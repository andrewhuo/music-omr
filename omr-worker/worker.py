import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from flask import Flask, jsonify, request

app = Flask(__name__)

AUDIVERIS_HOME = os.environ.get("AUDIVERIS_HOME", "/usr/share/audiveris")
GITHUB_API_BASE = os.environ.get("GITHUB_API_BASE", "https://api.github.com").rstrip("/")
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "andrewhuo")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "music-omr")
GITHUB_WORKFLOW_ID = os.environ.get("GITHUB_WORKFLOW_ID", "audiveris.yml")
GITHUB_REF = os.environ.get("GITHUB_REF", "main")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "gs://music-omr-bucket-777135743132/output")
RUN_DISCOVERY_TIMEOUT_SEC = int(os.environ.get("RUN_DISCOVERY_TIMEOUT_SEC", "20"))
RUN_DISCOVERY_POLL_SEC = float(os.environ.get("RUN_DISCOVERY_POLL_SEC", "2"))

# In-memory correlation for workflow dispatches that do not return run_id directly.
_PENDING_DISPATCHES: dict[str, dict] = {}


class GitHubAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = str(message)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_gh_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt)
    except Exception:
        return None


def _gh_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise GitHubAPIError(500, "GITHUB_TOKEN is not configured")
    return token


def _gh_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_gh_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def _gh_request(method: str, path: str, payload: dict | None = None, query: dict | None = None) -> dict | None:
    url = f"{GITHUB_API_BASE}{path}"
    if query:
        url = f"{url}?{urlparse.urlencode(query)}"

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urlrequest.Request(url, data=body, headers=_gh_headers(), method=method.upper())
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
            if not raw:
                return None
            return json.loads(raw)
    except urlerror.HTTPError as exc:
        raw_err = exc.read().decode("utf-8", errors="replace").strip()
        msg = raw_err or str(exc)
        raise GitHubAPIError(exc.code, msg) from exc
    except urlerror.URLError as exc:
        raise GitHubAPIError(502, f"GitHub API unreachable: {exc}") from exc


def _get_ref_sha(ref_name: str) -> str | None:
    data = _gh_request(
        "GET",
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{urlparse.quote(ref_name, safe='')}",
    )
    if not isinstance(data, dict):
        return None
    sha = str(data.get("sha") or "").strip()
    return sha or None


def _dispatch_workflow(pdf_gcs_uri: str) -> None:
    _gh_request(
        "POST",
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{urlparse.quote(GITHUB_WORKFLOW_ID, safe='')}/dispatches",
        payload={
            "ref": GITHUB_REF,
            "inputs": {
                "pdf_gcs_uri": pdf_gcs_uri,
            },
        },
    )


def _list_workflow_dispatch_runs(limit: int = 30) -> list[dict]:
    data = _gh_request(
        "GET",
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{urlparse.quote(GITHUB_WORKFLOW_ID, safe='')}/runs",
        query={
            "event": "workflow_dispatch",
            "branch": GITHUB_REF,
            "per_page": int(limit),
        },
    )
    if not isinstance(data, dict):
        return []
    runs = data.get("workflow_runs")
    if not isinstance(runs, list):
        return []
    return [r for r in runs if isinstance(r, dict)]


def _discover_run_id(dispatched_at: datetime, expected_sha: str | None) -> int | None:
    deadline = time.time() + max(2, int(RUN_DISCOVERY_TIMEOUT_SEC))
    lower_bound = dispatched_at - timedelta(minutes=2)

    while time.time() <= deadline:
        for run in _list_workflow_dispatch_runs():
            run_created = _parse_gh_datetime(run.get("created_at"))
            if run_created is None or run_created < lower_bound:
                continue
            run_sha = str(run.get("head_sha") or "").strip()
            if expected_sha and run_sha and run_sha != expected_sha:
                continue
            run_id = run.get("id")
            try:
                return int(run_id)
            except Exception:
                continue
        time.sleep(max(0.5, RUN_DISCOVERY_POLL_SEC))

    return None


def _artifact_uris_for_run(run_id: int) -> dict[str, str]:
    return {
        "annotated_pdf": f"{OUTPUT_PREFIX}/audiveris_out.pdf",
        "measure_mapping_debug": f"{OUTPUT_PREFIX}/artifacts/measure_mapping_debug.json",
        "mxl_page_manifest": f"{OUTPUT_PREFIX}/artifacts/mxl_page_manifest.json",
        "omr_debug_bundle": f"{OUTPUT_PREFIX}/artifacts/omr_debug_bundle.tar.gz",
        "run_info": f"{OUTPUT_PREFIX}/artifacts/run_info.json",
    }


def _run_public_status(run: dict) -> str:
    status = str(run.get("status") or "").strip().lower()
    conclusion = str(run.get("conclusion") or "").strip().lower()
    if status == "completed":
        if conclusion == "success":
            return "succeeded"
        return "failed"
    if status in ("queued", "requested", "waiting", "pending"):
        return "queued"
    if status in ("in_progress", "running"):
        return "running"
    return status or "unknown"


def _get_run(run_id: int) -> dict:
    data = _gh_request(
        "GET",
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{int(run_id)}",
    )
    if not isinstance(data, dict):
        raise GitHubAPIError(502, "GitHub run response was not an object")
    return data


def _pending_record(dispatch_id: str) -> dict | None:
    rec = _PENDING_DISPATCHES.get(dispatch_id)
    if not isinstance(rec, dict):
        return None
    return rec


def _ensure_run_id_for_pending(dispatch_id: str) -> tuple[int | None, dict | None]:
    rec = _pending_record(dispatch_id)
    if rec is None:
        return None, None
    run_id = rec.get("run_id")
    if isinstance(run_id, int):
        return run_id, rec

    dispatched_at = rec.get("dispatched_at")
    if not isinstance(dispatched_at, datetime):
        dispatched_at = _utc_now()
    expected_sha = rec.get("expected_sha")
    run_id = _discover_run_id(dispatched_at, expected_sha if isinstance(expected_sha, str) else None)
    if run_id is not None:
        rec["run_id"] = int(run_id)
        _PENDING_DISPATCHES[dispatch_id] = rec
    return run_id, rec


@app.route("/", methods=["GET"])
def health():
    return "omr-worker is running", 200


@app.route("/process", methods=["POST"])
def process_stub():
    # Backward-compatible stub endpoint.
    data = request.json or {}
    return jsonify(
        {
            "status": "ok",
            "message": "Use /api/omr/jobs for workflow dispatch integration",
            "audiveris_home": AUDIVERIS_HOME,
            "received": data,
        }
    ), 200


@app.route("/api/omr/jobs", methods=["POST"])
def create_job():
    data = request.json or {}
    pdf_gcs_uri = str(data.get("pdf_gcs_uri") or "").strip()
    if not pdf_gcs_uri:
        return jsonify({"error": "pdf_gcs_uri is required"}), 400
    if not pdf_gcs_uri.startswith("gs://"):
        return jsonify({"error": "pdf_gcs_uri must start with gs://"}), 400

    dispatch_id = str(uuid.uuid4())
    dispatched_at = _utc_now()
    try:
        expected_sha = _get_ref_sha(GITHUB_REF)
        _dispatch_workflow(pdf_gcs_uri)
        run_id = _discover_run_id(dispatched_at, expected_sha)
    except GitHubAPIError as exc:
        return jsonify({"error": exc.message, "status_code": exc.status_code}), (
            exc.status_code if 400 <= exc.status_code <= 599 else 500
        )

    _PENDING_DISPATCHES[dispatch_id] = {
        "dispatch_id": dispatch_id,
        "dispatched_at": dispatched_at,
        "expected_sha": expected_sha,
        "run_id": run_id,
        "pdf_gcs_uri": pdf_gcs_uri,
    }

    response = {
        "job_id": dispatch_id,
        "status": "queued",
        "run_id": run_id,
        "workflow": GITHUB_WORKFLOW_ID,
        "ref": GITHUB_REF,
        "pdf_gcs_uri": pdf_gcs_uri,
        "status_url": f"/api/omr/jobs/{dispatch_id}",
    }
    if run_id is not None:
        response["run_url"] = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}"
        response["artifacts"] = _artifact_uris_for_run(int(run_id))

    return jsonify(response), 202


@app.route("/api/omr/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    run_id = None
    if re.fullmatch(r"\d+", job_id or ""):
        run_id = int(job_id)
    else:
        run_id, rec = _ensure_run_id_for_pending(job_id)
        if rec is None:
            return jsonify({"error": f"unknown job_id: {job_id}"}), 404
        if run_id is None:
            return jsonify(
                {
                    "job_id": job_id,
                    "status": "dispatched",
                    "run_id": None,
                    "workflow": GITHUB_WORKFLOW_ID,
                    "ref": GITHUB_REF,
                    "status_url": f"/api/omr/jobs/{job_id}",
                }
            ), 202

    try:
        run = _get_run(int(run_id))
    except GitHubAPIError as exc:
        return jsonify({"error": exc.message, "status_code": exc.status_code}), (
            exc.status_code if 400 <= exc.status_code <= 599 else 500
        )

    response = {
        "job_id": job_id,
        "run_id": int(run_id),
        "status": _run_public_status(run),
        "github_status": run.get("status"),
        "github_conclusion": run.get("conclusion"),
        "ref": run.get("head_branch"),
        "sha": run.get("head_sha"),
        "run_attempt": run.get("run_attempt"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "run_url": run.get("html_url") or f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}",
        "artifacts": _artifact_uris_for_run(int(run_id)),
    }
    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
