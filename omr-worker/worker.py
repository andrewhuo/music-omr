import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import fitz  # PyMuPDF
from google.cloud import storage
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
RELABEL_MAX_VALUE = int(os.environ.get("RELABEL_MAX_VALUE", "1000000"))
RELABEL_MIN_VALUE = int(os.environ.get("RELABEL_MIN_VALUE", "0"))

MEASURE_TEXT_COLOR = (0, 0, 0)
MEASURE_TEXT_SIZE = 10.0
MEASURE_TEXT_Y_OFFSET = 6.0
MEASURE_TEXT_GUIDE_RIGHT_LIMIT = 6.0
MEASURE_TEXT_BG_COLOR = (1, 1, 1)

# In-memory correlation for workflow dispatches that do not return run_id directly.
_PENDING_DISPATCHES: dict[str, dict] = {}
_GCS_CLIENT: storage.Client | None = None


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
        "audiveris_out_pdf": f"{OUTPUT_PREFIX}/audiveris_out.pdf",
        "audiveris_out_corrected_pdf": f"{OUTPUT_PREFIX}/audiveris_out_corrected.pdf",
        "run_info": f"{OUTPUT_PREFIX}/artifacts/run_info.json",
        "mapping_summary": f"{OUTPUT_PREFIX}/artifacts/mapping_summary.json",
    }


def _gcs_client() -> storage.Client:
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    txt = str(uri or "").strip()
    if not txt.startswith("gs://"):
        raise ValueError(f"invalid gcs uri: {uri}")
    without = txt[5:]
    bucket, _, blob = without.partition("/")
    if not bucket or not blob:
        raise ValueError(f"invalid gcs uri: {uri}")
    return bucket, blob


def _download_gcs_json(uri: str) -> dict:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    raw = blob.download_as_bytes()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return data


def _download_gcs_to_file(uri: str, dest_path: Path) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))


def _upload_file_to_gcs(src_path: Path, dest_uri: str, content_type: str | None = None) -> None:
    bucket_name, blob_name = _parse_gs_uri(dest_uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(src_path), content_type=content_type)


def _upload_json_to_gcs(data: dict, dest_uri: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(dest_uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
    )


def _resolve_run_id_from_job_id(job_id: str) -> tuple[int | None, dict | None, str | None]:
    if re.fullmatch(r"\d+", job_id or ""):
        return int(job_id), None, None
    run_id, rec = _ensure_run_id_for_pending(job_id)
    if rec is None:
        return None, None, f"unknown job_id: {job_id}"
    if run_id is None:
        return None, rec, "job has been dispatched but run_id is not available yet"
    return int(run_id), rec, None


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _label_position(anchor_x: float, anchor_y_top: float, page_width: float, page_height: float, text: str) -> tuple[float, float, float]:
    tw = float(fitz.get_text_length(text, fontsize=MEASURE_TEXT_SIZE))
    x_centered = float(anchor_x) - (tw / 2.0)
    max_right = float(anchor_x) + MEASURE_TEXT_GUIDE_RIGHT_LIMIT
    if (x_centered + tw) > max_right:
        x_centered = max_right - tw
    x_text = min(max(0.0, x_centered), max(0.0, float(page_width) - tw - 2.0))
    y_text = max(MEASURE_TEXT_SIZE + 2.0, float(anchor_y_top) - MEASURE_TEXT_Y_OFFSET)
    y_text = min(y_text, max(MEASURE_TEXT_SIZE + 2.0, float(page_height) - 2.0))
    return x_text, y_text, tw
    return x_text, y_text, tw


def _draw_measure_label(page: fitz.Page, page_rect: fitz.Rect, anchor_x: float, anchor_y_top: float, text: str) -> None:
    x_text, y_text, tw = _label_position(anchor_x, anchor_y_top, float(page_rect.width), float(page_rect.height), text)
    th = float(MEASURE_TEXT_SIZE + 2.0)
    bg = fitz.Rect(x_text - 1.0, y_text - th + 1.0, x_text + tw + 1.0, y_text + 1.0)
    x0 = max(0.0, min(bg.x0, page_rect.width))
    y0 = max(0.0, min(bg.y0, page_rect.height))
    x1 = max(0.0, min(bg.x1, page_rect.width))
    y1 = max(0.0, min(bg.y1, page_rect.height))
    if x1 > x0 and y1 > y0:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=MEASURE_TEXT_BG_COLOR, fill=MEASURE_TEXT_BG_COLOR)
    page.insert_text((x_text, y_text), text, fontsize=MEASURE_TEXT_SIZE, color=MEASURE_TEXT_COLOR)


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


def _apply_relabel_edits(editable_state: dict, edits: list[dict]) -> tuple[list[dict], list[dict], list[dict], int]:
    systems = list(editable_state.get("systems") or [])
    if not systems:
        raise ValueError("editable_state.systems is missing or empty")

    systems = sorted(
        [s for s in systems if isinstance(s, dict)],
        key=lambda s: (_safe_int(s.get("page"), 0), _safe_int(s.get("system_index"), 0)),
    )
    if not systems:
        raise ValueError("editable_state.systems contains no valid rows")

    values: list[int] = []
    for row in systems:
        raw = row.get("current_value")
        if raw is None:
            raw = row.get("value")
        values.append(_safe_int(raw, 0))

    deltas: list[int] = []
    for idx in range(len(values) - 1):
        delta = values[idx + 1] - values[idx]
        deltas.append(max(0, int(delta)))

    id_to_index = {}
    for idx, row in enumerate(systems):
        sid = str(row.get("system_id") or "").strip()
        if sid:
            id_to_index[sid] = idx

    applied: list[dict] = []
    rejected: list[dict] = []
    for raw_edit in edits:
        if not isinstance(raw_edit, dict):
            rejected.append({"edit": raw_edit, "reason": "invalid_edit_object"})
            continue
        edit_type = str(raw_edit.get("type") or "").strip()
        if edit_type != "set_system_start":
            rejected.append({"edit": raw_edit, "reason": "unsupported_edit_type"})
            continue
        system_id = str(raw_edit.get("system_id") or "").strip()
        if not system_id or system_id not in id_to_index:
            rejected.append({"edit": raw_edit, "reason": "unknown_system_id"})
            continue
        try:
            new_value = int(raw_edit.get("value"))
        except Exception:
            rejected.append({"edit": raw_edit, "reason": "invalid_value"})
            continue
        if new_value < RELABEL_MIN_VALUE or new_value > RELABEL_MAX_VALUE:
            rejected.append(
                {
                    "edit": raw_edit,
                    "reason": "value_out_of_range",
                    "min": RELABEL_MIN_VALUE,
                    "max": RELABEL_MAX_VALUE,
                }
            )
            continue

        idx = id_to_index[system_id]
        values[idx] = int(new_value)
        for j in range(idx + 1, len(values)):
            values[j] = int(values[j - 1] + deltas[j - 1])
        applied.append({"type": "set_system_start", "system_id": system_id, "value": int(new_value)})

    for idx, row in enumerate(systems):
        row["current_value"] = str(values[idx])
        row["value"] = str(values[idx])
        row["render_label"] = str(values[idx])

    return systems, applied, rejected, len(values)


def _render_corrected_pdf(input_pdf: Path, output_pdf: Path, systems: list[dict], baseline_systems: dict[str, dict]) -> int:
    doc = fitz.open(str(input_pdf))
    drawn = 0
    for row in systems:
        sid = str(row.get("system_id") or "").strip()
        anchor = row.get("anchor") or {}
        page_no = _safe_int(row.get("page"), 0)
        if page_no <= 0 or page_no > doc.page_count:
            continue
        try:
            ax = float(anchor.get("x"))
            ay0 = float(anchor.get("y_top"))
        except Exception:
            continue
        page = doc[page_no - 1]
        rect = page.rect

        # Clear old label (if baseline exists), then draw new.
        base = baseline_systems.get(sid) or {}
        base_anchor = base.get("anchor") if isinstance(base, dict) else {}
        base_value = str(base.get("current_value") or base.get("value") or "").strip()
        if isinstance(base_anchor, dict) and base_value:
            try:
                bx = float(base_anchor.get("x"))
                by0 = float(base_anchor.get("y_top"))
                old_x, old_y, old_tw = _label_position(bx, by0, float(rect.width), float(rect.height), base_value)
                old_h = float(MEASURE_TEXT_SIZE + 2.0)
                erase = fitz.Rect(old_x - 2.0, old_y - old_h, old_x + old_tw + 2.0, old_y + 2.0)
                page.draw_rect(erase, color=MEASURE_TEXT_BG_COLOR, fill=MEASURE_TEXT_BG_COLOR)
            except Exception:
                pass

        label = str(row.get("current_value") or row.get("value") or "").strip()
        if not label:
            continue
        _draw_measure_label(page, rect, ax, ay0, label)
        drawn += 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()
    return drawn


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


@app.route("/api/omr/jobs/<job_id>/relabel", methods=["POST"])
def relabel_job(job_id: str):
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return jsonify({"error": err, "job_id": job_id}), 409

    payload = request.json or {}
    edits = payload.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        return jsonify({"error": "edits array is required", "job_id": job_id}), 400

    artifacts = _artifact_uris_for_run(int(run_id))
    mapping_uri = artifacts["mapping_summary"]
    baseline_pdf_uri = artifacts["audiveris_out_pdf"]
    corrected_pdf_uri = artifacts["audiveris_out_corrected_pdf"]
    run_info_uri = artifacts["run_info"]

    try:
        run_info = _download_gcs_json(run_info_uri)
        mapping_summary = _download_gcs_json(mapping_uri)
    except Exception as exc:
        return jsonify({"error": f"failed to load artifacts: {exc}", "job_id": job_id, "run_id": run_id}), 502

    summary_run_id = _safe_int(run_info.get("run_id"), 0)
    if summary_run_id and summary_run_id != int(run_id):
        return (
            jsonify(
                {
                    "error": "requested job_id does not match single-latest artifacts",
                    "job_id": job_id,
                    "requested_run_id": int(run_id),
                    "artifact_run_id": summary_run_id,
                }
            ),
            409,
        )

    editable_state = mapping_summary.get("editable_state") or {}
    if not isinstance(editable_state, dict):
        return jsonify({"error": "editable_state missing in mapping_summary", "job_id": job_id, "run_id": run_id}), 409

    started = time.time()
    try:
        baseline_systems = list(editable_state.get("systems") or [])
        baseline_by_id = {
            str(row.get("system_id")): row
            for row in baseline_systems
            if isinstance(row, dict) and str(row.get("system_id") or "").strip()
        }
        systems, applied, rejected, total_systems = _apply_relabel_edits(editable_state, edits)
    except ValueError as exc:
        return jsonify({"error": str(exc), "job_id": job_id, "run_id": run_id}), 400
    except Exception as exc:
        return jsonify({"error": f"failed to process edits: {exc}", "job_id": job_id, "run_id": run_id}), 500

    with TemporaryDirectory(prefix="omr-relabel-") as tmp:
        tmpdir = Path(tmp)
        in_pdf = tmpdir / "audiveris_out.pdf"
        out_pdf = tmpdir / "audiveris_out_corrected.pdf"
        try:
            _download_gcs_to_file(baseline_pdf_uri, in_pdf)
            redraw_started = time.time()
            labels_drawn = _render_corrected_pdf(in_pdf, out_pdf, systems, baseline_by_id)
            redraw_ms = int((time.time() - redraw_started) * 1000)
            _upload_file_to_gcs(out_pdf, corrected_pdf_uri, content_type="application/pdf")
        except Exception as exc:
            return jsonify({"error": f"failed to render corrected pdf: {exc}", "job_id": job_id, "run_id": run_id}), 500

    editable_state["systems"] = systems
    qa = editable_state.get("qa")
    if not isinstance(qa, dict):
        qa = {}
        editable_state["qa"] = qa
    qa["total_systems"] = len(systems)

    relabel_info = {
        "updated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "applied_edits": applied,
        "rejected_edits": rejected,
        "systems_updated_count": len(systems),
        "labels_redrawn_count": labels_drawn,
        "duration_ms": int((time.time() - started) * 1000),
        "redraw_duration_ms": redraw_ms,
    }
    mapping_summary["editable_state"] = editable_state
    mapping_summary["relabel"] = relabel_info

    try:
        _upload_json_to_gcs(mapping_summary, mapping_uri)
    except Exception as exc:
        return jsonify({"error": f"failed to upload mapping_summary: {exc}", "job_id": job_id, "run_id": run_id}), 500

    response = {
        "job_id": job_id,
        "run_id": int(run_id),
        "status": "succeeded",
        "artifacts": _artifact_uris_for_run(int(run_id)),
        "relabel": {
            "applied_edits": applied,
            "rejected_edits": rejected,
            "systems_updated_count": total_systems,
            "labels_redrawn_count": labels_drawn,
            "duration_ms": relabel_info["duration_ms"],
            "redraw_duration_ms": relabel_info["redraw_duration_ms"],
        },
        "single_latest_warning": "artifacts are single-latest; newer workflow runs overwrite prior baseline outputs",
    }
    if rec and isinstance(rec, dict) and rec.get("pdf_gcs_uri"):
        response["pdf_gcs_uri"] = rec.get("pdf_gcs_uri")
    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
