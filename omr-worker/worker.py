import json
import logging
import os
import re
import time
import threading
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import fitz  # PyMuPDF
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import storage
try:
    from google.cloud import firestore
except Exception:
    firestore = None
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger("omr-worker")

app = Flask(__name__)

AUDIVERIS_HOME = os.environ.get("AUDIVERIS_HOME", "/usr/share/audiveris")
GITHUB_API_BASE = os.environ.get("GITHUB_API_BASE", "https://api.github.com").rstrip("/")
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "andrewhuo")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "music-omr")
GITHUB_WORKFLOW_ID = os.environ.get("GITHUB_WORKFLOW_ID", "audiveris.yml")
GITHUB_REF = os.environ.get("GITHUB_REF", "main")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "gs://music-omr-bucket-777135743132/output")
INPUT_UPLOAD_PREFIX = os.environ.get("INPUT_UPLOAD_PREFIX", "gs://music-omr-bucket-777135743132/input/user-input")
RUN_DISCOVERY_TIMEOUT_SEC = int(os.environ.get("RUN_DISCOVERY_TIMEOUT_SEC", "20"))
RUN_DISCOVERY_POLL_SEC = float(os.environ.get("RUN_DISCOVERY_POLL_SEC", "2"))
RELABEL_MAX_VALUE = int(os.environ.get("RELABEL_MAX_VALUE", "1000000"))
RELABEL_MIN_VALUE = int(os.environ.get("RELABEL_MIN_VALUE", "0"))
ARTIFACT_SIGNED_URL_TTL_SEC = int(os.environ.get("ARTIFACT_SIGNED_URL_TTL_SEC", "1800"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
RELABEL_DEBUG_HISTORY_MAX = int(os.environ.get("RELABEL_DEBUG_HISTORY_MAX", "50"))
RELABEL_DEBUG_VERSION = "relabel_debug_v1"
CORS_ALLOW_ORIGINS_DEFAULT = "http://localhost:5173,http://localhost:3000"
RUNS_PREFIX = str(os.environ.get("RUNS_PREFIX", "runs") or "runs").strip().strip("/") or "runs"
ENABLE_JOB_STORE = str(os.environ.get("ENABLE_JOB_STORE", "1")).strip().lower() not in ("0", "false", "no")
JOB_STORE_COLLECTION = str(os.environ.get("JOB_STORE_COLLECTION", "omr_jobs") or "omr_jobs").strip() or "omr_jobs"
ALLOW_LEGACY_ARTIFACT_FALLBACK = (
    str(os.environ.get("ALLOW_LEGACY_ARTIFACT_FALLBACK", "1")).strip().lower() not in ("0", "false", "no")
)

MEASURE_TEXT_COLOR = (0, 0, 0)
MEASURE_TEXT_SIZE = 10.0
MEASURE_TEXT_Y_OFFSET = 8.0
MEASURE_TEXT_GUIDE_RIGHT_LIMIT = 6.0
MEASURE_TEXT_BG_COLOR = (1, 1, 1)
LABELS_MODE_SYSTEM_ONLY = "system_only"
LABELS_MODE_ALL_MEASURES = "all_measures"
LABELS_MODE_ALLOWED = {LABELS_MODE_SYSTEM_ONLY, LABELS_MODE_ALL_MEASURES}

# In-memory correlation for workflow dispatches that do not return run_id directly.
_PENDING_DISPATCHES: dict[str, dict] = {}
_PENDING_DISPATCHES_LOCK = threading.RLock()
_GCS_CLIENT: storage.Client | None = None
_GOOGLE_CREDENTIALS = None
_JOB_STORE_CLIENT = None


class GitHubAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = str(message)


class StaleArtifactsError(RuntimeError):
    def __init__(self, requested_run_id: int, artifact_run_id: int):
        super().__init__(
            f"requested job_id does not match single-latest artifacts: "
            f"requested_run_id={requested_run_id} artifact_run_id={artifact_run_id}"
        )
        self.requested_run_id = int(requested_run_id)
        self.artifact_run_id = int(artifact_run_id)


def _storage_mode_for_artifacts(artifacts: dict[str, str] | None) -> str:
    path = str((artifacts or {}).get("run_info") or "")
    marker = f"/{RUNS_PREFIX}/"
    if marker and marker in path:
        return "per_run_v1"
    return "legacy_single_latest"


def _api_path(path: str | None = None) -> bool:
    txt = str(path or request.path or "").strip()
    return txt.startswith("/api/omr/")


def _allowed_origins() -> set[str]:
    raw = os.environ.get("CORS_ALLOW_ORIGINS", CORS_ALLOW_ORIGINS_DEFAULT)
    return {entry.strip() for entry in str(raw or "").split(",") if entry.strip()}


def _origin_allowed(origin: str | None) -> bool:
    txt = str(origin or "").strip()
    if not txt:
        return False
    return txt in _allowed_origins()


def _apply_cors_headers(resp, origin: str | None):
    if not _origin_allowed(origin):
        return resp
    allow_origin = str(origin).strip()
    resp.headers["Access-Control-Allow-Origin"] = allow_origin
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    vary = str(resp.headers.get("Vary") or "").strip()
    if vary:
        if "Origin" not in [v.strip() for v in vary.split(",")]:
            resp.headers["Vary"] = f"{vary}, Origin"
    else:
        resp.headers["Vary"] = "Origin"
    return resp


@app.before_request
def _api_before_request():
    if not _api_path():
        return None

    origin = None
    try:
        origin = request.headers.get("Origin")
    except Exception:
        origin = None

    if request.method == "OPTIONS":
        try:
            return _apply_cors_headers(app.make_response(("", 204)), origin)
        except Exception as exc:
            # Never fail API preflight; frontend connectivity depends on a stable 204 response.
            print(f"CORS_PRECHECK_WARN detail={_safe_error_text(exc)}")
            return app.make_response(("", 204))
    return None


@app.after_request
def _api_after_request(resp):
    if _api_path():
        try:
            _apply_cors_headers(resp, request.headers.get("Origin"))
        except Exception as exc:
            print(f"CORS_AFTER_WARN detail={_safe_error_text(exc)}")
    return resp


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_z(raw: datetime | None) -> str:
    if not isinstance(raw, datetime):
        raw = _utc_now()
    return raw.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _workflow_id_candidates(primary: str) -> list[str]:
    ordered: list[str] = []

    def _add(value: str | None):
        txt = str(value or "").strip()
        if txt and txt not in ordered:
            ordered.append(txt)

    primary_txt = str(primary or "").strip()
    _add(primary_txt)

    base = Path(primary_txt).name if primary_txt else ""
    _add(base)
    if base:
        _add(f".github/workflows/{base}")

    try:
        payload = _gh_request(
            "GET",
            f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows",
            query={"per_page": 100},
        )
        workflows = payload.get("workflows") if isinstance(payload, dict) else None
        if isinstance(workflows, list):
            for wf in workflows:
                if not isinstance(wf, dict):
                    continue
                path = str(wf.get("path") or "").strip()
                wid = wf.get("id")
                name = str(wf.get("name") or "").strip()
                if base and (path.endswith(f"/{base}") or path.endswith(base)):
                    _add(path)
                    _add(str(wid) if wid is not None else "")
                if primary_txt and name == primary_txt:
                    _add(str(wid) if wid is not None else "")
    except Exception as exc:
        print(f"WORKFLOW_DISCOVERY_WARN detail={_safe_error_text(exc)}")

    return ordered


def _dispatch_workflow(pdf_gcs_uri: str, artifact_key: str | None = None) -> str:
    inputs = {
        "pdf_gcs_uri": pdf_gcs_uri,
    }
    key = str(artifact_key or "").strip()
    if key:
        inputs["artifact_key"] = key
    last_exc: GitHubAPIError | None = None
    for workflow_id in _workflow_id_candidates(GITHUB_WORKFLOW_ID):
        try:
            _gh_request(
                "POST",
                f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{urlparse.quote(workflow_id, safe='')}/dispatches",
                payload={
                    "ref": GITHUB_REF,
                    "inputs": inputs,
                },
            )
            if workflow_id != GITHUB_WORKFLOW_ID:
                print(f"WORKFLOW_DISPATCH_FALLBACK configured={GITHUB_WORKFLOW_ID} used={workflow_id}")
            return workflow_id
        except GitHubAPIError as exc:
            msg = str(exc.message or "")
            if exc.status_code == 422 and "workflow_dispatch" in msg:
                last_exc = exc
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise GitHubAPIError(500, "failed to dispatch workflow")


def _list_workflow_dispatch_runs(limit: int = 30, workflow_id: str | None = None) -> list[dict]:
    selector = str(workflow_id or GITHUB_WORKFLOW_ID).strip() or GITHUB_WORKFLOW_ID
    data = _gh_request(
        "GET",
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{urlparse.quote(selector, safe='')}/runs",
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


def _discover_run_id(dispatched_at: datetime, expected_sha: str | None, workflow_id: str | None = None) -> int | None:
    deadline = time.time() + max(2, int(RUN_DISCOVERY_TIMEOUT_SEC))
    lower_bound = dispatched_at - timedelta(minutes=2)

    while time.time() <= deadline:
        for run in _list_workflow_dispatch_runs(workflow_id=workflow_id):
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


def _output_prefix_normalized() -> str:
    return str(OUTPUT_PREFIX or "").rstrip("/")


def _normalize_artifact_key(value: str | int | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-.")


def _run_output_prefix(run_key: str | int) -> str:
    safe_key = _normalize_artifact_key(run_key)
    if not safe_key:
        safe_key = "unknown"
    return f"{_output_prefix_normalized()}/{RUNS_PREFIX}/{safe_key}"


def _legacy_output_prefix() -> str:
    return _output_prefix_normalized()


def _artifact_uris_for_run(run_id: int, artifact_key: str | None = None) -> dict[str, str]:
    run_key = _normalize_artifact_key(artifact_key) or str(int(run_id))
    out = _run_output_prefix(run_key)
    return {
        "audiveris_out_pdf": f"{out}/audiveris_out.pdf",
        "audiveris_out_corrected_pdf": f"{out}/audiveris_out_corrected.pdf",
        "run_info": f"{out}/artifacts/run_info.json",
        "mapping_summary": f"{out}/artifacts/mapping_summary.json",
    }


def _legacy_artifact_uris_for_run(run_id: int) -> dict[str, str]:
    out = _legacy_output_prefix()
    return {
        "audiveris_out_pdf": f"{out}/audiveris_out.pdf",
        "audiveris_out_corrected_pdf": f"{out}/audiveris_out_corrected.pdf",
        "run_info": f"{out}/artifacts/run_info.json",
        "mapping_summary": f"{out}/artifacts/mapping_summary.json",
    }


def _artifact_uris_for_existing_run(run_id: int, artifact_key: str | None = None) -> dict[str, str]:
    candidates: list[dict[str, str]] = []
    primary = _artifact_uris_for_run(int(run_id), artifact_key=artifact_key)
    candidates.append(primary)
    fallback_key = str(int(run_id))
    if _normalize_artifact_key(artifact_key) and _normalize_artifact_key(artifact_key) != fallback_key:
        candidates.append(_artifact_uris_for_run(int(run_id), artifact_key=fallback_key))
    for per_run in candidates:
        try:
            if _gcs_uri_exists(per_run["run_info"]):
                return per_run
        except Exception:
            return per_run
    if ALLOW_LEGACY_ARTIFACT_FALLBACK:
        return _legacy_artifact_uris_for_run(int(run_id))
    return primary


def _gs_uri_to_bucket_blob(uri: str) -> tuple[str, str]:
    return _parse_gs_uri(uri)


def _signed_http_url_for_gs(uri: str) -> str:
    try:
        bucket_name, blob_name = _gs_uri_to_bucket_blob(uri)
        bucket = _gcs_client().bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return ""
        ttl_sec = max(60, _safe_int(os.environ.get("ARTIFACT_SIGNED_URL_TTL_SEC"), ARTIFACT_SIGNED_URL_TTL_SEC))
        expiry = timedelta(seconds=ttl_sec)
        try:
            return str(
                blob.generate_signed_url(
                    version="v4",
                    expiration=expiry,
                    method="GET",
                )
            )
        except Exception as exc:
            # Cloud Run default credentials are token-only; retry with IAM signBlob flow.
            detail = _safe_error_text(exc).lower()
            if ("private key" not in detail) and ("sign credentials" not in detail):
                raise
            access_token = _runtime_access_token()
            service_account_email = _runtime_service_account_email()
            if not access_token or not service_account_email:
                raise
            return str(
                blob.generate_signed_url(
                    version="v4",
                    expiration=expiry,
                    method="GET",
                    service_account_email=service_account_email,
                    access_token=access_token,
                )
            )
    except Exception as exc:
        print(f"SIGNED_URL_WARN uri={uri} detail={_safe_error_text(exc)}")
        return ""


def _artifact_http_uris_for_run(run_id: int, artifacts: dict[str, str] | None = None) -> dict[str, str]:
    source = artifacts if isinstance(artifacts, dict) else _artifact_uris_for_run(run_id)
    out: dict[str, str] = {}
    for key, value in source.items():
        out[key] = _signed_http_url_for_gs(value)
    return out


def _gcs_client() -> storage.Client:
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def _runtime_credentials():
    global _GOOGLE_CREDENTIALS
    if _GOOGLE_CREDENTIALS is None:
        _GOOGLE_CREDENTIALS, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return _GOOGLE_CREDENTIALS


def _runtime_access_token() -> str:
    creds = _runtime_credentials()
    try:
        if not getattr(creds, "valid", False) or getattr(creds, "expired", False) or not getattr(creds, "token", None):
            creds.refresh(GoogleAuthRequest())
    except Exception:
        return ""
    return str(getattr(creds, "token", "") or "")


def _runtime_service_account_email() -> str:
    creds = _runtime_credentials()
    email = str(getattr(creds, "service_account_email", "") or "").strip()
    if email and email.lower() != "default":
        return email
    try:
        req = urlrequest.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"},
        )
        with urlrequest.urlopen(req, timeout=2) as resp:
            txt = resp.read().decode("utf-8", errors="replace").strip()
            return txt
    except Exception:
        return ""


def _job_store_client():
    global _JOB_STORE_CLIENT
    if not ENABLE_JOB_STORE or firestore is None:
        return None
    if _JOB_STORE_CLIENT is None:
        try:
            _JOB_STORE_CLIENT = firestore.Client()
        except Exception as exc:
            print(f"JOB_STORE_CLIENT_WARN detail={_safe_error_text(exc)}")
            return None
    return _JOB_STORE_CLIENT


def _job_store_upsert(job_id: str, payload: dict) -> None:
    client = _job_store_client()
    if client is None:
        return
    data = dict(payload or {})
    data["job_id"] = str(job_id)
    data["updated_at_utc"] = _to_utc_z(_utc_now())
    try:
        client.collection(JOB_STORE_COLLECTION).document(str(job_id)).set(data, merge=True)
    except Exception as exc:
        print(f"JOB_STORE_UPSERT_WARN job_id={job_id} detail={_safe_error_text(exc)}")


def _job_store_get(job_id: str) -> dict | None:
    client = _job_store_client()
    if client is None:
        return None
    try:
        snap = client.collection(JOB_STORE_COLLECTION).document(str(job_id)).get()
        if not bool(getattr(snap, "exists", False)):
            return None
        data = snap.to_dict()
        if isinstance(data, dict):
            return data
        return None
    except Exception as exc:
        print(f"JOB_STORE_GET_WARN job_id={job_id} detail={_safe_error_text(exc)}")
        return None


def _derive_job_id_from_pdf_uri(pdf_gcs_uri: str) -> str:
    try:
        _, blob_name = _parse_gs_uri(pdf_gcs_uri)
    except Exception:
        return ""
    base = Path(blob_name).name
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return _normalize_artifact_key(stem)[:96]


def _job_artifact_key(job_id: str, run_id: int | None = None, rec: dict | None = None) -> str:
    if isinstance(rec, dict):
        for key in ("artifact_key", "job_id", "dispatch_id"):
            val = _normalize_artifact_key(rec.get(key))
            if val:
                return val
    txt = _normalize_artifact_key(job_id)
    if txt:
        return txt
    if isinstance(run_id, int):
        return str(int(run_id))
    return ""


def _ensure_unique_job_id(base_job_id: str) -> str:
    base = _normalize_artifact_key(base_job_id)[:96] or str(uuid.uuid4())
    if _pending_record(base) is None and _job_store_get(base) is None:
        return base
    for idx in range(2, 1000):
        candidate = f"{base}-{idx}"
        if _pending_record(candidate) is None and _job_store_get(candidate) is None:
            return candidate
    return f"{base}-{uuid.uuid4().hex[:8]}"


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    txt = str(uri or "").strip()
    if not txt.startswith("gs://"):
        raise ValueError(f"invalid gcs uri: {uri}")
    without = txt[5:]
    bucket, _, blob = without.partition("/")
    if not bucket or not blob:
        raise ValueError(f"invalid gcs uri: {uri}")
    return bucket, blob


def _gcs_uri_exists(uri: str) -> bool:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return bool(blob.exists())


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


def _delete_gcs_uri_if_exists(uri: str) -> tuple[bool, bool]:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not bool(blob.exists()):
        return False, False
    blob.delete()
    return True, True


def _delete_gcs_prefix(prefix_uri: str, max_samples: int = 20) -> dict:
    bucket_name, blob_prefix = _parse_gs_uri(prefix_uri.rstrip("/") + "/_")
    blob_prefix = blob_prefix.rsplit("/_", 1)[0].rstrip("/") + "/"
    bucket = _gcs_client().bucket(bucket_name)
    deleted = 0
    found = 0
    samples: list[str] = []
    for blob in _gcs_client().list_blobs(bucket_name, prefix=blob_prefix):
        found += 1
        if len(samples) < max(1, int(max_samples)):
            samples.append(f"gs://{bucket_name}/{blob.name}")
        try:
            blob.delete()
            deleted += 1
        except Exception as exc:
            if len(samples) < max(1, int(max_samples)):
                samples.append(f"ERROR:{blob.name}:{_safe_error_text(exc)}")
    return {
        "prefix": f"gs://{bucket_name}/{blob_prefix.rstrip('/')}",
        "found_count": found,
        "deleted_count": deleted,
        "samples": samples,
    }


def _max_upload_bytes() -> int:
    return max(1, _safe_int(os.environ.get("MAX_UPLOAD_MB"), MAX_UPLOAD_MB)) * 1024 * 1024


def _resolve_run_id_from_job_id(job_id: str) -> tuple[int | None, dict | None, str | None]:
    run_id, rec = _ensure_run_id_for_pending(job_id)
    if isinstance(run_id, int):
        return int(run_id), rec, None
    store = _job_store_get(job_id)
    if isinstance(store, dict):
        store_run_id = _safe_int(store.get("run_id"), 0)
        if store_run_id > 0:
            merged = dict(store)
            if isinstance(rec, dict):
                merged = {**rec, **store}
            return int(store_run_id), merged, None
        return None, (rec if isinstance(rec, dict) else store), "job has been dispatched but run_id is not available yet"
    if re.fullmatch(r"\d+", job_id or ""):
        return int(job_id), None, None
    if rec is None:
        return None, None, f"unknown job_id: {job_id}"
    return None, rec, "job has been dispatched but run_id is not available yet"


def _load_mapping_for_run(run_id: int, artifact_key: str | None = None) -> tuple[dict, dict, int]:
    run_id_int = int(run_id)
    key = _normalize_artifact_key(artifact_key)
    candidate_keys: list[str] = []
    if key:
        candidate_keys.append(key)
    run_key = str(run_id_int)
    if run_key not in candidate_keys:
        candidate_keys.append(run_key)

    for candidate in candidate_keys:
        artifacts = _artifact_uris_for_run(run_id_int, artifact_key=candidate)
        if _gcs_uri_exists(artifacts["run_info"]) and _gcs_uri_exists(artifacts["mapping_summary"]):
            run_info = _download_gcs_json(artifacts["run_info"])
            mapping_summary = _download_gcs_json(artifacts["mapping_summary"])
            summary_run_id = _safe_int(run_info.get("run_id"), run_id_int)
            if summary_run_id and summary_run_id != run_id_int:
                print(
                    f"RUN_INFO_WARN requested_run_id={run_id_int} "
                    f"run_info_run_id={summary_run_id} mode=per_run_v1"
                )
            if not isinstance(mapping_summary, dict):
                raise ValueError("mapping_summary is not an object")
            return artifacts, mapping_summary, int(summary_run_id or run_id_int)

    if not ALLOW_LEGACY_ARTIFACT_FALLBACK:
        key_note = f" artifact_key={key}" if key else ""
        raise FileNotFoundError(f"per-run artifacts not found for run_id={run_id_int}{key_note}")

    legacy_artifacts = _legacy_artifact_uris_for_run(run_id_int)
    run_info = _download_gcs_json(legacy_artifacts["run_info"])
    mapping_summary = _download_gcs_json(legacy_artifacts["mapping_summary"])
    summary_run_id = _safe_int(run_info.get("run_id"), 0)
    if summary_run_id and summary_run_id != run_id_int:
        raise StaleArtifactsError(run_id_int, int(summary_run_id))
    if not isinstance(mapping_summary, dict):
        raise ValueError("mapping_summary is not an object")
    return legacy_artifacts, mapping_summary, int(summary_run_id or run_id_int)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    txt = str(value or "").strip().lower()
    if txt in ("1", "true", "yes", "y", "on"):
        return True
    if txt in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


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


def _editable_state_version(editable_state: dict) -> str:
    payload = {
        "version": editable_state.get("version"),
        "labels_mode": str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
        "systems": editable_state.get("systems") or [],
        "measures": editable_state.get("measures") or [],
        "measure_number_overrides": editable_state.get("measure_number_overrides") or {},
        "rest_systems": editable_state.get("rest_systems") or {},
        "endings": editable_state.get("endings") or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_error_text(exc: Exception | str, max_len: int = 220) -> str:
    txt = str(exc or "").strip().replace("\n", " ").replace("\r", " ")
    if len(txt) <= max_len:
        return txt
    return f"{txt[:max_len]}..."


def _rejected_reason_counts(rejected: list[dict] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rejected or []:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _append_relabel_trace(mapping_summary: dict, trace: dict, max_history: int = RELABEL_DEBUG_HISTORY_MAX) -> dict:
    if not isinstance(mapping_summary, dict):
        return {}

    relabel_debug = mapping_summary.get("relabel_debug")
    if not isinstance(relabel_debug, dict):
        relabel_debug = {}

    history = relabel_debug.get("history")
    if not isinstance(history, list):
        history = []
    clean_trace = {k: v for k, v in trace.items() if v is not None}
    history.append(clean_trace)

    max_keep = max(1, int(max_history))
    if len(history) > max_keep:
        history = history[-max_keep:]

    reason_counts: dict[str, int] = {}
    for row in history:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rejected_counts = row.get("rejected_reason_counts")
        if isinstance(rejected_counts, dict):
            for key, value in rejected_counts.items():
                code = str(key or "").strip()
                if not code:
                    continue
                reason_counts[code] = reason_counts.get(code, 0) + _safe_int(value, 0)

    relabel_debug["version"] = RELABEL_DEBUG_VERSION
    relabel_debug["history_max"] = max_keep
    relabel_debug["history"] = history
    relabel_debug["last_trace"] = history[-1] if history else {}
    relabel_debug["reason_counts"] = dict(sorted(reason_counts.items()))
    mapping_summary["relabel_debug"] = relabel_debug
    return relabel_debug


def _summarize_relabel_debug(mapping_summary: dict) -> dict:
    relabel_debug = mapping_summary.get("relabel_debug")
    if not isinstance(relabel_debug, dict):
        return {
            "history_count": 0,
            "history_max": max(1, RELABEL_DEBUG_HISTORY_MAX),
            "last_result": "",
            "last_trace_id": "",
            "reason_counts": {},
        }

    history = relabel_debug.get("history")
    if not isinstance(history, list):
        history = []
    last_trace = relabel_debug.get("last_trace")
    if not isinstance(last_trace, dict):
        last_trace = history[-1] if history and isinstance(history[-1], dict) else {}
    reason_counts = relabel_debug.get("reason_counts")
    if not isinstance(reason_counts, dict):
        reason_counts = {}

    return {
        "history_count": len(history),
        "history_max": max(1, _safe_int(relabel_debug.get("history_max"), RELABEL_DEBUG_HISTORY_MAX)),
        "last_result": str(last_trace.get("result") or ""),
        "last_trace_id": str(last_trace.get("trace_id") or ""),
        "reason_counts": reason_counts,
    }


def _persist_relabel_trace(mapping_summary: dict, mapping_uri: str, trace: dict, trace_id: str) -> bool:
    try:
        _append_relabel_trace(mapping_summary, trace)
        _upload_json_to_gcs(mapping_summary, mapping_uri)
        return True
    except Exception as exc:
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} "
            f"stage=trace_persist reason=mapping_upload_failed "
            f"detail={_safe_error_text(exc)}"
        )
        return False


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


def _draw_measure_label_left_barline(page: fitz.Page, page_rect: fitz.Rect, x_left: float, y_top: float, text: str) -> None:
    tw = float(fitz.get_text_length(text, fontsize=MEASURE_TEXT_SIZE))
    x_text = min(max(0.0, float(x_left) + 1.0), max(0.0, float(page_rect.width) - tw - 2.0))
    y_text = max(MEASURE_TEXT_SIZE + 2.0, float(y_top) - MEASURE_TEXT_Y_OFFSET)
    y_text = min(y_text, max(MEASURE_TEXT_SIZE + 2.0, float(page_rect.height) - 2.0))

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


def _pending_set(dispatch_id: str, record: dict) -> None:
    with _PENDING_DISPATCHES_LOCK:
        _PENDING_DISPATCHES[str(dispatch_id)] = dict(record or {})


def _pending_items_snapshot() -> list[tuple[str, dict]]:
    with _PENDING_DISPATCHES_LOCK:
        return [
            (str(dispatch_id), dict(record))
            for dispatch_id, record in _PENDING_DISPATCHES.items()
            if isinstance(record, dict)
        ]


def _pending_record(dispatch_id: str) -> dict | None:
    with _PENDING_DISPATCHES_LOCK:
        rec = _PENDING_DISPATCHES.get(dispatch_id)
    if not isinstance(rec, dict):
        return None
    return dict(rec)


def _ensure_run_id_for_pending(dispatch_id: str) -> tuple[int | None, dict | None]:
    rec = _pending_record(dispatch_id)
    if rec is None:
        store = _job_store_get(dispatch_id)
        if isinstance(store, dict):
            rec = dict(store)
            _pending_set(dispatch_id, rec)
    if rec is None:
        return None, None
    run_id = rec.get("run_id")
    if isinstance(run_id, int):
        return run_id, rec

    dispatched_at = rec.get("dispatched_at")
    if not isinstance(dispatched_at, datetime):
        dispatched_at = _utc_now()
    expected_sha = rec.get("expected_sha")
    workflow_id = str(rec.get("workflow_id") or GITHUB_WORKFLOW_ID).strip() or GITHUB_WORKFLOW_ID
    run_id = _discover_run_id(
        dispatched_at,
        expected_sha if isinstance(expected_sha, str) else None,
        workflow_id=workflow_id,
    )
    if run_id is not None:
        rec["run_id"] = int(run_id)
        _pending_set(dispatch_id, rec)
        _job_store_upsert(
            dispatch_id,
            {
                "run_id": int(run_id),
                "status": "queued",
                "mode": "per_run_v1",
                "workflow_id": workflow_id,
            },
        )
    return run_id, rec


def _pending_dispatched_at(rec: dict) -> datetime:
    raw = rec.get("dispatched_at")
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc)
    if isinstance(raw, str):
        parsed = _parse_gh_datetime(raw)
        if isinstance(parsed, datetime):
            return parsed.astimezone(timezone.utc)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _reassign_measures_to_nearest_system(systems: list[dict], measures: list[dict]) -> int:
    """Post-process measures to fix system misassignment from OMR.

    The OMR pipeline sometimes assigns measures to the wrong system based on
    XML element order rather than geometric position.  This function reassigns
    each measure to the system whose anchor y-range best overlaps the measure's
    y-range on the same page.  Mutates *measures* in place.

    Returns the number of measures that were reassigned.
    """
    # Step 1: Build per-page system lookup from anchors.
    page_systems: dict[int, list[tuple[str, int, float, float]]] = {}  # page -> [(system_id, system_index, y_top, y_bot)]
    for s in systems:
        anchor = s.get("anchor")
        if not isinstance(anchor, dict):
            continue
        try:
            page = int(s["page"])
            y_top = float(anchor["y_top"])
            y_bot = float(anchor["y_bottom"])
            sid = str(s["system_id"])
            sidx = int(s.get("system_index", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if y_bot <= y_top:
            continue
        page_systems.setdefault(page, []).append((sid, sidx, y_top, y_bot))
    # Sort each page's systems by y_top (top-to-bottom on page).
    for page in page_systems:
        page_systems[page].sort(key=lambda t: t[2])

    if not page_systems:
        return 0

    tolerance = 5.0  # PDF points tolerance for overlap
    reassigned = 0

    # Step 2: For each measure, find the best-matching system by y-overlap.
    for m in measures:
        try:
            m_page = int(m["page"])
            m_y_top = float(m["y_top"])
            m_y_bot = float(m["y_bottom"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates = page_systems.get(m_page)
        if not candidates:
            continue

        best_sid = None
        best_sidx = 0
        best_overlap = -999999.0
        best_center_dist = 999999.0
        m_center = (m_y_top + m_y_bot) / 2.0

        for sid, sidx, s_y_top, s_y_bot in candidates:
            overlap = min(m_y_bot, s_y_bot) - max(m_y_top, s_y_top) + tolerance
            s_center = (s_y_top + s_y_bot) / 2.0
            center_dist = abs(m_center - s_center)
            if overlap > best_overlap or (overlap == best_overlap and center_dist < best_center_dist):
                best_overlap = overlap
                best_center_dist = center_dist
                best_sid = sid
                best_sidx = sidx

        if best_sid is None:
            continue

        current_sid = str(m.get("system_id") or "")
        if current_sid != best_sid:
            print(
                f"MEASURE_REASSIGN measure={m.get('measure_id')} "
                f"from={current_sid} to={best_sid} "
                f"m_y=[{m_y_top:.1f},{m_y_bot:.1f}] page={m_page}"
            )
            m["system_id"] = best_sid
            m["system_index"] = best_sidx
            reassigned += 1

    # Step 3: Recompute measure_local_index and measure_id for all measures
    # (needed because reassignment changes group membership).
    from collections import defaultdict
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for m in measures:
        key = (int(m.get("page", 0)), str(m.get("system_id", "")))
        groups[key].append(m)

    for key, group in groups.items():
        group.sort(key=lambda m: float(m.get("x_left", 0)))
        page_no, sys_id = key
        for local_idx, m in enumerate(group):
            m["measure_local_index"] = local_idx
            sidx = m.get("system_index", 0)
            m["measure_id"] = f"p{page_no}_s{sidx}_m{local_idx}"

    return reassigned


def _sorted_measure_rows(measures: list[dict] | None) -> list[dict]:
    return sorted(
        [m for m in (measures or []) if isinstance(m, dict)],
        key=lambda m: (
            _safe_int(m.get("page"), 0),
            _safe_int(m.get("system_index"), 0),
            float(m.get("x_left") or 0),
            _safe_int(m.get("measure_local_index"), 0),
            str(m.get("measure_id") or ""),
        ),
    )


def _sorted_system_rows(systems: list[dict] | None) -> list[dict]:
    return sorted(
        [s for s in (systems or []) if isinstance(s, dict)],
        key=lambda s: (_safe_int(s.get("page"), 0), _safe_int(s.get("system_index"), 0)),
    )


def _measure_number_overrides(editable_state: dict) -> dict[str, int]:
    raw = editable_state.get("measure_number_overrides")
    if not isinstance(raw, dict):
        editable_state["measure_number_overrides"] = {}
        return {}

    cleaned: dict[str, int] = {}
    for raw_key, raw_value in raw.items():
        measure_id = str(raw_key or "").strip()
        if not measure_id:
            continue
        try:
            value = int(raw_value)
        except Exception:
            continue
        if value < RELABEL_MIN_VALUE or value > RELABEL_MAX_VALUE:
            continue
        cleaned[measure_id] = value

    editable_state["measure_number_overrides"] = cleaned
    return cleaned


def _recompute_measure_numbering(
    systems: list[dict] | None,
    measures: list[dict] | None,
    editable_state: dict | None = None,
) -> tuple[list[dict], list[dict], dict[str, str], dict[str, int]]:
    editable_state = editable_state or {}
    sorted_systems = _sorted_system_rows(systems)
    ordered_measures = _sorted_measure_rows(measures)

    first_start = 1
    if sorted_systems:
        first_start = _safe_int(
            sorted_systems[0].get("current_value") or sorted_systems[0].get("value"),
            1,
        )

    endings_map = editable_state.get("endings")
    if not isinstance(endings_map, dict):
        endings_map = {}
    rest_systems = editable_state.get("rest_systems")
    if not isinstance(rest_systems, dict):
        rest_systems = {}
    measure_overrides = _measure_number_overrides(editable_state)

    if ordered_measures:
        first_measure_id = str(ordered_measures[0].get("measure_id") or "").strip()
        if first_measure_id and first_measure_id in measure_overrides:
            first_start = int(measure_overrides[first_measure_id])

    result_labels: dict[str, str] = {}
    seq_starts_by_system: dict[str, int] = {}
    current_value = int(first_start)
    first_ending_start_value: int | None = None
    second_ending_local = 0
    current_sid: str | None = None

    for measure in ordered_measures:
        measure_id = str(measure.get("measure_id") or "").strip()
        system_id = str(measure.get("system_id") or "").strip()

        if system_id != current_sid:
            if current_sid is not None:
                rest_count = _safe_int(rest_systems.get(current_sid), 0)
                if rest_count > 0:
                    current_value += rest_count
            current_sid = system_id

        if measure_id and measure_id in measure_overrides:
            current_value = int(measure_overrides[measure_id])

        ending_type = str(endings_map.get(measure_id) or "").strip() if measure_id else ""
        if ending_type == "1":
            if first_ending_start_value is None:
                first_ending_start_value = current_value
            label_value = int(current_value)
            current_value += 1
        elif ending_type == "2":
            base = first_ending_start_value if first_ending_start_value is not None else current_value
            label_value = int(base + second_ending_local)
            second_ending_local += 1
        else:
            label_value = int(current_value)
            current_value += 1

        label = str(label_value)
        if measure_id:
            result_labels[measure_id] = label
        if system_id and system_id not in seq_starts_by_system:
            seq_starts_by_system[system_id] = label_value
        measure["current_value"] = label
        measure["value"] = label
        measure["render_label"] = label

    for system in sorted_systems:
        system_id = str(system.get("system_id") or "").strip()
        if system_id and system_id in seq_starts_by_system:
            label = seq_starts_by_system[system_id]
        else:
            label = _safe_int(system.get("current_value") or system.get("value"), first_start)
        system["current_value"] = str(label)
        system["value"] = str(label)
        system["render_label"] = str(label)

    return sorted_systems, ordered_measures, result_labels, seq_starts_by_system


def _apply_relabel_edits(editable_state: dict, edits: list[dict]) -> tuple[list[dict], list[dict], list[dict], int]:
    systems = _sorted_system_rows(editable_state.get("systems") or [])
    if not systems:
        raise ValueError("editable_state.systems is missing or empty")
    measures = _sorted_measure_rows(editable_state.get("measures") or [])
    editable_state["systems"] = systems
    editable_state["measures"] = measures

    id_to_index = {}
    for idx, row in enumerate(systems):
        sid = str(row.get("system_id") or "").strip()
        if sid:
            id_to_index[sid] = idx
    first_measure_by_system: dict[str, dict] = {}
    measure_ids = set()
    for measure in measures:
        measure_id = str(measure.get("measure_id") or "").strip()
        if measure_id:
            measure_ids.add(measure_id)
        system_id = str(measure.get("system_id") or "").strip()
        if system_id and system_id not in first_measure_by_system:
            first_measure_by_system[system_id] = measure

    applied: list[dict] = []
    rejected: list[dict] = []
    labels_mode = str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY).strip().lower()
    if labels_mode not in LABELS_MODE_ALLOWED:
        labels_mode = LABELS_MODE_SYSTEM_ONLY
    measure_overrides = _measure_number_overrides(editable_state)

    for raw_edit in edits:
        if not isinstance(raw_edit, dict):
            rejected.append({"edit": raw_edit, "reason": "invalid_edit_object"})
            continue
        edit_type = str(raw_edit.get("type") or "").strip()
        if edit_type == "set_system_start":
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

            first_measure = first_measure_by_system.get(system_id)
            if not isinstance(first_measure, dict):
                rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
                continue
            measure_id = str(first_measure.get("measure_id") or "").strip()
            if not measure_id:
                rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
                continue
            measure_overrides[measure_id] = int(new_value)
            applied.append({"type": "set_system_start", "system_id": system_id, "value": int(new_value)})
            continue

        if edit_type == "set_measure_number":
            measure_id = str(raw_edit.get("measure_id") or "").strip()
            if not measure_id:
                rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
                continue
            if measure_id not in measure_ids:
                rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
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

            measure_overrides[measure_id] = int(new_value)
            applied.append({"type": "set_measure_number", "measure_id": measure_id, "value": int(new_value)})
            continue

        if edit_type == "set_labels_mode":
            mode = str(raw_edit.get("value") or "").strip().lower()
            if mode not in LABELS_MODE_ALLOWED:
                rejected.append({"edit": raw_edit, "reason": "invalid_value"})
                continue
            labels_mode = mode
            applied.append({"type": "set_labels_mode", "value": labels_mode})
            continue

        if edit_type == "set_rest_staff":
            system_id = str(raw_edit.get("system_id") or "").strip()
            if not system_id or system_id not in id_to_index:
                rejected.append({"edit": raw_edit, "reason": "unknown_system_id"})
                continue
            measure_count = raw_edit.get("value")
            if not isinstance(measure_count, int) or measure_count < 0:
                rejected.append({"edit": raw_edit, "reason": "invalid_measure_count"})
                continue
            if "rest_systems" not in editable_state:
                editable_state["rest_systems"] = {}
            # Undo previously applied rest for this staff before applying new one
            prev_rest = editable_state["rest_systems"].get(system_id, 0)
            if measure_count == 0:
                editable_state["rest_systems"].pop(system_id, None)
            else:
                editable_state["rest_systems"][system_id] = measure_count
            diff = measure_count - prev_rest
            import sys
            msg1 = f"REST_DEBUG system_id={system_id} measure_count={measure_count} prev_rest={prev_rest} diff={diff}"
            logger.warning(msg1)
            print(msg1, file=sys.stderr, flush=True)
            applied.append({"type": "set_rest_staff", "system_id": system_id, "value": measure_count})
            continue

        if edit_type == "set_ending":
            measure_id = str(raw_edit.get("measure_id") or "").strip()
            ending_val = str(raw_edit.get("value") or "").strip()
            if not measure_id:
                rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
                continue
            if "endings" not in editable_state:
                editable_state["endings"] = {}
            if ending_val in ("", "none"):
                editable_state["endings"].pop(measure_id, None)
            elif ending_val in ("1", "2"):
                editable_state["endings"][measure_id] = ending_val
            else:
                rejected.append({"edit": raw_edit, "reason": "invalid_ending_value"})
                continue
            applied.append({"type": "set_ending", "measure_id": measure_id, "value": ending_val})
            continue

        rejected.append({"edit": raw_edit, "reason": "unsupported_edit_type"})

    editable_state["measure_number_overrides"] = measure_overrides
    systems, measures, _, _ = _recompute_measure_numbering(systems, measures, editable_state)
    editable_state["systems"] = systems
    editable_state["measures"] = measures
    editable_state["staff_boxes"] = []
    editable_state["labels_mode"] = labels_mode
    return systems, applied, rejected, len(systems)


def _render_corrected_pdf(
    input_pdf: Path,
    output_pdf: Path,
    systems: list[dict],
    baseline_systems: dict[str, dict],
    measures: list[dict],
    labels_mode: str,
    editable_state: dict | None = None,
) -> int:
    editable_state = editable_state or {}
    doc = fitz.open(str(input_pdf))
    drawn = 0

    def _erase_baseline_system_label(page, rect, base_row: dict) -> None:
        base_anchor = base_row.get("anchor") if isinstance(base_row, dict) else {}
        base_value = str(base_row.get("current_value") or base_row.get("value") or "").strip()
        if not isinstance(base_anchor, dict) or not base_value:
            return
        try:
            bx = float(base_anchor.get("x"))
            by0 = float(base_anchor.get("y_top"))
            old_x, old_y, old_tw = _label_position(bx, by0, float(rect.width), float(rect.height), base_value)
            old_h = float(MEASURE_TEXT_SIZE + 2.0)
            erase = fitz.Rect(old_x - 2.0, old_y - old_h, old_x + old_tw + 2.0, old_y + 2.0)
            page.draw_rect(erase, color=MEASURE_TEXT_BG_COLOR, fill=MEASURE_TEXT_BG_COLOR)
        except Exception:
            return

    # --- Shared: erase all baseline system labels ---
    for base in baseline_systems.values():
        if not isinstance(base, dict):
            continue
        page_no = _safe_int(base.get("page"), 0)
        if page_no <= 0 or page_no > doc.page_count:
            continue
        page = doc[page_no - 1]
        _erase_baseline_system_label(page, page.rect, base)

    sorted_systems, ordered_measures, result_labels, _ = _recompute_measure_numbering(
        systems,
        measures,
        editable_state,
    )

    if labels_mode == LABELS_MODE_ALL_MEASURES:
        # Draw every measure with its computed label
        for measure in ordered_measures:
            mid = str(measure.get("measure_id") or "").strip()
            page_no = _safe_int(measure.get("page"), 0)
            if page_no <= 0 or page_no > doc.page_count:
                continue
            label = result_labels.get(mid)
            if not label:
                continue
            try:
                x_left = float(measure.get("x_left"))
                y_top = float(measure.get("y_top"))
            except Exception:
                continue
            page = doc[page_no - 1]
            _draw_measure_label_left_barline(page, page.rect, x_left, y_top, label)
            drawn += 1
    else:
        # Staff-start mode still follows the measure sequence, but only the
        # first measure on each system gets a visible label.
        seen_system_ids: set[str] = set()
        for measure in ordered_measures:
            system_id = str(measure.get("system_id") or "").strip()
            if not system_id or system_id in seen_system_ids:
                continue
            seen_system_ids.add(system_id)
            page_no = _safe_int(measure.get("page"), 0)
            if page_no <= 0 or page_no > doc.page_count:
                continue
            try:
                x_left = float(measure.get("x_left"))
                y_top = float(measure.get("y_top"))
            except Exception:
                continue
            label = str(measure.get("current_value") or measure.get("render_label") or "").strip()
            if not label:
                continue
            page = doc[page_no - 1]
            _draw_measure_label_left_barline(page, page.rect, x_left, y_top, label)
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


@app.route("/api/omr/uploads", methods=["POST"])
def upload_pdf():
    file_obj = request.files.get("file")
    if file_obj is None:
        return jsonify({"error": "file is required"}), 400

    filename = str(file_obj.filename or "").strip()
    content_type = str(file_obj.mimetype or file_obj.content_type or "").strip().lower()
    looks_pdf = filename.lower().endswith(".pdf") or content_type in ("application/pdf", "application/x-pdf")
    if not looks_pdf:
        return jsonify({"error": "file must be a PDF"}), 400

    try:
        raw = file_obj.read()
    except Exception as exc:
        return jsonify({"error": f"failed to read upload: {_safe_error_text(exc)}"}), 400

    if not raw:
        return jsonify({"error": "empty file"}), 400

    max_bytes = _max_upload_bytes()
    if len(raw) > max_bytes:
        return (
            jsonify(
                {
                    "error": "file too large",
                    "max_upload_mb": max(1, _safe_int(os.environ.get("MAX_UPLOAD_MB"), MAX_UPLOAD_MB)),
                    "size_bytes": len(raw),
                }
            ),
            413,
        )

    upload_id = uuid.uuid4().hex[:16]
    upload_prefix = str(os.environ.get("INPUT_UPLOAD_PREFIX") or INPUT_UPLOAD_PREFIX).rstrip("/")
    pdf_gcs_uri = f"{upload_prefix}/{upload_id}.pdf"

    with TemporaryDirectory(prefix="omr-upload-") as tmp:
        tmp_pdf = Path(tmp) / f"{upload_id}.pdf"
        tmp_pdf.write_bytes(raw)
        try:
            _upload_file_to_gcs(tmp_pdf, pdf_gcs_uri, content_type="application/pdf")
        except Exception as exc:
            return jsonify({"error": f"failed to upload pdf: {_safe_error_text(exc)}"}), 500

    return (
        jsonify(
            {
                "upload_id": upload_id,
                "pdf_gcs_uri": pdf_gcs_uri,
                "size_bytes": len(raw),
                "content_type": "application/pdf",
            }
        ),
        201,
    )


@app.route("/api/omr/jobs", methods=["POST"])
def create_job():
    data = request.json or {}
    pdf_gcs_uri = str(data.get("pdf_gcs_uri") or "").strip()
    if not pdf_gcs_uri:
        return jsonify({"error": "pdf_gcs_uri is required"}), 400
    if not pdf_gcs_uri.startswith("gs://"):
        return jsonify({"error": "pdf_gcs_uri must start with gs://"}), 400

    requested_job_id = str(data.get("job_id") or "").strip()
    if not requested_job_id:
        requested_job_id = _derive_job_id_from_pdf_uri(pdf_gcs_uri)
    dispatch_id = _ensure_unique_job_id(requested_job_id or str(uuid.uuid4()))
    artifact_key = _job_artifact_key(dispatch_id)
    dispatched_at = _utc_now()
    try:
        expected_sha = _get_ref_sha(GITHUB_REF)
        workflow_id_used = _dispatch_workflow(pdf_gcs_uri, artifact_key=artifact_key) or GITHUB_WORKFLOW_ID
        run_id = _discover_run_id(dispatched_at, expected_sha, workflow_id=workflow_id_used)
    except GitHubAPIError as exc:
        return jsonify({"error": exc.message, "status_code": exc.status_code}), (
            exc.status_code if 400 <= exc.status_code <= 599 else 500
        )

    _pending_set(
        dispatch_id,
        {
            "dispatch_id": dispatch_id,
            "dispatched_at": dispatched_at,
            "expected_sha": expected_sha,
            "run_id": run_id,
            "pdf_gcs_uri": pdf_gcs_uri,
            "artifact_key": artifact_key,
            "workflow_id": workflow_id_used,
        },
    )
    _job_store_upsert(
        dispatch_id,
        {
            "created_at_utc": _to_utc_z(dispatched_at),
            "status": "queued",
            "run_id": int(run_id) if isinstance(run_id, int) else None,
            "pdf_gcs_uri": pdf_gcs_uri,
            "workflow": GITHUB_WORKFLOW_ID,
            "workflow_id": workflow_id_used,
            "ref": GITHUB_REF,
            "mode": "per_run_v1",
            "artifact_key": artifact_key,
        },
    )

    response = {
        "job_id": dispatch_id,
        "artifact_key": artifact_key,
        "status": "queued",
        "run_id": run_id,
        "workflow": workflow_id_used,
        "ref": GITHUB_REF,
        "pdf_gcs_uri": pdf_gcs_uri,
        "status_url": f"/api/omr/jobs/{dispatch_id}",
    }
    if run_id is not None:
        artifacts = _artifact_uris_for_existing_run(int(run_id), artifact_key=artifact_key)
        response["run_url"] = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}"
        response["artifacts"] = artifacts
        response["artifacts_http"] = _artifact_http_uris_for_run(int(run_id), artifacts)
        response["storage_mode"] = _storage_mode_for_artifacts(artifacts)

    return jsonify(response), 202


@app.route("/api/omr/jobs", methods=["GET"])
def list_jobs():
    try:
        items = _pending_items_snapshot()
        items.sort(key=lambda kv: _pending_dispatched_at(kv[1]), reverse=True)
    except Exception as exc:
        print(f"LIST_JOBS_WARN detail={_safe_error_text(exc)}")
        return jsonify({"jobs": []}), 200

    rows: list[dict] = []
    for dispatch_id, rec in items:
        try:
            created_at = _to_utc_z(_pending_dispatched_at(rec))
            status = "queued"
            run_id = _safe_int(rec.get("run_id"), 0)
            if run_id > 0:
                try:
                    run = _get_run(int(run_id))
                    status = _run_public_status(run)
                    created_at = str(run.get("created_at") or created_at)
                except Exception as exc:
                    # Keep list endpoint stable even when GitHub API is unavailable.
                    print(f"LIST_JOBS_RUN_WARN run_id={run_id} detail={_safe_error_text(exc)}")
                    status = "queued"
            rows.append(
                {
                    "job_id": str(dispatch_id),
                    "status": str(status),
                    "created_at": str(created_at),
                }
            )
        except Exception as exc:
            print(f"LIST_JOBS_ROW_WARN job_id={dispatch_id} detail={_safe_error_text(exc)}")
            continue
    return jsonify({"jobs": rows}), 200


@app.route("/api/omr/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    run_id = None
    rec = None
    if re.fullmatch(r"\d+", job_id or ""):
        run_id = int(job_id)
    else:
        run_id, rec, _ = _resolve_run_id_from_job_id(job_id)
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
    }
    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)
    artifacts = _artifact_uris_for_existing_run(int(run_id), artifact_key=artifact_key)
    response["artifacts"] = artifacts
    response["artifacts_http"] = _artifact_http_uris_for_run(int(run_id), artifacts)
    response["storage_mode"] = _storage_mode_for_artifacts(artifacts)
    if isinstance(rec, dict) and rec.get("pdf_gcs_uri"):
        response["pdf_gcs_uri"] = rec.get("pdf_gcs_uri")
    return jsonify(response), 200


@app.route("/api/omr/jobs/<job_id>/state", methods=["GET"])
def get_job_state(job_id: str):
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return jsonify({"error": err, "job_id": job_id}), 409
    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)

    try:
        artifacts, mapping_summary, _ = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
    except StaleArtifactsError as exc:
        return (
            jsonify(
                {
                    "error": "requested job_id does not match single-latest artifacts",
                    "job_id": job_id,
                    "requested_run_id": exc.requested_run_id,
                    "artifact_run_id": exc.artifact_run_id,
                }
            ),
            409,
        )
    except Exception as exc:
        return jsonify({"error": f"failed to load state: {exc}", "job_id": job_id, "run_id": run_id}), 502

    editable_state = mapping_summary.get("editable_state") or {}
    if not isinstance(editable_state, dict):
        return jsonify({"error": "editable_state missing in mapping_summary", "job_id": job_id, "run_id": run_id}), 409

    systems = editable_state.get("systems")
    if not isinstance(systems, list):
        systems = []
    measures = editable_state.get("measures")
    if not isinstance(measures, list):
        measures = []
    staff_boxes = editable_state.get("staff_boxes")
    if not isinstance(staff_boxes, list):
        staff_boxes = []
    qa = editable_state.get("qa")
    if not isinstance(qa, dict):
        qa = {}

    reassign_count = _reassign_measures_to_nearest_system(systems, measures)
    if reassign_count > 0:
        print(f"MEASURE_REASSIGN_SUMMARY job_id={job_id} reassigned={reassign_count}")
    systems, measures, _, _ = _recompute_measure_numbering(systems, measures, editable_state)
    editable_state["systems"] = systems
    editable_state["measures"] = measures
    editable_state["staff_boxes"] = []

    response = {
        "job_id": job_id,
        "run_id": int(run_id),
        "state_version": _editable_state_version(editable_state),
        "editable_state": {
            "version": str(editable_state.get("version") or "system_state_v1"),
            "labels_mode": str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
            "rest_systems": editable_state.get("rest_systems") or {},
            "qa": qa,
            "systems": systems,
            "measures": measures,
            "staff_boxes": [],
            "measure_number_overrides": editable_state.get("measure_number_overrides") or {},
            "endings": editable_state.get("endings") or {},
        },
        "relabel_debug_summary": _summarize_relabel_debug(mapping_summary),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(int(run_id), artifacts),
        "storage_mode": _storage_mode_for_artifacts(artifacts),
    }
    return jsonify(response), 200


@app.route("/api/omr/jobs/<job_id>/relabel", methods=["POST"])
def relabel_job(job_id: str):
    trace_id = _new_trace_id()
    started = time.time()
    payload = request.json or {}
    edits = payload.get("edits")
    edits_requested_count = len(edits) if isinstance(edits, list) else 0

    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    requested_run_id = int(run_id) if isinstance(run_id, int) else 0
    artifact_key = _job_artifact_key(job_id, requested_run_id or None, rec if isinstance(rec, dict) else None)
    print(
        f"RELABEL_TRACE_START trace_id={trace_id} job_id={job_id} "
        f"run_id={requested_run_id or 'unknown'} edits={edits_requested_count}"
    )

    if err:
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=resolve_run "
            f"reason=state_load_failed detail={_safe_error_text(err)}"
        )
        return jsonify({"error": err, "job_id": job_id, "trace_id": trace_id, "debug_result": "validation_error"}), 409

    try:
        artifacts, mapping_summary, artifact_run_id = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
    except StaleArtifactsError as exc:
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(exc.requested_run_id),
            "artifact_run_id": int(exc.artifact_run_id),
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": 0,
            "rejected_reason_counts": {},
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "stale_conflict",
            "reason": "stale_run_mismatch",
            "error_detail": "requested job_id does not match single-latest artifacts",
        }
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=load_artifacts "
            "reason=stale_run_mismatch detail=requested job_id does not match single-latest artifacts"
        )
        try:
            stale_artifacts, stale_mapping_summary, _ = _load_mapping_for_run(int(exc.artifact_run_id))
            _persist_relabel_trace(stale_mapping_summary, stale_artifacts["mapping_summary"], trace, trace_id)
        except Exception as trace_exc:
            print(
                f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=trace_persist "
                f"reason=mapping_upload_failed detail={_safe_error_text(trace_exc)}"
            )
        return (
            jsonify(
                {
                    "error": "requested job_id does not match single-latest artifacts",
                    "job_id": job_id,
                    "requested_run_id": exc.requested_run_id,
                    "artifact_run_id": exc.artifact_run_id,
                    "trace_id": trace_id,
                    "debug_result": "stale_conflict",
                }
            ),
            409,
        )
    except Exception as exc:
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=load_artifacts "
            f"reason=state_load_failed detail={_safe_error_text(exc)}"
        )
        return (
            jsonify(
                {
                    "error": f"failed to load artifacts: {exc}",
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "internal_error",
                }
            ),
            502,
        )

    mapping_uri = artifacts["mapping_summary"]
    baseline_pdf_uri = artifacts["audiveris_out_pdf"]
    corrected_pdf_uri = artifacts["audiveris_out_corrected_pdf"]

    editable_state = mapping_summary.get("editable_state") or {}
    if not isinstance(editable_state, dict):
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": 0,
            "rejected_reason_counts": {},
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "validation_error",
            "reason": "editable_state_missing",
            "error_detail": "editable_state missing in mapping_summary",
        }
        _persist_relabel_trace(mapping_summary, mapping_uri, trace, trace_id)
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=validate_state "
            "reason=editable_state_missing detail=editable_state missing in mapping_summary"
        )
        return (
            jsonify(
                {
                    "error": "editable_state missing in mapping_summary",
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "validation_error",
                }
            ),
            409,
        )
    labels_mode_before = str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY).strip().lower()
    if labels_mode_before not in LABELS_MODE_ALLOWED:
        labels_mode_before = LABELS_MODE_SYSTEM_ONLY
    editable_state["labels_mode"] = labels_mode_before
    systems_before = _sorted_system_rows(editable_state.get("systems") or [])
    measures_before = _sorted_measure_rows(editable_state.get("measures") or [])
    reassign_count = _reassign_measures_to_nearest_system(systems_before, measures_before)
    if reassign_count > 0:
        print(f"MEASURE_REASSIGN_SUMMARY job_id={job_id} reassigned={reassign_count}")
    editable_state["systems"] = systems_before
    editable_state["measures"] = measures_before
    editable_state["staff_boxes"] = []
    systems_before, measures_before, _, _ = _recompute_measure_numbering(
        systems_before,
        measures_before,
        editable_state,
    )
    editable_state["systems"] = systems_before
    editable_state["measures"] = measures_before

    if not isinstance(edits, list) or len(edits) == 0:
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "state_version_before": _editable_state_version(editable_state),
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": 0,
            "rejected_reason_counts": {"invalid_payload": 1},
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "validation_error",
            "reason": "invalid_payload",
            "error_detail": "edits array is required",
        }
        _persist_relabel_trace(mapping_summary, mapping_uri, trace, trace_id)
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=validate_payload "
            "reason=invalid_payload detail=edits array is required"
        )
        return (
            jsonify(
                {
                    "error": "edits array is required",
                    "job_id": job_id,
                    "trace_id": trace_id,
                    "debug_result": "validation_error",
                }
            ),
            400,
        )

    state_version_before = _editable_state_version(editable_state)

    try:
        baseline_systems = list(editable_state.get("systems") or [])
        baseline_by_id = {
            str(row.get("system_id")): row
            for row in baseline_systems
            if isinstance(row, dict) and str(row.get("system_id") or "").strip()
        }
        systems, applied, rejected, total_systems = _apply_relabel_edits(editable_state, edits)
    except ValueError as exc:
        reason = "invalid_payload"
        if "unknown_system_id" in str(exc):
            reason = "unknown_system_id"
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "state_version_before": state_version_before,
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": 0,
            "rejected_reason_counts": {reason: 1},
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "validation_error",
            "reason": reason,
            "error_detail": _safe_error_text(exc),
        }
        _persist_relabel_trace(mapping_summary, mapping_uri, trace, trace_id)
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=apply_edits "
            f"reason={reason} detail={_safe_error_text(exc)}"
        )
        return (
            jsonify(
                {
                    "error": str(exc),
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "validation_error",
                }
            ),
            400,
        )
    except Exception as exc:
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "state_version_before": state_version_before,
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": 0,
            "rejected_reason_counts": {"internal_error": 1},
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "internal_error",
            "reason": "internal_error",
            "error_detail": _safe_error_text(exc),
        }
        _persist_relabel_trace(mapping_summary, mapping_uri, trace, trace_id)
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=apply_edits "
            f"reason=internal_error detail={_safe_error_text(exc)}"
        )
        return (
            jsonify(
                {
                    "error": f"failed to process edits: {exc}",
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "internal_error",
                }
            ),
            500,
        )

    redraw_ms = 0
    try:
        with TemporaryDirectory(prefix="omr-relabel-") as tmp:
            tmpdir = Path(tmp)
            in_pdf = tmpdir / "audiveris_out.pdf"
            out_pdf = tmpdir / "audiveris_out_corrected.pdf"
            _download_gcs_to_file(baseline_pdf_uri, in_pdf)
            redraw_started = time.time()
            labels_drawn = _render_corrected_pdf(
                in_pdf,
                out_pdf,
                systems,
                baseline_by_id,
                list(editable_state.get("measures") or []),
                str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
                editable_state=editable_state,
            )
            redraw_ms = int((time.time() - redraw_started) * 1000)
            _upload_file_to_gcs(out_pdf, corrected_pdf_uri, content_type="application/pdf")
    except Exception as exc:
        reason = "pdf_render_failed"
        error_txt = _safe_error_text(exc)
        if "download" in error_txt.lower():
            reason = "pdf_download_failed"
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "state_version_before": state_version_before,
            "edits_requested_count": edits_requested_count,
            "applied_count": len(applied),
            "rejected_count": len(rejected),
            "rejected_reason_counts": _rejected_reason_counts(rejected),
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": redraw_ms,
            "result": "render_error",
            "reason": reason,
            "error_detail": error_txt,
        }
        _persist_relabel_trace(mapping_summary, mapping_uri, trace, trace_id)
        print(
            f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=render_pdf "
            f"reason={reason} detail={error_txt}"
        )
        return (
            jsonify(
                {
                    "error": f"failed to render corrected pdf: {exc}",
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "render_error",
                }
            ),
            500,
        )

    editable_state["systems"] = systems
    qa = editable_state.get("qa")
    if not isinstance(qa, dict):
        qa = {}
        editable_state["qa"] = qa
    qa["total_systems"] = len(systems)
    state_version_after = _editable_state_version(editable_state)

    before_values = {}
    for row in baseline_systems:
        if isinstance(row, dict):
            sid = str(row.get("system_id") or "").strip()
            if sid:
                before_values[sid] = str(row.get("current_value") or row.get("value") or "")
    updated_system_ids: list[str] = []
    for row in systems:
        sid = str(row.get("system_id") or "").strip()
        if not sid:
            continue
        after_value = str(row.get("current_value") or row.get("value") or "")
        if before_values.get(sid) != after_value:
            updated_system_ids.append(sid)

    relabel_info = {
        "updated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "applied_edits": applied,
        "rejected_edits": rejected,
        "labels_mode": str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
        "systems_updated_count": len(systems),
        "labels_redrawn_count": labels_drawn,
        "duration_ms": int((time.time() - started) * 1000),
        "redraw_duration_ms": redraw_ms,
    }
    mapping_summary["editable_state"] = editable_state
    mapping_summary["relabel"] = relabel_info

    trace = {
        "trace_id": trace_id,
        "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "job_id": job_id,
        "requested_run_id": int(run_id),
        "artifact_run_id": int(artifact_run_id),
        "state_version_before": state_version_before,
        "state_version_after": state_version_after,
        "edits_requested_count": edits_requested_count,
        "applied_count": len(applied),
        "rejected_count": len(rejected),
        "rejected_reason_counts": _rejected_reason_counts(rejected),
        "updated_system_ids_count": len(updated_system_ids),
        "labels_redrawn_count": labels_drawn,
        "duration_ms": relabel_info["duration_ms"],
        "redraw_duration_ms": relabel_info["redraw_duration_ms"],
        "result": "success",
    }
    if len(rejected) > 0:
        trace["reason"] = "invalid_payload"

    if not _persist_relabel_trace(mapping_summary, mapping_uri, trace, trace_id):
        return (
            jsonify(
                {
                    "error": "failed to upload mapping_summary",
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "upload_error",
                }
            ),
            500,
        )

    print(
        f"RELABEL_TRACE_RESULT trace_id={trace_id} result={trace['result']} "
        f"applied={len(applied)} rejected={len(rejected)} duration_ms={relabel_info['duration_ms']}"
    )

    response = {
        "job_id": job_id,
        "run_id": int(run_id),
        "status": "succeeded",
        "trace_id": trace_id,
        "debug_result": str(trace.get("result") or "success"),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(int(run_id), artifacts),
        "storage_mode": _storage_mode_for_artifacts(artifacts),
        "relabel": {
            "applied_edits": applied,
            "rejected_edits": rejected,
            "labels_mode": str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
            "state_version_before": state_version_before,
            "state_version_after": state_version_after,
            "updated_system_ids": updated_system_ids,
            "systems_updated_count": total_systems,
            "labels_redrawn_count": labels_drawn,
            "duration_ms": relabel_info["duration_ms"],
            "redraw_duration_ms": relabel_info["redraw_duration_ms"],
        },
    }
    if response["storage_mode"] == "legacy_single_latest":
        response["single_latest_warning"] = (
            "artifacts are legacy single-latest; newer workflow runs may overwrite prior baseline outputs"
        )
    if rec and isinstance(rec, dict) and rec.get("pdf_gcs_uri"):
        response["pdf_gcs_uri"] = rec.get("pdf_gcs_uri")
    return jsonify(response), 200


@app.route("/api/omr/jobs/<job_id>/cleanup", methods=["POST"])
def cleanup_job_artifacts(job_id: str):
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if run_id is None:
        if rec is None:
            return jsonify({"error": f"unknown job_id: {job_id}"}), 404
        return jsonify({"error": err or "run_id is not available yet", "job_id": job_id}), 409

    payload = request.json or {}
    delete_corrected_pdf = _safe_bool(payload.get("delete_corrected_pdf", True), True)
    delete_baseline_pdf = _safe_bool(payload.get("delete_baseline_pdf", False), False)
    delete_artifacts = _safe_bool(payload.get("delete_artifacts", False), False)
    delete_all_run_objects = _safe_bool(payload.get("delete_all_run_objects", False), False)
    delete_input_pdf = _safe_bool(payload.get("delete_input_pdf", False), False)

    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)
    artifacts = _artifact_uris_for_existing_run(int(run_id), artifact_key=artifact_key)
    targets: list[str] = []
    if delete_corrected_pdf:
        targets.append(artifacts["audiveris_out_corrected_pdf"])
    if delete_baseline_pdf:
        targets.append(artifacts["audiveris_out_pdf"])
    if delete_artifacts:
        targets.extend([artifacts["run_info"], artifacts["mapping_summary"]])

    results: list[dict] = []
    deleted_count = 0

    if delete_all_run_objects:
        run_prefix = str(artifacts.get("audiveris_out_pdf") or "").rsplit("/", 1)[0]
        if run_prefix.startswith("gs://"):
            try:
                prefix_result = _delete_gcs_prefix(run_prefix)
                deleted_count += _safe_int(prefix_result.get("deleted_count"), 0)
                results.append({"prefix_cleanup": prefix_result})
            except Exception as exc:
                results.append({"prefix_cleanup": {"prefix": run_prefix, "error": _safe_error_text(exc)}})

    for uri in targets:
        try:
            existed, deleted = _delete_gcs_uri_if_exists(uri)
            if deleted:
                deleted_count += 1
            results.append({"uri": uri, "existed": bool(existed), "deleted": bool(deleted)})
        except Exception as exc:
            results.append({"uri": uri, "error": _safe_error_text(exc)})
    if delete_input_pdf and isinstance(rec, dict):
        input_uri = str(rec.get("pdf_gcs_uri") or "").strip()
        if input_uri.startswith("gs://"):
            try:
                existed, deleted = _delete_gcs_uri_if_exists(input_uri)
                if deleted:
                    deleted_count += 1
                results.append({"uri": input_uri, "existed": bool(existed), "deleted": bool(deleted)})
            except Exception as exc:
                results.append({"uri": input_uri, "error": _safe_error_text(exc)})

    _job_store_upsert(
        job_id,
        {
            "cleanup": {
                "deleted_count": deleted_count,
                "targets": [row.get("uri") for row in results if isinstance(row, dict) and row.get("uri")],
            }
        },
    )

    response = {
        "job_id": job_id,
        "run_id": int(run_id),
        "storage_mode": _storage_mode_for_artifacts(artifacts),
        "results": results,
        "deleted_count": deleted_count,
    }
    if isinstance(rec, dict) and rec.get("pdf_gcs_uri"):
        response["pdf_gcs_uri"] = rec.get("pdf_gcs_uri")
    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
