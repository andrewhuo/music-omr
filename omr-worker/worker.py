import json
import logging
import os
import re
import time
import threading
import uuid
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
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
ANTHROPIC_API_BASE = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com").rstrip("/")
ANTHROPIC_MODEL = str(os.environ.get("ANTHROPIC_MODEL", "") or "").strip()
ANTHROPIC_VERSION = str(os.environ.get("ANTHROPIC_VERSION", "2023-06-01") or "2023-06-01").strip() or "2023-06-01"
ANTHROPIC_TIMEOUT_SEC = max(5.0, float(os.environ.get("ANTHROPIC_TIMEOUT_SEC", "90") or "90"))
ANTHROPIC_MAX_TOKENS = max(256, int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1800") or "1800"))
AI_PROVIDER = str(os.environ.get("AI_PROVIDER", "bedrock") or "bedrock").strip().lower()
AWS_REGION = str(os.environ.get("AWS_REGION", "us-east-1") or "us-east-1").strip()
BEDROCK_MODEL_ID = str(
    os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0") or ""
).strip()
BEDROCK_ANTHROPIC_VERSION = str(
    os.environ.get("BEDROCK_ANTHROPIC_VERSION", "bedrock-2023-05-31") or "bedrock-2023-05-31"
).strip() or "bedrock-2023-05-31"
AI_MEASURE_CROP_SCALE = max(1.0, float(os.environ.get("AI_MEASURE_CROP_SCALE", "2.0") or "2.0"))
AI_MEASURE_CROP_X_PAD_RATIO = max(0.0, float(os.environ.get("AI_MEASURE_CROP_X_PAD_RATIO", "0.08") or "0.08"))
AI_MEASURE_CROP_MIN_X_PAD = max(0.0, float(os.environ.get("AI_MEASURE_CROP_MIN_X_PAD", "8") or "8"))
AI_MEASURE_CROP_TOP_PAD_RATIO = max(0.0, float(os.environ.get("AI_MEASURE_CROP_TOP_PAD_RATIO", "1.00") or "1.00"))
AI_MEASURE_CROP_BOTTOM_PAD_RATIO = max(0.0, float(os.environ.get("AI_MEASURE_CROP_BOTTOM_PAD_RATIO", "1.00") or "1.00"))
AI_MEASURE_CROP_MIN_TOP_PAD = max(0.0, float(os.environ.get("AI_MEASURE_CROP_MIN_TOP_PAD", "20") or "20"))
AI_MEASURE_CROP_MIN_BOTTOM_PAD = max(0.0, float(os.environ.get("AI_MEASURE_CROP_MIN_BOTTOM_PAD", "10") or "10"))
AI_MEASURE_CROP_SYSTEM_GAP_CLAMP_RATIO = max(
    0.0,
    min(1.0, float(os.environ.get("AI_MEASURE_CROP_SYSTEM_GAP_CLAMP_RATIO", "0.75") or "0.75")),
)
SUSPICIOUS_PARTIAL_STAFF_HEIGHT_RATIO = 0.65
AI_SUGGEST_SAVE_DEBUG_CROPS = (
    str(os.environ.get("AI_SUGGEST_SAVE_DEBUG_CROPS", "0")).strip().lower() not in ("0", "false", "no", "")
)

MEASURE_TEXT_COLOR = (0, 0, 0)
MEASURE_TEXT_SIZE = 10.0
MEASURE_TEXT_Y_OFFSET = 8.0
MEASURE_TEXT_GUIDE_RIGHT_LIMIT = 6.0
MEASURE_TEXT_BG_COLOR = (1, 1, 1)
LABELS_MODE_SYSTEM_ONLY = "system_only"
LABELS_MODE_ALL_MEASURES = "all_measures"
LABELS_MODE_ALLOWED = {LABELS_MODE_SYSTEM_ONLY, LABELS_MODE_ALL_MEASURES}
ROW_SOURCE_AUTO = "auto"
ROW_SOURCE_MANUAL = "manual"
MANUAL_STAFF_KIND_SINGLE = "single"
MANUAL_STAFF_KIND_GRAND = "grand"
MANUAL_STAFF_KINDS_ALLOWED = {MANUAL_STAFF_KIND_SINGLE, MANUAL_STAFF_KIND_GRAND}
MANUAL_SYSTEM_ID_PREFIX = "manual_sys_"
MANUAL_MEASURE_ID_PREFIX = "manual_measure_"
MANUAL_ROW_OVERLAP_RATIO = 0.5
AUTO_ROW_RECT_TOLERANCE = 8.0
AUTO_BOX_MIN_WIDTH = 2.0
STAFF_START_SAME_ROW_OVERLAP_RATIO = 0.30
STAFF_START_SAME_ROW_CENTER_TOLERANCE_RATIO = 0.45
STAFF_START_SAME_ROW_MIN_HEIGHT_RATIO = 0.55
STAFF_START_SAME_ROW_MAX_HEIGHT_RATIO = 1.80
AI_SUGGESTIONS_VERSION = "ai_suggestions_v1"
AI_SUGGEST_RUN_STATUS_IDLE = "idle"
AI_SUGGEST_RUN_STATUS_RUNNING = "running"
AI_SUGGEST_RUN_STATUS_COMPLETED = "completed"
AI_SUGGEST_RUN_STATUS_FAILED = "failed"
AI_SUGGESTION_LABELS_ALLOWED = {"normal", "pickup", "multi_measure_rest", "uncertain"}
AI_SUGGESTION_CONFIDENCE_ALLOWED = {"low", "medium", "high"}
AI_SUGGESTION_MAYBE_LABELS_ALLOWED = {"pickup", "multi_measure_rest"}
AI_SUGGESTION_COMPLETENESS_ALLOWED = {"full", "incomplete", "unclear"}
AI_SCORE_TYPES_ALLOWED = {"single", "grand", "score"}
AI_SUGGESTION_UNCLEAR_REASONS_ALLOWED = {
    "time_signature_not_clear",
    "too_dense_to_count",
    "crop_cut_off",
    "split_may_be_wrong",
    "ornament_or_tie_confusion",
    "not_enough_visual_evidence",
}
AI_SUGGESTION_DEBUG_DURATION_ALLOWED = {"full", "short", "unclear"}
AI_SUGGESTION_DEBUG_RHYTHM_ALLOWED = {
    "single_event",
    "chord_single_event",
    "multiple_events",
    "rest_or_silence",
    "unclear",
}
AI_SUGGESTION_DEBUG_REASON_ALLOWED = {
    "fills_meter",
    "short_for_meter",
    "meter_unclear",
    "rhythm_unclear",
    "not_first_measure",
    "other",
}
AI_SUGGESTION_DEBUG_NOTEHEAD_FILL_ALLOWED = {"filled", "open", "unclear"}
AI_SUGGESTION_DEBUG_STEM_OR_BEAM_ALLOWED = {"stem", "flag_or_beam", "none", "unclear"}
AI_SUGGESTION_DEBUG_DOT_SEEN_ALLOWED = {"true", "false", "unclear"}
AI_SUGGESTION_DEBUG_NOTE_VALUE_ALLOWED = {"quarter", "half", "whole", "eighth", "other", "unclear"}
AI_SUGGEST_OVERLOAD_RETRY_DELAYS_SEC = (2.0, 5.0)
AI_REFERENCE_EXAMPLES_DIR = Path(__file__).resolve().parent / "reference_examples"
AI_OLD_STYLE_REFERENCE_EXAMPLES = (
    {
        "filename": "old_style_rest_negative_1.png",
        "caption": "Reference example A: visible count 1 with an old-style-looking symbol. This is normal, not multi_measure_rest.",
    },
    {
        "filename": "old_style_rest_positive_3.png",
        "caption": "Reference example B: visible count 3 with an old-style symbol. This is multi_measure_rest.",
    },
)

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


class AiSuggestError(RuntimeError):
    def __init__(
        self,
        message: str = "Claude suggestion request failed.",
        *,
        code: str = "ai_suggest_failed",
        retryable: bool = True,
        provider_status: int = 502,
        detail: str = "",
        debug_crops: dict | None = None,
    ):
        super().__init__(message)
        self.message = str(message or "Claude suggestion request failed.")
        self.code = str(code or "ai_suggest_failed")
        self.retryable = bool(retryable)
        self.provider_status = int(provider_status)
        self.detail = str(detail or "")
        self.debug_crops = dict(debug_crops) if isinstance(debug_crops, dict) else None


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


def _upload_bytes_to_gcs(data: bytes, dest_uri: str, content_type: str | None = None) -> None:
    bucket_name, blob_name = _parse_gs_uri(dest_uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type or "application/octet-stream")


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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cropbox_offsets(page: fitz.Page) -> tuple[float, float]:
    cropbox = getattr(page, "cropbox", None)
    if cropbox is None:
        return (0.0, 0.0)
    return (
        _safe_float(getattr(cropbox, "x0", 0.0), 0.0),
        _safe_float(getattr(cropbox, "y0", 0.0), 0.0),
    )


def _green_box_point_to_pdf_ink(page: fitz.Page, x: float, y: float) -> tuple[float, float]:
    crop_x, crop_y = _cropbox_offsets(page)
    return (float(x) - crop_x, float(y) - crop_y)


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
        "auto_rows": editable_state.get("auto_rows") or [],
        "manual_rows": editable_state.get("manual_rows") or [],
        "measure_number_overrides": editable_state.get("measure_number_overrides") or {},
        "rest_measures": editable_state.get("rest_measures") or {},
        "pickup_measures": editable_state.get("pickup_measures") or {},
        "rest_systems": editable_state.get("rest_systems") or {},
        "endings": editable_state.get("endings") or {},
        "label_erase_areas": editable_state.get("label_erase_areas") or [],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _row_source(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ROW_SOURCE_AUTO
    raw = str(row.get("source") or "").strip().lower()
    if raw == ROW_SOURCE_MANUAL:
        return ROW_SOURCE_MANUAL
    if row.get("manual_row_id") is not None:
        return ROW_SOURCE_MANUAL
    system_id = str(row.get("system_id") or "").strip().lower()
    if system_id.startswith(MANUAL_SYSTEM_ID_PREFIX):
        return ROW_SOURCE_MANUAL
    measure_id = str(row.get("measure_id") or "").strip().lower()
    if measure_id.startswith(MANUAL_MEASURE_ID_PREFIX):
        return ROW_SOURCE_MANUAL
    return ROW_SOURCE_AUTO


def _is_manual_row_source(row: dict | None) -> bool:
    return _row_source(row) == ROW_SOURCE_MANUAL


def _manual_system_id(manual_row_id: str) -> str:
    return f"{MANUAL_SYSTEM_ID_PREFIX}{_normalize_artifact_key(manual_row_id) or 'row'}"


def _manual_measure_id(manual_row_id: str, measure_local_index: int) -> str:
    safe_row_id = _normalize_artifact_key(manual_row_id) or "row"
    return f"{MANUAL_MEASURE_ID_PREFIX}{safe_row_id}_m{max(0, int(measure_local_index))}"


def _parse_manual_row_rect(raw_rect: dict | None) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_rect, dict):
        return None
    try:
        left = float(raw_rect.get("left"))
        right = float(raw_rect.get("right"))
        top = float(raw_rect.get("top"))
        bottom = float(raw_rect.get("bottom"))
    except Exception:
        return None
    if right <= left or bottom <= top:
        return None
    return (left, right, top, bottom)


def _normalize_label_erase_area(raw_area: dict | None) -> dict | None:
    if not isinstance(raw_area, dict):
        return None
    page = _safe_int(raw_area.get("page"), 0)
    if page <= 0:
        return None
    rect = raw_area.get("rect")
    parsed = _parse_manual_row_rect(rect if isinstance(rect, dict) else None)
    if parsed is None:
        return None
    left, right, top, bottom = parsed
    if (right - left) > 96.0 or (bottom - top) > 48.0:
        return None
    return {
        "page": int(page),
        "rect": {
            "left": float(left),
            "right": float(right),
            "top": float(top),
            "bottom": float(bottom),
        },
    }


def _editable_label_erase_areas(editable_state: dict) -> list[dict]:
    raw_areas = editable_state.get("label_erase_areas")
    if not isinstance(raw_areas, list):
        editable_state["label_erase_areas"] = []
        return []

    cleaned: list[dict] = []
    seen: set[str] = set()
    for raw_area in raw_areas:
        area = _normalize_label_erase_area(raw_area if isinstance(raw_area, dict) else None)
        if area is None:
            continue
        rect = area["rect"]
        key = (
            f"{area['page']}|{rect['left']:.2f}|{rect['right']:.2f}|"
            f"{rect['top']:.2f}|{rect['bottom']:.2f}"
        )
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(area)

    editable_state["label_erase_areas"] = cleaned
    return cleaned


def _is_excluded_from_counting(row: dict | None) -> bool:
    if not isinstance(row, dict):
        return False
    return _safe_bool(row.get("excluded_from_counting"), False)


def _editable_manual_rows(editable_state: dict) -> list[dict]:
    raw_rows = editable_state.get("manual_rows")
    if not isinstance(raw_rows, list):
        editable_state["manual_rows"] = []
        return []

    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        manual_row_id = _normalize_artifact_key(raw_row.get("manual_row_id"))[:64]
        if not manual_row_id or manual_row_id in seen_ids:
            continue
        page = _safe_int(raw_row.get("page"), 0)
        if page <= 0:
            continue
        staff_kind = str(raw_row.get("staff_kind") or "").strip().lower()
        if staff_kind not in MANUAL_STAFF_KINDS_ALLOWED:
            continue
        rect_tuple = _parse_manual_row_rect(raw_row.get("rect"))
        if rect_tuple is None:
            continue
        left, right, top, bottom = rect_tuple

        raw_cut_xs = raw_row.get("cut_xs")
        if not isinstance(raw_cut_xs, list):
            raw_cut_xs = []
        cut_xs: list[float] = []
        prev_cut: float | None = None
        valid = True
        for raw_cut in raw_cut_xs:
            try:
                cut = float(raw_cut)
            except Exception:
                valid = False
                break
            if cut <= left or cut >= right:
                valid = False
                break
            if prev_cut is not None and cut <= prev_cut:
                valid = False
                break
            cut_xs.append(float(cut))
            prev_cut = float(cut)
        if not valid:
            continue

        seen_ids.add(manual_row_id)
        cleaned.append(
            {
                "manual_row_id": manual_row_id,
                "page": int(page),
                "staff_kind": staff_kind,
                "rect": {
                    "left": float(left),
                    "right": float(right),
                    "top": float(top),
                    "bottom": float(bottom),
                },
                "cut_xs": [float(cut) for cut in cut_xs],
            }
        )

    cleaned.sort(
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float(((row.get("rect") or {}).get("top")) or 0.0),
            float(((row.get("rect") or {}).get("left")) or 0.0),
            str(row.get("manual_row_id") or ""),
        )
    )
    editable_state["manual_rows"] = cleaned
    return cleaned


def _build_auto_rows_from_state(editable_state: dict) -> list[dict]:
    systems = _clone_auto_system_rows(editable_state)
    measures = _clone_auto_measure_rows(editable_state)
    system_rows_by_id: dict[str, dict] = {}
    grouped_measures: dict[tuple[int, str], list[dict]] = {}
    for system in systems:
        system_id = str(system.get("system_id") or "").strip()
        if system_id:
            system_rows_by_id[system_id] = system
    for measure in measures:
        system_id = str(measure.get("system_id") or "").strip()
        page = _safe_int(measure.get("page"), 0)
        if not system_id or page <= 0:
            continue
        grouped_measures.setdefault((page, system_id), []).append(measure)

    rows: list[dict] = []
    for (page, system_id), group in grouped_measures.items():
        system_row = system_rows_by_id.get(system_id) or {}
        bounds = _system_visual_bounds(system_row, group)
        if bounds is None:
            continue
        left, right, top, bottom = bounds
        if right <= left or bottom <= top:
            continue
        ordered_group = sorted(group, key=lambda row: (_safe_float(row.get("x_left"), 0.0), str(row.get("measure_id") or "")))
        boxes: list[dict] = []
        for measure in ordered_group:
            measure_id = str(measure.get("measure_id") or "").strip()
            box_left = _safe_float(measure.get("x_left"), left)
            box_right = _safe_float(measure.get("x_right"), box_left)
            if not measure_id or box_right <= box_left:
                continue
            boxes.append(
                {
                    "measure_id": measure_id,
                    "left": float(box_left),
                    "right": float(box_right),
                    "excluded_from_counting": _is_excluded_from_counting(measure),
                }
            )
        if not boxes:
            continue
        row = {
            "system_id": system_id,
            "page": int(page),
            "rect": {
                "left": float(left),
                "right": float(right),
                "top": float(top),
                "bottom": float(bottom),
            },
            "boxes": boxes,
        }
        current_value = str(system_row.get("current_value") or system_row.get("value") or "").strip()
        if current_value:
            row["current_value"] = current_value
        staff_kind = str(system_row.get("staff_kind") or "").strip().lower()
        if staff_kind:
            row["staff_kind"] = staff_kind
        rows.append(row)

    rows.sort(
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("top")) or 0.0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("left")) or 0.0),
            str(row.get("system_id") or ""),
        )
    )
    return rows


def _editable_auto_rows(editable_state: dict) -> list[dict]:
    raw_rows = editable_state.get("auto_rows")
    if not isinstance(raw_rows, list):
        derived = _build_auto_rows_from_state(editable_state)
        editable_state["auto_rows"] = derived
        return derived

    cleaned: list[dict] = []
    seen_system_ids: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        system_id = _normalize_artifact_key(raw_row.get("system_id"))[:128]
        page = _safe_int(raw_row.get("page"), 0)
        rect_tuple = _parse_manual_row_rect(raw_row.get("rect"))
        raw_boxes = raw_row.get("boxes")
        if not system_id or page <= 0 or rect_tuple is None or not isinstance(raw_boxes, list):
            continue
        if system_id in seen_system_ids:
            continue
        left, right, top, bottom = rect_tuple
        boxes: list[dict] = []
        seen_measure_ids: set[str] = set()
        last_right = left
        for raw_box in raw_boxes:
            if not isinstance(raw_box, dict):
                continue
            measure_id = _normalize_artifact_key(raw_box.get("measure_id"))[:128]
            box_left = _safe_float(raw_box.get("left"), left)
            box_right = _safe_float(raw_box.get("right"), box_left)
            if (
                not measure_id
                or measure_id in seen_measure_ids
                or box_right <= box_left
                or box_left < left
                or box_right > right
                or box_left < last_right
            ):
                continue
            boxes.append(
                {
                    "measure_id": measure_id,
                    "left": float(box_left),
                    "right": float(box_right),
                    "excluded_from_counting": _safe_bool(raw_box.get("excluded_from_counting"), False),
                }
            )
            seen_measure_ids.add(measure_id)
            last_right = box_right
        if not boxes:
            continue
        cleaned_row = {
            "system_id": system_id,
            "page": int(page),
            "rect": {
                "left": float(left),
                "right": float(right),
                "top": float(top),
                "bottom": float(bottom),
            },
            "boxes": boxes,
        }
        current_value = str(raw_row.get("current_value") or "").strip()
        if current_value:
            cleaned_row["current_value"] = current_value
        staff_kind = str(raw_row.get("staff_kind") or "").strip().lower()
        if staff_kind:
            cleaned_row["staff_kind"] = staff_kind
        cleaned.append(cleaned_row)
        seen_system_ids.add(system_id)

    cleaned.sort(
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("top")) or 0.0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("left")) or 0.0),
            str(row.get("system_id") or ""),
        )
    )
    editable_state["auto_rows"] = cleaned
    return cleaned


def _clone_auto_system_rows(editable_state: dict) -> list[dict]:
    rows = editable_state.get("systems")
    if not isinstance(rows, list):
        return []
    cloned: list[dict] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict) or _is_manual_row_source(raw_row):
            continue
        row = dict(raw_row)
        row["source"] = ROW_SOURCE_AUTO
        row.pop("manual_row_id", None)
        cloned.append(row)
    return cloned


def _clone_auto_measure_rows(editable_state: dict) -> list[dict]:
    rows = editable_state.get("measures")
    if not isinstance(rows, list):
        return []
    cloned: list[dict] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict) or _is_manual_row_source(raw_row):
            continue
        row = dict(raw_row)
        row["source"] = ROW_SOURCE_AUTO
        row.pop("manual_row_id", None)
        cloned.append(row)
    return cloned


def _system_visual_bounds(system_row: dict | None, measures: list[dict] | None = None) -> tuple[float, float, float, float] | None:
    if not isinstance(system_row, dict):
        return None

    left_vals: list[float] = []
    right_vals: list[float] = []
    top_vals: list[float] = []
    bottom_vals: list[float] = []
    system_id = str(system_row.get("system_id") or "").strip()

    for measure in measures or []:
        if not isinstance(measure, dict):
            continue
        if str(measure.get("system_id") or "").strip() != system_id:
            continue
        try:
            left = float(measure.get("x_left"))
            right = float(measure.get("x_right")) if measure.get("x_right") is not None else float(measure.get("x_left"))
            top = float(measure.get("y_top"))
            bottom = float(measure.get("y_bottom")) if measure.get("y_bottom") is not None else float(measure.get("y_top"))
        except Exception:
            continue
        if right > left:
            left_vals.append(left)
            right_vals.append(right)
        top_vals.append(top)
        if bottom > top:
            bottom_vals.append(bottom)

    anchor = system_row.get("anchor")
    if isinstance(anchor, dict):
        try:
            top_vals.append(float(anchor.get("y_top")))
            bottom_vals.append(float(anchor.get("y_bottom")))
        except Exception:
            pass
        try:
            anchor_x = float(anchor.get("x"))
            if not left_vals:
                left_vals.append(anchor_x)
            if not right_vals:
                right_vals.append(anchor_x + 1.0)
        except Exception:
            pass

    for key, target in (("x_left", left_vals), ("x_right", right_vals), ("y_top", top_vals), ("y_bottom", bottom_vals)):
        try:
            value = float(system_row.get(key))
        except Exception:
            continue
        if key == "x_left":
            target.append(value)
        elif key == "x_right":
            if left_vals:
                target.append(value)
        elif key == "y_top":
            target.append(value)
        elif key == "y_bottom":
            target.append(value)

    if not left_vals or not right_vals or not top_vals or not bottom_vals:
        return None
    left = min(left_vals)
    right = max(right_vals)
    top = min(top_vals)
    bottom = max(bottom_vals)
    if right <= left or bottom <= top:
        return None
    return (float(left), float(right), float(top), float(bottom))


def _axis_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    span = min(max(0.0, a1 - a0), max(0.0, b1 - b0))
    if span <= 0.0:
        return 0.0
    overlap = min(a1, b1) - max(a0, b0)
    if overlap <= 0.0:
        return 0.0
    return float(overlap / span)


def _rects_strongly_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    x_ratio = _axis_overlap_ratio(ax0, ax1, bx0, bx1)
    y_ratio = _axis_overlap_ratio(ay0, ay1, by0, by1)
    return x_ratio >= MANUAL_ROW_OVERLAP_RATIO and y_ratio >= MANUAL_ROW_OVERLAP_RATIO


def _normalize_manual_rows_payload(
    page: int,
    raw_rows: list,
    editable_state: dict,
) -> tuple[list[dict] | None, str | None]:
    if page <= 0:
        return None, "invalid_page"
    if not isinstance(raw_rows, list):
        return None, "invalid_rows_payload"

    auto_systems = [row for row in _clone_auto_system_rows(editable_state) if _safe_int(row.get("page"), 0) == page]
    auto_measures = [row for row in _clone_auto_measure_rows(editable_state) if _safe_int(row.get("page"), 0) == page]
    auto_rects = [
        bounds
        for bounds in (_system_visual_bounds(system_row, auto_measures) for system_row in auto_systems)
        if bounds is not None
    ]

    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    seen_rects: list[tuple[float, float, float, float]] = []

    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            return None, "invalid_manual_row"
        manual_row_id = _normalize_artifact_key(raw_row.get("manual_row_id"))[:64]
        if not manual_row_id or manual_row_id in seen_ids:
            return None, "duplicate_manual_row_id"

        row_page = _safe_int(raw_row.get("page"), page)
        if row_page != page:
            return None, "manual_row_page_mismatch"

        staff_kind = str(raw_row.get("staff_kind") or "").strip().lower()
        if staff_kind not in MANUAL_STAFF_KINDS_ALLOWED:
            return None, "invalid_staff_kind"

        rect_tuple = _parse_manual_row_rect(raw_row.get("rect"))
        if rect_tuple is None:
            return None, "invalid_manual_rect"
        left, right, top, bottom = rect_tuple

        raw_cut_xs = raw_row.get("cut_xs")
        if not isinstance(raw_cut_xs, list):
            return None, "invalid_cut_xs"
        cut_xs: list[float] = []
        prev_cut: float | None = None
        for raw_cut in raw_cut_xs:
            try:
                cut = float(raw_cut)
            except Exception:
                return None, "invalid_cut_xs"
            if cut <= left or cut >= right:
                return None, "invalid_cut_xs"
            if prev_cut is not None and cut <= prev_cut:
                return None, "invalid_cut_xs"
            cut_xs.append(float(cut))
            prev_cut = float(cut)

        rect = (float(left), float(right), float(top), float(bottom))
        if any(_rects_strongly_overlap(rect, auto_rect) for auto_rect in auto_rects):
            return None, "manual_row_overlap_auto"
        if any(_rects_strongly_overlap(rect, prior_rect) for prior_rect in seen_rects):
            return None, "manual_row_overlap_manual"

        seen_ids.add(manual_row_id)
        seen_rects.append(rect)
        cleaned.append(
            {
                "manual_row_id": manual_row_id,
                "page": int(page),
                "staff_kind": staff_kind,
                "rect": {
                    "left": float(left),
                    "right": float(right),
                    "top": float(top),
                    "bottom": float(bottom),
                },
                "cut_xs": [float(cut) for cut in cut_xs],
            }
        )

    cleaned.sort(
        key=lambda row: (
            float(((row.get("rect") or {}).get("top")) or 0.0),
            float(((row.get("rect") or {}).get("left")) or 0.0),
            str(row.get("manual_row_id") or ""),
        )
    )
    return cleaned, None


def _build_auto_rows_overlay(auto_rows: list[dict], editable_state: dict) -> tuple[list[dict], list[dict]]:
    systems: list[dict] = []
    measures: list[dict] = []
    existing_systems_by_id: dict[str, dict] = {}
    for row in _clone_auto_system_rows(editable_state):
        system_id = str(row.get("system_id") or "").strip()
        if system_id:
            existing_systems_by_id[system_id] = row

    for auto_row in auto_rows or []:
        if not isinstance(auto_row, dict):
            continue
        system_id = str(auto_row.get("system_id") or "").strip()
        page = _safe_int(auto_row.get("page"), 0)
        rect = (auto_row.get("rect") or {}) if isinstance(auto_row.get("rect"), dict) else {}
        left = _safe_float(rect.get("left"), 0.0)
        right = _safe_float(rect.get("right"), left)
        top = _safe_float(rect.get("top"), 0.0)
        bottom = _safe_float(rect.get("bottom"), top)
        boxes = auto_row.get("boxes")
        if not system_id or page <= 0 or right <= left or bottom <= top or not isinstance(boxes, list):
            continue
        existing_system = existing_systems_by_id.get(system_id) or {}
        system_index = _safe_int(existing_system.get("system_index"), 0)
        current_value = str(auto_row.get("current_value") or existing_system.get("current_value") or existing_system.get("value") or "").strip()
        staff_kind = str(auto_row.get("staff_kind") or existing_system.get("staff_kind") or "").strip().lower()
        systems.append(
            {
                "system_id": system_id,
                "page": int(page),
                "system_index": int(system_index),
                "current_value": current_value,
                "value": current_value,
                "render_label": current_value,
                "source": ROW_SOURCE_AUTO,
                "staff_kind": staff_kind or None,
                "anchor": {"x": float(left), "y_top": float(top), "y_bottom": float(bottom)},
                "x_left": float(left),
                "x_right": float(right),
                "y_top": float(top),
                "y_bottom": float(bottom),
            }
        )
        ordered_boxes = sorted(
            [box for box in boxes if isinstance(box, dict)],
            key=lambda box: (_safe_float(box.get("left"), 0.0), str(box.get("measure_id") or "")),
        )
        for local_idx, box in enumerate(ordered_boxes):
            measure_id = str(box.get("measure_id") or "").strip()
            box_left = _safe_float(box.get("left"), left)
            box_right = _safe_float(box.get("right"), box_left)
            if not measure_id or box_right <= box_left:
                continue
            measures.append(
                {
                    "measure_id": measure_id,
                    "system_id": system_id,
                    "page": int(page),
                    "system_index": int(system_index),
                    "measure_local_index": int(local_idx),
                    "global_index": 0,
                    "x_left": float(box_left),
                    "x_right": float(box_right),
                    "y_top": float(top),
                    "y_bottom": float(bottom),
                    "source": ROW_SOURCE_AUTO,
                    "staff_kind": staff_kind or None,
                    "excluded_from_counting": _safe_bool(box.get("excluded_from_counting"), False),
                }
            )
    return systems, measures


def _normalize_auto_rows_payload(
    page: int,
    rows: list[dict] | None,
    editable_state: dict,
) -> tuple[list[dict] | None, str | None]:
    current_auto_rows = [row for row in _editable_auto_rows(editable_state) if _safe_int(row.get("page"), 0) == page]
    expected_system_ids = {
        str(row.get("system_id") or "").strip()
        for row in current_auto_rows
        if isinstance(row, dict) and str(row.get("system_id") or "").strip()
    }
    if not isinstance(rows, list):
        return None, "invalid_auto_rows"
    if not expected_system_ids and rows:
        return None, "unexpected_auto_rows"

    cleaned: list[dict] = []
    seen_system_ids: set[str] = set()
    seen_measure_ids: set[str] = set()
    current_by_system_id = {
        str(row.get("system_id") or "").strip(): row
        for row in current_auto_rows
        if isinstance(row, dict) and str(row.get("system_id") or "").strip()
    }

    for raw_row in rows:
        if not isinstance(raw_row, dict):
            return None, "invalid_auto_row"
        system_id = _normalize_artifact_key(raw_row.get("system_id"))[:128]
        if not system_id or system_id in seen_system_ids:
            return None, "duplicate_auto_system_id"
        if system_id not in expected_system_ids:
            return None, "unknown_auto_system_id"
        if _safe_int(raw_row.get("page"), 0) != page:
            return None, "auto_row_page_mismatch"
        rect_tuple = _parse_manual_row_rect(raw_row.get("rect"))
        if rect_tuple is None:
            return None, "invalid_auto_row_rect"
        left, right, top, bottom = rect_tuple
        current_row = current_by_system_id.get(system_id) or {}
        current_rect = _parse_manual_row_rect(current_row.get("rect"))
        if current_rect is None:
            return None, "missing_auto_row_baseline"
        raw_boxes = raw_row.get("boxes")
        if not isinstance(raw_boxes, list):
            return None, "invalid_auto_boxes"
        boxes: list[dict] = []
        last_right = left
        for raw_box in raw_boxes:
            if not isinstance(raw_box, dict):
                return None, "invalid_auto_box"
            measure_id = _normalize_artifact_key(raw_box.get("measure_id"))[:128]
            box_left = _safe_float(raw_box.get("left"), left)
            box_right = _safe_float(raw_box.get("right"), box_left)
            if (
                not measure_id
                or measure_id in seen_measure_ids
                or box_right <= box_left
                or (box_right - box_left) < AUTO_BOX_MIN_WIDTH
                or box_left < left
                or box_right > right
                or box_left < last_right
            ):
                return None, "invalid_auto_box"
            boxes.append(
                {
                    "measure_id": measure_id,
                    "left": float(box_left),
                    "right": float(box_right),
                    "excluded_from_counting": _safe_bool(raw_box.get("excluded_from_counting"), False),
                }
            )
            seen_measure_ids.add(measure_id)
            last_right = box_right
        if not boxes:
            return None, "auto_row_missing_boxes"
        cleaned_row = {
            "system_id": system_id,
            "page": int(page),
            "rect": {
                "left": float(left),
                "right": float(right),
                "top": float(top),
                "bottom": float(bottom),
            },
            "boxes": boxes,
        }
        current_value = str(current_row.get("current_value") or raw_row.get("current_value") or "").strip()
        if current_value:
            cleaned_row["current_value"] = current_value
        staff_kind = str(current_row.get("staff_kind") or raw_row.get("staff_kind") or "").strip().lower()
        if staff_kind:
            cleaned_row["staff_kind"] = staff_kind
        cleaned.append(cleaned_row)
        seen_system_ids.add(system_id)

    if seen_system_ids != expected_system_ids:
        return None, "auto_systems_mismatch"

    cleaned.sort(
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("top")) or 0.0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("left")) or 0.0),
            str(row.get("system_id") or ""),
        )
    )
    return cleaned, None


def _build_manual_rows_overlay(manual_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    systems: list[dict] = []
    measures: list[dict] = []
    for manual_row in manual_rows or []:
        if not isinstance(manual_row, dict):
            continue
        manual_row_id = str(manual_row.get("manual_row_id") or "").strip()
        rect = (manual_row.get("rect") or {}) if isinstance(manual_row.get("rect"), dict) else {}
        left = float(rect.get("left") or 0.0)
        right = float(rect.get("right") or 0.0)
        top = float(rect.get("top") or 0.0)
        bottom = float(rect.get("bottom") or 0.0)
        page = _safe_int(manual_row.get("page"), 0)
        if not manual_row_id or page <= 0 or right <= left or bottom <= top:
            continue
        staff_kind = str(manual_row.get("staff_kind") or MANUAL_STAFF_KIND_SINGLE).strip().lower()
        system_id = _manual_system_id(manual_row_id)
        systems.append(
            {
                "system_id": system_id,
                "page": int(page),
                "system_index": 0,
                "current_value": "",
                "value": "",
                "render_label": "",
                "source": ROW_SOURCE_MANUAL,
                "manual_row_id": manual_row_id,
                "staff_kind": staff_kind,
                "anchor": {"x": float(left), "y_top": float(top), "y_bottom": float(bottom)},
                "x_left": float(left),
                "x_right": float(right),
                "y_top": float(top),
                "y_bottom": float(bottom),
            }
        )
        boundaries = [float(left), *[float(cut) for cut in (manual_row.get("cut_xs") or [])], float(right)]
        for idx in range(len(boundaries) - 1):
            measures.append(
                {
                    "measure_id": _manual_measure_id(manual_row_id, idx),
                    "system_id": system_id,
                    "page": int(page),
                    "system_index": 0,
                    "measure_local_index": int(idx),
                    "global_index": 0,
                    "x_left": float(boundaries[idx]),
                    "x_right": float(boundaries[idx + 1]),
                    "y_top": float(top),
                    "y_bottom": float(bottom),
                    "source": ROW_SOURCE_MANUAL,
                    "manual_row_id": manual_row_id,
                    "staff_kind": staff_kind,
                }
            )
    return systems, measures


def _reindex_system_and_measure_order(systems: list[dict], measures: list[dict]) -> tuple[list[dict], list[dict]]:
    sorted_systems = sorted(
        [row for row in systems if isinstance(row, dict)],
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float((((row.get("anchor") or {}) if isinstance(row.get("anchor"), dict) else {}).get("y_top")) or row.get("y_top") or 0.0),
            float(row.get("x_left") or (((row.get("anchor") or {}) if isinstance(row.get("anchor"), dict) else {}).get("x")) or 0.0),
            0 if _row_source(row) == ROW_SOURCE_AUTO else 1,
            str(row.get("system_id") or ""),
        ),
    )

    system_index_by_id: dict[str, int] = {}
    next_index_by_page: dict[int, int] = {}
    for system in sorted_systems:
        page = max(1, _safe_int(system.get("page"), 0))
        system_id = str(system.get("system_id") or "").strip()
        if not system_id:
            continue
        system["page"] = int(page)
        system["source"] = _row_source(system)
        system["system_index"] = int(next_index_by_page.get(page, 0))
        next_index_by_page[page] = int(system["system_index"]) + 1
        system_index_by_id[system_id] = int(system["system_index"])

    grouped_measures: dict[tuple[int, str], list[dict]] = {}
    for measure in measures:
        if not isinstance(measure, dict):
            continue
        system_id = str(measure.get("system_id") or "").strip()
        page = max(1, _safe_int(measure.get("page"), 0))
        if not system_id:
            continue
        measure["page"] = int(page)
        measure["source"] = _row_source(measure)
        if system_id in system_index_by_id:
            measure["system_index"] = int(system_index_by_id[system_id])
        grouped_measures.setdefault((int(page), system_id), []).append(measure)

    ordered_measures: list[dict] = []
    global_index = 0
    for system in sorted_systems:
        system_id = str(system.get("system_id") or "").strip()
        page = _safe_int(system.get("page"), 0)
        group = grouped_measures.get((page, system_id), [])
        group.sort(
            key=lambda row: (
                float(row.get("x_left") or 0.0),
                float(row.get("x_right") or row.get("x_left") or 0.0),
                _safe_int(row.get("measure_local_index"), 0),
                str(row.get("measure_id") or ""),
            )
        )
        for local_index, measure in enumerate(group):
            measure["measure_local_index"] = int(local_index)
            measure["global_index"] = int(global_index)
            global_index += 1
            if _row_source(measure) == ROW_SOURCE_MANUAL:
                manual_row_id = _normalize_artifact_key(measure.get("manual_row_id"))[:64]
                if manual_row_id:
                    measure["measure_id"] = _manual_measure_id(manual_row_id, local_index)
            ordered_measures.append(measure)
    return sorted_systems, ordered_measures


def _merge_manual_rows_into_state(editable_state: dict) -> tuple[list[dict], list[dict]]:
    manual_rows = _editable_manual_rows(editable_state)
    base_auto_systems = _clone_auto_system_rows(editable_state)
    base_auto_measures = _clone_auto_measure_rows(editable_state)
    auto_rows = _editable_auto_rows(editable_state)
    overlay_auto_systems, overlay_auto_measures = _build_auto_rows_overlay(auto_rows, editable_state)
    if overlay_auto_systems:
        replaced_system_ids = {
            str(row.get("system_id") or "").strip()
            for row in overlay_auto_systems
            if isinstance(row, dict) and str(row.get("system_id") or "").strip()
        }
        auto_systems = [
            row for row in base_auto_systems
            if str(row.get("system_id") or "").strip() not in replaced_system_ids
        ]
        auto_systems.extend(overlay_auto_systems)
        auto_measures = [
            row for row in base_auto_measures
            if str(row.get("system_id") or "").strip() not in replaced_system_ids
        ]
        auto_measures.extend(overlay_auto_measures)
    else:
        auto_systems = base_auto_systems
        auto_measures = base_auto_measures
    manual_systems, manual_measures = _build_manual_rows_overlay(manual_rows)
    return _reindex_system_and_measure_order(auto_systems + manual_systems, auto_measures + manual_measures)


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


def _current_ai_suggestions(mapping_summary: dict | None) -> dict | None:
    ai_suggestions = (mapping_summary or {}).get("ai_suggestions")
    return ai_suggestions if isinstance(ai_suggestions, dict) else None


def _requested_ai_provider_name() -> str:
    provider = str(os.environ.get("AI_PROVIDER", AI_PROVIDER) or "").strip().lower()
    if provider in {"bedrock", "anthropic"}:
        return provider
    return "unknown"


def _configured_anthropic_model_name() -> str:
    return str(os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL) or "").strip()


def _configured_bedrock_model_id() -> str:
    return str(os.environ.get("BEDROCK_MODEL_ID", BEDROCK_MODEL_ID) or "").strip()


def _requested_ai_model_name() -> str:
    provider = _requested_ai_provider_name()
    if provider == "bedrock":
        return _configured_bedrock_model_id() or "unknown"
    if provider == "anthropic":
        return _configured_anthropic_model_name() or "unknown"
    return "unknown"


def _requested_anthropic_model_name() -> str:
    return _configured_anthropic_model_name() or "unknown"


def _current_ai_suggest_run(
    mapping_summary: dict | None,
    run_id: int | None = None,
    source_state_version: str | None = None,
) -> dict:
    raw = (mapping_summary or {}).get("ai_suggest_run")
    row = dict(raw) if isinstance(raw, dict) else {}
    status = str(row.get("status") or AI_SUGGEST_RUN_STATUS_IDLE).strip().lower()
    if status not in {
        AI_SUGGEST_RUN_STATUS_IDLE,
        AI_SUGGEST_RUN_STATUS_RUNNING,
        AI_SUGGEST_RUN_STATUS_COMPLETED,
        AI_SUGGEST_RUN_STATUS_FAILED,
    }:
        status = AI_SUGGEST_RUN_STATUS_IDLE
    remembered_time_signature = _normalize_ai_time_signature_value(row.get("remembered_time_signature"))
    last_time_signature_update = _normalize_ai_time_signature_update_row(row.get("last_time_signature_update"))
    time_signature_updates = _normalize_ai_time_signature_update_rows(row.get("time_signature_updates"))
    clean = {
        "status": status,
        "started_at_utc": str(row.get("started_at_utc") or "").strip() or None,
        "updated_at_utc": str(row.get("updated_at_utc") or "").strip() or None,
        "completed_at_utc": str(row.get("completed_at_utc") or "").strip() or None,
        "failed_at_utc": str(row.get("failed_at_utc") or "").strip() or None,
        "systems_total": max(0, _safe_int(row.get("systems_total"), 0)),
        "systems_completed": max(0, _safe_int(row.get("systems_completed"), 0)),
        "next_system_index": max(0, _safe_int(row.get("next_system_index"), 0)),
        "source_run_id": int(run_id) if isinstance(run_id, int) and run_id > 0 else _safe_int(row.get("source_run_id"), 0),
        "source_state_version": str(row.get("source_state_version") or source_state_version or "").strip() or None,
        "score_type": _normalize_ai_score_type(row.get("score_type")),
        "model": str(row.get("model") or _requested_ai_model_name()).strip() or "unknown",
        "last_error": row.get("last_error") if isinstance(row.get("last_error"), dict) else None,
        "remembered_time_signature": remembered_time_signature,
        "last_time_signature_update": last_time_signature_update,
        "time_signature_updates": time_signature_updates,
    }
    return clean


def _empty_ai_suggestions_state(
    run_id: int,
    source_state_version: str | None,
    measures_seen: int,
) -> dict:
    ai_suggestions = {
        "version": AI_SUGGESTIONS_VERSION,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "provider": _requested_ai_provider_name(),
        "model": _requested_ai_model_name(),
        "source_run_id": int(run_id),
        "by_measure_id": {},
        "decision_debug_by_measure_id": {},
        "time_signatures_by_measure_id": {},
        "measure_completeness_by_measure_id": {},
        "warnings": [],
        "summary": {
            "systems_processed": 0,
            "measures_seen": 0,
            "suggestions_kept": 0,
            "normal_measures_omitted": 0,
        },
    }
    source_state_version_txt = str(source_state_version or "").strip()
    if source_state_version_txt:
        ai_suggestions["source_state_version"] = source_state_version_txt
    return ai_suggestions


def _new_ai_suggest_run_state(
    run_id: int,
    source_state_version: str | None,
    systems_total: int,
    status: str = AI_SUGGEST_RUN_STATUS_RUNNING,
    score_type: str | None = None,
) -> dict:
    now_txt = _utc_now().isoformat().replace("+00:00", "Z")
    row = {
        "status": status,
        "started_at_utc": now_txt if status in {AI_SUGGEST_RUN_STATUS_RUNNING, AI_SUGGEST_RUN_STATUS_COMPLETED} else None,
        "updated_at_utc": now_txt,
        "completed_at_utc": now_txt if status == AI_SUGGEST_RUN_STATUS_COMPLETED else None,
        "failed_at_utc": now_txt if status == AI_SUGGEST_RUN_STATUS_FAILED else None,
        "systems_total": max(0, int(systems_total)),
        "systems_completed": max(0, int(systems_total)) if status == AI_SUGGEST_RUN_STATUS_COMPLETED else 0,
        "next_system_index": max(0, int(systems_total)) if status == AI_SUGGEST_RUN_STATUS_COMPLETED else 0,
        "source_run_id": int(run_id),
        "source_state_version": str(source_state_version or "").strip() or None,
        "score_type": _normalize_ai_score_type(score_type),
        "model": _requested_ai_model_name(),
        "last_error": None,
        "remembered_time_signature": None,
        "last_time_signature_update": None,
        "time_signature_updates": [],
    }
    return row


def _normalize_ai_score_type(raw_value) -> str | None:
    text = str(raw_value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in AI_SCORE_TYPES_ALLOWED else None


def _normalize_ai_time_signature_value(raw_value) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    compact = text.lower().replace("-", "_").replace(" ", "_")
    if compact in {"common_time", "common", "commonmeter", "c"}:
        return "common_time"
    if compact in {"cut_time", "cut", "alla_breve", "cuttime"}:
        return "cut_time"
    fraction = compact.replace("_", "")
    if re.fullmatch(r"\d{1,2}/\d{1,2}", fraction):
        return fraction
    return None


def _normalize_ai_time_signature_update_row(raw_row, system_id: str | None = None) -> dict | None:
    if not isinstance(raw_row, dict):
        return None
    new_time_signature = _normalize_ai_time_signature_value(raw_row.get("new_time_signature"))
    if not new_time_signature:
        return None
    measure_id = str(raw_row.get("measure_id") or "").strip() or None
    resolved_system_id = str(raw_row.get("system_id") or system_id or "").strip() or None
    return {
        "system_id": resolved_system_id,
        "measure_id": measure_id,
        "new_time_signature": new_time_signature,
    }


def _normalize_ai_time_signature_update_rows(raw_rows, system_id: str | None = None) -> list[dict]:
    clean: list[dict] = []
    if not isinstance(raw_rows, list):
        return clean
    for raw_row in raw_rows:
        normalized = _normalize_ai_time_signature_update_row(raw_row, system_id=system_id)
        if normalized:
            clean.append(normalized)
    return clean


def _ai_suggest_candidate_measures(editable_state: dict) -> list[dict]:
    measures = _sorted_measure_rows(editable_state.get("measures") or [])
    return [row for row in measures if not _is_excluded_from_counting(row)]


def _ai_suggest_system_batches(editable_state: dict) -> list[tuple[dict, list[dict]]]:
    systems = _sorted_system_rows(editable_state.get("systems") or [])
    measures = _ai_suggest_candidate_measures(editable_state)
    grouped_measures: dict[str, list[dict]] = {}
    for row in measures:
        system_id = str(row.get("system_id") or "").strip()
        if not system_id:
            continue
        grouped_measures.setdefault(system_id, []).append(row)
    batches: list[tuple[dict, list[dict]]] = []
    for system_row in systems:
        system_id = str(system_row.get("system_id") or "").strip()
        system_measures = grouped_measures.get(system_id) or []
        if not system_id or not system_measures:
            continue
        batches.append((system_row, system_measures))
    return batches


def _same_page_neighbor_systems(systems: list[dict], current_system_row: dict) -> tuple[dict | None, dict | None]:
    current_id = str((current_system_row or {}).get("system_id") or "").strip()
    current_page = _safe_int((current_system_row or {}).get("page"), 0)
    current_top, _ = _system_anchor_bounds(current_system_row)
    page_systems: list[tuple[float, dict]] = []
    for row in systems:
        if not isinstance(row, dict):
            continue
        if str(row.get("system_id") or "").strip() == current_id:
            continue
        if _safe_int(row.get("page"), 0) != current_page:
            continue
        row_top, _ = _system_anchor_bounds(row)
        if row_top is None:
            continue
        page_systems.append((row_top, row))
    page_systems.sort(key=lambda item: item[0])
    prev_system_row = None
    next_system_row = None
    if current_top is None:
        return (None, None)
    for row_top, row in page_systems:
        if row_top < current_top:
            prev_system_row = row
        elif next_system_row is None and row_top > current_top:
            next_system_row = row
            break
    return (prev_system_row, next_system_row)


def _merge_ai_suggestions_state(
    existing: dict | None,
    system_suggestions: dict,
    run_id: int,
    source_state_version: str | None,
) -> dict:
    base = dict(existing) if isinstance(existing, dict) else _empty_ai_suggestions_state(run_id, source_state_version, 0)
    by_measure_id = dict(base.get("by_measure_id") or {})
    by_measure_id.update(dict(system_suggestions.get("by_measure_id") or {}))
    decision_debug_by_measure_id = dict(base.get("decision_debug_by_measure_id") or {})
    decision_debug_by_measure_id.update(dict(system_suggestions.get("decision_debug_by_measure_id") or {}))
    time_signatures_by_measure_id = dict(base.get("time_signatures_by_measure_id") or {})
    time_signatures_by_measure_id.update(dict(system_suggestions.get("time_signatures_by_measure_id") or {}))
    measure_completeness_by_measure_id = dict(base.get("measure_completeness_by_measure_id") or {})
    measure_completeness_by_measure_id.update(dict(system_suggestions.get("measure_completeness_by_measure_id") or {}))
    warnings = list(base.get("warnings") or [])
    warnings.extend(list(system_suggestions.get("warnings") or []))
    base["version"] = AI_SUGGESTIONS_VERSION
    base["generated_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
    base["provider"] = str(system_suggestions.get("provider") or base.get("provider") or _requested_ai_provider_name()).strip() or _requested_ai_provider_name()
    base["model"] = _requested_ai_model_name()
    base["source_run_id"] = int(run_id)
    source_state_version_txt = str(source_state_version or "").strip()
    if source_state_version_txt:
        base["source_state_version"] = source_state_version_txt
    base["by_measure_id"] = by_measure_id
    base["decision_debug_by_measure_id"] = decision_debug_by_measure_id
    base["time_signatures_by_measure_id"] = time_signatures_by_measure_id
    base["measure_completeness_by_measure_id"] = measure_completeness_by_measure_id
    base["warnings"] = warnings
    summary = base.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    system_summary = system_suggestions.get("summary") if isinstance(system_suggestions.get("summary"), dict) else {}
    summary["systems_processed"] = max(0, _safe_int(summary.get("systems_processed"), 0)) + max(0, _safe_int(system_summary.get("systems_processed"), 0))
    summary["measures_seen"] = max(0, _safe_int(summary.get("measures_seen"), 0)) + max(0, _safe_int(system_summary.get("measures_seen"), 0))
    summary["normal_measures_omitted"] = max(0, _safe_int(summary.get("normal_measures_omitted"), 0)) + max(0, _safe_int(system_summary.get("normal_measures_omitted"), 0))
    summary["suggestions_kept"] = len(by_measure_id)
    base["summary"] = summary
    return base


def _ai_suggest_error_payload(exc: AiSuggestError | Exception, default_message: str = "Claude suggestion request failed.") -> dict:
    if isinstance(exc, AiSuggestError):
        payload = {
            "code": exc.code,
            "message": exc.message,
            "retryable": exc.retryable,
            "provider_status": exc.provider_status,
            "detail": exc.detail,
        }
        retry_attempts = getattr(exc, "retry_attempts", None)
        if isinstance(retry_attempts, int) and retry_attempts > 0:
            payload["retry_attempts"] = retry_attempts
        return payload
    return {
        "code": "ai_suggest_failed",
        "message": default_message,
        "retryable": True,
        "provider_status": 500,
        "detail": _safe_error_text(exc),
    }


def _refresh_ai_suggestions_summary(ai_suggestions: dict | None) -> dict:
    if not isinstance(ai_suggestions, dict):
        return {}
    by_measure_id = ai_suggestions.get("by_measure_id")
    if not isinstance(by_measure_id, dict):
        by_measure_id = {}
        ai_suggestions["by_measure_id"] = by_measure_id
    summary = ai_suggestions.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    summary["systems_processed"] = max(0, _safe_int(summary.get("systems_processed"), 0))
    summary["measures_seen"] = max(0, _safe_int(summary.get("measures_seen"), 0))
    summary["suggestions_kept"] = len(by_measure_id)
    summary["normal_measures_omitted"] = max(0, _safe_int(summary.get("normal_measures_omitted"), 0))
    ai_suggestions["summary"] = summary
    return summary


def _remove_ai_suggestion_entries(mapping_summary: dict | None, measure_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    ai_suggestions = _current_ai_suggestions(mapping_summary)
    if not isinstance(ai_suggestions, dict):
        return []
    by_measure_id = ai_suggestions.get("by_measure_id")
    if not isinstance(by_measure_id, dict):
        return []
    time_signatures_by_measure_id = ai_suggestions.get("time_signatures_by_measure_id")
    if not isinstance(time_signatures_by_measure_id, dict):
        time_signatures_by_measure_id = {}
    measure_completeness_by_measure_id = ai_suggestions.get("measure_completeness_by_measure_id")
    if not isinstance(measure_completeness_by_measure_id, dict):
        measure_completeness_by_measure_id = {}
    removed: list[str] = []
    maps_changed = False
    for measure_id in measure_ids or []:
        mid = str(measure_id or "").strip()
        if not mid:
            continue
        if mid in by_measure_id:
            by_measure_id.pop(mid, None)
            removed.append(mid)
        if mid in time_signatures_by_measure_id:
            time_signatures_by_measure_id.pop(mid, None)
            maps_changed = True
        if mid in measure_completeness_by_measure_id:
            measure_completeness_by_measure_id.pop(mid, None)
            maps_changed = True
    if removed or maps_changed:
        ai_suggestions["time_signatures_by_measure_id"] = time_signatures_by_measure_id
        ai_suggestions["measure_completeness_by_measure_id"] = measure_completeness_by_measure_id
        _refresh_ai_suggestions_summary(ai_suggestions)
    return removed


def _clear_measure_state_for_ids(editable_state: dict, measure_ids: set[str] | list[str] | tuple[str, ...]) -> None:
    ids = {str(measure_id or "").strip() for measure_id in (measure_ids or []) if str(measure_id or "").strip()}
    if not ids:
        return
    measure_overrides = _measure_number_overrides(editable_state)
    rest_measures = _editable_rest_measures(editable_state)
    pickup_measures = _editable_pickup_measures(editable_state)
    endings_map = _editable_endings_map(editable_state)
    for measure_id in ids:
        measure_overrides.pop(measure_id, None)
        rest_measures.pop(measure_id, None)
        pickup_measures.pop(measure_id, None)
        endings_map.pop(measure_id, None)


def _measure_ids_on_pages(
    measures: list[dict] | None,
    pages: set[int] | list[int] | tuple[int, ...],
    *,
    source: str | None = None,
) -> set[str]:
    page_set = {int(page) for page in (pages or []) if int(page) > 0}
    if not page_set:
        return set()
    ids: set[str] = set()
    for row in measures or []:
        if not isinstance(row, dict):
            continue
        if _safe_int(row.get("page"), 0) not in page_set:
            continue
        if source == ROW_SOURCE_MANUAL and _row_source(row) != ROW_SOURCE_MANUAL:
            continue
        if source == ROW_SOURCE_AUTO and _row_source(row) != ROW_SOURCE_AUTO:
            continue
        measure_id = str(row.get("measure_id") or "").strip()
        if measure_id:
            ids.add(measure_id)
    return ids


def _normalize_ai_suggest_warnings(raw_warnings) -> list[dict]:
    if raw_warnings is None or not isinstance(raw_warnings, list):
        return []
    clean: list[dict] = []
    for row in raw_warnings:
        if not isinstance(row, dict):
            continue
        warning_type = str(row.get("type") or "").strip()
        message = str(row.get("message") or "").strip()
        if not warning_type or not message:
            continue
        warning = {
            "type": warning_type,
            "message": message,
        }
        measure_id = str(row.get("measure_id") or "").strip()
        if measure_id:
            warning["measure_id"] = measure_id
        system_id = str(row.get("system_id") or "").strip()
        if system_id:
            warning["system_id"] = system_id
        if row.get("system_index") is not None:
            warning["system_index"] = _safe_int(row.get("system_index"), 0)
        clean.append(warning)
    return clean


def _normalize_ai_measure_completeness_value(raw_value) -> str | None:
    text = str(raw_value or "").strip().lower()
    return text if text in AI_SUGGESTION_COMPLETENESS_ALLOWED else None


def _normalize_ai_unclear_reason_value(raw_value) -> str | None:
    text = str(raw_value or "").strip().lower()
    return text if text in AI_SUGGESTION_UNCLEAR_REASONS_ALLOWED else None


def _normalize_ai_debug_note(raw_value, max_words: int = 50) -> str:
    text = re.sub(r"\s+", " ", str(raw_value or "").strip())
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def _normalize_ai_debug_short_text(raw_value, max_words: int = 8) -> str:
    if raw_value is None:
        return "unclear"
    text = re.sub(r"\s+", " ", str(raw_value).strip())
    if not text:
        return "unclear"
    lowered = text.lower()
    if lowered in {"unknown", "none", "null", "n/a"}:
        return "unclear"
    words = text.split()
    if len(words) > max_words:
        return "unclear"
    return text


def _normalize_ai_decision_debug(raw_debug) -> dict | None:
    if not isinstance(raw_debug, dict):
        return None
    active_meter = _normalize_ai_time_signature_value(raw_debug.get("active_meter_read")) or "unknown"
    duration = str(raw_debug.get("duration_judgment") or "").strip().lower()
    if duration not in AI_SUGGESTION_DEBUG_DURATION_ALLOWED:
        duration = "unclear"
    rhythm = str(raw_debug.get("rhythm_basis") or "").strip().lower()
    if rhythm not in AI_SUGGESTION_DEBUG_RHYTHM_ALLOWED:
        rhythm = "unclear"
    reason = str(raw_debug.get("decision_reason") or "").strip().lower()
    if reason not in AI_SUGGESTION_DEBUG_REASON_ALLOWED:
        reason = "other"
    notehead_fill = str(raw_debug.get("notehead_fill_read") or "").strip().lower()
    if notehead_fill not in AI_SUGGESTION_DEBUG_NOTEHEAD_FILL_ALLOWED:
        notehead_fill = "unclear"
    stem_or_beam = str(raw_debug.get("stem_or_beam_read") or "").strip().lower()
    if stem_or_beam not in AI_SUGGESTION_DEBUG_STEM_OR_BEAM_ALLOWED:
        stem_or_beam = "unclear"
    raw_dot_seen = raw_debug.get("dot_seen")
    if isinstance(raw_dot_seen, bool):
        dot_seen = "true" if raw_dot_seen else "false"
    else:
        dot_seen = str(raw_dot_seen or "").strip().lower()
    if dot_seen not in AI_SUGGESTION_DEBUG_DOT_SEEN_ALLOWED:
        dot_seen = "unclear"
    note_value = str(raw_debug.get("note_value_read") or "").strip().lower()
    if note_value not in AI_SUGGESTION_DEBUG_NOTE_VALUE_ALLOWED:
        note_value = "unclear"
    return {
        "active_meter_read": active_meter,
        "duration_judgment": duration,
        "rhythm_basis": rhythm,
        "decision_reason": reason,
        "notehead_fill_read": notehead_fill,
        "stem_or_beam_read": stem_or_beam,
        "dot_seen": dot_seen,
        "note_value_read": note_value,
        "counted_beat_units": _normalize_ai_debug_short_text(raw_debug.get("counted_beat_units")),
        "debug_note": _normalize_ai_debug_note(raw_debug.get("debug_note")),
    }


def _ai_suggest_normalization_warning(measure_row: dict | None, message: str) -> dict:
    warning = {
        "type": "normalization_adjusted",
        "message": message,
    }
    if isinstance(measure_row, dict):
        measure_id = str(measure_row.get("measure_id") or "").strip()
        if measure_id:
            warning["measure_id"] = measure_id
        system_id = str(measure_row.get("system_id") or "").strip()
        if system_id:
            warning["system_id"] = system_id
        if measure_row.get("system_index") is not None:
            warning["system_index"] = _safe_int(measure_row.get("system_index"), 0)
    return warning


def _derive_ai_measure_time_signatures_by_measure_id(
    ordered_measures: list[dict],
    valid_time_signature_updates: list[dict],
) -> dict[str, dict]:
    expected_measure_ids = {
        str((measure_row or {}).get("measure_id") or "").strip()
        for measure_row in ordered_measures
        if str((measure_row or {}).get("measure_id") or "").strip()
    }
    result: dict[str, dict] = {}
    for row in valid_time_signature_updates:
        measure_id = str((row or {}).get("measure_id") or "").strip()
        new_time_signature = _normalize_ai_time_signature_value((row or {}).get("new_time_signature"))
        if not measure_id or not new_time_signature:
            continue
        if measure_id not in expected_measure_ids:
            continue
        result[measure_id] = {
            "active_time_signature": new_time_signature,
            "time_signature_source": "explicit_here",
        }
    return result


def _normalize_ai_suggestions_result(
    raw_result: dict,
    editable_state: dict,
    run_id: int,
    source_state_version: str | None = None,
    remembered_time_signature_in: str | None = None,
) -> dict:
    if not isinstance(raw_result, dict):
        raise AiSuggestError(detail="malformed_response: root must be an object")

    ordered_measures = _sorted_measure_rows(editable_state.get("measures") or [])
    measure_rows_by_id = {
        str(row.get("measure_id") or "").strip(): row
        for row in ordered_measures
        if isinstance(row, dict) and str(row.get("measure_id") or "").strip()
    }
    expected_measure_ids = set(measure_rows_by_id.keys())
    raw_suggestions = raw_result.get("suggestions")
    if not isinstance(raw_suggestions, list):
        raise AiSuggestError(detail="malformed_response: suggestions must be an array")

    seen_measure_ids: set[str] = set()
    kept_by_measure_id: dict[str, dict] = {}
    decision_debug_by_measure_id: dict[str, dict] = {}
    measure_completeness_by_measure_id: dict[str, dict] = {}
    normal_measures_omitted = 0
    normalization_warnings: list[dict] = []
    fallback_measure_row = ordered_measures[0] if ordered_measures else {}
    system_id = str((ordered_measures[0] if ordered_measures else {}).get("system_id") or "").strip()

    for row in raw_suggestions:
        if not isinstance(row, dict):
            raise AiSuggestError(detail="malformed_response: suggestion entry must be an object")

        measure_id = str(row.get("measure_id") or "").strip()
        if not measure_id:
            raise AiSuggestError(detail="malformed_response: suggestion missing measure_id")
        if measure_id not in expected_measure_ids:
            raise AiSuggestError(detail=f"malformed_response: unknown measure_id {measure_id}")
        if measure_id in seen_measure_ids:
            raise AiSuggestError(detail=f"malformed_response: duplicate measure_id {measure_id}")
        seen_measure_ids.add(measure_id)

        label = str(row.get("label") or "").strip()
        if label not in AI_SUGGESTION_LABELS_ALLOWED:
            raise AiSuggestError(detail=f"malformed_response: invalid label for {measure_id}")
        confidence = str(row.get("confidence") or "").strip().lower()
        measure_row = measure_rows_by_id[measure_id]
        is_first_measure_of_score = _safe_int(measure_row.get("global_index"), -1) == 0
        if confidence not in AI_SUGGESTION_CONFIDENCE_ALLOWED:
            normalization_warnings.append(
                _ai_suggest_normalization_warning(
                    measure_row,
                    f"Invalid confidence for {measure_id}; defaulted to low.",
                )
            )
            confidence = "low"

        rest_count = row.get("rest_count")
        maybe_label = row.get("maybe_label")
        maybe_rest_count = row.get("maybe_rest_count")
        raw_unclear_reason = row.get("unclear_reason")
        decision_debug = None
        if is_first_measure_of_score:
            if row.get("decision_debug") is None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"first_measure_decision_debug_missing for {measure_id}.",
                    )
                )
            else:
                decision_debug = _normalize_ai_decision_debug(row.get("decision_debug"))
                if decision_debug is None:
                    normalization_warnings.append(
                        _ai_suggest_normalization_warning(
                            measure_row,
                            f"Dropped invalid decision_debug for {measure_id}.",
                        )
                    )
                else:
                    decision_debug_by_measure_id[measure_id] = decision_debug
        measure_completeness = _normalize_ai_measure_completeness_value(row.get("measure_completeness"))
        if measure_completeness is None:
            normalization_warnings.append(
                _ai_suggest_normalization_warning(
                    measure_row,
                    f"Missing or invalid measure_completeness for {measure_id}; defaulted to unclear.",
                )
            )
            measure_completeness = "unclear"

        if label == "pickup":
            measure_completeness = "incomplete"
        elif label == "multi_measure_rest":
            measure_completeness = "full"
        elif not is_first_measure_of_score and label == "normal" and measure_completeness == "incomplete":
            normalization_warnings.append(
                _ai_suggest_normalization_warning(
                    measure_row,
                    f"Downgraded later normal suggestion to uncertain for {measure_id} because measure_completeness was incomplete.",
                )
            )
            label = "uncertain"

        unclear_reason = None
        if raw_unclear_reason is not None:
            normalized_unclear_reason = _normalize_ai_unclear_reason_value(raw_unclear_reason)
            if normalized_unclear_reason is None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Dropped invalid unclear_reason for {measure_id}.",
                    )
                )
            else:
                unclear_reason = normalized_unclear_reason

        unclear_reason_allowed = label == "uncertain" or measure_completeness == "unclear"
        if unclear_reason is not None and not unclear_reason_allowed:
            normalization_warnings.append(
                _ai_suggest_normalization_warning(
                    measure_row,
                    f"Dropped unclear_reason for {measure_id} because the row was not uncertain or unclear.",
                )
            )
            unclear_reason = None

        measure_completeness_entry = {
            "measure_completeness": measure_completeness,
            "measure_completeness_source": "ai",
        }
        if unclear_reason is not None:
            measure_completeness_entry["unclear_reason"] = unclear_reason
        measure_completeness_by_measure_id[measure_id] = measure_completeness_entry

        if label == "normal":
            if rest_count is not None or maybe_label is not None or maybe_rest_count is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored extra fields on normal suggestion for {measure_id}.",
                    )
                )
            normal_measures_omitted += 1
            continue

        entry = {
            "label": label,
            "rest_count": None,
            "confidence": confidence,
            "system_id": str(measure_row.get("system_id") or "").strip(),
            "order_index_in_system": _safe_int(measure_row.get("measure_local_index"), 0),
            "is_first_measure_of_score": is_first_measure_of_score,
        }
        if unclear_reason is not None:
            entry["unclear_reason"] = unclear_reason
        if decision_debug is not None:
            entry["decision_debug"] = decision_debug

        if label == "pickup":
            if rest_count is not None or maybe_label is not None or maybe_rest_count is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored extra fields on pickup suggestion for {measure_id}.",
                    )
                )
        elif label == "multi_measure_rest":
            if not isinstance(rest_count, int) or int(rest_count) <= 1:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Downgraded multi_measure_rest to uncertain for {measure_id} because rest_count was missing or invalid.",
                    )
                )
                entry["label"] = "uncertain"
                kept_by_measure_id[measure_id] = entry
                continue
            if maybe_label is not None or maybe_rest_count is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored maybe fields on multi_measure_rest suggestion for {measure_id}.",
                    )
                )
            entry["rest_count"] = int(rest_count)
        else:
            if rest_count is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored rest_count on uncertain suggestion for {measure_id}.",
                    )
                )
            if maybe_label is not None:
                maybe_label = str(maybe_label or "").strip()
                if maybe_label not in AI_SUGGESTION_MAYBE_LABELS_ALLOWED:
                    normalization_warnings.append(
                        _ai_suggest_normalization_warning(
                            measure_row,
                            f"Dropped invalid maybe_label on uncertain suggestion for {measure_id}.",
                        )
                    )
                    maybe_label = None
                if maybe_label is not None:
                    entry["maybe_label"] = maybe_label
                if maybe_label == "multi_measure_rest":
                    if not isinstance(maybe_rest_count, int) or int(maybe_rest_count) <= 1:
                        normalization_warnings.append(
                            _ai_suggest_normalization_warning(
                                measure_row,
                                f"Downgraded uncertain multi_measure_rest guess to plain uncertain for {measure_id} because maybe_rest_count was missing or invalid.",
                            )
                        )
                        entry.pop("maybe_label", None)
                    else:
                        entry["maybe_rest_count"] = int(maybe_rest_count)
                elif maybe_rest_count is not None:
                    normalization_warnings.append(
                        _ai_suggest_normalization_warning(
                            measure_row,
                            f"Dropped invalid maybe fields on uncertain suggestion for {measure_id}.",
                        )
                    )
                    entry.pop("maybe_label", None)
            elif maybe_rest_count is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Dropped maybe_rest_count without maybe_label for {measure_id}.",
                    )
                )

        kept_by_measure_id[measure_id] = entry

    if seen_measure_ids != expected_measure_ids:
        missing = sorted(expected_measure_ids - seen_measure_ids)
        extra = sorted(seen_measure_ids - expected_measure_ids)
        detail_bits = []
        if missing:
            detail_bits.append(f"missing_measure_ids={','.join(missing[:10])}")
        if extra:
            detail_bits.append(f"unexpected_measure_ids={','.join(extra[:10])}")
        raise AiSuggestError(detail=f"malformed_response: incomplete suggestions {' '.join(detail_bits).strip()}".strip())

    provider = str(raw_result.get("provider") or _requested_ai_provider_name()).strip() or _requested_ai_provider_name()
    model = _requested_ai_model_name()
    systems_processed = len(_sorted_system_rows(editable_state.get("systems") or []))
    warnings = _normalize_ai_suggest_warnings(raw_result.get("warnings"))
    warnings.extend(normalization_warnings)
    raw_time_signature_updates = raw_result.get("time_signature_updates")
    valid_time_signature_updates: list[dict] = []
    if raw_time_signature_updates is not None and not isinstance(raw_time_signature_updates, list):
        warnings.append(
            _ai_suggest_normalization_warning(
                fallback_measure_row,
                "Ignored malformed time_signature_updates because it was not an array.",
            )
        )
    elif isinstance(raw_time_signature_updates, list):
        for raw_update in raw_time_signature_updates:
            if not isinstance(raw_update, dict):
                warnings.append(
                    _ai_suggest_normalization_warning(
                        fallback_measure_row,
                        "Ignored malformed time_signature update entry because it was not an object.",
                    )
                )
                continue
            measure_id = str(raw_update.get("measure_id") or "").strip()
            measure_row = measure_rows_by_id.get(measure_id) or fallback_measure_row
            if not measure_id or measure_id not in expected_measure_ids:
                warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        "Ignored time_signature update with missing or unknown measure_id.",
                    )
                )
                continue
            normalized_update = _normalize_ai_time_signature_update_row(raw_update, system_id=system_id)
            if not normalized_update:
                warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored invalid time_signature update for {measure_id}.",
                    )
                )
                continue
            normalized_update["measure_id"] = measure_id
            valid_time_signature_updates.append(normalized_update)

    raw_time_signature_out = raw_result.get("remembered_time_signature_out")
    if raw_time_signature_out is not None and _normalize_ai_time_signature_value(raw_time_signature_out) is None:
        warnings.append(
            _ai_suggest_normalization_warning(
                fallback_measure_row,
                "Ignored invalid remembered_time_signature_out because meter tracking is disabled.",
            )
        )

    remembered_time_signature_out = None
    last_time_signature_update = valid_time_signature_updates[-1] if valid_time_signature_updates else None

    time_signatures_by_measure_id = _derive_ai_measure_time_signatures_by_measure_id(
        ordered_measures,
        valid_time_signature_updates,
    )
    for measure_id, entry in kept_by_measure_id.items():
        time_signature_row = time_signatures_by_measure_id.get(measure_id)
        if isinstance(time_signature_row, dict):
            entry.update(time_signature_row)
        measure_completeness_row = measure_completeness_by_measure_id.get(measure_id)
        if isinstance(measure_completeness_row, dict):
            entry.update(measure_completeness_row)

    ai_suggestions = {
        "version": AI_SUGGESTIONS_VERSION,
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "provider": provider,
        "model": model,
        "source_run_id": int(run_id),
        "by_measure_id": kept_by_measure_id,
        "decision_debug_by_measure_id": decision_debug_by_measure_id,
        "time_signatures_by_measure_id": time_signatures_by_measure_id,
        "measure_completeness_by_measure_id": measure_completeness_by_measure_id,
        "warnings": warnings,
        "summary": {
            "systems_processed": systems_processed,
            "measures_seen": len(ordered_measures),
            "suggestions_kept": len(kept_by_measure_id),
            "normal_measures_omitted": normal_measures_omitted,
        },
    }
    source_state_version_txt = str(source_state_version or "").strip()
    if source_state_version_txt:
        ai_suggestions["source_state_version"] = source_state_version_txt
    ai_suggestions["remembered_time_signature_out"] = remembered_time_signature_out
    ai_suggestions["time_signature_updates"] = valid_time_signature_updates
    ai_suggestions["last_time_signature_update"] = last_time_signature_update
    return ai_suggestions


def _anthropic_api_key() -> str:
    return str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip()


def _aws_region_name() -> str:
    return str(os.environ.get("AWS_REGION", AWS_REGION) or "").strip()


def _bedrock_anthropic_version() -> str:
    return str(os.environ.get("BEDROCK_ANTHROPIC_VERSION", BEDROCK_ANTHROPIC_VERSION) or "").strip() or "bedrock-2023-05-31"


def _is_anthropic_overload_error(exc: AiSuggestError | Exception) -> bool:
    if not isinstance(exc, AiSuggestError):
        return False
    if int(getattr(exc, "provider_status", 0) or 0) == 529:
        return True
    detail = str(getattr(exc, "detail", "") or "").lower()
    return "overloaded_error" in detail


def _anthropic_messages_create_once(payload: dict) -> dict:
    api_key = _anthropic_api_key()
    if not api_key:
        raise AiSuggestError(provider_status=503, detail="provider_not_configured")
    req = urlrequest.Request(
        f"{ANTHROPIC_API_BASE}/v1/messages",
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urlrequest.urlopen(req, timeout=ANTHROPIC_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise AiSuggestError(provider_status=502, detail="malformed_provider_response")
            return data
    except urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        detail = body.strip() or _safe_error_text(exc)
        raise AiSuggestError(provider_status=int(getattr(exc, "code", 502) or 502), detail=detail)
    except urlerror.URLError as exc:
        raise AiSuggestError(provider_status=504, detail=_safe_error_text(exc))
    except TimeoutError as exc:
        raise AiSuggestError(provider_status=504, detail=_safe_error_text(exc))


def _anthropic_messages_create(payload: dict) -> dict:
    delays = tuple(float(delay) for delay in AI_SUGGEST_OVERLOAD_RETRY_DELAYS_SEC if float(delay) > 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            return _anthropic_messages_create_once(payload)
        except AiSuggestError as exc:
            if not _is_anthropic_overload_error(exc):
                raise
            if attempt > len(delays):
                detail = str(exc.detail or "").strip()
                delay_txt = ",".join(str(int(delay) if float(delay).is_integer() else delay) for delay in delays)
                suffix = f" overload_retry_attempts={attempt}"
                if delay_txt:
                    suffix += f" overload_retry_delays_sec={delay_txt}"
                exc.detail = f"{detail}{suffix}" if detail else suffix.strip()
                exc.retry_attempts = attempt
                raise
            delay_sec = delays[attempt - 1]
            logger.warning(
                "AI_SUGGEST_OVERLOAD_RETRY attempt=%s next_delay_sec=%s model=%s",
                attempt,
                delay_sec,
                str(payload.get("model") or _requested_ai_model_name()),
            )
            time.sleep(delay_sec)


def _bedrock_messages_create_once(payload: dict) -> dict:
    model_id = _configured_bedrock_model_id()
    region_name = _aws_region_name()
    if not model_id or not region_name:
        raise AiSuggestError(provider_status=503, detail="provider_not_configured")
    try:
        import boto3
    except Exception as exc:
        raise AiSuggestError(provider_status=503, detail=f"provider_not_configured: boto3 unavailable {_safe_error_text(exc)}") from exc

    body = dict(payload or {})
    body.pop("model", None)
    body["anthropic_version"] = _bedrock_anthropic_version()
    try:
        client = boto3.client("bedrock-runtime", region_name=region_name)
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        raw_body = response.get("body")
        raw = raw_body.read() if hasattr(raw_body, "read") else raw_body
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(str(raw or ""))
        if not isinstance(data, dict):
            raise AiSuggestError(provider_status=502, detail="malformed_provider_response")
        return data
    except AiSuggestError:
        raise
    except Exception as exc:
        response = getattr(exc, "response", None)
        metadata = response.get("ResponseMetadata") if isinstance(response, dict) else {}
        status = _safe_int((metadata or {}).get("HTTPStatusCode"), 502)
        raise AiSuggestError(provider_status=status or 502, detail=_safe_error_text(exc)) from exc


def _ai_messages_create(payload: dict) -> dict:
    provider = _requested_ai_provider_name()
    if provider == "bedrock":
        return _bedrock_messages_create_once(payload)
    if provider == "anthropic":
        return _anthropic_messages_create(payload)
    raise AiSuggestError(provider_status=503, detail="provider_not_configured")


def _strip_json_fences(text: str) -> str:
    txt = str(text or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    return txt.strip()


def _extract_json_object_text(text: str) -> str:
    txt = _strip_json_fences(text)
    if txt.startswith("{") and txt.endswith("}"):
        return txt
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        return txt[start : end + 1]
    raise AiSuggestError(detail="malformed_response: missing json object")


def _parse_anthropic_suggestions_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise AiSuggestError(detail="malformed_provider_response")
    content = message.get("content")
    if not isinstance(content, list):
        raise AiSuggestError(detail="malformed_provider_response: content missing")
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "").strip() == "text":
            text_parts.append(str(block.get("text") or ""))
    if not text_parts:
        raise AiSuggestError(detail="malformed_response: no text content")
    try:
        parsed = json.loads(_extract_json_object_text("\n".join(text_parts)))
    except json.JSONDecodeError as exc:
        raise AiSuggestError(detail=f"malformed_response: invalid json {_safe_error_text(exc)}")
    if not isinstance(parsed, dict):
        raise AiSuggestError(detail="malformed_response: root must be object")
    parsed.setdefault("provider", _requested_ai_provider_name())
    parsed.pop("model", None)
    return parsed


def _ai_suggest_debug_enabled() -> bool:
    return bool(AI_SUGGEST_SAVE_DEBUG_CROPS)


def _ai_debug_crops_prefix(artifacts: dict) -> str:
    mapping_uri = str((artifacts or {}).get("mapping_summary") or "").strip()
    if not mapping_uri:
        raise ValueError("mapping_summary artifact missing")
    bucket_name, blob_name = _parse_gs_uri(mapping_uri)
    base_dir = blob_name.rsplit("/", 1)[0].rstrip("/")
    return f"gs://{bucket_name}/{base_dir}/ai_debug_crops"


def _ai_debug_crop_manifest_uri(artifacts: dict) -> str:
    return f"{_ai_debug_crops_prefix(artifacts)}/manifest.json"


def _ai_debug_batch_trace_uri(artifacts: dict) -> str:
    return f"{_ai_debug_crops_prefix(artifacts)}/ai_batch_trace.json"


def _ai_debug_crop_measure_uri(artifacts: dict, system_id: str, measure_id: str) -> str:
    safe_system = _normalize_artifact_key(system_id) or "system"
    safe_measure = _normalize_artifact_key(measure_id) or "measure"
    return f"{_ai_debug_crops_prefix(artifacts)}/{safe_system}/{safe_measure}.png"


def _resolve_ai_crop_pdf_source(artifacts: dict, tmpdir: Path) -> tuple[Path, str]:
    corrected_pdf_uri = str((artifacts or {}).get("audiveris_out_corrected_pdf") or "").strip()
    baseline_pdf_uri = str((artifacts or {}).get("audiveris_out_pdf") or "").strip()
    if not baseline_pdf_uri:
        raise AiSuggestError(provider_status=500, detail="baseline_pdf_missing")

    if corrected_pdf_uri:
        corrected_pdf = tmpdir / "audiveris_out_corrected.pdf"
        try:
            _download_gcs_to_file(corrected_pdf_uri, corrected_pdf)
            return corrected_pdf, "corrected"
        except Exception as exc:
            logger.warning("AI_CROP_PDF_FALLBACK detail=%s", _safe_error_text(exc))

    baseline_pdf = tmpdir / "audiveris_out.pdf"
    _download_gcs_to_file(baseline_pdf_uri, baseline_pdf)
    return baseline_pdf, "baseline"


def _current_ai_crop_pdf_source_label(artifacts: dict) -> str:
    corrected_pdf_uri = str((artifacts or {}).get("audiveris_out_corrected_pdf") or "").strip()
    if corrected_pdf_uri and _gcs_uri_exists(corrected_pdf_uri):
        return "corrected"
    return "baseline"


def _system_anchor_bounds(system_row: dict | None) -> tuple[float | None, float | None]:
    if not isinstance(system_row, dict):
        return (None, None)
    anchor = system_row.get("anchor")
    if not isinstance(anchor, dict):
        return (None, None)
    try:
        top = float(anchor.get("y_top"))
        bottom = float(anchor.get("y_bottom"))
    except Exception:
        return (None, None)
    if bottom <= top:
        return (None, None)
    return (top, bottom)


def _system_gap_clamp_bounds(
    page_rect,
    system_row: dict | None,
    prev_system_row: dict | None = None,
    next_system_row: dict | None = None,
) -> tuple[float, float]:
    page_top = 0.0
    page_bottom = float(page_rect.height)
    system_top_raw, system_bottom_raw = _system_anchor_bounds(system_row)
    if system_top_raw is None or system_bottom_raw is None:
        return (page_top, page_bottom)

    system_top = max(page_top, min(system_top_raw, page_bottom))
    system_bottom = min(page_bottom, max(system_bottom_raw, system_top))
    clamp_top = system_top
    clamp_bottom = system_bottom

    prev_top_raw, prev_bottom_raw = _system_anchor_bounds(prev_system_row)
    if prev_top_raw is not None and prev_bottom_raw is not None and prev_bottom_raw < system_top:
        gap_above = max(0.0, system_top - prev_bottom_raw)
        clamp_top = max(page_top, system_top - (gap_above * AI_MEASURE_CROP_SYSTEM_GAP_CLAMP_RATIO))
    else:
        clamp_top = page_top

    next_top_raw, next_bottom_raw = _system_anchor_bounds(next_system_row)
    if next_top_raw is not None and next_bottom_raw is not None and next_top_raw > system_bottom:
        gap_below = max(0.0, next_top_raw - system_bottom)
        clamp_bottom = min(page_bottom, system_bottom + (gap_below * AI_MEASURE_CROP_SYSTEM_GAP_CLAMP_RATIO))
    else:
        clamp_bottom = page_bottom

    if clamp_bottom <= clamp_top:
        return (page_top, page_bottom)
    return (clamp_top, clamp_bottom)


def _measure_crop_spec(
    page_rect,
    measure_row: dict,
    next_measure_row: dict | None,
    system_row: dict | None,
    prev_system_row: dict | None = None,
    next_system_row: dict | None = None,
) -> dict:
    x_left = float(measure_row.get("x_left") or 0.0)
    x_right_raw = measure_row.get("x_right")
    if x_right_raw is None and isinstance(next_measure_row, dict):
        x_right_raw = next_measure_row.get("x_left")
    if x_right_raw is None:
        x_right_raw = page_rect.width
    x_right = float(x_right_raw or 0.0)

    y_top_raw = measure_row.get("y_top")
    if y_top_raw is None and isinstance(system_row, dict):
        y_top_raw = ((system_row.get("anchor") or {}) if isinstance(system_row.get("anchor"), dict) else {}).get("y_top")
    y_bottom_raw = measure_row.get("y_bottom")
    if y_bottom_raw is None and isinstance(system_row, dict):
        y_bottom_raw = ((system_row.get("anchor") or {}) if isinstance(system_row.get("anchor"), dict) else {}).get("y_bottom")
    y_top = float(y_top_raw or 0.0)
    y_bottom = float(y_bottom_raw or 0.0)

    if x_right <= x_left:
        x_right = min(float(page_rect.width), x_left + 40.0)
    if y_bottom <= y_top:
        y_bottom = min(float(page_rect.height), y_top + 40.0)

    width = max(1.0, x_right - x_left)
    height = max(1.0, y_bottom - y_top)
    x_pad = 0.0
    pad_top = max(AI_MEASURE_CROP_MIN_TOP_PAD, height * AI_MEASURE_CROP_TOP_PAD_RATIO)
    pad_bottom = max(AI_MEASURE_CROP_MIN_BOTTOM_PAD, height * AI_MEASURE_CROP_BOTTOM_PAD_RATIO)

    page_top = 0.0
    page_bottom = float(page_rect.height)
    system_top, system_bottom = _system_gap_clamp_bounds(page_rect, system_row, prev_system_row, next_system_row)

    clip = fitz.Rect(
        max(0.0, x_left - x_pad),
        max(system_top, y_top - pad_top),
        min(float(page_rect.width), x_right + x_pad),
        min(system_bottom, y_bottom + pad_bottom),
    )
    if clip.x1 <= clip.x0 or clip.y1 <= clip.y0:
        raise AiSuggestError(provider_status=500, detail="invalid_measure_crop")
    return {
        "clip": clip,
        "measure_bounds": {
            "left": float(x_left),
            "right": float(x_right),
            "top": float(y_top),
            "bottom": float(y_bottom),
            "width": float(width),
            "height": float(height),
        },
        "padding": {
            "left": float(x_pad),
            "right": float(x_pad),
            "top": float(pad_top),
            "bottom": float(pad_bottom),
        },
        "system_bounds": {
            "top": float(system_top),
            "bottom": float(system_bottom),
        },
    }


def _measure_crop_rect(
    page_rect,
    measure_row: dict,
    next_measure_row: dict | None,
    system_row: dict | None,
    prev_system_row: dict | None = None,
    next_system_row: dict | None = None,
) -> fitz.Rect:
    return _measure_crop_spec(page_rect, measure_row, next_measure_row, system_row, prev_system_row, next_system_row)["clip"]


def _render_measure_crop_png(page, clip: fitz.Rect) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(AI_MEASURE_CROP_SCALE, AI_MEASURE_CROP_SCALE), clip=clip, alpha=False)
    return bytes(pix.tobytes("png"))


def _build_ai_debug_crops_manifest(
    job_id: str,
    run_id: int,
    artifacts: dict,
    crop_rows: list[dict],
    pdf_source: str | None = None,
) -> dict:
    manifest_uri = _ai_debug_crop_manifest_uri(artifacts)
    payload = {
        "version": "ai_debug_crops_v1",
        "enabled": True,
        "job_id": str(job_id),
        "run_id": int(run_id),
        "pdf_source": str(pdf_source or "baseline"),
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "count": len(crop_rows),
        "crops": crop_rows,
    }
    _upload_json_to_gcs(payload, manifest_uri)
    return {
        "enabled": True,
        "manifest_uri": manifest_uri,
        "manifest_http": _signed_http_url_for_gs(manifest_uri),
        "pdf_source": str(pdf_source or "baseline"),
        "count": len(crop_rows),
    }


def _ai_batch_trace_before_snapshot(measures: list[dict] | None) -> list[dict | None]:
    snapshot: list[dict | None] = []
    for row in measures or []:
        if not isinstance(row, dict):
            snapshot.append(None)
            continue
        snapshot.append(
            {
                "system_id_before_reassign": str(row.get("system_id") or "").strip(),
                "system_index_before_reassign": _safe_int(row.get("system_index"), 0),
            }
        )
    return snapshot


def _debug_display_system_number(system_index: object) -> int:
    return max(1, _safe_int(system_index, 0) + 1)


def _debug_display_measure_number(measure_local_index: object) -> int:
    return max(1, _safe_int(measure_local_index, 0) + 1)


def _build_ai_batch_trace_payload(
    job_id: str,
    run_id: int,
    systems: list[dict] | None,
    measures: list[dict] | None,
    system_batches: list[tuple[dict, list[dict]]] | None,
    before_snapshot: list[dict | None] | None = None,
    processed_system_ids: list[str] | None = None,
    pdf_source: str | None = None,
) -> dict:
    ordered_systems = _sorted_system_rows(systems or [])
    ordered_measures = list(measures or [])
    before_rows = list(before_snapshot or [])
    valid_system_ids = {
        str(row.get("system_id") or "").strip()
        for row in ordered_systems
        if isinstance(row, dict) and str(row.get("system_id") or "").strip()
    }

    batched_measure_to_system: dict[str, str] = {}
    system_summaries: list[dict] = []
    processed_lookup = {
        str(system_id or "").strip()
        for system_id in (processed_system_ids or [])
        if str(system_id or "").strip()
    }

    for system_row, system_measures in system_batches or []:
        if not isinstance(system_row, dict):
            continue
        system_id = str(system_row.get("system_id") or "").strip()
        if not system_id:
            continue
        measure_ids_batched: list[str] = []
        for row in system_measures or []:
            if not isinstance(row, dict):
                continue
            measure_id = str(row.get("measure_id") or "").strip()
            if not measure_id:
                continue
            batched_measure_to_system[measure_id] = system_id
            measure_ids_batched.append(measure_id)
        system_summaries.append(
            {
                "system_id": system_id,
                "page": _safe_int(system_row.get("page"), 0),
                "display_system_number": _debug_display_system_number(system_row.get("system_index")),
                "display_location": f"Page {_safe_int(system_row.get('page'), 0)}, Staff {_debug_display_system_number(system_row.get('system_index'))}",
                "measure_ids_batched": measure_ids_batched,
                "count": len(measure_ids_batched),
                "processed": system_id in processed_lookup,
            }
        )

    trace_rows: list[dict] = []
    batched_count = 0
    skipped_count = 0

    for index, row in enumerate(ordered_measures):
        if not isinstance(row, dict):
            continue
        before_row = before_rows[index] if index < len(before_rows) else None
        before_system_id = str((before_row or {}).get("system_id_before_reassign") or str(row.get("system_id") or "")).strip()
        before_system_index = _safe_int((before_row or {}).get("system_index_before_reassign"), _safe_int(row.get("system_index"), 0))
        measure_id = str(row.get("measure_id") or "").strip()
        after_system_id = str(row.get("system_id") or "").strip()
        after_system_index = _safe_int(row.get("system_index"), 0)
        display_system_number = _debug_display_system_number(after_system_index)
        display_measure_number = _debug_display_measure_number(row.get("measure_local_index"))
        batch_system_id = str(batched_measure_to_system.get(measure_id) or "").strip() or None
        changed = before_system_id != after_system_id or before_system_index != after_system_index

        if batch_system_id:
            status = "reassigned_and_batched" if changed else "batched"
            batched_count += 1
        elif not after_system_id:
            status = "skipped_missing_system_id"
            skipped_count += 1
        elif after_system_id not in valid_system_ids:
            status = "reassigned_but_unbatched" if changed else "skipped_no_matching_system"
            skipped_count += 1
        else:
            status = "reassigned_but_unbatched" if changed else "skipped_no_matching_system"
            skipped_count += 1

        trace_rows.append(
            {
                "measure_id": measure_id,
                "page": _safe_int(row.get("page"), 0),
                "display_system_number": display_system_number,
                "display_measure_number": display_measure_number,
                "display_location": f"Page {_safe_int(row.get('page'), 0)}, Staff {display_system_number}, Measure {display_measure_number}",
                "system_id_before_reassign": before_system_id or None,
                "system_index_before_reassign": before_system_index,
                "system_id_after_reassign": after_system_id or None,
                "system_index_after_reassign": after_system_index,
                "measure_local_index": _safe_int(row.get("measure_local_index"), 0),
                "x_left": float(row.get("x_left") or 0.0),
                "y_top": float(row.get("y_top") or 0.0),
                "y_bottom": float(row.get("y_bottom") or 0.0),
                "batch_system_id": batch_system_id,
                "status": status,
                "processed": bool(batch_system_id and batch_system_id in processed_lookup),
            }
        )

    return {
        "version": "ai_batch_trace_v1",
        "enabled": True,
        "job_id": str(job_id),
        "run_id": int(run_id),
        "pdf_source": str(pdf_source or "baseline"),
        "generated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "updated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "measure_count": len(trace_rows),
        "batched_count": batched_count,
        "skipped_count": skipped_count,
        "processed_system_ids": sorted(processed_lookup),
        "systems": system_summaries,
        "measures": trace_rows,
    }


def _mark_ai_batch_trace_processed(
    payload: dict,
    system_row: dict | None,
    system_measures: list[dict] | None,
) -> dict:
    if not isinstance(payload, dict):
        return {}
    updated = json.loads(json.dumps(payload))
    system_id = str((system_row or {}).get("system_id") or "").strip()
    if not system_id:
        return updated
    processed_ids = [
        str(item or "").strip()
        for item in updated.get("processed_system_ids") or []
        if str(item or "").strip()
    ]
    if system_id not in processed_ids:
        processed_ids.append(system_id)
    updated["processed_system_ids"] = processed_ids

    for summary in updated.get("systems") or []:
        if isinstance(summary, dict) and str(summary.get("system_id") or "").strip() == system_id:
            summary["processed"] = True

    target_measure_ids = {
        str(row.get("measure_id") or "").strip()
        for row in system_measures or []
        if isinstance(row, dict) and str(row.get("measure_id") or "").strip()
    }
    for row in updated.get("measures") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("measure_id") or "").strip() in target_measure_ids:
            row["processed"] = True

    updated["updated_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
    return updated


def _write_ai_debug_batch_trace(payload: dict, artifacts: dict) -> dict:
    trace_uri = _ai_debug_batch_trace_uri(artifacts)
    _upload_json_to_gcs(payload, trace_uri)
    return {
        "enabled": True,
        "trace_uri": trace_uri,
        "trace_http": _signed_http_url_for_gs(trace_uri),
        "pdf_source": str(payload.get("pdf_source") or "baseline"),
        "measure_count": max(0, _safe_int(payload.get("measure_count"), 0)),
        "batched_count": max(0, _safe_int(payload.get("batched_count"), 0)),
        "skipped_count": max(0, _safe_int(payload.get("skipped_count"), 0)),
    }


def _load_ai_debug_batch_trace(artifacts: dict) -> dict | None:
    trace_uri = _ai_debug_batch_trace_uri(artifacts)
    if not _gcs_uri_exists(trace_uri):
        return None
    payload = _download_gcs_json(trace_uri)
    return payload if isinstance(payload, dict) else None


def _build_old_style_multi_rest_reference_content() -> tuple[list[dict], int]:
    content: list[dict] = []
    example_rows: list[dict] = []
    for row in AI_OLD_STYLE_REFERENCE_EXAMPLES:
        filename = str((row or {}).get("filename") or "").strip()
        if not filename:
            continue
        image_path = AI_REFERENCE_EXAMPLES_DIR / filename
        try:
            image_bytes = image_path.read_bytes()
        except FileNotFoundError:
            logger.warning("AI_REFERENCE_EXAMPLE_MISSING path=%s", image_path)
            continue
        except Exception as exc:
            logger.warning("AI_REFERENCE_EXAMPLE_LOAD_FAILED path=%s detail=%s", image_path, exc)
            continue
        example_rows.append(
            {
                "caption": str((row or {}).get("caption") or "").strip(),
                "image_bytes": image_bytes,
            }
        )
    if not example_rows:
        return content, 0
    content.append(
        {
            "type": "text",
            "text": (
                "Reference examples for old-style multi-measure rest recognition. "
                "The next example images are references only; the real measure crops follow after them."
            ),
        }
    )
    for row in example_rows:
        caption = str((row or {}).get("caption") or "").strip()
        if caption:
            content.append({"type": "text", "text": caption})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(row["image_bytes"]).decode("ascii"),
                },
            }
        )
    return content, len(example_rows)


def _ai_prompt_base_rules() -> list[str]:
    return [
        "Each image contains exactly one already-detected measure.",
        "Do not infer additional measures from rhythmic groupings, repeat dots, barline decorations, edge marks, spacing, or decorations.",
        "Process the provided measures left to right in order.",
        "A numeric time signature is two vertically stacked meter numbers immediately after the clef/key signature, such as 2 over 4.",
        "Ignore fingering/count numbers near notes, above the staff, or below the staff. They are not time signatures.",
        "Do not remember, inherit, carry, or track time signatures across measures.",
        "Only read a time signature if it is visible in the current crop.",
        "Only use meter for first-measure pickup judgment.",
        "For multi-measure rests, ignore meter completely.",
        "Only label pickup when is_first_measure_of_score is true.",
        "If is_first_measure_of_score is false, do not label pickup.",
        "Set measure_completeness only as needed: pickup = incomplete, multi_measure_rest = full, clear normal = full, unclear = unclear.",
        "Prefer useful suggestions over silence. If a crop strongly looks like pickup or multi_measure_rest, suggest it even with medium or low confidence.",
        "If a measure is uncertain or its measure_completeness is unclear, you may include unclear_reason using one of these exact codes only: time_signature_not_clear, too_dense_to_count, crop_cut_off, split_may_be_wrong, ornament_or_tie_confusion, not_enough_visual_evidence.",
        "Do not write sentences for unclear_reason. Use only one short code or omit the field.",
        "For later non-rest measures, do not judge beat completeness unless the visible notation makes uncertainty necessary.",
    ]


def _ai_prompt_duration_basics() -> list[str]:
    return [
        "Read meter as top/bottom: top is how many beat-units fill a full measure; bottom is which note value is one beat-unit.",
        "Count written note/rest durations only. Never count visual width, spacing, number of noteheads, or number of staves as beats.",
        "Chords or stacked notes count as exactly one rhythmic event using the written note value. Do not count each notehead separately.",
        "For later non-first measures, do not label pickup.",
        "Use uncertain with maybe_label pickup only when first-measure pickup is possible but the visible meter or rhythmic duration cannot be read reliably because the notation is unclear, dense, or cut off.",
        "A whole note, two half notes, or other sparse-looking content in the first measure is usually a slow full measure, not a pickup. Do not label pickup just because the first measure looks sparse or simple.",
        "Use uncertain instead of pickup only when there is not enough visual evidence to determine the first measure's visible meter or rhythmic duration.",
    ]


def _ai_prompt_single_pickup_rules() -> list[str]:
    return [
        "Single-staff pickup rules:",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use the visible meter in this crop only.",
        "Count this one staff's written duration only.",
        "If the first measure's written duration is less than the visible meter, label pickup and set measure_completeness to incomplete.",
        "If it reaches the visible meter, label normal and set measure_completeness to full.",
    ]


def _ai_prompt_grand_pickup_rules() -> list[str]:
    return [
        "Grand-staff pickup rules:",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use the visible meter in this crop only; if the same meter appears on both staves, treat it as one shared meter.",
        "Treble and bass happen at the same time; never add them as separate beats.",
        "Use one clear staff's written rhythm/rests as the timing guide for the whole measure.",
        "If the guide staff's duration is less than the visible meter, the whole first measure is pickup/incomplete.",
        "If the guide staff reaches the visible meter, label normal and set measure_completeness to full.",
    ]


def _ai_prompt_score_pickup_rules() -> list[str]:
    return [
        "Full-score pickup rules:",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use the visible meter in this crop only.",
        "Instruments happen at the same time; never add instruments as separate beats.",
        "Use the clearest staff/instrument's written rhythm/rests as the timing guide for the whole measure.",
        "If the guide staff's duration is less than the visible meter, the whole first measure is pickup/incomplete.",
        "If the guide staff reaches the visible meter, label normal and set measure_completeness to full.",
    ]


def _ai_prompt_single_rules() -> list[str]:
    return [
        "This is single-staff music. Judge rhythm using only this one staff.",
        "Single-staff pickup rules:",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use the visible meter in this crop only.",
        "Read meter as top/bottom: top = how many beat-units fill a full measure; bottom = which note value is one beat-unit.",
        "Bottom number examples: 4 means quarter-note beats, 8 means eighth-note beats, 2 means half-note beats.",
        "Common time looks like a large C after the clef/key signature and means 4/4.",
        "Cut time looks like a large C with a vertical slash through it and means 2/2.",
        "Count written note/rest durations only. Never count visual width, spacing, or number of noteheads as beats.",
        "Basic note values: filled notehead with stem = quarter note; open notehead with stem = half note; open notehead without stem = whole note; filled notehead with flag or beam = eighth note.",
        "For first-measure pickup debug, identify notehead fill before deciding note value.",
        "A filled black notehead cannot be a half note; half note requires an open/white notehead.",
        "If unsure, use notehead fill first: black = quarter/eighth family, open = half/whole family.",
        "A dot immediately to the right of a note/rest adds half its value.",
        "A triplet is marked by a small 3 above or below a group; the 3 may have a bracket or appear over beamed notes.",
        "Three triplet notes fit into the time normally taken by two of the same note value. Example: three triplet eighth notes equal one quarter-note beat.",
        "Rests count toward the meter exactly like notes.",
        "Chords/stacked notes count as exactly one rhythmic event using the written note value. Do not count each notehead separately.",
        "Count this one staff's written duration only.",
        "If the first measure's written duration is less than the visible meter, label pickup and set measure_completeness to incomplete.",
        "For the first measure, arithmetic wins over context; do not call a short first measure full because it looks musically complete.",
        "Only label normal/full if the written notes/rests clearly add up to the full visible meter.",
        "If first-measure pickup is possible but the visible meter or rhythm is unclear, use uncertain with maybe_label pickup.",
        "Single-staff multi-measure rest rules:",
        "Check multi-measure rests before pickup or meter. Do not inspect meter for multi-measure rests.",
        "Use the old-style reference images as examples only; the real measure crops come after them. Old-style rests can look different from modern H-bars.",
        "A multi-measure rest symbol can be a modern H-bar or thick horizontal block across the staff.",
        "A multi-measure rest symbol can also be old-style vertical bars, horizontal bars, or small bar pieces inside the staff.",
        "Normal quarter/eighth/half/whole rests are not multi-measure rests by themselves.",
        "Number first, symbol second: a readable big number 2 or higher above/near a rest-like old/modern symbol means multi_measure_rest.",
        "Use the printed big number as rest_count. Do not require the bar pattern to visually match the count.",
        "Do not return uncertain just because the symbol is messy, unfamiliar, or old-looking.",
        "A visible count of 1 is normal, not multi_measure_rest.",
        "If there is no readable count, do not return a confident multi_measure_rest.",
        "If it is clearly only a normal one-measure rest, label normal.",
        "Use uncertain only when the count is unreadable or the number may be a rehearsal/measure number instead of a rest count.",
    ]


def _ai_prompt_grand_rules() -> list[str]:
    return [
        "Grand-staff main rule:",
        "For all AI Suggest decisions in grand-staff/piano music, use only the top staff/treble staff.",
        "Ignore the bottom staff completely for time signature, pickup, and multi-measure rest.",
        "Do not inspect, compare, add, or use the bottom staff as fallback.",
        "If the top staff is hard to read, unreadable, empty, or cut off, do not switch to the bottom staff; use unknown or uncertain instead.",
        "Grand-staff pickup rules:",
        "This is grand-staff/piano music. For pickup counting, use only the top staff/treble staff.",
        "Ignore the bottom staff completely for pickup duration. Do not inspect it, use it as fallback, compare it, or add it.",
        "Do not use bottom-staff notes or rests to decide whether the measure is full.",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use only the top staff's visible meter in this crop.",
        "If the same meter appears on both staves, ignore the bottom duplicate.",
        "If the top staff meter is unreadable, use unknown/uncertain.",
        "Read meter as top/bottom: top = how many beat-units fill a full measure; bottom = which note value is one beat-unit.",
        "Bottom number examples: 4 means quarter-note beats, 8 means eighth-note beats, 2 means half-note beats.",
        "Common time looks like a large C after the clef/key signature and means 4/4.",
        "Cut time looks like a large C with a vertical slash through it and means 2/2.",
        "Count the top staff's written note/rest durations only. Never count visual width, spacing, number of noteheads, or number of staves as beats.",
        "Basic note values: filled notehead with stem = quarter note; open notehead with stem = half note; open notehead without stem = whole note; filled notehead with flag or beam = eighth note.",
        "For first-measure pickup debug, identify the top-staff notehead fill before deciding note value.",
        "A filled black notehead cannot be a half note; half note requires an open/white notehead.",
        "If unsure, use notehead fill first: black = quarter/eighth family, open = half/whole family.",
        "A dot immediately to the right of a note/rest adds half its value.",
        "A triplet is marked by a small 3 above or below a group; the 3 may have a bracket or appear over beamed notes.",
        "Three triplet notes fit into the time normally taken by two of the same note value. Example: three triplet eighth notes equal one quarter-note beat.",
        "A chord/stack on the top staff is exactly one rhythmic event, no matter how many noteheads it has. Use the written note value.",
        "If the top staff's written duration is less than the visible meter, the whole first measure is pickup/incomplete.",
        "For the first measure, arithmetic wins over context; do not call a short first measure full because it looks musically complete.",
        "Only label normal/full if the top staff clearly fills the visible meter.",
        "If the top staff meter/rhythm is unreadable or cut off, use uncertain with maybe_label pickup.",
        "Grand-staff multi-measure rest rules:",
        "For grand-staff multi-measure rest, use only the top staff/treble staff.",
        "Check multi-measure rests before pickup or meter. Do not inspect meter for multi-measure rests.",
        "Use the old-style reference images as examples only; the real measure crops come after them. Old-style rests can look different from modern H-bars.",
        "A multi-measure rest symbol can be a modern H-bar or thick horizontal block across the staff.",
        "A multi-measure rest symbol can also be old-style vertical bars, horizontal bars, or small bar pieces inside the staff.",
        "Normal quarter/eighth/half/whole rests are not multi-measure rests by themselves.",
        "Number first, symbol second: a readable big number 2 or higher above/near a rest-like old/modern symbol means multi_measure_rest.",
        "Use the printed big number as rest_count. Do not require the bar pattern to visually match the count.",
        "Do not return uncertain just because the symbol is messy, unfamiliar, or old-looking.",
        "A visible count of 1 is normal, not multi_measure_rest.",
        "If there is no readable count, do not return a confident multi_measure_rest.",
        "If it is clearly only a normal one-measure rest, label normal.",
        "Use uncertain only when the count is unreadable or the number may be a rehearsal/measure number instead of a rest count.",
    ]


def _ai_prompt_score_rules() -> list[str]:
    return [
        "Full-score pickup rules:",
        "For pickup detection in full scores, use only the top visible staff.",
        "Ignore all lower staves completely for pickup duration.",
        "Do not inspect, compare, add, or use lower staves as fallback.",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use only the top staff's visible meter in this crop.",
        "If the top staff meter is unreadable but common/cut/numeric meter is partially visible, make the best reasonable meter read and continue.",
        "Use uncertain only if no meter can be reasonably read at all.",
        "Read meter as top/bottom: top = how many beat-units fill a full measure; bottom = which note value is one beat-unit.",
        "Bottom number examples: 4 means quarter-note beats, 8 means eighth-note beats, 2 means half-note beats.",
        "Common time looks like a large C after the clef/key signature and means 4/4.",
        "Cut time looks like a large C with a vertical slash through it and means 2/2.",
        "Count the top staff's written note/rest durations only. Never count visual width, spacing, number of noteheads, number of staves, or number of instruments as beats.",
        "Basic note values: filled notehead with stem = quarter note; open notehead with stem = half note; open notehead without stem = whole note; filled notehead with flag or beam = eighth note.",
        "For first-measure pickup debug, identify the top-staff notehead fill before deciding note value.",
        "A filled black notehead cannot be a half note; half note requires an open/white notehead.",
        "If unsure, use notehead fill first: black = quarter/eighth family, open = half/whole family.",
        "A dot immediately to the right of a note/rest adds half its value.",
        "A triplet is marked by a small 3 above or below a group; the 3 may have a bracket or appear over beamed notes.",
        "Three triplet notes fit into the time normally taken by two of the same note value. Example: three triplet eighth notes equal one quarter-note beat.",
        "A chord/stack on the top staff is exactly one rhythmic event, no matter how many noteheads it has. Use the written note value.",
        "If the top staff's written duration is less than the visible meter, the whole first measure is pickup/incomplete.",
        "For the first measure, arithmetic wins over context; do not call a short first measure full because it looks intentional, musical, complete, or like an opening gesture.",
        "For the first measure, if the top-staff written duration looks clearly shorter than a normal full measure, prefer pickup over normal.",
        "For pickup, overusing uncertain is worse than a reasonable wrong pickup suggestion.",
        "Do not use uncertain just because the notation is old, small, light, or slightly messy.",
        "Only label normal/full if the top staff clearly fills the visible meter.",
        "If you cannot confidently choose pickup, but the top staff may be short, use uncertain with maybe_label pickup.",
        "If the top staff meter/rhythm is completely unreadable, empty, or cut off, use uncertain with maybe_label pickup.",
        "Full-score multi-measure rest rules:",
        "For full score V1, NEVER return multi_measure_rest.",
        "Do not look for multi-measure rests in full-score crops.",
        "Do not use any printed rest count, H-bar, old-style rest symbol, or instrument rest to skip score measures.",
        "A rest count on one staff only means that instrument may be resting; it does not mean the score should skip measures.",
        "Even if multiple staves show rest symbols, full-score V1 must still count visible score measures normally.",
        "If a full-score rest situation is confusing, label normal or uncertain, but never multi_measure_rest.",
        "rest_count must always be null for full-score prompts.",
    ]


def _ai_prompt_output_rules() -> list[str]:
    return [
        "If label is uncertain and you have a tentative guess, maybe_label may be pickup or multi_measure_rest, and maybe_rest_count is only allowed for maybe_label multi_measure_rest.",
        "If maybe_label is multi_measure_rest, always include maybe_rest_count if the count number is at all readable. Only omit maybe_rest_count if the number is completely unreadable.",
        "A wrong confident answer is worse than uncertain.",
        "Do not skip any provided measure_id.",
        "Do not output labels outside the allowed set.",
        "If label is multi_measure_rest, rest_count must be an integer >= 2. If label is not multi_measure_rest, rest_count must be null.",
        "For the first measure of the score only, decision_debug is required. Do not omit it. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_single_output_rules() -> list[str]:
    return [
        "Allowed labels: normal, pickup, multi_measure_rest, uncertain.",
        "Do not skip any provided measure_id.",
        "Do not output labels outside the allowed set.",
        "Overusing uncertain is worse than a reasonable confident suggestion.",
        "Use uncertain only when the visual evidence is truly unreadable or conflicting.",
        "For multi-rest: if readable count 2 or higher plus rest-like multi-rest symbol, output multi_measure_rest.",
        "For multi-rest: use uncertain only when the number is hard to read or might not be a rest count.",
        "If label is multi_measure_rest, rest_count must be an integer >= 2. If label is not multi_measure_rest, rest_count must be null.",
        "If label is uncertain with maybe_label = multi_measure_rest and the count is partly readable, include maybe_rest_count.",
        "For the first measure of the score only, decision_debug is required. Do not omit it. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_grand_output_rules() -> list[str]:
    return [
        "Allowed labels: normal, pickup, multi_measure_rest, uncertain.",
        "Do not skip any provided measure_id.",
        "Do not output labels outside the allowed set.",
        "Overusing uncertain is worse than a reasonable confident suggestion.",
        "Use uncertain only when the top-staff visual evidence is truly unreadable or conflicting.",
        "For multi-rest: if the top staff has readable count 2 or higher plus rest-like multi-rest symbol, output multi_measure_rest.",
        "For multi-rest: use uncertain only when the top-staff number is hard to read or might not be a rest count.",
        "If label is multi_measure_rest, rest_count must be an integer >= 2. If label is not multi_measure_rest, rest_count must be null.",
        "If label is uncertain with maybe_label = multi_measure_rest and the count is partly readable, include maybe_rest_count.",
        "For the first measure of the score only, decision_debug is required. Do not omit it. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_score_output_rules() -> list[str]:
    return [
        "Allowed labels: normal, pickup, uncertain.",
        "For pickup, overusing uncertain is worse than a reasonable pickup suggestion. For all other cases, use uncertain when evidence is truly unreadable.",
        "Do not skip any measure_id. Every input measure_id must appear exactly once.",
        "For full score V1, never output multi_measure_rest, and rest_count must always be null.",
        "For the first measure of the score only, decision_debug is required. Do not omit it. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_legacy_rules() -> list[str]:
    return [
        *_ai_prompt_base_rules(),
        "In grand-staff or piano crops, the same time signature may appear on both staves; if visible in this crop, use one shared meter for the whole measure.",
        *_ai_prompt_duration_basics(),
        *_ai_prompt_grand_pickup_rules(),
        "In grand-staff or full-score music, vertically aligned notes/rests across staves happen at the same time, not one after another. Do not add treble plus bass or multiple instruments as separate beats; count the timeline horizontally.",
        "For grand-staff/piano crops, judge pickup by the whole vertical measure across both staves. One staff may play while the other rests or is silent; do not require both staves to have notes.",
        "Examples: in 2/4, one quarter-note chord is 1 of 2 beats, so pickup if first measure. In 4/4, one half-note chord is 2 of 4 beats, so pickup if first measure. In 6/8, one dotted-quarter chord is 3 of 6 eighth-beats, so pickup if first unless more duration follows. If all visible staves show one aligned quarter-note event in 2/4, that is one beat total, so pickup if first unless another beat or rest follows.",
        "A multi-measure rest may use either the modern H-bar style or an older style made from a horizontal bar plus one or more vertical bars.",
        "In the older style, the vertical bars may be short or long, and there may be more than one.",
        "Check multi-measure rests before meter. A confident multi_measure_rest label requires a clearly readable count number of 2 or more. Without a visible count number, return uncertain.",
        "A visible count of 2 or more above that old-style symbol is strong evidence for multi_measure_rest.",
        "If label is multi_measure_rest, include integer rest_count of 2 or more. A visible count of 1 means the measure is normal, not multi_measure_rest.",
        "A plain one-measure rest without the old-style vertical-bar structure is normal, not multi_measure_rest.",
        "For grand-staff/piano crops, return multi_measure_rest only if both staves clearly share the same multi-measure rest count. If one staff has music, no count, or a different count, do not return multi_measure_rest.",
        *_ai_prompt_output_rules(),
    ]


def _ai_prompt_rules_for_score_type(score_type: str | None) -> list[str]:
    normalized = _normalize_ai_score_type(score_type)
    if normalized == "single":
        return [*_ai_prompt_base_rules(), *_ai_prompt_single_rules(), *_ai_prompt_single_output_rules()]
    if normalized == "grand":
        return [*_ai_prompt_base_rules(), *_ai_prompt_grand_rules(), *_ai_prompt_grand_output_rules()]
    if normalized == "score":
        return [*_ai_prompt_base_rules(), *_ai_prompt_score_rules(), *_ai_prompt_score_output_rules()]
    return _ai_prompt_legacy_rules()


def _build_system_measure_request(
    job_id: str,
    run_id: int,
    system_row: dict,
    measure_rows: list[dict],
    page,
    pdf_source: str = "baseline",
    prev_system_row: dict | None = None,
    next_system_row: dict | None = None,
    artifacts: dict | None = None,
    debug_crop_rows: list[dict] | None = None,
    remembered_time_signature_in: str | None = None,
    score_type: str | None = None,
) -> tuple[dict, int]:
    content: list[dict] = []
    system_id = str(system_row.get("system_id") or "").strip()
    page_number = _safe_int(system_row.get("page"), _safe_int((measure_rows[0] if measure_rows else {}).get("page"), 1))
    normalized_score_type = _normalize_ai_score_type(score_type)
    score_allowed_labels = ["normal", "pickup", "uncertain"] if normalized_score_type == "score" else ["normal", "pickup", "multi_measure_rest", "uncertain"]
    score_label_shape = "normal|pickup|uncertain" if normalized_score_type == "score" else "normal|pickup|multi_measure_rest|uncertain"
    score_maybe_label_shape = "pickup|null" if normalized_score_type == "score" else "pickup|multi_measure_rest|null"
    intro = {
        "job_id": str(job_id),
        "run_id": int(run_id),
        "system_id": system_id,
        "page_number": int(page_number),
        "score_type": normalized_score_type,
        "remembered_time_signature_in": None,
        "instructions": {
            "task": "Classify each already-detected sheet-music measure conservatively.",
            "allowed_labels": score_allowed_labels,
            "rules": _ai_prompt_rules_for_score_type(score_type),
            "output_shape": {
                "provider": _requested_ai_provider_name(),
                "suggestions": [
                    {
                        "measure_id": "string",
                        "label": score_label_shape,
                        "measure_completeness": "full|incomplete|unclear",
                        "unclear_reason": "time_signature_not_clear|too_dense_to_count|crop_cut_off|split_may_be_wrong|ornament_or_tie_confusion|not_enough_visual_evidence|null",
                        "rest_count": "integer|null",
                        "confidence": "low|medium|high",
                        "maybe_label": score_maybe_label_shape,
                        "maybe_rest_count": "integer|null",
                        "decision_debug": {
                            "active_meter_read": "2/4|3/4|4/4|6/8|common_time|cut_time|unknown|null",
                            "duration_judgment": "full|short|unclear|null",
                            "rhythm_basis": "single_event|chord_single_event|multiple_events|rest_or_silence|unclear|null",
                            "decision_reason": "fills_meter|short_for_meter|meter_unclear|rhythm_unclear|not_first_measure|other|null",
                            "notehead_fill_read": "filled|open|unclear|null",
                            "stem_or_beam_read": "stem|flag_or_beam|none|unclear|null",
                            "dot_seen": "true|false|unclear|null",
                            "note_value_read": "quarter|half|whole|eighth|other|unclear|null",
                            "counted_beat_units": "short text like 1 quarter beat|3 quarter beats|unclear|null",
                            "debug_note": "1-3 short sentences, max 50 words|null",
                        },
                    }
                ],
                "warnings": [{"type": "string", "system_id": "string", "system_index": "integer", "message": "string"}],
            },
        },
        "measures": [
            {
                "measure_id": str(row.get("measure_id") or "").strip(),
                "order_index_in_system": _safe_int(row.get("measure_local_index"), idx),
                "is_first_measure_of_score": _safe_int(row.get("global_index"), -1) == 0,
            }
            for idx, row in enumerate(measure_rows)
        ],
    }
    content.append({"type": "text", "text": json.dumps(intro, ensure_ascii=True)})
    reference_content, reference_examples_attached = _build_old_style_multi_rest_reference_content()
    content.extend(reference_content)

    for idx, row in enumerate(measure_rows):
        next_row = measure_rows[idx + 1] if idx + 1 < len(measure_rows) else None
        crop_spec = _measure_crop_spec(page.rect, row, next_row, system_row, prev_system_row, next_system_row)
        clip = crop_spec["clip"]
        image_bytes = _render_measure_crop_png(page, clip)
        if artifacts is not None and debug_crop_rows is not None:
            measure_id = str(row.get("measure_id") or "").strip()
            crop_uri = _ai_debug_crop_measure_uri(artifacts, system_id, measure_id)
            _upload_bytes_to_gcs(image_bytes, crop_uri, content_type="image/png")
            debug_crop_rows.append(
                {
                    "measure_id": measure_id,
                    "system_id": system_id,
                    "page_number": int(page_number),
                    "pdf_source": pdf_source,
                    "display_system_number": _debug_display_system_number(system_row.get("system_index")),
                    "display_measure_number": _debug_display_measure_number(row.get("measure_local_index")),
                    "display_location": (
                        f"Page {int(page_number)}, "
                        f"Staff {_debug_display_system_number(system_row.get('system_index'))}, "
                        f"Measure {_debug_display_measure_number(row.get('measure_local_index'))}"
                    ),
                    "order_index_in_system": _safe_int(row.get("measure_local_index"), idx),
                    "crop_uri": crop_uri,
                    "clip_rect": {
                        "x0": float(clip.x0),
                        "y0": float(clip.y0),
                        "x1": float(clip.x1),
                        "y1": float(clip.y1),
                    },
                    "measure_bounds": crop_spec["measure_bounds"],
                    "padding": crop_spec["padding"],
                    "system_bounds": crop_spec["system_bounds"],
                }
            )
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "measure_id": str(row.get("measure_id") or "").strip(),
                        "order_index_in_system": _safe_int(row.get("measure_local_index"), idx),
                        "is_first_measure_of_score": _safe_int(row.get("global_index"), -1) == 0,
                        "pdf_source": pdf_source,
                    },
                    ensure_ascii=True,
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                },
            }
        )

    return (
        {
            "model": _requested_ai_model_name(),
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        },
        int(reference_examples_attached),
    )


def _generate_ai_suggestions_for_system_batch(
    job_id: str,
    run_id: int,
    systems: list[dict] | None,
    system_row: dict,
    system_measures: list[dict],
    source_state_version: str | None,
    artifacts: dict,
    remembered_time_signature_in: str | None = None,
    score_type: str | None = None,
) -> dict:
    model_name = _requested_ai_model_name()
    if not model_name or model_name == "unknown":
        raise AiSuggestError(provider_status=503, detail="provider_not_configured")

    debug_enabled = _ai_suggest_debug_enabled()
    debug_crop_rows: list[dict] = []
    pdf_source = "baseline"
    reference_examples_attached = 0

    def _finalize_debug_crops() -> dict | None:
        if not debug_enabled or not debug_crop_rows:
            return None
        payload = _build_ai_debug_crops_manifest(job_id, int(run_id), artifacts, debug_crop_rows, pdf_source=pdf_source)
        payload["reference_examples_attached"] = int(reference_examples_attached)
        return payload

    with TemporaryDirectory(prefix="omr-ai-suggest-step-") as tmp:
        tmpdir = Path(tmp)
        in_pdf, pdf_source = _resolve_ai_crop_pdf_source(artifacts, tmpdir)
        doc = fitz.open(str(in_pdf))
        try:
            ordered_systems = _sorted_system_rows(systems or [])
            prev_system_row, next_system_row = _same_page_neighbor_systems(ordered_systems, system_row)
            page_number = _safe_int(system_row.get("page"), _safe_int(system_measures[0].get("page"), 1))
            page_index = max(0, int(page_number) - 1)
            if page_index >= len(doc):
                raise AiSuggestError(provider_status=500, detail=f"invalid_page_index:{page_number}")
            page = doc[page_index]
            payload, reference_examples_attached = _build_system_measure_request(
                job_id,
                int(run_id),
                system_row,
                system_measures,
                page,
                pdf_source=pdf_source,
                prev_system_row=prev_system_row,
                next_system_row=next_system_row,
                artifacts=artifacts if debug_enabled else None,
                debug_crop_rows=debug_crop_rows if debug_enabled else None,
                remembered_time_signature_in=remembered_time_signature_in,
                score_type=score_type,
            )
            message = _ai_messages_create(payload)
            parsed = _parse_anthropic_suggestions_message(message)
            system_suggestions = parsed.get("suggestions")
            if not isinstance(system_suggestions, list):
                raise AiSuggestError(detail=f"malformed_response: suggestions missing for {system_row.get('system_id')}")
            expected_ids = {str(row.get("measure_id") or "").strip() for row in system_measures}
            seen_ids: set[str] = set()
            for row in system_suggestions:
                if not isinstance(row, dict):
                    raise AiSuggestError(detail=f"malformed_response: suggestion entry must be object for {system_row.get('system_id')}")
                measure_id = str(row.get("measure_id") or "").strip()
                if measure_id not in expected_ids:
                    raise AiSuggestError(detail=f"malformed_response: unknown measure_id {measure_id} for {system_row.get('system_id')}")
                if measure_id in seen_ids:
                    raise AiSuggestError(detail=f"malformed_response: duplicate measure_id {measure_id} for {system_row.get('system_id')}")
                seen_ids.add(measure_id)
            if seen_ids != expected_ids:
                missing = sorted(expected_ids - seen_ids)
                raise AiSuggestError(detail=f"malformed_response: missing_measure_ids={','.join(missing[:10])} for {system_row.get('system_id')}")

            normalized = _normalize_ai_suggestions_result(
                parsed,
                {"systems": [system_row], "measures": list(system_measures)},
                int(run_id),
                source_state_version,
                remembered_time_signature_in=remembered_time_signature_in,
            )
            normalized["pdf_source"] = pdf_source
            normalized["reference_examples_attached"] = int(reference_examples_attached)
            debug_crops = _finalize_debug_crops()
            if debug_crops is not None:
                normalized["debug_crops"] = debug_crops
            return normalized
        except AiSuggestError as exc:
            debug_crops = _finalize_debug_crops()
            if debug_crops is not None:
                exc.debug_crops = debug_crops
            raise
        except Exception as exc:
            debug_crops = _finalize_debug_crops()
            raise AiSuggestError(provider_status=500, detail=_safe_error_text(exc), debug_crops=debug_crops) from exc
        finally:
            doc.close()


def _generate_ai_suggestions_for_job(
    job_id: str,
    run_id: int,
    editable_state: dict,
    mapping_summary: dict,
    artifacts: dict,
    score_type: str | None = None,
) -> dict:
    model_name = _requested_ai_model_name()
    if not model_name or model_name == "unknown":
        raise AiSuggestError(provider_status=503, detail="provider_not_configured")

    systems = _sorted_system_rows(editable_state.get("systems") or [])
    measures = _ai_suggest_candidate_measures(editable_state)
    grouped_measures: dict[str, list[dict]] = {}
    for row in measures:
        system_id = str(row.get("system_id") or "").strip()
        if not system_id:
            continue
        grouped_measures.setdefault(system_id, []).append(row)

    warnings: list[dict] = []
    suggestions: list[dict] = []
    debug_enabled = _ai_suggest_debug_enabled()
    debug_crop_rows: list[dict] = []
    pdf_source = "baseline"
    reference_examples_attached = 0

    def _finalize_debug_crops() -> dict | None:
        if not debug_enabled or not debug_crop_rows:
            return None
        payload = _build_ai_debug_crops_manifest(job_id, int(run_id), artifacts, debug_crop_rows, pdf_source=pdf_source)
        payload["reference_examples_attached"] = int(reference_examples_attached)
        return payload

    with TemporaryDirectory(prefix="omr-ai-suggest-") as tmp:
        tmpdir = Path(tmp)
        in_pdf, pdf_source = _resolve_ai_crop_pdf_source(artifacts, tmpdir)
        doc = fitz.open(str(in_pdf))
        try:
            for system_row in systems:
                system_id = str(system_row.get("system_id") or "").strip()
                system_measures = grouped_measures.get(system_id) or []
                if not system_id or not system_measures:
                    continue
                prev_system_row, next_system_row = _same_page_neighbor_systems(systems, system_row)
                page_number = _safe_int(system_row.get("page"), _safe_int(system_measures[0].get("page"), 1))
                page_index = max(0, int(page_number) - 1)
                if page_index >= len(doc):
                    raise AiSuggestError(provider_status=500, detail=f"invalid_page_index:{page_number}")
                page = doc[page_index]
                payload, payload_reference_examples_attached = _build_system_measure_request(
                    job_id,
                    int(run_id),
                    system_row,
                    system_measures,
                    page,
                    pdf_source=pdf_source,
                    prev_system_row=prev_system_row,
                    next_system_row=next_system_row,
                    artifacts=artifacts if debug_enabled else None,
                    debug_crop_rows=debug_crop_rows if debug_enabled else None,
                    score_type=score_type,
                )
                reference_examples_attached = max(
                    int(reference_examples_attached),
                    int(payload_reference_examples_attached),
                )
                message = _ai_messages_create(payload)
                parsed = _parse_anthropic_suggestions_message(message)
                system_suggestions = parsed.get("suggestions")
                if not isinstance(system_suggestions, list):
                    raise AiSuggestError(detail=f"malformed_response: suggestions missing for {system_id}")
                expected_ids = {str(row.get("measure_id") or "").strip() for row in system_measures}
                seen_ids: set[str] = set()
                for row in system_suggestions:
                    if not isinstance(row, dict):
                        raise AiSuggestError(detail=f"malformed_response: suggestion entry must be object for {system_id}")
                    measure_id = str(row.get("measure_id") or "").strip()
                    if measure_id not in expected_ids:
                        raise AiSuggestError(detail=f"malformed_response: unknown measure_id {measure_id} for {system_id}")
                    if measure_id in seen_ids:
                        raise AiSuggestError(detail=f"malformed_response: duplicate measure_id {measure_id} for {system_id}")
                    seen_ids.add(measure_id)
                    suggestions.append(row)
                if seen_ids != expected_ids:
                    missing = sorted(expected_ids - seen_ids)
                    raise AiSuggestError(detail=f"malformed_response: missing_measure_ids={','.join(missing[:10])} for {system_id}")

                system_warnings = parsed.get("warnings")
                if isinstance(system_warnings, list):
                    for warning in system_warnings:
                        if not isinstance(warning, dict):
                            continue
                        if not str(warning.get("system_id") or "").strip():
                            warning = dict(warning)
                            warning["system_id"] = system_id
                        if warning.get("system_index") is None:
                            warning = dict(warning)
                            warning["system_index"] = _safe_int(system_row.get("system_index"), 0)
                        warnings.append(warning)
        except AiSuggestError as exc:
            debug_crops = _finalize_debug_crops()
            if debug_crops is not None:
                exc.debug_crops = debug_crops
            raise
        except Exception as exc:
            debug_crops = _finalize_debug_crops()
            raise AiSuggestError(provider_status=500, detail=_safe_error_text(exc), debug_crops=debug_crops) from exc
        finally:
            doc.close()

    result = {
        "provider": _requested_ai_provider_name(),
        "model": model_name,
        "suggestions": suggestions,
        "warnings": warnings,
        "pdf_source": pdf_source,
        "reference_examples_attached": int(reference_examples_attached),
    }
    debug_crops = _finalize_debug_crops()
    if debug_crops is not None:
        result["debug_crops"] = debug_crops
    return result


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
    x_left, y_top = _green_box_point_to_pdf_ink(page, x_left, y_top)
    tw = float(fitz.get_text_length(text, fontsize=MEASURE_TEXT_SIZE))
    x_text = min(max(0.0, float(x_left) - (tw / 2.0)), max(0.0, float(page_rect.width) - tw - 2.0))
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


def _erase_label_area(page: fitz.Page, page_rect: fitz.Rect, area: dict) -> bool:
    if not isinstance(area, dict):
        return False
    rect = area.get("rect")
    if not isinstance(rect, dict):
        return False
    try:
        left = float(rect.get("left"))
        right = float(rect.get("right"))
        top = float(rect.get("top"))
        bottom = float(rect.get("bottom"))
    except Exception:
        return False
    x0 = max(0.0, min(left, float(page_rect.width)))
    x1 = max(0.0, min(right, float(page_rect.width)))
    y0 = max(0.0, min(top, float(page_rect.height)))
    y1 = max(0.0, min(bottom, float(page_rect.height)))
    if x1 <= x0 or y1 <= y0:
        return False
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=MEASURE_TEXT_BG_COLOR, fill=MEASURE_TEXT_BG_COLOR)
    return True


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
    layout_profile = _profile_system_layouts(systems)
    suspicious_system_ids = set(layout_profile.get("suspicious_system_ids") or set())

    # Step 1: Build per-page system lookup from anchors.
    page_systems: dict[int, list[tuple[str, int, float, float, bool, str]]] = {}  # page -> [(system_id, system_index, y_top, y_bot, suspicious, source)]
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
        page_systems.setdefault(page, []).append((sid, sidx, y_top, y_bot, sid in suspicious_system_ids, _row_source(s)))
    # Sort each page's systems by y_top (top-to-bottom on page).
    for page in page_systems:
        page_systems[page].sort(key=lambda t: t[2])

    if not page_systems:
        return 0

    tolerance = 5.0  # PDF points tolerance for overlap
    reassigned = 0

    # Step 2: For each measure, find the best-matching system by y-overlap.
    for m in measures:
        if _row_source(m) == ROW_SOURCE_MANUAL:
            m.pop("_system_reassigned", None)
            continue
        try:
            m_page = int(m["page"])
            m_y_top = float(m["y_top"])
            m_y_bot = float(m["y_bottom"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates = page_systems.get(m_page)
        if not candidates:
            continue

        m_center = (m_y_top + m_y_bot) / 2.0
        candidate_rows: list[dict] = []

        for sid, sidx, s_y_top, s_y_bot, suspicious, source in candidates:
            if source == ROW_SOURCE_MANUAL:
                continue
            overlap = min(m_y_bot, s_y_bot) - max(m_y_top, s_y_top) + tolerance
            s_center = (s_y_top + s_y_bot) / 2.0
            center_dist = abs(m_center - s_center)
            candidate_rows.append(
                {
                    "system_id": sid,
                    "system_index": sidx,
                    "overlap": overlap,
                    "center_dist": center_dist,
                    "suspicious": bool(suspicious),
                }
            )

        def _best_candidate(rows: list[dict]) -> dict | None:
            if not rows:
                return None
            return sorted(rows, key=lambda row: (-float(row["overlap"]), float(row["center_dist"])))[0]

        normal_positive = [row for row in candidate_rows if not row["suspicious"] and float(row["overlap"]) > 0.0]
        if normal_positive:
            best = _best_candidate(normal_positive)
        else:
            positive_rows = [row for row in candidate_rows if float(row["overlap"]) > 0.0]
            suspicious_positive = [row for row in positive_rows if row["suspicious"]]
            if len(positive_rows) == 1 and len(suspicious_positive) == 1:
                best = suspicious_positive[0]
            else:
                non_suspicious_rows = [row for row in candidate_rows if not row["suspicious"]]
                best = _best_candidate(non_suspicious_rows or candidate_rows)

        if not isinstance(best, dict):
            continue
        best_sid = str(best.get("system_id") or "").strip()
        best_sidx = _safe_int(best.get("system_index"), 0)
        if not best_sid:
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
            m["_system_reassigned"] = True
            reassigned += 1
        else:
            m.pop("_system_reassigned", None)

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
            if _row_source(m) == ROW_SOURCE_MANUAL:
                manual_row_id = _normalize_artifact_key(m.get("manual_row_id"))[:64]
                if manual_row_id:
                    m["measure_id"] = _manual_measure_id(manual_row_id, local_idx)
            elif bool(m.pop("_system_reassigned", False)) or not str(m.get("measure_id") or "").strip():
                m["measure_id"] = f"p{page_no}_s{sidx}_m{local_idx}"
            else:
                m.pop("_system_reassigned", None)

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


def _profile_system_layouts(systems: list[dict] | None) -> dict:
    sorted_systems = _sorted_system_rows(systems)
    by_page: dict[int, list[dict]] = {}
    suspicious_system_ids: set[str] = set()

    for system in sorted_systems:
        system["suspicious_partial_staff"] = False
        anchor = system.get("anchor")
        if not isinstance(anchor, dict):
            continue
        page = _safe_int(system.get("page"), 0)
        y_top = _safe_float(anchor.get("y_top"), 0.0)
        y_bottom = _safe_float(anchor.get("y_bottom"), 0.0)
        height = y_bottom - y_top
        if page <= 0 or height <= 0.0:
            continue
        by_page.setdefault(page, []).append(
            {
                "system_id": str(system.get("system_id") or "").strip(),
                "system_index": _safe_int(system.get("system_index"), 0),
                "height": float(height),
                "row": system,
            }
        )

    page_profiles: dict[int, dict] = {}
    for page, rows in by_page.items():
        heights = [row["height"] for row in rows if row["height"] > 0.0]
        page_median = float(median(heights)) if heights else 0.0
        threshold = page_median * SUSPICIOUS_PARTIAL_STAFF_HEIGHT_RATIO if page_median > 0.0 else 0.0
        suspicious_rows: list[dict] = []
        for row in rows:
            suspicious = bool(page_median > 0.0 and row["height"] < threshold)
            row["row"]["suspicious_partial_staff"] = suspicious
            if suspicious and row["system_id"]:
                suspicious_system_ids.add(row["system_id"])
                suspicious_rows.append(
                    {
                        "system_id": row["system_id"],
                        "system_index": row["system_index"],
                        "height": row["height"],
                    }
                )
        page_profiles[page] = {
            "median_height": page_median,
            "threshold_height": threshold,
            "suspicious_systems": suspicious_rows,
        }

    return {
        "by_page": page_profiles,
        "suspicious_system_ids": suspicious_system_ids,
    }


def _refresh_editable_state_qa(
    editable_state: dict | None,
    systems: list[dict] | None,
    measures: list[dict] | None,
) -> dict:
    if not isinstance(editable_state, dict):
        return {}

    sorted_systems = _sorted_system_rows(systems)
    ordered_measures = _sorted_measure_rows(measures)
    layout_profile = _profile_system_layouts(sorted_systems)
    by_page = layout_profile.get("by_page") or {}

    warnings: list[dict] = []
    warning_pages: set[int] = set()

    for page, profile in sorted(by_page.items()):
        for row in profile.get("suspicious_systems") or []:
            system_id = str(row.get("system_id") or "").strip()
            warnings.append(
                {
                    "type": "suspicious_partial_staff",
                    "page": int(page),
                    "system_id": system_id,
                    "message": f"System {system_id or '?'} looks unusually short for page {page}.",
                }
            )
            warning_pages.add(int(page))

    measure_counts_by_system: dict[str, int] = {}
    for measure in ordered_measures:
        system_id = str(measure.get("system_id") or "").strip()
        if system_id:
            measure_counts_by_system[system_id] = measure_counts_by_system.get(system_id, 0) + 1

    for system in sorted_systems:
        system_id = str(system.get("system_id") or "").strip()
        if not system_id or measure_counts_by_system.get(system_id, 0) > 0:
            continue
        page = _safe_int(system.get("page"), 0)
        warnings.append(
            {
                "type": "system_has_no_measures",
                "page": page,
                "system_id": system_id,
                "message": f"System {system_id} has no measures after reassignment.",
            }
        )
        if page > 0:
            warning_pages.add(page)

    systems_by_page: dict[int, list[dict]] = {}
    for system in sorted_systems:
        page = _safe_int(system.get("page"), 0)
        if page > 0:
            systems_by_page.setdefault(page, []).append(system)

    for page, rows in sorted(systems_by_page.items()):
        duplicate_values: dict[str, list[str]] = {}
        for row in rows[1:]:
            start_value = str(row.get("current_value") or row.get("value") or "").strip()
            if not start_value:
                continue
            duplicate_values.setdefault(start_value, []).append(str(row.get("system_id") or "").strip())
        for start_value, system_ids in sorted(duplicate_values.items()):
            unique_ids = [sid for sid in system_ids if sid]
            if len(unique_ids) < 2:
                continue
            warnings.append(
                {
                    "type": "duplicate_later_system_start",
                    "page": int(page),
                    "message": f"Later systems on page {page} share start value {start_value}: {', '.join(unique_ids)}.",
                }
            )
            warning_pages.add(int(page))

    qa = {
        "status": "warning" if warnings else "ok",
        "total_systems": len(sorted_systems),
        "warning_count": len(warnings),
        "warning_pages": sorted(warning_pages),
        "warnings": warnings,
    }
    editable_state["qa"] = qa
    return qa


def _refresh_editable_state_systems_and_measures(
    editable_state: dict,
    *,
    ending_debug_ctx: dict | None = None,
) -> tuple[list[dict], list[dict], int, dict]:
    systems, measures = _merge_manual_rows_into_state(editable_state)

    _editable_rest_measures(editable_state)
    _editable_pickup_measures(editable_state)
    reassign_count = _reassign_measures_to_nearest_system(systems, measures)
    systems, measures = _reindex_system_and_measure_order(systems, measures)
    systems, measures, _, _ = _recompute_measure_numbering(
        systems,
        measures,
        editable_state,
        ending_debug_ctx=ending_debug_ctx,
    )
    editable_state["systems"] = systems
    editable_state["measures"] = measures
    editable_state["staff_boxes"] = []
    qa = _refresh_editable_state_qa(editable_state, systems, measures)
    return systems, measures, reassign_count, qa


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


def _apply_legacy_system_rest_carryover(
    current_value: int,
    system_id: str | None,
    rest_systems: dict[str, int],
    exact_rest_system_ids: set[str],
) -> int:
    if not system_id or system_id in exact_rest_system_ids:
        return int(current_value)
    rest_count = _safe_int(rest_systems.get(system_id), 0)
    if rest_count <= 0:
        return int(current_value)
    return int(current_value) + rest_count


def _apply_measure_override_anchor(
    current_value: int,
    measure_id: str,
    measure_overrides: dict[str, int],
) -> int:
    if measure_id and measure_id in measure_overrides:
        return int(measure_overrides[measure_id])
    return int(current_value)


def _measure_override_value(
    measure_id: str,
    measure_overrides: dict[str, int],
) -> int | None:
    if measure_id and measure_id in measure_overrides:
        return int(measure_overrides[measure_id])
    return None


def _pickup_active_for_measure(measure_id: str, pickup_measures: dict[str, bool]) -> bool:
    return bool(pickup_measures.get(measure_id)) if measure_id else False


def _relabel_has_ending_debug(editable_state: dict | None, edits: list[dict] | None) -> bool:
    for raw_edit in edits or []:
        if isinstance(raw_edit, dict) and str(raw_edit.get("type") or "").strip() == "set_ending":
            return True
    endings_map = (editable_state or {}).get("endings")
    return bool(endings_map) if isinstance(endings_map, dict) else False


def _log_relabel_ending_debug(
    trace_id: str,
    job_id: str,
    run_id: int,
    stage: str,
    payload: dict | None = None,
) -> None:
    row = {
        "trace_id": trace_id,
        "job_id": job_id,
        "run_id": int(run_id),
        "stage": str(stage or "").strip(),
    }
    if isinstance(payload, dict):
        row.update(payload)
    print(f"RELABEL_ENDING_DEBUG {json.dumps(row, separators=(',', ':'), sort_keys=True, default=str)}")


def _build_ending_group_debug_snapshot(
    ordered_measures: list[dict],
    endings_map: dict[str, str],
    pickup_measures: dict[str, bool],
) -> dict:
    entries: dict[str, dict] = {}
    raw_rows: list[dict] = []
    ignored_rows: list[dict] = []
    groups: list[dict] = []
    pending_first_rows: list[dict] = []
    pending_second_rows: list[dict] = []
    group_id = 0

    def _base_row(measure: dict, raw_kind: str, pickup_active: bool) -> dict:
        return {
            "measure_id": str(measure.get("measure_id") or "").strip(),
            "kind": str(raw_kind or "").strip(),
            "page": _safe_int(measure.get("page"), 0),
            "system_id": str(measure.get("system_id") or "").strip(),
            "system_index": _safe_int(measure.get("system_index"), 0),
            "measure_local_index": _safe_int(measure.get("measure_local_index"), 0),
            "pickup_active": bool(pickup_active),
        }

    def _mark_pending_as_ignored(reason: str) -> None:
        nonlocal pending_first_rows, pending_second_rows
        for row in pending_first_rows:
            ignored_rows.append({**row, "reason": reason})
        for row in pending_second_rows:
            ignored_rows.append({**row, "reason": reason})

    def _flush_pending(reason_if_invalid: str = "incomplete_group") -> None:
        nonlocal pending_first_rows, pending_second_rows, group_id
        if pending_first_rows and pending_second_rows:
            groups.append(
                {
                    "group_id": group_id,
                    "ending1_ids": [row["measure_id"] for row in pending_first_rows],
                    "ending2_ids": [row["measure_id"] for row in pending_second_rows],
                }
            )
            for index, row in enumerate(pending_first_rows):
                entries[row["measure_id"]] = {
                    "group_id": group_id,
                    "kind": "1",
                    "branch_index": index,
                    "first_len": len(pending_first_rows),
                    "second_len": len(pending_second_rows),
                }
            for index, row in enumerate(pending_second_rows):
                entries[row["measure_id"]] = {
                    "group_id": group_id,
                    "kind": "2",
                    "branch_index": index,
                    "first_len": len(pending_first_rows),
                    "second_len": len(pending_second_rows),
                }
            group_id += 1
        elif pending_first_rows or pending_second_rows:
            _mark_pending_as_ignored(reason_if_invalid)
        pending_first_rows = []
        pending_second_rows = []

    for measure in ordered_measures:
        measure_id = str(measure.get("measure_id") or "").strip()
        raw_kind = str(endings_map.get(measure_id) or "").strip() if measure_id else ""
        if not raw_kind:
            _flush_pending()
            continue

        pickup_active = _pickup_active_for_measure(measure_id, pickup_measures)
        if _is_excluded_from_counting(measure):
            _flush_pending()
            ignored_rows.append({**_base_row(measure, raw_kind, pickup_active), "reason": "excluded_from_counting"})
            continue
        base_row = _base_row(measure, raw_kind, pickup_active)
        raw_rows.append(base_row)

        if raw_kind not in ("1", "2"):
            _flush_pending()
            ignored_rows.append({**base_row, "reason": "invalid_kind"})
            continue

        if pickup_active:
            _flush_pending()
            ignored_rows.append({**base_row, "reason": "pickup_blocked"})
            continue

        if raw_kind == "1":
            if pending_second_rows:
                _flush_pending()
            pending_first_rows.append(base_row)
            continue

        if pending_first_rows:
            pending_second_rows.append(base_row)
            continue

        ignored_rows.append({**base_row, "reason": "orphan_ending2"})

    _flush_pending()
    return {
        "entries": entries,
        "groups": groups,
        "raw_rows": raw_rows,
        "ignored_rows": ignored_rows,
    }


def _ending_group_entries_by_measure_id(
    ordered_measures: list[dict],
    endings_map: dict[str, str],
    pickup_measures: dict[str, bool],
) -> dict[str, dict]:
    snapshot = _build_ending_group_debug_snapshot(ordered_measures, endings_map, pickup_measures)
    return snapshot.get("entries") or {}


def _close_numbering_ending_group(current_value: int, group_state: dict | None) -> int:
    if not group_state:
        return int(current_value)
    next_values = [int(current_value)]
    first_next = group_state.get("first_next_value")
    second_next = group_state.get("second_next_value")
    if first_next is not None:
        next_values.append(int(first_next))
    if second_next is not None:
        next_values.append(int(second_next))
    return max(next_values)


def _resolve_grouped_ending_label(
    current_value: int,
    measure_override_value: int | None,
    ending_entry: dict,
    group_state: dict,
) -> tuple[int, int, dict]:
    kind = str(ending_entry.get("kind") or "")
    branch_index = _safe_int(ending_entry.get("branch_index"), 0)

    if group_state.get("start_value") is None:
        group_state["start_value"] = int(measure_override_value) if measure_override_value is not None else int(current_value)
    start_value = int(group_state["start_value"])

    if kind == "2":
        if group_state.get("second_next_value") is None:
            group_state["second_next_value"] = int(start_value)
        branch_value = int(group_state["second_next_value"])
        label_value = int(measure_override_value) if measure_override_value is not None else branch_value
        group_state["second_next_value"] = int(label_value) + 1
        return label_value, int(current_value), group_state

    if group_state.get("first_next_value") is None:
        group_state["first_next_value"] = int(start_value) + int(branch_index)
    branch_value = int(group_state["first_next_value"])
    label_value = int(measure_override_value) if measure_override_value is not None else branch_value
    group_state["first_next_value"] = int(label_value) + 1
    return label_value, int(group_state["first_next_value"]), group_state


def _apply_measure_label(
    measure: dict,
    measure_id: str,
    system_id: str,
    label: str,
    result_labels: dict[str, str],
    seq_starts_by_system: dict[str, int],
) -> None:
    if measure_id:
        result_labels[measure_id] = label
    if label:
        label_value = int(label)
        if system_id and system_id not in seq_starts_by_system:
            seq_starts_by_system[system_id] = label_value
    measure["current_value"] = label
    measure["value"] = label
    measure["render_label"] = label


def _apply_post_measure_rest(
    current_value: int,
    label_value: int,
    measure_id: str,
    rest_measures: dict[str, int],
) -> int:
    exact_rest_count = _safe_int(rest_measures.get(measure_id), 0) if measure_id else 0
    if exact_rest_count > 0:
        return int(label_value) + exact_rest_count
    return int(current_value)


def _system_start_anchor_measures(
    ordered_measures: list[dict] | None,
    result_labels: dict[str, str] | None,
    systems: list[dict] | None = None,
) -> list[tuple[dict, str]]:
    labels = result_labels if isinstance(result_labels, dict) else {}
    seen_system_ids: set[str] = set()
    anchor_rows: list[dict] = []
    system_rows_by_id = {
        str(system.get("system_id") or "").strip(): system
        for system in (systems or [])
        if isinstance(system, dict) and str(system.get("system_id") or "").strip()
    }

    def _fallback_measure_bounds(measure: dict) -> tuple[float, float, float, float] | None:
        if not isinstance(measure, dict):
            return None
        try:
            left = float(measure.get("x_left"))
        except Exception:
            return None
        try:
            right = float(measure.get("x_right")) if measure.get("x_right") is not None else float(left + 1.0)
        except Exception:
            right = float(left + 1.0)
        try:
            top = float(measure.get("y_top"))
        except Exception:
            return None
        try:
            bottom = float(measure.get("y_bottom")) if measure.get("y_bottom") is not None else float(top + 1.0)
        except Exception:
            bottom = float(top + 1.0)
        if right <= left:
            right = float(left + 1.0)
        if bottom <= top:
            bottom = float(top + 1.0)
        return (float(left), float(right), float(top), float(bottom))

    def _same_visual_row(
        left_bounds: tuple[float, float, float, float] | None,
        right_bounds: tuple[float, float, float, float] | None,
    ) -> bool:
        if left_bounds is None or right_bounds is None:
            return False
        _, _, left_top, left_bottom = left_bounds
        _, _, right_top, right_bottom = right_bounds
        left_height = float(left_bottom - left_top)
        right_height = float(right_bottom - right_top)
        if left_height <= 0.0 or right_height <= 0.0:
            return False
        shorter_height = min(left_height, right_height)
        taller_height = max(left_height, right_height)
        if shorter_height <= 0.0 or taller_height <= 0.0:
            return False
        height_ratio = shorter_height / taller_height
        if height_ratio < STAFF_START_SAME_ROW_MIN_HEIGHT_RATIO or height_ratio > STAFF_START_SAME_ROW_MAX_HEIGHT_RATIO:
            return False
        left_center = (left_top + left_bottom) / 2.0
        right_center = (right_top + right_bottom) / 2.0
        center_tolerance = shorter_height * STAFF_START_SAME_ROW_CENTER_TOLERANCE_RATIO
        if abs(left_center - right_center) > center_tolerance:
            return False
        overlap = max(0.0, min(left_bottom, right_bottom) - max(left_top, right_top))
        if overlap < (shorter_height * STAFF_START_SAME_ROW_OVERLAP_RATIO):
            return False
        return True

    for measure in ordered_measures or []:
        if not isinstance(measure, dict):
            continue
        system_id = str(measure.get("system_id") or "").strip()
        if not system_id or system_id in seen_system_ids:
            continue
        measure_id = str(measure.get("measure_id") or "").strip()
        label = str(labels.get(measure_id) or "").strip()
        if not label:
            continue
        seen_system_ids.add(system_id)
        system_row = system_rows_by_id.get(system_id)
        bounds = _system_visual_bounds(system_row, ordered_measures) if system_row is not None else None
        if bounds is None:
            bounds = _fallback_measure_bounds(measure)
        anchor_rows.append(
            {
                "measure": measure,
                "label": label,
                "system_id": system_id,
                "page": _safe_int(
                    measure.get("page"),
                    _safe_int(system_row.get("page"), 0) if isinstance(system_row, dict) else 0,
                ),
                "x_left": _safe_float(measure.get("x_left"), bounds[0] if bounds is not None else 0.0),
                "bounds": bounds,
            }
        )

    if not anchor_rows or not system_rows_by_id:
        return [(row["measure"], row["label"]) for row in anchor_rows]

    parent = list(range(len(anchor_rows)))

    def _find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def _union(left_idx: int, right_idx: int) -> None:
        left_root = _find(left_idx)
        right_root = _find(right_idx)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_idx in range(len(anchor_rows)):
        left_row = anchor_rows[left_idx]
        for right_idx in range(left_idx + 1, len(anchor_rows)):
            right_row = anchor_rows[right_idx]
            if left_row["page"] != right_row["page"]:
                continue
            if _same_visual_row(left_row.get("bounds"), right_row.get("bounds")):
                _union(left_idx, right_idx)

    chosen_by_group: dict[int, int] = {}
    for idx, row in enumerate(anchor_rows):
        root = _find(idx)
        chosen_idx = chosen_by_group.get(root)
        if chosen_idx is None or (row["x_left"], idx) < (anchor_rows[chosen_idx]["x_left"], chosen_idx):
            chosen_by_group[root] = idx

    selected_indices = {idx for idx in chosen_by_group.values()}
    return [
        (row["measure"], row["label"])
        for idx, row in enumerate(anchor_rows)
        if idx in selected_indices
    ]


def _apply_pickup_measure_rest(
    current_value: int,
    measure_id: str,
    rest_measures: dict[str, int],
) -> int:
    exact_rest_count = _safe_int(rest_measures.get(measure_id), 0) if measure_id else 0
    if exact_rest_count > 0:
        return int(current_value) + exact_rest_count
    return int(current_value)


def _recompute_measure_numbering(
    systems: list[dict] | None,
    measures: list[dict] | None,
    editable_state: dict | None = None,
    ending_debug_ctx: dict | None = None,
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
    rest_measures = _editable_rest_measures(editable_state)
    pickup_measures = _editable_pickup_measures(editable_state)
    measure_overrides = _measure_number_overrides(editable_state)
    ending_snapshot = _build_ending_group_debug_snapshot(ordered_measures, endings_map, pickup_measures)
    ending_entries = ending_snapshot.get("entries") or {}

    if isinstance(ending_debug_ctx, dict):
        _log_relabel_ending_debug(
            str(ending_debug_ctx.get("trace_id") or ""),
            str(ending_debug_ctx.get("job_id") or ""),
            _safe_int(ending_debug_ctx.get("run_id"), 0),
            "groups",
            {
                "saved_endings": dict(sorted(endings_map.items())),
                "ordered_measures": ending_snapshot.get("raw_rows") or [],
                "groups": ending_snapshot.get("groups") or [],
                "ignored": ending_snapshot.get("ignored_rows") or [],
            },
        )

    exact_rest_system_ids: set[str] = set()
    for measure in ordered_measures:
        if _is_excluded_from_counting(measure):
            continue
        measure_id = str(measure.get("measure_id") or "").strip()
        system_id = str(measure.get("system_id") or "").strip()
        if not measure_id or not system_id:
            continue
        if _safe_int(rest_measures.get(measure_id), 0) > 0:
            exact_rest_system_ids.add(system_id)

    if ordered_measures:
        first_measure_id = str(ordered_measures[0].get("measure_id") or "").strip()
        if first_measure_id and first_measure_id in measure_overrides:
            first_start = int(measure_overrides[first_measure_id])

    result_labels: dict[str, str] = {}
    seq_starts_by_system: dict[str, int] = {}
    current_value = int(first_start)
    active_ending_group_id: int | None = None
    active_ending_group_state: dict | None = None
    current_sid: str | None = None

    for measure in ordered_measures:
        measure_id = str(measure.get("measure_id") or "").strip()
        system_id = str(measure.get("system_id") or "").strip()
        ending_entry = ending_entries.get(measure_id) if measure_id else None
        ending_group_id = _safe_int(ending_entry.get("group_id"), -1) if ending_entry else None

        if active_ending_group_id is not None and ending_group_id != active_ending_group_id:
            close_state = dict(active_ending_group_state or {})
            resumed_value = _close_numbering_ending_group(current_value, active_ending_group_state)
            if isinstance(ending_debug_ctx, dict):
                _log_relabel_ending_debug(
                    str(ending_debug_ctx.get("trace_id") or ""),
                    str(ending_debug_ctx.get("job_id") or ""),
                    _safe_int(ending_debug_ctx.get("run_id"), 0),
                    "close",
                    {
                        "group_id": int(active_ending_group_id),
                        "next_measure_id": measure_id,
                        "resume_value": int(resumed_value),
                        "first_next_value": _safe_int(close_state.get("first_next_value"), 0) if close_state.get("first_next_value") is not None else None,
                        "second_next_value": _safe_int(close_state.get("second_next_value"), 0) if close_state.get("second_next_value") is not None else None,
                        "start_value": _safe_int(close_state.get("start_value"), 0) if close_state.get("start_value") is not None else None,
                    },
                )
            current_value = resumed_value
            active_ending_group_id = None
            active_ending_group_state = None

        # Stage 1: apply any legacy staff-level carryover when crossing a system boundary.
        if system_id != current_sid:
            if current_sid is not None:
                current_value = _apply_legacy_system_rest_carryover(
                    current_value,
                    current_sid,
                    rest_systems,
                    exact_rest_system_ids,
                )
            current_sid = system_id

        # Stage 2: determine whether this physical measure is marked as pickup.
        if _is_excluded_from_counting(measure):
            _apply_measure_label(
                measure,
                measure_id,
                system_id,
                "",
                result_labels,
                seq_starts_by_system,
            )
            continue

        pickup_active = _pickup_active_for_measure(measure_id, pickup_measures)

        # Stage 3: pickup wins over same-measure numbering anchors.
        if pickup_active:
            _apply_measure_label(
                measure,
                measure_id,
                system_id,
                "",
                result_labels,
                seq_starts_by_system,
            )
            current_value = _apply_pickup_measure_rest(
                current_value,
                measure_id,
                rest_measures,
            )
            continue

        # Stage 4: compute the local numbering anchor for this counted measure.
        measure_override_value = _measure_override_value(measure_id, measure_overrides)
        current_value_before_label = int(current_value)

        # Stage 5: resolve the final local label for this counted measure.
        ending_type = str(ending_entry.get("kind") or "").strip() if ending_entry else ""
        if ending_entry:
            if active_ending_group_id != ending_group_id:
                active_ending_group_id = ending_group_id
                active_ending_group_state = {}
            label_value, current_value, active_ending_group_state = _resolve_grouped_ending_label(
                current_value,
                measure_override_value,
                ending_entry,
                active_ending_group_state or {},
            )
        else:
            label_value = int(measure_override_value) if measure_override_value is not None else int(current_value)
            current_value = int(label_value) + 1

        final_label = str(label_value)
        _apply_measure_label(
            measure,
            measure_id,
            system_id,
            final_label,
            result_labels,
            seq_starts_by_system,
        )

        # Stage 6: apply any exact measure rest after the local label is finalized.
        if ending_type == "2":
            if active_ending_group_state is not None:
                active_ending_group_state["second_next_value"] = _apply_post_measure_rest(
                    active_ending_group_state.get("second_next_value") or current_value,
                    label_value,
                    measure_id,
                    rest_measures,
                )
        elif ending_type == "1":
            if active_ending_group_state is not None:
                active_ending_group_state["first_next_value"] = _apply_post_measure_rest(
                    active_ending_group_state.get("first_next_value") or current_value,
                    label_value,
                    measure_id,
                    rest_measures,
                )
                current_value = int(active_ending_group_state["first_next_value"])
        else:
            current_value = _apply_post_measure_rest(
                current_value,
                label_value,
                measure_id,
                rest_measures,
            )

        if ending_entry and isinstance(ending_debug_ctx, dict):
            _log_relabel_ending_debug(
                str(ending_debug_ctx.get("trace_id") or ""),
                str(ending_debug_ctx.get("job_id") or ""),
                _safe_int(ending_debug_ctx.get("run_id"), 0),
                "numbering",
                {
                    "group_id": int(ending_group_id),
                    "measure_id": measure_id,
                    "kind": ending_type,
                    "branch_index": _safe_int(ending_entry.get("branch_index"), 0),
                    "current_value_before": int(current_value_before_label),
                    "override_value": int(measure_override_value) if measure_override_value is not None else None,
                    "group_start_value": _safe_int(active_ending_group_state.get("start_value"), 0) if active_ending_group_state and active_ending_group_state.get("start_value") is not None else None,
                    "assigned_label": int(label_value),
                    "first_next_value": _safe_int(active_ending_group_state.get("first_next_value"), 0) if active_ending_group_state and active_ending_group_state.get("first_next_value") is not None else None,
                    "second_next_value": _safe_int(active_ending_group_state.get("second_next_value"), 0) if active_ending_group_state and active_ending_group_state.get("second_next_value") is not None else None,
                    "current_value_after": int(current_value),
                },
            )

    if active_ending_group_id is not None:
        close_state = dict(active_ending_group_state or {})
        resumed_value = _close_numbering_ending_group(current_value, active_ending_group_state)
        if isinstance(ending_debug_ctx, dict):
            _log_relabel_ending_debug(
                str(ending_debug_ctx.get("trace_id") or ""),
                str(ending_debug_ctx.get("job_id") or ""),
                _safe_int(ending_debug_ctx.get("run_id"), 0),
                "close",
                {
                    "group_id": int(active_ending_group_id),
                    "next_measure_id": "",
                    "resume_value": int(resumed_value),
                    "first_next_value": _safe_int(close_state.get("first_next_value"), 0) if close_state.get("first_next_value") is not None else None,
                    "second_next_value": _safe_int(close_state.get("second_next_value"), 0) if close_state.get("second_next_value") is not None else None,
                    "start_value": _safe_int(close_state.get("start_value"), 0) if close_state.get("start_value") is not None else None,
                },
            )
        current_value = resumed_value

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


def _editable_endings_map(editable_state: dict) -> dict[str, str]:
    endings_map = editable_state.get("endings")
    if not isinstance(endings_map, dict):
        editable_state["endings"] = {}
        return editable_state["endings"]
    return endings_map


def _editable_rest_systems(editable_state: dict) -> dict[str, int]:
    rest_systems = editable_state.get("rest_systems")
    if not isinstance(rest_systems, dict):
        editable_state["rest_systems"] = {}
        return editable_state["rest_systems"]
    return rest_systems


def _editable_rest_measures(editable_state: dict) -> dict[str, int]:
    raw = editable_state.get("rest_measures")
    if not isinstance(raw, dict):
        editable_state["rest_measures"] = {}
        return editable_state["rest_measures"]

    cleaned: dict[str, int] = {}
    for raw_key, raw_value in raw.items():
        measure_id = str(raw_key or "").strip()
        if not measure_id:
            continue
        try:
            value = int(raw_value)
        except Exception:
            continue
        if value <= 0:
            continue
        cleaned[measure_id] = value

    editable_state["rest_measures"] = cleaned
    return cleaned


def _editable_pickup_measures(editable_state: dict) -> dict[str, bool]:
    raw = editable_state.get("pickup_measures")
    if not isinstance(raw, dict):
        editable_state["pickup_measures"] = {}
        return editable_state["pickup_measures"]

    cleaned: dict[str, bool] = {}
    for raw_key, raw_value in raw.items():
        measure_id = str(raw_key or "").strip()
        if not measure_id:
            continue
        if _safe_bool(raw_value, False):
            cleaned[measure_id] = True

    editable_state["pickup_measures"] = cleaned
    return cleaned


def _relabel_number_value(raw_edit: dict, rejected: list[dict]) -> int | None:
    try:
        new_value = int(raw_edit.get("value"))
    except Exception:
        rejected.append({"edit": raw_edit, "reason": "invalid_value"})
        return None
    if new_value < RELABEL_MIN_VALUE or new_value > RELABEL_MAX_VALUE:
        rejected.append(
            {
                "edit": raw_edit,
                "reason": "value_out_of_range",
                "min": RELABEL_MIN_VALUE,
                "max": RELABEL_MAX_VALUE,
            }
        )
        return None
    return int(new_value)


def _apply_legacy_system_start_edit(
    raw_edit: dict,
    system_ids: set[str],
    first_measure_by_system: dict[str, dict],
    measure_overrides: dict[str, int],
    applied: list[dict],
    rejected: list[dict],
) -> None:
    system_id = str(raw_edit.get("system_id") or "").strip()
    if not system_id or system_id not in system_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_system_id"})
        return

    new_value = _relabel_number_value(raw_edit, rejected)
    if new_value is None:
        return

    first_measure = first_measure_by_system.get(system_id)
    if not isinstance(first_measure, dict):
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return

    measure_id = str(first_measure.get("measure_id") or "").strip()
    if not measure_id:
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return

    measure_overrides[measure_id] = int(new_value)
    applied.append({"type": "set_system_start", "system_id": system_id, "value": int(new_value)})


def _apply_measure_number_edit(
    raw_edit: dict,
    measure_ids: set[str],
    measure_rows_by_id: dict[str, dict],
    measure_overrides: dict[str, int],
    applied: list[dict],
    rejected: list[dict],
) -> None:
    measure_id = str(raw_edit.get("measure_id") or "").strip()
    if not measure_id:
        rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
        return
    if measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return
    if _is_excluded_from_counting(measure_rows_by_id.get(measure_id)):
        rejected.append({"edit": raw_edit, "reason": "measure_excluded_from_counting"})
        return

    new_value = _relabel_number_value(raw_edit, rejected)
    if new_value is None:
        return

    measure_overrides[measure_id] = int(new_value)
    applied.append({"type": "set_measure_number", "measure_id": measure_id, "value": int(new_value)})


def _apply_clear_measure_number_edit(
    raw_edit: dict,
    measure_ids: set[str],
    measure_rows_by_id: dict[str, dict],
    measure_overrides: dict[str, int],
    applied: list[dict],
    rejected: list[dict],
) -> None:
    measure_id = str(raw_edit.get("measure_id") or "").strip()
    if not measure_id:
        rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
        return
    if measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return
    if _is_excluded_from_counting(measure_rows_by_id.get(measure_id)):
        rejected.append({"edit": raw_edit, "reason": "measure_excluded_from_counting"})
        return

    measure_overrides.pop(measure_id, None)
    applied.append({"type": "clear_measure_number", "measure_id": measure_id})


def _apply_labels_mode_edit(
    raw_edit: dict,
    labels_mode: str,
    applied: list[dict],
    rejected: list[dict],
) -> str:
    mode = str(raw_edit.get("value") or "").strip().lower()
    if mode not in LABELS_MODE_ALLOWED:
        rejected.append({"edit": raw_edit, "reason": "invalid_value"})
        return labels_mode

    applied.append({"type": "set_labels_mode", "value": mode})
    return mode


def _apply_legacy_rest_staff_edit(
    raw_edit: dict,
    system_ids: set[str],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    system_id = str(raw_edit.get("system_id") or "").strip()
    if not system_id or system_id not in system_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_system_id"})
        return

    measure_count = raw_edit.get("value")
    if not isinstance(measure_count, int) or measure_count < 0:
        rejected.append({"edit": raw_edit, "reason": "invalid_measure_count"})
        return

    rest_systems = _editable_rest_systems(editable_state)
    prev_rest = rest_systems.get(system_id, 0)
    if measure_count == 0:
        rest_systems.pop(system_id, None)
    else:
        rest_systems[system_id] = measure_count

    diff = measure_count - prev_rest
    import sys

    msg1 = f"REST_DEBUG system_id={system_id} measure_count={measure_count} prev_rest={prev_rest} diff={diff}"
    logger.warning(msg1)
    print(msg1, file=sys.stderr, flush=True)
    applied.append({"type": "set_rest_staff", "system_id": system_id, "value": measure_count})


def _apply_measure_rest_edit(
    raw_edit: dict,
    measure_ids: set[str],
    measure_rows_by_id: dict[str, dict],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    measure_id = str(raw_edit.get("measure_id") or "").strip()
    if not measure_id:
        rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
        return
    if measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return
    if _is_excluded_from_counting(measure_rows_by_id.get(measure_id)):
        rejected.append({"edit": raw_edit, "reason": "measure_excluded_from_counting"})
        return

    measure_count = raw_edit.get("value")
    if not isinstance(measure_count, int) or measure_count < 0:
        rejected.append({"edit": raw_edit, "reason": "invalid_measure_count"})
        return

    rest_measures = _editable_rest_measures(editable_state)
    if measure_count == 0:
        rest_measures.pop(measure_id, None)
    else:
        rest_measures[measure_id] = measure_count

    applied.append({"type": "set_rest_measure", "measure_id": measure_id, "value": measure_count})


def _apply_measure_pickup_edit(
    raw_edit: dict,
    measure_ids: set[str],
    measure_rows_by_id: dict[str, dict],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    measure_id = str(raw_edit.get("measure_id") or "").strip()
    if not measure_id:
        rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
        return
    if measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return
    if _is_excluded_from_counting(measure_rows_by_id.get(measure_id)):
        rejected.append({"edit": raw_edit, "reason": "measure_excluded_from_counting"})
        return

    value = raw_edit.get("value")
    if not isinstance(value, bool):
        rejected.append({"edit": raw_edit, "reason": "invalid_value"})
        return

    pickup_measures = _editable_pickup_measures(editable_state)
    if value:
        target_row = measure_rows_by_id.get(measure_id) or {}
        target_system_id = str(target_row.get("system_id") or "").strip()
        if target_system_id:
            to_remove = [
                saved_measure_id
                for saved_measure_id in pickup_measures.keys()
                if str((measure_rows_by_id.get(saved_measure_id) or {}).get("system_id") or "").strip() == target_system_id
            ]
            for saved_measure_id in to_remove:
                pickup_measures.pop(saved_measure_id, None)
        pickup_measures[measure_id] = True
    else:
        pickup_measures.pop(measure_id, None)

    applied.append({"type": "set_pickup_measure", "measure_id": measure_id, "value": value})


def _apply_ending_edit(
    raw_edit: dict,
    measure_ids: set[str],
    measure_rows_by_id: dict[str, dict],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    measure_id = str(raw_edit.get("measure_id") or "").strip()
    ending_val = str(raw_edit.get("value") or "").strip()
    if not measure_id:
        rejected.append({"edit": raw_edit, "reason": "missing_measure_id"})
        return
    if measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "unknown_measure_id"})
        return
    if _is_excluded_from_counting(measure_rows_by_id.get(measure_id)):
        rejected.append({"edit": raw_edit, "reason": "measure_excluded_from_counting"})
        return

    endings = _editable_endings_map(editable_state)
    if ending_val in ("", "none"):
        endings.pop(measure_id, None)
    elif ending_val in ("1", "2"):
        endings[measure_id] = ending_val
    else:
        rejected.append({"edit": raw_edit, "reason": "invalid_ending_value"})
        return

    applied.append({"type": "set_ending", "measure_id": measure_id, "value": ending_val})


def _apply_replace_manual_rows_for_page_edit(
    raw_edit: dict,
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    page = _safe_int(raw_edit.get("page"), 0)
    rows = raw_edit.get("rows")
    cleaned_rows, error_reason = _normalize_manual_rows_payload(page, rows, editable_state)
    if error_reason:
        rejected.append({"edit": raw_edit, "reason": error_reason})
        return

    manual_rows = _editable_manual_rows(editable_state)
    kept_rows = [row for row in manual_rows if _safe_int(row.get("page"), 0) != page]
    kept_rows.extend(cleaned_rows or [])
    kept_rows.sort(
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("top")) or 0.0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("left")) or 0.0),
            str(row.get("manual_row_id") or ""),
        )
    )
    editable_state["manual_rows"] = kept_rows
    applied.append(
        {
            "type": "replace_manual_rows_for_page",
            "page": int(page),
            "rows_count": len(cleaned_rows or []),
        }
    )


def _apply_replace_auto_rows_for_page_edit(
    raw_edit: dict,
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    page = _safe_int(raw_edit.get("page"), 0)
    rows = raw_edit.get("rows")
    cleaned_rows, error_reason = _normalize_auto_rows_payload(page, rows, editable_state)
    if error_reason:
        rejected.append({"edit": raw_edit, "reason": error_reason})
        return

    auto_rows = _editable_auto_rows(editable_state)
    kept_rows = [row for row in auto_rows if _safe_int(row.get("page"), 0) != page]
    kept_rows.extend(cleaned_rows or [])
    kept_rows.sort(
        key=lambda row: (
            _safe_int(row.get("page"), 0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("top")) or 0.0),
            float((((row.get("rect") or {}) if isinstance(row.get("rect"), dict) else {}).get("left")) or 0.0),
            str(row.get("system_id") or ""),
        )
    )
    editable_state["auto_rows"] = kept_rows
    applied.append(
        {
            "type": "replace_auto_rows_for_page",
            "page": int(page),
            "rows_count": len(cleaned_rows or []),
        }
    )


def _apply_remove_label_area_edit(
    raw_edit: dict,
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    area = _normalize_label_erase_area(raw_edit)
    if area is None:
        rejected.append({"edit": raw_edit, "reason": "invalid_label_erase_area"})
        return

    erase_areas = _editable_label_erase_areas(editable_state)
    rect = area["rect"]
    key = (
        f"{area['page']}|{rect['left']:.2f}|{rect['right']:.2f}|"
        f"{rect['top']:.2f}|{rect['bottom']:.2f}"
    )
    existing = {
        (
            f"{saved['page']}|{saved['rect']['left']:.2f}|{saved['rect']['right']:.2f}|"
            f"{saved['rect']['top']:.2f}|{saved['rect']['bottom']:.2f}"
        )
        for saved in erase_areas
        if isinstance(saved, dict) and isinstance(saved.get("rect"), dict)
    }
    if key not in existing:
        erase_areas.append(area)
        editable_state["label_erase_areas"] = erase_areas

    applied.append({"type": "remove_label_area", "page": area["page"], "rect": rect})


def _apply_relabel_edits(
    editable_state: dict,
    edits: list[dict],
    ending_debug_ctx: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict], int]:
    systems = _sorted_system_rows(editable_state.get("systems") or [])
    if not systems:
        raise ValueError("editable_state.systems is missing or empty")
    measures = _sorted_measure_rows(editable_state.get("measures") or [])
    editable_state["systems"] = systems
    editable_state["measures"] = measures

    system_ids = set()
    for row in systems:
        sid = str(row.get("system_id") or "").strip()
        if sid:
            system_ids.add(sid)
    first_measure_by_system: dict[str, dict] = {}
    measure_ids = set()
    measure_rows_by_id: dict[str, dict] = {}
    for measure in measures:
        measure_id = str(measure.get("measure_id") or "").strip()
        if measure_id:
            measure_ids.add(measure_id)
            measure_rows_by_id[measure_id] = measure
        system_id = str(measure.get("system_id") or "").strip()
        if system_id and system_id not in first_measure_by_system:
            first_measure_by_system[system_id] = measure

    applied: list[dict] = []
    rejected: list[dict] = []
    labels_mode = str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY).strip().lower()
    if labels_mode not in LABELS_MODE_ALLOWED:
        labels_mode = LABELS_MODE_SYSTEM_ONLY
    measure_overrides = _measure_number_overrides(editable_state)
    _editable_pickup_measures(editable_state)

    for raw_edit in edits:
        if not isinstance(raw_edit, dict):
            rejected.append({"edit": raw_edit, "reason": "invalid_edit_object"})
            continue
        edit_type = str(raw_edit.get("type") or "").strip()
        if edit_type == "set_system_start":
            _apply_legacy_system_start_edit(
                raw_edit,
                system_ids,
                first_measure_by_system,
                measure_overrides,
                applied,
                rejected,
            )
            continue

        if edit_type == "set_measure_number":
            _apply_measure_number_edit(raw_edit, measure_ids, measure_rows_by_id, measure_overrides, applied, rejected)
            continue

        if edit_type == "clear_measure_number":
            _apply_clear_measure_number_edit(raw_edit, measure_ids, measure_rows_by_id, measure_overrides, applied, rejected)
            continue

        if edit_type == "set_labels_mode":
            labels_mode = _apply_labels_mode_edit(raw_edit, labels_mode, applied, rejected)
            continue

        if edit_type == "set_rest_measure":
            _apply_measure_rest_edit(raw_edit, measure_ids, measure_rows_by_id, editable_state, applied, rejected)
            continue

        if edit_type == "set_pickup_measure":
            _apply_measure_pickup_edit(raw_edit, measure_ids, measure_rows_by_id, editable_state, applied, rejected)
            continue

        if edit_type == "set_rest_staff":
            _apply_legacy_rest_staff_edit(raw_edit, system_ids, editable_state, applied, rejected)
            continue

        if edit_type == "set_ending":
            _apply_ending_edit(raw_edit, measure_ids, measure_rows_by_id, editable_state, applied, rejected)
            continue

        if edit_type == "replace_manual_rows_for_page":
            _apply_replace_manual_rows_for_page_edit(raw_edit, editable_state, applied, rejected)
            continue

        if edit_type == "replace_auto_rows_for_page":
            _apply_replace_auto_rows_for_page_edit(raw_edit, editable_state, applied, rejected)
            continue

        if edit_type == "remove_label_area":
            _apply_remove_label_area_edit(raw_edit, editable_state, applied, rejected)
            continue

        rejected.append({"edit": raw_edit, "reason": "unsupported_edit_type"})

    editable_state["measure_number_overrides"] = measure_overrides
    editable_state["labels_mode"] = labels_mode
    _editable_label_erase_areas(editable_state)
    systems, measures, _, _ = _refresh_editable_state_systems_and_measures(
        editable_state,
        ending_debug_ctx=ending_debug_ctx,
    )
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

    # Manual label erases are intentionally narrow and are applied before
    # current labels are redrawn.
    for area in _editable_label_erase_areas(editable_state):
        page_no = _safe_int(area.get("page"), 0)
        if page_no <= 0 or page_no > doc.page_count:
            continue
        page = doc[page_no - 1]
        _erase_label_area(page, page.rect, area)

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
        # Staff-start mode reuses the same computed measure labels, but only
        # draws the first visible numbered measure on each system.
        for measure, label in _system_start_anchor_measures(ordered_measures, result_labels, sorted_systems):
            page_no = _safe_int(measure.get("page"), 0)
            if page_no <= 0 or page_no > doc.page_count:
                continue
            try:
                x_left = float(measure.get("x_left"))
                y_top = float(measure.get("y_top"))
            except Exception:
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

    systems, measures, reassign_count, qa = _refresh_editable_state_systems_and_measures(editable_state)
    if reassign_count > 0:
        print(f"MEASURE_REASSIGN_SUMMARY job_id={job_id} reassigned={reassign_count}")

    response = {
        "job_id": job_id,
        "run_id": int(run_id),
        "state_version": _editable_state_version(editable_state),
        "editable_state": {
            "version": str(editable_state.get("version") or "system_state_v1"),
            "labels_mode": str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
            "auto_rows": editable_state.get("auto_rows") or [],
            "manual_rows": editable_state.get("manual_rows") or [],
            "rest_measures": editable_state.get("rest_measures") or {},
            "pickup_measures": editable_state.get("pickup_measures") or {},
            "label_erase_areas": _editable_label_erase_areas(editable_state),
            "rest_systems": editable_state.get("rest_systems") or {},
            "qa": qa,
            "systems": systems,
            "measures": measures,
            "staff_boxes": [],
            "measure_number_overrides": editable_state.get("measure_number_overrides") or {},
            "endings": editable_state.get("endings") or {},
        },
        "ai_suggestions": _current_ai_suggestions(mapping_summary),
        "ai_suggest_run": _current_ai_suggest_run(mapping_summary, int(run_id), _editable_state_version(editable_state)),
        "relabel_debug_summary": _summarize_relabel_debug(mapping_summary),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(int(run_id), artifacts),
        "storage_mode": _storage_mode_for_artifacts(artifacts),
    }
    return jsonify(response), 200


@app.route("/api/omr/jobs/<job_id>/ai-suggest", methods=["POST"])
def ai_suggest_job(job_id: str):
    started = time.time()
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": str(err),
                        "retryable": True,
                        "provider_status": 409,
                        "detail": "state_load_failed",
                    },
                }
            ),
            409,
        )

    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)
    try:
        artifacts, mapping_summary, artifact_run_id = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
    except StaleArtifactsError as exc:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(exc.requested_run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "requested job_id does not match single-latest artifacts",
                        "retryable": True,
                        "provider_status": 409,
                        "detail": "stale_run_mismatch",
                    },
                }
            ),
            409,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "failed to load state for AI suggestions",
                        "retryable": True,
                        "provider_status": 502,
                        "detail": _safe_error_text(exc),
                    },
                }
            ),
            502,
        )

    editable_state = mapping_summary.get("editable_state") or {}
    if not isinstance(editable_state, dict):
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "editable_state missing in mapping_summary",
                        "retryable": True,
                        "provider_status": 409,
                        "detail": "editable_state_missing",
                    },
                }
            ),
            409,
        )

    request_payload = request.get_json(silent=True) or {}
    if not isinstance(request_payload, dict):
        request_payload = {}
    raw_score_type = request_payload.get("score_type")
    score_type = _normalize_ai_score_type(raw_score_type)
    if raw_score_type is not None and score_type is None:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "invalid_score_type",
                        "message": "score_type must be single, grand, or score",
                        "retryable": False,
                        "provider_status": 400,
                        "detail": "invalid_score_type",
                    },
                }
            ),
            400,
        )

    systems = editable_state.get("systems")
    if not isinstance(systems, list):
        systems = []
    measures = editable_state.get("measures")
    if not isinstance(measures, list):
        measures = []
    batch_trace_before_rows = _ai_batch_trace_before_snapshot(measures) if _ai_suggest_debug_enabled() else None
    systems, measures, reassign_count, _ = _refresh_editable_state_systems_and_measures(editable_state)
    if reassign_count > 0:
        print(f"MEASURE_REASSIGN_SUMMARY job_id={job_id} reassigned={reassign_count}")

    source_state_version = _editable_state_version(editable_state)
    mapping_summary["editable_state"] = editable_state
    mapping_summary["ai_suggestions"] = _empty_ai_suggestions_state(
        int(artifact_run_id),
        source_state_version,
        len(_ai_suggest_candidate_measures(editable_state)),
    )
    system_batches = _ai_suggest_system_batches(editable_state)
    debug_batch_trace = None
    run_status = AI_SUGGEST_RUN_STATUS_RUNNING if system_batches else AI_SUGGEST_RUN_STATUS_COMPLETED
    mapping_summary["ai_suggest_run"] = _new_ai_suggest_run_state(
        int(artifact_run_id),
        source_state_version,
        len(system_batches),
        status=run_status,
        score_type=score_type,
    )
    try:
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
    except Exception as exc:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "failed to initialize AI suggestion run",
                        "retryable": True,
                        "provider_status": 500,
                        "detail": _safe_error_text(exc),
                    },
                }
            ),
            500,
        )

    if _ai_suggest_debug_enabled():
        try:
            payload = _build_ai_batch_trace_payload(
                job_id,
                int(artifact_run_id),
                systems,
                measures,
                system_batches,
                before_snapshot=batch_trace_before_rows,
                pdf_source=_current_ai_crop_pdf_source_label(artifacts),
            )
            debug_batch_trace = _write_ai_debug_batch_trace(payload, artifacts)
        except Exception as exc:
            logger.warning("AI_BATCH_TRACE_START_WARN job_id=%s detail=%s", job_id, _safe_error_text(exc))

    response = {
        "job_id": job_id,
        "run_id": int(artifact_run_id),
        "status": run_status,
        "ai_suggestions": mapping_summary["ai_suggestions"],
        "ai_suggest_run": mapping_summary["ai_suggest_run"],
        "storage_mode": _storage_mode_for_artifacts(artifacts),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
        "duration_ms": int((time.time() - started) * 1000),
    }
    if isinstance(debug_batch_trace, dict):
        response["debug_batch_trace"] = debug_batch_trace
    if rec and isinstance(rec, dict) and rec.get("pdf_gcs_uri"):
        response["pdf_gcs_uri"] = rec.get("pdf_gcs_uri")
    return jsonify(response), 200


@app.route("/api/omr/jobs/<job_id>/ai-suggest/step", methods=["POST"])
def ai_suggest_job_step(job_id: str):
    started = time.time()
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": str(err),
                        "retryable": True,
                        "provider_status": 409,
                        "detail": "state_load_failed",
                    },
                }
            ),
            409,
        )

    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)
    try:
        artifacts, mapping_summary, artifact_run_id = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
    except StaleArtifactsError as exc:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(exc.requested_run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "requested job_id does not match single-latest artifacts",
                        "retryable": True,
                        "provider_status": 409,
                        "detail": "stale_run_mismatch",
                    },
                }
            ),
            409,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "failed to load state for AI suggestions",
                        "retryable": True,
                        "provider_status": 502,
                        "detail": _safe_error_text(exc),
                    },
                }
            ),
            502,
        )

    editable_state = mapping_summary.get("editable_state") or {}
    if not isinstance(editable_state, dict):
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_failed",
                        "message": "editable_state missing in mapping_summary",
                        "retryable": True,
                        "provider_status": 409,
                        "detail": "editable_state_missing",
                    },
                }
            ),
            409,
        )

    systems = editable_state.get("systems")
    if not isinstance(systems, list):
        systems = []
    measures = editable_state.get("measures")
    if not isinstance(measures, list):
        measures = []
    batch_trace_before_rows = _ai_batch_trace_before_snapshot(measures) if _ai_suggest_debug_enabled() else None
    systems, measures, reassign_count, _ = _refresh_editable_state_systems_and_measures(editable_state)
    if reassign_count > 0:
        print(f"MEASURE_REASSIGN_SUMMARY job_id={job_id} reassigned={reassign_count}")

    source_state_version = _editable_state_version(editable_state)
    mapping_summary["editable_state"] = editable_state
    ai_suggestions = _current_ai_suggestions(mapping_summary)
    if not isinstance(ai_suggestions, dict):
        ai_suggestions = _empty_ai_suggestions_state(
            int(artifact_run_id),
            source_state_version,
            len(_ai_suggest_candidate_measures(editable_state)),
        )
        mapping_summary["ai_suggestions"] = ai_suggestions
    ai_suggest_run = _current_ai_suggest_run(mapping_summary, int(artifact_run_id), source_state_version)
    system_batches = _ai_suggest_system_batches(editable_state)
    systems_total = len(system_batches)
    ai_suggest_run["systems_total"] = systems_total
    debug_batch_trace = None
    debug_batch_trace_payload = None
    if _ai_suggest_debug_enabled():
        try:
            debug_batch_trace_payload = _load_ai_debug_batch_trace(artifacts)
            if not isinstance(debug_batch_trace_payload, dict):
                debug_batch_trace_payload = _build_ai_batch_trace_payload(
                    job_id,
                    int(artifact_run_id),
                    systems,
                    measures,
                    system_batches,
                    before_snapshot=batch_trace_before_rows,
                    pdf_source=_current_ai_crop_pdf_source_label(artifacts),
                )
            debug_batch_trace = _write_ai_debug_batch_trace(debug_batch_trace_payload, artifacts)
        except Exception as exc:
            logger.warning("AI_BATCH_TRACE_STEP_WARN job_id=%s detail=%s", job_id, _safe_error_text(exc))

    if ai_suggest_run.get("status") == AI_SUGGEST_RUN_STATUS_IDLE:
        response = {
            "job_id": job_id,
            "run_id": int(artifact_run_id),
            "status": AI_SUGGEST_RUN_STATUS_IDLE,
            "ai_suggestions": ai_suggestions,
            "ai_suggest_run": ai_suggest_run,
            "error": {
                "code": "ai_suggest_not_started",
                "message": "AI suggestion run has not been started.",
                "retryable": True,
                "provider_status": 409,
                "detail": "ai_suggest_not_started",
            },
        }
        if isinstance(debug_batch_trace, dict):
            response["debug_batch_trace"] = debug_batch_trace
        return jsonify(response), 409

    if ai_suggest_run.get("source_state_version") and ai_suggest_run.get("status") == AI_SUGGEST_RUN_STATUS_RUNNING:
        if str(ai_suggest_run.get("source_state_version") or "") != str(source_state_version or ""):
            now_txt = _utc_now().isoformat().replace("+00:00", "Z")
            error_payload = {
                "code": "ai_suggest_failed",
                "message": "AI suggestion source state changed during generation.",
                "retryable": True,
                "provider_status": 409,
                "detail": "source_state_version_mismatch",
            }
            ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_FAILED
            ai_suggest_run["updated_at_utc"] = now_txt
            ai_suggest_run["failed_at_utc"] = now_txt
            ai_suggest_run["last_error"] = error_payload
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
            return (
                jsonify(
                    {
                        "job_id": job_id,
                        "run_id": int(artifact_run_id),
                        "status": AI_SUGGEST_RUN_STATUS_FAILED,
                        "ai_suggestions": ai_suggestions,
                        "ai_suggest_run": ai_suggest_run,
                        "error": error_payload,
                        "storage_mode": _storage_mode_for_artifacts(artifacts),
                        "artifacts": artifacts,
                        "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
                        "duration_ms": int((time.time() - started) * 1000),
                        **({"debug_batch_trace": debug_batch_trace} if isinstance(debug_batch_trace, dict) else {}),
                    }
                ),
                200,
            )

    if ai_suggest_run.get("status") in {AI_SUGGEST_RUN_STATUS_COMPLETED, AI_SUGGEST_RUN_STATUS_FAILED}:
        response = {
            "job_id": job_id,
            "run_id": int(artifact_run_id),
            "status": str(ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_IDLE),
            "ai_suggestions": ai_suggestions,
            "ai_suggest_run": ai_suggest_run,
            "storage_mode": _storage_mode_for_artifacts(artifacts),
            "artifacts": artifacts,
            "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
            "duration_ms": int((time.time() - started) * 1000),
        }
        if isinstance(debug_batch_trace, dict):
            response["debug_batch_trace"] = debug_batch_trace
        if isinstance(ai_suggest_run.get("last_error"), dict):
            response["error"] = ai_suggest_run.get("last_error")
        return jsonify(response), 200

    next_system_index = max(0, _safe_int(ai_suggest_run.get("next_system_index"), 0))
    if systems_total == 0 or next_system_index >= systems_total:
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_COMPLETED
        ai_suggest_run["updated_at_utc"] = now_txt
        ai_suggest_run["completed_at_utc"] = now_txt
        ai_suggest_run["systems_completed"] = systems_total
        ai_suggest_run["next_system_index"] = systems_total
        ai_suggest_run["last_error"] = None
        mapping_summary["ai_suggest_run"] = ai_suggest_run
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": AI_SUGGEST_RUN_STATUS_COMPLETED,
                    "ai_suggestions": ai_suggestions,
                    "ai_suggest_run": ai_suggest_run,
                    "storage_mode": _storage_mode_for_artifacts(artifacts),
                    "artifacts": artifacts,
                    "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
                    "duration_ms": int((time.time() - started) * 1000),
                    **({"debug_batch_trace": debug_batch_trace} if isinstance(debug_batch_trace, dict) else {}),
                }
            ),
            200,
        )

    system_row, system_measures = system_batches[next_system_index]
    debug_crops = None
    reference_examples_attached = 0
    remembered_time_signature_in = _normalize_ai_time_signature_value(ai_suggest_run.get("remembered_time_signature"))
    score_type = _normalize_ai_score_type(ai_suggest_run.get("score_type"))
    try:
        system_result = _generate_ai_suggestions_for_system_batch(
            job_id,
            int(artifact_run_id),
            systems,
            system_row,
            system_measures,
            source_state_version,
            artifacts,
            remembered_time_signature_in=remembered_time_signature_in,
            score_type=score_type,
        )
        if isinstance(system_result, dict):
            debug_crops = system_result.pop("debug_crops", None)
            reference_examples_attached = _safe_int(system_result.pop("reference_examples_attached", 0), 0)
            current_system_id = str(system_row.get("system_id") or "").strip() or None
            system_result.pop("remembered_time_signature_out", None)
            ai_suggest_run["remembered_time_signature"] = None
            previous_last_time_signature_update = _normalize_ai_time_signature_update_row(
                ai_suggest_run.get("last_time_signature_update"),
            )
            current_time_signature_updates = _normalize_ai_time_signature_update_rows(
                ai_suggest_run.get("time_signature_updates"),
            )
            step_last_time_signature_update = _normalize_ai_time_signature_update_row(
                system_result.pop("last_time_signature_update", None),
                system_id=current_system_id,
            )
            step_time_signature_updates = _normalize_ai_time_signature_update_rows(
                system_result.pop("time_signature_updates", None),
                system_id=current_system_id,
            )
            if step_time_signature_updates:
                current_time_signature_updates.extend(step_time_signature_updates)
                previous_last_time_signature_update = step_time_signature_updates[-1]
            elif step_last_time_signature_update is not None:
                previous_last_time_signature_update = step_last_time_signature_update
            ai_suggest_run["last_time_signature_update"] = previous_last_time_signature_update
            ai_suggest_run["time_signature_updates"] = current_time_signature_updates
        ai_suggestions = _merge_ai_suggestions_state(ai_suggestions, system_result, int(artifact_run_id), source_state_version)
        if isinstance(debug_batch_trace_payload, dict):
            try:
                debug_batch_trace_payload = _mark_ai_batch_trace_processed(
                    debug_batch_trace_payload,
                    system_row,
                    system_measures,
                )
                debug_batch_trace = _write_ai_debug_batch_trace(debug_batch_trace_payload, artifacts)
            except Exception as exc:
                logger.warning("AI_BATCH_TRACE_MARK_WARN job_id=%s detail=%s", job_id, _safe_error_text(exc))
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        completed_count = min(systems_total, next_system_index + 1)
        ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_COMPLETED if completed_count >= systems_total else AI_SUGGEST_RUN_STATUS_RUNNING
        ai_suggest_run["updated_at_utc"] = now_txt
        ai_suggest_run["systems_completed"] = completed_count
        ai_suggest_run["next_system_index"] = completed_count
        ai_suggest_run["last_error"] = None
        ai_suggest_run["failed_at_utc"] = None
        if completed_count >= systems_total:
            ai_suggest_run["completed_at_utc"] = now_txt
        mapping_summary["ai_suggestions"] = ai_suggestions
        mapping_summary["ai_suggest_run"] = ai_suggest_run
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        response = {
            "job_id": job_id,
            "run_id": int(artifact_run_id),
            "status": str(ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_RUNNING),
            "ai_suggestions": ai_suggestions,
            "ai_suggest_run": ai_suggest_run,
            "reference_examples_attached": int(reference_examples_attached),
            "storage_mode": _storage_mode_for_artifacts(artifacts),
            "artifacts": artifacts,
            "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
            "duration_ms": int((time.time() - started) * 1000),
        }
        if isinstance(debug_crops, dict):
            response["debug_crops"] = debug_crops
        if isinstance(debug_batch_trace, dict):
            response["debug_batch_trace"] = debug_batch_trace
        return jsonify(response), 200
    except Exception as exc:
        error_payload = _ai_suggest_error_payload(exc)
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_FAILED
        ai_suggest_run["updated_at_utc"] = now_txt
        ai_suggest_run["failed_at_utc"] = now_txt
        ai_suggest_run["last_error"] = error_payload
        mapping_summary["ai_suggest_run"] = ai_suggest_run
        mapping_summary["ai_suggestions"] = ai_suggestions
        try:
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        except Exception as persist_exc:
            return (
                jsonify(
                    {
                        "job_id": job_id,
                        "run_id": int(artifact_run_id),
                        "status": "failed",
                        "error": {
                            "code": "ai_suggest_failed",
                            "message": "failed to persist AI suggestion failure state",
                            "retryable": True,
                            "provider_status": 500,
                            "detail": _safe_error_text(persist_exc),
                        },
                    }
                ),
                500,
            )
        response = {
            "job_id": job_id,
            "run_id": int(artifact_run_id),
            "status": AI_SUGGEST_RUN_STATUS_FAILED,
            "ai_suggestions": ai_suggestions,
            "ai_suggest_run": ai_suggest_run,
            "error": error_payload,
            "storage_mode": _storage_mode_for_artifacts(artifacts),
            "artifacts": artifacts,
            "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
            "duration_ms": int((time.time() - started) * 1000),
        }
        if isinstance(getattr(exc, "debug_crops", None), dict):
            response["debug_crops"] = getattr(exc, "debug_crops")
        if isinstance(debug_batch_trace, dict):
            response["debug_batch_trace"] = debug_batch_trace
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
    editable_state["systems"] = systems_before
    editable_state["measures"] = measures_before
    systems_before, measures_before, reassign_count, _ = _refresh_editable_state_systems_and_measures(editable_state)
    if reassign_count > 0:
        print(f"MEASURE_REASSIGN_SUMMARY job_id={job_id} reassigned={reassign_count}")

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
    ending_debug_ctx: dict | None = None
    if _relabel_has_ending_debug(editable_state, edits if isinstance(edits, list) else []):
        ending_debug_ctx = {
            "trace_id": trace_id,
            "job_id": job_id,
            "run_id": int(run_id),
        }
        _log_relabel_ending_debug(
            trace_id,
            job_id,
            int(run_id),
            "input",
            {
                "saved_endings_count": len(editable_state.get("endings") or {}) if isinstance(editable_state.get("endings"), dict) else 0,
                "all_edit_types": [
                    str(raw_edit.get("type") or "").strip()
                    for raw_edit in edits
                    if isinstance(raw_edit, dict)
                ],
                "ending_edits": [
                    {
                        "measure_id": str(raw_edit.get("measure_id") or "").strip(),
                        "value": str(raw_edit.get("value") or "").strip(),
                    }
                    for raw_edit in edits
                    if isinstance(raw_edit, dict) and str(raw_edit.get("type") or "").strip() == "set_ending"
                ],
            },
        )

    try:
        baseline_systems = list(editable_state.get("systems") or [])
        baseline_by_id = {
            str(row.get("system_id")): row
            for row in baseline_systems
            if isinstance(row, dict) and str(row.get("system_id") or "").strip()
        }
        systems, applied, rejected, total_systems = _apply_relabel_edits(
            editable_state,
            edits,
            ending_debug_ctx=ending_debug_ctx,
        )
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
    applied_measure_ids = {
        str(row.get("measure_id") or "").strip()
        for row in applied
        if isinstance(row, dict) and str(row.get("measure_id") or "").strip()
    }
    if applied_measure_ids:
        _remove_ai_suggestion_entries(mapping_summary, applied_measure_ids)
    manual_pages_updated = {
        _safe_int(row.get("page"), 0)
        for row in applied
        if isinstance(row, dict) and str(row.get("type") or "").strip() == "replace_manual_rows_for_page"
    }
    if manual_pages_updated:
        manual_measure_ids_to_clear = {
            str(row.get("measure_id") or "").strip()
            for row in list(measures_before) + list(editable_state.get("measures") or [])
            if isinstance(row, dict)
            and _row_source(row) == ROW_SOURCE_MANUAL
            and _safe_int(row.get("page"), 0) in manual_pages_updated
            and str(row.get("measure_id") or "").strip()
        }
        if manual_measure_ids_to_clear:
            _remove_ai_suggestion_entries(mapping_summary, manual_measure_ids_to_clear)
            _clear_measure_state_for_ids(editable_state, manual_measure_ids_to_clear)
    auto_pages_updated = {
        _safe_int(row.get("page"), 0)
        for row in applied
        if isinstance(row, dict) and str(row.get("type") or "").strip() == "replace_auto_rows_for_page"
    }
    if auto_pages_updated:
        auto_measure_ids_before = _measure_ids_on_pages(measures_before, auto_pages_updated, source=ROW_SOURCE_AUTO)
        auto_measure_ids_after = _measure_ids_on_pages(editable_state.get("measures") or [], auto_pages_updated, source=ROW_SOURCE_AUTO)
        excluded_auto_measure_ids = {
            str(row.get("measure_id") or "").strip()
            for row in (editable_state.get("measures") or [])
            if isinstance(row, dict)
            and _row_source(row) == ROW_SOURCE_AUTO
            and _safe_int(row.get("page"), 0) in auto_pages_updated
            and _is_excluded_from_counting(row)
            and str(row.get("measure_id") or "").strip()
        }
        stale_auto_measure_ids = (auto_measure_ids_before | auto_measure_ids_after) | excluded_auto_measure_ids
        if stale_auto_measure_ids:
            _remove_ai_suggestion_entries(mapping_summary, stale_auto_measure_ids)
        removed_auto_measure_ids = (auto_measure_ids_before - auto_measure_ids_after) | excluded_auto_measure_ids
        if removed_auto_measure_ids:
            _clear_measure_state_for_ids(editable_state, removed_auto_measure_ids)
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


@app.route("/api/omr/jobs/<job_id>/ai-suggestions/<measure_id>/dismiss", methods=["POST"])
def dismiss_ai_suggestion(job_id: str, measure_id: str):
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return jsonify({"error": err, "job_id": job_id}), 409
    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)

    try:
        _, mapping_summary, artifact_run_id = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
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

    ai_suggestions = _current_ai_suggestions(mapping_summary)
    target_measure_id = str(measure_id or "").strip()
    if not target_measure_id or not isinstance(ai_suggestions, dict):
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": "failed",
                    "error": {
                        "code": "suggestion_not_found",
                        "message": "AI suggestion not found for measure.",
                        "retryable": False,
                        "detail": target_measure_id or "missing_measure_id",
                    },
                }
            ),
            404,
        )

    by_measure_id = ai_suggestions.get("by_measure_id")
    if not isinstance(by_measure_id, dict) or target_measure_id not in by_measure_id:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": "failed",
                    "error": {
                        "code": "suggestion_not_found",
                        "message": "AI suggestion not found for measure.",
                        "retryable": False,
                        "detail": target_measure_id,
                    },
                }
            ),
            404,
        )

    _remove_ai_suggestion_entries(mapping_summary, {target_measure_id})
    try:
        artifacts = _artifact_uris_for_existing_run(int(artifact_run_id), artifact_key=artifact_key)
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
    except Exception as exc:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": "failed",
                    "error": {
                        "code": "ai_suggestion_dismiss_failed",
                        "message": "failed to dismiss AI suggestion",
                        "retryable": True,
                        "detail": _safe_error_text(exc),
                    },
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "job_id": job_id,
                "run_id": int(artifact_run_id),
                "status": "succeeded",
                "dismissed_measure_id": target_measure_id,
                "ai_suggestions": mapping_summary.get("ai_suggestions"),
            }
        ),
        200,
    )


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
