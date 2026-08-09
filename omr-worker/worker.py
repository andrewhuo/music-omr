import json
import hmac
import logging
import math
import os
import re
import secrets
import time
import threading
import uuid
import hashlib
import base64
from decimal import Decimal, InvalidOperation
from copy import deepcopy
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

_FIRESTORE_IMPORT_ERROR = None
try:
    from google.cloud import firestore
except Exception as exc:
    firestore = None
    _FIRESTORE_IMPORT_ERROR = exc
try:
    from appstoreserverlibrary.models.Environment import Environment as AppleEnvironment
    from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier, VerificationException
except Exception:
    AppleEnvironment = None
    SignedDataVerifier = None
    VerificationException = Exception
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
MANUAL_FIX_BATCH_HISTORY_MAX = 20
CORS_ALLOW_ORIGINS_DEFAULT = "http://localhost:5173,http://localhost:3000"
RUNS_PREFIX = str(os.environ.get("RUNS_PREFIX", "runs") or "runs").strip().strip("/") or "runs"
ENABLE_JOB_STORE = str(os.environ.get("ENABLE_JOB_STORE", "1")).strip().lower() not in ("0", "false", "no")
JOB_STORE_COLLECTION = str(os.environ.get("JOB_STORE_COLLECTION", "omr_jobs") or "omr_jobs").strip() or "omr_jobs"
FRIEND_ACCESS_COLLECTION = str(os.environ.get("FRIEND_ACCESS_COLLECTION", "omr_friend_devices") or "omr_friend_devices").strip() or "omr_friend_devices"
FRIEND_ACCESS_CONFIG_COLLECTION = str(os.environ.get("FRIEND_ACCESS_CONFIG_COLLECTION", "omr_access_config") or "omr_access_config").strip() or "omr_access_config"
FRIEND_ACCESS_CONFIG_DOCUMENT = "friend"
FRIEND_ACCESS_ATTEMPT_COLLECTION = str(os.environ.get("FRIEND_ACCESS_ATTEMPT_COLLECTION", "omr_friend_activation_attempts") or "omr_friend_activation_attempts").strip() or "omr_friend_activation_attempts"
FRIEND_ACCESS_DEFAULT_CREDITS = 500
FRIEND_ACCESS_RESERVATION_TTL_SEC = 15 * 60
FRIEND_ACCESS_HISTORY_MAX = 40
PAID_ACCESS_COLLECTION = str(os.environ.get("PAID_ACCESS_COLLECTION", "omr_paid_access") or "omr_paid_access").strip() or "omr_paid_access"
APPLE_PURCHASE_COLLECTION = str(os.environ.get("APPLE_PURCHASE_COLLECTION", "omr_apple_purchases") or "omr_apple_purchases").strip() or "omr_apple_purchases"
PAID_ACCESS_RESERVATION_TTL_SEC = 15 * 60
APPLE_BUNDLE_ID = str(os.environ.get("APPLE_BUNDLE_ID", "pineapple.Sheet-Music-Labeler") or "").strip()
APPLE_PRO_PRODUCT_ID = str(
    os.environ.get("APPLE_PRO_PRODUCT_ID", "pineapple.sheetmusiclabeler.pro.monthly") or ""
).strip()
APPLE_SUBSCRIPTION_PRODUCTS = {
    APPLE_PRO_PRODUCT_ID: {"plan": "pro", "display_name": "Pro", "credits": 400},
}
PAID_ACCESS_DEFAULT_CREDITS = int(APPLE_SUBSCRIPTION_PRODUCTS[APPLE_PRO_PRODUCT_ID]["credits"])
APPLE_CREDIT_PACKS = {
    "pineapple.sheetmusiclabeler.credits.60": 60,
    "pineapple.sheetmusiclabeler.credits.140": 140,
    "pineapple.sheetmusiclabeler.credits.240": 240,
}
ALLOW_LEGACY_ARTIFACT_FALLBACK = (
    str(os.environ.get("ALLOW_LEGACY_ARTIFACT_FALLBACK", "1")).strip().lower() not in ("0", "false", "no")
)
ANTHROPIC_API_BASE = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com").rstrip("/")
ANTHROPIC_MODEL = str(os.environ.get("ANTHROPIC_MODEL", "") or "").strip()
ANTHROPIC_VERSION = str(os.environ.get("ANTHROPIC_VERSION", "2023-06-01") or "2023-06-01").strip() or "2023-06-01"
ANTHROPIC_TIMEOUT_SEC = max(5.0, float(os.environ.get("ANTHROPIC_TIMEOUT_SEC", "90") or "90"))
ANTHROPIC_MAX_TOKENS = max(256, int(os.environ.get("ANTHROPIC_MAX_TOKENS", "5000") or "5000"))
AI_PROVIDER = str(os.environ.get("AI_PROVIDER", "bedrock") or "bedrock").strip().lower()
AWS_REGION = str(os.environ.get("AWS_REGION", "us-east-1") or "us-east-1").strip()
BEDROCK_MODEL_ID = str(
    os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0") or ""
).strip()
BEDROCK_GENERAL_MODEL_ID = str(
    os.environ.get("BEDROCK_GENERAL_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0") or ""
).strip()
BEDROCK_ENDING_MODEL_ID = str(
    os.environ.get("BEDROCK_ENDING_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0") or ""
).strip()
AI_ENDING_PASS_ENABLED = (
    str(os.environ.get("AI_ENDING_PASS_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes")
)
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
AI_SUGGESTIONS_ENDING_VERSION = "ai_ending_suggestions_v1"
AI_SUGGEST_RUN_VERSION = "ai_suggest_run_v2"
AI_CREDIT_SCHEME_TWO_SYSTEMS_V1 = "two_systems_per_credit_v1"
AI_CREDIT_SCHEME_VERSION = "general_per_system_plus_ending_pair_v2"
AI_COST_SUMMARY_VERSION = "ai_cost_summary_v1"
AI_COST_SUMMARY_KEY = "internal_ai_cost_summary"
AI_SUGGEST_RUN_STATUS_IDLE = "idle"
AI_SUGGEST_RUN_STATUS_RUNNING = "running"
AI_SUGGEST_RUN_STATUS_COMPLETED = "completed"
AI_SUGGEST_RUN_STATUS_FAILED = "failed"
AI_SUGGEST_RUN_STATUS_CANCELLED = "cancelled"
AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED = "partial_failed"
AI_SUGGEST_START_MODES_ALLOWED = {"start", "continue", "restart"}
AI_SUGGESTION_ENDING_LABELS = {"ending_1", "ending_2"}
AI_SUGGESTION_LABELS_ALLOWED = {"normal", "pickup", "multi_measure_rest", "false_measure", "uncertain"}
AI_SUGGESTION_CONFIDENCE_ALLOWED = {"low", "medium", "high"}
AI_SUGGESTION_MAYBE_LABELS_ALLOWED = {"pickup", "multi_measure_rest"}
AI_SUGGESTION_COMPLETENESS_ALLOWED = {"full", "incomplete", "unclear", "not_applicable"}
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
BEDROCK_RETRY_DELAYS_SEC = (0.7, 1.5, 3.0)
TRANSIENT_RETRY_DELAYS_SEC = (0.7, 1.5, 3.0)
AI_REFERENCE_EXAMPLES_DIR = Path(__file__).resolve().parent / "reference_examples"
AI_MULTI_REST_REFERENCE_EXAMPLES = (
    {
        "filename": "old_style_rest_negative_1.png",
        "caption": "Reference A: the printed count is 1, so this is one normal measure, not a multi_measure_rest.",
    },
    {
        "filename": "old_style_rest_positive_3.png",
        "caption": "Reference B: the printed count is 3 above an old-style rest symbol. Return multi_measure_rest with rest_count 3.",
    },
    {
        "filename": "modern_rest_positive_8.png",
        "caption": "Reference C: the printed count is 8 above a modern thick H-bar rest. Return multi_measure_rest with rest_count 8.",
    },
    {
        "filename": "old_style_rest_positive_16.png",
        "caption": "Reference D: the printed count is 16 above an old-style rest symbol. Return multi_measure_rest with rest_count 16.",
    },
)
AI_FALSE_MEASURE_REFERENCE_EXAMPLES = (
    {
        "filename": "false_measure_6_8_only.png",
        "caption": "False-measure reference A: This narrow detected box contains only a 6/8 time signature, staff lines, and barlines. It contains no notes or rests and consumes no musical time. Return false_measure.",
    },
    {
        "filename": "false_measure_common_time_only.png",
        "caption": "False-measure reference B: This narrow detected box contains only a clef, common-time symbol, staff lines, and barlines. It contains no notes or rests and consumes no musical time. Return false_measure.",
    },
)
AI_ENDING_REFERENCE_EXAMPLES = (
    {
        "filename": "ending_1_start_continues.png",
        "caption": "Ending reference A: Ending 1 starts here. The numbered left hook is visible and the horizontal bracket continues right.",
    },
    {
        "filename": "ending_1_starts_and_stops.png",
        "caption": "Ending reference B: Ending 1 starts and stops in this measure. Both the numbered left hook and right downward stop are visible.",
    },
    {
        "filename": "ending_2_start_continues.png",
        "caption": "Ending reference C: Ending 2 starts here. The numbered left hook is visible and the horizontal bracket continues right.",
    },
    {
        "filename": "ending_2_starts_and_stops.png",
        "caption": "Ending reference D: Ending 2 starts and stops in this measure. Both the numbered left hook and right downward stop are visible.",
    },
    {
        "filename": "active_ending_continues.png",
        "caption": "Ending reference E: An already-active ending continues through this measure. There is no new numbered start and no stop.",
    },
    {
        "filename": "active_ending_stops_closed.png",
        "caption": "Ending reference F: An already-active ending stops here with a downward right hook. This is a stop, not a new start.",
    },
    {
        "filename": "active_ending_stops_open.png",
        "caption": "Ending reference G: An already-active ending stops here without a downward hook. The horizontal line simply ends; this is an open stop, not continuation through the score.",
    },
)
AI_ENDING_START_VALUES = {"none", "ending_1", "ending_2", "unsupported", "uncertain"}
AI_ENDING_BOUNDARY_VALUES = {"none", "continues", "closed_stop", "open_stop", "system_edge", "uncertain"}

# In-memory correlation for workflow dispatches that do not return run_id directly.
_PENDING_DISPATCHES: dict[str, dict] = {}
_PENDING_DISPATCHES_LOCK = threading.RLock()
_GCS_CLIENT: storage.Client | None = None
_GOOGLE_CREDENTIALS = None
_JOB_STORE_CLIENT = None
_FRIEND_STORE_CLIENT = None
_APPLE_VERIFIERS: dict[str, object] = {}


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


class PaidAccessError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 403, *, retryable: bool = False):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.retryable = bool(retryable)


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


class FriendAccessError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int, *, retryable: bool = False):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.retryable = bool(retryable)


def _storage_mode_for_artifacts(artifacts: dict[str, str] | None) -> str:
    path = str((artifacts or {}).get("run_info") or "")
    marker = f"/{RUNS_PREFIX}/"
    if marker and marker in path:
        return "per_run_v1"
    return "legacy_single_latest"


def _api_path(path: str | None = None) -> bool:
    txt = str(path or request.path or "").strip()
    return txt.startswith("/api/omr/") or txt.startswith("/api/access/")


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
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-OMR-Paid-Token"
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


def _retry_delay_text(delays: tuple[float, ...] | list[float]) -> str:
    return ",".join(str(int(delay) if float(delay).is_integer() else delay) for delay in delays)


def _status_from_exception(exc: Exception) -> int:
    for attr in ("status_code", "code"):
        raw = getattr(exc, attr, None)
        try:
            status = int(raw)
            if status > 0:
                return status
        except Exception:
            pass
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status)
    except Exception:
        return 0


def _is_retryable_status(status: int) -> bool:
    return int(status or 0) in {408, 429, 500, 502, 503, 504}


def _is_retryable_text(exc: Exception) -> bool:
    text = _safe_error_text(exc).lower()
    retryable_markers = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporarily unavailable",
        "service unavailable",
        "try again",
        "rate limit",
        "too many requests",
        "backend error",
        "internal error",
    )
    return any(marker in text for marker in retryable_markers)


def _sleep_for_retry(delay_sec: float) -> None:
    time.sleep(float(delay_sec))


def _gh_request_once(method: str, path: str, payload: dict | None = None, query: dict | None = None) -> dict | None:
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


def _is_github_retryable_error(exc: GitHubAPIError) -> bool:
    return _is_retryable_status(int(getattr(exc, "status_code", 0) or 0))


def _gh_request(method: str, path: str, payload: dict | None = None, query: dict | None = None) -> dict | None:
    delays = tuple(float(delay) for delay in TRANSIENT_RETRY_DELAYS_SEC if float(delay) > 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            return _gh_request_once(method, path, payload=payload, query=query)
        except GitHubAPIError as exc:
            if not _is_github_retryable_error(exc):
                raise
            if attempt > len(delays):
                exc.message = f"{exc.message} github_retry_attempts={attempt} github_retry_delays_sec={_retry_delay_text(delays)}"
                raise
            delay_sec = delays[attempt - 1]
            logger.warning(
                "GITHUB_RETRY operation=%s path=%s attempt=%s/%s delay=%s status=%s",
                str(method or "").upper(),
                path,
                attempt + 1,
                len(delays) + 1,
                delay_sec,
                int(getattr(exc, "status_code", 0) or 0),
            )
            _sleep_for_retry(delay_sec)


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
        "page_omr_report": f"{out}/artifacts/page_omr_report.json",
    }


def _legacy_artifact_uris_for_run(run_id: int) -> dict[str, str]:
    out = _legacy_output_prefix()
    return {
        "audiveris_out_pdf": f"{out}/audiveris_out.pdf",
        "audiveris_out_corrected_pdf": f"{out}/audiveris_out_corrected.pdf",
        "run_info": f"{out}/artifacts/run_info.json",
        "mapping_summary": f"{out}/artifacts/mapping_summary.json",
        "page_omr_report": f"{out}/artifacts/page_omr_report.json",
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
        if not _with_gcs_retry("signed_url_exists", uri, lambda: blob.exists()):
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


def _is_gcs_retryable_error(exc: Exception) -> bool:
    status = _status_from_exception(exc)
    if status:
        return _is_retryable_status(status)
    return _is_retryable_text(exc)


def _with_gcs_retry(operation: str, uri: str, fn):
    delays = tuple(float(delay) for delay in TRANSIENT_RETRY_DELAYS_SEC if float(delay) > 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            if not _is_gcs_retryable_error(exc):
                raise
            if attempt > len(delays):
                raise
            delay_sec = delays[attempt - 1]
            logger.warning(
                "GCS_RETRY operation=%s attempt=%s/%s delay=%s uri=%s reason=%s",
                operation,
                attempt + 1,
                len(delays) + 1,
                delay_sec,
                uri,
                _safe_error_text(exc),
            )
            _sleep_for_retry(delay_sec)


def _with_job_store_retry(operation: str, job_id: str, fn) -> bool:
    delays = tuple(float(delay) for delay in TRANSIENT_RETRY_DELAYS_SEC if float(delay) > 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            fn()
            return True
        except Exception as exc:
            if not _is_gcs_retryable_error(exc):
                print(f"JOB_STORE_{operation}_WARN job_id={job_id} detail={_safe_error_text(exc)}")
                return False
            if attempt > len(delays):
                print(
                    f"JOB_STORE_{operation}_WARN job_id={job_id} "
                    f"detail={_safe_error_text(exc)} retry_attempts={attempt}"
                )
                return False
            delay_sec = delays[attempt - 1]
            logger.warning(
                "JOB_STORE_RETRY operation=%s job_id=%s attempt=%s/%s delay=%s reason=%s",
                operation,
                job_id,
                attempt + 1,
                len(delays) + 1,
                delay_sec,
                _safe_error_text(exc),
            )
            _sleep_for_retry(delay_sec)


def _job_store_upsert(job_id: str, payload: dict) -> bool:
    client = _job_store_client()
    if client is None:
        return True
    data = dict(payload or {})
    data["job_id"] = str(job_id)
    data["updated_at_utc"] = _to_utc_z(_utc_now())
    return _with_job_store_retry(
        "UPSERT",
        str(job_id),
        lambda: client.collection(JOB_STORE_COLLECTION).document(str(job_id)).set(data, merge=True),
    )


def _job_store_get(job_id: str) -> dict | None:
    client = _job_store_client()
    if client is None:
        return None
    holder: dict[str, dict | None] = {"value": None}

    def _read():
        snap = client.collection(JOB_STORE_COLLECTION).document(str(job_id)).get()
        if not bool(getattr(snap, "exists", False)):
            holder["value"] = None
            return
        data = snap.to_dict()
        if isinstance(data, dict):
            holder["value"] = data
            return
        holder["value"] = None

    if _with_job_store_retry("GET", str(job_id), _read):
        return holder["value"]
    return None


def _friend_access_enforced() -> bool:
    return str(os.environ.get("FRIEND_ACCESS_ENFORCED", "0") or "0").strip().lower() in {"1", "true", "yes"}


def _access_service_error_code(exc: Exception | None) -> str:
    if exc is None:
        return "none"
    if isinstance(exc, ModuleNotFoundError):
        module_name = str(getattr(exc, "name", "") or "").strip()
        return f"missing_module:{module_name}" if module_name else "missing_module"
    if isinstance(exc, NameError):
        missing_name = str(getattr(exc, "name", "") or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", missing_name):
            return f"missing_name:{missing_name}"
        return "missing_name"
    try:
        value = exc.code() if callable(getattr(exc, "code", None)) else getattr(exc, "code", None)
    except Exception:
        value = None
    name = str(getattr(value, "name", "") or "").strip()
    if name:
        return name.lower()
    if value is not None:
        text = re.sub(r"[^A-Za-z0-9_.:\-]", "_", str(value).strip())[:80]
        if text:
            return text
    errno = getattr(exc, "errno", None)
    return f"errno:{errno}" if errno is not None else "none"


def _log_access_store_failure(provider: str, stage: str, exc: Exception | None = None) -> None:
    logger.warning(
        "ACCESS_STORE_FAILURE provider=%s stage=%s error_type=%s service_code=%s",
        str(provider or "shared"),
        str(stage or "unknown"),
        type(exc).__name__ if exc is not None else "Unavailable",
        _access_service_error_code(exc),
    )


def _friend_store_client(provider: str = "shared"):
    global _FRIEND_STORE_CLIENT
    if firestore is None:
        _log_access_store_failure(provider, "library_load", _FIRESTORE_IMPORT_ERROR)
        return None
    if _FRIEND_STORE_CLIENT is None:
        try:
            _FRIEND_STORE_CLIENT = firestore.Client()
        except Exception as exc:
            _log_access_store_failure(provider, "client_connection", exc)
            return None
    return _FRIEND_STORE_CLIENT


def _friend_month_key(now: datetime | None = None) -> str:
    row = now if isinstance(now, datetime) else _utc_now()
    return row.astimezone(timezone.utc).strftime("%Y-%m")


def _friend_parse_time(value) -> datetime | None:
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        parsed = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _friend_history_append(data: dict, action: str, **details) -> list[dict]:
    history = [dict(row) for row in (data.get("admin_history") or []) if isinstance(row, dict)]
    row = {"action": str(action), "at_utc": _to_utc_z(_utc_now())}
    for key, value in details.items():
        if value is not None:
            row[str(key)] = value
    history.append(row)
    return history[-FRIEND_ACCESS_HISTORY_MAX:]


def _friend_run_transaction(client, callback):
    transaction = client.transaction()
    decorator = getattr(firestore, "transactional", None) if firestore is not None else None
    if callable(decorator):
        return decorator(callback)(transaction)
    return callback(transaction)


def _friend_config(client=None) -> dict:
    store = client or _friend_store_client("friend")
    if store is None:
        raise FriendAccessError(
            "friend_access_unavailable",
            "Friend Access is temporarily unavailable. Try again.",
            503,
            retryable=True,
        )
    try:
        snap = store.collection(FRIEND_ACCESS_CONFIG_COLLECTION).document(FRIEND_ACCESS_CONFIG_DOCUMENT).get()
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
    except Exception as exc:
        _log_access_store_failure("friend", "configuration_read", exc)
        raise FriendAccessError(
            "friend_access_unavailable",
            "Friend Access is temporarily unavailable. Try again.",
            503,
            retryable=True,
        ) from exc
    if not isinstance(data, dict):
        raise FriendAccessError("friend_code_not_configured", "Friend Access is not configured yet.", 503, retryable=False)
    return data


def _friend_b64decode(value: str) -> bytes:
    txt = str(value or "").strip()
    if not txt:
        return b""
    padding = "=" * ((4 - len(txt) % 4) % 4)
    try:
        return base64.urlsafe_b64decode((txt + padding).encode("ascii"))
    except Exception:
        return b""


def _friend_code_digest(code: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", str(code).encode("utf-8"), salt, max(100_000, int(iterations)))


def _friend_code_matches(code: str, config: dict) -> bool:
    salt = _friend_b64decode(config.get("code_salt"))
    expected = _friend_b64decode(config.get("code_hash"))
    iterations = max(100_000, _safe_int(config.get("code_iterations"), 210_000))
    if not salt or not expected or not str(code or "").strip():
        return False
    actual = _friend_code_digest(str(code).strip(), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _friend_device_key(device_id: str, config: dict) -> str:
    pepper = _friend_b64decode(config.get("device_pepper"))
    if not pepper:
        raise FriendAccessError("friend_code_not_configured", "Friend Access is not configured yet.", 503)
    return hmac.new(pepper, str(device_id).encode("utf-8"), hashlib.sha256).hexdigest()


def _friend_validate_device_id(device_id: str) -> str:
    txt = str(device_id or "").strip()
    if len(txt) < 16 or len(txt) > 128 or not re.fullmatch(r"[A-Za-z0-9._:\-]+", txt):
        raise FriendAccessError("friend_device_invalid", "This device could not be identified.", 400)
    return txt


def _friend_record_bad_activation(client, device_key: str) -> None:
    ref = client.collection(FRIEND_ACCESS_ATTEMPT_COLLECTION).document(device_key)
    now = _utc_now()

    def _update(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else {}
        data = dict(data) if isinstance(data, dict) else {}
        window_start = _friend_parse_time(data.get("window_started_at_utc"))
        if window_start is None or (now - window_start).total_seconds() >= 15 * 60:
            count = 1
            window_start = now
        else:
            count = max(0, _safe_int(data.get("failed_count"), 0)) + 1
        transaction.set(
            ref,
            {
                "failed_count": count,
                "window_started_at_utc": _to_utc_z(window_start),
                "last_failed_at_utc": _to_utc_z(now),
            },
            merge=True,
        )
        return count

    try:
        count = _friend_run_transaction(client, _update)
    except Exception as exc:
        if isinstance(exc, FriendAccessError):
            raise
        logger.warning("FRIEND_ACCESS_ATTEMPT_WARN detail=%s", _safe_error_text(exc))
        raise FriendAccessError("friend_access_unavailable", "Friend Access is temporarily unavailable. Try again.", 503, retryable=True) from exc
    if int(count) >= 5:
        raise FriendAccessError("friend_code_rate_limited", "Too many code attempts. Wait 15 minutes and try again.", 429, retryable=True)


def _friend_check_activation_rate(client, device_key: str) -> None:
    ref = client.collection(FRIEND_ACCESS_ATTEMPT_COLLECTION).document(device_key)
    try:
        snap = ref.get()
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else {}
    except Exception as exc:
        raise FriendAccessError("friend_access_unavailable", "Friend Access is temporarily unavailable. Try again.", 503, retryable=True) from exc
    if not isinstance(data, dict):
        return
    started = _friend_parse_time(data.get("window_started_at_utc"))
    if started is None or (_utc_now() - started).total_seconds() >= 15 * 60:
        return
    if _safe_int(data.get("failed_count"), 0) >= 5:
        raise FriendAccessError("friend_code_rate_limited", "Too many code attempts. Wait 15 minutes and try again.", 429, retryable=True)


def _friend_clear_activation_attempts(client, device_key: str) -> None:
    try:
        client.collection(FRIEND_ACCESS_ATTEMPT_COLLECTION).document(device_key).delete()
    except Exception as exc:
        logger.warning("FRIEND_ACCESS_ATTEMPT_CLEAR_WARN detail=%s", _safe_error_text(exc))


def _friend_default_credits(config: dict) -> int:
    return max(1, _safe_int(config.get("default_monthly_credits"), FRIEND_ACCESS_DEFAULT_CREDITS))


def _friend_reset_month(data: dict, config: dict, now: datetime | None = None) -> dict:
    result = dict(data or {})
    current_month = _friend_month_key(now)
    if str(result.get("credit_month") or "") != current_month:
        result["credit_month"] = current_month
        result["credits_remaining"] = _friend_default_credits(config)
        result["credits_used"] = 0
        result["reservations"] = {}
        result["admin_history"] = _friend_history_append(result, "monthly_reset", new_balance=result["credits_remaining"])
    return result


def _friend_release_stale_reservations(data: dict, now: datetime | None = None) -> dict:
    result = dict(data or {})
    current = now if isinstance(now, datetime) else _utc_now()
    reservations = dict(result.get("reservations") or {})
    kept: dict[str, dict] = {}
    released = 0
    for reservation_id, row in reservations.items():
        created = _friend_parse_time((row or {}).get("created_at_utc") if isinstance(row, dict) else None)
        if created is None or (current - created).total_seconds() >= FRIEND_ACCESS_RESERVATION_TTL_SEC:
            released += 1
            continue
        kept[str(reservation_id)] = dict(row)
    if released:
        result["credits_remaining"] = max(0, _safe_int(result.get("credits_remaining"), 0)) + released
        result["admin_history"] = _friend_history_append(result, "stale_reservations_released", amount=released)
    result["reservations"] = kept
    return result


def _friend_activate_device(device_id: str, code: str) -> dict:
    device_id = _friend_validate_device_id(device_id)
    client = _friend_store_client("friend")
    if client is None:
        raise FriendAccessError(
            "friend_access_unavailable",
            "Friend Access is temporarily unavailable. Try again.",
            503,
            retryable=True,
        )
    config = _friend_config(client)
    if not bool(config.get("enabled")):
        raise FriendAccessError("friend_code_disabled", "This Friend Code is not currently available.", 403)
    device_key = _friend_device_key(device_id, config)
    _friend_check_activation_rate(client, device_key)
    if not _friend_code_matches(code, config):
        _friend_record_bad_activation(client, device_key)
        raise FriendAccessError("friend_code_invalid", "That Friend Code is not valid.", 403)

    ref = client.collection(FRIEND_ACCESS_COLLECTION).document(device_key)
    token_secret = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token_secret.encode("utf-8")).hexdigest()
    now = _utc_now()

    def _activate(transaction):
        snap = ref.get(transaction=transaction)
        existing = snap.to_dict() if bool(getattr(snap, "exists", False)) else {}
        data = dict(existing) if isinstance(existing, dict) else {}
        if str(data.get("status") or "active").lower() == "banned":
            raise FriendAccessError("friend_access_banned", "Friend Access is unavailable for this device.", 403)
        data = _friend_reset_month(data, config, now)
        data = _friend_release_stale_reservations(data, now)
        friend_id = str(data.get("friend_id") or "").strip() or secrets.token_hex(5).upper()
        joined_at = str(data.get("joined_at_utc") or "").strip() or _to_utc_z(now)
        new_data = {
            **data,
            "friend_id": friend_id,
            "device_key": device_key,
            "token_hash": token_hash,
            "status": "active",
            "joined_at_utc": joined_at,
            "last_seen_at_utc": _to_utc_z(now),
            "updated_at_utc": _to_utc_z(now),
        }
        new_data["admin_history"] = _friend_history_append(new_data, "activated")
        transaction.set(ref, new_data, merge=False)
        return friend_id

    try:
        friend_id = _friend_run_transaction(client, _activate)
    except FriendAccessError:
        raise
    except Exception as exc:
        logger.warning("FRIEND_ACCESS_ACTIVATE_WARN detail=%s", _safe_error_text(exc))
        raise FriendAccessError("friend_access_unavailable", "Friend Access is temporarily unavailable. Try again.", 503, retryable=True) from exc
    _friend_clear_activation_attempts(client, device_key)
    return {"active": True, "friend_id": str(friend_id), "access_token": f"{device_key}.{token_secret}"}


def _friend_bearer_token() -> str:
    try:
        raw = str(request.headers.get("Authorization") or "").strip()
    except Exception:
        raw = ""
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return ""


def _friend_parse_access_token(token: str) -> tuple[str, str]:
    device_key, separator, secret = str(token or "").strip().partition(".")
    if separator != "." or not re.fullmatch(r"[0-9a-f]{64}", device_key) or len(secret) < 32:
        raise FriendAccessError("ai_access_required", "AI Suggestions require Pro or Friend Access.", 403)
    return device_key, secret


def _friend_verify_token(token: str, *, reserve: bool = False, job_id: str | None = None, system_id: str | None = None, charge_id: str | None = None) -> dict:
    device_key, secret = _friend_parse_access_token(token)
    client = _friend_store_client("friend")
    if client is None:
        raise FriendAccessError(
            "friend_access_unavailable",
            "Friend Access is temporarily unavailable. Try again.",
            503,
            retryable=True,
        )
    config = _friend_config(client)
    ref = client.collection(FRIEND_ACCESS_COLLECTION).document(device_key)
    now = _utc_now()
    stable_charge_id = str(charge_id or "").strip() or None
    reservation_id = stable_charge_id if stable_charge_id else (secrets.token_hex(16) if reserve else None)

    def _verify(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
        if not isinstance(data, dict):
            raise FriendAccessError("ai_access_required", "AI Suggestions require Pro or Friend Access.", 403)
        expected = str(data.get("token_hash") or "")
        actual = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        if not expected or not hmac.compare_digest(actual, expected):
            raise FriendAccessError("ai_access_required", "AI Suggestions require Pro or Friend Access.", 403)
        if str(data.get("status") or "").lower() == "banned":
            raise FriendAccessError("friend_access_banned", "Friend Access is unavailable for this device.", 403)
        data = _friend_reset_month(data, config, now)
        data = _friend_release_stale_reservations(data, now)
        charge_receipts = dict(data.get("ai_charge_receipts") or {})
        already_charged = bool(stable_charge_id and stable_charge_id in charge_receipts)
        existing_reservations = dict(data.get("reservations") or {})
        reservation_exists = bool(reservation_id and str(reservation_id) in existing_reservations)
        if reserve:
            reservations = dict(data.get("reservations") or {})
            if not already_charged and str(reservation_id) not in reservations:
                remaining = max(0, _safe_int(data.get("credits_remaining"), 0))
                if remaining <= 0:
                    raise FriendAccessError("ai_credits_exhausted", "Friend AI credits are used up for this month.", 403)
                data["credits_remaining"] = remaining - 1
                reservations[str(reservation_id)] = {
                    "created_at_utc": _to_utc_z(now),
                    "job_id": str(job_id or "") or None,
                    "system_id": str(system_id or "") or None,
                    "charge_id": stable_charge_id,
                }
            data["reservations"] = reservations
        data["last_seen_at_utc"] = _to_utc_z(now)
        data["updated_at_utc"] = _to_utc_z(now)
        transaction.set(ref, data, merge=False)
        return {
            "active": True,
            "friend_id": str(data.get("friend_id") or ""),
            "device_key": device_key,
            "reservation_id": reservation_id,
            "charge_id": stable_charge_id,
            "already_charged": already_charged,
            "reservation_exists": reservation_exists,
            "credits_remaining": max(0, _safe_int(data.get("credits_remaining"), 0)),
            "monthly_credit_capacity": _friend_default_credits(config),
            "expires_at_utc": _to_utc_z(datetime(now.year + (1 if now.month == 12 else 0), 1 if now.month == 12 else now.month + 1, 1, tzinfo=timezone.utc)),
        }

    try:
        return _friend_run_transaction(client, _verify)
    except FriendAccessError:
        raise
    except Exception as exc:
        _log_access_store_failure("friend", "credit_reservation" if reserve else "token_check", exc)
        raise FriendAccessError("friend_access_unavailable", "Friend Access is temporarily unavailable. Try again.", 503, retryable=True) from exc


def _friend_finish_reservation(access: dict | None, *, spent: bool) -> bool:
    if not isinstance(access, dict) or access.get("already_charged") or access.get("bypass"):
        return True
    if not access.get("reservation_id"):
        return False
    client = _friend_store_client()
    if client is None:
        logger.warning("FRIEND_ACCESS_FINISH_WARN reason=store_unavailable")
        return False
    device_key = str(access.get("device_key") or "")
    reservation_id = str(access.get("reservation_id") or "")
    ref = client.collection(FRIEND_ACCESS_COLLECTION).document(device_key)

    def _finish(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
        if not isinstance(data, dict):
            return False
        reservations = dict(data.get("reservations") or {})
        if reservation_id not in reservations:
            return False
        reservation = dict(reservations.get(reservation_id) or {})
        reservations.pop(reservation_id, None)
        data["reservations"] = reservations
        if spent:
            data["credits_used"] = max(0, _safe_int(data.get("credits_used"), 0)) + 1
            charge_id = str(reservation.get("charge_id") or access.get("charge_id") or "").strip()
            if charge_id:
                receipts = dict(data.get("ai_charge_receipts") or {})
                receipts[charge_id] = {"spent_at_utc": _to_utc_z(_utc_now())}
                data["ai_charge_receipts"] = dict(list(receipts.items())[-1000:])
            action = "credit_spent"
        else:
            data["credits_remaining"] = max(0, _safe_int(data.get("credits_remaining"), 0)) + 1
            action = "credit_released"
        data["updated_at_utc"] = _to_utc_z(_utc_now())
        data["admin_history"] = _friend_history_append(data, action, reservation_id=reservation_id)
        transaction.set(ref, data, merge=False)
        return True

    try:
        return bool(_friend_run_transaction(client, _finish))
    except Exception as exc:
        logger.warning("FRIEND_ACCESS_FINISH_WARN detail=%s", _safe_error_text(exc))
        return False


def _friend_ai_access(*, reserve: bool = False, job_id: str | None = None, system_id: str | None = None) -> dict:
    token = _friend_bearer_token()
    if not token and not _friend_access_enforced():
        return {"active": True, "bypass": True, "reservation_id": None}
    if not token:
        raise FriendAccessError("ai_access_required", "AI Suggestions require Pro or Friend Access.", 403)
    try:
        return _friend_verify_token(token, reserve=reserve, job_id=job_id, system_id=system_id)
    except FriendAccessError:
        if not _friend_access_enforced():
            return {"active": True, "bypass": True, "reservation_id": None}
        raise


def _friend_error_response(exc: FriendAccessError):
    return (
        jsonify(
            {
                "status": "failed",
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "provider_status": exc.status_code,
                    "detail": exc.code,
                },
            }
        ),
        exc.status_code,
    )


def _apple_iap_enabled() -> bool:
    return str(os.environ.get("APPLE_IAP_ENABLED", "0") or "0").strip().lower() in {"1", "true", "yes"}


def _apple_packs_enabled() -> bool:
    return str(os.environ.get("APPLE_PACKS_ENABLED", "0") or "0").strip().lower() in {"1", "true", "yes"}


def _apple_text(value) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _apple_millis_time(value) -> datetime | None:
    try:
        millis = int(value)
    except Exception:
        return None
    if millis <= 0:
        return None
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)


def _apple_field(payload, name: str, default=None):
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _apple_root_certificates() -> list[bytes]:
    cert_dir = Path(__file__).resolve().parent / "apple_root_certs"
    rows = []
    for path in sorted(cert_dir.glob("*.cer")):
        try:
            rows.append(path.read_bytes())
        except Exception:
            continue
    if not rows:
        raise PaidAccessError(
            "apple_verification_unavailable",
            "Apple purchase verification is not configured yet.",
            503,
            retryable=True,
        )
    return rows


def _apple_verifier(environment_name: str):
    env_name = str(environment_name or "").strip().lower()
    if env_name not in {"sandbox", "production"}:
        raise PaidAccessError("apple_purchase_invalid", "Apple purchase information is invalid.", 400)
    if SignedDataVerifier is None or AppleEnvironment is None:
        raise PaidAccessError(
            "apple_verification_unavailable",
            "Apple purchase verification is temporarily unavailable.",
            503,
            retryable=True,
        )
    cached = _APPLE_VERIFIERS.get(env_name)
    if cached is not None:
        return cached
    environment = AppleEnvironment.SANDBOX if env_name == "sandbox" else AppleEnvironment.PRODUCTION
    app_apple_id = None
    if env_name == "production":
        app_apple_id = _safe_int(os.environ.get("APPLE_APP_ID"), 0) or None
        if app_apple_id is None:
            raise PaidAccessError(
                "apple_verification_unavailable",
                "Apple purchase verification is not configured yet.",
                503,
                retryable=True,
            )
    verifier = SignedDataVerifier(
        _apple_root_certificates(),
        True,
        environment,
        APPLE_BUNDLE_ID,
        app_apple_id,
    )
    _APPLE_VERIFIERS[env_name] = verifier
    return verifier


def _apple_verify_transaction(signed_transaction: str):
    value = str(signed_transaction or "").strip()
    if len(value) < 100 or value.count(".") != 2:
        raise PaidAccessError("apple_purchase_invalid", "Apple purchase information is invalid.", 400)
    temporary_error = None
    for environment_name in ("production", "sandbox"):
        try:
            return _apple_verifier(environment_name).verify_and_decode_signed_transaction(value)
        except PaidAccessError as exc:
            if exc.code == "apple_verification_unavailable":
                temporary_error = exc
        except VerificationException:
            continue
        except Exception as exc:
            logger.warning("APPLE_TRANSACTION_VERIFY_WARN environment=%s detail=%s", environment_name, _safe_error_text(exc))
            temporary_error = PaidAccessError(
                "apple_verification_unavailable",
                "Apple purchase verification is temporarily unavailable.",
                503,
                retryable=True,
            )
    if temporary_error is not None and SignedDataVerifier is None:
        raise temporary_error
    raise PaidAccessError("apple_purchase_invalid", "Apple could not verify this purchase.", 403)


def _apple_verify_app_transaction(signed_app_transaction: str):
    value = str(signed_app_transaction or "").strip()
    if len(value) < 100 or value.count(".") != 2:
        raise PaidAccessError("apple_purchase_invalid", "Apple app purchase information is invalid.", 400)
    for environment_name in ("production", "sandbox"):
        try:
            return _apple_verifier(environment_name).verify_and_decode_app_transaction(value)
        except (PaidAccessError, VerificationException):
            continue
        except Exception as exc:
            logger.warning("APPLE_APP_TRANSACTION_VERIFY_WARN environment=%s detail=%s", environment_name, _safe_error_text(exc))
    raise PaidAccessError("apple_purchase_invalid", "Apple could not verify this app purchase.", 403)


def _apple_verify_notification(signed_payload: str):
    value = str(signed_payload or "").strip()
    if len(value) < 100 or value.count(".") != 2:
        raise PaidAccessError("apple_notification_invalid", "Apple notification is invalid.", 400)
    for environment_name in ("production", "sandbox"):
        try:
            return _apple_verifier(environment_name).verify_and_decode_notification(value)
        except (PaidAccessError, VerificationException):
            continue
        except Exception as exc:
            logger.warning("APPLE_NOTIFICATION_VERIFY_WARN environment=%s detail=%s", environment_name, _safe_error_text(exc))
    raise PaidAccessError("apple_notification_invalid", "Apple notification could not be verified.", 400)


def _apple_transaction_data(payload, *, allowed_products: set[str]) -> dict:
    product_id = _apple_text(_apple_field(payload, "productId"))
    bundle_id = _apple_text(_apple_field(payload, "bundleId"))
    transaction_id = _apple_text(_apple_field(payload, "transactionId"))
    original_transaction_id = _apple_text(_apple_field(payload, "originalTransactionId"))
    environment_name = _apple_text(_apple_field(payload, "environment") or _apple_field(payload, "receiptType")).lower()
    expires_at = _apple_millis_time(_apple_field(payload, "expiresDate"))
    purchase_at = _apple_millis_time(_apple_field(payload, "purchaseDate"))
    revocation_at = _apple_millis_time(_apple_field(payload, "revocationDate"))
    app_transaction_id = _apple_text(_apple_field(payload, "appTransactionId"))
    if product_id not in allowed_products or bundle_id != APPLE_BUNDLE_ID:
        raise PaidAccessError("apple_purchase_invalid", "This purchase does not belong to this app plan.", 403)
    if not transaction_id or not original_transaction_id or environment_name not in {"sandbox", "production"}:
        raise PaidAccessError("apple_purchase_invalid", "Apple purchase information is incomplete.", 400)
    return {
        "product_id": product_id,
        "bundle_id": bundle_id,
        "transaction_id": transaction_id,
        "original_transaction_id": original_transaction_id,
        "app_transaction_id": app_transaction_id,
        "environment": environment_name,
        "purchase_at": purchase_at,
        "expires_at": expires_at,
        "revocation_at": revocation_at,
    }


def _paid_transaction_data(payload) -> dict:
    return _apple_transaction_data(payload, allowed_products=set(APPLE_SUBSCRIPTION_PRODUCTS))


def _paid_plan_details(product_id: str) -> dict:
    details = APPLE_SUBSCRIPTION_PRODUCTS.get(str(product_id or "").strip())
    if not isinstance(details, dict):
        raise PaidAccessError("apple_purchase_invalid", "This purchase does not belong to this app plan.", 403)
    return {
        "plan": str(details.get("plan") or "pro"),
        "display_name": str(details.get("display_name") or "Pro"),
        "credits": max(0, _safe_int(details.get("credits"), 0)),
    }


def _pack_transaction_data(payload) -> dict:
    return _apple_transaction_data(payload, allowed_products=set(APPLE_CREDIT_PACKS))


def _apple_app_transaction_data(payload) -> dict:
    bundle_id = _apple_text(_apple_field(payload, "bundleId"))
    app_transaction_id = _apple_text(_apple_field(payload, "appTransactionId"))
    environment_name = _apple_text(_apple_field(payload, "receiptType")).lower()
    if bundle_id != APPLE_BUNDLE_ID or not app_transaction_id or environment_name not in {"sandbox", "production"}:
        raise PaidAccessError("apple_purchase_invalid", "Apple app purchase information is incomplete.", 400)
    return {
        "bundle_id": bundle_id,
        "app_transaction_id": app_transaction_id,
        "environment": environment_name,
    }


def _paid_record_key(original_transaction_id: str) -> str:
    return hashlib.sha256(str(original_transaction_id).encode("utf-8")).hexdigest()


def _apple_wallet_key(app_transaction_id: str) -> str:
    return hashlib.sha256(str(app_transaction_id).encode("utf-8")).hexdigest()


def _apple_purchase_key(transaction_id: str) -> str:
    return hashlib.sha256(str(transaction_id).encode("utf-8")).hexdigest()


def _paid_device_key(device_id: str) -> str:
    clean = _friend_validate_device_id(device_id)
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def _paid_release_stale_reservations(data: dict, now: datetime | None = None) -> dict:
    result = dict(data or {})
    current = now if isinstance(now, datetime) else _utc_now()
    reservations = dict(result.get("reservations") or {})
    kept = {}
    released = {"pro": 0, "purchased": 0}
    for reservation_id, row in reservations.items():
        created = _friend_parse_time((row or {}).get("created_at_utc") if isinstance(row, dict) else None)
        if created is None or (current - created).total_seconds() >= PAID_ACCESS_RESERVATION_TTL_SEC:
            source = str((row or {}).get("source") or "pro")
            released[source if source in released else "pro"] += 1
        else:
            kept[str(reservation_id)] = dict(row)
    if released["pro"]:
        result["subscription_credits_remaining"] = max(0, _safe_int(result.get("subscription_credits_remaining", result.get("credits_remaining")), 0)) + released["pro"]
        result["credits_remaining"] = result["subscription_credits_remaining"]
    if released["purchased"]:
        result["purchased_credits_remaining"] = max(0, _safe_int(result.get("purchased_credits_remaining"), 0)) + released["purchased"]
    result["reservations"] = kept
    return result


def _paid_is_active(data: dict, now: datetime | None = None) -> bool:
    current = now if isinstance(now, datetime) else _utc_now()
    status = str(data.get("status") or "").strip().lower()
    if status in {"refunded", "revoked", "expired", "inactive"}:
        return False
    expires_at = _friend_parse_time(data.get("expires_at_utc"))
    grace_at = _friend_parse_time(data.get("grace_expires_at_utc"))
    if expires_at is not None and current < expires_at:
        return True
    return bool(status == "billing_grace" and grace_at is not None and current < grace_at)


def _paid_apply_transaction(payload, *, app_transaction_id: str | None = None, device_id: str | None = None, issue_token: bool = False) -> dict:
    transaction_data = _paid_transaction_data(payload)
    plan = _paid_plan_details(transaction_data["product_id"])
    client = _friend_store_client()
    if client is None:
        raise PaidAccessError("paid_access_unavailable", "Paid Access is temporarily unavailable.", 503, retryable=True)
    wallet_identity = str(app_transaction_id or transaction_data.get("app_transaction_id") or transaction_data["original_transaction_id"])
    record_key = _apple_wallet_key(wallet_identity)
    ref = client.collection(PAID_ACCESS_COLLECTION).document(record_key)
    now = _utc_now()
    active = transaction_data["revocation_at"] is None and (
        transaction_data["expires_at"] is None or now < transaction_data["expires_at"]
    )
    device_key = _paid_device_key(device_id) if issue_token else None
    token_secret = secrets.token_urlsafe(32) if issue_token else None

    def _apply(transaction):
        snap = ref.get(transaction=transaction)
        existing = snap.to_dict() if bool(getattr(snap, "exists", False)) else {}
        data = dict(existing) if isinstance(existing, dict) else {}
        processed = [str(row) for row in (data.get("processed_transactions") or []) if str(row)]
        is_new_period = transaction_data["transaction_id"] not in processed
        if is_new_period:
            processed.append(transaction_data["transaction_id"])
            data["subscription_credits_remaining"] = plan["credits"] if active else 0
            data["subscription_credits_used"] = 0
            data["credits_remaining"] = data["subscription_credits_remaining"]
            data["credits_used"] = 0
            data["credit_period_transaction_id"] = transaction_data["transaction_id"]
        data.update(
            {
                "product_id": transaction_data["product_id"],
                "plan": plan["plan"],
                "plan_display_name": plan["display_name"],
                "monthly_credit_capacity": plan["credits"],
                "bundle_id": transaction_data["bundle_id"],
                "environment": transaction_data["environment"],
                "original_transaction_id": transaction_data["original_transaction_id"],
                "current_transaction_id": transaction_data["transaction_id"],
                "app_wallet_key": record_key,
                "purchase_at_utc": _to_utc_z(transaction_data["purchase_at"]) if transaction_data["purchase_at"] else None,
                "expires_at_utc": _to_utc_z(transaction_data["expires_at"]) if transaction_data["expires_at"] else None,
                "status": "active" if active else ("revoked" if transaction_data["revocation_at"] else "expired"),
                "processed_transactions": processed[-120:],
                "updated_at_utc": _to_utc_z(now),
            }
        )
        if not data.get("created_at_utc"):
            data["created_at_utc"] = _to_utc_z(now)
        if device_key and token_secret:
            device_tokens = dict(data.get("device_tokens") or {})
            device_tokens[device_key] = {
                "token_hash": hashlib.sha256(token_secret.encode("utf-8")).hexdigest(),
                "issued_at_utc": _to_utc_z(now),
                "last_seen_at_utc": _to_utc_z(now),
            }
            data["device_tokens"] = device_tokens
        transaction.set(ref, data, merge=False)
        return data, is_new_period

    try:
        saved, is_new_period = _friend_run_transaction(client, _apply)
    except PaidAccessError:
        raise
    except Exception as exc:
        logger.warning("PAID_ACCESS_SAVE_WARN detail=%s", _safe_error_text(exc))
        raise PaidAccessError("paid_access_unavailable", "Paid Access is temporarily unavailable.", 503, retryable=True) from exc
    return {
        "active": bool(active),
        "paid_id": record_key[:10].upper(),
        "plan": str(saved.get("plan") or plan["plan"]),
        "plan_display_name": str(saved.get("plan_display_name") or plan["display_name"]),
        "monthly_credit_capacity": max(0, _safe_int(saved.get("monthly_credit_capacity"), plan["credits"])) if active else 0,
        "status": str(saved.get("status") or ("active" if active else "inactive")),
        "credits_remaining": max(0, _safe_int(saved.get("credits_remaining", saved.get("subscription_credits_remaining")), 0)) if active else 0,
        "purchased_credits_remaining": max(0, _safe_int(saved.get("purchased_credits_remaining"), 0)),
        "expires_at_utc": saved.get("expires_at_utc"),
        "new_period": bool(is_new_period),
        "access_token": f"{record_key}.{device_key}.{token_secret}" if issue_token else None,
    }


def _paid_header_token() -> str:
    try:
        return str(request.headers.get("X-OMR-Paid-Token") or "").strip()
    except Exception:
        return ""


def _paid_issue_device_token(data: dict, *, record_key: str, device_id: str, now: datetime) -> tuple[dict, str]:
    device_key = _paid_device_key(device_id)
    token_secret = secrets.token_urlsafe(32)
    device_tokens = dict(data.get("device_tokens") or {})
    device_tokens[device_key] = {
        "token_hash": hashlib.sha256(token_secret.encode("utf-8")).hexdigest(),
        "issued_at_utc": _to_utc_z(now),
        "last_seen_at_utc": _to_utc_z(now),
    }
    data["device_tokens"] = device_tokens
    return data, f"{record_key}.{device_key}.{token_secret}"


def _pack_apply_transaction(payload, *, app_transaction_id: str, device_id: str) -> dict:
    purchase = _pack_transaction_data(payload)
    if purchase.get("revocation_at") is not None:
        raise PaidAccessError("apple_purchase_refunded", "This credit purchase was refunded.", 403)
    client = _friend_store_client()
    if client is None:
        raise PaidAccessError("paid_access_unavailable", "Credit purchases are temporarily unavailable.", 503, retryable=True)
    record_key = _apple_wallet_key(app_transaction_id)
    purchase_key = _apple_purchase_key(purchase["transaction_id"])
    wallet_ref = client.collection(PAID_ACCESS_COLLECTION).document(record_key)
    purchase_ref = client.collection(APPLE_PURCHASE_COLLECTION).document(purchase_key)
    now = _utc_now()
    granted = int(APPLE_CREDIT_PACKS[purchase["product_id"]])
    token_box = {"value": None}

    def _apply(transaction):
        receipt_snap = purchase_ref.get(transaction=transaction)
        receipt = receipt_snap.to_dict() if bool(getattr(receipt_snap, "exists", False)) else None
        wallet_snap = wallet_ref.get(transaction=transaction)
        wallet = wallet_snap.to_dict() if bool(getattr(wallet_snap, "exists", False)) else {}
        wallet = dict(wallet) if isinstance(wallet, dict) else {}
        is_new = not isinstance(receipt, dict)
        usable_added = 0
        debt_paid = 0
        if is_new:
            debt = max(0, _safe_int(wallet.get("purchased_credit_debt"), 0))
            debt_paid = min(debt, granted)
            usable_added = granted - debt_paid
            wallet["purchased_credit_debt"] = debt - debt_paid
            wallet["purchased_credits_remaining"] = max(0, _safe_int(wallet.get("purchased_credits_remaining"), 0)) + usable_added
            transaction.set(
                purchase_ref,
                {
                    "wallet_key": record_key,
                    "product_id": purchase["product_id"],
                    "credits_granted": granted,
                    "transaction_id_hash": purchase_key,
                    "environment": purchase["environment"],
                    "status": "granted",
                    "granted_at_utc": _to_utc_z(now),
                },
                merge=False,
            )
        wallet.update({
            "app_wallet_key": record_key,
            "bundle_id": purchase["bundle_id"],
            "environment": purchase["environment"],
            "updated_at_utc": _to_utc_z(now),
        })
        if not wallet.get("created_at_utc"):
            wallet["created_at_utc"] = _to_utc_z(now)
        wallet, token_box["value"] = _paid_issue_device_token(wallet, record_key=record_key, device_id=device_id, now=now)
        transaction.set(wallet_ref, wallet, merge=False)
        return wallet, is_new, usable_added, debt_paid

    try:
        wallet, is_new, usable_added, debt_paid = _friend_run_transaction(client, _apply)
    except PaidAccessError:
        raise
    except Exception as exc:
        logger.warning("APPLE_PACK_SAVE_WARN detail=%s", _safe_error_text(exc))
        raise PaidAccessError("paid_access_unavailable", "Credit purchases are temporarily unavailable.", 503, retryable=True) from exc
    return {
        "paid_id": record_key[:10].upper(),
        "access_token": token_box["value"],
        "new_purchase": bool(is_new),
        "credits_granted": granted if is_new else 0,
        "usable_credits_added": usable_added,
        "debt_repaid": debt_paid,
        "purchased_credits_remaining": max(0, _safe_int(wallet.get("purchased_credits_remaining"), 0)),
        "purchased_credit_debt": max(0, _safe_int(wallet.get("purchased_credit_debt"), 0)),
    }


def _pack_refund_transaction(payload) -> dict:
    purchase = _pack_transaction_data(payload)
    client = _friend_store_client()
    if client is None:
        raise PaidAccessError("paid_access_unavailable", "Credit refunds are temporarily unavailable.", 503, retryable=True)
    purchase_key = _apple_purchase_key(purchase["transaction_id"])
    purchase_ref = client.collection(APPLE_PURCHASE_COLLECTION).document(purchase_key)
    now = _utc_now()

    def _refund(transaction):
        receipt_snap = purchase_ref.get(transaction=transaction)
        receipt = receipt_snap.to_dict() if bool(getattr(receipt_snap, "exists", False)) else None
        if not isinstance(receipt, dict) or str(receipt.get("status")) == "refunded":
            return {"changed": False}
        wallet_key = str(receipt.get("wallet_key") or "")
        wallet_ref = client.collection(PAID_ACCESS_COLLECTION).document(wallet_key)
        wallet_snap = wallet_ref.get(transaction=transaction)
        wallet = wallet_snap.to_dict() if bool(getattr(wallet_snap, "exists", False)) else {}
        wallet = dict(wallet) if isinstance(wallet, dict) else {}
        amount = max(0, _safe_int(receipt.get("credits_granted"), 0))
        available = max(0, _safe_int(wallet.get("purchased_credits_remaining"), 0))
        wallet["purchased_credits_remaining"] = max(0, available - amount)
        wallet["purchased_credit_debt"] = max(0, _safe_int(wallet.get("purchased_credit_debt"), 0)) + max(0, amount - available)
        wallet["updated_at_utc"] = _to_utc_z(now)
        receipt["status"] = "refunded"
        receipt["refunded_at_utc"] = _to_utc_z(now)
        transaction.set(wallet_ref, wallet, merge=False)
        transaction.set(purchase_ref, receipt, merge=False)
        return {"changed": True, "wallet_key": wallet_key}

    return _friend_run_transaction(client, _refund)


def _paid_parse_access_token(token: str) -> tuple[str, str, str]:
    pieces = str(token or "").strip().split(".")
    if len(pieces) != 3 or not all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in pieces[:2]) or len(pieces[2]) < 32:
        raise PaidAccessError("paid_access_required", "AI Suggestions require Pro or Friend Access.", 403)
    return pieces[0], pieces[1], pieces[2]


def _paid_verify_token(token: str, *, reserve: bool = False, source: str | None = None, allow_empty: bool = False, job_id: str | None = None, system_id: str | None = None, charge_id: str | None = None) -> dict:
    record_key, device_key, secret = _paid_parse_access_token(token)
    client = _friend_store_client("paid")
    if client is None:
        raise PaidAccessError("paid_access_unavailable", "Paid Access is temporarily unavailable.", 503, retryable=True)
    ref = client.collection(PAID_ACCESS_COLLECTION).document(record_key)
    now = _utc_now()
    stable_charge_id = str(charge_id or "").strip() or None
    reservation_id = stable_charge_id if stable_charge_id else (secrets.token_hex(16) if reserve else None)

    def _verify(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
        if not isinstance(data, dict):
            raise PaidAccessError("paid_access_required", "AI Suggestions require Pro or Friend Access.", 403)
        device = (data.get("device_tokens") or {}).get(device_key) or {}
        expected = str(device.get("token_hash") or "")
        actual = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        if not expected or not hmac.compare_digest(expected, actual):
            raise PaidAccessError("paid_access_required", "AI Suggestions require Pro or Friend Access.", 403)
        data = _paid_release_stale_reservations(data, now)
        charge_receipts = dict(data.get("ai_charge_receipts") or {})
        already_charged = bool(stable_charge_id and stable_charge_id in charge_receipts)
        existing_reservations = dict(data.get("reservations") or {})
        reservation_exists = bool(reservation_id and str(reservation_id) in existing_reservations)
        pro_active = _paid_is_active(data, now)
        stored_plan = APPLE_SUBSCRIPTION_PRODUCTS.get(str(data.get("product_id") or "")) or {}
        plan_name = str(data.get("plan") or stored_plan.get("plan") or "pro")
        plan_display_name = str(data.get("plan_display_name") or stored_plan.get("display_name") or "Pro")
        monthly_capacity = max(
            0,
            _safe_int(data.get("monthly_credit_capacity"), stored_plan.get("credits", PAID_ACCESS_DEFAULT_CREDITS)),
        ) if pro_active else 0
        legacy_remaining = data.get("credits_remaining")
        pro_remaining = max(0, _safe_int(legacy_remaining if legacy_remaining is not None else data.get("subscription_credits_remaining"), 0)) if pro_active else 0
        purchased_remaining = max(0, _safe_int(data.get("purchased_credits_remaining"), 0))
        if pro_remaining <= 0 and purchased_remaining <= 0 and not allow_empty:
            if pro_active:
                raise PaidAccessError("paid_credits_exhausted", "Paid AI credits are used up for this billing period.", 403)
            raise PaidAccessError("paid_subscription_inactive", "No paid AI credits are available.", 403)
        selected_source = str(source or ("pro" if pro_remaining > 0 else "purchased"))
        if reserve and not already_charged and selected_source == "pro" and pro_remaining <= 0:
            raise PaidAccessError("paid_credits_exhausted", "Paid AI credits are used up for this billing period.", 403)
        if reserve and not already_charged and selected_source == "purchased" and purchased_remaining <= 0:
            raise PaidAccessError("paid_credits_exhausted", "Purchased AI credits are used up.", 403)
        if reserve and not already_charged:
            reservations = dict(data.get("reservations") or {})
            if str(reservation_id) in reservations:
                already_charged = False
            else:
                if selected_source == "pro":
                    data["subscription_credits_remaining"] = pro_remaining - 1
                    data["credits_remaining"] = pro_remaining - 1
                    pro_remaining -= 1
                else:
                    data["purchased_credits_remaining"] = purchased_remaining - 1
                    purchased_remaining -= 1
                reservations[str(reservation_id)] = {
                    "created_at_utc": _to_utc_z(now),
                    "job_id": str(job_id or "") or None,
                    "system_id": str(system_id or "") or None,
                    "source": selected_source,
                    "charge_id": stable_charge_id,
                }
                data["reservations"] = reservations
        device_tokens = dict(data.get("device_tokens") or {})
        device["last_seen_at_utc"] = _to_utc_z(now)
        device_tokens[device_key] = device
        data["device_tokens"] = device_tokens
        data["updated_at_utc"] = _to_utc_z(now)
        transaction.set(ref, data, merge=False)
        return {
            "active": True,
            "provider": "paid",
            "record_key": record_key,
            "reservation_id": reservation_id,
            "charge_id": stable_charge_id,
            "already_charged": already_charged,
            "reservation_exists": reservation_exists,
            "paid_id": record_key[:10].upper(),
            "plan": plan_name if pro_active else None,
            "plan_display_name": plan_display_name if pro_active else None,
            "monthly_credit_capacity": monthly_capacity,
            "subscription_status": str(data.get("status") or ("active" if pro_active else "inactive")),
            "source": selected_source,
            "credits_remaining": pro_remaining,
            "pro_credits_remaining": pro_remaining,
            "purchased_credits_remaining": purchased_remaining,
            "purchased_credit_debt": max(0, _safe_int(data.get("purchased_credit_debt"), 0)),
            "pro_active": pro_active,
            "expires_at_utc": data.get("expires_at_utc"),
        }

    try:
        return _friend_run_transaction(client, _verify)
    except PaidAccessError:
        raise
    except Exception as exc:
        _log_access_store_failure("paid", "credit_reservation" if reserve else "token_check", exc)
        raise PaidAccessError("paid_access_unavailable", "Paid Access is temporarily unavailable.", 503, retryable=True) from exc


def _paid_finish_reservation(access: dict | None, *, spent: bool) -> bool:
    if not isinstance(access, dict) or access.get("already_charged"):
        return True
    if access.get("provider") != "paid" or not access.get("reservation_id"):
        return False
    client = _friend_store_client()
    if client is None:
        logger.warning("PAID_ACCESS_FINISH_WARN reason=store_unavailable")
        return False
    record_key = str(access.get("record_key") or "")
    reservation_id = str(access.get("reservation_id") or "")
    ref = client.collection(PAID_ACCESS_COLLECTION).document(record_key)

    def _finish(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
        if not isinstance(data, dict):
            return False
        reservations = dict(data.get("reservations") or {})
        if reservation_id not in reservations:
            return False
        reservation = dict(reservations.get(reservation_id) or {})
        reservations.pop(reservation_id, None)
        data["reservations"] = reservations
        source = str(reservation.get("source") or access.get("source") or "pro")
        if spent:
            key = "subscription_credits_used" if source == "pro" else "purchased_credits_used"
            data[key] = max(0, _safe_int(data.get(key), 0)) + 1
            if source == "pro":
                data["credits_used"] = data[key]
            charge_id = str(reservation.get("charge_id") or access.get("charge_id") or "").strip()
            if charge_id:
                receipts = dict(data.get("ai_charge_receipts") or {})
                receipts[charge_id] = {"spent_at_utc": _to_utc_z(_utc_now()), "source": source}
                data["ai_charge_receipts"] = dict(list(receipts.items())[-1000:])
        elif source == "purchased":
            data["purchased_credits_remaining"] = max(0, _safe_int(data.get("purchased_credits_remaining"), 0)) + 1
        else:
            data["subscription_credits_remaining"] = max(0, _safe_int(data.get("subscription_credits_remaining", data.get("credits_remaining")), 0)) + 1
            data["credits_remaining"] = data["subscription_credits_remaining"]
        data["updated_at_utc"] = _to_utc_z(_utc_now())
        transaction.set(ref, data, merge=False)
        return True

    try:
        return bool(_friend_run_transaction(client, _finish))
    except Exception as exc:
        logger.warning("PAID_ACCESS_FINISH_WARN detail=%s", _safe_error_text(exc))
        return False


def _ai_access(*, reserve: bool = False, job_id: str | None = None, system_id: str | None = None, charge_id: str | None = None) -> dict:
    paid_token = _paid_header_token()
    friend_token = _friend_bearer_token()
    if not paid_token and not friend_token:
        friend = _friend_ai_access(reserve=reserve, job_id=job_id, system_id=system_id)
        if isinstance(friend, dict):
            friend["provider"] = "friend"
        return friend

    candidates = []
    access_errors = []
    retryable_errors = []
    verified_sources = 0
    if friend_token:
        try:
            friend = _friend_verify_token(friend_token, charge_id=charge_id)
            verified_sources += 1
            if charge_id and (friend.get("already_charged") or friend.get("reservation_exists")):
                friend["provider"] = "friend"
                return friend
            if max(0, _safe_int(friend.get("credits_remaining"), 0)) > 0:
                candidates.append((str(friend.get("expires_at_utc") or "9999"), 0, "friend"))
        except FriendAccessError as exc:
            if exc.retryable:
                retryable_errors.append(exc)
            else:
                access_errors.append(exc)
    if paid_token:
        try:
            paid = _paid_verify_token(paid_token, allow_empty=True, charge_id=charge_id)
            verified_sources += 1
            if charge_id and (paid.get("already_charged") or paid.get("reservation_exists")):
                return paid
            if paid.get("pro_active") and max(0, _safe_int(paid.get("pro_credits_remaining"), 0)) > 0:
                candidates.append((str(paid.get("expires_at_utc") or "9999"), 1, "pro"))
            if _apple_packs_enabled() and max(0, _safe_int(paid.get("purchased_credits_remaining"), 0)) > 0:
                candidates.append(("9999", 2, "purchased"))
        except PaidAccessError as exc:
            if exc.retryable:
                retryable_errors.append(exc)
            else:
                access_errors.append(exc)
    if not candidates:
        if retryable_errors:
            raise retryable_errors[0]
        if verified_sources:
            raise PaidAccessError("paid_credits_exhausted", "No AI credits are available.", 403)
        if access_errors:
            raise access_errors[0]
        raise PaidAccessError("paid_credits_exhausted", "No AI credits are available.", 403)
    candidates.sort(key=lambda row: (row[0], row[1]))
    selected = candidates[0][2]
    if not reserve:
        return {"active": True, "provider": selected}
    if selected == "friend":
        result = _friend_verify_token(friend_token, reserve=True, job_id=job_id, system_id=system_id, charge_id=charge_id)
        result["provider"] = "friend"
        return result
    return _paid_verify_token(paid_token, reserve=True, source=selected, job_id=job_id, system_id=system_id, charge_id=charge_id)


def _finish_ai_access(access: dict | None, *, spent: bool) -> bool:
    if isinstance(access, dict) and access.get("provider") == "paid":
        return _paid_finish_reservation(access, spent=spent)
    else:
        friend_access = dict(access) if isinstance(access, dict) else access
        if isinstance(friend_access, dict):
            friend_access.pop("provider", None)
        return _friend_finish_reservation(friend_access, spent=spent)
    return False


def _paid_error_response(exc: PaidAccessError):
    return (
        jsonify(
            {
                "status": "failed",
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "provider_status": exc.status_code,
                    "detail": exc.code,
                },
            }
        ),
        exc.status_code,
    )


def _verified_apple_identity(signed_app_transaction: str, transaction_payload=None) -> dict:
    app_data = _apple_app_transaction_data(_apple_verify_app_transaction(signed_app_transaction))
    if transaction_payload is not None:
        transaction_app_id = _apple_text(_apple_field(transaction_payload, "appTransactionId"))
        transaction_environment = _apple_text(_apple_field(transaction_payload, "environment")).lower()
        if transaction_app_id and transaction_app_id != app_data["app_transaction_id"]:
            raise PaidAccessError("apple_purchase_invalid", "Apple purchase wallet does not match.", 403)
        if transaction_environment and transaction_environment != app_data["environment"]:
            raise PaidAccessError("apple_purchase_invalid", "Apple purchase environment does not match.", 403)
    return app_data


def _paid_restore_wallet(*, app_transaction_id: str, device_id: str) -> dict:
    client = _friend_store_client()
    if client is None:
        raise PaidAccessError("paid_access_unavailable", "Paid Access is temporarily unavailable.", 503, retryable=True)
    record_key = _apple_wallet_key(app_transaction_id)
    ref = client.collection(PAID_ACCESS_COLLECTION).document(record_key)
    now = _utc_now()
    token_box = {"value": None}

    def _restore(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
        if not isinstance(data, dict):
            raise PaidAccessError("paid_access_required", "No Apple credit wallet was found.", 404)
        data, token_box["value"] = _paid_issue_device_token(data, record_key=record_key, device_id=device_id, now=now)
        data["updated_at_utc"] = _to_utc_z(now)
        transaction.set(ref, data, merge=False)
        return data

    saved = _friend_run_transaction(client, _restore)
    subscription_active = _paid_is_active(saved, now)
    stored_plan = APPLE_SUBSCRIPTION_PRODUCTS.get(str(saved.get("product_id") or "")) or {}
    monthly_capacity = max(
        0,
        _safe_int(saved.get("monthly_credit_capacity"), stored_plan.get("credits", PAID_ACCESS_DEFAULT_CREDITS)),
    ) if subscription_active else 0
    return {
        "paid_id": record_key[:10].upper(),
        "access_token": token_box["value"],
        "pro_active": subscription_active,
        "plan": str(saved.get("plan") or stored_plan.get("plan") or "pro") if subscription_active else None,
        "plan_display_name": str(saved.get("plan_display_name") or stored_plan.get("display_name") or "Pro") if subscription_active else None,
        "monthly_credit_capacity": monthly_capacity,
        "subscription_status": str(saved.get("status") or ("active" if subscription_active else "inactive")),
        "pro_credits_remaining": max(0, _safe_int(saved.get("subscription_credits_remaining", saved.get("credits_remaining")), 0)) if subscription_active else 0,
        "purchased_credits_remaining": max(0, _safe_int(saved.get("purchased_credits_remaining"), 0)),
        "purchased_credit_debt": max(0, _safe_int(saved.get("purchased_credit_debt"), 0)),
        "expires_at_utc": saved.get("expires_at_utc"),
    }


def _combined_credit_status() -> dict:
    friend_result = None
    paid_result = None
    friend_token = _friend_bearer_token()
    paid_token = _paid_header_token()
    friend_check = "absent"
    paid_check = "absent"
    retryable_errors = []
    if friend_token:
        try:
            friend_result = _friend_verify_token(friend_token)
            friend_check = "verified"
        except FriendAccessError as exc:
            if exc.retryable:
                friend_check = "temporarily_unavailable"
                retryable_errors.append(exc)
            else:
                friend_check = "banned" if exc.code == "friend_access_banned" else "invalid"
    if paid_token:
        try:
            paid_result = _paid_verify_token(paid_token, allow_empty=True)
            paid_check = "verified"
        except PaidAccessError as exc:
            if exc.retryable:
                paid_check = "temporarily_unavailable"
                retryable_errors.append(exc)
            else:
                paid_check = "invalid"
    if not friend_result and not paid_result and retryable_errors:
        raise retryable_errors[0]
    friend_credits = max(0, _safe_int((friend_result or {}).get("credits_remaining"), 0))
    pro_credits = max(0, _safe_int((paid_result or {}).get("pro_credits_remaining"), 0))
    purchased_credits = (
        max(0, _safe_int((paid_result or {}).get("purchased_credits_remaining"), 0))
        if _apple_packs_enabled()
        else 0
    )
    friend_capacity = max(0, _safe_int((friend_result or {}).get("monthly_credit_capacity"), 0))
    paid_capacity = max(0, _safe_int((paid_result or {}).get("monthly_credit_capacity"), 0))
    return {
        "friend_credits": friend_credits,
        "pro_credits": pro_credits,
        "purchased_credits": purchased_credits,
        "total_credits": friend_credits + pro_credits + purchased_credits,
        "friend_monthly_capacity": friend_capacity,
        "paid_monthly_capacity": paid_capacity,
        "total_monthly_capacity": friend_capacity + paid_capacity,
        "friend_active": bool(friend_result),
        "pro_active": bool((paid_result or {}).get("pro_active")),
        "paid_plan": (paid_result or {}).get("plan"),
        "paid_plan_display_name": (paid_result or {}).get("plan_display_name"),
        "paid_subscription_status": (paid_result or {}).get("subscription_status"),
        "paid_wallet_active": bool(paid_result and (pro_credits + purchased_credits) > 0),
        "friend_id": (friend_result or {}).get("friend_id"),
        "paid_id": (paid_result or {}).get("paid_id"),
        "pro_expires_at_utc": (paid_result or {}).get("expires_at_utc"),
        "purchased_credit_debt": max(0, _safe_int((paid_result or {}).get("purchased_credit_debt"), 0)),
        "friend_check": friend_check,
        "paid_check": paid_check,
        "partial": any(value in {"invalid", "banned", "temporarily_unavailable"} for value in (friend_check, paid_check)),
    }


def _paid_update_notification_state(payload, *, notification_type: str, grace_expires_at: datetime | None = None) -> dict:
    data = _paid_transaction_data(payload)
    client = _friend_store_client()
    if client is None:
        raise PaidAccessError("paid_access_unavailable", "Paid Access is temporarily unavailable.", 503, retryable=True)
    record_key = _apple_wallet_key(data.get("app_transaction_id") or data["original_transaction_id"])
    ref = client.collection(PAID_ACCESS_COLLECTION).document(record_key)
    event = str(notification_type or "").strip().upper()
    status = None
    if event in {"REFUND", "REVOKE"}:
        status = "refunded" if event == "REFUND" else "revoked"
    elif event in {"EXPIRED", "GRACE_PERIOD_EXPIRED"}:
        status = "expired"
    elif event in {"DID_FAIL_TO_RENEW"}:
        status = "billing_grace" if grace_expires_at is not None and _utc_now() < grace_expires_at else "billing_retry"

    def _update(transaction):
        snap = ref.get(transaction=transaction)
        row = snap.to_dict() if bool(getattr(snap, "exists", False)) else None
        if not isinstance(row, dict):
            return False
        if status:
            row["status"] = status
        if grace_expires_at is not None:
            row["grace_expires_at_utc"] = _to_utc_z(grace_expires_at)
        row["last_notification_type"] = event
        row["updated_at_utc"] = _to_utc_z(_utc_now())
        transaction.set(ref, row, merge=False)
        return True

    _friend_run_transaction(client, _update)
    return {"paid_id": record_key[:10].upper(), "status": status or "unchanged"}


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


def _ensure_unique_job_id(base_job_id: str, *, allow_same: bool = False) -> str:
    base = _normalize_artifact_key(base_job_id)[:96] or str(uuid.uuid4())
    if allow_same:
        return base
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
    return bool(_with_gcs_retry("exists", uri, lambda: blob.exists()))


def _download_gcs_json(uri: str) -> dict:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    raw = _with_gcs_retry("download_json", uri, lambda: blob.download_as_bytes())
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return data


def _download_gcs_to_file(uri: str, dest_path: Path) -> None:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _with_gcs_retry("download_file", uri, lambda: blob.download_to_filename(str(dest_path)))


def _upload_file_to_gcs(src_path: Path, dest_uri: str, content_type: str | None = None) -> None:
    if not Path(src_path).is_file():
        raise FileNotFoundError(str(src_path))
    bucket_name, blob_name = _parse_gs_uri(dest_uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    _with_gcs_retry("upload_file", dest_uri, lambda: blob.upload_from_filename(str(src_path), content_type=content_type))


def _upload_bytes_to_gcs(data: bytes, dest_uri: str, content_type: str | None = None) -> None:
    bucket_name, blob_name = _parse_gs_uri(dest_uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    _with_gcs_retry(
        "upload_bytes",
        dest_uri,
        lambda: blob.upload_from_string(data, content_type=content_type or "application/octet-stream"),
    )


def _upload_json_to_gcs(data: dict, dest_uri: str) -> None:
    bucket_name, blob_name = _parse_gs_uri(dest_uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    _with_gcs_retry(
        "upload_json",
        dest_uri,
        lambda: blob.upload_from_string(payload, content_type="application/json"),
    )


def _delete_gcs_uri_if_exists(uri: str) -> tuple[bool, bool]:
    bucket_name, blob_name = _parse_gs_uri(uri)
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not bool(_with_gcs_retry("exists", uri, lambda: blob.exists())):
        return False, False
    _with_gcs_retry("delete", uri, lambda: blob.delete())
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


def _label_rect_payload(rect) -> dict:
    return {
        "left": round(float(rect.x0), 3),
        "top": round(float(rect.y0), 3),
        "right": round(float(rect.x1), 3),
        "bottom": round(float(rect.y1), 3),
    }


def _measure_label_layout_left_barline(page: fitz.Page, page_rect: fitz.Rect, x_left: float, y_top: float, text: str) -> dict | None:
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
    if x1 <= x0 or y1 <= y0:
        return None
    bg_rect = fitz.Rect(x0, y0, x1, y1)
    return {
        "text_point": (float(x_text), float(y_text)),
        "text_width": float(tw),
        "text_height": float(th),
        "background_rect": bg_rect,
        "rect": _label_rect_payload(bg_rect),
    }


def _measure_label_layout_at_top_left(page_rect: fitz.Rect, left: float, top: float, text: str) -> dict | None:
    tw = float(fitz.get_text_length(text, fontsize=MEASURE_TEXT_SIZE))
    th = float(MEASURE_TEXT_SIZE + 2.0)
    x_text = min(max(1.0, float(left) + 1.0), max(1.0, float(page_rect.width) - tw - 1.0))
    y_bg_top = min(max(0.0, float(top)), max(0.0, float(page_rect.height) - th))
    y_text = y_bg_top + th - 1.0
    bg = fitz.Rect(x_text - 1.0, y_text - th + 1.0, x_text + tw + 1.0, y_text + 1.0)
    x0 = max(0.0, min(bg.x0, page_rect.width))
    y0 = max(0.0, min(bg.y0, page_rect.height))
    x1 = max(0.0, min(bg.x1, page_rect.width))
    y1 = max(0.0, min(bg.y1, page_rect.height))
    if x1 <= x0 or y1 <= y0:
        return None
    bg_rect = fitz.Rect(x0, y0, x1, y1)
    return {
        "text_point": (float(x_text), float(y_text)),
        "text_width": float(tw),
        "text_height": float(th),
        "background_rect": bg_rect,
        "rect": _label_rect_payload(bg_rect),
    }


def _draw_measure_label_layout(page: fitz.Page, layout: dict, text: str) -> None:
    bg_rect = layout.get("background_rect")
    if bg_rect is not None:
        page.draw_rect(bg_rect, color=MEASURE_TEXT_BG_COLOR, fill=MEASURE_TEXT_BG_COLOR)
    x_text, y_text = layout.get("text_point") or (0.0, 0.0)
    page.insert_text((x_text, y_text), text, fontsize=MEASURE_TEXT_SIZE, color=MEASURE_TEXT_COLOR)


def _label_box_from_layout(measure: dict, label: str, layout: dict, hidden: bool = False) -> dict | None:
    measure_id = str(measure.get("measure_id") or "").strip()
    page_no = _safe_int(measure.get("page"), 0)
    rect = layout.get("rect")
    if not measure_id or page_no <= 0 or not isinstance(rect, dict):
        return None
    return {
        "label_id": f"label:{measure_id}",
        "measure_id": measure_id,
        "page": page_no,
        "text": str(label),
        "rect": rect,
        "hidden": bool(hidden),
    }


def _editable_label_boxes(editable_state: dict) -> list[dict]:
    raw_boxes = editable_state.get("label_boxes")
    if isinstance(raw_boxes, list):
        return [box for box in raw_boxes if isinstance(box, dict)]
    editable_state["label_boxes"] = []
    return editable_state["label_boxes"]


def _editable_hidden_label_ids(editable_state: dict) -> list[str]:
    raw_ids = editable_state.get("hidden_label_ids")
    if not isinstance(raw_ids, list):
        editable_state["hidden_label_ids"] = []
        return editable_state["hidden_label_ids"]
    cleaned = sorted(
        {
            str(raw_id).strip()
            for raw_id in raw_ids
            if str(raw_id).strip().startswith("label:")
        }
    )
    editable_state["hidden_label_ids"] = cleaned
    return editable_state["hidden_label_ids"]


def _editable_forced_label_ids(editable_state: dict, valid_measure_ids: set[str] | None = None) -> list[str]:
    raw_ids = editable_state.get("forced_label_ids")
    if not isinstance(raw_ids, list):
        editable_state["forced_label_ids"] = []
        return editable_state["forced_label_ids"]
    cleaned = sorted(
        {
            str(raw_id).strip()
            for raw_id in raw_ids
            if str(raw_id).strip().startswith("label:")
            and (
                valid_measure_ids is None
                or str(raw_id).strip()[len("label:") :] in valid_measure_ids
            )
        }
    )
    editable_state["forced_label_ids"] = cleaned
    return editable_state["forced_label_ids"]


def _editable_label_positions(editable_state: dict) -> dict:
    raw_positions = editable_state.get("label_positions")
    if not isinstance(raw_positions, dict):
        editable_state["label_positions"] = {}
        return editable_state["label_positions"]
    cleaned = {}
    for raw_label_id, raw_position in raw_positions.items():
        label_id = str(raw_label_id or "").strip()
        if not label_id.startswith("label:") or not isinstance(raw_position, dict):
            continue
        try:
            left = float(raw_position.get("left"))
            top = float(raw_position.get("top"))
        except Exception:
            continue
        if not math.isfinite(left) or not math.isfinite(top):
            continue
        page = _safe_int(raw_position.get("page"), 0)
        saved = {"left": round(left, 3), "top": round(top, 3)}
        if page > 0:
            saved["page"] = page
        cleaned[label_id] = saved
    editable_state["label_positions"] = cleaned
    return editable_state["label_positions"]


def _editable_state_version(editable_state: dict) -> str:
    payload = {
        "version": editable_state.get("version"),
        "labels_mode": str(editable_state.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
        "systems": editable_state.get("systems") or [],
        "measures": editable_state.get("measures") or [],
        "auto_rows": editable_state.get("auto_rows") or [],
        "auto_rows_authoritative_pages": editable_state.get("auto_rows_authoritative_pages") or [],
        "manual_rows": editable_state.get("manual_rows") or [],
        "measure_number_overrides": editable_state.get("measure_number_overrides") or {},
        "rest_measures": editable_state.get("rest_measures") or {},
        "pickup_measures": editable_state.get("pickup_measures") or {},
        "rest_systems": editable_state.get("rest_systems") or {},
        "endings": editable_state.get("endings") or {},
        "label_erase_areas": editable_state.get("label_erase_areas") or [],
        "hidden_label_ids": editable_state.get("hidden_label_ids") or [],
        "forced_label_ids": editable_state.get("forced_label_ids") or [],
        "label_positions": editable_state.get("label_positions") or {},
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


def _manual_row_measure_ids(manual_row: dict, measure_count: int) -> list[str]:
    manual_row_id = _normalize_artifact_key(manual_row.get("manual_row_id"))[:64]
    raw_ids = manual_row.get("measure_ids")
    if isinstance(raw_ids, list) and len(raw_ids) == measure_count:
        cleaned = [_normalize_artifact_key(value)[:128] for value in raw_ids]
        if all(cleaned) and len(set(cleaned)) == len(cleaned):
            return cleaned
    return [_manual_measure_id(manual_row_id, index) for index in range(measure_count)]


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

        measure_ids = _manual_row_measure_ids(raw_row, len(cut_xs) + 1)
        source_manual_row_id = _normalize_artifact_key(
            raw_row.get("source_manual_row_id") or manual_row_id
        )[:64]
        if not source_manual_row_id:
            source_manual_row_id = manual_row_id

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
                "measure_ids": measure_ids,
                "source_manual_row_id": source_manual_row_id,
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
            "source_system_id": str(system_row.get("source_system_id") or system_id).strip(),
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


def _editable_auto_rows_authoritative_pages(editable_state: dict) -> list[int]:
    raw_pages = editable_state.get("auto_rows_authoritative_pages")
    pages = sorted({
        page
        for page in (_safe_int(value, 0) for value in (raw_pages if isinstance(raw_pages, list) else []))
        if page > 0
    })
    editable_state["auto_rows_authoritative_pages"] = pages
    return pages


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
            "source_system_id": _normalize_artifact_key(raw_row.get("source_system_id") or system_id)[:128],
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

    auto_rects = [
        rect
        for rect in (
            _parse_manual_row_rect(row.get("rect"))
            for row in _editable_auto_rows(editable_state)
            if _safe_int(row.get("page"), 0) == page
        )
        if rect is not None
    ]

    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    seen_measure_ids: set[str] = set()
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

        measure_ids = _manual_row_measure_ids(raw_row, len(cut_xs) + 1)
        if len(measure_ids) != len(cut_xs) + 1 or any(measure_id in seen_measure_ids for measure_id in measure_ids):
            return None, "invalid_manual_measure_ids"
        source_manual_row_id = _normalize_artifact_key(
            raw_row.get("source_manual_row_id") or manual_row_id
        )[:64]
        if not source_manual_row_id:
            return None, "invalid_source_manual_row_id"

        rect = (float(left), float(right), float(top), float(bottom))
        if any(_rects_strongly_overlap(rect, auto_rect) for auto_rect in auto_rects):
            return None, "manual_row_overlap_auto"
        if any(_rects_strongly_overlap(rect, prior_rect) for prior_rect in seen_rects):
            return None, "manual_row_overlap_manual"

        seen_ids.add(manual_row_id)
        seen_measure_ids.update(measure_ids)
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
                "measure_ids": measure_ids,
                "source_manual_row_id": source_manual_row_id,
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
        source_system_id = str(auto_row.get("source_system_id") or system_id).strip()
        page = _safe_int(auto_row.get("page"), 0)
        rect = (auto_row.get("rect") or {}) if isinstance(auto_row.get("rect"), dict) else {}
        left = _safe_float(rect.get("left"), 0.0)
        right = _safe_float(rect.get("right"), left)
        top = _safe_float(rect.get("top"), 0.0)
        bottom = _safe_float(rect.get("bottom"), top)
        boxes = auto_row.get("boxes")
        if not system_id or page <= 0 or right <= left or bottom <= top or not isinstance(boxes, list):
            continue
        existing_system = existing_systems_by_id.get(system_id) or existing_systems_by_id.get(source_system_id) or {}
        system_index = _safe_int(existing_system.get("system_index"), 0)
        current_value = str(auto_row.get("current_value") or existing_system.get("current_value") or existing_system.get("value") or "").strip()
        staff_kind = str(auto_row.get("staff_kind") or existing_system.get("staff_kind") or "").strip().lower()
        systems.append(
            {
                "system_id": system_id,
                "source_system_id": source_system_id,
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
                    "source_system_id": source_system_id,
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
    expected_source_system_ids = {
        str(row.get("source_system_id") or row.get("system_id") or "").strip()
        for row in current_auto_rows
        if isinstance(row, dict) and str(row.get("source_system_id") or row.get("system_id") or "").strip()
    }
    if not isinstance(rows, list):
        return None, "invalid_auto_rows"
    if not expected_source_system_ids and rows:
        return None, "unexpected_auto_rows"

    cleaned: list[dict] = []
    seen_system_ids: set[str] = set()
    seen_measure_ids: set[str] = set()
    current_by_system_id = {}
    current_by_source_system_id = {}
    for row in current_auto_rows:
        if not isinstance(row, dict):
            continue
        current_system_id = str(row.get("system_id") or "").strip()
        source_system_id = str(row.get("source_system_id") or current_system_id).strip()
        if current_system_id:
            current_by_system_id[current_system_id] = row
        if source_system_id and source_system_id not in current_by_source_system_id:
            current_by_source_system_id[source_system_id] = row

    for raw_row in rows:
        if not isinstance(raw_row, dict):
            return None, "invalid_auto_row"
        system_id = _normalize_artifact_key(raw_row.get("system_id"))[:128]
        source_system_id = _normalize_artifact_key(raw_row.get("source_system_id") or system_id)[:128]
        if not system_id or system_id in seen_system_ids:
            return None, "duplicate_auto_system_id"
        if not source_system_id or source_system_id not in expected_source_system_ids:
            return None, "unknown_auto_source_system_id"
        if _safe_int(raw_row.get("page"), 0) != page:
            return None, "auto_row_page_mismatch"
        rect_tuple = _parse_manual_row_rect(raw_row.get("rect"))
        if rect_tuple is None:
            return None, "invalid_auto_row_rect"
        left, right, top, bottom = rect_tuple
        current_row = current_by_system_id.get(system_id) or current_by_source_system_id.get(source_system_id) or {}
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
            "source_system_id": source_system_id,
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
        source_manual_row_id = str(manual_row.get("source_manual_row_id") or manual_row_id).strip()
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
                "source_manual_row_id": source_manual_row_id,
                "staff_kind": staff_kind,
                "anchor": {"x": float(left), "y_top": float(top), "y_bottom": float(bottom)},
                "x_left": float(left),
                "x_right": float(right),
                "y_top": float(top),
                "y_bottom": float(bottom),
            }
        )
        boundaries = [float(left), *[float(cut) for cut in (manual_row.get("cut_xs") or [])], float(right)]
        measure_ids = _manual_row_measure_ids(manual_row, len(boundaries) - 1)
        for idx in range(len(boundaries) - 1):
            measures.append(
                {
                    "measure_id": measure_ids[idx],
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
                    "source_manual_row_id": source_manual_row_id,
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
            if _row_source(measure) == ROW_SOURCE_MANUAL and not str(measure.get("measure_id") or "").strip():
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
    authoritative_pages = set(_editable_auto_rows_authoritative_pages(editable_state))
    replaced_system_ids = {
        str(row.get("system_id") or "").strip()
        for row in overlay_auto_systems
        if isinstance(row, dict) and str(row.get("system_id") or "").strip()
    }
    auto_systems = [
        row for row in base_auto_systems
        if _safe_int(row.get("page"), 0) not in authoritative_pages
        and str(row.get("system_id") or "").strip() not in replaced_system_ids
    ]
    auto_systems.extend(overlay_auto_systems)
    auto_measures = [
        row for row in base_auto_measures
        if _safe_int(row.get("page"), 0) not in authoritative_pages
        and str(row.get("system_id") or "").strip() not in replaced_system_ids
    ]
    auto_measures.extend(overlay_auto_measures)
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


def _manual_fix_batch_receipts(mapping_summary: dict) -> list[dict]:
    rows = mapping_summary.get("manual_fix_batches")
    if not isinstance(rows, list):
        rows = []
    cleaned = [row for row in rows if isinstance(row, dict) and str(row.get("request_id") or "").strip()]
    mapping_summary["manual_fix_batches"] = cleaned[-MANUAL_FIX_BATCH_HISTORY_MAX:]
    return mapping_summary["manual_fix_batches"]


def _find_manual_fix_batch_receipt(mapping_summary: dict, request_id: str) -> dict | None:
    clean_id = str(request_id or "").strip()
    if not clean_id:
        return None
    for row in reversed(_manual_fix_batch_receipts(mapping_summary)):
        if str(row.get("request_id") or "").strip() == clean_id:
            return row
    return None


def _append_manual_fix_batch_receipt(mapping_summary: dict, receipt: dict) -> None:
    request_id = str(receipt.get("request_id") or "").strip()
    if not request_id:
        return
    rows = [
        row
        for row in _manual_fix_batch_receipts(mapping_summary)
        if str(row.get("request_id") or "").strip() != request_id
    ]
    rows.append(deepcopy(receipt))
    mapping_summary["manual_fix_batches"] = rows[-MANUAL_FIX_BATCH_HISTORY_MAX:]


def _atomic_relabel_rejections(rejected: list[dict], editable_state: dict) -> list[dict]:
    measure_pages = {
        str(row.get("measure_id") or "").strip(): _safe_int(row.get("page"), 0)
        for row in (editable_state.get("measures") or [])
        if isinstance(row, dict) and str(row.get("measure_id") or "").strip()
    }
    output: list[dict] = []
    for row in rejected:
        if not isinstance(row, dict):
            continue
        raw_edit = row.get("edit") if isinstance(row.get("edit"), dict) else {}
        page = _safe_int(raw_edit.get("page"), 0)
        if page <= 0:
            measure_id = str(raw_edit.get("measure_id") or "").strip()
            page = measure_pages.get(measure_id, 0)
        item = {"reason": str(row.get("reason") or "edit_rejected").strip() or "edit_rejected"}
        if page > 0:
            item["page"] = page
        output.append(item)
    return output


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


def _configured_bedrock_general_model_id() -> str:
    explicit = str(os.environ.get("BEDROCK_GENERAL_MODEL_ID", "") or "").strip()
    if explicit:
        return explicit
    legacy = str(os.environ.get("BEDROCK_MODEL_ID", "") or "").strip()
    return legacy or BEDROCK_GENERAL_MODEL_ID


def _configured_bedrock_ending_model_id() -> str:
    explicit = str(os.environ.get("BEDROCK_ENDING_MODEL_ID", "") or "").strip()
    if explicit:
        return explicit
    legacy = str(os.environ.get("BEDROCK_MODEL_ID", "") or "").strip()
    return legacy or BEDROCK_ENDING_MODEL_ID


def _ending_pass_enabled() -> bool:
    return str(os.environ.get("AI_ENDING_PASS_ENABLED", "1" if AI_ENDING_PASS_ENABLED else "0") or "0").strip().lower() in ("1", "true", "yes")


def _requested_ai_model_name(pass_kind: str = "general") -> str:
    provider = _requested_ai_provider_name()
    if provider == "bedrock":
        if str(pass_kind or "").strip().lower() == "ending":
            return _configured_bedrock_ending_model_id() or _configured_bedrock_model_id() or "unknown"
        if str(pass_kind or "").strip().lower() == "general":
            return _configured_bedrock_general_model_id() or _configured_bedrock_model_id() or "unknown"
        return _configured_bedrock_model_id() or "unknown"
    if provider == "anthropic":
        return _configured_anthropic_model_name() or "unknown"
    return "unknown"


def _ai_cost_rate_snapshot() -> dict:
    def _rate_value(name: str) -> Decimal | None:
        raw = str(os.environ.get(name, "") or "").strip()
        if not raw:
            return None
        try:
            value = Decimal(raw)
        except (InvalidOperation, ValueError):
            return None
        return value if value >= 0 else None

    input_rate = _rate_value("BEDROCK_COST_INPUT_USD_PER_MILLION")
    output_rate = _rate_value("BEDROCK_COST_OUTPUT_USD_PER_MILLION")
    rate_version = str(os.environ.get("BEDROCK_COST_RATE_VERSION", "") or "").strip() or None
    return {
        "rate_version": rate_version,
        "input_usd_per_million": str(input_rate) if input_rate is not None else None,
        "output_usd_per_million": str(output_rate) if output_rate is not None else None,
        "available": input_rate is not None and output_rate is not None and rate_version is not None,
    }


def _safe_ai_usage_tokens(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _ai_usage_from_message(message: dict | None) -> dict | None:
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        return None
    input_tokens = _safe_ai_usage_tokens(usage.get("input_tokens"))
    output_tokens = _safe_ai_usage_tokens(usage.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retry_attempts": max(1, _safe_int(message.get("_internal_bedrock_attempts"), 1)),
    }


def _ai_cost_decimal(summary: dict) -> Decimal | None:
    rate = summary.get("rate") if isinstance(summary.get("rate"), dict) else {}
    if not rate.get("available"):
        return None
    try:
        input_rate = Decimal(str(rate.get("input_usd_per_million")))
        output_rate = Decimal(str(rate.get("output_usd_per_million")))
    except (InvalidOperation, ValueError):
        return None
    input_tokens = max(0, _safe_int(summary.get("input_tokens_total"), 0))
    output_tokens = max(0, _safe_int(summary.get("output_tokens_total"), 0))
    return (
        (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate)
        / Decimal(1_000_000)
    )


def _append_internal_ai_cost_usage(
    mapping_summary: dict,
    *,
    job_id: str,
    run_id: int,
    system_row: dict,
    model: str,
    usage: dict | None,
    pass_kind: str = "general",
    charge_id: str | None = None,
) -> dict:
    existing = mapping_summary.get(AI_COST_SUMMARY_KEY)
    summary = dict(existing) if isinstance(existing, dict) else {}
    history = [dict(row) for row in (summary.get("invocations") or []) if isinstance(row, dict)]
    rate = summary.get("rate") if isinstance(summary.get("rate"), dict) else _ai_cost_rate_snapshot()
    now_txt = _utc_now().isoformat().replace("+00:00", "Z")
    system_id = str(system_row.get("system_id") or "").strip() or None
    page = _safe_int(system_row.get("page"), 0) or None
    retry_attempts = max(1, _safe_int((usage or {}).get("retry_attempts"), 1))
    input_tokens = _safe_ai_usage_tokens((usage or {}).get("input_tokens"))
    output_tokens = _safe_ai_usage_tokens((usage or {}).get("output_tokens"))
    usage_available = input_tokens is not None and output_tokens is not None
    history.append(
        {
            "job_id": str(job_id),
            "run_id": int(run_id),
            "page": page,
            "system_id": system_id,
            "model": str(model or _requested_ai_model_name()).strip() or "unknown",
            "pass_kind": str(pass_kind or "general"),
            "charge_id": str(charge_id or "") or None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retry_attempts": retry_attempts,
            "usage_available": usage_available,
            "completed_at_utc": now_txt,
        }
    )
    usable_history = [row for row in history if row.get("usage_available")]
    complete_usage = len(usable_history) == len(history)
    summary = {
        "version": AI_COST_SUMMARY_VERSION,
        "currency": "USD",
        "job_id": str(job_id),
        "run_id": int(run_id),
        "provider": _requested_ai_provider_name(),
        "model": str(model or _requested_ai_model_name()).strip() or "unknown",
        "rate": rate,
        "successful_invocations": len(history),
        "usage_available_invocations": len(usable_history),
        "input_tokens_total": sum(max(0, _safe_int(row.get("input_tokens"), 0)) for row in usable_history),
        "output_tokens_total": sum(max(0, _safe_int(row.get("output_tokens"), 0)) for row in usable_history),
        "estimated_ai_cost_usd": None,
        "cost_status": "usage_missing" if not complete_usage else "rate_unavailable",
        "updated_at_utc": now_txt,
        "invocations": history,
    }
    estimated_cost = _ai_cost_decimal(summary) if complete_usage else None
    if estimated_cost is not None:
        summary["estimated_ai_cost_usd"] = format(estimated_cost, "f")
        summary["cost_status"] = "estimated"
    mapping_summary[AI_COST_SUMMARY_KEY] = summary
    if charge_id:
        logger.info(
            "AI_PASS_USAGE pass=%s system=%s model=%s input_tokens=%s output_tokens=%s retries=%s charge_id=%s usage_available=%s",
            str(pass_kind or "general"),
            system_id,
            str(model or "unknown"),
            input_tokens,
            output_tokens,
            retry_attempts,
            str(charge_id),
            usage_available,
        )
    return summary


def _requested_anthropic_model_name() -> str:
    return _configured_anthropic_model_name() or "unknown"


def _refresh_ai_run_recovery_flags(row: dict, current_source_state_version: str | None = None) -> dict:
    pass_state_by_system_id = row.get("pass_state_by_system_id") if isinstance(row.get("pass_state_by_system_id"), dict) else {}
    has_saved_progress = max(0, _safe_int(row.get("systems_completed"), 0)) > 0 or any(
        isinstance(pass_state, dict)
        and (
            str(pass_state.get("general") or "").strip() == "completed"
            or str(pass_state.get("ending") or "").strip() in {"completed", "retryable_failed"}
        )
        for pass_state in pass_state_by_system_id.values()
    )
    status = str(row.get("status") or AI_SUGGEST_RUN_STATUS_IDLE).strip().lower()
    systems_total = max(0, _safe_int(row.get("systems_total"), 0))
    next_system_index = max(0, _safe_int(row.get("next_system_index"), 0))
    score_type = _normalize_ai_score_type(row.get("score_type"))
    row["has_saved_progress"] = has_saved_progress
    row["can_continue"] = (
        status in {
            AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED,
            AI_SUGGEST_RUN_STATUS_CANCELLED,
            AI_SUGGEST_RUN_STATUS_FAILED,
        }
        and score_type is not None
        and systems_total > 0
        and next_system_index < systems_total
    )
    return row


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
        AI_SUGGEST_RUN_STATUS_CANCELLED,
        AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED,
    }:
        status = AI_SUGGEST_RUN_STATUS_IDLE
    remembered_time_signature = _normalize_ai_time_signature_value(row.get("remembered_time_signature"))
    last_time_signature_update = _normalize_ai_time_signature_update_row(row.get("last_time_signature_update"))
    time_signature_updates = _normalize_ai_time_signature_update_rows(row.get("time_signature_updates"))
    pass_state_by_system_id = row.get("pass_state_by_system_id") if isinstance(row.get("pass_state_by_system_id"), dict) else {}
    credit_groups = row.get("credit_groups") if isinstance(row.get("credit_groups"), dict) else {}
    run_source_state_version = str(row.get("source_state_version") or source_state_version or "").strip() or None
    clean = {
        "version": str(row.get("version") or "ai_suggest_run_v1"),
        "credit_scheme": str(row.get("credit_scheme") or "one_system_per_credit_v1"),
        "status": status,
        "started_at_utc": str(row.get("started_at_utc") or "").strip() or None,
        "updated_at_utc": str(row.get("updated_at_utc") or "").strip() or None,
        "completed_at_utc": str(row.get("completed_at_utc") or "").strip() or None,
        "failed_at_utc": str(row.get("failed_at_utc") or "").strip() or None,
        "cancelled_at_utc": str(row.get("cancelled_at_utc") or "").strip() or None,
        "systems_total": max(0, _safe_int(row.get("systems_total"), 0)),
        "systems_completed": max(0, _safe_int(row.get("systems_completed"), 0)),
        "next_system_index": max(0, _safe_int(row.get("next_system_index"), 0)),
        "source_run_id": int(run_id) if isinstance(run_id, int) and run_id > 0 else _safe_int(row.get("source_run_id"), 0),
        "source_state_version": run_source_state_version,
        "score_type": _normalize_ai_score_type(row.get("score_type")),
        "execution_id": str(row.get("execution_id") or "").strip() or None,
        "start_request_id": str(row.get("start_request_id") or "").strip() or None,
        "has_saved_progress": False,
        "can_continue": False,
        "model": str(row.get("model") or _requested_ai_model_name()).strip() or "unknown",
        "last_error": row.get("last_error") if isinstance(row.get("last_error"), dict) else None,
        "remembered_time_signature": remembered_time_signature,
        "last_time_signature_update": last_time_signature_update,
        "time_signature_updates": time_signature_updates,
        "pass_state_by_system_id": deepcopy(pass_state_by_system_id),
        "credit_groups": deepcopy(credit_groups),
        "ending_carry_kind": str(row.get("ending_carry_kind") or "").strip() or None,
    }
    return _refresh_ai_run_recovery_flags(clean, source_state_version)


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
        "ending_events_by_measure_id": {},
        "ending_pairs_by_id": {},
        "resolved_ending_pair_ids": [],
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


def _ending_confidence_rank(value: str | None) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value or "").strip().lower(), 0)


def _ending_pair_id(ending_1_start: str | None, ending_2_start: str | None) -> str:
    # Keep the candidate stable if a later system adds the matching Ending 2 start.
    raw = str(ending_1_start or ending_2_start or "missing")
    return "ending_pair_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _rebuild_ai_ending_pairs(ai_suggestions: dict, ordered_measures: list[dict], *, all_systems_completed: bool) -> dict:
    events = ai_suggestions.get("ending_events_by_measure_id")
    if not isinstance(events, dict):
        events = {}
    ordered = [
        str(row.get("measure_id") or "").strip()
        for row in _sorted_measure_rows(ordered_measures or [])
        if isinstance(row, dict) and str(row.get("measure_id") or "").strip()
    ]
    pairs: dict[str, dict] = {}
    resolved_pair_ids = {
        str(value or "").strip()
        for value in (ai_suggestions.get("resolved_ending_pair_ids") or [])
        if str(value or "").strip()
    }
    starts: list[dict] = []
    for measure_id in ordered:
        event = events.get(measure_id)
        if not isinstance(event, dict):
            continue
        kind = str(event.get("start") or "none").strip().lower()
        if kind not in {"ending_1", "ending_2"}:
            continue
        starts.append(
            {
                "measure_id": measure_id,
                "kind": kind,
                "confidence": str(event.get("confidence") or "low").strip().lower(),
            }
        )

    index = 0
    while index < len(starts):
        first = starts[index]
        candidates = [first]
        if (
            first["kind"] == "ending_1"
            and index + 1 < len(starts)
            and starts[index + 1]["kind"] == "ending_2"
        ):
            candidates.append(starts[index + 1])
            index += 1

        ending_1 = next((row for row in candidates if row["kind"] == "ending_1"), None)
        ending_2 = next((row for row in candidates if row["kind"] == "ending_2"), None)
        ending_1_start = (ending_1 or {}).get("measure_id")
        ending_2_start = (ending_2 or {}).get("measure_id")
        anchors = [row["measure_id"] for row in candidates]
        pair_id = _ending_pair_id(ending_1_start, ending_2_start)
        if pair_id in resolved_pair_ids:
            index += 1
            continue
        confidence = min(
            candidates,
            key=lambda row: _ending_confidence_rank(row.get("confidence")),
        ).get("confidence") or "low"
        pairs[pair_id] = {
            "pair_id": pair_id,
            "status": "candidate",
            "review_mode": "three_boundary",
            "review_anchor_measure_ids": anchors,
            "confidence": confidence,
            "order_measure_id": anchors[0],
            "ending_1_start_measure_id": ending_1_start,
            "ending_1_end_measure_id": ending_1_start,
            "ending_2_start_measure_id": ending_2_start,
            "ending_2_end_measure_id": ending_2_start,
            "ending_1_measure_ids": [ending_1_start] if ending_1_start else [],
            "ending_2_measure_ids": [ending_2_start] if ending_2_start else [],
            "missing_boundaries": ["ending_1_start", "ending_2_start", "ending_2_end"],
        }
        index += 1
    ai_suggestions["ending_pairs_by_id"] = pairs
    return pairs


def _new_ai_suggest_run_state(
    run_id: int,
    source_state_version: str | None,
    systems_total: int,
    status: str = AI_SUGGEST_RUN_STATUS_RUNNING,
    score_type: str | None = None,
    execution_id: str | None = None,
    start_request_id: str | None = None,
) -> dict:
    now_txt = _utc_now().isoformat().replace("+00:00", "Z")
    row = {
        "version": AI_SUGGEST_RUN_VERSION,
        "credit_scheme": AI_CREDIT_SCHEME_VERSION,
        "status": status,
        "started_at_utc": now_txt if status in {AI_SUGGEST_RUN_STATUS_RUNNING, AI_SUGGEST_RUN_STATUS_COMPLETED} else None,
        "updated_at_utc": now_txt,
        "completed_at_utc": now_txt if status == AI_SUGGEST_RUN_STATUS_COMPLETED else None,
        "failed_at_utc": now_txt if status == AI_SUGGEST_RUN_STATUS_FAILED else None,
        "cancelled_at_utc": now_txt if status == AI_SUGGEST_RUN_STATUS_CANCELLED else None,
        "systems_total": max(0, int(systems_total)),
        "systems_completed": max(0, int(systems_total)) if status == AI_SUGGEST_RUN_STATUS_COMPLETED else 0,
        "next_system_index": max(0, int(systems_total)) if status == AI_SUGGEST_RUN_STATUS_COMPLETED else 0,
        "source_run_id": int(run_id),
        "source_state_version": str(source_state_version or "").strip() or None,
        "score_type": _normalize_ai_score_type(score_type),
        "execution_id": str(execution_id or uuid.uuid4()).strip(),
        "start_request_id": str(start_request_id or "").strip() or None,
        "has_saved_progress": False,
        "can_continue": False,
        "model": _requested_ai_model_name(),
        "last_error": None,
        "remembered_time_signature": None,
        "last_time_signature_update": None,
        "time_signature_updates": [],
        "pass_state_by_system_id": {},
        "credit_groups": {},
        "ending_carry_kind": None,
    }
    return row


def _ai_credit_group_id(
    job_id: str,
    run_id: int,
    source_state_version: str | None,
    system_index: int,
    execution_id: str | None = None,
) -> str:
    pair_index = max(0, int(system_index)) // 2
    execution_part = str(execution_id or "").strip()
    if execution_part:
        raw = f"{job_id}|{int(run_id)}|{str(source_state_version or '')}|{execution_part}|{pair_index}"
    else:
        raw = f"{job_id}|{int(run_id)}|{str(source_state_version or '')}|{pair_index}"
    return "ai-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ai_credit_pass_id(
    job_id: str,
    run_id: int,
    source_state_version: str | None,
    system_index: int,
    pass_kind: str,
    execution_id: str | None = None,
) -> str:
    normalized_kind = "ending" if str(pass_kind or "").strip().lower() == "ending" else "general"
    charge_index = max(0, int(system_index)) // 2 if normalized_kind == "ending" else max(0, int(system_index))
    execution_part = str(execution_id or "").strip()
    raw = (
        f"{job_id}|{int(run_id)}|{str(source_state_version or '')}|{execution_part}|"
        f"{AI_CREDIT_SCHEME_VERSION}|{normalized_kind}|{charge_index}"
    )
    return "ai-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ai_execution_id(
    job_id: str,
    run_id: int,
    source_state_version: str | None,
    request_id: str,
) -> str:
    raw = f"{job_id}|{int(run_id)}|{str(source_state_version or '')}|{request_id}"
    return "airun-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _normalize_ai_start_request_id(raw_value) -> str | None:
    text = str(raw_value or "").strip()
    if not text or len(text) > 128:
        return None
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError, TypeError):
        return None


def _reconcile_ai_restart_credit_groups(ai_suggest_run: dict, job_id: str) -> None:
    credit_groups = ai_suggest_run.get("credit_groups") if isinstance(ai_suggest_run.get("credit_groups"), dict) else {}
    changed = False
    for charge_id, raw_group in credit_groups.items():
        group = dict(raw_group) if isinstance(raw_group, dict) else {}
        if group.get("charged"):
            continue
        status = str(group.get("status") or "").strip().lower()
        system_ids = group.get("system_ids") if isinstance(group.get("system_ids"), list) else []
        system_id = str(system_ids[0] or "").strip() if system_ids else None
        if status == "charge_pending":
            access = _ai_access(
                reserve=True,
                job_id=job_id,
                system_id=system_id,
                charge_id=str(charge_id),
            )
            if not _finish_ai_access(access, spent=True):
                raise AiSuggestError(
                    code="ai_restart_credit_pending",
                    message="The previous AI credit is still being confirmed. Try Restart again.",
                    retryable=True,
                    provider_status=503,
                    detail="credit_finalize_pending",
                )
            group.update(
                {
                    "status": "charged",
                    "charged": True,
                    "charged_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
                    "provider": str(access.get("provider") or "friend"),
                }
            )
            credit_groups[str(charge_id)] = group
            changed = True
        elif status in {"reserved", "reservation_pending"}:
            access = _ai_access(
                reserve=True,
                job_id=job_id,
                system_id=system_id,
                charge_id=str(charge_id),
            )
            if not _finish_ai_access(access, spent=False):
                raise AiSuggestError(
                    code="ai_restart_credit_pending",
                    message="The previous AI credit is still being released. Try Restart again.",
                    retryable=True,
                    provider_status=503,
                    detail="credit_release_pending",
                )
            group["status"] = "released"
            credit_groups[str(charge_id)] = group
            changed = True
    if changed:
        ai_suggest_run["credit_groups"] = credit_groups


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


def _merge_ai_ending_system_state(
    ai_suggestions: dict,
    ending_result: dict,
    ordered_measures: list[dict],
    *,
    all_systems_completed: bool,
) -> dict:
    result = dict(ai_suggestions or {})
    events_by_measure_id = dict(result.get("ending_events_by_measure_id") or {})
    warnings = [dict(row) for row in (result.get("warnings") or []) if isinstance(row, dict)]
    for event in ending_result.get("events") or []:
        if not isinstance(event, dict):
            continue
        measure_id = str(event.get("measure_id") or "").strip()
        if measure_id:
            events_by_measure_id[measure_id] = dict(event)
        if str(event.get("start") or "").strip() == "unsupported":
            warnings.append(
                {
                    "type": "unsupported_ending_ignored",
                    "system_id": str(ending_result.get("system_id") or "") or None,
                    "message": "A third or combined ending was ignored.",
                }
            )
    result["ending_events_by_measure_id"] = events_by_measure_id
    result["ending_version"] = AI_SUGGESTIONS_ENDING_VERSION
    result["ending_provider"] = str(ending_result.get("provider") or _requested_ai_provider_name())
    result["ending_model"] = str(ending_result.get("model") or _requested_ai_model_name("ending"))
    result["warnings"] = warnings
    _rebuild_ai_ending_pairs(result, ordered_measures, all_systems_completed=all_systems_completed)
    return result


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


def _remove_ai_ending_pairs(mapping_summary: dict | None, pair_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    ai_suggestions = _current_ai_suggestions(mapping_summary)
    if not isinstance(ai_suggestions, dict):
        return []
    pairs = ai_suggestions.get("ending_pairs_by_id")
    if not isinstance(pairs, dict):
        return []
    removed: list[str] = []
    for pair_id in pair_ids or []:
        clean_id = str(pair_id or "").strip()
        if clean_id and clean_id in pairs:
            pairs.pop(clean_id, None)
            removed.append(clean_id)
    ai_suggestions["ending_pairs_by_id"] = pairs
    return removed


def _resolve_ai_ending_pairs(mapping_summary: dict | None, pair_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    ai_suggestions = _current_ai_suggestions(mapping_summary)
    if not isinstance(ai_suggestions, dict):
        return []
    resolved = {
        str(value or "").strip()
        for value in (ai_suggestions.get("resolved_ending_pair_ids") or [])
        if str(value or "").strip()
    }
    clean_ids = {
        str(value or "").strip()
        for value in (pair_ids or [])
        if str(value or "").strip()
    }
    resolved.update(clean_ids)
    ai_suggestions["resolved_ending_pair_ids"] = sorted(resolved)
    _remove_ai_ending_pairs(mapping_summary, clean_ids)
    return sorted(clean_ids)


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
    label_ids = {f"label:{measure_id}" for measure_id in ids}
    editable_state["hidden_label_ids"] = [
        label_id for label_id in _editable_hidden_label_ids(editable_state)
        if label_id not in label_ids
    ]
    editable_state["forced_label_ids"] = [
        label_id for label_id in _editable_forced_label_ids(editable_state)
        if label_id not in label_ids
    ]
    positions = _editable_label_positions(editable_state)
    for label_id in label_ids:
        positions.pop(label_id, None)
    editable_state["label_positions"] = positions


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

        measure_row = measure_rows_by_id[measure_id]
        label = str(row.get("label") or "").strip()
        ignored_ending_label = label in AI_SUGGESTION_ENDING_LABELS
        if ignored_ending_label:
            normalization_warnings.append(
                _ai_suggest_normalization_warning(
                    measure_row,
                    f"Ignored unsupported AI ending label for {measure_id}; treated as normal.",
                )
            )
            label = "normal"
        if label not in AI_SUGGESTION_LABELS_ALLOWED:
            raise AiSuggestError(detail=f"malformed_response: invalid label for {measure_id}")
        confidence = str(row.get("confidence") or "").strip().lower()
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
        if is_first_measure_of_score and label != "false_measure":
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
        elif label == "false_measure":
            measure_completeness = "not_applicable"
        elif ignored_ending_label:
            measure_completeness = "full"
        elif not is_first_measure_of_score and label == "normal" and measure_completeness == "incomplete":
            normalization_warnings.append(
                _ai_suggest_normalization_warning(
                    measure_row,
                    f"Ignored later normal incomplete completeness for {measure_id}; later measures do not use pickup completeness.",
                )
            )
            measure_completeness = "full"

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
                        f"Treated multi_measure_rest as normal for {measure_id} because rest_count was missing or invalid.",
                    )
                )
                normal_measures_omitted += 1
                continue
            if maybe_label is not None or maybe_rest_count is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored maybe fields on multi_measure_rest suggestion for {measure_id}.",
                    )
                )
            entry["rest_count"] = int(rest_count)
        elif label == "false_measure":
            if rest_count is not None or maybe_label is not None or maybe_rest_count is not None or raw_unclear_reason is not None:
                normalization_warnings.append(
                    _ai_suggest_normalization_warning(
                        measure_row,
                        f"Ignored extra fields on false_measure suggestion for {measure_id}.",
                    )
                )
            entry.pop("unclear_reason", None)
            entry["rest_count"] = None
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


def _is_bedrock_retryable_error(exc: AiSuggestError | Exception) -> bool:
    if not isinstance(exc, AiSuggestError):
        return False
    detail = str(getattr(exc, "detail", "") or "").lower()
    if "provider_not_configured" in detail or "malformed_provider_response" in detail:
        return False
    permanent_markers = (
        "invalidsignatureexception",
        "accessdeniedexception",
        "unauthorized",
        "validationexception",
        "on-demand throughput isn",
        "not authorized",
    )
    if any(marker in detail for marker in permanent_markers):
        return False
    retryable_markers = (
        "throttlingexception",
        "serviceunavailableexception",
        "internalserverexception",
        "modeltimeoutexception",
        "timeout",
        "timed out",
        "connection",
    )
    if any(marker in detail for marker in retryable_markers):
        return True
    return int(getattr(exc, "provider_status", 0) or 0) in {408, 429, 500, 502, 503, 504}


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
    model_id = str((payload or {}).get("model") or _configured_bedrock_model_id()).strip()
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


def _bedrock_messages_create(payload: dict) -> dict:
    delays = tuple(float(delay) for delay in BEDROCK_RETRY_DELAYS_SEC if float(delay) > 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            response = _bedrock_messages_create_once(payload)
            if isinstance(response, dict):
                response = dict(response)
                response["_internal_bedrock_attempts"] = attempt
            return response
        except AiSuggestError as exc:
            if not _is_bedrock_retryable_error(exc):
                raise
            if attempt > len(delays):
                detail = str(exc.detail or "").strip()
                delay_txt = ",".join(str(int(delay) if float(delay).is_integer() else delay) for delay in delays)
                suffix = f" bedrock_retry_attempts={attempt}"
                if delay_txt:
                    suffix += f" bedrock_retry_delays_sec={delay_txt}"
                exc.detail = f"{detail}{suffix}" if detail else suffix.strip()
                exc.retry_attempts = attempt
                raise
            delay_sec = delays[attempt - 1]
            logger.warning(
                "AI_SUGGEST_BEDROCK_RETRY attempt=%s next_delay_sec=%s status=%s model=%s",
                attempt,
                delay_sec,
                int(getattr(exc, "provider_status", 0) or 0),
                _configured_bedrock_model_id() or str(payload.get("model") or _requested_ai_model_name()),
            )
            time.sleep(delay_sec)


def _ai_messages_create(payload: dict) -> dict:
    provider = _requested_ai_provider_name()
    if provider == "bedrock":
        return _bedrock_messages_create(payload)
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


def _ai_message_text_parts(message: dict | None) -> list[str]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and str(block.get("type") or "").strip() == "text"
    ]


def _ai_general_response_diagnostics(
    message: dict | None,
    *,
    system_id: str,
    measure_count: int,
    reference_count: int,
    model: str,
) -> dict:
    row = message if isinstance(message, dict) else {}
    content = row.get("content") if isinstance(row.get("content"), list) else []
    text_parts = _ai_message_text_parts(row)
    text = "\n".join(text_parts)
    stripped = _strip_json_fences(text).strip()
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    output_tokens = max(0, _safe_int(usage.get("output_tokens"), 0))
    raw_stop_reason = str(row.get("stop_reason") or "unknown").strip().lower()
    stop_reason = raw_stop_reason if raw_stop_reason in {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn"} else "other"
    block_types: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            block_type = "invalid"
        else:
            raw_type = str(block.get("type") or "").strip().lower()
            block_type = raw_type if raw_type in {"text", "tool_use", "thinking", "redacted_thinking"} else "other"
        if block_type not in block_types:
            block_types.append(block_type)
    return {
        "model": str(model or "unknown"),
        "system_id": str(system_id or "unknown"),
        "measure_count": max(0, int(measure_count)),
        "reference_count": max(0, int(reference_count)),
        "max_tokens": int(ANTHROPIC_MAX_TOKENS),
        "stop_reason": stop_reason,
        "input_tokens": max(0, _safe_int(usage.get("input_tokens"), 0)),
        "output_tokens": output_tokens,
        "content_blocks": len(content),
        "block_types": ",".join(block_types) if block_types else "none",
        "text_blocks": len(text_parts),
        "text_chars": len(text),
        "starts_object": stripped.startswith("{"),
        "ends_object": stripped.endswith("}"),
        "open_braces": text.count("{"),
        "close_braces": text.count("}"),
        "open_brackets": text.count("["),
        "close_brackets": text.count("]"),
        "provider_attempts": max(1, _safe_int(row.get("_internal_bedrock_attempts"), 1)),
        "output_limit_reached": stop_reason == "max_tokens" or output_tokens >= int(ANTHROPIC_MAX_TOKENS),
    }


def _ai_general_failure_diagnostics(exc: Exception, response_diagnostics: dict | None) -> dict:
    detail = str(getattr(exc, "detail", "") or "").lower()
    if "no text content" in detail or "content missing" in detail:
        category = "missing_text"
    elif "missing json object" in detail:
        category = "missing_object"
    elif "invalid json" in detail:
        category = "invalid_json"
    elif "duplicate measure_id" in detail:
        category = "duplicate_measures"
    elif "unknown measure_id" in detail:
        category = "unknown_measures"
    elif "suggestions missing" in detail or "missing_measure_ids" in detail:
        category = "missing_measures"
    else:
        category = "malformed_response"
    position = re.search(r"line\s+(\d+)\s+column\s+(\d+)\s+\(char\s+(\d+)\)", detail)
    diagnostics = dict(response_diagnostics or {})
    diagnostics.update(
        {
            "category": category,
            "error_line": int(position.group(1)) if position else 0,
            "error_column": int(position.group(2)) if position else 0,
            "error_char": int(position.group(3)) if position else 0,
        }
    )
    return diagnostics


def _log_ai_general_response_debug(diagnostics: dict) -> None:
    if not _ai_suggest_debug_enabled():
        return
    logger.info(
        "AI_GENERAL_RESPONSE_DEBUG model=%s system=%s measures=%s references=%s max_tokens=%s "
        "stop_reason=%s input_tokens=%s output_tokens=%s content_blocks=%s block_types=%s "
        "text_blocks=%s text_chars=%s starts_object=%s ends_object=%s open_braces=%s "
        "close_braces=%s open_brackets=%s close_brackets=%s provider_attempts=%s output_limit_reached=%s",
        diagnostics.get("model"), diagnostics.get("system_id"), diagnostics.get("measure_count"),
        diagnostics.get("reference_count"), diagnostics.get("max_tokens"), diagnostics.get("stop_reason"),
        diagnostics.get("input_tokens"), diagnostics.get("output_tokens"), diagnostics.get("content_blocks"),
        diagnostics.get("block_types"), diagnostics.get("text_blocks"), diagnostics.get("text_chars"),
        diagnostics.get("starts_object"), diagnostics.get("ends_object"), diagnostics.get("open_braces"),
        diagnostics.get("close_braces"), diagnostics.get("open_brackets"), diagnostics.get("close_brackets"),
        diagnostics.get("provider_attempts"), diagnostics.get("output_limit_reached"),
    )


def _log_ai_general_parse_failed(diagnostics: dict) -> None:
    if not _ai_suggest_debug_enabled():
        return
    logger.warning(
        "AI_GENERAL_PARSE_FAILED category=%s model=%s system=%s measures=%s references=%s max_tokens=%s "
        "stop_reason=%s input_tokens=%s output_tokens=%s text_blocks=%s text_chars=%s "
        "starts_object=%s ends_object=%s open_braces=%s close_braces=%s open_brackets=%s "
        "close_brackets=%s provider_attempts=%s output_limit_reached=%s error_line=%s error_column=%s error_char=%s",
        diagnostics.get("category"), diagnostics.get("model"), diagnostics.get("system_id"),
        diagnostics.get("measure_count"), diagnostics.get("reference_count"), diagnostics.get("max_tokens"),
        diagnostics.get("stop_reason"), diagnostics.get("input_tokens"), diagnostics.get("output_tokens"),
        diagnostics.get("text_blocks"), diagnostics.get("text_chars"), diagnostics.get("starts_object"),
        diagnostics.get("ends_object"), diagnostics.get("open_braces"), diagnostics.get("close_braces"),
        diagnostics.get("open_brackets"), diagnostics.get("close_brackets"), diagnostics.get("provider_attempts"),
        diagnostics.get("output_limit_reached"), diagnostics.get("error_line"), diagnostics.get("error_column"),
        diagnostics.get("error_char"),
    )


def _parse_anthropic_suggestions_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise AiSuggestError(detail="malformed_provider_response")
    content = message.get("content")
    if not isinstance(content, list):
        raise AiSuggestError(detail="malformed_provider_response: content missing")
    text_parts = _ai_message_text_parts(message)
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


def _ending_measure_crop_spec(
    page_rect,
    measure_row: dict,
    next_measure_row: dict | None,
    system_row: dict | None,
    prev_system_row: dict | None = None,
    next_system_row: dict | None = None,
) -> dict:
    spec = _measure_crop_spec(page_rect, measure_row, next_measure_row, system_row, prev_system_row, next_system_row)
    bounds = dict(spec.get("measure_bounds") or {})
    width = max(1.0, _safe_float(bounds.get("width"), 1.0))
    x_pad = min(width * 0.25, max(8.0, width * 0.08))
    clip = spec["clip"]
    spec["clip"] = fitz.Rect(
        max(0.0, float(clip.x0) - x_pad),
        float(clip.y0),
        min(float(page_rect.width), float(clip.x1) + x_pad),
        float(clip.y1),
    )
    spec["padding"] = dict(spec.get("padding") or {})
    spec["padding"]["left"] = min(x_pad, max(0.0, float(clip.x0)))
    spec["padding"]["right"] = min(x_pad, max(0.0, float(page_rect.width) - float(clip.x1)))
    return spec


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


def _build_multi_rest_reference_content() -> tuple[list[dict], int]:
    content: list[dict] = []
    example_rows: list[dict] = []
    for row in AI_MULTI_REST_REFERENCE_EXAMPLES:
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
                "Reference examples for multi-measure rest recognition. "
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


def _build_false_measure_reference_content() -> tuple[list[dict], int]:
    content: list[dict] = []
    rows: list[dict] = []
    for row in AI_FALSE_MEASURE_REFERENCE_EXAMPLES:
        image_path = AI_REFERENCE_EXAMPLES_DIR / str(row.get("filename") or "")
        try:
            image_bytes = image_path.read_bytes()
        except FileNotFoundError:
            logger.warning("AI_FALSE_MEASURE_REFERENCE_MISSING filename=%s", image_path.name)
            continue
        except Exception as exc:
            logger.warning(
                "AI_FALSE_MEASURE_REFERENCE_LOAD_FAILED filename=%s error_type=%s",
                image_path.name,
                type(exc).__name__,
            )
            continue
        rows.append({"caption": str(row.get("caption") or ""), "image_bytes": image_bytes})
    if not rows:
        return content, 0
    content.append(
        {
            "type": "text",
            "text": (
                "The next two images are false-measure references only. "
                "After them, classify the real target candidate-box crops."
            ),
        }
    )
    for row in rows:
        content.append({"type": "text", "text": row["caption"]})
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
    return content, len(rows)


def _build_ending_reference_content() -> tuple[list[dict], int]:
    content: list[dict] = []
    rows: list[dict] = []
    for row in AI_ENDING_REFERENCE_EXAMPLES:
        image_path = AI_REFERENCE_EXAMPLES_DIR / str(row.get("filename") or "")
        try:
            image_bytes = image_path.read_bytes()
        except FileNotFoundError:
            logger.warning("AI_ENDING_REFERENCE_MISSING filename=%s", image_path.name)
            continue
        except Exception as exc:
            logger.warning("AI_ENDING_REFERENCE_LOAD_FAILED filename=%s error_type=%s", image_path.name, type(exc).__name__)
            continue
        rows.append({"caption": str(row.get("caption") or ""), "image_bytes": image_bytes})
    if not rows:
        return content, 0
    content.append(
        {
            "type": "text",
            "text": (
                "The next seven images are labeled ending-bracket references only. "
                "After them, classify the real target-measure crops."
            ),
        }
    )
    for row in rows:
        content.append({"type": "text", "text": row["caption"]})
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
    return content, len(rows)


def _ai_prompt_false_measure_rules() -> list[str]:
    return [
        "False-measure check - perform this before every other classification:",
        "A false measure is an incorrect measure box containing no music that consumes time.",
        "Return false_measure only when the target box contains no notes, no chords, no ordinary rests, no full-measure rests, and no multi-measure-rest symbol.",
        "Clefs, key signatures, numeric time signatures, common time, cut time, staff lines, barlines, repeat barlines, repeat dots, and nearby text do not consume musical time by themselves.",
        "A target box containing only those setup symbols is false_measure.",
        "If any staff inside the target box contains a note, chord, ordinary rest, full-measure rest, or multi-measure-rest symbol, it is a real measure and must not be false_measure.",
        "Setup symbols followed by notes or rests inside the same target box belong to a real measure.",
        "Do not choose false_measure merely because the box is narrow, begins a page or system, or contains a time-signature change.",
        "For Grand Staff and Full Score, the target is a real measure if any staff contains music that consumes time.",
        "Classify only the target candidate box. Any neighboring material visible at the crop edges is context only.",
        "Always make the best allowed choice. Never return uncertain.",
    ]


def _ai_prompt_base_rules() -> list[str]:
    return [
        "Each image contains exactly one already-detected measure candidate box. Some candidate boxes may contain only setup notation and may not be real measures.",
        "Staff means one set of 5 horizontal music lines for one instrument or voice.",
        "System means the full horizontal row of music containing all staves on that line of the score.",
        "Do not infer additional measures from rhythmic groupings, repeat dots, barline decorations, edge marks, spacing, or decorations.",
        "Process the provided measures left to right in order.",
        *_ai_prompt_false_measure_rules(),
        "A numeric time signature is two vertically stacked meter numbers immediately after the clef/key signature, such as 2 over 4.",
        "Ignore fingering/count numbers near notes, above the staff, or below the staff. They are not time signatures.",
        "Do not remember, inherit, carry, or track time signatures across measures.",
        "Only read a time signature if it is visible in the current crop.",
        "Only use meter for first-measure pickup judgment.",
        "For multi-measure rests, ignore meter completely.",
        "Only label pickup when is_first_measure_of_score is true.",
        "If is_first_measure_of_score is false, do not label pickup.",
        "Set measure_completeness only as needed: pickup = incomplete, multi_measure_rest = full, false_measure = not_applicable, clear normal = full, unclear = unclear.",
        "Always make the best useful choice from the allowed labels, even when confidence is low.",
        "If measure_completeness is unclear, you may include unclear_reason using one of these exact codes only: time_signature_not_clear, too_dense_to_count, crop_cut_off, split_may_be_wrong, ornament_or_tie_confusion, not_enough_visual_evidence.",
        "Do not write sentences for unclear_reason. Use only one short code or omit the field.",
        "For non-first measures, do not judge pickup or beat completeness.",
        "Later measures must be normal unless they are a valid multi_measure_rest, false_measure, or a clearly numbered ending start.",
    ]


def _ai_prompt_duration_basics() -> list[str]:
    return [
        "Read meter as top/bottom: top is how many beat-units fill a full measure; bottom is which note value is one beat-unit.",
        "Count written note/rest durations only. Never count visual width, spacing, number of noteheads, or number of staves as beats.",
        "Chords or stacked notes count as exactly one rhythmic event using the written note value. Do not count each notehead separately.",
        "For later non-first measures, do not label pickup.",
        "If the first measure is hard to read, make the best pickup-or-normal choice and use low confidence.",
        "A whole note, two half notes, or other sparse-looking content in the first measure is usually a slow full measure, not a pickup. Do not label pickup just because the first measure looks sparse or simple.",
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


def _ai_prompt_ending_rules(score_type: str) -> list[str]:
    if score_type == "single":
        scope_rule = "For single-staff music, inspect above this one staff for ending brackets."
    elif score_type == "grand":
        scope_rule = "For grand-staff/piano music, inspect above the top staff only; do not duplicate ending labels for the bottom staff."
    elif score_type == "score":
        scope_rule = "For full-score music, inspect above the top visible staff/system only; do not duplicate ending labels for each instrument."
    else:
        scope_rule = "Inspect above the staff/system for ending brackets."
    return [
        "Ending / volta detection:",
        scope_rule,
        "Look above the staff/system for repeat-ending or volta brackets.",
        "First ending markers include 1, 1., 1st, or prima volta.",
        "Second ending markers include 2, 2., 2nd, or seconda volta.",
        "A bracket start often looks like an upside-down L: a left vertical line drops down and a horizontal line connects to its top and continues right.",
        "The bracket may end with a right vertical line, or the horizontal line may simply stop.",
        "Only label the measure where the bracket clearly starts.",
        "If the crop only shows a continuing horizontal line with no readable number/start, do not label it as a new ending.",
        "If the bracket number is unreadable, do not invent an ending label; make the best normal, pickup, or multi-measure-rest choice instead.",
        "Do not return a finish/end measure for endings.",
    ]


def _ai_prompt_multi_rest_rules(score_type: str) -> list[str]:
    if score_type == "grand":
        scope_rule = "For grand-staff multi-measure rests, inspect only the top staff/treble staff."
    elif score_type == "single":
        scope_rule = "For single-staff multi-measure rests, inspect the one visible staff."
    else:
        scope_rule = "Inspect the visible rest symbol and its printed count in the target measure."
    return [
        "Multi-measure rest decision order:",
        scope_rule,
        "Check this before pickup or meter; meter does not determine a multi-measure rest.",
        "The supplied reference images are examples only. The real target measure images follow them.",
        "First find a printed count that belongs to the rest symbol in the target measure.",
        "A readable count of 2 or more with a modern H-bar, thick horizontal block, or old-style bar-piece rest symbol means multi_measure_rest.",
        "Return the printed count exactly as rest_count; the symbol's number of bars does not need to match it.",
        "A count of 1 or an ordinary quarter, half, whole, or full-measure rest means normal.",
        "Ignore unrelated measure numbers, rehearsal marks, fingerings, lyrics, ending numbers, and numbers attached to notes.",
        "If the count or symbol is difficult to read, make the best normal-or-multi_measure_rest choice and use low confidence.",
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
        "If the first-measure meter or rhythm is unclear, make the best pickup-or-normal choice and use low confidence.",
        *_ai_prompt_ending_rules("single"),
        *_ai_prompt_multi_rest_rules("single"),
    ]


def _ai_prompt_grand_rules() -> list[str]:
    return [
        "Grand-staff main rule:",
        "For all AI Suggest decisions in grand-staff/piano music, use only the top staff/treble staff.",
        "Ignore the bottom staff completely for time signature, pickup, and multi-measure rest.",
        "Do not inspect, compare, add, or use the bottom staff as fallback.",
        "If the top staff is hard to read, unreadable, empty, or cut off, do not switch to the bottom staff; make the best top-staff choice with low confidence.",
        "Grand-staff pickup rules:",
        "This is grand-staff/piano music. For pickup counting, use only the top staff/treble staff.",
        "Ignore the bottom staff completely for pickup duration. Do not inspect it, use it as fallback, compare it, or add it.",
        "Do not use bottom-staff notes or rests to decide whether the measure is full.",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use only the top staff's visible meter in this crop.",
        "If the same meter appears on both staves, ignore the bottom duplicate.",
        "If the top staff meter is unreadable, make the best pickup-or-normal choice with low confidence.",
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
        "If the top staff meter or rhythm is unreadable or cut off, make the best pickup-or-normal choice with low confidence.",
        *_ai_prompt_ending_rules("grand"),
        *_ai_prompt_multi_rest_rules("grand"),
    ]


def _ai_prompt_score_rules() -> list[str]:
    return [
        "Full-score pickup rules:",
        "For score pickup, start at the top visible staff.",
        "If a staff shows only a full-measure rest in the first measure, skip to the next staff down.",
        "Use the first staff with notes or readable rhythm; do not add multiple staves.",
        "If every staff is resting or unreadable, make the best pickup-or-normal choice with low confidence.",
        "Only check pickup when is_first_measure_of_score is true.",
        "Use the first active staff's visible meter in this crop.",
        "If the first active staff meter is unreadable but common/cut/numeric meter is partially visible, make the best reasonable meter read and continue.",
        "Read meter as top/bottom: top = how many beat-units fill a full measure; bottom = which note value is one beat-unit.",
        "Bottom number examples: 4 means quarter-note beats, 8 means eighth-note beats, 2 means half-note beats.",
        "Common time looks like a large C after the clef/key signature and means 4/4.",
        "Cut time looks like a large C with a vertical slash through it and means 2/2.",
        "Count the first active staff's written note/rest durations only. Never count visual width, spacing, number of noteheads, number of staves, or number of instruments as beats.",
        "Basic note values: filled notehead with stem = quarter note; open notehead with stem = half note; open notehead without stem = whole note; filled notehead with flag or beam = eighth note.",
        "For first-measure pickup debug, identify the first active staff's notehead fill before deciding note value.",
        "A filled black notehead cannot be a half note; half note requires an open/white notehead.",
        "If unsure, use notehead fill first: black = quarter/eighth family, open = half/whole family.",
        "A dot immediately to the right of a note/rest adds half its value.",
        "A triplet is marked by a small 3 above or below a group; the 3 may have a bracket or appear over beamed notes.",
        "Three triplet notes fit into the time normally taken by two of the same note value. Example: three triplet eighth notes equal one quarter-note beat.",
        "A chord/stack on the first active staff is exactly one rhythmic event, no matter how many noteheads it has. Use the written note value.",
        "If the first active staff's written duration is less than the visible meter, the whole first measure is pickup/incomplete.",
        "For the first measure, arithmetic wins over context; do not call a short first measure full because it looks intentional, musical, complete, or like an opening gesture.",
        "Always make the best pickup-or-normal choice, even when the notation is old, small, light, or slightly messy.",
        "Only label normal/full if the first active staff clearly fills the visible meter.",
        "If the first active staff may be short, choose pickup with low confidence.",
        "If the first active staff meter or rhythm is completely unreadable, empty, or cut off, make the best pickup-or-normal choice with low confidence.",
        *_ai_prompt_ending_rules("score"),
        "Full-score multi-measure rest rules:",
        "For full score V1, NEVER return multi_measure_rest.",
        "Do not look for multi-measure rests in full-score crops.",
        "Do not use any printed rest count, H-bar, old-style rest symbol, or instrument rest to skip score measures.",
        "A rest count on one staff only means that instrument may be resting; it does not mean the score should skip measures.",
        "Even if multiple staves show rest symbols, full-score V1 must still count visible score measures normally.",
        "If a full-score rest situation is confusing, label normal, never multi_measure_rest.",
        "rest_count must always be null for full-score prompts.",
    ]


def _ai_prompt_output_rules() -> list[str]:
    return [
        "Do not skip any provided measure_id.",
        "Do not output labels outside the allowed set.",
        "Always choose normal, pickup, multi_measure_rest, false_measure, ending_1, or ending_2; never return uncertain or maybe fields.",
        "If label is multi_measure_rest, rest_count must be an integer >= 2. If label is not multi_measure_rest, rest_count must be null.",
        "For false_measure, measure_completeness must be not_applicable, unclear_reason and rest_count must be null, and decision_debug may be null.",
        "For the first measure of the score only, decision_debug is required unless label is false_measure. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_single_output_rules() -> list[str]:
    return [
        "Allowed labels: normal, pickup, multi_measure_rest, false_measure, ending_1, ending_2.",
        "Do not skip any provided measure_id.",
        "Do not output labels outside the allowed set.",
        "Never return uncertain, maybe_label, or maybe_rest_count; make the best allowed choice and use low confidence when needed.",
        "If label is multi_measure_rest, rest_count must be an integer >= 2. If label is not multi_measure_rest, rest_count must be null.",
        "For ending_1 and ending_2, rest_count must be null and measure_completeness should be full.",
        "For false_measure, measure_completeness must be not_applicable, unclear_reason and rest_count must be null, and decision_debug may be null.",
        "Do not output ending finish/end measures.",
        "For the first measure of the score only, decision_debug is required unless label is false_measure. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_grand_output_rules() -> list[str]:
    return [
        "Allowed labels: normal, pickup, multi_measure_rest, false_measure, ending_1, ending_2.",
        "Do not skip any provided measure_id.",
        "Do not output labels outside the allowed set.",
        "Never return uncertain, maybe_label, or maybe_rest_count; make the best allowed choice and use low confidence when needed.",
        "If label is multi_measure_rest, rest_count must be an integer >= 2. If label is not multi_measure_rest, rest_count must be null.",
        "For ending_1 and ending_2, rest_count must be null and measure_completeness should be full.",
        "For false_measure, measure_completeness must be not_applicable, unclear_reason and rest_count must be null, and decision_debug may be null.",
        "Do not output ending finish/end measures.",
        "For the first measure of the score only, decision_debug is required unless label is false_measure. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
        "Return JSON only.",
    ]


def _ai_prompt_score_output_rules() -> list[str]:
    return [
        "Allowed labels: normal, pickup, false_measure, ending_1, ending_2.",
        "Never return uncertain, maybe_label, or maybe_rest_count; make the best allowed choice and use low confidence when needed.",
        "Do not skip any measure_id. Every input measure_id must appear exactly once.",
        "For full score V1, never output multi_measure_rest, and rest_count must always be null.",
        "For ending_1 and ending_2, rest_count must be null and measure_completeness should be full.",
        "For false_measure, measure_completeness must be not_applicable, unclear_reason and rest_count must be null, and decision_debug may be null.",
        "Do not output ending finish/end measures.",
        "For the first measure of the score only, decision_debug is required unless label is false_measure. Include notehead_fill_read, stem_or_beam_read, dot_seen, note_value_read, counted_beat_units, and debug_note explaining what you saw rhythmically, what meter you used, and why you chose the label.",
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
        *_ai_prompt_multi_rest_rules("legacy"),
        *_ai_prompt_ending_rules("legacy"),
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
    score_allowed_labels = ["normal", "pickup", "false_measure", "ending_1", "ending_2"] if normalized_score_type == "score" else ["normal", "pickup", "multi_measure_rest", "false_measure", "ending_1", "ending_2"]
    score_label_shape = "normal|pickup|false_measure|ending_1|ending_2" if normalized_score_type == "score" else "normal|pickup|multi_measure_rest|false_measure|ending_1|ending_2"
    intro = {
        "job_id": str(job_id),
        "run_id": int(run_id),
        "system_id": system_id,
        "page_number": int(page_number),
        "score_type": normalized_score_type,
        "remembered_time_signature_in": None,
        "instructions": {
            "task": "Classify every already-detected sheet-music measure candidate box using the best allowed label.",
            "allowed_labels": score_allowed_labels,
            "rules": _ai_prompt_rules_for_score_type(score_type),
            "output_shape": {
                "provider": _requested_ai_provider_name(),
                "suggestions": [
                    {
                        "measure_id": "string",
                        "label": score_label_shape,
                        "measure_completeness": "full|incomplete|unclear|not_applicable",
                        "unclear_reason": "time_signature_not_clear|too_dense_to_count|crop_cut_off|split_may_be_wrong|ornament_or_tie_confusion|not_enough_visual_evidence|null",
                        "rest_count": "integer|null",
                        "confidence": "low|medium|high",
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
    false_reference_content, false_reference_count = _build_false_measure_reference_content()
    content.extend(false_reference_content)
    reference_examples_attached = int(false_reference_count)
    if normalized_score_type in ("single", "grand"):
        rest_reference_content, rest_reference_count = _build_multi_rest_reference_content()
        content.extend(rest_reference_content)
        reference_examples_attached += int(rest_reference_count)

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


def _build_ending_system_request(
    job_id: str,
    run_id: int,
    system_row: dict,
    measure_rows: list[dict],
    page,
    *,
    active_ending_in: str | None = None,
    pdf_source: str = "baseline",
    prev_system_row: dict | None = None,
    next_system_row: dict | None = None,
    score_type: str | None = None,
) -> tuple[dict, int]:
    system_id = str(system_row.get("system_id") or "").strip()
    normalized_score_type = _normalize_ai_score_type(score_type)
    scope = {
        "single": "Inspect above the single staff.",
        "grand": "Inspect above the top staff of the grand staff only.",
        "score": "Inspect above the top visible staff of the full system only.",
    }.get(normalized_score_type, "Inspect above the top staff of the system.")
    intro = {
        "job_id": str(job_id),
        "run_id": int(run_id),
        "system_id": system_id,
        "score_type": normalized_score_type,
        "active_ending_in": active_ending_in if active_ending_in in {"ending_1", "ending_2"} else None,
        "instructions": {
            "task": "Inspect every target measure for repeat-ending (volta) bracket structure.",
            "rules": [
                scope,
                "Return exactly one result for every target measure, in the supplied order.",
                "Each real crop contains the target measure plus a small amount of neighboring context. Classify only the center target measure.",
                "A repeat-ending bracket is a thin straight horizontal line above the staff, often beginning with a downward left hook and 1, 1., 1st, 2, 2., or 2nd.",
                "A start at a left barline belongs to the measure on its right. A stop at a right barline belongs to the measure on its left.",
                "A measure may contain both a start and a stop.",
                "closed_stop means a downward right hook. open_stop means the horizontal line visibly ends without a hook. Both stop the ending.",
                "continues means the bracket crosses the target measure's right boundary into the next measure.",
                "system_edge means the bracket reaches the right edge of the system and may continue on the next system or page.",
                "A continuing horizontal line without a number is not a new start.",
                "Do not treat fingering, rehearsal numbers, measure numbers, lyrics, repeat dots, slurs, ties, beams, hairpins, staff lines, or ordinary barlines as ending brackets.",
                "If a visible numbered bracket is neither Ending 1 nor Ending 2, use unsupported.",
                "Use uncertain instead of guessing when the bracket structure cannot be read.",
                "Return JSON only.",
            ],
            "output_shape": {
                "ending_measures": [
                    {
                        "measure_id": "string",
                        "start": "none|ending_1|ending_2|unsupported|uncertain",
                        "right_boundary": "none|continues|closed_stop|open_stop|system_edge|uncertain",
                        "confidence": "low|medium|high",
                        "evidence": "short visual description, maximum 20 words",
                    }
                ]
            },
        },
        "measures": [
            {
                "measure_id": str(row.get("measure_id") or "").strip(),
                "order_index_in_system": _safe_int(row.get("measure_local_index"), index),
            }
            for index, row in enumerate(measure_rows)
        ],
    }
    content: list[dict] = [{"type": "text", "text": json.dumps(intro, ensure_ascii=True)}]
    reference_content, reference_count = _build_ending_reference_content()
    content.extend(reference_content)
    for index, row in enumerate(measure_rows):
        next_row = measure_rows[index + 1] if index + 1 < len(measure_rows) else None
        crop_spec = _ending_measure_crop_spec(page.rect, row, next_row, system_row, prev_system_row, next_system_row)
        image_bytes = _render_measure_crop_png(page, crop_spec["clip"])
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "target_measure_id": str(row.get("measure_id") or "").strip(),
                        "order_index_in_system": _safe_int(row.get("measure_local_index"), index),
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
            "model": _requested_ai_model_name("ending"),
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        },
        int(reference_count),
    )


def _normalize_ending_system_response(parsed: dict, measure_rows: list[dict]) -> list[dict]:
    raw_rows = parsed.get("ending_measures") if isinstance(parsed, dict) else None
    if not isinstance(raw_rows, list):
        raise AiSuggestError(detail="malformed_response: ending_measures missing")
    expected = [str(row.get("measure_id") or "").strip() for row in measure_rows]
    if len(raw_rows) != len(expected):
        raise AiSuggestError(detail="malformed_response: ending measure count mismatch")
    normalized: list[dict] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise AiSuggestError(detail="malformed_response: ending entry must be object")
        measure_id = str(raw.get("measure_id") or "").strip()
        if measure_id != expected[index] or measure_id in seen:
            raise AiSuggestError(detail="malformed_response: ending measures must match supplied order")
        start = str(raw.get("start") or "none").strip().lower()
        boundary = str(raw.get("right_boundary") or "none").strip().lower()
        confidence = str(raw.get("confidence") or "low").strip().lower()
        if start not in AI_ENDING_START_VALUES or boundary not in AI_ENDING_BOUNDARY_VALUES:
            raise AiSuggestError(detail=f"malformed_response: invalid ending value for {measure_id}")
        if confidence not in AI_SUGGESTION_CONFIDENCE_ALLOWED:
            raise AiSuggestError(detail=f"malformed_response: invalid ending confidence for {measure_id}")
        evidence = " ".join(str(raw.get("evidence") or "").split())
        normalized.append(
            {
                "measure_id": measure_id,
                "start": start,
                "right_boundary": boundary,
                "confidence": confidence,
                "evidence": " ".join(evidence.split()[:20]),
            }
        )
        seen.add(measure_id)
    return normalized


def _ending_carry_after_system(active_ending_in: str | None, events: list[dict]) -> str | None:
    active = active_ending_in if active_ending_in in {"ending_1", "ending_2"} else None
    for event in events:
        start = str(event.get("start") or "none")
        boundary = str(event.get("right_boundary") or "none")
        if start in {"ending_1", "ending_2"}:
            active = start
        if boundary in {"closed_stop", "open_stop"}:
            active = None
        elif active and boundary == "none" and start == "none":
            active = None
    return active


def _generate_ai_endings_for_system_batch(
    job_id: str,
    run_id: int,
    systems: list[dict] | None,
    system_row: dict,
    system_measures: list[dict],
    artifacts: dict,
    *,
    active_ending_in: str | None,
    score_type: str | None,
) -> dict:
    model_name = _requested_ai_model_name("ending")
    if not model_name or model_name == "unknown":
        raise AiSuggestError(provider_status=503, detail="ending_provider_not_configured")
    with TemporaryDirectory(prefix="omr-ai-ending-step-") as tmp:
        in_pdf, pdf_source = _resolve_ai_crop_pdf_source(artifacts, Path(tmp))
        doc = fitz.open(str(in_pdf))
        try:
            ordered_systems = _sorted_system_rows(systems or [])
            prev_system_row, next_system_row = _same_page_neighbor_systems(ordered_systems, system_row)
            page_number = _safe_int(system_row.get("page"), _safe_int(system_measures[0].get("page"), 1))
            page_index = max(0, int(page_number) - 1)
            if page_index >= len(doc):
                raise AiSuggestError(provider_status=500, detail=f"invalid_page_index:{page_number}")
            payload, reference_count = _build_ending_system_request(
                job_id,
                run_id,
                system_row,
                system_measures,
                doc[page_index],
                active_ending_in=active_ending_in,
                pdf_source=pdf_source,
                prev_system_row=prev_system_row,
                next_system_row=next_system_row,
                score_type=score_type,
            )
            last_error: AiSuggestError | None = None
            for attempt in (1, 2):
                try:
                    message = _ai_messages_create(payload)
                    events = _normalize_ending_system_response(_parse_anthropic_suggestions_message(message), system_measures)
                    return {
                        "version": AI_SUGGESTIONS_ENDING_VERSION,
                        "provider": _requested_ai_provider_name(),
                        "model": model_name,
                        "system_id": str(system_row.get("system_id") or ""),
                        "events": events,
                        "active_ending_out": _ending_carry_after_system(active_ending_in, events),
                        "reference_examples_attached": reference_count,
                        "_internal_ai_usage": _ai_usage_from_message(message),
                        "retry_attempts": attempt,
                    }
                except AiSuggestError as exc:
                    last_error = exc
                    if attempt == 1 and "malformed_response" in str(exc.detail or ""):
                        logger.warning("AI_ENDING_MALFORMED_RETRY system_id=%s", str(system_row.get("system_id") or ""))
                        continue
                    raise
            raise last_error or AiSuggestError(detail="ending_pass_failed")
        finally:
            doc.close()


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
    response_diagnostics: dict | None = None

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
            response_diagnostics = _ai_general_response_diagnostics(
                message,
                system_id=str(system_row.get("system_id") or ""),
                measure_count=len(system_measures),
                reference_count=reference_examples_attached,
                model=model_name,
            )
            _log_ai_general_response_debug(response_diagnostics)
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
            normalized["_internal_ai_usage"] = _ai_usage_from_message(message)
            debug_crops = _finalize_debug_crops()
            if debug_crops is not None:
                normalized["debug_crops"] = debug_crops
            return normalized
        except AiSuggestError as exc:
            if response_diagnostics is not None:
                _log_ai_general_parse_failed(_ai_general_failure_diagnostics(exc, response_diagnostics))
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
    layout = _measure_label_layout_left_barline(page, page_rect, x_left, y_top, text)
    if layout is not None:
        _draw_measure_label_layout(page, layout, text)


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
        if conclusion == "cancelled":
            return "cancelled"
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


def _cancel_github_run(run_id: int) -> str:
    try:
        run = _get_run(int(run_id))
        if str(run.get("status") or "").strip().lower() == "completed":
            return "already_completed"
    except GitHubAPIError as exc:
        if exc.status_code == 404:
            return "run_not_found"
        raise

    try:
        _gh_request(
            "POST",
            f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{int(run_id)}/cancel",
        )
        return "cancel_requested"
    except GitHubAPIError as exc:
        if exc.status_code in {409, 422}:
            return "already_finished"
        raise


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


def _reassign_measures_to_nearest_system(
    systems: list[dict],
    measures: list[dict],
    *,
    skip_pages: set[int] | None = None,
) -> int:
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
    protected_pages = skip_pages or set()
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
        if m_page in protected_pages:
            m.pop("_system_reassigned", None)
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
            if _row_source(m) == ROW_SOURCE_MANUAL and not str(m.get("measure_id") or "").strip():
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
    authoritative_pages = set(_editable_auto_rows_authoritative_pages(editable_state))
    reassign_count = _reassign_measures_to_nearest_system(
        systems,
        measures,
        skip_pages=authoritative_pages,
    )
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
    valid_measure_ids = {
        str(measure.get("measure_id") or "").strip()
        for measure in measures
        if str(measure.get("measure_id") or "").strip()
    }
    _editable_forced_label_ids(editable_state, valid_measure_ids)
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
    first_counted_measure_id = next(
        (
            str(measure.get("measure_id") or "").strip()
            for measure in ordered_measures
            if not _is_excluded_from_counting(measure)
            and str(measure.get("measure_id") or "").strip()
        ),
        "",
    )
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
            is_opening_pickup = measure_id == first_counted_measure_id
            label_value = 0 if is_opening_pickup else int(current_value) - 1
            _apply_measure_label(
                measure,
                measure_id,
                system_id,
                str(label_value),
                result_labels,
                seq_starts_by_system,
            )
            next_value = 1 if is_opening_pickup else int(current_value)
            current_value = _apply_post_measure_rest(
                next_value,
                label_value,
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
    authoritative_pages = set(_editable_auto_rows_authoritative_pages(editable_state))
    authoritative_pages.add(int(page))
    editable_state["auto_rows_authoritative_pages"] = sorted(authoritative_pages)
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


def _apply_hide_label_edit(
    raw_edit: dict,
    measure_ids: set[str],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    raw_label_id = raw_edit.get("label_id")
    if raw_label_id is None:
        raw_label_id = raw_edit.get("value")
    label_id = str(raw_label_id or "").strip()
    if not label_id.startswith("label:"):
        rejected.append({"edit": raw_edit, "reason": "invalid_label_id"})
        return
    measure_id = label_id[len("label:") :].strip()
    if not measure_id or measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "invalid_label_id"})
        return

    hidden_ids = _editable_hidden_label_ids(editable_state)
    if label_id not in hidden_ids:
        hidden_ids.append(label_id)
        hidden_ids.sort()
        editable_state["hidden_label_ids"] = hidden_ids

    applied.append({"type": "hide_label", "label_id": label_id, "measure_id": measure_id})


def _apply_show_label_edit(
    raw_edit: dict,
    measure_ids: set[str],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    measure_id = str(raw_edit.get("measure_id") or "").strip()
    raw_label_id = raw_edit.get("label_id")
    if raw_label_id is None:
        raw_label_id = raw_edit.get("value")
    label_id = str(raw_label_id or "").strip()
    if not measure_id and label_id.startswith("label:"):
        measure_id = label_id[len("label:") :].strip()
    if not measure_id or measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "invalid_measure_id"})
        return

    label_id = f"label:{measure_id}"
    forced_ids = _editable_forced_label_ids(editable_state)
    if label_id not in forced_ids:
        forced_ids.append(label_id)
        forced_ids.sort()
        editable_state["forced_label_ids"] = forced_ids

    hidden_ids = _editable_hidden_label_ids(editable_state)
    if label_id in hidden_ids:
        hidden_ids.remove(label_id)
        editable_state["hidden_label_ids"] = hidden_ids

    positions = _editable_label_positions(editable_state)
    positions.pop(label_id, None)
    editable_state["label_positions"] = positions
    applied.append({"type": "show_label", "label_id": label_id, "measure_id": measure_id})


def _apply_move_label_edit(
    raw_edit: dict,
    measure_ids: set[str],
    editable_state: dict,
    applied: list[dict],
    rejected: list[dict],
) -> None:
    raw_label_id = raw_edit.get("label_id")
    if raw_label_id is None:
        raw_label_id = raw_edit.get("value")
    label_id = str(raw_label_id or "").strip()
    if not label_id.startswith("label:"):
        rejected.append({"edit": raw_edit, "reason": "invalid_label_id"})
        return
    measure_id = label_id[len("label:") :].strip()
    if not measure_id or measure_id not in measure_ids:
        rejected.append({"edit": raw_edit, "reason": "invalid_label_id"})
        return

    page = _safe_int(raw_edit.get("page"), 0)
    try:
        left = float(raw_edit.get("left"))
        top = float(raw_edit.get("top"))
    except Exception:
        rejected.append({"edit": raw_edit, "reason": "invalid_label_position"})
        return
    if page <= 0 or not math.isfinite(left) or not math.isfinite(top) or left < 0 or top < 0 or left > 5000 or top > 5000:
        rejected.append({"edit": raw_edit, "reason": "invalid_label_position"})
        return

    positions = _editable_label_positions(editable_state)
    positions[label_id] = {"page": page, "left": round(left, 3), "top": round(top, 3)}
    editable_state["label_positions"] = positions
    applied.append(
        {
            "type": "move_label",
            "label_id": label_id,
            "measure_id": measure_id,
            "page": page,
            "left": round(left, 3),
            "top": round(top, 3),
        }
    )


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

        if edit_type == "hide_label":
            _apply_hide_label_edit(raw_edit, measure_ids, editable_state, applied, rejected)
            continue

        if edit_type == "show_label":
            _apply_show_label_edit(raw_edit, measure_ids, editable_state, applied, rejected)
            continue

        if edit_type == "move_label":
            _apply_move_label_edit(raw_edit, measure_ids, editable_state, applied, rejected)
            continue

        rejected.append({"edit": raw_edit, "reason": "unsupported_edit_type"})

    editable_state["measure_number_overrides"] = measure_overrides
    editable_state["labels_mode"] = labels_mode
    _editable_label_erase_areas(editable_state)
    _editable_hidden_label_ids(editable_state)
    _editable_forced_label_ids(editable_state)
    _editable_label_positions(editable_state)
    systems, measures, _, _ = _refresh_editable_state_systems_and_measures(
        editable_state,
        ending_debug_ctx=ending_debug_ctx,
    )
    return systems, applied, rejected, len(systems)


def _label_render_rows(
    labels_mode: str,
    sorted_systems: list[dict],
    ordered_measures: list[dict],
    result_labels: dict[str, str],
    forced_label_ids: set[str],
) -> list[tuple[dict, str]]:
    render_rows: list[tuple[dict, str]] = []
    seen_measure_ids: set[str] = set()
    if labels_mode == LABELS_MODE_ALL_MEASURES:
        normal_rows = [
            (measure, result_labels.get(str(measure.get("measure_id") or "").strip()) or "")
            for measure in ordered_measures
        ]
    else:
        normal_rows = _system_start_anchor_measures(ordered_measures, result_labels, sorted_systems)
    for measure, label in normal_rows:
        measure_id = str(measure.get("measure_id") or "").strip()
        if not measure_id or not label or measure_id in seen_measure_ids:
            continue
        render_rows.append((measure, label))
        seen_measure_ids.add(measure_id)
    for measure in ordered_measures:
        measure_id = str(measure.get("measure_id") or "").strip()
        label = result_labels.get(measure_id) or ""
        if f"label:{measure_id}" not in forced_label_ids or not label or measure_id in seen_measure_ids:
            continue
        render_rows.append((measure, label))
        seen_measure_ids.add(measure_id)
    return render_rows


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
    label_boxes: list[dict] = []
    hidden_label_ids = set(_editable_hidden_label_ids(editable_state))
    forced_label_ids = set(_editable_forced_label_ids(editable_state))
    label_positions = _editable_label_positions(editable_state)

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

    render_rows = _label_render_rows(
        labels_mode,
        sorted_systems,
        ordered_measures,
        result_labels,
        forced_label_ids,
    )

    for measure, label in render_rows:
        measure_id = str(measure.get("measure_id") or "").strip()
        page_no = _safe_int(measure.get("page"), 0)
        if page_no <= 0 or page_no > doc.page_count:
            continue
        try:
            x_left = float(measure.get("x_left"))
            y_top = float(measure.get("y_top"))
        except Exception:
            continue
        page = doc[page_no - 1]
        layout = _measure_label_layout_left_barline(page, page.rect, x_left, y_top, label)
        label_id = f"label:{measure_id}"
        saved_position = label_positions.get(label_id)
        if isinstance(saved_position, dict):
            layout = _measure_label_layout_at_top_left(
                page.rect,
                saved_position.get("left", 0.0),
                saved_position.get("top", 0.0),
                label,
            )
        if layout is None:
            continue
        hidden = label_id in hidden_label_ids
        if not hidden:
            _draw_measure_label_layout(page, layout, label)
            drawn += 1
        label_box = _label_box_from_layout(measure, label, layout, hidden=hidden)
        if label_box is not None:
            label_boxes.append(label_box)

    editable_state["label_boxes"] = label_boxes
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()
    return drawn


@app.route("/", methods=["GET"])
def health():
    return "omr-worker is running", 200


@app.route("/privacy", methods=["GET"])
def privacy_policy():
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sheet Music Labeler Privacy Policy</title>
<style>body{font:16px -apple-system,BlinkMacSystemFont,sans-serif;line-height:1.55;max-width:760px;margin:40px auto;padding:0 20px;color:#202124}h1,h2{line-height:1.2}h2{margin-top:28px}small{color:#666}</style>
</head><body><h1>Sheet Music Labeler Privacy Policy</h1><small>Last updated July 17, 2026</small>
<p>Sheet Music Labeler processes sheet-music PDFs so it can detect measures, apply labels, provide manual editing, and, when enabled, generate AI suggestions.</p>
<h2>Information we process</h2><p>We process PDFs you upload, generated output files, job and error diagnostics, random device identifiers used for Friend or paid access, Apple transaction identifiers, subscription status, and AI credit usage. We do not collect or store your payment-card information.</p>
<h2>How information is used</h2><p>We use this information only to provide and secure the app, restore jobs, verify access, count AI usage, diagnose failures, and prevent duplicate purchases or charges.</p>
<h2>Service providers</h2><p>Processing may use Apple, Google Cloud, GitHub Actions, and Amazon Web Services, including Amazon Bedrock for AI features. These providers process data as needed to deliver their services.</p>
<h2>Storage and sharing</h2><p>We do not sell personal information. Files and operational records are retained only as needed to provide the service, restore results, protect purchases, and troubleshoot the app. Access records do not contain your plain Friend Code or plain access token.</p>
<h2>Your choices</h2><p>You can avoid AI processing by not using AI Suggestions. You may request help with, or deletion of, associated service data by contacting us.</p>
<h2>Contact</h2><p>Email <a href="mailto:suggestions.pineapple@gmail.com">suggestions.pineapple@gmail.com</a>.</p>
</body></html>""",
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "public, max-age=3600"},
    )


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
    requested_job_id = _ensure_unique_job_id(requested_job_id or str(uuid.uuid4()), allow_same=True)
    existing_rec = _pending_record(requested_job_id) or _job_store_get(requested_job_id)
    if isinstance(existing_rec, dict) and str(existing_rec.get("pdf_gcs_uri") or "").strip() == pdf_gcs_uri:
        run_id = _safe_int(existing_rec.get("run_id"), 0)
        artifact_key = _job_artifact_key(requested_job_id, run_id if run_id > 0 else None, existing_rec)
        _pending_set(requested_job_id, existing_rec)
        response = {
            "job_id": requested_job_id,
            "artifact_key": artifact_key,
            "status": str(existing_rec.get("status") or "queued"),
            "run_id": run_id if run_id > 0 else None,
            "workflow": existing_rec.get("workflow_id") or existing_rec.get("workflow") or GITHUB_WORKFLOW_ID,
            "ref": existing_rec.get("ref") or GITHUB_REF,
            "pdf_gcs_uri": pdf_gcs_uri,
            "status_url": f"/api/omr/jobs/{requested_job_id}",
            "warning": "reused existing job for repeated create request",
        }
        if run_id > 0:
            artifacts = _artifact_uris_for_existing_run(int(run_id), artifact_key=artifact_key)
            response["run_url"] = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}"
            response["artifacts"] = artifacts
            response["artifacts_http"] = _artifact_http_uris_for_run(int(run_id), artifacts)
            response["storage_mode"] = _storage_mode_for_artifacts(artifacts)
        return jsonify(response), 202
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
    durable_job_saved = _job_store_upsert(
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
    if not durable_job_saved:
        response["warning"] = "job started but durable job-store save failed; in-memory tracking is active"

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
            if str(rec.get("status") or "").strip().lower() == "cancelled":
                return jsonify(
                    {
                        "job_id": job_id,
                        "status": "cancelled",
                        "run_id": None,
                        "workflow": rec.get("workflow_id") or GITHUB_WORKFLOW_ID,
                        "ref": rec.get("ref") or GITHUB_REF,
                        "status_url": f"/api/omr/jobs/{job_id}",
                    }
                ), 200
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
        "status": "cancelled" if isinstance(rec, dict) and str(rec.get("status") or "").strip().lower() == "cancelled" else _run_public_status(run),
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


@app.route("/api/omr/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err and rec is None:
        return jsonify({"error": err, "job_id": job_id}), 404

    github_cancel_status = "not_started"
    if isinstance(run_id, int):
        try:
            github_cancel_status = _cancel_github_run(int(run_id))
        except GitHubAPIError as exc:
            return (
                jsonify(
                    {
                        "error": exc.message,
                        "job_id": job_id,
                        "run_id": int(run_id),
                        "status": "cancel_failed",
                        "status_code": exc.status_code,
                    }
                ),
                exc.status_code if 400 <= exc.status_code <= 599 else 502,
            )

    now_txt = _to_utc_z(_utc_now())
    merged = dict(rec or {})
    merged.update(
        {
            "status": "cancelled",
            "cancelled_at_utc": now_txt,
            "run_id": int(run_id) if isinstance(run_id, int) else None,
        }
    )
    _pending_set(job_id, merged)
    _job_store_upsert(job_id, merged)

    if isinstance(run_id, int):
        try:
            artifact_key = _job_artifact_key(job_id, int(run_id), merged)
            artifacts, mapping_summary, artifact_run_id = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
            ai_run = mapping_summary.get("ai_suggest_run")
            if isinstance(ai_run, dict) and str(ai_run.get("status") or "").strip().lower() == AI_SUGGEST_RUN_STATUS_RUNNING:
                ai_run["status"] = AI_SUGGEST_RUN_STATUS_CANCELLED
                ai_run["updated_at_utc"] = now_txt
                ai_run["cancelled_at_utc"] = now_txt
                mapping_summary["ai_suggest_run"] = ai_run
                _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        except Exception as exc:
            print(f"CANCEL_AI_STATE_WARN job_id={job_id} detail={_safe_error_text(exc)}")

    return jsonify(
        {
            "job_id": job_id,
            "run_id": int(run_id) if isinstance(run_id, int) else None,
            "status": "cancelled",
            "github_cancel_status": github_cancel_status,
        }
    ), 200


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
                    "error": {
                        "code": "resume_artifacts_mismatch",
                        "message": "requested job_id does not match single-latest artifacts",
                    },
                    "job_id": job_id,
                    "requested_run_id": exc.requested_run_id,
                    "artifact_run_id": exc.artifact_run_id,
                }
            ),
            409,
        )
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": {
                        "code": "resume_artifacts_missing",
                        "message": "Previous job artifacts are no longer available.",
                    },
                    "job_id": job_id,
                    "run_id": run_id,
                }
            ),
            410,
        )
    except Exception as exc:
        return jsonify({"error": f"failed to load state: {exc}", "job_id": job_id, "run_id": run_id}), 502

    editable_state = mapping_summary.get("editable_state")
    if not isinstance(editable_state, dict):
        return (
            jsonify(
                {
                    "error": {
                        "code": "resume_state_unusable",
                        "message": "Previous job state is no longer usable.",
                    },
                    "job_id": job_id,
                    "run_id": run_id,
                }
            ),
            410,
        )

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
            "auto_rows_authoritative_pages": _editable_auto_rows_authoritative_pages(editable_state),
            "manual_rows": editable_state.get("manual_rows") or [],
            "rest_measures": editable_state.get("rest_measures") or {},
            "pickup_measures": editable_state.get("pickup_measures") or {},
            "label_erase_areas": _editable_label_erase_areas(editable_state),
            "hidden_label_ids": _editable_hidden_label_ids(editable_state),
            "forced_label_ids": _editable_forced_label_ids(editable_state),
            "label_positions": _editable_label_positions(editable_state),
            "label_boxes": _editable_label_boxes(editable_state),
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


@app.route("/api/access/friend/activate", methods=["POST"])
def activate_friend_access():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        result = _friend_activate_device(payload.get("device_id"), payload.get("code"))
    except FriendAccessError as exc:
        return _friend_error_response(exc)
    return (
        jsonify(
            {
                "friend_access": {
                    "active": True,
                    "friend_id": result["friend_id"],
                },
                "access_token": result["access_token"],
            }
        ),
        200,
    )


@app.route("/api/access/status", methods=["GET"])
def get_access_status():
    try:
        result = _friend_verify_token(_friend_bearer_token())
    except FriendAccessError as exc:
        return _friend_error_response(exc)
    return jsonify({"friend_access": {"active": True, "friend_id": result["friend_id"]}}), 200


@app.route("/api/access/paid/verify", methods=["POST"])
def verify_paid_access():
    if not _apple_iap_enabled():
        return _paid_error_response(
            PaidAccessError(
                "apple_purchase_not_enabled",
                "Paid subscriptions are not available yet.",
                503,
                retryable=False,
            )
        )
    payload = request.get_json(silent=True) or {}
    try:
        decoded = _apple_verify_transaction(payload.get("signed_transaction"))
        identity = _verified_apple_identity(payload.get("signed_app_transaction"), decoded)
        result = _paid_apply_transaction(
            decoded,
            app_transaction_id=identity["app_transaction_id"],
            device_id=payload.get("device_id"),
            issue_token=True,
        )
    except PaidAccessError as exc:
        return _paid_error_response(exc)
    return (
        jsonify(
            {
                "paid_access": {
                    "active": bool(result["active"]),
                    "paid_id": result["paid_id"],
                    "plan": result["plan"],
                    "plan_display_name": result["plan_display_name"],
                    "monthly_credit_capacity": result["monthly_credit_capacity"],
                    "status": result["status"],
                    "credits_remaining": result["credits_remaining"],
                    "purchased_credits_remaining": result["purchased_credits_remaining"],
                    "expires_at_utc": result["expires_at_utc"],
                },
                "paid_access_token": result["access_token"],
            }
        ),
        200,
    )


@app.route("/api/access/paid/status", methods=["GET"])
def get_paid_access_status():
    try:
        result = _paid_verify_token(_paid_header_token(), allow_empty=True)
    except PaidAccessError as exc:
        return _paid_error_response(exc)
    return (
        jsonify(
            {
                "paid_access": {
                    "active": bool(result.get("pro_active")),
                    "paid_id": result["paid_id"],
                    "plan": result.get("plan"),
                    "plan_display_name": result.get("plan_display_name"),
                    "monthly_credit_capacity": result.get("monthly_credit_capacity", 0),
                    "status": result.get("subscription_status"),
                    "credits_remaining": result["pro_credits_remaining"],
                    "purchased_credits_remaining": result["purchased_credits_remaining"],
                    "expires_at_utc": result["expires_at_utc"],
                }
            }
        ),
        200,
    )


@app.route("/api/access/credits/status", methods=["GET"])
def get_combined_credit_status():
    try:
        result = _combined_credit_status()
    except FriendAccessError as exc:
        return _friend_error_response(exc)
    except PaidAccessError as exc:
        return _paid_error_response(exc)
    return jsonify({"credits": result}), 200


@app.route("/api/access/packs/verify", methods=["POST"])
def verify_credit_pack():
    if not _apple_packs_enabled():
        return _paid_error_response(PaidAccessError("apple_packs_not_enabled", "Credit packs are not available yet.", 503))
    payload = request.get_json(silent=True) or {}
    try:
        decoded = _apple_verify_transaction(payload.get("signed_transaction"))
        identity = _verified_apple_identity(payload.get("signed_app_transaction"), decoded)
        result = _pack_apply_transaction(
            decoded,
            app_transaction_id=identity["app_transaction_id"],
            device_id=payload.get("device_id"),
        )
    except PaidAccessError as exc:
        return _paid_error_response(exc)
    return jsonify({"pack_purchase": result, "paid_access_token": result["access_token"]}), 200


@app.route("/api/access/apple/restore", methods=["POST"])
def restore_apple_wallet():
    payload = request.get_json(silent=True) or {}
    try:
        identity = _verified_apple_identity(payload.get("signed_app_transaction"))
        result = _paid_restore_wallet(
            app_transaction_id=identity["app_transaction_id"],
            device_id=payload.get("device_id"),
        )
    except PaidAccessError as exc:
        return _paid_error_response(exc)
    return jsonify({"paid_access": result, "paid_access_token": result["access_token"]}), 200


@app.route("/api/apple/app-store/notifications", methods=["POST"])
def apple_app_store_notification():
    if not (_apple_iap_enabled() or _apple_packs_enabled()):
        return jsonify({"status": "disabled"}), 503
    payload = request.get_json(silent=True) or {}
    try:
        notification = _apple_verify_notification(payload.get("signedPayload"))
        notification_type = _apple_text(_apple_field(notification, "notificationType")).upper()
        notification_data = _apple_field(notification, "data")
        signed_transaction = _apple_field(notification_data, "signedTransactionInfo")
        if not signed_transaction:
            return jsonify({"status": "ignored", "reason": "no_transaction"}), 200
        transaction = _apple_verify_transaction(signed_transaction)
        product_id = _apple_text(_apple_field(transaction, "productId"))
        if product_id in APPLE_CREDIT_PACKS:
            if notification_type in {"REFUND", "REVOKE"}:
                result = _pack_refund_transaction(transaction)
            else:
                result = {"status": "ignored_pack_event"}
        elif notification_type in {"SUBSCRIBED", "DID_RENEW", "OFFER_REDEEMED", "DID_CHANGE_RENEWAL_PREF"}:
            result = _paid_apply_transaction(transaction)
        else:
            grace_expires_at = None
            signed_renewal = _apple_field(notification_data, "signedRenewalInfo")
            if signed_renewal:
                for environment_name in ("production", "sandbox"):
                    try:
                        renewal = _apple_verifier(environment_name).verify_and_decode_renewal_info(signed_renewal)
                        grace_expires_at = _apple_millis_time(_apple_field(renewal, "gracePeriodExpiresDate"))
                        break
                    except Exception:
                        continue
            result = _paid_update_notification_state(
                transaction,
                notification_type=notification_type,
                grace_expires_at=grace_expires_at,
            )
    except PaidAccessError as exc:
        return _paid_error_response(exc)
    return jsonify({"status": "ok", "notification_type": notification_type, "paid_id": result.get("paid_id")}), 200


@app.route("/api/omr/jobs/<job_id>/ai-suggest", methods=["POST"])
def ai_suggest_job(job_id: str):
    started = time.time()
    try:
        _ai_access()
    except FriendAccessError as exc:
        return _friend_error_response(exc)
    except PaidAccessError as exc:
        return _paid_error_response(exc)

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
    raw_mode = request_payload.get("mode")
    explicit_mode = raw_mode is not None
    mode = str(raw_mode or "").strip().lower() if explicit_mode else None
    if explicit_mode and mode not in AI_SUGGEST_START_MODES_ALLOWED:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "invalid_ai_start_mode",
                        "message": "mode must be start, continue, or restart",
                        "retryable": False,
                        "provider_status": 400,
                        "detail": "invalid_ai_start_mode",
                    },
                }
            ),
            400,
        )
    request_id = _normalize_ai_start_request_id(request_payload.get("request_id"))
    if mode in {"start", "restart"} and request_id is None:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "invalid_ai_request_id",
                        "message": "A stable request ID is required to start AI suggestions.",
                        "retryable": False,
                        "provider_status": 400,
                        "detail": "invalid_ai_request_id",
                    },
                }
            ),
            400,
        )
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
    if mode in {"start", "restart"} and score_type is None:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(run_id),
                    "status": "failed",
                    "error": {
                        "code": "score_type_required",
                        "message": "Choose Single Staff, Grand Staff, or Full Score.",
                        "retryable": False,
                        "provider_status": 400,
                        "detail": "score_type_required",
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
    existing_ai_suggestions = _current_ai_suggestions(mapping_summary)
    existing_ai_suggest_run = _current_ai_suggest_run(mapping_summary, int(artifact_run_id), source_state_version)
    system_batches = _ai_suggest_system_batches(editable_state)
    if (
        mode in {"start", "restart"}
        and request_id
        and str(existing_ai_suggest_run.get("start_request_id") or "") == request_id
        and existing_ai_suggest_run.get("execution_id")
    ):
        return jsonify(
            {
                "job_id": job_id,
                "run_id": int(artifact_run_id),
                "status": str(existing_ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_RUNNING),
                "ai_suggestions": existing_ai_suggestions,
                "ai_suggest_run": existing_ai_suggest_run,
                "storage_mode": _storage_mode_for_artifacts(artifacts),
                "artifacts": artifacts,
                "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
                "duration_ms": int((time.time() - started) * 1000),
            }
        ), 200

    if mode == "continue":
        if not existing_ai_suggest_run.get("can_continue") or not isinstance(existing_ai_suggestions, dict):
            return (
                jsonify(
                    {
                        "job_id": job_id,
                        "run_id": int(artifact_run_id),
                        "status": str(existing_ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_FAILED),
                        "ai_suggestions": existing_ai_suggestions,
                        "ai_suggest_run": existing_ai_suggest_run,
                        "error": {
                            "code": "ai_restart_required",
                            "message": "This AI run cannot continue. Restart AI Suggestions instead.",
                            "retryable": False,
                            "provider_status": 409,
                            "detail": "restart_required",
                        },
                    }
                ),
                409,
            )
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        existing_ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_RUNNING
        existing_ai_suggest_run["updated_at_utc"] = now_txt
        existing_ai_suggest_run["failed_at_utc"] = None
        existing_ai_suggest_run["cancelled_at_utc"] = None
        existing_ai_suggest_run["last_error"] = None
        existing_ai_suggest_run["systems_total"] = len(system_batches)
        _refresh_ai_run_recovery_flags(existing_ai_suggest_run, source_state_version)
        mapping_summary["ai_suggestions"] = existing_ai_suggestions
        mapping_summary["ai_suggest_run"] = existing_ai_suggest_run
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        return jsonify(
            {
                "job_id": job_id,
                "run_id": int(artifact_run_id),
                "status": AI_SUGGEST_RUN_STATUS_RUNNING,
                "ai_suggestions": existing_ai_suggestions,
                "ai_suggest_run": existing_ai_suggest_run,
                "storage_mode": _storage_mode_for_artifacts(artifacts),
                "artifacts": artifacts,
                "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
                "duration_ms": int((time.time() - started) * 1000),
            }
        ), 200

    if mode == "start" and str(existing_ai_suggest_run.get("status") or "") in {
        AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED,
        AI_SUGGEST_RUN_STATUS_CANCELLED,
        AI_SUGGEST_RUN_STATUS_FAILED,
    }:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": str(existing_ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_FAILED),
                    "ai_suggestions": existing_ai_suggestions,
                    "ai_suggest_run": existing_ai_suggest_run,
                    "error": {
                        "code": "ai_recovery_choice_required",
                        "message": "Choose Continue or Restart for the unfinished AI run.",
                        "retryable": False,
                        "provider_status": 409,
                        "detail": "recovery_choice_required",
                    },
                }
            ),
            409,
        )

    if mode == "restart" and str(existing_ai_suggest_run.get("status") or "") == AI_SUGGEST_RUN_STATUS_COMPLETED:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": AI_SUGGEST_RUN_STATUS_COMPLETED,
                    "ai_suggestions": existing_ai_suggestions,
                    "ai_suggest_run": existing_ai_suggest_run,
                    "error": {
                        "code": "ai_run_completed",
                        "message": "This AI run is already complete.",
                        "retryable": False,
                        "provider_status": 409,
                        "detail": "completed",
                    },
                }
            ),
            409,
        )

    if mode == "restart":
        try:
            _reconcile_ai_restart_credit_groups(existing_ai_suggest_run, job_id)
        except FriendAccessError as exc:
            return _friend_error_response(exc)
        except PaidAccessError as exc:
            return _paid_error_response(exc)
        except AiSuggestError as exc:
            error_payload = _ai_suggest_error_payload(exc)
            return jsonify(
                {
                    "job_id": job_id,
                    "run_id": int(artifact_run_id),
                    "status": str(existing_ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_FAILED),
                    "ai_suggestions": existing_ai_suggestions,
                    "ai_suggest_run": existing_ai_suggest_run,
                    "error": error_payload,
                }
            ), int(error_payload.get("provider_status") or 503)

    if (
        not explicit_mode
        and existing_ai_suggest_run.get("status") == AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED
        and isinstance(existing_ai_suggestions, dict)
    ):
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        existing_ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_RUNNING
        existing_ai_suggest_run["updated_at_utc"] = now_txt
        existing_ai_suggest_run["failed_at_utc"] = None
        existing_ai_suggest_run["last_error"] = None
        existing_ai_suggest_run["systems_total"] = len(system_batches)
        if score_type:
            existing_ai_suggest_run["score_type"] = score_type
        mapping_summary["ai_suggestions"] = existing_ai_suggestions
        mapping_summary["ai_suggest_run"] = existing_ai_suggest_run
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        return jsonify(
            {
                "job_id": job_id,
                "run_id": int(artifact_run_id),
                "status": AI_SUGGEST_RUN_STATUS_RUNNING,
                "ai_suggestions": existing_ai_suggestions,
                "ai_suggest_run": existing_ai_suggest_run,
                "storage_mode": _storage_mode_for_artifacts(artifacts),
                "artifacts": artifacts,
                "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
                "duration_ms": int((time.time() - started) * 1000),
            }
        ), 200

    mapping_summary["ai_suggestions"] = _empty_ai_suggestions_state(
        int(artifact_run_id),
        source_state_version,
        len(_ai_suggest_candidate_measures(editable_state)),
    )
    debug_batch_trace = None
    run_status = AI_SUGGEST_RUN_STATUS_RUNNING if system_batches else AI_SUGGEST_RUN_STATUS_COMPLETED
    mapping_summary["ai_suggest_run"] = _new_ai_suggest_run_state(
        int(artifact_run_id),
        source_state_version,
        len(system_batches),
        status=run_status,
        score_type=score_type,
        execution_id=(
            _ai_execution_id(job_id, int(artifact_run_id), source_state_version, request_id)
            if request_id
            else None
        ),
        start_request_id=request_id,
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


@app.route("/api/omr/jobs/<job_id>/ai-suggest/cancel", methods=["POST"])
def cancel_ai_suggest_job(job_id: str):
    started = time.time()
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": {
                        "code": "ai_suggest_cancel_failed",
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
                        "code": "ai_suggest_cancel_failed",
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
                        "code": "ai_suggest_cancel_failed",
                        "message": "failed to load AI suggestion state",
                        "retryable": True,
                        "provider_status": 502,
                        "detail": _safe_error_text(exc),
                    },
                }
            ),
            502,
        )

    editable_state = mapping_summary.get("editable_state") or {}
    source_state_version = _editable_state_version(editable_state) if isinstance(editable_state, dict) else None
    ai_suggestions = _current_ai_suggestions(mapping_summary)
    ai_suggest_run = _current_ai_suggest_run(mapping_summary, int(artifact_run_id), source_state_version)
    current_status = str(ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_IDLE)

    if current_status not in {AI_SUGGEST_RUN_STATUS_COMPLETED, AI_SUGGEST_RUN_STATUS_CANCELLED}:
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_CANCELLED
        ai_suggest_run["updated_at_utc"] = now_txt
        ai_suggest_run["cancelled_at_utc"] = now_txt
        _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
        mapping_summary["ai_suggest_run"] = ai_suggest_run
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
                            "code": "ai_suggest_cancel_failed",
                            "message": "failed to save stopped AI suggestion state",
                            "retryable": True,
                            "provider_status": 500,
                            "detail": _safe_error_text(exc),
                        },
                    }
                ),
                500,
            )

    response = {
        "job_id": job_id,
        "run_id": int(artifact_run_id),
        "status": str(ai_suggest_run.get("status") or AI_SUGGEST_RUN_STATUS_CANCELLED),
        "ai_suggestions": ai_suggestions,
        "ai_suggest_run": ai_suggest_run,
        "storage_mode": _storage_mode_for_artifacts(artifacts),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(int(artifact_run_id), artifacts),
        "duration_ms": int((time.time() - started) * 1000),
    }
    return jsonify(response), 200


def _step_ai_suggest_two_system_credit_v1(
    *,
    job_id: str,
    artifact_run_id: int,
    mapping_summary: dict,
    artifacts: dict,
    systems: list[dict],
    measures: list[dict],
    system_batches: list[tuple[dict, list[dict]]],
    source_state_version: str | None,
    ai_suggestions: dict,
    ai_suggest_run: dict,
    next_system_index: int,
    started: float,
) -> tuple:
    systems_total = len(system_batches)
    system_row, system_measures = system_batches[next_system_index]
    system_id = str(system_row.get("system_id") or "").strip()
    charge_id = _ai_credit_group_id(
        job_id,
        artifact_run_id,
        source_state_version,
        next_system_index,
        execution_id=ai_suggest_run.get("execution_id"),
    )
    pass_states = dict(ai_suggest_run.get("pass_state_by_system_id") or {})
    pass_state = dict(pass_states.get(system_id) or {})
    credit_groups = dict(ai_suggest_run.get("credit_groups") or {})
    credit_group = dict(credit_groups.get(charge_id) or {})
    credit_group.setdefault("group_index", next_system_index // 2)
    credit_group.setdefault("system_ids", [
        str(system_batches[index][0].get("system_id") or "")
        for index in range((next_system_index // 2) * 2, min(systems_total, (next_system_index // 2) * 2 + 2))
    ])

    try:
        ai_access = _ai_access(
            reserve=not bool(credit_group.get("charged")),
            job_id=job_id,
            system_id=system_id or None,
            charge_id=charge_id,
        )
    except FriendAccessError as exc:
        return _friend_error_response(exc)
    except PaidAccessError as exc:
        return _paid_error_response(exc)

    score_type = _normalize_ai_score_type(ai_suggest_run.get("score_type"))
    remembered_time_signature_in = _normalize_ai_time_signature_value(ai_suggest_run.get("remembered_time_signature"))
    debug_crops = None
    reference_examples_attached = 0

    if pass_state.get("general") != "completed":
        try:
            system_result = _generate_ai_suggestions_for_system_batch(
                job_id,
                artifact_run_id,
                systems,
                system_row,
                system_measures,
                source_state_version,
                artifacts,
                remembered_time_signature_in=remembered_time_signature_in,
                score_type=score_type,
            )
            debug_crops = system_result.pop("debug_crops", None)
            reference_examples_attached = _safe_int(system_result.pop("reference_examples_attached", 0), 0)
            usage = system_result.pop("_internal_ai_usage", None)
            system_result.pop("remembered_time_signature_out", None)
            system_result.pop("last_time_signature_update", None)
            system_result.pop("time_signature_updates", None)
            ai_suggestions = _merge_ai_suggestions_state(ai_suggestions, system_result, artifact_run_id, source_state_version)
            _append_internal_ai_cost_usage(
                mapping_summary,
                job_id=job_id,
                run_id=artifact_run_id,
                system_row=system_row,
                model=str(system_result.get("model") or _requested_ai_model_name("general")),
                usage=usage if isinstance(usage, dict) else None,
                pass_kind="general",
                charge_id=charge_id,
            )
            pass_state["general"] = "completed"
            pass_state["general_model"] = str(system_result.get("model") or _requested_ai_model_name("general"))
            pass_states[system_id] = pass_state
            ai_suggest_run["pass_state_by_system_id"] = pass_states
            mapping_summary["ai_suggestions"] = ai_suggestions
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        except Exception as exc:
            _finish_ai_access(ai_access, spent=False)
            error_payload = _ai_suggest_error_payload(exc)
            ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED if next_system_index > 0 else AI_SUGGEST_RUN_STATUS_FAILED
            ai_suggest_run["last_error"] = error_payload
            ai_suggest_run["failed_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
            _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            mapping_summary["ai_suggestions"] = ai_suggestions
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
            return jsonify({"job_id": job_id, "run_id": artifact_run_id, "status": ai_suggest_run["status"], "ai_suggestions": ai_suggestions, "ai_suggest_run": ai_suggest_run, "error": error_payload}), 200

    if not credit_group.get("charged"):
        if not _finish_ai_access(ai_access, spent=True):
            logger.warning(
                "AI_CREDIT_FINALIZE_PENDING provider=%s stage=spend",
                str(ai_access.get("provider") or "friend"),
            )
            credit_group["status"] = "charge_pending"
            credit_groups[charge_id] = credit_group
            ai_suggest_run["credit_groups"] = credit_groups
            ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED
            ai_suggest_run["last_error"] = {
                "code": "ai_credit_finalize_pending",
                "message": "AI results were saved, but credit confirmation must be retried.",
                "retryable": True,
                "provider_status": 503,
                "detail": "credit_finalize_pending",
            }
            _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
            return jsonify({"job_id": job_id, "run_id": artifact_run_id, "status": AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED, "ai_suggestions": ai_suggestions, "ai_suggest_run": ai_suggest_run, "error": ai_suggest_run["last_error"]}), 200
        credit_group.update(
            {
                "status": "charged",
                "charged": True,
                "charged_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
                "provider": str(ai_access.get("provider") or "friend"),
            }
        )
        credit_groups[charge_id] = credit_group
        ai_suggest_run["credit_groups"] = credit_groups
        mapping_summary["ai_suggest_run"] = ai_suggest_run
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])

    if _ending_pass_enabled() and pass_state.get("ending") != "completed":
        try:
            ending_result = _generate_ai_endings_for_system_batch(
                job_id,
                artifact_run_id,
                systems,
                system_row,
                system_measures,
                artifacts,
                active_ending_in=ai_suggest_run.get("ending_carry_kind"),
                score_type=score_type,
            )
            ending_usage = ending_result.pop("_internal_ai_usage", None)
            ai_suggest_run["ending_carry_kind"] = ending_result.get("active_ending_out")
            ai_suggestions = _merge_ai_ending_system_state(
                ai_suggestions,
                ending_result,
                measures,
                all_systems_completed=next_system_index + 1 >= systems_total,
            )
            _append_internal_ai_cost_usage(
                mapping_summary,
                job_id=job_id,
                run_id=artifact_run_id,
                system_row=system_row,
                model=str(ending_result.get("model") or _requested_ai_model_name("ending")),
                usage=ending_usage if isinstance(ending_usage, dict) else None,
                pass_kind="ending",
                charge_id=charge_id,
            )
            pass_state["ending"] = "completed"
            pass_state["ending_model"] = str(ending_result.get("model") or _requested_ai_model_name("ending"))
        except Exception as exc:
            pass_state["ending"] = "retryable_failed"
            pass_states[system_id] = pass_state
            error_payload = _ai_suggest_error_payload(exc, default_message="Ending detection temporarily failed.")
            ai_suggest_run["pass_state_by_system_id"] = pass_states
            ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED
            ai_suggest_run["last_error"] = error_payload
            ai_suggest_run["failed_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
            _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
            mapping_summary["ai_suggestions"] = ai_suggestions
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
            logger.warning("AI_ENDING_PASS_FAILED system_id=%s error_type=%s charge_id=%s", system_id, type(exc).__name__, charge_id)
            return jsonify({"job_id": job_id, "run_id": artifact_run_id, "status": AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED, "ai_suggestions": ai_suggestions, "ai_suggest_run": ai_suggest_run, "error": error_payload}), 200
    elif not _ending_pass_enabled():
        pass_state["ending"] = "disabled"

    pass_states[system_id] = pass_state
    completed_count = min(systems_total, next_system_index + 1)
    now_txt = _utc_now().isoformat().replace("+00:00", "Z")
    ai_suggest_run["pass_state_by_system_id"] = pass_states
    ai_suggest_run["systems_completed"] = completed_count
    ai_suggest_run["next_system_index"] = completed_count
    ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_COMPLETED if completed_count >= systems_total else AI_SUGGEST_RUN_STATUS_RUNNING
    ai_suggest_run["updated_at_utc"] = now_txt
    ai_suggest_run["last_error"] = None
    ai_suggest_run["failed_at_utc"] = None
    if completed_count >= systems_total:
        ai_suggest_run["completed_at_utc"] = now_txt
        _rebuild_ai_ending_pairs(ai_suggestions, measures, all_systems_completed=True)
    _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
    mapping_summary["ai_suggestions"] = ai_suggestions
    mapping_summary["ai_suggest_run"] = ai_suggest_run
    _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
    logger.info("AI_SYSTEM_COMPLETE system_id=%s pass=combined charge_id=%s", system_id, charge_id)
    response = {
        "job_id": job_id,
        "run_id": artifact_run_id,
        "status": ai_suggest_run["status"],
        "ai_suggestions": ai_suggestions,
        "ai_suggest_run": ai_suggest_run,
        "reference_examples_attached": int(reference_examples_attached),
        "storage_mode": _storage_mode_for_artifacts(artifacts),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(artifact_run_id, artifacts),
        "duration_ms": int((time.time() - started) * 1000),
    }
    if isinstance(debug_crops, dict):
        response["debug_crops"] = debug_crops
    return jsonify(response), 200


def _step_ai_suggest_split_credit_v2(
    *,
    job_id: str,
    artifact_run_id: int,
    mapping_summary: dict,
    artifacts: dict,
    systems: list[dict],
    measures: list[dict],
    system_batches: list[tuple[dict, list[dict]]],
    source_state_version: str | None,
    ai_suggestions: dict,
    ai_suggest_run: dict,
    next_system_index: int,
    started: float,
) -> tuple:
    systems_total = len(system_batches)
    system_row, system_measures = system_batches[next_system_index]
    system_id = str(system_row.get("system_id") or "").strip()
    execution_id = ai_suggest_run.get("execution_id")
    ending_enabled = _ending_pass_enabled()
    general_charge_id = _ai_credit_pass_id(
        job_id,
        artifact_run_id,
        source_state_version,
        next_system_index,
        "general",
        execution_id=execution_id,
    )
    ending_charge_id = _ai_credit_pass_id(
        job_id,
        artifact_run_id,
        source_state_version,
        next_system_index,
        "ending",
        execution_id=execution_id,
    ) if ending_enabled else None
    pass_states = dict(ai_suggest_run.get("pass_state_by_system_id") or {})
    pass_state = dict(pass_states.get(system_id) or {})
    credit_groups = dict(ai_suggest_run.get("credit_groups") or {})

    general_group = dict(credit_groups.get(general_charge_id) or {})
    general_group.setdefault("kind", "general")
    general_group.setdefault("group_index", next_system_index)
    general_group.setdefault("system_ids", [system_id])
    credit_groups[general_charge_id] = general_group

    ending_group = None
    if ending_charge_id:
        ending_group = dict(credit_groups.get(ending_charge_id) or {})
        ending_group.setdefault("kind", "ending")
        ending_group.setdefault("group_index", next_system_index // 2)
        ending_group.setdefault("system_ids", [
            str(system_batches[index][0].get("system_id") or "")
            for index in range((next_system_index // 2) * 2, min(systems_total, (next_system_index // 2) * 2 + 2))
        ])
        credit_groups[ending_charge_id] = ending_group

    def save_credit_state() -> None:
        ai_suggest_run["credit_groups"] = credit_groups
        mapping_summary["ai_suggest_run"] = ai_suggest_run
        mapping_summary["ai_suggestions"] = ai_suggestions
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])

    accesses: dict[str, dict] = {}

    def release_access(charge_id: str) -> bool:
        access = accesses.get(charge_id)
        group = dict(credit_groups.get(charge_id) or {})
        if not isinstance(access, dict) or access.get("already_charged") or group.get("charged"):
            return True
        released = _finish_ai_access(access, spent=False)
        group["status"] = "released" if released else "reservation_pending"
        credit_groups[charge_id] = group
        return released

    required_charges = [(general_charge_id, "general")]
    if ending_charge_id:
        required_charges.append((ending_charge_id, "ending"))

    try:
        for charge_id, kind in required_charges:
            group = dict(credit_groups.get(charge_id) or {})
            if group.get("charged"):
                continue
            access = _ai_access(
                reserve=True,
                job_id=job_id,
                system_id=system_id or None,
                charge_id=charge_id,
            )
            accesses[charge_id] = access
            if access.get("already_charged"):
                group.update(
                    {
                        "status": "charged",
                        "charged": True,
                        "provider": str(access.get("provider") or "friend"),
                    }
                )
            else:
                group.update(
                    {
                        "status": "reserved",
                        "charged": False,
                        "provider": str(access.get("provider") or "friend"),
                    }
                )
            group["kind"] = kind
            credit_groups[charge_id] = group
            save_credit_state()
    except (FriendAccessError, PaidAccessError) as exc:
        release_ok = True
        for charge_id in list(accesses):
            release_ok = release_access(charge_id) and release_ok
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED if next_system_index > 0 else AI_SUGGEST_RUN_STATUS_FAILED
        ai_suggest_run["failed_at_utc"] = now_txt
        ai_suggest_run["last_error"] = {
            "code": exc.code,
            "message": exc.message,
            "retryable": bool(exc.retryable),
            "provider_status": int(exc.status_code),
            "detail": exc.code if release_ok else "credit_release_pending",
        }
        _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
        save_credit_state()
        return _friend_error_response(exc) if isinstance(exc, FriendAccessError) else _paid_error_response(exc)

    def credit_pending_response(charge_id: str, kind: str) -> tuple:
        group = dict(credit_groups.get(charge_id) or {})
        group["status"] = "charge_pending"
        credit_groups[charge_id] = group
        logger.warning(
            "AI_CREDIT_FINALIZE_PENDING provider=%s stage=spend kind=%s",
            str((accesses.get(charge_id) or {}).get("provider") or group.get("provider") or "friend"),
            kind,
        )
        ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED
        ai_suggest_run["last_error"] = {
            "code": "ai_credit_finalize_pending",
            "message": "AI results were saved, but credit confirmation must be retried.",
            "retryable": True,
            "provider_status": 503,
            "detail": f"{kind}_credit_finalize_pending",
        }
        _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
        save_credit_state()
        return jsonify({
            "job_id": job_id,
            "run_id": artifact_run_id,
            "status": AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED,
            "ai_suggestions": ai_suggestions,
            "ai_suggest_run": ai_suggest_run,
            "error": ai_suggest_run["last_error"],
        }), 200

    def finalize_charge(charge_id: str, kind: str) -> tuple | None:
        group = dict(credit_groups.get(charge_id) or {})
        if group.get("charged"):
            return None
        access = accesses.get(charge_id)
        if not isinstance(access, dict) or not _finish_ai_access(access, spent=True):
            return credit_pending_response(charge_id, kind)
        group.update(
            {
                "status": "charged",
                "charged": True,
                "charged_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
                "provider": str(access.get("provider") or "friend"),
            }
        )
        credit_groups[charge_id] = group
        save_credit_state()
        return None

    score_type = _normalize_ai_score_type(ai_suggest_run.get("score_type"))
    remembered_time_signature_in = _normalize_ai_time_signature_value(ai_suggest_run.get("remembered_time_signature"))
    debug_crops = None
    reference_examples_attached = 0

    if pass_state.get("general") != "completed":
        try:
            system_result = _generate_ai_suggestions_for_system_batch(
                job_id,
                artifact_run_id,
                systems,
                system_row,
                system_measures,
                source_state_version,
                artifacts,
                remembered_time_signature_in=remembered_time_signature_in,
                score_type=score_type,
            )
            debug_crops = system_result.pop("debug_crops", None)
            reference_examples_attached = _safe_int(system_result.pop("reference_examples_attached", 0), 0)
            usage = system_result.pop("_internal_ai_usage", None)
            system_result.pop("remembered_time_signature_out", None)
            system_result.pop("last_time_signature_update", None)
            system_result.pop("time_signature_updates", None)
            ai_suggestions = _merge_ai_suggestions_state(ai_suggestions, system_result, artifact_run_id, source_state_version)
            _append_internal_ai_cost_usage(
                mapping_summary,
                job_id=job_id,
                run_id=artifact_run_id,
                system_row=system_row,
                model=str(system_result.get("model") or _requested_ai_model_name("general")),
                usage=usage if isinstance(usage, dict) else None,
                pass_kind="general",
                charge_id=general_charge_id,
            )
            pass_state["general"] = "completed"
            pass_state["general_model"] = str(system_result.get("model") or _requested_ai_model_name("general"))
            pass_states[system_id] = pass_state
            ai_suggest_run["pass_state_by_system_id"] = pass_states
            mapping_summary["ai_suggestions"] = ai_suggestions
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        except Exception as exc:
            release_access(general_charge_id)
            if ending_charge_id:
                release_access(ending_charge_id)
            error_payload = _ai_suggest_error_payload(exc)
            ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED if next_system_index > 0 else AI_SUGGEST_RUN_STATUS_FAILED
            ai_suggest_run["last_error"] = error_payload
            ai_suggest_run["failed_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
            _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
            save_credit_state()
            return jsonify({"job_id": job_id, "run_id": artifact_run_id, "status": ai_suggest_run["status"], "ai_suggestions": ai_suggestions, "ai_suggest_run": ai_suggest_run, "error": error_payload}), 200

    pending = finalize_charge(general_charge_id, "general")
    if pending is not None:
        if ending_charge_id:
            release_access(ending_charge_id)
            save_credit_state()
        return pending

    if ending_enabled and pass_state.get("ending") != "completed":
        try:
            ending_result = _generate_ai_endings_for_system_batch(
                job_id,
                artifact_run_id,
                systems,
                system_row,
                system_measures,
                artifacts,
                active_ending_in=ai_suggest_run.get("ending_carry_kind"),
                score_type=score_type,
            )
            ending_usage = ending_result.pop("_internal_ai_usage", None)
            ai_suggest_run["ending_carry_kind"] = ending_result.get("active_ending_out")
            ai_suggestions = _merge_ai_ending_system_state(
                ai_suggestions,
                ending_result,
                measures,
                all_systems_completed=next_system_index + 1 >= systems_total,
            )
            _append_internal_ai_cost_usage(
                mapping_summary,
                job_id=job_id,
                run_id=artifact_run_id,
                system_row=system_row,
                model=str(ending_result.get("model") or _requested_ai_model_name("ending")),
                usage=ending_usage if isinstance(ending_usage, dict) else None,
                pass_kind="ending",
                charge_id=ending_charge_id,
            )
            pass_state["ending"] = "completed"
            pass_state["ending_model"] = str(ending_result.get("model") or _requested_ai_model_name("ending"))
            pass_states[system_id] = pass_state
            ai_suggest_run["pass_state_by_system_id"] = pass_states
            mapping_summary["ai_suggestions"] = ai_suggestions
            mapping_summary["ai_suggest_run"] = ai_suggest_run
            _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
        except Exception as exc:
            if ending_charge_id:
                release_access(ending_charge_id)
            pass_state["ending"] = "retryable_failed"
            pass_states[system_id] = pass_state
            error_payload = _ai_suggest_error_payload(exc, default_message="Ending detection temporarily failed.")
            ai_suggest_run["pass_state_by_system_id"] = pass_states
            ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED
            ai_suggest_run["last_error"] = error_payload
            ai_suggest_run["failed_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
            _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
            save_credit_state()
            logger.warning("AI_ENDING_PASS_FAILED system_id=%s error_type=%s charge_id=%s", system_id, type(exc).__name__, ending_charge_id)
            return jsonify({"job_id": job_id, "run_id": artifact_run_id, "status": AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED, "ai_suggestions": ai_suggestions, "ai_suggest_run": ai_suggest_run, "error": error_payload}), 200
    elif not ending_enabled:
        pass_state["ending"] = "disabled"

    if ending_charge_id:
        pending = finalize_charge(ending_charge_id, "ending")
        if pending is not None:
            return pending

    pass_states[system_id] = pass_state
    completed_count = min(systems_total, next_system_index + 1)
    now_txt = _utc_now().isoformat().replace("+00:00", "Z")
    ai_suggest_run["pass_state_by_system_id"] = pass_states
    ai_suggest_run["systems_completed"] = completed_count
    ai_suggest_run["next_system_index"] = completed_count
    ai_suggest_run["status"] = AI_SUGGEST_RUN_STATUS_COMPLETED if completed_count >= systems_total else AI_SUGGEST_RUN_STATUS_RUNNING
    ai_suggest_run["updated_at_utc"] = now_txt
    ai_suggest_run["last_error"] = None
    ai_suggest_run["failed_at_utc"] = None
    if completed_count >= systems_total:
        ai_suggest_run["completed_at_utc"] = now_txt
        _rebuild_ai_ending_pairs(ai_suggestions, measures, all_systems_completed=True)
    _refresh_ai_run_recovery_flags(ai_suggest_run, source_state_version)
    mapping_summary["ai_suggestions"] = ai_suggestions
    mapping_summary["ai_suggest_run"] = ai_suggest_run
    _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
    logger.info(
        "AI_SYSTEM_COMPLETE system_id=%s pass=combined general_charge_id=%s ending_charge_id=%s",
        system_id,
        general_charge_id,
        ending_charge_id or "disabled",
    )
    response = {
        "job_id": job_id,
        "run_id": artifact_run_id,
        "status": ai_suggest_run["status"],
        "ai_suggestions": ai_suggestions,
        "ai_suggest_run": ai_suggest_run,
        "reference_examples_attached": int(reference_examples_attached),
        "storage_mode": _storage_mode_for_artifacts(artifacts),
        "artifacts": artifacts,
        "artifacts_http": _artifact_http_uris_for_run(artifact_run_id, artifacts),
        "duration_ms": int((time.time() - started) * 1000),
    }
    if isinstance(debug_crops, dict):
        response["debug_crops"] = debug_crops
    return jsonify(response), 200


def _step_ai_suggest_v2(**kwargs) -> tuple:
    ai_suggest_run = kwargs.get("ai_suggest_run") if isinstance(kwargs.get("ai_suggest_run"), dict) else {}
    if str(ai_suggest_run.get("credit_scheme") or "") == AI_CREDIT_SCHEME_VERSION:
        return _step_ai_suggest_split_credit_v2(**kwargs)
    return _step_ai_suggest_two_system_credit_v1(**kwargs)


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

    if ai_suggest_run.get("status") in {AI_SUGGEST_RUN_STATUS_COMPLETED, AI_SUGGEST_RUN_STATUS_FAILED, AI_SUGGEST_RUN_STATUS_CANCELLED}:
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

    if str(ai_suggest_run.get("version") or "") == AI_SUGGEST_RUN_VERSION:
        return _step_ai_suggest_v2(
            job_id=job_id,
            artifact_run_id=int(artifact_run_id),
            mapping_summary=mapping_summary,
            artifacts=artifacts,
            systems=systems,
            measures=measures,
            system_batches=system_batches,
            source_state_version=source_state_version,
            ai_suggestions=ai_suggestions,
            ai_suggest_run=ai_suggest_run,
            next_system_index=next_system_index,
            started=started,
        )

    system_row, system_measures = system_batches[next_system_index]
    try:
        ai_access = _ai_access(
            reserve=True,
            job_id=job_id,
            system_id=str(system_row.get("system_id") or "") or None,
        )
    except FriendAccessError as exc:
        return _friend_error_response(exc)
    except PaidAccessError as exc:
        return _paid_error_response(exc)

    debug_crops = None
    reference_examples_attached = 0
    internal_ai_usage = None
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
            internal_ai_usage = system_result.pop("_internal_ai_usage", None)
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
        cost_summary = _append_internal_ai_cost_usage(
            mapping_summary,
            job_id=job_id,
            run_id=int(artifact_run_id),
            system_row=system_row,
            model=str(system_result.get("model") or _requested_ai_model_name()),
            usage=internal_ai_usage if isinstance(internal_ai_usage, dict) else None,
        )
        latest_cost = (cost_summary.get("invocations") or [])[-1] if isinstance(cost_summary, dict) else {}
        logger.info(
            "AI_COST_USAGE job_id=%s run_id=%s system_id=%s page=%s input_tokens=%s output_tokens=%s retry_attempts=%s usage_available=%s",
            job_id,
            int(artifact_run_id),
            latest_cost.get("system_id"),
            latest_cost.get("page"),
            latest_cost.get("input_tokens"),
            latest_cost.get("output_tokens"),
            latest_cost.get("retry_attempts"),
            latest_cost.get("usage_available"),
        )
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
        _finish_ai_access(ai_access, spent=True)
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
        _finish_ai_access(ai_access, spent=False)
        error_payload = _ai_suggest_error_payload(exc)
        now_txt = _utc_now().isoformat().replace("+00:00", "Z")
        saved_completed_count = max(0, _safe_int(ai_suggest_run.get("systems_completed"), 0))
        failed_status = AI_SUGGEST_RUN_STATUS_PARTIAL_FAILED if saved_completed_count > 0 else AI_SUGGEST_RUN_STATUS_FAILED
        ai_suggest_run["status"] = failed_status
        ai_suggest_run["updated_at_utc"] = now_txt
        ai_suggest_run["failed_at_utc"] = now_txt
        ai_suggest_run["systems_completed"] = saved_completed_count
        ai_suggest_run["next_system_index"] = max(0, _safe_int(ai_suggest_run.get("next_system_index"), next_system_index))
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
            "status": failed_status,
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
    atomic = payload.get("atomic") is True
    request_id = str(payload.get("request_id") or "").strip()
    requested_state_version = str(payload.get("state_version") or "").strip()
    accepted_ai_ending_pair_ids = {
        str(value or "").strip()
        for value in (payload.get("accepted_ai_ending_pair_ids") or [])
        if str(value or "").strip()
    } if isinstance(payload.get("accepted_ai_ending_pair_ids"), list) else set()
    edits_requested_count = len(edits) if isinstance(edits, list) else 0

    if atomic and not re.fullmatch(r"[A-Za-z0-9-]{16,128}", request_id):
        return (
            jsonify(
                {
                    "error": "atomic relabel requires a valid request_id",
                    "code": "invalid_request_id",
                    "job_id": job_id,
                    "trace_id": trace_id,
                }
            ),
            400,
        )

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

    persisted_mapping_summary = deepcopy(mapping_summary)
    mapping_summary = deepcopy(mapping_summary)
    mapping_uri = artifacts["mapping_summary"]
    baseline_pdf_uri = artifacts["audiveris_out_pdf"]
    corrected_pdf_uri = artifacts["audiveris_out_corrected_pdf"]

    if atomic:
        duplicate_receipt = _find_manual_fix_batch_receipt(mapping_summary, request_id)
        if isinstance(duplicate_receipt, dict):
            duplicate_relabel = {
                "applied_edits": deepcopy(duplicate_receipt.get("applied_edits") or []),
                "rejected_edits": [],
                "labels_mode": str(duplicate_receipt.get("labels_mode") or LABELS_MODE_SYSTEM_ONLY),
                "state_version_before": duplicate_receipt.get("state_version_before"),
                "state_version_after": duplicate_receipt.get("state_version_after"),
                "updated_system_ids": deepcopy(duplicate_receipt.get("updated_system_ids") or []),
                "systems_updated_count": _safe_int(duplicate_receipt.get("systems_updated_count"), 0),
                "labels_redrawn_count": _safe_int(duplicate_receipt.get("labels_redrawn_count"), 0),
                "duration_ms": 0,
                "redraw_duration_ms": 0,
                "duplicate_request": True,
                "request_id": request_id,
            }
            response = {
                "job_id": job_id,
                "run_id": int(run_id),
                "status": "succeeded",
                "trace_id": trace_id,
                "debug_result": "duplicate_request",
                "artifacts": artifacts,
                "artifacts_http": _artifact_http_uris_for_run(int(run_id), artifacts),
                "storage_mode": _storage_mode_for_artifacts(artifacts),
                "relabel": duplicate_relabel,
            }
            if rec and isinstance(rec, dict) and rec.get("pdf_gcs_uri"):
                response["pdf_gcs_uri"] = rec.get("pdf_gcs_uri")
            print(
                f"RELABEL_TRACE_RESULT trace_id={trace_id} result=duplicate_request "
                f"request_id={request_id}"
            )
            return jsonify(response), 200

    def _persist_error_trace(trace_payload: dict) -> bool:
        return _persist_relabel_trace(deepcopy(persisted_mapping_summary), mapping_uri, trace_payload, trace_id)

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
        _persist_error_trace(trace)
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
        _persist_error_trace(trace)
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
    if atomic and (not requested_state_version or requested_state_version != state_version_before):
        first_page = 0
        version_edits = edits if isinstance(edits, list) else []
        for raw_edit in version_edits:
            if isinstance(raw_edit, dict):
                first_page = _safe_int(raw_edit.get("page"), 0)
                if first_page > 0:
                    break
        rejection = {"reason": "state_version_conflict"}
        if first_page > 0:
            rejection["page"] = first_page
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "state_version_before": state_version_before,
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": 1,
            "rejected_reason_counts": {"state_version_conflict": 1},
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "validation_error",
            "reason": "state_version_conflict",
        }
        _persist_error_trace(trace)
        return (
            jsonify(
                {
                    "error": "state version changed before manual fixes could save",
                    "code": "state_version_conflict",
                    "job_id": job_id,
                    "trace_id": trace_id,
                    "page": first_page if first_page > 0 else None,
                    "rejected_edits": [rejection],
                }
            ),
            409,
        )
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
        _persist_error_trace(trace)
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
        _persist_error_trace(trace)
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

    if atomic and rejected:
        rejection_rows = _atomic_relabel_rejections(rejected, editable_state)
        first_page = next(
            (_safe_int(row.get("page"), 0) for row in rejection_rows if _safe_int(row.get("page"), 0) > 0),
            0,
        )
        trace = {
            "trace_id": trace_id,
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "job_id": job_id,
            "requested_run_id": int(run_id),
            "artifact_run_id": int(artifact_run_id),
            "state_version_before": state_version_before,
            "edits_requested_count": edits_requested_count,
            "applied_count": 0,
            "rejected_count": len(rejection_rows),
            "rejected_reason_counts": _rejected_reason_counts(rejected),
            "updated_system_ids_count": 0,
            "labels_redrawn_count": 0,
            "duration_ms": int((time.time() - started) * 1000),
            "redraw_duration_ms": 0,
            "result": "validation_error",
            "reason": "atomic_relabel_rejected",
            "request_id": request_id,
        }
        _persist_error_trace(trace)
        return (
            jsonify(
                {
                    "error": "Manual Fix batch was rejected",
                    "code": "atomic_relabel_rejected",
                    "job_id": job_id,
                    "trace_id": trace_id,
                    "page": first_page if first_page > 0 else None,
                    "rejected_edits": rejection_rows,
                }
            ),
            409,
        )

    redraw_ms = 0
    corrected_backup_bytes: bytes | None = None
    corrected_pdf_uploaded = False
    try:
        with TemporaryDirectory(prefix="omr-relabel-") as tmp:
            tmpdir = Path(tmp)
            in_pdf = tmpdir / "audiveris_out.pdf"
            out_pdf = tmpdir / "audiveris_out_corrected.pdf"
            backup_pdf = tmpdir / "previous_audiveris_out_corrected.pdf"
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
            try:
                _download_gcs_to_file(corrected_pdf_uri, backup_pdf)
                if backup_pdf.is_file():
                    corrected_backup_bytes = backup_pdf.read_bytes()
            except Exception as exc:
                print(f"RELABEL_BACKUP_WARN trace_id={trace_id} detail={_safe_error_text(exc)}")
            _upload_file_to_gcs(out_pdf, corrected_pdf_uri, content_type="application/pdf")
            corrected_pdf_uploaded = True
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
        _persist_error_trace(trace)
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
    if accepted_ai_ending_pair_ids:
        _resolve_ai_ending_pairs(mapping_summary, accepted_ai_ending_pair_ids)
    manual_pages_updated = {
        _safe_int(row.get("page"), 0)
        for row in applied
        if isinstance(row, dict) and str(row.get("type") or "").strip() == "replace_manual_rows_for_page"
    }
    if manual_pages_updated:
        manual_measure_ids_before = _measure_ids_on_pages(measures_before, manual_pages_updated, source=ROW_SOURCE_MANUAL)
        manual_measure_ids_after = _measure_ids_on_pages(editable_state.get("measures") or [], manual_pages_updated, source=ROW_SOURCE_MANUAL)
        removed_manual_measure_ids = manual_measure_ids_before - manual_measure_ids_after
        if removed_manual_measure_ids:
            _remove_ai_suggestion_entries(mapping_summary, removed_manual_measure_ids)
            _clear_measure_state_for_ids(editable_state, removed_manual_measure_ids)
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
        removed_auto_measure_ids = (auto_measure_ids_before - auto_measure_ids_after) | excluded_auto_measure_ids
        if removed_auto_measure_ids:
            _remove_ai_suggestion_entries(mapping_summary, removed_auto_measure_ids)
            _clear_measure_state_for_ids(editable_state, removed_auto_measure_ids)
    system_ids_before = {
        str(row.get("system_id") or "").strip()
        for row in systems_before
        if isinstance(row, dict) and str(row.get("system_id") or "").strip()
    }
    system_ids_after = {
        str(row.get("system_id") or "").strip()
        for row in systems
        if isinstance(row, dict) and str(row.get("system_id") or "").strip()
    }
    removed_system_ids = system_ids_before - system_ids_after
    if removed_system_ids:
        rest_systems = _editable_rest_systems(editable_state)
        for system_id in removed_system_ids:
            rest_systems.pop(system_id, None)
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
    if atomic:
        _append_manual_fix_batch_receipt(
            mapping_summary,
            {
                "request_id": request_id,
                "saved_at_utc": relabel_info["updated_at_utc"],
                "state_version_before": state_version_before,
                "state_version_after": state_version_after,
                "applied_edits": deepcopy(applied),
                "labels_mode": relabel_info["labels_mode"],
                "updated_system_ids": deepcopy(updated_system_ids),
                "systems_updated_count": total_systems,
                "labels_redrawn_count": labels_drawn,
            },
        )

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
        rollback_result = "not_needed"
        if corrected_pdf_uploaded:
            try:
                if corrected_backup_bytes:
                    _upload_bytes_to_gcs(corrected_backup_bytes, corrected_pdf_uri, content_type="application/pdf")
                    rollback_result = "restored_previous_corrected_pdf"
                else:
                    _delete_gcs_uri_if_exists(corrected_pdf_uri)
                    rollback_result = "deleted_new_corrected_pdf"
            except Exception as rollback_exc:
                rollback_result = f"rollback_failed:{_safe_error_text(rollback_exc)}"
                print(
                    f"RELABEL_TRACE_ERROR trace_id={trace_id} stage=rollback_corrected_pdf "
                    f"reason=rollback_failed detail={_safe_error_text(rollback_exc)}"
                )
        return (
            jsonify(
                {
                    "error": "failed to upload mapping_summary",
                    "job_id": job_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "debug_result": "upload_error",
                    "rollback_result": rollback_result,
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
            "request_id": request_id if atomic else None,
            "duplicate_request": False,
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


@app.route("/api/omr/jobs/<job_id>/ai-ending-suggestions/<pair_id>/dismiss", methods=["POST"])
def dismiss_ai_ending_pair(job_id: str, pair_id: str):
    run_id, rec, err = _resolve_run_id_from_job_id(job_id)
    if err:
        return jsonify({"error": err, "job_id": job_id}), 409
    artifact_key = _job_artifact_key(job_id, int(run_id), rec if isinstance(rec, dict) else None)
    try:
        artifacts, mapping_summary, artifact_run_id = _load_mapping_for_run(int(run_id), artifact_key=artifact_key)
    except StaleArtifactsError as exc:
        return jsonify({"error": "requested job_id does not match artifacts", "requested_run_id": exc.requested_run_id, "artifact_run_id": exc.artifact_run_id}), 409
    except Exception as exc:
        return jsonify({"error": "failed to load state", "detail": _safe_error_text(exc), "job_id": job_id}), 502
    clean_pair_id = str(pair_id or "").strip()
    ai_suggestions = _current_ai_suggestions(mapping_summary)
    already_resolved = clean_pair_id in {
        str(value or "").strip()
        for value in ((ai_suggestions or {}).get("resolved_ending_pair_ids") or [])
        if str(value or "").strip()
    }
    if not already_resolved and not _remove_ai_ending_pairs(mapping_summary, {clean_pair_id}):
        return jsonify({"job_id": job_id, "run_id": int(artifact_run_id), "status": "failed", "error": {"code": "suggestion_not_found", "message": "AI ending suggestion not found.", "retryable": False, "detail": clean_pair_id}}), 404
    _resolve_ai_ending_pairs(mapping_summary, {clean_pair_id})
    try:
        _upload_json_to_gcs(mapping_summary, artifacts["mapping_summary"])
    except Exception as exc:
        return jsonify({"job_id": job_id, "run_id": int(artifact_run_id), "status": "failed", "error": {"code": "ai_suggestion_dismiss_failed", "message": "failed to dismiss AI ending suggestion", "retryable": True, "detail": _safe_error_text(exc)}}), 500
    return jsonify({"job_id": job_id, "run_id": int(artifact_run_id), "status": "succeeded", "dismissed_pair_id": clean_pair_id, "ai_suggestions": mapping_summary.get("ai_suggestions")}), 200


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
