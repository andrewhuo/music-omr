# OMR Backend API (Workflow Dispatch)

This API is implemented in:

- `/Users/andrew/Desktop/music-omr/omr-worker/worker.py`

Framework note:

- This backend is **Flask**, not FastAPI.

It is intended for website/app integration. The frontend calls your backend API, and the backend dispatches GitHub Actions.

## Authentication and CORS

- Header required for all `/api/omr/*` routes:
  - `X-Invite-Code: <INVITE_CODE>`
- If the invite code is missing/invalid:
  - `401 {"error":"invalid invite code"}`
- If server is missing `INVITE_CODE`:
  - `500 {"error":"server not configured: INVITE_CODE missing"}`
- CORS allowlist comes from:
  - `CORS_ALLOW_ORIGINS` (comma-separated)
- Preflight `OPTIONS` is supported for `/api/omr/*`.

## Endpoints

### `POST /api/omr/uploads`

Upload a PDF from browser/frontend (multipart form).

Request:

- `Content-Type: multipart/form-data`
- form field name: `file`

Response (201):

```json
{
  "upload_id": "91fa67040ce4495b",
  "pdf_gcs_uri": "gs://music-omr-bucket-777135743132/input/user-input/91fa67040ce4495b.pdf",
  "size_bytes": 142839,
  "content_type": "application/pdf"
}
```

Errors:

- `400` missing file / invalid type / empty file
- `413` file too large
- `500` upload failure

### `POST /api/omr/jobs`

Request body:

```json
{
  "pdf_gcs_uri": "gs://music-omr-bucket-777135743132/input/test.pdf"
}
```

Response (202):

```json
{
  "job_id": "uuid",
  "status": "queued",
  "run_id": 22209738954,
  "workflow": "audiveris.yml",
  "ref": "main",
  "pdf_gcs_uri": "gs://...",
  "status_url": "/api/omr/jobs/<job_id>",
  "run_url": "https://github.com/...",
  "artifacts": {
    "audiveris_out_pdf": "gs://.../output/audiveris_out.pdf",
    "audiveris_out_corrected_pdf": "gs://.../output/audiveris_out_corrected.pdf",
    "run_info": "gs://.../output/artifacts/run_info.json",
    "mapping_summary": "gs://.../output/artifacts/mapping_summary.json"
  },
  "artifacts_http": {
    "audiveris_out_pdf": "https://storage.googleapis.com/...",
    "audiveris_out_corrected_pdf": "",
    "run_info": "https://storage.googleapis.com/...",
    "mapping_summary": "https://storage.googleapis.com/..."
  }
}
```

### `GET /api/omr/jobs/{job_id}`

Response (200):

```json
{
  "job_id": "uuid or run_id",
  "run_id": 22209738954,
  "status": "queued|running|succeeded|failed",
  "github_status": "queued|in_progress|completed",
  "github_conclusion": "success|failure|cancelled|null",
  "ref": "main",
  "sha": "commit_sha",
  "run_attempt": 1,
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "run_url": "https://github.com/...",
  "artifacts": {
    "audiveris_out_pdf": "gs://.../output/audiveris_out.pdf",
    "audiveris_out_corrected_pdf": "gs://.../output/audiveris_out_corrected.pdf",
    "run_info": "gs://.../output/artifacts/run_info.json",
    "mapping_summary": "gs://.../output/artifacts/mapping_summary.json"
  },
  "artifacts_http": {
    "audiveris_out_pdf": "https://storage.googleapis.com/...",
    "audiveris_out_corrected_pdf": "",
    "run_info": "https://storage.googleapis.com/...",
    "mapping_summary": "https://storage.googleapis.com/..."
  }
}
```

### `POST /api/omr/jobs/{job_id}/relabel`

Applies user edits to system-start numbering and redraws labels without rerunning OMR.

Request body:

```json
{
  "edits": [
    { "type": "set_system_start", "system_id": "p4_s2", "value": 230 }
  ]
}
```

Response (200):

```json
{
  "job_id": "uuid or run_id",
  "run_id": 22209738954,
  "status": "succeeded",
  "trace_id": "a1b2c3d4e5f6",
  "debug_result": "success",
  "artifacts": {
    "audiveris_out_pdf": "gs://.../output/audiveris_out.pdf",
    "audiveris_out_corrected_pdf": "gs://.../output/audiveris_out_corrected.pdf",
    "run_info": "gs://.../output/artifacts/run_info.json",
    "mapping_summary": "gs://.../output/artifacts/mapping_summary.json"
  },
  "artifacts_http": {
    "audiveris_out_pdf": "https://storage.googleapis.com/...",
    "audiveris_out_corrected_pdf": "https://storage.googleapis.com/...",
    "run_info": "https://storage.googleapis.com/...",
    "mapping_summary": "https://storage.googleapis.com/..."
  },
  "relabel": {
    "applied_edits": [{ "type": "set_system_start", "system_id": "p4_s2", "value": 230 }],
    "rejected_edits": [],
    "state_version_before": "8c1f9a0e1f3a8d6b",
    "state_version_after": "5d40d4e5a76ec7f9",
    "updated_system_ids": ["p4_s2", "p4_s3", "p5_s0"],
    "systems_updated_count": 120,
    "labels_redrawn_count": 120,
    "duration_ms": 1800,
    "redraw_duration_ms": 950
  }
}
```

Error responses from relabel are additive and include:

- `trace_id`
- `debug_result` (`validation_error|stale_conflict|render_error|upload_error|internal_error`)

Debug taxonomy codes are tracked in `mapping_summary.json.relabel_debug.reason_counts`, including:

- `stale_run_mismatch`
- `editable_state_missing`
- `invalid_payload`
- `unknown_system_id`
- `invalid_value`
- `value_out_of_range`
- `pdf_download_failed`
- `pdf_render_failed`
- `mapping_upload_failed`
- `state_load_failed`
- `internal_error`

### `GET /api/omr/jobs/{job_id}/state`

Returns frontend-friendly clickable system state (no need to read raw GCS JSON directly).

Response (200):

```json
{
  "job_id": "uuid or run_id",
  "run_id": 22209738954,
  "state_version": "8c1f9a0e1f3a8d6b",
  "editable_state": {
    "version": "system_state_v1",
    "qa": {
      "ok": true,
      "reason_counts": {},
      "warnings": [],
      "total_systems": 120
    },
    "systems": [
      {
        "system_id": "p1_s0",
        "page": 1,
        "system_index": 0,
        "current_value": "1",
        "anchor": { "x": 44.2, "y_top": 71.3, "y_bottom": 122.7 },
        "in_bounds": true,
        "guide_build_source": "primary"
      }
    ]
  },
  "relabel_debug_summary": {
    "history_count": 12,
    "history_max": 50,
    "last_result": "success",
    "last_trace_id": "a1b2c3d4e5f6",
    "reason_counts": {
      "invalid_payload": 2,
      "unknown_system_id": 1
    }
  },
  "artifacts": {
    "audiveris_out_pdf": "gs://.../output/audiveris_out.pdf",
    "audiveris_out_corrected_pdf": "gs://.../output/audiveris_out_corrected.pdf",
    "run_info": "gs://.../output/artifacts/run_info.json",
    "mapping_summary": "gs://.../output/artifacts/mapping_summary.json"
  },
  "artifacts_http": {
    "audiveris_out_pdf": "https://storage.googleapis.com/...",
    "audiveris_out_corrected_pdf": "https://storage.googleapis.com/...",
    "run_info": "https://storage.googleapis.com/...",
    "mapping_summary": "https://storage.googleapis.com/..."
  }
}
```

## Required Environment Variables

- `GITHUB_TOKEN`: token with permissions to dispatch and read workflow runs.
- `GITHUB_OWNER` (default: `andrewhuo`)
- `GITHUB_REPO` (default: `music-omr`)
- `GITHUB_WORKFLOW_ID` (default: `audiveris.yml`)
- `GITHUB_REF` (default: `main`)
- `OUTPUT_PREFIX` (default: `gs://music-omr-bucket-777135743132/output`)
- `INVITE_CODE`: shared invite code required by `X-Invite-Code`.
- Google Cloud ADC credentials (service account/workload identity) with read/write access to storage prefixes.

Optional:

- `CORS_ALLOW_ORIGINS` (default: `http://localhost:5173,http://localhost:3000`)
- `ARTIFACT_SIGNED_URL_TTL_SEC` (default: `1800`)
- `INPUT_UPLOAD_PREFIX` (default: `gs://music-omr-bucket-777135743132/input/user-input`)
- `MAX_UPLOAD_MB` (default: `25`)
- `RUN_DISCOVERY_TIMEOUT_SEC` (default: `20`)
- `RUN_DISCOVERY_POLL_SEC` (default: `2`)
- `RELABEL_MIN_VALUE` (default: `0`)
- `RELABEL_MAX_VALUE` (default: `1000000`)
- `RELABEL_DEBUG_HISTORY_MAX` (default: `50`)

## Notes

- `workflow_dispatch` does not return `run_id` directly. The backend performs a short discovery poll to find the newly created run.
- Frontend should never call GitHub APIs directly.
- Storage mode is currently single-latest: each new run overwrites prior output at `OUTPUT_PREFIX`.
- `mapping_summary.json` includes `editable_state` and `relabel_debug`.
- When a different run overwrote single-latest artifacts, `/state` and `/relabel` return `409` with requested/artifact run IDs.
- Signed URL fields in `artifacts_http` are best-effort. If a file does not exist or signing fails, that field may be an empty string.

## Smoke Test Script

Use:

```bash
python3 /Users/andrew/Desktop/music-omr/omr-worker/scripts/relabel_smoke.py \
  --worker-url "https://<your-worker-url>" \
  --pdf-gcs-uri "gs://music-omr-bucket-777135743132/input/test-input/01_single_staff.pdf"
```

Expected:

- job reaches `succeeded`
- relabel call succeeds
- response contains `audiveris_out_corrected.pdf`
- script prints `SMOKE ok ...`
