# OMR Backend API (Workflow Dispatch)

This API is implemented in:

- `/Users/andrew/Desktop/music-omr/omr-worker/worker.py`

It is intended for website/app integration. The frontend calls your backend API, and the backend dispatches GitHub Actions.

## Endpoints

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
    "annotated_pdf": "gs://.../output/runs/<run_id>/annotated.pdf",
    "measure_mapping_debug": "gs://.../output/runs/<run_id>/measure_mapping_debug.json",
    "mxl_page_manifest": "gs://.../output/runs/<run_id>/mxl_page_manifest.json",
    "omr_debug_bundle": "gs://.../output/runs/<run_id>/omr_debug_bundle.tar.gz",
    "latest_annotated_pdf": "gs://.../output/annotated.pdf"
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
    "annotated_pdf": "gs://.../output/runs/<run_id>/annotated.pdf",
    "measure_mapping_debug": "gs://.../output/runs/<run_id>/measure_mapping_debug.json",
    "mxl_page_manifest": "gs://.../output/runs/<run_id>/mxl_page_manifest.json",
    "omr_debug_bundle": "gs://.../output/runs/<run_id>/omr_debug_bundle.tar.gz",
    "latest_annotated_pdf": "gs://.../output/annotated.pdf"
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

Optional:

- `RUN_DISCOVERY_TIMEOUT_SEC` (default: `20`)
- `RUN_DISCOVERY_POLL_SEC` (default: `2`)

## Notes

- `workflow_dispatch` does not return `run_id` directly. The backend performs a short discovery poll to find the newly created run.
- Frontend should never call GitHub APIs directly.
