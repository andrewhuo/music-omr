# OMR Operator Runbook

This runbook is for day-to-day operation of:

- Workflow: `Audiveris OMR + Measure Label Debug`
- File: `/Users/andrew/Desktop/music-omr/.github/workflows/audiveris.yml`
- API integration doc: `/Users/andrew/Desktop/music-omr/docs/omr-api.md`

## What this pipeline does

1. Downloads one input PDF from GCS.
2. Runs Audiveris OMR with pinned runtime image.
3. Builds per-page XML manifest in strict mode.
4. Annotates the PDF with guides/measure labels.
5. Uploads compact output files to GCS.

## How to run

1. Open GitHub Actions in `andrewhuo/music-omr`.
2. Select workflow `Audiveris OMR + Measure Label Debug`.
3. Click `Run workflow`.
4. Choose branch `main`.
5. Set `pdf_gcs_uri` to a `gs://...pdf` path.
6. Run.

## Output locations

Base prefix:

- `gs://music-omr-bucket-777135743132/output`

This workflow now uses a single-latest storage contract.  
Each run clears the output prefix first, then writes only:

- `gs://music-omr-bucket-777135743132/output/audiveris_out.pdf`
- `gs://music-omr-bucket-777135743132/output/artifacts/run_info.json`
- `gs://music-omr-bucket-777135743132/output/artifacts/mapping_summary.json`

No `runs/<run_id>/...` folders are written by this workflow.

`mapping_summary.json` now includes `editable_state` (system-level anchors + current values) for fast relabel.

When backend relabel is used, one additional output may be written:

- `gs://music-omr-bucket-777135743132/output/audiveris_out_corrected.pdf`

## 2E test-input prep

Use two prefixes under the existing `input/` path:

- `gs://music-omr-bucket-777135743132/input/user-input/`
- `gs://music-omr-bucket-777135743132/input/test-input/`

Keep a stable regression pack in `test-input/` with fixed filenames:

- `01_single_staff.pdf`
- `02_piano.pdf`
- `03_score_easy.pdf`
- `04_score_hard.pdf`
- `05_endings.pdf`

Treat these as canonical test fixtures; avoid replacing them casually.

## Backend browser mode (Flask API)

The backend service in `/Users/andrew/Desktop/music-omr/omr-worker/worker.py` is Flask.

For frontend/browser usage:

- CORS allowlist is controlled by:
  - `CORS_ALLOW_ORIGINS` (comma-separated)
- Browser upload endpoint:
  - `POST /api/omr/uploads` (multipart `file`)
- API responses include:
  - `artifacts` (`gs://...`)
  - `artifacts_http` (signed HTTPS URLs, best-effort)

If a signed URL cannot be generated, that field may be empty (`""`) while the API call still succeeds.

## Cloud Run deploy contract (omr-trigger)

Deploy `omr-trigger` with immutable image tags (do not use `:latest`) and explicit runtime env vars.

Recommended env vars:

- `CORS_ALLOW_ORIGINS=http://localhost:5173,https://measure-marker.created.app`
- `ARTIFACT_SIGNED_URL_TTL_SEC=1800`
- `MAX_UPLOAD_MB=25`
- `INPUT_UPLOAD_PREFIX=gs://music-omr-bucket-777135743132/input/user-input`

Remove stale vars if present:

- `INVITE_CODE`
- `UPLOAD_GCS_PREFIX`

Notes:

- `omr-worker` Cloud Run image is now Flask API only (no local Audiveris install).
- Audiveris execution remains in GitHub Actions workflow, not in Cloud Run API service.

### Deploy snippet (immutable image)

```bash
set -euo pipefail
PROJECT="music-omr-backend"
REGION="us-central1"
SERVICE="omr-trigger"
REPO="us-central1-docker.pkg.dev/${PROJECT}/omr-trigger-service/trigger"
SHA="$(git rev-parse --short HEAD)"
TS="$(date +%Y%m%d-%H%M%S)"
IMAGE="${REPO}:${SHA}-${TS}"

gcloud builds submit ./omr-worker --tag "${IMAGE}" --project "${PROJECT}" --region "${REGION}"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --set-env-vars "CORS_ALLOW_ORIGINS=http://localhost:5173,https://measure-marker.created.app,ARTIFACT_SIGNED_URL_TTL_SEC=1800,MAX_UPLOAD_MB=25,INPUT_UPLOAD_PREFIX=gs://music-omr-bucket-777135743132/input/user-input" \
  --remove-env-vars "INVITE_CODE,UPLOAD_GCS_PREFIX"
```

## Log modes

Default mode is quiet production logging.

- Env key: `PIPELINE_LOG_MODE`
- Allowed values: `quiet`, `debug`
- Default: `quiet`

`quiet` mode:

- suppresses user-facing debug marks in output PDF
- keeps strict mapping summary and system QA summary lines
- keeps compact JSON artifacts as the source of truth

`debug` mode:

- re-enables parser debug overlays/logging for investigations
- does not change strict gates or mapping behavior

Operational guidance:

- use `quiet` for normal runs
- switch to `debug` only while triaging a failure

## Success criteria

Look for these lines in logs:

- `WORKFLOW_SIGNATURE=engine-pin-v1`
- `AUDIVERIS_VERSION_NUM=5.9.0`
- `PAGE_SPLIT total_pages=<N>`
- `MXL_PAGE_MANIFEST path=... entries=<N> missing=0`
- `SUMMARY strict_xml_pages_ok=<N>/<N>`

If strict coverage fails, the run should fail.

## System QA Gate

This pipeline enforces system-level output consistency checks in baseline mode:

- one long guideline per system
- one system-start label per system

In Step 16, pages with `staff_start_source=mxl` and `mapping_status=ok` must satisfy:

- `system_guide_count == omr_system_count`
- `system_labels_drawn_count == omr_system_count`

Default policy is warning-first:

- `SYSTEM_QA_POLICY=warn` (default): run succeeds and prints warning lines.
- `SYSTEM_QA_POLICY=strict`: run fails on drift (QA mode).

Warning lines:

- `SUMMARY system_qa_warn page=... detail=...`

Strict mode failure lines:

- `failure_class=SYSTEM_COUNT_DRIFT`
- `SUMMARY strict_system_drift page=... detail=...`

## What to collect when a run fails

Download and share these step logs:

- `8_Run Audiveris OMR.txt`
- `13_Build page-split XML manifest (strict mode).txt`
- `16_Mapping debug summary.txt`
- `Upload outputs.txt`

Also share these storage lines from Step 18:

- `uploaded_audiveris_out=...`
- `uploaded_artifact_run_info=...`
- `uploaded_artifact_mapping_summary=...`

## Frontend/API troubleshooting

Common browser/API failures:

1. CORS blocked in browser
- Check request origin exactly matches an entry in `CORS_ALLOW_ORIGINS`.
- Confirm preflight `OPTIONS` is reaching backend.

2. `413 file too large` on upload
- Increase `MAX_UPLOAD_MB` or use smaller input file.

3. Missing PDF link in frontend (`artifacts_http` empty)
- File may not exist yet, or signed URL generation failed.
- Retry status/state API call after run completion.

## Anything handoff (copy/paste)

Use backend:

- `https://omr-trigger-777135743132.us-central1.run.app`

Use flow:

1. `GET /api/omr/jobs` (optional connection check)
2. `POST /api/omr/uploads`
3. `POST /api/omr/jobs`
4. Poll `GET /api/omr/jobs/{job_id}` until `succeeded`
5. `GET /api/omr/jobs/{job_id}/state`
6. `POST /api/omr/jobs/{job_id}/relabel`

Rules:

- No invite/auth header required.
- Use `artifacts_http` URLs for browser preview/download.
- Handle `409` stale mismatch by starting a new job.

## Fast relabel flow (no full OMR rerun)

The backend supports fast user corrections via:

- `GET /api/omr/jobs/{job_id}/state`
- `POST /api/omr/jobs/{job_id}/relabel`

This endpoint:

- reads baseline PDF + `editable_state` from `mapping_summary.json`
- applies `set_system_start` edits
- redraws labels only
- writes `audiveris_out_corrected.pdf`

It does not rerun Audiveris or GitHub Actions.

## Relabel debug traces

Relabel now writes structured traces into:

- `gs://music-omr-bucket-777135743132/output/artifacts/mapping_summary.json`
- path: `relabel_debug`

Fields:

- `relabel_debug.version` (`relabel_debug_v1`)
- `relabel_debug.history_max` (default `50`)
- `relabel_debug.history` (latest traces only)
- `relabel_debug.last_trace`
- `relabel_debug.reason_counts`

Each relabel response includes:

- `trace_id`
- `debug_result`

Each `/state` response includes:

- `relabel_debug_summary` (`history_count`, `history_max`, `last_result`, `last_trace_id`, `reason_counts`)

Common reason codes:

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

Quick troubleshooting flow:

1. Find `trace_id` from relabel API response.
2. Grep worker logs for `RELABEL_TRACE_* trace_id=<id>`.
3. Check `relabel_debug.last_trace` and `relabel_debug.reason_counts` in `mapping_summary.json`.
4. If reason is `stale_run_mismatch`, rerun baseline first, then relabel again.

Worker logs include grep-friendly lines:

- `RELABEL_TRACE_START ...`
- `RELABEL_TRACE_RESULT ...`
- `RELABEL_TRACE_ERROR ...`

## Fast relabel smoke test

Use the helper script:

- `/Users/andrew/Desktop/music-omr/omr-worker/scripts/relabel_smoke.py`

Example:

```bash
python3 /Users/andrew/Desktop/music-omr/omr-worker/scripts/relabel_smoke.py \
  --worker-url "https://<your-worker-url>" \
  --pdf-gcs-uri "gs://music-omr-bucket-777135743132/input/test-input/01_single_staff.pdf"
```

Expected result:

- Script prints `SMOKE ok ...`
- Relabel response includes `state_version_before`, `state_version_after`, `updated_system_ids`
- `audiveris_out_corrected.pdf` exists in output.

If a newer workflow run overwrote single-latest artifacts, `/state` or `/relabel` returns `409`
with requested/artifact run IDs.

## Troubleshooting note: system-count mismatch

In strict mode, if MusicXML system starts and OMR system count do not match on a page
(for example expand/compress mapping would be needed), the run now fails strict coverage
instead of drawing repeated labels.

Strict mode now first tries an exact-safe merge of per-page movement XML starts before
failing. If exact-safe merge cannot be proven, strict still fails.

This workflow currently runs with `ENDING_LABEL_MODE=system_only` to keep one label
per system start. `system_plus_endings` is available only as a debug/visualization mode.

Look for:

- `mapping_reason=manifest_system_count_mismatch` (manifest mode)
- `mapping_reason=mxl_system_count_mismatch` (full-book mode)

## Counting policy: 1st/2nd endings

Semantic numbering now uses `semantic_continuous_v2_longest_ending` in Step 13.

- Alternative endings share the same starting measure number.
- Continuation after the ending group uses the longest ending length.
- Labels remain one per system start (left-margin style).

If ending markers exist but the grouping cannot be resolved deterministically, strict mode
marks that page as missing with `semantic_ending_group_unresolved` and the run fails.
