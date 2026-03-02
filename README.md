# music-omr

Music OMR pipeline for annotating score PDFs with **system-start guides** and **measure labels**.

This repo includes:
- a GitHub Actions workflow that runs Audiveris + strict XML mapping + PDF annotation
- a backend API worker that dispatches workflow runs and reports status/artifact URIs

## Current Status

- Baseline numbering mode is active (`baseline_continuous_v1`).
- Guide rendering is system-level:
  - one long vertical guide per system
  - one label per system
- Strict quality gates are active:
  - XML coverage (`strict_xml_pages_ok`)
  - system drift guard (`SYSTEM_COUNT_DRIFT`)
- Storage mode is **single-latest** (each new run overwrites previous output).

## Repo Layout

- `.github/workflows/audiveris.yml`  
  Production workflow (`Audiveris OMR + Measure Label Debug`)
- `.github/workflows/omr-homr-ab.yml`  
  Experimental A/B workflow (Audiveris vs Homr)
- `parser-api/annotate_guides_from_omr.py`  
  Core PDF guide/label annotation logic
- `omr-worker/worker.py`  
  Backend API for workflow dispatch + status polling
- `docs/omr-runbook.md`  
  Operator runbook (logs, failure triage)
- `docs/omr-api.md`  
  API integration details

## How It Works

1. Input PDF is provided via `gs://...` URI.
2. Workflow downloads PDF and runs Audiveris OMR.
3. Workflow builds strict page-to-XML manifest.
4. Parser annotates PDF with system guides + labels.
5. Workflow uploads compact artifacts to GCS.

## Run the Production Workflow (Manual)

1. Open GitHub Actions in `andrewhuo/music-omr`.
2. Select workflow: `Audiveris OMR + Measure Label Debug`.
3. Click **Run workflow**.
4. Choose branch `main`.
5. Provide only:
   - `pdf_gcs_uri` = `gs://.../your.pdf`
6. Run.

## Output Contract (Single-Latest)

Base prefix:
- `gs://music-omr-bucket-777135743132/output`

Each run clears previous output and writes only:

- `gs://music-omr-bucket-777135743132/output/audiveris_out.pdf`
- `gs://music-omr-bucket-777135743132/output/artifacts/run_info.json`
- `gs://music-omr-bucket-777135743132/output/artifacts/mapping_summary.json`

No `runs/<run_id>/...` folders are used by the production workflow.

## Backend API (Workflow Dispatch)

Implemented in `omr-worker/worker.py`.

### `POST /api/omr/jobs`

Request:
```json
{
  "pdf_gcs_uri": "gs://music-omr-bucket-777135743132/input/test.pdf"
}
Response (202):

{
  "job_id": "uuid",
  "status": "queued",
  "run_id": 12345678901,
  "workflow": "audiveris.yml",
  "ref": "main",
  "pdf_gcs_uri": "gs://...",
  "status_url": "/api/omr/jobs/<job_id>",
  "run_url": "https://github.com/...",
  "artifacts": {
    "audiveris_out_pdf": "gs://.../output/audiveris_out.pdf",
    "run_info": "gs://.../output/artifacts/run_info.json",
    "mapping_summary": "gs://.../output/artifacts/mapping_summary.json"
  }
}
GET /api/omr/jobs/{job_id}
Returns run lifecycle status and artifact URIs:

queued | running | succeeded | failed
See full API details in docs/omr-api.md.

Required Backend Environment Variables
GITHUB_TOKEN (required)
GITHUB_OWNER (default: andrewhuo)
GITHUB_REPO (default: music-omr)
GITHUB_WORKFLOW_ID (default: audiveris.yml)
GITHUB_REF (default: main)
OUTPUT_PREFIX (default: gs://music-omr-bucket-777135743132/output)
Optional:

RUN_DISCOVERY_TIMEOUT_SEC (default: 20)
RUN_DISCOVERY_POLL_SEC (default: 2)
Quality Gates
A successful run should include:

SUMMARY strict_xml_pages_ok=N/N
No SUMMARY strict_system_drift ... lines
If strict checks fail, run failure is expected (by design).

Troubleshooting
When a run fails, collect these step logs:

8_Run Audiveris OMR.txt
13_Build page-split XML manifest (strict mode).txt
16_Mapping debug summary.txt
18_Upload outputs and debug artifacts.txt
Also capture upload lines:

uploaded_audiveris_out=...
uploaded_artifact_run_info=...
uploaded_artifact_mapping_summary=...
Then follow docs/omr-runbook.md.

Notes
Production flow is tuned for stable baseline output first.
Homr workflow is currently experimental (A/B path).
Frontend/manual correction UX is planned as a later phase.
