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
- Storage mode is **per-run** (`output/runs/<run_id>/...`) with legacy single-latest fallback.

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

## Output Contract (Per-Run)

Base prefix:
- `gs://music-omr-bucket-777135743132/output`

Each run writes to its own prefix:

- `gs://music-omr-bucket-777135743132/output/runs/<run_id>/audiveris_out.pdf`
- `gs://music-omr-bucket-777135743132/output/runs/<run_id>/artifacts/run_info.json`
- `gs://music-omr-bucket-777135743132/output/runs/<run_id>/artifacts/mapping_summary.json`
- relabel writes: `.../output/runs/<run_id>/audiveris_out_corrected.pdf`

Legacy single-latest paths may still be read for older jobs during migration.

## Backend API (Workflow Dispatch)

Implemented in `omr-worker/worker.py`.

Endpoints:

- `POST /api/omr/jobs`
- `GET /api/omr/jobs/{job_id}`
- `GET /api/omr/jobs/{job_id}/state`
- `POST /api/omr/jobs/{job_id}/relabel`
- `POST /api/omr/jobs/{job_id}/cleanup` (idempotent artifact cleanup helper)

The relabel flow is fast and does not rerun Audiveris:

1. Run OMR once with `POST /api/omr/jobs`.
2. Read clickable systems from `/state`.
3. Submit edit(s) to `/relabel`.
4. Download `audiveris_out_corrected.pdf`.

### Smoke test (no frontend required)

```bash
python3 /Users/andrew/Desktop/music-omr/omr-worker/scripts/relabel_smoke.py \
  --worker-url "https://<your-worker-url>" \
  --pdf-gcs-uri "gs://music-omr-bucket-777135743132/input/test-input/01_single_staff.pdf"
```

See full API details in `/Users/andrew/Desktop/music-omr/docs/omr-api.md`.
