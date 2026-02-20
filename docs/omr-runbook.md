# OMR Operator Runbook

This runbook is for day-to-day operation of:

- Workflow: `Audiveris OMR + Measure Label Debug`
- File: `/Users/andrew/Desktop/music-omr/.github/workflows/audiveris.yml`

## What this pipeline does

1. Downloads one input PDF from GCS.
2. Runs Audiveris OMR with pinned runtime image.
3. Builds per-page XML manifest in strict mode.
4. Annotates the PDF with guides/measure labels.
5. Uploads output files and debug artifacts to GCS.

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

Run-scoped prefix (for traceability):

- `gs://music-omr-bucket-777135743132/output/runs/<run_id>/`

Artifacts written per run:

- `annotated.pdf`
- `measure_mapping_debug.json`
- `mxl_page_manifest.json`
- `omr_debug_bundle.tar.gz`

Latest convenience copy:

- `gs://music-omr-bucket-777135743132/output/annotated.pdf`

## Success criteria

Look for these lines in logs:

- `WORKFLOW_SIGNATURE=engine-pin-v1`
- `AUDIVERIS_VERSION_NUM=5.9.0`
- `PAGE_SPLIT total_pages=<N>`
- `MXL_PAGE_MANIFEST path=... entries=<N> missing=0`
- `SUMMARY strict_xml_pages_ok=<N>/<N>`

If strict coverage fails, the run should fail.

## What to collect when a run fails

Download and share these step logs:

- `8_Run Audiveris OMR.txt`
- `13_Build page-split XML manifest (strict mode).txt`
- `16_Mapping debug summary.txt`
- `18_Upload outputs and debug artifacts.txt`

Also share run-scoped debug bundle URI:

- `gs://music-omr-bucket-777135743132/output/runs/<run_id>/omr_debug_bundle.tar.gz`

## Troubleshooting note: system-count mismatch

In strict mode, if MusicXML system starts and OMR system count do not match on a page
(for example expand/compress mapping would be needed), the run now fails strict coverage
instead of drawing repeated labels.

Strict mode now first tries an exact-safe merge of per-page movement XML starts before
failing. If exact-safe merge cannot be proven, strict still fails.

Look for:

- `mapping_reason=manifest_system_count_mismatch` (manifest mode)
- `mapping_reason=mxl_system_count_mismatch` (full-book mode)
