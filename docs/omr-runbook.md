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

Top-level objects (exact):

- `gs://music-omr-bucket-777135743132/output/audiveris_out.pdf`
- `gs://music-omr-bucket-777135743132/output/artifacts/`

Artifacts folder contents:

- `gs://music-omr-bucket-777135743132/output/artifacts/measure_mapping_debug.json`
- `gs://music-omr-bucket-777135743132/output/artifacts/mxl_page_manifest.json`
- `gs://music-omr-bucket-777135743132/output/artifacts/omr_debug_bundle.tar.gz`
- `gs://music-omr-bucket-777135743132/output/artifacts/run_info.json`

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

Also share debug bundle URI:

- `gs://music-omr-bucket-777135743132/output/artifacts/omr_debug_bundle.tar.gz`

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
