# OMR Engine A/B Test (Audiveris vs Homr)

This experiment is isolated from the production workflow.

- Production workflow remains: `/Users/andrew/Desktop/music-omr/.github/workflows/audiveris.yml`
- Experiment workflow: `/Users/andrew/Desktop/music-omr/.github/workflows/omr-homr-ab.yml`
- Branch for experiment: `codex/homr-ab-test`

## Goal

Compare Audiveris (control) and Homr (experiment) on the same input PDF while reusing existing OMR geometry for annotation.

## What stays the same

- Audiveris control path runs exactly as before.
- Strict gate (`strict_xml_pages_ok`) still applies to the Audiveris control path.

## What is new

- Homr runs page-by-page on PNG images generated from split pages.
- A Homr-specific manifest is generated: `/tmp/work/mxl_page_manifest_homr.json`.
- A second annotation pass is produced from Homr manifest + existing `.omr` geometry:
  - `/tmp/work/annotated_homr.pdf`
- A comparison file is generated:
  - `/tmp/work/engine_compare_summary.json`

## Runtime controls

Environment variables in the experiment workflow:

- `ENGINE_COMPARE_MODE=audiveris_vs_homr`
- `HOMR_ENABLE=1`
- `HOMR_TIMEOUT_SEC=900`
- `HOMR_IMAGE` (optional; if empty, local `homr`/`python -m homr` is used)

## Artifact paths

Flat layout (old Homr test outputs are cleared each run):

- Base prefix: `gs://music-omr-bucket-777135743132/output/homr-test/`
- Top-level objects (exact):
  - `.../audiveris_out.pdf`
  - `.../artifacts/`
  - `.../homr/`
- Control artifacts:
  - `.../audiveris_out.pdf`
  - `.../artifacts/measure_mapping_debug.json`
  - `.../artifacts/mxl_page_manifest.json`
  - `.../artifacts/omr_debug_bundle.tar.gz`
- Homr artifacts:
  - `.../homr/homr_out.pdf`
  - `.../homr/measure_mapping_debug.json`
  - `.../homr/mxl_page_manifest.json`
  - `.../homr/homr_debug_bundle.tar.gz`
- Compare + run metadata:
  - `.../artifacts/engine_compare_summary.json`
  - `.../artifacts/run_info.json`

## Notes

- This is an evaluation path. Homr failures are diagnostic and do not replace control pass/fail policy.
- Full replacement of Audiveris is out of scope here because parser geometry currently depends on Audiveris `.omr` internals.
