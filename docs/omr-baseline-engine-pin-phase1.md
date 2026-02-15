# OMR Baseline Snapshot (Before Engine Pin)

- Branch: `codex/engine-pin-phase1`
- Base commit: `c73ce35`
- Workflow signature: `xml-select-v4`
- Known failing run id: `57452085302`
- Run URL (fill if needed): `https://github.com/<owner>/<repo>/actions/runs/57452085302`

## Observed behavior

- Movement XML selection picked `input.mvt2.mxl`.
- Missing XML coverage pages: `1, 4, 5, 8`.
- Mapping errors: `mxl_page_index_missing`.
- Audiveris runtime showed internal graph/null crashes while processing some sheets.

## Why this snapshot exists

This note captures pre-change behavior so post-change runs can be compared quickly.
