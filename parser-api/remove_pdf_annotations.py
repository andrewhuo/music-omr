#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import fitz


def remove_annotations(input_pdf: str, output_pdf: str, report_path: str) -> dict:
    src = Path(input_pdf)
    out = Path(output_pdf)
    report = Path(report_path)
    doc = fitz.open(str(src))
    pages = []
    total = 0
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            removed = 0
            types = {}
            annot = page.first_annot
            while annot:
                next_annot = annot.next
                annot_type = annot.type[1] if annot.type and len(annot.type) > 1 else "unknown"
                types[annot_type] = types.get(annot_type, 0) + 1
                page.delete_annot(annot)
                removed += 1
                annot = next_annot
            total += removed
            pages.append(
                {
                    "page": page_index + 1,
                    "annotations_removed": removed,
                    "annotation_types": types,
                }
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"garbage": 4, "deflate": True}
        if total <= 0:
            doc.save(str(out))
        else:
            doc.save(str(out), **save_kwargs)
    finally:
        doc.close()

    payload = {
        "status": "ok",
        "input_pdf": str(src),
        "output_pdf": str(out),
        "annotations_removed": total,
        "clean_pdf_used_for_omr": total > 0,
        "final_pdf_uses_original": True,
        "page_count": len(pages),
        "pages": pages,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(json.dumps(remove_annotations(args.input_pdf, args.output_pdf, args.report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
