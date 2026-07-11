#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def _local(tag):
    return str(tag).split("}", 1)[-1] if "}" in str(tag) else str(tag)


def inspect_mxl(path):
    best = {"parts": 0, "measures": 0, "member": None, "error": None}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml"))
                and not name.lower().endswith("container.xml")
            )
            if not members:
                return {**best, "error": "no_xml_members"}
            for member in members:
                try:
                    root = ET.fromstring(archive.read(member))
                except Exception:
                    continue
                parts = sum(1 for child in list(root) if _local(child.tag) == "part")
                measures = sum(1 for node in root.iter() if _local(node.tag) == "measure")
                if (measures, parts) > (best["measures"], best["parts"]):
                    best = {
                        "parts": int(parts),
                        "measures": int(measures),
                        "member": member,
                        "error": None,
                    }
    except Exception as exc:
        return {**best, "error": f"mxl_open_failed:{type(exc).__name__}"}
    if best["parts"] <= 0 or best["measures"] <= 0:
        best["error"] = "parts_or_measures_zero"
    return best


def inspect_page_output(out_dir):
    candidates = sorted(glob.glob(os.path.join(out_dir, "**", "*.mxl"), recursive=True))
    omr_candidates = sorted(glob.glob(os.path.join(out_dir, "**", "*.omr"), recursive=True))
    if not candidates:
        return {
            "success": False,
            "reason": "no_mxl_candidates",
            "omr_created": bool(omr_candidates),
            "omr_path": omr_candidates[0] if omr_candidates else None,
            "mxl_created": False,
            "parts_count": 0,
            "measures_count": 0,
            "mxl_path": None,
        }
    rows = [(path, inspect_mxl(path)) for path in candidates]
    path, best = max(rows, key=lambda row: (row[1]["measures"], row[1]["parts"]))
    return {
        "success": best["parts"] > 0 and best["measures"] > 0,
        "reason": best.get("error") or "ok",
        "omr_created": bool(omr_candidates),
        "omr_path": omr_candidates[0] if omr_candidates else None,
        "mxl_created": True,
        "parts_count": int(best["parts"]),
        "measures_count": int(best["measures"]),
        "mxl_path": path,
    }


def _read_log(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _log_clues(log_paths):
    text = "\n".join(_read_log(path) for path in log_paths).lower()
    clues = []
    checks = [
        ("transcription_did_not_complete", r"transcription did not complete"),
        ("export_failed", r"could not export|export failed"),
        ("no_mxl_files", r"no_mxl|no mxl|no .*mxl"),
        ("no_parts_or_measures", r"parts_or_measures_zero|no parts|no measures"),
        ("staff_system_warning", r"staff|stave|system"),
        ("measure_warning", r"measure"),
        ("timeout_or_killed", r"timeout|timed out|killed|137"),
        ("crash_or_exception", r"exception|traceback|error: process completed with exit code"),
    ]
    for name, pattern in checks:
        if re.search(pattern, text):
            clues.append(name)
    return sorted(set(clues))


def _final_failure_stage(result):
    if not result.get("omr_created"):
        return "omr_missing"
    if not result.get("mxl_created"):
        return "mxl_missing_after_omr_created"
    if int(result.get("parts_count") or 0) <= 0 or int(result.get("measures_count") or 0) <= 0:
        return "mxl_has_no_parts_or_measures"
    if not result.get("success"):
        return str(result.get("reason") or "unknown_failure")
    return "ok"


def _recommended_next_try(final_stage, log_clues, image_clues):
    if "staff_lines_weak" in image_clues or "staff_lines_broken" in image_clues:
        return "medium_staff_line_repair"
    if "major_skew" in image_clues or "minor_skew" in image_clues:
        return "deskew_then_medium_raster"
    if "dark_borders" in image_clues:
        return "crop_or_border_cleanup"
    if final_stage in ("mxl_missing_after_omr_created", "mxl_has_no_parts_or_measures"):
        return "medium_raster_cleanup"
    if "crash_or_exception" in log_clues or "timeout_or_killed" in log_clues:
        return "retry_or_split_page"
    return "inspect_page_manually"


def write_attempt_status(
    out_dir,
    status_path,
    page,
    attempt,
    log_path,
    attempt_kind="normal",
    artifact_dir=None,
):
    result = inspect_page_output(out_dir)
    previous = {}
    path = Path(status_path)
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    first_failed = bool(previous) and not bool(previous.get("success"))
    payload = {
        "page": int(page),
        "attempts": int(attempt),
        "attempt_kind": str(attempt_kind),
        "success": bool(result["success"]),
        "recovered_on_retry": bool(int(attempt) > 1 and first_failed and result["success"]),
        "recovered_on_raster_fallback": bool(
            str(attempt_kind) == "medium_raster_cleanup" and result["success"]
        ),
        "reason": str(result["reason"]),
        "final_failure_stage": _final_failure_stage(result),
        "omr_created": bool(result.get("omr_created")),
        "omr_path": result.get("omr_path"),
        "mxl_created": bool(result.get("mxl_created")),
        "parts_count": int(result["parts_count"]),
        "measures_count": int(result["measures_count"]),
        "mxl_path": result.get("mxl_path"),
        "log_paths": list(previous.get("log_paths") or []) + [str(log_path)],
    }
    attempt_rows = list(previous.get("attempt_results") or [])
    attempt_rows.append(
        {
            "attempt": int(attempt),
            "attempt_kind": str(attempt_kind),
            "success": bool(result["success"]),
            "reason": str(result["reason"]),
            "final_failure_stage": _final_failure_stage(result),
            "omr_created": bool(result.get("omr_created")),
            "mxl_created": bool(result.get("mxl_created")),
            "parts_count": int(result["parts_count"]),
            "measures_count": int(result["measures_count"]),
            "mxl_path": result.get("mxl_path"),
            "log_path": str(log_path),
        }
    )
    payload["attempt_results"] = attempt_rows
    if artifact_dir:
        artifact_root = Path(artifact_dir)
        payload["fallback_artifacts"] = [
            f"artifacts/page_raster_fallback/page_{int(page):04d}/{path.relative_to(artifact_root)}"
            for path in sorted(artifact_root.rglob("*"))
            if path.is_file()
        ]
    payload["log_clues"] = _log_clues(payload["log_paths"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _image_libs():
    import cv2
    import fitz
    import numpy as np

    return cv2, fitz, np


def _page_image_diagnosis(page_pdf_path, output_dir, page):
    if not page_pdf_path or not Path(page_pdf_path).exists():
        return {"image_clues": ["page_pdf_missing"], "debug_images": []}
    cv2, fitz, np = _image_libs()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(page_pdf_path))
    try:
        page_obj = doc[0]
        pix = page_obj.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    finally:
        doc.close()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    clues = []
    dark_ratio = float(np.mean(gray < 80))
    border = max(8, min(h, w) // 35)
    border_pixels = np.concatenate(
        [gray[:border, :].ravel(), gray[-border:, :].ravel(), gray[:, :border].ravel(), gray[:, -border:].ravel()]
    )
    border_dark_ratio = float(np.mean(border_pixels < 90))
    if dark_ratio > 0.18:
        clues.append("noisy_or_dark_page")
    if border_dark_ratio > max(0.10, dark_ratio * 1.8):
        clues.append("dark_borders")

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_rows = []
    total_line_width = 0
    for contour in contours:
        x, y, ww, hh = cv2.boundingRect(contour)
        if ww < w * 0.12 or hh > 12:
            continue
        line_rows.append((x, y, ww, hh))
        total_line_width += ww
    line_count = len(line_rows)
    line_strength = float(total_line_width / max(1, w * max(1, line_count)))
    if line_count < 15:
        clues.append("staff_lines_weak")
    if line_strength < 0.35:
        clues.append("staff_lines_broken")

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(80, w // 12), minLineLength=max(50, w // 8), maxLineGap=12)
    angles = []
    if lines is not None:
        for row in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in row]
            if abs(x2 - x1) < 1:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= 8:
                angles.append(angle)
    skew = float(np.median(angles)) if angles else 0.0
    if abs(skew) > 2.0:
        clues.append("major_skew")
    elif abs(skew) > 0.8:
        clues.append("minor_skew")

    debug = bgr.copy()
    for x, y, ww, hh in line_rows:
        cv2.rectangle(debug, (x, y), (x + ww, y + hh), (0, 180, 0), 2)
    if "dark_borders" in clues:
        cv2.rectangle(debug, (0, 0), (w - 1, h - 1), (0, 0, 255), max(3, border // 3))
    image_name = f"page_{int(page):04d}_staff_candidates.png"
    image_path = out / image_name
    cv2.imwrite(str(image_path), debug)
    return {
        "image_clues": sorted(set(clues)),
        "image_metrics": {
            "dark_ratio": round(dark_ratio, 4),
            "border_dark_ratio": round(border_dark_ratio, 4),
            "staff_line_candidate_count": int(line_count),
            "staff_line_strength": round(line_strength, 4),
            "estimated_skew_degrees": round(skew, 3),
        },
        "debug_images": [f"artifacts/page_failure_debug/{image_name}"],
    }


def build_report(
    manifest_path,
    status_dir,
    output_path,
    full_document_success,
    page_pdf_dir=None,
    diagnostic_output_dir=None,
):
    manifest = {}
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    entries = {
        int(row.get("page_number") or int(row.get("page_index", 0)) + 1): row
        for row in (manifest.get("entries") or [])
        if isinstance(row, dict)
    }
    statuses = {}
    for path in sorted(Path(status_dir).glob("page_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            statuses[int(row["page"])] = row
        except Exception:
            continue

    page_numbers = sorted(set(entries) | set(statuses))
    page_results = []
    successful = []
    failed = []
    recovered = []
    recovered_on_raster_fallback = []
    for page in page_numbers:
        status = dict(statuses.get(page) or {})
        entry = dict(entries.get(page) or {})
        success = entry.get("status") == "ok"
        reason = "ok" if success else str(entry.get("error") or status.get("reason") or "unknown")
        log_clues = list(status.get("log_clues") or _log_clues(status.get("log_paths") or []))
        image_diag = {"image_clues": [], "image_metrics": {}, "debug_images": []}
        if not success and page_pdf_dir and diagnostic_output_dir:
            page_pdf = Path(page_pdf_dir) / f"page_{page:04d}.pdf"
            try:
                image_diag = _page_image_diagnosis(page_pdf, diagnostic_output_dir, page)
            except Exception as exc:
                image_diag = {"image_clues": [f"image_diagnosis_failed:{type(exc).__name__}"], "debug_images": []}
        image_clues = list(image_diag.get("image_clues") or [])
        final_stage = str(
            entry.get("final_failure_stage")
            or status.get("final_failure_stage")
            or ("ok" if success else "unknown_failure")
        )
        if success:
            successful.append(page)
        else:
            failed.append(page)
        if success and status.get("recovered_on_retry"):
            recovered.append(page)
        if success and status.get("recovered_on_raster_fallback"):
            recovered_on_raster_fallback.append(page)
        page_results.append(
            {
                "page": page,
                "status": "success" if success else "failed",
                "reason": reason,
                "attempts": int(status.get("attempts") or 0),
                "attempt_results": list(status.get("attempt_results") or []),
                "recovered_on_retry": bool(success and status.get("recovered_on_retry")),
                "recovered_on_raster_fallback": bool(
                    success and status.get("recovered_on_raster_fallback")
                ),
                "final_failure_stage": final_stage,
                "omr_created": bool(entry.get("omr_created") or status.get("omr_created")),
                "mxl_created": bool(entry.get("mxl_created") or status.get("mxl_created")),
                "parts_count": int(entry.get("parts_count") or status.get("parts_count") or 0),
                "measures_count": int(entry.get("measures_count") or status.get("measures_count") or 0),
                "system_count": len(entry.get("system_starts") or []),
                "log_clues": log_clues,
                "image_clues": image_clues,
                "image_metrics": dict(image_diag.get("image_metrics") or {}),
                "debug_images": list(image_diag.get("debug_images") or []),
                "recommended_next_try": (
                    "none"
                    if success
                    else _recommended_next_try(final_stage, log_clues, image_clues)
                ),
                "log_paths": list(status.get("log_paths") or []),
                "log_artifact": f"artifacts/page_omr_logs/page_{page:04d}.log",
                "fallback_artifacts": list(status.get("fallback_artifacts") or []),
            }
        )

    payload = {
        "version": "page-omr-recovery-v1",
        "full_document_success": bool(full_document_success),
        "total_pages": len(page_numbers),
        "successful_pages": successful,
        "failed_pages": failed,
        "recovered_on_retry_pages": recovered,
        "recovered_on_raster_fallback_pages": recovered_on_raster_fallback,
        "page_results": page_results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_compact_logs(report_path, output_dir):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in report.get("page_results") or []:
        page = int(row.get("page") or 0)
        failed = row.get("status") != "success"
        line_limit = 400 if failed else 80
        chunks = []
        for raw_path in row.get("log_paths") or []:
            path = Path(raw_path)
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            chunks.append(f"=== {path.name} (last {line_limit} lines) ===")
            chunks.extend(lines[-line_limit:])
        (out / f"page_{page:04d}.log").write_text("\n".join(chunks) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--out-dir", required=True)
    validate.add_argument("--status-path", required=True)
    validate.add_argument("--page", required=True, type=int)
    validate.add_argument("--attempt", required=True, type=int)
    validate.add_argument("--log-path", required=True)
    validate.add_argument("--attempt-kind", default="normal")
    validate.add_argument("--artifact-dir", required=False)

    report = sub.add_parser("report")
    report.add_argument("--manifest", required=True)
    report.add_argument("--status-dir", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--full-document-success", choices=("0", "1"), required=True)
    report.add_argument("--log-output-dir", required=True)
    report.add_argument("--page-pdf-dir", required=False)
    report.add_argument("--diagnostic-output-dir", required=False)

    args = parser.parse_args()
    if args.command == "validate":
        payload = write_attempt_status(
            args.out_dir,
            args.status_path,
            args.page,
            args.attempt,
            args.log_path,
            args.attempt_kind,
            args.artifact_dir,
        )
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload["success"] else 1)

    payload = build_report(
        args.manifest,
        args.status_dir,
        args.output,
        args.full_document_success == "1",
        page_pdf_dir=args.page_pdf_dir,
        diagnostic_output_dir=args.diagnostic_output_dir,
    )
    write_compact_logs(args.output, args.log_output_dir)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
