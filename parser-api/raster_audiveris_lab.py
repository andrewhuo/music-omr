#!/usr/bin/env python3
import argparse
import glob
import json
import os
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

VARIANTS = ("gentle", "strong")


def _image_libs():
    import cv2
    import fitz
    import numpy as np

    return cv2, fitz, np


def _local(tag):
    return str(tag).split("}", 1)[-1] if "}" in str(tag) else str(tag)


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _autocontrast(gray):
    _, _, np = _image_libs()
    lo, hi = np.percentile(gray, (1, 99))
    if hi <= lo:
        return gray
    out = (gray.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _normalize_background(gray):
    cv2, _, _ = _image_libs()
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    normalized = cv2.divide(gray, blur, scale=245)
    return _autocontrast(normalized)


def _clean_gentle(gray):
    cv2, _, _ = _image_libs()
    normalized = _normalize_background(gray)
    denoised = cv2.medianBlur(normalized, 3)
    return _autocontrast(denoised)


def _repair_staff_gaps(binary_inv):
    cv2, _, _ = _image_libs()
    h, w = binary_inv.shape[:2]
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 80), 1))
    repaired = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, horizontal, iterations=1)
    tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_OPEN, tiny, iterations=1)
    return repaired


def _clean_strong(gray):
    cv2, _, _ = _image_libs()
    normalized = _normalize_background(gray)
    thresholded = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )
    repaired = _repair_staff_gaps(thresholded)
    return 255 - repaired


def _render_page(input_pdf, page_number, output_dir, zoom=4.0):
    cv2, fitz, np = _image_libs()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(input_pdf)
    try:
        index = int(page_number) - 1
        if index < 0 or index >= doc.page_count:
            raise ValueError(f"page_out_of_range:{page_number}")
        page = doc[index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return {
            "bgr": bgr,
            "gray": gray,
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "pixel_width": int(pix.width),
            "pixel_height": int(pix.height),
            "zoom": float(zoom),
        }
    finally:
        doc.close()


def _estimate_skew(gray):
    cv2, _, np = _image_libs()
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(60, w // 16),
        minLineLength=max(60, w // 10),
        maxLineGap=max(8, w // 300),
    )
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if abs(float(x2) - float(x1)) < 1:
                continue
            angle = float(np.degrees(np.arctan2(float(y2) - float(y1), float(x2) - float(x1))))
            if abs(angle) <= 6:
                angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _rotate(gray, angle):
    cv2, _, _ = _image_libs()
    if abs(float(angle)) < 0.2:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _staff_rows(gray):
    """Return rows belonging to likely five-line staff groups."""
    cv2, _, np = _image_libs()
    h, w = gray.shape[:2]
    normalized = _normalize_background(gray)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, w // 70), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    projection = (horizontal > 0).sum(axis=1)
    threshold = max(8, int(w * 0.018))
    active = projection >= threshold
    runs = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, h - 1))
    centers = [int(round((a + b) / 2.0)) for a, b in runs if b - a <= 8]
    rows = set()
    for index in range(max(0, len(centers) - 4)):
        group = centers[index:index + 5]
        distances = [group[i + 1] - group[i] for i in range(4)]
        spacing = float(np.median(distances))
        if 5 <= spacing <= max(100, h // 20) and max(distances) - min(distances) <= max(3.0, spacing * 0.35):
            rows.update(group)
    return sorted(rows), binary, horizontal


def _repair_staff_rows(normalized, binary, rows):
    cv2, _, np = _image_libs()
    h, w = normalized.shape[:2]
    cleaned = normalized.copy()
    repaired_pixels = 0
    max_gap = max(5, min(18, w // 240))
    for row in rows:
        top = max(0, row - 1)
        bottom = min(h, row + 2)
        line = (binary[top:bottom] > 0).any(axis=0)
        index = 0
        while index < w:
            if line[index]:
                index += 1
                continue
            start = index
            while index < w and not line[index]:
                index += 1
            length = index - start
            if start > 0 and index < w and length <= max_gap:
                cleaned[top:bottom, start:index] = np.minimum(cleaned[top:bottom, start:index], 55)
                repaired_pixels += length * (bottom - top)
    return cleaned, repaired_pixels


def build_page_fallback(input_pdf, page_number, output_dir, zoom=4.0):
    cv2, _, np = _image_libs()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    render = _render_page(input_pdf, page_number, out, zoom)
    original = render["bgr"]
    cv2.imwrite(str(out / "original.png"), original)
    gray = render["gray"]
    normalized = _normalize_background(gray)
    cv2.imwrite(str(out / "grayscale_normalized.png"), normalized)
    skew = _estimate_skew(normalized)
    deskewed = _rotate(normalized, -skew)
    rows, binary, horizontal = _staff_rows(deskewed)
    cleaned, repaired_pixels = _repair_staff_rows(deskewed, binary, rows)
    cv2.imwrite(str(out / "medium_cleaned.png"), cleaned)

    debug = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
    for row in rows:
        cv2.line(debug, (0, row), (debug.shape[1] - 1, row), (0, 180, 0), 2)
    cv2.imwrite(str(out / "staff_candidates.png"), debug)
    _image_to_pdf(out / "medium_cleaned.png", out / "medium_input.pdf", render["page_width"], render["page_height"])
    report = {
        "status": "input_ready",
        "page": int(page_number),
        "cleanup": "medium_staff_line_repair",
        "zoom": float(zoom),
        "page_width": render["page_width"],
        "page_height": render["page_height"],
        "pixel_width": render["pixel_width"],
        "pixel_height": render["pixel_height"],
        "estimated_skew_degrees": round(float(skew), 3),
        "staff_line_rows": len(rows),
        "repaired_pixels": int(repaired_pixels),
        "files": {
            "original": "original.png",
            "grayscale_normalized": "grayscale_normalized.png",
            "medium_cleaned": "medium_cleaned.png",
            "staff_candidates": "staff_candidates.png",
            "medium_input_pdf": "medium_input.pdf",
        },
    }
    _write_json(out / "fallback_report.json", report)
    return report


def _render_first_page(input_pdf, output_dir, zoom):
    cv2, fitz, np = _image_libs()
    doc = fitz.open(input_pdf)
    try:
        if doc.page_count != 1:
            return None, {
                "status": "skipped",
                "reason": "raster_test_mode_supports_one_page_only",
                "page_count": int(doc.page_count),
            }
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        original_png = Path(output_dir) / "original_page.png"
        cv2.imwrite(str(original_png), bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return {
            "gray": gray,
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "pixel_width": int(pix.width),
            "pixel_height": int(pix.height),
            "zoom": float(zoom),
        }, None
    finally:
        doc.close()


def _image_to_pdf(image_path, pdf_path, page_width, page_height):
    _, fitz, _ = _image_libs()
    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)
    page.insert_image(fitz.Rect(0, 0, page_width, page_height), filename=str(image_path))
    doc.save(str(pdf_path))
    doc.close()


def build_inputs(input_pdf, output_dir, zoom=4.0):
    cv2, _, _ = _image_libs()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    render, skipped = _render_first_page(input_pdf, out, zoom)
    if skipped:
        _write_json(out / "raster_report.json", skipped)
        return skipped

    gray = render["gray"]
    report = {
        "status": "inputs_ready",
        "reason": "ok",
        "page_count": 1,
        "page_width": render["page_width"],
        "page_height": render["page_height"],
        "pixel_width": render["pixel_width"],
        "pixel_height": render["pixel_height"],
        "zoom": render["zoom"],
        "variants": {},
    }

    cleaners = {"gentle": _clean_gentle, "strong": _clean_strong}
    for variant, cleaner in cleaners.items():
        cleaned = cleaner(gray)
        cleaned_png = out / f"{variant}_cleaned.png"
        input_pdf_path = out / f"{variant}_input.pdf"
        cv2.imwrite(str(cleaned_png), cleaned)
        _image_to_pdf(cleaned_png, input_pdf_path, render["page_width"], render["page_height"])
        report["variants"][variant] = {
            "input_pdf": f"raster/{variant}_input.pdf",
            "cleaned_png": f"raster/{variant}_cleaned.png",
            "status": "input_ready",
        }

    _write_json(out / "raster_report.json", report)
    return report


def _inspect_mxl(path):
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


def _inspect_out_dir(out_dir):
    candidates = sorted(glob.glob(os.path.join(out_dir, "**", "*.mxl"), recursive=True))
    if not candidates:
        return {
            "success": False,
            "reason": "no_mxl_candidates",
            "parts_count": 0,
            "measures_count": 0,
            "mxl_path": None,
        }
    rows = [(path, _inspect_mxl(path)) for path in candidates]
    path, best = max(rows, key=lambda row: (row[1]["measures"], row[1]["parts"]))
    return {
        "success": best["parts"] > 0 and best["measures"] > 0,
        "reason": best.get("error") or "ok",
        "parts_count": int(best["parts"]),
        "measures_count": int(best["measures"]),
        "mxl_path": path,
    }


def _find_omr(out_dir):
    candidates = sorted(glob.glob(os.path.join(out_dir, "**", "*.omr"), recursive=True))
    return candidates[0] if candidates else None


def _merge_existing_report(report_path):
    path = Path(report_path)
    if not path.exists():
        return {"status": "missing_inputs", "variants": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {"status": "bad_report", "variants": {}}
    payload.setdefault("variants", {})
    return payload


def finalize_report(raster_dir):
    raster = Path(raster_dir)
    report_path = raster / "raster_report.json"
    payload = _merge_existing_report(report_path)
    if payload.get("status") == "skipped":
        return payload

    successes = []
    for variant in VARIANTS:
        variant_row = dict((payload.get("variants") or {}).get(variant) or {})
        out_dir = raster / f"{variant}_audiveris"
        result = _inspect_out_dir(str(out_dir))
        omr_path = _find_omr(str(out_dir))
        boxed_png = raster / f"{variant}_boxed.png"
        boxed_pdf = raster / f"{variant}_boxed.pdf"
        variant_row.update(
            {
                "success": bool(result["success"]),
                "reason": str(result["reason"]),
                "parts_count": int(result["parts_count"]),
                "measures_count": int(result["measures_count"]),
                "mxl_path": result.get("mxl_path"),
                "omr_path": omr_path,
                "boxed_png": f"raster/{variant}_boxed.png" if boxed_png.exists() else None,
                "boxed_pdf": f"raster/{variant}_boxed.pdf" if boxed_pdf.exists() else None,
            }
        )
        if variant_row["success"]:
            successes.append(variant)
        payload["variants"][variant] = variant_row

    payload["status"] = "success" if successes else "failed"
    payload["successful_variants"] = successes
    payload["best_variant"] = successes[0] if successes else None
    _write_json(report_path, payload)
    return payload


def copy_boxed_debug(raster_dir, variant):
    raster = Path(raster_dir)
    coord = raster / f"{variant}_coordinate_debug" / "coordinate_debug_page_1.png"
    if coord.exists():
        shutil.copyfile(coord, raster / f"{variant}_boxed.png")
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--input-pdf", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--zoom", type=float, default=4.0)

    boxed = sub.add_parser("copy-boxed")
    boxed.add_argument("--raster-dir", required=True)
    boxed.add_argument("--variant", choices=VARIANTS, required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--raster-dir", required=True)

    fallback = sub.add_parser("fallback-page")
    fallback.add_argument("--input-pdf", required=True)
    fallback.add_argument("--page", required=True, type=int)
    fallback.add_argument("--output-dir", required=True)
    fallback.add_argument("--zoom", type=float, default=4.0)

    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_inputs(args.input_pdf, args.output_dir, args.zoom), sort_keys=True))
    elif args.command == "fallback-page":
        print(json.dumps(build_page_fallback(args.input_pdf, args.page, args.output_dir, args.zoom), sort_keys=True))
    elif args.command == "copy-boxed":
        print(json.dumps({"copied": copy_boxed_debug(args.raster_dir, args.variant)}, sort_keys=True))
    else:
        print(json.dumps(finalize_report(args.raster_dir), sort_keys=True))


if __name__ == "__main__":
    main()
