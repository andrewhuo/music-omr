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


def _line_segments(gray):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, w // 90), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []

    def segment_angle(x, y, width, height):
        crop = normalized[max(0, y - 4):min(h, y + height + 4), x:min(w, x + width)]
        edges = cv2.Canny(crop, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(6, width // 20),
            minLineLength=max(12, width // 5),
            maxLineGap=max(3, width // 40),
        )
        candidates = []
        for x1, y1, x2, y2 in lines[:, 0, :] if lines is not None else []:
            dx = float(x2) - float(x1)
            if abs(dx) < 1:
                continue
            angle = float(np.degrees(np.arctan2(float(y2) - float(y1), dx)))
            length = float((dx * dx + (float(y2) - float(y1)) ** 2) ** 0.5)
            if abs(angle) <= 6:
                candidates.append((length, angle))
        return max(candidates, default=(0.0, 0.0))[1]

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < max(30, int(w * 0.08)) or height > 8:
            continue
        crop = normalized[y:min(h, y + height), x:min(w, x + width)]
        segments.append(
            {
                "left": int(x),
                "right": int(x + width - 1),
                "y": float(y + (height - 1) / 2.0),
                "height": int(height),
                "thickness": float(height),
                "angle": round(float(segment_angle(x, y, width, height)), 3),
                "darkness": round(float(np.mean(crop < 180)), 4),
                "length": int(width),
            }
        )
    return segments, binary, horizontal


def _group_staff_segments(segments, image_width, image_height):
    rows = []
    for segment in sorted(segments, key=lambda item: item["y"]):
        if rows and abs(segment["y"] - rows[-1]["y"]) <= 4:
            row = rows[-1]
            row["segments"].append(segment)
            row["left"] = min(row["left"], segment["left"])
            row["right"] = max(row["right"], segment["right"])
            row["angles"].append(segment["angle"])
            row["y"] = (row["y"] * (len(row["segments"]) - 1) + segment["y"]) / len(row["segments"])
        else:
            rows.append(
                {
                    "y": float(segment["y"]),
                    "left": int(segment["left"]),
                    "right": int(segment["right"]),
                    "angles": [segment["angle"]],
                    "segments": [segment],
                }
            )

    groups = []
    used_rows = set()
    max_spacing = max(100, image_height // 20)
    for index in range(max(0, len(rows) - 4)):
        candidate = rows[index:index + 5]
        distances = [candidate[i + 1]["y"] - candidate[i]["y"] for i in range(4)]
        spacing = sum(distances) / 4.0
        if not 5 <= spacing <= max_spacing:
            continue
        if max(distances) - min(distances) > max(3.0, spacing * 0.25):
            continue
        left = max(row["left"] for row in candidate)
        right = min(row["right"] for row in candidate)
        overlap = max(0, right - left + 1) / max(1, min(image_width, max(row["right"] for row in candidate) - min(row["left"] for row in candidate) + 1))
        if overlap < 0.20:
            continue
        thicknesses = [segment["thickness"] for row in candidate for segment in row["segments"]]
        angles = [segment["angle"] for row in candidate for segment in row["segments"]]
        thickness = sum(thicknesses) / max(1, len(thicknesses))
        if max(thicknesses) - min(thicknesses) > max(2.0, thickness):
            continue
        if max(angles) - min(angles) > 0.5:
            continue
        lengths = [segment["length"] for row in candidate for segment in row["segments"]]
        if min(lengths) / max(1.0, max(lengths)) < 0.35:
            continue
        row_ids = tuple(index + offset for offset in range(5))
        if any(row_id in used_rows for row_id in row_ids):
            continue
        used_rows.update(row_ids)
        groups.append(
            {
                "rows": [int(round(row["y"])) for row in candidate],
                "spacing": round(float(spacing), 3),
                "left": int(min(row["left"] for row in candidate)),
                "right": int(max(row["right"] for row in candidate)),
                "overlap": round(float(overlap), 3),
                "thickness": round(float(thickness), 3),
                "angle": round(float(sum(angles) / max(1, len(angles))), 3),
                "length_ratio": round(float(min(lengths) / max(1.0, max(lengths))), 3),
            }
        )
    accepted_rows = sorted({row for group in groups for row in group["rows"]})
    return groups, accepted_rows, rows


def _estimate_skew(gray, staff_groups):
    cv2, _, np = _image_libs()
    if len(staff_groups) < 2:
        return 0.0, "insufficient_staff_groups", 0
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
    staff_rows = [row for group in staff_groups for row in group["rows"]]
    angles = []
    for x1, y1, x2, y2 in lines[:, 0, :] if lines is not None else []:
        if abs(float(x2) - float(x1)) < 1:
            continue
        midpoint = (float(y1) + float(y2)) / 2.0
        if min(abs(midpoint - row) for row in staff_rows) > 10:
            continue
        angle = float(np.degrees(np.arctan2(float(y2) - float(y1), float(x2) - float(x1))))
        if abs(angle) <= 4:
            angles.append(angle)
    if len(angles) < 6:
        return 0.0, "insufficient_staff_line_angles", len(angles)
    median = float(np.median(angles))
    deviation = float(np.median(np.abs(np.asarray(angles) - median)))
    if deviation > 0.25 or abs(median) > 2.0:
        return 0.0, "conflicting_or_large_angle", len(angles)
    return median, "staff_group_consensus", len(angles)


def _rotate(gray, angle):
    cv2, _, _ = _image_libs()
    if abs(float(angle)) < 0.2:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _staff_rows(gray):
    """Return rows belonging to likely five-line staff groups."""
    segments, binary, horizontal = _line_segments(gray)
    _, rows, _ = _group_staff_segments(segments, gray.shape[1], gray.shape[0])
    return rows, binary, horizontal


def _repair_staff_rows(normalized, binary, rows):
    _, _, np = _image_libs()
    h, w = normalized.shape[:2]
    cleaned = normalized.copy()
    repaired_pixels = 0
    repaired_gaps = 0
    repaired_spans = []
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
            left_support = max(0, start - 16) < start and line[max(0, start - 16):start].any()
            right_support = index < w and line[index:min(w, index + 16)].any()
            if start > 0 and index < w and length <= max_gap and left_support and right_support:
                cleaned[top:bottom, start:index] = np.minimum(cleaned[top:bottom, start:index], 55)
                repaired_pixels += length * (bottom - top)
                repaired_gaps += 1
                repaired_spans.append((int(row), int(start), int(index)))
    return cleaned, {
        "repaired_gap_count": repaired_gaps,
        "repaired_length": repaired_pixels,
        "repaired_spans": repaired_spans,
    }


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
    segments, _, _ = _line_segments(normalized)
    staff_groups, rows, candidate_rows = _group_staff_segments(
        segments, normalized.shape[1], normalized.shape[0]
    )
    skew, skew_confidence, skew_samples = _estimate_skew(normalized, staff_groups)
    deskewed = _rotate(normalized, -skew)
    segments, binary, _ = _line_segments(deskewed)
    staff_groups, rows, candidate_rows = _group_staff_segments(
        segments, deskewed.shape[1], deskewed.shape[0]
    )
    cleaned, repair_info = _repair_staff_rows(deskewed, binary, rows)
    cv2.imwrite(str(out / "medium_cleaned.png"), cleaned)

    debug = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
    accepted_rows = set(rows)
    for row in candidate_rows:
        color = (0, 180, 0) if int(round(row["y"])) in accepted_rows else (0, 220, 220)
        cv2.line(debug, (int(row["left"]), int(round(row["y"]))), (int(row["right"]), int(round(row["y"]))), color, 2)
    for row, start, end in repair_info["repaired_spans"]:
        cv2.line(debug, (start, row), (end, row), (0, 0, 255), 3)
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
        "skew_confidence": skew_confidence,
        "skew_angle_samples": int(skew_samples),
        "coordinate_transform": {
            "type": "rotation_about_page_center",
            "angle_degrees_applied_to_clean_copy": round(float(-skew), 3),
            "center_pixels": [
                int(render["pixel_width"] // 2),
                int(render["pixel_height"] // 2),
            ],
            "mapping_inverse_applied": False,
        },
        "candidate_segment_count": int(len(segments)),
        "accepted_candidate_count": int(len(rows)),
        "rejected_candidate_count": int(max(0, len(candidate_rows) - len(rows))),
        "staff_groups": staff_groups,
        "staff_line_rows": len(rows),
        "repaired_gap_count": int(repair_info["repaired_gap_count"]),
        "repaired_length": int(repair_info["repaired_length"]),
        "rejection_reasons": [
            "not_in_five_line_group",
            "uneven_spacing_or_low_overlap",
            "large_or_unsupported_gap_left_unchanged",
        ],
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
