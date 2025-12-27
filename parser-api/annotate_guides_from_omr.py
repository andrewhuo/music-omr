#!/usr/bin/env python3
import math
import sys
import re
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF


GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0


_SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


def _sorted_sheet_xml_paths(z: zipfile.ZipFile) -> list[str]:
    """Return sheet#N/sheet#N.xml paths sorted by N."""
    found = []
    for name in z.namelist():
        m = _SHEET_XML_RE.match(name)
        if m:
            found.append((int(m.group(1)), name))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]

def _pct(values: list[float], p: float) -> float:
    """Simple percentile without numpy. p in [0,1]."""
    if not values:
        raise ValueError("empty")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    i = int(round(p * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, i))]

def _best_five_by_spacing(yxs: list[tuple[float, float]], expected_spacing: float) -> list[tuple[float, float]]:
    """
    Pick best 5 lines from candidates by minimizing spacing variance.
    yxs: list of (y, x_min_for_that_line), sorted by y.
    """
    if len(yxs) <= 5:
        return yxs
    best = None
    best_score = float("inf")
    for i in range(0, len(yxs) - 4):
        win = yxs[i:i+5]
        ys = [t[0] for t in win]
        diffs = [ys[j+1] - ys[j] for j in range(4)]
        med = sorted(diffs)[2]
        var = sum(abs(d - med) for d in diffs)
        # prefer windows close to expected spacing when available
        if expected_spacing > 0:
            var += 0.35 * sum(abs(d - expected_spacing) for d in diffs)
        if var < best_score:
            best_score = var
            best = win
    return best if best is not None else yxs[:5]

def _synthesize_five_lines(ys: list[float], expected_spacing: float) -> list[float] | None:
    """
    Given 3-4 observed line y's, build 5 y's using expected_spacing.
    Tries different alignment (which observed line corresponds to which index 0..4).
    """
    ys = sorted(ys)
    if expected_spacing <= 0:
        return None

    tol = max(2.0, expected_spacing * 0.35)

    best = None
    best_hits = -1
    best_err = float("inf")

    # try mapping each observed y to each line index k
    for anchor_y in ys:
        for k in range(5):
            y0 = anchor_y - k * expected_spacing
            pred = [y0 + i * expected_spacing for i in range(5)]

            hits = 0
            err = 0.0
            for y in ys:
                nearest = min(pred, key=lambda t: abs(t - y))
                d = abs(nearest - y)
                if d <= tol:
                    hits += 1
                    err += d
                else:
                    err += 2 * tol + d  # penalize outliers hard

            if hits > best_hits or (hits == best_hits and err < best_err):
                best_hits = hits
                best_err = err
                best = pred

    if best is None or best_hits < 2:
        return None

    # snap predicted lines to nearby observed lines when close
    snapped = []
    for y_pred in best:
        nearest_obs = min(ys, key=lambda t: abs(t - y_pred))
        if abs(nearest_obs - y_pred) <= tol:
            snapped.append(nearest_obs)
        else:
            snapped.append(y_pred)

    snapped = sorted(snapped)
    # sanity: must be increasing and reasonably tall
    if snapped[-1] - snapped[0] < 3.2 * expected_spacing:
        return None
    return snapped

def _parse_sheet(z: zipfile.ZipFile, sheet_xml_path: str):
    data = z.read(sheet_xml_path)
    root = ET.fromstring(data)

    pic = root.find("picture")
    if pic is None:
        raise ValueError(f"No <picture> in {sheet_xml_path}")

    pic_w = float(pic.get("width"))
    pic_h = float(pic.get("height"))
    if pic_w <= 0 or pic_h <= 0:
        raise ValueError(f"Bad picture size in {sheet_xml_path}: {pic_w}x{pic_h}")

    # Expected staff line spacing from Audiveris scale (very helpful for partial lines)
    expected_spacing = 0.0
    scale = root.find("scale")
    if scale is not None:
        inter = scale.find("interline")
        if inter is not None:
            try:
                expected_spacing = float(inter.get("main") or 0.0)
            except Exception:
                expected_spacing = 0.0

    guides = []
    page = root.find("page")
    if page is None:
        return pic_w, pic_h, guides

    for staff in page.findall(".//system//staff"):
        lines_node = staff.find("lines")
        if lines_node is None:
            continue

        # Prefer staff@left; else header/@start; else later fallback from line points
        x_left_attr = staff.get("left")
        x_anchor = float(x_left_attr) if x_left_attr is not None else None

        header = staff.find("header")
        if x_anchor is None and header is not None:
            hs = header.get("start")
            if hs is not None:
                try:
                    x_anchor = float(hs)
                except Exception:
                    pass

        line_nodes = lines_node.findall("line")
        if not line_nodes:
            continue

        # For each line, compute (y_at_left, min_x_of_that_line)
        yxs = []
        for ln in line_nodes:
            pts = ln.findall("point")
            if not pts:
                continue
            best = None  # (x, y)
            for p in pts:
                x = float(p.get("x"))
                y = float(p.get("y"))
                if best is None or x < best[0]:
                    best = (x, y)
            if best is None:
                continue
            yxs.append((best[1], best[0]))

        if len(yxs) < 3:
            continue

        # Sort by y and pick best 5 (or synthesize if only 3-4)
        yxs.sort(key=lambda t: t[0])

        if len(yxs) >= 5:
            chosen = _best_five_by_spacing(yxs, expected_spacing)
            chosen = sorted(chosen, key=lambda t: t[0])
            ys5 = [t[0] for t in chosen]
            xmins = [t[1] for t in chosen]
        else:
            ys_partial = [t[0] for t in yxs]
            ys5 = _synthesize_five_lines(ys_partial, expected_spacing)
            if ys5 is None:
                continue
            # xmins: use what we have (for fallback x computation)
            xmins = [t[1] for t in yxs]

        y_top = float(min(ys5))
        y_bot = float(max(ys5))
        if y_bot <= y_top or (y_bot - y_top) < 10:
            continue

        # X choice: anchor if present; else a stable percentile of per-line min-x
        if x_anchor is not None:
            x_left = x_anchor
        else:
            # Avoid extreme min/max outliers
            x_left = _pct(xmins, 0.30)

        guides.append((x_left, y_top, y_bot))

    return pic_w, pic_h, guides

def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    doc = fitz.open(input_pdf)

    with zipfile.ZipFile(omr_path, "r") as z:
        sheet_paths = _sorted_sheet_xml_paths(z)
        if not sheet_paths:
            raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

        # We expect one sheet per PDF page. If mismatch, we still do min() safely.
        for page_index in range(doc.page_count):
            page = doc[page_index]

            # Choose matching sheet; if fewer sheets than pages, reuse last.
            if page_index >= len(sheet_paths):
                # No matching sheet for this PDF page. Skip rather than reuse last sheet.
                continue
            sheet_i = page_index

            sheet_xml_path = sheet_paths[sheet_i]

            pic_w, pic_h, guides = _parse_sheet(z, sheet_xml_path)
            if not guides:
                continue

            # Map Audiveris pixel coords -> PDF coords for THIS page
            # (PyMuPDF page coords use page.rect; origin at top-left in MuPDF space). :contentReference[oaicite:1]{index=1}
            rect = page.rect
            scale_x = rect.width / pic_w
            scale_y = rect.height / pic_h

            for (x_px, y0_px, y1_px) in guides:
                x_pdf = x_px * scale_x
                y0_pdf = y0_px * scale_y
                y1_pdf = y1_px * scale_y
                page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=GUIDE_COLOR, width=GUIDE_WIDTH)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_guides_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
