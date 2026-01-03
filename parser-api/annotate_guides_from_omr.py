#!/usr/bin/env python3
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
import numpy as np
import cv2


GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0

# Small left shift so line sits just left of the staff start / clef
PAD_LEFT_PX = 6.0

_SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


def _debug_enabled() -> bool:
    v = os.getenv("DEBUG_GUIDES", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _debug_match(sheet_xml_path: str, staff_id: str) -> bool:
    # Optional filters to avoid log spam
    want_sheet = os.getenv("DEBUG_GUIDES_SHEET", "").strip()
    want_staff = os.getenv("DEBUG_GUIDES_STAFF_ID", "").strip()
    if want_sheet and want_sheet not in sheet_xml_path:
        return False
    if want_staff and want_staff != str(staff_id):
        return False
    return True


def _dbg(sheet_xml_path: str, staff_id: str, msg: str) -> None:
    if _debug_enabled() and _debug_match(sheet_xml_path, staff_id):
        print(msg, flush=True)


def _sorted_sheet_xml_paths(z: zipfile.ZipFile) -> list[str]:
    found = []
    for name in z.namelist():
        m = _SHEET_XML_RE.match(name)
        if m:
            found.append((int(m.group(1)), name))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _pct(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    i = int(round(p * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, i))]


def _best_five_by_spacing(
    yxs: list[tuple[float, float]], expected_spacing: float
) -> list[tuple[float, float]]:
    if len(yxs) <= 5:
        return yxs
    best = None
    best_score = float("inf")
    for i in range(0, len(yxs) - 4):
        win = yxs[i : i + 5]
        ys = [t[0] for t in win]
        diffs = [ys[j + 1] - ys[j] for j in range(4)]
        med = sorted(diffs)[2]
        var = sum(abs(d - med) for d in diffs)
        if expected_spacing > 0:
            var += 0.35 * sum(abs(d - expected_spacing) for d in diffs)
        if var < best_score:
            best_score = var
            best = win
    return best if best is not None else yxs[:5]


def _synthesize_five_lines(ys: list[float], expected_spacing: float) -> list[float] | None:
    ys = sorted(ys)
    if expected_spacing <= 0:
        return None

    tol = max(2.0, expected_spacing * 0.35)

    best = None
    best_hits = -1
    best_err = float("inf")

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
                    err += 2 * tol + d

            if hits > best_hits or (hits == best_hits and err < best_err):
                best_hits = hits
                best_err = err
                best = pred

    if best is None or best_hits < 2:
        return None

    snapped = []
    for y_pred in best:
        nearest_obs = min(ys, key=lambda t: abs(t - y_pred))
        if abs(nearest_obs - y_pred) <= tol:
            snapped.append(nearest_obs)
        else:
            snapped.append(y_pred)

    snapped = sorted(snapped)
    if snapped[-1] - snapped[0] < 3.2 * expected_spacing:
        return None
    return snapped


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _index_inters(page: ET.Element) -> dict[int, ET.Element]:
    out: dict[int, ET.Element] = {}
    inters = page.find(".//sig/inters")
    if inters is None:
        return out
    for el in list(inters):
        sid = el.get("id")
        if not sid:
            continue
        try:
            out[int(sid)] = el
        except Exception:
            continue
    return out


def _clef_bounds_from_header(
    inter_by_id: dict[int, ET.Element], staff: ET.Element
) -> tuple[float, float, float, float] | None:
    header = staff.find("header")
    if header is None:
        return None

    clef_id_txt = header.findtext("clef")
    if not clef_id_txt:
        return None

    try:
        clef_id = int(clef_id_txt.strip())
    except Exception:
        return None

    clef_el = inter_by_id.get(clef_id)
    if clef_el is None:
        return None

    b = clef_el.find("bounds")
    if b is None:
        return None

    x = _safe_float(b.get("x"))
    y = _safe_float(b.get("y"))
    w = _safe_float(b.get("w"))
    h = _safe_float(b.get("h"))
    if x is None or y is None or w is None or h is None:
        return None

    return (float(x), float(y), float(w), float(h))


def _clef_bounds_fallback(
    inters: ET.Element | None,
    staff_id: str,
    y_top: float,
    y_bot: float,
    header_start: float | None,
) -> tuple[float, float, float, float] | None:
    """
    If header doesn't link the clef (or staff attribute mismatches),
    try to find a clef inter by:
      1) matching clef@staff == staff_id (leftmost)
      2) otherwise, any clef whose bounds overlap the staff y-span (leftmost,
         optionally preferring ones near header_start)
    """
    if inters is None:
        return None

    best = None

    # 1) clef elements that explicitly match staff id
    if staff_id:
        for el in inters.findall("clef"):
            if el.get("staff") != staff_id:
                continue
            b = el.find("bounds")
            if b is None:
                continue
            x = _safe_float(b.get("x"))
            y = _safe_float(b.get("y"))
            w = _safe_float(b.get("w"))
            h = _safe_float(b.get("h"))
            if x is None or y is None or w is None or h is None:
                continue
            cand = (float(x), float(y), float(w), float(h))
            if best is None or cand[0] < best[0]:
                best = cand

    if best is not None:
        return best

    # 2) clef elements overlapping staff y-span
    #    Prefer ones near header_start if available.
    y_pad = 0.20 * (y_bot - y_top)
    want_y0 = y_top - y_pad
    want_y1 = y_bot + y_pad

    best_score = None
    for el in inters.findall("clef"):
        b = el.find("bounds")
        if b is None:
            continue
        x = _safe_float(b.get("x"))
        y = _safe_float(b.get("y"))
        w = _safe_float(b.get("w"))
        h = _safe_float(b.get("h"))
        if x is None or y is None or w is None or h is None:
            continue

        x, y, w, h = float(x), float(y), float(w), float(h)
        if h <= 0 or w <= 0:
            continue

        by0 = y
        by1 = y + h
        # overlap check
        if by1 < want_y0 or by0 > want_y1:
            continue

        # score: leftmost wins; add small penalty if far to the right of header_start
        penalty = 0.0
        if header_start is not None:
            # if clef is way to the right of header_start, it's probably not the header clef
            penalty = max(0.0, x - header_start) * 0.05
        score = x + penalty

        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def _parse_sheet(z: zipfile.ZipFile, sheet_xml_path: str):
    """
    Returns:
      pic_w, pic_h: Audiveris picture size
      guides_px: list of (x_left_px, y_top_px, y_bot_px) in picture pixels
      staff_total: how many <staff> nodes exist (target guide count)
    """
    data = z.read(sheet_xml_path)
    root = ET.fromstring(data)

    pic = root.find("picture")
    if pic is None:
        raise ValueError(f"No <picture> in {sheet_xml_path}")

    pic_w = float(pic.get("width"))
    pic_h = float(pic.get("height"))
    if pic_w <= 0 or pic_h <= 0:
        raise ValueError(f"Bad picture size in {sheet_xml_path}: {pic_w}x{pic_h}")

    expected_spacing = 0.0
    scale = root.find("scale")
    if scale is not None:
        inter = scale.find("interline")
        if inter is not None:
            expected_spacing = _safe_float(inter.get("main")) or 0.0

    guides_px: list[tuple[float, float, float]] = []

    pages = root.findall("page")
    if not pages:
        return pic_w, pic_h, guides_px, 0

    staff_total = 0

    for page in pages:
        inter_by_id = _index_inters(page)
        inters = page.find(".//sig/inters")

        def barline_span_for_staff(staff_id: str, y_hint: float | None) -> tuple[float, float] | None:
            if inters is None or not staff_id:
                return None

            best_covering = None   # (bx, y_top, y_bot)
            best_closest = None    # (dist, bx, y_top, y_bot)
            best_any = None        # (bx, y_top, y_bot)

            for el in inters.findall("barline"):
                if el.get("staff") != staff_id:
                    continue

                b = el.find("bounds")
                med = el.find("median")
                if b is None or med is None:
                    continue

                bx = _safe_float(b.get("x"))
                if bx is None:
                    continue

                p1 = med.find("p1")
                p2 = med.find("p2")
                if p1 is None or p2 is None:
                    continue

                y1 = _safe_float(p1.get("y"))
                y2 = _safe_float(p2.get("y"))
                if y1 is None or y2 is None:
                    continue

                y_top = float(min(y1, y2))
                y_bot = float(max(y1, y2))
                if y_bot <= y_top:
                    continue

                cand_any = (float(bx), y_top, y_bot)
                if best_any is None or cand_any[0] < best_any[0]:
                    best_any = cand_any

                if y_hint is not None:
                    if (y_top - 2.0) <= y_hint <= (y_bot + 2.0):
                        if best_covering is None or cand_any[0] < best_covering[0]:
                            best_covering = cand_any

                    mid = 0.5 * (y_top + y_bot)
                    dist = abs(mid - y_hint)
                    cand_closest = (dist, float(bx), y_top, y_bot)
                    if best_closest is None or cand_closest < best_closest:
                        best_closest = cand_closest

            if best_covering is not None:
                return (best_covering[1], best_covering[2])
            if y_hint is not None and best_closest is not None:
                return (best_closest[2], best_closest[3])
            if best_any is not None:
                return (best_any[1], best_any[2])
            return None

        systems = page.findall(".//system")
        if not systems:
            systems = page.findall("system")

        for system in systems:
            for staff in system.findall(".//staff"):
                staff_total += 1
                staff_id = staff.get("id") or ""

                header = staff.find("header")
                header_start = _safe_float(header.get("start")) if header is not None else None

                lines_node = staff.find("lines")
                line_nodes = [] if lines_node is None else lines_node.findall("line")

                yxs: list[tuple[float, float]] = []
                all_line_xmins: list[float] = []

                for ln in line_nodes:
                    pts = ln.findall("point")
                    if not pts:
                        continue

                    min_x = None
                    y_at_min_x = None

                    for p in pts:
                        x = _safe_float(p.get("x"))
                        y = _safe_float(p.get("y"))
                        if x is None or y is None:
                            continue
                        if min_x is None or x < min_x:
                            min_x = x
                            y_at_min_x = y

                    if min_x is None or y_at_min_x is None:
                        continue

                    yxs.append((float(y_at_min_x), float(min_x)))
                    all_line_xmins.append(float(min_x))

                yxs.sort(key=lambda t: t[0])

                # ---- Y span ----
                ys5: list[float] | None = None
                if len(yxs) >= 5:
                    chosen = _best_five_by_spacing(yxs, expected_spacing)
                    chosen = sorted(chosen, key=lambda t: t[0])
                    ys5 = [t[0] for t in chosen]
                elif len(yxs) >= 2:
                    ys_partial = [t[0] for t in yxs]
                    ys5 = _synthesize_five_lines(ys_partial, expected_spacing)

                clef_y_hint = None
                clef_b = None  # we’ll compute after y span, so we can do y-overlap fallback

                if ys5 is not None:
                    y_top = float(min(ys5))
                    y_bot = float(max(ys5))
                else:
                    span = barline_span_for_staff(staff_id, yxs[0][0] if len(yxs) >= 1 else None)
                    if span is None:
                        # try clef y-hint if available from header-linked clef
                        clef_b0 = _clef_bounds_from_header(inter_by_id, staff)
                        if clef_b0 is not None:
                            clef_y_hint = clef_b0[1] + 0.5 * clef_b0[3]
                            span = barline_span_for_staff(staff_id, clef_y_hint)
                    if span is None:
                        _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP no_y_span sheet={sheet_xml_path} staff={staff_id}")
                        continue
                    y_top, y_bot = span

                if y_bot <= y_top or (y_bot - y_top) < 10.0:
                    _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP bad_y_span sheet={sheet_xml_path} staff={staff_id} y_top={y_top} y_bot={y_bot}")
                    continue

                # ---- Clef bounds (after y span so we can do y-overlap fallback) ----
                clef_b = _clef_bounds_from_header(inter_by_id, staff)
                if clef_b is None:
                    clef_b = _clef_bounds_fallback(inters, staff_id, y_top, y_bot, header_start)

                if _debug_enabled() and _debug_match(sheet_xml_path, staff_id):
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] sheet={sheet_xml_path} sys={system.get('id')} staff={staff_id} indented={system.get('indented')} "
                        f"lines={len(line_nodes)} header_start={header_start} line_min_x={(min(all_line_xmins) if all_line_xmins else None)} "
                        f"clef_b={'present' if clef_b is not None else None} y_top={y_top:.1f} y_bot={y_bot:.1f}",
                    )

                # ---- X anchor ----
                x_left = _safe_float(staff.get("left"))
                line_min_x = float(min(all_line_xmins)) if all_line_xmins else None

                # Prefer real staff geometry if present
                if x_left is None and line_min_x is not None:
                    x_left = line_min_x

                # Next best: header_start is usually the staff left edge
                if x_left is None and header_start is not None:
                    x_left = float(header_start)

                # Last resort: left of clef by (clef width + a bit)
                if x_left is None and clef_b is not None:
                    clef_x, _, clef_w, _ = clef_b
                    extra = (expected_spacing if expected_spacing > 0 else 20.0)
                    x_left = float(clef_x) - float(clef_w) - float(extra)

                if x_left is None and all_line_xmins:
                    x_left = float(_pct(all_line_xmins, 0.05))

                if x_left is None:
                    _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP no_x sheet={sheet_xml_path} staff={staff_id}")
                    continue

                x_before = float(x_left)

                # 1) Don’t drift right of staff line start (if available)
                if line_min_x is not None and expected_spacing > 0.0:
                    if x_left > (line_min_x + 0.60 * expected_spacing):
                        x_left = line_min_x

                # 2) Never put guide inside header area (clef/key/time)
                if header_start is not None:
                    guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(header_start) - guard
                    if x_left > max_x:
                        x_left = max_x

                # 3) Never put guide inside clef box (strongest clamp)
                if clef_b is not None:
                    clef_x, _, _, _ = clef_b
                    guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(clef_x) - guard
                    if x_left > max_x:
                        x_left = max_x

                # Apply pad last
                x_postpad = max(0.0, float(x_left) - PAD_LEFT_PX)

                # Debug: detect if final guide still lands inside clef (should be rare now)
                if _debug_enabled() and _debug_match(sheet_xml_path, staff_id) and clef_b is not None:
                    clef_x, _, clef_w, _ = clef_b
                    if clef_x <= x_postpad <= (clef_x + clef_w):
                        _dbg(
                            sheet_xml_path,
                            staff_id,
                            "[DBG] GUIDE_IN_CLEF "
                            f"sheet={sheet_xml_path} staff={staff_id} "
                            f"x_before={x_before:.2f} x_after={float(x_left):.2f} x_postpad={x_postpad:.2f} "
                            f"clef_x={clef_x:.2f} clef_w={clef_w:.2f} staff_left={staff.get('left')} header_start={header_start} line_min_x={line_min_x}",
                        )

                guides_px.append((x_postpad, y_top, y_bot))

    return pic_w, pic_h, guides_px, staff_total


def _render_page_gray(page: fitz.Page, zoom: float = 2.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def _fallback_missing_staff_guides(
    page: fitz.Page,
    existing_guides_pdf: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Detect staff lines directly from the rendered page image.
    Return ONLY guides that don't overlap existing ones (by y-span).
    """
    gray = _render_page_gray(page, zoom=2.0)
    h, w = gray.shape[:2]

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    kernel_w = max(30, w // 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < 0.20 * w:
            continue
        if hh > 6:
            continue
        y_mid = y + 0.5 * hh
        lines.append((y_mid, x, ww))

    if len(lines) < 8:
        return []

    lines.sort(key=lambda t: t[0])
    ys = [t[0] for t in lines]

    gaps = []
    for i in range(len(ys) - 1):
        d = ys[i + 1] - ys[i]
        if 3 <= d <= 40:
            gaps.append(d)
    if not gaps:
        return []

    gaps.sort()
    gap = gaps[len(gaps) // 2]
    tol = max(2.0, 0.35 * gap)

    staff_candidates = []
    for i in range(0, len(lines) - 4):
        win = lines[i : i + 5]
        wy = [t[0] for t in win]
        diffs = [wy[j + 1] - wy[j] for j in range(4)]
        if all(abs(d - gap) <= tol for d in diffs):
            x_left = min(t[1] for t in win)
            y_top = wy[0]
            y_bot = wy[4]
            staff_candidates.append((x_left, y_top, y_bot))

    if not staff_candidates:
        return []

    staff_candidates.sort(key=lambda t: (t[1], t[0]))
    staves = []
    for x_left, y_top, y_bot in staff_candidates:
        cy = 0.5 * (y_top + y_bot)
        if staves and abs(cy - 0.5 * (staves[-1][1] + staves[-1][2])) < 0.6 * (y_bot - y_top):
            continue
        staves.append((x_left, y_top, y_bot))

    rect = page.rect
    pad_pdf = (PAD_LEFT_PX / float(w)) * rect.width

    extras = []
    for x_left_i, y_top_i, y_bot_i in staves:
        x_pdf = (x_left_i / float(w)) * rect.width - pad_pdf
        y0_pdf = (y_top_i / float(h)) * rect.height
        y1_pdf = (y_bot_i / float(h)) * rect.height
        if y1_pdf <= y0_pdf:
            continue

        cy = 0.5 * (y0_pdf + y1_pdf)
        h_staff = (y1_pdf - y0_pdf)

        overlaps = False
        for (_, ey0, ey1) in existing_guides_pdf:
            if (ey0 - 0.25 * h_staff) <= cy <= (ey1 + 0.25 * h_staff):
                overlaps = True
                break

        if not overlaps:
            extras.append((max(0.0, x_pdf), y0_pdf, y1_pdf))

    return extras


def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    doc = fitz.open(input_pdf)

    with zipfile.ZipFile(omr_path, "r") as z:
        sheet_paths = _sorted_sheet_xml_paths(z)
        if not sheet_paths:
            raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

        for page_index in range(doc.page_count):
            if page_index >= len(sheet_paths):
                continue

            page = doc[page_index]
            sheet_xml_path = sheet_paths[page_index]

            pic_w, pic_h, guides_px, staff_total = _parse_sheet(z, sheet_xml_path)
            if pic_w <= 0 or pic_h <= 0:
                continue

            rect = page.rect
            scale_x = rect.width / pic_w
            scale_y = rect.height / pic_h

            guides_pdf = []
            for (x_px, y0_px, y1_px) in guides_px:
                x_pdf = x_px * scale_x
                y0_pdf = y0_px * scale_y
                y1_pdf = y1_px * scale_y
                guides_pdf.append((x_pdf, y0_pdf, y1_pdf))
                page.draw_line(
                    (x_pdf, y0_pdf),
                    (x_pdf, y1_pdf),
                    color=GUIDE_COLOR,
                    width=GUIDE_WIDTH,
                )

            if staff_total > 0 and len(guides_pdf) < staff_total:
                extras = _fallback_missing_staff_guides(page, guides_pdf)
                if _debug_enabled():
                    print(
                        f"[DBG] page={page_index+1} sheet={sheet_xml_path} "
                        f"staff_total={staff_total} omr_guides={len(guides_pdf)} fallback_extras={len(extras)}",
                        flush=True,
                    )
                for (x_pdf, y0_pdf, y1_pdf) in extras:
                    page.draw_line(
                        (x_pdf, y0_pdf),
                        (x_pdf, y1_pdf),
                        color=GUIDE_COLOR,
                        width=GUIDE_WIDTH,
                    )

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_guides_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
