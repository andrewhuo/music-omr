#!/usr/bin/env python3
import sys
import re
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF

GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0

# Small left shift so line sits just left of the staff start / clef
PAD_LEFT_PX = 6.0

_SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


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


def _best_five_by_spacing(yxs: list[tuple[float, float]], expected_spacing: float) -> list[tuple[float, float]]:
    if len(yxs) <= 5:
        return yxs
    best = None
    best_score = float("inf")
    for i in range(0, len(yxs) - 4):
        win = yxs[i:i + 5]
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
    """
    Build a mapping id -> inter element from <sig><inters>.
    Useful to lookup clef bounds and (optionally) barline medians.
    """
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


def _barline_y_span_for_staff(page: ET.Element, staff_id: str) -> tuple[float, float] | None:
    """
    If staff lines are missing/partial, barline medians often still provide
    perfect staff-height vertical span. We use any barline for that staff.
    """
    inters = page.find(".//sig/inters")
    if inters is None:
        return None

    # pick a "reasonable" barline: smallest bounds.x tends to be earliest in the system
    best = None  # (x, y0, y1)
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

        cand = (bx, y_top, y_bot)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return None
    return (best[1], best[2])


def _clef_left_x(page: ET.Element, inter_by_id: dict[int, ET.Element], staff: ET.Element) -> float | None:
    """
    Use staff/header/clef-id -> <clef> inter -> <bounds x=...>
    """
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
    return _safe_float(b.get("x"))

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

    expected_spacing = 0.0
    scale = root.find("scale")
    if scale is not None:
        inter = scale.find("interline")
        if inter is not None:
            expected_spacing = _safe_float(inter.get("main")) or 0.0

    guides: list[tuple[float, float, float]] = []

    page = root.find("page")
    if page is None:
        return pic_w, pic_h, guides

    inter_by_id = _index_inters(page)

    inters = page.find(".//sig/inters")

    def barline_span_for_staff(staff_id: str, y_hint: float | None) -> tuple[float, float] | None:
        """
        Choose a barline span for this staff, preferably one whose median span covers y_hint.
        Among candidates, pick smallest bounds.x (earliest in the system).
        """
        if inters is None or not staff_id:
            return None

        best_any = None          # (bx, y_top, y_bot)
        best_covering = None     # (bx, y_top, y_bot)

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

            cand = (bx, y_top, y_bot)

            if best_any is None or cand[0] < best_any[0]:
                best_any = cand

            if y_hint is not None:
                # small slack
                if (y_top - 2.0) <= y_hint <= (y_bot + 2.0):
                    if best_covering is None or cand[0] < best_covering[0]:
                        best_covering = cand

        chosen = best_covering if best_covering is not None else best_any
        if chosen is None:
            return None
        return (chosen[1], chosen[2])

    # Iterate systems in order (more stable than ".//system//staff" in some XMLs)
    for system in page.findall("system"):
        # staff nodes live under system/part/staff
        for staff in system.findall(".//staff"):
            staff_id = staff.get("id") or ""

            lines_node = staff.find("lines")
            line_nodes = [] if lines_node is None else lines_node.findall("line")

            # Collect (y_at_left, x_min_for_that_line) for each detected staff line
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
                # IMPORTANT: allow 2–4 lines (this is the main missing-staff fix)
                ys_partial = [t[0] for t in yxs]
                ys5 = _synthesize_five_lines(ys_partial, expected_spacing)

            if ys5 is not None:
                y_top = float(min(ys5))
                y_bot = float(max(ys5))
            else:
                # Fallback: barline median span (prefer one covering a y hint if we have any)
                y_hint = yxs[0][0] if len(yxs) >= 1 else None
                span = barline_span_for_staff(staff_id, y_hint)
                if span is None:
                    continue
                y_top, y_bot = span

            if y_bot <= y_top or (y_bot - y_top) < 10.0:
                continue

            # ---- X anchor ----
            # Priority:
            # 1) staff@left
            # 2) min x from staff line points
            # 3) clef bounds x
            # 4) header@start
            # 5) percentile fallback
            x_left = _safe_float(staff.get("left"))

            line_min_x = float(min(all_line_xmins)) if all_line_xmins else None

            if x_left is None and line_min_x is not None:
                x_left = line_min_x

            if x_left is None:
                clef_x = _clef_left_x(page, inter_by_id, staff)
                if clef_x is not None:
                    x_left = float(clef_x)

            if x_left is None:
                header = staff.find("header")
                if header is not None:
                    hs = _safe_float(header.get("start"))
                    if hs is not None:
                        x_left = float(hs)

            if x_left is None and all_line_xmins:
                x_left = float(_pct(all_line_xmins, 0.05))

            if x_left is None:
                continue

            # Clamp: if chosen x is too far right compared to the staff-line edge,
            # force it back to the staff edge (fixes "line inside clef" safely).
            if line_min_x is not None and expected_spacing > 0.0:
                if x_left > (line_min_x + 0.60 * expected_spacing):
                    x_left = line_min_x

            x_left = max(0.0, float(x_left) - PAD_LEFT_PX)

            guides.append((x_left, y_top, y_bot))

    return pic_w, pic_h, guides


def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    doc = fitz.open(input_pdf)

    with zipfile.ZipFile(omr_path, "r") as z:
        sheet_paths = _sorted_sheet_xml_paths(z)
        if not sheet_paths:
            raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

        for page_index in range(doc.page_count):
            if page_index >= len(sheet_paths):
                # safer: skip pages we don't have a matching sheet for
                continue

            page = doc[page_index]
            sheet_xml_path = sheet_paths[page_index]

            pic_w, pic_h, guides = _parse_sheet(z, sheet_xml_path)
            if not guides:
                continue

            rect = page.rect
            scale_x = rect.width / pic_w
            scale_y = rect.height / pic_h

            for (x_px, y0_px, y1_px) in guides:
                x_pdf = x_px * scale_x
                y0_pdf = y0_px * scale_y
                y1_pdf = y1_px * scale_y
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
