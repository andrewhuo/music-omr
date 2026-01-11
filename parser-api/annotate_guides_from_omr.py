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


def _debug_dump_enabled() -> bool:
    v = os.getenv("DEBUG_GUIDES_DUMP_INTERS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


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
    """
    IMPORTANT: Audiveris sometimes nests inter elements; don't only look at direct children.
    """
    out: dict[int, ET.Element] = {}
    inters = page.find(".//sig/inters")
    if inters is None:
        return out
    for el in inters.iter():
        sid = el.get("id")
        if not sid:
            continue
        try:
            out[int(sid)] = el
        except Exception:
            continue
    return out


def _bounds_of(el: ET.Element) -> tuple[float, float, float, float] | None:
    b = el.find("bounds")
    if b is None:
        return None
    x = _safe_float(b.get("x"))
    y = _safe_float(b.get("y"))
    w = _safe_float(b.get("w"))
    h = _safe_float(b.get("h"))
    if x is None or y is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0:
        return None
    return (float(x), float(y), float(w), float(h))


def _tag_looks_like_clef(el: ET.Element) -> bool:
    t = (el.tag or "").lower()
    if t == "clef" or t.endswith("clef"):
        return True
    # Some Audiveris builds store as generic nodes with shape/type
    for k in ("shape", "type", "kind", "name", "family"):
        v = el.get(k)
        if v and "clef" in v.lower():
            return True
    return False


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

    b = _bounds_of(clef_el)
    if b is None:
        return None
    return b


def _clef_bounds_fallback(
    inters: ET.Element | None,
    staff_id: str,
    y_top: float,
    y_bot: float,
    header_start: float | None,
) -> tuple[float, float, float, float] | None:
    """
    Fallback when header doesn't link a clef.
    Robustly search *all descendants* under inters for clef-ish nodes with bounds.
    Prefer:
      - staff match when available
      - y overlap with staff
      - leftmost, with slight preference near header_start
    """
    if inters is None:
        return None

    y_pad = 0.20 * (y_bot - y_top)
    want_y0 = y_top - y_pad
    want_y1 = y_bot + y_pad

    best = None
    best_score = None

    for el in inters.iter():
        if not _tag_looks_like_clef(el):
            continue

        el_staff = el.get("staff")
        if el_staff and staff_id and el_staff != staff_id:
            continue

        b = _bounds_of(el)
        if b is None:
            continue

        x, y, w, h = b
        by0, by1 = y, y + h
        if by1 < want_y0 or by0 > want_y1:
            continue

        penalty = 0.0
        if header_start is not None:
            penalty = max(0.0, x - header_start) * 0.05
        score = x + penalty

        if best_score is None or score < best_score:
            best_score = score
            best = b

    return best


def _bounds_overlap_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _dump_inters_near_staff(
    sheet_xml_path: str,
    staff_id: str,
    inters: ET.Element | None,
    y_top: float,
    y_bot: float,
    x_center: float,
    x_window: float = 120.0,
) -> None:
    if not (_debug_enabled() and _debug_dump_enabled() and _debug_match(sheet_xml_path, staff_id)):
        return
    if inters is None:
        _dbg(sheet_xml_path, staff_id, "[DBG] DUMP_INTERS none (no <sig/inters>)")
        return

    y_pad = 0.25 * (y_bot - y_top)
    want_y0 = y_top - y_pad
    want_y1 = y_bot + y_pad

    rows = []
    for el in inters.iter():
        b = _bounds_of(el)
        if b is None:
            continue
        x, y, w, h = b
        x0, x1 = x, x + w
        y0, y1 = y, y + h

        if y1 < want_y0 or y0 > want_y1:
            continue
        if x1 < (x_center - x_window) or x0 > (x_center + x_window):
            continue

        shape = None
        for k in ("shape", "type", "kind", "name", "family"):
            if el.get(k):
                shape = f"{k}={el.get(k)}"
                break

        rows.append((x0, x1, el.tag, el.get("staff"), el.get("id"), shape, y0, y1))

    rows.sort(key=lambda r: r[0])
    _dbg(
        sheet_xml_path,
        staff_id,
        f"[DBG] DUMP_INTERS staff={staff_id} y=[{y_top:.1f},{y_bot:.1f}] x_center={x_center:.1f} found={len(rows)}",
    )
    for r in rows[:25]:
        x0, x1, tag, st, iid, shape, y0, y1 = r
        _dbg(
            sheet_xml_path,
            staff_id,
            f"  [DBG] inter tag={tag} staff={st} id={iid} {shape or ''} x=[{x0:.1f},{x1:.1f}] y=[{y0:.1f},{y1:.1f}]",
        )


def _looks_headerish_symbol(el: ET.Element) -> bool:
    tag = (el.tag or "").lower()
    if any(k in tag for k in ("clef", "keysig", "timesig", "brace", "bracket")):
        return True

    shape_blob = []
    for k in ("shape", "type", "kind", "name", "family"):
        v = el.get(k)
        if v:
            shape_blob.append(v.lower())
    s = " ".join(shape_blob)
    if not s:
        return False

    return any(k in s for k in ("clef", "key", "time", "brace", "bracket", "keysig", "timesig"))


def _has_headerish_symbol_near_header(
    inters: ET.Element | None,
    staff_id: str,
    y_top: float,
    y_bot: float,
    header_start: float | None,
    expected_spacing: float,
) -> bool:
    """
    When clef linkage is missing, only apply aggressive 'header guarding' if we can
    actually find header-ish symbols near the header_start region for this staff.
    """
    if inters is None or header_start is None:
        return False

    header_span = max(60.0, (4.0 * expected_spacing) if expected_spacing > 0 else 60.0)

    y_pad = 0.25 * (y_bot - y_top)
    want_y0 = y_top - y_pad
    want_y1 = y_bot + y_pad

    x0 = float(header_start) - header_span
    x1 = float(header_start) + header_span

    for el in inters.iter():
        b = _bounds_of(el)
        if b is None:
            continue

        bx, by, bw, bh = b
        bx0, bx1 = bx, bx + bw
        by0, by1 = by, by + bh

        if by1 < want_y0 or by0 > want_y1:
            continue
        if bx1 < x0 or bx0 > x1:
            continue

        el_staff = el.get("staff")
        if el_staff and staff_id and el_staff != staff_id:
            continue

        if _looks_headerish_symbol(el):
            return True

    return False


def _push_left_if_inside_any_symbol(
    inters: ET.Element | None,
    staff_id: str,
    y_top: float,
    y_bot: float,
    x_left: float,
    line_min_x: float | None,
    header_start: float | None,
    expected_spacing: float,
    sheet_xml_path: str,
) -> float:
    """
    If the guide (after PAD_LEFT_PX) lands inside ANY inter bounds overlapping this staff,
    push it left of that symbol.

    NOTE: scan inters.iter() (descendants), not just direct children.
    """
    if inters is None:
        return x_left

    guard = max(10.0, (0.35 * expected_spacing) if expected_spacing > 0 else 10.0)

    # Prefer header_start as the reference window anchor (safer than line_min_x).
    ref = None
    for v in (header_start, line_min_x, x_left):
        if v is not None:
            ref = float(v)
            break
    if ref is None:
        ref = float(x_left)

    header_span = 1.25 * (expected_spacing if expected_spacing > 0 else 60.0)

    for _ in range(4):
        x_postpad = max(0.0, float(x_left) - PAD_LEFT_PX)

        hits: list[tuple[float, float, str]] = []
        for el in inters.iter():
            b = _bounds_of(el)
            if b is None:
                continue
            x, y, w, h = b
            x0, x1 = x, x + w
            y0, y1 = y, y + h

            el_staff = el.get("staff")
            if el_staff and staff_id and el_staff != staff_id:
                continue

            if x0 > (ref + header_span):
                continue
            if not _bounds_overlap_1d(y0, y1, y_top, y_bot):
                continue

            if x0 <= x_postpad <= x1:
                hits.append((x0, x1, el.tag))

        if not hits:
            return x_left

        hits.sort(key=lambda t: t[0])
        hit_x0, hit_x1, hit_tag = hits[0]

        new_x_postpad = max(0.0, hit_x0 - guard)
        new_x_left = new_x_postpad + PAD_LEFT_PX

        _dbg(
            sheet_xml_path,
            staff_id,
            "[DBG] PUSH_LEFT "
            f"staff={staff_id} x_postpad={x_postpad:.2f} -> {new_x_postpad:.2f} "
            f"hit_tag={hit_tag} hit_x0={hit_x0:.2f} hit_x1={hit_x1:.2f} "
            f"guard={guard:.2f} ref={ref:.2f} header_span={header_span:.2f}",
        )

        if abs(new_x_left - x_left) < 0.01:
            return new_x_left
        x_left = new_x_left

    return x_left


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

            # barline tags tend to be direct children, but use iter() for safety
            for el in inters.iter():
                if (el.tag or "").lower() != "barline":
                    continue
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
                clef_b = None

                if ys5 is not None:
                    y_top = float(min(ys5))
                    y_bot = float(max(ys5))
                else:
                    span = barline_span_for_staff(staff_id, yxs[0][0] if len(yxs) >= 1 else None)
                    if span is None:
                        clef_b0 = _clef_bounds_from_header(inter_by_id, staff)
                        if clef_b0 is not None:
                            clef_y_hint = clef_b0[1] + 0.5 * clef_b0[3]
                            span = barline_span_for_staff(staff_id, clef_y_hint)
                    if span is None:
                        _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP no_y_span sheet={sheet_xml_path} staff={staff_id}")
                        continue
                    y_top, y_bot = span

                if y_bot <= y_top or (y_bot - y_top) < 10.0:
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] SKIP bad_y_span sheet={sheet_xml_path} staff={staff_id} y_top={y_top} y_bot={y_bot}",
                    )
                    continue

                # ---- Clef bounds (robust) ----
                clef_b = _clef_bounds_from_header(inter_by_id, staff)
                if clef_b is None:
                    clef_b = _clef_bounds_fallback(inters, staff_id, y_top, y_bot, header_start)

                # ---- X anchor ----
                x_left = _safe_float(staff.get("left"))
                line_min_x = float(min(all_line_xmins)) if all_line_xmins else None

                if x_left is None and line_min_x is not None:
                    x_left = line_min_x
                if x_left is None and header_start is not None:
                    x_left = float(header_start)
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

                # 1) Don’t drift right of staff line start
                if line_min_x is not None and expected_spacing > 0.0:
                    if x_left > (line_min_x + 0.60 * expected_spacing):
                        x_left = line_min_x

                # 2) Header clamp
                if header_start is not None:
                    base_guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(header_start) - base_guard
                    if x_left > max_x:
                        x_left = max_x

                    # BIG guard only when clef is missing *and* we actually detect header-ish symbols nearby
                    if clef_b is None:
                        if _has_headerish_symbol_near_header(
                            inters=inters,
                            staff_id=staff_id,
                            y_top=y_top,
                            y_bot=y_bot,
                            header_start=header_start,
                            expected_spacing=expected_spacing,
                        ):
                            big_guard = max(18.0, (1.6 * expected_spacing) if expected_spacing > 0 else 18.0)
                            max_x2 = float(header_start) - big_guard
                            if x_left > max_x2:
                                _dbg(
                                    sheet_xml_path,
                                    staff_id,
                                    f"[DBG] BIG_HEADER_GUARD staff={staff_id} header_start={header_start:.2f} "
                                    f"expected_spacing={expected_spacing:.2f} x={x_left:.2f} -> {max_x2:.2f}",
                                )
                                x_left = max_x2

                # 3) Clef clamp (when clef bounds exist)
                if clef_b is not None:
                    clef_x, _, _, _ = clef_b
                    guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(clef_x) - guard
                    if x_left > max_x:
                        x_left = max_x

                # 4) Generic collision clamp: scan ALL descendant inter bounds
                x_left = _push_left_if_inside_any_symbol(
                    inters=inters,
                    staff_id=staff_id,
                    y_top=y_top,
                    y_bot=y_bot,
                    x_left=float(x_left),
                    line_min_x=line_min_x,
                    header_start=header_start,
                    expected_spacing=expected_spacing,
                    sheet_xml_path=sheet_xml_path,
                )

                # Apply pad last
                x_postpad = max(0.0, float(x_left) - PAD_LEFT_PX)

                # Optional inter dump around where we ended up
                _dump_inters_near_staff(
                    sheet_xml_path=sheet_xml_path,
                    staff_id=staff_id,
                    inters=inters,
                    y_top=y_top,
                    y_bot=y_bot,
                    x_center=x_postpad,
                    x_window=140.0,
                )

                # Summary debug line (always useful)
                if _debug_enabled() and _debug_match(sheet_xml_path, staff_id):
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] sheet={sheet_xml_path} sys={system.get('id')} staff={staff_id} indented={system.get('indented')} "
                        f"lines={len(line_nodes)} header_start={header_start} line_min_x={line_min_x} "
                        f"expected_spacing={expected_spacing:.2f} "
                        f"clef_b={'present' if clef_b is not None else None} "
                        f"x_before={x_before:.2f} x_after={float(x_left):.2f} x_postpad={x_postpad:.2f} "
                        f"y_top={y_top:.1f} y_bot={y_bot:.1f}",
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
