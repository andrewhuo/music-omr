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

# Measure number rendering
MEASURE_TEXT_COLOR = (0, 0, 0)  # black
MEASURE_MIN_FONTSIZE = 7.0
MEASURE_MAX_FONTSIZE = 12.0

# Outlier forcing
OUTLIER_MZ_THRESHOLD = 3.5  # modified z-score threshold
OUTLIER_ABS_FLOOR_PX = 10.0  # minimum MAD floor to avoid divide-by-near-zero

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


def _meas_debug_enabled() -> bool:
    v = os.getenv("DEBUG_MEASURES", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _meas_debug_match(sheet_xml_path: str, system_id: str) -> bool:
    want_sheet = os.getenv("DEBUG_MEASURES_SHEET", "").strip()
    want_sys = os.getenv("DEBUG_MEASURES_SYS_ID", "").strip()
    if want_sheet and want_sheet not in sheet_xml_path:
        return False
    if want_sys and want_sys != str(system_id):
        return False
    return True


def _meas_dbg(sheet_xml_path: str, system_id: str, msg: str) -> None:
    if _meas_debug_enabled() and _meas_debug_match(sheet_xml_path, system_id):
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


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        raise ValueError("median of empty")
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def _mad(xs: list[float], med: float) -> float:
    return _median([abs(x - med) for x in xs])


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
    if inters is None:
        return x_left

    guard = max(10.0, (0.35 * expected_spacing) if expected_spacing > 0 else 10.0)

    ref = None
    for v in (line_min_x, header_start, x_left):
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

            # Only consider staff-owned symbols as collisions.
            el_staff = el.get("staff")
            if not el_staff or el_staff != staff_id:
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


def _barline_xs_for_system(
    inters: ET.Element | None,
    staff_ids_in_system: set[str],
    y_top: float,
    y_bot: float,
) -> list[float]:
    """
    Collect barlines for the whole system, not just one staff.

    Audiveris barline inters can be:
      - attached to a particular staff (staff="..."), but not necessarily the top staff
      - or sometimes not reliably staff-attached for system-wide constructs
    So we:
      - accept barlines with staff in this system
      - also accept barlines with no staff attribute
      - filter by vertical overlap with the system span
    """
    if inters is None:
        return []

    xs: list[float] = []
    for el in inters.iter():
        if (el.tag or "").lower() != "barline":
            continue

        el_staff = el.get("staff")
        if el_staff:
            if el_staff not in staff_ids_in_system:
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

        by0 = float(min(y1, y2))
        by1 = float(max(y1, y2))
        if not _bounds_overlap_1d(by0, by1, y_top, y_bot):
            continue

        xs.append(float(bx))

    xs.sort()
    return xs


def _dedupe_sorted_xs(xs: list[float], eps: float) -> list[float]:
    if not xs:
        return []
    out = [xs[0]]
    for x in xs[1:]:
        if abs(x - out[-1]) <= eps:
            out[-1] = 0.5 * (out[-1] + x)
        else:
            out.append(x)
    return out


def _parse_sheet(z: zipfile.ZipFile, sheet_xml_path: str, start_measure: int):
    """
    Returns:
      pic_w, pic_h: Audiveris picture size
      guides_px: list of (x_left_px, y_top_px, y_bot_px) in picture pixels
      staff_total: how many <staff> nodes exist (target guide count)
      measures_px: list of (x_px, y_px, label_text) in picture pixels
      next_measure: next measure number after this sheet
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

    pages = root.findall("page")
    if not pages:
        return pic_w, pic_h, [], 0, [], start_measure

    staff_total = 0

    staff_recs: list[dict] = []
    system_recs: dict[str, dict] = {}

    for page in pages:
        inter_by_id = _index_inters(page)
        inters = page.find(".//sig/inters")

        systems = page.findall(".//system")
        if not systems:
            systems = page.findall("system")

        for system in systems:
            sys_id = system.get("id") or ""
            sys_indented = system.get("indented")

            if sys_id not in system_recs:
                system_recs[sys_id] = {
                    "sys_id": sys_id,
                    "indented": sys_indented,
                    "staff_spans": [],  # list of (y_top, y_bot, staff_id)
                }

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

                ys5: list[float] | None = None
                if len(yxs) >= 5:
                    chosen = _best_five_by_spacing(yxs, expected_spacing)
                    chosen = sorted(chosen, key=lambda t: t[0])
                    ys5 = [t[0] for t in chosen]
                elif len(yxs) >= 2:
                    ys_partial = [t[0] for t in yxs]
                    ys5 = _synthesize_five_lines(ys_partial, expected_spacing)

                if ys5 is not None:
                    y_top = float(min(ys5))
                    y_bot = float(max(ys5))
                else:
                    _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP no_y_span sheet={sheet_xml_path} staff={staff_id}")
                    continue

                if y_bot <= y_top or (y_bot - y_top) < 10.0:
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] SKIP bad_y_span sheet={sheet_xml_path} staff={staff_id} y_top={y_top} y_bot={y_bot}",
                    )
                    continue

                clef_b = _clef_bounds_from_header(inter_by_id, staff)
                if clef_b is None:
                    clef_b = _clef_bounds_fallback(inters, staff_id, y_top, y_bot, header_start)

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

                if line_min_x is not None and expected_spacing > 0.0:
                    if x_left > (line_min_x + 0.60 * expected_spacing):
                        x_left = line_min_x

                if header_start is not None:
                    base_guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(header_start) - base_guard
                    if x_left > max_x:
                        x_left = max_x

                    if clef_b is None:
                        big_guard = max(28.0, (2.5 * expected_spacing) if expected_spacing > 0 else 28.0)
                        max_x2 = float(header_start) - big_guard
                        if x_left > max_x2:
                            _dbg(
                                sheet_xml_path,
                                staff_id,
                                f"[DBG] BIG_HEADER_GUARD staff={staff_id} header_start={header_start:.2f} "
                                f"expected_spacing={expected_spacing:.2f} x={x_left:.2f} -> {max_x2:.2f}",
                            )
                            x_left = max_x2

                if clef_b is not None:
                    clef_x, _, _, _ = clef_b
                    guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(clef_x) - guard
                    if x_left > max_x:
                        x_left = max_x

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

                x_postpad = max(0.0, float(x_left) - PAD_LEFT_PX)

                _dump_inters_near_staff(
                    sheet_xml_path=sheet_xml_path,
                    staff_id=staff_id,
                    inters=inters,
                    y_top=y_top,
                    y_bot=y_bot,
                    x_center=x_postpad,
                    x_window=140.0,
                )

                if _debug_enabled() and _debug_match(sheet_xml_path, staff_id):
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] sheet={sheet_xml_path} sys={sys_id} staff={staff_id} indented={system.get('indented')} "
                        f"lines={len(line_nodes)} header_start={header_start} line_min_x={line_min_x} "
                        f"expected_spacing={expected_spacing:.2f} "
                        f"clef_b={'present' if clef_b is not None else None} "
                        f"x_before={x_before:.2f} x_after={float(x_left):.2f} x_postpad={x_postpad:.2f} "
                        f"y_top={y_top:.1f} y_bot={y_bot:.1f}",
                    )

                staff_recs.append(
                    {
                        "sys_id": sys_id,
                        "staff_id": staff_id,
                        "indented": system.get("indented"),
                        "x_postpad": float(x_postpad),
                        "y_top": float(y_top),
                        "y_bot": float(y_bot),
                    }
                )
                system_recs[sys_id]["staff_spans"].append((float(y_top), float(y_bot), staff_id))

    non_indented_xs = [
        r["x_postpad"] for r in staff_recs if not (r["indented"] not in (None, "", "false", "False", "0"))
    ]

    forced_avg = None
    outlier_ids: set[tuple[str, str]] = set()

    if len(non_indented_xs) >= 6:
        med = _median(non_indented_xs)
        mad = _mad(non_indented_xs, med)
        mad = max(mad, OUTLIER_ABS_FLOOR_PX)

        def mz(x: float) -> float:
            return abs(0.6745 * (x - med) / mad)

        inliers = [x for x in non_indented_xs if mz(x) <= OUTLIER_MZ_THRESHOLD]
        if len(inliers) >= 4:
            forced_avg = sum(inliers) / float(len(inliers))

            for r in staff_recs:
                if r["indented"] not in (None, "", "false", "False", "0"):
                    continue
                score = mz(r["x_postpad"])
                if score > OUTLIER_MZ_THRESHOLD:
                    outlier_ids.add((r["sys_id"], r["staff_id"]))
                    old = r["x_postpad"]
                    r["x_postpad"] = float(forced_avg)
                    if _debug_enabled() and _debug_match(sheet_xml_path, r["staff_id"]):
                        _dbg(
                            sheet_xml_path,
                            r["staff_id"],
                            f"[DBG] FORCE_OUTLIER staff={r['staff_id']} sys={r['sys_id']} "
                            f"x={old:.2f} -> {forced_avg:.2f} mz={score:.2f} med={med:.2f} mad={mad:.2f} n_inliers={len(inliers)}",
                        )

            if _debug_enabled():
                print(
                    f"[DBG] OUTLIER_SUMMARY sheet={sheet_xml_path} forced_avg={forced_avg:.2f} "
                    f"non_indented={len(non_indented_xs)} outliers_forced={len(outlier_ids)}",
                    flush=True,
                )

    guides_px: list[tuple[float, float, float]] = [(r["x_postpad"], r["y_top"], r["y_bot"]) for r in staff_recs]

    # ---------------------------
    # Measure numbers: ONE per system start; advance by barline count in system
    # ---------------------------
    measures_px: list[tuple[float, float, str]] = []
    measure_no = int(start_measure)

    eps = max(6.0, 0.45 * expected_spacing) if expected_spacing > 0 else 10.0
    v_off = max(10.0, 0.90 * expected_spacing) if expected_spacing > 0 else 14.0

    staff_x_by_sys_staff: dict[tuple[str, str], float] = {
        (r["sys_id"], r["staff_id"]): r["x_postpad"] for r in staff_recs
    }

    for page in pages:
        inters = page.find(".//sig/inters")
        systems = page.findall(".//system")
        if not systems:
            systems = page.findall("system")

        for system in systems:
            sys_id = system.get("id") or ""
            spans = system_recs.get(sys_id, {}).get("staff_spans", [])
            if not spans:
                continue

            spans_sorted = sorted(spans, key=lambda t: t[0])
            top_y_top, top_y_bot, top_staff_id = spans_sorted[0]
            bottom_y_bot = max(t[1] for t in spans_sorted)
            sys_y_top = min(t[0] for t in spans_sorted)
            sys_y_bot = float(bottom_y_bot)

            sys_start_x = staff_x_by_sys_staff.get((sys_id, top_staff_id))
            if sys_start_x is None:
                continue

            staff_ids_in_system = {sid for (_, _, sid) in spans_sorted if sid}

            bar_xs = _barline_xs_for_system(
                inters=inters,
                staff_ids_in_system=staff_ids_in_system,
                y_top=float(sys_y_top),
                y_bot=float(sys_y_bot),
            )
            bar_xs = _dedupe_sorted_xs(bar_xs, eps=eps)

            bar_xs = [x for x in bar_xs if x > (sys_start_x + eps)]

            _meas_dbg(
                sheet_xml_path,
                sys_id,
                f"[DBG] MEAS_SYS sheet={sheet_xml_path} sys={sys_id} "
                f"top_staff={top_staff_id} start_x={sys_start_x:.2f} "
                f"staffs={sorted(list(staff_ids_in_system))[:12]} "
                f"sys_y=[{sys_y_top:.1f},{sys_y_bot:.1f}] "
                f"bar_count={len(bar_xs)} xs={[round(x,1) for x in bar_xs[:20]]}",
            )

            # Draw ONE label at system start (above guideline)
            x_text = float(sys_start_x) + (max(6.0, 0.30 * expected_spacing) if expected_spacing > 0 else 8.0)
            y_text = max(0.0, float(top_y_top) - v_off)
            measures_px.append((float(x_text), float(y_text), str(measure_no)))

            _meas_dbg(
                sheet_xml_path,
                sys_id,
                f"[DBG] MEAS_DRAW_SYS_START sheet={sheet_xml_path} sys={sys_id} "
                f"measure={measure_no} x={x_text:.2f} y={y_text:.2f}",
            )

            # Advance by number of barlines within this system.
            measure_no += int(len(bar_xs))

    return pic_w, pic_h, guides_px, staff_total, measures_px, measure_no


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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    doc = fitz.open(input_pdf)

    measure_no = 1

    with zipfile.ZipFile(omr_path, "r") as z:
        sheet_paths = _sorted_sheet_xml_paths(z)
        if not sheet_paths:
            raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

        for page_index in range(doc.page_count):
            if page_index >= len(sheet_paths):
                continue

            page = doc[page_index]
            sheet_xml_path = sheet_paths[page_index]

            pic_w, pic_h, guides_px, staff_total, measures_px, measure_no = _parse_sheet(
                z, sheet_xml_path, start_measure=measure_no
            )
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

            # Draw measure numbers (already filtered to one per system start)
            if measures_px:
                interline_pdf = None
                if len(guides_pdf) >= 1:
                    staff_h = abs(guides_pdf[0][2] - guides_pdf[0][1])
                    interline_pdf = staff_h / 4.0 if staff_h > 0 else None

                if interline_pdf is None:
                    interline_pdf = 16.0

                fontsize = _clamp(0.55 * interline_pdf, MEASURE_MIN_FONTSIZE, MEASURE_MAX_FONTSIZE)

                for (x_px, y_px, label) in measures_px:
                    x_pdf = x_px * scale_x
                    y_pdf = y_px * scale_y
                    page.insert_text(
                        (x_pdf, y_pdf),
                        label,
                        fontsize=fontsize,
                        color=MEASURE_TEXT_COLOR,
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
