#!/usr/bin/env python3
import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
import numpy as np
import cv2
from lxml import etree as LET


GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0
MEASURE_TEXT_COLOR = (0, 0, 0)  # black
MEASURE_TEXT_SIZE = 10.0
MEASURE_TEXT_X_OFFSET = 1.0
MEASURE_TEXT_BG_COLOR = (1, 1, 1)  # white

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


def _debug_dump_sigs_enabled() -> bool:
    v = os.getenv("DEBUG_GUIDES_DUMP_SIGS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _measure_labels_enabled() -> bool:
    v = os.getenv("ENABLE_MEASURE_LABELS", "1").strip()
    return v not in ("0", "false", "False", "no", "NO")


def _measure_label_mode() -> str:
    v = os.getenv("MEASURE_LABEL_MODE", "first_only").strip().lower()
    if v == "staff_start":
        return "staff_start"
    if v == "sequential":
        return "sequential"
    return "first_only"


def _measure_source_policy() -> str:
    v = os.getenv("MEASURE_SOURCE_POLICY", "mxl_strict").strip().lower()
    if v == "mxl_with_omr_fallback":
        return "mxl_with_omr_fallback"
    return "mxl_strict"


def _mxl_parser_policy() -> str:
    v = os.getenv("MXL_PARSER_POLICY", "auto").strip().lower()
    if v in ("stdlib", "lxml_recover", "auto"):
        return v
    return "auto"


def _debug_measure_labels_enabled() -> bool:
    v = os.getenv("DEBUG_MEASURE_LABELS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _debug_measure_markers_enabled() -> bool:
    v = os.getenv("DEBUG_MEASURE_MARKERS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _draw_measure_label(page: fitz.Page, rect: fitz.Rect, x: float, y: float, text: str) -> None:
    # Draw a tiny white background so labels stay visible over notation.
    tw = float(fitz.get_text_length(text, fontsize=MEASURE_TEXT_SIZE))
    th = float(MEASURE_TEXT_SIZE + 2.0)
    bg = fitz.Rect(x - 1.0, y - th + 1.0, x + tw + 1.0, y + 1.0)

    x0 = max(0.0, min(bg.x0, rect.width))
    y0 = max(0.0, min(bg.y0, rect.height))
    x1 = max(0.0, min(bg.x1, rect.width))
    y1 = max(0.0, min(bg.y1, rect.height))
    if x1 > x0 and y1 > y0:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=MEASURE_TEXT_BG_COLOR, fill=MEASURE_TEXT_BG_COLOR)

    page.insert_text(
        (x, y),
        text,
        fontsize=MEASURE_TEXT_SIZE,
        color=MEASURE_TEXT_COLOR,
    )

    if _debug_measure_markers_enabled():
        cx = x
        cy = max(0.0, y - 6.0)
        page.draw_circle((cx, cy), 3.0, color=(0, 0, 1), fill=(1, 1, 0), width=0.7)
        page.draw_line((cx - 5.0, cy), (cx + 5.0, cy), color=(0, 0, 1), width=0.7)
        page.draw_line((cx, cy - 5.0), (cx, cy + 5.0), color=(0, 0, 1), width=0.7)


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _children_named(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in list(el) if _local_name(c.tag) == name]


def _truthy_attr(v: str | None) -> bool:
    if not v:
        return False
    return v.strip().lower() in ("1", "true", "yes")


def _find_mxl_path_for_omr(omr_path: str) -> str | None:
    root, ext = os.path.splitext(omr_path)
    if ext.lower() != ".omr":
        return None
    mxl_path = f"{root}.mxl"
    return mxl_path if os.path.exists(mxl_path) else None


def _score_xml_member_from_mxl(z: zipfile.ZipFile) -> str | None:
    container_path = "META-INF/container.xml"
    if container_path in z.namelist():
        try:
            root = ET.fromstring(z.read(container_path))
            for rf in root.iter():
                if _local_name(rf.tag) == "rootfile":
                    full_path = rf.get("full-path")
                    if full_path and full_path in z.namelist():
                        return full_path
        except Exception:
            pass

    for name in z.namelist():
        lname = name.lower()
        if name.startswith("META-INF/"):
            continue
        if lname.endswith(".musicxml") or lname.endswith(".xml"):
            return name
    return None


def _parse_mxl_system_start_numbers_with_meta(mxl_path: str) -> dict:
    meta = {
        "mxl_path": mxl_path,
        "mxl_parse_status": "error",
        "mxl_member_path": None,
        "mxl_parser_used": "none",
        "mxl_pages": 0,
        "mxl_system_starts_per_page": [],
        "mxl_error": None,
    }

    parser_policy = _mxl_parser_policy()
    pages: list[list[str]] = []
    xml_bytes = b""
    try:
        with zipfile.ZipFile(mxl_path, "r") as z:
            score_member = _score_xml_member_from_mxl(z)
            if not score_member:
                meta["mxl_parse_status"] = "missing_score_member"
                return meta
            meta["mxl_member_path"] = score_member
            xml_bytes = z.read(score_member)
    except Exception as exc:
        meta["mxl_parse_status"] = "zip_error"
        meta["mxl_error"] = str(exc)
        return meta

    root = None
    stdlib_error = None
    lxml_error = None

    if parser_policy in ("stdlib", "auto"):
        try:
            root = ET.fromstring(xml_bytes)
            meta["mxl_parser_used"] = "stdlib"
        except Exception as exc:
            stdlib_error = str(exc)
            if parser_policy == "stdlib":
                meta["mxl_parse_status"] = "xml_parse_error"
                meta["mxl_error"] = stdlib_error
                return meta

    if root is None and parser_policy in ("lxml_recover", "auto"):
        try:
            lxml_parser = LET.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
            lxml_root = LET.fromstring(xml_bytes, parser=lxml_parser)
            xml_clean = LET.tostring(lxml_root, encoding="utf-8")
            root = ET.fromstring(xml_clean)
            meta["mxl_parser_used"] = "lxml_recover"
        except Exception as exc:
            lxml_error = str(exc)
            meta["mxl_parse_status"] = "zip_or_xml_error"
            if stdlib_error and lxml_error:
                meta["mxl_error"] = f"stdlib={stdlib_error} | lxml={lxml_error}"
            else:
                meta["mxl_error"] = stdlib_error or lxml_error
            return meta

    if root is None:
        meta["mxl_parse_status"] = "zip_or_xml_error"
        meta["mxl_error"] = stdlib_error or lxml_error or "unknown parse failure"
        return meta

    if _local_name(root.tag) != "score-partwise":
        meta["mxl_parse_status"] = "unsupported_root"
        return meta

    parts = _children_named(root, "part")
    if not parts:
        meta["mxl_parse_status"] = "missing_parts"
        return meta

    measures = _children_named(parts[0], "measure")
    if not measures:
        meta["mxl_parse_status"] = "missing_measures"
        return meta

    for i, measure in enumerate(measures):
        label = (measure.get("number") or "").strip() or str(i + 1)
        if i == 0:
            pages = [[label]]
            continue

        is_new_page = False
        is_new_system = False
        for pr in _children_named(measure, "print"):
            is_new_page = is_new_page or _truthy_attr(pr.get("new-page"))
            is_new_system = is_new_system or _truthy_attr(pr.get("new-system"))

        if is_new_page:
            pages.append([label])
        elif is_new_system:
            if not pages:
                pages = [[]]
            pages[-1].append(label)

    meta["mxl_parse_status"] = "ok"
    meta["mxl_pages"] = len(pages)
    meta["mxl_system_starts_per_page"] = pages
    return meta


def _apply_mxl_staff_start_labels(
    staff_start_labels_pdf: list[tuple[float, float, float, str]],
    system_staff_counts: list[int],
    mxl_page_system_starts: list[str],
) -> tuple[list[tuple[float, float, float, str]] | None, str]:
    if not staff_start_labels_pdf or not system_staff_counts or not mxl_page_system_starts:
        return None, "empty_inputs"
    if len(system_staff_counts) != len(mxl_page_system_starts):
        return None, "system_count_mismatch"

    out: list[tuple[float, float, float, str]] = []
    idx = 0
    for sys_idx, staff_cnt in enumerate(system_staff_counts):
        label = mxl_page_system_starts[sys_idx]
        for _ in range(staff_cnt):
            if idx >= len(staff_start_labels_pdf):
                return None, "staff_label_underflow"
            x, y0, y1, _ = staff_start_labels_pdf[idx]
            out.append((x, y0, y1, label))
            idx += 1

    if idx != len(staff_start_labels_pdf):
        return None, "staff_label_overflow"
    return out, "ok"


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


def _bounds_overlap_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _tag_looks_like_clef(el: ET.Element) -> bool:
    t = (el.tag or "").lower()
    if t == "clef" or t.endswith("clef"):
        return True
    for k in ("shape", "type", "kind", "name", "family"):
        v = el.get(k)
        if v and "clef" in v.lower():
            return True
    return False


def _clef_id_from_header(staff: ET.Element) -> int | None:
    header = staff.find("header")
    if header is None:
        return None
    txt = header.findtext("clef")
    if not txt:
        return None
    try:
        return int(txt.strip())
    except Exception:
        return None


def _index_inters(inters: ET.Element | None) -> dict[int, ET.Element]:
    """
    Index inter elements by id, scanning descendants.
    """
    out: dict[int, ET.Element] = {}
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


def _clef_bounds_from_header(
    inter_by_id: dict[int, ET.Element], staff: ET.Element
) -> tuple[float, float, float, float] | None:
    clef_id = _clef_id_from_header(staff)
    if clef_id is None:
        return None
    clef_el = inter_by_id.get(clef_id)
    if clef_el is None:
        return None
    return _bounds_of(clef_el)


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


def _dump_sigs_for_staff(
    sheet_xml_path: str,
    staff_id: str,
    page: ET.Element,
    system: ET.Element,
    chosen_inters: ET.Element | None,
    y_top: float,
    y_bot: float,
    clef_id: int | None,
) -> None:
    if not (_debug_enabled() and _debug_dump_sigs_enabled() and _debug_match(sheet_xml_path, staff_id)):
        return

    sig_inters_list = page.findall(".//sig/inters")
    _dbg(
        sheet_xml_path,
        staff_id,
        f"[DBG] SIGS sheet={sheet_xml_path} sys={system.get('id')} staff={staff_id} sig_count={len(sig_inters_list)} "
        f"chosen={'yes' if chosen_inters is not None else 'no'} clef_id={clef_id}",
    )

    chosen_key = id(chosen_inters) if chosen_inters is not None else None

    for idx, inters in enumerate(sig_inters_list):
        inter_by_id = _index_inters(inters)
        has_clef = (clef_id in inter_by_id) if clef_id is not None else False

        overlap_cnt = 0
        leftmost = None  # (x0, x1, tag, iid)
        for el in inters.iter():
            b = _bounds_of(el)
            if b is None:
                continue
            x, y, w, h = b
            x0, x1 = x, x + w
            y0, y1 = y, y + h
            if not _bounds_overlap_1d(y0, y1, y_top, y_bot):
                continue
            overlap_cnt += 1
            iid = el.get("id")
            if leftmost is None or x0 < leftmost[0]:
                leftmost = (x0, x1, el.tag, iid)

        is_chosen = (chosen_key is not None and id(inters) == chosen_key)
        _dbg(
            sheet_xml_path,
            staff_id,
            f"  [DBG] SIG idx={idx} chosen={is_chosen} overlap_bounds={overlap_cnt} has_clef_id={has_clef} "
            f"leftmost={None if leftmost is None else f'{leftmost[2]} id={leftmost[3]} x=[{leftmost[0]:.1f},{leftmost[1]:.1f}]'}",
        )


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
    measure_marks_px: list[tuple[float, float, float, str]] = []
    staff_start_marks_px: list[tuple[float, float, float, str]] = []
    system_staff_counts: list[int] = []

    pages = root.findall("page")
    if not pages:
        return pic_w, pic_h, guides_px, measure_marks_px, staff_start_marks_px, 0, system_staff_counts

    staff_total = 0

    measure_counter = 1

    for page in pages:
        systems = page.findall(".//system")
        if not systems:
            systems = page.findall("system")

        for sys_index, system in enumerate(systems):
            staff_nodes = system.findall(".//staff")
            system_staff_counts.append(len(staff_nodes))

            # FIX: SIG is per-system; prefer system-local inters
            system_inters = system.find(".//sig/inters")
            if system_inters is None:
                system_inters = page.find(".//sig/inters")

            inter_by_id = _index_inters(system_inters)

            def barline_span_for_staff(staff_id: str, y_hint: float | None) -> tuple[float, float] | None:
                if system_inters is None or not staff_id:
                    return None

                best_covering = None   # (bx, y_top, y_bot)
                best_closest = None    # (dist, bx, y_top, y_bot)
                best_any = None        # (bx, y_top, y_bot)

                for el in system_inters.iter():
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

            def barline_xs_for_staff(staff_id: str, y_top: float, y_bot: float) -> list[float]:
                if system_inters is None:
                    return []

                y_pad = max(2.0, 0.15 * (y_bot - y_top))
                want_y0 = y_top - y_pad
                want_y1 = y_bot + y_pad

                xs: list[float] = []
                for el in system_inters.iter():
                    if (el.tag or "").lower() != "barline":
                        continue

                    el_staff = el.get("staff")
                    if el_staff and staff_id and el_staff != staff_id:
                        continue

                    b = el.find("bounds")
                    if b is None:
                        continue
                    bx = _safe_float(b.get("x"))
                    if bx is None:
                        continue

                    med = el.find("median")
                    if med is not None:
                        p1 = med.find("p1")
                        p2 = med.find("p2")
                        if p1 is not None and p2 is not None:
                            y1 = _safe_float(p1.get("y"))
                            y2 = _safe_float(p2.get("y"))
                            if y1 is not None and y2 is not None:
                                by0 = float(min(y1, y2))
                                by1 = float(max(y1, y2))
                                if _bounds_overlap_1d(by0, by1, want_y0, want_y1):
                                    xs.append(float(bx))
                                continue

                    by = _safe_float(b.get("y"))
                    bh = _safe_float(b.get("h"))
                    if by is None or bh is None or bh <= 0:
                        continue
                    by0 = float(by)
                    by1 = float(by + bh)
                    if _bounds_overlap_1d(by0, by1, want_y0, want_y1):
                        xs.append(float(bx))

                if not xs:
                    return []

                xs.sort()
                deduped: list[float] = []
                merge_tol = max(2.0, (0.25 * expected_spacing) if expected_spacing > 0 else 3.0)
                for x in xs:
                    if deduped and abs(x - deduped[-1]) <= merge_tol:
                        continue
                    deduped.append(x)
                return deduped

            for staff in staff_nodes:
                staff_total += 1
                staff_id = staff.get("id") or ""
                system_id = system.get("id") or "?"
                page_id = page.get("id") or "?"

                header = staff.find("header")
                header_start = _safe_float(header.get("start")) if header is not None else None
                clef_id = _clef_id_from_header(staff)

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

                # DEBUG A+B: SIG inventory + which SIG has clef id
                _dump_sigs_for_staff(
                    sheet_xml_path=sheet_xml_path,
                    staff_id=staff_id,
                    page=page,
                    system=system,
                    chosen_inters=system_inters,
                    y_top=y_top,
                    y_bot=y_bot,
                    clef_id=clef_id,
                )

                # Clef bounds now use system-scoped inters
                clef_b = _clef_bounds_from_header(inter_by_id, staff)
                if clef_b is None:
                    clef_b = _clef_bounds_fallback(system_inters, staff_id, y_top, y_bot, header_start)

                x_left = _safe_float(staff.get("left"))
                line_min_x = float(min(all_line_xmins)) if all_line_xmins else None
                staff_left_raw = _safe_float(staff.get("left"))

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

                # 2) Header clamp (small, always safe)
                if header_start is not None:
                    base_guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(header_start) - base_guard
                    if x_left > max_x:
                        x_left = max_x

                # 3) Clef clamp (when clef bounds exist)
                if clef_b is not None:
                    clef_x, _, _, _ = clef_b
                    guard = (0.25 * expected_spacing) if expected_spacing > 0 else 6.0
                    max_x = float(clef_x) - guard
                    if x_left > max_x:
                        x_left = max_x

                # 4) Collision clamp (system-scoped inters)
                x_left = _push_left_if_inside_any_symbol(
                    inters=system_inters,
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
                    inters=system_inters,
                    y_top=y_top,
                    y_bot=y_bot,
                    x_center=x_postpad,
                    x_window=140.0,
                )

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
                staff_barline_xs = barline_xs_for_staff(staff_id, y_top, y_bot)
                staff_barline_ids_raw = (staff.findtext("barlines") or "").strip()
                staff_barline_ids = [tok for tok in staff_barline_ids_raw.split() if tok]
                counter_before = measure_counter
                increment_bars = len(staff_barline_xs)
                carryover_detected = False

                if increment_bars > 0 and sys_index > 0:
                    left_ref = None
                    for cand in (header_start, staff_left_raw, line_min_x):
                        if cand is not None:
                            left_ref = float(cand)
                            break

                    if left_ref is not None:
                        first_bar_x = float(min(staff_barline_xs))
                        carry_tol = max(8.0, (0.7 * expected_spacing) if expected_spacing > 0 else 8.0)
                        if abs(first_bar_x - left_ref) <= carry_tol:
                            carryover_detected = True
                            increment_bars = max(0, increment_bars - 1)

                if staff_barline_xs:
                    staff_start_marks_px.append((x_postpad, y_top, y_bot, str(measure_counter)))
                    measure_counter += increment_bars
                counter_after = measure_counter

                if _debug_measure_labels_enabled():
                    print(
                        f"[DBG] count page={page_id} sys={system_id} staff={staff_id} "
                        f"staff_xml_barlines={len(staff_barline_ids)} barline_ids={staff_barline_ids_raw or '-'} "
                        f"used_barlines={len(staff_barline_xs)} "
                        f"omr_carryover_detected={str(carryover_detected).lower()} "
                        f"increment_before={len(staff_barline_xs)} increment_after={increment_bars} "
                        f"counter_before={counter_before} counter_after={counter_after}",
                        flush=True,
                    )

                for measure_num, bx in enumerate(staff_barline_xs, start=1):
                    measure_marks_px.append((bx, y_top, y_bot, str(measure_num)))

    return pic_w, pic_h, guides_px, measure_marks_px, staff_start_marks_px, staff_total, system_staff_counts


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
    write_measure_labels = _measure_labels_enabled()
    measure_label_mode = _measure_label_mode()
    measure_source_policy = _measure_source_policy()
    mapping_debug_path = os.getenv("MEASURE_MAPPING_DEBUG_PATH", "").strip()
    mapping_debug: list[dict] = []

    mxl_meta = {
        "mxl_path": None,
        "mxl_parse_status": "not_requested",
        "mxl_member_path": None,
        "mxl_parser_used": "none",
        "mxl_pages": 0,
        "mxl_system_starts_per_page": [],
        "mxl_error": None,
    }
    mxl_path = _find_mxl_path_for_omr(omr_path)
    if measure_label_mode == "staff_start":
        if mxl_path:
            mxl_meta = _parse_mxl_system_start_numbers_with_meta(mxl_path)
        else:
            mxl_meta = {
                "mxl_path": None,
                "mxl_parse_status": "missing_mxl_file",
                "mxl_member_path": None,
                "mxl_parser_used": "none",
                "mxl_pages": 0,
                "mxl_system_starts_per_page": [],
                "mxl_error": None,
            }

    if _debug_measure_labels_enabled():
        print(
            "[DBG] measure_source "
            f"policy={measure_source_policy} "
            f"mxl_parse_status={mxl_meta.get('mxl_parse_status')} "
            f"mxl_parser_used={mxl_meta.get('mxl_parser_used')} "
            f"mxl_member_path={mxl_meta.get('mxl_member_path')} "
            f"mxl_pages={mxl_meta.get('mxl_pages')}",
            flush=True,
        )

    with zipfile.ZipFile(omr_path, "r") as z:
        sheet_paths = _sorted_sheet_xml_paths(z)
        if not sheet_paths:
            raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

        for page_index in range(doc.page_count):
            if page_index >= len(sheet_paths):
                continue

            page = doc[page_index]
            sheet_xml_path = sheet_paths[page_index]

            (
                pic_w,
                pic_h,
                guides_px,
                measure_marks_px,
                staff_start_marks_px,
                staff_total,
                system_staff_counts,
            ) = _parse_sheet(z, sheet_xml_path)
            if pic_w <= 0 or pic_h <= 0:
                continue

            rect = page.rect
            scale_x = rect.width / pic_w
            scale_y = rect.height / pic_h

            guides_pdf = []
            first_labels_pdf = []
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
                first_labels_pdf.append((x_pdf, y0_pdf, y1_pdf, "1"))

            sequential_labels_pdf = []
            for (x_px, y0_px, y1_px, text) in measure_marks_px:
                sequential_labels_pdf.append((x_px * scale_x, y0_px * scale_y, y1_px * scale_y, text))
            staff_start_labels_pdf = []
            for (x_px, y0_px, y1_px, text) in staff_start_marks_px:
                staff_start_labels_pdf.append((x_px * scale_x, y0_px * scale_y, y1_px * scale_y, text))

            staff_start_source = "none"
            mapping_status = "not_applicable"
            mapping_reason = "-"
            page_mxl_starts: list[str] = []

            if measure_label_mode == "staff_start":
                mxl_pages = mxl_meta.get("mxl_system_starts_per_page", [])
                if mxl_meta.get("mxl_parse_status") == "ok" and page_index < len(mxl_pages):
                    page_mxl_starts = list(mxl_pages[page_index])
                    mxl_mapped, map_reason = _apply_mxl_staff_start_labels(
                        staff_start_labels_pdf=staff_start_labels_pdf,
                        system_staff_counts=system_staff_counts,
                        mxl_page_system_starts=page_mxl_starts,
                    )
                    if mxl_mapped is not None:
                        staff_start_labels_pdf = mxl_mapped
                        staff_start_source = "mxl"
                        mapping_status = "ok"
                        mapping_reason = "mxl_mapping_ok"
                    else:
                        mapping_status = "error"
                        mapping_reason = f"mxl_map_{map_reason}"
                else:
                    mapping_status = "error"
                    if mxl_meta.get("mxl_parse_status") != "ok":
                        mapping_reason = f"mxl_parse_{mxl_meta.get('mxl_parse_status')}"
                    else:
                        mapping_reason = "mxl_page_index_missing"

                if staff_start_source != "mxl":
                    if measure_source_policy == "mxl_with_omr_fallback" and staff_start_labels_pdf:
                        staff_start_source = "omr_fallback"
                        mapping_status = "fallback"
                        mapping_reason = f"{mapping_reason}|fallback_to_omr"
                    else:
                        staff_start_labels_pdf = []
                        staff_start_source = "none"

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
                    first_labels_pdf.append((x_pdf, y0_pdf, y1_pdf, "1"))

            if write_measure_labels:
                labels_to_draw = first_labels_pdf
                if measure_label_mode == "staff_start":
                    labels_to_draw = staff_start_labels_pdf
                elif measure_label_mode == "sequential" and sequential_labels_pdf:
                    labels_to_draw = sequential_labels_pdf

                labels_drawn = 0
                labels_in_bounds = 0
                sample_positions = []
                for (x_pdf, y0_pdf, y1_pdf, label_text) in labels_to_draw:
                    staff_h = max(1.0, y1_pdf - y0_pdf)
                    y_offset = max(8.0, 0.45 * staff_h)
                    tw = float(fitz.get_text_length(label_text, fontsize=MEASURE_TEXT_SIZE))
                    x_text = min(max(0.0, x_pdf + MEASURE_TEXT_X_OFFSET), max(0.0, rect.width - tw - 2.0))
                    y_text = max(MEASURE_TEXT_SIZE + 2.0, y0_pdf - y_offset)
                    y_text = min(y_text, max(MEASURE_TEXT_SIZE + 2.0, rect.height - 2.0))

                    if 0.0 <= x_text <= max(0.0, rect.width - tw) and (MEASURE_TEXT_SIZE + 2.0) <= y_text <= rect.height:
                        labels_in_bounds += 1
                        if len(sample_positions) < 5:
                            sample_positions.append(
                                f"{label_text}@({x_text:.1f},{y_text:.1f}) staff_y=({y0_pdf:.1f},{y1_pdf:.1f})"
                            )

                    _draw_measure_label(page, rect, x_text, y_text, label_text)
                    labels_drawn += 1

                if _debug_measure_labels_enabled():
                    print(
                        f"[DBG] page={page_index+1} sheet={sheet_xml_path} "
                        f"staff_start_source={staff_start_source} "
                        f"mapping_status={mapping_status} mapping_reason={mapping_reason}",
                        flush=True,
                    )

            mapping_debug.append(
                {
                    "page_index": page_index,
                    "sheet_xml_path": sheet_xml_path,
                    "measure_mode": measure_label_mode,
                    "measure_source_policy": measure_source_policy,
                    "mxl_parse_status": mxl_meta.get("mxl_parse_status"),
                    "mxl_member_path": mxl_meta.get("mxl_member_path"),
                    "mxl_parser_used": mxl_meta.get("mxl_parser_used"),
                    "mxl_page_system_starts": page_mxl_starts,
                    "omr_system_staff_counts": system_staff_counts,
                    "staff_start_source": staff_start_source,
                    "mapping_status": mapping_status,
                    "mapping_reason": mapping_reason,
                    "assigned_labels": [t[3] for t in staff_start_labels_pdf],
                    "staff_start_candidate_count": len(staff_start_labels_pdf),
                }
            )

    doc.save(output_pdf)
    doc.close()

    if mapping_debug_path:
        out_dir = os.path.dirname(mapping_debug_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "measure_mode": measure_label_mode,
            "measure_source_policy": measure_source_policy,
            "mxl_parse_status": mxl_meta.get("mxl_parse_status"),
            "mxl_member_path": mxl_meta.get("mxl_member_path"),
            "mxl_parser_used": mxl_meta.get("mxl_parser_used"),
            "mxl_pages": mxl_meta.get("mxl_pages"),
            "mxl_system_starts_per_page": mxl_meta.get("mxl_system_starts_per_page"),
            "mxl_error": mxl_meta.get("mxl_error"),
            "pages": mapping_debug,
        }
        with open(mapping_debug_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_guides_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
