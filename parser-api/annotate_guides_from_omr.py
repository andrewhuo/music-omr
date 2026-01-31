#!/usr/bin/env python3
import os
import re
import sys
import zipfile
import math
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
import numpy as np
import cv2


# --------------------------
# Rendering knobs
# --------------------------
GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0

# Touch staff start by default.
PAD_LEFT_PX_DEFAULT = 0.0

MEASURE_TEXT_COLOR = (0, 0, 0)  # black
MEASURE_MIN_FONTSIZE = 7.0
MEASURE_MAX_FONTSIZE = 12.0

# DEBUG: barline overlay
DEBUG_BAR_COLOR = (0, 0, 1)  # blue
DEBUG_BAR_WIDTH = 0.9

_SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


# --------------------------
# Env helpers
# --------------------------
def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip()
    return v in ("1", "true", "True", "yes", "YES")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


def _debug_enabled() -> bool:
    return _env_truthy("DEBUG_GUIDES", "0")


def _debug_match(sheet_xml_path: str, staff_id: str) -> bool:
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
    return _env_truthy("DEBUG_GUIDES_DUMP_INTERS", "0")


def _debug_guides_summary_enabled() -> bool:
    return _env_truthy("DEBUG_GUIDES_SUMMARY", "0")


def _debug_guides_suspect_only_enabled() -> bool:
    return _env_truthy("DEBUG_GUIDES_SUSPECT_ONLY", "0")


def _meas_debug_enabled() -> bool:
    return _env_truthy("DEBUG_MEASURES", "0")


def _meas_debug_match(sheet_xml_path: str, system_key: str) -> bool:
    want_sheet = os.getenv("DEBUG_MEASURES_SHEET", "").strip()
    want_sys = os.getenv("DEBUG_MEASURES_SYS_ID", "").strip()
    if want_sheet and want_sheet not in sheet_xml_path:
        return False
    if want_sys and want_sys != str(system_key):
        return False
    return True


def _meas_dbg(sheet_xml_path: str, system_key: str, msg: str) -> None:
    if _meas_debug_enabled() and _meas_debug_match(sheet_xml_path, system_key):
        print(msg, flush=True)


def _bars_debug_enabled() -> bool:
    return _env_truthy("DEBUG_DRAW_BARS", "0")


def _bars_alpha() -> float:
    return _env_float("DEBUG_BARS_ALPHA", 0.85)


def _use_cv_bars() -> bool:
    return _env_truthy("USE_CV_BARS", "0")


def _cv_zoom() -> float:
    z = _env_float("CV_ZOOM", 2.0)
    return z if z > 0 else 2.0


def _pad_left_px() -> float:
    return _env_float("PAD_LEFT_PX", PAD_LEFT_PX_DEFAULT)


def _guide_consensus_enabled() -> bool:
    # Legacy "thresholded" clamp. You can keep it ON, but the new hard snap
    # (GUIDE_FORCE_NONINDENTED_X=1) should make this mostly redundant.
    return _env_truthy("GUIDE_CONSENSUS", "1")


def _guide_consensus_percentile() -> float:
    # Robust "system start" estimator across staves (legacy).
    p = _env_float("GUIDE_CONSENSUS_P", 0.10)
    return max(0.0, min(1.0, p))


def _guide_consensus_thresh_mult() -> float:
    # Threshold = max(10px, mult * expected_spacing)
    m = _env_float("GUIDE_CONSENSUS_THRESH_MULT", 0.60)
    return max(0.0, m)


def _guide_clef_clamp_enabled() -> bool:
    # ON by default: last-ditch safety to avoid landing inside clef bbox.
    return _env_truthy("GUIDE_CLEF_CLAMP", "1")


def _guide_force_nonindented_x_enabled() -> bool:
    # NOW: per-system one-sided snap (winsorize right tail) for non-indented systems.
    return _env_truthy("GUIDE_FORCE_NONINDENTED_X", "1")


def _guide_force_hard() -> bool:
    # If 1: force ALL non-indented to target_x (aggressive)
    return _env_truthy("GUIDE_FORCE_HARD", "0")


def _guide_target_left_frac() -> float:
    # Use the left cluster: take the lowest K = ceil(frac * n) and average.
    f = _env_float("GUIDE_TARGET_LEFT_FRAC", 0.75)
    return max(0.50, min(1.0, f))


def _guide_snap_eps_px() -> float:
    # Tiny epsilon in picture-space px. Anything to the right of target_x + eps snaps to target_x.
    e = _env_float("GUIDE_SNAP_EPS_PX", 2.0)
    return max(0.5, min(6.0, e))


def _guide_page_force_enabled() -> bool:
    # Optional legacy page-level forcing (disabled by default now).
    return _env_truthy("GUIDE_PAGE_FORCE", "0")


def _outlier_mz_threshold() -> float:
    return _env_float("OUTLIER_MZ_THRESHOLD", 3.5)


def _outlier_abs_floor_px() -> float:
    return _env_float("OUTLIER_ABS_FLOOR_PX", 10.0)


def _attr_truthy(v: str | None) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if s == "":
        return False
    return s in ("1", "true", "True", "yes", "YES")


# --------------------------
# Small helpers
# --------------------------
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


def _mean(xs: list[float]) -> float:
    return float(sum(xs)) / float(len(xs)) if xs else 0.0


def _mad(xs: list[float], med: float) -> float:
    return _median([abs(x - med) for x in xs])


def _best_five_by_spacing(yxs: list[tuple[float, float]], expected_spacing: float) -> list[tuple[float, float]]:
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


def _clef_bounds_from_header(inter_by_id: dict[int, ET.Element], staff: ET.Element) -> tuple[float, float, float, float] | None:
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
    return _bounds_of(clef_el)


def _bounds_overlap_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _looks_like_clef_inter(el: ET.Element) -> bool:
    t = (el.tag or "").lower()
    if "clef" in t:
        return True
    for k in ("shape", "type", "kind", "name", "family"):
        v = el.get(k)
        if v and "clef" in v.lower():
            return True
    return False


def _clef_bounds_fallback(inters: ET.Element | None, y_top: float, y_bot: float) -> tuple[float, float, float, float] | None:
    """
    When header->clef id is missing/odd, fall back to scanning <sig/inters> for clef-like
    symbols overlapping this staff's y-band. Choose the leftmost candidate.
    """
    if inters is None:
        return None

    y_pad = 0.25 * (y_bot - y_top)
    want_y0 = y_top - y_pad
    want_y1 = y_bot + y_pad

    best = None
    best_x = None

    for el in inters.iter():
        if not _looks_like_clef_inter(el):
            continue
        b = _bounds_of(el)
        if b is None:
            continue
        x, y, w, h = b
        y0 = y
        y1 = y + h
        if y1 < want_y0 or y0 > want_y1:
            continue
        if best_x is None or x < best_x:
            best_x = x
            best = b

    return best


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


def _count_measures_in_system(system: ET.Element) -> int:
    ms = system.findall("./measures/measure")
    if not ms:
        ms = system.findall("./measure")
    if not ms:
        ms = system.findall(".//measure")

    if not ms:
        return 0

    ids: list[str] = []
    for m in ms:
        mid = (m.get("id") or m.get("number") or "").strip()
        if mid:
            ids.append(mid)

    if ids:
        return len(set(ids))
    return len(ms)


def _looks_like_barline_inter(el: ET.Element) -> bool:
    t = (el.tag or "").lower()
    if "barline" in t:
        return True
    for k in ("shape", "type", "kind", "name", "family"):
        v = el.get(k)
        if v and "barline" in v.lower():
            return True
        if v and "bar line" in v.lower():
            return True
    return False


def _barline_xs_in_y_band(inters: ET.Element | None, y0: float, y1: float) -> list[float]:
    if inters is None:
        return []

    xs: list[float] = []
    for el in inters.iter():
        if not _looks_like_barline_inter(el):
            continue

        b = el.find("bounds")
        if b is None:
            continue

        bx = _safe_float(b.get("x"))
        by = _safe_float(b.get("y"))
        bw = _safe_float(b.get("w"))
        bh = _safe_float(b.get("h"))
        if bx is None:
            continue

        med = el.find("median")
        if med is not None:
            p1 = med.find("p1")
            p2 = med.find("p2")
            if p1 is not None and p2 is not None:
                yy1 = _safe_float(p1.get("y"))
                yy2 = _safe_float(p2.get("y"))
                if yy1 is not None and yy2 is not None:
                    by0 = float(min(yy1, yy2))
                    by1 = float(max(yy1, yy2))
                    if _bounds_overlap_1d(by0, by1, y0, y1):
                        xs.append(float(bx))
                    continue

        if by is None or bw is None or bh is None:
            continue
        by0 = float(by)
        by1 = float(by) + float(bh)
        if _bounds_overlap_1d(by0, by1, y0, y1):
            xs.append(float(bx))

    xs.sort()
    return xs


def _tail_xs(xs: list[float], n: int = 3) -> str:
    if not xs:
        return "[]"
    t = xs[-n:] if len(xs) > n else xs
    return "[" + ",".join(f"{x:.1f}" for x in t) + "]"


# --------------------------
# CV barline detection (page coordinate space)
# --------------------------
def _cv_barlines_xs_in_band_pdf(page: fitz.Page, y0_pdf: float, y1_pdf: float) -> tuple[list[float], float]:
    rect = page.rect

    y0_rel = float(y0_pdf) - float(rect.y0)
    y1_rel = float(y1_pdf) - float(rect.y0)

    y0_rel = max(0.0, min(float(rect.height), y0_rel))
    y1_rel = max(0.0, min(float(rect.height), y1_rel))
    if y1_rel <= y0_rel + 4.0:
        return ([], max(14.0, 0.02 * float(rect.width)))

    zoom = _cv_zoom()
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_y_px = float(pix.height) / float(rect.height)
    scale_x_px = float(pix.width) / float(rect.width)

    y0_px = int(max(0, min(pix.height - 1, round(y0_rel * scale_y_px))))
    y1_px = int(max(0, min(pix.height, round(y1_rel * scale_y_px))))
    if y1_px <= y0_px + 6:
        return ([], max(14.0, 0.02 * float(rect.width)))

    band = gray[y0_px:y1_px, :]
    thr = cv2.adaptiveThreshold(band, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    band_h = band.shape[0]
    k_h = max(18, int(round(band_h * 0.55)))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
    vertical = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs_px: list[float] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 0.60 * band_h:
            continue
        if w > max(6, int(0.012 * pix.width)):
            continue
        xs_px.append(x + 0.5 * w)

    xs_px.sort()
    eps_px = max(6.0, 0.012 * float(pix.width))
    xs_px = _dedupe_sorted_xs(xs_px, eps=eps_px)

    xs_pdf = [float(rect.x0) + float(x / scale_x_px) for x in xs_px]
    close_tol_pdf = max(14.0, 0.10 * (float(y1_rel - y0_rel) / 4.0))
    return (xs_pdf, close_tol_pdf)


# --------------------------
# Sheet parsing (OMR XML)
# --------------------------
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

    pages = root.findall("page")
    if not pages:
        return pic_w, pic_h, [], 0, []

    pad_left = _pad_left_px()
    staff_total = 0

    staff_recs: list[dict] = []
    system_recs: dict[str, dict] = {}

    for page_idx, page in enumerate(pages):
        inter_by_id = _index_inters(page)
        inters = page.find(".//sig/inters")

        systems = page.findall(".//system")
        if not systems:
            systems = page.findall("system")

        for sys_idx, system in enumerate(systems):
            sys_id = system.get("id") or ""
            sys_key = f"p{page_idx}_s{sys_idx}_id{sys_id}"
            sys_is_indented = _attr_truthy(system.get("indented"))

            if sys_key not in system_recs:
                system_recs[sys_key] = {
                    "sys_key": sys_key,
                    "sys_id": sys_id,
                    "page_idx": page_idx,
                    "sys_idx": sys_idx,
                    "is_indented": sys_is_indented,
                    "staff_spans": [],
                    "x_postpads": [],
                }

            for staff in system.findall(".//staff"):
                staff_total += 1
                staff_id = staff.get("id") or ""

                lines_node = staff.find("lines")
                line_nodes = [] if lines_node is None else lines_node.findall("line")

                yxs: list[tuple[float, float]] = []
                all_line_xmins: list[float] = []
                all_line_xmaxs: list[float] = []

                for ln in line_nodes:
                    pts = ln.findall("point")
                    if not pts:
                        continue

                    min_x = None
                    max_x = None
                    y_at_min_x = None

                    for p in pts:
                        x = _safe_float(p.get("x"))
                        y = _safe_float(p.get("y"))
                        if x is None or y is None:
                            continue
                        if min_x is None or x < min_x:
                            min_x = x
                            y_at_min_x = y
                        if max_x is None or x > max_x:
                            max_x = x

                    if min_x is None or y_at_min_x is None:
                        continue

                    yxs.append((float(y_at_min_x), float(min_x)))
                    all_line_xmins.append(float(min_x))
                    if max_x is not None:
                        all_line_xmaxs.append(float(max_x))

                yxs.sort(key=lambda t: t[0])

                ys5: list[float] | None = None
                if len(yxs) >= 5:
                    chosen = _best_five_by_spacing(yxs, expected_spacing)
                    chosen = sorted(chosen, key=lambda t: t[0])
                    ys5 = [t[0] for t in chosen]
                elif len(yxs) >= 2:
                    ys_partial = [t[0] for t in yxs]
                    ys5 = _synthesize_five_lines(ys_partial, expected_spacing)

                if ys5 is None:
                    _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP no_y_span sheet={sheet_xml_path} staff={staff_id}")
                    continue

                y_top = float(min(ys5))
                y_bot = float(max(ys5))
                if y_bot <= y_top or (y_bot - y_top) < 10.0:
                    _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP bad_y_span sheet={sheet_xml_path} staff={staff_id}")
                    continue

                # Anchor x to staff line start (robust percentile)
                if all_line_xmins:
                    staff_start_x = float(_pct(all_line_xmins, 0.05))
                else:
                    x_left = _safe_float(staff.get("left"))
                    if x_left is None:
                        _dbg(sheet_xml_path, staff_id, f"[DBG] SKIP no_x sheet={sheet_xml_path} staff={staff_id}")
                        continue
                    staff_start_x = float(x_left)

                x_postpad = max(0.0, staff_start_x - pad_left)

                system_recs[sys_key]["x_postpads"].append(float(x_postpad))
                system_recs[sys_key]["staff_spans"].append((float(y_top), float(y_bot), staff_id))

                # Clef bounds: header reference first, then fallback scan
                clef_b = _clef_bounds_from_header(inter_by_id, staff)
                clef_source = "header" if clef_b is not None else "none"
                if clef_b is None:
                    fb = _clef_bounds_fallback(inters, y_top, y_bot)
                    if fb is not None:
                        clef_b = fb
                        clef_source = "fallback"

                _dump_inters_near_staff(sheet_xml_path, staff_id, inters, y_top, y_bot, x_postpad, x_window=140.0)

                if _debug_enabled() and _debug_match(sheet_xml_path, staff_id):
                    line_min_x = float(min(all_line_xmins)) if all_line_xmins else None
                    line_max_x = float(max(all_line_xmaxs)) if all_line_xmaxs else None
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] sheet={sheet_xml_path} sys_key={sys_key} staff={staff_id} "
                        f"indented={int(sys_is_indented)} expected_spacing={expected_spacing:.2f} "
                        f"line_min_x={line_min_x} line_max_x={line_max_x} "
                        f"staff_start_x(p05)={staff_start_x:.2f} PAD_LEFT_PX={pad_left:.2f} x_postpad={x_postpad:.2f} "
                        f"clef_b={'present' if clef_b is not None else None} clef_src={clef_source} "
                        f"y_top={y_top:.1f} y_bot={y_bot:.1f}",
                    )

                line_max_x = float(max(all_line_xmaxs)) if all_line_xmaxs else None

                staff_recs.append(
                    {
                        "sys_key": sys_key,
                        "sys_id": sys_id,
                        "staff_id": staff_id,
                        "is_indented": bool(sys_is_indented),
                        "x_postpad": float(x_postpad),
                        "x_raw": float(x_postpad),  # keep raw for summaries
                        "y_top": float(y_top),
                        "y_bot": float(y_bot),
                        "line_max_x": float(line_max_x) if line_max_x is not None else None,
                        "clef_b": clef_b,
                        "clef_source": clef_source,
                    }
                )

    # Threshold in picture px (legacy blocks)
    thr_mult = _guide_consensus_thresh_mult()
    if expected_spacing > 0:
        thresh_px = max(10.0, thr_mult * float(expected_spacing))
    else:
        thresh_px = 14.0

    # ------------------------------------------------------------
    # 0) Build per-system target_x from left cluster (non-indented)
    # ------------------------------------------------------------
    sys_target: dict[str, float] = {}
    left_frac = _guide_target_left_frac()

    for sys_key, rec in system_recs.items():
        if rec.get("is_indented"):
            continue  # never touch indented systems
        xs = [float(x) for x in rec.get("x_postpads", []) if x is not None]
        if not xs:
            continue
        xs.sort()
        k = max(2, int(math.ceil(left_frac * len(xs))))
        k = min(k, len(xs))
        target_x = _mean(xs[:k]) if k > 0 else xs[0]
        sys_target[sys_key] = float(target_x)

    # ------------------------------------------------------------
    # 1) Legacy per-system "consensus clamp" (thresholded)
    #    (kept for compatibility; should be mostly redundant now)
    # ------------------------------------------------------------
    if _guide_consensus_enabled():
        p = _guide_consensus_percentile()

        sys_consensus: dict[str, float] = {}
        for sys_key, rec in system_recs.items():
            if rec.get("is_indented"):
                continue
            xs = [float(x) for x in rec.get("x_postpads", []) if x is not None]
            if not xs:
                continue
            try:
                sys_consensus[sys_key] = float(_pct(xs, p))
            except Exception:
                sys_consensus[sys_key] = float(min(xs))

        for r in staff_recs:
            if r.get("is_indented"):
                continue
            sk = r["sys_key"]
            c = sys_consensus.get(sk)
            if c is None:
                continue
            x0 = float(r["x_postpad"])
            if x0 > float(c) + thresh_px:
                if _debug_enabled() and _debug_match(sheet_xml_path, r["staff_id"]):
                    _dbg(sheet_xml_path, r["staff_id"], f"[DBG] SYS_CLAMP sys={sk} x={x0:.2f} -> {c:.2f} thresh={thresh_px:.2f}")
                r["x_postpad"] = float(c)

    # ------------------------------------------------------------
    # 2) NEW: per-system hard one-sided snap (winsorize right tail)
    #    Non-indented only, and snaps to *target_x* exactly.
    # ------------------------------------------------------------
    snapped_by_sys: dict[str, list[str]] = {}
    snap_eps = _guide_snap_eps_px()
    hard = _guide_force_hard()

    if _guide_force_nonindented_x_enabled():
        for r in staff_recs:
            if r.get("is_indented"):
                continue
            sk = r["sys_key"]
            tx = sys_target.get(sk)
            if tx is None:
                continue
            x0 = float(r["x_postpad"])
            if hard:
                if abs(x0 - tx) > 1e-6:
                    snapped_by_sys.setdefault(sk, []).append(str(r["staff_id"]))
                r["x_postpad"] = float(tx)
            else:
                if x0 > float(tx) + float(snap_eps):
                    if _debug_enabled() and _debug_match(sheet_xml_path, r["staff_id"]):
                        _dbg(sheet_xml_path, r["staff_id"], f"[DBG] SYS_SNAP sys={sk} x={x0:.2f} -> {tx:.2f} eps={snap_eps:.2f}")
                    snapped_by_sys.setdefault(sk, []).append(str(r["staff_id"]))
                    r["x_postpad"] = float(tx)

    # ------------------------------------------------------------
    # 3) Optional legacy page-level force (disabled by default)
    # ------------------------------------------------------------
    if _guide_page_force_enabled():
        xs = [float(r["x_postpad"]) for r in staff_recs if not r.get("is_indented")]
        if len(xs) >= 6:
            med = _median(xs)
            mad = max(_mad(xs, med), _outlier_abs_floor_px())
            mz_thr = _outlier_mz_threshold()

            def mz(x: float) -> float:
                return 0.6745 * (x - med) / mad

            inliers = [x for x in xs if abs(mz(x)) <= mz_thr or x <= med]
            forced_x = _mean(inliers) if len(inliers) >= 4 else med

            for r in staff_recs:
                if r.get("is_indented"):
                    continue
                x0 = float(r["x_postpad"])
                if (x0 > forced_x + thresh_px) or (mz(x0) > mz_thr):
                    if _debug_enabled() and _debug_match(sheet_xml_path, r["staff_id"]):
                        _dbg(
                            sheet_xml_path,
                            r["staff_id"],
                            f"[DBG] PAGE_FORCE x={x0:.2f} -> {forced_x:.2f} mz={mz(x0):.2f} thr={mz_thr:.2f} thresh={thresh_px:.2f}",
                        )
                    r["x_postpad"] = float(forced_x)

    # ------------------------------------------------------------
    # 4) Clef clamp (secondary safety), non-indented only
    #    Uses header OR fallback clef bounds.
    # ------------------------------------------------------------
    clef_fallback_used_by_sys: dict[str, int] = {}
    clef_header_used_by_sys: dict[str, int] = {}
    clef_none_by_sys: dict[str, int] = {}

    if _guide_clef_clamp_enabled():
        guard = (max(2.0, 0.25 * float(expected_spacing)) if expected_spacing > 0 else 6.0)
        pad_left_again = _pad_left_px()
        for r in staff_recs:
            if r.get("is_indented"):
                continue
            sk = r["sys_key"]
            src = str(r.get("clef_source") or "none")
            if src == "fallback":
                clef_fallback_used_by_sys[sk] = clef_fallback_used_by_sys.get(sk, 0) + 1
            elif src == "header":
                clef_header_used_by_sys[sk] = clef_header_used_by_sys.get(sk, 0) + 1
            else:
                clef_none_by_sys[sk] = clef_none_by_sys.get(sk, 0) + 1

            clef_b = r.get("clef_b")
            if clef_b is None:
                continue
            clef_x, _, _, _ = clef_b
            max_postpad = max(0.0, float(clef_x) - guard - float(pad_left_again))
            if float(r["x_postpad"]) > max_postpad:
                if _debug_enabled() and _debug_match(sheet_xml_path, r["staff_id"]):
                    _dbg(sheet_xml_path, r["staff_id"], f"[DBG] CLEF_CLAMP src={src} x={r['x_postpad']:.2f} -> {max_postpad:.2f} guard={guard:.2f}")
                r["x_postpad"] = float(max_postpad)

    # ------------------------------------------------------------
    # 5) System summary debug
    # ------------------------------------------------------------
    if _debug_enabled() and _debug_guides_summary_enabled():
        suspect_only = _debug_guides_suspect_only_enabled()

        # Group staff recs by sys
        by_sys: dict[str, list[dict]] = {}
        for r in staff_recs:
            by_sys.setdefault(r["sys_key"], []).append(r)

        for sk, recs in by_sys.items():
            # Skip indented systems entirely (we never touch them)
            if any(bool(r.get("is_indented")) for r in recs):
                continue

            raw_xs = [float(r.get("x_raw") or r["x_postpad"]) for r in recs]
            fin_xs = [float(r["x_postpad"]) for r in recs]

            if not raw_xs:
                continue

            tx = sys_target.get(sk)
            snap_list = snapped_by_sys.get(sk, [])
            snapped_count = len(snap_list)

            hcnt = clef_header_used_by_sys.get(sk, 0)
            fcnt = clef_fallback_used_by_sys.get(sk, 0)
            ncnt = clef_none_by_sys.get(sk, 0)

            if suspect_only and snapped_count == 0 and fcnt == 0:
                continue

            raw_min, raw_med, raw_max = min(raw_xs), _median(raw_xs), max(raw_xs)
            fin_min, fin_med, fin_max = min(fin_xs), _median(fin_xs), max(fin_xs)

            print(
                "[DBG] SYS_SUMMARY "
                f"sys={sk} target_x={(tx if tx is not None else 'None')} "
                f"raw[min/med/max]=[{raw_min:.2f},{raw_med:.2f},{raw_max:.2f}] "
                f"final[min/med/max]=[{fin_min:.2f},{fin_med:.2f},{fin_max:.2f}] "
                f"snapped={snapped_count} "
                f"clef(header/fallback/none)={hcnt}/{fcnt}/{ncnt} "
                f"snapped_staff={','.join(snap_list[:12])}{'...' if len(snap_list) > 12 else ''}",
                flush=True,
            )

    guides_px: list[tuple[float, float, float]] = [(r["x_postpad"], r["y_top"], r["y_bot"]) for r in staff_recs]

    # Build per-system descriptors (picture coords)
    systems_desc: list[dict] = []

    eps = max(6.0, 0.45 * expected_spacing) if expected_spacing > 0 else 10.0
    y_pad = max(8.0, 0.9 * expected_spacing) if expected_spacing > 0 else 14.0
    close_tol = max(14.0, 0.9 * expected_spacing) if expected_spacing > 0 else 20.0

    staff_x_by_sys_staff: dict[tuple[str, str], float] = {(r["sys_key"], r["staff_id"]): r["x_postpad"] for r in staff_recs}
    staff_right_by_sys_staff: dict[tuple[str, str], float | None] = {(r["sys_key"], r["staff_id"]): r["line_max_x"] for r in staff_recs}

    for page_idx, page in enumerate(pages):
        inters = page.find(".//sig/inters")
        systems = page.findall(".//system")
        if not systems:
            systems = page.findall("system")

        for sys_idx, system in enumerate(systems):
            sys_id = system.get("id") or ""
            sys_key = f"p{page_idx}_s{sys_idx}_id{sys_id}"

            spans = system_recs.get(sys_key, {}).get("staff_spans", [])
            if not spans:
                continue

            spans_sorted = sorted(spans, key=lambda t: t[0])
            top_y_top, _top_y_bot, top_staff_id = spans_sorted[0]
            sys_y_top = float(min(t[0] for t in spans_sorted))
            sys_y_bot = float(max(t[1] for t in spans_sorted))

            sys_start_x = staff_x_by_sys_staff.get((sys_key, top_staff_id))
            if sys_start_x is None:
                continue

            sys_right_x = None
            for (_, _, sid) in spans_sorted:
                mx = staff_right_by_sys_staff.get((sys_key, sid))
                if mx is None:
                    continue
                if sys_right_x is None or mx > sys_right_x:
                    sys_right_x = float(mx)

            xml_meas_count = _count_measures_in_system(system)

            xs_xml_used: list[float] = []
            if inters is not None:
                xs = _barline_xs_in_y_band(inters, sys_y_top - y_pad, sys_y_bot + y_pad)
                xs = _dedupe_sorted_xs(xs, eps=eps)
                xs = [x for x in xs if x > (float(sys_start_x) + eps)]
                if sys_right_x is not None:
                    xs = [x for x in xs if x <= (sys_right_x + 2.0 * close_tol)]
                xs_xml_used = xs

            systems_desc.append(
                {
                    "sys_key": sys_key,
                    "sys_id": sys_id,
                    "expected_spacing": float(expected_spacing),
                    "eps": float(eps),
                    "y_pad": float(y_pad),
                    "close_tol": float(close_tol),
                    "top_y_top": float(top_y_top),
                    "sys_y_top": float(sys_y_top),
                    "sys_y_bot": float(sys_y_bot),
                    "sys_start_x": float(sys_start_x),
                    "sys_right_x": float(sys_right_x) if sys_right_x is not None else None,
                    "xml_meas_count": int(xml_meas_count),
                    "xs_xml_used": xs_xml_used,
                }
            )

    return pic_w, pic_h, guides_px, staff_total, systems_desc


# --------------------------
# Missing staff guide fallback (unchanged)
# --------------------------
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

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

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
    base_x = float(rect.x0)
    base_y = float(rect.y0)

    pad_pdf = (_pad_left_px() / float(w)) * float(rect.width)

    extras = []
    for x_left_i, y_top_i, y_bot_i in staves:
        x_pdf = base_x + ((x_left_i / float(w)) * float(rect.width) - pad_pdf)
        y0_pdf = base_y + ((y_top_i / float(h)) * float(rect.height))
        y1_pdf = base_y + ((y_bot_i / float(h)) * float(rect.height))
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
            extras.append((max(base_x, x_pdf), y0_pdf, y1_pdf))

    return extras


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --------------------------
# Main annotation
# --------------------------
def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    print(
        "[DBG] startup "
        f"input_pdf={input_pdf} omr_path={omr_path} output_pdf={output_pdf} "
        f"DEBUG_GUIDES={os.getenv('DEBUG_GUIDES','')} DEBUG_MEASURES={os.getenv('DEBUG_MEASURES','')} "
        f"DEBUG_DRAW_BARS={os.getenv('DEBUG_DRAW_BARS','')} USE_CV_BARS={os.getenv('USE_CV_BARS','')} "
        f"CV_ZOOM={os.getenv('CV_ZOOM','')} PAD_LEFT_PX={os.getenv('PAD_LEFT_PX', str(PAD_LEFT_PX_DEFAULT))} "
        f"GUIDE_CONSENSUS={os.getenv('GUIDE_CONSENSUS','1')} GUIDE_CONSENSUS_P={os.getenv('GUIDE_CONSENSUS_P','0.10')} "
        f"GUIDE_CONSENSUS_THRESH_MULT={os.getenv('GUIDE_CONSENSUS_THRESH_MULT','0.60')} GUIDE_CLEF_CLAMP={os.getenv('GUIDE_CLEF_CLAMP','1')} "
        f"GUIDE_FORCE_NONINDENTED_X={os.getenv('GUIDE_FORCE_NONINDENTED_X','1')} GUIDE_FORCE_HARD={os.getenv('GUIDE_FORCE_HARD','0')} "
        f"GUIDE_TARGET_LEFT_FRAC={os.getenv('GUIDE_TARGET_LEFT_FRAC','0.75')} GUIDE_SNAP_EPS_PX={os.getenv('GUIDE_SNAP_EPS_PX','2.0')} "
        f"GUIDE_PAGE_FORCE={os.getenv('GUIDE_PAGE_FORCE','0')} "
        f"DEBUG_GUIDES_SUMMARY={os.getenv('DEBUG_GUIDES_SUMMARY','0')} DEBUG_GUIDES_SUSPECT_ONLY={os.getenv('DEBUG_GUIDES_SUSPECT_ONLY','0')} "
        f"OUTLIER_MZ_THRESHOLD={os.getenv('OUTLIER_MZ_THRESHOLD','3.5')} OUTLIER_ABS_FLOOR_PX={os.getenv('OUTLIER_ABS_FLOOR_PX','10.0')}",
        flush=True,
    )

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

            rect = page.rect
            base_x = float(rect.x0)
            base_y = float(rect.y0)

            pic_w, pic_h, guides_px, staff_total, systems_desc = _parse_sheet(z, sheet_xml_path)
            if pic_w <= 0 or pic_h <= 0:
                continue

            scale_x = float(rect.width) / float(pic_w)
            scale_y = float(rect.height) / float(pic_h)

            # Draw guides
            guides_pdf = []
            for (x_px, y0_px, y1_px) in guides_px:
                x_pdf = base_x + (x_px * scale_x)
                y0_pdf = base_y + (y0_px * scale_y)
                y1_pdf = base_y + (y1_px * scale_y)
                guides_pdf.append((x_pdf, y0_pdf, y1_pdf))
                page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=GUIDE_COLOR, width=GUIDE_WIDTH)

            # Font size
            interline_pdf = None
            if len(guides_pdf) >= 1:
                staff_h = abs(guides_pdf[0][2] - guides_pdf[0][1])
                interline_pdf = staff_h / 4.0 if staff_h > 0 else None
            if interline_pdf is None:
                interline_pdf = 16.0
            fontsize = _clamp(0.55 * interline_pdf, MEASURE_MIN_FONTSIZE, MEASURE_MAX_FONTSIZE)

            last_good_adv = 0

            for sysd in systems_desc:
                sys_key = sysd["sys_key"]
                sys_id = sysd["sys_id"]
                expected_spacing = float(sysd.get("expected_spacing") or 0.0)

                sys_start_x_pdf = base_x + (float(sysd["sys_start_x"]) * scale_x)
                sys_right_x_pdf = (
                    base_x + (float(sysd["sys_right_x"]) * scale_x) if sysd.get("sys_right_x") is not None else None
                )
                top_y_top_pdf = base_y + (float(sysd["top_y_top"]) * scale_y)
                sys_y_top_pdf = base_y + (float(sysd["sys_y_top"]) * scale_y)
                sys_y_bot_pdf = base_y + (float(sysd["sys_y_bot"]) * scale_y)

                eps_pdf = float(sysd.get("eps") or 10.0) * scale_x
                y_pad_pdf = float(sysd.get("y_pad") or 14.0) * scale_y
                close_tol_pdf_xml = float(sysd.get("close_tol") or 20.0) * scale_x

                xml_meas_count = int(sysd.get("xml_meas_count") or 0)

                method = ""
                reason = ""
                closing_present = False
                xs_used_pdf: list[float] = []

                if xml_meas_count > 0:
                    adv_raw = xml_meas_count
                    method = "xml_measures"
                    reason = "use_xml_measures"
                else:
                    if _use_cv_bars():
                        xs_pdf, close_tol_pdf = _cv_barlines_xs_in_band_pdf(
                            page,
                            y0_pdf=(sys_y_top_pdf - y_pad_pdf),
                            y1_pdf=(sys_y_bot_pdf + y_pad_pdf),
                        )
                        xs_pdf = [x for x in xs_pdf if x > (sys_start_x_pdf + eps_pdf)]
                        if sys_right_x_pdf is not None:
                            xs_pdf = [x for x in xs_pdf if x <= (sys_right_x_pdf + 2.0 * close_tol_pdf)]
                        xs_used_pdf = xs_pdf
                        method = "cv_bars"
                        close_tol_used = close_tol_pdf
                    else:
                        xs_pic = list(sysd.get("xs_xml_used") or [])
                        xs_pdf = [base_x + (float(x) * scale_x) for x in xs_pic]
                        xs_used_pdf = xs_pdf
                        method = "xml_bars"
                        close_tol_used = close_tol_pdf_xml

                    bar_count = len(xs_used_pdf)

                    if bar_count > 0:
                        cand0 = bar_count
                        cand1 = bar_count + 1

                        preferred = cand1
                        alt = cand0

                        if sys_right_x_pdf is not None:
                            rightmost = xs_used_pdf[-1]
                            if abs(rightmost - sys_right_x_pdf) <= close_tol_used or rightmost >= (
                                sys_right_x_pdf - close_tol_used
                            ):
                                closing_present = True
                                preferred = cand0
                                alt = cand1
                                reason = "bars"
                            else:
                                closing_present = False
                                preferred = cand1
                                alt = cand0
                                reason = "bars_plus_edge"
                        else:
                            closing_present = False
                            preferred = cand1
                            alt = cand0
                            reason = "no_right_edge"

                        if last_good_adv > 0:
                            if abs(preferred - last_good_adv) > abs(alt - last_good_adv):
                                preferred = alt
                                reason = reason + "_smooth"

                        adv_raw = preferred
                    else:
                        adv_raw = 0
                        reason = "no_bars"

                if adv_raw <= 0:
                    adv = last_good_adv if last_good_adv > 0 else 1
                    reason = reason + "_fallback"
                else:
                    adv = int(adv_raw)
                    last_good_adv = int(adv_raw)

                v_off = max(10.0, 0.90 * expected_spacing) if expected_spacing > 0 else 14.0
                x_text_pdf = sys_start_x_pdf + (
                    (max(6.0, 0.30 * expected_spacing) if expected_spacing > 0 else 8.0) * scale_x
                )
                y_text_pdf = max(base_y, top_y_top_pdf - (v_off * scale_y))
                page.insert_text((x_text_pdf, y_text_pdf), str(measure_no), fontsize=fontsize, color=MEASURE_TEXT_COLOR)

                _meas_dbg(
                    sheet_xml_path,
                    sys_key,
                    "[DBG] MEAS "
                    f"sys_key={sys_key} id={sys_id} "
                    f"start={measure_no} adv={adv} method={method} reason={reason} "
                    f"xml={xml_meas_count} bars={len(xs_used_pdf)} tail_xs={_tail_xs(xs_used_pdf, 3)} "
                    f"closing={closing_present} right={sys_right_x_pdf}",
                )

                if _bars_debug_enabled() and xs_used_pdf:
                    alpha = _bars_alpha()
                    for i, bx in enumerate(xs_used_pdf):
                        try:
                            page.draw_line(
                                (bx, sys_y_top_pdf),
                                (bx, sys_y_bot_pdf),
                                color=DEBUG_BAR_COLOR,
                                width=DEBUG_BAR_WIDTH,
                                stroke_opacity=alpha,
                            )
                            page.insert_text(
                                (bx + 1.5, max(base_y, sys_y_top_pdf - 6.0)),
                                str(measure_no + i),
                                fontsize=fontsize,
                                color=DEBUG_BAR_COLOR,
                                fill_opacity=alpha,
                            )
                        except Exception:
                            pass

                measure_no += int(max(1, adv))

            # Fallback staff detection if OMR missed some staves
            if staff_total > 0 and len(guides_pdf) < staff_total:
                extras = _fallback_missing_staff_guides(page, guides_pdf)
                if _debug_enabled():
                    print(
                        f"[DBG] page={page_index+1} sheet={sheet_xml_path} "
                        f"staff_total={staff_total} omr_guides={len(guides_pdf)} fallback_extras={len(extras)}",
                        flush=True,
                    )
                for (x_pdf, y0_pdf, y1_pdf) in extras:
                    page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=GUIDE_COLOR, width=GUIDE_WIDTH)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_guides_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
