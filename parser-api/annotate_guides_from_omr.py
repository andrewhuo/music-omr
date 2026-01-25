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

# Measure number rendering (system-start labels)
MEASURE_TEXT_COLOR = (0, 0, 0)  # black
MEASURE_MIN_FONTSIZE = 7.0
MEASURE_MAX_FONTSIZE = 12.0

# Outlier forcing
OUTLIER_MZ_THRESHOLD = 3.5  # modified z-score threshold
OUTLIER_ABS_FLOOR_PX = 10.0  # minimum MAD floor to avoid divide-by-near-zero

# DEBUG: barline overlay (testing only)
DEBUG_BAR_COLOR = (0, 0, 1)  # blue
DEBUG_BAR_WIDTH = 0.9

_SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


# --------------------------
# Env toggles
# --------------------------
def _debug_enabled() -> bool:
    v = os.getenv("DEBUG_GUIDES", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


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
    v = os.getenv("DEBUG_GUIDES_DUMP_INTERS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _meas_debug_enabled() -> bool:
    v = os.getenv("DEBUG_MEASURES", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


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
    v = os.getenv("DEBUG_DRAW_BARS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _bars_alpha() -> float:
    try:
        return float(os.getenv("DEBUG_BARS_ALPHA", "0.85").strip())
    except Exception:
        return 0.85


def _use_cv_bars() -> bool:
    v = os.getenv("USE_CV_BARS", "").strip()
    return v in ("1", "true", "True", "yes", "YES")


def _cv_zoom() -> float:
    try:
        z = float(os.getenv("CV_ZOOM", "2.5").strip())
        return z if z > 0 else 2.5
    except Exception:
        return 2.5


def _meas_sane_max() -> int:
    try:
        return int(os.getenv("MEAS_SANE_MAX", "32").strip())
    except Exception:
        return 32


def _meas_disagree_warn() -> int:
    try:
        return int(os.getenv("MEAS_DISAGREE_WARN", "2").strip())
    except Exception:
        return 2


def _allow_override_on_strong_disagree() -> bool:
    """
    If XML measures are present+sane, we still prefer them, but we *may* override
    when BOTH other independent signals agree against XML by >= warn.
    """
    v = os.getenv("ALLOW_XML_OVERRIDE", "1").strip()
    return v in ("1", "true", "True", "yes", "YES")


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


def _tag_looks_like_clef(el: ET.Element) -> bool:
    t = (el.tag or "").lower()
    if t == "clef" or t.endswith("clef"):
        return True
    for k in ("shape", "type", "kind", "name", "family"):
        v = el.get(k)
        if v and "clef" in v.lower():
            return True
    return False


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
    """
    Collect barline-like x positions by y-overlap, not staff linkage.
    Uses median p1/p2 when available; otherwise falls back to bounds y/h.
    """
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

        # Prefer the "median" vertical extent when it exists
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

        # Fallback: bounds vertical extent
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --------------------------
# CV barline detection (PDF space)
# --------------------------
def _cv_barlines_xs_in_band_pdf(page: fitz.Page, y0_pdf: float, y1_pdf: float) -> tuple[list[float], float]:
    """
    Detect barline x positions using image morphology in a PDF y-band.
    Returns: (xs_pdf, close_tol_pdf)
    """
    rect = page.rect
    y0_pdf = max(0.0, min(float(rect.height), float(y0_pdf)))
    y1_pdf = max(0.0, min(float(rect.height), float(y1_pdf)))
    if y1_pdf <= y0_pdf + 4.0:
        return ([], max(14.0, 0.02 * float(rect.width)))

    zoom = _cv_zoom()
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_y_px = float(pix.height) / float(rect.height)
    scale_x_px = float(pix.width) / float(rect.width)

    y0_px = int(max(0, min(pix.height - 1, round(y0_pdf * scale_y_px))))
    y1_px = int(max(0, min(pix.height, round(y1_pdf * scale_y_px))))
    if y1_px <= y0_px + 6:
        return ([], max(14.0, 0.02 * float(rect.width)))

    band = gray[y0_px:y1_px, :]

    # Adaptive threshold
    thr = cv2.adaptiveThreshold(band, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    # Vertical morphology (standard approach) :contentReference[oaicite:2]{index=2}
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
    xs_pdf = [float(x / scale_x_px) for x in xs_px]

    close_tol_pdf = max(14.0, 0.10 * (float(y1_pdf - y0_pdf) / 4.0))
    return (xs_pdf, close_tol_pdf)


# --------------------------
# Sheet parsing (OMR XML)
# --------------------------
def _parse_sheet(z: zipfile.ZipFile, sheet_xml_path: str):
    """
    Returns:
      pic_w, pic_h: Audiveris picture size
      guides_px: list of (x_left_px, y_top_px, y_bot_px) in picture pixels
      staff_total: how many <staff> nodes exist (target guide count)
      systems_desc: list of dict describing each system in picture coords, including:
         sys_key, sys_id, expected_spacing,
         sys_y_top, sys_y_bot, top_y_top,
         sys_start_x, sys_right_x,
         xml_meas_count,
         xs_xml_used, close_tol, eps, y_pad
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
        return pic_w, pic_h, [], 0, []

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

            if sys_key not in system_recs:
                system_recs[sys_key] = {
                    "sys_key": sys_key,
                    "sys_id": sys_id,
                    "page_idx": page_idx,
                    "sys_idx": sys_idx,
                    "indented": system.get("indented"),
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

                clef_b = _clef_bounds_from_header(inter_by_id, staff)
                if clef_b is None:
                    clef_b = _clef_bounds_fallback(inters, staff_id, y_top, y_bot, header_start)

                x_left = _safe_float(staff.get("left"))
                line_min_x = float(min(all_line_xmins)) if all_line_xmins else None
                line_max_x = float(max(all_line_xmaxs)) if all_line_xmaxs else None

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
                            _dbg(sheet_xml_path, staff_id, f"[DBG] BIG_HEADER_GUARD staff={staff_id} x={x_left:.2f} -> {max_x2:.2f}")
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

                _dump_inters_near_staff(sheet_xml_path, staff_id, inters, y_top, y_bot, x_postpad, x_window=140.0)

                if _debug_enabled() and _debug_match(sheet_xml_path, staff_id):
                    _dbg(
                        sheet_xml_path,
                        staff_id,
                        f"[DBG] sheet={sheet_xml_path} sys_key={sys_key} staff={staff_id} indented={system.get('indented')} "
                        f"header_start={header_start} line_min_x={line_min_x} line_max_x={line_max_x} expected_spacing={expected_spacing:.2f} "
                        f"clef_b={'present' if clef_b is not None else None} x_before={x_before:.2f} x_after={float(x_left):.2f} x_postpad={x_postpad:.2f} "
                        f"y_top={y_top:.1f} y_bot={y_bot:.1f}",
                    )

                staff_recs.append(
                    {
                        "sys_key": sys_key,
                        "sys_id": sys_id,
                        "staff_id": staff_id,
                        "indented": system.get("indented"),
                        "x_postpad": float(x_postpad),
                        "y_top": float(y_top),
                        "y_bot": float(y_bot),
                        "line_max_x": float(line_max_x) if line_max_x is not None else None,
                    }
                )
                system_recs[sys_key]["staff_spans"].append((float(y_top), float(y_bot), staff_id))

    # Outlier forcing for guide x (exclude indented)
    non_indented_xs = [
        r["x_postpad"] for r in staff_recs if not (r["indented"] not in (None, "", "false", "False", "0"))
    ]
    if len(non_indented_xs) >= 6:
        med = _median(non_indented_xs)
        mad = max(_mad(non_indented_xs, med), OUTLIER_ABS_FLOOR_PX)

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
                    r["x_postpad"] = float(forced_avg)

    guides_px: list[tuple[float, float, float]] = [(r["x_postpad"], r["y_top"], r["y_bot"]) for r in staff_recs]

    # Build per-system descriptors (so measure logic can happen later with CV + continuity)
    systems_desc: list[dict] = []

    # Picture-coords tolerances
    eps = max(6.0, 0.45 * expected_spacing) if expected_spacing > 0 else 10.0
    y_pad = max(8.0, 0.9 * expected_spacing) if expected_spacing > 0 else 14.0
    close_tol = max(14.0, 0.9 * expected_spacing) if expected_spacing > 0 else 20.0

    staff_x_by_sys_staff: dict[tuple[str, str], float] = {
        (r["sys_key"], r["staff_id"]): r["x_postpad"] for r in staff_recs
    }
    staff_right_by_sys_staff: dict[tuple[str, str], float | None] = {
        (r["sys_key"], r["staff_id"]): r["line_max_x"] for r in staff_recs
    }

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

            # XML barline xs (picture coords) for fallback/check
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
# Guide fallback (unchanged)
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


# --------------------------
# Measure helpers (integrated checks)
# --------------------------
def _count_from_bars(
    xs: list[float],
    sys_right_x: float | None,
    close_tol: float,
) -> tuple[int, bool, str]:
    """
    Return (count, closing_present, reason)
    """
    if not xs:
        return (0, False, "no_bars")

    bar_count = len(xs)
    # default: bars + system-edge
    preferred = bar_count + 1
    closing_present = False
    reason = "bars_plus_edge"

    if sys_right_x is not None:
        rightmost = xs[-1]
        if abs(rightmost - sys_right_x) <= close_tol or rightmost >= (sys_right_x - close_tol):
            # last barline is closing bar
            closing_present = True
            preferred = bar_count
            reason = "bars"
        else:
            closing_present = False
            preferred = bar_count + 1
            reason = "bars_plus_edge"
    else:
        reason = "no_right_edge"
        preferred = bar_count + 1

    return (int(preferred), bool(closing_present), reason)


def _is_sane_count(count: int, last_good_adv: int | None) -> bool:
    if count <= 0:
        return False
    if count > _meas_sane_max():
        return False
    # avoid wild spikes relative to neighbors
    if last_good_adv and last_good_adv > 0:
        if count > int(max(_meas_sane_max(), 2.2 * last_good_adv)):
            return False
    return True


def _absdiff(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return abs(int(a) - int(b))


def _choose_adv(
    xml_count: int | None,
    cv_count: int | None,
    xmlbar_count: int | None,
    last_good_adv: int,
) -> tuple[int, str, str, dict]:
    """
    Decide adv (measures in system) with cross-checking.
    Returns: (adv, chosen_method, status, debug_dict)
    """
    warn = _meas_disagree_warn()

    xml_sane = (xml_count is not None) and _is_sane_count(int(xml_count), last_good_adv)
    cv_sane = (cv_count is not None) and _is_sane_count(int(cv_count), last_good_adv)
    xmlbar_sane = (xmlbar_count is not None) and _is_sane_count(int(xmlbar_count), last_good_adv)

    d_xml_cv = _absdiff(xml_count, cv_count)
    d_xml_xmlbar = _absdiff(xml_count, xmlbar_count)
    d_cv_xmlbar = _absdiff(cv_count, xmlbar_count)

    status = "OK"
    chosen_method = "fallback"
    chosen = 0
    reason = ""

    # Strict Step 3: trust XML measures when present and sane
    if xml_sane:
        chosen = int(xml_count)
        chosen_method = "xml_measures"
        reason = "xml_sane_prefer"

        # If we have other sane signals and they disagree by >= warn, mark suspect
        strong_disagree = False
        if cv_sane and d_xml_cv is not None and d_xml_cv >= warn:
            strong_disagree = True
        if xmlbar_sane and d_xml_xmlbar is not None and d_xml_xmlbar >= warn:
            strong_disagree = True

        if strong_disagree:
            status = "SUSPECT"

            # Optional override only when BOTH independent signals agree against XML
            if _allow_override_on_strong_disagree():
                # Case 1: cv and xmlbar agree with each other and disagree with xml
                if cv_sane and xmlbar_sane and (d_cv_xmlbar == 0) and (d_xml_cv is not None) and d_xml_cv >= warn:
                    chosen = int(cv_count)
                    chosen_method = "override_cv_xmlbar_agree"
                    reason = "both_nonxml_agree"
                    status = "OVERRIDE"

                # Case 2: only CV available and continuity strongly favors it
                elif cv_sane and last_good_adv > 0 and d_xml_cv is not None and d_xml_cv >= warn:
                    if abs(int(cv_count) - last_good_adv) < abs(int(xml_count) - last_good_adv):
                        chosen = int(cv_count)
                        chosen_method = "override_cv_continuity"
                        reason = "cv_matches_prev"
                        status = "OVERRIDE"

    else:
        # XML missing/insane: fall back to CV then xmlbar then continuity
        if cv_sane:
            chosen = int(cv_count)
            chosen_method = "cv_bars"
            reason = "xml_missing_or_insane"
        elif xmlbar_sane:
            chosen = int(xmlbar_count)
            chosen_method = "xml_bars"
            reason = "xml_missing_or_insane"
        else:
            chosen = int(last_good_adv) if last_good_adv > 0 else 1
            chosen_method = "continuity"
            reason = "no_sane_sources"
            status = "SUSPECT"

    if chosen <= 0:
        chosen = int(last_good_adv) if last_good_adv > 0 else 1
        chosen_method = "continuity"
        reason = reason + "_forced"
        status = "SUSPECT"

    dbg = {
        "xml": int(xml_count) if xml_count is not None else None,
        "cv": int(cv_count) if cv_count is not None else None,
        "xmlbar": int(xmlbar_count) if xmlbar_count is not None else None,
        "xml_sane": bool(xml_sane),
        "cv_sane": bool(cv_sane),
        "xmlbar_sane": bool(xmlbar_sane),
        "d_xml_cv": d_xml_cv,
        "d_xml_xmlbar": d_xml_xmlbar,
        "d_cv_xmlbar": d_cv_xmlbar,
        "reason": reason,
    }
    return (int(chosen), chosen_method, status, dbg)


# --------------------------
# Main annotation
# --------------------------
def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    # Startup log always (so you can prove env got through)
    print(
        "[DBG] startup "
        f"input_pdf={input_pdf} omr_path={omr_path} output_pdf={output_pdf} "
        f"DEBUG_MEASURES={os.getenv('DEBUG_MEASURES','')} DEBUG_DRAW_BARS={os.getenv('DEBUG_DRAW_BARS','')} "
        f"USE_CV_BARS={os.getenv('USE_CV_BARS','')} CV_ZOOM={os.getenv('CV_ZOOM','')} "
        f"MEAS_SANE_MAX={os.getenv('MEAS_SANE_MAX','')} MEAS_DISAGREE_WARN={os.getenv('MEAS_DISAGREE_WARN','')} "
        f"ALLOW_XML_OVERRIDE={os.getenv('ALLOW_XML_OVERRIDE','')}",
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

            pic_w, pic_h, guides_px, staff_total, systems_desc = _parse_sheet(z, sheet_xml_path)
            if pic_w <= 0 or pic_h <= 0:
                continue

            rect = page.rect
            scale_x = rect.width / pic_w
            scale_y = rect.height / pic_h

            # Draw guides
            guides_pdf = []
            for (x_px, y0_px, y1_px) in guides_px:
                x_pdf = x_px * scale_x
                y0_pdf = y0_px * scale_y
                y1_pdf = y1_px * scale_y
                guides_pdf.append((x_pdf, y0_pdf, y1_pdf))
                page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=GUIDE_COLOR, width=GUIDE_WIDTH)

            # Font size (shared by measure labels + debug bar labels)
            interline_pdf = None
            if len(guides_pdf) >= 1:
                staff_h = abs(guides_pdf[0][2] - guides_pdf[0][1])
                interline_pdf = staff_h / 4.0 if staff_h > 0 else None
            if interline_pdf is None:
                interline_pdf = 16.0
            fontsize = _clamp(0.55 * interline_pdf, MEASURE_MIN_FONTSIZE, MEASURE_MAX_FONTSIZE)

            last_good_adv = 0

            for sys in systems_desc:
                sys_key = sys["sys_key"]
                sys_id = sys["sys_id"]
                expected_spacing = float(sys.get("expected_spacing") or 0.0)

                # Convert system geometry to PDF coords
                sys_start_x_pdf = float(sys["sys_start_x"]) * scale_x
                sys_right_x_pdf = (float(sys["sys_right_x"]) * scale_x) if sys.get("sys_right_x") is not None else None
                top_y_top_pdf = float(sys["top_y_top"]) * scale_y
                sys_y_top_pdf = float(sys["sys_y_top"]) * scale_y
                sys_y_bot_pdf = float(sys["sys_y_bot"]) * scale_y

                eps_pdf = float(sys.get("eps") or 10.0) * scale_x
                y_pad_pdf = float(sys.get("y_pad") or 14.0) * scale_y
                close_tol_pdf_xml = float(sys.get("close_tol") or 20.0) * scale_x

                xml_meas_count = int(sys.get("xml_meas_count") or 0)

                # --- Candidate 1: XML measures (count only)
                xml_count = xml_meas_count if xml_meas_count > 0 else None

                # --- Candidate 2: XML bar inters (convert stored pic-space xs to PDF)
                xs_xml_pdf: list[float] = [float(x) * scale_x for x in (sys.get("xs_xml_used") or [])]
                xs_xml_pdf = [x for x in xs_xml_pdf if x > (sys_start_x_pdf + eps_pdf)]
                if sys_right_x_pdf is not None:
                    xs_xml_pdf = [x for x in xs_xml_pdf if x <= (sys_right_x_pdf + 2.0 * close_tol_pdf_xml)]
                xmlbar_count, xmlbar_closing, xmlbar_reason = _count_from_bars(xs_xml_pdf, sys_right_x_pdf, close_tol_pdf_xml)
                xmlbar_count = xmlbar_count if xs_xml_pdf else 0
                xmlbar_count = xmlbar_count if xmlbar_count > 0 else 0

                # --- Candidate 3: CV bars (optional, but recommended as checker)
                xs_cv_pdf: list[float] = []
                cv_close_tol_used = None
                cv_count = None
                cv_closing = False
                cv_reason = ""
                if _use_cv_bars():
                    xs_cv_pdf, cv_close_tol_used = _cv_barlines_xs_in_band_pdf(
                        page,
                        y0_pdf=(sys_y_top_pdf - y_pad_pdf),
                        y1_pdf=(sys_y_bot_pdf + y_pad_pdf),
                    )
                    xs_cv_pdf = [x for x in xs_cv_pdf if x > (sys_start_x_pdf + eps_pdf)]
                    if sys_right_x_pdf is not None and cv_close_tol_used is not None:
                        xs_cv_pdf = [x for x in xs_cv_pdf if x <= (sys_right_x_pdf + 2.0 * float(cv_close_tol_used))]
                    cv_count_raw, cv_closing, cv_reason = _count_from_bars(
                        xs_cv_pdf,
                        sys_right_x_pdf,
                        float(cv_close_tol_used) if cv_close_tol_used is not None else close_tol_pdf_xml,
                    )
                    cv_count = cv_count_raw if xs_cv_pdf else 0
                    if cv_count <= 0:
                        cv_count = None

                xmlbar_count = xmlbar_count if xmlbar_count > 0 else None

                # Decide adv with cross-checks
                adv, method, status, dbg = _choose_adv(
                    xml_count=xml_count,
                    cv_count=cv_count,
                    xmlbar_count=xmlbar_count,
                    last_good_adv=last_good_adv,
                )
                if adv > 0:
                    last_good_adv = adv

                # Place system-start label (unchanged behavior)
                v_off = max(10.0, 0.90 * expected_spacing) if expected_spacing > 0 else 14.0
                x_text_pdf = sys_start_x_pdf + ((max(6.0, 0.30 * expected_spacing) if expected_spacing > 0 else 8.0) * scale_x)
                y_text_pdf = max(0.0, top_y_top_pdf - (v_off * scale_y))
                page.insert_text((x_text_pdf, y_text_pdf), str(measure_no), fontsize=fontsize, color=MEASURE_TEXT_COLOR)

                # Debug log (one line per system)
                _meas_dbg(
                    sheet_xml_path,
                    sys_key,
                    "[DBG] MEAS "
                    f"sys_key={sys_key} id={sys_id} start={measure_no} adv={adv} chosen={method} status={status} "
                    f"xml={dbg['xml']} xml_sane={dbg['xml_sane']} "
                    f"cv={dbg['cv']} cv_sane={dbg['cv_sane']} "
                    f"xmlbar={dbg['xmlbar']} xmlbar_sane={dbg['xmlbar_sane']} "
                    f"d_xml_cv={dbg['d_xml_cv']} d_xml_xmlbar={dbg['d_xml_xmlbar']} d_cv_xmlbar={dbg['d_cv_xmlbar']} "
                    f"right={sys_right_x_pdf} "
                    f"tail_xml={_tail_xs(xs_xml_pdf, 3)} tail_cv={_tail_xs(xs_cv_pdf, 3)} "
                    f"xmlbar_reason={xmlbar_reason} cv_reason={cv_reason} reason={dbg['reason']}",
                )

                # Debug overlay: draw/label barlines (CV + XML) when enabled
                if _bars_debug_enabled():
                    alpha = _bars_alpha()

                    # Draw XML bars
                    for i, bx in enumerate(xs_xml_pdf):
                        try:
                            page.draw_line(
                                (bx, sys_y_top_pdf),
                                (bx, sys_y_bot_pdf),
                                color=DEBUG_BAR_COLOR,
                                width=DEBUG_BAR_WIDTH,
                                stroke_opacity=alpha,
                            )
                            page.insert_text(
                                (bx + 1.5, max(0.0, sys_y_top_pdf - 6.0)),
                                f"X{i+1}",
                                fontsize=max(6.0, fontsize * 0.9),
                                color=DEBUG_BAR_COLOR,
                                fill_opacity=alpha,
                            )
                        except Exception:
                            pass

                    # Draw CV bars (slightly offset label so you can distinguish)
                    for i, bx in enumerate(xs_cv_pdf):
                        try:
                            page.draw_line(
                                (bx, sys_y_top_pdf),
                                (bx, sys_y_bot_pdf),
                                color=DEBUG_BAR_COLOR,
                                width=DEBUG_BAR_WIDTH,
                                stroke_opacity=alpha,
                            )
                            page.insert_text(
                                (bx + 1.5, max(0.0, sys_y_top_pdf - 14.0)),
                                f"C{i+1}",
                                fontsize=max(6.0, fontsize * 0.9),
                                color=DEBUG_BAR_COLOR,
                                fill_opacity=alpha,
                            )
                        except Exception:
                            pass

                measure_no += int(max(1, adv))

            # If OMR missed some staves, try fallback guide detection (unchanged)
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
