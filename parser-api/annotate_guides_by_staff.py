#!/usr/bin/env python3
"""
Draw a short RED vertical guide for EACH staff on each page.

Targets:
- Y span: staff line 1 to staff line 5 (best effort; never silently skip).
- X: just left of clef/key area (robust against barlines/stems and staffline subtraction).
- Each staff independent (handles indents).

Key robustness changes (based on your misfires):
1) Remove VERTICAL lines (barlines + stems) before blob selection using morphology.
   This prevents anchoring on internal barlines / note stems. (OpenCV morphology line extraction) :contentReference[oaicite:1]{index=1}
2) Reduce aggressiveness of staffline subtraction so clefs survive.
3) Always compute x_staffstart + x_blob; clamp blob choice so it cannot jump deep into a measure.
4) If staff refinement fails, fall back to approximate 5 lines (so no "missing red line" cases).
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2


# -----------------------------
# Tunables (start here)
# -----------------------------
ZOOM = 2.0

# Staff line row detection
ROW_HIT_FRAC = 0.07
ROW_HIT_MIN = 120
MAX_GAP_BETWEEN_LINES = 18
MIN_LINES_PER_STAFF = 4

# Morphology for staff lines (horizontal)
HORIZ_KERNEL_DIV = 25
HORIZ_CLOSE_LEN = 9

# Morphology for vertical lines (barlines/stems)
VERT_KERNEL_DIV = 25
VERT_CLOSE_LEN = 9

# Refinement retries
REFINE_PAD_BASE = 28
REFINE_PAD_MULTS = (0.6, 1.0, 1.5)
REFINE_THRESH_FRACS = (0.42, 0.35, 0.28, 0.22)

SPACING_TOL_FRAC = 0.30
MIN_STAFF_SPACING = 5.0
MAX_STAFF_SPACING = 22.0

# X anchor via ink blobs (clef/key)
STAFF_BAND_MARGIN_FRAC = 0.22

# Make staffline subtraction gentle (so clef survives)
STAFFLINE_SUBTRACT_DILATE = 7

# Remove vertical lines strongly (barlines/stems)
VERTLINE_SUBTRACT_DILATE = 7

INK_OPEN_K = 3

CLEF_MIN_AREA_FRAC = 0.010
CLEF_MIN_H_FRAC = 0.55
CLEF_MIN_OVERLAP_FRAC = 0.60
CLEF_MAX_X_FRAC = 0.55
PAD_LEFT_OF_CLEF_PX = 6

# Fallback X (staffline continuity)
X_RUN_WINDOW = 27
X_RUN_MIN_FRAC = 0.65

# Clamp blob X relative to staff-start X (prevents jumping into measures)
# These are in "staff spaces" (computed per staff).
CLAMP_RIGHT_SPACES = 2.5   # allow blob up to ~2.5 staff spaces right of staff-start
CLAMP_LEFT_SPACES = 1.5    # allow blob up to ~1.5 staff spaces left of staff-start

# Debug toggles (keep False for normal)
DEBUG_DRAW_CLEF_BOX = False
DEBUG_MARK_FALLBACK = False


def render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def binarize_ink(gray: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def extract_horizontal_lines_mask(ink: np.ndarray) -> np.ndarray:
    w = ink.shape[1]
    kernel_len = max(30, w // HORIZ_KERNEL_DIV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    tmp = cv2.erode(ink, k, iterations=1)
    horiz = cv2.dilate(tmp, k, iterations=1)

    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_CLOSE_LEN, 1))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, ck, iterations=1)
    return horiz


def extract_vertical_lines_mask(ink: np.ndarray) -> np.ndarray:
    # Same idea as OpenCV’s “extract vertical lines” tutorial: erode+dilate with a tall kernel. :contentReference[oaicite:2]{index=2}
    h = ink.shape[0]
    kernel_len = max(30, h // VERT_KERNEL_DIV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    tmp = cv2.erode(ink, k, iterations=1)
    vert = cv2.dilate(tmp, k, iterations=1)

    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERT_CLOSE_LEN))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, ck, iterations=1)
    return vert


def find_staff_line_ys(horiz_mask: np.ndarray) -> list[int]:
    row_sums = np.sum(horiz_mask > 0, axis=1)
    W = horiz_mask.shape[1]
    threshold = max(ROW_HIT_MIN, int(W * ROW_HIT_FRAC))

    ys: list[int] = []
    in_run = False
    run_start = 0

    for y, cnt in enumerate(row_sums):
        if cnt >= threshold and not in_run:
            in_run = True
            run_start = y
        elif cnt < threshold and in_run:
            in_run = False
            run_end = y - 1
            ys.append((run_start + run_end) // 2)

    if in_run:
        run_end = len(row_sums) - 1
        ys.append((run_start + run_end) // 2)

    return ys


def group_lines_into_staves(line_ys: list[int]) -> list[list[int]]:
    line_ys = sorted(line_ys)
    staves: list[list[int]] = []
    cur: list[int] = []

    for y in line_ys:
        if not cur:
            cur = [y]
            continue
        if y - cur[-1] <= MAX_GAP_BETWEEN_LINES:
            cur.append(y)
        else:
            if len(cur) >= MIN_LINES_PER_STAFF:
                staves.append(cur)
            cur = [y]

    if cur and len(cur) >= MIN_LINES_PER_STAFF:
        staves.append(cur)

    return staves


def best_five_line_window(ys: list[int]) -> list[int]:
    ys = sorted(ys)
    if len(ys) <= 5:
        return ys

    best = ys[:5]
    best_score = float("inf")
    for i in range(0, len(ys) - 4):
        w = ys[i:i + 5]
        diffs = np.diff(w)
        med = float(np.median(diffs))
        var = float(np.sum(np.abs(diffs - med)))
        score = var - 0.002 * w[-1]
        if score < best_score:
            best_score = score
            best = w
    return list(best)


def _run_centers_from_row_sums(row_sums: np.ndarray, threshold: int) -> list[int]:
    centers: list[int] = []
    in_run = False
    start = 0
    for i, v in enumerate(row_sums):
        if v >= threshold and not in_run:
            in_run = True
            start = i
        elif v < threshold and in_run:
            in_run = False
            end = i - 1
            centers.append((start + end) // 2)
    if in_run:
        end = len(row_sums) - 1
        centers.append((start + end) // 2)
    return centers


def _spacing_ok(lines5: list[int]) -> bool:
    if len(lines5) != 5:
        return False
    diffs = np.diff(sorted(lines5)).astype(np.float32)
    spacing = float(np.median(diffs))
    if spacing < MIN_STAFF_SPACING or spacing > MAX_STAFF_SPACING:
        return False
    tol = max(2.0, spacing * SPACING_TOL_FRAC)
    if float(np.max(np.abs(diffs - spacing))) > tol:
        return False
    return True


def staff_spacing(lines5: list[int]) -> float:
    if len(lines5) != 5:
        return 10.0
    diffs = np.diff(sorted(lines5)).astype(np.float32)
    s = float(np.median(diffs))
    return float(np.clip(s, MIN_STAFF_SPACING, MAX_STAFF_SPACING))


def _fit_5_line_pattern(cands: list[int], approx5: list[int]) -> list[int] | None:
    if len(cands) < 5:
        return None
    approx5 = sorted(approx5)
    s = staff_spacing(approx5)
    tol = max(2.0, s * SPACING_TOL_FRAC)

    best = None
    best_score = float("inf")
    cands = sorted(cands)
    approx_top, approx_bot = approx5[0], approx5[-1]

    for top in cands:
        chosen = [top]
        score = 0.0
        ok = True
        for k in range(1, 5):
            target = top + k * s
            idx = int(np.argmin([abs(y - target) for y in cands]))
            yk = cands[idx]
            err = abs(yk - target)
            if err > tol:
                ok = False
                break
            chosen.append(yk)
            score += err

        if not ok:
            continue

        chosen = sorted(chosen)
        if not _spacing_ok(chosen):
            continue

        anchor_pen = abs(chosen[0] - approx_top) + abs(chosen[-1] - approx_bot)
        total = score + 0.15 * anchor_pen
        if total < best_score:
            best_score = total
            best = chosen

    return best


def refine_staff_5_lines(horiz_mask: np.ndarray, approx5: list[int]) -> list[int] | None:
    approx5 = sorted(approx5)
    approx_top, approx_bot = approx5[0], approx5[-1]
    H = horiz_mask.shape[0]
    staff_h = max(1, approx_bot - approx_top)

    for pad_mult in REFINE_PAD_MULTS:
        pad = max(REFINE_PAD_BASE, int(staff_h * pad_mult))
        y0 = max(0, approx_top - pad)
        y1 = min(H, approx_bot + pad)
        band = horiz_mask[y0:y1, :]
        row_sums = np.sum(band > 0, axis=1)
        mx = int(row_sums.max()) if row_sums.size else 0
        if mx <= 0:
            continue

        for frac in REFINE_THRESH_FRACS:
            thr = max(20, int(mx * frac))
            centers = _run_centers_from_row_sums(row_sums, thr)
            if len(centers) < 5:
                continue
            cands_abs = [c + y0 for c in centers]

            fitted = _fit_5_line_pattern(cands_abs, approx5)
            if fitted is not None:
                return fitted

            win = sorted(best_five_line_window(cands_abs))
            if len(win) == 5 and _spacing_ok(win):
                if abs(win[0] - approx_top) <= staff_h and abs(win[-1] - approx_bot) <= staff_h:
                    return win

    return None


def _first_continuous_run_x(mask_1d: np.ndarray, win: int, min_on: int) -> int | None:
    if mask_1d.size < win:
        return None
    kernel = np.ones(win, dtype=np.int32)
    run = np.convolve(mask_1d.astype(np.int32), kernel, mode="same")
    hits = np.where(run >= min_on)[0]
    return int(hits[0]) if hits.size else None


def staffline_start_x(horiz_mask: np.ndarray, staff5: list[int]) -> int | None:
    H = horiz_mask.shape[0]
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    hm = cv2.dilate(horiz_mask, dil_k, iterations=1)

    win = X_RUN_WINDOW
    min_on = int(win * X_RUN_MIN_FRAC)

    xs = []
    for y in staff5:
        a = max(0, y - 1)
        b = min(H, y + 2)
        strip = hm[a:b, :]
        col_any = (np.any(strip > 0, axis=0)).astype(np.uint8)
        x0 = _first_continuous_run_x(col_any, win=win, min_on=min_on)
        if x0 is not None:
            xs.append(x0)

    if len(xs) >= 3:
        xs = sorted(xs)[:3]
        return int(np.median(xs))
    if xs:
        return int(np.median(xs))
    return None


def clef_blob_anchor_x(
    ink_mask: np.ndarray,
    horiz_mask: np.ndarray,
    vert_mask: np.ndarray,
    top_y: int,
    bot_y: int,
    spacing_px: float
) -> tuple[int | None, tuple[int, int, int, int] | None]:
    H, W = ink_mask.shape[:2]
    staff_h = max(1, bot_y - top_y)
    margin = int(staff_h * STAFF_BAND_MARGIN_FRAC)
    y0 = max(0, top_y - margin)
    y1 = min(H, bot_y + margin)

    band_ink = ink_mask[y0:y1, :].copy()

    # 1) Remove vertical lines (barlines/stems) so they can't win blob selection. :contentReference[oaicite:3]{index=3}
    vd = cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERTLINE_SUBTRACT_DILATE))
    vband = cv2.dilate(vert_mask[y0:y1, :], vd, iterations=1)
    band_ink[vband > 0] = 0

    # 2) Gently remove staff lines (don’t erase clef)
    hd = cv2.getStructuringElement(cv2.MORPH_RECT, (STAFFLINE_SUBTRACT_DILATE, 1))
    hband = cv2.dilate(horiz_mask[y0:y1, :], hd, iterations=1)
    band_ink[hband > 0] = 0

    # 3) Speckle cleanup
    if INK_OPEN_K >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (INK_OPEN_K, INK_OPEN_K))
        band_ink = cv2.morphologyEx(band_ink, cv2.MORPH_OPEN, k, iterations=1)

    bin_img = (band_ink > 0).astype(np.uint8)
    if np.count_nonzero(bin_img) == 0:
        return None, None

    # Connected components with stats (x,y,w,h,area). :contentReference[oaicite:4]{index=4}
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    band_area = float(bin_img.shape[0] * bin_img.shape[1])
    min_area = int(max(30, band_area * CLEF_MIN_AREA_FRAC))
    max_x = int(W * CLEF_MAX_X_FRAC)

    best = None  # (x, -area, bbox)
    for i in range(1, num):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue
        if x > max_x:
            continue
        if h < int(staff_h * CLEF_MIN_H_FRAC):
            continue

        # vertical overlap with the staff region
        comp_top = y0 + y
        comp_bot = y0 + y + h
        overlap = max(0, min(comp_bot, bot_y) - max(comp_top, top_y))
        if overlap < int(staff_h * CLEF_MIN_OVERLAP_FRAC):
            continue

        bbox = (x, y0 + y, w, h)
        cand = (x, -area, bbox)
        if best is None or cand < best:
            best = cand

    if best is None:
        return None, None

    x_blob, _, bbox = best
    x_guide = max(0, int(x_blob - PAD_LEFT_OF_CLEF_PX))
    return x_guide, bbox


def choose_x(x_staffstart: int | None, x_blob: int | None, spacing_px: float) -> int | None:
    if x_staffstart is None and x_blob is None:
        return None
    if x_staffstart is None:
        return x_blob
    if x_blob is None:
        return x_staffstart

    right_tol = int(CLAMP_RIGHT_SPACES * spacing_px)
    left_tol = int(CLAMP_LEFT_SPACES * spacing_px)

    # If blob is too far right (deep into measures), ignore it.
    if x_blob > x_staffstart + right_tol:
        return x_staffstart
    # If blob is too far left (likely header/text/noise), ignore it.
    if x_blob < x_staffstart - left_tol:
        return x_staffstart

    return x_blob


def annotate_guides(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)

    for page in doc:
        img = render_page_to_bgr(page, ZOOM)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ink = binarize_ink(gray)
        horiz = extract_horizontal_lines_mask(ink)
        vert = extract_vertical_lines_mask(ink)

        line_ys = find_staff_line_ys(horiz)
        staves = group_lines_into_staves(line_ys)

        if not staves:
            page.draw_rect(fitz.Rect(5, 5, 25, 25), color=(1, 0, 0), width=2)
            continue

        for staff_lines in staves:
            approx5 = sorted(best_five_line_window(staff_lines))

            refined = None
            if len(approx5) >= 5:
                refined = refine_staff_5_lines(horiz, approx5)

            # Never skip completely: if refine fails, fall back to approx (prevents "missing line")
            if refined is None or len(refined) != 5:
                if len(approx5) < 5:
                    continue
                staff5 = approx5
                used_fallback = True
            else:
                staff5 = sorted(refined)
                used_fallback = False

            s_px = staff_spacing(staff5)
            top_y, bot_y = staff5[0], staff5[-1]

            x_staff = staffline_start_x(horiz, staff5)
            x_blob, bbox = clef_blob_anchor_x(ink, horiz, vert, top_y, bot_y, s_px)

            x_left = choose_x(x_staff, x_blob, s_px)
            if x_left is None:
                continue

            if DEBUG_MARK_FALLBACK and used_fallback:
                # small orange marker near top line if we fell back
                r = fitz.Rect((x_left)/ZOOM, (top_y-6)/ZOOM, (x_left+6)/ZOOM, (top_y)/ZOOM)
                page.draw_rect(r, color=(1, 0.5, 0), width=1)

            if DEBUG_DRAW_CLEF_BOX and bbox is not None:
                bx, by, bw, bh = bbox
                r = fitz.Rect(bx/ZOOM, by/ZOOM, (bx+bw)/ZOOM, (by+bh)/ZOOM)
                page.draw_rect(r, color=(0, 1, 1), width=0.8)

            # Draw the guide
            x_pdf = x_left / ZOOM
            y0_pdf = top_y / ZOOM
            y1_pdf = bot_y / ZOOM
            page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=(1, 0, 0), width=1.0)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: annotate_guides_by_staff.py <input.pdf> <output.pdf>")
        sys.exit(1)

    annotate_guides(sys.argv[1], sys.argv[2])
