#!/usr/bin/env python3
"""
Draw a short RED vertical guide for EACH staff on each page.

What we want:
- Vertical span: exactly from staff line 1 to staff line 5.
- Horizontal position: just LEFT of the clef/key-signature ink (pre-measure marker).
- Each staff independent: different indents handled naturally.

Approach (image-only):
1) Binarize (Otsu) for ink mask.
2) Extract horizontal staff lines with morphology (erode/dilate).
3) Detect staff-line rows via projection profile and group into staves.
4) Refine each staff to an accurate 5-line pattern (guarded; retries if suspicious).
5) For X: remove staff lines from ink mask, then use connected components inside the staff band
   to find the leftmost "big" symbol blob (clef/key). Place guide slightly left of that.
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2


# -----------------------------
# Tunables (start here when tuning)
# -----------------------------
ZOOM = 2.0

# Staff line row detection (adaptive)
ROW_HIT_FRAC = 0.07     # row must have at least this fraction of width "on" to count as staff line
ROW_HIT_MIN = 120       # absolute floor

MAX_GAP_BETWEEN_LINES = 18
MIN_LINES_PER_STAFF = 4

# Horizontal morphology for staff lines (from OpenCV line extraction style)
HORIZ_KERNEL_DIV = 25   # kernel_len = max(30, width // HORIZ_KERNEL_DIV)
HORIZ_CLOSE_LEN = 9     # reconnect small breaks in staff lines

# Refinement retries
REFINE_PAD_BASE = 28
REFINE_PAD_MULTS = (0.6, 1.0, 1.5)                 # try increasing local search bands
REFINE_THRESH_FRACS = (0.42, 0.35, 0.28, 0.22)     # try progressively gentler thresholds

SPACING_TOL_FRAC = 0.30   # allowed deviation relative to staff spacing
MIN_STAFF_SPACING = 5.0
MAX_STAFF_SPACING = 22.0

# X anchor via ink blobs (clef/key)
STAFF_BAND_MARGIN_FRAC = 0.25    # band extends beyond staff by this fraction of staff height
STAFFLINE_SUBTRACT_DILATE = 17   # dilate staffline mask before subtracting from ink
INK_OPEN_K = 3                   # cleanup speckles in ink mask

CLEF_MIN_AREA_FRAC = 0.010       # min component area relative to staff band area
CLEF_MIN_H_FRAC = 0.55           # component height relative to staff height
CLEF_MIN_OVERLAP_FRAC = 0.60     # vertical overlap with staff region
CLEF_MAX_X_FRAC = 0.55           # only consider blobs in left portion of page
PAD_LEFT_OF_CLEF_PX = 6          # place guide this many px left of the clef/key blob

# Fallback X (only if blob method fails)
X_RUN_WINDOW = 27
X_RUN_MIN_FRAC = 0.65

# Debug toggles
DEBUG_DRAW_CLEF_BOX = False      # draws a thin cyan box around chosen blob (useful when tuning)
DEBUG_MARK_BAD_STAFF = False     # draws a small magenta square if staff refinement fails


def render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # PyMuPDF gives RGB for alpha=False; OpenCV expects BGR
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
    # ink -> 255, background -> 0
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def extract_horizontal_lines_mask(ink: np.ndarray) -> np.ndarray:
    w = ink.shape[1]
    kernel_len = max(30, w // HORIZ_KERNEL_DIV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    tmp = cv2.erode(ink, horiz_kernel, iterations=1)
    horiz = cv2.dilate(tmp, horiz_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_CLOSE_LEN, 1))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return horiz


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
    return best


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


def _fit_5_line_pattern(cands: list[int], approx5: list[int]) -> list[int] | None:
    if len(cands) < 5:
        return None
    approx5 = sorted(approx5)
    diffs = np.diff(approx5) if len(approx5) >= 2 else np.array([10.0])
    spacing = float(np.median(diffs)) if diffs.size else 10.0
    spacing = float(np.clip(spacing, MIN_STAFF_SPACING, MAX_STAFF_SPACING))
    tol = max(2.0, spacing * SPACING_TOL_FRAC)

    best = None
    best_score = float("inf")

    cands = sorted(cands)
    approx_top, approx_bot = approx5[0], approx5[-1]

    for top in cands:
        chosen = [top]
        score = 0.0
        ok = True
        for k in range(1, 5):
            target = top + k * spacing
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
    H, W = horiz_mask.shape[:2]
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


def fallback_staffline_start_x(horiz_mask: np.ndarray, staff5: list[int]) -> int | None:
    H, W = horiz_mask.shape[:2]
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
    top_y: int,
    bot_y: int
) -> tuple[int | None, tuple[int, int, int, int] | None]:
    H, W = ink_mask.shape[:2]
    staff_h = max(1, bot_y - top_y)
    margin = int(staff_h * STAFF_BAND_MARGIN_FRAC)
    y0 = max(0, top_y - margin)
    y1 = min(H, bot_y + margin)

    band_ink = ink_mask[y0:y1, :].copy()

    dil = cv2.getStructuringElement(cv2.MORPH_RECT, (STAFFLINE_SUBTRACT_DILATE, 1))
    staff_d = cv2.dilate(horiz_mask[y0:y1, :], dil, iterations=1)
    band_ink[staff_d > 0] = 0

    if INK_OPEN_K >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (INK_OPEN_K, INK_OPEN_K))
        band_ink = cv2.morphologyEx(band_ink, cv2.MORPH_OPEN, k, iterations=1)

    bin_img = (band_ink > 0).astype(np.uint8)
    if np.count_nonzero(bin_img) == 0:
        return None, None

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


def annotate_guides(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)

    for page in doc:
        img = render_page_to_bgr(page, ZOOM)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ink = binarize_ink(gray)
        horiz = extract_horizontal_lines_mask(ink)

        line_ys = find_staff_line_ys(horiz)
        staves = group_lines_into_staves(line_ys)

        if not staves:
            page.draw_rect(fitz.Rect(5, 5, 25, 25), color=(1, 0, 0), width=2)
            continue

        for staff_lines in staves:
            approx = sorted(best_five_line_window(staff_lines))
            if len(approx) < 2:
                continue

            refined = refine_staff_5_lines(horiz, approx)
            if refined is None or len(refined) != 5:
                if DEBUG_MARK_BAD_STAFF:
                    x_pdf = 8 / ZOOM
                    y_pdf = approx[0] / ZOOM
                    page.draw_rect(
                        fitz.Rect(x_pdf, y_pdf, x_pdf + 6/ZOOM, y_pdf + 6/ZOOM),
                        color=(1, 0, 1),
                        width=1
                    )
                continue

            staff5 = sorted(refined)
            top_y, bot_y = staff5[0], staff5[-1]

            x_left, bbox = clef_blob_anchor_x(ink, horiz, top_y, bot_y)
            if x_left is None:
                x_left = fallback_staffline_start_x(horiz, staff5)

            if x_left is None:
                continue

            if DEBUG_DRAW_CLEF_BOX and bbox is not None:
                bx, by, bw, bh = bbox
                r = fitz.Rect(bx/ZOOM, by/ZOOM, (bx+bw)/ZOOM, (by+bh)/ZOOM)
                page.draw_rect(r, color=(0, 1, 1), width=0.8)

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
