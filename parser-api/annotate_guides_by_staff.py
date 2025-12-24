#!/usr/bin/env python3
"""
Draw a short RED vertical guide for EACH staff on each page.

Goals:
- Each guide spans from the 1st staff line to the 5th staff line (robustly).
- X position is the left start of the staff lines for that staff (handles indents).
- Each staff is processed independently (no shared/global X).

Fixes:
- Prevent "short outliers" by refining staff lines using an expected 5-line pattern.
- Prevent "too far right" by scanning for the first continuous staff-line run.
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2

# -----------------------------
# Tunables
# -----------------------------
ZOOM = 2.0

# Staff line row detection (adaptive)
ROW_HIT_FRAC = 0.07
ROW_HIT_MIN = 120
MAX_GAP_BETWEEN_LINES = 18
MIN_LINES_PER_STAFF = 4

# Horizontal morphology
HORIZ_KERNEL_DIV = 25          # kernel_len = max(30, width // HORIZ_KERNEL_DIV)
HORIZ_CLOSE_LEN = 7            # reconnect small breaks

# Refinement
REFINE_PAD_PX = 28
REFINE_LOCAL_THRESH_FRAC = 0.38

# Pattern matching tolerance (relative to staff spacing)
SPACING_TOL_FRAC = 0.35

# Left-edge scan settings
X_RUN_WINDOW = 25              # pixels
X_RUN_MIN_FRAC = 0.60          # window must be at least this "on"
X_NUDGE_LEFT_PX = 4            # tiny nudge left to counter residual right bias
X_MIN = 0


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


def extract_horizontal_lines_mask(gray: np.ndarray) -> np.ndarray:
    # Dark lines -> white mask
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract horizontal lines
    kernel_len = max(30, gray.shape[1] // HORIZ_KERNEL_DIV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    tmp = cv2.erode(bw, horiz_kernel, iterations=1)
    horiz = cv2.dilate(tmp, horiz_kernel, iterations=1)

    # Reconnect small breaks (important for left-edge detection)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_CLOSE_LEN, 1))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return horiz


def find_staff_line_ys(horiz_mask: np.ndarray) -> list[int]:
    row_sums = np.sum(horiz_mask > 0, axis=1)
    W = horiz_mask.shape[1]
    row_hit_threshold = max(ROW_HIT_MIN, int(W * ROW_HIT_FRAC))

    ys: list[int] = []
    in_run = False
    run_start = 0

    for y, count in enumerate(row_sums):
        if count >= row_hit_threshold and not in_run:
            in_run = True
            run_start = y
        elif count < row_hit_threshold and in_run:
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
        diffs = [w[j + 1] - w[j] for j in range(4)]
        med = float(np.median(diffs))
        var = sum(abs(d - med) for d in diffs)

        # Slight preference lower on page
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


def refine_staff_5_lines(horiz_mask: np.ndarray, approx5: list[int]) -> list[int]:
    """
    Refine to the true 1st..5th staff line set.
    Uses a local band and fits a 5-line pattern based on the approx spacing.
    """
    approx5 = sorted(approx5)
    approx_top, approx_bot = approx5[0], approx5[-1]
    H, W = horiz_mask.shape[:2]

    diffs = np.diff(approx5) if len(approx5) >= 2 else np.array([10])
    spacing = float(np.median(diffs)) if diffs.size else 10.0
    spacing = max(6.0, spacing)

    staff_h = max(1, approx_bot - approx_top)
    pad = max(REFINE_PAD_PX, int(staff_h * 0.6))

    y0 = max(0, approx_top - pad)
    y1 = min(H, approx_bot + pad)
    band = horiz_mask[y0:y1, :]

    row_sums = np.sum(band > 0, axis=1)
    mx = int(row_sums.max()) if row_sums.size else 0
    if mx <= 0:
        return approx5

    local_thresh = max(20, int(mx * REFINE_LOCAL_THRESH_FRAC))
    centers = _run_centers_from_row_sums(row_sums, local_thresh)
    if len(centers) < 5:
        return approx5

    centers_abs = sorted([c + y0 for c in centers])

    # Fit 5-line pattern: pick a top, then seek 4 more lines near top + k*spacing
    tol = max(2.0, spacing * SPACING_TOL_FRAC)

    best = None
    best_score = float("inf")

    for top in centers_abs:
        chosen = [top]
        score = 0.0
        ok = True

        for k in range(1, 5):
            target = top + k * spacing
            # nearest candidate to target
            idx = int(np.argmin([abs(y - target) for y in centers_abs]))
            yk = centers_abs[idx]
            err = abs(yk - target)
            if err > tol:
                ok = False
                break
            chosen.append(yk)
            score += err

        if ok:
            chosen = sorted(chosen)
            # Keep it anchored near our original approx window (prevents snapping to wrong staff)
            anchor_penalty = abs(chosen[0] - approx_top) + abs(chosen[-1] - approx_bot)
            total = score + 0.15 * anchor_penalty
            if total < best_score:
                best_score = total
                best = chosen

    if best is not None:
        return best

    # Fallback: best 5-line window among candidates
    return sorted(best_five_line_window(centers_abs))


def _first_continuous_run_x(mask_1d: np.ndarray, win: int, min_on: int) -> int | None:
    """
    Find the first x where a window of length win has at least min_on "on" pixels.
    mask_1d: uint8 0/1
    """
    if mask_1d.size < win:
        return None
    kernel = np.ones(win, dtype=np.int32)
    run = np.convolve(mask_1d.astype(np.int32), kernel, mode="same")
    hits = np.where(run >= min_on)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def left_edge_x_for_staff(horiz_mask: np.ndarray, staff5: list[int]) -> int | None:
    """
    Determine where the staff lines *start* for this staff (handles indents).
    Uses an early "continuous run" detector per staff line, then aggregates.
    """
    H, W = horiz_mask.shape[:2]
    if len(staff5) < 5:
        return None

    # Light dilation to bridge tiny gaps (especially under clef)
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    hm = cv2.dilate(horiz_mask, dil_k, iterations=1)

    win = X_RUN_WINDOW
    min_on = int(win * X_RUN_MIN_FRAC)

    xs = []
    for y in staff5[:5]:
        a = max(0, y - 1)
        b = min(H, y + 2)
        strip = hm[a:b, :]
        col_any = (np.any(strip > 0, axis=0)).astype(np.uint8)

        x0 = _first_continuous_run_x(col_any, win=win, min_on=min_on)
        if x0 is not None:
            xs.append(x0)

    if len(xs) >= 3:
        # Use the median of the 3 smallest to lean left but stay stable
        xs = sorted(xs)[:3]
        x_left = int(np.median(xs))
        x_left = max(X_MIN, x_left - X_NUDGE_LEFT_PX)
        return x_left

    if len(xs) > 0:
        x_left = int(np.median(xs))
        x_left = max(X_MIN, x_left - X_NUDGE_LEFT_PX)
        return x_left

    return None


def annotate_guides(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)

    for page in doc:
        img = render_page_to_bgr(page, ZOOM)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        horiz = extract_horizontal_lines_mask(gray)
        line_ys = find_staff_line_ys(horiz)
        staves = group_lines_into_staves(line_ys)

        if not staves:
            page.draw_rect(fitz.Rect(5, 5, 25, 25), color=(1, 0, 0), width=2)
            continue

        for staff_lines in staves:
            approx5 = sorted(best_five_line_window(staff_lines))
            if len(approx5) < 2:
                continue

            # Refine to true 1st and 5th staff lines
            staff5 = refine_staff_5_lines(horiz, approx5)
            staff5 = sorted(staff5)
            if len(staff5) < 5:
                continue

            top_y, bot_y = staff5[0], staff5[-1]

            # Detect where this staff's lines start (independent per staff)
            x_left = left_edge_x_for_staff(horiz, staff5)
            if x_left is None:
                continue

            # Pixel -> PDF coords
            x_pdf = x_left / ZOOM
            y0_pdf = top_y / ZOOM
            y1_pdf = bot_y / ZOOM

            # Draw the guide (line between 1st and 5th staff line)
            page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=(1, 0, 0), width=1.0)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: annotate_guides_by_staff.py <input.pdf> <output.pdf>")
        sys.exit(1)

    annotate_guides(sys.argv[1], sys.argv[2])
