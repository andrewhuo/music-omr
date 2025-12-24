#!/usr/bin/env python3
"""
Debug step: draw a short RED vertical guide for EACH staff on each page.

Improvements in this version:
- More robust staff-line detection across different PDFs:
  * row hit threshold scales with page width (instead of fixed 200)
  * horizontal mask gets a light "close" to reconnect broken staff lines
- More precise X at the true left edge of the staff:
  * estimate left edge from the 5 staff-line pixels (percentile-based)
  * fall back to contour method if needed
- More reliable Y endpoints:
  * refinement pad scales with staff height
  * slightly gentler local threshold for faint lines
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2

# -----------------------------
# Tunables
# -----------------------------
ZOOM = 2.0

# Detect staff-line rows (global scan)
ROW_HIT_FRAC = 0.07      # fraction of page width that must be "ink" to count as staff line
ROW_HIT_MIN = 120        # minimum pixels, prevents too-low threshold on narrow pages
MAX_GAP_BETWEEN_LINES = 18
MIN_LINES_PER_STAFF = 4

# X detection
X_PAD_LEFT = 0
MIN_LONG_LINE_FRAC = 0.35
MAX_LINE_HEIGHT_PX = 8

# Y refinement (per staff)
REFINE_PAD_PX = 28               # baseline pad; actual pad scales with staff height
REFINE_LOCAL_THRESH_FRAC = 0.38  # gentler for faint lines (was 0.45)


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

    # Horizontal line extraction via morphology (erode + dilate with wide kernel)
    kernel_len = max(30, gray.shape[1] // 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    tmp = cv2.erode(bw, horiz_kernel, iterations=1)
    horiz = cv2.dilate(tmp, horiz_kernel, iterations=1)

    # Light close to reconnect small breaks in staff lines
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return horiz


def find_staff_line_ys(horiz_mask: np.ndarray) -> list[int]:
    row_sums = np.sum(horiz_mask > 0, axis=1)

    # Adaptive threshold based on page width
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

        # Prefer windows lower on the page a tiny bit (helps avoid header junk)
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


def refine_staff_lines(horiz_mask: np.ndarray, approx_lines: list[int]) -> list[int] | None:
    """
    Given approximate staff lines (ideally 5), search a local band and return a refined 5-line set if possible.
    """
    if not approx_lines:
        return None
    approx_lines = sorted(approx_lines)
    approx_top, approx_bot = approx_lines[0], approx_lines[-1]

    H, W = horiz_mask.shape[:2]
    staff_h = max(1, approx_bot - approx_top)
    pad = max(REFINE_PAD_PX, int(staff_h * 0.6))

    y0 = max(0, approx_top - pad)
    y1 = min(H, approx_bot + pad)
    band = horiz_mask[y0:y1, :]

    row_sums = np.sum(band > 0, axis=1)
    mx = int(row_sums.max()) if row_sums.size else 0
    if mx <= 0:
        return None

    local_thresh = max(20, int(mx * REFINE_LOCAL_THRESH_FRAC))
    centers = _run_centers_from_row_sums(row_sums, local_thresh)
    if len(centers) < 5:
        return None

    centers_abs = [c + y0 for c in centers]
    chosen = best_five_line_window(centers_abs)
    return sorted(chosen) if len(chosen) >= 5 else None


def left_edge_x_from_staff_lines(horiz_mask: np.ndarray, staff_lines: list[int]) -> int | None:
    """
    Estimate the staff's true left edge by looking at pixels on the staff lines themselves.
    This is usually more accurate than contours when the left edge is faint/broken.
    """
    H, W = horiz_mask.shape[:2]
    if not staff_lines:
        return None

    per_line_x = []
    for y in staff_lines[:5]:
        a = max(0, y - 1)
        b = min(H, y + 2)
        strip = horiz_mask[a:b, :]
        ys, xs = np.where(strip > 0)
        if xs.size == 0:
            continue
        # Use a low percentile to lean left but avoid a single stray pixel
        per_line_x.append(int(np.percentile(xs, 2)))

    if len(per_line_x) >= 3:
        # Take median of the 3 smallest values: "left-leaning but stable"
        vals = sorted(per_line_x)[:3]
        return int(np.median(vals))

    if len(per_line_x) > 0:
        return int(np.median(per_line_x))

    return None


def left_edge_x_from_contours(horiz_mask: np.ndarray, top_y: int, bot_y: int) -> int | None:
    y0 = max(0, top_y - 2)
    y1 = min(horiz_mask.shape[0], bot_y + 3)
    band = horiz_mask[y0:y1, :]

    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    band_w = band.shape[1]
    min_x_candidates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= int(band_w * MIN_LONG_LINE_FRAC) and h <= MAX_LINE_HEIGHT_PX:
            min_x_candidates.append(x)

    if min_x_candidates:
        return int(min(min_x_candidates))

    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    return int(np.percentile(xs, 1))


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
            # Initial pick of 5 lines
            approx5 = sorted(best_five_line_window(staff_lines))
            approx_top, approx_bot = approx5[0], approx5[-1]

            # Refine to true 5-line staff if possible
            refined5 = refine_staff_lines(horiz, approx5)
            use_lines = refined5 if refined5 is not None else approx5
            top_y, bot_y = use_lines[0], use_lines[-1]

            # Better X: try line-pixel based left edge first, fall back to contours
            x_left = left_edge_x_from_staff_lines(horiz, use_lines)
            if x_left is None:
                x_left = left_edge_x_from_contours(horiz, top_y, bot_y)
            if x_left is None:
                continue

            x_px = max(0, x_left + X_PAD_LEFT)

            # Pixel -> PDF coords
            x_pdf = x_px / ZOOM
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
