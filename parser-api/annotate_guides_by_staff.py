#!/usr/bin/env python3
"""
Debug step: draw a short RED vertical guide for EACH staff on each page.

Fix in this version:
- X stays as-is (already good)
- Y endpoints are refined per staff:
  snap top to the strongest line near the top
  snap bottom to the strongest line near the bottom
This makes the segment touch the 1st and 5th staff lines much more reliably.
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
ROW_HIT_THRESHOLD = 200
MAX_GAP_BETWEEN_LINES = 18
MIN_LINES_PER_STAFF = 4

# X detection
X_PAD_LEFT = 0
MIN_LONG_LINE_FRAC = 0.35
MAX_LINE_HEIGHT_PX = 8

# Y refinement (per staff)
REFINE_PAD_PX = 28          # how far above/below the detected staff to search
REFINE_LOCAL_THRESH_FRAC = 0.45  # local threshold relative to max row-sum in the band


def render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # PyMuPDF gives RGB for alpha=False; OpenCV likes BGR
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
    return horiz


def find_staff_line_ys(horiz_mask: np.ndarray) -> list[int]:
    row_sums = np.sum(horiz_mask > 0, axis=1)

    ys: list[int] = []
    in_run = False
    run_start = 0

    for y, count in enumerate(row_sums):
        if count >= ROW_HIT_THRESHOLD and not in_run:
            in_run = True
            run_start = y
        elif count < ROW_HIT_THRESHOLD and in_run:
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

        # Prefer windows that are lower on the page (helps include the true 5th line)
        # Smaller score is better.
        score = var - 0.002 * w[-1]
        if score < best_score:
            best_score = score
            best = w

    return best


def left_edge_x_for_staff(horiz_mask: np.ndarray, top_y: int, bot_y: int) -> int | None:
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

    # Fallback
    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    return int(np.percentile(xs, 1))


def _run_centers_from_row_sums(row_sums: np.ndarray, threshold: int) -> list[int]:
    """Return run-centers of rows where row_sum >= threshold."""
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


def refine_staff_endpoints(horiz_mask: np.ndarray, approx_top: int, approx_bot: int) -> tuple[int, int]:
    """
    Snap approx endpoints to the best local line rows inside a small band.
    Strategy:
    - Look in [approx_top-pad, approx_bot+pad]
    - Compute row_sums in that band
    - Find candidate line rows using a local threshold
    - Choose the best 5-line window; return its top and bottom
    """
    H, W = horiz_mask.shape[:2]
    y0 = max(0, approx_top - REFINE_PAD_PX)
    y1 = min(H, approx_bot + REFINE_PAD_PX)
    band = horiz_mask[y0:y1, :]

    row_sums = np.sum(band > 0, axis=1)
    mx = int(row_sums.max()) if row_sums.size else 0
    if mx <= 0:
        return approx_top, approx_bot

    local_thresh = max(20, int(mx * REFINE_LOCAL_THRESH_FRAC))
    centers = _run_centers_from_row_sums(row_sums, local_thresh)

    # If we found decent candidates, pick a 5-line staff and use its endpoints
    if len(centers) >= 5:
        centers_abs = [c + y0 for c in centers]
        chosen = best_five_line_window(centers_abs)
        chosen = sorted(chosen)
        return chosen[0], chosen[-1]

    # Fallback: snap to strongest rows near approx_top/bot
    # (helps when only a few lines are detected in the band)
    def snap_near(y_target: int, radius: int = 10) -> int:
        a = max(y0, y_target - radius)
        b = min(y1, y_target + radius + 1)
        sub = horiz_mask[a:b, :]
        rs = np.sum(sub > 0, axis=1)
        if rs.size == 0:
            return y_target
        return int(a + int(np.argmax(rs)))

    return snap_near(approx_top), snap_near(approx_bot)


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
            chosen = best_five_line_window(staff_lines)
            chosen = sorted(chosen)
            approx_top, approx_bot = chosen[0], chosen[-1]

            # Refine to true 1st/5th lines
            top_y, bot_y = refine_staff_endpoints(horiz, approx_top, approx_bot)

            x_left = left_edge_x_for_staff(horiz, top_y, bot_y)
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
