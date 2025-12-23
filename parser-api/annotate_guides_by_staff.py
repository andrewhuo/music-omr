#!/usr/bin/env python3
"""
Debug step: draw a short RED vertical guide for EACH staff on each page.

Improvements vs previous:
- X: use contours of long horizontal staff segments -> true left edge (per staff, supports indents)
- Y: pick the best 5-line window inside each staff group -> top touches 1st line, bottom touches 5th line
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2

# -----------------------------
# Tunables (only touch if needed)
# -----------------------------
ZOOM = 2.0                   # rendering scale; PDF coords = pixels / ZOOM
ROW_HIT_THRESHOLD = 200      # row "on" pixels required to count as a staff line row
MAX_GAP_BETWEEN_LINES = 18   # px gap threshold to group lines into one staff
MIN_LINES_PER_STAFF = 4      # tolerate missed/broken line: 4+ rows can still be a staff candidate
X_PAD_LEFT = 0               # per-staff shift (pixels) after detection; keep 0 for now

# Contour filtering to find "real" staff segments
MIN_LONG_LINE_FRAC = 0.35    # contour width must be >= this fraction of page width to count as staff line
MAX_LINE_HEIGHT_PX = 8       # contour bounding-rect height must be <= this to be considered a line


def render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Normalize to BGR for OpenCV
    if pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        # PyMuPDF yields RGB
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

    # Morphology for horizontal line extraction
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
    """
    If we have more than 5 candidate line rows, pick the best group of 5 that looks like a staff:
    - 5 lines
    - roughly even spacing
    """
    ys = sorted(ys)
    if len(ys) <= 5:
        return ys

    best = ys[:5]
    best_score = float("inf")

    for i in range(0, len(ys) - 4):
        w = ys[i:i+5]
        diffs = [w[j+1] - w[j] for j in range(4)]
        med = float(np.median(diffs))
        score = sum(abs(d - med) for d in diffs)
        if score < best_score:
            best_score = score
            best = w

    return best


def left_edge_x_for_staff(horiz_mask: np.ndarray, top_y: int, bot_y: int) -> int | None:
    """
    Find left edge of staff lines within a band by looking for LONG horizontal contours,
    then returning the minimum x among those long contours.
    """
    y0 = max(0, top_y - 2)
    y1 = min(horiz_mask.shape[0], bot_y + 3)
    band = horiz_mask[y0:y1, :]

    # Find contours in the band
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

    # Fallback: robust low percentile if no long contours matched
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
            # mark page so we know it ran
            page.draw_rect(fitz.Rect(5, 5, 25, 25), color=(1, 0, 0), width=2)
            continue

        for staff_lines in staves:
            # Choose the best 5 staff lines (so top/bottom are correct)
            chosen = best_five_line_window(staff_lines)
            chosen = sorted(chosen)

            top_y = chosen[0]     # 1st line
            bot_y = chosen[-1]    # 5th line

            x_left = left_edge_x_for_staff(horiz, top_y, bot_y)
            if x_left is None:
                continue

            x_px = max(0, x_left + X_PAD_LEFT)

            # Convert pixel coords -> PDF coords
            x_pdf = x_px / ZOOM
            y0_pdf = top_y / ZOOM
            y1_pdf = bot_y / ZOOM

            page.draw_line(
                (x_pdf, y0_pdf),
                (x_pdf, y1_pdf),
                color=(1, 0, 0),
                width=1.0,
            )

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: annotate_guides_by_staff.py <input.pdf> <output.pdf>")
        sys.exit(1)

    annotate_guides(sys.argv[1], sys.argv[2])
