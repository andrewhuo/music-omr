#!/usr/bin/env python3
"""
Debug step: draw a short RED vertical guide for EACH staff on each page.

Goal:
- Find staff horizontal lines (top & bottom per staff)
- For each staff: draw a vertical segment at the "beginning" (left edge of staff)
- No numbers yet. (Numbers come next step.)

Pipeline:
1) Render page -> image (PyMuPDF)
2) Binarize + morphology -> horizontal-line mask (OpenCV)
3) Detect staff-line rows (peaks in row sums)
4) Group rows into staves
5) For each staff: estimate left edge X (robust percentile)
6) Draw red segment from top staff line to bottom staff line (PDF coords)
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2


# -----------------------------
# Tunables (adjust only if needed)
# -----------------------------
ZOOM = 2.0                  # rendering scale; PDF coords = pixels / ZOOM
ROW_HIT_THRESHOLD = 200     # row "on" pixels required to count as a line row
MAX_GAP_BETWEEN_LINES = 18  # px gap threshold to group lines into one staff
MIN_LINES_PER_STAFF = 4     # accept 4+ lines to tolerate missed/broken line
X_PAD_LEFT = 0              # shift guide right (+) or left (-) in pixels


def render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # usually RGB, no alpha
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Normalize to BGR for OpenCV
    if pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        # PyMuPDF gives RGB; OpenCV prefers BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        # Fallback (rare): best-effort
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def extract_horizontal_lines_mask(gray: np.ndarray) -> np.ndarray:
    # Invert threshold so dark staff lines become white in the mask
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract horizontal lines: erode + dilate with a wide horizontal kernel
    kernel_len = max(30, gray.shape[1] // 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    tmp = cv2.erode(bw, horiz_kernel, iterations=1)
    horiz = cv2.dilate(tmp, horiz_kernel, iterations=1)
    return horiz


def find_staff_line_ys(horiz_mask: np.ndarray) -> list[int]:
    # staff lines create strong row-wise "on" counts
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


def left_edge_x_for_staff(horiz_mask: np.ndarray, top_y: int, bot_y: int) -> int | None:
    # Look in a thin band around the staff region
    y0 = max(0, top_y - 2)
    y1 = min(horiz_mask.shape[0], bot_y + 3)
    band = horiz_mask[y0:y1, :]

    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return None

    # Use a low percentile instead of absolute min to avoid tiny specks
    return int(np.percentile(xs, 2))


def annotate_guides(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)

    for page in doc:
        img = render_page_to_bgr(page, ZOOM)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        horiz = extract_horizontal_lines_mask(gray)
        line_ys = find_staff_line_ys(horiz)
        staves = group_lines_into_staves(line_ys)

        # If we detect nothing, mark the page so we know the script ran.
        if not staves:
            page.draw_rect(fitz.Rect(5, 5, 25, 25), color=(1, 0, 0), width=2)
            continue

        for staff_lines in staves:
            top_y = min(staff_lines)
            bot_y = max(staff_lines)

            x_left = left_edge_x_for_staff(horiz, top_y, bot_y)
            if x_left is None:
                continue

            x_px = max(0, x_left + X_PAD_LEFT)

            # Convert pixel coords -> PDF coords
            x_pdf = x_px / ZOOM
            y0_pdf = top_y / ZOOM
            y1_pdf = bot_y / ZOOM

            # Draw segmented guide in PDF space
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
