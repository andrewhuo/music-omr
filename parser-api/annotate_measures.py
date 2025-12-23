#!/usr/bin/env python3
"""
Annotate sheet music PDF with measure numbers.

Rules implemented (per your spec):
- Single staff
- Multiple systems per page
- Only number the first measure of each system
- Number is placed directly above the *first barline* of the system
- Centered on that barline
- If blocked by existing text (tempo/expression), nudge left/right but keep tied to same barline
- Output is a NEW PDF (original unchanged)

Dependencies:
- pymupdf (fitz)
- opencv-python
- numpy
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np


# ---------------------------
# PDF render helpers
# ---------------------------

def render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # pix.n should be 3 if alpha=False, but keep safe:
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


# ---------------------------
# Detection logic
# ---------------------------

@dataclass
class SystemInfo:
    y0: int
    y1: int
    staff_y_top: int
    staff_y_bottom: int
    first_bar_x: int
    bar_xs: List[int]


def _binarize(gray: np.ndarray) -> np.ndarray:
    # Robust binarization: invert so ink is 1/255
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    return thr


def _find_system_bands(bin_img: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find horizontal bands likely corresponding to systems.
    Works by horizontal projection: rows with lots of ink are likely staff regions.
    """
    h, w = bin_img.shape
    # Horizontal projection (ink per row)
    proj = (bin_img > 0).sum(axis=1).astype(np.float32)

    # Smooth projection to connect staff lines into system bands
    k = max(15, int(h * 0.01))
    k = k + 1 if k % 2 == 0 else k
    proj_smooth = cv2.GaussianBlur(proj.reshape(-1, 1), (1, k), 0).reshape(-1)

    # Threshold: rows with enough ink
    thresh = max(8, int(0.01 * w))
    mask = proj_smooth > thresh

    bands: List[Tuple[int, int]] = []
    in_band = False
    start = 0
    for y in range(h):
        if mask[y] and not in_band:
            in_band = True
            start = y
        elif not mask[y] and in_band:
            in_band = False
            end = y
            if end - start > 20:  # minimum height
                bands.append((start, end))

    if in_band:
        end = h - 1
        if end - start > 20:
            bands.append((start, end))

    # Merge close bands (staff lines can create fragmented bands)
    merged: List[Tuple[int, int]] = []
    for b in bands:
        if not merged:
            merged.append(b)
            continue
        prev = merged[-1]
        if b[0] - prev[1] < 30:  # gap small -> merge
            merged[-1] = (prev[0], b[1])
        else:
            merged.append(b)

    # Filter out very tall bands that are likely not a single staff system,
    # but still keep them (some scans include big slurs/text).
    # We’ll just use these as system ROIs.
    return merged


def _detect_barlines_in_roi(bin_img: np.ndarray, roi: Tuple[int, int, int, int]) -> List[int]:
    """
    Detect vertical barlines within a region of interest.
    Returns list of x positions (pixels) for barlines.
    """
    x0, y0, x1, y1 = roi
    sub = bin_img[y0:y1, x0:x1]
    if sub.size == 0:
        return []

    h, w = sub.shape

    # Remove staff lines (horizontal) to make verticals pop:
    # 1) detect horizontal lines using morphology
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 20), 1))
    horiz = cv2.morphologyEx(sub, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    # Subtract horizontals
    no_horiz = cv2.subtract(sub, horiz)

    # Now emphasize vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 3)))
    vert = cv2.morphologyEx(no_horiz, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    # Clean & connect
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Find contours of vertical segments
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs: List[int] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # Filter: tall-ish and thin-ish
        if ch < max(20, int(0.25 * h)):
            continue
        if cw > max(12, int(0.03 * w)):
            continue
        xs.append(x + cw // 2)

    if not xs:
        return []

    xs.sort()

    # Cluster nearby x's to one barline (avoid double detections)
    clustered: List[int] = []
    cluster = [xs[0]]
    for x in xs[1:]:
        if abs(x - cluster[-1]) <= 8:
            cluster.append(x)
        else:
            clustered.append(int(np.median(cluster)))
            cluster = [x]
    clustered.append(int(np.median(cluster)))

    # Convert back to full-image coordinates
    clustered = [x0 + x for x in clustered]
    return clustered


def detect_systems_and_first_barlines(bgr: np.ndarray) -> List[SystemInfo]:
    """
    Detect system regions and find the first barline of each system.
    Returns SystemInfo list (top-to-bottom).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bin_img = _binarize(gray)
    h, w = bin_img.shape

    bands = _find_system_bands(bin_img)

    systems: List[SystemInfo] = []
    for (y0, y1) in bands:
        # Expand a bit to include things above staff (tempo region)
        pad_top = 40
        pad_bot = 20
        yy0 = max(0, y0 - pad_top)
        yy1 = min(h, y1 + pad_bot)

        # Ignore very small bands
        if yy1 - yy0 < 40:
            continue

        # Define ROI for barline detection: avoid far-left margin where clef/time signature live
        # We'll still need the first barline, which is just to the right of clef/key/time.
        x_left = int(0.06 * w)
        x_right = int(0.98 * w)
        roi = (x_left, y0, x_right, y1)

        bar_xs = _detect_barlines_in_roi(bin_img, roi)
        if len(bar_xs) == 0:
            continue

        # Heuristic: first barline should not be extremely left (clef area).
        # We'll choose the leftmost barline that is to the right of x_left + small margin.
        min_x = x_left + int(0.01 * w)
        candidates = [x for x in bar_xs if x >= min_x]
        if not candidates:
            candidates = bar_xs

        first_bar_x = min(candidates)

        staff_y_top = y0
        staff_y_bottom = y1

        systems.append(SystemInfo(
            y0=yy0,
            y1=yy1,
            staff_y_top=staff_y_top,
            staff_y_bottom=staff_y_bottom,
            first_bar_x=first_bar_x,
            bar_xs=bar_xs,
        ))

    # Sort top-to-bottom
    systems.sort(key=lambda s: s.staff_y_top)
    return systems


# ---------------------------
# PDF annotation helpers
# ---------------------------

@dataclass
class TextBlock:
    rect: fitz.Rect


def get_text_blocks(page: fitz.Page) -> List[TextBlock]:
    """
    Collect text block rectangles for collision avoidance.
    """
    d = page.get_text("dict")
    blocks: List[TextBlock] = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        x0, y0, x1, y1 = b.get("bbox", [0, 0, 0, 0])
        blocks.append(TextBlock(rect=fitz.Rect(x0, y0, x1, y1)))
    return blocks


def collides(rect: fitz.Rect, blocks: List[TextBlock]) -> bool:
    for b in blocks:
        if rect.intersects(b.rect):
            return True
    return False


def place_measure_number(
    page: fitz.Page,
    blocks: List[TextBlock],
    number: int,
    x_pdf: float,
    y_pdf: float,
    font_size: float,
    max_nudge: float = 50.0,
    step: float = 6.0,
) -> None:
    """
    Place the number centered on x_pdf, with collision-aware left/right nudging.
    """
    text = str(number)

    # Rough width estimation: works well enough for small integers.
    # (We avoid depending on font metrics across environments.)
    est_char_w = font_size * 0.55
    est_w = max(font_size, est_char_w * len(text))
    est_h = font_size * 1.2

    def rect_for(center_x: float) -> fitz.Rect:
        return fitz.Rect(center_x - est_w / 2, y_pdf - est_h, center_x + est_w / 2, y_pdf)

    # Try centered first
    centers_to_try = [x_pdf]

    # Then try nudges alternating right/left
    steps = int(max_nudge // step)
    for i in range(1, steps + 1):
        centers_to_try.append(x_pdf + i * step)
        centers_to_try.append(x_pdf - i * step)

    chosen_center = x_pdf
    for cx in centers_to_try:
        r = rect_for(cx)
        if not collides(r, blocks):
            chosen_center = cx
            break

    # Final insertion point: left-bottom origin for insert_text
    # We'll place baseline slightly above y_pdf (which we chose as "just above staff").
    insert_x = chosen_center - est_w / 2
    insert_y = y_pdf - (est_h * 0.15)

    page.insert_text(
        fitz.Point(insert_x, insert_y),
        text,
        fontsize=font_size,
        fontname="helv",
        color=(0, 0, 0),
        overlay=True,
    )


def measures_in_system_from_barlines(bar_xs: List[int]) -> int:
    """
    Approximate how many measures are in this system.
    Typical case: measures = (#barlines - 1), since barlines mark boundaries.
    We clamp to at least 1.
    """
    if len(bar_xs) >= 2:
        return max(1, len(bar_xs) - 1)
    return 1


# ---------------------------
# Main
# ---------------------------

def annotate_pdf(
    input_pdf: str,
    output_pdf: str,
    start_measure: int = 1,
    zoom: float = 2.0,
    font_size: float = 10.0,
    y_offset_above_staff: float = 8.0,
) -> None:
    # Open original and create editable copy in memory
    doc = fitz.open(input_pdf)

    current_measure = start_measure

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)

        # Render for detection
        bgr = render_page_to_bgr(page, zoom=zoom)

        # Detect systems & first barlines in pixel space
        systems = detect_systems_and_first_barlines(bgr)

        # Get existing text blocks once per page for collision checks
        blocks = get_text_blocks(page)

        for sysinfo in systems:
            # Where to place number (pixel coords -> pdf coords)
            first_bar_x_px = sysinfo.first_bar_x

            # y position: just above staff top line (system staff_y_top),
            # but using expanded band y0/y1, we keep a stable staff anchor.
            y_above_staff_px = max(0, sysinfo.staff_y_top - int(y_offset_above_staff * zoom))

            x_pdf = first_bar_x_px / zoom
            y_pdf = y_above_staff_px / zoom

            # Place number (collision-aware)
            place_measure_number(
                page=page,
                blocks=blocks,
                number=current_measure,
                x_pdf=x_pdf,
                y_pdf=y_pdf,
                font_size=font_size,
            )

            # Update the running measure count using detected barlines
            current_measure += measures_in_system_from_barlines(sysinfo.bar_xs)

    # Save as a NEW PDF
    doc.save(output_pdf, deflate=True)
    doc.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate sheet music PDF with measure numbers.")
    p.add_argument("input_pdf", help="Path to input PDF (local file path).")
    p.add_argument("output_pdf", help="Path to output annotated PDF (local file path).")
    p.add_argument("--start-measure", type=int, default=1, help="Starting measure number (default: 1).")
    p.add_argument("--zoom", type=float, default=2.0, help="Render zoom for detection (default: 2.0).")
    p.add_argument("--font-size", type=float, default=10.0, help="Measure number font size (default: 10).")
    p.add_argument("--y-offset", type=float, default=8.0, help="Distance above staff (in PDF points-ish) (default: 8).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    annotate_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        start_measure=args.start_measure,
        zoom=args.zoom,
        font_size=args.font_size,
        y_offset_above_staff=args.y_offset,
    )
    print(f"Annotated PDF written to: {args.output_pdf}")
