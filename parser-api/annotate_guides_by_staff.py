#!/usr/bin/env python3
"""
Draw a RED vertical guide for EACH staff on each page.

Current state:
- X anchoring is working well (baseline + blob, with vertical-line suppression).
- Remaining issues are mostly Y:
  - Some staves get skipped when only 4 staff lines are detected.
  - Some guides are too short because the "5 lines" set is missing outer lines.

Fixes:
1) If a staff has only 3–4 detected lines, synthesize missing lines by snapping to
   the strongest horizontal-line rows near expected positions.
2) Post-check any 5-line set: if its vertical span is too small vs staff spacing,
   "repair" top/bottom by snapping near expected outer lines.

Uses OpenCV morphology for line extraction. :contentReference[oaicite:1]{index=1}
Uses connectedComponentsWithStats for blob picking. :contentReference[oaicite:2]{index=2}
"""

import sys
import fitz  # PyMuPDF
import numpy as np
import cv2

# -----------------------------
# Tunables
# -----------------------------
ZOOM = 2.0

# Staff line row detection
ROW_HIT_FRAC = 0.07
ROW_HIT_MIN = 120
MAX_GAP_BETWEEN_LINES = 18

# Allow smaller groups so we can synthesize missing lines (we still validate later)
MIN_LINES_PER_STAFF = 3

# Horizontal staffline morphology
HORIZ_KERNEL_DIV = 25
HORIZ_CLOSE_LEN = 9

# Vertical line morphology (barlines / stems)
VERT_KERNEL_DIV = 25
VERT_CLOSE_LEN = 9

# Staff refinement retries
REFINE_PAD_BASE = 28
REFINE_PAD_MULTS = (0.6, 1.0, 1.5)
REFINE_THRESH_FRACS = (0.42, 0.35, 0.28, 0.22)

SPACING_TOL_FRAC = 0.30
MIN_STAFF_SPACING = 5.0
MAX_STAFF_SPACING = 22.0

# X anchoring
STAFF_BAND_MARGIN_FRAC = 0.18
STAFFLINE_SUBTRACT_DILATE = 7
VERTLINE_SUBTRACT_DILATE = 9

# Reconnect clef/key fragments after subtraction (small)
RECONNECT_DILATE = 2
RECONNECT_CLOSE = 3
INK_OPEN_K = 3

# Blob filters
CLEF_MIN_AREA_FRAC = 0.008
CLEF_MIN_H_FRAC = 0.50
CLEF_MAX_X_FRAC = 0.55
PAD_LEFT_OF_BLOB_PX = 6

# Reject barline-like components even if they survive
BARLINE_THIN_PX = 3
BARLINE_TALL_FRAC = 0.90

# Clamp blob search region near baseline
CLAMP_RIGHT_SPACES = 2.5

# Y repair threshold: if staff span is too short, fix outer lines
SPAN_MIN_SPACES = 3.6  # ideal is ~4 staff-spaces between line1 and line5

# Debug toggles
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

    ys = []
    in_run = False
    start = 0
    for y, cnt in enumerate(row_sums):
        if cnt >= threshold and not in_run:
            in_run = True
            start = y
        elif cnt < threshold and in_run:
            in_run = False
            end = y - 1
            ys.append((start + end) // 2)
    if in_run:
        end = len(row_sums) - 1
        ys.append((start + end) // 2)
    return ys


def group_lines_into_staves(line_ys: list[int]) -> list[list[int]]:
    line_ys = sorted(line_ys)
    staves = []
    cur = []
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
    centers = []
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


def staff_spacing(lines: list[int]) -> float:
    lines = sorted(lines)
    if len(lines) < 2:
        return 10.0
    diffs = np.diff(lines).astype(np.float32)
    s = float(np.median(diffs))
    return float(np.clip(s, MIN_STAFF_SPACING, MAX_STAFF_SPACING))


def _spacing_ok(lines5: list[int]) -> bool:
    if len(lines5) != 5:
        return False
    lines5 = sorted(lines5)
    diffs = np.diff(lines5).astype(np.float32)
    spacing = float(np.median(diffs))
    if spacing < MIN_STAFF_SPACING or spacing > MAX_STAFF_SPACING:
        return False
    tol = max(2.0, spacing * SPACING_TOL_FRAC)
    return float(np.max(np.abs(diffs - spacing))) <= tol


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


def _snap_to_strongest_row(row_sums: np.ndarray, y_target: float, radius: int) -> tuple[int, int]:
    """Return (best_y, best_strength) near y_target."""
    H = row_sums.shape[0]
    yt = int(round(y_target))
    a = max(0, yt - radius)
    b = min(H, yt + radius + 1)
    if a >= b:
        return yt, int(row_sums[min(max(yt, 0), H - 1)])
    window = row_sums[a:b]
    idx = int(np.argmax(window))
    best_y = a + idx
    return best_y, int(window[idx])


def synthesize_or_repair_staff5(
    staff_lines: list[int],
    page_row_sums: np.ndarray,
) -> tuple[list[int] | None, bool]:
    """
    Return (staff5, used_fallback_like).
    - If we already have 5 but it's too short, repair outer lines.
    - If we have 3–4, synthesize missing lines by snapping near expected positions.
    """
    ys = sorted(staff_lines)
    used = False

    if len(ys) >= 5:
        staff5 = ys[:5] if len(ys) == 5 else best_five_line_window(ys)
    elif len(ys) in (3, 4):
        used = True
        s = staff_spacing(ys)
        # Build hypotheses by extending above/below, snapping to strongest rows
        radius = int(max(3, 0.9 * s))

        # Start from current ys as "core"
        core = ys.copy()

        # We will try to add missing lines until we reach 5
        # Prefer adding to both ends (outer lines) first.
        while len(core) < 5:
            core = sorted(core)
            s = staff_spacing(core)
            radius = int(max(3, 0.9 * s))

            top_expect = core[0] - s
            bot_expect = core[-1] + s

            top_y, top_strength = _snap_to_strongest_row(page_row_sums, top_expect, radius)
            bot_y, bot_strength = _snap_to_strongest_row(page_row_sums, bot_expect, radius)

            # Choose the stronger addition first (more likely a real staff line)
            if top_strength >= bot_strength:
                core = [top_y] + core
            else:
                core = core + [bot_y]

            # Prevent duplicates collapsing
            core = sorted(list(dict.fromkeys(core)))

            # If snapping created duplicates and we still can't grow, break
            if len(core) < 5 and (top_strength == 0 and bot_strength == 0):
                break

        if len(core) != 5 or not _spacing_ok(core):
            return None, True
        staff5 = core
    else:
        return None, True

    # Repair step: if span too small, outer lines are likely missing/wrong
    staff5 = sorted(staff5)
    s = staff_spacing(staff5)
    span = staff5[-1] - staff5[0]

    if span < SPAN_MIN_SPACES * s:
        used = True
        radius = int(max(3, 0.9 * s))
        # Keep middle three, re-snap outer expectations
        mid3 = staff5[1:4]
        top_expect = mid3[0] - s
        bot_expect = mid3[-1] + s
        top_y, _ = _snap_to_strongest_row(page_row_sums, top_expect, radius)
        bot_y, _ = _snap_to_strongest_row(page_row_sums, bot_expect, radius)
        candidate = sorted([top_y] + mid3 + [bot_y])
        if _spacing_ok(candidate):
            staff5 = candidate

    if len(staff5) != 5:
        return None, used
    return staff5, used


def staff_left_baseline_x(horiz_mask: np.ndarray, top_y: int, bot_y: int, spacing_px: float) -> int | None:
    H, W = horiz_mask.shape[:2]
    margin = int(max(2, spacing_px * 0.8))
    y0 = max(0, top_y - margin)
    y1 = min(H, bot_y + margin)
    band = horiz_mask[y0:y1, :]
    _, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    return int(np.percentile(xs, 1))


def clef_like_blob_x(
    ink: np.ndarray,
    horiz_mask: np.ndarray,
    vert_mask: np.ndarray,
    top_y: int,
    bot_y: int,
    spacing_px: float,
    baseline_x: int | None
) -> int | None:
    H, W = ink.shape[:2]
    staff_h = max(1, bot_y - top_y)
    margin = int(staff_h * STAFF_BAND_MARGIN_FRAC)
    y0 = max(0, top_y - margin)
    y1 = min(H, bot_y + margin)

    band = ink[y0:y1, :].copy()

    # Remove vertical lines (barlines/stems) so they can't win. :contentReference[oaicite:3]{index=3}
    vd = cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERTLINE_SUBTRACT_DILATE))
    vband = cv2.dilate(vert_mask[y0:y1, :], vd, iterations=1)
    band[vband > 0] = 0

    # Gently remove staff lines
    hd = cv2.getStructuringElement(cv2.MORPH_RECT, (STAFFLINE_SUBTRACT_DILATE, 1))
    hband = cv2.dilate(horiz_mask[y0:y1, :], hd, iterations=1)
    band[hband > 0] = 0

    # Reconnect fragments (helps key signatures)
    if RECONNECT_DILATE > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_RECT, (RECONNECT_DILATE, RECONNECT_DILATE))
        band = cv2.dilate(band, kd, iterations=1)
    if RECONNECT_CLOSE > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (RECONNECT_CLOSE, RECONNECT_CLOSE))
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kc, iterations=1)

    if INK_OPEN_K >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (INK_OPEN_K, INK_OPEN_K))
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k, iterations=1)

    bin_img = (band > 0).astype(np.uint8)
    if np.count_nonzero(bin_img) == 0:
        return None

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8) :contentReference[oaicite:4]{index=4}

    band_area = float(bin_img.shape[0] * bin_img.shape[1])
    min_area = int(max(25, band_area * CLEF_MIN_AREA_FRAC))
    max_x = int(W * CLEF_MAX_X_FRAC)

    if baseline_x is not None:
        max_near = baseline_x + int(CLAMP_RIGHT_SPACES * spacing_px)
    else:
        max_near = max_x

    best = None  # (x, -area)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if area < min_area:
            continue
        if x > max_x or x > max_near:
            continue
        if h < int(staff_h * CLEF_MIN_H_FRAC):
            continue

        if w <= BARLINE_THIN_PX and h >= int(staff_h * BARLINE_TALL_FRAC):
            continue

        cy_abs = y0 + cy
        if not (top_y <= cy_abs <= bot_y):
            continue

        cand = (x, -area)
        if best is None or cand < best:
            best = cand

    if best is None:
        return None

    x_blob, _ = best
    return max(0, int(x_blob - PAD_LEFT_OF_BLOB_PX))


def annotate_guides(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)

    for page in doc:
        img = render_page_to_bgr(page, ZOOM)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ink = binarize_ink(gray)
        horiz = extract_horizontal_lines_mask(ink)
        vert = extract_vertical_lines_mask(ink)

        # Precompute per-page row sums for snapping (cheap + stable)
        page_row_sums = np.sum(horiz > 0, axis=1).astype(np.int32)

        line_ys = find_staff_line_ys(horiz)
        staves = group_lines_into_staves(line_ys)

        if not staves:
            page.draw_rect(fitz.Rect(5, 5, 25, 25), color=(1, 0, 0), width=2)
            continue

        for staff_lines in staves:
            # Step 1: try normal path if we already have >=5 line candidates
            staff5 = None
            used_fallback = False

            if len(staff_lines) >= 5:
                approx5 = sorted(best_five_line_window(staff_lines))
                refined = refine_staff_5_lines(horiz, approx5)
                if refined is not None and len(refined) == 5 and _spacing_ok(refined):
                    staff5 = sorted(refined)
                else:
                    # fallback to approx and then repair if needed
                    staff5, used_fallback = synthesize_or_repair_staff5(approx5, page_row_sums)

            # Step 2: if fewer than 5, synthesize/repair from what we have
            if staff5 is None:
                staff5, used_fallback2 = synthesize_or_repair_staff5(staff_lines, page_row_sums)
                used_fallback = used_fallback or used_fallback2

            if staff5 is None or len(staff5) != 5:
                continue

            top_y, bot_y = staff5[0], staff5[-1]
            s_px = staff_spacing(staff5)

            baseline_x = staff_left_baseline_x(horiz, top_y, bot_y, s_px)
            x_blob = clef_like_blob_x(ink, horiz, vert, top_y, bot_y, s_px, baseline_x)

            x_left = x_blob if x_blob is not None else baseline_x
            if x_left is None:
                continue

            if DEBUG_MARK_FALLBACK and used_fallback:
                r = fitz.Rect(x_left/ZOOM, (top_y-6)/ZOOM, (x_left+6)/ZOOM, top_y/ZOOM)
                page.draw_rect(r, color=(1, 0.5, 0), width=1)

            page.draw_line(
                (x_left / ZOOM, top_y / ZOOM),
                (x_left / ZOOM, bot_y / ZOOM),
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
