import sys
import fitz
import numpy as np

def detect_staff_regions(page_image, threshold=200):
    """
    Detect approximate vertical bands where staff systems are.
    This uses horizontal projection to find clusters of horizontal lines.
    """
    gray = fitz.Pixmap(page_image, 0) if page_image.n < 4 else page_image.shrink()
    arr = np.frombuffer(gray.samples, dtype=np.uint8).reshape(gray.height, gray.width, gray.n)
    if arr.shape[2] > 1:
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    proj = np.sum(arr < threshold, axis=1)  # count dark pixels per row
    # find peaks where staff lines appear
    means = np.convolve(proj, np.ones(3)/3, mode='same')
    staff_rows = np.where(means > (0.5 * means.max()))[0]
    if len(staff_rows) == 0:
        return []
    diffs = np.diff(staff_rows)
    groups = [[staff_rows[0]]]
    for idx, d in enumerate(diffs):
        if d < 5:  # close rows
            groups[-1].append(staff_rows[idx + 1])
        else:
            groups.append([staff_rows[idx + 1]])
    regions = [(min(g), max(g)) for g in groups]
    return regions

def detect_vertical_barlines(page_image, staff_region, threshold=200):
    """
    Detect vertical lines (barlines) within a horizontal staff region.
    Returns sorted x coordinates of candidate vertical lines.
    """
    gray = fitz.Pixmap(page_image, 0) if page_image.n < 4 else page_image.shrink()
    arr = np.frombuffer(gray.samples, dtype=np.uint8).reshape(gray.height, gray.width, gray.n)
    if arr.shape[2] > 1:
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    (y0, y1) = staff_region
    arr_band = arr[y0:y1, :]
    col_dark_counts = np.sum(arr_band < threshold, axis=0)
    peaks = np.where(col_dark_counts > (0.7 * arr_band.shape[0]))[0]
    return sorted(peaks)

def annotate_pdf(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
    for page in doc:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        staff_regions = detect_staff_regions(pix)
        if not staff_regions:
            print("No staff regions found on this page.")
            continue

        top_staff = staff_regions[0]
        bar_xs = detect_vertical_barlines(pix, top_staff)
        if not bar_xs:
            print("No barlines detected on this page.")
            continue
        first_barline_x = bar_xs[0]

        offset = 20
        number_x = max(first_barline_x - offset, 10)
        top_y = top_staff[0]
        vertical_margin = 30

        text = "1"
        page.insert_text(
            (number_x / 2, (top_y - vertical_margin) / 2),
            text,
            fontsize=12,
            color=(0, 0, 0)
        )

    doc.save(output_pdf)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python annotate_measures.py <input.pdf> <output.pdf>")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]

    annotate_pdf(input_pdf, output_pdf)
