# parser-api/annotate_pdf_measures.py

import fitz  # PyMuPDF
import cv2
import numpy as np

def render_pdf_page_as_image(pdf_path: str, page_index: int, zoom: float = 2.0):
    """
    Render a PDF page as a high-resolution image for line detection.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    matrix = fitz.Matrix(zoom, zoom)  # zoom increases resolution
    pix = page.get_pixmap(matrix=matrix)
    doc.close()

    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    channels = pix.n
    height = pix.height
    width = pix.width

    img = img_data.reshape((height, width, channels))
    if channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img

def detect_vertical_barlines(image):
    """
    Discover vertical lines (barlines) in a music score page image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, image.shape[0] // 2)
    )
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    lines = cv2.HoughLinesP(
        vertical_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        minLineLength=int(image.shape[0] * 0.4),
        maxLineGap=10
    )

    line_positions = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x1 - x2) < 5:  # near-vertical
                line_positions.append((x1, min(y1, y2)))

    line_positions.sort(key=lambda pos: pos[0])
    return line_positions

def place_measure_numbers(
    pdf_path: str,
    measure_list: list[int],
    output_pdf: str
):
    """
    Read the original PDF, detect barlines, and place measure numbers above them.
    measure_list comes from your MusicXML parser.
    """
    doc = fitz.open(pdf_path)
    coords = []

    measure_index = 0

    for page_num in range(doc.page_count):
        image = render_pdf_page_as_image(pdf_path, page_num)

        barlines = detect_vertical_barlines(image)

        for (x, y) in barlines:
            if measure_index >= len(measure_list):
                break

            measure_num = measure_list[measure_index]
            coords.append((page_num, x, y, measure_num))
            measure_index += 1

        if measure_index >= len(measure_list):
            break

    for (pg, x, y, m) in coords:
        page = doc.load_page(pg)
        text = str(m)
        page.insert_text(
            (x, max(y - 30, 0)),
            text,
            fontsize=10,
            color=(0, 0, 0),
        )

    doc.save(output_pdf)
    doc.close()

if __name__ == "__main__":
    # Example usage:
    original_pdf_file = "input.pdf"
    measures = [1, 2, 3, 4]  # replace with your actual list
    place_measure_numbers(original_pdf_file, measures, "annotated_output.pdf")
