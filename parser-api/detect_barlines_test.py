import fitz
import cv2
import numpy as np
import sys
import json
import os

def render_pdf_page_to_image(pdf_path, page_index=0, zoom=2.0):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    doc.close()
    return img

def detect_vertical_barlines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # IMPORTANT FIX: barlines in violin music are short, not page-height
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, int(image.shape[0] * 0.08))  # ~8% of page height
    )

    vertical_mask = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel
    )

    lines = cv2.HoughLinesP(
        vertical_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(image.shape[0] * 0.05),
        maxLineGap=5
    )

    x_positions = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x1 - x2) < 3:
                x_positions.append(x1)

    return sorted(set(x_positions))

def annotate_pdf(pdf_path, x_positions, output_pdf):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)

    for i, x in enumerate(x_positions, start=1):
        page.insert_text(
            (x / 2, 40),   # convert from zoomed image coords
            str(i),
            fontsize=10,
            color=(0, 0, 0)
        )

    doc.save(output_pdf)
    doc.close()

def main():
    pdf_path = sys.argv[1]
    output_dir = sys.argv[sys.argv.index("--output-dir") + 1]
    os.makedirs(output_dir, exist_ok=True)

    image = render_pdf_page_to_image(pdf_path)
    x_coords = detect_vertical_barlines(image)

    print("Detected barline X positions:", x_coords)
    print("Total barlines on page 0:", len(x_coords))

    annotated_pdf_path = os.path.join(output_dir, "annotated.pdf")
    measures_json_path = os.path.join(output_dir, "measures.json")

    # Always write outputs
    annotate_pdf(pdf_path, x_coords, annotated_pdf_path)

    with open(measures_json_path, "w") as f:
        json.dump(
            [{"measure": i + 1, "x": x} for i, x in enumerate(x_coords)],
            f,
            indent=2
        )

    print("Wrote:", annotated_pdf_path)
    print("Wrote:", measures_json_path)

if __name__ == "__main__":
    main()
