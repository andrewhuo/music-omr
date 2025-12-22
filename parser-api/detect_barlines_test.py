# parser-api/detect_barlines_test.py

import fitz
import cv2
import numpy as np
import sys

def render_pdf_page_to_image(pdf_path, page_index=0, zoom=2.0):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img

def detect_vertical_barlines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0] // 2))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.HoughLinesP(
        vertical_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=150,
        minLineLength=int(image.shape[0] * 0.4),
        maxLineGap=10
    )
    x_positions = []
    if lines is not None:
        for l in lines:
            x1, _, x2, _ = l[0]
            if abs(x1 - x2) < 5:
                x_positions.append(x1)
    return sorted(set(x_positions))

def main():
    pdf_path = sys.argv[1]
    image = render_pdf_page_to_image(pdf_path, 0)
    x_coords = detect_vertical_barlines(image)
    print("Detected barline X positions:", x_coords)
    print("Total barlines on page 0:", len(x_coords))

if __name__ == "__main__":
    main()
