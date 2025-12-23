#!/usr/bin/env python3
"""
Phase A: draw a simple vertical guide on every page.
This will confirm the script runs and outputs get produced.
"""

import sys
import fitz  # PyMuPDF

def draw_guides(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)
    for page in doc:
        width = page.rect.width
        height = page.rect.height

        # Draw a vertical guide line at 10% of page width
        guide_x = width * 0.1

        page.draw_line(
            (guide_x, 0),
            (guide_x, height),
            color=(1, 0, 0),
            width=1.0,
        )

    doc.save(output_pdf)
    doc.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: annotate_measures_v0.py <input.pdf> <output.pdf>")
        sys.exit(1)

    draw_guides(sys.argv[1], sys.argv[2])
