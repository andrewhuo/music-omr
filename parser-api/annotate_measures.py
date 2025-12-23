#!/usr/bin/env python3
"""
Annotate measure numbers on a PDF.

Locked-in behavior:
- Only annotate the TOP staff of each system
- At the beginning of each system:
    draw a temporary vertical guide line
    write the measure number just to the right of it
    erase the guide line
- Numbers are placed ABOVE the top staff
"""

import sys
import fitz  # PyMuPDF

# -----------------------------
# Layout constants (tunable)
# -----------------------------
FONT_SIZE = 10
TEXT_X_OFFSET = 4     # pixels to the right of guide
TEXT_Y_OFFSET = 8     # pixels above staff top
GUIDE_HEIGHT = 40     # height of temporary guide line


def annotate_measures(input_pdf: str, output_pdf: str):
    doc = fitz.open(input_pdf)
    measure_number = 1

    for page in doc:
        page_width = page.rect.width

        # Use text blocks as a proxy for staff systems
        blocks = page.get_text("blocks")

        # Keep wide blocks only (likely music systems)
        systems = [
            b for b in blocks
            if (b[2] - b[0]) > page_width * 0.6
        ]

        # Sort top → bottom
        systems.sort(key=lambda b: b[1])

        for block in systems:
            x0, y0, x1, y1 = block[:4]

            staff_top_y = y0
            guide_x = x0

            # 1) draw temporary guide
            page.draw_line(
                (guide_x, staff_top_y),
                (guide_x, staff_top_y + GUIDE_HEIGHT),
                color=(1, 0, 0),
                width=0.5,
            )

            # 2) write measure number
            page.insert_text(
                (guide_x + TEXT_X_OFFSET, staff_top_y - TEXT_Y_OFFSET),
                str(measure_number),
                fontsize=FONT_SIZE,
                color=(0, 0, 0),
            )

            # 3) erase guide (overdraw in white)
            page.draw_line(
                (guide_x, staff_top_y),
                (guide_x, staff_top_y + GUIDE_HEIGHT),
                color=(1, 1, 1),
                width=1.2,
            )

            measure_number += 1

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: annotate_measures.py <input.pdf> <output.pdf>")
        sys.exit(1)

    annotate_measures(sys.argv[1], sys.argv[2])
