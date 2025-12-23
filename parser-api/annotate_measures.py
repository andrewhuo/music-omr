# parser-api/annotate_measures.py

import sys
import fitz  # PyMuPDF

def annotate_debug(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)

    print(f"[DEBUG] Opened PDF with {doc.page_count} pages")

    page = doc[0]

    # Draw a very obvious red rectangle
    rect = fitz.Rect(50, 50, 550, 180)
    page.draw_rect(
        rect,
        color=(1, 0, 0),      # red
        width=4
    )

    # Draw very obvious red text
    page.insert_text(
        (70, 120),
        "DEBUG: annotate_measures.py RAN",
        fontsize=24,
        color=(1, 0, 0)
    )

    print("[DEBUG] Drew rectangle and text on page 1")

    # Save as a brand new file
    doc.save(output_pdf)
    doc.close()

    print(f"[DEBUG] Saved annotated PDF to {output_pdf}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python annotate_measures.py <input.pdf> <output.pdf>")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]

    print("[DEBUG] annotate_measures.py starting")
    print(f"[DEBUG] input_pdf = {input_pdf}")
    print(f"[DEBUG] output_pdf = {output_pdf}")

    annotate_debug(input_pdf, output_pdf)

    print("[DEBUG] annotate_measures.py finished")
