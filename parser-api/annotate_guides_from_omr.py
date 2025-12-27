#!/usr/bin/env python3
import sys
import re
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF

# How many pixels (in Audiveris image coords) to shift LEFT of the staff start
PAD_LEFT_PX = 6

RED = (1, 0, 0)
LINE_WIDTH = 1.0

SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


def list_sheet_xml_paths(omr_path: str) -> list[str]:
    with zipfile.ZipFile(omr_path, "r") as z:
        matches = []
        for name in z.namelist():
            m = SHEET_XML_RE.match(name)
            if m:
                matches.append((int(m.group(1)), name))
    matches.sort(key=lambda t: t[0])
    return [p for _, p in matches]


def load_sheet_xml(omr_path: str, sheet_xml_path: str) -> ET.Element:
    with zipfile.ZipFile(omr_path, "r") as z:
        data = z.read(sheet_xml_path)
    return ET.fromstring(data)


def draw_guides_for_sheet(page: fitz.Page, sheet_root: ET.Element) -> None:
    """
    Uses Audiveris staff lines geometry:
    <staff left="..." ...>
      <lines>
        <line> <point x="..." y="..."/> ... </line>  (5 of these)
      </lines>
    """
    # Get the staff entries anywhere under the sheet
    for staff in sheet_root.findall(".//staff"):
        left_attr = staff.get("left")
        if left_attr is None:
            continue
        try:
            staff_left_x = float(left_attr)
        except ValueError:
            continue

        # Collect 5 staff line y-values at the LEFT edge (use the point with min x for each line)
        line_ys = []
        for line in staff.findall(".//lines/line"):
            pts = []
            for pt in line.findall("./point"):
                x = pt.get("x")
                y = pt.get("y")
                if x is None or y is None:
                    continue
                try:
                    pts.append((float(x), float(y)))
                except ValueError:
                    pass
            if not pts:
                continue
            x_min, y_at_left = min(pts, key=lambda t: t[0])
            line_ys.append(y_at_left)

        if len(line_ys) < 5:
            # If Audiveris didn’t give 5, skip this staff (rare, but safe)
            continue

        line_ys.sort()
        top_y = line_ys[0]
        bot_y = line_ys[-1]

        x = max(0.0, staff_left_x - PAD_LEFT_PX)

        # Audiveris coords are in the raster image space used for the sheet.
        # If your PDF rendering matches that scale 1:1 you can draw directly.
        # In practice: your earlier logs show Audiveris picture width/height match the rasterized PDF page size,
        # so this should align.
        page.draw_line((x, top_y), (x, bot_y), color=RED, width=LINE_WIDTH)


def annotate_pdf_from_omr(input_pdf: str, input_omr: str, output_pdf: str) -> None:
    sheet_paths = list_sheet_xml_paths(input_omr)
    if not sheet_paths:
        raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

    doc = fitz.open(input_pdf)

    # Map page i -> sheet i (1-indexed in Audiveris naming)
    for i in range(doc.page_count):
        sheet_idx = min(i, len(sheet_paths) - 1)  # safety if mismatch
        sheet_root = load_sheet_xml(input_omr, sheet_paths[sheet_idx])
        page = doc.load_page(i)
        draw_guides_for_sheet(page, sheet_root)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_pdf_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
