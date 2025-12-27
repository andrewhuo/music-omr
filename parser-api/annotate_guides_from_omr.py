#!/usr/bin/env python3
"""
Draw staff-bounded vertical guide lines using Audiveris .omr (XML) geometry.

Inputs:
  1) input.pdf
  2) input.omr  (zip containing sheet#N/sheet#N.xml)
  3) output.pdf

What it does:
- Reads the Audiveris sheet XML from the .omr archive
- For each PDF page, finds the matching <page id="...">
- For each <staff>, uses:
    x_draw = staff@left (optionally minus a tiny pad)
    y_top  = y at x_draw on staff line #1 (top line)
    y_bot  = y at x_draw on staff line #5 (bottom line)
- Converts from Audiveris pixel coords to PDF coords by scaling with page.rect
- Draws a red vertical line at that location

This should be robust to:
- indentation
- missing OpenCV detections
- clutter (text/slurs) near staves
"""

import sys
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF


GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0
X_PAD_PX = 0.0  # set to e.g. 2.0 if you want a tiny nudge left; keep 0 for now


def _find_first_sheet_xml_in_omr(omr_path: str) -> str:
    with zipfile.ZipFile(omr_path, "r") as z:
        names = z.namelist()
    # Prefer sheet#1/sheet#1.xml but support any sheet numbering
    sheet_xmls = [n for n in names if n.startswith("sheet#") and n.endswith(".xml") and "/sheet#" in n]
    if not sheet_xmls:
        raise RuntimeError("No sheet XML found inside .omr archive.")
    sheet_xmls.sort()
    return sheet_xmls[0]


def _load_sheet_root(omr_path: str) -> ET.Element:
    sheet_xml = _find_first_sheet_xml_in_omr(omr_path)
    with zipfile.ZipFile(omr_path, "r") as z:
        data = z.read(sheet_xml)
    return ET.fromstring(data)


def _interp_y_at_x(points, xq: float) -> float:
    """
    points: list of (x,y) along the staff line polyline
    Returns y at xq using linear interpolation along x.
    """
    if not points:
        raise ValueError("Empty point list")
    if len(points) == 1:
        return float(points[0][1])

    # If outside range, clamp to nearest endpoint
    xs = [p[0] for p in points]
    if xq <= xs[0]:
        return float(points[0][1])
    if xq >= xs[-1]:
        return float(points[-1][1])

    # Find segment containing xq
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= xq <= x1 or x1 <= xq <= x0:
            if x1 == x0:
                return float(y0)
            t = (xq - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

    # Fallback
    return float(points[-1][1])


def annotate_guides_from_omr(input_pdf: str, input_omr: str, output_pdf: str) -> None:
    root = _load_sheet_root(input_omr)

    picture = root.find("./picture")
    if picture is None:
        raise RuntimeError("No <picture> node found in sheet XML.")

    pic_w = float(picture.attrib["width"])
    pic_h = float(picture.attrib["height"])

    doc = fitz.open(input_pdf)

    for page_index, page in enumerate(doc):
        # Audiveris uses 1-based page ids in the XML snippet you posted
        page_id = str(page_index + 1)
        page_node = root.find(f".//page[@id='{page_id}']")
        if page_node is None:
            # If missing, just skip this page
            continue

        pdf_w = float(page.rect.width)
        pdf_h = float(page.rect.height)

        sx = pdf_w / pic_w
        sy = pdf_h / pic_h

        for system in page_node.findall("./system"):
            for part in system.findall("./part"):
                for staff in part.findall("./staff"):
                    left = staff.attrib.get("left")
                    if left is None:
                        continue
                    x_draw = float(left) - float(X_PAD_PX)
                    if x_draw < 0:
                        x_draw = 0.0

                    lines_node = staff.find("./lines")
                    if lines_node is None:
                        continue
                    line_nodes = lines_node.findall("./line")
                    if len(line_nodes) < 5:
                        continue  # skip malformed staves

                    def parse_line_points(line_elem):
                        pts = []
                        for p in line_elem.findall("./point"):
                            pts.append((float(p.attrib["x"]), float(p.attrib["y"])))
                        # Sort by x so interpolation works reliably
                        pts.sort(key=lambda t: t[0])
                        return pts

                    top_pts = parse_line_points(line_nodes[0])
                    bot_pts = parse_line_points(line_nodes[4])

                    if not top_pts or not bot_pts:
                        continue

                    y_top = _interp_y_at_x(top_pts, x_draw)
                    y_bot = _interp_y_at_x(bot_pts, x_draw)

                    # Convert to PDF coords
                    x_pdf = x_draw * sx
                    y0_pdf = y_top * sy
                    y1_pdf = y_bot * sy

                    page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=GUIDE_COLOR, width=GUIDE_WIDTH)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_guides_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
