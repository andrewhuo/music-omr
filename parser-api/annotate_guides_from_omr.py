#!/usr/bin/env python3
import sys
import re
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF


GUIDE_COLOR = (1, 0, 0)  # red
GUIDE_WIDTH = 1.0


_SHEET_XML_RE = re.compile(r"^sheet#(\d+)/sheet#\1\.xml$")


def _sorted_sheet_xml_paths(z: zipfile.ZipFile) -> list[str]:
    """Return sheet#N/sheet#N.xml paths sorted by N."""
    found = []
    for name in z.namelist():
        m = _SHEET_XML_RE.match(name)
        if m:
            found.append((int(m.group(1)), name))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _parse_sheet(z: zipfile.ZipFile, sheet_xml_path: str):
    data = z.read(sheet_xml_path)
    root = ET.fromstring(data)

    pic = root.find("picture")
    if pic is None:
        raise ValueError(f"No <picture> in {sheet_xml_path}")

    pic_w = float(pic.get("width"))
    pic_h = float(pic.get("height"))
    if pic_w <= 0 or pic_h <= 0:
        raise ValueError(f"Bad picture size in {sheet_xml_path}: {pic_w}x{pic_h}")

    # Collect staff guide specs: (x_left_px, y_top_px, y_bot_px)
    guides = []

    page = root.find("page")
    if page is None:
        return pic_w, pic_h, guides

    # systems -> parts -> staves
    for system in page.findall(".//system"):
        for staff in system.findall(".//staff"):
            lines_node = staff.find("lines")
            if lines_node is None:
                continue

            line_nodes = lines_node.findall("line")
            if len(line_nodes) < 5:
                # If Audiveris didn't detect 5 lines, skip (rare, but avoids garbage)
                continue

            # Prefer staff@left (accounts for indentation). Fallback to leftmost line point.
            x_left_attr = staff.get("left")
            x_left_px = float(x_left_attr) if x_left_attr is not None else None

            y_lefts = []
            x_lefts = []

            # Use only first 5 staff lines
            for ln in line_nodes[:5]:
                pts = ln.findall("point")
                if not pts:
                    break

                # pick leftmost point on this staffline
                best = None  # (x, y)
                for p in pts:
                    x = float(p.get("x"))
                    y = float(p.get("y"))
                    if best is None or x < best[0]:
                        best = (x, y)
                if best is None:
                    break
                x_lefts.append(best[0])
                y_lefts.append(best[1])

            if len(y_lefts) != 5:
                continue

            if x_left_px is None:
                x_left_px = min(x_lefts)

            y_top_px = min(y_lefts)
            y_bot_px = max(y_lefts)

            # Basic sanity: reject absurdly short/negative spans
            if y_bot_px <= y_top_px or (y_bot_px - y_top_px) < 10:
                continue

            guides.append((x_left_px, y_top_px, y_bot_px))

    return pic_w, pic_h, guides


def annotate_guides_from_omr(input_pdf: str, omr_path: str, output_pdf: str) -> None:
    doc = fitz.open(input_pdf)

    with zipfile.ZipFile(omr_path, "r") as z:
        sheet_paths = _sorted_sheet_xml_paths(z)
        if not sheet_paths:
            raise RuntimeError("No sheet#N/sheet#N.xml found inside .omr")

        # We expect one sheet per PDF page. If mismatch, we still do min() safely.
        for page_index in range(doc.page_count):
            page = doc[page_index]

            # Choose matching sheet; if fewer sheets than pages, reuse last.
            sheet_i = min(page_index, len(sheet_paths) - 1)
            sheet_xml_path = sheet_paths[sheet_i]

            pic_w, pic_h, guides = _parse_sheet(z, sheet_xml_path)
            if not guides:
                continue

            # Map Audiveris pixel coords -> PDF coords for THIS page
            # (PyMuPDF page coords use page.rect; origin at top-left in MuPDF space). :contentReference[oaicite:1]{index=1}
            rect = page.rect
            scale_x = rect.width / pic_w
            scale_y = rect.height / pic_h

            for (x_px, y0_px, y1_px) in guides:
                x_pdf = x_px * scale_x
                y0_pdf = y0_px * scale_y
                y1_pdf = y1_px * scale_y
                page.draw_line((x_pdf, y0_pdf), (x_pdf, y1_pdf), color=GUIDE_COLOR, width=GUIDE_WIDTH)

    doc.save(output_pdf)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: annotate_guides_from_omr.py <input.pdf> <input.omr> <output.pdf>")
        sys.exit(1)

    annotate_guides_from_omr(sys.argv[1], sys.argv[2], sys.argv[3])
