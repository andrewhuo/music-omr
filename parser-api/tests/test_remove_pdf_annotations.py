import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - local dev env may not have PyMuPDF.
    fitz = None


MODULE_PATH = Path(__file__).resolve().parents[1] / "remove_pdf_annotations.py"
if fitz is None:
    MODULE = None
else:
    SPEC = importlib.util.spec_from_file_location("remove_pdf_annotations", MODULE_PATH)
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC and SPEC.loader
    SPEC.loader.exec_module(MODULE)


class RemovePdfAnnotationsTests(unittest.TestCase):
    @unittest.skipIf(fitz is None, "PyMuPDF is not installed")
    def test_removes_annotations_from_copy_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.pdf"
            cleaned = root / "cleaned.pdf"
            report = root / "report.json"

            doc = fitz.open()
            page = doc.new_page(width=200, height=200)
            page.add_ink_annot([[(20, 20), (80, 80), (120, 30)]])
            page.add_highlight_annot(fitz.Rect(30, 120, 100, 135))
            doc.save(str(original))
            doc.close()

            payload = MODULE.remove_annotations(str(original), str(cleaned), str(report))

            self.assertEqual(payload["annotations_removed"], 2)
            self.assertTrue(payload["clean_pdf_used_for_omr"])
            self.assertTrue(payload["final_pdf_uses_original"])

            original_doc = fitz.open(str(original))
            cleaned_doc = fitz.open(str(cleaned))
            try:
                self.assertEqual(sum(1 for _ in original_doc[0].annots()), 2)
                self.assertEqual(list(cleaned_doc[0].annots() or []), [])
                self.assertEqual(original_doc[0].rect, cleaned_doc[0].rect)
            finally:
                original_doc.close()
                cleaned_doc.close()

            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved["annotations_removed"], 2)

    @unittest.skipIf(fitz is None, "PyMuPDF is not installed")
    def test_no_annotations_keeps_copy_and_reports_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.pdf"
            cleaned = root / "cleaned.pdf"
            report = root / "report.json"

            doc = fitz.open()
            doc.new_page(width=200, height=200)
            doc.save(str(original))
            doc.close()

            payload = MODULE.remove_annotations(str(original), str(cleaned), str(report))

            self.assertEqual(payload["annotations_removed"], 0)
            self.assertFalse(payload["clean_pdf_used_for_omr"])
            self.assertTrue(cleaned.exists())


if __name__ == "__main__":
    unittest.main()
