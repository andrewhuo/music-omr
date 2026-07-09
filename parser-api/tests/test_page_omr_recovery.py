import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "page_omr_recovery.py"
SPEC = importlib.util.spec_from_file_location("page_omr_recovery", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_mxl(path, parts=1, measures=2):
    measure_xml = "".join(f'<measure number="{idx + 1}"/>' for idx in range(measures))
    part_xml = "".join(f'<part id="P{idx + 1}">{measure_xml}</part>' for idx in range(parts))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("score.xml", f"<score-partwise>{part_xml}</score-partwise>")


class PageOmrRecoveryTests(unittest.TestCase):
    def test_page_output_requires_parts_and_measures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(MODULE.inspect_page_output(str(out))["reason"], "no_mxl_candidates")
            _write_mxl(out / "page.mxl", parts=1, measures=3)
            result = MODULE.inspect_page_output(str(out))
            self.assertTrue(result["success"])
            self.assertEqual(result["measures_count"], 3)

    def test_retry_status_marks_recovered_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            status = root / "status.json"
            MODULE.write_attempt_status(str(out), str(status), 2, 1, "/tmp/attempt1.log")
            _write_mxl(out / "page.mxl")
            result = MODULE.write_attempt_status(str(out), str(status), 2, 2, "/tmp/attempt2.log")
            self.assertTrue(result["success"])
            self.assertTrue(result["recovered_on_retry"])
            self.assertEqual(result["attempts"], 2)

    def test_report_separates_success_failed_and_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "entries": [
                    {
                        "page_number": 1,
                        "status": "ok",
                        "parts_count": 1,
                        "measures_count": 4,
                        "system_starts": ["1", "3"],
                    },
                    {
                        "page_number": 2,
                        "status": "missing",
                        "error": "no_mxl_candidates",
                    },
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            status_dir = root / "status"
            status_dir.mkdir()
            (status_dir / "page_0001.json").write_text(
                json.dumps(
                    {
                        "page": 1,
                        "attempts": 2,
                        "success": True,
                        "recovered_on_retry": True,
                        "log_paths": [],
                    }
                ),
                encoding="utf-8",
            )
            (status_dir / "page_0002.json").write_text(
                json.dumps(
                    {
                        "page": 2,
                        "attempts": 2,
                        "success": False,
                        "reason": "no_mxl_candidates",
                        "log_paths": [],
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.build_report(
                str(manifest_path),
                str(status_dir),
                str(root / "report.json"),
                False,
            )
            self.assertEqual(report["successful_pages"], [1])
            self.assertEqual(report["failed_pages"], [2])
            self.assertEqual(report["recovered_on_retry_pages"], [1])
            self.assertEqual(
                report["page_results"][1]["log_artifact"],
                "artifacts/page_omr_logs/page_0002.log",
            )

    def test_report_can_return_all_pages_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"page_number": 1, "status": "missing", "error": "no_mxl_candidates"},
                            {"page_number": 2, "status": "missing", "error": "parts_or_measures_zero"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            status_dir = root / "status"
            status_dir.mkdir()
            for page in (1, 2):
                (status_dir / f"page_{page:04d}.json").write_text(
                    json.dumps(
                        {
                            "page": page,
                            "attempts": 2,
                            "success": False,
                            "reason": "no_mxl_candidates",
                            "log_paths": [],
                        }
                    ),
                    encoding="utf-8",
                )
            report = MODULE.build_report(
                str(manifest_path),
                str(status_dir),
                str(root / "report.json"),
                False,
            )
            self.assertEqual(report["successful_pages"], [])
            self.assertEqual(report["failed_pages"], [1, 2])
            self.assertEqual(report["total_pages"], 2)


if __name__ == "__main__":
    unittest.main()
