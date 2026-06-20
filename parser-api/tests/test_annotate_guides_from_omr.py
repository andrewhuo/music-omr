import importlib.util
import os
import sys
import types
import unittest
import xml.etree.ElementTree as ET


def _install_import_stubs():
    fitz_mod = sys.modules.setdefault("fitz", types.ModuleType("fitz"))
    setattr(fitz_mod, "Page", type("Page", (), {}))
    setattr(fitz_mod, "Rect", type("Rect", (), {}))
    setattr(fitz_mod, "Matrix", type("Matrix", (), {}))
    setattr(fitz_mod, "get_text_length", lambda *args, **kwargs: 0.0)
    setattr(fitz_mod, "open", lambda *args, **kwargs: None)
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    numpy_mod = sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    setattr(numpy_mod, "ndarray", object)
    lxml_mod = sys.modules.setdefault("lxml", types.ModuleType("lxml"))
    etree_mod = sys.modules.setdefault("lxml.etree", types.ModuleType("lxml.etree"))
    setattr(lxml_mod, "etree", etree_mod)


def _load_module():
    _install_import_stubs()
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "annotate_guides_from_omr.py")
    )
    spec = importlib.util.spec_from_file_location("annotate_guides_from_omr_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


MOD = _load_module()


def _barline_el(el_id: int, staff: str, x: float, y_top: float, y_bottom: float, width: float = 0.0) -> ET.Element:
    el = ET.Element("barline", {"id": str(el_id), "staff": staff})
    ET.SubElement(
        el,
        "bounds",
        {
            "x": str(float(x)),
            "y": str(float(y_top)),
            "w": str(float(width)),
            "h": str(float(y_bottom - y_top)),
        },
    )
    med = ET.SubElement(el, "median")
    ET.SubElement(med, "p1", {"x": str(float(x)), "y": str(float(y_top))})
    ET.SubElement(med, "p2", {"x": str(float(x)), "y": str(float(y_bottom))})
    return el


class AnnotateGuidesFromOmrTests(unittest.TestCase):
    def test_grand_staff_connector_suppression_detects_tiny_shared_left_box(self):
        rows = [
            {
                "x_start": 30.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 520.0,
                "barline_xs": [62.0, 202.0, 342.0, 482.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 30.4,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 520.0,
                "barline_xs": [63.0, 203.0, 343.0, 483.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)
        starts_without_drop = MOD._build_measure_starts_for_system(rows, 30.0, 540.0)
        starts_with_drop = MOD._build_measure_starts_for_system(rows, 30.0, 540.0, drop_x_start=True)

        self.assertTrue(info["suppress"])
        self.assertEqual(info["reason"], "shared_small_left_connector_relative")
        self.assertAlmostEqual(info["next_3_median_width"], 140.0)
        self.assertLess(info["first_gap_ratio"], 0.25)
        self.assertEqual(starts_without_drop["measure_starts"], [30.0, 62.0, 202.0, 342.0, 482.0])
        self.assertEqual(starts_with_drop["measure_starts"], [62.0, 202.0, 342.0, 482.0])

    def test_grand_staff_connector_suppression_skips_normal_grand_staff(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 150.0,
                "barline_xs": [38.0, 72.0, 106.0, 140.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 10.0,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 150.0,
                "barline_xs": [38.2, 72.2, 106.2, 140.2],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)

        self.assertFalse(info["suppress"])
        self.assertEqual(info["reason"], "first_gap_not_small_enough_relative")

    def test_grand_staff_connector_suppression_respects_scaled_cap(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 150.0,
                "barline_xs": [60.0, 300.0, 540.0, 780.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 10.0,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 150.0,
                "barline_xs": [60.2, 300.2, 540.2, 780.2],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)

        self.assertFalse(info["suppress"])
        self.assertEqual(info["reason"], "first_gap_too_wide_absolute")

    def test_grand_staff_connector_suppression_does_not_run_on_one_staff(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 150.0,
                "barline_xs": [24.0, 70.0, 116.0, 162.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            }
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)

        self.assertFalse(info["suppress"])
        self.assertEqual(info["reason"], "not_multi_staff")

    def test_multi_staff_connector_suppression_detects_score_tiny_left_box(self):
        rows = [
            {
                "x_start": 30.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 520.0,
                "barline_xs": [62.0, 202.0, 342.0, 482.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 30.3,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 520.0,
                "barline_xs": [62.5, 202.5, 342.5, 482.5],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 29.8,
                "y_top": 80.0,
                "y_bottom": 100.0,
                "staff_right": 520.0,
                "barline_xs": [61.8, 201.8, 341.8, 481.8],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 30.2,
                "y_top": 110.0,
                "y_bottom": 130.0,
                "staff_right": 520.0,
                "barline_xs": [62.3, 202.3, 342.3, 482.3],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)
        starts_with_drop = MOD._build_measure_starts_for_system(rows, 30.0, 540.0, drop_x_start=True)

        self.assertTrue(info["suppress"])
        self.assertEqual(info["reason"], "shared_small_left_connector_relative")
        self.assertLess(info["first_gap_ratio"], 0.25)
        self.assertEqual(starts_with_drop["measure_starts"], [61.8, 201.8, 341.8, 481.8])

    def test_multi_staff_connector_suppression_skips_score_normal_first_measure(self):
        rows = [
            {
                "x_start": 30.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 520.0,
                "barline_xs": [96.0, 202.0, 342.0, 482.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 30.5,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 520.0,
                "barline_xs": [96.5, 202.5, 342.5, 482.5],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 29.8,
                "y_top": 80.0,
                "y_bottom": 100.0,
                "staff_right": 520.0,
                "barline_xs": [95.8, 201.8, 341.8, 481.8],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)

        self.assertFalse(info["suppress"])
        self.assertEqual(info["reason"], "first_gap_too_wide_absolute")

    def test_grand_staff_connector_suppression_keeps_staves_aligned(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 150.0,
                "barline_xs": [24.0, 70.0, 116.0, 162.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 10.5,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 150.0,
                "barline_xs": [24.8, 70.6, 116.4, 162.2],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        starts = MOD._build_measure_starts_for_system(rows, 10.0, 220.0, drop_x_start=True)

        self.assertEqual(starts["measure_starts"], [24.0, 70.0, 116.0, 162.0])

    def test_grand_staff_connector_suppression_requires_enough_later_measures(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 150.0,
                "barline_xs": [24.0, 70.0, 116.0],
                "barline_count": 3,
                "y_source": "staff_lines",
            },
            {
                "x_start": 10.5,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 150.0,
                "barline_xs": [24.8, 70.6, 116.4],
                "barline_count": 3,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)

        self.assertFalse(info["suppress"])
        self.assertEqual(info["reason"], "not_enough_later_measures")

    def test_grand_staff_connector_suppression_requires_aligned_first_barlines(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 150.0,
                "barline_xs": [24.0, 70.0, 116.0, 162.0],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
            {
                "x_start": 10.5,
                "y_top": 50.0,
                "y_bottom": 70.0,
                "staff_right": 150.0,
                "barline_xs": [36.8, 70.6, 116.4, 162.2],
                "barline_count": 4,
                "y_source": "staff_lines",
            },
        ]

        info = MOD._detect_grand_staff_connector_suppression(rows, 10.0)

        self.assertFalse(info["suppress"])
        self.assertEqual(info["reason"], "first_barline_misaligned")

    def test_system_x_debug_payload_includes_staff_and_final_measure_x(self):
        entry = {
            "selected_measure_rows": [{"x_start": 10.0}, {"x_start": 10.5}],
            "staff_contexts": [
                {
                    "staff_id": "1",
                    "line_min_x": 31.2345,
                    "line_max_x": 501.9876,
                    "header_start": 28.0,
                    "staff_left": 30.0,
                    "staff_right": 505.0,
                    "x_start": 24.0,
                    "y_top": 20.0,
                    "y_bottom": 40.0,
                    "y_source": "staff_lines",
                    "candidate_source": "staff_ids",
                    "barline_count": 4,
                    "barline_xs": [62.0, 202.0, 342.0, 482.0],
                },
                {
                    "staff_id": "2",
                    "line_min_x": 32.0,
                    "line_max_x": 502.0,
                    "header_start": 28.5,
                    "staff_left": 30.5,
                    "staff_right": 505.5,
                    "x_start": 24.5,
                    "y_top": 50.0,
                    "y_bottom": 70.0,
                    "y_source": "staff_lines",
                    "candidate_source": "staff_ids_plus_overlap",
                    "barline_count": 4,
                    "barline_xs": [62.5, 202.5, 342.5, 482.5],
                },
            ],
        }
        chosen_starts_info = {
            "measure_starts": [62.0, 202.0, 342.0, 482.0],
            "row_tail": 520.0,
        }

        payload = MOD._system_x_debug_payload(entry, chosen_starts_info)

        self.assertEqual(payload["staff_count"], 2)
        self.assertEqual(payload["selected_staff_count"], 2)
        self.assertEqual(payload["final_measure_starts"], [62.0, 202.0, 342.0, 482.0])
        self.assertEqual(payload["final_measure_right_edge"], 520.0)
        self.assertEqual(payload["staffs"][0]["line_min_x"], 31.235)
        self.assertEqual(payload["staffs"][0]["x_postpad"], 24.0)
        self.assertEqual(payload["staffs"][1]["candidate_source"], "staff_ids_plus_overlap")
        self.assertEqual(payload["staffs"][1]["barline_xs_preview"], [62.5, 202.5, 342.5, 482.5])

    def test_coordinate_trace_payload_includes_pixel_stages(self):
        entry = {
            "page_id": "1",
            "system_id": "1",
            "system_index": 0,
            "system_measure_rows_px": [
                {
                    "measure_local_index": 0,
                    "x_left": 10.0,
                    "x_right": 20.0,
                    "x_start": 10.0,
                    "y_top": 30.0,
                    "y_bottom": 80.0,
                    "barline_xs": [25.0, 60.0, 95.0, 130.0],
                }
            ],
            "first_pass_starts_info": {
                "measure_starts": [10.0, 25.0, 60.0, 95.0, 130.0],
                "system_y_top": 30.0,
                "system_y_bottom": 80.0,
                "row_tail": 160.0,
            },
        }
        chosen_starts_info = {
            "measure_starts": [25.0, 60.0, 95.0, 130.0],
            "system_y_top": 30.0,
            "system_y_bottom": 80.0,
            "row_tail": 160.0,
        }

        payload = MOD._coordinate_trace_payload(entry, chosen_starts_info)

        self.assertEqual(payload["raw_measure_starts_px"], [10.0, 25.0, 60.0, 95.0, 130.0])
        self.assertEqual(payload["adjusted_measure_starts_px"], [25.0, 60.0, 95.0, 130.0])
        self.assertEqual(payload["pixel_measure_box_count"], 4)
        self.assertEqual(payload["pixel_final_right_edge"], 160.0)
        self.assertEqual(payload["pixel_measure_boxes_preview"][0]["x_left"], 25.0)
        self.assertEqual(payload["pixel_measure_boxes_preview"][-1]["x_right"], 160.0)

    def test_coordinate_debug_image_filename_shape(self):
        path = "/tmp/work/coordinate_debug/coordinate_debug_page_12.png"
        row = {
            "page": 12,
            "path": path,
            "filename": os.path.basename(path),
            "written": True,
        }

        self.assertEqual(row["filename"], "coordinate_debug_page_12.png")
        self.assertTrue(row["written"])

    def test_omr_to_pdf_point_without_crop_offset_matches_old_math(self):
        class FakeRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0 = x0
                self.y0 = y0
                self.x1 = x1
                self.y1 = y1
                self.width = x1 - x0
                self.height = y1 - y0

        class FakePage:
            rect = FakeRect(0.0, 0.0, 200.0, 100.0)
            cropbox = FakeRect(0.0, 0.0, 200.0, 100.0)

        x_pdf, y_pdf = MOD._omr_to_pdf_point(20.0, 30.0, 0.5, 0.5, FakePage())

        self.assertEqual(x_pdf, 10.0)
        self.assertEqual(y_pdf, 15.0)

    def test_omr_to_pdf_point_adds_crop_offset(self):
        class FakeRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0 = x0
                self.y0 = y0
                self.x1 = x1
                self.y1 = y1
                self.width = x1 - x0
                self.height = y1 - y0

        class FakePage:
            rect = FakeRect(0.0, 0.0, 200.0, 100.0)
            cropbox = FakeRect(10.0, 20.0, 210.0, 120.0)

        x_pdf, y_pdf = MOD._omr_to_pdf_point(20.0, 30.0, 0.5, 0.5, FakePage())

        self.assertEqual(x_pdf, 20.0)
        self.assertEqual(y_pdf, 35.0)

    def test_crop_offset_keeps_converted_box_size_same(self):
        class FakeRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0 = x0
                self.y0 = y0
                self.x1 = x1
                self.y1 = y1
                self.width = x1 - x0
                self.height = y1 - y0

        class FakePage:
            rect = FakeRect(0.0, 0.0, 200.0, 100.0)
            cropbox = FakeRect(10.0, 20.0, 210.0, 120.0)

        x_left, y_top = MOD._omr_to_pdf_point(20.0, 30.0, 0.5, 0.5, FakePage())
        x_right, y_bottom = MOD._omr_to_pdf_point(80.0, 90.0, 0.5, 0.5, FakePage())

        self.assertEqual(x_right - x_left, 30.0)
        self.assertEqual(y_bottom - y_top, 30.0)

    def test_page_box_debug_payload_includes_boxes_and_conversion_samples(self):
        class FakeRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0 = x0
                self.y0 = y0
                self.x1 = x1
                self.y1 = y1
                self.width = x1 - x0
                self.height = y1 - y0

        class FakePage:
            rect = FakeRect(0.0, 0.0, 200.0, 100.0)
            cropbox = FakeRect(10.0, 20.0, 210.0, 120.0)
            mediabox = FakeRect(0.0, 0.0, 220.0, 130.0)
            rotation = 0

        payload = MOD._page_box_debug_payload(
            FakePage(),
            400.0,
            200.0,
            0.5,
            0.5,
            [
                {
                    "measure_local_index": 0,
                    "x_left": 20.0,
                    "x_right": 80.0,
                    "y_top": 30.0,
                    "y_bottom": 90.0,
                }
            ],
        )

        self.assertEqual(payload["rect"]["width"], 200.0)
        self.assertEqual(payload["cropbox"]["x0"], 10.0)
        self.assertEqual(payload["mediabox"]["height"], 130.0)
        self.assertEqual(payload["picture_width"], 400.0)
        self.assertEqual(payload["conversion_mode"], "cropbox_offset")
        self.assertEqual(payload["sample_current_pdf_boxes"][0]["x_left"], 20.0)
        self.assertEqual(payload["sample_current_pdf_boxes"][0]["y_top"], 35.0)
        self.assertEqual(payload["sample_rect_only_pdf_boxes"][0]["x_left"], 10.0)
        self.assertEqual(payload["sample_cropbox_offset_pdf_boxes"][0]["x_left"], 20.0)

    def test_second_pass_can_supplement_incomplete_staff_barline_ids(self):
        system_inters = ET.Element("inters")
        bar1 = _barline_el(1, "1", 100.0, 10.0, 50.0)
        bar2 = _barline_el(2, "1", 160.0, 10.0, 50.0)
        bar3 = _barline_el(3, "1", 220.0, 10.0, 50.0)
        for bar in (bar1, bar2, bar3):
            system_inters.append(bar)
        inter_by_id = {1: bar1, 2: bar2, 3: bar3}

        first_xs, first_source = MOD._barline_xs_for_staff_from_omr(
            system_inters,
            inter_by_id,
            10.0,
            "1",
            10.0,
            50.0,
            ["1"],
            0.0,
            260.0,
        )
        second_xs, second_source = MOD._barline_xs_for_staff_from_omr(
            system_inters,
            inter_by_id,
            10.0,
            "1",
            10.0,
            50.0,
            ["1"],
            0.0,
            260.0,
            allow_supplement=True,
            relaxed=True,
        )

        self.assertEqual(first_source, "staff_ids")
        self.assertEqual(first_xs, [100.0])
        self.assertEqual(second_source, "staff_ids_plus_overlap")
        self.assertEqual(second_xs, [100.0, 160.0, 220.0])

    def test_relaxed_second_pass_merges_less_aggressively(self):
        system_inters = ET.Element("inters")
        bar1 = _barline_el(1, "1", 100.0, 10.0, 50.0)
        bar2 = _barline_el(2, "1", 101.4, 10.0, 50.0)
        for bar in (bar1, bar2):
            system_inters.append(bar)
        inter_by_id = {1: bar1, 2: bar2}

        first_xs, _ = MOD._barline_xs_for_staff_from_omr(
            system_inters,
            inter_by_id,
            10.0,
            "1",
            10.0,
            50.0,
            ["1", "2"],
            0.0,
            260.0,
        )
        second_xs, _ = MOD._barline_xs_for_staff_from_omr(
            system_inters,
            inter_by_id,
            10.0,
            "1",
            10.0,
            50.0,
            ["1", "2"],
            0.0,
            260.0,
            allow_supplement=True,
            relaxed=True,
        )

        self.assertEqual(first_xs, [100.0])
        self.assertEqual(second_xs, [100.0, 101.4])

    def test_under_split_detection_uses_neighbor_median(self):
        rows = [
            {"first_pass_measure_count": 10, "system_width": 400.0},
            {"first_pass_measure_count": 2, "system_width": 400.0},
            {"first_pass_measure_count": 9, "system_width": 400.0},
        ]

        neighbor_median = MOD._neighbor_median_measure_count(rows, 1)

        self.assertEqual(neighbor_median, 9.5)
        self.assertTrue(MOD._is_under_split_suspect(2, neighbor_median))
        self.assertFalse(MOD._is_under_split_suspect(10, MOD._neighbor_median_measure_count(rows, 0)))

    def test_second_pass_acceptance_requires_clear_improvement(self):
        self.assertTrue(MOD._should_accept_second_pass(2, 6, 10.0))
        self.assertFalse(MOD._should_accept_second_pass(2, 2, 10.0))
        self.assertFalse(MOD._should_accept_second_pass(2, 18, 10.0))
        self.assertFalse(MOD._should_accept_second_pass(2, 3, None))

    def test_build_measure_starts_for_system_uses_relaxed_merge_tolerance(self):
        rows = [
            {
                "x_start": 10.0,
                "y_top": 20.0,
                "y_bottom": 40.0,
                "staff_right": 90.0,
                "barline_xs": [30.0, 31.4, 60.0],
            }
        ]

        first_info = MOD._build_measure_starts_for_system(rows, 10.0, 200.0, relaxed=False)
        second_info = MOD._build_measure_starts_for_system(rows, 10.0, 200.0, relaxed=True)

        self.assertEqual(first_info["measure_starts"], [10.0, 30.0, 60.0])
        self.assertEqual(second_info["measure_starts"], [10.0, 30.0, 31.4, 60.0])


if __name__ == "__main__":
    unittest.main()
