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
