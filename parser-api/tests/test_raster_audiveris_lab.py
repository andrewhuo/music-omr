import importlib.util
import unittest

from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "raster_audiveris_lab.py"
SPEC = importlib.util.spec_from_file_location("raster_audiveris_lab", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
except ModuleNotFoundError:
    CV_LIBS_AVAILABLE = False
else:
    CV_LIBS_AVAILABLE = True


@unittest.skipUnless(CV_LIBS_AVAILABLE, "OpenCV and NumPy are not installed")
class RasterFallbackTests(unittest.TestCase):
    def _staff_image(self, gap=None):
        import numpy as np
        import cv2

        image = np.full((360, 1000), 255, dtype=np.uint8)
        for base in (60, 180):
            for offset in range(5):
                y = base + offset * 12
                cv2.line(image, (40, y), (960, y), 0, 2)
                if gap and base == 60 and offset == 2:
                    image[y - 1:y + 2, gap[0]:gap[1]] = 255
        return image

    def test_detects_five_line_staff_groups(self):
        rows, _, _ = MODULE._staff_rows(self._staff_image())
        self.assertGreaterEqual(len(rows), 10)

    def test_repairs_only_small_staff_line_gap(self):
        image = self._staff_image(gap=(420, 428))
        rows, binary, _ = MODULE._staff_rows(image)
        cleaned, repair = MODULE._repair_staff_rows(image, binary, rows)
        self.assertGreater(repair["repaired_gap_count"], 0)
        self.assertLess(int(cleaned[84:87, 420:428].max()), 100)

    def test_large_gap_is_not_filled(self):
        image = self._staff_image(gap=(420, 500))
        rows, binary, _ = MODULE._staff_rows(image)
        cleaned, repair = MODULE._repair_staff_rows(image, binary, rows)
        self.assertEqual(repair["repaired_gap_count"], 0)
        self.assertGreater(int(cleaned[84:87, 420:500].min()), 200)

    def test_matching_segment_lengths_and_endpoints_form_staff_group(self):
        segments = []
        for y in (60, 72, 84, 96, 108):
            segments.append({"left": 40, "right": 960, "y": y, "thickness": 2.0, "length": 921})
        groups, rows, _ = MODULE._group_staff_segments(segments, 1000, 360)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(rows), 5)

    def test_mismatched_endpoints_are_rejected(self):
        segments = []
        for index, y in enumerate((60, 72, 84, 96, 108)):
            segments.append({
                "left": 40 if index < 3 else 700,
                "right": 960 if index < 3 else 760,
                "y": y,
                "thickness": 2.0,
                "length": 921 if index < 3 else 61,
            })
        groups, rows, _ = MODULE._group_staff_segments(segments, 1000, 360)
        self.assertEqual(groups, [])
        self.assertEqual(rows, [])

    def test_conflicting_angles_are_rejected(self):
        segments = []
        for index, y in enumerate((60, 72, 84, 96, 108)):
            segments.append({
                "left": 40,
                "right": 960,
                "y": y,
                "thickness": 2.0,
                "length": 921,
                "angle": 1.0 if index == 2 else 0.0,
            })
        groups, rows, _ = MODULE._group_staff_segments(segments, 1000, 360)
        self.assertEqual(groups, [])
        self.assertEqual(rows, [])

    def test_isolated_horizontal_marks_are_not_staff_group(self):
        segments = [
            {"left": 40, "right": 960, "y": y, "thickness": 2.0, "length": 921, "angle": 0.0}
            for y in (60, 75, 93, 116)
        ]
        groups, rows, _ = MODULE._group_staff_segments(segments, 1000, 360)
        self.assertEqual(groups, [])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
