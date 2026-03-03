import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_worker_module():
    if "fitz" not in sys.modules:
        fitz = types.ModuleType("fitz")
        fitz.get_text_length = lambda txt, fontsize=10.0: float(len(str(txt)) * 6.0)
        fitz.Rect = object
        fitz.Page = object
        sys.modules["fitz"] = fitz

    if "google.cloud.storage" not in sys.modules:
        google = sys.modules.get("google") or types.ModuleType("google")
        cloud = sys.modules.get("google.cloud") or types.ModuleType("google.cloud")
        storage = types.ModuleType("google.cloud.storage")

        class _DummyClient:
            pass

        storage.Client = _DummyClient
        cloud.storage = storage
        google.cloud = cloud
        sys.modules["google"] = google
        sys.modules["google.cloud"] = cloud
        sys.modules["google.cloud.storage"] = storage

    if "flask" not in sys.modules:
        flask = types.ModuleType("flask")

        class _DummyApp:
            def __init__(self, name):
                self.name = name

            def route(self, *_args, **_kwargs):
                def _decorator(fn):
                    return fn

                return _decorator

            def run(self, *args, **kwargs):
                return None

        flask.Flask = _DummyApp
        flask.jsonify = lambda payload, *args, **kwargs: payload
        flask.request = types.SimpleNamespace(json={})
        sys.modules["flask"] = flask

    worker_path = Path(__file__).resolve().parents[1] / "worker.py"
    spec = importlib.util.spec_from_file_location("omr_worker_worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


WORKER = _load_worker_module()


class RelabelLogicTests(unittest.TestCase):
    def _sample_state(self):
        return {
            "version": "system_state_v1",
            "systems": [
                {"system_id": "p1_s2", "page": 1, "system_index": 2, "current_value": "7"},
                {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1"},
                {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "4"},
                {"system_id": "p2_s0", "page": 2, "system_index": 0, "current_value": "10"},
            ],
        }

    def test_single_edit_reflows_forward(self):
        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            self._sample_state(),
            [{"type": "set_system_start", "system_id": "p1_s1", "value": 20}],
        )
        self.assertEqual(total, 4)
        self.assertEqual(len(applied), 1)
        self.assertEqual(rejected, [])
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 20, 23, 26])

    def test_multiple_edits_deterministic_order(self):
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            self._sample_state(),
            [
                {"type": "set_system_start", "system_id": "p1_s1", "value": 20},
                {"type": "set_system_start", "system_id": "p2_s0", "value": 50},
            ],
        )
        self.assertEqual(len(applied), 2)
        self.assertEqual(rejected, [])
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 20, 23, 50])

    def test_unknown_system_rejected(self):
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            self._sample_state(),
            [{"type": "set_system_start", "system_id": "p9_s9", "value": 3}],
        )
        self.assertEqual(len(applied), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_system_id")
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 10])

    def test_value_range_rejected(self):
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            self._sample_state(),
            [{"type": "set_system_start", "system_id": "p1_s0", "value": -1}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "value_out_of_range")
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 10])

    def test_sorted_by_page_and_system(self):
        systems, _, _, _ = WORKER._apply_relabel_edits(self._sample_state(), [])
        ordering = [(row["page"], row["system_index"]) for row in systems]
        self.assertEqual(ordering, [(1, 0), (1, 1), (1, 2), (2, 0)])

    def test_relabel_trace_history_cap(self):
        mapping_summary = {}
        for idx in range(55):
            WORKER._append_relabel_trace(
                mapping_summary,
                {
                    "trace_id": f"t{idx}",
                    "result": "success",
                    "rejected_reason_counts": {},
                },
                max_history=50,
            )
        relabel_debug = mapping_summary.get("relabel_debug") or {}
        history = relabel_debug.get("history") or []
        self.assertEqual(len(history), 50)
        self.assertEqual(history[0].get("trace_id"), "t5")
        self.assertEqual((relabel_debug.get("last_trace") or {}).get("trace_id"), "t54")

    def test_relabel_trace_reason_counts_aggregate(self):
        mapping_summary = {}
        WORKER._append_relabel_trace(
            mapping_summary,
            {
                "trace_id": "one",
                "result": "validation_error",
                "reason": "invalid_payload",
                "rejected_reason_counts": {"unknown_system_id": 2},
            },
            max_history=50,
        )
        WORKER._append_relabel_trace(
            mapping_summary,
            {
                "trace_id": "two",
                "result": "stale_conflict",
                "reason": "stale_run_mismatch",
                "rejected_reason_counts": {},
            },
            max_history=50,
        )
        summary = WORKER._summarize_relabel_debug(mapping_summary)
        self.assertEqual(summary["history_count"], 2)
        self.assertEqual(summary["reason_counts"].get("invalid_payload"), 1)
        self.assertEqual(summary["reason_counts"].get("unknown_system_id"), 2)
        self.assertEqual(summary["reason_counts"].get("stale_run_mismatch"), 1)


if __name__ == "__main__":
    unittest.main()
