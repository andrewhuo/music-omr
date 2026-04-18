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
        auth = types.ModuleType("google.auth")
        transport = types.ModuleType("google.auth.transport")
        requests_mod = types.ModuleType("google.auth.transport.requests")

        class _DummyClient:
            pass

        class _DummyCreds:
            valid = True
            expired = False
            token = "token"
            service_account_email = "service@example.com"

            def refresh(self, _request):
                self.token = "token"

        class _DummyRequest:
            pass

        storage.Client = _DummyClient
        auth.default = lambda scopes=None: (_DummyCreds(), None)
        requests_mod.Request = _DummyRequest
        transport.requests = requests_mod
        cloud.storage = storage
        google.cloud = cloud
        google.auth = auth
        sys.modules["google"] = google
        sys.modules["google.cloud"] = cloud
        sys.modules["google.cloud.storage"] = storage
        sys.modules["google.auth"] = auth
        sys.modules["google.auth.transport"] = transport
        sys.modules["google.auth.transport.requests"] = requests_mod

    if "flask" not in sys.modules:
        flask = types.ModuleType("flask")

        class _DummyApp:
            def __init__(self, name):
                self.name = name

            def route(self, *_args, **_kwargs):
                def _decorator(fn):
                    return fn

                return _decorator

            def before_request(self, fn=None, *_args, **_kwargs):
                if callable(fn):
                    return fn

                def _decorator(inner):
                    return inner

                return _decorator

            def after_request(self, fn=None, *_args, **_kwargs):
                if callable(fn):
                    return fn

                def _decorator(inner):
                    return inner

                return _decorator

            def make_response(self, payload):
                return payload

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
            "labels_mode": "system_only",
            "systems": [
                {"system_id": "p1_s2", "page": 1, "system_index": 2, "current_value": "7"},
                {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1"},
                {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "4"},
                {"system_id": "p2_s0", "page": 2, "system_index": 0, "current_value": "10"},
            ],
            "measures": [
                {"measure_id": "p1_s0_m0", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "y_top": 10},
                {"measure_id": "p1_s0_m1", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 1, "x_left": 30, "y_top": 10},
                {"measure_id": "p1_s0_m2", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 2, "x_left": 50, "y_top": 10},
                {"measure_id": "p1_s1_m0", "system_id": "p1_s1", "page": 1, "system_index": 1, "measure_local_index": 0, "x_left": 10, "y_top": 50},
                {"measure_id": "p1_s1_m1", "system_id": "p1_s1", "page": 1, "system_index": 1, "measure_local_index": 1, "x_left": 30, "y_top": 50},
                {"measure_id": "p1_s1_m2", "system_id": "p1_s1", "page": 1, "system_index": 1, "measure_local_index": 2, "x_left": 50, "y_top": 50},
                {"measure_id": "p1_s2_m0", "system_id": "p1_s2", "page": 1, "system_index": 2, "measure_local_index": 0, "x_left": 10, "y_top": 90},
                {"measure_id": "p1_s2_m1", "system_id": "p1_s2", "page": 1, "system_index": 2, "measure_local_index": 1, "x_left": 30, "y_top": 90},
                {"measure_id": "p1_s2_m2", "system_id": "p1_s2", "page": 1, "system_index": 2, "measure_local_index": 2, "x_left": 50, "y_top": 90},
                {"measure_id": "p2_s0_m0", "system_id": "p2_s0", "page": 2, "system_index": 0, "measure_local_index": 0, "x_left": 10, "y_top": 10},
                {"measure_id": "p2_s0_m1", "system_id": "p2_s0", "page": 2, "system_index": 0, "measure_local_index": 1, "x_left": 30, "y_top": 10},
                {"measure_id": "p2_s0_m2", "system_id": "p2_s0", "page": 2, "system_index": 0, "measure_local_index": 2, "x_left": 50, "y_top": 10},
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

    def test_measure_anchor_reflows_forward(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_measure_number", "measure_id": "p1_s1_m1", "value": 20}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_measure_number", "measure_id": "p1_s1_m1", "value": 20}])
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 22, 25])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m0"], 4)
        self.assertEqual(measure_values["p1_s1_m1"], 20)
        self.assertEqual(measure_values["p1_s1_m2"], 21)
        self.assertEqual(measure_values["p1_s2_m0"], 22)
        self.assertEqual(state.get("staff_boxes"), [])

    def test_clear_measure_number_removes_override_and_restores_sequence(self):
        state = self._sample_state()
        state["measure_number_overrides"] = {"p1_s1_m1": 20}
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "clear_measure_number", "measure_id": "p1_s1_m1"}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "clear_measure_number", "measure_id": "p1_s1_m1"}])
        self.assertEqual(state.get("measure_number_overrides"), {})
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 10])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m0"], 4)
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 6)
        self.assertEqual(measure_values["p1_s2_m0"], 7)

    def test_clear_measure_number_unknown_measure_rejected(self):
        state = self._sample_state()
        state["measure_number_overrides"] = {"p1_s1_m1": 20}
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "clear_measure_number", "measure_id": "missing"}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_measure_id")
        self.assertEqual(state.get("measure_number_overrides"), {"p1_s1_m1": 20})
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 22, 25])

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

    def test_unknown_measure_rejected(self):
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            self._sample_state(),
            [{"type": "set_measure_number", "measure_id": "missing", "value": 12}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_measure_id")
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

    def test_set_labels_mode_applied(self):
        state = self._sample_state()
        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_labels_mode", "value": "all_measures"}],
        )
        self.assertEqual(total, 4)
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_labels_mode", "value": "all_measures"}])
        self.assertEqual(state.get("labels_mode"), "all_measures")
        self.assertEqual(len(systems), 4)

    def test_invalid_labels_mode_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_labels_mode", "value": "bad_mode"}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "invalid_value")
        self.assertEqual(state.get("labels_mode"), "system_only")

    def test_mixed_edits_apply_mode_and_system_start(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_labels_mode", "value": "all_measures"},
                {"type": "set_system_start", "system_id": "p1_s1", "value": 20},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 2)
        self.assertEqual(state.get("labels_mode"), "all_measures")
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 20, 23, 26])

    def test_set_rest_staff_applied_and_reflows_following_systems(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_staff", "system_id": "p1_s1", "value": 2}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_rest_staff", "system_id": "p1_s1", "value": 2}])
        self.assertEqual(state.get("rest_systems"), {"p1_s1": 2})
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 9, 12])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s2_m0"], 9)
        self.assertEqual(measure_values["p2_s0_m0"], 12)

    def test_set_rest_staff_invalid_measure_count_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_staff", "system_id": "p1_s1", "value": -1}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "invalid_measure_count")
        self.assertEqual(state.get("rest_systems"), None)

    def test_set_rest_measure_applied_reflows_from_exact_anchor(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 3}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 3}])
        self.assertEqual(state.get("rest_measures"), {"p1_s1_m1": 3})
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 10, 13])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 9)
        self.assertEqual(measure_values["p1_s2_m0"], 10)
        self.assertEqual(measure_values["p2_s0_m0"], 13)

    def test_set_rest_measure_replaces_saved_count(self):
        state = self._sample_state()
        state["rest_measures"] = {"p1_s1_m1": 2}
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 5}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 5}])
        self.assertEqual(state.get("rest_measures"), {"p1_s1_m1": 5})

    def test_set_rest_measure_zero_removes_existing_value(self):
        state = self._sample_state()
        state["rest_measures"] = {"p1_s1_m1": 2}
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 0}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 0}])
        self.assertEqual(state.get("rest_measures"), {})

    def test_set_rest_measure_missing_measure_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_measure", "value": 2}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "missing_measure_id")
        self.assertEqual(state.get("rest_measures"), {})

    def test_set_rest_measure_unknown_measure_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_rest_measure", "measure_id": "missing", "value": 2}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_measure_id")
        self.assertEqual(state.get("rest_measures"), {})

    def test_set_rest_measure_rejects_invalid_or_negative_count(self):
        for bad_value in ("x", -1):
            with self.subTest(bad_value=bad_value):
                state = self._sample_state()
                _, applied, rejected, _ = WORKER._apply_relabel_edits(
                    state,
                    [{"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": bad_value}],
                )
                self.assertEqual(applied, [])
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0]["reason"], "invalid_measure_count")
                self.assertEqual(state.get("rest_measures"), {})

    def test_set_measure_number_and_rest_measure_compose_from_anchor_label(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_measure_number", "measure_id": "p1_s1_m1", "value": 20},
                {"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 3},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_measure_number", "measure_id": "p1_s1_m1", "value": 20},
                {"type": "set_rest_measure", "measure_id": "p1_s1_m1", "value": 3},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 25, 28])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], 20)
        self.assertEqual(measure_values["p1_s1_m2"], 24)
        self.assertEqual(measure_values["p1_s2_m0"], 25)
        self.assertEqual(measure_values["p2_s0_m0"], 28)

    def test_exact_rest_measure_wins_over_legacy_rest_staff_on_same_system(self):
        state = self._sample_state()
        state["rest_systems"] = {"p1_s1": 2}
        state["rest_measures"] = {"p1_s1_m1": 3}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 10, 13])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s2_m0"], 10)
        self.assertEqual(measure_values["p2_s0_m0"], 13)

    def test_set_ending_applied_recomputes_labels(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        self.assertEqual(state.get("endings"), {"p1_s1_m1": "1", "p1_s2_m0": "2"})
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 5, 9])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 6)
        self.assertEqual(measure_values["p1_s2_m0"], 5)
        self.assertEqual(measure_values["p1_s2_m1"], 7)

    def test_set_ending_clear_removes_existing_value(self):
        state = self._sample_state()
        state["endings"] = {"p1_s1_m1": "1"}
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_ending", "measure_id": "p1_s1_m1", "value": "none"}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_ending", "measure_id": "p1_s1_m1", "value": "none"}])
        self.assertEqual(state.get("endings"), {})

    def test_set_ending_unknown_measure_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_ending", "measure_id": "missing", "value": "1"}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_measure_id")

    def test_set_ending_invalid_value_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_ending", "measure_id": "p1_s1_m1", "value": "3"}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "invalid_ending_value")

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
