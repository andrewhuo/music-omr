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

    def _sample_state_with_bounds(self):
        state = self._sample_state()
        anchor_map = {
            "p1_s0": {"x": 10, "y_top": 10, "y_bottom": 20},
            "p1_s1": {"x": 10, "y_top": 50, "y_bottom": 60},
            "p1_s2": {"x": 10, "y_top": 90, "y_bottom": 100},
            "p2_s0": {"x": 10, "y_top": 10, "y_bottom": 20},
        }
        for system in state.get("systems") or []:
            system["anchor"] = dict(anchor_map.get(system["system_id"]) or {})
            system["source"] = "auto"
        for measure in state.get("measures") or []:
            measure["x_right"] = int(measure["x_left"]) + 15
            measure["y_bottom"] = int(measure["y_top"]) + 10
            measure["source"] = "auto"
        return state

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

    def test_remove_label_area_persists(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {
                    "type": "remove_label_area",
                    "page": 1,
                    "rect": {"left": 8, "right": 42, "top": 4, "bottom": 22},
                }
            ],
        )

        self.assertEqual(rejected, [])
        self.assertEqual(applied[0]["type"], "remove_label_area")
        self.assertEqual(state.get("label_erase_areas"), [{"page": 1, "rect": {"left": 8.0, "right": 42.0, "top": 4.0, "bottom": 22.0}}])

    def test_remove_label_area_rejects_invalid_rect(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "remove_label_area", "page": 0, "rect": {"left": 8, "right": 42, "top": 4, "bottom": 22}},
                {"type": "remove_label_area", "page": 1, "rect": {"left": 8, "right": 200, "top": 4, "bottom": 22}},
                {"type": "remove_label_area", "page": 1},
            ],
        )

        self.assertEqual(applied, [])
        self.assertEqual([row.get("reason") for row in rejected], ["invalid_label_erase_area", "invalid_label_erase_area", "invalid_label_erase_area"])
        self.assertEqual(state.get("label_erase_areas"), [])

    def test_hide_label_persists(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "hide_label", "value": "label:p1_s0_m0"}],
        )

        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "hide_label", "label_id": "label:p1_s0_m0", "measure_id": "p1_s0_m0"}])
        self.assertEqual(state.get("hidden_label_ids"), ["label:p1_s0_m0"])

    def test_hide_label_rejects_invalid_id(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "hide_label", "value": "p1_s0_m0"},
                {"type": "hide_label", "value": "label:not_a_measure"},
            ],
        )

        self.assertEqual(applied, [])
        self.assertEqual([row.get("reason") for row in rejected], ["invalid_label_id", "invalid_label_id"])
        self.assertEqual(state.get("hidden_label_ids"), [])

    def test_show_label_persists_and_restores_default_position(self):
        state = self._sample_state()
        state["hidden_label_ids"] = ["label:p1_s0_m1"]
        state["label_positions"] = {
            "label:p1_s0_m1": {"page": 1, "left": 42.25, "top": 13.75}
        }
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "show_label", "measure_id": "p1_s0_m1"},
                {"type": "show_label", "measure_id": "p1_s0_m1"},
            ],
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 2)
        self.assertEqual(state.get("forced_label_ids"), ["label:p1_s0_m1"])
        self.assertEqual(state.get("hidden_label_ids"), [])
        self.assertEqual(state.get("label_positions"), {})

    def test_show_label_rejects_invalid_measure(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "show_label", "measure_id": "missing"}],
        )

        self.assertEqual(applied, [])
        self.assertEqual([row.get("reason") for row in rejected], ["invalid_measure_id"])
        self.assertEqual(state.get("forced_label_ids"), [])

    def test_forced_label_adds_middle_measure_in_system_only_mode(self):
        state = self._sample_state()
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state["systems"], state["measures"], state
        )
        rows = WORKER._label_render_rows(
            "system_only",
            systems,
            measures,
            result_labels,
            {"label:p1_s0_m1"},
        )

        measure_ids = [row.get("measure_id") for row, _ in rows]
        self.assertIn("p1_s0_m0", measure_ids)
        self.assertIn("p1_s0_m1", measure_ids)
        self.assertEqual(measure_ids.count("p1_s0_m1"), 1)

    def test_forced_label_does_not_duplicate_all_measures_label(self):
        state = self._sample_state()
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state["systems"], state["measures"], state
        )
        rows = WORKER._label_render_rows(
            "all_measures",
            systems,
            measures,
            result_labels,
            {"label:p1_s0_m1"},
        )

        measure_ids = [row.get("measure_id") for row, _ in rows]
        self.assertEqual(measure_ids.count("p1_s0_m1"), 1)

    def test_forced_label_uses_recomputed_number(self):
        state = self._sample_state()
        WORKER._apply_relabel_edits(
            state,
            [
                {"type": "show_label", "measure_id": "p1_s0_m1"},
                {"type": "set_measure_number", "measure_id": "p1_s0_m1", "value": 42},
            ],
        )
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state["systems"], state["measures"], state
        )
        rows = WORKER._label_render_rows(
            "system_only",
            systems,
            measures,
            result_labels,
            set(state.get("forced_label_ids") or []),
        )

        rendered = {
            row.get("measure_id"): label
            for row, label in rows
        }
        self.assertEqual(rendered.get("p1_s0_m1"), "42")

    def test_forced_label_ids_drop_deleted_measures(self):
        state = {
            "forced_label_ids": ["label:keep", "label:deleted", "bad"],
        }
        cleaned = WORKER._editable_forced_label_ids(state, {"keep"})

        self.assertEqual(cleaned, ["label:keep"])

    def test_move_label_persists(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "move_label", "label_id": "label:p1_s0_m0", "page": 1, "left": 42.25, "top": 13.75}],
        )

        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {
                    "type": "move_label",
                    "label_id": "label:p1_s0_m0",
                    "measure_id": "p1_s0_m0",
                    "page": 1,
                    "left": 42.25,
                    "top": 13.75,
                }
            ],
        )
        self.assertEqual(state.get("label_positions"), {"label:p1_s0_m0": {"page": 1, "left": 42.25, "top": 13.75}})

    def test_move_label_rejects_invalid_id_or_position(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "move_label", "label_id": "p1_s0_m0", "page": 1, "left": 42, "top": 13},
                {"type": "move_label", "label_id": "label:not_a_measure", "page": 1, "left": 42, "top": 13},
                {"type": "move_label", "label_id": "label:p1_s0_m0", "page": 0, "left": 42, "top": 13},
                {"type": "move_label", "label_id": "label:p1_s0_m0", "page": 1, "left": -1, "top": 13},
            ],
        )

        self.assertEqual(applied, [])
        self.assertEqual(
            [row.get("reason") for row in rejected],
            ["invalid_label_id", "invalid_label_id", "invalid_label_position", "invalid_label_position"],
        )
        self.assertEqual(state.get("label_positions"), {})

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

    def test_replace_auto_rows_can_exclude_box_and_block_later_count_edit(self):
        state = self._sample_state_with_bounds()
        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            state,
            [
                {
                    "type": "replace_auto_rows_for_page",
                    "page": 1,
                    "rows": [
                        {
                            "system_id": "p1_s0",
                            "page": 1,
                            "rect": {"left": 10, "right": 65, "top": 10, "bottom": 20},
                            "boxes": [
                                {"measure_id": "p1_s0_m0", "left": 10, "right": 25, "excluded_from_counting": True},
                                {"measure_id": "p1_s0_m1", "left": 30, "right": 45, "excluded_from_counting": False},
                                {"measure_id": "p1_s0_m2", "left": 50, "right": 65, "excluded_from_counting": False},
                            ],
                        },
                        {
                            "system_id": "p1_s1",
                            "page": 1,
                            "rect": {"left": 10, "right": 65, "top": 50, "bottom": 60},
                            "boxes": [
                                {"measure_id": "p1_s1_m0", "left": 10, "right": 25, "excluded_from_counting": False},
                                {"measure_id": "p1_s1_m1", "left": 30, "right": 45, "excluded_from_counting": False},
                                {"measure_id": "p1_s1_m2", "left": 50, "right": 65, "excluded_from_counting": False},
                            ],
                        },
                        {
                            "system_id": "p1_s2",
                            "page": 1,
                            "rect": {"left": 10, "right": 65, "top": 90, "bottom": 100},
                            "boxes": [
                                {"measure_id": "p1_s2_m0", "left": 10, "right": 25, "excluded_from_counting": False},
                                {"measure_id": "p1_s2_m1", "left": 30, "right": 45, "excluded_from_counting": False},
                                {"measure_id": "p1_s2_m2", "left": 50, "right": 65, "excluded_from_counting": False},
                            ],
                        },
                    ],
                }
            ],
        )
        self.assertEqual(total, 4)
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "replace_auto_rows_for_page", "page": 1, "rows_count": 3}])
        excluded_measure = next(row for row in (state.get("measures") or []) if row.get("measure_id") == "p1_s0_m0")
        self.assertTrue(excluded_measure.get("excluded_from_counting"))
        self.assertEqual(str(excluded_measure.get("current_value") or ""), "")
        self.assertEqual(len(state.get("auto_rows") or []), 4)

        systems_after, applied_after, rejected_after, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_measure_number", "measure_id": "p1_s0_m0", "value": 99}],
        )
        self.assertEqual(applied_after, [])
        self.assertEqual(len(rejected_after), 1)
        self.assertEqual(rejected_after[0]["reason"], "measure_excluded_from_counting")
        self.assertEqual(len(systems_after), 4)

    def test_replace_auto_rows_can_resize_row_rect(self):
        state = self._sample_state_with_bounds()
        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            state,
            [
                {
                    "type": "replace_auto_rows_for_page",
                    "page": 1,
                    "rows": [
                        {
                            "system_id": "p1_s0",
                            "page": 1,
                            "rect": {"left": 8, "right": 70, "top": 8, "bottom": 24},
                            "boxes": [
                                {"measure_id": "p1_s0_m0", "left": 8, "right": 25, "excluded_from_counting": False},
                                {"measure_id": "p1_s0_m1", "left": 30, "right": 45, "excluded_from_counting": False},
                                {"measure_id": "p1_s0_m2", "left": 50, "right": 70, "excluded_from_counting": False},
                            ],
                        },
                        {
                            "system_id": "p1_s1",
                            "page": 1,
                            "rect": {"left": 10, "right": 65, "top": 50, "bottom": 60},
                            "boxes": [
                                {"measure_id": "p1_s1_m0", "left": 10, "right": 25, "excluded_from_counting": False},
                                {"measure_id": "p1_s1_m1", "left": 30, "right": 45, "excluded_from_counting": False},
                                {"measure_id": "p1_s1_m2", "left": 50, "right": 65, "excluded_from_counting": False},
                            ],
                        },
                        {
                            "system_id": "p1_s2",
                            "page": 1,
                            "rect": {"left": 10, "right": 65, "top": 90, "bottom": 100},
                            "boxes": [
                                {"measure_id": "p1_s2_m0", "left": 10, "right": 25, "excluded_from_counting": False},
                                {"measure_id": "p1_s2_m1", "left": 30, "right": 45, "excluded_from_counting": False},
                                {"measure_id": "p1_s2_m2", "left": 50, "right": 65, "excluded_from_counting": False},
                            ],
                        },
                    ],
                }
            ],
        )
        self.assertEqual(total, 4)
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "replace_auto_rows_for_page", "page": 1, "rows_count": 3}])
        auto_row = next(row for row in (state.get("auto_rows") or []) if row.get("system_id") == "p1_s0")
        self.assertEqual(auto_row.get("rect"), {"left": 8.0, "right": 70.0, "top": 8.0, "bottom": 24.0})
        resized_measure = next(row for row in (state.get("measures") or []) if row.get("measure_id") == "p1_s0_m0")
        self.assertEqual(resized_measure.get("x_left"), 8.0)
        self.assertEqual(resized_measure.get("y_top"), 8.0)
        self.assertEqual(resized_measure.get("y_bottom"), 24.0)
        self.assertEqual(len(systems), 4)

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
        self.assertEqual(values, [1, 4, 9, 12])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 8)
        self.assertEqual(measure_values["p1_s2_m0"], 9)
        self.assertEqual(measure_values["p2_s0_m0"], 12)

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

    def test_set_pickup_measure_saves_exact_anchor(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": True}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": True}])
        self.assertEqual(state.get("pickup_measures"), {"p1_s1_m1": True})

    def test_set_pickup_measure_replaces_existing_anchor_on_same_system(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m1": True}
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_pickup_measure", "measure_id": "p1_s1_m2", "value": True}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_pickup_measure", "measure_id": "p1_s1_m2", "value": True}])
        self.assertEqual(state.get("pickup_measures"), {"p1_s1_m2": True})

    def test_set_pickup_measure_allows_saved_anchors_on_different_systems(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": True},
                {"type": "set_pickup_measure", "measure_id": "p1_s2_m1", "value": True},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": True},
                {"type": "set_pickup_measure", "measure_id": "p1_s2_m1", "value": True},
            ],
        )
        self.assertEqual(state.get("pickup_measures"), {"p1_s1_m1": True, "p1_s2_m1": True})

    def test_set_pickup_measure_false_removes_existing_value(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m1": True}
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": False}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(applied, [{"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": False}])
        self.assertEqual(state.get("pickup_measures"), {})

    def test_set_pickup_measure_missing_measure_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_pickup_measure", "value": True}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "missing_measure_id")
        self.assertEqual(state.get("pickup_measures"), {})

    def test_set_pickup_measure_unknown_measure_rejected(self):
        state = self._sample_state()
        _, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [{"type": "set_pickup_measure", "measure_id": "missing", "value": True}],
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_measure_id")
        self.assertEqual(state.get("pickup_measures"), {})

    def test_set_pickup_measure_non_boolean_value_rejected(self):
        for bad_value in ("true", 1, 0, None):
            with self.subTest(bad_value=bad_value):
                state = self._sample_state()
                _, applied, rejected, _ = WORKER._apply_relabel_edits(
                    state,
                    [{"type": "set_pickup_measure", "measure_id": "p1_s1_m1", "value": bad_value}],
                )
                self.assertEqual(applied, [])
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0]["reason"], "invalid_value")
                self.assertEqual(state.get("pickup_measures"), {})

    def test_pickup_measure_in_middle_of_system_repeats_the_previous_number(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m1": True}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "6", "9"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m0"], "4")
        self.assertEqual(measure_values["p1_s1_m1"], "4")
        self.assertEqual(measure_values["p1_s1_m2"], "5")
        self.assertEqual(measure_values["p1_s2_m0"], "6")
        self.assertEqual(measure_values["p2_s0_m0"], "9")

    def test_pickup_measure_at_end_of_system_repeats_the_previous_number(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m2": True}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "6", "9"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], "5")
        self.assertEqual(measure_values["p1_s1_m2"], "5")
        self.assertEqual(measure_values["p1_s2_m0"], "6")
        self.assertEqual(measure_values["p2_s0_m0"], "9")

    def test_pickup_measure_at_start_of_system_repeats_the_previous_number(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m0": True}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "3", "6", "9"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m0"], "3")
        self.assertEqual(measure_values["p1_s1_m1"], "4")
        self.assertEqual(measure_values["p1_s1_m2"], "5")
        self.assertEqual(measure_values["p1_s2_m0"], "6")
        self.assertEqual(measure_values["p2_s0_m0"], "9")

    def test_opening_pickup_is_zero_and_the_next_measure_is_one(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s0_m0": True}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])

        self.assertEqual([str(row["current_value"]) for row in systems], ["0", "3", "6", "9"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s0_m0"], "0")
        self.assertEqual(measure_values["p1_s0_m1"], "1")
        self.assertEqual(measure_values["p1_s0_m2"], "2")

    def test_system_start_anchor_measures_uses_first_visible_numbered_measure(self):
        state = self._sample_state()
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state.get("systems"),
            state.get("measures"),
            state,
        )
        anchors = WORKER._system_start_anchor_measures(measures, result_labels)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_s0_m0", "1"),
                ("p1_s1_m0", "4"),
                ("p1_s2_m0", "7"),
                ("p2_s0_m0", "10"),
            ],
        )

    def test_system_start_anchor_measures_uses_pickup_at_start_of_system(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m0": True}
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state.get("systems"),
            state.get("measures"),
            state,
        )
        anchors = WORKER._system_start_anchor_measures(measures, result_labels)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_s0_m0", "1"),
                ("p1_s1_m0", "3"),
                ("p1_s2_m0", "6"),
                ("p2_s0_m0", "9"),
            ],
        )

    def test_system_start_anchor_measures_uses_pickup_and_rest_at_system_start(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m0": True}
        state["rest_measures"] = {"p1_s1_m0": 2}
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state.get("systems"),
            state.get("measures"),
            state,
        )
        anchors = WORKER._system_start_anchor_measures(measures, result_labels)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_s0_m0", "1"),
                ("p1_s1_m0", "3"),
                ("p1_s2_m0", "7"),
                ("p2_s0_m0", "10"),
            ],
        )

    def test_system_start_anchor_measures_uses_first_pickup_number(self):
        state = self._sample_state()
        state["pickup_measures"] = {
            "p1_s1_m0": True,
            "p1_s1_m1": True,
            "p1_s1_m2": True,
        }
        systems, measures, result_labels, _ = WORKER._recompute_measure_numbering(
            state.get("systems"),
            state.get("measures"),
            state,
        )
        anchors = WORKER._system_start_anchor_measures(measures, result_labels)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_s0_m0", "1"),
                ("p1_s1_m0", "3"),
                ("p1_s2_m0", "4"),
                ("p2_s0_m0", "7"),
            ],
        )

    def test_system_start_anchor_measures_groups_split_same_row_and_keeps_leftmost(self):
        systems = [
            {
                "system_id": "p1_left",
                "page": 1,
                "system_index": 0,
                "anchor": {"x": 10, "y_top": 40, "y_bottom": 60},
                "x_left": 10,
                "x_right": 65,
                "y_top": 40,
                "y_bottom": 60,
            },
            {
                "system_id": "p1_right",
                "page": 1,
                "system_index": 1,
                "anchor": {"x": 80, "y_top": 42, "y_bottom": 61},
                "x_left": 80,
                "x_right": 130,
                "y_top": 42,
                "y_bottom": 61,
            },
            {
                "system_id": "p1_next",
                "page": 1,
                "system_index": 2,
                "anchor": {"x": 10, "y_top": 90, "y_bottom": 110},
                "x_left": 10,
                "x_right": 130,
                "y_top": 90,
                "y_bottom": 110,
            },
        ]
        measures = [
            {"measure_id": "p1_left_m0", "system_id": "p1_left", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "x_right": 35, "y_top": 40, "y_bottom": 60},
            {"measure_id": "p1_left_m1", "system_id": "p1_left", "page": 1, "system_index": 0, "measure_local_index": 1, "x_left": 35, "x_right": 65, "y_top": 40, "y_bottom": 60},
            {"measure_id": "p1_right_m0", "system_id": "p1_right", "page": 1, "system_index": 1, "measure_local_index": 0, "x_left": 80, "x_right": 105, "y_top": 42, "y_bottom": 61},
            {"measure_id": "p1_right_m1", "system_id": "p1_right", "page": 1, "system_index": 1, "measure_local_index": 1, "x_left": 105, "x_right": 130, "y_top": 42, "y_bottom": 61},
            {"measure_id": "p1_next_m0", "system_id": "p1_next", "page": 1, "system_index": 2, "measure_local_index": 0, "x_left": 10, "x_right": 40, "y_top": 90, "y_bottom": 110},
        ]
        result_labels = {
            "p1_left_m0": "1",
            "p1_right_m0": "3",
            "p1_next_m0": "5",
        }

        anchors = WORKER._system_start_anchor_measures(measures, result_labels, systems)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_left_m0", "1"),
                ("p1_next_m0", "5"),
            ],
        )

    def test_system_start_anchor_measures_keeps_nearby_rows_separate_when_edges_do_not_match(self):
        systems = [
            {
                "system_id": "p1_top",
                "page": 1,
                "system_index": 0,
                "anchor": {"x": 10, "y_top": 40, "y_bottom": 60},
                "x_left": 10,
                "x_right": 70,
                "y_top": 40,
                "y_bottom": 60,
            },
            {
                "system_id": "p1_bottom",
                "page": 1,
                "system_index": 1,
                "anchor": {"x": 10, "y_top": 54, "y_bottom": 74},
                "x_left": 10,
                "x_right": 70,
                "y_top": 54,
                "y_bottom": 74,
            },
        ]
        measures = [
            {"measure_id": "p1_top_m0", "system_id": "p1_top", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "x_right": 40, "y_top": 40, "y_bottom": 60},
            {"measure_id": "p1_bottom_m0", "system_id": "p1_bottom", "page": 1, "system_index": 1, "measure_local_index": 0, "x_left": 10, "x_right": 40, "y_top": 54, "y_bottom": 74},
        ]
        result_labels = {
            "p1_top_m0": "1",
            "p1_bottom_m0": "2",
        }

        anchors = WORKER._system_start_anchor_measures(measures, result_labels, systems)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_top_m0", "1"),
                ("p1_bottom_m0", "2"),
            ],
        )

    def test_system_start_anchor_measures_groups_rows_with_close_centers_even_if_edges_differ(self):
        systems = [
            {
                "system_id": "p1_auto",
                "page": 1,
                "system_index": 0,
                "anchor": {"x": 10, "y_top": 40, "y_bottom": 50},
                "x_left": 10,
                "x_right": 70,
                "y_top": 40,
                "y_bottom": 50,
                "source": "auto",
            },
            {
                "system_id": "manual_sys_joined",
                "page": 1,
                "system_index": 1,
                "anchor": {"x": 76, "y_top": 37, "y_bottom": 55},
                "x_left": 76,
                "x_right": 130,
                "y_top": 37,
                "y_bottom": 55,
                "source": "manual",
                "manual_row_id": "joined",
                "staff_kind": "single",
            },
        ]
        measures = [
            {"measure_id": "p1_auto_m0", "system_id": "p1_auto", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "x_right": 40, "y_top": 40, "y_bottom": 50, "source": "auto"},
            {"measure_id": "manual_measure_joined_m0", "system_id": "manual_sys_joined", "page": 1, "system_index": 1, "measure_local_index": 0, "x_left": 76, "x_right": 100, "y_top": 37, "y_bottom": 55, "source": "manual", "manual_row_id": "joined", "staff_kind": "single"},
        ]
        result_labels = {
            "p1_auto_m0": "1",
            "manual_measure_joined_m0": "2",
        }

        anchors = WORKER._system_start_anchor_measures(measures, result_labels, systems)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [("p1_auto_m0", "1")],
        )

    def test_system_start_anchor_measures_keeps_rows_separate_when_height_ratio_is_too_different(self):
        systems = [
            {
                "system_id": "p1_small",
                "page": 1,
                "system_index": 0,
                "anchor": {"x": 10, "y_top": 40, "y_bottom": 50},
                "x_left": 10,
                "x_right": 70,
                "y_top": 40,
                "y_bottom": 50,
            },
            {
                "system_id": "p1_tall",
                "page": 1,
                "system_index": 1,
                "anchor": {"x": 76, "y_top": 35, "y_bottom": 60},
                "x_left": 76,
                "x_right": 130,
                "y_top": 35,
                "y_bottom": 60,
            },
        ]
        measures = [
            {"measure_id": "p1_small_m0", "system_id": "p1_small", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "x_right": 40, "y_top": 40, "y_bottom": 50},
            {"measure_id": "p1_tall_m0", "system_id": "p1_tall", "page": 1, "system_index": 1, "measure_local_index": 0, "x_left": 76, "x_right": 100, "y_top": 35, "y_bottom": 60},
        ]
        result_labels = {
            "p1_small_m0": "1",
            "p1_tall_m0": "2",
        }

        anchors = WORKER._system_start_anchor_measures(measures, result_labels, systems)

        self.assertEqual(
            [(row["measure_id"], label) for row, label in anchors],
            [
                ("p1_small_m0", "1"),
                ("p1_tall_m0", "2"),
            ],
        )

    def test_pickup_wins_over_same_measure_number_override(self):
        state = self._sample_state()
        state["measure_number_overrides"] = {"p1_s1_m1": 20}
        state["pickup_measures"] = {"p1_s1_m1": True}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "6", "9"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m0"], "4")
        self.assertEqual(measure_values["p1_s1_m1"], "4")
        self.assertEqual(measure_values["p1_s1_m2"], "5")
        self.assertEqual(measure_values["p1_s2_m0"], "6")

    def test_pickup_and_rest_on_same_middle_measure_shift_following_count(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m1": True}
        state["rest_measures"] = {"p1_s1_m1": 2}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "7", "10"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m0"], "4")
        self.assertEqual(measure_values["p1_s1_m1"], "4")
        self.assertEqual(measure_values["p1_s1_m2"], "6")
        self.assertEqual(measure_values["p1_s2_m0"], "7")

    def test_pickup_and_rest_at_start_of_system_shift_following_count(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m0": True}
        state["rest_measures"] = {"p1_s1_m0": 2}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "3", "7", "10"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m0"], "3")
        self.assertEqual(measure_values["p1_s1_m1"], "5")
        self.assertEqual(measure_values["p1_s1_m2"], "6")
        self.assertEqual(measure_values["p1_s2_m0"], "7")

    def test_pickup_and_rest_near_system_boundary_shift_later_systems(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s2_m2": True}
        state["rest_measures"] = {"p1_s2_m2": 2}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "7", "10"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s2_m1"], "8")
        self.assertEqual(measure_values["p1_s2_m2"], "8")
        self.assertEqual(measure_values["p2_s0_m0"], "10")

    def test_pickup_set_measure_number_and_rest_same_measure_follow_pickup_then_rest(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m1": True}
        state["measure_number_overrides"] = {"p1_s1_m1": 20}
        state["rest_measures"] = {"p1_s1_m1": 2}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "7", "10"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m0"], "4")
        self.assertEqual(measure_values["p1_s1_m1"], "4")
        self.assertEqual(measure_values["p1_s1_m2"], "6")
        self.assertEqual(measure_values["p1_s2_m0"], "7")
        self.assertEqual(measure_values["p2_s0_m0"], "10")

    def test_pickup_set_measure_number_ending_and_rest_same_measure_ignore_set_and_ending_but_keep_rest(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s2_m2": True}
        state["measure_number_overrides"] = {"p1_s2_m2": 30}
        state["endings"] = {"p1_s2_m2": "1"}
        state["rest_measures"] = {"p1_s2_m2": 2}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [str(row["current_value"]) for row in systems]
        self.assertEqual(values, ["1", "4", "7", "10"])
        measure_values = {row["measure_id"]: str(row.get("current_value") or "") for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s2_m1"], "8")
        self.assertEqual(measure_values["p1_s2_m2"], "8")
        self.assertEqual(measure_values["p2_s0_m0"], "10")
        self.assertEqual(measure_values["p2_s0_m1"], "11")

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
        self.assertEqual(values, [1, 4, 24, 27])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], 20)
        self.assertEqual(measure_values["p1_s1_m2"], 23)
        self.assertEqual(measure_values["p1_s2_m0"], 24)
        self.assertEqual(measure_values["p2_s0_m0"], 27)

    def test_set_measure_number_and_rest_measure_near_system_boundary_shift_downstream_systems(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_measure_number", "measure_id": "p1_s2_m2", "value": 30},
                {"type": "set_rest_measure", "measure_id": "p1_s2_m2", "value": 2},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_measure_number", "measure_id": "p1_s2_m2", "value": 30},
                {"type": "set_rest_measure", "measure_id": "p1_s2_m2", "value": 2},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 32])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s2_m2"], 30)
        self.assertEqual(measure_values["p2_s0_m0"], 32)
        self.assertEqual(measure_values["p2_s0_m1"], 33)

    def test_set_measure_number_and_larger_rest_follow_forced_number_not_natural_sequence(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_measure_number", "measure_id": "p1_s0_m2", "value": 40},
                {"type": "set_rest_measure", "measure_id": "p1_s0_m2", "value": 5},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_measure_number", "measure_id": "p1_s0_m2", "value": 40},
                {"type": "set_rest_measure", "measure_id": "p1_s0_m2", "value": 5},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 45, 48, 51])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s0_m2"], 40)
        self.assertEqual(measure_values["p1_s1_m0"], 45)
        self.assertEqual(measure_values["p1_s1_m1"], 46)
        self.assertEqual(measure_values["p1_s1_m2"], 47)

    def test_exact_rest_measure_wins_over_legacy_rest_staff_on_same_system(self):
        state = self._sample_state()
        state["rest_systems"] = {"p1_s1": 2}
        state["rest_measures"] = {"p1_s1_m1": 3}
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 9, 12])
        measure_values = {row["measure_id"]: int(row["current_value"]) for row in state.get("measures") or []}
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s2_m0"], 9)
        self.assertEqual(measure_values["p2_s0_m0"], 12)

    def test_set_ending_applied_recomputes_labels(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        self.assertEqual(state.get("endings"), {"p1_s1_m1": "1", "p1_s1_m2": "2"})
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 6, 9])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 5)
        self.assertEqual(measure_values["p1_s2_m0"], 6)
        self.assertEqual(measure_values["p1_s2_m1"], 7)

    def test_set_measure_number_and_ending_use_forced_local_number(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_measure_number", "measure_id": "p1_s1_m1", "value": 20},
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_measure_number", "measure_id": "p1_s1_m1", "value": 20},
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 21, 24])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 20)
        self.assertEqual(measure_values["p1_s1_m2"], 20)
        self.assertEqual(measure_values["p1_s2_m0"], 21)
        self.assertEqual(measure_values["p1_s2_m1"], 22)

    def test_set_measure_number_on_first_ending_2_measure_wins_and_carries_branch_forward(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_measure_number", "measure_id": "p1_s1_m2", "value": 20},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_measure_number", "measure_id": "p1_s1_m2", "value": 20},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 21, 24])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 20)
        self.assertEqual(measure_values["p1_s2_m0"], 21)
        self.assertEqual(measure_values["p1_s2_m1"], 22)
        self.assertEqual(measure_values["p2_s0_m0"], 24)

    def test_rest_on_first_ending_2_measure_shifts_later_numbering_correctly(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
                {"type": "set_rest_measure", "measure_id": "p1_s1_m2", "value": 2},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
                {"type": "set_rest_measure", "measure_id": "p1_s1_m2", "value": 2},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 10])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 5)
        self.assertEqual(measure_values["p1_s2_m0"], 7)
        self.assertEqual(measure_values["p2_s0_m0"], 10)

    def test_set_measure_number_ending_and_rest_cross_system_keep_downstream_labels_correct(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_measure_number", "measure_id": "p1_s1_m2", "value": 30},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "1"},
                {"type": "set_rest_measure", "measure_id": "p1_s1_m2", "value": 2},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_measure_number", "measure_id": "p1_s1_m2", "value": 30},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "1"},
                {"type": "set_rest_measure", "measure_id": "p1_s1_m2", "value": 2},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 30, 34])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m2"], 30)
        self.assertEqual(measure_values["p1_s2_m0"], 30)
        self.assertEqual(measure_values["p1_s2_m1"], 32)
        self.assertEqual(measure_values["p1_s2_m2"], 33)
        self.assertEqual(measure_values["p2_s0_m0"], 34)
        self.assertEqual(measure_values["p2_s0_m1"], 35)

    def test_pickup_at_ending_boundary_does_not_leak_stale_branch_state_forward(self):
        state = self._sample_state()
        state["pickup_measures"] = {"p1_s1_m2": True}
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            applied,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 6, 9])
        measure_values = {
            row["measure_id"]: str(row.get("current_value") or "")
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], "5")
        self.assertEqual(measure_values["p1_s1_m2"], "5")
        self.assertEqual(measure_values["p1_s2_m0"], "6")
        self.assertEqual(measure_values["p2_s0_m0"], "9")

    def test_malformed_saved_ending_fragments_are_ignored_for_numbering(self):
        state = self._sample_state()
        state["endings"] = {
            "p1_s1_m1": "1",
            "p1_s2_m0": "2",
            "p2_s0_m1": "2",
        }
        systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 10])
        measure_values = {
            row["measure_id"]: str(row.get("current_value") or "")
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], "5")
        self.assertEqual(measure_values["p1_s2_m0"], "7")
        self.assertEqual(measure_values["p2_s0_m1"], "11")

    def test_multiple_ending_groups_on_same_page_number_independently(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s0_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s0_m2", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 4)
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 3, 5, 8])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s0_m1"], 2)
        self.assertEqual(measure_values["p1_s0_m2"], 2)
        self.assertEqual(measure_values["p1_s1_m1"], 4)
        self.assertEqual(measure_values["p1_s1_m2"], 4)
        self.assertEqual(measure_values["p1_s2_m0"], 5)

    def test_multiple_ending_groups_across_pages_number_independently(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s2_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s2_m2", "value": "2"},
                {"type": "set_ending", "measure_id": "p2_s0_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p2_s0_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 4)
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 7, 9])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s2_m1"], 8)
        self.assertEqual(measure_values["p1_s2_m2"], 8)
        self.assertEqual(measure_values["p2_s0_m1"], 10)
        self.assertEqual(measure_values["p2_s0_m2"], 10)

    def test_second_ending_longer_than_first_starts_at_same_number(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 3)
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 6, 9])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 5)
        self.assertEqual(measure_values["p1_s2_m0"], 6)
        self.assertEqual(measure_values["p1_s2_m1"], 7)

    def test_second_ending_longer_than_two_measure_first_starts_at_same_number(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m1", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 5)
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 4, 5, 8])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s1_m1"], 5)
        self.assertEqual(measure_values["p1_s1_m2"], 6)
        self.assertEqual(measure_values["p1_s2_m0"], 5)
        self.assertEqual(measure_values["p1_s2_m1"], 6)
        self.assertEqual(measure_values["p1_s2_m2"], 7)
        self.assertEqual(measure_values["p2_s0_m0"], 8)

    def test_multiple_ending_groups_with_different_branch_lengths_stay_independent(self):
        state = self._sample_state()
        systems, applied, rejected, _ = WORKER._apply_relabel_edits(
            state,
            [
                {"type": "set_ending", "measure_id": "p1_s0_m0", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s0_m1", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s0_m2", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s1_m1", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s1_m2", "value": "1"},
                {"type": "set_ending", "measure_id": "p1_s2_m0", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m1", "value": "2"},
                {"type": "set_ending", "measure_id": "p1_s2_m2", "value": "2"},
            ],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(applied), 8)
        values = [int(row["current_value"]) for row in systems]
        self.assertEqual(values, [1, 3, 4, 7])
        measure_values = {
            row["measure_id"]: int(row["current_value"])
            for row in state.get("measures") or []
        }
        self.assertEqual(measure_values["p1_s0_m0"], 1)
        self.assertEqual(measure_values["p1_s0_m1"], 1)
        self.assertEqual(measure_values["p1_s0_m2"], 2)
        self.assertEqual(measure_values["p1_s1_m1"], 4)
        self.assertEqual(measure_values["p1_s1_m2"], 5)
        self.assertEqual(measure_values["p1_s2_m0"], 4)
        self.assertEqual(measure_values["p1_s2_m1"], 5)
        self.assertEqual(measure_values["p1_s2_m2"], 6)
        self.assertEqual(measure_values["p2_s0_m0"], 7)

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

    def test_numbering_baseline_scenarios_remain_stable_after_refactor(self):
        scenarios = [
            (
                "plain sequential",
                {},
                ["1", "4", "7", "10"],
                {"p1_s1_m1": "5", "p1_s2_m0": "7"},
            ),
            (
                "one measure-number override",
                {"measure_number_overrides": {"p1_s1_m1": 20}},
                ["1", "4", "22", "25"],
                {"p1_s1_m1": "20", "p1_s1_m2": "21"},
            ),
            (
                "one pickup",
                {"pickup_measures": {"p1_s1_m1": True}},
                ["1", "4", "6", "9"],
                {"p1_s1_m1": "4", "p1_s1_m2": "5"},
            ),
            (
                "one exact rest",
                {"rest_measures": {"p1_s1_m1": 3}},
                ["1", "4", "9", "12"],
                {"p1_s1_m1": "5", "p1_s1_m2": "8"},
            ),
            (
                "one ending branch pair",
                {"endings": {"p1_s1_m1": "1", "p1_s1_m2": "2"}},
                ["1", "4", "6", "9"],
                {"p1_s1_m1": "5", "p1_s1_m2": "5"},
            ),
        ]

        for name, updates, expected_systems, expected_measures in scenarios:
            with self.subTest(name=name):
                state = self._sample_state()
                state.update(updates)
                systems, _, _, _ = WORKER._apply_relabel_edits(state, [])
                self.assertEqual([str(row["current_value"]) for row in systems], expected_systems)

                measure_values = {
                    row["measure_id"]: str(row.get("current_value") or "")
                    for row in state.get("measures") or []
                }
                for measure_id, expected_value in expected_measures.items():
                    self.assertEqual(measure_values[measure_id], expected_value)

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

    def test_replace_manual_rows_for_page_creates_manual_system_and_measures(self):
        state = self._sample_state_with_bounds()

        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            state,
            [
                {
                    "type": "replace_manual_rows_for_page",
                    "page": 1,
                    "rows": [
                        {
                            "manual_row_id": "rowA",
                            "staff_kind": "single",
                            "rect": {"left": 12, "right": 90, "top": 70, "bottom": 80},
                            "cut_xs": [40, 65],
                        }
                    ],
                }
            ],
        )

        self.assertEqual(total, 5)
        self.assertEqual(applied, [{"type": "replace_manual_rows_for_page", "page": 1, "rows_count": 1}])
        self.assertEqual(rejected, [])
        self.assertEqual(
            state.get("manual_rows"),
            [
                {
                    "manual_row_id": "rowA",
                    "page": 1,
                    "staff_kind": "single",
                    "rect": {"left": 12.0, "right": 90.0, "top": 70.0, "bottom": 80.0},
                    "cut_xs": [40.0, 65.0],
                }
            ],
        )

        manual_system = next(row for row in systems if row.get("source") == "manual")
        self.assertEqual(manual_system.get("manual_row_id"), "rowA")
        self.assertEqual(manual_system.get("system_index"), 2)
        self.assertEqual(manual_system.get("staff_kind"), "single")
        self.assertEqual(int(manual_system.get("current_value")), 7)

        manual_measures = [row for row in (state.get("measures") or []) if row.get("manual_row_id") == "rowA"]
        self.assertEqual(len(manual_measures), 3)
        self.assertEqual([row.get("measure_id") for row in manual_measures], ["manual_measure_rowA_m0", "manual_measure_rowA_m1", "manual_measure_rowA_m2"])
        self.assertEqual([row.get("x_left") for row in manual_measures], [12.0, 40.0, 65.0])
        self.assertEqual([row.get("x_right") for row in manual_measures], [40.0, 65.0, 90.0])
        self.assertEqual([int(row.get("current_value")) for row in manual_measures], [7, 8, 9])

        auto_measure_ids = {str(row.get("measure_id") or "") for row in (state.get("measures") or [])}
        self.assertIn("p1_s2_m0", auto_measure_ids)
        shifted_measure = next(row for row in (state.get("measures") or []) if row.get("measure_id") == "p1_s2_m0")
        self.assertEqual(shifted_measure.get("system_index"), 3)

        batches = WORKER._ai_suggest_system_batches(state)
        manual_batch = next((batch for batch in batches if batch[0].get("manual_row_id") == "rowA"), None)
        self.assertIsNotNone(manual_batch)
        self.assertEqual(len(manual_batch[1]), 3)

    def test_replace_manual_rows_for_page_grand_staff_creates_shared_row(self):
        state = self._sample_state_with_bounds()

        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            state,
            [
                {
                    "type": "replace_manual_rows_for_page",
                    "page": 2,
                    "rows": [
                        {
                            "manual_row_id": "grand1",
                            "staff_kind": "grand",
                            "rect": {"left": 18, "right": 96, "top": 35, "bottom": 75},
                            "cut_xs": [58],
                        }
                    ],
                }
            ],
        )

        self.assertEqual(total, 5)
        self.assertEqual(applied, [{"type": "replace_manual_rows_for_page", "page": 2, "rows_count": 1}])
        self.assertEqual(rejected, [])

        manual_system = next(row for row in systems if row.get("manual_row_id") == "grand1")
        self.assertEqual(manual_system.get("staff_kind"), "grand")
        self.assertEqual(manual_system.get("source"), "manual")
        self.assertEqual(manual_system.get("page"), 2)
        self.assertEqual(manual_system.get("system_index"), 1)
        self.assertEqual(len([row for row in systems if row.get("page") == 2]), 2)

        manual_measures = [row for row in (state.get("measures") or []) if row.get("manual_row_id") == "grand1"]
        self.assertEqual(len(manual_measures), 2)
        self.assertTrue(all(row.get("staff_kind") == "grand" for row in manual_measures))
        self.assertTrue(all(row.get("y_top") == 35.0 and row.get("y_bottom") == 75.0 for row in manual_measures))

    def test_replace_manual_rows_for_page_rejected_on_strong_overlap(self):
        state = self._sample_state_with_bounds()

        systems, applied, rejected, total = WORKER._apply_relabel_edits(
            state,
            [
                {
                    "type": "replace_manual_rows_for_page",
                    "page": 1,
                    "rows": [
                        {
                            "manual_row_id": "badrow",
                            "staff_kind": "single",
                            "rect": {"left": 12, "right": 90, "top": 52, "bottom": 58},
                            "cut_xs": [],
                        }
                    ],
                }
            ],
        )

        self.assertEqual(total, 4)
        self.assertEqual(applied, [])
        self.assertEqual(rejected, [{"edit": {"type": "replace_manual_rows_for_page", "page": 1, "rows": [{"manual_row_id": "badrow", "staff_kind": "single", "rect": {"left": 12, "right": 90, "top": 52, "bottom": 58}, "cut_xs": []}]}, "reason": "manual_row_overlap_auto"}])
        self.assertEqual(state.get("manual_rows"), [])
        self.assertEqual(len([row for row in systems if row.get("source") == "manual"]), 0)

    def test_refresh_editable_state_systems_and_measures_rebuilds_manual_rows(self):
        state = self._sample_state_with_bounds()
        state["manual_rows"] = [
            {
                "manual_row_id": "persist1",
                "page": 1,
                "staff_kind": "single",
                "rect": {"left": 14, "right": 92, "top": 70, "bottom": 82},
                "cut_xs": [44],
            }
        ]

        systems, measures, reassign_count, qa = WORKER._refresh_editable_state_systems_and_measures(state)

        self.assertEqual(reassign_count, 0)
        self.assertEqual(qa.get("status"), "ok")
        self.assertEqual(len([row for row in systems if row.get("source") == "manual"]), 1)
        persisted_measures = [row for row in measures if row.get("manual_row_id") == "persist1"]
        self.assertEqual([row.get("measure_id") for row in persisted_measures], ["manual_measure_persist1_m0", "manual_measure_persist1_m1"])
        self.assertTrue(all(row.get("source") == "manual" for row in persisted_measures))


if __name__ == "__main__":
    unittest.main()
