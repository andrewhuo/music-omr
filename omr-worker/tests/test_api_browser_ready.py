import importlib.util
import json
import os
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


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

        class _DummyResponse:
            def __init__(self, payload=None, status_code=200):
                self.payload = payload
                self.status_code = int(status_code)
                self.headers: dict[str, str] = {}

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
                if isinstance(payload, tuple):
                    body, status = payload
                else:
                    body, status = payload, 200
                return _DummyResponse(body, status)

            def run(self, *args, **kwargs):
                return None

        flask.Flask = _DummyApp
        flask.jsonify = lambda payload, *args, **kwargs: payload
        flask.request = SimpleNamespace(path="", method="GET", headers={}, files={}, json={})
        sys.modules["flask"] = flask

    worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "worker.py"))
    spec = importlib.util.spec_from_file_location("omr_worker_browser_test", worker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _unpack(result):
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], int(result[1])
    return result, 200


WORKER = _load_worker_module()


class _FakeRect:
    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class _FakePage:
    def __init__(self, rect: _FakeRect):
        self.rect = rect


class _FakeDoc:
    def __init__(self, pages):
        self._pages = list(pages)

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        return None


class _FakeUploadFile:
    def __init__(self, filename: str, content_type: str, data: bytes):
        self.filename = filename
        self.mimetype = content_type
        self.content_type = content_type
        self._data = data

    def read(self):
        return self._data


class BrowserReadyApiTests(unittest.TestCase):
    def setUp(self):
        os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:5173"
        os.environ["ANTHROPIC_MODEL"] = "claude-sonnet-4-6"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        WORKER.request = SimpleNamespace(path="", method="GET", headers={}, files={}, json={})
        WORKER._PENDING_DISPATCHES.clear()

    def _sample_artifacts(self):
        return {
            "audiveris_out_pdf": "gs://x/output/audiveris_out.pdf",
            "audiveris_out_corrected_pdf": "gs://x/output/audiveris_out_corrected.pdf",
            "run_info": "gs://x/output/artifacts/run_info.json",
            "mapping_summary": "gs://x/output/artifacts/mapping_summary.json",
        }

    def _sample_mapping_summary(self):
        return {
            "editable_state": {
                "version": "system_state_v1",
                "time_signature_context": {
                    "system_rows": [
                        {"system_id": "p1_s0", "page": 1, "system_index": 0, "time_signature": "3/4", "source": "explicit"},
                        {"system_id": "p1_s1", "page": 1, "system_index": 1, "time_signature": "3/4", "source": "inherited"},
                    ],
                    "measure_rows": [
                        {"measure_id": "p1_s0_m0", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 0, "time_signature": "3/4", "source": "explicit"},
                        {"measure_id": "p1_s0_m1", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 1, "time_signature": "3/4", "source": "inherited"},
                        {"measure_id": "p1_s1_m0", "system_id": "p1_s1", "page": 1, "system_index": 1, "measure_local_index": 0, "time_signature": "3/4", "source": "inherited"},
                    ],
                },
                "systems": [
                    {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1", "anchor": {"x": 10, "y_top": 20, "y_bottom": 60}, "time_signature": "3/4", "time_signature_source": "explicit"},
                    {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "3", "anchor": {"x": 10, "y_top": 80, "y_bottom": 120}, "time_signature": "3/4", "time_signature_source": "inherited"},
                ],
                "measures": [
                    {
                        "measure_id": "p1_s0_m0",
                        "system_id": "p1_s0",
                        "page": 1,
                        "system_index": 0,
                        "measure_local_index": 0,
                        "global_index": 0,
                        "x_left": 30,
                        "y_top": 20,
                        "y_bottom": 40,
                        "time_signature": "3/4",
                        "time_signature_source": "explicit",
                    },
                    {
                        "measure_id": "p1_s0_m1",
                        "system_id": "p1_s0",
                        "page": 1,
                        "system_index": 0,
                        "measure_local_index": 1,
                        "global_index": 1,
                        "x_left": 90,
                        "y_top": 20,
                        "y_bottom": 40,
                        "time_signature": "3/4",
                        "time_signature_source": "inherited",
                    },
                    {
                        "measure_id": "p1_s1_m0",
                        "system_id": "p1_s1",
                        "page": 1,
                        "system_index": 1,
                        "measure_local_index": 0,
                        "global_index": 2,
                        "x_left": 30,
                        "y_top": 80,
                        "y_bottom": 100,
                        "time_signature": "3/4",
                        "time_signature_source": "inherited",
                    },
                ],
            }
        }

    def test_cors_allowed_and_disallowed(self):
        self.assertTrue(WORKER._origin_allowed("http://localhost:5173"))
        self.assertFalse(WORKER._origin_allowed("https://evil.example"))

        WORKER.request = SimpleNamespace(
            path="/api/omr/jobs",
            method="OPTIONS",
            headers={"Origin": "http://localhost:5173"},
            files={},
            json={},
        )
        resp = WORKER._api_before_request()
        self.assertEqual(getattr(resp, "status_code", 0), 204)
        self.assertEqual(getattr(resp, "headers", {}).get("Access-Control-Allow-Origin"), "http://localhost:5173")

    def test_non_options_requests_pass_without_auth_header(self):
        WORKER.request = SimpleNamespace(
            path="/api/omr/jobs",
            method="POST",
            headers={},
            files={},
            json={},
        )
        self.assertIsNone(WORKER._api_before_request())

    def test_options_preflight_fallback_never_500(self):
        WORKER.request = SimpleNamespace(
            path="/api/omr/jobs",
            method="OPTIONS",
            headers={"Origin": "http://localhost:5173"},
            files={},
            json={},
        )
        with patch.object(WORKER, "_apply_cors_headers", side_effect=RuntimeError("boom")):
            resp = WORKER._api_before_request()
        self.assertEqual(getattr(resp, "status_code", 0), 204)

    def test_signed_url_fallback_returns_empty(self):
        class _FakeBlob:
            def exists(self):
                return False

        class _FakeBucket:
            def blob(self, _name):
                return _FakeBlob()

        class _FakeClient:
            def bucket(self, _name):
                return _FakeBucket()

        with patch.object(WORKER, "_gcs_client", return_value=_FakeClient()):
            self.assertEqual(WORKER._signed_http_url_for_gs("gs://bucket/file.pdf"), "")

    def test_upload_validations(self):
        WORKER.request = SimpleNamespace(path="/api/omr/uploads", method="POST", headers={}, files={}, json={})
        body, status = _unpack(WORKER.upload_pdf())
        self.assertEqual(status, 400)
        self.assertEqual(body.get("error"), "file is required")

        WORKER.request = SimpleNamespace(
            path="/api/omr/uploads",
            method="POST",
            headers={},
            files={"file": _FakeUploadFile("note.txt", "text/plain", b"hello")},
            json={},
        )
        body, status = _unpack(WORKER.upload_pdf())
        self.assertEqual(status, 400)
        self.assertEqual(body.get("error"), "file must be a PDF")

        with patch.object(WORKER, "_max_upload_bytes", return_value=10):
            WORKER.request = SimpleNamespace(
                path="/api/omr/uploads",
                method="POST",
                headers={},
                files={"file": _FakeUploadFile("test.pdf", "application/pdf", b"%PDF " + b"x" * 20)},
                json={},
            )
            body, status = _unpack(WORKER.upload_pdf())
        self.assertEqual(status, 413)
        self.assertEqual(body.get("error"), "file too large")

        with patch.object(WORKER, "_upload_file_to_gcs", return_value=None):
            WORKER.request = SimpleNamespace(
                path="/api/omr/uploads",
                method="POST",
                headers={},
                files={"file": _FakeUploadFile("test.pdf", "application/pdf", b"%PDF-1.4\n")},
                json={},
            )
            body, status = _unpack(WORKER.upload_pdf())
        self.assertEqual(status, 201)
        self.assertTrue(str(body.get("pdf_gcs_uri") or "").startswith("gs://"))
        self.assertGreater(int(body.get("size_bytes") or 0), 0)

    def test_artifacts_http_present_on_responses(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}

        with (
            patch.object(WORKER, "_get_ref_sha", return_value="abc"),
            patch.object(WORKER, "_dispatch_workflow", return_value=None),
            patch.object(WORKER, "_discover_run_id", return_value=111),
            patch.object(WORKER, "_artifact_uris_for_run", return_value=artifacts),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
        ):
            WORKER.request = SimpleNamespace(
                path="/api/omr/jobs",
                method="POST",
                headers={},
                files={},
                json={"pdf_gcs_uri": "gs://bucket/input.pdf"},
            )
            body, status = _unpack(WORKER.create_job())
        self.assertEqual(status, 202)
        self.assertEqual(body.get("artifacts_http"), artifacts_http)

        with (
            patch.object(
                WORKER,
                "_get_run",
                return_value={
                    "status": "completed",
                    "conclusion": "success",
                    "head_branch": "main",
                    "head_sha": "abc",
                    "run_attempt": 1,
                    "created_at": "2026-03-03T00:00:00Z",
                    "updated_at": "2026-03-03T00:01:00Z",
                    "html_url": "https://github.com/example",
                },
            ),
            patch.object(WORKER, "_artifact_uris_for_run", return_value=artifacts),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
        ):
            body, status = _unpack(WORKER.get_job("111"))
        self.assertEqual(status, 200)
        self.assertEqual(body.get("artifacts_http"), artifacts_http)

        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, None, None)),
            patch.object(
                WORKER,
                "_load_mapping_for_run",
                return_value=(
                    artifacts,
                    {
                        "editable_state": {
                            "version": "system_state_v1",
                            "qa": {"ok": True},
                            "systems": [{"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1"}],
                        }
                    },
                    111,
                ),
            ),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
        ):
            body, status = _unpack(WORKER.get_job_state("111"))
        self.assertEqual(status, 200)
        self.assertEqual(body.get("artifacts_http"), artifacts_http)
        self.assertEqual((body.get("editable_state") or {}).get("labels_mode"), "system_only")
        self.assertEqual((body.get("editable_state") or {}).get("manual_rows"), [])
        self.assertEqual((body.get("editable_state") or {}).get("rest_measures"), {})
        self.assertEqual((body.get("editable_state") or {}).get("pickup_measures"), {})
        self.assertEqual((body.get("editable_state") or {}).get("staff_boxes"), [])
        self.assertIsNone(body.get("ai_suggestions"))
        self.assertEqual(((body.get("ai_suggest_run") or {}).get("status")), "idle")

    def test_list_jobs_endpoint_returns_simple_rows(self):
        now = WORKER._utc_now()
        WORKER._PENDING_DISPATCHES["abc123"] = {
            "dispatch_id": "abc123",
            "dispatched_at": now,
            "expected_sha": "abc",
            "run_id": 111,
        }

        with patch.object(WORKER, "_get_run", return_value={"status": "completed", "conclusion": "success", "created_at": "2026-03-08T20:00:00Z"}):
            WORKER.request = SimpleNamespace(path="/api/omr/jobs", method="GET", headers={}, files={}, json={})
            body, status = _unpack(WORKER.list_jobs())

        self.assertEqual(status, 200)
        self.assertIsInstance(body.get("jobs"), list)
        self.assertEqual(len(body["jobs"]), 1)
        self.assertEqual(body["jobs"][0].get("job_id"), "abc123")
        self.assertEqual(body["jobs"][0].get("status"), "succeeded")
        self.assertEqual(body["jobs"][0].get("created_at"), "2026-03-08T20:00:00Z")

    def test_list_jobs_returns_empty_array_when_snapshot_fails(self):
        WORKER.request = SimpleNamespace(path="/api/omr/jobs", method="GET", headers={}, files={}, json={})
        with patch.object(WORKER, "_pending_items_snapshot", side_effect=RuntimeError("snapshot failed")):
            body, status = _unpack(WORKER.list_jobs())
        self.assertEqual(status, 200)
        self.assertEqual(body.get("jobs"), [])

    def test_list_jobs_tolerates_malformed_records(self):
        WORKER._PENDING_DISPATCHES["good"] = {
            "dispatch_id": "good",
            "dispatched_at": WORKER._utc_now(),
            "expected_sha": "abc",
            "run_id": "123",
        }
        WORKER._PENDING_DISPATCHES["bad"] = {
            "dispatch_id": "bad",
            "dispatched_at": object(),
            "run_id": object(),
        }

        with patch.object(WORKER, "_get_run", side_effect=RuntimeError("github down")):
            WORKER.request = SimpleNamespace(path="/api/omr/jobs", method="GET", headers={}, files={}, json={})
            body, status = _unpack(WORKER.list_jobs())

        self.assertEqual(status, 200)
        self.assertEqual(len(body.get("jobs") or []), 2)

    def test_relabel_set_labels_mode_all_measures(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = {
            "editable_state": {
                "version": "system_state_v1",
                "systems": [
                    {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1", "anchor": {"x": 10, "y_top": 20}},
                    {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "5", "anchor": {"x": 10, "y_top": 80}},
                ],
                "measures": [
                    {"measure_id": "m0", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 0, "global_index": 0, "x_left": 30, "y_top": 20},
                    {"measure_id": "m1", "system_id": "p1_s1", "page": 1, "system_index": 1, "measure_local_index": 0, "global_index": 1, "x_left": 30, "y_top": 80},
                ],
            }
        }

        WORKER.request = SimpleNamespace(
            path="/api/omr/jobs/111/relabel",
            method="POST",
            headers={},
            files={},
            json={"edits": [{"type": "set_labels_mode", "value": "all_measures"}]},
        )
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_download_gcs_to_file", return_value=None),
            patch.object(WORKER, "_render_corrected_pdf", return_value=2),
            patch.object(WORKER, "_upload_file_to_gcs", return_value=None),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.relabel_job("111"))

        self.assertEqual(status, 200)
        relabel = body.get("relabel") or {}
        self.assertEqual(relabel.get("labels_mode"), "all_measures")
        self.assertEqual(relabel.get("labels_redrawn_count"), 2)

    def test_measure_crop_spec_uses_full_vertical_padding_when_room_exists(self):
        page_rect = _FakeRect(0, 0, 200, 160)
        measure_row = {"x_left": 30, "y_top": 40, "y_bottom": 60}
        next_measure_row = {"x_left": 90}
        system_row = {"anchor": {"y_top": 40, "y_bottom": 60}}
        prev_system_row = {"anchor": {"y_top": 0, "y_bottom": 20}}
        next_system_row = {"anchor": {"y_top": 100, "y_bottom": 120}}

        with patch.object(WORKER.fitz, "Rect", _FakeRect):
            spec = WORKER._measure_crop_spec(
                page_rect,
                measure_row,
                next_measure_row,
                system_row,
                prev_system_row,
                next_system_row,
            )

        clip = spec["clip"]
        padding = spec["padding"]
        self.assertEqual(padding.get("left"), 8.0)
        self.assertEqual(padding.get("right"), 8.0)
        self.assertEqual(padding.get("top"), 20.0)
        self.assertEqual(padding.get("bottom"), 20.0)
        self.assertEqual(clip.x0, 22.0)
        self.assertEqual(clip.x1, 98.0)
        self.assertEqual(clip.y0, 25.0)
        self.assertEqual(clip.y1, 80.0)

    def test_measure_crop_spec_uses_75_percent_gap_clamp(self):
        page_rect = _FakeRect(0, 0, 220, 200)
        measure_row = {"x_left": 25, "y_top": 95, "y_bottom": 125}
        next_measure_row = {"x_left": 110}
        system_row = {"anchor": {"y_top": 90, "y_bottom": 130}}
        prev_system_row = {"anchor": {"y_top": 40, "y_bottom": 70}}
        next_system_row = {"anchor": {"y_top": 170, "y_bottom": 190}}

        with patch.object(WORKER.fitz, "Rect", _FakeRect):
            spec = WORKER._measure_crop_spec(
                page_rect,
                measure_row,
                next_measure_row,
                system_row,
                prev_system_row,
                next_system_row,
            )

        clip = spec["clip"]
        self.assertEqual(clip.y0, 75.0)
        self.assertEqual(clip.y1, 155.0)
        self.assertEqual((spec.get("system_bounds") or {}).get("top"), 75.0)
        self.assertEqual((spec.get("system_bounds") or {}).get("bottom"), 160.0)

    def test_measure_crop_spec_handles_piano_measure_scale(self):
        page_rect = _FakeRect(0, 0, 260, 260)
        measure_row = {"x_left": 40, "y_top": 120, "y_bottom": 172}
        next_measure_row = {"x_left": 130}
        system_row = {"anchor": {"y_top": 118, "y_bottom": 172}}
        prev_system_row = {"anchor": {"y_top": 60, "y_bottom": 92}}
        next_system_row = {"anchor": {"y_top": 210, "y_bottom": 238}}

        with patch.object(WORKER.fitz, "Rect", _FakeRect):
            spec = WORKER._measure_crop_spec(
                page_rect,
                measure_row,
                next_measure_row,
                system_row,
                prev_system_row,
                next_system_row,
            )

        padding = spec["padding"]
        clip = spec["clip"]
        self.assertAlmostEqual(padding.get("top"), 52.0)
        self.assertAlmostEqual(padding.get("bottom"), 52.0)
        self.assertEqual(padding.get("left"), 8.0)
        self.assertEqual(padding.get("right"), 8.0)
        self.assertAlmostEqual(clip.y0, 98.5)
        self.assertAlmostEqual(clip.y1, 200.5)

    def test_measure_crop_spec_keeps_room_for_high_first_measure(self):
        page_rect = _FakeRect(0, 0, 240, 180)
        measure_row = {"x_left": 18, "y_top": 14, "y_bottom": 38}
        next_measure_row = {"x_left": 70}
        system_row = {"anchor": {"y_top": 14, "y_bottom": 38}}
        next_system_row = {"anchor": {"y_top": 90, "y_bottom": 120}}

        with patch.object(WORKER.fitz, "Rect", _FakeRect):
            spec = WORKER._measure_crop_spec(
                page_rect,
                measure_row,
                next_measure_row,
                system_row,
                None,
                next_system_row,
            )

        clip = spec["clip"]
        padding = spec["padding"]
        self.assertEqual(padding.get("top"), 24.0)
        self.assertEqual(padding.get("bottom"), 24.0)
        self.assertEqual(clip.y0, 0.0)
        self.assertEqual(clip.y1, 62.0)

    def test_measure_crop_spec_old_style_multi_rest_stays_generous_above(self):
        page_rect = _FakeRect(0, 0, 220, 180)
        measure_row = {"x_left": 24, "y_top": 86, "y_bottom": 104}
        next_measure_row = {"x_left": 76}
        system_row = {"anchor": {"y_top": 86, "y_bottom": 104}}
        prev_system_row = {"anchor": {"y_top": 38, "y_bottom": 62}}
        next_system_row = {"anchor": {"y_top": 138, "y_bottom": 160}}

        with patch.object(WORKER.fitz, "Rect", _FakeRect):
            spec = WORKER._measure_crop_spec(
                page_rect,
                measure_row,
                next_measure_row,
                system_row,
                prev_system_row,
                next_system_row,
            )

        clip = spec["clip"]
        padding = spec["padding"]
        self.assertEqual(padding.get("top"), 20.0)
        self.assertEqual(padding.get("bottom"), 18.0)
        self.assertEqual(clip.y0, 68.0)
        self.assertEqual(clip.y1, 122.0)

    def test_ai_suggest_start_initializes_running_state(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_editable_state_version", return_value="test-state"),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.ai_suggest_job("111"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "running")
        ai_suggestions = body.get("ai_suggestions") or {}
        summary = ai_suggestions.get("summary") or {}
        self.assertEqual(summary.get("measures_seen"), 3)
        self.assertEqual(summary.get("suggestions_kept"), 0)
        self.assertEqual(summary.get("systems_processed"), 0)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("systems_total"), 2)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("systems_completed"), 0)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("next_system_index"), 0)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("model"), "claude-sonnet-4-6")
        self.assertIn("ai_suggestions", mapping_summary)
        self.assertEqual(((mapping_summary.get("ai_suggest_run") or {}).get("status")), "running")

    def test_build_ai_batch_trace_payload_records_statuses_and_order(self):
        systems = [
            {"system_id": "p1_s0", "page": 1, "system_index": 0},
            {"system_id": "p1_s1", "page": 1, "system_index": 1},
        ]
        measures = [
            {
                "measure_id": "m0",
                "page": 1,
                "system_id": "p1_s0",
                "system_index": 0,
                "measure_local_index": 0,
                "x_left": 10,
                "y_top": 20,
                "y_bottom": 40,
            },
            {
                "measure_id": "m1",
                "page": 1,
                "system_id": "p1_s1",
                "system_index": 1,
                "measure_local_index": 0,
                "x_left": 50,
                "y_top": 20,
                "y_bottom": 40,
            },
            {
                "measure_id": "m2",
                "page": 1,
                "system_id": "",
                "system_index": 0,
                "measure_local_index": 0,
                "x_left": 90,
                "y_top": 20,
                "y_bottom": 40,
            },
            {
                "measure_id": "m3",
                "page": 1,
                "system_id": "missing_system",
                "system_index": 9,
                "measure_local_index": 0,
                "x_left": 130,
                "y_top": 20,
                "y_bottom": 40,
            },
            {
                "measure_id": "m4",
                "page": 1,
                "system_id": "missing_after_reassign",
                "system_index": 9,
                "measure_local_index": 0,
                "x_left": 170,
                "y_top": 20,
                "y_bottom": 40,
            },
        ]
        before_snapshot = [
            {"system_id_before_reassign": "p1_s0", "system_index_before_reassign": 0},
            {"system_id_before_reassign": "p1_s0", "system_index_before_reassign": 0},
            {"system_id_before_reassign": "", "system_index_before_reassign": 0},
            {"system_id_before_reassign": "missing_system", "system_index_before_reassign": 9},
            {"system_id_before_reassign": "p1_s0", "system_index_before_reassign": 0},
        ]
        system_batches = [
            (systems[0], [measures[0]]),
            (systems[1], [measures[1]]),
        ]

        payload = WORKER._build_ai_batch_trace_payload(
            "job-1",
            111,
            systems,
            measures,
            system_batches,
            before_snapshot=before_snapshot,
        )

        rows = {str(row.get("measure_id")): row for row in (payload.get("measures") or [])}
        self.assertEqual((rows.get("m0") or {}).get("status"), "batched")
        self.assertEqual((rows.get("m0") or {}).get("display_system_number"), 1)
        self.assertEqual((rows.get("m0") or {}).get("display_measure_number"), 1)
        self.assertEqual((rows.get("m1") or {}).get("status"), "reassigned_and_batched")
        self.assertEqual((rows.get("m1") or {}).get("display_system_number"), 2)
        self.assertEqual((rows.get("m1") or {}).get("display_measure_number"), 1)
        self.assertEqual((rows.get("m2") or {}).get("status"), "skipped_missing_system_id")
        self.assertEqual((rows.get("m3") or {}).get("status"), "skipped_no_matching_system")
        self.assertEqual((rows.get("m4") or {}).get("status"), "reassigned_but_unbatched")
        self.assertEqual(payload.get("measure_count"), 5)
        self.assertEqual(payload.get("batched_count"), 2)
        self.assertEqual(payload.get("skipped_count"), 3)
        systems_summary = {str(row.get("system_id")): row for row in (payload.get("systems") or [])}
        self.assertEqual((systems_summary.get("p1_s0") or {}).get("measure_ids_batched"), ["m0"])
        self.assertEqual((systems_summary.get("p1_s1") or {}).get("measure_ids_batched"), ["m1"])
        self.assertEqual((systems_summary.get("p1_s0") or {}).get("display_system_number"), 1)
        self.assertEqual((systems_summary.get("p1_s1") or {}).get("display_system_number"), 2)

    def test_ai_suggest_step_persists_one_system_and_advances_progress(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = WORKER._empty_ai_suggestions_state(111, "test-state", 3)
        mapping_summary["ai_suggest_run"] = {
            "status": "running",
            "started_at_utc": "2026-05-05T00:00:00Z",
            "updated_at_utc": "2026-05-05T00:00:00Z",
            "completed_at_utc": None,
            "failed_at_utc": None,
            "systems_total": 2,
            "systems_completed": 0,
            "next_system_index": 0,
            "source_run_id": 111,
            "source_state_version": "test-state",
            "last_error": None,
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest/step", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_editable_state_version", return_value="test-state"),
            patch.object(
                WORKER,
                "_generate_ai_suggestions_for_system_batch",
                return_value={
                    "version": "ai_suggestions_v1",
                    "generated_at_utc": "2026-05-05T00:00:01Z",
                    "provider": "claude",
                    "model": "claude-test",
                    "source_run_id": 111,
                    "by_measure_id": {
                        "p1_s0_m0": {"label": "pickup", "rest_count": None, "confidence": "medium"}
                    },
                    "warnings": [],
                    "summary": {"systems_processed": 1, "measures_seen": 2, "suggestions_kept": 1, "normal_measures_omitted": 1},
                },
            ),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.ai_suggest_job_step("111"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "running")
        self.assertEqual((body.get("ai_suggest_run") or {}).get("systems_completed"), 1)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("next_system_index"), 1)
        by_measure_id = ((body.get("ai_suggestions") or {}).get("by_measure_id") or {})
        self.assertEqual(sorted(by_measure_id.keys()), ["p1_s0_m0"])
        self.assertEqual((body.get("ai_suggestions") or {}).get("model"), "claude-sonnet-4-6")
        self.assertEqual((((body.get("ai_suggestions") or {}).get("summary") or {}).get("systems_processed")), 1)

    def test_ai_suggest_step_returns_debug_batch_trace_when_enabled(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = WORKER._empty_ai_suggestions_state(111, "test-state", 3)
        mapping_summary["ai_suggest_run"] = {
            "status": "running",
            "started_at_utc": "2026-05-05T00:00:00Z",
            "updated_at_utc": "2026-05-05T00:00:00Z",
            "completed_at_utc": None,
            "failed_at_utc": None,
            "systems_total": 2,
            "systems_completed": 0,
            "next_system_index": 0,
            "source_run_id": 111,
            "source_state_version": "test-state",
            "last_error": None,
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest/step", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_editable_state_version", return_value="test-state"),
            patch.object(
                WORKER,
                "_generate_ai_suggestions_for_system_batch",
                return_value={
                    "version": "ai_suggestions_v1",
                    "generated_at_utc": "2026-05-05T00:00:01Z",
                    "provider": "claude",
                    "model": "claude-test",
                    "source_run_id": 111,
                    "by_measure_id": {
                        "p1_s0_m0": {"label": "pickup", "rest_count": None, "confidence": "medium"}
                    },
                    "warnings": [],
                    "summary": {"systems_processed": 1, "measures_seen": 2, "suggestions_kept": 1, "normal_measures_omitted": 1},
                },
            ),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
            patch.object(WORKER, "_signed_http_url_for_gs", return_value="https://signed/debug"),
            patch.object(WORKER, "_gcs_uri_exists", return_value=False),
            patch.object(WORKER, "_ai_suggest_debug_enabled", return_value=True),
        ):
            body, status = _unpack(WORKER.ai_suggest_job_step("111"))

        self.assertEqual(status, 200)
        debug_batch_trace = body.get("debug_batch_trace") or {}
        self.assertEqual(debug_batch_trace.get("enabled"), True)
        self.assertEqual(debug_batch_trace.get("measure_count"), 3)
        self.assertEqual(debug_batch_trace.get("batched_count"), 3)
        self.assertEqual(debug_batch_trace.get("skipped_count"), 0)
        self.assertEqual(debug_batch_trace.get("trace_http"), "https://signed/debug")

    def test_resolve_ai_crop_pdf_source_prefers_corrected_pdf(self):
        artifacts = self._sample_artifacts()
        with TemporaryDirectory(prefix="ai-crop-source-test-") as tmp:
            tmpdir = Path(tmp)
            download_calls: list[str] = []

            def _fake_download(uri, dest):
                download_calls.append(str(uri))
                Path(dest).write_bytes(b"%PDF-1.4\n")

            with patch.object(WORKER, "_download_gcs_to_file", side_effect=_fake_download):
                pdf_path, pdf_source = WORKER._resolve_ai_crop_pdf_source(artifacts, tmpdir)

        self.assertEqual(pdf_source, "corrected")
        self.assertTrue(str(pdf_path).endswith("audiveris_out_corrected.pdf"))
        self.assertEqual(download_calls, ["gs://x/output/audiveris_out_corrected.pdf"])

    def test_resolve_ai_crop_pdf_source_falls_back_to_baseline(self):
        artifacts = self._sample_artifacts()
        with TemporaryDirectory(prefix="ai-crop-source-test-") as tmp:
            tmpdir = Path(tmp)
            download_calls: list[str] = []

            def _fake_download(uri, dest):
                download_calls.append(str(uri))
                if str(uri).endswith("audiveris_out_corrected.pdf"):
                    raise RuntimeError("missing corrected")
                Path(dest).write_bytes(b"%PDF-1.4\n")

            with patch.object(WORKER, "_download_gcs_to_file", side_effect=_fake_download):
                pdf_path, pdf_source = WORKER._resolve_ai_crop_pdf_source(artifacts, tmpdir)

        self.assertEqual(pdf_source, "baseline")
        self.assertTrue(str(pdf_path).endswith("audiveris_out.pdf"))
        self.assertEqual(
            download_calls,
            [
                "gs://x/output/audiveris_out_corrected.pdf",
                "gs://x/output/audiveris_out.pdf",
            ],
        )

    def test_generate_ai_suggestions_for_system_batch_uses_neighbor_systems_without_crashing(self):
        artifacts = self._sample_artifacts()
        editable_state = self._sample_mapping_summary().get("editable_state") or {}
        system_batches = WORKER._ai_suggest_system_batches(editable_state)
        system_row, system_measures = system_batches[0]
        systems = editable_state.get("systems") or []
        fake_doc = _FakeDoc([_FakePage(_FakeRect(0, 0, 200, 160))])
        provider_payload = {
            "provider": "claude",
            "suggestions": [
                {
                    "measure_id": "p1_s0_m0",
                    "label": "pickup",
                    "rest_count": None,
                    "confidence": "medium",
                },
                {
                    "measure_id": "p1_s0_m1",
                    "label": "normal",
                    "rest_count": None,
                    "confidence": "high",
                },
            ],
            "warnings": [],
        }
        message = {"content": [{"type": "text", "text": json.dumps(provider_payload)}]}

        with (
            patch.object(WORKER, "_resolve_ai_crop_pdf_source", return_value=(Path("/tmp/audiveris_out_corrected.pdf"), "corrected")),
            patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
            patch.object(WORKER, "_anthropic_messages_create", return_value=message),
            patch.object(WORKER.fitz, "open", return_value=fake_doc),
            patch.object(WORKER.fitz, "Rect", _FakeRect),
        ):
            result = WORKER._generate_ai_suggestions_for_system_batch(
                "111",
                111,
                systems,
                system_row,
                system_measures,
                "test-state",
                artifacts,
            )

        by_measure_id = result.get("by_measure_id") or {}
        self.assertEqual(sorted(by_measure_id.keys()), ["p1_s0_m0"])
        self.assertEqual((by_measure_id.get("p1_s0_m0") or {}).get("label"), "pickup")
        self.assertEqual((result.get("model") or ""), "claude-sonnet-4-6")
        self.assertEqual((result.get("pdf_source") or ""), "corrected")

    def test_ai_suggest_step_real_system_batch_path_no_longer_crashes(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = WORKER._empty_ai_suggestions_state(111, "test-state", 3)
        mapping_summary["ai_suggest_run"] = {
            "status": "running",
            "started_at_utc": "2026-05-05T00:00:00Z",
            "updated_at_utc": "2026-05-05T00:00:00Z",
            "completed_at_utc": None,
            "failed_at_utc": None,
            "systems_total": 2,
            "systems_completed": 0,
            "next_system_index": 0,
            "source_run_id": 111,
            "source_state_version": "test-state",
            "last_error": None,
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest/step", method="POST", headers={}, files={}, json={})
        provider_payload = {
            "provider": "claude",
            "suggestions": [
                {
                    "measure_id": "p1_s0_m0",
                    "label": "pickup",
                    "rest_count": None,
                    "confidence": "medium",
                },
                {
                    "measure_id": "p1_s0_m1",
                    "label": "normal",
                    "rest_count": None,
                    "confidence": "high",
                },
            ],
            "warnings": [],
        }
        message = {"content": [{"type": "text", "text": json.dumps(provider_payload)}]}
        fake_doc = _FakeDoc([_FakePage(_FakeRect(0, 0, 200, 160))])

        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_editable_state_version", return_value="test-state"),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
            patch.object(WORKER, "_resolve_ai_crop_pdf_source", return_value=(Path("/tmp/audiveris_out_corrected.pdf"), "corrected")),
            patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
            patch.object(WORKER, "_anthropic_messages_create", return_value=message),
            patch.object(WORKER.fitz, "open", return_value=fake_doc),
            patch.object(WORKER.fitz, "Rect", _FakeRect),
            patch.object(WORKER, "_ai_suggest_debug_enabled", return_value=True),
            patch.object(WORKER, "_signed_http_url_for_gs", return_value="https://signed/debug"),
            patch.object(WORKER, "_load_ai_debug_batch_trace", return_value=None),
            patch.object(WORKER, "_current_ai_crop_pdf_source_label", return_value="corrected"),
            patch.object(WORKER, "_upload_bytes_to_gcs", return_value=None),
            patch.object(
                WORKER,
                "_write_ai_debug_batch_trace",
                side_effect=lambda payload, _artifacts: {
                    "enabled": True,
                    "trace_uri": "gs://x/output/artifacts/ai_debug_crops/ai_batch_trace.json",
                    "trace_http": "https://signed/debug-trace",
                    "pdf_source": str(payload.get("pdf_source") or "baseline"),
                    "measure_count": int(payload.get("measure_count") or 0),
                    "batched_count": int(payload.get("batched_count") or 0),
                    "skipped_count": int(payload.get("skipped_count") or 0),
                },
            ),
        ):
            body, status = _unpack(WORKER.ai_suggest_job_step("111"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "running")
        self.assertEqual((body.get("ai_suggest_run") or {}).get("systems_completed"), 1)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("next_system_index"), 1)
        _reference_content, expected_reference_examples_attached = WORKER._build_old_style_multi_rest_reference_content()
        self.assertEqual(body.get("reference_examples_attached"), expected_reference_examples_attached)
        self.assertEqual(sorted(((body.get("ai_suggestions") or {}).get("by_measure_id") or {}).keys()), ["p1_s0_m0"])
        self.assertEqual(((body.get("debug_crops") or {}).get("pdf_source")), "corrected")
        self.assertEqual(
            ((body.get("debug_crops") or {}).get("reference_examples_attached")),
            expected_reference_examples_attached,
        )

    def test_ai_suggest_step_final_system_marks_completed(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = {
            "version": "ai_suggestions_v1",
            "generated_at_utc": "2026-05-05T00:00:00Z",
            "provider": "claude",
            "model": "claude-test",
            "source_run_id": 111,
            "by_measure_id": {
                "p1_s0_m0": {"label": "pickup", "rest_count": None, "confidence": "medium"}
            },
            "warnings": [],
            "summary": {"systems_processed": 1, "measures_seen": 2, "suggestions_kept": 1, "normal_measures_omitted": 1},
        }
        mapping_summary["ai_suggest_run"] = {
            "status": "running",
            "started_at_utc": "2026-05-05T00:00:00Z",
            "updated_at_utc": "2026-05-05T00:00:00Z",
            "completed_at_utc": None,
            "failed_at_utc": None,
            "systems_total": 2,
            "systems_completed": 1,
            "next_system_index": 1,
            "source_run_id": 111,
            "source_state_version": "test-state",
            "last_error": None,
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest/step", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_editable_state_version", return_value="test-state"),
            patch.object(
                WORKER,
                "_generate_ai_suggestions_for_system_batch",
                return_value={
                    "version": "ai_suggestions_v1",
                    "generated_at_utc": "2026-05-05T00:00:01Z",
                    "provider": "claude",
                    "model": "claude-test",
                    "source_run_id": 111,
                    "by_measure_id": {
                        "p1_s1_m0": {"label": "uncertain", "rest_count": None, "confidence": "low", "maybe_label": "multi_measure_rest", "maybe_rest_count": 2}
                    },
                    "warnings": [],
                    "summary": {"systems_processed": 1, "measures_seen": 1, "suggestions_kept": 1, "normal_measures_omitted": 0},
                },
            ),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.ai_suggest_job_step("111"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "completed")
        self.assertEqual((body.get("ai_suggest_run") or {}).get("systems_completed"), 2)
        self.assertEqual((body.get("ai_suggest_run") or {}).get("next_system_index"), 2)
        by_measure_id = ((body.get("ai_suggestions") or {}).get("by_measure_id") or {})
        self.assertEqual(sorted(by_measure_id.keys()), ["p1_s0_m0", "p1_s1_m0"])
        self.assertEqual((body.get("ai_suggestions") or {}).get("model"), "claude-sonnet-4-6")
        self.assertEqual((((body.get("ai_suggestions") or {}).get("summary") or {}).get("systems_processed")), 2)

    def test_ai_suggest_step_failure_marks_failed_and_keeps_partial_suggestions(self):
        artifacts = self._sample_artifacts()
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = {
            "version": "ai_suggestions_v1",
            "generated_at_utc": "2026-05-05T00:00:00Z",
            "provider": "claude",
            "model": "claude-test",
            "source_run_id": 111,
            "by_measure_id": {
                "p1_s0_m0": {"label": "pickup", "rest_count": None, "confidence": "medium"}
            },
            "warnings": [],
            "summary": {"systems_processed": 1, "measures_seen": 2, "suggestions_kept": 1, "normal_measures_omitted": 1},
        }
        original = deepcopy(mapping_summary["ai_suggestions"])
        mapping_summary["ai_suggest_run"] = {
            "status": "running",
            "started_at_utc": "2026-05-05T00:00:00Z",
            "updated_at_utc": "2026-05-05T00:00:00Z",
            "completed_at_utc": None,
            "failed_at_utc": None,
            "systems_total": 2,
            "systems_completed": 1,
            "next_system_index": 1,
            "source_run_id": 111,
            "source_state_version": "test-state",
            "last_error": None,
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest/step", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_editable_state_version", return_value="test-state"),
            patch.object(
                WORKER,
                "_generate_ai_suggestions_for_system_batch",
                side_effect=WORKER.AiSuggestError(provider_status=504, detail="timeout"),
            ),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.ai_suggest_job_step("111"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "failed")
        self.assertEqual(((body.get("error") or {}).get("detail")), "timeout")
        self.assertEqual(mapping_summary.get("ai_suggestions"), original)
        self.assertEqual((((mapping_summary.get("ai_suggest_run") or {}).get("status"))), "failed")

    def test_anthropic_messages_create_retries_overload_then_succeeds(self):
        overload_1 = WORKER.AiSuggestError(provider_status=529, detail='{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}')
        overload_2 = WORKER.AiSuggestError(provider_status=529, detail='{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}')
        success = {"content": [{"type": "text", "text": "{}"}], "model": "claude-sonnet-4-6"}
        with (
            patch.object(WORKER, "_anthropic_messages_create_once", side_effect=[overload_1, overload_2, success]) as create_once,
            patch.object(WORKER.time, "sleep", return_value=None) as sleep_mock,
        ):
            result = WORKER._anthropic_messages_create({"model": "claude-sonnet-4-6"})

        self.assertEqual(result, success)
        self.assertEqual(create_once.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [2.0, 5.0])

    def test_anthropic_messages_create_fails_after_overload_retries(self):
        overload_1 = WORKER.AiSuggestError(provider_status=529, detail='{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}')
        overload_2 = WORKER.AiSuggestError(provider_status=529, detail='{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}')
        overload_3 = WORKER.AiSuggestError(provider_status=529, detail='{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}')
        with (
            patch.object(WORKER, "_anthropic_messages_create_once", side_effect=[overload_1, overload_2, overload_3]) as create_once,
            patch.object(WORKER.time, "sleep", return_value=None) as sleep_mock,
        ):
            with self.assertRaises(WORKER.AiSuggestError) as ctx:
                WORKER._anthropic_messages_create({"model": "claude-sonnet-4-6"})

        self.assertEqual(create_once.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [2.0, 5.0])
        self.assertEqual(ctx.exception.provider_status, 529)
        self.assertEqual(getattr(ctx.exception, "retry_attempts", None), 3)
        self.assertIn("overload_retry_attempts=3", ctx.exception.detail)

    def test_anthropic_messages_create_does_not_retry_non_overload_error(self):
        malformed = WORKER.AiSuggestError(provider_status=502, detail="malformed_response: invalid json")
        with (
            patch.object(WORKER, "_anthropic_messages_create_once", side_effect=malformed) as create_once,
            patch.object(WORKER.time, "sleep", return_value=None) as sleep_mock,
        ):
            with self.assertRaises(WORKER.AiSuggestError):
                WORKER._anthropic_messages_create({"model": "claude-sonnet-4-6"})

        self.assertEqual(create_once.call_count, 1)
        self.assertEqual(sleep_mock.call_count, 0)

    def test_dismiss_ai_suggestion_removes_only_target(self):
        artifacts = self._sample_artifacts()
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = {
            "version": "ai_suggestions_v1",
            "generated_at_utc": "2026-04-30T12:00:00Z",
            "provider": "claude",
            "model": "test",
            "source_run_id": 111,
            "by_measure_id": {
                "p1_s0_m0": {"label": "pickup", "rest_count": None, "confidence": "medium"},
                "p1_s1_m0": {"label": "uncertain", "rest_count": None, "confidence": "low"},
            },
            "warnings": [],
            "summary": {"systems_processed": 2, "measures_seen": 3, "suggestions_kept": 2, "normal_measures_omitted": 1},
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggestions/p1_s0_m0/dismiss", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.dismiss_ai_suggestion("111", "p1_s0_m0"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("dismissed_measure_id"), "p1_s0_m0")
        by_measure_id = ((body.get("ai_suggestions") or {}).get("by_measure_id") or {})
        self.assertEqual(sorted(by_measure_id.keys()), ["p1_s1_m0"])
        self.assertEqual(((body.get("ai_suggestions") or {}).get("summary") or {}).get("suggestions_kept"), 1)

    def test_relabel_clears_touched_ai_suggestion(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = {
            "version": "ai_suggestions_v1",
            "generated_at_utc": "2026-04-30T12:00:00Z",
            "provider": "claude",
            "model": "test",
            "source_run_id": 111,
            "by_measure_id": {
                "p1_s0_m0": {"label": "pickup", "rest_count": None, "confidence": "medium"},
                "p1_s1_m0": {"label": "uncertain", "rest_count": None, "confidence": "low"},
            },
            "warnings": [],
            "summary": {"systems_processed": 2, "measures_seen": 3, "suggestions_kept": 2, "normal_measures_omitted": 1},
        }

        WORKER.request = SimpleNamespace(
            path="/api/omr/jobs/111/relabel",
            method="POST",
            headers={},
            files={},
            json={"edits": [{"type": "set_pickup_measure", "measure_id": "p1_s0_m0", "value": True}]},
        )
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_download_gcs_to_file", return_value=None),
            patch.object(WORKER, "_render_corrected_pdf", return_value=2),
            patch.object(WORKER, "_upload_file_to_gcs", return_value=None),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.relabel_job("111"))

        self.assertEqual(status, 200)
        remaining = ((mapping_summary.get("ai_suggestions") or {}).get("by_measure_id") or {})
        self.assertEqual(sorted(remaining.keys()), ["p1_s1_m0"])
        self.assertEqual((((mapping_summary.get("ai_suggestions") or {}).get("summary") or {}).get("suggestions_kept")), 1)

    def test_normalize_ai_suggestions_result_salvages_missing_maybe_rest_count_and_bad_warnings(self):
        editable_state = (self._sample_mapping_summary().get("editable_state") or {})
        raw_result = {
            "provider": "claude",
            "model": "claude-opus-4-5",
            "suggestions": [
                {
                    "measure_id": "p1_s0_m0",
                    "label": "uncertain",
                    "rest_count": None,
                    "confidence": "medium",
                    "maybe_label": "multi_measure_rest",
                },
                {
                    "measure_id": "p1_s0_m1",
                    "label": "normal",
                    "rest_count": 4,
                    "confidence": "high",
                },
                {
                    "measure_id": "p1_s1_m0",
                    "label": "pickup",
                    "rest_count": None,
                    "confidence": "medium",
                    "maybe_label": "pickup",
                },
            ],
            "warnings": [
                "bad-row",
                {"type": "", "message": "missing type"},
                {"type": "note", "message": "kept"},
            ],
        }

        normalized = WORKER._normalize_ai_suggestions_result(raw_result, editable_state, 111, "test-state")

        self.assertEqual(normalized.get("model"), "claude-sonnet-4-6")
        by_measure_id = normalized.get("by_measure_id") or {}
        self.assertEqual((by_measure_id.get("p1_s0_m0") or {}).get("label"), "uncertain")
        self.assertNotIn("maybe_label", by_measure_id.get("p1_s0_m0") or {})
        self.assertEqual((by_measure_id.get("p1_s1_m0") or {}).get("label"), "pickup")
        warnings = normalized.get("warnings") or []
        self.assertTrue(any((row or {}).get("type") == "note" for row in warnings))
        self.assertTrue(any((row or {}).get("type") == "normalization_adjusted" for row in warnings))
        self.assertEqual(((normalized.get("summary") or {}).get("normal_measures_omitted")), 1)

    def test_normalize_ai_suggestions_result_downgrades_bad_multirest_and_invalid_confidence(self):
        editable_state = (self._sample_mapping_summary().get("editable_state") or {})
        raw_result = {
            "provider": "claude",
            "suggestions": [
                {
                    "measure_id": "p1_s0_m0",
                    "label": "multi_measure_rest",
                    "rest_count": 1,
                    "confidence": "bad-confidence",
                },
                {
                    "measure_id": "p1_s0_m1",
                    "label": "uncertain",
                    "rest_count": None,
                    "confidence": "low",
                    "maybe_label": "pickup",
                    "maybe_rest_count": 3,
                },
                {
                    "measure_id": "p1_s1_m0",
                    "label": "normal",
                    "rest_count": None,
                    "confidence": "medium",
                },
            ],
        }

        normalized = WORKER._normalize_ai_suggestions_result(raw_result, editable_state, 111, "test-state")

        by_measure_id = normalized.get("by_measure_id") or {}
        first = by_measure_id.get("p1_s0_m0") or {}
        self.assertEqual(first.get("label"), "uncertain")
        self.assertEqual(first.get("confidence"), "low")
        second = by_measure_id.get("p1_s0_m1") or {}
        self.assertEqual(second.get("label"), "uncertain")
        self.assertNotIn("maybe_label", second)
        warnings = normalized.get("warnings") or []
        self.assertGreaterEqual(
            len([row for row in warnings if (row or {}).get("type") == "normalization_adjusted"]),
            2,
        )

    def test_parse_anthropic_suggestions_message_drops_model_field(self):
        message = {
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "text",
                    "text": """```json
{"model":"claude-opus-4-5","suggestions":[{"measure_id":"p1_s0_m0","label":"pickup","rest_count":null,"confidence":"medium"}],"warnings":[]}
```""",
                }
            ],
        }
        parsed = WORKER._parse_anthropic_suggestions_message(message)
        self.assertEqual(parsed.get("provider"), "claude")
        self.assertNotIn("model", parsed)
        self.assertEqual(((parsed.get("suggestions") or [])[0] or {}).get("measure_id"), "p1_s0_m0")

    def test_parse_anthropic_suggestions_message_accepts_json_fence(self):
        message = {
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "text",
                    "text": """```json
{"suggestions":[{"measure_id":"p1_s0_m0","label":"pickup","rest_count":null,"confidence":"medium"}],"warnings":[]}
```""",
                }
            ],
        }
        parsed = WORKER._parse_anthropic_suggestions_message(message)
        self.assertEqual(parsed.get("provider"), "claude")
        self.assertNotIn("model", parsed)
        self.assertEqual(((parsed.get("suggestions") or [])[0] or {}).get("measure_id"), "p1_s0_m0")

    def test_build_system_measure_request_includes_stronger_pickup_rules(self):
        mapping_summary = self._sample_mapping_summary()
        editable_state = mapping_summary.get("editable_state") or {}
        system_row = (editable_state.get("systems") or [])[0]
        measure_rows = [row for row in (editable_state.get("measures") or []) if row.get("system_id") == "p1_s0"]
        page = _FakePage(_FakeRect(0, 0, 200, 160))

        with (
            patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
            patch.object(WORKER.fitz, "Rect", _FakeRect),
        ):
            payload, reference_examples_attached = WORKER._build_system_measure_request(
                "111",
                111,
                system_row,
                measure_rows,
                page,
                pdf_source="corrected",
            )

        _reference_content, expected_reference_examples_attached = WORKER._build_old_style_multi_rest_reference_content()
        self.assertEqual(reference_examples_attached, expected_reference_examples_attached)
        self.assertNotIn("reference_examples_attached", payload)
        content = (((payload.get("messages") or [])[0] or {}).get("content")) or []
        intro = json.loads((content[0] or {}).get("text") or "{}")
        rules = ((intro.get("instructions") or {}).get("rules")) or []
        rules_text = "\n".join(str(row) for row in rules)
        measure_meta = intro.get("measures") or []
        self.assertIn("Use the visible time signature in the crop to judge completeness.", rules_text)
        self.assertIn("Each measure may include time_signature and time_signature_source metadata.", rules_text)
        self.assertIn("If time_signature metadata is provided, use it even when the crop does not visibly show the symbol.", rules_text)
        self.assertIn("If the first measure is clearly too short for the visible time signature, label pickup.", rules_text)
        self.assertIn("Examples: in 2/4, one quarter note in the first measure is pickup;", rules_text)
        self.assertIn("For measures that are not the first measure of the score, never use later time-signature context to label pickup in this version.", rules_text)
        self.assertIn("If the time signature is unclear but the first measure looks short, label uncertain with maybe_label pickup.", rules_text)
        self.assertEqual((measure_meta[0] or {}).get("time_signature"), "3/4")
        self.assertEqual((measure_meta[0] or {}).get("time_signature_source"), "explicit")
        self.assertEqual((measure_meta[1] or {}).get("time_signature_source"), "inherited")

    def test_build_system_measure_request_includes_old_style_multi_rest_guidance(self):
        mapping_summary = self._sample_mapping_summary()
        editable_state = mapping_summary.get("editable_state") or {}
        system_row = (editable_state.get("systems") or [])[0]
        measure_rows = [row for row in (editable_state.get("measures") or []) if row.get("system_id") == "p1_s0"]
        page = _FakePage(_FakeRect(0, 0, 200, 160))

        with (
            patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
            patch.object(WORKER.fitz, "Rect", _FakeRect),
        ):
            payload, _reference_examples_attached = WORKER._build_system_measure_request(
                "111",
                111,
                system_row,
                measure_rows,
                page,
                pdf_source="corrected",
            )

        content = (((payload.get("messages") or [])[0] or {}).get("content")) or []
        intro = json.loads((content[0] or {}).get("text") or "{}")
        rules = ((intro.get("instructions") or {}).get("rules")) or []
        rules_text = "\n".join(str(row) for row in rules)
        self.assertIn("A multi-measure rest may use either the modern H-bar style or an older style made from a horizontal bar plus one or more vertical bars.", rules_text)
        self.assertIn("In the older style, the vertical bars may be short or long, and there may be more than one.", rules_text)
        self.assertIn("A visible count of 2 or more above that old-style symbol is strong evidence for multi_measure_rest.", rules_text)
        self.assertIn("A plain one-measure rest without the old-style vertical-bar structure is normal, not multi_measure_rest.", rules_text)

    def test_build_system_measure_request_includes_reference_examples_before_real_measures(self):
        mapping_summary = self._sample_mapping_summary()
        editable_state = mapping_summary.get("editable_state") or {}
        system_row = (editable_state.get("systems") or [])[0]
        measure_rows = [row for row in (editable_state.get("measures") or []) if row.get("system_id") == "p1_s0"]
        page = _FakePage(_FakeRect(0, 0, 200, 160))

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "old_style_rest_negative_1.png").write_bytes(b"negative-reference")
            (tmp_path / "old_style_rest_positive_3.png").write_bytes(b"positive-reference")
            with (
                patch.object(WORKER, "AI_REFERENCE_EXAMPLES_DIR", tmp_path),
                patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
                patch.object(WORKER.fitz, "Rect", _FakeRect),
            ):
                payload, reference_examples_attached = WORKER._build_system_measure_request(
                    "111",
                    111,
                    system_row,
                    measure_rows,
                    page,
                    pdf_source="corrected",
                )

        self.assertEqual(reference_examples_attached, 2)
        self.assertNotIn("reference_examples_attached", payload)
        content = (((payload.get("messages") or [])[0] or {}).get("content")) or []
        self.assertEqual((content[0] or {}).get("type"), "text")
        self.assertIn("Reference examples for old-style multi-measure rest recognition.", (content[1] or {}).get("text") or "")
        self.assertIn("visible count 1", (content[2] or {}).get("text") or "")
        self.assertEqual((content[3] or {}).get("type"), "image")
        self.assertIn("visible count 3", (content[4] or {}).get("text") or "")
        self.assertEqual((content[5] or {}).get("type"), "image")
        self.assertEqual(json.loads((content[6] or {}).get("text") or "{}").get("measure_id"), "p1_s0_m0")

    def test_build_system_measure_request_skips_missing_reference_examples(self):
        mapping_summary = self._sample_mapping_summary()
        editable_state = mapping_summary.get("editable_state") or {}
        system_row = (editable_state.get("systems") or [])[0]
        measure_rows = [row for row in (editable_state.get("measures") or []) if row.get("system_id") == "p1_s0"]
        page = _FakePage(_FakeRect(0, 0, 200, 160))

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with (
                patch.object(WORKER, "AI_REFERENCE_EXAMPLES_DIR", tmp_path),
                patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
                patch.object(WORKER.fitz, "Rect", _FakeRect),
            ):
                payload, reference_examples_attached = WORKER._build_system_measure_request(
                    "111",
                    111,
                    system_row,
                    measure_rows,
                    page,
                    pdf_source="corrected",
                )

        self.assertEqual(reference_examples_attached, 0)
        self.assertNotIn("reference_examples_attached", payload)
        content = (((payload.get("messages") or [])[0] or {}).get("content")) or []
        self.assertEqual(json.loads((content[1] or {}).get("text") or "{}").get("measure_id"), "p1_s0_m0")
        self.assertEqual(len([row for row in content if (row or {}).get("type") == "image"]), len(measure_rows))

    def test_generate_ai_suggestions_system_batch_does_not_send_local_debug_field_to_provider(self):
        artifacts = self._sample_artifacts()
        mapping_summary = self._sample_mapping_summary()
        editable_state = mapping_summary.get("editable_state") or {}
        systems = editable_state.get("systems") or []
        system_row = systems[0]
        system_measures = [row for row in (editable_state.get("measures") or []) if row.get("system_id") == "p1_s0"]
        fake_doc = _FakeDoc([_FakePage(_FakeRect(0, 0, 200, 160))])
        provider_payload = {
            "provider": "claude",
            "suggestions": [
                {
                    "measure_id": "p1_s0_m0",
                    "label": "pickup",
                    "rest_count": None,
                    "confidence": "medium",
                },
                {
                    "measure_id": "p1_s0_m1",
                    "label": "normal",
                    "rest_count": None,
                    "confidence": "high",
                },
            ],
            "warnings": [],
        }
        message = {"content": [{"type": "text", "text": json.dumps(provider_payload)}]}
        captured_payloads: list[dict] = []

        def _capture_payload(payload):
            captured_payloads.append(payload)
            return message

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "old_style_rest_negative_1.png").write_bytes(b"negative-reference")
            (tmp_path / "old_style_rest_positive_3.png").write_bytes(b"positive-reference")
            with (
                patch.object(WORKER, "AI_REFERENCE_EXAMPLES_DIR", tmp_path),
                patch.object(WORKER, "_resolve_ai_crop_pdf_source", return_value=(Path("/tmp/audiveris_out_corrected.pdf"), "corrected")),
                patch.object(WORKER, "_render_measure_crop_png", return_value=b"png-bytes"),
                patch.object(WORKER, "_anthropic_messages_create", side_effect=_capture_payload),
                patch.object(WORKER.fitz, "open", return_value=fake_doc),
                patch.object(WORKER.fitz, "Rect", _FakeRect),
            ):
                result = WORKER._generate_ai_suggestions_for_system_batch(
                    "111",
                    111,
                    systems,
                    system_row,
                    system_measures,
                    "test-state",
                    artifacts,
                )

        self.assertEqual(len(captured_payloads), 1)
        self.assertNotIn("reference_examples_attached", captured_payloads[0])
        self.assertEqual(result.get("reference_examples_attached"), 2)

    def test_profile_system_layouts_flags_short_partial_staff(self):
        systems = [
            {"system_id": "p1_s0", "page": 1, "system_index": 0, "anchor": {"y_top": 0, "y_bottom": 40}},
            {"system_id": "p1_s1", "page": 1, "system_index": 1, "anchor": {"y_top": 50, "y_bottom": 70}},
            {"system_id": "p1_s2", "page": 1, "system_index": 2, "anchor": {"y_top": 80, "y_bottom": 120}},
        ]

        profile = WORKER._profile_system_layouts(systems)

        self.assertEqual(profile.get("suspicious_system_ids"), {"p1_s1"})
        self.assertFalse(bool(systems[0].get("suspicious_partial_staff")))
        self.assertTrue(bool(systems[1].get("suspicious_partial_staff")))
        self.assertFalse(bool(systems[2].get("suspicious_partial_staff")))

    def test_reassign_measures_prefers_normal_system_over_suspicious_partial_staff(self):
        systems = [
            {"system_id": "p1_s0", "page": 1, "system_index": 0, "anchor": {"y_top": 0, "y_bottom": 40}},
            {"system_id": "p1_s1", "page": 1, "system_index": 1, "anchor": {"y_top": 30, "y_bottom": 50}},
            {"system_id": "p1_s2", "page": 1, "system_index": 2, "anchor": {"y_top": 80, "y_bottom": 120}},
        ]
        measures = [
            {
                "measure_id": "p1_s1_m0",
                "system_id": "p1_s1",
                "page": 1,
                "system_index": 1,
                "measure_local_index": 0,
                "x_left": 25,
                "y_top": 32,
                "y_bottom": 38,
            }
        ]

        reassigned = WORKER._reassign_measures_to_nearest_system(systems, measures)

        self.assertEqual(reassigned, 1)
        self.assertEqual(measures[0].get("system_id"), "p1_s0")
        self.assertEqual(measures[0].get("system_index"), 0)
        self.assertEqual(measures[0].get("measure_id"), "p1_s0_m0")

    def test_refresh_editable_state_qa_warns_for_empty_suspicious_system(self):
        editable_state = {
            "systems": [
                {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1", "anchor": {"y_top": 0, "y_bottom": 40}},
                {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "5", "anchor": {"y_top": 50, "y_bottom": 70}},
                {"system_id": "p1_s2", "page": 1, "system_index": 2, "current_value": "9", "anchor": {"y_top": 80, "y_bottom": 120}},
            ],
            "measures": [
                {"measure_id": "p1_s0_m0", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "y_top": 0, "y_bottom": 20},
                {"measure_id": "p1_s2_m0", "system_id": "p1_s2", "page": 1, "system_index": 2, "measure_local_index": 0, "x_left": 10, "y_top": 80, "y_bottom": 100},
            ],
        }

        qa = WORKER._refresh_editable_state_qa(editable_state, editable_state["systems"], editable_state["measures"])
        warning_types = {(row or {}).get("type") for row in (qa.get("warnings") or [])}

        self.assertEqual(qa.get("status"), "warning")
        self.assertIn("suspicious_partial_staff", warning_types)
        self.assertIn("system_has_no_measures", warning_types)
        self.assertEqual(qa.get("warning_pages"), [1])

    def test_refresh_editable_state_qa_warns_on_duplicate_later_system_starts(self):
        editable_state = {
            "systems": [
                {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1", "anchor": {"y_top": 0, "y_bottom": 40}},
                {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "42", "anchor": {"y_top": 50, "y_bottom": 90}},
                {"system_id": "p1_s2", "page": 1, "system_index": 2, "current_value": "42", "anchor": {"y_top": 100, "y_bottom": 140}},
                {"system_id": "p1_s3", "page": 1, "system_index": 3, "current_value": "42", "anchor": {"y_top": 150, "y_bottom": 190}},
            ],
            "measures": [
                {"measure_id": "p1_s0_m0", "system_id": "p1_s0", "page": 1, "system_index": 0, "measure_local_index": 0, "x_left": 10, "y_top": 0, "y_bottom": 20},
                {"measure_id": "p1_s1_m0", "system_id": "p1_s1", "page": 1, "system_index": 1, "measure_local_index": 0, "x_left": 10, "y_top": 50, "y_bottom": 70},
                {"measure_id": "p1_s2_m0", "system_id": "p1_s2", "page": 1, "system_index": 2, "measure_local_index": 0, "x_left": 10, "y_top": 100, "y_bottom": 120},
                {"measure_id": "p1_s3_m0", "system_id": "p1_s3", "page": 1, "system_index": 3, "measure_local_index": 0, "x_left": 10, "y_top": 150, "y_bottom": 170},
            ],
        }

        qa = WORKER._refresh_editable_state_qa(editable_state, editable_state["systems"], editable_state["measures"])
        warning_types = {(row or {}).get("type") for row in (qa.get("warnings") or [])}

        self.assertEqual(qa.get("status"), "warning")
        self.assertIn("duplicate_later_system_start", warning_types)
        self.assertEqual(qa.get("warning_pages"), [1])

    def test_refresh_editable_state_systems_and_measures_keeps_clean_pages_quiet(self):
        mapping_summary = deepcopy(self._sample_mapping_summary())
        editable_state = mapping_summary.get("editable_state") or {}

        systems, measures, reassign_count, qa = WORKER._refresh_editable_state_systems_and_measures(editable_state)

        self.assertEqual(reassign_count, 0)
        self.assertEqual(len(systems), 2)
        self.assertEqual(len(measures), 3)
        self.assertEqual(qa.get("status"), "ok")
        self.assertEqual(qa.get("warning_count"), 0)
        self.assertEqual(qa.get("warning_pages"), [])
        self.assertEqual(qa.get("warnings"), [])
        self.assertTrue(all(not row.get("suspicious_partial_staff") for row in systems))



if __name__ == "__main__":
    unittest.main()
