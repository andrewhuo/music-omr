import importlib.util
import os
import sys
import types
import unittest
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
        WORKER.request = SimpleNamespace(path="", method="GET", headers={}, files={}, json={})
        WORKER._PENDING_DISPATCHES.clear()

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
        artifacts = {
            "audiveris_out_pdf": "gs://x/output/audiveris_out.pdf",
            "audiveris_out_corrected_pdf": "gs://x/output/audiveris_out_corrected.pdf",
            "run_info": "gs://x/output/artifacts/run_info.json",
            "mapping_summary": "gs://x/output/artifacts/mapping_summary.json",
        }
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
        self.assertEqual((body.get("editable_state") or {}).get("staff_boxes"), [])

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
        artifacts = {
            "audiveris_out_pdf": "gs://x/output/audiveris_out.pdf",
            "audiveris_out_corrected_pdf": "gs://x/output/audiveris_out_corrected.pdf",
            "run_info": "gs://x/output/artifacts/run_info.json",
            "mapping_summary": "gs://x/output/artifacts/mapping_summary.json",
        }
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



if __name__ == "__main__":
    unittest.main()
