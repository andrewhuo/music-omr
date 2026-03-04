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

        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, None, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(
                artifacts,
                {
                    "editable_state": {
                        "version": "system_state_v1",
                        "qa": {"ok": True, "total_systems": 1},
                        "systems": [
                            {
                                "system_id": "p1_s0",
                                "page": 1,
                                "system_index": 0,
                                "current_value": "1",
                                "anchor": {"x": 10.0, "y_top": 20.0, "y_bottom": 40.0},
                                "in_bounds": True,
                                "guide_build_source": "primary",
                            }
                        ],
                    }
                },
                111,
            )),
            patch.object(WORKER, "_download_gcs_to_file", return_value=None),
            patch.object(WORKER, "_render_corrected_pdf", return_value=1),
            patch.object(WORKER, "_upload_file_to_gcs", return_value=None),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
            patch.object(WORKER, "_artifact_uris_for_run", return_value=artifacts),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
        ):
            WORKER.request = SimpleNamespace(
                path="/api/omr/jobs/111/relabel",
                method="POST",
                headers={},
                files={},
                json={"edits": [{"type": "set_system_start", "system_id": "p1_s0", "value": 4}]},
            )
            body, status = _unpack(WORKER.relabel_job("111"))
        self.assertEqual(status, 200)
        self.assertEqual(body.get("artifacts_http"), artifacts_http)


if __name__ == "__main__":
    unittest.main()
