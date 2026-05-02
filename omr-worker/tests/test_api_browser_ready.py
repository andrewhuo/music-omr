import importlib.util
import os
import sys
import types
import unittest
from copy import deepcopy
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
                "systems": [
                    {"system_id": "p1_s0", "page": 1, "system_index": 0, "current_value": "1", "anchor": {"x": 10, "y_top": 20, "y_bottom": 60}},
                    {"system_id": "p1_s1", "page": 1, "system_index": 1, "current_value": "3", "anchor": {"x": 10, "y_top": 80, "y_bottom": 120}},
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
        self.assertEqual((body.get("editable_state") or {}).get("rest_measures"), {})
        self.assertEqual((body.get("editable_state") or {}).get("pickup_measures"), {})
        self.assertEqual((body.get("editable_state") or {}).get("staff_boxes"), [])
        self.assertIsNone(body.get("ai_suggestions"))

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

    def test_ai_suggest_success_persists_filtered_suggestions(self):
        artifacts = self._sample_artifacts()
        artifacts_http = {k: f"https://signed/{k}" for k in artifacts}
        mapping_summary = self._sample_mapping_summary()
        raw_result = {
            "provider": "claude",
            "model": "claude-test",
            "suggestions": [
                {"measure_id": "p1_s0_m0", "label": "pickup", "rest_count": None, "confidence": "medium"},
                {"measure_id": "p1_s0_m1", "label": "normal", "rest_count": None, "confidence": "high"},
                {
                    "measure_id": "p1_s1_m0",
                    "label": "uncertain",
                    "rest_count": None,
                    "confidence": "low",
                    "maybe_label": "multi_measure_rest",
                    "maybe_rest_count": 2,
                },
            ],
            "warnings": [{"type": "low_confidence_page", "system_id": "p1_s1", "system_index": 1, "message": "check m2"}],
        }
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(WORKER, "_generate_ai_suggestions_for_job", return_value=raw_result),
            patch.object(WORKER, "_artifact_http_uris_for_run", return_value=artifacts_http),
            patch.object(WORKER, "_upload_json_to_gcs", return_value=None),
        ):
            body, status = _unpack(WORKER.ai_suggest_job("111"))

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "succeeded")
        ai_suggestions = body.get("ai_suggestions") or {}
        by_measure_id = ai_suggestions.get("by_measure_id") or {}
        self.assertEqual(sorted(by_measure_id.keys()), ["p1_s0_m0", "p1_s1_m0"])
        self.assertEqual((by_measure_id.get("p1_s0_m0") or {}).get("label"), "pickup")
        self.assertEqual((by_measure_id.get("p1_s1_m0") or {}).get("maybe_label"), "multi_measure_rest")
        self.assertEqual((by_measure_id.get("p1_s1_m0") or {}).get("maybe_rest_count"), 2)
        summary = ai_suggestions.get("summary") or {}
        self.assertEqual(summary.get("measures_seen"), 3)
        self.assertEqual(summary.get("suggestions_kept"), 2)
        self.assertEqual(summary.get("normal_measures_omitted"), 1)
        self.assertIn("ai_suggestions", mapping_summary)

    def test_ai_suggest_failure_keeps_prior_suggestions(self):
        artifacts = self._sample_artifacts()
        mapping_summary = self._sample_mapping_summary()
        mapping_summary["ai_suggestions"] = {
            "version": "ai_suggestions_v1",
            "generated_at_utc": "2026-04-30T12:00:00Z",
            "provider": "claude",
            "model": "old",
            "source_run_id": 111,
            "by_measure_id": {"p1_s0_m1": {"label": "pickup", "rest_count": None, "confidence": "medium"}},
            "warnings": [],
            "summary": {"systems_processed": 2, "measures_seen": 3, "suggestions_kept": 1, "normal_measures_omitted": 2},
        }
        original = deepcopy(mapping_summary["ai_suggestions"])
        WORKER.request = SimpleNamespace(path="/api/omr/jobs/111/ai-suggest", method="POST", headers={}, files={}, json={})
        with (
            patch.object(WORKER, "_resolve_run_id_from_job_id", return_value=(111, {}, None)),
            patch.object(WORKER, "_load_mapping_for_run", return_value=(artifacts, mapping_summary, 111)),
            patch.object(
                WORKER,
                "_generate_ai_suggestions_for_job",
                side_effect=WORKER.AiSuggestError(provider_status=504, detail="timeout"),
            ),
        ):
            body, status = _unpack(WORKER.ai_suggest_job("111"))

        self.assertEqual(status, 504)
        self.assertEqual(body.get("status"), "failed")
        self.assertEqual(((body.get("error") or {}).get("detail")), "timeout")
        self.assertEqual(mapping_summary.get("ai_suggestions"), original)

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
        self.assertEqual(parsed.get("model"), "claude-sonnet-4-20250514")
        self.assertEqual(((parsed.get("suggestions") or [])[0] or {}).get("measure_id"), "p1_s0_m0")



if __name__ == "__main__":
    unittest.main()
