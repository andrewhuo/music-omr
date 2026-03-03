#!/usr/bin/env python3
import argparse
import json
import sys
import time
from urllib import error as urlerror
from urllib import request as urlrequest


def _request_json(method: str, url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=body, headers=headers, method=method.upper())
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return data


def _poll_job(worker_url: str, job_id: str, timeout_sec: int) -> dict:
    deadline = time.time() + int(timeout_sec)
    last = {}
    while time.time() < deadline:
        last = _request_json("GET", f"{worker_url}/api/omr/jobs/{job_id}")
        status = str(last.get("status") or "").strip().lower()
        if status in ("succeeded", "failed"):
            return last
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for job {job_id}")


def _find_system_value(systems: list[dict], system_id: str) -> str:
    for row in systems:
        if str(row.get("system_id") or "") == system_id:
            return str(row.get("current_value") or row.get("value") or "")
    raise KeyError(f"system_id not found: {system_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test fast relabel flow without frontend.")
    ap.add_argument("--worker-url", required=True, help="Base worker URL, e.g. https://omr-worker-xxx.run.app")
    ap.add_argument("--pdf-gcs-uri", required=True, help="Input PDF GCS URI")
    ap.add_argument("--timeout-sec", type=int, default=600, help="Overall job timeout (default: 600)")
    args = ap.parse_args()

    worker_url = str(args.worker_url).rstrip("/")
    print(f"SMOKE worker_url={worker_url}")
    print(f"SMOKE pdf_gcs_uri={args.pdf_gcs_uri}")

    create = _request_json(
        "POST",
        f"{worker_url}/api/omr/jobs",
        {"pdf_gcs_uri": args.pdf_gcs_uri},
    )
    job_id = str(create.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"missing job_id in create response: {create}")
    print(f"SMOKE job_id={job_id}")

    job = _poll_job(worker_url, job_id, args.timeout_sec)
    if str(job.get("status")) != "succeeded":
        raise RuntimeError(f"job did not succeed: {job}")
    run_id = int(job.get("run_id"))
    print(f"SMOKE run_id={run_id}")

    state_before = _request_json("GET", f"{worker_url}/api/omr/jobs/{job_id}/state")
    state_before_v = str(state_before.get("state_version") or "")
    editable = state_before.get("editable_state") or {}
    systems_before = editable.get("systems") or []
    if not isinstance(systems_before, list) or not systems_before:
        raise RuntimeError(f"no systems in state: {state_before}")

    first = systems_before[0]
    system_id = str(first.get("system_id") or "")
    old_val = int(str(first.get("current_value") or first.get("value") or "0"))
    new_val = old_val + 5
    print(f"SMOKE edit system_id={system_id} {old_val}->{new_val}")

    relabel = _request_json(
        "POST",
        f"{worker_url}/api/omr/jobs/{job_id}/relabel",
        {"edits": [{"type": "set_system_start", "system_id": system_id, "value": new_val}]},
    )
    relabel_info = relabel.get("relabel") or {}
    corrected_uri = ((relabel.get("artifacts") or {}).get("audiveris_out_corrected_pdf") or "").strip()
    if not corrected_uri.startswith("gs://"):
        raise RuntimeError(f"missing corrected PDF URI in relabel response: {relabel}")
    if relabel_info.get("state_version_before") == relabel_info.get("state_version_after"):
        raise RuntimeError(f"state version did not change after relabel: {relabel_info}")

    state_after = _request_json("GET", f"{worker_url}/api/omr/jobs/{job_id}/state")
    state_after_v = str(state_after.get("state_version") or "")
    systems_after = ((state_after.get("editable_state") or {}).get("systems") or [])
    if not isinstance(systems_after, list):
        raise RuntimeError(f"invalid state after relabel: {state_after}")
    changed = _find_system_value(systems_after, system_id)
    if int(changed) != int(new_val):
        raise RuntimeError(
            f"relabel not applied: system_id={system_id} expected={new_val} got={changed}"
        )
    if state_before_v == state_after_v:
        raise RuntimeError("state endpoint version did not update")

    print(f"SMOKE corrected_pdf={corrected_uri}")
    print(
        "SMOKE ok "
        f"state_version_before={state_before_v} "
        f"state_version_after={state_after_v} "
        f"duration_ms={relabel_info.get('duration_ms')}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP_ERROR status={exc.code} body={body}", file=sys.stderr)
        raise
