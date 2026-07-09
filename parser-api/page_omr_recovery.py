#!/usr/bin/env python3
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def _local(tag):
    return str(tag).split("}", 1)[-1] if "}" in str(tag) else str(tag)


def inspect_mxl(path):
    best = {"parts": 0, "measures": 0, "member": None, "error": None}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml"))
                and not name.lower().endswith("container.xml")
            )
            if not members:
                return {**best, "error": "no_xml_members"}
            for member in members:
                try:
                    root = ET.fromstring(archive.read(member))
                except Exception:
                    continue
                parts = sum(1 for child in list(root) if _local(child.tag) == "part")
                measures = sum(1 for node in root.iter() if _local(node.tag) == "measure")
                if (measures, parts) > (best["measures"], best["parts"]):
                    best = {
                        "parts": int(parts),
                        "measures": int(measures),
                        "member": member,
                        "error": None,
                    }
    except Exception as exc:
        return {**best, "error": f"mxl_open_failed:{type(exc).__name__}"}
    if best["parts"] <= 0 or best["measures"] <= 0:
        best["error"] = "parts_or_measures_zero"
    return best


def inspect_page_output(out_dir):
    candidates = sorted(glob.glob(os.path.join(out_dir, "**", "*.mxl"), recursive=True))
    if not candidates:
        return {
            "success": False,
            "reason": "no_mxl_candidates",
            "parts_count": 0,
            "measures_count": 0,
            "mxl_path": None,
        }
    rows = [(path, inspect_mxl(path)) for path in candidates]
    path, best = max(rows, key=lambda row: (row[1]["measures"], row[1]["parts"]))
    return {
        "success": best["parts"] > 0 and best["measures"] > 0,
        "reason": best.get("error") or "ok",
        "parts_count": int(best["parts"]),
        "measures_count": int(best["measures"]),
        "mxl_path": path,
    }


def write_attempt_status(out_dir, status_path, page, attempt, log_path):
    result = inspect_page_output(out_dir)
    previous = {}
    path = Path(status_path)
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    first_failed = bool(previous) and not bool(previous.get("success"))
    payload = {
        "page": int(page),
        "attempts": int(attempt),
        "success": bool(result["success"]),
        "recovered_on_retry": bool(int(attempt) > 1 and first_failed and result["success"]),
        "reason": str(result["reason"]),
        "parts_count": int(result["parts_count"]),
        "measures_count": int(result["measures_count"]),
        "mxl_path": result.get("mxl_path"),
        "log_paths": list(previous.get("log_paths") or []) + [str(log_path)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_report(manifest_path, status_dir, output_path, full_document_success):
    manifest = {}
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    entries = {
        int(row.get("page_number") or int(row.get("page_index", 0)) + 1): row
        for row in (manifest.get("entries") or [])
        if isinstance(row, dict)
    }
    statuses = {}
    for path in sorted(Path(status_dir).glob("page_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            statuses[int(row["page"])] = row
        except Exception:
            continue

    page_numbers = sorted(set(entries) | set(statuses))
    page_results = []
    successful = []
    failed = []
    recovered = []
    for page in page_numbers:
        status = dict(statuses.get(page) or {})
        entry = dict(entries.get(page) or {})
        success = entry.get("status") == "ok"
        reason = "ok" if success else str(entry.get("error") or status.get("reason") or "unknown")
        if success:
            successful.append(page)
        else:
            failed.append(page)
        if success and status.get("recovered_on_retry"):
            recovered.append(page)
        page_results.append(
            {
                "page": page,
                "status": "success" if success else "failed",
                "reason": reason,
                "attempts": int(status.get("attempts") or 0),
                "recovered_on_retry": bool(success and status.get("recovered_on_retry")),
                "parts_count": int(entry.get("parts_count") or status.get("parts_count") or 0),
                "measures_count": int(entry.get("measures_count") or status.get("measures_count") or 0),
                "system_count": len(entry.get("system_starts") or []),
                "log_paths": list(status.get("log_paths") or []),
                "log_artifact": f"artifacts/page_omr_logs/page_{page:04d}.log",
            }
        )

    payload = {
        "version": "page-omr-recovery-v1",
        "full_document_success": bool(full_document_success),
        "total_pages": len(page_numbers),
        "successful_pages": successful,
        "failed_pages": failed,
        "recovered_on_retry_pages": recovered,
        "page_results": page_results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_compact_logs(report_path, output_dir):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in report.get("page_results") or []:
        page = int(row.get("page") or 0)
        failed = row.get("status") != "success"
        line_limit = 400 if failed else 80
        chunks = []
        for raw_path in row.get("log_paths") or []:
            path = Path(raw_path)
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            chunks.append(f"=== {path.name} (last {line_limit} lines) ===")
            chunks.extend(lines[-line_limit:])
        (out / f"page_{page:04d}.log").write_text("\n".join(chunks) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--out-dir", required=True)
    validate.add_argument("--status-path", required=True)
    validate.add_argument("--page", required=True, type=int)
    validate.add_argument("--attempt", required=True, type=int)
    validate.add_argument("--log-path", required=True)

    report = sub.add_parser("report")
    report.add_argument("--manifest", required=True)
    report.add_argument("--status-dir", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--full-document-success", choices=("0", "1"), required=True)
    report.add_argument("--log-output-dir", required=True)

    args = parser.parse_args()
    if args.command == "validate":
        payload = write_attempt_status(
            args.out_dir,
            args.status_path,
            args.page,
            args.attempt,
            args.log_path,
        )
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload["success"] else 1)

    payload = build_report(
        args.manifest,
        args.status_dir,
        args.output,
        args.full_document_success == "1",
    )
    write_compact_logs(args.output, args.log_output_dir)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
