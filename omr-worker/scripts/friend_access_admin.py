#!/usr/bin/env python3
"""Private Cloud Shell controls for Friend Access."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import secrets
from datetime import datetime, timezone

try:
    from google.cloud import firestore
except Exception:  # The Cloud Shell setup command installs this dependency.
    firestore = None


DEVICE_COLLECTION = os.environ.get("FRIEND_ACCESS_COLLECTION", "omr_friend_devices")
CONFIG_COLLECTION = os.environ.get("FRIEND_ACCESS_CONFIG_COLLECTION", "omr_access_config")
CONFIG_DOCUMENT = "friend"
DEFAULT_CREDITS = 500
CODE_ITERATIONS = 210_000
HISTORY_LIMIT = 40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def history(data: dict, action: str, old_value, new_value, reason: str) -> list[dict]:
    rows = [dict(row) for row in (data.get("admin_history") or []) if isinstance(row, dict)]
    rows.append(
        {
            "action": action,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "at_utc": utc_now(),
        }
    )
    return rows[-HISTORY_LIMIT:]


def reset_month(data: dict) -> dict:
    result = dict(data)
    current = month_key()
    if str(result.get("credit_month") or "") != current:
        result["credit_month"] = current
        result["credits_remaining"] = DEFAULT_CREDITS
        result["credits_used"] = 0
        result["reservations"] = {}
    return result


def top_up_balance(balance: int) -> int:
    return max(DEFAULT_CREDITS, int(balance))


def find_device(client, friend_id: str):
    wanted = str(friend_id or "").strip().upper()
    matches = list(client.collection(DEVICE_COLLECTION).where("friend_id", "==", wanted).limit(2).stream())
    if not matches:
        raise SystemExit(f"Friend ID not found: {wanted}")
    if len(matches) > 1:
        raise SystemExit(f"Friend ID is not unique: {wanted}")
    return matches[0].reference, dict(matches[0].to_dict() or {})


def update_device(client, friend_id: str, action: str, reason: str, updater) -> None:
    ref, _ = find_device(client, friend_id)
    transaction = client.transaction()

    @firestore.transactional
    def apply(transaction):
        snap = ref.get(transaction=transaction)
        data = reset_month(dict(snap.to_dict() or {}))
        old_value, new_value = updater(data)
        data["admin_history"] = history(data, action, old_value, new_value, reason)
        data["updated_at_utc"] = utc_now()
        transaction.set(ref, data, merge=False)
        return old_value, new_value

    old_value, new_value = apply(transaction)
    print(f"{str(friend_id).upper()}: {old_value} -> {new_value}")


def command_list(client, _args) -> None:
    rows = []
    for snap in client.collection(DEVICE_COLLECTION).stream():
        data = dict(snap.to_dict() or {})
        rows.append(
            (
                str(data.get("friend_id") or "?"),
                str(data.get("status") or "?"),
                int(data.get("credits_remaining") or 0),
                str(data.get("credit_month") or "?"),
                str(data.get("last_seen_at_utc") or "?"),
            )
        )
    print("FRIEND_ID\tSTATUS\tBALANCE\tMONTH\tLAST_USED_UTC")
    for row in sorted(rows):
        print("\t".join(str(value) for value in row))


def command_top_up(client, args) -> None:
    def updater(data):
        old = int(data.get("credits_remaining") or 0)
        data["credits_remaining"] = top_up_balance(old)
        return old, data["credits_remaining"]

    update_device(client, args.friend_id, "top_up", args.reason, updater)


def command_add(client, args) -> None:
    if args.amount <= 0:
        raise SystemExit("AMOUNT must be greater than zero")

    def updater(data):
        old = int(data.get("credits_remaining") or 0)
        data["credits_remaining"] = old + args.amount
        return old, data["credits_remaining"]

    update_device(client, args.friend_id, "add_credits", args.reason, updater)


def command_status(client, args, status: str) -> None:
    def updater(data):
        old = str(data.get("status") or "active")
        data["status"] = status
        return old, status

    update_device(client, args.friend_id, status, args.reason, updater)


def update_config(client, action: str, reason: str, changes: dict) -> None:
    ref = client.collection(CONFIG_COLLECTION).document(CONFIG_DOCUMENT)
    transaction = client.transaction()

    @firestore.transactional
    def apply(transaction):
        snap = ref.get(transaction=transaction)
        data = dict(snap.to_dict() or {}) if snap.exists else {}
        old_value = {key: data.get(key) for key in changes}
        data.update(changes)
        data.setdefault("default_monthly_credits", DEFAULT_CREDITS)
        data.setdefault("enabled", False)
        data.setdefault("device_pepper", encode(secrets.token_bytes(32)))
        data["admin_history"] = history(data, action, old_value, changes, reason)
        data["updated_at_utc"] = utc_now()
        transaction.set(ref, data, merge=False)

    apply(transaction)
    print(f"Friend Code setting updated: {action}")


def command_set_code(client, args) -> None:
    code = getpass.getpass("New private Friend Code: ").strip()
    confirm = getpass.getpass("Confirm Friend Code: ").strip()
    if code != confirm:
        raise SystemExit("Codes did not match")
    if len(code) < 16:
        raise SystemExit("Use a private code with at least 16 characters")
    salt = secrets.token_bytes(24)
    digest = hashlib.pbkdf2_hmac("sha256", code.encode("utf-8"), salt, CODE_ITERATIONS)
    update_config(
        client,
        "set_code",
        args.reason,
        {
            "code_salt": encode(salt),
            "code_hash": encode(digest),
            "code_iterations": CODE_ITERATIONS,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage private Friend Access")
    parser.add_argument("--project", help="Google Cloud project ID")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    for name in ("top-up", "ban", "unban"):
        command = sub.add_parser(name)
        command.add_argument("friend_id")
        command.add_argument("--reason", default=name)
    add = sub.add_parser("add")
    add.add_argument("friend_id")
    add.add_argument("amount", type=int)
    add.add_argument("--reason", default="temporary friend credits")
    set_code = sub.add_parser("set-code")
    set_code.add_argument("--reason", default="private code rotation")
    for name in ("enable-code", "disable-code"):
        command = sub.add_parser(name)
        command.add_argument("--reason", default=name)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if firestore is None:
        raise SystemExit("Install google-cloud-firestore before running this command")
    client = firestore.Client(project=args.project)
    if args.command == "list":
        command_list(client, args)
    elif args.command == "top-up":
        command_top_up(client, args)
    elif args.command == "add":
        command_add(client, args)
    elif args.command == "ban":
        command_status(client, args, "banned")
    elif args.command == "unban":
        command_status(client, args, "active")
    elif args.command == "set-code":
        command_set_code(client, args)
    elif args.command == "enable-code":
        update_config(client, "enable_code", args.reason, {"enabled": True})
    elif args.command == "disable-code":
        update_config(client, "disable_code", args.reason, {"enabled": False})


if __name__ == "__main__":
    main()
