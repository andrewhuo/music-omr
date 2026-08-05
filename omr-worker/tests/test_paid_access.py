import unittest
import sys
import base64
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_api_browser_ready import WORKER, _unpack


class _Snapshot:
    def __init__(self, data):
        self._data = deepcopy(data) if isinstance(data, dict) else None
        self.exists = isinstance(data, dict)

    def to_dict(self):
        return deepcopy(self._data)


class _Reference:
    def __init__(self, store, collection, key):
        self.store = store
        self.collection_name = collection
        self.key = key

    def get(self, transaction=None):
        return _Snapshot(self.store.rows.get((self.collection_name, self.key)))


class _Collection:
    def __init__(self, store, name):
        self.store = store
        self.name = name

    def document(self, key):
        return _Reference(self.store, self.name, str(key))


class _Store:
    def __init__(self):
        self.rows = {}

    def collection(self, name):
        return _Collection(self, str(name))


class _Transaction:
    def set(self, ref, data, merge=False):
        ref.store.rows[(ref.collection_name, ref.key)] = deepcopy(data)


def _run_transaction(_client, callback):
    return callback(_Transaction())


def _apple_payload(transaction_id="tx-1", *, expires_days=30, revoked=False, bundle_id=None, product_id=None):
    now = WORKER._utc_now()
    return {
        "productId": product_id or WORKER.APPLE_PRO_PRODUCT_ID,
        "bundleId": bundle_id or WORKER.APPLE_BUNDLE_ID,
        "transactionId": transaction_id,
        "originalTransactionId": "original-1",
        "appTransactionId": "app-wallet-1",
        "environment": "Sandbox",
        "purchaseDate": int(now.timestamp() * 1000),
        "expiresDate": int((now + WORKER.timedelta(days=expires_days)).timestamp() * 1000),
        "revocationDate": int(now.timestamp() * 1000) if revoked else None,
    }


def _pack_payload(transaction_id="pack-1", product_id="pineapple.sheetmusiclabeler.credits.60", revoked=False):
    now = WORKER._utc_now()
    return {
        "productId": product_id,
        "bundleId": WORKER.APPLE_BUNDLE_ID,
        "transactionId": transaction_id,
        "originalTransactionId": transaction_id,
        "appTransactionId": "app-wallet-1",
        "environment": "Sandbox",
        "purchaseDate": int(now.timestamp() * 1000),
        "revocationDate": int(now.timestamp() * 1000) if revoked else None,
    }


class PaidAccessTests(unittest.TestCase):
    def setUp(self):
        self.store = _Store()
        self.patches = (
            patch.object(WORKER, "_friend_store_client", return_value=self.store),
            patch.object(WORKER, "_friend_run_transaction", side_effect=_run_transaction),
        )
        for row in self.patches:
            row.start()

    def tearDown(self):
        for row in reversed(self.patches):
            row.stop()

    def _record(self):
        key = WORKER._apple_wallet_key("app-wallet-1")
        return self.store.rows[(WORKER.PAID_ACCESS_COLLECTION, key)]

    def test_initial_pro_purchase_grants_exactly_400(self):
        result = WORKER._paid_apply_transaction(
            _apple_payload(),
            device_id="device-identifier-1234",
            issue_token=True,
        )
        self.assertTrue(result["active"])
        self.assertEqual(result["credits_remaining"], 400)
        self.assertTrue(result["new_period"])
        self.assertTrue(result["access_token"])
        self.assertEqual(result["plan"], "pro")
        self.assertEqual(result["monthly_credit_capacity"], 400)

    def test_repeated_transaction_does_not_grant_again(self):
        WORKER._paid_apply_transaction(_apple_payload())
        self._record()["credits_remaining"] = 137
        result = WORKER._paid_apply_transaction(_apple_payload())
        self.assertFalse(result["new_period"])
        self.assertEqual(result["credits_remaining"], 137)

    def test_renewal_resets_to_400_without_rollover(self):
        WORKER._paid_apply_transaction(_apple_payload())
        self._record()["credits_remaining"] = 11
        result = WORKER._paid_apply_transaction(_apple_payload("tx-2"))
        self.assertTrue(result["new_period"])
        self.assertEqual(result["credits_remaining"], 400)
        self.assertEqual(self._record()["credits_used"], 0)

    def test_expired_and_revoked_purchase_are_inactive(self):
        expired = WORKER._paid_apply_transaction(_apple_payload(expires_days=-1))
        self.assertFalse(expired["active"])
        self.assertEqual(expired["credits_remaining"], 0)
        revoked = WORKER._paid_apply_transaction(_apple_payload("tx-2", revoked=True))
        self.assertFalse(revoked["active"])
        self.assertEqual(revoked["status"], "revoked")
        self.assertEqual(revoked["credits_remaining"], 0)

    def test_wrong_bundle_or_product_fails_closed(self):
        for payload in (
            _apple_payload(bundle_id="wrong.bundle"),
            _apple_payload(product_id="wrong.product"),
            _apple_payload(product_id="pineapple.sheetmusiclabeler.pro500.monthly"),
        ):
            with self.assertRaises(WORKER.PaidAccessError) as raised:
                WORKER._paid_apply_transaction(payload)
            self.assertEqual(raised.exception.code, "apple_purchase_invalid")

    def test_credit_reservation_spends_or_releases_once(self):
        issued = WORKER._paid_apply_transaction(
            _apple_payload(),
            device_id="device-identifier-1234",
            issue_token=True,
        )
        reserved = WORKER._paid_verify_token(issued["access_token"], reserve=True, job_id="job", system_id="s1")
        self.assertEqual(reserved["credits_remaining"], 399)
        WORKER._paid_finish_reservation(reserved, spent=False)
        self.assertEqual(self._record()["credits_remaining"], 400)
        WORKER._paid_finish_reservation(reserved, spent=False)
        self.assertEqual(self._record()["credits_remaining"], 400)
        reserved = WORKER._paid_verify_token(issued["access_token"], reserve=True, job_id="job", system_id="s2")
        WORKER._paid_finish_reservation(reserved, spent=True)
        self.assertEqual(self._record()["credits_remaining"], 399)
        self.assertEqual(self._record()["credits_used"], 1)

    def test_grace_period_is_active_only_until_its_end(self):
        WORKER._paid_apply_transaction(
            _apple_payload(),
            device_id="device-identifier-1234",
            issue_token=True,
        )
        row = self._record()
        row["status"] = "billing_grace"
        row["expires_at_utc"] = "2026-01-01T00:00:00Z"
        row["grace_expires_at_utc"] = WORKER._to_utc_z(WORKER._utc_now() + WORKER.timedelta(days=2))
        self.assertTrue(WORKER._paid_is_active(row))
        row["grace_expires_at_utc"] = "2026-01-02T00:00:00Z"
        self.assertFalse(WORKER._paid_is_active(row))

    def test_failed_renewal_records_billing_problem(self):
        WORKER._paid_apply_transaction(_apple_payload())
        result = WORKER._paid_update_notification_state(
            _apple_payload(), notification_type="DID_FAIL_TO_RENEW"
        )
        self.assertEqual(result["status"], "billing_retry")
        self.assertEqual(self._record()["status"], "billing_retry")

    def test_disabled_rollout_rejects_purchase_without_touching_store(self):
        WORKER.request = SimpleNamespace(
            get_json=lambda silent=True: {"device_id": "device-identifier-1234", "signed_transaction": "x"},
            headers={},
        )
        with patch.dict(WORKER.os.environ, {"APPLE_IAP_ENABLED": "0"}):
            body, status = _unpack(WORKER.verify_paid_access())
        self.assertEqual(status, 503)
        self.assertEqual((body.get("error") or {}).get("code"), "apple_purchase_not_enabled")
        self.assertEqual(self.store.rows, {})

    def test_privacy_policy_is_public_html(self):
        body, status, headers = WORKER.privacy_policy()
        self.assertEqual(status, 200)
        self.assertIn("Sheet Music Labeler Privacy Policy", body)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")

    def test_invalid_paid_token_stops_ai_before_processing(self):
        WORKER.request = SimpleNamespace(headers={"X-OMR-Paid-Token": "invalid"})
        with patch.object(WORKER, "_resolve_run_id_from_job_id") as resolve:
            body, status = _unpack(WORKER.ai_suggest_job("job"))
        self.assertEqual(status, 403)
        self.assertEqual((body.get("error") or {}).get("code"), "paid_access_required")
        resolve.assert_not_called()

    def test_each_pack_grants_exact_credit_amount_once(self):
        expected = {
            "pineapple.sheetmusiclabeler.credits.60": 60,
            "pineapple.sheetmusiclabeler.credits.140": 140,
            "pineapple.sheetmusiclabeler.credits.240": 240,
        }
        total = 0
        for index, (product, amount) in enumerate(expected.items(), start=1):
            result = WORKER._pack_apply_transaction(
                _pack_payload(f"pack-{index}", product),
                app_transaction_id="app-wallet-1",
                device_id="device-identifier-1234",
            )
            total += amount
            self.assertTrue(result["new_purchase"])
            self.assertEqual(result["credits_granted"], amount)
            self.assertEqual(result["purchased_credits_remaining"], total)

    def test_duplicate_pack_does_not_grant_twice(self):
        first = WORKER._pack_apply_transaction(
            _pack_payload(), app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
        )
        second = WORKER._pack_apply_transaction(
            _pack_payload(), app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
        )
        self.assertEqual(first["purchased_credits_remaining"], 60)
        self.assertFalse(second["new_purchase"])
        self.assertEqual(second["purchased_credits_remaining"], 60)

    def test_pack_refund_creates_debt_and_later_pack_repays_it(self):
        result = WORKER._pack_apply_transaction(
            _pack_payload(), app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
        )
        token = result["access_token"]
        for index in range(50):
            reservation = WORKER._paid_verify_token(token, reserve=True, source="purchased", system_id=str(index))
            WORKER._paid_finish_reservation(reservation, spent=True)
        WORKER._pack_refund_transaction(_pack_payload())
        wallet_key = WORKER._apple_wallet_key("app-wallet-1")
        wallet = self.store.rows[(WORKER.PAID_ACCESS_COLLECTION, wallet_key)]
        self.assertEqual(wallet["purchased_credits_remaining"], 0)
        self.assertEqual(wallet["purchased_credit_debt"], 50)
        added = WORKER._pack_apply_transaction(
            _pack_payload("pack-2", "pineapple.sheetmusiclabeler.credits.140"),
            app_transaction_id="app-wallet-1",
            device_id="device-identifier-1234",
        )
        self.assertEqual(added["debt_repaid"], 50)
        self.assertEqual(added["purchased_credits_remaining"], 90)

    def test_subscription_renewal_does_not_change_purchased_credits(self):
        WORKER._pack_apply_transaction(
            _pack_payload(), app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
        )
        WORKER._paid_apply_transaction(_apple_payload(), app_transaction_id="app-wallet-1")
        WORKER._paid_apply_transaction(_apple_payload("tx-2"), app_transaction_id="app-wallet-1")
        wallet = self.store.rows[(WORKER.PAID_ACCESS_COLLECTION, WORKER._apple_wallet_key("app-wallet-1"))]
        self.assertEqual(wallet["subscription_credits_remaining"], 400)
        self.assertEqual(wallet["purchased_credits_remaining"], 60)

    def test_restore_uses_same_apple_wallet_on_new_device(self):
        original = WORKER._pack_apply_transaction(
            _pack_payload(), app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
        )
        restored = WORKER._paid_restore_wallet(app_transaction_id="app-wallet-1", device_id="different-device-5678")
        self.assertEqual(restored["paid_id"], original["paid_id"])
        self.assertEqual(restored["purchased_credits_remaining"], 60)
        self.assertTrue(restored["access_token"])

    def test_combined_access_uses_earliest_expiring_then_purchased(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        friend_status = {
            "active": True,
            "friend_id": "F",
            "credits_remaining": 5,
            "expires_at_utc": "2026-08-01T00:00:00Z",
        }
        paid_status = {
            "active": True,
            "provider": "paid",
            "pro_active": True,
            "pro_credits_remaining": 5,
            "purchased_credits_remaining": 60,
            "expires_at_utc": "2026-07-20T00:00:00Z",
        }
        with patch.object(WORKER, "_friend_verify_token", return_value=friend_status) as friend_verify, patch.object(
            WORKER, "_paid_verify_token", return_value=paid_status
        ) as paid_verify:
            result = WORKER._ai_access(reserve=True, job_id="job", system_id="s1")
        self.assertEqual(result["provider"], "paid")
        self.assertEqual(paid_verify.call_args.kwargs["source"], "pro")
        self.assertEqual(friend_verify.call_count, 1)

    def test_combined_status_reports_plan_and_monthly_capacity(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        friend_status = {
            "active": True,
            "friend_id": "F",
            "credits_remaining": 500,
            "monthly_credit_capacity": 500,
        }
        paid_status = {
            "active": True,
            "paid_id": "P",
            "pro_active": True,
            "pro_credits_remaining": 400,
            "purchased_credits_remaining": 0,
            "monthly_credit_capacity": 400,
            "plan": "pro",
            "plan_display_name": "Pro",
        }
        with patch.object(WORKER, "_friend_verify_token", return_value=friend_status), patch.object(
            WORKER, "_paid_verify_token", return_value=paid_status
        ):
            result = WORKER._combined_credit_status()
        self.assertEqual(result["paid_plan"], "pro")
        self.assertEqual(result["total_credits"], 900)
        self.assertEqual(result["total_monthly_capacity"], 900)

    def test_combined_status_endpoint_returns_source_checks_without_private_keys(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend"})
        friend_status = {
            "active": True,
            "friend_id": "F",
            "credits_remaining": 500,
            "monthly_credit_capacity": 500,
        }
        with patch.object(WORKER, "_friend_verify_token", return_value=friend_status):
            body, status = _unpack(WORKER.get_combined_credit_status())
        self.assertEqual(status, 200)
        credits = body["credits"]
        self.assertEqual(credits["friend_check"], "verified")
        self.assertEqual(credits["paid_check"], "absent")
        self.assertFalse(credits["partial"])
        for private_key in ("access_token", "device_key", "record_key", "token_hash"):
            self.assertNotIn(private_key, credits)

    def test_combined_status_uses_paid_when_friend_is_temporarily_unavailable(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        paid_status = {
            "paid_id": "P",
            "pro_active": True,
            "pro_credits_remaining": 42,
            "purchased_credits_remaining": 0,
            "monthly_credit_capacity": 400,
            "plan": "pro",
            "plan_display_name": "Pro",
        }
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError(
                "friend_access_unavailable", "temporary", 503, retryable=True
            ),
        ), patch.object(WORKER, "_paid_verify_token", return_value=paid_status):
            result = WORKER._combined_credit_status()
        self.assertEqual(result["friend_check"], "temporarily_unavailable")
        self.assertEqual(result["paid_check"], "verified")
        self.assertTrue(result["partial"])
        self.assertEqual(result["total_credits"], 42)

    def test_combined_status_uses_friend_when_paid_is_temporarily_unavailable(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        friend_status = {
            "active": True,
            "friend_id": "F",
            "credits_remaining": 37,
            "monthly_credit_capacity": 500,
        }
        with patch.object(WORKER, "_friend_verify_token", return_value=friend_status), patch.object(
            WORKER,
            "_paid_verify_token",
            side_effect=WORKER.PaidAccessError(
                "paid_access_unavailable", "temporary", 503, retryable=True
            ),
        ):
            result = WORKER._combined_credit_status()
        self.assertEqual(result["friend_check"], "verified")
        self.assertEqual(result["paid_check"], "temporarily_unavailable")
        self.assertTrue(result["partial"])
        self.assertEqual(result["total_credits"], 37)

    def test_invalid_or_banned_friend_does_not_block_paid_status(self):
        paid_status = {
            "paid_id": "P",
            "pro_active": True,
            "pro_credits_remaining": 20,
            "purchased_credits_remaining": 0,
            "monthly_credit_capacity": 400,
        }
        for code, expected_check in (("ai_access_required", "invalid"), ("friend_access_banned", "banned")):
            with self.subTest(code=code):
                WORKER.request = SimpleNamespace(
                    headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"}
                )
                with patch.object(
                    WORKER,
                    "_friend_verify_token",
                    side_effect=WORKER.FriendAccessError(code, "denied", 403),
                ), patch.object(WORKER, "_paid_verify_token", return_value=paid_status):
                    result = WORKER._combined_credit_status()
                self.assertEqual(result["friend_check"], expected_check)
                self.assertEqual(result["total_credits"], 20)

    def test_combined_status_fails_when_no_wallet_can_be_verified(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError(
                "friend_access_unavailable", "temporary", 503, retryable=True
            ),
        ), patch.object(
            WORKER,
            "_paid_verify_token",
            side_effect=WORKER.PaidAccessError(
                "paid_access_unavailable", "temporary", 503, retryable=True
            ),
        ), self.assertRaises(WORKER.FriendAccessError) as raised:
            WORKER._combined_credit_status()
        self.assertTrue(raised.exception.retryable)

    def test_combined_status_reports_invalid_tokens_without_exposing_them(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError("ai_access_required", "denied", 403),
        ), patch.object(
            WORKER,
            "_paid_verify_token",
            side_effect=WORKER.PaidAccessError("paid_access_required", "denied", 403),
        ):
            result = WORKER._combined_credit_status()
        self.assertEqual(result["friend_check"], "invalid")
        self.assertEqual(result["paid_check"], "invalid")
        self.assertEqual(result["total_credits"], 0)
        for private_key in ("access_token", "device_key", "record_key", "token_hash"):
            self.assertNotIn(private_key, result)

    def test_ai_uses_verified_paid_wallet_when_friend_check_is_temporary(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        paid_status = {
            "pro_active": True,
            "pro_credits_remaining": 9,
            "purchased_credits_remaining": 0,
            "expires_at_utc": "2026-09-01T00:00:00Z",
        }
        reserved = {**paid_status, "provider": "paid", "reservation_id": "reservation"}
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError(
                "friend_access_unavailable", "temporary", 503, retryable=True
            ),
        ), patch.object(WORKER, "_paid_verify_token", side_effect=[paid_status, reserved]) as paid_verify:
            result = WORKER._ai_access(reserve=True, job_id="job", system_id="system")
        self.assertEqual(result["provider"], "paid")
        self.assertEqual(paid_verify.call_count, 2)
        self.assertTrue(paid_verify.call_args.kwargs["reserve"])
        self.assertEqual(paid_verify.call_args.kwargs["source"], "pro")

    def test_ai_uses_verified_friend_wallet_when_paid_check_is_temporary(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        friend_status = {
            "credits_remaining": 9,
            "expires_at_utc": "2026-09-01T00:00:00Z",
        }
        reserved = {**friend_status, "reservation_id": "reservation"}
        with patch.object(WORKER, "_friend_verify_token", side_effect=[friend_status, reserved]) as friend_verify, patch.object(
            WORKER,
            "_paid_verify_token",
            side_effect=WORKER.PaidAccessError(
                "paid_access_unavailable", "temporary", 503, retryable=True
            ),
        ):
            result = WORKER._ai_access(reserve=True, job_id="job", system_id="system")
        self.assertEqual(result["provider"], "friend")
        self.assertEqual(friend_verify.call_count, 2)
        self.assertTrue(friend_verify.call_args.kwargs["reserve"])

    def test_banned_friend_does_not_block_verified_paid_ai(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        paid_status = {
            "pro_active": True,
            "pro_credits_remaining": 9,
            "purchased_credits_remaining": 0,
            "expires_at_utc": "2026-09-01T00:00:00Z",
        }
        reserved = {**paid_status, "provider": "paid", "reservation_id": "reservation"}
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError("friend_access_banned", "banned", 403),
        ), patch.object(WORKER, "_paid_verify_token", side_effect=[paid_status, reserved]):
            result = WORKER._ai_access(reserve=True)
        self.assertEqual(result["provider"], "paid")

    def test_ai_fails_closed_when_no_wallet_verifies(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError(
                "friend_access_unavailable", "temporary", 503, retryable=True
            ),
        ), patch.object(
            WORKER,
            "_paid_verify_token",
            side_effect=WORKER.PaidAccessError(
                "paid_access_unavailable", "temporary", 503, retryable=True
            ),
        ), self.assertRaises(WORKER.FriendAccessError):
            WORKER._ai_access(reserve=True, job_id="job", system_id="system")

    def test_ai_endpoint_stops_before_loading_job_when_no_wallet_verifies(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        with patch.object(
            WORKER,
            "_friend_verify_token",
            side_effect=WORKER.FriendAccessError(
                "friend_access_unavailable", "temporary", 503, retryable=True
            ),
        ), patch.object(
            WORKER,
            "_paid_verify_token",
            side_effect=WORKER.PaidAccessError(
                "paid_access_unavailable", "temporary", 503, retryable=True
            ),
        ), patch.object(WORKER, "_resolve_run_id_from_job_id") as resolve:
            body, status = _unpack(WORKER.ai_suggest_job("job"))
        self.assertEqual(status, 503)
        self.assertEqual((body.get("error") or {}).get("code"), "friend_access_unavailable")
        resolve.assert_not_called()

    def test_verified_empty_wallets_return_exhausted(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer friend", "X-OMR-Paid-Token": "paid"})
        friend_status = {"credits_remaining": 0}
        paid_status = {"pro_active": True, "pro_credits_remaining": 0, "purchased_credits_remaining": 0}
        with patch.object(WORKER, "_friend_verify_token", return_value=friend_status), patch.object(
            WORKER, "_paid_verify_token", return_value=paid_status
        ), self.assertRaises(WORKER.PaidAccessError) as raised:
            WORKER._ai_access(reserve=True)
        self.assertEqual(raised.exception.code, "paid_credits_exhausted")

    def test_friend_reactivation_keeps_identity_and_balance_but_rotates_token(self):
        pepper = base64.urlsafe_b64encode(b"p" * 32).decode("ascii").rstrip("=")
        config = {
            "enabled": True,
            "device_pepper": pepper,
            "default_monthly_credits": 500,
        }
        with patch.object(WORKER, "_friend_config", return_value=config), patch.object(
            WORKER, "_friend_check_activation_rate"
        ), patch.object(WORKER, "_friend_code_matches", return_value=True), patch.object(
            WORKER, "_friend_clear_activation_attempts"
        ):
            first = WORKER._friend_activate_device("device-identifier-1234", "private-code")
            device_key = first["access_token"].split(".", 1)[0]
            record = self.store.rows[(WORKER.FRIEND_ACCESS_COLLECTION, device_key)]
            record["credits_remaining"] = 123
            second = WORKER._friend_activate_device("device-identifier-1234", "private-code")
        self.assertEqual(second["friend_id"], first["friend_id"])
        self.assertNotEqual(second["access_token"], first["access_token"])
        self.assertEqual(
            self.store.rows[(WORKER.FRIEND_ACCESS_COLLECTION, device_key)]["credits_remaining"],
            123,
        )

    def test_friend_verification_returns_current_balance_and_monthly_capacity(self):
        pepper = base64.urlsafe_b64encode(b"p" * 32).decode("ascii").rstrip("=")
        config = {
            "enabled": True,
            "device_pepper": pepper,
            "default_monthly_credits": 500,
        }
        with patch.object(WORKER, "_friend_config", return_value=config), patch.object(
            WORKER, "_friend_check_activation_rate"
        ), patch.object(WORKER, "_friend_code_matches", return_value=True), patch.object(
            WORKER, "_friend_clear_activation_attempts"
        ):
            activated = WORKER._friend_activate_device("device-identifier-1234", "private-code")
            verified = WORKER._friend_verify_token(activated["access_token"])
        self.assertTrue(verified["active"])
        self.assertEqual(verified["credits_remaining"], 500)
        self.assertEqual(verified["monthly_credit_capacity"], 500)

    def test_access_store_logs_safe_library_connection_and_config_stages(self):
        import_error = ModuleNotFoundError("private-token-should-not-appear")
        import_error.name = "google.cloud.firestore"
        with patch.object(WORKER, "firestore", None), patch.object(
            WORKER, "_FIRESTORE_IMPORT_ERROR", import_error
        ), patch.object(WORKER, "_FRIEND_STORE_CLIENT", None), self.assertLogs(
            WORKER.logger, level="WARNING"
        ) as captured:
            self.assertIsNone(self.patches[0].temp_original("friend"))
        library_log = " ".join(captured.output)
        self.assertIn("stage=library_load", library_log)
        self.assertIn("missing_module:google.cloud.firestore", library_log)
        self.assertNotIn("private-token-should-not-appear", library_log)

        class PermissionFailure(RuntimeError):
            def code(self):
                return SimpleNamespace(name="PERMISSION_DENIED")

        class BrokenFirestore:
            @staticmethod
            def Client():
                raise PermissionFailure("private-client-detail-should-not-appear")

        with patch.object(WORKER, "firestore", BrokenFirestore), patch.object(
            WORKER, "_FRIEND_STORE_CLIENT", None
        ), self.assertLogs(WORKER.logger, level="WARNING") as captured:
            self.assertIsNone(self.patches[0].temp_original("paid"))
        client_log = " ".join(captured.output)
        self.assertIn("provider=paid", client_log)
        self.assertIn("stage=client_connection", client_log)
        self.assertNotIn("private-client-detail-should-not-appear", client_log)

        class BrokenDocument:
            def get(self):
                raise PermissionFailure("private-device-id-should-not-appear")

        class BrokenCollection:
            def document(self, _key):
                return BrokenDocument()

        class BrokenStore:
            def collection(self, _name):
                return BrokenCollection()

        with self.assertLogs(WORKER.logger, level="WARNING") as captured, self.assertRaises(
            WORKER.FriendAccessError
        ):
            WORKER._friend_config(BrokenStore())
        config_log = " ".join(captured.output)
        self.assertIn("stage=configuration_read", config_log)
        self.assertIn("service_code=permission_denied", config_log)
        self.assertNotIn("private-device-id-should-not-appear", config_log)

    def test_access_store_logs_token_and_reservation_stages_without_private_data(self):
        config = {"default_monthly_credits": 500}
        token = f"{'a' * 64}.{'private-secret-value-' * 3}"
        for reserve, stage in ((False, "token_check"), (True, "credit_reservation")):
            with self.subTest(stage=stage), patch.object(
                WORKER, "_friend_config", return_value=config
            ), patch.object(
                WORKER, "_friend_run_transaction", side_effect=RuntimeError(token)
            ), self.assertLogs(WORKER.logger, level="WARNING") as captured, self.assertRaises(
                WORKER.FriendAccessError
            ):
                WORKER._friend_verify_token(token, reserve=reserve)
            log = " ".join(captured.output)
            self.assertIn(f"stage={stage}", log)
            self.assertNotIn("private-secret-value", log)

        paid_token = f"{'a' * 64}.{'b' * 64}.{'private-paid-secret-' * 3}"
        with patch.object(
            WORKER, "_friend_run_transaction", side_effect=RuntimeError(paid_token)
        ), self.assertLogs(WORKER.logger, level="WARNING") as captured, self.assertRaises(
            WORKER.PaidAccessError
        ):
            WORKER._paid_verify_token(paid_token)
        paid_log = " ".join(captured.output)
        self.assertIn("provider=paid", paid_log)
        self.assertIn("stage=token_check", paid_log)
        self.assertNotIn("private-paid-secret", paid_log)

    def test_access_store_log_reports_only_safe_missing_name(self):
        error = NameError("name 'friend_credit_helper' is not defined")
        error.name = "friend_credit_helper"
        with self.assertLogs(WORKER.logger, level="WARNING") as captured:
            WORKER._log_access_store_failure("friend", "token_check", error)
        log = " ".join(captured.output)
        self.assertIn("missing_name:friend_credit_helper", log)
        self.assertNotIn("name 'friend_credit_helper' is not defined", log)

        private_error = NameError("private-token-should-not-appear")
        private_error.name = "private-token-should-not-appear"
        with self.assertLogs(WORKER.logger, level="WARNING") as captured:
            WORKER._log_access_store_failure("friend", "token_check", private_error)
        private_log = " ".join(captured.output)
        self.assertIn("service_code=missing_name", private_log)
        self.assertNotIn("private-token-should-not-appear", private_log)

    def test_disabled_pack_balance_does_not_unlock_ai(self):
        WORKER.request = SimpleNamespace(headers={"Authorization": "Bearer bad", "X-OMR-Paid-Token": "paid"})
        paid_status = {
            "paid_id": "P",
            "pro_active": False,
            "pro_credits_remaining": 0,
            "purchased_credits_remaining": 60,
            "purchased_credit_debt": 0,
        }
        with patch.object(
            WORKER, "_friend_verify_token", side_effect=WORKER.FriendAccessError("ai_access_required", "bad", 403)
        ), patch.object(WORKER, "_paid_verify_token", return_value=paid_status):
            result = WORKER._combined_credit_status()
        self.assertEqual(result["purchased_credits"], 0)
        self.assertEqual(result["total_credits"], 0)

        WORKER.request = SimpleNamespace(headers={"X-OMR-Paid-Token": "paid"})
        with patch.object(WORKER, "_paid_verify_token", return_value=paid_status), self.assertRaises(
            WORKER.PaidAccessError
        ) as raised:
            WORKER._ai_access()
        self.assertEqual(raised.exception.code, "paid_credits_exhausted")

    def test_invalid_pack_product_fails_closed(self):
        with self.assertRaises(WORKER.PaidAccessError) as raised:
            WORKER._pack_apply_transaction(
                _pack_payload(product_id="wrong.product"),
                app_transaction_id="app-wallet-1",
                device_id="device-identifier-1234",
            )
        self.assertEqual(raised.exception.code, "apple_purchase_invalid")

    def test_invalid_pack_environment_fails_closed(self):
        payload = _pack_payload()
        payload["environment"] = "Unknown"
        with self.assertRaises(WORKER.PaidAccessError) as raised:
            WORKER._pack_apply_transaction(
                payload, app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
            )
        self.assertEqual(raised.exception.code, "apple_purchase_invalid")

    def test_pro_is_spent_before_permanent_pack_credits(self):
        packed = WORKER._pack_apply_transaction(
            _pack_payload(), app_transaction_id="app-wallet-1", device_id="device-identifier-1234"
        )
        WORKER._paid_apply_transaction(_apple_payload(), app_transaction_id="app-wallet-1")
        reserved = WORKER._paid_verify_token(packed["access_token"], reserve=True)
        self.assertEqual(reserved["source"], "pro")
        wallet = self.store.rows[(WORKER.PAID_ACCESS_COLLECTION, WORKER._apple_wallet_key("app-wallet-1"))]
        self.assertEqual(wallet["subscription_credits_remaining"], 399)
        self.assertEqual(wallet["purchased_credits_remaining"], 60)

    def test_packs_deploy_disabled(self):
        WORKER.request = SimpleNamespace(get_json=lambda silent=True: {}, headers={})
        with patch.dict(WORKER.os.environ, {"APPLE_PACKS_ENABLED": "0"}):
            body, status = _unpack(WORKER.verify_credit_pack())
        self.assertEqual(status, 503)
        self.assertEqual((body.get("error") or {}).get("code"), "apple_packs_not_enabled")


if __name__ == "__main__":
    unittest.main()
