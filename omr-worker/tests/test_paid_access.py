import unittest
import sys
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
        "productId": product_id or WORKER.APPLE_MONTHLY_PRODUCT_ID,
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

    def test_initial_purchase_grants_exactly_200(self):
        result = WORKER._paid_apply_transaction(
            _apple_payload(),
            device_id="device-identifier-1234",
            issue_token=True,
        )
        self.assertTrue(result["active"])
        self.assertEqual(result["credits_remaining"], 200)
        self.assertTrue(result["new_period"])
        self.assertTrue(result["access_token"])

    def test_repeated_transaction_does_not_grant_again(self):
        WORKER._paid_apply_transaction(_apple_payload())
        self._record()["credits_remaining"] = 137
        result = WORKER._paid_apply_transaction(_apple_payload())
        self.assertFalse(result["new_period"])
        self.assertEqual(result["credits_remaining"], 137)

    def test_renewal_resets_to_200_without_rollover(self):
        WORKER._paid_apply_transaction(_apple_payload())
        self._record()["credits_remaining"] = 11
        result = WORKER._paid_apply_transaction(_apple_payload("tx-2"))
        self.assertTrue(result["new_period"])
        self.assertEqual(result["credits_remaining"], 200)
        self.assertEqual(self._record()["credits_used"], 0)

    def test_expired_and_revoked_purchase_are_inactive(self):
        with self.assertRaises(WORKER.PaidAccessError) as expired:
            WORKER._paid_apply_transaction(_apple_payload(expires_days=-1))
        self.assertEqual(expired.exception.code, "paid_subscription_inactive")
        with self.assertRaises(WORKER.PaidAccessError) as revoked:
            WORKER._paid_apply_transaction(_apple_payload("tx-2", revoked=True))
        self.assertEqual(revoked.exception.code, "paid_subscription_inactive")

    def test_wrong_bundle_or_product_fails_closed(self):
        for payload in (
            _apple_payload(bundle_id="wrong.bundle"),
            _apple_payload(product_id="wrong.product"),
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
        self.assertEqual(reserved["credits_remaining"], 199)
        WORKER._paid_finish_reservation(reserved, spent=False)
        self.assertEqual(self._record()["credits_remaining"], 200)
        WORKER._paid_finish_reservation(reserved, spent=False)
        self.assertEqual(self._record()["credits_remaining"], 200)
        reserved = WORKER._paid_verify_token(issued["access_token"], reserve=True, job_id="job", system_id="s2")
        WORKER._paid_finish_reservation(reserved, spent=True)
        self.assertEqual(self._record()["credits_remaining"], 199)
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
        self.assertEqual(wallet["subscription_credits_remaining"], 200)
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

    def test_bad_friend_token_does_not_block_valid_paid_wallet(self):
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
        self.assertEqual(result["purchased_credits"], 60)
        self.assertEqual(result["total_credits"], 60)

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
        self.assertEqual(wallet["subscription_credits_remaining"], 199)
        self.assertEqual(wallet["purchased_credits_remaining"], 60)

    def test_packs_deploy_disabled(self):
        WORKER.request = SimpleNamespace(get_json=lambda silent=True: {}, headers={})
        with patch.dict(WORKER.os.environ, {"APPLE_PACKS_ENABLED": "0"}):
            body, status = _unpack(WORKER.verify_credit_pack())
        self.assertEqual(status, 503)
        self.assertEqual((body.get("error") or {}).get("code"), "apple_packs_not_enabled")


if __name__ == "__main__":
    unittest.main()
