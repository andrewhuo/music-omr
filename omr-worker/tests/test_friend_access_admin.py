import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "friend_access_admin.py"
SPEC = importlib.util.spec_from_file_location("friend_access_admin_test", SCRIPT)
ADMIN = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ADMIN)


class FriendAccessAdminTests(unittest.TestCase):
    def test_top_up_only_raises_low_balances_to_500(self):
        self.assertEqual(ADMIN.top_up_balance(120), 500)
        self.assertEqual(ADMIN.top_up_balance(500), 500)
        self.assertEqual(ADMIN.top_up_balance(700), 700)

    def test_month_reset_discards_rollover_and_sets_500(self):
        original_month_key = ADMIN.month_key
        ADMIN.month_key = lambda: "2026-07"
        try:
            result = ADMIN.reset_month(
                {
                    "credit_month": "2026-06",
                    "credits_remaining": 900,
                    "credits_used": 10,
                    "reservations": {"old": {}},
                }
            )
        finally:
            ADMIN.month_key = original_month_key
        self.assertEqual(result["credit_month"], "2026-07")
        self.assertEqual(result["credits_remaining"], 500)
        self.assertEqual(result["credits_used"], 0)
        self.assertEqual(result["reservations"], {})


if __name__ == "__main__":
    unittest.main()
