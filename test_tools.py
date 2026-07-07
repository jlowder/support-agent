"""
Quick test for tool functions.
"""
import sys
sys.path.insert(0, "/Users/jlowder/dev/pi-vs-claude-code/support-agent/backend/app")

from main import (
    get_user_profile_fn,
    check_policy_validity_fn,
    process_refund_transaction_fn,
    escalate_to_human_fn,
)


def test_get_user_profile():
    print("Testing get_user_profile_fn...")
    result = get_user_profile_fn("emily.j@example.net")
    print(f"  Result: {result}")
    assert result.get("found"), "Customer should be found"
    assert result.get("customer_name") == "Emily Johnson"
    print("  ✓ PASS")


def test_get_user_profile_not_found():
    print("Testing get_user_profile_fn (not found)...")
    result = get_user_profile_fn("nonexistent@email.com")
    print(f"  Result: {result}")
    assert not result.get("found"), "Customer should not be found"
    assert "error" in result
    print("  ✓ PASS")


def test_check_policy_validity():
    print("Testing check_policy_validity_fn...")
    result = check_policy_validity_fn("ORD-000030", "full")
    print(f"  Result: {result}")
    assert result.get("valid"), "Order should be valid"
    assert result.get("order_id") == "ORD-000030"
    assert result.get("days_since_purchase") is not None
    print("  ✓ PASS")


def test_check_policy_not_found():
    print("Testing check_policy_validity_fn (not found)...")
    result = check_policy_validity_fn("ORD-999999", "full")
    print(f"  Result: {result}")
    assert not result.get("valid"), "Order should not be valid"
    assert "error" in result
    print("  ✓ PASS")


def test_process_refund_success():
    print("Testing process_refund_transaction_fn (success)...")
    # Use an even amount to avoid 503 simulation
    # ORD-000030 is already refunded in mock data, let's use ORD-000050
    result = process_refund_transaction_fn("ORD-000050", 100.00)
    print(f"  Result: {result}")
    assert result.get("success"), "Refund should succeed"
    assert result.get("transaction_id") is not None
    assert result.get("status") == "Refunded"
    print("  ✓ PASS")


def test_escalate_to_human():
    print("Testing escalate_to_human_fn...")
    result = escalate_to_human_fn("Customer requests supervisor")
    print(f"  Result: {result}")
    assert result.get("status") == "logged", "Status should be 'logged'"
    assert result.get("escalation_id") is not None
    assert "ESC-" in result.get("escalation_id", "")
    print("  ✓ PASS")


if __name__ == "__main__":
    test_get_user_profile()
    test_get_user_profile_not_found()
    test_check_policy_validity()
    test_check_policy_not_found()
    test_process_refund_success()
    test_escalate_to_human()
    print("\n✅ All tests passed!")
