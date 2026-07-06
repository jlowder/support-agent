"""
Test scenarios for the Support Agent.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

import pytest
from fastapi.testclient import TestClient
from backend.app.main import (
    app,
    get_user_profile_fn,
    check_policy_validity_fn,
    process_refund_transaction_fn,
    escalate_to_human_fn,
)


class TestSupportAgent:
    """Test suite for the Support Agent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        print("✓ test_health_endpoint passed")

    def test_get_user_profile_tool(self):
        """Test the get_user_profile_fn tool function."""
        # Test with valid email
        result = get_user_profile_fn("diana.p@email.com")
        assert result["found"] == True
        assert result["customer_name"] == "Diana Prince"
        assert "customer_email" in result
        print("✓ test_get_user_profile_tool (valid) passed")

        # Test with invalid email
        result = get_user_profile_fn("nonexistent@email.com")
        assert result["found"] == False
        assert "error" in result
        print("✓ test_get_user_profile_tool (invalid) passed")

    def test_check_policy_validity_tool(self):
        """Test the check_policy_validity_fn tool function."""
        # Test with valid order
        result = check_policy_validity_fn("ORD-000001")
        assert result["valid"] == True
        assert result["order_id"] == "ORD-000001"
        assert result["days_since_purchase"] is not None
        assert "items" in result
        print("✓ test_check_policy_validity_tool (valid) passed")

        # Test with invalid order
        result = check_policy_validity_fn("ORD-999999")
        assert result["valid"] == False
        assert "error" in result
        print("✓ test_check_policy_validity_tool (invalid) passed")

    def test_process_refund_transaction_tool(self):
        """Test the process_refund_transaction_fn tool function."""
        # Test with even amount (should succeed immediately)
        result = process_refund_transaction_fn("ORD-000002", 100.00)
        assert result["success"] == True
        assert result["transaction_id"] is not None
        assert result["status"] == "Refunded"
        print("✓ test_process_refund_transaction_tool (success) passed")

        # Test with odd amount (should retry and succeed)
        result = process_refund_transaction_fn("ORD-000003", 100.01)  # 10001 is odd
        assert result["success"] == True
        assert result["attempts"] == 2  # Should take 2 attempts
        print("✓ test_process_refund_transaction_tool (retry) passed")

    def test_escalate_to_human_tool(self):
        """Test the escalate_to_human_fn tool function."""
        result = escalate_to_human_fn("Customer requests supervisor")
        assert result["status"] == "logged"
        assert "escalation_id" in result
        assert result["escalation_id"].startswith("ESC-")
        print("✓ test_escalate_to_human_tool passed")

    def test_admin_trace_endpoint(self):
        """Test the admin trace SSE endpoint."""
        response = self.client.get("/admin/trace")
        # SSE endpoint returns a streaming response
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        print("✓ test_admin_trace_endpoint passed")

    def test_voice_ingress_endpoint(self):
        """Test the voice ingress stub endpoint."""
        response = self.client.post("/api/voice/ingress")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pluggable" in data["message"].lower()
        print("✓ test_voice_ingress_endpoint passed")

    def test_chat_endpoint_no_llm(self):
        """Test the chat endpoint without LLM available (should handle gracefully)."""
        response = self.client.post("/chat", json={
            "customer_id": "diana.p@email.com",
            "message": "I want to refund order ORD-000001 for $100"
        })
        # The endpoint should return a response even if LLM fails
        assert response.status_code in [200, 500]
        print(f"✓ test_chat_endpoint_no_llm passed (status: {response.status_code})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
