# Support Agent

LangGraph-based customer service agent with LLM-driven refund decisions.

## Overview

This is a LangGraph-powered agent that handles customer refund requests using natural language. The agent:

- Authenticates customers via CRM lookup
- Lists order history when needed
- Extracts order details from customer messages using LLM
- Checks policy validity using factual tools
- Makes refund decisions via LLM (not hardcoded logic)
- Processes refunds with retry logic
- Escalates to humans when needed
- Streams admin trace events via SSE

## Architecture

### Agent State Machine (LangGraph)

```
init -> authenticate -> (request_auth_info | list_orders | extract)
                                        |
                                    request_auth_info -> generate_response
                                        |
                                    list_orders -> extract -> check_policy -> process -> generate_response
                                                              |
                                                          generate_response
```

### Nodes

1. **init** - Initialize the agent loop
2. **authenticate** - Verify customer identity via CRM
3. **request_auth_info** - Ask for customer ID/email
4. **list_orders** - Display customer's order history
5. **extract** - Extract order_id and amount from customer message using LLM
6. **check_policy** - Check policy validity and make LLM-based decision
7. **process** - Process refund transaction with retry logic
8. **generate_response** - Generate final response (fallback)

### Tool Functions

All tool functions are importable and testable independently:

```python
from backend.app.main import (
    get_user_profile_fn,
    check_policy_validity_fn,
    process_refund_transaction_fn,
    escalate_to_human_fn,
)
```

#### `get_user_profile_fn(customer_id: str) -> dict`
Look up customer profile from CRM. Returns customer data if found, or error JSON.

#### `check_policy_validity_fn(order_id: str, check_type: str = "full") -> dict`
Factual tool that gathers policy-relevant data about an order. Returns:
- order_id, valid, days_since_purchase
- within_30_day_window, within_60_day_window
- items, return_history
- Does NOT make policy decisions (that's the LLM's job)

#### `process_refund_transaction_fn(order_id: str, amount: float) -> dict`
Process a refund transaction. Includes retry logic:
- Fails on first attempt if `(amount * 100)` is odd (simulated 503)
- Retries up to 3 times
- Returns transaction_id on success

#### `escalate_to_human_fn(reason: str) -> dict`
Create an escalation record. Returns:
- escalation_id (format: ESC-XXXXXXXX)
- status: "logged"
- reason, timestamp, priority

## API Endpoints

### POST /chat
Primary chat endpoint that runs the full agent loop.

**Request:**
```json
{
  "customer_id": "diana.p@email.com",
  "message": "I want to refund order ORD-000001 for $100"
}
```

**Response:**
```json
{
  "response": "Your refund of $100.00 for order ORD-000001 has been successfully processed. Transaction ID: refund_abc123..."
}
```

### GET /admin/trace
SSE endpoint for real-time admin trace streaming.

**Event Schema:**
```json
{
  "timestamp": "2026-07-02T12:00:05.123Z",
  "type": "trace",
  "component": "authenticate",
  "message": "Customer authenticated: Diana Prince",
  "payload": { "customer_email": "diana.p@email.com" }
}
```

### POST /api/voice/ingress
Voice ingress stub (pending integration).

**Response:**
```json
{
  "status": "success",
  "message": "Voice ingress is pluggable and pending integration"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-07-06T18:50:00.000Z"
}
```

## Configuration

### LLM Configuration (`llm_config.json`)
```json
{
  "model": "local-model",
  "base_url": "http://localhost:8080/v1",
  "api_key": "not-needed-for-local",
  "max_tokens": 1024,
  "temperature": 0.3
}
```

### Policy Rules (`policy_rules.md`)
Defines refund eligibility:
- 30-day full refund window
- 31-60 day partial refund window
- Digital products non-refundable once accessed
- Damaged/defective products always eligible
- Restocking fees after 14 days for opened items

## Installation

```bash
cd support-agent/backend/app
pip install -r requirements.txt
```

## Running

### 1. Install Dependencies

```bash
cd support-agent/backend/app
pip install -r requirements.txt
```

### 2. Start the Backend API

```bash
cd backend/app
pip install -r requirements.txt
uvicorn main:app --reload --port 8050
```

## Testing Tool Functions

```bash
cd support-agent
python test_tools.py
```

## Key Design Decisions

1. **LLM-Driven Decisions**: All refund decisions are made by the LLM based on factual data from tools, not hardcoded Python logic.

2. **Factual Tools**: Tools like `check_policy_validity_fn` return raw data (days_since_purchase, item conditions, etc.). The LLM interprets this data and makes policy decisions.

3. **Retry Logic**: The `process_refund_transaction_fn` includes simulated 503 errors for odd-digit amounts to test retry behavior.

4. **Admin Tracing**: All agent actions are broadcast via SSE for real-time monitoring.

5. **Stateless Tools**: Tool functions can be called independently without the LLM or FastAPI app, making them easy to test.

## File Structure

```
support-agent/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py              # Main agent code + FastAPI app
│       ├── requirements.txt     # Python dependencies
│       ├── llm_config.json      # LLM configuration
│       ├── policy_rules.md      # Refund policy rules
│       └── local_crm.json       # Customer order data
└── test_tools.py                # Tool function tests
```
