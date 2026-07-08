# Support Agent

Customer service agent with LLM-driven refund decisions.

## What it does

The agent handles customer refund requests:

- Looks up customer profiles in the CRM
- Lists order history
- Extracts order details from messages
- Checks refund eligibility
- Processes refunds with retry logic
- Escalates to humans when needed
- Streams actions to admins in real time

## Structure

```
support-agent/
├── backend/app/      # FastAPI server with LangGraph agent
├── datagen/          # CRM data generator
├── frontend/         # Simple UI for testing
├── data-viewer/      # CSV/JSON viewer
├── local_crm.json    # Generated customer order data
├── llm_config.json   # LLM settings
└── policy_rules.md   # Refund eligibility rules
```

## Running the agent

```bash
cd backend/app
uvicorn main:app --reload --port 8050
```

## API endpoints

### POST /chat

Send a refund request.

```bash
curl -X POST http://localhost:8050/chat \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "diana.p@email.com", "message": "I want to refund order ORD-000001 for $100"}'
```

### GET /admin/trace

Server-sent events for admin monitoring.

### GET /health

Returns `{"status": "healthy"}`.

## Data generation

Generate CRM data with `datagen/generate-crm.py`:

```bash
python3 datagen/generate-crm.py -n 50
```

The `-n` flag controls customer count and always triggers LLM generation (if configured).

## Tool functions

Importable and testable standalone:

- `get_user_profile_fn(customer_id)` — lookup customer
- `check_policy_validity_fn(order_id, check_type)` — policy check
- `process_refund_transaction_fn(order_id, amount)` — refund with retries
- `escalate_to_human_fn(reason)` — human escalation

See `test_tools.py` for examples.