# Support Agent - AI Customer Support System

A FastAPI-based AI customer support agent that processes e-commerce refund requests using LangGraph for tool orchestration.

## Architecture

```
+--------------------------------------------------------------------------+
|                               FRONTEND (UI)                              |
|  - Chat Interface | Voice Capture | Admin Trace Panel                    |
+---------------------------+-----------------------------+---------------+
                            |                             |
+---------------------------v-----------------------------v---------------+
|                    AGENT LAYER (LangGraph)                             |
|  - State Machine | Policy Engine | Tool Orchestration                   |
+---------------------------+-----------------------------+---------------+
                            |                             |
+---------------------------v-----------------------------v---------------+
|                    TOOL REGISTRY & DATA LAYER                          |
|  - CRM DB Reader | Policy Rules | Refund Processor                     |
+--------------------------------------------------------------------------+
```

## Features

- **LLM-Powered Agent**: Uses LangGraph for dynamic tool orchestration
- **Policy Validation**: Enforces refund policies (time windows, item conditions)
- **Error Recovery**: Implements retry logic for transient failures
- **SSE Streaming**: Real-time admin trace logging
- **Voice Pipeline**: Pluggable architecture for STT/TTS integration

## Requirements

- Python 3.10+
- FastAPI
- LangGraph
- LangChain OpenAI
- OpenAI API (for LLM integration)

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Configuration

### LLM Configuration (`llm_config.json`)
```json
{
  "url": "http://localhost:8080/v1/chat/completions",
  "api_key": "omlx-om5hh4rsln2h3f8w",
  "model_name": "gemma-4-31B-it-MLX-8bit"
}
```

## Running the Server

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### `GET /health`
Health check endpoint.

### `POST /chat`
Main chat endpoint for customer interactions.

**Request:**
```json
{
  "customer_id": "usr_001",
  "email": "customer@example.com",
  "message": "I want to request a refund for order ORD-2025-02-25-001"
}
```

**Response:**
```json
{
  "response": "I've successfully processed your refund...",
  "messages": [...]
}
```

### `GET /admin/trace`
SSE endpoint for admin trace logging.

### `POST /api/voice/ingress`
Stub endpoint for future voice pipeline integration.

## Tool Registry

### `get_user_profile(customer_id)`
Retrieve user profile and order history from CRM.

### `check_policy_validity(order_id, clause)`
Check if a refund request matches policy rules.

### `process_refund_transaction(order_id, amount)`
Process a refund transaction for an order.

### `escalate_to_human(reason)`
Escalate an issue to a human agent.

## Test Scenarios

### SCENARIO-01: Happy Path
- Customer within 30-day window (or 45 for Gold tier)
- Unopened item
- Expected: Successful refund

### SCENARIO-02: Holding the Line
- Customer outside refund window
- Expected: Refusal with policy explanation

### SCENARIO-03: Error Recovery
- Odd-amount refund triggers simulated transient failure
- Expected: Automatic retry and successful completion

## Running Tests

```bash
cd tests
pytest test_scenarios.py -v
```

## Project Structure

```
support-agent/
├── llm_config.json          # LLM configuration
├── policy_rules.md          # Refund policy document
├── local_crm.json           # Mock CRM database
├── spec.md                  # Technical specification
├── README.md                # This file
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI application
│   ├── requirements.txt
│   └── tests/
│       ├── __init__.py
│       └── test_scenarios.py
└── docs/
```

## License

MIT License
