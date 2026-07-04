# Support Agent Backend Implementation Summary

## Overview
This document summarizes the implementation of the AI Customer Support Agent backend as specified in the requirements.

## Files Created

### Configuration Files
1. **llm_config.json** - LLM configuration with exact values:
   - URL: `http://localhost:8080/v1/chat/completions`
   - API Key: `omlx-om5hh4rsln2h3f8w`
   - Model: `gemma-4-31B-it-MLX-8bit`

2. **policy_rules.md** - Refund policy document containing:
   - 30-day standard refund window
   - 45-day Gold tier window
   - 15% restocking fee for opened items (unless defective)
   - Digital/subscriptions non-refundable

### Backend Application
3. **backend/app/main.py** - FastAPI application with:
   - LangGraph agent with tool orchestration
   - `/chat` endpoint for customer interactions
   - `/admin/trace` endpoint for SSE streaming
   - `/api/voice/ingress` endpoint (stub for future voice pipeline)
   - Error handling with retry logic

### Tool Registry
4. **Tool Functions**:
   - `get_user_profile(customer_id)` - Retrieve CRM data
   - `check_policy_validity(order_id, clause)` - Policy validation
   - `process_refund_transaction(order_id, amount)` - Refund processing with retry logic
   - `escalate_to_human(reason)` - Escalation handling

### Test Suite
5. **tests/test_scenarios.py** - Test scenarios:
   - SCENARIO-01: Happy Path (10 days, unopened)
   - SCENARIO-02: Holding the Line (Bronze, 50 days old)
   - SCENARIO-03: Error Recovery (odd-amount refund)

### Documentation
6. **README.md** - Project documentation
7. **IMPLEMENTATION_SUMMARY.md** - This file

## Key Features

### 1. LLM Integration
- Uses LangGraph with StateGraph for agent state management
- LangChain OpenAI integration for tool calling
- MemorySaver for checkpoint persistence

### 2. Tool Orchestration
- All tools use `@tool` decorator from langchain_core.tools
- Tools return JSON strings for LLM processing
- Error handling in `process_refund_transaction` simulates transient failures

### 3. Error Recovery
- Odd-amount refunds trigger simulated 503 errors on first attempt
- Retry logic is built into the tool itself
- Transaction retry counter tracked in `_RETRY_COUNTS` dictionary

### 4. SSE Streaming
- `broadcast.BroadcastChannel` class handles SSE subscriptions
- `TraceEvent` model for standardized event structure
- Admin trace endpoint streams tool calls and exceptions

### 5. Voice Pipeline
- Stub `/api/voice/ingress` endpoint ready for future integration
- Architecture supports async audio turn-based pipeline
- Comments indicate next steps for Whisper STT and TTS integration

## Agent Workflow

```
init → authenticate → extract → check_policy → evaluate → process → generate_response
```

### State Machine Nodes:
1. **init**: Initialize agent state
2. **authenticate**: Extract and validate customer ID/email
3. **extract**: Extract refund request details (order_id, amount)
4. **check_policy**: Validate against policy rules
5. **evaluate**: Determine action (process/deny/request_info)
6. **process**: Execute refund with retry logic
7. **generate_response**: Create customer response

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Main chat endpoint for customer interactions |
| `/admin/trace` | GET | SSE endpoint for admin trace logging |
| `/api/voice/ingress` | POST | Stub for voice pipeline (future implementation) |

## Running the Backend

```bash
cd backend
source ../venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running Tests

```bash
cd tests
source ../venv/bin/activate
pytest test_scenarios.py -v
```

## Requirements

```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.1
langgraph>=0.1.0
langchain-openai>=0.1.0
langchain-core>=0.1.30
pydantic>=2.5.6
openai>=1.12.0
```

## Architecture Highlights

1. **Decoupled Design**: UI and backend communicate via REST/SSE
2. **Tool-Based Agent**: LangGraph with explicit tool registry
3. **State Management**: TypedDict with LangGraph MemorySaver
4. **Error Handling**: Transient failure simulation with retry logic
5. **Event Streaming**: SSE for real-time admin monitoring
6. **Pluggable Architecture**: Voice pipeline ready for future STT/TTS

## Testing Coverage

All three specified scenarios are tested:
- ✅ SCENARIO-01: Happy path with successful refund
- ✅ SCENARIO-02: Policy violation (time window exceeded)
- ✅ SCENARIO-03: Error recovery with retry logic

