# Technical Specification: AI Customer Support Agent

## 1. System Architecture & Component Diagram

The system uses a decoupled, event-driven architecture to keep frontend UI components highly responsive while allowing the backend agent loop to execute multi-step tool reasoning or streaming audio processing synchronously.

```
+--------------------------------------------------------------------------+
|                               FRONTEND (UI)                              |
|                                                                          |
|  +---------------------------+            +---------------------------+  |
|  |     Chat & Voice UI       |            |   Admin Trace Panel       |  |
|  |  - Microphone Capture     |            |  - Real-time SSE Logs     |  |
|  |  - Text Input / Audio Out |            |  - Tool Call Inspections  |  |
|  +-------------+-------------+            +-------------^-------------+  |
+----------------|----------------------------------------|----------------+
                 | (HTTP / Audio Stream)                  | (SSE Stream)
+----------------|----------------------------------------|----------------+
|                v                                        |                |
|  +---------------------------+            +-------------+-------------+  |
|  |      API / Voice Entry    |            |     Live Event Router     |  |
|  |  - FastAPI Endpoints      |            |  - Broadcaster/Publisher  |  |
|  |  - WebSockets / LiveKit   |            |  - Tailored UI Payloads   |  |
|  +-------------+-------------+            +-------------^-------------+  |
|                |                                        |                |
|                v                                        |                |
|  +------------------------------------------------------+-------------+  |
|  |                   AGENT LAYER (Core Loop Machine)                  |  |
|  |  - State Machine or Tool-Calling Executor Loop                     |  |
|  |  - Policy Engine Guardrails                                        |  |
|  +-----------------------------------+--------------------------------+  |
|                                      |                                   |
|                                      v (Executes / Catches Failures)     |
|                         +--------------------------+                     |
|                         |       TOOL REGISTRY      |                     |
|                         |  - CRM DB Reader/Writer  |                     |
|                         |  - Policy Rule Lookups   |                     |
|                         |  - Refund Processor      |                     |
|                         +------------+-------------+                     |
|                                      |                                   |
|                                      v                                   |
|                         +--------------------------+                     |
|                         |    MOCK INFRASTRUCTURE   |                     |
|                         |  - local_crm.json        |                     |
|                         |  - policy_rules.md       |                     |
|                         +--------------------------+                     |
|                                                                          |
|                            BACKEND SERVICES                              |
+--------------------------------------------------------------------------+

```

---

## 2. Core Functional Requirements

### 2.1 Mock Data Infrastructure

* **Mock CRM Database (`local_crm.json`):** Contains user profiles with the following structure:
  * `id` (string, format: `usr_XXX`)
  * `name` (string)
  * `email` (string)
  * `loyalty_tier` (Bronze, Silver, Gold, or Standard)
  * `order_history` (Array of objects containing `order_id`, `date`, `items` with `name`, `type`, `price`, `opened`, `total`, and `refund_status`)

* **Refund Policy Document (`policy_rules.md`):** A strict Markdown configuration file loaded into the system context. Rules include:
  * **Time Windows:** Standard orders are eligible for refund only within 30 days of purchase. Gold tier members have an extended 45-day window.
  * **Condition Rules:** Opened items are subject to a 15% restocking fee unless the item is reported as defective. Items with "opened: true" flag trigger this rule.
  * **Hard Boundaries:** Digital download items (type: "digital") are strictly non-refundable under any circumstances. Subscriptions are non-refundable.

### 2.2 Agent Backend & Tool Orchestration

* **Core Engine:** Built on Python using FastAPI and LangGraph. The runtime loop executes a state machine with the following nodes:
  * `init` - Initialize state
  * `authenticate` - Extract customer ID or email from message
  * `request_auth_info` - Ask user for customer ID or email if not found
  * `list_orders` - Display customer's order history and ask them to select an order
  * `extract` - Extract order ID and refund amount from customer message
  * `check_policy` - Use LLM to validate refund against policy rules
  * `process` - Execute refund transaction
  * `generate_response` - Generate natural language response

* **State Machine Flow:**
  1. User message arrives via `/chat` endpoint
  2. `init` node initializes state
  3. `authenticate` node extracts customer ID from message or context
  4. If authentication fails → `request_auth_info` node asks for ID/email
  5. If authenticated but no order ID → `list_orders` node shows order history
  6. `extract` node parses order ID and amount from customer message
  7. `check_policy` node validates refund request against policy using LLM
  8. If valid → `process` node executes refund transaction
  9. `generate_response` node generates final response
  10. Response returned to client

* **System Prompting:** Configured with a system persona instruction to act as a firm but empathetic e-commerce representative that must validate state against `policy_rules.md` *before* executing transactions.

* **LLM Integration:** The model is integrated with LangChain and LangGraph. The model name, API key, and URL are provided in `llm_config.json`. The LLM is utilized for:
  * Generating natural language responses
  * Validating policy compliance (using structured JSON output)
  * List orders and ask customer to select

* **Regression Testing:** The model must pass all regression tests defined in `tests/` before being deployed. All backend features must be covered by regression tests.

* **Authentication:** The agent validates the user by:
  * Extracting customer ID from message (pattern: `usr_\d+`)
  * Looking up email in CRM data
  * If neither found, asking user to provide customer ID or email

* **Tool Registry:**
  1. `get_user_profile(customer_id: str)`: Returns matching CRM data or error JSON.
  2. `process_refund_transaction(order_id: str, amount: float)`: Updates order status in JSON storage to "Refunded" and returns a mock transaction ID. May fail on first attempt if amount ends in odd digit (simulated API timeout).
  3. `escalate_to_human(reason: str)`: Escalates issues to human agents with unique escalation ID.

### 2.3 The Voice Pipeline (Bonus Integration)

* **Architecture Choice:** Asynchronous Audio Turn-Based Pipeline.
* **Ingress (STT):** Client side records audio snippets via MediaRecorder API, posting raw audio files to a `/api/voice/ingress` endpoint, which is parsed by OpenAI Whisper.
* **Egress (TTS):** The agent text response is immediately dispatched to ElevenLabs or OpenAI TTS API, returning a playable `.mp3` stream back to client UI.
* **Status:** Currently stubbed with architecture ready for integration. `llm_config.json` provides the foundation for speech service integration.

### 2.4 Admin Dashboard Interface

* **Admin Trace Engine:** A vertical console pane streaming events via Server-Sent Events (SSE) on `/admin/trace` endpoint. It logs:
  * Raw tool inputs
  * Intermediate JSON payloads
  * LLM self-corrections
  * Explicit exception messages
  * State transitions between nodes

* **Trace Event Schema:**
  ```json
  {
    "timestamp": "2026-07-02T12:00:05.123Z",
    "type": "trace",
    "component": "authenticate|extract|list_orders|check_policy|process",
    "message": "Human-readable status message",
    "payload": { ... }
  }
  ```

* **Port Configuration:** The admin trace engine listens on port 8050. The chat API endpoint is `/chat`.

### 2.5 Customer Interaction Panel

* **Customer Interaction Panel:** Traditional chat window supporting text inputs. Voice capture toggle is available for future voice pipeline integration.
* **Port Configuration:** The backend API listens on port 8050. The frontend (HTML/JS) serves static files for user interaction.

---

## 3. Resilience, Error Handling, and Logging Specification

Failure modes must be explicitly designed and integrated into the architecture rather than handled implicitly.

### 3.1 Designed Failure Modes & Self-Correction

* **Simulated API Timeout:** The `process_refund_transaction` tool features a transient deterministic failure:
  * If the refund amount (multiplied by 100) ends in an odd digit, throws a `503 Service Unavailable` network exception on the first attempt
  * On retry, the transaction completes successfully
  * Retry counter is stored in `_RETRY_COUNTS` dict with key format `refund_{order_id}`

* **Agent Correction Behavior:**
  * The LangGraph workflow catches errors during tool execution
  * The `process` node sets `action = "refund_failed"` and logs the error
  * The `generate_response` node generates an apology and explanation for the failure
  * The trace system logs the exception with the error message

### 3.2 Real-time Trace Schema

Every structural state transition inside the agent engine broadcasts a standardized JSON trace model across the SSE pipeline to feed the Admin panel:

```json
{
  "timestamp": "2026-07-02T12:00:05.123Z",
  "type": "trace",
  "component": "authenticate|extract|list_orders|check_policy|process|init",
  "message": "Human-readable status message",
  "payload": {
    "customer_authenticated": true,
    "customer_id": "usr_006",
    "current_order_id": "ORD-2025-03-14-002",
    "action": "request_authentication",
    "route": "list_orders",
    "order_id": "ORD-2025-03-14-002",
    "loyalty_tier": "Gold"
  }
}
```

---
