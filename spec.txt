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

* **Mock CRM Database (`local_crm.json`):** Must contain at least 15 detailed user profiles. Each profile requires:
* `customer_id` (string)
* `name` (string)
* `email` (string)
* `loyalty_tier` (Bronze, Silver, Gold)
* `order_history` (Array of objects containing `order_id`, `purchase_date`, `amount`, `item_description`, `status`)


* **Refund Policy Document (`policy_rules.md`):** A strict Markdown configuration file loaded into the system context. Rules must include:
* **Time Windows:** Standard orders are eligible for refund only within 30 days of purchase. Gold tier members have an extended 45-day window.
* **Condition Rules:** Opened items are subject to a 15% restocking fee unless the item is reported as defective.
* **Hard Boundaries:** Subscriptions or digital download items are strictly non-refundable under any circumstances.


### 2.2 Agent Backend & Tool Orchestration

* **Core Engine:** Built on Python utilizing FastAPI. The runtime loop handles dynamic tool execution via raw function-calling loops.
* **System Prompting:** Configured with a system persona instruction to act as a firm but empathetic e-commerce representative that must validate state against `policy_rules.md` *before* executing transactions.
* **LLM Integration:** The model must be integrated with an LLM which may be either local or remote. The information required to integrate with the LLM is provided in the `llm_config.json` file, including the URL, API key, and model name. The LLM is utilized for generating responses and validating policy compliance. Integration with the LLM is handled using LangGraph.
* **Regression Testing:** The model must pass all regression tests defined in `tests/` before being deployed. All backend features must be covered by regression tests.
* **Authentication:** The LLM must authenticate the user before processing any refund requests or providing any other sensitive information. Authentication is simulated by having the user identify themselves using customer id or email.
* **Tool Registry:** The model must be bounded to the following explicit tools:
1. `get_user_profile(customer_id: str)`: Returns matching CRM data.
2. `check_policy_validity(order_id: str, clause: str)`: Cross-references the target item attributes against the markdown rules.
3. `process_refund_transaction(order_id: str, amount: float)`: Updates the item status in the JSON storage engine to "Refunded" and returns a mock transaction transaction ID.
4. `escalate_to_human(reason: str)`: Flagged whenever user intent conflicts with policy but user safety/fraud escalations are triggered.


### 2.3 The Voice Pipeline (Bonus Integration)

* **Architecture Choice:** Asynchronous Audio Turn-Based Pipeline.
* **Ingress (STT):** Client side records audio snippets via MediaRecorder API, posting raw audio files to a `/api/voice/ingress` endpoint, which is parsed by OpenAI Whisper.
* **Egress (TTS):** The agent text response is immediately dispatched to ElevenLabs or OpenAI TTS API, returning a playable `.mp3` stream back to the client UI.

### 2.4 Admin Dashboard Interface

* **Admin Trace Engine:** A vertical console pane streaming events via Server-Sent Events (SSE). It logs raw tool inputs, intermediate JSON payloads, LLM self-corrections, and explicit exception messages.
* **Port Configuration:** The admin trace engine listens on port 8050.

### 2.5 Customer Interaction Panel

* **Customer Interaction Panel:** Traditional chat window supporting text inputs alongside a microphone button for voice capture toggles.
* **Port Configuration:** The admin trace engine listens on port 3050.

---

## 3. Resilience, Error Handling, and Logging Specification

Failure modes must be explicitly designed and integrated into the architecture rather than handled implicitly.

### 3.1 Designed Failure Modes & Self-Correction

* **Simulated API Timeout:** The `process_refund_transaction` tool will feature a transient deterministic failure (e.g., if the refund amount ends in an odd number, throw a `503 Service Unavailable` network exception on the first attempt).
* **Agent Correction Behavior:** The backend loop must catch this specific exception within its execution chain, construct an inner-monologue trace indicating a connection retry, wait 1 second, and attempt the call again.

### 3.2 Real-time Trace Schema

Every structural state transition inside the agent engine must broadcast a standardized JSON trace model across the SSE pipeline to feed the Admin panel:

```json
{
  "timestamp": "2026-07-02T12:00:05.123Z",
  "type": "tool_call | tool_exception | internal_thought | execution_success",
  "component": "RefundAgentLoop",
  "message": "Attempting to call process_refund_transaction for order_id ORD-9912.",
  "payload": {
    "retry_count": 1,
    "raw_error": "Connection reset by peer"
  }
}

```

---

## 4. Verification & Demo Scenario Matrix

These scenarios are mapped directly to the evaluation script requirements for the final delivery video:

| Scenario ID | Target Path | Input Condition | Expected Agent Behavior | Admin Dashboard Visibility |
| --- | --- | --- | --- | --- |
| **SCENARIO-01** | Happy Path | User orders clothing item 10 days ago, item is unopened, requests refund. | Agent calls `get_user_profile`, verifies date is under 30 days, references policy, executes `process_refund_transaction`. | Shows clear tool invocation chain and successful transaction ID generation. |
| **SCENARIO-02** | Holding the Line | User has a Bronze profile, bought an item 50 days ago, insists on a refund. | Agent calls `get_user_profile`, identifies date violates the 30-day limit, politely denies refund, offers store credit alternative. | Displays policy checking step returning false, followed by the text-generation output generation. |
| **SCENARIO-03** | Error Recovery | User triggers a valid refund path that targets the transient failure mock function. | Agent encounters database exception on first execution attempt, logs error, re-invokes tool successfully. | Highlighted warning/exception block flashing on screen, followed immediately by a successful execution block. |

---
