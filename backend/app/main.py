"""Main FastAPI application for the support-agent backend."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

# Load configuration
LLM_CONFIG_PATH = Path(__file__).parent.parent.parent / "llm_config.json"
CRM_DATA_PATH = Path(__file__).parent.parent.parent / "local_crm.json"
POLICY_PATH = Path(__file__).parent.parent.parent / "policy_rules.md"

with open(LLM_CONFIG_PATH) as f:
    LLM_CONFIG = json.load(f)

with open(CRM_DATA_PATH) as f:
    CRM_DATA = json.load(f)

with open(POLICY_PATH) as f:
    POLICY_RULES = f.read()


class ChatRequest(BaseModel):
    customer_id: Optional[str] = None
    email: Optional[str] = None
    message: str


class TraceEvent(BaseModel):
    timestamp: str
    type: str
    component: str
    message: str
    payload: Optional[Dict[str, Any]] = None


# Global broadcast channel for SSE
class BroadcastChannel:
    def __init__(self):
        self._clients: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._clients.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            if queue in self._clients:
                self._clients.remove(queue)

    async def broadcast(self, event: TraceEvent) -> None:
        async with self._lock:
            for client in self._clients:
                try:
                    client.put_nowait(event)
                except asyncio.QueueFull:
                    pass


broadcast = BroadcastChannel()

# Retry counter
_RETRY_COUNTS: Dict[str, int] = {}


def load_llm():
    """Load the LLM with configuration."""
    return ChatOpenAI(
        model=LLM_CONFIG["model_name"],
        openai_api_key=LLM_CONFIG["api_key"],
        openai_api_base=LLM_CONFIG["url"],
        temperature=0.7,
    )


# Tool Registry
def get_user_profile_fn(customer_id: str) -> str:
    """Retrieve user profile and order history from CRM."""
    for user in CRM_DATA:
        if user["id"] == customer_id:
            return json.dumps(user, indent=2)
    return json.dumps({"error": f"User {customer_id} not found"})


get_user_profile = tool(get_user_profile_fn)


def process_refund_transaction_fn(order_id: str, amount: float) -> str:
    """Process a refund transaction for an order."""
    key = f"refund_{order_id}"

    if key not in _RETRY_COUNTS:
        _RETRY_COUNTS[key] = 0

    if _RETRY_COUNTS[key] == 0 and int(amount * 100) % 2 == 1:
        _RETRY_COUNTS[key] += 1
        raise Exception(
            "503 Service Unavailable: Connection reset by peer during payment gateway transaction"
        )

    _RETRY_COUNTS.pop(key, None)

    import uuid

    transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"

    for user in CRM_DATA:
        for order in user.get("order_history", []):
            if order.get("order_id") == order_id:
                order["refund_status"] = "Refunded"
                order["refund_transaction_id"] = transaction_id
                order["refund_amount"] = amount
                order["refund_date"] = datetime.now().isoformat()

                with open(CRM_DATA_PATH, "w") as f:
                    json.dump(CRM_DATA, f, indent=2)

                return json.dumps(
                    {
                        "order_id": order_id,
                        "transaction_id": transaction_id,
                        "amount": amount,
                        "status": "success",
                        "refund_date": order["refund_date"],
                    },
                    indent=2,
                )

    return json.dumps({"error": f"Order {order_id} not found"}, indent=2)


process_refund_transaction = tool(process_refund_transaction_fn)


def escalate_to_human_fn(reason: str) -> str:
    """Escalate an issue to a human agent."""
    import uuid

    escalation_id = f"ESC-{uuid.uuid4().hex[:12].upper()}"

    print(f"[ESCALATION] {escalation_id}: {reason}")

    return json.dumps(
        {
            "escalation_id": escalation_id,
            "reason": reason,
            "status": "logged",
            "next_steps": "Human agent will contact customer within 24 hours",
        },
        indent=2,
    )


escalate_to_human = tool(escalate_to_human_fn)


async def broadcast_trace(event: TraceEvent) -> None:
    """Broadcast a trace event to all SSE clients."""
    await broadcast.broadcast(event)


# Build the LangGraph agent


def build_agent():
    """Build the LangGraph agent with tool orchestration."""

    class AgentState(TypedDict):
        messages: List[BaseMessage]
        context: Dict[str, Any]
        current_order_id: Optional[str]
        refund_amount: Optional[float]

    def init_state(state: AgentState) -> AgentState:
        return {
            **state,
            "context": {
                "customer_authenticated": False,
                "loyalty_tier": None,
                "policy_violations": [],
            },
        }

    def authenticate_customer(state: AgentState) -> AgentState:
        messages = state["messages"]
        latest_msg = messages[-1]
        content = latest_msg.content

        customer_id = None

        # Check if customer_id is already provided in context (from chat_endpoint)
        if state["context"].get("customer_id"):
            customer_id = state["context"]["customer_id"]

        if not customer_id and "usr_" in content.lower():
            import re

            matches = re.findall(r"usr_\d+", content.lower())
            if matches:
                customer_id = matches[0].upper()

        if customer_id:
            user_data = json.loads(get_user_profile_fn(customer_id))
            if "error" not in user_data:
                state["context"]["customer_authenticated"] = True
                state["context"]["customer_id"] = customer_id
                state["context"]["loyalty_tier"] = user_data.get(
                    "loyalty_tier", "Standard"
                )
                state["context"]["user_profile"] = user_data

        return state

    def extract_refund_request(state: AgentState) -> AgentState:
        messages = state["messages"]
        # Process ALL messages to find the request, not just the last one
        full_content = " ".join([m.content for m in messages]).lower()

        refund_keywords = ["refund", "return", "back money", "cancel order"]
        is_refund_request = any(kw in full_content for kw in refund_keywords)

        if is_refund_request:
            state["context"]["refund_requested"] = True

            import re

            # Use full_content to find order ID
            order_matches = re.findall(
                r"(ORD-\d{4}-\d{2}-\d{2}-\d+)", full_content.upper()
            )
            if order_matches:
                state["current_order_id"] = order_matches[0]

            # Use full_content to find amount
            amount_matches = re.findall(r"\$?(\d+\.?\d*)", full_content)
            if amount_matches:
                state["refund_amount"] = float(amount_matches[0])

        return state

    def check_policy(state: AgentState) -> AgentState:
        """Use LLM to check refund policy compliance."""
        order_id = state.get("current_order_id")

        if not order_id:
            state["context"]["policy_check_required"] = True
            return state

        # Get user profile for context
        customer_id = state["context"].get("customer_id")
        user_profile = {}
        if customer_id:
            try:
                user_profile = json.loads(get_user_profile_fn(customer_id))
            except:
                pass

        # Get order details for context
        order_details = {}
        if customer_id and "error" not in user_profile:
            for order in user_profile.get("order_history", []):
                if order.get("order_id") == order_id:
                    order_details = order
                    break

        # Build policy check prompt for LLM
        prompt = f"""You are a policy validation assistant. Review the refund request against the policy rules and customer context.

POLICY RULES:
{POLICY_RULES}

CUSTOMER CONTEXT:
- Customer ID: {customer_id}
- Loyalty Tier: {user_profile.get("loyalty_tier", "Unknown")}

ORDER DETAILS:
{json.dumps(order_details, indent=2) if order_details else "Order not found"}

REQUEST:
Customer is requesting a refund for order {order_id}.

Please analyze this refund request against the policy rules and provide:
1. Whether the refund is VALID or INVALID
2. Specific reasons based on policy violations or exceptions
3. Any applicable fees or conditions

Respond in JSON format with: {{"valid": true/false, "reasons": ["list of reasons"], "recommended_action": "approve"|"deny"|"conditional"}}"""

        try:
            llm = load_llm()
            response = llm.invoke(prompt)
            policy_result = json.loads(response.content)

            state["context"]["order_details"] = order_details
            state["context"]["policy_result"] = policy_result
            state["context"]["time_valid"] = policy_result.get("valid", False)
            state["context"]["reasons"] = policy_result.get("reasons", [])
            state["context"]["action"] = policy_result.get(
                "recommended_action", "unknown"
            )

        except Exception as e:
            # Fallback if LLM fails
            state["context"]["policy_result"] = {
                "valid": False,
                "reasons": [f"LLM validation failed: {str(e)}"],
                "recommended_action": "deny",
            }
            state["context"]["time_valid"] = False
            state["context"]["reasons"] = [f"Validation error: {str(e)}"]
            state["context"]["action"] = "deny_refund_policy_violation"

        return state

    def evaluate_policy(state: AgentState) -> AgentState:
        """Use LLM's policy evaluation result to determine action."""
        policy_result = state["context"].get("policy_result", {})

        action = policy_result.get("recommended_action", "unknown")

        if action == "deny":
            state["context"]["action"] = "deny_refund_policy_violation"
            state["context"]["deny_reason"] = ", ".join(
                policy_result.get("reasons", ["Policy violation"])
            )
        elif action == "conditional":
            state["context"]["action"] = "process_refund"
            # Conditional approvals still count as valid, may need additional processing
            state["context"]["conditional_fees"] = policy_result.get("reasons", [])
        else:  # approve or valid
            state["context"]["action"] = "process_refund"

        return state

    def process_refund(state: AgentState) -> AgentState:
        order_id = state.get("current_order_id")
        amount = state.get("refund_amount")

        if not order_id or not amount:
            state["context"]["action"] = "request_missing_info"
            return state

        try:
            result = json.loads(process_refund_transaction_fn(order_id, amount))
            if "error" not in result:
                state["context"]["action"] = "refund_success"
                state["context"]["transaction_id"] = result.get("transaction_id")
            else:
                state["context"]["action"] = "refund_failed"
                state["context"]["failure_reason"] = result.get("error")
        except Exception as e:
            state["context"]["action"] = "refund_failed"
            state["context"]["failure_reason"] = str(e)

        return state

    def generate_response(state: AgentState) -> AgentState:
        action = state["context"].get("action")
        policy_result = state["context"].get("policy_result", {})

        if action == "refund_success":
            response = f"I've successfully processed your refund for order {state.get('current_order_id')}. "
            response += (
                f"A refund of ${state.get('refund_amount'):.2f} has been issued. "
            )
            response += f"Transaction ID: {state['context'].get('transaction_id')}. "
            response += (
                "You should see the credit on your statement within 3-5 business days."
            )
        elif action == "deny_refund_policy_violation":
            # Use the LLM's detailed policy explanation
            reasons = policy_result.get("reasons", [])
            if reasons:
                response = "I've reviewed your request, and unfortunately I cannot process a refund for this order. "
                response += " ".join(reasons)
            else:
                response = "I've reviewed your request, and according to our policy analysis, this refund request does not meet the required criteria. "
                response += "Could you please review our return policy or let me know if you'd like to speak with a human agent?"
        elif action == "request_missing_info":
            response = (
                "I need a bit more information to help you with your refund request. "
            )
            response += "Could you please provide the order ID and the amount you'd like refunded?"
        elif action == "refund_failed":
            response = "I apologize, but I encountered an issue processing your refund transaction. "
            response += (
                f"Error: {state['context'].get('failure_reason', 'Unknown error')}. "
            )
            response += "Please try again or let me know if you'd like to speak with a human agent."
        else:
            response = "I apologize, but I encountered an issue processing your refund request. "
            response += "Could you please try again or let me know if you'd like to speak with a human agent?"

        state["messages"].append(AIMessage(content=response))
        return state

    workflow = StateGraph(AgentState)

    workflow.add_node("init", init_state)
    workflow.add_node("authenticate", authenticate_customer)
    workflow.add_node("extract", extract_refund_request)
    workflow.add_node("check_policy", check_policy)
    workflow.add_node("evaluate", evaluate_policy)
    workflow.add_node("process", process_refund)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("init")

    workflow.add_conditional_edges(
        "init",
        lambda s: (
            "authenticate"
            if not s["context"].get("customer_authenticated")
            else "extract"
        ),
        ["authenticate", "extract"],
    )

    workflow.add_conditional_edges(
        "authenticate",
        lambda s: (
            "extract" if not s["context"].get("current_order_id") else "check_policy"
        ),
        ["extract", "check_policy"],
    )

    workflow.add_conditional_edges(
        "extract",
        lambda s: (
            "check_policy"
            if s["context"].get("current_order_id")
            else "generate_response"
        ),
        ["check_policy", "generate_response"],
    )

    workflow.add_conditional_edges("check_policy", evaluate_policy, ["evaluate"])

    workflow.add_conditional_edges(
        "evaluate",
        lambda s: (
            "process"
            if s["context"].get("action") == "process_refund"
            else "generate_response"
        ),
        ["process", "generate_response"],
    )

    workflow.add_conditional_edges(
        "process",
        lambda s: "generate_response" if s["context"].get("action") else "process",
        ["generate_response"],
    )

    workflow.add_edge("generate_response", END)

    return workflow.compile(checkpointer=MemorySaver())


# Create FastAPI app
app = FastAPI(title="Support Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        messages = [HumanMessage(content=request.message)]

        if request.customer_id:
            messages.append(
                HumanMessage(content=f"My customer ID is {request.customer_id}")
            )
        elif request.email:
            messages.append(HumanMessage(content=f"My email is {request.email}"))

        config = {"configurable": {"thread_id": f"chat_{datetime.now().timestamp()}"}}

        result = await agent.ainvoke(
            {
                "messages": messages,
                "context": {},
                "current_order_id": None,
                "refund_amount": None,
            },
            config=config,
        )

        return {
            "response": result["messages"][-1].content
            if result["messages"]
            else "I couldn't process your request.",
            "messages": [m.dict() for m in result["messages"]],
        }

    except Exception as e:
        await broadcast_trace(
            TraceEvent(
                timestamp=datetime.now().isoformat(),
                type="tool_exception",
                component="ChatEndpoint",
                message=f"Error processing chat: {str(e)}",
                payload={"error": str(e)},
            )
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/trace")
async def admin_trace_endpoint(request: Request):
    queue = await broadcast.subscribe()

    async def event_stream():
        try:
            while True:
                event = await queue.get()
                yield f"data: {event.model_dump_json()}\n\n"
        except asyncio.CancelledError:
            await broadcast.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/voice/ingress")
async def voice_ingress_endpoint():
    """Stub for future voice pipeline integration."""
    await broadcast_trace(
        TraceEvent(
            timestamp=datetime.now().isoformat(),
            type="internal_thought",
            component="VoicePipeline",
            message="Voice ingress endpoint called (pluggable architecture ready)",
            payload={"status": "stub_implementation"},
        )
    )

    return {
        "status": "success",
        "message": "Voice pipeline endpoint ready (STT/TTS integration pending)",
        "architecture": "async_audio_turn_based",
        "next_steps": [
            "Integrate OpenAI Whisper for STT",
            "Integrate ElevenLabs or OpenAI TTS for Egress",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
from langchain_core.messages import BaseMessage
