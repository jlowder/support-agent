"""
Support Agent - LLM-driven customer service agent using LangChain tools.

All CRM lookups and refund decisions are made by the LLM via bound tools.
The agent uses LangGraph for state management and tool orchestration.
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict, cast
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool, ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

# Pre-build graph at module level for efficiency

from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CRM_PATH = BASE_DIR.parent.parent / "local_crm.json"
LLM_CONFIG_PATH = BASE_DIR.parent.parent / "llm_config.json"
POLICY_RULES_PATH = BASE_DIR / "policy_rules.md"

# ---------------------------------------------------------------------------
# CRM / LLM / Policy loading
# ---------------------------------------------------------------------------

def _load_crm() -> dict:
    """Load CRM data from local_crm.json."""
    with open(CRM_PATH, "r") as f:
        return json.load(f)


def _load_llm_config() -> dict:
    """Load LLM configuration."""
    with open(LLM_CONFIG_PATH, "r") as f:
        return json.load(f)


def _load_policy_rules() -> str:
    """Load policy rules from policy_rules.md."""
    return POLICY_RULES_PATH.read_text()


# In-memory data stores (kept in memory for the lifetime of the process)
_crm_data: Optional[dict] = None
_policy_rules_text: str = ""


def get_crm() -> dict:
    global _crm_data
    if _crm_data is None:
        _crm_data = _load_crm()
    return _crm_data


def get_policy_rules() -> str:
    global _policy_rules_text
    if not _policy_rules_text:
        _policy_rules_text = _load_policy_rules()
    return _policy_rules_text


# ---------------------------------------------------------------------------
# Admin Trace SSE
# ---------------------------------------------------------------------------

class _TraceBroadcaster:
    """Publishes trace events to all connected SSE clients."""

    def __init__(self) -> None:
        self._subscriptions: List[asyncio.Queue] = []  # type: ignore

    def subscribe(self) -> "asyncio.Queue":
        import asyncio
        q: asyncio.Queue = asyncio.Queue()
        self._subscriptions.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        import asyncio
        if q in self._subscriptions:
            self._subscriptions.remove(q)

    def broadcast(self, component: str, message: str, payload: Optional[dict] = None) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "trace",
            "component": component,
            "message": message,
            "payload": payload or {},
        }
        # Create a copy of subscriptions to avoid mutation issues
        for q in list(self._subscriptions.copy()):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def close_all(self) -> None:
        for q in self._subscriptions:
            try:
                q.put_nowait(None)  # sentinel
            except Exception:
                pass
        self._subscriptions.clear()


trace_broadcaster = _TraceBroadcaster()


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    """Get configured LLM client."""
    config = _load_llm_config()
    return ChatOpenAI(
        model=config.get("model", "Qwen3-Coder-Next-MLX-6bit"),
        base_url=config.get("base_url", "http://localhost:8080/v1"),
        api_key=config.get("api_key", "omlx-om5hh4rsln2h3f8w"),
        max_tokens=config.get("max_tokens", 1024),
        temperature=config.get("temperature", 0.7),
    )


# ---------------------------------------------------------------------------
# CRM Data Helpers
# ---------------------------------------------------------------------------

def _find_customer_by_identifier(identifier: str) -> Optional[dict]:
    """Find a customer by ID (usr_XXX) or email address."""
    crm = get_crm()
    customers = crm.get("customers", [])
    for customer in customers:
        if customer.get("id") == identifier or customer.get("email") == identifier:
            return customer
    return None


def _find_order_by_id(order_id: str) -> tuple:
    """Find an order and its associated customer by order_id.
    Returns (order, customer) tuple, or (None, None) if not found."""
    crm = get_crm()
    customers = crm.get("customers", [])
    for customer in customers:
        for order in customer.get("order_history", []):
            if order.get("order_id") == order_id:
                return order, customer
    return None, None


# ---------------------------------------------------------------------------
# LangChain Tools (LLM-driven operations)
# ---------------------------------------------------------------------------

@tool(description="Look up customer profile by customer ID (e.g., usr_001) or email address. Returns customer details including loyalty tier, order count, and order history.", response_format="content_and_artifact")
def get_customer_profile(customer_id: str) -> tuple[str, dict]:
    """Look up a customer profile from CRM data by ID or email."""
    customer = _find_customer_by_identifier(customer_id)
    
    if not customer:
        return json.dumps({
            "found": False,
            "error": f"Customer not found for identifier: {customer_id}",
            "suggestion": "Please provide a valid customer ID (e.g., usr_001) or email address",
        }, default=str), {"found": False, "error": f"Customer not found: {customer_id}"}
    
    profile = {
        "found": True,
        "id": customer.get("id", ""),
        "customer_email": customer.get("email", ""),
        "customer_name": customer.get("name", ""),
        "loyalty_tier": customer.get("loyalty_tier", "standard"),
        "orders_count": len(customer.get("order_history", [])),
        "order_history": customer.get("order_history", []),
    }
    
    return json.dumps(profile, default=str), profile


@tool(description="Check if an order is eligible for refund. Returns order details, days since purchase, and refund eligibility factors.", response_format="content_and_artifact")
def check_order_eligibility(order_id: str) -> tuple[str, dict]:
    """
    Check if an order is eligible for refund.
    Returns factual data about the order including days since purchase, order status, and item details.
    Does NOT make the refund decision - that's for the LLM based on this data.
    """
    order, customer = _find_order_by_id(order_id)
    
    if not order:
        return json.dumps({
            "order_id": order_id,
            "valid": False,
            "error": f"Order {order_id} not found in CRM",
            "days_since_purchase": None,
            "is_eligible": False,
        }, default=str), {"found": False, "error": f"Order {order_id} not found"}
    
    order_date_str = order.get("order_date", "")
    try:
        order_date = datetime.strptime(order_date_str, "%Y-%m-%d")
    except ValueError:
        return json.dumps({
            "order_id": order_id,
            "valid": False,
            "error": f"Could not parse order date: {order_date_str}",
            "days_since_purchase": None,
            "is_eligible": False,
        }, default=str), {"found": False, "error": "Could not parse order date"}
    
    now = datetime.utcnow()
    days_since_purchase = (now - order_date).days
    
    # Compute eligibility windows
    within_30_days = days_since_purchase <= 30
    within_60_days = days_since_purchase <= 60
    
    # Check item conditions
    items_data = []
    for item in order.get("items", []):
        item_info = {
            "item_name": item.get("name", ""),
            "item_type": item.get("item_type", ""),
            "is_opened": item.get("is_opened", False),
            "quantity": item.get("quantity", 1),
            "price": item.get("price", 0),
            "category": item.get("category", ""),
            "is_digital": item.get("item_type") in ("digital", "subscription"),
        }
        items_data.append(item_info)
    
    return json.dumps({
        "order_id": order_id,
        "valid": True,
        "customer_name": customer.get("name", "") if customer else "",
        "customer_email": customer.get("email", "") if customer else "",
        "order_status": order.get("status", ""),
        "order_date": order_date_str,
        "days_since_purchase": days_since_purchase,
        "within_30_day_window": within_30_days,
        "within_60_day_window": within_60_days,
        "total_amount": order.get("total_amount", 0),
        "items": items_data,
        "is_eligible": within_60_days and order.get("status") not in ("Refunded", "Cancelled"),
    }, default=str), {
        "found": True,
        "order_id": order_id,
        "customer_name": customer.get("name", "") if customer else "",
        "customer_email": customer.get("email", "") if customer else "",
        "days_since_purchase": days_since_purchase,
        "within_30_day_window": within_30_days,
        "within_60_day_window": within_60_days,
        "order_status": order.get("status", ""),
        "total_amount": order.get("total_amount", 0),
        "items": items_data,
        "is_eligible": within_60_days and order.get("status") not in ("Refunded", "Cancelled"),
    }


@tool(description="Process a refund for an order. Returns transaction details or error if refund cannot be processed.", response_format="content_and_artifact")
def process_refund(order_id: str, refund_amount: float, reason: str) -> tuple[str, dict]:
    """
    Process a refund transaction for an order.
    Updates order status to 'Refunded' and returns transaction ID.
    Note: This tool performs validation - returns error if order not found or already refunded.
    """
    order, customer = _find_order_by_id(order_id)
    
    if not order:
        return json.dumps({
            "success": False,
            "error": f"Order {order_id} not found",
            "transaction_id": None,
        }, default=str), {"success": False, "error": f"Order {order_id} not found"}
    
    # Check if already refunded
    if order.get("status") == "Refunded" or order.get("refund_status") == "Full Refund":
        return json.dumps({
            "success": False,
            "error": f"Order {order_id} has already been refunded",
            "transaction_id": None,
        }, default=str), {"success": False, "error": "Order already refunded"}
    
    # Simulated 503 on odd digit amounts: (amount * 100) is odd
    digit_check = int(refund_amount * 100)
    is_odd = digit_check % 2 != 0
    
    if is_odd:
        # Simulate occasional failure
        return json.dumps({
            "success": False,
            "error": f"Payment service unavailable (amount validation failed)",
            "transaction_id": None,
        }, default=str), {"success": False, "error": "Payment service temporarily unavailable"}
    
    # Process refund
    order["status"] = "Refunded"
    order["refund_status"] = "Full Refund"
    order["refund_amount"] = refund_amount
    
    transaction_id = f"refund_{uuid.uuid4().hex[:12]}"
    
    return json.dumps({
        "success": True,
        "transaction_id": transaction_id,
        "order_id": order_id,
        "amount": refund_amount,
        "status": "Refunded",
        "reason": reason,
    }, default=str), {
        "success": True,
        "transaction_id": transaction_id,
        "order_id": order_id,
        "amount": refund_amount,
        "status": "Refunded",
    }


@tool(description="Escalate an issue to a human customer service representative. Use when the LLM cannot make a decision, customer requests supervisor, or refund amount exceeds $500.")
def escalate_to_human(reason: str) -> str:
    """
    Create an escalation record for a human representative.
    Returns escalation ID that can be shared with customer.
    """
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    return json.dumps({
        "escalation_id": escalation_id,
        "status": "logged",
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "priority": "high",
    }, default=str)


# ---------------------------------------------------------------------------
# System Prompt with Tool Instructions
# ---------------------------------------------------------------------------

def _get_system_prompt() -> str:
    """Build the system prompt with agent persona and policy rules."""
    policy_rules = get_policy_rules()
    return f"""You are a firm but empathetic e-commerce customer service assistant.

Your capabilities:
- Look up customer profiles by ID or email
- Check order eligibility for refunds
- Process refunds when eligible
- Escalate complex issues to humans

REFUND POLICY (from policy_rules.md):
{policy_rules}

CRITICAL RULES:
1. ALWAYS start by looking up the customer's profile to verify their identity
2. ALWAYS check order eligibility BEFORE processing any refund
3. Only approve refunds that meet the policy requirements
4. For amounts over $500 or when policy is unclear, escalate to human
5. Be empathetic but firm about policy restrictions

DECISION FLOW:
1. If customer identity is unclear: Ask for customer ID or email
2. Look up customer profile using get_customer_profile tool
3. If they want a refund: Ask for order ID
4. Check order eligibility using check_order_eligibility tool
5. If eligible: Process refund using process_refund tool
6. If not eligible or unsure: Explain why and offer escalation

RESPONSE REQUIREMENTS:
- State decisions clearly: approved, denied, or escalated
- Include transaction IDs when refunds are processed
- Explain denial reasons with specific policy references
- Use escalation_id when human intervention is needed"""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    customer_profile: Optional[dict]
    order_info: Optional[dict]
    refund_result: Optional[dict]
    messages: List[BaseMessage]  # conversation history
    needs_human: bool
    error: Optional[str]
    response: Optional[str]  # final response to return


# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------

def node_handle_tool_calls(state: AgentState) -> AgentState:
    """Route to tool execution for LLM tool calls."""
    trace_broadcaster.broadcast("agent", "Processing tool calls", {})
    return state


def node_generate_response(state: AgentState) -> AgentState:
    """Generate final response after tools have been called."""
    trace_broadcaster.broadcast("agent", "Generating final response", {})
    
    # If response is already set in state, use it
    if state.get("response"):
        return state
    
    # Extract conversation from state
    messages = state.get("messages", [])
    
    # Get LLM
    llm = _get_llm()
    
    # Build messages with system prompt
    system_prompt = _get_system_prompt()
    
    # Convert state messages to LangChain format if needed
    langchain_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "system":
                langchain_messages.append(SystemMessage(content=msg.get("content", "")))
            elif msg.get("role") == "human":
                langchain_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                langchain_messages.append(AIMessage(content=msg.get("content", "")))
            elif msg.get("role") == "tool":
                langchain_messages.append(ToolMessage(content=msg.get("content", ""), tool_call_id=msg.get("tool_call_id", "")))
        elif isinstance(msg, BaseMessage):
            langchain_messages.append(msg)
    
    # Add system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in langchain_messages):
        langchain_messages.insert(0, SystemMessage(content=system_prompt))
    
    try:
        # Invoke LLM synchronously (LangGraph's ToolNode handles async context)
        response = llm.invoke(langchain_messages)
        state["response"] = response.content
    except Exception as e:
        state["response"] = (
            "I'm sorry, but I'm experiencing technical difficulties. "
            "A human representative will follow up with you shortly."
        )
        trace_broadcaster.broadcast("error", f"Response generation failed: {str(e)}", {"error": str(e)})
    
    return state


# ---------------------------------------------------------------------------
# Tool Execution Setup
# ---------------------------------------------------------------------------

# Create the tools list
tools = [
    get_customer_profile,
    check_order_eligibility,
    process_refund,
    escalate_to_human,
]

# Create the tool node
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# Agent Router
# ---------------------------------------------------------------------------

def should_call_tools(state: AgentState) -> str:
    """Determine if the LLM wants to call tools based on tool calls in state."""
    messages = state.get("messages", [])
    
    # Check if there are tool calls in the last message
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            # Check for tool_call_id which indicates this was a tool response
            if last_msg.get("role") == "tool":
                return "generate_response"
        elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
    
    return "generate_response"


# ---------------------------------------------------------------------------
# Build Agent Graph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent with bound tools."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("tools", node_handle_tool_calls)
    graph.add_node("generate_response", node_generate_response)
    
    # Set entry point
    graph.set_entry_point("generate_response")
    
    # Add edges
    # From generate_response, check if tools are needed
    graph.add_conditional_edges(
        "generate_response",
        should_call_tools,
        {
            "tools": "tools",
            "generate_response": END,
        },
    )
    
    # Tools node loops back to generate_response
    graph.add_edge("tools", "generate_response")
    
    return graph


# Pre-build graph at module level for efficiency (one-time initialization)
agent_graph = build_agent_graph().compile()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Support Agent", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    customer_id: Optional[str] = None
    message: str
    messages: Optional[List[dict]] = None  # Conversation history from frontend


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Primary chat endpoint. Runs the LLM-driven agent with bound tools.
    """
    # Build initial state with proper defaults
    messages = []
    if request.messages:
        messages = request.messages + [{"role": "human", "content": request.message}]
    else:
        messages = [{"role": "human", "content": request.message}]
    
    state: AgentState = {
        "customer_profile": None,
        "order_info": None,
        "refund_result": None,
        "messages": messages,
        "needs_human": False,
        "error": None,
        "response": None,
    }
    
    # Use pre-built compiled graph
    try:
        result = agent_graph.invoke(state)
        return ChatResponse(response=result.get("response", "I'm sorry, I couldn't process your request."))
    except Exception as e:
        trace_broadcaster.broadcast("error", f"Agent loop error: {str(e)}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/trace")
async def admin_trace():
    """SSE endpoint for real-time admin trace streaming."""
    async def event_generator() -> AsyncGenerator[dict, None]:
        q = trace_broadcaster.subscribe()
        try:
            while True:
                event = await q.get()
                if event is None:
                    break  # sentinel
                yield {
                    "data": json.dumps(event),
                    "event": event.get("type", "trace"),
                }
        finally:
            trace_broadcaster.unsubscribe(q)
    
    return EventSourceResponse(event_generator())


@app.post("/api/voice/ingress")
async def voice_ingress():
    """Voice ingress stub - pending integration."""
    return {
        "status": "success",
        "message": "Voice ingress is pluggable and pending integration",
    }


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize data on startup."""
    global _crm_data, _policy_rules_text
    _crm_data = _load_crm()
    _policy_rules_text = _load_policy_rules()


@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown."""
    trace_broadcaster.close_all()


# ---------------------------------------------------------------------------
# Main (for running directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)