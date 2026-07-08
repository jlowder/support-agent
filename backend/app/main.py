"""
Support Agent - LLM-driven customer service agent using LangChain tools.

All CRM lookups and refund decisions are made by the LLM via bound tools.
The agent uses LangGraph for state management and tool orchestration.

This code should be implemented according to the specification file ./spec.md
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict, cast, Annotated
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage, AnyMessage
from langchain_core.tools import tool, ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_config_list, get_executor_for_config, ensure_config
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
        # Use Z suffix for UTC timezone (ISO 8601)
        # datetime.now(timezone.utc).isoformat() returns something like 2026-07-07T21:28:50.780583+00:00
        # We need to replace +00:00 with Z for valid JSON timestamp
        ts = datetime.now(timezone.utc).isoformat()
        clean_timestamp = ts.replace('+00:00', 'Z')
        event = {
            "timestamp": clean_timestamp,
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
    # Support both config formats
    model = config.get("model") or config.get("model_name", "Qwen3-Coder-Next-MLX-6bit")
    base_url = config.get("base_url") or config.get("url", "http://localhost:8080/v1")
    api_key = config.get("api_key", "omlx-om5hh4rsln2h3f8w")
    max_tokens = config.get("max_tokens", 1024)
    temperature = config.get("temperature", 0.7)
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
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


def _ensure_base_messages(messages: List[Any]) -> List[BaseMessage]:
    """Ensure all messages are proper BaseMessage objects."""
    result = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            result.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "human" or role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                result.append(AIMessage(content=content, tool_calls=msg.get("tool_calls")))
            elif role == "tool":
                result.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
    return result


# ---------------------------------------------------------------------------
# LangChain Tools (LLM-driven operations)
# ---------------------------------------------------------------------------

@tool(description="Look up customer profile by customer ID (e.g., usr_001) or email address. Returns customer details including loyalty tier, order count, and order history.", response_format="content")
def get_customer_profile(customer_id: str) -> str:
    """Look up a customer profile from CRM data by ID or email."""
    customer = _find_customer_by_identifier(customer_id)
    
    if not customer:
        return json.dumps({
            "found": False,
            "error": f"Customer not found for identifier: {customer_id}",
            "suggestion": "Please provide a valid customer ID (e.g., usr_001) or email address",
        }, default=str)
    
    profile = {
        "found": True,
        "id": customer.get("id", ""),
        "customer_email": customer.get("email", ""),
        "customer_name": customer.get("name", ""),
        "loyalty_tier": customer.get("loyalty_tier", "standard"),
        "orders_count": len(customer.get("order_history", [])),
        # Include recent order details with items for eligibility evaluation
        "recent_orders": customer.get("order_history", [])[:3],  # Last 3 orders with full item details
        "order_history": customer.get("order_history", []),
    }
    
    return json.dumps(profile, default=str)


@tool(description="Get raw order items for an order. Returns item details including name, price, quantity, is_opened status, and item_type. No eligibility determination - just raw data for the LLM to evaluate.", response_format="content")
def get_order_items(order_id: str) -> str:
    """
    Get raw order item data for an order.
    Returns factual data about items - no eligibility determination.
    The LLM should use policy rules to determine eligibility.
    
    Args:
        order_id: The order ID to get items for
    Returns:
        JSON with order items including: name, price, quantity, is_opened, item_type, category
    """
    order, customer = _find_order_by_id(order_id)
    
    if not order:
        return json.dumps({
            "order_id": order_id,
            "valid": False,
            "error": f"Order {order_id} not found in CRM",
        }, default=str)
    
    # Get current date
    current_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    order_date_str = order.get("order_date", "")
    try:
        order_date = datetime.strptime(order_date_str, "%Y-%m-%d")
        days_since_purchase = (current_dt - order_date).days
    except ValueError:
        days_since_purchase = None
    
    # Build items list with all relevant data for eligibility calculation
    items_data = []
    for item in order.get("items", []):
        item_info = {
            "item_name": item.get("name", ""),
            "item_type": item.get("item_type", ""),
            "is_opened": item.get("is_opened", False),
            "quantity": item.get("quantity", 1),
            "price": item.get("price", 0),
            "category": item.get("category", ""),
            "item_id": item.get("item_id", ""),
        }
        items_data.append(item_info)
    
    return json.dumps({
        "order_id": order_id,
        "valid": True,
        "order_date": order_date_str,
        "days_since_purchase": days_since_purchase,
        "order_status": order.get("status", ""),
        "items": items_data,
    }, default=str)


@tool(description="Process a refund for an order. Returns transaction details or error if refund cannot be processed.", response_format="content")
def process_refund(order_id: str, refund_amount: float, reason: str) -> str:
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
        }, default=str)
    
    # Check if already refunded
    if order.get("status") == "Refunded" or order.get("refund_status") == "Full Refund":
        return json.dumps({
            "success": False,
            "error": f"Order {order_id} has already been refunded",
            "transaction_id": None,
        }, default=str)
    
    # Simulated 503 on odd digit amounts: (amount * 100) is odd
    digit_check = int(refund_amount * 100)
    is_odd = digit_check % 2 != 0
    
    if is_odd:
        # Simulate occasional failure
        return json.dumps({
            "success": False,
            "error": f"Payment service unavailable (amount validation failed)",
            "transaction_id": None,
        }, default=str)
    
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
    }, default=str)


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
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "priority": "high",
    }, default=str)


@tool(description="Select specific items from an order for refund. Takes order_id and item description/pattern to find matching items. Use this when customer specifies which items they want to return.")
def select_items_for_refund(order_id: str, item_selection: str) -> str:
    """
    Select specific items from an order for refund.
    Returns matched items with their details and refund eligibility.
    
    Args:
        order_id: The order ID containing the items
        item_selection: Description of items to select (e.g., 'water bottle', 'all items', 'first item')
    
    Returns:
        JSON string with matched items and their details
    """
    order, customer = _find_order_by_id(order_id)
    if not order:
        return json.dumps({
            "found": False,
            "error": f"Order not found: {order_id}",
        }, default=str)
    
    items = order.get("items", [])
    if not items:
        return json.dumps({
            "found": False,
            "error": "No items found in this order",
        }, default=str)
    
    # Parse item selection
    # Support patterns like:
    # - "water bottle" - match items containing this text
    # - "all items" - return all items
    # - "first item", "second item", etc. - select by index
    # - "item 1", "item 2" - select by 1-indexed position
    
    item_selection_lower = item_selection.lower().strip()
    matched_items = []
    
    # Handle "all items"
    if "all items" in item_selection_lower or item_selection_lower == "all":
        matched_items = items
    # Handle numeric selection (1-indexed)
    elif item_selection_lower.startswith("item "):
        try:
            idx = int(item_selection_lower.split()[1]) - 1  # Convert to 0-indexed
            if 0 <= idx < len(items):
                matched_items = [items[idx]]
            else:
                return json.dumps({
                    "found": False,
                    "error": f"Invalid item number: {item_selection}. Order has {len(items)} items.",
                }, default=str)
        except (ValueError, IndexError):
            # Try to find matching text
            for item in items:
                item_name = item.get("name", "").lower()
                if item_selection_lower in item_name or item_name in item_selection_lower:
                    matched_items.append(item)
                    break
    # Handle "first", "second", etc.
    elif "first" in item_selection_lower:
        matched_items = [items[0]] if items else []
    elif "second" in item_selection_lower:
        matched_items = [items[1]] if len(items) > 1 else []
    elif "third" in item_selection_lower:
        matched_items = [items[2]] if len(items) > 2 else []
    # Handle "item 1", "item 2", etc.
    elif any(f"item {i}" in item_selection_lower for i in range(1, 10)):
        for i in range(1, 10):
            if f"item {i}" in item_selection_lower:
                idx = i - 1
                if 0 <= idx < len(items):
                    matched_items = [items[idx]]
                break
    # Handle text matching
    else:
        for item in items:
            item_name = item.get("name", "").lower()
            if item_selection_lower in item_name or item_name in item_selection_lower:
                matched_items.append(item)
                break
    
    if not matched_items:
        return json.dumps({
            "found": False,
            "error": f"No items matched selection: '{item_selection}'. Available items: {[item.get('name', 'unnamed') for item in items]}",
        }, default=str)
    
    # Build response with matched items
    response = {
        "found": True,
        "order_id": order_id,
        "matched_items": [],
    }
    
    for item in matched_items:
        item_detail = {
            "name": item.get("name", "unnamed"),
            "quantity": item.get("quantity", 1),
            "price": item.get("price", 0),
            "quantity": item.get("quantity", 1),
            "is_opened": item.get("is_opened", False),
            "category": item.get("category", ""),
            "item_type": item.get("item_type", "physical"),
            "eligibility": "eligible",
            "refunded_amount": 0,
        }
        
        # Calculate refund amount
        base_price = item.get("price", 0) * item.get("quantity", 1)
        if item.get("is_opened", False):
            # 15% restocking fee for opened items
            item_detail["restocking_fee"] = base_price * 0.15
            item_detail["refunded_amount"] = base_price * 0.85
        else:
            item_detail["restocking_fee"] = 0
            item_detail["refunded_amount"] = base_price
        
        response["matched_items"].append(item_detail)
    
    response["total_refund"] = sum(item["refunded_amount"] for item in response["matched_items"])
    
    return json.dumps(response, default=str)




# ---------------------------------------------------------------------------
# System Prompt with Tool Instructions
# ---------------------------------------------------------------------------

def _get_system_prompt() -> str:
    """Build the system prompt with agent persona and policy rules."""
    policy_rules = get_policy_rules()
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""You are a firm but empathetic e-commerce customer service assistant.

IMPORTANT: Use the conversation history to remember customer details (ID, email, orders) and previous tool outputs. Do not ask for information already provided.

CURRENT DATE: {current_date}

REFUND POLICY RULES:
{policy_rules}

CRITICAL FLOW (call tools in order, do NOT skip steps):
1. get_customer_profile(customer_id=...) - Use customer_id from extraction or history
2. get_order_items(order_id=...) - Use order_id from extraction or history
3. select_items_for_refund() - If user specified items
4. process_refund() - If eligible
5. Generate response - ONLY at the end

INSTRUCTIONS:
- Extract: customer_id (usr_XXX), order_id (ORD-XXX), items from user message or conversation history.
- Call tools in the order above - DO NOT skip steps.
- Use the CURRENT DATE ({current_date}) for all eligibility calculations.
- Generate response ONLY after all necessary tools have been called and their output processed.
- If you have all information needed from previous turns, proceed directly to the next step in the flow.
- NEVER mention "calling a tool", "let me call", or anything about your internal processing.
- NEVER hallucinate tools that are not in the AVAILABLE TOOLS list.
- If you need information, ask the customer directly.

AVAILABLE TOOLS:
- get_customer_profile(customer_id: str) - Look up customer
- get_order_items(order_id: str) - Get order items
- select_items_for_refund(order_id: str, item_selection: str) - Select items
- process_refund(order_id: str, refund_amount: float, reason: str) - Process refund
- escalate_to_human(reason: str) - Escalate if needed

DO NOT write Python code. Use proper tool calls only.
DO NOT hallucinate tool calls or "thinking" blocks like "Let me call the tool...". Just call the tool.
YOUR FINAL RESPONSE TO THE CUSTOMER SHOULD ONLY BE THE TEXT YOU WANT THEM TO SEE."""


class AgentState(MessagesState):
    response: Optional[str]


# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------

def node_generate_response(state: AgentState, config: RunnableConfig) -> dict:
    """Extract final response from the last AI message."""
    messages = _ensure_base_messages(state.get("messages", []))
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    
    trace_broadcaster.broadcast("agent", "Generating final response", {
        "thread_id": thread_id,
        "message_count": len(messages),
        "message_types": [type(m).__name__ for m in messages],
        "has_tool_messages": any(isinstance(m, ToolMessage) for m in messages),
    })

    # Find the last AIMessage that is NOT a tool call
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            return {"response": msg.content}
    
    # Fallback if no appropriate message found
    return {"response": "I'm sorry, I couldn't process your request."}


# ---------------------------------------------------------------------------
# Tool Execution Setup
# ---------------------------------------------------------------------------

# Create the tools list
tools = [
    get_customer_profile,
    get_order_items,
    process_refund,
    escalate_to_human,
    # Redundant - current date is now injected into system prompt
    # get_current_datetime,
    select_items_for_refund,
]

# Use standard ToolNode
tool_node = ToolNode(tools)

def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    """Wrapper for ToolNode that adds tracing and ensures message objects."""
    # Ensure messages are BaseMessage objects
    messages = _ensure_base_messages(state.get("messages", []))
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    # Extract tool calls from the last message
    tool_calls = []
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            tool_calls = last_msg.tool_calls

    # Trace tool execution START
    if tool_calls:
        trace_broadcaster.broadcast("tools", "Tool execution started", {
            "thread_id": thread_id,
            "tool_call_count": len(tool_calls),
            "tool_calls": [
                {"name": tc.get("name", "unknown"), "args": tc.get("args", {}), "id": tc.get("id", "unknown")}
                for tc in tool_calls
            ],
        })

    # Run tool node
    result = tool_node.invoke({"messages": messages}, config=config)

    # Trace tool execution END - include tool responses
    tool_results = []
    if "messages" in result:
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                tool_results.append({
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                    "name": msg.name if hasattr(msg, 'name') else None,
                })
    
    trace_broadcaster.broadcast("tools", "Tool execution completed", {
        "thread_id": thread_id,
        "result_type": type(result).__name__,
        "tool_results": tool_results,
        "result_messages": [
            {
                "type": type(m).__name__,
                "content": m.content if isinstance(m, BaseMessage) else str(m),
            }
            for m in result.get("messages", [])
        ],
    })

    return result


# ---------------------------------------------------------------------------
# Build Agent Graph
# ---------------------------------------------------------------------------

# Shared checkpointer
memory = MemorySaver()

def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent with bound tools."""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        # Ensure messages are BaseMessage objects (handling restored dicts)
        messages = _ensure_base_messages(state.get("messages", []))
        system_prompt = _get_system_prompt()
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")

        # Build prompt with system message
        langchain_messages = [SystemMessage(content=system_prompt)] + messages

        trace_broadcaster.broadcast("agent", "Agent processing message", {
            "thread_id": thread_id,
            "message_count": len(messages),
            "input_messages": [
                {
                    "role": type(m).__name__.lower().replace("message", ""),
                    "content": m.content if isinstance(m, BaseMessage) else str(m),
                }
                for m in langchain_messages
            ],
        })
        
        response = llm_with_tools.invoke(langchain_messages, config=config)
        
        # Extract tool calls for tracing
        tool_calls_info = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls_info = [
                {
                    "name": tc.get("name", "unknown"),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", "unknown"),
                }
                for tc in response.tool_calls
            ]
        
        trace_broadcaster.broadcast("agent", "LLM response received", {
            "thread_id": thread_id,
            "has_tool_calls": bool(tool_calls_info),
            "content": response.content if hasattr(response, 'content') else str(response),
            "tool_calls": tool_calls_info,
        })
        
        return {"messages": [response]}
    
    def route_after_agent(state: AgentState) -> str:
        messages = state.get("messages", [])
        if messages and getattr(messages[-1], 'tool_calls', None):
            return "tools"
        return "generate_response"
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("generate_response", node_generate_response)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "generate_response": "generate_response"})
    graph.add_edge("tools", "agent")
    graph.add_edge("generate_response", END)
    
    return graph.compile(checkpointer=memory)


# Initialize the graph once
agent_graph = build_agent_graph()


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
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Primary chat endpoint. Runs the LLM-driven agent with bound tools.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Only send the new message, let LangGraph handle history via checkpointer
    input_state = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        # agent_graph is already compiled with checkpointer
        result = agent_graph.invoke(input_state, config=config)
        return ChatResponse(response=result.get("response", "I'm sorry, I couldn't process your request."))
    except Exception as e:
        import traceback
        traceback.print_exc()
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
# Aliases for testing
# ---------------------------------------------------------------------------
def get_user_profile_fn(customer_id: str) -> dict:
    return json.loads(get_customer_profile.run(customer_id))

def check_policy_validity_fn(order_id: str, check_type: str = "full") -> dict:
    return json.loads(get_order_items.run(order_id))

def process_refund_transaction_fn(order_id: str, amount: float) -> dict:
    return json.loads(process_refund.run({"order_id": order_id, "refund_amount": amount, "reason": "Refund requested via test"}))

def escalate_to_human_fn(reason: str) -> dict:
    return json.loads(escalate_to_human.run(reason))


# ---------------------------------------------------------------------------
# Main (for running directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
