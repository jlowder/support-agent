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
from langgraph.prebuilt import ToolNode, create_react_agent
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
        "order_history": customer.get("order_history", []),
    }
    
    return json.dumps(profile, default=str)


@tool(description="Check if an order is eligible for refund. Returns order details, days since purchase, and refund eligibility factors. Include current_date parameter for accurate calculations.", response_format="content")
def check_order_eligibility(order_id: str, current_date: str = None) -> str:
    """
    Check if an order is eligible for refund.
    Returns factual data about the order including days since purchase, order status, and item details.
    Does NOT make the refund decision - that's for the LLM based on this data.
    
    Args:
        order_id: The order ID to check
        current_date: Optional current date in YYYY-MM-DD format. If not provided, uses system date.
    """
    order, customer = _find_order_by_id(order_id)
    
    if not order:
        return json.dumps({
            "order_id": order_id,
            "valid": False,
            "error": f"Order {order_id} not found in CRM",
            "days_since_purchase": None,
            "is_eligible": False,
        }, default=str)
    
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
        }, default=str)
    
    # Use provided current_date or system date
    if current_date:
        try:
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        except ValueError:
            current_dt = datetime.utcnow()
    else:
        current_dt = datetime.utcnow()
    
    days_since_purchase = (current_dt - order_date).days
    
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "priority": "high",
    }, default=str)


@tool(description="Get the current date and time. Use this to calculate days since purchase for order eligibility.")
def get_current_datetime() -> str:
    """
    Get the current date and time for calculating order eligibility.
    Returns ISO formatted datetime string.
    """
    return json.dumps({
        "current_datetime": datetime.utcnow().isoformat() + "Z",
        "current_date": datetime.utcnow().strftime("%Y-%m-%d"),
    }, default=str)


# ---------------------------------------------------------------------------
# System Prompt with Tool Instructions
# ---------------------------------------------------------------------------

def _get_system_prompt() -> str:
    """Build the system prompt with agent persona and policy rules."""
    policy_rules = get_policy_rules()
    # Get current date for accurate calculations
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    return f"""You are a firm but empathetic e-commerce customer service assistant.

CURRENT DATE: {current_date}

Your capabilities:
- Look up customer profiles by ID or email
- Check order eligibility for refunds
- Process refunds when eligible
- Escalate complex issues to humans
- Get current date/time for accurate calculations

REFUND POLICY (from policy_rules.md):
{policy_rules}

CRITICAL RULES:
1. ALWAYS start by looking up the customer's profile to verify their identity
2. ALWAYS get current date using get_current_datetime tool
3. ALWAYS check order eligibility using check_order_eligibility tool (include current_date)
4. Only approve refunds that meet the policy requirements
5. For amounts over $500 or when policy is unclear, escalate to human
6. Be empathetic but firm about policy restrictions

ORDER REFUND FLOW:
1. Verify customer identity with get_customer_profile
2. Get current date with get_current_datetime
3. If order ID provided: Check eligibility with check_order_eligibility (include current_date)
4. If order not provided: Ask customer for order ID or list their orders
5. Check order eligibility and show items to customer
6. Ask customer which items they want to refund
7. Calculate refund amounts (15% restocking fee for opened items)
8. If eligible: Process refund using process_refund tool
9. If not eligible or unsure: Explain why and offer escalation

ITEM-LEVEL REFUNDS:
- Customers can select specific items to refund
- Unopened items: Full refund
- Opened items: 15% restocking fee applied
- Digital/subscription items: Non-refundable
- Calculate partial refunds based on selected items

RESPONSE REQUIREMENTS:
- State decisions clearly: approved, denied, or escalated
- Include transaction IDs when refunds are processed
- Explain denial reasons with specific policy references
- Use escalation_id when human intervention is needed
- For partial refunds: Show itemized breakdown with amounts"""


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
    messages = state.get("messages", [])
    
    # Get the last assistant message with tool calls
    tool_calls_to_execute = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_to_execute = msg.tool_calls
            break
    
    # Trace: tools being invoked
    trace_broadcaster.broadcast("agent", "Executing tool calls", {
        "tool_call_count": len(tool_calls_to_execute),
        "tool_calls": [
            {"name": tc.get("name", "unknown"), "args": tc.get("args", {}), "id": tc.get("id", "unknown")}
            for tc in tool_calls_to_execute
        ],
    })
    
    return state


def node_generate_response(state: AgentState) -> AgentState:
    """Generate final response after tools have been called."""
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
    
    # Trace: generating response with conversation context
    trace_broadcaster.broadcast("agent", "Generating final response", {
        "message_count": len(langchain_messages),
        "message_types": [msg.__class__.__name__ for msg in langchain_messages],
        "has_tool_messages": any(isinstance(m, ToolMessage) for m in langchain_messages),
    })
    
    try:
        # Invoke LLM synchronously (LangGraph's ToolNode handles async context)
        response = llm.invoke(langchain_messages)
        state["response"] = response.content
        
        # Trace: response generated
        trace_broadcaster.broadcast("agent", "Response generated", {
            "response_length": len(response.content) if response.content else 0,
            "response_preview": response.content[:100] if response.content else None,
        })
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
    get_current_datetime,
]

# Create the tool node with tracing
class TracingToolNode(ToolNode):
    """ToolNode that broadcasts tool execution events."""
    
    def _apply(self, state, input):
        # Get tool calls from input
        tool_calls = []
        if hasattr(input, 'tool_calls') and input.tool_calls:
            tool_calls = input.tool_calls
        elif isinstance(input, dict) and 'tool_calls' in input:
            tool_calls = input['tool_calls']
        
        # Trace tool execution
        if tool_calls:
            trace_broadcaster.broadcast("tools", "Tool execution started", {
                "tool_call_count": len(tool_calls),
                "tool_calls": [
                    {"name": tc.get("name", "unknown"), "args": tc.get("args", {}), "id": tc.get("id", "unknown")}
                    for tc in tool_calls
                ],
            })
        
        # Execute tools
        result = super()._apply(state, input)
        
        # Trace tool completion
        if tool_calls:
            trace_broadcaster.broadcast("tools", "Tool execution completed", {
                "tool_call_count": len(tool_calls),
                "results_count": len(result.get("messages", [])),
            })
        
        return result

tool_node = TracingToolNode(tools)


# ---------------------------------------------------------------------------
# Agent Router
# ---------------------------------------------------------------------------

def should_call_tools(state: AgentState) -> str:
    """Determine if the LLM wants to call tools based on tool calls in state."""
    messages = state.get("messages", [])
    
    # Check if there are tool calls in the last message
    if messages:
        last_msg = messages[-1]
        # Check for ToolMessage (tool results ready for LLM)
        if isinstance(last_msg, ToolMessage) or hasattr(last_msg, 'tool_call_id'):
            return "agent"
        elif isinstance(last_msg, dict):
            # Check for tool_call_id which indicates this was a tool response
            if last_msg.get("role") == "tool":
                return "agent"
        elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
    
    return "generate_response"


# ---------------------------------------------------------------------------
# Build Agent Graph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent with bound tools using create_react_agent."""
    # Get LLM and bind tools
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    # Create a simple graph that handles tool calling
    # We'll use a manual approach since create_react_agent has different state management
    def agent_node(state: AgentState) -> AgentState:
        """Run the agent and return updated state."""
        # Get messages from state
        messages = state.get("messages", [])
        
        # Convert dict messages to LangChain messages
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
        
        # Add current date system prompt if not already present
        # This ensures LLM has accurate date for eligibility calculations
        system_prompt = _get_system_prompt()
        if not any(isinstance(m, SystemMessage) for m in langchain_messages):
            langchain_messages.insert(0, SystemMessage(content=system_prompt))
        
        # Trace: agent started
        trace_broadcaster.broadcast("agent", "Agent processing message", {
            "message_count": len(messages),
            "message_types": [msg.__class__.__name__ if hasattr(msg, '__class__') else type(msg).__name__ for msg in messages[-3:]],  # Last 3 messages
        })
        
        # Invoke LLM with tools
        response = llm_with_tools.invoke(langchain_messages)
        
        # Trace: LLM response
        tool_calls_info = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls_info = [{"name": tc.get("name", "unknown"), "args": tc.get("args", {})} for tc in response.tool_calls]
        
        trace_broadcaster.broadcast("agent", "LLM response received", {
            "content": response.content[:200] if response.content else None,
            "has_tool_calls": len(tool_calls_info) > 0,
            "tool_calls": tool_calls_info,
        })
        
        # Add response to messages
        state["messages"].append(response)
        
        # Check if there are tool calls in the response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            state["has_tool_calls"] = True
        else:
            state["has_tool_calls"] = False
        
        return state
    
    def route_after_agent(state: AgentState) -> str:
        """Route based on whether agent wants to call tools or has tool results."""
        messages = state.get("messages", [])
        
        if not messages:
            return "generate_response"
        
        last_msg = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            # LLM wants to call tools
            trace_broadcaster.broadcast("routing", "Routing to tools", {
                "decision": "tool_calls_detected",
                "tool_call_count": len(last_msg.tool_calls),
                "tool_names": [tc.get("name", "unknown") for tc in last_msg.tool_calls],
            })
            return "tools"
        elif isinstance(last_msg, ToolMessage):
            # Tool results are in history, LLM should process them
            trace_broadcaster.broadcast("routing", "Routing to agent with tool results", {
                "decision": "tool_result_received",
                "tool_call_id": getattr(last_msg, 'tool_call_id', None),
                "content_length": len(last_msg.content) if last_msg.content else 0,
            })
            return "agent"
        
        # No tool calls and no tool results, generate response
        trace_broadcaster.broadcast("routing", "Routing to generate_response", {
            "decision": "no_more_tools",
            "last_message_type": last_msg.__class__.__name__,
        })
        return "generate_response"
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("generate_response", node_generate_response)
    
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "generate_response": "generate_response",
        },
    )
    
    graph.add_edge("tools", "agent")
    
    return graph


# Pre-build graph at module level for efficiency (one-time initialization)
agent_graph = None


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
    
    # Compile graph at runtime (since build_agent_graph() is called)
    graph = build_agent_graph()
    compiled = graph.compile()
    
    try:
        result = compiled.invoke(state)
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