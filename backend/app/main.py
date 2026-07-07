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
from langchain_core.runnables.config import get_config_list, get_executor_for_config
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
    current_dt = datetime.utcnow()
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
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
    # NOTE: CURRENT DATE is NOT hardcoded here because it changes per request.
    # The LLM should call get_current_datetime() to get the actual current date.
    # This ensures accurate eligibility calculations regardless of when the request is made.
    return f"""You are a firm but empathetic e-commerce customer service assistant.

IMPORTANT: You MUST call get_current_datetime() first to get the current date before calculating order eligibility.

CRITICAL FLOW (call tools in order, do NOT skip steps):
1. get_current_datetime() - Get current date (REQUIRED FIRST STEP)
2. get_customer_profile(customer_id=...) - Use customer_id from extraction
3. get_order_items(order_id=...) - Use order_id from extraction
4. select_items_for_refund() - If user specified items
5. process_refund() - If eligible
6. Generate response - ONLY at the end

INSTRUCTIONS:
- Extract: customer_id (usr_XXX), order_id (ORD-XXX), items from user message
- Call tools in the order above - DO NOT skip steps
- ALWAYS call get_current_datetime() as your FIRST tool call
- Generate response ONLY after all tools are called

AVAILABLE TOOLS:
- get_customer_profile(customer_id: str) - Look up customer
- get_current_datetime() - Get current date (CALL THIS FIRST)
- get_order_items(order_id: str) - Get order items
- select_items_for_refund(order_id: str, item_selection: str) - Select items
- process_refund(order_id: str, refund_amount: float, reason: str) - Process refund
- escalate_to_human(reason: str) - Escalate if needed

DO NOT write Python code - use proper tool calls only."""


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
    
    # Bind tools for consistent tool calling behavior
    # Use tool_choice='none' to prevent tool calls in final response
    llm_with_tools = llm.bind_tools(tools, tool_choice='none')
    
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
        # Use llm_with_tools for consistent tool calling behavior
        response = llm_with_tools.invoke(langchain_messages)
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
    get_order_items,
    process_refund,
    escalate_to_human,
    get_current_datetime,
    select_items_for_refund,
]

# Create the tool node with tracing
class TracingToolNode(ToolNode):
    """ToolNode that broadcasts tool execution events and properly manages message history."""
    
    def _func(
        self,
        input,
        config,
        *,
        store=None,
    ):
        """Override _func to preserve conversation history."""
        # Convert config dict to RunnableConfig if needed
        from langchain_core.runnables import RunnableConfig
        
        # Filter config to only include valid RunnableConfig keys
        valid_keys = {'tags', 'metadata', 'callbacks', 'run_name', 'max_concurrency', 'recursion_limit', 'configurable', 'run_id'}
        if type(config).__name__ == 'dict':
            filtered_config = {k: v for k, v in config.items() if k in valid_keys}
            config = RunnableConfig(**filtered_config)
        
        # DEBUG: Log incoming state
        print(f"DEBUG: BEFORE executor - config type = {type(config).__name__}, keys = {list(config.keys())}")
        if isinstance(input, dict):
            incoming_messages = input.get("messages", [])
        elif hasattr(input, "__iter__"):
            incoming_messages = list(input)
        else:
            incoming_messages = []
        
        trace_broadcaster.broadcast("tools", "ToolNode _func started", {
            "incoming_message_count": len(incoming_messages),
            "incoming_message_types": [type(m).__name__ if hasattr(m, '__class__') else type(m).__name__ for m in incoming_messages],
            "input_type": type(input).__name__,
        })
        
        # Parse tool calls from input
        tool_calls, input_type = self._parse_input(input)
        
        # Trace tool execution
        if tool_calls:
            trace_broadcaster.broadcast("tools", "Tool execution started", {
                "tool_call_count": len(tool_calls),
                "tool_calls": [
                    {"name": tc.get("name", "unknown"), "args": tc.get("args", {}), "id": tc.get("id", "unknown")}
                    for tc in tool_calls
                ],
            })
        
        # Get existing messages from input for history preservation
        existing_messages = list(incoming_messages)
        
        # Execute tools
        config_list = get_config_list(config, len(tool_calls))
        input_types = [input_type] * len(tool_calls)
        with get_executor_for_config(config) as executor:
            outputs = [
                *executor.map(self._run_one, tool_calls, input_types, config_list)
            ]
        
        # Combine tool outputs
        raw_result = self._combine_tool_outputs(outputs, input_type)
        
        # DEBUG: Log result before merge
        print(f"DEBUG: AFTER executor - raw_result type = {type(raw_result).__name__}")
        if isinstance(raw_result, dict):
            tool_results = raw_result.get("messages", [])
        else:
            tool_results = raw_result if isinstance(raw_result, list) else []
        
        trace_broadcaster.broadcast("tools", "ToolNode result before merge", {
            "tool_result_count": len(tool_results),
            "tool_result_types": [type(m).__name__ for m in tool_results],
            "existing_message_count": len(existing_messages),
            "raw_result_type": type(raw_result).__name__,
        })
        
        # Merge tool results with existing messages to preserve conversation history
        merged_messages = list(existing_messages) + list(tool_results)
        
        # Return merged messages in the expected format
        if isinstance(raw_result, dict):
            raw_result["messages"] = merged_messages
            result = raw_result
        else:
            result = {self.messages_key: merged_messages}
        
        # DEBUG: Log final merged state
        trace_broadcaster.broadcast("tools", "ToolNode _func completed", {
            "merged_message_count": len(merged_messages),
            "merged_message_types": [type(m).__name__ if hasattr(m, '__class__') else type(m).__name__ for m in merged_messages],
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
