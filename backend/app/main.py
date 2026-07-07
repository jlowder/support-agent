"""
Support Agent - LangGraph-based customer service agent with LLM-driven decisions.

Tool Functions (importable directly):
- get_user_profile_fn(customer_id)
- check_policy_validity_fn(order_id, check_type)
- process_refund_transaction_fn(order_id, amount)
- escalate_to_human_fn(reason)

API Endpoints:
- POST /chat - Primary chat endpoint (runs agent loop)
- GET /admin/trace - SSE stream for admin trace panel
- POST /api/voice/ingress - Voice stub
- GET /health - Health check
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CRM_PATH = BASE_DIR / "local_crm.json"
LLM_CONFIG_PATH = BASE_DIR / "llm_config.json"
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
        for q in list(self._subscriptions):
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
# Tool Functions (standalone, importable)
# ---------------------------------------------------------------------------

def get_user_profile_fn(customer_id: str) -> str:
    """Look up a customer profile from CRM data."""
    crm = get_crm()

    # Use customers array (new CRM structure)
    customers = crm.get("customers", [])

    # Build customer_map directly from customers
    customer_map = {}
    for customer in customers:
        cust_id = customer.get("id", "")
        email = customer.get("email", "")

        customer_data = {
            "id": cust_id,
            "customer_email": email,
            "customer_name": customer.get("name", ""),
            "loyalty_tier": customer.get("loyalty_tier", "standard"),
            "orders_count": len(customer.get("order_history", [])),
            "order_history": customer.get("order_history", []),
        }

        # Map both customer ID and email
        customer_map[cust_id] = customer_data
        customer_map[email] = customer_data

    # Look up customer by ID or email
    profile = customer_map.get(customer_id)

    if profile:
        profile["found"] = True
        return json.dumps(profile, default=str)

    return json.dumps({
        "found": False,
        "error": f"Customer not found for identifier: {customer_id}",
        "suggestion": "Please provide a valid customer ID (e.g., usr_001) or email address",
    }, default=str)


def _find_order_in_customers(order_id: str):
    """Find an order by ID in the customers-based CRM structure.
    Returns (order, customer) tuple, or (None, None) if not found."""
    crm = get_crm()
    customers = crm.get("customers", [])
    for customer in customers:
        for order in customer.get("order_history", []):
            if order.get("order_id") == order_id:
                return order, customer
    return None, None


def check_policy_validity_fn(order_id: str, check_type: str = "full") -> str:
    """
    Factual tool that gathers policy-relevant data about an order.
    Returns JSON with order_id, valid, days_since_purchase, and other relevant fields.
    Does NOT make policy decisions - that's the LLM's job.
    """
    crm = get_crm()
    order, customer = _find_order_in_customers(order_id)

    if not order:
        return json.dumps({
            "order_id": order_id,
            "valid": False,
            "error": f"Order {order_id} not found in CRM",
            "days_since_purchase": None,
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
        }, default=str)

    now = datetime.utcnow()
    days_since_purchase = (now - order_date).days

    # Compute eligibility windows (factual data)
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
            "has_return_requests": len(item.get("return_requests", [])) > 0,
        }
        items_data.append(item_info)

    # Check return history
    return_requests = []
    for item in order.get("items", []):
        for rr in item.get("return_requests", []):
            return_requests.append({
                "item_name": item.get("name", ""),
                "status": rr.get("status", ""),
                "reason": rr.get("reason", ""),
                "amount": rr.get("refund_amount", 0),
            })

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
        "return_history": return_requests,
        "check_type": check_type,
    }, default=str)


def process_refund_transaction_fn(order_id: str, amount: float) -> str:
    """
    Process a refund transaction. Updates order status to 'Refunded'.
    Includes retry logic - fails on first attempt if (amount * 100) is odd.
    Returns mock transaction ID on success.
    """
    crm = get_crm()

    # Find order and its location in the nested structure
    order = None
    customer_idx = None
    order_idx = None
    for ci, customer in enumerate(crm.get("customers", [])):
        for oi, o in enumerate(customer.get("order_history", [])):
            if o.get("order_id") == order_id:
                order = o
                customer_idx = ci
                order_idx = oi
                break
        if order:
            break

    if not order:
        return json.dumps({
            "success": False,
            "error": f"Order {order_id} not found",
            "transaction_id": None,
        }, default=str)

    # Simulated 503 on odd digit amounts: (amount * 100) is odd
    digit_check = int(amount * 100)
    is_odd = digit_check % 2 != 0

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        if is_odd and attempt == 0:
            # First attempt fails with simulated 503
            last_error = f"Simulated 503 Service Unavailable (odd digit amount: {digit_check})"
            if attempt < max_retries - 1:
                continue  # retry on next attempt
            break

        # Success path: update order status in nested structure
        crm["customers"][customer_idx]["order_history"][order_idx]["status"] = "Refunded"
        crm["customers"][customer_idx]["order_history"][order_idx]["refund_status"] = "Refunded"

        transaction_id = f"refund_{uuid.uuid4().hex[:12]}"
        return json.dumps({
            "success": True,
            "transaction_id": transaction_id,
            "order_id": order_id,
            "amount": amount,
            "status": "Refunded",
            "attempts": attempt + 1,
        }, default=str)

    return json.dumps({
        "success": False,
        "error": last_error or "Max retries exceeded",
        "transaction_id": None,
        "order_id": order_id,
    }, default=str)


def escalate_to_human_fn(reason: str) -> str:
    """
    Create an escalation record. Returns JSON with escalation_id and status 'logged'.
    """
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    return json.dumps({
        "escalation_id": escalation_id,
        "status": "logged",
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "priority": "high" if "supervisor" in reason.lower() else "medium",
    }, default=str)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    customer_id: Optional[str]
    customer_email: Optional[str]
    customer_name: Optional[str]
    customer_profile: Optional[dict]
    order_id: Optional[str]
    amount: Optional[float]
    messages: List[dict]  # conversation history
    refund_result: Optional[dict]
    authenticated: bool
    order_selected: bool
    order_history: Optional[list]
    response: str
    needs_human: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# LLM initialization
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    """Get configured LLM client."""
    config = _load_llm_config()
    return ChatOpenAI(
        model=config.get("model", "local-model"),
        base_url=config.get("base_url", "http://localhost:8080/v1"),
        api_key=config.get("api_key", "not-needed"),
        max_tokens=config.get("max_tokens", 1024),
        temperature=config.get("temperature", 0.3),
    )


def _get_system_prompt() -> str:
    """Build the system prompt with agent persona and policy rules."""
    policy_rules = get_policy_rules()
    return f"""You are a firm but empathetic e-commerce customer service representative.

Your role:
- Help customers with refund requests
- Verify customer identity before processing any requests
- Make refund decisions based on company policy (see below)
- Use tools to gather facts, then use your judgment to decide

REFUND POLICY RULES:
{policy_rules}

IMPORTANT DECISION RULES:
- When a customer provides an order_id and amount, check policy validity using the check_policy_validity_fn tool
- Only approve refunds that comply with the policy rules above
- If the refund does NOT comply with policy, respond with "cannot" or "unable" and explain why based on policy
- If the refund DOES comply, respond with "refund" or "success" language
- For amounts over $500, or when in doubt, escalate to a human using escalate_to_human_fn
- If the customer is not authenticated (no valid customer_id), ask for their customer ID or email
- If the customer doesn't provide an order_id, list their order history using list_orders equivalent
- Always be empathetic but firm about policy

TOOL USAGE:
- get_user_profile_fn: Use to verify customer identity. Pass customer_email or customer_id.
- check_policy_validity_fn: Use to get factual data about an order's eligibility. Pass order_id and check_type.
- process_refund_transaction_fn: Use ONLY after policy check passes. Pass order_id and amount.
- escalate_to_human_fn: Use when policy is ambiguous, amount > $500, or customer requests supervisor.

RESPONSE FORMAT:
- Give a clear, empathetic response in natural language
- State the decision (approved/denied) clearly
- If approved, include the transaction ID
- If denied, explain which policy rule was violated
- If escalating, explain that a human will follow up"""


# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------

def node_init(state: AgentState) -> AgentState:
    """Initialize the agent loop."""
    trace_broadcaster.broadcast(
        "init",
        "Agent loop initialized",
        {"customer_id": state.get("customer_id")},
    )
    return state


def node_authenticate(state: AgentState) -> AgentState:
    """Authenticate the customer using CRM lookup."""
    customer_id = state.get("customer_id")
    messages = state.get("messages", [])

    trace_broadcaster.broadcast(
        "authenticate",
        f"Authenticating customer: {customer_id}",
        {"customer_id": customer_id},
    )

    if not customer_id:
        # Scan messages for customer ID or email pattern
        # Matches: usr_001, usr_047, or any valid email address
        for message in messages:
            content = message.get("content", "")
            match = re.search(r'(usr_\d+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', content, re.IGNORECASE)
            if match:
                customer_id = match.group(1)
                state["customer_id"] = customer_id
                trace_broadcaster.broadcast(
                    "authenticate",
                    f"Extracted customer_id from message: {customer_id}",
                    {"customer_id": customer_id},
                )
                break

        if not customer_id:
            # No customer_id found at all - ask for it
            trace_broadcaster.broadcast(
                "authenticate",
                "No customer_id found in input",
                {},
            )
            state["authenticated"] = False
            return state

    # Look up customer
    profile = get_user_profile_fn(customer_id)
    profile = json.loads(profile)  # Parse JSON string response into dict
    state["customer_profile"] = profile

    if profile.get("found"):
        state["authenticated"] = True
        state["customer_email"] = profile.get("customer_email", customer_id)
        state["customer_name"] = profile.get("customer_name", "")
        trace_broadcaster.broadcast(
            "authenticate",
            f"Customer authenticated: {profile.get('customer_name')}",
            {"customer_email": profile.get("customer_email")},
        )
    else:
        state["authenticated"] = False
        state["error"] = profile.get("error", "Customer not found")
        trace_broadcaster.broadcast(
            "authenticate",
            f"Authentication failed: {profile.get('error')}",
            {"error": profile.get("error")},
        )

    return state


def node_request_auth_info(state: AgentState) -> AgentState:
    """Request customer authentication info."""
    trace_broadcaster.broadcast(
        "authenticate",
        "Requesting authentication info from customer",
        {},
    )

    state["response"] = (
        "I'd be happy to help you with your refund request! "
        "Could you please provide your customer ID (e.g., usr_001) or email address "
        "so I can look up your account?"
    )
    return state


def node_list_orders(state: AgentState) -> AgentState:
    """List customer's order history and ask them to select one."""
    customer_email = state.get("customer_email")
    crm = get_crm()

    # Get customer's orders from their order_history
    customer_orders = []
    for customer in crm.get("customers", []):
        if customer.get("email") == customer_email:
            customer_orders = customer.get("order_history", [])
            break

    state["order_history"] = customer_orders

    if not customer_orders:
        state["response"] = (
            "I wasn't able to find any orders associated with your account. "
            "Could you provide an order ID directly (e.g., ORD-000001)?"
        )
        trace_broadcaster.broadcast(
            "list_orders",
            "No orders found for customer",
            {"customer_email": customer_email},
        )
        return state

    # Format order list for display
    order_list = []
    for i, o in enumerate(customer_orders, 1):
        order_list.append(
            f"{i}. {o['order_id']} - {o.get('order_date', 'N/A')} - "
            f"${o.get('total_amount', 0):.2f} - Status: {o.get('status', 'N/A')}"
        )

    state["response"] = (
        f"I found {len(customer_orders)} order(s) in your history. "
        "Which order would you like to request a refund for? "
        "Please provide the order ID (e.g., ORD-000001):\n\n"
        + "\n".join(order_list)
    )

    trace_broadcaster.broadcast(
        "list_orders",
        f"Listed {len(customer_orders)} orders for customer",
        {"orders": len(customer_orders)},
    )
    return state


def node_extract(state: AgentState) -> AgentState:
    """Extract order_id and amount from customer message using regex + LLM fallback."""
    messages = state.get("messages", [])
    state["order_selected"] = True

    # Get the latest human message
    human_msgs = [m for m in messages if m.get("role") == "human"]
    latest_human = human_msgs[-1].get("content", "") if human_msgs else ""

    order_id = None
    amount = None

    # Try regex extraction first (fast, no LLM dependency)
    order_id_match = re.search(r'(ORD-\d{6})', latest_human, re.IGNORECASE)
    if order_id_match:
        order_id = order_id_match.group(1).upper()  # Normalize to uppercase

    # Try to extract amount if present
    amount_match = re.search(r'\$(\d+\.?\d*)', latest_human)
    if amount_match:
        amount = float(amount_match.group(1))

    # Only try LLM if regex didn't find an order ID
    if not order_id:
        try:
            llm = _get_llm()
            system = SystemMessage(content=(
                "Extract the order_id and amount from the customer's message. "
                "Return JSON with keys: order_id (string), amount (float). "
                "If either is missing, set it to null."
            ))
            response = llm.invoke([system, HumanMessage(content=latest_human)])
            content = response.content
            # Try to parse JSON
            try:
                extracted = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                match = re.search(r'\{[^}]+\}', content)
                if match:
                    extracted = json.loads(match.group())
                else:
                    extracted = {"order_id": None, "amount": None}
            order_id = extracted.get("order_id") or order_id
            amount = extracted.get("amount") if extracted.get("amount") is not None else amount
        except Exception as e:
            trace_broadcaster.broadcast(
                "extract",
                f"LLM extraction failed: {str(e)}",
                {"error": str(e)},
            )
            # Keep regex-extracted values (may be None)

    state["order_id"] = order_id
    state["amount"] = amount

    trace_broadcaster.broadcast(
        "extract",
        f"Extracted: order_id={order_id}, amount={amount}",
        {"order_id": order_id, "amount": amount},
    )

    return state


def node_show_items(state: AgentState) -> AgentState:
    """Show order line items and ask customer to select which to refund."""
    order_id = state.get("order_id")

    if not order_id:
        state["response"] = "I'm sorry, but I couldn't identify an order ID."
        return state

    # Get order details with items
    policy_data_str = check_policy_validity_fn(order_id, "full")
    policy_data = json.loads(policy_data_str)

    items = policy_data.get("items", [])
    if not items:
        state["response"] = f"No items found in order {order_id}."
        return state

    # Format items for display
    item_lines = []
    for i, item in enumerate(items, 1):
        item_name = item.get("name", "Unknown Item")
        quantity = item.get("quantity", 1)
        price = item.get("price", 0)
        item_type = item.get("item_type", "physical")
        is_opened = item.get("is_opened", False)

        # Determine status
        if item_type in ("digital", "subscription"):
            status = "Non-refundable"
        elif is_opened:
            status = "Opened (15% restocking fee)"
        else:
            status = "Unopened (Full refund)"

        item_lines.append(
            f"  {i}. {item_name} - Qty: {quantity} - ${price:.2f} each - {status}"
        )

    state["response"] = (
        f"Here are the items in order {order_id}:\n\n"
        + "\n".join(item_lines) +
        "\n\nWhich items would you like to refund? "
        "Please provide the item numbers (e.g., '1, 3' or 'all')."
    )

    # Store items in state for later use
    state["order_items"] = items
    state["order_items_formatted"] = item_lines

    return state


def node_select_items(state: AgentState) -> AgentState:
    """Process customer's item selection and calculate refund amount."""
    order_id = state.get("order_id")
    items = state.get("order_items", [])

    if not items:
        state["response"] = "No items found in this order."
        return state

    # Get customer's selection
    messages = state.get("messages", [])
    human_msgs = [m for m in messages if m.get("role") == "human"]
    latest_human = human_msgs[-1].get("content", "") if human_msgs else ""

    # Parse selection (e.g., "1, 3" or "all")
    selected_indices = []
    if latest_human.lower().strip() in ("all", "everything", "the whole order"):
        selected_indices = list(range(len(items)))
    else:
        numbers = re.findall(r'\d+', latest_human)
        selected_indices = [int(n) - 1 for n in numbers if 1 <= int(n) <= len(items)]

    if not selected_indices:
        state["response"] = "I didn't understand your selection. Please provide item numbers (e.g., '1, 3') or 'all'."
        state["_select_error"] = True
        return state

    # Calculate refund per item using rule-based policy
    approved_items = []
    denied_items = []
    total_refund = 0.0

    for idx in selected_indices:
        if idx >= len(items):
            continue

        item = items[idx]
        item_name = item.get("name", "Unknown")
        item_type = item.get("item_type", "physical")
        is_opened = item.get("is_opened", False)
        price = item.get("price", 0)
        quantity = item.get("quantity", 1)

        # Check non-refundable items
        if item_type in ("digital", "subscription"):
            denied_items.append((item, "Digital/subscription items are non-refundable"))
            continue

        # Calculate refund amount
        if is_opened:
            refund_amount = price * quantity * 0.85  # 15% restocking fee
        else:
            refund_amount = price * quantity  # Full refund

        approved_items.append((item, refund_amount))
        total_refund += refund_amount

    # Update state
    state["selected_items"] = approved_items
    state["denied_items"] = denied_items
    state["amount"] = total_refund
    state["refund_result"] = {
        "approved_items": [(i.get("name"), a) for i, a in approved_items],
        "denied_items": [(i.get("name"), r) for i, r in denied_items],
        "total_refund": total_refund,
        "approved": len(denied_items) == 0,
    }

    # Build response
    response_parts = []
    if approved_items:
        response_parts.append(f"Approved refund of ${total_refund:.2f} for:")
        for item, amount in approved_items:
            response_parts.append(f"  - {item.get('name')}: ${amount:.2f}")

    if denied_items:
        response_parts.append("\nDenied items:")
        for item, reason in denied_items:
            response_parts.append(f"  - {item.get('name')}: {reason}")

    state["response"] = "\n".join(response_parts)

    # Clear any error flag on success
    state.pop("_select_error", None)
    return state


def node_check_policy(state: AgentState) -> AgentState:
    """Check policy validity using factual tool, then apply rule-based policy."""
    order_id = state.get("order_id")
    amount = state.get("amount")

    # If amount not provided, get full order amount from CRM
    if amount is None:
        policy_data_str = check_policy_validity_fn(order_id, "full")
        policy_data = json.loads(policy_data_str)
        order_amount = policy_data.get("total_amount")
        if order_amount:
            amount = order_amount
            state["amount"] = amount

    if not order_id:
        state["response"] = "I'm sorry, I couldn't identify an order ID from your message. " \
                           "Could you please provide the order ID (e.g., ORD-000001)?"
        return state

    trace_broadcaster.broadcast(
        "check_policy",
        f"Checking policy for order: {order_id}",
        {"order_id": order_id},
    )

    # Get factual policy data
    policy_data_str = check_policy_validity_fn(order_id, "full")
    policy_data = json.loads(policy_data_str)

    # Validate order ownership - order must belong to authenticated customer
    authenticated_email = state.get("customer_email")
    order_email = policy_data.get("customer_email", "")
    if authenticated_email and order_email and authenticated_email != order_email:
        state["response"] = (
            f"I'm sorry, but order {order_id} does not belong to your account. "
            f"Refund requests can only be processed for orders associated with your account."
        )
        trace_broadcaster.broadcast(
            "check_policy",
            f"Order ownership mismatch: authenticated={authenticated_email}, order={order_email}",
            {"authenticated_email": authenticated_email, "order_email": order_email},
        )
        return state

    if not policy_data.get("valid"):
        state["response"] = (
            f"I'm sorry, but I cannot process a refund for order {order_id} "
            f"because it does not meet our policy requirements. "
            f"Error: {policy_data.get('error', 'Unknown error')}"
        )
        return state

    # Rule-based policy check (no LLM dependency)
    days_since = policy_data.get("days_since_purchase", 0)
    within_30 = policy_data.get("within_30_day_window", False)
    within_60 = policy_data.get("within_60_day_window", False)
    order_status = policy_data.get("order_status", "")

    # Basic approval logic
    approved = False
    reason = ""

    if order_status == "Refunded":
        approved = False
        reason = "This order has already been refunded."
    elif within_30:
        approved = True
        reason = "Order is within the 30-day full refund window."
    elif within_60:
        approved = True
        reason = "Order is within the 60-day partial refund window."
    else:
        approved = False
        reason = f"Order is {days_since} days old, which exceeds the refund policy window."

    # Customer tier can extend the window
    customer_tier = state.get("customer_profile", {}).get("loyalty_tier", "standard")
    if customer_tier == "gold" and days_since <= 45:
        approved = True
        reason = "Gold tier member - extended 45-day refund window applies."

    state["refund_result"] = {
        "policy_data": policy_data,
        "approved": approved,
        "reason": reason,
        "tier": customer_tier,
    }

    trace_broadcaster.broadcast(
        "check_policy",
        f"Policy decision: approved={approved}, reason={reason}",
        {"approved": approved, "reason": reason},
    )

    return state


def node_process(state: AgentState) -> AgentState:
    """Process the refund transaction."""
    order_id = state.get("order_id")
    amount = state.get("amount")
    approved = state.get("refund_result", {}).get("approved", False)

    if not approved:
        state["response"] = (
            f"I'm sorry, but I cannot process a refund for order {order_id} "
            f"because it does not meet our refund policy requirements. "
            f"Reason: {state.get('refund_result', {}).get('reason', 'Policy violation')}. "
            f"If you believe this is an error, I can escalate this to a human representative."
        )
        return state

    if not order_id or amount is None:
        state["response"] = "I'm sorry, but I need an order ID and refund amount to process this request."
        return state

    trace_broadcaster.broadcast(
        "process",
        f"Processing refund: ${amount:.2f} for order {order_id}",
        {"order_id": order_id, "amount": amount},
    )

    result_str = process_refund_transaction_fn(order_id, amount)
    result = json.loads(result_str)

    state["refund_result"]["transaction"] = result

    if result.get("success"):
        state["response"] = (
            f"Great news! Your refund of ${amount:.2f} for order {order_id} has been "
            f"successfully processed. Transaction ID: {result.get('transaction_id')}. "
            f"The refund should appear in your account within 5-7 business days."
        )
        trace_broadcaster.broadcast(
            "process",
            f"Refund successful: {result.get('transaction_id')}",
            {"transaction_id": result.get("transaction_id")},
        )
    else:
        # Try escalation on failure
        reason = f"Refund processing failed for order {order_id}: {result.get('error', 'Unknown error')}"
        esc_result = escalate_to_human_fn(reason)

        state["response"] = (
            f"I'm sorry, but we encountered an issue processing your refund of ${amount:.2f} "
            f"for order {order_id}. "
            f"I've escalated this to a human representative who will follow up with you soon. "
            f"Reference: {esc_result.get('escalation_id')}"
        )

        state["needs_human"] = True
        trace_broadcaster.broadcast(
            "process",
            f"Refund failed, escalation created: {esc_result.get('escalation_id')}",
            {"escalation_id": esc_result.get("escalation_id")},
        )

    return state


def node_generate_response(state: AgentState) -> AgentState:
    """Generate final response using LLM (for unhandled cases)."""
    trace_broadcaster.broadcast(
        "generate_response",
        "Generating final response",
        {},
    )

    # If response is already set, use it
    if state.get("response"):
        return state

    # Fallback: generate a response using LLM
    llm = _get_llm()
    system = SystemMessage(content=_get_system_prompt())

    # For local LLMs, first message must be HumanMessage (user query)
    # Extract actual user input from state messages if available
    customer_input = ""
    if state.get("messages"):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage) and msg.content.strip():
                customer_input = msg.content.strip()
                break

    messages = [
        HumanMessage(content=customer_input or "Please help me with my refund request."),
    ]

    try:
        response = llm.invoke(messages)
        state["response"] = response.content
    except Exception as e:
        state["response"] = (
            "I'm sorry, but I'm experiencing technical difficulties. "
            "A human representative will follow up with you shortly."
        )
        trace_broadcaster.broadcast(
            "generate_response",
            f"LLM response generation failed: {str(e)}",
            {"error": str(e)},
        )

    return state


# ---------------------------------------------------------------------------
# LangGraph Router
# ---------------------------------------------------------------------------

def route_after_authenticate(state: AgentState) -> str:
    """Route after authentication step."""
    if state.get("authenticated"):
        # Check if order_id is provided
        if state.get("order_id"):
            return "extract"
        else:
            return "list_orders"
    else:
        return "request_auth_info"


def route_after_list_orders(state: AgentState) -> str:
    """After listing orders, wait for customer to provide order_id."""
    # This is a terminal node - we return the response and let the client send another message
    return "extract"


def route_after_show_items(state: AgentState) -> str:
    """After showing items, wait for customer selection."""
    return "select_items"


def route_after_extract(state: AgentState) -> str:
    """After extraction, check if we have order_id and amount."""
    if not state.get("order_id"):
        state["response"] = "I couldn't find an order ID in your request. Could you please provide the order ID?"
        return "generate_response"

    # Show items first so customer can select which to refund
    return "show_items"


def route_after_check_policy(state: AgentState) -> str:
    """After policy check, decide whether to process or deny."""
    approved = state.get("refund_result", {}).get("approved", False)
    if approved:
        return "process"
    else:
        return "generate_response"


def route_after_process(state: AgentState) -> str:
    """After processing, end."""
    return "generate_response"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("init", node_init)
    graph.add_node("authenticate", node_authenticate)
    graph.add_node("request_auth_info", node_request_auth_info)
    graph.add_node("list_orders", node_list_orders)
    graph.add_node("extract", node_extract)
    graph.add_node("show_items", node_show_items)
    graph.add_node("select_items", node_select_items)
    graph.add_node("check_policy", node_check_policy)
    graph.add_node("process", node_process)
    graph.add_node("generate_response", node_generate_response)

    # Set entry point
    graph.set_entry_point("init")

    # Add edges
    graph.add_edge("init", "authenticate")

    # Authenticate -> either request_auth_info or list_orders/extract
    graph.add_conditional_edges(
        "authenticate",
        route_after_authenticate,
        {
            "request_auth_info": "request_auth_info",
            "list_orders": "list_orders",
            "extract": "extract",
        },
    )

    # request_auth_info -> end (response is set)
    graph.add_edge("request_auth_info", "generate_response")

    # list_orders -> extract (customer will provide order_id in next message)
    graph.add_edge("list_orders", "extract")

    # extract -> show_items or generate_response
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "show_items": "show_items",
            "generate_response": "generate_response",
        },
    )

    # show_items -> select_items
    graph.add_edge("show_items", "select_items")

    # select_items -> check_policy (for ownership validation) or generate_response (if error)
    graph.add_conditional_edges(
        "select_items",
        lambda s: "generate_response" if s.get("_select_error") else "check_policy",
        {
            "check_policy": "check_policy",
            "generate_response": "generate_response",
        },
    )

    # check_policy -> process or generate_response
    graph.add_conditional_edges(
        "check_policy",
        route_after_check_policy,
        {
            "process": "process",
            "generate_response": "generate_response",
        },
    )

    # process -> generate_response
    graph.add_edge("process", "generate_response")

    # generate_response -> end
    graph.add_edge("generate_response", END)

    return graph


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
    Primary chat endpoint. Runs the full agent loop and returns the LLM-generated response.
    """
    # Build initial state
    if request.messages:
        # Use provided conversation history, ensuring system prompt is included
        if request.messages[0].get("role") == "system":
            messages = request.messages + [{"role": "human", "content": request.message}]
        else:
            # Prepend system prompt; request.messages already contains the current user message
            messages = [
                {"role": "system", "content": _get_system_prompt()},
            ] + request.messages
    else:
        # Fresh conversation - build from scratch
        messages = [
            {"role": "system", "content": _get_system_prompt()},
            {"role": "human", "content": request.message},
        ]

    state: AgentState = {
        "customer_id": request.customer_id,
        "customer_email": None,
        "customer_name": None,
        "customer_profile": None,
        "order_id": None,
        "amount": None,
        "messages": messages,
        "refund_result": None,
        "authenticated": False,
        "order_selected": False,
        "order_history": None,
        "response": "",
        "needs_human": False,
        "error": None,
    }

    # Build and run graph
    graph = build_agent_graph()
    compiled = graph.compile()

    try:
        result = compiled.invoke(state)
        return ChatResponse(response=result.get("response", "I'm sorry, I couldn't process your request."))
    except Exception as e:
        trace_broadcaster.broadcast(
            "error",
            f"Agent loop error: {str(e)}",
            {"error": str(e)},
        )
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
