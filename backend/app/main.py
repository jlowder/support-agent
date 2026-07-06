"""
Backend API for Support-Agent Application with Item-Level Returns

This module provides REST API endpoints for managing orders and processing returns
at the item level rather than order level.

Data structure: Hierarchical (Customers -> Orders -> Items)
"""

import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Support-Agent API", version="1.1.0")

# Data models
class ReturnRequestStatus(str, Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    DENIED = "Denied"
    PROCESSING = "Processing"
    COMPLETED = "Completed"

class ItemType(str, Enum):
    PHYSICAL = "Physical"
    DIGITAL = "Digital"

class OrderStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    SHIPPED = "Shipped"
    DELIVERED = "Delivered"
    CANCELLED = "Cancelled"

class LoyaltyTier(str, Enum):
    STANDARD = "Standard"
    BRONZE = "Bronze"
    SILVER = "Silver"
    GOLD = "Gold"

# Request/Response Models
class ReturnRequest(BaseModel):
    item_index: int
    request_date: str
    reason: str
    status: ReturnRequestStatus
    refund_amount: float
    refund_date: Optional[str] = None
    transaction_id: Optional[str] = None
    restocking_fee_applied: bool = False

class OrderItem(BaseModel):
    item_id: str
    name: str
    category: str
    quantity: int
    price: float
    item_type: ItemType
    is_opened: bool = False
    return_requests: List[ReturnRequest] = []

class Order(BaseModel):
    order_id: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    order_date: str
    total_amount: float
    shipping_address: str
    status: OrderStatus
    items: List[OrderItem] = []
    refund_status: str = "Not Refunded"
    refund_amount: float = 0.0

class Customer(BaseModel):
    id: str
    name: str
    email: str
    address: str
    loyalty_tier: LoyaltyTier
    order_history: List[Order] = []

class CreateReturnRequest(BaseModel):
    order_id: str
    item_indices: List[int] = Field(default_factory=list)
    item_names: List[str] = Field(default_factory=list)
    reason: str
    request_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

class ProcessRefundRequest(BaseModel):
    order_id: str
    item_indices: Optional[List[int]] = None
    item_names: Optional[List[str]] = None
    confirm: bool = False

class ProcessRefundResponse(BaseModel):
    success: bool
    message: str
    return_requests: List[ReturnRequest] = []
    total_refund_amount: float = 0.0

class ReturnPolicy(BaseModel):
    digital_items_non_refundable: bool = True
    opened_physical_items_fee: float = 0.15  # 15% restocking fee
    unopened_physical_items_fee: float = 0.0
    processing_fee: float = 0.0
    max_days_since_delivery: int = 30

# Data storage
CRM_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "local_crm.json")

# Loyalty tier return policy days
LOYALTY_RETURN_DAYS = {
    "Gold": 45,
    "Silver": 35,
    "Bronze": 30,
    "Standard": 30
}

def load_crm_data() -> dict:
    """Load CRM data from JSON file."""
    try:
        with open(CRM_DATA_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "generated_at": datetime.now().isoformat(),
            "total_customers": 0,
            "total_orders": 0,
            "customers": []
        }
    except json.JSONDecodeError:
        return {
            "generated_at": datetime.now().isoformat(),
            "total_customers": 0,
            "total_orders": 0,
            "customers": []
        }

def save_crm_data(data: dict):
    """Save CRM data to JSON file."""
    with open(CRM_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def get_customer_by_id(customer_id: str) -> Optional[dict]:
    """Find a customer by their ID (usr_XXX format)."""
    data = load_crm_data()
    for customer in data.get("customers", []):
        if customer.get("id") == customer_id:
            return customer
    return None

def get_order_by_id(order_id: str) -> tuple:
    """Find an order by its ID across all customers. Returns (customer, order) tuple."""
    data = load_crm_data()
    for customer in data.get("customers", []):
        for order in customer.get("order_history", []):
            if order.get("order_id") == order_id:
                return customer, order
    return None, None

def get_item_by_index(order: dict, item_index: int) -> Optional[dict]:
    """Get an item by its index in the order."""
    if 0 <= item_index < len(order.get("items", [])):
        return order["items"][item_index]
    return None

def get_item_by_name(order: dict, item_name: str) -> Optional[dict]:
    """Get an item by its name in the order."""
    for item in order.get("items", []):
        if item.get("name") == item_name:
            return item
    return None

def calculate_refund(item: dict, restocking_fee: float = 0.15) -> float:
    """Calculate refund amount for an item."""
    base_amount = item["price"] * item["quantity"]
    
    # Apply restocking fee if item is opened
    if item.get("is_opened", False):
        return base_amount * (1 - restocking_fee)
    
    return base_amount

def validate_refund_policy(item: dict) -> dict:
    """Validate if an item can be refunded and return policy details."""
    item_type = item.get("item_type", "Physical")
    is_opened = item.get("is_opened", False)
    
    result = {
        "can_refund": True,
        "reason": "",
        "restocking_fee": 0.0,
        "refund_percentage": 100.0
    }
    
    # Digital items are non-refundable
    if item_type == "Digital":
        result["can_refund"] = False
        result["reason"] = "Digital items are non-refundable per our return policy"
        return result
    
    # Physical items with restocking fees
    if is_opened:
        result["restocking_fee"] = 0.15  # 15% fee for opened items
        result["refund_percentage"] = 85.0
        result["reason"] = "Opened item - 15% restocking fee applied"
    else:
        result["reason"] = "Unopened item - full refund eligible"
    
    return result

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Support-Agent API", "version": "1.1.0"}

@app.get("/customers", response_model=List[Customer])
async def list_customers():
    """List all customers with their profiles."""
    data = load_crm_data()
    customers = data.get("customers", [])
    return customers

@app.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: str):
    """Get a specific customer by ID."""
    customer = get_customer_by_id(customer_id)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    return customer

@app.get("/customers/{customer_id}/orders", response_model=List[Order])
async def get_customer_orders(customer_id: str):
    """Get all orders for a specific customer."""
    customer = get_customer_by_id(customer_id)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    return customer.get("order_history", [])

@app.get("/customers/{customer_id}/orders/{order_id}", response_model=Order)
async def get_customer_order(customer_id: str, order_id: str):
    """Get a specific order for a specific customer."""
    customer = get_customer_by_id(customer_id)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    for order in customer.get("order_history", []):
        if order.get("order_id") == order_id:
            return order
    
    raise HTTPException(status_code=404, detail=f"Order {order_id} not found for customer {customer_id}")

@app.get("/orders", response_model=List[Order])
async def list_orders(status: Optional[str] = None):
    """List all orders from all customers, optionally filtered by status."""
    data = load_crm_data()
    all_orders = []
    
    for customer in data.get("customers", []):
        all_orders.extend(customer.get("order_history", []))
    
    if status:
        all_orders = [o for o in all_orders if o.get("status") == status]
    
    return all_orders

@app.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str):
    """Get a specific order by ID (searches all customers)."""
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    return order

@app.get("/orders/{order_id}/items", response_model=List[OrderItem])
async def list_order_items(order_id: str):
    """List all items in an order."""
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    return order.get("items", [])

@app.post("/orders/{order_id}/return-requests", response_model=ReturnRequest)
async def create_return_request(order_id: str, request: CreateReturnRequest):
    """Create a new return request for an item."""
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Determine which items to process
    items_to_process = []
    
    # If item_indices specified, use those
    if request.item_indices:
        for idx in request.item_indices:
            item = get_item_by_index(order, idx)
            if item:
                items_to_process.append(item)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Item index {idx} not found in order"
                )
    # If item_names specified, use those
    elif request.item_names:
        for name in request.item_names:
            item = get_item_by_name(order, name)
            if item:
                items_to_process.append(item)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Item '{name}' not found in order"
                )
    else:
        raise HTTPException(
            status_code=400,
            detail="Must specify either item_indices or item_names"
        )
    
    # Create return requests for each item
    new_return_requests = []
    validation_results = []
    
    for item in items_to_process:
        # Validate refund policy
        policy_result = validate_refund_policy(item)
        
        if not policy_result["can_refund"]:
            validation_results.append({
                "item_index": order["items"].index(item),
                "item_name": item["name"],
                "valid": False,
                "reason": policy_result["reason"]
            })
            continue
        
        # Create return request
        return_request = ReturnRequest(
            item_index=order["items"].index(item),
            request_date=request.request_date,
            reason=request.reason,
            status=ReturnRequestStatus.PENDING,
            refund_amount=calculate_refund(item, policy_result["restocking_fee"]),
            restocking_fee_applied=policy_result["restocking_fee"] > 0
        )
        
        new_return_requests.append(return_request)
        validation_results.append({
            "item_index": order["items"].index(item),
            "item_name": item["name"],
            "valid": True,
            "reason": "Return request created"
        })
    
    # Update order with new return requests
    for return_request in new_return_requests:
        item_index = return_request.item_index
        if 0 <= item_index < len(order["items"]):
            order["items"][item_index]["return_requests"].append(return_request.dict())
    
    # Update order-level refund information
    total_refund = sum(rr.refund_amount for rr in new_return_requests)
    if total_refund > 0:
        order["refund_amount"] = round(total_refund, 2)
        order["refund_status"] = "Partial Refund"
    
    # Save updated data
    save_crm_data(load_crm_data())
    
    return JSONResponse(
        content={
            "success": True,
            "message": f"Created {len(new_return_requests)} return request(s)",
            "validation_results": validation_results,
            "return_requests": [rr.dict() for rr in new_return_requests]
        }
    )

@app.post("/orders/{order_id}/process-refund", response_model=ProcessRefundResponse)
async def process_refund_transaction(
    order_id: str,
    request: ProcessRefundRequest,
    background_tasks: BackgroundTasks
):
    """
    Process refund transactions for specific items in an order.
    
    This function handles item-level refunds rather than order-level refunds.
    It validates each item against the return policy and creates/approves
    return requests accordingly.
    
    Loyalty tier enforcement:
    - Gold: 45-day return window
    - Silver: 35-day return window
    - Standard/Bronze: 30-day return window
    """
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Get customer's loyalty tier for return window enforcement
    customer_data = load_crm_data()
    customer_info = None
    for c in customer_data.get("customers", []):
        if c.get("id") == customer.get("id") if customer else None:
            customer_info = c
            break
    
    if not customer_info:
        # Fallback to customer data from order
        customer_info = customer
    
    loyalty_tier = customer_info.get("loyalty_tier", "Standard")
    return_days = LOYALTY_RETURN_DAYS.get(loyalty_tier, 30)
    
    # Check if order is within return window
    order_date = datetime.strptime(order.get("order_date", ""), "%Y-%m-%d")
    days_since_order = (datetime.now() - order_date).days
    
    if days_since_order > return_days:
        raise HTTPException(
            status_code=403,
            detail=f"Order is outside the return window. Loyalty tier: {loyalty_tier}, "
                   f"Return window: {return_days} days, Days since order: {days_since_order}"
        )
    
    # Determine which items to refund
    items_to_refund = []
    
    if request.item_indices is not None:
        for idx in request.item_indices:
            item = get_item_by_index(order, idx)
            if item:
                items_to_refund.append(item)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Item index {idx} not found in order"
                )
    elif request.item_names is not None:
        for name in request.item_names:
            item = get_item_by_name(order, name)
            if item:
                items_to_refund.append(item)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Item '{name}' not found in order"
                )
    else:
        # Process all items with pending return requests
        for item in order.get("items", []):
            for rr in item.get("return_requests", []):
                if rr.get("status") == ReturnRequestStatus.PENDING:
                    items_to_refund.append(item)
                    break
    
    if not items_to_refund:
        raise HTTPException(
            status_code=400,
            detail="No items found to process refunds for"
        )
    
    # Process refunds for each item
    processed_requests = []
    total_refund_amount = 0.0
    validation_results = []
    
    for item in items_to_refund:
        item_index = order["items"].index(item)
        
        # Validate refund policy
        policy_result = validate_refund_policy(item)
        
        if not policy_result["can_refund"]:
            validation_results.append({
                "item_index": item_index,
                "item_name": item["name"],
                "processed": False,
                "reason": policy_result["reason"]
            })
            continue
        
        # Find or create return request for this item
        item_return_requests = item.get("return_requests", [])
        existing_request = None
        
        for rr in item_return_requests:
            if rr.get("status") in [ReturnRequestStatus.PENDING, ReturnRequestStatus.PROCESSING]:
                existing_request = rr
                break
        
        if existing_request:
            # Update existing return request
            existing_request["status"] = ReturnRequestStatus.APPROVED
            existing_request["restocking_fee_applied"] = policy_result["restocking_fee"] > 0
            refund_amount = calculate_refund(item, policy_result["restocking_fee"])
            existing_request["refund_amount"] = round(refund_amount, 2)
            existing_request["transaction_id"] = f"refund_{datetime.now().strftime('%Y%m%d')}-{order_id[:8]}-{item_index}"
            existing_request["refund_date"] = datetime.now().strftime("%Y-%m-%d")
            
            processed_requests.append(ReturnRequest(**existing_request))
            total_refund_amount += refund_amount
            validation_results.append({
                "item_index": item_index,
                "item_name": item["name"],
                "processed": True,
                "reason": "Return request approved and refund processed",
                "refund_amount": round(refund_amount, 2)
            })
        else:
            # Create new return request (shouldn't happen in normal flow)
            new_request = ReturnRequest(
                item_index=item_index,
                request_date=datetime.now().strftime("%Y-%m-%d"),
                reason="Refund processed by support agent",
                status=ReturnRequestStatus.APPROVED,
                refund_amount=calculate_refund(item, policy_result["restocking_fee"]),
                refund_date=datetime.now().strftime("%Y-%m-%d"),
                transaction_id=f"refund_{datetime.now().strftime('%Y%m%d')}-{order_id[:8]}-{item_index}",
                restocking_fee_applied=policy_result["restocking_fee"] > 0
            )
            
            item["return_requests"].append(new_request.dict())
            processed_requests.append(new_request)
            total_refund_amount += new_request.refund_amount
            validation_results.append({
                "item_index": item_index,
                "item_name": item["name"],
                "processed": True,
                "reason": "New return request created and approved",
                "refund_amount": round(new_request.refund_amount, 2)
            })
    
    # Update order-level refund information
    if total_refund_amount > 0:
        order["refund_amount"] = round(total_refund_amount, 2)
        if total_refund_amount >= order["total_amount"]:
            order["refund_status"] = "Full Refund"
        else:
            order["refund_status"] = "Partial Refund"
    
    # Save updated data
    save_crm_data(load_crm_data())
    
    return ProcessRefundResponse(
        success=True,
        message=f"Processed refunds for {len(processed_requests)} item(s)",
        return_requests=[rr.dict() for rr in processed_requests],
        total_refund_amount=round(total_refund_amount, 2),
        loyalty_tier=loyalty_tier,
        return_window_days=return_days,
        days_since_order=days_since_order
    )

@app.get("/policy", response_model=ReturnPolicy)
async def get_return_policy():
    """Get the current return policy."""
    return ReturnPolicy()

@app.get("/orders/{order_id}/return-requests", response_model=List[ReturnRequest])
async def list_return_requests(order_id: str):
    """List all return requests for an order."""
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    all_requests = []
    for item in order.get("items", []):
        for rr in item.get("return_requests", []):
            all_requests.append(ReturnRequest(**rr))
    
    return all_requests

@app.patch("/orders/{order_id}/return-requests/{request_index}", response_model=ReturnRequest)
async def update_return_request(
    order_id: str,
    request_index: int,
    status: ReturnRequestStatus
):
    """
    Update the status of a specific return request.
    
    This endpoint updates individual return requests at the item level,
    not the entire order.
    """
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Find the return request across all items
    updated_request = None
    item_index = None
    
    for i, item in enumerate(order.get("items", [])):
        for j, rr in enumerate(item.get("return_requests", [])):
            # Create a simple index system
            total_before = sum(len(it.get("return_requests", [])) for it in order["items"][:i])
            full_index = total_before + j
            
            if full_index == request_index:
                item_index = i
                rr["status"] = status.value
                updated_request = ReturnRequest(**rr)
                
                # Update refund date if completed/approved
                if status in [ReturnRequestStatus.APPROVED, ReturnRequestStatus.COMPLETED]:
                    if not rr.get("refund_date"):
                        rr["refund_date"] = datetime.now().strftime("%Y-%m-%d")
                    if not rr.get("transaction_id"):
                        rr["transaction_id"] = f"refund_{datetime.now().strftime('%Y%m%d')}-{order_id[:8]}-{i}"
                
                # Update refund amount if applicable
                if status == ReturnRequestStatus.DENIED:
                    rr["refund_amount"] = 0.0
                
                break
        if updated_request:
            break
    
    if not updated_request:
        raise HTTPException(
            status_code=404,
            detail=f"Return request at index {request_index} not found"
        )
    
    # Update order-level refund information
    total_refund = 0.0
    for item in order.get("items", []):
        for rr in item.get("return_requests", []):
            if rr.get("status") in [ReturnRequestStatus.APPROVED, ReturnRequestStatus.COMPLETED]:
                total_refund += rr.get("refund_amount", 0.0)
    
    if total_refund > 0:
        order["refund_amount"] = round(total_refund, 2)
        if total_refund >= order["total_amount"]:
            order["refund_status"] = "Full Refund"
        else:
            order["refund_status"] = "Partial Refund"
    
    # Save updated data
    save_crm_data(load_crm_data())
    
    return updated_request

@app.delete("/orders/{order_id}/return-requests/{request_index}")
async def delete_return_request(order_id: str, request_index: int):
    """Delete a specific return request."""
    customer, order = get_order_by_id(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Find and remove the return request
    removed = False
    item_index = None
    
    for i, item in enumerate(order.get("items", [])):
        for j, rr in enumerate(item.get("return_requests", [])):
            total_before = sum(len(it.get("return_requests", [])) for it in order["items"][:i])
            full_index = total_before + j
            
            if full_index == request_index:
                item["return_requests"].pop(j)
                removed = True
                item_index = i
                break
        if removed:
            break
    
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Return request at index {request_index} not found"
        )
    
    # Recalculate order-level refund
    total_refund = sum(
        rr.get("refund_amount", 0.0)
        for item in order.get("items", [])
        for rr in item.get("return_requests", [])
        if rr.get("status") in [ReturnRequestStatus.APPROVED, ReturnRequestStatus.COMPLETED]
    )
    
    if total_refund > 0:
        order["refund_amount"] = round(total_refund, 2)
        if total_refund >= order["total_amount"]:
            order["refund_status"] = "Full Refund"
        else:
            order["refund_status"] = "Partial Refund"
    
    save_crm_data(load_crm_data())
    
    return {"success": True, "message": "Return request deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
