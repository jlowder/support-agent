#!/usr/bin/env python3
"""
CRM Data Generator with Item-Level Returns Support (Hierarchical)

This script generates realistic CRM data with hierarchical structure:
Customers -> Orders -> Items.

Each customer has a profile with loyalty tier, and their order history
is nested directly under them. Each item can have multiple return requests.
"""

import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal
import uuid

# Order status constants
OrderStatus = Literal["Pending", "Processing", "Shipped", "Delivered", "Cancelled"]
ReturnRequestStatus = Literal["Pending", "Approved", "Denied", "Processing", "Completed"]
ItemType = Literal["Physical", "Digital"]
LoyaltyTier = Literal["Bronze", "Silver", "Gold", "Standard"]

@dataclass
class Customer:
    """Represents a customer profile with loyalty tier."""
    id: str
    name: str
    email: str
    address: str
    loyalty_tier: LoyaltyTier
    order_history: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "address": self.address,
            "loyalty_tier": self.loyalty_tier,
            "order_history": self.order_history
        }

@dataclass
class ReturnRequest:
    """Represents a return request for a specific item."""
    item_index: int
    request_date: str
    reason: str
    status: ReturnRequestStatus
    refund_amount: float
    refund_date: Optional[str] = None
    transaction_id: Optional[str] = None
    restocking_fee_applied: bool = False

@dataclass
class OrderItem:
    """Represents a single item in an order."""
    item_id: str
    name: str
    category: str
    quantity: int
    price: float
    item_type: ItemType
    is_opened: bool = False
    return_requests: List[ReturnRequest] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class Order:
    """Represents a customer order with multiple items."""
    order_id: str
    customer_name: str
    customer_email: str
    order_date: str
    total_amount: float
    shipping_address: str
    status: OrderStatus
    items: List[OrderItem] = field(default_factory=list)
    refund_status: str = "Not Refunded"
    refund_amount: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "order_date": self.order_date,
            "total_amount": self.total_amount,
            "shipping_address": self.shipping_address,
            "status": self.status,
            "refund_status": self.refund_status,
            "refund_amount": self.refund_amount,
            "items": [item.to_dict() for item in self.items]
        }

class CRMDataGenerator:
    """Generates realistic CRM data with hierarchical structure:
    Customers -> Orders -> Items."""
    
    # All 8 customers with IDs and loyalty tiers
    ALL_CUSTOMERS = [
        ("usr_001", "John Smith", "john.smith@email.com", "123 Main St, Springfield"),
        ("usr_002", "Jane Doe", "jane.doe@email.com", "456 Oak Ave, Portland"),
        ("usr_003", "Bob Johnson", "bob.j@email.com", "789 Pine Rd, Austin"),
        ("usr_004", "Alice Williams", "alice.w@email.com", "321 Elm St, Denver"),
        ("usr_005", "Charlie Brown", "charlie.b@email.com", "654 Maple Dr, Seattle"),
        ("usr_006", "Diana Prince", "diana.p@email.com", "987 Cedar Ln, Boston"),
        ("usr_007", "Eve Wilson", "eve.w@email.com", "147 Birch Way, Chicago"),
        ("usr_008", "Frank Miller", "frank.m@email.com", "258 Spruce Ct, Miami"),
    ]
    
    LOYALTY_TIERS: List[LoyaltyTier] = ["Bronze", "Silver", "Gold", "Standard"]
    
    def __init__(self, num_orders: int = 50):
        self.num_orders = num_orders
        self.customers: List[Customer] = []
        self.physical_items = [
            ("Wireless Headphones", "Electronics"),
            ("Bluetooth Speaker", "Electronics"),
            ("Kitchen Blender", "Home & Kitchen"),
            ("Yoga Mat", "Fitness"),
            ("Desk Lamp", "Home & Kitchen"),
            ("Running Shoes", "Apparel"),
            ("Coffee Maker", "Home & Kitchen"),
            ("Backpack", "Apparel"),
            ("Water Bottle", "Fitness"),
            ("Tablet Stand", "Electronics"),
        ]
        self.digital_items = [
            ("E-book: Python Programming", "Books"),
            ("Online Course: Web Development", "Education"),
            ("Software License: Design Tool", "Software"),
            ("Music Album: Best Hits", "Entertainment"),
            ("E-book: Data Science", "Books"),
            ("Video Course: AI Fundamentals", "Education"),
        ]
        self.return_reasons = [
            "Defective product",
            "Wrong size",
            "Not as described",
            "Changed mind",
            "Better price elsewhere",
            "Received duplicate",
            "Item damaged in shipping",
            "Not working as expected",
        ]
    
    def generate_customer_profiles(self) -> List[Customer]:
        """Generate customer profiles with IDs and random loyalty tiers."""
        customers = []
        for i, (cid, name, email, address) in enumerate(self.ALL_CUSTOMERS):
            tier = random.choice(self.LOYALTY_TIERS)
            customers.append(Customer(
                id=f"usr_{i+1:03d}",
                name=name,
                email=email,
                address=address,
                loyalty_tier=tier,
                order_history=[]
            ))
        self.customers = customers
        return customers
    
    def get_customer_for_order(self) -> tuple:
        """Get a random customer as (name, email, address, customer)."""
        customer = random.choice(self.customers)
        return customer.name, customer.email, customer.address, customer
    
    def generate_item(self, item_index: int, is_digital: bool = False) -> OrderItem:
        """Generate a single order item."""
        item_id = str(uuid.uuid4())[:8]
        
        if is_digital:
            name, category = random.choice(self.digital_items)
            price = round(random.uniform(9.99, 199.99), 2)
            item_type: ItemType = "Digital"
        else:
            name, category = random.choice(self.physical_items)
            price = round(random.uniform(19.99, 499.99), 2)
            item_type: ItemType = "Physical"
        
        quantity = random.randint(1, 3)
        is_opened = random.random() < 0.3  # 30% chance item was opened
        
        return OrderItem(
            item_id=item_id,
            name=name,
            category=category,
            quantity=quantity,
            price=price,
            item_type=item_type,
            is_opened=is_opened,
            return_requests=[]
        )
    
    def generate_return_request(self, item_index: int, item: OrderItem) -> Optional[ReturnRequest]:
        """Generate a return request for an item."""
        # Only generate return requests for some items
        if random.random() > 0.4:  # 40% chance
            return None
        
        # Digital items are non-refundable
        if item.item_type == "Digital":
            return None
        
        # Determine reason
        reason = random.choice(self.return_reasons)
        
        # Determine status
        status_options = ["Pending", "Processing", "Approved", "Completed", "Denied"]
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # More pending/processing
        status = random.choices(status_options, weights=weights)[0]
        
        # Calculate refund amount
        base_refund = item.price * item.quantity
        restocking_fee_applied = False
        
        # Denied requests have no refund
        if status == "Denied":
            base_refund = 0.0
        # 15% restocking fee for opened items in active status
        elif item.is_opened and status in ["Approved", "Processing", "Completed"]:
            restocking_fee_applied = True
            base_refund = base_refund * 0.85
        
        # Generate dates
        request_date = datetime.now() - timedelta(days=random.randint(1, 30))
        request_date_str = request_date.strftime("%Y-%m-%d")
        
        refund_date = None
        if status in ["Approved", "Processing", "Completed"]:
            refund_date = (request_date + timedelta(days=random.randint(3, 7))).strftime("%Y-%m-%d")
        
        transaction_id = None
        if status in ["Approved", "Completed"]:
            transaction_id = f"refund_{uuid.uuid4().hex[:12]}"
        
        return ReturnRequest(
            item_index=item_index,
            request_date=request_date_str,
            reason=reason,
            status=status,
            refund_amount=round(base_refund, 2),
            refund_date=refund_date,
            transaction_id=transaction_id,
            restocking_fee_applied=restocking_fee_applied
        )
    
    def generate_order(self, order_num: int) -> Order:
        """Generate a complete order with items and potential return requests."""
        customer_name, customer_email, shipping_address, customer = self.get_customer_for_order()
        
        order_id = f"ORD-{order_num:06d}"
        order_date = datetime.now() - timedelta(days=random.randint(0, 60))
        order_date_str = order_date.strftime("%Y-%m-%d")
        
        # Generate 2-5 items per order
        num_items = random.randint(2, 5)
        items = []
        
        for i in range(num_items):
            # 20% chance of digital item
            is_digital = random.random() < 0.2
            item = self.generate_item(i, is_digital)
            
            # Generate return requests for this item
            return_request = self.generate_return_request(i, item)
            if return_request:
                item.return_requests.append(return_request)
            
            items.append(item)
        
        # Calculate total
        total_amount = sum(item.price * item.quantity for item in items)
        
        # Determine order status
        status_options = ["Delivered", "Shipped", "Processing", "Pending", "Cancelled"]
        weights = [0.4, 0.25, 0.15, 0.1, 0.1]
        status = random.choices(status_options, weights=weights)[0]
        
        order = Order(
            order_id=order_id,
            customer_name=customer_name,
            customer_email=customer_email,
            order_date=order_date_str,
            total_amount=round(total_amount, 2),
            shipping_address=shipping_address,
            status=status,
            items=items
        )
        
        # If any items have return requests, update order-level refund status
        total_refund = sum(rr.refund_amount for item in items for rr in item.return_requests)
        if total_refund > 0:
            order.refund_amount = round(total_refund, 2)
            order.refund_status = "Partial Refund" if total_refund < total_amount else "Full Refund"
        
        # Assign order to customer's order_history
        customer.order_history.append(order.to_dict())
        
        return order
    
    def generate_data(self) -> dict:
        """Generate complete hierarchical CRM data.
        
        Structure:
        {
            "generated_at": "...",
            "total_customers": 8,
            "total_orders": 50,
            "customers": [
                {
                    "id": "usr_001",
                    "name": "...",
                    "email": "...",
                    "address": "...",
                    "loyalty_tier": "Gold",
                    "order_history": [...]
                },
                ...
            ]
        }
        """
        # Generate customer profiles with loyalty tiers
        self.generate_customer_profiles()
        
        # Generate orders and assign to customers
        orders = [self.generate_order(i + 1) for i in range(self.num_orders)]
        total_orders = len(orders)
        
        return {
            "generated_at": datetime.now().isoformat(),
            "total_customers": len(self.customers),
            "total_orders": total_orders,
            "customers": [customer.to_dict() for customer in self.customers]
        }
    
    def save_json(self, data: dict, filename: str):
        """Save data to JSON file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_csv(self, customers: List[Customer], filename: str):
        """Save data to CSV file (flat format for legacy compatibility)."""
        lines = []
        header = ("order_id,customer_name,customer_email,order_date,total_amount,"
                  "status,item_index,item_name,category,quantity,price,item_type,"
                  "is_opened,return_request_date,return_reason,return_status,"
                  "refund_amount,refund_date,transaction_id,restocking_fee_applied")
        lines.append(header)
        
        for customer in customers:
            for order in customer.order_history:
                for item_idx, item in enumerate(order["items"]):
                    base = [
                        order["order_id"],
                        customer.name,
                        customer.email,
                        order["order_date"],
                        str(order["total_amount"]),
                        order["status"],
                        str(item_idx),
                        f'"{item["name"]}"',
                        item["category"],
                        str(item["quantity"]),
                        str(item["price"]),
                        item["item_type"],
                        str(item["is_opened"]),
                        "", "", "", "", "", "", "", "", "", ""
                    ]
                    lines.append(",".join(base))
                    
                    for rr in item["return_requests"]:
                        rr_line = [
                            "", "", "", "", "", "",
                            "", "", "", "", "", "", "", "", "", "", "",
                            rr["request_date"],
                            f'"{rr["reason"]}"',
                            rr["status"],
                            str(rr["refund_amount"]),
                            rr.get("refund_date") or "",
                            rr.get("transaction_id") or "",
                            str(rr["restocking_fee_applied"])
                        ]
                        lines.append(",".join(rr_line))
        
        with open(filename, 'w') as f:
            f.write("\n".join(lines))


def main():
    """Main entry point."""
    print("Generating hierarchical CRM data with item-level returns support...")
    
    generator = CRMDataGenerator(num_orders=50)
    data = generator.generate_data()
    
    # Save JSON
    generator.save_json(data, "local_crm.json")
    print(f"Saved {data['total_orders']} orders across {data['total_customers']} customers to local_crm.json")
    
    # Generate CSV for legacy compatibility
    generator.save_csv(generator.customers, "crm_orders.csv")
    print("Saved CSV data to crm_orders.csv")
    
    # Print summary
    total_items = sum(
        len(order["items"])
        for customer in data["customers"]
        for order in customer["order_history"]
    )
    total_return_requests = sum(
        len(item["return_requests"])
        for customer in data["customers"]
        for order in customer["order_history"]
        for item in order["items"]
    )
    
    print(f"\n=== Hierarchical Data Summary ===")
    print(f"  Total Customers: {data['total_customers']}")
    print(f"  Total Orders: {data['total_orders']}")
    print(f"  Total Items: {total_items}")
    print(f"  Total Return Requests: {total_return_requests}")
    print(f"  Avg Orders per Customer: {data['total_orders'] / data['total_customers']:.1f}")
    print(f"  Avg Items per Order: {total_items / data['total_orders']:.1f}")
    
    # Customer breakdown
    print(f"\n=== Customer Profiles ===")
    for customer in data["customers"]:
        num_orders = len(customer["order_history"])
        print(f"  {customer['id']} | {customer['name']} | {customer['loyalty_tier']} | {num_orders} orders")
    
    # Loyalty tier distribution
    tier_counts = {}
    for customer in data["customers"]:
        tier = customer["loyalty_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    print(f"\n=== Loyalty Tier Distribution ===")
    for tier in ["Standard", "Bronze", "Silver", "Gold"]:
        count = tier_counts.get(tier, 0)
        print(f"  {tier}: {count}")
    
    # Count return request statuses
    status_counts = {}
    for customer in data["customers"]:
        for order in customer["order_history"]:
            for item in order["items"]:
                for rr in item["return_requests"]:
                    status_counts[rr["status"]] = status_counts.get(rr["status"], 0) + 1
    
    print(f"\n=== Return Request Statuses ===")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
