import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field, RootModel, field_validator

# example usage: python generate_crm.py --seed 42 --validate --export-crm
#
# ==========================================
# 🔐 Pydantic Schema for Validation
# ==========================================


class OrderItem(BaseModel):
    name: str
    type: Literal["physical", "digital"]
    price: float = Field(gt=0)
    opened: bool = False


class Order(BaseModel):
    order_id: str
    date: str  # ISO 8601 string (e.g., "2025-04-10T16:00:00Z")
    items: List[OrderItem]
    total: float = Field(gt=0)
    refund_status: Literal["None", "Pending", "Refunded", "Denied", "Escalated"]

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: float, info) -> float:
        items = info.data.get("items")
        if items:
            computed = sum(item.price for item in items)
            # Allow small floating-point tolerance
            if abs(v - round(computed, 2)) > 0.01:
                raise ValueError(f"total {v} != computed {round(computed, 2)}")
        return v


class User(BaseModel):
    id: str
    name: str
    email: str
    loyalty_tier: Literal["Standard", "Silver", "Gold"]
    order_history: List[Order]


class CRM(RootModel):
    root: List[User]


# ==========================================
# 🛠️ CRM Generator Logic
# ==========================================

POLICY_WINDOW_STANDARD = 30
POLICY_WINDOW_GOLD = 45
REFERENCE_DATE = datetime(2025, 4, 15)

# Product catalog: (name, type, price, opened_default)
PRODUCTS = [
    ("Wireless Headphones", "physical", 89.99, True),
    ("Smart Watch Pro", "physical", 249.99, False),
    ("Organic Coffee Beans", "physical", 24.50, True),
    ("Digital Guidebook", "digital", 19.99, False),
    ("Ergonomic Keyboard", "physical", 129.99, False),
    ("Noise-Canceling Earbuds", "physical", 159.99, True),
    ("Laptop Stand", "physical", 39.99, False),
    ("Smart Home Hub", "physical", 119.99, True),
    ("Gaming Mouse", "physical", 79.99, False),
    ("Mechanical Keyboard", "physical", 149.99, True),
    ("Webcam HD", "physical", 69.99, False),
    ("USB-C Docking Station", "physical", 99.99, True),
    ("Monitor Light Bar", "physical", 34.99, False),
    ("Ergonomic Chair", "physical", 299.99, False),
    ("External SSD 1TB", "physical", 129.99, False),
    ("Gaming Headset", "physical", 59.99, True),
    ("Wireless Charging Pad", "physical", 24.99, False),
    ("Online Course Access", "digital", 149.99, False),
    ("Subscription Monthly", "digital", 9.99, False),
    ("Digital Access Pass", "digital", 49.99, False),
    ("E-Book Bundle", "digital", 29.99, False),
]

NAMES = [
    "Alice Thompson",
    "Marcus Chen",
    "Priya Patel",
    "David Kim",
    "Emma Williams",
    "James Rodriguez",
    "Linda Martinez",
    "Robert Taylor",
    "Sarah Johnson",
    "Michael Lee",
    "Olivia Brown",
    "William Garcia",
    "Sophia Davis",
    "Daniel Miller",
    "Isabella Wilson",
]


def random_past_date(max_days_old: int = 60) -> datetime:
    """Generate a random date in the past, within `max_days_old` days from REFERENCE_DATE."""
    days_ago = random.randint(1, max_days_old)
    return REFERENCE_DATE - timedelta(days=days_ago)


def generate_crm(seed: int = 42) -> CRM:
    """Generate a deterministic CRM dataset."""
    random.seed(seed)

    users = []
    for i, name in enumerate(NAMES, 1):
        user_id = f"usr_{i:03d}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        tier = random.choice(["Standard", "Silver", "Gold"])

        # 60% chance of 2 orders, 40% chance of 1
        num_orders = 2 if random.random() < 0.6 else 1
        order_history = []

        for j in range(num_orders):
            order_date = random_past_date(50)
            days_old = (REFERENCE_DATE - order_date).days

            # Select 1–3 items
            num_items = random.randint(1, 3)
            items = []
            total = 0.0

            for _ in range(num_items):
                name_, type_, price, opened = random.choice(PRODUCTS)
                item = OrderItem(name=name_, type=type_, price=price, opened=opened)
                items.append(item)
                total += price

            # Determine refund_status deterministically
            is_outside_window = (
                tier == "Standard" and days_old > POLICY_WINDOW_STANDARD
            ) or (tier == "Gold" and days_old > POLICY_WINDOW_GOLD)
            has_digital_only = all(item.type == "digital" for item in items)
            has_opened_physical = any(
                item.type == "physical" and item.opened for item in items
            )

            if is_outside_window or has_digital_only:
                refund_status = "Denied"
            elif days_old <= 7:
                refund_status = "Refunded"
            else:
                r = random.random()
                if r < 0.60:
                    refund_status = "Pending"  # Active refund request
                elif r < 0.90:
                    refund_status = "None"
                else:
                    refund_status = "Refunded"

            order = Order(
                order_id=f"ORD-2025-{order_date.strftime('%m-%d')}-{j + 1:03d}",
                date=order_date.isoformat() + "Z",
                items=items,
                total=round(total, 2),
                refund_status=refund_status,
            )
            order_history.append(order)

        users.append(
            User(
                id=user_id,
                name=name,
                email=email,
                loyalty_tier=tier,
                order_history=order_history,
            )
        )

    return CRM(root=users)


# ==========================================
# 📤 Export Functions
# ==========================================


def export_json(crm: CRM, output_path: str = "local_crm.json"):
    """Write CRM to JSON (pretty-printed)."""
    # Use model_dump() on the RootModel, which handles nested models
    data = crm.model_dump()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Wrote {len(data)} users → {output_path}")


def export_csv(crm: CRM, output_path: str = "crm_orders.csv"):
    """Export flattened order-level data to CSV."""
    rows = []
    for user in crm.root:
        for order in user.order_history:
            for item in order.items:
                rows.append(
                    {
                        "user_id": user.id,
                        "user_name": user.name,
                        "user_email": user.email,
                        "user_tier": user.loyalty_tier,
                        "order_id": order.order_id,
                        "order_date": order.date,
                        "order_total": order.total,
                        "order_refund_status": order.refund_status,
                        "item_name": item.name,
                        "item_type": item.type,
                        "item_price": item.price,
                        "item_opened": item.opened,
                    }
                )

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Wrote {len(rows)} order items → {output_path}")
    else:
        print("⚠️  No orders to export.")


# ==========================================
# 🖥️ CLI Entry Point
# ==========================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate mock CRM data for SupportAgent"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output against Pydantic schema",
    )
    parser.add_argument(
        "--export-crm",
        action="store_true",
        help="Alias for --export-csv",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export flattened orders to crm_orders.csv",
    )
    parser.add_argument(
        "--export-json", action="store_true", help="Export to local_crm.json"
    )
    parser.add_argument(
        "--json-path", default="local_crm.json", help="Output JSON path"
    )
    parser.add_argument("--csv-path", default="crm_orders.csv", help="Output CSV path")

    args = parser.parse_args()

    # Generate
    crm = generate_crm(seed=args.seed)

    # Validate (optional)
    if args.validate:
        try:
            crm.model_validate(crm)  # Pydantic v2 style
            print("✅ Schema validation passed.")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return

    # Export
    if args.export_json:
        export_json(crm, args.json_path)
    if args.export_csv or args.export_crm:
        export_csv(crm, args.csv_path)

    if not (args.export_json or args.export_csv or args.export_crm):
        print(
            "ℹ️  No export flag passed. Use --export-json and/or --export-csv to write files."
        )

    # Summary
    total_orders = sum(len(u.order_history) for u in crm.root)
    statuses = {"None": 0, "Pending": 0, "Refunded": 0, "Denied": 0}
    for u in crm.root:
        for o in u.order_history:
            statuses[o.refund_status] += 1
    print(f"📊 Generated {total_orders} orders across {len(crm.root)} users:")
    for s, count in statuses.items():
        print(f"   - {s}: {count}")


if __name__ == "__main__":
    main()
