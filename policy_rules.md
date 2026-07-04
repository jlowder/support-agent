# Refund Policy Rules

## Overview

This document outlines the refund policies for customer orders. All refund requests must be evaluated against these rules before processing.

---

## Time Window Eligibility

### Standard Orders (Bronze & Silver Tiers)
- Refund requests must be submitted within **30 days** of the purchase date.
- Orders older than 30 days are **not eligible** for refunds.

### Gold Tier Members
- Gold tier members have an extended **45-day** refund window.
- Refund requests must be submitted within 45 days of the purchase date.

---

## Item Condition Requirements

### Unopened Items
- Items that remain in original packaging and are unopened are eligible for full refund.

### Opened Items
- Opened items are subject to a **15% restocking fee**.
- The restocking fee is applied to the item price before refund calculation.

### Defective Items
- Items reported as defective bypass the 15% restocking fee.
- Customers must provide evidence of defect (photos or description).
- Defective items receive full refund regardless of opening status.

---

## Non-Refundable Items

### Digital Products
- Digital downloads, e-books, and digital guides are **strictly non-refundable**.
- This applies regardless of time window or customer tier.

### Subscriptions
- Monthly, yearly, or recurring subscription services are **strictly non-refundable**.
- Pro-rated refunds are not provided for subscription cancellations.

---

## Final Decision Logic

1. Check customer loyalty tier and purchase date against time windows.
2. If outside eligible window, deny refund with explanation.
3. If within window, check item type (digital/subscription vs physical).
4. For physical items, verify opening status and apply restocking fee if applicable.
5. For defective items, skip restocking fee and process full refund.
6. For non-refundable items, deny with explanation regardless of other factors.