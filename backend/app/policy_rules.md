# Refund Policy Rules

## Core Principles
- Be firm but empathetic in all communications
- Always verify customer identity before processing any requests
- Only process refunds for eligible orders within the policy window

## Eligibility Rules
1. Orders must be within 30 days of purchase to qualify for a full refund
2. Orders between 31-60 days may qualify for a partial refund
3. Orders older than 60 days are not eligible for refunds
4. Digital products are non-refundable once accessed/downloaded
5. Damaged or defective products are eligible for refund regardless of timing
6. "Changed mind" returns within 30 days are eligible for full refund
7. Restocking fees (15%) apply to opened physical items after 14 days

## Validation Checks
- `purchase_date`: Check if within 30-day full refund window
- `product_type`: Verify digital vs physical eligibility
- `item_condition`: Check if opened/closed affects refund amount
- `return_history`: Check for duplicate refund requests

## Escalation Triggers
- Customer requests supervisor
- Refund amount exceeds $500
- System errors persist after retries
- Ambiguous policy edge cases
