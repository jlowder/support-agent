# Support-Agent Application

A comprehensive customer support application with item-level returns management.

## Features

- **Item-Level Returns**: Each order item can have multiple return requests, not just order-level
- **Flexible Refund Policy**: 
  - Digital items: Non-refundable
  - Physical items: 
    - Unopened: Full refund
    - Opened: 15% restocking fee
- **Real-time Return Request Tracking**: View and manage return requests per item
- **REST API**: FastAPI-based backend for all operations
- **Data Viewer**: Interactive web interface for exploring CRM data

## Project Structure

```
support-agent/
├── generate-crm.py          # Data generator with item-level returns
├── backend/
│   └── app/
│       └── main.py          # FastAPI backend with item-level refund logic
├── data-viewer/
│   └── index.html           # Interactive data viewer
├── local_crm.json           # Generated CRM data (gitignored)
└── README.md
```

## Quick Start

### 1. Generate CRM Data

```bash
python3 generate-crm.py
```

This creates `local_crm.json` with:
- 50 sample orders
- 170+ items
- 60+ return requests in various statuses (Pending, Processing, Approved, Denied, Completed)

### 2. Start the Backend API

```bash
cd backend/app
pip install fastapi uvicorn
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Open the Data Viewer

Open `data-viewer/index.html` in your browser to:
- View all orders with item-level details
- See return requests for each item
- Create new return requests
- Filter and search orders

## Data Structure

### Order Item

```json
{
  "item_id": "abc123",
  "name": "Wireless Headphones",
  "category": "Electronics",
  "quantity": 2,
  "price": 149.99,
  "item_type": "Physical",
  "is_opened": false,
  "return_requests": []
}
```

### Return Request

```json
{
  "item_index": 0,
  "request_date": "2026-06-15",
  "reason": "Defective product",
  "status": "Pending",
  "refund_amount": 299.98,
  "refund_date": null,
  "transaction_id": null,
  "restocking_fee_applied": false
}
```

## API Endpoints

### Orders
- `GET /orders` - List all orders
- `GET /orders/{order_id}` - Get specific order
- `GET /orders/{order_id}/items` - List items in order
- `GET /orders/{order_id}/return-requests` - List return requests

### Return Requests
- `POST /orders/{order_id}/return-requests` - Create return request
- `PATCH /orders/{order_id}/return-requests/{request_index}` - Update return status
- `DELETE /orders/{order_id}/return-requests/{request_index}` - Delete return request

### Refunds
- `POST /orders/{order_id}/process-refund` - Process refund for specific items
- `GET /policy` - Get return policy

## Refund Policy Logic

### Digital Items
- Always non-refundable
- Status: 100% non-refundable

### Physical Items - Unopened
- Full refund eligible
- No restocking fee

### Physical Items - Opened
- 15% restocking fee applied
- 85% refund eligible

## Return Request Workflow

1. **Create Request**: Customer/agent creates return request (status: Pending)
2. **Review**: Support reviews the request
3. **Approve/Deny**: Status updated to Approved, Processing, or Denied
4. **Process**: Refund is processed (status: Processing → Completed)
5. **Complete**: Return request closed (status: Completed)

## Usage Examples

### Create a Return Request (API)

```bash
curl -X POST http://localhost:8000/orders/ORD-000001/return-requests \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD-000001",
    "item_indices": [0, 2],
    "reason": "Defective product"
  }'
```

### Process a Refund (API)

```bash
curl -X POST http://localhost:8000/orders/ORD-000001/process-refund \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD-000001",
    "item_indices": [0, 2],
    "confirm": true
  }'
```

### View Return Policy

```bash
curl http://localhost:8000/policy
```

## Development

### Testing the Backend

```bash
cd backend/app
pip install -r requirements.txt
uvicorn main:app --reload

# Test endpoints
curl http://localhost:8000/
curl http://localhost:8000/orders | jq
curl http://localhost:8000/orders/ORD-000001
```

### Modifying Data Generator

Edit `generate-crm.py` to:
- Change number of orders
- Adjust return request rates
- Modify product categories
- Customize customer data

## License

MIT