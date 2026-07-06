# Support-Agent Application

A comprehensive customer support application with item-level returns management.

## Features

- **Item-Level Returns**: Each order item can have multiple return requests, not just order-level
- **Flexible Refund Policy**: 
  - Digital items (`digital`): Non-refundable
  - Physical items (`physical`): 
    - Unopened: Full refund
    - Opened: 15% restocking fee
- **Real-time Return Request Tracking**: View and manage return requests per item
- **REST API**: FastAPI-based backend for all operations
- **Interactive Data Viewer**: Web interface for exploring CRM data (Node.js server on port 3200)
- **Customer Chat UI**: Frontend interface for customer support interactions
- **LangGraph AI Agent**: AI-powered support agent with state-machine workflow (see `spec.md`)

## Project Structure

```
support-agent/
├── README.md                  # You are here
├── spec.md                    # LangGraph agent architecture specification
├── llm_config.json            # LLM configuration (local ollama endpoint)
├── policy_rules.md            # Refund policy rules
├── challenge.md               # Challenge / exercise documentation
├── local_crm.json             # Generated CRM data (gitignored, ~170+ items)
├── crm_orders.csv             # Flat CSV export of CRM orders
├── .gitignore
│
├── datagen/                   # Data generation tools
│   ├── spec.md                # Data generator specification
│   ├── generate-crm.py        # Python script: generates hierarchical CRM data
│   ├── requirements.txt       # Dependencies (requests)
│   └── crm_orders.csv         # Flat CSV output
│
├── backend/                   # FastAPI backend
│   └── app/
│       ├── main.py            # REST API: orders, returns, refunds
│       └── requirements.txt   # fastapi, uvicorn, pydantic, python-multipart
│
├── frontend/                  # Customer chat UI
│   ├── index.html             # Customer support chat interface (Tailwind CSS)
│   └── src/components/        # UI components (placeholder)
│
├── data-viewer/               # Interactive CRM data viewer
│   ├── index.html             # Visual CRM explorer (1000+ lines)
│   └── server.js              # Static file server (port 3200)
│
├── tests/                     # Regression tests
│   ├── conftest.py            # Pytest fixtures (30s timeout)
│   └── test_scenarios.py      # 3 refund scenario tests
│
└── doc/                       # Documentation & diagrams
    ├── state-flow.mmd         # Mermaid diagram source
    └── state-flow.png         # Rendered state flow diagram
```

## Quick Start

### 1. Generate CRM Data

```bash
# Default: generate 15 customers
python3 datagen/generate-crm.py

# Custom count: generate 20 customers
python3 datagen/generate-crm.py -n 20
```

This creates `local_crm.json` with:
- N customers (configurable via `-n`, default 15)
- ~50 sample orders
- 170+ items
- 60+ return requests in various statuses (`pending`, `processing`, `approved`, `denied`, `completed`)

**For >15 customers**, the generator uses an OpenAI-compatible LLM for realistic name generation. Set these environment variables:

```bash
export LLM_URL=https://your-openai-compatible-api.com/v1
export LLM_MODEL=gemma-4-31b-it
export LLM_API_KEY=your-api-key-here
```

Install the datagen dependency first:

```bash
pip install -r datagen/requirements.txt
```

### 2. Start the Backend API

```bash
cd backend/app
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Start the Data Viewer

```bash
cd data-viewer
node server.js
```

Then open `http://localhost:3200` in your browser to:
- View all orders with item-level details
- See return requests for each item
- Create new return requests
- Filter and search orders

## Data Model

The data is hierarchical: **Customers → Orders → Items → Return Requests**.

### Customer

```json
{
  "id": "usr_001",
  "name": "John Smith",
  "email": "john.smith@email.com",
  "address": "123 Main St, Springfield",
  "loyalty_tier": "gold",
  "order_history": [...]
}
```

**Loyalty tiers**: `standard`, `silver`, `gold`

### Order

```json
{
  "order_id": "ORD-000001",
  "order_date": "2026-06-15",
  "total_amount": 299.98,
  "shipping_address": "123 Main St, Springfield",
  "status": "shipped",
  "refund_status": "Partially Refunded",
  "refund_amount": 0.0,
  "items": [...]
}
```

**Order statuses**: `pending`, `processing`, `delivered`, `shipped`, `cancelled`

### Order Item

```json
{
  "item_id": "abc123",
  "name": "Wireless Headphones",
  "category": "Electronics",
  "quantity": 2,
  "price": 149.99,
  "item_type": "physical",
  "is_opened": false,
  "return_requests": []
}
```

**Item types**: `physical`, `digital`

### Return Request

```json
{
  "item_index": 0,
  "request_date": "2026-06-15",
  "reason": "Defective product",
  "status": "pending",
  "refund_amount": 299.98,
  "refund_date": null,
  "transaction_id": null,
  "restocking_fee_applied": false
}
```

**Return request statuses**: `pending`, `approved`, `denied`, `processing`, `completed`

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

### Digital Items (`digital`)
- Always non-refundable
- Status: 100% non-refundable

### Physical Items - Unopened (`is_opened: false`)
- Full refund eligible
- No restocking fee

### Physical Items - Opened (`is_opened: true`)
- 15% restocking fee applied
- 85% refund eligible

## Return Request Workflow

1. **Create Request**: Customer/agent creates return request (status: `pending`)
2. **Review**: Support reviews the request
3. **Approve/Deny**: Status updated to `approved`, `processing`, or `denied`
4. **Process**: Refund is processed (status: `processing` → `completed`)
5. **Complete**: Return request closed (status: `completed`)

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

## Specification & Design

- **`spec.md`** — Full specification for the LangGraph-based AI support agent, including the state machine flow (`init → authenticate → list_orders → extract → check_policy → process → generate_response`), tool registry, and SSE-based admin trace engine.
- **`llm_config.json`** — LLM configuration (local ollama endpoint, gemma-4-31B-it model).
- **`policy_rules.md`** — Detailed refund policy rules referenced by both the data generator and backend.
- **`datagen/spec.md`** — Specification for the data generator tool.
- **`doc/state-flow.mmd`** — Mermaid source for the agent state flow diagram.
- **`doc/state-flow.png`** — Rendered state flow diagram.

## Testing

Three regression test scenarios cover the refund workflow:

```bash
cd tests
pip install pytest
pytest test_scenarios.py -v
```

| Scenario | Description |
|---|---|
| Happy Path | Complete refund for an unopened physical item |
| Bronze Tier Denial | Loyalty-tier-based return denial |
| Error Recovery | Handling backend failures and retry logic |

## Development

### Modifying the Data Generator

Edit `datagen/generate-crm.py` to:
- Change the number of orders (modify `num_orders` parameter)
- Adjust return request generation rates
- Modify product categories and item pools
- Customize the LLM prompt for customer name generation

### Running the Data Generator with LLM

```bash
LLM_URL=https://api.openai.com/v1 \
LLM_MODEL=gpt-4 \
LLM_API_KEY=sk-xxx \
python3 datagen/generate-crm.py -n 25
```

## License

MIT
