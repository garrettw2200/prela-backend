# n8n Integration API Guide

This document describes the n8n-specific API endpoints for the Prela API Gateway.

## Overview

The n8n integration API provides specialized endpoints for capturing traces from n8n workflows. These endpoints handle:

- Starting and ending workflow traces
- Logging custom spans within workflows
- Recording AI/LLM calls
- Receiving webhook payloads from n8n

## Authentication

All endpoints require API key authentication via the `X-API-Key` header:

```bash
curl -X POST https://api.prela.io/api/v1/n8n/traces/start \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '...'
```

## Endpoints

### 1. Start Trace

**POST** `/api/v1/n8n/traces/start`

Start a new trace for an n8n workflow execution. This creates the root span.

**Request Body:**
```json
{
  "project_id": "my-project",
  "workflow_id": "abc123",
  "workflow_name": "Customer Support Bot",
  "execution_id": "exec-456",
  "trace_name": "Support Query Processing",
  "attributes": {
    "environment": "production",
    "customer_id": "cust-789"
  }
}
```

**Response:**
```json
{
  "trace_id": "n8n-exec-456",
  "span_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

**Usage in n8n:**

Add an HTTP Request node at the start of your workflow:

```javascript
// Method: POST
// URL: {{$env.PRELA_API_URL}}/api/v1/n8n/traces/start
// Headers: X-API-Key = {{$env.PRELA_API_KEY}}
// Body (JSON):
{
  "project_id": "{{$env.PROJECT_ID}}",
  "workflow_id": "{{$workflow.id}}",
  "workflow_name": "{{$workflow.name}}",
  "execution_id": "{{$execution.id}}"
}
```

---

### 2. End Trace

**POST** `/api/v1/n8n/traces/end`

End a workflow trace with success or error status.

**Request Body:**
```json
{
  "project_id": "my-project",
  "execution_id": "exec-456",
  "status": "success",
  "error_message": null
}
```

For errors:
```json
{
  "project_id": "my-project",
  "execution_id": "exec-456",
  "status": "error",
  "error_message": "Failed to process customer query"
}
```

**Response:**
```json
{
  "status": "ended",
  "trace_id": "n8n-exec-456"
}
```

**Usage in n8n:**

Add an HTTP Request node at the end of your workflow:

```javascript
// Success path:
{
  "project_id": "{{$env.PROJECT_ID}}",
  "execution_id": "{{$execution.id}}",
  "status": "success"
}

// Error path (in error workflow):
{
  "project_id": "{{$env.PROJECT_ID}}",
  "execution_id": "{{$execution.id}}",
  "status": "error",
  "error_message": "{{$json.error}}"
}
```

---

### 3. Log Span

**POST** `/api/v1/n8n/spans`

Log a custom span within a workflow trace. Useful for tracking specific operations.

**Request Body:**
```json
{
  "project_id": "my-project",
  "execution_id": "exec-456",
  "span_name": "Database Query",
  "span_type": "tool",
  "input_data": {
    "query": "SELECT * FROM customers WHERE id = ?",
    "params": ["cust-789"]
  },
  "output_data": {
    "rows": 1,
    "customer": {"id": "cust-789", "name": "John Doe"}
  },
  "attributes": {
    "database": "postgres",
    "table": "customers"
  }
}
```

**Response:**
```json
{
  "span_id": "660e8400-e29b-41d4-a716-446655440000",
  "trace_id": "n8n-exec-456"
}
```

**Span Types:**
- `agent` - Agent/orchestrator operations
- `llm` - LLM calls (use `/ai-calls` endpoint instead)
- `tool` - Tool/function calls
- `retrieval` - Document/data retrieval
- `embedding` - Embedding generation
- `custom` - Custom operations

---

### 4. Log AI Call

**POST** `/api/v1/n8n/ai-calls`

Log an AI/LLM call with full details including prompts, responses, and token usage.

**Request Body:**
```json
{
  "project_id": "my-project",
  "execution_id": "exec-456",
  "model": "gpt-4",
  "provider": "openai",
  "prompt": [
    {
      "role": "system",
      "content": "You are a customer support assistant."
    },
    {
      "role": "user",
      "content": "How do I reset my password?"
    }
  ],
  "response": "To reset your password, follow these steps...",
  "token_usage": {
    "promptTokens": 150,
    "completionTokens": 89,
    "totalTokens": 239
  },
  "latency_ms": 1234.5,
  "attributes": {
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

**Response:**
```json
{
  "span_id": "770e8400-e29b-41d4-a716-446655440000",
  "trace_id": "n8n-exec-456"
}
```

**Usage in n8n:**

After an OpenAI/Anthropic node, add an HTTP Request node:

```javascript
{
  "project_id": "{{$env.PROJECT_ID}}",
  "execution_id": "{{$execution.id}}",
  "model": "{{$json.model}}",
  "provider": "openai",
  "prompt": "{{$('OpenAI').item.json.messages}}",
  "response": "{{$json.choices[0].message.content}}",
  "token_usage": {
    "promptTokens": "{{$json.usage.prompt_tokens}}",
    "completionTokens": "{{$json.usage.completion_tokens}}"
  }
}
```

---

### 5. Webhook Ingestion

**POST** `/api/v1/n8n/webhook`

Alternative method: receive complete workflow data via webhook.

**Request Body:**
```json
{
  "workflow": {
    "id": "abc123",
    "name": "Customer Support Bot"
  },
  "execution": {
    "id": "exec-456",
    "mode": "manual"
  },
  "node": {
    "name": "OpenAI Chat",
    "type": "n8n-nodes-langchain.chatOpenAi"
  },
  "data": [
    {
      "json": {
        "output": "To reset your password..."
      }
    }
  ]
}
```

**Response:**
```json
{
  "trace_id": "n8n-exec-456",
  "spans_created": 2,
  "status": "accepted"
}
```

**Usage in n8n:**

Add an HTTP Request node that sends workflow context:

```javascript
// Method: POST
// URL: {{$env.PRELA_API_URL}}/api/v1/n8n/webhook
// Body:
{
  "workflow": {{$workflow}},
  "execution": {{$execution}},
  "node": {{$node}},
  "data": {{$json}}
}
```

---

## Complete Workflow Example

Here's a complete n8n workflow with Prela tracing:

```
┌─────────────────────┐
│   Start Workflow    │
│  (Webhook Trigger)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Start Trace       │
│  (HTTP Request)     │  → POST /n8n/traces/start
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  OpenAI Chat Node   │
│   (Get Response)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Log AI Call       │
│  (HTTP Request)     │  → POST /n8n/ai-calls
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Custom Processing  │
│   (Code Node)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Log Span          │
│  (HTTP Request)     │  → POST /n8n/spans
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   End Trace         │
│  (HTTP Request)     │  → POST /n8n/traces/end
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Return Result     │
└─────────────────────┘
```

---

## Environment Variables

Set these in your n8n environment:

```bash
# Prela Configuration
PRELA_API_URL=https://api.prela.io
PRELA_API_KEY=your-api-key-here
PROJECT_ID=my-project
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request body
- `401 Unauthorized` - Missing or invalid API key
- `404 Not Found` - Trace not found (for end/span/ai-call endpoints)
- `500 Internal Server Error` - Server error

**Error Response Format:**
```json
{
  "detail": "Error message here"
}
```

---

## Best Practices

1. **Start trace early**: Call `/traces/start` as the first node after trigger
2. **End trace always**: Use error workflows to ensure `/traces/end` is called even on failures
3. **Use appropriate span types**: Choose the correct `span_type` for better categorization
4. **Include metadata**: Add relevant `attributes` for richer trace context
5. **Log AI calls**: Always log LLM calls with token usage for cost tracking
6. **Handle errors**: Always send error messages in `/traces/end` when workflow fails

---

## Redis Caching

The API uses Redis to maintain execution_id → trace_id mappings:

- **TTL**: 24 hours (86400 seconds)
- **Key format**: `n8n:execution:{execution_id}`
- **Cleanup**: Extended to 1 hour after trace ends

This allows child spans (logged after trace end) to still find their parent trace.

---

## Trace ID Format

Trace IDs follow the format: `n8n-{execution_id}`

This ensures:
- Unique trace per execution
- Easy correlation with n8n execution logs
- Consistent with SDK-based n8n instrumentation

---

## Next Steps

1. Set up environment variables in n8n
2. Create a test workflow using the example above
3. View traces in Prela dashboard
4. Integrate with production workflows

For questions or issues, contact support@prela.io
