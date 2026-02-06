#!/usr/bin/env python3
"""
Test script for n8n API endpoints.

Usage:
    python test_n8n_api.py

Requirements:
    pip install requests
"""

import json
import time
from typing import Any

import requests

# Configuration
BASE_URL = "http://localhost:8000/api/v1/n8n"
API_KEY = "test-api-key-1234567890"
PROJECT_ID = "test-project"
WORKFLOW_ID = "wf-test-123"
WORKFLOW_NAME = "Test Support Workflow"
EXECUTION_ID = f"exec-test-{int(time.time())}"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def print_response(title: str, response: requests.Response) -> dict[str, Any]:
    """Print formatted response."""
    print(f"\n{'='*60}")
    print(f"TEST: {title}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Response:\n{json.dumps(data, indent=2)}")
        return data
    except Exception:
        print(f"Response: {response.text}")
        return {}


def test_start_trace() -> dict[str, Any]:
    """Test starting a trace."""
    url = f"{BASE_URL}/traces/start"
    payload = {
        "project_id": PROJECT_ID,
        "workflow_id": WORKFLOW_ID,
        "workflow_name": WORKFLOW_NAME,
        "execution_id": EXECUTION_ID,
        "trace_name": "Test Customer Query",
        "attributes": {
            "environment": "test",
            "customer_id": "cust-789",
        },
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    return print_response("Start Trace", response)


def test_log_ai_call() -> dict[str, Any]:
    """Test logging an AI call."""
    url = f"{BASE_URL}/ai-calls"
    payload = {
        "project_id": PROJECT_ID,
        "execution_id": EXECUTION_ID,
        "model": "gpt-4",
        "provider": "openai",
        "prompt": [
            {"role": "system", "content": "You are a customer support assistant."},
            {"role": "user", "content": "How do I reset my password?"},
        ],
        "response": "To reset your password, follow these steps: 1. Go to the login page...",
        "token_usage": {
            "promptTokens": 150,
            "completionTokens": 89,
            "totalTokens": 239,
        },
        "latency_ms": 1234.5,
        "attributes": {
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    return print_response("Log AI Call", response)


def test_log_span() -> dict[str, Any]:
    """Test logging a custom span."""
    url = f"{BASE_URL}/spans"
    payload = {
        "project_id": PROJECT_ID,
        "execution_id": EXECUTION_ID,
        "span_name": "Database Query",
        "span_type": "tool",
        "input_data": {
            "query": "SELECT * FROM customers WHERE id = ?",
            "params": ["cust-789"],
        },
        "output_data": {
            "rows": 1,
            "customer": {"id": "cust-789", "name": "John Doe", "email": "john@example.com"},
        },
        "attributes": {
            "database": "postgres",
            "table": "customers",
            "duration_ms": 45.2,
        },
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    return print_response("Log Span", response)


def test_end_trace() -> dict[str, Any]:
    """Test ending a trace."""
    url = f"{BASE_URL}/traces/end"
    payload = {
        "project_id": PROJECT_ID,
        "execution_id": EXECUTION_ID,
        "status": "success",
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    return print_response("End Trace", response)


def test_webhook() -> dict[str, Any]:
    """Test webhook ingestion."""
    url = f"{BASE_URL}/webhook"
    payload = {
        "workflow": {
            "id": WORKFLOW_ID,
            "name": WORKFLOW_NAME,
        },
        "execution": {
            "id": f"exec-webhook-{int(time.time())}",
            "mode": "manual",
        },
        "node": {
            "name": "OpenAI Chat",
            "type": "n8n-nodes-langchain.chatOpenAi",
        },
        "data": [
            {
                "json": {
                    "output": "To reset your password, follow these steps...",
                    "model": "gpt-4",
                    "usage": {"prompt_tokens": 150, "completion_tokens": 89},
                }
            }
        ],
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    return print_response("Webhook Ingestion", response)


def test_error_handling():
    """Test error cases."""
    print(f"\n{'='*60}")
    print("TEST: Error Handling")
    print(f"{'='*60}")

    # Missing API key
    print("\n1. Missing API Key:")
    response = requests.post(
        f"{BASE_URL}/traces/start",
        headers={"Content-Type": "application/json"},
        json={"project_id": "test"},
    )
    print(f"Status: {response.status_code} (expected: 422)")

    # Invalid API key
    print("\n2. Invalid API Key:")
    response = requests.post(
        f"{BASE_URL}/traces/start",
        headers={"X-API-Key": "short", "Content-Type": "application/json"},
        json={"project_id": "test"},
    )
    print(f"Status: {response.status_code} (expected: 401)")

    # Trace not found
    print("\n3. Trace Not Found:")
    response = requests.post(
        f"{BASE_URL}/traces/end",
        headers=HEADERS,
        json={"project_id": "test", "execution_id": "nonexistent", "status": "success"},
    )
    print(f"Status: {response.status_code} (expected: 404)")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("n8n API Integration Tests")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Execution ID: {EXECUTION_ID}")

    try:
        # Test complete workflow
        start_result = test_start_trace()
        trace_id = start_result.get("trace_id")

        if trace_id:
            print(f"\n✅ Trace started successfully: {trace_id}")

            # Wait a moment for Redis propagation
            time.sleep(0.5)

            # Log AI call
            ai_result = test_log_ai_call()
            if ai_result.get("span_id"):
                print(f"\n✅ AI call logged: {ai_result['span_id']}")

            # Log custom span
            span_result = test_log_span()
            if span_result.get("span_id"):
                print(f"\n✅ Custom span logged: {span_result['span_id']}")

            # End trace
            end_result = test_end_trace()
            if end_result.get("status") == "ended":
                print(f"\n✅ Trace ended successfully")

        # Test webhook endpoint
        webhook_result = test_webhook()
        if webhook_result.get("spans_created"):
            print(
                f"\n✅ Webhook processed: {webhook_result['spans_created']} spans created"
            )

        # Test error handling
        test_error_handling()

        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)

        print("\n📝 Summary:")
        print(f"   - Trace ID: {trace_id}")
        print(f"   - Execution ID: {EXECUTION_ID}")
        print(f"   - API Key: {API_KEY}")
        print("\n💡 Check Redis for cached mappings:")
        print(f"   redis-cli GET n8n:execution:{EXECUTION_ID}")
        print("\n💡 Check Kafka topics for messages:")
        print("   - Topic: traces")
        print("   - Topic: spans")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API Gateway")
        print("   Make sure the service is running:")
        print("   cd backend/services/api-gateway")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
