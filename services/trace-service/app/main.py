"""Trace Service main worker.

This service consumes traces and spans from Kafka and writes them to ClickHouse.
"""

import asyncio
import json
import logging
import signal
from datetime import datetime

from shared import (
    get_clickhouse_client,
    get_consumer,
    init_clickhouse_schema,
    publish_event,
    settings,
)

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Graceful shutdown flag
shutdown_event = asyncio.Event()


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


async def process_span(client, span_data: dict) -> None:
    """Process and insert a span into ClickHouse.

    Args:
        client: ClickHouse client.
        span_data: Span data dictionary.
    """
    try:
        # Extract fields
        span_id = span_data.get("span_id")
        trace_id = span_data.get("trace_id")
        parent_span_id = span_data.get("parent_span_id", "")
        name = span_data.get("name", "")
        span_type = span_data.get("span_type", "")
        service_name = span_data.get("service_name", "unknown")
        started_at = span_data.get("started_at")
        ended_at = span_data.get("ended_at")
        duration_ms = span_data.get("duration_ms", 0.0)
        status = span_data.get("status", "UNKNOWN")
        attributes = json.dumps(span_data.get("attributes", {}))
        events = json.dumps(span_data.get("events", []))

        # Insert into ClickHouse
        client.insert(
            "spans",
            [
                [
                    span_id,
                    trace_id,
                    parent_span_id,
                    name,
                    span_type,
                    service_name,
                    started_at,
                    ended_at,
                    duration_ms,
                    status,
                    attributes,
                    events,
                ]
            ],
            column_names=[
                "span_id",
                "trace_id",
                "parent_span_id",
                "name",
                "span_type",
                "service_name",
                "started_at",
                "ended_at",
                "duration_ms",
                "status",
                "attributes",
                "events",
            ],
        )

        logger.debug(f"Span inserted: {span_id}")

    except Exception as e:
        logger.error(f"Failed to process span {span_data.get('span_id')}: {e}")


async def process_trace(client, trace_data: dict) -> None:
    """Process and insert a trace into ClickHouse.

    Args:
        client: ClickHouse client.
        trace_data: Trace data dictionary.
    """
    try:
        # Extract fields
        trace_id = trace_data.get("trace_id")
        service_name = trace_data.get("service_name", "unknown")
        started_at = trace_data.get("started_at")
        completed_at = trace_data.get("completed_at")
        duration_ms = trace_data.get("duration_ms", 0.0)
        status = trace_data.get("status", "UNKNOWN")
        root_span_id = trace_data.get("root_span_id", "")
        spans = trace_data.get("spans", [])
        span_count = len(spans)
        metadata = json.dumps(trace_data.get("metadata", {}))

        # Insert trace record
        client.insert(
            "traces",
            [
                [
                    trace_id,
                    service_name,
                    started_at,
                    completed_at,
                    duration_ms,
                    status,
                    root_span_id,
                    span_count,
                    metadata,
                ]
            ],
            column_names=[
                "trace_id",
                "service_name",
                "started_at",
                "completed_at",
                "duration_ms",
                "status",
                "root_span_id",
                "span_count",
                "metadata",
            ],
        )

        # Insert all spans
        for span_data in spans:
            span_data["trace_id"] = trace_id  # Ensure trace_id is set
            span_data.setdefault("service_name", service_name)
            await process_span(client, span_data)

        logger.info(f"Trace inserted: {trace_id} ({span_count} spans)")

        # Publish event to Redis for real-time updates
        # Extract project_id from metadata or attributes
        metadata_dict = trace_data.get("metadata", {})
        project_id = metadata_dict.get("project_id", "default")

        # Extract workflow_id for n8n traces (if available)
        workflow_id = metadata_dict.get("workflow_id")

        event_data = {
            "event": "trace.created",
            "trace_id": trace_id,
            "project_id": project_id,
            "service_name": service_name,
            "status": status,
            "duration_ms": duration_ms,
            "span_count": span_count,
            "started_at": started_at,
        }

        # Add workflow_id if this is an n8n trace
        if workflow_id:
            event_data["workflow_id"] = workflow_id

        # Publish to project-specific channel
        channel = f"project:{project_id}:events"
        await publish_event(channel, event_data)
        logger.debug(f"Published trace.created event to {channel}")

    except Exception as e:
        logger.error(f"Failed to process trace {trace_data.get('trace_id')}: {e}")


async def consume_spans():
    """Consume spans from Kafka and write to ClickHouse."""
    logger.info("Starting span consumer")

    client = get_clickhouse_client()
    consumer = await get_consumer(
        settings.kafka_topic_spans, group_id="trace-service-spans"
    )

    try:
        while not shutdown_event.is_set():
            # Poll for messages with timeout
            data = await consumer.getmany(timeout_ms=1000, max_records=100)

            for tp, messages in data.items():
                for message in messages:
                    span_data = message.value
                    await process_span(client, span_data)

    except Exception as e:
        logger.error(f"Error in span consumer: {e}")
    finally:
        await consumer.stop()
        logger.info("Span consumer stopped")


async def consume_traces():
    """Consume traces from Kafka and write to ClickHouse."""
    logger.info("Starting trace consumer")

    client = get_clickhouse_client()
    consumer = await get_consumer(
        settings.kafka_topic_traces, group_id="trace-service-traces"
    )

    try:
        while not shutdown_event.is_set():
            # Poll for messages with timeout
            data = await consumer.getmany(timeout_ms=1000, max_records=50)

            for tp, messages in data.items():
                for message in messages:
                    trace_data = message.value
                    await process_trace(client, trace_data)

    except Exception as e:
        logger.error(f"Error in trace consumer: {e}")
    finally:
        await consumer.stop()
        logger.info("Trace consumer stopped")


async def main():
    """Main entry point."""
    logger.info(f"Starting Trace Service ({settings.environment})")

    # Initialize ClickHouse schema
    try:
        client = get_clickhouse_client()
        await init_clickhouse_schema(client)
    except Exception as e:
        logger.error(f"Failed to initialize ClickHouse schema: {e}")
        return

    # Start consumers
    tasks = [
        asyncio.create_task(consume_spans()),
        asyncio.create_task(consume_traces()),
    ]

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Wait for tasks to complete
    logger.info("Waiting for consumers to finish...")
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Trace Service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
