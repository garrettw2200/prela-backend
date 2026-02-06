"""Upstash Kafka client utilities."""

import json
import logging
import ssl
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from .config import settings

logger = logging.getLogger(__name__)


def get_kafka_config() -> dict[str, Any]:
    """Get Upstash Kafka config with SASL authentication.

    Returns:
        Dictionary of Kafka connection configuration.
    """
    return {
        "bootstrap_servers": settings.kafka_bootstrap_servers,
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "SCRAM-SHA-256",
        "sasl_plain_username": settings.kafka_username,
        "sasl_plain_password": settings.kafka_password,
        "ssl_context": ssl.create_default_context(),
    }


async def get_producer() -> AIOKafkaProducer:
    """Create and start a Kafka producer.

    Returns:
        Started AIOKafkaProducer instance.

    Raises:
        Exception: If producer fails to start.
    """
    try:
        producer = AIOKafkaProducer(
            **get_kafka_config(),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            compression_type="gzip",
        )
        await producer.start()
        logger.info("Kafka producer started successfully")
        return producer
    except Exception as e:
        logger.error(f"Failed to start Kafka producer: {e}")
        raise


async def get_consumer(topic: str, group_id: str) -> AIOKafkaConsumer:
    """Create and start a Kafka consumer.

    Args:
        topic: Kafka topic to consume from.
        group_id: Consumer group ID.

    Returns:
        Started AIOKafkaConsumer instance.

    Raises:
        Exception: If consumer fails to start.
    """
    try:
        consumer = AIOKafkaConsumer(
            topic,
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            **get_kafka_config(),
        )
        await consumer.start()
        logger.info(f"Kafka consumer started for topic '{topic}' with group '{group_id}'")
        return consumer
    except Exception as e:
        logger.error(f"Failed to start Kafka consumer: {e}")
        raise


async def send_message(
    producer: AIOKafkaProducer, topic: str, message: dict[str, Any], key: str | None = None
) -> None:
    """Send a message to a Kafka topic.

    Args:
        producer: Kafka producer instance.
        topic: Topic to send to.
        message: Message payload (will be JSON-encoded).
        key: Optional message key for partitioning.
    """
    try:
        key_bytes = key.encode("utf-8") if key else None
        await producer.send_and_wait(topic, value=message, key=key_bytes)
        logger.debug(f"Message sent to topic '{topic}': {key}")
    except Exception as e:
        logger.error(f"Failed to send message to topic '{topic}': {e}")
        raise
