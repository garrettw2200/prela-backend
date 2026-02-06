"""Redis client utilities for caching and session management."""

import logging
from typing import Any, Optional

import redis.asyncio as redis

from .config import settings

logger = logging.getLogger(__name__)

# Global Redis client
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """Get or create Redis client.

    Returns:
        Redis client instance.

    Raises:
        Exception: If Redis connection fails.
    """
    global _redis_client

    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await _redis_client.ping()
            logger.info("Redis client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    return _redis_client


async def close_redis_client() -> None:
    """Close Redis client connection."""
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")


async def cache_set(
    key: str,
    value: Any,
    ttl: int = 3600,
) -> None:
    """Set a value in Redis cache with TTL.

    Args:
        key: Cache key.
        value: Value to cache (will be stringified).
        ttl: Time-to-live in seconds (default: 1 hour).
    """
    try:
        client = await get_redis_client()
        await client.setex(key, ttl, str(value))
    except Exception as e:
        logger.error(f"Failed to set cache key '{key}': {e}")
        # Don't raise - cache failures shouldn't break the app


async def cache_get(key: str) -> Optional[str]:
    """Get a value from Redis cache.

    Args:
        key: Cache key.

    Returns:
        Cached value or None if not found.
    """
    try:
        client = await get_redis_client()
        return await client.get(key)
    except Exception as e:
        logger.error(f"Failed to get cache key '{key}': {e}")
        return None


async def cache_delete(key: str) -> None:
    """Delete a key from Redis cache.

    Args:
        key: Cache key to delete.
    """
    try:
        client = await get_redis_client()
        await client.delete(key)
    except Exception as e:
        logger.error(f"Failed to delete cache key '{key}': {e}")


async def publish_event(channel: str, message: dict) -> None:
    """Publish an event to a Redis channel.

    Args:
        channel: Redis channel name (e.g., "project:prod-n8n:events").
        message: Event data dictionary (will be JSON-serialized).
    """
    import json

    try:
        client = await get_redis_client()
        await client.publish(channel, json.dumps(message))
        logger.debug(f"Published event to channel '{channel}'")
    except Exception as e:
        logger.error(f"Failed to publish to channel '{channel}': {e}")
        # Don't raise - pub/sub failures shouldn't break the app


async def subscribe_to_channel(channel: str):
    """Subscribe to a Redis channel and yield messages.

    Args:
        channel: Redis channel name to subscribe to.

    Yields:
        Dict: Parsed JSON message from the channel.

    Example:
        async for message in subscribe_to_channel("project:prod-n8n:events"):
            print(message)
    """
    import json

    try:
        client = await get_redis_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        logger.info(f"Subscribed to Redis channel: {channel}")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    yield data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message from '{channel}': {e}")
                    continue

    except Exception as e:
        logger.error(f"Error in Redis subscription to '{channel}': {e}")
        raise
