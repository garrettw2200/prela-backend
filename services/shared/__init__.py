"""Shared utilities for Prela backend services."""

from .config import settings
from .clickhouse import (
    get_clickhouse_client,
    init_clickhouse_schema,
    insert_span,
    insert_trace,
    query_spans,
    query_traces,
)
from .database import (
    get_database_pool,
    close_database_pool,
    get_db_connection,
    fetch_one,
    fetch_all,
    execute,
    insert_returning,
    get_user_by_clerk_id,
    create_user,
    get_subscription_by_user_id,
    create_free_subscription,
    update_subscription_tier,
    increment_usage,
    reset_monthly_usage,
    verify_api_key,
    update_api_key_last_used,
)
# Kafka removed from infrastructure - direct ClickHouse writes instead
# from .kafka import get_consumer, get_kafka_config, get_producer, send_message
from .redis import (
    cache_delete,
    cache_get,
    cache_set,
    get_redis_client,
    publish_event,
    subscribe_to_channel,
)
from .notifications import (
    send_email_notification,
    send_slack_notification,
    format_drift_alert_email,
    format_drift_alert_slack,
)
from .task_clusterer import TaskClusterer, PromptCluster
from .model_recommender import ModelRecommender, ModelRecommendation, ModelUsageStats
from .cache_analyzer import CacheAnalyzer, CacheRecommendation
from .otlp_normalizer import normalize_otlp_traces

__all__ = [
    "settings",
    "get_clickhouse_client",
    "init_clickhouse_schema",
    "insert_span",
    "insert_trace",
    "query_spans",
    "query_traces",
    "get_database_pool",
    "close_database_pool",
    "get_db_connection",
    "fetch_one",
    "fetch_all",
    "execute",
    "insert_returning",
    "get_user_by_clerk_id",
    "create_user",
    "get_subscription_by_user_id",
    "create_free_subscription",
    "update_subscription_tier",
    "increment_usage",
    "reset_monthly_usage",
    "verify_api_key",
    "update_api_key_last_used",
    "get_redis_client",
    "cache_get",
    "cache_set",
    "cache_delete",
    "publish_event",
    "subscribe_to_channel",
    "send_email_notification",
    "send_slack_notification",
    "format_drift_alert_email",
    "format_drift_alert_slack",
    "TaskClusterer",
    "PromptCluster",
    "ModelRecommender",
    "ModelRecommendation",
    "ModelUsageStats",
    "CacheAnalyzer",
    "CacheRecommendation",
    "normalize_otlp_traces",
]
