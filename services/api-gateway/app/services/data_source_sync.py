"""Background sync service for external data sources.

Periodically polls active data sources (e.g. Langfuse) for new traces,
normalizes them, and inserts into ClickHouse.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from shared import settings
from shared.clickhouse import get_clickhouse_client
from shared.database import (
    get_active_data_sources,
    get_data_source_by_id,
    update_data_source_status,
    update_data_source_last_sync,
    fetch_one,
)
from shared.encryption import decrypt_secret
from shared.langfuse_normalizer import (
    fetch_langfuse_traces,
    fetch_langfuse_observations,
    normalize_langfuse_traces,
)

logger = logging.getLogger(__name__)


async def sync_data_source(source_id: str) -> dict[str, Any]:
    """Sync a single data source — fetch from Langfuse, normalize, insert to ClickHouse.

    Args:
        source_id: Data source UUID.

    Returns:
        Dict with sync results: traces_imported, spans_imported, success.

    Raises:
        Exception: On unrecoverable sync errors.
    """
    # Fetch data source (without user check — used by background sync)
    source = await fetch_one(
        "SELECT * FROM data_sources WHERE id = $1",
        source_id,
    )
    if not source:
        raise ValueError(f"Data source {source_id} not found")

    config = source["config"]
    if isinstance(config, str):
        config = json.loads(config)

    # Decrypt secret key
    host = config["host"]
    public_key = config["public_key"]
    secret_key = decrypt_secret(config["encrypted_secret_key"])
    project_id = source["project_id"]
    from_timestamp = config.get("last_synced_timestamp")

    logger.info(
        f"[SYNC] Starting sync for data_source={source_id}, "
        f"project={project_id}, from_timestamp={from_timestamp}"
    )

    # Fetch traces from Langfuse
    langfuse_traces = await fetch_langfuse_traces(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        from_timestamp=from_timestamp,
    )

    if not langfuse_traces:
        logger.info(f"[SYNC] No new traces for data_source={source_id}")
        # Still update last_sync_at to show we checked
        now = datetime.now(tz=timezone.utc)
        await update_data_source_last_sync(source_id, now, config)
        return {"traces_imported": 0, "spans_imported": 0, "success": True}

    # Fetch observations for each trace
    observations_by_trace: dict[str, list[dict[str, Any]]] = {}
    for lf_trace in langfuse_traces:
        trace_id = lf_trace.get("id", "")
        if not trace_id:
            continue

        observations = await fetch_langfuse_observations(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            trace_id=trace_id,
        )
        observations_by_trace[trace_id] = observations

    # Normalize to Prela format
    trace_rows, span_rows = normalize_langfuse_traces(
        langfuse_traces, observations_by_trace, project_id
    )

    # Batch insert to ClickHouse
    traces_inserted = 0
    spans_inserted = 0

    if trace_rows:
        ch_client = get_clickhouse_client()
        try:
            # Insert traces
            trace_columns = list(trace_rows[0].keys())
            ch_client.insert(
                "traces",
                [list(row.values()) for row in trace_rows],
                column_names=trace_columns,
            )
            traces_inserted = len(trace_rows)

            # Insert spans
            if span_rows:
                span_columns = list(span_rows[0].keys())
                ch_client.insert(
                    "spans",
                    [list(row.values()) for row in span_rows],
                    column_names=span_columns,
                )
                spans_inserted = len(span_rows)
        finally:
            ch_client.close()

    # Update sync state
    now = datetime.now(tz=timezone.utc)

    # Track the latest trace timestamp for incremental sync
    latest_timestamp = _find_latest_timestamp(langfuse_traces)
    config["last_synced_timestamp"] = latest_timestamp
    await update_data_source_last_sync(source_id, now, config)

    logger.info(
        f"[SYNC] Completed data_source={source_id}: "
        f"{traces_inserted} traces, {spans_inserted} spans imported"
    )

    return {
        "traces_imported": traces_inserted,
        "spans_imported": spans_inserted,
        "success": True,
    }


def _find_latest_timestamp(langfuse_traces: list[dict[str, Any]]) -> str | None:
    """Find the latest timestamp from a list of Langfuse traces.

    Used to set the from_timestamp for the next incremental sync.

    Args:
        langfuse_traces: Raw Langfuse trace dicts.

    Returns:
        ISO8601 timestamp string, or None.
    """
    latest: datetime | None = None

    for trace in langfuse_traces:
        ts_str = trace.get("timestamp")
        if not ts_str:
            continue
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if latest is None or dt > latest:
                latest = dt
        except (ValueError, TypeError):
            continue

    return latest.isoformat() if latest else None


async def sync_all_active_sources() -> None:
    """Sync all active data sources.

    Processes each source sequentially. Errors on one source don't
    stop sync of remaining sources.
    """
    sources = await get_active_data_sources()
    if not sources:
        return

    logger.info(f"[SYNC] Starting background sync for {len(sources)} active data sources")

    for source in sources:
        source_id = str(source["id"])
        try:
            result = await sync_data_source(source_id)
            logger.info(
                f"[SYNC] {source['name']}: "
                f"{result['traces_imported']} traces, {result['spans_imported']} spans"
            )
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync {source['name']} ({source_id}): {e}")
            try:
                await update_data_source_status(
                    source_id,
                    status="error",
                    error_message=str(e)[:500],
                )
            except Exception as status_err:
                logger.error(f"[SYNC] Failed to update status for {source_id}: {status_err}")


async def background_sync_loop() -> None:
    """Background loop that syncs all active data sources periodically.

    Runs every settings.data_source_sync_interval_minutes minutes.
    Designed to be launched via asyncio.create_task() in the API Gateway lifespan.
    """
    interval = settings.data_source_sync_interval_minutes * 60
    logger.info(
        f"[SYNC] Background sync loop started (interval={settings.data_source_sync_interval_minutes}m)"
    )

    # Initial delay to let the app fully start
    await asyncio.sleep(10)

    while True:
        try:
            await sync_all_active_sources()
        except Exception as e:
            logger.error(f"[SYNC] Background sync loop error: {e}")

        await asyncio.sleep(interval)
