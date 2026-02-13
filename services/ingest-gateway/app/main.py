"""Ingest Gateway main application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from shared import settings
from shared.clickhouse import get_clickhouse_client
from shared.database import get_user_by_clerk_id, get_subscription_by_user_id, verify_api_key
from shared.rate_limiter import get_rate_limiter, close_rate_limiter
from shared.otlp_normalizer import normalize_otlp_traces
import hashlib

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global ClickHouse client
clickhouse_client = None

# HTTP Bearer security scheme
http_bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global clickhouse_client

    logger.info(f"Starting Ingest Gateway ({settings.environment})")

    # Initialize ClickHouse client
    clickhouse_client = get_clickhouse_client()
    logger.info("ClickHouse client initialized")

    # Initialize rate limiter
    await get_rate_limiter()
    logger.info("Rate limiter initialized")

    yield

    # Shutdown
    logger.info("Shutting down Ingest Gateway")
    await close_rate_limiter()


app = FastAPI(
    title="Prela Ingest Gateway",
    description="High-throughput ingestion gateway for traces and spans",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "prela-ingest-gateway",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy"}


async def authenticate_and_check_rate_limit(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> dict:
    """Authenticate API key and check rate limit.

    Args:
        credentials: HTTP Bearer credentials.

    Returns:
        User data dictionary with user_id, tier, etc.

    Raises:
        HTTPException: If authentication fails or rate limit exceeded.
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Set PRELA_API_KEY environment variable or pass in Authorization header."
        )

    api_key = credentials.credentials

    # Verify API key format
    if not (api_key.startswith("prela_sk_") or api_key.startswith("sk_")):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. API keys should start with 'prela_sk_' or 'sk_'"
        )

    # Hash the API key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Verify API key and get user + subscription
    result = await verify_api_key(key_hash)

    if not result:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Generate a new API key at https://prela.app/settings"
        )

    # Check subscription status
    if result["subscription_status"] not in ("active", "trialing"):
        raise HTTPException(
            status_code=403,
            detail=f"Subscription {result['subscription_status']}. Please update your payment method at https://prela.app/billing"
        )

    # Check rate limit
    rate_limiter = await get_rate_limiter()
    allowed, current_usage, limit = await rate_limiter.check_limit(
        user_id=result["user_id"],
        tier=result["tier"],
        traces_count=1,
    )

    if not allowed:
        limit_str = f"{limit:,}" if limit else "unlimited"
        raise HTTPException(
            status_code=429,
            detail=f"Monthly trace limit exceeded ({current_usage:,}/{limit_str} traces). "
                   f"Upgrade at https://prela.app/pricing",
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"}
        )

    return {
        "user_id": str(result["user_id"]),
        "tier": result["tier"],
        "current_usage": current_usage,
        "limit": limit,
    }


@app.post("/v1/traces")
async def ingest_trace(
    request: Request,
    user: dict = Depends(authenticate_and_check_rate_limit),
):
    """Ingest a complete trace with all spans.

    Accepts trace data from the Prela SDK and writes directly to ClickHouse.

    Expected format:
    {
        "trace_id": "...",
        "service_name": "...",
        "spans": [...]
    }
    """
    import json

    try:
        trace_data = await request.json()

        # Validate required fields
        if "trace_id" not in trace_data:
            raise HTTPException(status_code=400, detail="Missing trace_id")

        trace_id = trace_data["trace_id"]

        # Use user_id as project_id (each user has their own project)
        project_id = user["user_id"]

        # Write directly to ClickHouse
        # Extract trace-level data
        trace_row = {
            "trace_id": trace_id,
            "project_id": project_id,
            "service_name": trace_data.get("service_name", "unknown"),
            "started_at": trace_data.get("started_at"),
            "completed_at": trace_data.get("completed_at"),
            "duration_ms": trace_data.get("duration_ms", 0),
            "status": trace_data.get("status", "unknown"),
            "root_span_id": trace_data.get("root_span_id", ""),
            "span_count": len(trace_data.get("spans", [])),
            "attributes": json.dumps(trace_data.get("attributes", {})),
            "source": "native",
        }

        # Insert trace
        trace_columns = list(trace_row.keys())
        clickhouse_client.insert("traces", [list(trace_row.values())], column_names=trace_columns)

        # Insert spans if present
        spans = trace_data.get("spans", [])
        if spans:
            span_rows = []
            for span in spans:
                span_row = {
                    "span_id": span.get("span_id", ""),
                    "trace_id": trace_id,
                    "project_id": project_id,
                    "parent_span_id": span.get("parent_span_id", ""),
                    "name": span.get("name", ""),
                    "span_type": span.get("span_type", "unknown"),
                    "service_name": trace_data.get("service_name", "unknown"),
                    "started_at": span.get("started_at"),
                    "ended_at": span.get("ended_at"),
                    "duration_ms": span.get("duration_ms", 0),
                    "status": span.get("status", "unknown"),
                    "attributes": json.dumps(span.get("attributes", {})),
                    "events": json.dumps(span.get("events", [])),
                    "replay_snapshot": json.dumps(span.get("replay_snapshot", {})),
                    "source": "native",
                }
                span_rows.append(span_row)

            span_columns = list(span_rows[0].keys())
            clickhouse_client.insert(
                "spans",
                [list(row.values()) for row in span_rows],
                column_names=span_columns,
            )

        # Increment rate limit counter
        rate_limiter = await get_rate_limiter()
        await rate_limiter.increment(user["user_id"], traces_count=1)

        logger.debug(f"Trace ingested: {trace_id} ({len(spans)} spans) - User: {user['user_id']}")

        # Build response with tier information in headers
        from fastapi.responses import JSONResponse

        response = JSONResponse(
            content={
                "status": "accepted",
                "trace_id": trace_id,
                "usage": {
                    "current": user["current_usage"] + 1,
                    "limit": user["limit"],
                }
            },
            headers={
                "X-Prela-Tier": user["tier"],
                "X-RateLimit-Limit": str(user["limit"]) if user["limit"] else "unlimited",
                "X-RateLimit-Remaining": str(max(0, (user["limit"] or 0) - user["current_usage"] - 1)),
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest trace: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest trace")


@app.post("/v1/spans")
async def ingest_span(
    request: Request,
    user: dict = Depends(authenticate_and_check_rate_limit),
):
    """Ingest a single span.

    Accepts span data from the Prela SDK and writes directly to ClickHouse.

    Expected format:
    {
        "span_id": "...",
        "trace_id": "...",
        "name": "...",
        ...
    }
    """
    import json

    try:
        span_data = await request.json()

        # Validate required fields
        if "span_id" not in span_data or "trace_id" not in span_data:
            raise HTTPException(status_code=400, detail="Missing span_id or trace_id")

        span_id = span_data["span_id"]
        trace_id = span_data["trace_id"]
        project_id = user["user_id"]

        # Write directly to ClickHouse
        span_row = {
            "span_id": span_id,
            "trace_id": trace_id,
            "project_id": project_id,
            "parent_span_id": span_data.get("parent_span_id", ""),
            "name": span_data.get("name", ""),
            "span_type": span_data.get("span_type", "unknown"),
            "service_name": span_data.get("service_name", "unknown"),
            "started_at": span_data.get("started_at"),
            "ended_at": span_data.get("ended_at"),
            "duration_ms": span_data.get("duration_ms", 0),
            "status": span_data.get("status", "unknown"),
            "attributes": json.dumps(span_data.get("attributes", {})),
            "events": json.dumps(span_data.get("events", [])),
            "replay_snapshot": json.dumps(span_data.get("replay_snapshot", {})),
            "source": "native",
        }

        span_columns = list(span_row.keys())
        clickhouse_client.insert("spans", [list(span_row.values())], column_names=span_columns)

        logger.debug(f"Span ingested: {span_id} (trace: {trace_id}) - User: {user['user_id']}")

        return {"status": "accepted", "span_id": span_id, "trace_id": trace_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest span: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest span")


@app.post("/v1/batch")
async def ingest_batch(
    request: Request,
    user: dict = Depends(authenticate_and_check_rate_limit),
):
    """Ingest a batch of spans (more efficient for high volume).

    Expected format:
    {
        "spans": [
            {"span_id": "...", "trace_id": "...", ...},
            ...
        ]
    }
    """
    import json
    import gzip
    import zlib

    try:
        # Handle gzip compression if present
        content_encoding = request.headers.get("content-encoding", "").lower()
        body = await request.body()

        # Security: Check compressed payload size
        if len(body) > settings.max_compressed_payload_size:
            logger.warning(
                f"Compressed payload too large: {len(body)} bytes "
                f"(max: {settings.max_compressed_payload_size}) - User: {user['user_id']}"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Compressed payload too large. Maximum: {settings.max_compressed_payload_size} bytes"
            )

        if content_encoding == "gzip":
            try:
                # Decompress with size limit to prevent zip bombs
                decompressed = gzip.decompress(body)

                # Security: Check decompressed size
                if len(decompressed) > settings.max_decompressed_payload_size:
                    logger.warning(
                        f"Decompressed payload too large: {len(decompressed)} bytes "
                        f"(max: {settings.max_decompressed_payload_size}) - User: {user['user_id']}"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"Decompressed payload too large. Maximum: {settings.max_decompressed_payload_size} bytes"
                    )

                # Security: Check decompression ratio (zip bomb detection)
                ratio = len(decompressed) / len(body) if len(body) > 0 else 0
                if ratio > settings.max_decompression_ratio:
                    logger.warning(
                        f"Suspicious decompression ratio: {ratio:.1f}x "
                        f"(compressed: {len(body)}, decompressed: {len(decompressed)}) - User: {user['user_id']}"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"Suspicious compression ratio detected (potential zip bomb). "
                               f"Maximum ratio: {settings.max_decompression_ratio}x"
                    )

                body = decompressed

            except zlib.error as e:
                logger.error(f"Invalid gzip data: {e} - User: {user['user_id']}")
                raise HTTPException(status_code=400, detail=f"Invalid gzip data: {str(e)}")
            except MemoryError:
                logger.error(f"Memory error during decompression - User: {user['user_id']}")
                raise HTTPException(status_code=413, detail="Payload too large to decompress")
        else:
            # Security: Also check uncompressed payload size
            if len(body) > settings.max_decompressed_payload_size:
                logger.warning(
                    f"Uncompressed payload too large: {len(body)} bytes "
                    f"(max: {settings.max_decompressed_payload_size}) - User: {user['user_id']}"
                )
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload too large. Maximum: {settings.max_decompressed_payload_size} bytes"
                )

        # Parse JSON
        batch_data = json.loads(body)

        if "spans" not in batch_data or not isinstance(batch_data["spans"], list):
            raise HTTPException(status_code=400, detail="Invalid batch format")

        spans = batch_data["spans"]
        project_id = user["user_id"]
        span_rows = []

        # Convert spans to ClickHouse format
        for span_data in spans:
            if "span_id" not in span_data or "trace_id" not in span_data:
                logger.warning(f"Skipping invalid span: {span_data}")
                continue

            span_row = {
                "span_id": span_data["span_id"],
                "trace_id": span_data["trace_id"],
                "project_id": project_id,
                "parent_span_id": span_data.get("parent_span_id", ""),
                "name": span_data.get("name", ""),
                "span_type": span_data.get("span_type", "unknown"),
                "service_name": span_data.get("service_name", "unknown"),
                "started_at": span_data.get("started_at"),
                "ended_at": span_data.get("ended_at"),
                "duration_ms": span_data.get("duration_ms", 0),
                "status": span_data.get("status", "unknown"),
                "attributes": json.dumps(span_data.get("attributes", {})),
                "events": json.dumps(span_data.get("events", [])),
                "replay_snapshot": json.dumps(span_data.get("replay_snapshot", {})),
                "source": "native",
            }
            span_rows.append(span_row)

        # Batch insert to ClickHouse
        if span_rows:
            span_columns = list(span_rows[0].keys())
            clickhouse_client.insert(
                "spans",
                [list(row.values()) for row in span_rows],
                column_names=span_columns,
            )

        # Increment rate limit counter
        rate_limiter = await get_rate_limiter()
        await rate_limiter.increment(user["user_id"], traces_count=1)

        ingested = len(span_rows)
        logger.info(f"Batch ingested: {ingested}/{len(spans)} spans - User: {user['user_id']}")

        # Build response with tier information in headers
        from fastapi.responses import JSONResponse

        response = JSONResponse(
            content={
                "status": "accepted",
                "ingested": ingested,
                "total": len(spans),
                "usage": {
                    "current": user["current_usage"] + 1,
                    "limit": user["limit"],
                }
            },
            headers={
                "X-Prela-Tier": user["tier"],
                "X-RateLimit-Limit": str(user["limit"]) if user["limit"] else "unlimited",
                "X-RateLimit-Remaining": str(max(0, (user["limit"] or 0) - user["current_usage"] - 1)),
            }
        )

        return response

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Failed to ingest batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest batch")


@app.post("/v1/otlp/v1/traces")
async def ingest_otlp_traces(
    request: Request,
    user: dict = Depends(authenticate_and_check_rate_limit),
):
    """Ingest traces in OTLP JSON format.

    Accepts traces from any OpenTelemetry-compatible exporter (LangChain,
    CrewAI, OpenLIT, etc.) and normalizes them into Prela's internal format.

    Expected format: OTLP JSON (https://opentelemetry.io/docs/specs/otlp/)
    {
        "resourceSpans": [{
            "resource": {"attributes": [...]},
            "scopeSpans": [{"spans": [...]}]
        }]
    }

    Supports gzip compression via Content-Encoding: gzip header.
    """
    import json
    import gzip
    import zlib

    try:
        # Handle gzip decompression
        content_encoding = request.headers.get("content-encoding", "").lower()
        body = await request.body()

        if len(body) > settings.max_compressed_payload_size:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large. Maximum: {settings.max_compressed_payload_size} bytes"
            )

        if content_encoding == "gzip":
            try:
                decompressed = gzip.decompress(body)

                if len(decompressed) > settings.max_decompressed_payload_size:
                    raise HTTPException(status_code=413, detail="Decompressed payload too large")

                ratio = len(decompressed) / len(body) if len(body) > 0 else 0
                if ratio > settings.max_decompression_ratio:
                    raise HTTPException(status_code=413, detail="Suspicious compression ratio")

                body = decompressed
            except zlib.error as e:
                raise HTTPException(status_code=400, detail=f"Invalid gzip data: {str(e)}")
            except MemoryError:
                raise HTTPException(status_code=413, detail="Payload too large to decompress")
        else:
            if len(body) > settings.max_decompressed_payload_size:
                raise HTTPException(status_code=413, detail="Payload too large")

        # Parse JSON
        otlp_data = json.loads(body)

        if "resourceSpans" not in otlp_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid OTLP format: missing 'resourceSpans' field"
            )

        project_id = user["user_id"]

        # Normalize OTLP to Prela format
        trace_rows, span_rows = normalize_otlp_traces(otlp_data, project_id)

        # Insert traces
        if trace_rows:
            trace_columns = list(trace_rows[0].keys())
            trace_data = [list(row.values()) for row in trace_rows]
            logger.info(f"OTLP trace columns: {trace_columns}")
            logger.info(f"OTLP trace data types: {[(type(v).__name__, v) for v in trace_data[0]]}")
            result = clickhouse_client.insert(
                "traces",
                trace_data,
                column_names=trace_columns,
            )
            logger.info(f"OTLP trace insert result: written_rows={result.written_rows}, summary={result.summary}")

        # Insert spans
        if span_rows:
            span_columns = list(span_rows[0].keys())
            span_data = [list(row.values()) for row in span_rows]
            logger.info(f"OTLP span columns: {span_columns}")
            logger.info(f"OTLP span data types: {[(type(v).__name__, v) for v in span_data[0]]}")
            result = clickhouse_client.insert(
                "spans",
                span_data,
                column_names=span_columns,
            )
            logger.info(f"OTLP span insert result: written_rows={result.written_rows}, summary={result.summary}")

        # Increment rate limit (count each trace as 1 against the limit)
        traces_count = len(trace_rows) or 1
        rate_limiter = await get_rate_limiter()
        await rate_limiter.increment(user["user_id"], traces_count=traces_count)

        logger.info(
            f"OTLP ingested: {len(trace_rows)} traces, {len(span_rows)} spans - "
            f"User: {user['user_id']}"
        )

        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "status": "accepted",
                "traces_ingested": len(trace_rows),
                "spans_ingested": len(span_rows),
                "usage": {
                    "current": user["current_usage"] + traces_count,
                    "limit": user["limit"],
                }
            },
            headers={
                "X-Prela-Tier": user["tier"],
                "X-RateLimit-Limit": str(user["limit"]) if user["limit"] else "unlimited",
                "X-RateLimit-Remaining": str(
                    max(0, (user["limit"] or 0) - user["current_usage"] - traces_count)
                ),
            }
        )

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        import traceback
        logger.error(f"Failed to ingest OTLP traces: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to ingest OTLP traces")
