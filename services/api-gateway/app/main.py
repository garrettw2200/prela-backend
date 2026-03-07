"""API Gateway main application."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from .routers import alerts, api_keys, billing, comments, cost_optimization, data_sources, debug, drift, errors, eval_generation, guardrails, health, insights, multi_agent, n8n, projects, prompts, replay, security, teams, traces
from .websocket import websocket_endpoint
from shared import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info(f"Starting API Gateway ({settings.environment})")

    # Start background data source sync loop
    from .services.data_source_sync import background_sync_loop

    sync_task = asyncio.create_task(background_sync_loop())

    # Start background security scanning loop
    from .services.security_scan import background_security_scan_loop

    security_scan_task = asyncio.create_task(background_security_scan_loop())

    # Start background drift detection loop
    from .services.drift_detection import background_drift_detection_loop

    drift_task = asyncio.create_task(background_drift_detection_loop())

    # Start background alert evaluation loop
    from .services.alert_evaluation import background_alert_evaluation_loop

    alert_task = asyncio.create_task(background_alert_evaluation_loop())

    yield

    # Cancel background tasks on shutdown
    sync_task.cancel()
    security_scan_task.cancel()
    drift_task.cancel()
    alert_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    try:
        await security_scan_task
    except asyncio.CancelledError:
        pass
    try:
        await drift_task
    except asyncio.CancelledError:
        pass
    try:
        await alert_task
    except asyncio.CancelledError:
        pass

    logger.info("Shutting down API Gateway")


app = FastAPI(
    title="Prela API Gateway",
    description="API Gateway for Prela observability platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Security headers middleware (added first so it runs last, after CORS)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(billing.router, tags=["billing"])
app.include_router(api_keys.router, tags=["api-keys"])
app.include_router(projects.router, prefix="/api/v1", tags=["projects"])
app.include_router(traces.router, prefix="/api/v1", tags=["traces"])
app.include_router(n8n.router, prefix="/api/v1/n8n", tags=["n8n"])
app.include_router(multi_agent.router, prefix="/api/v1/multi-agent", tags=["multi-agent"])
app.include_router(replay.router, prefix="/api/v1/replay", tags=["replay"])
app.include_router(errors.router, prefix="/api/v1", tags=["errors"])
app.include_router(drift.router, prefix="/api/v1/drift", tags=["drift"])
app.include_router(cost_optimization.router, prefix="/api/v1/cost-optimization", tags=["cost-optimization"])
app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
app.include_router(security.router, prefix="/api/v1/security", tags=["security"])
app.include_router(debug.router, prefix="/api/v1/debug", tags=["debug"])
app.include_router(eval_generation.router, prefix="/api/v1/eval-generation", tags=["eval-generation"])
app.include_router(data_sources.router, tags=["data-sources"])
app.include_router(teams.router, tags=["teams"])
app.include_router(comments.router, tags=["comments"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(guardrails.router, prefix="/api/v1/guardrails", tags=["guardrails"])


@app.websocket("/api/v1/ws/{project_id}")
async def websocket_route(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time project updates.

    Args:
        websocket: WebSocket connection instance.
        project_id: Project identifier for event filtering.
    """
    await websocket_endpoint(websocket, project_id)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "prela-api-gateway",
        "version": "0.1.0",
        "status": "running",
    }
