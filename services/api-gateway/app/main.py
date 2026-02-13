"""API Gateway main application."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routers import api_keys, billing, cost_optimization, data_sources, drift, errors, health, insights, multi_agent, n8n, projects, replay, security, traces
from .websocket import websocket_endpoint
from shared import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


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

    yield

    # Cancel background tasks on shutdown
    sync_task.cancel()
    security_scan_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    try:
        await security_scan_task
    except asyncio.CancelledError:
        pass

    logger.info("Shutting down API Gateway")


app = FastAPI(
    title="Prela API Gateway",
    description="API Gateway for Prela observability platform",
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

# Include routers
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
app.include_router(data_sources.router, tags=["data-sources"])


@app.websocket("/ws/{project_id}")
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
