"""API Gateway main application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routers import api_keys, billing, cost_optimization, drift, errors, health, multi_agent, n8n, projects, replay, traces
from .websocket import websocket_endpoint
from shared import get_producer, settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global Kafka producer
kafka_producer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global kafka_producer

    logger.info(f"Starting API Gateway ({settings.environment})")

    # Initialize Kafka producer for n8n routes
    try:
        kafka_producer = await get_producer()
        n8n.set_kafka_producer(kafka_producer)
        logger.info("Kafka producer initialized for n8n routes")
    except Exception as e:
        logger.warning(f"Failed to initialize Kafka producer: {e}")
        logger.warning("n8n routes will not function without Kafka")

    yield

    # Cleanup
    if kafka_producer:
        await kafka_producer.stop()
        logger.info("Kafka producer stopped")

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
