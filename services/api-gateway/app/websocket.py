"""WebSocket server for real-time updates.

Manages WebSocket connections and broadcasts events from Redis to connected clients.
"""

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect
from shared import subscribe_to_channel

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections per project."""

    def __init__(self):
        """Initialize connection manager with empty project connections."""
        # Map of project_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of project_id -> asyncio.Task for Redis subscription
        self.subscription_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, project_id: str) -> None:
        """Accept WebSocket connection and subscribe to project events.

        Args:
            websocket: WebSocket connection instance.
            project_id: Project identifier for event filtering.
        """
        await websocket.accept()

        # Add to active connections
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()

        self.active_connections[project_id].add(websocket)
        logger.info(
            f"WebSocket connected for project '{project_id}' "
            f"({len(self.active_connections[project_id])} total)"
        )

        # Start Redis subscription if this is the first connection for this project
        if len(self.active_connections[project_id]) == 1:
            task = asyncio.create_task(self._subscribe_to_redis(project_id))
            self.subscription_tasks[project_id] = task
            logger.info(f"Started Redis subscription for project '{project_id}'")

    def disconnect(self, websocket: WebSocket, project_id: str) -> None:
        """Remove WebSocket connection and cleanup if no more connections.

        Args:
            websocket: WebSocket connection instance.
            project_id: Project identifier.
        """
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)
            remaining = len(self.active_connections[project_id])

            logger.info(
                f"WebSocket disconnected for project '{project_id}' "
                f"({remaining} remaining)"
            )

            # Stop Redis subscription if no more connections
            if remaining == 0:
                if project_id in self.subscription_tasks:
                    task = self.subscription_tasks[project_id]
                    task.cancel()
                    del self.subscription_tasks[project_id]
                    logger.info(f"Stopped Redis subscription for project '{project_id}'")

                del self.active_connections[project_id]

    async def broadcast(self, project_id: str, message: dict) -> None:
        """Broadcast message to all WebSocket connections for a project.

        Args:
            project_id: Project identifier.
            message: Message dictionary (will be JSON-serialized).
        """
        if project_id not in self.active_connections:
            return

        # Get copy of connections to avoid modification during iteration
        connections = list(self.active_connections[project_id])
        message_json = json.dumps(message)

        # Send to all connections, removing stale ones
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                self.disconnect(websocket, project_id)

    async def _subscribe_to_redis(self, project_id: str) -> None:
        """Subscribe to Redis channel and broadcast events to WebSocket clients.

        Args:
            project_id: Project identifier for channel subscription.
        """
        channel = f"project:{project_id}:events"

        try:
            async for message in subscribe_to_channel(channel):
                await self.broadcast(project_id, message)

        except asyncio.CancelledError:
            logger.info(f"Redis subscription cancelled for project '{project_id}'")
            raise
        except Exception as e:
            logger.error(
                f"Error in Redis subscription for project '{project_id}': {e}"
            )


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, project_id: str) -> None:
    """WebSocket endpoint for project-specific real-time updates.

    Args:
        websocket: WebSocket connection instance.
        project_id: Project identifier from path parameter.
    """
    await manager.connect(websocket, project_id)

    try:
        # Keep connection alive and listen for client messages
        while True:
            # Receive messages from client (e.g., ping/pong, subscribe requests)
            data = await websocket.receive_text()

            # Echo acknowledgment
            await websocket.send_text(
                json.dumps({"type": "ack", "message": "Message received"})
            )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from project '{project_id}'")
    except Exception as e:
        logger.error(f"WebSocket error for project '{project_id}': {e}")
    finally:
        manager.disconnect(websocket, project_id)
