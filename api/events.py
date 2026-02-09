"""
Event emitter system for broadcasting agent events to WebSocket clients.

Agents emit events → EventEmitter → WebSocket → UI
"""
import asyncio
import json
from typing import Dict, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class AgentEvent:
    """Agent event data structure."""

    def __init__(
        self,
        session_id: str,
        event_type: str,
        agent: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.session_id = session_id
        self.event_type = event_type
        self.agent = agent
        self.data = data
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "event_type": self.event_type,
            "agent": self.agent,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class EventEmitter:
    """
    Singleton event emitter for broadcasting agent events.

    Usage:
        emitter = EventEmitter()
        await emitter.emit(session_id, "thinking", "director", {...})
    """

    _instance: Optional['EventEmitter'] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Session ID -> Set of WebSocket connections
        self._subscribers: Dict[str, Set[WebSocket]] = {}
        self._subscribers_lock = asyncio.Lock()
        self._initialized = True

        logger.info("EventEmitter initialized")

    async def subscribe(self, session_id: str, websocket: WebSocket):
        """Subscribe a WebSocket to events for a specific session."""
        async with self._subscribers_lock:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = set()
            self._subscribers[session_id].add(websocket)
            logger.info(f"WebSocket subscribed to session {session_id}")

    async def unsubscribe(self, session_id: str, websocket: WebSocket):
        """Unsubscribe a WebSocket from a session."""
        async with self._subscribers_lock:
            if session_id in self._subscribers:
                self._subscribers[session_id].discard(websocket)
                if not self._subscribers[session_id]:
                    del self._subscribers[session_id]
                logger.info(f"WebSocket unsubscribed from session {session_id}")

    async def emit(
        self,
        session_id: str,
        event_type: str,
        agent: str,
        data: Dict[str, Any]
    ):
        """
        Emit an event to all subscribers of a session.

        Args:
            session_id: Session ID
            event_type: Event type (thinking, action, finding, synthesis, error)
            agent: Agent name (director, manager, intern)
            data: Event data
        """
        event = AgentEvent(session_id, event_type, agent, data)

        async with self._subscribers_lock:
            if session_id not in self._subscribers:
                # No subscribers, but log the event
                logger.debug(f"Event emitted but no subscribers for {session_id}: {event_type}")
                return

            subscribers = list(self._subscribers[session_id])

        # Send to all subscribers (outside lock to avoid blocking)
        dead_connections = []
        for websocket in subscribers:
            try:
                await websocket.send_json(event.to_dict())
                logger.debug(f"Sent {event_type} event to WebSocket for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.append(websocket)

        # Clean up dead connections
        if dead_connections:
            async with self._subscribers_lock:
                for ws in dead_connections:
                    self._subscribers[session_id].discard(ws)

    def get_subscriber_count(self, session_id: str) -> int:
        """Get number of active subscribers for a session."""
        return len(self._subscribers.get(session_id, set()))

    def get_all_sessions(self) -> list[str]:
        """Get all sessions with active subscribers."""
        return list(self._subscribers.keys())


# Global singleton instance
_emitter = EventEmitter()


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    return _emitter


# Convenience function for emitting events
async def emit_event(
    session_id: str,
    event_type: str,
    agent: str,
    data: Dict[str, Any]
):
    """Emit an event to all subscribers."""
    emitter = get_event_emitter()
    await emitter.emit(session_id, event_type, agent, data)
