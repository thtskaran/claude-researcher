"""
Event emission routes.

Allows external processes (e.g., CLI) to forward agent events to the API
so connected WebSocket clients can receive them in real time.
"""
from typing import Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.events import emit_event, get_event_emitter

router = APIRouter(prefix="/api/events", tags=["events"])


class EventEmitRequest(BaseModel):
    """Request payload for emitting an agent event."""
    session_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    agent: str = Field(..., min_length=1)
    data: Dict[str, Any] = Field(default_factory=dict)


@router.post("/emit")
async def emit_event_endpoint(payload: EventEmitRequest):
    """Emit an event to all WebSocket subscribers for a session."""
    await emit_event(
        session_id=payload.session_id,
        event_type=payload.event_type,
        agent=payload.agent,
        data=payload.data,
    )

    emitter = get_event_emitter()
    return {
        "status": "emitted",
        "session_id": payload.session_id,
        "event_type": payload.event_type,
        "subscribers": emitter.get_subscriber_count(payload.session_id),
    }
