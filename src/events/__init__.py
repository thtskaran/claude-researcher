"""
Event emission helpers for agents to send real-time updates to WebSocket clients.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add api directory to path for importing
api_path = str(Path(__file__).parent.parent.parent / "api")
if api_path not in sys.path:
    sys.path.insert(0, api_path)


def _should_proxy_event(local_subscribers: int | None) -> bool:
    """Decide whether to forward events to the API server."""
    if os.environ.get("CLAUDE_RESEARCHER_DISABLE_EVENT_PROXY") == "1":
        return False
    if os.environ.get("CLAUDE_RESEARCHER_IN_API") == "1":
        return False
    if local_subscribers is not None and local_subscribers > 0:
        return False
    return True


async def _emit_remote_event(
    session_id: str,
    event_type: str,
    agent: str,
    data: dict[str, Any],
) -> None:
    """Forward event to the API server for WebSocket broadcasting."""
    base_url = os.environ.get("CLAUDE_RESEARCHER_API_URL", "http://localhost:8080").rstrip("/")
    if not base_url:
        return

    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{base_url}/api/events/emit",
                json={
                    "session_id": session_id,
                    "event_type": event_type,
                    "agent": agent,
                    "data": data,
                },
            )
    except Exception:
        # Silent failure - don't interrupt research if API is unavailable
        return


async def emit_agent_event(
    session_id: str,
    event_type: str,
    agent: str,
    data: dict[str, Any],
) -> None:
    """
    Emit an event from an agent to all WebSocket subscribers.

    This is fire-and-forget - errors are logged but don't interrupt research.

    Args:
        session_id: Research session ID
        event_type: Event type (thinking, action, finding, synthesis, error)
        agent: Agent name (director, manager, intern)
        data: Event data dict

    Examples:
        await emit_agent_event(
            session_id="abc123",
            event_type="thinking",
            agent="director",
            data={"message": "Analyzing research goal..."}
        )
    """
    local_subscribers: int | None = None
    try:
        # Import here to avoid circular dependencies
        from api.events import emit_event, get_event_emitter

        # Fire and forget - don't block research if WebSocket fails
        try:
            asyncio.create_task(emit_event(session_id, event_type, agent, data))
        except RuntimeError:
            # Fallback if no running loop (should be rare)
            await emit_event(session_id, event_type, agent, data)

        local_subscribers = get_event_emitter().get_subscriber_count(session_id)
    except Exception as e:
        # Log errors but don't interrupt research
        print(f"⚠️  Event emission error: {e}")
        import traceback
        traceback.print_exc()

    # Proxy to API server if we have no local subscribers (CLI -> UI bridge)
    if _should_proxy_event(local_subscribers):
        try:
            asyncio.create_task(
                _emit_remote_event(session_id, event_type, agent, data)
            )
        except RuntimeError:
            await _emit_remote_event(session_id, event_type, agent, data)


async def emit_thinking(session_id: str, agent: str, thought: str) -> None:
    """Emit a thinking event."""
    await emit_agent_event(
        session_id=session_id,
        event_type="thinking",
        agent=agent,
        data={"thought": thought}
    )


async def emit_action(
    session_id: str,
    agent: str,
    action: str,
    details: dict[str, Any] | None = None
) -> None:
    """Emit an action event."""
    data = {"action": action}
    if details:
        data.update(details)

    await emit_agent_event(
        session_id=session_id,
        event_type="action",
        agent=agent,
        data=data
    )


async def emit_finding(
    session_id: str,
    agent: str,
    content: str,
    source: str | None = None,
    confidence: float | None = None
) -> None:
    """Emit a finding discovery event."""
    data = {"content": content}
    if source:
        data["source"] = source
    if confidence is not None:
        data["confidence"] = confidence

    await emit_agent_event(
        session_id=session_id,
        event_type="finding",
        agent=agent,
        data=data
    )


async def emit_synthesis(
    session_id: str,
    agent: str,
    message: str,
    progress: int | None = None
) -> None:
    """Emit a synthesis/report generation event."""
    data = {"message": message}
    if progress is not None:
        data["progress"] = progress

    await emit_agent_event(
        session_id=session_id,
        event_type="synthesis",
        agent=agent,
        data=data
    )


async def emit_error(
    session_id: str,
    agent: str,
    error: str,
    recoverable: bool = True
) -> None:
    """Emit an error event."""
    await emit_agent_event(
        session_id=session_id,
        event_type="error",
        agent=agent,
        data={"error": error, "recoverable": recoverable}
    )
