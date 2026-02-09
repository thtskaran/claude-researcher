"""
Research session routes.

Handles CRUD operations for research sessions.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from datetime import datetime

from api.models import ResearchSessionCreate, ResearchSessionResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# In-memory storage for now (will connect to SQLite later)
sessions_db = {}


@router.post("/", response_model=ResearchSessionResponse, status_code=201)
async def create_session(session: ResearchSessionCreate):
    """
    Create a new research session.

    This will be connected to the actual ResearchHarness later.
    For now, it just creates a session record.
    """
    import secrets

    session_id = secrets.token_hex(4)  # 8-char hex

    session_data = {
        "session_id": session_id,
        "goal": session.goal,
        "time_limit": session.time_limit,
        "status": "created",
        "created_at": datetime.now(),
        "completed_at": None
    }

    sessions_db[session_id] = session_data

    return ResearchSessionResponse(**session_data)


@router.get("/", response_model=List[ResearchSessionResponse])
async def list_sessions():
    """List all research sessions."""
    return list(sessions_db.values())


@router.get("/{session_id}", response_model=ResearchSessionResponse)
async def get_session(session_id: str):
    """Get a specific research session."""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return ResearchSessionResponse(**sessions_db[session_id])


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a research session."""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    del sessions_db[session_id]
