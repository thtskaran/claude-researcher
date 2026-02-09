"""
Research session routes.

Handles CRUD operations for research sessions.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from datetime import datetime

from api.models import ResearchSessionCreate, ResearchSessionResponse
from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/", response_model=ResearchSessionResponse, status_code=201)
async def create_session(session: ResearchSessionCreate):
    """
    Create a new research session in the database.

    This creates a session record that can be used for CLI or UI research.
    """
    db = await get_db()
    db_session = await db.create_session(session.goal, session.time_limit)

    return ResearchSessionResponse(
        session_id=db_session.id,
        goal=db_session.goal,
        time_limit=db_session.time_limit_minutes,
        status=db_session.status,
        created_at=db_session.started_at,
        completed_at=db_session.ended_at
    )


@router.get("/", response_model=List[ResearchSessionResponse])
async def list_sessions(limit: int = 100):
    """List all research sessions from the database."""
    db = await get_db()
    sessions = await db.list_sessions(limit)

    return [
        ResearchSessionResponse(
            session_id=s.id,
            goal=s.goal,
            time_limit=s.time_limit_minutes,
            status=s.status,
            created_at=s.started_at,
            completed_at=s.ended_at
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=ResearchSessionResponse)
async def get_session(session_id: str):
    """Get a specific research session from the database."""
    db = await get_db()
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return ResearchSessionResponse(
        session_id=session.id,
        goal=session.goal,
        time_limit=session.time_limit_minutes,
        status=session.status,
        created_at=session.started_at,
        completed_at=session.ended_at
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a research session from the database."""
    db = await get_db()
    deleted = await db.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
