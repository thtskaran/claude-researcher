"""
Findings and sources endpoints for a session.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["findings"])


@router.get("/{session_id}/findings")
async def list_findings(
    session_id: str,
    limit: int = 200,
    offset: int = 0,
    order: str = "desc",
    search: Optional[str] = None,
    finding_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
):
    """List findings for a session with optional filters."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rows = await db.list_findings(
        session_id=session_id,
        limit=limit,
        offset=offset,
        order=order,
        search=search,
        finding_type=finding_type,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    return rows


@router.get("/{session_id}/sources")
async def list_sources(
    session_id: str,
    limit: int = 200,
    offset: int = 0,
):
    """List source index for a session."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rows = await db.list_sources(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )

    return rows
