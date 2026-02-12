"""
Findings and sources endpoints for a session.
"""

from fastapi import APIRouter, HTTPException, Query

from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["findings"])


@router.get("/{session_id}/findings")
async def list_findings(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    order: str = "desc",
    search: str | None = None,
    finding_type: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
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
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
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
