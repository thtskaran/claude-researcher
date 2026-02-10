"""
Verification pipeline API routes.

Serves verification results and stats for the CoVe Pipeline page.
"""
from fastapi import APIRouter, HTTPException

from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["verification"])


@router.get("/{session_id}/verification/results")
async def list_verification_results(
    session_id: str,
    limit: int = 200,
    offset: int = 0,
):
    """List verification results with joined finding content."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return await db.list_verification_results(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )


@router.get("/{session_id}/verification/stats")
async def get_verification_stats(session_id: str):
    """Get aggregate verification statistics."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return await db.get_verification_stats(session_id)
