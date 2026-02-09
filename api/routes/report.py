"""
Report endpoints for a session.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.db import get_db

router = APIRouter(prefix="/api/sessions", tags=["report"])


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    """Return the report markdown for a session, if available."""
    db = await get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    slug = session.slug or "research"
    output_dir = Path("output") / f"{slug}_{session_id}"
    report_path = output_dir / "report.md"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "session_id": session_id,
        "report": report_path.read_text(),
        "path": str(report_path),
    }
