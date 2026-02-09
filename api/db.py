"""
Database service for API.

Direct SQLite wrapper to avoid import issues with src/ modules.
"""
import aiosqlite
import secrets
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class SessionRecord(BaseModel):
    """Session record from database."""
    id: str
    goal: str
    slug: str
    time_limit_minutes: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str = "active"
    total_findings: int = 0
    total_searches: int = 0
    depth_reached: int = 0


class APIDatabase:
    """Database wrapper for API endpoints."""

    def __init__(self, db_path: str = "research.db"):
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Connect to database."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row

    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def list_sessions(self, limit: int = 100) -> List[SessionRecord]:
        """List all research sessions, most recent first."""
        cursor = await self._connection.execute(
            """
            SELECT * FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = await cursor.fetchall()

        sessions = []
        for row in rows:
            sessions.append(SessionRecord(
                id=row["id"],
                goal=row["goal"],
                slug=row["slug"],
                time_limit_minutes=row["time_limit_minutes"],
                started_at=datetime.fromisoformat(row["started_at"]),
                ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                status=row["status"],
                total_findings=row["total_findings"],
                total_searches=row["total_searches"],
                depth_reached=row["depth_reached"],
            ))

        return sessions

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get a specific session."""
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return SessionRecord(
            id=row["id"],
            goal=row["goal"],
            slug=row["slug"],
            time_limit_minutes=row["time_limit_minutes"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            status=row["status"],
            total_findings=row["total_findings"],
            total_searches=row["total_searches"],
            depth_reached=row["depth_reached"],
        )

    async def create_session(self, goal: str, time_limit: int = 60) -> SessionRecord:
        """Create a new research session."""
        session_id = secrets.token_hex(4)[:7]  # 7-char hex like the main code

        # Simple slug generation
        slug = goal.lower()[:50].replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")

        now = datetime.now()

        try:
            await self._connection.execute(
                """
                INSERT INTO sessions (id, goal, slug, time_limit_minutes, started_at, status)
                VALUES (?, ?, ?, ?, ?, 'created')
                """,
                (session_id, goal, slug, time_limit, now.isoformat()),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

        return SessionRecord(
            id=session_id,
            goal=goal,
            slug=slug,
            time_limit_minutes=time_limit,
            started_at=now,
            status="created",
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        try:
            # Delete related data first (foreign key constraints)
            await self._connection.execute(
                "DELETE FROM findings WHERE session_id = ?", (session_id,)
            )
            await self._connection.execute(
                "DELETE FROM topics WHERE session_id = ?", (session_id,)
            )
            await self._connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            await self._connection.execute(
                "DELETE FROM verification_results WHERE session_id = ?", (session_id,)
            )
            await self._connection.execute(
                "DELETE FROM credibility_audit WHERE session_id = ?", (session_id,)
            )
            await self._connection.execute(
                "DELETE FROM agent_decisions WHERE session_id = ?", (session_id,)
            )

            # Delete the session
            cursor = await self._connection.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            await self._connection.commit()

            return cursor.rowcount > 0
        except Exception:
            await self._connection.rollback()
            raise


# Global database instance
_db_instance: Optional[APIDatabase] = None


async def get_db() -> APIDatabase:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = APIDatabase()
        await _db_instance.connect()
    return _db_instance


async def close_db():
    """Close the global database instance."""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
