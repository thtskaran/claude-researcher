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
            await self._ensure_schema()

    async def _ensure_schema(self) -> None:
        """Ensure core schema exists (shared with CLI database)."""
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                slug TEXT,
                time_limit_minutes INTEGER DEFAULT 60,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active',
                total_findings INTEGER DEFAULT 0,
                total_searches INTEGER DEFAULT 0,
                depth_reached INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                parent_topic_id INTEGER,
                depth INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                assigned_at TEXT,
                completed_at TEXT,
                findings_count INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
            );

            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic_id INTEGER,
                content TEXT NOT NULL,
                finding_type TEXT NOT NULL,
                source_url TEXT,
                confidence REAL DEFAULT 0.8,
                search_query TEXT,
                created_at TEXT NOT NULL,
                validated_by_manager INTEGER DEFAULT 0,
                manager_notes TEXT,
                verification_status TEXT,
                verification_method TEXT,
                kg_support_score REAL DEFAULT 0.0,
                original_confidence REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            );

            CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                finding_id INTEGER NOT NULL,
                original_confidence REAL,
                verified_confidence REAL,
                verification_status TEXT,
                verification_method TEXT,
                consistency_score REAL DEFAULT 0.0,
                kg_support_score REAL DEFAULT 0.0,
                kg_entity_matches INTEGER DEFAULT 0,
                kg_supporting_relations INTEGER DEFAULT 0,
                critic_iterations INTEGER DEFAULT 0,
                corrections_made TEXT,
                external_verification_used INTEGER DEFAULT 0,
                contradictions TEXT,
                verification_time_ms REAL,
                created_at TEXT NOT NULL,
                error TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (finding_id) REFERENCES findings(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS credibility_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                finding_id TEXT,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                final_score REAL NOT NULL,
                domain_authority_score REAL,
                recency_score REAL,
                source_type_score REAL,
                https_score REAL,
                path_depth_score REAL,
                credibility_label TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                decision_outcome TEXT NOT NULL,
                reasoning TEXT,
                inputs_json TEXT,
                metrics_json TEXT,
                iteration INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                agent TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
            CREATE INDEX IF NOT EXISTS idx_topics_session ON topics(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_verification_session ON verification_results(session_id);
            CREATE INDEX IF NOT EXISTS idx_verification_finding ON verification_results(finding_id);
            CREATE INDEX IF NOT EXISTS idx_credibility_session ON credibility_audit(session_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_session ON agent_decisions(session_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_type ON agent_decisions(decision_type);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """
        )
        await self._connection.commit()

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
            await self._connection.execute(
                "DELETE FROM events WHERE session_id = ?", (session_id,)
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

    async def save_event(
        self,
        session_id: str,
        event_type: str,
        agent: str,
        timestamp: str,
        data_json: str,
        created_at: str,
    ) -> None:
        """Persist an emitted event for later playback."""
        try:
            await self._connection.execute(
                """
                INSERT INTO events (session_id, event_type, agent, timestamp, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, event_type, agent, timestamp, data_json, created_at),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def list_events(self, session_id: str, limit: int = 500, order: str = "desc"):
        """List persisted events for a session."""
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        cursor = await self._connection.execute(
            f"""
            SELECT session_id, event_type, agent, timestamp, data_json
            FROM events
            WHERE session_id = ?
            ORDER BY timestamp {order_sql}
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return rows


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
