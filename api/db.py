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

    async def list_findings(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
        order: str = "desc",
        search: Optional[str] = None,
        finding_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
    ):
        """List findings with optional filters."""
        order_sql = "DESC" if order.lower() == "desc" else "ASC"
        params: list = [session_id]
        where = ["session_id = ?"]

        if search:
            where.append("(content LIKE ? OR source_url LIKE ? OR search_query LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])

        if finding_type and finding_type.lower() != "all":
            where.append("LOWER(finding_type) = ?")
            params.append(finding_type.lower())

        if min_confidence is not None:
            where.append("confidence >= ?")
            params.append(min_confidence)

        if max_confidence is not None:
            where.append("confidence <= ?")
            params.append(max_confidence)

        where_sql = " AND ".join(where)

        cursor = await self._connection.execute(
            f"""
            SELECT
                id,
                session_id,
                content,
                finding_type,
                source_url,
                confidence,
                search_query,
                created_at,
                verification_status,
                verification_method,
                kg_support_score
            FROM findings
            WHERE {where_sql}
            ORDER BY created_at {order_sql}
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "content": row["content"],
                "finding_type": row["finding_type"],
                "source_url": row["source_url"],
                "confidence": row["confidence"],
                "search_query": row["search_query"],
                "created_at": row["created_at"],
                "verification_status": row["verification_status"],
                "verification_method": row["verification_method"],
                "kg_support_score": row["kg_support_score"],
            }
            for row in rows
        ]

    async def list_sources(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ):
        """List source index for a session."""
        cursor = await self._connection.execute(
            """
            WITH source_stats AS (
                SELECT
                    source_url,
                    COUNT(*) as findings_count,
                    AVG(confidence) as avg_confidence,
                    MAX(created_at) as last_seen
                FROM findings
                WHERE session_id = ?
                  AND source_url IS NOT NULL
                  AND source_url != ''
                GROUP BY source_url
            ),
            latest_audit AS (
                SELECT url, MAX(created_at) as max_created_at
                FROM credibility_audit
                WHERE session_id = ?
                GROUP BY url
            )
            SELECT
                s.source_url,
                s.findings_count,
                s.avg_confidence,
                s.last_seen,
                ca.domain,
                ca.final_score,
                ca.credibility_label
            FROM source_stats s
            LEFT JOIN latest_audit la ON la.url = s.source_url
            LEFT JOIN credibility_audit ca
              ON ca.url = la.url
             AND ca.created_at = la.max_created_at
             AND ca.session_id = ?
            ORDER BY s.findings_count DESC, s.avg_confidence DESC
            LIMIT ? OFFSET ?
            """,
            (session_id, session_id, session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "source_url": row["source_url"],
                "findings_count": row["findings_count"],
                "avg_confidence": row["avg_confidence"],
                "last_seen": row["last_seen"],
                "domain": row["domain"],
                "final_score": row["final_score"],
                "credibility_label": row["credibility_label"],
            }
            for row in rows
        ]

    async def get_session_slug(self, session_id: str) -> Optional[str]:
        """Get the slug for a session ID."""
        cursor = await self._connection.execute(
            "SELECT slug FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return row["slug"]


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
