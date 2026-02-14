"""
Database service for API.

Direct SQLite wrapper to avoid import issues with src/ modules.
"""
import secrets
from datetime import datetime
from pathlib import Path

import aiosqlite
from pydantic import BaseModel


class SessionRecord(BaseModel):
    """Session record from database."""
    id: str
    goal: str
    slug: str
    max_iterations: int = 5
    time_limit: int = 0  # Kept for backward compat display
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "active"
    total_findings: int = 0
    total_searches: int = 0
    depth_reached: int = 0
    elapsed_seconds: float = 0.0
    paused_at: datetime | None = None
    iteration_count: int = 0
    phase: str = "init"


def _row_to_session(row) -> SessionRecord:
    """Convert a DB row to SessionRecord, handling old schemas without max_iterations."""
    keys = row.keys() if hasattr(row, "keys") else []
    return SessionRecord(
        id=row["id"],
        goal=row["goal"],
        slug=row["slug"],
        max_iterations=row["max_iterations"] if "max_iterations" in keys else 5,
        time_limit=row["time_limit_minutes"] if "time_limit_minutes" in keys else 0,
        started_at=datetime.fromisoformat(row["started_at"]),
        ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
        status=row["status"],
        total_findings=row["total_findings"],
        total_searches=row["total_searches"],
        depth_reached=row["depth_reached"],
        elapsed_seconds=row["elapsed_seconds"] or 0.0,
        paused_at=datetime.fromisoformat(row["paused_at"]) if row["paused_at"] else None,
        iteration_count=row["iteration_count"] or 0,
        phase=row["phase"] or "init",
    )


class APIDatabase:
    """Database wrapper for API endpoints."""

    def __init__(self, db_path: str = "research.db"):
        self.db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def connect(self):
        """Connect to database."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
            await self._connection.execute("PRAGMA busy_timeout=5000")
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._ensure_schema()

    async def _ensure_schema(self) -> None:
        """Ensure core schema exists (shared with CLI database)."""
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                slug TEXT,
                max_iterations INTEGER DEFAULT 5,
                time_limit_minutes INTEGER DEFAULT 0,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active',
                total_findings INTEGER DEFAULT 0,
                total_searches INTEGER DEFAULT 0,
                depth_reached INTEGER DEFAULT 0,
                elapsed_seconds REAL DEFAULT 0.0,
                paused_at TEXT,
                iteration_count INTEGER DEFAULT 0,
                phase TEXT DEFAULT 'init'
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
                consistency_score REAL,
                kg_support_score REAL,
                kg_entity_matches INTEGER,
                kg_supporting_relations INTEGER,
                critic_iterations INTEGER,
                corrections_made TEXT,
                questions_asked TEXT,
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

        # Migration: add max_iterations column for existing databases
        try:
            await self._connection.execute(
                "ALTER TABLE sessions ADD COLUMN max_iterations INTEGER DEFAULT 5"
            )
            await self._connection.commit()
        except Exception:
            pass  # Column already exists

    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def list_sessions(self, limit: int = 100) -> list[SessionRecord]:
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
            sessions.append(_row_to_session(row))

        return sessions

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Get a specific session."""
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return _row_to_session(row)

    async def create_session(self, goal: str, max_iterations: int = 5) -> SessionRecord:
        """Create a new research session."""
        session_id = secrets.token_hex(4)[:7]  # 7-char hex like the main code

        # Simple slug generation
        slug = goal.lower()[:50].replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")

        now = datetime.now()

        try:
            await self._connection.execute(
                """
                INSERT INTO sessions (id, goal, slug, max_iterations, started_at, status)
                VALUES (?, ?, ?, ?, ?, 'created')
                """,
                (session_id, goal, slug, max_iterations, now.isoformat()),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

        return SessionRecord(
            id=session_id,
            goal=goal,
            slug=slug,
            max_iterations=max_iterations,
            started_at=now,
            status="created",
        )

    async def update_session_status(
        self, session_id: str, status: str, ended_at: datetime | None = None
    ) -> bool:
        """Update session status and optionally set ended_at."""
        try:
            if ended_at:
                cursor = await self._connection.execute(
                    "UPDATE sessions SET status = ?, ended_at = ? WHERE id = ?",
                    (status, ended_at.isoformat(), session_id),
                )
            else:
                cursor = await self._connection.execute(
                    "UPDATE sessions SET status = ? WHERE id = ?",
                    (status, session_id),
                )
            await self._connection.commit()
            return cursor.rowcount > 0
        except Exception:
            await self._connection.rollback()
            raise

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
        search: str | None = None,
        finding_type: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
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
        from src.knowledge.credibility import CredibilityScorer

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

        scorer = CredibilityScorer()
        results = []
        for row in rows:
            domain = row["domain"]
            final_score = row["final_score"]
            credibility_label = row["credibility_label"]

            # Compute credibility on-the-fly if audit data is missing
            if final_score is None:
                score_result = scorer.score_source(row["source_url"])
                domain = score_result.domain
                final_score = score_result.score
                credibility_label = scorer.get_credibility_label(score_result.score)

            results.append({
                "source_url": row["source_url"],
                "findings_count": row["findings_count"],
                "avg_confidence": row["avg_confidence"],
                "last_seen": row["last_seen"],
                "domain": domain,
                "final_score": final_score,
                "credibility_label": credibility_label,
            })
        return results

    async def list_verification_results(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ):
        """List verification results joined with finding content."""
        cursor = await self._connection.execute(
            """
            SELECT
                vr.id,
                vr.session_id,
                vr.finding_id,
                vr.original_confidence,
                vr.verified_confidence,
                vr.verification_status,
                vr.verification_method,
                vr.consistency_score,
                vr.kg_support_score,
                vr.kg_entity_matches,
                vr.kg_supporting_relations,
                vr.critic_iterations,
                vr.corrections_made,
                vr.questions_asked,
                vr.external_verification_used,
                vr.contradictions,
                vr.verification_time_ms,
                vr.created_at,
                vr.error,
                f.content AS finding_content,
                f.finding_type,
                f.source_url,
                f.confidence AS current_confidence
            FROM verification_results vr
            LEFT JOIN findings f ON f.id = vr.finding_id AND f.session_id = vr.session_id
            WHERE vr.session_id = ?
            ORDER BY vr.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        import json as _json
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "finding_id": row["finding_id"],
                "original_confidence": row["original_confidence"],
                "verified_confidence": row["verified_confidence"],
                "verification_status": row["verification_status"],
                "verification_method": row["verification_method"],
                "consistency_score": row["consistency_score"],
                "kg_support_score": row["kg_support_score"],
                "kg_entity_matches": row["kg_entity_matches"],
                "kg_supporting_relations": row["kg_supporting_relations"],
                "critic_iterations": row["critic_iterations"],
                "corrections_made": _json.loads(row["corrections_made"]) if row["corrections_made"] else None,
                "questions_asked": _json.loads(row["questions_asked"]) if row["questions_asked"] else None,
                "external_verification_used": bool(row["external_verification_used"]),
                "contradictions": _json.loads(row["contradictions"]) if row["contradictions"] else None,
                "verification_time_ms": row["verification_time_ms"],
                "created_at": row["created_at"],
                "error": row["error"],
                "finding_content": row["finding_content"],
                "finding_type": row["finding_type"],
                "source_url": row["source_url"],
                "current_confidence": row["current_confidence"],
            })
        return results

    async def get_verification_stats(self, session_id: str) -> dict:
        """Get aggregate verification stats for a session."""
        cursor = await self._connection.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                SUM(CASE WHEN verification_status = 'flagged' THEN 1 ELSE 0 END) as flagged,
                SUM(CASE WHEN verification_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN verification_status = 'pending' THEN 1 ELSE 0 END) as pending,
                AVG(verified_confidence) as avg_confidence,
                AVG(verification_time_ms) as avg_time_ms
            FROM verification_results
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row or row["total"] == 0:
            return {
                "total": 0, "verified": 0, "flagged": 0,
                "rejected": 0, "pending": 0,
                "avg_confidence": None, "avg_time_ms": None,
            }
        return {
            "total": row["total"],
            "verified": row["verified"],
            "flagged": row["flagged"],
            "rejected": row["rejected"],
            "pending": row["pending"],
            "avg_confidence": row["avg_confidence"],
            "avg_time_ms": row["avg_time_ms"],
        }

    async def list_agent_decisions(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ):
        """List agent decision logs for a session."""
        cursor = await self._connection.execute(
            """
            SELECT
                id, session_id, agent_role, decision_type,
                decision_outcome, reasoning, inputs_json,
                metrics_json, iteration, created_at
            FROM agent_decisions
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        import json as _json
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "agent_role": row["agent_role"],
                "decision_type": row["decision_type"],
                "decision_outcome": row["decision_outcome"],
                "reasoning": row["reasoning"],
                "inputs": _json.loads(row["inputs_json"]) if row["inputs_json"] else None,
                "metrics": _json.loads(row["metrics_json"]) if row["metrics_json"] else None,
                "iteration": row["iteration"],
                "created_at": row["created_at"],
            })
        return results

    async def list_topics(
        self,
        session_id: str,
        limit: int = 200,
    ):
        """List topic tree for a session."""
        cursor = await self._connection.execute(
            """
            SELECT
                id, session_id, topic, parent_topic_id, depth,
                status, priority, assigned_at, completed_at, findings_count
            FROM topics
            WHERE session_id = ?
            ORDER BY depth ASC, priority DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "topic": row["topic"],
                "parent_topic_id": row["parent_topic_id"],
                "depth": row["depth"],
                "status": row["status"],
                "priority": row["priority"],
                "assigned_at": row["assigned_at"],
                "completed_at": row["completed_at"],
                "findings_count": row["findings_count"],
            }
            for row in rows
        ]

    async def get_session_slug(self, session_id: str) -> str | None:
        """Get the slug for a session ID."""
        cursor = await self._connection.execute(
            "SELECT slug FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return row["slug"]

    async def get_session_stats(self, session_id: str) -> dict:
        """Get aggregate stats for a session (findings, sources, topics)."""
        findings_cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM findings WHERE session_id = ?",
            (session_id,),
        )
        findings_row = await findings_cursor.fetchone()

        sources_cursor = await self._connection.execute(
            """
            SELECT COUNT(DISTINCT source_url) as count
            FROM findings
            WHERE session_id = ? AND source_url IS NOT NULL AND source_url != ''
            """,
            (session_id,),
        )
        sources_row = await sources_cursor.fetchone()

        topics_cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM topics WHERE session_id = ?",
            (session_id,),
        )
        topics_row = await topics_cursor.fetchone()

        return {
            "findings": findings_row["count"] if findings_row else 0,
            "sources": sources_row["count"] if sources_row else 0,
            "topics": topics_row["count"] if topics_row else 0,
        }


# Global database instance
_db_instance: APIDatabase | None = None


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
