"""SQLite database for persisting research state and findings."""

import asyncio
import json
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite

from ..logging_config import get_logger
from ..models.findings import (
    AgentMessage,
    AgentRole,
    Finding,
    FindingType,
    ResearchSession,
    ResearchTopic,
)

logger = get_logger(__name__)


def _generate_session_id() -> str:
    """Generate a unique 7-character hexadecimal session ID."""
    return secrets.token_hex(4)[:7]  # 7 hex chars


def _generate_slug(goal: str) -> str:
    """Generate a URL-friendly slug from the research goal.

    Examples:
        "What are the latest AI safety research?" -> "ai-safety-research"
        "History of quantum computing" -> "quantum-computing-history"
    """
    # Remove common question words and punctuation
    text = goal.lower()
    text = re.sub(r"\?+$", "", text)

    # Remove stop words
    stop_words = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "can",
        "latest",
        "current",
        "recent",
        "new",
        "best",
        "top",
    }
    words = text.split()
    words = [w for w in words if w not in stop_words]

    # Take first 4-5 meaningful words
    words = words[:5]

    # Clean and join
    slug = "-".join(words)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")

    # Limit length
    if len(slug) > 50:
        slug = slug[:50].rsplit("-", 1)[0]

    return slug or "research"


class _ConnectionPool:
    """Simple async SQLite connection pool with WAL mode."""

    def __init__(self, db_path: Path, size: int = 5):
        self._db_path = db_path
        self._size = size
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=size)
        self._initialized = False

    async def init(self):
        """Create pool connections with WAL mode."""
        for _ in range(self._size):
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            # Set busy_timeout BEFORE WAL so the journal mode switch can wait for locks
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("PRAGMA journal_mode=WAL")
            await self._pool.put(conn)
        self._initialized = True

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def close(self):
        """Close all pooled connections."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        self._initialized = False


class ResearchDatabase:
    """SQLite database manager for research persistence."""

    def __init__(self, db_path: str = "research.db"):
        self.db_path = Path(db_path)
        self._pool: _ConnectionPool | None = None

    async def connect(self) -> None:
        """Connect to the database and create tables if needed."""
        if self._pool is not None and self._pool._initialized:
            return  # Already connected
        logger.info("Database connecting: %s", self.db_path)
        self._pool = _ConnectionPool(self.db_path)
        await self._pool.init()
        async with self._pool.acquire() as conn:
            await self._create_tables(conn)

    async def close(self) -> None:
        """Close all database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _check_pool(self) -> _ConnectionPool:
        """Get the active connection pool.

        Raises RuntimeError if not connected.
        """
        if self._pool is None or not self._pool._initialized:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    async def _create_tables(self, conn: aiosqlite.Connection) -> None:
        """Create database tables if they don't exist."""
        await conn.executescript("""
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

            CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
            CREATE INDEX IF NOT EXISTS idx_topics_session ON topics(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_verification_session ON verification_results(session_id);
            CREATE INDEX IF NOT EXISTS idx_verification_finding ON verification_results(finding_id);

            -- Audit trail tables for explainability
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
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_credibility_session ON credibility_audit(session_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_session ON agent_decisions(session_id);
            CREATE INDEX IF NOT EXISTS idx_decisions_type ON agent_decisions(decision_type);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        """)
        await conn.commit()

        # Migration: add max_iterations column for existing databases
        try:
            await conn.execute(
                "ALTER TABLE sessions ADD COLUMN max_iterations INTEGER DEFAULT 5"
            )
            await conn.commit()
        except Exception as e:
            # [HARDENED] ERR-002: Only suppress expected "duplicate column" errors
            if "duplicate column" in str(e).lower():
                pass
            else:
                raise

    # Session methods
    async def create_session(self, goal: str, max_iterations: int = 5) -> ResearchSession:
        """Create a new research session with unique hex ID."""
        pool = self._check_pool()
        # [HARDENED] BUG-011: Retry on ID collision (up to 3 attempts)
        last_err: Exception | None = None
        for _attempt in range(3):
            session_id = _generate_session_id()
            slug = _generate_slug(goal)
            now = datetime.now().isoformat()
            async with pool.acquire() as conn:
                try:
                    await conn.execute(
                        """
                        INSERT INTO sessions (id, goal, slug, max_iterations, started_at, status)
                        VALUES (?, ?, ?, ?, ?, 'active')
                        """,
                        (session_id, goal, slug, max_iterations, now),
                    )
                    await conn.commit()
                    break
                except Exception as e:
                    await conn.rollback()
                    logger.error("DB error in create_session: %s", e, exc_info=True)
                    last_err = e
                    if "unique" in str(e).lower() or "constraint" in str(e).lower():
                        continue
                    raise
        else:
            raise last_err  # type: ignore[misc]
        logger.info("Session created: %s", session_id)
        return ResearchSession(
            id=session_id,
            goal=goal,
            slug=slug,
            max_iterations=max_iterations,
            started_at=datetime.fromisoformat(now),
            status="active",
        )

    async def get_session(self, session_id: str) -> ResearchSession | None:
        """Get a session by ID."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return ResearchSession(
            id=row["id"],
            goal=row["goal"],
            slug=row["slug"],
            max_iterations=row["max_iterations"] if "max_iterations" in row.keys() else 5,
            time_limit_minutes=row["time_limit_minutes"] if "time_limit_minutes" in row.keys() else 0,
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

    async def update_session(self, session: ResearchSession) -> None:
        """Update a session."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    UPDATE sessions SET
                        status = ?,
                        ended_at = ?,
                        total_findings = ?,
                        total_searches = ?,
                        depth_reached = ?,
                        elapsed_seconds = ?,
                        paused_at = ?,
                        iteration_count = ?,
                        phase = ?
                    WHERE id = ?
                    """,
                    (
                        session.status,
                        session.ended_at.isoformat() if session.ended_at else None,
                        session.total_findings,
                        session.total_searches,
                        session.depth_reached,
                        session.elapsed_seconds,
                        session.paused_at.isoformat() if session.paused_at else None,
                        session.iteration_count,
                        session.phase,
                        session.id,
                    ),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in update_session: %s", e, exc_info=True)
                raise

    # Topic methods
    async def create_topic(
        self,
        session_id: str,
        topic: str,
        parent_topic_id: int | None = None,
        depth: int = 0,
        priority: int = 5,
    ) -> ResearchTopic:
        """Create a new research topic."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO topics (session_id, topic, parent_topic_id, depth, priority)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, topic, parent_topic_id, depth, priority),
                )
                await conn.commit()
                lastrowid = cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in create_topic: %s", e, exc_info=True)
                raise
        return ResearchTopic(
            id=lastrowid,
            session_id=session_id,
            topic=topic,
            parent_topic_id=parent_topic_id,
            depth=depth,
            priority=priority,
        )

    async def get_pending_topics(self, session_id: str, limit: int = 10) -> list[ResearchTopic]:
        """Get pending topics ordered by priority."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM topics
                WHERE session_id = ? AND status = 'pending'
                ORDER BY priority DESC, depth ASC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = await cursor.fetchall()
        return [
            ResearchTopic(
                id=row["id"],
                session_id=row["session_id"],
                topic=row["topic"],
                parent_topic_id=row["parent_topic_id"],
                depth=row["depth"],
                status=row["status"],
                priority=row["priority"],
            )
            for row in rows
        ]

    async def update_topic_status(
        self, topic_id: int, status: str, findings_count: int = 0
    ) -> None:
        """Update topic status."""
        pool = self._check_pool()
        completed_at = datetime.now().isoformat() if status == "completed" else None
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    UPDATE topics SET status = ?, completed_at = ?, findings_count = ?
                    WHERE id = ?
                    """,
                    (status, completed_at, findings_count, topic_id),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in update_topic_status: %s", e, exc_info=True)
                raise

    async def get_all_topics(self, session_id: str) -> list[ResearchTopic]:
        """Get all topics for a session (for state reconstruction on resume)."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM topics
                WHERE session_id = ?
                ORDER BY priority DESC, depth ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [
            ResearchTopic(
                id=row["id"],
                session_id=row["session_id"],
                topic=row["topic"],
                parent_topic_id=row["parent_topic_id"],
                depth=row["depth"],
                status=row["status"],
                priority=row["priority"],
            )
            for row in rows
        ]

    async def reset_in_progress_topics(self, session_id: str) -> None:
        """Reset any in_progress topics back to pending (for resume after pause/crash)."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    UPDATE topics SET status = 'pending', assigned_at = NULL
                    WHERE session_id = ? AND status = 'in_progress'
                    """,
                    (session_id,),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in reset_in_progress_topics: %s", e, exc_info=True)
                raise

    # Finding methods
    async def save_finding(self, finding: Finding, topic_id: int | None = None) -> Finding:
        """Save a research finding."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO findings (
                        session_id, topic_id, content, finding_type, source_url,
                        confidence, search_query, created_at, validated_by_manager, manager_notes,
                        verification_status, verification_method, kg_support_score, original_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        finding.session_id,
                        topic_id,
                        finding.content,
                        finding.finding_type.value,
                        finding.source_url,
                        finding.confidence,
                        finding.search_query,
                        finding.created_at.isoformat(),
                        1 if finding.validated_by_manager else 0,
                        finding.manager_notes,
                        finding.verification_status,
                        finding.verification_method,
                        finding.kg_support_score,
                        finding.original_confidence,
                    ),
                )
                await conn.commit()
                finding.id = cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in save_finding: %s", e, exc_info=True)
                raise
        return finding

    async def update_finding_verification(
        self,
        finding_id: int,
        verification_status: str,
        verification_method: str,
        kg_support_score: float = 0.0,
        original_confidence: float | None = None,
        new_confidence: float | None = None,
    ) -> None:
        """Update a finding's verification status."""
        pool = self._check_pool()
        # [HARDENED] DRY-002: Consolidated duplicated SQL branches
        fields = [
            "verification_status = ?",
            "verification_method = ?",
            "kg_support_score = ?",
            "original_confidence = ?",
        ]
        params: list = [
            verification_status,
            verification_method,
            kg_support_score,
            original_confidence,
        ]
        if new_confidence is not None:
            fields.append("confidence = ?")
            params.append(new_confidence)
        params.append(finding_id)

        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    f"UPDATE findings SET {', '.join(fields)} WHERE id = ?",
                    tuple(params),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in update_finding_verification: %s", e, exc_info=True)
                raise

    async def save_verification_result(
        self,
        session_id: str,
        finding_id: int,
        result_dict: dict,
    ) -> int:
        """Save a verification result."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO verification_results (
                        session_id, finding_id, original_confidence, verified_confidence,
                        verification_status, verification_method, consistency_score,
                        kg_support_score, kg_entity_matches, kg_supporting_relations,
                        critic_iterations, corrections_made, questions_asked, external_verification_used,
                        contradictions, verification_time_ms, created_at, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        finding_id,
                        result_dict.get("original_confidence"),
                        result_dict.get("verified_confidence"),
                        result_dict.get("verification_status"),
                        result_dict.get("verification_method"),
                        result_dict.get("consistency_score"),
                        result_dict.get("kg_support_score"),
                        result_dict.get("kg_entity_matches"),
                        result_dict.get("kg_supporting_relations"),
                        result_dict.get("critic_iterations"),
                        json.dumps(result_dict.get("corrections_made", [])),
                        json.dumps(result_dict.get("questions_asked", [])),
                        1 if result_dict.get("external_verification_used") else 0,
                        json.dumps(result_dict.get("contradictions", [])),
                        result_dict.get("verification_time_ms", 0.0),
                        datetime.now().isoformat(),
                        result_dict.get("error"),
                    ),
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in save_verification_result: %s", e, exc_info=True)
                raise

    async def get_verification_stats(self, session_id: str) -> dict:
        """Get verification statistics for a session."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified,
                    SUM(CASE WHEN verification_status = 'flagged' THEN 1 ELSE 0 END) as flagged,
                    SUM(CASE WHEN verification_status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN verification_status = 'skipped' THEN 1 ELSE 0 END) as skipped,
                    AVG(original_confidence) as avg_original,
                    AVG(verified_confidence) as avg_calibrated,
                    AVG(verification_time_ms) as avg_time_ms
                FROM verification_results WHERE session_id = ?
                """,
                (session_id,),
            )
            row = await cursor.fetchone()

        if not row or row["total"] == 0:
            return {
                "total": 0,
                "verified": 0,
                "flagged": 0,
                "rejected": 0,
                "skipped": 0,
                "verification_rate": 0.0,
                "avg_original_confidence": 0.0,
                "avg_calibrated_confidence": 0.0,
                "avg_time_ms": 0.0,
            }

        total = row["total"]
        verified = row["verified"] or 0
        effective_total = total - (row["skipped"] or 0)
        verification_rate = verified / effective_total * 100 if effective_total > 0 else 0.0

        return {
            "total": total,
            "verified": verified,
            "flagged": row["flagged"] or 0,
            "rejected": row["rejected"] or 0,
            "skipped": row["skipped"] or 0,
            "verification_rate": verification_rate,
            "avg_original_confidence": row["avg_original"] or 0.0,
            "avg_calibrated_confidence": row["avg_calibrated"] or 0.0,
            "avg_time_ms": row["avg_time_ms"] or 0.0,
        }

    async def get_session_findings(self, session_id: str) -> list[Finding]:
        """Get all findings for a session."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                "SELECT * FROM findings WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [
            Finding(
                id=row["id"],
                session_id=row["session_id"],
                topic_id=row["topic_id"],
                content=row["content"],
                finding_type=FindingType(row["finding_type"]),
                source_url=row["source_url"],
                confidence=row["confidence"],
                search_query=row["search_query"],
                created_at=datetime.fromisoformat(row["created_at"]),
                validated_by_manager=bool(row["validated_by_manager"]),
                manager_notes=row["manager_notes"],
                verification_status=row["verification_status"],
                verification_method=row["verification_method"],
                kg_support_score=row["kg_support_score"],
                original_confidence=row["original_confidence"],
            )
            for row in rows
        ]

    # Message methods
    async def save_message(self, message: AgentMessage) -> AgentMessage:
        """Save an agent message."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO messages (
                        session_id, from_agent, to_agent, message_type, content, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.session_id,
                        message.from_agent.value,
                        message.to_agent.value,
                        message.message_type,
                        message.content,
                        json.dumps(message.metadata),
                        message.created_at.isoformat(),
                    ),
                )
                await conn.commit()
                message.id = cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in save_message: %s", e, exc_info=True)
                raise
        return message

    async def get_session_messages(
        self, session_id: str, agent_filter: AgentRole | None = None
    ) -> list[AgentMessage]:
        """Get messages for a session, optionally filtered by agent."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            if agent_filter:
                cursor = await conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ? AND (from_agent = ? OR to_agent = ?)
                    ORDER BY created_at
                    """,
                    (session_id, agent_filter.value, agent_filter.value),
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                    (session_id,),
                )
            rows = await cursor.fetchall()
        return [
            AgentMessage(
                id=row["id"],
                session_id=row["session_id"],
                from_agent=AgentRole(row["from_agent"]),
                to_agent=AgentRole(row["to_agent"]),
                message_type=row["message_type"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    # Stats methods
    async def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a session."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    COUNT(*) as total_findings,
                    COUNT(DISTINCT search_query) as unique_searches,
                    AVG(confidence) as avg_confidence
                FROM findings WHERE session_id = ?
                """,
                (session_id,),
            )
            findings_row = await cursor.fetchone()

            cursor = await conn.execute(
                """
                SELECT
                    COUNT(*) as total_topics,
                    MAX(depth) as max_depth,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_topics
                FROM topics WHERE session_id = ?
                """,
                (session_id,),
            )
            topics_row = await cursor.fetchone()

        return {
            "total_findings": findings_row["total_findings"] or 0,
            "unique_searches": findings_row["unique_searches"] or 0,
            "avg_confidence": findings_row["avg_confidence"] or 0,
            "total_topics": topics_row["total_topics"] or 0,
            "max_depth": topics_row["max_depth"] or 0,
            "completed_topics": topics_row["completed_topics"] or 0,
        }

    # Credibility Audit methods
    async def save_credibility_audit(
        self,
        session_id: str,
        finding_id: str | None,
        url: str,
        domain: str,
        final_score: float,
        domain_authority_score: float,
        recency_score: float,
        source_type_score: float,
        https_score: float,
        path_depth_score: float,
        credibility_label: str,
    ) -> int:
        """Save a credibility audit record."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO credibility_audit (
                        session_id, finding_id, url, domain, final_score,
                        domain_authority_score, recency_score, source_type_score,
                        https_score, path_depth_score, credibility_label, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        finding_id,
                        url,
                        domain,
                        final_score,
                        domain_authority_score,
                        recency_score,
                        source_type_score,
                        https_score,
                        path_depth_score,
                        credibility_label,
                        datetime.now().isoformat(),
                    ),
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in save_credibility_audit: %s", e, exc_info=True)
                raise

    async def get_credibility_audits(self, session_id: str) -> list[dict]:
        """Get all credibility audit records for a session."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM credibility_audit
                WHERE session_id = ?
                ORDER BY created_at
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Agent Decision methods
    async def save_agent_decision(
        self,
        session_id: str,
        agent_role: str,
        decision_type: str,
        decision_outcome: str,
        reasoning: str | None = None,
        inputs_json: str | None = None,
        metrics_json: str | None = None,
        iteration: int | None = None,
    ) -> int:
        """Save an agent decision record."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO agent_decisions (
                        session_id, agent_role, decision_type, decision_outcome,
                        reasoning, inputs_json, metrics_json, iteration, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        agent_role,
                        decision_type,
                        decision_outcome,
                        reasoning[:500] if reasoning else None,  # Truncate to 500 chars
                        inputs_json,
                        metrics_json,
                        iteration,
                        datetime.now().isoformat(),
                    ),
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in save_agent_decision: %s", e, exc_info=True)
                raise

    async def save_agent_decisions_batch(
        self,
        decisions: list[dict],
    ) -> int:
        """Save multiple agent decisions in a batch."""
        if not decisions:
            return 0

        pool = self._check_pool()
        async with pool.acquire() as conn:
            try:
                await conn.executemany(
                    """
                    INSERT INTO agent_decisions (
                        session_id, agent_role, decision_type, decision_outcome,
                        reasoning, inputs_json, metrics_json, iteration, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            d.get("session_id"),
                            d.get("agent_role"),
                            d.get("decision_type"),
                            d.get("decision_outcome"),
                            d.get("reasoning", "")[:500] if d.get("reasoning") else None,
                            d.get("inputs_json"),
                            d.get("metrics_json"),
                            d.get("iteration"),
                            d.get("created_at", datetime.now().isoformat()),
                        )
                        for d in decisions
                    ],
                )
                await conn.commit()
                return len(decisions)
            except Exception as e:
                await conn.rollback()
                logger.error("DB error in save_agent_decisions_batch: %s", e, exc_info=True)
                raise

    async def get_agent_decisions(
        self,
        session_id: str,
        agent_role: str | None = None,
        decision_type: str | None = None,
    ) -> list[dict]:
        """Get agent decisions for a session, optionally filtered."""
        pool = self._check_pool()
        query = "SELECT * FROM agent_decisions WHERE session_id = ?"
        params = [session_id]

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)
        if decision_type:
            query += " AND decision_type = ?"
            params.append(decision_type)

        query += " ORDER BY created_at"

        async with pool.acquire() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_decision_stats(self, session_id: str) -> dict:
        """Get decision statistics for a session."""
        pool = self._check_pool()
        async with pool.acquire() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    agent_role,
                    decision_type,
                    COUNT(*) as count
                FROM agent_decisions
                WHERE session_id = ?
                GROUP BY agent_role, decision_type
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()

        stats = {"by_agent": {}, "by_type": {}, "total": 0}
        for row in rows:
            agent = row["agent_role"]
            dtype = row["decision_type"]
            count = row["count"]

            stats["by_agent"].setdefault(agent, 0)
            stats["by_agent"][agent] += count

            stats["by_type"].setdefault(dtype, 0)
            stats["by_type"][dtype] += count

            stats["total"] += count

        return stats
