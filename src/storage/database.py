"""SQLite database for persisting research state and findings."""

import aiosqlite
import secrets
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.findings import (
    Finding,
    FindingType,
    ResearchSession,
    ResearchTopic,
    AgentMessage,
    AgentRole,
)


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
    text = re.sub(r'\?+$', '', text)

    # Remove stop words
    stop_words = {
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can',
        'latest', 'current', 'recent', 'new', 'best', 'top',
    }
    words = text.split()
    words = [w for w in words if w not in stop_words]

    # Take first 4-5 meaningful words
    words = words[:5]

    # Clean and join
    slug = '-'.join(words)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')

    # Limit length
    if len(slug) > 50:
        slug = slug[:50].rsplit('-', 1)[0]

    return slug or 'research'


class ResearchDatabase:
    """SQLite database manager for research persistence."""

    def __init__(self, db_path: str = "research.db"):
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect to the database and create tables if needed."""
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        await self._connection.executescript("""
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
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id)
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
        """)
        await self._connection.commit()

    # Session methods
    async def create_session(self, goal: str, time_limit_minutes: int = 60) -> ResearchSession:
        """Create a new research session with unique hex ID."""
        session_id = _generate_session_id()
        slug = _generate_slug(goal)
        now = datetime.now().isoformat()

        await self._connection.execute(
            """
            INSERT INTO sessions (id, goal, slug, time_limit_minutes, started_at, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            (session_id, goal, slug, time_limit_minutes, now),
        )
        await self._connection.commit()
        return ResearchSession(
            id=session_id,
            goal=goal,
            slug=slug,
            time_limit_minutes=time_limit_minutes,
            started_at=datetime.fromisoformat(now),
            status="active",
        )

    async def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get a session by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return ResearchSession(
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

    async def update_session(self, session: ResearchSession) -> None:
        """Update a session."""
        await self._connection.execute(
            """
            UPDATE sessions SET
                status = ?,
                ended_at = ?,
                total_findings = ?,
                total_searches = ?,
                depth_reached = ?
            WHERE id = ?
            """,
            (
                session.status,
                session.ended_at.isoformat() if session.ended_at else None,
                session.total_findings,
                session.total_searches,
                session.depth_reached,
                session.id,
            ),
        )
        await self._connection.commit()

    # Topic methods
    async def create_topic(
        self,
        session_id: str,
        topic: str,
        parent_topic_id: Optional[int] = None,
        depth: int = 0,
        priority: int = 5,
    ) -> ResearchTopic:
        """Create a new research topic."""
        cursor = await self._connection.execute(
            """
            INSERT INTO topics (session_id, topic, parent_topic_id, depth, priority)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, topic, parent_topic_id, depth, priority),
        )
        await self._connection.commit()
        return ResearchTopic(
            id=cursor.lastrowid,
            session_id=session_id,
            topic=topic,
            parent_topic_id=parent_topic_id,
            depth=depth,
            priority=priority,
        )

    async def get_pending_topics(self, session_id: str, limit: int = 10) -> list[ResearchTopic]:
        """Get pending topics ordered by priority."""
        cursor = await self._connection.execute(
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
        completed_at = datetime.now().isoformat() if status == "completed" else None
        await self._connection.execute(
            """
            UPDATE topics SET status = ?, completed_at = ?, findings_count = ?
            WHERE id = ?
            """,
            (status, completed_at, findings_count, topic_id),
        )
        await self._connection.commit()

    # Finding methods
    async def save_finding(self, finding: Finding, topic_id: Optional[int] = None) -> Finding:
        """Save a research finding."""
        cursor = await self._connection.execute(
            """
            INSERT INTO findings (
                session_id, topic_id, content, finding_type, source_url,
                confidence, search_query, created_at, validated_by_manager, manager_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        await self._connection.commit()
        finding.id = cursor.lastrowid
        return finding

    async def get_session_findings(self, session_id: str) -> list[Finding]:
        """Get all findings for a session."""
        cursor = await self._connection.execute(
            "SELECT * FROM findings WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            Finding(
                id=row["id"],
                session_id=row["session_id"],
                content=row["content"],
                finding_type=FindingType(row["finding_type"]),
                source_url=row["source_url"],
                confidence=row["confidence"],
                search_query=row["search_query"],
                created_at=datetime.fromisoformat(row["created_at"]),
                validated_by_manager=bool(row["validated_by_manager"]),
                manager_notes=row["manager_notes"],
            )
            for row in rows
        ]

    # Message methods
    async def save_message(self, message: AgentMessage) -> AgentMessage:
        """Save an agent message."""
        import json

        cursor = await self._connection.execute(
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
        await self._connection.commit()
        message.id = cursor.lastrowid
        return message

    async def get_session_messages(
        self, session_id: str, agent_filter: Optional[AgentRole] = None
    ) -> list[AgentMessage]:
        """Get messages for a session, optionally filtered by agent."""
        import json

        if agent_filter:
            cursor = await self._connection.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND (from_agent = ? OR to_agent = ?)
                ORDER BY created_at
                """,
                (session_id, agent_filter.value, agent_filter.value),
            )
        else:
            cursor = await self._connection.execute(
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
        cursor = await self._connection.execute(
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

        cursor = await self._connection.execute(
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
