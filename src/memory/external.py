"""External memory storage for findings and context overflow."""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class StoredMemory:
    """A memory item stored externally."""
    id: str
    session_id: int
    content: str
    memory_type: str  # 'finding', 'summary', 'context', 'decision'
    tags: list[str]
    created_at: datetime
    metadata: dict

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
        }


class ExternalMemoryStore:
    """External storage for long-term memory and context overflow.

    Based on Anthropic's approach of storing plan details and subagent
    outputs to filesystem/database when context approaches limits.

    Provides:
    - Persistent storage of findings and context
    - Searchable archive by tags and content
    - Session-scoped memory management
    """

    def __init__(self, db_path: str = "memory.db"):
        """Initialize external memory store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                session_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tags TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_session
            ON memories(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memories(memory_type)
        """)

        # Full-text search table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, tags, tokenize='porter')
        """)

        conn.commit()
        conn.close()

    def store(
        self,
        session_id: int,
        content: str,
        memory_type: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store content to external memory.

        Args:
            session_id: Current research session ID
            content: Content to store
            memory_type: Type of memory (finding, summary, context, decision)
            tags: Optional tags for retrieval
            metadata: Optional metadata

        Returns:
            ID of stored memory
        """
        import uuid

        memory_id = str(uuid.uuid4())[:8]
        tags = tags or []
        metadata = metadata or {}
        created_at = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Store in main table
        cursor.execute("""
            INSERT INTO memories (id, session_id, content, memory_type, tags, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            session_id,
            content,
            memory_type,
            json.dumps(tags),
            created_at.isoformat(),
            json.dumps(metadata),
        ))

        # Store in FTS table
        cursor.execute("""
            INSERT INTO memories_fts (rowid, content, tags)
            VALUES (last_insert_rowid(), ?, ?)
        """, (content, ' '.join(tags)))

        conn.commit()
        conn.close()

        return memory_id

    def search(
        self,
        query: str,
        session_id: Optional[int] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Search external memory.

        Args:
            query: Search query (full-text search)
            session_id: Optional filter by session
            memory_type: Optional filter by type
            limit: Maximum results to return

        Returns:
            List of matching memories
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        sql = """
            SELECT m.id, m.session_id, m.content, m.memory_type, m.tags, m.created_at, m.metadata
            FROM memories m
            JOIN memories_fts fts ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
        """
        params = [query]

        if session_id is not None:
            sql += " AND m.session_id = ?"
            params.append(session_id)

        if memory_type is not None:
            sql += " AND m.memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # FTS query failed, fall back to LIKE
            sql = """
                SELECT id, session_id, content, memory_type, tags, created_at, metadata
                FROM memories
                WHERE content LIKE ?
            """
            params = [f"%{query}%"]

            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)

            if memory_type is not None:
                sql += " AND memory_type = ?"
                params.append(memory_type)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

        conn.close()

        return [
            StoredMemory(
                id=row[0],
                session_id=row[1],
                content=row[2],
                memory_type=row[3],
                tags=json.loads(row[4]) if row[4] else [],
                created_at=datetime.fromisoformat(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]

    def get_by_session(
        self,
        session_id: int,
        memory_type: Optional[str] = None,
    ) -> list[StoredMemory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            memory_type: Optional filter by type

        Returns:
            List of memories
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if memory_type:
            cursor.execute("""
                SELECT id, session_id, content, memory_type, tags, created_at, metadata
                FROM memories
                WHERE session_id = ? AND memory_type = ?
                ORDER BY created_at
            """, (session_id, memory_type))
        else:
            cursor.execute("""
                SELECT id, session_id, content, memory_type, tags, created_at, metadata
                FROM memories
                WHERE session_id = ?
                ORDER BY created_at
            """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        return [
            StoredMemory(
                id=row[0],
                session_id=row[1],
                content=row[2],
                memory_type=row[3],
                tags=json.loads(row[4]) if row[4] else [],
                created_at=datetime.fromisoformat(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]

    def get_recent(
        self,
        session_id: int,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Get most recent memories for a session.

        Args:
            session_id: Session ID
            limit: Maximum results

        Returns:
            List of recent memories
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, session_id, content, memory_type, tags, created_at, metadata
            FROM memories
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            StoredMemory(
                id=row[0],
                session_id=row[1],
                content=row[2],
                memory_type=row[3],
                tags=json.loads(row[4]) if row[4] else [],
                created_at=datetime.fromisoformat(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]

    def delete_session(self, session_id: int) -> int:
        """Delete all memories for a session.

        Args:
            session_id: Session to delete

        Returns:
            Number of memories deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ?",
            (session_id,)
        )
        count = cursor.fetchone()[0]

        cursor.execute(
            "DELETE FROM memories WHERE session_id = ?",
            (session_id,)
        )

        conn.commit()
        conn.close()

        return count

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM memories")
        sessions = cursor.fetchone()[0]

        conn.close()

        return {
            'total_memories': total,
            'by_type': by_type,
            'sessions': sessions,
        }
