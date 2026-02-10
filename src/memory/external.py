"""External memory storage for findings and context overflow."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite


@dataclass
class StoredMemory:
    """A memory item stored externally."""
    id: str
    session_id: str
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
        self._connection: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized (lazy initialization)."""
        if self._initialized:
            return
        async with self._connection_lock:
            if self._initialized:
                return
            await self._init_db()
            self._initialized = True

    async def _init_db(self) -> None:
        """Initialize database schema."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_session
                ON memories(session_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type
                ON memories(memory_type)
            """)

            # Full-text search table
            await conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, tags, tokenize='porter')
            """)

            await conn.commit()

    async def store(
        self,
        session_id: str,
        content: str,
        memory_type: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
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
        await self._ensure_initialized()
        import uuid

        memory_id = str(uuid.uuid4())[:8]
        tags = tags or []
        metadata = metadata or {}
        created_at = datetime.now()

        async with aiosqlite.connect(self.db_path) as conn:
            try:
                # Store in main table
                await conn.execute("""
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
                await conn.execute("""
                    INSERT INTO memories_fts (rowid, content, tags)
                    VALUES (last_insert_rowid(), ?, ?)
                """, (content, ' '.join(tags)))

                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        return memory_id

    # Synchronous wrapper for backward compatibility (runs in executor)
    def store_sync(
        self,
        session_id: str,
        content: str,
        memory_type: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Synchronous store for backward compatibility.

        Note: Prefer using the async `store` method when possible.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self.store(session_id, content, memory_type, tags, metadata),
                loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            # No running event loop, create one
            return asyncio.run(self.store(session_id, content, memory_type, tags, metadata))

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        memory_type: str | None = None,
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
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row

            # Build query
            sql = """
                SELECT m.id, m.session_id, m.content, m.memory_type, m.tags, m.created_at, m.metadata
                FROM memories m
                JOIN memories_fts fts ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ?
            """
            params: list = [query]

            if session_id is not None:
                sql += " AND m.session_id = ?"
                params.append(session_id)

            if memory_type is not None:
                sql += " AND m.memory_type = ?"
                params.append(memory_type)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            try:
                cursor = await conn.execute(sql, params)
                rows = await cursor.fetchall()
            except aiosqlite.OperationalError:
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

                cursor = await conn.execute(sql, params)
                rows = await cursor.fetchall()

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

    async def get_by_session(
        self,
        session_id: str,
        memory_type: str | None = None,
    ) -> list[StoredMemory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            memory_type: Optional filter by type

        Returns:
            List of memories
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as conn:
            if memory_type:
                cursor = await conn.execute("""
                    SELECT id, session_id, content, memory_type, tags, created_at, metadata
                    FROM memories
                    WHERE session_id = ? AND memory_type = ?
                    ORDER BY created_at
                """, (session_id, memory_type))
            else:
                cursor = await conn.execute("""
                    SELECT id, session_id, content, memory_type, tags, created_at, metadata
                    FROM memories
                    WHERE session_id = ?
                    ORDER BY created_at
                """, (session_id,))

            rows = await cursor.fetchall()

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

    async def get_recent(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """Get most recent memories for a session.

        Args:
            session_id: Session ID
            limit: Maximum results

        Returns:
            List of recent memories
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("""
                SELECT id, session_id, content, memory_type, tags, created_at, metadata
                FROM memories
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit))

            rows = await cursor.fetchall()

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

    async def delete_session(self, session_id: str) -> int:
        """Delete all memories for a session.

        Args:
            session_id: Session to delete

        Returns:
            Number of memories deleted
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM memories WHERE session_id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            count = row[0] if row else 0

            try:
                await conn.execute(
                    "DELETE FROM memories WHERE session_id = ?",
                    (session_id,)
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        return count

    async def get_stats(self) -> dict:
        """Get memory store statistics."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM memories")
            row = await cursor.fetchone()
            total = row[0] if row else 0

            cursor = await conn.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
            """)
            rows = await cursor.fetchall()
            by_type = {row[0]: row[1] for row in rows}

            cursor = await conn.execute("SELECT COUNT(DISTINCT session_id) FROM memories")
            row = await cursor.fetchone()
            sessions = row[0] if row else 0

        return {
            'total_memories': total,
            'by_type': by_type,
            'sessions': sessions,
        }
