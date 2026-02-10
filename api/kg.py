"""
Knowledge graph query wrapper for API endpoints.

Reads from the research_kg.db SQLite database (synchronous) via asyncio executor.
"""
import json
import sqlite3
from functools import partial
from pathlib import Path
from typing import Any

import asyncio


# Default KG database path (relative to project root)
_DEFAULT_KG_PATH = Path("research_kg.db")


def _query_sync(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only query synchronously and return list of dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _scalar_sync(db_path: str, sql: str, params: tuple = ()) -> Any:
    """Execute a query and return a single scalar value."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


class KnowledgeGraphAPI:
    """Async wrapper around the KG SQLite database."""

    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or _DEFAULT_KG_PATH)

    def _db_exists(self) -> bool:
        return Path(self.db_path).exists()

    async def _run(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, self.db_path, *args))

    async def get_entities(self, entity_type: str | None = None, limit: int = 500) -> list[dict]:
        """Get all KG entities, optionally filtered by type."""
        if not self._db_exists():
            return []
        if entity_type:
            rows = await self._run(
                _query_sync,
                """
                SELECT id, name, entity_type, properties, created_at
                FROM kg_entities
                WHERE entity_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (entity_type, limit),
            )
        else:
            rows = await self._run(
                _query_sync,
                """
                SELECT id, name, entity_type, properties, created_at
                FROM kg_entities
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        # Parse properties JSON
        for row in rows:
            try:
                if row.get("properties"):
                    row["properties"] = json.loads(row["properties"])
                else:
                    row["properties"] = {}
            except Exception:
                row["properties"] = {}
        return rows

    async def get_relations(self, limit: int = 1000) -> list[dict]:
        """Get all KG relations."""
        if not self._db_exists():
            return []
        rows = await self._run(
            _query_sync,
            """
            SELECT
                r.id, r.subject_id, r.predicate, r.object_id,
                r.source_id, r.confidence, r.properties, r.created_at,
                s.name AS subject_name, s.entity_type AS subject_type,
                o.name AS object_name, o.entity_type AS object_type
            FROM kg_relations r
            LEFT JOIN kg_entities s ON s.id = r.subject_id
            LEFT JOIN kg_entities o ON o.id = r.object_id
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            try:
                if row.get("properties"):
                    row["properties"] = json.loads(row["properties"])
                else:
                    row["properties"] = {}
            except Exception:
                row["properties"] = {}
        return rows

    async def get_contradictions(self, limit: int = 100) -> list[dict]:
        """Get KG contradictions."""
        if not self._db_exists():
            return []
        return await self._run(
            _query_sync,
            """
            SELECT id, relation1_id, relation2_id, contradiction_type,
                   description, severity, resolution, resolved_at, created_at
            FROM kg_contradictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    async def get_stats(self) -> dict:
        """Get KG statistics."""
        if not self._db_exists():
            return {"entities": 0, "relations": 0, "contradictions": 0, "entity_types": {}}

        entities_row = await self._run(
            _scalar_sync,
            "SELECT COUNT(*) as count FROM kg_entities",
        )
        relations_row = await self._run(
            _scalar_sync,
            "SELECT COUNT(*) as count FROM kg_relations",
        )
        contradictions_row = await self._run(
            _scalar_sync,
            "SELECT COUNT(*) as count FROM kg_contradictions",
        )

        # Entity type breakdown
        type_rows = await self._run(
            _query_sync,
            "SELECT entity_type, COUNT(*) as count FROM kg_entities GROUP BY entity_type ORDER BY count DESC",
        )
        entity_types = {row["entity_type"]: row["count"] for row in type_rows}

        return {
            "entities": entities_row["count"] if entities_row else 0,
            "relations": relations_row["count"] if relations_row else 0,
            "contradictions": contradictions_row["count"] if contradictions_row else 0,
            "entity_types": entity_types,
        }

    async def get_entity(self, entity_id: str) -> dict | None:
        """Get a single entity by ID with its relations."""
        if not self._db_exists():
            return None
        rows = await self._run(
            _query_sync,
            "SELECT id, name, entity_type, properties, created_at FROM kg_entities WHERE id = ?",
            (entity_id,),
        )
        if not rows:
            return None
        entity = rows[0]
        try:
            if entity.get("properties"):
                entity["properties"] = json.loads(entity["properties"])
            else:
                entity["properties"] = {}
        except Exception:
            entity["properties"] = {}

        # Get relations for this entity
        relations = await self._run(
            _query_sync,
            """
            SELECT r.id, r.subject_id, r.predicate, r.object_id, r.confidence,
                   s.name AS subject_name, o.name AS object_name
            FROM kg_relations r
            LEFT JOIN kg_entities s ON s.id = r.subject_id
            LEFT JOIN kg_entities o ON o.id = r.object_id
            WHERE r.subject_id = ? OR r.object_id = ?
            """,
            (entity_id, entity_id),
        )
        entity["relations"] = relations
        return entity


# Global instance
_kg_instance: KnowledgeGraphAPI | None = None


def get_kg() -> KnowledgeGraphAPI:
    """Get or create the global KG API instance."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraphAPI()
    return _kg_instance
