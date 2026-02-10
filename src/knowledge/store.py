"""Hybrid NetworkX + SQLite storage for knowledge graphs."""

import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .models import Contradiction, Entity, Relation


class HybridKnowledgeGraphStore:
    """Hybrid storage: NetworkX for in-memory graph algorithms, SQLite for persistence.

    This provides:
    - Fast graph queries via NetworkX (PageRank, centrality, paths)
    - Persistent storage via SQLite
    - Automatic sync between the two
    """

    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = Path(db_path)

        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = None  # Fallback mode

        self._init_db()
        self._load_from_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                source_id TEXT,
                confidence REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
                FOREIGN KEY (object_id) REFERENCES kg_entities(id)
            )
        """)

        # Findings/Sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_findings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT,
                source_title TEXT,
                credibility_score REAL,
                finding_type TEXT,
                search_query TEXT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Contradictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_contradictions (
                id TEXT PRIMARY KEY,
                relation1_id TEXT NOT NULL,
                relation2_id TEXT NOT NULL,
                contradiction_type TEXT NOT NULL,
                description TEXT,
                severity TEXT DEFAULT 'medium',
                resolution TEXT,
                resolved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (relation1_id) REFERENCES kg_relations(id),
                FOREIGN KEY (relation2_id) REFERENCES kg_relations(id)
            )
        """)

        # Indexes for fast queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON kg_entities(entity_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_name ON kg_entities(name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_predicate ON kg_relations(predicate)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_subject ON kg_relations(subject_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_object ON kg_relations(object_id)"
        )

        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Load graph from SQLite into NetworkX."""
        if not HAS_NETWORKX or self.graph is None:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load entities as nodes
        cursor.execute("SELECT id, name, entity_type, properties FROM kg_entities")
        for row in cursor.fetchall():
            entity_id, name, entity_type, properties = row
            props = json.loads(properties) if properties else {}
            self.graph.add_node(
                entity_id,
                name=name,
                entity_type=entity_type,
                **props
            )

        # Load relations as edges
        cursor.execute("""
            SELECT id, subject_id, predicate, object_id, confidence, properties
            FROM kg_relations
        """)
        for row in cursor.fetchall():
            rel_id, subj, pred, obj, conf, properties = row
            props = json.loads(properties) if properties else {}
            if subj in self.graph and obj in self.graph:
                self.graph.add_edge(
                    subj, obj,
                    relation_id=rel_id,
                    predicate=pred,
                    confidence=conf,
                    **props
                )

        conn.close()

    def add_entity(self, entity: Entity) -> str:
        """Add entity to both NetworkX and SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize embedding
        embedding_blob = pickle.dumps(entity.embedding) if entity.embedding is not None else None
        properties = json.dumps({
            'aliases': entity.aliases,
            'sources': entity.sources,
            'confidence': entity.confidence,
            **entity.properties,
        })

        cursor.execute("""
            INSERT OR REPLACE INTO kg_entities (id, name, entity_type, properties, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (entity.id, entity.name, entity.entity_type, properties, embedding_blob, datetime.now().isoformat()))

        conn.commit()
        conn.close()

        # Add to NetworkX
        if HAS_NETWORKX and self.graph is not None:
            self.graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                aliases=entity.aliases,
                sources=entity.sources,
                confidence=entity.confidence,
                **entity.properties
            )

        return entity.id

    def add_relation(self, relation: Relation) -> str:
        """Add relation to both NetworkX and SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        properties = json.dumps({
            'timestamp': relation.timestamp,
            **relation.properties,
        })

        cursor.execute("""
            INSERT INTO kg_relations
            (id, subject_id, predicate, object_id, source_id, confidence, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            relation.id, relation.subject_id, relation.predicate,
            relation.object_id, relation.source_id, relation.confidence,
            properties
        ))

        conn.commit()
        conn.close()

        # Add to NetworkX
        if HAS_NETWORKX and self.graph is not None:
            self.graph.add_edge(
                relation.subject_id,
                relation.object_id,
                relation_id=relation.id,
                predicate=relation.predicate,
                confidence=relation.confidence,
                source_id=relation.source_id
            )

        return relation.id

    def add_contradiction(self, contradiction: Contradiction) -> str:
        """Record a detected contradiction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO kg_contradictions
            (id, relation1_id, relation2_id, contradiction_type, description, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            contradiction.id,
            contradiction.relation1_id,
            contradiction.relation2_id,
            contradiction.contradiction_type,
            contradiction.description,
            contradiction.severity,
        ))

        conn.commit()
        conn.close()
        return contradiction.id

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, name, entity_type, properties, embedding FROM kg_entities WHERE id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        entity_id, name, entity_type, properties, embedding_blob = row
        props = json.loads(properties) if properties else {}

        return Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            aliases=props.get('aliases', []),
            sources=props.get('sources', []),
            confidence=props.get('confidence', 1.0),
            embedding=pickle.loads(embedding_blob) if embedding_blob else None,
            properties={k: v for k, v in props.items() if k not in ['aliases', 'sources', 'confidence']},
        )

    def query_by_entity_type(self, entity_type: str) -> list[dict]:
        """Query entities by type using SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, name, properties FROM kg_entities WHERE entity_type = ?",
            (entity_type,)
        )

        results = []
        for row in cursor.fetchall():
            entity_id, name, properties = row
            props = json.loads(properties) if properties else {}
            results.append({
                'id': entity_id,
                'name': name,
                'type': entity_type,
                **props
            })

        conn.close()
        return results

    def get_entity_relations(self, entity_id: str) -> dict:
        """Get all relations for an entity."""
        if not HAS_NETWORKX or self.graph is None or entity_id not in self.graph:
            # Fallback to SQLite
            return self._get_entity_relations_sql(entity_id)

        outgoing = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            outgoing.append({
                'predicate': data.get('predicate', ''),
                'target_id': target,
                'target_name': self.graph.nodes[target].get('name', target),
                'confidence': data.get('confidence', 1.0)
            })

        incoming = []
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            incoming.append({
                'predicate': data.get('predicate', ''),
                'source_id': source,
                'source_name': self.graph.nodes[source].get('name', source),
                'confidence': data.get('confidence', 1.0)
            })

        return {'outgoing': outgoing, 'incoming': incoming}

    def _get_entity_relations_sql(self, entity_id: str) -> dict:
        """Get entity relations via SQLite (fallback)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Outgoing relations
        cursor.execute("""
            SELECT r.predicate, r.object_id, e.name, r.confidence
            FROM kg_relations r
            JOIN kg_entities e ON r.object_id = e.id
            WHERE r.subject_id = ?
        """, (entity_id,))

        outgoing = [
            {'predicate': row[0], 'target_id': row[1], 'target_name': row[2], 'confidence': row[3]}
            for row in cursor.fetchall()
        ]

        # Incoming relations
        cursor.execute("""
            SELECT r.predicate, r.subject_id, e.name, r.confidence
            FROM kg_relations r
            JOIN kg_entities e ON r.subject_id = e.id
            WHERE r.object_id = ?
        """, (entity_id,))

        incoming = [
            {'predicate': row[0], 'source_id': row[1], 'source_name': row[2], 'confidence': row[3]}
            for row in cursor.fetchall()
        ]

        conn.close()
        return {'outgoing': outgoing, 'incoming': incoming}

    def get_unresolved_contradictions(self) -> list[dict]:
        """Get all unresolved contradictions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, relation1_id, relation2_id, contradiction_type, description, severity
            FROM kg_contradictions
            WHERE resolved_at IS NULL
        """)

        contradictions = []
        for row in cursor.fetchall():
            contradictions.append({
                'id': row[0],
                'relation1_id': row[1],
                'relation2_id': row[2],
                'type': row[3],
                'description': row[4],
                'severity': row[5],
            })

        conn.close()
        return contradictions

    def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        if HAS_NETWORKX and self.graph is not None:
            num_entities = self.graph.number_of_nodes()
            num_relations = self.graph.number_of_edges()
            num_components = nx.number_weakly_connected_components(self.graph) if num_entities > 0 else 0
            density = nx.density(self.graph) if num_entities > 1 else 0
        else:
            # Fallback to SQL counts
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM kg_entities")
            num_entities = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM kg_relations")
            num_relations = cursor.fetchone()[0]
            conn.close()
            num_components = 0
            density = 0

        # Get entity type counts
        type_counts = {}
        if HAS_NETWORKX and self.graph is not None:
            for _, data in self.graph.nodes(data=True):
                etype = data.get('entity_type', 'UNKNOWN')
                type_counts[etype] = type_counts.get(etype, 0) + 1

        return {
            'num_entities': num_entities,
            'num_relations': num_relations,
            'num_components': num_components,
            'density': density,
            'entity_types': type_counts,
        }

    def clear(self):
        """Clear all data from the knowledge graph."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM kg_contradictions")
        cursor.execute("DELETE FROM kg_relations")
        cursor.execute("DELETE FROM kg_entities")
        cursor.execute("DELETE FROM kg_findings")

        conn.commit()
        conn.close()

        if HAS_NETWORKX and self.graph is not None:
            self.graph.clear()
