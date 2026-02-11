"""Real-time incremental knowledge graph construction."""

import asyncio
import json
import re
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .credibility import CredibilityScorer
from .fast_ner import FastNER, get_fast_ner
from .models import ENTITY_TYPES, Contradiction, Entity, KGFinding, Relation
from .store import HybridKnowledgeGraphStore


# Canonical predicate mapping — normalizes free-form predicates to a fixed set
# for better graph connectivity. ~130 synonyms → ~20 canonical forms.
PREDICATE_CANONICAL_MAP: dict[str, str] = {
    # === Canonical predicates (identity mappings) ===
    'supports': 'supports',
    'contradicts': 'contradicts',
    'qualifies': 'qualifies',
    'cites': 'cites',
    'is_a': 'is_a',
    'part_of': 'part_of',
    'causes': 'causes',
    'correlates_with': 'correlates_with',
    'enables': 'enables',
    'implements': 'implements',
    'outperforms': 'outperforms',
    'similar_to': 'similar_to',
    'alternative_to': 'alternative_to',
    'authored_by': 'authored_by',
    'published_in': 'published_in',
    'mentioned_in': 'mentioned_in',
    'co_occurs_with': 'co_occurs_with',
    # Directional predicates (kept separate for contradiction detection)
    'increases': 'increases',
    'decreases': 'decreases',
    'prevents': 'prevents',
    'blocks': 'blocks',
    'improves': 'improves',
    'worsens': 'worsens',
    'underperforms': 'underperforms',
    # === Synonyms → supports ===
    'evidence_for': 'supports', 'backs': 'supports', 'confirms': 'supports',
    'validates': 'supports', 'demonstrates': 'supports', 'proves': 'supports',
    'shows': 'supports', 'indicates': 'supports', 'corroborates': 'supports',
    'suggests': 'supports', 'backs_up': 'supports',
    # === Synonyms → contradicts ===
    'conflicts_with': 'contradicts', 'opposes': 'contradicts', 'refutes': 'contradicts',
    'disproves': 'contradicts', 'challenges': 'contradicts', 'negates': 'contradicts',
    'disputes': 'contradicts', 'denies': 'contradicts',
    # === Synonyms → qualifies ===
    'limits': 'qualifies', 'conditions': 'qualifies', 'constrains': 'qualifies',
    'modifies': 'qualifies',
    # === Synonyms → cites ===
    'references': 'cites', 'refers_to': 'cites', 'based_on': 'cites',
    'builds_on': 'cites', 'draws_from': 'cites', 'derived_from': 'cites',
    # === Synonyms → is_a ===
    'is': 'is_a', 'type_of': 'is_a', 'kind_of': 'is_a', 'instance_of': 'is_a',
    'form_of': 'is_a', 'classified_as': 'is_a', 'subtype_of': 'is_a',
    'category_of': 'is_a',
    # === Synonyms → part_of ===
    'component_of': 'part_of', 'belongs_to': 'part_of', 'subset_of': 'part_of',
    'included_in': 'part_of', 'element_of': 'part_of', 'contains': 'part_of',
    'comprises': 'part_of', 'consists_of': 'part_of', 'has': 'part_of',
    'includes': 'part_of',
    # === Synonyms → causes ===
    'leads_to': 'causes', 'results_in': 'causes', 'produces': 'causes',
    'triggers': 'causes', 'induces': 'causes', 'generates': 'causes',
    'creates': 'causes', 'drives': 'causes', 'contributes_to': 'causes',
    'influences': 'causes', 'affects': 'causes', 'impacts': 'causes',
    'determines': 'causes',
    # === Synonyms → correlates_with ===
    'associated_with': 'correlates_with', 'related_to': 'correlates_with',
    'linked_to': 'correlates_with', 'connected_to': 'correlates_with',
    'measures': 'correlates_with', 'predicts': 'correlates_with',
    'corresponds_to': 'correlates_with',
    # === Synonyms → enables ===
    'allows': 'enables', 'facilitates': 'enables', 'permits': 'enables',
    'requires': 'enables', 'depends_on': 'enables', 'needs': 'enables',
    # === Synonyms → implements ===
    'realizes': 'implements', 'applies': 'implements', 'uses': 'implements',
    'employs': 'implements', 'utilizes': 'implements', 'leverages': 'implements',
    'adopts': 'implements', 'extends': 'implements',
    # === Synonyms → outperforms ===
    'better_than': 'outperforms', 'surpasses': 'outperforms', 'exceeds': 'outperforms',
    'superior_to': 'outperforms', 'beats': 'outperforms', 'faster_than': 'outperforms',
    # === Synonyms → similar_to ===
    'resembles': 'similar_to', 'analogous_to': 'similar_to', 'comparable_to': 'similar_to',
    'equivalent_to': 'similar_to', 'same_as': 'similar_to',
    # === Synonyms → alternative_to ===
    'replaces': 'alternative_to', 'substitute_for': 'alternative_to',
    'competes_with': 'alternative_to', 'instead_of': 'alternative_to',
    # === Synonyms → authored_by ===
    'written_by': 'authored_by', 'created_by': 'authored_by', 'developed_by': 'authored_by',
    'invented_by': 'authored_by', 'designed_by': 'authored_by', 'proposed_by': 'authored_by',
    'introduced_by': 'authored_by',
    # === Synonyms → published_in ===
    'appeared_in': 'published_in', 'presented_at': 'published_in',
    'reported_in': 'published_in', 'released_in': 'published_in',
    # === Synonyms → mentioned_in ===
    'described_in': 'mentioned_in', 'discussed_in': 'mentioned_in',
    'found_in': 'mentioned_in', 'cited_in': 'mentioned_in',
}

# Canonical predicate names for inclusion in LLM prompts
CANONICAL_PREDICATES_PROMPT = (
    "supports, contradicts, causes, enables, implements, is_a, part_of, "
    "correlates_with, outperforms, similar_to, alternative_to, authored_by, "
    "published_in, increases, decreases, qualifies, cites, mentioned_in"
)


class IncrementalKnowledgeGraph:
    """Real-time knowledge graph that builds as findings stream in.

    Based on iText2KG, KGGen, and Graphiti patterns:
    - Async processing for non-blocking LLM I/O
    - Entity resolution via embedding similarity
    - Automatic contradiction detection
    - Incremental updates without full reprocessing
    """

    # Predicate pairs that are contradictory
    CONTRADICTORY_PREDICATES = [
        ('increases', 'decreases'),
        ('causes', 'prevents'),
        ('supports', 'contradicts'),
        ('enables', 'blocks'),
        ('improves', 'worsens'),
        ('is', 'is not'),
        ('has', 'lacks'),
        ('before', 'after'),
        ('greater than', 'less than'),
        ('outperforms', 'underperforms'),
    ]

    def __init__(
        self,
        llm_callback: Callable[[str], Any],
        store: HybridKnowledgeGraphStore | None = None,
        similarity_threshold: float = 0.7,
        use_fast_ner: bool = True,
        credibility_audit_callback: Callable[[dict], Any] | None = None,
        session_id: str | None = None,
    ):
        """Initialize the incremental knowledge graph.

        Args:
            llm_callback: Async function to call LLM for extraction
            store: Storage backend (creates new if not provided)
            similarity_threshold: Threshold for entity matching (0.7 = 70% similar)
            use_fast_ner: Use spaCy for fast NER (with LLM fallback for domain types)
            credibility_audit_callback: Optional async callback to persist credibility audits
            session_id: Session ID to associate with entities/relations
        """
        self.llm_callback = llm_callback
        self.session_id = session_id
        self.store = store or HybridKnowledgeGraphStore()
        self.similarity_threshold = similarity_threshold
        self.credibility_scorer = CredibilityScorer()
        self.credibility_audit_callback = credibility_audit_callback

        # Fast NER using spaCy (falls back to LLM for domain types)
        self.use_fast_ner = use_fast_ner
        self.fast_ner = get_fast_ner() if use_fast_ner else None

        # In-memory indexes for fast entity resolution
        self.entity_embeddings: dict[str, Any] = {}
        self.entity_by_name: dict[str, str] = {}  # lowercase name -> entity_id
        self.entity_by_type: dict[str, list[str]] = {}

        # Contradiction tracking
        self.contradictions: list[Contradiction] = []

        # Processing lock for thread safety
        self._processing_lock = asyncio.Lock()

    async def _save_credibility_audit(self, audit_data: dict) -> None:
        """Fire-and-forget save of credibility audit data."""
        if not self.credibility_audit_callback:
            return
        try:
            result = self.credibility_audit_callback(audit_data)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Don't let audit errors affect main processing
            pass

    async def add_finding(self, finding: KGFinding, fast_mode: bool = True) -> dict:
        """Process a new finding and integrate it into the knowledge graph.

        Args:
            finding: The finding to process
            fast_mode: If True, use single-pass extraction (fewer LLM calls)

        Returns:
            Dict with extracted entities, relations, and any contradictions found
        """
        async with self._processing_lock:
            result = {
                'entities': [],
                'relations': [],
                'contradictions_found': 0,
                'finding_id': finding.id,
            }

            # Score source credibility with full audit trail
            credibility, audit_data = self.credibility_scorer.score_source_with_audit(
                finding.source_url,
                finding.timestamp,
            )
            finding.credibility_score = credibility.score

            # Queue async credibility audit write if callback provided
            if self.credibility_audit_callback:
                audit_data['finding_id'] = finding.id
                asyncio.create_task(self._save_credibility_audit(audit_data))

            if fast_mode:
                # Fast mode: Single LLM call for both entities and relations
                return await self._fast_extract(finding)

            # Full mode: Multi-step extraction (slower but more thorough)
            # Step 1: Extract atomic facts
            atomic_facts = await self._extract_atomic_facts(finding)

            # Step 2: Extract entities from each fact
            all_entities = []
            for fact in atomic_facts:
                entities = await self._extract_entities(fact, finding.id)
                all_entities.extend(entities)

            # Step 3: Resolve entities against existing graph
            resolved_entities = []
            for entity in all_entities:
                resolved = await self._resolve_entity(entity)
                resolved_entities.append(resolved)
                result['entities'].append(resolved)

            # Step 4: Extract relations using resolved entities
            for fact in atomic_facts:
                relations = await self._extract_relations(fact, resolved_entities, finding.id)
                for relation in relations:
                    # Check for contradictions before adding
                    contradiction = self._check_contradiction(relation)
                    if contradiction:
                        self.contradictions.append(contradiction)
                        self.store.add_contradiction(contradiction)
                        result['contradictions_found'] += 1

                    # Add relation to store
                    self.store.add_relation(relation, self.session_id)
                    result['relations'].append(relation)

            return result

    async def _fast_extract(self, finding: KGFinding) -> dict:
        """Fast extraction using spaCy NER + LLM for relations.

        Uses spaCy for ~100x faster entity extraction, with LLM only for:
        - Domain-specific entity types (CONCEPT, CLAIM, METHOD, etc.)
        - Relation extraction between entities
        """
        result = {
            'entities': [],
            'relations': [],
            'contradictions_found': 0,
            'finding_id': finding.id,
        }

        # Step 1: Fast entity extraction with spaCy + LLM for domain types
        if self.use_fast_ner and self.fast_ner and self.fast_ner.enabled:
            # Use spaCy for standard entities + LLM for domain-specific types
            entities = await self.fast_ner.extract_with_llm(
                finding.content,
                self.llm_callback,
                source_id=finding.id,
                extract_domain_types=True,  # Also use LLM for CONCEPT, CLAIM, etc.
            )

            # Resolve and add entities (improvement #1: raised cap 5→8)
            for entity in entities[:8]:
                # Improvement #7: propagate source URL to entity properties
                if finding.source_url:
                    entity.properties.setdefault('source_urls', [])
                    if finding.source_url not in entity.properties['source_urls']:
                        entity.properties['source_urls'].append(finding.source_url)
                resolved = await self._resolve_entity(entity)
                result['entities'].append(resolved)

        else:
            # Fallback to LLM-only extraction
            return await self._llm_only_extract(finding)

        # Step 2: Extract relations using LLM (needs semantic understanding)
        if len(result['entities']) >= 2:
            relations = await self._extract_relations_fast(
                finding.content,
                result['entities'],
                finding.id,
                credibility_score=finding.credibility_score,  # improvement #5
            )

            for relation in relations:
                # Check for contradictions
                contradiction = self._check_contradiction(relation)
                if contradiction:
                    self.contradictions.append(contradiction)
                    self.store.add_contradiction(contradiction)
                    result['contradictions_found'] += 1

                self.store.add_relation(relation, self.session_id)
                result['relations'].append(relation)

            # Improvement #4: cross-finding co-occurrence links
            co_relations = self._build_co_occurrence_links(
                result['entities'], result['relations'], finding,
            )
            for relation in co_relations:
                self.store.add_relation(relation, self.session_id)
                result['relations'].append(relation)

        return result

    async def _llm_only_extract(self, finding: KGFinding) -> dict:
        """Fallback LLM-only extraction when spaCy is unavailable."""
        result = {
            'entities': [],
            'relations': [],
            'contradictions_found': 0,
            'finding_id': finding.id,
        }

        entity_types_list = ", ".join(list(ENTITY_TYPES.keys())[:10])

        prompt = f"""Extract key entities and their relationships from this research finding.

Finding: {finding.content}

Entity Types: {entity_types_list}

For each entity provide:
- name: Canonical name
- type: One of the entity types above
- description: One-sentence description of this entity

For METRIC entities, also include:
- value: The numeric value (e.g. "3.5", "92%")
- unit: The unit of measurement (e.g. "%", "ms", "users")

For CLAIM entities, also include:
- attributed_to: Who made this claim (if known)
- date_claimed: When it was claimed (if known)

Allowed predicates for relations (use ONLY these):
{CANONICAL_PREDICATES_PROMPT}

Return as JSON:
{{
  "entities": [
    {{"name": "Entity name", "type": "CONCEPT", "description": "Brief description"}}
  ],
  "relations": [
    {{"subject": "Entity1", "predicate": "causes", "object": "Entity2"}}
  ]
}}

Extract up to 8 entities and 5 relations. Focus on the most important facts."""

        try:
            response = await self.llm_callback(prompt)

            # Parse response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())

                # Improvement #5: use source credibility for relation confidence
                confidence = finding.credibility_score if finding.credibility_score else 0.8

                # Process entities (improvement #1: raised cap 5→8)
                for e in data.get('entities', [])[:8]:
                    if isinstance(e, dict) and e.get('name'):
                        entity_type = e.get('type', 'CONCEPT').upper()

                        # Build properties (improvements #2, #3, #7, #8)
                        props: dict[str, Any] = {}
                        if e.get('description'):
                            props['description'] = e['description']
                        if finding.source_url:
                            props['source_urls'] = [finding.source_url]
                        if entity_type == 'METRIC' and e.get('value'):
                            props['metric_value'] = e['value']
                            props['metric_unit'] = e.get('unit', '')
                        if entity_type == 'CLAIM':
                            if e.get('attributed_to'):
                                props['attributed_to'] = e['attributed_to']
                            if e.get('date_claimed'):
                                props['date_claimed'] = e['date_claimed']

                        entity = Entity(
                            id=self._generate_id(),
                            name=e['name'],
                            entity_type=entity_type,
                            aliases=[],
                            sources=[finding.id],
                            properties=props,
                        )
                        resolved = await self._resolve_entity(entity)
                        result['entities'].append(resolved)

                # Process relations (improvement #1: raised cap 3→5)
                name_to_id = {e.name.lower(): e.id for e in result['entities']}
                for r in data.get('relations', [])[:5]:
                    if isinstance(r, dict):
                        subject_id = name_to_id.get(r.get('subject', '').lower())
                        object_id = name_to_id.get(r.get('object', '').lower())

                        if subject_id and object_id and subject_id != object_id:
                            relation = Relation(
                                id=self._generate_id(),
                                subject_id=subject_id,
                                predicate=self._normalize_predicate(
                                    r.get('predicate', 'related_to')),
                                object_id=object_id,
                                source_id=finding.id,
                                confidence=confidence,
                            )

                            # Check for contradictions
                            contradiction = self._check_contradiction(relation)
                            if contradiction:
                                self.contradictions.append(contradiction)
                                self.store.add_contradiction(contradiction)
                                result['contradictions_found'] += 1

                            self.store.add_relation(relation, self.session_id)
                            result['relations'].append(relation)

                # Improvement #4: co-occurrence links
                co_relations = self._build_co_occurrence_links(
                    result['entities'], result['relations'], finding,
                )
                for relation in co_relations:
                    self.store.add_relation(relation, self.session_id)
                    result['relations'].append(relation)

        except Exception:
            # Silently fail - KG is optional
            pass

        return result

    async def _extract_relations_fast(
        self,
        text: str,
        entities: list[Entity],
        source_id: str,
        credibility_score: float | None = None,
    ) -> list[Relation]:
        """Fast relation extraction for pre-extracted entities.

        Uses a compact prompt focused only on relations.
        """
        if len(entities) < 2:
            return []

        entity_names = ", ".join([e.name for e in entities[:8]])

        prompt = f"""Given these entities: {entity_names}

Extract relationships from this text:
{text[:500]}

Allowed predicates (use ONLY these):
{CANONICAL_PREDICATES_PROMPT}

Return JSON array (max 5 relations):
[{{"subject": "Entity1", "predicate": "causes", "object": "Entity2"}}]"""

        try:
            response = await self.llm_callback(prompt)

            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not match:
                return []

            data = json.loads(match.group())

            # Map entity names to IDs
            name_to_id = {e.name.lower(): e.id for e in entities}
            for e in entities:
                for alias in e.aliases:
                    name_to_id[alias.lower()] = e.id

            confidence = credibility_score if credibility_score else 0.8

            relations = []
            for r in data[:5]:
                if not isinstance(r, dict):
                    continue

                subject_id = name_to_id.get(r.get('subject', '').lower())
                object_id = name_to_id.get(r.get('object', '').lower())

                if subject_id and object_id and subject_id != object_id:
                    relation = Relation(
                        id=self._generate_id(),
                        subject_id=subject_id,
                        predicate=self._normalize_predicate(r.get('predicate', 'related_to')),
                        object_id=object_id,
                        source_id=source_id,
                        confidence=confidence,
                    )
                    relations.append(relation)

            return relations

        except Exception:
            return []

    async def add_findings_batch(self, findings: list[KGFinding], batch_size: int = 5) -> dict:
        """Process multiple findings in batches for efficiency.

        Groups findings and extracts from multiple in a single LLM call.

        Args:
            findings: List of findings to process
            batch_size: Number of findings per LLM call

        Returns:
            Aggregated results from all findings
        """
        result = {
            'total_entities': 0,
            'total_relations': 0,
            'total_contradictions': 0,
            'processed': 0,
        }

        # Score credibility for all findings first
        for finding in findings:
            # Score source credibility with full audit trail
            credibility, audit_data = self.credibility_scorer.score_source_with_audit(
                finding.source_url,
                finding.timestamp,
            )
            finding.credibility_score = credibility.score

            # Queue async credibility audit write if callback provided
            if self.credibility_audit_callback:
                audit_data['finding_id'] = finding.id
                asyncio.create_task(self._save_credibility_audit(audit_data))

        # Process in batches
        for i in range(0, len(findings), batch_size):
            batch = findings[i:i + batch_size]
            batch_result = await self._extract_batch(batch)

            result['total_entities'] += batch_result.get('entities_count', 0)
            result['total_relations'] += batch_result.get('relations_count', 0)
            result['total_contradictions'] += batch_result.get('contradictions', 0)
            result['processed'] += len(batch)

        return result

    async def _extract_batch(self, findings: list[KGFinding]) -> dict:
        """Extract entities and relations from a batch of findings in one LLM call."""
        result = {'entities_count': 0, 'relations_count': 0, 'contradictions': 0}

        if not findings:
            return result

        entity_types_list = ", ".join(list(ENTITY_TYPES.keys())[:10])

        # Format findings for batch processing
        findings_text = "\n\n".join([
            f"Finding {i+1}: {f.content[:500]}"
            for i, f in enumerate(findings)
        ])

        prompt = f"""Extract key entities and relationships from these research findings.

{findings_text}

Entity Types: {entity_types_list}

For each entity provide name, type, and a brief description.
For METRIC entities, also include "value" and "unit".

Allowed predicates for relations (use ONLY these):
{CANONICAL_PREDICATES_PROMPT}

Return as JSON array, one object per finding:
[
  {{
    "finding": 1,
    "entities": [{{"name": "Entity", "type": "CONCEPT", "description": "Brief description"}}],
    "relations": [{{"subject": "Entity1", "predicate": "causes", "object": "Entity2"}}]
  }}
]

Max 5 entities and 4 relations per finding."""

        try:
            response = await self.llm_callback(prompt)

            # Parse response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                data = json.loads(match.group())

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    finding_idx = item.get('finding', 1) - 1
                    if finding_idx < 0 or finding_idx >= len(findings):
                        finding_idx = 0
                    finding = findings[finding_idx]
                    confidence = finding.credibility_score if finding.credibility_score else 0.8

                    # Process entities (raised cap 3→5)
                    entities = []
                    for e in item.get('entities', [])[:5]:
                        if isinstance(e, dict) and e.get('name'):
                            entity_type = e.get('type', 'CONCEPT').upper()
                            props: dict[str, Any] = {}
                            if e.get('description'):
                                props['description'] = e['description']
                            if finding.source_url:
                                props['source_urls'] = [finding.source_url]
                            if entity_type == 'METRIC' and e.get('value'):
                                props['metric_value'] = e['value']
                                props['metric_unit'] = e.get('unit', '')

                            entity = Entity(
                                id=self._generate_id(),
                                name=e['name'],
                                entity_type=entity_type,
                                aliases=[],
                                sources=[finding.id],
                                properties=props,
                            )
                            resolved = await self._resolve_entity(entity)
                            entities.append(resolved)
                            result['entities_count'] += 1

                    # Process relations (raised cap 2→4)
                    name_to_id = {e.name.lower(): e.id for e in entities}
                    for r in item.get('relations', [])[:4]:
                        if isinstance(r, dict):
                            subject_id = name_to_id.get(r.get('subject', '').lower())
                            object_id = name_to_id.get(r.get('object', '').lower())

                            if subject_id and object_id and subject_id != object_id:
                                relation = Relation(
                                    id=self._generate_id(),
                                    subject_id=subject_id,
                                    predicate=self._normalize_predicate(
                                    r.get('predicate', 'related_to')),
                                    object_id=object_id,
                                    source_id=finding.id,
                                    confidence=confidence,
                                )

                                # Check for contradictions
                                contradiction = self._check_contradiction(relation)
                                if contradiction:
                                    self.contradictions.append(contradiction)
                                    self.store.add_contradiction(contradiction)
                                    result['contradictions'] += 1

                                self.store.add_relation(relation, self.session_id)
                                result['relations_count'] += 1

        except Exception:
            pass

        return result

    async def _extract_atomic_facts(self, finding: KGFinding) -> list[str]:
        """Break finding into minimal, self-contained atomic facts."""
        prompt = f"""Extract atomic facts from this research finding.
Each atomic fact should be:
- Self-contained (understandable without context)
- Minimal (single piece of information)
- Factual (not opinion unless attributed)

Finding: {finding.content}

Return as JSON array of strings.
Example: ["Fact 1", "Fact 2", "Fact 3"]"""

        response = await self.llm_callback(prompt)
        return self._parse_json_array(response)

    async def _extract_entities(self, text: str, source_id: str) -> list[Entity]:
        """Extract entities from text using LLM."""
        entity_types_list = "\n".join([f"- {k}: {v}" for k, v in list(ENTITY_TYPES.items())[:8]])

        prompt = f"""Extract key entities from this text.

Text: {text}

Entity Types:
{entity_types_list}

For each entity provide:
- name: The canonical name (singular, consistent capitalization)
- type: One of the entity types above
- aliases: Other names/mentions that refer to this entity

Return as JSON array:
[
  {{"name": "Entity name", "type": "CONCEPT", "aliases": ["alias1"]}}
]"""

        response = await self.llm_callback(prompt)
        entities_data = self._parse_json_array(response)

        entities = []
        for e in entities_data:
            if not isinstance(e, dict):
                continue

            entity = Entity(
                id=self._generate_id(),
                name=e.get('name', ''),
                entity_type=e.get('type', 'CONCEPT').upper(),
                aliases=e.get('aliases', []),
                sources=[source_id],
            )
            entities.append(entity)

        return entities

    async def _resolve_entity(self, entity: Entity) -> Entity:
        """Resolve entity against existing global set.

        Uses name matching (embedding similarity if available).
        """
        # First check exact name match
        name_key = entity.name.lower().strip()
        if name_key in self.entity_by_name:
            existing_id = self.entity_by_name[name_key]
            existing = self.store.get_entity(existing_id)
            if existing:
                # Merge: add this as a source
                if entity.sources:
                    existing.sources = list(set(existing.sources + entity.sources))
                existing.aliases = list(set(existing.aliases + entity.aliases + [entity.name]))
                # Merge source_urls from properties
                existing_urls = set(existing.properties.get('source_urls', []))
                new_urls = set(entity.properties.get('source_urls', []))
                if new_urls:
                    existing.properties['source_urls'] = list(existing_urls | new_urls)
                # Keep first description, don't overwrite
                if entity.properties.get('description'):
                    existing.properties.setdefault('description', entity.properties['description'])
                self.store.add_entity(existing, self.session_id)  # Update
                return existing

        # Check aliases
        for existing_name, existing_id in list(self.entity_by_name.items()):
            existing = self.store.get_entity(existing_id)
            if existing and entity.name.lower() in [a.lower() for a in existing.aliases]:
                # Found via alias
                if entity.sources:
                    existing.sources = list(set(existing.sources + entity.sources))
                # Merge source_urls
                existing_urls = set(existing.properties.get('source_urls', []))
                new_urls = set(entity.properties.get('source_urls', []))
                if new_urls:
                    existing.properties['source_urls'] = list(existing_urls | new_urls)
                self.store.add_entity(existing, self.session_id)
                return existing

        # No match found - add as new entity
        self.store.add_entity(entity, self.session_id)
        self.entity_by_name[name_key] = entity.id
        self.entity_by_type.setdefault(entity.entity_type, []).append(entity.id)

        # Also index aliases
        for alias in entity.aliases:
            alias_key = alias.lower().strip()
            if alias_key not in self.entity_by_name:
                self.entity_by_name[alias_key] = entity.id

        return entity

    async def _extract_relations(
        self, text: str, entities: list[Entity], source_id: str
    ) -> list[Relation]:
        """Extract relations using resolved entities as context."""
        if not entities:
            return []

        entity_context = "\n".join([
            f"- {e.name} ({e.entity_type})" for e in entities
        ])

        prompt = f"""Extract relationships between these entities from the text.

Text: {text}

Known entities:
{entity_context}

For each relationship provide:
- subject: The subject entity name (must be from known entities)
- predicate: The relationship (1-3 words max, e.g., "causes", "is part of")
- object: The object entity name (must be from known entities)
- confidence: 0.0-1.0 confidence score

Return as JSON array:
[
  {{"subject": "Entity1", "predicate": "causes", "object": "Entity2", "confidence": 0.9}}
]"""

        response = await self.llm_callback(prompt)
        relations_data = self._parse_json_array(response)

        # Map entity names to IDs
        name_to_id = {e.name.lower(): e.id for e in entities}
        for e in entities:
            for alias in e.aliases:
                name_to_id[alias.lower()] = e.id

        relations = []
        for r in relations_data:
            if not isinstance(r, dict):
                continue

            subject_name = r.get('subject', '').lower()
            object_name = r.get('object', '').lower()

            subject_id = name_to_id.get(subject_name)
            object_id = name_to_id.get(object_name)

            if subject_id and object_id and subject_id != object_id:
                relation = Relation(
                    id=self._generate_id(),
                    subject_id=subject_id,
                    predicate=self._normalize_predicate(r.get('predicate', 'related_to')),
                    object_id=object_id,
                    source_id=source_id,
                    confidence=r.get('confidence', 0.8),
                )
                relations.append(relation)

        return relations

    def _check_contradiction(self, new_relation: Relation) -> Contradiction | None:
        """Check if new relation contradicts existing relations."""
        # Look for relations with same subject and object but different predicate
        existing_relations = self.store.get_entity_relations(new_relation.subject_id)

        for existing in existing_relations.get('outgoing', []):
            if existing.get('target_id') == new_relation.object_id:
                existing_pred = existing.get('predicate', '')
                if self._predicates_contradict(existing_pred, new_relation.predicate):
                    return Contradiction(
                        id=self._generate_id(),
                        relation1_id=existing.get('relation_id', ''),
                        relation2_id=new_relation.id,
                        contradiction_type='direct',
                        description=(
                            f"Conflicting predicates: '{existing_pred}' vs '{new_relation.predicate}'"
                        ),
                        severity='high' if new_relation.confidence > 0.8 else 'medium',
                    )

        return None

    def _predicates_contradict(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        p1 = pred1.lower().strip()
        p2 = pred2.lower().strip()

        for contra_pair in self.CONTRADICTORY_PREDICATES:
            if (contra_pair[0] in p1 and contra_pair[1] in p2) or \
               (contra_pair[1] in p1 and contra_pair[0] in p2):
                return True

        return False

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to canonical form using PREDICATE_CANONICAL_MAP.

        Tries exact match, then 2-word prefix, then first word.
        Falls back to underscore-joined truncation if no canonical match.
        """
        words = predicate.lower().strip().split()[:4]
        if not words:
            return 'correlates_with'

        normalized = '_'.join(words)

        # Try exact match
        if normalized in PREDICATE_CANONICAL_MAP:
            return PREDICATE_CANONICAL_MAP[normalized]

        # Try first 2 words
        if len(words) >= 2:
            two_word = '_'.join(words[:2])
            if two_word in PREDICATE_CANONICAL_MAP:
                return PREDICATE_CANONICAL_MAP[two_word]

        # Try first word only
        if words[0] in PREDICATE_CANONICAL_MAP:
            return PREDICATE_CANONICAL_MAP[words[0]]

        # Fallback: return cleaned form (max 3 words)
        return '_'.join(words[:3])

    def _build_co_occurrence_links(
        self,
        entities: list[Entity],
        explicit_relations: list[Relation],
        finding: KGFinding,
    ) -> list[Relation]:
        """Create co_occurs_with relations between entities from the same finding
        that don't already have an explicit relation. Capped at 5 per finding."""
        if len(entities) < 2:
            return []

        # Build set of entity pairs that already have explicit relations
        linked_pairs: set[tuple[str, str]] = set()
        for rel in explicit_relations:
            linked_pairs.add((rel.subject_id, rel.object_id))
            linked_pairs.add((rel.object_id, rel.subject_id))

        co_relations: list[Relation] = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1.id == e2.id:
                    continue
                if (e1.id, e2.id) in linked_pairs:
                    continue

                relation = Relation(
                    id=self._generate_id(),
                    subject_id=e1.id,
                    predicate='co_occurs_with',
                    object_id=e2.id,
                    source_id=finding.id,
                    confidence=0.4,
                )
                co_relations.append(relation)

                if len(co_relations) >= 5:
                    return co_relations

        return co_relations

    def _generate_id(self) -> str:
        """Generate unique ID."""
        return str(uuid.uuid4())[:8]

    def _parse_json_array(self, text: str) -> list:
        """Parse JSON array from LLM response."""
        # Extract JSON from potential markdown code blocks
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing the whole response
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return []

    def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        stats = self.store.get_stats()
        stats['contradictions'] = len(self.contradictions)
        stats['indexed_names'] = len(self.entity_by_name)
        stats['fast_ner_enabled'] = self.use_fast_ner and self.fast_ner is not None
        if self.fast_ner:
            stats['fast_ner'] = self.fast_ner.get_stats()
        return stats

    async def get_kg_support_score(
        self,
        content: str,
        source_url: str | None = None,
    ) -> float:
        """Calculate KG support score for a finding.

        Returns 0-1 score based on:
        - Entity matches in the KG
        - Supporting relations in the KG

        Args:
            content: The finding content to check
            source_url: Optional source URL for context

        Returns:
            Support score between 0 and 1
        """
        # Extract entities from the content (quick extraction)
        entities = await self._quick_entity_extract(content)
        if not entities:
            return 0.0

        entity_match_count = 0
        supporting_relations = 0

        for entity_name in entities:
            # Check if entity exists in KG
            entity_key = entity_name.lower().strip()
            if entity_key in self.entity_by_name:
                entity_match_count += 1
                entity_id = self.entity_by_name[entity_key]

                # Check for supporting relations
                relations = self.store.get_entity_relations(entity_id)
                if relations:
                    supporting_relations += len(relations.get('outgoing', []))
                    supporting_relations += len(relations.get('incoming', []))

        if not entities:
            return 0.0

        # Calculate score: 60% entity matches + 40% relation support
        entity_score = min(entity_match_count / len(entities), 1.0)
        relation_score = min(supporting_relations / (len(entities) * 2), 1.0)

        return (entity_score * 0.6) + (relation_score * 0.4)

    async def _quick_entity_extract(self, content: str) -> list[str]:
        """Quick entity extraction without LLM call.

        Uses spaCy if available, otherwise simple NLP heuristics.
        """
        # Use fast NER if available (much more accurate)
        if self.use_fast_ner and self.fast_ner and self.fast_ner.enabled:
            extracted = self.fast_ner.extract(content)
            return [e.name for e in extracted[:10]]

        # Fallback to regex heuristics
        import re

        entities = []

        # Extract capitalized phrases (likely proper nouns)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(cap_pattern, content)
        entities.extend(matches)

        # Extract quoted terms
        quoted_pattern = r'"([^"]+)"|\'([^\']+)\''
        for match in re.findall(quoted_pattern, content):
            for group in match:
                if group:
                    entities.append(group)

        # Extract terms with numbers (likely specific things)
        num_pattern = r'\b([A-Za-z]+[-\s]?\d+(?:\.\d+)?)\b'
        entities.extend(re.findall(num_pattern, content))

        # Deduplicate and limit
        seen = set()
        unique_entities = []
        for e in entities:
            e_lower = e.lower()
            if e_lower not in seen and len(e) > 2:
                seen.add(e_lower)
                unique_entities.append(e)

        return unique_entities[:10]  # Limit to 10 entities

    async def check_contradictions_detailed(
        self,
        content: str,
    ) -> dict:
        """Check for contradictions with detailed results.

        Args:
            content: The finding content to check

        Returns:
            Dict with has_contradictions bool and list of contradiction details
        """
        entities = await self._quick_entity_extract(content)
        contradictions = []

        for entity_name in entities:
            entity_key = entity_name.lower().strip()
            if entity_key not in self.entity_by_name:
                continue

            entity_id = self.entity_by_name[entity_key]
            relations = self.store.get_entity_relations(entity_id)

            if not relations:
                continue

            # Check for contradictory predicates in the content
            content_lower = content.lower()
            for relation in relations.get('outgoing', []):
                predicate = relation.get('predicate', '')

                # Check if content contradicts existing relation
                for contra_pair in self.CONTRADICTORY_PREDICATES:
                    # Check if content has opposite predicate
                    if contra_pair[0] in predicate and contra_pair[1] in content_lower:
                        contradictions.append({
                            "conflicting_id": relation.get('relation_id', 'unknown'),
                            "description": f"Content suggests '{contra_pair[1]}' but KG has '{contra_pair[0]}' for {entity_name}",
                            "severity": "medium",
                        })
                    elif contra_pair[1] in predicate and contra_pair[0] in content_lower:
                        contradictions.append({
                            "conflicting_id": relation.get('relation_id', 'unknown'),
                            "description": f"Content suggests '{contra_pair[0]}' but KG has '{contra_pair[1]}' for {entity_name}",
                            "severity": "medium",
                        })

        return {
            "has_contradictions": len(contradictions) > 0,
            "contradictions": contradictions,
        }
