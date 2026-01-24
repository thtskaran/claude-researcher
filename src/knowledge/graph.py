"""Real-time incremental knowledge graph construction."""

import asyncio
import json
import re
import uuid
from typing import Optional, Callable, Any
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .models import Entity, Relation, KGFinding, Contradiction, ENTITY_TYPES
from .store import HybridKnowledgeGraphStore
from .credibility import CredibilityScorer
from .fast_ner import FastNER, get_fast_ner


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
        store: Optional[HybridKnowledgeGraphStore] = None,
        similarity_threshold: float = 0.7,
        use_fast_ner: bool = True,
    ):
        """Initialize the incremental knowledge graph.

        Args:
            llm_callback: Async function to call LLM for extraction
            store: Storage backend (creates new if not provided)
            similarity_threshold: Threshold for entity matching (0.7 = 70% similar)
            use_fast_ner: Use spaCy for fast NER (with LLM fallback for domain types)
        """
        self.llm_callback = llm_callback
        self.store = store or HybridKnowledgeGraphStore()
        self.similarity_threshold = similarity_threshold
        self.credibility_scorer = CredibilityScorer()

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

            # Score source credibility
            credibility = self.credibility_scorer.score_source(
                finding.source_url,
                finding.timestamp,
            )
            finding.credibility_score = credibility.score

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
                    self.store.add_relation(relation)
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

            # Resolve and add entities
            for entity in entities[:5]:
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
            )

            for relation in relations:
                # Check for contradictions
                contradiction = self._check_contradiction(relation)
                if contradiction:
                    self.contradictions.append(contradiction)
                    self.store.add_contradiction(contradiction)
                    result['contradictions_found'] += 1

                self.store.add_relation(relation)
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

        entity_types_list = ", ".join(list(ENTITY_TYPES.keys())[:8])

        prompt = f"""Extract key entities and their relationships from this research finding.

Finding: {finding.content}

Entity Types: {entity_types_list}

Return as JSON with entities and relations:
{{
  "entities": [
    {{"name": "Entity name", "type": "CONCEPT"}}
  ],
  "relations": [
    {{"subject": "Entity1", "predicate": "causes", "object": "Entity2"}}
  ]
}}

Keep it concise: max 5 entities, max 3 relations. Focus on the most important facts."""

        try:
            response = await self.llm_callback(prompt)

            # Parse response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())

                # Process entities
                for e in data.get('entities', [])[:5]:
                    if isinstance(e, dict) and e.get('name'):
                        entity = Entity(
                            id=self._generate_id(),
                            name=e['name'],
                            entity_type=e.get('type', 'CONCEPT').upper(),
                            aliases=[],
                            sources=[finding.id],
                        )
                        resolved = await self._resolve_entity(entity)
                        result['entities'].append(resolved)

                # Process relations
                name_to_id = {e.name.lower(): e.id for e in result['entities']}
                for r in data.get('relations', [])[:3]:
                    if isinstance(r, dict):
                        subject_id = name_to_id.get(r.get('subject', '').lower())
                        object_id = name_to_id.get(r.get('object', '').lower())

                        if subject_id and object_id and subject_id != object_id:
                            relation = Relation(
                                id=self._generate_id(),
                                subject_id=subject_id,
                                predicate=self._normalize_predicate(r.get('predicate', 'related_to')),
                                object_id=object_id,
                                source_id=finding.id,
                                confidence=0.8,
                            )

                            # Check for contradictions
                            contradiction = self._check_contradiction(relation)
                            if contradiction:
                                self.contradictions.append(contradiction)
                                self.store.add_contradiction(contradiction)
                                result['contradictions_found'] += 1

                            self.store.add_relation(relation)
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
    ) -> list[Relation]:
        """Fast relation extraction for pre-extracted entities.

        Uses a compact prompt focused only on relations.
        """
        if len(entities) < 2:
            return []

        entity_names = ", ".join([e.name for e in entities[:5]])

        prompt = f"""Given these entities: {entity_names}

Extract relationships from this text:
{text[:500]}

Return JSON array (max 3 relations):
[{{"subject": "Entity1", "predicate": "verb phrase", "object": "Entity2"}}]"""

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

            relations = []
            for r in data[:3]:
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
                        confidence=0.8,
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

        entity_types_list = ", ".join(list(ENTITY_TYPES.keys())[:8])

        # Format findings for batch processing
        findings_text = "\n\n".join([
            f"Finding {i+1}: {f.content[:500]}"
            for i, f in enumerate(findings)
        ])

        prompt = f"""Extract key entities and relationships from these research findings.

{findings_text}

Entity Types: {entity_types_list}

Return as JSON array, one object per finding:
[
  {{
    "finding": 1,
    "entities": [{{"name": "Entity", "type": "CONCEPT"}}],
    "relations": [{{"subject": "Entity1", "predicate": "causes", "object": "Entity2"}}]
  }}
]

Keep concise: max 3 entities and 2 relations per finding."""

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

                    # Process entities
                    entities = []
                    for e in item.get('entities', [])[:3]:
                        if isinstance(e, dict) and e.get('name'):
                            entity = Entity(
                                id=self._generate_id(),
                                name=e['name'],
                                entity_type=e.get('type', 'CONCEPT').upper(),
                                aliases=[],
                                sources=[finding.id],
                            )
                            resolved = await self._resolve_entity(entity)
                            entities.append(resolved)
                            result['entities_count'] += 1

                    # Process relations
                    name_to_id = {e.name.lower(): e.id for e in entities}
                    for r in item.get('relations', [])[:2]:
                        if isinstance(r, dict):
                            subject_id = name_to_id.get(r.get('subject', '').lower())
                            object_id = name_to_id.get(r.get('object', '').lower())

                            if subject_id and object_id and subject_id != object_id:
                                relation = Relation(
                                    id=self._generate_id(),
                                    subject_id=subject_id,
                                    predicate=self._normalize_predicate(r.get('predicate', 'related_to')),
                                    object_id=object_id,
                                    source_id=finding.id,
                                    confidence=0.8,
                                )

                                # Check for contradictions
                                contradiction = self._check_contradiction(relation)
                                if contradiction:
                                    self.contradictions.append(contradiction)
                                    self.store.add_contradiction(contradiction)
                                    result['contradictions'] += 1

                                self.store.add_relation(relation)
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
                self.store.add_entity(existing)  # Update
                return existing

        # Check aliases
        for existing_name, existing_id in list(self.entity_by_name.items()):
            existing = self.store.get_entity(existing_id)
            if existing and entity.name.lower() in [a.lower() for a in existing.aliases]:
                # Found via alias
                if entity.sources:
                    existing.sources = list(set(existing.sources + entity.sources))
                self.store.add_entity(existing)
                return existing

        # No match found - add as new entity
        self.store.add_entity(entity)
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

    def _check_contradiction(self, new_relation: Relation) -> Optional[Contradiction]:
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
        """Normalize predicate to canonical form."""
        # Lowercase, strip, limit to 3 words
        words = predicate.lower().strip().split()[:3]
        return '_'.join(words) if words else 'related_to'

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
        source_url: Optional[str] = None,
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
