"""Knowledge graph processing extracted from ManagerAgent.

Handles converting findings to KG entities/relations and indexing
for hybrid retrieval. Reduces ManagerAgent's responsibilities.
"""

import logging
from collections.abc import Callable

from ..knowledge import IncrementalKnowledgeGraph, KGFinding
from ..models.findings import Finding
from ..retrieval import FindingsRetriever

logger = logging.getLogger(__name__)

# Type alias for the log callback
LogCallback = Callable[[str, str], None]


class KGProcessor:
    """Processes findings into the knowledge graph and retrieval index.

    Extracted from ManagerAgent to reduce its size and isolate KG concerns.
    """

    def __init__(
        self,
        knowledge_graph: IncrementalKnowledgeGraph,
        findings_retriever: FindingsRetriever,
        log: LogCallback | None = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.findings_retriever = findings_retriever
        self._log = log or (lambda msg, style="": None)

    async def process_findings(
        self, findings: list[Finding], session_id: str
    ) -> None:
        """Process findings into the knowledge graph and hybrid retrieval index.

        Uses batch processing for speed (multiple findings per LLM call)
        while still building the full KG that agents can query during research.
        Also indexes findings for semantic search via hybrid retrieval.

        Args:
            findings: List of findings to process
            session_id: Current research session ID
        """
        if not findings:
            return

        self._log(
            f"[KG] Processing {len(findings)} findings into knowledge graph",
            "dim",
        )

        # Index findings for hybrid retrieval (semantic + lexical search)
        self._index_for_retrieval(findings, session_id)

        # Convert and add to knowledge graph
        kg_findings = self._convert_to_kg_findings(findings)
        await self._add_to_graph(kg_findings)

    def _index_for_retrieval(
        self, findings: list[Finding], session_id: str
    ) -> None:
        """Index findings for semantic + lexical search."""
        try:
            self.findings_retriever.add_findings(
                findings=findings,
                session_id=session_id,
            )
            self._log(
                f"[RETRIEVAL] Indexed {len(findings)} findings for semantic search",
                "dim",
            )
        except Exception as e:
            self._log(f"[RETRIEVAL] Error indexing findings: {e}", "yellow")
            logger.warning("Retrieval indexing error: %s", e, exc_info=True)

    def _convert_to_kg_findings(self, findings: list[Finding]) -> list[KGFinding]:
        """Convert Finding models to KGFinding format for the knowledge graph."""
        kg_findings = []
        for finding in findings:
            try:
                kg_finding = KGFinding(
                    id=str(finding.id or hash(finding.content)),
                    content=finding.content,
                    source_url=finding.source_url or "",
                    source_title=(
                        finding.source_url.split("/")[-1] if finding.source_url else ""
                    ),
                    timestamp=finding.created_at.isoformat(),
                    credibility_score=finding.confidence,
                    finding_type=finding.finding_type.value,
                    search_query=finding.search_query,
                )
                kg_findings.append(kg_finding)
            except Exception as e:
                self._log(f"[KG] Error converting finding: {e}", "dim")
                logger.warning("KG conversion error: %s", e, exc_info=True)
        return kg_findings

    async def _add_to_graph(self, kg_findings: list[KGFinding]) -> None:
        """Add KG findings to the knowledge graph (batch or individual)."""
        if not kg_findings:
            return

        if len(kg_findings) > 3:
            result = await self.knowledge_graph.add_findings_batch(
                kg_findings, batch_size=5
            )
            self._log(
                f"[KG] Extracted {result['total_entities']} entities, "
                f"{result['total_relations']} relations",
                "dim",
            )
            if result["total_contradictions"] > 0:
                self._log(
                    f"[KG] Contradictions detected: {result['total_contradictions']}",
                    "yellow",
                )
        else:
            for kg_finding in kg_findings:
                try:
                    result = await self.knowledge_graph.add_finding(
                        kg_finding, fast_mode=True
                    )
                    if result.get("contradictions_found", 0) > 0:
                        self._log(
                            f"[KG] Contradiction detected: "
                            f"{result['contradictions_found']} conflicts",
                            "yellow",
                        )
                except Exception as e:
                    self._log(f"[KG] Error processing finding: {e}", "dim")
                    logger.warning("KG processing error: %s", e, exc_info=True)
