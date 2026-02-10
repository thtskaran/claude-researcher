"""Findings retriever for research-specific semantic search.

This module provides specialized retrieval for research findings,
integrated with the knowledge graph and manager agent workflow.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..models.findings import Finding, FindingType
from .embeddings import EmbeddingConfig
from .hybrid import HybridConfig, HybridRetriever, RetrievalResult
from .reranker import RerankerConfig
from .vectorstore import Document


@dataclass
class FindingSearchResult:
    """Result from finding search."""

    finding: Finding
    score: float
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    reranked: bool = False


class FindingsRetriever:
    """Hybrid retriever specialized for research findings.

    Provides:
    - Semantic search over findings
    - Finding type filtering
    - Source URL deduplication
    - Confidence-weighted ranking
    - Cross-session retrieval

    Usage:
        retriever = FindingsRetriever(persist_dir=".findings_index")

        # Index findings as they come in
        retriever.add_finding(finding, session_id, topic_id)

        # Search for relevant findings
        results = retriever.search("quantum computing applications", limit=10)
    """

    def __init__(
        self,
        persist_dir: str = ".findings_retrieval",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-large",
    ):
        """Initialize findings retriever.

        Args:
            persist_dir: Directory for persisting indices
            embedding_model: Sentence transformer model for embeddings
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Cross-encoder model for reranking
        """
        config = HybridConfig(
            persist_directory=persist_dir,
            embedding=EmbeddingConfig(
                model_name=embedding_model,
                # Optimize query prefix for research queries
                query_prefix="Represent this research query for finding relevant findings: ",
            ),
            reranker=RerankerConfig(model_name=reranker_model) if use_reranker else None,
            use_reranker=use_reranker,
            # Retrieval parameters optimized for findings
            initial_k=100,  # Get more candidates for better coverage
            rerank_k=30,    # Rerank top 30
            final_k=10,     # Return top 10
            semantic_weight=0.6,  # Slightly favor semantic for research
        )

        self._retriever = HybridRetriever(config)
        self._finding_cache: dict[str, Finding] = {}  # id -> Finding

    def add_finding(
        self,
        finding: Finding,
        session_id: str,
        topic_id: str | None = None,
    ) -> str:
        """Add a finding to the index.

        Args:
            finding: The finding to index
            session_id: Current research session ID
            topic_id: Optional topic ID

        Returns:
            Finding ID
        """
        # Create searchable content (combine claim with context)
        content = finding.content
        if hasattr(finding, 'supporting_quote') and finding.supporting_quote:
            content = f"{finding.content}\n\nContext: {finding.supporting_quote}"

        # Build metadata for filtering
        metadata = {
            "session_id": session_id,
            "finding_type": finding.finding_type.value if hasattr(finding.finding_type, 'value') else str(finding.finding_type),
            "confidence": finding.confidence,
            "source_url": finding.source_url or "",
            "created_at": datetime.now().isoformat(),
        }

        if topic_id:
            metadata["topic_id"] = topic_id

        if hasattr(finding, 'source_title') and finding.source_title:
            metadata["source_title"] = finding.source_title

        # Generate ID if not present (ensure it's a string for ChromaDB)
        finding_id = getattr(finding, 'id', None)
        if not finding_id:
            import hashlib
            finding_id = hashlib.md5(content.encode()).hexdigest()[:12]
        finding_id = str(finding_id)  # ChromaDB requires string IDs

        # Cache the finding object
        self._finding_cache[finding_id] = finding

        # Add to retriever
        self._retriever.add_texts(
            texts=[content],
            metadatas=[metadata],
            ids=[finding_id],
        )

        return finding_id

    def add_findings(
        self,
        findings: list[Finding],
        session_id: str,
        topic_id: str | None = None,
    ) -> list[str]:
        """Add multiple findings efficiently.

        Args:
            findings: Findings to index
            session_id: Current research session ID
            topic_id: Optional topic ID

        Returns:
            List of finding IDs
        """
        if not findings:
            return []

        documents = []
        ids = []

        for finding in findings:
            # Create searchable content
            content = finding.content
            if hasattr(finding, 'supporting_quote') and finding.supporting_quote:
                content = f"{finding.content}\n\nContext: {finding.supporting_quote}"

            # Build metadata
            metadata = {
                "session_id": session_id,
                "finding_type": finding.finding_type.value if hasattr(finding.finding_type, 'value') else str(finding.finding_type),
                "confidence": finding.confidence,
                "source_url": finding.source_url or "",
                "created_at": datetime.now().isoformat(),
            }

            if topic_id:
                metadata["topic_id"] = topic_id

            if hasattr(finding, 'source_title') and finding.source_title:
                metadata["source_title"] = finding.source_title

            # Generate ID (ensure it's a string for ChromaDB)
            finding_id = getattr(finding, 'id', None)
            if not finding_id:
                import hashlib
                finding_id = hashlib.md5(content.encode()).hexdigest()[:12]
            finding_id = str(finding_id)  # ChromaDB requires string IDs

            # Cache
            self._finding_cache[finding_id] = finding

            documents.append(Document(
                id=finding_id,
                content=content,
                metadata=metadata,
            ))
            ids.append(finding_id)

        self._retriever.add(documents)
        return ids

    def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
        finding_types: list[FindingType] | None = None,
        min_confidence: float | None = None,
        use_reranker: bool | None = None,
    ) -> list[FindingSearchResult]:
        """Search for relevant findings.

        Args:
            query: Search query
            limit: Maximum results
            session_id: Optional filter by session
            finding_types: Optional filter by finding types
            min_confidence: Optional minimum confidence threshold
            use_reranker: Override default reranker setting

        Returns:
            List of FindingSearchResult sorted by relevance
        """
        # Build filter
        filter_dict = {}
        if session_id:
            filter_dict["session_id"] = session_id

        # Note: ChromaDB doesn't support IN queries directly,
        # so we'll filter finding types post-retrieval
        chroma_filter = filter_dict if filter_dict else None

        # Get more results if we need to filter
        fetch_k = limit * 3 if (finding_types or min_confidence) else limit

        # Run hybrid search
        results = self._retriever.search(
            query=query,
            k=fetch_k,
            filter=chroma_filter,
            use_reranker=use_reranker,
        )

        # Convert and filter results
        search_results = []

        for result in results:
            finding_id = result.document.id

            # Get finding from cache or reconstruct
            finding = self._finding_cache.get(finding_id)
            if not finding:
                # Reconstruct from document metadata
                finding = self._reconstruct_finding(result)

            # Apply filters
            if finding_types:
                finding_type = finding.finding_type
                if hasattr(finding_type, 'value'):
                    if finding_type not in finding_types:
                        continue
                else:
                    type_str = str(finding_type)
                    if not any(ft.value == type_str or ft.name == type_str for ft in finding_types):
                        continue

            if min_confidence and finding.confidence < min_confidence:
                continue

            search_results.append(FindingSearchResult(
                finding=finding,
                score=result.score,
                bm25_rank=result.bm25_rank,
                semantic_rank=result.semantic_rank,
                reranked=result.reranker_score is not None,
            ))

            if len(search_results) >= limit:
                break

        return search_results

    def _reconstruct_finding(self, result: RetrievalResult) -> Finding:
        """Reconstruct a Finding object from retrieval result."""
        metadata = result.document.metadata

        # Determine finding type
        type_str = metadata.get("finding_type", "FACT")
        try:
            finding_type = FindingType(type_str)
        except (ValueError, KeyError):
            finding_type = FindingType.FACT

        return Finding(
            content=result.document.content.split("\n\nContext:")[0],  # Extract main content
            finding_type=finding_type,
            confidence=float(metadata.get("confidence", 0.5)),
            source_url=metadata.get("source_url"),
            source_title=metadata.get("source_title"),
        )

    def find_similar(
        self,
        finding: Finding,
        limit: int = 5,
        exclude_self: bool = True,
        session_id: str | None = None,
    ) -> list[FindingSearchResult]:
        """Find findings similar to a given finding.

        Useful for:
        - Deduplication
        - Finding supporting evidence
        - Detecting contradictions

        Args:
            finding: Finding to find similar ones for
            limit: Maximum results
            exclude_self: Whether to exclude the input finding
            session_id: Optional filter by session

        Returns:
            List of similar findings
        """
        results = self.search(
            query=finding.content,
            limit=limit + (1 if exclude_self else 0),
            session_id=session_id,
        )

        if exclude_self:
            # Remove the input finding if it appears in results
            results = [
                r for r in results
                if r.finding.content != finding.content
            ][:limit]

        return results

    def find_by_source(
        self,
        source_url: str,
        limit: int = 20,
    ) -> list[Finding]:
        """Find all findings from a specific source.

        Args:
            source_url: Source URL to search for
            limit: Maximum results

        Returns:
            List of findings from that source
        """
        # Use metadata filter
        results = self._retriever.search(
            query="",  # Empty query, rely on filter
            k=limit,
            filter={"source_url": source_url},
            use_reranker=False,  # No need for reranking with filter-only
        )

        findings = []
        for result in results:
            finding = self._finding_cache.get(result.document.id)
            if finding:
                findings.append(finding)
            else:
                findings.append(self._reconstruct_finding(result))

        return findings

    def get_session_findings(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Finding]:
        """Get all findings for a session.

        Args:
            session_id: Session ID
            limit: Maximum results

        Returns:
            List of findings
        """
        results = self._retriever.search(
            query="research findings",  # Generic query
            k=limit,
            filter={"session_id": session_id},
            use_reranker=False,
        )

        findings = []
        for result in results:
            finding = self._finding_cache.get(result.document.id)
            if finding:
                findings.append(finding)
            else:
                findings.append(self._reconstruct_finding(result))

        return findings

    def delete_session(self, session_id: str) -> int:
        """Delete all findings for a session.

        Args:
            session_id: Session to delete

        Returns:
            Number of findings deleted
        """
        # Get findings for this session
        findings = self.get_session_findings(session_id, limit=10000)

        if not findings:
            return 0

        # Delete from retriever
        # Note: ChromaDB doesn't support delete by filter directly
        # We need to delete by IDs
        ids_to_delete = []
        for finding in findings:
            finding_id = getattr(finding, 'id', None)
            if finding_id:
                ids_to_delete.append(finding_id)
                self._finding_cache.pop(finding_id, None)

        if ids_to_delete:
            self._retriever.delete(ids_to_delete)

        return len(ids_to_delete)

    def count(self) -> int:
        """Get total number of indexed findings."""
        return self._retriever.count()

    def stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "total_findings": self.count(),
            "cached_findings": len(self._finding_cache),
            "retriever": self._retriever.stats(),
        }

    def clear(self) -> None:
        """Clear all indexed findings."""
        self._retriever.clear()
        self._finding_cache.clear()


# Singleton instance for global access
_findings_retriever: FindingsRetriever | None = None


def get_findings_retriever(
    persist_dir: str = ".findings_retrieval",
    **kwargs,
) -> FindingsRetriever:
    """Get or create the global findings retriever.

    Args:
        persist_dir: Directory for persistence
        **kwargs: Additional arguments for FindingsRetriever

    Returns:
        FindingsRetriever instance
    """
    global _findings_retriever
    if _findings_retriever is None:
        _findings_retriever = FindingsRetriever(persist_dir=persist_dir, **kwargs)
    return _findings_retriever


def reset_findings_retriever() -> None:
    """Reset the global findings retriever."""
    global _findings_retriever
    _findings_retriever = None
