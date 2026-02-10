"""Hybrid retrieval system combining semantic and lexical search.

This module provides high-quality retrieval combining:
- Semantic search (BGE embeddings + ChromaDB)
- Lexical search (BM25)
- Reciprocal Rank Fusion for hybrid combination
- Cross-encoder reranking for final quality boost

Usage:
    from src.retrieval import HybridRetriever, create_retriever

    # Quick setup
    retriever = create_retriever()
    retriever.add_texts(["doc1", "doc2", "doc3"])
    results = retriever.search("my query")

    # For research findings
    from src.retrieval import FindingsRetriever
    findings_retriever = FindingsRetriever()
    findings_retriever.add_finding(finding, session_id)
    results = findings_retriever.search("quantum computing")
"""

from .bm25 import BM25Config, BM25Index
from .deduplication import (
    DeduplicationConfig,
    DeduplicationResult,
    FindingDeduplicator,
    get_deduplicator,
    reset_deduplicator,
)
from .embeddings import EmbeddingConfig, EmbeddingService, get_embedding_service
from .findings import (
    FindingSearchResult,
    FindingsRetriever,
    get_findings_retriever,
    reset_findings_retriever,
)
from .hybrid import HybridConfig, HybridRetriever, RetrievalResult, create_retriever
from .memory_integration import (
    SemanticMemoryStore,
    SemanticSearchResult,
    create_semantic_memory,
)
from .query_expansion import (
    ExpandedQuery,
    QueryExpander,
    QueryExpansionConfig,
    QueryExpansionResult,
    merge_search_results,
)
from .reranker import LightweightReranker, Reranker, RerankerConfig
from .vectorstore import Document, VectorStore, VectorStoreConfig

__all__ = [
    # Core components
    "EmbeddingService",
    "EmbeddingConfig",
    "get_embedding_service",
    "VectorStore",
    "VectorStoreConfig",
    "Document",
    "BM25Index",
    "BM25Config",
    "Reranker",
    "RerankerConfig",
    "LightweightReranker",
    # Hybrid retriever
    "HybridRetriever",
    "HybridConfig",
    "RetrievalResult",
    "create_retriever",
    # Findings retriever
    "FindingsRetriever",
    "FindingSearchResult",
    "get_findings_retriever",
    "reset_findings_retriever",
    # Memory integration
    "SemanticMemoryStore",
    "SemanticSearchResult",
    "create_semantic_memory",
    # Deduplication
    "FindingDeduplicator",
    "DeduplicationConfig",
    "DeduplicationResult",
    "get_deduplicator",
    "reset_deduplicator",
    # Query expansion
    "QueryExpander",
    "QueryExpansionConfig",
    "QueryExpansionResult",
    "ExpandedQuery",
    "merge_search_results",
]
