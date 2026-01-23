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

from .embeddings import EmbeddingService, EmbeddingConfig, get_embedding_service
from .vectorstore import VectorStore, VectorStoreConfig, Document
from .bm25 import BM25Index, BM25Config
from .reranker import Reranker, RerankerConfig, LightweightReranker
from .hybrid import HybridRetriever, HybridConfig, RetrievalResult, create_retriever
from .findings import (
    FindingsRetriever,
    FindingSearchResult,
    get_findings_retriever,
    reset_findings_retriever,
)
from .memory_integration import (
    SemanticMemoryStore,
    SemanticSearchResult,
    create_semantic_memory,
)

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
]
