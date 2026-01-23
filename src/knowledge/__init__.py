"""Knowledge graph module for real-time research synthesis."""

from .graph import IncrementalKnowledgeGraph
from .store import HybridKnowledgeGraphStore
from .models import Entity, Relation, KGFinding, Contradiction, KnowledgeGap
from .query import ManagerQueryInterface
from .credibility import CredibilityScorer
from .visualize import KnowledgeGraphVisualizer

__all__ = [
    "IncrementalKnowledgeGraph",
    "HybridKnowledgeGraphStore",
    "Entity",
    "Relation",
    "KGFinding",
    "Contradiction",
    "KnowledgeGap",
    "ManagerQueryInterface",
    "CredibilityScorer",
    "KnowledgeGraphVisualizer",
]
