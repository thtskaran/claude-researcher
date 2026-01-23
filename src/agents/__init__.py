"""Hierarchical research agents."""

from .base import BaseAgent, AgentConfig, ModelRouter
from .intern import InternAgent
from .manager import ManagerAgent
from .director import DirectorAgent
from .parallel import ParallelInternPool, ParallelResearchResult

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "ModelRouter",
    "InternAgent",
    "ManagerAgent",
    "DirectorAgent",
    "ParallelInternPool",
    "ParallelResearchResult",
]
