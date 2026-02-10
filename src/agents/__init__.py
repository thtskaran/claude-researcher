"""Hierarchical research agents."""

from .base import AgentConfig, BaseAgent, ModelRouter
from .director import DirectorAgent
from .intern import InternAgent
from .manager import ManagerAgent
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
