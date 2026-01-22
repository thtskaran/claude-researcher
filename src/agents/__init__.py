"""Hierarchical research agents."""

from .base import BaseAgent, AgentConfig
from .intern import InternAgent
from .manager import ManagerAgent
from .director import DirectorAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "InternAgent",
    "ManagerAgent",
    "DirectorAgent",
]
