"""User interaction module for research sessions.

This module provides interactive features during research:
- Pre-research clarification questions
- Async mid-research questions with timeout
- User message queue for injecting guidance
"""

from .models import (
    ClarificationQuestion,
    ClarifiedGoal,
    PendingQuestion,
    UserMessage,
)
from .config import InteractionConfig
from .handler import UserInteraction
from .listener import InputListener

__all__ = [
    # Models
    "ClarificationQuestion",
    "ClarifiedGoal",
    "PendingQuestion",
    "UserMessage",
    # Config
    "InteractionConfig",
    # Handler
    "UserInteraction",
    # Listener
    "InputListener",
]
