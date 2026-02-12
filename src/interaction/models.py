"""Data models for user interaction features."""

from datetime import datetime

from pydantic import BaseModel, Field


class ClarificationQuestion(BaseModel):
    """A question to ask the user before starting research."""
    id: int
    question: str
    options: list[str] = Field(default_factory=list)
    default: str | None = None
    category: str = "general"


class ClarifiedGoal(BaseModel):
    """The result of the clarification process."""
    original: str
    clarifications: dict[int, str] = Field(default_factory=dict)
    enriched_context: str
    skipped: bool = False


class PendingQuestion(BaseModel):
    """A question waiting for user response during research."""
    text: str
    context: str = ""
    asked_at: datetime = Field(default_factory=datetime.now)
    options: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60


class UserMessage(BaseModel):
    """A message injected by the user during research."""
    content: str
    injected_at: datetime = Field(default_factory=datetime.now)
    processed: bool = False
