"""
Pydantic models for API request/response validation.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResearchSessionCreate(BaseModel):
    """Request model for creating a new research session."""
    goal: str = Field(..., min_length=1, description="Research goal/question")
    max_iterations: int = Field(default=5, ge=1, le=30, description="Number of research iterations")
    autonomous: bool = Field(default=False, description="Run without user interaction")
    db_path: str | None = Field(default=None, description="Custom database path")


class ResearchSessionResponse(BaseModel):
    """Response model for research session."""
    session_id: str
    goal: str
    max_iterations: int = 5
    time_limit: int = 0  # Backward compat
    status: str  # 'running', 'completed', 'paused', 'crashed', 'error'
    created_at: datetime
    completed_at: datetime | None = None
    elapsed_seconds: float = 0.0
    paused_at: datetime | None = None
    iteration_count: int = 0


class AgentEvent(BaseModel):
    """WebSocket event from agents."""
    session_id: str
    event_type: str  # 'thinking', 'action', 'finding', 'synthesis', 'error'
    agent: str  # 'director', 'manager', 'intern'
    timestamp: datetime
    data: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
