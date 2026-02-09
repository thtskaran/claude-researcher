"""
Pydantic models for API request/response validation.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ResearchSessionCreate(BaseModel):
    """Request model for creating a new research session."""
    goal: str = Field(..., min_length=1, description="Research goal/question")
    time_limit: int = Field(default=60, ge=1, le=240, description="Time limit in minutes")
    autonomous: bool = Field(default=False, description="Run without user interaction")
    db_path: Optional[str] = Field(default=None, description="Custom database path")


class ResearchSessionResponse(BaseModel):
    """Response model for research session."""
    session_id: str
    goal: str
    time_limit: int
    status: str  # 'running', 'completed', 'paused', 'error'
    created_at: datetime
    completed_at: Optional[datetime] = None


class AgentEvent(BaseModel):
    """WebSocket event from agents."""
    session_id: str
    event_type: str  # 'thinking', 'action', 'finding', 'synthesis', 'error'
    agent: str  # 'director', 'manager', 'intern'
    timestamp: datetime
    data: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
