"""Data models for research findings and agent communication."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FindingType(str, Enum):
    """Types of research findings."""

    FACT = "fact"
    INSIGHT = "insight"
    CONNECTION = "connection"
    SOURCE = "source"
    QUESTION = "question"
    CONTRADICTION = "contradiction"


class AgentRole(str, Enum):
    """Agent roles in the hierarchy."""

    INTERN = "intern"
    MANAGER = "manager"
    DIRECTOR = "director"


class Finding(BaseModel):
    """A single research finding from the Intern agent."""

    id: int | None = None
    session_id: str  # 7-char hex ID
    content: str
    finding_type: FindingType
    source_url: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    search_query: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    validated_by_manager: bool = False
    manager_notes: str | None = None

    topic_id: int | None = None

    # Verification fields
    verification_status: str | None = None  # verified/flagged/rejected/skipped
    verification_method: str | None = None  # cove/critic/kg_match/streaming/batch
    kg_support_score: float = 0.0  # KG corroboration score (0-1)
    original_confidence: float | None = None  # Confidence before verification calibration


class ResearchTopic(BaseModel):
    """A research topic or subtopic to investigate."""

    id: int | None = None
    session_id: str  # 7-char hex ID
    topic: str
    parent_topic_id: int | None = None
    depth: int = 0
    status: str = "pending"  # pending, in_progress, completed, blocked
    priority: int = Field(default=5, ge=1, le=10)
    assigned_at: datetime | None = None
    completed_at: datetime | None = None
    findings_count: int = 0


class ResearchSession(BaseModel):
    """A research session with time limits and goals."""

    id: str | None = None  # 7-char hex ID
    goal: str
    slug: str | None = None  # AI-generated short name for output folder
    time_limit_minutes: int = 60
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    status: str = "active"  # active, paused, completed, timeout
    total_findings: int = 0
    total_searches: int = 0
    depth_reached: int = 0


class AgentMessage(BaseModel):
    """Message passed between agents in the hierarchy."""

    id: int | None = None
    session_id: str  # 7-char hex ID
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str  # task, report, critique, question, directive
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ManagerDirective(BaseModel):
    """Directive from Manager to Intern."""

    action: str  # search, deep_dive, verify, expand, stop
    topic: str
    instructions: str
    priority: int = Field(default=5, ge=1, le=10)
    max_searches: int = 5


class InternReport(BaseModel):
    """Report from Intern to Manager."""

    topic: str
    findings: list[Finding]
    searches_performed: int
    suggested_followups: list[str]
    blockers: list[str] = Field(default_factory=list)


class ManagerReport(BaseModel):
    """Report from Manager to Director."""

    summary: str
    key_findings: list[Finding]
    topics_explored: list[str]
    topics_remaining: list[str]
    quality_assessment: str
    recommended_next_steps: list[str]
    time_elapsed_minutes: float
    time_remaining_minutes: float
