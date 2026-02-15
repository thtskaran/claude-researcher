"""Decision logging for explainable agent reasoning.

This module provides async batched writes for agent decision logging,
enabling post-hoc analysis of agent reasoning without impacting performance.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..storage.database import ResearchDatabase


class DecisionType(Enum):
    """Types of agent decisions that can be logged."""

    # Manager decisions
    SYNTHESIS_TRIGGER = "synthesis_trigger"
    TOPIC_SELECTION = "topic_selection"
    DIRECTIVE_CREATE = "directive_create"

    # Intern decisions
    STOP_SEARCHING = "stop_searching"
    QUERY_EXPAND = "query_expand"
    DEDUP_SKIP = "dedup_skip"

    # Query expansion decisions
    MULTI_QUERY_GEN = "multi_query_gen"
    CONTEXTUAL_EXPAND = "contextual_expand"
    SUFFICIENCY_CHECK = "sufficiency_check"
    QUERY_MERGE = "query_merge"

    # Common decisions
    ERROR_RECOVERY = "error_recovery"
    PRIORITY_CHANGE = "priority_change"


@dataclass
class AgentDecisionRecord:
    """Record of an agent decision for audit trail."""

    session_id: str
    agent_role: str  # "manager" or "intern"
    decision_type: DecisionType
    decision_outcome: str
    reasoning: str | None = None
    inputs: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    iteration: int | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "session_id": self.session_id,
            "agent_role": self.agent_role,
            "decision_type": self.decision_type.value,
            "decision_outcome": self.decision_outcome,
            "reasoning": self.reasoning[:500] if self.reasoning else None,
            "inputs_json": json.dumps(self.inputs) if self.inputs else None,
            "metrics_json": json.dumps(self.metrics) if self.metrics else None,
            "iteration": self.iteration,
            "created_at": self.created_at.isoformat(),
        }


class DecisionLogger:
    """Async batched decision logger for agent explainability.

    Features:
    - Async fire-and-forget writes via asyncio.create_task()
    - Batched DB writes (every N records or timeout)
    - Reasoning truncated to 500 chars to limit DB size
    - Thread-safe queue access
    """

    def __init__(
        self,
        db: "ResearchDatabase",
        batch_size: int = 10,
        flush_interval_seconds: float = 1.0,
    ):
        """Initialize the decision logger.

        Args:
            db: Database connection for persistence
            batch_size: Number of records to batch before writing
            flush_interval_seconds: Time interval to force flush
        """
        self.db = db
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds

        self._queue: list[AgentDecisionRecord] = []
        self._queue_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop the logger and flush remaining records."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()

    async def log(self, record: AgentDecisionRecord) -> None:
        """Log a decision record (async fire-and-forget).

        This method adds the record to the queue and triggers a flush
        if the batch size is reached. The actual DB write happens
        asynchronously in the background.

        Args:
            record: The decision record to log
        """
        async with self._queue_lock:
            self._queue.append(record)

            # Trigger immediate flush if batch size reached
            if len(self._queue) >= self.batch_size:
                asyncio.create_task(self._flush())

    def log_sync(self, record: AgentDecisionRecord) -> None:
        """Synchronous log method for use in non-async contexts.

        Creates a task to handle the async logging.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.log(record))
        except RuntimeError:
            # No running loop, queue directly (will be flushed later)
            self._queue.append(record)

    async def _flush_loop(self) -> None:
        """Background loop to periodically flush the queue."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception:
                # Don't let errors stop the flush loop
                logger.debug("Decision logger flush failed", exc_info=True)

    async def _flush(self) -> None:
        """Flush all queued records to the database."""
        async with self._queue_lock:
            if not self._queue:
                return

            records = self._queue.copy()
            self._queue.clear()

        if not records:
            return

        try:
            # Batch write to database
            await self.db.save_agent_decisions_batch([r.to_dict() for r in records])
        except Exception:
            # Re-queue on failure (best effort)
            logger.debug("Decision flush error", exc_info=True)
            async with self._queue_lock:
                self._queue.extend(records)

    async def log_decision(
        self,
        session_id: str,
        agent_role: str,
        decision_type: DecisionType,
        decision_outcome: str,
        reasoning: str | None = None,
        inputs: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        iteration: int | None = None,
    ) -> None:
        """Convenience method to log a decision.

        Args:
            session_id: The research session ID
            agent_role: "manager" or "intern"
            decision_type: Type of decision
            decision_outcome: The outcome/choice made
            reasoning: Agent's reasoning (truncated to 500 chars)
            inputs: Key inputs that influenced the decision
            metrics: Metrics at decision time
            iteration: Current iteration number
        """
        record = AgentDecisionRecord(
            session_id=session_id,
            agent_role=agent_role,
            decision_type=decision_type,
            decision_outcome=decision_outcome,
            reasoning=reasoning,
            inputs=inputs,
            metrics=metrics,
            iteration=iteration,
        )
        await self.log(record)

    def get_queue_size(self) -> int:
        """Get current queue size (for monitoring)."""
        return len(self._queue)


# Global instance for shared access
_decision_logger: DecisionLogger | None = None


def get_decision_logger(db: Optional["ResearchDatabase"] = None) -> DecisionLogger | None:
    """Get or create the global decision logger.

    Args:
        db: Database connection (required for first call)

    Returns:
        The decision logger instance, or None if db not provided and not initialized
    """
    global _decision_logger

    if _decision_logger is None and db is not None:
        _decision_logger = DecisionLogger(db)

    return _decision_logger


async def init_decision_logger(db: "ResearchDatabase") -> DecisionLogger:
    """Initialize and start the global decision logger.

    Args:
        db: Database connection

    Returns:
        The initialized and started decision logger
    """
    global _decision_logger

    if _decision_logger is None:
        _decision_logger = DecisionLogger(db)

    await _decision_logger.start()
    return _decision_logger
