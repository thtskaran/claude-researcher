"""Parallel research execution with multiple interns."""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from ..events import emit_agent_event
from ..logging_config import get_logger
from ..models.findings import Finding, InternReport, ManagerDirective
from ..storage.database import ResearchDatabase
from .base import AgentConfig
from .intern import InternAgent

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..verification import VerificationPipeline


@dataclass
class ParallelResearchResult:
    """Result from parallel research execution."""
    reports: list[InternReport]
    total_findings: list[Finding]
    total_searches: int
    execution_time_seconds: float
    errors: list[str] = field(default_factory=list)


class ParallelInternPool:
    """Pool of intern agents for parallel research execution.

    Based on Anthropic's multi-agent pattern and GPT Researcher's
    parallel execution via asyncio.gather().

    Key principles:
    - Each intern gets a clear, focused objective
    - Interns operate independently on different aspects
    - Results are aggregated by the manager
    """

    def __init__(
        self,
        db: ResearchDatabase,
        pool_size: int = 3,
        config: AgentConfig | None = None,
        console: Console | None = None,
        verification_pipeline: Optional["VerificationPipeline"] = None,
    ):
        """Initialize the intern pool.

        Args:
            db: Database for persistence
            pool_size: Number of interns to run in parallel (default: 3)
            config: Agent configuration
            console: Rich console for output
            verification_pipeline: Optional verification pipeline for hallucination reduction
        """
        self.db = db
        self.pool_size = pool_size
        self.config = config or AgentConfig()
        self.console = console or Console()
        self.verification_pipeline = verification_pipeline
        self._current_session_id: str | None = None
        self._pause_requested = False
        self._active_interns: list[InternAgent] = []

        # Note: We no longer store intern instances - they are created fresh per directive
        # to prevent state corruption from concurrent access

    async def research_parallel(
        self,
        directives: list[ManagerDirective],
        session_id: str,
    ) -> ParallelResearchResult:
        """Execute multiple research directives in parallel.

        Args:
            directives: List of directives for each intern
            session_id: Current research session ID

        Returns:
            ParallelResearchResult with aggregated findings
        """
        start_time = datetime.now()
        self._current_session_id = session_id

        # Check if already paused before starting
        if self._pause_requested:
            return ParallelResearchResult(
                reports=[], total_findings=[], total_searches=0,
                execution_time_seconds=0, errors=["Paused before start"],
            )

        # Limit directives to pool size
        active_directives = directives[:self.pool_size]

        logger.info("Parallel research starting: %d directives", len(active_directives))
        self._log(f"Starting parallel research with {len(active_directives)} interns")

        # Create tasks for parallel execution
        # Each task gets a FRESH intern instance to prevent state corruption
        tasks = []
        self._active_interns = []
        for i, directive in enumerate(active_directives):
            # Create a fresh intern for each directive with unique ID
            intern = InternAgent(
                self.db,
                self.config,
                self.console,
                self.verification_pipeline,
                agent_id=f"intern_{i}"  # Unique ID for each intern
            )
            # Propagate pause flag to new interns
            if self._pause_requested:
                intern._pause_requested = True
            self._active_interns.append(intern)
            task = self._execute_with_error_handling(intern, directive, session_id, i)
            tasks.append(task)

        # Execute all tasks in parallel with 15-minute timeout per batch.
        # Use asyncio.wait instead of wait_for(gather(...)) so we preserve
        # results from interns that completed before the timeout.
        task_objects = [asyncio.ensure_future(t) for t in tasks]
        try:
            done, pending = await asyncio.wait(task_objects, timeout=900)
        except asyncio.CancelledError:
            # Parent was cancelled — cancel children and re-raise
            for t in task_objects:
                t.cancel()
            await asyncio.gather(*task_objects, return_exceptions=True)
            self._active_interns = []
            raise

        if pending:
            logger.warning(
                "Parallel research timed out: %d/%d tasks still pending",
                len(pending), len(task_objects),
            )
            self._log(
                f"Parallel research timed out: {len(done)}/{len(task_objects)} interns completed",
                style="yellow",
            )
            for t in pending:
                t.cancel()
            # Wait for cancellation to finish cleanly
            await asyncio.gather(*pending, return_exceptions=True)
            for intern in self._active_interns:
                intern.stop()

        self._active_interns = []

        # Collect results — completed tasks keep their results,
        # timed-out/cancelled tasks are recorded as errors
        results = []
        for task_obj in task_objects:
            if task_obj.done() and not task_obj.cancelled():
                exc = task_obj.exception()
                results.append(exc if exc else task_obj.result())
            else:
                results.append(TimeoutError("Intern timed out"))

        # Aggregate results
        reports = []
        all_findings = []
        total_searches = 0
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Intern {i}: {str(result)}")
                logger.error("Intern %d failed: %s", i, result)
                self._log(f"Intern {i} failed: {result}", style="red")
            elif isinstance(result, InternReport):
                reports.append(result)
                all_findings.extend(result.findings)
                total_searches += result.searches_performed
                self._log(
                    f"Intern {i} completed: {len(result.findings)} findings from {result.searches_performed} searches",
                    style="green"
                )

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info("Parallel research complete: %d findings in %.1fs", len(all_findings), execution_time)
        self._log(
            f"Parallel research complete: {len(all_findings)} total findings in {execution_time:.1f}s"
        )

        result = ParallelResearchResult(
            reports=reports,
            total_findings=all_findings,
            total_searches=total_searches,
            execution_time_seconds=execution_time,
            errors=errors,
        )
        self._current_session_id = None
        return result

    async def _execute_with_error_handling(
        self,
        intern: InternAgent,
        directive: ManagerDirective,
        session_id: str,
        intern_id: int,
    ) -> InternReport:
        """Execute a directive with error handling.

        Args:
            intern: The intern agent to use
            directive: The directive to execute
            session_id: Current session ID
            intern_id: ID of this intern for logging

        Returns:
            InternReport with findings
        """
        try:
            logger.info("Intern %d starting: %s", intern_id, directive.topic)
            self._log(f"Intern {intern_id} starting: {directive.topic}", style="cyan")
            report = await intern.execute_directive(directive, session_id)
            return report
        except asyncio.CancelledError:
            # Don't log as error — this is expected during timeout cancellation
            logger.info("Intern %d cancelled: %s", intern_id, directive.topic)
            raise
        except Exception as e:
            logger.error("Intern %d error on topic '%s': %s", intern_id, directive.topic, e, exc_info=True)
            self._log(f"Intern {intern_id} error: {e}", style="red")
            raise

    async def decompose_and_research(
        self,
        goal: str,
        session_id: str,
        llm_callback,
        max_aspects: int = 3,
    ) -> ParallelResearchResult:
        """Decompose a goal into aspects and research in parallel.

        This implements the "perspective-guided question asking" pattern
        from Stanford STORM.

        Args:
            goal: The main research goal
            session_id: Current session ID
            llm_callback: Async function to call LLM for decomposition
            max_aspects: Maximum number of aspects to research in parallel

        Returns:
            ParallelResearchResult with aggregated findings
        """
        logger.info("Decomposing goal into aspects: %s", goal[:200])
        # Decompose the goal into research aspects
        aspects = await self._decompose_goal(goal, llm_callback, max_aspects)

        self._log(f"Decomposed goal into {len(aspects)} aspects:")
        for i, aspect in enumerate(aspects):
            self._log(f"  {i+1}. {aspect}", style="cyan")

        # Create directives for each aspect
        directives = [
            ManagerDirective(
                action="search",
                topic=aspect,
                instructions=f"Research this specific aspect thoroughly: {aspect}",
                priority=8,
                max_searches=5,
            )
            for aspect in aspects
        ]

        # Execute in parallel
        return await self.research_parallel(directives, session_id)

    async def _decompose_goal(
        self,
        goal: str,
        llm_callback,
        max_aspects: int,
    ) -> list[str]:
        """Decompose a research goal into distinct aspects.

        Args:
            goal: The main research goal
            llm_callback: Async function to call LLM
            max_aspects: Maximum number of aspects

        Returns:
            List of research aspects
        """
        import json
        import re

        prompt = (
            f"Decompose this research goal into {max_aspects} "
            "distinct aspects that can be researched independently "
            "and in parallel.\n\n"
            f"Research Goal: {goal}\n\n"
            "Each aspect should be specific, focused, cover a "
            "different angle, and be searchable standalone."
        )

        decompose_schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["aspects"],
            },
        }

        try:
            response = await llm_callback(
                prompt, output_format=decompose_schema,
            )
        except TypeError:
            response = await llm_callback(prompt)

        if isinstance(response, dict):
            return response.get("aspects", [goal])[:max_aspects]

        # Text fallback
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                aspects = json.loads(match.group())
                return aspects[:max_aspects]
            except json.JSONDecodeError:
                pass

        return [goal]

    def _log(self, message: str, style: str | None = None) -> None:
        """Log a message to the console."""
        prefix = "[PARALLEL]"
        if style:
            self.console.print(f"{prefix} {message}", style=style)
        else:
            self.console.print(f"{prefix} {message}")

        if (
            self._current_session_id
            and os.environ.get("CLAUDE_RESEARCHER_DISABLE_LOG_EVENTS") != "1"
        ):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(
                emit_agent_event(
                    session_id=self._current_session_id,
                    event_type="system",
                    agent="parallel",
                    data={"message": f"{prefix} {message}"},
                )
            )

    def pause(self) -> None:
        """Request all active interns to pause."""
        self._pause_requested = True
        for intern in self._active_interns:
            intern.pause()
        self._log("Pause requested for all active interns")

    def reset_all(self) -> None:
        """Reset method - clears pause flag since interns are created fresh."""
        self._pause_requested = False
        self._active_interns = []

    def set_verification_pipeline(self, pipeline: "VerificationPipeline") -> None:
        """Set the verification pipeline for future intern instances.

        Args:
            pipeline: The verification pipeline to use
        """
        self.verification_pipeline = pipeline
        # Note: New interns created in research_parallel() will use this pipeline
