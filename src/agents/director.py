"""Director agent - top-level interface for user interaction."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..costs.tracker import CostSummary, get_cost_tracker, reset_cost_tracker
from ..events import emit_synthesis
from ..interaction import InteractionConfig, UserInteraction
from ..models.findings import AgentRole, ManagerReport, ResearchSession
from ..reports.writer import DeepReportWriter
from ..storage.database import ResearchDatabase
from .base import AgentConfig, BaseAgent
from .intern import InternAgent
from .manager import ManagerAgent


class DirectorAgent(BaseAgent):
    """The Director agent is the top-level interface that the user interacts with.

    Responsibilities:
    - Receive and interpret user research requests
    - Translate user goals into clear research objectives
    - Manage research sessions (start, pause, resume, stop)
    - Report progress and findings to the user
    - Handle time limits and session management
    """

    def __init__(
        self,
        db: ResearchDatabase,
        config: AgentConfig | None = None,
        console: Console | None = None,
        interaction_config: InteractionConfig | None = None,
        owns_db: bool = False,
    ):
        super().__init__(AgentRole.DIRECTOR, db, config, console)
        self._owns_db = owns_db  # Only close db if we own it (not injected by caller)
        self.intern = InternAgent(db, config, console)

        # Set up interaction handler
        self.interaction_config = interaction_config or InteractionConfig()
        self.interaction = UserInteraction(
            config=self.interaction_config,
            console=self.console,
            llm_callback=self._interaction_llm_callback,
        )

        # Pass interaction to manager
        self.manager = ManagerAgent(
            db, self.intern, config, console,
            interaction=self.interaction,
        )
        self.current_session: ResearchSession | None = None
        self._progress_task = None
        self._progress: Progress | None = None  # For pause/resume support

        # Wire up progress callbacks to interaction handler
        self.interaction.set_progress_callbacks(
            on_pause=self.pause_progress,
            on_resume=self.resume_progress,
        )

        # Input listener (set later, started after clarification)
        self._input_listener = None

    def set_input_listener(self, listener) -> None:
        """Set the input listener (will be started after clarification)."""
        self._input_listener = listener

    async def _start_input_listener(self) -> None:
        """Start the input listener (called after clarification is done)."""
        if self._input_listener:
            await self._input_listener.start()

    def pause_progress(self) -> None:
        """Pause the progress spinner (for interact mode)."""
        if self._progress:
            self._progress.stop()

    def resume_progress(self) -> None:
        """Resume the progress spinner (after interact mode)."""
        if self._progress:
            self._progress.start()

    async def _interaction_llm_callback(self, prompt: str) -> str:
        """LLM callback for interaction module (uses fast model)."""
        original_model = self.config.model
        self.config.model = "haiku"  # Fast model for clarification questions
        try:
            return await self.call_claude(prompt)
        finally:
            self.config.model = original_model

    @property
    def system_prompt(self) -> str:
        return """You are the Research Director. You interface with the user and oversee the entire research operation.

RESPONSIBILITIES:
1. Understand and clarify user research requests
2. Set clear objectives and success criteria
3. Monitor research progress and quality
4. Provide meaningful updates to the user
5. Synthesize final results into actionable insights

COMMUNICATION STYLE:
- Be professional but approachable
- Provide context for your decisions
- Be transparent about limitations
- Offer actionable recommendations

When presenting results:
- Lead with key insights
- Support claims with evidence
- Note confidence levels
- Suggest next steps"""

    async def think(self, context: dict[str, Any]) -> str:
        """Not used for Director - it's event-driven from user input."""
        return ""

    async def act(self, thought: str, context: dict[str, Any]) -> dict[str, Any]:
        """Not used for Director - it's event-driven from user input."""
        return {}

    async def observe(self, action_result: dict[str, Any]) -> str:
        """Not used for Director - it's event-driven from user input."""
        return ""

    def is_done(self, context: dict[str, Any]) -> bool:
        """Director is done when the session ends."""
        return context.get("session_ended", False)

    async def clarify_research_goal(self, goal: str) -> str:
        """Ask clarification questions before starting research.

        Args:
            goal: The original research goal

        Returns:
            The enriched goal after clarification, or original if skipped
        """
        clarified = await self.interaction.clarify_research_goal(goal)

        if not clarified.skipped and clarified.clarifications:
            self.console.print()
            self.console.print(Panel(
                f"[bold]Original:[/bold] {clarified.original}\n\n"
                f"[bold]Refined:[/bold] {clarified.enriched_context}",
                title="[green]Research Goal Refined[/green]",
                border_style="green",
            ))
            self.console.print()

        return clarified.enriched_context

    async def start_research(
        self,
        goal: str,
        time_limit_minutes: int = 60,
        skip_clarification: bool = False,
        existing_session_id: str | None = None,
        resume: bool = False,
    ) -> ManagerReport:
        """Start a new research session or resume a paused/crashed one.

        Args:
            goal: The research goal/question to investigate
            time_limit_minutes: Maximum time for the research session
            skip_clarification: Skip pre-research clarification questions
            existing_session_id: Optional existing session ID to use (for UI/API)
            resume: If True, resume a paused or crashed session

        Returns:
            ManagerReport with findings and recommendations
        """
        # Reset interaction state and cost tracker
        self.interaction.reset()
        reset_cost_tracker()

        if resume and existing_session_id:
            # Resume flow: load session, skip clarification
            self.current_session = await self.db.get_session(existing_session_id)
            if not self.current_session:
                raise ValueError(f"Session {existing_session_id} not found")
            if self.current_session.status not in ("paused", "crashed"):
                raise ValueError(
                    f"Session {existing_session_id} is "
                    f"{self.current_session.status}, not paused/crashed"
                )
            effective_goal = self.current_session.goal
            time_limit_minutes = self.current_session.time_limit_minutes
        else:
            # Clarify goal if enabled (skip when using existing session from UI)
            if not skip_clarification and self.interaction_config.enable_clarification and not existing_session_id:
                effective_goal = await self.clarify_research_goal(goal)
            else:
                effective_goal = goal

        # Start input listener AFTER clarification is done
        await self._start_input_listener()

        if not resume:
            # Use existing session or create new one
            if existing_session_id:
                # Load existing session from database (created by API)
                self.current_session = await self.db.get_session(existing_session_id)
                if not self.current_session:
                    raise ValueError(f"Session {existing_session_id} not found")
            else:
                # Create new session (CLI flow)
                self.current_session = await self.db.create_session(
                    goal=effective_goal,
                    time_limit_minutes=time_limit_minutes,
                )

        # Set session ID on all agents for WebSocket events
        self.session_id = self.current_session.id
        self.manager.session_id = self.current_session.id
        self.intern.session_id = self.current_session.id

        self._log_header(effective_goal, time_limit_minutes)

        # Run research with progress display
        try:
            report = await self._run_with_progress(
                effective_goal, time_limit_minutes, resume=resume
            )

            # Check if we paused (don't mark as completed)
            if self.current_session.status == "paused":
                self.console.print("\n[yellow]Research paused. Progress saved.[/yellow]")
                return report

            # Update session
            self.current_session.status = "completed"
            self.current_session.ended_at = datetime.now()
            self.current_session.total_findings = len(report.key_findings)
            self.current_session.phase = "done"
            await self.db.update_session(self.current_session)

            # Display results
            await self._display_report(report)

            # Auto-export all outputs
            output_path = await self.export_findings()
            self.console.print(f"\n[bold green]Research saved to: {output_path}/[/bold green]")

            return report

        except asyncio.CancelledError:
            self.console.print("\n[yellow]Research interrupted by user[/yellow]")
            self.current_session.status = "interrupted"
            self.current_session.ended_at = datetime.now()
            await self.db.update_session(self.current_session)
            raise

        except Exception as e:
            self.console.print(f"\n[red]Error during research: {e}[/red]")
            self.current_session.status = "error"
            self.current_session.ended_at = datetime.now()
            await self.db.update_session(self.current_session)
            raise

        finally:
            # Only close database if we own it (not injected by caller like ResearchHarness)
            if self._owns_db and self.db:
                await self.db.close()

    def pause_research(self) -> None:
        """Request the research to pause gracefully (saves state for resume)."""
        self.manager.pause()
        self.intern.pause()
        if self.manager.intern_pool:
            self.manager.intern_pool.pause()
        self._log("Pause requested - finishing current operation")

    async def _run_with_progress(
        self, goal: str, time_limit_minutes: int, resume: bool = False
    ) -> ManagerReport:
        """Run research with progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

        with self._progress as progress:
            label = "Resuming" if resume else "Researching"
            task = progress.add_task(
                f"[cyan]{label}: {goal[:50]}...",
                total=None,
            )

            # Add callback to update progress
            def update_progress(agent, ctx):
                iteration = ctx.get("iteration", 0)
                findings = len(self.manager.all_findings)
                progress.update(
                    task,
                    description=f"[cyan]Iteration {iteration} | Findings: {findings}[/cyan]",
                )

            self.manager.add_callback(update_progress)

            # Run research
            report = await self.manager.run_research(
                goal=goal,
                session_id=self.current_session.id,
                time_limit_minutes=time_limit_minutes,
                resume=resume,
                session=self.current_session if resume else None,
            )

            # Check if paused
            if self.manager._pause_requested:
                progress.update(task, description="[yellow]Research paused")
                # Refresh session from DB (checkpoint_state updated it)
                self.current_session = await self.db.get_session(self.current_session.id)
            else:
                progress.update(task, description="[green]Research complete!")
            self._progress = None

            return report

    def _log_header(self, goal: str, time_limit_minutes: int) -> None:
        """Log research session header."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Research Goal:[/bold] {goal}\n"
            f"[bold]Time Limit:[/bold] {time_limit_minutes} minutes\n"
            f"[bold]Session ID:[/bold] {self.current_session.id}",
            title="[bold blue]Deep Research Session[/bold blue]",
            border_style="blue",
        ))
        self.console.print()

    async def _display_report(self, report: ManagerReport) -> None:
        """Display the final research report."""
        self.console.print()
        self.console.print(Panel(
            report.summary,
            title="[bold green]Research Summary[/bold green]",
            border_style="green",
        ))

        # Findings table
        if report.key_findings:
            table = Table(title="Key Findings", show_header=True, header_style="bold")
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Finding", style="white")
            table.add_column("Confidence", style="yellow", width=10)

            for finding in report.key_findings[:15]:
                table.add_row(
                    finding.finding_type.value.upper(),
                    finding.content[:100] + "..." if len(finding.content) > 100 else finding.content,
                    f"{finding.confidence:.0%}",
                )

            self.console.print(table)

        # Stats
        stats = await self.db.get_session_stats(self.current_session.id)
        self.console.print()
        self.console.print(Panel(
            f"[bold]Topics Explored:[/bold] {len(report.topics_explored)}\n"
            f"[bold]Total Findings:[/bold] {stats['total_findings']}\n"
            f"[bold]Unique Searches:[/bold] {stats['unique_searches']}\n"
            f"[bold]Max Depth:[/bold] {stats['max_depth']}\n"
            f"[bold]Time Used:[/bold] {report.time_elapsed_minutes:.1f} minutes",
            title="[bold]Session Statistics[/bold]",
            border_style="dim",
        ))

        # Remaining topics
        if report.topics_remaining:
            self.console.print()
            self.console.print("[bold]Topics for further investigation:[/bold]")
            for topic in report.topics_remaining[:5]:
                self.console.print(f"  - {topic}")

        # Display cost summary
        self._display_costs()

    def _display_costs(self) -> None:
        """Display API cost summary."""
        cost_summary = get_cost_tracker().get_summary()

        # Build cost table
        table = Table(title="API Cost Estimate", show_header=True, header_style="bold")
        table.add_column("Model", style="cyan", width=10)
        table.add_column("Calls", justify="right", width=6)
        table.add_column("Input", justify="right", width=10)
        table.add_column("Output", justify="right", width=10)
        table.add_column("Thinking", justify="right", width=10)
        table.add_column("Cost", justify="right", style="green", width=10)

        # Add rows for each model (only if used)
        if cost_summary.sonnet_usage.calls > 0:
            table.add_row(
                "Sonnet 4.5",
                str(cost_summary.sonnet_usage.calls),
                f"{cost_summary.sonnet_usage.input_tokens:,}",
                f"{cost_summary.sonnet_usage.output_tokens:,}",
                f"{cost_summary.sonnet_usage.thinking_tokens:,}",
                f"${cost_summary.sonnet_cost:.4f}",
            )

        if cost_summary.opus_usage.calls > 0:
            table.add_row(
                "Opus 4.5",
                str(cost_summary.opus_usage.calls),
                f"{cost_summary.opus_usage.input_tokens:,}",
                f"{cost_summary.opus_usage.output_tokens:,}",
                f"{cost_summary.opus_usage.thinking_tokens:,}",
                f"${cost_summary.opus_cost:.4f}",
            )

        if cost_summary.haiku_usage.calls > 0:
            table.add_row(
                "Haiku 4.5",
                str(cost_summary.haiku_usage.calls),
                f"{cost_summary.haiku_usage.input_tokens:,}",
                f"{cost_summary.haiku_usage.output_tokens:,}",
                f"{cost_summary.haiku_usage.thinking_tokens:,}",
                f"${cost_summary.haiku_cost:.4f}",
            )

        # Add web searches row if any
        if cost_summary.web_searches > 0 or cost_summary.web_fetches > 0:
            table.add_row(
                "Web",
                f"{cost_summary.web_searches + cost_summary.web_fetches}",
                f"{cost_summary.web_searches} srch",
                f"{cost_summary.web_fetches} fetch",
                "-",
                f"${cost_summary.search_cost:.4f}",
            )

        # Add total row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{cost_summary.total_calls}[/bold]",
            f"[bold]{cost_summary.total_input_tokens:,}[/bold]",
            f"[bold]{cost_summary.total_output_tokens - cost_summary.total_thinking_tokens:,}[/bold]",
            f"[bold]{cost_summary.total_thinking_tokens:,}[/bold]",
            f"[bold]${cost_summary.total_cost:.4f}[/bold]",
            style="bold",
        )

        self.console.print()
        self.console.print(table)
        self.console.print("[dim]Pricing: Opus $5/$25, Sonnet $3/$15, Haiku $1/$5 per MTok (input/output). Search: $0.01/search[/dim]")
        self.console.print("[dim]Note: Token counts are estimates (~4 chars/token). Thinking tokens billed as output.[/dim]")

    async def get_session_findings(self) -> list:
        """Get all findings from the current session."""
        if not self.current_session:
            return []
        return await self.db.get_session_findings(self.current_session.id)

    async def export_findings(self) -> str:
        """Export all research outputs to a dedicated folder.

        Creates: output/{slug}_{session_id}/
            - report.md       - Narrative report
            - findings.json   - Structured data

        Returns:
            Path to the output directory
        """
        import json

        if not self.current_session:
            raise ValueError("No active session")

        findings = await self.get_session_findings()

        # Create output directory: output/{slug}_{session_id}/
        slug = self.current_session.slug or "research"
        session_id = self.current_session.id
        output_dir = Path("output") / f"{slug}_{session_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold cyan]Exporting research to: {output_dir}/[/bold cyan]")

        # 1. Export JSON data
        json_output = {
            "session": {
                "id": self.current_session.id,
                "goal": self.current_session.goal,
                "slug": self.current_session.slug,
                "started_at": self.current_session.started_at.isoformat(),
                "ended_at": self.current_session.ended_at.isoformat() if self.current_session.ended_at else None,
                "status": self.current_session.status,
            },
            "findings": [
                {
                    "content": f.content,
                    "type": f.finding_type.value,
                    "source_url": f.source_url,
                    "confidence": f.confidence,
                    "search_query": f.search_query,
                }
                for f in findings
            ],
            "topics_explored": [t.topic for t in self.manager.completed_topics] if self.manager.completed_topics else [],
            "topics_remaining": [t.topic for t in self.manager.topics_queue] if self.manager.topics_queue else [],
            "costs": get_cost_tracker().get_summary().to_dict(),
        }
        json_file = output_dir / "findings.json"
        json_file.write_text(json.dumps(json_output, indent=2))
        self.console.print("  [dim]Saved findings.json[/dim]")

        # 2. Export knowledge graph
        try:
            kg_exports = self.manager.get_knowledge_graph_exports(str(output_dir))
            self.console.print(f"  [dim]Knowledge graph: {kg_exports.get('stats', {}).get('num_entities', 0)} entities, {kg_exports.get('stats', {}).get('num_relations', 0)} relations[/dim]")
        except Exception as e:
            self.console.print(f"  [dim]Knowledge graph export skipped: {e}[/dim]")
            kg_exports = None

        # 3. Generate markdown report
        self.console.print("[bold cyan]Generating deep research report...[/bold cyan]")
        self.console.print("[dim]This uses extended thinking to synthesize findings into a narrative report.[/dim]\n")

        # Emit synthesis start event
        if self.session_id:
            await emit_synthesis(
                session_id=self.session_id,
                agent="director",
                message=f"Synthesizing {len(findings)} findings into report...",
                progress=0
            )

        writer = DeepReportWriter(model="opus")

        async def report_progress(message: str, progress: int) -> None:
            if self.session_id:
                await emit_synthesis(
                    session_id=self.session_id,
                    agent="director",
                    message=message,
                    progress=progress
                )

        topics_explored = [t.topic for t in self.manager.completed_topics] if self.manager.completed_topics else []
        topics_remaining = [t.topic for t in self.manager.topics_queue] if self.manager.topics_queue else []

        report = await writer.generate_report(
            session=self.current_session,
            findings=findings,
            topics_explored=topics_explored,
            topics_remaining=topics_remaining,
            kg_exports=kg_exports,
            progress_callback=report_progress,
        )

        # Emit synthesis complete event
        if self.session_id:
            await emit_synthesis(
                session_id=self.session_id,
                agent="director",
                message="Report synthesis complete",
                progress=100
            )

        md_file = output_dir / "report.md"
        md_file.write_text(report)
        self.console.print("  [dim]Saved report.md[/dim]")

        return str(output_dir)

    def stop_research(self) -> None:
        """Request the research to stop gracefully."""
        self.manager.stop()
        self.intern.stop()
        self._log("Stop requested - finishing current operation")


class ResearchHarness:
    """Main harness for running deep research sessions.

    This is the primary entry point for running the hierarchical research system.
    """

    def __init__(
        self,
        db_path: str = "research.db",
        interaction_config: InteractionConfig | None = None,
    ):
        self.db_path = db_path
        self.db: ResearchDatabase | None = None
        self.director: DirectorAgent | None = None
        self.console = Console()
        self.interaction_config = interaction_config

    async def __aenter__(self):
        """Async context manager entry."""
        self.db = ResearchDatabase(self.db_path)
        await self.db.connect()
        self.director = DirectorAgent(
            self.db,
            console=self.console,
            interaction_config=self.interaction_config,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.db:
            await self.db.close()

    async def research(
        self,
        goal: str,
        time_limit_minutes: int = 60,
        existing_session_id: str | None = None,
        resume: bool = False,
    ) -> ManagerReport:
        """Run a research session.

        Args:
            goal: The research question or topic to investigate
            time_limit_minutes: Maximum time for the session (default: 60)
            existing_session_id: Optional existing session ID (for UI/API)
            resume: If True, resume a paused/crashed session

        Returns:
            ManagerReport with findings and analysis
        """
        if not self.director:
            raise RuntimeError("Harness not initialized - use async with")

        return await self.director.start_research(
            goal,
            time_limit_minutes,
            existing_session_id=existing_session_id,
            resume=resume,
        )

    def stop(self) -> None:
        """Stop the current research session."""
        if self.director:
            self.director.stop_research()
