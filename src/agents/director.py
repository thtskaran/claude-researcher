"""Director agent - top-level interface for user interaction."""

import asyncio
from typing import Any, Optional
from datetime import datetime
from pathlib import Path

from .base import BaseAgent, AgentConfig
from .manager import ManagerAgent
from .intern import InternAgent
from ..models.findings import AgentRole, ManagerReport, ResearchSession
from ..storage.database import ResearchDatabase
from ..reports.writer import DeepReportWriter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


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
        config: Optional[AgentConfig] = None,
        console: Optional[Console] = None,
    ):
        super().__init__(AgentRole.DIRECTOR, db, config, console)
        self.intern = InternAgent(db, config, console)
        self.manager = ManagerAgent(db, self.intern, config, console)
        self.current_session: Optional[ResearchSession] = None
        self._progress_task = None

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

    async def start_research(
        self,
        goal: str,
        time_limit_minutes: int = 60,
    ) -> ManagerReport:
        """Start a new research session.

        Args:
            goal: The research goal/question to investigate
            time_limit_minutes: Maximum time for the research session

        Returns:
            ManagerReport with findings and recommendations
        """
        # Create session
        self.current_session = await self.db.create_session(
            goal=goal,
            time_limit_minutes=time_limit_minutes,
        )

        self._log_header(goal, time_limit_minutes)

        # Run research with progress display
        try:
            report = await self._run_with_progress(goal, time_limit_minutes)

            # Update session
            self.current_session.status = "completed"
            self.current_session.ended_at = datetime.now()
            self.current_session.total_findings = len(report.key_findings)
            await self.db.update_session(self.current_session)

            # Display results
            await self._display_report(report)

            # Auto-export to markdown
            md_file = await self.export_findings("markdown")
            self.console.print(f"\n[bold green]Report saved to: {md_file}[/bold green]")

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

    async def _run_with_progress(
        self, goal: str, time_limit_minutes: int
    ) -> ManagerReport:
        """Run research with progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Researching: {goal[:50]}...",
                total=None,
            )

            # Add callback to update progress
            def update_progress(agent, ctx):
                iteration = ctx.get("iteration", 0)
                findings = len(self.manager.all_findings)
                progress.update(
                    task,
                    description=f"[cyan]Iteration {iteration} | Findings: {findings}",
                )

            self.manager.add_callback(update_progress)

            # Run research
            report = await self.manager.run_research(
                goal=goal,
                session_id=self.current_session.id,
                time_limit_minutes=time_limit_minutes,
            )

            progress.update(task, description="[green]Research complete!")

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

    async def get_session_findings(self) -> list:
        """Get all findings from the current session."""
        if not self.current_session:
            return []
        return await self.db.get_session_findings(self.current_session.id)

    async def export_findings(self, format: str = "json") -> str:
        """Export findings to a file."""
        import json

        if not self.current_session:
            raise ValueError("No active session")

        findings = await self.get_session_findings()

        if format == "json":
            output = {
                "session": {
                    "id": self.current_session.id,
                    "goal": self.current_session.goal,
                    "started_at": self.current_session.started_at.isoformat(),
                    "ended_at": self.current_session.ended_at.isoformat() if self.current_session.ended_at else None,
                },
                "findings": [
                    {
                        "content": f.content,
                        "type": f.finding_type.value,
                        "source_url": f.source_url,
                        "confidence": f.confidence,
                    }
                    for f in findings
                ],
            }
            # Save to output folder
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            filename = output_dir / f"research_{self.current_session.id}.json"
            filename.write_text(json.dumps(output, indent=2))
            return str(filename)

        elif format == "markdown":
            # Use the deep report writer for narrative synthesis
            self.console.print("\n[bold cyan]Generating deep research report...[/bold cyan]")
            self.console.print("[dim]This uses extended thinking to synthesize findings into a narrative report.[/dim]\n")

            writer = DeepReportWriter(model="opus")

            # Get topics explored and remaining from manager
            topics_explored = [t.topic for t in self.manager.completed_topics] if self.manager.completed_topics else []
            topics_remaining = [t.topic for t in self.manager.topics_queue] if self.manager.topics_queue else []

            # Get knowledge graph exports
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            try:
                kg_exports = self.manager.get_knowledge_graph_exports(str(output_dir))
                self.console.print(f"[dim]Knowledge graph: {kg_exports.get('stats', {}).get('num_entities', 0)} entities, {kg_exports.get('stats', {}).get('num_relations', 0)} relations[/dim]")
            except Exception as e:
                self.console.print(f"[dim]Knowledge graph export skipped: {e}[/dim]")
                kg_exports = None

            report = await writer.generate_report(
                session=self.current_session,
                findings=findings,
                topics_explored=topics_explored,
                topics_remaining=topics_remaining,
                kg_exports=kg_exports,
            )

            # Save to output folder
            filename = output_dir / f"research_{self.current_session.id}.md"
            filename.write_text(report)
            return str(filename)

        raise ValueError(f"Unknown format: {format}")

    def stop_research(self) -> None:
        """Request the research to stop gracefully."""
        self.manager.stop()
        self.intern.stop()
        self._log("Stop requested - finishing current operation")


class ResearchHarness:
    """Main harness for running deep research sessions.

    This is the primary entry point for running the hierarchical research system.
    """

    def __init__(self, db_path: str = "research.db"):
        self.db_path = db_path
        self.db: Optional[ResearchDatabase] = None
        self.director: Optional[DirectorAgent] = None
        self.console = Console()

    async def __aenter__(self):
        """Async context manager entry."""
        self.db = ResearchDatabase(self.db_path)
        await self.db.connect()
        self.director = DirectorAgent(self.db, console=self.console)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.db:
            await self.db.close()

    async def research(
        self,
        goal: str,
        time_limit_minutes: int = 60,
    ) -> ManagerReport:
        """Run a research session.

        Args:
            goal: The research question or topic to investigate
            time_limit_minutes: Maximum time for the session (default: 60)

        Returns:
            ManagerReport with findings and analysis
        """
        if not self.director:
            raise RuntimeError("Harness not initialized - use async with")

        return await self.director.start_research(goal, time_limit_minutes)

    def stop(self) -> None:
        """Stop the current research session."""
        if self.director:
            self.director.stop_research()
