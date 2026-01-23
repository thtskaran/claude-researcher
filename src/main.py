"""Main CLI entry point for Claude Researcher."""

import asyncio
import signal
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .agents.director import ResearchHarness
from .interaction import InteractionConfig, InputListener

console = Console()

# Create Typer app - no subcommands needed
app = typer.Typer(
    name="researcher",
    help="Deep hierarchical research agent powered by Claude.",
    add_completion=False,
)

# Global harness for signal handling
_harness: Optional[ResearchHarness] = None


def _handle_interrupt(signum, frame):
    """Handle interrupt signal gracefully."""
    console.print("\n[yellow]Interrupt received - stopping research...[/yellow]")
    if _harness:
        _harness.stop()


@app.command()
def main(
    goal: str = typer.Argument(..., help="The research goal or question to investigate"),
    time_limit: int = typer.Option(
        60,
        "--time", "-t",
        help="Time limit in minutes (default: 60)",
        min=1,
        max=480,
    ),
    db_path: str = typer.Option(
        "research.db",
        "--db", "-d",
        help="Path to SQLite database file",
    ),
    no_clarify: bool = typer.Option(
        False,
        "--no-clarify",
        help="Skip pre-research clarification questions",
    ),
    autonomous: bool = typer.Option(
        False,
        "--autonomous", "-a",
        help="Run fully autonomous (no user interaction)",
    ),
    timeout: int = typer.Option(
        60,
        "--timeout",
        help="Timeout in seconds for mid-research questions (default: 60)",
        min=10,
        max=300,
    ),
):
    """Run a deep research session on a topic.

    Output is automatically saved to: output/{topic-slug}_{session-id}/
      - report.md         Narrative research report
      - findings.json     Structured findings data
      - knowledge_graph.html  Interactive visualization

    Interactive Features:
        By default, the researcher asks 2-4 clarification questions before
        starting to refine the research scope. During research, you can type
        messages to inject guidance. Use --autonomous to disable all interaction.

    Examples:
        researcher "What are the latest advances in fusion energy?"
        researcher "History of the Internet" --time 120
        researcher "Climate change solutions" -t 30 --no-clarify
        researcher "AI developments" --autonomous
    """
    global _harness

    console.print()
    console.print(Panel(
        "[bold]Claude Deep Researcher[/bold]\n"
        "Hierarchical multi-agent research system",
        border_style="blue",
    ))

    # Set up signal handlers
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    # Create interaction config from CLI args
    interaction_config = InteractionConfig.from_cli_args(
        no_clarify=no_clarify,
        autonomous=autonomous,
        timeout=timeout,
    )

    # Show interaction mode
    if autonomous:
        console.print("[dim]Running in autonomous mode (no user interaction)[/dim]")
    elif no_clarify:
        console.print("[dim]Skipping clarification questions. Type + Enter during research to inject guidance.[/dim]")
    else:
        console.print("[dim]Interactive mode enabled. Type + Enter during research to inject guidance.[/dim]")

    async def run():
        global _harness

        async with ResearchHarness(db_path, interaction_config=interaction_config) as harness:
            _harness = harness

            # Create listener but don't start it yet - will be started after clarification
            listener: Optional[InputListener] = None
            if not interaction_config.autonomous_mode:
                listener = InputListener(
                    harness.director.interaction,
                    console=console,
                    on_interact_start=harness.director.pause_progress,
                    on_interact_end=harness.director.resume_progress,
                )
                # Pass listener to director so it can start it after clarification
                harness.director.set_input_listener(listener)

            try:
                report = await harness.research(goal, time_limit)
                return report

            except asyncio.CancelledError:
                console.print("[yellow]Research cancelled[/yellow]")
                return None

            finally:
                if listener:
                    listener.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)


def cli():
    """Entry point."""
    app()


if __name__ == "__main__":
    cli()
