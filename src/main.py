"""Main CLI entry point for Claude Researcher."""

import asyncio
import signal
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .agents.director import ResearchHarness

console = Console()

# Global harness for signal handling
_harness: Optional[ResearchHarness] = None


def _handle_interrupt(signum, frame):
    """Handle interrupt signal gracefully."""
    console.print("\n[yellow]Interrupt received - stopping research...[/yellow]")
    if _harness:
        _harness.stop()


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
):
    """Deep hierarchical research agent powered by Claude.

    Run autonomous research sessions that search the web, analyze findings,
    and dive deeper into promising threads until the time limit is reached.

    Output is automatically saved to: output/{topic-slug}_{session-id}/
      - report.md         Narrative research report
      - findings.json     Structured findings data
      - knowledge_graph.html  Interactive visualization

    Examples:
        researcher "What are the latest advances in fusion energy?"
        researcher "History of the Internet" --time 120
        researcher "Climate change solutions" -t 30
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

    async def run():
        global _harness
        async with ResearchHarness(db_path) as harness:
            _harness = harness
            try:
                report = await harness.research(goal, time_limit)
                return report

            except asyncio.CancelledError:
                console.print("[yellow]Research cancelled[/yellow]")
                return None

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)


def app():
    """Entry point."""
    typer.run(main)


if __name__ == "__main__":
    app()
