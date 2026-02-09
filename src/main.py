"""Main CLI entry point for Claude Researcher."""

import asyncio
import signal
import subprocess
import sys
import webbrowser
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


@app.command()
def ui(
    session_id: Optional[str] = typer.Argument(
        None,
        help="Optional session ID to open directly in UI"
    ),
    port: int = typer.Option(
        8080,
        "--port", "-p",
        help="Port for API server (default: 8080)",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't auto-open browser",
    ),
):
    """
    Launch the web UI for claude-researcher.

    This starts the FastAPI backend and opens the UI in your browser.
    The UI provides a visual interface for managing research sessions,
    viewing agent thinking in real-time, and exploring knowledge graphs.

    Examples:
        researcher ui                  # Launch UI server
        researcher ui abc123ef         # Open UI with specific session
        researcher ui --port 9000      # Use custom port
        researcher ui --no-browser     # Start server without opening browser
    """
    console.print()
    console.print(Panel(
        "[bold]Claude Researcher UI[/bold]\n"
        "Launching web interface...",
        border_style="blue",
    ))

    # Check if servers are already running
    import socket
    import time
    import os
    from pathlib import Path

    def check_port(port_num):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port_num)) == 0
        sock.close()
        return result

    api_running = check_port(port)
    ui_running = check_port(3000)

    # Start API server if not running
    if api_running:
        console.print(f"[yellow]API server already running on port {port}[/yellow]")
    else:
        console.print(f"[cyan]Starting API server on port {port}...[/cyan]")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "api.server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for API server to start
            max_wait = 10
            for i in range(max_wait):
                if check_port(port):
                    console.print("[green]✓ API server started[/green]")
                    break
                time.sleep(0.5)
            else:
                console.print("[red]✗ Failed to start API server[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error starting API server: {e}[/red]")
            sys.exit(1)

    # Start Next.js UI server if not running
    if ui_running:
        console.print("[yellow]UI server already running on port 3000[/yellow]")
    else:
        console.print("[cyan]Starting Next.js UI server on port 3000...[/cyan]")
        try:
            ui_path = Path(__file__).parent.parent / "ui"
            if not ui_path.exists():
                console.print("[red]✗ UI directory not found[/red]")
                sys.exit(1)

            # Start Next.js dev server
            subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for UI server to start
            max_wait = 15
            for i in range(max_wait):
                if check_port(3000):
                    console.print("[green]✓ UI server started[/green]")
                    break
                time.sleep(1)
            else:
                console.print("[red]✗ Failed to start UI server[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error starting UI server: {e}[/red]")
            sys.exit(1)

    # Construct UI URL (port 3000, not API port)
    url = f"http://localhost:3000"
    if session_id:
        url += f"/session/{session_id}"

    console.print(f"\n[bold green]✓ All servers ready[/bold green]")
    console.print(f"[dim]Frontend: http://localhost:3000[/dim]")
    console.print(f"[dim]API: http://localhost:{port}[/dim]")
    console.print(f"[dim]API Docs: http://localhost:{port}/docs[/dim]")

    # Open browser
    if not no_browser:
        console.print(f"\n[cyan]Opening {url} in browser...[/cyan]")
        webbrowser.open(url)
    else:
        console.print(f"\n[cyan]Server URL: {url}[/cyan]")

    console.print("\n[dim]Press Ctrl+C to stop the server[/dim]\n")

    # Keep the process running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping server...[/yellow]")
        sys.exit(0)


def cli():
    """Entry point."""
    app()


if __name__ == "__main__":
    cli()
