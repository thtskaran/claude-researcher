#!/usr/bin/env python3
"""Example with custom agent configuration."""

import asyncio
from src.agents.base import AgentConfig
from src.agents.director import DirectorAgent
from src.storage.database import ResearchDatabase
from rich.console import Console


async def main():
    """Run research with custom configuration."""
    # Custom configuration for deeper research
    config = AgentConfig(
        model="sonnet",  # or "opus" for deeper reasoning, "haiku" for faster
        max_tokens=16000,
        max_iterations=200,  # Allow more iterations for longer sessions
        thinking_enabled=True,  # Enable extended thinking prompts
    )

    console = Console()
    db = ResearchDatabase("custom_research.db")
    await db.connect()

    try:
        director = DirectorAgent(db, config=config, console=console)

        # Run a 2-hour deep research session
        report = await director.start_research(
            goal="Comprehensive analysis of quantum computing's impact on cryptography",
            time_limit_minutes=120,
        )

        print(f"\nCompleted with {len(report.key_findings)} findings")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
