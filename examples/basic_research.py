#!/usr/bin/env python3
"""Basic example of running a research session."""

import asyncio
from src.agents.director import ResearchHarness


async def main():
    """Run a basic research session."""
    async with ResearchHarness("research.db") as harness:
        # Run a 30-minute research session
        report = await harness.research(
            goal="What are the current leading approaches to achieving AGI?",
            time_limit_minutes=30,
        )

        # Print key findings
        print("\n=== Key Findings ===")
        for finding in report.key_findings[:10]:
            print(f"[{finding.finding_type.value}] {finding.content}")

        # Export to markdown
        filename = await harness.director.export_findings("markdown")
        print(f"\nExported to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
