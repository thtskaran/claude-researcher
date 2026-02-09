#!/usr/bin/env python3
"""
Quick test to verify agents emit WebSocket events during research.

This runs a short 2-minute research and shows events in the terminal.
Open http://localhost:3000/test-websocket to see events in the UI!
"""
import asyncio
import sys

async def main():
    # Import after adding to path
    from src.agents.director import ResearchHarness

    print("=" * 60)
    print("Testing Live WebSocket Events")
    print("=" * 60)
    print("\n📡 IMPORTANT: Open http://localhost:3000/test-websocket")
    print("   to see events appear in real-time!\n")
    print("Starting 2-minute research session...")
    print("=" * 60)
    print()

    async with ResearchHarness() as harness:
        try:
            await harness.research(
                goal="What are the latest developments in quantum computing?",
                time_limit_minutes=2
            )
            print("\n✅ Research complete!")
            print("Check the WebSocket test page to see the events that were emitted.")
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
