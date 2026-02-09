"""
Research execution routes.

Handles starting and managing actual research runs from the UI.
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

router = APIRouter(prefix="/api/research", tags=["research"])

# Track running research sessions
running_research: Dict[str, asyncio.Task] = {}


class StartResearchRequest(BaseModel):
    """Request to start research."""
    goal: str
    time_limit: int = 60
    autonomous: bool = True


class ResearchStatusResponse(BaseModel):
    """Response with research status."""
    session_id: str
    status: str  # 'starting', 'running', 'completed', 'error'
    message: Optional[str] = None


async def run_research_background(session_id: str, goal: str, time_limit: int):
    """
    Run research in the background.

    This imports and runs the actual ResearchHarness.
    """
    print(f"\n{'='*60}")
    print(f"🔬 STARTING BACKGROUND RESEARCH")
    print(f"   Session ID: {session_id}")
    print(f"   Goal: {goal}")
    print(f"   Time Limit: {time_limit} minutes")
    print(f"{'='*60}\n")

    try:
        print("📦 Importing ResearchHarness...")
        from src.agents.director import ResearchHarness
        from src.interaction import InteractionConfig
        print("✓ Import successful")

        # Create autonomous interaction config (no user prompts)
        print("⚙️  Creating interaction config...")
        interaction_config = InteractionConfig(
            enable_clarification=False,
            autonomous_mode=True,
        )
        print("✓ Config created")

        print("🚀 Starting research harness...")
        async with ResearchHarness(
            db_path="research.db",
            interaction_config=interaction_config
        ) as harness:
            print(f"✓ Harness initialized")
            print(f"🔍 Starting research for: {goal}")
            print(f"   Using existing session ID: {session_id}")

            # Pass existing session_id so we don't create a duplicate session
            result = await harness.research(
                goal=goal,
                time_limit_minutes=time_limit,
                existing_session_id=session_id
            )

            print(f"✅ Research completed successfully!")
            print(f"   Findings: {len(result.key_findings) if hasattr(result, 'key_findings') else 'N/A'}")
            return result

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR in background research for {session_id}")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        print(f"🧹 Cleaning up session {session_id}")
        if session_id in running_research:
            del running_research[session_id]
            print(f"   ✓ Removed from running_research")


@router.post("/start", response_model=ResearchStatusResponse)
async def start_research(
    request: StartResearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new research session.

    This creates the session AND starts the actual research process.
    Events will be emitted via WebSocket as research progresses.
    """
    from api.db import get_db

    # Create session in database
    db = await get_db()
    session = await db.create_session(request.goal, request.time_limit)
    session_id = session.id

    # Start research in background
    task = asyncio.create_task(
        run_research_background(session_id, request.goal, request.time_limit)
    )
    running_research[session_id] = task

    return ResearchStatusResponse(
        session_id=session_id,
        status="running",
        message=f"Research started. Connect to WebSocket /ws/{session_id} for live updates."
    )


@router.get("/{session_id}/status", response_model=ResearchStatusResponse)
async def get_research_status(session_id: str):
    """Get the status of a research session."""
    from api.db import get_db

    db = await get_db()
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if research is running
    is_running = session_id in running_research

    return ResearchStatusResponse(
        session_id=session_id,
        status="running" if is_running else session.status,
        message=f"Research is {'currently running' if is_running else session.status}"
    )


@router.post("/{session_id}/stop")
async def stop_research(session_id: str):
    """Stop a running research session."""
    if session_id not in running_research:
        raise HTTPException(status_code=404, detail="No running research found for this session")

    task = running_research[session_id]
    task.cancel()
    del running_research[session_id]

    return {"status": "stopped", "session_id": session_id}
