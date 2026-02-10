"""
Research execution routes.

Handles starting and managing actual research runs from the UI.
"""
import asyncio
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

router = APIRouter(prefix="/api/research", tags=["research"])

# Track running research sessions
running_research: dict[str, asyncio.Task] = {}


class StartResearchRequest(BaseModel):
    """Request to start research."""
    goal: str
    time_limit: int = 60
    autonomous: bool = True


class ClarifyRequest(BaseModel):
    """Request to generate clarification questions."""
    goal: str
    max_questions: int = 4


class EnrichRequest(BaseModel):
    """Request to enrich a goal using clarifications."""
    goal: str
    questions: list[dict]
    answers: dict[str, str]


async def _haiku_callback(prompt: str) -> str:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    options = ClaudeAgentOptions(
        model="haiku",
        max_turns=1,
        allowed_tools=[],
    )

    response_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
    return response_text


def _build_interaction(max_questions: int):
    from src.interaction import InteractionConfig, UserInteraction

    config = InteractionConfig.interactive()
    config.max_clarification_questions = max(1, min(6, max_questions))
    return UserInteraction(config=config, llm_callback=_haiku_callback)


class ResearchStatusResponse(BaseModel):
    """Response with research status."""
    session_id: str
    status: str  # 'starting', 'running', 'completed', 'error'
    message: str | None = None


async def run_research_background(session_id: str, goal: str, time_limit: int):
    """
    Run research in the background.

    This imports and runs the actual ResearchHarness.
    """
    print(f"\n{'='*60}")
    print("🔬 STARTING BACKGROUND RESEARCH")
    print(f"   Session ID: {session_id}")
    print(f"   Goal: {goal}")
    print(f"   Time Limit: {time_limit} minutes")
    print(f"{'='*60}\n")

    try:
        print("📦 Importing ResearchHarness...")
        from src.agents.director import ResearchHarness
        from src.interaction import InteractionConfig
        print("✓ Import successful")

        # Update session status to running
        from api.db import get_db
        db = await get_db()
        await db.update_session_status(session_id, "running")

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
            print("✓ Harness initialized")
            print(f"🔍 Starting research for: {goal}")
            print(f"   Using existing session ID: {session_id}")

            # Pass existing session_id so we don't create a duplicate session
            result = await harness.research(
                goal=goal,
                time_limit_minutes=time_limit,
                existing_session_id=session_id
            )

            print("✅ Research completed successfully!")
            findings_count = len(result.key_findings) if hasattr(result, 'key_findings') else 'N/A'
            print(f"   Findings: {findings_count}")

            # Update session status to completed
            from datetime import datetime
            await db.update_session_status(session_id, "completed", ended_at=datetime.now())

            return result

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR in background research for {session_id}")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()

        # Update session status to error
        try:
            from api.db import get_db
            db = await get_db()
            await db.update_session_status(session_id, "error")
        except Exception:
            pass

        raise
    finally:
        # Clean up
        print(f"🧹 Cleaning up session {session_id}")
        if session_id in running_research:
            del running_research[session_id]
            print("   ✓ Removed from running_research")


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


@router.post("/clarify")
async def clarify_goal(request: ClarifyRequest):
    """Generate clarification questions for a research goal."""
    interaction = _build_interaction(request.max_questions)

    questions = await interaction._generate_clarification_questions(request.goal)
    questions = questions[: interaction.config.max_clarification_questions]

    return {"questions": [q.model_dump() for q in questions]}


@router.post("/enrich")
async def enrich_goal(request: EnrichRequest):
    """Enrich a research goal with user clarifications."""
    from src.interaction import UserInteraction
    from src.interaction.models import ClarificationQuestion

    interaction = UserInteraction(config=None, llm_callback=_haiku_callback)

    questions = [ClarificationQuestion(**q) for q in request.questions]
    answers: dict[int, str] = {}
    for key, value in request.answers.items():
        try:
            answers[int(key)] = value
        except Exception:
            continue

    enriched = await interaction._enrich_goal(request.goal, questions, answers)
    return {"enriched_goal": enriched}


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
