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

# Track running research sessions and their harnesses
running_research: dict[str, asyncio.Task] = {}
running_harnesses: dict[str, object] = {}  # session_id -> ResearchHarness


class StartResearchRequest(BaseModel):
    """Request to start research."""
    goal: str
    max_iterations: int = 5
    autonomous: bool = True
    enable_mid_questions: bool = False


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


async def run_research_background(session_id: str, goal: str, max_iterations: int, autonomous: bool = True, enable_mid_questions: bool = False):
    """
    Run research in the background.

    This imports and runs the actual ResearchHarness.
    """
    print(f"\n{'='*60}")
    print("🔬 STARTING BACKGROUND RESEARCH")
    print(f"   Session ID: {session_id}")
    print(f"   Goal: {goal}")
    print(f"   Iterations: {max_iterations}")
    print(f"   Autonomous Mode: {autonomous}")
    print(f"   Mid-Research Questions: {enable_mid_questions}")
    print(f"{'='*60}\n")

    try:
        print("📦 Importing ResearchHarness...")
        from api.ui_interaction import UIInteraction
        from src.agents.director import ResearchHarness
        from src.interaction import InteractionConfig
        print("✓ Import successful")

        # Update session status to running
        from api.db import get_db
        db = await get_db()
        await db.update_session_status(session_id, "running")

        # Create interaction config
        print("⚙️  Creating interaction config...")
        interaction_config = InteractionConfig(
            enable_clarification=False,  # Pre-research clarification handled separately
            enable_async_questions=enable_mid_questions,  # Enable mid-research questions
            autonomous_mode=autonomous and not enable_mid_questions,
            question_timeout=60,
            max_questions_per_session=5,
        )
        print(f"✓ Config created (mid_questions: {enable_mid_questions}, autonomous: {autonomous})")

        print("🚀 Starting research harness...")
        async with ResearchHarness(
            db_path="research.db",
            interaction_config=interaction_config
        ) as harness:
            print("✓ Harness initialized")

            # Save harness reference so pause endpoint can signal it
            running_harnesses[session_id] = harness

            # Replace CLI interaction with UI interaction if mid-questions enabled
            if enable_mid_questions:
                print("🔄 Setting up UI interaction handler...")
                ui_interaction = UIInteraction(
                    session_id=session_id,
                    config=interaction_config,
                    llm_callback=harness.director._interaction_llm_callback,
                )
                # Replace manager's interaction handler
                harness.director.manager.interaction = ui_interaction
                print("✓ UI interaction configured")

            print(f"🔍 Starting research for: {goal}")
            print(f"   Using existing session ID: {session_id}")

            # Pass existing session_id so we don't create a duplicate session
            result = await harness.research(
                goal=goal,
                max_iterations=max_iterations,
                existing_session_id=session_id
            )

            # Check if research was paused (don't overwrite status)
            session = await db.get_session(session_id)
            if session and session.status == "paused":
                print("⏸️  Research paused. State saved.")
                return result

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
        if session_id in running_harnesses:
            del running_harnesses[session_id]
            print("   ✓ Removed from running_harnesses")


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
    session = await db.create_session(request.goal, request.max_iterations)
    session_id = session.id

    # Start research in background
    task = asyncio.create_task(
        run_research_background(
            session_id,
            request.goal,
            request.max_iterations,
            request.autonomous,
            request.enable_mid_questions
        )
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


@router.post("/{session_id}/pause")
async def pause_research(session_id: str):
    """Pause a running research session. State is saved for later resume."""
    harness = running_harnesses.get(session_id)
    if not harness or not hasattr(harness, "director"):
        raise HTTPException(status_code=404, detail="No running research found for this session")

    harness.director.pause_research()

    # Immediately emit a WebSocket event so the UI gets feedback
    try:
        from src.events import emit_agent_event
        await emit_agent_event(
            session_id=session_id,
            event_type="system",
            agent="director",
            data={"message": "Pause requested - finishing current operation..."},
        )
    except Exception:
        pass

    return {"status": "pausing", "session_id": session_id}


async def run_research_resume_background(session_id: str):
    """Resume a paused or crashed research session in the background."""
    print(f"\n{'='*60}")
    print("🔄 RESUMING BACKGROUND RESEARCH")
    print(f"   Session ID: {session_id}")
    print(f"{'='*60}\n")

    try:
        from api.db import get_db
        from src.agents.director import ResearchHarness
        from src.interaction import InteractionConfig

        db = await get_db()
        session = await db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Don't update status here - the Director's resume flow handles it
        # after loading and validating the session status

        interaction_config = InteractionConfig(
            enable_clarification=False,
            autonomous_mode=True,
        )

        async with ResearchHarness(
            db_path="research.db",
            interaction_config=interaction_config,
        ) as harness:
            # Save harness reference
            running_harnesses[session_id] = harness

            print(f"🔍 Resuming research: {session.goal}")

            result = await harness.research(
                goal=session.goal,
                max_iterations=session.max_iterations if hasattr(session, "max_iterations") else 5,
                existing_session_id=session_id,
                resume=True,
            )

            # Check if paused again
            refreshed = await db.get_session(session_id)
            if refreshed and refreshed.status == "paused":
                print("⏸️  Research paused again. State saved.")
                return result

            print("✅ Resumed research completed successfully!")

            from datetime import datetime
            await db.update_session_status(session_id, "completed", ended_at=datetime.now())
            return result

    except Exception as e:
        print(f"❌ ERROR resuming research for {session_id}: {e}")
        import traceback
        traceback.print_exc()

        try:
            from api.db import get_db
            db = await get_db()
            await db.update_session_status(session_id, "error")
        except Exception:
            pass
        raise

    finally:
        print(f"🧹 Cleaning up resumed session {session_id}")
        if session_id in running_research:
            del running_research[session_id]
        if session_id in running_harnesses:
            del running_harnesses[session_id]


@router.post("/{session_id}/resume", response_model=ResearchStatusResponse)
async def resume_research(session_id: str):
    """Resume a paused or crashed research session."""
    from api.db import get_db

    db = await get_db()
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("paused", "crashed"):
        raise HTTPException(
            status_code=400,
            detail=f"Session is '{session.status}', not 'paused' or 'crashed'"
        )

    # Start research resume in background
    task = asyncio.create_task(run_research_resume_background(session_id))
    running_research[session_id] = task

    return ResearchStatusResponse(
        session_id=session_id,
        status="running",
        message=f"Research resuming. Connect to WebSocket /ws/{session_id} for live updates.",
    )


class AnswerQuestionRequest(BaseModel):
    """Request to answer a mid-research question."""
    question_id: str
    response: str


@router.post("/{session_id}/answer")
async def answer_question(session_id: str, request: AnswerQuestionRequest):
    """Answer a mid-research question."""
    from api.question_manager import get_question_manager

    question_manager = get_question_manager()
    success = question_manager.answer_question(request.question_id, request.response)

    if not success:
        raise HTTPException(status_code=404, detail="Question not found or already answered")

    return {"status": "answered", "question_id": request.question_id}
