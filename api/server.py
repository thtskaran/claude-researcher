"""
FastAPI server for claude-researcher UI.

Provides REST API + WebSocket endpoints for real-time research monitoring.
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
# Mark this process as the API server to avoid event proxy loops
os.environ.setdefault("CLAUDE_RESEARCHER_IN_API", "1")

from api.db import close_db, get_db
from api.events import emit_event, get_event_emitter
from api.models import HealthResponse
from api.routes import agents as agents_routes
from api.routes import events, findings, report, research, sessions
from api.routes import knowledge as knowledge_routes
from api.routes import verification as verification_routes

# Server state
START_TIME = time.time()
active_websockets: set[WebSocket] = set()
active_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 FastAPI server starting...")
    # Initialize database connection
    db = await get_db()
    print("✓ Database connected")

    # Crash recovery: mark any sessions left as 'running' as 'crashed'
    try:
        cursor = await db._connection.execute(
            "SELECT COUNT(*) as count FROM sessions WHERE status = 'running'"
        )
        row = await cursor.fetchone()
        crashed_count = row["count"] if row else 0

        if crashed_count > 0:
            await db._connection.execute(
                "UPDATE sessions SET status = 'crashed' WHERE status = 'running'"
            )
            await db._connection.commit()
            print(f"⚠️  Marked {crashed_count} previously-running session(s) as 'crashed'")
    except Exception as e:
        print(f"⚠️  Could not check for crashed sessions: {e}")

    yield
    print("🛑 FastAPI server shutting down...")
    # Close database connection
    await close_db()
    print("✓ Database closed")


# Create FastAPI app
app = FastAPI(
    title="Claude Researcher API",
    description="Backend API for hierarchical AI research system",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions.router)
app.include_router(research.router)
app.include_router(events.router)
app.include_router(findings.router)
app.include_router(report.router)
app.include_router(knowledge_routes.router)
app.include_router(verification_routes.router)
app.include_router(agents_routes.router)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime=time.time() - START_TIME
    )


@app.post("/api/test/emit/{session_id}")
async def test_emit_event(session_id: str, event_type: str = "test", message: str = "Test event"):
    """
    Test endpoint to emit events manually.

    Useful for testing WebSocket functionality without running actual research.
    """
    # [HARDENED] SEC-003: Only allow in debug mode to prevent arbitrary event injection
    if not os.environ.get("CLAUDE_RESEARCHER_DEBUG"):
        return JSONResponse(status_code=404, content={"error": "Not found"})

    emitter = get_event_emitter()

    await emit_event(
        session_id=session_id,
        event_type=event_type,
        agent="test",
        data={
            "message": message,
            "timestamp": time.time()
        }
    )

    return {
        "status": "emitted",
        "session_id": session_id,
        "event_type": event_type,
        "subscribers": emitter.get_subscriber_count(session_id)
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time research updates.

    Agents emit events → EventEmitter → WebSocket → UI
    """
    await websocket.accept()
    active_websockets.add(websocket)

    # Get event emitter and subscribe to this session
    emitter = get_event_emitter()
    await emitter.subscribe(session_id, websocket)

    try:
        print(f"📡 WebSocket connected for session: {session_id}")
        print(f"   Active subscribers: {emitter.get_subscriber_count(session_id)}")

        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": time.time(),
            "subscribers": emitter.get_subscriber_count(session_id)
        })

        # Send a welcome event
        await emit_event(
            session_id=session_id,
            event_type="system",
            agent="server",
            data={
                "message": "WebSocket connected successfully",
                "status": "ready"
            }
        )

        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()

            # Handle ping messages
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time()
                })
            else:
                # Echo other messages for debugging
                await websocket.send_json({
                    "type": "echo",
                    "message": data
                })

    except (WebSocketDisconnect, RuntimeError):
        print(f"📡 WebSocket disconnected for session: {session_id}")
    finally:
        # Unsubscribe from events
        await emitter.unsubscribe(session_id, websocket)
        active_websockets.discard(websocket)
        print(f"   Remaining subscribers: {emitter.get_subscriber_count(session_id)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    import logging

    logging.getLogger(__name__).error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🔬 Claude Researcher API Server")
    print("=" * 60)
    print("📍 Server: http://localhost:8080")
    print("📍 Docs: http://localhost:8080/docs")
    print("📍 WebSocket: ws://localhost:8080/ws/{session_id}")
    print("=" * 60)

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
