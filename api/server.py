"""
FastAPI server for claude-researcher UI.

Provides REST API + WebSocket endpoints for real-time research monitoring.
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Set

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
# Mark this process as the API server to avoid event proxy loops
os.environ.setdefault("CLAUDE_RESEARCHER_IN_API", "1")

from api.models import HealthResponse
from api.routes import sessions
from api.routes import research
from api.routes import events
from api.routes import findings
from api.db import get_db, close_db
from api.events import get_event_emitter, emit_event

# Server state
START_TIME = time.time()
active_websockets: Set[WebSocket] = set()
active_sessions: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 FastAPI server starting...")
    # Initialize database connection
    await get_db()
    print("✓ Database connected")
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

# Configure CORS for Next.js frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions.router)
app.include_router(research.router)
app.include_router(events.router)
app.include_router(findings.router)


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

    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected for session: {session_id}")
    finally:
        # Unsubscribe from events
        await emitter.unsubscribe(session_id, websocket)
        active_websockets.discard(websocket)
        print(f"   Remaining subscribers: {emitter.get_subscriber_count(session_id)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
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
