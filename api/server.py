"""
FastAPI server for claude-researcher UI.

Provides REST API + WebSocket endpoints for real-time research monitoring.
"""
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
# Mark this process as the API server to avoid event proxy loops
os.environ.setdefault("CLAUDE_RESEARCHER_IN_API", "1")

from src.logging_config import setup_logging, get_logger

# Initialize centralized file logging for API server
setup_logging()
_api_logger = get_logger(__name__)

from api.db import close_db, get_db
from api.events import emit_event, get_event_emitter
from api.models import HealthResponse
from api.routes import agents as agents_routes
from api.routes import events, findings, report, research, sessions
from api.routes import knowledge as knowledge_routes
from api.routes import verification as verification_routes

# Server state
START_TIME = time.time()

# Auth: optional API key from env. When set, all HTTP endpoints require
# Authorization: Bearer <key>. When unset, no auth (local dev mode).
_API_KEY = os.environ.get("CLAUDE_RESEARCHER_API_KEY")


class _AuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid Bearer token (if API key is configured)."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth when no key is configured (local dev)
        if not _API_KEY:
            return await call_next(request)
        # Always allow health check and docs
        if request.url.path in ("/", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {_API_KEY}":
            return await call_next(request)
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-IP sliding-window rate limiter (in-memory)."""

    def __init__(self, app, requests_per_minute: int = 120):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = now - 60.0
        # Prune old entries
        hits = self._hits[ip]
        self._hits[ip] = hits = [t for t in hits if t > window]
        if len(hits) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests"},
            )
        hits.append(now)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("FastAPI server starting...")
    # Initialize database connection
    db = await get_db()
    print("Database connected")

    # Register the event emitter with the agent-layer event system
    # so agents can emit events without importing from api/
    try:
        from src.events import register_emitter
        emitter = get_event_emitter()
        register_emitter(
            emitter=emit_event,
            subscriber_count=emitter.get_subscriber_count,
        )
        print("Event emitter registered with agent layer")
    except Exception as e:
        print(f"Warning: Could not register event emitter: {e}")

    # Crash recovery: mark any sessions left as 'running' as 'crashed'
    try:
        crashed_count = await db.mark_crashed_sessions()
        if crashed_count > 0:
            print(f"Marked {crashed_count} previously-running session(s) as 'crashed'")
    except Exception as e:
        print(f"Could not check for crashed sessions: {e}")

    yield
    print("FastAPI server shutting down...")
    # Close active WebSocket subscribers
    emitter = get_event_emitter()
    for sid in emitter.get_all_sessions():
        _api_logger.info("Closing subscribers for session %s", sid)
    # Close database connection
    await close_db()
    print("Database closed")


# Create FastAPI app
app = FastAPI(
    title="Claude Researcher API",
    description="Backend API for hierarchical AI research system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: restrict to localhost origins by default.
# Set CLAUDE_RESEARCHER_CORS_ORIGINS=* to allow all (not recommended).
_cors_env = os.environ.get("CLAUDE_RESEARCHER_CORS_ORIGINS", "")
_cors_origins = (
    ["*"] if _cors_env == "*"
    else [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_env != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (applied before auth so abusers are blocked early)
app.add_middleware(_RateLimitMiddleware, requests_per_minute=120)
# API key auth (optional — only active when CLAUDE_RESEARCHER_API_KEY is set)
app.add_middleware(_AuthMiddleware)

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

    Agents emit events -> EventEmitter -> WebSocket -> UI
    """
    # Validate session_id format (7-char hex)
    if not session_id or len(session_id) > 16 or not all(
        c in "0123456789abcdef" for c in session_id
    ):
        await websocket.close(code=1008, reason="Invalid session_id")
        return

    await websocket.accept()

    # Get event emitter and subscribe to this session
    emitter = get_event_emitter()
    await emitter.subscribe(session_id, websocket)

    try:
        _api_logger.info(
            "WebSocket connected: session=%s subscribers=%d",
            session_id, emitter.get_subscriber_count(session_id),
        )

        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": time.time(),
            "subscribers": emitter.get_subscriber_count(session_id),
        })

        # Send a welcome event
        await emit_event(
            session_id=session_id,
            event_type="system",
            agent="server",
            data={
                "message": "WebSocket connected successfully",
                "status": "ready",
            },
        )

        # Keep connection alive and listen for client messages.
        # Server-side heartbeat: if no message arrives within 60s
        # we send a ping ourselves to detect dead connections.
        import asyncio as _aio

        while True:
            try:
                data = await _aio.wait_for(
                    websocket.receive_text(), timeout=60.0,
                )
            except TimeoutError:
                # No message in 60s — send a server-side ping
                try:
                    await websocket.send_json({
                        "type": "ping", "timestamp": time.time(),
                    })
                except Exception:
                    break  # connection dead
                continue

            # Handle ping messages
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time(),
                })

    except (WebSocketDisconnect, RuntimeError):
        _api_logger.info("WebSocket disconnected: session=%s", session_id)
    finally:
        # Unsubscribe from events
        await emitter.unsubscribe(session_id, websocket)


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
