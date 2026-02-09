"""
FastAPI server for claude-researcher UI.

Provides REST API + WebSocket endpoints for real-time research monitoring.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import HealthResponse
from api.routes import sessions

# Server state
START_TIME = time.time()
active_websockets: Set[WebSocket] = set()
active_sessions: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 FastAPI server starting...")
    yield
    print("🛑 FastAPI server shutting down...")


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


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime=time.time() - START_TIME
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time research updates.

    Agents emit events → EventEmitter → WebSocket → UI
    """
    await websocket.accept()
    active_websockets.add(websocket)

    try:
        print(f"📡 WebSocket connected for session: {session_id}")

        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": time.time()
        })

        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            # Echo back for now (will implement proper handling later)
            await websocket.send_json({
                "type": "echo",
                "message": data
            })

    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected for session: {session_id}")
    finally:
        active_websockets.discard(websocket)


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
