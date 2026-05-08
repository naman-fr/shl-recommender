"""
SHL Assessment Recommender — FastAPI Application

Exposes:
  GET  /health  → Health check (returns {"status": "ok"})
  POST /chat    → Conversational chat endpoint
  GET  /        → Frontend UI
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from app.models import ChatRequest, ChatResponse, HealthResponse
from app.agent import SHLAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global agent instance ────────────────────────────────────────────────
agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent on startup."""
    global agent
    logger.info("Initializing SHL Assessment Recommender agent...")
    agent = SHLAgent()
    logger.info(
        f"Agent ready. Catalog loaded with {agent.retriever.get_catalog_size()} assessments."
    )
    yield
    logger.info("Shutting down agent.")


# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Health check endpoint. Returns {"status": "ok"}."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and return the agent's response.
    
    Schema contract (non-negotiable for evaluator):
    - recommendations: EMPTY when clarifying/refusing, 1-10 items when recommending
    - end_of_conversation: true only when agent considers task complete
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        response_text, recommendations, end_of_conversation = await agent.chat(
            session_id=request.session_id,
            user_message=request.message,
        )

        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred processing your message: {str(e)}",
        )


@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_frontend():
    """Serve the chat frontend."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Frontend not found</h1><p>Place index.html in frontend/</p>",
        status_code=404,
    )
