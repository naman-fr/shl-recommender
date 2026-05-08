"""
Pydantic models for the SHL Assessment Recommender API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""
    session_id: str = Field(..., description="Unique session identifier for conversation continuity")
    message: str = Field(..., min_length=1, max_length=2000, description="User's message text")


class AssessmentCard(BaseModel):
    """A single assessment recommendation returned to the user."""
    name: str
    url: str
    description: str
    duration: str
    remote: str
    adaptive: str
    job_levels: list[str]
    categories: list[str]


class ChatResponse(BaseModel):
    """Response from the chat agent."""
    session_id: str
    response: str = Field(..., description="Agent's text reply")
    recommendations: list[AssessmentCard] = Field(
        default_factory=list,
        description="Assessment recommendations (if any)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
