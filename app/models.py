"""
Pydantic models for the SHL Assessment Recommender API.

CRITICAL: The response schema is non-negotiable for the automated evaluator.
- recommendations: EMPTY array when clarifying/refusing, 1-10 items when recommending
- end_of_conversation: true only when the agent considers the task complete
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")


class AssessmentCard(BaseModel):
    """A single assessment recommendation. Fields match SHL catalog."""
    name: str
    url: str
    description: str
    duration: str
    remote: str
    adaptive: str
    job_levels: list[str]
    categories: list[str]


class ChatResponse(BaseModel):
    """
    Response schema — non-negotiable for the evaluator.
    - recommendations is EMPTY when agent is clarifying or refusing.
    - recommendations has 1-10 items when agent commits to a shortlist.
    - end_of_conversation is true only when the agent considers task complete.
    """
    session_id: str
    response: str = Field(..., description="Agent's text reply")
    recommendations: list[AssessmentCard] = Field(
        default_factory=list,
        description="Empty when clarifying; 1-10 items when recommending"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )


class HealthResponse(BaseModel):
    """Health check — must return status: ok"""
    status: str = "ok"
