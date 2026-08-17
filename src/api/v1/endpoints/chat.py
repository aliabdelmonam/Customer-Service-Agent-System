"""Chat endpoint for the customer-service orchestration pipeline."""

from __future__ import annotations

import logging
import os
import uuid
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.agents.orchestrator import Orchestrator
from src.llm_providers import ProviderFactory


router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """One customer message in an ongoing conversation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    user_id: str | None = Field(
        default=None, min_length=1, max_length=128, description="Stable customer ID, if available."
    )
    session_id: str | None = Field(
        default=None, min_length=1, max_length=128, description="Conversation ID returned by this endpoint."
    )
    message: str = Field(
        min_length=1,
        max_length=10_000,
        validation_alias=AliasChoices("message", "query"),
        description="Customer's message. The frontend may also send this as 'query'.",
    )


class ChatResponse(BaseModel):
    """The response and lifecycle state for this conversation turn."""

    message: str
    ticket_id: str | None = None
    session_id: str
    finished: bool


_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.1-flash-lite",
    "cohere": "command-r-plus",
}


@lru_cache(maxsize=1)
def build_orchestrator() -> Orchestrator:
    """Create the process-wide orchestrator from environment configuration."""
    provider = os.getenv("GENERATION_BACKEND", "gemini").lower()
    model = os.getenv("GENERATION_MODEL", _DEFAULT_MODELS.get(provider, "openai/gpt-oss-120b"))
    return Orchestrator(ProviderFactory.create(provider, model=model))


def get_orchestrator(request: Request) -> Orchestrator:
    """Use an app-provided orchestrator when testing, otherwise use the default."""
    return getattr(request.app.state, "orchestrator", None) or build_orchestrator()


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    payload: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Send a customer message through triage, resolution, and escalation."""
    try:
        # A newly generated session ID is returned so a frontend without a
        # logged-in user can send it back as ``session_id`` on the next turn.
        conversation_id = payload.user_id or payload.session_id or str(uuid.uuid4())
        result = await orchestrator.handle_message(conversation_id, payload.message)
    except Exception as exc:  # noqa: BLE001 - backend/provider failures are unavailable service errors
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The customer-service agent is temporarily unavailable. Please try again.",
        ) from exc

    return ChatResponse(
        message=result.message,
        ticket_id=result.ticket_id,
        session_id=conversation_id,
        finished=result.finished,
    )


__all__ = ["router", "ChatRequest", "ChatResponse", "build_orchestrator", "get_orchestrator"]
