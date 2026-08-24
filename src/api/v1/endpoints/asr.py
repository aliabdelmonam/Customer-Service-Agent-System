"""Audio transcription endpoint used by the browser microphone control."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from src.asr_service import ASRError, GroqASRService, MAX_AUDIO_BYTES


router = APIRouter()
logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    text: str


@lru_cache(maxsize=1)
def build_asr_service() -> GroqASRService:
    return GroqASRService()


def get_asr_service(request: Request) -> GroqASRService:
    """Allow an app-provided service for tests; otherwise use the Groq service."""
    return getattr(request.app.state, "asr_service", None) or build_asr_service()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    asr_service: GroqASRService = Depends(get_asr_service),
) -> TranscriptionResponse:
    """Transcribe an audio upload into text for the normal chat endpoint."""
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload an audio file.")

    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Audio must be 25 MB or smaller.")

    try:
        result = await asr_service.transcribe(
            content,
            audio.filename or "recording.webm",
            audio.content_type or "audio/webm",
            language=language,
        )
    except ASRError as exc:
        logger.exception("Audio transcription failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech transcription is temporarily unavailable. Please try again.",
        ) from exc
    finally:
        await audio.close()

    return TranscriptionResponse(text=result.text)


__all__ = ["router", "TranscriptionResponse", "build_asr_service", "get_asr_service"]
