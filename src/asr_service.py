"""Speech-to-text component for customer audio messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.Utils import settings


MAX_AUDIO_BYTES = 25 * 1024 * 1024


class ASRError(RuntimeError):
    """Raised when an audio transcription cannot be completed."""


@dataclass(frozen=True)
class Transcription:
    text: str


class GroqASRService:
    """Transcribes browser-recorded audio using Groq Whisper.

    The class is intentionally isolated from the chat agent so another ASR
    provider or a local Whisper implementation can replace it later.
    """

    def __init__(self, *, model: str = "whisper-large-v3-turbo") -> None:
        if not settings.GROQ_API_KEY:
            raise ASRError("GROQ_API_KEY is required for Groq speech-to-text")
        self.model = model
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

    async def transcribe(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        *,
        language: Optional[str] = None,
    ) -> Transcription:
        if not audio:
            raise ASRError("The recorded audio is empty")
        if len(audio) > MAX_AUDIO_BYTES:
            raise ASRError("The recorded audio exceeds the 25 MB upload limit")

        try:
            response = await self._client.audio.transcriptions.create(
                file=(filename, audio, content_type),
                model=self.model,
                response_format="json",
                language=language,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - expose a safe API error upstream
            raise ASRError("Speech transcription provider failed") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ASRError("No speech was detected in the recording")
        return Transcription(text=text)


__all__ = ["ASRError", "GroqASRService", "MAX_AUDIO_BYTES", "Transcription"]
