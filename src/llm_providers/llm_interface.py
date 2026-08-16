
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pydantic_settings import BaseModel

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    COHERE = "cohere"
    GEMINI = "gemini"
    GROK = "grok"


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResponse:
    text: str
    provider: str
    model: str
    raw: Any = None                      # original SDK response, for debugging
    usage: dict = field(default_factory=dict)   # {"input_tokens": .., "output_tokens": ..}
    finish_reason: Optional[str] = None


class ProviderError(Exception):
    """Raised when a provider call fails, wrapping the original exception."""
    def __init__(self, provider: str, original: Exception):
        self.provider = provider
        self.original = original
        super().__init__(f"[{provider}] generation failed: {original}")


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------

class GenerationClient(ABC):
    """Every provider implementation must satisfy this interface."""

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.extra_config = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Send messages, return a normalized GenerationResponse."""
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        raise NotImplementedError
