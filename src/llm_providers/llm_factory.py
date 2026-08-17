"""
LLM Provider Factory
=====================

Unified interface over multiple LLM providers (Cohere, Google Gemini,  Groq)
so the rest of the agent (triage, resolution, etc.) calls one consistent
`GenerationClient.generate(...)` regardless of which provider is behind it.

Install what you need:
    pip install cohere --break-system-packages
    pip install google-genai --break-system-packages
    pip install openai --break-system-packages   #  Groq is OpenAI-compatible
"""
from __future__ import annotations
from typing import Any, Optional
from .llm_interface import GenerationClient,Provider, Message
from .cohere_provider import CohereClient
from .google_provider import GeminiClient
from .groq_provider import GroqClient
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[Provider, type[GenerationClient]] = {
    Provider.COHERE: CohereClient,
    Provider.GEMINI: GeminiClient,
    Provider.GROQ: GroqClient,
}


class ProviderFactory:
    """Single entry point for building a GenerationClient.

    Usage:
        client = ProviderFactory.create("cohere", model="command-r-plus")
        client = ProviderFactory.create(Provider.GEMINI, model="gemini-2.0-flash")
    """

    @staticmethod
    def create(
        provider: str | Provider,
        model: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerationClient:
        if isinstance(provider, str):
            try:
                provider = Provider(provider.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Valid options: {[p.value for p in Provider]}"
                )

        client_cls = _PROVIDER_REGISTRY.get(provider)
        if client_cls is None:
            raise ValueError(f"No client registered for provider '{provider}'")

        return client_cls(model=model, **kwargs)

    @staticmethod
    def register(provider: Provider, client_cls: type[GenerationClient]) -> None:
        """Allows adding a new provider later without editing this file."""
        _PROVIDER_REGISTRY[provider] = client_cls


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def _example():
    client = ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite")

    messages = [
        Message(role="system", content="You are a concise customer support triage agent."),
        Message(role="user", content="I want to cancel order #4471."),
    ]

    response = await client.generate(messages, temperature=0.2, max_tokens=256)
    print(response.provider, response.model)
    print(response.text)
    print(response.usage)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())