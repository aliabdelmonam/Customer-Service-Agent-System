# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------
from .llm_interface import GenerationClient, GenerationResponse, Message, ProviderError, Provider
from typing import Any, Optional
import os
from src.Utils import settings
from pydantic import BaseModel

class CohereClient(GenerationClient):
    """Wraps Cohere's chat API (ClientV2 / async)."""

    def __init__(self, model: str = "command-r-plus",  **kwargs):
        api_key = settings.COHERE_API_KEY
        if not api_key:
            raise ValueError("COHERE_API_KEY not set and no api_key provided")
        super().__init__(model=model, api_key=api_key, **kwargs)

        import cohere  # lazy import so unused providers don't require the dep
        self._client = cohere.AsyncClientV2(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return Provider.COHERE.value

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        try:
            cohere_messages = [{"role": m.role, "content": m.content} for m in messages]

            chat_kwargs = {
            "model": self.model,
            "messages": cohere_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
            }

            if output_schema is not None:
                chat_kwargs["response_format"] = {
                    "type": "json_object",
                    "json_schema": output_schema,
                }

            resp = await self._client.chat(**chat_kwargs)
            text = "".join(
                block.text for block in resp.message.content if block.type == "text"
            )
            usage = {}
            if resp.usage and resp.usage.billed_units:
                usage = {
                    "input_tokens": resp.usage.billed_units.input_tokens,
                    "output_tokens": resp.usage.billed_units.output_tokens,
                }
            return GenerationResponse(
                text=text,
                provider=self.provider_name,
                model=self.model,
                raw=resp,
                usage=usage,
                finish_reason=resp.finish_reason,
            )
        except Exception as e:
            raise ProviderError(self.provider_name, e) from e

