# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible API)
# ---------------------------------------------------------------------------
from .llm_interface import GenerationClient, GenerationResponse, Message, ProviderError, Provider
from typing import Any
from src.Utils import settings
from pydantic import BaseModel

class GroqClient(GenerationClient):
    """Wraps Groq's API, which is OpenAI-compatible — reuses the
    `openai` SDK pointed at Groq's base URL instead of a bespoke client."""

    def __init__(self, model: str = "openai/gpt-oss-120b", **kwargs):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not set and no api_key provided")
        super().__init__(model=model, api_key=api_key, **kwargs)

        from openai import AsyncOpenAI  # lazy import
        self._client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    @property
    def provider_name(self) -> str:
        return Provider.GROQ.value

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        try:
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]

            request_params: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

            # Inject JSON schema into Groq / OpenAI response_format if provided
            if output_schema is not None:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_schema.__name__,
                        "schema": output_schema.model_json_schema(),
                    },
                }

            resp = await self._client.chat.completions.create(**request_params)
            choice = resp.choices[0]
            usage = {}
            if resp.usage:
                usage = {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                }
            return GenerationResponse(
                text=choice.message.content or "",
                provider=self.provider_name,
                model=self.model,
                raw=resp,
                usage=usage,
                finish_reason=choice.finish_reason,
            )
        except Exception as e:
            raise ProviderError(self.provider_name, e) from e
