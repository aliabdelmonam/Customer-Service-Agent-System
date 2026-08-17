
# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------
from pyexpat.errors import messages

from .llm_interface import GenerationClient, GenerationResponse, Message, ProviderError, Provider
from typing import Any
from src.Utils import config, settings
from pydantic import BaseModel
from google.genai import types


class GeminiClient(GenerationClient):
    """Wraps Google's Gen AI SDK (google-genai)."""

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set and no api_key provided")
        super().__init__(model=model, api_key=api_key, **kwargs)

        from google import genai  # lazy import
        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return Provider.GEMINI.value

    async def generate(
    self,
    messages: list[Message],
    temperature: float = 0.3,
    max_tokens: int = 1024,
    output_schema: type[BaseModel] | None = None,
    **kwargs: Any,
) -> GenerationResponse:
        try:
            system_msgs = [m.content for m in messages if m.role == "system"]
            turn_msgs = [m for m in messages if m.role != "system"]

            contents = [
                {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]}
                for m in turn_msgs
            ]

            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                **kwargs,
            }

            if system_msgs:
                generation_config["system_instruction"] = "\n".join(system_msgs)

            if output_schema is not None:
                generation_config["response_mime_type"] = "application/json"
                # response_json_schema (not response_schema) accepts raw JSON
                # Schema as a plain dict — it isn't validated against Google's
                # restricted Schema pydantic type, so keywords like
                # "additionalProperties" from a normal Pydantic-generated
                # schema won't be rejected.
                generation_config["response_json_schema"] = output_schema.model_json_schema()

            resp = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**generation_config),
            )

            usage = {}
            if resp.usage_metadata:
                usage = {
                    "input_tokens": resp.usage_metadata.prompt_token_count,
                    "output_tokens": resp.usage_metadata.candidates_token_count,
                }

            return GenerationResponse(
                text=resp.text,
                provider=self.provider_name,
                model=self.model,
                raw=resp,
                usage=usage,
                finish_reason=(
                    resp.candidates[0].finish_reason.name if resp.candidates else None
                ),
            )
        except Exception as e:
            raise ProviderError(self.provider_name, e) from e

