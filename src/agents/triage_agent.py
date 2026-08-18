from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.llm_providers import GenerationClient, Message, ProviderError
from src.agents.helpers.triage_taxonomy import FLOW_SUBFLOWS, SUBFLOW_DESCRIPTIONS
from src.agents.helpers.triage_agent_system_prompt import TRIAGE_SYSTEM_PROMPT

_ALL_FLOWS = tuple(FLOW_SUBFLOWS.keys())
_ALL_SUBFLOWS = tuple({sf for subs in FLOW_SUBFLOWS.values() for sf in subs})


# ---------------------------------------------------------------------------
# Chitchat categories (absorbed from chitchat_gate.py)
# ---------------------------------------------------------------------------

class ChitchatType(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    SMALL_TALK = "small_talk"
    NONE = "none"  # no chitchat framing present


_CANNED_RESPONSES: dict[ChitchatType, str] = {
    ChitchatType.GREETING: "Hi there! How can I help you today?",
    ChitchatType.FAREWELL: "Take care! Reach out anytime if you need anything else.",
    ChitchatType.THANKS: "You're welcome! Let me know if there's anything else I can help with.",
    ChitchatType.SMALL_TALK: "Happy to chat, but let me know if there's something I can help you with today!",
}
_DEFAULT_CANNED_RESPONSE = "How can I help you today?"


# ---------------------------------------------------------------------------
# Combined structured output
# ---------------------------------------------------------------------------

class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_actionable_request: bool = Field(
        description="True if the message contains any real customer-service request, "
                     "even alongside a greeting/thanks/farewell.",
    )
    chitchat_type: ChitchatType = Field(
        description="The greeting/farewell/thanks/small_talk framing present, if any. "
                     "'none' if has_actionable_request is true with no chitchat framing.",
    )

    flow: Optional[Literal[_ALL_FLOWS]] = Field(default=None)
    subflow: Optional[Literal[_ALL_SUBFLOWS]] = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=500)
    needs_clarification: bool = Field(
        description="True when there's an actionable request but it can't be reliably mapped.",
    )

    @field_validator("subflow")
    @classmethod
    def validate_subflow(cls, value: Optional[str], info) -> Optional[str]:
        if value is not None:
            flow = info.data.get("flow")
            if flow is None or value not in FLOW_SUBFLOWS.get(flow, []):
                raise ValueError(f"Invalid subflow '{value}' for flow '{flow}'")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> "TriageResult":
        if not self.has_actionable_request:
            if self.flow is not None or self.subflow is not None:
                raise ValueError("flow/subflow must be null when there is no actionable request")
        elif not self.needs_clarification:
            if self.flow is None or self.subflow is None:
                raise ValueError(
                    "an actionable, non-clarification result requires both flow and subflow"
                )
        return self

    def canned_response(self, *, active_ticket: bool = False, pending_question: Optional[str] = None) -> str:
        """Only meaningful when has_actionable_request is False."""
        base = _CANNED_RESPONSES.get(self.chitchat_type, _DEFAULT_CANNED_RESPONSE)
        if active_ticket and pending_question:
            return f"{base} {pending_question}"
        return base



# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TriageAgent:
    def __init__(
        self,
        llm: GenerationClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _messages(user_message: str, conversation_history: Optional[list[Message]]) -> list[Message]:
        messages = [Message(role="system", content=TRIAGE_SYSTEM_PROMPT)]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(Message(role="user", content=user_message))
        return messages

    async def classify(
        self,
        user_message: str,
        conversation_history: Optional[list[Message]] = None,
    ) -> TriageResult:
        if not user_message or not user_message.strip():
            return TriageResult(
                has_actionable_request=False,
                chitchat_type=ChitchatType.NONE,
                confidence=0.0,
                reasoning="Empty message.",
                needs_clarification=False,
            )

        try:
            response = await self.llm.generate(
                messages=self._messages(user_message.strip(), conversation_history),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                output_schema=TriageResult,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise RuntimeError("Triage generation failed") from exc

        return TriageResult.model_validate_json(response.text)


__all__ = [
    "FLOW_SUBFLOWS",
    "SUBFLOW_DESCRIPTIONS",
    "ChitchatType",
    "TriageResult",
    "TriageAgent",
]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def _example():
    from src.llm_providers import ProviderFactory, Provider

    llm = ProviderFactory.create(Provider.GEMINI, model="gemini-2.0-flash")
    agent = TriageAgent(llm=llm)

    for msg in ["hi", "hi, I want to cancel my order", "thanks so much!", "cancel order 4471"]:
        result = await agent.classify(msg)
        if not result.has_actionable_request:
            print(f"{msg!r} -> chitchat ({result.chitchat_type.value}): {result.canned_response()!r}")
        else:
            print(f"{msg!r} -> flow={result.flow} subflow={result.subflow} confidence={result.confidence}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())
