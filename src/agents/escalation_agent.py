"""Escalation agent.

Consumes a `ResolutionOutcome` with `outcome_type == HANDOFF_ESCALATION`
(from resolution_agent.py) plus its `TicketState`, and turns it into a
structured payload for a human -- not a free-text summary. Also phrases a
short customer-facing message.

Design choices, consistent with the rest of the project:
- Reason classification and priority are deterministic code, not an LLM
  judgment call. The resolution agent's escalation reasons are generated
  by our own code in a small number of consistent shapes (see
  FunctionExecutionError / _escalate call sites), so pattern-matching on
  them is reliable -- this is not fuzzy text the LLM needs to interpret.
- The LLM is used narrowly, once, only to phrase the customer-facing
  message naturally. It never decides priority, category, or what to tell
  the human team. If the LLM call fails, a fixed template is used instead
  -- a customer in an escalation moment must always get a response.
- Emotion is NOT computed here. Per resolution_agent's own docstring,
  emotion scoring lives in the orchestrator, upstream of both agents. This
  agent accepts an already-computed emotion dict as an optional input.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.llm_providers import GenerationClient, Message, ProviderError
from src.agents.resolution_agent import (
    AuditEntry,
    OutcomeType,
    ResolutionOutcome,
    StepType,
    TicketState,
)


# ---------------------------------------------------------------------------
# Classification -- deterministic, pattern-matched against OUR OWN reason
# strings (see resolution_agent.FunctionExecutionError / _escalate).
# ---------------------------------------------------------------------------

class EscalationReasonCategory(str, Enum):
    FUNCTION_FAILURE = "function_failure"       # a registered backend function raised
    MISSING_CAPABILITY = "missing_capability"   # no function registered for this step
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"   # malformed/looping sequence
    UNKNOWN = "unknown"


class EscalationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_FUNCTION_FAILURE_PATTERN = re.compile(r"\[(?P<function_name>[\w\-]+)\] failed")
_MISSING_FUNCTION_PATTERN = re.compile(r"No backend function named '(?P<function_name>[\w\-]+)'")


def classify_reason(reason: str) -> tuple[EscalationReasonCategory, Optional[str]]:
    """Returns (category, function_name_if_known). Pattern-matched against
    the fixed set of reason strings resolution_agent.py actually produces."""
    if match := _MISSING_FUNCTION_PATTERN.search(reason):
        return EscalationReasonCategory.MISSING_CAPABILITY, match.group("function_name")
    if match := _FUNCTION_FAILURE_PATTERN.search(reason):
        return EscalationReasonCategory.FUNCTION_FAILURE, match.group("function_name")
    if "exceeded max steps" in reason:
        return EscalationReasonCategory.MAX_STEPS_EXCEEDED, None
    return EscalationReasonCategory.UNKNOWN, None


def determine_priority(
    category: EscalationReasonCategory,
    emotion: Optional[dict[str, Any]],
) -> EscalationPriority:
    # Emotion overrides category-based priority when it's available and high --
    # an angry customer waiting on a config bug is still a HIGH priority wait,
    # regardless of how mundane the underlying cause is.
    if emotion is not None and emotion.get("anger", 0) >= 0.7:
        return EscalationPriority.HIGH
    if category in (EscalationReasonCategory.FUNCTION_FAILURE, EscalationReasonCategory.MISSING_CAPABILITY):
        return EscalationPriority.MEDIUM
    if category == EscalationReasonCategory.MAX_STEPS_EXCEEDED:
        return EscalationPriority.MEDIUM
    return EscalationPriority.LOW


_SUGGESTED_NEXT_STEP = {
    EscalationReasonCategory.FUNCTION_FAILURE: (
        "A backend function ('{function_name}') raised an error while processing this. "
        "Check that integration's logs, then either retry manually or complete the action by hand."
    ),
    EscalationReasonCategory.MISSING_CAPABILITY: (
        "No backend function is registered for '{function_name}'. This is a configuration gap, "
        "not a customer-caused issue -- likely needs an engineering fix, not just a manual resolution."
    ),
    EscalationReasonCategory.MAX_STEPS_EXCEEDED: (
        "The step sequence for {flow} / {subflow} looped without resolving -- likely a missing "
        "branch target in that sequence definition. Resolve manually for this customer, then flag "
        "the sequence for review."
    ),
    EscalationReasonCategory.UNKNOWN: (
        "Review the attempted steps below and determine the appropriate manual resolution."
    ),
}


def suggested_next_step(
    category: EscalationReasonCategory,
    function_name: Optional[str],
    flow: str,
    subflow: str,
) -> str:
    template = _SUGGESTED_NEXT_STEP[category]
    return template.format(function_name=function_name or "unknown", flow=flow, subflow=subflow)


# ---------------------------------------------------------------------------
# Attempted-steps summary -- pulled straight from the audit log already
# built by resolution_agent.py, filtered to the events a human actually
# needs, not every internal bookkeeping entry.
# ---------------------------------------------------------------------------

_RELEVANT_EVENTS = {
    "slot_extracted", "slot_derived", "function_executed",
    "check_evaluated", "asked_customer", "escalated",
}


class AttemptedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    step_type: StepType
    event: str
    detail: str


def summarize_attempted_steps(audit_log: list[AuditEntry]) -> list[AttemptedStep]:
    return [
        AttemptedStep(step_id=e.step_id, step_type=e.step_type, event=e.event, detail=e.detail)
        for e in audit_log
        if e.event in _RELEVANT_EVENTS
    ]


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

class EscalationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correlation_id: str
    flow: str
    subflow: str
    filled_slots: dict[str, Any]
    reason: str
    reason_category: EscalationReasonCategory
    priority: EscalationPriority
    attempted_steps: list[AttemptedStep]
    suggested_next_step: str
    emotion: Optional[dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Customer-facing message -- narrow LLM phrasing with a fixed fallback.
# ---------------------------------------------------------------------------

ESCALATION_PHRASING_SYSTEM_PROMPT = """You write ONE short, calm, reassuring message
telling a customer their request has been passed to a human team member. Do not
apologize excessively, do not explain the technical cause, do not invent a timeline
you were not given. Output only the message text, nothing else."""

_FALLBACK_MESSAGE = (
    "I've passed this along to a team member who can help further -- "
    "they'll follow up with you shortly."
)


class EscalationAgent:
    def __init__(
        self,
        llm: Optional[GenerationClient] = None,
        queue: Optional[Callable[[EscalationPayload], None]] = None,
        *,
        phrasing_temperature: float = 0.3,
    ) -> None:
        self.llm = llm
        self.queue = queue or self._default_queue
        self.phrasing_temperature = phrasing_temperature

    @staticmethod
    def _default_queue(payload: EscalationPayload) -> None:
        # Placeholder: replace with a real push (Slack, internal ticket
        # system, notify_internal_team backend function, etc).
        print(f"[escalation queue] {payload.model_dump_json(indent=2)}")

    def build_payload(
        self,
        state: TicketState,
        outcome: ResolutionOutcome,
        emotion: Optional[dict[str, Any]] = None,
    ) -> EscalationPayload:
        if outcome.outcome_type != OutcomeType.HANDOFF_ESCALATION:
            raise ValueError(
                f"EscalationAgent expects HANDOFF_ESCALATION, got {outcome.outcome_type!r}"
            )

        reason = outcome.reason or state.escalation_reason or "Unknown escalation reason"
        category, function_name = classify_reason(reason)
        priority = determine_priority(category, emotion)

        return EscalationPayload(
            correlation_id=state.correlation_id,
            flow=state.flow,
            subflow=state.subflow,
            filled_slots=dict(state.filled_slots),
            reason=reason,
            reason_category=category,
            priority=priority,
            attempted_steps=summarize_attempted_steps(state.audit_log),
            suggested_next_step=suggested_next_step(category, function_name, state.flow, state.subflow),
            emotion=emotion,
        )

    async def phrase_customer_message(self) -> str:
        if self.llm is None:
            return _FALLBACK_MESSAGE
        try:
            response = await self.llm.generate(
                messages=[Message(role="system", content=ESCALATION_PHRASING_SYSTEM_PROMPT)],
                temperature=self.phrasing_temperature,
                max_tokens=60,
            )
            text = response.text.strip()
            return text or _FALLBACK_MESSAGE
        except (ProviderError, Exception):  # noqa: BLE001 -- never let phrasing block an escalation
            return _FALLBACK_MESSAGE

    async def handle(
        self,
        state: TicketState,
        outcome: ResolutionOutcome,
        emotion: Optional[dict[str, Any]] = None,
    ) -> tuple[EscalationPayload, str]:
        """One-call entry point for the orchestrator: builds the payload,
        pushes it to the human queue, and returns (payload, customer_message)."""
        payload = self.build_payload(state, outcome, emotion)
        self.queue(payload)
        message = await self.phrase_customer_message()
        return payload, message


__all__ = [
    "EscalationReasonCategory",
    "EscalationPriority",
    "AttemptedStep",
    "EscalationPayload",
    "EscalationAgent",
    "classify_reason",
    "determine_priority",
    "suggested_next_step",
    "summarize_attempted_steps",
]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def _example():
    from resolution_agent import (
        AuditEntry, StepType, TicketState, TicketStatus, ResolutionOutcome, OutcomeType,
    )

    state = TicketState(
        correlation_id="corr-123",
        flow="Order Issue",
        subflow="Manage Cancel",
        current_step_id="cancel_action",
        filled_slots={"order_id": "4471", "shipping_status": "not_shipped"},
        status=TicketStatus.ESCALATED,
        escalation_reason="Function 'cancel_order' failed: [cancel_order] failed: connection timeout",
        audit_log=[
            AuditEntry(
                timestamp="2026-08-17T10:00:00Z", correlation_id="corr-123",
                step_id="ask_order_id", step_type=StepType.SLOT,
                event="slot_extracted", detail="order_id='4471' (confidence=0.95)",
            ),
            AuditEntry(
                timestamp="2026-08-17T10:00:05Z", correlation_id="corr-123",
                step_id="cancel_action", step_type=StepType.ACTION,
                event="escalated", detail="Function 'cancel_order' failed: connection timeout",
            ),
        ],
    )
    outcome = ResolutionOutcome(
        outcome_type=OutcomeType.HANDOFF_ESCALATION,
        state=state,
        reason=state.escalation_reason,
    )

    agent = EscalationAgent(llm=None)  # no LLM -- exercises the fallback message path
    payload, message = await agent.handle(state, outcome, emotion={"anger": 0.8})

    print(payload.model_dump_json(indent=2))
    print("Customer message:", message)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())
