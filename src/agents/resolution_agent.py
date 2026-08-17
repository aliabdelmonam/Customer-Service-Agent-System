"""Resolution agent — deterministic step-walker over a subflow's step sequence.

Design principle (matches the rule-engine / triage split used elsewhere in this
project): the LLM never decides control flow. It is invoked narrowly, per step,
for exactly two jobs:

  1. Extracting a value from what the customer just said (slot steps where
     source == "customer_provided").
  2. Phrasing a question or response naturally (slot steps that need to ask,
     and response steps).

Everything else — which step we're on, whether a check passes, whether an
action requires human approval, whether we advance/branch/stop — is plain
code with an explicit pointer (`current_step_id`) into the sequence. This is
the same principle as the rule engine: the LLM must never improvise on checks
or actions.

SCHEMA NOTE: `sequence_table_refined_examples.json` / `rule_engine_examples.json`
were not available when this file was written — only their READMEs. The step
schema below (step_id / next_step_id / on_true_step_id / on_false_step_id /
requires_human_approval) is MY OWN convention to make branching and exits
concrete. Reconcile field names with the real refined JSON when you convert
the remaining 53 subflows — the shape (slot/check/action/response, per-slot
`source`) matches the README; the exact branching field names may not.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.llm_providers import GenerationClient, Message, ProviderError
from src.agents.resolution_functions import FUNCTIONS, ResolutionFunction


# ---------------------------------------------------------------------------
# Step schema
# ---------------------------------------------------------------------------

class SlotSource(str, Enum):
    CUSTOMER_PROVIDED = "customer_provided"
    SYSTEM_DERIVABLE = "system_derivable"


class StepType(str, Enum):
    SLOT = "slot"
    CHECK = "check"
    ACTION = "action"
    RESPONSE = "response"


class CheckOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    EXISTS = "exists"


class BaseStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    step_type: StepType
    instruction: str = Field(description="Original guideline text, for phrasing/audit.")


class SlotStep(BaseStep):
    step_type: Literal[StepType.SLOT] = StepType.SLOT
    slot_name: str
    source: SlotSource
    # For system_derivable slots: backend function that computes the value.
    derive_function: Optional[str] = None
    next_step_id: Optional[str] = None  # None => sequence ends after this step


class CheckStep(BaseStep):
    step_type: Literal[StepType.CHECK] = StepType.CHECK
    condition_slot: str
    operator: CheckOperator
    condition_value: Optional[Any] = None
    on_true_step_id: Optional[str] = None
    on_false_step_id: Optional[str] = None


class ActionStep(BaseStep):
    step_type: Literal[StepType.ACTION] = StepType.ACTION
    function_name: str
    requires_human_approval: bool = False  # e.g. money-moving actions
    next_step_id: Optional[str] = None


class ResponseStep(BaseStep):
    step_type: Literal[StepType.RESPONSE] = StepType.RESPONSE
    next_step_id: Optional[str] = None


Step = Union[SlotStep, CheckStep, ActionStep, ResponseStep]


class StepSequence(BaseModel):
    """A subflow's full step sequence, keyed for O(1) branching."""

    model_config = ConfigDict(extra="forbid")
    flow: str
    subflow: str
    start_step_id: str
    steps: dict[str, Step]

    def get(self, step_id: str) -> Step:
        try:
            return self.steps[step_id]
        except KeyError as exc:
            raise ValueError(f"Unknown step_id '{step_id}' in {self.subflow}") from exc


# ---------------------------------------------------------------------------
# Ticket state — must survive across turns, so it is plain serializable data.
# The orchestrator is responsible for persisting/loading this; this module
# does not assume a storage backend.
# ---------------------------------------------------------------------------

class TicketStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    AWAITING_CUSTOMER = "awaiting_customer"
    AWAITING_HUMAN_APPROVAL = "awaiting_human_approval"
    ESCALATED = "escalated"
    COMPLETED = "completed"


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: str
    correlation_id: str
    step_id: str
    step_type: StepType
    event: str
    detail: str = ""


class TicketState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correlation_id: str
    flow: str
    subflow: str
    current_step_id: str
    filled_slots: dict[str, Any] = Field(default_factory=dict)
    status: TicketStatus = TicketStatus.IN_PROGRESS
    audit_log: list[AuditEntry] = Field(default_factory=list)
    # Populated on terminal outcomes for the orchestrator to act on.
    pending_action: Optional[str] = None
    escalation_reason: Optional[str] = None

    def log(self, step_id: str, step_type: StepType, event: str, detail: str = "") -> None:
        self.audit_log.append(
            AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id=self.correlation_id,
                step_id=step_id,
                step_type=step_type,
                event=event,
                detail=detail,
            )
        )

    @classmethod
    def start(cls, flow: str, subflow: str, sequence: StepSequence) -> "TicketState":
        return cls(
            correlation_id=str(uuid.uuid4()),
            flow=flow,
            subflow=subflow,
            current_step_id=sequence.start_step_id,
        )


# ---------------------------------------------------------------------------
# Outcome — what process_turn hands back to the orchestrator each call.
# ---------------------------------------------------------------------------

class OutcomeType(str, Enum):
    ASK_CUSTOMER = "ask_customer"          # need a slot; message is the question
    INFORM_CUSTOMER = "inform_customer"    # a response step fired; message is FYI text
    HANDOFF_APPROVAL = "handoff_approval"   # hit a money-moving action; stop for human
    HANDOFF_ESCALATION = "handoff_escalation"  # backend-function failure / capability gap
    COMPLETED = "completed"                # sequence finished


class ResolutionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome_type: OutcomeType
    message: Optional[str] = None
    state: TicketState
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend functions — deterministic operations. The LLM never decides what
# these return; it may reach a step that calls a function, but the outcome is
# code, not a model generation.
# ---------------------------------------------------------------------------

class FunctionExecutionError(Exception):
    """Raised when a backend function fails and the ticket must escalate."""

    def __init__(self, function_name: str, original: Exception):
        self.function_name = function_name
        self.original = original
        super().__init__(f"[{function_name}] failed: {original}")


class BackendFunctionRegistry:
    """Executes the concrete functions defined in ``resolution_functions``."""

    def __init__(self, functions: Optional[dict[str, ResolutionFunction]] = None) -> None:
        self._functions = dict(FUNCTIONS if functions is None else functions)

    async def call(self, function_name: str, filled_slots: dict[str, Any]) -> Any:
        function = self._functions.get(function_name)
        if function is None:
            raise FunctionExecutionError(
                function_name, RuntimeError(f"No backend function named '{function_name}'")
            )
        try:
            return await function(dict(filled_slots))
        except Exception as exc:  # noqa: BLE001 -- always becomes an escalation
            raise FunctionExecutionError(function_name, exc) from exc


# ---------------------------------------------------------------------------
# Audit logger — pluggable so it can be swapped for the project's existing
# correlation_id-based logging.
# ---------------------------------------------------------------------------

class AuditLogger(Protocol):
    def emit(self, entry: AuditEntry) -> None: ...


class NullAuditLogger:
    """Default no-op logger; TicketState.audit_log already captures history
    in-band, this is only for out-of-band sinks (e.g. shipping to Axiom)."""

    def emit(self, entry: AuditEntry) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Narrow LLM calls
# ---------------------------------------------------------------------------

class SlotExtraction(BaseModel):
    """Structured output for a single, scoped extraction call. The agent asks
    for exactly one slot at a time — never "extract everything you can"."""

    model_config = ConfigDict(extra="forbid")
    found: bool = Field(description="True only if the customer's message actually states this value.")
    value: Optional[str] = Field(default=None, description="The extracted value, if found.")
    confidence: float = Field(ge=0.0, le=1.0)


SLOT_EXTRACTION_SYSTEM_PROMPT = """You extract exactly one piece of information from a
customer's message. You do not answer the customer, take actions, or infer values that
were not stated. If the customer did not state this value, set found=false and value=null.
Do not guess or default. Respond only with the requested JSON schema."""


RESPONSE_PHRASING_SYSTEM_PROMPT = """You phrase a single message to a customer service
customer, based on an internal instruction and known facts. Be concise and natural. Do not
invent facts beyond what is given. Do not mention internal step names, policies, or system
details. Output only the message text, nothing else."""


# ---------------------------------------------------------------------------
# Resolution agent
# ---------------------------------------------------------------------------

class ResolutionAgent:
    """Deterministic step-walker. One instance can serve any ticket; all
    per-ticket state lives in the `TicketState` passed in and returned.

    NOTE — explicitly out of scope here, per design: emotion scoring and
    safety/guardrail checks run on every incoming message regardless of
    which step the ticket is on. They are the orchestrator's job, upstream
    of `process_turn`, not a step in the sequence.
    """

    def __init__(
        self,
        llm: GenerationClient,
        functions: BackendFunctionRegistry,
        *,
        audit_logger: Optional[AuditLogger] = None,
        extraction_temperature: float = 0.0,
        phrasing_temperature: float = 0.3,
    ) -> None:
        self.llm = llm
        self.functions = functions
        self.audit_logger = audit_logger or NullAuditLogger()
        self.extraction_temperature = extraction_temperature
        self.phrasing_temperature = phrasing_temperature

    # -- public entry point --------------------------------------------------

    async def process_turn(
        self,
        state: TicketState,
        sequence: StepSequence,
        customer_message: Optional[str] = None,
    ) -> ResolutionOutcome:
        """Advance the ticket as far as it can go on this turn.

        Runs a bounded loop: keeps walking steps (system-derivable slots,
        checks, non-approval actions, response steps) without waiting on the
        customer, and stops the instant it needs customer input, hits an
        approval gate, hits a backend-function failure, or finishes the sequence.
        """
        if state.status in (TicketStatus.COMPLETED, TicketStatus.ESCALATED):
            return ResolutionOutcome(outcome_type=OutcomeType.COMPLETED, state=state)

        state.status = TicketStatus.IN_PROGRESS
        max_steps_per_turn = 8  # guard against a malformed sequence looping forever
        pending_customer_message = customer_message

        for _ in range(max_steps_per_turn):
            step = sequence.get(state.current_step_id)

            if isinstance(step, SlotStep):
                outcome = await self._handle_slot_step(state, step, pending_customer_message)
                pending_customer_message = None  # only consumed once
                if outcome is not None:
                    return outcome
                if state.status == TicketStatus.COMPLETED:
                    return ResolutionOutcome(outcome_type=OutcomeType.COMPLETED, state=state)
                continue

            if isinstance(step, CheckStep):
                self._handle_check_step(state, step)
                continue

            if isinstance(step, ActionStep):
                outcome = await self._handle_action_step(state, step)
                if outcome is not None:
                    return outcome
                if state.status == TicketStatus.COMPLETED:
                    return ResolutionOutcome(outcome_type=OutcomeType.COMPLETED, state=state)
                continue

            if isinstance(step, ResponseStep):
                return await self._handle_response_step(state, step)

            raise ValueError(f"Unhandled step type at {state.current_step_id!r}")

        state.status = TicketStatus.ESCALATED
        state.escalation_reason = "Step sequence exceeded max steps per turn; likely a malformed loop."
        state.log(state.current_step_id, sequence.get(state.current_step_id).step_type, "escalated", state.escalation_reason)
        return ResolutionOutcome(
            outcome_type=OutcomeType.HANDOFF_ESCALATION,
            state=state,
            reason=state.escalation_reason,
        )

    # -- step handlers ---------------------------------------------------------

    async def _handle_slot_step(
        self,
        state: TicketState,
        step: SlotStep,
        customer_message: Optional[str],
    ) -> Optional[ResolutionOutcome]:
        """Returns an outcome if we must stop here; None if we can advance."""

        # Already filled (e.g. resumed turn) — just advance.
        if step.slot_name in state.filled_slots:
            state.log(step.step_id, step.step_type, "slot_already_filled")
            self._advance(state, step.next_step_id)
            return None

        if step.source == SlotSource.SYSTEM_DERIVABLE:
            if not step.derive_function:
                raise ValueError(f"Slot step {step.step_id!r} is system_derivable but has no derive_function")
            try:
                value = await self.functions.call(step.derive_function, state.filled_slots)
            except FunctionExecutionError as exc:
                return self._escalate(state, step, f"Failed to derive slot '{step.slot_name}': {exc}")
            state.filled_slots[step.slot_name] = value
            state.log(step.step_id, step.step_type, "slot_derived", f"{step.slot_name}={value!r}")
            self._advance(state, step.next_step_id)
            return None

        # customer_provided: try to extract from the message just given.
        if customer_message:
            extraction = await self._extract_slot(step, customer_message)
            if extraction.found and extraction.value is not None:
                state.filled_slots[step.slot_name] = extraction.value
                state.log(
                    step.step_id, step.step_type, "slot_extracted",
                    f"{step.slot_name}={extraction.value!r} (confidence={extraction.confidence})",
                )
                self._advance(state, step.next_step_id)
                return None

        # Nothing to extract (or extraction failed) — ask the customer.
        question = await self._phrase(
            instruction=step.instruction,
            known_facts=state.filled_slots,
        )
        state.status = TicketStatus.AWAITING_CUSTOMER
        state.log(step.step_id, step.step_type, "asked_customer", question)
        return ResolutionOutcome(
            outcome_type=OutcomeType.ASK_CUSTOMER,
            message=question,
            state=state,
        )

    def _handle_check_step(self, state: TicketState, step: CheckStep) -> None:
        """Pure code — the LLM is never involved in evaluating a check."""
        actual = state.filled_slots.get(step.condition_slot)
        passed = self._evaluate_condition(actual, step.operator, step.condition_value)
        next_id = step.on_true_step_id if passed else step.on_false_step_id
        state.log(
            step.step_id, step.step_type, "check_evaluated",
            f"{step.condition_slot}={actual!r} {step.operator.value} {step.condition_value!r} -> {passed}",
        )
        if next_id is None:
            raise ValueError(
                f"Check step {step.step_id!r} has no {'on_true' if passed else 'on_false'} target"
            )
        state.current_step_id = next_id

    @staticmethod
    def _evaluate_condition(actual: Any, operator: CheckOperator, expected: Any) -> bool:
        if operator == CheckOperator.EQUALS:
            return actual == expected
        if operator == CheckOperator.NOT_EQUALS:
            return actual != expected
        if operator == CheckOperator.IS_TRUE:
            return bool(actual) is True
        if operator == CheckOperator.IS_FALSE:
            return bool(actual) is False
        if operator == CheckOperator.EXISTS:
            return actual is not None
        raise ValueError(f"Unknown operator {operator!r}")

    async def _handle_action_step(
        self, state: TicketState, step: ActionStep
    ) -> Optional[ResolutionOutcome]:
        if step.requires_human_approval:
            state.status = TicketStatus.AWAITING_HUMAN_APPROVAL
            state.pending_action = step.function_name
            state.log(step.step_id, step.step_type, "awaiting_human_approval", step.function_name)
            return ResolutionOutcome(
                outcome_type=OutcomeType.HANDOFF_APPROVAL,
                state=state,
                reason=f"Function '{step.function_name}' requires human approval before executing.",
            )

        try:
            result = await self.functions.call(step.function_name, state.filled_slots)
        except FunctionExecutionError as exc:
            return self._escalate(state, step, f"Function '{step.function_name}' failed: {exc}")

        state.log(step.step_id, step.step_type, "function_executed", f"{step.function_name} -> {result!r}")
        self._advance(state, step.next_step_id)
        return None

    async def _handle_response_step(self, state: TicketState, step: ResponseStep) -> ResolutionOutcome:
        message = await self._phrase(instruction=step.instruction, known_facts=state.filled_slots)
        state.log(step.step_id, step.step_type, "response_sent", message)

        if step.next_step_id is None:
            state.status = TicketStatus.COMPLETED
            state.log(step.step_id, step.step_type, "sequence_completed")
            return ResolutionOutcome(outcome_type=OutcomeType.COMPLETED, message=message, state=state)

        self._advance(state, step.next_step_id)
        return ResolutionOutcome(outcome_type=OutcomeType.INFORM_CUSTOMER, message=message, state=state)

    # -- helpers -----------------------------------------------------------

    def _advance(self, state: TicketState, next_step_id: Optional[str]) -> None:
        if next_step_id is None:
            state.status = TicketStatus.COMPLETED
            return
        state.current_step_id = next_step_id

    def _escalate(self, state: TicketState, step: Step, reason: str) -> ResolutionOutcome:
        state.status = TicketStatus.ESCALATED
        state.escalation_reason = reason
        state.log(step.step_id, step.step_type, "escalated", reason)
        return ResolutionOutcome(outcome_type=OutcomeType.HANDOFF_ESCALATION, state=state, reason=reason)

    async def _extract_slot(self, step: SlotStep, customer_message: str) -> SlotExtraction:
        messages = [
            Message(
                role="system",
                content=(
                    f"{SLOT_EXTRACTION_SYSTEM_PROMPT}\n\n"
                    f"Slot to extract: {step.slot_name}\n"
                    f"Context / instruction: {step.instruction}"
                ),
            ),
            Message(role="user", content=customer_message),
        ]
        try:
            response = await self.llm.generate(
                messages=messages,
                temperature=self.extraction_temperature,
                max_tokens=128,
                output_schema=SlotExtraction,
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Slot extraction generation failed") from exc

        try:
            return SlotExtraction.model_validate_json(response.text)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ValueError(f"Provider returned invalid extraction JSON: {response.text!r}") from exc

    async def _phrase(self, instruction: str, known_facts: dict[str, Any]) -> str:
        facts_text = ", ".join(f"{k}={v!r}" for k, v in known_facts.items()) or "(none yet)"
        messages = [
            Message(role="system", content=RESPONSE_PHRASING_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Internal instruction: {instruction}\nKnown facts: {facts_text}",
            ),
        ]
        try:
            response = await self.llm.generate(
                messages=messages,
                temperature=self.phrasing_temperature,
                max_tokens=200,
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Response phrasing generation failed") from exc
        return response.text.strip()


__all__ = [
    "SlotSource",
    "StepType",
    "CheckOperator",
    "SlotStep",
    "CheckStep",
    "ActionStep",
    "ResponseStep",
    "Step",
    "StepSequence",
    "TicketStatus",
    "AuditEntry",
    "TicketState",
    "OutcomeType",
    "ResolutionOutcome",
    "FunctionExecutionError",
    "BackendFunctionRegistry",
    "AuditLogger",
    "NullAuditLogger",
    "ResolutionAgent",
]


# ---------------------------------------------------------------------------
# Example usage — Manage Cancel, roughly matching the README's
# "item shipped -> cannot cancel" rule.
# ---------------------------------------------------------------------------

async def _example():
    from src.llm_providers import ProviderFactory, Provider

    async def lookup_shipping_status(slots: dict[str, Any]) -> str:
        return "not_shipped"  # stub

    async def cancel_order(slots: dict[str, Any]) -> str:
        return f"order {slots.get('order_id')} cancelled"

    functions = BackendFunctionRegistry({
        "lookup_shipping_status": lookup_shipping_status,
        "cancel_order": cancel_order,
    })

    sequence = StepSequence(
        flow="Order Issue",
        subflow="Manage Cancel",
        start_step_id="ask_order_id",
        steps={
            "ask_order_id": SlotStep(
                step_id="ask_order_id",
                instruction="Ask the customer for their order ID.",
                slot_name="order_id",
                source=SlotSource.CUSTOMER_PROVIDED,
                next_step_id="derive_shipping_status",
            ),
            "derive_shipping_status": SlotStep(
                step_id="derive_shipping_status",
                instruction="Look up whether the order has shipped.",
                slot_name="shipping_status",
                source=SlotSource.SYSTEM_DERIVABLE,
                derive_function="lookup_shipping_status",
                next_step_id="check_shipped",
            ),
            "check_shipped": CheckStep(
                step_id="check_shipped",
                instruction="If shipped, cannot cancel.",
                condition_slot="shipping_status",
                operator=CheckOperator.EQUALS,
                condition_value="not_shipped",
                on_true_step_id="cancel_action",
                on_false_step_id="cannot_cancel_response",
            ),
            "cancel_action": ActionStep(
                step_id="cancel_action",
                instruction="Cancel the order.",
                function_name="cancel_order",
                requires_human_approval=False,
                next_step_id="cancelled_response",
            ),
            "cancelled_response": ResponseStep(
                step_id="cancelled_response",
                instruction="Tell the customer their order has been cancelled.",
                next_step_id=None,
            ),
            "cannot_cancel_response": ResponseStep(
                step_id="cannot_cancel_response",
                instruction="Tell the customer the order has already shipped and cannot be cancelled.",
                next_step_id=None,
            ),
        },
    )

    llm = ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite")
    agent = ResolutionAgent(llm=llm, functions=functions)

    state = TicketState.start(flow="Order Issue", subflow="Manage Cancel", sequence=sequence)

    outcome = await agent.process_turn(state, sequence)
    print(outcome.outcome_type, outcome.message)

    outcome = await agent.process_turn(
        outcome.state, sequence, customer_message="It's order #4471"
    )
    print(outcome.outcome_type, outcome.message)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())
