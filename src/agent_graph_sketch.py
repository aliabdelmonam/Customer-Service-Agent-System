"""
Customer Service Agent — LangGraph state machine sketch.

This is a SKELETON, not production code: node bodies are stubs showing
what each step should call and return. Fill in the LLM prompts, tool
implementations, and policy engine rules to match your real system.

Design decisions baked in (from spec discussion):
- user_id NEVER comes from LLM-parsed text — always injected from the
  authenticated session before the graph runs.
- Policy lookups are deterministic function calls (hierarchical dict/DB
  lookup), never RAG/ANN. RAG is reserved for the FAQ node only.
- Money/state-changing actions always route to human approval for now.
- Emotion is re-scored every turn, not just at triage.
- Open tickets are checked before a new ticket is created (dedup).
- Every node writes to an audit log keyed by correlation_id / ticket_id.
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
import uuid
import time


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------

class Slot(TypedDict):
    name: str
    value: Optional[str]
    required: bool
    validated: bool


class EmotionScore(TypedDict):
    anger: float
    frustration: float
    urgency: float
    trend: Literal["improving", "stable", "declining"]


class ToolCallLog(TypedDict):
    tool_name: str
    args: dict
    result: dict
    timestamp: float


class AgentState(TypedDict):
    # Identity — injected by the API layer, never by the LLM
    correlation_id: str
    user_id: str

    # Conversation
    messages: list  # running chat history
    last_user_message: str

    # Guardrails
    guardrail_flag: Optional[str]  # None | "injection" | "scope_violation"

    # Ticket / dedup
    ticket_id: Optional[str]
    open_tickets: list  # result of get_open_tickets()
    is_duplicate: bool

    # Triage
    intent: Optional[str]
    category: Optional[str]
    emotion: EmotionScore

    # Slot filling
    required_slots: list[Slot]
    slots_complete: bool

    # Policy + resolution
    policy_result: Optional[dict]
    action_plan: Optional[dict]

    # Gates
    needs_human_approval: bool   # money / state-changing action
    needs_escalation: bool       # stuck / emotion / tool failure
    escalation_payload: Optional[dict]

    # Output
    response: str
    tool_call_log: list[ToolCallLog]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------

def guardrail_check(state: AgentState) -> AgentState:
    """
    Runs on every turn, before anything else.
    Checks: prompt injection patterns, attempts to reference another
    user's data (order ids / emails not belonging to user_id), attempts
    to override system instructions.
    This is a classifier / rule check, NOT something the resolution
    LLM is trusted to self-police.
    """
    flag = run_injection_and_scope_classifier(
        text=state["last_user_message"],
        user_id=state["user_id"],
    )
    state["guardrail_flag"] = flag
    log_audit_event(state, node="guardrail_check", data={"flag": flag})
    return state


def route_after_guardrail(state: AgentState) -> str:
    if state["guardrail_flag"] is not None:
        return "blocked_response"
    return "emotion_scorer"


def blocked_response(state: AgentState) -> AgentState:
    """Terminal node for a guardrail violation — no tool calls, no LLM
    reasoning about the blocked content, just a safe fixed response."""
    state["response"] = (
        "I can only help with your own account and orders, and I'm not "
        "able to process that request. Let me know if there's something "
        "else I can help with."
    )
    log_audit_event(state, node="blocked_response", data={})
    return state


def emotion_scorer(state: AgentState) -> AgentState:
    """Re-scored every turn, not just once at triage — lets the graph
    catch a customer escalating mid-conversation even if intent/category
    hasn't changed."""
    state["emotion"] = score_emotion(
        text=state["last_user_message"],
        history=state["messages"],
        previous=state.get("emotion"),
    )
    log_audit_event(state, node="emotion_scorer", data=state["emotion"])
    return state


def ticket_lookup(state: AgentState) -> AgentState:
    """Dedup check: is there already an open ticket for this kind of
    request? Exact match on user_id + category/order_id, not fuzzy."""
    open_tickets = get_open_tickets(user_id=state["user_id"])
    state["open_tickets"] = open_tickets
    log_tool_call(state, "get_open_tickets", {"user_id": state["user_id"]}, open_tickets)
    return state


def triage(state: AgentState) -> AgentState:
    """LLM call: classify intent + category from the message + any open
    ticket context. Does NOT decide policy — just routes."""
    result = triage_llm_call(
        message=state["last_user_message"],
        open_tickets=state["open_tickets"],
    )
    state["intent"] = result["intent"]
    state["category"] = result["category"]
    state["is_duplicate"] = result.get("matches_open_ticket", False)

    if state["is_duplicate"]:
        matched = result["matched_ticket_id"]
        state["ticket_id"] = matched
    else:
        state["ticket_id"] = str(uuid.uuid4())

    state["required_slots"] = get_required_slots_for_intent(state["intent"])
    log_audit_event(state, node="triage", data=result)
    return state


def route_after_triage(state: AgentState) -> str:
    if state["is_duplicate"]:
        return "duplicate_ticket_response"
    return "slot_filling"


def duplicate_ticket_response(state: AgentState) -> AgentState:
    state["response"] = (
        f"Looks like you already have an open request for this "
        f"(ticket {state['ticket_id']}). Here's the latest status..."
    )
    # fetch and append status of matched ticket here
    log_audit_event(state, node="duplicate_ticket_response", data={})
    return state


def slot_filling(state: AgentState) -> AgentState:
    """Determine which required slots are still missing. If any are
    missing, generate a clarifying question and stop here for this turn
    (graph re-enters on next user message). If complete, proceed."""
    filled, missing = check_slots(
        required=state["required_slots"],
        message=state["last_user_message"],
        history=state["messages"],
    )
    state["required_slots"] = filled
    state["slots_complete"] = len(missing) == 0

    if not state["slots_complete"]:
        state["response"] = generate_clarifying_question(missing)

    log_audit_event(state, node="slot_filling", data={"missing": missing})
    return state


def route_after_slots(state: AgentState) -> str:
    if not state["slots_complete"]:
        return END  # wait for next user turn with the missing info
    return "policy_lookup"


def policy_lookup(state: AgentState) -> AgentState:
    """
    Deterministic, hierarchical lookup — NOT RAG/ANN.
    e.g. policies[category][intent] -> ordered precondition checks,
    same shape as rule_engine_examples.json from the ABCD-derived set.
    """
    result = evaluate_policy(
        category=state["category"],
        intent=state["intent"],
        slots={s["name"]: s["value"] for s in state["required_slots"]},
        user_id=state["user_id"],
    )
    state["policy_result"] = result
    log_tool_call(state, "evaluate_policy", {"intent": state["intent"]}, result)
    return state


def route_after_policy(state: AgentState) -> str:
    result = state["policy_result"]
    if result.get("requires_money_action"):
        return "human_approval_gate"
    if result.get("requires_capability_we_dont_have"):  # e.g. images, complex judgment
        return "escalation_builder"
    if state["emotion"]["anger"] > 0.8 or state["emotion"]["trend"] == "declining":
        return "escalation_builder"
    return "execute_action"


def human_approval_gate(state: AgentState) -> AgentState:
    """
    Money / state-changing actions always land here for now (per current
    scope decision). This is NOT the same as escalation — it's a fast,
    async human click-to-approve queue, with the AI's drafted action and
    reasoning attached, not a full ticket handoff.
    """
    state["needs_human_approval"] = True
    queue_for_approval(
        ticket_id=state["ticket_id"],
        user_id=state["user_id"],
        proposed_action=state["policy_result"],
    )
    state["response"] = (
        "I've prepared this for approval and someone will confirm it "
        "shortly — you'll get an update as soon as it's processed."
    )
    log_audit_event(state, node="human_approval_gate", data=state["policy_result"])
    return state


def execute_action(state: AgentState) -> AgentState:
    """Non-money, policy-approved action — call the actual system tool."""
    result = call_system_tool(
        action=state["policy_result"]["action"],
        user_id=state["user_id"],   # session-scoped, never from message text
        slots={s["name"]: s["value"] for s in state["required_slots"]},
    )
    log_tool_call(state, state["policy_result"]["action"], {}, result)

    if not result.get("success"):
        state["needs_escalation"] = True
        return state  # falls through to escalation_builder via conditional edge

    state["action_plan"] = result
    state["response"] = generate_response_from_result(result)
    return state


def route_after_execute(state: AgentState) -> str:
    if state.get("needs_escalation"):
        return "escalation_builder"
    return "close_ticket"


def escalation_builder(state: AgentState) -> AgentState:
    """
    Structured payload, not a free-text summary — per the enhancement
    decision. Includes what the policy engine returned and what was
    already tried, so the human isn't starting from zero.
    """
    state["escalation_payload"] = {
        "ticket_id": state["ticket_id"],
        "user_id": state["user_id"],
        "category": state["category"],
        "intent": state["intent"],
        "emotion": state["emotion"],
        "policy_result": state["policy_result"],
        "tool_call_log": state["tool_call_log"],
        "reason": determine_escalation_reason(state),
        "suggested_next_step": suggest_next_step(state),
    }
    state["needs_escalation"] = True
    push_to_human_queue(state["escalation_payload"])
    state["response"] = (
        "I've escalated this to our support team with full context — "
        "they'll follow up with you shortly."
    )
    log_audit_event(state, node="escalation_builder", data=state["escalation_payload"])
    return state


def close_ticket(state: AgentState) -> AgentState:
    """Terminal success node. Logs outcome for the eval set."""
    log_ticket_outcome(
        ticket_id=state["ticket_id"],
        resolved_by="ai",
        category=state["category"],
    )
    return state


# ---------------------------------------------------------------------------
# GRAPH WIRING
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail_check", guardrail_check)
    graph.add_node("blocked_response", blocked_response)
    graph.add_node("emotion_scorer", emotion_scorer)
    graph.add_node("ticket_lookup", ticket_lookup)
    graph.add_node("triage", triage)
    graph.add_node("duplicate_ticket_response", duplicate_ticket_response)
    graph.add_node("slot_filling", slot_filling)
    graph.add_node("policy_lookup", policy_lookup)
    graph.add_node("human_approval_gate", human_approval_gate)
    graph.add_node("execute_action", execute_action)
    graph.add_node("escalation_builder", escalation_builder)
    graph.add_node("close_ticket", close_ticket)

    graph.set_entry_point("guardrail_check")

    graph.add_conditional_edges(
        "guardrail_check", route_after_guardrail,
        {"blocked_response": "blocked_response", "emotion_scorer": "emotion_scorer"},
    )
    graph.add_edge("blocked_response", END)

    graph.add_edge("emotion_scorer", "ticket_lookup")
    graph.add_edge("ticket_lookup", "triage")

    graph.add_conditional_edges(
        "triage", route_after_triage,
        {"duplicate_ticket_response": "duplicate_ticket_response", "slot_filling": "slot_filling"},
    )
    graph.add_edge("duplicate_ticket_response", END)

    graph.add_conditional_edges(
        "slot_filling", route_after_slots,
        {END: END, "policy_lookup": "policy_lookup"},
    )

    graph.add_conditional_edges(
        "policy_lookup", route_after_policy,
        {
            "human_approval_gate": "human_approval_gate",
            "escalation_builder": "escalation_builder",
            "execute_action": "execute_action",
        },
    )

    graph.add_edge("human_approval_gate", END)

    graph.add_conditional_edges(
        "execute_action", route_after_execute,
        {"escalation_builder": "escalation_builder", "close_ticket": "close_ticket"},
    )

    graph.add_edge("escalation_builder", END)
    graph.add_edge("close_ticket", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# STUBS — implement these against your real system
# ---------------------------------------------------------------------------

def run_injection_and_scope_classifier(text: str, user_id: str) -> Optional[str]: ...
def score_emotion(text: str, history: list, previous: Optional[EmotionScore]) -> EmotionScore: ...
def get_open_tickets(user_id: str) -> list: ...
def triage_llm_call(message: str, open_tickets: list) -> dict: ...
def get_required_slots_for_intent(intent: str) -> list[Slot]: ...
def check_slots(required: list[Slot], message: str, history: list) -> tuple[list[Slot], list[str]]: ...
def generate_clarifying_question(missing: list[str]) -> str: ...
def evaluate_policy(category: str, intent: str, slots: dict, user_id: str) -> dict: ...
def call_system_tool(action: str, user_id: str, slots: dict) -> dict: ...
def generate_response_from_result(result: dict) -> str: ...
def determine_escalation_reason(state: AgentState) -> str: ...
def suggest_next_step(state: AgentState) -> str: ...
def queue_for_approval(ticket_id: str, user_id: str, proposed_action: dict) -> None: ...
def push_to_human_queue(payload: dict) -> None: ...
def log_ticket_outcome(ticket_id: str, resolved_by: str, category: str) -> None: ...


def log_audit_event(state: AgentState, node: str, data: dict) -> None:
    """Every node writes here. Backing store: same OTel/Axiom pattern
    already used on the mental health RAG project — correlation_id ties
    every node's output to one ticket for later eval/debugging."""
    print({
        "correlation_id": state["correlation_id"],
        "ticket_id": state.get("ticket_id"),
        "node": node,
        "timestamp": time.time(),
        "data": data,
    })


def log_tool_call(state: AgentState, tool_name: str, args: dict, result: dict) -> None:
    entry: ToolCallLog = {
        "tool_name": tool_name,
        "args": args,
        "result": result,
        "timestamp": time.time(),
    }
    state.setdefault("tool_call_log", []).append(entry)
    log_audit_event(state, node=f"tool:{tool_name}", data=entry)
