"""Triage agent — now merged with the chitchat gate into a single LLM call.

Previously this was two calls per turn (chitchat_gate.classify, then
triage.classify). Since both are narrow structured-output classifications
with no decision-making, they're combined into one schema and one call:
the model first states whether there's an actionable request, and only
fills in flow/subflow if there is. chitchat_gate.py's classification logic
is superseded by this file; its canned-response templates are reused here
unchanged.

Cost note: this call is still small (few output fields, low max_tokens) --
same model-swap-later story as before applies to the whole thing now,
not just the taxonomy part.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.llm_providers import GenerationClient, Message, ProviderError


# ---------------------------------------------------------------------------
# Taxonomy (unchanged)
# ---------------------------------------------------------------------------

FLOW_SUBFLOWS: dict[str, list[str]] = {
    "Product Defect": [
        "Initiate Refund", "Update Refund", "Refund Status",
        "Return Due to Stain", "Return Due to Color", "Return Due to Size",
    ],
    "Order Issue": [
        "Status Mystery Fee", "Status Delivery Time", "Status Payment Method",
        "Status Quantity", "Manage Upgrade", "Manage Downgrade",
        "Manage Create", "Manage Cancel",
    ],
    "Account Access": [
        "Recover Username", "Recover Password", "Reset Two-Factor Auth",
    ],
    "Troubleshoot Site": [
        "Invalid Credit Card", "Cart Not Updating", "Search Not Working",
        "Website Too Slow",
    ],
    "Manage Account": [
        "Status Service Added", "Status Service Removed", "Status Shipping Question",
        "Status Credit Missing", "Manage Change Address", "Manage Change Name",
        "Manage Change Phone", "Manage Payment Method",
    ],
    "Purchase Dispute": [
        "Bad Price Competitor", "Bad Price Yesterday", "Out-of-Stock General",
        "Out-of-Stock One Item", "Promo Code Invalid", "Promo Code Out of Date",
        "Mistimed Billing Already Returned", "Mistimed Billing Never Bought",
    ],
    "Shipping Issue": [
        "Shipping Status", "Manage Shipping", "Missing Item", "Shipping Cost",
    ],
    "Subscription Inquiry": [
        "Status Active", "Status Due Amount", "Status Due Date",
        "Manage Pay Bill", "Manage Extension", "Manage Dispute Bill",
    ],
    "Single-Item Query": [
        "Boots FAQ", "Shirt FAQ", "Jeans FAQ", "Jacket FAQ",
    ],
    "Storewide Query": [
        "Pricing FAQ", "Membership FAQ", "Timing FAQ", "Policy FAQ",
    ],
}

SUBFLOW_DESCRIPTIONS: dict[str, str] = {
    "Initiate Refund": "Customer wants to start a refund for a damaged or wrong item.",
    "Update Refund": "Customer wants to change details of an already-requested refund.",
    "Refund Status": "Customer is asking about the status of an existing refund.",
    "Return Due to Stain": "Customer wants to return an item because it arrived stained.",
    "Return Due to Color": "Customer wants to return an item because the color is wrong.",
    "Return Due to Size": "Customer wants to return an item because the size is wrong.",
    "Status Mystery Fee": "Customer is disputing an unexpected charge on an order.",
    "Status Delivery Time": "Customer is asking when an order will arrive.",
    "Status Payment Method": "Customer is asking which payment method was used or charged.",
    "Status Quantity": "Customer is asking about the quantity of items in an order.",
    "Manage Upgrade": "Customer wants to upgrade an item or plan on an order.",
    "Manage Downgrade": "Customer wants to downgrade an item or plan on an order.",
    "Manage Create": "Customer wants to place a new order.",
    "Manage Cancel": "Customer wants to cancel all or part of an existing order.",
    "Recover Username": "Customer forgot their username and wants to recover it.",
    "Recover Password": "Customer forgot their password and wants to reset it.",
    "Reset Two-Factor Auth": "Customer needs help resetting two-factor authentication.",
    "Invalid Credit Card": "Customer's credit card is being rejected at checkout.",
    "Cart Not Updating": "Customer's shopping cart is not reflecting changes correctly.",
    "Search Not Working": "Customer cannot search for products on the site.",
    "Website Too Slow": "Customer reports the website is slow or unresponsive.",
    "Status Service Added": "Customer is asking about a service added to their account.",
    "Status Service Removed": "Customer is asking about a service removed from their account.",
    "Status Shipping Question": "Customer has a general shipping question about their account.",
    "Status Credit Missing": "Customer expected a credit on their account that is not there.",
    "Manage Change Address": "Customer wants to update their shipping or billing address.",
    "Manage Change Name": "Customer wants to update the name on their account.",
    "Manage Change Phone": "Customer wants to update their phone number.",
    "Manage Payment Method": "Customer wants to update their saved payment method.",
    "Bad Price Competitor": "Customer says a competitor has a lower price and wants a match.",
    "Bad Price Yesterday": "Customer says the price was lower yesterday and wants that price honored.",
    "Out-of-Stock General": "Customer is asking about general product availability.",
    "Out-of-Stock One Item": "Customer is asking about availability of one specific item.",
    "Promo Code Invalid": "Customer's promo code is not working.",
    "Promo Code Out of Date": "Customer is trying to use an expired promo code.",
    "Mistimed Billing Already Returned": "Customer was billed for an item they already returned.",
    "Mistimed Billing Never Bought": "Customer was billed for an item they never purchased.",
    "Shipping Status": "Customer wants to know the shipping status of an order.",
    "Manage Shipping": "Customer wants to change the shipping method or address for an order.",
    "Missing Item": "Customer's order arrived with an item missing.",
    "Shipping Cost": "Customer has a question about shipping charges.",
    "Status Active": "Customer is asking whether their subscription is active.",
    "Status Due Amount": "Customer is asking how much they owe on their subscription or bill.",
    "Status Due Date": "Customer is asking when their next payment is due.",
    "Manage Pay Bill": "Customer wants to pay their bill now.",
    "Manage Extension": "Customer wants an extension on a payment deadline.",
    "Manage Dispute Bill": "Customer is disputing a subscription or bill charge.",
    "Boots FAQ": "General question about boots, such as sizing, material, or care.",
    "Shirt FAQ": "General question about shirts, such as sizing, material, or care.",
    "Jeans FAQ": "General question about jeans, such as sizing, material, or care.",
    "Jacket FAQ": "General question about jackets, such as sizing, material, or care.",
    "Pricing FAQ": "General question about store pricing policy.",
    "Membership FAQ": "General question about membership tiers or benefits.",
    "Timing FAQ": "General question about store hours or processing times.",
    "Policy FAQ": "General question about store policies, such as returns or shipping.",
}

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


def _taxonomy_text() -> str:
    lines: list[str] = []
    for flow, subflows in FLOW_SUBFLOWS.items():
        lines.append(f"\n{flow}:")
        for subflow in subflows:
            lines.append(f"  - {subflow}: {SUBFLOW_DESCRIPTIONS[subflow]}")
    return "\n".join(lines)


TRIAGE_SYSTEM_PROMPT = f"""You are the triage classifier for a customer service system.

STEP 1 — Determine whether the customer's latest message contains an actionable
customer-service request (anything about an order, account, refund, product,
shipping, billing, or similar).
- Set has_actionable_request=true if there is ANY such request, even if it's
  also preceded or followed by a greeting, thanks, or small talk
  (e.g. "hi, I want to cancel my order" -> has_actionable_request=true).
- Set has_actionable_request=false ONLY if the entire message is purely a
  greeting, farewell, thanks, or small talk with nothing else in it.
- Set chitchat_type to the greeting/farewell/thanks/small_talk framing present,
  or "none" if there isn't any.

STEP 2 — Only if has_actionable_request is true, classify the request into
exactly ONE (flow, subflow) pair from the fixed taxonomy below. If false,
leave flow and subflow null and skip this step.

TAXONOMY:
{_taxonomy_text()}

CLASSIFICATION RULES:
1. Choose the flow and subflow that best match the customer's request.
2. If the message is ambiguous between two categories, pick the more specific
   one and lower your confidence score accordingly.
3. If there's an actionable request but it's too vague to classify confidently,
   set needs_clarification=true and give your best guess with low confidence.
4. If there are multiple distinct requests, classify only the PRIMARY one.
   Mention the secondary request in "reasoning" only.
5. Never invent information about the customer's order, account, or identity.
6. Do not answer the customer, take actions, or decide policy -- classify only.
"""


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
