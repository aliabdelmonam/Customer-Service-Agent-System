"""Provider-agnostic customer-service triage agent.

The agent depends only on the project's GenerationClient interface. It does not
use LangChain or provider-specific SDKs. Structured output is requested through
`response_format`, which is supported directly by the Cohere provider shown in
this project.
"""

from __future__ import annotations

import json
from typing import Optional, Any,Literal

from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator

from src.llm_providers import GenerationClient, Message, ProviderError


# ---------------------------------------------------------------------------
# Taxonomy
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


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------
_ALL_FLOWS = tuple(FLOW_SUBFLOWS.keys())
_ALL_SUBFLOWS = tuple({sf for subs in FLOW_SUBFLOWS.values() for sf in subs})

class TriageResult(BaseModel):
    """Validated result returned by the triage node."""

    model_config = ConfigDict(extra="forbid")

    flow: Optional[Literal[_ALL_FLOWS]] = Field(
        default=None,
        description="Exact flow from the allowed taxonomy",
    )
    subflow: Optional[Literal[_ALL_SUBFLOWS]] = Field(
        default=None,
        description="Exact subflow belonging to the selected flow",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification confidence from 0.0 to 1.0.",
    )
    reasoning: str = Field(
        min_length=1,
        max_length=500,
        description="Short explanation for the selected intent; do not include private chain-of-thought.",
    )
    needs_clarification: bool = Field(
        description="True when the request cannot be reliably mapped to the taxonomy.",
    )

    @field_validator("flow")
    @classmethod
    def validate_flow(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in FLOW_SUBFLOWS:
            raise ValueError(f"Invalid flow: {value}")
        return value

    @field_validator("subflow")
    @classmethod
    def validate_subflow(cls, value: Optional[str], info) -> Optional[str]:
        if value is not None:
            flow = info.data.get("flow")
            if flow is None:
                raise ValueError("subflow cannot be set when flow is null")
            if value not in FLOW_SUBFLOWS.get(flow, []):
                raise ValueError(f"Invalid subflow '{value}' for flow '{flow}'")
        return value


def _taxonomy_text() -> str:
    lines: list[str] = []
    for flow, subflows in FLOW_SUBFLOWS.items():
        lines.append(f"\n{flow}:")
        for subflow in subflows:
            lines.append(f"  - {subflow}: {SUBFLOW_DESCRIPTIONS[subflow]}")
    return "\n".join(lines)


# def _structured_response_format() -> dict[str, Any]:
#     """JSON Schema for the triage result."""
#     return {
#         "type": "json_object",
#         "schema": {
#             "type": "object",
#             "additionalProperties": False,
#             "required": [
#                 "flow",
#                 "subflow",
#                 "confidence",
#                 "reasoning",
#                 "needs_clarification",
#             ],
#             "properties": {
#                 "flow": {
#                     "type": ["string", "null"],
#                     "enum": list(FLOW_SUBFLOWS.keys()) + [None],
#                     "description": "Exact taxonomy flow.",
#                 },
#                 "subflow": {
#                     "type": ["string", "null"],
#                     "enum": list(SUBFLOW_DESCRIPTIONS.keys()) + [None],
#                     "description": "Exact taxonomy subflow.",
#                 },
#                 "confidence": {
#                     "type": "number",
#                     "minimum": 0.0,
#                     "maximum": 1.0,
#                 },
#                 "reasoning": {
#                     "type": "string",
#                 },
#                 "needs_clarification": {
#                     "type": "boolean",
#                 },
#             },
#         },
#     }


TRIAGE_SYSTEM_PROMPT = f"""You are the triage classifier for a customer service system.

Your ONLY task is to classify the customer's latest message into the single
best (flow, subflow) pair from the fixed taxonomy below.

Do not answer the customer.
Do not perform actions.
Do not decide business policy.
Do not infer customer/account/order facts that were not provided.
Do not invent categories.

TAXONOMY:
{_taxonomy_text()}

CLASSIFICATION RULES:
1. Return exactly one valid flow/subflow pair when the request is classifiable.
2. The subflow MUST belong to the selected flow.
3. If the request is genuinely too vague, off-topic, or impossible to map,
   set flow=null, subflow=null, needs_clarification=true, and use a low confidence.
4. If two categories are plausible, choose the most specific category supported
   by the message and reduce confidence.
5. If there are multiple requests, classify the primary request expressed in the
   latest user message. Mention only the existence of the secondary request in
   the short reasoning; do not classify it as a second intent.
6. Conversation history is context only. Do not treat assistant statements as
   facts unless the customer confirms them.
7. Confidence must represent classification certainty, not whether the customer
   is likely to receive the requested outcome.
8. Reasoning must be one short factual sentence. Do not expose hidden chain of
   thought, internal deliberation, or policy reasoning.
9. The response MUST be valid JSON matching the supplied schema.
"""


class TriageAgent:
    """Triage node using the application's GenerationClient abstraction."""

    def __init__(
        self,
        llm: GenerationClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.response_format = _structured_response_format()

    @staticmethod
    def _messages(
        user_message: str,
        conversation_history: Optional[list[Message]],
    ) -> list[Message]:
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
        """Classify a customer message and return a validated TriageResult."""
        if not user_message or not user_message.strip():
            return TriageResult(
                flow=None,
                subflow=None,
                confidence=0.0,
                reasoning="The customer message is empty and cannot be classified.",
                needs_clarification=True,
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

        try:
            result = TriageResult.model_validate_json(response.text)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Provider returned invalid triage JSON: {response.text!r}"
            ) from exc

        # Final deterministic consistency checks. Never trust the model alone.
        if result.needs_clarification:
            if result.flow is not None or result.subflow is not None:
                raise ValueError(
                    "Invalid triage result: clarification must have null flow and subflow."
                )
            return result

        if result.flow is None or result.subflow is None:
            raise ValueError(
                "Invalid triage result: a non-clarification result requires flow and subflow."
            )

        if result.subflow not in FLOW_SUBFLOWS[result.flow]:
            raise ValueError(
                f"Invalid taxonomy pair: {result.flow!r} / {result.subflow!r}"
            )

        return result


__all__ = [
    "FLOW_SUBFLOWS",
    "SUBFLOW_DESCRIPTIONS",
    "TriageResult",
    "TriageAgent",
]

async def _example():
    import os
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from src.llm_providers import ProviderFactory, Provider

    llm = ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite")
    
    agent = TriageAgent(llm=llm)
    
    result = await agent.classify(
        user_message="I want to cancel my order, it hasn't shipped yet",
    )

    print(f"Flow: {result.flow}")
    print(f"Subflow: {result.subflow}")
    print(f"Confidence: {result.confidence}")
    print(f"Needs Clarification: {result.needs_clarification}")
    print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())