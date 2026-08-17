"""Temporary backend functions used by the resolution agent.

Each function is deliberately small and deterministic.  Replace its return
value with the real database, payment, shipping, FAQ, or notification logic
when the backend services are connected.  Keeping one function per resolution
action makes that replacement local and explicit.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable


FunctionResult = dict[str, Any]
ResolutionFunction = Callable[[dict[str, Any]], Awaitable[FunctionResult]]


def _placeholder(action: str, slots: dict[str, Any], **details: Any) -> FunctionResult:
    """Return a predictable temporary result without changing customer data."""
    return {"action": action, "status": "placeholder", "input": dict(slots), **details}


async def subscription_status(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("subscription_status", slots, subscription_status="Gold")


async def instructions(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("instructions", slots, instructions="No instructions configured.")


async def make_purchase(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("make_purchase", slots, purchase_created=True)


async def ask_the_oracle(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("ask_the_oracle", slots, answer=None)


async def shipping_status(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("shipping_status", slots, shipping_status="unknown")


async def update_account(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("update_account", slots, updated=False)


async def verify_identity(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("verify_identity", slots, verified=False)


async def select_answer(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("select_answer", slots, selected_answer=None)


async def pull_up_account(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("pull_up_account", slots, account=None)


async def update_order(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("update_order", slots, updated=False)


async def search_faq(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("search_faq", slots, results=[])


async def make_password(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("make_password", slots, password_created=False)


async def boots(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("boots", slots, results=[])


async def shirt(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("shirt", slots, results=[])


async def record_reason(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("record_reason", slots, recorded=False)


async def timing(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("timing", slots, results=[])


async def enter_details(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("enter_details", slots, saved=False)


async def membership(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("membership", slots, membership=None)


async def try_again(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("try_again", slots, retry_scheduled=False)


async def membership_privileges(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("membership_privileges", slots, privileges=[])


async def jacket(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("jacket", slots, results=[])


async def enter_detail(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("enter_detail", slots, saved=False)


async def jeans(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("jeans", slots, results=[])


async def send_link(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("send_link", slots, sent=False)


async def validate_purchase(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("validate_purchase", slots, valid=None)


async def end_conversation(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("end_conversation", slots, conversation_ended=True)


async def not_applicable(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("not_applicable", slots)


async def log_out_in(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("log_out_in", slots, session_reset=False)


async def policy(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("policy", slots, policy=None)


async def promo_code(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("promo_code", slots, valid=None)


async def offer_refund(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("offer_refund", slots, refund_offered=False)


async def pricing(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("pricing", slots, price=None)


async def notify_internal_team(slots: dict[str, Any]) -> FunctionResult:
    return _placeholder("notify_internal_team", slots, notified=False)


# Policy labels are accepted as aliases because existing flow data uses them.
FUNCTIONS: dict[str, ResolutionFunction] = {
    "Subscription Status": subscription_status,
    "Instructions": instructions,
    "Make Purchase": make_purchase,
    "Ask the Oracle": ask_the_oracle,
    "Shipping Status": shipping_status,
    "Update Account": update_account,
    "Verify Identity": verify_identity,
    "Select Answer": select_answer,
    "Pull up Account": pull_up_account,
    "Update Order": update_order,
    "Search FAQ": search_faq,
    "Make Password": make_password,
    "Boots": boots,
    "Shirt": shirt,
    "Record Reason": record_reason,
    "Timing": timing,
    "Enter Details": enter_details,
    "Membership": membership,
    "Try Again": try_again,
    "Membership Privileges": membership_privileges,
    "Jacket": jacket,
    "Enter Detail": enter_detail,
    "Jeans": jeans,
    "Send Link": send_link,
    "Validate Purchase": validate_purchase,
    "End Conversation": end_conversation,
    "N/A": not_applicable,
    "Log Out/In": log_out_in,
    "Policy": policy,
    "Promo Code": promo_code,
    "Offer Refund": offer_refund,
    "Pricing": pricing,
    "Notify Internal Team": notify_internal_team,
}
FUNCTIONS.update({function.__name__: function for function in tuple(FUNCTIONS.values())})


__all__ = ["FunctionResult", "ResolutionFunction", "FUNCTIONS", *[function.__name__ for function in FUNCTIONS.values()]]
