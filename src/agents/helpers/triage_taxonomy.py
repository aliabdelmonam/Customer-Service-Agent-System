
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