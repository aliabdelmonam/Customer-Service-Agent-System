# Policy Dataset for AI Customer Service Agent

Source: derived from the Action-Based Conversations Dataset (ABCD, Chen et al. 2021,
asappresearch/abcd on GitHub). ABCD is the only public dataset built specifically around
company-policy-constrained agent actions (as opposed to just conversational phrasing).

## Files

- **policies_structured.json** — all 55 real policy subflows (e.g. "Order Issue -> Manage Cancel",
  "Product Defect -> Initiate Refund") converted into a clean, consistent schema:
  `{flow, subflow, scenario, steps: [{step_type, action_name, instruction, details}]}`.
  This is the raw policy logic an agent originally followed, in structured form.

- **rule_engine_examples.json** — 3 fully worked examples (order cancellation, product-defect
  refund, mystery-fee dispute) rewritten as explicit `precondition -> action` rules —
  the exact "item shipped -> cannot cancel" pattern you asked for. Use this as the template
  to convert the rest of policies_structured.json, and to encode your own real business rules
  in the same shape.

- **raw_abcd_guidelines.json** — original ABCD guideline documents (unmodified).
- **raw_abcd_kb.json** — action sequences per intent (55 intents -> ordered action lists).
- **raw_abcd_sample_conversations.json** — 3 sample human-agent dialogues showing the
  policy being applied in a live conversation (useful for eval / conversational tone).

## How to use this

1. Treat `rule_engine_examples.json`'s schema as your policy engine's format — this is what
   your agent should query (deterministically) before taking any money-related action, not
   something you fine-tune into the LLM's weights.
2. Walk through `policies_structured.json` and convert the ones relevant to your product
   (order cancel, refund, shipping, subscription) into the same precondition/action shape.
3. Replace ABCD's fictional membership tiers / dollar amounts / day-windows with your actual
   business policy once you have it.
4. Use `raw_abcd_sample_conversations.json` (and the full dataset, linked below) as eval
   conversations to test whether your agent's decisions match the policy given the same inputs.

## Full dataset (not included here — 10,042 conversations)
Full ABCD dataset: https://github.com/asappresearch/abcd (data/abcd_v1.1.json.gz)
Paper: https://arxiv.org/pdf/2104.00783
