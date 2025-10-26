"""Review assistant module orchestrating the issues/edits flow."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, Optional

Action = Literal[
    "generate_edits",
    "find_competitors",
    "stop",
    "clarify",
    "answer_question",
]

_ALLOWED_ACTIONS = {
    "generate_edits",
    "find_competitors",
    "stop",
    "clarify",
    "answer_question",
}

_TOOL_METADATA = [
    {
        "name": "answer_question",
        "description": (
            "Synthesize a direct answer for questions about the rules, review summary, "
            "client SKU details, or competitor data using available context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "issues_summary": {"type": "string"},
                "rules": {"type": "object"},
                "client_sku": {"type": "object"},
                "competitor_sku": {"type": "object"},
            },
            "required": ["question"],
        },
    }
]

_LLM_SYSTEM_PROMPT = (
    "You orchestrate a review workflow for Amazon PDP content. "
    "Choose one action based on the latest issues summary and the user's reply. "
    "Actions: generate_edits (draft compliant edits now), find_competitors (surface similar competitor products "
    "automatically), stop (end the workflow), clarify (ask for more info), answer_question (respond directly to a "
    "question about the rules or provided product data using the answer_question tool). Return JSON with a single "
    "key 'action' using one of the allowed values."
)


class _LLMResponseError(RuntimeError):
    """Raised when the LLM result cannot be parsed."""


def _heuristic_classification(summary: str, user_input: str) -> Action:
    text = (user_input or "").strip()
    if not text:
        return "clarify"

    normalized = re.sub(r"\s+", " ", text.lower())

    find_competitor_triggers = [
        "find competitor",
        "find competitors",
        "show similar product",
        "show similar products",
        "recommend competitor",
        "recommend competitors",
        "suggest competitor",
        "suggest competitors",
        "show competitors",
        "show competitor products",
        "similar competitors",
    ]
    for trigger in find_competitor_triggers:
        if trigger in normalized:
            return "find_competitors"

    competitor_triggers = [
        "different competitor",
        "another competitor",
        "switch competitor",
        "change competitor",
        "pick a competitor",
        "choose competitor",
        "new competitor",
    ]
    for trigger in competitor_triggers:
        if trigger in normalized:
            return "find_competitors"
    if "competitor" in normalized and any(
        word in normalized for word in ["change", "switch", "different", "another"]
    ):
        return "find_competitors"

    stop_triggers = [
        "stop",
        "no thanks",
        "no,",
        "no.",
        "not now",
        "not yet",
        "hold off",
        "that's all",
        "that is all",
        "cancel",
        "done",
        "all set",
        "good for now",
    ]
    for trigger in stop_triggers:
        if trigger in normalized:
            return "stop"
    if normalized in {"no", "nah", "nope"}:
        return "stop"

    generate_triggers = [
        "yes",
        "yeah",
        "yep",
        "sure",
        "go ahead",
        "do it",
        "ready",
        "generate",
        "draft",
        "create the edits",
        "produce edits",
        "make the edits",
    ]
    for trigger in generate_triggers:
        if trigger in normalized:
            return "generate_edits"

    question_words = ("what", "how", "why", "where", "who", "when", "which")
    informational_triggers = [
        "rule",
        "guideline",
        "requirement",
        "allow",
        "limit",
        "title",
        "bullet",
        "description",
        "sku",
        "product",
        "issue",
        "gap",
        "compare",
        "difference",
        "explain",
        "detail",
        "info",
        "information",
    ]
    info_request_triggers = [
        "tell me",
        "share",
        "show",
        "remind",
        "need",
        "give",
        "provide",
        "explain",
        "details",
        "information",
    ]
    is_question_like = False
    if normalized.endswith("?") or normalized.startswith("?"):
        is_question_like = True
    if any(normalized.startswith(word + " ") for word in question_words):
        is_question_like = True
    if normalized.startswith("tell me") or normalized.startswith("share"):
        is_question_like = True
    if any(token in normalized for token in informational_triggers) and any(
        trigger in normalized for trigger in info_request_triggers
    ):
        is_question_like = True
    if is_question_like:
        return "answer_question"

    return "clarify"


def classify_review_followup(
    summary: str,
    user_input: str,
    *,
    client: Optional[Any] = None,
    additional_products: Optional[Any] = None,
) -> Action:
    """Classify how the review flow should proceed based on the user's reply."""

    heuristic_action = _heuristic_classification(summary, user_input)

    if client is not None:
        try:
            payload: Dict[str, Any] = {
                "issues_summary": summary or "",
                "user_input": user_input or "",
                "tools": _TOOL_METADATA,
            }
            if additional_products is not None:
                payload["additional_products"] = additional_products

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            data: Dict[str, Any] = json.loads(content)
            action = data.get("action")
            if action in _ALLOWED_ACTIONS:
                if action == "answer_question" or heuristic_action == "answer_question":
                    return "answer_question"
                return action  # type: ignore[return-value]
            raise _LLMResponseError(f"Invalid action from LLM: {action}")
        except Exception:
            # Fall back to heuristics on any LLM failure
            pass

    return heuristic_action


__all__ = ["Action", "classify_review_followup"]
