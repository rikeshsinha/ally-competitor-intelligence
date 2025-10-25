"""Review interaction orchestrator for the issues/edits flow."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, Optional

Action = Literal["generate_edits", "select_competitor", "stop", "clarify"]

_ALLOWED_ACTIONS = {"generate_edits", "select_competitor", "stop", "clarify"}

_LLM_SYSTEM_PROMPT = (
    "You orchestrate a review workflow for Amazon PDP content. "
    "Choose one action based on the latest issues summary and the user's reply. "
    "Actions: generate_edits (draft compliant edits now), select_competitor (ask user to pick a new competitor), "
    "stop (end the workflow), clarify (ask for more info). Return JSON with a single key 'action' using one of the "
    "allowed values."
)


class _LLMResponseError(RuntimeError):
    """Raised when the LLM result cannot be parsed."""


def _heuristic_classification(summary: str, user_input: str) -> Action:
    text = (user_input or "").strip()
    if not text:
        return "clarify"

    normalized = re.sub(r"\s+", " ", text.lower())

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
            return "select_competitor"
    if "competitor" in normalized and any(
        word in normalized for word in ["change", "switch", "different", "another"]
    ):
        return "select_competitor"

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

    if normalized.endswith("?") or normalized.startswith("?"):
        return "clarify"
    question_words = ("what", "how", "why", "where", "who", "when")
    if any(normalized.startswith(word + " ") for word in question_words):
        return "clarify"

    return "clarify"


def classify_review_followup(
    summary: str,
    user_input: str,
    *,
    client: Optional[Any] = None,
) -> Action:
    """Classify how the review flow should proceed based on the user's reply."""

    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "issues_summary": summary or "",
                                "user_input": user_input or "",
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            data: Dict[str, Any] = json.loads(content)
            action = data.get("action")
            if action in _ALLOWED_ACTIONS:
                return action  # type: ignore[return-value]
            raise _LLMResponseError(f"Invalid action from LLM: {action}")
        except Exception:
            # Fall back to heuristics on any LLM failure
            pass

    return _heuristic_classification(summary, user_input)


__all__ = ["Action", "classify_review_followup"]
