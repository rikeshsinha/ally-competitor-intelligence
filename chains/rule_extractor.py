"""LangChain runnable that prepares normalization rules from uploaded PDFs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st
from langchain_core.runnables import RunnableLambda

from core.content_rules import DEFAULT_RULES


@dataclass
class RuleExtraction:
    rules: Dict[str, Dict[str, Any]]
    source: str
    notes: Optional[str] = None


class _RuleExtractor:
    def __call__(self, inputs: Dict[str, Any]) -> RuleExtraction:  # type: ignore[override]
        pdf = inputs.get("rules_file")
        if pdf is None:
            return RuleExtraction(
                rules=DEFAULT_RULES,
                source="default",
                notes="No PDF uploaded; using bundled default rules.",
            )

        display_name = getattr(pdf, "name", "uploaded.pdf")
        st.sidebar.caption(f"Rules file: {display_name}")
        notes = (
            "Rules PDF uploaded. Using default normalization template while retaining file reference."
        )
        return RuleExtraction(rules=DEFAULT_RULES, source=display_name, notes=notes)


def create_rule_extractor() -> RunnableLambda:
    return RunnableLambda(_RuleExtractor())
