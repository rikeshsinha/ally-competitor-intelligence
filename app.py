from __future__ import annotations
import math
import os
import re
import json
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple
import textwrap

import pandas as pd
import streamlit as st

from core.content_rules import (
    DEFAULT_RULES,
    compare_fields,
    enforce_title_caps,
    extract_image_urls,
    limit_text_with_sentence_guard,
    split_bullets,
)
from graph.product_validation import build_product_validation_graph
from chains.rule_extractor import RuleExtraction
from chains.review_assistant import classify_review_followup
from chains.sku_extractor import (
    COMPETITOR_SELECTION_SESSION_KEYS,
    CLIENT_SELECTION_SESSION_KEYS,
    SKUData,
    _SKUExtractor,
    prime_competitor_chat_state,
)
from chains.competitor_finder import find_similar_competitors

# Optional OpenAI SDK (gracefully handle if not installed or no key)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Standard assistant follow-up after presenting rule checks
RULE_CHECKS_FOLLOWUP_PROMPT = (
    "Would you like me to generate a draft of compliant edits now, or do you have any other questions?"
)

# ---------------------------
# LLM prompt & call
# ---------------------------

SYSTEM_PROMPT = (
    "You are a meticulous Amazon PDP content editor for Pet Supplies. "
    "Follow the provided style rules strictly. Return compliant copy and explain briefly how each edit improves "
    "against the competitor. Always craft exactly three distinct edit variants when asked for draft edits."
)

USER_PROMPT_TEMPLATE = (
    "CLIENT SKU (brand={brand}):\n"
    "- Title: {c_title}\n"
    "- Bullets (rephrase these; do not introduce new claims beyond client copy. If fewer than five, pull any extra "
    "bullets only from the client description): {c_bullets}\n"
    "- Description (generate the new description strictly from this client description): {c_desc}\n"
    "\n"
    "COMPETITOR SKU (brand={comp_brand}) — context for rationale/comparison only. Do NOT use competitor language or "
    "claims in the proposed client copy.\n"
    "- Title: {k_title}\n"
    "- Bullets: {k_bullets}\n"
    "- Description: {k_desc}\n"
    "\n"
    "Rules JSON (style guide extraction):\n{rule_json}\n"
    "\n"
    "TASK: Propose an improved TITLE, 3-5 BULLETS, and a short DESCRIPTION for the CLIENT that stay grounded in "
    "the client's source content. Rephrase each client bullet, and only create additional bullets when needed using "
    "details from the client description. Also provide a brief rationale for each change that references what the "
    "competitor does while keeping all proposed client copy free of competitor language or claims.\n"
    "Return a JSON object with a `variants` array containing exactly three objects (no more, no fewer). Each variant "
    "object must include keys: title_edit (string), bullets_edits (array of 3-5 strings), description_edit (string), "
    "and rationales (array of strings). You may include an optional metadata object for any additional context you "
    "need."
)


def get_openai_client() -> Optional[OpenAI]:
    """Return an OpenAI client if a key is present (UI > env), else None."""
    # UI-provided key takes precedence
    ui_key = st.session_state.get("OPENAI_API_KEY_UI")
    api_key = ui_key or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def validate_openai_key() -> Tuple[bool, str]:
    """Try a very light call to validate the key. Stores result in session_state."""
    client = get_openai_client()
    if client is None:
        return False, "No key set or SDK unavailable"
    try:
        # Light call; models.list() is sufficient and inexpensive
        _ = client.models.list()
        st.session_state["openai_valid"] = True
        st.session_state["openai_error"] = ""
        return True, ""
    except Exception as e:
        st.session_state["openai_valid"] = False
        st.session_state["openai_error"] = str(e)
        return False, str(e)


def get_validation_graph() -> Any:
    if "product_validation_graph" not in st.session_state:

        def _run_validation(
            sku_data: SKUData, rule_data: RuleExtraction
        ) -> Dict[str, Any]:
            rules = rule_data.rules or DEFAULT_RULES
            client = getattr(sku_data, "client", {})
            competitor = getattr(sku_data, "competitor", {})
            return compare_fields(client, competitor, rules=rules)

        st.session_state["product_validation_graph"] = build_product_validation_graph(
            _run_validation
        )
    return st.session_state["product_validation_graph"]


def _trigger_rerun() -> None:
    """Call the appropriate Streamlit rerun helper across API versions."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is None:  # pragma: no cover - defensive guard for unexpected versions
        raise AttributeError("Streamlit rerun helper is unavailable")
    rerun_fn()


def _format_brand_dropdown_label(option: Optional[Dict[str, Any]]) -> str:
    if not option:
        return "Select a brand"
    brand_name = str(option.get("brand", "")).strip() or "Unnamed brand"
    ordinal = option.get("ordinal")
    if ordinal is not None:
        return f"{ordinal}. {brand_name}"
    return brand_name


def _format_product_dropdown_label(option: Optional[Dict[str, Any]]) -> str:
    if not option:
        return "Select a product"
    title = (
        str(option.get("title_label") or option.get("title_value") or "").strip()
        or "Unnamed product"
    )
    ordinal = option.get("brand_option_ordinal") or option.get("ordinal")
    if ordinal is not None:
        return f"{ordinal}. {title}"
    return title


def _get_brand_record(
    brand_groups: List[Dict[str, Any]], brand_name: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not brand_name:
        return None
    for record in brand_groups:
        if record.get("brand") == brand_name:
            return record
    return None


def _get_option_for_selection(
    selection: Dict[str, Any] | None, options: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not isinstance(selection, dict):
        return None
    target_row = selection.get("row_index")
    target_brand = selection.get("brand")
    if target_row is None or target_brand is None:
        return None
    for option in options:
        if (
            option.get("row_index") == target_row
            and option.get("brand") == target_brand
            ):
                return option
    return None


_CLIENT_SELECTION_RESET_KEYS = tuple(
    key
    for key in CLIENT_SELECTION_SESSION_KEYS
    if key not in {"client_chat_rendered_version"}
)


_AUTO_COMPETITOR_SESSION_KEYS = (
    "auto_competitor_recommendations",
    "auto_competitor_choice",
    "auto_competitor_confirmed_index",
    "auto_competitor_reference",
    "auto_competitor_error",
)


def _clear_auto_competitor_state() -> None:
    for key in _AUTO_COMPETITOR_SESSION_KEYS:
        st.session_state.pop(key, None)


def _handle_variant_checkbox_selection(index: int) -> None:
    """Ensure only a single LLM variant checkbox remains selected."""
    key = f"variant_accept_{index}"
    checked = st.session_state.get(key, False)
    if checked:
        st.session_state["selected_variant_index"] = index
        variants = st.session_state.get("llm_out", []) or []
        for idx in range(len(variants)):
            if idx == index:
                continue
            other_key = f"variant_accept_{idx}"
            if st.session_state.get(other_key):
                st.session_state[other_key] = False
    else:
        if st.session_state.get("selected_variant_index") == index:
            st.session_state["selected_variant_index"] = None


def _build_final_outputs(
    variant: Dict[str, Any],
    client_data: Dict[str, Any],
    rules: Dict[str, Any],
    *,
    title_max: int,
    bullet_max: int,
    desc_max: int,
) -> Tuple[str, str]:
    """Assemble the final Markdown/email strings for downloads."""
    title_text = limit_text_with_sentence_guard(
        str(variant.get("title_edit", "")),
        title_max,
        prefer_sentence=False,
    )
    desc_text = limit_text_with_sentence_guard(
        str(variant.get("description_edit", "")),
        desc_max,
        prefer_sentence=True,
    )

    final_md: List[str] = []
    final_md.append(f"# Final Content — Client SKU {client_data['sku']}")
    final_md.append("\n## Title (proposed)\n")
    final_md.append(title_text)
    final_md.append("\n## Bullets (proposed)\n")
    for b in variant.get("bullets_edits", [])[:bullet_max]:
        final_md.append(f"- {re.sub(r'[.!?]+$', '', str(b)).strip()}")
    final_md.append("\n\n## Description (proposed)\n")
    final_md.append(desc_text)

    final_md.append("\n\n## Rationale & Rule Compliance\n")
    final_md.append(
        f"- Title ≤ {title_max} chars; "
        f"{'brand required' if rules['title']['brand_required'] else 'brand optional'}; "
        f"{'avoid ALL CAPS' if rules['title']['no_all_caps'] else 'ALL CAPS allowed'}; "
        f"{'no promo language' if rules['title']['no_promo'] else 'promo allowed'}"
    )
    final_md.append(
        f"- Up to {bullet_max} bullets; "
        f"{'start with capitals' if rules['bullets']['start_capital'] else 'any case allowed'}; "
        f"{'no ending punctuation' if rules['bullets']['no_end_punct'] else 'ending punctuation allowed'}; "
        f"{'no promo/seller info' if rules['bullets']['no_promo_or_seller_info'] else 'promo allowed'}"
    )
    final_md.append(
        f"- Description ≤ {desc_max} chars; "
        f"{'no promo language' if rules['description']['no_promo'] else 'promo allowed'}; "
        f"{'avoid ALL CAPS' if rules['description']['sentence_caps'] else 'ALL CAPS allowed'}"
    )
    for r in variant.get("rationales", []):
        final_md.append(f"- {r}")

    final_md_str = "\n".join(final_md).strip()

    email_md = f"""
Subject: Approved PDP Edits for SKU {client_data["sku"]}

Hi team,

Please find the approved PDP content updates for SKU {client_data["sku"]} below. These adhere to the style guide (title≤{title_max}, ≤{bullet_max} bullets, description≤{desc_max}; no promo/seller info when restricted).

Title
-----
{title_text}

Bullets
-------
{chr(10).join([f"- {re.sub(r'[.!?]+$', '', str(b)).strip()}" for b in variant.get("bullets_edits", [])[:bullet_max]])}

Description
----------
{desc_text}

Rationale
---------
- Alignment with style rules; improved specificity vs competitor
{chr(10).join([f"- {r}" for r in variant.get("rationales", [])])}

Thanks,
Ally (Competitor Content Intelligence)
""".strip()

    return final_md_str, email_md


def _clear_client_selection_state(
    *, keep_choices: bool = True, reset_version: bool = False, clear_widget_keys: bool = False
) -> None:
    product_key = st.session_state.pop("client_product_select_key", None)
    if product_key:
        st.session_state.pop(product_key, None)
    for key in _CLIENT_SELECTION_RESET_KEYS:
        st.session_state.pop(key, None)
    if reset_version:
        st.session_state.pop("client_chat_rendered_version", None)
    if not keep_choices:
        st.session_state.pop("client_choices", None)
    if clear_widget_keys:
        for key in list(st.session_state.keys()):
            if key.startswith("client_brand_select_") or key.startswith(
                "client_product_select_"
            ):
                st.session_state.pop(key, None)


def _clear_competitor_selection_state(*, keep_choices: bool = False) -> None:
    product_key = st.session_state.pop("competitor_product_select_key", None)
    if product_key:
        st.session_state.pop(product_key, None)
    competitor_choices = (
        st.session_state.get("competitor_choices") if keep_choices else None
    )
    for key in COMPETITOR_SELECTION_SESSION_KEYS:
        if keep_choices and key == "competitor_choices":
            continue
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("competitor_brand_select_") or key.startswith(
            "competitor_product_select_"
        ):
            st.session_state.pop(key, None)
    _clear_auto_competitor_state()
    if keep_choices and competitor_choices is not None:
        st.session_state["competitor_choices"] = competitor_choices


def _set_competitor_selection_mode(mode: str) -> None:
    normalized = mode.strip().lower()
    current = st.session_state.get("competitor_selection_mode")
    if current == normalized:
        return
    _clear_competitor_selection_state(keep_choices=True)
    if normalized not in {"manual", "auto"}:
        st.session_state.pop("competitor_selection_mode", None)
        return
    st.session_state["competitor_selection_mode"] = normalized
    _trigger_rerun()


def _format_auto_competitor_option(
    idx: int, recommendations: List[Dict[str, Any]]
) -> str:
    if idx < 0 or idx >= len(recommendations):
        return "Select a recommended competitor"
    candidate = recommendations[idx]
    brand = str(candidate.get("brand", "")).strip() or "Unnamed brand"
    title = str(candidate.get("title", "")).strip() or "Untitled product"
    parts = [f"{brand} — {title}"]
    similarity = candidate.get("similarity")
    extras: List[str] = []
    if isinstance(similarity, (int, float)):
        try:
            extras.append(f"{similarity * 100:.1f}% match")
        except Exception:
            pass
    rank_value = candidate.get("rank")
    if isinstance(rank_value, (int, float)) and math.isfinite(float(rank_value)):
        if isinstance(rank_value, float) and not rank_value.is_integer():
            rank_display = f"Avg rank {rank_value:.1f}"
        else:
            rank_display = f"Avg rank {int(round(rank_value))}"
        extras.append(rank_display)
    if extras:
        parts.append(f"({' • '.join(extras)})")
    return " ".join(parts)


def _build_sku_data_for_auto() -> Optional[SKUData]:
    state = st.session_state.get("_sku_extractor_state")
    if not isinstance(state, dict):
        return None
    df = state.get("dataframe")
    column_map = state.get("column_map")
    client_idx = state.get("client_row_index")
    if not isinstance(client_idx, int) or client_idx < 0:
        selected_client = st.session_state.get("selected_client")
        client_idx = (
            selected_client.get("row_index")
            if isinstance(selected_client, dict)
            else None
        )
    if not isinstance(client_idx, int) or client_idx < 0:
        return None
    if df is None or column_map is None:
        return None
    if client_idx >= len(df):
        return None
    extractor = _SKUExtractor()
    client_row = df.iloc[[client_idx]]
    client_record = extractor._record_from_row(client_row, column_map)
    client_record["row_index"] = client_idx
    return SKUData(
        dataframe=df,
        column_map=column_map,
        client=client_record,
        competitor={},
    )


def _match_auto_candidate_to_option(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    choices = st.session_state.get("competitor_choices")
    if not isinstance(choices, dict):
        return None
    options: List[Dict[str, Any]] = choices.get("options") or []
    if not options:
        return None
    state = st.session_state.get("_sku_extractor_state")
    df = state.get("dataframe") if isinstance(state, dict) else None
    column_map = state.get("column_map") if isinstance(state, dict) else None
    candidate_brand = str(candidate.get("brand", "")).strip().lower()
    candidate_title = str(candidate.get("title", "")).strip().lower()
    candidate_sku = str(candidate.get("sku", "")).strip()

    def _option_matches(option: Dict[str, Any]) -> bool:
        brand = str(option.get("brand", "")).strip().lower()
        if candidate_brand and brand != candidate_brand:
            return False
        title_value = str(option.get("title_value", "")).strip().lower()
        if candidate_title and title_value == candidate_title:
            return True
        row_index = option.get("row_index")
        if (
            df is not None
            and column_map
            and isinstance(row_index, int)
            and 0 <= row_index < len(df)
        ):
            row = df.iloc[row_index]
            sku_col = column_map.get("sku_col")
            possible_skus: List[str] = []
            if sku_col and sku_col in df.columns:
                possible_skus.append(str(row.get(sku_col, "")).strip())
            if "_display_sku" in df.columns:
                possible_skus.append(str(row.get("_display_sku", "")).strip())
            if candidate_sku and candidate_sku in possible_skus:
                return True
        return False

    for option in options:
        if _option_matches(option):
            return option

    if candidate_brand:
        for option in options:
            brand = str(option.get("brand", "")).strip().lower()
            if brand == candidate_brand:
                return option

    return None


def _render_competitor_mode_prompt(*, key_namespace: str = "competitor_mode") -> None:
    mode = st.session_state.get("competitor_selection_mode")

    with st.chat_message("assistant"):
        st.markdown("How should Ally line up the competitor SKU for this matchup?")
        if mode == "auto":
            st.caption("Current mode: Ally is auto-scouting competitors for you.")
        elif mode == "manual":
            st.caption("Current mode: You're hand-picking the competitor and Ally will follow your lead.")
        else:
            st.caption("Pick a mode to continue.")
        manual_col, auto_col = st.columns(2)
        manual_clicked = manual_col.button(
            "I'll pick the competitor", key=f"{key_namespace}_manual"
        )
        auto_clicked = auto_col.button(
            "Find best competitor", key=f"{key_namespace}_auto"
        )
    if manual_clicked:
        _clear_auto_competitor_state()
        _set_competitor_selection_mode("manual")
    elif auto_clicked:
        _set_competitor_selection_mode("auto")


def _apply_auto_selection(
    choice_idx: int, recommendations: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if choice_idx < 0 or choice_idx >= len(recommendations):
        return None
    candidate = recommendations[choice_idx]
    matched_option = _match_auto_candidate_to_option(candidate)
    if not matched_option:
        st.session_state["auto_competitor_error"] = (
            "Unable to map the suggested competitor to the catalog. Try manual selection."
        )
        return None

    st.session_state.pop("auto_competitor_error", None)
    selection_record = {
        "brand": matched_option.get("brand"),
        "row_index": matched_option.get("row_index"),
        "label": matched_option.get("title_label")
        or matched_option.get("title_value"),
        "ordinal": matched_option.get("ordinal")
        or matched_option.get("brand_option_ordinal"),
    }

    st.session_state["selected_competitor"] = selection_record
    brand = selection_record.get("brand")
    if brand:
        st.session_state["selected_competitor_brand"] = brand
        st.session_state["competitor_brand_user_ack"] = brand
    else:
        st.session_state.pop("selected_competitor_brand", None)
        st.session_state.pop("competitor_brand_user_ack", None)

    label = selection_record.get("label")
    if label and brand:
        st.session_state["competitor_product_user_ack"] = f"{brand} — {label}"
    elif label:
        st.session_state["competitor_product_user_ack"] = label
    elif brand:
        st.session_state["competitor_product_user_ack"] = brand
    else:
        st.session_state.pop("competitor_product_user_ack", None)

    st.session_state["competitor_chat_confirmed"] = True
    st.session_state["auto_competitor_confirmed_index"] = choice_idx
    return selection_record


def _render_auto_competitor_selection_ui() -> Optional[Dict[str, Any]]:
    sku_data = _build_sku_data_for_auto()
    if sku_data is None:
        return None

    current_reference = st.session_state.get("auto_competitor_reference")
    if current_reference != st.session_state.get("selected_client", {}).get("row_index"):
        _clear_auto_competitor_state()

    recommendations: Optional[List[Dict[str, Any]]] = st.session_state.get(
        "auto_competitor_recommendations"
    )
    if not isinstance(recommendations, list):
        recommendations = find_similar_competitors(sku_data)
        st.session_state["auto_competitor_recommendations"] = recommendations
        st.session_state["auto_competitor_reference"] = st.session_state.get(
            "selected_client", {}
        ).get("row_index")
    if not recommendations:
        with st.chat_message("assistant"):
            st.warning(
                "Ally couldn't spot a close competitor automatically. Give manual mode a whirl instead."
            )
        return None

    choice_key = "auto_competitor_choice"
    choice_options = [-1] + list(range(len(recommendations)))
    confirmed_idx = st.session_state.get("auto_competitor_confirmed_index")

    with st.chat_message("assistant"):
        st.markdown(
            "Ally's shortlist of lookalike competitors is ready—choose your challenger to compare."
        )
        selected_option = st.selectbox(
            "Select an automatically suggested competitor",
            options=choice_options,
            index=choice_options.index(
                st.session_state.get(choice_key, -1)
                if st.session_state.get(choice_key, -1) in choice_options
                else -1
            ),
            format_func=lambda idx: _format_auto_competitor_option(idx, recommendations),
            key=choice_key,
            label_visibility="collapsed",
        )
        error_message = st.session_state.get("auto_competitor_error")
        if error_message:
            st.warning(error_message)

    final_selection = st.session_state.get("selected_competitor")

    if selected_option == -1:
        pass
    elif selected_option == confirmed_idx and final_selection:
        pass
    else:
        selection_record = _apply_auto_selection(selected_option, recommendations)
        if selection_record is not None:
            _trigger_rerun()
            return selection_record
        final_selection = st.session_state.get("selected_competitor")

    product_ack = st.session_state.get("competitor_product_user_ack")
    if product_ack:
        with st.chat_message("user"):
            st.markdown(f"Compare against **{product_ack}**.")

    if final_selection and st.session_state.get("competitor_chat_confirmed"):
        competitor_brand = final_selection.get("brand") or "the selected brand"
        competitor_label = final_selection.get("label")
        if competitor_label:
            confirmation_text = (
                f"Locked in **{competitor_brand} — {competitor_label}** as the competitor."
            )
        else:
            confirmation_text = f"Locked in **{competitor_brand}** as the competitor."
        with st.chat_message("assistant"):
            st.markdown(confirmation_text)

    return final_selection


def _render_client_selection_ui() -> Optional[Dict[str, Any]]:
    choices = st.session_state.get("client_choices")
    if not isinstance(choices, dict):
        return st.session_state.get("selected_client")

    brand_groups: List[Dict[str, Any]] = choices.get("brand_groups") or []
    if not brand_groups:
        return st.session_state.get("selected_client")

    version = choices.get("version")
    if version and st.session_state.get("client_chat_rendered_version") != version:
        _clear_client_selection_state(reset_version=True, clear_widget_keys=True)
        _clear_competitor_selection_state()
        st.session_state["client_chat_rendered_version"] = version

    previous_selection = st.session_state.get("selected_client")
    previous_brand = st.session_state.get("selected_client_brand")

    brand_groups_with_placeholder: List[Optional[Dict[str, Any]]] = [None]
    brand_groups_with_placeholder.extend(brand_groups)
    brand_select_key = f"client_brand_select_{version or 'default'}"
    current_brand_record = _get_brand_record(
        brand_groups, st.session_state.get("selected_client_brand")
    )
    brand_index = 0
    if current_brand_record and current_brand_record in brand_groups_with_placeholder:
        brand_index = brand_groups_with_placeholder.index(current_brand_record)

    with st.chat_message("assistant"):
        st.markdown("Let's choose the client brand Ally should polish up:")
        selected_brand_option = st.selectbox(
            "Select a client brand",
            options=brand_groups_with_placeholder,
            index=brand_index,
            format_func=_format_brand_dropdown_label,
            key=brand_select_key,
            label_visibility="collapsed",
        )

    if selected_brand_option is None:
        if previous_brand is not None or previous_selection is not None:
            _clear_client_selection_state(keep_choices=True)
    else:
        brand_name = str(selected_brand_option.get("brand", "")).strip()
        if not brand_name:
            if previous_brand is not None or previous_selection is not None:
                _clear_client_selection_state(keep_choices=True)
        elif previous_brand != brand_name:
            _clear_client_selection_state(keep_choices=True)
            st.session_state["selected_client_brand"] = brand_name
        else:
            st.session_state["selected_client_brand"] = brand_name

    brand_name = st.session_state.get("selected_client_brand")
    if brand_name:
        st.session_state["client_brand_user_ack"] = brand_name
    else:
        st.session_state.pop("client_brand_user_ack", None)

    brand_ack = st.session_state.get("client_brand_user_ack")
    if brand_ack:
        with st.chat_message("user"):
            st.markdown(f"Let's optimize **{brand_ack}**.")

    brand_record = _get_brand_record(brand_groups, brand_name)
    selected_product_option: Optional[Dict[str, Any]] = None

    if brand_record:
        product_options = brand_record.get("options") or []
        matched_option = _get_option_for_selection(
            st.session_state.get("selected_client"), product_options
        )
        product_options_with_placeholder: List[Optional[Dict[str, Any]]] = [None]
        product_options_with_placeholder.extend(product_options)
        product_index = 0
        if matched_option and matched_option in product_options_with_placeholder:
            product_index = product_options_with_placeholder.index(matched_option)
        product_key = (
            f"client_product_select_{version or 'default'}_{brand_record.get('brand')}"
        )
        st.session_state["client_product_select_key"] = product_key

        with st.chat_message("assistant"):
            st.markdown(
                f"Now pick which **{brand_record.get('brand', 'this brand')}** product Ally should glow-up next:"
            )
            selected_product_option = st.selectbox(
                "Select a client product",
                options=product_options_with_placeholder,
                index=product_index,
                format_func=_format_product_dropdown_label,
                key=product_key,
                label_visibility="collapsed",
            )

    if selected_product_option is None:
        st.session_state.pop("client_product_user_ack", None)
        st.session_state.pop("client_chat_confirmed", None)
        if st.session_state.get("selected_client") is not None:
            st.session_state.pop("selected_client", None)
    else:
        selection_record = {
            "brand": str(selected_product_option.get("brand", "")).strip(),
            "row_index": selected_product_option.get("row_index"),
            "label": str(
                selected_product_option.get("title_label")
                or selected_product_option.get("title_value")
                or ""
            ).strip(),
            "ordinal": selected_product_option.get("ordinal")
            or selected_product_option.get("brand_option_ordinal"),
        }
        st.session_state["selected_client"] = selection_record
        st.session_state["client_chat_confirmed"] = True
        label = selection_record.get("label")
        brand_for_ack = selection_record.get("brand", "") or ""
        if brand_for_ack:
            brand_for_ack = str(brand_for_ack).strip()
        if label and brand_for_ack:
            ack_text = f"{brand_for_ack} — {label}"
        elif label:
            ack_text = label
        else:
            ack_text = brand_for_ack
        st.session_state["client_product_user_ack"] = ack_text

    product_ack = st.session_state.get("client_product_user_ack")
    if product_ack:
        with st.chat_message("user"):
            st.markdown(f"Focus on **{product_ack}**.")

    final_selection = st.session_state.get("selected_client")
    if final_selection and st.session_state.get("client_chat_confirmed"):
        brand_label = final_selection.get("brand") or "your selected brand"
        product_label = final_selection.get("label")
        if product_label:
            confirmation_text = (
                f"Fantastic! Ally is on **{brand_label} — {product_label}** duty."
            )
        else:
            confirmation_text = f"Fantastic! Ally is on **{brand_label}** duty."
        with st.chat_message("assistant"):
            st.markdown(confirmation_text)

    if previous_selection != final_selection:
        _clear_competitor_selection_state()
        _trigger_rerun()

    return final_selection


def _render_competitor_selection_ui() -> Optional[Dict[str, Any]]:
    if st.session_state.get("competitor_selection_mode") != "manual":
        return st.session_state.get("selected_competitor")

    choices = st.session_state.get("competitor_choices")
    if not isinstance(choices, dict):
        return st.session_state.get("selected_competitor")
    options: List[Dict[str, Any]] = choices.get("options") or []
    brand_groups: List[Dict[str, Any]] = choices.get("brand_groups") or []
    if not options or not brand_groups:
        return st.session_state.get("selected_competitor")

    version = choices.get("version")
    if version and st.session_state.get("competitor_chat_rendered_version") != version:
        st.session_state["competitor_chat_rendered_version"] = version
        st.session_state.pop("competitor_chat_confirmed", None)
        st.session_state.pop("selected_competitor_brand", None)
        st.session_state.pop("competitor_product_prompt_brand", None)
        st.session_state.pop("competitor_brand_user_ack", None)
        st.session_state.pop("competitor_product_user_ack", None)
        product_key = st.session_state.pop("competitor_product_select_key", None)
        if product_key:
            st.session_state.pop(product_key, None)
        for key in list(st.session_state.keys()):
            if key.startswith("competitor_brand_select_") or key.startswith(
                "competitor_product_select_"
            ):
                st.session_state.pop(key, None)

    current_selection = st.session_state.get("selected_competitor")
    if current_selection and not st.session_state.get("competitor_chat_confirmed"):
        matched_option = _get_option_for_selection(current_selection, options)
        if matched_option:
            confirmation = (
                f"Locked in **{matched_option['brand']} — {matched_option['title_label']}** as the competitor."
            )
        else:
            confirmation = "Using your previously selected competitor."
        st.session_state["competitor_chat_confirmed"] = True

    brand_groups_with_placeholder: List[Optional[Dict[str, Any]]] = [None]
    brand_groups_with_placeholder.extend(brand_groups)
    brand_select_key = f"competitor_brand_select_{version or 'default'}"
    selected_brand_name = st.session_state.get("selected_competitor_brand")
    current_brand_record = _get_brand_record(brand_groups, selected_brand_name)
    brand_index = 0
    if current_brand_record and current_brand_record in brand_groups_with_placeholder:
        brand_index = brand_groups_with_placeholder.index(current_brand_record)

    with st.chat_message("assistant"):
        st.markdown("First up, pick the competitor brand Ally should scout:")
        selected_brand_option = st.selectbox(
            "Select a competitor brand",
            options=brand_groups_with_placeholder,
            index=brand_index,
            format_func=_format_brand_dropdown_label,
            key=brand_select_key,
            label_visibility="collapsed",
        )

    if selected_brand_option is None:
        st.session_state.pop("selected_competitor_brand", None)
        st.session_state.pop("competitor_brand_user_ack", None)
        prev_product_key = st.session_state.pop("competitor_product_select_key", None)
        if prev_product_key:
            st.session_state.pop(prev_product_key, None)
        st.session_state.pop("competitor_product_user_ack", None)
        st.session_state.pop("selected_competitor", None)
        st.session_state.pop("competitor_chat_confirmed", None)
    else:
        brand_name = str(selected_brand_option.get("brand", "")).strip()
        previous_brand = st.session_state.get("selected_competitor_brand")
        if brand_name:
            st.session_state["selected_competitor_brand"] = brand_name
            if previous_brand != brand_name:
                prev_product_key = st.session_state.pop(
                    "competitor_product_select_key", None
                )
                if prev_product_key:
                    st.session_state.pop(prev_product_key, None)
                st.session_state.pop("competitor_product_user_ack", None)
                st.session_state.pop("selected_competitor", None)
                st.session_state.pop("competitor_chat_confirmed", None)
        else:
            st.session_state.pop("selected_competitor_brand", None)

    brand_name = st.session_state.get("selected_competitor_brand")
    if brand_name:
        st.session_state["competitor_brand_user_ack"] = brand_name
    else:
        st.session_state.pop("competitor_brand_user_ack", None)

    brand_ack = st.session_state.get("competitor_brand_user_ack")
    if brand_ack:
        with st.chat_message("user"):
            st.markdown(f"Let's review **{brand_ack}**.")

    brand_record = _get_brand_record(brand_groups, brand_name)
    selected_product_option: Optional[Dict[str, Any]] = None

    if brand_record:
        product_options = brand_record.get("options") or []
        matched_option = _get_option_for_selection(
            st.session_state.get("selected_competitor"), product_options
        )
        product_index = 0
        product_options_with_placeholder: List[Optional[Dict[str, Any]]] = [None]
        product_options_with_placeholder.extend(product_options)
        if matched_option and matched_option in product_options_with_placeholder:
            product_index = product_options_with_placeholder.index(matched_option)
        product_key = f"competitor_product_select_{version or 'default'}_{brand_record.get('brand')}"
        st.session_state["competitor_product_select_key"] = product_key

        with st.chat_message("assistant"):
            st.markdown(
                f"Next, choose which **{brand_record.get('brand', 'this brand')}** product Ally should stack up against your client:"
            )
            selected_product_option = st.selectbox(
                "Select a competitor product",
                options=product_options_with_placeholder,
                index=product_index,
                format_func=_format_product_dropdown_label,
                key=product_key,
                label_visibility="collapsed",
            )

    if selected_product_option is None:
        if brand_record is None:
            st.session_state.pop("competitor_product_select_key", None)
        st.session_state.pop("competitor_product_user_ack", None)
        st.session_state.pop("selected_competitor", None)
        st.session_state.pop("competitor_chat_confirmed", None)
    else:
        selection_record = {
            "brand": selected_product_option.get("brand"),
            "row_index": selected_product_option.get("row_index"),
            "label": selected_product_option.get("title_label")
            or selected_product_option.get("title_value"),
            "ordinal": selected_product_option.get("ordinal")
            or selected_product_option.get("brand_option_ordinal"),
        }
        st.session_state["selected_competitor"] = selection_record
        st.session_state["competitor_chat_confirmed"] = True
        label = selection_record.get("label")
        if label:
            st.session_state["competitor_product_user_ack"] = (
                f"{selection_record.get('brand', '')} — {label}"
            )
        else:
            st.session_state["competitor_product_user_ack"] = selection_record.get(
                "brand"
            )

    product_ack = st.session_state.get("competitor_product_user_ack")
    if product_ack:
        with st.chat_message("user"):
            st.markdown(f"Compare against **{product_ack}**.")

    final_selection = st.session_state.get("selected_competitor")
    if final_selection and st.session_state.get("competitor_chat_confirmed"):
        competitor_brand = final_selection.get("brand") or "the selected brand"
        competitor_label = final_selection.get("label")
        if competitor_label:
            confirmation_text = (
                f"Locked in **{competitor_brand} — {competitor_label}** as the competitor."
            )
        else:
            confirmation_text = f"Locked in **{competitor_brand}** as the competitor."
        with st.chat_message("assistant"):
            st.markdown(confirmation_text)

    return final_selection


def call_llm(
    client_data: Dict[str, Any],
    comp_data: Dict[str, Any],
    rules: Dict[str, Any],
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    client = get_openai_client()
    prompt = USER_PROMPT_TEMPLATE.format(
        brand=client_data.get("brand", ""),
        c_title=client_data.get("title", ""),
        c_bullets=split_bullets(client_data.get("bullets", "")),
        c_desc=client_data.get("description", ""),
        comp_brand=comp_data.get("brand", ""),
        k_title=comp_data.get("title", ""),
        k_bullets=split_bullets(comp_data.get("bullets", "")),
        k_desc=comp_data.get("description", ""),
        rule_json=json.dumps(rules, indent=2, sort_keys=True),
    )

    normalized_context = (user_context or "").strip()
    if normalized_context:
        prompt = "\n\n".join(
            [prompt, f"USER GUIDANCE:\n{normalized_context}"]
        )

    if client is None:
        # Heuristic fallback: create simple suggestions without LLM
        def _normalize_clause(text: str) -> str:
            text = re.sub(r"\s+", " ", text.strip())
            text = re.sub(r"[.!?;,:]+$", "", text)
            return text.strip()

        bullet_input = client_data.get("bullets", "")
        split_candidates = split_bullets(bullet_input)
        original_bullets: List[str] = []
        for entry in split_candidates:
            if isinstance(entry, str) and re.search(r"[\n\r\|•;]+", entry):
                original_bullets.extend(
                    [seg for seg in re.split(r"[\n\r\|•;]+", entry) if seg.strip()]
                )
            else:
                text = str(entry).strip()
                if text:
                    original_bullets.append(text)
        if not original_bullets and isinstance(bullet_input, str):
            original_bullets = [
                seg for seg in re.split(r"[\n\r\|•;]+", bullet_input) if seg.strip()
            ]
        normalized_bullets = []
        for bullet in original_bullets:
            normalized = _normalize_clause(bullet)
            if normalized:
                normalized_bullets.append(normalized)

        max_bullets = rules["bullets"].get("max_count", 5)
        if len(normalized_bullets) < max_bullets:
            description_text = client_data.get("description") or ""
            # Split description into clauses/sentences using punctuation/newlines
            clauses = re.split(r"[\n\r;]+|(?<=[.!?])\s+", description_text)
            for clause in clauses:
                normalized_clause = _normalize_clause(clause)
                if normalized_clause and normalized_clause not in normalized_bullets:
                    normalized_bullets.append(normalized_clause)
                if len(normalized_bullets) >= max_bullets:
                    break

        fixed_bullets: List[str] = normalized_bullets[:max_bullets]

        base_brand = (client_data.get("brand") or "").strip()
        # Title heuristic: ensure brand at start, trim to rule max characters
        raw_title = client_data.get("title") or ""
        if base_brand and base_brand.lower() not in raw_title.lower():
            title = f"{base_brand} " + raw_title
        else:
            title = raw_title
        title = enforce_title_caps(title)
        title = limit_text_with_sentence_guard(
            title,
            rules["title"].get("max_chars", len(title)),
            prefer_sentence=False,
        )

        # Description heuristic: keep client description if present, otherwise reuse bullet text
        desc = (client_data.get("description") or "").strip()
        if not desc and fixed_bullets:
            desc = " ".join(fixed_bullets)
        desc = re.sub(r"\s+", " ", desc).strip()
        desc = limit_text_with_sentence_guard(
            desc,
            rules["description"].get("max_chars", len(desc)),
            prefer_sentence=True,
        )

        variant = {
            "title_edit": title,
            "bullets_edits": fixed_bullets,
            "description_edit": desc,
            "rationales": [
                f"Ensure brand appears in title and stays within {rules['title']['max_chars']} characters",
                f"Use up to {rules['bullets']['max_count']} concise bullets starting with a capital letter, no ending punctuation",
                f"Short description (<= {rules['description']['max_chars']} chars) with clear benefit; remove promo language if present",
            ],
        }
        response: Dict[str, Any] = {"variants": [variant], "_llm": False}
        if normalized_context:
            response["metadata"] = {"user_guidance": f"USER GUIDANCE:\n{normalized_context}"}
        return response

    # Real LLM call
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = rsp.choices[0].message.content
        data = json.loads(content)

        variants_payload = data.get("variants") if isinstance(data, dict) else None
        normalized_variants: List[Dict[str, Any]] = []
        if isinstance(variants_payload, list):
            for variant in variants_payload:
                if not isinstance(variant, dict):
                    continue
                title_edit = str(variant.get("title_edit", ""))
                bullets = variant.get("bullets_edits", [])
                if not isinstance(bullets, list):
                    bullets = [bullets]
                bullets_edits = [str(b).strip() for b in bullets if str(b).strip()]
                description_edit = str(variant.get("description_edit", ""))
                rationales = variant.get("rationales", [])
                if not isinstance(rationales, list):
                    rationales = [rationales]
                rationale_entries = [str(r).strip() for r in rationales if str(r).strip()]
                normalized_variants.append(
                    {
                        "title_edit": title_edit,
                        "bullets_edits": bullets_edits,
                        "description_edit": description_edit,
                        "rationales": rationale_entries,
                    }
                )

        # Legacy compatibility: wrap single-response payloads into variants list
        if not normalized_variants and isinstance(data, dict):
            candidate_keys = {"title_edit", "bullets_edits", "description_edit", "rationales"}
            if candidate_keys.intersection(data.keys()):
                single_variant = {
                    "title_edit": str(data.get("title_edit", "")),
                    "bullets_edits": [
                        str(b).strip()
                        for b in (
                            data.get("bullets_edits")
                            if isinstance(data.get("bullets_edits"), list)
                            else [data.get("bullets_edits")]
                        )
                        if b is not None and str(b).strip()
                    ],
                    "description_edit": str(data.get("description_edit", "")),
                    "rationales": [
                        str(r).strip()
                        for r in (
                            data.get("rationales")
                            if isinstance(data.get("rationales"), list)
                            else [data.get("rationales")]
                        )
                        if r is not None and str(r).strip()
                    ],
                }
                normalized_variants.append(single_variant)

        if not normalized_variants:
            raise ValueError("LLM response did not include any variants")

        data["variants"] = normalized_variants[:3]
        data["_llm"] = True
        return data
    except Exception as e:
        # Fallback to heuristic
        variant = {
            "title_edit": limit_text_with_sentence_guard(
                enforce_title_caps(client_data.get("title") or ""),
                rules["title"].get("max_chars", 0),
                prefer_sentence=False,
            ),
            "bullets_edits": [
                "Durable construction",
                "Comfortable fit",
                "Easy to clean",
            ],
            "description_edit": limit_text_with_sentence_guard(
                "Compact design for everyday use; easy to clean; ideal for most pets",
                rules["description"].get("max_chars", 0),
                prefer_sentence=True,
            ),
            "rationales": ["LLM error; generated heuristic placeholders"],
        }
        response: Dict[str, Any] = {"variants": [variant], "_llm": False}
        metadata: Dict[str, Any] = {"error": str(e)}
        if normalized_context:
            metadata["user_guidance"] = f"USER GUIDANCE:\n{normalized_context}"
        if metadata:
            response["metadata"] = metadata
        return response


def _format_rules_for_answer(rules: Dict[str, Any]) -> str:
    title_rules = rules.get("title", {})
    bullet_rules = rules.get("bullets", {})
    desc_rules = rules.get("description", {})
    parts = ["Key rule constraints:"]
    if title_rules:
        parts.append(
            f"- Title: ≤{title_rules.get('max_chars', 'n/a')} characters; case guidance: {title_rules.get('style', '—')}"
        )
    if bullet_rules:
        parts.append(
            f"- Bullets: up to {bullet_rules.get('max_count', 'n/a')} entries; max length ≈ {bullet_rules.get('max_chars', 'n/a')} each"
        )
    if desc_rules:
        parts.append(f"- Description: ≤{desc_rules.get('max_chars', 'n/a')} characters")
    other_rules = [
        key for key in rules.keys() if key not in {"title", "bullets", "description"}
    ]
    if other_rules:
        parts.append("- Additional policies: " + ", ".join(sorted(other_rules)))
    return "\n".join(parts)


def _summarize_product(label: str, data: Dict[str, Any]) -> str:
    bullets = [
        f"• {b}" for b in split_bullets(data.get("bullets", "")) if str(b).strip()
    ]
    details = [
        f"Brand: {data.get('brand', '—')}",
        f"SKU: {data.get('sku', data.get('sku_id', '—'))}",
        f"Title: {data.get('title', '—')}",
    ]
    description = data.get("description")
    if description:
        details.append(f"Description: {description}")
    if bullets:
        details.append("Bullets:\n" + "\n".join(bullets))
    avg_rank = data.get("avg_rank_search")
    if avg_rank not in (None, "", "nan"):
        details.append(f"Average search rank: {avg_rank}")
    return f"{label} details:\n" + "\n".join(details)


def _fallback_answer(
    question: str,
    *,
    issues_summary: str,
    rule_text: str,
    client_text: str,
    competitor_text: str,
) -> str:
    lower_q = question.lower()
    sections: List[str] = []
    if any(token in lower_q for token in ["rule", "guideline", "limit", "allow"]):
        sections.append(rule_text)
    if any(
        token in lower_q
        for token in ["client", "sku", "product", "title", "bullet", "description"]
    ):
        sections.append(client_text)
    if any(token in lower_q for token in ["competitor", "compare", "gap", "issue"]):
        sections.append(competitor_text)
    if issues_summary and ("issue" in lower_q or "gap" in lower_q):
        sections.append("Issues & gaps summary:\n" + issues_summary)
    if not sections:
        sections.extend([rule_text, client_text, competitor_text])
    header = "I don't have model access right now, but here's what I can confirm from the uploaded materials:"
    return header + "\n\n" + "\n\n".join(sections)


def answer_review_question(
    question: str,
    *,
    issues_summary: str,
    current_rules: Dict[str, Any],
    client_data: Dict[str, Any],
    competitor_data: Dict[str, Any],
    client: Optional[OpenAI] = None,
) -> Tuple[str, List[str]]:
    rule_text = _format_rules_for_answer(current_rules)
    client_text = _summarize_product("Client SKU", client_data)
    competitor_text = _summarize_product("Competitor SKU", competitor_data)

    context_sections = [rule_text, client_text, competitor_text]
    if issues_summary:
        context_sections.append("Issues & gaps summary:\n" + issues_summary)
    context_blob = "\n\n".join(context_sections)

    answer_text = ""
    if client is None:
        client = get_openai_client()

    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a retail content specialist. Use only the provided context to answer "
                            "questions about rules, client SKU details, or competitor insights. Reference key "
                            "constraints numerically when possible."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": question,
                                "context": context_blob,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0,
            )
            answer_text = response.choices[0].message.content or ""
        except Exception:
            answer_text = ""

    if not answer_text:
        answer_text = _fallback_answer(
            question,
            issues_summary=issues_summary,
            rule_text=rule_text,
            client_text=client_text,
            competitor_text=competitor_text,
        )

    normalized = textwrap.dedent(answer_text).strip()
    if not normalized:
        normalized = "I'm unable to find that information right now. Please review the uploaded summary above."

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks = [p + "\n\n" for p in paragraphs]
    if not chunks:
        chunks = [normalized]
    return normalized, chunks


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Competitor Content Intelligence", layout="wide")

# --- High-contrast styling for selectbox across themes ---
st.markdown(
    """
    <style>
    :root {
        --_ally-select-fg: var(--text-color, #111111);
        --_ally-select-bg: var(--background-color, #ffffff);
        --_ally-select-bg-active: var(--secondary-background-color, #f3f4f6);
        --_ally-select-muted: var(--secondary-text-color, #6b7280);
    }

    .stSelectbox [data-baseweb="select"] > div {
        color: var(--_ally-select-fg) !important;
        background-color: var(--_ally-select-bg) !important;
    }

    .stSelectbox [data-baseweb="select"] span {
        color: var(--_ally-select-fg) !important;
    }

    .stSelectbox [data-baseweb="select"] [aria-hidden="true"] {
        color: var(--_ally-select-muted) !important;
    }

    [data-baseweb="menu"] {
        background-color: var(--_ally-select-bg) !important;
        color: var(--_ally-select-fg) !important;
    }

    [data-baseweb="menu"] [role="option"] {
        color: var(--_ally-select-fg) !important;
    }

    [data-baseweb="menu"] [aria-selected="true"] {
        background-color: var(--_ally-select-bg-active) !important;
        color: var(--_ally-select-fg) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ally's Competitor Content Intelligence Lab")

if not st.session_state.get("ally_welcome_announced"):
    with st.chat_message("assistant"):
        st.markdown(
            "Hi there! I'm Ally—your upbeat PDP wingmate. Let's turn these listings into standouts together!"
        )
    st.session_state["ally_welcome_announced"] = True

with st.expander("About this demo"):
    st.markdown(
        "Ally compares your client SKU to a competitor and drafts compliant edits (title, bullets, description) with flair.\n"
        "Rules stay grounded in the Amazon Pet Supplies style guide for UAE (title≤50, ≤5 bullets, description≤200, no promo)."
    )

# Sidebar: LLM settings
st.sidebar.header("LLM Settings")
_ui_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.get("OPENAI_API_KEY_UI", ""),
)
if _ui_key != st.session_state.get("OPENAI_API_KEY_UI"):
    st.session_state["OPENAI_API_KEY_UI"] = _ui_key

col_a, col_b = st.sidebar.columns([1, 1])
if col_a.button("Validate key"):
    ok, err = validate_openai_key()
    if ok:
        st.sidebar.success("Key validated")
    else:
        st.sidebar.error("Invalid key")
        if err:
            st.sidebar.caption(err)

# Status pill
valid_flag = st.session_state.get("openai_valid")
if valid_flag is True:
    st.sidebar.markdown(
        '<div style="display:inline-block;padding:4px 10px;border-radius:999px;background:#16a34a;color:#fff;font-weight:600">OpenAI: Valid</div>',
        unsafe_allow_html=True,
    )
elif (_ui_key or os.getenv("OPENAI_API_KEY")) and valid_flag is False:
    st.sidebar.markdown(
        '<div style="display:inline-block;padding:4px 10px;border-radius:999px;background:#dc2626;color:#fff;font-weight:600">OpenAI: Invalid</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div style="display:inline-block;padding:4px 10px;border-radius:999px;background:#6b7280;color:#fff;font-weight:600">OpenAI: Not set</div>',
        unsafe_allow_html=True,
    )

# Sidebar: file inputs
st.sidebar.header("Inputs")
rules_file = st.sidebar.file_uploader("Upload Rules PDF", type=["pdf"], key="rules_pdf")

csv_file = st.sidebar.file_uploader(
    "Upload SKUs CSV (asin_data_filled.csv)", type=["csv"], key="csv_uploader"
)

if csv_file is None and "uploaded_df" not in st.session_state:
    st.info(
        "Upload a CSV to continue. Expected columns: sku_id/product_id, title, bullet_points/bullets, description, image_url(s), brand, category."
    )
    st.stop()

prime_competitor_chat_state(csv_file)
_render_client_selection_ui()

if st.session_state.get("client_choices") and not st.session_state.get(
    "selected_client"
):
    st.stop()

mode = st.session_state.get("competitor_selection_mode")
if mode not in {"manual", "auto"}:
    _render_competitor_mode_prompt(key_namespace="competitor_mode_main")
    st.stop()

if mode == "auto":
    _render_auto_competitor_selection_ui()
elif mode == "manual":
    _render_competitor_selection_ui()

if st.session_state.get("competitor_choices") and not st.session_state.get(
    "selected_competitor"
):
    st.stop()

validation_graph = get_validation_graph()
with st.spinner("Analyzing the competitor matchup and fetching insights..."):
    result = validation_graph.invoke({"rules_file": rules_file, "sku_file": csv_file})
sku_data = result.get("sku_data")
rule_data = result.get("rule_data")
summary = result.get("validation", {})

if sku_data is None or rule_data is None:
    st.error("Failed to process the uploaded files. Please re-upload and try again.")
    st.stop()

if isinstance(rule_data, RuleExtraction):
    current_rules = rule_data.rules or DEFAULT_RULES
    rules_source = rule_data.source
    rules_notes = rule_data.messages
else:
    current_rules = getattr(rule_data, "rules", DEFAULT_RULES) or DEFAULT_RULES
    rules_source = getattr(rule_data, "source", "Built-in defaults")
    rules_notes = getattr(rule_data, "messages", [])

with st.sidebar.expander("Active rules JSON"):
    st.json(current_rules)


def _display_rule_messages(messages: Any) -> bool:
    seen = set()
    displayed = False
    for item in messages or []:
        level = "info"
        text = ""
        if isinstance(item, tuple) and len(item) >= 2:
            level, text = item[0], item[1]
        elif isinstance(item, dict):
            level = item.get("level", "info")
            text = item.get("text", "")
        else:
            text = str(item)
        text = str(text).strip()
        level = str(level).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        displayed = True
        if level == "success":
            st.sidebar.success(text)
        elif level == "warning":
            st.sidebar.warning(text)
        elif level in {"error", "danger"}:
            st.sidebar.error(text)
        else:
            st.sidebar.info(text)
    return displayed


if not _display_rule_messages(rules_notes):
    st.sidebar.info(f"Using rules from {rules_source}")

client_data = getattr(sku_data, "client", {})
comp_data = getattr(sku_data, "competitor", {})

# Two-column layout for side-by-side comparison
left, right = st.columns(2)
with left:
    st.subheader("Client SKU")
    client_sku_display = client_data.get("sku", "")
    client_sku_original = client_data.get("sku_original")
    if client_sku_original and client_sku_original != client_sku_display:
        client_sku_display = f"{client_sku_display} (original: {client_sku_original})"
    st.write(f"**SKU**: {client_sku_display or '—'}")
    st.write(f"**Brand**: {client_data.get('brand', '')}")
    st.write(f"**Universe**: {client_data.get('universe', '') or '—'}")
    st.write(f"**Title**: {client_data.get('title', '')}")
    st.write("**Bullets**:")
    for b in split_bullets(client_data.get("bullets", "")):
        st.write(f"• {b}")
    st.write("**Description**:")
    st.write(client_data.get("description", ""))
    client_avg_rank = client_data.get("avg_rank_search")
    client_avg_rank_display = (
        "—"
        if client_avg_rank is None or pd.isna(client_avg_rank)
        else str(client_avg_rank)
    )
    st.write(f"**Average Search Rank**: {client_avg_rank_display}")
    client_image_urls = extract_image_urls(client_data.get("image_urls", ""))
    client_image_count = len(client_image_urls)
    st.write("**Images**:")
    st.write(f"Total: {client_image_count}")
    if client_image_urls:
        for idx, url in enumerate(client_image_urls, 1):
            st.markdown(f"{idx}. [{url}]({url})")
    else:
        st.markdown("_No image URLs provided._")

with right:
    st.subheader("Competitor SKU")
    comp_sku_display = comp_data.get("sku", "")
    comp_sku_original = comp_data.get("sku_original")
    if comp_sku_original and comp_sku_original != comp_sku_display:
        comp_sku_display = f"{comp_sku_display} (original: {comp_sku_original})"
    st.write(f"**SKU**: {comp_sku_display or '—'}")
    st.write(f"**Brand**: {comp_data.get('brand', '')}")
    st.write(f"**Universe**: {comp_data.get('universe', '') or '—'}")
    st.write(f"**Title**: {comp_data.get('title', '')}")
    st.write("**Bullets**:")
    for b in split_bullets(comp_data.get("bullets", "")):
        st.write(f"• {b}")
    st.write("**Description**:")
    st.write(comp_data.get("description", ""))
    comp_avg_rank = comp_data.get("avg_rank_search")
    comp_avg_rank_display = (
        "—" if comp_avg_rank is None or pd.isna(comp_avg_rank) else str(comp_avg_rank)
    )
    st.write(f"**Average Search Rank**: {comp_avg_rank_display}")
    comp_image_urls = extract_image_urls(comp_data.get("image_urls", ""))
    comp_image_count = len(comp_image_urls)
    st.write("**Images**:")
    st.write(f"Total: {comp_image_count}")
    if comp_image_urls:
        for idx, url in enumerate(comp_image_urls, 1):
            st.markdown(f"{idx}. [{url}]({url})")
    else:
        st.markdown("_No image URLs provided._")

st.divider()

# Rule checks & summary
rules_for_display = current_rules
title_limit = rules_for_display["title"]["max_chars"]
bullet_limit = rules_for_display["bullets"]["max_count"]
desc_limit = rules_for_display["description"]["max_chars"]


def _format_rules_notes(notes: Any) -> str:
    if not notes:
        return ""
    if isinstance(notes, (list, tuple)):
        filtered = [str(n) for n in notes if n]
        return "; ".join(filtered)
    return str(notes)


formatted_rules_notes = _format_rules_notes(rules_notes)

st.subheader("Insights")
if formatted_rules_notes:
    st.caption(f"Rules source: {rules_source} — {formatted_rules_notes}")
else:
    st.caption(f"Rules source: {rules_source}")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Title score",
        summary["title"]["client_score"],
        help=f"Length ≤ {title_limit}, brand present, no promo, no ALL CAPS",
    )
with col2:
    st.metric(
        "Bullets score",
        summary["bullets"]["client_score"],
        help=f"≤{bullet_limit}, start caps, no end punctuation, no promo",
    )
with col3:
    st.metric(
        "Description score",
        summary["description"]["client_score"],
        help=f"≤{desc_limit} chars, no promo, no ALL CAPS",
    )
with col4:
    st.metric(
        "Images (client vs comp)",
        f"{summary['images']['client_count']} vs {summary['images']['comp_count']}",
    )

issues_gaps_source = {
    "summary": summary,
    "competitor": st.session_state.get("selected_competitor"),
    "client_sku": client_data.get("sku"),
}
issues_gaps_version_payload = json.dumps(
    issues_gaps_source, sort_keys=True, default=str
).encode("utf-8")
issues_gaps_version = hashlib.md5(issues_gaps_version_payload).hexdigest()
stored_issues_gaps_version = st.session_state.get("issues_gaps_rendered_version")

issues_gaps_sections = []
for label, issues in (
    ("Title", summary.get("title", {}).get("issues", [])),
    ("Bullets", summary.get("bullets", {}).get("issues", [])),
    ("Description", summary.get("description", {}).get("issues", [])),
    ("Images", summary.get("images", {}).get("issues", [])),
):
    filtered = [str(i).strip() for i in issues if str(i).strip()]
    if filtered:
        block_lines = [f"**{label}**"]
        block_lines.extend(f"- {issue}" for issue in filtered)
        issues_gaps_sections.append("\n".join(block_lines))

gaps_vs_competitor = [
    str(g).strip() for g in summary.get("gaps_vs_competitor", []) if str(g).strip()
]
if gaps_vs_competitor:
    gap_lines = ["**Gaps vs competitor**"]
    gap_lines.extend(f"- {gap}" for gap in gaps_vs_competitor)
    issues_gaps_sections.append("\n".join(gap_lines))

if not issues_gaps_sections:
    issues_gaps_sections.append(
        "**All clear**\n- No rule violations or competitive gaps detected."
    )

issues_gaps_intro = "I am going to highlight the Issues and gaps:"
issues_gaps_body = "\n\n".join(issues_gaps_sections)
issues_gaps_message = f"{issues_gaps_intro}\n\n{issues_gaps_body}".strip()

chat_history_key = "issues_gaps_chat_history"
stop_key = "issues_gaps_stop"

if issues_gaps_version != stored_issues_gaps_version:
    st.session_state["issues_gaps_rendered_version"] = issues_gaps_version
    st.session_state["issues_gaps_message"] = issues_gaps_message
    st.session_state[chat_history_key] = [
        {"role": "assistant", "content": issues_gaps_message},
        {"role": "assistant", "content": RULE_CHECKS_FOLLOWUP_PROMPT},
    ]
    st.session_state.pop("issues_gaps_decision", None)
    st.session_state.pop("issues_gaps_decision_version", None)
    st.session_state.pop("llm_out", None)
    st.session_state.pop("llm_out_meta", None)
    st.session_state.pop("selected_variant_index", None)
    st.session_state.pop(stop_key, None)
else:
    st.session_state.setdefault("issues_gaps_message", issues_gaps_message)
    history = st.session_state.setdefault(chat_history_key, [])
    if history:
        history[0] = {"role": "assistant", "content": issues_gaps_message}
    else:
        history.append({"role": "assistant", "content": issues_gaps_message})

    if not any(
        entry.get("role") == "assistant"
        and entry.get("content") == RULE_CHECKS_FOLLOWUP_PROMPT
        for entry in history
    ):
        insert_at = 1 if history else 0
        history.insert(
            insert_at,
            {"role": "assistant", "content": RULE_CHECKS_FOLLOWUP_PROMPT},
        )

chat_history = st.session_state.get(chat_history_key, [])
for entry in chat_history:
    role = entry.get("role", "assistant")
    content = entry.get("content", "")
    if not content:
        continue
    with st.chat_message(role):
        st.markdown(content)

user_reply = st.chat_input("How would you like to continue with these findings?")
if user_reply:
    user_reply = user_reply.strip()
    if user_reply:
        chat_history.append({"role": "user", "content": user_reply})
        with st.chat_message("user"):
            st.markdown(user_reply)

        client = get_openai_client()
        with st.spinner(
            "Reviewing your reply with Ally's policy brain to pick the best next step..."
        ):
            action = classify_review_followup(
                st.session_state.get("issues_gaps_message", issues_gaps_message),
                user_reply,
                client=client,
                additional_products=st.session_state.get(
                    "orchestrator_additional_products"
                ),
            )
        st.session_state["issues_gaps_last_action"] = action

        if action == "generate_edits":
            acknowledgement = (
                "Copy that! I'm spinning up some rule-hugging sparkle for these edits."
            )
            chat_history.append({"role": "assistant", "content": acknowledgement})
            with st.chat_message("assistant"):
                st.markdown(acknowledgement)

            user_context_entries = []
            for item in chat_history:
                if item.get("role") != "user":
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                normalized = re.sub(r"\s+", " ", content)
                if normalized:
                    user_context_entries.append(normalized)
            st.session_state["issues_gaps_user_context"] = user_context_entries
            user_context_str = "\n".join(user_context_entries) if user_context_entries else None

            with st.spinner(
                "Consulting OpenAI for brand-safe edits and preparing fallbacks..."
            ):
                llm_result = call_llm(
                    client_data,
                    comp_data,
                    current_rules,
                    user_context=user_context_str,
                )

            variants = []
            metadata: Dict[str, Any] = {}
            if isinstance(llm_result, dict):
                variants = [
                    v for v in llm_result.get("variants", []) if isinstance(v, dict)
                ]
                metadata = {k: v for k, v in llm_result.items() if k != "variants"}

            st.session_state["llm_out"] = variants
            st.session_state["llm_out_meta"] = metadata
            st.session_state["selected_variant_index"] = (
                0 if not metadata.get("_llm") else None
            )
            st.session_state.pop(stop_key, None)

            if metadata.get("_llm"):
                followup = (
                    "Ta-da! Fresh from Ally's idea oven (with a dash of OpenAI magic) — check out these draft edits."
                )
            else:
                if not st.session_state.get("openai_valid"):
                    followup = (
                        "Ta-da! I whipped these up with Ally's built-in heuristics since we don't have a valid OpenAI key right now."
                    )
                else:
                    followup = (
                        "Ta-da! OpenAI took a rain check, so I jazzed these up with Ally's heuristic toolkit instead."
                    )
            chat_history.append({"role": "assistant", "content": followup})
            with st.chat_message("assistant"):
                st.markdown(followup)

        elif action == "answer_question":
            with st.spinner(
                "Gathering a grounded answer from the context and OpenAI (with heuristics on standby)..."
            ):
                answer_text, chunks = answer_review_question(
                    user_reply,
                    issues_summary=st.session_state.get(
                        "issues_gaps_message", issues_gaps_message
                    ),
                    current_rules=current_rules,
                    client_data=client_data,
                    competitor_data=comp_data,
                    client=client,
                )
            chat_history.append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                writer = getattr(st, "write_stream", None)
                if callable(writer):

                    def _stream() -> Iterable[str]:
                        for chunk in chunks:
                            yield chunk

                    writer(_stream())
                else:
                    st.markdown(answer_text)

        elif action == "find_competitors":
            _clear_competitor_selection_state(keep_choices=True)
            st.session_state.pop("issues_gaps_found_competitors", None)
            st.session_state.pop("orchestrator_additional_products", None)

            message = (
                "Let's revisit the competitor selection. Choose how to pick the competitor SKU "
                "using the options above."
            )
            chat_history.append({"role": "assistant", "content": message})
            with st.chat_message("assistant"):
                st.markdown(message)

            _trigger_rerun()

        elif action == "stop":
            st.session_state[stop_key] = True
            note = (
                "Understood — I'll pause here. Let me know if you need anything else."
            )
            chat_history.append({"role": "assistant", "content": note})
            with st.chat_message("assistant"):
                st.markdown(note)

        else:  # clarify
            clarification = (
                "Happy to help — let me know what you’d like me to clarify or adjust."
            )
            chat_history.append({"role": "assistant", "content": clarification})
            with st.chat_message("assistant"):
                st.markdown(clarification)

st.divider()

if "llm_out" in st.session_state:
    variants: List[Dict[str, Any]] = st.session_state.get("llm_out", [])
    metadata: Dict[str, Any] = st.session_state.get("llm_out_meta", {})
    rules = current_rules
    title_max = rules["title"]["max_chars"]
    bullet_max = rules["bullets"]["max_count"]
    desc_max = rules["description"]["max_chars"]

    st.subheader("Drafted compliant edits")
    _mode = (
        "OpenAI (validated)"
        if metadata.get("_llm") and st.session_state.get("openai_valid")
        else (
            "OpenAI (unvalidated)"
            if (
                metadata.get("_llm")
                and (
                    st.session_state.get("OPENAI_API_KEY_UI")
                    or os.getenv("OPENAI_API_KEY")
                )
            )
            else "Heuristic fallback (no key)"
        )
    )
    st.caption(f"Mode: {_mode}")

    if not variants:
        st.warning(
            "No draft variants are available right now. Try requesting new edits to regenerate suggestions."
        )
    else:
        if metadata.get("_llm"):
            valid_indexes = set(range(len(variants)))
            for key in list(st.session_state.keys()):
                if not key.startswith("variant_accept_"):
                    continue
                try:
                    idx = int(key.rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                if idx not in valid_indexes:
                    st.session_state.pop(key, None)

            selected_index = st.session_state.get("selected_variant_index")
            if not isinstance(selected_index, int) or not (0 <= selected_index < len(variants)):
                selected_index = None
                st.session_state["selected_variant_index"] = None

            for idx, variant in enumerate(variants):
                st.markdown(f"### Variant {idx + 1}")
                st.markdown("**Proposed Title**")
                st.code(
                    limit_text_with_sentence_guard(
                        str(variant.get("title_edit", "")),
                        title_max,
                        prefer_sentence=False,
                    )
                )

                st.markdown(f"**Proposed Bullets (up to {bullet_max})**")
                for b in variant.get("bullets_edits", [])[:bullet_max]:
                    bullet_text = str(b)
                    st.write(f"• {re.sub(r'[.!?]+$', '', bullet_text).strip()}")

                st.markdown(f"**Proposed Description (<= {desc_max} chars)**")
                st.code(
                    limit_text_with_sentence_guard(
                        str(variant.get("description_edit", "")),
                        desc_max,
                        prefer_sentence=True,
                    )
                )

                rationales = variant.get("rationales", [])
                if rationales:
                    with st.expander("Why these edits?", expanded=False):
                        for r in rationales:
                            st.write("- ", r)

                checkbox_key = f"variant_accept_{idx}"
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = selected_index == idx
                st.checkbox(
                    "accept this version",
                    key=checkbox_key,
                    on_change=_handle_variant_checkbox_selection,
                    args=(idx,),
                )

            st.caption(f"Source: {rules_source}")

            final_index = st.session_state.get("selected_variant_index")
            if (
                isinstance(final_index, int)
                and 0 <= final_index < len(variants)
                and st.session_state.get(f"variant_accept_{final_index}")
            ):
                selected_variant = variants[final_index]
                final_md_str, email_md = _build_final_outputs(
                    selected_variant,
                    client_data,
                    rules,
                    title_max=title_max,
                    bullet_max=bullet_max,
                    desc_max=desc_max,
                )

                st.markdown("---")
                st.subheader("Final Markdown")
                st.code(final_md_str, language="markdown")
                st.download_button(
                    "Download final.md", final_md_str.encode("utf-8"), file_name="final.md"
                )

                with st.expander("Email draft to stakeholders"):
                    st.code(email_md)
                    st.download_button(
                        "Download email.txt", email_md.encode("utf-8"), file_name="email.txt"
                    )
            else:
                st.info("Select a variant to accept and unlock the final downloads.")

        else:
            selected_variant = variants[0]
            st.markdown("**Proposed Title**")
            st.code(
                limit_text_with_sentence_guard(
                    str(selected_variant.get("title_edit", "")),
                    title_max,
                    prefer_sentence=False,
                )
            )

            st.markdown(f"**Proposed Bullets (up to {bullet_max})**")
            for b in selected_variant.get("bullets_edits", [])[:bullet_max]:
                st.write(f"• {re.sub(r'[.!?]+$', '', str(b)).strip()}")

            st.markdown(f"**Proposed Description (<= {desc_max} chars)**")
            st.code(
                limit_text_with_sentence_guard(
                    str(selected_variant.get("description_edit", "")),
                    desc_max,
                    prefer_sentence=True,
                )
            )

            rationales = selected_variant.get("rationales", [])
            if rationales:
                with st.expander("Why these edits?", expanded=False):
                    for r in rationales:
                        st.write("- ", r)

            st.caption(f"Source: {rules_source}")

            approved = st.checkbox("I approve these edits")
            if approved:
                final_md_str, email_md = _build_final_outputs(
                    selected_variant,
                    client_data,
                    rules,
                    title_max=title_max,
                    bullet_max=bullet_max,
                    desc_max=desc_max,
                )

                st.markdown("---")
                st.subheader("Final Markdown")
                st.code(final_md_str, language="markdown")
                st.download_button(
                    "Download final.md", final_md_str.encode("utf-8"), file_name="final.md"
                )

                with st.expander("Email draft to stakeholders"):
                    st.code(email_md)
                    st.download_button(
                        "Download email.txt", email_md.encode("utf-8"), file_name="email.txt"
                    )
else:
    st.info(
        "Use the chat below to ask for edits, generate suggested edits, choose another competitor, or wrap up the review."
    )
