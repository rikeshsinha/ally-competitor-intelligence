"""
Streamlit app: Competitor Content Intelligence (Ally skill demo)

What it does
------------
- Load a CSV of SKUs (client + competitors)
- Let you pick a client SKU and a competitor SKU
- Run rule checks based on the provided Amazon Pet Supplies style guide
- Compare PDP fields (title, bullets, description, images)
- Ask an LLM to draft brand- & Amazon‑compliant edits for the client SKU (title/bullets/description)
- Pause for human approval
- On approval, output a Final Markdown summary and optional email draft and allow download

How to run
----------
1) Install deps:  
   pip install streamlit pandas openai tiktoken python-dotenv

2) Put your files in the same folder as this script:
   - asin_data_filled.csv
   - PetSupplies_PetFood_Styleguide_EN_AE._CB1198675309_.pdf (for reference only; rules are encoded below)

3) Set your OpenAI key (or provider of choice):  
   export OPENAI_API_KEY=sk-...  

4) Start the app:  
   streamlit run app.py

Notes
-----
- If no API key is set, the app will generate heuristic (non‑LLM) suggestions so you can still demo the flow.
- CSV column names are auto‑detected; expected columns include: sku_id, title, bullets, description, image_urls, brand, category.
- Bullets can be delimited by newlines, "|", "•", or semicolons; the app tries to auto‑split.
"""

from __future__ import annotations
import math
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

from core.content_rules import (
    DEFAULT_RULES,
    compare_fields,
    enforce_title_caps,
    extract_image_urls,
    split_bullets,
)
from graph.product_validation import build_product_validation_graph
from chains.rule_extractor import RuleExtraction

# Optional OpenAI SDK (gracefully handle if not installed or no key)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ---------------------------
# LLM prompt & call
# ---------------------------

SYSTEM_PROMPT = (
    "You are a meticulous Amazon PDP content editor for Pet Supplies. "
    "Follow the provided style rules strictly. Return compliant copy and explain briefly how each edit improves "
    "against the competitor."
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
    "Return JSON with keys: title_edit, bullets_edits (list), description_edit, rationales (list of strings)."
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


def get_validation_graph():
    if "product_validation_graph" not in st.session_state:
        def _run_validation(sku_data, rule_data: RuleExtraction):
            rules = rule_data.rules or DEFAULT_RULES
            client = getattr(sku_data, "client", {})
            competitor = getattr(sku_data, "competitor", {})
            return compare_fields(client, competitor, rules=rules)

        st.session_state["product_validation_graph"] = build_product_validation_graph(
            _run_validation
        )
    return st.session_state["product_validation_graph"]


def call_llm(
    client_data: Dict[str, Any],
    comp_data: Dict[str, Any],
    rules: Dict[str, Any],
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
                if (
                    normalized_clause
                    and normalized_clause not in normalized_bullets
                ):
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
        title = enforce_title_caps(title)[: rules["title"]["max_chars"]].strip()

        # Description heuristic: keep client description if present, otherwise reuse bullet text
        desc = (client_data.get("description") or "").strip()
        if not desc and fixed_bullets:
            desc = " ".join(fixed_bullets)
        desc = re.sub(r"\s+", " ", desc)[: rules["description"]["max_chars"]].strip()
        return {
            "title_edit": title,
            "bullets_edits": fixed_bullets,
            "description_edit": desc,
            "rationales": [
                f"Ensure brand appears in title and stays within {rules['title']['max_chars']} characters",
                f"Use up to {rules['bullets']['max_count']} concise bullets starting with a capital letter, no ending punctuation",
                f"Short description (<= {rules['description']['max_chars']} chars) with clear benefit; remove promo language if present",
            ],
            "_llm": False,
        }

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
        data["_llm"] = True
        return data
    except Exception as e:
        # Fallback to heuristic
        return {
            "title_edit": enforce_title_caps((client_data.get("title") or "")[: rules["title"]["max_chars"]]),
            "bullets_edits": ["Durable construction", "Comfortable fit", "Easy to clean"],
            "description_edit": "Compact design for everyday use; easy to clean; ideal for most pets"[: rules["description"]["max_chars"]],
            "rationales": ["LLM error; generated heuristic placeholders"],
            "_llm": False,
        }


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

st.title("Competitor Content Intelligence — Ally skill demo")

with st.expander("About this demo"):
    st.markdown(
        "This app compares a client SKU to a competitor and drafts compliant edits (title, bullets, description).\n"
        "Rules encoded from the Amazon Pet Supplies style guide for UAE (title≤50, ≤5 bullets, description≤200, no promo)."
    )

# Sidebar: LLM settings
st.sidebar.header("LLM Settings")
_ui_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get("OPENAI_API_KEY_UI", ""))
if _ui_key != st.session_state.get("OPENAI_API_KEY_UI"):
    st.session_state["OPENAI_API_KEY_UI"] = _ui_key

col_a, col_b = st.sidebar.columns([1,1])
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
    st.sidebar.markdown('<div style="display:inline-block;padding:4px 10px;border-radius:999px;background:#16a34a;color:#fff;font-weight:600">OpenAI: Valid</div>', unsafe_allow_html=True)
elif (_ui_key or os.getenv("OPENAI_API_KEY")) and valid_flag is False:
    st.sidebar.markdown('<div style="display:inline-block;padding:4px 10px;border-radius:999px;background:#dc2626;color:#fff;font-weight:600">OpenAI: Invalid</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div style="display:inline-block;padding:4px 10px;border-radius:999px;background:#6b7280;color:#fff;font-weight:600">OpenAI: Not set</div>', unsafe_allow_html=True)

# Sidebar: file inputs
st.sidebar.header("Inputs")
rules_file = st.sidebar.file_uploader("Upload Rules PDF", type=["pdf"], key="rules_pdf")

csv_file = st.sidebar.file_uploader("Upload SKUs CSV (asin_data_filled.csv)", type=["csv"], key="csv_uploader")

if csv_file is None and "uploaded_df" not in st.session_state:
    st.info("Upload a CSV to continue. Expected columns: sku_id/product_id, title, bullet_points/bullets, description, image_url(s), brand, category.")
    st.stop()

validation_graph = get_validation_graph()
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


def _format_avg_rank(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or "—"
    if math.isnan(numeric) or math.isinf(numeric):
        return "—"
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    formatted = f"{numeric:.2f}".rstrip("0").rstrip(".")
    return formatted or f"{numeric:.2f}"


def _to_display_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(value).strip()
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"", "nan", "na", "none", "null"}:
        return ""
    return text


if not _display_rule_messages(rules_notes):
    st.sidebar.info(f"Using rules from {rules_source}")

client_data = getattr(sku_data, "client", {})
comp_data = getattr(sku_data, "competitor", {})
similar_candidates = getattr(sku_data, "similar_candidates", [])
selection_metadata = getattr(sku_data, "selection_metadata", {})

if similar_candidates:
    with st.sidebar.expander("Similar competitor suggestions"):
        for idx, candidate in enumerate(similar_candidates, 1):
            label = candidate.get("title") or candidate.get("sku") or f"Candidate {idx}"
            brand = candidate.get("brand") or "Unknown"
            rank_text = _format_avg_rank(candidate.get("avg_rank_search"))
            st.write(f"{idx}. {label} — {brand} (avg rank: {rank_text})")
            reason = candidate.get("reason")
            if reason:
                st.caption(reason)

selection_source = selection_metadata.get("source") if isinstance(selection_metadata, dict) else None
if selection_source == "recommended":
    st.sidebar.success("Competitor selection: Recommended similar match")
elif selection_source == "manual" and similar_candidates:
    st.sidebar.caption("Competitor selection: Manual choice")

# Two-column layout for side-by-side comparison
left, right = st.columns(2)
with left:
    st.subheader("Client SKU")
    client_sku_display = client_data.get("sku", "")
    client_sku_original = client_data.get("sku_original")
    if client_sku_original and client_sku_original != client_sku_display:
        client_sku_display = f"{client_sku_display} (original: {client_sku_original})"
    st.write(f"**SKU**: {client_sku_display or '—'}")
    st.write(f"**Brand**: {client_data.get('brand','')}")
    st.write(
        f"**Avg Search Rank**: {_format_avg_rank(client_data.get('avg_rank_search'))}"
    )
    client_bundle = _to_display_text(client_data.get("bundle_size"))
    if client_bundle:
        st.write(f"**Bundle Size**: {client_bundle}")
    st.write(f"**Universe**: {client_data.get('universe', '') or '—'}")
    st.write(f"**Title**: {client_data.get('title','')}")
    st.write("**Bullets**:")
    for b in split_bullets(client_data.get("bullets", "")):
        st.write(f"• {b}")
    st.write("**Description**:")
    st.write(client_data.get("description", ""))
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
    st.write(f"**Brand**: {comp_data.get('brand','')}")
    st.write(
        f"**Avg Search Rank**: {_format_avg_rank(comp_data.get('avg_rank_search'))}"
    )
    comp_bundle = _to_display_text(comp_data.get("bundle_size"))
    if comp_bundle:
        st.write(f"**Bundle Size**: {comp_bundle}")
    st.write(f"**Universe**: {comp_data.get('universe', '') or '—'}")
    st.write(f"**Title**: {comp_data.get('title','')}")
    if comp_data.get("selection_source") == "recommended":
        reason = _to_display_text(comp_data.get("similarity_reason"))
        caption_text = "Recommended competitor selection"
        if reason:
            caption_text = f"{caption_text}: {reason}"
        st.caption(caption_text)
    st.write("**Bullets**:")
    for b in split_bullets(comp_data.get("bullets", "")):
        st.write(f"• {b}")
    st.write("**Description**:")
    st.write(comp_data.get("description", ""))
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

st.subheader("Rule checks (Client)")
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
    st.metric("Images (client vs comp)", f"{summary['images']['client_count']} vs {summary['images']['comp_count']}")

with st.expander("Issues & Gaps"):
    if summary["title"]["issues"]:
        st.markdown("**Title**")
        for i in summary["title"]["issues"]:
            st.write("- ", i)
    if summary["bullets"]["issues"]:
        st.markdown("**Bullets**")
        for i in summary["bullets"]["issues"]:
            st.write("- ", i)
    if summary["description"]["issues"]:
        st.markdown("**Description**")
        for i in summary["description"]["issues"]:
            st.write("- ", i)
    if summary["images"]["issues"]:
        st.markdown("**Images**")
        for i in summary["images"]["issues"]:
            st.write("- ", i)
    if summary.get("gaps_vs_competitor"):
        st.markdown("**Gaps vs competitor**")
        for g in summary["gaps_vs_competitor"]:
            st.write("- ", g)

st.divider()

st.subheader("Generate suggested edits")

# LLM mode indicator
_mode = "OpenAI (validated)" if st.session_state.get("openai_valid") else ("OpenAI (unvalidated)" if (st.session_state.get("OPENAI_API_KEY_UI") or os.getenv("OPENAI_API_KEY")) else "Heuristic fallback (no key)")
st.caption(f"Mode: {_mode}")
if st.button("Draft compliant edits with LLM / heuristic"):
    with st.spinner("Generating suggestions..."):
        llm_out = call_llm(client_data, comp_data, current_rules)
    st.session_state["llm_out"] = llm_out
    if llm_out.get("_llm"):
        st.success("Used OpenAI LLM")
    else:
        if not st.session_state.get("openai_valid"):
            st.warning("No/invalid OpenAI key → used heuristic fallback")
        else:
            st.warning("LLM call failed → used heuristic fallback")

if "llm_out" in st.session_state:
    out = st.session_state["llm_out"]
    rules = current_rules
    title_max = rules["title"]["max_chars"]
    bullet_max = rules["bullets"]["max_count"]
    desc_max = rules["description"]["max_chars"]
    st.markdown("**Proposed Title**")
    st.code(out.get("title_edit", ""))

    st.markdown(f"**Proposed Bullets (up to {bullet_max})**")
    for b in out.get("bullets_edits", [])[: bullet_max]:
        st.write(f"• {re.sub(r'[.!?]+$', '', b).strip()}")

    st.markdown(f"**Proposed Description (<= {desc_max} chars)**")
    st.code((out.get("description_edit", ""))[: desc_max])

    if out.get("rationales"):
        with st.expander("Why these edits?"):
            for r in out["rationales"]:
                st.write("- ", r)

    st.caption(f"Source: {rules_source}")

    approved = st.checkbox("I approve these edits and confirm they follow brand & Amazon rules")
    if approved:
        # Final Markdown summary
        final_md = []
        final_md.append(f"# Final Content — Client SKU {client_data['sku']}")
        final_md.append("\n## Title (proposed)\n")
        final_md.append(out.get("title_edit", ""))
        final_md.append("\n## Bullets (proposed)\n")
        for b in out.get("bullets_edits", [])[: bullet_max]:
            final_md.append(f"- {re.sub(r'[.!?]+$','', b).strip()}")
        final_md.append("\n\n## Description (proposed)\n")
        final_md.append((out.get("description_edit", ""))[: desc_max])

        final_md.append("\n\n## Rationale & Rule Compliance\n")
        final_md.append(
            f"- Title ≤ {title_max} chars; "
            f"{'brand required' if rules['title']['brand_required'] else 'brand optional'}; "
            f"{'avoid ALL CAPS' if rules['title']['no_all_caps'] else 'ALL CAPS allowed'}; "
            f"{'no promo language' if rules['title']['no_promo'] else 'promo allowed'}"
        )
        final_md.append(
            f"- Up to {bullet_max} bullets; {'start with capitals' if rules['bullets']['start_capital'] else 'any case allowed'}; "
            f"{'no ending punctuation' if rules['bullets']['no_end_punct'] else 'ending punctuation allowed'}; "
            f"{'no promo/seller info' if rules['bullets']['no_promo_or_seller_info'] else 'promo allowed'}"
        )
        final_md.append(
            f"- Description ≤ {desc_max} chars; "
            f"{'no promo language' if rules['description']['no_promo'] else 'promo allowed'}; "
            f"{'avoid ALL CAPS' if rules['description']['sentence_caps'] else 'ALL CAPS allowed'}"
        )
        for r in out.get("rationales", []):
            final_md.append(f"- {r}")

        final_md_str = "\n".join(final_md).strip()
        st.markdown("---")
        st.subheader("Final Markdown")
        st.code(final_md_str, language="markdown")
        st.download_button("Download final.md", final_md_str.encode("utf-8"), file_name="final.md")

        with st.expander("Email draft to stakeholders"):
            email_md = f"""
Subject: Approved PDP Edits for SKU {client_data['sku']}

Hi team,

Please find the approved PDP content updates for SKU {client_data['sku']} below. These adhere to the style guide (title≤{title_max}, ≤{bullet_max} bullets, description≤{desc_max}; no promo/seller info when restricted).

Title
-----
{out.get('title_edit','')}

Bullets
-------
{chr(10).join([f"- {re.sub(r'[.!?]+$','', b).strip()}" for b in out.get('bullets_edits', [])[: bullet_max]])}

Description
----------
{(out.get('description_edit',''))[: desc_max]}

Rationale
---------
- Alignment with style rules; improved specificity vs competitor
{chr(10).join([f"- {r}" for r in out.get('rationales', [])])}

Thanks,
Ally (Competitor Content Intelligence)
""".strip()
            st.code(email_md)
            st.download_button("Download email.txt", email_md.encode("utf-8"), file_name="email.txt")

else:
    st.info("Click the button above to generate suggested edits.")
