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
import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# Optional OpenAI SDK (gracefully handle if not installed or no key)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from chains.rule_extractor import RuleExtraction
from chains.sku_extractor import SKUData
from core.content_rules import (
    DEFAULT_RULES,
    compare_fields,
    count_images,
    enforce_title_caps,
    split_bullets,
)
from graph.product_validation import build_product_validation_graph


# ---------------------------
# LLM prompt & call
# ---------------------------

SYSTEM_PROMPT = (
    "You are a meticulous Amazon PDP content editor for Pet Supplies. "
    "Follow the provided style rules strictly (title<=50 chars; up to 5 bullets; description<=200 chars; "
    "no promotional or seller info; start bullets with a capital letter; avoid ending punctuation; keep sentences clear). "
    "Return compliant copy and explain briefly how each edit improves against the competitor."
)

USER_PROMPT_TEMPLATE = (
    "CLIENT SKU (brand={brand}):\n"
    "- Title: {c_title}\n"
    "- Bullets: {c_bullets}\n"
    "- Description: {c_desc}\n"
    "\n"
    "COMPETITOR SKU (brand={comp_brand}):\n"
    "- Title: {k_title}\n"
    "- Bullets: {k_bullets}\n"
    "- Description: {k_desc}\n"
    "\n"
    "Rules summary:\n"
    "- Title: <=50 chars, include brand, no ALL CAPS, no promo, '(pack of X)' only for bundles.\n"
    "- Bullets: up to 5; start with capital; sentence fragments; no ending punctuation; no promo/seller info.\n"
    "- Description: <=200 chars; no promo or seller info; truthful claims; avoid ALL CAPS.\n"
    "\n"
    "TASK: Propose an improved TITLE, 3-5 BULLETS, and a short DESCRIPTION for the CLIENT that are compliant.\n"
    "Also provide a brief rationale for each change that references what the competitor does.\n"
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


def call_llm(
    client_data: Dict[str, Any],
    comp_data: Dict[str, Any],
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rules = rules or DEFAULT_RULES
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
    )

    if client is None:
        # Heuristic fallback: create simple suggestions without LLM
        bullets = split_bullets(client_data.get("bullets", ""))
        base_brand = (client_data.get("brand") or "").strip()
        # Title heuristic: ensure brand at start, trim to 50 chars
        raw_title = client_data.get("title") or ""
        if base_brand and base_brand.lower() not in raw_title.lower():
            title = f"{base_brand} " + raw_title
        else:
            title = raw_title
        title = enforce_title_caps(title)[: rules["title"]["max_chars"]].strip()
        # Bullets heuristic: take up to 5, strip punctuation at end, capitalize first letter
        fixed_bullets: List[str] = []
        for b in bullets[:5] or ["Durable construction", "Comfortable fit", "Easy to clean"]:
            b = b.strip()
            b = re.sub(r"[.!?]+$", "", b)
            if b and b[0].isalpha():
                b = b[0].upper() + b[1:]
            fixed_bullets.append(b)
        # Description heuristic
        desc = (client_data.get("description") or "").strip()
        if not desc:
            desc = "Compact design for everyday use; easy to clean; ideal for most pets"
        desc = desc[: rules["description"]["max_chars"]]
        return {
            "title_edit": title,
            "bullets_edits": fixed_bullets,
            "description_edit": desc,
            "rationales": [
                "Ensure brand appears in title and stays within 50 characters",
                "Use 3–5 concise bullets starting with a capital letter, no ending punctuation",
                "Short description (<=200 chars) with clear benefit; remove promo language if present",
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
            "description_edit": "Compact design for everyday use; easy to clean; ideal for most pets"[
                : rules["description"]["max_chars"]
            ],
            "rationales": ["LLM error; generated heuristic placeholders"],
            "_llm": False,
        }


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Competitor Content Intelligence", layout="wide")

# --- High-contrast fix for selectbox in dark mode ---
st.markdown(
    """
    <style>
    /* Selected value text inside the selectbox control */
    .stSelectbox [data-baseweb="select"] > div { color: #ffffff !important; }
    .stSelectbox [data-baseweb="select"] span { color: #ffffff !important; }

    /* The dropdown menu portal */
    [data-baseweb="menu"] { background-color: #ffffff !important; color: #111111 !important; }
    [data-baseweb="menu"] [role="option"] { color: #111111 !important; }
    [data-baseweb="menu"] [aria-selected="true"] { background-color: #f3f4f6 !important; }

    /* Placeholder */
    .stSelectbox [data-baseweb="select"] [aria-hidden="true"] { color: #e5e7eb !important; }
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
csv_file = st.sidebar.file_uploader(
    "Upload SKUs CSV (asin_data_filled.csv)", type=["csv"], key="csv_uploader"
)
rules_pdf = st.sidebar.file_uploader(
    "Upload style guide PDF (optional)", type=["pdf"], key="rules_uploader"
)


def _run_validation_graph(csv, pdf):
    graph = build_product_validation_graph(
        lambda sku_data, rule_data: compare_fields(
            sku_data.client,
            sku_data.competitor,
            rules=rule_data.rules,
            available_universes=sku_data.available_universes,
        )
    )
    state_inputs = {"sku_file": csv, "rules_file": pdf}
    return graph.invoke(state_inputs)


graph_state = _run_validation_graph(csv_file, rules_pdf)
sku_result: SKUData = graph_state["sku_data"]
rule_result: RuleExtraction = graph_state["rule_data"]
summary = graph_state["validation"]

client_data = sku_result.client
comp_data = sku_result.competitor
active_rules = rule_result.rules if rule_result and rule_result.rules else DEFAULT_RULES

# Two-column layout for side-by-side comparison
left, right = st.columns(2)
with left:
    st.subheader("Client SKU")
    client_sku_display = client_data["sku"]
    if client_data["sku_original"] and client_data["sku_original"] != client_data["sku"]:
        client_sku_display = f"{client_sku_display} (original: {client_data['sku_original']})"
    st.write(f"**SKU**: {client_sku_display}")
    st.write(f"**Brand**: {client_data.get('brand','')}")
    st.write(f"**Universe**: {client_data.get('universe', '') or '—'}")
    st.write(f"**Title**: {client_data.get('title','')}")
    st.write("**Bullets**:")
    for b in split_bullets(client_data.get("bullets", "")):
        st.write(f"• {b}")
    st.write("**Description**:")
    st.write(client_data.get("description", ""))
    st.write("**Images (count)**:")
    st.write(count_images(client_data.get("image_urls", "")))

with right:
    st.subheader("Competitor SKU")
    comp_sku_display = comp_data["sku"]
    if comp_data["sku_original"] and comp_data["sku_original"] != comp_data["sku"]:
        comp_sku_display = f"{comp_sku_display} (original: {comp_data['sku_original']})"
    st.write(f"**SKU**: {comp_sku_display}")
    st.write(f"**Brand**: {comp_data.get('brand','')}")
    st.write(f"**Universe**: {comp_data.get('universe', '') or '—'}")
    st.write(f"**Title**: {comp_data.get('title','')}")
    st.write("**Bullets**:")
    for b in split_bullets(comp_data.get("bullets", "")):
        st.write(f"• {b}")
    st.write("**Description**:")
    st.write(comp_data.get("description", ""))
    st.write("**Images (count)**:")
    st.write(count_images(comp_data.get("image_urls", "")))

st.divider()

st.subheader("Rule checks (Client)")
if rule_result.notes:
    st.caption(f"Rules source: {rule_result.source} — {rule_result.notes}")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Title score", summary["title"]["client_score"], help="Length, brand present, no promo, no ALL CAPS")
with col2:
    st.metric("Bullets score", summary["bullets"]["client_score"], help="<=5, start caps, no end punctuation, no promo")
with col3:
    st.metric("Description score", summary["description"]["client_score"], help="<=200 chars, no promo, no ALL CAPS")
with col4:
    st.metric("Images (client vs comp)", f"{summary['images']['client_count']} vs {summary['images']['comp_count']}")
with col5:
    _u = summary.get('universe', {})
    _client_uni = _u.get('client_provided') or '—'
    _suggested_uni = _u.get('suggested') or _client_uni
    _comp_uni = _u.get('competitor_provided') or '—'
    _inferred_uni = _u.get('inferred') or '—'
    st.metric("Universe", f"{_client_uni} → {_suggested_uni}", help=f"Competitor: {_comp_uni}; Inferred: {_inferred_uni}")

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
    if summary.get("universe", {}).get("issues"):
        st.markdown("**Universe**")
        for i in summary["universe"]["issues"]:
            st.write("- ", i)

st.divider()

st.subheader("Generate suggested edits")

# LLM mode indicator
_mode = "OpenAI (validated)" if st.session_state.get("openai_valid") else ("OpenAI (unvalidated)" if (st.session_state.get("OPENAI_API_KEY_UI") or os.getenv("OPENAI_API_KEY")) else "Heuristic fallback (no key)")
st.caption(f"Mode: {_mode}")
if st.button("Draft compliant edits with LLM / heuristic"):
    with st.spinner("Generating suggestions..."):
        llm_out = call_llm(client_data, comp_data, active_rules)
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
    st.markdown("**Proposed Title**")
    st.code(out.get("title_edit", ""))

    st.markdown("**Proposed Bullets (3–5)**")
    for b in out.get("bullets_edits", [])[:5]:
        st.write(f"• {re.sub(r'[.!?]+$', '', b).strip()}")

    st.markdown("**Proposed Description (<=200 chars)**")
    st.code((out.get("description_edit", ""))[: active_rules["description"]["max_chars"]])

    if out.get("rationales"):
        with st.expander("Why these edits?"):
            for r in out["rationales"]:
                st.write("- ", r)

    st.caption("Source: Amazon Pet Supplies style guide rules encoded in app")

    approved = st.checkbox("I approve these edits and confirm they follow brand & Amazon rules")
    if approved:
        # Final Markdown summary
        final_md = []
        final_md.append(f"# Final Content — Client SKU {client_data['sku']}")
        final_md.append("\n## Title (proposed)\n")
        final_md.append(out.get("title_edit", ""))
        final_md.append("\n## Bullets (proposed)\n")
        for b in out.get("bullets_edits", [])[:5]:
            final_md.append(f"- {re.sub(r'[.!?]+$','', b).strip()}")
        final_md.append("\n\n## Description (proposed)\n")
        final_md.append((out.get("description_edit", ""))[: active_rules["description"]["max_chars"]])

        uni = summary.get("universe", {})
        if uni.get("suggested"):
            final_md.append("\n\n## Universe suggestion\n")
            final_md.append(
                f"Suggested universe: **{uni['suggested']}** "
                f"(client: {uni.get('client_provided') or '—'}, "
                f"competitor: {uni.get('competitor_provided') or '—'})\n"
                f"Reason: {uni.get('reason','')}"
            )

        final_md.append("\n\n## Rationale & Rule Compliance\n")
        final_md.append(
            "- Title ≤ 50 chars, includes brand, avoids ALL CAPS & promo\n"
            "- Up to 5 bullets; capitalized starts; no ending punctuation; no promo/seller info\n"
            "- Description ≤ 200 chars; plain language; no promo/seller info"
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

Please find the approved PDP content updates for SKU {client_data['sku']} below. These adhere to the Pet Supplies style guide (title≤50, ≤5 bullets, description≤200; no promo/seller info).

Title
-----
{out.get('title_edit','')}

Bullets
-------
{chr(10).join([f"- {re.sub(r'[.!?]+$','', b).strip()}" for b in out.get('bullets_edits', [])[:5]])}

Description
----------
{(out.get('description_edit',''))[: active_rules['description']['max_chars']]}

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
