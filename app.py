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

import pandas as pd
import streamlit as st

# Optional OpenAI SDK (gracefully handle if not installed or no key)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ---------------------------
# Constants & Rulebook (from PDF)
# ---------------------------
RULES: Dict[str, Any] = {
    "title": {
        "max_chars": 50,  # Pet Supplies guide (UAE)
        "brand_required": True,
        "capitalize_words": True,  # First letter of each word; exceptions not strictly enforced
        "no_all_caps": True,
        "no_promo": True,  # no promo text (sale/free ship)
        "allow_pack_of": True,  # (pack of X) only for bundles
    },
    "bullets": {
        "max_count": 5,
        "start_capital": True,
        "sentence_fragments": True,
        "no_end_punct": True,  # avoid ending punctuation
        "no_promo_or_seller_info": True,
        "numbers_as_numerals": True,
        "semicolons_ok": True,
    },
    "description": {
        "max_chars": 200,
        "no_promo": True,
        "no_seller_info": True,
        "sentence_caps": True,  # avoid ALL CAPS
        "truthful_claims": True,
    },
    "images": {
        "min_count": 1,
        # Preferred 1000px for zoom; can’t check px without fetching images, so we surface as advice
        "preferred_min_px": 1000,
        "white_bg_required": True,  # can’t verify programmatically here
        "no_text_watermarks": True,  # can’t verify programmatically here
    },
}

EXCEPTIONS_LOWERCASE_WORDS = {
    "and", "or", "for", "the", "a", "an", "in", "on", "over", "with", "to", "of"
}

# ---------------------------
# Helpers
# ---------------------------

# --- Universe detection ---
UNIVERSE_KEYWORDS: Dict[str, List[str]] = {
    "Beverages": ["beverage", "drink", "juice", "soda", "cola", "water", "tea", "coffee", "sparkling", "energy", "soft drink"],
    "Pantry Staples": ["staple", "rice", "flour", "atta", "sugar", "salt", "lentil", "dal", "pasta", "noodle", "oil", "ghee", "spice", "masala"],
    "Breakfast Cereals": ["cereal", "cornflakes", "corn flakes", "oats", "oatmeal", "granola", "muesli", "bran", "wheat flakes"],
    "Snacks": ["snack", "chips", "crisps", "namkeen", "nuts", "trail mix", "popcorn", "biscuits", "cookies", "pretzel"],
    "Canned & Packaged Foods": ["canned", "can", "tin", "packaged", "soup", "beans", "tomato", "corn", "tuna"],
    "Condiments & Sauces": ["ketchup", "mayonnaise", "mayo", "mustard", "sauce", "hot sauce", "soy", "salsa", "dressing", "vinegar"],
    "Baking Supplies": ["baking", "bake", "bicarbonate", "baking soda", "yeast", "cocoa", "chocolate chips", "vanilla", "baking powder"],
    "Oils & Vinegars": ["oil", "olive", "sunflower", "canola", "mustard oil", "sesame", "vinegar", "apple cider"],
    "Tea & Coffee": ["tea", "chai", "green tea", "black tea", "coffee", "espresso", "instant coffee", "grounds", "beans"],
    "Juices": ["juice", "mango", "orange", "apple", "cranberry", "pulp", "nectar"],
    "Water": ["water", "mineral", "spring", "purified", "sparkling"],
    "Dairy": ["milk", "cream", "butter", "ghee", "cheese", "yogurt", "curd"],
    "Baby Food": ["baby", "infant", "toddler", "puree", "cerelac", "formula"],
}

AVAILABLE_UNIVERSES: List[str] = []

def infer_universe_from_text(title: str, bullets: List[str], desc: str, category: str = "", allowed: Optional[List[str]] = None) -> Tuple[Optional[str], Dict[str, int]]:
    """Infer universe (top-level grocery-like category) from text using simple keyword scores.
    If `allowed` is provided, restrict candidates to that set and add each label's own tokens as keywords.
    Returns (best_universe_or_None, keyword_counts).
    """
    text = " ".join([
        str(title or ""),
        " ".join([b for b in bullets or [] if isinstance(b, str)]),
        str(desc or ""),
        str(category or ""),
    ]).lower()
    # Build candidate vocab
    vocab: Dict[str, List[str]] = dict(UNIVERSE_KEYWORDS)
    if allowed:
        allowed_set = {a.strip().title() for a in allowed if isinstance(a, str)}
        # Remove labels not allowed
        vocab = {u: kws for u, kws in vocab.items() if u in allowed_set}
        # Add any allowed labels missing from the base dictionary using label tokens
        for lab in allowed_set:
            if lab not in vocab:
                toks = [t for t in re.split(r"[^a-zA-Z0-9]+", lab.lower()) if t]
                vocab[lab] = toks or [lab.lower()]
    counts: Dict[str, int] = {u: 0 for u in vocab}
    for uni, kws in vocab.items():
        for kw in kws + [uni.lower()]:
            if kw and kw in text:
                counts[uni] += text.count(kw)
    best = max(counts, key=lambda k: counts[k]) if counts else None
    if best and counts[best] > 0:
        return best, counts
    return None, counts

def analyze_universe(client: Dict[str, Any], comp: Dict[str, Any]) -> Dict[str, Any]:
    client_bullets = split_bullets(client.get("bullets", ""))
    allowed = AVAILABLE_UNIVERSES if AVAILABLE_UNIVERSES else None
    inferred, counts = infer_universe_from_text(
        client.get("title", ""), client_bullets, client.get("description", ""), client.get("category", ""), allowed=allowed
    )
    client_prov = (client.get("universe") or "").strip()
    comp_prov = (comp.get("universe") or "").strip()

    issues: List[str] = []
    suggested: Optional[str] = None
    reason = ""

    if inferred and (not client_prov or client_prov.lower() != inferred.lower()):
        suggested = inferred
        reason = f"Inferred '{inferred}' from keywords in title/bullets/description"
        issues.append(f"Client universe '{client_prov or '—'}' may be incorrect; inferred '{inferred}'")
    elif not client_prov and comp_prov:
        suggested = comp_prov
        reason = "Client universe missing; align with competitor if same category"
        issues.append("Client universe missing; suggest aligning with competitor")
    elif comp_prov and client_prov and client_prov.lower() != comp_prov.lower() and inferred and inferred.lower() == comp_prov.lower():
        suggested = inferred
        reason = "Competitor universe matches inferred signals"
        issues.append("Client universe differs from competitor and inferred signals")

    return {
        "client_provided": client_prov or None,
        "competitor_provided": comp_prov or None,
        "inferred": inferred,
        "suggested": suggested,
        "reason": reason,
        "issues": issues,
        "keyword_counts": counts,
    }


def col_or_fallback(df: pd.DataFrame, names: List[str], default: str = "") -> str:
    for name in names:
        if name in df.columns:
            return name
    # if nothing found, create a missing column with default value
    missing = names[0]
    df[missing] = default
    return missing


def split_bullets(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    # Try the most common delimiters first
    candidates = ["\n", "||", "|", "•", ";", " • "]
    for delim in candidates:
        if delim in raw:
            parts = [p.strip() for p in raw.split(delim) if p.strip()]
            if len(parts) > 1:
                return parts
    # Fallback: split on periods for long texts, but only when safe
    if "." in raw and len(raw) > 120:
        parts = [p.strip() for p in raw.split(".") if p.strip()]
        return parts[:5]
    return [raw.strip()]


def count_images(raw: str) -> int:
    if not isinstance(raw, str) or not raw.strip():
        return 0
    # Split on whitespace or comma/pipe
    pieces = re.split(r"[\s,|]+", raw.strip())
    urls = [p for p in pieces if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    return len(urls)


def has_promo_terms(text: str) -> bool:
    if not isinstance(text, str):
        return False
    promo = ["free shipping", "sale", "discount", "% off", "%off", "deal", "hurry", "limited time"]
    t = text.lower()
    return any(p in t for p in promo)


def is_all_caps(text: str) -> bool:
    if not isinstance(text, str):
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def enforce_title_caps(text: str) -> str:
    words = text.split()
    fixed = []
    for i, w in enumerate(words):
        wl = w.lower()
        if i != 0 and wl in EXCEPTIONS_LOWERCASE_WORDS:
            fixed.append(wl)
        else:
            fixed.append(w.capitalize())
    return " ".join(fixed)


def rule_check_title(title: str, brand: str, is_bundle: bool = False) -> Tuple[int, List[str]]:
    score = 100
    issues = []
    if not isinstance(title, str) or not title.strip():
        return 0, ["Title is missing"]
    t = title.strip()
    # length
    if len(t) > RULES["title"]["max_chars"]:
        issues.append(f"Title exceeds {RULES['title']['max_chars']} characters (len={len(t)})")
        score -= 25
    # brand presence
    if RULES["title"]["brand_required"] and isinstance(brand, str) and brand.strip():
        if brand.lower() not in t.lower():
            issues.append("Brand name is missing from title")
            score -= 20
    # promo text
    if RULES["title"]["no_promo"] and has_promo_terms(t):
        issues.append("Remove promotional language in title")
        score -= 15
    # all caps
    if RULES["title"]["no_all_caps"] and is_all_caps(t):
        issues.append("Avoid ALL CAPS in title")
        score -= 10
    # (pack of X) handling (informational only)
    if not is_bundle and re.search(r"pack of\s*\d+", t, re.I):
        issues.append("'(pack of X)' should only be used for bundles; verify correctness")
        score -= 5
    return max(score, 0), issues


def rule_check_bullets(bullets: List[str]) -> Tuple[int, List[str]]:
    score = 100
    issues = []
    if not bullets:
        return 0, ["No bullets present (aim for up to 5 key features)"]
    if len(bullets) > RULES["bullets"]["max_count"]:
        issues.append(f"Too many bullets: {len(bullets)} (max {RULES['bullets']['max_count']})")
        score -= 15
    # Validate each bullet
    for i, b in enumerate(bullets, 1):
        if not b:
            continue
        # starting capital
        if RULES["bullets"]["start_capital"] and b[0].isalpha() and not b[0].isupper():
            issues.append(f"Bullet {i}: start with a capital letter")
            score -= 5
        # ending punctuation
        if RULES["bullets"]["no_end_punct"] and re.search(r"[.!?]$", b.strip()):
            issues.append(f"Bullet {i}: remove ending punctuation")
            score -= 3
        # promo/seller info
        if RULES["bullets"]["no_promo_or_seller_info"] and has_promo_terms(b):
            issues.append(f"Bullet {i}: remove promotional messaging")
            score -= 5
    return max(score, 0), issues


def rule_check_description(desc: str) -> Tuple[int, List[str]]:
    score = 100
    issues = []
    if not isinstance(desc, str) or not desc.strip():
        return 0, ["Description is missing (<= 200 chars)"]
    d = desc.strip()
    if len(d) > RULES["description"]["max_chars"]:
        issues.append(f"Description exceeds {RULES['description']['max_chars']} characters (len={len(d)})")
        score -= 20
    if RULES["description"]["no_promo"] and has_promo_terms(d):
        issues.append("Remove promotional language in description")
        score -= 10
    if RULES["description"]["sentence_caps"] and is_all_caps(d):
        issues.append("Avoid ALL CAPS in description")
        score -= 5
    return max(score, 0), issues


def compare_fields(client: Dict[str, Any], comp: Dict[str, Any]) -> Dict[str, Any]:
    client_bullets = split_bullets(client.get("bullets", ""))
    comp_bullets = split_bullets(comp.get("bullets", ""))

    client_images = count_images(client.get("image_urls", ""))
    comp_images = count_images(comp.get("image_urls", ""))

    title_score, title_issues = rule_check_title(client.get("title", ""), client.get("brand", ""))
    bullets_score, bullets_issues = rule_check_bullets(client_bullets)
    desc_score, desc_issues = rule_check_description(client.get("description", ""))

    # Universe analysis
    uni = analyze_universe(client, comp)

    summary = {
        "title": {
            "client_len": len((client.get("title") or "")),
            "comp_len": len((comp.get("title") or "")),
            "client_score": title_score,
            "issues": title_issues,
        },
        "bullets": {
            "client_count": len(client_bullets),
            "comp_count": len(comp_bullets),
            "client_score": bullets_score,
            "issues": bullets_issues,
        },
        "description": {
            "client_len": len((client.get("description") or "")),
            "comp_len": len((comp.get("description") or "")),
            "client_score": desc_score,
            "issues": desc_issues,
        },
        "images": {
            "client_count": client_images,
            "comp_count": comp_images,
            "issues": [] if client_images >= RULES["images"]["min_count"] else ["Add at least one image"],
        },
        "universe": uni,
    }

    # Heuristic improvement ideas based on competitor
    gaps = []
    if summary["title"]["client_len"] < summary["title"]["comp_len"]:
        gaps.append("Competitor title may include more attributes (e.g., size/material/use-case)")
    if summary["bullets"]["client_count"] < min(5, summary["bullets"]["comp_count"]):
        gaps.append("Add missing bullets to reach 5 concise key features")
    if summary["description"]["client_len"] < min(200, summary["description"]["comp_len"]):
        gaps.append("Add a concise use-case/benefit sentence in the description (<= 200 chars)")
    if summary["images"]["client_count"] < summary["images"]["comp_count"]:
        gaps.append("Upload additional images to match competitor coverage (white background, high-res)")
    if summary["universe"].get("suggested"):
        gaps.append(f"Universe: suggest '{summary['universe']['suggested']}' (reason: {summary['universe']['reason']})")

    summary["gaps_vs_competitor"] = gaps
    return summary


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


def call_llm(client_data: Dict[str, Any], comp_data: Dict[str, Any]) -> Dict[str, Any]:
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
        title = enforce_title_caps(title)[: RULES["title"]["max_chars"]].strip()
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
        desc = desc[: RULES["description"]["max_chars"]]
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
            "title_edit": enforce_title_caps((client_data.get("title") or "")[: RULES["title"]["max_chars"]]),
            "bullets_edits": ["Durable construction", "Comfortable fit", "Easy to clean"],
            "description_edit": "Compact design for everyday use; easy to clean; ideal for most pets"[: RULES["description"]["max_chars"]],
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
csv_file = st.sidebar.file_uploader("Upload SKUs CSV (asin_data_filled.csv)", type=["csv"], key="csv_uploader")

# Load data (require upload; no default path fallback)
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.session_state["uploaded_df"] = df
elif "uploaded_df" in st.session_state:
    # If user already uploaded once in this session, keep using it across reruns
    df = st.session_state["uploaded_df"]
else:
    st.info("Upload a CSV to continue. Expected columns: sku_id/product_id, title, bullet_points/bullets, description, image_url(s), brand, category.")
    st.stop()

# Column mapping
# Add common aliases from your CSV: product_id, bullet_points, image_url, retailer_brand_name, description_filled, retailer_category_node
_cols_lower = {c.lower(): c for c in df.columns}

def pick(*candidates: str, default: str = "") -> str:
    for cand in candidates:
        key = cand.lower()
        if key in _cols_lower:
            return _cols_lower[key]
    df[candidates[0]] = default
    return candidates[0]

sku_col = pick("sku_id", "asin", "sku", "id", "product_id")
title_col = pick("title", "product_title")
bullets_col = pick("bullets", "features", "key_features", "bullet_points")
desc_col = pick("description", "product_description", "desc", "description_filled")
images_col = pick("image_urls", "images", "image", "image_url")
brand_col = pick("brand", "brand_name", "retailer_brand_name")
category_col = pick("category", "node", "retailer_category_node")
universe_col = pick("universe")

# Build allowed universes from the CSV values (restrict inference to these)
raw_unis = df[universe_col].astype(str).fillna("").str.strip()
AVAILABLE_UNIVERSES = sorted({u.title() for u in raw_unis if u and u.lower() != "nan"})

# Debug: show which columns were mapped
with st.expander("Detected column mapping"):
    mapping = {
        "sku_col": sku_col,
        "title_col": title_col,
        "bullets_col": bullets_col,
        "desc_col": desc_col,
        "images_col": images_col,
        "brand_col": brand_col,
        "category_col": category_col,
        "universe_col": universe_col,
        "available_universes": AVAILABLE_UNIVERSES[:20],
    }
    st.write(mapping)

# Simple selection lists
sku_series = df[sku_col].astype(str).fillna("").str.strip()
sku_list = [s for s in sku_series.tolist() if s]
selected_client = st.sidebar.selectbox("Select Client SKU", sku_list)
selected_comp = st.sidebar.selectbox("Select Competitor SKU", sku_list, index=min(1, len(sku_list)-1))

# Extract records
client_row = df[df[sku_col].astype(str) == str(selected_client)].head(1)
comp_row = df[df[sku_col].astype(str) == str(selected_comp)].head(1)

if client_row.empty or comp_row.empty:
    st.warning("Please select valid SKUs")
    st.stop()

client_data = {
    "sku": client_row.iloc[0][sku_col],
    "title": client_row.iloc[0][title_col],
    "bullets": client_row.iloc[0][bullets_col],
    "description": client_row.iloc[0][desc_col],
    "image_urls": client_row.iloc[0][images_col],
    "brand": client_row.iloc[0][brand_col],
    "category": client_row.iloc[0][category_col],
    "universe": client_row.iloc[0][universe_col] if universe_col in df.columns else None,
}
comp_data = {
    "sku": comp_row.iloc[0][sku_col],
    "title": comp_row.iloc[0][title_col],
    "bullets": comp_row.iloc[0][bullets_col],
    "description": comp_row.iloc[0][desc_col],
    "image_urls": comp_row.iloc[0][images_col],
    "brand": comp_row.iloc[0][brand_col],
    "category": comp_row.iloc[0][category_col],
    "universe": comp_row.iloc[0][universe_col] if universe_col in df.columns else None,
}

# Two-column layout for side-by-side comparison
left, right = st.columns(2)
with left:
    st.subheader("Client SKU")
    st.write(f"**SKU**: {client_data['sku']}")
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
    st.write(f"**SKU**: {comp_data['sku']}")
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

# Rule checks & summary
summary = compare_fields(client_data, comp_data)

st.subheader("Rule checks (Client)")
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
        llm_out = call_llm(client_data, comp_data)
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
    st.code((out.get("description_edit", ""))[: RULES["description"]["max_chars"]])

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
        final_md.append((out.get("description_edit", ""))[: RULES["description"]["max_chars"]])

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
{(out.get('description_edit',''))[: RULES['description']['max_chars']]}

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
