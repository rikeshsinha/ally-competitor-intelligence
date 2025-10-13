"""Reusable content validation helpers and rule definitions."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    "title": {
        "max_chars": 50,
        "brand_required": True,
        "capitalize_words": True,
        "no_all_caps": True,
        "no_promo": True,
        "allow_pack_of": True,
    },
    "bullets": {
        "max_count": 5,
        "start_capital": True,
        "sentence_fragments": True,
        "no_end_punct": True,
        "no_promo_or_seller_info": True,
        "numbers_as_numerals": True,
        "semicolons_ok": True,
    },
    "description": {
        "max_chars": 200,
        "no_promo": True,
        "no_seller_info": True,
        "sentence_caps": True,
        "truthful_claims": True,
    },
    "images": {
        "min_count": 1,
        "preferred_min_px": 1000,
        "white_bg_required": True,
        "no_text_watermarks": True,
    },
}

EXCEPTIONS_LOWERCASE_WORDS = {
    "and",
    "or",
    "for",
    "the",
    "a",
    "an",
    "in",
    "on",
    "over",
    "with",
    "to",
    "of",
}


@dataclass
class UniverseAnalysis:
    client_provided: Optional[str]
    competitor_provided: Optional[str]
    inferred: Optional[str]
    suggested: Optional[str]
    reason: str
    issues: List[str]
    keyword_counts: Dict[str, int]


UNIVERSE_KEYWORDS: Dict[str, List[str]] = {
    "Beverages": [
        "beverage",
        "drink",
        "juice",
        "soda",
        "cola",
        "water",
        "tea",
        "coffee",
        "sparkling",
        "energy",
        "soft drink",
    ],
    "Pantry Staples": [
        "staple",
        "rice",
        "flour",
        "atta",
        "sugar",
        "salt",
        "lentil",
        "dal",
        "pasta",
        "noodle",
        "oil",
        "ghee",
        "spice",
        "masala",
    ],
    "Breakfast Cereals": [
        "cereal",
        "cornflakes",
        "corn flakes",
        "oats",
        "oatmeal",
        "granola",
        "muesli",
        "bran",
        "wheat flakes",
    ],
    "Snacks": [
        "snack",
        "chips",
        "crisps",
        "namkeen",
        "nuts",
        "trail mix",
        "popcorn",
        "biscuits",
        "cookies",
        "pretzel",
    ],
    "Canned & Packaged Foods": [
        "canned",
        "can",
        "tin",
        "packaged",
        "soup",
        "beans",
        "tomato",
        "corn",
        "tuna",
    ],
    "Condiments & Sauces": [
        "ketchup",
        "mayonnaise",
        "mayo",
        "mustard",
        "sauce",
        "hot sauce",
        "soy",
        "salsa",
        "dressing",
        "vinegar",
    ],
    "Baking Supplies": [
        "baking",
        "bake",
        "bicarbonate",
        "baking soda",
        "yeast",
        "cocoa",
        "chocolate chips",
        "vanilla",
        "baking powder",
    ],
    "Oils & Vinegars": [
        "oil",
        "olive",
        "sunflower",
        "canola",
        "mustard oil",
        "sesame",
        "vinegar",
        "apple cider",
    ],
    "Tea & Coffee": [
        "tea",
        "chai",
        "green tea",
        "black tea",
        "coffee",
        "espresso",
        "instant coffee",
        "grounds",
        "beans",
    ],
    "Juices": ["juice", "mango", "orange", "apple", "cranberry", "pulp", "nectar"],
    "Water": ["water", "mineral", "spring", "purified", "sparkling"],
    "Dairy": ["milk", "cream", "butter", "ghee", "cheese", "yogurt", "curd"],
    "Baby Food": ["baby", "infant", "toddler", "puree", "cerelac", "formula"],
}


def split_bullets(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    candidates = ["\n", "||", "|", "•", ";", " • "]
    for delim in candidates:
        if delim in raw:
            parts = [p.strip() for p in raw.split(delim) if p.strip()]
            if len(parts) > 1:
                return parts
    if "." in raw and len(raw) > 120:
        parts = [p.strip() for p in raw.split(".") if p.strip()]
        return parts[:5]
    return [raw.strip()]


def count_images(raw: str) -> int:
    if not isinstance(raw, str) or not raw.strip():
        return 0
    pieces = re.split(r"[\s,|]+", raw.strip())
    urls = [p for p in pieces if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    return len(urls)


def has_promo_terms(text: str) -> bool:
    if not isinstance(text, str):
        return False
    promo = [
        "free shipping",
        "sale",
        "discount",
        "% off",
        "%off",
        "deal",
        "hurry",
        "limited time",
    ]
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


def rule_check_title(title: str, brand: str, rules: Dict[str, Any], is_bundle: bool = False) -> Tuple[int, List[str]]:
    score = 100
    issues: List[str] = []
    if not isinstance(title, str) or not title.strip():
        return 0, ["Title is missing"]
    t = title.strip()
    if len(t) > rules.get("max_chars", 0):
        issues.append(f"Title exceeds {rules['max_chars']} characters (len={len(t)})")
        score -= 25
    if rules.get("brand_required") and isinstance(brand, str) and brand.strip():
        if brand.lower() not in t.lower():
            issues.append("Brand name is missing from title")
            score -= 20
    if rules.get("no_promo") and has_promo_terms(t):
        issues.append("Remove promotional language in title")
        score -= 15
    if rules.get("no_all_caps") and is_all_caps(t):
        issues.append("Avoid ALL CAPS in title")
        score -= 10
    if not is_bundle and re.search(r"pack of\s*\d+", t, re.I):
        issues.append("'(pack of X)' should only be used for bundles; verify correctness")
        score -= 5
    return max(score, 0), issues


def rule_check_bullets(bullets: List[str], rules: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 100
    issues: List[str] = []
    if not bullets:
        return 0, ["No bullets present (aim for up to 5 key features)"]
    if len(bullets) > rules.get("max_count", len(bullets)):
        issues.append(f"Too many bullets: {len(bullets)} (max {rules['max_count']})")
        score -= 15
    for i, b in enumerate(bullets, 1):
        if not b:
            continue
        if rules.get("start_capital") and b[0].isalpha() and not b[0].isupper():
            issues.append(f"Bullet {i}: start with a capital letter")
            score -= 5
        if rules.get("no_end_punct") and re.search(r"[.!?]$", b.strip()):
            issues.append(f"Bullet {i}: remove ending punctuation")
            score -= 3
        if rules.get("no_promo_or_seller_info") and has_promo_terms(b):
            issues.append(f"Bullet {i}: remove promotional messaging")
            score -= 5
    return max(score, 0), issues


def rule_check_description(desc: str, rules: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 100
    issues: List[str] = []
    if not isinstance(desc, str) or not desc.strip():
        return 0, ["Description is missing (<= 200 chars)"]
    d = desc.strip()
    if len(d) > rules.get("max_chars", len(d)):
        issues.append(f"Description exceeds {rules['max_chars']} characters (len={len(d)})")
        score -= 20
    if rules.get("no_promo") and has_promo_terms(d):
        issues.append("Remove promotional language in description")
        score -= 10
    if rules.get("sentence_caps") and is_all_caps(d):
        issues.append("Avoid ALL CAPS in description")
        score -= 5
    return max(score, 0), issues


def infer_universe_from_text(
    title: str,
    bullets: List[str],
    desc: str,
    category: str = "",
    allowed: Optional[Iterable[str]] = None,
) -> Tuple[Optional[str], Dict[str, int]]:
    text = " ".join([
        str(title or ""),
        " ".join([b for b in bullets or [] if isinstance(b, str)]),
        str(desc or ""),
        str(category or ""),
    ]).lower()
    vocab: Dict[str, List[str]] = dict(UNIVERSE_KEYWORDS)
    if allowed:
        allowed_set = {a.strip().title() for a in allowed if isinstance(a, str)}
        vocab = {u: kws for u, kws in vocab.items() if u in allowed_set}
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


def analyze_universe(
    client: Dict[str, Any],
    comp: Dict[str, Any],
    available_universes: Optional[Iterable[str]] = None,
) -> UniverseAnalysis:
    client_bullets = split_bullets(client.get("bullets", ""))
    inferred, counts = infer_universe_from_text(
        client.get("title", ""),
        client_bullets,
        client.get("description", ""),
        client.get("category", ""),
        allowed=available_universes,
    )
    client_prov = (client.get("universe") or "").strip() or None
    comp_prov = (comp.get("universe") or "").strip() or None

    issues: List[str] = []
    suggested: Optional[str] = None
    reason = ""

    if inferred and (not client_prov or client_prov.lower() != inferred.lower()):
        suggested = inferred
        reason = f"Inferred '{inferred}' from keywords in title/bullets/description"
        issues.append(
            f"Client universe '{client_prov or '—'}' may be incorrect; inferred '{inferred}'"
        )
    elif not client_prov and comp_prov:
        suggested = comp_prov
        reason = "Client universe missing; align with competitor if same category"
        issues.append("Client universe missing; suggest aligning with competitor")
    elif (
        comp_prov
        and client_prov
        and client_prov.lower() != comp_prov.lower()
        and inferred
        and inferred.lower() == comp_prov.lower()
    ):
        suggested = inferred
        reason = "Competitor universe matches inferred signals"
        issues.append("Client universe differs from competitor and inferred signals")

    return UniverseAnalysis(
        client_provided=client_prov,
        competitor_provided=comp_prov,
        inferred=inferred,
        suggested=suggested,
        reason=reason,
        issues=issues,
        keyword_counts=counts,
    )


def compare_fields(
    client: Dict[str, Any],
    comp: Dict[str, Any],
    *,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
    available_universes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    rules = rules or DEFAULT_RULES
    client_bullets = split_bullets(client.get("bullets", ""))
    comp_bullets = split_bullets(comp.get("bullets", ""))

    client_images = count_images(client.get("image_urls", ""))
    comp_images = count_images(comp.get("image_urls", ""))

    title_score, title_issues = rule_check_title(
        client.get("title", ""), client.get("brand", ""), rules.get("title", {})
    )
    bullets_score, bullets_issues = rule_check_bullets(
        client_bullets, rules.get("bullets", {})
    )
    desc_score, desc_issues = rule_check_description(
        client.get("description", ""), rules.get("description", {})
    )

    uni = analyze_universe(client, comp, available_universes=available_universes)

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
            "issues": []
            if client_images >= rules.get("images", {}).get("min_count", 0)
            else ["Add at least one image"],
        },
        "universe": {
            "client_provided": uni.client_provided,
            "competitor_provided": uni.competitor_provided,
            "inferred": uni.inferred,
            "suggested": uni.suggested,
            "reason": uni.reason,
            "issues": uni.issues,
            "keyword_counts": uni.keyword_counts,
        },
    }

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
        gaps.append(
            f"Universe: suggest '{summary['universe']['suggested']}' (reason: {summary['universe']['reason']})"
        )

    summary["gaps_vs_competitor"] = gaps
    return summary


def dataframe_from_uploaded_file(uploaded) -> pd.DataFrame:
    """Safely load a CSV from a Streamlit UploadedFile."""
    if uploaded is None:
        raise ValueError("No CSV uploaded")
    position = uploaded.tell()
    uploaded.seek(0)
    df = pd.read_csv(uploaded)
    uploaded.seek(position)
    return df
