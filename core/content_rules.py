"""Reusable content validation helpers and rule definitions."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

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


def _coerce_string_list(raw: Any) -> List[str]:
    """Return a list of cleaned strings from list-like inputs."""

    items: List[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        parsed: Optional[Any] = None
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, list):
            items = parsed
        else:
            matches = re.findall(r'"(.*?)"', stripped)
            if matches:
                items = matches
            else:
                items = [stripped]
    else:
        return []

    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if text.startswith('["') and text.endswith('"]') and len(text) >= 4:
            text = text[2:-2].strip()
        if text.startswith('"') and text.endswith('"') and len(text) >= 2:
            text = text[1:-1].strip()
        cleaned.append(text)
    return cleaned


def split_bullets(raw: Any) -> List[str]:
    return _coerce_string_list(raw)


def _extract_urls_from_text(text: str) -> List[str]:
    pieces = re.split(r"[\s,|]+", text)
    return [
        p
        for p in pieces
        if p and p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".res"))
    ]


def _count_urls_from_text(text: str) -> int:
    return len(_extract_urls_from_text(text))


def extract_image_urls(raw: Any) -> List[str]:
    urls = _coerce_string_list(raw)
    if urls:
        if isinstance(raw, str):
            stripped = raw.strip()
            if urls == [stripped]:
                fallback_urls = _extract_urls_from_text(stripped)
                if fallback_urls:
                    return fallback_urls
        return [str(url).strip() for url in urls if str(url).strip()]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            return _extract_urls_from_text(stripped)
    return []


def count_images(raw: Any) -> int:
    return len(extract_image_urls(raw))


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


def limit_text_with_sentence_guard(
    text: str,
    max_chars: int,
    *,
    prefer_sentence: bool = True,
) -> str:
    """Truncate text without cutting mid-sentence or mid-word."""

    if not isinstance(text, str):
        return ""

    normalized = re.sub(r"\s+", " ", text).strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rstrip()

    if prefer_sentence:
        sentence_break = -1
        for marker in ".!?":
            idx = truncated.rfind(marker)
            if idx > sentence_break:
                sentence_break = idx
        if sentence_break != -1 and sentence_break >= max(0, max_chars // 2):
            return truncated[: sentence_break + 1].strip()

    word_break = truncated.rfind(" ")
    if word_break > 0:
        candidate = truncated[:word_break].rstrip(",;:-")
        if candidate:
            return candidate.strip()

    return truncated.strip()


def rule_check_title(
    title: str, brand: str, rules: Dict[str, Any], is_bundle: bool = False
) -> Tuple[int, List[str]]:
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
        issues.append(
            "'(pack of X)' should only be used for bundles; verify correctness"
        )
        score -= 5
    return max(score, 0), issues


def rule_check_bullets(
    bullets: List[str], rules: Dict[str, Any]
) -> Tuple[int, List[str]]:
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
        issues.append(
            f"Description exceeds {rules['max_chars']} characters (len={len(d)})"
        )
        score -= 20
    if rules.get("no_promo") and has_promo_terms(d):
        issues.append("Remove promotional language in description")
        score -= 10
    if rules.get("sentence_caps") and is_all_caps(d):
        issues.append("Avoid ALL CAPS in description")
        score -= 5
    return max(score, 0), issues


def compare_fields(
    client: Dict[str, Any],
    comp: Dict[str, Any],
    *,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rules = rules or DEFAULT_RULES
    client_bullets = split_bullets(client.get("bullets", ""))
    comp_bullets = split_bullets(comp.get("bullets", ""))

    client_image_urls = extract_image_urls(client.get("image_urls", ""))
    comp_image_urls = extract_image_urls(comp.get("image_urls", ""))
    client_images = len(client_image_urls)
    comp_images = len(comp_image_urls)

    title_score, title_issues = rule_check_title(
        client.get("title", ""), client.get("brand", ""), rules.get("title", {})
    )
    bullets_score, bullets_issues = rule_check_bullets(
        client_bullets, rules.get("bullets", {})
    )
    desc_score, desc_issues = rule_check_description(
        client.get("description", ""), rules.get("description", {})
    )

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
    }

    gaps = []
    if summary["title"]["client_len"] < summary["title"]["comp_len"]:
        gaps.append(
            "Competitor title may include more attributes (e.g., size/material/use-case)"
        )
    if summary["bullets"]["client_count"] < min(5, summary["bullets"]["comp_count"]):
        gaps.append("Add missing bullets to reach 5 concise key features")
    if summary["description"]["client_len"] < min(
        200, summary["description"]["comp_len"]
    ):
        gaps.append(
            "Add a concise use-case/benefit sentence in the description (<= 200 chars)"
        )
    if summary["images"]["client_count"] < summary["images"]["comp_count"]:
        gaps.append(
            "Upload additional images to match competitor coverage (white background, high-res)"
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
