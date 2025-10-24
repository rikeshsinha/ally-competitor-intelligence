"""Chain for finding similar competitor products for a selected client SKU."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.runnables import RunnableLambda

try:  # Optional Streamlit dependency for session-based API keys
    import streamlit as st
except Exception:  # pragma: no cover - streamlit not present during tests
    st = None  # type: ignore

try:  # Optional OpenAI dependency – mirror behaviour used elsewhere
    from openai import OpenAI
except Exception:  # pragma: no cover - SDK may be missing
    OpenAI = None  # type: ignore


BundleValue = Optional[int]


@dataclass
class SimilarProductMatch:
    sku: str
    brand: str
    title: str
    bundle_size: BundleValue = None
    score: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class SimilarProductsResult:
    matches: List[SimilarProductMatch] = field(default_factory=list)
    using_llm: bool = False
    message: Optional[str] = None


_BUNDLE_KEYWORDS = {
    "bundle",
    "pack",
    "packs",
    "count",
    "ct",
    "qty",
    "quantity",
    "pieces",
    "pcs",
    "pack_size",
    "bundle_size",
}

_BUNDLE_PATTERNS = [
    re.compile(r"(?:pack|set|bundle|pk)\s*(?:of)?\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*[-\s]*(?:pack|ct|count|qty|pcs|pieces)", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*[xX]\s*(?:count|ct|pcs|pieces)?", re.IGNORECASE),
]


def _tokenize(text: str) -> Sequence[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):  # type: ignore[no-untyped-call]
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d{1,4}", text)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None


def _infer_bundle_from_texts(texts: Iterable[Any]) -> Optional[int]:
    for raw in texts:
        if raw is None:
            continue
        text = str(raw)
        for pattern in _BUNDLE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1)
                try:
                    return int(value)
                except ValueError:
                    continue
    return None


def _infer_bundle_from_row(row: pd.Series, bundle_columns: Sequence[str], text_fields: Sequence[str]) -> Optional[int]:
    for column in bundle_columns:
        if column not in row:
            continue
        value = row[column]
        bundle = _coerce_int(value)
        if bundle:
            return bundle
    texts: List[Any] = []
    for field in text_fields:
        if field in row:
            texts.append(row[field])
    return _infer_bundle_from_texts(texts)


def _infer_bundle_from_dict(data: Dict[str, Any]) -> Optional[int]:
    for key, value in data.items():
        if not key:
            continue
        key_lower = str(key).lower()
        if key_lower in _BUNDLE_KEYWORDS:
            bundle = _coerce_int(value)
            if bundle:
                return bundle
    return _infer_bundle_from_texts(
        [data.get("title"), data.get("description"), data.get("bullets")]
    )


def _find_bundle_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for column in df.columns:
        lower = str(column).lower()
        if any(keyword in lower for keyword in _BUNDLE_KEYWORDS):
            columns.append(column)
    return columns


def _prepare_candidate(
    row: pd.Series,
    column_map: Dict[str, str],
    bundle_columns: Sequence[str],
    text_fields: Sequence[str],
) -> Dict[str, Any]:
    sku_col = column_map.get("sku_col")
    title_col = column_map.get("title_col")
    desc_col = column_map.get("desc_col")
    brand_col = column_map.get("brand_col")

    sku = str(row.get(sku_col, "") or "").strip()
    title = str(row.get(title_col, "") or "").strip()
    description = str(row.get(desc_col, "") or "").strip()
    brand = str(row.get(brand_col, "") or "").strip()

    bundle_size = _infer_bundle_from_row(row, bundle_columns, text_fields)
    return {
        "sku": sku,
        "title": title,
        "description": description,
        "brand": brand,
        "bundle_size": bundle_size,
        "_row": row,
    }


def _heuristic_score_matches(
    client: Dict[str, Any],
    client_bundle: Optional[int],
    candidates: List[Dict[str, Any]],
) -> List[SimilarProductMatch]:
    client_text_parts = [
        str(client.get("title", "") or ""),
        str(client.get("description", "") or ""),
    ]
    client_text = " ".join(part for part in client_text_parts if part).lower()
    client_tokens = _tokenize(client_text)
    client_token_set = set(client_tokens)

    scored: List[Tuple[float, int, int, SimilarProductMatch]] = []
    for idx, candidate in enumerate(candidates):
        title = candidate.get("title", "")
        description = candidate.get("description", "")
        candidate_text = f"{title} {description}".strip().lower()
        cand_tokens = _tokenize(candidate_text)
        cand_token_set = set(cand_tokens)
        overlap = len(client_token_set & cand_token_set)
        union = len(client_token_set | cand_token_set)
        token_score = (overlap / union) if union else 0.0
        substring_bonus = 0.0
        if title and title.lower() in client_text:
            substring_bonus += 0.05
        if client.get("title") and str(client.get("title")).lower() in candidate_text:
            substring_bonus += 0.05

        candidate_bundle = candidate.get("bundle_size")
        bundle_bonus = 0.0
        bundle_reason = ""
        if client_bundle and candidate_bundle:
            diff = abs(client_bundle - candidate_bundle)
            if diff == 0:
                bundle_bonus = 0.15
                bundle_reason = "Matching bundle size"
            else:
                bundle_bonus = max(0.0, 0.12 - 0.02 * diff)
                if bundle_bonus > 0:
                    bundle_reason = f"Bundle difference of {diff}"

        score = token_score + substring_bonus + bundle_bonus

        reason_parts: List[str] = []
        if overlap:
            reason_parts.append(
                f"{overlap} overlapping terms (token score {token_score:.2f})"
            )
        if bundle_reason:
            reason_parts.append(bundle_reason)
        if not reason_parts:
            reason_parts.append("Text similarity heuristic")

        match = SimilarProductMatch(
            sku=str(candidate.get("sku", "")),
            brand=str(candidate.get("brand", "")),
            title=str(candidate.get("title", "")),
            bundle_size=candidate_bundle,
            score=round(score, 4),
            reason="; ".join(reason_parts),
        )

        scored.append((score, overlap, idx, match))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [entry[-1] for entry in scored[:5]]


def _serialize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sku": candidate.get("sku", ""),
        "brand": candidate.get("brand", ""),
        "title": candidate.get("title", ""),
        "description": candidate.get("description", ""),
        "bundle_size": candidate.get("bundle_size"),
    }


class _SimilarProductsRunnable:
    def __call__(self, inputs: Dict[str, Any]) -> SimilarProductsResult:  # type: ignore[override]
        dataframe = inputs.get("dataframe")
        column_map = inputs.get("column_map") or {}
        client = inputs.get("client") or {}

        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            return SimilarProductsResult(matches=[], using_llm=False, message="No data available")

        brand_col = column_map.get("brand_col")
        if not brand_col or brand_col not in dataframe.columns:
            return SimilarProductsResult(
                matches=[],
                using_llm=False,
                message="Brand column not found",
            )

        bundle_columns = _find_bundle_columns(dataframe)
        text_fields = [
            column_map.get("title_col", "title"),
            column_map.get("desc_col", "description"),
        ]

        client_brand = str(client.get("brand", "") or "").strip().lower()
        client_sku = str(client.get("sku_original") or client.get("sku") or "").strip()
        client_bundle = _infer_bundle_from_dict(client)

        candidates: List[Dict[str, Any]] = []
        for _, row in dataframe.iterrows():
            candidate = _prepare_candidate(row, column_map, bundle_columns, text_fields)
            if not candidate.get("brand"):
                continue
            if candidate["brand"].strip().lower() == client_brand:
                continue
            if client_sku and candidate.get("sku") == client_sku:
                continue
            candidates.append(candidate)

        if not candidates:
            return SimilarProductsResult(
                matches=[],
                using_llm=False,
                message="No competitor rows found",
            )

        llm_client = self._get_llm_client()
        if llm_client is not None:
            matches, message = self._call_openai(llm_client, client, candidates)
            if matches is not None:
                return SimilarProductsResult(
                    matches=matches,
                    using_llm=True,
                    message=message,
                )
            fallback_message = message or "OpenAI scoring unavailable; used heuristics"
        else:
            fallback_message = "No OpenAI client available; used heuristics"

        matches = _heuristic_score_matches(client, client_bundle, candidates)
        return SimilarProductsResult(
            matches=matches,
            using_llm=False,
            message=fallback_message,
        )

    def _get_llm_client(self) -> Any:
        if OpenAI is None:
            return None

        api_key: Optional[str] = None
        if st is not None:
            api_key = st.session_state.get("OPENAI_API_KEY_UI") or os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _call_openai(
        self,
        client: Any,
        client_record: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[List[SimilarProductMatch]], Optional[str]]:
        request_payload = {
            "client": {
                "sku": client_record.get("sku"),
                "brand": client_record.get("brand"),
                "title": client_record.get("title"),
                "description": client_record.get("description"),
                "bundle_size": _infer_bundle_from_dict(client_record),
            },
            "candidates": [_serialize_candidate(c) for c in candidates],
        }

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an ecommerce analyst."
                            " Select up to five competitor products that best match"
                            " the client SKU based on title, description, and bundle size."
                            " Return JSON with a 'matches' list."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Choose the closest competitors and provide a short reason."
                            f"\n\nData: {json.dumps(request_payload, ensure_ascii=False)}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except Exception as exc:  # pragma: no cover - depends on network
            return None, f"OpenAI call failed: {exc}"

        if not response.choices:
            return None, "OpenAI returned no choices"
        content = response.choices[0].message.content
        if not content:
            return None, "OpenAI returned empty content"

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None, "OpenAI response was not valid JSON"

        if isinstance(parsed, dict):
            raw_matches = parsed.get("matches")
        elif isinstance(parsed, list):
            raw_matches = parsed
        else:
            raw_matches = None

        if not isinstance(raw_matches, list):
            return None, "OpenAI response missing 'matches'"

        matches: List[SimilarProductMatch] = []
        for item in raw_matches[:5]:
            if not isinstance(item, dict):
                continue
            match = SimilarProductMatch(
                sku=str(item.get("sku", "")),
                brand=str(item.get("brand", "")),
                title=str(item.get("title", "")),
                bundle_size=_coerce_int(item.get("bundle_size")),
                score=(
                    float(item["score"])
                    if isinstance(item.get("score"), (int, float))
                    else None
                ),
                reason=str(
                    item.get("reason")
                    or item.get("explanation")
                    or item.get("rationale", "")
                ).strip()
                or None,
            )
            matches.append(match)

        return matches, "Used OpenAI ranking"


def create_similar_products_chain() -> RunnableLambda:
    """Return a runnable that produces :class:`SimilarProductsResult`."""

    return RunnableLambda(_SimilarProductsRunnable())


__all__ = [
    "SimilarProductMatch",
    "SimilarProductsResult",
    "create_similar_products_chain",
]

