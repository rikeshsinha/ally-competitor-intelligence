"""Chain utilities for recommending similar competitor products."""
from __future__ import annotations

import difflib
import json
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.runnables import RunnableLambda

try:  # Streamlit is optional during testing
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency guard
    st = None  # type: ignore

try:  # Optional OpenAI dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are an ecommerce analyst who evaluates product similarity across brands. "
    "Return responses strictly in JSON."
)


MATCH_PROMPT_TEMPLATE = (
    "You are given JSON describing a base product and a set of candidate competitor products.\n"
    "Select up to {max_matches} candidates that best match the base product based on title, description, and bundle_size.\n"
    "Do not include any candidate whose brand matches the base product brand.\n"
    "When two candidates are equally relevant, prefer the one with the lower average_search_rank value.\n"
    "Respond with a JSON object of the form {{\"matches\": [{{\"candidate_id\": str, \"reason\": str}}...]}}.\n"
    "Order matches from the strongest match to the weakest match.\n"
    "JSON payload:\n{payload}"
)


def _get_openai_client() -> Optional["OpenAI"]:
    api_key: Optional[str] = None
    if st is not None:
        api_key = st.session_state.get("OPENAI_API_KEY_UI")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:  # pragma: no cover - runtime safety
        return None


class _SimilarProductFinder:
    def __call__(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
        df = inputs.get("dataframe")
        column_map: Dict[str, str] = inputs.get("column_map", {})
        selected_index = inputs.get("selected_index")

        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        try:
            base_idx = int(selected_index)
        except (TypeError, ValueError):
            return []
        if base_idx < 0 or base_idx >= len(df):
            return []

        base_row = df.iloc[base_idx]
        sku_col = column_map.get("sku_col")
        brand_col = column_map.get("brand_col")
        title_col = column_map.get("title_col")
        desc_col = column_map.get("desc_col")
        bundle_col = column_map.get("bundle_size_col")
        avg_rank_col = column_map.get("avg_rank_search_col")

        base_brand = self._normalize_str(base_row.get(brand_col))
        base_title = self._to_text(base_row.get(title_col))
        base_desc = self._to_text(base_row.get(desc_col))
        base_bundle = self._to_text(base_row.get(bundle_col))
        base_bundle_norm = self._normalize_str(base_bundle)
        base_text = self._normalize_text(" ".join(part for part in [base_title, base_desc, base_bundle] if part))

        candidates: List[Dict[str, Any]] = []
        for row_idx in range(len(df)):
            if row_idx == base_idx:
                continue
            row = df.iloc[row_idx]
            candidate_brand_raw = self._normalize_str(row.get(brand_col))
            if base_brand and candidate_brand_raw and candidate_brand_raw == base_brand:
                continue

            candidate_title = self._to_text(row.get(title_col))
            candidate_desc = self._to_text(row.get(desc_col))
            candidate_bundle = self._to_text(row.get(bundle_col))
            candidate_bundle_norm = self._normalize_str(candidate_bundle)
            candidate_text = self._normalize_text(
                " ".join(part for part in [candidate_title, candidate_desc, candidate_bundle] if part)
            )
            similarity = difflib.SequenceMatcher(None, base_text, candidate_text).ratio()
            if base_bundle_norm and candidate_bundle_norm and candidate_bundle_norm == base_bundle_norm:
                similarity = min(1.0, similarity + 0.1)

            avg_rank_value = self._coerce_avg_rank(row.get(avg_rank_col))
            sku_value = self._to_text(row.get(sku_col))
            candidates.append(
                {
                    "row_index": row_idx,
                    "sku": sku_value,
                    "title": candidate_title,
                    "brand": self._to_text(row.get(brand_col)),
                    "description": candidate_desc,
                    "bundle_size": candidate_bundle,
                    "bundle_norm": candidate_bundle_norm,
                    "avg_rank_search": avg_rank_value,
                    "similarity": similarity,
                }
            )

        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda item: item["similarity"], reverse=True)
        limited_candidates = sorted_candidates[: min(25, len(sorted_candidates))]
        for idx, candidate in enumerate(limited_candidates):
            candidate["candidate_id"] = f"C{idx}"

        max_matches = min(5, len(limited_candidates))
        payload = self._build_payload(
            base_row,
            sku_col,
            brand_col,
            title_col,
            desc_col,
            bundle_col,
            limited_candidates,
        )

        matches = self._invoke_llm(payload, max_matches)
        candidate_lookup = {entry["candidate_id"]: entry for entry in limited_candidates}
        results: List[Dict[str, Any]] = []

        if matches:
            used_ids = set()
            for match in matches:
                if not isinstance(match, dict):
                    continue
                candidate_id = str(match.get("candidate_id") or match.get("id") or "").strip()
                if not candidate_id or candidate_id in used_ids:
                    continue
                candidate = candidate_lookup.get(candidate_id)
                if candidate is None:
                    continue
                reason = self._to_text(match.get("reason") or match.get("justification"))
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "row_index": candidate["row_index"],
                        "sku": candidate["sku"],
                        "title": candidate["title"],
                        "brand": candidate["brand"],
                        "bundle_size": candidate["bundle_size"],
                        "avg_rank_search": candidate["avg_rank_search"],
                        "reason": reason or "LLM-selected similar product",
                        "score": candidate["similarity"],
                    }
                )
                used_ids.add(candidate_id)
                if len(results) >= max_matches:
                    break

            if len(results) < max_matches:
                remaining = [
                    cand
                    for cand in limited_candidates
                    if cand["candidate_id"] not in {entry["candidate_id"] for entry in results}
                ]
                results.extend(self._heuristic_fill(remaining, base_bundle_norm, max_matches - len(results)))
        else:
            results = self._heuristic_fill(limited_candidates, base_bundle_norm, max_matches)

        results = self._rank_by_avg_search(results)
        return results[:max_matches]

    def _build_payload(
        self,
        base_row: pd.Series,
        sku_col: Optional[str],
        brand_col: Optional[str],
        title_col: Optional[str],
        desc_col: Optional[str],
        bundle_col: Optional[str],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_payload = {
            "sku": self._to_text(base_row.get(sku_col)),
            "brand": self._to_text(base_row.get(brand_col)),
            "title": self._to_text(base_row.get(title_col)),
            "description": self._truncate(self._to_text(base_row.get(desc_col)), 600),
            "bundle_size": self._to_text(base_row.get(bundle_col)),
        }
        candidate_payload = []
        for candidate in candidates:
            candidate_payload.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "sku": candidate.get("sku"),
                    "brand": candidate.get("brand"),
                    "title": candidate.get("title"),
                    "description": self._truncate(candidate.get("description"), 400),
                    "bundle_size": candidate.get("bundle_size"),
                    "average_search_rank": candidate.get("avg_rank_search"),
                    "similarity_hint": round(float(candidate.get("similarity", 0)), 3),
                }
            )
        return {"base_product": base_payload, "candidates": candidate_payload}

    def _invoke_llm(
        self, payload: Dict[str, Any], max_matches: int
    ) -> Optional[List[Dict[str, Any]]]:
        client = _get_openai_client()
        if client is None:
            return None
        prompt = MATCH_PROMPT_TEMPLATE.format(
            max_matches=max_matches,
            payload=json.dumps(payload, indent=2, ensure_ascii=False),
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except Exception:  # pragma: no cover - runtime guard
            return None

        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            return None
        if not content:
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        matches = parsed.get("matches") or parsed.get("candidates")
        if not isinstance(matches, list):
            return None
        return matches

    def _heuristic_fill(
        self,
        candidates: List[Dict[str, Any]],
        base_bundle_norm: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        sorted_candidates = sorted(
            candidates,
            key=lambda cand: (
                -float(cand.get("similarity", 0.0)),
                self._avg_rank_sort_key(cand.get("avg_rank_search")),
                cand.get("row_index", 0),
            ),
        )
        results: List[Dict[str, Any]] = []
        for candidate in sorted_candidates[:limit]:
            reason_parts = [f"Similarity score {candidate.get('similarity', 0.0):.2f}"]
            if base_bundle_norm and candidate.get("bundle_norm") == base_bundle_norm:
                reason_parts.append("Bundle size matches")
            results.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "row_index": candidate.get("row_index"),
                    "sku": candidate.get("sku"),
                    "title": candidate.get("title"),
                    "brand": candidate.get("brand"),
                    "bundle_size": candidate.get("bundle_size"),
                    "avg_rank_search": candidate.get("avg_rank_search"),
                    "reason": "; ".join(reason_parts),
                    "score": candidate.get("similarity", 0.0),
                }
            )
        return results

    def _rank_by_avg_search(
        self, matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return sorted(
            matches,
            key=lambda entry: (
                self._avg_rank_sort_key(entry.get("avg_rank_search")),
                -float(entry.get("score", 0.0)),
            ),
        )

    def _avg_rank_sort_key(self, value: Any) -> float:
        numeric = self._coerce_avg_rank(value)
        if numeric is None or not math.isfinite(numeric):
            return math.inf
        return numeric

    @staticmethod
    def _truncate(text: Any, limit: int) -> str:
        value = "" if text is None else str(text)
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).lower()

    @staticmethod
    def _normalize_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _coerce_avg_rank(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                value = float(stripped)
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if pd.isna(numeric):
                return None
            return numeric
        return None


def create_similar_product_chain() -> RunnableLambda:
    """Return a runnable chain that surfaces similar competitor products."""
    return RunnableLambda(_SimilarProductFinder())
