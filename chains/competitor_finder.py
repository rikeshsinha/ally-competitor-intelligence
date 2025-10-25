"""Utilities for surfacing competitor recommendations based on similarity."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from typing import Any, Dict, Iterable, List

import pandas as pd

from chains.sku_extractor import SKUData


@dataclass
class _Candidate:
    brand: str
    sku: str
    title: str
    rank: float
    similarity: float


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if pd.isna(value):  # type: ignore[arg-type]
        return ""
    return str(value).strip()


def _normalize_rank(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if math.isnan(number) or math.isinf(number):
        return float("inf")
    return number


def _gather_candidates(
    df: pd.DataFrame,
    column_map: Dict[str, str],
    *,
    target_sku: str,
    target_text: str,
) -> Iterable[_Candidate]:
    sku_col = column_map["sku_col"]
    title_col = column_map["title_col"]
    bullets_col = column_map["bullets_col"]
    desc_col = column_map["desc_col"]
    brand_col = column_map["brand_col"]
    avg_rank_col = column_map.get("avg_rank_col")

    for _, row in df.iterrows():
        row_sku = _coerce_text(row.get("_display_sku", row.get(sku_col, "")))
        base_sku = _coerce_text(row.get(sku_col, ""))
        if base_sku == target_sku or row_sku == target_sku:
            continue

        title = _coerce_text(row.get(title_col, ""))
        bullets = _coerce_text(row.get(bullets_col, ""))
        description = _coerce_text(row.get(desc_col, ""))
        compare_text = " \n ".join(part for part in (title, bullets, description) if part)
        if not compare_text.strip():
            continue

        similarity = SequenceMatcher(None, target_text, compare_text).ratio()
        rank_raw = row.get(avg_rank_col) if avg_rank_col else None
        rank_value = _normalize_rank(rank_raw)

        yield _Candidate(
            brand=_coerce_text(row.get(brand_col, "")),
            sku=row_sku or base_sku,
            title=title,
            rank=rank_value,
            similarity=similarity,
        )


def find_similar_competitors(
    sku_data: SKUData,
    limit: int = 5,
    candidate_limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return competitor SKUs similar to the selected client SKU."""

    df = getattr(sku_data, "dataframe", None)
    column_map = getattr(sku_data, "column_map", {})
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if not column_map:
        return []

    client = getattr(sku_data, "client", {}) or {}
    target_title = _coerce_text(client.get("title"))
    target_bullets = _coerce_text(client.get("bullets"))
    target_description = _coerce_text(client.get("description"))
    target_text = " \n ".join(
        part for part in (target_title, target_bullets, target_description) if part
    )
    if not target_text.strip():
        return []

    target_sku = _coerce_text(client.get("sku_original") or client.get("sku"))

    candidates = list(
        _gather_candidates(
            df,
            column_map,
            target_sku=target_sku,
            target_text=target_text,
        )
    )
    if not candidates:
        return []

    candidates.sort(key=lambda item: item.similarity, reverse=True)
    top_candidates = candidates[: max(candidate_limit, limit)]
    top_candidates.sort(key=lambda item: (item.rank, -item.similarity))

    results: List[Dict[str, Any]] = []
    for entry in top_candidates[:limit]:
        results.append(
            {
                "brand": entry.brand,
                "sku": entry.sku,
                "title": entry.title,
                "rank": entry.rank,
                "similarity": entry.similarity,
            }
        )
    return results


__all__ = ["find_similar_competitors"]
