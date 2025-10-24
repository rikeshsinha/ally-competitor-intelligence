"""LangChain runnable that loads SKU data and captures user selections."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from langchain_core.runnables import RunnableLambda

from core.content_rules import dataframe_from_uploaded_file
from .similar_product_chain import create_similar_product_chain


@dataclass
class SKUData:
    dataframe: pd.DataFrame
    column_map: Dict[str, str]
    client: Dict[str, Any]
    competitor: Dict[str, Any]
    similar_candidates: List[Dict[str, Any]] = field(default_factory=list)
    selection_metadata: Dict[str, Any] = field(default_factory=dict)


class _SKUExtractor:
    def __init__(self) -> None:
        self._similar_chain = create_similar_product_chain()

    def __call__(self, inputs: Dict[str, Any]) -> SKUData:  # type: ignore[override]
        csv_file = inputs.get("sku_file")
        if csv_file is None and "uploaded_df" not in st.session_state:
            st.info(
                "Upload a CSV to continue. Expected columns: sku_id/product_id, title, bullets, description, image_urls, brand, category."
            )
            st.stop()
        df = self._load_dataframe(csv_file)
        column_map = self._resolve_columns(df)
        df = self._deduplicate_products(df, column_map)
        self._show_mapping(column_map)

        brands, brand_map = self._build_sku_list(df, column_map)
        if not brands:
            st.warning("No SKU identifiers found in the uploaded file.")
            st.stop()

        selected_client_brand = st.sidebar.selectbox(
            "Select Client Brand", brands
        )
        client_title_idx = self._select_title_for_brand(
            brand_map, selected_client_brand, "Select Client Title"
        )

        client_idx = self._resolve_index(client_title_idx, df, fallback=0)
        client_row = df.iloc[[client_idx]] if len(df) else df.iloc[[]]

        if client_row.empty:
            st.warning("Please select a valid client SKU")
            st.stop()

        similar_candidates, recommended_choice = self._render_similar_dropdown(
            df, column_map, client_idx
        )

        selected_comp_brand = st.sidebar.selectbox(
            "Select Competitor Brand",
            brands,
            index=min(1, len(brands) - 1),
        )
        comp_title_idx = self._select_title_for_brand(
            brand_map,
            selected_comp_brand,
            "Select Competitor Title",
            default_index=min(1, len(brand_map.get(selected_comp_brand, [])) - 1)
            if brand_map.get(selected_comp_brand)
            else 0,
        )

        comp_idx = self._resolve_index(
            comp_title_idx,
            df,
            fallback=min(1, len(df) - 1) if len(df) > 1 else 0,
        )

        selection_metadata: Dict[str, Any] = {
            "source": "manual",
            "recommended_choice_id": None,
            "recommended_reason": None,
            "manual_brand": selected_comp_brand,
            "manual_title_index": comp_title_idx,
        }

        if recommended_choice is not None:
            comp_idx = recommended_choice.get("row_index", comp_idx)
            selection_metadata.update(
                {
                    "source": "recommended",
                    "recommended_choice_id": recommended_choice.get("candidate_id"),
                    "recommended_reason": recommended_choice.get("reason"),
                }
            )

        comp_row = df.iloc[[comp_idx]] if len(df) else df.iloc[[]]

        if comp_row.empty:
            st.warning("Please select valid SKUs")
            st.stop()

        client_data = self._record_from_row(client_row, column_map)
        comp_data = self._record_from_row(comp_row, column_map)

        if recommended_choice is not None:
            comp_data["selection_source"] = "recommended"
            reason = str(recommended_choice.get("reason", "")).strip()
            if reason:
                comp_data["similarity_reason"] = reason
        else:
            comp_data["selection_source"] = "manual"

        return SKUData(
            dataframe=df,
            column_map=column_map,
            client=client_data,
            competitor=comp_data,
            similar_candidates=similar_candidates,
            selection_metadata=selection_metadata,
        )

    def _load_dataframe(self, csv_file) -> pd.DataFrame:
        if csv_file is not None:
            df = dataframe_from_uploaded_file(csv_file)
            st.session_state["uploaded_df"] = df
        else:
            df = st.session_state["uploaded_df"]
        return df.copy()

    def _resolve_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        def pick(*candidates: str, default: Any = "") -> str:
            for cand in candidates:
                key = cand.lower()
                if key in cols_lower:
                    return cols_lower[key]
            df[candidates[0]] = default
            return candidates[0]

        return {
            "sku_col": pick("sku_id", "asin", "sku", "id", "product_id"),
            "title_col": pick("title", "product_title"),
            "bullets_col": pick("bullets", "features", "key_features", "bullet_points"),
            "desc_col": pick("description", "product_description", "desc", "description_filled"),
            "images_col": pick("image_urls", "images", "image", "image_url"),
            "brand_col": pick("brand", "brand_name", "retailer_brand_name"),
            "category_col": pick("category", "node", "retailer_category_node"),
            "universe_col": pick("universe"),
            "avg_rank_search_col": pick("avg_rank_search", default=pd.NA),
            "bundle_size_col": pick("bundle_size", "pack_size", "pack_of", default=""),
        }

    def _show_mapping(self, column_map: Dict[str, str]) -> None:
        with st.expander("Detected column mapping"):
            st.write(dict(column_map))

    def _deduplicate_products(
        self, df: pd.DataFrame, column_map: Dict[str, str]
    ) -> pd.DataFrame:
        if df.empty:
            return df

        sku_col = column_map["sku_col"]
        desc_col = column_map["desc_col"]
        bullets_col = column_map["bullets_col"]
        images_col = column_map["images_col"]
        avg_rank_col = column_map.get("avg_rank_search_col")

        working_df = df.copy()

        working_df["_desc_len"] = (
            working_df[desc_col].fillna("").astype(str).str.len()
        )
        working_df["_bullet_len"] = (
            working_df[bullets_col].fillna("").astype(str).str.len()
        )

        def _count_images(value: Any) -> int:
            if isinstance(value, list):
                iterable = value
            else:
                if pd.isna(value):
                    return 0
                value_str = str(value).strip()
                if not value_str:
                    return 0
                for sep in ("\n", "|", ";", ","):
                    if sep in value_str:
                        parts = [part.strip() for part in value_str.split(sep)]
                        return sum(1 for part in parts if part)
                return 1
            return sum(1 for item in iterable if pd.notna(item) and str(item).strip())

        working_df["_image_count"] = working_df[images_col].apply(_count_images)
        working_df["_original_order"] = range(len(working_df))

        sort_columns: List[str] = [sku_col]
        ascending: List[bool] = [True]

        if avg_rank_col and avg_rank_col in working_df.columns:
            working_df["_has_avg_rank"] = working_df[avg_rank_col].apply(
                lambda value: 1
                if not self._is_missing_avg_rank(value)
                else 0
            )
            sort_columns.append("_has_avg_rank")
            ascending.append(False)

        sort_columns.extend(["_desc_len", "_bullet_len", "_image_count", "_original_order"])
        ascending.extend([False, False, False, True])

        sorted_df = working_df.sort_values(
            by=sort_columns,
            ascending=ascending,
            kind="mergesort",
        )

        deduped = (
            sorted_df.groupby(sku_col, sort=False, group_keys=False).head(1).copy()
        )
        drop_columns = ["_desc_len", "_bullet_len", "_image_count", "_original_order"]
        if "_has_avg_rank" in deduped.columns:
            drop_columns.append("_has_avg_rank")

        deduped = deduped.sort_values("_original_order", kind="mergesort").drop(
            columns=drop_columns
        )
        deduped.reset_index(drop=True, inplace=True)
        return deduped

    def _build_sku_list(
        self, df: pd.DataFrame, column_map: Dict[str, str]
    ) -> tuple[List[str], Dict[str, List[tuple[str, int]]]]:
        sku_col = column_map["sku_col"]
        raw_series = df[sku_col].fillna("")
        sku_series = raw_series.astype(str).str.strip()
        sku_series = sku_series.replace({"nan": "", "NaN": "", "None": ""})
        duplicate_counts: Dict[str, int] = {}
        display_skus: List[str] = []
        for sku_val in sku_series:
            base = sku_val
            label = base if base else "(blank SKU)"
            count = duplicate_counts.get(base, 0)
            display = label if count == 0 else f"{label}_{count}"
            duplicate_counts[base] = count + 1
            display_skus.append(display)
        df["_display_sku"] = display_skus
        brand_col = column_map["brand_col"]
        title_col = column_map["title_col"]

        brand_map: Dict[str, List[tuple[str, int]]] = {}
        brands: List[str] = []
        for idx, row in df.iterrows():
            brand_value = row[brand_col]
            brand_label = "" if pd.isna(brand_value) else str(brand_value).strip()
            if not brand_label:
                brand_label = "(Unspecified Brand)"
            if brand_label not in brand_map:
                brand_map[brand_label] = []
                brands.append(brand_label)

            title_value = row[title_col]
            title_label = "" if pd.isna(title_value) else str(title_value).strip()
            if not title_label:
                title_label = df.at[idx, "_display_sku"]
            brand_map[brand_label].append((title_label, idx))

        return brands, brand_map

    def _select_title_for_brand(
        self,
        brand_map: Dict[str, List[tuple[str, int]]],
        brand: str,
        prompt: str,
        default_index: int = 0,
    ) -> int | None:
        options = brand_map.get(brand, [])
        if not options:
            st.warning(f"No titles found for brand '{brand}'.")
            return None
        counts: Dict[str, int] = {}
        labels: List[str] = []
        for title, _ in options:
            count = counts.get(title, 0)
            label = title if count == 0 else f"{title} ({count + 1})"
            counts[title] = count + 1
            labels.append(label)
        selected_title = st.sidebar.selectbox(
            prompt,
            labels,
            index=min(default_index, len(labels) - 1) if labels else 0,
        )
        selected_idx = labels.index(selected_title)
        return options[selected_idx][1]

    def _record_from_row(self, row: pd.DataFrame, column_map: Dict[str, str]) -> Dict[str, Any]:
        sku_display = row.iloc[0]["_display_sku"] if "_display_sku" in row.columns else ""
        sku_original_series = row[column_map["sku_col"]].astype(str)
        sku_original = sku_original_series.iloc[0] if not sku_original_series.empty else ""
        record = {
            "sku": sku_display,
            "sku_original": sku_original,
            "title": row.iloc[0][column_map["title_col"]],
            "bullets": row.iloc[0][column_map["bullets_col"]],
            "description": row.iloc[0][column_map["desc_col"]],
            "image_urls": row.iloc[0][column_map["images_col"]],
            "brand": row.iloc[0][column_map["brand_col"]],
            "category": row.iloc[0][column_map["category_col"]],
        }
        uni_col = column_map.get("universe_col")
        if uni_col in row.columns:
            record["universe"] = row.iloc[0][uni_col]
        else:
            record["universe"] = None
        avg_rank_col = column_map.get("avg_rank_search_col")
        if avg_rank_col in row.columns:
            record["avg_rank_search"] = self._coerce_avg_rank(row.iloc[0][avg_rank_col])
        else:
            record["avg_rank_search"] = None
        bundle_col = column_map.get("bundle_size_col")
        if bundle_col in row.columns:
            record["bundle_size"] = row.iloc[0][bundle_col]
        else:
            record["bundle_size"] = None
        return record

    def _coerce_avg_rank(self, value: Any) -> Optional[float]:
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

    def _is_missing_avg_rank(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _resolve_index(
        self, selected_idx: Optional[int], df: pd.DataFrame, fallback: int = 0
    ) -> int:
        if df.empty:
            return 0
        if selected_idx is None:
            target = fallback
        else:
            try:
                target = int(selected_idx)
            except (TypeError, ValueError):
                target = fallback
        target = max(0, min(target, len(df) - 1))
        return target

    def _render_similar_dropdown(
        self, df: pd.DataFrame, column_map: Dict[str, str], client_idx: int
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if df.empty:
            return [], None

        try:
            raw_results = self._similar_chain.invoke(
                {
                    "dataframe": df,
                    "column_map": column_map,
                    "selected_index": client_idx,
                }
            )
        except Exception as exc:
            st.sidebar.warning(f"Similar product suggestion failed: {exc}")
            return [], None

        if not raw_results:
            return [], None

        def _format_option(idx: int) -> str:
            if idx == -1:
                return "Choose competitor manually"
            entry = raw_results[idx]
            title = str(entry.get("title", "")).strip() or "(Untitled)"
            brand = str(entry.get("brand", "")).strip() or "Unknown"
            avg_rank = entry.get("avg_rank_search")
            if isinstance(avg_rank, (int, float)) and not pd.isna(avg_rank):
                rank_display = (
                    f"avg rank {avg_rank:.2f}" if isinstance(avg_rank, float) else f"avg rank {avg_rank}"
                )
            else:
                rank_display = "avg rank N/A"
            return f"{title} — {brand} ({rank_display})"

        option_indices = list(range(len(raw_results)))
        selected_option = st.sidebar.selectbox(
            "Similar competitor suggestions",
            [-1] + option_indices,
            format_func=_format_option,
        )

        chosen_entry: Optional[Dict[str, Any]] = None
        if selected_option != -1:
            chosen_entry = raw_results[selected_option]
            reason = str(chosen_entry.get("reason", "")).strip()
            if reason:
                st.sidebar.caption(f"Why suggested: {reason}")

        return raw_results, chosen_entry


def create_sku_extractor() -> RunnableLambda:
    """Return a runnable instance for SKU extraction."""
    return RunnableLambda(_SKUExtractor())
