"""LangChain runnable that loads SKU data and captures user selections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from langchain_core.runnables import RunnableLambda

from core.content_rules import dataframe_from_uploaded_file


@dataclass
class SKUData:
    dataframe: pd.DataFrame
    column_map: Dict[str, str]
    client: Dict[str, Any]
    competitor: Dict[str, Any]
    available_universes: List[str]


class _SKUExtractor:
    def __call__(self, inputs: Dict[str, Any]) -> SKUData:  # type: ignore[override]
        csv_file = inputs.get("sku_file")
        if csv_file is None and "uploaded_df" not in st.session_state:
            st.info(
                "Upload a CSV to continue. Expected columns: sku_id/product_id, title, bullets, description, image_urls, brand, category."
            )
            st.stop()
        df = self._load_dataframe(csv_file)
        column_map = self._resolve_columns(df)
        available_universes = self._available_universes(df, column_map["universe_col"])
        self._show_mapping(column_map, available_universes)

        sku_list, display_to_index = self._build_sku_list(df, column_map["sku_col"])
        if not sku_list:
            st.warning("No SKU identifiers found in the uploaded file.")
            st.stop()

        selected_client = st.sidebar.selectbox("Select Client SKU", sku_list)
        selected_comp = st.sidebar.selectbox(
            "Select Competitor SKU",
            sku_list,
            index=min(1, len(sku_list) - 1),
        )

        client_idx = display_to_index.get(selected_client, 0)
        comp_idx = display_to_index.get(selected_comp, min(1, len(sku_list) - 1))

        client_row = df.iloc[[client_idx]] if len(df) else df.iloc[[]]
        comp_row = df.iloc[[comp_idx]] if len(df) else df.iloc[[]]

        if client_row.empty or comp_row.empty:
            st.warning("Please select valid SKUs")
            st.stop()

        client_data = self._record_from_row(client_row, column_map)
        comp_data = self._record_from_row(comp_row, column_map)

        return SKUData(
            dataframe=df,
            column_map=column_map,
            client=client_data,
            competitor=comp_data,
            available_universes=available_universes,
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

        def pick(*candidates: str, default: str = "") -> str:
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
        }

    def _available_universes(self, df: pd.DataFrame, universe_col: str) -> List[str]:
        if universe_col not in df.columns:
            return []
        raw_unis = df[universe_col].astype(str).fillna("").str.strip()
        universes = sorted({u.title() for u in raw_unis if u and u.lower() != "nan"})
        return universes

    def _show_mapping(self, column_map: Dict[str, str], universes: List[str]) -> None:
        with st.expander("Detected column mapping"):
            mapping = dict(column_map)
            mapping["available_universes"] = universes[:20]
            st.write(mapping)

    def _build_sku_list(
        self, df: pd.DataFrame, sku_col: str
    ) -> tuple[List[str], Dict[str, int]]:
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
        display_to_index = {label: pos for pos, label in enumerate(display_skus)}
        return display_skus, display_to_index

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
        return record


def create_sku_extractor() -> RunnableLambda:
    """Return a runnable instance for SKU extraction."""
    return RunnableLambda(_SKUExtractor())
