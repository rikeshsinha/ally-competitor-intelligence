"""LangChain runnable that loads SKU data and captures user selections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


class _SKUExtractor:
    _STATE_KEY = "_sku_extractor_state"

    def __call__(self, inputs: Dict[str, Any]) -> SKUData:  # type: ignore[override]
        csv_file = inputs.get("sku_file")
        if csv_file is None and "uploaded_df" not in st.session_state:
            st.info(
                "Upload a CSV to continue. Expected columns: sku_id/product_id, title, bullets, description, image_urls, brand, category."
            )
            st.stop()

        state = st.session_state.get(self._STATE_KEY)
        if state is None:
            state = self._prime_state(csv_file)

        if state is None:
            st.stop()

        df: pd.DataFrame = state["dataframe"]
        column_map: Dict[str, str] = state["column_map"]
        competitor_options: List[Dict[str, Any]] = state["competitor_options"]
        client_idx: int = state["client_row_index"]

        selection = st.session_state.get("selected_competitor")
        matched_option = self._match_competitor_selection(
            selection, competitor_options
        )
        if matched_option is None:
            st.stop()

        comp_idx = matched_option["row_index"]

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
        )

    def _prime_state(self, csv_file) -> Optional[Dict[str, Any]]:
        df = self._load_dataframe(csv_file)
        column_map = self._resolve_columns(df)
        df = self._deduplicate_products(df, column_map)
        self._show_mapping(column_map)

        brands, brand_map = self._build_sku_list(df, column_map)
        if not brands:
            st.warning("No SKU identifiers found in the uploaded file.")
            st.stop()

        selected_client_brand = st.sidebar.selectbox(
            "Select Client Brand",
            brands,
            index=None,
            placeholder="Choose a client brand",
        )
        if selected_client_brand is None:
            st.sidebar.info("Select a client brand to continue.")
            st.stop()

        client_title_idx = self._select_title_for_brand(
            brand_map,
            selected_client_brand,
            "Select Client Title",
            default_index=None,
        )
        if client_title_idx is None:
            st.sidebar.info("Select a client product to continue.")
            st.stop()

        competitor_brands = [b for b in brands if b != selected_client_brand]
        if not competitor_brands:
            st.sidebar.warning("No competitor brands available for comparison.")
            st.stop()

        brand_groups, competitor_options = self._build_competitor_catalog(
            competitor_brands, brand_map
        )
        if not competitor_options:
            st.sidebar.warning("No competitor SKUs available for comparison.")
            st.stop()

        version_key = self._competitor_version_key(
            selected_client_brand, competitor_options
        )
        previous_state = st.session_state.get("competitor_choices")
        previous_version: Optional[str] = None
        if isinstance(previous_state, dict):
            previous_version = previous_state.get("version")
        if previous_version != version_key:
            for key in (
                "selected_competitor",
                "selected_competitor_brand",
                "competitor_choices",
                "competitor_chat_log",
                "competitor_chat_confirmed",
                "competitor_chat_rendered_version",
                "competitor_product_prompt_brand",
                "competitor_brand_user_ack",
                "competitor_product_user_ack",
                "competitor_product_select_key",
            ):
                st.session_state.pop(key, None)

        st.session_state["competitor_choices"] = {
            "client_brand": selected_client_brand,
            "options": competitor_options,
            "brand_groups": brand_groups,
            "version": version_key,
        }

        client_idx = client_title_idx if client_title_idx is not None else 0
        state = {
            "dataframe": df,
            "column_map": column_map,
            "competitor_options": competitor_options,
            "client_row_index": client_idx,
        }
        st.session_state[self._STATE_KEY] = state
        return state

    def _load_dataframe(self, csv_file) -> pd.DataFrame:
        if csv_file is not None:
            file_signature = (
                getattr(csv_file, "name", None),
                getattr(csv_file, "size", None),
            )
            if st.session_state.get("_uploaded_csv_signature") != file_signature:
                st.session_state["_uploaded_csv_signature"] = file_signature
                for key in (
                    "selected_competitor",
                    "selected_competitor_brand",
                    "competitor_choices",
                    "competitor_chat_log",
                    "competitor_chat_confirmed",
                    "competitor_chat_rendered_version",
                    "competitor_product_prompt_brand",
                    "competitor_brand_user_ack",
                    "competitor_product_user_ack",
                    "competitor_product_select_key",
                    self._STATE_KEY,
                ):
                    st.session_state.pop(key, None)
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
            "avg_rank_col": pick("avg_rank_search", default=pd.NA),
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
        avg_rank_col = column_map.get("avg_rank_col")

        working_df = df.copy()

        working_df["_desc_len"] = (
            working_df[desc_col].fillna("").astype(str).str.len()
        )
        working_df["_bullet_len"] = (
            working_df[bullets_col].fillna("").astype(str).str.len()
        )

        if avg_rank_col and avg_rank_col in working_df.columns:
            avg_rank_numeric = pd.to_numeric(
                working_df[avg_rank_col], errors="coerce"
            )
        else:
            avg_rank_numeric = pd.Series(
                float("nan"), index=working_df.index, dtype="float64"
            )

        working_df["_avg_rank_value"] = avg_rank_numeric
        working_df["_has_avg_rank"] = avg_rank_numeric.notna()

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

        sorted_df = working_df.sort_values(
            by=[
                sku_col,
                "_has_avg_rank",
                "_avg_rank_value",
                "_desc_len",
                "_bullet_len",
                "_image_count",
                "_original_order",
            ],
            ascending=[True, False, True, False, False, False, True],
            kind="mergesort",
        )

        deduped = (
            sorted_df.groupby(sku_col, sort=False, group_keys=False).head(1).copy()
        )
        deduped = deduped.sort_values("_original_order", kind="mergesort").drop(
            columns=
            [
                "_desc_len",
                "_bullet_len",
                "_image_count",
                "_original_order",
                "_avg_rank_value",
                "_has_avg_rank",
            ]
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
        default_index: Optional[int] = 0,
    ) -> int | None:
        option_records = self._options_for_brand(brand_map, brand)
        if not option_records:
            st.warning(f"No titles found for brand '{brand}'.")
            return None
        labels = [record["title_label"] for record in option_records]
        selectbox_kwargs: Dict[str, Any] = {}
        if default_index is None:
            selectbox_kwargs.update({"index": None, "placeholder": "Choose a client product"})
        else:
            selectbox_kwargs["index"] = min(default_index, len(labels) - 1) if labels else 0

        selected_title = st.sidebar.selectbox(
            prompt,
            labels,
            **selectbox_kwargs,
        )
        if selected_title is None:
            return None
        selected_idx = labels.index(selected_title)
        return option_records[selected_idx]["row_index"]

    def _options_for_brand(
        self, brand_map: Dict[str, List[tuple[str, int]]], brand: str
    ) -> List[Dict[str, Any]]:
        raw_options = brand_map.get(brand, [])
        if not raw_options:
            return []
        counts: Dict[str, int] = {}
        options: List[Dict[str, Any]] = []
        for title_value, row_index in raw_options:
            base_title = str(title_value)
            count = counts.get(base_title, 0)
            label = base_title if count == 0 else f"{base_title} ({count + 1})"
            counts[base_title] = count + 1
            options.append(
                {
                    "title_value": base_title,
                    "title_label": label,
                    "row_index": row_index,
                }
            )
        return options

    def _build_competitor_catalog(
        self,
        competitor_brands: List[str],
        brand_map: Dict[str, List[tuple[str, int]]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        brand_groups: List[Dict[str, Any]] = []
        flat_options: List[Dict[str, Any]] = []
        global_counter = 1
        for brand_idx, brand in enumerate(competitor_brands, start=1):
            brand_options: List[Dict[str, Any]] = []
            for option_idx, record in enumerate(
                self._options_for_brand(brand_map, brand), start=1
            ):
                option = {
                    "brand": brand,
                    "title_label": record["title_label"],
                    "title_value": record["title_value"],
                    "row_index": record["row_index"],
                    "ordinal": global_counter,
                    "brand_option_ordinal": option_idx,
                }
                brand_options.append(option)
                flat_options.append(option)
                global_counter += 1
            if brand_options:
                brand_groups.append(
                    {
                        "brand": brand,
                        "ordinal": brand_idx,
                        "options": brand_options,
                    }
                )
        return brand_groups, flat_options

    def _competitor_version_key(
        self, client_brand: str, options: List[Dict[str, Any]]
    ) -> str:
        parts = [client_brand]
        parts.extend(f"{opt['brand']}::{opt['row_index']}" for opt in options)
        return "|".join(parts)

    def _match_competitor_selection(
        self,
        selection: Any,
        options: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(selection, dict):
            return None
        row_index = selection.get("row_index")
        brand = selection.get("brand")
        if row_index is None or brand is None:
            return None
        for option in options:
            if option["row_index"] == row_index and option["brand"] == brand:
                return option
        return None

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
            "avg_rank_search": row.iloc[0][column_map["avg_rank_col"]],
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


def prime_competitor_chat_state(csv_file) -> None:
    """Ensure SKU data and competitor options are ready for chat selection."""
    if csv_file is None and "uploaded_df" not in st.session_state:
        return
    extractor = _SKUExtractor()
    extractor._prime_state(csv_file)
