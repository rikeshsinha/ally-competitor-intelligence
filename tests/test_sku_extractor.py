import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chains import sku_extractor


class _StubStreamlit:
    def __init__(self) -> None:
        self.session_state: Dict[str, Any] = {}

    def info(self, *_: Any, **__: Any) -> None:  # pragma: no cover - noop for tests
        return None

    def warning(self, *_: Any, **__: Any) -> None:  # pragma: no cover - noop for tests
        return None

    def stop(self) -> None:  # pragma: no cover - fail fast if called unexpectedly
        raise RuntimeError("st.stop should not be called in tests")


@pytest.fixture
def stub_streamlit(monkeypatch: pytest.MonkeyPatch) -> _StubStreamlit:
    stub = _StubStreamlit()
    monkeypatch.setattr(sku_extractor, "st", stub)
    return stub


def test_selected_rows_use_dataframe_labels(stub_streamlit: _StubStreamlit) -> None:
    extractor = sku_extractor._SKUExtractor()

    data = [
        {
            "sku_id": "OTHER-000",
            "title": "Other Item",
            "bullets": "",
            "description": "",
            "image_urls": "",
            "brand": "OtherBrand",
            "category": "Food",
            "avg_rank_search": "",
            "universe": "US",
        },
        {
            "sku_id": "CLIENT-002",
            "title": "Client Selection",
            "bullets": "Client bullet",
            "description": "Client description",
            "image_urls": "client.jpg",
            "brand": "ClientBrand",
            "category": "Food",
            "avg_rank_search": "1",
            "universe": "US",
        },
        {
            "sku_id": "COMP-003",
            "title": "Competitor (other)",
            "bullets": "",
            "description": "",
            "image_urls": "",
            "brand": "CompetitorBrand",
            "category": "Food",
            "avg_rank_search": "2",
            "universe": "US",
        },
        {
            "sku_id": "OTHER-004",
            "title": "Another Item",
            "bullets": "",
            "description": "",
            "image_urls": "",
            "brand": "OtherBrand",
            "category": "Food",
            "avg_rank_search": "",
            "universe": "US",
        },
        {
            "sku_id": "COMP-005",
            "title": "Competitor Selection",
            "bullets": "",
            "description": "",
            "image_urls": "",
            "brand": "CompetitorBrand",
            "category": "Food",
            "avg_rank_search": "3",
            "universe": "US",
        },
        {
            "sku_id": "EXTRA-006",
            "title": "Extra Item",
            "bullets": "",
            "description": "",
            "image_urls": "",
            "brand": "ExtraBrand",
            "category": "Food",
            "avg_rank_search": "",
            "universe": "US",
        },
    ]

    df = pd.DataFrame(data, index=[0, 2, 3, 4, 5, 6])

    column_map = {
        "sku_col": "sku_id",
        "title_col": "title",
        "bullets_col": "bullets",
        "desc_col": "description",
        "images_col": "image_urls",
        "brand_col": "brand",
        "category_col": "category",
        "avg_rank_col": "avg_rank_search",
        "universe_col": "universe",
    }

    brands, brand_map = extractor._build_sku_list(df, column_map)
    competitor_brands = [b for b in brands if b != "ClientBrand"]
    brand_groups, competitor_options = extractor._build_competitor_catalog(
        competitor_brands, brand_map
    )
    version_key = extractor._competitor_version_key("ClientBrand", competitor_options)

    stub_streamlit.session_state.update(
        {
            "uploaded_df": df,
            extractor._STATE_KEY: {
                "dataframe": df,
                "column_map": column_map,
                "brand_map": brand_map,
                "brands": brands,
                "competitor_options": competitor_options,
                "client_row_index": None,
                "mode_snapshot": "manual",
            },
            "selected_client": {"brand": "ClientBrand", "row_index": 2},
            "selected_competitor": {"brand": "CompetitorBrand", "row_index": 5},
            "competitor_selection_mode": "manual",
            "competitor_choices": {
                "client_brand": "ClientBrand",
                "options": competitor_options,
                "brand_groups": brand_groups,
                "version": version_key,
            },
        }
    )

    result = extractor({"sku_file": None})

    assert result.client["sku_original"] == "CLIENT-002"
    assert result.client["title"] == "Client Selection"
    assert result.competitor["sku_original"] == "COMP-005"
    assert result.competitor["title"] == "Competitor Selection"
