import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chains.competitor_finder import find_similar_competitors
from chains.sku_extractor import SKUData


def test_find_similar_competitors_limits_and_ordering():
    data = [
        {
            "sku_id": "CLIENT1",
            "_display_sku": "CLIENT1",
            "title": "Premium Salmon Dog Food",
            "bullets": "High protein | Grain free",
            "description": "A balanced salmon recipe for adult dogs.",
            "brand": "ClientBrand",
            "category": "Food",
            "images": None,
            "avg_rank_search": "",
        },
        {
            "sku_id": "COMP1",
            "_display_sku": "SKU-1",
            "title": "Salmon Recipe Dog Food",
            "bullets": "High protein | Grain free",
            "description": "Balanced salmon dog food with omega-3s.",
            "brand": "BrandA",
            "category": "Food",
            "images": None,
            "avg_rank_search": "2",
        },
        {
            "sku_id": "COMP2",
            "_display_sku": "SKU-2",
            "title": "Premium Salmon & Sweet Potato Dog Food",
            "bullets": "Grain free | Added vitamins",
            "description": "Crafted salmon kibble for adult dogs.",
            "brand": "BrandB",
            "category": "Food",
            "images": None,
            "avg_rank_search": "1",
        },
        {
            "sku_id": "COMP3",
            "_display_sku": "SKU-3",
            "title": "Chicken and Salmon Blend",
            "bullets": "High protein | Added probiotics",
            "description": "Chicken and salmon mix for dogs.",
            "brand": "BrandC",
            "category": "Food",
            "images": None,
            "avg_rank_search": "5",
        },
        {
            "sku_id": "COMP4",
            "_display_sku": "SKU-4",
            "title": "Adult Dog Salmon Dinner",
            "bullets": "Protein rich | Balanced",
            "description": "Adult dog salmon dinner with vitamins.",
            "brand": "BrandD",
            "category": "Food",
            "images": None,
            "avg_rank_search": None,
        },
        {
            "sku_id": "COMP5",
            "_display_sku": "SKU-5",
            "title": "Cat Toy",
            "bullets": "Feather teaser",
            "description": "Interactive cat feather toy.",
            "brand": "BrandE",
            "category": "Toys",
            "images": None,
            "avg_rank_search": "invalid",
        },
        {
            "sku_id": "COMP6",
            "_display_sku": "SKU-6",
            "title": "Salmon and Brown Rice Dog Food",
            "bullets": "High protein | Gentle grains",
            "description": "Salmon dog food with brown rice.",
            "brand": "BrandF",
            "category": "Food",
            "images": None,
            "avg_rank_search": "3",
        },
    ]

    df = pd.DataFrame(data)
    column_map = {
        "sku_col": "sku_id",
        "title_col": "title",
        "bullets_col": "bullets",
        "desc_col": "description",
        "images_col": "images",
        "brand_col": "brand",
        "category_col": "category",
        "avg_rank_col": "avg_rank_search",
    }

    client_record = {
        "sku": "CLIENT1",
        "sku_original": "CLIENT1",
        "title": data[0]["title"],
        "bullets": data[0]["bullets"],
        "description": data[0]["description"],
        "brand": data[0]["brand"],
        "category": data[0]["category"],
        "avg_rank_search": data[0]["avg_rank_search"],
    }

    sku_data = SKUData(
        dataframe=df,
        column_map=column_map,
        client=client_record,
        competitor={},
    )

    results = find_similar_competitors(sku_data)

    assert len(results) == 5
    skus = [item["sku"] for item in results]
    assert skus == ["SKU-2", "SKU-1", "SKU-6", "SKU-3", "SKU-4"]

    ranks = [item["rank"] for item in results]
    assert ranks[:4] == [1.0, 2.0, 3.0, 5.0]
    assert ranks[-1] == float("inf")
