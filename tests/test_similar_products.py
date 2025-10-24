import pandas as pd

from chains.similar_products import create_similar_products_chain, SimilarProductsResult


def test_similar_products_heuristic_fallback(monkeypatch):
    monkeypatch.setattr("chains.similar_products.OpenAI", None, raising=False)

    data = [
        {
            "sku": "A1",
            "title": "Ultra Clean Wipes Pack of 3",
            "description": "Set of 3 antibacterial wipes for pets",
            "brand": "Alpha",
            "bullets": "",
            "images": "",
            "category": "Cleaning",
            "avg_rank_search": 5,
            "universe": "Online",
        },
        {
            "sku": "B1",
            "title": "Ultra Clean Wipes Pack of 3",
            "description": "Pack of 3 scented wipes",
            "brand": "Beta",
            "bullets": "",
            "images": "",
            "category": "Cleaning",
            "avg_rank_search": 12,
            "universe": "Online",
        },
        {
            "sku": "B2",
            "title": "Ultra Clean Wipes Pack of 6",
            "description": "Bulk pack of 6 wipes for multi-pet homes",
            "brand": "Beta",
            "bullets": "",
            "images": "",
            "category": "Cleaning",
            "avg_rank_search": 18,
            "universe": "Online",
        },
        {
            "sku": "G1",
            "title": "Kitchen Sponges 4 pack",
            "description": "Set of 4 absorbent kitchen sponges",
            "brand": "Gamma",
            "bullets": "",
            "images": "",
            "category": "Kitchen",
            "avg_rank_search": 22,
            "universe": "Retail",
        },
    ]
    df = pd.DataFrame(data)
    column_map = {
        "sku_col": "sku",
        "title_col": "title",
        "desc_col": "description",
        "brand_col": "brand",
        "bullets_col": "bullets",
        "images_col": "images",
        "category_col": "category",
        "avg_rank_col": "avg_rank_search",
        "universe_col": "universe",
    }
    client_record = {
        "sku": "A1",
        "sku_original": "A1",
        "title": data[0]["title"],
        "description": data[0]["description"],
        "bullets": "",
        "brand": "Alpha",
        "category": "Cleaning",
        "image_urls": "",
        "avg_rank_search": 5,
        "universe": "Online",
    }

    chain = create_similar_products_chain()
    result = chain.invoke(
        {"dataframe": df, "column_map": column_map, "client": client_record}
    )

    assert isinstance(result, SimilarProductsResult)
    assert result.matches
    assert not result.using_llm
    assert result.message and "heuristic" in result.message.lower()

    top_match = result.matches[0]
    assert top_match.sku == "B1"
    assert top_match.bundle_size == 3
    assert top_match.reason and "bundle" in top_match.reason.lower()

    skus = [m.sku for m in result.matches]
    assert skus.count("A1") == 0
    assert len(result.matches) <= 5
