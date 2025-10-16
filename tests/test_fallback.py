import re
import sys
import types
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide lightweight stubs for optional dependencies to load app
langgraph_module = types.ModuleType("langgraph")
langgraph_graph_module = types.ModuleType("langgraph.graph")
langgraph_graph_module.StateGraph = object  # type: ignore[attr-defined]
langgraph_graph_module.END = object()
langgraph_module.graph = langgraph_graph_module  # type: ignore[attr-defined]
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)

graph_module = types.ModuleType("graph")
graph_product_module = types.ModuleType("graph.product_validation")


class _DummyGraph:
    def invoke(self, *_, **__):
        rules = {
            "title": {
                "max_chars": 200,
                "brand_required": False,
                "no_all_caps": True,
                "no_promo": True,
            },
            "bullets": {
                "max_count": 5,
                "start_capital": True,
                "no_end_punct": True,
                "no_promo_or_seller_info": True,
            },
            "description": {
                "max_chars": 400,
                "no_promo": True,
                "sentence_caps": True,
            },
            "images": {},
        }
        dummy_sku = types.SimpleNamespace(
            dataframe=None,
            column_map={},
            brands=["Acme", "Rival"],
            brand_map={
                "Acme": [("Client Product", 0)],
                "Rival": [("Competitor Product", 1)],
            },
            client_selection=types.SimpleNamespace(
                brand="Acme", title_index=0, row_index=0
            ),
            competitor_selection=types.SimpleNamespace(
                brand="Rival", title_index=0, row_index=1
            ),
            client={"sku": "CLIENT-1", "brand": "Acme", "title": "Client Product"},
            competitor={
                "sku": "RIVAL-1",
                "brand": "Rival",
                "title": "Competitor Product",
            },
        )
        return {
            "sku_data": dummy_sku,
            "rule_data": types.SimpleNamespace(rules=rules, source="tests", messages=[]),
            "validation": {
                "title": {"client_score": 0, "issues": []},
                "bullets": {"client_score": 0, "issues": []},
                "description": {"client_score": 0, "issues": []},
                "images": {"client_count": 0, "comp_count": 0, "issues": []},
                "gaps_vs_competitor": [],
            },
        }


graph_product_module.build_product_validation_graph = lambda *_, **__: _DummyGraph()
graph_module.product_validation = graph_product_module  # type: ignore[attr-defined]
sys.modules.setdefault("graph", graph_module)
sys.modules.setdefault("graph.product_validation", graph_product_module)

import app


def _call_fallback(client_data):
    """Invoke call_llm with fallback path forced."""
    # Ensure fallback path by forcing get_openai_client to return None
    original_get_client = app.get_openai_client
    app.get_openai_client = lambda: None  # type: ignore
    try:
        return app.call_llm(
            client_data=client_data,
            comp_data={},
            rules={
                "title": {"max_chars": 200},
                "bullets": {"max_count": 5},
                "description": {"max_chars": 400},
            },
        )
    finally:
        app.get_openai_client = original_get_client  # type: ignore


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _assert_from_sources(output_bullets: List[str], source_bullets: str, source_description: str):
    combined_source = _normalize(source_bullets + " " + source_description)
    for bullet in output_bullets:
        assert _normalize(bullet) in combined_source


def test_fallback_reuses_client_bullets_and_description_clauses():
    client_data = {
        "brand": "Acme Pets",
        "title": "Orthopedic Dog Bed",
        "bullets": "soft cushioning.|Waterproof lining ;  removable cover",
        "description": "Machine washable cover. Provides joint support for older dogs!",
    }
    response = _call_fallback(client_data)

    expected_original_bullets = {
        "soft cushioning",
        "waterproof lining",
        "removable cover",
    }

    normalized_output = {_normalize(b) for b in response["bullets_edits"]}

    # Original bullets must be preserved after normalization
    assert expected_original_bullets.issubset(normalized_output)
    # Additional bullets must come from the client description
    _assert_from_sources(response["bullets_edits"], client_data["bullets"], client_data["description"])

    # Description must derive from client description text
    if response["description_edit"]:
        assert _normalize(response["description_edit"]) in _normalize(client_data["description"])


def test_fallback_handles_missing_bullets_with_description():
    client_data = {
        "brand": "Acme Pets",
        "title": "Cat Tunnel",
        "bullets": "",
        "description": "Durable ripstop material keeps its shape. Collapsible design for easy storage.",
    }
    response = _call_fallback(client_data)

    assert response["bullets_edits"], "Expected bullets extracted from description"
    _assert_from_sources(response["bullets_edits"], client_data["bullets"], client_data["description"])

    if response["description_edit"]:
        assert _normalize(response["description_edit"]) in _normalize(client_data["description"])


def test_fallback_empty_inputs_return_no_new_claims():
    client_data = {
        "brand": "Acme Pets",
        "title": "Pet Harness",
        "bullets": "",
        "description": "",
    }
    response = _call_fallback(client_data)

    assert response["bullets_edits"] == []
    assert response["description_edit"] == ""
