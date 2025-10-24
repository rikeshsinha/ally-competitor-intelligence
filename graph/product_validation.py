"""LangGraph orchestration for the product validation workflow."""
from __future__ import annotations

from typing import Any, Callable, Dict, TypedDict

from langgraph.graph import END, StateGraph

from chains.rule_extractor import RuleExtraction, create_rule_extractor
from chains.similar_products import (
    SimilarProductsResult,
    create_similar_products_chain,
)
from chains.sku_extractor import SKUData, create_sku_extractor


class ProductValidationState(TypedDict, total=False):
    sku_file: Any
    rules_file: Any
    sku_data: SKUData
    rule_data: RuleExtraction
    validation: Dict[str, Any]
    similar_products: SimilarProductsResult


def build_product_validation_graph(
    validation_fn: Callable[[SKUData, RuleExtraction], Dict[str, Any]]
):
    sku_chain = create_sku_extractor()
    rule_chain = create_rule_extractor()
    similar_chain = create_similar_products_chain()

    def run_sku(state: ProductValidationState) -> Dict[str, Any]:
        result = sku_chain.invoke({"sku_file": state.get("sku_file")})
        return {"sku_data": result}

    def run_similar(state: ProductValidationState) -> Dict[str, Any]:
        sku_data = state["sku_data"]
        result = similar_chain.invoke(
            {
                "dataframe": sku_data.dataframe,
                "column_map": sku_data.column_map,
                "client": sku_data.client,
            }
        )
        sku_data.similar_competitors = list(result.matches)
        return {"sku_data": sku_data, "similar_products": result}

    def run_rules(state: ProductValidationState) -> Dict[str, Any]:
        result = rule_chain.invoke({"rules_file": state.get("rules_file")})
        return {"rule_data": result}

    def run_validation(state: ProductValidationState) -> Dict[str, Any]:
        sku_data = state["sku_data"]
        rule_data = state["rule_data"]
        summary = validation_fn(sku_data, rule_data)
        return {"validation": summary}

    graph = StateGraph(ProductValidationState)
    graph.add_node("sku_extraction", run_sku)
    graph.add_node("rule_extraction", run_rules)
    graph.add_node("similar_products", run_similar)
    graph.add_node("validation", run_validation)

    graph.set_entry_point("sku_extraction")
    graph.add_edge("sku_extraction", "similar_products")
    graph.add_edge("similar_products", "rule_extraction")
    graph.add_edge("rule_extraction", "validation")
    graph.add_edge("validation", END)

    return graph.compile()
