from __future__ import annotations

import pyarrow.parquet as pq

from bmscientist.graph_enrichment import (
    APPLICATION_NODE_SCHEMA,
    MARKET_APPLICATION_SCHEMA,
    MARKET_NODE_SCHEMA,
    empty_row,
    write_rows,
)
from bmscientist.graph_estimate import GraphEstimateAgent
from bmscientist.graph_query import DuckDBGraphQueryEngine


class EstimateLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.0):
        assert "Relevant graph evidence rows" in user_prompt
        return response_model.model_validate(
            {
                "market_name": "medical packaging",
                "application_name": "medical trays",
                "total_substrate_volume_value": 80000,
                "total_substrate_volume_unit": "metric_tons_per_year",
                "volume_year": 2026,
                "revenue_value": 1200,
                "revenue_unit": "USD million",
                "revenue_year": 2026,
                "confidence": 0.54,
                "rationale": "Estimated from tray market revenue and implied substrate pricing in graph evidence.",
                "material_volumes": [
                    {
                        "material_name": "PETG",
                        "volume_value": 44000,
                        "volume_unit": "metric_tons_per_year",
                        "share_of_total": 0.55,
                        "confidence": 0.5,
                        "rationale": "PETG is estimated as the leading tray substrate.",
                    },
                    {
                        "material_name": "PVC",
                        "volume_value": 4000,
                        "volume_unit": "metric_tons_per_year",
                        "share_of_total": 0.05,
                        "confidence": 0.44,
                        "rationale": "PVC is estimated as a smaller legacy tray substrate.",
                    },
                ],
                "source_citations": [
                    {
                        "chunk_id": "graph:Market_HAS_APPLICATION_Application:edge-1",
                        "source_url": "https://example.com/stats",
                        "source_title": "Offline graph market data",
                    }
                ],
            }
        )


def test_graph_estimate_agent_persists_ai_market_volume_estimate(tmp_path):
    graph_path = tmp_path / "graph"
    write_rows(
        graph_path / "nodes" / "Market.parquet",
        [
            {
                "market_id": "market:medical-packaging",
                "name": "medical packaging",
                "normalized_name": "medical packaging",
                "primary_slug": "medical-packaging-market",
                "source_vendor": "test",
            }
        ],
        MARKET_NODE_SCHEMA,
    )
    write_rows(
        graph_path / "nodes" / "Application.parquet",
        [
            {
                "application_id": "application:medical-trays",
                "name": "medical trays",
                "normalized_name": "medical trays",
                "node_type": "application",
            }
        ],
        APPLICATION_NODE_SCHEMA,
    )
    edge = empty_row(MARKET_APPLICATION_SCHEMA)
    edge.update(
        {
            "edge_id": "edge-1",
            "market_id": "market:medical-packaging",
            "application_id": "application:medical-trays",
            "scope_type": "statistics",
            "source_node_type": "application",
            "page_url": "https://example.com/stats",
            "target_url": "https://example.com/stats",
            "status": "fetched",
            "revenue_value": 1200.0,
            "revenue_year": 2026,
            "unit": "USD million",
            "source_url": "https://example.com/stats",
            "source_title": "Offline graph market data",
            "supporting_quote": "Medical tray market revenue was estimated at $1.2B.",
            "confidence": 0.7,
            "validation_status": "accepted",
        }
    )
    write_rows(graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet", [edge], MARKET_APPLICATION_SCHEMA)

    agent = GraphEstimateAgent(EstimateLLM(), DuckDBGraphQueryEngine(graph_path))
    result = agent.run(
        "Estimate the market share percentage for different materials in thermoformed sterile medical trays and back into annual tonnage.",
        persist=True,
    )

    assert result.persisted is True
    assert result.persisted_rows == 3
    assert result.matched_entity_buckets.applications[0].name == "medical trays"
    assert result.estimate.total_substrate_volume_value == 80000

    product_edges = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
    volumes_by_product = {edge["product_id"]: edge["volume_value"] for edge in product_edges}
    assert volumes_by_product["product:petg"] == 44000
    assert volumes_by_product["product:pvc"] == 4000

    market_edges = pq.read_table(graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet").to_pylist()
    ai_edges = [edge for edge in market_edges if edge["source_node_type"] == "ai_volume_estimate"]
    assert ai_edges[0]["volume_value"] == 80000
