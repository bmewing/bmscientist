from __future__ import annotations

from bmscientist.graph_enrichment import GraphEnrichmentStore
from bmscientist.graph_query import DuckDBGraphQueryEngine, GraphQueryAgent
from bmscientist.models import GraphQueryPlan


def test_duckdb_graph_query_engine_lists_tables_and_runs_select(tmp_path):
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)
    store._ensure_node("Product", "Tritan", "product_id", "product", aliases=["Eastman Tritan"])
    store._ensure_node("Application", "medical trays", "application_id", "application")

    engine = DuckDBGraphQueryEngine(graph_path)
    table_names = {table.table_name for table in engine.list_tables()}

    assert "Product" in table_names
    assert "Application" in table_names

    result = engine.query("SELECT name, product_id FROM Product ORDER BY name", limit=10)

    assert result.columns == ["name", "product_id"]
    assert result.rows[0]["name"] == "Tritan"
    assert result.rows[0]["product_id"] == "product:tritan"


def test_duckdb_graph_query_engine_rejects_non_select_sql(tmp_path):
    engine = DuckDBGraphQueryEngine(tmp_path / "graph")

    try:
        engine.query("DELETE FROM Product")
    except ValueError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Expected read-only SQL validation to reject DELETE")


def test_graph_query_agent_uses_llm_plan_then_executes(tmp_path):
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)
    store._ensure_node("Product", "ABS", "product_id", "product")

    class FakeLLM:
        def complete_json(self, response_model, system_prompt, user_prompt):
            assert "Available graph tables and columns" in user_prompt
            assert "Candidate entity matches from the graph" in user_prompt
            return GraphQueryPlan(
                sql="SELECT name FROM Product WHERE normalized_name = 'abs'",
                rationale="Return the canonical ABS product row.",
                assumptions=[],
            )

    engine = DuckDBGraphQueryEngine(graph_path)
    agent = GraphQueryAgent(FakeLLM(), engine)

    result = agent.run("Show me ABS rows", limit=10)

    assert result.rows == [{"name": "ABS"}]
    assert result.rationale == "Return the canonical ABS product row."
    assert result.assumptions == []
    assert any(match.node_label == "Product" and match.name == "ABS" for match in result.matched_entities)
    assert any(match.name == "ABS" for match in result.matched_entity_buckets.materials)


def test_duckdb_graph_query_engine_matches_entities_with_alias_and_fuzzy_overlap(tmp_path):
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)
    family_id = store.ensure_material_family(
        "ABS",
        canonical_name="ABS",
        family_type="polymer",
        aliases=["Acrylonitrile butadiene styrene"],
    )
    store.ensure_material_alias(
        "Acrylonitrile butadiene styrene",
        family_id,
        "MaterialFamily",
        alias_type="chemical_name",
        source_vendor="curated",
    )
    store._ensure_node("Market", "thermoformed sterile medical device trays", "market_id", None)

    engine = DuckDBGraphQueryEngine(graph_path)
    matches = engine.match_entities("Show me the volume of Acrylonitrile butadiene styrene being sold in thermoformed sterile medical device tray market")

    assert any(match.node_label == "MaterialFamily" and match.name == "ABS" for match in matches)
    assert any(
        match.node_label == "Market" and match.name == "thermoformed sterile medical device trays"
        for match in matches
    )


def test_format_graph_query_result_includes_match_breadcrumbs(tmp_path):
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)
    store._ensure_node("Product", "PVC", "product_id", "product")

    class FakeLLM:
        def complete_json(self, response_model, system_prompt, user_prompt):
            return GraphQueryPlan(
                sql="SELECT name FROM Product WHERE normalized_name = 'pvc'",
                rationale="Return the matched PVC product row.",
                assumptions=["Using Product because PVC exists there in this graph."],
            )

    engine = DuckDBGraphQueryEngine(graph_path)
    agent = GraphQueryAgent(FakeLLM(), engine)
    result = agent.run("Show me PVC", limit=10)

    assert result.matched_entities
    assert result.matched_entities[0].node_label == "Product"
    assert result.matched_entities[0].name == "PVC"
    assert result.matched_entity_buckets.materials
    assert result.matched_entity_buckets.materials[0].name == "PVC"


def test_graph_query_result_groups_market_matches_for_frontend(tmp_path):
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)
    store._ensure_node("Product", "PVC", "product_id", "product")
    store._ensure_node("Market", "thermoformed sterile medical device trays", "market_id", None)

    class FakeLLM:
        def complete_json(self, response_model, system_prompt, user_prompt):
            return GraphQueryPlan(
                sql="SELECT name FROM Product WHERE normalized_name = 'pvc'",
                rationale="Return PVC while preserving matched market breadcrumbs.",
                assumptions=[],
            )

    engine = DuckDBGraphQueryEngine(graph_path)
    agent = GraphQueryAgent(FakeLLM(), engine)
    result = agent.run("Show me PVC sold in thermoformed sterile medical device tray market", limit=10)

    assert any(match.name == "PVC" for match in result.matched_entity_buckets.materials)
    assert any(
        match.name == "thermoformed sterile medical device trays"
        for match in result.matched_entity_buckets.markets
    )
