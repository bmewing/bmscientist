from __future__ import annotations

import json
from datetime import datetime, timezone

import pyarrow.parquet as pq

from bmscientist.graph_enrichment import GraphEnrichmentProposer, GraphEnrichmentStore, GraphEnrichmentValidator
from bmscientist.graph_market import GraphMarketEvidence
from bmscientist.coscientist_models import MarketVolumeEstimateOutput, ReflectionAssessment
from bmscientist.models import (
    ChunkRecord,
    GraphEnrichmentValidationOutput,
)
from tests.test_coscientist import make_document, make_generic_document, make_hypothesis


def make_chunk() -> ChunkRecord:
    return ChunkRecord(
        id="chunk-graph-1",
        run_id="run-graph-1",
        original_query="PETG medical tray applications",
        search_query="PETG medical trays clarity thermoforming",
        source_title="Medical tray material guide",
        source_url="https://example.com/trays",
        source_domain="example.com",
        retrieved_at=datetime.now(timezone.utc),
        chunk_index=0,
        chunk_text="PETG sheet is used in medical trays where clarity and thermoformability are critical requirements.",
        vector=[0.1, 0.2, 0.3],
        application="medical trays",
        incumbent_material="PVC",
        candidate_materials=["PETG"],
        evidence_type="application requirements",
        application_requirements=["clarity", "thermoformability"],
        substitution_drivers=["PVC reduction"],
        relevance_score=0.8,
        confidence_score=0.7,
    )


class ProposalLLM:
    def complete_json(self, response_model, system_prompt, user_prompt):
        assert "Product_USED_IN_Application" in user_prompt
        return response_model.model_validate(
            {
                "proposals": [
                    {
                        "edge_type": "Product_USED_IN_Application",
                        "product_name": "PETG sheet",
                        "application_name": "medical trays",
                        "market_name": "medical packaging",
                        "relationship_role": "candidate material",
                        "critical_to_quality": ["clarity", "thermoformability"],
                        "metrics": [
                            {"name": "volume", "value": 1200, "unit": "tonnes", "year": 2025, "basis": "segment use"}
                        ],
                        "source_chunk_id": "chunk-graph-1",
                        "source_url": "https://example.com/trays",
                        "source_title": "Medical tray material guide",
                        "supporting_quote": "PETG sheet is used in medical trays",
                        "rationale": "Direct product/application relationship.",
                        "confidence_score": 0.74,
                    }
                ]
            }
        )


class ValidationLLM:
    def complete_json(self, response_model, system_prompt, user_prompt):
        assert "PETG sheet is used in medical trays" in user_prompt
        import json

        proposal_rows = json.loads(user_prompt.split("Source evidence chunks:")[0].split("Candidate graph enrichment proposals:")[1])
        return response_model.model_validate(
            {
                "validations": [
                    {
                        "proposal_id": proposal_rows[0]["proposal_id"],
                        "accepted": True,
                        "confidence_score": 0.82,
                        "rationale": "The cited quote directly supports the relationship.",
                        "corrected_relationship_role": "candidate material",
                        "corrected_critical_to_quality": ["clarity", "thermoformability"],
                    }
                ]
            }
        )


def test_graph_enrichment_proposes_validates_writes_and_retrieves_product_application(tmp_path):
    record = make_chunk()
    proposals = GraphEnrichmentProposer(ProposalLLM()).propose(record.original_query, [record])
    validations = GraphEnrichmentValidator(ValidationLLM()).validate(proposals, [record])

    graph_path = tmp_path / "graph"
    accepted = GraphEnrichmentStore(graph_path).write(proposals, validations, record.run_id, record.original_query)

    assert accepted == 1
    claim_rows = pq.read_table(graph_path / "enrichment" / "GraphEnrichmentClaim.parquet").to_pylist()
    assert claim_rows[0]["validation_status"] == "accepted"

    edge_rows = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
    assert edge_rows[0]["product_id"] == "product:petg-sheet"
    assert edge_rows[0]["application_id"] == "application:medical-trays"
    assert edge_rows[0]["volume_value"] == 1200
    assert edge_rows[0]["evidence_hash"]

    evidence_rows = GraphMarketEvidence(graph_path).build_evidence_rows(make_document(), make_hypothesis())
    assert any(row["metadata"]["edge_type"] == "Product_USED_IN_Application" for row in evidence_rows)


def test_graph_enrichment_store_keeps_rejected_claim_out_of_edges(tmp_path):
    record = make_chunk()
    proposals = GraphEnrichmentProposer(ProposalLLM()).propose(record.original_query, [record])
    validations = GraphEnrichmentValidationOutput.model_validate(
        {
            "validations": [
                {
                    "proposal_id": proposals[0].proposal_id,
                    "accepted": False,
                    "confidence_score": 0.2,
                    "rationale": "Relationship not direct enough.",
                }
            ]
        }
    ).validations

    graph_path = tmp_path / "graph"
    accepted = GraphEnrichmentStore(graph_path).write(proposals, validations, record.run_id, record.original_query)

    assert accepted == 0
    assert (graph_path / "enrichment" / "GraphEnrichmentClaim.parquet").exists()
    assert not (graph_path / "edges" / "Product_USED_IN_Application.parquet").exists()


def test_graph_enrichment_store_keeps_low_confidence_acceptance_out_of_edges(tmp_path):
    record = make_chunk()
    proposals = GraphEnrichmentProposer(ProposalLLM()).propose(record.original_query, [record])
    validations = GraphEnrichmentValidationOutput.model_validate(
        {
            "validations": [
                {
                    "proposal_id": proposals[0].proposal_id,
                    "accepted": True,
                    "confidence_score": 0.4,
                    "rationale": "Technically accepted, but too weak to promote.",
                }
            ]
        }
    ).validations

    graph_path = tmp_path / "graph"
    accepted = GraphEnrichmentStore(graph_path).write(proposals, validations, record.run_id, record.original_query)

    assert accepted == 0
    claim_rows = pq.read_table(graph_path / "enrichment" / "GraphEnrichmentClaim.parquet").to_pylist()
    assert claim_rows[0]["validation_status"] == "accepted"
    assert not (graph_path / "edges" / "Product_USED_IN_Application.parquet").exists()


def test_graph_enrichment_merges_product_aliases_without_collapsing_master_data(tmp_path):
    record = make_chunk().model_copy(
        update={
            "id": "chunk-ps-1",
            "chunk_text": "PS (polystyrene) is used in rigid display trays where stiffness matters.",
            "candidate_materials": ["PS"],
        }
    )
    proposal = {
        "edge_type": "Product_USED_IN_Application",
        "product_name": "PS",
        "product_aliases": ["polystyrene"],
        "application_name": "rigid display trays",
        "relationship_role": "material",
        "source_chunk_id": "chunk-ps-1",
        "source_url": "https://example.com/ps",
        "supporting_quote": "PS (polystyrene) is used in rigid display trays",
        "confidence_score": 0.8,
    }

    class AliasProposalLLM:
        def complete_json(self, response_model, system_prompt, user_prompt):
            return response_model.model_validate({"proposals": [proposal]})

    proposals = GraphEnrichmentProposer(AliasProposalLLM()).propose(record.original_query, [record])
    validations = GraphEnrichmentValidationOutput.model_validate(
        {
            "validations": [
                {
                    "proposal_id": proposals[0].proposal_id,
                    "accepted": True,
                    "confidence_score": 0.8,
                    "rationale": "The parenthetical expansion supports the alias.",
                    "corrected_product_aliases": ["polystyrene"],
                }
            ]
        }
    ).validations

    graph_path = tmp_path / "graph"
    accepted = GraphEnrichmentStore(graph_path).write(proposals, validations, record.run_id, record.original_query)

    assert accepted == 1
    product_rows = pq.read_table(graph_path / "nodes" / "Product.parquet").to_pylist()
    assert product_rows[0]["product_id"] == "product:polystyrene"
    assert product_rows[0]["name"] == "polystyrene"
    assert json.loads(product_rows[0]["aliases_json"]) == ["PS"]

    claim_rows = pq.read_table(graph_path / "enrichment" / "GraphEnrichmentClaim.parquet").to_pylist()
    assert json.loads(claim_rows[0]["product_aliases_json"]) == ["polystyrene"]


def test_promote_hypothesis_writes_to_graph(tmp_path):
    from tests.test_coscientist import make_reflected_hypothesis

    hypothesis = make_reflected_hypothesis()
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)

    store.promote_hypothesis(hypothesis)

    # Verify nodes
    products = pq.read_table(graph_path / "nodes" / "Product.parquet").to_pylist()
    # Should have candidate (PETG) and incumbent (PVC)
    product_names = {p["name"] for p in products}
    assert "PETG" in product_names
    assert "PVC" in product_names

    applications = pq.read_table(graph_path / "nodes" / "Application.parquet").to_pylist()
    assert applications[0]["name"] == "medical trays"

    markets = pq.read_table(graph_path / "nodes" / "Market.parquet").to_pylist()
    assert markets[0]["name"] == "medical packaging"

    # Verify edges
    edges = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
    # There should be 2 edges: PETG -> medical trays and PVC -> medical trays
    assert len(edges) == 2
    roles = {e["relationship_role"] for e in edges}
    assert "candidate_replacement" in roles
    assert "incumbent" in roles

    mkt_app_edges = pq.read_table(graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet").to_pylist()
    assert len(mkt_app_edges) == 1
    assert mkt_app_edges[0]["market_id"] == "market:medical-packaging"
    assert mkt_app_edges[0]["application_id"] == "application:medical-trays"

    mkt_prod_edges = pq.read_table(graph_path / "edges" / "Market_USES_Product.parquet").to_pylist()
    assert len(mkt_prod_edges) == 1
    assert mkt_prod_edges[0]["market_id"] == "market:medical-packaging"
    assert mkt_prod_edges[0]["product_id"] == "product:petg"


def test_write_ai_market_volume_estimate_updates_market_and_material_edges(tmp_path):
    hypothesis = make_hypothesis().model_copy(
        update={
            "application": "thermoformed medical trays",
            "market_segment": "medical packaging",
            "incumbent_material": "PVC",
        }
    )
    estimate = MarketVolumeEstimateOutput.model_validate(
        {
            "market_name": "medical packaging",
            "application_name": "thermoformed medical trays",
            "total_substrate_volume_value": 80000,
            "total_substrate_volume_unit": "metric_tons_per_year",
            "volume_year": 2026,
            "revenue_value": 1200,
            "revenue_unit": "USD million",
            "revenue_year": 2026,
            "confidence": 0.52,
            "rationale": "Estimated from reported tray market revenue and average substrate price.",
            "material_volumes": [
                {
                    "material_name": "PETG",
                    "volume_value": 44000,
                    "volume_unit": "metric_tons_per_year",
                    "share_of_total": 0.55,
                    "confidence": 0.5,
                    "rationale": "PETG is estimated as the largest current substrate share.",
                },
                {
                    "material_name": "PVC",
                    "volume_value": 4000,
                    "volume_unit": "metric_tons_per_year",
                    "share_of_total": 0.05,
                    "confidence": 0.45,
                    "rationale": "PVC is estimated as a residual legacy share.",
                },
            ],
            "source_citations": [
                {
                    "chunk_id": "graph:market",
                    "source_url": "https://example.com/tray-market",
                    "source_title": "Tray market estimate",
                }
            ],
        }
    )

    graph_path = tmp_path / "graph"
    rows = GraphEnrichmentStore(graph_path).write_ai_market_volume_estimate(hypothesis, estimate)

    assert len(rows) == 3
    mkt_app_edges = pq.read_table(graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet").to_pylist()
    assert mkt_app_edges[0]["volume_value"] == 80000
    assert mkt_app_edges[0]["source_node_type"] == "ai_volume_estimate"
    assert mkt_app_edges[0]["confidence"] == 0.52
    assert "AI generated estimate" in mkt_app_edges[0]["highlights_json"]

    product_edges = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
    volumes_by_product = {edge["product_id"]: edge for edge in product_edges}
    assert volumes_by_product["product:pvc"]["volume_value"] == 4000
    assert volumes_by_product["product:pvc"]["relationship_role"] == "incumbent_ai_estimated_share"
    assert volumes_by_product["product:petg"]["volume_value"] == 44000


def test_apply_edge_feedback_updates_graph(tmp_path):
    from tests.test_coscientist import make_reflected_hypothesis

    hypothesis = make_reflected_hypothesis()
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)

    store.promote_hypothesis(hypothesis)

    updated_count = store.apply_edge_feedback(
        candidate_material="PETG",
        incumbent_material="PVC",
        application="medical trays",
        volume=0.0,
        status="rejected",
        comment="absurdly low volume",
    )

    assert updated_count == 2

    edges = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
    for edge in edges:
        assert edge["volume_value"] == 0.0
        assert edge["validation_status"] == "rejected"
        assert edge["supporting_quote"] == "absurdly low volume"


def test_apply_hypothesis_feedback_updates_hypothesis_and_graph(tmp_path):
    from tests.test_coscientist import make_reflected_hypothesis
    from bmscientist.coscientist_store import CoScientistStore

    coscientist_path = tmp_path / "coscientist"
    graph_path = tmp_path / "graph"

    import bmscientist.graph_enrichment as ge
    original_graph_path = ge.GRAPH_PATH
    ge.GRAPH_PATH = graph_path

    try:
        cosc_store = CoScientistStore(coscientist_path)
        hypothesis = make_reflected_hypothesis()

        cosc_store.save_hypothesis(hypothesis)

        updated = cosc_store.apply_hypothesis_feedback(
            research_id=hypothesis.research_id,
            hypothesis_id=hypothesis.hypothesis_id,
            volume=0.01,
            status="rejected",
            comment="obsolete application",
        )

        assert updated is not None
        assert updated.is_active is False
        assert updated.status == "retired"
        assert updated.retired_reason == "obsolete application"

        saved = cosc_store.load_hypotheses(hypothesis.research_id, stages={"retired"})
        assert len(saved) == 1
        assert saved[0].is_active is False

        edges = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
        assert len(edges) == 2
        for edge in edges:
            assert edge["volume_value"] == 0.01
            assert edge["validation_status"] == "rejected"
    finally:
        ge.GRAPH_PATH = original_graph_path


def test_promote_generic_candidate_design_hypothesis_writes_flexible_graph(tmp_path):
    hypothesis = make_hypothesis().model_copy(
        update={
            "status": "reflected",
            "candidate_material": None,
            "title": "Candidate coalescent A for acrylic latex",
            "summary": "Candidate coalescing aid for acrylic latex with lower aquatic toxicity risk.",
            "application": "waterborne coatings",
            "market_segment": "architectural coatings",
            "candidate_artifact": {
                "name_or_label": "Candidate coalescent A",
                "smiles": "CCOC(=O)OCC",
                "intended_binder_system": "acrylic latex",
                "chemistry_class": "ester alcohol",
                "functional_role": "coalescing aid",
                "manufacturer": "Eastman Chemical Company",
            },
            "reflection_assessment": ReflectionAssessment.model_validate(
                {
                    "criterion_results": [
                        {
                            "criterion_name": "aquatic_toxicity_risk",
                            "value": "lower concern",
                            "normalized_score": 0.76,
                            "confidence": 0.62,
                            "evidence_mode": "external_tool",
                            "tool_id": "opera_qsar",
                            "is_inferred": True,
                        },
                        {
                            "criterion_name": "water_compatibility",
                            "value": "moderate",
                            "normalized_score": 0.68,
                            "confidence": 0.55,
                            "evidence_mode": "literature",
                            "is_inferred": True,
                        },
                    ]
                }
            ),
        }
    )
    graph_path = tmp_path / "graph"
    store = GraphEnrichmentStore(graph_path)

    store.promote_hypothesis(hypothesis)

    products = pq.read_table(graph_path / "nodes" / "Product.parquet").to_pylist()
    assert products[0]["name"] == "Candidate coalescent A"
    assert products[0]["canonical_smiles"] == "CCOC(=O)OCC"

    chemistry_classes = pq.read_table(graph_path / "nodes" / "ChemistryClass.parquet").to_pylist()
    assert chemistry_classes[0]["name"] == "ester alcohol"

    functions = pq.read_table(graph_path / "nodes" / "Function.parquet").to_pylist()
    assert functions[0]["name"] == "coalescing aid"

    binders = pq.read_table(graph_path / "nodes" / "BinderSystem.parquet").to_pylist()
    assert binders[0]["name"] == "acrylic latex"

    endpoints = pq.read_table(graph_path / "nodes" / "Endpoint.parquet").to_pylist()
    endpoint_names = {row["name"] for row in endpoints}
    assert "aquatic_toxicity_risk" in endpoint_names
    assert "water_compatibility" in endpoint_names

    companies = pq.read_table(graph_path / "nodes" / "Company.parquet").to_pylist()
    assert companies[0]["name"] == "Eastman Chemical Company"

    endpoint_edges = pq.read_table(graph_path / "edges" / "Product_HAS_Endpoint.parquet").to_pylist()
    assert len(endpoint_edges) == 2
    assert {edge["tool_id"] for edge in endpoint_edges} == {"opera_qsar", None}

    evidence_rows = GraphMarketEvidence(graph_path).build_evidence_rows(make_generic_document(), hypothesis)
    edge_types = {row["metadata"]["edge_type"] for row in evidence_rows}
    assert "Product_HAS_Endpoint" in edge_types
    assert "Product_TARGETS_BinderSystem" in edge_types
