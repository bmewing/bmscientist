from __future__ import annotations

import json
from datetime import datetime, timezone

import pyarrow.parquet as pq

from app_discovery_agent.graph_enrichment import GraphEnrichmentProposer, GraphEnrichmentStore, GraphEnrichmentValidator
from app_discovery_agent.graph_market import GraphMarketEvidence
from app_discovery_agent.models import (
    ChunkRecord,
    GraphEnrichmentValidationOutput,
)
from tests.test_coscientist import make_document, make_hypothesis


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
