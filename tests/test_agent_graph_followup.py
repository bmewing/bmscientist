from __future__ import annotations

from datetime import datetime, timezone

from bmscientist.agent import DiscoveryAgent
from bmscientist.graph_enrichment import GraphEnrichmentFollowUpQuestion
from bmscientist.models import ChunkRecord, GraphEnrichmentProposal, GraphEnrichmentValidationOutput


def make_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        run_id="run-follow-up-1",
        original_query="CTQ requirements for extruded light diffusers",
        search_query="light diffuser ctq",
        source_title="Primary source",
        source_url="https://example.com/primary",
        source_domain="example.com",
        retrieved_at=datetime.now(timezone.utc),
        chunk_index=0,
        chunk_text=text,
        application="extruded light diffusers",
        incumbent_material=None,
        candidate_materials=["Exolon DX"],
        evidence_type="application requirements",
        application_requirements=["light diffusion"],
        substitution_drivers=[],
        relevance_score=0.8,
        confidence_score=0.7,
    )


def test_discovery_agent_expansion_can_trigger_external_follow_up_search():
    agent = DiscoveryAgent.__new__(DiscoveryAgent)

    class FakeExpander:
        def expand(self, original_query, proposals, validations, records, max_questions_per_chunk=3):
            return (
                [
                    GraphEnrichmentFollowUpQuestion(
                        source_chunk_id=records[0].id,
                        question="Who makes Exolon DX polycarbonate sheet?",
                        rationale="Need supplier edge.",
                        target_edge_types=["Company_PRODUCES_Product"],
                    )
                ],
                [],
            )

    class FakeProposer:
        def propose(self, original_query, records, limit=24):
            assert records[0].source_title == "Follow-up source"
            return [
                GraphEnrichmentProposal.model_validate(
                    {
                        "proposal_id": "claim-follow-up-1",
                        "edge_type": "Company_PRODUCES_Product",
                        "product_name": "Exolon DX",
                        "company_name": "Covestro",
                        "source_chunk_id": records[0].id,
                        "source_url": records[0].source_url,
                        "source_title": records[0].source_title,
                        "supporting_quote": "Exolon DX is offered by Covestro.",
                        "confidence_score": 0.78,
                    }
                )
            ]

    class FakeValidator:
        def validate(self, proposals, records):
            return GraphEnrichmentValidationOutput.model_validate(
                {
                    "validations": [
                        {
                            "proposal_id": proposal.proposal_id,
                            "accepted": True,
                            "confidence_score": 0.81,
                            "rationale": "Direct support.",
                        }
                        for proposal in proposals
                    ]
                }
            ).validations

    follow_up_chunk = make_chunk("chunk-follow-up-2", "Exolon DX polycarbonate sheet is offered by Covestro.")
    follow_up_chunk = follow_up_chunk.model_copy(
        update={
            "search_query": "Who makes Exolon DX polycarbonate sheet?",
            "source_title": "Follow-up source",
            "source_url": "https://example.com/follow-up",
        }
    )

    agent._graph_enrichment_expander = FakeExpander()
    agent._graph_enrichment_proposer = FakeProposer()
    agent._graph_enrichment_validator = FakeValidator()
    agent._run_graph_follow_up_search = lambda state, questions: [follow_up_chunk]

    initial_chunk = make_chunk("chunk-initial-1", "Exolon DX polycarbonate sheet is used in extruded light diffusers.")
    initial_proposals = [
        GraphEnrichmentProposal.model_validate(
            {
                "proposal_id": "claim-1",
                "edge_type": "Product_USED_IN_Application",
                "product_name": "Exolon DX",
                "application_name": "extruded light diffusers",
                "source_chunk_id": initial_chunk.id,
                "confidence_score": 0.8,
            }
        )
    ]
    initial_validations = GraphEnrichmentValidationOutput.model_validate(
        {
            "validations": [
                {
                    "proposal_id": "claim-1",
                    "accepted": True,
                    "confidence_score": 0.82,
                    "rationale": "Direct support.",
                }
            ]
        }
    ).validations

    state = {
        "run_id": "run-follow-up-1",
        "original_query": "CTQ requirements for extruded light diffusers",
        "chunk_records": [initial_chunk],
        "graph_enrichment_proposals": initial_proposals,
        "graph_enrichment_validations": initial_validations,
        "skipped_pages": [],
        "errors": [],
    }

    result = agent.expand_graph_enrichments(state)

    assert result["graph_enrichment_external_search_queries"] == ["Who makes Exolon DX polycarbonate sheet?"]
    assert len(result["graph_enrichment_expansion_proposals"]) == 1
    assert result["graph_enrichment_expansion_proposals"][0].edge_type == "Company_PRODUCES_Product"
    assert result["graph_enrichment_expansion_proposals"][0].company_name == "Covestro"
    assert len(result["chunk_records"]) == 2
