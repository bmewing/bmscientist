from datetime import datetime, timezone

from app_discovery_agent.models import ChunkRecord, EvidenceClassification


def test_evidence_classification_parses_expected_shape():
    classification = EvidenceClassification.model_validate(
        {
            "relevant": True,
            "relevance_score": 0.78,
            "confidence_score": 0.61,
            "application": "blister packaging",
            "incumbent_material": "PVC",
            "candidate_materials": ["PETG", "Eastman Tritan"],
            "evidence_type": "application requirements",
            "application_requirements": ["clarity", "impact resistance"],
            "substitution_drivers": ["recyclability"],
            "rationale": "The page describes clear rigid packaging needs.",
            "supporting_quotes": ["clear rigid film used in packaging"],
            "metadata": {"source_type": "product page"},
        }
    )

    assert classification.relevant is True
    assert classification.candidate_materials == ["PETG", "Eastman Tritan"]


def test_chunk_record_parses_nested_metadata():
    record = ChunkRecord.model_validate(
        {
            "id": "chunk-1",
            "run_id": "run-1",
            "original_query": "PVC alternatives",
            "search_query": "PVC PETG application",
            "source_title": "Example",
            "source_url": "https://example.com/page",
            "source_domain": "example.com",
            "retrieved_at": datetime.now(timezone.utc),
            "chunk_index": 0,
            "chunk_text": "PVC is used for a clear rigid application.",
            "vector": [0.1, 0.2, 0.3],
            "application": "medical packaging",
            "incumbent_material": "PVC",
            "candidate_materials": ["PET"],
            "evidence_type": "application currently uses PVC",
            "application_requirements": ["clarity"],
            "substitution_drivers": ["regulatory pressure"],
            "relevance_score": 0.84,
            "confidence_score": 0.59,
            "metadata": {"quotes": ["PVC tray"]},
        }
    )

    assert record.metadata["quotes"] == ["PVC tray"]

