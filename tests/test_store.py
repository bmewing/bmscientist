from datetime import datetime, timezone

from app_discovery_agent.models import ChunkRecord
from app_discovery_agent.store import LanceEvidenceStore


def test_lancedb_insert_and_vector_retrieval(tmp_path):
    store = LanceEvidenceStore(tmp_path / "lancedb")
    now = datetime.now(timezone.utc)
    records = [
        ChunkRecord(
            id="1",
            run_id="run-1",
            original_query="PVC alternatives",
            search_query="PVC PETG application",
            source_title="A",
            source_url="https://example.com/a",
            source_domain="example.com",
            retrieved_at=now,
            chunk_index=0,
            chunk_text="PVC is used in a clear rigid tray.",
            vector=[1.0, 0.0, 0.0],
            application="tray",
            incumbent_material="PVC",
            candidate_materials=["PETG"],
            evidence_type="application currently uses PVC",
            application_requirements=["clarity"],
            substitution_drivers=["recyclability"],
            relevance_score=0.8,
            confidence_score=0.6,
            metadata={"quotes": ["clear rigid tray"]},
        ),
        ChunkRecord(
            id="2",
            run_id="run-1",
            original_query="PVC alternatives",
            search_query="PVC PETG application",
            source_title="B",
            source_url="https://example.com/b",
            source_domain="example.com",
            retrieved_at=now,
            chunk_index=1,
            chunk_text="PETG is positioned for transparent packaging.",
            vector=[0.0, 1.0, 0.0],
            application="packaging",
            incumbent_material="PVC",
            candidate_materials=["PETG"],
            evidence_type="PET/PETG/Tritan capability evidence",
            application_requirements=["transparency"],
            substitution_drivers=["performance"],
            relevance_score=0.75,
            confidence_score=0.58,
            metadata={"quotes": ["transparent packaging"]},
        ),
    ]

    inserted = store.add_chunks(records)
    results = store.search_by_vector([1.0, 0.0, 0.0], top_k=1)

    assert inserted == 2
    assert len(results) == 1
    assert results[0]["id"] == "1"
