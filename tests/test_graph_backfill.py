from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from app_discovery_agent.graph_backfill import chunk_record_from_lancedb_row, existing_claimed_chunk_ids


def test_chunk_record_from_lancedb_row_reconstructs_model():
    record = chunk_record_from_lancedb_row(
        {
            "id": "chunk-1",
            "run_id": "run-1",
            "original_query": "PVC applications",
            "search_query": "PVC blister packaging",
            "source_title": "Example",
            "source_url": "https://example.com",
            "source_domain": "example.com",
            "retrieved_at": "2026-06-14T12:00:00+00:00",
            "chunk_index": 2,
            "chunk_text": "PVC is used in blister packaging.",
            "vector": [0.1, 0.2],
            "application": "blister packaging",
            "incumbent_material": "PVC",
            "candidate_materials": ["PETG"],
            "evidence_type": "application currently uses PVC",
            "application_requirements": ["clarity"],
            "substitution_drivers": ["PVC reduction"],
            "relevance_score": 0.8,
            "confidence_score": 0.7,
            "metadata": {"ok": True},
        }
    )

    assert record is not None
    assert record.id == "chunk-1"
    assert record.application == "blister packaging"
    assert record.candidate_materials == ["PETG"]


def test_existing_claimed_chunk_ids_reads_claim_ledger(tmp_path):
    path = tmp_path / "GraphEnrichmentClaim.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"claim_id": "claim-1", "source_chunk_id": "chunk-1"}],
            schema=pa.schema([("claim_id", pa.string()), ("source_chunk_id", pa.string())]),
        ),
        path,
    )

    assert existing_claimed_chunk_ids(path) == {"chunk-1"}
