from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from bmscientist.graph_enrichment import GraphEnrichmentProposer, GraphEnrichmentStore, GraphEnrichmentValidator
from bmscientist.models import ChunkRecord
from bmscientist.store import LanceEvidenceStore


LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path("data")
DEFAULT_CLAIM_LEDGER_PATH = DEFAULT_DATA_DIR / "graph" / "enrichment" / "GraphEnrichmentClaim.parquet"


@dataclass(frozen=True)
class GraphBackfillResult:
    scanned_chunks: int
    eligible_chunks: int
    proposed_claims: int
    accepted_claims: int
    batches: int
    output_path: Path


class LanceGraphBackfiller:
    def __init__(
        self,
        store: LanceEvidenceStore,
        proposer: GraphEnrichmentProposer,
        validator: GraphEnrichmentValidator,
        graph_store: GraphEnrichmentStore,
        data_dir: Path | None = None,
    ):
        self._store = store
        self._proposer = proposer
        self._validator = validator
        self._graph_store = graph_store
        self._data_dir = data_dir if data_dir is not None else DEFAULT_DATA_DIR

    def run(
        self,
        query: str = "backfill existing LanceDB evidence into graph enrichment proposals",
        batch_size: int = 12,
        limit: int | None = None,
        skip_claimed: bool = True,
        records: list[ChunkRecord] | None = None,
    ) -> GraphBackfillResult:
        if records is None:
            rows = self._store.all_rows()
            records = [record for row in rows if (record := chunk_record_from_lancedb_row(row)) is not None]
            scanned = len(records)
        else:
            scanned = len(records)

        if skip_claimed:
            claim_ledger = self._data_dir / "graph" / "enrichment" / "GraphEnrichmentClaim.parquet"
            claimed_chunk_ids = existing_claimed_chunk_ids(claim_ledger)
            records = [record for record in records if record.id not in claimed_chunk_ids]
        if limit is not None:
            records = records[:limit]

        run_id = f"graph-backfill-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        total_proposed = 0
        total_accepted = 0
        batches = 0
        details: list[dict[str, Any]] = []

        for batch in batched(records, batch_size):
            batches += 1
            proposals = self._proposer.propose(query, batch, limit=batch_size)
            validations = self._validator.validate(proposals, batch)
            accepted = self._graph_store.write(proposals, validations, run_id, query)
            total_proposed += len(proposals)
            total_accepted += accepted
            details.append(
                {
                    "batch": batches,
                    "chunk_ids": [record.id for record in batch],
                    "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                    "validations": [validation.model_dump(mode="json") for validation in validations],
                    "accepted": accepted,
                }
            )
            LOGGER.info(
                "Graph backfill batch %s produced %s proposals and %s accepted claims",
                batches,
                len(proposals),
                accepted,
            )

        output_path = self._data_dir / "raw" / f"{run_id}_graph_backfill.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "query": query,
                    "scanned_chunks": scanned,
                    "eligible_chunks": len(records),
                    "proposed_claims": total_proposed,
                    "accepted_claims": total_accepted,
                    "batches": batches,
                    "skip_claimed": skip_claimed,
                    "details": details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return GraphBackfillResult(
            scanned_chunks=scanned,
            eligible_chunks=len(records),
            proposed_claims=total_proposed,
            accepted_claims=total_accepted,
            batches=batches,
            output_path=output_path.resolve(),
        )


def chunk_record_from_lancedb_row(row: dict[str, Any]) -> ChunkRecord | None:
    chunk_text = str(row.get("chunk_text") or "").strip()
    chunk_id = str(row.get("id") or "").strip()
    if not chunk_id or not chunk_text:
        return None
    retrieved_at = parse_datetime(row.get("retrieved_at"))
    return ChunkRecord(
        id=chunk_id,
        run_id=str(row.get("run_id") or "unknown"),
        original_query=str(row.get("original_query") or ""),
        search_query=str(row.get("search_query") or ""),
        source_title=str(row.get("source_title") or ""),
        source_url=str(row.get("source_url") or ""),
        source_domain=str(row.get("source_domain") or ""),
        retrieved_at=retrieved_at,
        chunk_index=int(row.get("chunk_index") or 0),
        chunk_text=chunk_text,
        vector=[float(value) for value in row.get("vector") or []],
        application=none_if_blank(row.get("application")),
        incumbent_material=none_if_blank(row.get("incumbent_material")),
        candidate_materials=[str(item) for item in row.get("candidate_materials") or [] if item],
        evidence_type=str(row.get("evidence_type") or "market or customer need"),
        application_requirements=[str(item) for item in row.get("application_requirements") or [] if item],
        substitution_drivers=[str(item) for item in row.get("substitution_drivers") or [] if item],
        relevance_score=float(row.get("relevance_score") or 0.0),
        confidence_score=float(row.get("confidence_score") or 0.0),
        metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
    )


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def none_if_blank(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def existing_claimed_chunk_ids(path: Path = DEFAULT_CLAIM_LEDGER_PATH) -> set[str]:
    if not path.exists():
        return set()
    try:
        return {
            str(row.get("source_chunk_id"))
            for row in pq.read_table(path, columns=["source_chunk_id"]).to_pylist()
            if row.get("source_chunk_id")
        }
    except Exception:
        LOGGER.exception("Unable to read graph enrichment claim ledger at %s", path)
        return set()


def batched(records: list[ChunkRecord], batch_size: int) -> list[list[ChunkRecord]]:
    size = max(1, batch_size)
    return [records[index : index + size] for index in range(0, len(records), size)]
