from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from app_discovery_agent.models import ChunkRecord


TABLE_NAME = "evidence_chunks"


class LanceEvidenceStore:
    def __init__(self, db_path: Path):
        self._db = lancedb.connect(str(db_path))
        self._table = None
        self._vector_dim: int | None = None

    def _table_names(self) -> set[str]:
        if hasattr(self._db, "list_tables"):
            listing = self._db.list_tables()
        else:
            listing = self._db.table_names()
        names: set[str] = set()
        listing_items = [listing] if isinstance(listing, dict) else list(listing)
        for item in listing_items:
            for normalized in self._normalize_table_names(item):
                names.add(normalized)
        return names

    @staticmethod
    def _normalize_table_name(item: Any) -> str | None:
        names = LanceEvidenceStore._normalize_table_names(item)
        return names[0] if names else None

    @staticmethod
    def _normalize_table_names(item: Any) -> list[str]:
        if isinstance(item, str):
            return [item]
        if isinstance(item, dict):
            table_names = item.get("tables")
            if isinstance(table_names, list):
                return [str(name) for name in table_names if name]
            name = item.get("name")
            return [str(name)] if name else []
        if isinstance(item, (list, tuple)) and item:
            if len(item) >= 2 and item[0] == "tables" and isinstance(item[1], list):
                return [str(name) for name in item[1] if name]
            if len(item) >= 2 and item[0] == "page_token":
                return []
            head = item[0]
            return [str(head)] if isinstance(head, str) else []
        return []

    def _build_schema(self, vector_dim: int) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("run_id", pa.string()),
                pa.field("original_query", pa.string()),
                pa.field("search_query", pa.string()),
                pa.field("source_title", pa.string()),
                pa.field("source_url", pa.string()),
                pa.field("source_domain", pa.string()),
                pa.field("retrieved_at", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("chunk_text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("application", pa.string()),
                pa.field("incumbent_material", pa.string()),
                pa.field("candidate_materials", pa.list_(pa.string())),
                pa.field("evidence_type", pa.string()),
                pa.field("application_requirements", pa.list_(pa.string())),
                pa.field("substitution_drivers", pa.list_(pa.string())),
                pa.field("relevance_score", pa.float32()),
                pa.field("confidence_score", pa.float32()),
                pa.field("metadata", pa.string()),
            ]
        )

    def _ensure_table(self, vector_dim: int):
        if self._table is not None:
            return self._table
        self._vector_dim = vector_dim
        if TABLE_NAME in self._table_names():
            self._table = self._db.open_table(TABLE_NAME)
            return self._table
        schema = self._build_schema(vector_dim)
        try:
            self._table = self._db.create_table(TABLE_NAME, schema=schema)
            return self._table
        except Exception as exc:
            # LanceDB can report "already exists" even when the preceding table listing
            # did not surface the table in a stable shape. In that case, open and append.
            if "already exists" in str(exc).lower():
                self._table = self._db.open_table(TABLE_NAME)
                return self._table
            raise

    def _serialize_record(self, record: ChunkRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "run_id": record.run_id,
            "original_query": record.original_query,
            "search_query": record.search_query,
            "source_title": record.source_title,
            "source_url": str(record.source_url),
            "source_domain": record.source_domain,
            "retrieved_at": record.retrieved_at.isoformat(),
            "chunk_index": record.chunk_index,
            "chunk_text": record.chunk_text,
            "vector": [float(value) for value in record.vector],
            "application": record.application,
            "incumbent_material": record.incumbent_material,
            "candidate_materials": list(record.candidate_materials),
            "evidence_type": record.evidence_type,
            "application_requirements": list(record.application_requirements),
            "substitution_drivers": list(record.substitution_drivers),
            "relevance_score": float(record.relevance_score),
            "confidence_score": float(record.confidence_score),
            "metadata": json.dumps(record.metadata, sort_keys=True),
        }

    @staticmethod
    def _deserialize_row(row: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(row)
        metadata = parsed.get("metadata")
        if isinstance(metadata, str) and metadata:
            parsed["metadata"] = json.loads(metadata)
        return parsed

    def add_chunks(self, records: list[ChunkRecord]) -> int:
        if not records:
            return 0
        table = self._ensure_table(len(records[0].vector))
        payload = [self._serialize_record(record) for record in records]
        table.add(payload)
        return len(payload)

    def search_by_vector(self, vector: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        table = self._ensure_table(len(vector))
        rows = table.search(vector).limit(top_k).to_list()
        return [self._deserialize_row(row) for row in rows]

    def all_rows(self) -> list[dict[str, Any]]:
        table = self._table
        if table is None:
            if TABLE_NAME not in self._table_names():
                return []
            table = self._db.open_table(TABLE_NAME)
        if hasattr(table, "to_arrow"):
            return [self._deserialize_row(row) for row in table.to_arrow().to_pylist()]
        return [self._deserialize_row(row) for row in table.search([0.0] * (self._vector_dim or 1)).limit(1000).to_list()]
