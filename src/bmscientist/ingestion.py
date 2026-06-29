from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, Field

from bmscientist.chunking import TextChunker
from bmscientist.extract import extract_pdf_text, extract_readable_text
from bmscientist.models import ChunkRecord, EvidenceClassification, PageContent


PRIVATE_QUERY = "private uploaded project document evidence"
ENCRYPTED_PARQUET_SUFFIX = ".parquet.aesgcm"
ENCRYPTED_PAYLOAD_MAGIC = b"BMSA1"
NONCE_BYTES = 12


class PrivateEmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class PrivateClassifier(Protocol):
    def classify(self, query: str, page: PageContent) -> EvidenceClassification: ...


class VectorEvidenceStore(Protocol):
    def search_by_vector(self, vector: list[float], top_k: int = 8) -> list[dict[str, Any]]: ...

    def all_rows(self, where: str | None = None) -> list[dict[str, Any]]: ...


class IngestionResult(BaseModel):
    stored_chunks_count: int = 0
    new_nodes_count: int = 0
    new_edges_count: int = 0
    status: str
    document_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    private_graph_path: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class PrivateDocumentIngestor:
    encryption_key: bytes
    private_graph_path: Path
    embedding_model: str = "local"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedder: PrivateEmbeddingProvider | None = None
    classifier: PrivateClassifier | None = None

    def __post_init__(self) -> None:
        validate_aes256_key(self.encryption_key)
        self.private_graph_path = Path(self.private_graph_path)
        self.private_graph_path.mkdir(parents=True, exist_ok=True)

    def ingest_bytes(self, filename: str, content_bytes: bytes, mime_type: str) -> IngestionResult:
        try:
            now = datetime.now(timezone.utc)
            text, content_type, extraction_metadata = extract_private_document_text(
                filename=filename,
                content_bytes=content_bytes,
                mime_type=mime_type,
            )
            if len(" ".join(text.split())) < 40:
                return IngestionResult(
                    status="failed",
                    error_message="Private document did not contain enough extractable text.",
                    private_graph_path=str(self.private_graph_path),
                )

            content_hash = sha256(content_bytes).hexdigest()
            document_id = str(uuid5(NAMESPACE_URL, f"private-document:{filename}:{content_hash}"))
            page = PageContent(
                title=Path(filename).stem or filename,
                url=f"private://documents/{document_id}/{filename}",
                search_query=f"private-file:{filename}",
                source_domain="private-file",
                fetched_at=now,
                text=text,
                content_type=content_type,
                raw_excerpt=text[:500],
                metadata={
                    "source_type": "private-file",
                    "file_name": filename,
                    "mime_type": mime_type,
                    "content_sha256": content_hash,
                    "embedding_model": self.embedding_model,
                    **extraction_metadata,
                },
            )
            classification = (
                self.classifier.classify(PRIVATE_QUERY, page)
                if self.classifier is not None
                else default_private_classification(page)
            )
            chunks = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).chunk_text(text)
            vectors = self._embed_chunks(chunks)
            run_id = str(uuid4())
            records = build_private_chunk_records(
                page=page,
                classification=classification,
                chunks=chunks,
                vectors=vectors,
                run_id=run_id,
                document_id=document_id,
                content_hash=content_hash,
            )

            append_encrypted_table(
                self.private_graph_path / "nodes" / f"PrivateDocument{ENCRYPTED_PARQUET_SUFFIX}",
                [
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "mime_type": mime_type,
                        "content_type": content_type,
                        "content_sha256": content_hash,
                        "chunk_count": len(records),
                        "source_url": page.url,
                        "created_at": now.isoformat(),
                        "metadata_json": json.dumps(page.metadata, sort_keys=True),
                    }
                ],
                PRIVATE_DOCUMENT_SCHEMA,
                self.encryption_key,
                unique_key="document_id",
            )
            append_encrypted_table(
                self.private_graph_path / "chunks" / f"PrivateChunk{ENCRYPTED_PARQUET_SUFFIX}",
                [private_chunk_row(record, document_id) for record in records],
                PRIVATE_CHUNK_SCHEMA,
                self.encryption_key,
                unique_key="chunk_id",
            )
            append_encrypted_table(
                self.private_graph_path / "edges" / f"PrivateDocument_HAS_CHUNK_PrivateChunk{ENCRYPTED_PARQUET_SUFFIX}",
                [
                    {
                        "edge_id": str(uuid5(NAMESPACE_URL, f"{document_id}:has-chunk:{record.id}")),
                        "document_id": document_id,
                        "chunk_id": record.id,
                        "source_url": page.url,
                        "created_at": now.isoformat(),
                    }
                    for record in records
                ],
                PRIVATE_DOCUMENT_CHUNK_EDGE_SCHEMA,
                self.encryption_key,
                unique_key="edge_id",
            )
            return IngestionResult(
                status="success",
                stored_chunks_count=len(records),
                new_nodes_count=1 + len(records),
                new_edges_count=len(records),
                document_id=document_id,
                chunk_ids=[record.id for record in records],
                private_graph_path=str(self.private_graph_path),
            )
        except Exception as exc:
            return IngestionResult(
                status="failed",
                error_message=str(exc),
                private_graph_path=str(self.private_graph_path),
            )

    def _embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if self.embedder is None:
            return [[] for _ in chunks]
        vectors = self.embedder.embed_texts(chunks)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding provider returned a different number of vectors than chunks.")
        return vectors


class EncryptedPrivateEvidenceStore:
    def __init__(self, private_graph_path: Path, decryption_key: bytes):
        validate_aes256_key(decryption_key)
        self._private_graph_path = Path(private_graph_path)
        self._decryption_key = decryption_key

    def search_by_vector(self, vector: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        rows = self.all_rows()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            row_vector = row.get("vector") or []
            score = cosine_similarity(vector, row_vector) if vector and row_vector else 0.0
            scored.append((score, {**row, "_distance": 1.0 - score, "graph_scope": "private"}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _score, row in scored[:top_k]]

    def all_rows(self, where: str | None = None) -> list[dict[str, Any]]:
        path = self._private_graph_path / "chunks" / f"PrivateChunk{ENCRYPTED_PARQUET_SUFFIX}"
        if not path.exists():
            return []
        table = read_encrypted_table(path, self._decryption_key)
        return [private_chunk_store_row(row) for row in table.to_pylist()]


class MergedEvidenceStore:
    def __init__(self, *stores: VectorEvidenceStore):
        self._stores = [store for store in stores if store is not None]

    def search_by_vector(self, vector: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for store in self._stores:
            rows.extend(store.search_by_vector(vector, top_k=top_k))
        rows.sort(key=lambda row: float(row.get("_distance", 1.0) or 1.0))
        deduped: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = str(row.get("id") or "")
            if row_id and row_id not in deduped:
                deduped[row_id] = row
        return list(deduped.values())[:top_k]

    def all_rows(self, where: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for store in self._stores:
            rows.extend(store.all_rows(where=where))
        return rows


def extract_private_document_text(filename: str, content_bytes: bytes, mime_type: str) -> tuple[str, str, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    normalized_mime = (mime_type or "").split(";")[0].strip().lower()
    if normalized_mime == "application/pdf" or suffix == ".pdf" or content_bytes.startswith(b"%PDF-"):
        text, metadata = extract_pdf_text(content_bytes)
        return text, "application/pdf", metadata
    if normalized_mime in {"text/html", "application/xhtml+xml"} or suffix in {".html", ".htm"}:
        html = content_bytes.decode("utf-8", errors="ignore")
        return extract_readable_text(html), normalized_mime or "text/html", {}
    if normalized_mime.startswith("text/") or suffix in {".txt", ".md", ".rst", ".text", ".csv", ".tsv", ".json"}:
        return content_bytes.decode("utf-8", errors="ignore"), normalized_mime or "text/plain", {}
    raise ValueError(f"Unsupported private document type: {mime_type or suffix or 'unknown'}")


def default_private_classification(page: PageContent) -> EvidenceClassification:
    return EvidenceClassification.model_validate(
        {
            "relevant": True,
            "relevance_score": 0.65,
            "confidence_score": 0.55,
            "application": None,
            "incumbent_material": None,
            "candidate_materials": [],
            "evidence_type": "market or customer need",
            "application_requirements": [],
            "substitution_drivers": [],
            "rationale": "Private uploaded document retained as project-specific evidence.",
            "supporting_quotes": [page.text[:300]] if page.text else [],
            "metadata": {"source_type": "private-file"},
        }
    )


def build_private_chunk_records(
    *,
    page: PageContent,
    classification: EvidenceClassification,
    chunks: list[str],
    vectors: list[list[float]],
    run_id: str,
    document_id: str,
    content_hash: str,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for index, chunk in enumerate(chunks):
        records.append(
            ChunkRecord(
                id=str(uuid5(NAMESPACE_URL, f"{run_id}:{document_id}:{index}")),
                run_id=run_id,
                original_query=PRIVATE_QUERY,
                search_query=page.search_query,
                source_title=page.title,
                source_url=page.url,
                source_domain=page.source_domain,
                retrieved_at=page.fetched_at,
                chunk_index=index,
                chunk_text=chunk,
                vector=vectors[index] if index < len(vectors) else [],
                application=classification.application,
                incumbent_material=classification.incumbent_material,
                candidate_materials=classification.candidate_materials,
                evidence_type=classification.evidence_type,
                application_requirements=classification.application_requirements,
                substitution_drivers=classification.substitution_drivers,
                relevance_score=classification.relevance_score,
                confidence_score=classification.confidence_score,
                metadata={
                    "private": True,
                    "source_type": "private-file",
                    "document_id": document_id,
                    "content_sha256": content_hash,
                    "rationale": classification.rationale,
                    "supporting_quotes": classification.supporting_quotes,
                    "classification_relevant": classification.relevant,
                    "classification_relevance_score": classification.relevance_score,
                    "classification_confidence_score": classification.confidence_score,
                    "page_metadata": page.metadata,
                },
            )
        )
    return records


def private_chunk_row(record: ChunkRecord, document_id: str) -> dict[str, Any]:
    return {
        "chunk_id": record.id,
        "document_id": document_id,
        "run_id": record.run_id,
        "original_query": record.original_query,
        "search_query": record.search_query,
        "source_title": record.source_title,
        "source_url": record.source_url,
        "source_domain": record.source_domain,
        "retrieved_at": record.retrieved_at.isoformat(),
        "chunk_index": record.chunk_index,
        "chunk_text": record.chunk_text,
        "vector_json": json.dumps([float(item) for item in record.vector]),
        "application": record.application,
        "incumbent_material": record.incumbent_material,
        "candidate_materials_json": json.dumps(record.candidate_materials, sort_keys=True),
        "evidence_type": record.evidence_type,
        "application_requirements_json": json.dumps(record.application_requirements, sort_keys=True),
        "substitution_drivers_json": json.dumps(record.substitution_drivers, sort_keys=True),
        "relevance_score": float(record.relevance_score),
        "confidence_score": float(record.confidence_score),
        "metadata_json": json.dumps(record.metadata, sort_keys=True),
    }


def private_chunk_store_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("chunk_id"),
        "run_id": row.get("run_id"),
        "original_query": row.get("original_query"),
        "search_query": row.get("search_query"),
        "source_title": row.get("source_title"),
        "source_url": row.get("source_url"),
        "source_domain": row.get("source_domain"),
        "retrieved_at": row.get("retrieved_at"),
        "chunk_index": row.get("chunk_index"),
        "chunk_text": row.get("chunk_text"),
        "vector": parse_json_list(row.get("vector_json")),
        "application": row.get("application"),
        "incumbent_material": row.get("incumbent_material"),
        "candidate_materials": parse_json_list(row.get("candidate_materials_json")),
        "evidence_type": row.get("evidence_type"),
        "application_requirements": parse_json_list(row.get("application_requirements_json")),
        "substitution_drivers": parse_json_list(row.get("substitution_drivers_json")),
        "relevance_score": row.get("relevance_score"),
        "confidence_score": row.get("confidence_score"),
        "metadata": parse_json_dict(row.get("metadata_json")),
    }


def parse_json_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
    left_norm = sum(float(a) ** 2 for a in left) ** 0.5
    right_norm = sum(float(b) ** 2 for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def validate_aes256_key(key: bytes) -> None:
    if not isinstance(key, bytes):
        raise TypeError("Private ingestion encryption key must be bytes.")
    if len(key) != 32:
        raise ValueError("Private ingestion requires a 32-byte AES-256 key.")


def encrypt_payload(data: bytes, key: bytes) -> bytes:
    validate_aes256_key(key)
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return ENCRYPTED_PAYLOAD_MAGIC + nonce + ciphertext


def decrypt_payload(encrypted: bytes, key: bytes) -> bytes:
    validate_aes256_key(key)
    if not encrypted.startswith(ENCRYPTED_PAYLOAD_MAGIC):
        raise ValueError("Encrypted payload has an unknown format.")
    nonce_start = len(ENCRYPTED_PAYLOAD_MAGIC)
    nonce_end = nonce_start + NONCE_BYTES
    nonce = encrypted[nonce_start:nonce_end]
    ciphertext = encrypted[nonce_end:]
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def table_to_parquet_bytes(rows: list[dict[str, Any]], schema: pa.Schema) -> bytes:
    sink = BytesIO()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, sink)
    return sink.getvalue()


def parquet_bytes_to_rows(payload: bytes) -> list[dict[str, Any]]:
    table = pq.read_table(BytesIO(payload))
    return table.to_pylist()


def read_encrypted_table(path: Path, key: bytes) -> pa.Table:
    plaintext = decrypt_payload(path.read_bytes(), key)
    return pq.read_table(BytesIO(plaintext))


def append_encrypted_table(
    path: Path,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    key: bytes,
    *,
    unique_key: str | None = None,
) -> None:
    if not rows:
        return
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = parquet_bytes_to_rows(decrypt_payload(path.read_bytes(), key))
    merged = [*existing, *rows]
    if unique_key:
        deduped: dict[Any, dict[str, Any]] = {}
        for row in merged:
            deduped[row.get(unique_key)] = row
        merged = list(deduped.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    plaintext = table_to_parquet_bytes(merged, schema)
    path.write_bytes(encrypt_payload(plaintext, key))


PRIVATE_DOCUMENT_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("filename", pa.string()),
        ("mime_type", pa.string()),
        ("content_type", pa.string()),
        ("content_sha256", pa.string()),
        ("chunk_count", pa.int32()),
        ("source_url", pa.string()),
        ("created_at", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

PRIVATE_CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("document_id", pa.string()),
        ("run_id", pa.string()),
        ("original_query", pa.string()),
        ("search_query", pa.string()),
        ("source_title", pa.string()),
        ("source_url", pa.string()),
        ("source_domain", pa.string()),
        ("retrieved_at", pa.string()),
        ("chunk_index", pa.int32()),
        ("chunk_text", pa.string()),
        ("vector_json", pa.string()),
        ("application", pa.string()),
        ("incumbent_material", pa.string()),
        ("candidate_materials_json", pa.string()),
        ("evidence_type", pa.string()),
        ("application_requirements_json", pa.string()),
        ("substitution_drivers_json", pa.string()),
        ("relevance_score", pa.float64()),
        ("confidence_score", pa.float64()),
        ("metadata_json", pa.string()),
    ]
)

PRIVATE_DOCUMENT_CHUNK_EDGE_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("document_id", pa.string()),
        ("chunk_id", pa.string()),
        ("source_url", pa.string()),
        ("created_at", pa.string()),
    ]
)
