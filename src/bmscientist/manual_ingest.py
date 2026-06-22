from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from bmscientist.chunking import TextChunker
from bmscientist.classify import EvidenceClassifier
from bmscientist.config import AppConfig
from bmscientist.embeddings import LocalEmbedder
from bmscientist.extract import extract_pdf_text, extract_readable_text
from bmscientist.models import ChunkRecord, PageContent
from bmscientist.store import LanceEvidenceStore


LOGGER = logging.getLogger(__name__)
MANUAL_QUERY = "manually obtained application development evidence for material substitution research"
TEXT_SUFFIXES = {".txt", ".md", ".rst", ".text", ".csv", ".tsv", ".json", ".yaml", ".yml"}
HTML_SUFFIXES = {".html", ".htm"}
MIN_MANUAL_TEXT_CHARACTERS = 40


class ManualEvidenceIngestor:
    DEFAULT_ROOT = Path("data/manually-obtained")

    def __init__(
        self,
        config: AppConfig,
        classifier: EvidenceClassifier,
        chunker: TextChunker,
        embedder: LocalEmbedder,
        store: LanceEvidenceStore,
        graph_enrichment_callback: Callable[[str, list[ChunkRecord]], None] | None = None,
        root_path: Path | None = None,
    ):
        self._config = config
        self._classifier = classifier
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._graph_enrichment_callback = graph_enrichment_callback
        self._root = root_path if root_path is not None else self.DEFAULT_ROOT
        self._processed_root = self._root / "processed"
        self._root.mkdir(parents=True, exist_ok=True)
        self._processed_root.mkdir(parents=True, exist_ok=True)

    def ingest_pending_files(self) -> int:
        pending = self._pending_files()
        if not pending:
            return 0

        stored_chunks = 0
        for path in pending:
            try:
                stored_chunks += self._ingest_file(path)
            except ValueError as exc:
                LOGGER.warning("%s", exc)
                target_path = path if self._processed_root in path.parents else self._target_path(path)
                if path != target_path:
                    path.replace(target_path)
            except Exception:
                LOGGER.exception("Manual evidence ingest failed for %s", path)
        if stored_chunks:
            LOGGER.info("Stored %s chunks from manually obtained files", stored_chunks)
        return stored_chunks

    def _pending_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if self._processed_root in path.parents:
                continue
            files.append(path)
        files.extend(self._processed_files_missing_from_store())
        return sorted(files)

    def _processed_files_missing_from_store(self) -> list[Path]:
        if not hasattr(self._store, "all_rows"):
            return []
        try:
            rows = self._store.all_rows(where=f"original_query = '{MANUAL_QUERY}'")
        except Exception:
            LOGGER.exception("Unable to inspect stored manual evidence; skipping processed-file recovery")
            return []
        stored_paths = {
            str(row.get("source_url", ""))
            for row in rows
            if row.get("metadata", {}).get("page_metadata", {}).get("source_type") == "manual-file"
        }
        stored_paths.update(
            str(row.get("metadata", {}).get("page_metadata", {}).get("local_processed_path", ""))
            for row in rows
            if row.get("metadata", {}).get("page_metadata", {}).get("local_processed_path")
        )
        missing: list[Path] = []
        for path in self._processed_root.rglob("*"):
            if not path.is_file():
                continue
            if str(path.resolve()) not in stored_paths:
                missing.append(path)
        return missing

    def _ingest_file(self, path: Path) -> int:
        extracted_text, content_type, extra_metadata = self._read_file(path)
        target_path = path if self._processed_root in path.parents else self._target_path(path)
        if len(" ".join(extracted_text.split())) < MIN_MANUAL_TEXT_CHARACTERS:
            LOGGER.warning("Skipping manual file with too little text: %s", path)
            if path != target_path:
                path.replace(target_path)
            return 0

        page = PageContent(
            title=path.stem,
            url=str(target_path.resolve()),
            search_query=f"manual-file:{path.name}",
            source_domain="local-file",
            fetched_at=datetime.now(timezone.utc),
            text=extracted_text,
            status_code=None,
            content_type=content_type,
            raw_excerpt=extracted_text[:500],
            metadata={
                "source_type": "manual-file",
                "local_original_path": str(path.resolve()),
                "local_processed_path": str(target_path.resolve()),
                "file_name": path.name,
                "file_suffix": path.suffix.lower(),
                **extra_metadata,
            },
        )
        classification = self._classifier.classify(MANUAL_QUERY, page)
        chunk_records = self._build_chunk_records(page, classification)
        vectors = self._embedder.embed_texts([record.chunk_text for record in chunk_records])
        embedded_records = [record.model_copy(update={"vector": vector}) for record, vector in zip(chunk_records, vectors, strict=False)]

        if path != target_path:
            path.replace(target_path)
        try:
            stored_count = self._store.add_chunks(embedded_records)
            if self._graph_enrichment_callback is not None:
                self._graph_enrichment_callback(MANUAL_QUERY, embedded_records)
            return stored_count
        except Exception:
            if target_path.exists() and not path.exists():
                target_path.replace(path)
            raise

    def _target_path(self, path: Path) -> Path:
        candidate = self._processed_root / path.name
        if not candidate.exists():
            return candidate
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return self._processed_root / f"{path.stem}_{timestamp}{path.suffix}"

    def _read_file(self, path: Path) -> tuple[str, str, dict]:
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            return path.read_text(encoding="utf-8", errors="ignore"), "text/plain", {}
        if suffix in HTML_SUFFIXES:
            html = path.read_text(encoding="utf-8", errors="ignore")
            return extract_readable_text(html), "text/html", {}
        if suffix == ".pdf":
            text, metadata = extract_pdf_text(path.read_bytes())
            return text, "application/pdf", metadata
        raise ValueError(f"Unsupported manually obtained file type: {path.suffix}")

    def _build_chunk_records(self, page: PageContent, classification) -> list[ChunkRecord]:
        run_id = str(uuid4())
        records: list[ChunkRecord] = []
        for index, chunk in enumerate(self._chunker.chunk_text(page.text)):
            records.append(
                ChunkRecord(
                    id=str(uuid5(NAMESPACE_URL, f"{run_id}::{page.url}::{index}")),
                    run_id=run_id,
                    original_query=MANUAL_QUERY,
                    search_query=page.search_query,
                    source_title=page.title,
                    source_url=page.url,
                    source_domain=page.source_domain,
                    retrieved_at=page.fetched_at,
                    chunk_index=index,
                    chunk_text=chunk,
                    application=classification.application,
                    incumbent_material=classification.incumbent_material,
                    candidate_materials=classification.candidate_materials,
                    evidence_type=classification.evidence_type,
                    application_requirements=classification.application_requirements,
                    substitution_drivers=classification.substitution_drivers,
                    relevance_score=classification.relevance_score,
                    confidence_score=classification.confidence_score,
                    metadata={
                        "rationale": classification.rationale,
                        "supporting_quotes": classification.supporting_quotes,
                        "classification_relevant": classification.relevant,
                        "classification_relevance_score": classification.relevance_score,
                        "classification_confidence_score": classification.confidence_score,
                        "retained_for_reflection": True,
                        "retention_policy": "retain_manual_file_for_reflection",
                        "page_metadata": page.metadata,
                    },
                )
            )
        return records
