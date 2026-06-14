from pathlib import Path

from app_discovery_agent.chunking import TextChunker
from app_discovery_agent.config import AppConfig
from app_discovery_agent.manual_ingest import ManualEvidenceIngestor
from app_discovery_agent.models import EvidenceClassification


class FakeClassifier:
    def classify(self, query, page):
        return EvidenceClassification.model_validate(
            {
                "relevant": True,
                "relevance_score": 0.82,
                "confidence_score": 0.74,
                "application": "medical trays",
                "incumbent_material": "PVC",
                "candidate_materials": ["PETG"],
                "evidence_type": "application requirements",
                "application_requirements": ["clarity", "thermoformability"],
                "substitution_drivers": ["PVC reduction"],
                "rationale": "Manual document contains direct application evidence.",
                "supporting_quotes": ["PVC trays require clarity."],
                "metadata": {"source_type": "manual-file"},
            }
        )


class LowRelevanceClassifier:
    def classify(self, query, page):
        return EvidenceClassification.model_validate(
            {
                "relevant": False,
                "relevance_score": 0.12,
                "confidence_score": 0.3,
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [],
                "evidence_type": "market or customer need",
                "application_requirements": [],
                "substitution_drivers": [],
                "rationale": "Classifier judged this document only indirectly useful.",
                "supporting_quotes": [],
                "metadata": {"source_type": "manual-file"},
            }
        )


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self):
        self.records = []

    def add_chunks(self, records):
        self.records.extend(records)
        return len(records)

    def all_rows(self):
        return [
            {
                "source_url": record.source_url,
                "metadata": record.metadata,
            }
            for record in self.records
        ]


def test_manual_ingest_moves_file_and_stores_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    config.ensure_directories()

    source_path = Path("data/manually-obtained/manual-note.txt")
    source_path.write_text(
        "PVC is used in clear rigid medical trays where clarity, thermoformability, and sterilization compatibility matter.",
        encoding="utf-8",
    )

    store = FakeStore()
    ingestor = ManualEvidenceIngestor(
        config,
        FakeClassifier(),
        TextChunker(chunk_size=50, chunk_overlap=0),
        FakeEmbedder(),
        store,
    )

    stored_count = ingestor.ingest_pending_files()

    processed_path = tmp_path / "data" / "manually-obtained" / "processed" / "manual-note.txt"
    assert stored_count == len(store.records)
    assert stored_count > 0
    assert not source_path.exists()
    assert processed_path.exists()
    assert store.records[0].source_url == str(processed_path.resolve())


def test_manual_ingest_retains_low_relevance_files_for_reflection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    config.ensure_directories()

    source_path = Path("data/manually-obtained/market-note.txt")
    source_path.write_text(
        "The adjacent packaging market generated USD 800 million in revenue and is forecast to grow at a 6% CAGR.",
        encoding="utf-8",
    )

    store = FakeStore()
    ingestor = ManualEvidenceIngestor(
        config,
        LowRelevanceClassifier(),
        TextChunker(chunk_size=80, chunk_overlap=0),
        FakeEmbedder(),
        store,
    )

    stored_count = ingestor.ingest_pending_files()

    assert stored_count == len(store.records)
    assert stored_count > 0
    assert store.records[0].metadata["retained_for_reflection"] is True
    assert store.records[0].metadata["classification_relevant"] is False


def test_manual_ingest_recovers_processed_files_missing_from_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    config.ensure_directories()

    processed_path = Path("data/manually-obtained/processed/previously-skipped.txt")
    processed_path.write_text(
        "The market generated USD 900 million in revenue and the fastest growing segment uses clear plastic film.",
        encoding="utf-8",
    )

    store = FakeStore()
    ingestor = ManualEvidenceIngestor(
        config,
        LowRelevanceClassifier(),
        TextChunker(chunk_size=80, chunk_overlap=0),
        FakeEmbedder(),
        store,
    )

    stored_count = ingestor.ingest_pending_files()

    assert stored_count == len(store.records)
    assert stored_count > 0
    assert processed_path.exists()
    assert store.records[0].source_url == str(processed_path.resolve())


def test_manual_ingest_does_not_reprocess_stored_processed_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    config.ensure_directories()

    processed_path = Path("data/manually-obtained/processed/already-stored.txt")
    processed_path.write_text(
        "PVC is used in clear rigid medical trays where clarity and thermoformability matter.",
        encoding="utf-8",
    )

    store = FakeStore()
    ingestor = ManualEvidenceIngestor(
        config,
        FakeClassifier(),
        TextChunker(chunk_size=80, chunk_overlap=0),
        FakeEmbedder(),
        store,
    )

    assert ingestor.ingest_pending_files() > 0
    first_count = len(store.records)
    assert ingestor.ingest_pending_files() == 0
    assert len(store.records) == first_count


def test_manual_ingest_skips_when_no_pending_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    config.ensure_directories()

    store = FakeStore()
    ingestor = ManualEvidenceIngestor(
        config,
        FakeClassifier(),
        TextChunker(),
        FakeEmbedder(),
        store,
    )

    assert ingestor.ingest_pending_files() == 0
    assert store.records == []
