from __future__ import annotations

from bmscientist.graph_query import DuckDBGraphQueryEngine
from bmscientist.ingestion import EncryptedPrivateEvidenceStore, PrivateDocumentIngestor, read_encrypted_table


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_private_document_ingestor_writes_encrypted_chunks_and_graph_tables(tmp_path):
    key = b"1" * 32
    private_graph_path = tmp_path / "private-graph"
    content = (
        "Confidential coating formulation notes: Binder X uses a private coalescent package. "
        "The uploaded manual says the target film needs corrosion resistance and low VOC."
    ).encode("utf-8")

    result = PrivateDocumentIngestor(
        encryption_key=key,
        private_graph_path=private_graph_path,
        chunk_size=80,
        chunk_overlap=0,
        embedder=FakeEmbedder(),
    ).ingest_bytes("private-note.txt", content, "text/plain")

    assert result.status == "success"
    assert result.stored_chunks_count > 0
    assert result.new_nodes_count == result.stored_chunks_count + 1
    assert result.new_edges_count == result.stored_chunks_count

    chunk_path = private_graph_path / "chunks" / "PrivateChunk.parquet.aesgcm"
    document_path = private_graph_path / "nodes" / "PrivateDocument.parquet.aesgcm"
    assert chunk_path.exists()
    assert document_path.exists()
    assert b"Confidential coating formulation" not in chunk_path.read_bytes()

    chunks = read_encrypted_table(chunk_path, key).to_pylist()
    assert chunks[0]["chunk_text"].startswith("Confidential coating formulation notes")
    assert chunks[0]["document_id"] == result.document_id


def test_private_graph_query_engine_reads_encrypted_private_tables(tmp_path):
    key = b"2" * 32
    private_graph_path = tmp_path / "private-graph"
    PrivateDocumentIngestor(
        encryption_key=key,
        private_graph_path=private_graph_path,
        embedder=FakeEmbedder(),
    ).ingest_bytes(
        "secure-manual.md",
        b"Private customer manual says Product Z is used in aerospace seals and requires low extractables.",
        "text/markdown",
    )

    engine = DuckDBGraphQueryEngine(tmp_path / "public-graph", private_graph_path=private_graph_path, decryption_key=key)
    table_names = {table.table_name for table in engine.list_tables()}

    assert "PrivateDocument" in table_names
    assert "PrivateChunk" in table_names

    result = engine.query("SELECT filename FROM PrivateDocument", limit=5)

    assert result.rows == [{"filename": "secure-manual.md"}]


def test_encrypted_private_evidence_store_searches_private_chunks(tmp_path):
    key = b"3" * 32
    private_graph_path = tmp_path / "private-graph"
    PrivateDocumentIngestor(
        encryption_key=key,
        private_graph_path=private_graph_path,
        embedder=FakeEmbedder(),
    ).ingest_bytes(
        "retrieval-note.txt",
        b"Private test note about corrosion resistant aerospace seal materials and low extractables.",
        "text/plain",
    )

    rows = EncryptedPrivateEvidenceStore(private_graph_path, key).search_by_vector([0.1, 0.2, 0.3], top_k=3)

    assert len(rows) == 1
    assert rows[0]["graph_scope"] == "private"
    assert "aerospace seal materials" in rows[0]["chunk_text"]
    assert rows[0]["metadata"]["private"] is True


def test_private_graph_query_engine_requires_decryption_key(tmp_path):
    try:
        DuckDBGraphQueryEngine(tmp_path / "public-graph", private_graph_path=tmp_path / "private-graph")
    except ValueError as exc:
        assert "decryption key" in str(exc)
    else:
        raise AssertionError("Expected private graph query engine to require a decryption key.")
