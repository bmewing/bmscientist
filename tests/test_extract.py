from app_discovery_agent.extract import PageFetcher, extract_pdf_text


def test_extract_pdf_text_joins_page_text(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, _stream):
            self.pages = [FakePage("PVC pipe requirements"), FakePage("chemical resistance and durability")]
            self.metadata = {"/Title": "PVC spec"}

    monkeypatch.setattr("app_discovery_agent.extract.PdfReader", FakeReader)

    text, metadata = extract_pdf_text(b"%PDF-1.7 fake")

    assert "PVC pipe requirements" in text
    assert "chemical resistance and durability" in text
    assert metadata["page_count"] == 2
    assert metadata["extraction_method"] == "pypdf"


def test_pdf_cache_entry_with_extracted_text_is_kept(tmp_path):
    from datetime import datetime, timezone
    import json

    from app_discovery_agent.agent import DiscoveryAgent

    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    path = tmp_path / "fetched_pages.json"
    path.write_text(
        json.dumps(
            [
                {
                    "title": "PVC PDF",
                    "url": "https://example.com/pvc.pdf",
                    "search_query": "pvc pdf",
                    "source_domain": "example.com",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "status_code": 200,
                    "content_type": "application/pdf",
                    "text": "PVC is used in piping because of chemical resistance.",
                    "metadata": {"source_type": "pdf", "page_count": 1},
                }
            ]
        ),
        encoding="utf-8",
    )

    pages, skipped = agent._load_cached_fetched_pages(path, max_pages=20)

    assert len(pages) == 1
    assert pages[0].metadata["source_type"] == "pdf"
    assert skipped == []
