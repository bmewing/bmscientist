from __future__ import annotations

from bmscientist.config import AppConfig
from bmscientist.models import PageContent, SearchResultItem
from bmscientist.retrieval import ExaPageRetriever
from bmscientist.search import ContentsResponse


class FakeSearchClient:
    def __init__(self, contents_response: ContentsResponse | None = None):
        self.contents_response = contents_response
        self.calls: list[dict[str, object]] = []

    def get_contents(self, urls, options):
        self.calls.append({"urls": urls, "options": options})
        if self.contents_response is None:
            raise RuntimeError("No contents response configured")
        return self.contents_response


class FakeFetcher:
    def __init__(self):
        self.safe_fetch_calls: list[str] = []
        self.safe_fetch_result: tuple[PageContent | None, dict | None] = (None, None)

    def safe_fetch(self, result: SearchResultItem):
        self.safe_fetch_calls.append(str(result.url))
        return self.safe_fetch_result


def make_config(**overrides) -> AppConfig:
    payload = {
        "deepseek_api_key": "x",
        "exa_api_key": "y",
        "min_page_characters": 100,
        "min_snippet_characters": 40,
    }
    payload.update(overrides)
    return AppConfig.model_validate(payload)


def test_retriever_uses_search_content_without_direct_fetch():
    config = make_config()
    search = FakeSearchClient()
    fetcher = FakeFetcher()
    retriever = ExaPageRetriever(config, search, fetcher)
    result = SearchResultItem(
        title="Rigid packaging requirements",
        url="https://example.com/packaging",
        search_query="rigid packaging pvc alternatives",
        snippet="Snippet",
        summary="Summary",
        content_text="PVC medical trays require clarity, toughness, and sterilization compatibility. " * 4,
        content_text_characters=320,
        highlights=["Clarity and sterilization compatibility matter."],
        score=0.8,
        request_id="req-search",
        cost_dollars=0.01,
    )

    batch = retriever.retrieve_pages("rigid packaging pvc alternatives", [result], max_pages=1)

    assert len(batch.pages) == 1
    assert batch.pages[0].metadata["source_type"] == "exa_search_contents"
    assert fetcher.safe_fetch_calls == []
    assert search.calls == []


def test_retriever_uses_contents_followup_before_direct_fetch():
    config = make_config(exa_search_content_text_chars=500, min_page_characters=200)
    search = FakeSearchClient(
        ContentsResponse(
            urls=["https://polymart.info/news"],
            results=[
                {
                    "url": "https://polymart.info/news",
                    "title": "Industry news",
                    "text": "Expanded article body with enough detail for chunking and evidence extraction." * 3,
                    "summary": "Expanded summary",
                    "highlights": [{"text": "Demand remains strong"}],
                }
            ],
            raw_payload={"requestId": "req-contents", "costDollars": 0.02, "results": []},
            request_id="req-contents",
            cost_dollars=0.02,
            statuses_by_url={"https://polymart.info/news": {"status": "success"}},
        )
    )
    fetcher = FakeFetcher()
    retriever = ExaPageRetriever(config, search, fetcher)
    result = SearchResultItem(
        title="Industry news",
        url="https://polymart.info/news",
        search_query="polymer market news",
        snippet="Short snippet",
        summary="Summary",
        content_text="Demand remains strong for PET applications. " * 12,
        content_text_characters=492,
        highlights=["Demand remains strong for PET applications."],
        highlight_scores=[0.9],
        score=0.85,
        category="news",
    )

    batch = retriever.retrieve_pages("polymer market news", [result], max_pages=1)

    assert len(batch.pages) == 1
    assert batch.pages[0].metadata["source_type"] == "exa_contents_api"
    assert batch.stats["contents_followups_attempted"] == 1
    assert fetcher.safe_fetch_calls == []
    assert search.calls[0]["urls"] == ["https://polymart.info/news"]


def test_retriever_attempts_direct_fetch_when_exa_content_is_insufficient():
    config = make_config()
    search = FakeSearchClient(contents_response=None)
    fetcher = FakeFetcher()
    retriever = ExaPageRetriever(config, search, fetcher)
    result = SearchResultItem(
        title="Sparse page",
        url="https://blocked.example.com/page",
        search_query="blocked domain query",
        snippet="Tiny note",
        summary="",
        content_text="",
        highlights=[],
        score=0.05,
    )

    batch = retriever.retrieve_pages("blocked domain query", [result], max_pages=1)

    assert len(batch.pages) == 1
    assert batch.pages[0].metadata["source_type"] == "search_snippet_partial"
    assert fetcher.safe_fetch_calls == ["https://blocked.example.com/page"]
