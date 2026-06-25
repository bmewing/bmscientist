import json

from bmscientist.config import AppConfig
from bmscientist.models import SearchResultItem
from bmscientist.search import ExaSearchClient, canonicalize_url, deduplicate_search_results, load_search_results_file


def test_canonicalize_url_removes_tracking_and_www():
    normalized = canonicalize_url("https://www.example.com/path/?utm_source=test&x=1#fragment")
    assert normalized == "https://example.com/path?x=1"


def test_deduplicate_search_results_keeps_first_unique_url():
    results = [
        SearchResultItem(title="A", url="https://example.com/item?utm_source=x", search_query="q"),
        SearchResultItem(title="B", url="https://www.example.com/item", search_query="q"),
        SearchResultItem(title="C", url="https://example.com/other", search_query="q"),
    ]

    deduped = deduplicate_search_results(results)

    assert len(deduped) == 2
    assert deduped[0].title == "A"
    assert deduped[1].title == "C"


def test_load_search_results_file_rehydrates_items(tmp_path):
    path = tmp_path / "cached_search.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "pvc applications",
                    "payload": {
                        "results": [
                            {
                                "title": "PVC overview",
                                "url": "https://example.com/pvc",
                                "summary": "Major uses of PVC.",
                            }
                        ]
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    results = load_search_results_file(path)

    assert len(results) == 1
    assert results[0].search_query == "pvc applications"
    assert results[0].summary == "Major uses of PVC."


def test_load_search_results_file_rehydrates_exa_content_fields(tmp_path):
    path = tmp_path / "cached_search.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "polymer market news",
                    "payload": {
                        "requestId": "req-search-1",
                        "costDollars": {"total": 0.02},
                        "results": [
                            {
                                "id": "res-1",
                                "title": "Polymart industry update",
                                "url": "https://polymart.info/news",
                                "summary": "A summary.",
                                "text": "Expanded extracted content from Exa search.",
                                "highlights": [{"text": "Key highlighted passage", "score": 0.81}],
                                "category": "news",
                            }
                        ],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    results = load_search_results_file(path)

    assert len(results) == 1
    assert results[0].exa_id == "res-1"
    assert results[0].request_id == "req-search-1"
    assert results[0].cost_dollars == 0.02
    assert results[0].content_text == "Expanded extracted content from Exa search."
    assert results[0].highlights == ["Key highlighted passage"]
    assert results[0].highlight_scores == [0.81]
    assert results[0].category == "news"


def test_exa_search_client_requests_contents_payload_for_search():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "requestId": "req-123",
                "costDollars": {"total": 0.01, "search": {"neural": 0.01}},
                "results": [
                    {
                        "id": "res-1",
                        "title": "Market news",
                        "url": "https://example.com/news",
                        "text": "A long extracted result body.",
                        "highlights": [{"text": "Important highlight", "score": 0.7}],
                        "category": "news",
                    }
                ],
            }

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def post(self, url, json, timeout):
            self.calls.append({"url": url, "json": json, "timeout": timeout})
            return FakeResponse()

    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
        exa_news_domains=["polymart.info"],
    )
    client = ExaSearchClient(config)
    fake_session = FakeSession()
    client._thread_local.session = fake_session

    response = client.search("polymer market news polymart.info", num_results=5)

    payload = fake_session.calls[0]["json"]
    assert payload["type"] == config.exa_default_search_type
    assert payload["category"] == "news"
    assert payload["contents"]["text"]["maxCharacters"] == config.exa_search_content_text_chars
    assert payload["contents"]["highlights"]["maxCharacters"] == config.exa_highlights_max_chars
    assert payload["contents"]["highlights"]["query"] == "polymer market news polymart.info"
    assert response.request_id == "req-123"
    assert response.cost_dollars == 0.01
    assert response.results[0].content_text == "A long extracted result body."
    assert response.results[0].highlights == ["Important highlight"]
