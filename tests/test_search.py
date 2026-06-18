import json

from bmscientist.config import AppConfig
from bmscientist.extract import PageFetcher
from bmscientist.models import SearchResultItem
from bmscientist.search import canonicalize_url, deduplicate_search_results, load_search_results_file


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


def test_skip_fetch_domain_matches_subdomain():
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
        skip_fetch_domains=["sciencedirect.com"],
    )
    fetcher = PageFetcher(config)

    assert fetcher.should_skip_direct_fetch("https://www.sciencedirect.com/topics/test") is True
    assert fetcher.should_skip_direct_fetch("https://example.com/page") is False


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
