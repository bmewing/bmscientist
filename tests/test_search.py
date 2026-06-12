from app_discovery_agent.models import SearchResultItem
from app_discovery_agent.search import canonicalize_url, deduplicate_search_results


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

