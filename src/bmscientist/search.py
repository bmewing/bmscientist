from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from bmscientist.config import AppConfig
from bmscientist.models import SearchResultItem


LOGGER = logging.getLogger(__name__)
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResultItem]
    raw_payload: dict[str, Any]


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(filtered_query, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def deduplicate_search_results(results: list[SearchResultItem]) -> list[SearchResultItem]:
    deduplicated: dict[str, SearchResultItem] = {}
    for item in results:
        normalized = canonicalize_url(str(item.url))
        if normalized not in deduplicated:
            deduplicated[normalized] = item
    return list(deduplicated.values())


class ExaSearchClient:
    def __init__(self, config: AppConfig):
        self._api_key = config.exa_api_key
        self._timeout = config.request_timeout_seconds
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": config.user_agent,
            }
        )

    def search(self, query: str, num_results: int) -> SearchResponse:
        payload = {
            "query": query,
            "numResults": num_results,
            "moderation": True,
        }
        response = self._session.post("https://api.exa.ai/search", json=payload, timeout=self._timeout)
        response.raise_for_status()
        raw_payload = response.json()
        items = [self._to_result_item(query, item) for item in raw_payload.get("results", [])]
        LOGGER.info("Exa returned %s results for query: %s", len(items), query)
        return SearchResponse(query=query, results=items, raw_payload=raw_payload)

    @staticmethod
    def _to_result_item(query: str, item: dict[str, Any]) -> SearchResultItem:
        return SearchResultItem(
            title=item.get("title") or item.get("url") or "Untitled",
            url=item["url"],
            search_query=query,
            snippet=item.get("text", "") or item.get("summary", ""),
            summary=item.get("summary", "") or "",
            published_date=item.get("publishedDate"),
            score=item.get("score"),
            raw=item,
        )


def load_search_results_file(path: Path) -> list[SearchResultItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[SearchResultItem] = []
    for entry in payload:
        query = entry.get("query", "")
        raw_payload = entry.get("payload", {})
        for result in raw_payload.get("results", []):
            items.append(ExaSearchClient._to_result_item(query, result))
    return items
