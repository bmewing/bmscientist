from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from bmscientist.config import AppConfig
from bmscientist.cost_tracking import CostTracker
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
class ExaSearchOptions:
    category: str | None = None
    search_type: str = "auto"
    text_max_characters: int = 8000
    highlights_max_characters: int = 2000
    include_text: bool = True
    include_highlights: bool = True
    include_summary: bool = False
    max_age_hours: int | None = None
    start_published_date: str | None = None
    end_published_date: str | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None


@dataclass(slots=True)
class ExaContentsOptions:
    query: str
    text_max_characters: int = 12000
    highlights_max_characters: int = 2000
    summary_max_tokens: int | None = 250
    max_age_hours: int | None = 168


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResultItem]
    raw_payload: dict[str, Any]
    request_id: str | None = None
    cost_dollars: float | None = None


@dataclass(slots=True)
class ContentsResponse:
    urls: list[str]
    results: list[dict[str, Any]]
    raw_payload: dict[str, Any]
    request_id: str | None = None
    cost_dollars: float | None = None
    statuses_by_url: dict[str, Any] | None = None


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
    def __init__(self, config: AppConfig, cost_tracker: CostTracker | None = None):
        self._api_key = config.exa_api_key
        self._timeout = config.request_timeout_seconds
        self._config = config
        self._cost_tracker = cost_tracker
        self._thread_local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self._thread_local, "session"):
            session = requests.Session()
            session.headers.update(
                {
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                    "User-Agent": self._config.user_agent,
                }
            )
            self._thread_local.session = session
        return self._thread_local.session

    def search(self, query: str, num_results: int, options: ExaSearchOptions | None = None) -> SearchResponse:
        options = options or default_search_options(self._config, query)
        payload = {
            "query": query,
            "numResults": num_results,
            "moderation": True,
            "type": options.search_type,
        }
        if options.category:
            payload["category"] = options.category
        if options.start_published_date:
            payload["startPublishedDate"] = options.start_published_date
        if options.end_published_date:
            payload["endPublishedDate"] = options.end_published_date
        if options.include_domains:
            payload["includeDomains"] = options.include_domains
        if options.exclude_domains:
            payload["excludeDomains"] = options.exclude_domains
        if options.max_age_hours is not None:
            payload["maxAgeHours"] = options.max_age_hours
        contents_payload = self._build_search_contents_payload(query, options)
        if contents_payload:
            payload["contents"] = contents_payload

        response = self.session.post("https://api.exa.ai/search", json=payload, timeout=self._timeout)
        response.raise_for_status()
        raw_payload = response.json()
        request_id = raw_payload.get("requestId")
        cost_dollars = _extract_cost_dollars(raw_payload.get("costDollars"))
        if self._cost_tracker is not None:
            self._cost_tracker.record_exa_call(operation="search", cost_dollars=cost_dollars)
        items = [
            self._to_result_item(
                query,
                item,
                request_id=request_id,
                cost_dollars=cost_dollars,
            )
            for item in raw_payload.get("results", [])
        ]
        LOGGER.info("Exa returned %s results for query: %s", len(items), query)
        return SearchResponse(
            query=query,
            results=items,
            raw_payload=raw_payload,
            request_id=request_id,
            cost_dollars=cost_dollars,
        )

    def get_contents(self, urls: list[str], options: ExaContentsOptions) -> ContentsResponse:
        payload: dict[str, Any] = {"urls": urls}
        if options.max_age_hours is not None:
            payload["maxAgeHours"] = options.max_age_hours
        if options.text_max_characters > 0:
            payload["text"] = {
                "maxCharacters": options.text_max_characters,
                "includeHtmlTags": False,
            }
        if options.highlights_max_characters > 0:
            payload["highlights"] = {
                "query": options.query,
                "maxCharacters": options.highlights_max_characters,
            }
        if options.summary_max_tokens:
            payload["summary"] = {
                "query": f"Summarize material, application, company, performance, regulatory, and market evidence relevant to: {options.query}",
                "maxTokens": options.summary_max_tokens,
            }

        response = self.session.post("https://api.exa.ai/contents", json=payload, timeout=self._timeout)
        response.raise_for_status()
        raw_payload = response.json()
        statuses_by_url = self._statuses_by_url(raw_payload.get("statuses", []))
        cost_dollars = _extract_cost_dollars(raw_payload.get("costDollars"))
        if self._cost_tracker is not None:
            self._cost_tracker.record_exa_call(operation="contents", cost_dollars=cost_dollars)
        return ContentsResponse(
            urls=urls,
            results=list(raw_payload.get("results", [])),
            raw_payload=raw_payload,
            request_id=raw_payload.get("requestId"),
            cost_dollars=cost_dollars,
            statuses_by_url=statuses_by_url,
        )

    @staticmethod
    def _build_search_contents_payload(query: str, options: ExaSearchOptions) -> dict[str, Any]:
        contents: dict[str, Any] = {}
        if options.include_text and options.text_max_characters > 0:
            contents["text"] = {
                "maxCharacters": options.text_max_characters,
                "includeHtmlTags": False,
            }
        if options.include_highlights and options.highlights_max_characters > 0:
            contents["highlights"] = {
                "query": query,
                "maxCharacters": options.highlights_max_characters,
            }
        if options.include_summary:
            contents["summary"] = {
                "query": query,
                "maxTokens": 200,
            }
        return contents

    @staticmethod
    def _to_result_item(
        query: str,
        item: dict[str, Any],
        *,
        request_id: str | None = None,
        cost_dollars: float | None = None,
    ) -> SearchResultItem:
        highlights, highlight_scores = _normalize_highlights(item.get("highlights"))
        content_text = (item.get("text") or "").strip()
        summary = _normalize_summary(item.get("summary"))
        snippet = _build_snippet(highlights, summary, content_text)
        content_source = "search_contents" if content_text else None
        return SearchResultItem(
            title=item.get("title") or item.get("url") or "Untitled",
            url=item["url"],
            search_query=query,
            snippet=snippet,
            summary=summary,
            exa_id=item.get("id"),
            request_id=request_id,
            cost_dollars=cost_dollars,
            highlights=highlights,
            highlight_scores=highlight_scores,
            content_text=content_text,
            content_text_characters=len(content_text) or None,
            published_date=item.get("publishedDate"),
            score=_coerce_float(item.get("score")),
            category=item.get("category"),
            image_url=item.get("image"),
            favicon_url=item.get("favicon"),
            content_source=content_source,
            raw=item,
        )

    @staticmethod
    def _statuses_by_url(statuses: list[Any]) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for status in statuses:
            if isinstance(status, dict):
                url = status.get("url") or status.get("id")
                if url:
                    mapping[str(url)] = status
            elif isinstance(status, str):
                mapping[status] = {"status": status}
        return mapping


def _normalize_summary(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("summary")
        return str(text).strip() if text else ""
    return ""


def _normalize_highlights(value: Any) -> tuple[list[str], list[float]]:
    if not isinstance(value, list):
        return [], []
    texts: list[str] = []
    scores: list[float] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                texts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("highlight") or "").strip()
        if text:
            texts.append(text)
        score = _coerce_float(item.get("score"))
        if score is not None:
            scores.append(score)
    return texts, scores


def _build_snippet(highlights: list[str], summary: str, content_text: str) -> str:
    if highlights:
        return "\n\n".join(item.strip() for item in highlights if item.strip())[:1000]
    if summary:
        return summary[:1000]
    return content_text[:1000]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_cost_dollars(value: Any) -> float | None:
    if isinstance(value, dict):
        total = value.get("total")
        if total is None:
            return None
        return _coerce_float(total)
    return _coerce_float(value)


def _is_news_query(query: str, news_domains: list[str]) -> bool:
    lowered = query.lower()
    news_terms = (
        "news",
        "industry",
        "market",
        "headlines",
        "pricing",
        "price",
        "capacity",
        "expansion",
        "launch",
        "acquisition",
        "regulatory",
        "recycling",
        "sustainability",
    )
    if any(term in lowered for term in news_terms):
        return True
    return any(domain and domain in lowered for domain in news_domains)


def default_search_options(
    config: AppConfig,
    query: str,
    *,
    search_type: str | None = None,
    category: str | None = None,
) -> ExaSearchOptions:
    resolved_category = category or config.exa_search_category
    if resolved_category is None and _is_news_query(query, config.exa_news_domains):
        resolved_category = "news"
    max_age_hours = config.exa_news_max_age_hours if resolved_category == "news" else config.exa_default_max_age_hours
    return ExaSearchOptions(
        category=resolved_category,
        search_type=search_type or config.exa_default_search_type,
        text_max_characters=config.exa_search_content_text_chars if config.exa_enable_search_contents else 0,
        highlights_max_characters=config.exa_highlights_max_chars,
        include_text=config.exa_enable_search_contents,
        include_highlights=True,
        include_summary=False,
        max_age_hours=max_age_hours,
    )


def load_search_results_file(path: Path) -> list[SearchResultItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[SearchResultItem] = []
    for entry in payload:
        query = entry.get("query", "")
        raw_payload = entry.get("payload", {})
        request_id = raw_payload.get("requestId")
        cost_dollars = _extract_cost_dollars(raw_payload.get("costDollars"))
        for result in raw_payload.get("results", []):
            items.append(
                ExaSearchClient._to_result_item(
                    query,
                    result,
                    request_id=request_id,
                    cost_dollars=cost_dollars,
                )
            )
    return items
