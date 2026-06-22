from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from bmscientist.config import AppConfig
from bmscientist.extract import PageFetcher, extract_domain
from bmscientist.models import PageContent, SearchResultItem
from bmscientist.search import ExaContentsOptions, ExaSearchClient, canonicalize_url


LOGGER = logging.getLogger(__name__)
RetrievalAction = Literal["use_search_text", "fetch_contents", "direct_fetch_fallback", "partial_only", "skip"]


@dataclass(slots=True)
class RetrievalDecision:
    action: RetrievalAction
    score: float
    rationale: str


@dataclass(slots=True)
class RetrievalBatchResult:
    pages: list[PageContent]
    skipped: list[dict[str, Any]]
    stats: dict[str, Any]


class ExaResultAssessor:
    def __init__(self, config: AppConfig):
        self._config = config

    def assess(self, query: str, result: SearchResultItem) -> RetrievalDecision:
        content_text = result.content_text.strip()
        highlights_text = "\n\n".join(result.highlights).strip()
        partial_text = compose_partial_text(result)

        text_score = self._heuristic_relevance(query, content_text)
        highlights_score = self._heuristic_relevance(query, highlights_text)
        partial_score = self._heuristic_relevance(query, partial_text)
        exa_score = max(min(result.score or 0.0, 1.0), 0.0)
        composite_score = min(max(text_score, highlights_score, partial_score) * 0.75 + (exa_score * 0.25), 1.0)

        if result.category == "news" and self._is_news_query(query):
            composite_score = min(composite_score + 0.1, 1.0)

        strong_highlights = bool(result.highlights) and (
            highlights_score >= 0.3 or max(result.highlight_scores or [0.0]) >= 0.35
        )
        has_search_text = len(content_text) >= self._config.min_page_characters
        looks_truncated = bool(content_text) and len(content_text) >= int(self._config.exa_search_content_text_chars * 0.95)

        if has_search_text and not looks_truncated and composite_score >= 0.2:
            return RetrievalDecision("use_search_text", composite_score, "search_contents_sufficient")

        if self._config.exa_enable_contents_followup and (
            strong_highlights
            or composite_score >= self._config.exa_deep_fetch_min_score
            or (bool(content_text) and looks_truncated)
        ):
            return RetrievalDecision("fetch_contents", composite_score, "follow_up_for_deeper_content")

        if len(partial_text) >= self._config.min_snippet_characters and (strong_highlights or composite_score >= 0.15):
            return RetrievalDecision("partial_only", composite_score, "partial_evidence_from_highlights")

        if self._config.exa_enable_direct_fetch_fallback:
            return RetrievalDecision("direct_fetch_fallback", composite_score, "exa_content_insufficient_try_direct")

        return RetrievalDecision("skip", composite_score, "insufficient_exa_signal")

    @staticmethod
    def _heuristic_relevance(query: str, text: str) -> float:
        query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*", query) if len(term) > 3}
        if not query_terms or not text:
            return 0.0
        lowered = text.lower()
        hits = sum(1 for term in query_terms if term in lowered)
        density_bonus = min(len(text) / 4000.0, 0.35)
        return min((hits / max(len(query_terms), 1)) * 0.7 + density_bonus, 1.0)

    @staticmethod
    def _is_news_query(query: str) -> bool:
        lowered = query.lower()
        return any(
            term in lowered
            for term in (
                "news",
                "industry",
                "market",
                "pricing",
                "price",
                "launch",
                "acquisition",
                "regulatory",
                "sustainability",
                "recycling",
            )
        )


class ExaPageRetriever:
    def __init__(self, config: AppConfig, search_client: ExaSearchClient, fetcher: PageFetcher):
        self._config = config
        self._search = search_client
        self._fetcher = fetcher
        self._assessor = ExaResultAssessor(config)

    def retrieve_pages(
        self,
        query: str,
        results: list[SearchResultItem],
        *,
        max_pages: int,
        contents_query: str | None = None,
    ) -> RetrievalBatchResult:
        pages: list[PageContent] = []
        skipped: list[dict[str, Any]] = []
        contents_candidates: list[tuple[SearchResultItem, RetrievalDecision]] = []
        direct_fallback_candidates: list[tuple[SearchResultItem, RetrievalDecision]] = []
        stats = self._initial_stats(results)

        for result in results[:max_pages]:
            decision = self._assessor.assess(query, result)
            stats["decisions"][decision.action] = stats["decisions"].get(decision.action, 0) + 1

            if decision.action == "use_search_text":
                page = build_page_from_exa_search_result(
                    result,
                    decision,
                    search_text_max_characters=self._config.exa_search_content_text_chars,
                )
                if page:
                    pages.append(page)
                    stats["pages_by_source"]["exa_search_contents"] += 1
                    continue
            elif decision.action == "fetch_contents":
                contents_candidates.append((result, decision))
                continue
            elif decision.action == "partial_only":
                page = build_partial_page_from_search_result(result, decision.rationale)
                if page:
                    pages.append(page)
                    stats["pages_by_source"]["search_snippet_partial"] += 1
                    continue
            elif decision.action == "direct_fetch_fallback":
                direct_fallback_candidates.append((result, decision))
                continue

            skipped.append(
                {
                    "url": str(result.url),
                    "search_query": result.search_query,
                    "reason": decision.action,
                    "note": decision.rationale,
                }
            )

        selected_contents = self._select_contents_candidates(contents_candidates)
        if selected_contents:
            contents_pages, contents_skips, contents_stats = self._fetch_contents_for_results(
                query=contents_query or query,
                selected_results=[result for result, _ in selected_contents],
            )
            pages.extend(contents_pages)
            skipped.extend(contents_skips)
            stats["contents_followups_attempted"] += contents_stats["contents_followups_attempted"]
            stats["contents_followups_succeeded"] += contents_stats["contents_followups_succeeded"]
            stats["pages_by_source"]["exa_contents_api"] += contents_stats["pages_by_source"]["exa_contents_api"]
            stats["pages_by_source"]["search_snippet_partial"] += contents_stats["pages_by_source"]["search_snippet_partial"]
            stats["contents_request_ids"].extend(contents_stats["contents_request_ids"])
            stats["contents_cost_dollars"] += contents_stats["contents_cost_dollars"]

            retrieved_urls = {str(page.url) for page in contents_pages}
            for result, decision in selected_contents:
                if str(result.url) not in retrieved_urls and self._config.exa_enable_direct_fetch_fallback:
                    direct_fallback_candidates.append((result, decision))

        for result, decision in direct_fallback_candidates:
            if self._fetcher.should_skip_direct_fetch(str(result.url)):
                page = build_partial_page_from_search_result(result, "blocked_domain")
                if page:
                    pages.append(page)
                    stats["pages_by_source"]["search_snippet_partial"] += 1
                else:
                    skipped.append(
                        {
                            "url": str(result.url),
                            "search_query": result.search_query,
                            "reason": "blocked_domain",
                            "note": "Direct fetch skipped by policy after Exa content was insufficient.",
                        }
                    )
                continue

            stats["direct_fetch_attempted"] += 1
            page, error = self._fetcher.safe_fetch(result)
            if page:
                page = page.model_copy(
                    update={
                        "metadata": {
                            **page.metadata,
                            "retrieval_decision": decision.rationale,
                        }
                    }
                )
                pages.append(page)
                source_type = page.metadata.get("source_type") or "direct_html"
                stats["pages_by_source"][source_type] = stats["pages_by_source"].get(source_type, 0) + 1
                continue

            if error:
                skipped.append(error)
            fallback_page = build_partial_page_from_search_result(result, "fetch_error")
            if fallback_page:
                pages.append(fallback_page)
                stats["pages_by_source"]["search_snippet_partial"] += 1

        stats["retrieved_pages"] = len(pages)
        stats["skipped_pages"] = len(skipped)
        return RetrievalBatchResult(pages=pages, skipped=skipped, stats=stats)

    def _select_contents_candidates(
        self,
        contents_candidates: list[tuple[SearchResultItem, RetrievalDecision]],
    ) -> list[tuple[SearchResultItem, RetrievalDecision]]:
        selected: list[tuple[SearchResultItem, RetrievalDecision]] = []
        counts_by_query: dict[str, int] = {}
        for result, decision in sorted(contents_candidates, key=lambda item: item[1].score, reverse=True):
            if len(selected) >= self._config.exa_deep_fetch_max_per_run:
                break
            query_count = counts_by_query.get(result.search_query, 0)
            if query_count >= self._config.exa_deep_fetch_max_per_query:
                continue
            selected.append((result, decision))
            counts_by_query[result.search_query] = query_count + 1
        return selected

    def _fetch_contents_for_results(
        self,
        *,
        query: str,
        selected_results: list[SearchResultItem],
    ) -> tuple[list[PageContent], list[dict[str, Any]], dict[str, Any]]:
        urls = [str(result.url) for result in selected_results]
        stats = {
            "contents_followups_attempted": len(urls),
            "contents_followups_succeeded": 0,
            "contents_request_ids": [],
            "contents_cost_dollars": 0.0,
            "pages_by_source": {
                "exa_contents_api": 0,
                "search_snippet_partial": 0,
            },
        }
        if not urls:
            return [], [], stats

        try:
            response = self._search.get_contents(
                urls,
                ExaContentsOptions(
                    query=query,
                    text_max_characters=self._config.exa_contents_initial_text_chars,
                    highlights_max_characters=self._config.exa_highlights_max_chars,
                    summary_max_tokens=250,
                    max_age_hours=self._config.exa_default_max_age_hours,
                ),
            )
        except Exception as exc:
            LOGGER.warning("Exa contents follow-up failed for %s urls: %s", len(urls), exc)
            return [], [
                {
                    "url": url,
                    "search_query": query,
                    "reason": "exa_contents_error",
                    "error": str(exc),
                }
                for url in urls
            ], stats
        if response.request_id:
            stats["contents_request_ids"].append(response.request_id)
        stats["contents_cost_dollars"] += response.cost_dollars or 0.0

        result_by_url = {
            canonicalize_url(str(item.get("url") or "")): item
            for item in response.results
            if item.get("url")
        }
        pages: list[PageContent] = []
        skipped: list[dict[str, Any]] = []

        for result in selected_results:
            raw = result_by_url.get(canonicalize_url(str(result.url)))
            status = (response.statuses_by_url or {}).get(str(result.url)) or (response.statuses_by_url or {}).get(
                canonicalize_url(str(result.url))
            )
            if raw:
                page = build_page_from_exa_contents_result(
                    result,
                    raw,
                    status=status,
                    request_id=response.request_id,
                    cost_dollars=response.cost_dollars,
                    text_max_characters=self._config.exa_contents_initial_text_chars,
                )
                if page:
                    pages.append(page)
                    stats["contents_followups_succeeded"] += 1
                    stats["pages_by_source"]["exa_contents_api"] += 1
                    continue
            partial_page = build_partial_page_from_search_result(result, "exa_contents_incomplete")
            if partial_page:
                pages.append(partial_page)
                stats["pages_by_source"]["search_snippet_partial"] += 1
            else:
                skipped.append(
                    {
                        "url": str(result.url),
                        "search_query": result.search_query,
                        "reason": "exa_contents_incomplete",
                        "status": status,
                    }
                )
        return pages, skipped, stats

    @staticmethod
    def _initial_stats(results: list[SearchResultItem]) -> dict[str, Any]:
        search_requests: dict[str, float] = {}
        for result in results:
            if result.request_id and result.request_id not in search_requests:
                search_requests[result.request_id] = result.cost_dollars or 0.0
        return {
            "search_request_ids": list(search_requests.keys()),
            "search_cost_dollars": round(sum(search_requests.values()), 6),
            "search_results_seen": len(results),
            "retrieved_pages": 0,
            "skipped_pages": 0,
            "contents_followups_attempted": 0,
            "contents_followups_succeeded": 0,
            "contents_request_ids": [],
            "contents_cost_dollars": 0.0,
            "direct_fetch_attempted": 0,
            "decisions": {},
            "pages_by_source": {
                "exa_search_contents": 0,
                "exa_contents_api": 0,
                "search_snippet_partial": 0,
                "direct_html": 0,
                "direct_pdf": 0,
            },
        }


def compose_partial_text(result: SearchResultItem) -> str:
    parts = [result.title.strip()]
    parts.extend(item.strip() for item in result.highlights if item.strip())
    if result.summary.strip():
        parts.append(result.summary.strip())
    if result.snippet.strip() and result.snippet.strip() not in parts:
        parts.append(result.snippet.strip())
    return "\n\n".join(part for part in parts if part)


def build_page_from_exa_search_result(
    result: SearchResultItem,
    decision: RetrievalDecision,
    *,
    search_text_max_characters: int,
) -> PageContent | None:
    text = result.content_text.strip()
    if not text:
        return None
    metadata = _base_search_metadata(result)
    metadata.update(
        {
            "source_type": "exa_search_contents",
            "exa_text_max_characters": search_text_max_characters,
            "exa_text_truncated": bool(result.content_text)
            and len(result.content_text) >= int(search_text_max_characters * 0.95),
            "retrieval_decision": decision.rationale,
        }
    )
    return PageContent(
        title=result.title,
        url=str(result.url),
        search_query=result.search_query,
        source_domain=extract_domain(str(result.url)),
        fetched_at=datetime.now(timezone.utc),
        text=text,
        status_code=200,
        content_type="text/plain",
        raw_excerpt=result.snippet[:500],
        metadata=metadata,
    )


def build_page_from_exa_contents_result(
    result: SearchResultItem,
    contents_result: dict[str, Any],
    *,
    status: Any = None,
    request_id: str | None = None,
    cost_dollars: float | None = None,
    text_max_characters: int | None = None,
) -> PageContent | None:
    text = str(contents_result.get("text") or "").strip()
    if not text:
        return None
    highlights = result.highlights
    if isinstance(contents_result.get("highlights"), list):
        highlights = [str(item.get("text") if isinstance(item, dict) else item).strip() for item in contents_result["highlights"]]
        highlights = [item for item in highlights if item]
    summary = result.summary
    if contents_result.get("summary"):
        summary_value = contents_result.get("summary")
        if isinstance(summary_value, dict):
            summary = str(summary_value.get("text") or summary_value.get("summary") or "").strip()
        else:
            summary = str(summary_value).strip()
    metadata = _base_search_metadata(result)
    metadata.update(
        {
            "source_type": "exa_contents_api",
            "exa_request_id": request_id or result.request_id,
            "exa_cost_dollars": cost_dollars if cost_dollars is not None else result.cost_dollars,
            "exa_highlights": highlights,
            "exa_summary": summary,
            "exa_content_status": status,
            "exa_content_source": "contents_api",
            "exa_text_max_characters": text_max_characters,
            "exa_text_truncated": bool(text_max_characters) and len(text) >= int(text_max_characters * 0.95),
            "retrieval_decision": "follow_up_for_deeper_content",
        }
    )
    return PageContent(
        title=str(contents_result.get("title") or result.title),
        url=str(contents_result.get("url") or result.url),
        search_query=result.search_query,
        source_domain=extract_domain(str(contents_result.get("url") or result.url)),
        fetched_at=datetime.now(timezone.utc),
        text=text,
        status_code=200,
        content_type="text/plain",
        raw_excerpt=result.snippet[:500],
        metadata=metadata,
    )


def build_partial_page_from_search_result(result: SearchResultItem, reason: str) -> PageContent | None:
    partial_text = compose_partial_text(result)
    if not partial_text:
        return None
    metadata = _base_search_metadata(result)
    metadata.update(
        {
            "is_partial_evidence": True,
            "partial_evidence_reason": reason,
            "source_type": "search_snippet_partial",
            "retrieval_decision": reason,
        }
    )
    return PageContent(
        title=result.title,
        url=str(result.url),
        search_query=result.search_query,
        source_domain=extract_domain(str(result.url)),
        fetched_at=datetime.now(timezone.utc),
        text=partial_text,
        status_code=None,
        content_type="application/x-search-snippet",
        raw_excerpt=partial_text[:500],
        metadata=metadata,
    )


def _base_search_metadata(result: SearchResultItem) -> dict[str, Any]:
    return {
        "search_result_published_date": result.published_date,
        "search_result_score": result.score,
        "search_result_summary": result.summary,
        "search_result_snippet": result.snippet,
        "search_result_category": result.category,
        "exa_id": result.exa_id,
        "exa_request_id": result.request_id,
        "exa_cost_dollars": result.cost_dollars,
        "exa_highlights": result.highlights,
        "exa_highlight_scores": result.highlight_scores,
        "exa_content_source": result.content_source,
    }
