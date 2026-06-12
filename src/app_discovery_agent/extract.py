from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from app_discovery_agent.config import AppConfig
from app_discovery_agent.models import PageContent, SearchResultItem


LOGGER = logging.getLogger(__name__)


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def extract_readable_text(html: str) -> str:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_formatting=False,
        include_tables=True,
        favor_precision=True,
    )
    if text:
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    fallback = " ".join(soup.get_text(separator=" ").split())
    return fallback.strip()


class PageFetcher:
    def __init__(self, config: AppConfig):
        self._timeout = config.request_timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})

    def fetch(self, result: SearchResultItem) -> PageContent:
        response = self._session.get(str(result.url), timeout=self._timeout)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        text = extract_readable_text(response.text)
        title = result.title

        if "<title" in response.text.lower():
            soup = BeautifulSoup(response.text, "html.parser")
            title = (soup.title.string or title).strip() if soup.title else title

        return PageContent(
            title=title,
            url=str(response.url),
            search_query=result.search_query,
            source_domain=extract_domain(str(response.url)),
            fetched_at=datetime.now(timezone.utc),
            text=text,
            status_code=response.status_code,
            content_type=content_type,
            raw_excerpt=result.snippet[:500],
            metadata={
                "search_result_published_date": result.published_date,
                "search_result_score": result.score,
            },
        )

    def safe_fetch(self, result: SearchResultItem) -> tuple[PageContent | None, dict[str, Any] | None]:
        try:
            page = self.fetch(result)
            return page, None
        except Exception as exc:
            LOGGER.warning("Failed to fetch %s: %s", result.url, exc)
            return None, {"url": str(result.url), "search_query": result.search_query, "error": str(exc)}

