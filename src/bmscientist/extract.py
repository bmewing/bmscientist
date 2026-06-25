from __future__ import annotations

from io import BytesIO
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader

from bmscientist.config import AppConfig
from bmscientist.models import PageContent, SearchResultItem


LOGGER = logging.getLogger(__name__)
SUPPORTED_TEXT_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


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


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        if normalized:
            parts.append(normalized)

    metadata = {
        "page_count": len(reader.pages),
        "pdf_metadata": {str(key): str(value) for key, value in (reader.metadata or {}).items()},
        "extraction_method": "pypdf",
    }
    return "\n\n".join(parts).strip(), metadata


class PageFetcher:
    def __init__(self, config: AppConfig):
        self._config = config
        self._timeout = config.request_timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})

    def fetch(self, result: SearchResultItem) -> PageContent:
        response = self._session.get(str(result.url), timeout=self._timeout)
        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        title = result.title
        metadata = {
            "search_result_published_date": result.published_date,
            "search_result_score": result.score,
        }

        if self._is_pdf_response(content_type, response.content):
            text, pdf_metadata = extract_pdf_text(response.content)
            metadata.update(pdf_metadata)
            metadata["source_type"] = "direct_pdf"
            if not text:
                raise ValueError("No readable text extracted from PDF")
        elif self._is_supported_text_content_type(content_type):
            text = extract_readable_text(response.text)
            if not text:
                raise ValueError("No readable text extracted from page")

            if "<title" in response.text.lower():
                soup = BeautifulSoup(response.text, "html.parser")
                title = (soup.title.string or title).strip() if soup.title else title
            metadata["source_type"] = "direct_html"
        else:
            raise ValueError(f"Unsupported content type for extraction: {content_type or 'unknown'}")

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
            metadata=metadata,
        )

    @staticmethod
    def _is_pdf_response(content_type: str, content: bytes) -> bool:
        return "application/pdf" in content_type or content.startswith(b"%PDF-")

    @staticmethod
    def _is_supported_text_content_type(content_type: str) -> bool:
        if not content_type:
            return True
        return any(item in content_type for item in SUPPORTED_TEXT_CONTENT_TYPES)

    def safe_fetch(self, result: SearchResultItem) -> tuple[PageContent | None, dict[str, Any] | None]:
        try:
            page = self.fetch(result)
            return page, None
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            reason = "blocked_domain" if status_code in (403, 401) else "fetch_error"
            LOGGER.warning("Failed to fetch %s directly: %s", result.url, exc)
            return None, {
                "url": str(result.url),
                "search_query": result.search_query,
                "error": str(exc),
                "status_code": status_code,
                "reason": reason,
                "note": "Direct fetch failed.",
            }
        except Exception as exc:
            return None, {
                "url": str(result.url),
                "search_query": result.search_query,
                "error": str(exc),
                "status_code": None,
                "reason": "fetch_error",
                "note": "Direct fetch failed.",
            }
