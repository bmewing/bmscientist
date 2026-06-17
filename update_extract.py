import sys

path = "src/app_discovery_agent/extract.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add exa dependency to imports
import_block = """from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader"""

new_import_block = """from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader
from exa_py import Exa"""

content = content.replace(import_block, new_import_block)

# Add fallback method to PageFetcher
old_safe_fetch = """    def safe_fetch(self, result: SearchResultItem) -> tuple[PageContent | None, dict[str, Any] | None]:
        try:
            page = self.fetch(result)
            return page, None
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            reason = "blocked_domain" if status_code == 403 else "fetch_error"
            LOGGER.warning("Failed to fetch %s: %s", result.url, exc)
            return None, {
                "url": str(result.url),
                "search_query": result.search_query,
                "error": str(exc),
                "status_code": status_code,
                "reason": reason,
            }
        except Exception as exc:
            LOGGER.warning("Failed to fetch %s: %s", result.url, exc)
            return None, {
                "url": str(result.url),
                "search_query": result.search_query,
                "error": str(exc),
                "reason": "fetch_error",
            }"""

new_safe_fetch = """    def safe_fetch(self, result: SearchResultItem) -> tuple[PageContent | None, dict[str, Any] | None]:
        try:
            page = self.fetch(result)
            return page, None
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            reason = "blocked_domain" if status_code in (403, 401) else "fetch_error"
            LOGGER.warning("Failed to fetch %s directly: %s. Attempting Exa fallback...", result.url, exc)
            return self._exa_fallback_fetch(result, reason, status_code, str(exc))
        except Exception as exc:
            LOGGER.warning("Failed to fetch %s directly: %s. Attempting Exa fallback...", result.url, exc)
            return self._exa_fallback_fetch(result, "fetch_error", None, str(exc))

    def _exa_fallback_fetch(self, result: SearchResultItem, reason: str, status_code: int | None, error_msg: str) -> tuple[PageContent | None, dict[str, Any] | None]:
        if not self._config.exa_api_key:
            return None, {
                "url": str(result.url),
                "search_query": result.search_query,
                "error": error_msg,
                "status_code": status_code,
                "reason": reason,
                "note": "Exa fallback skipped: No API key configured."
            }
            
        try:
            exa = Exa(api_key=self._config.exa_api_key)
            contents_response = exa.get_contents([str(result.url)], text=True)
            
            if not contents_response.results or not contents_response.results[0].text:
                 return None, {
                    "url": str(result.url),
                    "search_query": result.search_query,
                    "error": error_msg,
                    "status_code": status_code,
                    "reason": reason,
                    "note": "Exa fallback failed: No content returned."
                }
            
            exa_result = contents_response.results[0]
            
            metadata = {
                "search_result_published_date": result.published_date,
                "search_result_score": result.score,
                "source_type": "exa_contents_api",
                "original_fetch_error": error_msg,
                "original_fetch_status": status_code
            }
            
            page = PageContent(
                title=exa_result.title or result.title,
                url=str(result.url),
                search_query=result.search_query,
                source_domain=extract_domain(str(result.url)),
                fetched_at=datetime.now(timezone.utc),
                text=exa_result.text,
                status_code=200, # Fake 200 since Exa succeeded
                content_type="text/plain",
                raw_excerpt=result.snippet[:500],
                metadata=metadata,
            )
            LOGGER.info("Successfully recovered content for %s via Exa API", result.url)
            return page, None
            
        except Exception as exa_exc:
            LOGGER.warning("Exa fallback also failed for %s: %s", result.url, exa_exc)
            return None, {
                "url": str(result.url),
                "search_query": result.search_query,
                "error": error_msg,
                "status_code": status_code,
                "reason": reason,
                "exa_fallback_error": str(exa_exc)
            }"""

content = content.replace(old_safe_fetch, new_safe_fetch)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
