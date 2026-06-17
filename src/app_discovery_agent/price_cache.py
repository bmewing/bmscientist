from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app_discovery_agent.config import AppConfig
from app_discovery_agent.coscientist_models import PriceMetric


LOGGER = logging.getLogger(__name__)
PRICE_CACHE_PATH = Path("data/pricing/plasticportal_prices.json")
WEEKLY_URL = "https://www.plasticportal.eu/price-reports"
AVERAGE_URL = "https://www.plasticportal.eu/polymer-prices"
FX_URL = "https://currencyapi.net/api/v2/rates?base=USD&output=json&key=2b75799b749c380bee772512d4dc883a9f6c"
PRICE_EVIDENCE_SOURCE_TITLE = "PlasticPortal structured price cache"
PRICE_CACHE_QUERY = "structured polymer price reference data from PlasticPortal"
FALLBACK_EUR_USD_RATE = 1.08


class PriceCacheEntry(BaseModel):
    source: Literal["weekly_commodity", "average_resin"]
    page_url: str
    label: str
    polymer_name: str
    normalized_polymer: str
    price_eur_per_kg: float = Field(ge=0.0)
    price_usd_per_kg: float | None = Field(default=None, ge=0.0)
    raw_price_text: str
    difference_text: str = ""
    yoy_text: str = ""
    fetched_at: datetime


class PriceCacheDocument(BaseModel):
    fetched_at: datetime
    eur_usd_rate: float | None = Field(default=None, ge=0.0)
    eur_usd_rate_fetched_at: datetime | None = None
    entries: list[PriceCacheEntry] = Field(default_factory=list)


class StructuredPriceCache:
    def __init__(self, config: AppConfig, cache_path: Path = PRICE_CACHE_PATH):
        self._config = config
        self._cache_path = cache_path
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})

    def ensure_fresh(self, max_age_days: int = 7) -> PriceCacheDocument:
        cached = self.load()
        if cached and datetime.now(timezone.utc) - cached.fetched_at < timedelta(days=max_age_days):
            return cached
        try:
            return self.refresh(existing=cached)
        except Exception:
            if cached is not None:
                LOGGER.exception("Structured price cache refresh failed; using stale cached prices")
                return cached
            raise

    def load(self) -> PriceCacheDocument | None:
        if not self._cache_path.exists():
            return None
        return PriceCacheDocument.model_validate_json(self._cache_path.read_text(encoding="utf-8"))

    def refresh(self, existing: PriceCacheDocument | None = None) -> PriceCacheDocument:
        fetched_at = datetime.now(timezone.utc)
        weekly_html = self._fetch_text(WEEKLY_URL)
        average_html = self._fetch_text(AVERAGE_URL)
        fx_rate, fx_fetched_at = self._fetch_fx_rate(existing)

        entries = self._parse_weekly_prices(weekly_html, fetched_at, fx_rate)
        entries.extend(self._parse_average_prices(average_html, fetched_at, fx_rate))

        document = PriceCacheDocument(
            fetched_at=fetched_at,
            eur_usd_rate=fx_rate,
            eur_usd_rate_fetched_at=fx_fetched_at,
            entries=entries,
        )
        self._cache_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        LOGGER.info("Updated structured price cache with %s entries", len(entries))
        return document

    def build_price_evidence_rows(
        self,
        incumbent_material: str | None,
        nbca_material: str | None,
        candidate_material: str | None,
        document: PriceCacheDocument | None = None,
    ) -> list[dict[str, Any]]:
        document = document or self.ensure_fresh()
        rows: list[dict[str, Any]] = []
        for role, material_name in (
            ("incumbent", incumbent_material),
            ("nbca", nbca_material),
            ("candidate", candidate_material),
        ):
            if not material_name:
                continue
            for entry in self.entries_for_material(material_name, document=document)[:3]:
                usd_value, usd_is_inferred = self._usd_price_for_entry(entry, document)
                usd_text = (
                    f"{usd_value:.3f} USD/kg"
                    if usd_value is not None and not usd_is_inferred
                    else f"about {usd_value:.3f} USD/kg using fallback FX"
                    if usd_value is not None
                    else "USD conversion unavailable"
                )
                chunk_text = (
                    f"Structured price reference from PlasticPortal ({entry.source}, {entry.label}): "
                    f"{entry.polymer_name} = {entry.price_eur_per_kg:.3f} EUR/kg ({usd_text}). "
                    f"Original text: {entry.raw_price_text}. "
                    f"Difference text: {entry.difference_text or 'n/a'}. "
                    f"Year-over-year text: {entry.yoy_text or 'n/a'}. "
                    f"Remote source: {entry.page_url}."
                )
                rows.append(
                    {
                        "id": f"price-cache:{role}:{entry.source}:{entry.normalized_polymer}:{entry.label}",
                        "source_url": str(self._cache_path.resolve()),
                        "source_title": PRICE_EVIDENCE_SOURCE_TITLE,
                        "application": None,
                        "incumbent_material": material_name if role == "incumbent" else None,
                        "candidate_materials": [candidate_material] if candidate_material else [],
                        "relevance_score": 0.92,
                        "retrieved_at": document.fetched_at.isoformat(),
                        "chunk_text": chunk_text,
                        "metadata": {
                            "source_type": "structured-price-cache",
                            "price_role": role,
                            "price_source": entry.source,
                            "page_url": entry.page_url,
                            "label": entry.label,
                            "polymer_name": entry.polymer_name,
                            "price_eur_per_kg": entry.price_eur_per_kg,
                            "price_usd_per_kg": usd_value,
                            "price_usd_is_inferred": usd_is_inferred,
                            "cache_path": str(self._cache_path.resolve()),
                        },
                    }
                )
        return rows

    def metric_for_material(
        self,
        material_name: str | None,
        document: PriceCacheDocument | None = None,
    ) -> PriceMetric | None:
        if not material_name:
            return None
        document = document or self.ensure_fresh()
        entries = self.entries_for_material(material_name, document=document)
        if not entries:
            return None
        entry = entries[0]
        value, is_inferred = self._usd_price_for_entry(entry, document)
        rationale = (
            f"Structured PlasticPortal price reference for {entry.polymer_name} "
            f"({entry.source}, {entry.label}) from {entry.raw_price_text}."
        )
        if value is not None and is_inferred:
            rationale += f" USD/kg converted from EUR/kg with fallback EUR/USD rate {FALLBACK_EUR_USD_RATE:.2f}."
        return PriceMetric(
            value=value,
            rationale=rationale,
            confidence=0.8 if value is not None and not is_inferred else 0.62 if value is not None else 0.45,
            citation_chunk_ids=[f"price-cache:{entry.source}:{entry.normalized_polymer}:{entry.label}"],
            citation_urls=[str(self._cache_path.resolve())],
            is_inferred=is_inferred,
        )

    def entries_for_material(
        self,
        material_name: str,
        document: PriceCacheDocument | None = None,
    ) -> list[PriceCacheEntry]:
        document = document or self.ensure_fresh()
        target_aliases = set(self._aliases_for_material(material_name))
        ranked: list[tuple[int, PriceCacheEntry]] = []
        for entry in document.entries:
            score = self._match_score(target_aliases, entry.normalized_polymer)
            if score > 0:
                ranked.append((score, entry))
        ranked.sort(key=lambda item: (item[0], item[1].fetched_at.isoformat(), item[1].source == "average_resin"), reverse=True)
        return [entry for _, entry in ranked]

    def _fetch_text(self, url: str) -> str:
        response = self._session.get(url, timeout=self._config.request_timeout_seconds)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _usd_price_for_entry(
        entry: PriceCacheEntry,
        document: PriceCacheDocument | None,
    ) -> tuple[float | None, bool]:
        if entry.price_usd_per_kg is not None:
            return entry.price_usd_per_kg, False
        if document and document.eur_usd_rate is not None:
            return entry.price_eur_per_kg * document.eur_usd_rate, False
        return entry.price_eur_per_kg * FALLBACK_EUR_USD_RATE, True

    def _fetch_fx_rate(self, existing: PriceCacheDocument | None) -> tuple[float | None, datetime | None]:
        try:
            response = self._session.get(FX_URL, timeout=self._config.request_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            rates = payload.get("rates", {})
            eur_rate = rates.get("EUR")
            if eur_rate is None:
                eur_rate = rates.get("eur")
            if eur_rate is None:
                raise ValueError("EUR rate missing from exchange response")
            eur_rate = float(eur_rate)
            if eur_rate == 0:
                raise ValueError("EUR rate cannot be zero")
            return 1.0 / eur_rate, datetime.now(timezone.utc)
        except Exception as exc:
            LOGGER.warning("Failed to refresh USD-per-EUR rate from %s: %s", FX_URL, exc)
            if existing and existing.eur_usd_rate is not None:
                return existing.eur_usd_rate, existing.eur_usd_rate_fetched_at
            return None, None

    def _parse_weekly_prices(
        self,
        html: str,
        fetched_at: datetime,
        eur_usd_rate: float | None,
    ) -> list[PriceCacheEntry]:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.find("div", class_="prices-history")
        if container is None:
            return []
        label_node = container.find("h3")
        label = " ".join(label_node.get_text(" ", strip=True).split()) if label_node else "Weekly commodity prices"
        table = container.find("table")
        if table is None:
            return []

        entries: list[PriceCacheEntry] = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            polymer_name = " ".join(cells[0].get_text(" ", strip=True).split())
            raw_price_text = " ".join(cells[1].get_text(" ", strip=True).split())
            if not polymer_name or not raw_price_text:
                continue
            price_eur_per_kg = self._parse_eur_price(raw_price_text)
            if price_eur_per_kg is None:
                continue
            difference_text = " ".join(cells[2].get_text(" ", strip=True).split()) if len(cells) > 2 else ""
            entries.append(
                PriceCacheEntry(
                    source="weekly_commodity",
                    page_url=WEEKLY_URL,
                    label=label,
                    polymer_name=polymer_name,
                    normalized_polymer=self._normalize_text(polymer_name),
                    price_eur_per_kg=price_eur_per_kg,
                    price_usd_per_kg=(price_eur_per_kg * eur_usd_rate) if eur_usd_rate is not None else None,
                    raw_price_text=raw_price_text,
                    difference_text=difference_text,
                    fetched_at=fetched_at,
                )
            )
        return entries

    def _parse_average_prices(
        self,
        html: str,
        fetched_at: datetime,
        eur_usd_rate: float | None,
    ) -> list[PriceCacheEntry]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="table")
        if table is None:
            return []
        entries: list[PriceCacheEntry] = []
        label = "Average resin prices"
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            polymer_name = " ".join(cells[0].get_text(" ", strip=True).split())
            raw_price_text = " ".join(cells[1].get_text(" ", strip=True).split())
            if not polymer_name or not raw_price_text:
                continue
            try:
                price_eur_per_kg = float(raw_price_text.replace(",", "."))
            except ValueError:
                continue
            difference_text = " ".join(cells[2].get_text(" ", strip=True).split()) if len(cells) > 2 else ""
            yoy_text = " ".join(cells[3].get_text(" ", strip=True).split()) if len(cells) > 3 else ""
            entries.append(
                PriceCacheEntry(
                    source="average_resin",
                    page_url=AVERAGE_URL,
                    label=label,
                    polymer_name=polymer_name,
                    normalized_polymer=self._normalize_text(polymer_name),
                    price_eur_per_kg=price_eur_per_kg,
                    price_usd_per_kg=(price_eur_per_kg * eur_usd_rate) if eur_usd_rate is not None else None,
                    raw_price_text=f"{raw_price_text} EUR/kg",
                    difference_text=difference_text,
                    yoy_text=yoy_text,
                    fetched_at=fetched_at,
                )
            )
        return entries

    @staticmethod
    def _parse_eur_price(raw_price_text: str) -> float | None:
        normalized = raw_price_text.replace("\xa0", " ").strip()
        amount_match = re.search(r"([\d\s]+(?:[.,]\d+)?)\s*(?:€|EUR)\s*/\s*([tk]g?)", normalized, flags=re.IGNORECASE)
        if not amount_match:
            return None
        amount = float(amount_match.group(1).replace(" ", "").replace(",", "."))
        unit = amount_match.group(2).lower()
        if unit == "t":
            return amount / 1000.0
        return amount

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower())
        return " ".join(cleaned.split())

    def _aliases_for_material(self, material_name: str) -> list[str]:
        normalized = self._normalize_text(material_name)
        aliases = {normalized}
        canonical_aliases = {
            "polyvinyl chloride": ["pvc"],
            "pvc": ["polyvinyl chloride"],
            "polyethylene terephthalate": ["pet"],
            "pet": ["polyethylene terephthalate"],
            "petg": ["pet", "pet g", "pet-g", "polyethylene terephthalate glycol"],
            "polyethylene terephthalate glycol": ["petg", "pet g", "pet-g", "pet"],
            "high impact polystyrene": ["hips", "ps impact", "polystyrene"],
            "hips": ["high impact polystyrene", "ps impact", "polystyrene"],
            "general purpose polystyrene": ["gpps", "ps crystal", "polystyrene"],
            "gpps": ["general purpose polystyrene", "ps crystal", "polystyrene"],
            "polystyrene": ["ps", "ps impact", "ps crystal", "hips", "gpps"],
            "ps": ["polystyrene", "ps impact", "ps crystal"],
            "styrene acrylonitrile": ["san"],
            "san": ["styrene acrylonitrile"],
            "acrylonitrile butadiene styrene": ["abs"],
            "abs": ["acrylonitrile butadiene styrene"],
            "polycarbonate": ["pc"],
            "pc": ["polycarbonate"],
            "polyamide": ["pa"],
            "pa": ["polyamide"],
            "pmma": ["acrylic"],
            "polyoxymethylene": ["pom"],
            "pom": ["polyoxymethylene"],
            "polybutylene terephthalate": ["pbt"],
            "pbt": ["polybutylene terephthalate"],
            "polypropylene": ["pp", "pp homo", "pp copo"],
            "pp": ["polypropylene", "pp homo", "pp copo"],
            "high density polyethylene": ["hdpe"],
            "hdpe": ["high density polyethylene"],
            "low density polyethylene": ["ldpe"],
            "ldpe": ["low density polyethylene"],
            "linear low density polyethylene": ["lldpe"],
            "lldpe": ["linear low density polyethylene"],
        }
        for key, values in canonical_aliases.items():
            if self._contains_alias(normalized, key):
                aliases.update(values)
        return sorted({self._normalize_text(alias) for alias in aliases if alias})

    @staticmethod
    def _contains_alias(normalized_text: str, normalized_alias: str) -> bool:
        if normalized_text == normalized_alias:
            return True
        return re.search(rf"(^|\s){re.escape(normalized_alias)}(\s|$)", normalized_text) is not None

    @staticmethod
    def _match_score(target_aliases: set[str], normalized_polymer: str) -> int:
        for alias in target_aliases:
            if alias == normalized_polymer:
                return 100
        for alias in target_aliases:
            if alias and alias in normalized_polymer:
                return 80
        polymer_tokens = set(normalized_polymer.split())
        for alias in target_aliases:
            alias_tokens = set(alias.split())
            if alias_tokens and alias_tokens <= polymer_tokens:
                return 60
        return 0
