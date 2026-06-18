from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import pyarrow.parquet as pq

from bmscientist.coscientist_models import Hypothesis, ResearchGoalDocument


LOGGER = logging.getLogger(__name__)
DEFAULT_GRAPH_PATH = Path("data/graph")
GRAPH_PATH = DEFAULT_GRAPH_PATH



class GraphMarketEvidence:
    def __init__(self, graph_path: Path | None = None):
        self._graph_path = graph_path if graph_path is not None else GRAPH_PATH
        self._lock = Lock()
        self._loaded = False
        self._nodes: dict[str, dict[str, dict[str, Any]]] = {}
        self._edges: dict[str, list[dict[str, Any]]] = {}

    def build_evidence_rows(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if not self._edges:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for edge_type in (
            "Market_HAS_APPLICATION_Application",
            "Market_USES_Product",
            "Market_IN_GEOGRAPHY_Geography",
            "Product_USED_IN_Application",
        ):
            for edge in self._edges.get(edge_type, []):
                score = self._score_edge(edge_type, edge, document, hypothesis)
                if score <= 0:
                    continue
                row = self._row_from_edge(edge_type, edge, score)
                if row is not None:
                    scored.append((score, row))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1]["metadata"].get("revenue_value") or 0,
                item[1]["metadata"].get("forecast_revenue_value") or 0,
            ),
            reverse=True,
        )
        deduped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for _, row in scored:
            deduped.setdefault(row["id"], row)
            if len(deduped) >= limit:
                break
        return list(deduped.values())

    def build_evidence_rows_for_goal(
        self,
        document: ResearchGoalDocument,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if not self._edges:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for edge_type in (
            "Market_HAS_APPLICATION_Application",
            "Market_USES_Product",
            "Market_IN_GEOGRAPHY_Geography",
            "Product_USED_IN_Application",
        ):
            for edge in self._edges.get(edge_type, []):
                score = self._score_edge_for_goal(edge_type, edge, document)
                if score <= 0:
                    continue
                row = self._row_from_edge(edge_type, edge, score)
                if row is not None:
                    scored.append((score, row))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1]["metadata"].get("revenue_value") or 0,
                item[1]["metadata"].get("forecast_revenue_value") or 0,
            ),
            reverse=True,
        )
        deduped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for _, row in scored:
            deduped.setdefault(row["id"], row)
            if len(deduped) >= limit:
                break
        return list(deduped.values())

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._nodes = {
                    "Market": self._load_node_map("Market", "market_id"),
                    "Application": self._load_node_map("Application", "application_id"),
                    "Product": self._load_node_map("Product", "product_id"),
                    "Geography": self._load_node_map("Geography", "geo_id"),
                }
                self._edges = {
                    edge_type: self._load_rows(self._graph_path / "edges" / f"{edge_type}.parquet")
                    for edge_type in (
                        "Market_HAS_APPLICATION_Application",
                        "Market_USES_Product",
                        "Market_IN_GEOGRAPHY_Geography",
                        "Product_USED_IN_Application",
                    )
                }
            except Exception:
                LOGGER.exception("Failed to load graph market parquet evidence from %s", self._graph_path)
                self._nodes = {}
                self._edges = {}
            self._loaded = True

    def _load_node_map(self, label: str, key: str) -> dict[str, dict[str, Any]]:
        return {str(row[key]): row for row in self._load_rows(self._graph_path / "nodes" / f"{label}.parquet") if row.get(key)}

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    def _score_edge(
        self,
        edge_type: str,
        edge: dict[str, Any],
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
    ) -> float:
        if str(edge.get("validation_status")).lower() == "rejected":
            return 0.0
        market = self._nodes.get("Market", {}).get(str(edge.get("market_id")), {})
        if edge_type == "Product_USED_IN_Application":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            application = self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
            product_tokens = self._tokens(product.get("name")) | self._tokens(edge.get("product_id"))
            application_tokens = self._tokens(application.get("name")) | self._tokens(edge.get("application_id"))
            material_terms = self._tokens(
                " ".join(
                    item
                    for item in [
                        hypothesis.candidate_material,
                        hypothesis.incumbent_material,
                        hypothesis.next_best_competitive_alternative,
                        " ".join(document.material_scope),
                        " ".join(document.preferred_candidate_materials),
                        " ".join(document.target_incumbent_materials),
                    ]
                    if item
                )
            )
            application_terms = self._tokens(
                " ".join(
                    item
                    for item in [
                        hypothesis.application,
                        hypothesis.market_segment,
                        hypothesis.product_type,
                        " ".join(document.application_scope),
                        document.raw_goal,
                    ]
                    if item
                )
            )
            score = 0.0
            if product_tokens & material_terms:
                score += 4.0
            if application_tokens & application_terms:
                score += 4.0
            if edge.get("critical_to_quality_json"):
                score += 1.0
            if self._has_market_metrics(edge) or self._number(edge.get("volume_value")) is not None:
                score += 1.0
            return score
        target = self._target_node(edge_type, edge)
        market_tokens = self._tokens(market.get("name")) | self._tokens(market.get("primary_slug"))
        target_tokens = self._tokens(target.get("name")) | self._tokens(target.get("node_type"))
        geo = self._nodes.get("Geography", {}).get(str(edge.get("geo_id")), {})
        geo_tokens = self._tokens(geo.get("name"))

        material_terms = self._tokens(
            " ".join(
                item
                for item in [
                    hypothesis.candidate_material,
                    hypothesis.incumbent_material,
                    hypothesis.next_best_competitive_alternative,
                    " ".join(document.material_scope),
                    " ".join(document.preferred_candidate_materials),
                    " ".join(document.target_incumbent_materials),
                ]
                if item
            )
        )
        application_terms = self._tokens(
            " ".join(
                item
                for item in [
                    hypothesis.application,
                    hypothesis.market_segment,
                    hypothesis.product_type,
                    " ".join(document.application_scope),
                    document.raw_goal,
                ]
                if item
            )
        )
        region_terms = self._tokens(" ".join(document.regions))

        score = 0.0
        if market_tokens & material_terms:
            score += 3.0
        if market_tokens & application_terms:
            score += 2.0
        if target_tokens & application_terms:
            score += 4.0
        if target_tokens & material_terms:
            score += 3.0
        if geo_tokens and (geo_tokens & region_terms or "global" in geo_tokens):
            score += 1.0
        if self._has_market_metrics(edge):
            score += 1.0
        return score

    def _score_edge_for_goal(
        self,
        edge_type: str,
        edge: dict[str, Any],
        document: ResearchGoalDocument,
    ) -> float:
        if str(edge.get("validation_status")).lower() == "rejected":
            return 0.0
        market = self._nodes.get("Market", {}).get(str(edge.get("market_id")), {})
        if edge_type == "Product_USED_IN_Application":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            application = self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
            product_tokens = self._tokens(product.get("name")) | self._tokens(edge.get("product_id"))
            application_tokens = self._tokens(application.get("name")) | self._tokens(edge.get("application_id"))
            material_terms = self._tokens(
                " ".join(
                    item
                    for item in [
                        " ".join(document.material_scope),
                        " ".join(document.preferred_candidate_materials),
                        " ".join(document.target_incumbent_materials),
                    ]
                    if item
                )
            )
            application_terms = self._tokens(
                " ".join(
                    item
                    for item in [
                        " ".join(document.application_scope),
                        document.raw_goal,
                    ]
                    if item
                )
            )
            score = 0.0
            if product_tokens & material_terms:
                score += 4.0
            if application_tokens & application_terms:
                score += 4.0
            if self._has_market_metrics(edge) or self._number(edge.get("volume_value")) is not None:
                score += 1.0
            return score
        target = self._target_node(edge_type, edge)
        market_tokens = self._tokens(market.get("name")) | self._tokens(market.get("primary_slug"))
        target_tokens = self._tokens(target.get("name")) | self._tokens(target.get("node_type"))
        geo = self._nodes.get("Geography", {}).get(str(edge.get("geo_id")), {})
        geo_tokens = self._tokens(geo.get("name"))

        material_terms = self._tokens(
            " ".join(
                item
                for item in [
                    " ".join(document.material_scope),
                    " ".join(document.preferred_candidate_materials),
                    " ".join(document.target_incumbent_materials),
                ]
                if item
            )
        )
        application_terms = self._tokens(
            " ".join(
                item
                for item in [
                    " ".join(document.application_scope),
                    document.raw_goal,
                ]
                if item
            )
        )
        region_terms = self._tokens(" ".join(document.regions))

        score = 0.0
        if market_tokens & material_terms:
            score += 3.0
        if market_tokens & application_terms:
            score += 2.0
        if target_tokens & application_terms:
            score += 4.0
        if target_tokens & material_terms:
            score += 3.0
        if geo_tokens and (geo_tokens & region_terms or "global" in geo_tokens):
            score += 1.0
        if self._has_market_metrics(edge):
            score += 1.0
        return score

    def _target_node(self, edge_type: str, edge: dict[str, Any]) -> dict[str, Any]:
        if edge_type == "Market_HAS_APPLICATION_Application":
            return self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
        if edge_type == "Market_USES_Product":
            return self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
        if edge_type == "Market_IN_GEOGRAPHY_Geography":
            return self._nodes.get("Geography", {}).get(str(edge.get("geo_id")), {})
        if edge_type == "Product_USED_IN_Application":
            return self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
        return {}

    def _row_from_edge(self, edge_type: str, edge: dict[str, Any], score: float) -> dict[str, Any] | None:
        market = self._nodes.get("Market", {}).get(str(edge.get("market_id")), {})
        target = self._target_node(edge_type, edge)
        if not market and edge_type != "Product_USED_IN_Application":
            return None
        relationship = {
            "Market_HAS_APPLICATION_Application": "has application",
            "Market_USES_Product": "uses product",
            "Market_IN_GEOGRAPHY_Geography": "is measured in geography",
            "Product_USED_IN_Application": "is used in application",
        }.get(edge_type, edge_type)
        product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
        target_name = target.get("name") or edge.get("application_id") or edge.get("product_id") or edge.get("geo_id")
        metrics_text = self._metrics_text(edge)
        notes = self._json_notes(
            edge,
            ["critical_to_quality_json", "highlights_json", "industry_trends_json", "data_book_summary_json"],
            limit=4,
        )
        source_url = (
            edge.get("source_url")
            or edge.get("page_url")
            or edge.get("target_url")
            or market.get("canonical_url")
            or str(self._graph_path.resolve())
        )
        subject_name = product.get("name") if edge_type == "Product_USED_IN_Application" else market.get("name")
        chunk_text = " ".join(
            item
            for item in [
                f"Graph market data from {market.get('source_vendor') or 'offline market graph'}: "
                f"{subject_name} {relationship} {target_name}.",
                metrics_text,
                " ".join(notes),
                f"Source URL: {source_url}.",
            ]
            if item
        )
        row_id = f"graph:{edge_type}:{edge.get('edge_id')}"
        return {
            "id": row_id,
            "source_url": source_url,
            "source_title": "Offline graph market data",
            "application": target_name if edge_type in ("Market_HAS_APPLICATION_Application", "Product_USED_IN_Application") else None,
            "incumbent_material": None,
            "candidate_materials": (
                [target_name]
                if edge_type == "Market_USES_Product" and target_name
                else [product.get("name")]
                if edge_type == "Product_USED_IN_Application" and product.get("name")
                else []
            ),
            "relevance_score": min(0.98, 0.55 + (score * 0.04)),
            "retrieved_at": edge.get("retrieved_at") or edge.get("updated_at") or edge.get("created_at"),
            "chunk_text": chunk_text[:1800],
            "metadata": {
                "source_type": "offline-graph-market-data",
                "edge_type": edge_type,
                "market_id": edge.get("market_id"),
                "market_name": market.get("name"),
                "product_id": edge.get("product_id"),
                "application_id": edge.get("application_id"),
                "target_name": target_name,
                "geo_id": edge.get("geo_id"),
                "revenue_value": self._number(edge.get("revenue_value")),
                "revenue_year": self._number(edge.get("revenue_year")),
                "forecast_revenue_value": self._number(edge.get("forecast_revenue_value")),
                "forecast_revenue_year": self._number(edge.get("forecast_revenue_year")),
                "cagr_value": self._number(edge.get("cagr_value")),
                "volume_value": self._number(edge.get("volume_value")),
                "volume_unit": edge.get("volume_unit"),
                "volume_year": self._number(edge.get("volume_year")),
                "price_value": self._number(edge.get("price_value")),
                "price_currency": edge.get("price_currency"),
                "price_unit": edge.get("price_unit"),
                "price_year": self._number(edge.get("price_year")),
                "evidence_hash": edge.get("evidence_hash"),
                "source_chunk_id": edge.get("source_chunk_id"),
                "unit": edge.get("unit"),
                "currency": edge.get("currency"),
                "unit_scale": edge.get("unit_scale"),
            },
        }

    @staticmethod
    def _metrics_text(edge: dict[str, Any]) -> str:
        pieces: list[str] = []
        revenue = GraphMarketEvidence._number(edge.get("revenue_value"))
        forecast = GraphMarketEvidence._number(edge.get("forecast_revenue_value"))
        cagr = GraphMarketEvidence._number(edge.get("cagr_value"))
        unit = edge.get("unit") or "reported units"
        if revenue is not None:
            pieces.append(f"Revenue was {revenue:g} {unit} in {GraphMarketEvidence._year(edge.get('revenue_year'))}.")
        if forecast is not None:
            pieces.append(
                f"Forecast revenue is {forecast:g} {unit} by {GraphMarketEvidence._year(edge.get('forecast_revenue_year'))}."
            )
        if cagr is not None:
            pieces.append(
                f"CAGR is {cagr:g}% from {GraphMarketEvidence._year(edge.get('cagr_start_year'))} "
                f"to {GraphMarketEvidence._year(edge.get('cagr_end_year'))}."
            )
        volume = GraphMarketEvidence._number(edge.get("volume_value"))
        if volume is not None:
            pieces.append(f"Volume was {volume:g} {edge.get('volume_unit') or 'reported units'} in {GraphMarketEvidence._year(edge.get('volume_year'))}.")
        price = GraphMarketEvidence._number(edge.get("price_value"))
        if price is not None:
            unit_text = edge.get("price_unit") or "reported unit"
            currency = edge.get("price_currency") or ""
            pieces.append(f"Price was {price:g} {currency}/{unit_text} in {GraphMarketEvidence._year(edge.get('price_year'))}.")
        return " ".join(pieces)

    @staticmethod
    def _json_notes(edge: dict[str, Any], fields: list[str], limit: int) -> list[str]:
        notes: list[str] = []
        for field in fields:
            raw_value = edge.get(field)
            if not raw_value:
                continue
            try:
                parsed = json.loads(raw_value)
            except (TypeError, ValueError):
                parsed = raw_value
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        text = str(item.get("text") or item.get("source_text") or "").strip()
                    else:
                        text = str(item).strip()
                    if text:
                        notes.append(text)
            elif isinstance(parsed, dict):
                for key, value in parsed.items():
                    if key and value:
                        notes.append(f"{key}: {value}")
            elif parsed:
                notes.append(str(parsed))
            if len(notes) >= limit:
                return notes[:limit]
        return notes[:limit]

    @staticmethod
    def _has_market_metrics(edge: dict[str, Any]) -> bool:
        return any(
            GraphMarketEvidence._number(edge.get(key)) is not None
            for key in ("revenue_value", "forecast_revenue_value", "cagr_value", "volume_value", "price_value")
        )

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            return None if math.isnan(number) else number
        try:
            number = float(str(value))
        except ValueError:
            return None
        return None if math.isnan(number) else number

    @staticmethod
    def _year(value: Any) -> str:
        number = GraphMarketEvidence._number(value)
        if number is None:
            return "n/a"
        return str(int(number))

    @staticmethod
    def _tokens(value: Any) -> set[str]:
        normalized = unicodedata.normalize("NFKD", str(value or ""))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        tokens = set(re.findall(r"[a-z0-9]+", ascii_text.lower()))
        stopwords = {
            "and",
            "for",
            "from",
            "general",
            "market",
            "material",
            "materials",
            "of",
            "or",
            "plastic",
            "plastics",
            "polymer",
            "polymers",
            "purpose",
            "recycled",
            "resin",
            "resins",
            "standard",
            "the",
            "to",
            "use",
            "uses",
            "with",
        }
        return {token for token in tokens if token not in stopwords and len(token) > 1}
