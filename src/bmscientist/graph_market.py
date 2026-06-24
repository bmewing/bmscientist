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

GRAPH_EVIDENCE_EDGE_TYPES = (
    "Market_HAS_APPLICATION_Application",
    "Market_USES_Product",
    "Market_IN_GEOGRAPHY_Geography",
    "Product_USED_IN_Application",
    "Company_PRODUCES_Product",
    "Market_HAS_COMPANY_Company",
    "Product_HAS_MaterialGrade",
    "MaterialGrade_BELONGS_TO_MaterialFamily",
    "Company_PRODUCES_MaterialGrade",
    "MaterialGrade_HAS_Endpoint",
    "Application_REQUIRES_CriticalToQuality",
    "CriticalToQuality_INDICATED_BY_Endpoint",
    "Product_HAS_ChemistryClass",
    "Product_HAS_Function",
    "Product_TARGETS_BinderSystem",
    "Product_HAS_Endpoint",
)



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
        for edge_type in GRAPH_EVIDENCE_EDGE_TYPES:
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
        for edge_type in GRAPH_EVIDENCE_EDGE_TYPES:
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
                    "Company": self._load_node_map("Company", "company_id"),
                    "MaterialFamily": self._load_node_map("MaterialFamily", "material_family_id"),
                    "MaterialGrade": self._load_node_map("MaterialGrade", "material_grade_id"),
                    "ChemistryClass": self._load_node_map("ChemistryClass", "chemistry_class_id"),
                    "Function": self._load_node_map("Function", "function_id"),
                    "BinderSystem": self._load_node_map("BinderSystem", "binder_system_id"),
                    "Endpoint": self._load_node_map("Endpoint", "endpoint_id"),
                    "CriticalToQuality": self._load_node_map("CriticalToQuality", "ctq_id"),
                }
                self._edges = {
                    edge_type: self._load_rows(self._graph_path / "edges" / f"{edge_type}.parquet")
                    for edge_type in GRAPH_EVIDENCE_EDGE_TYPES
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
        combined_terms = self._document_terms(document) | self._hypothesis_terms(document, hypothesis)
        criterion_terms = self._criterion_terms(document)
        market = self._nodes.get("Market", {}).get(str(edge.get("market_id")), {})
        if edge_type == "Product_USED_IN_Application":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            application = self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
            product_tokens = self._node_tokens(product) | self._tokens(edge.get("product_id"))
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
        if edge_type == "Company_PRODUCES_Product":
            company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            score = 0.0
            if self._tokens(company.get("name")) & combined_terms:
                score += 4.0
            if self._node_tokens(product) & combined_terms:
                score += 3.0
            return score
        if edge_type == "Product_HAS_MaterialGrade":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            score = 0.0
            if self._node_tokens(product) & combined_terms:
                score += 3.0
            if self._node_tokens(material_grade) & combined_terms:
                score += 4.0
            return score
        if edge_type == "MaterialGrade_BELONGS_TO_MaterialFamily":
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            material_family = self._nodes.get("MaterialFamily", {}).get(str(edge.get("material_family_id")), {})
            score = 0.0
            if self._node_tokens(material_grade) & combined_terms:
                score += 3.0
            if self._node_tokens(material_family) & combined_terms:
                score += 4.0
            return score
        if edge_type == "Company_PRODUCES_MaterialGrade":
            company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            score = 0.0
            if self._tokens(company.get("name")) & combined_terms:
                score += 3.0
            if self._node_tokens(material_grade) & combined_terms:
                score += 4.0
            return score
        if edge_type == "MaterialGrade_HAS_Endpoint":
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            endpoint = self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
            score = 0.0
            if self._node_tokens(material_grade) & combined_terms:
                score += 3.0
            if self._node_tokens(endpoint) & combined_terms:
                score += 3.0
            if self._node_tokens(endpoint) & criterion_terms:
                score += 3.0
            if self._number(edge.get("value_numeric")) is not None or edge.get("value_text"):
                score += 1.0
            return score
        if edge_type == "Application_REQUIRES_CriticalToQuality":
            application = self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
            ctq = self._nodes.get("CriticalToQuality", {}).get(str(edge.get("ctq_id")), {})
            score = 0.0
            if self._tokens(application.get("name")) & combined_terms:
                score += 4.0
            if self._node_tokens(ctq) & combined_terms:
                score += 3.0
            if self._node_tokens(ctq) & criterion_terms:
                score += 2.0
            if edge.get("property_requirements_json"):
                score += 1.0
            return score
        if edge_type == "CriticalToQuality_INDICATED_BY_Endpoint":
            ctq = self._nodes.get("CriticalToQuality", {}).get(str(edge.get("ctq_id")), {})
            endpoint = self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
            score = 0.0
            if self._node_tokens(ctq) & combined_terms:
                score += 3.0
            if self._node_tokens(endpoint) & combined_terms:
                score += 3.0
            if self._node_tokens(endpoint) & criterion_terms:
                score += 2.0
            return score
        if edge_type == "Market_HAS_COMPANY_Company":
            company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
            score = 0.0
            if self._tokens(market.get("name")) & combined_terms:
                score += 2.0
            if self._tokens(company.get("name")) & combined_terms:
                score += 4.0
            return score
        if edge_type == "Product_HAS_ChemistryClass":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            chemistry_class = self._nodes.get("ChemistryClass", {}).get(str(edge.get("chemistry_class_id")), {})
            score = 0.0
            if self._node_tokens(product) & combined_terms:
                score += 2.0
            if self._tokens(chemistry_class.get("name")) & combined_terms:
                score += 4.0
            return score
        if edge_type == "Product_HAS_Function":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            function = self._nodes.get("Function", {}).get(str(edge.get("function_id")), {})
            score = 0.0
            if self._node_tokens(product) & combined_terms:
                score += 2.0
            if self._tokens(function.get("name")) & combined_terms:
                score += 4.0
            return score
        if edge_type == "Product_TARGETS_BinderSystem":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            binder_system = self._nodes.get("BinderSystem", {}).get(str(edge.get("binder_system_id")), {})
            score = 0.0
            if self._node_tokens(product) & combined_terms:
                score += 2.0
            if self._tokens(binder_system.get("name")) & combined_terms:
                score += 4.0
            return score
        if edge_type == "Product_HAS_Endpoint":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            endpoint = self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
            score = 0.0
            endpoint_tokens = self._node_tokens(endpoint)
            if self._node_tokens(product) & combined_terms:
                score += 2.0
            if endpoint_tokens & combined_terms:
                score += 3.0
            if endpoint_tokens & criterion_terms:
                score += 3.0
            if self._number(edge.get("normalized_score")) is not None or edge.get("value_text"):
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
        goal_terms = self._document_terms(document)
        criterion_terms = self._criterion_terms(document)
        market = self._nodes.get("Market", {}).get(str(edge.get("market_id")), {})
        if edge_type == "Product_USED_IN_Application":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            application = self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
            product_tokens = self._node_tokens(product) | self._tokens(edge.get("product_id"))
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
        if edge_type == "Company_PRODUCES_Product":
            company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            score = 0.0
            if self._tokens(company.get("name")) & goal_terms:
                score += 4.0
            if self._node_tokens(product) & goal_terms:
                score += 3.0
            return score
        if edge_type == "Product_HAS_MaterialGrade":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            score = 0.0
            if self._node_tokens(product) & goal_terms:
                score += 3.0
            if self._node_tokens(material_grade) & goal_terms:
                score += 4.0
            return score
        if edge_type == "MaterialGrade_BELONGS_TO_MaterialFamily":
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            material_family = self._nodes.get("MaterialFamily", {}).get(str(edge.get("material_family_id")), {})
            score = 0.0
            if self._node_tokens(material_grade) & goal_terms:
                score += 3.0
            if self._node_tokens(material_family) & goal_terms:
                score += 4.0
            return score
        if edge_type == "Company_PRODUCES_MaterialGrade":
            company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            score = 0.0
            if self._tokens(company.get("name")) & goal_terms:
                score += 3.0
            if self._node_tokens(material_grade) & goal_terms:
                score += 4.0
            return score
        if edge_type == "MaterialGrade_HAS_Endpoint":
            material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
            endpoint = self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
            score = 0.0
            if self._node_tokens(material_grade) & goal_terms:
                score += 3.0
            if self._node_tokens(endpoint) & goal_terms:
                score += 3.0
            if self._node_tokens(endpoint) & criterion_terms:
                score += 3.0
            if self._number(edge.get("value_numeric")) is not None or edge.get("value_text"):
                score += 1.0
            return score
        if edge_type == "Application_REQUIRES_CriticalToQuality":
            application = self._nodes.get("Application", {}).get(str(edge.get("application_id")), {})
            ctq = self._nodes.get("CriticalToQuality", {}).get(str(edge.get("ctq_id")), {})
            score = 0.0
            if self._tokens(application.get("name")) & goal_terms:
                score += 4.0
            if self._node_tokens(ctq) & goal_terms:
                score += 3.0
            if self._node_tokens(ctq) & criterion_terms:
                score += 2.0
            if edge.get("property_requirements_json"):
                score += 1.0
            return score
        if edge_type == "CriticalToQuality_INDICATED_BY_Endpoint":
            ctq = self._nodes.get("CriticalToQuality", {}).get(str(edge.get("ctq_id")), {})
            endpoint = self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
            score = 0.0
            if self._node_tokens(ctq) & goal_terms:
                score += 3.0
            if self._node_tokens(endpoint) & goal_terms:
                score += 3.0
            if self._node_tokens(endpoint) & criterion_terms:
                score += 2.0
            return score
        if edge_type == "Market_HAS_COMPANY_Company":
            company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
            score = 0.0
            if self._tokens(market.get("name")) & goal_terms:
                score += 2.0
            if self._tokens(company.get("name")) & goal_terms:
                score += 4.0
            return score
        if edge_type == "Product_HAS_ChemistryClass":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            chemistry_class = self._nodes.get("ChemistryClass", {}).get(str(edge.get("chemistry_class_id")), {})
            score = 0.0
            if self._node_tokens(product) & goal_terms:
                score += 2.0
            if self._tokens(chemistry_class.get("name")) & goal_terms:
                score += 4.0
            return score
        if edge_type == "Product_HAS_Function":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            function = self._nodes.get("Function", {}).get(str(edge.get("function_id")), {})
            score = 0.0
            if self._node_tokens(product) & goal_terms:
                score += 2.0
            if self._tokens(function.get("name")) & goal_terms:
                score += 4.0
            return score
        if edge_type == "Product_TARGETS_BinderSystem":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            binder_system = self._nodes.get("BinderSystem", {}).get(str(edge.get("binder_system_id")), {})
            score = 0.0
            if self._node_tokens(product) & goal_terms:
                score += 2.0
            if self._tokens(binder_system.get("name")) & goal_terms:
                score += 4.0
            return score
        if edge_type == "Product_HAS_Endpoint":
            product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
            endpoint = self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
            score = 0.0
            endpoint_tokens = self._node_tokens(endpoint)
            if self._node_tokens(product) & goal_terms:
                score += 2.0
            if endpoint_tokens & goal_terms:
                score += 3.0
            if endpoint_tokens & criterion_terms:
                score += 3.0
            if self._number(edge.get("normalized_score")) is not None or edge.get("value_text"):
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
        if edge_type == "Company_PRODUCES_Product":
            return self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
        if edge_type == "Product_HAS_MaterialGrade":
            return self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
        if edge_type == "MaterialGrade_BELONGS_TO_MaterialFamily":
            return self._nodes.get("MaterialFamily", {}).get(str(edge.get("material_family_id")), {})
        if edge_type == "Company_PRODUCES_MaterialGrade":
            return self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
        if edge_type == "MaterialGrade_HAS_Endpoint":
            return self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
        if edge_type == "Application_REQUIRES_CriticalToQuality":
            return self._nodes.get("CriticalToQuality", {}).get(str(edge.get("ctq_id")), {})
        if edge_type == "CriticalToQuality_INDICATED_BY_Endpoint":
            return self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
        if edge_type == "Market_HAS_COMPANY_Company":
            return self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
        if edge_type == "Product_HAS_ChemistryClass":
            return self._nodes.get("ChemistryClass", {}).get(str(edge.get("chemistry_class_id")), {})
        if edge_type == "Product_HAS_Function":
            return self._nodes.get("Function", {}).get(str(edge.get("function_id")), {})
        if edge_type == "Product_TARGETS_BinderSystem":
            return self._nodes.get("BinderSystem", {}).get(str(edge.get("binder_system_id")), {})
        if edge_type == "Product_HAS_Endpoint":
            return self._nodes.get("Endpoint", {}).get(str(edge.get("endpoint_id")), {})
        return {}

    def _row_from_edge(self, edge_type: str, edge: dict[str, Any], score: float) -> dict[str, Any] | None:
        market = self._nodes.get("Market", {}).get(str(edge.get("market_id")), {})
        target = self._target_node(edge_type, edge)
        if not market and edge_type in (
            "Market_HAS_APPLICATION_Application",
            "Market_USES_Product",
            "Market_IN_GEOGRAPHY_Geography",
            "Market_HAS_COMPANY_Company",
        ):
            return None
        relationship = {
            "Market_HAS_APPLICATION_Application": "has application",
            "Market_USES_Product": "uses product",
            "Market_IN_GEOGRAPHY_Geography": "is measured in geography",
            "Product_USED_IN_Application": "is used in application",
            "Company_PRODUCES_Product": "produces product",
            "Product_HAS_MaterialGrade": "has material grade",
            "MaterialGrade_BELONGS_TO_MaterialFamily": "belongs to material family",
            "Company_PRODUCES_MaterialGrade": "produces material grade",
            "MaterialGrade_HAS_Endpoint": "has endpoint",
            "Application_REQUIRES_CriticalToQuality": "requires critical to quality feature",
            "CriticalToQuality_INDICATED_BY_Endpoint": "is indicated by endpoint",
            "Market_HAS_COMPANY_Company": "includes company",
            "Product_HAS_ChemistryClass": "belongs to chemistry class",
            "Product_HAS_Function": "has function",
            "Product_TARGETS_BinderSystem": "targets binder system",
            "Product_HAS_Endpoint": "has endpoint",
        }.get(edge_type, edge_type)
        product = self._nodes.get("Product", {}).get(str(edge.get("product_id")), {})
        company = self._nodes.get("Company", {}).get(str(edge.get("company_id")), {})
        material_grade = self._nodes.get("MaterialGrade", {}).get(str(edge.get("material_grade_id")), {})
        ctq = self._nodes.get("CriticalToQuality", {}).get(str(edge.get("ctq_id")), {})
        target_name = (
            target.get("name")
            or edge.get("application_id")
            or edge.get("product_id")
            or edge.get("material_grade_id")
            or edge.get("geo_id")
            or edge.get("company_id")
            or edge.get("ctq_id")
        )
        metrics_text = self._metrics_text(edge)
        notes = self._json_notes(
            edge,
            ["critical_to_quality_json", "property_requirements_json", "highlights_json", "industry_trends_json", "data_book_summary_json"],
            limit=4,
        )
        source_url = (
            edge.get("source_url")
            or edge.get("page_url")
            or edge.get("target_url")
            or market.get("canonical_url")
            or str(self._graph_path.resolve())
        )
        subject_name = self._subject_name(edge_type, market, product, company, material_grade, ctq)
        if edge_type in ("Product_HAS_Endpoint", "MaterialGrade_HAS_Endpoint"):
            endpoint_value_parts = [
                f"value={edge.get('value_numeric')}" if edge.get("value_numeric") is not None else None,
                f"text={edge.get('value_text')}" if edge.get("value_text") else None,
                f"normalized_score={edge.get('normalized_score')}" if edge.get("normalized_score") is not None else None,
                f"evidence_mode={edge.get('evidence_mode')}" if edge.get("evidence_mode") else None,
                f"tool_id={edge.get('tool_id')}" if edge.get("tool_id") else None,
            ]
            notes.append("Endpoint details: " + ", ".join(part for part in endpoint_value_parts if part))
        chunk_text = " ".join(
            item
            for item in [
                f"Graph evidence from {market.get('source_vendor') or 'offline graph'}: "
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
                if edge_type in ("Product_USED_IN_Application", "Product_HAS_ChemistryClass", "Product_HAS_Function", "Product_TARGETS_BinderSystem", "Product_HAS_Endpoint")
                and product.get("name")
                else [material_grade.get("name")]
                if edge_type in ("Product_HAS_MaterialGrade", "MaterialGrade_BELONGS_TO_MaterialFamily", "Company_PRODUCES_MaterialGrade", "MaterialGrade_HAS_Endpoint")
                and material_grade.get("name")
                else [ctq.get("name")]
                if edge_type in ("Application_REQUIRES_CriticalToQuality", "CriticalToQuality_INDICATED_BY_Endpoint")
                and ctq.get("name")
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
                "material_grade_id": edge.get("material_grade_id"),
                "material_family_id": edge.get("material_family_id"),
                "application_id": edge.get("application_id"),
                "ctq_id": edge.get("ctq_id"),
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
                "company_id": edge.get("company_id"),
                "chemistry_class_id": edge.get("chemistry_class_id"),
                "function_id": edge.get("function_id"),
                "binder_system_id": edge.get("binder_system_id"),
                "endpoint_id": edge.get("endpoint_id"),
                "value_text": edge.get("value_text"),
                "value_numeric": self._number(edge.get("value_numeric")),
                "value_min": self._number(edge.get("value_min")),
                "value_max": self._number(edge.get("value_max")),
                "normalized_score": self._number(edge.get("normalized_score")),
                "evidence_mode": edge.get("evidence_mode"),
                "tool_id": edge.get("tool_id"),
                "is_inferred": edge.get("is_inferred"),
                "property_requirements_json": edge.get("property_requirements_json"),
                "evidence_hash": edge.get("evidence_hash"),
                "source_chunk_id": edge.get("source_chunk_id"),
                "unit": edge.get("unit"),
                "currency": edge.get("currency"),
                "unit_scale": edge.get("unit_scale"),
            },
        }

    @staticmethod
    def _subject_name(
        edge_type: str,
        market: dict[str, Any],
        product: dict[str, Any],
        company: dict[str, Any],
        material_grade: dict[str, Any],
        ctq: dict[str, Any],
    ) -> str:
        if edge_type in (
            "Product_USED_IN_Application",
            "Product_HAS_MaterialGrade",
            "MaterialGrade_HAS_Endpoint",
            "Product_HAS_ChemistryClass",
            "Product_HAS_Function",
            "Product_TARGETS_BinderSystem",
            "Product_HAS_Endpoint",
        ):
            return product.get("name")
        if edge_type in (
            "MaterialGrade_BELONGS_TO_MaterialFamily",
            "Company_PRODUCES_MaterialGrade",
        ):
            return material_grade.get("name")
        if edge_type in (
            "Application_REQUIRES_CriticalToQuality",
            "CriticalToQuality_INDICATED_BY_Endpoint",
        ):
            return ctq.get("name")
        if edge_type == "Company_PRODUCES_Product":
            return company.get("name")
        return market.get("name")

    def _node_tokens(self, node: dict[str, Any]) -> set[str]:
        parts = [
            node.get("name"),
            node.get("canonical_name"),
            node.get("normalized_name"),
            " ".join(self._parse_json_list(node.get("aliases_json"))),
        ]
        return self._tokens(" ".join(str(part or "") for part in parts if part))

    @staticmethod
    def _parse_json_list(value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item]
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if item]

    def _document_terms(self, document: ResearchGoalDocument) -> set[str]:
        parts = [
            document.raw_goal,
            " ".join(document.material_scope),
            " ".join(document.application_scope),
            " ".join(document.preferred_candidate_materials),
            " ".join(document.target_incumbent_materials),
            " ".join(document.search_strategy_notes),
            document.candidate_artifact_schema.artifact_type,
        ]
        parts.extend(criterion.name for criterion in document.evaluation_criteria)
        parts.extend(criterion.description for criterion in document.evaluation_criteria)
        return self._tokens(" ".join(part for part in parts if part))

    def _hypothesis_terms(self, document: ResearchGoalDocument, hypothesis: Hypothesis) -> set[str]:
        artifact = hypothesis.candidate_artifact or {}
        parts = [
            hypothesis.title,
            hypothesis.summary,
            hypothesis.application,
            hypothesis.market_segment,
            hypothesis.candidate_material,
            hypothesis.incumbent_material,
            hypothesis.product_type,
            " ".join(str(value) for value in artifact.values() if value),
            " ".join(hypothesis.application_requirements),
            " ".join(hypothesis.substitution_drivers),
        ]
        if hypothesis.reflection_assessment is not None:
            parts.extend(hypothesis.reflection_assessment.evidence_gap_notes)
        return self._tokens(" ".join(part for part in parts if part)) | self._document_terms(document)

    def _criterion_terms(self, document: ResearchGoalDocument) -> set[str]:
        parts = []
        for criterion in document.evaluation_criteria:
            parts.append(criterion.name)
            parts.append(criterion.description)
        return self._tokens(" ".join(part for part in parts if part))

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
