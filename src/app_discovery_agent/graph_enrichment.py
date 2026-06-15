from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pyarrow as pa
import pyarrow.parquet as pq

from app_discovery_agent.models import (
    ChunkRecord,
    GraphEnrichmentMetric,
    GraphEnrichmentProposal,
    GraphEnrichmentProposalOutput,
    GraphEnrichmentValidation,
    GraphEnrichmentValidationOutput,
)
from app_discovery_agent.prompt_library import PROMPTS


LOGGER = logging.getLogger(__name__)
GRAPH_PATH = Path("data/graph")


class GraphEnrichmentProposer:
    def __init__(self, llm):
        self._llm = llm

    def propose(self, original_query: str, records: list[ChunkRecord], limit: int = 24) -> list[GraphEnrichmentProposal]:
        evidence_rows = [self._evidence_row(record) for record in records[:limit]]
        if not evidence_rows:
            return []
        system_prompt = PROMPTS.render("graph_enrichment_agent", "propose.system")
        user_prompt = PROMPTS.render(
            "graph_enrichment_agent",
            "propose.user",
            original_query=original_query,
            evidence_json=json.dumps(evidence_rows, indent=2),
        )
        output = self._llm.complete_json(GraphEnrichmentProposalOutput, system_prompt, user_prompt)
        return self._normalize_proposals(output.proposals, records)

    @staticmethod
    def _evidence_row(record: ChunkRecord) -> dict[str, Any]:
        return {
            "chunk_id": record.id,
            "source_title": record.source_title,
            "source_url": record.source_url,
            "application": record.application,
            "incumbent_material": record.incumbent_material,
            "candidate_materials": record.candidate_materials,
            "evidence_type": record.evidence_type,
            "application_requirements": record.application_requirements,
            "substitution_drivers": record.substitution_drivers,
            "confidence_score": record.confidence_score,
            "chunk_text": record.chunk_text[:2400],
        }

    @staticmethod
    def _normalize_proposals(
        proposals: list[GraphEnrichmentProposal],
        records: list[ChunkRecord],
    ) -> list[GraphEnrichmentProposal]:
        records_by_id = {record.id: record for record in records}
        normalized: "OrderedDict[str, GraphEnrichmentProposal]" = OrderedDict()
        for proposal in proposals:
            if proposal.source_chunk_id not in records_by_id:
                continue
            record = records_by_id[proposal.source_chunk_id]
            evidence_hash = proposal.evidence_hash or evidence_hash_for_record(record)
            proposal_id = proposal.proposal_id or proposal_id_for(proposal, evidence_hash)
            enriched = proposal.model_copy(
                update={
                    "proposal_id": proposal_id,
                    "source_url": proposal.source_url or record.source_url,
                    "source_title": proposal.source_title or record.source_title,
                    "evidence_hash": evidence_hash,
                }
            )
            normalized.setdefault(proposal_id, enriched)
        return list(normalized.values())


class GraphEnrichmentValidator:
    def __init__(self, llm):
        self._llm = llm

    def validate(
        self,
        proposals: list[GraphEnrichmentProposal],
        records: list[ChunkRecord],
    ) -> list[GraphEnrichmentValidation]:
        if not proposals:
            return []
        source_records = {record.id: record for record in records}
        proposal_rows = [proposal.model_dump(mode="json") for proposal in proposals]
        evidence_rows = [
            GraphEnrichmentProposer._evidence_row(source_records[proposal.source_chunk_id])
            for proposal in proposals
            if proposal.source_chunk_id in source_records
        ]
        system_prompt = PROMPTS.render("graph_enrichment_agent", "validate.system")
        user_prompt = PROMPTS.render(
            "graph_enrichment_agent",
            "validate.user",
            proposals_json=json.dumps(proposal_rows, indent=2),
            evidence_json=json.dumps(evidence_rows, indent=2),
        )
        output = self._llm.complete_json(GraphEnrichmentValidationOutput, system_prompt, user_prompt)
        known_ids = {proposal.proposal_id for proposal in proposals if proposal.proposal_id}
        return [validation for validation in output.validations if validation.proposal_id in known_ids]


class GraphEnrichmentStore:
    def __init__(self, graph_path: Path = GRAPH_PATH, min_promotion_confidence: float = 0.6):
        self._graph_path = graph_path
        self._nodes_path = graph_path / "nodes"
        self._edges_path = graph_path / "edges"
        self._enrichment_path = graph_path / "enrichment"
        self._min_promotion_confidence = min_promotion_confidence

    def write(
        self,
        proposals: list[GraphEnrichmentProposal],
        validations: list[GraphEnrichmentValidation],
        run_id: str,
        original_query: str,
    ) -> int:
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        self._edges_path.mkdir(parents=True, exist_ok=True)
        self._enrichment_path.mkdir(parents=True, exist_ok=True)

        validations_by_id = {validation.proposal_id: validation for validation in validations}
        self._append_claim_rows(proposals, validations_by_id, run_id, original_query)

        accepted_count = 0
        for proposal in proposals:
            if not proposal.proposal_id:
                continue
            validation = validations_by_id.get(proposal.proposal_id)
            if validation is None or not validation.accepted:
                continue
            if validation.confidence_score < self._min_promotion_confidence:
                continue
            self._write_accepted_edge(proposal, validation)
            accepted_count += 1
        return accepted_count

    def _append_claim_rows(
        self,
        proposals: list[GraphEnrichmentProposal],
        validations_by_id: dict[str, GraphEnrichmentValidation],
        run_id: str,
        original_query: str,
    ) -> None:
        now = now_iso()
        rows = []
        for proposal in proposals:
            validation = validations_by_id.get(proposal.proposal_id or "")
            rows.append(
                {
                    "claim_id": proposal.proposal_id,
                    "run_id": run_id,
                    "original_query": original_query,
                    "edge_type": proposal.edge_type,
                    "product_name": proposal.product_name,
                    "product_aliases_json": json.dumps(effective_product_aliases(proposal, validation), sort_keys=True),
                    "application_name": proposal.application_name,
                    "market_name": proposal.market_name,
                    "company_name": proposal.company_name,
                    "geography_name": proposal.geography_name,
                    "relationship_role": validation.corrected_relationship_role
                    if validation and validation.corrected_relationship_role
                    else proposal.relationship_role,
                    "critical_to_quality_json": json.dumps(
                        validation.corrected_critical_to_quality
                        if validation and validation.corrected_critical_to_quality
                        else proposal.critical_to_quality,
                        sort_keys=True,
                    ),
                    "metrics_json": json.dumps(
                        [
                            metric.model_dump(mode="json")
                            for metric in (
                                validation.corrected_metrics
                                if validation and validation.corrected_metrics
                                else proposal.metrics
                            )
                        ],
                        sort_keys=True,
                    ),
                    "source_chunk_id": proposal.source_chunk_id,
                    "source_url": proposal.source_url,
                    "source_title": proposal.source_title,
                    "supporting_quote": proposal.supporting_quote,
                    "proposal_rationale": proposal.rationale,
                    "proposal_confidence": proposal.confidence_score,
                    "validation_status": "accepted" if validation and validation.accepted else "rejected",
                    "validation_confidence": validation.confidence_score if validation else None,
                    "validation_rationale": validation.rationale if validation else None,
                    "evidence_hash": proposal.evidence_hash,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        self._append_unique_rows(self._enrichment_path / "GraphEnrichmentClaim.parquet", rows, CLAIM_SCHEMA, "claim_id")

    def _write_accepted_edge(self, proposal: GraphEnrichmentProposal, validation: GraphEnrichmentValidation) -> None:
        edge_type = validation.corrected_edge_type or proposal.edge_type
        if edge_type == "Product_USED_IN_Application":
            product_id = self._ensure_product_node(proposal, validation)
            application_id = self._ensure_node("Application", proposal.application_name, "application_id", "application")
            self._append_product_application_edge(proposal, validation, product_id, application_id)
        elif edge_type == "Company_PRODUCES_Product":
            company_id = self._ensure_node("Company", proposal.company_name, "company_id", None)
            product_id = self._ensure_product_node(proposal, validation)
            self._append_company_product_edge(proposal, validation, company_id, product_id)
        elif edge_type == "Market_USES_Product":
            market_id = self._ensure_market(proposal.market_name)
            product_id = self._ensure_product_node(proposal, validation)
            self._append_market_product_edge(proposal, validation, market_id, product_id)
        elif edge_type == "Market_HAS_APPLICATION_Application":
            market_id = self._ensure_market(proposal.market_name)
            application_id = self._ensure_node("Application", proposal.application_name, "application_id", "application")
            self._append_market_application_edge(proposal, validation, market_id, application_id)
        elif edge_type == "Market_HAS_COMPANY_Company":
            market_id = self._ensure_market(proposal.market_name)
            company_id = self._ensure_node("Company", proposal.company_name, "company_id", None)
            self._append_market_company_edge(proposal, validation, market_id, company_id)

    def _ensure_market(self, market_name: str | None) -> str:
        return self._ensure_node("Market", market_name, "market_id", None)

    def _ensure_product_node(self, proposal: GraphEnrichmentProposal, validation: GraphEnrichmentValidation) -> str:
        aliases = effective_product_aliases(proposal, validation)
        canonical_name = canonical_product_name(proposal.product_name, aliases)
        product_aliases = sorted(
            {
                alias.strip()
                for alias in [proposal.product_name, *aliases]
                if alias and normalize_name(alias) != normalize_name(canonical_name)
            },
            key=str.lower,
        )
        return self._ensure_node("Product", canonical_name, "product_id", "product", aliases=product_aliases)

    def _ensure_node(
        self,
        label: str,
        name: str | None,
        key: str,
        node_type: str | None,
        aliases: list[str] | None = None,
    ) -> str:
        safe_name = (name or "Unknown").strip() or "Unknown"
        normalized_name = normalize_name(safe_name)
        prefix = label.lower()
        node_id = f"{prefix}:{slugify(safe_name)}"
        schema = NODE_SCHEMAS[label]
        path = self._nodes_path / f"{label}.parquet"
        rows = rows_by_key(path, schema, key)
        now = now_iso()
        if node_id not in rows:
            row = empty_row(schema)
            row.update(
                {
                    key: node_id,
                    "name": safe_name,
                    "normalized_name": normalized_name,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            if "node_type" in schema.names and node_type:
                row["node_type"] = node_type
            if "aliases_json" in schema.names:
                row["aliases_json"] = json.dumps(sorted(set(aliases or []), key=str.lower), sort_keys=True)
            if label == "Market":
                row["primary_slug"] = f"{slugify(safe_name)}-market"
                row["canonical_url"] = None
                row["source_vendor"] = "evidence-enrichment"
            rows[node_id] = row
        else:
            if "aliases_json" in schema.names:
                existing_aliases = parse_json_list(rows[node_id].get("aliases_json"))
                merged_aliases = sorted(
                    {
                        alias.strip()
                        for alias in [*existing_aliases, *(aliases or [])]
                        if alias and normalize_name(alias) != normalize_name(rows[node_id].get("name"))
                    },
                    key=str.lower,
                )
                rows[node_id]["aliases_json"] = json.dumps(merged_aliases, sort_keys=True)
            rows[node_id]["updated_at"] = now
        write_rows(path, list(rows.values()), schema)
        return node_id

    def _append_product_application_edge(
        self,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
        product_id: str,
        application_id: str,
    ) -> None:
        metrics = effective_metrics(proposal, validation)
        row = self._base_edge_row(PRODUCT_APPLICATION_SCHEMA, proposal, validation)
        row.update(
            {
                "edge_id": stable_id("Product_USED_IN_Application", product_id, application_id, proposal.evidence_hash),
                "product_id": product_id,
                "application_id": application_id,
                "market_id": self._ensure_market(proposal.market_name) if proposal.market_name else None,
                "relationship_role": validation.corrected_relationship_role
                or proposal.relationship_role
                or "used_in",
                "volume_value": metric_value(metrics, "volume"),
                "volume_unit": metric_unit(metrics, "volume"),
                "volume_year": metric_year(metrics, "volume"),
                "price_value": metric_value(metrics, "price"),
                "price_currency": metric_currency(metrics, "price"),
                "price_unit": metric_unit(metrics, "price"),
                "price_year": metric_year(metrics, "price"),
                "critical_to_quality_json": json.dumps(effective_ctqs(proposal, validation), sort_keys=True),
            }
        )
        self._append_unique_rows(
            self._edges_path / "Product_USED_IN_Application.parquet",
            [row],
            PRODUCT_APPLICATION_SCHEMA,
            "edge_id",
        )

    def _append_company_product_edge(
        self,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
        company_id: str,
        product_id: str,
    ) -> None:
        row = self._base_edge_row(COMPANY_PRODUCT_SCHEMA, proposal, validation)
        row.update(
            {
                "edge_id": stable_id("Company_PRODUCES_Product", company_id, product_id, proposal.evidence_hash),
                "company_id": company_id,
                "product_id": product_id,
                "role": validation.corrected_relationship_role or proposal.relationship_role or "producer",
            }
        )
        self._append_unique_rows(self._edges_path / "Company_PRODUCES_Product.parquet", [row], COMPANY_PRODUCT_SCHEMA, "edge_id")

    def _append_market_product_edge(
        self,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
        market_id: str,
        product_id: str,
    ) -> None:
        metrics = effective_metrics(proposal, validation)
        row = self._base_market_edge_row(MARKET_PRODUCT_SCHEMA, proposal, validation, market_id)
        row.update(
            {
                "edge_id": stable_id("Market_USES_Product", market_id, product_id, proposal.evidence_hash),
                "product_id": product_id,
                "volume_value": metric_value(metrics, "volume"),
                "volume_unit": metric_unit(metrics, "volume"),
                "volume_year": metric_year(metrics, "volume"),
                "price_value": metric_value(metrics, "price"),
                "price_currency": metric_currency(metrics, "price"),
                "price_unit": metric_unit(metrics, "price"),
                "price_year": metric_year(metrics, "price"),
            }
        )
        self._append_unique_rows(self._edges_path / "Market_USES_Product.parquet", [row], MARKET_PRODUCT_SCHEMA, "edge_id")

    def _append_market_application_edge(
        self,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
        market_id: str,
        application_id: str,
    ) -> None:
        row = self._base_market_edge_row(MARKET_APPLICATION_SCHEMA, proposal, validation, market_id)
        row.update(
            {
                "edge_id": stable_id("Market_HAS_APPLICATION_Application", market_id, application_id, proposal.evidence_hash),
                "application_id": application_id,
                "critical_to_quality_json": json.dumps(effective_ctqs(proposal, validation), sort_keys=True),
            }
        )
        self._append_unique_rows(
            self._edges_path / "Market_HAS_APPLICATION_Application.parquet",
            [row],
            MARKET_APPLICATION_SCHEMA,
            "edge_id",
        )

    def _append_market_company_edge(
        self,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
        market_id: str,
        company_id: str,
    ) -> None:
        row = empty_row(MARKET_COMPANY_SCHEMA)
        now = now_iso()
        row.update(
            {
                "edge_id": stable_id("Market_HAS_COMPANY_Company", market_id, company_id, proposal.evidence_hash),
                "market_id": market_id,
                "company_id": company_id,
                "role": validation.corrected_relationship_role or proposal.relationship_role or "participant",
                "source": proposal.source_url,
                "queue_candidate": False,
                "source_chunk_id": proposal.source_chunk_id,
                "evidence_hash": proposal.evidence_hash,
                "supporting_quote": proposal.supporting_quote,
                "confidence": validation.confidence_score,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(self._edges_path / "Market_HAS_COMPANY_Company.parquet", [row], MARKET_COMPANY_SCHEMA, "edge_id")

    @staticmethod
    def _base_edge_row(
        schema: pa.Schema,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
    ) -> dict[str, Any]:
        row = empty_row(schema)
        now = now_iso()
        row.update(
            {
                "source_chunk_id": proposal.source_chunk_id,
                "source_url": proposal.source_url,
                "source_title": proposal.source_title,
                "evidence_hash": proposal.evidence_hash,
                "supporting_quote": proposal.supporting_quote,
                "confidence": validation.confidence_score,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        return row

    @classmethod
    def _base_market_edge_row(
        cls,
        schema: pa.Schema,
        proposal: GraphEnrichmentProposal,
        validation: GraphEnrichmentValidation,
        market_id: str,
    ) -> dict[str, Any]:
        row = cls._base_edge_row(schema, proposal, validation)
        metrics = effective_metrics(proposal, validation)
        row.update(
            {
                "market_id": market_id,
                "scope_type": "evidence",
                "source_node_type": "evidence_enrichment",
                "source_path": proposal.source_chunk_id,
                "geo_id": None,
                "page_type": "evidence",
                "page_url": proposal.source_url,
                "retrieved_at": None,
                "target_url": proposal.source_url,
                "target_page_type": "evidence",
                "queue_candidate": False,
                "status": "accepted",
                "revenue_value": metric_value(metrics, "revenue"),
                "revenue_year": metric_year(metrics, "revenue"),
                "forecast_revenue_value": metric_value(metrics, "forecast_revenue"),
                "forecast_revenue_year": metric_year(metrics, "forecast_revenue"),
                "cagr_value": metric_value(metrics, "cagr"),
                "cagr_start_year": None,
                "cagr_end_year": metric_year(metrics, "cagr"),
                "unit": metric_unit(metrics, "revenue") or metric_unit(metrics, "forecast_revenue"),
                "currency": metric_currency(metrics, "revenue") or metric_currency(metrics, "forecast_revenue"),
                "unit_scale": None,
                "summary_metrics_json": json.dumps([metric.model_dump(mode="json") for metric in metrics], sort_keys=True),
                "highlights_json": json.dumps([proposal.supporting_quote], sort_keys=True),
                "industry_trends_json": None,
                "data_book_summary_json": None,
                "source_market_slug": None,
            }
        )
        return row

    @staticmethod
    def _append_unique_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema, key: str) -> None:
        if not rows:
            return
        merged = rows_by_key(path, schema, key)
        for row in rows:
            row_key = row.get(key)
            if not row_key:
                continue
            existing = merged.get(str(row_key), empty_row(schema))
            existing.update({name: row.get(name) for name in schema.names if row.get(name) is not None})
            if existing.get("created_at") is None:
                existing["created_at"] = row.get("created_at")
            existing["updated_at"] = row.get("updated_at") or now_iso()
            merged[str(row_key)] = existing
        write_rows(path, list(merged.values()), schema)


def rows_by_key(path: Path, schema: pa.Schema, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = OrderedDict()
    for row in read_rows(path, schema):
        value = row.get(key)
        if value:
            rows[str(value)] = row
    return rows


def read_rows(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw_rows = pq.read_table(path).to_pylist()
    return [coerce_row(row, schema) for row in raw_rows]


def write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([coerce_row(row, schema) for row in rows], schema=schema)
    pq.write_table(table, path)


def coerce_row(row: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    coerced = empty_row(schema)
    for field in schema:
        value = row.get(field.name)
        if value is None:
            continue
        if pa.types.is_integer(field.type):
            try:
                coerced[field.name] = int(value)
            except (TypeError, ValueError):
                coerced[field.name] = None
        elif pa.types.is_floating(field.type):
            try:
                coerced[field.name] = float(value)
            except (TypeError, ValueError):
                coerced[field.name] = None
        elif pa.types.is_boolean(field.type):
            coerced[field.name] = bool(value)
        else:
            coerced[field.name] = str(value)
    return coerced


def empty_row(schema: pa.Schema) -> dict[str, Any]:
    return {name: None for name in schema.names}


def evidence_hash_for_record(record: ChunkRecord) -> str:
    payload = f"{record.id}\n{record.source_url}\n{record.chunk_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def proposal_id_for(proposal: GraphEnrichmentProposal, evidence_hash: str) -> str:
    identity = "::".join(
        [
            proposal.edge_type,
            proposal.product_name or "",
            proposal.application_name or "",
            proposal.market_name or "",
            proposal.company_name or "",
            proposal.source_chunk_id,
            evidence_hash,
        ]
    )
    return f"claim:{uuid5(NAMESPACE_URL, identity)}"


def stable_id(*parts: Any) -> str:
    return f"edge:{uuid5(NAMESPACE_URL, '::'.join(str(part or '') for part in parts))}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "unknown")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug or "unknown"


def market_id_or_none(market_name: str | None) -> str | None:
    if not market_name:
        return None
    return f"market:{slugify(market_name)}"


def effective_metrics(proposal: GraphEnrichmentProposal, validation: GraphEnrichmentValidation) -> list[GraphEnrichmentMetric]:
    return validation.corrected_metrics or proposal.metrics


def effective_ctqs(proposal: GraphEnrichmentProposal, validation: GraphEnrichmentValidation) -> list[str]:
    return validation.corrected_critical_to_quality or proposal.critical_to_quality


def effective_product_aliases(proposal: GraphEnrichmentProposal, validation: GraphEnrichmentValidation) -> list[str]:
    aliases = validation.corrected_product_aliases or proposal.product_aliases
    deduped: dict[str, str] = OrderedDict()
    for alias in aliases:
        text = str(alias or "").strip()
        if text:
            deduped.setdefault(normalize_name(text), text)
    return list(deduped.values())


def canonical_product_name(product_name: str | None, aliases: list[str]) -> str | None:
    candidates = [item.strip() for item in [product_name, *aliases] if item and item.strip()]
    if not candidates:
        return product_name
    current = candidates[0]
    if is_probable_abbreviation(current):
        fuller = [candidate for candidate in candidates[1:] if not is_probable_abbreviation(candidate)]
        if fuller:
            return max(fuller, key=len)
    return current


def is_probable_abbreviation(value: str) -> bool:
    text = re.sub(r"[^A-Za-z0-9]", "", value or "")
    return 1 < len(text) <= 5 and text.upper() == text


def parse_json_list(value: Any) -> list[str]:
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


def metric_value(metrics: list[GraphEnrichmentMetric], name: str) -> float | None:
    metric = next((item for item in metrics if item.name == name), None)
    return metric.value if metric else None


def metric_unit(metrics: list[GraphEnrichmentMetric], name: str) -> str | None:
    metric = next((item for item in metrics if item.name == name), None)
    return metric.unit if metric else None


def metric_currency(metrics: list[GraphEnrichmentMetric], name: str) -> str | None:
    metric = next((item for item in metrics if item.name == name), None)
    return metric.currency if metric else None


def metric_year(metrics: list[GraphEnrichmentMetric], name: str) -> int | None:
    metric = next((item for item in metrics if item.name == name), None)
    return metric.year if metric else None


CLAIM_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("run_id", pa.string()),
        ("original_query", pa.string()),
        ("edge_type", pa.string()),
        ("product_name", pa.string()),
        ("product_aliases_json", pa.string()),
        ("application_name", pa.string()),
        ("market_name", pa.string()),
        ("company_name", pa.string()),
        ("geography_name", pa.string()),
        ("relationship_role", pa.string()),
        ("critical_to_quality_json", pa.string()),
        ("metrics_json", pa.string()),
        ("source_chunk_id", pa.string()),
        ("source_url", pa.string()),
        ("source_title", pa.string()),
        ("supporting_quote", pa.string()),
        ("proposal_rationale", pa.string()),
        ("proposal_confidence", pa.float64()),
        ("validation_status", pa.string()),
        ("validation_confidence", pa.float64()),
        ("validation_rationale", pa.string()),
        ("evidence_hash", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)

PRODUCT_NODE_SCHEMA = pa.schema(
    [
        ("product_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("node_type", pa.string()),
        ("url", pa.string()),
        ("aliases_json", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
APPLICATION_NODE_SCHEMA = pa.schema(
    [
        ("application_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("node_type", pa.string()),
        ("url", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
MARKET_NODE_SCHEMA = pa.schema(
    [
        ("market_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("primary_slug", pa.string()),
        ("canonical_url", pa.string()),
        ("source_vendor", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
COMPANY_NODE_SCHEMA = pa.schema(
    [
        ("company_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("profile_url", pa.string()),
        ("website", pa.string()),
        ("website_domain", pa.string()),
        ("employees_raw", pa.string()),
        ("hq_raw", pa.string()),
        ("hq_country", pa.string()),
        ("hq_region", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
NODE_SCHEMAS = {
    "Product": PRODUCT_NODE_SCHEMA,
    "Application": APPLICATION_NODE_SCHEMA,
    "Market": MARKET_NODE_SCHEMA,
    "Company": COMPANY_NODE_SCHEMA,
}

MARKET_EDGE_FIELDS = [
    ("edge_id", pa.string()),
    ("market_id", pa.string()),
    ("scope_type", pa.string()),
    ("source_node_type", pa.string()),
    ("source_path", pa.string()),
    ("geo_id", pa.string()),
    ("page_type", pa.string()),
    ("page_url", pa.string()),
    ("retrieved_at", pa.string()),
    ("target_url", pa.string()),
    ("target_page_type", pa.string()),
    ("queue_candidate", pa.bool_()),
    ("status", pa.string()),
    ("revenue_value", pa.float64()),
    ("revenue_year", pa.int32()),
    ("forecast_revenue_value", pa.float64()),
    ("forecast_revenue_year", pa.int32()),
    ("cagr_value", pa.float64()),
    ("cagr_start_year", pa.int32()),
    ("cagr_end_year", pa.int32()),
    ("unit", pa.string()),
    ("currency", pa.string()),
    ("unit_scale", pa.string()),
    ("summary_metrics_json", pa.string()),
    ("highlights_json", pa.string()),
    ("industry_trends_json", pa.string()),
    ("data_book_summary_json", pa.string()),
    ("source_market_slug", pa.string()),
    ("volume_value", pa.float64()),
    ("volume_unit", pa.string()),
    ("volume_year", pa.int32()),
    ("price_value", pa.float64()),
    ("price_currency", pa.string()),
    ("price_unit", pa.string()),
    ("price_year", pa.int32()),
    ("source_chunk_id", pa.string()),
    ("evidence_hash", pa.string()),
    ("supporting_quote", pa.string()),
    ("confidence", pa.float64()),
    ("validation_status", pa.string()),
    ("source_title", pa.string()),
    ("source_url", pa.string()),
    ("created_at", pa.string()),
    ("updated_at", pa.string()),
]

MARKET_PRODUCT_SCHEMA = pa.schema(MARKET_EDGE_FIELDS[:2] + [("product_id", pa.string())] + MARKET_EDGE_FIELDS[2:])
MARKET_APPLICATION_SCHEMA = pa.schema(
    MARKET_EDGE_FIELDS[:2]
    + [("application_id", pa.string())]
    + MARKET_EDGE_FIELDS[2:]
    + [("critical_to_quality_json", pa.string())]
)

PRODUCT_APPLICATION_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("product_id", pa.string()),
        ("application_id", pa.string()),
        ("market_id", pa.string()),
        ("geo_id", pa.string()),
        ("relationship_role", pa.string()),
        ("volume_value", pa.float64()),
        ("volume_unit", pa.string()),
        ("volume_year", pa.int32()),
        ("price_value", pa.float64()),
        ("price_currency", pa.string()),
        ("price_unit", pa.string()),
        ("price_year", pa.int32()),
        ("critical_to_quality_json", pa.string()),
        ("source_chunk_id", pa.string()),
        ("source_url", pa.string()),
        ("source_title", pa.string()),
        ("evidence_hash", pa.string()),
        ("supporting_quote", pa.string()),
        ("confidence", pa.float64()),
        ("validation_status", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
COMPANY_PRODUCT_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("company_id", pa.string()),
        ("product_id", pa.string()),
        ("role", pa.string()),
        ("source_chunk_id", pa.string()),
        ("source_url", pa.string()),
        ("source_title", pa.string()),
        ("evidence_hash", pa.string()),
        ("supporting_quote", pa.string()),
        ("confidence", pa.float64()),
        ("validation_status", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
MARKET_COMPANY_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("market_id", pa.string()),
        ("company_id", pa.string()),
        ("role", pa.string()),
        ("source", pa.string()),
        ("queue_candidate", pa.bool_()),
        ("source_chunk_id", pa.string()),
        ("evidence_hash", pa.string()),
        ("supporting_quote", pa.string()),
        ("confidence", pa.float64()),
        ("validation_status", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
