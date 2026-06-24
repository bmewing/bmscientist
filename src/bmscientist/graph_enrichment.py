from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pyarrow as pa
import pyarrow.parquet as pq


from bmscientist.models import (
    ChunkRecord,
    GraphEnrichmentMetric,
    GraphEnrichmentProposal,
    GraphEnrichmentProposalOutput,
    GraphEnrichmentValidation,
    GraphEnrichmentValidationOutput,
)
from bmscientist.prompt_library import PROMPTS


LOGGER = logging.getLogger(__name__)
DEFAULT_GRAPH_PATH = Path("data/graph")
GRAPH_PATH = DEFAULT_GRAPH_PATH



class FileLock:
    def __init__(self, file_path: Path, timeout_seconds: float = 30.0, poll_interval: float = 0.1):
        self.lock_file = file_path.with_suffix(".lock")
        self.timeout = timeout_seconds
        self.poll_interval = poll_interval
        self.has_lock = False

    def __enter__(self):
        start_time = time.time()
        lock_id = secrets.token_hex(8)
        while True:
            try:
                # Try to create the lock file atomically
                self.lock_file.parent.mkdir(parents=True, exist_ok=True)
                self.lock_file.touch(exist_ok=False)
                self.lock_file.write_text(lock_id, encoding="utf-8")
                self.has_lock = True
                return self
            except (FileExistsError, OSError):
                # Clean up old stale locks (> 5 minutes)
                try:
                    if self.lock_file.exists():
                        mtime = self.lock_file.stat().st_mtime
                        if time.time() - mtime > 300:
                            self.lock_file.unlink(missing_ok=True)
                except Exception:
                    pass

                if time.time() - start_time > self.timeout:
                    LOGGER.warning("File lock timeout for %s, proceeding without lock", self.lock_file)
                    return self
                time.sleep(self.poll_interval + secrets.SystemRandom().uniform(0.0, 0.05))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_lock:
            try:
                self.lock_file.unlink(missing_ok=True)
            except Exception:
                pass


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
    def __init__(self, graph_path: Path | None = None, min_promotion_confidence: float = 0.6):
        self._graph_path = graph_path if graph_path is not None else GRAPH_PATH
        self._nodes_path = self._graph_path / "nodes"
        self._edges_path = self._graph_path / "edges"
        self._enrichment_path = self._graph_path / "enrichment"
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

    def ensure_material_family(
        self,
        name: str,
        *,
        canonical_name: str | None = None,
        family_type: str = "unknown",
        aliases: list[str] | None = None,
        alias_sources: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> str:
        return self._ensure_node(
            "MaterialFamily",
            canonical_name or name,
            "material_family_id",
            None,
            aliases=aliases,
            extra_fields={
                "canonical_name": canonical_name or name,
                "family_type": family_type,
                "alias_sources_json": json.dumps(alias_sources or {}, sort_keys=True),
                "description": description,
            },
        )

    def ensure_material_grade(
        self,
        name: str,
        *,
        record_type: str = "unknown",
        source_vendor: str = "matweb",
        source_url: str | None = None,
        source_record_id: str | None = None,
        source_search_query: str | None = None,
        manufacturer_name: str | None = None,
        trade_name: str | None = None,
        grade_name: str | None = None,
        material_family_name: str | None = None,
        category_path: list[str] | None = None,
        processing_methods: list[str] | None = None,
        applications: list[str] | None = None,
        property_table: list[dict[str, Any]] | None = None,
        raw_record: dict[str, Any] | None = None,
    ) -> str:
        grade_id = (
            f"material_grade:matweb:{slugify(source_record_id)}"
            if source_vendor == "matweb" and source_record_id
            else f"material_grade:{slugify(name)}"
        )
        path = self._nodes_path / "MaterialGrade.parquet"
        schema = NODE_SCHEMAS["MaterialGrade"]
        with FileLock(path):
            rows = rows_by_key(path, schema, "material_grade_id")
            now = now_iso()
            existing = rows.get(grade_id, empty_row(schema))
            existing.update(
                {
                    "material_grade_id": grade_id,
                    "name": name,
                    "normalized_name": normalize_name(name),
                    "record_type": record_type,
                    "source_vendor": source_vendor,
                    "source_url": source_url,
                    "source_record_id": source_record_id,
                    "source_search_query": source_search_query,
                    "manufacturer_name": manufacturer_name,
                    "trade_name": trade_name,
                    "grade_name": grade_name,
                    "material_family_name": material_family_name,
                    "category_path_json": json.dumps(category_path or [], sort_keys=True),
                    "processing_methods_json": json.dumps(processing_methods or [], sort_keys=True),
                    "applications_json": json.dumps(applications or [], sort_keys=True),
                    "property_table_json": json.dumps(property_table or [], sort_keys=True),
                    "raw_record_json": json.dumps(raw_record or {}, sort_keys=True),
                    "last_seen_at": now,
                    "updated_at": now,
                }
            )
            if not existing.get("created_at"):
                existing["created_at"] = now
            rows[grade_id] = existing
            write_rows(path, list(rows.values()), schema)
        return grade_id

    def ensure_critical_to_quality(
        self,
        name: str,
        *,
        requirement_type: str = "other",
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        return self._ensure_node(
            "CriticalToQuality",
            name,
            "ctq_id",
            None,
            aliases=aliases,
            extra_fields={"requirement_type": requirement_type, "description": description},
        )

    def ensure_material_alias(
        self,
        alias_text: str,
        canonical_node_id: str,
        canonical_node_label: str,
        *,
        alias_type: str = "unknown",
        source_vendor: str | None = None,
        source_url: str | None = None,
        evidence_hash: str | None = None,
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> str:
        alias_id = f"material_alias:{slugify(canonical_node_id)}:{slugify(alias_text)}"
        path = self._nodes_path / "MaterialAlias.parquet"
        schema = NODE_SCHEMAS["MaterialAlias"]
        with FileLock(path):
            rows = rows_by_key(path, schema, "material_alias_id")
            now = now_iso()
            existing = rows.get(alias_id, empty_row(schema))
            existing.update(
                {
                    "material_alias_id": alias_id,
                    "alias_text": alias_text,
                    "normalized_alias": normalize_name(alias_text),
                    "canonical_node_id": canonical_node_id,
                    "canonical_node_label": canonical_node_label,
                    "alias_type": alias_type,
                    "source_vendor": source_vendor,
                    "source_url": source_url,
                    "evidence_hash": evidence_hash,
                    "confidence": confidence,
                    "validation_status": validation_status,
                    "updated_at": now,
                }
            )
            if not existing.get("created_at"):
                existing["created_at"] = now
            rows[alias_id] = existing
            write_rows(path, list(rows.values()), schema)
        return alias_id

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
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        safe_name = (name or "Unknown").strip() or "Unknown"
        normalized_name = normalize_name(safe_name)
        prefix = {
            "MaterialFamily": "material_family",
            "CriticalToQuality": "ctq",
        }.get(label, label.lower())
        node_id = f"{prefix}:{slugify(safe_name)}"
        schema = NODE_SCHEMAS[label]
        path = self._nodes_path / f"{label}.parquet"
        with FileLock(path):
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
                if extra_fields:
                    for field_name, field_value in extra_fields.items():
                        if field_name in schema.names and field_value is not None:
                            row[field_name] = field_value
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
                if extra_fields:
                    for field_name, field_value in extra_fields.items():
                        if field_name in schema.names and field_value is not None:
                            rows[node_id][field_name] = field_value
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

    def write_ai_market_volume_estimate(self, hypothesis: Any, estimate: Any) -> list[dict[str, Any]]:
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        self._edges_path.mkdir(parents=True, exist_ok=True)
        self._enrichment_path.mkdir(parents=True, exist_ok=True)

        app_name = (getattr(estimate, "application_name", None) or getattr(hypothesis, "application", None) or "").strip()
        market_name = (getattr(estimate, "market_name", None) or getattr(hypothesis, "market_segment", None) or "").strip()
        if not app_name or not market_name:
            return []

        total_volume = getattr(estimate, "total_substrate_volume_value", None)
        material_volumes = list(getattr(estimate, "material_volumes", []) or [])
        if total_volume is None and not any(getattr(item, "volume_value", None) is not None for item in material_volumes):
            return []

        market_id = self._ensure_market(market_name)
        application_id = self._ensure_node("Application", app_name, "application_id", "application")
        now = now_iso()
        year = getattr(estimate, "volume_year", None) or datetime.now(timezone.utc).year
        confidence = max(0.35, min(0.65, float(getattr(estimate, "confidence", 0.5) or 0.5)))
        estimate_payload = estimate.model_dump(mode="json") if hasattr(estimate, "model_dump") else dict(estimate)
        evidence_hash = hashlib.sha256(
            json.dumps(
                {
                    "hypothesis_id": getattr(hypothesis, "hypothesis_id", None),
                    "market": market_name,
                    "application": app_name,
                    "estimate": estimate_payload,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        source_chunk_id = f"ai_volume_estimate:{getattr(hypothesis, 'hypothesis_id', 'unknown')}"
        source_url = self._estimate_source_url(hypothesis, estimate)
        source_title = "AI generated market volume estimate"
        rationale = str(getattr(estimate, "rationale", "") or "AI-generated market volume estimate.").strip()
        highlights = ["AI generated estimate", rationale]
        highlights.extend(self._estimate_citation_notes(estimate))

        written_rows: list[dict[str, Any]] = []
        if total_volume is not None:
            market_row = empty_row(MARKET_APPLICATION_SCHEMA)
            market_row.update(
                {
                    "edge_id": stable_id("Market_HAS_APPLICATION_Application", market_id, application_id, evidence_hash),
                    "market_id": market_id,
                    "application_id": application_id,
                    "scope_type": "ai_estimate",
                    "source_node_type": "ai_volume_estimate",
                    "source_path": source_chunk_id,
                    "geo_id": None,
                    "page_type": "ai_estimate",
                    "page_url": source_url,
                    "retrieved_at": now,
                    "target_url": source_url,
                    "target_page_type": "ai_estimate",
                    "queue_candidate": False,
                    "status": "accepted",
                    "revenue_value": getattr(estimate, "revenue_value", None),
                    "revenue_year": getattr(estimate, "revenue_year", None),
                    "unit": getattr(estimate, "revenue_unit", None),
                    "summary_metrics_json": json.dumps(estimate_payload, sort_keys=True),
                    "highlights_json": json.dumps(highlights, sort_keys=True),
                    "volume_value": total_volume,
                    "volume_unit": getattr(estimate, "total_substrate_volume_unit", None) or "metric_tons_per_year",
                    "volume_year": year,
                    "source_chunk_id": source_chunk_id,
                    "source_url": source_url,
                    "source_title": source_title,
                    "evidence_hash": evidence_hash,
                    "supporting_quote": rationale,
                    "confidence": confidence,
                    "validation_status": "accepted",
                    "critical_to_quality_json": json.dumps(getattr(hypothesis, "application_requirements", []) or [], sort_keys=True),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            self._append_unique_rows(
                self._edges_path / "Market_HAS_APPLICATION_Application.parquet",
                [market_row],
                MARKET_APPLICATION_SCHEMA,
                "edge_id",
            )
            written_rows.append(market_row)

        for material_estimate in material_volumes:
            material_name = str(getattr(material_estimate, "material_name", "") or "").strip()
            material_volume = getattr(material_estimate, "volume_value", None)
            if not material_name or material_volume is None:
                continue
            product_id = self._ensure_node("Product", material_name, "product_id", "product")
            material_confidence = max(0.3, min(0.65, float(getattr(material_estimate, "confidence", confidence) or confidence)))
            role = "ai_estimated_material_share"
            incumbent = str(getattr(hypothesis, "incumbent_material", "") or "").strip()
            candidate = str(getattr(hypothesis, "candidate_material", "") or "").strip()
            if incumbent and normalize_name(material_name) == normalize_name(incumbent):
                role = "incumbent_ai_estimated_share"
            elif candidate and normalize_name(material_name) == normalize_name(candidate):
                role = "candidate_ai_estimated_share"
            material_row = empty_row(PRODUCT_APPLICATION_SCHEMA)
            material_row.update(
                {
                    "edge_id": stable_id("Product_USED_IN_Application", product_id, application_id, evidence_hash),
                    "product_id": product_id,
                    "application_id": application_id,
                    "market_id": market_id,
                    "relationship_role": role,
                    "volume_value": material_volume,
                    "volume_unit": getattr(material_estimate, "volume_unit", None) or getattr(estimate, "total_substrate_volume_unit", None) or "metric_tons_per_year",
                    "volume_year": year,
                    "critical_to_quality_json": json.dumps(getattr(hypothesis, "application_requirements", []) or [], sort_keys=True),
                    "source_chunk_id": source_chunk_id,
                    "source_url": source_url,
                    "source_title": source_title,
                    "evidence_hash": evidence_hash,
                    "supporting_quote": str(getattr(material_estimate, "rationale", "") or rationale).strip(),
                    "confidence": material_confidence,
                    "validation_status": "accepted",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            self._append_unique_rows(
                self._edges_path / "Product_USED_IN_Application.parquet",
                [material_row],
                PRODUCT_APPLICATION_SCHEMA,
                "edge_id",
            )
            written_rows.append(material_row)
        return written_rows

    @staticmethod
    def _estimate_source_url(hypothesis: Any, estimate: Any) -> str:
        for citation in getattr(estimate, "source_citations", []) or []:
            source_url = getattr(citation, "source_url", None)
            if source_url:
                return str(source_url)
        return f"coscientist://research/{getattr(hypothesis, 'research_id', 'unknown')}/hypothesis/{getattr(hypothesis, 'hypothesis_id', 'unknown')}"

    @staticmethod
    def _estimate_citation_notes(estimate: Any) -> list[str]:
        notes: list[str] = []
        for citation in getattr(estimate, "source_citations", []) or []:
            title = getattr(citation, "source_title", None)
            url = getattr(citation, "source_url", None)
            if title or url:
                notes.append("Source: " + " ".join(str(item) for item in [title, url] if item))
        return notes[:6]

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

    def promote_hypothesis(self, hypothesis: Any) -> None:
        """
        Promotes a reflected hypothesis directly to the knowledge graph under data/graph/.
        This maps:
        - Product (candidate_material or generic candidate artifact identifier)
        - Product (incumbent_material)
        - Application (application)
        - Market (market_segment)
        - Company (manufacturer / supplier when explicit)
        - ChemistryClass, Function, BinderSystem, Endpoint when explicit in candidate artifacts or reflection outputs
        - Edges:
          - Product_USED_IN_Application (candidate_material -> application)
          - Product_USED_IN_Application (incumbent_material -> application)
          - Market_HAS_APPLICATION_Application (market -> application)
          - Market_USES_Product (market -> candidate_material)
          - Market_USES_Product (market -> incumbent_material)
        """
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        self._edges_path.mkdir(parents=True, exist_ok=True)
        self._enrichment_path.mkdir(parents=True, exist_ok=True)

        candidate_artifact = dict(getattr(hypothesis, "candidate_artifact", {}) or {})
        candidate_mat = self._candidate_label(hypothesis, candidate_artifact)
        incumbent_mat = (hypothesis.incumbent_material or "").strip()
        app_name = (hypothesis.application or "").strip()
        market_name = (hypothesis.market_segment or "").strip()

        if not candidate_mat:
            return

        now = now_iso()
        evidence_hash = hashlib.sha256(f"hypothesis:{hypothesis.hypothesis_id}".encode("utf-8")).hexdigest()
        source_chunk_id = f"hypothesis:{hypothesis.hypothesis_id}"
        source_title = f"Hypothesis: {hypothesis.title}"
        source_url = f"coscientist://research/{hypothesis.research_id}/hypothesis/{hypothesis.hypothesis_id}"

        # 1. Create/Ensure Nodes
        # Candidate Product
        candidate_id = self._ensure_product_node_from_hypothesis(candidate_mat, candidate_artifact)

        # Incumbent Product
        incumbent_id = None
        if incumbent_mat:
            incumbent_id = self._ensure_node("Product", incumbent_mat, "product_id", "product")

        # Application
        application_id = None
        if app_name:
            application_id = self._ensure_node("Application", app_name, "application_id", "application")

        # Market
        market_id = None
        if market_name:
            self._ensure_node("Market", market_name, "market_id", None)
            market_id = f"market:{slugify(market_name)}"

        # Retrieve reflection assessments if available
        assessment = hypothesis.reflection_assessment
        confidence = 0.8  # default high confidence since it was generated and reflected
        if assessment:
            # Derive confidence from technical & commercial success probabilities if available
            tech_prob = getattr(assessment.technical_success_probability, "value", None)
            comm_prob = getattr(assessment.commercial_success_probability, "value", None)
            if tech_prob is not None and comm_prob is not None:
                confidence = float(tech_prob + comm_prob) / 2.0
            elif tech_prob is not None:
                confidence = float(tech_prob)
            elif comm_prob is not None:
                confidence = float(comm_prob)

        ctq_json = json.dumps(hypothesis.application_requirements, sort_keys=True)

        # 2. Candidate Material -> Application Edge
        if application_id is not None:
            cand_edge_id = stable_id("Product_USED_IN_Application", candidate_id, application_id, evidence_hash)

            nbca_price = None
            price_unit = None
            price_currency = None
            if assessment and assessment.nbca_price_usd_per_kg and assessment.nbca_price_usd_per_kg.value is not None:
                nbca_price = float(assessment.nbca_price_usd_per_kg.value)
                price_unit = "kg"
                price_currency = "USD"

            candidate_edge_row = empty_row(PRODUCT_APPLICATION_SCHEMA)
            candidate_edge_row.update({
                "edge_id": cand_edge_id,
                "product_id": candidate_id,
                "application_id": application_id,
                "market_id": market_id,
                "relationship_role": "candidate_replacement",
                "volume_value": None,
                "volume_unit": None,
                "volume_year": None,
                "price_value": nbca_price,
                "price_currency": price_currency,
                "price_unit": price_unit,
                "price_year": datetime.now(timezone.utc).year,
                "critical_to_quality_json": ctq_json,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": hypothesis.summary,
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            })
            self._append_unique_rows(
                self._edges_path / "Product_USED_IN_Application.parquet",
                [candidate_edge_row],
                PRODUCT_APPLICATION_SCHEMA,
                "edge_id",
            )

        # 3. Incumbent Material -> Application Edge (if incumbent exists)
        if incumbent_id and application_id is not None:
            inc_edge_id = stable_id("Product_USED_IN_Application", incumbent_id, application_id, evidence_hash)
            
            inc_price = None
            if assessment and assessment.incumbent_price_usd_per_kg and assessment.incumbent_price_usd_per_kg.value is not None:
                inc_price = float(assessment.incumbent_price_usd_per_kg.value)
                price_unit = "kg"
                price_currency = "USD"

            incumbent_edge_row = empty_row(PRODUCT_APPLICATION_SCHEMA)
            incumbent_edge_row.update({
                "edge_id": inc_edge_id,
                "product_id": incumbent_id,
                "application_id": application_id,
                "market_id": market_id,
                "relationship_role": "incumbent",
                "volume_value": None,
                "volume_unit": None,
                "volume_year": None,
                "price_value": inc_price,
                "price_currency": price_currency,
                "price_unit": price_unit,
                "price_year": datetime.now(timezone.utc).year,
                "critical_to_quality_json": ctq_json,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": f"Displaced by {candidate_mat} in {app_name}. Rationale: {hypothesis.strategic_rationale}",
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            })
            self._append_unique_rows(
                self._edges_path / "Product_USED_IN_Application.parquet",
                [incumbent_edge_row],
                PRODUCT_APPLICATION_SCHEMA,
                "edge_id",
            )

        # 4. Market Edges (if market exists)
        if market_id:
            if application_id is not None:
                # Market HAS APPLICATION Application
                mkt_app_edge_id = stable_id("Market_HAS_APPLICATION_Application", market_id, application_id, evidence_hash)
                mkt_app_row = empty_row(MARKET_APPLICATION_SCHEMA)
                mkt_app_row.update({
                    "edge_id": mkt_app_edge_id,
                    "market_id": market_id,
                    "application_id": application_id,
                    "scope_type": "evidence",
                    "source_node_type": "hypothesis_promotion",
                    "source_path": source_chunk_id,
                    "geo_id": None,
                    "page_type": "evidence",
                    "page_url": source_url,
                    "retrieved_at": None,
                    "target_url": source_url,
                    "target_page_type": "evidence",
                    "queue_candidate": False,
                    "status": "accepted",
                    "revenue_value": None,
                    "revenue_year": None,
                    "forecast_revenue_value": None,
                    "forecast_revenue_year": None,
                    "cagr_value": None,
                    "cagr_start_year": None,
                    "cagr_end_year": None,
                    "unit": None,
                    "currency": None,
                    "unit_scale": None,
                    "summary_metrics_json": "[]",
                    "highlights_json": json.dumps([hypothesis.summary], sort_keys=True),
                    "industry_trends_json": None,
                    "data_book_summary_json": None,
                    "source_market_slug": None,
                    "critical_to_quality_json": ctq_json,
                    "source_chunk_id": source_chunk_id,
                    "source_url": source_url,
                    "source_title": source_title,
                    "evidence_hash": evidence_hash,
                    "supporting_quote": hypothesis.summary,
                    "confidence": confidence,
                    "validation_status": "accepted",
                    "created_at": now,
                    "updated_at": now,
                })
                self._append_unique_rows(
                    self._edges_path / "Market_HAS_APPLICATION_Application.parquet",
                    [mkt_app_row],
                    MARKET_APPLICATION_SCHEMA,
                    "edge_id",
                )

            # Market USES Product (for Candidate)
            mkt_cand_edge_id = stable_id("Market_USES_Product", market_id, candidate_id, evidence_hash)
            mkt_cand_row = empty_row(MARKET_PRODUCT_SCHEMA)
            mkt_cand_row.update({
                "edge_id": mkt_cand_edge_id,
                "market_id": market_id,
                "product_id": candidate_id,
                "scope_type": "evidence",
                "source_node_type": "hypothesis_promotion",
                "source_path": source_chunk_id,
                "geo_id": None,
                "page_type": "evidence",
                "page_url": source_url,
                "retrieved_at": None,
                "target_url": source_url,
                "target_page_type": "evidence",
                "queue_candidate": False,
                "status": "accepted",
                "revenue_value": None,
                "revenue_year": None,
                "forecast_revenue_value": None,
                "forecast_revenue_year": None,
                "cagr_value": None,
                "cagr_start_year": None,
                "cagr_end_year": None,
                "unit": None,
                "currency": None,
                "unit_scale": None,
                "summary_metrics_json": "[]",
                "highlights_json": json.dumps([hypothesis.summary], sort_keys=True),
                "industry_trends_json": None,
                "data_book_summary_json": None,
                "source_market_slug": None,
                "volume_value": None,
                "volume_unit": None,
                "volume_year": None,
                "price_value": nbca_price,
                "price_currency": price_currency,
                "price_unit": price_unit,
                "price_year": datetime.now(timezone.utc).year,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": hypothesis.summary,
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            })
            self._append_unique_rows(
                self._edges_path / "Market_USES_Product.parquet",
                [mkt_cand_row],
                MARKET_PRODUCT_SCHEMA,
                "edge_id",
            )

        self._promote_generic_candidate_context(
            hypothesis=hypothesis,
            candidate_id=candidate_id,
            candidate_name=candidate_mat,
            candidate_artifact=candidate_artifact,
            source_chunk_id=source_chunk_id,
            source_url=source_url,
            source_title=source_title,
            evidence_hash=evidence_hash,
            confidence=confidence,
            now=now,
        )

    def _promote_generic_candidate_context(
        self,
        hypothesis: Any,
        candidate_id: str,
        candidate_name: str,
        candidate_artifact: dict[str, Any],
        source_chunk_id: str,
        source_url: str,
        source_title: str,
        evidence_hash: str,
        confidence: float,
        now: str,
    ) -> None:
        chemistry_class = self._first_artifact_value(
            candidate_artifact,
            "chemistry_class",
            "chemistry_family",
            "chemical_family",
            "chemistry_style",
            "scaffold",
        )
        if chemistry_class:
            chemistry_class_id = self._ensure_node("ChemistryClass", chemistry_class, "chemistry_class_id", None)
            self._append_product_chemistry_class_edge(
                candidate_id,
                chemistry_class_id,
                source_chunk_id,
                source_url,
                source_title,
                evidence_hash,
                hypothesis.summary,
                confidence,
                now,
            )

        binder_system = self._first_artifact_value(
            candidate_artifact,
            "intended_binder_system",
            "binder_system",
            "target_binder_system",
        )
        if binder_system:
            binder_system_id = self._ensure_node("BinderSystem", binder_system, "binder_system_id", None)
            self._append_product_binder_system_edge(
                candidate_id,
                binder_system_id,
                source_chunk_id,
                source_url,
                source_title,
                evidence_hash,
                hypothesis.summary,
                confidence,
                now,
            )

        manufacturer = self._first_artifact_value(
            candidate_artifact,
            "manufacturer",
            "supplier",
            "vendor",
            "produced_by",
            "company_name",
        )
        if manufacturer:
            company_id = self._ensure_node("Company", manufacturer, "company_id", None)
            self._append_company_product_edge_from_hypothesis(
                company_id,
                candidate_id,
                source_chunk_id,
                source_url,
                source_title,
                evidence_hash,
                hypothesis.summary,
                confidence,
                now,
            )

        for function_name in self._candidate_function_names(hypothesis, candidate_artifact):
            function_id = self._ensure_node("Function", function_name, "function_id", None)
            self._append_product_function_edge(
                candidate_id,
                function_id,
                source_chunk_id,
                source_url,
                source_title,
                evidence_hash,
                hypothesis.summary,
                confidence,
                now,
            )

        for result in self._candidate_endpoint_results(hypothesis):
            endpoint_name = str(result.criterion_name or "").strip()
            if not endpoint_name:
                continue
            endpoint_id = self._ensure_node(
                "Endpoint",
                endpoint_name,
                "endpoint_id",
                None,
                extra_fields={"endpoint_category": self._endpoint_category(endpoint_name)},
            )
            self._append_product_endpoint_edge(
                candidate_id,
                endpoint_id,
                result,
                source_chunk_id,
                source_url,
                source_title,
                evidence_hash,
                confidence,
                now,
            )

    def _ensure_product_node_from_hypothesis(self, candidate_name: str, candidate_artifact: dict[str, Any]) -> str:
        aliases = [
            value
            for value in [
                candidate_artifact.get("candidate_material"),
                candidate_artifact.get("name"),
                candidate_artifact.get("trade_name"),
                candidate_artifact.get("common_name"),
                candidate_artifact.get("name_or_label"),
            ]
            if value and str(value).strip() != candidate_name
        ]
        return self._ensure_node(
            "Product",
            candidate_name,
            "product_id",
            "product",
            aliases=[str(alias).strip() for alias in aliases if str(alias).strip()],
            extra_fields={
                "canonical_smiles": self._first_artifact_value(candidate_artifact, "smiles", "canonical_smiles"),
                "inchi_key": self._first_artifact_value(candidate_artifact, "inchi_key", "inchikey"),
                "cas_number": self._first_artifact_value(candidate_artifact, "cas_number", "cas"),
                "product_family": self._first_artifact_value(
                    candidate_artifact,
                    "chemistry_class",
                    "chemistry_family",
                    "chemical_family",
                ),
            },
        )

    def _append_company_product_edge_from_hypothesis(
        self,
        company_id: str,
        product_id: str,
        source_chunk_id: str,
        source_url: str,
        source_title: str,
        evidence_hash: str,
        supporting_quote: str,
        confidence: float,
        now: str,
    ) -> None:
        row = empty_row(COMPANY_PRODUCT_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Company_PRODUCES_Product", company_id, product_id, evidence_hash),
                "company_id": company_id,
                "product_id": product_id,
                "role": "producer_or_supplier",
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(self._edges_path / "Company_PRODUCES_Product.parquet", [row], COMPANY_PRODUCT_SCHEMA, "edge_id")

    def _append_product_chemistry_class_edge(
        self,
        product_id: str,
        chemistry_class_id: str,
        source_chunk_id: str,
        source_url: str,
        source_title: str,
        evidence_hash: str,
        supporting_quote: str,
        confidence: float,
        now: str,
    ) -> None:
        row = empty_row(PRODUCT_CHEMISTRY_CLASS_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Product_HAS_ChemistryClass", product_id, chemistry_class_id, evidence_hash),
                "product_id": product_id,
                "chemistry_class_id": chemistry_class_id,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Product_HAS_ChemistryClass.parquet",
            [row],
            PRODUCT_CHEMISTRY_CLASS_SCHEMA,
            "edge_id",
        )

    def _append_product_function_edge(
        self,
        product_id: str,
        function_id: str,
        source_chunk_id: str,
        source_url: str,
        source_title: str,
        evidence_hash: str,
        supporting_quote: str,
        confidence: float,
        now: str,
    ) -> None:
        row = empty_row(PRODUCT_FUNCTION_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Product_HAS_Function", product_id, function_id, evidence_hash),
                "product_id": product_id,
                "function_id": function_id,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Product_HAS_Function.parquet",
            [row],
            PRODUCT_FUNCTION_SCHEMA,
            "edge_id",
        )

    def _append_product_binder_system_edge(
        self,
        product_id: str,
        binder_system_id: str,
        source_chunk_id: str,
        source_url: str,
        source_title: str,
        evidence_hash: str,
        supporting_quote: str,
        confidence: float,
        now: str,
    ) -> None:
        row = empty_row(PRODUCT_BINDER_SYSTEM_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Product_TARGETS_BinderSystem", product_id, binder_system_id, evidence_hash),
                "product_id": product_id,
                "binder_system_id": binder_system_id,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Product_TARGETS_BinderSystem.parquet",
            [row],
            PRODUCT_BINDER_SYSTEM_SCHEMA,
            "edge_id",
        )

    def _append_product_endpoint_edge(
        self,
        product_id: str,
        endpoint_id: str,
        result: Any,
        source_chunk_id: str,
        source_url: str,
        source_title: str,
        evidence_hash: str,
        confidence: float,
        now: str,
    ) -> None:
        row = empty_row(PRODUCT_ENDPOINT_SCHEMA)
        numeric_value = None
        if isinstance(result.value, (int, float)):
            numeric_value = float(result.value)
        row.update(
            {
                "edge_id": stable_id("Product_HAS_Endpoint", product_id, endpoint_id, evidence_hash),
                "product_id": product_id,
                "endpoint_id": endpoint_id,
                "value_text": None if numeric_value is not None else (str(result.value) if result.value is not None else None),
                "value_numeric": numeric_value,
                "unit": result.unit,
                "normalized_score": result.normalized_score,
                "evidence_mode": result.evidence_mode,
                "tool_id": result.tool_id,
                "is_inferred": result.is_inferred,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": result.rationale or "",
                "confidence": max(confidence, result.confidence),
                "validation_status": "accepted",
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Product_HAS_Endpoint.parquet",
            [row],
            PRODUCT_ENDPOINT_SCHEMA,
            "edge_id",
        )

    def append_product_material_grade_edge(
        self,
        product_id: str,
        material_grade_id: str,
        *,
        relationship_role: str = "product_line_grade",
        source_url: str | None = None,
        source_title: str | None = None,
        evidence_hash: str | None = None,
        supporting_quote: str = "",
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> None:
        now = now_iso()
        row = empty_row(PRODUCT_MATERIAL_GRADE_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Product_HAS_MaterialGrade", product_id, material_grade_id, evidence_hash or ""),
                "product_id": product_id,
                "material_grade_id": material_grade_id,
                "relationship_role": relationship_role,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": validation_status,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Product_HAS_MaterialGrade.parquet",
            [row],
            PRODUCT_MATERIAL_GRADE_SCHEMA,
            "edge_id",
        )

    def append_material_grade_family_edge(
        self,
        material_grade_id: str,
        material_family_id: str,
        *,
        relationship_role: str = "base_family",
        source_url: str | None = None,
        source_title: str | None = None,
        evidence_hash: str | None = None,
        supporting_quote: str = "",
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> None:
        now = now_iso()
        row = empty_row(MATERIAL_GRADE_MATERIAL_FAMILY_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("MaterialGrade_BELONGS_TO_MaterialFamily", material_grade_id, material_family_id, evidence_hash or ""),
                "material_grade_id": material_grade_id,
                "material_family_id": material_family_id,
                "relationship_role": relationship_role,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": validation_status,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "MaterialGrade_BELONGS_TO_MaterialFamily.parquet",
            [row],
            MATERIAL_GRADE_MATERIAL_FAMILY_SCHEMA,
            "edge_id",
        )

    def append_company_material_grade_edge(
        self,
        company_id: str,
        material_grade_id: str,
        *,
        role: str = "manufacturer",
        source_url: str | None = None,
        source_title: str | None = None,
        evidence_hash: str | None = None,
        supporting_quote: str = "",
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> None:
        now = now_iso()
        row = empty_row(COMPANY_MATERIAL_GRADE_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Company_PRODUCES_MaterialGrade", company_id, material_grade_id, evidence_hash or ""),
                "company_id": company_id,
                "material_grade_id": material_grade_id,
                "role": role,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": validation_status,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Company_PRODUCES_MaterialGrade.parquet",
            [row],
            COMPANY_MATERIAL_GRADE_SCHEMA,
            "edge_id",
        )

    def append_material_grade_endpoint_edge(
        self,
        material_grade_id: str,
        endpoint_id: str,
        *,
        value_text: str | None = None,
        value_numeric: float | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        unit: str | None = None,
        condition_text: str | None = None,
        test_method: str | None = None,
        original_property_name: str | None = None,
        original_value_text: str | None = None,
        original_unit: str | None = None,
        normalized_score: float | None = None,
        evidence_mode: str = "matweb_table",
        tool_id: str | None = None,
        is_inferred: bool = False,
        source_chunk_id: str | None = None,
        source_url: str | None = None,
        source_title: str | None = None,
        evidence_hash: str | None = None,
        supporting_quote: str = "",
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> None:
        now = now_iso()
        row = empty_row(MATERIAL_GRADE_ENDPOINT_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("MaterialGrade_HAS_Endpoint", material_grade_id, endpoint_id, evidence_hash or ""),
                "material_grade_id": material_grade_id,
                "endpoint_id": endpoint_id,
                "value_text": value_text,
                "value_numeric": value_numeric,
                "value_min": value_min,
                "value_max": value_max,
                "unit": unit,
                "condition_text": condition_text,
                "test_method": test_method,
                "original_property_name": original_property_name,
                "original_value_text": original_value_text,
                "original_unit": original_unit,
                "normalized_score": normalized_score,
                "evidence_mode": evidence_mode,
                "tool_id": tool_id,
                "is_inferred": is_inferred,
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": validation_status,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "MaterialGrade_HAS_Endpoint.parquet",
            [row],
            MATERIAL_GRADE_ENDPOINT_SCHEMA,
            "edge_id",
        )

    def append_application_ctq_edge(
        self,
        application_id: str,
        ctq_id: str,
        *,
        market_id: str | None = None,
        requirement_role: str = "must_have",
        requirement_text: str = "",
        property_requirements: dict[str, Any] | None = None,
        source_chunk_id: str | None = None,
        source_url: str | None = None,
        source_title: str | None = None,
        evidence_hash: str | None = None,
        supporting_quote: str = "",
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> None:
        now = now_iso()
        row = empty_row(APPLICATION_CTQ_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("Application_REQUIRES_CriticalToQuality", application_id, ctq_id, evidence_hash or ""),
                "application_id": application_id,
                "ctq_id": ctq_id,
                "market_id": market_id,
                "requirement_role": requirement_role,
                "requirement_text": requirement_text,
                "property_requirements_json": json.dumps(property_requirements or {}, sort_keys=True),
                "source_chunk_id": source_chunk_id,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": validation_status,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "Application_REQUIRES_CriticalToQuality.parquet",
            [row],
            APPLICATION_CTQ_SCHEMA,
            "edge_id",
        )

    def append_ctq_endpoint_edge(
        self,
        ctq_id: str,
        endpoint_id: str,
        *,
        direction: str,
        default_threshold_value: float | None = None,
        default_threshold_min: float | None = None,
        default_threshold_max: float | None = None,
        unit: str | None = None,
        condition_text: str | None = None,
        rationale: str = "",
        source_url: str | None = None,
        source_title: str | None = None,
        evidence_hash: str | None = None,
        supporting_quote: str = "",
        confidence: float = 1.0,
        validation_status: str = "accepted",
    ) -> None:
        now = now_iso()
        row = empty_row(CTQ_ENDPOINT_SCHEMA)
        row.update(
            {
                "edge_id": stable_id("CriticalToQuality_INDICATED_BY_Endpoint", ctq_id, endpoint_id, evidence_hash or ""),
                "ctq_id": ctq_id,
                "endpoint_id": endpoint_id,
                "direction": direction,
                "default_threshold_value": default_threshold_value,
                "default_threshold_min": default_threshold_min,
                "default_threshold_max": default_threshold_max,
                "unit": unit,
                "condition_text": condition_text,
                "rationale": rationale,
                "source_url": source_url,
                "source_title": source_title,
                "evidence_hash": evidence_hash,
                "supporting_quote": supporting_quote,
                "confidence": confidence,
                "validation_status": validation_status,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._append_unique_rows(
            self._edges_path / "CriticalToQuality_INDICATED_BY_Endpoint.parquet",
            [row],
            CTQ_ENDPOINT_SCHEMA,
            "edge_id",
        )

    @staticmethod
    def _candidate_label(hypothesis: Any, candidate_artifact: dict[str, Any]) -> str:
        values = [
            getattr(hypothesis, "candidate_material", None),
            candidate_artifact.get("name_or_label"),
            candidate_artifact.get("trade_name"),
            candidate_artifact.get("name"),
            candidate_artifact.get("smiles"),
        ]
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _first_artifact_value(candidate_artifact: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = candidate_artifact.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return None

    @staticmethod
    def _candidate_function_names(hypothesis: Any, candidate_artifact: dict[str, Any]) -> list[str]:
        explicit = [
            GraphEnrichmentStore._first_artifact_value(
                candidate_artifact,
                "functional_role",
                "function",
                "role",
                "product_role",
            )
        ]
        inferred: list[str] = []
        text = " ".join(
            str(item or "")
            for item in [
                getattr(hypothesis, "title", ""),
                getattr(hypothesis, "summary", ""),
                getattr(hypothesis, "application", ""),
            ]
        ).lower()
        phrase_map = {
            "coalescing aid": "coalescing aid",
            "coalescent": "coalescing aid",
            "plasticizer": "plasticizer",
            "solvent": "solvent",
            "surfactant": "surfactant",
        }
        for phrase, normalized in phrase_map.items():
            if phrase in text:
                inferred.append(normalized)
        deduped: "OrderedDict[str, None]" = OrderedDict()
        for item in [*explicit, *inferred]:
            text = str(item or "").strip()
            if text:
                deduped[text] = None
        return list(deduped.keys())

    @staticmethod
    def _candidate_endpoint_results(hypothesis: Any) -> list[Any]:
        results: "OrderedDict[str, Any]" = OrderedDict()
        for result in getattr(hypothesis, "evaluation_results", []) or []:
            if getattr(result, "criterion_name", None):
                results[result.criterion_name] = result
        assessment = getattr(hypothesis, "reflection_assessment", None)
        for result in getattr(assessment, "criterion_results", []) or []:
            if getattr(result, "criterion_name", None):
                results[result.criterion_name] = result
        return list(results.values())

    @staticmethod
    def _endpoint_category(endpoint_name: str) -> str:
        text = endpoint_name.lower()
        if any(token in text for token in ("tox", "ecotox", "fish", "hazard")):
            return "toxicity"
        if any(token in text for token in ("solubility", "logp", "vapor", "boiling")):
            return "physicochemical"
        if any(token in text for token in ("coales", "mfft", "tg", "film")):
            return "performance"
        return "general"

    def apply_edge_feedback(
        self,
        candidate_material: str,
        incumbent_material: str | None,
        application: str,
        volume: float | None = None,
        volume_unit: str | None = None,
        status: str | None = None,
        confidence: float | None = None,
        comment: str | None = None,
    ) -> int:
        """
        Updates edges in the knowledge graph matching candidate_material -> application
        and incumbent_material -> application with human feedback.
        Returns the number of edges updated.
        """
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        self._edges_path.mkdir(parents=True, exist_ok=True)

        candidate_slug = slugify(candidate_material)
        candidate_id = f"product:{candidate_slug}"
        
        incumbent_id = f"product:{slugify(incumbent_material)}" if incumbent_material else None
        application_id = f"application:{slugify(application)}"

        path = self._edges_path / "Product_USED_IN_Application.parquet"
        
        updated_count = 0
        now = now_iso()
        
        with FileLock(path):
            rows = read_rows(path, PRODUCT_APPLICATION_SCHEMA)
            for row in rows:
                is_candidate_match = row.get("product_id") == candidate_id and row.get("application_id") == application_id
                is_incumbent_match = incumbent_id and row.get("product_id") == incumbent_id and row.get("application_id") == application_id
                
                if is_candidate_match or is_incumbent_match:
                    if volume is not None:
                        row["volume_value"] = float(volume)
                        row["volume_year"] = datetime.now(timezone.utc).year
                    if volume_unit is not None:
                        row["volume_unit"] = volume_unit
                    if status is not None:
                        row["validation_status"] = status
                    if confidence is not None:
                        row["confidence"] = float(confidence)
                    if comment is not None:
                        row["supporting_quote"] = comment
                    row["updated_at"] = now
                    updated_count += 1
            
            if updated_count > 0:
                write_rows(path, rows, PRODUCT_APPLICATION_SCHEMA)
                
        return updated_count

    @staticmethod
    def _append_unique_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema, key: str) -> None:
        if not rows:
            return
        with FileLock(path):
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
        ("canonical_smiles", pa.string()),
        ("inchi_key", pa.string()),
        ("cas_number", pa.string()),
        ("product_family", pa.string()),
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
MATERIAL_FAMILY_NODE_SCHEMA = pa.schema(
    [
        ("material_family_id", pa.string()),
        ("name", pa.string()),
        ("canonical_name", pa.string()),
        ("normalized_name", pa.string()),
        ("family_type", pa.string()),
        ("aliases_json", pa.string()),
        ("alias_sources_json", pa.string()),
        ("description", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
MATERIAL_GRADE_NODE_SCHEMA = pa.schema(
    [
        ("material_grade_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("record_type", pa.string()),
        ("source_vendor", pa.string()),
        ("source_url", pa.string()),
        ("source_record_id", pa.string()),
        ("source_search_query", pa.string()),
        ("manufacturer_name", pa.string()),
        ("trade_name", pa.string()),
        ("grade_name", pa.string()),
        ("material_family_name", pa.string()),
        ("category_path_json", pa.string()),
        ("processing_methods_json", pa.string()),
        ("applications_json", pa.string()),
        ("property_table_json", pa.string()),
        ("raw_record_json", pa.string()),
        ("last_seen_at", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
MATERIAL_ALIAS_NODE_SCHEMA = pa.schema(
    [
        ("material_alias_id", pa.string()),
        ("alias_text", pa.string()),
        ("normalized_alias", pa.string()),
        ("canonical_node_id", pa.string()),
        ("canonical_node_label", pa.string()),
        ("alias_type", pa.string()),
        ("source_vendor", pa.string()),
        ("source_url", pa.string()),
        ("evidence_hash", pa.string()),
        ("confidence", pa.float64()),
        ("validation_status", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
CHEMISTRY_CLASS_NODE_SCHEMA = pa.schema(
    [
        ("chemistry_class_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
FUNCTION_NODE_SCHEMA = pa.schema(
    [
        ("function_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
BINDER_SYSTEM_NODE_SCHEMA = pa.schema(
    [
        ("binder_system_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
ENDPOINT_NODE_SCHEMA = pa.schema(
    [
        ("endpoint_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("endpoint_category", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
CRITICAL_TO_QUALITY_NODE_SCHEMA = pa.schema(
    [
        ("ctq_id", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("description", pa.string()),
        ("requirement_type", pa.string()),
        ("aliases_json", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)
NODE_SCHEMAS = {
    "Product": PRODUCT_NODE_SCHEMA,
    "Application": APPLICATION_NODE_SCHEMA,
    "Market": MARKET_NODE_SCHEMA,
    "Company": COMPANY_NODE_SCHEMA,
    "MaterialFamily": MATERIAL_FAMILY_NODE_SCHEMA,
    "MaterialGrade": MATERIAL_GRADE_NODE_SCHEMA,
    "MaterialAlias": MATERIAL_ALIAS_NODE_SCHEMA,
    "ChemistryClass": CHEMISTRY_CLASS_NODE_SCHEMA,
    "Function": FUNCTION_NODE_SCHEMA,
    "BinderSystem": BINDER_SYSTEM_NODE_SCHEMA,
    "Endpoint": ENDPOINT_NODE_SCHEMA,
    "CriticalToQuality": CRITICAL_TO_QUALITY_NODE_SCHEMA,
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
PRODUCT_CHEMISTRY_CLASS_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("product_id", pa.string()),
        ("chemistry_class_id", pa.string()),
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
PRODUCT_FUNCTION_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("product_id", pa.string()),
        ("function_id", pa.string()),
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
PRODUCT_BINDER_SYSTEM_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("product_id", pa.string()),
        ("binder_system_id", pa.string()),
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
PRODUCT_ENDPOINT_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("product_id", pa.string()),
        ("endpoint_id", pa.string()),
        ("value_text", pa.string()),
        ("value_numeric", pa.float64()),
        ("unit", pa.string()),
        ("normalized_score", pa.float64()),
        ("evidence_mode", pa.string()),
        ("tool_id", pa.string()),
        ("is_inferred", pa.bool_()),
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
PRODUCT_MATERIAL_GRADE_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("product_id", pa.string()),
        ("material_grade_id", pa.string()),
        ("relationship_role", pa.string()),
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
MATERIAL_GRADE_MATERIAL_FAMILY_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("material_grade_id", pa.string()),
        ("material_family_id", pa.string()),
        ("relationship_role", pa.string()),
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
COMPANY_MATERIAL_GRADE_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("company_id", pa.string()),
        ("material_grade_id", pa.string()),
        ("role", pa.string()),
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
MATERIAL_GRADE_ENDPOINT_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("material_grade_id", pa.string()),
        ("endpoint_id", pa.string()),
        ("value_text", pa.string()),
        ("value_numeric", pa.float64()),
        ("value_min", pa.float64()),
        ("value_max", pa.float64()),
        ("unit", pa.string()),
        ("condition_text", pa.string()),
        ("test_method", pa.string()),
        ("original_property_name", pa.string()),
        ("original_value_text", pa.string()),
        ("original_unit", pa.string()),
        ("normalized_score", pa.float64()),
        ("evidence_mode", pa.string()),
        ("tool_id", pa.string()),
        ("is_inferred", pa.bool_()),
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
APPLICATION_CTQ_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("application_id", pa.string()),
        ("ctq_id", pa.string()),
        ("market_id", pa.string()),
        ("requirement_role", pa.string()),
        ("requirement_text", pa.string()),
        ("property_requirements_json", pa.string()),
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
CTQ_ENDPOINT_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("ctq_id", pa.string()),
        ("endpoint_id", pa.string()),
        ("direction", pa.string()),
        ("default_threshold_value", pa.float64()),
        ("default_threshold_min", pa.float64()),
        ("default_threshold_max", pa.float64()),
        ("unit", pa.string()),
        ("condition_text", pa.string()),
        ("rationale", pa.string()),
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
