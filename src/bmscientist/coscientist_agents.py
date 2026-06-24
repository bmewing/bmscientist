from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from bmscientist.chunking import TextChunker
from bmscientist.classify import EvidenceClassifier
from bmscientist.config import AppConfig
from bmscientist.cost_tracking import CostTracker
from bmscientist.coscientist_models import (
    AssessmentMetric,
    CandidateEvaluationResult,
    CoScientistRunResult,
    CoScientistLoopResult,
    EvidenceCitation,
    EvaluationCriterion,
    EvolutionHypothesisSeed,
    GapShrinkageStatus,
    Hypothesis,
    HypothesisEvolutionOutput,
    HypothesisGenerationOutput,
    MetaReviewOutput,
    MetaReviewRound,
    MarketVolumeEstimateOutput,
    PriceMetric,
    ProximityConcept,
    ProximityMergePolicy,
    ProximityReviewOutput,
    ProximityRound,
    RankingAction,
    RankedHypothesis,
    RankingOutput,
    RankingRound,
    ReflectionAssessment,
    ReflectionReviewOutput,
    ReflectionSearchLimits,
    ResearchGoalDocument,
    ResearchPlanDraft,
    SynthesizedHypothesisSeed,
    UpdatedResearchPlan,
)
from bmscientist.coscientist_store import CoScientistStore
from bmscientist.extract import PageFetcher, extract_domain
from bmscientist.graph_market import GraphMarketEvidence
from bmscientist.llm import DeepSeekLLM
from bmscientist.manual_ingest import ManualEvidenceIngestor
from bmscientist.models import ChunkRecord, DiscoverySummary, PageContent, SearchResultItem
from bmscientist.price_cache import StructuredPriceCache
from bmscientist.prompt_library import PROMPTS
from bmscientist.retrieval import ExaPageRetriever, build_partial_page_from_search_result, compose_partial_text
from bmscientist.search import ExaSearchClient, deduplicate_search_results, default_search_options


if TYPE_CHECKING:
    from bmscientist.agent import DiscoveryAgent
    from bmscientist.embeddings import LocalEmbedder
    from bmscientist.store import LanceEvidenceStore


LOGGER = logging.getLogger(__name__)


class ProgressReporter(Protocol):
    def start(self, phase: str, message: str, total: int | None = None) -> None: ...

    def advance(self, phase: str, message: str, completed: int, total: int | None = None) -> None: ...

    def complete(
        self,
        phase: str,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None: ...

    def details(self, phase: str, lines: list[str]) -> None: ...


class NullProgressReporter:
    def start(self, phase: str, message: str, total: int | None = None) -> None:
        return None

    def advance(self, phase: str, message: str, completed: int, total: int | None = None) -> None:
        return None

    def complete(
        self,
        phase: str,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        return None

    def details(self, phase: str, lines: list[str]) -> None:
        return None


class LocalEvidenceRetriever:
    def __init__(self, store: LanceEvidenceStore, embedder: LocalEmbedder):
        self._store = store
        self._embedder = embedder
        self._lock = Lock()

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        with self._lock:
            vector = self._embedder.embed_query(query)
            return self._store.search_by_vector(vector, top_k=top_k)

    def search_many(self, queries: list[str], top_k_per_query: int = 5, max_results: int = 20) -> list[dict[str, Any]]:
        merged: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for query in queries:
            for row in self.search(query, top_k=top_k_per_query):
                row_id = row.get("id")
                if row_id and row_id not in merged:
                    merged[row_id] = row
                if len(merged) >= max_results:
                    return list(merged.values())
        return list(merged.values())

    def retrieve_for_goal(self, document: ResearchGoalDocument, max_results: int) -> list[dict[str, Any]]:
        queries = [document.raw_goal]
        queries.extend(document.material_scope[:3])
        queries.extend(document.application_scope[:3])
        if not self._requires_novel_candidates(document):
            queries.extend(document.target_incumbent_materials[:2])
            queries.extend(document.preferred_candidate_materials[:2])
        else:
            queries.extend(document.novelty_requirements[:2])
            queries.extend(document.known_candidate_exclusion_terms[:2])
        queries.extend(document.recycling_or_sustainability_angles[:2])
        queries.extend(self._goal_queries_from_contract(document))
        return self.search_many(queries, top_k_per_query=6, max_results=max_results)

    def retrieve_for_hypothesis(self, document: ResearchGoalDocument, hypothesis: Hypothesis, max_results: int = 16) -> list[dict[str, Any]]:
        queries = [
            document.raw_goal,
            hypothesis.title,
            f"{hypothesis.application or ''} {hypothesis.candidate_material or ''} {hypothesis.incumbent_material or ''}".strip(),
            f"{hypothesis.market_segment or ''} {hypothesis.candidate_material or ''} strategic fit".strip(),
            f"{hypothesis.application or ''} {hypothesis.incumbent_material or ''} price usd kg".strip(),
            f"{hypothesis.application or ''} {hypothesis.next_best_competitive_alternative or ''} price usd kg".strip(),
            f"{hypothesis.application or ''} {hypothesis.market_segment or ''} market size {' '.join(document.regions)}".strip(),
            f"{hypothesis.application or ''} replacement drivers regulatory sustainability".strip(),
            f"{hypothesis.application or ''} {hypothesis.candidate_material or ''} drop in replacement {hypothesis.incumbent_material or ''}".strip(),
        ]
        queries.extend(self._hypothesis_queries_from_contract(document, hypothesis))
        normalized_queries = [query for query in queries if query]
        return self.search_many(normalized_queries, top_k_per_query=5, max_results=max_results)

    @staticmethod
    def _goal_queries_from_contract(document: ResearchGoalDocument) -> list[str]:
        queries: list[str] = []
        schema = document.candidate_artifact_schema
        if schema.artifact_type:
            queries.append(schema.artifact_type)
        if schema.primary_identifier_field:
            queries.append(f"{document.raw_goal} {schema.primary_identifier_field}")
        if LocalEvidenceRetriever._requires_novel_candidates(document):
            queries.extend(document.novelty_requirements[:2])
            if schema.primary_identifier_field:
                queries.append(f"{document.raw_goal} novel {schema.primary_identifier_field}")
        queries.extend(document.search_strategy_notes[:3])
        for criterion in document.evaluation_criteria[:5]:
            queries.append(criterion.name)
            if criterion.description:
                queries.append(f"{document.raw_goal} {criterion.description}")
            queries.extend(criterion.suggested_search_queries[:2])
        return queries

    @staticmethod
    def _hypothesis_queries_from_contract(document: ResearchGoalDocument, hypothesis: Hypothesis) -> list[str]:
        queries: list[str] = []
        artifact = hypothesis.candidate_artifact or {}
        primary_field = document.candidate_artifact_schema.primary_identifier_field
        primary_value = artifact.get(primary_field)
        if primary_value:
            queries.append(str(primary_value))
            if LocalEvidenceRetriever._requires_novel_candidates(document):
                queries.append(f"{primary_value} known chemical")
                queries.append(f"{primary_value} PubChem")
        for key, value in artifact.items():
            text = str(value).strip()
            if text:
                queries.append(f"{key} {text}")
        for criterion in document.evaluation_criteria[:5]:
            queries.append(f"{hypothesis.title} {criterion.name}")
            if primary_value:
                queries.append(f"{primary_value} {criterion.name}")
            if criterion.description:
                queries.append(f"{hypothesis.title} {criterion.description}")
            queries.extend(criterion.suggested_search_queries[:1])
        queries.extend(hypothesis.unknowns[:3])
        if hypothesis.reflection_assessment is not None:
            queries.extend(hypothesis.reflection_assessment.evidence_gap_notes[:3])
        return queries

    @staticmethod
    def _requires_novel_candidates(document: ResearchGoalDocument) -> bool:
        return document.candidate_origin_policy in {"novel_candidates", "novel_analogs", "de_novo_design"}

    @staticmethod
    def citations_from_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[EvidenceCitation]:
        citations: list[EvidenceCitation] = []
        for row in rows[:limit]:
            citations.append(
                EvidenceCitation(
                    chunk_id=str(row.get("id", "")),
                    source_url=str(row.get("source_url", "")),
                    source_title=str(row.get("source_title", "")),
                    relevance_score=row.get("relevance_score"),
                    retrieved_at=row.get("retrieved_at"),
                )
            )
        return citations

    @staticmethod
    def is_stale(rows: list[dict[str, Any]], preferred_recency_days: int) -> bool:
        if not rows:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(days=preferred_recency_days)
        for row in rows:
            try:
                retrieved_at = datetime.fromisoformat(str(row.get("retrieved_at")))
            except (TypeError, ValueError):
                continue
            if retrieved_at >= cutoff:
                return False
        return True


class DiscoveryEvidenceTool:
    def __init__(
        self,
        source: DiscoveryAgent | AppConfig,
        llm: DeepSeekLLM | None = None,
        embedder: LocalEmbedder | None = None,
        store: LanceEvidenceStore | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self._legacy_discovery_agent = source if hasattr(source, "discover") else None
        self._config = source if isinstance(source, AppConfig) else None
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._search = ExaSearchClient(source, cost_tracker=cost_tracker) if isinstance(source, AppConfig) else None
        self._fetcher = PageFetcher(source) if isinstance(source, AppConfig) else None
        self._retriever = ExaPageRetriever(source, self._search, self._fetcher) if isinstance(source, AppConfig) else None
        self._classifier = EvidenceClassifier(llm) if isinstance(source, AppConfig) and llm else None
        self._chunker = TextChunker()
        self._write_lock = Lock()

    def run(self, query: str, limits: ReflectionSearchLimits):
        if self._legacy_discovery_agent is not None:
            return self._legacy_discovery_agent.discover(
                query=query,
                max_search_queries=1,
                results_per_query=limits.results_per_query,
                max_pages=limits.max_pages_per_search,
            )
        if not all([self._config, self._search, self._fetcher, self._classifier, self._embedder, self._store]):
            raise ValueError("DiscoveryEvidenceTool requires either a DiscoveryAgent or reflection tool dependencies.")

        run_id = str(uuid4())
        skipped_pages: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            search_response = self._search.search(
                query=query,
                num_results=limits.results_per_query,
                options=default_search_options(
                    self._config,
                    query,
                    search_type=self._config.exa_reflection_search_type,
                ),
            )
            search_results = search_response.results
        except Exception as exc:
            LOGGER.exception("Reflection search failed for query %s", query)
            errors.append(f"search:{query}:{exc}")
            search_results = []

        unique_results = deduplicate_search_results(search_results)
        fetched_pages = self._fetch_pages(query, unique_results[: limits.max_pages_per_search], skipped_pages)
        candidate_pages = self._filter_pages(query, fetched_pages, skipped_pages)
        chunk_records = self._classify_and_chunk(run_id, query, candidate_pages, skipped_pages, errors)
        vectors = self._embedder.embed_texts([record.chunk_text for record in chunk_records])
        embedded_records = [record.model_copy(update={"vector": vector}) for record, vector in zip(chunk_records, vectors, strict=False)]
        with self._write_lock:
            stored_chunks = self._store.add_chunks(embedded_records)

        return DiscoverySummary(
            run_id=run_id,
            original_query=query,
            total_search_queries=1,
            total_search_results=len(search_results),
            unique_urls=len(unique_results),
            fetched_pages=len(fetched_pages),
            relevant_pages=len(candidate_pages),
            stored_chunks=stored_chunks,
            opportunity_summary="Reflection evidence ingestion completed without broad discovery summarization.",
            notable_applications=sorted({record.application for record in embedded_records if record.application}),
            evidence_gaps=[item.get("reason", "unknown") for item in skipped_pages[:10]],
            recommended_next_steps=[],
            output_path="",
        )

    def _fetch_pages(
        self,
        query: str,
        results: list[SearchResultItem],
        skipped_pages: list[dict[str, Any]],
    ) -> list[PageContent]:
        fetched_pages: list[PageContent] = []
        assert self._fetcher is not None
        assert self._retriever is not None
        retrieval = self._retriever.retrieve_pages(query, results, max_pages=len(results), contents_query=query)
        skipped_pages.extend(retrieval.skipped)
        return retrieval.pages

    def _filter_pages(
        self,
        query: str,
        pages: list[PageContent],
        skipped_pages: list[dict[str, Any]],
    ) -> list[PageContent]:
        assert self._classifier is not None
        assert self._config is not None
        candidates: list[PageContent] = []
        for page in pages:
            is_partial = bool(page.metadata.get("is_partial_evidence"))
            min_characters = self._config.min_snippet_characters if is_partial else self._config.min_page_characters
            if len(page.text) < min_characters:
                skipped_pages.append({"url": str(page.url), "reason": "too_little_text"})
                continue
            heuristic_score = self._classifier.heuristic_relevance(query, page.text)
            metadata = {
                **page.metadata,
                "heuristic_relevance_score": heuristic_score,
                "retention_policy": "retain_reflection_search_text",
            }
            candidates.append(page.model_copy(update={"metadata": metadata}))
        return candidates

    def _classify_and_chunk(
        self,
        run_id: str,
        query: str,
        pages: list[PageContent],
        skipped_pages: list[dict[str, Any]],
        errors: list[str],
    ) -> list[ChunkRecord]:
        assert self._classifier is not None
        assert self._config is not None
        chunk_records: list[ChunkRecord] = []
        for page in pages:
            try:
                classification = self._classifier.classify(query, page)
            except Exception as exc:
                LOGGER.exception("Reflection classification failed for %s", page.url)
                errors.append(f"classify:{page.url}:{exc}")
                skipped_pages.append({"url": str(page.url), "reason": "classification_error", "error": str(exc)})
                continue
            for index, chunk in enumerate(self._chunker.chunk_text(page.text)):
                chunk_records.append(
                    ChunkRecord(
                        id=str(uuid5(NAMESPACE_URL, f"{run_id}::{page.url}::{index}")),
                        run_id=run_id,
                        original_query=query,
                        search_query=page.search_query,
                        source_title=page.title,
                        source_url=str(page.url),
                        source_domain=page.source_domain,
                        retrieved_at=page.fetched_at,
                        chunk_index=index,
                        chunk_text=chunk,
                        application=classification.application,
                        incumbent_material=classification.incumbent_material,
                        candidate_materials=classification.candidate_materials,
                        evidence_type=classification.evidence_type,
                        application_requirements=classification.application_requirements,
                        substitution_drivers=classification.substitution_drivers,
                        relevance_score=classification.relevance_score,
                        confidence_score=classification.confidence_score,
                        metadata={
                            "rationale": classification.rationale,
                            "supporting_quotes": classification.supporting_quotes,
                            "classification_relevant": classification.relevant,
                            "classification_relevance_score": classification.relevance_score,
                            "classification_confidence_score": classification.confidence_score,
                            "retained_for_reflection": True,
                            "retention_policy": page.metadata.get("retention_policy", "retain_reflection_search_text"),
                            "page_metadata": page.metadata,
                            "source": "coscientist_reflection",
                        },
                    )
                )
        return chunk_records

    @staticmethod
    def _build_partial_page_from_search_result(query: str, result: SearchResultItem, reason: str) -> PageContent | None:
        parts = compose_partial_text(result)
        partial_text = parts
        if len(partial_text) < 80:
            return None
        partial_page = build_partial_page_from_search_result(result, reason)
        if partial_page is None:
            return None
        return partial_page.model_copy(update={"search_query": query})


class ReflectionSearchPlanner:
    def plan(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        assessment: ReflectionAssessment,
        suggested_queries: list[str],
    ) -> list[str]:
        if document.evaluation_criteria:
            return self._plan_for_criteria(document, hypothesis, assessment, suggested_queries)
        needs = self._evidence_needs(assessment)
        queries: "OrderedDict[str, None]" = OrderedDict()
        region_text = " ".join(document.regions).strip()
        incumbent = hypothesis.incumbent_material or "incumbent material"
        candidate = hypothesis.candidate_material or "candidate material"
        nbca = hypothesis.next_best_competitive_alternative or assessment.nbca_material or "competitive alternative"
        application = hypothesis.application or hypothesis.product_type or "application"
        form = hypothesis.incumbent_form or hypothesis.candidate_form or hypothesis.product_type or ""
        process = hypothesis.conversion_process or ""
        segment = hypothesis.market_segment or ""

        templates_by_need = {
            "market": [
                f"{application} {segment} market size {region_text}",
                f"{application} packaging demand volume {region_text}",
            ],
            "pricing": [
                f"{incumbent} price usd kg {application} {region_text}",
                f"{nbca} price usd kg {application} {region_text}",
            ],
            "technical": [
                f"{candidate} {application} replacement for {incumbent} {form} {process}",
                f"{candidate} vs {incumbent} {application} requirements {form}",
            ],
            "drivers": [
                f"{application} {incumbent} regulatory sustainability recycled content {region_text}",
                f"{candidate} recycled content value {application} {region_text}",
            ],
            "strategic": [
                f"{candidate} drop in replacement {incumbent} {application} {process}",
                f"{application} {candidate} commercialization lead time {region_text}",
            ],
        }
        ordered_needs = [need for need in ("technical", "market", "pricing", "drivers", "strategic") if need in needs]
        if not ordered_needs:
            ordered_needs = ["technical", "market", "pricing"]
        for need in ordered_needs:
            for query in templates_by_need[need]:
                normalized = " ".join(query.split())
                if normalized:
                    queries[normalized] = None
        for query in suggested_queries:
            normalized = " ".join(query.split())
            if normalized:
                queries[normalized] = None
        return list(queries.keys())

    def _plan_for_criteria(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        assessment: ReflectionAssessment,
        suggested_queries: list[str],
    ) -> list[str]:
        queries: "OrderedDict[str, None]" = OrderedDict()
        region_text = " ".join(document.regions).strip()
        artifact = hypothesis.candidate_artifact or {}
        primary_field = document.candidate_artifact_schema.primary_identifier_field
        primary_value = str(artifact.get(primary_field) or "").strip()
        unresolved = self._criteria_needing_evidence(document.evaluation_criteria, assessment)
        if not unresolved:
            unresolved = document.evaluation_criteria[:3]
        for criterion in unresolved:
            for query in criterion.suggested_search_queries:
                normalized = " ".join(query.split())
                if normalized:
                    queries[normalized] = None
            if primary_value:
                queries[f"{primary_value} {criterion.name}".strip()] = None
                if criterion.description:
                    queries[f"{primary_value} {criterion.description}".strip()] = None
                if document.candidate_origin_policy in {"novel_candidates", "novel_analogs", "de_novo_design"}:
                    queries[f"{primary_value} known chemical".strip()] = None
                    queries[f"{primary_value} PubChem".strip()] = None
            elif hypothesis.title:
                queries[f"{hypothesis.title} {criterion.name}".strip()] = None
            if region_text:
                queries[f"{hypothesis.title} {criterion.name} {region_text}".strip()] = None
            for field_name in criterion.required_candidate_fields[:2]:
                field_value = artifact.get(field_name)
                if field_value:
                    queries[f"{field_value} {criterion.name}".strip()] = None
        for query in suggested_queries:
            normalized = " ".join(query.split())
            if normalized:
                queries[normalized] = None
        return [query for query in queries.keys() if query]

    @staticmethod
    def _criteria_needing_evidence(
        criteria: list[EvaluationCriterion],
        assessment: ReflectionAssessment,
    ) -> list[EvaluationCriterion]:
        results_by_name = {
            result.criterion_name: result
            for result in assessment.criterion_results
            if result.criterion_name
        }
        unresolved: list[EvaluationCriterion] = []
        for criterion in criteria:
            result = results_by_name.get(criterion.name)
            if result is None or result.normalized_score is None or result.confidence < 0.35:
                unresolved.append(criterion)
        return unresolved

    @staticmethod
    def _evidence_needs(assessment: ReflectionAssessment) -> set[str]:
        needs: set[str] = set()
        if assessment.market_size_score.value is None or assessment.market_size_score.confidence < 0.35:
            needs.add("market")
        if assessment.incumbent_price_usd_per_kg.value is None or assessment.nbca_price_usd_per_kg.value is None:
            needs.add("pricing")
        if (
            assessment.replacement_fit_score.value is None
            or assessment.activation_ease_score.value is None
            or assessment.technical_success_probability.value is None
        ):
            needs.add("technical")
        if assessment.replacement_driver_strength_score.value is None or assessment.replacement_driver_strength_score.confidence < 0.35:
            needs.add("drivers")
        if assessment.strategic_fit_score.value is None or assessment.commercial_success_probability.value is None:
            needs.add("strategic")
        return needs


class ResearchPlanningAgent:
    _NOVEL_DESIGN_CUES = (
        "brand-new",
        "brand new",
        "never before seen",
        "never-before-seen",
        "de novo",
        "de-novo",
        "invent",
        "novel structure",
        "novel structures",
        "design molecule",
        "design molecules",
        "generate smiles",
        "smiles strings",
        "smiles string",
        "not existing materials",
        "not existing material",
        "not looking for substitution",
        "not looking for substitutions",
        "not substitutions",
    )
    _NOVEL_ANALOG_CUES = ("analog", "analogs", "analogue", "analogues")
    _STRUCTURE_CUES = ("smiles", "molecule", "molecules", "structure", "structures")
    _SUBSTITUTION_CUES = (
        "replacement",
        "replace",
        "substitute",
        "substitution",
        "drop-in",
        "drop in",
        "alternative to",
        "commercially available",
        "supplier",
        "qualified material",
    )

    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    @staticmethod
    def _default_generated_count(target_hypotheses_final: int) -> int:
        return min(max(target_hypotheses_final * 2, target_hypotheses_final), target_hypotheses_final * 3)

    def create_research_goal(
        self,
        research_id: str,
        raw_goal: str,
        target_hypotheses_final: int,
        regions: list[str] | None,
        strategic_fit_notes: str | None,
        preferred_evidence_recency_days: int,
        reflection_search_limits: ReflectionSearchLimits,
        proximity_merge_policy: ProximityMergePolicy | None = None,
    ) -> ResearchGoalDocument:
        system_prompt = PROMPTS.render("research_planning_agent", "create_research_goal.system")
        user_prompt = PROMPTS.render(
            "research_planning_agent",
            "create_research_goal.user",
            raw_goal=raw_goal,
            target_hypotheses_final=target_hypotheses_final,
            regions=regions or [],
            strategic_fit_notes=strategic_fit_notes or "",
        )
        draft = self._llm.complete_json(ResearchPlanDraft, system_prompt, user_prompt)
        draft = self._stabilize_plan_draft(raw_goal, draft)
        return ResearchGoalDocument(
            research_id=research_id,
            raw_goal=raw_goal,
            target_hypotheses_final=target_hypotheses_final,
            target_hypotheses_generated=self._default_generated_count(target_hypotheses_final),
            research_mode=draft.research_mode,
            regions=regions or [],
            strategic_fit_criteria=draft.strategic_fit_criteria,
            target_incumbent_materials=draft.target_incumbent_materials,
            preferred_candidate_materials=draft.preferred_candidate_materials,
            candidate_material_preferences=draft.candidate_material_preferences,
            candidate_origin_policy=draft.candidate_origin_policy,
            novelty_requirements=draft.novelty_requirements,
            known_candidate_exclusion_terms=draft.known_candidate_exclusion_terms,
            novelty_check_policy=draft.novelty_check_policy,
            recycling_or_sustainability_angles=draft.recycling_or_sustainability_angles,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            reflection_search_limits=reflection_search_limits,
            material_scope=draft.material_scope,
            application_scope=draft.application_scope,
            opportunity_modes=draft.opportunity_modes,
            opportunity_speed_horizon_months=draft.opportunity_speed_horizon_months,
            commercialization_constraints=draft.commercialization_constraints,
            ranking_weights=draft.ranking_weights,
            success_definition=draft.success_definition,
            candidate_artifact_schema=draft.candidate_artifact_schema,
            evaluation_criteria=draft.evaluation_criteria,
            reflection_guidance=draft.reflection_guidance,
            tool_requests=draft.tool_requests,
            search_strategy_notes=draft.search_strategy_notes,
            strategic_fit_notes=strategic_fit_notes,
            proximity_merge_policy=proximity_merge_policy or ProximityMergePolicy(),
        )

    def update_research_goal(
        self,
        document: ResearchGoalDocument,
        feedback: str,
    ) -> ResearchGoalDocument:
        system_prompt = PROMPTS.render("research_planning_agent", "update_research_goal.system")
        user_prompt = PROMPTS.render(
            "research_planning_agent",
            "update_research_goal.user",
            current_goal_json=document.model_dump_json(indent=2),
            feedback=feedback,
        )
        updated = self._llm.complete_json(UpdatedResearchPlan, system_prompt, user_prompt)
        updated = self._stabilize_updated_plan(updated.raw_goal or document.raw_goal, updated, feedback)
        return document.model_copy(
            update={
                "raw_goal": updated.raw_goal,
                "research_mode": updated.research_mode,
                "regions": updated.regions,
                "strategic_fit_criteria": updated.strategic_fit_criteria,
                "target_incumbent_materials": updated.target_incumbent_materials,
                "preferred_candidate_materials": updated.preferred_candidate_materials,
                "candidate_material_preferences": updated.candidate_material_preferences,
                "candidate_origin_policy": updated.candidate_origin_policy,
                "novelty_requirements": updated.novelty_requirements,
                "known_candidate_exclusion_terms": updated.known_candidate_exclusion_terms,
                "novelty_check_policy": updated.novelty_check_policy,
                "recycling_or_sustainability_angles": updated.recycling_or_sustainability_angles,
                "material_scope": updated.material_scope,
                "application_scope": updated.application_scope,
                "opportunity_modes": updated.opportunity_modes,
                "opportunity_speed_horizon_months": updated.opportunity_speed_horizon_months,
                "commercialization_constraints": updated.commercialization_constraints,
                "ranking_weights": updated.ranking_weights,
                "success_definition": updated.success_definition,
                "candidate_artifact_schema": updated.candidate_artifact_schema,
                "evaluation_criteria": updated.evaluation_criteria,
                "reflection_guidance": updated.reflection_guidance,
                "tool_requests": updated.tool_requests,
                "search_strategy_notes": updated.search_strategy_notes,
                "strategic_fit_notes": updated.strategic_fit_notes,
            }
        )

    @classmethod
    def _stabilize_plan_draft(cls, raw_goal: str, draft: ResearchPlanDraft) -> ResearchPlanDraft:
        origin_policy = cls._resolved_candidate_origin_policy(
            raw_goal=raw_goal,
            research_mode=draft.research_mode,
            primary_identifier_field=draft.candidate_artifact_schema.primary_identifier_field,
            current_policy=draft.candidate_origin_policy,
        )
        novelty_check_policy = cls._resolved_novelty_check_policy(
            primary_identifier_field=draft.candidate_artifact_schema.primary_identifier_field,
            candidate_origin_policy=origin_policy,
            current_policy=draft.novelty_check_policy,
        )
        candidate_artifact_schema = cls._stabilize_candidate_artifact_schema(
            draft.candidate_artifact_schema,
            origin_policy,
        )
        novelty_requirements = cls._resolved_novelty_requirements(
            current_requirements=draft.novelty_requirements,
            candidate_origin_policy=origin_policy,
            primary_identifier_field=candidate_artifact_schema.primary_identifier_field,
        )
        return draft.model_copy(
            update={
                "candidate_origin_policy": origin_policy,
                "novelty_check_policy": novelty_check_policy,
                "novelty_requirements": novelty_requirements,
                "candidate_artifact_schema": candidate_artifact_schema,
            }
        )

    @classmethod
    def _stabilize_updated_plan(
        cls,
        raw_goal: str,
        updated: UpdatedResearchPlan,
        feedback: str,
    ) -> UpdatedResearchPlan:
        combined_goal = " ".join(part for part in [raw_goal, feedback] if part).strip()
        origin_policy = cls._resolved_candidate_origin_policy(
            raw_goal=combined_goal,
            research_mode=updated.research_mode,
            primary_identifier_field=updated.candidate_artifact_schema.primary_identifier_field,
            current_policy=updated.candidate_origin_policy,
        )
        novelty_check_policy = cls._resolved_novelty_check_policy(
            primary_identifier_field=updated.candidate_artifact_schema.primary_identifier_field,
            candidate_origin_policy=origin_policy,
            current_policy=updated.novelty_check_policy,
        )
        candidate_artifact_schema = cls._stabilize_candidate_artifact_schema(
            updated.candidate_artifact_schema,
            origin_policy,
        )
        novelty_requirements = cls._resolved_novelty_requirements(
            current_requirements=updated.novelty_requirements,
            candidate_origin_policy=origin_policy,
            primary_identifier_field=candidate_artifact_schema.primary_identifier_field,
        )
        return updated.model_copy(
            update={
                "candidate_origin_policy": origin_policy,
                "novelty_check_policy": novelty_check_policy,
                "novelty_requirements": novelty_requirements,
                "candidate_artifact_schema": candidate_artifact_schema,
            }
        )

    @classmethod
    def _resolved_candidate_origin_policy(
        cls,
        raw_goal: str,
        research_mode: str,
        primary_identifier_field: str,
        current_policy: str,
    ) -> str:
        if current_policy and current_policy != "unspecified":
            return current_policy
        goal_text = str(raw_goal or "").lower()
        has_novel_cue = any(cue in goal_text for cue in cls._NOVEL_DESIGN_CUES)
        has_structure_cue = any(cue in goal_text for cue in cls._STRUCTURE_CUES)
        has_substitution_cue = any(cue in goal_text for cue in cls._SUBSTITUTION_CUES)
        has_analog_cue = any(cue in goal_text for cue in cls._NOVEL_ANALOG_CUES)
        if has_novel_cue or (has_structure_cue and not has_substitution_cue):
            return "novel_analogs" if has_analog_cue else "de_novo_design"
        if research_mode == "materials_opportunity" or has_substitution_cue:
            return "known_candidates"
        if research_mode == "candidate_design" and primary_identifier_field.strip().lower() == "smiles":
            return "unspecified"
        return "unspecified"

    @staticmethod
    def _resolved_novelty_check_policy(
        primary_identifier_field: str,
        candidate_origin_policy: str,
        current_policy: str,
    ) -> str:
        if current_policy and current_policy != "none":
            return current_policy
        if candidate_origin_policy not in {"novel_candidates", "novel_analogs", "de_novo_design"}:
            return "none"
        if primary_identifier_field.strip().lower() == "smiles":
            return "identifier_lookup"
        return "name_only"

    @staticmethod
    def _stabilize_candidate_artifact_schema(schema, candidate_origin_policy: str):
        required_fields = list(schema.required_fields)
        primary_field = schema.primary_identifier_field.strip()
        if (
            candidate_origin_policy in {"novel_candidates", "novel_analogs", "de_novo_design"}
            and primary_field
            and primary_field not in required_fields
        ):
            required_fields.append(primary_field)
        return schema.model_copy(update={"required_fields": required_fields})

    @staticmethod
    def _resolved_novelty_requirements(
        current_requirements: list[str],
        candidate_origin_policy: str,
        primary_identifier_field: str,
    ) -> list[str]:
        requirements = list(current_requirements)
        if candidate_origin_policy not in {"novel_candidates", "novel_analogs", "de_novo_design"}:
            return requirements
        defaults = [
            "Do not return existing commercial materials or direct substitution recommendations as final candidates.",
            "Use known materials as constraints, benchmarks, or exclusions rather than as the final answer.",
        ]
        if primary_identifier_field.strip().lower() == "smiles":
            defaults.append("Provide explicit SMILES strings for each candidate.")
        for item in defaults:
            if item not in requirements:
                requirements.append(item)
        return requirements


class GenerationAgent:
    _minimum_batch_size = 5

    def __init__(self, llm: DeepSeekLLM, retriever: LocalEvidenceRetriever, graph_evidence: GraphMarketEvidence | None = None):
        self._llm = llm
        self._retriever = retriever
        self._graph_evidence = graph_evidence

    @property
    def batch_size(self) -> int:
        return self._minimum_batch_size

    def batch_size_for(self, requested_hypotheses: int) -> int:
        requested = max(0, requested_hypotheses)
        if requested < 20:
            return requested
        return max(self._minimum_batch_size, (requested + 3) // 4)

    def generate(
        self,
        document: ResearchGoalDocument,
        on_progress: Callable[[int, int], None] | None = None,
        on_batch: Callable[[list[Hypothesis]], None] | None = None,
        avoid_hypotheses: list[Hypothesis] | None = None,
    ) -> list[Hypothesis]:
        evidence_rows = self._retriever.retrieve_for_goal(
            document,
            max_results=max(document.target_hypotheses_generated * 4, 12),
        )
        if self._graph_evidence:
            try:
                graph_rows = self._graph_evidence.build_evidence_rows_for_goal(document)
                evidence_rows.extend(graph_rows)
            except Exception:
                LOGGER.exception("Offline graph market evidence unavailable during initial generation")
        evidence_payload = [
            {
                "chunk_id": row.get("id"),
                "application": row.get("application"),
                "incumbent_material": row.get("incumbent_material"),
                "candidate_materials": row.get("candidate_materials"),
                "application_requirements": row.get("application_requirements"),
                "substitution_drivers": row.get("substitution_drivers"),
                "metadata": row.get("metadata", {}),
                "source_url": row.get("source_url"),
                "source_title": row.get("source_title"),
                "relevance_score": row.get("relevance_score"),
                "excerpt": str(row.get("chunk_text", ""))[:500],
            }
            for row in evidence_rows[:40]
        ]
        return self._generate_in_batches(
            document=document,
            evidence_payload=evidence_payload,
            limit=document.target_hypotheses_generated,
            generation_source="initial",
            round_index=0,
            system_section="generate.system",
            user_section="generate.user",
            progress_total=document.target_hypotheses_generated,
            on_progress=on_progress,
            on_batch=on_batch,
            avoid_hypotheses=avoid_hypotheses,
        )

    def generate_from_meta_review(
        self,
        document: ResearchGoalDocument,
        meta_review_round: MetaReviewRound,
        target_count: int,
        round_index: int,
        on_progress: Callable[[int, int], None] | None = None,
        on_batch: Callable[[list[Hypothesis]], None] | None = None,
        avoid_hypotheses: list[Hypothesis] | None = None,
    ) -> list[Hypothesis]:
        if target_count <= 0:
            return []
        evidence_rows = self._retriever.retrieve_for_goal(document, max_results=max(target_count * 5, 12))
        if self._graph_evidence:
            try:
                graph_rows = self._graph_evidence.build_evidence_rows_for_goal(document)
                evidence_rows.extend(graph_rows)
            except Exception:
                LOGGER.exception("Offline graph market evidence unavailable during regeneration")
        evidence_payload = [
            {
                "chunk_id": row.get("id"),
                "application": row.get("application"),
                "incumbent_material": row.get("incumbent_material"),
                "candidate_materials": row.get("candidate_materials"),
                "metadata": row.get("metadata", {}),
                "source_url": row.get("source_url"),
                "source_title": row.get("source_title"),
                "excerpt": str(row.get("chunk_text", ""))[:500],
            }
            for row in evidence_rows[:35]
        ]
        return self._generate_in_batches(
            document=document,
            evidence_payload=evidence_payload,
            limit=target_count,
            generation_source="regenerated",
            round_index=round_index,
            system_section="generate_from_meta_review.system",
            user_section="generate_from_meta_review.user",
            progress_total=target_count,
            on_progress=on_progress,
            generation_guidance_json=json.dumps(meta_review_round.generation_guidance, indent=2),
            whitespace_gaps_json=json.dumps(meta_review_round.whitespace_gaps, indent=2),
            on_batch=on_batch,
            avoid_hypotheses=avoid_hypotheses,
        )

    def _generate_in_batches(
        self,
        document: ResearchGoalDocument,
        evidence_payload: list[dict[str, Any]],
        limit: int,
        generation_source: str,
        round_index: int,
        system_section: str,
        user_section: str,
        progress_total: int,
        on_progress: Callable[[int, int], None] | None = None,
        on_batch: Callable[[list[Hypothesis]], None] | None = None,
        avoid_hypotheses: list[Hypothesis] | None = None,
        **extra_context: Any,
    ) -> list[Hypothesis]:
        hypotheses: "OrderedDict[str, Hypothesis]" = OrderedDict()
        avoided = list(avoid_hypotheses or [])
        batch_size = max(1, min(self.batch_size_for(limit), limit))
        max_attempts = max(2, ((limit + batch_size - 1) // batch_size) * 2)
        attempts_without_progress = 0

        while len(hypotheses) < limit and attempts_without_progress < max_attempts:
            remaining = limit - len(hypotheses)
            batch_target = min(batch_size, remaining)
            system_prompt = PROMPTS.render("generation_agent", system_section)
            user_prompt = PROMPTS.render(
                "generation_agent",
                user_section,
                research_goal=document.raw_goal,
                document_json=document.model_dump_json(indent=2),
                evidence_payload_json=json.dumps(evidence_payload, indent=2),
                target_hypotheses_generated=batch_target,
                target_count=batch_target,
                existing_hypotheses_json=json.dumps(
                    self._existing_hypothesis_prompt_payload(document, list(hypotheses.values())),
                    indent=2,
                ),
                avoided_hypotheses_json=json.dumps(
                    self._avoided_hypothesis_prompt_payload(document, avoided),
                    indent=2,
                ),
                **extra_context,
            )
            output = self._llm.complete_json(HypothesisGenerationOutput, system_prompt, user_prompt)
            batch_hypotheses = self._seeds_to_hypotheses(
                document=document,
                seeds=output.hypotheses,
                limit=batch_target,
                generation_source=generation_source,
                round_index=round_index,
            )
            before_count = len(hypotheses)
            new_hypotheses: list[Hypothesis] = []
            for hypothesis in batch_hypotheses:
                hypothesis = self._prepare_hypothesis(document, hypothesis)
                if hypothesis is None:
                    continue
                if hypothesis.hypothesis_id in hypotheses:
                    continue
                if self._duplicates_existing_opportunity(document, hypothesis, hypotheses.values()):
                    continue
                if self._duplicates_existing_opportunity(document, hypothesis, avoided):
                    continue
                hypotheses[hypothesis.hypothesis_id] = hypothesis
                new_hypotheses.append(hypothesis)

            if len(hypotheses) == before_count:
                attempts_without_progress += 1
                continue

            attempts_without_progress = 0
            if on_batch is not None and new_hypotheses:
                on_batch(new_hypotheses)
            if on_progress is not None and len(hypotheses) < progress_total:
                on_progress(len(hypotheses), progress_total)

        return list(hypotheses.values())[:limit]

    @classmethod
    def _existing_hypothesis_prompt_payload(
        cls,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
    ) -> list[str]:
        return [cls._hypothesis_duplicate_signature(document, hypothesis) for hypothesis in hypotheses]

    @classmethod
    def _avoided_hypothesis_prompt_payload(
        cls,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
    ) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        seen_signatures: set[str] = set()
        for hypothesis in hypotheses:
            signature = cls._hypothesis_duplicate_signature(document, hypothesis)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            payload.append(
                {
                    "signature": signature,
                    "reason": cls._compact_prompt_text(
                        hypothesis.user_feedback_comment
                        or hypothesis.ranking_rationale
                        or hypothesis.retired_reason
                    ),
                }
            )
        return payload

    @staticmethod
    def _duplicates_existing_opportunity(
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        existing_hypotheses: Any,
    ) -> bool:
        return any(
            GenerationAgent._same_primary_artifact(document, existing, hypothesis)
            or ProximityCheckAgent._should_cluster(document, existing, hypothesis)
            for existing in existing_hypotheses
        )

    @classmethod
    def _hypothesis_duplicate_signature(
        cls,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
    ) -> str:
        candidate = cls._compact_prompt_text(hypothesis.candidate_material)
        if not candidate:
            candidate = cls._compact_prompt_text(cls._primary_artifact_identifier(document, hypothesis))
        incumbent = cls._compact_prompt_text(hypothesis.incumbent_material)
        artifact_identifier = cls._compact_prompt_text(cls._primary_artifact_identifier(document, hypothesis))
        parts = [
            cls._compact_prompt_text(hypothesis.title),
            f"app={cls._compact_prompt_text(hypothesis.application or hypothesis.product_type)}",
            f"market={cls._compact_prompt_text(hypothesis.market_segment or hypothesis.buyer_type)}",
        ]
        if candidate and incumbent:
            parts.append(f"{candidate} -> {incumbent}")
        elif candidate:
            parts.append(f"candidate={candidate}")
        elif incumbent:
            parts.append(f"incumbent={incumbent}")
        if artifact_identifier and artifact_identifier not in {candidate, cls._compact_prompt_text(hypothesis.title)}:
            parts.append(f"artifact={artifact_identifier}")
        compact_parts = [part for part in parts if part and not part.endswith("=")]
        return " | ".join(compact_parts) or hypothesis.hypothesis_id

    @staticmethod
    def _primary_artifact_identifier(document: ResearchGoalDocument, hypothesis: Hypothesis) -> str:
        artifact = hypothesis.candidate_artifact or {}
        primary_field = document.candidate_artifact_schema.primary_identifier_field
        if primary_field:
            value = artifact.get(primary_field)
            if value not in (None, "", []):
                return str(value)
        for fallback_field in ("name_or_label", "candidate_material", "application"):
            value = artifact.get(fallback_field)
            if value not in (None, "", []):
                return str(value)
        return ""

    @staticmethod
    def _compact_prompt_text(value: Any) -> str:
        if value in (None, "", []):
            return ""
        return " ".join(str(value).split())

    @classmethod
    def _prepare_hypothesis(
        cls,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
    ) -> Hypothesis | None:
        artifact = dict(hypothesis.candidate_artifact or {})
        primary_field = document.candidate_artifact_schema.primary_identifier_field.strip()
        if primary_field:
            primary_value = artifact.get(primary_field)
            if primary_field.lower() == "smiles":
                normalized_identifier = cls._normalize_smiles_identifier(primary_value)
                if normalized_identifier:
                    artifact[primary_field] = normalized_identifier
            elif primary_value not in (None, "", []):
                artifact[primary_field] = cls._compact_prompt_text(primary_value)
        missing_fields = [
            field
            for field in document.candidate_artifact_schema.required_fields
            if artifact.get(field) in (None, "", [])
        ]
        if missing_fields:
            return None
        prepared = hypothesis.model_copy(update={"candidate_artifact": artifact})
        if cls._violates_candidate_origin_policy(document, prepared):
            if cls._requires_novel_candidates(document):
                artifact["novelty_check_status"] = "failed_known_substance_check"
            return None
        if cls._requires_novel_candidates(document):
            artifact["novelty_check_status"] = "passed_exact_identifier_check"
        return prepared.model_copy(update={"candidate_artifact": artifact})

    @staticmethod
    def _requires_novel_candidates(document: ResearchGoalDocument) -> bool:
        return document.candidate_origin_policy in {"novel_candidates", "novel_analogs", "de_novo_design"}

    @classmethod
    def _violates_candidate_origin_policy(
        cls,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
    ) -> bool:
        if not cls._requires_novel_candidates(document):
            return False
        exclusion_terms = [cls._normalize_text_key(item) for item in document.known_candidate_exclusion_terms if item]
        if not exclusion_terms:
            return False
        candidate_texts = [
            hypothesis.candidate_material,
            hypothesis.title,
            (hypothesis.candidate_artifact or {}).get("name_or_label"),
        ]
        for value in candidate_texts:
            normalized_value = cls._normalize_text_key(value)
            if normalized_value and any(
                normalized_value == exclusion or normalized_value in exclusion or exclusion in normalized_value
                for exclusion in exclusion_terms
            ):
                return True
        primary_key = cls._artifact_identifier_key(document, hypothesis)
        if not primary_key:
            return False
        if document.candidate_artifact_schema.primary_identifier_field.strip().lower() == "smiles":
            normalized_exclusions = {cls._normalize_smiles_identifier(item) for item in document.known_candidate_exclusion_terms if item}
        else:
            normalized_exclusions = {cls._normalize_text_key(item) for item in document.known_candidate_exclusion_terms if item}
        return primary_key in normalized_exclusions

    @classmethod
    def _same_primary_artifact(
        cls,
        document: ResearchGoalDocument,
        left: Hypothesis,
        right: Hypothesis,
    ) -> bool:
        primary_field = document.candidate_artifact_schema.primary_identifier_field.strip().lower()
        if not primary_field or primary_field == "candidate_material":
            return False
        left_key = cls._artifact_identifier_key(document, left)
        right_key = cls._artifact_identifier_key(document, right)
        return bool(left_key and right_key and left_key == right_key)

    @classmethod
    def _artifact_identifier_key(
        cls,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
    ) -> str:
        value = cls._primary_artifact_identifier(document, hypothesis)
        if not value:
            return ""
        if document.candidate_artifact_schema.primary_identifier_field.strip().lower() == "smiles":
            return cls._normalize_smiles_identifier(value)
        return cls._normalize_text_key(value)

    @staticmethod
    def _normalize_text_key(value: Any) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        return " ".join(text.split())

    @staticmethod
    def _normalize_smiles_identifier(value: Any) -> str:
        text = "".join(str(value or "").split())
        if not text:
            return ""
        try:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                return ""
            return Chem.MolToSmiles(molecule, canonical=True)
        except Exception:
            return text

    @staticmethod
    def _seeds_to_hypotheses(
        document: ResearchGoalDocument,
        seeds: list[Any],
        limit: int,
        generation_source: str,
        round_index: int,
    ) -> list[Hypothesis]:
        hypotheses: "OrderedDict[str, Hypothesis]" = OrderedDict()
        for seed in seeds:
            primary_field = document.candidate_artifact_schema.primary_identifier_field
            primary_identifier = str((seed.candidate_artifact or {}).get(primary_field) or "").strip()
            hypothesis_id = str(
                uuid5(
                    NAMESPACE_URL,
                    (
                        f"{document.research_id}::{generation_source}::{round_index}::"
                        f"{seed.title}::{seed.application or ''}::{seed.candidate_material or ''}::{primary_identifier}"
                    ),
                )
            )
            if hypothesis_id in hypotheses:
                continue
            candidate_artifact = dict(seed.candidate_artifact or {})
            if not candidate_artifact:
                candidate_artifact = GenerationAgent._fallback_candidate_artifact(document, seed)
            hypotheses[hypothesis_id] = Hypothesis(
                hypothesis_id=hypothesis_id,
                research_id=document.research_id,
                status="generated",
                title=seed.title,
                summary=seed.summary,
                application=seed.application,
                market_segment=seed.market_segment,
                region_scope=document.regions,
                candidate_material=seed.candidate_material,
                incumbent_material=seed.incumbent_material,
                next_best_competitive_alternative=seed.next_best_competitive_alternative,
                incumbent_form=seed.incumbent_form,
                candidate_form=seed.candidate_form,
                conversion_process=seed.conversion_process,
                product_type=seed.product_type,
                buyer_type=seed.buyer_type,
                application_requirements=seed.application_requirements,
                substitution_drivers=seed.substitution_drivers,
                strategic_rationale=seed.strategic_rationale,
                supporting_chunk_ids=seed.supporting_chunk_ids,
                supporting_urls=seed.supporting_urls,
                assumptions=seed.assumptions,
                unknowns=seed.unknowns,
                candidate_artifact=candidate_artifact,
                evaluation_results=seed.evaluation_results,
                generation_confidence=seed.generation_confidence,
                round_index=round_index,
                generation_source=generation_source,
            )
            if len(hypotheses) >= limit:
                break
        return list(hypotheses.values())

    @staticmethod
    def _fallback_candidate_artifact(document: ResearchGoalDocument, seed: Any) -> dict[str, Any]:
        artifact = {
            "candidate_material": seed.candidate_material,
            "incumbent_material": seed.incumbent_material,
            "application": seed.application,
        }
        primary_field = document.candidate_artifact_schema.primary_identifier_field
        if primary_field and primary_field not in artifact and primary_field.strip().lower() != "smiles":
            if getattr(seed, "candidate_material", None):
                artifact[primary_field] = seed.candidate_material
            elif getattr(seed, "title", None):
                artifact[primary_field] = seed.title
        return {key: value for key, value in artifact.items() if value not in (None, "", [])}


class RankingAgent:
    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def rank(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        round_index: int,
        target_final_count: int,
        evolve_top_k: int,
    ) -> tuple[RankingRound, list[Hypothesis]]:
        candidates = [hypothesis for hypothesis in hypotheses if hypothesis.status == "reflected"]
        if not candidates:
            ranking_round = RankingRound(
                ranking_round_id=str(uuid4()),
                research_id=document.research_id,
                round_index=round_index,
                candidate_count=0,
                target_final_count=target_final_count,
            )
            return ranking_round, []

        heuristic_rankings = self._heuristic_rankings(candidates)
        try:
            output = self._llm_rank(document, candidates, heuristic_rankings, target_final_count)
        except Exception as exc:
            LOGGER.warning(
                "Ranking LLM failed (%s); falling back to deterministic reflected-score ranking",
                exc,
            )
            output = RankingOutput(
                rankings=heuristic_rankings,
                best_patterns=["High aggregate reflected assessment scores."],
                worst_patterns=["Weak or missing reflected assessment scores."],
            )

        rankings = self._merge_rankings(candidates, heuristic_rankings, output.rankings)
        rankings.sort(key=lambda item: (-item.score, item.rank or 10_000, item.hypothesis_id))
        normalized_rankings = [
            item.model_copy(
                update={
                    "rank": index + 1,
                    "recommended_action": self._normalized_action(index, item, target_final_count, evolve_top_k),
                }
            )
            for index, item in enumerate(rankings)
        ]
        candidates_by_id = {h.hypothesis_id: h for h in candidates}
        promoted_ids = [item.hypothesis_id for item in normalized_rankings[:target_final_count]]
        
        evolved_parent_ids = []
        for item in normalized_rankings:
            hyp = candidates_by_id.get(item.hypothesis_id)
            is_accepted = hyp and hyp.user_feedback_status == "accepted"
            is_high_score = item.score >= 0.5
            
            is_normal_evolve = (item.recommended_action in {"advance", "evolve"} and len(evolved_parent_ids) < evolve_top_k)
            if is_normal_evolve or (is_accepted and is_high_score):
                if item.hypothesis_id not in evolved_parent_ids:
                    evolved_parent_ids.append(item.hypothesis_id)
                    
        rejected_ids = [item.hypothesis_id for item in normalized_rankings if item.recommended_action == "reject"]
        scores = [item.score for item in normalized_rankings]
        ranking_round = RankingRound(
            ranking_round_id=str(uuid4()),
            research_id=document.research_id,
            round_index=round_index,
            candidate_count=len(candidates),
            target_final_count=target_final_count,
            ranked_hypothesis_ids=[item.hypothesis_id for item in normalized_rankings],
            promoted_hypothesis_ids=promoted_ids,
            evolved_parent_hypothesis_ids=evolved_parent_ids,
            rejected_hypothesis_ids=rejected_ids,
            rankings=normalized_rankings,
            best_patterns=output.best_patterns,
            worst_patterns=output.worst_patterns,
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
        )
        rankings_by_id = {item.hypothesis_id: item for item in normalized_rankings}
        ranked_hypotheses = [
            hypothesis.model_copy(
                update={
                    "ranking_score": rankings_by_id[hypothesis.hypothesis_id].score,
                    "ranking_rationale": rankings_by_id[hypothesis.hypothesis_id].rationale,
                    "ranking_round_id": ranking_round.ranking_round_id,
                    "ranking_status": rankings_by_id[hypothesis.hypothesis_id].recommended_action,
                }
            )
            for hypothesis in candidates
            if hypothesis.hypothesis_id in rankings_by_id
        ]
        return ranking_round, ranked_hypotheses

    def _llm_rank(
        self,
        document: ResearchGoalDocument,
        candidates: list[Hypothesis],
        heuristic_rankings: list[RankedHypothesis],
        target_final_count: int,
    ) -> RankingOutput:
        heuristic_by_id = {item.hypothesis_id: item.score for item in heuristic_rankings}
        payload = [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "summary": hypothesis.summary,
                "application": hypothesis.application,
                "market_segment": hypothesis.market_segment,
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
                "candidate_artifact": hypothesis.candidate_artifact,
                "evaluation_results": [result.model_dump() for result in hypothesis.evaluation_results],
                "nbca": hypothesis.next_best_competitive_alternative,
                "generation_source": hypothesis.generation_source,
                "round_index": hypothesis.round_index,
                "heuristic_score": round(heuristic_by_id.get(hypothesis.hypothesis_id, 0.0), 3),
                "reflection": self._assessment_payload(hypothesis.reflection_assessment),
                "evidence_gaps": (hypothesis.reflection_assessment.evidence_gap_notes if hypothesis.reflection_assessment else []),
                "tool_request_notes": (hypothesis.reflection_assessment.tool_request_notes if hypothesis.reflection_assessment else []),
                "user_feedback_status": hypothesis.user_feedback_status,
                "user_feedback_comment": hypothesis.user_feedback_comment,
            }
            for hypothesis in candidates
        ]
        system_prompt = PROMPTS.render("ranking_agent", "rank.system")
        user_prompt = PROMPTS.render(
            "ranking_agent",
            "rank.user",
            research_goal=document.raw_goal,
            document_json=document.model_dump_json(indent=2),
            target_final_count=target_final_count,
            hypotheses_json=json.dumps(payload, indent=2),
        )
        return self._llm.complete_json(RankingOutput, system_prompt, user_prompt)

    @classmethod
    def _heuristic_rankings(cls, hypotheses: list[Hypothesis]) -> list[RankedHypothesis]:
        rankings = [
            RankedHypothesis(
                hypothesis_id=hypothesis.hypothesis_id,
                score=cls._heuristic_score(hypothesis),
                recommended_action="hold",
                rationale="Deterministic weighted score from reflected assessment metrics.",
            )
            for hypothesis in hypotheses
        ]
        rankings.sort(key=lambda item: (-item.score, item.hypothesis_id))
        return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(rankings)]

    @staticmethod
    def _merge_rankings(
        candidates: list[Hypothesis],
        heuristic_rankings: list[RankedHypothesis],
        llm_rankings: list[RankedHypothesis],
    ) -> list[RankedHypothesis]:
        candidate_ids = {hypothesis.hypothesis_id for hypothesis in candidates}
        heuristic_by_id = {item.hypothesis_id: item for item in heuristic_rankings}
        merged: dict[str, RankedHypothesis] = {}
        for item in llm_rankings:
            if item.hypothesis_id in candidate_ids:
                merged[item.hypothesis_id] = item
        for hypothesis_id in candidate_ids:
            if hypothesis_id not in merged:
                merged[hypothesis_id] = heuristic_by_id[hypothesis_id]
        return list(merged.values())

    @staticmethod
    def _normalized_action(index: int, ranking: RankedHypothesis, target_final_count: int, evolve_top_k: int) -> RankingAction:
        if index < min(target_final_count, evolve_top_k):
            return "evolve"
        if index < target_final_count:
            return "advance"
        if ranking.score < 0.35:
            return "reject"
        return "hold"

    @classmethod
    def _heuristic_score(cls, hypothesis: Hypothesis) -> float:
        assessment = hypothesis.reflection_assessment
        if assessment is None:
            return max(0.0, min(1.0, hypothesis.generation_confidence * 0.5))
        if assessment.criterion_results:
            return cls._criterion_score(assessment)
        weighted_values = [
            (assessment.strategic_fit_score.value, 0.16),
            (assessment.market_size_score.value, 0.16),
            (assessment.replacement_fit_score.value, 0.16),
            (assessment.activation_ease_score.value, 0.12),
            (assessment.replacement_driver_strength_score.value, 0.12),
            (assessment.technical_success_probability.value, 0.14),
            (assessment.commercial_success_probability.value, 0.14),
        ]
        present = [(float(value), weight) for value, weight in weighted_values if value is not None]
        if not present:
            return max(0.0, min(1.0, hypothesis.generation_confidence * 0.5))
        score = sum(value * weight for value, weight in present) / sum(weight for _, weight in present)
        gap_penalty = min(len(assessment.evidence_gap_notes) * 0.025, 0.15)
        return max(0.0, min(1.0, score - gap_penalty))

    @staticmethod
    def _criterion_score(assessment: ReflectionAssessment) -> float:
        present = [
            (float(result.normalized_score), max(0.25, result.confidence))
            for result in assessment.criterion_results
            if result.normalized_score is not None
        ]
        if not present:
            return 0.25
        score = sum(value * weight for value, weight in present) / sum(weight for _, weight in present)
        gap_penalty = min(len(assessment.evidence_gap_notes) * 0.025, 0.15)
        tool_penalty = min(len(assessment.tool_request_notes) * 0.02, 0.1)
        return max(0.0, min(1.0, score - gap_penalty - tool_penalty))

    @staticmethod
    def _assessment_payload(assessment: ReflectionAssessment | None) -> dict[str, Any]:
        if assessment is None:
            return {}
        return {
            "strategic_fit_score": assessment.strategic_fit_score.value,
            "market_size_score": assessment.market_size_score.value,
            "incumbent_price_usd_per_kg": assessment.incumbent_price_usd_per_kg.value,
            "nbca_material": assessment.nbca_material,
            "nbca_price_usd_per_kg": assessment.nbca_price_usd_per_kg.value,
            "replacement_fit_score": assessment.replacement_fit_score.value,
            "activation_ease_score": assessment.activation_ease_score.value,
            "replacement_driver_strength_score": assessment.replacement_driver_strength_score.value,
            "technical_success_probability": assessment.technical_success_probability.value,
            "commercial_success_probability": assessment.commercial_success_probability.value,
            "criterion_results": [result.model_dump() for result in assessment.criterion_results],
            "tool_request_notes": assessment.tool_request_notes,
        }


class EvolutionAgent:
    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def evolve(
        self,
        document: ResearchGoalDocument,
        parent_hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
        target_count: int,
        round_index: int,
    ) -> list[Hypothesis]:
        if target_count <= 0 or not parent_hypotheses:
            return []
        try:
            output = self._llm_evolve(document, parent_hypotheses, ranking_round, target_count)
            seeds = output.hypotheses
        except Exception as exc:
            LOGGER.warning(
                "Evolution LLM failed (%s); falling back to conservative parent variants",
                exc,
            )
            seeds = self._fallback_seeds(parent_hypotheses, target_count)
        return self._seeds_to_hypotheses(document, seeds, target_count, round_index)

    def _llm_evolve(
        self,
        document: ResearchGoalDocument,
        parent_hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
        target_count: int,
    ) -> HypothesisEvolutionOutput:
        parent_payload = [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "summary": hypothesis.summary,
                "application": hypothesis.application,
                "market_segment": hypothesis.market_segment,
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
                "nbca": hypothesis.next_best_competitive_alternative,
                "application_requirements": hypothesis.application_requirements,
                "substitution_drivers": hypothesis.substitution_drivers,
                "ranking_score": hypothesis.ranking_score,
                "ranking_rationale": hypothesis.ranking_rationale,
                "reflection": RankingAgent._assessment_payload(hypothesis.reflection_assessment),
            }
            for hypothesis in parent_hypotheses
        ]
        system_prompt = PROMPTS.render("evolution_agent", "evolve.system")
        user_prompt = PROMPTS.render(
            "evolution_agent",
            "evolve.user",
            research_goal=document.raw_goal,
            best_patterns_json=json.dumps(ranking_round.best_patterns, indent=2),
            worst_patterns_json=json.dumps(ranking_round.worst_patterns, indent=2),
            parent_hypotheses_json=json.dumps(parent_payload, indent=2),
            target_count=target_count,
        )
        return self._llm.complete_json(HypothesisEvolutionOutput, system_prompt, user_prompt)

    @staticmethod
    def _fallback_seeds(parent_hypotheses: list[Hypothesis], target_count: int) -> list[EvolutionHypothesisSeed]:
        seeds: list[EvolutionHypothesisSeed] = []
        for parent in parent_hypotheses:
            seeds.append(
                EvolutionHypothesisSeed(
                    title=f"Focused variant: {parent.title}",
                    summary=(
                        f"A narrower variant of '{parent.title}' that prioritizes the fastest addressable "
                        "activation path and explicitly tests unresolved commercial gaps."
                    ),
                    application=parent.application,
                    market_segment=parent.market_segment,
                    candidate_material=parent.candidate_material,
                    incumbent_material=parent.incumbent_material,
                    next_best_competitive_alternative=parent.next_best_competitive_alternative,
                    incumbent_form=parent.incumbent_form,
                    candidate_form=parent.candidate_form,
                    conversion_process=parent.conversion_process,
                    product_type=parent.product_type,
                    buyer_type=parent.buyer_type,
                    application_requirements=parent.application_requirements,
                    substitution_drivers=parent.substitution_drivers,
                    strategic_rationale=parent.strategic_rationale,
                    supporting_chunk_ids=parent.supporting_chunk_ids,
                    supporting_urls=parent.supporting_urls,
                    assumptions=parent.assumptions,
                    unknowns=list(OrderedDict.fromkeys(parent.unknowns + ["Validate evolved activation pathway."])),
                    generation_confidence=max(0.0, min(parent.generation_confidence, 0.55)),
                    parent_hypothesis_ids=[parent.hypothesis_id],
                    mutation_strategy="Conservative narrowing of the parent opportunity.",
                    evolution_notes=["Fallback variant generated without LLM output."],
                )
            )
            if len(seeds) >= target_count:
                break
        return seeds

    @staticmethod
    def _seeds_to_hypotheses(
        document: ResearchGoalDocument,
        seeds: list[EvolutionHypothesisSeed],
        target_count: int,
        round_index: int,
    ) -> list[Hypothesis]:
        hypotheses: "OrderedDict[str, Hypothesis]" = OrderedDict()
        for seed in seeds:
            parent_ids = seed.parent_hypothesis_ids
            hypothesis_id = str(
                uuid5(
                    NAMESPACE_URL,
                    (
                        f"{document.research_id}::evolved::{round_index}::{seed.title}::"
                        f"{seed.application or ''}::{seed.candidate_material or ''}::{','.join(parent_ids)}"
                    ),
                )
            )
            if hypothesis_id in hypotheses:
                continue
            notes = list(OrderedDict.fromkeys([seed.mutation_strategy] + seed.evolution_notes))
            hypotheses[hypothesis_id] = Hypothesis(
                hypothesis_id=hypothesis_id,
                research_id=document.research_id,
                status="generated",
                title=seed.title,
                summary=seed.summary,
                application=seed.application,
                market_segment=seed.market_segment,
                region_scope=document.regions,
                candidate_material=seed.candidate_material,
                incumbent_material=seed.incumbent_material,
                next_best_competitive_alternative=seed.next_best_competitive_alternative,
                incumbent_form=seed.incumbent_form,
                candidate_form=seed.candidate_form,
                conversion_process=seed.conversion_process,
                product_type=seed.product_type,
                buyer_type=seed.buyer_type,
                application_requirements=seed.application_requirements,
                substitution_drivers=seed.substitution_drivers,
                strategic_rationale=seed.strategic_rationale,
                supporting_chunk_ids=seed.supporting_chunk_ids,
                supporting_urls=seed.supporting_urls,
                assumptions=seed.assumptions,
                unknowns=seed.unknowns,
                generation_confidence=seed.generation_confidence,
                round_index=round_index,
                generation_source="evolved",
                parent_hypothesis_ids=parent_ids,
                evolution_notes=notes,
            )
            if len(hypotheses) >= target_count:
                break
        return list(hypotheses.values())


class ProximityCheckAgent:
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    _REGION_TOKENS = {
        "apac",
        "asia",
        "america",
        "americas",
        "emea",
        "eu",
        "europe",
        "european",
        "global",
        "latin",
        "latam",
        "na",
        "north",
        "pacific",
        "south",
        "worldwide",
    }
    _FAMILY_GENERIC_TOKENS = {
        "application",
        "applications",
        "clear",
        "clinical",
        "conversion",
        "device",
        "devices",
        "format",
        "formats",
        "hospital",
        "market",
        "markets",
        "medical",
        "packaging",
        "primary",
        "product",
        "products",
        "rigid",
        "secondary",
        "segment",
        "segments",
        "sterile",
        "surgical",
        "thermoform",
        "thermoformed",
        "transparent",
    }

    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def review(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        round_index: int,
        max_synthesized_hypotheses: int,
    ) -> tuple[ProximityRound, list[Hypothesis], list[Hypothesis]]:
        active_reflected = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.status == "reflected" and hypothesis.is_active
        ]
        return self._review_candidates(
            document=document,
            candidates=active_reflected,
            round_index=round_index,
            max_synthesized_hypotheses=max_synthesized_hypotheses,
            empty_note="No active reflected hypotheses available for proximity review.",
        )

    def review_generated(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        round_index: int,
        max_synthesized_hypotheses: int,
    ) -> tuple[ProximityRound, list[Hypothesis], list[Hypothesis]]:
        active_generated = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.status == "generated" and hypothesis.is_active
        ]
        return self._review_candidates(
            document=document,
            candidates=active_generated,
            round_index=round_index,
            max_synthesized_hypotheses=max_synthesized_hypotheses,
            empty_note="No active generated hypotheses available for pre-reflection proximity review.",
        )

    def _review_candidates(
        self,
        document: ResearchGoalDocument,
        candidates: list[Hypothesis],
        round_index: int,
        max_synthesized_hypotheses: int,
        empty_note: str,
    ) -> tuple[ProximityRound, list[Hypothesis], list[Hypothesis]]:
        if not candidates:
            proximity_round = ProximityRound(
                proximity_round_id=str(uuid4()),
                research_id=document.research_id,
                round_index=round_index,
                notes=[empty_note],
            )
            return proximity_round, [], []
        clusters = self._deterministic_clusters(document, candidates)
        if not clusters:
            policy = self._policy_for(document)
            proximity_round = ProximityRound(
                proximity_round_id=str(uuid4()),
                research_id=document.research_id,
                round_index=round_index,
                notes=[
                    (
                        "No overlapping hypothesis clusters found under proximity policy "
                        f"{policy.merge_mode}/{policy.granularity}."
                    )
                ],
            )
            return proximity_round, [], []

        active_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in candidates}
        updated_hypotheses: list[Hypothesis] = []
        synthesized_hypotheses: list[Hypothesis] = []
        concepts: list[ProximityConcept] = []
        retired_hypothesis_ids: list[str] = []
        labeled_hypothesis_ids: list[str] = []
        notes = [
            (
                "Applied proximity policy "
                f"{self._policy_for(document).merge_mode}/{self._policy_for(document).granularity} "
                "with region union enabled for synthesized hypotheses."
            )
        ]

        for index, cluster in enumerate(clusters):
            concept, seed, cluster_notes = self._review_cluster(
                document=document,
                cluster=cluster,
                synthesize=index < max(0, max_synthesized_hypotheses),
            )
            concepts.append(concept)
            notes.extend(cluster_notes)
            cluster_ids = [hypothesis.hypothesis_id for hypothesis in cluster]
            labeled_hypothesis_ids.extend(cluster_ids)
            for hypothesis in cluster:
                labels = list(OrderedDict.fromkeys(hypothesis.concept_labels + [concept.concept_label]))
                updated_hypotheses.append(
                    hypothesis.model_copy(
                        update={
                            "concept_labels": labels,
                            "concept_cluster_id": hypothesis.concept_cluster_id or concept.concept_label,
                        }
                    )
                )
            if seed is None:
                continue
            synthesized = self._seed_to_hypothesis(document, seed, round_index, cluster_ids, cluster)
            synthesized_hypotheses.append(synthesized)
            for member_id in cluster_ids:
                retired_hypothesis_ids.append(member_id)
                member = active_by_id[member_id]
                merged_labels = list(
                    OrderedDict.fromkeys(member.concept_labels + [concept.concept_label])
                )
                updated_hypotheses.append(
                    member.model_copy(
                        update={
                            "status": "retired",
                            "is_active": False,
                            "retired_reason": "merged_into_synthesized_hypothesis",
                            "superseded_by_hypothesis_id": synthesized.hypothesis_id,
                            "concept_labels": merged_labels,
                            "concept_cluster_id": member.concept_cluster_id or concept.concept_label,
                        }
                    )
                )

        proximity_round = ProximityRound(
            proximity_round_id=str(uuid4()),
            research_id=document.research_id,
            round_index=round_index,
            concepts=concepts,
            synthesized_hypothesis_ids=[hypothesis.hypothesis_id for hypothesis in synthesized_hypotheses],
            retired_hypothesis_ids=list(OrderedDict.fromkeys(retired_hypothesis_ids)),
            labeled_hypothesis_ids=list(OrderedDict.fromkeys(labeled_hypothesis_ids)),
            notes=list(OrderedDict.fromkeys(notes)),
        )
        deduped_updates: "OrderedDict[str, Hypothesis]" = OrderedDict()
        for hypothesis in updated_hypotheses:
            deduped_updates[hypothesis.hypothesis_id] = hypothesis
        return proximity_round, list(deduped_updates.values()), synthesized_hypotheses

    def _review_cluster(
        self,
        document: ResearchGoalDocument,
        cluster: list[Hypothesis],
        synthesize: bool,
    ) -> tuple[ProximityConcept, SynthesizedHypothesisSeed | None, list[str]]:
        cluster_ids = [hypothesis.hypothesis_id for hypothesis in cluster]
        output: ProximityReviewOutput | None = None
        notes: list[str] = []
        if synthesize:
            try:
                output = self._llm_review(document, cluster, max_synthesized_hypotheses=1)
                notes.extend(output.notes)
            except Exception as exc:
                LOGGER.warning("Proximity review failed for cluster %s (%s); falling back to deterministic synthesis", cluster_ids, exc)
                notes.append(f"Used deterministic synthesis fallback for cluster {', '.join(cluster_ids)}.")
        concept_label = self._concept_label_from_output(output) or self._fallback_concept_label(document, cluster)
        concept_description = self._concept_description_from_output(output) or (
            "Deterministic overlap cluster using the project proximity merge policy."
        )
        concept = ProximityConcept(
            concept_label=concept_label,
            description=concept_description,
            member_hypothesis_ids=cluster_ids,
        )
        if not synthesize:
            return concept, None, notes

        seed = self._synthesized_seed_from_output(output)
        if seed is None:
            seed = self._fallback_synthesized_seed(document, cluster, concept)
        else:
            seed = seed.model_copy(
                update={
                    "merged_from_hypothesis_ids": cluster_ids,
                    "concept_label": concept.concept_label,
                }
            )
        return concept, seed, notes

    def _llm_review(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        max_synthesized_hypotheses: int,
    ) -> ProximityReviewOutput:
        payload = [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "summary": hypothesis.summary,
                "application": hypothesis.application,
                "market_segment": hypothesis.market_segment,
                "region_scope": hypothesis.region_scope,
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
                "product_type": hypothesis.product_type,
                "buyer_type": hypothesis.buyer_type,
                "application_requirements": hypothesis.application_requirements,
                "substitution_drivers": hypothesis.substitution_drivers,
                "concept_labels": hypothesis.concept_labels,
                "ranking_score": hypothesis.ranking_score,
                "reflection": RankingAgent._assessment_payload(hypothesis.reflection_assessment),
            }
            for hypothesis in hypotheses
        ]
        system_prompt = PROMPTS.render("proximity_check_agent", "review.system")
        user_prompt = PROMPTS.render(
            "proximity_check_agent",
            "review.user",
            research_goal=document.raw_goal,
            document_json=document.model_dump_json(indent=2),
            hypotheses_json=json.dumps(payload, indent=2),
            max_synthesized_hypotheses=max_synthesized_hypotheses,
        )
        return self._llm.complete_json(ProximityReviewOutput, system_prompt, user_prompt)

    @classmethod
    def _deterministic_clusters(
        cls,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
    ) -> list[list[Hypothesis]]:
        if len(hypotheses) < 2:
            return []

        parent = list(range(len(hypotheses)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left in range(len(hypotheses)):
            for right in range(left + 1, len(hypotheses)):
                if cls._should_cluster(document, hypotheses[left], hypotheses[right]):
                    union(left, right)

        grouped: dict[int, list[Hypothesis]] = {}
        for index, hypothesis in enumerate(hypotheses):
            grouped.setdefault(find(index), []).append(hypothesis)
        clusters = [group for group in grouped.values() if len(group) >= 2]
        clusters.sort(key=cls._cluster_sort_key)
        return clusters

    @classmethod
    def _should_cluster(
        cls,
        document: ResearchGoalDocument,
        left: Hypothesis,
        right: Hypothesis,
    ) -> bool:
        if not cls._same_core_opportunity(document, left, right):
            return False
        if GenerationAgent._same_primary_artifact(document, left, right):
            return True

        policy = cls._policy_for(document)
        family_overlap = bool(cls._family_tokens(left) & cls._family_tokens(right))
        market_overlap = bool(cls._market_tokens(left) & cls._market_tokens(right))
        support_overlap = cls._supporting_overlap(left, right)
        similarity = cls._token_similarity(cls._context_tokens(left), cls._context_tokens(right))

        if policy.granularity == "device_subtype":
            return cls._application_signature(left) == cls._application_signature(right) != ""

        if policy.granularity == "application_family":
            if family_overlap:
                if policy.merge_mode == "conservative":
                    return support_overlap or similarity >= 0.35
                return True
            return policy.merge_mode == "aggressive" and market_overlap and similarity >= 0.45

        if policy.merge_mode == "conservative":
            return family_overlap and (market_overlap or support_overlap)
        if policy.merge_mode == "balanced":
            return family_overlap or market_overlap
        return family_overlap or market_overlap or similarity >= 0.45

    @classmethod
    def _same_core_opportunity(
        cls,
        document: ResearchGoalDocument,
        left: Hypothesis,
        right: Hypothesis,
    ) -> bool:
        left_candidate = cls._candidate_key(document, left)
        right_candidate = cls._candidate_key(document, right)
        left_incumbent = cls._normalize_text_key(left.incumbent_material)
        right_incumbent = cls._normalize_text_key(right.incumbent_material)
        if left_candidate and right_candidate and left_candidate != right_candidate:
            return False
        if left_incumbent and right_incumbent and left_incumbent != right_incumbent:
            return False
        return bool(left_candidate or right_candidate or left_incumbent or right_incumbent)

    @classmethod
    def _candidate_key(cls, document: ResearchGoalDocument, hypothesis: Hypothesis) -> str:
        if document.research_mode != "materials_opportunity":
            identifier = GenerationAgent._artifact_identifier_key(document, hypothesis)
            if identifier:
                return identifier
            candidate = hypothesis.candidate_material
            return cls._normalize_text_key(candidate)
        candidate = hypothesis.candidate_material or GenerationAgent._primary_artifact_identifier(document, hypothesis)
        return cls._material_family_key(candidate)

    @staticmethod
    def _policy_for(document: ResearchGoalDocument) -> ProximityMergePolicy:
        return ProximityMergePolicy.model_validate(document.proximity_merge_policy)

    @classmethod
    def _application_signature(cls, hypothesis: Hypothesis) -> str:
        tokens = cls._token_set(
            [hypothesis.application, hypothesis.product_type],
            drop_family_generic=False,
        )
        return " ".join(sorted(tokens))

    @classmethod
    def _family_tokens(cls, hypothesis: Hypothesis) -> set[str]:
        tokens = cls._token_set(
            [hypothesis.application, hypothesis.product_type],
            drop_family_generic=True,
        )
        if tokens:
            return tokens
        return cls._token_set([hypothesis.application, hypothesis.product_type], drop_family_generic=False)

    @classmethod
    def _market_tokens(cls, hypothesis: Hypothesis) -> set[str]:
        tokens = cls._token_set(
            [hypothesis.market_segment, hypothesis.buyer_type],
            drop_family_generic=True,
        )
        if tokens:
            return tokens
        return cls._token_set([hypothesis.market_segment, hypothesis.buyer_type], drop_family_generic=False)

    @classmethod
    def _context_tokens(cls, hypothesis: Hypothesis) -> set[str]:
        return cls._token_set(
            [
                hypothesis.application,
                hypothesis.product_type,
                hypothesis.market_segment,
                hypothesis.buyer_type,
                " ".join(hypothesis.application_requirements),
                " ".join(hypothesis.substitution_drivers),
            ],
            drop_family_generic=False,
        )

    @classmethod
    def _supporting_overlap(cls, left: Hypothesis, right: Hypothesis) -> bool:
        left_support = cls._token_set(
            [" ".join(left.application_requirements), " ".join(left.substitution_drivers)],
            drop_family_generic=False,
        )
        right_support = cls._token_set(
            [" ".join(right.application_requirements), " ".join(right.substitution_drivers)],
            drop_family_generic=False,
        )
        return bool(left_support & right_support)

    @classmethod
    def _token_set(cls, values: list[str | None], drop_family_generic: bool) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
                normalized = cls._singularize_token(token)
                if not normalized or normalized in cls._STOPWORDS or normalized in cls._REGION_TOKENS:
                    continue
                if drop_family_generic and normalized in cls._FAMILY_GENERIC_TOKENS:
                    continue
                tokens.add(normalized)
        return tokens

    @staticmethod
    def _singularize_token(token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    @staticmethod
    def _token_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / union if union else 0.0

    @classmethod
    def _cluster_sort_key(cls, cluster: list[Hypothesis]) -> tuple[float, float, str]:
        mean_score = sum(hypothesis.ranking_score or 0.0 for hypothesis in cluster) / len(cluster)
        title_key = min(hypothesis.title for hypothesis in cluster)
        return (-len(cluster), -mean_score, title_key)

    @staticmethod
    def _concept_label_from_output(output: ProximityReviewOutput | None) -> str:
        if output is None:
            return ""
        for concept in output.concepts:
            if concept.concept_label.strip():
                return concept.concept_label.strip()
        return ""

    @staticmethod
    def _concept_description_from_output(output: ProximityReviewOutput | None) -> str:
        if output is None:
            return ""
        for concept in output.concepts:
            if concept.description.strip():
                return concept.description.strip()
        return ""

    @staticmethod
    def _synthesized_seed_from_output(output: ProximityReviewOutput | None) -> SynthesizedHypothesisSeed | None:
        if output is None or not output.synthesized_hypotheses:
            return None
        return output.synthesized_hypotheses[0]

    @classmethod
    def _fallback_concept_label(
        cls,
        document: ResearchGoalDocument,
        cluster: list[Hypothesis],
    ) -> str:
        candidate = cls._cluster_candidate_label(document, cluster) or "candidate"
        focus = cls._cluster_focus_label(cluster) or "opportunity"
        return f"{candidate} {focus} cluster".strip()

    @classmethod
    def _fallback_synthesized_seed(
        cls,
        document: ResearchGoalDocument,
        cluster: list[Hypothesis],
        concept: ProximityConcept,
    ) -> SynthesizedHypothesisSeed:
        candidate = cls._cluster_candidate_label(document, cluster) or "Candidate"
        incumbent = cls._most_common_text([hypothesis.incumbent_material for hypothesis in cluster])
        focus = cls._cluster_focus_label(cluster) or "opportunity"
        application = cls._synthesized_application(cluster, focus)
        market_segment = cls._most_common_text([hypothesis.market_segment for hypothesis in cluster])
        title = f"{candidate} platform for {application}".strip()
        summary = (
            f"A merged opportunity covering overlapping {focus} variants that share the same core replacement thesis."
        )
        if market_segment:
            summary = f"{summary} The combined opportunity is anchored in {market_segment}."
        return SynthesizedHypothesisSeed(
            title=title,
            summary=summary,
            application=application,
            market_segment=market_segment or None,
            candidate_material=cls._most_common_text([hypothesis.candidate_material for hypothesis in cluster]) or None,
            incumbent_material=incumbent or None,
            next_best_competitive_alternative=cls._most_common_text(
                [hypothesis.next_best_competitive_alternative for hypothesis in cluster]
            )
            or None,
            incumbent_form=cls._most_common_text([hypothesis.incumbent_form for hypothesis in cluster]) or None,
            candidate_form=cls._most_common_text([hypothesis.candidate_form for hypothesis in cluster]) or None,
            conversion_process=cls._most_common_text([hypothesis.conversion_process for hypothesis in cluster]) or None,
            product_type=cls._most_common_text([hypothesis.product_type for hypothesis in cluster]) or None,
            buyer_type=cls._most_common_text([hypothesis.buyer_type for hypothesis in cluster]) or None,
            application_requirements=cls._union_lists(hypothesis.application_requirements for hypothesis in cluster),
            substitution_drivers=cls._union_lists(hypothesis.substitution_drivers for hypothesis in cluster),
            strategic_rationale=(
                "Synthesizes overlapping reflected hypotheses into a single opportunity family and "
                "combines regional scope into one thesis."
            ),
            supporting_chunk_ids=cls._union_lists(hypothesis.supporting_chunk_ids for hypothesis in cluster),
            supporting_urls=cls._union_lists(hypothesis.supporting_urls for hypothesis in cluster),
            assumptions=cls._union_lists(hypothesis.assumptions for hypothesis in cluster),
            unknowns=cls._union_lists(hypothesis.unknowns for hypothesis in cluster),
            candidate_artifact=cls._first_non_empty_artifact(cluster),
            evaluation_results=[],
            generation_confidence=max(
                0.0,
                min(
                    1.0,
                    sum(hypothesis.generation_confidence for hypothesis in cluster) / len(cluster),
                ),
            ),
            merged_from_hypothesis_ids=[hypothesis.hypothesis_id for hypothesis in cluster],
            concept_label=concept.concept_label,
            synthesis_rationale="Deterministic synthesis for a policy-defined overlap cluster.",
        )

    @classmethod
    def _cluster_candidate_label(
        cls,
        document: ResearchGoalDocument,
        cluster: list[Hypothesis],
    ) -> str:
        candidates = [
            hypothesis.candidate_material or GenerationAgent._primary_artifact_identifier(document, hypothesis)
            for hypothesis in cluster
        ]
        return cls._most_common_text(candidates)

    @classmethod
    def _cluster_focus_label(cls, cluster: list[Hypothesis]) -> str:
        family_sets = [cls._family_tokens(hypothesis) for hypothesis in cluster if cls._family_tokens(hypothesis)]
        if family_sets:
            shared = set.intersection(*family_sets)
            if shared:
                return " ".join(sorted(shared))
        return (
            cls._most_common_text([hypothesis.application for hypothesis in cluster])
            or cls._most_common_text([hypothesis.product_type for hypothesis in cluster])
            or cls._most_common_text([hypothesis.market_segment for hypothesis in cluster])
            or "opportunity"
        )

    @classmethod
    def _synthesized_application(cls, cluster: list[Hypothesis], focus: str) -> str:
        application = cls._most_common_text([hypothesis.application for hypothesis in cluster])
        if application:
            return application
        if focus == "opportunity":
            return "combined opportunity"
        return f"{focus} opportunities"

    @staticmethod
    def _most_common_text(values: list[str | None]) -> str:
        filtered = [str(value).strip() for value in values if str(value or "").strip()]
        if not filtered:
            return ""
        counts = Counter(filtered)
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @staticmethod
    def _union_lists(values: list[list[str]] | Any) -> list[str]:
        merged: list[str] = []
        for items in values:
            for item in items:
                if item and item not in merged:
                    merged.append(item)
        return merged

    @staticmethod
    def _first_non_empty_artifact(cluster: list[Hypothesis]) -> dict[str, Any]:
        for hypothesis in cluster:
            if hypothesis.candidate_artifact:
                return dict(hypothesis.candidate_artifact)
        return {}

    @staticmethod
    def _normalize_text_key(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @classmethod
    def _material_family_key(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"\b(?:e\.?g\.?|grade|grades|injection|molding|extrusion|resin|series|type)\b", " ", text)
        text = re.sub(r"\b[a-z]{1,4}\d{2,}[a-z0-9-]*\b", " ", text)
        return cls._normalize_text_key(text)

    @staticmethod
    def _seed_to_hypothesis(
        document: ResearchGoalDocument,
        seed: SynthesizedHypothesisSeed,
        round_index: int,
        member_ids: list[str],
        members: list[Hypothesis],
    ) -> Hypothesis:
        hypothesis_id = str(
            uuid5(
                NAMESPACE_URL,
                (
                    f"{document.research_id}::synthesized::{round_index}::{seed.title}::"
                    f"{seed.application or ''}::{seed.candidate_material or ''}::{','.join(member_ids)}"
                ),
            )
        )
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            research_id=document.research_id,
            status="generated",
            title=seed.title,
            summary=seed.summary,
            application=seed.application,
            market_segment=seed.market_segment,
            region_scope=ProximityCheckAgent._union_lists([member.region_scope for member in members]) or document.regions,
            candidate_material=seed.candidate_material,
            incumbent_material=seed.incumbent_material,
            next_best_competitive_alternative=seed.next_best_competitive_alternative,
            incumbent_form=seed.incumbent_form,
            candidate_form=seed.candidate_form,
            conversion_process=seed.conversion_process,
            product_type=seed.product_type,
            buyer_type=seed.buyer_type,
            application_requirements=seed.application_requirements,
            substitution_drivers=seed.substitution_drivers,
            strategic_rationale=seed.synthesis_rationale or seed.strategic_rationale,
            supporting_chunk_ids=seed.supporting_chunk_ids,
            supporting_urls=seed.supporting_urls,
            assumptions=seed.assumptions,
            unknowns=seed.unknowns,
            candidate_artifact=seed.candidate_artifact or ProximityCheckAgent._first_non_empty_artifact(members),
            evaluation_results=seed.evaluation_results,
            generation_confidence=seed.generation_confidence,
            round_index=round_index,
            generation_source="synthesized",
            parent_hypothesis_ids=member_ids,
            merged_from_hypothesis_ids=member_ids,
            concept_labels=[seed.concept_label] if seed.concept_label else [],
            concept_cluster_id=seed.concept_label,
            evolution_notes=["Synthesized from overlapping reflected hypotheses."],
        )


class MetaReviewAgent:
    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def review(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
        round_index: int,
        gap_overlap_threshold: float,
        max_gap_persistence_rounds: int,
    ) -> tuple[ResearchGoalDocument, MetaReviewRound]:
        active_reflected = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.status == "reflected" and hypothesis.is_active
        ]
        feedback_hypotheses = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.user_feedback_status is not None or hypothesis.user_feedback_comment
        ]
        try:
            output = self._llm_review(document, active_reflected, ranking_round, feedback_hypotheses)
        except Exception as exc:
            LOGGER.warning("Meta-review failed (%s); falling back to deterministic gap review", exc)
            output = self._fallback_review(document, active_reflected, ranking_round, feedback_hypotheses)

        previous_gaps = document.whitespace_gap_notes
        current_gaps = output.whitespace_gaps
        gap_overlap = self._gap_overlap(previous_gaps, current_gaps)
        gap_count_shrank = len(current_gaps) < len(previous_gaps) if previous_gaps else False
        meaningfully_shrunk = gap_overlap < gap_overlap_threshold or gap_count_shrank
        if previous_gaps and current_gaps and not meaningfully_shrunk:
            persistence_count = document.whitespace_gap_persistence_count + 1
        elif current_gaps:
            persistence_count = 0
        else:
            persistence_count = 0

        should_continue = True
        stop_reason = None
        if current_gaps and persistence_count > max_gap_persistence_rounds:
            should_continue = False
            stop_reason = "meta_review_gap_persistence"
        elif not current_gaps and output.coverage_sufficient:
            should_continue = False
            stop_reason = "meta_review_coverage_sufficient"

        shrinkage_status: GapShrinkageStatus
        if not previous_gaps:
            shrinkage_status = output.gap_shrinkage_status if output.gap_shrinkage_status != "unknown" else "unknown"
        elif meaningfully_shrunk:
            shrinkage_status = "improved"
        elif len(current_gaps) > len(previous_gaps):
            shrinkage_status = "worse"
        else:
            shrinkage_status = "stable"

        updated_document = document.model_copy(
            update={
                "whitespace_gap_notes": current_gaps,
                "whitespace_gap_persistence_count": persistence_count,
                "meta_review_generation_guidance": output.generation_guidance,
                "emerging_concept_labels": self._collect_concepts(active_reflected),
                "last_meta_review_round_index": round_index,
            }
        )
        meta_review_round = MetaReviewRound(
            meta_review_round_id=str(uuid4()),
            research_id=document.research_id,
            round_index=round_index,
            whitespace_gaps=current_gaps,
            generation_guidance=output.generation_guidance,
            coverage_assessment=output.coverage_assessment,
            gap_shrinkage_status=shrinkage_status,
            coverage_sufficient=output.coverage_sufficient,
            should_continue=should_continue,
            stop_reason=stop_reason,
            gap_persistence_count=persistence_count,
        )
        return updated_document, meta_review_round

    def _llm_review(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
        feedback_hypotheses: list[Hypothesis],
    ) -> MetaReviewOutput:
        payload = [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "application": hypothesis.application,
                "market_segment": hypothesis.market_segment,
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
                "concept_labels": hypothesis.concept_labels,
                "ranking_score": hypothesis.ranking_score,
                "ranking_status": hypothesis.ranking_status,
                "reflection": RankingAgent._assessment_payload(hypothesis.reflection_assessment),
                "evidence_gaps": (
                    hypothesis.reflection_assessment.evidence_gap_notes if hypothesis.reflection_assessment else []
                ),
                "user_feedback_status": hypothesis.user_feedback_status,
                "user_feedback_comment": hypothesis.user_feedback_comment,
            }
            for hypothesis in hypotheses
        ]
        feedback_payload = [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "application": hypothesis.application,
                "market_segment": hypothesis.market_segment,
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
                "status": hypothesis.status,
                "is_active": hypothesis.is_active,
                "ranking_status": hypothesis.ranking_status,
                "user_feedback_status": hypothesis.user_feedback_status,
                "user_feedback_comment": hypothesis.user_feedback_comment,
                "retired_reason": hypothesis.retired_reason,
            }
            for hypothesis in feedback_hypotheses
        ]
        system_prompt = PROMPTS.render("meta_review_agent", "review.system")
        user_prompt = PROMPTS.render(
            "meta_review_agent",
            "review.user",
            research_goal=document.raw_goal,
            document_json=document.model_dump_json(indent=2),
            best_patterns_json=json.dumps(ranking_round.best_patterns, indent=2),
            worst_patterns_json=json.dumps(ranking_round.worst_patterns, indent=2),
            previous_gaps_json=json.dumps(document.whitespace_gap_notes, indent=2),
            previous_guidance_json=json.dumps(document.meta_review_generation_guidance, indent=2),
            gap_persistence_count=document.whitespace_gap_persistence_count,
            hypotheses_json=json.dumps(payload, indent=2),
            feedback_hypotheses_json=json.dumps(feedback_payload, indent=2),
        )
        return self._llm.complete_json(MetaReviewOutput, system_prompt, user_prompt)

    @staticmethod
    def _fallback_review(
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
        feedback_hypotheses: list[Hypothesis],
    ) -> MetaReviewOutput:
        active_concepts = sorted(
            {
                label
                for hypothesis in hypotheses
                for label in hypothesis.concept_labels
                if label
            }
        )
        guidance = list(
            OrderedDict.fromkeys(
                document.meta_review_generation_guidance
                + ["Explore applications, regions, or buyer types not represented in the current reflected set."]
            )
        )
        accepted_or_edited = [
            hypothesis
            for hypothesis in feedback_hypotheses
            if hypothesis.user_feedback_status in {"accepted", "edited"}
        ]
        rejected = [
            hypothesis
            for hypothesis in feedback_hypotheses
            if hypothesis.user_feedback_status in {"rejected", "retired"}
        ]
        if accepted_or_edited:
            guidance.append(
                "Build on user-endorsed directions and sharpen the edited hypotheses with stronger evidence and narrower activation plans."
            )
        if rejected:
            guidance.append(
                "Avoid regenerating ideas that match user-rejected directions unless new evidence materially changes the thesis."
            )
        gaps = []
        if len(active_concepts) < max(1, min(3, document.target_hypotheses_final // 2)):
            gaps.append("Broaden concept diversity beyond the currently concentrated application clusters.")
        if not any(hypothesis.region_scope for hypothesis in hypotheses):
            gaps.append("Strengthen region-specific opportunity coverage.")
        return MetaReviewOutput(
            whitespace_gaps=gaps,
            generation_guidance=guidance,
            coverage_assessment="Fallback review based on concept diversity and reflected portfolio breadth.",
            gap_shrinkage_status="unknown",
            coverage_sufficient=not gaps and len(hypotheses) >= document.target_hypotheses_final,
        )

    @staticmethod
    def _gap_overlap(previous_gaps: list[str], current_gaps: list[str]) -> float:
        if not previous_gaps or not current_gaps:
            return 0.0
        previous_tokens = {
            " ".join(sorted(set(item.lower().split())))
            for item in previous_gaps
            if item.strip()
        }
        current_tokens = {
            " ".join(sorted(set(item.lower().split())))
            for item in current_gaps
            if item.strip()
        }
        if not previous_tokens or not current_tokens:
            return 0.0
        intersection = len(previous_tokens & current_tokens)
        return intersection / max(len(previous_tokens), len(current_tokens))

    @staticmethod
    def _collect_concepts(hypotheses: list[Hypothesis]) -> list[str]:
        return list(
            OrderedDict.fromkeys(
                label
                for hypothesis in hypotheses
                for label in hypothesis.concept_labels
                if label
            )
        )


class FinalPortfolioAgent:
    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def build_report(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        ranking_round: RankingRound | None,
        meta_review_round: MetaReviewRound | None,
        stop_reason: str,
        target_count: int,
    ) -> str:
        ranked = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.ranking_round_id and hypothesis.status == "reflected" and hypothesis.is_active
        ]
        ranked.sort(key=lambda item: (-(item.ranking_score or 0.0), item.title))
        top_ranked = ranked[:target_count]
        validation_gaps = self._collect_validation_gaps(top_ranked)
        ranking_payload = (
            {
                "ranking_round_id": ranking_round.ranking_round_id,
                "round_index": ranking_round.round_index,
                "candidate_count": ranking_round.candidate_count,
                "promoted_hypothesis_ids": ranking_round.promoted_hypothesis_ids,
                "best_patterns": ranking_round.best_patterns,
                "worst_patterns": ranking_round.worst_patterns,
                "mean_score": ranking_round.mean_score,
                "max_score": ranking_round.max_score,
            }
            if ranking_round is not None
            else {}
        )
        meta_payload = (
            {
                "round_index": meta_review_round.round_index,
                "whitespace_gaps": meta_review_round.whitespace_gaps,
                "generation_guidance": meta_review_round.generation_guidance,
                "coverage_assessment": meta_review_round.coverage_assessment,
                "gap_shrinkage_status": meta_review_round.gap_shrinkage_status,
                "coverage_sufficient": meta_review_round.coverage_sufficient,
                "should_continue": meta_review_round.should_continue,
                "stop_reason": meta_review_round.stop_reason,
            }
            if meta_review_round is not None
            else {}
        )
        opportunity_payload = [self._opportunity_payload(hypothesis) for hypothesis in top_ranked]
        system_prompt = PROMPTS.render("final_portfolio_agent", "build_report.system")
        user_prompt = PROMPTS.render(
            "final_portfolio_agent",
            "build_report.user",
            research_goal=document.raw_goal,
            document_json=document.model_dump_json(indent=2),
            stop_reason=stop_reason,
            ranking_round_json=json.dumps(ranking_payload, indent=2),
            meta_review_round_json=json.dumps(meta_payload, indent=2),
            top_opportunities_json=json.dumps(opportunity_payload, indent=2),
            validation_gaps_json=json.dumps(validation_gaps, indent=2),
        )
        try:
            return self._llm.complete_text(system_prompt, user_prompt)
        except Exception as exc:
            LOGGER.warning("Final portfolio report generation failed (%s); using deterministic fallback", exc)
            return CoScientistRunner._build_loop_report(document, ranking_round, meta_review_round, hypotheses, stop_reason)

    @staticmethod
    def _opportunity_payload(hypothesis: Hypothesis) -> dict[str, Any]:
        assessment = hypothesis.reflection_assessment or ReflectionAssessment()
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "summary": hypothesis.summary,
            "ranking_score": hypothesis.ranking_score,
            "ranking_status": hypothesis.ranking_status,
            "generation_source": hypothesis.generation_source,
            "application": hypothesis.application,
            "market_segment": hypothesis.market_segment,
            "candidate_material": hypothesis.candidate_material,
            "incumbent_material": hypothesis.incumbent_material,
            "concept_labels": hypothesis.concept_labels,
            "strategic_fit_score": assessment.strategic_fit_score.value,
            "market_size_score": assessment.market_size_score.value,
            "replacement_fit_score": assessment.replacement_fit_score.value,
            "activation_ease_score": assessment.activation_ease_score.value,
            "technical_success_probability": assessment.technical_success_probability.value,
            "commercial_success_probability": assessment.commercial_success_probability.value,
            "evidence_gap_notes": assessment.evidence_gap_notes,
            "unknowns": hypothesis.unknowns,
            "supporting_urls": hypothesis.supporting_urls,
        }

    @staticmethod
    def _collect_validation_gaps(hypotheses: list[Hypothesis]) -> list[dict[str, Any]]:
        gaps: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        metric_labels = {
            "market_size_score": "Market size validation missing or weak",
            "incumbent_price_usd_per_kg": "Incumbent pricing validation missing or weak",
            "nbca_price_usd_per_kg": "Competitive pricing validation missing or weak",
            "replacement_fit_score": "Replacement fit validation missing or weak",
            "activation_ease_score": "Activation ease validation missing or weak",
            "technical_success_probability": "Technical validation missing or weak",
            "commercial_success_probability": "Commercial validation missing or weak",
        }

        def bump(label: str) -> None:
            bucket = gaps.setdefault(label, {"gap": label, "count": 0})
            bucket["count"] += 1

        for hypothesis in hypotheses:
            assessment = hypothesis.reflection_assessment or ReflectionAssessment()
            for note in assessment.evidence_gap_notes:
                if note:
                    bump(note)
            metrics = {
                "market_size_score": assessment.market_size_score.value,
                "incumbent_price_usd_per_kg": assessment.incumbent_price_usd_per_kg.value,
                "nbca_price_usd_per_kg": assessment.nbca_price_usd_per_kg.value,
                "replacement_fit_score": assessment.replacement_fit_score.value,
                "activation_ease_score": assessment.activation_ease_score.value,
                "technical_success_probability": assessment.technical_success_probability.value,
                "commercial_success_probability": assessment.commercial_success_probability.value,
            }
            for field_name, value in metrics.items():
                if value is None:
                    bump(metric_labels[field_name])
        return sorted(gaps.values(), key=lambda item: (-item["count"], item["gap"]))


class ReflectionAgent:
    def __init__(
        self,
        llm: DeepSeekLLM,
        retriever: LocalEvidenceRetriever,
        discovery_tool: DiscoveryEvidenceTool,
        price_cache: StructuredPriceCache | None = None,
        graph_evidence: GraphMarketEvidence | None = None,
        volume_estimation_llm: DeepSeekLLM | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._discovery_tool = discovery_tool
        self._search_planner = ReflectionSearchPlanner()
        self._price_cache = price_cache
        self._graph_evidence = graph_evidence
        self._volume_estimation_llm = volume_estimation_llm

    def reflect(self, document: ResearchGoalDocument, hypothesis: Hypothesis) -> tuple[Hypothesis, int]:
        use_material_path = not document.evaluation_criteria and document.research_mode == "materials_opportunity"
        price_document = None
        if use_material_path and self._price_cache is not None:
            try:
                price_document = self._price_cache.ensure_fresh()
            except Exception:
                LOGGER.exception("Structured price cache unavailable during reflection for %s", hypothesis.hypothesis_id)
        local_rows = self._retriever.retrieve_for_hypothesis(document, hypothesis)
        evidence_rows = self._augment_evidence_rows(document, hypothesis, local_rows, price_document)
        stale = self._retriever.is_stale(local_rows, document.preferred_evidence_recency_days)
        initial_review = self._review(document, hypothesis, evidence_rows)

        discovery_run_ids: list[str] = []
        search_queries: list[str] = []

        if stale or initial_review.needs_additional_search:
            candidate_queries = self._search_planner.plan(
                document,
                hypothesis,
                initial_review.assessment,
                initial_review.follow_up_search_queries,
            )
            for query in candidate_queries[: document.reflection_search_limits.max_reflection_searches_per_hypothesis]:
                summary = self._discovery_tool.run(query, document.reflection_search_limits)
                discovery_run_ids.append(summary.run_id)
                search_queries.append(query)
            if search_queries:
                refreshed_rows = self._retriever.retrieve_for_hypothesis(document, hypothesis)
                evidence_rows = self._augment_evidence_rows(document, hypothesis, refreshed_rows, price_document)

        if use_material_path:
            evidence_rows = self._augment_with_ai_market_volume_estimate(document, hypothesis, evidence_rows)

        final_review = self._review(document, hypothesis, evidence_rows)
        assessment = final_review.assessment
        if use_material_path:
            assessment = self._merge_price_metrics(
                assessment,
                hypothesis,
                price_document,
            )
        assessment = self._finalize_missing_assessment_fields(document, hypothesis, assessment, evidence_rows)
        assessment = assessment.model_copy(
            update={
                "reflection_search_queries": search_queries,
                "reflection_discovery_run_ids": discovery_run_ids,
                "evidence_gap_notes": final_review.assessment.evidence_gap_notes,
            }
        )
        reflected = hypothesis.model_copy(
            update={
                "status": "reflected",
                "reflection_assessment": assessment,
            }
        )
        return reflected, len(discovery_run_ids)

    def _augment_evidence_rows(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        evidence_rows: list[dict[str, Any]],
        price_document,
    ) -> list[dict[str, Any]]:
        augmented: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for row in evidence_rows:
            row_id = str(row.get("id", ""))
            if row_id:
                augmented[row_id] = row
        if not document.evaluation_criteria and document.research_mode == "materials_opportunity":
            for row in self._graph_market_rows(document, hypothesis):
                augmented[str(row["id"])] = row
            for row in self._price_rows(hypothesis, price_document):
                augmented[str(row["id"])] = row
        return list(augmented.values())

    def _price_rows(
        self,
        hypothesis: Hypothesis,
        price_document,
    ) -> list[dict[str, Any]]:
        if self._price_cache is None or price_document is None:
            return []
        return self._price_cache.build_price_evidence_rows(
            hypothesis.incumbent_material,
            hypothesis.next_best_competitive_alternative,
            hypothesis.candidate_material,
            document=price_document,
        )

    def _graph_market_rows(self, document: ResearchGoalDocument, hypothesis: Hypothesis) -> list[dict[str, Any]]:
        if self._graph_evidence is None:
            return []
        try:
            return self._graph_evidence.build_evidence_rows(document, hypothesis)
        except Exception:
            LOGGER.exception("Offline graph market evidence unavailable during reflection for %s", hypothesis.hypothesis_id)
            return []

    def _augment_with_ai_market_volume_estimate(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        evidence_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._volume_estimation_llm is None:
            return evidence_rows
        if not self._needs_ai_market_volume_estimate(hypothesis, evidence_rows):
            return evidence_rows
        try:
            estimate = self._estimate_market_volume(document, hypothesis, evidence_rows)
            if estimate.total_substrate_volume_value is None and not estimate.material_volumes:
                return evidence_rows
            from bmscientist.graph_enrichment import GraphEnrichmentStore

            GraphEnrichmentStore().write_ai_market_volume_estimate(hypothesis, estimate)
            return self._merge_evidence_rows(evidence_rows, self._market_volume_estimate_rows(hypothesis, estimate))
        except Exception:
            LOGGER.exception("AI market-volume estimate failed for hypothesis %s", hypothesis.hypothesis_id)
            return evidence_rows

    @staticmethod
    def _needs_ai_market_volume_estimate(hypothesis: Hypothesis, evidence_rows: list[dict[str, Any]]) -> bool:
        if not (hypothesis.application and hypothesis.market_segment):
            return False
        graph_rows = [
            row
            for row in evidence_rows
            if row.get("metadata", {}).get("source_type") == "offline-graph-market-data"
        ]
        if not graph_rows:
            return False
        if any(row.get("metadata", {}).get("volume_value") is not None for row in graph_rows):
            return False
        return any(
            row.get("metadata", {}).get(key) is not None
            for row in graph_rows
            for key in ("revenue_value", "forecast_revenue_value", "cagr_value", "price_value")
        )

    def _estimate_market_volume(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        evidence_rows: list[dict[str, Any]],
    ) -> MarketVolumeEstimateOutput:
        evidence_payload = [
            {
                "chunk_id": row.get("id"),
                "source_url": row.get("source_url"),
                "source_title": row.get("source_title"),
                "application": row.get("application"),
                "incumbent_material": row.get("incumbent_material"),
                "candidate_materials": row.get("candidate_materials"),
                "metadata": row.get("metadata", {}),
                "excerpt": str(row.get("chunk_text", ""))[:1000],
            }
            for row in evidence_rows[:24]
        ]
        system_prompt = (
            "You estimate annual material volumes for market/application combinations. "
            "Use careful quantitative reasoning, but return strict JSON only. "
            "Prefer conservative estimates. Do not invent source URLs; cite only source rows supplied by the user. "
            "If you infer volume from revenue, explain the price and conversion assumptions. "
            "Use metric_tons_per_year for volume units where possible."
        )
        user_prompt = (
            "Estimate the current annual substrate/material volume for this hypothesis's market/application. "
            "Also estimate material-level volume shares for the incumbent and relevant alternatives when evidence supports it.\n\n"
            f"Research goal:\n{document.model_dump_json(indent=2)}\n\n"
            f"Hypothesis:\n{hypothesis.model_dump_json(indent=2)}\n\n"
            f"Evidence rows:\n{json.dumps(evidence_payload, indent=2)}\n\n"
            "Return JSON with market_name, application_name, total_substrate_volume_value, "
            "total_substrate_volume_unit, volume_year, revenue_value, revenue_unit, revenue_year, "
            "assumed_average_price_value, assumed_average_price_unit, material_volumes, confidence, rationale, "
            "and source_citations. Keep confidence medium unless there is direct source evidence."
        )
        return self._volume_estimation_llm.complete_json(
            MarketVolumeEstimateOutput,
            system_prompt,
            user_prompt,
            temperature=0.0,
        )

    @staticmethod
    def _market_volume_estimate_rows(hypothesis: Hypothesis, estimate: MarketVolumeEstimateOutput) -> list[dict[str, Any]]:
        source_url = ReflectionAgent._estimate_source_url(hypothesis, estimate)
        source_title = "AI generated market volume estimate"
        retrieved_at = datetime.now(timezone.utc).isoformat()
        market_name = estimate.market_name or hypothesis.market_segment
        application_name = estimate.application_name or hypothesis.application
        rows: list[dict[str, Any]] = []
        if estimate.total_substrate_volume_value is not None:
            rows.append(
                {
                    "id": f"ai-volume:{hypothesis.hypothesis_id}:total",
                    "source_url": source_url,
                    "source_title": source_title,
                    "application": application_name,
                    "incumbent_material": hypothesis.incumbent_material,
                    "candidate_materials": [],
                    "relevance_score": 0.86,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"AI generated estimate: {market_name} / {application_name} total substrate volume is "
                        f"{estimate.total_substrate_volume_value:g} {estimate.total_substrate_volume_unit} "
                        f"in {estimate.volume_year or 'the current period'}. {estimate.rationale}"
                    )[:1800],
                    "metadata": {
                        "source_type": "offline-graph-market-data",
                        "source_node_type": "ai_volume_estimate",
                        "edge_type": "Market_HAS_APPLICATION_Application",
                        "market_name": market_name,
                        "target_name": application_name,
                        "volume_value": estimate.total_substrate_volume_value,
                        "volume_unit": estimate.total_substrate_volume_unit,
                        "volume_year": estimate.volume_year,
                        "revenue_value": estimate.revenue_value,
                        "revenue_year": estimate.revenue_year,
                        "unit": estimate.revenue_unit,
                        "is_inferred": True,
                    },
                }
            )
        for item in estimate.material_volumes:
            if item.volume_value is None:
                continue
            rows.append(
                {
                    "id": f"ai-volume:{hypothesis.hypothesis_id}:{item.material_name.lower().replace(' ', '-')}",
                    "source_url": source_url,
                    "source_title": source_title,
                    "application": application_name,
                    "incumbent_material": hypothesis.incumbent_material,
                    "candidate_materials": [item.material_name],
                    "relevance_score": 0.86,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"AI generated estimate: {item.material_name} volume in {application_name} is "
                        f"{item.volume_value:g} {item.volume_unit} "
                        f"({item.share_of_total:.1%} of total substrate volume). {item.rationale}"
                        if item.share_of_total is not None
                        else f"AI generated estimate: {item.material_name} volume in {application_name} is "
                        f"{item.volume_value:g} {item.volume_unit}. {item.rationale}"
                    )[:1800],
                    "metadata": {
                        "source_type": "offline-graph-market-data",
                        "source_node_type": "ai_volume_estimate",
                        "edge_type": "Product_USED_IN_Application",
                        "market_name": market_name,
                        "target_name": application_name,
                        "volume_value": item.volume_value,
                        "volume_unit": item.volume_unit,
                        "volume_year": estimate.volume_year,
                        "is_inferred": True,
                    },
                }
            )
        return rows

    @staticmethod
    def _estimate_source_url(hypothesis: Hypothesis, estimate: MarketVolumeEstimateOutput) -> str:
        for citation in estimate.source_citations:
            if citation.source_url:
                return citation.source_url
        return f"coscientist://research/{hypothesis.research_id}/hypothesis/{hypothesis.hypothesis_id}"

    @staticmethod
    def _merge_evidence_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for row in [*existing_rows, *new_rows]:
            row_id = str(row.get("id") or "")
            if row_id:
                merged[row_id] = row
        return list(merged.values())

    def _review(self, document: ResearchGoalDocument, hypothesis: Hypothesis, evidence_rows: list[dict[str, Any]]) -> ReflectionReviewOutput:
        if document.evaluation_criteria or document.research_mode != "materials_opportunity":
            return self._review_generic_criteria(document, hypothesis, evidence_rows)
        technical = self._review_category(document, hypothesis, evidence_rows, "technical")
        commercial = self._review_category(document, hypothesis, evidence_rows, "commercial")
        strategic = self._review_category(document, hypothesis, evidence_rows, "strategic")
        return ReflectionReviewOutput(
            assessment=self._merge_assessments([technical.assessment, commercial.assessment, strategic.assessment]),
            needs_additional_search=technical.needs_additional_search
            or commercial.needs_additional_search
            or strategic.needs_additional_search,
            follow_up_search_queries=list(
                OrderedDict.fromkeys(
                    technical.follow_up_search_queries
                    + commercial.follow_up_search_queries
                    + strategic.follow_up_search_queries
                )
            ),
        )

    def _review_generic_criteria(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        evidence_rows: list[dict[str, Any]],
    ) -> ReflectionReviewOutput:
        criteria = document.evaluation_criteria or self._default_generic_criteria(document, hypothesis)
        if not criteria:
            return ReflectionReviewOutput(assessment=ReflectionAssessment(), needs_additional_search=False, follow_up_search_queries=[])

        reviews: list[ReflectionReviewOutput] = []
        batch_size = 3
        for offset in range(0, len(criteria), batch_size):
            batch = criteria[offset : offset + batch_size]
            evidence_payload = [
                {
                    "chunk_id": row.get("id"),
                    "source_url": row.get("source_url"),
                    "source_title": row.get("source_title"),
                    "application": row.get("application"),
                    "incumbent_material": row.get("incumbent_material"),
                    "candidate_materials": row.get("candidate_materials"),
                    "metadata": row.get("metadata", {}),
                    "relevance_score": row.get("relevance_score"),
                    "retrieved_at": row.get("retrieved_at"),
                    "excerpt": str(row.get("chunk_text", ""))[:600],
                }
                for row in evidence_rows[:30]
            ]
            system_prompt = PROMPTS.render("reflection_agent", "review_criteria.system")
            user_prompt = PROMPTS.render(
                "reflection_agent",
                "review_criteria.user",
                document_json=document.model_dump_json(indent=2),
                hypothesis_json=hypothesis.model_dump_json(indent=2),
                evidence_payload_json=json.dumps(evidence_payload, indent=2),
                criteria_json=json.dumps([criterion.model_dump() for criterion in batch], indent=2),
                tool_requests_json=json.dumps([request.model_dump() for request in document.tool_requests], indent=2),
            )
            reviews.append(self._llm.complete_json(ReflectionReviewOutput, system_prompt, user_prompt))
        return ReflectionReviewOutput(
            assessment=self._merge_assessments([review.assessment for review in reviews]),
            needs_additional_search=any(review.needs_additional_search for review in reviews),
            follow_up_search_queries=list(
                OrderedDict.fromkeys(
                    query
                    for review in reviews
                    for query in review.follow_up_search_queries
                    if query
                )
            ),
        )

    def _review_category(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        evidence_rows: list[dict[str, Any]],
        category: str,
    ) -> ReflectionReviewOutput:
        evidence_payload = [
            {
                "chunk_id": row.get("id"),
                "source_url": row.get("source_url"),
                "source_title": row.get("source_title"),
                "application": row.get("application"),
                "incumbent_material": row.get("incumbent_material"),
                "candidate_materials": row.get("candidate_materials"),
                "metadata": row.get("metadata", {}),
                "relevance_score": row.get("relevance_score"),
                "retrieved_at": row.get("retrieved_at"),
                "excerpt": str(row.get("chunk_text", ""))[:600],
            }
            for row in evidence_rows[:30]
        ]
        category_fields = {
            "technical": [
                "replacement_fit_score",
                "activation_ease_score",
                "technical_success_probability",
            ],
            "commercial": [
                "market_size_score",
                "incumbent_price_usd_per_kg",
                "nbca_material",
                "nbca_price_usd_per_kg",
                "commercial_success_probability",
            ],
            "strategic": [
                "strategic_fit_score",
                "replacement_driver_strength_score",
            ],
        }
        system_prompt = PROMPTS.render("reflection_agent", "review_category.system")
        user_prompt = PROMPTS.render(
            "reflection_agent",
            "review_category.user",
            document_json=document.model_dump_json(indent=2),
            hypothesis_json=hypothesis.model_dump_json(indent=2),
            evidence_payload_json=json.dumps(evidence_payload, indent=2),
            category=category,
            focus_fields_json=json.dumps(category_fields[category], indent=2),
        )
        return self._llm.complete_json(ReflectionReviewOutput, system_prompt, user_prompt)

    def _merge_price_metrics(
        self,
        assessment: ReflectionAssessment,
        hypothesis: Hypothesis,
        price_document,
    ) -> ReflectionAssessment:
        if self._price_cache is None:
            return assessment
        incumbent_metric = self._price_cache.metric_for_material(hypothesis.incumbent_material, document=price_document)
        nbca_name = assessment.nbca_material or hypothesis.next_best_competitive_alternative
        nbca_metric = self._price_cache.metric_for_material(nbca_name, document=price_document)
        updates: dict[str, Any] = {}
        if assessment.incumbent_price_usd_per_kg.value is None and incumbent_metric is not None:
            updates["incumbent_price_usd_per_kg"] = incumbent_metric
        if assessment.nbca_price_usd_per_kg.value is None and nbca_metric is not None:
            updates["nbca_price_usd_per_kg"] = nbca_metric
        if not assessment.nbca_material and nbca_name:
            updates["nbca_material"] = nbca_name
        if not updates:
            return assessment
        return assessment.model_copy(update=updates)

    def _finalize_missing_assessment_fields(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        assessment: ReflectionAssessment,
        evidence_rows: list[dict[str, Any]],
    ) -> ReflectionAssessment:
        if document.evaluation_criteria or document.research_mode != "materials_opportunity":
            return self._finalize_generic_assessment_fields(document, hypothesis, assessment)
        updates: dict[str, Any] = {}
        metric_fields = [
            "strategic_fit_score",
            "market_size_score",
            "replacement_fit_score",
            "activation_ease_score",
            "replacement_driver_strength_score",
            "technical_success_probability",
            "commercial_success_probability",
        ]
        graph_rows = [row for row in evidence_rows if row.get("metadata", {}).get("source_type") == "offline-graph-market-data"]
        working_assessment = assessment
        for field_name in metric_fields:
            metric = getattr(working_assessment, field_name)
            if metric.value is None:
                inferred_metric = self._inferred_metric(field_name, document, hypothesis, working_assessment, graph_rows)
                updates[field_name] = inferred_metric
                working_assessment = working_assessment.model_copy(update={field_name: inferred_metric})
        if assessment.incumbent_price_usd_per_kg.value is None:
            updates["incumbent_price_usd_per_kg"] = self._inferred_price_metric(hypothesis.incumbent_material)
        if assessment.nbca_price_usd_per_kg.value is None:
            nbca_name = assessment.nbca_material or hypothesis.next_best_competitive_alternative
            updates["nbca_price_usd_per_kg"] = self._inferred_price_metric(nbca_name)
        if not assessment.nbca_material and hypothesis.next_best_competitive_alternative:
            updates["nbca_material"] = hypothesis.next_best_competitive_alternative
        if not updates:
            return assessment
        return assessment.model_copy(update=updates)

    @staticmethod
    def _finalize_generic_assessment_fields(
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        assessment: ReflectionAssessment,
    ) -> ReflectionAssessment:
        if assessment.criterion_results:
            return assessment
        criteria = document.evaluation_criteria
        if not criteria:
            return assessment
        results: list[CandidateEvaluationResult] = []
        for criterion in criteria:
            results.append(
                CandidateEvaluationResult(
                    criterion_name=criterion.name,
                    value=None,
                    normalized_score=0.5 if criterion.direction == "describe" else None,
                    confidence=0.1,
                    rationale=(
                        f"No direct evidence resolved `{criterion.name}` for candidate "
                        f"`{hypothesis.title}` during reflection."
                    ),
                    evidence_mode="mixed",
                    is_inferred=True,
                )
            )
        return assessment.model_copy(update={"criterion_results": results})

    @staticmethod
    def _inferred_metric(
        field_name: str,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        assessment: ReflectionAssessment,
        graph_rows: list[dict[str, Any]],
    ) -> AssessmentMetric:
        graph_ids = [str(row.get("id")) for row in graph_rows[:4] if row.get("id")]
        graph_urls = [str(row.get("source_url")) for row in graph_rows[:4] if row.get("source_url")]
        graph_stats = ReflectionAgent._graph_market_stats(graph_rows)
        confidence = 0.35 + (0.15 if graph_rows else 0.0)
        value = 0.5
        rationale = "Inferred from the hypothesis structure and research configuration after reflection evidence remained incomplete."

        if field_name == "strategic_fit_score":
            value = 0.5
            if hypothesis.candidate_material and hypothesis.candidate_material in document.preferred_candidate_materials:
                value += 0.12
            if hypothesis.incumbent_material and hypothesis.incumbent_material in document.target_incumbent_materials:
                value += 0.1
            if hypothesis.substitution_drivers:
                value += 0.08
            if hypothesis.strategic_rationale:
                value += 0.07
            rationale = "Inferred from candidate/incumbent alignment, substitution drivers, and stated strategic rationale."
        elif field_name == "market_size_score":
            value = graph_stats["market_score"] if graph_rows else 0.45
            rationale = (
                "Inferred from offline graph market revenue, forecast revenue, and CAGR signals."
                if graph_rows
                else "Inferred conservatively because no usable market-size evidence was available after reflection search."
            )
        elif field_name == "replacement_fit_score":
            value = 0.45 + (0.12 if hypothesis.application_requirements else 0.0) + (0.1 if hypothesis.conversion_process else 0.0)
            if hypothesis.candidate_form and hypothesis.incumbent_form and hypothesis.candidate_form.lower() == hypothesis.incumbent_form.lower():
                value += 0.1
            rationale = "Inferred from available requirements, material forms, and conversion-process detail."
        elif field_name == "activation_ease_score":
            value = 0.5 + (0.12 if hypothesis.conversion_process else 0.0)
            if document.opportunity_speed_horizon_months and document.opportunity_speed_horizon_months <= 6:
                value += 0.1
            if any("drop" in item.lower() for item in document.opportunity_modes):
                value += 0.08
            rationale = "Inferred from process specificity and the configured speed/drop-in opportunity horizon."
        elif field_name == "replacement_driver_strength_score":
            value = 0.45 + min(0.25, 0.07 * len(hypothesis.substitution_drivers))
            if document.recycling_or_sustainability_angles:
                value += 0.08
            rationale = "Inferred from substitution drivers and sustainability or recycling angles in the research configuration."
        elif field_name == "technical_success_probability":
            replacement_fit = assessment.replacement_fit_score.value
            value = replacement_fit if replacement_fit is not None else 0.55
            rationale = "Inferred from replacement-fit signals where direct technical validation was incomplete."
        elif field_name == "commercial_success_probability":
            components = [
                assessment.strategic_fit_score.value,
                assessment.market_size_score.value,
                assessment.replacement_driver_strength_score.value,
                assessment.activation_ease_score.value,
            ]
            present = [item for item in components if item is not None]
            value = sum(present) / len(present) if present else graph_stats["market_score"] if graph_rows else 0.5
            rationale = "Inferred from the available strategic, market, driver, and activation signals."

        return AssessmentMetric(
            value=max(0.0, min(1.0, round(value, 3))),
            rationale=rationale,
            confidence=min(0.65, confidence),
            citation_chunk_ids=graph_ids,
            citation_urls=list(OrderedDict.fromkeys(graph_urls)),
            is_inferred=True,
        )

    @staticmethod
    def _graph_market_stats(graph_rows: list[dict[str, Any]]) -> dict[str, float]:
        revenues = [
            row.get("metadata", {}).get("revenue_value")
            for row in graph_rows
            if row.get("metadata", {}).get("revenue_value") is not None
        ]
        forecasts = [
            row.get("metadata", {}).get("forecast_revenue_value")
            for row in graph_rows
            if row.get("metadata", {}).get("forecast_revenue_value") is not None
        ]
        cagrs = [
            row.get("metadata", {}).get("cagr_value")
            for row in graph_rows
            if row.get("metadata", {}).get("cagr_value") is not None
        ]
        revenue = max([float(item) for item in revenues + forecasts], default=0.0)
        cagr = max([float(item) for item in cagrs], default=0.0)
        revenue_score = 0.35
        if revenue >= 5000:
            revenue_score = 0.85
        elif revenue >= 1000:
            revenue_score = 0.72
        elif revenue >= 250:
            revenue_score = 0.58
        elif revenue > 0:
            revenue_score = 0.45
        cagr_bonus = 0.08 if cagr >= 8 else 0.04 if cagr >= 4 else 0.0
        return {"market_score": min(0.92, revenue_score + cagr_bonus)}

    @staticmethod
    def _inferred_price_metric(material_name: str | None) -> PriceMetric:
        if not material_name:
            return PriceMetric(
                value=2.0,
                rationale="No material name was available for price lookup; populated with a broad polymer reference placeholder.",
                confidence=0.05,
                citation_chunk_ids=[],
                citation_urls=[],
                is_inferred=True,
            )
        normalized = material_name.lower()
        reference_prices = [
            (("petg",), 2.4),
            (("pet", "polyethylene terephthalate", "apet", "rpet"), 1.7),
            (("pvc", "polyvinyl chloride"), 1.35),
            (("abs",), 2.6),
            (("polycarbonate", " pc"), 3.2),
            (("polypropylene", " pp"), 1.15),
            (("hdpe", "high density polyethylene"), 1.2),
            (("ldpe", "low density polyethylene", "lldpe"), 1.25),
            (("polystyrene", " ps", "hips", "gpps"), 1.55),
            (("acrylic", "pmma"), 2.5),
            (("polyamide", "nylon", " pa"), 2.8),
        ]
        value = next((price for aliases, price in reference_prices if any(alias in normalized for alias in aliases)), 2.0)
        return PriceMetric(
            value=value,
            rationale=(
                f"Inferred broad USD/kg reference for {material_name} because no structured price evidence "
                "resolved after local cache lookup and reflection search."
            ),
            confidence=0.25,
            citation_chunk_ids=[],
            citation_urls=[],
            is_inferred=True,
        )

    @staticmethod
    def _merge_assessments(assessments: list[ReflectionAssessment]) -> ReflectionAssessment:
        merged = ReflectionAssessment()
        metric_fields = [
            "strategic_fit_score",
            "market_size_score",
            "replacement_fit_score",
            "activation_ease_score",
            "replacement_driver_strength_score",
            "technical_success_probability",
            "commercial_success_probability",
        ]
        price_fields = ["incumbent_price_usd_per_kg", "nbca_price_usd_per_kg"]
        for field_name in metric_fields + price_fields:
            current = getattr(merged, field_name)
            best = ReflectionAgent._best_metric([getattr(assessment, field_name) for assessment in assessments], current)
            merged = merged.model_copy(update={field_name: best})
        merged = merged.model_copy(
            update={
                "nbca_material": next((assessment.nbca_material for assessment in assessments if assessment.nbca_material), None),
                "criterion_results": ReflectionAgent._merge_criterion_results(assessments),
                "tool_request_notes": list(
                    OrderedDict.fromkeys(
                        note for assessment in assessments for note in assessment.tool_request_notes if note
                    )
                ),
                "evidence_gap_notes": list(
                    OrderedDict.fromkeys(
                        note for assessment in assessments for note in assessment.evidence_gap_notes if note
                    )
                ),
            }
        )
        return merged

    @staticmethod
    def _best_metric(metrics: list[AssessmentMetric | PriceMetric], default: AssessmentMetric | PriceMetric):
        supported = [
            metric
            for metric in metrics
            if metric.value is not None or metric.rationale or metric.citation_chunk_ids or metric.citation_urls
        ]
        if not supported:
            return default
        return max(supported, key=lambda metric: (metric.value is not None, metric.confidence, bool(metric.citation_chunk_ids)))

    @staticmethod
    def _merge_criterion_results(assessments: list[ReflectionAssessment]) -> list[CandidateEvaluationResult]:
        merged: "OrderedDict[str, CandidateEvaluationResult]" = OrderedDict()
        for assessment in assessments:
            for result in assessment.criterion_results:
                existing = merged.get(result.criterion_name)
                if existing is None or ReflectionAgent._criterion_result_rank(result) > ReflectionAgent._criterion_result_rank(existing):
                    merged[result.criterion_name] = result
        return list(merged.values())

    @staticmethod
    def _criterion_result_rank(result: CandidateEvaluationResult) -> tuple[bool, float, bool]:
        return (
            result.normalized_score is not None,
            result.confidence,
            bool(result.citation_chunk_ids),
        )

    @staticmethod
    def _default_generic_criteria(
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
    ) -> list[EvaluationCriterion]:
        primary_field = document.candidate_artifact_schema.primary_identifier_field
        return [
            EvaluationCriterion(
                name="overall_fit",
                description="Overall fit of the candidate to the stated research goal.",
                direction="maximize",
                required_candidate_fields=[primary_field] if primary_field else [],
                suggested_search_queries=[f"{hypothesis.title} evidence"],
                reflection_guidance=["Check whether the candidate plausibly addresses the goal without obvious fatal flaws."],
            )
        ]

    @staticmethod
    def _fallback_search_queries(document: ResearchGoalDocument, hypothesis: Hypothesis) -> list[str]:
        region_text = " ".join(document.regions)
        return [
            f"{hypothesis.application or ''} {hypothesis.incumbent_material or ''} market size {region_text}".strip(),
            f"{hypothesis.application or ''} {hypothesis.incumbent_material or ''} price usd kg {region_text}".strip(),
            f"{hypothesis.application or ''} {hypothesis.candidate_material or ''} replacement fit requirements".strip(),
            f"{hypothesis.application or ''} {hypothesis.candidate_material or ''} drop in replacement lead time".strip(),
        ]


class CoScientistRunner:
    DEFAULT_AGENTIC_LOOP_SAFETY_CAP = 12

    def __init__(
        self,
        config: AppConfig,
        llm: DeepSeekLLM | None = None,
        evidence_store: LanceEvidenceStore | None = None,
        embedder: LocalEmbedder | None = None,
        discovery_agent: DiscoveryAgent | None = None,
        artifact_store: CoScientistStore | None = None,
    ):
        self._progress_reporter: ProgressReporter = NullProgressReporter()
        self._config = config
        self._config.ensure_directories()
        self._cost_tracker = CostTracker(config)
        self._cost_tracker_seeded_research_ids: set[str] = set()
        self._llm = llm or DeepSeekLLM(
            config,
            request_profile=config.chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="chat",
        )
        self._planning_llm = DeepSeekLLM(
            config,
            model=config.planning_chat_model,
            request_profile=config.planning_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="planning",
        )
        self._generation_llm = DeepSeekLLM(
            config,
            model=config.generation_chat_model,
            request_profile=config.generation_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="generation",
        )
        self._reflection_llm = DeepSeekLLM(
            config,
            model=config.reflection_chat_model,
            request_profile=config.reflection_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="reflection",
        )
        self._market_volume_llm = DeepSeekLLM(
            config,
            model=config.market_volume_estimation_chat_model or config.reflection_chat_model,
            request_profile=config.market_volume_estimation_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="market_volume_estimation",
        )
        self._ranking_llm = DeepSeekLLM(
            config,
            model=config.ranking_chat_model,
            request_profile=config.ranking_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="ranking",
        )
        self._evolution_llm = DeepSeekLLM(
            config,
            model=config.evolution_chat_model,
            request_profile=config.evolution_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="evolution",
        )
        self._proximity_llm = DeepSeekLLM(
            config,
            model=config.proximity_chat_model,
            request_profile=config.proximity_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="proximity",
        )
        self._meta_review_llm = DeepSeekLLM(
            config,
            model=config.meta_review_chat_model,
            request_profile=config.meta_review_chat_profile,
            cost_tracker=self._cost_tracker,
            client_name="meta_review",
        )
        if evidence_store is None:
            from bmscientist.store import LanceEvidenceStore

            self._evidence_store = LanceEvidenceStore(config.resolved_lancedb_path())
        else:
            self._evidence_store = evidence_store
        if embedder is None:
            from bmscientist.embeddings import LocalEmbedder

            self._embedder = LocalEmbedder(config)
        else:
            self._embedder = embedder
        self._discovery_agent = discovery_agent
        self._artifact_store = artifact_store or CoScientistStore()
        self._retriever = LocalEvidenceRetriever(self._evidence_store, self._embedder)
        self._price_cache = StructuredPriceCache(config)
        self._graph_evidence = GraphMarketEvidence()
        self._manual_ingestor = ManualEvidenceIngestor(
            config,
            EvidenceClassifier(self._reflection_llm),
            TextChunker(),
            self._embedder,
            self._evidence_store,
        )
        self._manual_ingestor.ingest_pending_files()
        try:
            self._price_cache.ensure_fresh()
        except Exception:
            LOGGER.exception("Initial structured price cache refresh failed")
        self._planning_agent = ResearchPlanningAgent(self._planning_llm)
        self._generation_agent = GenerationAgent(self._generation_llm, self._retriever, self._graph_evidence)
        self._ranking_agent = RankingAgent(self._ranking_llm)
        self._evolution_agent = EvolutionAgent(self._evolution_llm)
        self._proximity_agent = ProximityCheckAgent(self._proximity_llm)
        self._meta_review_agent = MetaReviewAgent(self._meta_review_llm)
        self._final_portfolio_agent = FinalPortfolioAgent(self._meta_review_llm)
        discovery_tool = (
            DiscoveryEvidenceTool(discovery_agent)
            if discovery_agent is not None
            else DiscoveryEvidenceTool(
                config,
                self._reflection_llm,
                self._embedder,
                self._evidence_store,
                cost_tracker=self._cost_tracker,
            )
        )
        self._reflection_agent = ReflectionAgent(
            self._reflection_llm,
            self._retriever,
            discovery_tool,
            self._price_cache,
            self._graph_evidence,
            self._market_volume_llm,
        )

    def set_progress_reporter(self, reporter: ProgressReporter | None) -> None:
        self._progress_reporter = reporter or NullProgressReporter()

    def prepare_project_name(self, preferred_name: str | None = None) -> str:
        return self._artifact_store.claim_project_name(preferred_name)

    def reflect_hypothesis(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        persist: bool = False,
    ) -> tuple[Hypothesis, int]:
        reflected, discovery_runs = self._reflection_agent.reflect(document, hypothesis)
        if persist:
            self._artifact_store.append_hypothesis_snapshot(reflected)
        return reflected, discovery_runs

    def run(
        self,
        goal: str,
        target_hypotheses: int,
        project_name: str | None = None,
        regions: list[str] | None = None,
        strategic_fit_notes: str | None = None,
        preferred_evidence_recency_days: int = 180,
        max_reflection_searches_per_hypothesis: int = 3,
        results_per_query: int = 5,
        max_pages_per_search: int = 8,
        reflection_concurrency: int = 3,
        proximity_merge_mode: str = "balanced",
        proximity_granularity: str = "application_family",
        spawn_reflection_daemons: bool = False,
    ) -> CoScientistRunResult:
        limits = ReflectionSearchLimits(
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
        research_id = project_name or self.prepare_project_name()
        self._seed_cost_tracking_for_research(research_id)
        self._progress_reporter.start("planning", "Processing goal")
        document = self._planning_agent.create_research_goal(
            research_id=research_id,
            raw_goal=goal,
            target_hypotheses_final=target_hypotheses,
            regions=regions,
            strategic_fit_notes=strategic_fit_notes,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            reflection_search_limits=limits,
            proximity_merge_policy=ProximityMergePolicy(
                merge_mode=proximity_merge_mode,
                granularity=proximity_granularity,
            ),
        )
        research_goal_path = self._artifact_store.save_research_goal(document)
        self._progress_reporter.complete("planning", "Goal processed")

        generated_by_id: "OrderedDict[str, Hypothesis]" = OrderedDict()
        generated_lock = Lock()
        stop_reflection_monitor: Callable[[], None] | None = None
        if spawn_reflection_daemons:
            batch_size_for = getattr(self._generation_agent, "batch_size_for", None)
            generation_batch_size = (
                batch_size_for(document.target_hypotheses_generated)
                if callable(batch_size_for)
                else self._generation_agent.batch_size
            )
            self._spawn_reflection_daemons(
                research_id=document.research_id,
                worker_count=min(max(1, generation_batch_size), document.target_hypotheses_generated),
                preferred_evidence_recency_days=preferred_evidence_recency_days,
                max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
                results_per_query=results_per_query,
                max_pages_per_search=max_pages_per_search,
                idle_exit_after_seconds=max(300, self._config.request_timeout_seconds * 3 + 15),
            )

        def persist_generated_batch(batch: list[Hypothesis]) -> None:
            with generated_lock:
                for hypothesis in batch:
                    self._artifact_store.append_hypothesis_snapshot(hypothesis)
                    generated_by_id.setdefault(hypothesis.hypothesis_id, hypothesis)

        self._progress_reporter.start("generation", "Generating ideas", document.target_hypotheses_generated)
        if spawn_reflection_daemons:
            stop_reflection_monitor = self._start_reflection_progress_monitor(
                research_id=document.research_id,
                tracked_hypothesis_ids=lambda: self._tracked_hypothesis_ids(generated_by_id, generated_lock),
                phase="reflection",
                progress_message="Reflecting on ideas",
            )
        generated = self._generation_agent.generate(
            document,
            on_progress=lambda completed, total: self._progress_reporter.advance(
                "generation",
                "Generating ideas",
                completed,
                total,
            ),
            on_batch=persist_generated_batch,
        )
        self._progress_reporter.complete(
            "generation",
            "Ideas generated",
            len(generated),
            document.target_hypotheses_generated,
        )
        persisted_ids = {
            hypothesis.hypothesis_id for hypothesis in self._artifact_store.latest_hypotheses(document.research_id)
        }
        for hypothesis in generated:
            with generated_lock:
                generated_by_id.setdefault(hypothesis.hypothesis_id, hypothesis)
            if hypothesis.hypothesis_id not in persisted_ids:
                self._artifact_store.append_hypothesis_snapshot(hypothesis)
                persisted_ids.add(hypothesis.hypothesis_id)

        if not spawn_reflection_daemons:
            generated = self._apply_pre_reflection_proximity(document, generated, round_index=0)

        if spawn_reflection_daemons:
            try:
                with generated_lock:
                    tracked_ids = set(generated_by_id.keys())
                reflected_hypotheses = self._wait_for_reflection_completion(
                    research_id=document.research_id,
                    hypothesis_ids=tracked_ids,
                    phase="reflection",
                    progress_message="Reflecting on ideas",
                    report_progress=False,
                )
                automatic_discovery_runs = self._automatic_discovery_runs_for_hypotheses(reflected_hypotheses)
                self._progress_reporter.complete(
                    "reflection",
                    "Reflecting on ideas complete",
                    len(reflected_hypotheses),
                    len(tracked_ids),
                )
            finally:
                if stop_reflection_monitor is not None:
                    stop_reflection_monitor()
        else:
            reflected_hypotheses, automatic_discovery_runs = self._reflect_and_append(
                document,
                generated,
                concurrency=reflection_concurrency,
                phase="reflection",
                progress_message="Reflecting on ideas",
            )

        self._progress_reporter.start("reporting", "Writing report")
        report_path = self._artifact_store.write_report(
            document.research_id,
            self._build_report(document, reflected_hypotheses),
        )
        self._write_tool_report(document, reflected_hypotheses)
        cost_path = self._write_cost_report(document.research_id)
        self._progress_reporter.complete("reporting", "Report written")
        return CoScientistRunResult(
            research_id=document.research_id,
            generated_hypotheses=len(generated),
            reflected_hypotheses=len(reflected_hypotheses),
            automatic_discovery_runs=automatic_discovery_runs,
            research_goal_path=str(research_goal_path.resolve()),
            hypothesis_path=str(self._artifact_store.hypothesis_path(document.research_id).resolve()),
            report_path=str(report_path.resolve()),
            cost_path=cost_path,
        )

    def run_loop(
        self,
        research_id: str,
        target_final_hypotheses: int | None = None,
        max_rounds: int | None = None,
        evolve_top_k: int = 5,
        evolved_per_round: int = 5,
        regenerated_per_round: int = 5,
        proximity_check_every: int = 1,
        max_synthesized_per_round: int = 3,
        promotion_score_threshold: float = 0.72,
        gap_overlap_threshold: float = 0.6,
        max_gap_persistence_rounds: int = 1,
        preferred_evidence_recency_days: int | None = None,
        max_reflection_searches_per_hypothesis: int | None = None,
        results_per_query: int | None = None,
        max_pages_per_search: int | None = None,
        reflection_concurrency: int = 3,
        proximity_merge_mode: str | None = None,
        proximity_granularity: str | None = None,
    ) -> CoScientistLoopResult:
        self._seed_cost_tracking_for_research(research_id)
        self._progress_reporter.start("loop_load", "Loading research run")
        document = self._artifact_store.load_research_goal(research_id)
        document = self._with_reflection_overrides(
            document=document,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
        updated_proximity_document = self._with_proximity_overrides(
            document=document,
            proximity_merge_mode=proximity_merge_mode,
            proximity_granularity=proximity_granularity,
        )
        if updated_proximity_document != document:
            document = updated_proximity_document
            self._artifact_store.save_research_goal(document)
        self._progress_reporter.complete("loop_load", "Research run loaded")
        
        # Self-healing: check if there are generated hypotheses in queue that need reflection
        generated_dir = self._artifact_store.hypotheses_dir(research_id) / "generated"
        generated_files = list(generated_dir.glob("*.json")) if generated_dir.exists() else []
        latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
        unreflected = [
            h for h in latest_hypotheses
            if h.status in ("generated", "generation", "reflecting")
        ]
        
        if generated_files or unreflected:
            self._progress_reporter.start(
                "reflection", 
                "Resuming reflection on queued ideas", 
                max(len(generated_files), len(unreflected))
            )
            self._reflect_from_queue(
                document=document,
                research_id=research_id,
                concurrency=reflection_concurrency,
                daemon=False,
                phase="reflection",
                progress_message="Reflecting on ideas"
            )
            # Reload latest hypotheses to reflect status changes
            latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)

        target_final = target_final_hypotheses or document.target_hypotheses_final
        rounds_completed = 0
        total_evolved = 0
        total_regenerated = 0
        total_synthesized = 0
        total_reflected = 0
        automatic_discovery_runs = 0
        round_limit = max_rounds if max_rounds is not None else self.DEFAULT_AGENTIC_LOOP_SAFETY_CAP
        stop_reason = "safety_round_limit_reached" if max_rounds is None else "max_rounds_reached"
        latest_ranking: RankingRound | None = None
        latest_meta_review: MetaReviewRound | None = None

        for round_index in range(1, round_limit + 1):
            latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
            reflected = [
                hypothesis
                for hypothesis in latest_hypotheses
                if hypothesis.status == "reflected" and hypothesis.is_active
            ]
            if not reflected:
                stop_reason = "no_active_reflected_hypotheses"
                break
            round_label = self._loop_round_label(round_index, max_rounds)
            self._progress_reporter.start(
                "ranking",
                f"Ranking ideas ({round_label})",
                len(reflected),
            )
            ranking_round, ranked_hypotheses = self._ranking_agent.rank(
                document=document,
                hypotheses=reflected,
                round_index=round_index,
                target_final_count=target_final,
                evolve_top_k=evolve_top_k,
            )
            self._progress_reporter.complete(
                "ranking",
                f"Ideas ranked ({round_label})",
                len(ranked_hypotheses),
                len(reflected),
            )
            latest_ranking = ranking_round
            self._artifact_store.append_ranking_round(ranking_round)
            ranked_hypotheses = self._apply_ranking_outcomes(ranked_hypotheses, ranking_round)
            for hypothesis in ranked_hypotheses:
                self._artifact_store.append_hypothesis_snapshot(hypothesis)
            rounds_completed += 1

            if proximity_check_every > 0 and round_index % proximity_check_every == 0:
                proximity_hypotheses = self._artifact_store.latest_hypotheses(research_id)
                self._progress_reporter.start(
                    "proximity",
                    f"Checking idea overlap ({round_label})",
                    len(proximity_hypotheses),
                )
                proximity_round, proximity_updates, synthesized_hypotheses = self._proximity_agent.review(
                    document=document,
                    hypotheses=proximity_hypotheses,
                    round_index=round_index,
                    max_synthesized_hypotheses=max_synthesized_per_round,
                )
                self._artifact_store.append_proximity_round(proximity_round)
                for hypothesis in proximity_updates:
                    self._artifact_store.append_hypothesis_snapshot(hypothesis)
                synthesized_hypotheses = self._dedupe_new_hypotheses(
                    synthesized_hypotheses,
                    existing_hypothesis_ids={
                        hypothesis.hypothesis_id
                        for hypothesis in self._artifact_store.latest_hypotheses(research_id)
                    },
                )
                self._progress_reporter.complete(
                    "proximity",
                    f"Idea overlap checked ({round_label})",
                    len(proximity_hypotheses),
                    len(proximity_hypotheses),
                )
                for hypothesis in synthesized_hypotheses:
                    self._artifact_store.append_hypothesis_snapshot(hypothesis)
                total_synthesized += len(synthesized_hypotheses)
                reflected_synthesized, run_count = self._reflect_and_append(
                    document,
                    synthesized_hypotheses,
                    concurrency=reflection_concurrency,
                    phase="synthesized_reflection",
                    progress_message=f"Reflecting on synthesized ideas ({round_label})",
                )
                total_reflected += len(reflected_synthesized)
                automatic_discovery_runs += run_count

            latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
            self._progress_reporter.start(
                "meta_review",
                f"Reviewing portfolio gaps ({round_label})",
            )
            document, meta_review_round = self._meta_review_agent.review(
                document=document,
                hypotheses=latest_hypotheses,
                ranking_round=ranking_round,
                round_index=round_index,
                gap_overlap_threshold=gap_overlap_threshold,
                max_gap_persistence_rounds=max_gap_persistence_rounds,
            )
            self._progress_reporter.complete(
                "meta_review",
                f"Portfolio gaps reviewed ({round_label})",
            )
            latest_meta_review = meta_review_round
            self._artifact_store.save_research_goal(document)
            self._artifact_store.append_meta_review_round(meta_review_round)

            promoted_scores = [
                ranking.score
                for ranking in ranking_round.rankings
                if ranking.hypothesis_id in ranking_round.promoted_hypothesis_ids
            ]
            if (
                len(ranking_round.promoted_hypothesis_ids) >= target_final
                and promoted_scores
                and min(promoted_scores[:target_final]) >= promotion_score_threshold
                and meta_review_round.coverage_sufficient
            ):
                stop_reason = "target_portfolio_reached"
                break
            if not meta_review_round.should_continue:
                stop_reason = meta_review_round.stop_reason or "meta_review_stop"
                break

            parent_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in ranked_hypotheses}
            parents = [
                parent_by_id[hypothesis_id]
                for hypothesis_id in ranking_round.evolved_parent_hypothesis_ids
                if hypothesis_id in parent_by_id
            ]
            for parent in parents:
                self._artifact_store.append_hypothesis_snapshot(parent.model_copy(update={"status": "evolve"}))
            self._progress_reporter.start(
                "evolution",
                f"Evolving top ideas ({round_label})",
                len(parents),
            )
            evolved = self._evolution_agent.evolve(
                document=document,
                parent_hypotheses=parents,
                ranking_round=ranking_round,
                target_count=evolved_per_round,
                round_index=round_index,
            )
            self._progress_reporter.complete(
                "evolution",
                f"Top ideas evolved ({round_label})",
                len(evolved),
                len(parents),
            )
            self._progress_reporter.start(
                "regeneration",
                f"Generating replacement ideas ({round_label})",
                regenerated_per_round,
            )
            rejected_hypotheses = self._generation_avoid_hypotheses(
                self._artifact_store.latest_hypotheses(research_id)
            )
            regenerated = self._generation_agent.generate_from_meta_review(
                document=document,
                meta_review_round=meta_review_round,
                target_count=regenerated_per_round,
                round_index=round_index,
                avoid_hypotheses=rejected_hypotheses,
                on_progress=lambda completed, total, round_index=round_index: self._progress_reporter.advance(
                    "regeneration",
                    f"Generating replacement ideas ({self._loop_round_label(round_index, max_rounds)})",
                    completed,
                    total,
                ),
            )
            self._progress_reporter.complete(
                "regeneration",
                f"Replacement ideas generated ({round_label})",
                len(regenerated),
                regenerated_per_round,
            )
            new_hypotheses = self._dedupe_new_hypotheses(
                evolved + regenerated,
                existing_hypothesis_ids={
                    hypothesis.hypothesis_id
                    for hypothesis in self._artifact_store.latest_hypotheses(research_id)
                },
            )
            for parent in parents:
                self._artifact_store.append_hypothesis_snapshot(parent.model_copy(update={"status": "reflected"}))
            for hypothesis in new_hypotheses:
                self._artifact_store.append_hypothesis_snapshot(hypothesis)
            new_hypotheses = self._apply_pre_reflection_proximity(document, new_hypotheses, round_index=round_index)
            total_evolved += len([hypothesis for hypothesis in new_hypotheses if hypothesis.generation_source == "evolved"])
            total_regenerated += len(
                [hypothesis for hypothesis in new_hypotheses if hypothesis.generation_source == "regenerated"]
            )
            total_synthesized += len(
                [hypothesis for hypothesis in new_hypotheses if hypothesis.generation_source == "synthesized"]
            )
            reflected_new, run_count = self._reflect_and_append(
                document,
                new_hypotheses,
                concurrency=reflection_concurrency,
                phase="new_reflection",
                progress_message=f"Reflecting on new ideas ({round_label})",
            )
            total_reflected += len(reflected_new)
            automatic_discovery_runs += run_count
            if not new_hypotheses:
                stop_reason = "no_new_hypotheses_generated"
                break

        latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
        ranked_count = len([hypothesis for hypothesis in latest_hypotheses if hypothesis.ranking_round_id])
        self._progress_reporter.start("loop_reporting", "Writing loop report")
        final_report = self._final_portfolio_agent.build_report(
            document=document,
            hypotheses=latest_hypotheses,
            ranking_round=latest_ranking,
            meta_review_round=latest_meta_review,
            stop_reason=stop_reason,
            target_count=target_final,
        )
        report_path = self._artifact_store.write_loop_report(
            research_id,
            final_report,
        )
        self._write_tool_report(document, latest_hypotheses)
        cost_path = self._write_cost_report(research_id)
        self._progress_reporter.complete("loop_reporting", "Loop report written")
        return CoScientistLoopResult(
            research_id=research_id,
            rounds_completed=rounds_completed,
            ranked_hypotheses=ranked_count,
            evolved_hypotheses=total_evolved,
            regenerated_hypotheses=total_regenerated,
            synthesized_hypotheses=total_synthesized,
            reflected_hypotheses=total_reflected,
            automatic_discovery_runs=automatic_discovery_runs,
            ranking_path=str(self._artifact_store.ranking_path(research_id).resolve()),
            hypothesis_path=str(self._artifact_store.hypothesis_path(research_id).resolve()),
            report_path=str(report_path.resolve()),
            stop_reason=stop_reason,
            cost_path=cost_path,
        )

    def _apply_pre_reflection_proximity(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        round_index: int,
    ) -> list[Hypothesis]:
        if len(hypotheses) < 2 or not hasattr(self, "_proximity_agent"):
            return hypotheses

        proximity_round, proximity_updates, synthesized_hypotheses = self._proximity_agent.review_generated(
            document=document,
            hypotheses=hypotheses,
            round_index=round_index,
            max_synthesized_hypotheses=len(hypotheses),
        )
        if not proximity_updates and not synthesized_hypotheses:
            return hypotheses

        if hasattr(self._artifact_store, "append_proximity_round"):
            self._artifact_store.append_proximity_round(proximity_round)
        for hypothesis in proximity_updates:
            self._artifact_store.append_hypothesis_snapshot(hypothesis)
        for hypothesis in synthesized_hypotheses:
            self._artifact_store.append_hypothesis_snapshot(hypothesis)

        retired_ids = {
            hypothesis.hypothesis_id
            for hypothesis in proximity_updates
            if hypothesis.status == "retired" or not hypothesis.is_active
        }
        active_by_id: "OrderedDict[str, Hypothesis]" = OrderedDict()
        for hypothesis in hypotheses:
            if hypothesis.hypothesis_id not in retired_ids and hypothesis.status == "generated" and hypothesis.is_active:
                active_by_id[hypothesis.hypothesis_id] = hypothesis
        for hypothesis in synthesized_hypotheses:
            active_by_id[hypothesis.hypothesis_id] = hypothesis
        return list(active_by_id.values())

    @staticmethod
    def _loop_round_label(round_index: int, max_rounds: int | None) -> str:
        if max_rounds is None:
            return f"round {round_index}"
        return f"round {round_index}/{max_rounds}"

    @staticmethod
    def _apply_ranking_outcomes(
        ranked_hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
    ) -> list[Hypothesis]:
        rejected_ids = set(ranking_round.rejected_hypothesis_ids)
        updated: list[Hypothesis] = []
        for hypothesis in ranked_hypotheses:
            if hypothesis.hypothesis_id in rejected_ids and hypothesis.user_feedback_status != "accepted":
                hypothesis = hypothesis.model_copy(
                    update={
                        "status": "retired",
                        "is_active": False,
                        "retired_reason": "ranking_rejected",
                    }
                )
            updated.append(hypothesis)
        return updated

    @staticmethod
    def _generation_avoid_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        avoided: list[Hypothesis] = []
        for hypothesis in hypotheses:
            if hypothesis.user_feedback_status == "rejected":
                avoided.append(hypothesis)
                continue
            if hypothesis.retired_reason == "ranking_rejected":
                avoided.append(hypothesis)
        return avoided

    def reflect_existing(
        self,
        research_id: str,
        preferred_evidence_recency_days: int | None = None,
        max_reflection_searches_per_hypothesis: int | None = None,
        results_per_query: int | None = None,
        max_pages_per_search: int | None = None,
        max_hypotheses: int | None = None,
        concurrency: int = 3,
        daemon: bool = False,
        worker_id: str | None = None,
        lease_seconds: int = 1800,
        poll_interval_seconds: int = 5,
        idle_exit_after_seconds: int | None = None,
    ) -> CoScientistRunResult:
        self._seed_cost_tracking_for_research(research_id)
        self._progress_reporter.start("resume_load", "Loading research run")
        document = self._artifact_store.load_research_goal(research_id)
        document = self._with_reflection_overrides(
            document=document,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
        self._progress_reporter.complete("resume_load", "Research run loaded")

        automatic_discovery_runs = self._reflect_from_queue(
            document,
            research_id=research_id,
            concurrency=concurrency,
            max_hypotheses=max_hypotheses,
            daemon=daemon,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            poll_interval_seconds=poll_interval_seconds,
            idle_exit_after_seconds=idle_exit_after_seconds,
            phase="resume_reflection",
            progress_message="Reflecting on queued ideas",
        )

        refreshed_hypotheses = self._artifact_store.latest_hypotheses(research_id)
        reflected_hypotheses = [hypothesis for hypothesis in refreshed_hypotheses if hypothesis.status == "reflected"]
        self._progress_reporter.start("resume_reporting", "Writing report")
        report_path = self._artifact_store.write_report(
            document.research_id,
            self._build_report(document, reflected_hypotheses),
        )
        self._write_tool_report(document, reflected_hypotheses)
        cost_path = self._write_cost_report(document.research_id)
        self._progress_reporter.complete("resume_reporting", "Report written")
        return CoScientistRunResult(
            research_id=document.research_id,
            generated_hypotheses=len(refreshed_hypotheses),
            reflected_hypotheses=len(reflected_hypotheses),
            automatic_discovery_runs=automatic_discovery_runs,
            research_goal_path=str(self._artifact_store.research_goal_path(document.research_id).resolve()),
            hypothesis_path=str(self._artifact_store.hypothesis_path(document.research_id).resolve()),
            report_path=str(report_path.resolve()),
            cost_path=cost_path,
        )

    def _reflect_from_queue(
        self,
        document: ResearchGoalDocument,
        research_id: str,
        concurrency: int,
        max_hypotheses: int | None = None,
        daemon: bool = False,
        worker_id: str | None = None,
        lease_seconds: int = 1800,
        poll_interval_seconds: int = 5,
        idle_exit_after_seconds: int | None = None,
        phase: str = "resume_reflection",
        progress_message: str = "Reflecting on queued ideas",
    ) -> int:
        worker_count = max(1, concurrency)
        base_worker_id = worker_id or self._default_reflection_worker_id(research_id)
        state_lock = Lock()
        completed = 0
        discovery_runs = 0
        remaining_claims = max_hypotheses
        max_idle_window = max(1, idle_exit_after_seconds) if idle_exit_after_seconds is not None else None

        self._progress_reporter.start(phase, progress_message)

        def reserve_claim_slot() -> bool:
            nonlocal remaining_claims
            with state_lock:
                if remaining_claims is None:
                    return True
                if remaining_claims <= 0:
                    return False
                remaining_claims -= 1
                return True

        def release_claim_slot() -> None:
            nonlocal remaining_claims
            with state_lock:
                if remaining_claims is not None:
                    remaining_claims += 1

        def record_progress(run_count: int) -> None:
            nonlocal completed, discovery_runs
            with state_lock:
                completed += 1
                discovery_runs += run_count
                current_completed = completed
            self._progress_reporter.advance(phase, progress_message, current_completed)

        def worker_loop(worker_index: int) -> None:
            idle_started_at: float | None = None
            while True:
                if not reserve_claim_slot():
                    return
                claimed = self._artifact_store.claim_next_generated_hypothesis(
                    research_id=research_id,
                    worker_id=f"{base_worker_id}-t{worker_index}",
                    lease_seconds=lease_seconds,
                )
                if claimed is None:
                    release_claim_slot()
                    if self._artifact_store.requeue_expired_reflection_claims(research_id):
                        continue
                    if not daemon:
                        return
                    now = time.monotonic()
                    if idle_started_at is None:
                        idle_started_at = now
                    elif max_idle_window is not None and (now - idle_started_at) >= max_idle_window:
                        return
                    time.sleep(max(1, poll_interval_seconds))
                    continue

                idle_started_at = None
                try:
                    reflected, run_count = self.reflect_hypothesis(document, claimed)
                except Exception as exc:
                    LOGGER.exception("Reflection failed for hypothesis %s", claimed.hypothesis_id)
                    self._artifact_store.release_reflection_claim(claimed, str(exc))
                    record_progress(0)
                    continue
                self._artifact_store.complete_reflection_claim(reflected)
                record_progress(run_count)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(worker_loop, worker_index) for worker_index in range(1, worker_count + 1)]
            for future in as_completed(futures):
                future.result()

        self._progress_reporter.complete(phase, f"{progress_message} complete", completed)
        return discovery_runs

    def _reflect_and_append(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        concurrency: int,
        phase: str = "reflection",
        progress_message: str = "Reflecting on ideas",
    ) -> tuple[list[Hypothesis], int]:
        if not hypotheses:
            self._progress_reporter.complete(phase, f"{progress_message} complete", 0, 0)
            return [], 0
        worker_count = max(1, min(concurrency, len(hypotheses)))
        total_hypotheses = len(hypotheses)
        self._progress_reporter.start(phase, progress_message, total_hypotheses)
        if worker_count == 1:
            reflected_hypotheses: list[Hypothesis] = []
            discovery_runs = 0
            for index, hypothesis in enumerate(hypotheses, start=1):
                reflected, run_count = self.reflect_hypothesis(document, hypothesis, persist=True)
                reflected_hypotheses.append(reflected)
                discovery_runs += run_count
                self._progress_reporter.advance(phase, progress_message, index, total_hypotheses)
            self._progress_reporter.complete(phase, f"{progress_message} complete", total_hypotheses, total_hypotheses)
            return reflected_hypotheses, discovery_runs

        reflected_hypotheses = []
        discovery_runs = 0
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self.reflect_hypothesis, document, hypothesis): hypothesis
                for hypothesis in hypotheses
            }
            for future in as_completed(future_map):
                hypothesis = future_map[future]
                completed += 1
                try:
                    reflected, run_count = future.result()
                except Exception:
                    LOGGER.exception("Reflection failed for hypothesis %s", hypothesis.hypothesis_id)
                    self._progress_reporter.advance(phase, progress_message, completed, total_hypotheses)
                    continue
                reflected_hypotheses.append(reflected)
                discovery_runs += run_count
                self._artifact_store.append_hypothesis_snapshot(reflected)
                self._progress_reporter.advance(phase, progress_message, completed, total_hypotheses)
        reflected_hypotheses.sort(key=lambda item: [hyp.hypothesis_id for hyp in hypotheses].index(item.hypothesis_id))
        completion_message = f"{progress_message} complete"
        if len(reflected_hypotheses) != total_hypotheses:
            completion_message = (
                f"{progress_message} complete ({len(reflected_hypotheses)}/{total_hypotheses} succeeded)"
            )
        self._progress_reporter.complete(phase, completion_message, total_hypotheses, total_hypotheses)
        return reflected_hypotheses, discovery_runs

    def _spawn_reflection_daemons(
        self,
        research_id: str,
        worker_count: int,
        preferred_evidence_recency_days: int,
        max_reflection_searches_per_hypothesis: int,
        results_per_query: int,
        max_pages_per_search: int,
        lease_seconds: int = 1800,
        poll_interval_seconds: int = 1,
        idle_exit_after_seconds: int = 300,
    ) -> None:
        if worker_count <= 0:
            return
        project_root = self._config.data_dir.resolve().parent
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for worker_index in range(1, worker_count + 1):
            worker_id = f"{research_id}-reflector-{worker_index}"
            command = [
                sys.executable,
                "-m",
                "bmscientist.cli",
                "coscientist-reflect",
                "--research-id",
                research_id,
                "--concurrency",
                "1",
                "--daemon",
                "--worker-id",
                worker_id,
                "--lease-seconds",
                str(lease_seconds),
                "--poll-interval-seconds",
                str(poll_interval_seconds),
                "--idle-exit-after-seconds",
                str(idle_exit_after_seconds),
                "--preferred-evidence-recency-days",
                str(preferred_evidence_recency_days),
                "--max-reflection-searches-per-hypothesis",
                str(max_reflection_searches_per_hypothesis),
                "--results-per-query",
                str(results_per_query),
                "--max-pages-per-search",
                str(max_pages_per_search),
            ]
            subprocess.Popen(
                command,
                cwd=str(project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

    def _wait_for_reflection_completion(
        self,
        research_id: str,
        hypothesis_ids: set[str],
        phase: str,
        progress_message: str,
        poll_interval_seconds: float = 1.0,
        report_progress: bool = True,
    ) -> list[Hypothesis]:
        total = len(hypothesis_ids)
        completed = -1
        while True:
            latest_by_id = {
                hypothesis.hypothesis_id: hypothesis
                for hypothesis in self._artifact_store.latest_hypotheses(research_id)
                if hypothesis.hypothesis_id in hypothesis_ids
            }
            finished = [
                hypothesis
                for hypothesis in latest_by_id.values()
                if hypothesis.status == "reflected" or hypothesis.status == "retired"
            ]
            reflected = [hypothesis for hypothesis in finished if hypothesis.status == "reflected"]
            current_completed = len(finished)
            if report_progress and current_completed != completed:
                completed = current_completed
                self._progress_reporter.advance(phase, progress_message, completed, total)
            if current_completed >= total:
                if report_progress:
                    self._progress_reporter.complete(phase, f"{progress_message} complete", current_completed, total)
                return reflected
            time.sleep(max(0.2, poll_interval_seconds))

    @staticmethod
    def _automatic_discovery_runs_for_hypotheses(hypotheses: list[Hypothesis]) -> int:
        return sum(
            len((hypothesis.reflection_assessment or ReflectionAssessment()).reflection_discovery_run_ids)
            for hypothesis in hypotheses
        )

    def _start_reflection_progress_monitor(
        self,
        research_id: str,
        tracked_hypothesis_ids: Callable[[], set[str]],
        phase: str,
        progress_message: str,
        poll_interval_seconds: float = 1.0,
    ) -> Callable[[], None]:
        stop_event = Event()
        self._progress_reporter.start(phase, progress_message)

        def monitor() -> None:
            last_signature: tuple[int, int | None, tuple[str, ...]] | None = None
            while not stop_event.wait(max(0.2, poll_interval_seconds)):
                hypothesis_ids = tracked_hypothesis_ids()
                reflected_count, total, active_worker_lines = self._reflection_progress_counts(research_id, hypothesis_ids)
                display_total = total if total > 0 else None
                signature = (reflected_count, display_total, tuple(active_worker_lines))
                if signature == last_signature:
                    continue
                last_signature = signature
                self._progress_reporter.advance(phase, progress_message, reflected_count, display_total)
                self._set_progress_details(phase, active_worker_lines)

        thread = Thread(target=monitor, name=f"reflection-progress-{research_id}", daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join(timeout=2)

        return stop

    def _reflection_progress_counts(self, research_id: str, hypothesis_ids: set[str]) -> tuple[int, int, list[str]]:
        if not hypothesis_ids:
            return 0, 0, []
        latest_by_id = {
            hypothesis.hypothesis_id: hypothesis
            for hypothesis in self._artifact_store.latest_hypotheses(research_id)
            if hypothesis.hypothesis_id in hypothesis_ids
        }
        reflected_count = len(
            [
                hypothesis
                for hypothesis in latest_by_id.values()
                if hypothesis.status == "reflected" or hypothesis.status == "retired"
            ]
        )
        active_worker_lines = self._active_reflection_worker_lines(latest_by_id.values())
        return reflected_count, len(hypothesis_ids), active_worker_lines

    @staticmethod
    def _active_reflection_worker_lines(hypotheses: Any) -> list[str]:
        active = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.status == "reflecting" and hypothesis.reflection_worker_id
        ]
        active.sort(key=lambda hypothesis: (hypothesis.reflection_worker_id or "", hypothesis.title))
        lines: list[str] = []
        for hypothesis in active:
            worker_id = hypothesis.reflection_worker_id or "worker"
            title = " ".join(hypothesis.title.split())
            if len(title) > 90:
                title = f"{title[:87]}..."
            lines.append(f"{worker_id}: {title}")
        return lines

    def _set_progress_details(self, phase: str, lines: list[str]) -> None:
        details_method = getattr(self._progress_reporter, "details", None)
        if callable(details_method):
            details_method(phase, lines)

    @staticmethod
    def _tracked_hypothesis_ids(
        generated_by_id: "OrderedDict[str, Hypothesis]",
        generated_lock: Lock,
    ) -> set[str]:
        with generated_lock:
            return set(generated_by_id.keys())

    def _seed_cost_tracking_for_research(self, research_id: str) -> None:
        tracker = getattr(self, "_cost_tracker", None)
        if tracker is None:
            return
        seeded_ids = getattr(self, "_cost_tracker_seeded_research_ids", None)
        if seeded_ids is None:
            seeded_ids = set()
            self._cost_tracker_seeded_research_ids = seeded_ids
        if research_id in seeded_ids:
            return
        loader = getattr(self._artifact_store, "load_cost_report", None)
        if callable(loader):
            report = loader(research_id)
            if isinstance(report, dict):
                tracker.seed_from_report(report)
        seeded_ids.add(research_id)

    def _write_tool_report(self, document: ResearchGoalDocument, hypotheses: list[Hypothesis]) -> None:
        writer = getattr(self._artifact_store, "write_tool_report", None)
        if callable(writer):
            writer(
                document.research_id,
                self._build_tool_request_report(document, hypotheses),
            )

    def _write_cost_report(self, research_id: str) -> str | None:
        tracker = getattr(self, "_cost_tracker", None)
        writer = getattr(self._artifact_store, "write_cost_report", None)
        if tracker is None or not callable(writer):
            return None
        self._seed_cost_tracking_for_research(research_id)
        path = writer(research_id, tracker.build_report(research_id))
        return str(path.resolve())

    @staticmethod
    def _default_reflection_worker_id(research_id: str) -> str:
        return f"{research_id}-{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"

    @staticmethod
    def _with_reflection_overrides(
        document: ResearchGoalDocument,
        preferred_evidence_recency_days: int | None,
        max_reflection_searches_per_hypothesis: int | None,
        results_per_query: int | None,
        max_pages_per_search: int | None,
    ) -> ResearchGoalDocument:
        limits = document.reflection_search_limits.model_copy(
            update={
                "max_reflection_searches_per_hypothesis": (
                    max_reflection_searches_per_hypothesis
                    if max_reflection_searches_per_hypothesis is not None
                    else document.reflection_search_limits.max_reflection_searches_per_hypothesis
                ),
                "results_per_query": (
                    results_per_query
                    if results_per_query is not None
                    else document.reflection_search_limits.results_per_query
                ),
                "max_pages_per_search": (
                    max_pages_per_search
                    if max_pages_per_search is not None
                    else document.reflection_search_limits.max_pages_per_search
                ),
            }
        )
        return document.model_copy(
            update={
                "preferred_evidence_recency_days": (
                    preferred_evidence_recency_days
                    if preferred_evidence_recency_days is not None
                    else document.preferred_evidence_recency_days
                ),
                "reflection_search_limits": limits,
            }
        )

    @staticmethod
    def _with_proximity_overrides(
        document: ResearchGoalDocument,
        proximity_merge_mode: str | None,
        proximity_granularity: str | None,
    ) -> ResearchGoalDocument:
        current_policy = ProximityMergePolicy.model_validate(document.proximity_merge_policy)
        next_policy = current_policy.model_copy(
            update={
                "merge_mode": proximity_merge_mode or current_policy.merge_mode,
                "granularity": proximity_granularity or current_policy.granularity,
            }
        )
        if next_policy == current_policy:
            return document
        return document.model_copy(update={"proximity_merge_policy": next_policy})

    @staticmethod
    def _dedupe_new_hypotheses(
        hypotheses: list[Hypothesis],
        existing_hypothesis_ids: set[str] | None = None,
    ) -> list[Hypothesis]:
        deduped: "OrderedDict[str, Hypothesis]" = OrderedDict()
        for hypothesis in hypotheses:
            if existing_hypothesis_ids and hypothesis.hypothesis_id in existing_hypothesis_ids:
                continue
            deduped.setdefault(hypothesis.hypothesis_id, hypothesis)
        return list(deduped.values())

    @staticmethod
    def _build_report(document: ResearchGoalDocument, hypotheses: list[Hypothesis]) -> str:
        lines = [
            f"# Co-Scientist Reflection Report",
            "",
            f"Research ID: `{document.research_id}`",
            f"Goal: {document.raw_goal}",
            f"Research mode: {document.research_mode}",
            f"Candidate artifact type: {document.candidate_artifact_schema.artifact_type}",
            "",
        ]
        if document.evaluation_criteria:
            lines.extend(["Evaluation criteria:", *[f"- {criterion.name}: {criterion.description or criterion.direction}" for criterion in document.evaluation_criteria], ""])
        for hypothesis in hypotheses:
            assessment = hypothesis.reflection_assessment or ReflectionAssessment()
            lines.extend(
                [
                    f"## {hypothesis.title}",
                    "",
                    f"- Application: {hypothesis.application or 'Unknown'}",
                    f"- Market segment: {hypothesis.market_segment or 'Unknown'}",
                    f"- Candidate material: {hypothesis.candidate_material or 'Unknown'}",
                    f"- Incumbent material: {hypothesis.incumbent_material or 'Unknown'}",
                    f"- Candidate artifact: {json.dumps(hypothesis.candidate_artifact, ensure_ascii=True) if hypothesis.candidate_artifact else 'None'}",
                    f"- NBCA material: {assessment.nbca_material or 'Unknown'}",
                    f"- Strategic fit score: {assessment.strategic_fit_score.value}",
                    f"- Market size score: {assessment.market_size_score.value}",
                    f"- Technical success probability: {assessment.technical_success_probability.value}",
                    f"- Commercial success probability: {assessment.commercial_success_probability.value}",
                    f"- Evidence gaps: {', '.join(assessment.evidence_gap_notes) or 'None recorded'}",
                    f"- Tool request notes: {', '.join(assessment.tool_request_notes) or 'None recorded'}",
                    "",
                    "Summary:",
                    hypothesis.summary,
                    "",
                ]
            )
            if assessment.criterion_results:
                lines.append("Criterion results:")
                for result in assessment.criterion_results:
                    lines.append(
                        f"- {result.criterion_name}: score={result.normalized_score} confidence={result.confidence} value={result.value}"
                    )
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_loop_report(
        document: ResearchGoalDocument,
        ranking_round: RankingRound | None,
        meta_review_round: MetaReviewRound | None,
        hypotheses: list[Hypothesis],
        stop_reason: str,
    ) -> str:
        ranked = [hypothesis for hypothesis in hypotheses if hypothesis.ranking_round_id]
        ranked.sort(key=lambda item: (-(item.ranking_score or 0.0), item.title))
        lines = [
            "# Co-Scientist Ranking Loop Report",
            "",
            f"Research ID: `{document.research_id}`",
            f"Goal: {document.raw_goal}",
            f"Research mode: {document.research_mode}",
            f"Stop reason: `{stop_reason}`",
            "",
        ]
        if ranking_round is not None:
            lines.extend(
                [
                    "## Latest Ranking Round",
                    "",
                    f"- Ranking round: `{ranking_round.ranking_round_id}`",
                    f"- Round index: {ranking_round.round_index}",
                    f"- Candidate count: {ranking_round.candidate_count}",
                    f"- Mean score: {ranking_round.mean_score:.2f}",
                    f"- Max score: {ranking_round.max_score:.2f}",
                    f"- Promoted hypotheses: {len(ranking_round.promoted_hypothesis_ids)}",
                    f"- Evolution parents: {len(ranking_round.evolved_parent_hypothesis_ids)}",
                    "",
                    "Best patterns:",
                ]
            )
            lines.extend([f"- {pattern}" for pattern in ranking_round.best_patterns] or ["- None recorded"])
            lines.append("")
            lines.append("Worst patterns:")
            lines.extend([f"- {pattern}" for pattern in ranking_round.worst_patterns] or ["- None recorded"])
            lines.append("")
        if meta_review_round is not None:
            lines.extend(
                [
                    "## Meta-review",
                    "",
                    f"- Gap shrinkage status: {meta_review_round.gap_shrinkage_status}",
                    f"- Coverage sufficient: {meta_review_round.coverage_sufficient}",
                    f"- Gap persistence count: {meta_review_round.gap_persistence_count}",
                    f"- Continue loop: {meta_review_round.should_continue}",
                    "",
                    "Whitespace gaps:",
                ]
            )
            lines.extend([f"- {item}" for item in meta_review_round.whitespace_gaps] or ["- None recorded"])
            lines.append("")
            lines.append("Meta-review generation guidance:")
            lines.extend([f"- {item}" for item in meta_review_round.generation_guidance] or ["- None recorded"])
            lines.append("")

        lines.extend(["## Ranked Opportunities", ""])
        for index, hypothesis in enumerate(ranked[: document.target_hypotheses_final * 2], start=1):
            assessment = hypothesis.reflection_assessment or ReflectionAssessment()
            lines.extend(
                [
                    f"### {index}. {hypothesis.title}",
                    "",
                    f"- Ranking score: {hypothesis.ranking_score}",
                    f"- Ranking status: {hypothesis.ranking_status or 'Unknown'}",
                    f"- Generation source: {hypothesis.generation_source}",
                    f"- Active: {hypothesis.is_active}",
                    f"- Application: {hypothesis.application or 'Unknown'}",
                    f"- Market segment: {hypothesis.market_segment or 'Unknown'}",
                    f"- Candidate material: {hypothesis.candidate_material or 'Unknown'}",
                    f"- Incumbent material: {hypothesis.incumbent_material or 'Unknown'}",
                    f"- Candidate artifact: {json.dumps(hypothesis.candidate_artifact, ensure_ascii=True) if hypothesis.candidate_artifact else 'None'}",
                    f"- Concepts: {', '.join(hypothesis.concept_labels) or 'None'}",
                    f"- Strategic fit: {assessment.strategic_fit_score.value}",
                    f"- Market size: {assessment.market_size_score.value}",
                    f"- Technical success: {assessment.technical_success_probability.value}",
                    f"- Commercial success: {assessment.commercial_success_probability.value}",
                    f"- Evidence gaps: {', '.join(assessment.evidence_gap_notes) or 'None recorded'}",
                    f"- Tool request notes: {', '.join(assessment.tool_request_notes) or 'None recorded'}",
                    "",
                    hypothesis.ranking_rationale or hypothesis.summary,
                    "",
                ]
            )
            if assessment.criterion_results:
                lines.append("Criterion results:")
                for result in assessment.criterion_results:
                    lines.append(
                        f"- {result.criterion_name}: score={result.normalized_score} confidence={result.confidence} value={result.value}"
                    )
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_tool_request_report(document: ResearchGoalDocument, hypotheses: list[Hypothesis]) -> str:
        lines = [
            "# Tool Requests",
            "",
            f"Research ID: `{document.research_id}`",
            f"Goal: {document.raw_goal}",
            f"Research mode: {document.research_mode}",
            "",
        ]
        if not document.tool_requests:
            lines.append("No tool requests recorded.")
            return "\n".join(lines)
        for request in document.tool_requests:
            dependent_criteria = [
                criterion.name
                for criterion in document.evaluation_criteria
                if request.tool_id in criterion.suggested_tool_ids
            ]
            lines.extend(
                [
                    f"## {request.tool_id}",
                    "",
                    f"- Purpose: {request.purpose}",
                    f"- Status: {request.status}",
                    f"- Candidate packages: {', '.join(request.candidate_packages) or 'None recorded'}",
                    f"- Required inputs: {', '.join(request.required_inputs) or 'None recorded'}",
                    f"- Expected outputs: {', '.join(request.expected_outputs) or 'None recorded'}",
                    f"- Dependent criteria: {', '.join(dependent_criteria) or 'None recorded'}",
                    f"- Limitations: {', '.join(request.limitations) or 'None recorded'}",
                    "",
                ]
            )
        hypothesis_notes = list(
            OrderedDict.fromkeys(
                note
                for hypothesis in hypotheses
                for note in (hypothesis.reflection_assessment.tool_request_notes if hypothesis.reflection_assessment else [])
                if note
            )
        )
        if hypothesis_notes:
            lines.extend(["## Reflection Notes", ""])
            lines.extend([f"- {note}" for note in hypothesis_notes])
        return "\n".join(lines)
