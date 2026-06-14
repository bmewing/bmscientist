from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from app_discovery_agent.chunking import TextChunker
from app_discovery_agent.classify import EvidenceClassifier
from app_discovery_agent.config import AppConfig
from app_discovery_agent.coscientist_models import (
    AssessmentMetric,
    CoScientistRunResult,
    CoScientistLoopResult,
    EvidenceCitation,
    EvolutionHypothesisSeed,
    GapShrinkageStatus,
    Hypothesis,
    HypothesisEvolutionOutput,
    HypothesisGenerationOutput,
    MetaReviewOutput,
    MetaReviewRound,
    PriceMetric,
    ProximityConcept,
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
)
from app_discovery_agent.coscientist_store import CoScientistStore
from app_discovery_agent.extract import PageFetcher, extract_domain
from app_discovery_agent.graph_market import GraphMarketEvidence
from app_discovery_agent.llm import DeepSeekLLM
from app_discovery_agent.manual_ingest import ManualEvidenceIngestor
from app_discovery_agent.models import ChunkRecord, DiscoverySummary, PageContent, SearchResultItem
from app_discovery_agent.price_cache import StructuredPriceCache
from app_discovery_agent.prompt_library import PROMPTS
from app_discovery_agent.search import ExaSearchClient, deduplicate_search_results


if TYPE_CHECKING:
    from app_discovery_agent.agent import DiscoveryAgent
    from app_discovery_agent.embeddings import LocalEmbedder
    from app_discovery_agent.store import LanceEvidenceStore


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
        queries.extend(document.target_incumbent_materials[:2])
        queries.extend(document.preferred_candidate_materials[:2])
        queries.extend(document.recycling_or_sustainability_angles[:2])
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
        normalized_queries = [query for query in queries if query]
        return self.search_many(normalized_queries, top_k_per_query=5, max_results=max_results)

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
    ):
        self._legacy_discovery_agent = source if hasattr(source, "discover") else None
        self._config = source if isinstance(source, AppConfig) else None
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._search = ExaSearchClient(source) if isinstance(source, AppConfig) else None
        self._fetcher = PageFetcher(source) if isinstance(source, AppConfig) else None
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
            search_response = self._search.search(query=query, num_results=limits.results_per_query)
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
        for result in results:
            if self._fetcher.should_skip_direct_fetch(str(result.url)):
                skipped_pages.append({"url": str(result.url), "search_query": result.search_query, "reason": "blocked_domain"})
                fallback = self._build_partial_page_from_search_result(query, result, "blocked_domain")
                if fallback:
                    fetched_pages.append(fallback)
                continue
            page, error = self._fetcher.safe_fetch(result)
            if page:
                fetched_pages.append(page)
                continue
            if error:
                skipped_pages.append(error)
            fallback = self._build_partial_page_from_search_result(query, result, "fetch_error")
            if fallback:
                fetched_pages.append(fallback)
        return fetched_pages

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
        parts = [result.title.strip(), result.summary.strip(), result.snippet.strip()]
        partial_text = "\n\n".join(part for part in parts if part)
        if len(partial_text) < 80:
            return None
        return PageContent(
            title=result.title,
            url=str(result.url),
            search_query=query,
            source_domain=extract_domain(str(result.url)),
            fetched_at=datetime.now(timezone.utc),
            text=partial_text,
            status_code=None,
            content_type="application/x-search-snippet",
            raw_excerpt=partial_text[:500],
            metadata={
                "is_partial_evidence": True,
                "partial_evidence_reason": reason,
                "source_type": "exa_search_result",
                "search_result_summary": result.summary,
                "search_result_snippet": result.snippet,
                "search_result_score": result.score,
                "search_result_published_date": result.published_date,
            },
        )


class ReflectionSearchPlanner:
    def plan(
        self,
        document: ResearchGoalDocument,
        hypothesis: Hypothesis,
        assessment: ReflectionAssessment,
        suggested_queries: list[str],
    ) -> list[str]:
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
        return ResearchGoalDocument(
            research_id=research_id,
            raw_goal=raw_goal,
            target_hypotheses_final=target_hypotheses_final,
            target_hypotheses_generated=self._default_generated_count(target_hypotheses_final),
            regions=regions or [],
            strategic_fit_criteria=draft.strategic_fit_criteria,
            target_incumbent_materials=draft.target_incumbent_materials,
            preferred_candidate_materials=draft.preferred_candidate_materials,
            candidate_material_preferences=draft.candidate_material_preferences,
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
            strategic_fit_notes=strategic_fit_notes,
        )


class GenerationAgent:
    _batch_size = 5

    def __init__(self, llm: DeepSeekLLM, retriever: LocalEvidenceRetriever):
        self._llm = llm
        self._retriever = retriever

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def generate(
        self,
        document: ResearchGoalDocument,
        on_progress: Callable[[int, int], None] | None = None,
        on_batch: Callable[[list[Hypothesis]], None] | None = None,
    ) -> list[Hypothesis]:
        evidence_rows = self._retriever.retrieve_for_goal(
            document,
            max_results=max(document.target_hypotheses_generated * 4, 12),
        )
        evidence_payload = [
            {
                "chunk_id": row.get("id"),
                "application": row.get("application"),
                "incumbent_material": row.get("incumbent_material"),
                "candidate_materials": row.get("candidate_materials"),
                "application_requirements": row.get("application_requirements"),
                "substitution_drivers": row.get("substitution_drivers"),
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
        )

    def generate_from_meta_review(
        self,
        document: ResearchGoalDocument,
        meta_review_round: MetaReviewRound,
        target_count: int,
        round_index: int,
        on_progress: Callable[[int, int], None] | None = None,
        on_batch: Callable[[list[Hypothesis]], None] | None = None,
    ) -> list[Hypothesis]:
        if target_count <= 0:
            return []
        evidence_rows = self._retriever.retrieve_for_goal(document, max_results=max(target_count * 5, 12))
        evidence_payload = [
            {
                "chunk_id": row.get("id"),
                "application": row.get("application"),
                "incumbent_material": row.get("incumbent_material"),
                "candidate_materials": row.get("candidate_materials"),
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
        **extra_context: Any,
    ) -> list[Hypothesis]:
        hypotheses: "OrderedDict[str, Hypothesis]" = OrderedDict()
        batch_size = max(1, min(self._batch_size, limit))
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
                    self._existing_hypothesis_prompt_payload(list(hypotheses.values())),
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
                if hypothesis.hypothesis_id in hypotheses:
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

    @staticmethod
    def _existing_hypothesis_prompt_payload(hypotheses: list[Hypothesis]) -> list[dict[str, Any]]:
        return [
            {
                "title": hypothesis.title,
                "application": hypothesis.application,
                "market_segment": hypothesis.market_segment,
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
            }
            for hypothesis in hypotheses
        ]

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
            hypothesis_id = str(
                uuid5(
                    NAMESPACE_URL,
                    (
                        f"{document.research_id}::{generation_source}::{round_index}::"
                        f"{seed.title}::{seed.application or ''}::{seed.candidate_material or ''}"
                    ),
                )
            )
            if hypothesis_id in hypotheses:
                continue
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
                generation_source=generation_source,
            )
            if len(hypotheses) >= limit:
                break
        return list(hypotheses.values())


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
        promoted_ids = [item.hypothesis_id for item in normalized_rankings[:target_final_count]]
        evolved_parent_ids = [
            item.hypothesis_id
            for item in normalized_rankings
            if item.recommended_action in {"advance", "evolve"}
        ][:evolve_top_k]
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
                "nbca": hypothesis.next_best_competitive_alternative,
                "generation_source": hypothesis.generation_source,
                "round_index": hypothesis.round_index,
                "heuristic_score": round(heuristic_by_id.get(hypothesis.hypothesis_id, 0.0), 3),
                "reflection": self._assessment_payload(hypothesis.reflection_assessment),
                "evidence_gaps": (hypothesis.reflection_assessment.evidence_gap_notes if hypothesis.reflection_assessment else []),
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
        if not active_reflected:
            proximity_round = ProximityRound(
                proximity_round_id=str(uuid4()),
                research_id=document.research_id,
                round_index=round_index,
                notes=["No active reflected hypotheses available for proximity review."],
            )
            return proximity_round, [], []

        try:
            output = self._llm_review(document, active_reflected, max_synthesized_hypotheses)
        except Exception as exc:
            LOGGER.warning("Proximity review failed (%s); falling back to deterministic concept grouping", exc)
            output = self._fallback_review(active_reflected)

        concepts = [concept for concept in output.concepts if concept.member_hypothesis_ids]
        concepts_by_member: dict[str, list[ProximityConcept]] = {}
        for concept in concepts:
            for hypothesis_id in concept.member_hypothesis_ids:
                concepts_by_member.setdefault(hypothesis_id, []).append(concept)

        updated_hypotheses: list[Hypothesis] = []
        active_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in active_reflected}
        for hypothesis in active_reflected:
            member_concepts = concepts_by_member.get(hypothesis.hypothesis_id, [])
            if not member_concepts:
                continue
            labels = list(
                OrderedDict.fromkeys(
                    hypothesis.concept_labels + [concept.concept_label for concept in member_concepts]
                )
            )
            updated_hypotheses.append(
                hypothesis.model_copy(
                    update={
                        "concept_labels": labels,
                        "concept_cluster_id": hypothesis.concept_cluster_id or member_concepts[0].concept_label,
                    }
                )
            )

        synthesized_hypotheses: list[Hypothesis] = []
        retired_hypothesis_ids: list[str] = []
        labeled_hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in updated_hypotheses]
        for seed in output.synthesized_hypotheses[:max_synthesized_hypotheses]:
            member_ids = [item for item in seed.merged_from_hypothesis_ids if item in active_by_id]
            if len(member_ids) < 2:
                continue
            synthesized = self._seed_to_hypothesis(document, seed, round_index, member_ids)
            synthesized_hypotheses.append(synthesized)
            for member_id in member_ids:
                retired_hypothesis_ids.append(member_id)
                member = active_by_id[member_id]
                merged_labels = list(
                    OrderedDict.fromkeys(
                        member.concept_labels + ([seed.concept_label] if seed.concept_label else [])
                    )
                )
                updated_hypotheses.append(
                    member.model_copy(
                        update={
                            "status": "retired",
                            "is_active": False,
                            "retired_reason": "merged_into_synthesized_hypothesis",
                            "superseded_by_hypothesis_id": synthesized.hypothesis_id,
                            "concept_labels": merged_labels,
                            "concept_cluster_id": member.concept_cluster_id or seed.concept_label,
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
            notes=output.notes,
        )
        deduped_updates: "OrderedDict[str, Hypothesis]" = OrderedDict()
        for hypothesis in updated_hypotheses:
            deduped_updates[hypothesis.hypothesis_id] = hypothesis
        return proximity_round, list(deduped_updates.values()), synthesized_hypotheses

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
                "candidate_material": hypothesis.candidate_material,
                "incumbent_material": hypothesis.incumbent_material,
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

    @staticmethod
    def _fallback_review(hypotheses: list[Hypothesis]) -> ProximityReviewOutput:
        grouped: dict[tuple[str, str], list[str]] = {}
        for hypothesis in hypotheses:
            key = (
                (hypothesis.application or "unknown").strip().lower(),
                (hypothesis.candidate_material or "unknown").strip().lower(),
            )
            grouped.setdefault(key, []).append(hypothesis.hypothesis_id)
        fallback_concepts: list[ProximityConcept] = []
        for (application, candidate_material), member_ids in grouped.items():
            if len(member_ids) < 2:
                continue
            label = f"{(candidate_material or 'material').upper()} in {application.title()}"
            fallback_concepts.append(
                ProximityConcept(
                    concept_label=label,
                    description="Deterministic grouping by application and candidate material.",
                    member_hypothesis_ids=member_ids,
                )
            )
        return ProximityReviewOutput(
            concepts=fallback_concepts,
            synthesized_hypotheses=[],
            notes=["Used deterministic proximity grouping fallback."],
        )

    @staticmethod
    def _seed_to_hypothesis(
        document: ResearchGoalDocument,
        seed: SynthesizedHypothesisSeed,
        round_index: int,
        member_ids: list[str],
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
            strategic_rationale=seed.synthesis_rationale or seed.strategic_rationale,
            supporting_chunk_ids=seed.supporting_chunk_ids,
            supporting_urls=seed.supporting_urls,
            assumptions=seed.assumptions,
            unknowns=seed.unknowns,
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
        try:
            output = self._llm_review(document, active_reflected, ranking_round)
        except Exception as exc:
            LOGGER.warning("Meta-review failed (%s); falling back to deterministic gap review", exc)
            output = self._fallback_review(document, active_reflected, ranking_round)

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
                "evidence_gaps": (
                    hypothesis.reflection_assessment.evidence_gap_notes if hypothesis.reflection_assessment else []
                ),
            }
            for hypothesis in hypotheses
        ]
        system_prompt = PROMPTS.render("meta_review_agent", "review.system")
        user_prompt = PROMPTS.render(
            "meta_review_agent",
            "review.user",
            research_goal=document.raw_goal,
            document_json=document.model_dump_json(indent=2),
            best_patterns_json=json.dumps(ranking_round.best_patterns, indent=2),
            worst_patterns_json=json.dumps(ranking_round.worst_patterns, indent=2),
            hypotheses_json=json.dumps(payload, indent=2),
        )
        return self._llm.complete_json(MetaReviewOutput, system_prompt, user_prompt)

    @staticmethod
    def _fallback_review(
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        ranking_round: RankingRound,
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
    ):
        self._llm = llm
        self._retriever = retriever
        self._discovery_tool = discovery_tool
        self._search_planner = ReflectionSearchPlanner()
        self._price_cache = price_cache
        self._graph_evidence = graph_evidence

    def reflect(self, document: ResearchGoalDocument, hypothesis: Hypothesis) -> tuple[Hypothesis, int]:
        price_document = None
        if self._price_cache is not None:
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

        final_review = self._review(document, hypothesis, evidence_rows)
        assessment = self._merge_price_metrics(
            final_review.assessment,
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

    def _review(self, document: ResearchGoalDocument, hypothesis: Hypothesis, evidence_rows: list[dict[str, Any]]) -> ReflectionReviewOutput:
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
    def _fallback_search_queries(document: ResearchGoalDocument, hypothesis: Hypothesis) -> list[str]:
        region_text = " ".join(document.regions)
        return [
            f"{hypothesis.application or ''} {hypothesis.incumbent_material or ''} market size {region_text}".strip(),
            f"{hypothesis.application or ''} {hypothesis.incumbent_material or ''} price usd kg {region_text}".strip(),
            f"{hypothesis.application or ''} {hypothesis.candidate_material or ''} replacement fit requirements".strip(),
            f"{hypothesis.application or ''} {hypothesis.candidate_material or ''} drop in replacement lead time".strip(),
        ]


class CoScientistRunner:
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
        self._llm = llm or DeepSeekLLM(config)
        self._planning_llm = DeepSeekLLM(config, model=config.planning_chat_model)
        self._generation_llm = DeepSeekLLM(config, model=config.generation_chat_model)
        self._reflection_llm = DeepSeekLLM(config, model=config.reflection_chat_model)
        self._ranking_llm = DeepSeekLLM(config, model=config.ranking_chat_model)
        self._evolution_llm = DeepSeekLLM(config, model=config.evolution_chat_model)
        self._proximity_llm = DeepSeekLLM(config, model=config.proximity_chat_model)
        self._meta_review_llm = DeepSeekLLM(config, model=config.meta_review_chat_model)
        if evidence_store is None:
            from app_discovery_agent.store import LanceEvidenceStore

            self._evidence_store = LanceEvidenceStore(config.resolved_lancedb_path())
        else:
            self._evidence_store = evidence_store
        if embedder is None:
            from app_discovery_agent.embeddings import LocalEmbedder

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
        self._generation_agent = GenerationAgent(self._generation_llm, self._retriever)
        self._ranking_agent = RankingAgent(self._ranking_llm)
        self._evolution_agent = EvolutionAgent(self._evolution_llm)
        self._proximity_agent = ProximityCheckAgent(self._proximity_llm)
        self._meta_review_agent = MetaReviewAgent(self._meta_review_llm)
        self._final_portfolio_agent = FinalPortfolioAgent(self._meta_review_llm)
        discovery_tool = (
            DiscoveryEvidenceTool(discovery_agent)
            if discovery_agent is not None
            else DiscoveryEvidenceTool(config, self._reflection_llm, self._embedder, self._evidence_store)
        )
        self._reflection_agent = ReflectionAgent(
            self._reflection_llm,
            self._retriever,
            discovery_tool,
            self._price_cache,
            self._graph_evidence,
        )

    def set_progress_reporter(self, reporter: ProgressReporter | None) -> None:
        self._progress_reporter = reporter or NullProgressReporter()

    def prepare_project_name(self, preferred_name: str | None = None) -> str:
        return self._artifact_store.claim_project_name(preferred_name)

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
        spawn_reflection_daemons: bool = False,
    ) -> CoScientistRunResult:
        limits = ReflectionSearchLimits(
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
        research_id = project_name or self.prepare_project_name()
        self._progress_reporter.start("planning", "Processing goal")
        document = self._planning_agent.create_research_goal(
            research_id=research_id,
            raw_goal=goal,
            target_hypotheses_final=target_hypotheses,
            regions=regions,
            strategic_fit_notes=strategic_fit_notes,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            reflection_search_limits=limits,
        )
        research_goal_path = self._artifact_store.save_research_goal(document)
        self._progress_reporter.complete("planning", "Goal processed")

        generated_by_id: "OrderedDict[str, Hypothesis]" = OrderedDict()
        generated_lock = Lock()
        stop_reflection_monitor: Callable[[], None] | None = None
        if spawn_reflection_daemons:
            self._spawn_reflection_daemons(
                research_id=document.research_id,
                worker_count=min(max(1, self._generation_agent.batch_size), document.target_hypotheses_generated),
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
        self._progress_reporter.complete("reporting", "Report written")
        return CoScientistRunResult(
            research_id=document.research_id,
            generated_hypotheses=len(generated),
            reflected_hypotheses=len(reflected_hypotheses),
            automatic_discovery_runs=automatic_discovery_runs,
            research_goal_path=str(research_goal_path.resolve()),
            hypothesis_path=str(self._artifact_store.hypothesis_path(document.research_id).resolve()),
            report_path=str(report_path.resolve()),
        )

    def run_loop(
        self,
        research_id: str,
        target_final_hypotheses: int | None = None,
        max_rounds: int = 1,
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
    ) -> CoScientistLoopResult:
        self._progress_reporter.start("loop_load", "Loading research run")
        document = self._artifact_store.load_research_goal(research_id)
        document = self._with_reflection_overrides(
            document=document,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
        self._progress_reporter.complete("loop_load", "Research run loaded")
        target_final = target_final_hypotheses or document.target_hypotheses_final
        rounds_completed = 0
        total_evolved = 0
        total_regenerated = 0
        total_synthesized = 0
        total_reflected = 0
        automatic_discovery_runs = 0
        stop_reason = "max_rounds_reached"
        latest_ranking: RankingRound | None = None
        latest_meta_review: MetaReviewRound | None = None

        for round_index in range(1, max_rounds + 1):
            latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
            reflected = [
                hypothesis
                for hypothesis in latest_hypotheses
                if hypothesis.status == "reflected" and hypothesis.is_active
            ]
            if not reflected:
                stop_reason = "no_active_reflected_hypotheses"
                break
            self._progress_reporter.start(
                "ranking",
                f"Ranking ideas (round {round_index}/{max_rounds})",
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
                f"Ideas ranked (round {round_index}/{max_rounds})",
                len(ranked_hypotheses),
                len(reflected),
            )
            latest_ranking = ranking_round
            self._artifact_store.append_ranking_round(ranking_round)
            for hypothesis in ranked_hypotheses:
                self._artifact_store.append_hypothesis_snapshot(hypothesis)
            rounds_completed += 1

            if proximity_check_every > 0 and round_index % proximity_check_every == 0:
                proximity_hypotheses = self._artifact_store.latest_hypotheses(research_id)
                self._progress_reporter.start(
                    "proximity",
                    f"Checking idea overlap (round {round_index}/{max_rounds})",
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
                    f"Idea overlap checked (round {round_index}/{max_rounds})",
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
                    progress_message=f"Reflecting on synthesized ideas (round {round_index}/{max_rounds})",
                )
                total_reflected += len(reflected_synthesized)
                automatic_discovery_runs += run_count

            latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
            self._progress_reporter.start(
                "meta_review",
                f"Reviewing portfolio gaps (round {round_index}/{max_rounds})",
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
                f"Portfolio gaps reviewed (round {round_index}/{max_rounds})",
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
                and (meta_review_round.coverage_sufficient or not meta_review_round.whitespace_gaps)
            ):
                stop_reason = "target_portfolio_reached"
                break
            if not meta_review_round.should_continue:
                stop_reason = meta_review_round.stop_reason or "meta_review_stop"
                break
            if round_index >= max_rounds:
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
                f"Evolving top ideas (round {round_index}/{max_rounds})",
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
                f"Top ideas evolved (round {round_index}/{max_rounds})",
                len(evolved),
                len(parents),
            )
            self._progress_reporter.start(
                "regeneration",
                f"Generating replacement ideas (round {round_index}/{max_rounds})",
                regenerated_per_round,
            )
            regenerated = self._generation_agent.generate_from_meta_review(
                document=document,
                meta_review_round=meta_review_round,
                target_count=regenerated_per_round,
                round_index=round_index,
                on_progress=lambda completed, total, round_index=round_index: self._progress_reporter.advance(
                    "regeneration",
                    f"Generating replacement ideas (round {round_index}/{max_rounds})",
                    completed,
                    total,
                ),
            )
            self._progress_reporter.complete(
                "regeneration",
                f"Replacement ideas generated (round {round_index}/{max_rounds})",
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
            total_evolved += len([hypothesis for hypothesis in new_hypotheses if hypothesis.generation_source == "evolved"])
            total_regenerated += len(
                [hypothesis for hypothesis in new_hypotheses if hypothesis.generation_source == "regenerated"]
            )
            reflected_new, run_count = self._reflect_and_append(
                document,
                new_hypotheses,
                concurrency=reflection_concurrency,
                phase="new_reflection",
                progress_message=f"Reflecting on new ideas (round {round_index}/{max_rounds})",
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
        )

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
        self._progress_reporter.complete("resume_reporting", "Report written")
        return CoScientistRunResult(
            research_id=document.research_id,
            generated_hypotheses=len(refreshed_hypotheses),
            reflected_hypotheses=len(reflected_hypotheses),
            automatic_discovery_runs=automatic_discovery_runs,
            research_goal_path=str(self._artifact_store.research_goal_path(document.research_id).resolve()),
            hypothesis_path=str(self._artifact_store.hypothesis_path(document.research_id).resolve()),
            report_path=str(report_path.resolve()),
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
                    reflected, run_count = self._reflection_agent.reflect(document, claimed)
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
                reflected, run_count = self._reflection_agent.reflect(document, hypothesis)
                reflected_hypotheses.append(reflected)
                discovery_runs += run_count
                self._artifact_store.append_hypothesis_snapshot(reflected)
                self._progress_reporter.advance(phase, progress_message, index, total_hypotheses)
            self._progress_reporter.complete(phase, f"{progress_message} complete", total_hypotheses, total_hypotheses)
            return reflected_hypotheses, discovery_runs

        reflected_hypotheses = []
        discovery_runs = 0
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._reflection_agent.reflect, document, hypothesis): hypothesis
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
        project_root = Path(__file__).resolve().parents[2]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for worker_index in range(1, worker_count + 1):
            worker_id = f"{research_id}-reflector-{worker_index}"
            command = [
                sys.executable,
                "-m",
                "app_discovery_agent.cli",
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
            "",
        ]
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
                    f"- NBCA material: {assessment.nbca_material or 'Unknown'}",
                    f"- Strategic fit score: {assessment.strategic_fit_score.value}",
                    f"- Market size score: {assessment.market_size_score.value}",
                    f"- Technical success probability: {assessment.technical_success_probability.value}",
                    f"- Commercial success probability: {assessment.commercial_success_probability.value}",
                    f"- Evidence gaps: {', '.join(assessment.evidence_gap_notes) or 'None recorded'}",
                    "",
                    "Summary:",
                    hypothesis.summary,
                    "",
                ]
            )
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
                    f"- Concepts: {', '.join(hypothesis.concept_labels) or 'None'}",
                    f"- Strategic fit: {assessment.strategic_fit_score.value}",
                    f"- Market size: {assessment.market_size_score.value}",
                    f"- Technical success: {assessment.technical_success_probability.value}",
                    f"- Commercial success: {assessment.commercial_success_probability.value}",
                    f"- Evidence gaps: {', '.join(assessment.evidence_gap_notes) or 'None recorded'}",
                    "",
                    hypothesis.ranking_rationale or hypothesis.summary,
                    "",
                ]
            )
        return "\n".join(lines)
