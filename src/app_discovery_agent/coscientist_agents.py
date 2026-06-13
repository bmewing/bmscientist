from __future__ import annotations

import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any
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
from app_discovery_agent.llm import DeepSeekLLM
from app_discovery_agent.manual_ingest import ManualEvidenceIngestor
from app_discovery_agent.models import ChunkRecord, DiscoverySummary, PageContent, SearchResultItem
from app_discovery_agent.price_cache import StructuredPriceCache
from app_discovery_agent.search import ExaSearchClient, deduplicate_search_results


if TYPE_CHECKING:
    from app_discovery_agent.agent import DiscoveryAgent
    from app_discovery_agent.embeddings import LocalEmbedder
    from app_discovery_agent.store import LanceEvidenceStore


LOGGER = logging.getLogger(__name__)


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
            min_score = 0.08 if is_partial else 0.15
            heuristic_score = self._classifier.heuristic_relevance(query, page.text)
            if heuristic_score < min_score:
                skipped_pages.append({"url": str(page.url), "reason": "low_heuristic_relevance", "score": heuristic_score})
                continue
            candidates.append(page)
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
            if classification.relevance_score < self._config.min_relevance_score or not classification.relevant:
                skipped_pages.append(
                    {
                        "url": str(page.url),
                        "reason": "below_relevance_threshold",
                        "relevance_score": classification.relevance_score,
                    }
                )
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
        raw_goal: str,
        target_hypotheses_final: int,
        regions: list[str] | None,
        strategic_fit_notes: str | None,
        preferred_evidence_recency_days: int,
        reflection_search_limits: ReflectionSearchLimits,
    ) -> ResearchGoalDocument:
        system_prompt = (
            "You are a research planning agent. Convert a research goal into a structured plan for downstream "
            "hypothesis generation and reflection. Return strict JSON only."
        )
        user_prompt = f"""
Raw research goal:
{raw_goal}

Target final hypotheses: {target_hypotheses_final}
Regions: {regions or []}
Strategic fit notes: {strategic_fit_notes or ""}

Return JSON with:
- strategic_fit_criteria (array of strings)
- target_incumbent_materials (array of strings)
- preferred_candidate_materials (array of strings)
- candidate_material_preferences (array of strings)
- recycling_or_sustainability_angles (array of strings)
- material_scope (array of strings)
- application_scope (array of strings)
- opportunity_modes (array of strings)
- opportunity_speed_horizon_months (integer or null)
- commercialization_constraints (array of strings)
- ranking_weights (object with numeric weights like speed, volume, strategic_fit, sustainability)
- success_definition (string)

Be concise and specific. Do not invent constraints not implied by the goal.
"""
        draft = self._llm.complete_json(ResearchPlanDraft, system_prompt, user_prompt)
        return ResearchGoalDocument(
            research_id=str(uuid4()),
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
    def __init__(self, llm: DeepSeekLLM, retriever: LocalEvidenceRetriever):
        self._llm = llm
        self._retriever = retriever

    def generate(self, document: ResearchGoalDocument) -> list[Hypothesis]:
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
        system_prompt = (
            "You are a generation agent for industrial material opportunity research. "
            "Create hypotheses grounded in the supplied evidence. Return strict JSON only."
        )
        user_prompt = f"""
Research goal:
{document.raw_goal}

Research configuration:
{document.model_dump_json(indent=2)}

Available evidence:
{json.dumps(evidence_payload, indent=2)}

Generate {document.target_hypotheses_generated} distinct hypotheses grounded in the evidence.

Each hypothesis must include:
- title
- summary
- application
- market_segment
- candidate_material
- incumbent_material
- next_best_competitive_alternative
- incumbent_form
- candidate_form
- conversion_process
- product_type
- buyer_type
- application_requirements
- substitution_drivers
- strategic_rationale
- supporting_chunk_ids
- supporting_urls
- assumptions
- unknowns
- generation_confidence

Rules:
- Use evidence, not pure brainstorming.
- Cite chunk IDs and URLs already present in the evidence.
- Capture material form, product type, buyer type, and conversion process when supported or clearly implied.
- If a detail is unclear, leave it in unknowns rather than inventing it.
"""
        output = self._llm.complete_json(HypothesisGenerationOutput, system_prompt, user_prompt)
        return self._seeds_to_hypotheses(
            document=document,
            seeds=output.hypotheses,
            limit=document.target_hypotheses_generated,
            generation_source="initial",
            round_index=0,
        )

    def generate_from_meta_review(
        self,
        document: ResearchGoalDocument,
        meta_review_round: MetaReviewRound,
        target_count: int,
        round_index: int,
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
        system_prompt = (
            "You are a generation agent improving an industrial material opportunity portfolio. "
            "Create new hypotheses grounded in local evidence and meta-review whitespace guidance. Return strict JSON only."
        )
        user_prompt = f"""
Research goal:
{document.raw_goal}

Research configuration:
{document.model_dump_json(indent=2)}

Meta-review guidance:
{json.dumps(meta_review_round.generation_guidance, indent=2)}

Whitespace gaps:
{json.dumps(meta_review_round.whitespace_gaps, indent=2)}

Evidence available for new ideas:
{json.dumps(evidence_payload, indent=2)}

Generate {target_count} new hypotheses that directly address the whitespace gaps and follow the meta-review guidance.
Use the same schema as prior hypotheses. Cite only provided chunk IDs and URLs.
"""
        output = self._llm.complete_json(HypothesisGenerationOutput, system_prompt, user_prompt)
        return self._seeds_to_hypotheses(
            document=document,
            seeds=output.hypotheses,
            limit=target_count,
            generation_source="regenerated",
            round_index=round_index,
        )

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
        system_prompt = (
            "You are the Ranking Agent in a local AI co-scientist system. "
            "Judge reflected industrial-material opportunities conservatively. Return strict JSON only."
        )
        user_prompt = f"""
Research goal:
{document.raw_goal}

Research configuration:
{document.model_dump_json(indent=2)}

Rank the reflected hypotheses below as a tournament judge. Strong opportunities should fit the research strategy,
have credible technical and commercial paths, cite evidence, and avoid unresolved fatal gaps.
Target final portfolio size: {target_final_count}

Hypotheses:
{json.dumps(payload, indent=2)}

Return:
- rankings: one item per hypothesis_id with score 0.0-1.0, rank, recommended_action
  (advance, hold, evolve, reject), rationale, strengths, weaknesses, improvement_directions.
- best_patterns: what the best hypotheses have in common.
- worst_patterns: what the weakest hypotheses have in common.
"""
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
        system_prompt = (
            "You are the Evolution Agent in a local AI co-scientist system. "
            "Create mutated, improved variants of promising material-opportunity hypotheses. Return strict JSON only."
        )
        user_prompt = f"""
Research goal:
{document.raw_goal}

Ranking feedback:
Best patterns:
{json.dumps(ranking_round.best_patterns, indent=2)}

Weakness patterns:
{json.dumps(ranking_round.worst_patterns, indent=2)}

Improvement guidance:
[
  "Use the ranking rationale, best patterns, and worst patterns to mutate promising hypotheses toward stronger variants."
]

Parent hypotheses:
{json.dumps(parent_payload, indent=2)}

Generate {target_count} evolved hypotheses. Use genetic-algorithm style mutations such as:
- narrowing application form
- changing buyer segment
- changing NBCA
- shifting region or activation pathway
- tightening the rPET value proposition
- reducing evidence gaps

Each evolved hypothesis must include parent_hypothesis_ids, mutation_strategy, evolution_notes, and the standard hypothesis seed fields.
Do not simply rename the parent; make a meaningful variant.
"""
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
        system_prompt = (
            "You are the Proximity Check Agent in a local AI co-scientist system. "
            "Cluster related reflected hypotheses into higher-order concepts and identify truly mergeable ideas. "
            "Return strict JSON only."
        )
        user_prompt = f"""
Research goal:
{document.raw_goal}

Research configuration:
{document.model_dump_json(indent=2)}

Hypotheses:
{json.dumps(payload, indent=2)}

Tasks:
1. Identify emerging concepts across the reflected hypotheses.
2. Label only hypotheses that genuinely belong in a concept.
3. If multiple hypotheses are similar enough that they should be combined into a higher-level opportunity,
   create at most {max_synthesized_hypotheses} synthesized hypotheses that merge the reflected insights.

Rules:
- Do not force every hypothesis into a concept.
- Do not synthesize unless the underlying ideas are genuinely overlapping and combinable.
- Synthesized hypotheses should be broader, cleaner, and more informative than the source hypotheses.
- Use merged_from_hypothesis_ids to identify the originals being superseded.
"""
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
        system_prompt = (
            "You are the Meta-review Agent in a local AI co-scientist system. "
            "Assess research-space coverage, identify whitespace, and write guidance for the next generation pass. "
            "Return strict JSON only."
        )
        user_prompt = f"""
Research goal:
{document.raw_goal}

Research configuration:
{document.model_dump_json(indent=2)}

Latest ranking patterns:
Best patterns:
{json.dumps(ranking_round.best_patterns, indent=2)}

Worst patterns:
{json.dumps(ranking_round.worst_patterns, indent=2)}

Active reflected hypotheses:
{json.dumps(payload, indent=2)}

Tasks:
1. Identify whitespace gaps versus the research goal.
2. Determine whether coverage is already sufficient.
3. Write concrete guidance for the next generation pass that targets missing areas.

Rules:
- Gaps should be substantive missing regions of the research space, not stylistic complaints.
- Generation guidance should be specific enough to drive better search-grounded hypotheses.
"""
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


class ReflectionAgent:
    def __init__(
        self,
        llm: DeepSeekLLM,
        retriever: LocalEvidenceRetriever,
        discovery_tool: DiscoveryEvidenceTool,
        price_cache: StructuredPriceCache | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._discovery_tool = discovery_tool
        self._search_planner = ReflectionSearchPlanner()
        self._price_cache = price_cache

    def reflect(self, document: ResearchGoalDocument, hypothesis: Hypothesis) -> tuple[Hypothesis, int]:
        price_document = None
        if self._price_cache is not None:
            try:
                price_document = self._price_cache.ensure_fresh()
            except Exception:
                LOGGER.exception("Structured price cache unavailable during reflection for %s", hypothesis.hypothesis_id)
        local_rows = self._retriever.retrieve_for_hypothesis(document, hypothesis)
        evidence_rows = self._augment_with_price_rows(local_rows, hypothesis, price_document)
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
                evidence_rows = self._augment_with_price_rows(refreshed_rows, hypothesis, price_document)

        final_review = self._review(document, hypothesis, evidence_rows)
        assessment = self._merge_price_metrics(
            final_review.assessment,
            hypothesis,
            price_document,
        ).model_copy(
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

    def _augment_with_price_rows(
        self,
        evidence_rows: list[dict[str, Any]],
        hypothesis: Hypothesis,
        price_document,
    ) -> list[dict[str, Any]]:
        if self._price_cache is None or price_document is None:
            return evidence_rows
        augmented: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for row in evidence_rows:
            row_id = str(row.get("id", ""))
            if row_id:
                augmented[row_id] = row
        for row in self._price_cache.build_price_evidence_rows(
            hypothesis.incumbent_material,
            hypothesis.next_best_competitive_alternative,
            hypothesis.candidate_material,
            document=price_document,
        ):
            augmented[str(row["id"])] = row
        return list(augmented.values())

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
        system_prompt = (
            "You are a reflection agent acting as a skeptical peer reviewer for industrial material hypotheses. "
            "Use only the supplied evidence. Return strict JSON only."
        )
        user_prompt = f"""
Research configuration:
{document.model_dump_json(indent=2)}

Hypothesis:
{hypothesis.model_dump_json(indent=2)}

Available evidence:
{json.dumps(evidence_payload, indent=2)}

Evaluate only the {category} dimension of the hypothesis and return JSON with:
- assessment
- needs_additional_search
- follow_up_search_queries

Focus fields:
{json.dumps(category_fields[category], indent=2)}

Leave fields outside this focus unset unless the evidence directly resolves them.
- evidence_gap_notes

For every scored or priced field:
- for score and probability fields, set value to a normalized number from 0.0 to 1.0 when supported
- for price fields, set value to a numeric USD/kg amount when supported
- otherwise set value to null
- include rationale
- include confidence from 0 to 1
- include citation_chunk_ids and citation_urls
- set is_inferred accurately

If evidence is weak or stale, set needs_additional_search to true and propose targeted web search queries using material, application, incumbent material, form, and conversion process terms.
Do not invent citations.
"""
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
        )

    def run(
        self,
        goal: str,
        target_hypotheses: int,
        regions: list[str] | None = None,
        strategic_fit_notes: str | None = None,
        preferred_evidence_recency_days: int = 180,
        max_reflection_searches_per_hypothesis: int = 3,
        results_per_query: int = 5,
        max_pages_per_search: int = 8,
        reflection_concurrency: int = 3,
        ) -> CoScientistRunResult:
        limits = ReflectionSearchLimits(
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
        document = self._planning_agent.create_research_goal(
            raw_goal=goal,
            target_hypotheses_final=target_hypotheses,
            regions=regions,
            strategic_fit_notes=strategic_fit_notes,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            reflection_search_limits=limits,
        )
        research_goal_path = self._artifact_store.save_research_goal(document)

        generated = self._generation_agent.generate(document)
        for hypothesis in generated:
            self._artifact_store.append_hypothesis_snapshot(hypothesis)

        reflected_hypotheses, automatic_discovery_runs = self._reflect_and_append(
            document,
            generated,
            concurrency=reflection_concurrency,
        )

        report_path = self._artifact_store.write_report(
            document.research_id,
            self._build_report(document, reflected_hypotheses),
        )
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
        document = self._artifact_store.load_research_goal(research_id)
        document = self._with_reflection_overrides(
            document=document,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )
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
            ranking_round, ranked_hypotheses = self._ranking_agent.rank(
                document=document,
                hypotheses=reflected,
                round_index=round_index,
                target_final_count=target_final,
                evolve_top_k=evolve_top_k,
            )
            latest_ranking = ranking_round
            self._artifact_store.append_ranking_round(ranking_round)
            for hypothesis in ranked_hypotheses:
                self._artifact_store.append_hypothesis_snapshot(hypothesis)
            rounds_completed += 1

            if proximity_check_every > 0 and round_index % proximity_check_every == 0:
                proximity_round, proximity_updates, synthesized_hypotheses = self._proximity_agent.review(
                    document=document,
                    hypotheses=self._artifact_store.latest_hypotheses(research_id),
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
                for hypothesis in synthesized_hypotheses:
                    self._artifact_store.append_hypothesis_snapshot(hypothesis)
                total_synthesized += len(synthesized_hypotheses)
                reflected_synthesized, run_count = self._reflect_and_append(
                    document,
                    synthesized_hypotheses,
                    concurrency=reflection_concurrency,
                )
                total_reflected += len(reflected_synthesized)
                automatic_discovery_runs += run_count

            latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
            document, meta_review_round = self._meta_review_agent.review(
                document=document,
                hypotheses=latest_hypotheses,
                ranking_round=ranking_round,
                round_index=round_index,
                gap_overlap_threshold=gap_overlap_threshold,
                max_gap_persistence_rounds=max_gap_persistence_rounds,
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
            evolved = self._evolution_agent.evolve(
                document=document,
                parent_hypotheses=parents,
                ranking_round=ranking_round,
                target_count=evolved_per_round,
                round_index=round_index,
            )
            regenerated = self._generation_agent.generate_from_meta_review(
                document=document,
                meta_review_round=meta_review_round,
                target_count=regenerated_per_round,
                round_index=round_index,
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
            )
            total_reflected += len(reflected_new)
            automatic_discovery_runs += run_count
            if not new_hypotheses:
                stop_reason = "no_new_hypotheses_generated"
                break

        latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
        ranked_count = len([hypothesis for hypothesis in latest_hypotheses if hypothesis.ranking_round_id])
        report_path = self._artifact_store.write_loop_report(
            research_id,
            self._build_loop_report(document, latest_ranking, latest_meta_review, latest_hypotheses, stop_reason),
        )
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
    ) -> CoScientistRunResult:
        document = self._artifact_store.load_research_goal(research_id)
        document = self._with_reflection_overrides(
            document=document,
            preferred_evidence_recency_days=preferred_evidence_recency_days,
            max_reflection_searches_per_hypothesis=max_reflection_searches_per_hypothesis,
            results_per_query=results_per_query,
            max_pages_per_search=max_pages_per_search,
        )

        latest_hypotheses = self._artifact_store.latest_hypotheses(research_id)
        pending_hypotheses = [hypothesis for hypothesis in latest_hypotheses if hypothesis.status == "generated"]
        if max_hypotheses is not None:
            pending_hypotheses = pending_hypotheses[:max_hypotheses]

        _, automatic_discovery_runs = self._reflect_and_append(
            document,
            pending_hypotheses,
            concurrency=concurrency,
        )

        refreshed_hypotheses = self._artifact_store.latest_hypotheses(research_id)
        reflected_hypotheses = [hypothesis for hypothesis in refreshed_hypotheses if hypothesis.status == "reflected"]
        report_path = self._artifact_store.write_report(
            document.research_id,
            self._build_report(document, reflected_hypotheses),
        )
        return CoScientistRunResult(
            research_id=document.research_id,
            generated_hypotheses=len(latest_hypotheses),
            reflected_hypotheses=len(reflected_hypotheses),
            automatic_discovery_runs=automatic_discovery_runs,
            research_goal_path=str(self._artifact_store.research_goal_path(document.research_id).resolve()),
            hypothesis_path=str(self._artifact_store.hypothesis_path(document.research_id).resolve()),
            report_path=str(report_path.resolve()),
        )

    def _reflect_and_append(
        self,
        document: ResearchGoalDocument,
        hypotheses: list[Hypothesis],
        concurrency: int,
    ) -> tuple[list[Hypothesis], int]:
        if not hypotheses:
            return [], 0
        worker_count = max(1, min(concurrency, len(hypotheses)))
        if worker_count == 1:
            reflected_hypotheses: list[Hypothesis] = []
            discovery_runs = 0
            for hypothesis in hypotheses:
                reflected, run_count = self._reflection_agent.reflect(document, hypothesis)
                reflected_hypotheses.append(reflected)
                discovery_runs += run_count
                self._artifact_store.append_hypothesis_snapshot(reflected)
            return reflected_hypotheses, discovery_runs

        reflected_hypotheses = []
        discovery_runs = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._reflection_agent.reflect, document, hypothesis): hypothesis
                for hypothesis in hypotheses
            }
            for future in as_completed(future_map):
                hypothesis = future_map[future]
                try:
                    reflected, run_count = future.result()
                except Exception:
                    LOGGER.exception("Reflection failed for hypothesis %s", hypothesis.hypothesis_id)
                    continue
                reflected_hypotheses.append(reflected)
                discovery_runs += run_count
                self._artifact_store.append_hypothesis_snapshot(reflected)
        reflected_hypotheses.sort(key=lambda item: [hyp.hypothesis_id for hyp in hypotheses].index(item.hypothesis_id))
        return reflected_hypotheses, discovery_runs

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
