from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4, uuid5, NAMESPACE_URL

from langgraph.graph import END, START, StateGraph

from app_discovery_agent.chunking import TextChunker
from app_discovery_agent.classify import EvidenceClassifier
from app_discovery_agent.config import AppConfig
from app_discovery_agent.embeddings import LocalEmbedder
from app_discovery_agent.extract import PageFetcher, extract_domain
from app_discovery_agent.graph_enrichment import GraphEnrichmentProposer, GraphEnrichmentStore, GraphEnrichmentValidator
from app_discovery_agent.llm import DeepSeekLLM
from app_discovery_agent.manual_ingest import ManualEvidenceIngestor
from app_discovery_agent.models import (
    ChunkRecord,
    DiscoverySummary,
    EvidenceClassification,
    GraphEnrichmentProposal,
    GraphEnrichmentValidation,
    PageContent,
    SearchQueryPlan,
    SearchResultItem,
)
from app_discovery_agent.prompt_library import PROMPTS
from app_discovery_agent.search import ExaSearchClient, deduplicate_search_results, load_search_results_file
from app_discovery_agent.store import LanceEvidenceStore


LOGGER = logging.getLogger(__name__)


class DiscoveryState(TypedDict, total=False):
    run_id: str
    original_query: str
    max_search_queries: int
    results_per_query: int
    max_pages: int
    search_queries: list[str]
    search_results: list[SearchResultItem]
    unique_results: list[SearchResultItem]
    fetched_pages: list[PageContent]
    candidate_pages: list[PageContent]
    classifications: list[dict[str, Any]]
    chunk_records: list[ChunkRecord]
    graph_enrichment_proposals: list[GraphEnrichmentProposal]
    graph_enrichment_validations: list[GraphEnrichmentValidation]
    graph_enrichment_accepted: int
    summary: DiscoverySummary
    skipped_pages: list[dict[str, Any]]
    errors: list[str]
    output_path: str


class DiscoveryAgent:
    def __init__(self, config: AppConfig):
        self._config = config
        self._config.ensure_directories()
        self._llm = DeepSeekLLM(config)
        self._search = ExaSearchClient(config)
        self._fetcher = PageFetcher(config)
        self._classifier = EvidenceClassifier(self._llm)
        self._chunker = TextChunker()
        self._embedder = LocalEmbedder(config)
        self._store = LanceEvidenceStore(config.resolved_lancedb_path())
        self._graph_enrichment_proposer = GraphEnrichmentProposer(self._llm)
        self._graph_enrichment_validator = GraphEnrichmentValidator(self._llm)
        self._graph_enrichment_store = GraphEnrichmentStore()
        self._manual_ingestor = ManualEvidenceIngestor(
            config,
            self._classifier,
            self._chunker,
            self._embedder,
            self._store,
            self._enrich_manual_records,
        )
        self._manual_ingestor.ingest_pending_files()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(DiscoveryState)
        graph.add_node("plan_search_queries", self.plan_search_queries)
        graph.add_node("run_external_search", self.run_external_search)
        graph.add_node("deduplicate_results", self.deduplicate_results)
        graph.add_node("fetch_pages", self.fetch_pages)
        graph.add_node("filter_relevance", self.filter_relevance)
        graph.add_node("classify_evidence", self.classify_evidence)
        graph.add_node("chunk_content", self.chunk_content)
        graph.add_node("embed_chunks", self.embed_chunks)
        graph.add_node("write_to_lancedb", self.write_to_lancedb)
        graph.add_node("propose_graph_enrichments", self.propose_graph_enrichments)
        graph.add_node("validate_graph_enrichments", self.validate_graph_enrichments)
        graph.add_node("write_graph_enrichments", self.write_graph_enrichments)
        graph.add_node("summarize_discoveries", self.summarize_discoveries)

        graph.add_edge(START, "plan_search_queries")
        graph.add_edge("plan_search_queries", "run_external_search")
        graph.add_edge("run_external_search", "deduplicate_results")
        graph.add_edge("deduplicate_results", "fetch_pages")
        graph.add_edge("fetch_pages", "filter_relevance")
        graph.add_edge("filter_relevance", "classify_evidence")
        graph.add_edge("classify_evidence", "chunk_content")
        graph.add_edge("chunk_content", "embed_chunks")
        graph.add_edge("embed_chunks", "write_to_lancedb")
        graph.add_edge("write_to_lancedb", "propose_graph_enrichments")
        graph.add_edge("propose_graph_enrichments", "validate_graph_enrichments")
        graph.add_edge("validate_graph_enrichments", "write_graph_enrichments")
        graph.add_edge("write_graph_enrichments", "summarize_discoveries")
        graph.add_edge("summarize_discoveries", END)
        return graph.compile()

    def discover(
        self,
        query: str,
        max_search_queries: int = 8,
        results_per_query: int = 5,
        max_pages: int = 20,
    ) -> DiscoverySummary:
        state: DiscoveryState = {
            "run_id": str(uuid4()),
            "original_query": query,
            "max_search_queries": max_search_queries,
            "results_per_query": results_per_query,
            "max_pages": max_pages,
            "skipped_pages": [],
            "errors": [],
        }
        final_state = self._graph.invoke(state)
        return final_state["summary"]

    def replay_discovery(
        self,
        query: str,
        search_results_path: Path | None = None,
        fetched_pages_path: Path | None = None,
        max_pages: int = 20,
    ) -> DiscoverySummary:
        state: DiscoveryState = {
            "run_id": str(uuid4()),
            "original_query": query,
            "max_search_queries": 0,
            "results_per_query": 0,
            "max_pages": max_pages,
            "search_queries": [],
            "skipped_pages": [],
            "errors": [],
        }

        if search_results_path and search_results_path.exists():
            search_results = load_search_results_file(search_results_path)
            state["search_results"] = search_results
            state.update(self.deduplicate_results(state))
        else:
            state["search_results"] = []
            state["unique_results"] = []

        if fetched_pages_path and fetched_pages_path.exists():
            fetched_pages, preload_skips = self._load_cached_fetched_pages(fetched_pages_path, max_pages=max_pages)
            state["fetched_pages"] = fetched_pages
            state["skipped_pages"] = state.get("skipped_pages", []) + preload_skips
        else:
            fetched_update = self.fetch_pages(state)
            state.update(fetched_update)

        state.update(self.filter_relevance(state))
        state.update(self.classify_evidence(state))
        state.update(self.chunk_content(state))
        state.update(self.embed_chunks(state))
        state.update(self.write_to_lancedb(state))
        state.update(self.propose_graph_enrichments(state))
        state.update(self.validate_graph_enrichments(state))
        state.update(self.write_graph_enrichments(state))
        state.update(self.summarize_discoveries(state))
        return state["summary"]

    def plan_search_queries(self, state: DiscoveryState) -> DiscoveryState:
        system_prompt = PROMPTS.render("discovery_agent", "plan_search_queries.system")
        example_output = json.dumps(
            {
                "queries": [
                    "material application market segment form performance requirements",
                    "material thermoformed tray clarity impact resistance food packaging alternatives",
                    "material vs glass stainless steel aluminum application requirements substitution",
                ]
            },
            indent=2,
        )
        user_prompt = PROMPTS.render(
            "discovery_agent",
            "plan_search_queries.user",
            original_query=state["original_query"],
            max_search_queries=state["max_search_queries"],
            example_output=example_output,
        )
        plan = self._llm.complete_json(SearchQueryPlan, system_prompt, user_prompt)
        queries = plan.queries[: state["max_search_queries"]]
        if state["original_query"] not in queries:
            queries.insert(0, state["original_query"])
        return {"search_queries": queries[: state["max_search_queries"]]}

    def run_external_search(self, state: DiscoveryState) -> DiscoveryState:
        all_results: list[SearchResultItem] = []
        raw_payloads: list[dict[str, Any]] = []
        for query in state["search_queries"]:
            try:
                response = self._search.search(query=query, num_results=state["results_per_query"])
                all_results.extend(response.results)
                raw_payloads.append({"query": query, "payload": response.raw_payload})
            except Exception as exc:
                LOGGER.exception("Search failed for query %s", query)
                state.setdefault("errors", []).append(f"search:{query}:{exc}")

        raw_path = Path("data/raw") / f'{state["run_id"]}_search_results.json'
        raw_path.write_text(json.dumps(raw_payloads, indent=2), encoding="utf-8")
        return {"search_results": all_results}

    def deduplicate_results(self, state: DiscoveryState) -> DiscoveryState:
        unique_results = deduplicate_search_results(state.get("search_results", []))
        return {"unique_results": unique_results}

    def fetch_pages(self, state: DiscoveryState) -> DiscoveryState:
        fetched_pages: list[PageContent] = []
        skipped = list(state.get("skipped_pages", []))
        for result in state.get("unique_results", [])[: state["max_pages"]]:
            if self._fetcher.should_skip_direct_fetch(str(result.url)):
                skipped.append(
                    {
                        "url": str(result.url),
                        "search_query": result.search_query,
                        "reason": "blocked_domain",
                        "error": "Domain skipped for direct fetch based on configured policy",
                    }
                )
                fallback_page = self._build_partial_page_from_search_result(result, "blocked_domain")
                if fallback_page:
                    fetched_pages.append(fallback_page)
                continue
            page, error = self._fetcher.safe_fetch(result)
            if error:
                skipped.append(error)
                fallback_page = self._build_partial_page_from_search_result(result, error.get("reason", "fetch_error"))
                if fallback_page:
                    fetched_pages.append(fallback_page)
                continue
            if page:
                fetched_pages.append(page)
        raw_pages_path = Path("data/raw") / f'{state["run_id"]}_fetched_pages.json'
        raw_pages_path.write_text(
            json.dumps(
                [
                    {
                        "title": page.title,
                        "url": str(page.url),
                        "search_query": page.search_query,
                        "source_domain": page.source_domain,
                        "fetched_at": page.fetched_at.isoformat(),
                        "status_code": page.status_code,
                        "content_type": page.content_type,
                        "text": page.text,
                        "metadata": page.metadata,
                    }
                    for page in fetched_pages
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"fetched_pages": fetched_pages, "skipped_pages": skipped}

    def filter_relevance(self, state: DiscoveryState) -> DiscoveryState:
        candidates: list[PageContent] = []
        skipped = list(state.get("skipped_pages", []))
        for page in state.get("fetched_pages", []):
            is_partial = bool(page.metadata.get("is_partial_evidence"))
            min_characters = self._config.min_snippet_characters if is_partial else self._config.min_page_characters
            if len(page.text) < min_characters:
                skipped.append({"url": str(page.url), "reason": "too_little_text"})
                continue
            heuristic_score = self._classifier.heuristic_relevance(state["original_query"], page.text)
            metadata = {
                **page.metadata,
                "heuristic_relevance_score": heuristic_score,
                "retention_policy": "retain_fetched_text_for_reflection",
            }
            candidates.append(page.model_copy(update={"metadata": metadata}))
        return {"candidate_pages": candidates, "skipped_pages": skipped}

    def classify_evidence(self, state: DiscoveryState) -> DiscoveryState:
        classifications: list[dict[str, Any]] = []
        skipped = list(state.get("skipped_pages", []))
        for page in state.get("candidate_pages", []):
            try:
                classification = self._classifier.classify(state["original_query"], page)
            except Exception as exc:
                LOGGER.exception("Classification failed for %s", page.url)
                state.setdefault("errors", []).append(f"classify:{page.url}:{exc}")
                skipped.append({"url": str(page.url), "reason": "classification_error", "error": str(exc)})
                continue

            classifications.append({"page": page, "classification": classification})
        return {"classifications": classifications, "skipped_pages": skipped}

    def chunk_content(self, state: DiscoveryState) -> DiscoveryState:
        chunk_records: list[ChunkRecord] = []
        for item in state.get("classifications", []):
            page: PageContent = item["page"]
            classification: EvidenceClassification = item["classification"]
            chunks = self._chunker.chunk_text(page.text)
            for index, chunk in enumerate(chunks):
                chunk_id = str(uuid5(NAMESPACE_URL, f"{state['run_id']}::{page.url}::{index}"))
                chunk_records.append(
                    ChunkRecord(
                        id=chunk_id,
                        run_id=state["run_id"],
                        original_query=state["original_query"],
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
                            "retention_policy": page.metadata.get("retention_policy", "retain_classified_text_for_reflection"),
                            "page_metadata": page.metadata,
                        },
                    )
                )
        return {"chunk_records": chunk_records}

    def embed_chunks(self, state: DiscoveryState) -> DiscoveryState:
        records = state.get("chunk_records", [])
        vectors = self._embedder.embed_texts([record.chunk_text for record in records])
        embedded_records = [record.model_copy(update={"vector": vector}) for record, vector in zip(records, vectors, strict=False)]
        return {"chunk_records": embedded_records}

    def write_to_lancedb(self, state: DiscoveryState) -> DiscoveryState:
        stored_count = self._store.add_chunks(state.get("chunk_records", []))
        LOGGER.info("Stored %s chunks in LanceDB", stored_count)
        return state

    def propose_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            proposals = self._graph_enrichment_proposer.propose(
                state["original_query"],
                state.get("chunk_records", []),
            )
            return {"graph_enrichment_proposals": proposals}
        except Exception as exc:
            LOGGER.exception("Graph enrichment proposal failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_proposal:{exc}")
            return {"graph_enrichment_proposals": []}

    def validate_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            validations = self._graph_enrichment_validator.validate(
                state.get("graph_enrichment_proposals", []),
                state.get("chunk_records", []),
            )
            return {"graph_enrichment_validations": validations}
        except Exception as exc:
            LOGGER.exception("Graph enrichment validation failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_validation:{exc}")
            return {"graph_enrichment_validations": []}

    def write_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            accepted = self._graph_enrichment_store.write(
                state.get("graph_enrichment_proposals", []),
                state.get("graph_enrichment_validations", []),
                state["run_id"],
                state["original_query"],
            )
            raw_path = Path("data/raw") / f'{state["run_id"]}_graph_enrichments.json'
            raw_path.write_text(
                json.dumps(
                    {
                        "proposals": [
                            proposal.model_dump(mode="json")
                            for proposal in state.get("graph_enrichment_proposals", [])
                        ],
                        "validations": [
                            validation.model_dump(mode="json")
                            for validation in state.get("graph_enrichment_validations", [])
                        ],
                        "accepted": accepted,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {"graph_enrichment_accepted": accepted}
        except Exception as exc:
            LOGGER.exception("Graph enrichment write failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_write:{exc}")
            return {"graph_enrichment_accepted": 0}

    def summarize_discoveries(self, state: DiscoveryState) -> DiscoveryState:
        notable_applications = sorted(
            {
                item["classification"].application
                for item in state.get("classifications", [])
                if item["classification"].application
            }
        )
        evidence_preview = [
            {
                "application": record.application,
                "evidence_type": record.evidence_type,
                "source_title": record.source_title,
                "source_url": str(record.source_url),
                "relevance_score": record.relevance_score,
                "confidence_score": record.confidence_score,
                "chunk_id": record.id,
            }
            for record in state.get("chunk_records", [])[:20]
        ]
        system_prompt = PROMPTS.render("discovery_agent", "summarize_discoveries.system")
        user_prompt = PROMPTS.render(
            "discovery_agent",
            "summarize_discoveries.user",
            original_query=state["original_query"],
            evidence_preview_json=json.dumps(evidence_preview, indent=2),
        )
        narrative = self._llm.complete_text(system_prompt, user_prompt)

        output_path = Path("data/outputs") / f'{state["run_id"]}_summary.md'
        output_path.write_text(narrative, encoding="utf-8")

        summary = DiscoverySummary(
            run_id=state["run_id"],
            original_query=state["original_query"],
            total_search_queries=len(state.get("search_queries", [])),
            total_search_results=len(state.get("search_results", [])),
            unique_urls=len(state.get("unique_results", [])),
            fetched_pages=len(state.get("fetched_pages", [])),
            relevant_pages=len(state.get("classifications", [])),
            stored_chunks=len(state.get("chunk_records", [])),
            graph_enrichment_proposals=len(state.get("graph_enrichment_proposals", [])),
            graph_enrichment_accepted=state.get("graph_enrichment_accepted", 0),
            opportunity_summary=narrative,
            notable_applications=notable_applications,
            evidence_gaps=[
                "Direct comparative performance data may still be missing for many applications.",
                "Commercial fit should be validated with application-specific requirements and testing.",
            ],
            recommended_next_steps=[
                "Prioritize applications with explicit PVC use evidence and documented clear rigid requirements.",
                "Seek supplier datasheets, regulatory notes, and competitor positioning for shortlisted applications.",
            ],
            output_path=str(output_path.resolve()),
        )
        summary_json_path = Path("data/outputs") / f'{state["run_id"]}_summary.json'
        summary_json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        skipped_path = Path("data/raw") / f'{state["run_id"]}_skipped_pages.json'
        skipped_path.write_text(json.dumps(state.get("skipped_pages", []), indent=2), encoding="utf-8")
        errors_path = Path("data/raw") / f'{state["run_id"]}_errors.json'
        errors_path.write_text(json.dumps(state.get("errors", []), indent=2), encoding="utf-8")
        return {"summary": summary, "output_path": str(output_path.resolve())}

    def _load_cached_fetched_pages(self, path: Path, max_pages: int) -> tuple[list[PageContent], list[dict[str, Any]]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages: list[PageContent] = []
        skipped: list[dict[str, Any]] = []

        for entry in payload[:max_pages]:
            content_type = (entry.get("content_type") or "").lower()
            text = entry.get("text") or ""
            if ("application/pdf" in content_type and text.startswith("%PDF-")) or (not text.strip()):
                skipped.append(
                    {
                        "url": entry.get("url"),
                        "search_query": entry.get("search_query"),
                        "reason": "unsupported_cached_content_type",
                        "content_type": content_type or "application/pdf",
                    }
                )
                continue
            try:
                pages.append(PageContent.model_validate(entry))
            except Exception as exc:
                skipped.append(
                    {
                        "url": entry.get("url"),
                        "search_query": entry.get("search_query"),
                        "reason": "invalid_cached_page",
                        "error": str(exc),
                    }
                )
        return pages, skipped

    def _build_partial_page_from_search_result(self, result: SearchResultItem, reason: str) -> PageContent | None:
        partial_text = self._compose_partial_text(result)
        if len(partial_text) < self._config.min_snippet_characters:
            return None
        return PageContent(
            title=result.title,
            url=str(result.url),
            search_query=result.search_query,
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

    @staticmethod
    def _compose_partial_text(result: SearchResultItem) -> str:
        parts = [result.title.strip(), result.summary.strip(), result.snippet.strip()]
        return "\n\n".join(part for part in parts if part)

    def _enrich_manual_records(self, original_query: str, records: list[ChunkRecord]) -> None:
        try:
            proposals = self._graph_enrichment_proposer.propose(original_query, records)
            validations = self._graph_enrichment_validator.validate(proposals, records)
            accepted = self._graph_enrichment_store.write(proposals, validations, records[0].run_id if records else "manual", original_query)
            LOGGER.info(
                "Graph enrichment from manual evidence produced %s proposals and %s accepted claims",
                len(proposals),
                accepted,
            )
        except Exception:
            LOGGER.exception("Manual graph enrichment failed")

    @property
    def store(self) -> LanceEvidenceStore:
        return self._store

    @property
    def embedder(self) -> LocalEmbedder:
        return self._embedder

    @property
    def llm(self) -> DeepSeekLLM:
        return self._llm


def build_opportunity_report(
    llm: DeepSeekLLM,
    rows: list[dict[str, Any]],
    incumbent_material: str,
    candidate_material: str,
) -> str:
    system_prompt = PROMPTS.render("discovery_agent", "opportunity_report.system")
    user_prompt = PROMPTS.render(
        "discovery_agent",
        "opportunity_report.user",
        incumbent_material=incumbent_material,
        candidate_material=candidate_material,
        rows_json=json.dumps(rows[:30], indent=2),
    )
    return llm.complete_text(system_prompt, user_prompt)
