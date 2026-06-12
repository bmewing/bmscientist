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
from app_discovery_agent.extract import PageFetcher
from app_discovery_agent.llm import DeepSeekLLM
from app_discovery_agent.models import (
    ChunkRecord,
    DiscoverySummary,
    EvidenceClassification,
    PageContent,
    SearchQueryPlan,
    SearchResultItem,
)
from app_discovery_agent.search import ExaSearchClient, deduplicate_search_results
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
        graph.add_edge("write_to_lancedb", "summarize_discoveries")
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

    def plan_search_queries(self, state: DiscoveryState) -> DiscoveryState:
        system_prompt = (
            "You create precise web search queries for technical application discovery. "
            "Return JSON only."
        )
        user_prompt = f"""
Original query:
{state["original_query"]}

Generate up to {state["max_search_queries"]} search queries that can uncover:
- applications currently using PVC
- clear rigid or semi-rigid product requirements
- references to PET, PETG, or Eastman Tritan as alternatives or comparable materials
- sustainability, regulatory, or customer pressure affecting material choices

Favor targeted, evidence-rich queries over generic ones.
Return JSON with a single field: queries
"""
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
            page, error = self._fetcher.safe_fetch(result)
            if error:
                skipped.append({"reason": "fetch_error", **error})
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
            if len(page.text) < self._config.min_page_characters:
                skipped.append({"url": str(page.url), "reason": "too_little_text"})
                continue
            heuristic_score = self._classifier.heuristic_relevance(state["original_query"], page.text)
            if heuristic_score < 0.2:
                skipped.append({"url": str(page.url), "reason": "low_heuristic_relevance", "score": heuristic_score})
                continue
            candidates.append(page)
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

            if classification.relevance_score < self._config.min_relevance_score or not classification.relevant:
                skipped.append(
                    {
                        "url": str(page.url),
                        "reason": "below_relevance_threshold",
                        "relevance_score": classification.relevance_score,
                    }
                )
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
        system_prompt = (
            "You write conservative research summaries for materials opportunity discovery. "
            "Do not make commercial suitability claims that the evidence does not support."
        )
        user_prompt = f"""
Original query:
{state["original_query"]}

Evidence preview:
{json.dumps(evidence_preview, indent=2)}

Write a concise summary with:
- plausible application clusters
- what evidence currently supports
- what evidence is missing
- next best research steps

Keep the tone cautious and cite chunk IDs and URLs when referencing evidence.
"""
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
    system_prompt = (
        "You summarize local evidence for materials substitution opportunities. "
        "Be conservative and never claim fit without evidence."
    )
    user_prompt = f"""
Incumbent material: {incumbent_material}
Candidate material: {candidate_material}

Evidence rows:
{json.dumps(rows[:30], indent=2)}

Write a concise report that:
- highlights promising application areas
- distinguishes direct evidence from partial evidence
- references chunk IDs and source URLs
- avoids final commercial claims
"""
    return llm.complete_text(system_prompt, user_prompt)
