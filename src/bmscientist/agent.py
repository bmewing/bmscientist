from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from uuid import uuid4, uuid5, NAMESPACE_URL

from langgraph.graph import END, START, StateGraph

from bmscientist.chunking import TextChunker
from bmscientist.classify import EvidenceClassifier
from bmscientist.config import AppConfig
from bmscientist.embeddings import LocalEmbedder
from bmscientist.extract import PageFetcher, extract_domain
from bmscientist.graph_enrichment import (
    GraphEnrichmentExpander,
    GraphEnrichmentProposer,
    GraphEnrichmentStore,
    GraphEnrichmentValidator,
)
from bmscientist.llm import DeepSeekLLM
from bmscientist.manual_ingest import ManualEvidenceIngestor
from bmscientist.models import (
    ChunkRecord,
    DiscoverySummary,
    EvidenceClassification,
    GraphEnrichmentProposal,
    GraphEnrichmentValidation,
    PageContent,
    SearchQueryPlan,
    SearchResultItem,
)
from bmscientist.prompt_library import PROMPTS
from bmscientist.retrieval import ExaPageRetriever, build_partial_page_from_search_result, compose_partial_text
from bmscientist.search import ExaSearchClient, deduplicate_search_results, default_search_options, load_search_results_file
from bmscientist.skills import (
    EPISuiteSkill,
    MoleculeAvailabilitySkill,
    MoleculeIdentityPubChemSkill,
    PubChemProfileSkill,
    RDKitProfileSkill,
    SafetyTriageSkill,
    SkillContext,
    SkillRegistry,
    SkillRunner,
)
from bmscientist.store import LanceEvidenceStore


LOGGER = logging.getLogger(__name__)

GRAPH_ENRICHMENT_SKILL_IDS = (
    "molecule_identity_pubchem",
    "rdkit_profile",
    "pubchem_profile",
    "safety_triage",
    "molecule_availability",
    "epa_episuite",
)


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
    graph_enrichment_follow_up_questions: list[dict[str, Any]]
    graph_enrichment_expansion_proposals: list[GraphEnrichmentProposal]
    graph_enrichment_expansion_validations: list[GraphEnrichmentValidation]
    graph_enrichment_external_search_queries: list[str]
    graph_enrichment_skill_outputs: list[dict[str, Any]]
    graph_enrichment_skill_writes: int
    graph_enrichment_accepted: int
    summary: DiscoverySummary
    skipped_pages: list[dict[str, Any]]
    retrieval_stats: dict[str, Any]
    errors: list[str]
    output_path: str


class DiscoveryAgent:
    def __init__(self, config: AppConfig):
        self._config = config
        self._config.ensure_directories()
        self._llm = DeepSeekLLM(config)
        self._search = ExaSearchClient(config)
        self._fetcher = PageFetcher(config)
        self._retriever = ExaPageRetriever(config, self._search, self._fetcher)
        self._classifier = EvidenceClassifier(self._llm)
        self._chunker = TextChunker()
        self._embedder = LocalEmbedder(config)
        self._store = LanceEvidenceStore(config.resolved_lancedb_path())
        self._graph_enrichment_proposer = GraphEnrichmentProposer(self._llm)
        self._graph_enrichment_validator = GraphEnrichmentValidator(self._llm)
        self._graph_enrichment_expander = GraphEnrichmentExpander(self._llm)
        self._graph_enrichment_store = GraphEnrichmentStore()
        self._graph_skill_runner = SkillRunner(
            SkillRegistry(
                [
                    SafetyTriageSkill(config),
                    MoleculeIdentityPubChemSkill(config),
                    RDKitProfileSkill(config),
                    PubChemProfileSkill(config),
                    MoleculeAvailabilitySkill(config),
                    EPISuiteSkill(config),
                ]
            )
        )
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
        graph.add_node("run_enrichment_skills", self.run_enrichment_skills)
        graph.add_node("propose_graph_enrichments", self.propose_graph_enrichments)
        graph.add_node("validate_graph_enrichments", self.validate_graph_enrichments)
        graph.add_node("expand_graph_enrichments", self.expand_graph_enrichments)
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
        graph.add_edge("write_to_lancedb", "run_enrichment_skills")
        graph.add_edge("run_enrichment_skills", "propose_graph_enrichments")
        graph.add_edge("propose_graph_enrichments", "validate_graph_enrichments")
        graph.add_edge("validate_graph_enrichments", "expand_graph_enrichments")
        graph.add_edge("expand_graph_enrichments", "write_graph_enrichments")
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
        state.update(self.expand_graph_enrichments(state))
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
                response = self._search.search(
                    query=query,
                    num_results=state["results_per_query"],
                    options=default_search_options(self._config, query),
                )
                all_results.extend(response.results)
                raw_payloads.append({"query": query, "payload": response.raw_payload})
            except Exception as exc:
                LOGGER.exception("Search failed for query %s", query)
                state.setdefault("errors", []).append(f"search:{query}:{exc}")

        raw_path = self._config.data_dir / "raw" / f'{state["run_id"]}_search_results.json'
        raw_path.write_text(json.dumps(raw_payloads, indent=2), encoding="utf-8")
        return {"search_results": all_results}

    def deduplicate_results(self, state: DiscoveryState) -> DiscoveryState:
        unique_results = deduplicate_search_results(state.get("search_results", []))
        return {"unique_results": unique_results}

    def fetch_pages(self, state: DiscoveryState) -> DiscoveryState:
        retrieval = self._retriever.retrieve_pages(
            state["original_query"],
            state.get("unique_results", [])[: state["max_pages"]],
            max_pages=state["max_pages"],
        )
        fetched_pages = retrieval.pages
        skipped = list(state.get("skipped_pages", [])) + retrieval.skipped
        raw_pages_path = self._config.data_dir / "raw" / f'{state["run_id"]}_fetched_pages.json'
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
        retrieval_stats_path = self._config.data_dir / "raw" / f'{state["run_id"]}_retrieval_stats.json'
        retrieval_stats_path.write_text(json.dumps(retrieval.stats, indent=2), encoding="utf-8")
        return {"fetched_pages": fetched_pages, "skipped_pages": skipped, "retrieval_stats": retrieval.stats}

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

    def run_enrichment_skills(self, state: DiscoveryState) -> DiscoveryState:
        try:
            available_skills, skill_outputs, write_count = self._run_graph_enrichment_skills_for_records(
                state["original_query"],
                state.get("chunk_records", []),
            )
            LOGGER.info(
                "Graph enrichment skills produced %s outputs and %s graph writes",
                len(skill_outputs),
                write_count,
            )
            return {
                "graph_enrichment_skill_outputs": skill_outputs,
                "graph_enrichment_skill_writes": write_count,
            }
        except Exception as exc:
            LOGGER.exception("Graph enrichment skill run failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_skills:{exc}")
            return {
                "graph_enrichment_skill_outputs": [],
                "graph_enrichment_skill_writes": 0,
            }

    def propose_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            available_skills = self._available_enrichment_skill_catalog()
            skill_outputs = self._filter_skill_outputs_for_records(
                state.get("graph_enrichment_skill_outputs", []),
                state.get("chunk_records", []),
            )
            proposals = self._propose_with_skill_context(
                state["original_query"],
                state.get("chunk_records", []),
                available_skills,
                skill_outputs,
            )
            return {"graph_enrichment_proposals": proposals}
        except Exception as exc:
            LOGGER.exception("Graph enrichment proposal failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_proposal:{exc}")
            return {"graph_enrichment_proposals": []}

    def validate_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            skill_outputs = self._filter_skill_outputs_for_records(
                state.get("graph_enrichment_skill_outputs", []),
                state.get("chunk_records", []),
            )
            validations = self._validate_with_skill_context(
                state.get("graph_enrichment_proposals", []),
                state.get("chunk_records", []),
                skill_outputs,
            )
            return {"graph_enrichment_validations": validations}
        except Exception as exc:
            LOGGER.exception("Graph enrichment validation failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_validation:{exc}")
            return {"graph_enrichment_validations": []}

    def expand_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            current_skill_outputs = self._filter_skill_outputs_for_records(
                state.get("graph_enrichment_skill_outputs", []),
                state.get("chunk_records", []),
            )
            questions, proposals = self._graph_enrichment_expander.expand(
                state["original_query"],
                state.get("graph_enrichment_proposals", []),
                state.get("graph_enrichment_validations", []),
                state.get("chunk_records", []),
            )
            validations = (
                self._validate_with_skill_context(
                    proposals,
                    state.get("chunk_records", []),
                    current_skill_outputs,
                )
                if proposals
                else []
            )
            external_queries = self._select_graph_follow_up_search_questions(
                questions,
                state.get("graph_enrichment_proposals", []),
                state.get("graph_enrichment_validations", []),
                proposals,
                validations,
            )
            external_records = self._run_graph_follow_up_search(state, external_queries) if external_queries else []
            external_skill_outputs: list[dict[str, Any]] = []
            external_skill_writes = 0
            if external_records:
                _available_skills, external_skill_outputs, external_skill_writes = self._run_graph_enrichment_skills_for_records(
                    state["original_query"],
                    external_records,
                )
            external_proposals = (
                self._propose_with_skill_context(
                    state["original_query"],
                    external_records,
                    self._available_enrichment_skill_catalog(),
                    external_skill_outputs,
                )
                if external_records
                else []
            )
            external_validations = (
                self._validate_with_skill_context(
                    external_proposals,
                    external_records,
                    external_skill_outputs,
                )
                if external_proposals
                else []
            )
            merged_chunk_records = self._merge_chunk_records(state.get("chunk_records", []), external_records)
            merged_skill_outputs = [*state.get("graph_enrichment_skill_outputs", []), *external_skill_outputs]
            return {
                "graph_enrichment_follow_up_questions": [item.model_dump(mode="json") for item in questions],
                "graph_enrichment_external_search_queries": [item.question for item in external_queries],
                "graph_enrichment_expansion_proposals": [*proposals, *external_proposals],
                "graph_enrichment_expansion_validations": [*validations, *external_validations],
                "graph_enrichment_skill_outputs": merged_skill_outputs,
                "graph_enrichment_skill_writes": state.get("graph_enrichment_skill_writes", 0) + external_skill_writes,
                "chunk_records": merged_chunk_records,
            }
        except Exception as exc:
            LOGGER.exception("Graph enrichment expansion failed for run %s", state.get("run_id"))
            state.setdefault("errors", []).append(f"graph_enrichment_expansion:{exc}")
            return {
                "graph_enrichment_follow_up_questions": [],
                "graph_enrichment_external_search_queries": [],
                "graph_enrichment_expansion_proposals": [],
                "graph_enrichment_expansion_validations": [],
            }

    def write_graph_enrichments(self, state: DiscoveryState) -> DiscoveryState:
        try:
            proposals = [
                *state.get("graph_enrichment_proposals", []),
                *state.get("graph_enrichment_expansion_proposals", []),
            ]
            validations = [
                *state.get("graph_enrichment_validations", []),
                *state.get("graph_enrichment_expansion_validations", []),
            ]
            accepted = self._graph_enrichment_store.write(
                proposals,
                validations,
                state["run_id"],
                state["original_query"],
            )
            raw_path = self._config.data_dir / "raw" / f'{state["run_id"]}_graph_enrichments.json'
            raw_path.write_text(
                json.dumps(
                    {
                        "proposals": [
                            proposal.model_dump(mode="json")
                            for proposal in proposals
                        ],
                        "validations": [
                            validation.model_dump(mode="json")
                            for validation in validations
                        ],
                        "follow_up_questions": state.get("graph_enrichment_follow_up_questions", []),
                        "external_search_queries": state.get("graph_enrichment_external_search_queries", []),
                        "skill_outputs": state.get("graph_enrichment_skill_outputs", []),
                        "skill_writes": state.get("graph_enrichment_skill_writes", 0),
                        "initial_proposal_count": len(state.get("graph_enrichment_proposals", [])),
                        "expansion_proposal_count": len(state.get("graph_enrichment_expansion_proposals", [])),
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

        output_path = self._config.data_dir / "outputs" / f'{state["run_id"]}_summary.md'
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
            graph_enrichment_proposals=len(state.get("graph_enrichment_proposals", []))
            + len(state.get("graph_enrichment_expansion_proposals", [])),
            graph_enrichment_accepted=state.get("graph_enrichment_accepted", 0),
            graph_enrichment_follow_up_questions=len(state.get("graph_enrichment_follow_up_questions", [])),
            graph_enrichment_expansion_proposals=len(state.get("graph_enrichment_expansion_proposals", [])),
            graph_enrichment_external_search_queries=len(state.get("graph_enrichment_external_search_queries", [])),
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
        summary_json_path = self._config.data_dir / "outputs" / f'{state["run_id"]}_summary.json'
        summary_json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        skipped_path = self._config.data_dir / "raw" / f'{state["run_id"]}_skipped_pages.json'
        skipped_path.write_text(json.dumps(state.get("skipped_pages", []), indent=2), encoding="utf-8")
        errors_path = self._config.data_dir / "raw" / f'{state["run_id"]}_errors.json'
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
        partial_text = compose_partial_text(result)
        if len(partial_text) < self._config.min_snippet_characters:
            return None
        return build_partial_page_from_search_result(result, reason)

    @staticmethod
    def _compose_partial_text(result: SearchResultItem) -> str:
        return compose_partial_text(result)

    def _run_graph_enrichment_skills_for_records(
        self,
        original_query: str,
        records: list[ChunkRecord],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        available_skills = self._available_enrichment_skill_catalog()
        if not hasattr(self, "_graph_skill_runner"):
            return available_skills, [], 0
        outputs: list[dict[str, Any]] = []
        writes = 0
        for record, candidate_artifact in self._iter_enrichment_candidates(records):
            context = self._build_enrichment_skill_context(original_query, record, candidate_artifact)
            skill_results = self._graph_skill_runner.run_auto(context)
            completed_results = [result for result in skill_results if result.status == "completed"]
            if not completed_results:
                continue
            enriched_artifact = self._merge_candidate_artifact(candidate_artifact, completed_results)
            outputs.append(
                {
                    "source_chunk_id": record.id,
                    "source_url": record.source_url,
                    "source_title": record.source_title,
                    "candidate_artifact": enriched_artifact,
                    "skill_results": [result.as_prompt_dict() for result in skill_results],
                }
            )
            writes += self._graph_enrichment_store.write_skill_enrichments(
                candidate_artifact=enriched_artifact,
                skill_results=completed_results,
                source_chunk_id=record.id,
                source_url=record.source_url,
                source_title=record.source_title,
                supporting_quote=record.chunk_text[:1200],
                confidence=max(0.7, float(record.confidence_score or 0.0)),
            )
        return available_skills, outputs, writes

    def _available_enrichment_skill_catalog(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_graph_skill_runner"):
            return []
        return [
            item
            for item in self._graph_skill_runner.all_skill_catalog()
            if "enrichment" in item.get("phases", [])
        ]

    def _filter_skill_outputs_for_records(
        self,
        skill_outputs: list[dict[str, Any]],
        records: list[ChunkRecord],
    ) -> list[dict[str, Any]]:
        if not skill_outputs or not records:
            return []
        chunk_ids = {record.id for record in records}
        return [row for row in skill_outputs if row.get("source_chunk_id") in chunk_ids]

    def _propose_with_skill_context(
        self,
        original_query: str,
        records: list[ChunkRecord],
        available_skills: list[dict[str, Any]],
        skill_outputs: list[dict[str, Any]],
    ) -> list[GraphEnrichmentProposal]:
        try:
            return self._graph_enrichment_proposer.propose(
                original_query,
                records,
                available_skills=available_skills,
                skill_outputs=skill_outputs,
            )
        except TypeError:
            return self._graph_enrichment_proposer.propose(original_query, records)

    def _validate_with_skill_context(
        self,
        proposals: list[GraphEnrichmentProposal],
        records: list[ChunkRecord],
        skill_outputs: list[dict[str, Any]],
    ) -> list[GraphEnrichmentValidation]:
        try:
            return self._graph_enrichment_validator.validate(
                proposals,
                records,
                skill_outputs=skill_outputs,
            )
        except TypeError:
            return self._graph_enrichment_validator.validate(proposals, records)

    def _iter_enrichment_candidates(self, records: list[ChunkRecord]) -> list[tuple[ChunkRecord, dict[str, Any]]]:
        candidates: list[tuple[ChunkRecord, dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            metadata = record.metadata or {}
            explicit_artifacts = metadata.get("candidate_artifacts")
            if isinstance(explicit_artifacts, list):
                for item in explicit_artifacts:
                    if not isinstance(item, dict):
                        continue
                    artifact = dict(item)
                    label = self._artifact_label(artifact)
                    if not label:
                        continue
                    key = (record.id, json.dumps(artifact, sort_keys=True, default=str))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((record, artifact))
            for name in [*record.candidate_materials, record.incumbent_material]:
                text = str(name or "").strip()
                if not text:
                    continue
                artifact = {"name_or_label": text}
                key = (record.id, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((record, artifact))
        return candidates

    def _build_enrichment_skill_context(
        self,
        original_query: str,
        record: ChunkRecord,
        candidate_artifact: dict[str, Any],
    ) -> SkillContext:
        primary_identifier_field = "name_or_label"
        if candidate_artifact.get("canonical_smiles") or candidate_artifact.get("smiles"):
            primary_identifier_field = "smiles"
        elif candidate_artifact.get("inchi"):
            primary_identifier_field = "inchi"
        elif candidate_artifact.get("cas_number") or candidate_artifact.get("cas"):
            primary_identifier_field = "cas_number"
        document = SimpleNamespace(
            research_mode="candidate_design",
            candidate_artifact_schema=SimpleNamespace(primary_identifier_field=primary_identifier_field),
            evaluation_criteria=[],
            reflection_guidance=[],
            novelty_check_policy="",
            known_candidate_exclusion_terms=[],
        )
        hypothesis = SimpleNamespace(
            title=self._artifact_label(candidate_artifact) or record.incumbent_material or "Candidate",
            summary=record.chunk_text[:1200],
            candidate_material=self._artifact_label(candidate_artifact),
            application=record.application,
            incumbent_material=record.incumbent_material,
            candidate_artifact=dict(candidate_artifact),
        )
        return SkillContext(
            phase="enrichment",
            document=document,
            hypothesis=hypothesis,
            purpose=f"Enrich graph material nodes for query: {original_query}",
            requested_skill_ids=GRAPH_ENRICHMENT_SKILL_IDS,
            evidence_rows=(
                {
                    "id": record.id,
                    "source_url": record.source_url,
                    "source_title": record.source_title,
                    "application": record.application,
                    "incumbent_material": record.incumbent_material,
                    "candidate_materials": record.candidate_materials,
                    "relevance_score": record.relevance_score,
                    "chunk_text": record.chunk_text[:1800],
                    "metadata": record.metadata,
                },
            ),
            metadata={"source_chunk_id": record.id},
        )

    @staticmethod
    def _merge_candidate_artifact(candidate_artifact: dict[str, Any], skill_results: list[Any]) -> dict[str, Any]:
        merged = dict(candidate_artifact)
        for result in skill_results:
            for key, value in getattr(result, "resolved_identifiers", {}).items():
                if value not in (None, "", []):
                    merged[key] = value
            cid = getattr(result, "metadata", {}).get("cid")
            if cid not in (None, "", []):
                merged["cid"] = cid
        if "inchikey" in merged and "inchi_key" not in merged:
            merged["inchi_key"] = merged["inchikey"]
        if "canonical_smiles" in merged and "smiles" not in merged:
            merged["smiles"] = merged["canonical_smiles"]
        return merged

    @staticmethod
    def _artifact_label(candidate_artifact: dict[str, Any]) -> str:
        for key in ("name_or_label", "trade_name", "name", "candidate_material", "canonical_smiles", "smiles"):
            value = candidate_artifact.get(key)
            if value not in (None, "", []):
                return str(value).strip()
        return ""

    def _enrich_manual_records(self, original_query: str, records: list[ChunkRecord]) -> None:
        try:
            available_skills, skill_outputs, _write_count = self._run_graph_enrichment_skills_for_records(
                original_query,
                records,
            )
            proposals = self._propose_with_skill_context(
                original_query,
                records,
                available_skills,
                skill_outputs,
            )
            validations = self._validate_with_skill_context(
                proposals,
                records,
                skill_outputs,
            )
            _, expanded = self._graph_enrichment_expander.expand(original_query, proposals, validations, records)
            expanded_validations = (
                self._validate_with_skill_context(expanded, records, skill_outputs)
                if expanded
                else []
            )
            accepted = self._graph_enrichment_store.write(
                [*proposals, *expanded],
                [*validations, *expanded_validations],
                records[0].run_id if records else "manual",
                original_query,
            )
            LOGGER.info(
                "Graph enrichment from manual evidence produced %s proposals and %s accepted claims",
                len(proposals) + len(expanded),
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

    @staticmethod
    def _merge_chunk_records(existing: list[ChunkRecord], additional: list[ChunkRecord]) -> list[ChunkRecord]:
        merged: dict[str, ChunkRecord] = {record.id: record for record in existing}
        for record in additional:
            merged.setdefault(record.id, record)
        return list(merged.values())

    @staticmethod
    def _accepted_edge_types(
        proposals: list[GraphEnrichmentProposal],
        validations: list[GraphEnrichmentValidation],
    ) -> set[str]:
        proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals if proposal.proposal_id}
        accepted: set[str] = set()
        for validation in validations:
            if not validation.accepted or validation.confidence_score < 0.6:
                continue
            proposal = proposals_by_id.get(validation.proposal_id)
            if proposal is None:
                continue
            accepted.add(validation.corrected_edge_type or proposal.edge_type)
        return accepted

    def _select_graph_follow_up_search_questions(
        self,
        questions: list[Any],
        initial_proposals: list[GraphEnrichmentProposal],
        initial_validations: list[GraphEnrichmentValidation],
        expansion_proposals: list[GraphEnrichmentProposal],
        expansion_validations: list[GraphEnrichmentValidation],
        limit: int = 4,
    ) -> list[Any]:
        present_edge_types = self._accepted_edge_types(initial_proposals, initial_validations) | self._accepted_edge_types(
            expansion_proposals,
            expansion_validations,
        )
        selected: list[Any] = []
        seen_questions: set[str] = set()
        for question in questions:
            question_text = str(getattr(question, "question", "") or "").strip()
            if not question_text:
                continue
            normalized = question_text.lower()
            if normalized in seen_questions:
                continue
            targets = {str(item) for item in getattr(question, "target_edge_types", []) if item}
            if targets and targets <= present_edge_types:
                continue
            selected.append(question)
            seen_questions.add(normalized)
            if len(selected) >= limit:
                break
        return selected

    def _run_graph_follow_up_search(self, state: DiscoveryState, questions: list[Any]) -> list[ChunkRecord]:
        if not questions:
            return []

        all_results: list[SearchResultItem] = []
        for question in questions:
            query = str(getattr(question, "question", "") or "").strip()
            if not query:
                continue
            try:
                response = self._search.search(
                    query=query,
                    num_results=2,
                    options=default_search_options(self._config, query),
                )
                all_results.extend(response.results)
            except Exception as exc:
                LOGGER.exception("Graph follow-up search failed for query %s", query)
                state.setdefault("errors", []).append(f"graph_follow_up_search:{query}:{exc}")

        if not all_results:
            return []

        unique_results = deduplicate_search_results(all_results)
        retrieval = self._retriever.retrieve_pages(
            state["original_query"],
            unique_results[:6],
            max_pages=6,
        )
        state["skipped_pages"] = list(state.get("skipped_pages", [])) + retrieval.skipped

        question_text = " | ".join(str(getattr(item, "question", "") or "").strip() for item in questions if getattr(item, "question", None))
        chunk_records: list[ChunkRecord] = []
        for page in retrieval.pages:
            if len(page.text) < self._config.min_snippet_characters:
                continue
            page_chunks = self._chunker.chunk_text(page.text)[:2]
            for index, chunk in enumerate(page_chunks):
                chunk_id = str(uuid5(NAMESPACE_URL, f"{state['run_id']}::graph-follow-up::{page.url}::{index}"))
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
                        application=None,
                        incumbent_material=None,
                        candidate_materials=[],
                        evidence_type="market or customer need",
                        application_requirements=[],
                        substitution_drivers=[],
                        relevance_score=0.72,
                        confidence_score=0.55,
                        metadata={
                            "graph_follow_up_search": True,
                            "follow_up_questions": question_text,
                            "page_metadata": page.metadata,
                        },
                    )
                )

        if not chunk_records:
            return []

        vectors = self._embedder.embed_texts([record.chunk_text for record in chunk_records])
        embedded_records = [record.model_copy(update={"vector": vector}) for record, vector in zip(chunk_records, vectors, strict=False)]
        self._store.add_chunks(embedded_records)
        return embedded_records


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
