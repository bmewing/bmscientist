from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, Field

from bmscientist.coscientist_models import MarketVolumeEstimateOutput, ResearchGoalDocument
from bmscientist.graph_enrichment import GraphEnrichmentStore
from bmscientist.graph_market import GraphMarketEvidence
from bmscientist.graph_query import DuckDBGraphQueryEngine, group_entity_matches
from bmscientist.llm import DeepSeekLLM
from bmscientist.models import GraphEntityMatch, GraphEntityMatchBuckets
from bmscientist.prompt_library import PROMPTS


class GraphEstimateResult(BaseModel):
    question: str
    matched_entities: list[GraphEntityMatch] = Field(default_factory=list)
    matched_entity_buckets: GraphEntityMatchBuckets = Field(default_factory=GraphEntityMatchBuckets)
    evidence_rows_count: int = 0
    estimate: MarketVolumeEstimateOutput
    persisted: bool = False
    persisted_rows: int = 0


class GraphEstimateAgent:
    def __init__(self, llm: DeepSeekLLM, engine: DuckDBGraphQueryEngine):
        self._llm = llm
        self._engine = engine
        self._graph_market = GraphMarketEvidence(engine._graph_path)
        self._graph_store = GraphEnrichmentStore(engine._graph_path)

    def run(self, question: str, *, limit: int = 24, persist: bool = True) -> GraphEstimateResult:
        matches = self._engine.match_entities(question)
        buckets = group_entity_matches(matches)
        document = self._estimate_document(question, buckets)
        evidence_rows = self._graph_market.build_evidence_rows_for_goal(document, limit=limit)

        system_prompt = PROMPTS.render("graph_estimate_agent", "estimate.system")
        user_prompt = PROMPTS.render(
            "graph_estimate_agent",
            "estimate.user",
            user_question=question,
            graph_entity_matches=self._engine.entity_match_summary_from_matches(matches),
            graph_evidence_rows=json.dumps(evidence_rows[:limit], indent=2, default=str),
        )
        estimate = self._llm.complete_json(
            MarketVolumeEstimateOutput,
            system_prompt,
            user_prompt,
            temperature=0.0,
        )

        persisted_rows = 0
        if persist:
            pseudo_hypothesis = SimpleNamespace(
                hypothesis_id=f"graph-estimate-{abs(hash(question))}",
                research_id="graph-estimate",
                application=estimate.application_name or (buckets.applications[0].name if buckets.applications else None),
                market_segment=estimate.market_name or (buckets.markets[0].name if buckets.markets else None),
                incumbent_material=None,
                candidate_material=None,
                next_best_competitive_alternative=None,
            )
            rows = self._graph_store.write_ai_market_volume_estimate(pseudo_hypothesis, estimate)
            persisted_rows = len(rows)

        return GraphEstimateResult(
            question=question,
            matched_entities=matches,
            matched_entity_buckets=buckets,
            evidence_rows_count=len(evidence_rows),
            estimate=estimate,
            persisted=persist,
            persisted_rows=persisted_rows,
        )

    @staticmethod
    def _estimate_document(question: str, buckets: GraphEntityMatchBuckets) -> ResearchGoalDocument:
        return ResearchGoalDocument(
            research_id="graph-estimate",
            raw_goal=question,
            target_hypotheses_final=1,
            target_hypotheses_generated=1,
            material_scope=[item.name for item in buckets.materials[:6]],
            application_scope=[item.name for item in buckets.applications[:4]],
            preferred_candidate_materials=[item.name for item in buckets.materials[:6]],
            success_definition="Produce a conservative graph-grounded estimate.",
        )


def format_graph_estimate_result(result: GraphEstimateResult) -> str:
    payload = {
        "question": result.question,
        "matched_entities": [item.model_dump(mode="json") for item in result.matched_entities],
        "matched_entity_buckets": result.matched_entity_buckets.model_dump(mode="json"),
        "evidence_rows_count": result.evidence_rows_count,
        "persisted": result.persisted,
        "persisted_rows": result.persisted_rows,
        "estimate": result.estimate.model_dump(mode="json"),
    }
    return json.dumps(payload, indent=2, default=str)
