from __future__ import annotations

import argparse

from app_discovery_agent.config import AppConfig
from app_discovery_agent.coscientist_agents import (
    CoScientistRunner,
    DiscoveryEvidenceTool,
    EvolutionAgent,
    GenerationAgent,
    LocalEvidenceRetriever,
    MetaReviewAgent,
    ProximityCheckAgent,
    RankingAgent,
    ReflectionAgent,
    ResearchPlanningAgent,
)
from app_discovery_agent.coscientist_cli import run_coscientist_command, run_coscientist_loop_command
from app_discovery_agent.coscientist_models import (
    Hypothesis,
    HypothesisEvolutionOutput,
    HypothesisGenerationOutput,
    MetaReviewOutput,
    PriceMetric,
    ProximityReviewOutput,
    RankedHypothesis,
    RankingOutput,
    ReflectionAssessment,
    ReflectionReviewOutput,
    ReflectionSearchLimits,
    ResearchGoalDocument,
)
from app_discovery_agent.coscientist_store import CoScientistStore


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def search_by_vector(self, vector, top_k=8):
        return self.rows[:top_k]


class PlanningLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "strategic_fit_criteria": ["favors regulated durable applications"],
                "target_incumbent_materials": ["PVC"],
                "preferred_candidate_materials": ["PETG"],
                "candidate_material_preferences": ["favor drop-in fit"],
                "recycling_or_sustainability_angles": ["recycled content value"],
                "material_scope": ["PETG", "PVC"],
                "application_scope": ["clear rigid packaging"],
                "opportunity_modes": ["rapid_win", "large_volume"],
                "opportunity_speed_horizon_months": 6,
                "commercialization_constraints": ["sales realization under 6 months"],
                "ranking_weights": {"speed": 0.5, "volume": 0.3, "sustainability": 0.2},
                "success_definition": "Find plausible near-term substitution targets.",
            }
        )


class GenerationLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "hypotheses": [
                    {
                        "title": "PETG for rigid medical trays",
                        "summary": "PETG may fit rigid medical tray applications where clarity and processability matter.",
                        "application": "medical trays",
                        "market_segment": "medical packaging",
                        "candidate_material": "PETG",
                        "incumbent_material": "PVC",
                        "next_best_competitive_alternative": "APET",
                        "application_requirements": ["clarity", "thermoformability"],
                        "substitution_drivers": ["PVC reduction"],
                        "strategic_rationale": "Existing evidence ties PVC use to clear rigid trays and notes replacement pressure.",
                        "supporting_chunk_ids": ["chunk-1"],
                        "supporting_urls": ["https://example.com/1"],
                        "assumptions": ["medical-grade PETG supply is available"],
                        "unknowns": ["regional certification timing"],
                        "generation_confidence": 0.7,
                    }
                ]
            }
        )


class MetaGuidedGenerationLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        assert "Meta-review guidance" in user_prompt
        assert "Whitespace gaps" in user_prompt
        assert "Ranking feedback from the judge" not in user_prompt
        assert "Best patterns:" not in user_prompt
        return response_model.model_validate(
            {
                "hypotheses": [
                    {
                        "title": "PETG for non-medical rigid trays",
                        "summary": "Targets whitespace outside medical packaging.",
                        "application": "consumer trays",
                        "market_segment": "consumer packaging",
                        "candidate_material": "PETG",
                        "incumbent_material": "PVC",
                        "application_requirements": ["clarity"],
                        "substitution_drivers": ["PVC reduction"],
                        "generation_confidence": 0.55,
                    }
                ]
            }
        )


class ReflectionLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return response_model.model_validate(
                {
                    "assessment": {
                        "strategic_fit_score": {
                            "value": 0.7,
                            "rationale": "The application aligns with the goal.",
                            "confidence": 0.6,
                            "citation_chunk_ids": ["chunk-1"],
                            "citation_urls": ["https://example.com/1"],
                            "is_inferred": False,
                        },
                        "technical_success_probability": {
                            "value": 0.65,
                            "rationale": "Moderate fit based on requirements evidence.",
                            "confidence": 0.55,
                            "citation_chunk_ids": ["chunk-1"],
                            "citation_urls": ["https://example.com/1"],
                            "is_inferred": True,
                        },
                        "evidence_gap_notes": ["Need market size evidence."],
                    },
                    "needs_additional_search": True,
                    "follow_up_search_queries": ["medical trays PVC market size North America"],
                }
            )
        return response_model.model_validate(
            {
                "assessment": {
                    "strategic_fit_score": {
                        "value": 0.7,
                        "rationale": "The application aligns with the goal.",
                        "confidence": 0.6,
                        "citation_chunk_ids": ["chunk-1"],
                        "citation_urls": ["https://example.com/1"],
                        "is_inferred": False,
                    },
                    "market_size_score": {
                        "value": 0.5,
                        "rationale": "Some market demand evidence was found.",
                        "confidence": 0.4,
                        "citation_chunk_ids": ["chunk-2"],
                        "citation_urls": ["https://example.com/2"],
                        "is_inferred": True,
                    },
                    "technical_success_probability": {
                        "value": 0.65,
                        "rationale": "Moderate fit based on requirements evidence.",
                        "confidence": 0.55,
                        "citation_chunk_ids": ["chunk-1"],
                        "citation_urls": ["https://example.com/1"],
                        "is_inferred": True,
                    },
                    "evidence_gap_notes": ["Incumbent price remains unresolved."],
                },
                "needs_additional_search": False,
                "follow_up_search_queries": [],
            }
        )


class ReflectionNoSearchLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "assessment": {
                    "strategic_fit_score": {
                        "value": 0.8,
                        "rationale": "Strong local evidence.",
                        "confidence": 0.8,
                        "citation_chunk_ids": ["chunk-1"],
                        "citation_urls": ["https://example.com/1"],
                        "is_inferred": False,
                    },
                    "evidence_gap_notes": [],
                },
                "needs_additional_search": False,
                "follow_up_search_queries": [],
            }
        )


class RankingLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "rankings": [
                    {
                        "hypothesis_id": "hyp-1",
                        "rank": 1,
                        "score": 0.82,
                        "recommended_action": "evolve",
                        "rationale": "Strong strategic and technical profile.",
                        "strengths": ["good fit"],
                        "weaknesses": ["price gap"],
                        "improvement_directions": ["narrow buyer segment"],
                    }
                ],
                "best_patterns": ["fast activation"],
                "worst_patterns": ["weak evidence"],
            }
        )


class EvolutionLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "hypotheses": [
                    {
                        "title": "PETG for hospital tray lids",
                        "summary": "A narrower evolved variant focused on tray lids.",
                        "application": "hospital tray lids",
                        "market_segment": "medical packaging",
                        "candidate_material": "PETG",
                        "incumbent_material": "PVC",
                        "next_best_competitive_alternative": "APET",
                        "application_requirements": ["clarity"],
                        "substitution_drivers": ["PVC reduction"],
                        "strategic_rationale": "Narrower product form may improve activation.",
                        "supporting_chunk_ids": ["chunk-1"],
                        "supporting_urls": ["https://example.com/1"],
                        "assumptions": ["same thermoforming assets"],
                        "unknowns": ["buyer qualification timing"],
                        "generation_confidence": 0.58,
                        "parent_hypothesis_ids": ["hyp-1"],
                        "mutation_strategy": "Narrow application form.",
                        "evolution_notes": ["Focus on lids rather than all trays."],
                    }
                ]
            }
        )


class ProximityLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "concepts": [
                    {
                        "concept_label": "PETG medical tray conversion cluster",
                        "description": "Medical tray PETG opportunities with overlapping activation logic.",
                        "member_hypothesis_ids": ["hyp-1", "hyp-2"],
                    }
                ],
                "synthesized_hypotheses": [
                    {
                        "title": "PETG platform for rigid medical tray formats",
                        "summary": "A higher-level opportunity combining overlapping medical tray PETG variants.",
                        "application": "medical trays",
                        "market_segment": "medical packaging",
                        "candidate_material": "PETG",
                        "incumbent_material": "PVC",
                        "next_best_competitive_alternative": "APET",
                        "application_requirements": ["clarity", "thermoformability"],
                        "substitution_drivers": ["PVC reduction"],
                        "strategic_rationale": "Combines the strongest overlapping signals.",
                        "supporting_chunk_ids": ["chunk-1"],
                        "supporting_urls": ["https://example.com/1"],
                        "assumptions": ["shared thermoforming base"],
                        "unknowns": ["qualification speed by buyer"],
                        "generation_confidence": 0.61,
                        "merged_from_hypothesis_ids": ["hyp-1", "hyp-2"],
                        "concept_label": "PETG medical tray conversion cluster",
                        "synthesis_rationale": "Merge overlapping tray ideas into a broader platform thesis.",
                    }
                ],
                "notes": ["Clustered overlapping tray ideas."],
            }
        )


class MetaReviewLLM:
    def __init__(self, gaps=None, coverage_sufficient=False):
        self.gaps = gaps if gaps is not None else ["Need stronger coverage in non-medical rigid clear applications."]
        self.coverage_sufficient = coverage_sufficient

    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "whitespace_gaps": self.gaps,
                "generation_guidance": [
                    "Look for clear rigid applications outside medical trays with cited qualification speed signals."
                ],
                "coverage_assessment": "Coverage is still concentrated in medical packaging.",
                "gap_shrinkage_status": "stable",
                "coverage_sufficient": self.coverage_sufficient,
            }
        )


class FakeDiscoveryAgent:
    def __init__(self):
        self.queries = []

    def discover(self, query, max_search_queries=1, results_per_query=5, max_pages=8):
        self.queries.append(query)
        return type("Summary", (), {"run_id": "run-123"})()


class FakePriceCache:
    def ensure_fresh(self):
        return {"cache": "ok"}

    def build_price_evidence_rows(self, incumbent_material, nbca_material, candidate_material, document=None):
        rows = []
        if incumbent_material:
            rows.append(
                {
                    "id": "price-cache:incumbent",
                    "source_url": "C:/tmp/prices.json",
                    "source_title": "Structured price cache",
                    "application": None,
                    "incumbent_material": incumbent_material,
                    "candidate_materials": [candidate_material] if candidate_material else [],
                    "relevance_score": 0.95,
                    "retrieved_at": "2099-01-01T00:00:00+00:00",
                    "chunk_text": f"Structured price for {incumbent_material}: 1.23 USD/kg",
                    "metadata": {"price_role": "incumbent"},
                }
            )
        return rows

    def metric_for_material(self, material_name, document=None):
        if material_name == "PVC":
            return PriceMetric.model_validate(
                {
                    "value": 1.23,
                    "rationale": "Structured PlasticPortal reference.",
                    "confidence": 0.8,
                    "citation_chunk_ids": ["price-cache:incumbent"],
                    "citation_urls": ["C:/tmp/prices.json"],
                    "is_inferred": False,
                }
            )
        if material_name == "APET":
            return PriceMetric.model_validate(
                {
                    "value": 1.45,
                    "rationale": "Structured PlasticPortal reference.",
                    "confidence": 0.8,
                    "citation_chunk_ids": ["price-cache:nbca"],
                    "citation_urls": ["C:/tmp/prices.json"],
                    "is_inferred": False,
                }
            )
        return None


def make_row(chunk_id: str, text: str = "PVC trays are used in medical packaging.", retrieved_at: str = "2099-01-01T00:00:00+00:00"):
    return {
        "id": chunk_id,
        "source_url": f"https://example.com/{chunk_id}",
        "source_title": f"Source {chunk_id}",
        "application": "medical trays",
        "incumbent_material": "PVC",
        "candidate_materials": ["PETG"],
        "application_requirements": ["clarity", "thermoformability"],
        "substitution_drivers": ["PVC reduction"],
        "relevance_score": 0.8,
        "retrieved_at": retrieved_at,
        "chunk_text": text,
    }


def make_document() -> ResearchGoalDocument:
    return ResearchGoalDocument(
        research_id="research-1",
        raw_goal="Find PVC replacement opportunities in clear rigid packaging.",
        target_hypotheses_final=3,
        target_hypotheses_generated=6,
        regions=["North America"],
        strategic_fit_criteria=["fit regulated packaging"],
        reflection_search_limits=ReflectionSearchLimits(),
        material_scope=["PETG", "PVC"],
        application_scope=["medical trays"],
        success_definition="Identify plausible substitution targets.",
    )


def make_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="hyp-1",
        research_id="research-1",
        status="generated",
        title="PETG for rigid medical trays",
        summary="PETG may fit rigid medical tray applications.",
        application="medical trays",
        market_segment="medical packaging",
        region_scope=["North America"],
        candidate_material="PETG",
        incumbent_material="PVC",
        next_best_competitive_alternative="APET",
        application_requirements=["clarity"],
        substitution_drivers=["PVC reduction"],
        strategic_rationale="Relevant local evidence exists.",
        supporting_chunk_ids=["chunk-1"],
        supporting_urls=["https://example.com/1"],
        assumptions=[],
        unknowns=[],
        generation_confidence=0.7,
    )


def make_reflected_hypothesis(hypothesis_id: str = "hyp-1", title: str = "PETG for rigid medical trays") -> Hypothesis:
    return make_hypothesis().model_copy(
        update={
            "hypothesis_id": hypothesis_id,
            "title": title,
            "status": "reflected",
            "reflection_assessment": ReflectionAssessment.model_validate(
                {
                    "strategic_fit_score": {"value": 0.8, "confidence": 0.7},
                    "market_size_score": {"value": 0.7, "confidence": 0.5},
                    "replacement_fit_score": {"value": 0.75, "confidence": 0.6},
                    "activation_ease_score": {"value": 0.8, "confidence": 0.6},
                    "technical_success_probability": {"value": 0.7, "confidence": 0.6},
                    "commercial_success_probability": {"value": 0.65, "confidence": 0.5},
                }
            ),
        }
    )


def test_research_planning_sets_generated_target_default():
    agent = ResearchPlanningAgent(PlanningLLM())
    document = agent.create_research_goal(
        raw_goal="Find PVC replacement opportunities.",
        target_hypotheses_final=5,
        regions=["North America"],
        strategic_fit_notes=None,
        preferred_evidence_recency_days=180,
        reflection_search_limits=ReflectionSearchLimits(),
    )

    assert document.target_hypotheses_generated == 10
    assert document.opportunity_speed_horizon_months == 6
    assert document.target_incumbent_materials == ["PVC"]
    assert document.ranking_weights["speed"] == 0.5


def test_agent_specific_model_selection_from_config():
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
        chat_model="deepseek-v4-flash",
        generation_chat_model="deepseek-v4-pro",
    )

    from app_discovery_agent.llm import DeepSeekLLM

    generation_llm = DeepSeekLLM(config, model=config.generation_chat_model)
    reflection_llm = DeepSeekLLM(config, model=config.reflection_chat_model)

    assert generation_llm._model == "deepseek-v4-pro"
    assert reflection_llm._model == "deepseek-v4-flash"


def test_generation_uses_local_evidence_with_citations():
    retriever = LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder())
    agent = GenerationAgent(GenerationLLM(), retriever)

    hypotheses = agent.generate(make_document())

    assert len(hypotheses) == 1
    assert hypotheses[0].supporting_chunk_ids == ["chunk-1"]
    assert hypotheses[0].candidate_material == "PETG"


def test_generation_from_meta_review_uses_only_meta_review_guidance():
    retriever = LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder())
    agent = GenerationAgent(MetaGuidedGenerationLLM(), retriever)
    meta_review_round = MetaReviewOutput.model_validate(
        {
            "whitespace_gaps": ["Need stronger coverage outside medical trays."],
            "generation_guidance": ["Target consumer rigid clear trays with faster qualification paths."],
            "coverage_assessment": "Medical concentration remains too high.",
            "coverage_sufficient": False,
        }
    )
    from app_discovery_agent.coscientist_models import MetaReviewRound

    generated = agent.generate_from_meta_review(
        make_document(),
        MetaReviewRound(
            meta_review_round_id="meta-1",
            research_id="research-1",
            round_index=1,
            whitespace_gaps=meta_review_round.whitespace_gaps,
            generation_guidance=meta_review_round.generation_guidance,
            coverage_assessment=meta_review_round.coverage_assessment,
            coverage_sufficient=meta_review_round.coverage_sufficient,
        ),
        target_count=1,
        round_index=1,
    )

    assert len(generated) == 1
    assert generated[0].application == "consumer trays"


def test_hypothesis_generation_output_accepts_loose_llm_aliases():
    output = HypothesisGenerationOutput.model_validate(
        {
            "hypotheses": [
                {
                    "hypothesis_text": (
                        "Replace styrenic display trays with PET where clarity and rapid qualification matter."
                    ),
                    "replacement_material": "PET",
                    "target_material": "Styrenics",
                    "requirements": ["clarity", "stiffness"],
                    "drivers": ["rPET value"],
                    "confidence": 0.52,
                    "chunk_ids": ["chunk-1"],
                    "source_urls": ["https://example.com/1"],
                }
            ]
        }
    )

    seed = output.hypotheses[0]
    assert seed.title.startswith("Replace styrenic display trays")
    assert seed.summary.startswith("Replace styrenic display trays")
    assert seed.candidate_material == "PET"
    assert seed.incumbent_material == "Styrenics"
    assert seed.application_requirements == ["clarity", "stiffness"]
    assert seed.substitution_drivers == ["rPET value"]
    assert seed.supporting_chunk_ids == ["chunk-1"]
    assert seed.generation_confidence == 0.52


def test_reflection_checks_local_evidence_first_and_skips_discovery_when_sufficient():
    discovery = FakeDiscoveryAgent()
    agent = ReflectionAgent(
        ReflectionNoSearchLLM(),
        LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder()),
        DiscoveryEvidenceTool(discovery),
    )

    reflected, run_count = agent.reflect(make_document(), make_hypothesis())

    assert reflected.status == "reflected"
    assert run_count == 0
    assert discovery.queries == []


def test_reflection_calls_discovery_when_evidence_missing_and_records_run_ids():
    discovery = FakeDiscoveryAgent()
    agent = ReflectionAgent(
        ReflectionLLM(),
        LocalEvidenceRetriever(FakeStore([make_row("chunk-1"), make_row("chunk-2")]), FakeEmbedder()),
        DiscoveryEvidenceTool(discovery),
    )

    reflected, run_count = agent.reflect(make_document(), make_hypothesis())

    assert run_count == 3
    assert discovery.queries[:2] == [
        "PETG medical trays replacement for PVC",
        "PETG vs PVC medical trays requirements",
    ]
    assert "medical trays" in discovery.queries[2]
    assert reflected.reflection_assessment is not None
    assert reflected.reflection_assessment.reflection_discovery_run_ids == ["run-123", "run-123", "run-123"]


def test_reflection_applies_structured_price_cache_when_prices_are_missing():
    agent = ReflectionAgent(
        ReflectionNoSearchLLM(),
        LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder()),
        DiscoveryEvidenceTool(FakeDiscoveryAgent()),
        price_cache=FakePriceCache(),
    )

    reflected, run_count = agent.reflect(make_document(), make_hypothesis())

    assert run_count == 0
    assert reflected.reflection_assessment is not None
    assert reflected.reflection_assessment.incumbent_price_usd_per_kg.value == 1.23
    assert reflected.reflection_assessment.nbca_price_usd_per_kg.value == 1.45


def test_reflection_leaves_unknown_fields_null_when_unresolved():
    assessment = ReflectionAssessment.model_validate(
        {
            "incumbent_price_usd_per_kg": {
                "value": None,
                "rationale": "No reliable pricing evidence.",
                "confidence": 0.1,
                "citation_chunk_ids": [],
                "citation_urls": [],
                "is_inferred": False,
            },
            "evidence_gap_notes": ["Missing incumbent price."],
        }
    )

    assert assessment.incumbent_price_usd_per_kg.value is None
    assert assessment.evidence_gap_notes == ["Missing incumbent price."]


def test_ranking_agent_scores_and_marks_ranked_snapshots():
    reflected = make_reflected_hypothesis()
    agent = RankingAgent(RankingLLM())

    ranking_round, ranked_hypotheses = agent.rank(make_document(), [reflected], 1, 1, 1)

    assert ranking_round.candidate_count == 1
    assert ranking_round.promoted_hypothesis_ids == ["hyp-1"]
    assert ranking_round.best_patterns == ["fast activation"]
    assert ranked_hypotheses[0].ranking_score == 0.82
    assert ranked_hypotheses[0].ranking_status == "evolve"


def test_evolution_agent_creates_parent_linked_generated_variant():
    parent = make_reflected_hypothesis().model_copy(
        update={
            "ranking_score": 0.82,
            "ranking_rationale": "Strong opportunity.",
        }
    )
    ranking_round = RankingAgent(RankingLLM()).rank(make_document(), [parent], 1, 1, 1)[0]
    agent = EvolutionAgent(EvolutionLLM())

    evolved = agent.evolve(make_document(), [parent], ranking_round, 1, 1)

    assert len(evolved) == 1
    assert evolved[0].status == "generated"
    assert evolved[0].generation_source == "evolved"
    assert evolved[0].parent_hypothesis_ids == ["hyp-1"]
    assert "Narrow application form." in evolved[0].evolution_notes


def test_proximity_agent_labels_and_synthesizes_overlapping_hypotheses():
    hyp_1 = make_reflected_hypothesis("hyp-1", "PETG for rigid medical trays")
    hyp_2 = make_reflected_hypothesis("hyp-2", "PETG for hospital tray lids")
    agent = ProximityCheckAgent(ProximityLLM())

    proximity_round, updated, synthesized = agent.review(make_document(), [hyp_1, hyp_2], 1, 2)

    assert proximity_round.labeled_hypothesis_ids == ["hyp-1", "hyp-2"]
    assert proximity_round.synthesized_hypothesis_ids
    assert synthesized[0].generation_source == "synthesized"
    assert synthesized[0].merged_from_hypothesis_ids == ["hyp-1", "hyp-2"]
    updates_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in updated}
    assert "PETG medical tray conversion cluster" in updates_by_id["hyp-1"].concept_labels
    assert updates_by_id["hyp-2"].retired_reason == "merged_into_synthesized_hypothesis"
    assert updates_by_id["hyp-2"].status == "retired"


def test_meta_review_tracks_gap_persistence_and_requests_one_last_loop():
    document = make_document().model_copy(
        update={
            "whitespace_gap_notes": ["Need stronger coverage in non-medical rigid clear applications."],
            "whitespace_gap_persistence_count": 1,
        }
    )
    ranking_round = RankingAgent(RankingLLM()).rank(make_document(), [make_reflected_hypothesis()], 1, 1, 1)[0]
    agent = MetaReviewAgent(MetaReviewLLM())

    updated_document, meta_round = agent.review(
        document=document,
        hypotheses=[make_reflected_hypothesis()],
        ranking_round=ranking_round,
        round_index=2,
        gap_overlap_threshold=0.6,
        max_gap_persistence_rounds=1,
    )

    assert updated_document.whitespace_gap_persistence_count == 2
    assert meta_round.should_continue is False
    assert meta_round.stop_reason == "meta_review_gap_persistence"


def test_reflection_review_output_accepts_null_metric_objects():
    review = ReflectionReviewOutput.model_validate(
        {
            "assessment": {
                "strategic_fit_score": None,
                "replacement_driver_strength_score": None,
                "incumbent_price_usd_per_kg": None,
                "nbca_price_usd_per_kg": None,
            },
            "needs_additional_search": True,
            "follow_up_search_queries": None,
        }
    )

    assert review.assessment.strategic_fit_score.value is None
    assert review.assessment.replacement_driver_strength_score.value is None
    assert review.assessment.incumbent_price_usd_per_kg.value is None
    assert review.assessment.nbca_price_usd_per_kg.value is None
    assert review.follow_up_search_queries == []


def test_cli_smoke_writes_expected_summary(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            from app_discovery_agent.coscientist_models import CoScientistRunResult

            return CoScientistRunResult(
                research_id="research-123",
                generated_hypotheses=8,
                reflected_hypotheses=8,
                automatic_discovery_runs=2,
                research_goal_path=str(tmp_path / "goal.json"),
                hypothesis_path=str(tmp_path / "hypotheses.jsonl"),
                report_path=str(tmp_path / "report.md"),
            )

    args = argparse.Namespace(
        goal="Find PVC replacement opportunities.",
        target_hypotheses=4,
        regions="North America,Europe",
        strategic_fit_notes="Prefer regulated applications.",
        preferred_evidence_recency_days=180,
        max_reflection_searches_per_hypothesis=3,
        results_per_query=5,
        max_pages_per_search=8,
        reflection_concurrency=2,
    )
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )

    exit_code = run_coscientist_command(args, config, runner_cls=FakeRunner)

    assert exit_code == 0


def test_coscientist_reflect_cli_smoke_writes_expected_summary(tmp_path):
    from app_discovery_agent.coscientist_cli import run_coscientist_reflect_command

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def reflect_existing(self, **kwargs):
            from app_discovery_agent.coscientist_models import CoScientistRunResult

            assert kwargs["research_id"] == "research-123"
            return CoScientistRunResult(
                research_id="research-123",
                generated_hypotheses=8,
                reflected_hypotheses=5,
                automatic_discovery_runs=2,
                research_goal_path=str(tmp_path / "goal.json"),
                hypothesis_path=str(tmp_path / "hypotheses.jsonl"),
                report_path=str(tmp_path / "report.md"),
            )

    args = argparse.Namespace(
        research_id="research-123",
        preferred_evidence_recency_days=None,
        max_reflection_searches_per_hypothesis=None,
        results_per_query=None,
        max_pages_per_search=None,
        max_hypotheses=None,
        concurrency=2,
    )
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )

    exit_code = run_coscientist_reflect_command(args, config, runner_cls=FakeRunner)

    assert exit_code == 0


def test_coscientist_loop_cli_smoke_writes_expected_summary(tmp_path):
    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run_loop(self, **kwargs):
            from app_discovery_agent.coscientist_models import CoScientistLoopResult

            assert kwargs["research_id"] == "research-123"
            return CoScientistLoopResult(
                research_id="research-123",
                rounds_completed=1,
                ranked_hypotheses=8,
                evolved_hypotheses=2,
                regenerated_hypotheses=2,
                synthesized_hypotheses=1,
                reflected_hypotheses=4,
                automatic_discovery_runs=1,
                ranking_path=str(tmp_path / "rankings.jsonl"),
                hypothesis_path=str(tmp_path / "hypotheses.jsonl"),
                report_path=str(tmp_path / "loop.md"),
                stop_reason="max_rounds_reached",
            )

    args = argparse.Namespace(
        research_id="research-123",
        target_final_hypotheses=5,
        max_rounds=1,
        evolve_top_k=3,
        evolved_per_round=2,
        regenerated_per_round=2,
        proximity_check_every=1,
        max_synthesized_per_round=2,
        promotion_score_threshold=0.72,
        gap_overlap_threshold=0.6,
        max_gap_persistence_rounds=1,
        preferred_evidence_recency_days=None,
        max_reflection_searches_per_hypothesis=None,
        results_per_query=None,
        max_pages_per_search=None,
        reflection_concurrency=2,
    )
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )

    exit_code = run_coscientist_loop_command(args, config, runner_cls=FakeRunner)

    assert exit_code == 0


def test_coscientist_store_moves_hypothesis_file_between_stage_folders(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    hypothesis = make_hypothesis()
    store.append_hypothesis_snapshot(hypothesis)
    store.append_hypothesis_snapshot(hypothesis.model_copy(update={"status": "reflected"}))

    snapshots = store.load_hypothesis_snapshots(hypothesis.research_id)
    generated_path = (
        tmp_path / "coscientist" / hypothesis.research_id / "hypotheses" / "generated" / f"{hypothesis.hypothesis_id}.json"
    )
    reflected_path = (
        tmp_path / "coscientist" / hypothesis.research_id / "hypotheses" / "reflected" / f"{hypothesis.hypothesis_id}.json"
    )

    assert len(snapshots) == 1
    assert not generated_path.exists()
    assert reflected_path.exists()
    assert snapshots[0].status == "reflected"


def test_coscientist_store_uses_evolve_and_retired_queue_folders(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    hypothesis = make_reflected_hypothesis()

    evolve_path = store.append_hypothesis_snapshot(hypothesis.model_copy(update={"status": "evolve"}))
    retired_path = store.append_hypothesis_snapshot(
        hypothesis.model_copy(update={"status": "retired", "is_active": False, "retired_reason": "merged"})
    )

    assert evolve_path.parent.name == "evolve"
    assert not evolve_path.exists()
    assert retired_path.parent.name == "retired"
    assert retired_path.exists()


def test_reflect_existing_only_processes_pending_hypotheses(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    document = make_document()
    store.save_research_goal(document)

    pending = make_hypothesis()
    already_reflected = make_hypothesis().model_copy(
        update={
            "hypothesis_id": "hyp-2",
            "title": "Already reflected hypothesis",
            "status": "reflected",
            "reflection_assessment": ReflectionAssessment.model_validate(
                {
                    "strategic_fit_score": {
                        "value": 0.8,
                        "rationale": "Existing reflected record.",
                        "confidence": 0.7,
                        "citation_chunk_ids": ["chunk-1"],
                        "citation_urls": ["https://example.com/1"],
                        "is_inferred": False,
                    }
                }
            ),
        }
    )
    store.append_hypothesis_snapshot(pending)
    store.append_hypothesis_snapshot(already_reflected)

    class FakeReflectRunner:
        def __init__(self):
            self.calls = []

        def reflect(self, document, hypothesis):
            self.calls.append(hypothesis.hypothesis_id)
            return (
                hypothesis.model_copy(
                    update={
                        "status": "reflected",
                        "reflection_assessment": ReflectionAssessment.model_validate(
                            {
                                "strategic_fit_score": {
                                    "value": 0.6,
                                    "rationale": "Reflected during resume.",
                                    "confidence": 0.5,
                                    "citation_chunk_ids": ["chunk-1"],
                                    "citation_urls": ["https://example.com/1"],
                                    "is_inferred": False,
                                }
                            }
                        ),
                    }
                ),
                0,
            )

    runner = object.__new__(CoScientistRunner)
    runner._artifact_store = store
    runner._reflection_agent = FakeReflectRunner()

    result = runner.reflect_existing("research-1")
    latest = {item.hypothesis_id: item for item in store.latest_hypotheses("research-1")}

    assert runner._reflection_agent.calls == ["hyp-1"]
    assert result.generated_hypotheses == 2
    assert result.reflected_hypotheses == 2
    assert latest["hyp-1"].status == "reflected"
    assert latest["hyp-2"].status == "reflected"
