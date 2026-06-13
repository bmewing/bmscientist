from __future__ import annotations

import argparse

from app_discovery_agent.config import AppConfig
from app_discovery_agent.coscientist_agents import (
    CoScientistRunner,
    DiscoveryEvidenceTool,
    GenerationAgent,
    LocalEvidenceRetriever,
    ReflectionAgent,
    ResearchPlanningAgent,
)
from app_discovery_agent.coscientist_cli import run_coscientist_command
from app_discovery_agent.coscientist_models import (
    Hypothesis,
    HypothesisGenerationOutput,
    PriceMetric,
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


def test_coscientist_runs_append_hypothesis_snapshots_without_overwrite(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    hypothesis = make_hypothesis()
    store.append_hypothesis_snapshot(hypothesis)
    store.append_hypothesis_snapshot(hypothesis.model_copy(update={"status": "reflected"}))

    snapshots = store.load_hypothesis_snapshots(hypothesis.research_id)

    assert len(snapshots) == 2
    assert snapshots[0].status == "generated"
    assert snapshots[1].status == "reflected"


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
