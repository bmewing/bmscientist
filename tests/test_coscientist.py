from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bmscientist.chunking import TextChunker
from bmscientist.config import AppConfig
from bmscientist.cost_tracking import CostTracker
from bmscientist.coscientist_agents import (
    CoScientistRunner,
    DiscoveryEvidenceTool,
    EvolutionAgent,
    FinalPortfolioAgent,
    GenerationAgent,
    LocalEvidenceRetriever,
    MetaReviewAgent,
    ProximityCheckAgent,
    RankingAgent,
    ReflectionAgent,
    ReflectionSearchPlanner,
    ResearchPlanningAgent,
)
from bmscientist.coscientist_cli import run_coscientist_command, run_coscientist_loop_command
from bmscientist.coscientist_models import (
    EvaluationCriterion,
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
from bmscientist.coscientist_store import CoScientistStore
from bmscientist.graph_market import GraphMarketEvidence
from bmscientist.models import EvidenceClassification, PageContent


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def search_by_vector(self, vector, top_k=8):
        return self.rows[:top_k]


class RecordingProgressReporter:
    def __init__(self):
        self.events = []

    def start(self, phase, message, total=None):
        self.events.append(("start", phase, message, total))

    def advance(self, phase, message, completed, total=None):
        self.events.append(("advance", phase, message, completed, total))

    def complete(self, phase, message, completed=None, total=None):
        self.events.append(("complete", phase, message, completed, total))


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


class GenericPlanningLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        return response_model.model_validate(
            {
                "research_mode": "candidate_design",
                "strategic_fit_criteria": ["low aquatic toxicity", "waterborne compatibility"],
                "target_incumbent_materials": ["traditional coalescing aids"],
                "preferred_candidate_materials": [],
                "candidate_material_preferences": ["small molecules with plausible commercial use"],
                "recycling_or_sustainability_angles": ["lower hazard profile"],
                "material_scope": ["waterborne coating additives"],
                "application_scope": ["latex coatings"],
                "opportunity_modes": ["candidate_screening"],
                "opportunity_speed_horizon_months": 12,
                "commercialization_constraints": ["avoid obvious regulatory hazards"],
                "ranking_weights": {"strategic_fit": 0.5, "toxicity": 0.5},
                "success_definition": "Find candidate coalescing aids worth deeper screening.",
                "candidate_artifact_schema": {
                    "artifact_type": "small_molecule",
                    "primary_identifier_field": "smiles",
                    "required_fields": ["name_or_label", "smiles", "intended_binder_system"],
                    "optional_fields": ["boiling_point_c"],
                    "validation_rules": ["SMILES should be syntactically valid where possible"],
                },
                "evaluation_criteria": [
                    {
                        "name": "aquatic_toxicity_risk",
                        "description": "Avoid candidates with high aquatic toxicity concern.",
                        "direction": "avoid",
                        "required_candidate_fields": ["smiles"],
                        "suggested_tool_ids": ["opera_qsar"],
                        "suggested_search_queries": ["candidate aquatic toxicity prediction"],
                        "reflection_guidance": ["Look for toxicity signals or missing data."],
                    }
                ],
                "reflection_guidance": ["Flag candidates that need a QSAR tool before proceeding."],
                "tool_requests": [
                    {
                        "tool_id": "opera_qsar",
                        "purpose": "Estimate aquatic toxicity and related properties from SMILES.",
                        "candidate_packages": ["OPERA command-line application"],
                        "required_inputs": ["smiles"],
                        "expected_outputs": ["toxicity_endpoints"],
                    }
                ],
                "search_strategy_notes": ["Combine molecule identity terms with toxicity/property terms."],
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


class GenericReflectionLLM:
    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        assert "Criteria to review" in user_prompt
        return response_model.model_validate(
            {
                "assessment": {
                    "criterion_results": [
                        {
                            "criterion_name": "aquatic_toxicity_risk",
                            "value": "low concern from literature proxy",
                            "normalized_score": 0.76,
                            "confidence": 0.62,
                            "rationale": "Indirect evidence suggests a lower hazard profile, but no direct tool output was supplied.",
                            "evidence_mode": "literature",
                            "tool_id": None,
                            "citation_chunk_ids": ["chunk-1"],
                            "citation_urls": ["https://example.com/1"],
                            "is_inferred": True,
                        }
                    ],
                    "tool_request_notes": ["Run opera_qsar before making a final toxicity call."],
                    "evidence_gap_notes": ["No direct aquatic toxicity model output was available."],
                },
                "needs_additional_search": True,
                "follow_up_search_queries": ["CCOC(=O)OCC aquatic toxicity coating additive"],
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


def make_generic_document() -> ResearchGoalDocument:
    return ResearchGoalDocument(
        research_id="generic-run-1",
        raw_goal="Find a waterborne coalescing aid with lower aquatic toxicity risk.",
        target_hypotheses_final=2,
        target_hypotheses_generated=4,
        research_mode="candidate_design",
        regions=["North America"],
        strategic_fit_criteria=["low aquatic toxicity", "film formation support"],
        candidate_artifact_schema={
            "artifact_type": "small_molecule",
            "primary_identifier_field": "smiles",
            "required_fields": ["name_or_label", "smiles", "intended_binder_system"],
        },
        evaluation_criteria=[
            EvaluationCriterion(
                name="aquatic_toxicity_risk",
                description="Avoid candidates with high aquatic toxicity concern.",
                direction="avoid",
                required_candidate_fields=["smiles"],
                suggested_tool_ids=["opera_qsar"],
                suggested_search_queries=["candidate aquatic toxicity prediction"],
            )
        ],
        tool_requests=[
            {
                "tool_id": "opera_qsar",
                "purpose": "Estimate aquatic toxicity and related properties from SMILES.",
                "candidate_packages": ["OPERA command-line application"],
                "required_inputs": ["smiles"],
                "expected_outputs": ["toxicity_endpoints"],
            }
        ],
        search_strategy_notes=["Use SMILES and toxicity language in discovery queries."],
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
        research_id="calm-river-beacon",
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


def test_coscientist_store_claim_project_name_normalizes_and_avoids_collisions(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")

    first = store.claim_project_name("My Great Project")
    second = store.claim_project_name("My Great Project")

    assert first == "my-great-project"
    assert second == "my-great-project-2"
    assert (tmp_path / "coscientist" / first).exists()
    assert (tmp_path / "coscientist" / second).exists()


def test_agent_specific_model_selection_from_config():
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
        chat_model="deepseek-v4-flash",
        generation_chat_model="deepseek-v4-pro",
    )

    from bmscientist.llm import DeepSeekLLM

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


def test_generation_reports_batch_progress_for_large_targets():
    class MultiBatchGenerationLLM:
        def __init__(self):
            self.calls = 0

        def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
            self.calls += 1
            if self.calls == 1:
                payload = {
                    "hypotheses": [
                        {"title": f"Batch 1 idea {index}", "summary": "Idea", "candidate_material": "PETG"}
                        for index in range(1, 6)
                    ]
                }
            else:
                payload = {
                    "hypotheses": [
                        {"title": "Batch 2 idea 1", "summary": "Idea", "candidate_material": "PETG"}
                    ]
                }
            return response_model.model_validate(payload)

    retriever = LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder())
    agent = GenerationAgent(MultiBatchGenerationLLM(), retriever)
    progress_updates = []

    hypotheses = agent.generate(
        make_document(),
        on_progress=lambda completed, total: progress_updates.append((completed, total)),
    )

    assert len(hypotheses) == 6
    assert progress_updates == [(5, 6)]


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
    from bmscientist.coscientist_models import MetaReviewRound

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


def test_hypothesis_generation_output_accepts_top_level_list_payload():
    output = HypothesisGenerationOutput.model_validate(
        [
            {
                "title": "Clear deli containers with PETG",
                "summary": "PETG may fit clear deli container applications.",
                "candidate_material": "PETG",
                "incumbent_material": "PVC",
            }
        ]
    )

    assert len(output.hypotheses) == 1
    assert output.hypotheses[0].title == "Clear deli containers with PETG"


def test_hypothesis_generation_output_normalizes_named_confidence_levels():
    output = HypothesisGenerationOutput.model_validate(
        {
            "hypotheses": [
                {
                    "title": "Clear deli containers with PETG",
                    "summary": "PETG may fit clear deli container applications.",
                    "candidate_material": "PETG",
                    "incumbent_material": "PVC",
                    "generation_confidence": "Medium",
                },
                {
                    "title": "PETG tray lids",
                    "summary": "Another idea.",
                    "candidate_material": "PETG",
                    "incumbent_material": "PVC",
                    "generation_confidence": "80%",
                },
            ]
        }
    )

    assert output.hypotheses[0].generation_confidence == 0.55
    assert output.hypotheses[1].generation_confidence == 0.8


def test_research_planning_accepts_generic_candidate_design_contract():
    agent = ResearchPlanningAgent(GenericPlanningLLM())

    document = agent.create_research_goal(
        research_id="screen-1",
        raw_goal="Find a waterborne coalescing aid with lower aquatic toxicity risk.",
        target_hypotheses_final=3,
        regions=["North America"],
        strategic_fit_notes=None,
        preferred_evidence_recency_days=180,
        reflection_search_limits=ReflectionSearchLimits(),
    )

    assert document.research_mode == "candidate_design"
    assert document.candidate_artifact_schema.primary_identifier_field == "smiles"
    assert document.evaluation_criteria[0].name == "aquatic_toxicity_risk"
    assert document.tool_requests[0].tool_id == "opera_qsar"


def test_hypothesis_seed_accepts_candidate_artifact():
    output = HypothesisGenerationOutput.model_validate(
        {
            "hypotheses": [
                {
                    "title": "Candidate coalescent A",
                    "summary": "A small-molecule coalescent candidate.",
                    "candidate_artifact": {
                        "name_or_label": "Candidate A",
                        "smiles": "CCOC(=O)OCC",
                        "intended_binder_system": "acrylic latex",
                    },
                    "evaluation_results": [
                        {
                            "criterion_name": "aquatic_toxicity_risk",
                            "normalized_score": 0.6,
                            "confidence": 0.4,
                            "rationale": "Preliminary estimate.",
                            "is_inferred": True,
                        }
                    ],
                    "generation_confidence": 0.5,
                }
            ]
        }
    )

    assert output.hypotheses[0].candidate_artifact["smiles"] == "CCOC(=O)OCC"
    assert output.hypotheses[0].evaluation_results[0].criterion_name == "aquatic_toxicity_risk"


def test_hypothesis_seed_normalizes_evaluation_result_aliases():
    output = HypothesisGenerationOutput.model_validate(
        {
            "hypotheses": [
                {
                    "title": "Candidate coalescent B",
                    "summary": "A candidate with aliased evaluation fields.",
                    "evaluation_results": [
                        {
                            "criterion": "aquatic_toxicity_risk",
                            "score": 0.7,
                            "reasoning": "Predicted to be less toxic than the benchmark.",
                            "tool": "opera_qsar",
                            "chunk_ids": ["chunk-1"],
                            "urls": ["https://example.com/opera"],
                            "inferred": True,
                        }
                    ],
                }
            ]
        }
    )

    result = output.hypotheses[0].evaluation_results[0]
    assert result.criterion_name == "aquatic_toxicity_risk"
    assert result.normalized_score == 0.7
    assert result.rationale == "Predicted to be less toxic than the benchmark."
    assert result.tool_id == "opera_qsar"
    assert result.citation_chunk_ids == ["chunk-1"]
    assert result.citation_urls == ["https://example.com/opera"]
    assert result.is_inferred is True


def test_hypothesis_seed_accepts_single_entry_evaluation_result_maps():
    output = HypothesisGenerationOutput.model_validate(
        {
            "hypotheses": [
                {
                    "title": "Candidate coalescent C",
                    "summary": "A candidate with one-entry evaluation summaries.",
                    "evaluation_results": [
                        {
                            "Aquatic toxicity": "Predicted low acute aquatic toxicity; no structural alerts.",
                        },
                        {
                            "Coalescing efficiency": 0.62,
                        },
                    ],
                }
            ]
        }
    )

    first_result = output.hypotheses[0].evaluation_results[0]
    second_result = output.hypotheses[0].evaluation_results[1]
    assert first_result.criterion_name == "Aquatic toxicity"
    assert first_result.rationale == "Predicted low acute aquatic toxicity; no structural alerts."
    assert first_result.value is None
    assert second_result.criterion_name == "Coalescing efficiency"
    assert second_result.value == 0.62


def test_reflection_assessment_normalizes_criterion_result_aliases():
    assessment = ReflectionAssessment.model_validate(
        {
            "criterion_results": [
                {
                    "criterion": "water_compatibility",
                    "score": 0.55,
                    "description": "Estimated to have acceptable water compatibility.",
                }
            ]
        }
    )

    result = assessment.criterion_results[0]
    assert result.criterion_name == "water_compatibility"
    assert result.normalized_score == 0.55
    assert result.rationale == "Estimated to have acceptable water compatibility."


def test_hypothesis_seed_normalizes_non_unit_score_scales():
    output = HypothesisGenerationOutput.model_validate(
        {
            "hypotheses": [
                {
                    "title": "Candidate coalescent D",
                    "summary": "A candidate with mixed score scales.",
                    "evaluation_results": [
                        {
                            "criterion_name": "film_forming_potential",
                            "normalized_score": 4,
                        },
                        {
                            "criterion_name": "water_compatibility",
                            "normalized_score": "8/10",
                        },
                        {
                            "criterion_name": "aquatic_toxicity_risk",
                            "normalized_score": "80%",
                        },
                    ],
                }
            ]
        }
    )

    results = output.hypotheses[0].evaluation_results
    assert results[0].normalized_score == 0.8
    assert results[1].normalized_score == 0.8
    assert results[2].normalized_score == 0.8


def test_generation_preserves_candidate_artifact_primary_identifier():
    class GenericGenerationLLM:
        def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
            return response_model.model_validate(
                {
                    "hypotheses": [
                        {
                            "title": "Candidate coalescent A",
                            "summary": "Candidate with a plausible low-toxicity profile.",
                            "candidate_artifact": {
                                "name_or_label": "Candidate A",
                                "smiles": "CCOC(=O)OCC",
                                "intended_binder_system": "acrylic latex",
                            },
                            "generation_confidence": 0.61,
                        }
                    ]
                }
            )

    retriever = LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder())
    agent = GenerationAgent(GenericGenerationLLM(), retriever)

    hypotheses = agent.generate(make_generic_document())

    assert len(hypotheses) == 1
    assert hypotheses[0].candidate_artifact["smiles"] == "CCOC(=O)OCC"
    assert hypotheses[0].candidate_artifact["name_or_label"] == "Candidate A"


def test_reflection_reviews_generic_criteria_without_running_tools():
    discovery = FakeDiscoveryAgent()
    agent = ReflectionAgent(
        GenericReflectionLLM(),
        LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder()),
        DiscoveryEvidenceTool(discovery),
    )
    hypothesis = make_hypothesis().model_copy(
        update={
            "title": "Candidate coalescent A",
            "candidate_artifact": {
                "name_or_label": "Candidate A",
                "smiles": "CCOC(=O)OCC",
                "intended_binder_system": "acrylic latex",
            },
        }
    )

    reflected, run_count = agent.reflect(make_generic_document(), hypothesis)

    assert run_count == 3
    assert discovery.queries[0] == "candidate aquatic toxicity prediction"
    assert reflected.reflection_assessment is not None
    assert reflected.reflection_assessment.criterion_results[0].criterion_name == "aquatic_toxicity_risk"
    assert reflected.reflection_assessment.tool_request_notes == ["Run opera_qsar before making a final toxicity call."]


def test_reflection_search_planner_uses_evaluation_criteria_queries():
    planner = ReflectionSearchPlanner()
    hypothesis = make_hypothesis().model_copy(
        update={
            "title": "Candidate coalescent A",
            "candidate_artifact": {"smiles": "CCOC(=O)OCC"},
        }
    )

    queries = planner.plan(
        make_generic_document(),
        hypothesis,
        ReflectionAssessment(),
        suggested_queries=[],
    )

    assert "candidate aquatic toxicity prediction" in queries
    assert "CCOC(=O)OCC aquatic_toxicity_risk" in queries


def test_ranking_uses_criterion_results_when_present():
    reflected = make_hypothesis().model_copy(
        update={
            "status": "reflected",
            "reflection_assessment": ReflectionAssessment.model_validate(
                {
                    "criterion_results": [
                        {
                            "criterion_name": "aquatic_toxicity_risk",
                            "normalized_score": 0.81,
                            "confidence": 0.7,
                            "rationale": "Looks promising.",
                            "is_inferred": True,
                        }
                    ]
                }
            ),
        }
    )

    score = RankingAgent._heuristic_score(reflected)

    assert score > 0.7


def test_report_includes_tool_requests():
    hypothesis = make_hypothesis().model_copy(
        update={
            "reflection_assessment": ReflectionAssessment.model_validate(
                {
                    "criterion_results": [
                        {
                            "criterion_name": "aquatic_toxicity_risk",
                            "normalized_score": 0.76,
                            "confidence": 0.62,
                            "rationale": "Indirect evidence only.",
                            "is_inferred": True,
                        }
                    ],
                    "tool_request_notes": ["Run opera_qsar before final selection."],
                }
            )
        }
    )

    report = CoScientistRunner._build_tool_request_report(make_generic_document(), [hypothesis])

    assert "opera_qsar" in report
    assert "Dependent criteria: aquatic_toxicity_risk" in report
    assert "Run opera_qsar before final selection." in report


def test_evaluation_criterion_normalizes_evidence_mode_aliases():
    document = ResearchGoalDocument.model_validate(
        {
            "research_id": "screen-2",
            "raw_goal": "Screen coalescing aids.",
            "target_hypotheses_final": 2,
            "target_hypotheses_generated": 4,
            "evaluation_criteria": [
                {
                    "name": "aquatic_toxicity_risk",
                    "evidence_mode": "computational_prediction",
                },
                {
                    "name": "film_formation_proxy",
                    "evidence_mode": "qsar",
                },
                {
                    "name": "synthesis_feasibility",
                    "evidence_mode": "computational_retrosynthesis",
                },
                {
                    "name": "bench_validation",
                    "evidence_mode": "computational_and_experimental",
                },
            ],
            "tool_requests": [
                {
                    "tool_id": "custom_qsar",
                    "purpose": "Prototype a toxicity-screening helper.",
                    "status": "to_be_developed",
                }
            ],
        }
    )

    assert document.evaluation_criteria[0].evidence_mode == "external_tool"
    assert document.evaluation_criteria[1].evidence_mode == "external_tool"
    assert document.evaluation_criteria[2].evidence_mode == "external_tool"
    assert document.evaluation_criteria[3].evidence_mode == "mixed"
    assert document.tool_requests[0].status == "requested"


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


def test_graph_market_evidence_matches_market_and_application(tmp_path):
    graph_path = tmp_path / "graph"
    (graph_path / "nodes").mkdir(parents=True)
    (graph_path / "edges").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "market_id": "market:medical-packaging",
                    "name": "Medical Packaging",
                    "normalized_name": "Medical Packaging",
                    "primary_slug": "medical-packaging-market",
                    "canonical_url": "https://example.com/market",
                    "source_vendor": "Grand View Research",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        graph_path / "nodes" / "Market.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "application_id": "application:medical-trays",
                    "name": "Medical Trays",
                    "normalized_name": "Medical Trays",
                    "node_type": "application",
                    "url": "https://example.com/application",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        graph_path / "nodes" / "Application.parquet",
    )
    pq.write_table(pa.Table.from_pylist([], schema=pa.schema([("product_id", pa.string())])), graph_path / "nodes" / "Product.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "geo_id": "geo:north-america",
                    "name": "North America",
                    "normalized_name": "North America",
                    "geo_type": "region",
                    "parent_geo_id": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        graph_path / "nodes" / "Geography.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "edge_id": "edge-1",
                    "market_id": "market:medical-packaging",
                    "application_id": "application:medical-trays",
                    "scope_type": "statistics",
                    "source_node_type": "application",
                    "geo_id": "geo:north-america",
                    "page_url": "https://example.com/stats",
                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                    "status": "fetched",
                    "revenue_value": 1200.0,
                    "revenue_year": 2025,
                    "forecast_revenue_value": 1700.0,
                    "forecast_revenue_year": 2030,
                    "cagr_value": 5.4,
                    "cagr_start_year": 2026,
                    "cagr_end_year": 2030,
                    "unit": "USD million",
                    "currency": "USD",
                    "unit_scale": "million",
                    "highlights_json": '[{"text":"The North America medical trays segment is growing."}]',
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet",
    )
    empty_edge_schema = pa.schema([("edge_id", pa.string()), ("market_id", pa.string())])
    pq.write_table(pa.Table.from_pylist([], schema=empty_edge_schema), graph_path / "edges" / "Market_USES_Product.parquet")
    pq.write_table(pa.Table.from_pylist([], schema=empty_edge_schema), graph_path / "edges" / "Market_IN_GEOGRAPHY_Geography.parquet")

    rows = GraphMarketEvidence(graph_path).build_evidence_rows(make_document(), make_hypothesis())

    assert rows
    assert rows[0]["metadata"]["source_type"] == "offline-graph-market-data"
    assert rows[0]["metadata"]["revenue_value"] == 1200.0
    assert "Medical Packaging" in rows[0]["chunk_text"]


def test_reflection_infers_missing_assessment_fields_after_review():
    agent = ReflectionAgent(
        ReflectionNoSearchLLM(),
        LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder()),
        DiscoveryEvidenceTool(FakeDiscoveryAgent()),
    )

    reflected, _ = agent.reflect(make_document(), make_hypothesis())

    assessment = reflected.reflection_assessment
    assert assessment is not None
    assert assessment.market_size_score.value is not None
    assert assessment.replacement_fit_score.value is not None
    assert assessment.activation_ease_score.value is not None
    assert assessment.replacement_driver_strength_score.value is not None
    assert assessment.technical_success_probability.value is not None
    assert assessment.commercial_success_probability.value is not None
    assert assessment.market_size_score.is_inferred is True
    assert assessment.incumbent_price_usd_per_kg.value is not None
    assert assessment.incumbent_price_usd_per_kg.is_inferred is True


def test_reflection_estimates_and_persists_missing_graph_volume(tmp_path):
    import bmscientist.graph_enrichment as ge
    from bmscientist.graph_enrichment import (
        APPLICATION_NODE_SCHEMA,
        MARKET_APPLICATION_SCHEMA,
        MARKET_NODE_SCHEMA,
        empty_row,
        write_rows,
    )

    class VolumeEstimateLLM:
        def __init__(self):
            self.calls = 0

        def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
            self.calls += 1
            assert "Estimate the current annual substrate/material volume" in user_prompt
            assert "revenue_value" in user_prompt
            return response_model.model_validate(
                {
                    "market_name": "medical packaging",
                    "application_name": "medical trays",
                    "total_substrate_volume_value": 80000,
                    "total_substrate_volume_unit": "metric_tons_per_year",
                    "volume_year": 2026,
                    "revenue_value": 1200,
                    "revenue_unit": "USD million",
                    "revenue_year": 2026,
                    "confidence": 0.52,
                    "rationale": "Estimated from tray market revenue and substrate price assumptions.",
                    "material_volumes": [
                        {
                            "material_name": "PETG",
                            "volume_value": 44000,
                            "volume_unit": "metric_tons_per_year",
                            "share_of_total": 0.55,
                            "confidence": 0.5,
                            "rationale": "PETG is the leading tray substrate.",
                        },
                        {
                            "material_name": "PVC",
                            "volume_value": 4000,
                            "volume_unit": "metric_tons_per_year",
                            "share_of_total": 0.05,
                            "confidence": 0.45,
                            "rationale": "PVC is a small legacy share.",
                        },
                    ],
                    "source_citations": [
                        {
                            "chunk_id": "graph:Market_HAS_APPLICATION_Application:edge-1",
                            "source_url": "https://example.com/stats",
                            "source_title": "Offline graph market data",
                        }
                    ],
                }
            )

    graph_path = tmp_path / "graph"
    write_rows(
        graph_path / "nodes" / "Market.parquet",
        [
            {
                "market_id": "market:medical-packaging",
                "name": "medical packaging",
                "normalized_name": "medical packaging",
                "primary_slug": "medical-packaging-market",
                "source_vendor": "test",
            }
        ],
        MARKET_NODE_SCHEMA,
    )
    write_rows(
        graph_path / "nodes" / "Application.parquet",
        [
            {
                "application_id": "application:medical-trays",
                "name": "medical trays",
                "normalized_name": "medical trays",
                "node_type": "application",
            }
        ],
        APPLICATION_NODE_SCHEMA,
    )
    edge = empty_row(MARKET_APPLICATION_SCHEMA)
    edge.update(
        {
            "edge_id": "edge-1",
            "market_id": "market:medical-packaging",
            "application_id": "application:medical-trays",
            "scope_type": "statistics",
            "source_node_type": "application",
            "page_url": "https://example.com/stats",
            "target_url": "https://example.com/stats",
            "status": "fetched",
            "revenue_value": 1200.0,
            "revenue_year": 2026,
            "unit": "USD million",
            "source_url": "https://example.com/stats",
            "source_title": "Offline graph market data",
            "supporting_quote": "Medical tray market revenue was estimated at $1.2B.",
            "confidence": 0.7,
            "validation_status": "accepted",
        }
    )
    write_rows(graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet", [edge], MARKET_APPLICATION_SCHEMA)

    original_graph_path = ge.GRAPH_PATH
    ge.GRAPH_PATH = graph_path
    volume_llm = VolumeEstimateLLM()
    try:
        agent = ReflectionAgent(
            ReflectionNoSearchLLM(),
            LocalEvidenceRetriever(FakeStore([make_row("chunk-1")]), FakeEmbedder()),
            DiscoveryEvidenceTool(FakeDiscoveryAgent()),
            graph_evidence=GraphMarketEvidence(graph_path),
            volume_estimation_llm=volume_llm,
        )

        reflected, _ = agent.reflect(make_document(), make_hypothesis())
    finally:
        ge.GRAPH_PATH = original_graph_path

    assert reflected.reflection_assessment is not None
    assert volume_llm.calls == 1

    product_edges = pq.read_table(graph_path / "edges" / "Product_USED_IN_Application.parquet").to_pylist()
    volumes_by_product = {edge["product_id"]: edge["volume_value"] for edge in product_edges}
    assert volumes_by_product["product:pvc"] == 4000
    assert volumes_by_product["product:petg"] == 44000

    market_edges = pq.read_table(graph_path / "edges" / "Market_HAS_APPLICATION_Application.parquet").to_pylist()
    ai_edges = [edge for edge in market_edges if edge["source_node_type"] == "ai_volume_estimate"]
    assert ai_edges[0]["volume_value"] == 80000


def test_reflection_discovery_retains_low_relevance_pages_for_later_scoring():
    class LowRelevanceClassifier:
        def heuristic_relevance(self, query, text):
            return 0.01

        def classify(self, query, page):
            return EvidenceClassification.model_validate(
                {
                    "relevant": False,
                    "relevance_score": 0.05,
                    "confidence_score": 0.2,
                    "application": None,
                    "incumbent_material": None,
                    "candidate_materials": [],
                    "evidence_type": "market or customer need",
                    "application_requirements": [],
                    "substitution_drivers": [],
                    "rationale": "Indirect market context.",
                    "supporting_quotes": [],
                    "metadata": {},
                }
            )

    tool = DiscoveryEvidenceTool.__new__(DiscoveryEvidenceTool)
    tool._classifier = LowRelevanceClassifier()
    tool._config = type("Config", (), {"min_page_characters": 20, "min_snippet_characters": 20})()
    tool._chunker = TextChunker(chunk_size=80, chunk_overlap=0)
    page = PageContent(
        title="Adjacent market report",
        url="https://example.com/market",
        search_query="application market size",
        source_domain="example.com",
        fetched_at=datetime.now(timezone.utc),
        text="Market revenue, CAGR, application segmentation, and customer demand context for an adjacent market.",
    )
    skipped_pages = []

    candidates = tool._filter_pages("PETG application evidence", [page], skipped_pages)
    records = tool._classify_and_chunk("run-1", "PETG application evidence", candidates, skipped_pages, [])

    assert len(candidates) == 1
    assert skipped_pages == []
    assert records
    assert records[0].metadata["retained_for_reflection"] is True
    assert records[0].metadata["classification_relevant"] is False


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


def test_meta_review_allows_one_retry_when_same_gap_reappears():
    document = make_document().model_copy(
        update={
            "whitespace_gap_notes": ["Need stronger coverage in non-medical rigid clear applications."],
            "whitespace_gap_persistence_count": 0,
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

    assert updated_document.whitespace_gap_persistence_count == 1
    assert meta_round.should_continue is True
    assert meta_round.stop_reason is None


def test_meta_review_prompt_includes_previous_gaps_and_persistence_state():
    class RecordingMetaReviewLLM:
        def __init__(self):
            self.user_prompt = None

        def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
            self.user_prompt = user_prompt
            return response_model.model_validate(
                {
                    "whitespace_gaps": [],
                    "generation_guidance": [],
                    "coverage_assessment": "Coverage is sufficient.",
                    "gap_shrinkage_status": "improved",
                    "coverage_sufficient": True,
                }
            )

    llm = RecordingMetaReviewLLM()
    document = make_document().model_copy(
        update={
            "whitespace_gap_notes": ["Need stronger coverage in non-medical rigid clear applications."],
            "meta_review_generation_guidance": ["Expand into adjacent clear rigid consumer applications."],
            "whitespace_gap_persistence_count": 1,
        }
    )
    ranking_round = RankingAgent(RankingLLM()).rank(make_document(), [make_reflected_hypothesis()], 1, 1, 1)[0]

    MetaReviewAgent(llm).review(
        document=document,
        hypotheses=[make_reflected_hypothesis()],
        ranking_round=ranking_round,
        round_index=2,
        gap_overlap_threshold=0.6,
        max_gap_persistence_rounds=1,
    )

    assert "Previous whitespace gaps" in llm.user_prompt
    assert "Current unresolved-gap persistence count" in llm.user_prompt
    assert "Expand into adjacent clear rigid consumer applications." in llm.user_prompt


def test_final_portfolio_agent_writes_conclusive_report_with_validation_gaps():
    class FinalReportLLM:
        def __init__(self):
            self.user_prompt = None

        def complete_text(self, system_prompt, user_prompt, temperature=0.1):
            self.user_prompt = user_prompt
            return "# Final Opportunity Portfolio\n\nConclusive report."

    llm = FinalReportLLM()
    agent = FinalPortfolioAgent(llm)
    hypothesis = make_reflected_hypothesis().model_copy(
        update={
            "ranking_round_id": "ranking-1",
            "ranking_score": 0.82,
            "ranking_status": "advance",
            "reflection_assessment": ReflectionAssessment.model_validate(
                {
                    "strategic_fit_score": {"value": 0.8, "confidence": 0.7},
                    "market_size_score": {"value": None, "confidence": 0.0},
                    "technical_success_probability": {"value": 0.7, "confidence": 0.6},
                    "commercial_success_probability": {"value": None, "confidence": 0.0},
                    "evidence_gap_notes": ["Missing market size validation."],
                }
            ),
        }
    )
    ranking_round = RankingAgent(RankingLLM()).rank(make_document(), [hypothesis], 1, 1, 1)[0]
    meta_round = MetaReviewAgent(MetaReviewLLM()).review(make_document(), [hypothesis], ranking_round, 1, 0.6, 1)[1]

    report = agent.build_report(
        document=make_document(),
        hypotheses=[hypothesis],
        ranking_round=ranking_round,
        meta_review_round=meta_round,
        stop_reason="target_portfolio_reached",
        target_count=1,
    )

    assert report.startswith("# Final Opportunity Portfolio")
    assert "Missing market size validation." in llm.user_prompt
    assert "Market size validation missing or weak" in llm.user_prompt


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
            self.prepared = []

        def prepare_project_name(self, preferred_name=None):
            self.prepared.append(preferred_name)
            return preferred_name or "calm-river-beacon"

        def run(self, **kwargs):
            from bmscientist.coscientist_models import CoScientistRunResult
            assert kwargs["project_name"] == "calm-river-beacon"
            assert kwargs["spawn_reflection_daemons"] is False

            return CoScientistRunResult(
                research_id="research-123",
                generated_hypotheses=8,
                reflected_hypotheses=8,
                automatic_discovery_runs=2,
                research_goal_path=str(tmp_path / "goal.json"),
                hypothesis_path=str(tmp_path / "hypotheses.jsonl"),
                report_path=str(tmp_path / "report.md"),
            )

        def run_loop(self, **kwargs):
            from bmscientist.coscientist_models import CoScientistLoopResult

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
        goal="Find PVC replacement opportunities.",
        project_name=None,
        target_hypotheses=4,
        regions="North America,Europe",
        strategic_fit_notes="Prefer regulated applications.",
        preferred_evidence_recency_days=180,
        max_reflection_searches_per_hypothesis=3,
        results_per_query=5,
        max_pages_per_search=8,
        reflection_concurrency=2,
        skip_loop=False,
        target_final_hypotheses=None,
        max_rounds=2,
        evolve_top_k=5,
        evolved_per_round=5,
        regenerated_per_round=5,
        proximity_check_every=1,
        max_synthesized_per_round=3,
        promotion_score_threshold=0.72,
        gap_overlap_threshold=0.6,
        max_gap_persistence_rounds=1,
        spawn_reflection_daemons=False,
    )
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )

    exit_code = run_coscientist_command(args, config, runner_cls=FakeRunner)

    assert exit_code == 0


def test_cli_can_opt_into_legacy_reflection_daemons(tmp_path):
    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def prepare_project_name(self, preferred_name=None):
            return preferred_name or "calm-river-beacon"

        def run(self, **kwargs):
            from bmscientist.coscientist_models import CoScientistRunResult

            assert kwargs["spawn_reflection_daemons"] is True
            return CoScientistRunResult(
                research_id="research-123",
                generated_hypotheses=4,
                reflected_hypotheses=4,
                automatic_discovery_runs=1,
                research_goal_path=str(tmp_path / "goal.json"),
                hypothesis_path=str(tmp_path / "hypotheses.jsonl"),
                report_path=str(tmp_path / "report.md"),
            )

        def run_loop(self, **kwargs):
            raise AssertionError("skip_loop should avoid loop execution")

    args = argparse.Namespace(
        goal="Find PVC replacement opportunities.",
        project_name=None,
        target_hypotheses=4,
        regions="",
        strategic_fit_notes=None,
        preferred_evidence_recency_days=180,
        max_reflection_searches_per_hypothesis=3,
        results_per_query=5,
        max_pages_per_search=8,
        reflection_concurrency=2,
        spawn_reflection_daemons=True,
        skip_loop=True,
        target_final_hypotheses=None,
        max_rounds=None,
        evolve_top_k=5,
        evolved_per_round=5,
        regenerated_per_round=5,
        proximity_check_every=1,
        max_synthesized_per_round=3,
        promotion_score_threshold=0.72,
        gap_overlap_threshold=0.6,
        max_gap_persistence_rounds=1,
    )
    config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )

    exit_code = run_coscientist_command(args, config, runner_cls=FakeRunner)

    assert exit_code == 0


def test_coscientist_reflect_cli_smoke_writes_expected_summary(tmp_path):
    from bmscientist.coscientist_cli import run_coscientist_reflect_command

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def reflect_existing(self, **kwargs):
            from bmscientist.coscientist_models import CoScientistRunResult

            assert kwargs["research_id"] == "research-123"
            assert kwargs["daemon"] is False
            assert kwargs["lease_seconds"] == 1800
            assert kwargs["poll_interval_seconds"] == 5
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
        daemon=False,
        worker_id=None,
        lease_seconds=1800,
        poll_interval_seconds=5,
        idle_exit_after_seconds=None,
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
            from bmscientist.coscientist_models import CoScientistLoopResult

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


def test_coscientist_store_load_hypothesis_snapshots_tolerates_transient_missing_files(tmp_path, monkeypatch):
    store = CoScientistStore(tmp_path / "coscientist")
    hypothesis = make_hypothesis()
    store.append_hypothesis_snapshot(hypothesis)
    path = store.hypothesis_file_path(hypothesis)
    original_read_text = Path.read_text
    state = {"raised": False}

    def flaky_read_text(self, *args, **kwargs):
        if self == path and not state["raised"]:
            state["raised"] = True
            raise FileNotFoundError(path)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    snapshots = store.load_hypothesis_snapshots(hypothesis.research_id)

    assert len(snapshots) == 1
    assert snapshots[0].hypothesis_id == hypothesis.hypothesis_id


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


def test_coscientist_store_claims_and_releases_generated_hypothesis(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    hypothesis = make_hypothesis()
    store.append_hypothesis_snapshot(hypothesis)

    claimed = store.claim_next_generated_hypothesis("research-1", worker_id="worker-a", lease_seconds=120)

    assert claimed is not None
    assert claimed.status == "reflecting"
    assert claimed.reflection_worker_id == "worker-a"
    assert claimed.reflection_attempt_count == 1
    assert (
        tmp_path / "coscientist" / "research-1" / "hypotheses" / "reflecting" / f"{hypothesis.hypothesis_id}.json"
    ).exists()

    store.release_reflection_claim(claimed, "temporary failure")
    latest = {item.hypothesis_id: item for item in store.latest_hypotheses("research-1")}

    assert latest["hyp-1"].status == "generated"
    assert latest["hyp-1"].reflection_worker_id is None
    assert latest["hyp-1"].reflection_error == "temporary failure"
    assert (tmp_path / "coscientist" / "research-1" / "hypotheses" / "generated" / "hyp-1.json").exists()


def test_coscientist_store_retries_transient_claim_file_locks(tmp_path, monkeypatch):
    store = CoScientistStore(tmp_path / "coscientist")
    hypothesis = make_hypothesis()
    store.append_hypothesis_snapshot(hypothesis)
    generated_path = store.hypothesis_file_path(hypothesis)
    original_rename = Path.rename
    state = {"attempts": 0}

    def flaky_rename(self, target):
        if self == generated_path and state["attempts"] == 0:
            state["attempts"] += 1
            raise PermissionError("simulated Windows file lock")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    claimed = store.claim_next_generated_hypothesis("research-1", worker_id="worker-a", lease_seconds=120)

    assert claimed is not None
    assert claimed.hypothesis_id == hypothesis.hypothesis_id
    assert claimed.status == "reflecting"
    assert state["attempts"] == 1


def test_coscientist_store_requeues_expired_reflection_claims(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    expired = make_hypothesis().model_copy(
        update={
            "status": "reflecting",
            "reflection_worker_id": "worker-a",
            "reflection_claimed_at": datetime(2000, 1, 1, tzinfo=timezone.utc),
            "reflection_lease_expires_at": datetime(2000, 1, 1, 0, 5, tzinfo=timezone.utc),
            "reflection_attempt_count": 1,
        }
    )
    store.append_hypothesis_snapshot(expired)

    reclaimed = store.requeue_expired_reflection_claims("research-1")
    latest = {item.hypothesis_id: item for item in store.latest_hypotheses("research-1")}

    assert reclaimed == 1
    assert latest["hyp-1"].status == "generated"
    assert latest["hyp-1"].reflection_worker_id is None
    assert latest["hyp-1"].reflection_error == "Reflection lease expired before completion."


def test_reflect_hypothesis_exposes_single_hypothesis_api(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    document = make_document()
    hypothesis = make_hypothesis()

    class FakeReflectRunner:
        def __init__(self):
            self.calls = []

        def reflect(self, document_arg, hypothesis_arg):
            self.calls.append((document_arg.research_id, hypothesis_arg.hypothesis_id))
            return (
                hypothesis_arg.model_copy(
                    update={
                        "status": "reflected",
                        "reflection_assessment": ReflectionAssessment.model_validate(
                            {
                                "strategic_fit_score": {
                                    "value": 0.6,
                                    "rationale": "Reflected through public API.",
                                    "confidence": 0.5,
                                }
                            }
                        ),
                    }
                ),
                2,
            )

    runner = object.__new__(CoScientistRunner)
    runner._artifact_store = store
    runner._reflection_agent = FakeReflectRunner()

    reflected, discovery_runs = runner.reflect_hypothesis(document, hypothesis, persist=True)
    latest = {item.hypothesis_id: item for item in store.latest_hypotheses("research-1")}

    assert reflected.status == "reflected"
    assert discovery_runs == 2
    assert runner._reflection_agent.calls == [("research-1", "hyp-1")]
    assert latest["hyp-1"].status == "reflected"


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
    runner._progress_reporter = RecordingProgressReporter()

    result = runner.reflect_existing("research-1")
    latest = {item.hypothesis_id: item for item in store.latest_hypotheses("research-1")}

    assert runner._reflection_agent.calls == ["hyp-1"]
    assert result.generated_hypotheses == 2
    assert result.reflected_hypotheses == 2
    assert latest["hyp-1"].status == "reflected"
    assert latest["hyp-2"].status == "reflected"


def test_reflect_existing_requeues_failed_claims(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    document = make_document()
    store.save_research_goal(document)
    store.append_hypothesis_snapshot(make_hypothesis())

    class FailingReflectRunner:
        def reflect(self, document, hypothesis):
            raise RuntimeError("simulated reflection failure")

    runner = object.__new__(CoScientistRunner)
    runner._artifact_store = store
    runner._reflection_agent = FailingReflectRunner()
    runner._progress_reporter = RecordingProgressReporter()

    result = runner.reflect_existing("research-1", max_hypotheses=1, concurrency=1)
    latest = {item.hypothesis_id: item for item in store.latest_hypotheses("research-1")}

    assert result.reflected_hypotheses == 0
    assert latest["hyp-1"].status == "generated"
    assert latest["hyp-1"].reflection_attempt_count == 1
    assert latest["hyp-1"].reflection_error == "simulated reflection failure"


def test_active_reflection_worker_lines_show_worker_and_hypothesis_title():
    lines = CoScientistRunner._active_reflection_worker_lines(
        [
            make_hypothesis().model_copy(
                update={
                    "status": "reflecting",
                    "title": "PETG for rigid medical trays",
                    "reflection_worker_id": "reflector-a",
                }
            ),
            make_hypothesis().model_copy(
                update={
                    "hypothesis_id": "hyp-2",
                    "status": "reflecting",
                    "title": "APET for food service lids",
                    "reflection_worker_id": "reflector-b",
                }
            ),
            make_hypothesis().model_copy(
                update={
                    "hypothesis_id": "hyp-3",
                    "status": "generated",
                    "title": "Should not appear",
                }
            ),
        ]
    )

    assert lines == [
        "reflector-a: PETG for rigid medical trays",
        "reflector-b: APET for food service lids",
    ]


def test_reflect_and_append_reports_incremental_progress(tmp_path):
    store = CoScientistStore(tmp_path / "coscientist")
    reporter = RecordingProgressReporter()

    class FakeReflectRunner:
        def reflect(self, document, hypothesis):
            return (
                hypothesis.model_copy(
                    update={
                        "status": "reflected",
                        "reflection_assessment": ReflectionAssessment.model_validate(
                            {
                                "strategic_fit_score": {
                                    "value": 0.6,
                                    "rationale": "Reflected during progress test.",
                                    "confidence": 0.5,
                                    "citation_chunk_ids": ["chunk-1"],
                                    "citation_urls": ["https://example.com/1"],
                                    "is_inferred": False,
                                }
                            }
                        ),
                    }
                ),
                1,
            )

    runner = object.__new__(CoScientistRunner)
    runner._artifact_store = store
    runner._reflection_agent = FakeReflectRunner()
    runner._progress_reporter = reporter

    hypotheses = [make_hypothesis(), make_hypothesis().model_copy(update={"hypothesis_id": "hyp-2"})]
    reflected, discovery_runs = runner._reflect_and_append(make_document(), hypotheses, concurrency=1)

    assert len(reflected) == 2
    assert discovery_runs == 2
    assert reporter.events == [
        ("start", "reflection", "Reflecting on ideas", 2),
        ("advance", "reflection", "Reflecting on ideas", 1, 2),
        ("advance", "reflection", "Reflecting on ideas", 2, 2),
        ("complete", "reflection", "Reflecting on ideas complete", 2, 2),
    ]


def test_run_reports_stage_progress(tmp_path):
    document = make_document().model_copy(update={"target_hypotheses_generated": 2})
    reporter = RecordingProgressReporter()
    generated = [
        make_hypothesis(),
        make_hypothesis().model_copy(update={"hypothesis_id": "hyp-2", "title": "Hypothesis 2"}),
    ]
    reflected = [
        hypothesis.model_copy(
            update={
                "status": "reflected",
                "reflection_assessment": ReflectionAssessment.model_validate(
                    {
                        "strategic_fit_score": {
                            "value": 0.7,
                            "rationale": "Strong fit.",
                            "confidence": 0.7,
                            "citation_chunk_ids": ["chunk-1"],
                            "citation_urls": ["https://example.com/1"],
                            "is_inferred": False,
                        }
                    }
                ),
            }
        )
        for hypothesis in generated
    ]

    class FakePlanningAgent:
        def create_research_goal(self, **kwargs):
            return document

    class FakeGenerationAgent:
        batch_size = 5

        def generate(self, doc, on_progress=None, on_batch=None):
            assert doc.research_id == document.research_id
            if on_batch is not None:
                on_batch(generated)
            return generated

    class FakeArtifactStore:
        def __init__(self):
            self.saved = []

        def claim_project_name(self, preferred_name=None):
            return preferred_name or document.research_id

        def save_research_goal(self, doc):
            return tmp_path / "goal.json"

        def append_hypothesis_snapshot(self, hypothesis):
            self.saved.append(hypothesis.hypothesis_id)

        def latest_hypotheses(self, research_id):
            return []

        def write_report(self, research_id, report):
            return tmp_path / "report.md"

        def write_cost_report(self, research_id, payload):
            import json

            path = tmp_path / "cost.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path

        def hypothesis_path(self, research_id):
            return tmp_path / "hypotheses"

    runner = object.__new__(CoScientistRunner)
    runner._config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
    )
    runner._cost_tracker = CostTracker(runner._config)
    runner._cost_tracker_seeded_research_ids = set()
    runner._planning_agent = FakePlanningAgent()
    runner._generation_agent = FakeGenerationAgent()
    runner._artifact_store = FakeArtifactStore()
    runner._progress_reporter = reporter
    runner._reflect_and_append = lambda document, hypotheses, concurrency, phase="reflection", progress_message="Reflecting on ideas": (  # noqa: E731
        reflected,
        3,
    )
    runner._build_report = lambda document, hypotheses: "report"

    result = runner.run("Find PET opportunities", 1, reflection_concurrency=2)

    assert result.generated_hypotheses == 2
    assert result.reflected_hypotheses == 2
    assert result.cost_path == str((tmp_path / "cost.json").resolve())
    assert reporter.events == [
        ("start", "planning", "Processing goal", None),
        ("complete", "planning", "Goal processed", None, None),
        ("start", "generation", "Generating ideas", 2),
        ("complete", "generation", "Ideas generated", 2, 2),
        ("start", "reporting", "Writing report", None),
        ("complete", "reporting", "Report written", None, None),
    ]


def test_run_can_spawn_reflection_daemons_and_wait_for_queue(tmp_path):
    document = make_document().model_copy(update={"target_hypotheses_generated": 2})
    reporter = RecordingProgressReporter()
    generated = [
        make_hypothesis(),
        make_hypothesis().model_copy(update={"hypothesis_id": "hyp-2", "title": "Hypothesis 2"}),
    ]
    reflected = [
        hypothesis.model_copy(
            update={
                "status": "reflected",
                "reflection_assessment": ReflectionAssessment.model_validate(
                    {
                        "reflection_discovery_run_ids": ["run-1"],
                    }
                ),
            }
        )
        for hypothesis in generated
    ]

    class FakePlanningAgent:
        def create_research_goal(self, **kwargs):
            return document

    class FakeGenerationAgent:
        batch_size = 5

        def generate(self, doc, on_progress=None, on_batch=None):
            if on_batch is not None:
                on_batch([generated[0]])
                on_batch([generated[1]])
            return generated

    class FakeArtifactStore:
        def __init__(self):
            self.saved = []

        def claim_project_name(self, preferred_name=None):
            return preferred_name or document.research_id

        def save_research_goal(self, doc):
            return tmp_path / "goal.json"

        def append_hypothesis_snapshot(self, hypothesis):
            self.saved.append(hypothesis.hypothesis_id)

        def latest_hypotheses(self, research_id):
            return generated

        def write_report(self, research_id, report):
            return tmp_path / "report.md"

        def hypothesis_path(self, research_id):
            return tmp_path / "hypotheses"

    spawn_calls = []

    runner = object.__new__(CoScientistRunner)
    runner._config = AppConfig(
        deepseek_api_key="x",
        exa_api_key="y",
        request_timeout_seconds=240,
    )
    runner._planning_agent = FakePlanningAgent()
    runner._generation_agent = FakeGenerationAgent()
    runner._artifact_store = FakeArtifactStore()
    runner._progress_reporter = reporter
    runner._spawn_reflection_daemons = lambda **kwargs: spawn_calls.append(kwargs)  # noqa: E731
    runner._start_reflection_progress_monitor = lambda **kwargs: (lambda: None)  # noqa: E731
    runner._wait_for_reflection_completion = lambda **kwargs: reflected  # noqa: E731
    runner._automatic_discovery_runs_for_hypotheses = lambda hypotheses: 2  # noqa: E731
    runner._build_report = lambda document, hypotheses: "report"

    result = runner.run(
        "Find PET opportunities",
        1,
        reflection_concurrency=2,
        spawn_reflection_daemons=True,
    )

    assert result.generated_hypotheses == 2
    assert result.reflected_hypotheses == 2
    assert spawn_calls
    assert spawn_calls[0]["idle_exit_after_seconds"] == 735
