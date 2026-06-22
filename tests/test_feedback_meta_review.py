import json
from datetime import datetime, timezone
import pytest
from pydantic import BaseModel

from bmscientist.coscientist_models import (
    CandidateArtifactSchema,
    EvaluationCriterion,
    Hypothesis,
    ResearchGoalDocument,
    ReflectionAssessment,
    UpdatedResearchPlan,
    RankingRound,
)
from bmscientist.coscientist_store import CoScientistStore
from bmscientist.coscientist_agents import (
    ResearchPlanningAgent,
    RankingAgent,
    MetaReviewAgent,
)


class MockLLM:
    def __init__(self, response_data):
        self.response_data = response_data
        self.last_system_prompt = None
        self.last_user_prompt = None

    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return response_model.model_validate(self.response_data)


def make_test_hypothesis(hypothesis_id="hyp-1", title="PETG for rigid medical trays") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        research_id="run-1",
        status="reflected",
        title=title,
        summary="A summary of rigid medical trays PETG substitution",
        candidate_material="PETG",
        incumbent_material="PVC",
        application="rigid medical trays",
        strategic_rationale="PETG is sterile-friendly and non-toxic compared to legacy PVC.",
        is_active=True,
        reflection_assessment=ReflectionAssessment.model_validate({
            "strategic_fit_score": {"value": 0.8, "confidence": 0.7},
            "market_size_score": {"value": 0.7, "confidence": 0.5},
            "replacement_fit_score": {"value": 0.75, "confidence": 0.6},
            "activation_ease_score": {"value": 0.8, "confidence": 0.6},
            "technical_success_probability": {"value": 0.7, "confidence": 0.6},
            "commercial_success_probability": {"value": 0.65, "confidence": 0.5},
        }),
    )


def make_test_document() -> ResearchGoalDocument:
    return ResearchGoalDocument(
        research_id="run-1",
        raw_goal="Find bio-friendly rigid material substitutions for PVC.",
        target_hypotheses_final=2,
        target_hypotheses_generated=4,
        regions=["North America"],
        strategic_fit_criteria=["bio-friendly", "rigid"],
    )


def test_hypothesis_feedback_store_integration(tmp_path, monkeypatch):
    # Setup temporary store path
    monkeypatch.setenv("BMSCIENTIST_DATA_DIR", str(tmp_path))
    store = CoScientistStore()

    # Create dummy hypothesis in store
    hyp = make_test_hypothesis()
    store.save_hypothesis(hyp)

    # 1. Apply acceptance feedback
    updated = store.apply_hypothesis_feedback(
        research_id="run-1",
        hypothesis_id="hyp-1",
        status="accepted",
        comment="Strong regulatory pathway.",
    )

    assert updated is not None
    assert updated.user_feedback_status == "accepted"
    assert updated.user_feedback_comment == "Strong regulatory pathway."
    assert updated.is_active is True

    # Reload from store
    loaded = store.load_hypotheses("run-1")[0]
    assert loaded.user_feedback_status == "accepted"
    assert loaded.user_feedback_comment == "Strong regulatory pathway."

    # 2. Apply rejection feedback
    updated = store.apply_hypothesis_feedback(
        research_id="run-1",
        hypothesis_id="hyp-1",
        status="rejected",
        comment="Too expensive for current scope.",
    )
    assert updated.is_active is False
    assert updated.user_feedback_status == "rejected"
    assert updated.user_feedback_comment == "Too expensive for current scope."

    # 3. Apply edit feedback
    updated = store.apply_hypothesis_feedback(
        research_id="run-1",
        hypothesis_id="hyp-1",
        title="Super PETG for medical trays",
        summary="A modified summary",
        status="edited",
    )
    assert updated.is_active is True
    assert updated.user_feedback_status == "edited"
    assert updated.title == "Super PETG for medical trays"
    assert updated.summary == "A modified summary"


def test_ranking_agent_includes_feedback_in_payload():
    hyp = make_test_hypothesis()
    hyp.user_feedback_status = "accepted"
    hyp.user_feedback_comment = "Highly relevant niche"

    mock_llm = MockLLM({
        "rankings": [
            {
                "hypothesis_id": "hyp-1",
                "rank": 1,
                "score": 0.85,
                "recommended_action": "advance",
                "rationale": "High score",
                "strengths": [],
                "weaknesses": [],
                "improvement_directions": [],
            }
        ],
        "best_patterns": [],
        "worst_patterns": [],
    })
    agent = RankingAgent(mock_llm)
    doc = make_test_document()
    agent.rank(doc, [hyp], round_index=1, target_final_count=2, evolve_top_k=1)

    assert mock_llm.last_user_prompt is not None
    assert "user_feedback_status" in mock_llm.last_user_prompt
    assert "user_feedback_comment" in mock_llm.last_user_prompt
    assert "accepted" in mock_llm.last_user_prompt
    assert "Highly relevant niche" in mock_llm.last_user_prompt


def test_accepted_hypothesis_forced_evolution():
    # Make a hypothesis with status accepted and score >= 0.5 (but not in normal evolve_top_k range)
    # Let's say evolve_top_k = 0, but we have an accepted hypothesis with high score.
    hyp1 = make_test_hypothesis("hyp-1")
    hyp1.user_feedback_status = "accepted"
    
    mock_llm = MockLLM({
        "rankings": [
            {
                "hypothesis_id": "hyp-1",
                "rank": 1,
                "score": 0.85,
                "recommended_action": "advance", # normally not evolved if evolve_top_k is 0
                "rationale": "Good",
                "strengths": [],
                "weaknesses": [],
                "improvement_directions": [],
            }
        ],
        "best_patterns": [],
        "worst_patterns": [],
    })
    agent = RankingAgent(mock_llm)
    doc = make_test_document()
    ranking_round, ranked_hypotheses = agent.rank(
        doc, [hyp1], round_index=1, target_final_count=1, evolve_top_k=0
    )
    # It should be forced to evolve because it's accepted and has score >= 0.5!
    assert "hyp-1" in ranking_round.evolved_parent_hypothesis_ids


def test_meta_review_receives_user_feedback():
    hyp = make_test_hypothesis()
    hyp.user_feedback_status = "rejected"
    hyp.user_feedback_comment = "Bad smell"
    
    mock_llm = MockLLM({
        "whitespace_gaps": ["smell issues"],
        "generation_guidance": ["avoid PVC substitutes that smell bad"],
        "coverage_assessment": "Moderate",
        "gap_shrinkage_status": "stable",
        "coverage_sufficient": False,
    })
    agent = MetaReviewAgent(mock_llm)
    doc = make_test_document()
    
    mock_ranking = RankingRound(
        ranking_round_id="r-1",
        research_id="run-1",
        round_index=1,
        candidate_count=1,
        target_final_count=2,
        best_patterns=[],
        worst_patterns=[],
    )
    
    agent.review(doc, [hyp], mock_ranking, round_index=1, gap_overlap_threshold=0.6, max_gap_persistence_rounds=1)
    
    assert mock_llm.last_user_prompt is not None
    assert "user_feedback_status" in mock_llm.last_user_prompt
    assert "rejected" in mock_llm.last_user_prompt
    assert "Bad smell" in mock_llm.last_user_prompt


def test_update_project_goal():
    mock_llm = MockLLM({
        "raw_goal": "Europe focus on recyclability.",
        "regions": ["Europe"],
        "strategic_fit_criteria": ["recyclable"],
        "target_incumbent_materials": ["PVC"],
        "preferred_candidate_materials": [],
        "candidate_material_preferences": [],
        "recycling_or_sustainability_angles": ["recyclable design"],
        "material_scope": ["polymers"],
        "application_scope": ["packaging"],
        "opportunity_modes": [],
        "opportunity_speed_horizon_months": 12,
        "commercialization_constraints": [],
        "ranking_weights": {"strategic_fit": 0.8},
        "success_definition": "Packaging recyclable PVC alternative",
        "strategic_fit_notes": "Focus on EU compliance",
    })
    
    agent = ResearchPlanningAgent(mock_llm)
    doc = make_test_document()
    updated = agent.update_research_goal(doc, "Shift focus to Europe and packaging recyclability.")
    
    assert updated.raw_goal == "Europe focus on recyclability."
    assert updated.regions == ["Europe"]
    assert updated.strategic_fit_criteria == ["recyclable"]
    assert updated.ranking_weights == {"strategic_fit": 0.8}


def test_update_research_goal_preserves_or_updates_evaluation_criteria():
    mock_llm = MockLLM({
        "raw_goal": "Focus on low-toxicity coalescing aids for acrylic latex.",
        "research_mode": "candidate_design",
        "regions": ["North America"],
        "strategic_fit_criteria": ["low aquatic toxicity"],
        "target_incumbent_materials": ["traditional coalescing aids"],
        "preferred_candidate_materials": [],
        "candidate_material_preferences": ["small molecules"],
        "recycling_or_sustainability_angles": ["safer chemistry"],
        "material_scope": ["coating additives"],
        "application_scope": ["waterborne coatings"],
        "opportunity_modes": ["candidate_screening"],
        "opportunity_speed_horizon_months": 12,
        "commercialization_constraints": [],
        "ranking_weights": {"strategic_fit": 0.6, "toxicity": 0.4},
        "success_definition": "Identify candidates worth deeper tool-assisted screening.",
        "candidate_artifact_schema": {
            "artifact_type": "small_molecule",
            "primary_identifier_field": "smiles",
            "required_fields": ["name_or_label", "smiles"],
        },
        "evaluation_criteria": [
            {
                "name": "aquatic_toxicity_risk",
                "description": "Avoid candidates with high aquatic toxicity concern.",
                "direction": "avoid",
                "required_candidate_fields": ["smiles"],
                "suggested_tool_ids": ["opera_qsar"],
            }
        ],
        "reflection_guidance": ["Ask for tool support when toxicity evidence is indirect."],
        "tool_requests": [
            {
                "tool_id": "opera_qsar",
                "purpose": "Predict toxicity-related endpoints from SMILES.",
                "required_inputs": ["smiles"],
                "expected_outputs": ["toxicity_endpoints"],
            }
        ],
        "search_strategy_notes": ["Use SMILES and toxicity terms together."],
        "strategic_fit_notes": "Coatings focus",
    })

    agent = ResearchPlanningAgent(mock_llm)
    doc = make_test_document().model_copy(
        update={
            "research_mode": "candidate_design",
            "candidate_artifact_schema": CandidateArtifactSchema(
                artifact_type="small_molecule",
                primary_identifier_field="smiles",
            ),
            "evaluation_criteria": [
                EvaluationCriterion(
                    name="water_compatibility",
                    description="Look for waterborne suitability.",
                )
            ],
        }
    )

    updated = agent.update_research_goal(doc, "Focus more explicitly on aquatic toxicity.")

    assert updated.research_mode == "candidate_design"
    assert updated.candidate_artifact_schema.primary_identifier_field == "smiles"
    assert updated.evaluation_criteria[0].name == "aquatic_toxicity_risk"
    assert updated.tool_requests[0].tool_id == "opera_qsar"
