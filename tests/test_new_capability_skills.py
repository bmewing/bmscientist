from __future__ import annotations

from bmscientist.coscientist_models import Hypothesis, ResearchGoalDocument
from bmscientist.skills import (
    HansenSolubilityXGBoostSkill,
    MolToxPredScreenSkill,
    PolymerPropertyProfileSkill,
    SkillContext,
    SkillRegistry,
    SkillRunner,
)


def make_small_molecule_document() -> ResearchGoalDocument:
    return ResearchGoalDocument.model_validate(
        {
            "research_id": "tox-1",
            "raw_goal": "Find safer small-molecule coalescents with low toxicity.",
            "target_hypotheses_final": 1,
            "target_hypotheses_generated": 1,
            "research_mode": "candidate_design",
            "candidate_artifact_schema": {
                "artifact_type": "small_molecule",
                "primary_identifier_field": "smiles",
                "required_fields": ["smiles"],
            },
            "evaluation_criteria": [
                {
                    "name": "toxicity",
                    "description": "Prefer lower toxicity and fewer Tox21-like structural alerts.",
                }
            ],
        }
    )


def make_polymer_document() -> ResearchGoalDocument:
    return ResearchGoalDocument.model_validate(
        {
            "research_id": "poly-1",
            "raw_goal": "Profile polymer repeat units for Tg, density, permeability, and solubility parameter.",
            "target_hypotheses_final": 1,
            "target_hypotheses_generated": 1,
            "research_mode": "formulation_design",
            "candidate_artifact_schema": {
                "artifact_type": "polymer",
                "primary_identifier_field": "repeat_unit_smiles",
                "required_fields": ["repeat_unit_smiles"],
            },
            "evaluation_criteria": [
                {
                    "name": "glass_transition",
                    "description": "Estimate polymer glass transition temperature and density.",
                }
            ],
        }
    )


def test_moltoxpred_screen_returns_structured_toxicity_outputs(tmp_path):
    skill = MolToxPredScreenSkill(cache_dir=tmp_path / "tox-cache")
    context = SkillContext(
        phase="reflection",
        document=make_small_molecule_document(),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-tox",
                "research_id": "tox-1",
                "status": "generated",
                "title": "Nitrobenzene analog",
                "summary": "Small molecule candidate.",
                "candidate_artifact": {"smiles": "O=[N+]([O-])c1ccccc1", "name_or_label": "Nitrobenzene analog"},
            }
        ),
    )

    result = skill.run(context)
    results_by_name = {item.criterion_name: item for item in result.criterion_results}

    assert result.status == "completed"
    assert results_by_name["moltoxpred_toxicity_score"].value > 0.4
    assert results_by_name["moltoxpred_toxicity_label"].value in {"moderate_concern", "higher_concern"}
    assert results_by_name["tox21_structural_alert_count"].value >= 1
    assert result.metadata["matched_alerts"]
    assert "bioinformatics-cdac/MolToxPred" in result.notes[1]


def test_polymer_property_profile_returns_agent_usable_values(tmp_path):
    skill = PolymerPropertyProfileSkill(cache_dir=tmp_path / "poly-cache")
    context = SkillContext(
        phase="reflection",
        document=make_polymer_document(),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-poly",
                "research_id": "poly-1",
                "status": "generated",
                "title": "Polyester repeat unit",
                "summary": "Polymer repeat-unit candidate.",
                "candidate_artifact": {
                    "repeat_unit_smiles": "[*:1]CC(=O)O[*:2]",
                    "name_or_label": "Polyester repeat unit",
                },
            }
        ),
    )

    result = skill.run(context)
    results_by_name = {item.criterion_name: item for item in result.criterion_results}

    assert result.status == "completed"
    assert results_by_name["polymer_tg_estimate_k"].value > 170
    assert results_by_name["polymer_density_estimate_g_cm3"].value > 0
    assert results_by_name["polymer_solubility_parameter_mpa05"].value > 0
    assert result.resolved_identifiers["canonical_repeat_unit_smiles"]
    assert result.metadata["reference_project"] == "polymer_property_prediction"


def test_new_capability_skills_are_discoverable_and_autorun(tmp_path):
    runner = SkillRunner(
        SkillRegistry(
            [
                MolToxPredScreenSkill(cache_dir=tmp_path / "tox-cache"),
                PolymerPropertyProfileSkill(cache_dir=tmp_path / "poly-cache"),
            ]
        )
    )
    context = SkillContext(
        phase="reflection",
        document=make_small_molecule_document(),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-safe",
                "research_id": "tox-1",
                "status": "generated",
                "title": "Ethyl acetate",
                "summary": "Small molecule candidate.",
                "candidate_artifact": {"smiles": "CCOC(C)=O"},
            }
        ),
        requested_skill_ids=("moltoxpred",),
    )

    catalog = runner.catalog_for_context(context)
    results = runner.run_auto(context)

    assert any(item["skill_id"] == "moltoxpred_screen" for item in catalog)
    assert results[0].skill_id == "moltoxpred_screen"
    assert results[0].status == "completed"


def test_hansen_solubility_xgboost_skill_predicts_hsp_components(tmp_path):
    skill = HansenSolubilityXGBoostSkill(cache_dir=tmp_path / "hsp-cache")
    context = SkillContext(
        phase="reflection",
        document=ResearchGoalDocument.model_validate(
            {
                "research_id": "hsp-1",
                "raw_goal": "Assess Hansen solubility parameters for binder compatibility.",
                "target_hypotheses_final": 1,
                "target_hypotheses_generated": 1,
                "research_mode": "candidate_design",
                "candidate_artifact_schema": {
                    "artifact_type": "small_molecule",
                    "primary_identifier_field": "smiles",
                    "required_fields": ["smiles"],
                },
                "evaluation_criteria": [
                    {
                        "name": "binder compatibility",
                        "description": "Use Hansen solubility parameters for latex compatibility.",
                    }
                ],
            }
        ),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-hsp",
                "research_id": "hsp-1",
                "status": "generated",
                "title": "Ethyl acetate",
                "summary": "Small molecule HSP test candidate.",
                "candidate_artifact": {"smiles": "CCOC(C)=O", "name_or_label": "Ethyl acetate"},
            }
        ),
        requested_skill_ids=("hsp_prediction",),
    )

    result = skill.run(context)
    results_by_name = {item.criterion_name: item for item in result.criterion_results}

    assert result.status == "completed"
    assert results_by_name["hsp_delta_d_mpa05"].value > 0
    assert results_by_name["hsp_delta_p_mpa05"].value >= 0
    assert results_by_name["hsp_delta_h_mpa05"].value >= 0
    assert results_by_name["hsp_total_mpa05"].value > results_by_name["hsp_delta_d_mpa05"].value
    assert result.metadata["model_variant"] == "50"
    assert result.metadata["reference_repo"].endswith("HSP-predictions")


def test_hansen_solubility_skill_is_discoverable_by_alias(tmp_path):
    runner = SkillRunner(SkillRegistry([HansenSolubilityXGBoostSkill(cache_dir=tmp_path / "hsp-cache")]))
    context = SkillContext(
        phase="reflection",
        document=make_small_molecule_document(),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-hsp-runner",
                "research_id": "tox-1",
                "status": "generated",
                "title": "Ethyl lactate",
                "summary": "Small molecule candidate.",
                "candidate_artifact": {"smiles": "CCOC(=O)C(O)C"},
            }
        ),
        requested_skill_ids=("hansen_solubility",),
    )

    results = runner.run_auto(context)

    assert results[0].skill_id == "hansen_solubility_xgboost"
    assert results[0].status == "completed"
