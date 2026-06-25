from __future__ import annotations

from pathlib import Path

from bmscientist.coscientist_models import Hypothesis, ResearchGoalDocument
from bmscientist.skills import SkillContext, SkillRegistry, SkillRunner
from bmscientist.skills.rxn4chemistry import RXN4CHEMISTRY_TOOL_ID, RXN4ChemistryRetrosynthesisSkill


class FakeRXNWrapper:
    def __init__(self, *, api_key, project_id=None, base_url=None):
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = base_url
        self.calls = []
        self.created_projects = []

    def list_all_projects(self):
        return {"content": [{"id": "project-1", "name": "bmscientist"}]}

    def create_project(self, name):
        self.created_projects.append(name)
        self.project_id = "project-created"
        return {"response": {"payload": {"id": self.project_id}}}

    def set_project(self, project_id):
        self.project_id = project_id

    def predict_automatic_retrosynthesis(self, smiles, **kwargs):
        self.calls.append(("predict", smiles, kwargs))
        return {"prediction_id": "prediction-1"}

    def get_predict_automatic_retrosynthesis_results(self, prediction_id):
        self.calls.append(("results", prediction_id))
        return {
            "status": "SUCCESS",
            "retrosynthetic_paths": [
                {
                    "sequenceId": "route-1",
                    "children": [
                        {"smiles": "CCO"},
                        {"smiles": "O=C=O"},
                    ],
                },
                {
                    "sequenceId": "route-2",
                    "children": [
                        {
                            "children": [
                                {"smiles": "BrCC"},
                                {"smiles": "O"},
                            ]
                        }
                    ],
                },
            ],
        }


def test_rxn_skill_extracts_synthesis_and_route_metrics(tmp_path: Path):
    wrappers: list[FakeRXNWrapper] = []

    def factory(**kwargs):
        wrapper = FakeRXNWrapper(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    skill = RXN4ChemistryRetrosynthesisSkill(
        api_key="secret",
        project_id="project-1",
        cache_dir=tmp_path / "rxn-cache",
        wrapper_factory=factory,
    )

    summary, results = skill.predict_smiles("CCOC(=O)OCC")
    results_by_name = {result.criterion_name: result for result in results}

    assert summary.route_count == 2
    assert summary.best_route_depth == 2
    assert summary.max_route_depth == 3
    assert results_by_name["retrosynthesis_route_count"].value == 2.0
    assert results_by_name["retrosynthesis_best_route_depth"].value == 2.0
    assert results_by_name["retrosynthesis_max_route_depth"].value == 3.0
    assert results_by_name["synthesis_feasibility"].normalized_score >= 0.7
    assert results_by_name["synthesis_feasibility"].tool_id == RXN4CHEMISTRY_TOOL_ID
    assert wrappers[0].calls[0][0] == "predict"


def test_rxn_skill_uses_cache_for_repeat_calls(tmp_path: Path):
    wrappers: list[FakeRXNWrapper] = []

    def factory(**kwargs):
        wrapper = FakeRXNWrapper(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    skill = RXN4ChemistryRetrosynthesisSkill(
        api_key="secret",
        project_id="project-1",
        cache_dir=tmp_path / "rxn-cache",
        wrapper_factory=factory,
    )

    first_summary, first_results = skill.predict_smiles("CCOC(=O)OCC")
    second_summary, second_results = skill.predict_smiles("CCOC(=O)OCC")

    assert first_summary.route_count == second_summary.route_count == 2
    assert len(first_results) == len(second_results)
    assert len(wrappers) == 1
    assert [call[0] for call in wrappers[0].calls].count("predict") == 1


def test_rxn_skill_blocks_when_configuration_is_missing(tmp_path: Path):
    skill = RXN4ChemistryRetrosynthesisSkill(cache_dir=tmp_path / "rxn-cache")
    context = SkillContext(
        phase="reflection",
        document=ResearchGoalDocument.model_validate(
            {
                "research_id": "rxn-1",
                "raw_goal": "Find makeable coalescent candidates.",
                "target_hypotheses_final": 1,
                "target_hypotheses_generated": 1,
                "research_mode": "candidate_design",
                "candidate_artifact_schema": {
                    "artifact_type": "small_molecule",
                    "primary_identifier_field": "smiles",
                    "required_fields": ["smiles"],
                },
            }
        ),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-1",
                "research_id": "rxn-1",
                "status": "generated",
                "title": "Candidate 1",
                "summary": "Small molecule candidate.",
                "candidate_artifact": {"smiles": "CCOC(=O)OCC"},
            }
        ),
    )

    result = skill.run(context)

    assert result.status == "blocked"
    assert "RXN4CHEMISTRY_API_KEY" in result.notes[0]


def test_rxn_skill_is_discoverable_and_runs_for_generic_rqn_tool_request(tmp_path: Path):
    def factory(**kwargs):
        return FakeRXNWrapper(**kwargs)

    skill = RXN4ChemistryRetrosynthesisSkill(
        api_key="secret",
        project_name="bmscientist",
        cache_dir=tmp_path / "rxn-cache",
        wrapper_factory=factory,
    )
    runner = SkillRunner(SkillRegistry([skill]))
    context = SkillContext(
        phase="reflection",
        document=ResearchGoalDocument.model_validate(
            {
                "research_id": "rxn-2",
                "raw_goal": "Screen novel molecules for synthesis feasibility.",
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
                        "name": "synthesis_feasibility",
                        "description": "Prefer candidates with plausible retrosynthetic routes.",
                        "suggested_tool_ids": ["rxn4chemistry"],
                    }
                ],
                "tool_requests": [
                    {
                        "tool_id": "rxn4chemistry",
                        "purpose": "Run hosted retrosynthesis for makeability checks.",
                        "status": "available",
                    }
                ],
            }
        ),
        hypothesis=Hypothesis.model_validate(
            {
                "hypothesis_id": "hyp-2",
                "research_id": "rxn-2",
                "status": "generated",
                "title": "Candidate 2",
                "summary": "Small molecule candidate.",
                "candidate_artifact": {"smiles": "CCOC(=O)OCC"},
            }
        ),
        purpose="Assess synthesis feasibility for a SMILES candidate.",
        requested_skill_ids=("rxn4chemistry",),
    )

    catalog = runner.catalog_for_context(context)
    results = runner.run_auto(context)

    assert len(catalog) == 1
    assert catalog[0]["skill_id"] == RXN4CHEMISTRY_TOOL_ID
    assert "synthesis_feasibility" in catalog[0]["expected_outputs"]
    assert results
    assert results[0].skill_id == RXN4CHEMISTRY_TOOL_ID
    assert results[0].status == "completed"
