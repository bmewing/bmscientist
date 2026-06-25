from __future__ import annotations

from pathlib import Path

from bmscientist.coscientist_models import Hypothesis, ResearchGoalDocument
from bmscientist.skills import SkillContext, SkillRegistry, SkillRunner
from bmscientist.skills.episuite import EPISuiteSkill


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.payload)


def test_episuite_skill_extracts_curated_endpoints_from_nested_payload(tmp_path: Path):
    payload = {
        "physical_properties": {
            "boilingPoint": {"value": 248.3, "unit": "C"},
            "vaporPressure": {"value": "0.12 mm Hg", "unit": "mm Hg"},
            "waterSolubility": {"value": 1850, "unit": "mg/L"},
            "logKow": 2.14,
        },
        "fate": {
            "henryLawConstant": {"value": 1.3e-07, "unit": "atm-m3/mol"},
            "soilAdsorption": {"logKoc": 1.9},
            "bioconcentration": {"bcf": 12.7},
            "biodegradationProbability": {"value": 0.81},
        },
        "ecotoxicity": {
            "fishLC50": {"value": 48.5, "unit": "mg/L"},
            "daphniaEC50": {"value": 65.0, "unit": "mg/L"},
        },
    }
    session = FakeSession(payload)
    skill = EPISuiteSkill(session=session, cache_dir=tmp_path / "episuite-cache", timeout_seconds=9)

    results = skill.predict_smiles("CCOC(=O)OCC")
    results_by_name = {result.criterion_name: result for result in results}

    assert results_by_name["boiling_point_c"].value == 248.3
    assert results_by_name["boiling_point_c"].unit == "C"
    assert results_by_name["water_solubility_mg_l"].value == 1850.0
    assert results_by_name["log_kow"].value == 2.14
    assert results_by_name["log_koc"].value == 1.9
    assert results_by_name["bioconcentration_factor_bcf"].value == 12.7
    assert results_by_name["ready_biodegradation_probability"].value == 0.81
    assert results_by_name["fish_lc50_mg_l"].value == 48.5
    assert results_by_name["daphnia_ec50_mg_l"].value == 65.0
    assert all(result.tool_id == "epa_episuite" for result in results)
    assert len(session.calls) == 1


def test_episuite_skill_uses_cache_for_repeat_calls(tmp_path: Path):
    payload = {"physical_properties": {"logKow": 2.14}}
    session = FakeSession(payload)
    skill = EPISuiteSkill(session=session, cache_dir=tmp_path / "episuite-cache")

    first = skill.predict_smiles("CCOC(=O)OCC")
    second = skill.predict_smiles("CCOC(=O)OCC")

    assert len(first) == 1
    assert len(second) == 1
    assert len(session.calls) == 1


def test_episuite_skill_is_discoverable_through_registry(tmp_path: Path):
    skill = EPISuiteSkill(cache_dir=tmp_path / "episuite-cache")
    runner = SkillRunner(SkillRegistry([skill]))
    context = SkillContext(
        phase="reflection",
        document=ResearchGoalDocument.model_validate(
            {
                "research_id": "skills-1",
                "raw_goal": "Screen SMILES candidates for toxicity and fate.",
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
                "research_id": "skills-1",
                "status": "generated",
                "title": "Candidate 1",
                "summary": "Small molecule candidate.",
                "candidate_artifact": {"smiles": "CCOC(=O)OCC"},
            }
        ),
        purpose="Assess toxicity and fate signals for a SMILES candidate.",
        requested_skill_ids=("epa_episuite",),
    )

    catalog = runner.catalog_for_context(context)

    assert len(catalog) == 1
    assert catalog[0]["skill_id"] == "epa_episuite"
    assert "log_kow" in catalog[0]["expected_outputs"]
