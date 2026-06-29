from __future__ import annotations

import json
from types import SimpleNamespace

import pyarrow.parquet as pq

from bmscientist.coscientist_agents import GenerationAgent, LocalEvidenceRetriever, ResearchPlanningAgent
from bmscientist.coscientist_models import Hypothesis, HypothesisGenerationOutput, ReflectionSearchLimits, ResearchGoalDocument
from bmscientist.graph_enrichment import GraphEnrichmentStore
from bmscientist.skills import (
    MoleculeAvailabilitySkill,
    MoleculeIdentityPubChemSkill,
    NoveltyPatentScreenSkill,
    RDKitProfileSkill,
    RXN4ChemistryRetrosynthesisSkill,
    SafetyTriageSkill,
    SkillContext,
    SkillRegistry,
    SkillRunResult,
    SkillRunner,
    SkillSpec,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakePubChemSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if "/compound/smiles/" in url and "/cids/JSON" in url:
            return FakeResponse({"IdentifierList": {"CID": [123]}})
        if "/compound/name/" in url and "/cids/JSON" in url:
            return FakeResponse({"IdentifierList": {"CID": [123]}})
        if "/compound/cid/123/property/" in url:
            return FakeResponse(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CanonicalSMILES": "CCOC(=O)OCC",
                                "IsomericSMILES": "CCOC(=O)OCC",
                                "MolecularFormula": "C6H12O3",
                                "MolecularWeight": 132.16,
                                "IUPACName": "ethyl 2-hydroxypropanoate",
                                "XLogP": 1.5,
                                "TPSA": 35.5,
                                "RotatableBondCount": 4,
                            }
                        ]
                    }
                }
            )
        if "/compound/cid/123/synonyms/JSON" in url:
            return FakeResponse({"InformationList": {"Information": [{"Synonym": ["Ethyl lactate", "97-64-3"]}]}})
        if "/compound/cid/123/sids/JSON" in url:
            return FakeResponse({"InformationList": {"Information": [{"SID": [1, 2, 3, 4, 5, 6]}]}})
        if "/pug_view/data/compound/123/JSON" in url:
            return FakeResponse(
                {
                    "Record": {
                        "Section": [
                            {
                                "TOCHeading": "Safety and Hazards",
                                "Section": [
                                    {
                                        "TOCHeading": "Hazards Identification",
                                        "Information": [
                                            {"Value": {"StringWithMarkup": [{"String": "Explosive when heated."}]}}
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                }
            )
        raise AssertionError(f"Unexpected URL {url}")


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows

    def retrieve_for_goal(self, document, max_results=12):
        return list(self.rows)


class FakeNoveltySearchClient:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, num_results, options=None):
        self.calls.append((query, num_results, options))
        return SimpleNamespace(results=list(self.results))


class PlanningCaptureLLM:
    def __init__(self):
        self.user_prompts = []

    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        self.user_prompts.append(user_prompt)
        return response_model.model_validate(
            {
                "research_mode": "candidate_design",
                "candidate_artifact_schema": {
                    "artifact_type": "small_molecule",
                    "primary_identifier_field": "smiles",
                    "required_fields": ["smiles"],
                },
                "tool_requests": [
                    {
                        "tool_id": "molecule_identity_pubchem",
                        "purpose": "Resolve candidate identifiers through PubChem.",
                    }
                ],
            }
        )


class GenerationCaptureLLM:
    def __init__(self):
        self.user_prompts = []

    def complete_json(self, response_model, system_prompt, user_prompt, temperature=0.1):
        self.user_prompts.append(user_prompt)
        return HypothesisGenerationOutput.model_validate(
            {
                "hypotheses": [
                    {
                        "title": "Designed coalescent A",
                        "summary": "Generated from a molecule neighbor seed.",
                        "application": "waterborne coatings",
                        "candidate_artifact": {
                            "name_or_label": "Designed coalescent A",
                            "smiles": "CCOC(=O)OCC",
                        },
                        "generation_confidence": 0.5,
                    }
                ]
            }
        )


class FakeGenerationSkill:
    def __init__(self):
        self._spec = SkillSpec(
            skill_id="molecule_neighbor_expansion",
            description="Fake molecule seed expansion skill.",
            phases=("generation",),
            expected_outputs=("generation_seed_candidates",),
            priority=25,
        )

    @property
    def spec(self):
        return self._spec

    def is_applicable(self, context):
        return True

    def should_run(self, context):
        return True

    def run(self, context):
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            seed_candidates=[
                {
                    "title": "Analog seed from benchmark",
                    "candidate_artifact": {"name_or_label": "Seed A", "smiles": "CCOC(=O)OCC"},
                    "rationale": "Similarity neighbor.",
                }
            ],
            evidence_rows=[
                {
                    "id": "seed-1",
                    "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                    "source_title": "PubChem analog seed",
                    "chunk_text": "Analog seed evidence row.",
                }
            ],
            rationale="Generated one seed candidate.",
        )


class FakeRXNWrapper:
    def __init__(self, *, api_key, project_id=None, base_url=None):
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = base_url

    def predict_automatic_retrosynthesis(self, smiles, **kwargs):
        raise AssertionError("RXN wrapper should not run when safety triage blocks synthesis.")

    def get_predict_automatic_retrosynthesis_results(self, prediction_id):
        raise AssertionError("RXN wrapper should not run when safety triage blocks synthesis.")


def make_smiles_document() -> ResearchGoalDocument:
    return ResearchGoalDocument.model_validate(
        {
            "research_id": "mol-1",
            "raw_goal": "Find novel coalescing-aid molecules.",
            "target_hypotheses_final": 1,
            "target_hypotheses_generated": 1,
            "research_mode": "candidate_design",
            "candidate_artifact_schema": {
                "artifact_type": "small_molecule",
                "primary_identifier_field": "smiles",
                "required_fields": ["smiles"],
            },
            "known_candidate_exclusion_terms": ["CCOCC"],
            "candidate_origin_policy": "de_novo_design",
        }
    )


def make_smiles_hypothesis() -> Hypothesis:
    return Hypothesis.model_validate(
        {
            "hypothesis_id": "hyp-1",
            "research_id": "mol-1",
            "status": "generated",
            "title": "Candidate A",
            "summary": "Small molecule candidate.",
            "candidate_artifact": {"smiles": "CCOC(=O)OCC", "name_or_label": "Candidate A"},
        }
    )


def make_name_document() -> SimpleNamespace:
    return SimpleNamespace(
        research_mode="candidate_design",
        candidate_artifact_schema=SimpleNamespace(primary_identifier_field="name_or_label"),
        evaluation_criteria=[],
        reflection_guidance=[],
        novelty_check_policy="",
        known_candidate_exclusion_terms=[],
    )


def make_name_hypothesis() -> SimpleNamespace:
    return SimpleNamespace(
        title="Ethyl lactate candidate",
        summary="Candidate identified from enrichment evidence.",
        candidate_material="Ethyl lactate",
        application="coalescing aid",
        incumbent_material=None,
        candidate_artifact={"name_or_label": "Ethyl lactate"},
    )


def test_rdkit_profile_skill_extracts_descriptor_results():
    skill = RDKitProfileSkill()
    context = SkillContext(phase="reflection", document=make_smiles_document(), hypothesis=make_smiles_hypothesis())

    result = skill.run(context)
    results_by_name = {item.criterion_name: item for item in result.criterion_results}

    assert result.status == "completed"
    assert results_by_name["molecular_weight_rdkit"].value > 100
    assert results_by_name["tpsa_rdkit"].value > 0
    assert result.metadata["functional_groups"]


def test_pubchem_identity_and_availability_skills_use_fake_pubchem_session(tmp_path):
    from bmscientist.skills.pubchem_support import PubChemClient

    client = PubChemClient(session=FakePubChemSession(), cache_dir=tmp_path / "pubchem-cache")
    context = SkillContext(phase="reflection", document=make_smiles_document(), hypothesis=make_smiles_hypothesis())

    identity = MoleculeIdentityPubChemSkill(pubchem_client=client).run(context)
    availability = MoleculeAvailabilitySkill(pubchem_client=client).run(context)

    assert identity.status == "completed"
    assert identity.metadata["cid"] == 123
    assert identity.resolved_identifiers["cas_number"] == "97-64-3"
    assert availability.status == "completed"
    assert availability.metadata["source_record_count"] == 6


def test_safety_triage_blocks_rxn_skill_through_runner(tmp_path):
    from bmscientist.skills.pubchem_support import PubChemClient

    client = PubChemClient(session=FakePubChemSession(), cache_dir=tmp_path / "pubchem-cache")
    safety = SafetyTriageSkill(pubchem_client=client)
    rxn = RXN4ChemistryRetrosynthesisSkill(
        api_key="secret",
        project_id="project-1",
        wrapper_factory=lambda **kwargs: FakeRXNWrapper(**kwargs),
        cache_dir=tmp_path / "rxn-cache",
    )
    runner = SkillRunner(SkillRegistry([rxn, safety]))
    document = make_smiles_document().model_copy(
        update={
            "tool_requests": [
                {
                    "tool_id": "rxn4chemistry",
                    "purpose": "Assess synthesis feasibility.",
                    "status": "available",
                }
            ]
        }
    )
    context = SkillContext(
        phase="reflection",
        document=document,
        hypothesis=make_smiles_hypothesis(),
        requested_skill_ids=("rxn4chemistry",),
    )

    results = runner.run_auto(context)

    assert results[0].skill_id == "safety_triage"
    assert results[0].metadata["synthesis_blocked"] is True
    assert results[1].skill_id == "rxn4chemistry_retrosynthesis"
    assert results[1].status == "blocked"


def test_enrichment_runner_propagates_pubchem_identity_into_rdkit_profile(tmp_path):
    from bmscientist.skills.pubchem_support import PubChemClient

    client = PubChemClient(session=FakePubChemSession(), cache_dir=tmp_path / "pubchem-cache")
    runner = SkillRunner(
        SkillRegistry(
            [
                MoleculeIdentityPubChemSkill(pubchem_client=client),
                RDKitProfileSkill(cache_dir=tmp_path / "rdkit-cache"),
            ]
        )
    )
    context = SkillContext(
        phase="enrichment",
        document=make_name_document(),
        hypothesis=make_name_hypothesis(),
        requested_skill_ids=("molecule_identity_pubchem", "rdkit_profile"),
    )

    results = runner.run_auto(context)

    assert results[0].skill_id == "molecule_identity_pubchem"
    assert results[0].status == "completed"
    assert results[1].skill_id == "rdkit_profile"
    assert results[1].status == "completed"
    assert any(item.criterion_name == "molecular_weight_rdkit" for item in results[1].criterion_results)


def test_graph_enrichment_store_writes_skill_results_to_product_and_endpoint_graph(tmp_path):
    from bmscientist.skills.pubchem_support import PubChemClient

    client = PubChemClient(session=FakePubChemSession(), cache_dir=tmp_path / "pubchem-cache")
    runner = SkillRunner(
        SkillRegistry(
            [
                MoleculeIdentityPubChemSkill(pubchem_client=client),
                RDKitProfileSkill(cache_dir=tmp_path / "rdkit-cache"),
            ]
        )
    )
    context = SkillContext(
        phase="enrichment",
        document=make_name_document(),
        hypothesis=make_name_hypothesis(),
        requested_skill_ids=("molecule_identity_pubchem", "rdkit_profile"),
    )
    results = [result for result in runner.run_auto(context) if result.status == "completed"]

    graph_path = tmp_path / "graph"
    write_count = GraphEnrichmentStore(graph_path).write_skill_enrichments(
        candidate_artifact={"name_or_label": "Ethyl lactate"},
        skill_results=results,
        source_chunk_id="chunk-skill-1",
        source_url="https://example.com/ethyl-lactate",
        source_title="Ethyl lactate note",
        supporting_quote="Ethyl lactate was identified in the evidence.",
    )

    assert write_count >= 2


def test_novelty_skill_uses_free_prior_art_search_domains(tmp_path):
    from bmscientist.models import SearchResultItem
    from bmscientist.skills.pubchem_support import PubChemClient

    client = PubChemClient(session=FakePubChemSession(), cache_dir=tmp_path / "pubchem-cache")
    search_client = FakeNoveltySearchClient(
        results=[
            SearchResultItem.model_validate(
                {
                    "title": "Google Patents result for ethyl lactate",
                    "url": "https://patents.google.com/patent/US1234567A/en",
                    "search_query": "\"Ethyl lactate\"",
                    "snippet": "Ethyl lactate is described as a solvent candidate.",
                }
            )
        ]
    )
    skill = NoveltyPatentScreenSkill(pubchem_client=client, search_client=search_client)
    context = SkillContext(phase="reflection", document=make_name_document(), hypothesis=make_name_hypothesis())

    result = skill.run(context)

    assert result.status == "completed"
    assert result.metadata["free_prior_art_hits"]
    assert any("patents/preprints" in note for note in result.notes)
    assert search_client.calls
    first_call = search_client.calls[0]
    assert "patents.google.com" in first_call[2].include_domains
    assert "chemrxiv.org" in first_call[2].include_domains


def test_research_planning_prompt_includes_available_skills():
    llm = PlanningCaptureLLM()
    runner = SkillRunner(SkillRegistry([MoleculeIdentityPubChemSkill(pubchem_client=None), RDKitProfileSkill()]))
    agent = ResearchPlanningAgent(llm, runner)

    document = agent.create_research_goal(
        research_id="plan-1",
        raw_goal="Find coalescent molecules with lower toxicity risk.",
        target_hypotheses_final=2,
        regions=["North America"],
        strategic_fit_notes=None,
        preferred_evidence_recency_days=180,
        reflection_search_limits=ReflectionSearchLimits(),
    )

    assert document.tool_requests[0].tool_id == "molecule_identity_pubchem"
    assert '"skill_id": "molecule_identity_pubchem"' in llm.user_prompts[0]


def test_generation_prompt_includes_seed_candidates_from_generation_skills():
    llm = GenerationCaptureLLM()
    retriever = FakeRetriever(rows=[{"id": "chunk-1", "source_url": "https://example.com", "source_title": "Example", "chunk_text": "Goal evidence."}])
    runner = SkillRunner(SkillRegistry([FakeGenerationSkill()]))
    agent = GenerationAgent(llm, retriever, skill_runner=runner)

    hypotheses = agent.generate(make_smiles_document().model_copy(update={"target_hypotheses_generated": 1}))

    assert len(hypotheses) == 1
    assert '"title": "Analog seed from benchmark"' in llm.user_prompts[0]
    assert '"skill_id": "molecule_neighbor_expansion"' in llm.user_prompts[0]
