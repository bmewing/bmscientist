from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

from bmscientist.skills.base import SkillContext, SkillRunResult
from bmscientist.skills.registry import SkillRegistry


LOGGER = logging.getLogger(__name__)


class SkillRunner:
    def __init__(self, registry: SkillRegistry):
        self._registry = registry

    def catalog_for_context(self, context: SkillContext) -> list[dict]:
        return self._registry.catalog_for_context(context)

    def all_skill_catalog(self) -> list[dict]:
        return [spec.as_prompt_dict() for spec in self._registry.all_specs()]

    def run_auto(self, context: SkillContext) -> list[SkillRunResult]:
        requested_ids = OrderedDict.fromkeys(self._registry.resolve_ids(context.requested_skill_ids))
        phase_skills = [skill for skill in self._registry.all_skills() if context.phase in skill.spec.phases]
        skills_by_id = {skill.spec.skill_id: skill for skill in phase_skills}

        ordered = []
        for skill_id in requested_ids:
            skill = skills_by_id.get(skill_id)
            if skill is not None and skill not in ordered:
                ordered.append(skill)
        for skill in sorted(phase_skills, key=lambda item: (item.spec.priority, item.spec.skill_id)):
            if skill not in ordered and skill.is_applicable(context) and skill.should_run(context):
                ordered.append(skill)

        if any(skill.spec.requires_safety_review for skill in ordered):
            safety_skill = skills_by_id.get("safety_triage")
            if safety_skill is not None and safety_skill not in ordered:
                ordered.insert(0, safety_skill)
            elif safety_skill is not None:
                ordered = [safety_skill, *[skill for skill in ordered if skill is not safety_skill]]

        results: list[SkillRunResult] = []
        synthesis_blocked = False
        safety_notes: list[str] = []
        current_context = context
        for skill in ordered:
            if not skill.is_applicable(current_context):
                results.append(
                    SkillRunResult(
                        skill_id=skill.spec.skill_id,
                        status="skipped",
                        notes=["Skill inputs were not available in the current context."],
                        rationale="The skill did not have the identifiers or fields it requires.",
                    )
                )
                continue
            if synthesis_blocked and skill.spec.requires_safety_review:
                results.append(
                    SkillRunResult(
                        skill_id=skill.spec.skill_id,
                        status="blocked",
                        notes=safety_notes or ["Blocked by safety triage policy."],
                        rationale="This synthesis-oriented skill was blocked by a prior safety triage result.",
                        metadata={"blocked_by": "safety_triage", "synthesis_blocked": True},
                    )
                )
                continue
            try:
                result = skill.run(current_context)
            except Exception as exc:
                LOGGER.exception("Skill %s failed during %s phase", skill.spec.skill_id, current_context.phase)
                result = SkillRunResult(
                    skill_id=skill.spec.skill_id,
                    status="failed",
                    notes=[str(exc)],
                    rationale=f"Skill `{skill.spec.skill_id}` failed during execution.",
                )
            results.append(result)
            if skill.spec.skill_id == "safety_triage" and result.metadata.get("synthesis_blocked"):
                synthesis_blocked = True
                safety_notes = list(result.notes)
            current_context = self._context_with_skill_result(current_context, result)
        return results

    @staticmethod
    def _context_with_skill_result(context: SkillContext, result: SkillRunResult) -> SkillContext:
        hypothesis = context.hypothesis
        if hypothesis is None or not result.resolved_identifiers:
            return context

        existing_artifact = dict(getattr(hypothesis, "candidate_artifact", {}) or {})
        merged_artifact = dict(existing_artifact)
        for key, value in result.resolved_identifiers.items():
            if value not in (None, "", []):
                merged_artifact[key] = value
        if "inchikey" in merged_artifact and "inchi_key" not in merged_artifact:
            merged_artifact["inchi_key"] = merged_artifact["inchikey"]
        if "canonical_smiles" in merged_artifact and "smiles" not in merged_artifact:
            merged_artifact["smiles"] = merged_artifact["canonical_smiles"]

        proxy_hypothesis = SimpleNamespace(
            candidate_artifact=merged_artifact,
            title=getattr(hypothesis, "title", None),
            summary=getattr(hypothesis, "summary", None),
            candidate_material=getattr(hypothesis, "candidate_material", None),
            application=getattr(hypothesis, "application", None),
            incumbent_material=getattr(hypothesis, "incumbent_material", None),
            hypothesis_id=getattr(hypothesis, "hypothesis_id", None),
            research_id=getattr(hypothesis, "research_id", None),
        )
        return replace(context, hypothesis=proxy_hypothesis)
