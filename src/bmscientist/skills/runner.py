from __future__ import annotations

import logging
from collections import OrderedDict

from bmscientist.skills.base import SkillContext, SkillRunResult
from bmscientist.skills.registry import SkillRegistry


LOGGER = logging.getLogger(__name__)


class SkillRunner:
    def __init__(self, registry: SkillRegistry):
        self._registry = registry

    def catalog_for_context(self, context: SkillContext) -> list[dict]:
        return self._registry.catalog_for_context(context)

    def run_auto(self, context: SkillContext) -> list[SkillRunResult]:
        requested_ids = OrderedDict.fromkeys(context.requested_skill_ids)
        skills = self._registry.skills_for_context(context)

        ordered = []
        for skill_id in requested_ids:
            skill = self._registry.get(skill_id)
            if skill is not None and skill in skills:
                ordered.append(skill)
        for skill in skills:
            if skill not in ordered and skill.should_run(context):
                ordered.append(skill)

        results: list[SkillRunResult] = []
        for skill in ordered:
            try:
                results.append(skill.run(context))
            except Exception as exc:
                LOGGER.exception("Skill %s failed during %s phase", skill.spec.skill_id, context.phase)
                results.append(
                    SkillRunResult(
                        skill_id=skill.spec.skill_id,
                        status="failed",
                        notes=[str(exc)],
                        rationale=f"Skill `{skill.spec.skill_id}` failed during execution.",
                    )
                )
        return results
