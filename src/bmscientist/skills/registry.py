from __future__ import annotations

from dataclasses import dataclass

from bmscientist.skills.base import Skill, SkillContext, SkillSpec


@dataclass
class SkillRegistry:
    _skills_by_id: dict[str, Skill]

    def __init__(self, skills: list[Skill] | None = None):
        self._skills_by_id = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        self._skills_by_id[skill.spec.skill_id] = skill

    def get(self, skill_id: str) -> Skill | None:
        return self._skills_by_id.get(skill_id)

    def specs_for_context(self, context: SkillContext) -> list[SkillSpec]:
        return [
            skill.spec
            for skill in self._skills_by_id.values()
            if context.phase in skill.spec.phases and skill.is_applicable(context)
        ]

    def catalog_for_context(self, context: SkillContext) -> list[dict]:
        return [spec.as_prompt_dict() for spec in self.specs_for_context(context)]

    def skills_for_context(self, context: SkillContext) -> list[Skill]:
        return [
            skill
            for skill in self._skills_by_id.values()
            if context.phase in skill.spec.phases and skill.is_applicable(context)
        ]
