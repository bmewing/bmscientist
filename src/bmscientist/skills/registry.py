from __future__ import annotations

from dataclasses import dataclass

from bmscientist.skills.base import Skill, SkillContext, SkillSpec


@dataclass
class SkillRegistry:
    _skills_by_id: dict[str, Skill]
    _aliases_to_id: dict[str, str]

    def __init__(self, skills: list[Skill] | None = None):
        self._skills_by_id = {}
        self._aliases_to_id = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        self._skills_by_id[skill.spec.skill_id] = skill
        for alias in (skill.spec.skill_id, *skill.spec.aliases):
            normalized = str(alias).strip().lower()
            if normalized:
                self._aliases_to_id[normalized] = skill.spec.skill_id

    def get(self, skill_id: str) -> Skill | None:
        resolved = self.resolve_id(skill_id)
        if resolved is None:
            return None
        return self._skills_by_id.get(resolved)

    def all_specs(self) -> list[SkillSpec]:
        return [skill.spec for skill in self._skills_by_id.values()]

    def all_skills(self) -> list[Skill]:
        return list(self._skills_by_id.values())

    def resolve_id(self, skill_id: str) -> str | None:
        normalized = str(skill_id or "").strip().lower()
        if not normalized:
            return None
        return self._aliases_to_id.get(normalized)

    def resolve_ids(self, skill_ids: list[str] | tuple[str, ...]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()
        for item in skill_ids:
            canonical = self.resolve_id(item)
            if canonical and canonical not in seen:
                seen.add(canonical)
                resolved.append(canonical)
        return resolved

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
