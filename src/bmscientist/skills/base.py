from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from bmscientist.coscientist_models import CandidateEvaluationResult


SkillPhase = Literal["reflection", "generation", "planning", "meta_review"]
SkillStatus = Literal["completed", "skipped", "blocked", "failed"]


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    description: str
    phases: tuple[SkillPhase, ...]
    supported_research_modes: tuple[str, ...] = ()
    required_candidate_fields: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    trigger_keywords: tuple[str, ...] = ()
    provider: str = "local"

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "phases": list(self.phases),
            "supported_research_modes": list(self.supported_research_modes),
            "required_candidate_fields": list(self.required_candidate_fields),
            "expected_outputs": list(self.expected_outputs),
            "trigger_keywords": list(self.trigger_keywords),
            "provider": self.provider,
        }


@dataclass(frozen=True)
class SkillContext:
    phase: SkillPhase
    document: Any
    hypothesis: Any | None = None
    purpose: str = ""
    requested_skill_ids: tuple[str, ...] = ()


@dataclass
class SkillRunResult:
    skill_id: str
    status: SkillStatus
    criterion_results: list[CandidateEvaluationResult] = field(default_factory=list)
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rationale: str = ""

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "criterion_results": [item.model_dump(mode="json") for item in self.criterion_results],
            "evidence_rows": self.evidence_rows,
            "notes": self.notes,
            "rationale": self.rationale,
        }


class Skill(Protocol):
    @property
    def spec(self) -> SkillSpec: ...

    def is_applicable(self, context: SkillContext) -> bool: ...

    def should_run(self, context: SkillContext) -> bool: ...

    def run(self, context: SkillContext) -> SkillRunResult: ...
