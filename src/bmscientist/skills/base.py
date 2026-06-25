from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from bmscientist.coscientist_models import CandidateEvaluationResult


SkillPhase = Literal["reflection", "generation", "planning", "meta_review", "enrichment"]
SkillStatus = Literal["completed", "skipped", "blocked", "failed"]


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    description: str
    phases: tuple[SkillPhase, ...]
    aliases: tuple[str, ...] = ()
    supported_research_modes: tuple[str, ...] = ()
    required_candidate_fields: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    trigger_keywords: tuple[str, ...] = ()
    provider: str = "local"
    priority: int = 100
    requires_safety_review: bool = False

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "phases": list(self.phases),
            "aliases": list(self.aliases),
            "supported_research_modes": list(self.supported_research_modes),
            "required_candidate_fields": list(self.required_candidate_fields),
            "expected_outputs": list(self.expected_outputs),
            "trigger_keywords": list(self.trigger_keywords),
            "provider": self.provider,
            "priority": self.priority,
            "requires_safety_review": self.requires_safety_review,
        }


@dataclass(frozen=True)
class SkillContext:
    phase: SkillPhase
    document: Any
    hypothesis: Any | None = None
    purpose: str = ""
    requested_skill_ids: tuple[str, ...] = ()
    evidence_rows: tuple[dict[str, Any], ...] = ()
    target_count: int | None = None
    avoid_hypotheses: tuple[Any, ...] = ()
    question_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRunResult:
    skill_id: str
    status: SkillStatus
    criterion_results: list[CandidateEvaluationResult] = field(default_factory=list)
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rationale: str = ""
    seed_candidates: list[dict[str, Any]] = field(default_factory=list)
    resolved_identifiers: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "criterion_results": [item.model_dump(mode="json") for item in self.criterion_results],
            "evidence_rows": self.evidence_rows,
            "notes": self.notes,
            "rationale": self.rationale,
            "seed_candidates": self.seed_candidates,
            "resolved_identifiers": self.resolved_identifiers,
            "metadata": self.metadata,
        }


class Skill(Protocol):
    @property
    def spec(self) -> SkillSpec: ...

    def is_applicable(self, context: SkillContext) -> bool: ...

    def should_run(self, context: SkillContext) -> bool: ...

    def run(self, context: SkillContext) -> SkillRunResult: ...
