from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


HypothesisStatus = Literal["generated", "reflected", "evolve", "retired"]
HypothesisGenerationSource = Literal["initial", "evolved", "regenerated", "synthesized"]
RankingAction = Literal["advance", "hold", "evolve", "reject"]
GapShrinkageStatus = Literal["improved", "stable", "worse", "unknown"]


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(item).strip() for item in value if str(item).strip()]


def coerce_metric_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def short_title_from_text(text: Any) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "Untitled hypothesis"
    first_sentence = cleaned.split(".")[0].strip()
    return (first_sentence or cleaned)[:120]


def normalize_confidence_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))

    text = str(value).strip().lower()
    if not text:
        return 0.0
    named_levels = {
        "very low": 0.15,
        "low": 0.3,
        "medium-low": 0.4,
        "medium": 0.55,
        "moderate": 0.55,
        "medium-high": 0.7,
        "high": 0.8,
        "very high": 0.92,
    }
    if text in named_levels:
        return named_levels[text]
    if text.endswith("%"):
        try:
            return max(0.0, min(1.0, float(text[:-1].strip()) / 100.0))
        except ValueError:
            return 0.0
    try:
        numeric = float(text)
    except ValueError:
        return 0.0
    if numeric > 1.0 and numeric <= 100.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


class ReflectionSearchLimits(BaseModel):
    max_reflection_searches_per_hypothesis: int = Field(default=3, ge=1, le=10)
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_pages_per_search: int = Field(default=8, ge=1, le=50)


class ResearchGoalDocument(BaseModel):
    research_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_goal: str
    target_hypotheses_final: int = Field(ge=1)
    target_hypotheses_generated: int = Field(ge=1)
    regions: list[str] = Field(default_factory=list)
    strategic_fit_criteria: list[str] = Field(default_factory=list)
    target_incumbent_materials: list[str] = Field(default_factory=list)
    preferred_candidate_materials: list[str] = Field(default_factory=list)
    candidate_material_preferences: list[str] = Field(default_factory=list)
    recycling_or_sustainability_angles: list[str] = Field(default_factory=list)
    preferred_evidence_recency_days: int = Field(default=180, ge=1)
    reflection_search_limits: ReflectionSearchLimits = Field(default_factory=ReflectionSearchLimits)
    material_scope: list[str] = Field(default_factory=list)
    application_scope: list[str] = Field(default_factory=list)
    opportunity_modes: list[str] = Field(default_factory=list)
    opportunity_speed_horizon_months: int | None = Field(default=None, ge=1)
    commercialization_constraints: list[str] = Field(default_factory=list)
    ranking_weights: dict[str, float] = Field(default_factory=dict)
    success_definition: str = Field(default="")
    strategic_fit_notes: str | None = None
    whitespace_gap_notes: list[str] = Field(default_factory=list)
    whitespace_gap_persistence_count: int = Field(default=0, ge=0)
    meta_review_generation_guidance: list[str] = Field(default_factory=list)
    emerging_concept_labels: list[str] = Field(default_factory=list)
    last_meta_review_round_index: int | None = Field(default=None, ge=0)

    @field_validator(
        "regions",
        "strategic_fit_criteria",
        "target_incumbent_materials",
        "preferred_candidate_materials",
        "candidate_material_preferences",
        "recycling_or_sustainability_angles",
        "material_scope",
        "application_scope",
        "opportunity_modes",
        "commercialization_constraints",
        "whitespace_gap_notes",
        "meta_review_generation_guidance",
        "emerging_concept_labels",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("ranking_weights", mode="before")
    @classmethod
    def default_weights(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        return {str(key): float(raw_value) for key, raw_value in value.items()}


class ResearchPlanDraft(BaseModel):
    strategic_fit_criteria: list[str] = Field(default_factory=list)
    target_incumbent_materials: list[str] = Field(default_factory=list)
    preferred_candidate_materials: list[str] = Field(default_factory=list)
    candidate_material_preferences: list[str] = Field(default_factory=list)
    recycling_or_sustainability_angles: list[str] = Field(default_factory=list)
    material_scope: list[str] = Field(default_factory=list)
    application_scope: list[str] = Field(default_factory=list)
    opportunity_modes: list[str] = Field(default_factory=list)
    opportunity_speed_horizon_months: int | None = Field(default=None, ge=1)
    commercialization_constraints: list[str] = Field(default_factory=list)
    ranking_weights: dict[str, float] = Field(default_factory=dict)
    success_definition: str = Field(default="")

    @field_validator(
        "strategic_fit_criteria",
        "target_incumbent_materials",
        "preferred_candidate_materials",
        "candidate_material_preferences",
        "recycling_or_sustainability_angles",
        "material_scope",
        "application_scope",
        "opportunity_modes",
        "commercialization_constraints",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("ranking_weights", mode="before")
    @classmethod
    def default_weights(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        return {str(key): float(raw_value) for key, raw_value in value.items()}


class EvidenceCitation(BaseModel):
    chunk_id: str
    source_url: str
    source_title: str
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieved_at: str | None = None


class AssessmentMetric(BaseModel):
    value: float | None = None
    rationale: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    is_inferred: bool = False

    @field_validator("citation_chunk_ids", "citation_urls", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class PriceMetric(BaseModel):
    value: float | None = None
    rationale: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    is_inferred: bool = False

    @field_validator("citation_chunk_ids", "citation_urls", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class ReflectionAssessment(BaseModel):
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategic_fit_score: AssessmentMetric = Field(default_factory=AssessmentMetric)
    market_size_score: AssessmentMetric = Field(default_factory=AssessmentMetric)
    incumbent_price_usd_per_kg: PriceMetric = Field(default_factory=PriceMetric)
    nbca_material: str | None = None
    nbca_price_usd_per_kg: PriceMetric = Field(default_factory=PriceMetric)
    replacement_fit_score: AssessmentMetric = Field(default_factory=AssessmentMetric)
    activation_ease_score: AssessmentMetric = Field(default_factory=AssessmentMetric)
    replacement_driver_strength_score: AssessmentMetric = Field(default_factory=AssessmentMetric)
    technical_success_probability: AssessmentMetric = Field(default_factory=AssessmentMetric)
    commercial_success_probability: AssessmentMetric = Field(default_factory=AssessmentMetric)
    reflection_search_queries: list[str] = Field(default_factory=list)
    reflection_discovery_run_ids: list[str] = Field(default_factory=list)
    evidence_gap_notes: list[str] = Field(default_factory=list)

    @field_validator(
        "strategic_fit_score",
        "market_size_score",
        "replacement_fit_score",
        "activation_ease_score",
        "replacement_driver_strength_score",
        "technical_success_probability",
        "commercial_success_probability",
        mode="before",
    )
    @classmethod
    def normalize_assessment_metrics(cls, value: Any) -> dict[str, Any]:
        return coerce_metric_payload(value)

    @field_validator("incumbent_price_usd_per_kg", "nbca_price_usd_per_kg", mode="before")
    @classmethod
    def normalize_price_metrics(cls, value: Any) -> dict[str, Any]:
        return coerce_metric_payload(value)

    @field_validator("nbca_material", mode="before")
    @classmethod
    def normalize_nbca_material(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):
            raw_value = value.get("value")
            if raw_value is None:
                return None
            text = str(raw_value).strip()
            return text or None
        text = str(value).strip()
        return text or None

    @field_validator("reflection_search_queries", "reflection_discovery_run_ids", "evidence_gap_notes", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class ReflectionReviewOutput(BaseModel):
    assessment: ReflectionAssessment = Field(default_factory=ReflectionAssessment)
    needs_additional_search: bool = False
    follow_up_search_queries: list[str] = Field(default_factory=list)

    @field_validator("assessment", mode="before")
    @classmethod
    def normalize_assessment(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {}

    @field_validator("follow_up_search_queries", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class Hypothesis(BaseModel):
    hypothesis_id: str
    research_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: HypothesisStatus
    title: str
    summary: str
    application: str | None = None
    market_segment: str | None = None
    region_scope: list[str] = Field(default_factory=list)
    candidate_material: str | None = None
    incumbent_material: str | None = None
    next_best_competitive_alternative: str | None = None
    incumbent_form: str | None = None
    candidate_form: str | None = None
    conversion_process: str | None = None
    product_type: str | None = None
    buyer_type: str | None = None
    application_requirements: list[str] = Field(default_factory=list)
    substitution_drivers: list[str] = Field(default_factory=list)
    strategic_rationale: str = Field(default="")
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_urls: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    generation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reflection_assessment: ReflectionAssessment | None = None
    round_index: int = Field(default=0, ge=0)
    generation_source: HypothesisGenerationSource = "initial"
    parent_hypothesis_ids: list[str] = Field(default_factory=list)
    ranking_score: float | None = Field(default=None, ge=0.0, le=1.0)
    ranking_rationale: str = Field(default="")
    ranking_round_id: str | None = None
    ranking_status: RankingAction | None = None
    evolution_notes: list[str] = Field(default_factory=list)
    concept_labels: list[str] = Field(default_factory=list)
    concept_cluster_id: str | None = None
    is_active: bool = True
    retired_reason: str | None = None
    superseded_by_hypothesis_id: str | None = None
    merged_from_hypothesis_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "region_scope",
        "application_requirements",
        "substitution_drivers",
        "supporting_chunk_ids",
        "supporting_urls",
        "assumptions",
        "unknowns",
        "parent_hypothesis_ids",
        "evolution_notes",
        "concept_labels",
        "merged_from_hypothesis_ids",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class HypothesisSeed(BaseModel):
    title: str
    summary: str
    application: str | None = None
    market_segment: str | None = None
    candidate_material: str | None = None
    incumbent_material: str | None = None
    next_best_competitive_alternative: str | None = None
    incumbent_form: str | None = None
    candidate_form: str | None = None
    conversion_process: str | None = None
    product_type: str | None = None
    buyer_type: str | None = None
    application_requirements: list[str] = Field(default_factory=list)
    substitution_drivers: list[str] = Field(default_factory=list)
    strategic_rationale: str = Field(default="")
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_urls: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    generation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_seed_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        hypothesis_text = first_present(
            payload,
            "hypothesis_text",
            "hypothesis",
            "idea",
            "opportunity",
            "opportunity_description",
            "description",
        )
        if "title" not in payload or not str(payload.get("title") or "").strip():
            payload["title"] = first_present(payload, "hypothesis_title", "opportunity_title", "name") or short_title_from_text(
                hypothesis_text or payload.get("summary")
            )
        if "summary" not in payload or not str(payload.get("summary") or "").strip():
            payload["summary"] = hypothesis_text or first_present(
                payload,
                "rationale",
                "strategic_rationale",
                "description",
            ) or payload["title"]
        alias_map = {
            "application": ("end_use", "use_case", "product_application"),
            "market_segment": ("segment", "market", "industry_segment"),
            "candidate_material": ("replacement_material", "proposed_material", "alternative_material", "new_material"),
            "incumbent_material": ("target_material", "existing_material", "current_material", "material_to_replace"),
            "next_best_competitive_alternative": ("nbca", "competitive_alternative", "next_best_alternative"),
            "incumbent_form": ("target_form", "existing_material_form"),
            "candidate_form": ("replacement_form", "proposed_material_form"),
            "conversion_process": ("process", "manufacturing_process"),
            "product_type": ("product", "form_factor"),
            "buyer_type": ("customer_type", "buyer", "customer"),
            "application_requirements": ("requirements", "performance_requirements", "key_requirements"),
            "substitution_drivers": ("drivers", "replacement_drivers", "substitution_rationale"),
            "strategic_rationale": ("rationale", "strategic_fit", "why_it_fits"),
            "supporting_chunk_ids": ("citation_chunk_ids", "chunk_ids", "supporting_evidence_chunk_ids"),
            "supporting_urls": ("citation_urls", "urls", "source_urls", "sources"),
            "assumptions": ("key_assumptions",),
            "unknowns": ("evidence_gaps", "gaps", "open_questions"),
            "generation_confidence": ("confidence", "confidence_score"),
        }
        for canonical, aliases in alias_map.items():
            if canonical in payload and payload[canonical] is not None:
                continue
            alias_value = first_present(payload, *aliases)
            if alias_value is not None:
                payload[canonical] = alias_value
        if payload.get("generation_confidence") is None:
            payload["generation_confidence"] = 0.0
        return payload

    @field_validator("generation_confidence", mode="before")
    @classmethod
    def normalize_generation_confidence(cls, value: Any) -> float:
        return normalize_confidence_value(value)

    @field_validator(
        "application_requirements",
        "substitution_drivers",
        "supporting_chunk_ids",
        "supporting_urls",
        "assumptions",
        "unknowns",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class HypothesisGenerationOutput(BaseModel):
    hypotheses: list[HypothesisSeed] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_top_level_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {"hypotheses": value}
        return value

    @field_validator("hypotheses", mode="before")
    @classmethod
    def default_hypotheses(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value


class RankedHypothesis(BaseModel):
    hypothesis_id: str
    rank: int | None = Field(default=None, ge=1)
    score: float = Field(ge=0.0, le=1.0)
    recommended_action: RankingAction = "hold"
    rationale: str = Field(default="")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvement_directions: list[str] = Field(default_factory=list)

    @field_validator("strengths", "weaknesses", "improvement_directions", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class RankingOutput(BaseModel):
    rankings: list[RankedHypothesis] = Field(default_factory=list)
    best_patterns: list[str] = Field(default_factory=list)
    worst_patterns: list[str] = Field(default_factory=list)

    @field_validator("rankings", mode="before")
    @classmethod
    def default_rankings(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @field_validator("best_patterns", "worst_patterns", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class RankingRound(BaseModel):
    ranking_round_id: str
    research_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_index: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    target_final_count: int = Field(ge=1)
    ranked_hypothesis_ids: list[str] = Field(default_factory=list)
    promoted_hypothesis_ids: list[str] = Field(default_factory=list)
    evolved_parent_hypothesis_ids: list[str] = Field(default_factory=list)
    rejected_hypothesis_ids: list[str] = Field(default_factory=list)
    rankings: list[RankedHypothesis] = Field(default_factory=list)
    best_patterns: list[str] = Field(default_factory=list)
    worst_patterns: list[str] = Field(default_factory=list)
    mean_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "ranked_hypothesis_ids",
        "promoted_hypothesis_ids",
        "evolved_parent_hypothesis_ids",
        "rejected_hypothesis_ids",
        "best_patterns",
        "worst_patterns",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class EvolutionHypothesisSeed(HypothesisSeed):
    parent_hypothesis_ids: list[str] = Field(default_factory=list)
    mutation_strategy: str = Field(default="")
    evolution_notes: list[str] = Field(default_factory=list)

    @field_validator("parent_hypothesis_ids", "evolution_notes", mode="before")
    @classmethod
    def default_evolution_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class HypothesisEvolutionOutput(BaseModel):
    hypotheses: list[EvolutionHypothesisSeed] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_top_level_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {"hypotheses": value}
        return value

    @field_validator("hypotheses", mode="before")
    @classmethod
    def default_hypotheses(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value


class ProximityConcept(BaseModel):
    concept_label: str
    description: str = Field(default="")
    member_hypothesis_ids: list[str] = Field(default_factory=list)

    @field_validator("member_hypothesis_ids", mode="before")
    @classmethod
    def default_member_ids(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class SynthesizedHypothesisSeed(HypothesisSeed):
    merged_from_hypothesis_ids: list[str] = Field(default_factory=list)
    concept_label: str | None = None
    synthesis_rationale: str = Field(default="")

    @field_validator("merged_from_hypothesis_ids", mode="before")
    @classmethod
    def default_merged_ids(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class ProximityReviewOutput(BaseModel):
    concepts: list[ProximityConcept] = Field(default_factory=list)
    synthesized_hypotheses: list[SynthesizedHypothesisSeed] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("concepts", "synthesized_hypotheses", mode="before")
    @classmethod
    def default_collection_payloads(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @field_validator("notes", mode="before")
    @classmethod
    def default_notes(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class ProximityRound(BaseModel):
    proximity_round_id: str
    research_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_index: int = Field(ge=0)
    concepts: list[ProximityConcept] = Field(default_factory=list)
    synthesized_hypothesis_ids: list[str] = Field(default_factory=list)
    retired_hypothesis_ids: list[str] = Field(default_factory=list)
    labeled_hypothesis_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator(
        "concepts",
        mode="before",
    )
    @classmethod
    def default_concepts(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @field_validator(
        "synthesized_hypothesis_ids",
        "retired_hypothesis_ids",
        "labeled_hypothesis_ids",
        "notes",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class MetaReviewOutput(BaseModel):
    whitespace_gaps: list[str] = Field(default_factory=list)
    generation_guidance: list[str] = Field(default_factory=list)
    coverage_assessment: str = Field(default="")
    gap_shrinkage_status: GapShrinkageStatus = "unknown"
    coverage_sufficient: bool = False

    @field_validator("whitespace_gaps", "generation_guidance", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class MetaReviewRound(BaseModel):
    meta_review_round_id: str
    research_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_index: int = Field(ge=0)
    whitespace_gaps: list[str] = Field(default_factory=list)
    generation_guidance: list[str] = Field(default_factory=list)
    coverage_assessment: str = Field(default="")
    gap_shrinkage_status: GapShrinkageStatus = "unknown"
    coverage_sufficient: bool = False
    should_continue: bool = True
    stop_reason: str | None = None
    gap_persistence_count: int = Field(default=0, ge=0)

    @field_validator("whitespace_gaps", "generation_guidance", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)


class CoScientistRunResult(BaseModel):
    research_id: str
    generated_hypotheses: int
    reflected_hypotheses: int
    automatic_discovery_runs: int
    research_goal_path: str
    hypothesis_path: str
    report_path: str


class CoScientistLoopResult(BaseModel):
    research_id: str
    rounds_completed: int
    ranked_hypotheses: int
    evolved_hypotheses: int
    regenerated_hypotheses: int
    synthesized_hypotheses: int
    reflected_hypotheses: int
    automatic_discovery_runs: int
    ranking_path: str
    hypothesis_path: str
    report_path: str
    stop_reason: str
