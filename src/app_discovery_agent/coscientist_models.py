from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


HypothesisStatus = Literal["generated", "reflected"]


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

    @field_validator(
        "region_scope",
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

    @field_validator("hypotheses", mode="before")
    @classmethod
    def default_hypotheses(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value


class CoScientistRunResult(BaseModel):
    research_id: str
    generated_hypotheses: int
    reflected_hypotheses: int
    automatic_discovery_runs: int
    research_goal_path: str
    hypothesis_path: str
    report_path: str
