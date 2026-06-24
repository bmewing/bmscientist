from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


HypothesisStatus = Literal["generated", "reflecting", "reflected", "evolve", "retired"]
HypothesisGenerationSource = Literal["initial", "evolved", "regenerated", "synthesized"]
RankingAction = Literal["advance", "hold", "evolve", "reject"]
GapShrinkageStatus = Literal["improved", "stable", "worse", "unknown"]
ProximityMergeMode = Literal["conservative", "balanced", "aggressive"]
ProximityGranularity = Literal["device_subtype", "application_family", "global"]
ResearchMode = Literal[
    "materials_opportunity",
    "candidate_design",
    "formulation_design",
    "process_design",
    "literature_map",
    "generic_screening",
]
CriterionDirection = Literal["maximize", "minimize", "target", "avoid", "classify", "describe"]
CriterionEvidenceMode = Literal["literature", "local_tool", "external_tool", "human_review", "mixed"]
ToolRequestStatus = Literal["requested", "available", "blocked", "deferred"]
EVALUATION_RESULT_FIELD_NAMES = {
    "criterion_name",
    "criterion",
    "name",
    "criterion_id",
    "metric",
    "evaluation_criterion",
    "value",
    "unit",
    "normalized_score",
    "score",
    "criterion_score",
    "normalized",
    "normalized_value",
    "confidence",
    "rationale",
    "reason",
    "reasoning",
    "justification",
    "notes",
    "description",
    "explanation",
    "evidence_mode",
    "tool_id",
    "tool",
    "tool_name",
    "citation_chunk_ids",
    "chunk_ids",
    "supporting_chunk_ids",
    "evidence_chunk_ids",
    "citation_urls",
    "urls",
    "supporting_urls",
    "source_urls",
    "sources",
    "is_inferred",
    "inferred",
}


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


def coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return [value]
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, BaseModel):
            normalized.append(item.model_dump())
        elif isinstance(item, dict):
            normalized.append(item)
    return normalized


def normalize_evaluation_result_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        return {}
    payload = dict(value)
    if "criterion_name" not in payload or not str(payload.get("criterion_name") or "").strip():
        criterion_name = first_present(payload, "criterion", "name", "criterion_id", "metric", "evaluation_criterion")
        if criterion_name is not None:
            payload["criterion_name"] = criterion_name
    if ("criterion_name" not in payload or not str(payload.get("criterion_name") or "").strip()) and len(payload) == 1:
        key, raw_result = next(iter(payload.items()))
        nested = normalize_evaluation_result_payload(raw_result)
        if nested:
            nested["criterion_name"] = nested.get("criterion_name") or str(key).strip()
            return nested
        result = {"criterion_name": str(key).strip()}
        if isinstance(raw_result, str):
            result["rationale"] = raw_result.strip()
        elif raw_result is not None:
            result["value"] = raw_result
        return result
    if payload.get("normalized_score") is None:
        normalized_score = first_present(payload, "score", "criterion_score", "normalized", "normalized_value")
        if normalized_score is not None:
            payload["normalized_score"] = normalized_score
    if not str(payload.get("rationale") or "").strip():
        rationale = first_present(payload, "reason", "reasoning", "justification", "notes", "description", "explanation")
        if rationale is not None:
            payload["rationale"] = rationale
    if payload.get("tool_id") is None:
        tool_id = first_present(payload, "tool", "tool_name")
        if tool_id is not None:
            payload["tool_id"] = tool_id
    if payload.get("citation_chunk_ids") is None:
        chunk_ids = first_present(payload, "chunk_ids", "supporting_chunk_ids", "evidence_chunk_ids")
        if chunk_ids is not None:
            payload["citation_chunk_ids"] = chunk_ids
    if payload.get("citation_urls") is None:
        citation_urls = first_present(payload, "urls", "supporting_urls", "source_urls", "sources")
        if citation_urls is not None:
            payload["citation_urls"] = citation_urls
    if payload.get("is_inferred") is None:
        is_inferred = first_present(payload, "inferred")
        if is_inferred is not None:
            payload["is_inferred"] = is_inferred
    return payload


def coerce_evaluation_results(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, BaseModel):
        value = value.model_dump()
    
    # If LLM returned a dictionary mapping criterion_name -> result details
    if isinstance(value, dict):
        payload_keys = {str(key).strip().lower() for key in value.keys()}
        is_single_result_payload = bool(payload_keys & EVALUATION_RESULT_FIELD_NAMES)
        if not is_single_result_payload:
            results = []
            for k, v in value.items():
                normalized_result = normalize_evaluation_result_payload({k: v})
                if normalized_result:
                    results.append(normalized_result)
            return results
        else:
            normalized = normalize_evaluation_result_payload(value)
            return [normalized] if normalized else []
            
    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            if isinstance(item, BaseModel):
                item = item.model_dump()
            if isinstance(item, dict):
                normalized.extend(coerce_evaluation_results(item))
        return normalized
        
    return []



def coerce_string_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return {str(key): raw_value for key, raw_value in value.items()}
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


def coerce_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        for key in ("value", "text", "rationale", "summary", "description"):
            nested = value.get(key)
            if nested is not None:
                return coerce_text_value(nested)
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [coerce_text_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


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


def normalize_score_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 1.0:
            return max(0.0, min(1.0, numeric))
        if numeric <= 5.0:
            return max(0.0, min(1.0, numeric / 5.0))
        if numeric <= 10.0:
            return max(0.0, min(1.0, numeric / 10.0))
        if numeric <= 100.0:
            return max(0.0, min(1.0, numeric / 100.0))
        return 1.0

    text = str(value).strip().lower()
    if not text:
        return None
    if text.endswith("%"):
        try:
            return max(0.0, min(1.0, float(text[:-1].strip()) / 100.0))
        except ValueError:
            return None
    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)
        try:
            numerator = float(numerator_text.strip())
            denominator = float(denominator_text.strip())
        except ValueError:
            numerator = denominator = 0.0
        if denominator > 0:
            return max(0.0, min(1.0, numerator / denominator))

    named_levels = {
        "very low": 0.1,
        "low": 0.25,
        "medium-low": 0.4,
        "medium": 0.55,
        "moderate": 0.55,
        "medium-high": 0.7,
        "high": 0.85,
        "very high": 0.95,
    }
    if text in named_levels:
        return named_levels[text]
    try:
        return normalize_score_value(float(text))
    except ValueError:
        return None


def normalize_evidence_mode(value: Any) -> str:
    if value is None:
        return "mixed"
    text = str(value).strip().lower()
    if not text:
        return "mixed"
    alias_map = {
        "literature": "literature",
        "paper": "literature",
        "papers": "literature",
        "publication": "literature",
        "publications": "literature",
        "experimental": "literature",
        "experimental_literature": "literature",
        "human_review": "human_review",
        "human": "human_review",
        "expert_review": "human_review",
        "manual_review": "human_review",
        "local_tool": "local_tool",
        "tool": "local_tool",
        "local_model": "local_tool",
        "descriptor_model": "local_tool",
        "external_tool": "external_tool",
        "external_model": "external_tool",
        "web_tool": "external_tool",
        "api": "external_tool",
        "service": "external_tool",
        "mixed": "mixed",
        "hybrid": "mixed",
        "combined": "mixed",
        "computational_prediction": "external_tool",
        "computational": "external_tool",
        "prediction": "external_tool",
        "predictive_model": "external_tool",
        "predictive": "external_tool",
        "qsar": "external_tool",
        "in_silico": "external_tool",
        "computational_retrosynthesis": "external_tool",
        "retrosynthesis": "external_tool",
        "computational_and_experimental": "mixed",
        "experimental_and_computational": "mixed",
    }
    if text in alias_map:
        return alias_map[text]
    if "comput" in text and "experiment" in text:
        return "mixed"
    # Check substrings as fallback
    if "literature" in text or "paper" in text or "report" in text or "publication" in text or "article" in text:
        return "literature"
    if "human" in text or "expert" in text or "manual" in text or "review" in text:
        return "human_review"
    if "external" in text or "web" in text or "api" in text or "lca" in text or "computation" in text or "predict" in text:
        return "external_tool"
    if "local" in text or "tool" in text or "model" in text:
        return "local_tool"
    if "mixed" in text or "hybrid" in text or "combine" in text:
        return "mixed"
    return "mixed"


def normalize_direction(value: Any) -> str:
    if value is None:
        return "describe"
    text = str(value).strip().lower()
    if not text:
        return "describe"
    
    # Common mappings/aliases
    if "higher" in text or "maximize" in text or "increase" in text or "greater" in text or "max" in text:
        return "maximize"
    if "lower" in text or "minimize" in text or "decrease" in text or "less" in text or "reduce" in text or "min" in text:
        return "minimize"
    if "target" in text or "equal" in text or "meet" in text or "exceed" in text:
        return "target"
    if "avoid" in text or "exclude" in text or "prevent" in text:
        return "avoid"
    if "classify" in text or "category" in text or "type" in text:
        return "classify"
        
    valid = {"maximize", "minimize", "target", "avoid", "classify", "describe"}
    if text in valid:
        return text
    return "describe"


def normalize_tool_request_status(value: Any) -> str:
    if value is None:
        return "requested"
    text = str(value).strip().lower()
    if not text:
        return "requested"
    alias_map = {
        "to_be_developed": "requested",
        "needs_development": "requested",
        "requires_development": "requested",
        "needs_building": "requested",
        "planned": "requested",
        "not_yet_available": "requested",
        "unavailable": "blocked",
        "not_available": "blocked",
    }
    if text in alias_map:
        return alias_map[text]
    if "proposed" in text or "request" in text or "pending" in text:
        return "requested"
    if "available" in text or "ready" in text or "active" in text:
        return "available"
    if "blocked" in text or "prevent" in text:
        return "blocked"
    if "develop" in text or "build" in text or "implement" in text:
        return "requested"
    if "deferred" in text or "postpone" in text or "later" in text:
        return "deferred"
        
    valid = {"requested", "available", "blocked", "deferred"}
    if text in valid:
        return text
    return "requested"



class ReflectionSearchLimits(BaseModel):
    max_reflection_searches_per_hypothesis: int = Field(default=3, ge=1, le=10)
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_pages_per_search: int = Field(default=8, ge=1, le=50)


class CandidateArtifactSchema(BaseModel):
    artifact_type: str = "material_opportunity"
    primary_identifier_field: str = "candidate_material"
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("required_fields", "optional_fields", "validation_rules", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("examples", mode="before")
    @classmethod
    def default_examples(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_dict_list(value)


class EvaluationCriterion(BaseModel):
    name: str
    description: str = ""
    direction: CriterionDirection = "describe"
    target_value: str | None = None
    weight: float = Field(default=1.0, ge=0.0)
    required_candidate_fields: list[str] = Field(default_factory=list)
    evidence_mode: CriterionEvidenceMode = "mixed"
    suggested_search_queries: list[str] = Field(default_factory=list)
    suggested_tool_ids: list[str] = Field(default_factory=list)
    reflection_guidance: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)

    @field_validator(
        "required_candidate_fields",
        "suggested_search_queries",
        "suggested_tool_ids",
        "reflection_guidance",
        "failure_modes",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("evidence_mode", mode="before")
    @classmethod
    def normalize_evidence_mode_field(cls, value: Any) -> str:
        return normalize_evidence_mode(value)

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction_field(cls, value: Any) -> str:
        return normalize_direction(value)


class ToolRequest(BaseModel):
    tool_id: str
    purpose: str
    status: ToolRequestStatus = "requested"

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status_field(cls, value: Any) -> str:
        return normalize_tool_request_status(value)
    candidate_packages: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    installation_notes: list[str] = Field(default_factory=list)
    execution_notes: list[str] = Field(default_factory=list)
    validation_examples: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator(
        "candidate_packages",
        "required_inputs",
        "expected_outputs",
        "installation_notes",
        "execution_notes",
        "limitations",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("validation_examples", mode="before")
    @classmethod
    def default_examples(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_dict_list(value)


class ProximityMergePolicy(BaseModel):
    merge_mode: ProximityMergeMode = "balanced"
    granularity: ProximityGranularity = "application_family"

    @field_validator("merge_mode", mode="before")
    @classmethod
    def normalize_merge_mode(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "balanced"
        alias_map = {
            "strict": "conservative",
            "merged": "balanced",
            "normal": "balanced",
            "broad": "aggressive",
        }
        return alias_map.get(text, text)

    @field_validator("granularity", mode="before")
    @classmethod
    def normalize_granularity(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "application_family"
        alias_map = {
            "device": "device_subtype",
            "subtype": "device_subtype",
            "family": "application_family",
            "application": "application_family",
            "broad": "global",
        }
        return alias_map.get(text, text)


class CandidateEvaluationResult(BaseModel):
    criterion_name: str
    value: str | float | bool | None = None
    unit: str | None = None
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence_mode: CriterionEvidenceMode = "mixed"
    tool_id: str | None = None
    citation_chunk_ids: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    is_inferred: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        return normalize_evaluation_result_payload(value)

    @field_validator("citation_chunk_ids", "citation_urls", mode="before")
    @classmethod
    def default_list_fields(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("normalized_score", mode="before")
    @classmethod
    def normalize_score_field(cls, value: Any) -> float | None:
        return normalize_score_value(value)

    @field_validator("evidence_mode", mode="before")
    @classmethod
    def normalize_evidence_mode_field(cls, value: Any) -> str:
        return normalize_evidence_mode(value)


class ResearchGoalDocument(BaseModel):
    research_id: str
    project_title: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_goal: str
    original_goal: str | None = None
    goal_changelog: list[dict[str, Any]] = Field(default_factory=list)
    target_hypotheses_final: int = Field(ge=1)
    target_hypotheses_generated: int = Field(ge=1)
    research_mode: ResearchMode = "materials_opportunity"
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
    candidate_artifact_schema: CandidateArtifactSchema = Field(default_factory=CandidateArtifactSchema)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)
    reflection_guidance: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    search_strategy_notes: list[str] = Field(default_factory=list)
    strategic_fit_notes: str | None = None
    proximity_merge_policy: ProximityMergePolicy = Field(default_factory=ProximityMergePolicy)
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
        "reflection_guidance",
        "search_strategy_notes",
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

    @field_validator("candidate_artifact_schema", mode="before")
    @classmethod
    def default_candidate_artifact_schema(cls, value: Any) -> dict[str, Any]:
        return coerce_metric_payload(value)

    @field_validator("proximity_merge_policy", mode="before")
    @classmethod
    def default_proximity_merge_policy(cls, value: Any) -> dict[str, Any]:
        return coerce_metric_payload(value)

    @field_validator("evaluation_criteria", "tool_requests", mode="before")
    @classmethod
    def default_collection_payloads(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_dict_list(value)


class ResearchPlanDraft(BaseModel):
    research_mode: ResearchMode = "materials_opportunity"
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
    candidate_artifact_schema: CandidateArtifactSchema = Field(default_factory=CandidateArtifactSchema)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)
    reflection_guidance: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    search_strategy_notes: list[str] = Field(default_factory=list)

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
        "reflection_guidance",
        "search_strategy_notes",
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

    @field_validator("candidate_artifact_schema", mode="before")
    @classmethod
    def default_candidate_artifact_schema(cls, value: Any) -> dict[str, Any]:
        return coerce_metric_payload(value)

    @field_validator("evaluation_criteria", "tool_requests", mode="before")
    @classmethod
    def default_collection_payloads(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_dict_list(value)


class UpdatedResearchPlan(BaseModel):
    raw_goal: str
    research_mode: ResearchMode = "materials_opportunity"
    regions: list[str] = Field(default_factory=list)
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
    candidate_artifact_schema: CandidateArtifactSchema = Field(default_factory=CandidateArtifactSchema)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)
    reflection_guidance: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    search_strategy_notes: list[str] = Field(default_factory=list)
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
        "reflection_guidance",
        "search_strategy_notes",
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

    @field_validator("candidate_artifact_schema", mode="before")
    @classmethod
    def default_candidate_artifact_schema(cls, value: Any) -> dict[str, Any]:
        return coerce_metric_payload(value)

    @field_validator("evaluation_criteria", "tool_requests", mode="before")
    @classmethod
    def default_collection_payloads(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_dict_list(value)


class EvidenceCitation(BaseModel):
    chunk_id: str
    source_url: str
    source_title: str
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieved_at: str | None = None


class MarketMaterialVolumeEstimate(BaseModel):
    material_name: str
    volume_value: float | None = Field(default=None, ge=0.0)
    volume_unit: str = "metric_tons_per_year"
    share_of_total: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("volume_unit", mode="before")
    @classmethod
    def default_volume_unit(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "metric_tons_per_year"


class MarketVolumeEstimateOutput(BaseModel):
    market_name: str | None = None
    application_name: str | None = None
    region_scope: list[str] = Field(default_factory=list)
    total_substrate_volume_value: float | None = Field(default=None, ge=0.0)
    total_substrate_volume_unit: str = "metric_tons_per_year"
    volume_year: int | None = None
    revenue_value: float | None = Field(default=None, ge=0.0)
    revenue_unit: str | None = None
    revenue_year: int | None = None
    assumed_average_price_value: float | None = Field(default=None, ge=0.0)
    assumed_average_price_unit: str | None = None
    material_volumes: list[MarketMaterialVolumeEstimate] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    source_citations: list[EvidenceCitation] = Field(default_factory=list)

    @field_validator("region_scope", mode="before")
    @classmethod
    def default_region_scope(cls, value: Any) -> list[str]:
        return coerce_string_list(value)

    @field_validator("total_substrate_volume_unit", mode="before")
    @classmethod
    def default_total_volume_unit(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "metric_tons_per_year"

    @field_validator("source_citations", mode="before")
    @classmethod
    def default_source_citations(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_dict_list(value)


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
    criterion_results: list[CandidateEvaluationResult] = Field(default_factory=list)
    tool_request_notes: list[str] = Field(default_factory=list)
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

    @field_validator("criterion_results", mode="before")
    @classmethod
    def default_criterion_results(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_evaluation_results(value)

    @field_validator(
        "tool_request_notes",
        "reflection_search_queries",
        "reflection_discovery_run_ids",
        "evidence_gap_notes",
        mode="before",
    )
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
    candidate_artifact: dict[str, Any] = Field(default_factory=dict)
    evaluation_results: list[CandidateEvaluationResult] = Field(default_factory=list)
    generation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reflection_assessment: ReflectionAssessment | None = None
    reflection_worker_id: str | None = None
    reflection_claimed_at: datetime | None = None
    reflection_lease_expires_at: datetime | None = None
    reflection_attempt_count: int = Field(default=0, ge=0)
    reflection_error: str | None = None
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
    user_feedback_status: Literal["accepted", "rejected", "edited", None] = None
    user_feedback_comment: str | None = None

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

    @field_validator("candidate_artifact", mode="before")
    @classmethod
    def default_candidate_artifact(cls, value: Any) -> dict[str, Any]:
        return coerce_string_dict(value)

    @field_validator("evaluation_results", mode="before")
    @classmethod
    def default_evaluation_results(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_evaluation_results(value)

    @field_validator("strategic_rationale", mode="before")
    @classmethod
    def normalize_hypothesis_strategic_rationale(cls, value: Any) -> str:
        return coerce_text_value(value)


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
    candidate_artifact: dict[str, Any] = Field(default_factory=dict)
    evaluation_results: list[CandidateEvaluationResult] = Field(default_factory=list)
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
            "candidate_artifact": ("artifact", "candidate", "structure", "molecule"),
            "evaluation_results": ("criterion_results", "preliminary_evaluation", "estimated_metrics"),
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

    @field_validator("candidate_artifact", mode="before")
    @classmethod
    def default_candidate_artifact(cls, value: Any) -> dict[str, Any]:
        return coerce_string_dict(value)

    @field_validator("evaluation_results", mode="before")
    @classmethod
    def default_evaluation_results(cls, value: Any) -> list[dict[str, Any]]:
        return coerce_evaluation_results(value)

    @field_validator("strategic_rationale", mode="before")
    @classmethod
    def normalize_seed_strategic_rationale(cls, value: Any) -> str:
        return coerce_text_value(value)


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
    cost_path: str | None = None


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
    cost_path: str | None = None
