from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


EVIDENCE_TYPES = (
    "application currently uses PVC",
    "application requirements",
    "PET/PETG/Tritan capability evidence",
    "regulatory or sustainability pressure",
    "competitor alternative positioning",
    "market or customer need",
)


class SearchQueryPlan(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("queries")
    @classmethod
    def strip_queries(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class SearchResultItem(BaseModel):
    title: str = Field(default="")
    url: HttpUrl
    search_query: str
    snippet: str = Field(default="")
    summary: str = Field(default="")
    exa_id: str | None = None
    request_id: str | None = None
    cost_dollars: float | None = None
    highlights: list[str] = Field(default_factory=list)
    highlight_scores: list[float] = Field(default_factory=list)
    content_text: str = Field(default="")
    content_text_characters: int | None = None
    published_date: str | None = None
    score: float | None = None
    category: str | None = None
    image_url: str | None = None
    favicon_url: str | None = None
    content_source: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PageContent(BaseModel):
    title: str
    url: str
    search_query: str
    source_domain: str
    fetched_at: datetime
    text: str
    status_code: int | None = None
    content_type: str | None = None
    raw_excerpt: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fetched_at", mode="before")
    @classmethod
    def default_timestamp(cls, value: datetime | None) -> datetime:
        return value or datetime.now(timezone.utc)


class EvidenceClassification(BaseModel):
    relevant: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    application: str | None = None
    incumbent_material: str | None = None
    candidate_materials: list[str] = Field(default_factory=list)
    evidence_type: Literal[
        "application currently uses PVC",
        "application requirements",
        "PET/PETG/Tritan capability evidence",
        "regulatory or sustainability pressure",
        "competitor alternative positioning",
        "market or customer need",
    ]
    application_requirements: list[str] = Field(default_factory=list)
    substitution_drivers: list[str] = Field(default_factory=list)
    rationale: str = Field(default="")
    supporting_quotes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceClassificationDraft(BaseModel):
    relevant: bool | None = None
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    application: str | None = None
    incumbent_material: str | None = None
    candidate_materials: list[str] = Field(default_factory=list)
    evidence_type: str | None = None
    application_requirements: list[str] = Field(default_factory=list)
    substitution_drivers: list[str] = Field(default_factory=list)
    rationale: str = Field(default="")
    supporting_quotes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "candidate_materials",
        "application_requirements",
        "substitution_drivers",
        "supporting_quotes",
        mode="before",
    )
    @classmethod
    def default_list_fields(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def default_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        return value


class ChunkRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    run_id: str
    original_query: str
    search_query: str
    source_title: str
    source_url: str
    source_domain: str
    retrieved_at: datetime
    chunk_index: int = Field(ge=0)
    chunk_text: str = Field(min_length=1)
    vector: list[float] = Field(default_factory=list)
    application: str | None = None
    incumbent_material: str | None = None
    candidate_materials: list[str] = Field(default_factory=list)
    evidence_type: str
    application_requirements: list[str] = Field(default_factory=list)
    substitution_drivers: list[str] = Field(default_factory=list)
    relevance_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


GRAPH_ENRICHMENT_EDGE_TYPES = (
    "Product_USED_IN_Application",
    "Market_USES_Product",
    "Market_HAS_APPLICATION_Application",
    "Market_HAS_COMPANY_Company",
    "Company_PRODUCES_Product",
)


class GraphEnrichmentMetric(BaseModel):
    name: Literal["volume", "price", "revenue", "forecast_revenue", "cagr"]
    value: float | None = None
    unit: str | None = None
    currency: str | None = None
    year: int | None = None
    basis: str | None = None


class GraphEnrichmentProposal(BaseModel):
    proposal_id: str | None = None
    edge_type: Literal[
        "Product_USED_IN_Application",
        "Market_USES_Product",
        "Market_HAS_APPLICATION_Application",
        "Market_HAS_COMPANY_Company",
        "Company_PRODUCES_Product",
    ]
    product_name: str | None = None
    product_aliases: list[str] = Field(default_factory=list)
    application_name: str | None = None
    market_name: str | None = None
    company_name: str | None = None
    geography_name: str | None = None
    relationship_role: str | None = None
    critical_to_quality: list[str] = Field(default_factory=list)
    metrics: list[GraphEnrichmentMetric] = Field(default_factory=list)
    source_chunk_id: str
    source_url: str | None = None
    source_title: str | None = None
    supporting_quote: str = Field(default="")
    rationale: str = Field(default="")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_hash: str | None = None

    @field_validator(
        "critical_to_quality",
        "product_aliases",
        "metrics",
        mode="before",
    )
    @classmethod
    def default_graph_list_fields(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        return value


class GraphEnrichmentProposalOutput(BaseModel):
    proposals: list[GraphEnrichmentProposal] = Field(default_factory=list)


class GraphEnrichmentValidation(BaseModel):
    proposal_id: str
    accepted: bool
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="")
    corrected_edge_type: Literal[
        "Product_USED_IN_Application",
        "Market_USES_Product",
        "Market_HAS_APPLICATION_Application",
        "Market_HAS_COMPANY_Company",
        "Company_PRODUCES_Product",
    ] | None = None
    corrected_relationship_role: str | None = None
    corrected_product_aliases: list[str] = Field(default_factory=list)
    corrected_metrics: list[GraphEnrichmentMetric] = Field(default_factory=list)
    corrected_critical_to_quality: list[str] = Field(default_factory=list)

    @field_validator(
        "corrected_metrics",
        "corrected_product_aliases",
        "corrected_critical_to_quality",
        mode="before",
    )
    @classmethod
    def default_validation_list_fields(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        return value


class GraphEnrichmentValidationOutput(BaseModel):
    validations: list[GraphEnrichmentValidation] = Field(default_factory=list)


class MaterialAliasRecord(BaseModel):
    material_alias_id: str | None = None
    alias_text: str
    normalized_alias: str | None = None
    canonical_node_id: str
    canonical_node_label: Literal[
        "MaterialFamily",
        "Product",
        "MaterialGrade",
        "Company",
        "Endpoint",
        "CriticalToQuality",
    ]
    alias_type: Literal[
        "abbreviation",
        "chemical_name",
        "trade_name",
        "synonym",
        "spelling_variant",
        "legacy_name",
        "source_label",
        "unknown",
    ] = "unknown"
    source_vendor: str | None = None
    source_url: str | None = None
    evidence_hash: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    validation_status: Literal["accepted", "pending_review", "rejected", "conflict"] = "accepted"


class AliasResolution(BaseModel):
    status: Literal["exact", "normalized", "fuzzy", "ambiguous", "missing"]
    canonical_node_id: str | None = None
    canonical_node_label: str | None = None
    matched_alias: str | None = None
    matched_by: str | None = None
    candidate_node_ids: list[str] = Field(default_factory=list)


class CTQPropertyRequirement(BaseModel):
    bound: Literal["min", "max", "range", "qualitative"]
    value: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    unit: str | None = None
    condition_text: str | None = None
    basis: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CriticalToQualityRequirement(BaseModel):
    ctq_name: str
    requirement_text: str
    property_requirements: dict[str, CTQPropertyRequirement] = Field(default_factory=dict)
    requirement_role: Literal["must_have", "nice_to_have", "differentiator", "failure_mode"] = "must_have"


class EndpointIndicatorRule(BaseModel):
    endpoint_name: str
    direction: Literal["higher_is_better", "lower_is_better", "within_range", "qualitative"]
    default_threshold_value: float | None = None
    default_threshold_min: float | None = None
    default_threshold_max: float | None = None
    unit: str | None = None
    condition_text: str | None = None
    rationale: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CTQMappingProposal(BaseModel):
    ctq_name: str
    requirement_text: str
    property_requirements: dict[str, CTQPropertyRequirement] = Field(default_factory=dict)
    indicator_rules: list[EndpointIndicatorRule] = Field(default_factory=list)
    review_required: bool = False
    review_reason: str = Field(default="")


class DiscoverySummary(BaseModel):
    run_id: str
    original_query: str
    total_search_queries: int
    total_search_results: int
    unique_urls: int
    fetched_pages: int
    relevant_pages: int
    stored_chunks: int
    graph_enrichment_proposals: int = 0
    graph_enrichment_accepted: int = 0
    opportunity_summary: str
    notable_applications: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    output_path: str | None = None


class OpportunityItem(BaseModel):
    application: str
    evidence_type: str
    incumbent_material: str | None = None
    candidate_materials: list[str] = Field(default_factory=list)
    relevance_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    chunk_id: str
    source_title: str
    source_url: str
    summary: str


class OpportunityReport(BaseModel):
    incumbent_material: str
    candidate_material: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    narrative: str
    items: list[OpportunityItem] = Field(default_factory=list)
