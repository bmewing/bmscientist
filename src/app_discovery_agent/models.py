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
    published_date: str | None = None
    score: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PageContent(BaseModel):
    title: str
    url: HttpUrl
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


class ChunkRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    run_id: str
    original_query: str
    search_query: str
    source_title: str
    source_url: HttpUrl
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


class DiscoverySummary(BaseModel):
    run_id: str
    original_query: str
    total_search_queries: int
    total_search_results: int
    unique_urls: int
    fetched_pages: int
    relevant_pages: int
    stored_chunks: int
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
    source_url: HttpUrl
    summary: str


class OpportunityReport(BaseModel):
    incumbent_material: str
    candidate_material: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    narrative: str
    items: list[OpportunityItem] = Field(default_factory=list)

