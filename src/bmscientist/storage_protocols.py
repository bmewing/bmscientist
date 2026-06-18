"""Protocol definitions for storage backends.

These protocols define the structural interfaces that storage implementations
must satisfy.  They are *runtime-checkable* so that ``isinstance`` guards
work for lightweight dependency-injection patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Evidence / vector store
# ---------------------------------------------------------------------------


@runtime_checkable
class EvidenceStore(Protocol):
    """Structural interface for a vector-backed evidence store.

    Any object whose public surface matches these signatures is accepted
    wherever an ``EvidenceStore`` is expected — no explicit subclassing
    required.
    """

    def add_chunks(self, records: list) -> int:
        """Persist *records* and return the number actually written."""
        ...

    def search_by_vector(
        self, vector: list[float], top_k: int = 8
    ) -> list[dict[str, Any]]:
        """Return up to *top_k* nearest neighbours for *vector*."""
        ...

    def all_rows(self) -> list[dict[str, Any]]:
        """Return every row in the store as plain dicts."""
        ...


# ---------------------------------------------------------------------------
# Research / co-scientist store
# ---------------------------------------------------------------------------


@runtime_checkable
class ResearchStore(Protocol):
    """Structural interface for the co-scientist research store.

    Covers research-goal CRUD, hypothesis lifecycle, tournament rounds,
    and report persistence.
    """

    # -- research goals -----------------------------------------------------

    def save_research_goal(self, document: Any) -> Path:
        """Persist a research-goal document and return its path."""
        ...

    def load_research_goal(self, research_id: str) -> Any:
        """Load and return the research-goal document for *research_id*."""
        ...

    # -- hypotheses ---------------------------------------------------------

    def save_hypothesis(self, hypothesis: Any) -> Path:
        """Write a hypothesis snapshot and return its path."""
        ...

    def load_hypotheses(
        self,
        research_id: str,
        stages: set[str] | None = None,
        active_only: bool = False,
    ) -> list:
        """Return hypotheses, optionally filtered by *stages* and activity."""
        ...

    def latest_hypotheses(self, research_id: str) -> list:
        """Return the most-recent snapshot of every hypothesis."""
        ...

    def claim_next_generated_hypothesis(
        self,
        research_id: str,
        worker_id: str,
        lease_seconds: int = 1800,
    ) -> Any | None:
        """Atomically claim the next generated hypothesis for reflection."""
        ...

    def complete_reflection_claim(self, hypothesis: Any) -> Path:
        """Mark a claimed hypothesis as fully reflected."""
        ...

    def release_reflection_claim(
        self, hypothesis: Any, error: str | None = None
    ) -> Path:
        """Release a reflection claim, optionally recording an *error*."""
        ...

    # -- reports ------------------------------------------------------------

    def write_report(self, research_id: str, content: str) -> Path:
        """Write (overwrite) the reflection report for *research_id*."""
        ...

    def write_loop_report(self, research_id: str, content: str) -> Path:
        """Write (overwrite) the loop report for *research_id*."""
        ...

    # -- tournament rounds --------------------------------------------------

    def append_ranking_round(self, ranking_round: Any) -> Path:
        """Append a ranking round to the JSONL log."""
        ...

    def load_ranking_rounds(self, research_id: str) -> list:
        """Load all ranking rounds for *research_id*."""
        ...

    def append_proximity_round(self, proximity_round: Any) -> Path:
        """Append a proximity round to the JSONL log."""
        ...

    def load_proximity_rounds(self, research_id: str) -> list:
        """Load all proximity rounds for *research_id*."""
        ...

    def append_meta_review_round(self, meta_review_round: Any) -> Path:
        """Append a meta-review round to the JSONL log."""
        ...

    def load_meta_review_rounds(self, research_id: str) -> list:
        """Load all meta-review rounds for *research_id*."""
        ...

    # -- project management -------------------------------------------------

    def project_exists(self, research_id: str) -> bool:
        """Return ``True`` if a project directory already exists."""
        ...

    def claim_project_name(self, preferred_name: str | None = None) -> str:
        """Reserve and return a unique project name."""
        ...

    # -- feedback -----------------------------------------------------------

    def apply_hypothesis_feedback(
        self,
        research_id: str,
        hypothesis_id: str,
        volume: float | None = None,
        volume_unit: str | None = None,
        status: str | None = None,
        confidence: float | None = None,
        comment: str | None = None,
    ) -> Any | None:
        """Apply human feedback to a hypothesis and propagate to the graph."""
        ...
