from __future__ import annotations

import logging
from typing import Iterable

from bmscientist.llm import DeepSeekLLM
from bmscientist.models import EVIDENCE_TYPES, EvidenceClassification, EvidenceClassificationDraft, PageContent
from bmscientist.prompt_library import PROMPTS


LOGGER = logging.getLogger(__name__)


class EvidenceClassifier:
    def __init__(self, llm: DeepSeekLLM):
        self._llm = llm

    def heuristic_relevance(self, query: str, text: str) -> float:
        query_terms = {term.lower() for term in query.split() if len(term) > 3}
        if not query_terms or not text:
            return 0.0
        lowered = text.lower()
        hits = sum(1 for term in query_terms if term in lowered)
        density_bonus = min(len(text) / 4000.0, 0.35)
        return min((hits / max(len(query_terms), 1)) * 0.7 + density_bonus, 1.0)

    def classify(self, original_query: str, page: PageContent) -> EvidenceClassification:
        system_prompt = PROMPTS.render("evidence_classifier", "classify.system")
        user_prompt = PROMPTS.render(
            "evidence_classifier",
            "classify.user",
            original_query=original_query,
            allowed_evidence_types=", ".join(EVIDENCE_TYPES),
            source_url=page.url,
            source_title=page.title,
            search_query=page.search_query,
            source_metadata=page.metadata,
            page_text=page.text[:16000],
        )
        draft = self._llm.complete_json(EvidenceClassificationDraft, system_prompt, user_prompt)
        return self._normalize_classification(draft)

    def _normalize_classification(self, draft: EvidenceClassificationDraft) -> EvidenceClassification:
        evidence_type = self._normalize_evidence_type(draft.evidence_type)
        relevant = bool(draft.relevant) if draft.relevant is not None else False
        relevance_score = draft.relevance_score if draft.relevance_score is not None else 0.0
        confidence_score = draft.confidence_score if draft.confidence_score is not None else 0.0

        return EvidenceClassification(
            relevant=relevant,
            relevance_score=relevance_score,
            confidence_score=confidence_score,
            application=draft.application,
            incumbent_material=draft.incumbent_material,
            candidate_materials=[item for item in draft.candidate_materials if item],
            evidence_type=evidence_type,
            application_requirements=[item for item in draft.application_requirements if item],
            substitution_drivers=[item for item in draft.substitution_drivers if item],
            rationale=draft.rationale,
            supporting_quotes=[item for item in draft.supporting_quotes if item],
            metadata=draft.metadata,
        )

    @staticmethod
    def _normalize_evidence_type(value: str | None) -> str:
        if value in EVIDENCE_TYPES:
            return value
        if not value:
            return "market or customer need"

        lowered = value.strip().lower()
        aliases = {
            "application uses pvc": "application currently uses PVC",
            "current pvc use": "application currently uses PVC",
            "application requirements and specs": "application requirements",
            "requirements": "application requirements",
            "capability evidence": "PET/PETG/Tritan capability evidence",
            "pet capability evidence": "PET/PETG/Tritan capability evidence",
            "petg capability evidence": "PET/PETG/Tritan capability evidence",
            "tritan capability evidence": "PET/PETG/Tritan capability evidence",
            "sustainability pressure": "regulatory or sustainability pressure",
            "regulatory pressure": "regulatory or sustainability pressure",
            "competitor positioning": "competitor alternative positioning",
            "customer need": "market or customer need",
            "market need": "market or customer need",
        }
        return aliases.get(lowered, "market or customer need")

    @staticmethod
    def filter_supported(pages: Iterable[PageContent], threshold: float, query: str, classifier: "EvidenceClassifier") -> list[PageContent]:
        kept: list[PageContent] = []
        for page in pages:
            score = classifier.heuristic_relevance(query, page.text)
            LOGGER.info("Heuristic relevance for %s: %.2f", page.url, score)
            if score >= threshold:
                kept.append(page)
        return kept
