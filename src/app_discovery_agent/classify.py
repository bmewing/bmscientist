from __future__ import annotations

import logging
from typing import Iterable

from app_discovery_agent.llm import DeepSeekLLM
from app_discovery_agent.models import EVIDENCE_TYPES, EvidenceClassification, PageContent


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
        system_prompt = (
            "You are a conservative technical research analyst. "
            "Classify evidence about application-development opportunities. "
            "Never overstate suitability. If evidence is partial, preserve it with modest confidence. "
            "Return strict JSON only."
        )
        user_prompt = f"""
Original query:
{original_query}

Allowed evidence_type values:
{", ".join(EVIDENCE_TYPES)}

Source URL: {page.url}
Source title: {page.title}
Search query: {page.search_query}

Text:
\"\"\"
{page.text[:16000]}
\"\"\"

Return JSON with:
- relevant (boolean)
- relevance_score (0 to 1)
- confidence_score (0 to 1)
- application (string or null)
- incumbent_material (string or null)
- candidate_materials (array of strings)
- evidence_type (one allowed value)
- application_requirements (array of strings)
- substitution_drivers (array of strings)
- rationale (short string)
- supporting_quotes (array of short excerpts copied from the page)
- metadata (object)

Rules:
- Do not claim PET, PETG, or Tritan is suitable unless the text supports that.
- If the page mentions PVC use or requirements without proving substitution, that can still be relevant.
- Keep confidence conservative.
- Prefer null over guessing.
"""
        return self._llm.complete_json(EvidenceClassification, system_prompt, user_prompt)

    @staticmethod
    def filter_supported(pages: Iterable[PageContent], threshold: float, query: str, classifier: "EvidenceClassifier") -> list[PageContent]:
        kept: list[PageContent] = []
        for page in pages:
            score = classifier.heuristic_relevance(query, page.text)
            LOGGER.info("Heuristic relevance for %s: %.2f", page.url, score)
            if score >= threshold:
                kept.append(page)
        return kept

