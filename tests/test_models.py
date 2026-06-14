from datetime import datetime, timezone
import json
from pathlib import Path

from app_discovery_agent.agent import DiscoveryAgent
from app_discovery_agent.classify import EvidenceClassifier
from app_discovery_agent.chunking import TextChunker
from app_discovery_agent.llm import DeepSeekLLM
from app_discovery_agent.models import ChunkRecord, EvidenceClassification, EvidenceClassificationDraft, PageContent, SearchResultItem


def test_evidence_classification_parses_expected_shape():
    classification = EvidenceClassification.model_validate(
        {
            "relevant": True,
            "relevance_score": 0.78,
            "confidence_score": 0.61,
            "application": "blister packaging",
            "incumbent_material": "PVC",
            "candidate_materials": ["PETG", "Eastman Tritan"],
            "evidence_type": "application requirements",
            "application_requirements": ["clarity", "impact resistance"],
            "substitution_drivers": ["recyclability"],
            "rationale": "The page describes clear rigid packaging needs.",
            "supporting_quotes": ["clear rigid film used in packaging"],
            "metadata": {"source_type": "product page"},
        }
    )

    assert classification.relevant is True
    assert classification.candidate_materials == ["PETG", "Eastman Tritan"]


def test_chunk_record_parses_nested_metadata():
    record = ChunkRecord.model_validate(
        {
            "id": "chunk-1",
            "run_id": "run-1",
            "original_query": "PVC alternatives",
            "search_query": "PVC PETG application",
            "source_title": "Example",
            "source_url": "https://example.com/page",
            "source_domain": "example.com",
            "retrieved_at": datetime.now(timezone.utc),
            "chunk_index": 0,
            "chunk_text": "PVC is used for a clear rigid application.",
            "vector": [0.1, 0.2, 0.3],
            "application": "medical packaging",
            "incumbent_material": "PVC",
            "candidate_materials": ["PET"],
            "evidence_type": "application currently uses PVC",
            "application_requirements": ["clarity"],
            "substitution_drivers": ["regulatory pressure"],
            "relevance_score": 0.84,
            "confidence_score": 0.59,
            "metadata": {"quotes": ["PVC tray"]},
        }
    )

    assert record.metadata["quotes"] == ["PVC tray"]


def test_classifier_normalizes_missing_evidence_type():
    classifier = EvidenceClassifier.__new__(EvidenceClassifier)
    normalized = classifier._normalize_classification(
        EvidenceClassificationDraft(
            relevant=True,
            relevance_score=0.7,
            confidence_score=0.5,
            evidence_type=None,
            candidate_materials=["PETG"],
        )
    )

    assert normalized.evidence_type == "market or customer need"
    assert normalized.relevant is True


def test_evidence_classification_draft_allows_null_collections():
    draft = EvidenceClassificationDraft.model_validate(
        {
            "relevant": True,
            "relevance_score": 0.6,
            "confidence_score": 0.4,
            "candidate_materials": None,
            "application_requirements": None,
            "substitution_drivers": None,
            "supporting_quotes": None,
            "metadata": None,
        }
    )

    assert draft.candidate_materials == []
    assert draft.application_requirements == []
    assert draft.substitution_drivers == []
    assert draft.supporting_quotes == []
    assert draft.metadata == {}


def test_partial_page_is_built_from_search_result():
    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    agent._config = type("Config", (), {"min_snippet_characters": 20})()
    result = SearchResultItem(
        title="PVC applications overview",
        url="https://example.com/pvc",
        search_query="pvc applications",
        snippet="PVC is used in pipe, siding, and medical packaging because of durability and clarity.",
        summary="Overview of major applications and requirements.",
    )

    page = agent._build_partial_page_from_search_result(result, "blocked_domain")

    assert page is not None
    assert page.metadata["is_partial_evidence"] is True
    assert page.metadata["partial_evidence_reason"] == "blocked_domain"
    assert "PVC applications overview" in page.text


def test_discovery_retains_low_relevance_fetched_pages_for_reflection():
    class LowRelevanceClassifier:
        def heuristic_relevance(self, query, text):
            return 0.01

        def classify(self, query, page):
            return EvidenceClassification.model_validate(
                {
                    "relevant": False,
                    "relevance_score": 0.05,
                    "confidence_score": 0.2,
                    "application": None,
                    "incumbent_material": None,
                    "candidate_materials": [],
                    "evidence_type": "market or customer need",
                    "application_requirements": [],
                    "substitution_drivers": [],
                    "rationale": "Indirect market context.",
                    "supporting_quotes": [],
                    "metadata": {},
                }
            )

    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    agent._config = type("Config", (), {"min_page_characters": 20, "min_snippet_characters": 20})()
    agent._classifier = LowRelevanceClassifier()
    agent._chunker = TextChunker(chunk_size=80, chunk_overlap=0)
    page = PageContent(
        title="Adjacent market report",
        url="https://example.com/market",
        search_query="application market size",
        source_domain="example.com",
        fetched_at=datetime.now(timezone.utc),
        text="Market revenue, CAGR, application segmentation, and customer demand context for an adjacent market.",
    )

    state = {"fetched_pages": [page], "skipped_pages": [], "original_query": "PETG application evidence"}
    state.update(agent.filter_relevance(state))
    state.update(agent.classify_evidence(state))
    state.update(agent.chunk_content({**state, "run_id": "run-1"}))

    assert len(state["candidate_pages"]) == 1
    assert state["candidate_pages"][0].metadata["heuristic_relevance_score"] == 0.01
    assert len(state["classifications"]) == 1
    assert state["skipped_pages"] == []
    assert state["chunk_records"][0].metadata["retained_for_reflection"] is True
    assert state["chunk_records"][0].metadata["classification_relevant"] is False


def test_load_cached_fetched_pages_skips_pdf(tmp_path):
    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    path = tmp_path / "fetched_pages.json"
    path.write_text(
        json.dumps(
            [
                {
                    "title": "PDF",
                    "url": "https://example.com/file.pdf",
                    "search_query": "pvc",
                    "source_domain": "example.com",
                    "fetched_at": "2026-06-12T17:29:19.538021+00:00",
                    "status_code": 200,
                    "content_type": "application/pdf",
                    "text": "%PDF-1.7 ...",
                    "metadata": {},
                },
                {
                    "title": "HTML",
                    "url": "https://example.com/page",
                    "search_query": "pvc",
                    "source_domain": "example.com",
                    "fetched_at": "2026-06-12T17:29:19.538021+00:00",
                    "status_code": 200,
                    "content_type": "text/html",
                    "text": "PVC is used in rigid pipe applications.",
                    "metadata": {},
                },
            ]
        ),
        encoding="utf-8",
    )

    pages, skipped = agent._load_cached_fetched_pages(path, max_pages=20)

    assert len(pages) == 1
    assert pages[0].title == "HTML"
    assert skipped[0]["reason"] == "unsupported_cached_content_type"


def test_plan_search_queries_handles_embedded_json_example():
    class FakeLLM:
        def complete_json(self, response_model, system_prompt, user_prompt):
            assert '"queries"' in user_prompt
            return response_model.model_validate({"queries": ["pvc applications rigid pipe requirements"]})

    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    agent._llm = FakeLLM()

    result = agent.plan_search_queries(
        {
            "original_query": "major applications of PVC material and key performance requirements",
            "max_search_queries": 8,
        }
    )

    assert result["search_queries"][0] == "major applications of PVC material and key performance requirements"


def test_llm_json_coercion_handles_fenced_json_and_trailing_commas():
    content = """
Here is the ranking:
```json
{
  "rankings": [
    {"hypothesis_id": "h1", "score": 0.8,}
  ],
}
```
"""

    payload = DeepSeekLLM._coerce_json(content)

    assert payload["rankings"][0]["hypothesis_id"] == "h1"


def test_llm_json_coercion_uses_first_balanced_object():
    content = '{"ok": true}\n\nExtra explanation with another object: {"ignored": true}'

    payload = DeepSeekLLM._coerce_json(content)

    assert payload == {"ok": True}
