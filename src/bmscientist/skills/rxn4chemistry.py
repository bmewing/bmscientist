from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec


LOGGER = logging.getLogger(__name__)

RXN4CHEMISTRY_TOOL_ID = "rxn4chemistry_retrosynthesis"
RXN4CHEMISTRY_TOOL_ALIASES = {
    RXN4CHEMISTRY_TOOL_ID,
    "rxn4chemistry",
    "ibm_rxn",
    "ibm_rxn_retrosynthesis",
    "retrosynthesis",
}
DEFAULT_RXN_BASE_URL = "https://rxn.res.ibm.com"


class RXN4ChemistryConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _RetrosynthesisSummary:
    prediction_id: str
    status: str
    route_count: int
    best_route_depth: int | None
    max_route_depth: int | None


class RXN4ChemistryRetrosynthesisSkill:
    def __init__(
        self,
        config: Any | None = None,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        project_name: str | None = None,
        base_url: str | None = None,
        ai_model: str | None = None,
        max_wait_seconds: float | None = None,
        poll_interval_seconds: float = 2.0,
        cache_dir: Path | None = None,
        wrapper_factory: Callable[..., Any] | None = None,
    ):
        self._api_key = (api_key or os.getenv("RXN4CHEMISTRY_API_KEY") or "").strip()
        self._project_id = (project_id or os.getenv("RXN4CHEMISTRY_PROJECT_ID") or "").strip() or None
        self._project_name = (project_name or os.getenv("RXN4CHEMISTRY_PROJECT_NAME") or "").strip() or None
        configured_base_url = base_url or os.getenv("RXN4CHEMISTRY_BASE_URL") or DEFAULT_RXN_BASE_URL
        self._base_url = configured_base_url.rstrip("/")
        self._ai_model = (ai_model or os.getenv("RXN4CHEMISTRY_RETROSYNTHESIS_MODEL") or "").strip() or None
        default_wait = max(90.0, float(getattr(config, "request_timeout_seconds", 60) or 60) * 2.0)
        self._max_wait_seconds = max_wait_seconds or default_wait
        self._poll_interval_seconds = max(0.1, poll_interval_seconds)
        self._wrapper_factory = wrapper_factory
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif config is not None and getattr(config, "data_dir", None) is not None:
            self._cache_dir = Path(config.data_dir) / "skills" / "rxn4chemistry"
        else:
            self._cache_dir = Path("data") / "skills" / "rxn4chemistry"
        self._spec = SkillSpec(
            skill_id=RXN4CHEMISTRY_TOOL_ID,
            description=(
                "Evaluate retrosynthetic accessibility for SMILES candidates using RXN for Chemistry. "
                "Use when reflection needs synthesis-feasibility, route-complexity, or makeability signals."
            ),
            phases=("reflection",),
            aliases=tuple(sorted(RXN4CHEMISTRY_TOOL_ALIASES - {RXN4CHEMISTRY_TOOL_ID})),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            required_candidate_fields=("smiles",),
            expected_outputs=(
                "synthesis_feasibility",
                "retrosynthesis_route_count",
                "retrosynthesis_best_route_depth",
                "retrosynthesis_max_route_depth",
            ),
            trigger_keywords=(
                "synthesis",
                "synthetic",
                "retrosynthesis",
                "makeability",
                "route",
                "precursor",
            ),
            provider="python_package",
            priority=90,
            requires_safety_review=True,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    @property
    def tool_id(self) -> str:
        return RXN4CHEMISTRY_TOOL_ID

    def is_applicable(self, context: SkillContext) -> bool:
        if context.hypothesis is None:
            return False
        candidate_artifact = getattr(context.hypothesis, "candidate_artifact", {}) or {}
        return bool(str(candidate_artifact.get("smiles") or candidate_artifact.get("canonical_smiles") or "").strip())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False

        requested = {
            str(item).strip().lower()
            for item in (*context.requested_skill_ids, *(request.tool_id for request in context.document.tool_requests))
            if str(item).strip()
        }
        if requested & RXN4CHEMISTRY_TOOL_ALIASES:
            return True

        document = context.document
        keyword_text = " ".join(
            [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
            + [criterion.evidence_mode for criterion in document.evaluation_criteria]
            + document.reflection_guidance
            + [context.purpose]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        hypothesis = context.hypothesis
        candidate_artifact = getattr(hypothesis, "candidate_artifact", {}) or {}
        smiles = str(candidate_artifact.get("smiles") or candidate_artifact.get("canonical_smiles") or "").strip()
        if not smiles:
            return SkillRunResult(
                skill_id=self.tool_id,
                status="skipped",
                notes=["No SMILES string was available on the candidate artifact."],
                rationale="RXN retrosynthesis requires a SMILES string.",
            )

        candidate_name = (
            str(
                candidate_artifact.get("name_or_label")
                or candidate_artifact.get("trade_name")
                or candidate_artifact.get("name")
                or getattr(hypothesis, "candidate_material", None)
                or getattr(hypothesis, "title", "")
            ).strip()
            or getattr(hypothesis, "title", "Unknown candidate")
        )
        try:
            summary, results = self.predict_smiles(smiles)
        except RXN4ChemistryConfigurationError as exc:
            return SkillRunResult(
                skill_id=self.tool_id,
                status="blocked",
                notes=[str(exc)],
                rationale="RXN retrosynthesis could not run because its hosted-service configuration is incomplete.",
            )
        except TimeoutError as exc:
            return SkillRunResult(
                skill_id=self.tool_id,
                status="failed",
                notes=[str(exc)],
                rationale="RXN retrosynthesis did not finish before the configured timeout.",
            )

        return SkillRunResult(
            skill_id=self.tool_id,
            status="completed",
            criterion_results=results,
            evidence_rows=self.build_evidence_rows(
                smiles=smiles,
                candidate_name=candidate_name,
                application=getattr(hypothesis, "application", None),
                incumbent_material=getattr(hypothesis, "incumbent_material", None),
            ),
            notes=[
                (
                    f"RXN found {summary.route_count} retrosynthetic route(s) for `{candidate_name}`"
                    f" with best route depth {summary.best_route_depth or 'unknown'}."
                )
            ],
            rationale=(
                "Executed hosted RXN for Chemistry retrosynthesis against the candidate SMILES and normalized "
                "route-count and route-complexity signals."
            ),
        )

    def predict_smiles(self, smiles: str) -> tuple[_RetrosynthesisSummary, list[CandidateEvaluationResult]]:
        normalized_smiles = str(smiles or "").strip()
        if not normalized_smiles:
            raise RXN4ChemistryConfigurationError("No SMILES string was supplied.")

        cached = self._load_cached_prediction(normalized_smiles)
        if cached is not None:
            return cached

        wrapper = self._build_wrapper()
        prediction_id = self._submit_prediction(wrapper, normalized_smiles)
        results_payload = self._poll_for_results(wrapper, prediction_id)
        summary = self._summarize_results(prediction_id, results_payload)
        results = self._build_results(normalized_smiles, summary)
        self._store_cached_prediction(normalized_smiles, summary, results, results_payload)
        return summary, results

    def build_evidence_rows(
        self,
        *,
        smiles: str,
        candidate_name: str,
        application: str | None = None,
        incumbent_material: str | None = None,
    ) -> list[dict[str, Any]]:
        summary, results = self.predict_smiles(smiles)
        if not results:
            return []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []
        for result in results:
            value_text = self._render_result_value(result)
            rows.append(
                {
                    "id": f"rxn4chemistry:{self._cache_key(smiles)}:{result.criterion_name}",
                    "source_url": self._base_url,
                    "source_title": "RXN4Chemistry retrosynthesis prediction",
                    "application": application,
                    "incumbent_material": incumbent_material,
                    "candidate_materials": [candidate_name] if candidate_name else [],
                    "relevance_score": 0.8,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"RXN retrosynthesis predicted {result.criterion_name} for SMILES {smiles}: {value_text}. "
                        f"Prediction {summary.prediction_id} returned {summary.route_count} route(s) with "
                        f"best route depth {summary.best_route_depth or 'unknown'}."
                    )[:1800],
                    "metadata": {
                        "source_type": "external-tool",
                        "tool_id": self.tool_id,
                        "smiles": smiles,
                        "prediction_id": summary.prediction_id,
                        "prediction_status": summary.status,
                        "route_count": summary.route_count,
                        "best_route_depth": summary.best_route_depth,
                        "max_route_depth": summary.max_route_depth,
                        "endpoint_name": result.criterion_name,
                        "value": result.value,
                        "unit": result.unit,
                        "is_inferred": result.is_inferred,
                    },
                }
            )
        return rows

    def _build_wrapper(self) -> Any:
        if not self._api_key:
            raise RXN4ChemistryConfigurationError("Set RXN4CHEMISTRY_API_KEY to enable the RXN retrosynthesis skill.")

        wrapper_factory = self._wrapper_factory
        if wrapper_factory is None:
            try:
                from rxn4chemistry import RXN4ChemistryWrapper
            except ImportError as exc:
                raise RXN4ChemistryConfigurationError(
                    "Install RXN4Chemistry to enable the RXN retrosynthesis skill."
                ) from exc
            wrapper_factory = RXN4ChemistryWrapper

        wrapper = wrapper_factory(api_key=self._api_key, project_id=self._project_id, base_url=self._base_url)
        project_id = self._ensure_project(wrapper)
        if not project_id:
            raise RXN4ChemistryConfigurationError(
                "Set RXN4CHEMISTRY_PROJECT_ID or RXN4CHEMISTRY_PROJECT_NAME to enable RXN retrosynthesis."
            )
        return wrapper

    def _ensure_project(self, wrapper: Any) -> str | None:
        existing_project_id = str(getattr(wrapper, "project_id", "") or "").strip()
        if existing_project_id:
            return existing_project_id
        if not self._project_name:
            return None

        project_id = self._find_project_id_by_name(wrapper, self._project_name)
        if project_id:
            self._set_project(wrapper, project_id)
            return project_id

        created = wrapper.create_project(self._project_name)
        created_id = self._extract_project_id(created) or str(getattr(wrapper, "project_id", "") or "").strip() or None
        if created_id:
            self._set_project(wrapper, created_id)
        return created_id

    @staticmethod
    def _set_project(wrapper: Any, project_id: str) -> None:
        setter = getattr(wrapper, "set_project", None)
        if callable(setter):
            setter(project_id)
        else:
            wrapper.project_id = project_id

    def _find_project_id_by_name(self, wrapper: Any, project_name: str) -> str | None:
        try:
            payload = wrapper.list_all_projects()
        except Exception:
            LOGGER.exception("Failed listing RXN projects while resolving %s", project_name)
            return None
        normalized_target = project_name.strip().lower()
        for item in self._iter_named_payloads(payload):
            name = str(item.get("name") or item.get("title") or "").strip()
            project_id = str(item.get("id") or item.get("project_id") or "").strip()
            if name.lower() == normalized_target and project_id:
                return project_id
        return None

    def _submit_prediction(self, wrapper: Any, smiles: str) -> str:
        kwargs: dict[str, Any] = {}
        if self._ai_model:
            kwargs["ai_model"] = self._ai_model
        response = wrapper.predict_automatic_retrosynthesis(smiles, **kwargs)
        prediction_id = self._extract_prediction_id(response)
        if not prediction_id:
            raise RuntimeError("RXN retrosynthesis response did not contain a prediction id.")
        return prediction_id

    def _poll_for_results(self, wrapper: Any, prediction_id: str) -> Any:
        deadline = time.monotonic() + self._max_wait_seconds
        last_status = "UNKNOWN"
        while True:
            payload = wrapper.get_predict_automatic_retrosynthesis_results(prediction_id)
            status = self._extract_status(payload)
            if status:
                last_status = status
            if self._has_paths(payload) or last_status in {"SUCCESS", "DONE", "COMPLETED", "FINISHED"}:
                return payload
            if last_status in {"FAILED", "ERROR", "ABORTED", "CANCELLED"}:
                raise RuntimeError(f"RXN retrosynthesis failed with status `{last_status}`.")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"RXN retrosynthesis timed out after {self._max_wait_seconds:.0f}s waiting for prediction {prediction_id}."
                )
            time.sleep(self._poll_interval_seconds)

    def _summarize_results(self, prediction_id: str, payload: Any) -> _RetrosynthesisSummary:
        status = self._extract_status(payload) or "SUCCESS"
        paths = self._extract_retrosynthetic_paths(payload)
        route_count = len(paths)
        depths = [self._route_depth(path) for path in paths]
        nonzero_depths = [depth for depth in depths if depth > 0]
        best_route_depth = min(nonzero_depths) if nonzero_depths else None
        max_route_depth = max(nonzero_depths) if nonzero_depths else None
        return _RetrosynthesisSummary(
            prediction_id=prediction_id,
            status=status,
            route_count=route_count,
            best_route_depth=best_route_depth,
            max_route_depth=max_route_depth,
        )

    def _build_results(
        self,
        smiles: str,
        summary: _RetrosynthesisSummary,
    ) -> list[CandidateEvaluationResult]:
        citation_urls = [self._base_url]
        route_count_value = float(summary.route_count)
        best_route_depth = float(summary.best_route_depth) if summary.best_route_depth is not None else None
        max_route_depth = float(summary.max_route_depth) if summary.max_route_depth is not None else None
        feasibility_score = self._feasibility_score(summary.route_count, summary.best_route_depth)
        feasibility_label = self._feasibility_label(feasibility_score, summary.route_count, summary.best_route_depth)
        results = [
            CandidateEvaluationResult(
                criterion_name="synthesis_feasibility",
                value=feasibility_label,
                normalized_score=feasibility_score,
                confidence=0.7,
                rationale=(
                    f"RXN retrosynthesis found {summary.route_count} route(s)"
                    f" with best route depth {summary.best_route_depth or 'unknown'} for SMILES `{smiles}`."
                ),
                evidence_mode="external_tool",
                tool_id=self.tool_id,
                citation_urls=citation_urls,
                is_inferred=True,
            ),
            CandidateEvaluationResult(
                criterion_name="retrosynthesis_route_count",
                value=route_count_value,
                confidence=0.78,
                rationale=f"RXN retrosynthesis returned {summary.route_count} candidate route(s) for `{smiles}`.",
                evidence_mode="external_tool",
                tool_id=self.tool_id,
                citation_urls=citation_urls,
                is_inferred=True,
            ),
        ]
        if best_route_depth is not None:
            results.append(
                CandidateEvaluationResult(
                    criterion_name="retrosynthesis_best_route_depth",
                    value=best_route_depth,
                    confidence=0.74,
                    rationale=(
                        f"The shortest RXN retrosynthetic route for `{smiles}` had depth {summary.best_route_depth}."
                    ),
                    evidence_mode="external_tool",
                    tool_id=self.tool_id,
                    citation_urls=citation_urls,
                    is_inferred=True,
                )
            )
        if max_route_depth is not None:
            results.append(
                CandidateEvaluationResult(
                    criterion_name="retrosynthesis_max_route_depth",
                    value=max_route_depth,
                    confidence=0.74,
                    rationale=(
                        f"The deepest RXN retrosynthetic route for `{smiles}` had depth {summary.max_route_depth}."
                    ),
                    evidence_mode="external_tool",
                    tool_id=self.tool_id,
                    citation_urls=citation_urls,
                    is_inferred=True,
                )
            )
        return results

    @staticmethod
    def _feasibility_score(route_count: int, best_route_depth: int | None) -> float:
        if route_count <= 0:
            return 0.12
        score = 0.42 + min(route_count, 4) * 0.08
        if best_route_depth is None:
            return min(0.78, round(score, 2))
        if best_route_depth <= 2:
            score += 0.18
        elif best_route_depth <= 3:
            score += 0.12
        elif best_route_depth <= 5:
            score += 0.05
        else:
            score -= min(best_route_depth - 5, 4) * 0.05
        return max(0.05, min(0.95, round(score, 2)))

    @staticmethod
    def _feasibility_label(score: float, route_count: int, best_route_depth: int | None) -> str:
        if score >= 0.8:
            label = "strong"
        elif score >= 0.6:
            label = "moderate"
        elif score >= 0.4:
            label = "challenging"
        else:
            label = "weak"
        depth_text = str(best_route_depth) if best_route_depth is not None else "unknown"
        return f"{label} retrosynthetic accessibility ({route_count} route(s), best depth {depth_text})"

    @staticmethod
    def _extract_prediction_id(payload: Any) -> str | None:
        direct = RXN4ChemistryRetrosynthesisSkill._first_text(
            payload,
            "prediction_id",
            "predictionId",
        )
        if direct:
            return direct
        for nested in ("response", "payload"):
            if isinstance(payload, dict) and isinstance(payload.get(nested), dict):
                nested_id = RXN4ChemistryRetrosynthesisSkill._first_text(payload[nested], "id", "prediction_id")
                if nested_id:
                    return nested_id
        return None

    @staticmethod
    def _extract_project_id(payload: Any) -> str | None:
        direct = RXN4ChemistryRetrosynthesisSkill._first_text(payload, "project_id", "projectId", "id")
        if direct:
            return direct
        if isinstance(payload, dict):
            response = payload.get("response")
            if isinstance(response, dict):
                nested = RXN4ChemistryRetrosynthesisSkill._first_text(response.get("payload"), "id", "project_id")
                if nested:
                    return nested
        return None

    @staticmethod
    def _extract_status(payload: Any) -> str | None:
        if isinstance(payload, dict):
            status = payload.get("status")
            if status is not None:
                return str(status).strip().upper()
            for key in ("response", "payload"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    nested_status = RXN4ChemistryRetrosynthesisSkill._extract_status(nested)
                    if nested_status:
                        return nested_status
        return None

    @staticmethod
    def _extract_retrosynthetic_paths(payload: Any) -> list[Any]:
        if isinstance(payload, dict):
            paths = payload.get("retrosynthetic_paths")
            if isinstance(paths, list):
                return paths
            for value in payload.values():
                nested = RXN4ChemistryRetrosynthesisSkill._extract_retrosynthetic_paths(value)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for item in payload:
                nested = RXN4ChemistryRetrosynthesisSkill._extract_retrosynthetic_paths(item)
                if nested:
                    return nested
        return []

    @staticmethod
    def _has_paths(payload: Any) -> bool:
        return bool(RXN4ChemistryRetrosynthesisSkill._extract_retrosynthetic_paths(payload))

    @staticmethod
    def _route_depth(route: Any) -> int:
        if isinstance(route, dict):
            children = route.get("children")
            if isinstance(children, list) and children:
                return 1 + max(RXN4ChemistryRetrosynthesisSkill._route_depth(child) for child in children)
            for key in ("steps", "reactions"):
                value = route.get(key)
                if isinstance(value, list) and value:
                    return len(value)
            return 1 if route else 0
        if isinstance(route, list):
            if not route:
                return 0
            return max(RXN4ChemistryRetrosynthesisSkill._route_depth(item) for item in route)
        return 0

    @staticmethod
    def _first_text(payload: Any, *keys: str) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _iter_named_payloads(payload: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("title")
            identifier = payload.get("id") or payload.get("project_id")
            if name is not None and identifier is not None:
                items.append(payload)
            for value in payload.values():
                items.extend(RXN4ChemistryRetrosynthesisSkill._iter_named_payloads(value))
        elif isinstance(payload, list):
            for item in payload:
                items.extend(RXN4ChemistryRetrosynthesisSkill._iter_named_payloads(item))
        return items

    @staticmethod
    def _render_result_value(result: CandidateEvaluationResult) -> str:
        if result.value is None:
            return "unknown"
        if result.unit:
            return f"{result.value} {result.unit}"
        return str(result.value)

    def _load_cached_prediction(
        self,
        smiles: str,
    ) -> tuple[_RetrosynthesisSummary, list[CandidateEvaluationResult]] | None:
        path = self._cache_path(smiles)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed to read RXN cache file %s", path)
            return None
        raw_summary = payload.get("summary")
        raw_results = payload.get("results")
        if not isinstance(raw_summary, dict) or not isinstance(raw_results, list):
            return None
        try:
            summary = _RetrosynthesisSummary(
                prediction_id=str(raw_summary.get("prediction_id") or "").strip(),
                status=str(raw_summary.get("status") or "").strip() or "SUCCESS",
                route_count=int(raw_summary.get("route_count") or 0),
                best_route_depth=(
                    int(raw_summary["best_route_depth"]) if raw_summary.get("best_route_depth") is not None else None
                ),
                max_route_depth=(
                    int(raw_summary["max_route_depth"]) if raw_summary.get("max_route_depth") is not None else None
                ),
            )
            results = [CandidateEvaluationResult.model_validate(item) for item in raw_results]
        except Exception:
            LOGGER.warning("Failed to validate cached RXN predictions from %s", path)
            return None
        return summary, results

    def _store_cached_prediction(
        self,
        smiles: str,
        summary: _RetrosynthesisSummary,
        results: list[CandidateEvaluationResult],
        raw_payload: Any,
    ) -> None:
        path = self._cache_path(smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "smiles": smiles,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "prediction_id": summary.prediction_id,
                "status": summary.status,
                "route_count": summary.route_count,
                "best_route_depth": summary.best_route_depth,
                "max_route_depth": summary.max_route_depth,
            },
            "results": [result.model_dump(mode="json") for result in results],
            "raw_payload": raw_payload,
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed to write RXN cache file %s", path)

    def _cache_path(self, smiles: str) -> Path:
        return self._cache_dir / f"{self._cache_key(smiles)}.json"

    @staticmethod
    def _cache_key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()
