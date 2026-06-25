from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec


LOGGER = logging.getLogger(__name__)

EPISUITE_TOOL_ID = "epa_episuite"
DEFAULT_EPISUITE_BASE_URL = "https://episuite.dev"


@dataclass(frozen=True)
class _EndpointSpec:
    criterion_name: str
    selectors: tuple[str, ...]
    default_unit: str | None = None


@dataclass(frozen=True)
class _PayloadCandidate:
    path_text: str
    value: Any
    unit: str | None = None


ENDPOINT_SPECS: tuple[_EndpointSpec, ...] = (
    _EndpointSpec("boiling_point_c", ("boiling point",), "C"),
    _EndpointSpec("melting_point_c", ("melting point",), "C"),
    _EndpointSpec("vapor_pressure_mm_hg", ("vapor pressure", "vapour pressure"), "mm Hg"),
    _EndpointSpec("water_solubility_mg_l", ("water solubility", "solubility in water"), "mg/L"),
    _EndpointSpec("log_kow", ("log kow", "log k ow", "logp"), None),
    _EndpointSpec("henrys_law_constant_atm_m3_per_mol", ("henry law", "henry's law"), "atm-m3/mol"),
    _EndpointSpec("log_koc", ("log koc", "soil adsorption"), None),
    _EndpointSpec("bioconcentration_factor_bcf", ("bioconcentration factor", "bcf"), None),
    _EndpointSpec(
        "ready_biodegradation_probability",
        ("ready biodegradation", "biodegradation probability", "biowin"),
        None,
    ),
    _EndpointSpec("atmospheric_half_life_hours", ("atmospheric oxidation half life", "atmospheric half-life"), "hours"),
    _EndpointSpec("fish_lc50_mg_l", ("fish lc50", "fathead minnow lc50"), "mg/L"),
    _EndpointSpec("daphnia_ec50_mg_l", ("daphnia ec50", "daphnia lc50"), "mg/L"),
    _EndpointSpec("algae_ec50_mg_l", ("algae ec50", "green algae ec50"), "mg/L"),
)


class EPISuiteSkill:
    def __init__(
        self,
        config: Any | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
        cache_dir: Path | None = None,
        session: requests.Session | None = None,
    ):
        configured_base_url = base_url or os.getenv("EPISUITE_API_BASE_URL") or DEFAULT_EPISUITE_BASE_URL
        self._base_url = configured_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds or int(getattr(config, "request_timeout_seconds", 60) or 60)
        self._session = session or requests.Session()
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif config is not None and getattr(config, "data_dir", None) is not None:
            self._cache_dir = Path(config.data_dir) / "skills" / "episuite"
        else:
            self._cache_dir = Path("data") / "skills" / "episuite"
        self._spec = SkillSpec(
            skill_id=EPISUITE_TOOL_ID,
            description="Predict physicochemical, fate, and ecotoxicity endpoints from SMILES using the EPA EPISuite API.",
            phases=("reflection", "enrichment"),
            aliases=("episuite", "epa_episuite_api"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            required_candidate_fields=("smiles",),
            expected_outputs=tuple(spec.criterion_name for spec in ENDPOINT_SPECS),
            trigger_keywords=(
                "smiles",
                "toxicity",
                "ecotox",
                "solubility",
                "log_kow",
                "vapor pressure",
                "biodegradation",
                "bcf",
            ),
            provider="http_api",
            priority=80,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    @property
    def tool_id(self) -> str:
        return EPISUITE_TOOL_ID

    def is_applicable(self, context: SkillContext) -> bool:
        if context.hypothesis is None:
            return False
        candidate_artifact = getattr(context.hypothesis, "candidate_artifact", {}) or {}
        return bool(str(candidate_artifact.get("smiles") or candidate_artifact.get("canonical_smiles") or "").strip())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False

        requested = {item.strip().lower() for item in context.requested_skill_ids if item.strip()}
        if self.tool_id in requested:
            return True

        document = context.document
        primary_identifier = document.candidate_artifact_schema.primary_identifier_field.strip().lower()
        if document.research_mode == "candidate_design" and primary_identifier == "smiles":
            return True

        keyword_text = " ".join(
            [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
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
                rationale="EPISuite requires a SMILES string.",
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
        results = self.predict_smiles(smiles)
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
            notes=[f"Predicted {len(results)} EPISuite endpoints for `{candidate_name}`."] if results else [],
            rationale="Executed the EPA EPISuite API against the candidate SMILES and normalized selected endpoints.",
        )

    def request_url_for(self, smiles: str) -> str:
        return f"{self._base_url}/api/submit?smiles={quote(smiles, safe='')}"

    def predict_smiles(self, smiles: str) -> list[CandidateEvaluationResult]:
        normalized_smiles = str(smiles or "").strip()
        if not normalized_smiles:
            return []

        cached = self._load_cached_predictions(normalized_smiles)
        if cached is not None:
            return cached

        payload = self._fetch_payload(normalized_smiles)
        results = self._extract_results(payload, normalized_smiles)
        self._store_cached_predictions(normalized_smiles, payload, results)
        return results

    def build_evidence_rows(
        self,
        *,
        smiles: str,
        candidate_name: str,
        application: str | None = None,
        incumbent_material: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self.predict_smiles(smiles)
        if not results:
            return []

        source_url = self.request_url_for(smiles)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []
        for result in results:
            value_text = self._render_result_value(result)
            rows.append(
                {
                    "id": f"episuite:{self._cache_key(smiles)}:{result.criterion_name}",
                    "source_url": source_url,
                    "source_title": "EPA EPISuite prediction",
                    "application": application,
                    "incumbent_material": incumbent_material,
                    "candidate_materials": [candidate_name] if candidate_name else [],
                    "relevance_score": 0.82,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"EPA EPISuite predicted {result.criterion_name} for SMILES {smiles}: {value_text}. "
                        f"{result.rationale}"
                    )[:1800],
                    "metadata": {
                        "source_type": "external-tool",
                        "tool_id": self.tool_id,
                        "smiles": smiles,
                        "endpoint_name": result.criterion_name,
                        "value": result.value,
                        "unit": result.unit,
                        "is_inferred": result.is_inferred,
                    },
                }
            )
        return rows

    def _fetch_payload(self, smiles: str) -> Any:
        response = self._session.get(
            f"{self._base_url}/api/submit",
            params={"smiles": smiles},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _extract_results(self, payload: Any, smiles: str) -> list[CandidateEvaluationResult]:
        candidates = list(self._iter_candidates(payload))
        if not candidates:
            return []

        results: list[CandidateEvaluationResult] = []
        for spec in ENDPOINT_SPECS:
            candidate = self._best_candidate_for_spec(spec, candidates)
            if candidate is None:
                continue
            parsed = self._coerce_candidate_value(candidate.value)
            if parsed is None:
                continue
            numeric_value, text_value, parsed_unit = parsed
            value: str | float | bool | None = numeric_value if numeric_value is not None else text_value
            if value in (None, ""):
                continue
            results.append(
                CandidateEvaluationResult(
                    criterion_name=spec.criterion_name,
                    value=value,
                    unit=candidate.unit or parsed_unit or spec.default_unit,
                    confidence=0.72 if numeric_value is not None else 0.6,
                    rationale=(
                        f"EPA EPISuite API prediction extracted from response field `{candidate.path_text}` "
                        f"for SMILES `{smiles}`."
                    ),
                    evidence_mode="external_tool",
                    tool_id=self.tool_id,
                    citation_urls=[self.request_url_for(smiles)],
                    is_inferred=True,
                )
            )

        deduped: "OrderedDict[str, CandidateEvaluationResult]" = OrderedDict()
        for result in results:
            deduped.setdefault(result.criterion_name, result)
        return list(deduped.values())

    def _best_candidate_for_spec(
        self,
        spec: _EndpointSpec,
        candidates: list[_PayloadCandidate],
    ) -> _PayloadCandidate | None:
        ranked: list[tuple[tuple[int, int, int], _PayloadCandidate]] = []
        for candidate in candidates:
            selector_score = max((len(selector) for selector in spec.selectors if selector in candidate.path_text), default=0)
            if selector_score <= 0:
                continue
            parsed = self._coerce_candidate_value(candidate.value)
            has_numeric = 1 if parsed is not None and parsed[0] is not None else 0
            ranked.append(((has_numeric, selector_score, len(candidate.path_text)), candidate))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    def _iter_candidates(self, payload: Any, path: tuple[str, ...] = ()) -> list[_PayloadCandidate]:
        candidates: list[_PayloadCandidate] = []
        if isinstance(payload, dict):
            unit = self._extract_unit(payload)
            value_key = self._extract_value_key(payload)
            if value_key is not None:
                value = payload.get(value_key)
                if self._is_scalar(value):
                    candidates.append(_PayloadCandidate(self._normalize_path(path), value, unit))
            for key, value in payload.items():
                next_path = (*path, str(key))
                if self._is_scalar(value):
                    if str(key).lower() not in {"unit", "units"}:
                        candidates.append(_PayloadCandidate(self._normalize_path(next_path), value, unit))
                    continue
                candidates.extend(self._iter_candidates(value, next_path))
            return candidates
        if isinstance(payload, list):
            for index, item in enumerate(payload):
                candidates.extend(self._iter_candidates(item, (*path, str(index))))
            return candidates
        if self._is_scalar(payload):
            return [_PayloadCandidate(self._normalize_path(path), payload, None)]
        return []

    @staticmethod
    def _extract_value_key(payload: dict[str, Any]) -> str | None:
        for key in ("value", "prediction", "predicted", "result", "estimate", "mean", "median"):
            if key in payload and EPISuiteSkill._is_scalar(payload.get(key)):
                return key
        return None

    @staticmethod
    def _extract_unit(payload: dict[str, Any]) -> str | None:
        for key in ("unit", "units"):
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _normalize_path(path: tuple[str, ...]) -> str:
        normalized = " ".join(path)
        normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
        normalized = normalized.lower()
        normalized = re.sub(r"[_/\\-]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    @staticmethod
    def _coerce_candidate_value(value: Any) -> tuple[float | None, str | None, str | None] | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None, str(value), None
        if isinstance(value, (int, float)):
            return float(value), None, None

        text = str(value).strip()
        if not text:
            return None
        if text.lower() in {"nan", "na", "n/a", "none", "null"}:
            return None

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match is None:
            return None, text, None
        try:
            numeric = float(match.group(0))
        except ValueError:
            return None, text, None
        unit = text[match.end():].strip(" ,;:()[]") or None
        return numeric, None, unit

    @staticmethod
    def _render_result_value(result: CandidateEvaluationResult) -> str:
        if result.value is None:
            return "unknown"
        if result.unit:
            return f"{result.value} {result.unit}"
        return str(result.value)

    def _load_cached_predictions(self, smiles: str) -> list[CandidateEvaluationResult] | None:
        path = self._cache_path(smiles)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed to read EPISuite cache file %s", path)
            return None
        results = payload.get("results")
        if not isinstance(results, list):
            return None
        try:
            return [CandidateEvaluationResult.model_validate(item) for item in results]
        except Exception:
            LOGGER.warning("Failed to validate cached EPISuite predictions from %s", path)
            return None

    def _store_cached_predictions(
        self,
        smiles: str,
        raw_payload: Any,
        results: list[CandidateEvaluationResult],
    ) -> None:
        path = self._cache_path(smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "smiles": smiles,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "request_url": self.request_url_for(smiles),
            "results": [result.model_dump(mode="json") for result in results],
            "raw_payload": raw_payload,
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed to write EPISuite cache file %s", path)

    def _cache_path(self, smiles: str) -> Path:
        return self._cache_dir / f"{self._cache_key(smiles)}.json"

    @staticmethod
    def _cache_key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()
