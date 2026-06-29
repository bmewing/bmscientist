from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec
from bmscientist.skills.molecule_support import compact_text, extract_molecule_identifiers


LOGGER = logging.getLogger(__name__)

HANSEN_XGBOOST_TOOL_ID = "hansen_solubility_xgboost"
HSP_REPO_URL = "https://github.com/darjacvetkovic/HSP-predictions"
HSP_PAPER_URL = "https://doi.org/10.1016/j.chemolab.2024.105168"


@dataclass(frozen=True)
class HansenPrediction:
    canonical_smiles: str
    delta_d: float
    delta_p: float
    delta_h: float
    total_hsp: float
    model_variant: str
    feature_counts: dict[str, int]


class HansenSolubilityXGBoostPredictor:
    def __init__(self, model_variant: str = "50"):
        self._model_variant = model_variant
        self._models: dict[str, Any] | None = None
        self._features: dict[str, list[str]] | None = None

    def predict_smiles(self, smiles: str) -> HansenPrediction:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES string for Hansen solubility prediction: {smiles}")
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        descriptors = self._mordred_descriptors(molecule)
        models = self._load_models()
        features = self._load_feature_lists()

        predictions: dict[str, float] = {}
        for component in ("D", "P", "H"):
            frame = self._feature_frame(descriptors, features[component], component)
            predictions[component] = float(models[component].predict(frame)[0])

        total_hsp = math.sqrt(predictions["D"] ** 2 + predictions["P"] ** 2 + predictions["H"] ** 2)
        return HansenPrediction(
            canonical_smiles=canonical_smiles,
            delta_d=predictions["D"],
            delta_p=predictions["P"],
            delta_h=predictions["H"],
            total_hsp=total_hsp,
            model_variant=self._model_variant,
            feature_counts={component: len(items) for component, items in features.items()},
        )

    def _load_models(self) -> dict[str, Any]:
        if self._models is not None:
            return self._models
        import xgboost as xgb

        model_root = files("bmscientist.resources.hsp_xgboost").joinpath(self._model_variant)
        models: dict[str, Any] = {}
        for component in ("D", "P", "H"):
            model = xgb.XGBRegressor(enable_categorical=True)
            model.load_model(str(model_root.joinpath(f"MODEL_{component}_XGBOOST.json")))
            models[component] = model
        self._models = models
        return models

    def _load_feature_lists(self) -> dict[str, list[str]]:
        if self._features is not None:
            return self._features
        models = self._load_models()
        model_root = files("bmscientist.resources.hsp_xgboost").joinpath(self._model_variant)
        feature_lists: dict[str, list[str]] = {}
        for component, model in models.items():
            model_features = getattr(model, "feature_names_in_", None)
            if model_features is None:
                model_features = getattr(model.get_booster(), "feature_names", None)
            if model_features is not None and len(model_features) > 0:
                feature_lists[component] = [str(item) for item in model_features]
                continue
            feature_lists[component] = [
                line.strip()
                for line in model_root.joinpath(f"{component}_{self._model_variant}.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self._features = feature_lists
        return self._features

    @staticmethod
    def _mordred_descriptors(molecule: Any) -> dict[str, Any]:
        import numpy as np
        from mordred import Calculator, descriptors

        # Mordred 1.2 still references removed numpy aliases in some environments.
        for alias, value in {
            "float": float,
            "int": int,
            "object": object,
            "bool": bool,
        }.items():
            if not hasattr(np, alias):
                setattr(np, alias, value)

        calc = Calculator(descriptors, ignore_3D=True)
        result = calc(molecule)
        return {str(descriptor): value for descriptor, value in zip(result.keys(), result.values(), strict=False)}

    @staticmethod
    def _feature_frame(descriptors: dict[str, Any], feature_names: list[str], component: str) -> Any:
        import pandas as pd

        payload: dict[str, list[float]] = {}
        missing: list[str] = []
        invalid: list[str] = []
        for feature in feature_names:
            if feature not in descriptors:
                missing.append(feature)
                continue
            try:
                value = float(descriptors[feature])
            except (TypeError, ValueError):
                invalid.append(feature)
                continue
            if not math.isfinite(value):
                invalid.append(feature)
                continue
            payload[feature] = [value]
        if missing or invalid:
            details = []
            if missing:
                details.append(f"missing features for {component}: {', '.join(missing[:8])}")
            if invalid:
                details.append(f"invalid Mordred values for {component}: {', '.join(invalid[:8])}")
            raise ValueError("; ".join(details))
        return pd.DataFrame(payload, columns=feature_names)


class HansenSolubilityXGBoostSkill:
    def __init__(
        self,
        config: Any | None = None,
        *,
        predictor: HansenSolubilityXGBoostPredictor | None = None,
        cache_dir: Path | None = None,
    ):
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif config is not None and getattr(config, "data_dir", None) is not None:
            self._cache_dir = Path(config.data_dir) / "skills" / "hansen_solubility_xgboost"
        else:
            self._cache_dir = Path("data") / "skills" / "hansen_solubility_xgboost"
        self._predictor = predictor or HansenSolubilityXGBoostPredictor()
        self._spec = SkillSpec(
            skill_id=HANSEN_XGBOOST_TOOL_ID,
            description=(
                "Predict Hansen solubility parameters from SMILES using the MIT-licensed XGBoost models from "
                "darjacvetkovic/HSP-predictions."
            ),
            phases=("reflection", "enrichment"),
            aliases=("hansen_solubility", "hsp_prediction", "hsp_xgboost", "binder_hsp_compatibility"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            required_candidate_fields=("smiles",),
            expected_outputs=(
                "hsp_delta_d_mpa05",
                "hsp_delta_p_mpa05",
                "hsp_delta_h_mpa05",
                "hsp_total_mpa05",
            ),
            trigger_keywords=(
                "hansen",
                "hsp",
                "solubility parameter",
                "binder compatibility",
                "latex compatibility",
                "miscibility",
            ),
            provider="python_package",
            priority=42,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        identifiers = extract_molecule_identifiers(context)
        return bool(identifiers.canonical_smiles or identifiers.smiles)

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        requested = {str(item).strip().lower() for item in context.requested_skill_ids if str(item).strip()}
        if requested & {self.spec.skill_id, *self.spec.aliases}:
            return True
        keyword_text = " ".join(_document_text_fields(context.document) + [context.purpose, context.question_text]).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        smiles = identifiers.canonical_smiles or identifiers.smiles
        if not smiles:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="skipped",
                notes=["No SMILES string was available for Hansen solubility prediction."],
                rationale="HSP prediction requires a molecule SMILES string.",
            )
        try:
            payload = self._load_cache(smiles)
            if payload is None:
                prediction = self._predictor.predict_smiles(smiles)
                payload = {
                    "canonical_smiles": prediction.canonical_smiles,
                    "hsp_delta_d_mpa05": prediction.delta_d,
                    "hsp_delta_p_mpa05": prediction.delta_p,
                    "hsp_delta_h_mpa05": prediction.delta_h,
                    "hsp_total_mpa05": prediction.total_hsp,
                    "model_variant": prediction.model_variant,
                    "feature_counts": prediction.feature_counts,
                    "reference_repo": HSP_REPO_URL,
                    "reference_paper": HSP_PAPER_URL,
                }
                self._store_cache(smiles, payload)
        except Exception as exc:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="blocked",
                notes=[str(exc)],
                rationale="HSP XGBoost prediction could not be completed for this molecule.",
            )

        results = self._results_from_payload(smiles, payload)
        evidence_rows = self._evidence_rows(smiles, identifiers.name or "Unknown candidate", payload, results)
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=[
                "Predicted HSP values with the 50-feature XGBoost models from darjacvetkovic/HSP-predictions.",
                "Units are MPa^0.5 for delta_d, delta_p, delta_h, and total HSP.",
            ],
            rationale="Computed Mordred 2D descriptors from SMILES and applied bundled XGBoost HSP models.",
            resolved_identifiers={"canonical_smiles": payload["canonical_smiles"], "smiles": payload["canonical_smiles"]},
            metadata=payload,
        )

    def _results_from_payload(self, smiles: str, payload: dict[str, Any]) -> list[CandidateEvaluationResult]:
        outputs = (
            ("hsp_delta_d_mpa05", "dispersion Hansen parameter"),
            ("hsp_delta_p_mpa05", "polar Hansen parameter"),
            ("hsp_delta_h_mpa05", "hydrogen-bond Hansen parameter"),
            ("hsp_total_mpa05", "total Hansen solubility parameter"),
        )
        return [
            CandidateEvaluationResult(
                criterion_name=criterion_name,
                value=float(payload[criterion_name]),
                unit="MPa^0.5",
                confidence=0.72 if criterion_name == "hsp_delta_d_mpa05" else 0.66,
                rationale=(
                    f"Predicted {label} from Mordred descriptors using the MIT-licensed "
                    "HSP-predictions XGBoost model."
                ),
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_urls=[HSP_REPO_URL, HSP_PAPER_URL],
                citation_chunk_ids=[f"hsp-xgboost:{self._cache_key(smiles)}:{criterion_name}"],
                is_inferred=True,
            )
            for criterion_name, label in outputs
        ]

    def _evidence_rows(
        self,
        smiles: str,
        candidate_name: str,
        payload: dict[str, Any],
        results: list[CandidateEvaluationResult],
    ) -> list[dict[str, Any]]:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        return [
            {
                "id": f"hsp-xgboost:{self._cache_key(smiles)}:{result.criterion_name}",
                "source_url": HSP_REPO_URL,
                "source_title": "HSP-predictions XGBoost model",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [candidate_name],
                "relevance_score": 0.86,
                "retrieved_at": retrieved_at,
                "chunk_text": (
                    f"HSP XGBoost prediction for {candidate_name} ({payload['canonical_smiles']}): "
                    f"delta_d={payload['hsp_delta_d_mpa05']:.3f}, delta_p={payload['hsp_delta_p_mpa05']:.3f}, "
                    f"delta_h={payload['hsp_delta_h_mpa05']:.3f}, total={payload['hsp_total_mpa05']:.3f} MPa^0.5. "
                    "Models from darjacvetkovic/HSP-predictions."
                )[:1800],
                "metadata": {
                    "source_type": "local-tool",
                    "tool_id": self.spec.skill_id,
                    "smiles": payload["canonical_smiles"],
                    "endpoint_name": result.criterion_name,
                    "value": result.value,
                    "unit": result.unit,
                    "reference_repo": HSP_REPO_URL,
                    "reference_paper": HSP_PAPER_URL,
                },
            }
            for result in results
        ]

    def _load_cache(self, smiles: str) -> dict[str, Any] | None:
        path = self._cache_path(smiles)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed reading HSP cache file %s", path)
            return None

    def _store_cache(self, smiles: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed writing HSP cache file %s", path)

    def _cache_path(self, smiles: str) -> Path:
        return self._cache_dir / f"{self._cache_key(smiles)}.json"

    @staticmethod
    def _cache_key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()


def _document_text_fields(document: Any) -> list[str]:
    values: list[str] = []
    for field in ("research_mode", "raw_goal", "candidate_origin_policy", "novelty_check_policy"):
        text = compact_text(getattr(document, field, None))
        if text:
            values.append(text)
    for field in ("evaluation_criteria", "reflection_guidance", "material_scope", "application_scope"):
        for item in getattr(document, field, []) or []:
            if hasattr(item, "name"):
                values.append(str(getattr(item, "name", "")))
                values.append(str(getattr(item, "description", "")))
            else:
                values.append(str(item))
    return values
