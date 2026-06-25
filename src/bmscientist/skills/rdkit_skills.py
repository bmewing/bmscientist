from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec
from bmscientist.skills.molecule_support import (
    FUNCTIONAL_GROUP_SMARTS,
    HAZARD_ALERT_SMARTS,
    compact_text,
    extract_molecule_identifiers,
)


LOGGER = logging.getLogger(__name__)

RDKIT_PROFILE_TOOL_ID = "rdkit_profile"
RDKIT_SIMILARITY_TOOL_ID = "rdkit_similarity_and_alerts"


class RDKitProfileSkill:
    def __init__(self, config: Any | None = None, *, cache_dir: Path | None = None):
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif config is not None and getattr(config, "data_dir", None) is not None:
            self._cache_dir = Path(config.data_dir) / "skills" / "rdkit"
        else:
            self._cache_dir = Path("data") / "skills" / "rdkit"
        self._spec = SkillSpec(
            skill_id=RDKIT_PROFILE_TOOL_ID,
            description=(
                "Compute local RDKit descriptors for SMILES candidates, including molecular weight, TPSA, "
                "LogP proxy, hydrogen-bond counts, ring counts, and functional groups."
            ),
            phases=("reflection", "enrichment"),
            aliases=("rdkit", "molecule_descriptors", "rdkit_descriptors"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            required_candidate_fields=("smiles",),
            expected_outputs=(
                "molecular_weight_rdkit",
                "tpsa_rdkit",
                "logp_rdkit",
                "hbond_donor_count",
                "hbond_acceptor_count",
                "rotatable_bond_count",
                "ring_count",
                "heavy_atom_count",
            ),
            trigger_keywords=("smiles", "descriptor", "logp", "tpsa", "molecular weight", "functional group"),
            provider="python_package",
            priority=30,
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
        document = context.document
        if document.research_mode == "candidate_design" and document.candidate_artifact_schema.primary_identifier_field.strip().lower() == "smiles":
            return True
        keyword_text = " ".join(
            [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
            + document.reflection_guidance
            + [context.purpose]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        smiles = identifiers.canonical_smiles or identifiers.smiles
        if not smiles:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="skipped",
                notes=["No SMILES string was available for RDKit profiling."],
                rationale="RDKit descriptor profiling requires a SMILES string.",
            )

        cached = self._load_cache(smiles)
        if cached is None:
            cached = self._compute_profile(smiles)
            self._store_cache(smiles, cached)
        results = self._results_from_profile(smiles, cached)
        evidence_rows = self._evidence_rows(smiles, identifiers.name or "Unknown candidate", cached, results)
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=[f"Computed {len(results)} RDKit descriptor signals for `{identifiers.name or smiles}`."],
            rationale="Computed local RDKit descriptors and functional-group annotations from the candidate SMILES.",
            metadata={"functional_groups": cached["functional_groups"]},
        )

    @staticmethod
    def _compute_profile(smiles: str) -> dict[str, Any]:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES string for RDKit profiling: {smiles}")
        functional_groups = []
        for name, smarts in FUNCTIONAL_GROUP_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and molecule.HasSubstructMatch(pattern):
                functional_groups.append(name)
        return {
            "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
            "molecular_weight_rdkit": round(rdMolDescriptors.CalcExactMolWt(molecule), 6),
            "molecular_formula_rdkit": rdMolDescriptors.CalcMolFormula(molecule),
            "tpsa_rdkit": round(rdMolDescriptors.CalcTPSA(molecule), 6),
            "logp_rdkit": round(Crippen.MolLogP(molecule), 6),
            "hbond_donor_count": int(Lipinski.NumHDonors(molecule)),
            "hbond_acceptor_count": int(Lipinski.NumHAcceptors(molecule)),
            "rotatable_bond_count": int(Lipinski.NumRotatableBonds(molecule)),
            "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
            "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
            "fingerprint_density_morgan2": round(Descriptors.FpDensityMorgan2(molecule), 6),
            "functional_groups": functional_groups,
        }

    def _results_from_profile(self, smiles: str, payload: dict[str, Any]) -> list[CandidateEvaluationResult]:
        outputs = (
            ("molecular_weight_rdkit", "Da"),
            ("tpsa_rdkit", "A2"),
            ("logp_rdkit", None),
            ("hbond_donor_count", None),
            ("hbond_acceptor_count", None),
            ("rotatable_bond_count", None),
            ("ring_count", None),
            ("heavy_atom_count", None),
        )
        results: list[CandidateEvaluationResult] = []
        for criterion_name, unit in outputs:
            value = payload.get(criterion_name)
            if value is None:
                continue
            results.append(
                CandidateEvaluationResult(
                    criterion_name=criterion_name,
                    value=float(value) if isinstance(value, (int, float)) else value,
                    unit=unit,
                    confidence=0.94,
                    rationale=f"Computed directly with RDKit from canonical SMILES `{payload['canonical_smiles'] or smiles}`.",
                    evidence_mode="local_tool",
                    tool_id=self.spec.skill_id,
                    citation_chunk_ids=[f"rdkit:{self._cache_key(smiles)}:{criterion_name}"],
                    is_inferred=False,
                )
            )
        return results

    def _evidence_rows(
        self,
        smiles: str,
        candidate_name: str,
        payload: dict[str, Any],
        results: list[CandidateEvaluationResult],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        functional_groups_text = ", ".join(payload["functional_groups"]) or "none detected"
        for result in results:
            rows.append(
                {
                    "id": f"rdkit:{self._cache_key(smiles)}:{result.criterion_name}",
                    "source_url": "local://rdkit",
                    "source_title": "Local RDKit profile",
                    "application": None,
                    "incumbent_material": None,
                    "candidate_materials": [candidate_name],
                    "relevance_score": 0.9,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"RDKit computed {result.criterion_name} for {candidate_name} ({payload['canonical_smiles']}): "
                        f"{result.value}{f' {result.unit}' if result.unit else ''}. "
                        f"Functional groups detected: {functional_groups_text}."
                    )[:1800],
                    "metadata": {
                        "source_type": "local-tool",
                        "tool_id": self.spec.skill_id,
                        "smiles": payload["canonical_smiles"],
                        "functional_groups": payload["functional_groups"],
                        "endpoint_name": result.criterion_name,
                        "value": result.value,
                        "unit": result.unit,
                    },
                }
            )
        return rows

    def _load_cache(self, smiles: str) -> dict[str, Any] | None:
        path = self._cache_path(smiles)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed reading RDKit cache file %s", path)
            return None

    def _store_cache(self, smiles: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed writing RDKit cache file %s", path)

    def _cache_path(self, smiles: str) -> Path:
        return self._cache_dir / "profile" / f"{self._cache_key(smiles)}.json"

    @staticmethod
    def _cache_key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()


class RDKitSimilarityAndAlertsSkill:
    def __init__(self):
        self._spec = SkillSpec(
            skill_id=RDKIT_SIMILARITY_TOOL_ID,
            description=(
                "Compute RDKit similarity against reference molecules when available and flag simple hazard motifs "
                "such as nitro groups, organic peroxides, azides, and diazo patterns."
            ),
            phases=("reflection",),
            aliases=("molecule_similarity", "hazard_alerts", "rdkit_similarity"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            required_candidate_fields=("smiles",),
            expected_outputs=("hazard_motif_count", "explosive_alert_score", "reference_similarity_max"),
            trigger_keywords=("similarity", "substructure", "alert", "explosive", "hazard", "novelty"),
            provider="python_package",
            priority=35,
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
        document = context.document
        keyword_text = " ".join(
            [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
            + document.reflection_guidance
            + [context.purpose, document.novelty_check_policy]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        smiles = identifiers.canonical_smiles or identifiers.smiles
        if not smiles:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="skipped",
                notes=["No SMILES string was available for similarity and alert analysis."],
                rationale="RDKit similarity and alert screening requires a SMILES string.",
            )
        payload = self._compute(smiles, context.document.known_candidate_exclusion_terms)
        results = self._results_from_payload(smiles, payload)
        evidence_rows = self._evidence_rows(smiles, identifiers.name or "Unknown candidate", payload, results)
        notes = []
        if payload["hazard_alert_names"]:
            notes.append(f"Hazard motifs detected: {', '.join(payload['hazard_alert_names'])}.")
        if payload["nearest_reference"]:
            notes.append(
                f"Nearest reference analog `{payload['nearest_reference']['reference']}` had similarity "
                f"{payload['nearest_reference']['similarity']:.3f}."
            )
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=notes,
            rationale="Computed local RDKit hazard-pattern alerts and reference similarity signals.",
            metadata={
                "hazard_alert_names": payload["hazard_alert_names"],
                "nearest_reference": payload["nearest_reference"],
            },
        )

    @staticmethod
    def _compute(smiles: str, reference_values: list[str]) -> dict[str, Any]:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES string for RDKit similarity screening: {smiles}")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        hazard_alert_names: list[str] = []
        explosive_alert_score = 0.0
        for name, smarts, score in HAZARD_ALERT_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and molecule.HasSubstructMatch(pattern):
                hazard_alert_names.append(name)
                explosive_alert_score = max(explosive_alert_score, score)
        nearest_reference: dict[str, Any] | None = None
        for raw_reference in reference_values:
            reference_smiles = compact_text(raw_reference)
            if not reference_smiles:
                continue
            reference_mol = Chem.MolFromSmiles(reference_smiles)
            if reference_mol is None:
                continue
            reference_fp = AllChem.GetMorganFingerprintAsBitVect(reference_mol, 2, nBits=2048)
            similarity = float(DataStructs.TanimotoSimilarity(fingerprint, reference_fp))
            if nearest_reference is None or similarity > nearest_reference["similarity"]:
                nearest_reference = {"reference": reference_smiles, "similarity": similarity}
        return {
            "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
            "hazard_alert_names": hazard_alert_names,
            "hazard_motif_count": len(hazard_alert_names),
            "explosive_alert_score": round(explosive_alert_score, 6),
            "nearest_reference": nearest_reference,
            "reference_similarity_max": round(nearest_reference["similarity"], 6) if nearest_reference else None,
        }

    def _results_from_payload(self, smiles: str, payload: dict[str, Any]) -> list[CandidateEvaluationResult]:
        results = [
            CandidateEvaluationResult(
                criterion_name="hazard_motif_count",
                value=float(payload["hazard_motif_count"]),
                confidence=0.9,
                rationale=f"Counted RDKit hazard motifs for canonical SMILES `{payload['canonical_smiles'] or smiles}`.",
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_chunk_ids=[f"rdkit-alerts:{self._cache_key(smiles)}:hazard_motif_count"],
                is_inferred=False,
            ),
            CandidateEvaluationResult(
                criterion_name="explosive_alert_score",
                value=float(payload["explosive_alert_score"]),
                normalized_score=float(payload["explosive_alert_score"]),
                confidence=0.82,
                rationale=(
                    "Computed a heuristic explosive-concern score from matched RDKit hazard-alert SMARTS patterns."
                ),
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_chunk_ids=[f"rdkit-alerts:{self._cache_key(smiles)}:explosive_alert_score"],
                is_inferred=True,
            ),
        ]
        if payload["reference_similarity_max"] is not None:
            results.append(
                CandidateEvaluationResult(
                    criterion_name="reference_similarity_max",
                    value=float(payload["reference_similarity_max"]),
                    normalized_score=float(payload["reference_similarity_max"]),
                    confidence=0.85,
                    rationale=(
                        f"Computed maximum Tanimoto similarity against provided reference molecules for `{payload['canonical_smiles']}`."
                    ),
                    evidence_mode="local_tool",
                    tool_id=self.spec.skill_id,
                    citation_chunk_ids=[f"rdkit-alerts:{self._cache_key(smiles)}:reference_similarity_max"],
                    is_inferred=False,
                )
            )
        return results

    def _evidence_rows(
        self,
        smiles: str,
        candidate_name: str,
        payload: dict[str, Any],
        results: list[CandidateEvaluationResult],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        alert_text = ", ".join(payload["hazard_alert_names"]) or "none detected"
        for result in results:
            rows.append(
                {
                    "id": f"rdkit-alerts:{self._cache_key(smiles)}:{result.criterion_name}",
                    "source_url": "local://rdkit",
                    "source_title": "Local RDKit similarity and alerts",
                    "application": None,
                    "incumbent_material": None,
                    "candidate_materials": [candidate_name],
                    "relevance_score": 0.86,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"RDKit similarity/alert analysis for {candidate_name} ({payload['canonical_smiles']}): "
                        f"{result.criterion_name}={result.value}. Hazard alerts: {alert_text}."
                    )[:1800],
                    "metadata": {
                        "source_type": "local-tool",
                        "tool_id": self.spec.skill_id,
                        "smiles": payload["canonical_smiles"],
                        "hazard_alert_names": payload["hazard_alert_names"],
                        "nearest_reference": payload["nearest_reference"],
                        "endpoint_name": result.criterion_name,
                        "value": result.value,
                    },
                }
            )
        return rows

    @staticmethod
    def _cache_key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()
