from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec
from bmscientist.skills.molecule_support import compact_text


LOGGER = logging.getLogger(__name__)

POLYMER_PROPERTY_TOOL_ID = "polymer_property_profile"

_DUMMY_PATTERN = re.compile(r"\[\*:?\d*\]|\*")
_WATER_SOLUBILITY_PARAMETER = math.sqrt(15.1**2 + 20.4**2 + 16.5**2)

_ATOM_MOLAR_VOLUMES_CM3_MOL = {
    1: 8.71,
    5: 18.0,
    6: 16.35,
    7: 14.39,
    8: 5.42,
    9: 12.5,
    14: 26.5,
    15: 24.0,
    16: 22.9,
    17: 22.45,
    35: 26.52,
    53: 32.5,
}

_ATOM_COHESIVE_ENERGIES_J_MOL = {
    1: 1000.0,
    5: 3600.0,
    6: 4200.0,
    7: 7600.0,
    8: 8200.0,
    9: 2800.0,
    14: 3100.0,
    15: 5400.0,
    16: 6100.0,
    17: 3300.0,
    35: 3600.0,
    53: 3900.0,
}

_POLAR_GROUP_SMARTS: tuple[tuple[str, str, float], ...] = (
    ("hydroxyl", "[OX2H]", 11000.0),
    ("carboxylic_acid", "C(=O)[OX2H1]", 15000.0),
    ("amide", "C(=O)N", 13000.0),
    ("ester", "[CX3](=O)[OX2][#6]", 8500.0),
    ("ether", "[OD2]([#6])[#6]", 4200.0),
    ("nitrile", "C#N", 7000.0),
    ("sulfone", "S(=O)(=O)", 12000.0),
    ("carbonate", "O[C](=O)O", 9500.0),
)


class PolymerPropertyProfileSkill:
    def __init__(self, config: Any | None = None, *, cache_dir: Path | None = None):
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif config is not None and getattr(config, "data_dir", None) is not None:
            self._cache_dir = Path(config.data_dir) / "skills" / "polymer_property_profile"
        else:
            self._cache_dir = Path("data") / "skills" / "polymer_property_profile"
        self._spec = SkillSpec(
            skill_id=POLYMER_PROPERTY_TOOL_ID,
            description=(
                "Estimate homopolymer repeat-unit properties from polymer SMILES, inspired by the "
                "polymer_property_prediction project and Bicerano-style group contributions. Returns structured "
                "values instead of CSV output."
            ),
            phases=("reflection", "generation", "enrichment"),
            aliases=("polymer_properties", "polymer_property_prediction", "bicerano_polymer_profile"),
            supported_research_modes=("candidate_design", "formulation_design", "generic_screening"),
            required_candidate_fields=("repeat_unit_smiles",),
            expected_outputs=(
                "polymer_tg_estimate_k",
                "polymer_density_estimate_g_cm3",
                "polymer_solubility_parameter_mpa05",
                "polymer_water_affinity_score",
                "polymer_o2_permeability_proxy",
            ),
            trigger_keywords=(
                "polymer",
                "homopolymer",
                "repeat unit",
                "glass transition",
                "tg",
                "density",
                "permeability",
                "solubility parameter",
            ),
            provider="python_package",
            priority=45,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        return bool(_extract_polymer_smiles(context))

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        requested = {str(item).strip().lower() for item in context.requested_skill_ids if str(item).strip()}
        if requested & {self.spec.skill_id, *self.spec.aliases}:
            return True
        keyword_text = " ".join(
            _document_text_fields(context.document)
            + [context.purpose, context.question_text]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        raw_smiles = _extract_polymer_smiles(context)
        if not raw_smiles:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="skipped",
                notes=["No polymer repeat-unit SMILES was available."],
                rationale="Polymer property profiling requires a repeat-unit or polymer SMILES string.",
            )
        cached = self._load_cache(raw_smiles)
        if cached is None:
            cached = self._compute_profile(raw_smiles)
            self._store_cache(raw_smiles, cached)
        results = self._results_from_profile(raw_smiles, cached)
        evidence_rows = self._evidence_rows(raw_smiles, _candidate_label(context), cached, results)
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=[
                "Polymer estimates are local group-contribution approximations; use calibrated data when available.",
                "Reference basis: polymer_property_prediction project.",
            ],
            rationale=(
                "Estimated polymer repeat-unit descriptors with local RDKit parsing and group-contribution heuristics "
                "adapted for structured agent consumption."
            ),
            resolved_identifiers={
                "repeat_unit_smiles": cached["repeat_unit_smiles"],
                "canonical_repeat_unit_smiles": cached["canonical_repeat_unit_smiles"],
            },
            metadata=cached,
        )

    @staticmethod
    def _compute_profile(raw_smiles: str) -> dict[str, Any]:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        repeat_smiles = _strip_polymer_markers(raw_smiles)
        molecule = Chem.MolFromSmiles(repeat_smiles)
        if molecule is None:
            raise ValueError(f"Invalid polymer repeat-unit SMILES: {raw_smiles}")
        hydrogen_count = sum(atom.GetTotalNumHs() for atom in molecule.GetAtoms())
        marker_count = len(_DUMMY_PATTERN.findall(raw_smiles))
        exact_mass = rdMolDescriptors.CalcExactMolWt(molecule)
        repeat_mass = max(1.0, exact_mass - 1.00784 * min(marker_count, 2))
        molar_volume = _estimate_molar_volume(molecule, hydrogen_count, marker_count)
        cohesive_energy = _estimate_cohesive_energy(molecule, hydrogen_count)
        solubility_parameter = math.sqrt(max(cohesive_energy, 1.0) / max(molar_volume, 1.0))
        density = repeat_mass / molar_volume
        ring_count = int(rdMolDescriptors.CalcNumRings(molecule))
        aromatic_ring_count = int(rdMolDescriptors.CalcNumAromaticRings(molecule))
        rotatable = int(Lipinski.NumRotatableBonds(molecule))
        hbd = int(Lipinski.NumHDonors(molecule))
        hba = int(Lipinski.NumHAcceptors(molecule))
        hetero_fraction = _hetero_fraction(molecule)
        flexible_fraction = rotatable / max(1, molecule.GetNumHeavyAtoms())
        tg = (
            175.0
            + 4.8 * solubility_parameter
            + 18.0 * ring_count
            + 12.0 * aromatic_ring_count
            + 10.0 * (hbd + min(hba, 4))
            + 65.0 * hetero_fraction
            - 45.0 * flexible_fraction
        )
        tg = max(170.0, min(650.0, tg))
        water_distance = abs(solubility_parameter - _WATER_SOLUBILITY_PARAMETER)
        water_affinity_score = max(0.0, min(1.0, 1.0 - water_distance / 24.0))
        oxygen_permeability_proxy = math.exp(
            0.18 * (23.0 - solubility_parameter)
            + 0.85 * flexible_fraction
            - 0.13 * ring_count
            - 0.08 * (hbd + hba)
        )
        co2_selectivity_proxy = 1.0 + 0.08 * solubility_parameter + 0.22 * hetero_fraction + 0.03 * hba
        return {
            "input_smiles": raw_smiles,
            "repeat_unit_smiles": repeat_smiles,
            "canonical_repeat_unit_smiles": Chem.MolToSmiles(molecule, canonical=True),
            "repeat_unit_exact_mass_da": round(repeat_mass, 6),
            "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
            "molar_volume_estimate_cm3_mol": round(molar_volume, 6),
            "cohesive_energy_estimate_j_mol": round(cohesive_energy, 6),
            "polymer_density_estimate_g_cm3": round(density, 6),
            "polymer_solubility_parameter_mpa05": round(solubility_parameter, 6),
            "polymer_solubility_ratio_to_water": round(solubility_parameter / _WATER_SOLUBILITY_PARAMETER, 6),
            "polymer_water_affinity_score": round(water_affinity_score, 6),
            "polymer_tg_estimate_k": round(tg, 6),
            "polymer_o2_permeability_proxy": round(oxygen_permeability_proxy, 6),
            "polymer_co2_n2_selectivity_proxy": round(co2_selectivity_proxy, 6),
            "logp_repeat_unit_rdkit": round(Crippen.MolLogP(molecule), 6),
            "tpsa_repeat_unit_rdkit": round(rdMolDescriptors.CalcTPSA(molecule), 6),
            "hbond_donor_count": hbd,
            "hbond_acceptor_count": hba,
            "rotatable_bond_count": rotatable,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "hetero_atom_fraction": round(hetero_fraction, 6),
            "fingerprint_density_morgan2": round(Descriptors.FpDensityMorgan2(molecule), 6),
            "matched_polar_groups": _matched_polar_groups(molecule),
            "polymer_marker_count": marker_count,
            "reference_project": "polymer_property_prediction",
        }

    def _results_from_profile(self, raw_smiles: str, payload: dict[str, Any]) -> list[CandidateEvaluationResult]:
        outputs = (
            ("polymer_tg_estimate_k", "K", 0.62),
            ("polymer_density_estimate_g_cm3", "g/cm3", 0.6),
            ("polymer_solubility_parameter_mpa05", "MPa^0.5", 0.58),
            ("polymer_water_affinity_score", None, 0.55),
            ("polymer_o2_permeability_proxy", None, 0.5),
            ("polymer_co2_n2_selectivity_proxy", None, 0.48),
            ("repeat_unit_exact_mass_da", "Da", 0.9),
            ("molar_volume_estimate_cm3_mol", "cm3/mol", 0.55),
        )
        results: list[CandidateEvaluationResult] = []
        for criterion_name, unit, confidence in outputs:
            value = payload.get(criterion_name)
            if value is None:
                continue
            normalized_score = None
            if criterion_name == "polymer_water_affinity_score":
                normalized_score = float(value)
            results.append(
                CandidateEvaluationResult(
                    criterion_name=criterion_name,
                    value=float(value),
                    unit=unit,
                    normalized_score=normalized_score,
                    confidence=confidence,
                    rationale=(
                        f"Estimated from repeat-unit SMILES `{payload['canonical_repeat_unit_smiles']}` using local "
                        "RDKit descriptors and group-contribution heuristics inspired by polymer_property_prediction."
                    ),
                    evidence_mode="local_tool",
                    tool_id=self.spec.skill_id,
                    citation_chunk_ids=[f"polymer-property:{self._cache_key(raw_smiles)}:{criterion_name}"],
                    is_inferred=True,
                )
            )
        return results

    def _evidence_rows(
        self,
        raw_smiles: str,
        candidate_name: str,
        payload: dict[str, Any],
        results: list[CandidateEvaluationResult],
    ) -> list[dict[str, Any]]:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        group_text = ", ".join(payload["matched_polar_groups"]) or "none detected"
        return [
            {
                "id": f"polymer-property:{self._cache_key(raw_smiles)}:{result.criterion_name}",
                "source_url": "local://polymer_property_profile",
                "source_title": "Local polymer property profile",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [candidate_name],
                "relevance_score": 0.82,
                "retrieved_at": retrieved_at,
                "chunk_text": (
                    f"Polymer property profile for {candidate_name} repeat unit "
                    f"{payload['canonical_repeat_unit_smiles']}: {result.criterion_name}={result.value}"
                    f"{f' {result.unit}' if result.unit else ''}. Polar groups: {group_text}. "
                    "Reference project: polymer_property_prediction."
                )[:1800],
                "metadata": {
                    "source_type": "local-tool",
                    "tool_id": self.spec.skill_id,
                    "smiles": payload["canonical_repeat_unit_smiles"],
                    "endpoint_name": result.criterion_name,
                    "value": result.value,
                    "unit": result.unit,
                    "reference_project": "polymer_property_prediction",
                },
            }
            for result in results
        ]

    def _load_cache(self, raw_smiles: str) -> dict[str, Any] | None:
        path = self._cache_path(raw_smiles)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Failed reading polymer property cache file %s", path)
            return None

    def _store_cache(self, raw_smiles: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(raw_smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed writing polymer property cache file %s", path)

    def _cache_path(self, raw_smiles: str) -> Path:
        return self._cache_dir / f"{self._cache_key(raw_smiles)}.json"

    @staticmethod
    def _cache_key(raw_smiles: str) -> str:
        return sha256(raw_smiles.encode("utf-8")).hexdigest()


def _extract_polymer_smiles(context: SkillContext) -> str:
    hypothesis = getattr(context, "hypothesis", None)
    artifact = getattr(hypothesis, "candidate_artifact", {}) or {}
    for key in (
        "repeat_unit_smiles",
        "polymer_repeat_unit_smiles",
        "polymer_smiles",
        "opsin_smiles",
        "monomer_smiles",
    ):
        value = compact_text(artifact.get(key))
        if value:
            return value
    primary_field = str(getattr(getattr(context.document, "candidate_artifact_schema", None), "primary_identifier_field", "") or "")
    if primary_field.lower() in {"repeat_unit_smiles", "polymer_smiles", "opsin_smiles"}:
        return compact_text(artifact.get(primary_field))
    smiles = compact_text(artifact.get("smiles"))
    if _DUMMY_PATTERN.search(smiles):
        return smiles
    return ""


def _strip_polymer_markers(raw_smiles: str) -> str:
    stripped = _DUMMY_PATTERN.sub("", compact_text(raw_smiles))
    stripped = stripped.replace("()", "")
    return stripped


def _estimate_molar_volume(molecule: Any, hydrogen_count: int, marker_count: int) -> float:
    volume = hydrogen_count * _ATOM_MOLAR_VOLUMES_CM3_MOL[1]
    for atom in molecule.GetAtoms():
        volume += _ATOM_MOLAR_VOLUMES_CM3_MOL.get(atom.GetAtomicNum(), 18.0)
    volume -= 2.1 * molecule.GetRingInfo().NumRings()
    volume -= 1.5 * min(marker_count, 2)
    return max(volume, 5.0)


def _estimate_cohesive_energy(molecule: Any, hydrogen_count: int) -> float:
    cohesive = hydrogen_count * _ATOM_COHESIVE_ENERGIES_J_MOL[1]
    for atom in molecule.GetAtoms():
        cohesive += _ATOM_COHESIVE_ENERGIES_J_MOL.get(atom.GetAtomicNum(), 4000.0)
    for _name, smarts, increment in _POLAR_GROUP_SMARTS:
        from rdkit import Chem

        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            cohesive += increment * len(molecule.GetSubstructMatches(pattern))
    aromatic_rings = sum(
        1
        for ring in molecule.GetRingInfo().AtomRings()
        if all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    )
    cohesive += 3500.0 * aromatic_rings
    return max(cohesive, 1.0)


def _matched_polar_groups(molecule: Any) -> list[str]:
    from rdkit import Chem

    matched: list[str] = []
    for name, smarts, _increment in _POLAR_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and molecule.HasSubstructMatch(pattern):
            matched.append(name)
    return matched


def _hetero_fraction(molecule: Any) -> float:
    heavy_atoms = max(1, molecule.GetNumHeavyAtoms())
    hetero_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    return hetero_atoms / heavy_atoms


def _document_text_fields(document: Any) -> list[str]:
    values: list[str] = []
    for field in ("research_mode", "raw_goal", "candidate_origin_policy", "novelty_check_policy"):
        value = getattr(document, field, None)
        if value not in (None, "", []):
            values.append(str(value))
    for field in ("evaluation_criteria", "reflection_guidance", "material_scope", "application_scope"):
        for item in getattr(document, field, []) or []:
            if hasattr(item, "name"):
                values.append(str(getattr(item, "name", "")))
                values.append(str(getattr(item, "description", "")))
            else:
                values.append(str(item))
    return values


def _candidate_label(context: SkillContext) -> str:
    hypothesis = getattr(context, "hypothesis", None)
    artifact = getattr(hypothesis, "candidate_artifact", {}) or {}
    return (
        compact_text(artifact.get("name_or_label"))
        or compact_text(getattr(hypothesis, "candidate_material", None))
        or compact_text(getattr(hypothesis, "title", None))
        or "Polymer candidate"
    )
