from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec
from bmscientist.skills.molecule_support import compact_text, extract_molecule_identifiers


LOGGER = logging.getLogger(__name__)

MOLTOXPRED_SCREEN_TOOL_ID = "moltoxpred_screen"

_MOLTOXPRED_REPO_URL = "https://github.com/bioinformatics-cdac/MolToxPred"
_MOLTOXPRED_PAPER_URL = "https://doi.org/10.1039/d3ra07322j"

_TOX21_ALERTS: tuple[dict[str, Any], ...] = (
    {
        "name": "aromatic_nitro",
        "smarts": "[N+](=O)[O-]",
        "endpoints": ("genotoxicity", "Ames-like mutagenicity", "Tox21 stress response"),
        "weight": 0.2,
    },
    {
        "name": "aniline_or_arylamine",
        "smarts": "[$([NX3H2][c]),$([NX3H1]([#6])[c])]",
        "endpoints": ("mutagenicity", "skin sensitization", "Tox21 nuclear receptor panel"),
        "weight": 0.16,
    },
    {
        "name": "michael_acceptor",
        "smarts": "[C,c]=[C,c]-[C,S](=O)",
        "endpoints": ("covalent reactivity", "skin sensitization", "oxidative stress"),
        "weight": 0.18,
    },
    {
        "name": "epoxide",
        "smarts": "C1OC1",
        "endpoints": ("alkylating reactivity", "genotoxicity"),
        "weight": 0.18,
    },
    {
        "name": "aldehyde",
        "smarts": "[CX3H1](=O)[#6]",
        "endpoints": ("protein reactivity", "irritation"),
        "weight": 0.11,
    },
    {
        "name": "alkyl_halide",
        "smarts": "[CX4][Cl,Br,I]",
        "endpoints": ("alkylating reactivity", "genotoxicity"),
        "weight": 0.13,
    },
    {
        "name": "hydrazine",
        "smarts": "NN",
        "endpoints": ("hepatotoxicity", "genotoxicity"),
        "weight": 0.16,
    },
    {
        "name": "isocyanate",
        "smarts": "N=C=O",
        "endpoints": ("respiratory sensitization", "skin sensitization"),
        "weight": 0.16,
    },
    {
        "name": "phenolic_bisaryl",
        "smarts": "c1ccc(O)cc1",
        "endpoints": ("endocrine activity screen", "Tox21 nuclear receptor panel"),
        "weight": 0.08,
    },
)


class MolToxPredScreenSkill:
    def __init__(self, config: Any | None = None, *, cache_dir: Path | None = None):
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif config is not None and getattr(config, "data_dir", None) is not None:
            self._cache_dir = Path(config.data_dir) / "skills" / "moltoxpred_screen"
        else:
            self._cache_dir = Path("data") / "skills" / "moltoxpred_screen"
        self._spec = SkillSpec(
            skill_id=MOLTOXPRED_SCREEN_TOOL_ID,
            description=(
                "Run a local small-molecule toxicity screen inspired by MolToxPred: RDKit descriptors, Morgan-style "
                "structure features, and Tox21-like structural alerts. This does not require TensorFlow, PaDEL, Java, "
                "or the MolToxPred trained model files."
            ),
            phases=("reflection", "enrichment"),
            aliases=("toxicity_qsar_screen", "small_molecule_toxicity", "moltoxpred", "tox21_alert_screen"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            required_candidate_fields=("smiles",),
            expected_outputs=(
                "moltoxpred_toxicity_score",
                "moltoxpred_toxicity_label",
                "tox21_structural_alert_count",
                "tox21_alert_endpoint_summary",
            ),
            trigger_keywords=(
                "toxicity",
                "toxic",
                "tox21",
                "mutagenicity",
                "genotoxicity",
                "endocrine",
                "sensitization",
                "safety",
            ),
            provider="python_package",
            priority=25,
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
                notes=["No SMILES string was available for toxicity screening."],
                rationale="MolToxPred-style screening requires a small-molecule SMILES string.",
            )
        cached = self._load_cache(smiles)
        if cached is None:
            cached = self._compute_profile(smiles)
            self._store_cache(smiles, cached)
        results = self._results_from_profile(smiles, cached)
        evidence_rows = self._evidence_rows(smiles, identifiers.name or "Unknown candidate", cached, results)
        notes = [
            "This is a lightweight local screen inspired by MolToxPred; it is not the original trained ensemble.",
            f"Reference repo: {_MOLTOXPRED_REPO_URL}",
        ]
        if cached["matched_alerts"]:
            notes.append("Matched structural alerts: " + ", ".join(item["name"] for item in cached["matched_alerts"]) + ".")
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=notes,
            rationale=(
                "Computed RDKit descriptors and Tox21-like structural alerts using a local MolToxPred-inspired "
                "screen suitable for agent scoring."
            ),
            metadata=cached,
        )

    @staticmethod
    def _compute_profile(smiles: str) -> dict[str, Any]:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES string for toxicity screening: {smiles}")
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        mol_weight = float(rdMolDescriptors.CalcExactMolWt(molecule))
        logp = float(Crippen.MolLogP(molecule))
        tpsa = float(rdMolDescriptors.CalcTPSA(molecule))
        hbd = int(Lipinski.NumHDonors(molecule))
        hba = int(Lipinski.NumHAcceptors(molecule))
        rotatable = int(Lipinski.NumRotatableBonds(molecule))
        aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(molecule))
        halogen_count = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})
        formal_charge_abs = sum(abs(atom.GetFormalCharge()) for atom in molecule.GetAtoms())
        matched_alerts = _matched_alerts(molecule)

        descriptor_pressure = 0.0
        descriptor_pressure += _sigmoid((logp - 3.0) / 1.2) * 0.14
        descriptor_pressure += _sigmoid((mol_weight - 450.0) / 90.0) * 0.07
        descriptor_pressure += _sigmoid((tpsa - 120.0) / 35.0) * 0.05
        descriptor_pressure += min(0.08, 0.018 * aromatic_rings)
        descriptor_pressure += min(0.09, 0.015 * halogen_count)
        descriptor_pressure += min(0.05, 0.015 * formal_charge_abs)
        descriptor_pressure += min(0.04, 0.006 * max(0, hba + hbd - 8))
        descriptor_pressure += min(0.03, 0.004 * max(0, rotatable - 8))

        alert_pressure = 1.0
        for alert in matched_alerts:
            alert_pressure *= 1.0 - float(alert["weight"])
        alert_score = 1.0 - alert_pressure
        toxicity_score = max(0.02, min(0.98, 0.18 + descriptor_pressure + alert_score))
        label = _toxicity_label(toxicity_score, len(matched_alerts))
        endpoint_summary = _endpoint_summary(matched_alerts)
        return {
            "canonical_smiles": canonical_smiles,
            "moltoxpred_toxicity_score": round(toxicity_score, 6),
            "moltoxpred_toxicity_label": label,
            "tox21_structural_alert_count": len(matched_alerts),
            "tox21_alert_endpoint_summary": endpoint_summary,
            "matched_alerts": matched_alerts,
            "descriptor_profile": {
                "molecular_weight_da": round(mol_weight, 6),
                "logp_rdkit": round(logp, 6),
                "tpsa_a2": round(tpsa, 6),
                "hbond_donor_count": hbd,
                "hbond_acceptor_count": hba,
                "rotatable_bond_count": rotatable,
                "aromatic_ring_count": aromatic_rings,
                "halogen_count": halogen_count,
                "formal_charge_abs": formal_charge_abs,
            },
            "reference_repo": _MOLTOXPRED_REPO_URL,
            "reference_paper": _MOLTOXPRED_PAPER_URL,
        }

    def _results_from_profile(self, smiles: str, payload: dict[str, Any]) -> list[CandidateEvaluationResult]:
        toxicity_score = float(payload["moltoxpred_toxicity_score"])
        return [
            CandidateEvaluationResult(
                criterion_name="moltoxpred_toxicity_score",
                value=toxicity_score,
                normalized_score=1.0 - toxicity_score,
                confidence=0.58,
                rationale=(
                    "Estimated with local RDKit descriptors and Tox21-like structural alerts inspired by the "
                    "MolToxPred workflow; lower normalized score means higher predicted concern."
                ),
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_urls=[_MOLTOXPRED_REPO_URL, _MOLTOXPRED_PAPER_URL],
                citation_chunk_ids=[f"moltoxpred-screen:{self._cache_key(smiles)}:toxicity_score"],
                is_inferred=True,
            ),
            CandidateEvaluationResult(
                criterion_name="moltoxpred_toxicity_label",
                value=payload["moltoxpred_toxicity_label"],
                normalized_score=1.0 - toxicity_score,
                confidence=0.56,
                rationale="Bucketed from the local MolToxPred-inspired toxicity score and structural-alert count.",
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_urls=[_MOLTOXPRED_REPO_URL, _MOLTOXPRED_PAPER_URL],
                citation_chunk_ids=[f"moltoxpred-screen:{self._cache_key(smiles)}:toxicity_label"],
                is_inferred=True,
            ),
            CandidateEvaluationResult(
                criterion_name="tox21_structural_alert_count",
                value=float(payload["tox21_structural_alert_count"]),
                confidence=0.82,
                rationale="Counted local SMARTS alerts mapped to Tox21-like endpoint families.",
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_urls=[_MOLTOXPRED_REPO_URL, _MOLTOXPRED_PAPER_URL],
                citation_chunk_ids=[f"moltoxpred-screen:{self._cache_key(smiles)}:alert_count"],
                is_inferred=False,
            ),
            CandidateEvaluationResult(
                criterion_name="tox21_alert_endpoint_summary",
                value=payload["tox21_alert_endpoint_summary"],
                confidence=0.68,
                rationale="Summarized endpoint families associated with matched structural alerts.",
                evidence_mode="local_tool",
                tool_id=self.spec.skill_id,
                citation_urls=[_MOLTOXPRED_REPO_URL, _MOLTOXPRED_PAPER_URL],
                citation_chunk_ids=[f"moltoxpred-screen:{self._cache_key(smiles)}:endpoint_summary"],
                is_inferred=True,
            ),
        ]

    def _evidence_rows(
        self,
        smiles: str,
        candidate_name: str,
        payload: dict[str, Any],
        results: list[CandidateEvaluationResult],
    ) -> list[dict[str, Any]]:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        alert_text = ", ".join(alert["name"] for alert in payload["matched_alerts"]) or "none detected"
        descriptor_text = ", ".join(f"{key}={value}" for key, value in payload["descriptor_profile"].items())
        return [
            {
                "id": f"moltoxpred-screen:{self._cache_key(smiles)}:{result.criterion_name}",
                "source_url": _MOLTOXPRED_REPO_URL,
                "source_title": "MolToxPred-inspired local toxicity screen",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [candidate_name],
                "relevance_score": 0.86,
                "retrieved_at": retrieved_at,
                "chunk_text": (
                    f"MolToxPred-inspired local toxicity screen for {candidate_name} ({payload['canonical_smiles']}): "
                    f"{result.criterion_name}={result.value}. Alerts: {alert_text}. Descriptors: {descriptor_text}."
                )[:1800],
                "metadata": {
                    "source_type": "local-tool",
                    "tool_id": self.spec.skill_id,
                    "smiles": payload["canonical_smiles"],
                    "endpoint_name": result.criterion_name,
                    "value": result.value,
                    "matched_alerts": payload["matched_alerts"],
                    "reference_repo": _MOLTOXPRED_REPO_URL,
                    "reference_paper": _MOLTOXPRED_PAPER_URL,
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
            LOGGER.warning("Failed reading MolToxPred screen cache file %s", path)
            return None

    def _store_cache(self, smiles: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed writing MolToxPred screen cache file %s", path)

    def _cache_path(self, smiles: str) -> Path:
        return self._cache_dir / f"{self._cache_key(smiles)}.json"

    @staticmethod
    def _cache_key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()


def _matched_alerts(molecule: Any) -> list[dict[str, Any]]:
    from rdkit import Chem

    matched: list[dict[str, Any]] = []
    for alert in _TOX21_ALERTS:
        pattern = Chem.MolFromSmarts(alert["smarts"])
        if pattern is not None and molecule.HasSubstructMatch(pattern):
            matched.append(
                {
                    "name": alert["name"],
                    "endpoints": list(alert["endpoints"]),
                    "weight": float(alert["weight"]),
                }
            )
    return matched


def _endpoint_summary(matched_alerts: list[dict[str, Any]]) -> str:
    endpoints: list[str] = []
    seen: set[str] = set()
    for alert in matched_alerts:
        for endpoint in alert["endpoints"]:
            key = endpoint.lower()
            if key not in seen:
                seen.add(key)
                endpoints.append(endpoint)
    return ", ".join(endpoints) if endpoints else "no_local_tox21_alert_endpoints"


def _toxicity_label(score: float, alert_count: int) -> str:
    if score >= 0.68 or alert_count >= 3:
        return "higher_concern"
    if score >= 0.45 or alert_count:
        return "moderate_concern"
    return "lower_concern"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


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
