from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from bmscientist.coscientist_models import CandidateEvaluationResult
from bmscientist.skills.base import SkillContext, SkillRunResult, SkillSpec
from bmscientist.skills.molecule_support import (
    HAZARD_ALERT_SMARTS,
    MoleculeIdentifiers,
    compact_text,
    extract_molecule_identifiers,
)
from bmscientist.skills.pubchem_support import PubChemClient, flatten_pubchem_strings


LOGGER = logging.getLogger(__name__)

PUBCHEM_IDENTITY_TOOL_ID = "molecule_identity_pubchem"
PUBCHEM_PROFILE_TOOL_ID = "pubchem_profile"
MOLECULE_AVAILABILITY_TOOL_ID = "molecule_availability"
SAFETY_TRIAGE_TOOL_ID = "safety_triage"
MOLECULE_PRICING_TOOL_ID = "molecule_pricing_optional"
MOLECULE_NEIGHBOR_EXPANSION_TOOL_ID = "molecule_neighbor_expansion"
NOVELTY_PATENT_TOOL_ID = "novelty_patent_screen"
LITERATURE_ANSWER_TOOL_ID = "literature_answer"


class _BasePubChemSkill:
    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        timeout_seconds = int(getattr(config, "request_timeout_seconds", 30) or 30)
        cache_dir = None
        if config is not None and getattr(config, "data_dir", None) is not None:
            cache_dir = getattr(config, "data_dir") / "skills" / "pubchem"
        self._pubchem = pubchem_client or PubChemClient(timeout_seconds=timeout_seconds, cache_dir=cache_dir)

    @staticmethod
    def _identifier_text(identifiers: dict[str, Any]) -> str:
        for key in ("canonical_smiles", "smiles", "cas_number", "name", "cid"):
            value = identifiers.get(key)
            if value not in (None, "", []):
                return f"{key}={value}"
        return "unknown identifier"

    @staticmethod
    def _requested(context: SkillContext, spec: SkillSpec) -> bool:
        requested = {str(item).strip().lower() for item in context.requested_skill_ids if str(item).strip()}
        return bool(requested & {spec.skill_id, *spec.aliases})


class MoleculeIdentityPubChemSkill(_BasePubChemSkill):
    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        super().__init__(config, pubchem_client=pubchem_client)
        self._spec = SkillSpec(
            skill_id=PUBCHEM_IDENTITY_TOOL_ID,
            description=(
                "Resolve molecule names, CAS numbers, SMILES, and InChI strings through PubChem into canonical "
                "identifiers, CID, synonyms, and canonical SMILES."
            ),
            phases=("reflection", "generation", "enrichment"),
            aliases=("pubchem_identity", "pubchem_lookup", "molecule_identity"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            expected_outputs=("pubchem_identifier_confidence", "pubchem_cid"),
            trigger_keywords=("pubchem", "cid", "identifier", "cas", "canonical smiles"),
            provider="http_api",
            priority=20,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        identifiers = extract_molecule_identifiers(context)
        return bool(identifiers.best_query())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        if self._requested(context, self.spec):
            return True
        document = context.document
        return document.candidate_artifact_schema.primary_identifier_field.strip().lower() in {"smiles", "inchi", "cas_number"}

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        resolved = self._pubchem.resolve(identifiers)
        resolved_identifiers = resolved.get("identifiers", {})
        cid = resolved.get("cid")
        confidence = 0.9 if cid is not None else 0.15
        results = [
            CandidateEvaluationResult(
                criterion_name="pubchem_identifier_confidence",
                value=float(confidence),
                normalized_score=float(confidence),
                confidence=0.88,
                rationale="Estimated from whether PubChem returned a CID and canonicalized identifiers.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=True,
            )
        ]
        if cid is not None:
            results.append(
                CandidateEvaluationResult(
                    criterion_name="pubchem_cid",
                    value=float(cid),
                    confidence=0.96,
                    rationale="Resolved directly from PubChem identifier lookup.",
                    evidence_mode="external_tool",
                    tool_id=self.spec.skill_id,
                    citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                    is_inferred=False,
                )
            )
        notes = []
        if resolved.get("synonyms"):
            notes.append(f"Resolved {min(len(resolved['synonyms']), 20)} PubChem synonyms.")
        candidate_label = compact_text(
            resolved_identifiers.get("name")
            or (context.hypothesis.title if context.hypothesis is not None else "Candidate")
        )
        evidence_rows = [
            {
                "id": f"pubchem-identity:{cid or self._identifier_text(resolved_identifiers)}",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                "source_title": "PubChem identifier lookup",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [candidate_label],
                "relevance_score": 0.91,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "chunk_text": (
                    f"PubChem identity resolution returned CID {cid or 'none'} for {self._identifier_text(resolved_identifiers)}. "
                    f"Canonical SMILES: {resolved_identifiers.get('canonical_smiles') or 'n/a'}. "
                    f"Name: {resolved_identifiers.get('name') or 'n/a'}."
                )[:1800],
                "metadata": {
                    "source_type": "external-tool",
                    "tool_id": self.spec.skill_id,
                    "cid": cid,
                    "resolved_identifiers": resolved_identifiers,
                    "synonyms": resolved.get("synonyms", []),
                },
            }
        ]
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=notes,
            rationale="Resolved molecule identifiers through PubChem PUG endpoints.",
            resolved_identifiers=resolved_identifiers,
            metadata={"cid": cid, "synonyms": resolved.get("synonyms", [])},
        )


class PubChemProfileSkill(_BasePubChemSkill):
    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        super().__init__(config, pubchem_client=pubchem_client)
        self._spec = SkillSpec(
            skill_id=PUBCHEM_PROFILE_TOOL_ID,
            description=(
                "Retrieve PubChem computed properties, synonyms, and view metadata for a molecule, including "
                "MolecularWeight, XLogP, TPSA, InChI, and related profile facts."
            ),
            phases=("reflection", "enrichment"),
            aliases=("pubchem_properties", "pubchem_molecule_profile"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            expected_outputs=(
                "molecular_weight_pubchem",
                "xlogp_pubchem",
                "tpsa_pubchem",
                "rotatable_bond_count_pubchem",
            ),
            trigger_keywords=("pubchem", "molecular weight", "xlogp", "tpsa", "property"),
            provider="http_api",
            priority=40,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        return bool(extract_molecule_identifiers(context).best_query())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        if self._requested(context, self.spec):
            return True
        document = context.document
        keyword_text = " ".join(
            [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
            + [context.purpose]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        resolved = self._pubchem.resolve(identifiers)
        cid = resolved.get("cid")
        if cid is None:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="blocked",
                notes=["PubChem could not resolve the molecule to a CID."],
                rationale="PubChem property profiling requires a resolved PubChem CID.",
                resolved_identifiers=resolved.get("identifiers", {}),
            )
        properties = resolved.get("properties", {})
        outputs = (
            ("molecular_weight_pubchem", properties.get("MolecularWeight"), "Da"),
            ("xlogp_pubchem", properties.get("XLogP"), None),
            ("tpsa_pubchem", properties.get("TPSA"), "A2"),
            ("rotatable_bond_count_pubchem", properties.get("RotatableBondCount"), None),
        )
        results: list[CandidateEvaluationResult] = []
        evidence_rows: list[dict[str, Any]] = []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        for criterion_name, raw_value, unit in outputs:
            if raw_value in (None, "", []):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            result = CandidateEvaluationResult(
                criterion_name=criterion_name,
                value=value,
                unit=unit,
                confidence=0.9,
                rationale=f"Retrieved directly from PubChem properties for CID {cid}.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=False,
            )
            results.append(result)
            evidence_rows.append(
                {
                    "id": f"pubchem-profile:{cid}:{criterion_name}",
                    "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                    "source_title": "PubChem molecule profile",
                    "application": None,
                    "incumbent_material": None,
                    "candidate_materials": [compact_text(resolved["identifiers"].get("name") or f"CID {cid}")],
                    "relevance_score": 0.88,
                    "retrieved_at": retrieved_at,
                    "chunk_text": (
                        f"PubChem property profile for CID {cid}: {criterion_name}={value}"
                        f"{f' {unit}' if unit else ''}. Canonical SMILES: {resolved['identifiers'].get('canonical_smiles') or 'n/a'}."
                    )[:1800],
                    "metadata": {
                        "source_type": "external-tool",
                        "tool_id": self.spec.skill_id,
                        "cid": cid,
                        "endpoint_name": criterion_name,
                        "value": value,
                        "unit": unit,
                    },
                }
            )
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=[f"Retrieved {len(results)} PubChem property signals for CID {cid}."],
            rationale="Retrieved PubChem computed-property data for the resolved molecule.",
            resolved_identifiers=resolved.get("identifiers", {}),
            metadata={"cid": cid, "properties": properties},
        )


class SafetyTriageSkill(_BasePubChemSkill):
    _CONTROL_KEYWORDS = ("controlled substance", "schedule i", "schedule ii", "dea list")
    _EXPLOSIVE_KEYWORDS = ("explosive", "detonation", "shock sensitive", "peroxide", "violent decomposition")

    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        super().__init__(config, pubchem_client=pubchem_client)
        self._spec = SkillSpec(
            skill_id=SAFETY_TRIAGE_TOOL_ID,
            description=(
                "Perform a molecule safety triage using PubChem safety sections and local hazard-pattern heuristics. "
                "Use before retrosynthesis or other synthesis-oriented molecule skills."
            ),
            phases=("reflection", "enrichment"),
            aliases=("molecule_safety", "safety", "safety_summary"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            expected_outputs=("safety_triage_status", "explosive_concern_score", "control_chemical_concern_score"),
            trigger_keywords=("safety", "hazard", "explosive", "controlled", "synthesis"),
            provider="http_api",
            priority=10,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        return bool(extract_molecule_identifiers(context).best_query())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        if self._requested(context, self.spec):
            return True
        document = context.document
        keyword_text = " ".join(
            [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
            + document.reflection_guidance
            + [context.purpose]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        resolved = self._pubchem.resolve(identifiers)
        cid = resolved.get("cid")
        hazard_rows: list[dict[str, str]] = []
        if cid is not None:
            view = self._pubchem.view_for_cid(cid)
            hazard_rows = [
                row
                for row in flatten_pubchem_strings(view.get("Record", view))
                if any(token in row["heading_path"].lower() for token in ("safety", "hazard", "toxicity"))
            ]
        hazard_text = " ".join(row["text"] for row in hazard_rows).lower()
        explosive_score = max(
            [0.0]
            + [score for name, _smarts, score in HAZARD_ALERT_SMARTS if name.replace("_alert", "").split("_")[0] in hazard_text]
        )
        control_score = 0.9 if any(keyword in hazard_text for keyword in self._CONTROL_KEYWORDS) else 0.0
        if any(keyword in hazard_text for keyword in self._EXPLOSIVE_KEYWORDS):
            explosive_score = max(explosive_score, 0.9)
        synthesis_blocked = bool(explosive_score >= 0.85 or control_score >= 0.8)
        status_label = (
            "blocked_high_risk"
            if synthesis_blocked
            else "warning_signals_present"
            if max(explosive_score, control_score) >= 0.45
            else "no_major_alerts_found"
        )
        results = [
            CandidateEvaluationResult(
                criterion_name="safety_triage_status",
                value=status_label,
                normalized_score=0.05 if synthesis_blocked else 0.45 if "warning" in status_label else 0.8,
                confidence=0.78,
                rationale="Derived from PubChem safety/hazard text plus deterministic high-risk keyword checks.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=True,
            ),
            CandidateEvaluationResult(
                criterion_name="explosive_concern_score",
                value=float(explosive_score),
                normalized_score=float(explosive_score),
                confidence=0.8,
                rationale="Heuristic explosive-risk score derived from PubChem safety text and deterministic alert keywords.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=True,
            ),
            CandidateEvaluationResult(
                criterion_name="control_chemical_concern_score",
                value=float(control_score),
                normalized_score=float(control_score),
                confidence=0.7,
                rationale="Keyword-based controlled-chemical concern estimate from PubChem safety text.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=True,
            ),
        ]
        notes = []
        if synthesis_blocked:
            notes.append("Safety triage blocked synthesis-oriented skills for this molecule.")
        if hazard_rows:
            notes.append(f"Reviewed {len(hazard_rows)} PubChem hazard/safety text entries.")
        evidence_rows = [
            {
                "id": f"safety-triage:{cid or compact_text(identifiers.best_query())}",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                "source_title": "PubChem safety triage",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [compact_text(resolved.get("identifiers", {}).get("name") or identifiers.name or "Candidate")],
                "relevance_score": 0.94,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "chunk_text": (
                    f"Safety triage for CID {cid or 'unresolved'} returned status `{status_label}` with explosive score "
                    f"{explosive_score:.2f} and control concern score {control_score:.2f}."
                )[:1800],
                "metadata": {
                    "source_type": "external-tool",
                    "tool_id": self.spec.skill_id,
                    "cid": cid,
                    "synthesis_blocked": synthesis_blocked,
                    "hazard_row_count": len(hazard_rows),
                },
            }
        ]
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=notes,
            rationale="Performed a deterministic PubChem-backed molecule safety triage.",
            resolved_identifiers=resolved.get("identifiers", {}),
            metadata={
                "cid": cid,
                "synthesis_blocked": synthesis_blocked,
                "hazard_row_count": len(hazard_rows),
                "status_label": status_label,
            },
        )


class MoleculeAvailabilitySkill(_BasePubChemSkill):
    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        super().__init__(config, pubchem_client=pubchem_client)
        self._spec = SkillSpec(
            skill_id=MOLECULE_AVAILABILITY_TOOL_ID,
            description=(
                "Estimate whether a molecule appears commercially or experimentally available based on PubChem source "
                "records and vendor-linked metadata. This does not fabricate numeric pricing."
            ),
            phases=("reflection", "enrichment"),
            aliases=("availability", "purchasable_check", "supplier_signal"),
            supported_research_modes=("candidate_design", "generic_screening", "formulation_design"),
            expected_outputs=("molecule_availability_signal", "pubchem_source_record_count"),
            trigger_keywords=("availability", "supplier", "commercial", "purchasable"),
            provider="http_api",
            priority=60,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        return bool(extract_molecule_identifiers(context).best_query())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        if self._requested(context, self.spec):
            return True
        keyword_text = " ".join(
            [criterion.name for criterion in context.document.evaluation_criteria]
            + [criterion.description for criterion in context.document.evaluation_criteria]
            + [context.purpose]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        resolved = self._pubchem.resolve(identifiers)
        cid = resolved.get("cid")
        if cid is None:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="blocked",
                notes=["PubChem could not resolve the molecule to a CID for availability checks."],
                rationale="Availability screening requires a resolved PubChem CID.",
            )
        sids = self._pubchem.sids_for_cid(cid)
        count = len(sids)
        normalized = min(1.0, count / 25.0)
        signal = "likely_purchasable" if count >= 5 else "possibly_known_but_not_vendor_rich" if count > 0 else "no_vendor_signal_found"
        results = [
            CandidateEvaluationResult(
                criterion_name="molecule_availability_signal",
                value=signal,
                normalized_score=normalized,
                confidence=0.76,
                rationale="Estimated from PubChem source-record counts; this is an availability signal, not a price quote.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=True,
            ),
            CandidateEvaluationResult(
                criterion_name="pubchem_source_record_count",
                value=float(count),
                confidence=0.9,
                rationale="Counted directly from PubChem source/depositor records for the resolved CID.",
                evidence_mode="external_tool",
                tool_id=self.spec.skill_id,
                citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
                is_inferred=False,
            ),
        ]
        evidence_rows = [
            {
                "id": f"pubchem-availability:{cid}",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                "source_title": "PubChem availability signal",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [compact_text(resolved["identifiers"].get("name") or f"CID {cid}")],
                "relevance_score": 0.83,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "chunk_text": (
                    f"PubChem availability screen for CID {cid} found {count} source/depositor records, giving signal "
                    f"`{signal}`. No numeric price was inferred."
                )[:1800],
                "metadata": {
                    "source_type": "external-tool",
                    "tool_id": self.spec.skill_id,
                    "cid": cid,
                    "source_record_count": count,
                },
            }
        ]
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=results,
            evidence_rows=evidence_rows,
            notes=["No numeric price was inferred from the PubChem availability check."],
            rationale="Estimated molecule availability from PubChem source-record counts.",
            resolved_identifiers=resolved.get("identifiers", {}),
            metadata={"cid": cid, "source_record_count": count, "signal": signal},
        )


class MoleculePricingOptionalSkill:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        session: requests.Session | None = None,
        base_url: str | None = None,
    ):
        self._api_key = (api_key or os.getenv("CHEMSPACE_API_KEY") or "").strip()
        self._base_url = (base_url or os.getenv("CHEMSPACE_BASE_URL") or "https://api.chem-space.com").rstrip("/")
        self._session = session or requests.Session()
        self._token: str | None = None
        self._spec = SkillSpec(
            skill_id=MOLECULE_PRICING_TOOL_ID,
            description=(
                "Fetch optional freemium molecule purchase pricing from ChemSpace when an API key is configured. "
                "Use for quote-style pricing, not free public pricing."
            ),
            phases=("reflection",),
            aliases=("chemspace_price", "molecule_price", "price_quote"),
            supported_research_modes=("candidate_design", "generic_screening"),
            expected_outputs=("molecule_price_quote_usd",),
            trigger_keywords=("price", "quote", "vendor", "chemspace"),
            provider="http_api",
            priority=95,
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
        if self._requested(context, self.spec):
            return True
        keyword_text = " ".join(
            [criterion.name for criterion in context.document.evaluation_criteria]
            + [criterion.description for criterion in context.document.evaluation_criteria]
            + [context.purpose]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        if not self._api_key:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="blocked",
                notes=["Set CHEMSPACE_API_KEY to enable optional molecule pricing quotes."],
                rationale="ChemSpace pricing is optional and requires a configured API key.",
            )
        identifiers = extract_molecule_identifiers(context)
        smiles = identifiers.canonical_smiles or identifiers.smiles
        if not smiles:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="skipped",
                notes=["No SMILES string was available for ChemSpace price lookup."],
                rationale="ChemSpace price lookup requires a SMILES string.",
            )
        quote = self._fetch_cheapest_quote(smiles)
        if quote is None:
            return SkillRunResult(
                skill_id=self.spec.skill_id,
                status="completed",
                notes=["ChemSpace returned no exact quote; price remains unknown."],
                rationale="Executed optional ChemSpace price lookup but found no quote.",
                metadata={"smiles": smiles, "quote_found": False},
            )
        result = CandidateEvaluationResult(
            criterion_name="molecule_price_quote_usd",
            value=float(quote["price_usd"]),
            unit="USD",
            confidence=0.82,
            rationale=(
                f"Cheapest ChemSpace quote was {quote['price_usd']} USD for {quote['quantity']} from {quote['vendor']}."
            ),
            evidence_mode="external_tool",
            tool_id=self.spec.skill_id,
            citation_urls=[self._base_url],
            is_inferred=False,
        )
        evidence_row = {
            "id": f"chemspace-price:{smiles}",
            "source_url": self._base_url,
            "source_title": "ChemSpace price quote",
            "application": None,
            "incumbent_material": None,
            "candidate_materials": [identifiers.name or smiles],
            "relevance_score": 0.82,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "chunk_text": (
                f"ChemSpace quote for {identifiers.name or smiles}: {quote['quantity']} costs {quote['price_usd']} USD "
                f"from {quote['vendor']} and ships within {quote['ships_within_days']} days."
            )[:1800],
            "metadata": {
                "source_type": "external-tool",
                "tool_id": self.spec.skill_id,
                "smiles": smiles,
                "price_usd": quote["price_usd"],
                "quantity": quote["quantity"],
                "vendor": quote["vendor"],
                "ships_within_days": quote["ships_within_days"],
            },
        }
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=[result],
            evidence_rows=[evidence_row],
            notes=["Optional ChemSpace quote retrieved."],
            rationale="Retrieved the cheapest exact ChemSpace quote for the candidate SMILES.",
            metadata={"smiles": smiles, "quote_found": True, "quote": quote},
        )

    def _fetch_cheapest_quote(self, smiles: str) -> dict[str, Any] | None:
        token = self._token or self._renew_token()
        response = self._session.post(
            f"{self._base_url}/v3/search/exact?count=1&page=1&categories=CSMB,CSSB",
            headers={
                "Accept": "application/json; version=3.1",
                "Authorization": f"Bearer {token}",
            },
            data={"SMILES": smiles},
            timeout=30,
        )
        payload = response.json()
        if payload.get("message") == "Your request was made with invalid credentials.":
            token = self._renew_token(force=True)
            response = self._session.post(
                f"{self._base_url}/v3/search/exact?count=1&page=1&categories=CSMB,CSSB",
                headers={
                    "Accept": "application/json; version=3.1",
                    "Authorization": f"Bearer {token}",
                },
                data={"SMILES": smiles},
                timeout=30,
            )
            payload = response.json()
        if int(payload.get("count", 0) or 0) <= 0:
            return None
        cheapest: dict[str, Any] | None = None
        for item in payload.get("items", []):
            for offer in item.get("offers", []):
                for price in offer.get("prices", []):
                    try:
                        price_usd = float(price.get("priceUsd"))
                    except (TypeError, ValueError):
                        continue
                    quote = {
                        "price_usd": price_usd,
                        "quantity": f"{price.get('pack')}{price.get('uom')}",
                        "vendor": price.get("vendorName") or offer.get("vendorName") or "unknown vendor",
                        "ships_within_days": offer.get("shipsWithin"),
                    }
                    if cheapest is None or quote["price_usd"] < cheapest["price_usd"]:
                        cheapest = quote
        return cheapest

    def _renew_token(self, force: bool = False) -> str:
        if self._token and not force:
            return self._token
        response = self._session.get(
            f"{self._base_url}/auth/token",
            headers={"Accept": "application/json", "Authorization": f"Bearer {self._api_key}"},
            timeout=30,
        )
        payload = response.json()
        token = compact_text(payload.get("access_token"))
        if not token:
            raise RuntimeError("ChemSpace token request did not return an access token.")
        self._token = token
        return token


class MoleculeNeighborExpansionSkill(_BasePubChemSkill):
    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        super().__init__(config, pubchem_client=pubchem_client)
        self._spec = SkillSpec(
            skill_id=MOLECULE_NEIGHBOR_EXPANSION_TOOL_ID,
            description=(
                "Generate molecule analog seed candidates for generation-time use by expanding known benchmark or "
                "exclusion molecules through PubChem identity and 2D similarity."
            ),
            phases=("generation",),
            aliases=("molecule_analogs", "analog_expansion", "pubchem_neighbors"),
            supported_research_modes=("candidate_design", "generic_screening"),
            expected_outputs=("generation_seed_candidates",),
            trigger_keywords=("analog", "neighbor", "similarity", "seed"),
            provider="http_api",
            priority=25,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        document = context.document
        seeds = self._seed_terms(document)
        return context.phase == "generation" and bool(seeds) and document.candidate_artifact_schema.primary_identifier_field.strip().lower() == "smiles"

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        if self._requested(context, self.spec):
            return True
        document = context.document
        return document.candidate_origin_policy in {"novel_candidates", "novel_analogs", "de_novo_design"}

    def run(self, context: SkillContext) -> SkillRunResult:
        document = context.document
        target_count = max(1, context.target_count or 3)
        seeds = self._seed_terms(document)
        seed_candidates: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        for seed in seeds[:5]:
            identifiers = MoleculeIdentifiers(name=seed)
            resolved = self._pubchem.resolve(identifiers)
            cid = resolved.get("cid")
            query_smiles = compact_text(resolved.get("identifiers", {}).get("canonical_smiles") or resolved.get("identifiers", {}).get("smiles"))
            if cid is None or not query_smiles:
                continue
            similar_cids = self._pubchem.similar_cids_from_smiles(query_smiles, threshold=85, max_records=max(target_count * 2, 6))
            for similar_cid in similar_cids:
                if similar_cid == cid:
                    continue
                similar_resolved = self._pubchem.resolve(MoleculeIdentifiers(cid=similar_cid))
                similar_identifiers = similar_resolved.get("identifiers", {})
                candidate_smiles = compact_text(similar_identifiers.get("canonical_smiles") or similar_identifiers.get("smiles"))
                if not candidate_smiles:
                    continue
                seed_candidates.append(
                    {
                        "title": f"Analog seed from {seed}",
                        "candidate_artifact": {
                            "name_or_label": compact_text(similar_identifiers.get("name")) or f"CID {similar_cid}",
                            "smiles": candidate_smiles,
                            "pubchem_cid": similar_cid,
                            "seed_origin": seed,
                        },
                        "rationale": f"PubChem 2D similarity neighbor of benchmark `{seed}` (CID {cid}).",
                        "source_skill_id": self.spec.skill_id,
                    }
                )
                evidence_rows.append(
                    {
                        "id": f"pubchem-neighbor:{cid}:{similar_cid}",
                        "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                        "source_title": "PubChem analog seed",
                        "application": None,
                        "incumbent_material": None,
                        "candidate_materials": [compact_text(similar_identifiers.get("name") or f"CID {similar_cid}")],
                        "relevance_score": 0.79,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "chunk_text": (
                            f"PubChem analog seed: CID {similar_cid} ({candidate_smiles}) was retrieved as a 2D-similarity "
                            f"neighbor of benchmark `{seed}` (CID {cid})."
                        )[:1800],
                        "metadata": {
                            "source_type": "external-tool",
                            "tool_id": self.spec.skill_id,
                            "seed_origin": seed,
                            "source_cid": cid,
                            "neighbor_cid": similar_cid,
                        },
                    }
                )
                if len(seed_candidates) >= max(target_count * 3, 6):
                    break
            if len(seed_candidates) >= max(target_count * 3, 6):
                break
        deduped: list[dict[str, Any]] = []
        seen_smiles: set[str] = set()
        for seed in seed_candidates:
            smiles = compact_text(seed.get("candidate_artifact", {}).get("smiles"))
            if not smiles or smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            deduped.append(seed)
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            evidence_rows=evidence_rows[: max(target_count * 3, 6)],
            notes=[f"Generated {len(deduped)} PubChem analog seed candidates."],
            rationale="Expanded benchmark molecules into generation-time analog seeds using PubChem 2D similarity.",
            seed_candidates=deduped[: max(target_count * 3, 6)],
            metadata={"seed_terms": seeds},
        )

    @staticmethod
    def _seed_terms(document: Any) -> list[str]:
        values = []
        for item in (
            *document.known_candidate_exclusion_terms,
            *document.preferred_candidate_materials,
            *document.target_incumbent_materials,
        ):
            text = compact_text(item)
            if text:
                values.append(text)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(value)
        return deduped


class NoveltyPatentScreenSkill(_BasePubChemSkill):
    def __init__(self, config: Any | None = None, *, pubchem_client: PubChemClient | None = None):
        super().__init__(config, pubchem_client=pubchem_client)
        self._spec = SkillSpec(
            skill_id=NOVELTY_PATENT_TOOL_ID,
            description=(
                "Perform a free, approximate novelty screen based on PubChem identity resolution and source-record "
                "presence. This is not a legal patent opinion."
            ),
            phases=("reflection",),
            aliases=("novelty_screen", "patent_screen", "novelty_check"),
            supported_research_modes=("candidate_design", "generic_screening"),
            expected_outputs=("novelty_patent_screen",),
            trigger_keywords=("novelty", "patent", "known substance", "commercial"),
            provider="http_api",
            priority=70,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        return bool(extract_molecule_identifiers(context).best_query())

    def should_run(self, context: SkillContext) -> bool:
        if not self.is_applicable(context):
            return False
        if self._requested(context, self.spec):
            return True
        document = context.document
        keyword_text = " ".join(
            [document.novelty_check_policy, context.purpose]
            + [criterion.name for criterion in document.evaluation_criteria]
            + [criterion.description for criterion in document.evaluation_criteria]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords) or document.candidate_origin_policy in {
            "novel_candidates",
            "novel_analogs",
            "de_novo_design",
        }

    def run(self, context: SkillContext) -> SkillRunResult:
        identifiers = extract_molecule_identifiers(context)
        resolved = self._pubchem.resolve(identifiers)
        cid = resolved.get("cid")
        score = 0.3
        label = "unresolved"
        notes: list[str] = [
            "This is a free, approximate novelty screen and not a legal patent determination."
        ]
        if cid is None:
            score = 0.72
            label = "possibly_novel_unresolved"
            notes.append("PubChem did not resolve the candidate to a CID.")
        else:
            source_count = len(self._pubchem.sids_for_cid(cid))
            if source_count >= 5:
                score = 0.12
                label = "likely_known_and_commercially_disclosed"
                notes.append(f"PubChem returned CID {cid} with {source_count} source records.")
            elif source_count > 0:
                score = 0.28
                label = "known_compound_limited_vendor_signal"
                notes.append(f"PubChem returned CID {cid} with {source_count} source records.")
            else:
                score = 0.45
                label = "known_compound_no_vendor_signal"
                notes.append(f"PubChem returned CID {cid} but no source records were found.")
        result = CandidateEvaluationResult(
            criterion_name="novelty_patent_screen",
            value=label,
            normalized_score=score,
            confidence=0.68,
            rationale="Approximate novelty score derived from PubChem resolution and source-record presence only.",
            evidence_mode="external_tool",
            tool_id=self.spec.skill_id,
            citation_urls=["https://pubchem.ncbi.nlm.nih.gov"],
            is_inferred=True,
        )
        evidence_rows = [
            {
                "id": f"novelty-screen:{cid or compact_text(identifiers.best_query())}",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                "source_title": "PubChem novelty screen",
                "application": None,
                "incumbent_material": None,
                "candidate_materials": [compact_text(resolved.get("identifiers", {}).get("name") or identifiers.name or "Candidate")],
                "relevance_score": 0.8,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "chunk_text": (
                    f"Approximate novelty screen labeled the molecule `{label}` with score {score:.2f}. "
                    f"CID resolved: {cid or 'none'}."
                )[:1800],
                "metadata": {
                    "source_type": "external-tool",
                    "tool_id": self.spec.skill_id,
                    "cid": cid,
                    "novelty_label": label,
                    "novelty_score": score,
                },
            }
        ]
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status="completed",
            criterion_results=[result],
            evidence_rows=evidence_rows,
            notes=notes,
            rationale="Performed a free approximate novelty screen using PubChem identity and source-record data.",
            resolved_identifiers=resolved.get("identifiers", {}),
            metadata={"cid": cid, "novelty_label": label, "novelty_score": score},
        )


class LiteratureAnswerSkill:
    def __init__(self, *, answer_builder: Callable[[SkillContext], dict[str, Any]] | None = None):
        self._answer_builder = answer_builder
        self._spec = SkillSpec(
            skill_id=LITERATURE_ANSWER_TOOL_ID,
            description=(
                "Answer molecule-specific literature questions using existing retrieval evidence, with an optional "
                "provider hook for custom literature backends."
            ),
            phases=("reflection",),
            aliases=("literature", "paper_answer"),
            supported_research_modes=("candidate_design", "generic_screening", "literature_map"),
            expected_outputs=("literature_support_signal",),
            trigger_keywords=("paper", "literature", "publication", "study"),
            provider="local_retrieval",
            priority=75,
        )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def is_applicable(self, context: SkillContext) -> bool:
        return bool(context.evidence_rows or context.question_text or extract_molecule_identifiers(context).best_query())

    def should_run(self, context: SkillContext) -> bool:
        if self._requested(context, self.spec):
            return True
        keyword_text = " ".join(
            [context.question_text, context.purpose, context.document.research_mode]
            + [criterion.name for criterion in context.document.evaluation_criteria]
            + [criterion.description for criterion in context.document.evaluation_criteria]
        ).lower()
        return any(token in keyword_text for token in self.spec.trigger_keywords)

    def run(self, context: SkillContext) -> SkillRunResult:
        if self._answer_builder is not None:
            payload = self._answer_builder(context)
            status = "completed"
            notes = payload.get("notes", [])
            rationale = payload.get("rationale", "Built from a configured literature-answer provider.")
            result_value = payload.get("support_signal", 0.5)
        else:
            relevant_rows = [
                row
                for row in context.evidence_rows
                if row.get("source_url") or row.get("chunk_text")
            ]
            status = "completed" if relevant_rows else "blocked"
            notes = (
                [f"Used {len(relevant_rows)} existing evidence rows as literature context."]
                if relevant_rows
                else ["No retrieval evidence was available for a literature-backed answer."]
            )
            rationale = "Used existing retrieval evidence as a lightweight literature-answer backend."
            result_value = min(1.0, len(relevant_rows) / 8.0) if relevant_rows else 0.0
        result = CandidateEvaluationResult(
            criterion_name="literature_support_signal",
            value=float(result_value),
            normalized_score=float(result_value),
            confidence=0.65,
            rationale=rationale,
            evidence_mode="literature",
            tool_id=self.spec.skill_id,
            is_inferred=True,
        )
        return SkillRunResult(
            skill_id=self.spec.skill_id,
            status=status,
            criterion_results=[result] if status == "completed" else [],
            notes=notes,
            rationale=rationale,
            metadata={"provider": "custom_answer_builder" if self._answer_builder is not None else "local_retrieval"},
        )
