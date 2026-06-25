from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")
INCHI_PREFIX = "InChI="

FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str], ...] = (
    ("alcohol", "[OX2H]"),
    ("phenol", "c[OX2H]"),
    ("amine", "[NX3;H2,H1;!$(NC=O)]"),
    ("amide", "C(=O)N"),
    ("ester", "[#6][CX3](=O)[OX2H0][#6]"),
    ("carboxylic_acid", "C(=O)[OX2H1]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("nitrile", "[CX2]#N"),
    ("nitro", "[NX3](=O)=O"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("halogen", "[F,Cl,Br,I]"),
)

HAZARD_ALERT_SMARTS: tuple[tuple[str, str, float], ...] = (
    ("nitro_group_alert", "[NX3](=O)=O", 0.62),
    ("organic_peroxide_alert", "OO", 0.88),
    ("azide_alert", "[N-]=[N+]=N", 0.9),
    ("diazo_alert", "[N]=[N]", 0.76),
    ("hydrazine_alert", "NN", 0.68),
)


@dataclass(frozen=True)
class MoleculeIdentifiers:
    smiles: str | None = None
    canonical_smiles: str | None = None
    name: str | None = None
    cas_number: str | None = None
    cid: int | None = None
    inchi: str | None = None
    inchikey: str | None = None

    def best_query(self) -> str | None:
        for value in (
            self.cid,
            self.canonical_smiles,
            self.smiles,
            self.inchi,
            self.cas_number,
            self.name,
        ):
            if value not in (None, "", []):
                return str(value)
        return None

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "name": self.name,
            "cas_number": self.cas_number,
            "cid": self.cid,
            "inchi": self.inchi,
            "inchikey": self.inchikey,
        }


def compact_text(value: Any) -> str:
    if value in (None, "", []):
        return ""
    return " ".join(str(value).split())


def normalize_cas_number(value: Any) -> str | None:
    text = compact_text(value)
    if not text:
        return None
    return text if CAS_PATTERN.match(text) else None


def is_inchi(value: Any) -> bool:
    return compact_text(value).startswith(INCHI_PREFIX)


def extract_molecule_identifiers(context: Any) -> MoleculeIdentifiers:
    hypothesis = getattr(context, "hypothesis", None)
    artifact = getattr(hypothesis, "candidate_artifact", {}) or {}
    document = getattr(context, "document", None)
    primary_field = ""
    if document is not None:
        primary_field = str(document.candidate_artifact_schema.primary_identifier_field or "").strip()

    name = first_present(
        artifact,
        "name_or_label",
        "trade_name",
        "name",
        "candidate_material",
    )
    primary_value = artifact.get(primary_field) if primary_field else None
    smiles = first_nonempty(
        artifact.get("canonical_smiles"),
        artifact.get("smiles"),
        primary_value if primary_field.lower() == "smiles" else None,
    )
    inchi = first_nonempty(
        artifact.get("inchi"),
        primary_value if primary_field.lower() == "inchi" else None,
    )
    cas_number = first_nonempty(
        normalize_cas_number(artifact.get("cas_number")),
        normalize_cas_number(artifact.get("cas")),
        normalize_cas_number(primary_value if primary_field.lower() == "cas_number" else None),
    )
    cid_value = first_nonempty(artifact.get("cid"), artifact.get("pubchem_cid"))
    cid = None
    if cid_value not in (None, "", []):
        try:
            cid = int(str(cid_value).strip())
        except ValueError:
            cid = None
    if not name:
        name = first_nonempty(getattr(hypothesis, "candidate_material", None), getattr(hypothesis, "title", None))
    return MoleculeIdentifiers(
        smiles=compact_text(smiles) or None,
        canonical_smiles=compact_text(artifact.get("canonical_smiles")) or None,
        name=compact_text(name) or None,
        cas_number=cas_number,
        cid=cid,
        inchi=compact_text(inchi) or None,
        inchikey=compact_text(artifact.get("inchikey")) or None,
    )


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None
