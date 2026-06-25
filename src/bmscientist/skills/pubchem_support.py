from __future__ import annotations

import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from bmscientist.skills.molecule_support import MoleculeIdentifiers, compact_text, normalize_cas_number


LOGGER = logging.getLogger(__name__)

PUBCHEM_PUG_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUG_VIEW_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
DEFAULT_PUBCHEM_PROPERTIES = (
    "CanonicalSMILES",
    "IsomericSMILES",
    "MolecularFormula",
    "MolecularWeight",
    "InChI",
    "InChIKey",
    "IUPACName",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
)


class PubChemClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 30,
        cache_dir: Path | None = None,
    ):
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._cache_dir = cache_dir or Path("data") / "skills" / "pubchem"

    def resolve(self, identifiers: MoleculeIdentifiers) -> dict[str, Any]:
        cid = identifiers.cid
        if cid is None:
            cid = self._cid_from_identifiers(identifiers)
        if cid is None:
            return {"identifiers": identifiers.as_prompt_dict(), "cid": None}
        payload = self.properties_for_cid(cid)
        synonyms = self.synonyms_for_cid(cid)
        properties = self._first_property_row(payload)
        return {
            "cid": cid,
            "properties": properties,
            "synonyms": synonyms[:20],
            "identifiers": {
                **identifiers.as_prompt_dict(),
                "canonical_smiles": compact_text(properties.get("CanonicalSMILES")) or identifiers.canonical_smiles,
                "smiles": compact_text(properties.get("IsomericSMILES")) or identifiers.smiles,
                "name": compact_text(properties.get("IUPACName")) or identifiers.name,
                "inchi": compact_text(properties.get("InChI")) or identifiers.inchi,
                "inchikey": compact_text(properties.get("InChIKey")) or identifiers.inchikey,
                "cas_number": self._pick_cas_number(synonyms, identifiers.cas_number),
            },
        }

    def properties_for_cid(self, cid: int, properties: tuple[str, ...] = DEFAULT_PUBCHEM_PROPERTIES) -> dict[str, Any]:
        property_text = ",".join(properties)
        url = f"{PUBCHEM_PUG_BASE_URL}/compound/cid/{cid}/property/{property_text}/JSON"
        return self._cached_json("properties", url)

    def synonyms_for_cid(self, cid: int) -> list[str]:
        url = f"{PUBCHEM_PUG_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        payload = self._cached_json("synonyms", url)
        return [
            str(item).strip()
            for item in payload.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            if str(item).strip()
        ]

    def view_for_cid(self, cid: int) -> dict[str, Any]:
        url = f"{PUBCHEM_PUG_VIEW_BASE_URL}/data/compound/{cid}/JSON"
        return self._cached_json("view", url)

    def sids_for_cid(self, cid: int) -> list[int]:
        url = f"{PUBCHEM_PUG_BASE_URL}/compound/cid/{cid}/sids/JSON"
        payload = self._cached_json("sids", url)
        values = payload.get("InformationList", {}).get("Information", [{}])[0].get("SID", [])
        result: list[int] = []
        for value in values:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    def similar_cids_from_smiles(self, smiles: str, threshold: int = 90, max_records: int = 10) -> list[int]:
        escaped = quote(smiles, safe="")
        url = (
            f"{PUBCHEM_PUG_BASE_URL}/compound/fastsimilarity_2d/smiles/{escaped}/cids/JSON"
            f"?Threshold={int(threshold)}&MaxRecords={int(max_records)}"
        )
        payload = self._cached_json("similarity", url)
        values = payload.get("IdentifierList", {}).get("CID", [])
        result: list[int] = []
        for value in values:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    def related_compounds_from_cid(self, cid: int, max_records: int = 10) -> list[int]:
        # Placeholder for a richer related-compound strategy. The first wave uses explicit
        # similarity search instead of depending on view-record linkage semantics.
        return []

    def _cid_from_identifiers(self, identifiers: MoleculeIdentifiers) -> int | None:
        lookup_order: list[tuple[str, str | None]] = [
            ("smiles", identifiers.canonical_smiles or identifiers.smiles),
            ("inchi", identifiers.inchi if identifiers.inchi and identifiers.inchi.startswith("InChI=") else None),
            ("name", identifiers.cas_number or identifiers.name),
        ]
        for namespace, value in lookup_order:
            if not value:
                continue
            escaped = quote(str(value), safe="")
            url = f"{PUBCHEM_PUG_BASE_URL}/compound/{namespace}/{escaped}/cids/JSON"
            try:
                payload = self._cached_json("cid_lookup", url)
            except Exception:
                LOGGER.exception("PubChem CID lookup failed for %s=%s", namespace, value)
                continue
            candidates = payload.get("IdentifierList", {}).get("CID", [])
            for candidate in candidates:
                try:
                    return int(candidate)
                except (TypeError, ValueError):
                    continue
        return None

    def _cached_json(self, category: str, url: str) -> dict[str, Any]:
        cache_path = self._cache_path(category, url)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("Failed reading PubChem cache file %s", cache_path)
        response = self._session.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Failed writing PubChem cache file %s", cache_path)
        return payload

    def _cache_path(self, category: str, url: str) -> Path:
        key = sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / category / f"{key}.json"

    @staticmethod
    def _first_property_row(payload: dict[str, Any]) -> dict[str, Any]:
        return payload.get("PropertyTable", {}).get("Properties", [{}])[0]

    @staticmethod
    def _pick_cas_number(synonyms: list[str], fallback: str | None) -> str | None:
        values = [normalize_cas_number(item) for item in synonyms]
        values.extend([normalize_cas_number(fallback)])
        for value in values:
            if value:
                return value
        return None


def flatten_pubchem_strings(payload: Any, headings: tuple[str, ...] = ()) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        heading = compact_text(payload.get("TOCHeading"))
        next_headings = headings + ((heading,) if heading else ())
        for info in payload.get("Information", []) or []:
            rows.extend(_extract_information_strings(info, next_headings))
        for section in payload.get("Section", []) or []:
            rows.extend(flatten_pubchem_strings(section, next_headings))
        return rows
    if isinstance(payload, list):
        for item in payload:
            rows.extend(flatten_pubchem_strings(item, headings))
    return rows


def _extract_information_strings(payload: dict[str, Any], headings: tuple[str, ...]) -> list[dict[str, str]]:
    value = payload.get("Value", {})
    strings: list[str] = []
    if isinstance(value, dict):
        if "StringWithMarkup" in value:
            for item in value.get("StringWithMarkup", []) or []:
                text = compact_text(item.get("String"))
                if text:
                    strings.append(text)
        if "String" in value:
            string_value = value.get("String")
            if isinstance(string_value, list):
                strings.extend(compact_text(item) for item in string_value if compact_text(item))
            else:
                text = compact_text(string_value)
                if text:
                    strings.append(text)
        if "Number" in value:
            numbers = value.get("Number")
            if isinstance(numbers, list):
                strings.extend(compact_text(item) for item in numbers if compact_text(item))
            else:
                text = compact_text(numbers)
                if text:
                    strings.append(text)
    return [
        {"heading_path": " > ".join(part for part in headings if part), "text": text}
        for text in strings
        if text
    ]
