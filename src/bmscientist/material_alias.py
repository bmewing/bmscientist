from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from bmscientist.graph_enrichment import (
    MATERIAL_ALIAS_NODE_SCHEMA,
    NODE_SCHEMAS,
    read_rows,
)
from bmscientist.models import AliasResolution


def normalize_alias_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().replace("/", " ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


class AliasResolver:
    def __init__(self, graph_path: Path | None = None):
        base_path = Path("data/graph") if graph_path is None else Path(graph_path)
        self._nodes_path = base_path / "nodes"

    def resolve_material(self, query: str) -> AliasResolution:
        normalized_query = normalize_alias_text(query)
        if not normalized_query:
            return AliasResolution(status="missing")

        accepted_alias_rows = [
            row
            for row in read_rows(self._nodes_path / "MaterialAlias.parquet", MATERIAL_ALIAS_NODE_SCHEMA)
            if str(row.get("validation_status") or "accepted") == "accepted"
        ]

        exact_matches = [
            row for row in accepted_alias_rows if normalize_alias_text(str(row.get("alias_text") or "")) == normalized_query
        ]
        if len(exact_matches) == 1:
            row = exact_matches[0]
            return AliasResolution(
                status="exact",
                canonical_node_id=row.get("canonical_node_id"),
                canonical_node_label=row.get("canonical_node_label"),
                matched_alias=row.get("alias_text"),
                matched_by="material_alias",
            )
        if len(exact_matches) > 1:
            return AliasResolution(
                status="ambiguous",
                candidate_node_ids=sorted({str(row.get("canonical_node_id")) for row in exact_matches if row.get("canonical_node_id")}),
            )

        family_rows = read_rows(self._nodes_path / "MaterialFamily.parquet", NODE_SCHEMAS["MaterialFamily"])
        matches: list[tuple[str, str]] = []
        for row in family_rows:
            material_family_id = str(row.get("material_family_id") or "")
            if not material_family_id:
                continue
            names_to_check = [row.get("name"), row.get("canonical_name"), row.get("normalized_name")]
            names_to_check.extend(_parse_json_list(row.get("aliases_json")))
            normalized_candidates = {normalize_alias_text(str(item or "")) for item in names_to_check if item}
            if normalized_query in normalized_candidates:
                matches.append((material_family_id, str(row.get("name") or "")))

        if len(matches) == 1:
            return AliasResolution(
                status="normalized",
                canonical_node_id=matches[0][0],
                canonical_node_label="MaterialFamily",
                matched_alias=matches[0][1],
                matched_by="material_family",
            )
        if len(matches) > 1:
            return AliasResolution(
                status="ambiguous",
                candidate_node_ids=sorted({match[0] for match in matches}),
            )
        return AliasResolution(status="missing")


def _parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]
