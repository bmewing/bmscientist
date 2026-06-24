from __future__ import annotations

from bmscientist.graph_enrichment import GraphEnrichmentStore
from bmscientist.material_alias import AliasResolver, normalize_alias_text


def test_normalize_alias_text_collapses_punctuation_and_case():
    assert normalize_alias_text("Acrylonitrile-butadiene-styrene") == "acrylonitrile butadiene styrene"


def test_alias_resolver_matches_material_family_aliases(tmp_path):
    store = GraphEnrichmentStore(tmp_path / "graph")
    family_id = store.ensure_material_family(
        "ABS",
        canonical_name="ABS",
        family_type="polymer",
        aliases=["Acrylonitrile butadiene styrene"],
    )
    store.ensure_material_alias(
        "Acrylonitrile butadiene styrene",
        family_id,
        "MaterialFamily",
        alias_type="chemical_name",
        source_vendor="curated",
    )

    resolution = AliasResolver(tmp_path / "graph").resolve_material("Acrylonitrile-butadiene-styrene")

    assert resolution.status == "exact"
    assert resolution.canonical_node_id == family_id
    assert resolution.canonical_node_label == "MaterialFamily"


def test_alias_resolver_flags_conflicting_aliases(tmp_path):
    store = GraphEnrichmentStore(tmp_path / "graph")
    abs_id = store.ensure_material_family("ABS", canonical_name="ABS", family_type="polymer")
    psa_id = store.ensure_material_family("PSA", canonical_name="PSA", family_type="polymer")
    store.ensure_material_alias("impact grade", abs_id, "MaterialFamily", source_vendor="curated")
    store.ensure_material_alias("impact grade", psa_id, "MaterialFamily", source_vendor="curated")

    resolution = AliasResolver(tmp_path / "graph").resolve_material("impact grade")

    assert resolution.status == "ambiguous"
    assert sorted(resolution.candidate_node_ids) == sorted([abs_id, psa_id])
