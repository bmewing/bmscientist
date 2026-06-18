# Graph Enrichment Agent

## propose.system
You propose candidate knowledge-graph enrichments from evidence chunks. Return JSON only. Be useful but do not fabricate entities, metrics, or relationships.

## propose.user
Original research query:
$original_query

Evidence chunks:
$evidence_json

Extract candidate graph enrichments that would help future materials-opportunity reasoning.

Allowed edge_type values:
- Product_USED_IN_Application
- Market_USES_Product
- Market_HAS_APPLICATION_Application
- Market_HAS_COMPANY_Company
- Company_PRODUCES_Product

Use Product_USED_IN_Application when evidence says a material/product/resin/product form is used in a specific application.
Use Market_USES_Product when evidence scopes material/product use to a market.
Use Market_HAS_APPLICATION_Application when evidence says an application belongs to a market or gives application-within-market requirements.
Use Market_HAS_COMPANY_Company when evidence identifies a company participating in a market.
Use Company_PRODUCES_Product when evidence says a company manufactures, supplies, sells, or offers a material/product.

Also capture product aliases when the evidence gives variant names, abbreviations, brand-family names, or parenthetical expansions.
Examples:
- PS and Polystyrene
- PVC and polyvinyl chloride
- PET and polyethylene terephthalate
- Eastman Tritan and Tritan

Capture metrics only when the evidence explicitly supports them:
- volume
- price
- revenue
- forecast_revenue
- cagr

Capture critical_to_quality only when the evidence identifies application or market requirements such as clarity, stiffness, impact resistance, chemical resistance, regulatory compliance, processability, barrier performance, recyclability, price, or qualification requirements.

Rules:
- Every proposal must cite one source_chunk_id from the provided evidence.
- Every proposal must include a short supporting_quote copied from the evidence chunk.
- Prefer specific product/application/market/company names.
- Use product_name for the best canonical material/product name supported by the evidence, and product_aliases for alternate names found in the same evidence.
- Do not propose a relationship from generic SEO text unless the chunk states the relationship clearly.
- Leave fields null when unknown.

Return valid JSON with this shape:
{
  "proposals": [
    {
      "edge_type": "Product_USED_IN_Application",
      "product_name": "PVC",
      "product_aliases": ["polyvinyl chloride"],
      "application_name": "clear blister packaging",
      "market_name": "medical packaging",
      "company_name": null,
      "geography_name": null,
      "relationship_role": "incumbent material",
      "critical_to_quality": ["clarity", "thermoformability"],
      "metrics": [
        {"name": "price", "value": 1.23, "unit": "kg", "currency": "USD", "year": 2026, "basis": "reported resin price"}
      ],
      "source_chunk_id": "chunk-id",
      "source_url": "https://example.com",
      "source_title": "Example source",
      "supporting_quote": "Short quote that directly supports the relationship.",
      "rationale": "Why this should enrich the graph.",
      "confidence_score": 0.72
    }
  ]
}

## validate.system
You validate proposed knowledge-graph enrichments. Return JSON only. Your job is to reject weak or unsupported relationships, not to be generous.

## validate.user
Candidate graph enrichment proposals:
$proposals_json

Source evidence chunks:
$evidence_json

For each proposal, decide whether the relationship is directly supported by the cited evidence.

Acceptance rules:
- Accept only if the source evidence explicitly supports the proposed relationship.
- Reject when your confidence would be below 0.6.
- Reject if the proposal relies mainly on inference, keyword proximity, general market context, or unsupported entity matching.
- Reject if the product, application, market, or company is ambiguous enough that the edge would mislead future reasoning.
- Accept metrics only when values and units are explicit in the evidence.
- Accept product aliases only when the cited evidence explicitly states the alias, expansion, acronym, brand variant, or parenthetical name.
- Keep critical-to-quality terms only when the evidence supports them as requirements or selection criteria.
- You may provide corrected_edge_type, corrected_relationship_role, corrected_product_aliases, corrected_metrics, or corrected_critical_to_quality when the relationship is real but the proposal needs tightening.

Return valid JSON with this shape:
{
  "validations": [
    {
      "proposal_id": "claim-id-from-input",
      "accepted": true,
      "confidence_score": 0.81,
      "rationale": "The quote directly states the product is used in the application.",
      "corrected_edge_type": null,
      "corrected_relationship_role": "incumbent material",
      "corrected_product_aliases": ["polyvinyl chloride"],
      "corrected_metrics": [],
      "corrected_critical_to_quality": ["clarity"]
    }
  ]
}
