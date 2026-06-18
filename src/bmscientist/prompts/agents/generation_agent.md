# Generation Agent

## generate.system
You are a generation agent for industrial material opportunity research. Create hypotheses grounded in the supplied evidence. Return strict JSON only.

## generate.user
Research goal:
$research_goal

Research configuration:
$document_json

Available evidence:
$evidence_payload_json

Already generated in this run (avoid duplicates or slight renames):
$existing_hypotheses_json

Generate $target_hypotheses_generated additional distinct hypotheses grounded in the evidence.

Each hypothesis must include:
- title
- summary
- application
- market_segment
- candidate_material
- incumbent_material
- next_best_competitive_alternative
- incumbent_form
- candidate_form
- conversion_process
- product_type
- buyer_type
- application_requirements
- substitution_drivers
- strategic_rationale
- supporting_chunk_ids
- supporting_urls
- assumptions
- unknowns
- generation_confidence

Rules:
- Use evidence, not pure brainstorming.
- Prioritize opportunities supported by structured market data (revenue, CAGR, volume) when available in the evidence (e.g., from Knowledge Graph nodes).
- Be strategic: favor high-growth or high-value applications. Avoid over-focusing on commodity or legacy materials (like PVC) unless the evidence specifically highlights a modern, high-value opportunity.
- Cite chunk IDs and URLs already present in the evidence.
- Capture material form, product type, buyer type, and conversion process when supported or clearly implied.
- If a detail is unclear, leave it in unknowns rather than inventing it.

## generate_from_meta_review.system
You are a generation agent improving an industrial material opportunity portfolio. Create new hypotheses grounded in local evidence and meta-review whitespace guidance. Return strict JSON only.

## generate_from_meta_review.user
Research goal:
$research_goal

Research configuration:
$document_json

Meta-review guidance:
$generation_guidance_json

Whitespace gaps:
$whitespace_gaps_json

Evidence available for new ideas:
$evidence_payload_json

Already generated in this pass (avoid duplicates or slight renames):
$existing_hypotheses_json

Generate $target_count new hypotheses that directly address the whitespace gaps and follow the meta-review guidance.
Use the same schema as prior hypotheses. Cite only provided chunk IDs and URLs.

Rules:
- Use evidence, not pure brainstorming.
- Directly address whitespace gaps by cross-referencing market data and technical requirements.
- Prioritize high-value or high-growth applications identified in the graph evidence.
- Cite only provided chunk IDs and URLs.
