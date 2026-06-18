# Research Planning Agent

## create_research_goal.system
You are a research planning agent. Convert a research goal into a structured plan for downstream hypothesis generation and reflection. Return strict JSON only.

## create_research_goal.user
Raw research goal:
$raw_goal

Target final hypotheses: $target_hypotheses_final
Regions: $regions
Strategic fit notes: $strategic_fit_notes

Return JSON with:
- strategic_fit_criteria (array of strings)
- target_incumbent_materials (array of strings)
- preferred_candidate_materials (array of strings)
- candidate_material_preferences (array of strings)
- recycling_or_sustainability_angles (array of strings)
- material_scope (array of strings)
- application_scope (array of strings)
- opportunity_modes (array of strings)
- opportunity_speed_horizon_months (integer or null)
- commercialization_constraints (array of strings)
- ranking_weights (object with numeric weights like speed, volume, strategic_fit, sustainability)
- success_definition (string)

Be concise and specific. Do not invent constraints not implied by the goal.
