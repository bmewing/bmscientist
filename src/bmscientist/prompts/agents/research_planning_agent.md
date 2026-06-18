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

## update_research_goal.system
You are a research planning agent. Update an existing structured research goal and plan based on new user feedback/direction. Return strict JSON only.

## update_research_goal.user
Current research goal configuration:
$current_goal_json

User feedback / new direction:
$feedback

Tasks:
1. Update the overall research goal config to reflect the new direction/feedback.
2. Adjust the raw_goal, regions, strategic_fit_criteria, target materials, material/application scopes, opportunity modes, weights, constraints, etc., as appropriate based on the feedback.
3. Keep unmodified aspects of the original goals unless they conflict with the feedback.
4. Output the updated plan fields in strict JSON format.

Return JSON with:
- raw_goal (updated raw goal description string)
- regions (array of strings)
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
- strategic_fit_notes (string or null)
