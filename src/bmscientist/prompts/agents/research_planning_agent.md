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
- research_mode (one of: materials_opportunity, candidate_design, formulation_design, process_design, literature_map, generic_screening)
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
- candidate_artifact_schema (object with artifact_type, primary_identifier_field, required_fields, optional_fields, validation_rules, examples)
- evaluation_criteria (array of objects with name, description, direction, target_value, weight, required_candidate_fields, evidence_mode, suggested_search_queries, suggested_tool_ids, reflection_guidance, failure_modes)
- reflection_guidance (array of strings)
- tool_requests (array of objects with tool_id, purpose, status, candidate_packages, required_inputs, expected_outputs, installation_notes, execution_notes, validation_examples, limitations)
- search_strategy_notes (array of strings)

Rules:
- Be concise and specific. Do not invent constraints not implied by the goal.
- Infer the right candidate representation for the goal. For molecule discovery or screening, prefer an artifact schema with identifiers such as `smiles`.
- Keep `research_mode` as `materials_opportunity` when the goal is clearly about incumbent-material replacement in applications.
- If the goal is broader, use the most fitting research mode and artifact schema.
- Evaluation criteria should explain what makes a good candidate and what evidence would be convincing.
- Tool requests should be concrete capability requests, not instructions to execute code. Do not assume requested tools are installed.
- Reflection guidance should help downstream reviewers know what to validate, what to falsify, and what missing evidence matters most.

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
- research_mode (one of: materials_opportunity, candidate_design, formulation_design, process_design, literature_map, generic_screening)
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
- candidate_artifact_schema (object with artifact_type, primary_identifier_field, required_fields, optional_fields, validation_rules, examples)
- evaluation_criteria (array of objects with name, description, direction, target_value, weight, required_candidate_fields, evidence_mode, suggested_search_queries, suggested_tool_ids, reflection_guidance, failure_modes)
- reflection_guidance (array of strings)
- tool_requests (array of objects with tool_id, purpose, status, candidate_packages, required_inputs, expected_outputs, installation_notes, execution_notes, validation_examples, limitations)
- search_strategy_notes (array of strings)
- strategic_fit_notes (string or null)

Rules:
- Keep unmodified aspects of the original goals unless they conflict with the new feedback.
- Preserve the existing research mode and generic contract unless the feedback clearly changes what kind of candidates or evaluation logic is needed.
