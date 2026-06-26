# Changelog

All notable changes to this project will be documented in this file.

## 0.9.4

### Fixed
- Tightened meta-review guidance so follow-up work stays grounded in the project target size instead of expanding into oversized top-k lists or unnecessary extra candidate batches.
- Updated shortlist sufficiency framing so a project can stop once the required number of novel, well-supported candidates is covered, even if weaker surplus ideas remain elsewhere in the portfolio.

## 0.9.3

### Fixed
- Hardened research-plan parsing so planner outputs can recover from wrapped payloads, alias-heavy field names, and mildly messy nested criterion/tool definitions instead of failing on strict schema mismatches.
- Added normalization for common planner near-misses such as wrapped `plan`/`updated_plan` payloads, alternate criterion/tool keys, nested ranking-weight objects, and month strings like `"12 months"`.

## 0.9.2

### Fixed
- Removed the legacy `SKIP_FETCH_DOMAINS` direct-fetch denylist so retrieval stays lean and Exa-guided fallback no longer carries stale domain-specific policy.
- Simplified the Exa retrieval fallback path to always attempt direct fetch when configured, while still preserving partial evidence when direct access fails.
- Cleaned the public configuration examples and docs to match the slimmer retrieval model.

## 0.9.1

### Fixed
- Tightened saved `research_goal.json` and hypothesis JSON so candidate-design and molecule-discovery runs no longer persist large amounts of empty or irrelevant materials-opportunity boilerplate by default.
- Made generation and planning prompts more mode-aware so molecule-design workflows stop overproducing substitution/commercial-comparison fields unless those details are actually relevant to the task.
- Updated local hypothesis retrieval to avoid automatically manufacturing price-comparison style search queries for non-`materials_opportunity` research modes.

## 0.9.0

### Added
- Added a phase-aware chemistry skill layer for molecule-first workflows, including PubChem identity resolution, RDKit descriptor profiling, PubChem property profiling, safety triage, availability screening, EPISuite integration, RXN4Chemistry retrosynthesis support, novelty screening, and molecule-neighbor seed expansion.
- Added skill registry aliases, per-skill priorities, deterministic safety gating, and prompt-visible skill catalogs so planning, generation, and reflection can request real capabilities instead of inventing tool names.
- Added graph-enrichment access to the skill layer so discovery-time enrichment can resolve molecule identifiers, derive SMILES-backed properties, and persist product and endpoint facts into the graph.
- Added regression coverage for molecule-skill chaining, graph skill writeback, prompt rendering, and safety-gated chemistry workflows.

### Changed
- Expanded the hybrid agent-plus-skills architecture so planning, generation, reflection, and graph enrichment all share the same typed skill runtime.
- Updated generation and planning prompts to include available skill catalogs and generation-phase molecule seed candidates.
- Updated graph-enrichment prompts so LLM proposal and validation steps can use skill outputs for canonical naming and identifier consistency without bypassing evidence-based relationship validation.
- Refined the active runtime catalog to keep chemistry enrichment focused on PubChem, RDKit, EPISuite, RXN4Chemistry, safety, and availability, while leaving deferred integrations such as ChemSpace pricing out of the live skill set for now.

## 0.8.0

### Added
- Added explicit novelty-intent planning fields for co-scientist candidate-design runs, including `candidate_origin_policy`, `novelty_requirements`, `known_candidate_exclusion_terms`, and `novelty_check_policy`, so the system can distinguish de novo scientific design requests from ordinary substitution searches.

### Changed
- Updated research planning, generation, retrieval, reflection, ranking, and final-portfolio prompts so novel molecule-design goals use evidence as design constraints and benchmarks rather than defaulting to known commercial substitution candidates.

### Fixed
- Fixed co-scientist hypothesis generation for de novo SMILES-style goals so requests like "find brand-new, never-before-seen molecules" no longer return known commercial coalescing aids as final answers, enforce required artifact identifiers such as `smiles`, and dedupe exact structures before reflection.

## 0.7.0

### Added
- Formalized hypothesis-level human feedback states so co-scientist records can carry explicit `accepted`, `rejected`, `edited`, `retired`, and `low_volume` user signals instead of treating feedback as loose comments only.

### Changed
- Updated co-scientist feedback handling so user-edited or user-accepted hypotheses can be revived into the active reflected set and rescored in later `coscientist-loop` passes.
- Expanded meta-review inputs to include feedback history across accepted, rejected, and edited hypotheses, including retired rejected ideas, so whitespace analysis and next-pass generation guidance can respond to user expertise directly.
- Updated regeneration guidance and filtering so previously rejected ideas are treated as "avoid" context and are not reintroduced unless new evidence materially changes the thesis.
- Integration note for web clients such as `ypotheto.com`: edited hypotheses should be treated as candidates for renewed scoring, while rejected hypotheses should remain visible as user feedback constraints that inform meta-review and future generation rather than disappearing from project context.

## 0.6.0

### Added
- Introduced a more flexible research-planning contract so planning agents can define output artifacts, evaluation criteria, reflection guidance, and tool requests for broader scientific discovery tasks beyond simple material/application matching.
- Added graph support for richer hypothesis context, including new node and edge capture for flexible planning outputs and AI-assisted market volume estimates with confidence and provenance metadata.
- Added reflection-time market volume estimation so the system can infer approximate material tonnage from market revenue and pricing signals when graph coverage is missing.
- Added role-specific DeepSeek request profiles, including support for thinking-mode configuration and per-profile request timeouts.
- Added run-level cost tracking for co-scientist workflows, including `reports/cost.json` with cumulative Exa and DeepSeek spend totals plus provider breakdowns across `coscientist`, `coscientist-loop`, and `coscientist-reflect`.

### Changed
- Expanded co-scientist prompts and agent behavior to handle more open-ended formulation and materials-invention goals, including requests for structures like SMILES and tool-gap reporting.
- Switched `coscientist` to default to in-process threaded reflection, reducing RAM usage compared with spawning multiple reflection subprocesses.
- Improved concurrency behavior around reflection workers, including safer queue claiming and better support for parallel processing.
- Updated configuration so model/profile routing can be controlled independently for generation, reflection, planning, ranking, evolution, proximity checks, meta-review, and market-volume estimation.
- Added configurable DeepSeek pricing overrides so local cost reports can track updated or custom model rate cards without code changes.

### Fixed
- Normalized planning output aliases to avoid Pydantic validation failures for legacy or model-generated enum variants.
- Fixed DeepSeek thinking-mode request mapping to align with the provider's documented OpenAI-compatible API shape.
- Fixed LLM profile resolution so a profile's explicit model is not accidentally overridden by the global default model.
- Fixed Exa cost parsing to read the provider's `costDollars.total` response shape correctly when computing run-level spend.
