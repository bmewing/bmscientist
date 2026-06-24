# Changelog

All notable changes to this project will be documented in this file.

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
