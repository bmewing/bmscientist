# Changelog

All notable changes to this project will be documented in this file.

## 0.6.0

### Added
- Introduced a more flexible research-planning contract so planning agents can define output artifacts, evaluation criteria, reflection guidance, and tool requests for broader scientific discovery tasks beyond simple material/application matching.
- Added graph support for richer hypothesis context, including new node and edge capture for flexible planning outputs and AI-assisted market volume estimates with confidence and provenance metadata.
- Added reflection-time market volume estimation so the system can infer approximate material tonnage from market revenue and pricing signals when graph coverage is missing.
- Added role-specific DeepSeek request profiles, including support for thinking-mode configuration and per-profile request timeouts.

### Changed
- Expanded co-scientist prompts and agent behavior to handle more open-ended formulation and materials-invention goals, including requests for structures like SMILES and tool-gap reporting.
- Switched `coscientist` to default to in-process threaded reflection, reducing RAM usage compared with spawning multiple reflection subprocesses.
- Improved concurrency behavior around reflection workers, including safer queue claiming and better support for parallel processing.
- Updated configuration so model/profile routing can be controlled independently for generation, reflection, planning, ranking, evolution, proximity checks, meta-review, and market-volume estimation.

### Fixed
- Normalized planning output aliases to avoid Pydantic validation failures for legacy or model-generated enum variants.
- Fixed DeepSeek thinking-mode request mapping to align with the provider's documented OpenAI-compatible API shape.
- Fixed LLM profile resolution so a profile's explicit model is not accidentally overridden by the global default model.
