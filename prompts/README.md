# Prompt Files

Prompt templates live in `prompts/agents/`.

Each agent has its own Markdown file, and each prompt block is stored under a `## section-name` heading.

Examples:

- `prompts/agents/research_planning_agent.md`
- `prompts/agents/generation_agent.md`
- `prompts/agents/reflection_agent.md`
- `prompts/agents/discovery_agent.md`

Template variables use Python `string.Template` syntax, for example:

- `$research_goal`
- `$document_json`
- `$target_count`

To change an agent's behavior, edit the matching prompt section in its Markdown file. No code changes are needed unless you want to add a brand-new prompt section or a new agent.
