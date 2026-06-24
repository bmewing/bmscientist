# bmscientist

See [CHANGELOG.md](C:/Users/bmark/PycharmProjects/bmscientist/CHANGELOG.md) for release notes.

Agentic framework for scientific materials discovery — hypothesis generation, evidence retrieval, and knowledge graph enrichment.

## Setup

Use the local virtual environment for all Python commands on this project:

```powershell
# Install the package in editable mode with development dependencies
.\.venv\Scripts\python.exe -m pip install -e .[dev]

# Run tests
.\.venv\Scripts\python.exe -m pytest
```

Create a `.env` from `.env.example` and add your API keys.

## Environment & Configuration

Create a `.env` file in the project root containing:

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
EXA_API_KEY=your_exa_api_key
HF_TOKEN=your_huggingface_token

# Base directory for all cache and generated outputs (defaults to ./data)
BMSCIENTIST_DATA_DIR=./data

# Optional explicit LanceDB override. If omitted, defaults to BMSCIENTIST_DATA_DIR/lancedb.
# LANCEDB_PATH=./data/lancedb

EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
REQUEST_TIMEOUT_SECONDS=60
SKIP_FETCH_DOMAINS=sciencedirect.com

# Exa retrieval controls
EXA_SEARCH_CONTENT_TEXT_CHARS=8000
EXA_CONTENTS_INITIAL_TEXT_CHARS=12000
EXA_CONTENTS_DEEP_TEXT_CHARS=50000
EXA_HIGHLIGHTS_MAX_CHARS=2000
EXA_ENABLE_SEARCH_CONTENTS=true
EXA_ENABLE_CONTENTS_FOLLOWUP=true
EXA_ENABLE_DIRECT_FETCH_FALLBACK=true
EXA_DEFAULT_SEARCH_TYPE=auto
EXA_REFLECTION_SEARCH_TYPE=fast
EXA_DEFAULT_MAX_AGE_HOURS=168
EXA_NEWS_MAX_AGE_HOURS=24
EXA_DEEP_FETCH_MIN_SCORE=0.78
EXA_DEEP_FETCH_MAX_PER_QUERY=2
EXA_DEEP_FETCH_MAX_PER_RUN=10
EXA_SEARCH_CATEGORY=
EXA_NEWS_DOMAINS=polymart.info

# Simple model-only configuration still works.
CHAT_MODEL=deepseek-v4-flash
GENERATION_CHAT_MODEL=deepseek-v4-pro
REFLECTION_CHAT_MODEL=deepseek-v4-flash
PLANNING_CHAT_MODEL=deepseek-v4-pro
RANKING_CHAT_MODEL=deepseek-v4-pro
EVOLUTION_CHAT_MODEL=deepseek-v4-pro
PROXIMITY_CHAT_MODEL=deepseek-v4-pro
META_REVIEW_CHAT_MODEL=deepseek-v4-pro
MARKET_VOLUME_ESTIMATION_CHAT_MODEL=deepseek-v4-pro

# Optional DeepSeek thinking-mode request profiles.
# These map to:
#   extra_body={"thinking":{"type":"enabled|disabled"}}
#   reasoning_effort="high|max"
#
# If both *_CHAT_MODEL and *_CHAT_PROFILE are present, *_CHAT_PROFILE.model wins.
# timeout_seconds is optional per profile and overrides REQUEST_TIMEOUT_SECONDS for that role.
REFLECTION_CHAT_PROFILE={"model":"deepseek-v4-pro","thinking":{"enabled":true,"effort":"high"},"timeout_seconds":120}
MARKET_VOLUME_ESTIMATION_CHAT_PROFILE={"model":"deepseek-v4-pro","thinking":{"enabled":true,"effort":"max"},"timeout_seconds":180}

# Optional DeepSeek per-model pricing overrides for cost tracking.
# Values are USD per 1M tokens.
# DEEPSEEK_MODEL_PRICING={"deepseek-v4-pro":{"input_cost_per_million_tokens":0.435,"output_cost_per_million_tokens":0.87,"cached_input_cost_per_million_tokens":0.003625}}
```

* `DEEPSEEK_API_KEY` (**Required**): Your DeepSeek API key for all LLM-backed reasoning.
* `EXA_API_KEY` (**Required**): Your Exa search API key for web discovery and page retrieval.
* `EXA_*` retrieval settings (*Optional*): Tune how much Exa text/highlight content is requested at search time, when deeper `/contents` follow-ups happen, how stale cached content can be, and how aggressively direct HTTP fetch is used as a fallback.
* `BMSCIENTIST_DATA_DIR` (*Optional*): Base directory for raw data, caches, graphs, and co-scientist run outputs.
* `LANCEDB_PATH` (*Optional*): Explicit LanceDB directory override. Defaults to `BMSCIENTIST_DATA_DIR/lancedb`.
* `*_CHAT_MODEL` (*Optional*): Simple per-role model override keys. These still work and are the easiest way to route different stages to different DeepSeek models.
* `*_CHAT_PROFILE` (*Optional*): JSON request-profile keys for DeepSeek thinking mode. Use these when you need to control both model selection and request structure such as `thinking.enabled`, `thinking.effort`, and role-specific `timeout_seconds`.
* `DEEPSEEK_MODEL_PRICING` (*Optional*): JSON object of DeepSeek model pricing overrides used by run-level cost tracking. This is useful if provider pricing changes or you want your local `cost.json` totals to reflect a custom rate card.
* `MARKET_VOLUME_ESTIMATION_CHAT_PROFILE` (*Optional*): Useful for the reflection-time AI volume estimator, where a stronger reasoning model can infer tonnage from revenue and pricing signals before writing medium-confidence AI-generated graph evidence.
* `HF_TOKEN` (*Optional*): Hugging Face token, recommended for avoiding rate limits when downloading embedding models.

DeepSeek thinking-mode note:
Use a JSON profile like `{"model":"deepseek-v4-pro","thinking":{"enabled":true,"effort":"max"},"timeout_seconds":180}`. The library maps that to DeepSeek's OpenAI-format request fields (`extra_body.thinking` and `reasoning_effort`) automatically, and `timeout_seconds` overrides the global `REQUEST_TIMEOUT_SECONDS` for that specific role.

Cost tracking note:
Co-scientist runs now write `reports/cost.json` with `total_exa_usd`, `total_deepseek_usd`, and a provider breakdown. DeepSeek totals are computed from API token usage plus the configured pricing table, while Exa totals come from the provider-returned `costDollars.total` field.

Exa retrieval note:
Discovery and reflection search now use Exa search-time extracted content first, preserve Exa highlights and request metadata, and only escalate to `/contents` follow-ups for high-value or truncated pages. Direct HTTP fetch is now a fallback path rather than the default source of truth.


## Usage

Once installed, you can invoke the CLI using the package module syntax:

```powershell
.\.venv\Scripts\python.exe -m bmscientist <command> [arguments]
```

Or, if your virtual environment's `Scripts\` folder is in your PATH, simply:

```powershell
bmscientist <command> [arguments]
```

### 1. Discovery

Run external web discovery, classification, chunking, embedding, and cumulative LanceDB storage:

```powershell
.\.venv\Scripts\python.exe -m bmscientist discover --query "major applications of PVC material and key performance requirements" --max-search-queries 8 --results-per-query 10 --max-pages 30
```

Search local evidence:

```powershell
.\.venv\Scripts\python.exe -m bmscientist search --query "Where is PVC used in clear rigid applications?" --top-k 8
```

Inspect and query the local graph with DuckDB:

```powershell
.\.venv\Scripts\python.exe -m bmscientist graph-schema
.\.venv\Scripts\python.exe -m bmscientist graph-sql --sql "SELECT name, product_id FROM Product ORDER BY name LIMIT 20"
.\.venv\Scripts\python.exe -m bmscientist graph-ask --question "Show me the material grades linked to Tritan"
```

Manually obtained files can be dropped into the configured data directory: `data/manually-obtained/`. Workflows ingest them and move processed files to `data/manually-obtained/processed/`.

### 2. Co-Scientist Workflow

Create a research goal, generate hypotheses from local evidence, and reflect on them immediately:

```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist --goal "I want to find rapid drop-in/drop-out flywheel opportunities for PET against a target material of Styrenics..." --target-hypotheses 25 --regions "North America,Europe" --reflection-concurrency 4 --proximity-merge-mode balanced --proximity-granularity application_family
```

`Proximity Check Agent` tuning now lives on the research goal:
- `--proximity-merge-mode`: `conservative`, `balanced`, or `aggressive`
- `--proximity-granularity`: `device_subtype`, `application_family`, or `global`

Regions are always combined into synthesized hypotheses rather than split into separate regional variants.

The `coscientist` command now defaults to in-process threaded reflection, which keeps RAM usage much lower than spawning multiple background reflector subprocesses. If you explicitly want the legacy subprocess behavior, add:

```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist ... --spawn-reflection-daemons
```

When a run completes, the CLI prints a `Costs:` path pointing to the generated `reports/cost.json` file.

Resume reflection for generated hypotheses without regenerating:

```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist-reflect --research-id YOUR_RESEARCH_ID --concurrency 4
```

Run multiple long-lived reflection workers in parallel against the same research run (using the file-rename queue):

```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist-reflect --research-id YOUR_RESEARCH_ID --daemon --worker-id reflector-a --concurrency 2 --poll-interval-seconds 5
.\.venv\Scripts\python.exe -m bmscientist coscientist-reflect --research-id YOUR_RESEARCH_ID --daemon --worker-id reflector-b --concurrency 2 --poll-interval-seconds 5
```

Each worker atomically claims hypotheses by renaming files from `generated/` to `reflecting/`, then moves completed work into `reflected/`. If a worker dies, the lease expires and another worker automatically reclaims it.

### 3. Iteration Loops

Run the Ranking Agent, Proximity Check Agent, Meta-review Agent, Evolution Agent, feedback generation, and reflection loop over an existing research run:

```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist-loop --research-id YOUR_RESEARCH_ID --target-final-hypotheses 10 --max-rounds 2 --evolve-top-k 5 --evolved-per-round 5 --regenerated-per-round 5 --proximity-check-every 1 --max-synthesized-per-round 3 --max-gap-persistence-rounds 1 --reflection-concurrency 4 --proximity-merge-mode aggressive --proximity-granularity application_family
```

The loop ranks active reflected hypotheses, clusters concepts, synthesizes overlapping ideas, reflects on synthesized variants, and uses meta-review whitespace analysis to control loop continuation. For existing projects, the proximity flags on `coscientist-loop` act as explicit policy overrides.

### 4. Human Feedback & Steering

Provide human feedback on hypotheses (accepting, rejecting, or editing specific fields), update project-level goals, or run explicit meta-reviews to steer generation.

#### Apply feedback/edits to a hypothesis:
```powershell
# Accept a hypothesis with a comment
.\.venv\Scripts\python.exe -m bmscientist coscientist-feedback --research-id YOUR_RESEARCH_ID --hypothesis-id HYPOTHESIS_ID --status accepted --comment "Highly viable regulatory path"

# Reject/retire a hypothesis
.\.venv\Scripts\python.exe -m bmscientist coscientist-feedback --research-id YOUR_RESEARCH_ID --hypothesis-id HYPOTHESIS_ID --status rejected --comment "Material too expensive"

# Edit specific fields on a hypothesis (marks status as 'edited')
.\.venv\Scripts\python.exe -m bmscientist coscientist-feedback --research-id YOUR_RESEARCH_ID --hypothesis-id HYPOTHESIS_ID --title "New Hypothesis Title" --summary "Updated summary of hypothesis"
```

#### Update project direction & auto-re-rank:
Updating the overall project goals/criteria triggers the ranking agent to automatically re-evaluate and re-rank all active hypotheses under the new direction:
```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist-feedback --research-id YOUR_RESEARCH_ID --project-feedback "Shift focus to European regions and prioritize recyclability/circular economy drivers."
```

#### Explicitly trigger a meta-review round:
Explicitly run the Meta-Review Agent to assess portfolio gaps, update whitespace guidance, and evolve high-scoring/accepted hypotheses:
```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist-meta-review --research-id YOUR_RESEARCH_ID --evolve-top-k 5 --evolved-per-round 5
```

## Directory Structure

Under the directory configured by `BMSCIENTIST_DATA_DIR` (defaults to `./data`):

* Vector Database: `lancedb/`
* Graph Database: `graph/`
* Co-scientist Projects: `coscientist/{research_id}/`
  * Research goal: `research_goal.json`
  * Hypotheses: `hypotheses/{generated,reflecting,reflected,evolve,retired}/{hypothesis_id}.json`
  * Execution logs: `rounds/rankings.jsonl`, `rounds/proximity.jsonl`, `rounds/meta_reviews.jsonl`
  * Output Reports: `reports/reflection.md`, `reports/loop.md`, `reports/tool_requests.md`, `reports/cost.json`
* Cache: `raw/`, `pricing/`

For a given `research_id`, the cost report is cumulative across `coscientist`, `coscientist-loop`, and `coscientist-reflect`, so later commands update the same `reports/cost.json` instead of starting a separate spend ledger.

## Persistence & Validation Rules

* Evidence remains cumulative in LanceDB.
* The system preserves source URLs and chunk IDs, avoids invented citations, and keeps confidence scores conservative when evidence is incomplete.
