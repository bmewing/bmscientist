# bmscientist

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

# Optional: Override LanceDB path (defaults to BMSCIENTIST_DATA_DIR/lancedb)
LANCEDB_PATH=

EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
REQUEST_TIMEOUT_SECONDS=60
SKIP_FETCH_DOMAINS=sciencedirect.com
CHAT_MODEL=deepseek-v4-flash
GENERATION_CHAT_MODEL=deepseek-v4-pro
REFLECTION_CHAT_MODEL=deepseek-v4-flash
PLANNING_CHAT_MODEL=deepseek-v4-pro
RANKING_CHAT_MODEL=deepseek-v4-pro
EVOLUTION_CHAT_MODEL=deepseek-v4-pro
PROXIMITY_CHAT_MODEL=deepseek-v4-pro
META_REVIEW_CHAT_MODEL=deepseek-v4-pro
```

* `DEEPSEEK_API_KEY` (**Required**): Your DeepSeek API key (used for all LLM reasoning, planning, reflection, and evolution).
* `EXA_API_KEY` (**Required**): Your Exa search API key (used for web search and page retrieval).
* `BMSCIENTIST_DATA_DIR` (*Optional*): Configures the base directory where raw data, cache files, knowledge graphs, and generated co-scientist run logs are saved (defaults to `./data`).
* `LANCEDB_PATH` (*Optional*): Explicit override for the LanceDB directory. If left blank or omitted, it defaults to `BMSCIENTIST_DATA_DIR/lancedb`.
* `CHAT_MODEL` (*Optional*): Default LLM to use (defaults to `deepseek-v4-flash`). Agent-specific model overrides (e.g., `GENERATION_CHAT_MODEL`, `EVOLUTION_CHAT_MODEL`) allow routing reasoning-heavy tasks to more capable models like `deepseek-v4-pro`.
* `HF_TOKEN` (*Optional*): Hugging Face token, recommended for avoiding rate limits when downloading embedding models.


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

Manually obtained files can be dropped into the configured data directory: `data/manually-obtained/`. Workflows ingest them and move processed files to `data/manually-obtained/processed/`.

### 2. Co-Scientist Workflow

Create a research goal, generate hypotheses from local evidence, and reflect on them immediately:

```powershell
.\.venv\Scripts\python.exe -m bmscientist coscientist --goal "I want to find rapid drop-in/drop-out flywheel opportunities for PET against a target material of Styrenics..." --target-hypotheses 25 --regions "North America,Europe" --reflection-concurrency 4
```

The `coscientist` command auto-starts background reflection workers per generation batch slot, so as soon as each hypothesis batch is written to the queue, reflection begins immediately.

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
.\.venv\Scripts\python.exe -m bmscientist coscientist-loop --research-id YOUR_RESEARCH_ID --target-final-hypotheses 10 --max-rounds 2 --evolve-top-k 5 --evolved-per-round 5 --regenerated-per-round 5 --proximity-check-every 1 --max-synthesized-per-round 3 --max-gap-persistence-rounds 1 --reflection-concurrency 4
```

The loop ranks active reflected hypotheses, clusters concepts, synthesizes overlapping ideas, reflects on synthesized variants, and uses meta-review whitespace analysis to control loop continuation.

## Directory Structure

Under the directory configured by `BMSCIENTIST_DATA_DIR` (defaults to `./data`):

* Vector Database: `lancedb/`
* Graph Database: `graph/`
* Co-scientist Projects: `coscientist/{research_id}/`
  * Research goal: `research_goal.json`
  * Hypotheses: `hypotheses/{generated,reflecting,reflected,evolve,retired}/{hypothesis_id}.json`
  * Execution logs: `rounds/rankings.jsonl`, `rounds/proximity.jsonl`, `rounds/meta_reviews.jsonl`
  * Output Reports: `reports/`
* Cache: `raw/`, `pricing/`

## Persistence & Validation Rules

* Evidence remains cumulative in LanceDB.
* The system preserves source URLs and chunk IDs, avoids invented citations, and keeps confidence scores conservative when evidence is incomplete.
