# app-discovery-agent

Local Python MVP for discovering material-application evidence, storing cumulative vectorized chunks in LanceDB, and running an early multi-agent co-scientist workflow over the stored evidence.

## Setup

Use the local virtual environment for all Python commands on this project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Create a `.env` from `.env.example` and add your keys.

## Environment

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
EXA_API_KEY=your_exa_api_key
HF_TOKEN=your_huggingface_token
LANCEDB_PATH=./data/lancedb
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

`CHAT_MODEL` remains the default. Agent-specific variables let you route expensive reasoning tasks to stronger DeepSeek models without changing the whole app.
`HF_TOKEN` is optional but recommended for higher Hugging Face Hub rate limits and authenticated model downloads.

## Discovery

Run external discovery, classification, chunking, embedding, and cumulative LanceDB storage:

```powershell
.\.venv\Scripts\python.exe -m app_discovery_agent.cli discover --query "major applications of PVC material and key performance requirements" --max-search-queries 8 --results-per-query 10 --max-pages 30
```

Search local evidence:

```powershell
.\.venv\Scripts\python.exe -m app_discovery_agent.cli search --query "Where is PVC used in clear rigid applications?" --top-k 8
```

Manually obtained files can be dropped into `data/manually-obtained/`; DB-oriented workflows ingest them and move processed files into `data/manually-obtained/processed/`.

## Co-Scientist

Create a research goal, generate hypotheses from local evidence, and reflect them:

```powershell
.\.venv\Scripts\python.exe -m app_discovery_agent.cli coscientist --goal "I want to find rapid drop-in/drop-out flywheel opportunities for PET against a target material of Styrenics..." --target-hypotheses 25 --regions "North America,Europe" --reflection-concurrency 4
```

The `coscientist` command now auto-starts one background reflection worker per generation batch slot, so as soon as each hypothesis batch is written to the queue, reflection begins immediately instead of waiting for all generation to finish.

Resume reflection for generated hypotheses without regenerating:

```powershell
.\.venv\Scripts\python.exe -m app_discovery_agent.cli coscientist-reflect --research-id YOUR_RESEARCH_ID --concurrency 4
```

Run multiple long-lived reflection workers against the same research run with the file-rename queue:

```powershell
.\.venv\Scripts\python.exe -m app_discovery_agent.cli coscientist-reflect --research-id YOUR_RESEARCH_ID --daemon --worker-id reflector-a --concurrency 2 --poll-interval-seconds 5
.\.venv\Scripts\python.exe -m app_discovery_agent.cli coscientist-reflect --research-id YOUR_RESEARCH_ID --daemon --worker-id reflector-b --concurrency 2 --poll-interval-seconds 5
```

Each worker atomically claims hypotheses by renaming files from `generated/` to `reflecting/`, then moves completed work into `reflected/`. If a worker dies mid-task, the reflection lease eventually expires and another worker can requeue that hypothesis automatically.

Run the Ranking Agent, Proximity Check Agent, Meta-review Agent, Evolution Agent, feedback generation, and bounded reflection loop over an existing research run:

```powershell
.\.venv\Scripts\python.exe -m app_discovery_agent.cli coscientist-loop --research-id YOUR_RESEARCH_ID --target-final-hypotheses 10 --max-rounds 2 --evolve-top-k 5 --evolved-per-round 5 --regenerated-per-round 5 --proximity-check-every 1 --max-synthesized-per-round 3 --max-gap-persistence-rounds 1 --reflection-concurrency 4
```

The loop ranks active reflected hypotheses, clusters concepts, can synthesize overlapping ideas into new higher-level hypotheses, reflects any synthesized/generated variants, and uses meta-review whitespace analysis to decide whether one more loop is worth doing. The ranker judges; the meta-review agent is the only agent that writes guidance for the next generation pass.

The loop keeps each research run under `data/coscientist/{research_id}/`. Hypotheses are individual JSON files that move through queue-like folders (`generated`, `reflecting`, `reflected`, `evolve`, `retired`), while ranking, proximity, and meta-review rounds remain append-only JSONL logs under the run's `rounds/` directory.

## Persistence Rules

Evidence remains cumulative in LanceDB under `data/lancedb`. Co-scientist artifacts are local and inspectable:

- Research goal: `data/coscientist/{research_id}/research_goal.json`
- Hypotheses: `data/coscientist/{research_id}/hypotheses/{generated,reflecting,reflected,evolve,retired}/{hypothesis_id}.json`
- Ranking rounds: `data/coscientist/{research_id}/rounds/rankings.jsonl`
- Proximity rounds: `data/coscientist/{research_id}/rounds/proximity.jsonl`
- Meta-review rounds: `data/coscientist/{research_id}/rounds/meta_reviews.jsonl`
- Reports: `data/coscientist/{research_id}/reports/`

The system should preserve source URLs and chunk IDs, avoid invented citations, and keep scores conservative when evidence is incomplete.
