from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from bmscientist.agent import DiscoveryAgent, build_opportunity_report
from bmscientist.config import AppConfig
from bmscientist.coscientist_cli import (
    add_coscientist_parser,
    run_coscientist_command,
    run_coscientist_loop_command,
    run_coscientist_reflect_command,
    run_coscientist_feedback_command,
    run_coscientist_meta_review_command,
)
from bmscientist.graph_backfill import LanceGraphBackfiller
from bmscientist.graph_enrichment import GraphEnrichmentProposer, GraphEnrichmentStore, GraphEnrichmentValidator
from bmscientist.graph_backfill import chunk_record_from_lancedb_row
from bmscientist.embeddings import LocalEmbedder
from bmscientist.llm import DeepSeekLLM
from bmscientist.store import LanceEvidenceStore


console = Console()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    if not verbose:
        for logger_name in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger("bmscientist.extract").setLevel(logging.ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app-discovery-agent")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover, classify, and ingest evidence.")
    discover.add_argument("--query", required=True)
    discover.add_argument("--max-search-queries", type=int, default=8)
    discover.add_argument("--results-per-query", type=int, default=5)
    discover.add_argument("--max-pages", type=int, default=20)

    search = subparsers.add_parser("search", help="Search local stored evidence.")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=8)

    opportunities = subparsers.add_parser("opportunities", help="Summarize opportunities from local evidence.")
    opportunities.add_argument("--incumbent-material", required=True)
    opportunities.add_argument("--candidate-material", required=True)

    replay = subparsers.add_parser("replay", help="Replay a discovery run from cached raw artifacts without using Exa.")
    replay.add_argument("--query")
    replay.add_argument("--run-id")
    replay.add_argument("--search-results-file")
    replay.add_argument("--fetched-pages-file")
    replay.add_argument("--max-pages", type=int, default=20)

    graph_backfill = subparsers.add_parser("graph-backfill", help="Backfill governed graph enrichments from existing LanceDB evidence.")
    graph_backfill.add_argument(
        "--query",
        default="backfill existing LanceDB evidence into graph enrichment proposals",
        help="Context query to give the enrichment proposer.",
    )
    graph_backfill.add_argument("--batch-size", type=int, default=12)
    graph_backfill.add_argument("--limit", type=int)
    graph_backfill.add_argument("--include-claimed", action="store_true", help="Reprocess chunks already present in the graph claim ledger.")
    graph_backfill.add_argument(
        "--search-query",
        action="append",
        dest="search_queries",
        help="Vector-search LanceDB first and backfill only matching chunks. May be repeated.",
    )
    graph_backfill.add_argument("--top-k-per-query", type=int, default=25)
    add_coscientist_parser(subparsers)

    return parser


def run_discover(args: argparse.Namespace, config: AppConfig) -> int:
    agent = DiscoveryAgent(config)
    summary = agent.discover(
        query=args.query,
        max_search_queries=args.max_search_queries,
        results_per_query=args.results_per_query,
        max_pages=args.max_pages,
    )
    console.print(f"[bold]Run ID:[/bold] {summary.run_id}")
    console.print(f"[bold]Stored chunks:[/bold] {summary.stored_chunks}")
    console.print(f"[bold]Relevant pages:[/bold] {summary.relevant_pages}")
    console.print(f"[bold]Summary:[/bold] {summary.output_path}")
    console.print(summary.opportunity_summary)
    return 0


def run_search(args: argparse.Namespace, config: AppConfig) -> int:
    agent = DiscoveryAgent(config)
    query_vector = agent.embedder.embed_query(args.query)
    rows = agent.store.search_by_vector(query_vector, top_k=args.top_k)

    table = Table(title="Local Evidence Search Results")
    table.add_column("Application")
    table.add_column("Evidence Type")
    table.add_column("Relevance")
    table.add_column("Chunk ID")
    table.add_column("Source Ref")
    table.add_column("Excerpt")

    for row in rows:
        table.add_row(
            row.get("application") or "-",
            row.get("evidence_type") or "-",
            f'{row.get("relevance_score", 0):.2f}',
            row.get("id", ""),
            row.get("source_url", ""),
            (row.get("chunk_text", "")[:140] + "...") if len(row.get("chunk_text", "")) > 140 else row.get("chunk_text", ""),
        )
    console.print(table)
    return 0


def run_opportunities(args: argparse.Namespace, config: AppConfig) -> int:
    agent = DiscoveryAgent(config)
    rows = agent.store.all_rows()
    filtered = []
    incumbent = args.incumbent_material.lower()
    candidate = args.candidate_material.lower()

    for row in rows:
        incumbent_match = (row.get("incumbent_material") or "").lower() == incumbent or incumbent in row.get("chunk_text", "").lower()
        candidates = [item.lower() for item in row.get("candidate_materials", [])]
        candidate_match = candidate in candidates or candidate in row.get("chunk_text", "").lower()
        if incumbent_match and candidate_match:
            filtered.append(row)

    if not filtered:
        console.print("No locally stored evidence matched that incumbent/candidate pair.")
        return 0

    report = build_opportunity_report(agent.llm, filtered, args.incumbent_material, args.candidate_material)
    report_path = config.data_dir / "outputs" / f'opportunities_{args.incumbent_material}_{args.candidate_material}.md'
    report_path.write_text(report, encoding="utf-8")
    console.print(report)
    console.print(f"\nSaved report to {report_path.resolve()}")
    return 0


def run_replay(args: argparse.Namespace, config: AppConfig) -> int:
    agent = DiscoveryAgent(config)
    search_results_path, fetched_pages_path = resolve_replay_paths(args.run_id, args.search_results_file, args.fetched_pages_file, data_dir=config.data_dir)
    query = args.query or infer_query_from_search_results(search_results_path)

    if not query:
        raise ValueError("Replay requires --query, or a search_results.json file containing the original query.")

    summary = agent.replay_discovery(
        query=query,
        search_results_path=search_results_path,
        fetched_pages_path=fetched_pages_path,
        max_pages=args.max_pages,
    )
    console.print(f"[bold]Replay run ID:[/bold] {summary.run_id}")
    console.print(f"[bold]Stored chunks:[/bold] {summary.stored_chunks}")
    console.print(f"[bold]Relevant pages:[/bold] {summary.relevant_pages}")
    console.print(f"[bold]Summary:[/bold] {summary.output_path}")
    console.print(summary.opportunity_summary)
    return 0


def run_graph_backfill(args: argparse.Namespace, config: AppConfig) -> int:
    llm = DeepSeekLLM(config)
    store = LanceEvidenceStore(config.resolved_lancedb_path())
    records = None
    if args.search_queries:
        embedder = LocalEmbedder(config)
        records_by_id = {}
        for search_query in args.search_queries:
            vector = embedder.embed_query(search_query)
            for row in store.search_by_vector(vector, top_k=args.top_k_per_query):
                record = chunk_record_from_lancedb_row(row)
                if record is not None:
                    records_by_id.setdefault(record.id, record)
        records = list(records_by_id.values())
    backfiller = LanceGraphBackfiller(
        store,
        GraphEnrichmentProposer(llm),
        GraphEnrichmentValidator(llm),
        GraphEnrichmentStore(),
    )
    result = backfiller.run(
        query=args.query,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_claimed=not args.include_claimed,
        records=records,
    )
    console.print(f"[bold]Scanned chunks:[/bold] {result.scanned_chunks}")
    console.print(f"[bold]Eligible chunks:[/bold] {result.eligible_chunks}")
    console.print(f"[bold]Batches:[/bold] {result.batches}")
    console.print(f"[bold]Proposed claims:[/bold] {result.proposed_claims}")
    console.print(f"[bold]Accepted claims:[/bold] {result.accepted_claims}")
    console.print(f"[bold]Backfill details:[/bold] {result.output_path}")
    return 0


def resolve_replay_paths(
    run_id: str | None,
    search_results_file: str | None,
    fetched_pages_file: str | None,
    *,
    data_dir: Path = Path("data"),
) -> tuple[Path | None, Path | None]:
    if search_results_file or fetched_pages_file:
        return (
            Path(search_results_file) if search_results_file else None,
            Path(fetched_pages_file) if fetched_pages_file else None,
        )

    raw_dir = data_dir / "raw"
    if run_id:
        return raw_dir / f"{run_id}_search_results.json", raw_dir / f"{run_id}_fetched_pages.json"

    latest_search = max(raw_dir.glob("*_search_results.json"), key=lambda path: path.stat().st_mtime, default=None)
    latest_pages = max(raw_dir.glob("*_fetched_pages.json"), key=lambda path: path.stat().st_mtime, default=None)
    return latest_search, latest_pages


def infer_query_from_search_results(search_results_path: Path | None) -> str | None:
    if not search_results_path or not search_results_path.exists():
        return None
    import json

    payload = json.loads(search_results_path.read_text(encoding="utf-8"))
    for entry in payload:
        query = entry.get("query")
        if query:
            return query
    return None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)
    config = AppConfig.from_env(args.env_file)

    if args.command == "discover":
        return run_discover(args, config)
    if args.command == "search":
        return run_search(args, config)
    if args.command == "opportunities":
        return run_opportunities(args, config)
    if args.command == "replay":
        return run_replay(args, config)
    if args.command == "graph-backfill":
        return run_graph_backfill(args, config)
    if args.command == "coscientist":
        return run_coscientist_command(args, config)
    if args.command == "coscientist-reflect":
        return run_coscientist_reflect_command(args, config)
    if args.command == "coscientist-loop":
        return run_coscientist_loop_command(args, config)
    if args.command == "coscientist-feedback":
        return run_coscientist_feedback_command(args, config)
    if args.command == "coscientist-meta-review":
        return run_coscientist_meta_review_command(args, config)
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
