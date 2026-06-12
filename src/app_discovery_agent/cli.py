from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from app_discovery_agent.agent import DiscoveryAgent, build_opportunity_report
from app_discovery_agent.config import AppConfig


console = Console()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


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
    table.add_column("Source URL")
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
    report_path = Path("data/outputs") / f'opportunities_{args.incumbent_material}_{args.candidate_material}.md'
    report_path.write_text(report, encoding="utf-8")
    console.print(report)
    console.print(f"\nSaved report to {report_path.resolve()}")
    return 0


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
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
