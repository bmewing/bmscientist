from __future__ import annotations

import argparse

from rich.console import Console

from app_discovery_agent.config import AppConfig
from app_discovery_agent.coscientist_agents import CoScientistRunner


console = Console()


def add_coscientist_parser(subparsers: argparse._SubParsersAction) -> None:
    coscientist = subparsers.add_parser("coscientist", help="Run the co-scientist generation and reflection workflow.")
    coscientist.add_argument("--goal", required=True)
    coscientist.add_argument("--target-hypotheses", required=True, type=int)
    coscientist.add_argument("--regions")
    coscientist.add_argument("--strategic-fit-notes")
    coscientist.add_argument("--preferred-evidence-recency-days", type=int, default=180)
    coscientist.add_argument("--max-reflection-searches-per-hypothesis", type=int, default=3)
    coscientist.add_argument("--results-per-query", type=int, default=5)
    coscientist.add_argument("--max-pages-per-search", type=int, default=8)
    coscientist.add_argument("--reflection-concurrency", type=int, default=3)

    reflect = subparsers.add_parser("coscientist-reflect", help="Resume reflection for an existing co-scientist research run.")
    reflect.add_argument("--research-id", required=True)
    reflect.add_argument("--preferred-evidence-recency-days", type=int)
    reflect.add_argument("--max-reflection-searches-per-hypothesis", type=int)
    reflect.add_argument("--results-per-query", type=int)
    reflect.add_argument("--max-pages-per-search", type=int)
    reflect.add_argument("--max-hypotheses", type=int)
    reflect.add_argument("--concurrency", type=int, default=3)


def run_coscientist_command(
    args: argparse.Namespace,
    config: AppConfig,
    runner_cls: type[CoScientistRunner] = CoScientistRunner,
) -> int:
    regions = [item.strip() for item in (args.regions or "").split(",") if item.strip()]
    runner = runner_cls(config)
    result = runner.run(
        goal=args.goal,
        target_hypotheses=args.target_hypotheses,
        regions=regions,
        strategic_fit_notes=args.strategic_fit_notes,
        preferred_evidence_recency_days=args.preferred_evidence_recency_days,
        max_reflection_searches_per_hypothesis=args.max_reflection_searches_per_hypothesis,
        results_per_query=args.results_per_query,
        max_pages_per_search=args.max_pages_per_search,
        reflection_concurrency=args.reflection_concurrency,
    )
    console.print(f"[bold]Research ID:[/bold] {result.research_id}")
    console.print(f"[bold]Generated hypotheses:[/bold] {result.generated_hypotheses}")
    console.print(f"[bold]Reflected hypotheses:[/bold] {result.reflected_hypotheses}")
    console.print(f"[bold]Automatic discovery runs:[/bold] {result.automatic_discovery_runs}")
    console.print(f"[bold]Research goal:[/bold] {result.research_goal_path}")
    console.print(f"[bold]Hypotheses:[/bold] {result.hypothesis_path}")
    console.print(f"[bold]Report:[/bold] {result.report_path}")
    return 0


def run_coscientist_reflect_command(
    args: argparse.Namespace,
    config: AppConfig,
    runner_cls: type[CoScientistRunner] = CoScientistRunner,
) -> int:
    runner = runner_cls(config)
    result = runner.reflect_existing(
        research_id=args.research_id,
        preferred_evidence_recency_days=args.preferred_evidence_recency_days,
        max_reflection_searches_per_hypothesis=args.max_reflection_searches_per_hypothesis,
        results_per_query=args.results_per_query,
        max_pages_per_search=args.max_pages_per_search,
        max_hypotheses=args.max_hypotheses,
        concurrency=args.concurrency,
    )
    console.print(f"[bold]Research ID:[/bold] {result.research_id}")
    console.print(f"[bold]Generated hypotheses:[/bold] {result.generated_hypotheses}")
    console.print(f"[bold]Reflected hypotheses:[/bold] {result.reflected_hypotheses}")
    console.print(f"[bold]Automatic discovery runs:[/bold] {result.automatic_discovery_runs}")
    console.print(f"[bold]Research goal:[/bold] {result.research_goal_path}")
    console.print(f"[bold]Hypotheses:[/bold] {result.hypothesis_path}")
    console.print(f"[bold]Report:[/bold] {result.report_path}")
    return 0
