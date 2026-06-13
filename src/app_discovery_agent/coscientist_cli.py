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

    loop = subparsers.add_parser("coscientist-loop", help="Run ranking, evolution, and reflection loops for an existing research run.")
    loop.add_argument("--research-id", required=True)
    loop.add_argument("--target-final-hypotheses", type=int)
    loop.add_argument("--max-rounds", type=int, default=1)
    loop.add_argument("--evolve-top-k", type=int, default=5)
    loop.add_argument("--evolved-per-round", type=int, default=5)
    loop.add_argument("--regenerated-per-round", type=int, default=5)
    loop.add_argument("--proximity-check-every", type=int, default=1)
    loop.add_argument("--max-synthesized-per-round", type=int, default=3)
    loop.add_argument("--promotion-score-threshold", type=float, default=0.72)
    loop.add_argument("--gap-overlap-threshold", type=float, default=0.6)
    loop.add_argument("--max-gap-persistence-rounds", type=int, default=1)
    loop.add_argument("--preferred-evidence-recency-days", type=int)
    loop.add_argument("--max-reflection-searches-per-hypothesis", type=int)
    loop.add_argument("--results-per-query", type=int)
    loop.add_argument("--max-pages-per-search", type=int)
    loop.add_argument("--reflection-concurrency", type=int, default=3)


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


def run_coscientist_loop_command(
    args: argparse.Namespace,
    config: AppConfig,
    runner_cls: type[CoScientistRunner] = CoScientistRunner,
) -> int:
    runner = runner_cls(config)
    result = runner.run_loop(
        research_id=args.research_id,
        target_final_hypotheses=args.target_final_hypotheses,
        max_rounds=args.max_rounds,
        evolve_top_k=args.evolve_top_k,
        evolved_per_round=args.evolved_per_round,
        regenerated_per_round=args.regenerated_per_round,
        proximity_check_every=args.proximity_check_every,
        max_synthesized_per_round=args.max_synthesized_per_round,
        promotion_score_threshold=args.promotion_score_threshold,
        gap_overlap_threshold=args.gap_overlap_threshold,
        max_gap_persistence_rounds=args.max_gap_persistence_rounds,
        preferred_evidence_recency_days=args.preferred_evidence_recency_days,
        max_reflection_searches_per_hypothesis=args.max_reflection_searches_per_hypothesis,
        results_per_query=args.results_per_query,
        max_pages_per_search=args.max_pages_per_search,
        reflection_concurrency=args.reflection_concurrency,
    )
    console.print(f"[bold]Research ID:[/bold] {result.research_id}")
    console.print(f"[bold]Rounds completed:[/bold] {result.rounds_completed}")
    console.print(f"[bold]Ranked hypotheses:[/bold] {result.ranked_hypotheses}")
    console.print(f"[bold]Evolved hypotheses:[/bold] {result.evolved_hypotheses}")
    console.print(f"[bold]Regenerated hypotheses:[/bold] {result.regenerated_hypotheses}")
    console.print(f"[bold]Synthesized hypotheses:[/bold] {result.synthesized_hypotheses}")
    console.print(f"[bold]Newly reflected hypotheses:[/bold] {result.reflected_hypotheses}")
    console.print(f"[bold]Automatic discovery runs:[/bold] {result.automatic_discovery_runs}")
    console.print(f"[bold]Stop reason:[/bold] {result.stop_reason}")
    console.print(f"[bold]Rankings:[/bold] {result.ranking_path}")
    console.print(f"[bold]Hypotheses:[/bold] {result.hypothesis_path}")
    console.print(f"[bold]Report:[/bold] {result.report_path}")
    return 0
