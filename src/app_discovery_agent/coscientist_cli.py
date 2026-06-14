from __future__ import annotations

import argparse
from threading import Lock

from rich.console import Console
from rich.status import Status

from app_discovery_agent.config import AppConfig
from app_discovery_agent.coscientist_agents import CoScientistRunner


console = Console()


class RichProgressReporter:
    def __init__(self, console: Console):
        self._console = console
        self._lock = Lock()
        self._status: Status | None = None
        self._active_phase: str | None = None

    def start(self, phase: str, message: str, total: int | None = None) -> None:
        with self._lock:
            self._stop_status()
            self._status = self._console.status(self._format(message, 0 if total is not None else None, total))
            self._status.start()
            self._active_phase = phase

    def advance(self, phase: str, message: str, completed: int, total: int | None = None) -> None:
        with self._lock:
            if self._status is None or self._active_phase != phase:
                self._stop_status()
                self._status = self._console.status(self._format(message, completed, total))
                self._status.start()
                self._active_phase = phase
                return
            self._status.update(self._format(message, completed, total))

    def complete(
        self,
        phase: str,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            if self._active_phase == phase:
                self._stop_status()
            self._console.print(self._format(message, completed, total, complete=True))

    @staticmethod
    def _format(message: str, completed: int | None, total: int | None, complete: bool = False) -> str:
        suffix = ""
        if total is not None:
            current = completed if completed is not None else 0
            suffix = f" [bold]{current}/{total}[/bold]"
        status = " complete" if complete and "complete" not in message.lower() and "processed" not in message.lower() else ""
        return f"[cyan]{message}{status}[/cyan][dim].....[/dim]{suffix}"

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        self._active_phase = None


def add_coscientist_parser(subparsers: argparse._SubParsersAction) -> None:
    coscientist = subparsers.add_parser("coscientist", help="Run the co-scientist generation and reflection workflow.")
    coscientist.add_argument("--goal", required=True)
    coscientist.add_argument("--project-name")
    coscientist.add_argument("--target-hypotheses", required=True, type=int)
    coscientist.add_argument("--regions")
    coscientist.add_argument("--strategic-fit-notes")
    coscientist.add_argument("--preferred-evidence-recency-days", type=int, default=180)
    coscientist.add_argument("--max-reflection-searches-per-hypothesis", type=int, default=3)
    coscientist.add_argument("--results-per-query", type=int, default=5)
    coscientist.add_argument("--max-pages-per-search", type=int, default=8)
    coscientist.add_argument("--reflection-concurrency", type=int, default=3)
    coscientist.add_argument("--skip-loop", action="store_true")
    coscientist.add_argument("--target-final-hypotheses", type=int)
    coscientist.add_argument("--max-rounds", type=int, default=2)
    coscientist.add_argument("--evolve-top-k", type=int, default=5)
    coscientist.add_argument("--evolved-per-round", type=int, default=5)
    coscientist.add_argument("--regenerated-per-round", type=int, default=5)
    coscientist.add_argument("--proximity-check-every", type=int, default=1)
    coscientist.add_argument("--max-synthesized-per-round", type=int, default=3)
    coscientist.add_argument("--promotion-score-threshold", type=float, default=0.72)
    coscientist.add_argument("--gap-overlap-threshold", type=float, default=0.6)
    coscientist.add_argument("--max-gap-persistence-rounds", type=int, default=1)

    reflect = subparsers.add_parser("coscientist-reflect", help="Resume reflection for an existing co-scientist research run.")
    reflect.add_argument("--research-id", "--project-name", dest="research_id", required=True)
    reflect.add_argument("--preferred-evidence-recency-days", type=int)
    reflect.add_argument("--max-reflection-searches-per-hypothesis", type=int)
    reflect.add_argument("--results-per-query", type=int)
    reflect.add_argument("--max-pages-per-search", type=int)
    reflect.add_argument("--max-hypotheses", type=int)
    reflect.add_argument("--concurrency", type=int, default=3)

    loop = subparsers.add_parser("coscientist-loop", help="Run ranking, evolution, and reflection loops for an existing research run.")
    loop.add_argument("--research-id", "--project-name", dest="research_id", required=True)
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
    if hasattr(runner, "set_progress_reporter"):
        runner.set_progress_reporter(RichProgressReporter(console))
    project_name = getattr(args, "project_name", None)
    if hasattr(runner, "prepare_project_name"):
        project_name = runner.prepare_project_name(args.project_name)
    if project_name:
        console.print(f"[bold]Creating new project named:[/bold] {project_name}")
    result = runner.run(
        goal=args.goal,
        project_name=project_name,
        target_hypotheses=args.target_hypotheses,
        regions=regions,
        strategic_fit_notes=args.strategic_fit_notes,
        preferred_evidence_recency_days=args.preferred_evidence_recency_days,
        max_reflection_searches_per_hypothesis=args.max_reflection_searches_per_hypothesis,
        results_per_query=args.results_per_query,
        max_pages_per_search=args.max_pages_per_search,
        reflection_concurrency=args.reflection_concurrency,
    )
    loop_result = None
    if not args.skip_loop and hasattr(runner, "run_loop"):
        console.print("[bold]Continuing into ranking and evolution loop...[/bold]")
        loop_result = runner.run_loop(
            research_id=result.research_id,
            target_final_hypotheses=args.target_final_hypotheses or args.target_hypotheses,
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
    console.print(f"[bold]Project Name:[/bold] {result.research_id}")
    console.print(f"[bold]Generated hypotheses:[/bold] {result.generated_hypotheses}")
    console.print(f"[bold]Initial reflected hypotheses:[/bold] {result.reflected_hypotheses}")
    if loop_result is not None:
        console.print(f"[bold]Rounds completed:[/bold] {loop_result.rounds_completed}")
        console.print(f"[bold]Ranked hypotheses:[/bold] {loop_result.ranked_hypotheses}")
        console.print(f"[bold]Evolved hypotheses:[/bold] {loop_result.evolved_hypotheses}")
        console.print(f"[bold]Regenerated hypotheses:[/bold] {loop_result.regenerated_hypotheses}")
        console.print(f"[bold]Synthesized hypotheses:[/bold] {loop_result.synthesized_hypotheses}")
        console.print(f"[bold]Newly reflected hypotheses:[/bold] {loop_result.reflected_hypotheses}")
        console.print(f"[bold]Stop reason:[/bold] {loop_result.stop_reason}")
        console.print(f"[bold]Automatic discovery runs:[/bold] {result.automatic_discovery_runs + loop_result.automatic_discovery_runs}")
        console.print(f"[bold]Research goal:[/bold] {result.research_goal_path}")
        console.print(f"[bold]Hypotheses:[/bold] {loop_result.hypothesis_path}")
        console.print(f"[bold]Rankings:[/bold] {loop_result.ranking_path}")
        console.print(f"[bold]Report:[/bold] {loop_result.report_path}")
        return 0
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
    if hasattr(runner, "set_progress_reporter"):
        runner.set_progress_reporter(RichProgressReporter(console))
    result = runner.reflect_existing(
        research_id=args.research_id,
        preferred_evidence_recency_days=args.preferred_evidence_recency_days,
        max_reflection_searches_per_hypothesis=args.max_reflection_searches_per_hypothesis,
        results_per_query=args.results_per_query,
        max_pages_per_search=args.max_pages_per_search,
        max_hypotheses=args.max_hypotheses,
        concurrency=args.concurrency,
    )
    console.print(f"[bold]Project Name:[/bold] {result.research_id}")
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
    if hasattr(runner, "set_progress_reporter"):
        runner.set_progress_reporter(RichProgressReporter(console))
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
    console.print(f"[bold]Project Name:[/bold] {result.research_id}")
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
