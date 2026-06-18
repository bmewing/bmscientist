from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.text import Text

from bmscientist.config import AppConfig
from bmscientist.coscientist_agents import CoScientistRunner


console = Console()


@dataclass
class PhaseState:
    message: str
    completed: int | None = None
    total: int | None = None
    is_complete: bool = False
    details: list[str] | None = None


class RichProgressReporter:
    def __init__(self, console: Console):
        self._console = console
        self._lock = Lock()
        self._live: Live | None = None
        self._phases: "OrderedDict[str, PhaseState]" = OrderedDict()

    def start(self, phase: str, message: str, total: int | None = None) -> None:
        with self._lock:
            self._phases[phase] = PhaseState(
                message=message,
                completed=0 if total is not None else None,
                total=total,
                is_complete=False,
            )
            self._refresh_live()

    def advance(self, phase: str, message: str, completed: int, total: int | None = None) -> None:
        with self._lock:
            current = self._phases.get(phase, PhaseState(message=message))
            self._phases[phase] = PhaseState(
                message=message,
                completed=completed,
                total=total if total is not None else current.total,
                is_complete=False,
            )
            self._refresh_live()

    def complete(
        self,
        phase: str,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            current = self._phases.get(phase, PhaseState(message=message))
            self._phases[phase] = PhaseState(
                message=message,
                completed=completed if completed is not None else current.completed,
                total=total if total is not None else current.total,
                is_complete=True,
            )
            self._refresh_live()
            if self._all_phases_complete():
                self._stop_live()
                self._phases.clear()

    def details(self, phase: str, lines: list[str]) -> None:
        with self._lock:
            current = self._phases.get(phase)
            if current is None:
                return
            self._phases[phase] = PhaseState(
                message=current.message,
                completed=current.completed,
                total=current.total,
                is_complete=current.is_complete,
                details=lines,
            )
            self._refresh_live()

    @staticmethod
    def _format(message: str, completed: int | None, total: int | None, complete: bool = False) -> Text:
        status_message = message
        if complete and "complete" not in message.lower() and "processed" not in message.lower():
            status_message = f"{message} complete"
        text = Text()
        text.append(status_message, style="cyan")
        text.append(".....", style="dim")
        if total is not None:
            current = completed if completed is not None else 0
            text.append(f" {current}/{total}", style="bold")
        return text

    def _render(self) -> Group:
        lines: list[Text] = []
        for state in self._phases.values():
            lines.append(self._format(state.message, state.completed, state.total, complete=state.is_complete))
            for detail in state.details or []:
                detail_text = Text()
                detail_text.append("  ", style="dim")
                detail_text.append(detail, style="dim")
                lines.append(detail_text)
        if not lines:
            lines = [Text("")]
        return Group(*lines)

    def _refresh_live(self) -> None:
        if self._live is None:
            self._live = Live(self._render(), console=self._console, refresh_per_second=8, transient=False)
            self._live.start()
            return
        self._live.update(self._render(), refresh=True)

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _all_phases_complete(self) -> bool:
        return bool(self._phases) and all(state.is_complete for state in self._phases.values())


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

    feedback = subparsers.add_parser("coscientist-feedback", help="Provide human feedback on hypotheses or graph edges after a run.")
    feedback.add_argument("--research-id", "--project-name", dest="research_id", required=True)
    feedback.add_argument("--hypothesis-id")
    feedback.add_argument("--candidate", help="Candidate material (optional if --hypothesis-id is specified)")
    feedback.add_argument("--incumbent", help="Incumbent material (optional)")
    feedback.add_argument("--application", help="Application name (optional if --hypothesis-id is specified)")
    feedback.add_argument("--volume", type=float, help="Corrected volume value (e.g. 0.0 for near-zero volume)")
    feedback.add_argument("--volume-unit", default="tonnes", help="Volume unit")
    feedback.add_argument("--status", choices=["accepted", "rejected", "retired", "low_volume", "edited"], help="Feedback status")
    feedback.add_argument("--confidence", type=float, help="Feedback confidence score (0.0 to 1.0)")
    feedback.add_argument("--comment", help="Feedback comment or retired reason")
    feedback.add_argument("--title", help="Edit: new title for the hypothesis")
    feedback.add_argument("--summary", help="Edit: new summary for the hypothesis")
    feedback.add_argument("--strategic-rationale", help="Edit: new strategic rationale for the hypothesis")
    feedback.add_argument("--project-feedback", help="Provide overall project-level feedback or new direction to update project goals")

    meta_review = subparsers.add_parser("coscientist-meta-review", help="Explicitly run a meta-review pass to identify gaps and update project guidance.")
    meta_review.add_argument("--research-id", "--project-name", dest="research_id", required=True)
    meta_review.add_argument("--evolve-top-k", type=int, default=5, help="Number of top hypotheses to consider for evolution.")
    meta_review.add_argument("--evolved-per-round", type=int, default=5, help="Number of evolved hypotheses to generate.")
    meta_review.add_argument("--reflection-concurrency", type=int, default=3, help="Reflection concurrency.")

    coscientist.add_argument("--skip-loop", action="store_true")
    coscientist.add_argument("--target-final-hypotheses", type=int)
    coscientist.add_argument("--max-rounds", type=int, help="Optional safety cap for loop rounds; omit to let meta-review control stopping.")
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
    reflect.add_argument("--daemon", action="store_true")
    reflect.add_argument("--worker-id")
    reflect.add_argument("--lease-seconds", type=int, default=1800)
    reflect.add_argument("--poll-interval-seconds", type=int, default=5)
    reflect.add_argument("--idle-exit-after-seconds", type=int)

    loop = subparsers.add_parser("coscientist-loop", help="Run ranking, evolution, and reflection loops for an existing research run.")
    loop.add_argument("--research-id", "--project-name", dest="research_id", required=True)
    loop.add_argument("--target-final-hypotheses", type=int)
    loop.add_argument("--max-rounds", type=int, help="Optional safety cap for loop rounds; omit to let meta-review control stopping.")
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
        spawn_reflection_daemons=True,
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
        daemon=args.daemon,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        idle_exit_after_seconds=args.idle_exit_after_seconds,
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


def run_coscientist_feedback_command(
    args: argparse.Namespace,
    config: AppConfig,
) -> int:
    from bmscientist.coscientist_store import CoScientistStore
    from bmscientist.graph_enrichment import GraphEnrichmentStore

    store = CoScientistStore()

    project_feedback = getattr(args, "project_feedback", None)
    if project_feedback:
        document = store.load_research_goal(args.research_id)
        if not document:
            console.print(f"[bold red]Research project {args.research_id} not found.[/bold red]")
            return 1

        from bmscientist.coscientist_agents import ResearchPlanningAgent, RankingAgent, DeepSeekLLM
        
        console.print(f"[bold]Updating project goals for {args.research_id} using feedback: '{project_feedback}'...[/bold]")
        planning_llm = DeepSeekLLM(config, model=config.planning_chat_model)
        planning_agent = ResearchPlanningAgent(planning_llm)
        updated_document = planning_agent.update_research_goal(document, project_feedback)
        store.save_research_goal(updated_document)
        console.print("[bold green]Successfully updated project goals.[/bold green]")

        # Trigger ranking agent to re-rank the existing hypotheses
        latest_hypotheses = store.latest_hypotheses(args.research_id)
        active_reflected = [
            h for h in latest_hypotheses
            if h.status == "reflected" and h.is_active
        ]
        if active_reflected:
            console.print(f"[bold]Re-ranking {len(active_reflected)} active reflected hypotheses...[/bold]")
            ranking_llm = DeepSeekLLM(config, model=config.ranking_chat_model)
            ranking_agent = RankingAgent(ranking_llm)
            
            rounds = store.load_ranking_rounds(args.research_id)
            next_round_index = len(rounds) + 1
            
            ranking_round, ranked_hypotheses = ranking_agent.rank(
                document=updated_document,
                hypotheses=active_reflected,
                round_index=next_round_index,
                target_final_count=updated_document.target_hypotheses_final,
                evolve_top_k=5,
            )
            store.append_ranking_round(ranking_round)
            for hypothesis in ranked_hypotheses:
                store.save_hypothesis(hypothesis)
            console.print(f"[bold green]Successfully re-ranked hypotheses (ranking round {next_round_index} saved).[/bold green]")
        else:
            console.print("[yellow]No active reflected hypotheses to re-rank.[/yellow]")

        return 0

    if args.hypothesis_id:
        console.print(f"[bold]Applying feedback to hypothesis {args.hypothesis_id} in run {args.research_id}...[/bold]")
        updated = store.apply_hypothesis_feedback(
            research_id=args.research_id,
            hypothesis_id=args.hypothesis_id,
            volume=args.volume,
            volume_unit=args.volume_unit,
            status=args.status,
            confidence=args.confidence,
            comment=args.comment,
            title=getattr(args, "title", None),
            summary=getattr(args, "summary", None),
            candidate_material=getattr(args, "candidate", None),
            incumbent_material=getattr(args, "incumbent", None),
            application=getattr(args, "application", None),
            strategic_rationale=getattr(args, "strategic_rationale", None),
        )
        if not updated:
            console.print(f"[bold red]Hypothesis {args.hypothesis_id} not found in run {args.research_id}.[/bold red]")
            return 1
        console.print(f"[bold green]Successfully updated hypothesis {args.hypothesis_id} (new status: {updated.status}).[/bold green]")
    else:
        if not args.candidate or not args.application:
            console.print("[bold red]Error: --candidate and --application must be specified if --hypothesis-id is omitted.[/bold red]")
            return 1

        console.print(f"[bold]Applying direct graph feedback for {args.candidate} replacing {args.incumbent} in {args.application}...[/bold]")
        graph_store = GraphEnrichmentStore()
        status_val = args.status or "rejected"
        graph_status = "rejected" if status_val in ("rejected", "retired") else status_val
        updated_count = graph_store.apply_edge_feedback(
            candidate_material=args.candidate,
            incumbent_material=args.incumbent,
            application=args.application,
            volume=args.volume,
            volume_unit=args.volume_unit,
            status=graph_status,
            confidence=args.confidence,
            comment=args.comment,
        )
        console.print(f"[bold green]Successfully updated {updated_count} edges in the knowledge graph.[/bold green]")

    return 0


def run_coscientist_meta_review_command(
    args: argparse.Namespace,
    config: AppConfig,
) -> int:
    from bmscientist.coscientist_store import CoScientistStore
    from bmscientist.coscientist_agents import CoScientistRunner, DeepSeekLLM
    
    store = CoScientistStore()
    document = store.load_research_goal(args.research_id)
    if not document:
        console.print(f"[bold red]Research project {args.research_id} not found.[/bold red]")
        return 1
        
    console.print(f"[bold]Running explicit meta-review and evolution for run {args.research_id}...[/bold]")
    
    runner = CoScientistRunner(config, artifact_store=store)
    
    meta_rounds = store.load_meta_review_rounds(args.research_id)
    round_index = len(meta_rounds) + 1
    
    # 1. Rank current hypotheses
    latest_hypotheses = store.latest_hypotheses(args.research_id)
    active_reflected = [
        h for h in latest_hypotheses
        if h.status == "reflected" and h.is_active
    ]
    
    if not active_reflected:
        console.print("[bold red]No active reflected hypotheses found. Cannot run meta-review.[/bold red]")
        return 1
        
    console.print(f"[bold]Ranking {len(active_reflected)} active reflected hypotheses...[/bold]")
    ranking_round, ranked_hypotheses = runner._ranking_agent.rank(
        document=document,
        hypotheses=active_reflected,
        round_index=round_index,
        target_final_count=document.target_hypotheses_final,
        evolve_top_k=args.evolve_top_k,
    )
    store.append_ranking_round(ranking_round)
    for hypothesis in ranked_hypotheses:
        store.save_hypothesis(hypothesis)
        
    # 2. Run meta-review
    console.print("[bold]Reviewing portfolio gaps...[/bold]")
    updated_document, meta_review_round = runner._meta_review_agent.review(
        document=document,
        hypotheses=ranked_hypotheses,
        ranking_round=ranking_round,
        round_index=round_index,
        gap_overlap_threshold=0.6,
        max_gap_persistence_rounds=1,
    )
    store.save_research_goal(updated_document)
    store.append_meta_review_round(meta_review_round)
    
    console.print("[bold green]Meta-Review completed successfully![/bold green]")
    console.print(f"[bold]Whitespace Gaps:[/bold] {meta_review_round.whitespace_gaps}")
    console.print(f"[bold]Generation Guidance:[/bold] {meta_review_round.generation_guidance}")
    
    # 3. Evolve parent hypotheses
    parent_by_id = {h.hypothesis_id: h for h in ranked_hypotheses}
    parents = [
        parent_by_id[h_id]
        for h_id in ranking_round.evolved_parent_hypothesis_ids
        if h_id in parent_by_id
    ]
    
    if parents:
        console.print(f"[bold]Evolving {len(parents)} parent hypotheses (including accepted ones)...[/bold]")
        for parent in parents:
            store.append_hypothesis_snapshot(parent.model_copy(update={"status": "evolve"}))
            
        evolved = runner._evolution_agent.evolve(
            document=updated_document,
            parent_hypotheses=parents,
            ranking_round=ranking_round,
            target_count=args.evolved_per_round,
            round_index=round_index,
        )
        
        new_hypotheses = runner._dedupe_new_hypotheses(
            evolved,
            existing_hypothesis_ids={h.hypothesis_id for h in store.latest_hypotheses(args.research_id)},
        )
        
        for parent in parents:
            store.append_hypothesis_snapshot(parent.model_copy(update={"status": "reflected"}))
            
        for h in new_hypotheses:
            store.append_hypothesis_snapshot(h)
            
        if new_hypotheses:
            console.print(f"[bold]Reflecting on {len(new_hypotheses)} new evolved hypotheses...[/bold]")
            reflected_new, run_count = runner._reflect_and_append(
                updated_document,
                new_hypotheses,
                concurrency=args.reflection_concurrency,
                phase="new_reflection",
                progress_message="Reflecting on new evolved ideas",
            )
            console.print(f"[bold green]Successfully generated and reflected {len(reflected_new)} evolved hypotheses.[/bold green]")
        else:
            console.print("[yellow]No new evolved hypotheses generated (all duplicates).[/yellow]")
    else:
        console.print("[yellow]No parent hypotheses identified for evolution.[/yellow]")
        
    return 0

