from __future__ import annotations

import json
from pathlib import Path

from app_discovery_agent.coscientist_models import (
    Hypothesis,
    MetaReviewRound,
    ProximityRound,
    RankingRound,
    ResearchGoalDocument,
)


class CoScientistStore:
    HYPOTHESIS_STAGES = ("generated", "reflected", "evolve", "retired")

    def __init__(self, root: Path = Path("data/coscientist")):
        self.root = root
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def research_dir(self, research_id: str) -> Path:
        return self.root / research_id

    def hypotheses_dir(self, research_id: str) -> Path:
        return self.research_dir(research_id) / "hypotheses"

    def rounds_dir(self, research_id: str) -> Path:
        return self.research_dir(research_id) / "rounds"

    def reports_dir(self, research_id: str) -> Path:
        return self.research_dir(research_id) / "reports"

    def ensure_run_directories(self, research_id: str) -> None:
        self.research_dir(research_id).mkdir(parents=True, exist_ok=True)
        for stage in self.HYPOTHESIS_STAGES:
            (self.hypotheses_dir(research_id) / stage).mkdir(parents=True, exist_ok=True)
        self.rounds_dir(research_id).mkdir(parents=True, exist_ok=True)
        self.reports_dir(research_id).mkdir(parents=True, exist_ok=True)

    def research_goal_path(self, research_id: str) -> Path:
        return self.research_dir(research_id) / "research_goal.json"

    def hypothesis_path(self, research_id: str) -> Path:
        return self.hypotheses_dir(research_id)

    def report_path(self, research_id: str) -> Path:
        return self.reports_dir(research_id) / "reflection.md"

    def loop_report_path(self, research_id: str) -> Path:
        return self.reports_dir(research_id) / "loop.md"

    def ranking_path(self, research_id: str) -> Path:
        return self.rounds_dir(research_id) / "rankings.jsonl"

    def proximity_path(self, research_id: str) -> Path:
        return self.rounds_dir(research_id) / "proximity.jsonl"

    def meta_review_path(self, research_id: str) -> Path:
        return self.rounds_dir(research_id) / "meta_reviews.jsonl"

    def save_research_goal(self, document: ResearchGoalDocument) -> Path:
        self.ensure_run_directories(document.research_id)
        path = self.research_goal_path(document.research_id)
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_research_goal(self, research_id: str) -> ResearchGoalDocument:
        path = self.research_goal_path(research_id)
        return ResearchGoalDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def append_hypothesis_snapshot(self, hypothesis: Hypothesis) -> Path:
        return self.save_hypothesis(hypothesis)

    def save_hypothesis(self, hypothesis: Hypothesis) -> Path:
        self.ensure_run_directories(hypothesis.research_id)
        path = self.hypothesis_file_path(hypothesis)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(hypothesis.model_dump_json(indent=2), encoding="utf-8")
        self._remove_hypothesis_from_other_stages(hypothesis, keep_stage=path.parent.name)
        tmp_path.replace(path)
        return path

    def hypothesis_file_path(self, hypothesis: Hypothesis) -> Path:
        return self.hypotheses_dir(hypothesis.research_id) / self.hypothesis_stage(hypothesis) / f"{hypothesis.hypothesis_id}.json"

    def hypothesis_stage(self, hypothesis: Hypothesis) -> str:
        if not hypothesis.is_active or hypothesis.retired_reason or hypothesis.status == "retired":
            return "retired"
        if hypothesis.status == "evolve":
            return "evolve"
        if hypothesis.status == "generated":
            return "generated"
        return "reflected"

    def _remove_hypothesis_from_other_stages(self, hypothesis: Hypothesis, keep_stage: str) -> None:
        for stage in self.HYPOTHESIS_STAGES:
            if stage == keep_stage:
                continue
            path = self.hypotheses_dir(hypothesis.research_id) / stage / f"{hypothesis.hypothesis_id}.json"
            if path.exists():
                path.unlink()

    def load_hypothesis_snapshots(self, research_id: str) -> list[Hypothesis]:
        return [
            Hypothesis.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self._hypothesis_files(research_id)
        ]

    def latest_hypotheses(self, research_id: str) -> list[Hypothesis]:
        latest: dict[str, Hypothesis] = {}
        for snapshot in self.load_hypothesis_snapshots(research_id):
            latest[snapshot.hypothesis_id] = snapshot
        return list(latest.values())

    def load_hypotheses(
        self,
        research_id: str,
        stages: set[str] | None = None,
        active_only: bool = False,
    ) -> list[Hypothesis]:
        hypotheses = self.latest_hypotheses(research_id)
        if stages is not None:
            hypotheses = [hypothesis for hypothesis in hypotheses if self.hypothesis_stage(hypothesis) in stages]
        if active_only:
            hypotheses = [hypothesis for hypothesis in hypotheses if hypothesis.is_active]
        return hypotheses

    def _hypothesis_files(self, research_id: str) -> list[Path]:
        root = self.hypotheses_dir(research_id)
        if not root.exists():
            return []
        files: list[Path] = []
        for stage in self.HYPOTHESIS_STAGES:
            files.extend(sorted((root / stage).glob("*.json")))
        return files

    def write_report(self, research_id: str, content: str) -> Path:
        self.ensure_run_directories(research_id)
        path = self.report_path(research_id)
        path.write_text(content, encoding="utf-8")
        return path

    def write_loop_report(self, research_id: str, content: str) -> Path:
        self.ensure_run_directories(research_id)
        path = self.loop_report_path(research_id)
        path.write_text(content, encoding="utf-8")
        return path

    def append_ranking_round(self, ranking_round: RankingRound) -> Path:
        self.ensure_run_directories(ranking_round.research_id)
        path = self.ranking_path(ranking_round.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(ranking_round.model_dump_json())
            handle.write("\n")
        return path

    def load_ranking_rounds(self, research_id: str) -> list[RankingRound]:
        rounds: list[RankingRound] = []
        path = self.ranking_path(research_id)
        if not path.exists():
            return rounds
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(RankingRound.model_validate_json(line))
        return rounds

    def append_proximity_round(self, proximity_round: ProximityRound) -> Path:
        self.ensure_run_directories(proximity_round.research_id)
        path = self.proximity_path(proximity_round.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(proximity_round.model_dump_json())
            handle.write("\n")
        return path

    def load_proximity_rounds(self, research_id: str) -> list[ProximityRound]:
        rounds: list[ProximityRound] = []
        path = self.proximity_path(research_id)
        if not path.exists():
            return rounds
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(ProximityRound.model_validate_json(line))
        return rounds

    def append_meta_review_round(self, meta_review_round: MetaReviewRound) -> Path:
        self.ensure_run_directories(meta_review_round.research_id)
        path = self.meta_review_path(meta_review_round.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(meta_review_round.model_dump_json())
            handle.write("\n")
        return path

    def load_meta_review_rounds(self, research_id: str) -> list[MetaReviewRound]:
        rounds: list[MetaReviewRound] = []
        path = self.meta_review_path(research_id)
        if not path.exists():
            return rounds
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(MetaReviewRound.model_validate_json(line))
        return rounds

    @staticmethod
    def to_pretty_json(data: dict) -> str:
        return json.dumps(data, indent=2)
