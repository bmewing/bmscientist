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
    def __init__(self, root: Path = Path("data/coscientist")):
        self.root = root
        self.research_goals_dir = self.root / "research_goals"
        self.hypotheses_dir = self.root / "hypotheses"
        self.rankings_dir = self.root / "rankings"
        self.proximity_dir = self.root / "proximity"
        self.meta_reviews_dir = self.root / "meta_reviews"
        self.reports_dir = self.root / "reports"
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.research_goals_dir.mkdir(parents=True, exist_ok=True)
        self.hypotheses_dir.mkdir(parents=True, exist_ok=True)
        self.rankings_dir.mkdir(parents=True, exist_ok=True)
        self.proximity_dir.mkdir(parents=True, exist_ok=True)
        self.meta_reviews_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def research_goal_path(self, research_id: str) -> Path:
        return self.research_goals_dir / f"{research_id}.json"

    def hypothesis_path(self, research_id: str) -> Path:
        return self.hypotheses_dir / f"{research_id}.jsonl"

    def report_path(self, research_id: str) -> Path:
        return self.reports_dir / f"{research_id}_reflection.md"

    def loop_report_path(self, research_id: str) -> Path:
        return self.reports_dir / f"{research_id}_loop.md"

    def ranking_path(self, research_id: str) -> Path:
        return self.rankings_dir / f"{research_id}.jsonl"

    def proximity_path(self, research_id: str) -> Path:
        return self.proximity_dir / f"{research_id}.jsonl"

    def meta_review_path(self, research_id: str) -> Path:
        return self.meta_reviews_dir / f"{research_id}.jsonl"

    def save_research_goal(self, document: ResearchGoalDocument) -> Path:
        path = self.research_goal_path(document.research_id)
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_research_goal(self, research_id: str) -> ResearchGoalDocument:
        path = self.research_goal_path(research_id)
        return ResearchGoalDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def append_hypothesis_snapshot(self, hypothesis: Hypothesis) -> Path:
        path = self.hypothesis_path(hypothesis.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(hypothesis.model_dump_json())
            handle.write("\n")
        return path

    def load_hypothesis_snapshots(self, research_id: str) -> list[Hypothesis]:
        path = self.hypothesis_path(research_id)
        if not path.exists():
            return []
        snapshots: list[Hypothesis] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            snapshots.append(Hypothesis.model_validate_json(line))
        return snapshots

    def latest_hypotheses(self, research_id: str) -> list[Hypothesis]:
        latest: dict[str, Hypothesis] = {}
        for snapshot in self.load_hypothesis_snapshots(research_id):
            latest[snapshot.hypothesis_id] = snapshot
        return list(latest.values())

    def write_report(self, research_id: str, content: str) -> Path:
        path = self.report_path(research_id)
        path.write_text(content, encoding="utf-8")
        return path

    def write_loop_report(self, research_id: str, content: str) -> Path:
        path = self.loop_report_path(research_id)
        path.write_text(content, encoding="utf-8")
        return path

    def append_ranking_round(self, ranking_round: RankingRound) -> Path:
        path = self.ranking_path(ranking_round.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(ranking_round.model_dump_json())
            handle.write("\n")
        return path

    def load_ranking_rounds(self, research_id: str) -> list[RankingRound]:
        path = self.ranking_path(research_id)
        if not path.exists():
            return []
        rounds: list[RankingRound] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(RankingRound.model_validate_json(line))
        return rounds

    def append_proximity_round(self, proximity_round: ProximityRound) -> Path:
        path = self.proximity_path(proximity_round.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(proximity_round.model_dump_json())
            handle.write("\n")
        return path

    def load_proximity_rounds(self, research_id: str) -> list[ProximityRound]:
        path = self.proximity_path(research_id)
        if not path.exists():
            return []
        rounds: list[ProximityRound] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(ProximityRound.model_validate_json(line))
        return rounds

    def append_meta_review_round(self, meta_review_round: MetaReviewRound) -> Path:
        path = self.meta_review_path(meta_review_round.research_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(meta_review_round.model_dump_json())
            handle.write("\n")
        return path

    def load_meta_review_rounds(self, research_id: str) -> list[MetaReviewRound]:
        path = self.meta_review_path(research_id)
        if not path.exists():
            return []
        rounds: list[MetaReviewRound] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rounds.append(MetaReviewRound.model_validate_json(line))
        return rounds

    @staticmethod
    def to_pretty_json(data: dict) -> str:
        return json.dumps(data, indent=2)
