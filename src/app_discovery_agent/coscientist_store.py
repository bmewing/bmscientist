from __future__ import annotations

import json
from pathlib import Path

from app_discovery_agent.coscientist_models import Hypothesis, ResearchGoalDocument


class CoScientistStore:
    def __init__(self, root: Path = Path("data/coscientist")):
        self.root = root
        self.research_goals_dir = self.root / "research_goals"
        self.hypotheses_dir = self.root / "hypotheses"
        self.reports_dir = self.root / "reports"
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.research_goals_dir.mkdir(parents=True, exist_ok=True)
        self.hypotheses_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def research_goal_path(self, research_id: str) -> Path:
        return self.research_goals_dir / f"{research_id}.json"

    def hypothesis_path(self, research_id: str) -> Path:
        return self.hypotheses_dir / f"{research_id}.jsonl"

    def report_path(self, research_id: str) -> Path:
        return self.reports_dir / f"{research_id}_reflection.md"

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

    @staticmethod
    def to_pretty_json(data: dict) -> str:
        return json.dumps(data, indent=2)
