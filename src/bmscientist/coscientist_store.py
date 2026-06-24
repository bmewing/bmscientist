from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from bmscientist.coscientist_models import (
    Hypothesis,
    MetaReviewRound,
    ProximityRound,
    RankingRound,
    ResearchGoalDocument,
)


LOGGER = logging.getLogger(__name__)


class CoScientistStore:
    HYPOTHESIS_STAGES = ("generated", "reflecting", "reflected", "evolve", "retired")
    REFLECTION_QUEUE_LOCK_TIMEOUT_SECONDS = 30.0
    REFLECTION_QUEUE_LOCK_POLL_SECONDS = 0.05
    REFLECTION_FILE_RETRY_ATTEMPTS = 5
    _ADJECTIVES = (
        "amber",
        "ancient",
        "autumn",
        "bold",
        "brisk",
        "calm",
        "cedar",
        "clear",
        "cobalt",
        "crimson",
        "curious",
        "daring",
        "deep",
        "eager",
        "ember",
        "gentle",
        "golden",
        "grand",
        "hidden",
        "ivory",
        "jolly",
        "kind",
        "lively",
        "lunar",
        "mellow",
        "mossy",
        "nimble",
        "novel",
        "quiet",
        "rapid",
        "royal",
        "silver",
        "steady",
        "sunny",
        "swift",
        "tidy",
        "vivid",
        "wise",
    )
    _NATURE_WORDS = (
        "brook",
        "canyon",
        "cliff",
        "cloud",
        "coast",
        "creek",
        "dawn",
        "desert",
        "field",
        "forest",
        "garden",
        "glade",
        "harbor",
        "hill",
        "lake",
        "meadow",
        "mesa",
        "mist",
        "moon",
        "ocean",
        "pine",
        "prairie",
        "reef",
        "river",
        "shore",
        "sky",
        "spring",
        "stone",
        "summit",
        "valley",
        "willow",
        "wind",
    )
    _OBJECT_WORDS = (
        "anchor",
        "arrow",
        "atlas",
        "beacon",
        "bridge",
        "compass",
        "engine",
        "falcon",
        "forge",
        "harvest",
        "lantern",
        "ledger",
        "market",
        "matrix",
        "meridian",
        "module",
        "monarch",
        "orbit",
        "otter",
        "pilot",
        "ranger",
        "rocket",
        "signal",
        "sparrow",
        "tandem",
        "thunder",
        "voyage",
        "weaver",
    )

    def __init__(self, root: Path = Path("data/coscientist")):
        self.root = root
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def research_dir(self, research_id: str) -> Path:
        return self.root / research_id

    def project_exists(self, research_id: str) -> bool:
        return self.research_dir(research_id).exists()

    def claim_project_name(self, preferred_name: str | None = None) -> str:
        if preferred_name:
            project_name = self._uniquify_project_name(self._normalize_project_name(preferred_name))
        else:
            project_name = self._random_project_name()
        self.ensure_run_directories(project_name)
        return project_name

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

    def tool_report_path(self, research_id: str) -> Path:
        return self.reports_dir(research_id) / "tool_requests.md"

    def cost_report_path(self, research_id: str) -> Path:
        return self.reports_dir(research_id) / "cost.json"

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
        tmp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        tmp_path.write_text(hypothesis.model_dump_json(indent=2), encoding="utf-8")
        self._remove_hypothesis_from_other_stages(hypothesis, keep_stage=path.parent.name)
        tmp_path.replace(path)

        if self.hypothesis_stage(hypothesis) == "reflected":
            try:
                from bmscientist.graph_enrichment import GraphEnrichmentStore
                store = GraphEnrichmentStore()
                store.promote_hypothesis(hypothesis)
            except Exception as exc:
                LOGGER.warning("Failed to promote hypothesis %s to graph: %s", hypothesis.hypothesis_id, exc)

        return path

    def hypothesis_file_path(self, hypothesis: Hypothesis) -> Path:
        return self.hypotheses_dir(hypothesis.research_id) / self.hypothesis_stage(hypothesis) / f"{hypothesis.hypothesis_id}.json"

    def hypothesis_stage(self, hypothesis: Hypothesis) -> str:
        if not hypothesis.is_active or hypothesis.retired_reason or hypothesis.status == "retired":
            return "retired"
        if hypothesis.status == "evolve":
            return "evolve"
        if hypothesis.status == "reflecting":
            return "reflecting"
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

    def claim_next_generated_hypothesis(
        self,
        research_id: str,
        worker_id: str,
        lease_seconds: int = 1800,
    ) -> Hypothesis | None:
        self.ensure_run_directories(research_id)
        generated_dir = self.hypotheses_dir(research_id) / "generated"
        reflecting_dir = self.hypotheses_dir(research_id) / "reflecting"
        now = datetime.now(timezone.utc)
        lease_window = timedelta(seconds=max(1, lease_seconds))

        with self._reflection_queue_lock(research_id):
            for source_path in sorted(generated_dir.glob("*.json")):
                claimed_path = reflecting_dir / source_path.name
                if not self._claim_generated_file(source_path, claimed_path):
                    continue
                try:
                    claimed_path.touch()
                    hypothesis = Hypothesis.model_validate_json(claimed_path.read_text(encoding="utf-8"))
                except (OSError, PermissionError):
                    LOGGER.exception("Failed to read claimed hypothesis %s; releasing claim", claimed_path)
                    try:
                        claimed_path.rename(source_path)
                    except OSError:
                        LOGGER.exception("Failed to release unreadable claimed hypothesis %s", claimed_path)
                    continue

                claimed = hypothesis.model_copy(
                    update={
                        "status": "reflecting",
                        "reflection_worker_id": worker_id,
                        "reflection_claimed_at": now,
                        "reflection_lease_expires_at": now + lease_window,
                        "reflection_attempt_count": hypothesis.reflection_attempt_count + 1,
                        "reflection_error": None,
                    }
                )
                self.save_hypothesis(claimed)
                return claimed
        return None

    @contextmanager
    def _reflection_queue_lock(self, research_id: str) -> Iterator[None]:
        lock_path = self.hypotheses_dir(research_id) / ".reflection_queue.lock"
        deadline = time.monotonic() + self.REFLECTION_QUEUE_LOCK_TIMEOUT_SECONDS
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}".encode("utf-8"))
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for reflection queue lock: {lock_path}")
                time.sleep(self.REFLECTION_QUEUE_LOCK_POLL_SECONDS)
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(self.REFLECTION_QUEUE_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                LOGGER.warning("Could not remove reflection queue lock %s", lock_path, exc_info=True)

    def _claim_generated_file(self, source_path: Path, claimed_path: Path) -> bool:
        for attempt in range(self.REFLECTION_FILE_RETRY_ATTEMPTS):
            try:
                source_path.rename(claimed_path)
                return True
            except FileExistsError:
                return False
            except FileNotFoundError:
                return False
            except (OSError, PermissionError) as exc:
                if not source_path.exists():
                    return False
                if attempt + 1 >= self.REFLECTION_FILE_RETRY_ATTEMPTS:
                    LOGGER.warning(
                        "Could not claim generated hypothesis %s after %s attempts: %s",
                        source_path,
                        self.REFLECTION_FILE_RETRY_ATTEMPTS,
                        exc,
                    )
                    return False
                time.sleep(self.REFLECTION_QUEUE_LOCK_POLL_SECONDS * (2**attempt))
        return False

    def complete_reflection_claim(self, hypothesis: Hypothesis) -> Path:
        completed = hypothesis.model_copy(
            update={
                "status": "reflected",
                "reflection_worker_id": None,
                "reflection_claimed_at": None,
                "reflection_lease_expires_at": None,
                "reflection_error": None,
            }
        )
        return self.save_hypothesis(completed)

    def release_reflection_claim(self, hypothesis: Hypothesis, error: str | None = None) -> Path:
        released = hypothesis.model_copy(
            update={
                "status": "generated",
                "reflection_worker_id": None,
                "reflection_claimed_at": None,
                "reflection_lease_expires_at": None,
                "reflection_error": error.strip()[:500] if error else None,
            }
        )
        return self.save_hypothesis(released)

    def requeue_expired_reflection_claims(self, research_id: str) -> int:
        self.ensure_run_directories(research_id)
        reclaimed = 0
        now = datetime.now(timezone.utc)
        grace_cutoff = now - timedelta(seconds=5)
        reflecting_dir = self.hypotheses_dir(research_id) / "reflecting"

        for path in sorted(reflecting_dir.glob("*.json")):
            hypothesis = Hypothesis.model_validate_json(path.read_text(encoding="utf-8"))
            lease_expiry = hypothesis.reflection_lease_expires_at
            if lease_expiry is not None and lease_expiry > now:
                continue
            if lease_expiry is None:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if modified_at > grace_cutoff:
                    continue
            released = hypothesis.model_copy(
                update={
                    "status": "generated",
                    "reflection_worker_id": None,
                    "reflection_claimed_at": None,
                    "reflection_lease_expires_at": None,
                    "reflection_error": "Reflection lease expired before completion.",
                }
            )
            self.save_hypothesis(released)
            reclaimed += 1
        return reclaimed

    def load_hypothesis_snapshots(self, research_id: str) -> list[Hypothesis]:
        last_missing: FileNotFoundError | None = None
        for attempt in range(3):
            snapshots: list[Hypothesis] = []
            paths = self._hypothesis_files(research_id)
            saw_missing = False
            for path in paths:
                try:
                    snapshots.append(Hypothesis.model_validate_json(path.read_text(encoding="utf-8")))
                except FileNotFoundError as exc:
                    last_missing = exc
                    saw_missing = True
                    break
            if not saw_missing:
                return snapshots
        if last_missing is not None:
            LOGGER.warning(
                "Hypothesis snapshot changed during load for %s; returning best-effort snapshot set. Last missing path: %s",
                research_id,
                last_missing,
            )
        snapshots: list[Hypothesis] = []
        for path in self._hypothesis_files(research_id):
            try:
                snapshots.append(Hypothesis.model_validate_json(path.read_text(encoding="utf-8")))
            except FileNotFoundError:
                continue
        return snapshots

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

    def write_tool_report(self, research_id: str, content: str) -> Path:
        self.ensure_run_directories(research_id)
        path = self.tool_report_path(research_id)
        path.write_text(content, encoding="utf-8")
        return path

    def write_cost_report(self, research_id: str, payload: dict[str, object]) -> Path:
        self.ensure_run_directories(research_id)
        path = self.cost_report_path(research_id)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_cost_report(self, research_id: str) -> dict[str, object] | None:
        path = self.cost_report_path(research_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

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

    def _random_project_name(self) -> str:
        rng = secrets.SystemRandom()
        for _ in range(100):
            candidate = "-".join(
                (
                    rng.choice(self._ADJECTIVES),
                    rng.choice(self._NATURE_WORDS),
                    rng.choice(self._OBJECT_WORDS),
                )
            )
            if not self.project_exists(candidate):
                return candidate
        return self._uniquify_project_name(
            "-".join(
                (
                    rng.choice(self._ADJECTIVES),
                    rng.choice(self._NATURE_WORDS),
                    rng.choice(self._OBJECT_WORDS),
                )
            )
        )

    def _uniquify_project_name(self, base_name: str) -> str:
        if not self.project_exists(base_name):
            return base_name
        suffix = 2
        while self.project_exists(f"{base_name}-{suffix}"):
            suffix += 1
        return f"{base_name}-{suffix}"

    @staticmethod
    def _normalize_project_name(name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        return normalized or "project"

    def apply_hypothesis_feedback(
        self,
        research_id: str,
        hypothesis_id: str,
        volume: float | None = None,
        volume_unit: str | None = None,
        status: str | None = None,
        confidence: float | None = None,
        comment: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        candidate_material: str | None = None,
        incumbent_material: str | None = None,
        application: str | None = None,
        strategic_rationale: str | None = None,
    ) -> Hypothesis | None:
        """
        Applies feedback to a specific hypothesis in the research run.
        If status is 'retired' or 'rejected', retires the hypothesis.
        Also propagates the feedback to the graph.
        """
        hypotheses = self.load_hypotheses(research_id)
        target = None
        for hyp in hypotheses:
            if hyp.hypothesis_id == hypothesis_id:
                target = hyp
                break
                
        if not target:
            return None
            
        is_edited = any(v is not None for v in [title, summary, candidate_material, incumbent_material, application, strategic_rationale])
        if is_edited and not status:
            status = "edited"

        assessment = target.reflection_assessment
        if assessment:
            if comment:
                assessment.evidence_gap_notes.append(f"Human feedback: {comment}")
            if status:
                assessment.evidence_gap_notes.append(f"Human feedback status: {status}")
                
        update_dict = {}
        if status:
            update_dict["user_feedback_status"] = status
        if comment:
            update_dict["user_feedback_comment"] = comment
            
        if title is not None:
            update_dict["title"] = title
        if summary is not None:
            update_dict["summary"] = summary
        if candidate_material is not None:
            update_dict["candidate_material"] = candidate_material
        if incumbent_material is not None:
            update_dict["incumbent_material"] = incumbent_material
        if application is not None:
            update_dict["application"] = application
        if strategic_rationale is not None:
            update_dict["strategic_rationale"] = strategic_rationale

        if status == "rejected" or status == "retired":
            update_dict.update({
                "is_active": False,
                "status": "retired",
                "retired_reason": comment or f"Retired by user feedback (status: {status})",
            })
        else:
            notes = list(target.evolution_notes)
            if comment:
                notes.append(f"Feedback: {comment}")
            update_dict.update({
                "evolution_notes": notes,
                "is_active": True,
                "retired_reason": None,
            })
            if target.status == "retired" or target.retired_reason or not target.is_active:
                update_dict.update({
                    "status": "reflected",
                    "superseded_by_hypothesis_id": None,
                })
            
        target = target.model_copy(update=update_dict)
        self.save_hypothesis(target)
        
        from bmscientist.graph_enrichment import GraphEnrichmentStore
        graph_store = GraphEnrichmentStore()
        graph_status = "rejected" if status in ("rejected", "retired") else status
        graph_store.apply_edge_feedback(
            candidate_material=target.candidate_material or "",
            incumbent_material=target.incumbent_material,
            application=target.application or "",
            volume=volume,
            volume_unit=volume_unit,
            status=graph_status,
            confidence=confidence,
            comment=comment,
        )
        
        return target
