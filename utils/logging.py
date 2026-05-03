"""
utils/logging.py — per-episode metrics tracking and persistence.

Tracks:
  - Scalar reward components per episode
  - Action histogram (what kinds of commands the agent issues)
  - Training loss per RL update step (written by rl_train.py)

Usage:
    logger = MetricsLogger(log_dir="logs/run1")
    logger.log_episode(episode=0, result)
    logger.log_rl_step(step=0, loss=0.42)
    logger.save()                        # writes logs/run1/metrics.jsonl
    summary = logger.summary()           # dict of means over all episodes
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Import lazily so this module has no heavy deps
try:
    from environment.judge import JudgeResult
except ImportError:
    JudgeResult = None


class MetricsLogger:
    def __init__(self, log_dir: str, print_every: int = 1):
        """
        Args:
            log_dir     : directory where metrics.jsonl and summary.json are written
            print_every : print a one-line summary every N episodes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.print_every = print_every

        self._episode_records: list[dict] = []
        self._rl_step_records: list[dict] = []
        self._action_counts: dict[str, int] = defaultdict(int)

        self._metrics_path = self.log_dir / "metrics.jsonl"
        self._rl_path = self.log_dir / "rl_steps.jsonl"
        self._summary_path = self.log_dir / "summary.json"

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        reward: float,
        n_steps: int,
        submitted: bool,
        defense_rate: Optional[float] = None,
        compliance_rate: Optional[float] = None,
        quality_score: Optional[float] = None,
        actions: Optional[list[str]] = None,
        wall_time: Optional[float] = None,
    ):
        """
        Record one completed episode.

        Args:
            episode        : episode index
            reward         : final scalar reward
            n_steps        : number of step() calls in this episode
            submitted      : whether agent submitted before budget exhaustion
            defense_rate   : judge component score (if available)
            compliance_rate: judge component score (if available)
            quality_score  : judge component score (if available)
            actions        : list of action strings from this episode
            wall_time      : elapsed seconds for the episode
        """
        record = {
            "type": "episode",
            "episode": episode,
            "reward": reward,
            "n_steps": n_steps,
            "submitted": submitted,
            "defense_rate": defense_rate,
            "compliance_rate": compliance_rate,
            "quality_score": quality_score,
            "wall_time": wall_time,
            "timestamp": time.time(),
        }
        self._episode_records.append(record)

        # Action histogram
        if actions:
            for action in actions:
                bucket = self._action_bucket(action)
                self._action_counts[bucket] += 1

        # Append to file immediately (crash-safe)
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if self.print_every > 0 and (episode + 1) % self.print_every == 0:
            self._print_episode(record)

    def log_episode_from_result(
        self,
        episode: int,
        result,          # JudgeResult or None
        n_steps: int,
        submitted: bool,
        actions: Optional[list[str]] = None,
        wall_time: Optional[float] = None,
    ):
        """Convenience wrapper when you have a JudgeResult directly."""
        if result is not None:
            self.log_episode(
                episode=episode,
                reward=result.reward,
                n_steps=n_steps,
                submitted=submitted,
                defense_rate=result.defense_success_rate,
                compliance_rate=result.compliance_rate,
                quality_score=result.quality_score,
                actions=actions,
                wall_time=wall_time,
            )
        else:
            self.log_episode(
                episode=episode,
                reward=0.0,
                n_steps=n_steps,
                submitted=submitted,
                actions=actions,
                wall_time=wall_time,
            )

    # ------------------------------------------------------------------
    # RL step logging
    # ------------------------------------------------------------------

    def log_rl_step(self, step: int, loss: float, learning_rate: Optional[float] = None):
        """Record a single policy gradient update step."""
        record = {
            "type": "rl_step",
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "timestamp": time.time(),
        }
        self._rl_step_records.append(record)
        with self._rl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return aggregate stats over all logged episodes."""
        if not self._episode_records:
            return {}

        rewards = [r["reward"] for r in self._episode_records]
        n_steps_list = [r["n_steps"] for r in self._episode_records]
        submit_rate = sum(1 for r in self._episode_records if r["submitted"]) / len(
            self._episode_records
        )

        def _mean_optional(key):
            vals = [r[key] for r in self._episode_records if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        s = {
            "n_episodes": len(self._episode_records),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "submit_rate": submit_rate,
            "mean_steps": sum(n_steps_list) / len(n_steps_list),
            "mean_defense_rate": _mean_optional("defense_rate"),
            "mean_compliance_rate": _mean_optional("compliance_rate"),
            "mean_quality_score": _mean_optional("quality_score"),
            "action_counts": dict(self._action_counts),
            "n_rl_steps": len(self._rl_step_records),
        }
        return s

    def save(self):
        """Write summary.json to log_dir."""
        s = self.summary()
        with self._summary_path.open("w") as f:
            json.dump(s, f, indent=2)
        return s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _action_bucket(self, action: str) -> str:
        """Categorise an action string into a coarse bucket for the histogram."""
        lower = action.lower()
        if any(k in lower for k in ("train", "finetune", "fine-tune")):
            return "train"
        if any(k in lower for k in ("eval", "evaluate", "inference")):
            return "eval"
        if any(k in lower for k in ("cat", "head", "tail", "less", "grep", "ls")):
            return "inspect"
        if any(k in lower for k in ("python", "bash", "sh ")):
            return "script"
        if any(k in lower for k in ("cp ", "mv ", "mkdir", "rm ", "write")):
            return "file_op"
        if "pip install" in lower or "apt" in lower:
            return "install"
        return "other"

    def _print_episode(self, record: dict):
        dr = f"{record['defense_rate']:.3f}" if record["defense_rate"] is not None else "N/A"
        cr = f"{record['compliance_rate']:.3f}" if record["compliance_rate"] is not None else "N/A"
        qs = f"{record['quality_score']:.3f}" if record["quality_score"] is not None else "N/A"
        wt = f"{record['wall_time']:.1f}s" if record["wall_time"] is not None else "N/A"
        print(
            f"[Episode {record['episode']:>4}] "
            f"reward={record['reward']:.4f}  "
            f"defense={dr}  compliance={cr}  quality={qs}  "
            f"steps={record['n_steps']}  submitted={record['submitted']}  "
            f"time={wt}"
        )
