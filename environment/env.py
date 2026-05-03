"""
HardeningEnv: RL environment for hardening an LLM against indirect prompt injections.

Each episode:
  - Resets to a fresh workspace with a base model, datasets, and training scripts
  - Exposes a bash action space (read / write / run files, invoke training)
  - Terminates when the agent submits /submission/model.py or exhausts its budget
  - Scores the submission with a Judge and returns a scalar reward in [0, 1]
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from agent.base import SYSTEM_PROMPT


@dataclass
class EpisodeState:
    compute_budget_remaining: float
    submitted: bool = False
    step_count: int = 0
    action_history: list = field(default_factory=list)


class HardeningEnv:
    """
    Gym-style RL environment.

    Observation space : str  (stdout/stderr of last action + budget status)
    Action space      : str  (bash command string)
    Reward            : float in [0, 1], returned only on terminal step
    """

    def __init__(
        self,
        base_model_name: str,
        workspace_dir: str,
        compute_budget: float = 100.0,
        max_steps: int = 50,
        judge=None,
        command_timeout: int = 300,
    ):
        """
        Args:
            base_model_name:  HuggingFace model id or local path for the base model.
            workspace_dir:    Root directory the agent operates in.
            compute_budget:   Abstract compute units. Training costs 10/run,
                              evaluation 2/run, other commands ~0.1/run.
            max_steps:        Hard cap on number of actions per episode.
            judge:            Judge instance. If None, reward is always 0.
            command_timeout:  Per-command timeout in seconds.
        """
        self.base_model_name = base_model_name
        self.workspace_dir = Path(workspace_dir)
        self.compute_budget = compute_budget
        self.max_steps = max_steps
        self.judge = judge
        self.command_timeout = command_timeout

        self._state: Optional[EpisodeState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Reset the environment and return the initial observation string."""
        self._state = EpisodeState(compute_budget_remaining=self.compute_budget)
        self._setup_workspace()
        return self._initial_observation()

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """
        Execute a bash command and advance the episode.

        Returns:
            observation : command output + budget footer
            reward      : 0.0 unless terminal, then judge score
            done        : True if submitted, budget exhausted, or max_steps hit
            info        : diagnostic dict
        """
        assert self._state is not None, "Call reset() before step()"

        self._state.step_count += 1
        self._state.action_history.append(action)

        stdout, cost = self._execute(action)
        self._state.compute_budget_remaining -= cost

        # Check for submission
        submission_path = self.workspace_dir / "submission" / "model.py"
        if submission_path.exists():
            self._state.submitted = True

        reward, done, terminal_msg = self._check_termination(submission_path)

        observation = stdout
        if terminal_msg:
            observation += f"\n\n[Environment] {terminal_msg}"
        observation += (
            f"\n\n[Budget: {self._state.compute_budget_remaining:.1f}"
            f" / {self.compute_budget:.1f} | Step {self._state.step_count}/{self.max_steps}]"
        )

        info = {
            "step": self._state.step_count,
            "compute_remaining": self._state.compute_budget_remaining,
            "submitted": self._state.submitted,
            "reward": reward,
        }
        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, action: str) -> Tuple[str, float]:
        """Run a bash command in the workspace. Returns (output, cost)."""
        try:
            result = subprocess.run(
                action,
                shell=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
            output = result.stdout
            if result.stderr.strip():
                output += f"\n[stderr]\n{result.stderr.strip()}"
            return (output.strip() or "[No output]"), self._cost(action)
        except subprocess.TimeoutExpired:
            return f"[Error] Command timed out after {self.command_timeout}s.", 5.0
        except Exception as exc:
            return f"[Error] {exc}", 0.0

    def _cost(self, action: str) -> float:
        """Heuristic compute cost for an action string."""
        lower = action.lower()
        if any(k in lower for k in ("train", "finetune", "fine-tune", "fit")):
            return 10.0
        if any(k in lower for k in ("eval", "evaluate", "test", "inference")):
            return 2.0
        if any(k in lower for k in ("pip install", "apt-get")):
            return 0.5
        return 0.1

    def _check_termination(
        self, submission_path: Path
    ) -> Tuple[float, bool, str]:
        """Return (reward, done, message)."""
        if self._state.submitted:
            reward = 0.0
            if self.judge is not None:
                try:
                    reward = self.judge.score(str(submission_path))
                except Exception as exc:
                    reward = 0.0
                    print(f"[Judge error] {exc}")
            return reward, True, f"Submission detected. Reward = {reward:.4f}. Episode complete."

        if self._state.compute_budget_remaining <= 0:
            return 0.0, True, "Compute budget exhausted. Episode failed."

        if self._state.step_count >= self.max_steps:
            return 0.0, True, "Max steps reached. Episode failed."

        return 0.0, False, ""

    def _setup_workspace(self):
        for subdir in ("data", "scripts", "checkpoints", "submission", "logs"):
            (self.workspace_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _initial_observation(self) -> str:
        files = sorted(self.workspace_dir.rglob("*"))
        file_list = "\n".join(
            f"  {f.relative_to(self.workspace_dir)}"
            for f in files
            if f.is_file()
        ) or "  (empty)"

        return (
            f"{SYSTEM_PROMPT}\n"
            f"{'='*60}\n"
            f"Workspace\n{file_list}\n\n"
            f"Base model : {self.base_model_name}\n"
            f"Budget     : {self.compute_budget:.1f} compute units\n"
            f"Max steps  : {self.max_steps}\n"
            f"{'='*60}\n\n"
            f"What would you like to do first?"
        )
