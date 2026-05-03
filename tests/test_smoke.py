"""
Smoke test suite.

Verifies the full pipeline runs syntactically and logically without real compute
or network access.

Model strategy
--------------
Tests that need a real model use a locally-constructed GPT-2-sized mock
(~50 k parameters, pure PyTorch, no downloads required).  The mock generates
plausible-length token sequences so the judge can run end-to-end.

For integration testing against a real downloaded model, set the environment
variable SMOKE_USE_HF=1 and the tests will fall back to sshleifer/tiny-gpt2.

Run with pytest:
    pytest tests/test_smoke.py -v

Or standalone (no pytest required):
    python tests/test_smoke.py
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.datasets import (
    load_adversarial_dataset,
    load_benign_dataset,
    save_dataset,
)
from environment.judge import Judge
from environment.env import HardeningEnv

TINY_MODEL = "sshleifer/tiny-gpt2"
USE_HF = os.environ.get("SMOKE_USE_HF", "0") == "1"


# ---------------------------------------------------------------------------
# Minimal mock model + tokenizer (no network required)
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Mimics the PreTrainedTokenizer interface used by Judge._generate()."""

    eos_token_id = 0
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        # Encode as byte values mod 50 → vocab_size
        ids = [b % 50 for b in text.encode("utf-8")][:max_length]
        id_tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": id_tensor, "attention_mask": torch.ones_like(id_tensor)}

    def decode(self, token_ids, skip_special_tokens=True):
        # Return a plausible, non-empty string so quality heuristic scores it
        return "This is a helpful response to the user's request."

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class _MockModel(nn.Module):
    """
    Minimal causal LM: embedding → single linear head.
    Supports .generate() by greedily sampling from the head logits.
    """

    vocab_size = 50

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(self.vocab_size, 16)
        self.head = nn.Linear(16, self.vocab_size, bias=False)
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed(input_ids)          # (B, T, 16)
        logits = self.head(h)              # (B, T, vocab)
        # Return a namespace-like object with .logits
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        return out

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=10,
                 do_sample=False, pad_token_id=0, **kwargs):
        B, T = input_ids.shape
        cur = input_ids
        for _ in range(max_new_tokens):
            logits = self.head(self.embed(cur))   # (B, T+i, vocab)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
            cur = torch.cat([cur, next_id], dim=1)
        return cur

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "mock_weights.pt")


def _make_mock_fixture():
    """Return (model, tokenizer) backed by the local mock — no downloads."""
    model = _MockModel()
    model.eval()
    tokenizer = _MockTokenizer()
    return model, tokenizer


def _make_mock_model_py(tmp_path: Path) -> Path:
    """Write a model.py that loads _MockModel, usable by the Judge."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent.parent
    model_py = tmp_path / "model.py"
    model_py.write_text(
        f"""
import sys
sys.path.insert(0, "{src_dir}")
from tests.test_smoke import _MockModel, _MockTokenizer

def load_model():
    model = _MockModel()
    model.eval()
    return model, _MockTokenizer()
"""
    )
    return model_py


def _load_hf_tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    return model, tokenizer


def _load_fixture():
    if USE_HF:
        print(f"  (using HF model: {TINY_MODEL})")
        return _load_hf_tiny()
    return _make_mock_fixture()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_synthetic_datasets_nonempty():
    """Synthetic datasets are non-empty and have required keys."""
    adv = load_adversarial_dataset()
    ben = load_benign_dataset()

    assert len(adv) > 0
    assert len(ben) > 0
    for ex in adv:
        assert "prompt" in ex and "injected_instruction" in ex
    for ex in ben:
        assert "prompt" in ex


def test_dataset_save_reload(tmp_path):
    """Round-trip: save dataset to JSONL and reload it."""
    adv = load_adversarial_dataset()
    path = str(tmp_path / "adv.jsonl")
    save_dataset(adv, path)
    reloaded = load_adversarial_dataset(path)
    assert len(reloaded) == len(adv)
    assert reloaded[0]["prompt"] == adv[0]["prompt"]


def test_judge_defense_rate_in_range(model_fixture):
    """Defense success rate must be in [0, 1]."""
    model, tokenizer = model_fixture
    judge = Judge(load_adversarial_dataset(), load_benign_dataset(), verbose=False)
    rate, _ = judge._evaluate_defense(model, tokenizer)
    assert 0.0 <= rate <= 1.0, f"Out of range: {rate}"


def test_judge_compliance_rate_in_range(model_fixture):
    """Compliance rate and quality score must be in [0, 1]."""
    model, tokenizer = model_fixture
    judge = Judge(load_adversarial_dataset(), load_benign_dataset(), verbose=False)
    compliance, quality, _ = judge._evaluate_benign(model, tokenizer)
    assert 0.0 <= compliance <= 1.0, f"compliance out of range: {compliance}"
    assert 0.0 <= quality <= 1.0, f"quality out of range: {quality}"


def test_reward_is_multiplicative(model_fixture):
    """Reward must equal the product of the three component scores."""
    model, tokenizer = model_fixture
    judge = Judge(load_adversarial_dataset(), load_benign_dataset(), verbose=False)
    result = judge.evaluate(model, tokenizer)

    expected = (
        result.defense_success_rate
        * result.compliance_rate
        * result.quality_score
    )
    assert abs(result.reward - expected) < 1e-9, (
        f"Reward {result.reward} ≠ product {expected}"
    )
    assert 0.0 <= result.reward <= 1.0


def test_env_reset_returns_prompt(tmp_path):
    """reset() returns an observation that mentions the key task."""
    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(tmp_path / "env_reset"),
        compute_budget=50.0,
        max_steps=10,
    )
    obs = env.reset()
    assert "prompt injection" in obs.lower() or "hardening" in obs.lower()
    assert "mock-model" in obs


def test_env_step_echo(tmp_path):
    """A benign step should capture stdout and not be terminal."""
    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(tmp_path / "env_step"),
        compute_budget=50.0,
        max_steps=10,
    )
    env.reset()
    obs, reward, done, info = env.step("echo 'hello agent'")
    assert "hello agent" in obs
    assert reward == 0.0
    assert not done


def test_env_budget_depletes(tmp_path):
    """Budget decreases after each step."""
    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(tmp_path / "env_budget"),
        compute_budget=10.0,
        max_steps=100,
    )
    env.reset()
    _, _, _, info = env.step("echo x")
    assert info["compute_remaining"] < 10.0


def test_env_max_steps_terminates(tmp_path):
    """Episode ends after max_steps even without submission."""
    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(tmp_path / "env_steps"),
        compute_budget=999.0,
        max_steps=2,
    )
    env.reset()
    env.step("echo step1")
    _, _, done, _ = env.step("echo step2")
    assert done


def test_env_submission_detected(tmp_path):
    """Placing model.py in submission/ triggers episode end."""
    workspace = tmp_path / "env_submit"
    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(workspace),
        compute_budget=50.0,
        max_steps=20,
        judge=None,
    )
    env.reset()

    # Write a minimal submission before the next step
    (workspace / "submission").mkdir(parents=True, exist_ok=True)
    _make_mock_model_py(workspace / "submission")
    # Rename to model.py (mock writer already names it model.py)

    _, _, done, info = env.step("ls submission/")
    assert done
    assert info["submitted"]


def test_dynamic_submission_load(tmp_path):
    """Judge can dynamically import a model.py and call load_model()."""
    model_py = _make_mock_model_py(tmp_path / "dyn")
    judge = Judge(
        load_adversarial_dataset(),
        load_benign_dataset(),
        verbose=False,
    )
    model, tokenizer = judge._load_submission(str(model_py))
    assert model is not None
    assert tokenizer is not None


def test_end_to_end_score(tmp_path):
    """Full judge.score() path: load submission → evaluate → return float."""
    model_py = _make_mock_model_py(tmp_path / "e2e")
    judge = Judge(
        load_adversarial_dataset(),
        load_benign_dataset(),
        verbose=False,
    )
    reward = judge.score(str(model_py))
    assert 0.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# New module tests
# ---------------------------------------------------------------------------

# ── utils/logging ──────────────────────────────────────────────────────────

def test_logger_episode_record(tmp_path):
    """MetricsLogger records an episode and produces a correct summary."""
    from utils.logging import MetricsLogger

    logger = MetricsLogger(log_dir=str(tmp_path / "logs"), print_every=0)
    logger.log_episode(
        episode=0,
        reward=0.6,
        n_steps=5,
        submitted=True,
        defense_rate=0.8,
        compliance_rate=0.9,
        quality_score=0.75,
        actions=["python train.py", "ls", "cat data/adv.jsonl"],
        wall_time=12.3,
    )
    s = logger.summary()
    assert s["n_episodes"] == 1
    assert abs(s["mean_reward"] - 0.6) < 1e-9
    assert s["submit_rate"] == 1.0
    assert s["action_counts"].get("train") == 1
    assert s["action_counts"].get("inspect") == 2   # ls + cat


def test_logger_save_creates_file(tmp_path):
    """save() writes summary.json to log_dir."""
    import json
    from utils.logging import MetricsLogger

    logger = MetricsLogger(log_dir=str(tmp_path / "logs2"), print_every=0)
    logger.log_episode(episode=0, reward=0.1, n_steps=3, submitted=False)
    logger.save()
    summary_path = tmp_path / "logs2" / "summary.json"
    assert summary_path.exists()
    with summary_path.open() as f:
        data = json.load(f)
    assert "mean_reward" in data


def test_logger_rl_step(tmp_path):
    """log_rl_step() appends to rl_steps.jsonl."""
    from utils.logging import MetricsLogger

    logger = MetricsLogger(log_dir=str(tmp_path / "logs3"), print_every=0)
    logger.log_rl_step(step=0, loss=0.42, learning_rate=1e-5)
    assert logger._rl_step_records[0]["loss"] == 0.42


# ── training/buffer ────────────────────────────────────────────────────────

def test_buffer_push_and_len():
    """Buffer tracks pushed trajectories and enforces capacity."""
    from training.buffer import Trajectory, TrajectoryBuffer

    buf = TrajectoryBuffer(capacity=3)
    for i in range(4):
        t = Trajectory(episode_id=i, reward=float(i))
        buf.push(t)

    assert len(buf) == 3  # capacity enforced
    # Oldest (episode 0) should have been evicted
    assert buf.all()[0].episode_id == 1


def test_buffer_returns_are_constant():
    """Sparse reward: every step in a trajectory gets the same return."""
    from training.buffer import Step, Trajectory

    t = Trajectory(episode_id=0, reward=0.42)
    t.add_step("obs1", "action1")
    t.add_step("obs2", "action2")
    t.add_step("obs3", "action3")
    returns = t.returns()
    assert all(r == 0.42 for r in returns)
    assert len(returns) == 3


def test_buffer_clear():
    """clear() empties the buffer."""
    from training.buffer import Trajectory, TrajectoryBuffer

    buf = TrajectoryBuffer(capacity=10)
    buf.push(Trajectory(episode_id=0, reward=0.5))
    buf.clear()
    assert len(buf) == 0


def test_buffer_mean_reward():
    """mean_reward() averages correctly."""
    from training.buffer import Trajectory, TrajectoryBuffer

    buf = TrajectoryBuffer()
    buf.push(Trajectory(episode_id=0, reward=0.2))
    buf.push(Trajectory(episode_id=1, reward=0.8))
    assert abs(buf.mean_reward() - 0.5) < 1e-9


# ── training/checkpoint ────────────────────────────────────────────────────

class _SaveableAgent:
    """Minimal agent that supports save/load for checkpoint tests."""
    def __init__(self):
        self.saved_to = None
        self.loaded_from = None

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved_to = path

    def load(self, path):
        self.loaded_from = path


def test_checkpoint_save_creates_latest(tmp_path):
    from training.checkpoint import CheckpointManager
    agent = _SaveableAgent()
    ckpt = CheckpointManager(str(tmp_path / "ckpts"), save_every=5)
    ckpt.save(agent, episode=0, reward=0.3)
    assert (tmp_path / "ckpts" / "latest").exists()
    assert (tmp_path / "ckpts" / "latest" / "meta.json").exists()


def test_checkpoint_best_updated_on_improvement(tmp_path):
    from training.checkpoint import CheckpointManager
    agent = _SaveableAgent()
    ckpt = CheckpointManager(str(tmp_path / "ckpts2"), save_every=5)
    ckpt.save(agent, episode=0, reward=0.2)
    ckpt.save(agent, episode=1, reward=0.5)  # improvement → best updated
    assert (tmp_path / "ckpts2" / "best").exists()
    assert ckpt.best_reward == 0.5


def test_checkpoint_load_latest(tmp_path):
    from training.checkpoint import CheckpointManager
    agent = _SaveableAgent()
    ckpt = CheckpointManager(str(tmp_path / "ckpts3"), save_every=5)
    ckpt.save(agent, episode=0, reward=0.1)
    loaded_agent, meta = ckpt.load_latest(_SaveableAgent())
    assert loaded_agent.loaded_from is not None
    assert "episode" in meta


def test_checkpoint_has_checkpoint(tmp_path):
    from training.checkpoint import CheckpointManager
    ckpt = CheckpointManager(str(tmp_path / "empty_ckpts"), save_every=5)
    assert not ckpt.has_checkpoint()
    ckpt.save(_SaveableAgent(), episode=0, reward=0.0)
    assert ckpt.has_checkpoint()


# ── agent/agent ────────────────────────────────────────────────────────────

class _EchoAgent:
    """Minimal agent that echoes a fixed action — no model required."""

    def __init__(self, action="echo hello"):
        self._action = action
        self._history = []

    def act(self, obs):
        self._history.append({"role": "user", "content": obs})
        self._history.append({"role": "assistant", "content": self._action})
        return self._action

    def reset_history(self):
        self._history = []

    def save(self, path): pass
    def load(self, path): pass


def test_base_agent_history_trim():
    """BaseAgent trims history to max_history_turns."""
    from agent.base import BaseAgent

    class _TestAgent(BaseAgent):
        def _call_model(self, messages):
            return "echo ok"

    agent = _TestAgent(max_history_turns=2)
    for i in range(10):
        agent.act(f"observation {i}")

    # max_history_turns=2 → keep last 4 messages (2 user + 2 assistant)
    assert len(agent.history) <= 4


def test_base_agent_clean_action():
    """BaseAgent strips markdown fences from LLM responses."""
    from agent.base import BaseAgent

    class _FenceAgent(BaseAgent):
        def _call_model(self, messages):
            return "```bash\necho hello\n```"

    agent = _FenceAgent()
    action = agent.act("some observation")
    assert action == "echo hello"
    assert "```" not in action


# ── runner ─────────────────────────────────────────────────────────────────

def test_runner_produces_trajectory(tmp_path):
    """EpisodeRunner runs an episode and returns a Trajectory."""
    from environment.env import HardeningEnv
    from training.runner import EpisodeRunner

    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(tmp_path / "runner_env"),
        compute_budget=50.0,
        max_steps=3,
    )
    agent = _EchoAgent(action="echo step")
    runner = EpisodeRunner(env=env, agent=agent, verbose=False)
    traj = runner.run_episode(episode_id=0)

    assert traj.episode_id == 0
    assert traj.n_steps == 3            # max_steps reached
    assert traj.reward == 0.0           # no submission
    assert not traj.submitted


def test_runner_submission_terminates(tmp_path):
    """EpisodeRunner ends when agent writes submission/model.py."""
    from environment.env import HardeningEnv
    from training.runner import EpisodeRunner

    workspace = tmp_path / "runner_submit"
    env = HardeningEnv(
        base_model_name="mock-model",
        workspace_dir=str(workspace),
        compute_budget=50.0,
        max_steps=20,
    )

    src_dir = Path(__file__).resolve().parent.parent
    submission_cmd = (
        f"mkdir -p submission && python3 -c \""
        f"import sys; sys.path.insert(0, '{src_dir}'); "
        f"from tests.test_smoke import _MockModel, _MockTokenizer; "
        f"open('submission/model.py', 'w').write("
        f"\\\"import sys; sys.path.insert(0, '{src_dir}')\\\\n"
        f"from tests.test_smoke import _MockModel, _MockTokenizer\\\\n"
        f"def load_model(): return _MockModel(), _MockTokenizer()\\\\n\\\")\""
    )

    # Two-step agent: first echo, then submit
    actions = ["echo warming up", submission_cmd]
    call_count = [0]

    class _TwoStepAgent:
        def act(self, obs):
            idx = min(call_count[0], len(actions) - 1)
            call_count[0] += 1
            return actions[idx]
        def reset_history(self): pass
        def save(self, p): pass
        def load(self, p): pass

    runner = EpisodeRunner(env=env, agent=_TwoStepAgent(), verbose=False)
    traj = runner.run_episode(episode_id=0)
    assert traj.submitted
    assert traj.n_steps == 2


# ── training/rollout ───────────────────────────────────────────────────────

def test_rollout_collector_returns_k_trajectories(tmp_path):
    """RolloutCollector produces exactly episodes_per_update trajectories."""
    from environment.env import HardeningEnv
    from training.buffer import TrajectoryBuffer
    from training.rollout import RolloutCollector

    def _make_env():
        return HardeningEnv(
            base_model_name="mock-model",
            workspace_dir=str(tmp_path / f"rollout_env_{id(object())}"),
            compute_budget=10.0,
            max_steps=2,
        )

    buf = TrajectoryBuffer()
    collector = RolloutCollector(
        env_factory=_make_env,
        agent=_EchoAgent(),
        episodes_per_update=3,
        buffer=buf,
        verbose=False,
    )
    trajectories = collector.collect()
    assert len(trajectories) == 3
    assert len(buf) == 3
    assert collector.total_episodes == 3


# ---------------------------------------------------------------------------
# Pytest fixtures (auto-used when run under pytest)
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.fixture(scope="module")
    def model_fixture():
        return _load_fixture()

except ImportError:
    pass  # Running standalone below


# ---------------------------------------------------------------------------
# Standalone runner (no pytest dependency)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model fixture…")
    _fixture = _load_fixture()
    print("Ready.\n")

    with tempfile.TemporaryDirectory() as _tmp_str:
        _tmp = Path(_tmp_str)

        _tests = [
            # Original environment tests
            ("Synthetic datasets nonempty",
             lambda: test_synthetic_datasets_nonempty()),
            ("Dataset save/reload",
             lambda: test_dataset_save_reload(_tmp / "ds")),
            ("Defense rate in range",
             lambda: test_judge_defense_rate_in_range(_fixture)),
            ("Compliance rate in range",
             lambda: test_judge_compliance_rate_in_range(_fixture)),
            ("Reward is multiplicative",
             lambda: test_reward_is_multiplicative(_fixture)),
            ("Env reset returns prompt",
             lambda: test_env_reset_returns_prompt(_tmp)),
            ("Env step echo",
             lambda: test_env_step_echo(_tmp)),
            ("Budget depletes",
             lambda: test_env_budget_depletes(_tmp)),
            ("Max steps terminates",
             lambda: test_env_max_steps_terminates(_tmp)),
            ("Submission detected",
             lambda: test_env_submission_detected(_tmp)),
            ("Dynamic submission load",
             lambda: test_dynamic_submission_load(_tmp / "dyn")),
            ("End-to-end score",
             lambda: test_end_to_end_score(_tmp / "e2e")),
            # utils/logging
            ("Logger episode record",
             lambda: test_logger_episode_record(_tmp / "log1")),
            ("Logger save creates file",
             lambda: test_logger_save_creates_file(_tmp / "log2")),
            ("Logger RL step",
             lambda: test_logger_rl_step(_tmp / "log3")),
            # training/buffer
            ("Buffer push and len",
             lambda: test_buffer_push_and_len()),
            ("Buffer returns are constant",
             lambda: test_buffer_returns_are_constant()),
            ("Buffer clear",
             lambda: test_buffer_clear()),
            ("Buffer mean reward",
             lambda: test_buffer_mean_reward()),
            # training/checkpoint
            ("Checkpoint save creates latest",
             lambda: test_checkpoint_save_creates_latest(_tmp / "ck1")),
            ("Checkpoint best updated on improvement",
             lambda: test_checkpoint_best_updated_on_improvement(_tmp / "ck2")),
            ("Checkpoint load latest",
             lambda: test_checkpoint_load_latest(_tmp / "ck3")),
            ("Checkpoint has_checkpoint",
             lambda: test_checkpoint_has_checkpoint(_tmp / "ck4")),
            # agent
            ("Base agent history trim",
             lambda: test_base_agent_history_trim()),
            ("Base agent clean action",
             lambda: test_base_agent_clean_action()),
            # runner
            ("Runner produces trajectory",
             lambda: test_runner_produces_trajectory(_tmp / "run1")),
            ("Runner submission terminates",
             lambda: test_runner_submission_terminates(_tmp / "run2")),
            # rollout
            ("Rollout collector returns K trajectories",
             lambda: test_rollout_collector_returns_k_trajectories(_tmp / "rc1")),
        ]

        passed, failed = 0, 0
        for name, fn in _tests:
            try:
                fn()
                print(f"  ✓  {name}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {name}")
                traceback.print_exc()
                failed += 1

        print(f"\n{'─'*40}")
        print(f"  {passed}/{passed + failed} tests passed.")
        sys.exit(0 if failed == 0 else 1)

