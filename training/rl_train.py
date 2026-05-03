"""
training/rl_train.py — REINFORCE policy gradient training loop.

Algorithm: REINFORCE with a running mean baseline.

  For each update step:
    1. Collect K episodes via RolloutCollector
    2. For each trajectory, compute returns (sparse: same reward for all steps)
    3. Subtract the running mean baseline to reduce variance
    4. For each (observation, action) step, compute log π(action | obs)
    5. Loss = -mean(advantage × log_prob)   [policy gradient]
    6. Backprop + optimizer step
    7. Log metrics, checkpoint if improved

Why REINFORCE and not PPO/GRPO?
  REINFORCE is the simplest correct policy gradient algorithm. Given the
  environment is still being validated, starting simple makes debugging easier.
  PPO or GRPO can be swapped in by replacing the loss computation in
  _compute_loss() — the rest of the loop is algorithm-agnostic.

Usage:
    trainer = RLTrainer(
        agent=agent,
        env_factory=lambda: HardeningEnv(...),
        config=RLConfig(n_updates=100, episodes_per_update=4),
    )
    trainer.train()
"""
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from training.buffer import Trajectory, TrajectoryBuffer
from training.checkpoint import CheckpointManager
from training.rollout import RolloutCollector
from utils.logging import MetricsLogger


@dataclass
class RLConfig:
    # Rollout
    n_updates: int = 100               # number of policy update steps
    episodes_per_update: int = 4       # episodes collected before each update

    # Optimiser
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0         # gradient clipping

    # Baseline
    baseline_momentum: float = 0.95   # EMA coefficient for running mean baseline

    # Entropy bonus (encourages exploration)
    entropy_coef: float = 0.01

    # Checkpointing
    checkpoint_dir: str = "checkpoints/rl_run"
    save_every: int = 10
    keep_snapshots: bool = False

    # Logging
    log_dir: str = "logs/rl_run"
    print_every: int = 1


class RLTrainer:
    """
    Trains a LocalModelAgent via REINFORCE.

    Args:
        agent       : LocalModelAgent instance (must have .compute_log_probs())
        env_factory : zero-arg callable returning a fresh HardeningEnv
        config      : RLConfig dataclass
        resume      : if True, attempt to load from checkpoint_dir/latest before training
    """

    def __init__(
        self,
        agent,
        env_factory: Callable,
        config: RLConfig = None,
        resume: bool = False,
    ):
        self.agent = agent
        self.env_factory = env_factory
        self.config = config or RLConfig()

        import torch

        self.optimizer = torch.optim.AdamW(
            self.agent.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.buffer = TrajectoryBuffer(capacity=self.config.episodes_per_update * 4)

        self.collector = RolloutCollector(
            env_factory=env_factory,
            agent=agent,
            episodes_per_update=self.config.episodes_per_update,
            buffer=self.buffer,
            verbose=True,
        )

        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            save_every=self.config.save_every,
            keep_snapshots=self.config.keep_snapshots,
        )

        self.logger = MetricsLogger(
            log_dir=self.config.log_dir,
            print_every=self.config.print_every,
        )

        # Running baseline (EMA of mean return)
        self._baseline: float = 0.0

        self._start_update: int = 0
        if resume and self.checkpoint_mgr.has_checkpoint():
            self.agent, meta = self.checkpoint_mgr.load_latest(self.agent)
            self._start_update = meta.get("episode", 0)
            print(f"[RLTrainer] Resuming from update {self._start_update}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full RL training loop."""
        import torch

        print(
            f"\n{'='*60}\n"
            f"  RL Training\n"
            f"  Updates     : {self.config.n_updates}\n"
            f"  Episodes/upd: {self.config.episodes_per_update}\n"
            f"  LR          : {self.config.learning_rate}\n"
            f"{'='*60}\n"
        )

        for update in range(self._start_update, self.config.n_updates):
            t_update_start = time.time()

            # 1. Collect rollouts
            trajectories = self.collector.collect()

            # 2. Compute policy gradient loss
            self.agent.model.train()
            self.optimizer.zero_grad()

            loss, mean_reward = self._compute_loss(trajectories)

            # 3. Backprop + clip + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.agent.model.eval()

            elapsed = time.time() - t_update_start

            # 4. Log
            self.logger.log_rl_step(
                step=update,
                loss=loss.item(),
                learning_rate=self.config.learning_rate,
            )
            for i, traj in enumerate(trajectories):
                self.logger.log_episode(
                    episode=self.collector.total_episodes - len(trajectories) + i,
                    reward=traj.reward,
                    n_steps=traj.n_steps,
                    submitted=traj.submitted,
                    actions=[s.action for s in traj.steps],
                )

            print(
                f"[Update {update:>4}/{self.config.n_updates}] "
                f"loss={loss.item():.4f}  "
                f"mean_reward={mean_reward:.4f}  "
                f"baseline={self._baseline:.4f}  "
                f"time={elapsed:.1f}s"
            )

            # 5. Checkpoint
            self.checkpoint_mgr.save(
                agent=self.agent,
                episode=update,
                reward=mean_reward,
                optimizer_state=self.optimizer.state_dict(),
            )

            # 6. Clear buffer (on-policy)
            self.buffer.clear()

        summary = self.logger.save()
        print(f"\n[RLTrainer] Training complete. Summary:\n{summary}")
        return summary

    # ------------------------------------------------------------------
    # Loss computation (REINFORCE with running baseline)
    # ------------------------------------------------------------------

    def _compute_loss(self, trajectories: List[Trajectory]):
        """
        REINFORCE loss:  L = -E[ advantage × log π(a|s) ]

        advantage = return - baseline
        return    = episode reward (same for all steps, sparse reward)
        baseline  = EMA of mean return (reduces variance)
        """
        import torch

        rewards = [t.reward for t in trajectories]
        mean_reward = sum(rewards) / len(rewards)

        # Update EMA baseline
        self._baseline = (
            self.config.baseline_momentum * self._baseline
            + (1 - self.config.baseline_momentum) * mean_reward
        )

        policy_losses = []
        for trajectory in trajectories:
            advantage = trajectory.reward - self._baseline
            for step in trajectory.steps:
                # Reconstruct message history up to this step
                step_idx = trajectory.steps.index(step)
                messages = self._reconstruct_messages(trajectory, step_idx)

                log_prob = self.agent.compute_log_probs(messages, step.action)
                policy_losses.append(-advantage * log_prob)

        if not policy_losses:
            return torch.tensor(0.0, requires_grad=True), mean_reward

        loss = torch.stack(policy_losses).mean()
        return loss, mean_reward

    def _reconstruct_messages(self, trajectory: Trajectory, up_to_step: int) -> list[dict]:
        """
        Reconstruct the message history the agent had at step `up_to_step`.

        Format: alternating user (observation) / assistant (action) messages,
        up to but not including the current step's action.
        """
        messages = []
        for i, step in enumerate(trajectory.steps[:up_to_step]):
            messages.append({"role": "user", "content": step.observation})
            messages.append({"role": "assistant", "content": step.action})
        # Add current step's observation (the model must predict the action)
        messages.append({"role": "user", "content": trajectory.steps[up_to_step].observation})
        return messages
