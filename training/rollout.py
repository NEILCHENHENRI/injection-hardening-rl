"""
training/rollout.py — collect multiple episode trajectories before each RL update.

Why collect multiple episodes before updating?
  Policy gradient estimates from a single episode have high variance.
  Averaging the gradient signal over K episodes per update step reduces
  variance substantially — this is the standard "batch of rollouts" pattern
  used in REINFORCE, PPO, and GRPO.

Sequential vs parallel
  True parallelism (running K episodes simultaneously) would require either:
    - K separate environment processes (expensive in memory)
    - K threads sharing the same environment (not safe)
  
  This implementation runs episodes sequentially and aggregates trajectories.
  A comment marks where you would plug in concurrent.futures or Ray if you
  wanted actual parallelism later.

Usage:
    collector = RolloutCollector(
        env_factory=lambda: HardeningEnv(...),
        agent=agent,
        episodes_per_update=4,
    )
    trajectories = collector.collect()   # runs 4 episodes, returns trajectories
"""
import time
from typing import Callable, List

from environment.env import HardeningEnv
from training.runner import EpisodeRunner
from training.buffer import Trajectory, TrajectoryBuffer


class RolloutCollector:
    """
    Runs `episodes_per_update` episodes sequentially and returns the trajectories.

    Args:
        env_factory         : zero-argument callable that returns a fresh HardeningEnv.
                              Called once per episode so each episode starts clean.
        agent               : agent instance (shared across episodes within an update)
        episodes_per_update : how many episodes to collect before each RL update
        buffer              : optional TrajectoryBuffer to push completed trajectories into
        verbose             : passed through to EpisodeRunner
    """

    def __init__(
        self,
        env_factory: Callable[[], HardeningEnv],
        agent,
        episodes_per_update: int = 4,
        buffer: TrajectoryBuffer = None,
        verbose: bool = True,
    ):
        self.env_factory = env_factory
        self.agent = agent
        self.episodes_per_update = episodes_per_update
        self.buffer = buffer
        self.verbose = verbose

        self._global_episode_counter = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def collect(self) -> List[Trajectory]:
        """
        Run `episodes_per_update` episodes and return the trajectory list.

        If a buffer was provided, trajectories are also pushed into it.
        """
        trajectories: List[Trajectory] = []
        t_start = time.time()

        for i in range(self.episodes_per_update):
            episode_id = self._global_episode_counter
            self._global_episode_counter += 1

            if self.verbose:
                print(
                    f"\n{'─'*50}\n"
                    f"Collecting episode {episode_id} "
                    f"({i+1}/{self.episodes_per_update} in this batch)\n"
                    f"{'─'*50}"
                )

            # Each episode gets a fresh environment instance.
            # To run in parallel: replace this loop with a ThreadPoolExecutor
            # or Ray remote call, one env per worker.
            env = self.env_factory()
            runner = EpisodeRunner(env=env, agent=self.agent, verbose=self.verbose)
            trajectory = runner.run_episode(episode_id=episode_id)

            trajectories.append(trajectory)
            if self.buffer is not None:
                self.buffer.push(trajectory)

        elapsed = time.time() - t_start
        mean_reward = sum(t.reward for t in trajectories) / len(trajectories)
        submit_rate = sum(1 for t in trajectories if t.submitted) / len(trajectories)

        if self.verbose:
            print(
                f"\n[RolloutCollector] Batch complete — "
                f"{len(trajectories)} episodes  "
                f"mean_reward={mean_reward:.4f}  "
                f"submit_rate={submit_rate:.2f}  "
                f"time={elapsed:.1f}s"
            )

        return trajectories

    @property
    def total_episodes(self) -> int:
        return self._global_episode_counter
