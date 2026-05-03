"""
training/runner.py — drives a single episode from reset() to done=True.

The runner is the glue between the environment and the agent:
  - calls env.reset() to get the first observation
  - feeds observations to agent.act() to get actions
  - calls env.step(action) and collects the result
  - assembles the completed Trajectory

Usage:
    runner = EpisodeRunner(env, agent, max_action_retries=2)
    trajectory = runner.run_episode(episode_id=0)
"""
import time

from environment.env import HardeningEnv
from training.buffer import Trajectory


class EpisodeRunner:
    """
    Runs one episode at a time.

    Args:
        env               : HardeningEnv instance (already configured)
        agent             : any agent with .act(obs) -> str and .reset_history()
        max_action_retries: if the agent returns an empty action, retry this
                            many times before substituting a safe no-op
        verbose           : print each (step, action, reward) line
    """

    def __init__(
        self,
        env: HardeningEnv,
        agent,
        max_action_retries: int = 2,
        verbose: bool = True,
    ):
        self.env = env
        self.agent = agent
        self.max_action_retries = max_action_retries
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_episode(self, episode_id: int = 0) -> Trajectory:
        """
        Run one complete episode.

        Returns a Trajectory containing every (observation, action) pair
        and the final reward signal.
        """
        self.agent.reset_history()
        trajectory = Trajectory(episode_id=episode_id)

        t_start = time.time()
        obs = self.env.reset()
        done = False
        reward = 0.0

        while not done:
            action = self._get_action(obs)
            trajectory.add_step(observation=obs, action=action)

            obs, reward, done, info = self.env.step(action)

            if self.verbose:
                step = info["step"]
                budget = info["compute_remaining"]
                print(
                    f"  [ep={episode_id} step={step:>3}] "
                    f"budget={budget:.1f}  "
                    f"action={action[:60]!r}"
                )

        trajectory.reward = reward
        trajectory.submitted = self.env._state.submitted if self.env._state else False

        elapsed = time.time() - t_start
        if self.verbose:
            print(
                f"[Episode {episode_id}] done — "
                f"reward={reward:.4f}  steps={trajectory.n_steps}  "
                f"submitted={trajectory.submitted}  time={elapsed:.1f}s"
            )

        return trajectory

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_action(self, obs: str) -> str:
        """
        Ask the agent for an action, retrying on empty responses.
        Falls back to a safe no-op if all retries fail.
        """
        for attempt in range(self.max_action_retries + 1):
            try:
                action = self.agent.act(obs)
                if action.strip():
                    return action
                if self.verbose and attempt < self.max_action_retries:
                    print(f"  [runner] Empty action (attempt {attempt+1}), retrying…")
            except Exception as exc:
                if self.verbose:
                    print(f"  [runner] Agent error on attempt {attempt+1}: {exc}")
        return "echo '[agent produced no action]'"
