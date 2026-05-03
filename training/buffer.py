"""
training/buffer.py — trajectory storage for RL.

A Trajectory is the full record of one completed episode:
    steps : list of (observation, action) pairs
    reward: scalar reward returned by the judge at episode end

The buffer accumulates trajectories across episodes and exposes them
for the RL update in batches.

Design note — sparse reward
---------------------------
The judge only scores the final submission, so reward is received once
at the end of the episode. REINFORCE assigns this reward to every step
in the trajectory (no discount needed since the reward isn't sequential;
every action contributed equally to the final submission quality).

If you later add intermediate rewards (e.g. partial credit for eval
score improvements mid-episode), a discounted return is easy to add here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Step:
    """A single (observation, action) pair within an episode."""
    observation: str
    action: str


@dataclass
class Trajectory:
    """
    One completed episode.

    Attributes:
        episode_id : unique index assigned by the runner
        steps      : ordered list of (observation, action) pairs
        reward     : scalar reward from the judge (0.0 if not submitted)
        submitted  : whether the agent produced a submission
        n_steps    : convenience alias for len(steps)
    """
    episode_id: int
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    submitted: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def add_step(self, observation: str, action: str):
        self.steps.append(Step(observation=observation, action=action))

    def returns(self) -> list[float]:
        """
        Return the return (cumulative reward) assigned to each step.

        With sparse end-of-episode reward and no discounting, every step
        in the trajectory gets the same return: the final reward.
        """
        return [self.reward] * len(self.steps)


class TrajectoryBuffer:
    """
    Fixed-capacity FIFO buffer of completed trajectories.

    The RL training loop calls push() after each episode and
    sample_batch() when it's ready to do a policy update.

    Args:
        capacity : maximum number of trajectories to retain.
                   Oldest are dropped when capacity is exceeded.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._buffer: list[Trajectory] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, trajectory: Trajectory):
        """Add a completed trajectory, evicting the oldest if at capacity."""
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(trajectory)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sample_batch(self, n: int) -> list[Trajectory]:
        """
        Return up to n trajectories (most recent first).

        Does not remove them from the buffer — call clear() explicitly
        after an on-policy update.
        """
        return list(reversed(self._buffer))[:n]

    def all(self) -> list[Trajectory]:
        return list(self._buffer)

    def clear(self):
        """Empty the buffer. Call after an on-policy update step."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def mean_reward(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(t.reward for t in self._buffer) / len(self._buffer)

    def submit_rate(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(1 for t in self._buffer if t.submitted) / len(self._buffer)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._buffer)
