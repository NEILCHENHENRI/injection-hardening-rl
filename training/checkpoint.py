"""
training/checkpoint.py — save and restore agent training state.

Saves:
  - Agent model weights (via agent.save())
  - Training metadata: episode count, best reward, optimizer state

Layout on disk:
  checkpoints/
    latest/          ← always overwritten
    best/            ← overwritten only when reward improves
    episode_0042/    ← optional periodic snapshots

Usage:
    ckpt = CheckpointManager("checkpoints/run1", save_every=10)
    ckpt.save(agent, episode=0, reward=0.0)
    ckpt.save(agent, episode=10, reward=0.31)   # saves to latest + best
    agent = ckpt.load_best(agent)
"""
import json
import shutil
from pathlib import Path
from typing import Optional


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 10,
        keep_snapshots: bool = False,
    ):
        """
        Args:
            checkpoint_dir  : root directory for all checkpoints
            save_every      : save a periodic snapshot every N episodes
            keep_snapshots  : if True, keep episode_XXXX/ directories;
                              if False, only latest/ and best/ are kept
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_snapshots = keep_snapshots

        self._best_reward: float = -float("inf")
        self._meta_path = self.checkpoint_dir / "meta.json"

        # Resume metadata if it exists
        if self._meta_path.exists():
            with self._meta_path.open() as f:
                meta = json.load(f)
            self._best_reward = meta.get("best_reward", -float("inf"))
            print(
                f"[CheckpointManager] Resumed — best reward so far: {self._best_reward:.4f}"
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        agent,
        episode: int,
        reward: float,
        optimizer_state: Optional[dict] = None,
    ) -> str:
        """
        Persist agent state.

        Always writes to latest/.
        Writes to best/ if reward exceeds previous best.
        Writes to episode_XXXX/ every save_every episodes if keep_snapshots=True.

        Returns the path that was written to.
        """
        latest_dir = self.checkpoint_dir / "latest"
        self._write(agent, latest_dir, episode, reward, optimizer_state)

        if reward > self._best_reward:
            self._best_reward = reward
            best_dir = self.checkpoint_dir / "best"
            self._write(agent, best_dir, episode, reward, optimizer_state)
            print(f"[Checkpoint] New best reward {reward:.4f} → saved to {best_dir}")

        if self.keep_snapshots and (episode % self.save_every == 0):
            snap_dir = self.checkpoint_dir / f"episode_{episode:04d}"
            self._write(agent, snap_dir, episode, reward, optimizer_state)

        self._save_meta(episode, reward)
        return str(latest_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(self, agent) -> tuple:
        """
        Load weights from latest/ into agent.
        Returns (agent, metadata_dict).
        """
        return self._load(agent, self.checkpoint_dir / "latest")

    def load_best(self, agent) -> tuple:
        """
        Load weights from best/ into agent.
        Returns (agent, metadata_dict).
        """
        return self._load(agent, self.checkpoint_dir / "best")

    def has_checkpoint(self) -> bool:
        return (self.checkpoint_dir / "latest" / "meta.json").exists()

    @property
    def best_reward(self) -> float:
        return self._best_reward

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, agent, directory: Path, episode: int, reward: float, optimizer_state):
        directory.mkdir(parents=True, exist_ok=True)
        agent.save(str(directory))

        meta = {
            "episode": episode,
            "reward": reward,
            "best_reward": self._best_reward,
            "has_optimizer_state": optimizer_state is not None,
        }
        with (directory / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

        if optimizer_state is not None:
            import torch
            torch.save(optimizer_state, directory / "optimizer.pt")

    def _load(self, agent, directory: Path) -> tuple:
        if not directory.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {directory}. "
                "Run at least one episode before loading."
            )
        agent.load(str(directory))
        meta_path = directory / "meta.json"
        meta = {}
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)

        opt_path = directory / "optimizer.pt"
        if opt_path.exists():
            import torch
            meta["optimizer_state"] = torch.load(opt_path, map_location="cpu")

        print(
            f"[Checkpoint] Loaded from {directory} "
            f"(episode={meta.get('episode')}, reward={meta.get('reward')})"
        )
        return agent, meta

    def _save_meta(self, episode: int, reward: float):
        with self._meta_path.open("w") as f:
            json.dump(
                {
                    "last_episode": episode,
                    "last_reward": reward,
                    "best_reward": self._best_reward,
                },
                f,
                indent=2,
            )
