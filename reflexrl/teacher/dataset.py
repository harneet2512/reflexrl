"""Teacher label store: (student_obs, pi_T, action) shards on disk.

Every Qwen forward pass is paid for once. Labels are written as compressed
shards so they can be reused by BC, the teacher proxy, and distillation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SHARD_SIZE = 2000


class LabelWriter:
    def __init__(self, out_dir: Path, shard_size: int = SHARD_SIZE):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._n_shards = len(list(self.out_dir.glob("shard_*.npz")))
        self._reset()

    def _reset(self) -> None:
        self._obs, self._probs, self._act, self._ep, self._step = [], [], [], [], []

    def add(self, obs: np.ndarray, probs: np.ndarray, action: int, episode: int,
            step: int) -> None:
        self._obs.append(obs)
        self._probs.append(probs.astype(np.float16))
        self._act.append(action)
        self._ep.append(episode)
        self._step.append(step)
        if len(self._obs) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._obs:
            return
        path = self.out_dir / f"shard_{self._n_shards:04d}.npz"
        np.savez_compressed(
            path, obs=np.stack(self._obs), probs=np.stack(self._probs),
            action=np.asarray(self._act, np.int16), episode=np.asarray(self._ep, np.int32),
            step=np.asarray(self._step, np.int32))
        self._n_shards += 1
        self._reset()

    def close(self) -> None:
        self.flush()


def load_labels(label_dir: Path) -> dict[str, np.ndarray]:
    shards = sorted(Path(label_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no label shards in {label_dir}")
    parts = [dict(np.load(s)) for s in shards]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
