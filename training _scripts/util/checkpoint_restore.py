"""Per-episode SOFA-state restore wrapper for training.

Wraps a BenchEnv so that every `env.reset(...)` picks a random `.npz` from
`checkpoint_dir` and injects it into `options["restore_checkpoint"]`, which
`MonoPlaneStatic.reset()` then forwards to `SofaBeamAdapter.reset(checkpoint=...)`.

Intended for use with checkpoints produced by
`collect_sofa_checkpoints.py` (each `.npz` contains xtip, rotation_instrument,
index_first_node, dof_positions, tracking3d).

Usage:
    from util.checkpoint_restore import CheckpointRestoreWrapper
    env_train = CheckpointRestoreWrapper(
        env_train, checkpoint_dir="/opt/eve_training/results/sofa_checkpoints/selected",
        rng_seed=0,
    )
"""

import glob
import os
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np


_REQUIRED_KEYS = ("xtip", "rotation_instrument", "index_first_node", "dof_positions")


class CheckpointRestoreWrapper(gym.Wrapper):
    """Random-checkpoint injector for per-episode SOFA state restore.

    Checkpoint selection is designed to be diverse across *parallel workers*:
    the wrapper is deepcopied into each worker process, so a shared seed
    produces the same sequence in every worker (the bug seen in the first
    test run of 2026-04-19 where every worker picked the same `.npz` per
    episode). This class works around that by choosing the checkpoint index
    via one of two mechanisms, in priority order:

    1. If the caller passes ``seed=K`` to ``reset()`` (as ``heuristic_seed``
       does per episode via ``episode_schedule``), derive a fresh
       ``np.random.default_rng(K)`` and use it — gives a unique pick per
       (worker, episode) because the schedule hands out unique seeds.
    2. Else, fall back to a lazily-initialised per-process RNG seeded with
       ``(rng_seed or 0) + os.getpid()`` so workers diverge even after
       deepcopy, and keep drawing from that RNG across episodes.
    """

    def __init__(
        self,
        env,
        checkpoint_dir: str,
        rng_seed: Optional[int] = None,
        pattern: str = "*.npz",
    ):
        super().__init__(env)
        self.checkpoint_dir = checkpoint_dir
        self._rng_seed = rng_seed
        self._rng = None  # lazy, per-process seeded on first use

        self.checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, pattern))
        )
        if not self.checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoints matching {pattern} found in {checkpoint_dir}"
            )

        # Sanity check the first one — fail fast if the format is wrong.
        sample = np.load(self.checkpoint_files[0])
        missing = [k for k in _REQUIRED_KEYS if k not in sample.files]
        if missing:
            raise ValueError(
                f"Checkpoint {self.checkpoint_files[0]} missing keys: {missing}"
            )

    def _pick_index(self, seed: Optional[int]) -> int:
        if seed is not None:
            # Per-episode, per-worker reproducible: unique seeds from the
            # schedule produce unique checkpoint picks.
            return int(
                np.random.default_rng(seed).integers(0, len(self.checkpoint_files))
            )
        if self._rng is None:
            base = self._rng_seed if self._rng_seed is not None else 0
            self._rng = np.random.default_rng(base + os.getpid())
        return int(self._rng.integers(0, len(self.checkpoint_files)))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = dict(options) if options else {}
        # Only inject if caller didn't already supply a checkpoint.
        if "restore_checkpoint" not in options:
            idx = self._pick_index(seed)
            path = self.checkpoint_files[idx]
            with np.load(path) as data:
                options["restore_checkpoint"] = {k: np.array(data[k]) for k in data.files}
            options["_restore_checkpoint_file"] = os.path.basename(path)
        return self.env.reset(seed=seed, options=options)

    # Forward everything else untouched (gym.Wrapper does this by default).
