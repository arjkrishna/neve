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
        self._pattern = pattern
        self._rng_seed = rng_seed
        self._rng = None  # lazy, per-process seeded on first use
        self._validated = False

        # Plan v9 Change 8 — lazy scan. The pool may be EMPTY at
        # construction time when caches + checkpoints are generated in
        # the SAME run (heuristic seeding fills the pool before online
        # explore needs it). So do NOT raise on an empty/absent dir;
        # rescan lazily at reset time. Only validate the format once a
        # file actually appears.
        self.checkpoint_files = []
        self._rescan()

    def _rescan(self) -> int:
        """(Re)glob the checkpoint dir. Returns the file count. Validates
        the npz format once, the first time a file appears."""
        self.checkpoint_files = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, self._pattern))
        )
        if self.checkpoint_files and not self._validated:
            sample = np.load(self.checkpoint_files[0])
            missing = [k for k in _REQUIRED_KEYS if k not in sample.files]
            if missing:
                raise ValueError(
                    f"Checkpoint {self.checkpoint_files[0]} missing keys: {missing}"
                )
            self._validated = True
        return len(self.checkpoint_files)

    def _pick_index(self, seed: Optional[int],
                    explicit_id: Optional[int] = None) -> int:
        # Plan v3: schedule may pass `options["checkpoint_id"]` to pick
        # a SPECIFIC checkpoint deterministically (factorial grid testing).
        # Falls back to seed-based random pick otherwise.
        if explicit_id is not None:
            return int(explicit_id) % len(self.checkpoint_files)
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
        # Plan v9 Change 8 — heuristic-mode bypass. Heuristic seeding
        # episodes are supposed to run from z=345 (the original
        # insertion point) to produce demos with full-trunk action
        # sequences. If the trainer happens to invoke a live
        # heuristic_seed() pass after wrapping env_train with this
        # restorer, we'd otherwise SOFA-restore the heuristic too, which
        # would defeat the purpose. Skip injection in that case.
        if options.get("heuristic_mode", False):
            return self.env.reset(seed=seed, options=options)
        # Only inject if caller didn't already supply a checkpoint.
        if "restore_checkpoint" not in options:
            # Plan v9 Change 8 — pool may have been filled by heuristic
            # seeding since construction; rescan if we still have none.
            if not self.checkpoint_files:
                self._rescan()
            if self.checkpoint_files:
                explicit_id = options.get("checkpoint_id")
                idx = self._pick_index(seed, explicit_id=explicit_id)
                path = self.checkpoint_files[idx]
                with np.load(path) as data:
                    options["restore_checkpoint"] = {k: np.array(data[k]) for k in data.files}
                options["_restore_checkpoint_file"] = os.path.basename(path)
                options["_restore_checkpoint_idx"] = int(idx)
            # else: pool still empty -> fall through; episode starts at the
            # normal insertion point (z=345). Defensive: never crash.
        return self.env.reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # EveRLObject interop. agent.save_checkpoint() calls
    # self.env_eval.get_config_dict() (eve_rl/agent/agent.py:327). gym.Wrapper
    # does NOT forward custom EveRLObject methods, so without this pass-through
    # the offline runner crashes on the first eval that triggers a checkpoint
    # save. We delegate to the wrapped env (the bare BenchEnv5, which IS an
    # EveRLObject) and merge in this wrapper's metadata so post-hoc analysis
    # can tell the env_eval used a checkpoint-restore pool.
    # ------------------------------------------------------------------
    def get_config_dict(self):
        """Delegate to wrapped env, augmenting with wrapper metadata.

        Returns the inner env's config dict (or ``None`` if the inner env
        doesn't implement get_config_dict) merged with this wrapper's
        identifying fields: ``checkpoint_dir``, ``rng_seed``, and the
        current number of files in the restore pool.
        """
        inner = (
            self.env.get_config_dict()
            if hasattr(self.env, "get_config_dict")
            else None
        )
        meta = {
            "_wrapper": "CheckpointRestoreWrapper",
            "_checkpoint_dir": self.checkpoint_dir,
            "_pattern": self._pattern,
            "_rng_seed": self._rng_seed,
            "_n_checkpoint_files": len(self.checkpoint_files),
        }
        if isinstance(inner, dict):
            merged = dict(inner)
            merged.update(meta)
            return merged
        # Inner env had no get_config_dict — still return a serializable
        # dict so callers (save_checkpoint) don't get None and skip env
        # metadata persistence.
        return meta

    # Forward everything else untouched (gym.Wrapper does this by default).
