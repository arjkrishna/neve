"""A SET of fixed patient anatomies, each with its own baked mesh.

This is the fixed-mesh approach the shipped patient uses (``FromMesh``: load an
.obj, set mesh_path, done), extended to hold more than one and switch between
them on the episode schedule. It is deliberately NOT built on
``RCCAVariedFromMesh``: that class exists to SYNTHESISE a vessel per
generation, so its mesh is a runtime artifact and its identity is an RNG
replay. These anatomies are fixed, so their meshes are baked to disk by
``topbrain_tools/bake_meshes.py`` and simply loaded, and their identity is
their name.

That distinction buys three things:

  portable   an anatomy folder is self-contained: centerlines plus the .obj.
             Copy it to another machine and the geometry is byte-identical,
             instead of being re-derived by whatever scikit-image happens to
             be installed there.
  cheap      switching anatomy is a path swap, not seconds of marching cubes.
  exact      the checkpoint fingerprint IS the anatomy, so restoring stuck
             SOFA state means looking the anatomy up by name rather than
             re-seeding an RNG and replaying N generations and trusting that
             it lands on the same geometry.

Every anatomy shares the host tree up to the graft at 130 mm, including branch
(11), so the insertion point and the RCCA/RVA fork are identical across the
set and the observation frame matches ``DualDeviceNav``.

Fingerprints are stripped to ``[A-Za-z0-9]`` because env5 embeds them in
checkpoint FILENAMES and checkpoint_restore matches them with
``_mesh-([A-Za-z0-9]+)_``; anything else silently fails to match.
"""

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vesseltree import VesselTree, Insertion, gym
from .util.branch import BranchWithRadii, calc_branching_with_radii


def fingerprint_of(name: str) -> str:
    """Alphanumeric id for an anatomy, safe to embed in a filename."""
    return re.sub(r"[^A-Za-z0-9]", "", str(name))


class TopBrainAnatomySet(VesselTree):
    """Switch between pre-built anatomies, each with a mesh already on disk.

    Args:
        branch_lists: one loaded branch list per anatomy. Loading is the
            caller's job so this module stays independent of eve_bench.
        mesh_paths: the baked collision .obj per anatomy, same order.
        anatomy_names: labels, same order. These become the fingerprints.
        episodes_between_change: switch anatomy every N episodes.
        seed: base RNG seed; distinct per worker for distinct orderings.
        visu_mesh_paths: optional visual meshes, same order.
        start_point_offset: how many points into branch (11) to start.
    """

    def __init__(
        self,
        branch_lists: Sequence[List[BranchWithRadii]],
        mesh_paths: Sequence[str],
        anatomy_names: Optional[Sequence[str]] = None,
        episodes_between_change: int = 10,
        seed: Optional[int] = None,
        visu_mesh_paths: Optional[Sequence[str]] = None,
        start_point_offset: int = 2,
        rcca_name: str = "Centerline curve - RCCA.mrk",
    ) -> None:
        if not branch_lists:
            raise ValueError("TopBrainAnatomySet needs at least one anatomy")
        if len(mesh_paths) != len(branch_lists):
            raise ValueError("mesh_paths must match branch_lists in length")
        for p in mesh_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "no baked mesh at %s (run topbrain_tools/bake_meshes.py)" % p
                )
        self._sets = [list(bl) for bl in branch_lists]
        self._meshes = list(mesh_paths)
        self._visu = list(visu_mesh_paths) if visu_mesh_paths else [None] * len(self._sets)
        self._names = list(anatomy_names) if anatomy_names else [
            "anatomy%02d" % i for i in range(len(self._sets))
        ]
        if len(self._names) != len(self._sets):
            raise ValueError("anatomy_names must match branch_lists in length")
        self._fps = [fingerprint_of(n) for n in self._names]
        if len(set(self._fps)) != len(self._fps):
            raise ValueError(
                "anatomy names collide once stripped to alphanumerics: %s"
                % self._fps
            )

        self.episodes_between_change = int(episodes_between_change)
        self.start_point_offset = int(start_point_offset)
        self.rcca_name = rcca_name
        self._seed = int(seed) if seed is not None else int(
            np.random.default_rng().integers(0, 2 ** 31 - 1)
        )
        self._generation = 0
        self._current = 0
        self._pending_fingerprint: Optional[str] = None
        self._cache: Dict[int, Dict[str, Any]] = {}

        # ConfigHandler reads every __init__ param name off the instance; the
        # branch lists are heavy, so record a placeholder as branch_list does.
        self.branch_lists = None
        self.mesh_paths = list(self._meshes)
        self.anatomy_names = list(self._names)
        self.visu_mesh_paths = list(self._visu)
        self.seed = self._seed

        self._insertion = self._derive_insertion(self._sets[0])
        self.insertion = self._insertion
        self._select(self._index_for(0))
        self._generation = 1

    # -- insertion -------------------------------------------------------
    def _derive_insertion(self, branches: List[BranchWithRadii]) -> Insertion:
        """A few points into the (11) bridge, the shared RCCA/RVA parent.

        Starting inside the bridge rather than at the RCCA ostium forces the
        wire to traverse it and deflect INTO the RCCA away from the RVA, which
        is the fork-discrimination skill. Not the very first (11) point: that
        sits in the junction and is slightly outside the lumen. Identical for
        every anatomy, since none of them touch this part of the tree.
        """
        rcca = None
        for b in branches:
            if b.name == self.rcca_name or "RCCA" in str(b.name).upper():
                rcca = b
                break
        if rcca is None:
            raise ValueError("no RCCA branch in the anatomy")
        ostium = np.asarray(rcca.coordinates[0], dtype=np.float64)

        bridge = None
        for b in branches:
            if "(11)" in str(getattr(b, "name", "")):
                bridge = np.asarray(b.coordinates, dtype=np.float64)
                break
        if bridge is not None and len(bridge) >= 3:
            # orient from entry (far from the ostium) toward the fork
            if np.linalg.norm(bridge[0] - ostium) < np.linalg.norm(bridge[-1] - ostium):
                bridge = bridge[::-1]
            k = int(min(max(1, self.start_point_offset), len(bridge) - 2))
            entry, nxt = bridge[k], bridge[k + 1]
        else:
            entry, nxt = ostium, np.asarray(rcca.coordinates[1], dtype=np.float64)
        direction = nxt - entry
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        return Insertion(entry, direction)

    # -- selection -------------------------------------------------------
    def _index_for(self, generation: int) -> int:
        """Which anatomy generation g uses: a pure function of (seed, g).

        Permuted rather than iid so a run covers the set evenly, and stateless
        so re-seeding cannot desynchronise it.
        """
        n = len(self._sets)
        rng = np.random.default_rng([self._seed & 0x7FFFFFFF, generation // n])
        return int(rng.permutation(n)[generation % n])

    def _select(self, index: int) -> None:
        self._current = int(index)
        if index not in self._cache:
            branches = self._sets[index]
            highs = np.max([b.high for b in branches], axis=0)
            lows = np.min([b.low for b in branches], axis=0)
            self._cache[index] = {
                "branches": tuple(branches),
                "branching_points": calc_branching_with_radii(branches),
                "centerline_coordinates": np.concatenate(
                    [b.coordinates for b in branches]
                ),
                "space": gym.spaces.Box(lows, highs),
            }
        c = self._cache[index]
        self.branches = c["branches"]
        self.branching_points = c["branching_points"]
        self.centerline_coordinates = c["centerline_coordinates"]
        self.coordinate_space = c["space"]
        self.coordinate_space_episode = c["space"]
        self.mesh_path = self._meshes[index]
        self.visu_mesh_path = self._visu[index]
        # The insertion is in the shared part of the tree and never re-derived.
        self.insertion = self._insertion

    # -- identity (recovery-restore matching) ----------------------------
    @property
    def anatomy_count(self) -> int:
        return len(self._sets)

    @property
    def current_anatomy(self) -> str:
        return self._names[self._current]

    @property
    def mesh_fingerprint(self) -> str:
        """The anatomy IS the identity: no seed or generation involved."""
        return self._fps[self._current]

    @staticmethod
    def parse_fingerprint(fingerprint: str) -> Optional[str]:
        if not fingerprint or fingerprint == "fixed":
            return None
        return str(fingerprint)

    def regenerate_to_fingerprint(self, fingerprint: str) -> None:
        """Select the anatomy a checkpoint was captured on. Exact, not replayed."""
        fp = self.parse_fingerprint(fingerprint)
        if fp is None or fp not in self._fps:
            raise ValueError(
                "fingerprint %r is not one of this set's anatomies" % fingerprint
            )
        self._select(self._fps.index(fp))

    def pin_next(self, fingerprint: str) -> None:
        """Ask the NEXT reset() to select ``fingerprint``, overriding the
        schedule, so the mesh matches the checkpoint about to be restored."""
        self._pending_fingerprint = fingerprint

    # -- VesselTree interface --------------------------------------------
    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        # A pin is authoritative: a stuck checkpoint is mesh-bound SOFA state,
        # so the anatomy must match it, schedule and seed notwithstanding.
        if self._pending_fingerprint is not None:
            fp = self._pending_fingerprint
            self._pending_fingerprint = None
            self.regenerate_to_fingerprint(fp)
            return
        if seed is not None:
            self._seed = int(seed)
            self._generation = 0
        if episode_nr > 0 and (episode_nr % self.episodes_between_change == 0):
            self._select(self._index_for(self._generation))
            self._generation += 1
