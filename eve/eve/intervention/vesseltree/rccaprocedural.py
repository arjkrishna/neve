"""Procedural RCCA-to-siphon vessel tree (per-worker anatomy variation).

Reuses eve's proven aorta + brachiocephalic-trunk CHS generators for the
proximal arch (consistent scale, proven meshing) and swaps the common
carotid for the full RCCA -> RICA -> cavernous-siphon -> terminus chain
(``carotidsiphon.right_carotid_siphon``). The arch sisters (RSA, LCCA,
LSA) are kept so the wire still faces the real fork-commit / wrong-daughter
problem at the arch top. Everything is generated from CENTERLINES and
meshed by the voxel -> marching-cubes pipeline (watertight by
construction — no surface-mesh boolean), so real siphon tortuosity costs
only new CHS control points.

``reset()`` regenerates the geometry every ``episodes_between_change``
episodes (default 10), exactly like ``AorticArchRandom`` — drop a fresh
instance into each worker's env and each worker sees a different RCCA that
rotates every N episodes. Exposes the standard VesselTree interface
(branches / branching_points / centerline_coordinates / insertion /
mesh_path), so env5 / DualDeviceNav / the path-context state machine
consume it exactly like ``FromMesh``.

NOTE: SOFA-restore checkpoints (the "pid" pool) are captured against ONE
fixed mesh and CANNOT be reused here — a regenerated mesh re-tessellates
the collision surface and a restored wire pose is invalid. Use this tree
for the from-z=345 generalization run (Gen-4 recovery training replaces
the restore crutch); keep the fixed FromMesh tree for restore-from-pids.
"""

from typing import Optional, Tuple

import numpy as np

from .vesseltree import VesselTree, Insertion, gym
from .util.branch import (
    BranchWithRadii,
    calc_branching_with_radii,
    scale_branches_xyzd,
    rotate_branches,
)
from .util.insertionfromcenterline import calc_insertion_from_branch_start
from .util.meshing import generate_temp_mesh
from .aorticarcharteries import (
    aorta_generator,
    brachiocephalic_trunk_static,
    left_common_carotid,
    left_subclavian,
    right_subclavian,
)
from .aorticarcharteries.carotidsiphon import right_carotid_siphon, SiphonParams


class RCCAProcedural(VesselTree):
    """Type-I aortic arch whose RCCA extends through the carotid siphon,
    procedurally varied per generation.

    Args:
        siphon_params: sampling ranges for the RCCA->siphon anatomy.
        episodes_between_change: regenerate the vessel every N episodes.
        scale_diameter_mean_sigma: whole-tree diameter scale (on top of the
            per-segment siphon taper) — inter-subject calibre variation.
        rotation_yzx_deg: fixed registration rotation applied to every
            generated vessel (align to SOFA / the C-arm view). Held CONSTANT
            across generations so the obs direction features keep their
            meaning (mesh-invariance rule).
        seed: base RNG seed (per-worker distinctness = distinct seeds).
    """

    def __init__(
        self,
        siphon_params: Optional[SiphonParams] = None,
        episodes_between_change: int = 10,
        scale_diameter_mean_sigma: Tuple[float, float] = (1.0, 0.06),
        rotation_yzx_deg: Optional[Tuple[float, float, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.siphon_params = siphon_params or SiphonParams()
        self.episodes_between_change = int(episodes_between_change)
        self.scale_diameter_mean_sigma = scale_diameter_mean_sigma
        self.rotation_yzx_deg = rotation_yzx_deg
        self._seed = int(seed) if seed is not None else int(
            np.random.default_rng().integers(0, 2**31 - 1)
        )
        self._rng = np.random.default_rng(self._seed)
        self._gen_count = 0

        self._mesh_path = None
        self.visu_mesh_path = None
        self.branches = None
        self.branching_points = None
        self.centerline_coordinates = None
        self.insertion = None

        self._generate()

    # -- VesselTree interface -------------------------------------------
    @property
    def mesh_path(self) -> str:
        # Lazy: voxel->marching-cubes is only paid when SOFA first asks for
        # the mesh (once per generation), not at every reset.
        if self._mesh_path is None:
            self._mesh_path = generate_temp_mesh(self.branches, "rcca_proc", 0.99)
        return self._mesh_path

    @mesh_path.setter
    def mesh_path(self, value):
        self._mesh_path = value

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        # Regenerate every N episodes (episode 0 already generated in ctor).
        if episode_nr > 0 and (episode_nr % self.episodes_between_change == 0):
            self._generate()

    # -- generation ------------------------------------------------------
    def _generate(self) -> None:
        rng = self._rng
        self._gen_count += 1
        n = rng.normal

        aorta_res = 2
        aorta, _ = aorta_generator(aorta_res, rng)

        # Type-I arch: BCT off the aorta, RSA + RCCA off the BCT, LCCA + LSA
        # off the arch (mirrors AorticArch._create_type_I, minus the LCCA/LSA
        # spacing jitter that isn't load-bearing here).
        d_bct = int(np.round(abs(n(36, 5)) / aorta_res))
        bct, bct_chs = brachiocephalic_trunk_static(
            aorta.coordinates[-d_bct], 1, rng
        )
        rsa, _ = right_subclavian(bct.coordinates[-1], bct_chs[-1], 1, rng)

        # THE swap — RCCA continues through the siphon instead of ending at
        # the common carotid. Named "RCCA" so classify_physical_branch tags
        # it as the target daughter.
        rcca, _ = right_carotid_siphon(
            bct.coordinates[-1], resolution=1.0, rng=rng,
            params=self.siphon_params, name="RCCA",
        )

        d_lcca = int(np.round(abs(n(16, 4)) / aorta_res))
        lcca, _ = left_common_carotid(aorta.coordinates[-d_lcca], 1, rng)
        d_lsa = d_lcca + int(np.round(abs(n(28, 3)) / aorta_res))
        lsa, _ = left_subclavian(aorta.coordinates[-d_lsa], 1, rng)

        branches = [aorta, bct, rcca, rsa, lcca, lsa]

        # Whole-tree diameter scale (per-subject calibre) + fixed
        # registration rotation. Diameter-only scale keeps the siphon
        # taper; rotation is constant across generations (obs invariance).
        dscale = float(max(0.75, n(*self.scale_diameter_mean_sigma)))
        if abs(dscale - 1.0) > 1e-6:
            branches = list(scale_branches_xyzd(
                branches, (1.0, 1.0, 1.0, dscale)
            ))
        if self.rotation_yzx_deg is not None:
            branches = list(rotate_branches(branches, self.rotation_yzx_deg))

        # Insertion from the TRANSFORMED aorta (branches[0] after scale/
        # rotate), NOT the local `aorta` binding which stays untransformed —
        # else with rotation_yzx_deg set the insertion lands in a different
        # frame than the (rotated) lumen/mesh (mirrors aorticarch.py's
        # calc_insertion_from_branch_start(branches[0]) after transforms).
        insertion_point, ip_dir = calc_insertion_from_branch_start(branches[0])

        self.branches = tuple(branches)
        self.branching_points = calc_branching_with_radii(branches)
        self.centerline_coordinates = np.concatenate(
            [b.coordinates for b in branches]
        )
        self.insertion = Insertion(insertion_point, ip_dir)
        highs = np.max([b.high for b in branches], axis=0)
        lows = np.min([b.low for b in branches], axis=0)
        self.coordinate_space = gym.spaces.Box(lows, highs)
        self.coordinate_space_episode = self.coordinate_space
        # Delete the prior generation's temp .obj (avoid /tmp exhaustion
        # over a long multi-worker run) then force regen on next access.
        _old = getattr(self, "_mesh_path", None)
        if _old:
            try:
                import os as _os
                _os.remove(_old)
            except OSError:
                pass
        self._mesh_path = None
