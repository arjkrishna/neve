"""Loaded arch mesh with ONLY the RCCA->RICA->siphon varied per worker.

Unlike ``RCCAProcedural`` (which synthesizes the WHOLE arch), this keeps
every non-RCCA branch centerline EXACTLY as loaded from the fixed
DualDeviceNav mesh and perturbs ONLY the RCCA — anchored at the real RCCA
ostium so the ostium point, the bridge junction, and the wire's fixed
start are identical across workers and generations. The distal siphon
bends and calibre vary per worker; the proximal ~15 mm stays pinned.

Because the whole tree is re-meshed from centerlines (voxel ->
marching-cubes), the non-RCCA arch is a re-mesh of the SAME loaded
centerlines (identical lumen — and the wire never navigates there anyway,
since it starts at the RCCA ostium), and the varied RCCA is watertight by
construction (no surface-mesh boolean).

Insertion is FIXED at the RCCA ostium (position + loaded takeoff
direction) — identical for every worker; only the RCCA downstream of it
varies. Regenerates every ``episodes_between_change`` episodes.

NOTE: a per-generation full-tree re-mesh is not free (marching-cubes over
the arch bounding box, ~seconds). Amortized over N minutes-long episodes
it is a few percent overhead; if it dominates, lower the voxel resolution
or restrict meshing to the navigated subtree.
"""

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .vesseltree import VesselTree, Insertion, gym
from .util.branch import (
    BranchWithRadii,
    calc_branching_with_radii,
)
from .util.meshing import generate_temp_mesh


def _perpendicular_basis(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors perpendicular to ``tangent``."""
    t = tangent / max(np.linalg.norm(tangent), 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n1 = np.cross(t, ref)
    n1 = n1 / max(np.linalg.norm(n1), 1e-9)
    n2 = np.cross(t, n1)
    n2 = n2 / max(np.linalg.norm(n2), 1e-9)
    return n1, n2


def perturb_rcca(
    rcca: BranchWithRadii,
    rng: np.random.Generator,
    tortuosity: float = 1.0,
    radius_scale: float = 1.0,
    anchor_mm: float = 15.0,
    ramp_mm: float = 25.0,
    distal_anchor_mm: float = 25.0,
    base_amp_mm: float = 4.0,
    freqs: Sequence[float] = (0.7, 1.3, 2.1),
    name: Optional[str] = None,
) -> BranchWithRadii:
    """Return a perturbed copy of the loaded RCCA centerline.

    BELL-shaped envelope: the displacement is zero over the first
    ``anchor_mm`` (pins the ostium + the (11)/RVA bridge junction) AND over
    the last ``distal_anchor_mm`` (pins the terminus + the distal
    Circle-of-Willis (13)/(24) junction), ramping up/down over ``ramp_mm``
    and full through the MIDDLE — i.e. it varies the cavernous-siphon course
    (the genua, the region of interest) while keeping BOTH ends connected
    to the rest of the loaded tree. Amplitude scales with ``tortuosity``.
    Radii are scaled toward ``radius_scale`` in the mid-vessel (both ends'
    calibre preserved so the junctions stay consistent). The displacement
    is a smooth low-frequency field -> a plausibly tortuous vessel, not a
    kink.
    """
    coords = np.asarray(rcca.coordinates, dtype=np.float64).copy()
    radii = np.asarray(rcca.radii, dtype=np.float64).copy()
    n = len(coords)
    if n < 4:
        return rcca

    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # arclength per point
    total = float(s[-1])

    # Bell envelope: smoothstep up from the ostium anchor, smoothstep down
    # into the terminus anchor. 0 at both ends, 1 through the middle.
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    w_up = _smoothstep((s - anchor_mm) / max(ramp_mm, 1e-6))
    d_from_end = total - s
    w_down = _smoothstep((d_from_end - distal_anchor_mm) / max(ramp_mm, 1e-6))
    w = np.minimum(w_up, w_down)

    # Overall tangent -> a stable perpendicular basis (the displacement
    # stays roughly perpendicular to the vessel so it bends rather than
    # stretches it).
    overall_t = coords[-1] - coords[0]
    n1, n2 = _perpendicular_basis(overall_t)

    u = s / max(total, 1e-6)
    amp = base_amp_mm * float(tortuosity)
    disp1 = np.zeros(n)
    disp2 = np.zeros(n)
    for f in freqs:
        a1 = rng.normal(0.0, 1.0)
        a2 = rng.normal(0.0, 1.0)
        p1 = rng.uniform(0.0, 2.0 * np.pi)
        p2 = rng.uniform(0.0, 2.0 * np.pi)
        disp1 += a1 * np.sin(2.0 * np.pi * f * u + p1)
        disp2 += a2 * np.sin(2.0 * np.pi * f * u + p2)
    # Normalize the summed modes so amplitude is controlled by amp, not by
    # the number of frequencies.
    norm = max(len(freqs) ** 0.5, 1.0)
    offset = (w * amp / norm)[:, None] * (
        disp1[:, None] * n1[None, :] + disp2[:, None] * n2[None, :]
    )
    new_coords = coords + offset

    # Radius: bell-blend toward radius_scale in the mid-vessel; BOTH ends'
    # calibre preserved (same envelope as the displacement) so neither
    # junction's radius-based detection is disturbed.
    new_radii = radii * (1.0 + (float(radius_scale) - 1.0) * w)

    return BranchWithRadii(
        name or rcca.name,
        new_coords.astype(np.float32),
        new_radii.astype(np.float32),
    )


class RCCAVariedFromMesh(VesselTree):
    """Loaded arch centerlines with only the RCCA perturbed per generation.

    Args:
        branch_list: the loaded branch centerlines (from
            eve_bench.dualdevicenav.load_branches) — non-RCCA kept as-is.
        rcca_name: exact name of the RCCA branch in branch_list.
        episodes_between_change: regenerate every N episodes.
        tortuosity_mean_sigma / tortuosity_clip: distal-bend variation.
        radius_scale_mean_sigma: distal calibre variation.
        seed: base RNG seed (distinct per worker for distinct anatomies).
        decimate_factor: passed to generate_temp_mesh.
    """

    def __init__(
        self,
        branch_list: List[BranchWithRadii],
        rcca_name: str = "Centerline curve - RCCA.mrk",
        episodes_between_change: int = 10,
        tortuosity_mean_sigma: Tuple[float, float] = (1.0, 0.3),
        tortuosity_clip: Tuple[float, float] = (0.4, 1.6),
        radius_scale_mean_sigma: Tuple[float, float] = (1.0, 0.07),
        base_amp_mm: float = 4.0,
        seed: Optional[int] = None,
        decimate_factor: float = 0.99,
        visu_mesh_path: Optional[str] = None,
        fork_anchor_mm: float = 3.0,
        rva_amp_mm: float = 3.0,
        rva_proximal_mm: float = 35.0,
        start_point_offset: int = 2,
    ) -> None:
        self._loaded = list(branch_list)
        self._rcca_idx = None
        for i, b in enumerate(self._loaded):
            if b.name == rcca_name or "RCCA" in str(b.name).upper():
                self._rcca_idx = i
                break
        if self._rcca_idx is None:
            raise ValueError(
                f"RCCA branch '{rcca_name}' not in branch_list "
                f"({[b.name for b in self._loaded]})"
            )
        self._loaded_rcca = self._loaded[self._rcca_idx]
        # RVA shares the (11) ostium with RCCA (the bifurcation). Perturbing
        # its PROXIMAL takeoff too varies the fork geometry so the wire cannot
        # memorize a fixed deflection into RCCA. Optional (fall back to fixed
        # RVA if the branch is absent).
        self._rva_idx = None
        for i, b in enumerate(self._loaded):
            if "RVA" in str(b.name).upper():
                self._rva_idx = i
                break
        self._loaded_rva = (
            self._loaded[self._rva_idx] if self._rva_idx is not None else None
        )

        self.episodes_between_change = int(episodes_between_change)
        self.tortuosity_mean_sigma = tortuosity_mean_sigma
        self.tortuosity_clip = tortuosity_clip
        self.radius_scale_mean_sigma = radius_scale_mean_sigma
        self.base_amp_mm = float(base_amp_mm)
        self.decimate_factor = float(decimate_factor)
        # Bifurcation-variation knobs (ConfigHandler-visible floats):
        #   fork_anchor_mm  — RCCA ostium pin length. SMALL (was effectively
        #     15) so the RCCA takeoff/deflection at the fork varies, not just
        #     the mid-siphon. The ostium POINT (s=0) stays pinned regardless.
        #   rva_amp_mm      — proximal-RVA displacement amplitude ("a little").
        #   rva_proximal_mm — length of the RVA takeoff window that varies
        #     (the rest of the RVA, incl. any distal junction, stays fixed).
        self.fork_anchor_mm = float(fork_anchor_mm)
        self.rva_amp_mm = float(rva_amp_mm)
        self.rva_proximal_mm = float(rva_proximal_mm)
        # Insert this many centerline points INTO the (11) bridge (past the
        # junction, where the first point is slightly outside the lumen).
        self.start_point_offset = int(start_point_offset)

        self._seed = int(seed) if seed is not None else int(
            np.random.default_rng().integers(0, 2**31 - 1)
        )
        self._rng = np.random.default_rng(self._seed)
        # Mesh identity (Gen-4 recovery-restore mesh matching). The current
        # mesh is uniquely reproducible from (seed, generation): _generation
        # counts _generate() calls since the RNG was last (re)seeded, and
        # self._rng is used ONLY inside _generate(), so re-seeding with the
        # same seed and replaying `generation` _generate() calls recreates
        # the identical siphon geometry (see regenerate_to_fingerprint).
        # This lets stuck-state SOFA checkpoints be tagged with the mesh
        # they were captured on and restored ONLY into that exact mesh —
        # a checkpoint is mesh-bound SOFA state, so restoring it into a
        # different siphon would be a geometrically invalid teleport.
        self._generation = 0
        # When set (via pin_next), the NEXT reset() regenerates the mesh to
        # this fingerprint instead of following the episode schedule — used
        # by CheckpointRestoreWrapper to make the mesh match the checkpoint
        # it is about to restore.
        self._pending_fingerprint: Optional[str] = None

        # eve ConfigHandler (save_config) reads EVERY __init__ param name via
        # getattr(self, name) — this class lives in the eve.* namespace so its
        # OWN signature is used (confighandler.py:146). Expose the params that
        # are otherwise stored under private names. branch_list is the heavy
        # Branch list: store None (the LocalGuidance path_context convention)
        # so config records a placeholder instead of serializing every
        # centerline; the real data stays in self._loaded.
        self.branch_list = None
        self.rcca_name = rcca_name
        self.seed = self._seed

        # FIXED insertion — a FEW POINTS inside the (11) bridge, the shared
        # parent of RCCA and RVA (both daughters start at the (11) distal
        # fork). Starting inside the bridge (rather than at the RCCA ostium)
        # forces the wire to traverse it and DEFLECT into RCCA, away from RVA,
        # at the fork — the RCCA-vs-RVA discrimination skill, and it re-enables
        # the +1/-1 daughter-commit signal at that fork. NOT the very first
        # (11) point: that sits in the (0)/(11)/LCCA junction and is slightly
        # OUTSIDE the (11) lumen — inserting there put the wire outside the
        # vessel (every episode immediately hit vessel-end). start_point_offset
        # (default 2) steps a couple points in, clear of the junction. The
        # (11) bridge is NOT perturbed, so the start is identical across
        # workers/generations. Falls back to the RCCA ostium if (11) is absent.
        ostium = np.asarray(self._loaded_rcca.coordinates[0], dtype=np.float64)
        bridge = None
        for _b in self._loaded:
            if "(11)" in str(getattr(_b, "name", "")):
                bridge = _b
                break
        if bridge is not None and len(bridge.coordinates) >= 3:
            bc = np.asarray(bridge.coordinates, dtype=np.float64)
            # Orient index from ENTRY (far from ostium) -> FORK (at ostium).
            if np.linalg.norm(bc[0] - ostium) < np.linalg.norm(bc[-1] - ostium):
                bc = bc[::-1]
            k = int(min(max(1, self.start_point_offset), len(bc) - 2))
            entry = bc[k]
            nxt = bc[k + 1]  # next point toward the fork
            ip_dir = nxt - entry
            ip_dir = ip_dir / max(np.linalg.norm(ip_dir), 1e-9)
            self._insertion = Insertion(entry, ip_dir)
        else:
            second = np.asarray(self._loaded_rcca.coordinates[1], dtype=np.float64)
            ip_dir = second - ostium
            ip_dir = ip_dir / max(np.linalg.norm(ip_dir), 1e-9)
            self._insertion = Insertion(ostium, ip_dir)

        self._mesh_path = None
        self.visu_mesh_path = visu_mesh_path
        self.branches = None
        self.branching_points = None
        self.centerline_coordinates = None
        self.insertion = self._insertion
        self._generate()

    # -- VesselTree interface -------------------------------------------
    @property
    def mesh_path(self) -> str:
        if self._mesh_path is None:
            self._mesh_path = generate_temp_mesh(
                self.branches, "rcca_varied", self.decimate_factor
            )
        return self._mesh_path

    @mesh_path.setter
    def mesh_path(self, value):
        self._mesh_path = value

    # -- mesh identity (recovery-restore matching) ----------------------
    @property
    def mesh_fingerprint(self) -> str:
        """Stable id of the CURRENT siphon geometry: ``s{seed}g{gen}``.
        Reproducible via regenerate_to_fingerprint on any worker."""
        return f"s{self._seed}g{self._generation}"

    @staticmethod
    def parse_fingerprint(fingerprint: str) -> Optional[Tuple[int, int]]:
        """(seed, generation) from an ``s{seed}g{gen}`` string, or None if
        it is not a RCCAVariedFromMesh fingerprint (e.g. a fixed-mesh
        checkpoint's absent/``fixed`` tag)."""
        if not fingerprint:
            return None
        m = re.fullmatch(r"s(-?\d+)g(\d+)", str(fingerprint))
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def regenerate_to_fingerprint(self, fingerprint: str) -> None:
        """Rebuild the exact mesh identified by ``fingerprint`` by re-seeding
        and replaying generations. Deterministic because self._rng feeds
        ONLY _generate(). Raises on a fingerprint this class cannot parse."""
        parsed = self.parse_fingerprint(fingerprint)
        if parsed is None:
            raise ValueError(
                f"cannot regenerate to non-RCCAVaried fingerprint "
                f"{fingerprint!r}"
            )
        seed, generation = parsed
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._generation = 0
        for _ in range(int(generation)):
            self._generate()

    def pin_next(self, fingerprint: str) -> None:
        """Ask the NEXT reset() to regenerate to ``fingerprint`` (overriding
        the episode schedule). No-op-safe: None clears the pin."""
        self._pending_fingerprint = fingerprint

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        # A pin (from CheckpointRestoreWrapper) is authoritative: match the
        # mesh to the checkpoint about to be restored, ignoring both the
        # per-episode seed and the regeneration schedule for this episode.
        if self._pending_fingerprint is not None:
            fp = self._pending_fingerprint
            self._pending_fingerprint = None
            self.regenerate_to_fingerprint(fp)
            return
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
            # A re-seed restarts the generation stream, so the fingerprint
            # stays a faithful (seed, gens-since-seed) identity.
            self._generation = 0
        if episode_nr > 0 and (episode_nr % self.episodes_between_change == 0):
            self._generate()

    # -- generation ------------------------------------------------------
    def _generate(self) -> None:
        rng = self._rng
        tort = float(np.clip(
            rng.normal(*self.tortuosity_mean_sigma), *self.tortuosity_clip
        ))
        rscale = float(max(0.75, rng.normal(*self.radius_scale_mean_sigma)))
        varied_rcca = perturb_rcca(
            self._loaded_rcca, rng,
            tortuosity=tort, radius_scale=rscale,
            anchor_mm=self.fork_anchor_mm,   # small ostium pin -> deflection varies
            ramp_mm=15.0,
            base_amp_mm=self.base_amp_mm,
            name=self._loaded_rcca.name,
        )
        branches = list(self._loaded)
        branches[self._rcca_idx] = varied_rcca

        # Vary the PROXIMAL RVA takeoff a little so the bifurcation deflection
        # is not memorizable. Reuse the bell envelope but pin everything past
        # rva_proximal_mm (distal RVA + any distal junction unchanged); the
        # shared ostium point (s=0) stays pinned, so RCCA[0] == RVA[0] and the
        # fork remains connected to (11).
        if self._rva_idx is not None and self._loaded_rva is not None:
            _rc = np.asarray(self._loaded_rva.coordinates, dtype=np.float64)
            rva_len = float(np.sum(np.linalg.norm(np.diff(_rc, axis=0), axis=1)))
            varied_rva = perturb_rcca(
                self._loaded_rva, rng,
                tortuosity=1.0, radius_scale=1.0,
                anchor_mm=3.0, ramp_mm=12.0,
                distal_anchor_mm=max(5.0, rva_len - self.rva_proximal_mm),
                base_amp_mm=self.rva_amp_mm,
                name=self._loaded_rva.name,
            )
            branches[self._rva_idx] = varied_rva

        self.branches = tuple(branches)
        self.branching_points = calc_branching_with_radii(branches)
        self.centerline_coordinates = np.concatenate(
            [b.coordinates for b in branches]
        )
        # Insertion stays FIXED at the loaded ostium (never re-derived).
        self.insertion = self._insertion
        highs = np.max([b.high for b in branches], axis=0)
        lows = np.min([b.low for b in branches], axis=0)
        self.coordinate_space = gym.spaces.Box(lows, highs)
        self.coordinate_space_episode = self.coordinate_space
        # Delete the PRIOR generation's temp .obj before dropping the ref —
        # else every regen leaks one file into gettempdir(), filling /tmp
        # over a long 16-worker run.
        _old = getattr(self, "_mesh_path", None)
        if _old:
            try:
                import os as _os
                _os.remove(_old)
            except OSError:
                pass
        self._mesh_path = None  # force re-mesh on next mesh_path access
        # This mesh is now the (generation)-th since the last (re)seed.
        self._generation += 1
