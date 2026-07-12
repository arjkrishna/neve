"""Mesh-invariant observation components (Gen-4 obs redesign).

Three components that remove the mesh-memorization channels from the
observation stack and add the privileged tail for the asymmetric critic:

- ``TipRelativeTracking2D``: wraps a Tracking2D-style observation and
  re-expresses the wire shape relative to the tip at a FIXED mm scale.
  Replaces the mesh-bbox ``NormalizeTracking2DEpisode`` (whose reference
  quantities are exactly the mesh's size/aspect/placement — the dominant
  memorization channel, and anisotropic per-mesh).
- ``TargetTipOffset2D``: clipped tip→target offset. Replaces the absolute
  bbox-normalized target coordinate (a mesh-identity fingerprint; on one
  mesh the 4 daughters form 4 recognizable clusters).
- ``PrivilegedState``: sim-side state the critic may see but a deployed
  policy could not (SOFA node forces/velocities, contact proxy, device
  rotations, catheter tip, ground-truth branch). Appended LAST in the
  ObsDict so it rides inside flat_obs through buffer/PER/caches with no
  plumbing changes; the policy slices it off internally
  (GaussianPolicy.forward keeps ``[..., :n_observations]``).
"""

from typing import Optional
import numpy as np

from .observation import Observation, gym
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_2d


class TipRelativeTracking2D(Observation):
    """Tip-relative wire shape at fixed mm scale.

    Row 0 = per-step tip displacement / ``delta_scale_mm`` (restores the
    rigid-translation velocity information that tip-relative frames lose
    under Memory stacking). Rows 1..N-1 = (p_i - tip) / ``offset_scale_mm``.
    Offsets are hard-bounded by the tracking resolution
    (|p_i - p_0| <= resolution*i), so the fixed scale is exact and
    identical for every mesh — the mesh bbox never enters the observation,
    and the scale is isotropic (no per-mesh angle/curvature distortion).
    Offsets stay in the fixed image axes (the action's rotation semantics
    are defined relative to the C-arm view; heading is supplied by the
    guidance features).
    """

    def __init__(
        self,
        wrapped_obs: Observation,
        delta_scale_mm: float = 4.0,
        offset_scale_mm: float = 135.0,
        name: Optional[str] = None,
    ) -> None:
        self.name = name or wrapped_obs.name
        self.wrapped_obs = wrapped_obs
        self.delta_scale_mm = float(delta_scale_mm)
        self.offset_scale_mm = float(offset_scale_mm)
        self.obs = None
        self._prev_tip: Optional[np.ndarray] = None

    @property
    def space(self) -> gym.spaces.Box:
        shape = self.wrapped_obs.space.shape
        high = np.ones(shape, dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._recompute()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        # Seed prev tip with the reset tip so row 0 reads 0 at episode
        # start (correct for a stationary wire; matches Memory FILL).
        self._prev_tip = np.array(self.wrapped_obs.obs[0], dtype=np.float32)
        self._recompute()

    def _recompute(self) -> None:
        points = np.asarray(self.wrapped_obs.obs, dtype=np.float32)
        tip = points[0]
        out = np.empty_like(points)
        if self._prev_tip is None:
            self._prev_tip = tip.copy()
        out[0] = (tip - self._prev_tip) / self.delta_scale_mm
        out[1:] = (points[1:] - tip) / self.offset_scale_mm
        self._prev_tip = tip.copy()
        self.obs = np.clip(out, -1.0, 1.0)


class TargetTipOffset2D(Observation):
    """Clipped tip→target 2-D offset: ``(target2d - tip2d) / clip_mm``.

    Far targets saturate at ±1 (routing is the guidance block's job); the
    offset only sharpens near-target homing. Supports the grader-mode
    ``target_coord3d`` override with the same semantics as Target2D (env5
    updates ``.target_coord3d`` per grader episode).
    """

    def __init__(
        self,
        intervention: Intervention,
        clip_mm: float = 50.0,
        name: str = "target_offset2d",
        target_coord3d: Optional[np.ndarray] = None,
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.clip_mm = float(clip_mm)
        self.obs = None
        self.target_coord3d = (
            np.array(target_coord3d, dtype=np.float32)
            if target_coord3d is not None
            else None
        )

    @property
    def space(self) -> gym.spaces.Box:
        high = np.ones((2,), dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self) -> None:
        fluoro = self.intervention.fluoroscopy
        tip2d = np.asarray(fluoro.tracking2d[0], dtype=np.float32)
        if self.target_coord3d is not None:
            target2d = np.asarray(
                tracking3d_to_2d(self.target_coord3d), dtype=np.float32
            )
        else:
            target2d = np.asarray(
                self.intervention.target.coordinates2d, dtype=np.float32
            )
        self.obs = np.clip(
            (target2d - tip2d) / self.clip_mm, -1.0, 1.0
        ).astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


class PrivilegedState(Observation):
    """Privileged sim-state tail for the asymmetric critic (24 dims).

    Everything here is either read directly from the SOFA scene or
    mirrored onto the intervention by env5; a deployed system would not
    have it, so ONLY the critics may consume it. Every accessor is
    exception-guarded: any failure yields 0 for that block, never a crash
    (this is observation-support code inside env.step).

    Layout (all clipped to [-1, 1] / [0, 1]):
      0  mean |node velocity| over the K tip-most nodes, /50 mm/s
      1  max  |node velocity| (all nodes), /50
      2  log1p(mean |node force|)/10   (log scale: force units vary)
      3  log1p(max  |node force|)/10
      4  arg-max-force position along the wire, 0=proximal .. 1=tip
      5  mean |position - free_position| (contact impulse proxy), /2 mm
      6  max  |position - free_position|, /10 mm
      7-10   sin/cos of accumulated device rotations (gw, cath)
      11-13  (catheter tip - guidewire tip) 3-D offset, /150 mm
      14-18  physical-branch one-hot: [target daughter, other daughter,
             trunk, bridge/on-path other, off/unknown]
      19  off-branch step counter /50
      20  fold-stall counter /20
      21  slip (commanded-vs-achieved insertion mismatch) /4 mm
      22  cross-track excess beyond radius-aware tolerance, /10 mm
      23  guidewire slack (inserted - tip arclength), /50 mm
    """

    N_DIMS = 24
    _TIP_NODES = 3

    def __init__(
        self,
        intervention: Intervention,
        name: str = "privileged",
        path_context=None,
    ) -> None:
        self.name = name
        self.intervention = intervention
        # ConfigHandler expects self.path_context to match __init__ param.
        # Store None for serialization; actual cache in _path_context
        # (same pattern as LocalGuidance).
        self.path_context = None
        self._path_context = path_context
        self.obs = np.zeros(self.N_DIMS, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        high = np.ones((self.N_DIMS,), dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()

    def step(self) -> None:
        obs = np.zeros(self.N_DIMS, dtype=np.float32)

        # --- SOFA beam state (velocity / force / contact proxy) ---
        try:
            ic = self.intervention.simulation._instruments_combined
            if ic is not None:
                dofs = ic.DOFs
                pos = np.asarray(dofs.position.value)[:, 0:3]
                vel = getattr(dofs, "velocity", None)
                if vel is not None:
                    v = np.linalg.norm(
                        np.asarray(vel.value)[:, 0:3], axis=1
                    )
                    k = min(self._TIP_NODES, len(v))
                    # DOFs are stored proximal->distal reversed for
                    # tracking; tip nodes are at the END of the array
                    # (tracking3d reverses; see sofabeamadapter).
                    obs[0] = np.mean(v[-k:]) / 50.0
                    obs[1] = np.max(v) / 50.0
                force = getattr(dofs, "force", None)
                if force is not None:
                    f = np.linalg.norm(
                        np.asarray(force.value)[:, 0:3], axis=1
                    )
                    if len(f) > 0:
                        obs[2] = np.log1p(float(np.mean(f))) / 10.0
                        obs[3] = np.log1p(float(np.max(f))) / 10.0
                        obs[4] = float(np.argmax(f)) / max(1, len(f) - 1)
                free_pos = getattr(dofs, "free_position", None)
                if free_pos is not None:
                    fp = np.asarray(free_pos.value)[:, 0:3]
                    if fp.shape == pos.shape:
                        d = np.linalg.norm(pos - fp, axis=1)
                        obs[5] = float(np.mean(d)) / 2.0
                        obs[6] = float(np.max(d)) / 10.0
        except Exception:
            pass

        # --- device rotations (torsional windup) ---
        try:
            rot = self.intervention.device_rotations
            for i in range(min(2, len(rot))):
                obs[7 + 2 * i] = np.sin(rot[i])
                obs[8 + 2 * i] = np.cos(rot[i])
        except Exception:
            pass

        # --- catheter tip offset from guidewire tip (3-D, tracking frame) ---
        try:
            trackings = self.intervention.fluoroscopy.device_trackings3d
            if trackings is not None and len(trackings) >= 2:
                gw_tip = np.asarray(trackings[0][0], dtype=np.float32)
                cath = trackings[1]
                if len(cath) > 0:
                    cath_tip = np.asarray(cath[0], dtype=np.float32)
                    obs[11:14] = (cath_tip - gw_tip) / 150.0
        except Exception:
            pass

        # --- ground-truth physical branch one-hot ---
        try:
            from ..util.pathcontext import classify_physical_branch

            short = classify_physical_branch(self.intervention)
            target_short = getattr(
                self.intervention, "_env_target_branch_short", None
            )
            daughters = ("RCCA", "LCCA", "RVA", "LVA")
            if target_short and short == target_short:
                obs[14] = 1.0
            elif short in daughters:
                obs[15] = 1.0
            elif short == "trunk":
                obs[16] = 1.0
            elif short == "bridge":
                obs[17] = 1.0
            else:
                obs[18] = 1.0
        except Exception:
            pass

        # --- env-mirrored counters + slip (env5 pre-step mirrors) ---
        try:
            obs[19] = (
                float(getattr(self.intervention, "_env_off_branch_steps", 0))
                / 50.0
            )
            obs[20] = (
                float(getattr(self.intervention, "_env_fold_stall_count", 0))
                / 20.0
            )
            obs[21] = (
                float(getattr(self.intervention, "_env_slip_mm", 0.0)) / 4.0
            )
        except Exception:
            pass

        # --- cross-track excess + guidewire slack (path context) ---
        try:
            pc = self._path_context
            if pc is not None:
                proj = pc.get_projection()
                tol = pc.get_local_tolerance()
                obs[22] = max(0.0, proj.cross_track_dist - tol) / 10.0
                inserted_gw = self.intervention.device_lengths_inserted[0]
                obs[23] = (float(inserted_gw) - proj.s) / 50.0
        except Exception:
            pass

        self.obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
