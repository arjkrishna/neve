from typing import Optional
import numpy as np

from .observation import Observation, gym
from ..intervention import Intervention
from ..interimtarget import InterimTarget
from ..util.coordtransform import tracking3d_to_2d


class Target2D(Observation):
    def __init__(
        self,
        intervention: Intervention,
        interim_target: Optional[InterimTarget] = None,
        name: str = "target2d",
        target_coord3d: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            target_coord3d: Plan v12 Stage 1 — multi-target heatup. A
                tracking3d-frame coord (as produced by CenterlineRandom).
                When provided, THIS observation instance projects this
                coord to 2-D via `tracking3d_to_2d` (drops y axis, same
                projection `CenterlineRandom.coordinates2d` uses, see
                `eve/util/coordtransform.py:65`) and stores it in obs.
                The 4 virtual envs in `MultiTargetEnv5` each construct a
                standalone `Target2D(intervention, ..., target_coord3d=
                daughter_k_centerline_random.coordinates3d)` so their
                `obs[40:42]` stored in each per-target Episode is
                target-coherent. Default `None` → falls back to shared
                `intervention.target.coordinates2d` (single-target
                behavior unchanged).
        """
        self.name = name
        self.intervention = intervention
        self.interim_target = interim_target
        self.obs = None
        # Plan v12 — own per-instance target 3-D coord; lazily projected
        # to 2-D in step() so step-time refreshes match new physics state.
        self.target_coord3d = (
            np.array(target_coord3d, dtype=np.float32)
            if target_coord3d is not None
            else None
        )

    @property
    def space(self) -> gym.spaces.Box:
        return self.intervention.fluoroscopy.tracking2d_space

    def step(self) -> None:
        if (
            self.interim_target is not None
            and self.interim_target.coordinates2d is not None
        ):
            self.obs = np.array(self.interim_target.coordinates2d, dtype=np.float32)
        elif self.target_coord3d is not None:
            # Plan v12 — own target_coord3d. Project via the same function
            # CenterlineRandom.coordinates2d uses (drops y-axis only).
            self.obs = np.array(
                tracking3d_to_2d(self.target_coord3d), dtype=np.float32
            )
        else:
            self.obs = np.array(
                self.intervention.target.coordinates2d, dtype=np.float32
            )

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
