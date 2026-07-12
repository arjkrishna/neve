"""DualDeviceNav with a PROCEDURAL, per-worker-varied RCCA->siphon anatomy.

Same guidewire + catheter, SOFA backend, fluoroscopy, and symmetric action
space as ``DualDeviceNav`` — but the vessel tree is
``eve.intervention.vesseltree.RCCAProcedural`` (aorta + BCT + RCCA-through-
siphon + arch sisters, generated from CHS centerlines, re-randomized every
``episodes_between_change`` episodes). The wire inserts at the aortic root
(full-trunk "Challenge-1" task) — there is NO ``insertion_z`` and NO SOFA
restore, because a regenerating mesh invalidates restore checkpoints (use
the fixed-mesh ``DualDeviceNav`` for restore-from-pids runs). Gen-4
recovery training replaces the restore crutch.

Give each worker its OWN instance with a distinct ``seed`` so the 16
workers see 16 different RCCA anatomies, each rotating every N episodes:
    DualDeviceNavProcedural(seed=base_seed + worker_id,
                            episodes_between_change=10)
"""

from typing import Optional, Tuple

import eve
from eve.intervention.vesseltree import RCCAProcedural
from eve.intervention.vesseltree.aorticarcharteries import SiphonParams


class DualDeviceNavProcedural(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        seed: Optional[int] = None,
        episodes_between_change: int = 10,
        siphon_params: Optional[SiphonParams] = None,
        scale_diameter_mean_sigma: Tuple[float, float] = (1.0, 0.06),
        rotation_yzx_deg: Optional[Tuple[float, float, float]] = None,
        image_rot_zx: Optional[Tuple[float, float]] = None,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        vessel_tree = RCCAProcedural(
            siphon_params=siphon_params,
            episodes_between_change=episodes_between_change,
            scale_diameter_mean_sigma=scale_diameter_mean_sigma,
            rotation_yzx_deg=rotation_yzx_deg,
            seed=seed,
        )

        # Same devices as DualDeviceNav (frame-independent properties).
        device1 = eve.intervention.device.JShaped(
            name="mic_guide",
            length=900,
            velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.36,
            straight_outer_diameter=0.36,
            tip_inner_diameter=0,
            straight_inner_diameter=0.36,
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )
        device2 = eve.intervention.device.JShaped(
            name="mic_cath",
            length=900,
            velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.6,
            straight_outer_diameter=0.7,
            tip_inner_diameter=0.57,
            straight_inner_diameter=0.57,
            color=(1.0, 0.0, 0.0),
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

        # image_rot_zx held CONSTANT across generations so the obs 2D
        # direction features keep their meaning (mesh-invariance rule).
        # Default matches DualDeviceNav so obs semantics align across the
        # fixed-mesh and procedural runs.
        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=list(image_rot_zx) if image_rot_zx else [20, 5],
        )

        # Target = the RCCA daughter (named "RCCA" by RCCAProcedural). The
        # siphon depth is sampled per episode along the whole RCCA
        # centerline, so the target lands anywhere from proximal CCA to the
        # cavernous terminus — exactly the depth-generalization we want.
        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=["RCCA"],
        )

        # Symmetric velocity limits (review 2.4) — omit velocity_limit_low
        # so MonoPlaneStatic defaults to -velocity_limits.
        super().__init__(
            vessel_tree,
            [device1, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )
