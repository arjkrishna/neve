"""DualDeviceNav with the LOADED arch fixed and ONLY the RCCA varied.

Loads the exact DualDeviceNav branch centerlines, keeps every non-RCCA
branch as-is, and perturbs only the RCCA->RICA->siphon per worker
(``eve.intervention.vesseltree.RCCAVariedFromMesh``). The wire's start is
FIXED at the real RCCA ostium (identical for every worker); only the RCCA
downstream of it varies, regenerating every ``episodes_between_change``
episodes. The whole tree is re-meshed from centerlines (watertight, no
surface boolean) and stays in the SAME vessel-CS frame as ``DualDeviceNav``
— so the observation semantics match and a fixed-mesh RCCA policy can warm-
start this run.

This isolates exactly the siphon-navigation problem: no arch/trunk/fork-
commit (the wire starts at the RCCA ostium), just varied siphon anatomy.

Per worker: distinct ``seed`` -> distinct RCCA anatomy sequence:
    DualDeviceNavRCCAVaried(seed=base_seed + worker_id,
                            episodes_between_change=10)
"""

import os
from typing import Optional, Tuple

import eve
from eve.intervention.vesseltree import RCCAVariedFromMesh

# Reuse DualDeviceNav's proven DATA_DIR + loader (resolves the centerline
# data correctly regardless of which install path eve_bench is imported
# from — recomputing it here would be fragile across the dual mount).
from .dualdevicenav import load_branches, DATA_DIR

_RCCA_NAME = "Centerline curve - RCCA.mrk"


class DualDeviceNavRCCAVaried(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        seed: Optional[int] = None,
        episodes_between_change: int = 10,
        tortuosity_mean_sigma: Tuple[float, float] = (1.0, 0.3),
        tortuosity_clip: Tuple[float, float] = (0.4, 1.6),
        radius_scale_mean_sigma: Tuple[float, float] = (1.0, 0.07),
        base_amp_mm: float = 4.0,
        image_rot_zx: Optional[Tuple[float, float]] = None,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
        target_min_arclength_mm: float = 40.0,
    ) -> None:
        centerline_folder = os.path.join(DATA_DIR, "Centrelines_comb")
        branches = load_branches(centerline_folder)  # same (y,-z,-x) frame

        vessel_tree = RCCAVariedFromMesh(
            branch_list=branches,
            rcca_name=_RCCA_NAME,
            episodes_between_change=episodes_between_change,
            tortuosity_mean_sigma=tortuosity_mean_sigma,
            tortuosity_clip=tortuosity_clip,
            radius_scale_mean_sigma=radius_scale_mean_sigma,
            base_amp_mm=base_amp_mm,
            seed=seed,
        )

        # Devices, sim, view identical to DualDeviceNav (frame-consistent).
        device1 = eve.intervention.device.JShaped(
            name="mic_guide", length=900, velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5, tip_outer_diameter=0.36,
            straight_outer_diameter=0.36, tip_inner_diameter=0,
            straight_inner_diameter=0.36, mass_density_tip=0.000005,
            mass_density_straight=0.000005, young_modulus_tip=1e3,
            young_modulus_straight=1e3, beams_per_mm_straight=0.6,
        )
        device2 = eve.intervention.device.JShaped(
            name="mic_cath", length=900, velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5, tip_outer_diameter=0.6,
            straight_outer_diameter=0.7, tip_inner_diameter=0.57,
            straight_inner_diameter=0.57, color=(1.0, 0.0, 0.0),
            mass_density_tip=0.000005, mass_density_straight=0.000005,
            young_modulus_tip=1e3, young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )
        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)
        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=list(image_rot_zx) if image_rot_zx else [20, 5],
        )
        # Skip near-ostium RCCA targets: with the wire now starting at the
        # (11) bridge entry, a target < target_min_arclength_mm into the RCCA
        # would be a trivial "deflect and stop" — exclude those so every
        # target sits in the siphon and the wire must actually navigate it.
        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=[_RCCA_NAME],
            min_arclength_from_start=target_min_arclength_mm,
        )

        # Symmetric velocity limits (review 2.4) — omit velocity_limit_low.
        super().__init__(
            vessel_tree,
            [device1, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )
