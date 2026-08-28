"""DualDeviceNav over REAL patient siphons grafted onto the fixed arch.

Drop-in counterpart to ``DualDeviceNavRCCAVaried``: identical devices,
simulation, fluoroscopy, insertion and target semantics, so a policy trained
under the procedural variation can warm-start here and the observation
semantics match. The only difference is where the siphon comes from. Instead
of perturbing one loaded RCCA with sinusoids, each generation loads one of the
anatomies built by ``topbrain_tools/graft_siphon.py``, whose distal RCCA is a
real TopBrain patient's internal carotid.

    DualDeviceNavTopBrain(anatomy_dir="topbrain_data/anatomies",
                          seed=base_seed + worker_id,
                          episodes_between_change=10)

Hold anatomies out for evaluation with ``exclude`` on the training env and
``only`` on the test env, so a transfer number is measured on siphons the
policy has never seen:

    train = DualDeviceNavTopBrain(exclude=HELD_OUT)
    test  = DualDeviceNavTopBrain(only=HELD_OUT)
"""

import os
from typing import List, Optional, Sequence, Tuple

import eve
# Imported from its own module, NOT via eve.intervention.vesseltree: the
# package __init__ is bind-mounted into the container by every launcher,
# so adding an import there would break every run that does not also
# mount this file. This way only a TopBrain run needs the extra mount.
from eve.intervention.vesseltree.topbrainanatomyset import TopBrainAnatomySet

from .dualdevicenav import load_branches

_RCCA_NAME = "Centerline curve - RCCA.mrk"
_CENTERLINE_SUBDIR = "Centrelines_comb"
_MESH_NAME = "vessel_architecture_collision.obj"


def find_anatomies(
    anatomy_dir: str,
    only: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """(roots, names) of the anatomy folders to use, in a stable order."""
    if not os.path.isdir(anatomy_dir):
        raise FileNotFoundError("no anatomy directory at %s" % anatomy_dir)
    names = sorted(
        d for d in os.listdir(anatomy_dir)
        if os.path.isdir(os.path.join(anatomy_dir, d, _CENTERLINE_SUBDIR))
    )
    if only is not None:
        wanted = set(only)
        missing = wanted - set(names)
        if missing:
            raise ValueError("anatomies not found: %s" % sorted(missing))
        names = [n for n in names if n in wanted]
    if exclude is not None:
        names = [n for n in names if n not in set(exclude)]
    if not names:
        raise ValueError(
            "no anatomies left in %s after only/exclude" % anatomy_dir
        )
    return [os.path.join(anatomy_dir, n) for n in names], names


class DualDeviceNavTopBrain(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        anatomy_dir: str = "topbrain_data/anatomies",
        seed: Optional[int] = None,
        episodes_between_change: int = 10,
        only: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        image_rot_zx: Optional[Tuple[float, float]] = None,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
        target_min_arclength_mm: float = 40.0,
    ) -> None:
        roots, names = find_anatomies(anatomy_dir, only, exclude)
        branch_lists = [load_branches(os.path.join(r, _CENTERLINE_SUBDIR))
                        for r in roots]                    # same (y,-z,-x) frame
        mesh_paths = [os.path.join(r, _MESH_NAME) for r in roots]

        vessel_tree = TopBrainAnatomySet(
            branch_lists=branch_lists,
            mesh_paths=mesh_paths,
            anatomy_names=names,
            rcca_name=_RCCA_NAME,
            episodes_between_change=episodes_between_change,
            seed=seed,
        )

        # Devices, sim and view identical to DualDeviceNav / RCCAVaried, so
        # the frames and the action-to-mm mapping stay comparable.
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
        # Same near-ostium exclusion as the procedural env: a target within
        # target_min_arclength_mm of the RCCA entry would be a trivial
        # deflect-and-stop, so every target sits in the siphon.
        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=[_RCCA_NAME],
            min_arclength_from_start=target_min_arclength_mm,
        )

        super().__init__(
            vessel_tree,
            [device1, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )
