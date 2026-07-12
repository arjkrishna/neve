"""Procedural right carotid -> internal carotid -> cavernous siphon.

Extends the eve CHS artery library (which stops at the COMMON carotid,
~z=230) with the full RCCA -> RICA -> carotid-siphon -> terminus chain as
ONE ``BranchWithRadii`` named "rcca". The mesh is built from centerlines
(voxel -> marching cubes, watertight by construction), so real siphon
tortuosity requires NO surface-mesh boolean — the "hybrid" approach: the
distal control points can be hand-seeded from real anatomy (below) or,
later, fit from a dataset (TopCoW ICA centerlines).

Coordinate convention (matches ``right_common_carotid``): local frame with
+z cranial, built from ``start`` (the arch take-off); AorticArch rotates /
scales the whole tree afterwards. CHS ``radius`` param is DIAMETER — the
stored branch radius is half (``chs_to_cl_points`` divides by 2), so the
diameter values below read as real millimetres.

Anatomy (Bouthillier ICA segments C1-C7), cumulative length from the CCA
origin, and the CHS control point that encodes each landmark:

  landmark                    cum.len  diam   feature (turn / radius-of-curv)
  CCA origin (arch take-off)     0mm    8.0   straight ascent
  carotid bifurcation (->ICA)  ~105mm   6.5   bulb, then ICA taper
  cervical ICA (C1) mid        ~150mm   5.0   straight OR coil/kink (tortuosity)
  petrous genu (C2)            ~195mm   4.5   vertical->horizontal ~90deg, R~6mm
  cavernous ascend (C4)        ~215mm   4.0   ascends into the sinus
  posterior genu (C4)          ~228mm   4.0   ~180deg bend, R~4mm  <- sharpest
  anterior genu (C4)           ~240mm   3.8   ~180deg bend, R~5mm  <- sharpest
  supraclinoid (C6)            ~255mm   3.5   ascends, posterior turn
  terminus (C7, MCA/ACA bif)   ~265mm   3.2   bifurcation

The two cavernous genua are the catheter-overshoot region: tight ~180deg
bends within ~15mm. ``direction_magnitude`` at a genu control point sets
how loosely the Hermite spline bows through it — SMALL magnitude at a genu
makes the spline overshoot into a tight loop (a tortuous siphon), large
magnitude keeps it taut (a gentle siphon). ``tortuosity`` scales the
lateral genu offsets and inversely the magnitudes, so one scalar sweeps
gentle -> severe siphons (elderly / atherosclerotic vessels are more
tortuous).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..util.branch import BranchWithRadii
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points


@dataclass
class SiphonParams:
    """Per-worker sampling ranges for the RCCA->siphon anatomy. Means are
    anatomically reasonable; sigmas give inter-subject variation. Sample a
    concrete instance per generated vessel with :meth:`sample`."""

    # Entry / proximal geometry -----------------------------------------
    cca_length_mean_sigma: Tuple[float, float] = (105.0, 12.0)  # mm, CCA->bif
    cervical_length_mean_sigma: Tuple[float, float] = (48.0, 8.0)  # mm, C1
    # Diameters (mm) at each landmark; per-vessel diameter scale on top.
    diam_cca: float = 8.0
    diam_bifurcation: float = 6.5
    diam_cervical: float = 5.0
    diam_petrous: float = 4.5
    diam_cavernous: float = 4.0
    diam_supraclinoid: float = 3.5
    diam_terminus: float = 3.2
    diameter_scale_mean_sigma: Tuple[float, float] = (1.0, 0.08)

    # Tortuosity — 0.6 (gentle) .. 1.25 (severe siphon/coiling). Scales the
    # lateral genu offsets and inversely the spline magnitudes. MESHING
    # CONSTRAINT: the voxel->marching-cubes mesher self-intersects the
    # vessel's INNER wall wherever the centerline radius-of-curvature drops
    # below the local vessel radius (~2 mm at the cavernous genua). Verified
    # numerically: tort=1.0 -> min ROC ~2.1 mm (safe), tort=1.4 -> ~1.3 mm
    # (self-intersects). The ceiling is clipped to 1.25 (min ROC ~1.7 mm,
    # rounded out by the 2x gaussian smooth); raise only after a SOFA smoke
    # test confirms the collision surface stays manifold.
    tortuosity_mean_sigma: Tuple[float, float] = (1.0, 0.22)
    tortuosity_clip: Tuple[float, float] = (0.6, 1.25)

    # Cervical-ICA coil/kink lateral excursion (mm) — the biggest realistic
    # C1 variation; scaled by tortuosity.
    cervical_kink_mm_sigma: float = 6.0

    # Petrous knee turn strength (anterior component of the tangent) and
    # cavernous genu lateral offsets (mm, before tortuosity scaling).
    petrous_turn_mean_sigma: Tuple[float, float] = (0.75, 0.12)
    post_genu_offset_mm: float = 7.0
    ant_genu_offset_mm: float = 8.0
    # Genu spline magnitudes (small = tighter loop). Scaled by 1/tortuosity.
    genu_magnitude_mean_sigma: Tuple[float, float] = (0.9, 0.12)

    def sample(self, rng: np.random.Generator) -> "SampledSiphon":
        n = rng.normal
        tort = float(np.clip(
            n(*self.tortuosity_mean_sigma), *self.tortuosity_clip
        ))
        dscale = float(max(0.7, n(*self.diameter_scale_mean_sigma)))
        return SampledSiphon(
            cca_length=float(max(70.0, n(*self.cca_length_mean_sigma))),
            cervical_length=float(max(25.0, n(*self.cervical_length_mean_sigma))),
            tortuosity=tort,
            diameter_scale=dscale,
            cervical_kink=float(n(0.0, self.cervical_kink_mm_sigma) * tort),
            petrous_turn=float(n(*self.petrous_turn_mean_sigma)),
            genu_magnitude=float(max(
                0.35, n(*self.genu_magnitude_mean_sigma) / max(tort, 0.5)
            )),
            post_genu_offset=self.post_genu_offset_mm * tort,
            ant_genu_offset=self.ant_genu_offset_mm * tort,
            _params=self,
        )


@dataclass
class SampledSiphon:
    """One concrete draw of the siphon geometry (per vessel)."""

    cca_length: float
    cervical_length: float
    tortuosity: float
    diameter_scale: float
    cervical_kink: float
    petrous_turn: float
    genu_magnitude: float
    post_genu_offset: float
    ant_genu_offset: float
    _params: SiphonParams = field(repr=False, default=None)


def right_carotid_siphon(
    start: Tuple[float, float, float],
    resolution: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    sampled: Optional[SampledSiphon] = None,
    params: Optional[SiphonParams] = None,
    name: str = "rcca",
) -> Tuple[BranchWithRadii, List[CHSPoint]]:
    """Generate the RCCA->RICA->cavernous-siphon->terminus centerline as one
    ``BranchWithRadii``. Returns (branch, chs_points).

    Pass a ``sampled`` draw for reproducibility, or ``params`` (or neither,
    for defaults) to sample internally from ``rng``.
    """
    rng = rng or np.random.default_rng()
    if sampled is None:
        sampled = (params or SiphonParams()).sample(rng)
    p = sampled._params or SiphonParams()
    ds = sampled.diameter_scale

    def dia(x: float) -> float:
        # Floor the diameter so a rare unlucky radius draw can't go
        # non-positive (chs_to_cl_points would emit a negative stored
        # radius -> the mesher marks an inverted tube).
        return max(0.4, x * ds)

    chs: List[CHSPoint] = []

    # CP0 — arch take-off (CCA origin). No positional jitter (it must sit on
    # the arch); ascend +z with a slight anterior lean.
    chs.append(chs_point_normal(
        coords_mean=(0.0, 0.0, 0.0), coords_sigma=(0.0, 0.0, 0.0),
        direction_mean=(0.15, 0.0, 1.0), direction_sigma=(0.15, 0.15, 0.2),
        direction_magnitude_mean_and_sigma=(1.5, 0.2),
        radius_mean_and_sigma=(dia(p.diam_cca), 0.3),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP1 — carotid bifurcation (CCA -> ICA). Straight ascent; mild lateral.
    z1 = sampled.cca_length
    chs.append(chs_point_normal(
        coords_mean=(0.0, 0.0, z1), coords_sigma=(2.0, 2.0, 0.0),
        direction_mean=(0.0, 0.0, 1.0), direction_sigma=(0.1, 0.1, 0.0),
        direction_magnitude_mean_and_sigma=(1.4, 0.2),
        radius_mean_and_sigma=(dia(p.diam_bifurcation), 0.3),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP2 — cervical ICA (C1). Coil/kink via a tortuosity-scaled lateral
    # excursion; this is the biggest realistic proximal-ICA variation.
    z2 = z1 + sampled.cervical_length
    chs.append(chs_point_normal(
        coords_mean=(sampled.cervical_kink, 0.5 * sampled.cervical_kink, z2),
        coords_sigma=(2.0, 2.0, 0.0),
        direction_mean=(0.0, 0.0, 1.0),
        direction_sigma=(0.15 * sampled.tortuosity, 0.15 * sampled.tortuosity, 0.0),
        direction_magnitude_mean_and_sigma=(1.4, 0.25),
        radius_mean_and_sigma=(dia(p.diam_cervical), 0.25),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP3 — petrous genu (C2): vertical -> horizontal (anteromedial) ~90deg.
    z3 = z2 + 30.0
    chs.append(chs_point_normal(
        coords_mean=(6.0, 0.0, z3), coords_sigma=(2.0, 2.0, 1.0),
        direction_mean=(sampled.petrous_turn, 0.0, 0.5),
        direction_sigma=(0.12, 0.12, 0.1),
        direction_magnitude_mean_and_sigma=(1.0, 0.15),
        radius_mean_and_sigma=(dia(p.diam_petrous), 0.2),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP4 — cavernous ascending: turn back upward into the sinus.
    z4 = z3 + 14.0
    chs.append(chs_point_normal(
        coords_mean=(9.0, 0.0, z4), coords_sigma=(1.5, 1.5, 1.0),
        direction_mean=(0.1, 0.0, 1.0), direction_sigma=(0.1, 0.1, 0.1),
        direction_magnitude_mean_and_sigma=(sampled.genu_magnitude, 0.1),
        radius_mean_and_sigma=(dia(p.diam_cavernous), 0.2),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP5 — posterior genu: the first ~180deg cavernous bend. Lateral/anterior
    # offset scaled by tortuosity; small magnitude -> tight loop.
    z5 = z4 + 12.0
    chs.append(chs_point_normal(
        coords_mean=(9.0 - sampled.post_genu_offset, 0.0, z5),
        coords_sigma=(1.5, 1.5, 1.0),
        direction_mean=(-0.6, 0.0, 0.5), direction_sigma=(0.15, 0.12, 0.12),
        direction_magnitude_mean_and_sigma=(sampled.genu_magnitude, 0.1),
        radius_mean_and_sigma=(dia(p.diam_cavernous), 0.2),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP6 — anterior genu: the second ~180deg bend (completes the S/siphon).
    z6 = z5 + 12.0
    chs.append(chs_point_normal(
        coords_mean=(9.0 - sampled.post_genu_offset + sampled.ant_genu_offset,
                     0.0, z6),
        coords_sigma=(1.5, 1.5, 1.0),
        direction_mean=(0.5, 0.0, 0.7), direction_sigma=(0.15, 0.12, 0.12),
        direction_magnitude_mean_and_sigma=(sampled.genu_magnitude, 0.1),
        radius_mean_and_sigma=(dia(p.diam_cavernous - 0.2), 0.2),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP7 — supraclinoid (C6): ascends with a posterior turn.
    z7 = z6 + 15.0
    chs.append(chs_point_normal(
        coords_mean=(9.0 - sampled.post_genu_offset + 0.4 * sampled.ant_genu_offset,
                     0.0, z7),
        coords_sigma=(1.5, 1.5, 1.0),
        direction_mean=(-0.2, 0.0, 1.0), direction_sigma=(0.12, 0.1, 0.1),
        direction_magnitude_mean_and_sigma=(1.0, 0.15),
        radius_mean_and_sigma=(dia(p.diam_supraclinoid), 0.2),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    # CP8 — terminus (C7, MCA/ACA bifurcation).
    z8 = z7 + 10.0
    chs.append(chs_point_normal(
        coords_mean=(9.0 - sampled.post_genu_offset + 0.6 * sampled.ant_genu_offset,
                     0.0, z8),
        coords_sigma=(1.5, 1.5, 1.0),
        direction_mean=(0.3, 0.0, 0.8), direction_sigma=(0.12, 0.1, 0.1),
        direction_magnitude_mean_and_sigma=(1.0, 0.15),
        radius_mean_and_sigma=(dia(p.diam_terminus), 0.2),
        d_radius_mean_and_sigma=(0.0, 0.0), coord_offset=start, rng=rng,
    ))

    cl_coordinates, cl_radii = chs_to_cl_points(chs, resolution)
    return BranchWithRadii(name, cl_coordinates, cl_radii), chs
