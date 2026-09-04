#!/usr/bin/env python3
"""Three-source graft: host arch -> real carotid bifurcation -> TopBrain siphon.

The existing 49 anatomies vary only in the siphon; every one of them shares a
single cervical carotid and has no carotid bifurcation at all, because the
host's RCCA is one continuous branch. This composes three real sources instead:

    host        ostium to the first seam, the arch and proximal CCA
    lower       a real CCA, a real ICA/ECA bifurcation, and cervical ICA
    siphon      a real TopBrain ICA from the skull base to the terminus

The ECA becomes a NEW branch in the tree. That is the point: entering the
external carotid is a real clinical error, and the current anatomy cannot
represent it because no such fork exists.

SEAM PLACEMENT. The siphon seam is pinned at 130 mm of composed arclength so
these anatomies stay comparable with the 49 that already exist, and the host
cut falls out of that: host_cut = 130 - (cca + ica). Clamped to [15, 72] mm,
since below 15 the host contributes nothing meaningful and above 72 it has
stopped being CCA-calibre. When a lower is long enough to overrun, its CCA is
trimmed proximally rather than the seam being moved.

Both seams get the frame match (tangent onto tangent, superior onto superior)
and the smoothstep radius ramp that graft_siphon.py established, for the same
reasons: a tangent-only match leaves the roll to chance, and an unramped
calibre step is a discontinuity the mesher turns into a shelf.

CALIBRE. Two invariants and one thing that is deliberately NOT one.

  * The ostium radius is 5.8121 mm in every anatomy of both sets, and more
    generally the shipped host's first 10 mm are byte-for-byte the same
    everywhere. That is what makes the two sets comparable; the ramp is
    bounded by BLEND_ANCHOR_MM so it can never reach into that stretch.
  * A donor's calibre at its own CUT FACE is a measurement artifact and is
    never used as a join target; see CCA_ANCHOR_MM.
  * The step in calibre ACROSS THE CCA/ICA BIFURCATION is left exactly as the
    donor measured it. It is the only seam here with no ramp, and that is
    correct: the bifurcation is interior to one donor, not a join between two
    sources. Composition reproduces the donor's own profile there to within
    0.09 mm, the step runs from -1.59 to +1.55 mm across 4 mm with the sign
    split 26/23 between donors, and in 42 of 49 donors some other 4 mm window
    of the same vessel carries a larger step. It is the carotid bulb and its
    taper into the ICA -- the clinical reason these donors are here at all --
    and ramping it would erase the one calibre feature the set exists to add.

    python carotid_tools/graft_three.py --pairing <json> --out <dir> [--only NAME]
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graft_siphon import (RCCA_FILE, RVA_FILE, anchor_trim, angle_deg, arclength,
                          frame_rotation, is_cranial_stub, kink_deg,
                          min_clearance, prep_siphon, read_curve, resample,
                          rva_deflect, rva_shorten, smoothstep, tangent_at,
                          unit, write_curve, json_to_branch, branch_to_json)
from analyze_bifurcations import read_centerlines, split_tree

SIPHON_SEAM_MM = 130.0
HOST_CUT_MIN, HOST_CUT_MAX = 15.0, 72.0
BLEND_MM = 25.0
# The radius ramp is allowed to overwrite host calibre, but only where the host
# is cervical vessel. The shipped RCCA opens INSIDE the aortic arch: its radius
# falls 5.81 -> 2.49 mm monotonically over the first 10 mm and is then flat
# (2.27-2.63 mm) all the way to 55 mm. Those first 10 mm are the ostium flare,
# they are identical in every anatomy of both sets, and the ostium radius of
# 5.81 mm is the invariant every anatomy is supposed to share.
#
# A fixed 25 mm window does not respect that. host_cut lands at 15.0-53.6 mm
# across the 229 pairs, so the window start (host_cut - 25) fell below 10 mm in
# 48 of them and below ZERO in ten -- and where it goes negative, smoothstep is
# already off zero at s=0 and rewrites the first point itself: the ostium came
# out at 5.00 mm (case_m_006_left, host_cut 15.0, weight 0.352) and 5.60 mm
# (case_w_008_right, host_cut 20.0, weight 0.103) instead of 5.81. Measured
# against the shipped host, the ramp moved proximal radii by up to 1.81 mm.
#
# Anchoring the window at 10 mm instead of at a fixed length leaves the flare
# alone by construction. HOST_CUT_MIN (15) > BLEND_ANCHOR_MM (10) keeps the
# shortest ramp at 5 mm.
BLEND_ANCHOR_MM = 10.0
# How much of a donor section's calibre at its own CUT FACE to distrust.
#
# The bifurcation models are cropped lumens, and every face where the lumen was
# cut is narrower than the vessel behind it. Measured over all 49 donors, median
# radius against distance from the face:
#
#     face          0.0    1.0    2.0    4.0    8.0   12.0 mm
#     CCA inlet    2.445  2.851  3.015  3.092  3.110  3.235   (+0.64 by 4 mm)
#     ICA tip      1.781  2.070  2.080  2.153  2.125  2.138   (+0.30)
#     ECA tip      0.511  0.760  0.806  0.898  1.281  1.562   (+0.42)
#     CCA at bif   3.332  3.376  3.364  3.342  3.305  3.261   (-0.05)
#
# Every cut face droops and recovers within ~4 mm; the one end of the CCA that
# is NOT a cut face -- its distal end, inside the model at the bifurcation --
# does not droop at all, it tapers the way a real vessel does.
#
# This is NOT, as an earlier version of this comment claimed, an artifact of
# VMTK's inscribed sphere having nothing to grow into past the boundary. The
# SURFACE really is pinched there. Probing the donor STL directly with a
# cap-independent measure -- minimum wall distance inside a 1.5 mm slab normal
# to the centerline, so a flat cut face is parallel to the probe and cannot be
# hit -- tracks the MISR value to 0.03-0.04 mm at every distance:
#
#     distance      0.0    1.0    2.0    4.0    8.0   12.0 mm
#     MISR         2.808  3.081  3.152  3.229  3.274  3.239
#     slab probe   2.779  2.998  3.085  3.192  3.232  3.214
#
# The radii are faithfully reporting a segmented surface that converges toward a
# point at each crop (the ECA tip's true half-width at 0 mm is 0.040 mm). So the
# value at a crop is model-preparation geometry, not vessel calibre -- and a
# common carotid does not taper as it approaches the aorta, which makes a
# proximal-CCA minimum anti-anatomical: 49 of 49 donors are wider 4 mm in than
# at the face, and in 39 the inlet is the narrowest point of the entire CCA.
# Worth stating precisely, because it is also what stops someone re-running
# vmtkCenterlines with different settings expecting the droop to disappear.
#
# It matters because the ramp used to take cca_r[0] as its target. That aimed
# the host at the artifact and put a spurious waist at the seam in all 229
# anatomies -- the composed radius rose 0.38 mm within 1 mm of the seam and
# 0.61 mm within 4 mm, a taper reversal absent from set A, whose donors (the
# TopBrain siphons) have no such droop (+0.05 mm over their first 4 mm). Aim at
# the calibre 4 mm in and hold the donor's first 4 mm there. It costs 4 mm of
# measured-but-untrustworthy radii per anatomy and no geometry at all. The ICA
# tip's own droop needs no such repair: the seam-2 ramp already overwrites the
# last 25 mm of the ICA, which is why seam 2 is clean (+0.01 mm at 1 mm).
CCA_ANCHOR_MM = 4.0
RESAMPLE_MM = 1.0
DISTAL_TRIM_MM = 4.0
ECA_NAME = "Centerline curve - RECA.mrk.json"
# The ECA is here to be a WRONG TURN, not a complete external carotid. The
# donors carry 17-70 mm of it, and past roughly the first third that length is
# doing nothing for the task while being the single largest source of
# collisions with the host's vertebral artery -- a collision that is an
# artifact of dropping a donor bifurcation into a different host, not anything
# real. Cap it, and trim further if it still hits something; below ECA_MIN_MM
# there is no longer enough vessel to enter and the anatomy is rejected instead.
ECA_MAX_MM = 30.0
# 17.0, not 20.0, and the number had to be re-derived once ECA_MESH_R_MM landed
# because that fix changed what "enough vessel to enter" means. The guard was
# never enforced in that sense before: with the fork eroding out of the mesh,
# 23 anatomies shipped with under 10 mm of ENTERABLE ECA and around 100 with
# under 20 mm, and none were rejected. Now that declared length is enterable
# length, the three donors sitting at 17.46 / 17.87 / 18.54 mm deliver a fully
# present fork -- better than what those 100 had -- and a 20.0 floor would
# delete 14 real anatomies to enforce a standard the set has never met. Next
# shortest donor is 21.41 mm, so 17.0 is a floor with room, not a fitted value.
ECA_MIN_MM = 17.0
# The mesher erodes before it iso-surfaces. mark_centerline_in_array lays down a
# BINARY tube, gaussian_smooth(1) then runs TWICE -- sigma sqrt(2) voxels, which
# is 0.85 mm across x/y and 1.27 mm along z -- and marching cubes cuts the half
# level. A tube thinner than about one voxel-sigma never reaches that level, so
# it is not in the .obj at all. Measured by pushing straight test tubes through
# the real pipeline: a superiorly-running tube first appears at r = 1.15 mm, one
# running in the axial plane at r = 1.45 mm, and whatever survives is roughly
# 0.55 mm narrower than declared. Fitted against the 49 ECAs actually built, the
# effective threshold is 1.30 mm: "first station where r drops below 1.30" and
# "first station where the centerline leaves the mesh" agree to r = 0.998, mean
# absolute error 0.44 mm, which is under one voxel.
#
# 1.6 mm is the smallest floor that puts the whole ECA into the mesh in ANY
# orientation and still leaves a lumen (0.75-0.95 mm radius) that the 0.7 mm
# catheter can enter. It stays below these same 49 donors' own median ECA radius
# at every arclength (1.85 mm at 20 mm, 1.65 mm at 25 mm), so it never inflates a
# fork past what the cohort calls normal there; it only lifts the bottom quartile,
# which is where centerline radius estimation goes to pieces anyway -- 7 of the
# 38 affected donors dip under 1.6 mm and then recover above 1.8 mm further out,
# which no real terminal taper does.
ECA_MESH_R_MM = 1.6
# The same erosion floor, applied to the ROUTE through the donor section. This
# is the one that decides whether an anatomy is navigable at all.
#
# The bifurcation donors are STENOSIS patients -- lower_manifest.json carries
# stenosis_pct 3-74%, and corr(ica_min_d, stenosis_pct) = -0.92 -- so their real
# lumen dips below the floor above MID-VESSEL, not at either end. 27 of 229
# composed routes closed completely at 29-54% of the path: the field on the
# centerline fell to 0.210-0.495 against the 0.500 iso level, the meshed lumen
# radius there was 0.00 mm, and 83-158 route points ended up on a mesh component
# that does not contain the arch. The device cannot reach any target past that,
# and nothing in the pipeline noticed, because match_sections.MIN_TIP_D guards
# the ICA TIP -- which the seam-2 ramp overwrites anyway.
#
# It pre-compensates the DECIMATION as much as the smoothing, which matters to
# anyone re-deriving it: on the undecimated iso-surface a 1.45 floor already
# leaves 0.35-0.74 mm of lumen and nothing seals, and at decimate(0.95) none of
# the ten anatomies that seal at 1.45 still do. Roughly half of what this floor
# buys is compensation for decimate(0.99), which is kept because the shipped
# baseline patient's own collision mesh is 3,583 triangles and SOFA collision
# cost is the training bottleneck. Re-derive against the un-decimated surface
# and you will get ~1.35 and wrongly conclude 1.60 is too high.
#
# 1.60 mm chosen by sweep: 1.30 still leaves case_w_047_left__topcow_mr_005 at
# field 0.487 with 134 off-arch points; 1.45 clears every seal but leaves a
# meshed lumen as thin as 0.40 mm radius against a 0.35 mm catheter radius;
# 1.60 gives 0.75-1.35 mm, which is what the healthy anatomies of this same set
# reach at their tightest (0.95-1.65 mm). Hard, not smoothstepped: a C1 floor at
# the same plateau rewrites 2-2.5x more arclength for a field minimum identical
# to three decimals, and the mesher's own ~1 mm Gaussian rounds the corner
# either way.
#
# Scoped to the DONOR section only. The host arch and the TopBrain siphon keep
# exactly the calibre set A ships, so the pre-existing distal-siphon taper stays
# identical in both sets rather than being quietly repaired in B alone.
#
# What this costs is the DEPTH of the stenosis over a short stretch -- median
# 7.0 mm of route rewritten, median radius raise 0.41 mm, grades falling from
# 59-74% to 20-30%. That is a real loss of pathology and it is disclosed per
# anatomy in provenance.json. But the grade being erased was never in the
# collision mesh: at those calibres the mesher produced no lumen at all, so the
# choice is between a 27% stenosis and a sealed vessel, not between 73% and 27%.
ROUTE_MIN_R = 1.60
# Floor on the SIPHON radius. 0 = off (as shipped). ROUTE_MIN_R is scoped to the
# donor section on purpose, which left the TopBrain siphon at whatever the label
# gave it -- including the one-voxel necks label_necks.py finds in 20 of 50
# vessels. The v2 mesher renders those faithfully, as 0.3-0.6 mm pinches: the
# first v2 bake of set B flagged three anatomies as non-navigable at 0.37-0.57 mm,
# every one on a necked _L siphon. Set A gets the same floor from graft_siphon.
SIPHON_MIN_R = 0.0
CLEAR_MARGIN_MM = 0.5
# Positive clearance is not enough: two lumens closer than this MERGE in the
# collision mesh even though their centerlines never overlap.
#
# The mesher smooths a binary tube twice at sigma sqrt(2) voxels and iso-
# surfaces at the half level, so it cannot resolve a wall thinner than roughly
# one voxel-sigma. Measured on the shipped set: case_m_024_left sits at a
# centerline gap of +0.057 to +0.075 mm between its route and its own ECA, well
# clear by the >0 test that was applied -- and 4 of its 5 anatomies bake into a
# single continuous lumen there, with a 1.60 mm channel against a 0.35 mm
# catheter radius. A ring the device can drive around, that no patient has.
#
# So every clearance gate here is the fusing band, not zero.
FUSE_BAND_MM = 0.35
# How the tangent that ORIENTS a donor section is measured.
#
# place() rotates an ENTIRE section -- and everything grafted downstream of it
# -- so that its start tangent meets the anchor tangent. Measured over the
# default 5 mm that start tangent is a local sample taken exactly where
# centerline seeding artifacts live, and when those 5 mm are unrepresentative
# the whole section tumbles. case_w_024_right leaves its CCA inlet 54 deg away
# from the direction its own vessel runs, so its composed lower sat 43 deg off
# vertical where the donor's own scan has it at 12; case_w_037_right sat at 53
# against a native 1.4.
#
# The ground truth is the donor's OWN elevation in its own scan frame -- all
# three sources carry superior as +z, so a correctly placed lower should keep
# roughly the tilt the donor really has. Measured against that over 49 donors,
# widening the span helps, but a GLOBAL widening is the wrong instrument:
#
# A GLOBAL widening is the wrong instrument. Measured over the full builder,
# all 240 pairs, a fixed 15 mm span costs 16 anatomies and gains 7 (net -9):
# pushing every donor's lower into a new pose moves 46 routes into a
# neighbouring lumen that were clear before, kills ten ECAs on ECA_MIN_MM, and
# churns 36 repairs on / 21 off. case_w_046_left__topcow_mr_027 is rejected
# outright at -3.44 mm inside the LVA.
#
# But widening only WHERE the sample is bad is also wrong, and this is the part
# that took a second look. Widening keeps the unrepresentative points AND
# measures the tangent over a chord that is no longer the tangent AT the joint,
# so the joint itself bends: every one of the 25 pairs it fired on got a worse
# seam-1 kink, median +6.1 deg and worst +28.1 (12.6 -> 40.7 on
# case_w_024_right__topcow_mr_026_L, the very case it was meant to fix), taking
# the set-wide seam-1 maximum past anything in set A. It also pushed three
# case_w_040_left anatomies from clear into the host's vertebral artery,
# earning them an RVA deflection they had not needed.
#
# TRIM, do not average -- exactly what graft_siphon.anchor_trim() does to a
# siphon that opens by doubling back. Advance the CCA inlet until its own local
# tangent agrees with where the section goes, then frame-match that. Same gate,
# same 5 donors, same 25 pairs, and measured over the full builder:
#
#                       built  tilt p90/max  >40  seam1 med/p90/max  new RVA
#     span 5             227    33.3 / 52.7   15   15.9 / 21.5 / 29.1    -
#     adaptive span      227    27.4 / 35.8    0   16.6 / 24.4 / 40.7    3
#     inlet trim 15/15   226    27.4 / 52.6    9   13.7 / 19.9 / 29.1    4
#     inlet trim 25/12   216    22.0 / 33.2    0   ... measured on the set
#
# The 15/15 row is what actually shipped first and it did NOT deliver the row
# above it: the budget was smaller than the problem on 8 donors, head_trim fell
# through to 0, and those 38 anatomies got the untrimmed inlet. 25/12 is the
# budget that reproduces the intended result. See INLET_MAX_TRIM_MM.
#
# The seam-1 regression disappears and the tilt fix is kept. It costs 9-14 mm
# of proximal CCA on those five donors -- which host_cut = 130 - cca - ica
# hands straight back to the host -- and that stretch is the cut-face region
# this file already declares untrustworthy under CCA_ANCHOR_MM. One extra RVA
# deflection (case_w_047_left__topcow_mr_006, 2.8 mm).
#
# START_TC_THR is 27, not 25. The donor set is stable for any threshold in
# (24.997, 29.507], and 25.0 sits 0.0033 deg from case_w_007_right -- a
# knife-edge on a data point, where re-running vmtkCenterlines with different
# settings could flip which donors are treated. 27 is the middle of that range.
#
# NOT applied to the siphon, which place() also positions. The siphon is the
# LAST section: nothing is grafted downstream of it, so a start-tangent error
# there has nowhere to propagate. Taking its start over 15 mm drives the worst
# seam-2 kink 52.0 -> 65.3 and pushes case_k_005_right__topcow_mr_024_L past
# the 60 deg --max-kink gate for nothing.
#
# NOT applied to ANCHORS either (the host cut point, the ICA tip). There the
# local tangent IS the direction of travel, and a long backward chord chases
# the curve instead of following it: spanning the ICA tip over 15 mm drives the
# worst seam-2 kink to 102.1 deg and puts 10 pairs over the gate.
#
# Note tangent_at(p, i, span) takes span as an INDEX offset. It is millimetres
# here only because everything is resampled to RESAMPLE_MM = 1.0.
START_TC_THR = 27.0
# 25/12, not 15/15. At 15/15 the search budget was smaller than the problem:
# case_w_037_right needs 21 mm of trim to bring its inlet under the threshold
# and case_w_047_left cleared it at 13.9 mm but was refused because that left
# 14.88 mm of CCA against a 15 mm floor -- a 0.12 mm miss. head_trim then fell
# through to `return 0` and shipped the untrimmed inlet, so the gate fired on 19
# donors, actually trimmed 11, and silently gave up on 8 (38 anatomies) --
# including the very case this fix was written around. Those 38 carried a
# composed-vs-native pose error of median 15.1 / max 51.2 deg against 1.9 / 12.8
# for the donors that did get trimmed. 25/12 reproduces the table below.
INLET_MAX_TRIM_MM = 25.0
INLET_MIN_KEEP_MM = 12.0
SIPHON_SPAN = 5


def clear_profile(p, r, branches):
    """Per-point lumen gap to the nearest neighbouring branch (negative = inside)."""
    out = []
    for _, bp, br in branches:
        d = np.linalg.norm(p[:, None, :] - bp[None, :, :], axis=2)
        out.append((d - r[:, None] - br[None, :]).min(axis=1))
    return np.min(np.stack(out), axis=0) if out else np.full(len(p), np.inf)


def eca_reentry(rp, rr, ep, er):
    """Where the ECA re-enters the route AWAY from the bifurcation, if it does.

    The fork legitimately shares lumen with the route at its origin -- that is
    what a bifurcation is -- so a plain route-vs-ECA clearance test is useless
    here: it reads -3.7 to -10.2 mm in all 220 anatomies and would reject every
    one of them. What is NOT legitimate is a SECOND communication further out,
    which is a closed ICA-ECA ring the catheter can drive around and no patient
    has.

    So the test is topological, not metric. Walk the ECA from its origin: the
    contiguous opening run at the start is the bifurcation and is expected. Any
    overlap that resumes after that run has ended is a re-entry, and the ECA is
    trimmed back before it.

    It happens because ECA_MESH_R_MM inflates a thin distal fork back into the
    vessel it came from: at the offending stations the donor's native ECA
    radius is 0.50-0.73 mm with a real positive gap to the ICA. Measured on the
    shipped set, 8 of 220 anatomies across 2 donors carried such a ring.

    Returns the ECA arclength to cut at, or None if the fork is clean.
    """
    d = np.linalg.norm(rp[:, None, :] - ep[None, :, :], axis=2)
    overlapping = ((d - rr[:, None] - er[None, :]) < FUSE_BAND_MM).any(axis=0)
    if not overlapping.any():
        return None
    end = int(np.argmax(~overlapping)) if (~overlapping).any() else len(overlapping)
    if not overlapping[end:].any():
        return None
    es = arclength(ep)
    return float(es[end + int(np.argmax(overlapping[end:]))])


def ramp_radius(s, r, join_s, target, blend=BLEND_MM):
    """Ease the calibre to `target` over the last `blend` mm before `join_s`.

    The window is [join_s - blend, join_s] and the CALLER owns it. Note what
    happens if it is allowed to reach past s[0]: the clamp on `a` keeps the
    anchor value in range, but smoothstep is evaluated at (s[0] - join_s +
    blend) / blend > 0, so the weight at the very first sample is already off
    zero and r[0] itself is rewritten. That is how ten anatomies lost the
    invariant ostium radius. Both callers keep the window strictly inside the
    section: seam 1 through BLEND_ANCHOR_MM, seam 2 because every ICA reaching
    it is at least 57.9 mm long against a 25 mm window.
    """
    r = np.asarray(r, float).copy()
    a = float(np.interp(max(join_s - blend, s[0]), s, r))
    w = smoothstep((s - (join_s - blend)) / max(blend, 1e-6))
    inside = s >= (join_s - blend)
    return np.where(inside, a + (target - a) * w, r)


def place(src_p, anchor_p, anchor_t, span=5):
    """Translate src onto anchor_p and frame-match its start tangent to anchor_t.

    `span` is the chord length, in mm, the start tangent is measured over; see
    START_TC_THR and SIPHON_SPAN for why the two seams are treated differently.

    Returns (moved, R, origin) so a caller can carry SIBLING branches through
    the identical transform. The ICA and ECA have to move with their own CCA,
    and recomputing the rotation for them separately is a standing invitation
    for the two to drift apart.

    There is deliberately no `up` argument. All three donors -- the host in
    branch coordinates, the bifurcation models in scanner coordinates and the
    TopBrain siphons in RAS -- already carry superior as +z, so the module UP
    that frame_rotation uses is the right axis for every seam here. An earlier
    version took an `up` parameter and silently ignored it, which read as if
    the axis were configurable when it never was.
    """
    R = frame_rotation(tangent_at(src_p, 0, span), anchor_t)
    return (src_p - src_p[0]) @ R.T + anchor_p, R, src_p[0]


def head_trim(cca_p, ica_p):
    """Index to start the donor CCA from, so its inlet tangent means something.

    Reads the module globals rather than binding them as defaults, so a sweep
    can set START_TC_THR at runtime and actually get a different policy. The
    previous version bound them in the signature and silently ignored any
    override.
    """
    chord = ica_p[-1] - cca_p[0]
    if angle_deg(tangent_at(cca_p, 0), chord) <= START_TC_THR:
        return 0
    s = arclength(cca_p)
    best_i, best_a = 0, angle_deg(tangent_at(cca_p, 0), chord)
    for i in range(len(cca_p)):
        if s[i] > INLET_MAX_TRIM_MM or s[-1] - s[i] < INLET_MIN_KEEP_MM:
            break
        a = angle_deg(tangent_at(cca_p[i:], 0), ica_p[-1] - cca_p[i])
        if a <= START_TC_THR:
            return i
        if a < best_a:
            best_i, best_a = i, a
    # Budget exhausted without clearing the threshold. Return the BEST index
    # found rather than 0: falling back to the untrimmed inlet throws away a
    # real improvement and silently restores the defect this exists to fix.
    return best_i


def load_lower(rec):
    """(cca, ica, eca) as (points, radii), ICA already extended if it was."""
    paths = read_centerlines(rec["path"])
    cca, dau = split_tree(paths)
    if cca is None or len(dau) < 2:
        return None

    def cal(r):
        return float(np.median(r[:max(len(r) // 2, 1)])) if r is not None and len(r) else 0.0

    dau = sorted(dau, key=lambda d: -cal(d[1]))
    ica, eca = dau[0], dau[1]
    if rec.get("ext_json"):
        e = json.load(open(rec["ext_json"], encoding="utf-8"))
        ica = (np.asarray(e["points"], float),
               None if e["radii"] is None else np.asarray(e["radii"], float))
    return cca, ica, eca


def compose(host_p, host_r, lower, siphon_p, siphon_r):
    """Return (route_points, route_radii, eca_points, eca_radii, diagnostics)."""
    (cca_p, cca_r), (ica_p, ica_r), (eca_p, eca_r) = lower
    cca_p, cca_r = resample(cca_p, cca_r, RESAMPLE_MM)
    ica_p, ica_r = resample(ica_p, ica_r, RESAMPLE_MM)
    eca_p, eca_r = resample(eca_p, eca_r, RESAMPLE_MM)

    # Drop an inlet that points somewhere the vessel does not go. host_cut is
    # 130 - cca - ica, so the host simply supplies what is trimmed here.
    j = head_trim(cca_p, ica_p)
    inlet_trim = float(arclength(cca_p)[j]) if j else 0.0
    if j:
        cca_p, cca_r = cca_p[j:], cca_r[j:]

    cca_L, ica_L = arclength(cca_p)[-1], arclength(ica_p)[-1]
    host_cut = SIPHON_SEAM_MM - (cca_L + ica_L)
    trim_cca = 0.0
    if host_cut < HOST_CUT_MIN:                 # lower overruns: trim its CCA
        trim_cca = HOST_CUT_MIN - host_cut
        s = arclength(cca_p)
        if s[-1] - trim_cca < 5:
            return None, None, None, None, {"why": "CCA too short to trim"}
        keep = s >= trim_cca
        cca_p, cca_r = cca_p[keep], (None if cca_r is None else cca_r[keep])
        cca_L = arclength(cca_p)[-1]
        host_cut = HOST_CUT_MIN
    host_cut = float(min(host_cut, HOST_CUT_MAX))

    hs = arclength(host_p)
    k = int(np.searchsorted(hs, host_cut))
    k = min(max(k, 5), len(host_p) - 2)
    keep_p, keep_r = host_p[:k + 1].copy(), host_r[:k + 1].copy()

    # seam 1: host -> CCA. The ICA and ECA are carried through the CCA's own
    # transform rather than a separately recomputed one, so they cannot drift
    # away from the bifurcation they belong to.
    #
    # Join on the donor's calibre CCA_ANCHOR_MM into its CCA, not on the value
    # at its cut face, and flatten that untrustworthy stretch to the same
    # number so the two meet without a step. A CCA trimmed proximally has
    # already had its cut face removed, so it only needs whatever is left of
    # the window. Ramp the host over what is available above BLEND_ANCHOR_MM,
    # which keeps the ostium flare out of the blend entirely.
    cca_s = arclength(cca_p)
    cca_r0_face = float(cca_r[0])
    anchor = max(CCA_ANCHOR_MM - trim_cca, 0.0)
    join_r = float(np.interp(anchor, cca_s, cca_r))
    if anchor > 0.0:
        cca_r = np.where(cca_s < anchor, join_r, cca_r)
    blend = min(BLEND_MM, max(hs[k] - BLEND_ANCHOR_MM, 0.0))
    keep_r = ramp_radius(hs[:k + 1], keep_r, hs[k], join_r, blend)
    cca_moved, R, origin = place(cca_p, keep_p[-1],
                                 tangent_at(keep_p, len(keep_p) - 1),
                                 SIPHON_SPAN)

    def move(p):
        return (p - origin) @ R.T + keep_p[-1]

    ica_moved, eca_moved = move(ica_p), move(eca_p)
    es = arclength(eca_moved)
    ek = es <= ECA_MAX_MM
    if ek.sum() >= 5:
        eca_moved, eca_r = eca_moved[ek], eca_r[ek]
    # Pre-compensate the mesher's erosion. Left alone, 102 of 229 anatomies had
    # a fork that existed as a centerline and was missing from the collision
    # mesh past 20 mm, 23 of them past 10 mm, worst case 3.2 mm of a 29.5 mm
    # ECA. That is worse than having no fork: the catheter cannot enter it, so
    # there is no decision to get wrong, which is the only reason set B exists.
    eca_floored = float(np.mean(eca_r < ECA_MESH_R_MM))
    eca_r = np.maximum(eca_r, ECA_MESH_R_MM)

    # seam 2: ICA tip -> siphon
    sip_moved, R2, origin2 = place(siphon_p, ica_moved[-1],
                                   tangent_at(ica_moved, len(ica_moved) - 1), SIPHON_SPAN)
    # Both section maps, for the v3 mesher to carry the real surfaces through:
    # v_branch = R (v - origin) + anchor. The lower's ICA and ECA share the
    # CCA's map, exactly as their centerlines did.
    xform = {"lower": {"R": R.tolist(), "origin": origin.tolist(), "anchor": keep_p[-1].tolist()},
             "siphon": {"R": R2.tolist(), "origin": np.asarray(origin2).tolist(),
                        "anchor": ica_moved[-1].tolist()}}
    siphon_min_raw = float(siphon_r.min())
    siphon_floored = 0.0
    if SIPHON_MIN_R > 0:
        siphon_floored = float(np.mean(siphon_r < SIPHON_MIN_R))
        siphon_r = np.maximum(siphon_r, SIPHON_MIN_R)   # before the ramp aims at siphon_r[0]
    ica_s = arclength(ica_moved)
    ica_r2 = ramp_radius(ica_s, ica_r, ica_s[-1], float(siphon_r[0]))

    # Pre-compensate the mesher's erosion along the route, AFTER the seam-2 ramp
    # so the floor is applied to the calibre that actually ships. Donor section
    # only: keep_r (host) and siphon_r are deliberately untouched. Doing it here
    # rather than in main() also puts it ahead of the clearance checks, so any
    # overlap the widening could create is caught by the guard already there.
    raw_min = float(min(cca_r.min(), ica_r2.min()))
    route_floored = float(np.mean(np.concatenate([cca_r, ica_r2]) < ROUTE_MIN_R))
    cca_r = np.maximum(cca_r, ROUTE_MIN_R)
    ica_r2 = np.maximum(ica_r2, ROUTE_MIN_R)

    route_p = np.vstack([keep_p, cca_moved[1:], ica_moved[1:], sip_moved[1:]])
    route_r = np.concatenate([keep_r, cca_r[1:], ica_r2[1:], siphon_r[1:]])
    route_p, route_r = resample(route_p, route_r, RESAMPLE_MM)
    s_end = arclength(route_p)
    trim = s_end <= (s_end[-1] - DISTAL_TRIM_MM)
    if trim.sum() >= 30:
        route_p, route_r = route_p[trim], route_r[trim]

    kk = kink_deg(route_p)
    diag = {"host_cut_mm": host_cut, "cca_mm": cca_L, "ica_mm": ica_L,
            "trim_cca_mm": trim_cca, "inlet_trim_mm": inlet_trim, "blend_mm": float(blend),
            "join_r_mm": join_r, "cca_face_r_mm": float(cca_r0_face),
            "total_mm": float(arclength(route_p)[-1]),
            "max_kink": float(kk.max()),
            "seam1_kink": float(kk[max(k - 4, 0):k + 4].max()),
            "eca_mm": float(arclength(eca_moved)[-1]),
            "eca_floored_frac": eca_floored,
            # what the donor's real stenosis measured before the floor, and how
            # much of the donor section had to be lifted off it
            "route_min_r_raw": raw_min,
            "route_floor_mm": ROUTE_MIN_R,
            "route_floored_frac": route_floored,
            "siphon_min_r_raw": siphon_min_raw,
            "siphon_floor_mm": SIPHON_MIN_R,
            "siphon_floored_frac": siphon_floored,
            "xform": xform}
    return route_p, route_r, eca_moved, eca_r, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairing", default="carotid_data/pairing.json")
    ap.add_argument("--host", default="eve_bench/data/dualdevicenav/Centrelines_comb")
    ap.add_argument("--out", default="carotid_data/anatomies")
    ap.add_argument("--only", default=None, help="comma-separated pair indices or 'a:b' range")
    ap.add_argument("--max-kink", type=float, default=60.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave already-built anatomies alone, so a replacement "
                         "round costs only the new pairs")
    # The four constants that pre-compensate the mesher. Defaults reproduce the
    # shipped set; a v2 build passes the values re-derived for the SDF mesher.
    ap.add_argument("--route-min-r", type=float, default=ROUTE_MIN_R)
    ap.add_argument("--eca-mesh-r", type=float, default=ECA_MESH_R_MM)
    ap.add_argument("--distal-trim", type=float, default=DISTAL_TRIM_MM)
    ap.add_argument("--fuse-band", type=float, default=FUSE_BAND_MM)
    ap.add_argument("--siphon-min-r", type=float, default=SIPHON_MIN_R)
    a = ap.parse_args()
    globals().update(ROUTE_MIN_R=float(a.route_min_r), ECA_MESH_R_MM=float(a.eca_mesh_r),
                     DISTAL_TRIM_MM=float(a.distal_trim), FUSE_BAND_MM=float(a.fuse_band),
                     SIPHON_MIN_R=float(a.siphon_min_r))

    plan = json.load(open(a.pairing, encoding="utf-8"))
    pairs = plan["pairs"]
    if a.only:
        if ":" in a.only:
            lo, hi = a.only.split(":"); pairs = pairs[int(lo):int(hi)]
        else:
            pairs = [pairs[int(i)] for i in a.only.split(",")]

    tmpl, hp, hr = read_curve(os.path.join(a.host, RCCA_FILE))
    hp, hr = resample(hp, hr, RESAMPLE_MM)
    rva_tmpl, vp, vr = read_curve(os.path.join(a.host, RVA_FILE))
    rva_name = RVA_FILE.replace("Centerline curve", "cc").replace(".mrk.json", "")

    others, neigh = [], []
    for f in sorted(os.listdir(a.host)):
        if not f.endswith(".json") or f == RCCA_FILE:
            continue
        _, bp, br = read_curve(os.path.join(a.host, f))
        if is_cranial_stub(f, bp):
            continue
        others.append(f)
        if br is not None:
            neigh.append((f.replace("Centerline curve", "cc").replace(".mrk.json", ""), bp, br))

    os.makedirs(a.out, exist_ok=True)
    ok, fail = 0, []
    print("%-34s %8s %8s %7s %7s %s" % ("anatomy", "total", "host_cut", "kink", "clear", "note"))
    for idx, pr in enumerate(pairs):
        name = "%s__%s" % (pr["lower"], pr["siphon"])
        if a.skip_existing and os.path.isdir(os.path.join(a.out, name, "Centrelines_comb")):
            ok += 1
            continue
        rec = plan["lowers"][pr["lower"]]
        sinfo = plan["siphons"][pr["siphon"]]
        try:
            lower = load_lower(rec)
            if lower is None:
                raise ValueError("lower tree would not split")
            j = json.load(open(sinfo["src"], encoding="utf-8"))
            sp = np.array(j["points"], float)
            sr = np.array(j["radii"], float)
            if sinfo["mirror"]:
                sp = sp * np.array([-1.0, 1.0, 1.0])
            sp, sr = prep_siphon(sp, sr)
            # Some siphons open with the petrous segment doubling back on
            # itself, so their first point's tangent aims AWAY from the
            # direction the vessel actually goes. Frame-matching that tangent
            # onto the ICA tip then applies a near-180 deg rotation and lays
            # the whole siphon over on its side: mr_021 rose 1 mm instead of
            # 65 mm, mr_024_L 20 mm instead of 42 mm. graft_siphon.graft()
            # has always trimmed those few points before reading the proximal
            # tangent; this path did not, which is the whole defect. Only 2 of
            # 45 siphons need it, and it costs at most 4 points of anatomy the
            # lower section supplies anyway.
            j = anchor_trim(sp)
            sp, sr = sp[j:], sr[j:]
            rp, rr, ep, er, diag = compose(hp, hr, lower, sp, sr)
            if rp is None:
                raise ValueError(diag.get("why", "compose failed"))
            if diag["max_kink"] > a.max_kink:
                raise ValueError("kink %.0f deg" % diag["max_kink"])
            # ECA_MIN_MM used to govern only the clearance-driven trim further
            # down, so a donor whose NATIVE ECA was already under the floor was
            # never checked on any path: three of them (17.6, 18.0 and 18.7 mm)
            # shipped 14 anatomies with a fork shorter than the builder's own
            # declared minimum. Check the length compose() actually produced.
            if diag["eca_mm"] < ECA_MIN_MM:
                raise ValueError("ECA %.1f mm (need %.0f)"
                                 % (diag["eca_mm"], ECA_MIN_MM))

            nb = [(n, p, q) for n, p, q in neigh]
            # The graft starts at the host cut (15-54 mm), not at 130 mm.
            # Measuring only past 130 mm left the whole new CCA/ICA unchecked,
            # and the ECA -- new geometry in its entirety -- unchecked at all:
            # 27 routes and 32 ECAs were interpenetrating a neighbour, which
            # the mesher fuses into a shortcut no patient has. Below the cut
            # the route is shipped host geometry meeting its neighbours at
            # real junctions, where touching is correct and expected.
            graft_from = diag["host_cut_mm"] + 2.0

            def worst(branches):
                c, w = min_clearance(rp, rr, branches, graft_from)
                ec, ew = min_clearance(ep, er, branches, 0.0)
                return (ec, ew + " (ECA)") if ec < c else (c, w)

            note = ""
            # Trim the ECA back to its last clear point before anything else.
            # It is the cheapest thing in the anatomy to give up -- a shorter
            # fork is still a fork -- so spend it before deforming a real host
            # vessel to make room.
            eg = clear_profile(ep, er, nb)
            if eg.min() < FUSE_BAND_MM:
                es = arclength(ep)
                cutoff = float(es[int(np.argmax(eg < FUSE_BAND_MM))]) - CLEAR_MARGIN_MM
                if cutoff < ECA_MIN_MM:
                    raise ValueError("ECA clear for only %.0f mm (need %.0f)"
                                     % (max(cutoff, 0.0), ECA_MIN_MM))
                keep = es <= cutoff
                ep, er = ep[keep], er[keep]
                note = "ECA trimmed to %.0f mm" % cutoff

            # ...and separately, cut it back before any re-entry into the route
            # itself. See eca_reentry(): the fork sharing lumen with the route
            # at its ORIGIN is the bifurcation, but a second communication
            # further out is a ring the catheter can circle.
            reent = eca_reentry(rp, rr, ep, er)
            if reent is not None:
                cutoff = reent - CLEAR_MARGIN_MM
                if cutoff < ECA_MIN_MM:
                    raise ValueError("ECA re-enters the route at %.0f mm (need %.0f)"
                                     % (max(cutoff, 0.0), ECA_MIN_MM))
                es = arclength(ep)
                keep = es <= cutoff
                ep, er = ep[keep], er[keep]
                note = (note + "; " if note else "") + "ECA re-entry cut at %.0f mm" % cutoff

            clear, who = worst(nb)
            rva = None
            if clear < FUSE_BAND_MM and who.startswith(rva_name):
                # Repair against the whole GRAFTED route, not just past 130 mm.
                # Most of these conflicts sit at 70-130 mm, in the new CCA/ICA,
                # so measuring only the siphon meant the repair was aimed at a
                # stretch that was not the one colliding.
                s2 = int(np.searchsorted(arclength(rp), graft_from))
                # The ECA has to be in the comparison set. `nb` is built from
                # the host folder before the fork exists, so the repair could
                # never see it -- yet the ECA is the RVA's nearest neighbour in
                # 224 of 229 anatomies and its gap to the shipped RVA bottoms
                # out at +0.12 mm. Bending the RVA blind to it pushed one
                # anatomy straight out of the set.
                # NB not `others` -- that name already holds this host's branch
                # FILENAMES in this scope, and shadowing it feeds tuples to
                # os.path.join when the anatomy is written out.
                rva_others = ([n for n in nb if n[0] != rva_name]
                              + [(ECA_NAME.replace("Centerline curve", "cc")
                                          .replace(".mrk.json", ""), ep, er)])
                # Deflect BEFORE shortening. A bend of a couple of millimetres
                # leaves the decoy its full length, where rva_shorten was taking
                # a median 33 mm off the vessel; it only ran first because the
                # deflections used to be forced up to 4 mm and over by the
                # confluence baseline that rva_deflect now handles per point.
                bent = rva_deflect(rp[s2:], rr[s2:], vp, vr, rva_others)
                if bent is not None:
                    rva = (bent[0], bent[1])
                    note = (note + "; " if note else "") + "RVA deflected %.1f mm" % bent[2]
                else:
                    cut = rva_shorten(rp[s2:], rr[s2:], vp, vr)
                    if cut is not None:
                        rva = (cut[0], cut[1])
                        note = (note + "; " if note else "") + "RVA shortened %.0f mm" % cut[2]
                if rva is not None:
                    patched = [(n[0], rva[0], rva[1]) if n[0] == rva_name else n for n in nb]
                    clear, who = worst(patched)
            if clear < FUSE_BAND_MM:
                raise ValueError("clearance %.2f mm with %s (fuses below %.2f)"
                                 % (clear, who, FUSE_BAND_MM))

            folder = os.path.join(a.out, name, "Centrelines_comb")
            os.makedirs(folder, exist_ok=True)
            for o in others:
                if o == RVA_FILE and rva is not None:
                    write_curve(rva_tmpl, os.path.join(folder, o), rva[0], rva[1])
                else:
                    shutil.copy2(os.path.join(a.host, o), os.path.join(folder, o))
            write_curve(tmpl, os.path.join(folder, RCCA_FILE), rp, rr)
            write_curve(tmpl, os.path.join(folder, ECA_NAME), ep, er)
            with open(os.path.join(a.out, name, "provenance.json"), "w", encoding="utf-8") as fh:
                diag["eca_mm"] = float(arclength(ep)[-1])   # after any trim
                # v3: which real surfaces these sections came from. The lower's
                # lumen STL sits beside its centerline .vtp in the database and
                # shares its frame; the siphon's surface is the label surface
                # stage A wrote, mirrored exactly as its centerline was.
                lx, sx = diag["xform"]["lower"], diag["xform"]["siphon"]
                lx.update(kind="zenodo", mirror=[1.0, 1.0, 1.0],
                          surface=rec["path"].replace("_lumen_centerlines.vtp", "_lumen.stl"),
                          ica_real_mm=float(rec.get("ica_mm", 0.0)),
                          extend_mm=float(rec.get("extend_mm", 0.0)))
                src = sinfo["src"]
                left = "centerlines_left" in src.replace("\\", "/")
                stem0 = os.path.basename(src).replace("_ica.json", "")
                sx.update(kind="topbrain", centerline_src=src,
                          mirror=[-1.0, 1.0, 1.0] if sinfo["mirror"] else [1.0, 1.0, 1.0],
                          surface=os.path.join(os.path.dirname(os.path.dirname(src)),
                                               "surfaces_left" if left else "surfaces",
                                               stem0 + ("_lICA.vtp" if left else "_rICA.vtp")))
                # `note` records every repair applied -- an ECA trim, an RVA
                # shortening or deflection. It was printed to the console and
                # then dropped, so nothing on disk said which anatomies had a
                # deformed host vessel, and no downstream filter could tell a
                # pristine anatomy from a repaired one.
                json.dump({"lower": pr["lower"], "siphon": pr["siphon"],
                           "mismatch_mm": pr["mismatch_mm"],
                           "clearance_mm": float(clear),
                           "repairs": note or None, **diag}, fh, indent=1)
            ok += 1
            print("%-34s %8.1f %8.1f %7.1f %+7.2f %s"
                  % (name[:34], diag["total_mm"], diag["host_cut_mm"],
                     diag["max_kink"], clear, note))
        except Exception as e:                                # noqa: BLE001
            fail.append((name, str(e)[:60]))
            print("%-34s %s" % (name[:34], "FAILED: " + str(e)[:50]))

    print("\nbuilt %d, failed %d" % (ok, len(fail)))
    for n, why in fail[:20]:
        print("   %-40s %s" % (n[:40], why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
