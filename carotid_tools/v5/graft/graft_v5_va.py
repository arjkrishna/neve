#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\graft_v5_va.py (= /sp/v5_va/graft_v5_va.py in the build containers)
"""v5 PROTOTYPE: a real cervical vertebral artery from the subclavian origin, spliced to v4's TopBrain distal VA.

v4 (carotid_tools/graft_v4.py) replaced each host VA only ABOVE a cut (median 169 mm along a 208-242 mm
host VA) with a TopBrain MR label piece (18-93 mm). Everything below -- the origin at the subclavian, V1, V2
and most of V3 -- is still the host's smooth tube (union > 0.3 mm: 0.0003). v5 replaces that stretch too.

STRATEGY (B, "cervical donor + v4's distal piece, two seams"; argued in notes/graft.md):

    host subclavian (kept byte-for-byte) --[ORIGIN SEAM]--> donor VA, origin -> splice
        --[SPLICE SEAM: v4's repairs, TopBrain piece rigidly fitted]--> the SAME TopBrain distal VA piece v4 used -> blind end

  * The donor is VMR 0248's CT segmentation (ct_seg4, stage-A surface, VMTK MISR centerline) -- the only
    whole-length cervical VA with a real (CT-resolution) wall available. Its upper V3 / V4 are smooth at CT
    resolution, so the distal VA stays TopBrain's (strategy A, the whole VMR VA, would trade TopBrain's
    0.08 / 0.23 distal detail for VMR's 0.00 / 0.01).
  * ORIGIN SEAM. The donor carries its own branch point: the foot of its ostium on its own subclavian axis
    (O_v). O_v is pinned onto the host branch point O_h (RVA: where branch (17) leaves it, s = 26.8 mm;
    LVA: its first point, where (18) meets (19)). The rotation is a weighted Kabsch fit of the donor VA onto
    the v4 VA's corridor (21 points at matched normalised arclength, origin weight 10) -- the corridor v3 / v4
    already cleared against both carotid routes -- not a parent-tangent frame match: the subclavian bends
    hard at the VA origin (VMR's right subclavian rises at 63 deg there, the host's runs at 20 deg), and on
    all 223 a parent-frame match (--pose-mode parent) passes every clearance gate while sending the VAs out of
    the neck (tips 134-195 mm apart; v4 35). The take-off angle that results is recorded. A straight stub joins O_h to the placed ostium inside the parent, its calibre flaring from the
    parent's radius to the donor's (the host VA's own junction does the same: 5.9 -> 3.1 mm over 5 mm).
    The donor lumen must leave its parent within EXIT_MAX_MM and never re-enter it.
  * SPLICE SEAM. Length-matched as v4's cut (donor VA length - TopBrain piece length). The TopBrain piece is
    rigidly FITTED onto the donor's own distal remainder, start pinned (TB_PLACE "fit", see tb_rotation), not
    frame-matched on the local tangent as v4 does on the host: a real donor's V3 tangent swings through the C1
    loop and a frame-matched piece's blind end moved up to 104 mm (median 20). Moving the splice (v4's
    VA_CUT_STEPS) and advancing the TopBrain start (DONOR_TRIM_STEPS_MM) are the repairs, as in v4; every
    join <= 60 deg.
  * CALIBRE RAMP (RAMP_RULE "v4"). v4's own rule: the kept proximal vessel -- here the donor's last 25 mm, its
    CT-smooth upper V3 -- is eased to the TopBrain calibre. The alternative "narrow" (only ever narrow the
    fatter side) was tested and rejected: it puts the seam on TopBrain's C1-loop stretch, its richest wall.
  * POSE SEARCH = the first repair. Rigid rotations about O_h (3, 6, ..., 15 deg about the three axes), nearest
    first; each keeps the real surface exactly where its transform puts it (no tube fallback needed).
  * HANDEDNESS (VMR_HAND). VMR is a different patient from the TopBrain head. Default "native", like the
    Zenodo forks: VMR right VA -> RVA, left VA -> LVA, in every anatomy; the TopBrain distal piece keeps v4's
    handedness. Option "follow" (--vmr-hand follow): a mirrored anatomy takes VMR's opposite VA, x-mirrored,
    so its whole posterior circulation is one mirror image (v4's rule) and the set carries two cervical
    shapes per side; 223 / 223 pass the gates either way (the vmr agent's 0 / 223 was a rigid whole-VA
    probe with no pose / splice search).
  * GATES (v4's, NEW_CLEAR_MM = 0.6 mm): all new geometry from the parent exit on, against every other branch
    (incl. the stub and the ramps -- here all of it is new, so the ramp check is part of it), the parent
    after exit, the other VA, and itself (points > 12 mm apart in arclength); new joins <= 60 deg
    (join_kink); TopBrain blob cap and floor (VA_MIN_R); proper rotations only.
  * TAKE-OFF (second session). Recorded for every VA: donor native vs placed take-off at 5 / 10 mm (5 mm forward
    parent tangent on host and donor alike), elevation, parent-tangent mismatch, course RMS to the corridor.
    The corridor pose does NOT preserve it (VMR's subclavians rise 64 / 36 deg at the VA origin, the host's
    20 / 14). --pose-mode takeoff_frame (donor-pool plan Stage 6) / takeoff_roll preserve it and are dry-run
    options only: the first sends the VAs out of the neck (tips 164 mm apart), the second halves the error at
    twice the course RMS. notes/graft.md, "Take-off".
  * LEFT V1. VMR's left VA runs against a vein full of undiluted contrast for s ~ 10-40 mm (vmr agent): its
    wall there is not arterial. The left real section therefore starts LEFT_REAL_FROM_MM past the ostium; the
    stretch below is tube (declared VMTK radii).

Outputs, per anatomy, a COPY of the v4 anatomy (never anatomies_v4 itself) under --out with new RVA / LVA
centerline files and provenance.json extended by "v5_va" and xform["vaC_R"/"vaC_L"] (the donor sections) and
updated xform["va_R"/"va_L"] (the re-placed TopBrain pieces). Bake with /sp/v5_va/bake_v5.py.

    python3 /sp/v5_va/graft_v5_va.py --only NAME[,NAME] [--out ...]      # write prototypes
    python3 /sp/v5_va/graft_v5_va.py --dry [--workers 6]                 # centerline-level feasibility, all 223
Runs in eve-training-fixed. Licence of the donor: VMR research use, acknowledge (LICENSE_VMR_0248.txt).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/repo/carotid_tools")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/sp/v5_va")

import graft_three as G3  # noqa: E402
import graft_v4 as G4  # noqa: E402
from graft_siphon import (arclength, pair_gaps, prep_siphon, read_curve, resample,  # noqa: E402
                          smoothstep, tangent_at, unit, write_curve)

VERSION = "graft_v5_va prototype 0.1 (2026-10-01)"
NEW_CLEAR_MM = G4.NEW_CLEAR_MM            # 0.6
VA_MIN_R = G4.VA_MIN_R                    # 1.0
VA_BLEND_MM = G4.VA_BLEND_MM              # 25
VA_ANCHOR_MM = G4.VA_ANCHOR_MM            # 4
VA_BLOB_FACTOR, VA_BLOB_SPAN_MM = G4.VA_BLOB_FACTOR, G4.VA_BLOB_SPAN_MM
VA_CUT_STEPS = G4.VA_CUT_STEPS            # splice moves, nearest first
DONOR_TRIM_STEPS_MM = G4.DONOR_TRIM_STEPS_MM
JOIN_MAX_KINK = G4.JOIN_MAX_KINK          # 60
SELF_WINDOW_MM = G4.SELF_WINDOW_MM        # 12
VA_MIN_LEN_MM = G4.VA_MIN_LEN_MM          # 15
VA_TAIL_MM = G4.VA_TAIL_MM                # 10
EXIT_MAX_MM = 12.0          # donor lumen must clear its parent within this (host: 10.1 R / 7.6 L)
SPLICE_MIN_MM = 100.0       # keep at least V1, V2 and the C2 loop of the donor (C2 exit ~112 mm)
POSE_STEPS_DEG = (3.0, 6.0, 9.0, 12.0, 15.0)
LEFT_REAL_FROM_MM = 40.0    # VMR left VA: venous-contrast contact s ~ 10-40 mm (notes/vmr.md)
STEP = G3.RESAMPLE_MM       # 1 mm, as every v4 curve
# How the TopBrain piece is oriented at the splice (see tb_rotation): "frame" = v4's frame match on the local
# tangent; "fit" = Kabsch onto the donor's own distal remainder, start pinned; "fitslerp" = fit, slerped back
# towards the frame match only as far as the join needs to turn <= JOIN_MAX_KINK.
TB_PLACE = "fit"            # chosen on the 223-anatomy dry runs (notes/graft.md)
# Calibre ramp at the splice. "v4": v4's own rule (graft_va) -- the KEPT proximal vessel (here the donor's last
# 25 mm, its smooth upper V3: union > 0.3 mm 0.005 / 0.010 in VMR) is eased to the TopBrain calibre, widening or
# narrowing. "narrow": only ever narrow the fatter side -- tested first; it puts the 25 mm seam on the TopBrain
# piece's first 25 mm whenever TopBrain is fatter (190 / 146 of 223 VAs), which is its C1-loop stretch, the
# richest real wall it has (TopBrain 40-80 mm-from-distal median 0.23), and the measure then reads the calibre
# offset there as detail (0.25-0.72). See notes/graft.md.
RAMP_RULE = "v4"
# Dry-run variants for the design argument only (the prototypes use the defaults):
#   POSE_MODE "corridor" (default) = Kabsch onto the v4 VA corridor; "parent" = frame-match the donor's subclavian
#   tangent + superior onto the host subclavian at the branch point (v4's seam rule applied to the parent).
#   VMR_HAND "native" (default) = VMR never mirrored; "follow" = a mirrored anatomy takes VMR's opposite VA, x-mirrored.
POSE_MODE = "corridor"
VMR_HAND = "native"
# Take-off (added after the donor-pool plan, Stage 6). Take-off at d mm = angle between the VA chord ostium -> the
# point d mm along it and the parent subclavian's distal tangent, a TAKEOFF_TANGENT_MM forward chord from the branch
# point -- the host's t_sub (5 points of a 1 mm curve) and the donor's measured identically. Recorded in every mode.
#   POSE_MODE "takeoff_frame": the plan's proposal -- the donor's own subclavian tangent frame-matched onto the host's
#     (minimal rotation), roll about it taken from the v4 corridor (best weighted fit); preserves the take-off exactly.
#   POSE_MODE "takeoff_roll": the corridor fit, then rolled about its own origin -> distal-end chord (which keeps the
#     far end where the corridor fit put it) to the smallest roll that restores the donor's take-off.
# Both gate |delta take-off at 10 mm| <= TAKEOFF_TOL_DEG on every pose candidate (the pose search perturbs it).
TAKEOFF_TANGENT_MM = 5.0
TAKEOFF_TOL_DEG = 15.0
TAKEOFF_ROLL_AIM_DEG = 3.0

RVA, LVA = G4.RVA, G4.LVA
SUB_R = "Centerline curve (17).mrk.json"      # right subclavian, continuing from the RVA at its VA origin
SUB_L_IN = "Centerline curve (18).mrk.json"   # left subclavian from the arch, ends at the LVA origin
SUB_L_OUT = "Centerline curve (19).mrk.json"  # left subclavian distal, starts at the LVA origin
IDENT = [1.0, 1.0, 1.0]

VMR_ROOT = "/repo/carotid_data/va_sources/derived/vmr0248"
DONORS = {
    # side -> (centerline json with points/radii (VMTK MISR), closed surface)
    "vmr_ct": {"right": ("ct_seg4/rva_stageA_vmtk.json", "ct_seg4/rva_stageA.vtp"),
               "left": ("ct_seg4/lva_stageA_vmtk.json", "ct_seg4/lva_stageA.vtp"),
               "kind": "VMR 0248 CT segmentation ct_seg4, stage-A surface (marching cubes + windowed sinc 20/0.1), "
                       "VMTK MISR centerline", "left_real_from_mm": LEFT_REAL_FROM_MM},
}


class Unresolvable(Exception):
    pass


# ------------------------------------------------------------------ donor ----
def vmr_foot_points():
    """VMR branch point per side: the foot of the VA ostium (first VMTK point) on the expert subclavian path,
    and the subclavian distal tangent there (RAS mm)."""
    from sources import load_vmr0248
    v = load_vmr0248(with_ct=False, with_model=False, with_walls=False)
    out = {}
    for side in ("right", "left"):
        sp = np.asarray(v["vessels"][side + "_subclavian"]["points"], float)
        sr = np.asarray(v["vessels"][side + "_subclavian"]["radius_mm"], float)
        out[side] = (sp, sr)
    return out


def load_donor(donor, side, sub):
    """(points, radii, meta) of the donor VA, ostium -> distal, 1 mm, seed jogs trimmed, with O_v."""
    cl, sf = DONORS[donor][side]
    j = json.load(open(os.path.join(VMR_ROOT, cl)))
    p, r = np.asarray(j["points"], float), np.asarray(j["radii"], float)
    p, r = prep_siphon(p, r, STEP)                        # dedup, 1 mm, VMTK seed jogs (v4's rule)
    sp, sr = sub
    d = np.linalg.norm(sp - p[0], axis=1)
    k = int(np.argmin(d))
    # foot point on the subclavian polyline (project onto the two adjacent segments)
    best = (np.inf, sp[k])
    for a, b in ((max(k - 1, 0), k), (k, min(k + 1, len(sp) - 1))):
        if a == b:
            continue
        ab = sp[b] - sp[a]
        t = float(np.clip((p[0] - sp[a]) @ ab / (ab @ ab), 0.0, 1.0))
        q = sp[a] + t * ab
        if np.linalg.norm(q - p[0]) < best[0]:
            best = (float(np.linalg.norm(q - p[0])), q)
    O_v = best[1]
    t_sub = unit(sp[min(k + 5, len(sp) - 1)] - sp[max(k - 5, 0)])
    t_fwd = forward_chord(sp, O_v, TAKEOFF_TANGENT_MM)
    meta = {"donor": donor, "side": side, "centerline_src": os.path.join(VMR_ROOT, cl),
            "surface": os.path.join(VMR_ROOT, sf), "kind": DONORS[donor]["kind"],
            "len_mm": float(arclength(p)[-1]), "ostium_to_axis_mm": best[0], "sub_r_at_origin": float(sr[k]),
            "O_v": O_v.tolist(), "t_sub": t_sub.tolist(), "t_sub_fwd": t_fwd.tolist(),
            "takeoff_native": {"5": takeoff_of(p, t_fwd, 5.0), "10": takeoff_of(p, t_fwd, 10.0),
                               "10_central_tangent": takeoff_of(p, t_sub, 10.0)},
            "real_from_mm": float(DONORS[donor].get("left_real_from_mm", 0.0)) if side == "left" else 0.0}
    return p, r, meta


def forward_chord(sp, O, L):
    """Unit chord from O (on the polyline sp, index increasing distally) to the point L mm further along sp."""
    sp = np.asarray(sp, float)
    s = arclength(sp)
    k = int(np.argmin(np.linalg.norm(sp - O, axis=1)))
    # arclength of O's foot: project onto the neighbouring segments
    s0, best = s[k], np.inf
    for a, b in ((max(k - 1, 0), k), (k, min(k + 1, len(sp) - 1))):
        if a == b:
            continue
        ab = sp[b] - sp[a]
        t = float(np.clip((O - sp[a]) @ ab / (ab @ ab), 0.0, 1.0))
        d = float(np.linalg.norm(sp[a] + t * ab - O))
        if d < best:
            best, s0 = d, s[a] + t * (s[b] - s[a])
    q = np.array([np.interp(s0 + L, s, sp[:, i]) for i in range(3)])
    return unit(q - np.asarray(O, float))


def d_chord(p, d):
    """Chord vector p[0] -> the first point at >= d mm along p (va_metrics' take-off chord)."""
    s = arclength(p)
    return p[min(int(np.searchsorted(s, d)), len(p) - 1)] - p[0]


def takeoff_of(p, t_parent, d):
    return angle(d_chord(p, d), t_parent)


# --------------------------------------------------------------- geometry ----
def kabsch(A, B, w):
    w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def at_frac(P, q):
    s = arclength(P)
    return np.stack([np.interp(q * s[-1], s, P[:, i]) for i in range(3)], 1)


def rot(axis, deg):
    a = unit(np.asarray(axis, float))
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def base_pose(ext, corridor):
    """Proper rotation of the donor (O_v + VA) onto the corridor, origin-weighted, O_v -> corridor[0]."""
    q = np.linspace(0.0, 1.0, 21)
    A, B = at_frac(ext, q), at_frac(corridor, q)
    w = np.ones(len(q))
    w[0] = 10.0
    A = A - ext[0]
    B = B - corridor[0]
    return kabsch(A, B, w)


def corridor_misfit(R, ext, corridor, weighted=True):
    """RMS distance (mm) of the rotated donor from the corridor at 21 matched normalised-arclength points, origins
    pinned together; weighted = base_pose's objective (origin weight 10), else the plain course RMS."""
    q = np.linspace(0.0, 1.0, 21)
    A, B = at_frac(ext, q) - ext[0], at_frac(corridor, q) - corridor[0]
    w = np.ones(len(q))
    if weighted:
        w[0] = 10.0
    return float(np.sqrt((w * ((A @ np.asarray(R).T - B) ** 2).sum(1)).sum() / w.sum()))


def min_rotation(a, b):
    """Smallest proper rotation taking direction a onto direction b."""
    a, b = unit(np.asarray(a, float)), unit(np.asarray(b, float))
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        return rot(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]), 180.0)
    return rot(v, float(np.degrees(np.arctan2(np.linalg.norm(v), c))))


def takeoff_pose(mode, ext, corridor, d10, t_d, t_h, native):
    """Base rotation for the take-off-preserving pose modes (see TAKEOFF_* above). Returns (R, record)."""
    phis = np.arange(-180.0, 180.0, 1.0)
    if mode == "takeoff_frame":
        Ra = min_rotation(t_d, t_h)
        f = min(phis, key=lambda x: corridor_misfit(rot(t_h, x) @ Ra, ext, corridor))
        R = rot(t_h, f) @ Ra
        return R, {"mode": mode, "roll_deg": float(f), "about": "host subclavian distal tangent",
                   "residual_deg": abs(angle(R @ d10, t_h) - native)}
    R0 = base_pose(ext, corridor)
    a = unit(R0 @ (ext[-1] - ext[0]))
    dl = np.array([abs(angle(rot(a, f) @ R0 @ d10, t_h) - native) for f in phis])
    i = None
    for tol in (TAKEOFF_ROLL_AIM_DEG, TAKEOFF_TOL_DEG):
        ok = np.nonzero(dl <= tol)[0]
        if len(ok):
            i = int(ok[np.argmin(np.abs(phis[ok]))])
            break
    if i is None:
        i = int(np.argmin(dl))
    return rot(a, phis[i]) @ R0, {"mode": mode, "roll_deg": float(phis[i]),
                                  "about": "corridor-fit chord origin -> donor distal end", "residual_deg": float(dl[i]),
                                  "corridor_fit_delta_deg": float(angle(R0 @ d10, t_h) - native)}


def poses(R0):
    yield 0.0, None, R0
    for d in POSE_STEPS_DEG:
        for ax_name, ax in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
            for sgn in (1.0, -1.0):
                yield sgn * d, ax_name, rot(ax, sgn * d) @ R0


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


# ------------------------------------------------------------- clearance ----
class Neighbours:
    """Per-branch KD-trees of every branch but the one being grafted."""

    def __init__(self, items):
        self.items = []
        for name, p, r in items:
            p, r = np.asarray(p, float), np.asarray(r, float)
            if len(p) < 2:
                continue
            self.items.append((name, p, r, cKDTree(p), float(r.max())))

    def profile(self, q, qr, k=8):
        """Per point: (smallest gap, branch name) over all branches (k nearest centres per branch)."""
        best = np.full(len(q), np.inf)
        who = np.full(len(q), "", dtype=object)
        for name, p, r, tree, rmax in self.items:
            kk = min(k, len(p))
            d, i = tree.query(q, k=kk)
            if kk == 1:
                d, i = d[:, None], i[:, None]
            g = (d - qr[:, None] - r[i]).min(axis=1)
            m = g < best
            best[m] = g[m]
            who[m] = name
        return best, who


def self_gaps(p, r, from_idx, window=SELF_WINDOW_MM):
    s = arclength(p)
    q = np.arange(from_idx, len(p))
    if not len(q):
        return np.inf, None
    g = pair_gaps(p[q], r[q], p, r)
    far = np.abs(s[q][:, None] - s[None, :]) > window
    g = np.where(far, g, np.inf)
    k = np.unravel_index(int(np.argmin(g)), g.shape)
    return float(g[k]), [float(s[q[k[0]]]), float(s[k[1]])]


def gate_va(p, r, i_o, parents, others):
    """v4's gates on a whole new VA polyline p (branch arclength), origin at index i_o.

    parents: Neighbours of the parent subclavian (+ the branch's own kept subclavian stretch);
    others: Neighbours of every other branch. Returns dict(ok, clear, who, exit_mm, ...)."""
    s = arclength(p)
    q, qr = p[i_o:], r[i_o:]
    sq = s[i_o:] - s[i_o]
    gp, _ = parents.profile(q, qr)
    clear_par = np.nonzero(gp >= NEW_CLEAR_MM)[0]
    if not len(clear_par):
        return {"ok": False, "why": "never leaves its parent"}
    e = int(clear_par[0])
    exit_mm = float(sq[e])
    reentry = bool((gp[e:] < NEW_CLEAR_MM).any())
    go, gw = others.profile(q, qr)
    k = int(np.argmin(go))
    sc, spair = self_gaps(p, r, i_o + e)
    res = {"exit_mm": exit_mm, "parent_reentry": reentry,
           "parent_min_after_exit_mm": float(gp[e:].min()),
           "clear_others_mm": float(go[k]), "clear_others_who": str(gw[k]), "clear_others_at_mm": float(sq[k]),
           "self_mm": sc, "self_pair_mm": spair}
    worst = min((float(go[k]), str(gw[k])), (sc, "itself"),
                (float(gp[e:].min()) if not reentry else -1.0, "parent (after exit)" if not reentry else "parent (re-entry)"), key=lambda t: t[0])
    res["clear"], res["who"] = worst
    res["ok"] = bool(exit_mm <= EXIT_MAX_MM and not reentry and worst[0] >= NEW_CLEAR_MM)
    if exit_mm > EXIT_MAX_MM:
        res["why"] = "leaves its parent only at %.1f mm" % exit_mm
    elif not res["ok"]:
        res["why"] = "clearance %.2f mm with %s" % worst
    return res


# ------------------------------------------------------------------ joins ----
def stub(O_h, P0, r_parent, r0):
    L = float(np.linalg.norm(P0 - O_h))
    n = max(int(np.ceil(L / STEP)), 1)
    t = np.linspace(0.0, 1.0, n + 1)
    pts = O_h + t[:, None] * (P0 - O_h)
    w = smoothstep(t)
    return pts, r_parent + (r0 - r_parent) * w, L


def graft_splice(keep_p, keep_r, donor_p, donor_r, target=None):
    """graft_v4.graft_va with v5's ramp rule: the TopBrain piece frame-matched onto the end of `keep`
    (tangent + superior, graft_three.place), its first 4 mm flattened and its first 10 mm capped at
    1.3 x its median (v4's blob rule), floored at VA_MIN_R -- and the calibre ramp easing the FATTER side down
    to the thinner one over VA_BLEND_MM, so neither real section's declared tube is ever widened."""
    keep_p, keep_r = keep_p.copy(), np.asarray(keep_r, float).copy()
    hs = arclength(keep_p)
    k = len(keep_p) - 1
    ds = arclength(donor_p)
    donor_r = np.maximum(np.asarray(donor_r, float), VA_MIN_R)
    cap = VA_BLOB_FACTOR * float(np.median(donor_r))
    r_join = min(float(np.interp(min(VA_ANCHOR_MM, ds[-1]), ds, donor_r)), cap)
    donor_r = np.where(ds < VA_ANCHOR_MM, r_join, donor_r)
    donor_r = np.where(ds < VA_BLOB_SPAN_MM, np.minimum(donor_r, cap), donor_r)
    r_keep = float(np.interp(max(hs[k] - VA_ANCHOR_MM, 0.0), hs, keep_r))
    if RAMP_RULE == "v4":
        blend = min(VA_BLEND_MM, hs[k] - 10.0)
        keep_r = G3.ramp_radius(hs, keep_r, hs[k], r_join, blend)
        ramp = {"side": "donor", "rule": "v4", "from_r": r_keep, "to_r": r_join, "span_mm": [-blend, 0.0]}
    elif r_join > r_keep:
        # TopBrain fatter: its first VA_BLEND_MM rise from the donor's calibre to its own
        w = smoothstep(ds / VA_BLEND_MM)
        donor_r = r_keep + (donor_r - r_keep) * w
        keep_r[-1] = r_keep
        ramp = {"side": "topbrain", "rule": "narrow", "from_r": r_keep, "to_r": r_join, "span_mm": [0.0, VA_BLEND_MM]}
    else:
        blend = min(VA_BLEND_MM, hs[k] - 10.0)
        keep_r = G3.ramp_radius(hs, keep_r, hs[k], r_join, blend)
        ramp = {"side": "donor", "rule": "narrow", "from_r": r_keep, "to_r": r_join, "span_mm": [-blend, 0.0]}
    R, t_fit = tb_rotation(donor_p, keep_p, target)
    origin = donor_p[0]
    moved = (donor_p - origin) @ R.T + keep_p[-1]
    p = np.vstack([keep_p, moved[1:]])
    r = np.concatenate([keep_r, donor_r[1:]])
    xf = {"R": R.tolist(), "origin": np.asarray(origin).tolist(), "anchor": keep_p[-1].tolist(),
          "placement": TB_PLACE, "fit_weight": t_fit}
    return p, r, float(hs[k]), xf, ramp


def slerp(Ra, Rb, t):
    from scipy.spatial.transform import Rotation, Slerp
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([t]).as_matrix()[0]


def tb_rotation(donor_p, keep_p, target):
    """Rotation of the TopBrain piece about its start, which lands on the end of `keep`.

    v4 frame-matches the piece's first 5 mm chord to the host tangent at the cut (graft_three.place). On a
    host tube that tangent is the vessel's course. On a REAL donor spliced in V3 it is not: the splice falls in
    or next to the C1 loop (V3 DM 2.2 in VMR 0248), where the local tangent swings through 180 deg in 30 mm,
    and a frame-matched 50-90 mm piece points wherever that swing points -- its blind end moved a median
    20-26 mm (max 100) from v4's in the first dry run. "fit" instead puts the piece where the donor's own
    distal VA runs (Kabsch at 21 matched normalised-arclength points, start pinned), so its blind end sits by
    the donor's own vertebrobasilar confluence; "fitslerp" keeps as much of that as the 60 deg join rule allows.
    Returns (R, weight of the fit: 1 = pure fit, 0 = v4's frame match)."""
    k = len(keep_p) - 1
    R_frame = G3.frame_rotation(tangent_at(donor_p, 0, G3.SIPHON_SPAN), tangent_at(keep_p, k))
    if TB_PLACE == "frame" or target is None or len(target) < 5:
        return R_frame, 0.0
    q = np.linspace(0.0, 1.0, 21)
    A = at_frac(donor_p, q) - donor_p[0]
    B = at_frac(target, q) - target[0]
    R_fit = kabsch(A, B, np.ones(len(q)))
    if TB_PLACE == "fit":
        return R_fit, 1.0
    hs = arclength(keep_p)
    for t in np.arange(1.0, -1e-9, -0.1):
        R = slerp(R_frame, R_fit, t) if t < 1.0 else R_fit
        moved = (donor_p - donor_p[0]) @ R.T + keep_p[-1]
        if G4.join_kink(np.vstack([keep_p, moved[1:]]), hs[k]) <= JOIN_MAX_KINK:
            return R, float(round(t, 2))
    return R_frame, 0.0


# ------------------------------------------------------------------ build ----
def side_setup(side, br, donor_p, donor_r, dmeta):
    """Host branch point, kept subclavian stretch and the parent set for one side."""
    if side == "right":
        _, vp, vr = br[RVA]
        _, sp, sr = br[SUB_R]
        d = np.linalg.norm(vp - sp[0], axis=1)
        i0 = int(np.argmin(d))
        if d[i0] > 0.5:          # v4 resampled the whole RVA at 1 mm: its subclavian stretch sits <= 0.34 mm off
            raise Unresolvable("RVA does not pass through the (17) branch point (%.2f mm)" % d[i0])
        sub_p, sub_r = vp[:i0 + 1].copy(), vr[:i0 + 1].copy()
        parents = [("(17)", sp, sr), ("RVA subclavian", sub_p, sub_r)]
        t_sub = unit(sp[min(5, len(sp) - 1)] - sp[0])
    else:
        _, vp, vr = br[LVA]
        _, sp_in, sr_in = br[SUB_L_IN]
        _, sp, sr = br[SUB_L_OUT]
        if np.linalg.norm(vp[0] - sp[0]) > 0.05:
            raise Unresolvable("LVA does not start at the (19) branch point")
        sub_p, sub_r = vp[:1].copy(), vr[:1].copy()
        parents = [("(18)", sp_in, sr_in), ("(19)", sp, sr)]
        t_sub = unit(sp[min(5, len(sp) - 1)] - sp[0])
    return {"O_h": sub_p[-1].copy(), "sub_p": sub_p, "sub_r": sub_r, "r_parent": float(sub_r[-1]),
            "parents": parents, "t_sub": t_sub, "v4_va": (vp, vr), "i_o": len(sub_p) - 1}


def others_for(side, fixed, other_va=None):
    """Every other branch but this side's parent subclavian (gated by the exit rule instead)."""
    excl = {"cc (17)"} if side == "right" else {"cc (18)", "cc (19)"}
    items = [n for n in fixed if n[0] not in excl]
    if other_va is not None:
        items.append(other_va)
    return items


def s_of(p, q):
    """Arclength of the curve point nearest q."""
    return float(arclength(p)[int(np.argmin(np.linalg.norm(p - np.asarray(q), axis=1)))])


def compose(st, R, donor_p, donor_r, dmeta, splice_mm, tb_p, tb_r):
    """kept subclavian + stub + placed donor [0, splice] + TopBrain piece. Returns the curve and records."""
    O_v = np.asarray(dmeta["O_v"])
    placed = (donor_p - O_v) @ R.T + st["O_h"]
    ds = arclength(donor_p)
    k = int(np.searchsorted(ds, splice_mm))
    k = int(np.clip(k, 5, len(donor_p) - 1))
    sp_, sr_, L_stub = stub(st["O_h"], placed[0], st["r_parent"], float(max(donor_r[0], VA_MIN_R)))
    dr = np.maximum(donor_r[:k + 1], VA_MIN_R)
    keep_p = np.vstack([sp_, placed[1:k + 1]])
    keep_r = np.concatenate([sr_, dr[1:]])
    p, r, cut_at, xf, ramp = graft_splice(keep_p, keep_r, tb_p, tb_r, target=placed[k:])
    p, r = resample(p, r, STEP)
    # the kept host subclavian stretch stays byte-identical (RVA only)
    full_p = np.vstack([st["sub_p"][:-1], p])
    full_r = np.concatenate([st["sub_r"][:-1], r])
    # seams located on the curve as written (the 1 mm resample shortens tight bends slightly)
    rec = {"stub_mm": L_stub, "ostium_at_mm": s_of(full_p, placed[0]), "splice_on_donor_mm": float(ds[k]),
           "cut_at_mm": s_of(full_p, xf["anchor"]), "cut_at_mm_unresampled": float(arclength(st["sub_p"])[-1]) + cut_at,
           "ramp": ramp, "tb_xform": xf, "R": R}
    return full_p, full_r, rec


def build(name, a, foot, donors, dry=False):
    v4 = os.path.join(a.v4, name)
    prov4 = json.load(open(os.path.join(v4, "provenance.json"), encoding="utf-8"))
    mirrored = prov4["handedness"] == "mirrored"
    pid = prov4["patient"]
    cdir = os.path.join(v4, "Centrelines_comb")
    br = {}
    for f in sorted(os.listdir(cdir)):
        if f.endswith(".json"):
            br[f] = read_curve(os.path.join(cdir, f))
    t0 = time.time()
    out = {"name": name, "handedness": prov4["handedness"], "patient": pid, "sides": {}}
    fixed = [(G4.short(f), p, r) for f, (_, p, r) in br.items() if f not in (RVA, LVA) and r is not None]
    placed = {}
    for side, fname in (("right", RVA), ("left", LVA)):
        dp, dr, dmeta = donors["mirrored" if (VMR_HAND == "follow" and mirrored) else "native"][side]
        st = side_setup(side, br, dp, dr, dmeta)
        par = Neighbours(st["parents"])
        # the right VA is placed first, against everything but both VAs (the v4 LVA is replaced next);
        # the left one then against everything, the placed right VA included
        other_va = (G4.short(RVA), *placed["right"][:2]) if side == "left" else None
        others = Neighbours(others_for(side, fixed, other_va))
        # TopBrain piece exactly as v4 sourced it (same patient side, same mirror)
        tbp0, tbr0, tmeta = G4.tb_part("va", side, pid, mirrored, "/repo")
        if tbp0 is None:
            raise Unresolvable("VA %s: no TopBrain piece (%s)" % (side, tmeta.get("why")))
        tds0 = arclength(tbp0)
        vp4, vr4 = st["v4_va"]
        corridor = vp4[st["i_o"]:]
        ext = np.vstack([np.asarray(dmeta["O_v"])[None, :], dp])
        d10 = d_chord(dp, 10.0)
        t_d = np.asarray(dmeta["t_sub_fwd"])
        native10 = float(dmeta["takeoff_native"]["10"])
        pose_info = {"mode": POSE_MODE}
        if POSE_MODE == "corridor":
            R0 = base_pose(ext, corridor)
        elif POSE_MODE == "parent":
            R0 = G3.frame_rotation(np.asarray(dmeta["t_sub"]), st["t_sub"])
        else:
            R0, pose_info = takeoff_pose(POSE_MODE, ext, corridor, d10, t_d, st["t_sub"], native10)
        gate_takeoff = POSE_MODE.startswith("takeoff")
        L_d = float(arclength(dp)[-1])
        tried, ok = [], None
        n_eval = 0
        for ang, ax, R in poses(R0):
            if gate_takeoff and abs(angle(R @ d10, st["t_sub"]) - native10) > TAKEOFF_TOL_DEG:
                tried.append("pose %+.0f%s: take-off off by > %.0f deg" % (ang, ax or "", TAKEOFF_TOL_DEG))
                continue
            for dc in VA_CUT_STEPS:
                for tr in DONOR_TRIM_STEPS_MM:
                    tb_p, tb_r = tbp0[tds0 >= tr], tbr0[tds0 >= tr]
                    Ltb = float(arclength(tb_p)[-1])
                    if Ltb < VA_MIN_LEN_MM:
                        break
                    splice = float(np.clip(L_d - Ltb, SPLICE_MIN_MM, L_d - VA_TAIL_MM)) + dc
                    if splice < SPLICE_MIN_MM or splice > L_d - VA_TAIL_MM:
                        break
                    p, r, rec = compose(st, R, dp, dr, dmeta, splice, tb_p, tb_r)
                    n_eval += 1
                    kink = G4.join_kink(p, rec["cut_at_mm"])
                    if kink > JOIN_MAX_KINK:
                        tried.append("pose %+.0f%s splice %+.0f trim %.0f: join %.0f deg" % (ang, ax or "", dc, tr, kink))
                        continue                       # advance the TopBrain start
                    g = gate_va(p, r, st["i_o"], par, others)
                    if g["ok"]:
                        ok = (p, r, rec, g, kink, ang, ax, dc, tr, tmeta, Ltb)
                    else:
                        tried.append("pose %+.0f%s splice %+.0f trim %.0f: %s" % (ang, ax or "", dc, tr, g.get("why")))
                    break                              # the clearance, not the join, decided: next splice
                if ok:
                    break
            if ok:
                break
        if ok is None:
            raise Unresolvable("VA %s: no pose / splice passes (%d tried; first: %s)" % (side, n_eval, "; ".join(tried[:4])))
        p, r, rec, g, kink, ang, ax, dc, tr, tmeta, Ltb = ok
        placed[side] = (p, r, rec, g, kink, ang, ax, dc, tr, tmeta, Ltb, st, dmeta)
        # take-off: first 10 mm of the donor VA (from the ostium) against the parent's distal tangent
        s = arclength(p)
        io = int(np.searchsorted(s, rec["ostium_at_mm"]))
        i10 = int(np.searchsorted(s, rec["ostium_at_mm"] + 10.0))
        # strategy C would keep v4's TopBrain piece where v4 put it and make the donor reach its start:
        # how far is the base-pose donor from that point, and how much would the join turn there
        v4cut = float(prov4["vertebrals"][side]["cut_mm"])
        s4 = arclength(vp4)
        k4 = int(np.searchsorted(s4, v4cut))
        k4 = int(np.clip(k4, 1, len(vp4) - 2))
        placed0 = (dp - np.asarray(dmeta["O_v"])) @ R0.T + st["O_h"]
        dd = np.linalg.norm(placed0 - vp4[k4], axis=1)
        j = int(np.argmin(dd))
        c_gap = {"v4_cut_mm": v4cut, "nearest_donor_mm": float(dd[j]), "at_donor_s_mm": float(arclength(dp)[j]),
                 "tangent_mismatch_deg": angle(tangent_at(placed0, j), tangent_at(vp4, k4)),
                 "length_matched_point_gap_mm": float(np.linalg.norm(
                     placed0[int(np.searchsorted(arclength(dp), np.clip(L_d - float(arclength(tbp0)[-1]), 0, L_d)))] - vp4[k4]))}
        placed_full = (dp - np.asarray(dmeta["O_v"])) @ np.asarray(rec["R"]).T + st["O_h"]
        out["sides"][side] = {
            "strategy_C_gap": c_gap, "tip_moved_vs_v4_mm": float(np.linalg.norm(p[-1] - vp4[-1])),
            "tip_to_donor_confluence_mm": float(np.linalg.norm(p[-1] - placed_full[-1])),
            "tb_placement": rec["tb_xform"].get("placement"), "tb_fit_weight": rec["tb_xform"].get("fit_weight"),
            "pose_deg": ang, "pose_axis": ax, "splice_move_mm": dc, "tb_start_trim_mm": tr,
            "n_candidates_evaluated": n_eval, "first_rejections": tried[:6],
            "splice_on_donor_mm": rec["splice_on_donor_mm"], "cut_at_mm": rec["cut_at_mm"],
            "ostium_at_mm": rec["ostium_at_mm"], "stub_mm": rec["stub_mm"], "join_kink_deg": kink,
            "tb_len_mm": Ltb, "tb_patient_side": tmeta["patient_side"], "tb_mirror": tmeta["mirror"],
            "ramp": rec["ramp"], "gate": g, "total_mm": float(s[-1]),
            "takeoff_10mm_deg": angle(p[i10] - p[io], st["t_sub"]),
            "takeoff_5mm_deg": angle(p[int(np.searchsorted(s, rec["ostium_at_mm"] + 5.0))] - p[io], st["t_sub"]),
            "donor_takeoff_10mm_deg_native": native10,
            "donor_takeoff_5mm_deg_native": float(dmeta["takeoff_native"]["5"]),
            "takeoff_delta_10mm_deg": angle(p[i10] - p[io], st["t_sub"]) - native10,
            "takeoff_preserved": bool(abs(angle(p[i10] - p[io], st["t_sub"]) - native10) <= TAKEOFF_TOL_DEG),
            "parent_tangent_mismatch_deg": angle(np.asarray(rec["R"]) @ t_d, st["t_sub"]),
            "va_10mm_elevation_deg": 90.0 - angle(p[i10] - p[io], np.array([0.0, 0.0, 1.0])),
            "donor_va_10mm_elevation_deg_native": 90.0 - angle(d10, np.array([0.0, 0.0, 1.0])),
            "host_subclavian_elevation_deg": 90.0 - angle(st["t_sub"], np.array([0.0, 0.0, 1.0])),
            "donor_subclavian_elevation_deg_native": 90.0 - angle(t_d, np.array([0.0, 0.0, 1.0])),
            "course_rms_to_corridor_mm": corridor_misfit(np.asarray(rec["R"]), ext, corridor, weighted=False),
            "pose_mode": POSE_MODE, "pose_base": pose_info}
    out["seconds"] = round(time.time() - t0, 1)
    out["tip_to_tip_mm"] = float(np.linalg.norm(placed["right"][0][-1] - placed["left"][0][-1]))
    _, v4r, _ = br[RVA]
    _, v4l, _ = br[LVA]
    out["v4_tip_to_tip_mm"] = float(np.linalg.norm(v4r[-1] - v4l[-1]))
    # final clearances on everything as it will be written
    for side, fname in (("right", RVA), ("left", LVA)):
        p, r, rec, g, kink, ang, ax, dc, tr, tmeta, Ltb, st, dmeta = placed[side]
        oth = "left" if side == "right" else "right"
        par = Neighbours(st["parents"])
        other_items = others_for(side, fixed, (G4.short(LVA if side == "right" else RVA), *placed[oth][:2]))
        gf = gate_va(p, r, st["i_o"], par, Neighbours(other_items))
        # exact (all-pairs) recheck of the k-NN figure
        allp = np.vstack([x[1] for x in other_items]); allr = np.concatenate([x[2] for x in other_items])
        s = arclength(p)
        e = int(np.searchsorted(s - s[st["i_o"]], gf.get("exit_mm", 0.0)))
        exact = float(pair_gaps(p[e:], r[e:], allp, allr).min()) if e < len(p) else np.inf
        out["sides"][side]["final"] = {k_: gf.get(k_) for k_ in ("ok", "clear", "who", "exit_mm", "self_mm",
                                                                "clear_others_mm", "clear_others_who")}
        out["sides"][side]["final"]["exact_all_pairs_from_exit_mm"] = exact
    if dry:
        return out, placed, br, prov4
    write(name, a, out, placed, br, prov4)
    return out, placed, br, prov4


def write(name, a, out, placed, br, prov4):
    src = os.path.join(a.v4, name)
    dst = os.path.join(a.out, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "Centrelines_comb"))
    for f in os.listdir(os.path.join(src, "Centrelines_comb")):
        if f not in (RVA, LVA):
            shutil.copy2(os.path.join(src, "Centrelines_comb", f), os.path.join(dst, "Centrelines_comb", f))
    for side, fname in (("right", RVA), ("left", LVA)):
        p, r = placed[side][:2]
        write_curve(br[fname][0], os.path.join(dst, "Centrelines_comb", fname), p, r)
    prov = json.loads(json.dumps(prov4))
    xform = prov["xform"]
    v5 = {"version": VERSION, "strategy": "B: donor cervical VA (origin -> splice) + v4's TopBrain distal piece",
          "tb_placement": TB_PLACE, "ramp_rule": RAMP_RULE,
          "donor": "vmr0248 ct_seg4 stage-A (VMTK MISR)", "donor_mirror": IDENT,
          "donor_licence": "VMR research use; cite Wilson et al. 2013 (doi 10.1115/1.4025983), NLM R01LM013120",
          "new_graft_clearance_mm": NEW_CLEAR_MM, "exit_max_mm": EXIT_MAX_MM, "seconds": out["seconds"]}
    for side, key, ckey in (("right", "va_R", "vaC_R"), ("left", "va_L", "vaC_L")):
        p, r, rec, g, kink, ang, ax, dc, tr, tmeta, Ltb, st, dmeta = placed[side]
        old = xform.get(key, {})
        xform[key] = dict(rec["tb_xform"], kind="topbrain", centerline_src=tmeta["centerline_src"],
                          mirror=tmeta["mirror"], surface=tmeta["surface"], cut_mm=rec["cut_at_mm"],
                          donor_start_trim_mm=float(tr), v4_cut_mm=old.get("cut_mm"))
        lo = rec["ostium_at_mm"] + dmeta["real_from_mm"]
        xform[ckey] = {"R": np.asarray(rec["R"]).tolist(), "origin": list(dmeta["O_v"]), "anchor": st["O_h"].tolist(),
                       "mirror": dmeta.get("mirror", IDENT), "surface": dmeta["surface"], "centerline_src": dmeta["centerline_src"],
                       "kind": dmeta["kind"], "lo_mm": lo, "hi_mm": rec["cut_at_mm"], "branch": "RVA" if side == "right" else "LVA",
                       "real_from_mm_past_ostium": dmeta["real_from_mm"]}
        v5[side] = out["sides"][side]
    prov["v5_va"] = v5
    with open(os.path.join(dst, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    for f in ("LICENSE_VMR_0248.txt",):
        shutil.copy2(os.path.join(VMR_ROOT, f), os.path.join(dst, f))


def _init(a_):
    global A, FOOT, DON, TB_PLACE, RAMP_RULE, POSE_MODE, VMR_HAND
    A = a_
    TB_PLACE = a_.tb_place
    RAMP_RULE = a_.ramp
    POSE_MODE = a_.pose_mode
    VMR_HAND = a_.vmr_hand
    FOOT = vmr_foot_points()
    nat = {side: load_donor(A.donor, side, FOOT[side]) for side in ("right", "left")}
    mir = {}
    for side, other in (("right", "left"), ("left", "right")):
        p, r, m = nat[other]
        M = np.array([-1.0, 1.0, 1.0])
        m = dict(m, O_v=(np.asarray(m["O_v"]) * M).tolist(), t_sub=(np.asarray(m["t_sub"]) * M).tolist(),
                 t_sub_fwd=(np.asarray(m["t_sub_fwd"]) * M).tolist(),
                 mirror=M.tolist(), side=side + " (VMR %s VA, mirrored)" % other)
        mir[side] = (p * M, r, m)
    DON = {"native": nat, "mirrored": mir}


def _one(name):
    try:
        out = build(name, A, FOOT, DON, dry=A.dry)[0]
        return name, out, None
    except Unresolvable as e:
        return name, None, str(e)
    except Exception as e:                                          # noqa: BLE001
        import traceback
        return name, None, "ERROR %s: %s | %s" % (type(e).__name__, e, traceback.format_exc()[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default="/repo/carotid_data/anatomies_v4")
    ap.add_argument("--out", default="/repo/carotid_data/va_sources/v5_proto")
    ap.add_argument("--donor", default="vmr_ct", choices=sorted(DONORS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tb-place", default=TB_PLACE, choices=("frame", "fit", "fitslerp"))
    ap.add_argument("--ramp", default=RAMP_RULE, choices=("v4", "narrow"))
    ap.add_argument("--pose-mode", default=POSE_MODE, choices=("corridor", "parent", "takeoff_frame", "takeoff_roll"))
    ap.add_argument("--vmr-hand", default=VMR_HAND, choices=("native", "follow"))
    a = ap.parse_args()
    names = sorted(d for d in os.listdir(a.v4) if os.path.isdir(os.path.join(a.v4, d, "Centrelines_comb")))
    if a.only:
        want = a.only.split(",")
        names = [n for n in names if n in want]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init, initargs=(a,)) as pool:
            rows = pool.map(_one, names, chunksize=2)
    else:
        _init(a)
        rows = [_one(n) for n in names]
    ok = [r for r in rows if r[1] is not None]
    bad = [r for r in rows if r[1] is None]
    for n, o, e in rows:
        if o is None:
            print("DROP  %-44s %s" % (n[:44], e[:300]))
        else:
            sR, sL = o["sides"]["right"], o["sides"]["left"]
            print("OK    %-44s R pose %+3.0f%-1s spl %+3.0f clr %.2f (%s) exit %.1f to %.0f | L pose %+3.0f%-1s spl %+3.0f clr %.2f (%s) exit %.1f to %.0f" % (
                n[:44], sR["pose_deg"], sR["pose_axis"] or "", sR["splice_move_mm"], sR["final"]["clear"], sR["final"]["who"][-12:],
                sR["final"]["exit_mm"], sR["takeoff_10mm_deg"], sL["pose_deg"], sL["pose_axis"] or "", sL["splice_move_mm"],
                sL["final"]["clear"], sL["final"]["who"][-12:], sL["final"]["exit_mm"], sL["takeoff_10mm_deg"]), flush=True)
    print("\nbuilt %d, dropped %d, %.0f s" % (len(ok), len(bad), time.time() - t0))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"version": VERSION, "dry": a.dry, "built": len(ok), "dropped": len(bad),
                       "rows": {n: o for n, o, e in rows if o is not None},
                       "drops": {n: e for n, o, e in rows if o is None}}, f, indent=1,
                      default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, (np.floating,)) else str(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
