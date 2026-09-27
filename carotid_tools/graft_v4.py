#!/usr/bin/env python3
"""v4: complete the three-source anatomies -- left carotid, both vertebrals, ECA extensions.

Set B (graft_three.py) grafts ONE route: host arch -> a real carotid fork -> a
TopBrain siphon, on the right. Every other neck vessel is still the shipped
host's: the LCCA runs unbroken to the left intracranial ICA with no fork, and
both vertebrals are the host's, meeting at its basilar confluence. v4 finishes
the neck. Starting from each v3 anatomy, with its right route kept byte-for-byte:

  left carotid  host LCCA -> the OTHER carotid of the same fork patient -> the
                same TopBrain patient's other ICA siphon. Built by graft_three's
                own compose() with the left-side constants below, so both sides
                are composed by one method.
  vertebrals    host RVA / LVA (proximal) -> the same TopBrain patient's VA
                distal segment, frame-matched at a cut exactly as a siphon is.
                Independent segments with blind ends: no basilar, no circle of
                Willis, so the host's vertebrobasilar ring is gone by design.
  ECA extension a TopBrain ECA piece frame-matched onto each fork ECA's tip,
                only where it is convenient (see ECA_EXT_*); otherwise the fork
                ECA stands alone.

HANDEDNESS. A v3 anatomy whose siphon id ends in _L carries the patient's LEFT
ICA, x-mirrored, on its right route. Such an anatomy is "mirrored", and every
TopBrain part of it follows suit: its left siphon is the patient's RIGHT ICA
mirrored, its right vertebral the patient's LEFT VA mirrored, and so on, so the
donor head is one consistent mirror image rather than a mix. A "native" anatomy
takes every part from its own side unmirrored. Frame matching is a proper
rotation and cannot change handedness, so this is the only place chirality is
decided. Lowers are never mirrored (graft_three convention; fork handedness is
not a side property).

    python3 carotid_tools/graft_v4.py --v3 carotid_data/anatomies_v3 \
        --out carotid_data/anatomies_v4 [--only NAME,NAME | --shard i/n]
"""
import argparse
import contextlib
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graft_three as G3
from graft_siphon import (anchor_trim, arclength, kink_deg,
                          min_clearance, pair_gaps, prep_siphon, read_curve,
                          resample, rva_deflect,
                          tangent_at, write_curve)

RCCA = "Centerline curve - RCCA.mrk.json"
RECA = "Centerline curve - RECA.mrk.json"
RVA = "Centerline curve - RVA.mrk.json"
LCCA = "Centerline curve - LCCA.mrk.json"
LECA = "Centerline curve - LECA.mrk.json"
LVA = "Centerline curve - LVA.mrk.json"

# ---- v2/v3 constants (BUILD_v2.json flags), applied to every new graft ------
V2 = dict(ROUTE_MIN_R=1.0, ECA_MESH_R_MM=1.0, DISTAL_TRIM_MM=0.0,
          FUSE_BAND_MM=0.35, SIPHON_MIN_R=1.0)
# Every clearance gate on NEW v4 geometry (left carotid, vertebral grafts, ECA
# extensions) uses a wider band than v2/v3's 0.35 mm. fuse_band_v2.py measured
# a 0.35 mm wall fusing in 3 of 6 grid alignments on the 0.45 mm SDF grid and
# 0.45 mm never; the first v4 bake confirmed it at the anatomy level -- grafts
# passed at 0.35-0.45 mm came out as tube-level handles (genus > 0 with no real
# surface at all). 0.6 mm leaves a grid step plus the radius ramps' slack. The
# v3 right side is copied untouched and keeps its own 0.35 mm history.
NEW_CLEAR_MM = 0.6
V4_NEW = dict(V2, FUSE_BAND_MM=NEW_CLEAR_MM)   # also what eca_reentry sees on new forks

# ---- left carotid -----------------------------------------------------------
# The right seam sits at 130 mm of RCCA arclength, level with z = 530.3 in the
# branch frame. The host LCCA reaches that same height at 157.6 mm and has
# 108.2 mm left above it, against 105.8 mm on the right: the two definitions a
# left seam could have (same level, same remaining length) agree, so the left
# siphon enters the skull where the right one does. Computed from the host at
# run time rather than hard-coded; LEFT_SEAM_FALLBACK only documents the value.
LEFT_SEAM_FALLBACK = 157.6
# The LCCA opens INSIDE the aortic arch at 14.68 mm radius and does not reach
# CCA calibre until ~24 mm (the RCCA's flare is over by 8 mm). The ramp must
# never reach into that flare, so the anchor sits at 30 mm, and the shortest
# ramp keeps graft_three's 5 mm margin above it.
BLEND_ANCHOR_L = 30.0
HOST_CUT_MIN_L = 35.0
# The host LCCA holds CCA calibre (radius 2.40 / 2.52 / 2.53 mm at 100 / 110 / 120 mm,
# falling to 1.50 by 130) to ~120 mm. 110 keeps the cut 10 mm inside that; 100 was
# the first choice and left one fork patient (m_024_right, CCA+ICA 44.5 mm) short.
# A left donor then needs cca+ica >= 47.6 mm, less the seam tolerance.
HOST_CUT_MAX_L = 110.0
MIN_TIP_D = 2.0            # match_sections' tip rule; recorded, NOT a gate here (see lower_records)

# ---- vertebrals ---------------------------------------------------------------
VA_MIN_R = 1.0             # same floor as the siphons (V2 SIPHON_MIN_R)
VA_MIN_LEN_MM = 15.0       # a shorter donor VA is a label fragment, not a vessel
VA_CUT_MIN_MM = 80.0       # keep the host's subclavian origin and cervical VA
VA_TAIL_MM = 10.0          # ...and never cut closer than this to its end
VA_BLEND_MM = 25.0
VA_MAX_SHORTEN_MM = 30.0
VA_CUT_STEPS = (0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25, -30, 30)   # mm, nearest first

# ---- ECA extension ------------------------------------------------------------
# "Convenient, without a long gap": the same (handedness-mapped) patient side,
# one connected label piece with a usable centerline of at least ECA_EXT_MIN_MM,
# and calibres that already agree at the join -- both measured ECA_EXT_ANCHOR_MM
# in from their cut faces, which droop (graft_three.CCA_ANCHOR_MM). A worse
# mismatch would need an invented taper, which is what the fork ECA alone avoids.
ECA_EXT_MIN_MM = 10.0
ECA_EXT_MAX_MISMATCH_MM = 1.0
ECA_EXT_ANCHOR_MM = 4.0
ECA_EXT_BLEND_MM = 10.0

# ---- shape gates on every new join (added after the v4 audit) -----------------
# vmtkCenterlines' AppendEndPoints seeds sit on the wall, 1-2 mm off the axis.
# prep_siphon trims the resulting jog only while a curve has more than 21
# points, so short TopBrain VA/ECA pieces kept it, and place() frame-matches
# the chord p0 -> p5 -- the jog landed at the join: 57 ECA extensions turned
# 60-98 deg there. Short pieces' proximal seeds are trimmed here, and every new join is held to
# the left route's own kink limit (--max-kink), measured the same way
# (kink_deg on the 1 mm resampled curve) within JOIN_KINK_SPAN_MM of the join.
SEED_TRIM_MAX_MM = 3.0     # per end
SEED_KEEP_MIN = 8          # points
SEED_KINK_DEG = 40.0       # prep_siphon's threshold
JOIN_MAX_KINK = 60.0
JOIN_KINK_SPAN_MM = 3.0
DONOR_TRIM_STEPS_MM = (0, 1, 2, 3, 4, 5)   # a donor that opens on a turn: advance its start
# A donor VA whose first millimetres are a label blob (mr_025 lVA: 5.3 mm radius
# against a 1.4 mm median) must not set the calibre the host is ramped to: join
# at the radius VA_ANCHOR_MM in, capped at VA_BLOB_FACTOR x the donor median,
# and flatten the donor's first VA_ANCHOR_MM to it, as extend_eca does.
VA_ANCHOR_MM = 4.0
VA_BLOB_FACTOR = 1.3
VA_BLOB_SPAN_MM = 10.0
# A fork ECA that runs inside its own route is no fork. v3's right forks keep
# at most 15.7 mm of ECA within the fusing band of their route; five left
# lowers that v4 newly admitted kept 23-40 mm (graft_three.load_lower ranks the
# daughters by calibre, and there an ICA-calibre sibling path came second).
ECA_FUSED_MAX_MM = 16.0
# A new branch against ITSELF: points more than this far apart in arclength
# (sdf_union_v4.WINDOW_MM, the ownership self-contact window) must clear.
SELF_WINDOW_MM = 12.0

TB = "topbrain_data"
TB_PARTS = {  # (kind, patient side) -> (centerline json, surface vtp); {p} = topcow_mr_NNN
    ("ica", "right"): ("centerlines/{p}_ica.json", "surfaces/{p}_rICA.vtp"),
    ("ica", "left"): ("centerlines_left/{p}_ica.json", "surfaces_left/{p}_lICA.vtp"),
    ("va", "right"): ("centerlines_rva/{p}_va.json", "surfaces_va/{p}_rVA.vtp"),
    ("va", "left"): ("centerlines_lva/{p}_va.json", "surfaces_va/{p}_lVA.vtp"),
    ("eca", "right"): ("centerlines_reca/{p}_eca.json", "surfaces_eca/{p}_rECA.vtp"),
    ("eca", "left"): ("centerlines_leca/{p}_eca.json", "surfaces_eca/{p}_lECA.vtp"),
}
MIRROR = [-1.0, 1.0, 1.0]
IDENT = [1.0, 1.0, 1.0]


class Unresolvable(Exception):
    """A required component could not be grafted cleanly: the anatomy is dropped."""


@contextlib.contextmanager
def g3_constants(**kw):
    """Run graft_three with different module constants, then restore them.

    compose(), eca_reentry() and friends read graft_three's module globals at
    call time, so this is how the left side reuses the right side's method
    exactly rather than through a copy that could drift."""
    old = {k: getattr(G3, k) for k in kw}
    for k, v in kw.items():
        setattr(G3, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(G3, k, v)


def short(fname):
    return fname.replace("Centerline curve", "cc").replace(".mrk.json", "")


# ------------------------------------------------------------------ TopBrain --
def tb_part(kind, anat_side, pid, mirrored, root="."):
    """(points, radii, meta) for a TopBrain part, in RAS mm, mirrored as needed.

    Native anatomies take the part from their own side; mirrored ones from the
    opposite side, x-mirrored. Returns (None, None, meta-with-why) if absent."""
    src_side = anat_side if not mirrored else ("left" if anat_side == "right" else "right")
    cl, sf = (os.path.join(root, TB, t.format(p=pid)) for t in TB_PARTS[(kind, src_side)])
    meta = {"kind": kind, "patient": pid, "patient_side": src_side, "centerline_src": cl,
            "surface": sf, "mirror": MIRROR if mirrored else IDENT}
    if not os.path.exists(cl):
        meta["why"] = "no %s centerline for %s %s" % (kind, pid, src_side)
        return None, None, meta
    j = json.load(open(cl, encoding="utf-8"))
    p, r = np.asarray(j["points"], float), np.asarray(j["radii"], float)
    if mirrored:
        p = p * np.asarray(MIRROR)
    p, r = prep_siphon(p, r)
    if kind in ("va", "eca"):
        p, r, lo, hi = trim_short_seeds(p, r)
        meta["seed_trim_pts"] = [lo, hi]
    k = anchor_trim(p)
    p, r = p[k:], r[k:]
    meta["anchor_trim_idx"] = int(k)
    meta["len_mm"] = float(arclength(p)[-1])
    return p, r, meta


def trim_short_seeds(p, r, step=G3.RESAMPLE_MM):
    """prep_siphon's seed-jog trim, at the PROXIMAL end, for pieces it leaves
    alone (21 points or fewer): drop first points while one of the first three
    turns exceeds SEED_KINK_DEG, at most SEED_TRIM_MAX_MM, keeping SEED_KEEP_MIN
    points. Only the proximal seed lands on a join; the distal one sits at a
    blind tip, and trimming it too cost 43 of 135 ECA extensions their 10 mm."""
    lo = 0
    nmax = int(round(SEED_TRIM_MAX_MM / step))
    while len(p) - lo > SEED_KEEP_MIN and lo < nmax:
        k = kink_deg(p[lo:])
        if len(k) >= 3 and k[:3].max() > SEED_KINK_DEG:
            lo += 1
            continue
        break
    return p[lo:], r[lo:], int(lo), 0


def join_kink(p, at_mm, span=JOIN_KINK_SPAN_MM):
    """Largest 1 mm-step turn within `span` of arclength `at_mm` (kink_deg, as
    compose measures the left route's max_kink)."""
    s = arclength(p)
    idx = np.nonzero((s >= at_mm - span) & (s <= at_mm + span))[0]
    if len(idx) < 2:
        return 0.0
    lo, hi = max(int(idx[0]) - 1, 0), min(int(idx[-1]) + 1, len(p) - 1)
    k = kink_deg(p[lo:hi + 1])
    return float(k.max()) if len(k) else 0.0


def self_clear(p, r, from_mm, window=SELF_WINDOW_MM):
    """Smallest lumen gap between points of one curve more than `window` apart
    in arclength, at least one of them at or past `from_mm`."""
    s = arclength(p)
    new = np.nonzero(s >= from_mm)[0]
    if not len(new):
        return np.inf
    g = pair_gaps(p[new], r[new], p, r)
    far = np.abs(s[new][:, None] - s[None, :]) > window
    return float(g[far].min()) if far.any() else np.inf


def fused_run(ep, er, rp, rr, band=None):
    """Arclength from an ECA's origin to its first point clear of its own route
    by `band` (default NEW_CLEAR_MM); the ECA's full length if it never clears."""
    band = NEW_CLEAR_MM if band is None else band
    s = arclength(ep)
    g = pair_gaps(ep, er, rp, rr).min(axis=1)
    clear = np.nonzero(g >= band)[0]
    return float(s[clear[0]]) if len(clear) else float(s[-1])


def place(src_p, anchor_p, anchor_t, span=G3.SIPHON_SPAN):
    """graft_three.place: frame-match (tangent + superior) and translate."""
    return G3.place(src_p, anchor_p, anchor_t, span)


# ---------------------------------------------------------------- lowers ------
def lower_records(manifest_path, ext_dir, need_mm):
    """Every manifest model as a graft_three lower record for the LEFT route.

    Usability is what compose() actually needs, not the manifest's rule. The
    manifest marks a lower usable when its ICA ALONE reaches 58 mm (extended by
    at most 10), but compose() places seam 2 at host_cut + cca + ica with
    host_cut <= HOST_CUT_MAX, so what has to reach is CCA + ICA. Applying the
    ICA-only rule rejected the patient's own other carotid for three of the
    first four anatomies tried (w_047_right, k_004_right, m_024_right).

    The donor's NATIVE ICA is preferred -- a copied 10 mm extension is synthetic
    -- and the extended version is used only when the native one cannot reach.
    `need_mm` = left seam - HOST_CUT_MAX_L; head_trim can still shorten the CCA
    later, which the builder's seam-landing check then catches."""
    man = json.load(open(manifest_path, encoding="utf-8"))
    ext = {}
    for f in glob.glob(os.path.join(ext_dir, "*_ica_extended.json")):
        d = json.load(open(f, encoding="utf-8"))
        ext[d["name"]] = (f, d)
    recs = {}
    for m in man["models"]:
        base = {"name": m["name"], "side": m["side"], "path": m["path"],
                "cca_mm": m["cca_mm"], "eca_mm": m["eca_mm"],
                "stenosis_pct": m.get("stenosis_pct")}
        native = dict(base, extended=False, ext_json=None, ica_mm=m["ica_mm"],
                      ica_real_mm=m["ica_mm"], tip_d=2 * (m.get("ica_tip_radius") or 0))
        # NOT a gate: the ICA tip diameter. match_sections rejected tips under
        # MIN_TIP_D, but tip_d is read at the donor's CROP FACE, which droops
        # (graft_three.CCA_ANCHOR_MM table: ICA tip 1.78 mm median at the face,
        # 2.15 mm 4 mm in), and compose() overwrites the last 25 mm of the ICA
        # with the seam-2 ramp to the siphon's calibre and then floors it at
        # ROUTE_MIN_R. Gating on it rejected the patient's own other carotid
        # for 67 anatomies (15 fork patients) over a value the build discards.
        # It is kept in the record as information.
        rec, why = None, None
        if m["cca_mm"] + m["ica_mm"] >= need_mm:
            rec = native
        elif m["name"] in ext:
            f, e = ext[m["name"]]
            tip = 2 * float(e["radii"][-1]) if e.get("radii") else native["tip_d"]
            if m["cca_mm"] + e["ica_mm_after"] >= need_mm:
                rec = dict(base, extended=True, ext_json=f.replace("\\", "/"), ica_mm=e["ica_mm_after"],
                           ica_real_mm=e["ica_mm_before"], tip_d=tip)
        if rec is None and why is None:
            why = "CCA+ICA %.1f mm < %.1f mm needed to reach the left seam" % (
                m["cca_mm"] + m["ica_mm"], need_mm)
        recs[m["name"]] = dict(rec or native, unusable=why)
    for u in man.get("unparsed", []):
        n = os.path.basename(u if isinstance(u, str) else u[0]).replace("_lumen_centerlines.vtp", "")
        recs.setdefault(n, {"name": n, "unusable": "tree would not split"})
    return recs


def right_records(manifest_path, ext_dir):
    """The right lowers' real (pre-extension) ICA lengths, for the v3 correction."""
    man = json.load(open(manifest_path, encoding="utf-8"))
    out = {m["name"]: {"ica_real_mm": m["ica_mm"], "extended": False, "extend_mm": 0.0} for m in man["models"]}
    for f in glob.glob(os.path.join(ext_dir, "*_ica_extended.json")):
        d = json.load(open(f, encoding="utf-8"))
        # v3 (match_sections) used the extended ICA whenever this file exists
        out.setdefault(d["name"], {}).update(ica_real_mm=d["ica_mm_before"], extended=True,
                                             extend_mm=float(d["ica_mm_after"] - d["ica_mm_before"]))
    return out


def other_side(lower_name):
    case, side = lower_name.rsplit("_", 1)
    return "%s_%s" % (case, "right" if side == "left" else "left")


def left_lower_candidates(right_lower, recs, siphon_prox_d):
    """The patient's own other carotid first; then calibre-matched substitutes.

    Substitutes are every usable lower ranked by |tip_d - siphon proximal d|,
    left carotids before right ones (the route is a left route), then by name."""
    own = other_side(right_lower)
    out = []
    r = recs.get(own)
    if r is not None and not r.get("unusable"):
        out.append((r, None))
    own_why = ("other side %s: %s" % (own, (r or {}).get("unusable") or "absent from the database")
               if not out else None)
    # never the anatomy's own right fork: that would put one carotid on both sides
    subs = [x for x in recs.values() if not x.get("unusable") and x["name"] not in (own, right_lower)]
    subs.sort(key=lambda x: (abs(x["tip_d"] - siphon_prox_d), x["side"] != "left", x["name"]))
    out += [(x, own_why or "own other side failed its gates") for x in subs]
    return out


# ---------------------------------------------------------------- grafting ----
def left_seam_mm(host_rcca, host_lcca):
    hs = arclength(host_rcca)
    z = host_rcca[int(np.searchsorted(hs, G3.SIPHON_SEAM_MM)), 2]
    ls = arclength(host_lcca)
    above = np.nonzero(host_lcca[:, 2] >= z)[0]
    return float(ls[above[0]]) if len(above) else LEFT_SEAM_FALLBACK


def graft_va(keep_p, keep_r, donor_p, donor_r):
    """The kept host VA (already cut), then the donor VA frame-matched onto its
    end, the host ramped to the donor's calibre and the donor floored."""
    keep_p, keep_r = keep_p.copy(), keep_r.copy()
    hs = arclength(keep_p)
    k = len(keep_p) - 1
    ds = arclength(donor_p)
    donor_r = np.maximum(np.asarray(donor_r, float), VA_MIN_R)
    cap = VA_BLOB_FACTOR * float(np.median(donor_r))
    r_join = min(float(np.interp(min(VA_ANCHOR_MM, ds[-1]), ds, donor_r)), cap)
    donor_r = np.where(ds < VA_ANCHOR_MM, r_join, donor_r)
    donor_r = np.where(ds < VA_BLOB_SPAN_MM, np.minimum(donor_r, cap), donor_r)
    keep_r = G3.ramp_radius(hs[:k + 1], keep_r, hs[k], r_join,
                            min(VA_BLEND_MM, hs[k] - 10.0))
    moved, R, origin = place(donor_p, keep_p[-1], tangent_at(keep_p, len(keep_p) - 1))
    p = np.vstack([keep_p, moved[1:]])
    r = np.concatenate([keep_r, donor_r[1:]])
    p, r = resample(p, r, G3.RESAMPLE_MM)
    xf = {"R": R.tolist(), "origin": np.asarray(origin).tolist(), "anchor": keep_p[-1].tolist()}
    return p, r, float(hs[k]), xf


def extend_eca(ep, er, dp, dr):
    """Append a TopBrain ECA piece to a fork ECA at its tip. None if inconvenient."""
    L = float(arclength(dp)[-1])
    if L < ECA_EXT_MIN_MM:
        return None, "TopBrain ECA piece %.1f mm < %.0f" % (L, ECA_EXT_MIN_MM)
    es, ds = arclength(ep), arclength(dp)
    r_fork = float(np.interp(max(es[-1] - ECA_EXT_ANCHOR_MM, 0.0), es, er))
    r_tb = float(np.interp(min(ECA_EXT_ANCHOR_MM, ds[-1]), ds, dr))
    if abs(r_fork - r_tb) > ECA_EXT_MAX_MISMATCH_MM:
        return None, "calibre mismatch %.2f mm at the join (fork %.2f, TopBrain %.2f)" % (
            abs(r_fork - r_tb), r_fork, r_tb)
    dr = np.maximum(dr, G3.ECA_MESH_R_MM)
    dr = np.where(ds < ECA_EXT_ANCHOR_MM, max(r_tb, G3.ECA_MESH_R_MM), dr)   # flatten its drooping face
    er2 = G3.ramp_radius(es, er, es[-1], float(dr[0]), min(ECA_EXT_BLEND_MM, es[-1] * 0.5))
    moved, R, origin = place(dp, ep[-1], tangent_at(ep, len(ep) - 1))
    p = np.vstack([ep, moved[1:]])
    r = np.concatenate([er2, dr[1:]])
    p, r = resample(p, r, G3.RESAMPLE_MM)
    xf = {"R": R.tolist(), "origin": np.asarray(origin).tolist(), "anchor": ep[-1].tolist()}
    return (p, r, float(es[-1]), xf), None


def clear_from(p, r, from_mm, neighbours):
    return min_clearance(p, r, neighbours, from_mm) if len(neighbours) else (np.inf, "")


def load_lower_v4(rec):
    """graft_three.load_lower, with the ECA re-picked where its choice is no fork.

    load_lower takes the second daughter by calibre as the ECA. In five left
    lowers v4 newly admitted (case_w_018_left, w_002_left, w_050_right,
    w_042_right, w_026_right) that daughter is an ICA-calibre sibling path which
    shares the ICA for 23-40 mm; the true ECA is a lower-ranked daughter that
    clears the ICA within 8-15 mm. Wherever load_lower's ECA clears its ICA
    within ECA_FUSED_MAX_MM its choice stands unchanged; otherwise the daughter
    that clears earliest, of at least ECA_MIN_MM and a third of the ICA's
    calibre, is taken. Returns (lower, note) or (None, why)."""
    lower = G3.load_lower(rec)
    if lower is None:
        return None, "lower tree would not split"
    (cca, ica, eca) = lower
    ip, ir = resample(ica[0], ica[1], G3.RESAMPLE_MM)
    ep, er = resample(eca[0], eca[1], G3.RESAMPLE_MM)
    if fused_run(ep, er, ip, ir) <= ECA_FUSED_MAX_MM:
        return lower, None
    _, dau = G3.split_tree(G3.read_centerlines(rec["path"]))
    ical = float(np.median(ir[:max(len(ir) // 2, 1)]))
    best = None
    for d in dau:
        if d[1] is None or len(d[0]) < 3:
            continue
        dp, dr = resample(d[0], d[1], G3.RESAMPLE_MM)
        if arclength(dp)[-1] < G3.ECA_MIN_MM or float(np.median(dr[:max(len(dr) // 2, 1)])) < ical / 3.0:
            continue
        fr = fused_run(dp, dr, ip, ir)
        if fr <= ECA_FUSED_MAX_MM and (best is None or fr < best[0]):
            best = (fr, d)
    if best is None:
        return None, "no daughter clears the ICA within %.0f mm" % ECA_FUSED_MAX_MM
    return (cca, ica, best[1]), "ECA re-picked (calibre rank 2 fused with the ICA; this one clears at %.1f mm)" % best[0]


def self_gap_profile(p, r, window=SELF_WINDOW_MM):
    """Per point: smallest lumen gap to points of the same curve more than
    `window` apart in arclength (inf where there are none)."""
    s = arclength(p)
    g = pair_gaps(p, r, p, r)
    g = np.where(np.abs(s[:, None] - s[None, :]) > window, g, np.inf)
    return g.min(axis=1)


def grown_clear(p, r, host_p, host_r, upto_mm, neighbours, host_names=None):
    """Clearance of the kept host stretch that a radius ramp WIDENED.

    Every graft here blends the kept host vessel to the donor's calibre over
    the stretch just below the cut (graft_va, compose's CCA ramp, extend_eca),
    and that adds volume where the donor-side gates -- which start at the cut
    -- never look. The first v4 bake found it: case_k_011_left__topcow_mr_013_L's
    RVA ramp (143-168 mm) grew into the right route at 155-160 mm, a tube-level
    handle behind a recorded 2.42 mm clearance. The kept host points sit where
    they always did, so a gap is NEW only where it was open before the ramp
    (host radius) and the ramp closed it below the band; contacts that already
    overlapped are the host's own junctions and stay as shipped. That excuse
    holds only against unchanged geometry: `host_names` lists the neighbours it
    applies to (None: all); against a new neighbour every contact counts."""
    s = arclength(p)
    m = s <= upto_mm + 1e-6
    rh = np.interp(s[m], arclength(host_p), host_r)
    grown = r[m] > rh + 0.01
    worst, who = np.inf, ""
    if not grown.any():
        return worst, who
    q, qr, dr = p[m][grown], r[m][grown], (r[m] - rh)[grown]
    for name, bp, br in neighbours:
        g = pair_gaps(q, qr, bp, br)
        if host_names is None or name in host_names:
            g = np.where(g + dr[:, None] > 0.0, g, np.inf)
        if g.min() < worst:
            worst, who = float(g.min()), name + " (ramp)"
    return worst, who


def new_clear(p, r, cut_mm, host_p, host_r, neighbours, host_names=None, from_mm=None):
    """Clearance of a graft: every point from the cut on (from `from_mm` if
    given -- a VA cut that moved up also keeps host VA past the planned cut),
    the ramp-widened stretch below it, and the curve against itself."""
    c, w = clear_from(p, r, cut_mm if from_mm is None else min(from_mm, cut_mm), neighbours)
    c2, w2 = grown_clear(p, r, host_p, host_r, cut_mm, neighbours, host_names)
    c3 = self_clear(p, r, cut_mm)
    return min((c, w), (c2, w2), (c3, "itself"), key=lambda t: t[0])


def trim_to_clear(p, r, start_mm, neighbours, keep_min_mm):
    """Cut a free end back to its last clear point (past start_mm). None if too short."""
    s = arclength(p)
    g = G3.clear_profile(p, r, neighbours)
    bad = (g < NEW_CLEAR_MM) & (s > start_mm)
    if not bad.any():
        return p, r
    cut = float(s[int(np.argmax(bad))]) - G3.CLEAR_MARGIN_MM
    if cut - start_mm < keep_min_mm:
        return None
    keep = s <= cut
    return p[keep], r[keep]


# ------------------------------------------------------------------- build ----
def build(name, a, rrecs, recs):
    v3 = os.path.join(a.v3, name)
    prov3 = json.load(open(os.path.join(v3, "provenance.json"), encoding="utf-8"))
    siphon_R = prov3["siphon"]
    mirrored = siphon_R.endswith("_L")
    pid = siphon_R[:-2] if mirrored else siphon_R
    cdir = os.path.join(v3, "Centrelines_comb")
    br = {}
    for f in sorted(os.listdir(cdir)):
        if f.endswith(".json"):
            br[f] = read_curve(os.path.join(cdir, f))          # (template, points, radii)
    notes = []

    # ---- 1. vertebral cut plan: the donor VA lengths fix where each host VA ends
    va = {}
    for side, fname in (("right", RVA), ("left", LVA)):
        dp, dr, meta = tb_part("va", side, pid, mirrored, a.root)
        if dp is None or meta["len_mm"] < VA_MIN_LEN_MM:
            raise Unresolvable("VA %s: %s" % (side, meta.get("why") or "donor VA %.1f mm < %.0f"
                                               % (meta["len_mm"], VA_MIN_LEN_MM)))
        _, hp, hr = br[fname]
        hp, hr = resample(hp, hr, G3.RESAMPLE_MM)
        Lh = float(arclength(hp)[-1])
        cut = float(np.clip(Lh - meta["len_mm"], VA_CUT_MIN_MM, Lh - VA_TAIL_MM))
        k = int(np.searchsorted(arclength(hp), cut))
        va[side] = {"file": fname, "donor": (dp, dr), "meta": meta, "cut_mm": cut,
                    "prox": (hp[:k + 1], hr[:k + 1]), "host_len_mm": Lh,
                    "host": (hp, hr), "prox_fixed": False}

    # ---- 2. left carotid: the patient's other carotid + other ICA siphon
    sp, sr, smeta = tb_part("ica", "left", pid, mirrored, a.root)
    if sp is None or smeta["len_mm"] < 60.0:
        # mr_006's left ICA is a 19.5 mm fragment. The fallback keeps the donor
        # head one patient but not one vessel per side: it is the RIGHT route's
        # own ICA with the opposite handedness (a native anatomy takes the
        # patient's right ICA mirrored, a mirrored one the left ICA as is).
        why = smeta.get("why") or "left siphon %.1f mm < 60" % smeta["len_mm"]
        sp, sr, smeta = tb_part("ica", "left", pid, not mirrored, a.root)
        if sp is None or smeta["len_mm"] < 60.0:
            raise Unresolvable("left siphon: " + why)
        smeta["fallback"] = "%s; used the patient's %s ICA%s -- the same vessel as the right siphon, in the opposite handedness" % (
            why, smeta["patient_side"], ", mirrored" if smeta["mirror"] == MIRROR else ", unmirrored")
        notes.append("left siphon fallback")
    _, lcca_p, lcca_r = br[LCCA]
    lcca_p, lcca_r = resample(lcca_p, lcca_r, G3.RESAMPLE_MM)
    seam_L = left_seam_mm(resample(*read_curve(os.path.join(a.host, RCCA))[1:], G3.RESAMPLE_MM)[0], lcca_p)

    right_route = br[RCCA][1:]
    fixed = [(short(f), p, r) for f, (_, p, r) in br.items()
             if f not in (LCCA, LVA, RVA) and r is not None]
    fixed_names = {n[0] for n in fixed}
    va_prox = [(short(RVA), *va["right"]["prox"]), (short(LVA), *va["left"]["prox"])]

    left = None
    tried = []
    for rec, fb in left_lower_candidates(prov3["lower"], recs, 2 * float(sr[0]))[:a.max_left_tries]:
        try:
            with g3_constants(SIPHON_SEAM_MM=seam_L, HOST_CUT_MIN=HOST_CUT_MIN_L,
                              HOST_CUT_MAX=HOST_CUT_MAX_L, BLEND_ANCHOR_MM=BLEND_ANCHOR_L, **V4_NEW):
                lower, lnote = load_lower_v4(rec)
                if lower is None:
                    raise ValueError(lnote)
                rp, rr, ep, er, diag = G3.compose(lcca_p, lcca_r, lower, sp, sr)
                if rp is None:
                    raise ValueError(diag.get("why", "compose failed"))
                if diag["max_kink"] > a.max_kink:
                    raise ValueError("kink %.0f deg" % diag["max_kink"])
                if diag["eca_mm"] < G3.ECA_MIN_MM:
                    raise ValueError("ECA %.1f mm" % diag["eca_mm"])
                seam_err = diag["host_cut_mm"] + diag["cca_mm"] + diag["ica_mm"] - seam_L
                if seam_err < -a.max_seam_short:
                    raise ValueError("left seam lands %.1f mm short" % -seam_err)
                nb = fixed + va_prox
                graft_from = diag["host_cut_mm"]
                note = [lnote] if lnote else []
                eg = np.minimum(G3.clear_profile(ep, er, nb), self_gap_profile(ep, er))
                if eg.min() < NEW_CLEAR_MM:
                    es = arclength(ep)
                    cutoff = float(es[int(np.argmax(eg < NEW_CLEAR_MM))]) - G3.CLEAR_MARGIN_MM
                    if cutoff < G3.ECA_MIN_MM:
                        raise ValueError("LECA clear for only %.0f mm" % max(cutoff, 0))
                    k = arclength(ep) <= cutoff
                    ep, er = ep[k], er[k]
                    note.append("LECA trimmed to %.0f mm" % cutoff)
                reent = G3.eca_reentry(rp, rr, ep, er)
                if reent is not None:
                    cutoff = reent - G3.CLEAR_MARGIN_MM
                    if cutoff < G3.ECA_MIN_MM:
                        raise ValueError("LECA re-enters the route at %.0f mm" % max(cutoff, 0))
                    k = arclength(ep) <= cutoff
                    ep, er = ep[k], er[k]
                    note.append("LECA re-entry cut at %.0f mm" % cutoff)
                fr = fused_run(ep, er, rp, rr)
                if fr > ECA_FUSED_MAX_MM:
                    raise ValueError("LECA inside its route for %.0f mm" % fr)
                diag["eca_mm"] = float(arclength(ep)[-1])
                diag["eca_fused_mm"] = fr

                def worst(nbx):
                    c, w = min_clearance(rp, rr, nbx, graft_from)
                    gc, gw = grown_clear(rp, rr, lcca_p, lcca_r, diag["host_cut_mm"], nbx)
                    ec, ew = min_clearance(ep, er, nbx, 0.0)
                    sc = min(self_clear(rp, rr, graft_from), self_clear(ep, er, 0.0))
                    return min((c, w), (gc, gw), (ec, ew + " (LECA)"), (sc, "itself"), key=lambda t: t[0])

                clear, who = worst(nb)
                lva_prox = va["left"]["prox"]
                if clear < NEW_CLEAR_MM and who in (short(LVA), short(LVA) + " (ramp)"):
                    # bend the LVA off the grafted route, or off the host stretch
                    # the CCA ramp widened if that is where it touches; the host
                    # LCCA below s2 stays a constraint (rva_deflect holds each
                    # point to its own baseline where it already touches)
                    s2 = int(np.searchsorted(arclength(rp), BLEND_ANCHOR_L if who.endswith("(ramp)") else graft_from))
                    others = ([n for n in nb if n[0] != short(LVA)] + [(short(LECA), ep, er)]
                              + ([(short(LCCA), rp[:s2], rr[:s2])] if s2 > 1 else []))
                    bent = rva_deflect(rp[s2:], rr[s2:], *lva_prox, others)
                    if bent is not None:
                        lva_prox = (bent[0], bent[1])
                        note.append("LVA deflected %.1f mm" % bent[2])
                        nb = [(n[0], *lva_prox) if n[0] == short(LVA) else n for n in nb]
                        clear, who = worst(nb)
                if clear < NEW_CLEAR_MM:
                    raise ValueError("clearance %.2f mm with %s" % (clear, who))
            left = {"route": (rp, rr), "eca": (ep, er), "diag": diag, "rec": rec,
                    "fallback": fb, "clear": float(clear), "notes": note,
                    "lva_prox": lva_prox, "seam_target_mm": seam_L, "tried": list(tried)}
            break
        except Exception as e:                                          # noqa: BLE001
            tried.append("%s: %s" % (rec["name"], str(e)[:80]))
    if left is None:
        raise Unresolvable("left carotid: " + " | ".join(tried[:4]))
    va["left"]["prox"] = left["lva_prox"]
    va["left"]["prox_fixed"] = any(n.startswith("LVA deflected") for n in left["notes"])
    if left["fallback"]:
        notes.append("left lower fallback")

    # ---- 3. vertebral grafts, each against everything else as it now stands
    placed = {}
    base_nb = fixed + [(short(LCCA), *left["route"]), (short(LECA), *left["eca"])]
    for side in ("right", "left"):
        v = va[side]
        dp0, dr0 = v["donor"]
        ds0 = arclength(dp0)
        other = "left" if side == "right" else "right"
        other_va = placed[other][:2] if other in placed else va[other]["prox"]
        nb = base_nb + [(short(va[other]["file"]), *(other_va[:2]))]
        host_names = fixed_names | ({short(va[other]["file"])} if other not in placed else set())
        # First repair, and the only one that keeps the protocol intact: move
        # the CUT. The donor segment is still frame-matched (tangent + superior)
        # onto the host VA, just a few millimetres higher or lower, and where it
        # lands changes with the host tangent there. The first run dropped 23
        # anatomies, every one a right donor VA crossing the right carotid route
        # at the length-matched cut -- the same crowding that made v3 bend the
        # host RVA in 100 anatomies. Nearest cut first; a host LVA that step 2
        # already bent can only be cut further down, not extended. At each cut
        # a donor that opens on a turn sharper than JOIN_MAX_KINK has its start
        # advanced (DONOR_TRIM_STEPS_MM) before the cut is given up.

        def donor(tr):
            keep = ds0 >= tr
            return dp0[keep], dr0[keep]

        placed_ok, tried_cuts = None, []
        for dc in VA_CUT_STEPS:
            cut = v["cut_mm"] + dc
            if cut < VA_CUT_MIN_MM or cut > v["host_len_mm"] - VA_TAIL_MM or (v["prox_fixed"] and dc > 0):
                continue
            hp, hr = v["prox"] if v["prox_fixed"] else v["host"]
            k = int(np.clip(np.searchsorted(arclength(hp), cut), 5, len(hp) - 1))
            for tr in DONOR_TRIM_STEPS_MM:
                dp, dr = donor(tr)
                if arclength(dp)[-1] < VA_MIN_LEN_MM:
                    break
                p, r, cut_at, xf = graft_va(hp[:k + 1], hr[:k + 1], dp, dr)
                kink = join_kink(p, cut_at)
                if kink > JOIN_MAX_KINK:
                    tried_cuts.append("%+.0f/trim %.0f: join %.0f deg" % (dc, tr, kink))
                    continue
                clear, who = new_clear(p, r, cut_at, hp, hr, nb, host_names, from_mm=v["cut_mm"])
                if clear >= NEW_CLEAR_MM:
                    rep = ", ".join(x for x in ("cut moved %+.0f mm" % dc if dc else "",
                                                "donor start trimmed %.0f mm" % tr if tr else "") if x) or None
                    placed_ok = (p, r, cut_at, dict(xf, donor_start_trim_mm=float(tr)), rep, float(clear), kink)
                    break
                tried_cuts.append("%+.0f/trim %.0f: %.2f mm with %s" % (dc, tr, clear, who))
                break           # the clearance, not the join, failed: next cut
            if placed_ok is not None:
                break
        if placed_ok is not None:
            placed[side] = placed_ok
            continue
        # then bend or shorten at the length-matched cut, as the RVA repair does,
        # with the least donor trim whose join passes
        hp, hr = v["prox"]
        for tr in DONOR_TRIM_STEPS_MM:
            dp, dr = donor(tr)
            p, r, cut_at, xf = graft_va(hp, hr, dp, dr)
            kink = join_kink(p, cut_at)
            if kink <= JOIN_MAX_KINK and arclength(dp)[-1] >= VA_MIN_LEN_MM:
                break
        else:
            raise Unresolvable("VA %s: every join turns more than %.0f deg (%s)"
                               % (side, JOIN_MAX_KINK, "; ".join(tried_cuts[:6])))
        xf = dict(xf, donor_start_trim_mm=float(tr))
        repair = "donor start trimmed %.0f mm" % tr if tr else None
        clear, who = new_clear(p, r, cut_at, hp, hr, nb, host_names)
        if clear < NEW_CLEAR_MM:
            who_ = who.replace(" (ramp)", "")
            offenders = [n for n in nb if n[0] == who_] or nb
            q = np.vstack([o[1] for o in offenders]); qr = np.concatenate([o[2] for o in offenders])
            rest = [n for n in nb if n[0] != who_]
            bent = rva_deflect(q, qr, p, r, rest)
            if bent is not None:
                p, r = bent[0], bent[1]
                repair = ", ".join(x for x in (repair, "deflected %.1f mm from %s" % (bent[2], who)) if x)
                # the donor segment no longer sits where its transform put it,
                # so its real surface would float off the bent centerline:
                # bake that section as a tube (bake_meshes_v4 honours this)
                xf = dict(xf, real_surface=False, why_tube="VA deflected %.1f mm after grafting" % bent[2])
            else:
                t = trim_to_clear(p, r, cut_at, nb, VA_MIN_LEN_MM)
                if t is not None:
                    lost = float(arclength(p)[-1] - arclength(t[0])[-1])
                    if lost <= VA_MAX_SHORTEN_MM:
                        p, r = t
                        repair = ", ".join(x for x in (repair, "shortened %.0f mm (clear of %s)" % (lost, who)) if x)
            clear, who = new_clear(p, r, cut_at, hp, hr, nb, host_names)
            if clear < NEW_CLEAR_MM:
                raise Unresolvable("VA %s: clearance %.2f mm with %s after repair (cuts tried: %s)"
                                   % (side, clear, who, "; ".join(tried_cuts[:6])))
        placed[side] = (p, r, cut_at, xf, repair, float(clear), join_kink(p, cut_at))

    # ---- 4. ECA extensions (optional; never drops an anatomy)
    ecas = {"right": br[RECA][1:], "left": left["eca"]}
    eca_ext = {}
    all_nb = (fixed + [(short(LCCA), *left["route"])]
              + [(short(RVA), *placed["right"][:2]), (short(LVA), *placed["left"][:2])])
    for side in ("right", "left"):
        ep, er = ecas[side]
        own_route = right_route if side == "right" else left["route"]
        dp0, dr0, meta = tb_part("eca", side, pid, mirrored, a.root)
        if dp0 is None:
            eca_ext[side] = {"extended": False, "why": meta.get("why")}
            continue
        me = short(RECA if side == "right" else LECA)
        other_eca = short(LECA if side == "right" else RECA)
        nb = [n for n in all_nb if n[0] != me] + [(other_eca, *ecas["left" if side == "right" else "right"])]
        host_names = fixed_names - {me} - ({other_eca} if eca_ext.get("right", {}).get("extended") else set())
        ds0 = arclength(dp0)
        res = why = None
        with g3_constants(**V4_NEW):
            for tr in DONOR_TRIM_STEPS_MM:
                keep = ds0 >= tr
                res, why = extend_eca(ep, er, dp0[keep], dr0[keep])
                if res is None:
                    break
                kink = join_kink(res[0], res[2])
                if kink <= JOIN_MAX_KINK:
                    break
                why = "the join turns %.0f deg" % kink
                res = None
            if res is None:
                eca_ext[side] = {"extended": False, "why": why, **meta}
                continue
            p, r, join_at, xf = res
            xf = dict(xf, donor_start_trim_mm=float(tr))
            # clear of its neighbours and of itself past the join
            g = np.minimum(G3.clear_profile(p, r, nb), self_gap_profile(p, r))
            s = arclength(p)
            bad = (g < NEW_CLEAR_MM) & (s > join_at)
            if bad.any():
                cut = float(s[int(np.argmax(bad))]) - G3.CLEAR_MARGIN_MM
                if cut - join_at < ECA_EXT_MIN_MM:
                    eca_ext[side] = {"extended": False, "why": "extension collides within %.0f mm" % ECA_EXT_MIN_MM, **meta}
                    continue
                p, r = p[s <= cut], r[s <= cut]
            gc, gw = grown_clear(p, r, ep, er, join_at, nb, host_names)       # nb holds both routes
            if gc < NEW_CLEAR_MM:
                eca_ext[side] = {"extended": False, "why": "calibre ramp at the join comes %.2f mm from %s" % (gc, gw), **meta}
                continue
            reent = G3.eca_reentry(*own_route, p, r)
            if reent is not None and reent - G3.CLEAR_MARGIN_MM < join_at + ECA_EXT_MIN_MM:
                eca_ext[side] = {"extended": False, "why": "extension re-enters its own route", **meta}
                continue
            if reent is not None:
                k = arclength(p) <= reent - G3.CLEAR_MARGIN_MM
                p, r = p[k], r[k]
        ecas[side] = (p, r)
        eca_ext[side] = {"extended": True, "join_at_mm": join_at, "join_kink_deg": join_kink(p, join_at),
                         "added_mm": float(arclength(p)[-1] - join_at), "xform": xf, **meta}

    # ---- final clearances, measured on everything as it is written
    final = {short(f): (p, r) for f, (_, p, r) in br.items() if r is not None}
    final.update({short(LCCA): left["route"], short(LECA): ecas["left"], short(RECA): ecas["right"],
                  short(RVA): placed["right"][:2], short(LVA): placed["left"][:2]})

    def against(*skip):
        return [(n, p, r) for n, (p, r) in final.items() if n not in skip]

    lrp, lrr = left["route"]
    hc = left["diag"]["host_cut_mm"]
    left_final = min(min_clearance(lrp, lrr, against(short(LCCA), short(LECA)), hc)[0],
                     grown_clear(lrp, lrr, lcca_p, lcca_r, hc, against(short(LCCA), short(LECA)), fixed_names)[0],
                     min_clearance(*ecas["left"], against(short(LCCA), short(LECA)), 0.0)[0],
                     self_clear(lrp, lrr, hc), self_clear(*ecas["left"], 0.0))
    va_final = {}
    for side, fname in (("right", RVA), ("left", LVA)):
        p, r, cut_at = placed[side][:3]
        hp, hr = va[side]["prox"] if va[side]["prox_fixed"] else va[side]["host"]
        va_final[side] = new_clear(p, r, cut_at, hp, hr, against(short(fname)), fixed_names)[0]

    # ---- 5. write
    out = os.path.join(a.out, name)
    cout = os.path.join(out, "Centrelines_comb")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(cout)
    for f in br:
        if f not in (RCCA, RECA, RVA, LCCA, LVA):
            shutil.copy2(os.path.join(cdir, f), os.path.join(cout, f))
    shutil.copy2(os.path.join(cdir, RCCA), os.path.join(cout, RCCA))    # byte-identical right route
    if eca_ext["right"].get("extended"):
        write_curve(br[RECA][0], os.path.join(cout, RECA), *ecas["right"])
    else:
        shutil.copy2(os.path.join(cdir, RECA), os.path.join(cout, RECA))
    write_curve(br[LCCA][0], os.path.join(cout, LCCA), *left["route"])
    write_curve(br[RECA][0], os.path.join(cout, LECA), *ecas["left"])
    write_curve(br[RVA][0], os.path.join(cout, RVA), *placed["right"][:2])
    write_curve(br[LVA][0], os.path.join(cout, LVA), *placed["left"][:2])

    # provenance: v3's right record (with the real ICA length corrected) + the new parts
    rrec = rrecs.get(prov3["lower"], {})
    xform = dict(prov3["xform"])
    xform["lower"] = dict(xform["lower"], ica_real_mm=float(rrec.get("ica_real_mm", prov3["ica_mm"])),
                          ica_extended=bool(rrec.get("extended", False)),
                          extend_mm=float(rrec.get("extend_mm", 0.0)))
    ld, lrec = left["diag"], left["rec"]
    lx = ld["xform"]
    lx["lower"].update(kind="zenodo", mirror=IDENT,
                       surface=lrec["path"].replace("_lumen_centerlines.vtp", "_lumen.stl"),
                       ica_real_mm=float(lrec["ica_real_mm"]),
                       extend_mm=float(lrec["ica_mm"] - lrec["ica_real_mm"]))
    lx["siphon"].update(kind="topbrain", centerline_src=smeta["centerline_src"],
                        mirror=smeta["mirror"], surface=smeta["surface"])
    xform.update(lower_L=lx["lower"], siphon_L=lx["siphon"])
    for side, key in (("right", "va_R"), ("left", "va_L")):
        m = va[side]["meta"]
        xform[key] = dict(placed[side][3], kind="topbrain", centerline_src=m["centerline_src"],
                          mirror=m["mirror"], surface=m["surface"], cut_mm=placed[side][2])
    for side, key in (("right", "ecaext_R"), ("left", "ecaext_L")):
        e = eca_ext[side]
        if e.get("extended"):
            xform[key] = dict(e["xform"], kind="topbrain", centerline_src=e["centerline_src"],
                              mirror=e["mirror"], surface=e["surface"], join_at_mm=e["join_at_mm"])
    ld = {k: v for k, v in ld.items() if k != "xform"}
    prov = {k: v for k, v in prov3.items() if k != "xform"}
    # v3's right-route gate value, measured against v3's neighbours under its
    # 0.35 mm band; kept, but named so it is not read as a v4 clearance
    if "clearance_mm" in prov:
        prov["v3_clearance_mm"] = prov.pop("clearance_mm")

    def va_repair(side):
        step2 = [n for n in left["notes"] if n.startswith("LVA deflected")] if side == "left" else []
        parts = ["host " + n + " for the left carotid" for n in step2] + ([placed[side][4]] if placed[side][4] else [])
        return "; ".join(parts) or None
    prov.update(
        v4=True, new_graft_clearance_mm=NEW_CLEAR_MM, v3_name=name, handedness="mirrored" if mirrored else "native", patient=pid,
        left={"lower": lrec["name"], "lower_side": lrec["side"], "lower_fallback": left["fallback"],
              "lower_extended": bool(lrec["extended"]), "siphon": smeta["patient"] + (
                  "_L" if smeta["patient_side"] == "left" else "") + (" (mirrored)" if smeta["mirror"] == MIRROR else ""),
              "siphon_fallback": smeta.get("fallback"), "seam_target_mm": left["seam_target_mm"],
              "clearance_mm": left_final, "clearance_at_gate_mm": left["clear"],
              "repairs": "; ".join(left["notes"]) or None,
              "rejected_candidates": left["tried"] or None, **ld},
        vertebrals={side: {"patient_side": va[side]["meta"]["patient_side"], "donor_len_mm": va[side]["meta"]["len_mm"],
                           "host_len_mm": va[side]["host_len_mm"], "cut_mm": placed[side][2],
                           "total_mm": float(arclength(placed[side][0])[-1]),
                           "repair": va_repair(side), "clearance_mm": va_final[side],
                           "clearance_at_gate_mm": placed[side][5], "join_kink_deg": placed[side][6],
                           "donor_start_trim_mm": placed[side][3].get("donor_start_trim_mm", 0.0),
                           "seed_trim_pts": va[side]["meta"].get("seed_trim_pts")} for side in ("right", "left")},
        eca_extension={side: {k: v for k, v in eca_ext[side].items() if k != "xform"} for side in ("right", "left")},
        notes=notes or None, xform=xform)
    with open(os.path.join(out, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1)
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="carotid_data/anatomies_v3")
    ap.add_argument("--out", default="carotid_data/anatomies_v4")
    ap.add_argument("--host", default="eve_bench/data/dualdevicenav/Centrelines_comb")
    ap.add_argument("--manifest", default="carotid_data/lower_manifest.json")
    ap.add_argument("--extended", default="carotid_data/extended")
    ap.add_argument("--root", default=".", help="repo root holding topbrain_data/")
    ap.add_argument("--only", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--max-kink", type=float, default=60.0)
    ap.add_argument("--max-seam-short", type=float, default=5.0,
                    help="reject a left lower whose seam lands further short of the target than this")
    ap.add_argument("--max-left-tries", type=int, default=6)
    a = ap.parse_args()

    _, hr_p, hr_r = read_curve(os.path.join(a.host, RCCA))
    _, hl_p, hl_r = read_curve(os.path.join(a.host, LCCA))
    seam_L = left_seam_mm(resample(hr_p, hr_r, G3.RESAMPLE_MM)[0], resample(hl_p, hl_r, G3.RESAMPLE_MM)[0])
    need_mm = seam_L - HOST_CUT_MAX_L - a.max_seam_short
    recs = lower_records(a.manifest, a.extended, need_mm)
    rrecs = right_records(a.manifest, a.extended)
    usable = sum(1 for r in recs.values() if not r.get("unusable"))
    print("left seam %.1f mm; a left lower needs CCA+ICA >= %.1f mm; %d of %d lowers usable" % (seam_L, need_mm, usable, len(recs)))
    names = sorted(d for d in os.listdir(a.v3) if os.path.isdir(os.path.join(a.v3, d, "Centrelines_comb")))
    if a.only:
        want = set(a.only.split(","))
        names = [n for n in names if n in want]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        names = names[i::n]
    os.makedirs(a.out, exist_ok=True)
    ok, drops = 0, []
    for name in names:
        try:
            p = build(name, a, rrecs, recs)
            ok += 1
            L = p["left"]
            print("OK    %-44s L:%s%s  VA:%s/%s  ECAext:%s/%s  %s" % (
                name[:44], L["lower"], " (fb)" if L["lower_fallback"] else "",
                p["vertebrals"]["right"]["repair"] and "rep" or "ok",
                p["vertebrals"]["left"]["repair"] and "rep" or "ok",
                "Y" if p["eca_extension"]["right"].get("extended") else "-",
                "Y" if p["eca_extension"]["left"].get("extended") else "-",
                "; ".join(filter(None, [L["repairs"]] + (p["notes"] or [])))), flush=True)
        except Unresolvable as e:
            drops.append((name, str(e)))
            print("DROP  %-44s %s" % (name[:44], str(e)[:160]), flush=True)
        except Exception as e:                                          # noqa: BLE001
            import traceback
            drops.append((name, "ERROR %s: %s" % (type(e).__name__, e)))
            print("ERROR %-44s %s: %s" % (name[:44], type(e).__name__, str(e)[:160]), flush=True)
            traceback.print_exc()
    print("\nbuilt %d, dropped %d" % (ok, len(drops)))
    with open(os.path.join(a.out, "_graft_log%s.json" % ("" if not a.shard else "_" + a.shard.replace("/", "of"))),
              "w", encoding="utf-8") as fh:
        json.dump({"built": ok, "dropped": [{"name": n, "why": w} for n, w in drops]}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
