#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\graft_v5_reg.py (= /sp/v5_va/graft_v5_reg.py in the build containers)
"""graft_v5_reg.py -- v5 REGISTRY graft: strategy B for ANY donor record (graft_v5_reg_donor.py), with the
feasibility report's fixes ("Fixes before scaling") and every v4 gate.

    host subclavian (kept byte for byte) --[ORIGIN SEAM: stub inside the parent]--> donor VA, ostium -> splice in V3
        --[SPLICE: v4 repair ladder, TopBrain piece Kabsch-FITTED onto the donor's own distal VA]--> v4's TopBrain piece

What is imported UNCHANGED (the prototype record /sp/v5_va/graft_v5_va.py and the repo's graft_v4 / graft_three /
graft_siphon): base_pose (weighted Kabsch onto the v4 VA corridor, origin weight 10), poses (rigid rotations about O_h,
3..15 deg about x / y / z), compose (kept subclavian + stub + placed donor [0, splice] + TopBrain piece, 1 mm resample),
graft_splice / tb_rotation (TB_PLACE "fit", RAMP_RULE "v4"), side_setup, others_for, gate_va (0.6 mm clearance to every
other branch incl. the stub and the ramps, the parent after exit, the other VA and itself; exit <= 12 mm, no re-entry),
Neighbours, corridor_misfit, forward_chord, d_chord, G4.join_kink (largest 1 mm step turn within +-3 mm), G4.tb_part
(the TopBrain piece exactly as v4 sourced it: same patient, same mirror).

What is NEW (report "Fixes before scaling" / "Take-off (corrected)"):
  * DONOR REGISTRY: the donor is a side record (centerline, MISR / calibre, O_v on its own subclavian axis, 5 mm forward
    subclavian chord, landmarks, real_from_mm, wall class); real-surface donors declare MISR radii (the inscribed radius,
    as every v4 section), tube donors their CT-half-max (or MISR x 1.10) calibre and bake with real_surface false.
  * HANDEDNESS "follow" (default): a mirrored ("_L") anatomy takes the donor's OPPOSITE VA, x-mirrored, so the whole
    posterior circulation is one mirror image (v4's rule for TopBrain parts); native anatomies take the same side.
  * ORIGIN-SEAM TURN GATE: the largest 1 mm-step turn within +-3 mm of O_h and of the placed ostium <= 60 deg
    (v4's join rule; ungated in the prototype: 22 / 223 failed).
  * COURSE-PLAUSIBILITY GATE (clearance gates pass absurd placements): course RMS to the v4 corridor <= 20 mm (21
    matched normalised-arclength points, origins pinned); blind end moved vs v4's <= 60 mm; take-off at 10 mm within
    [20, 120] deg; the VA's 10 mm chord rises >= 15 deg.
  * HOST PARENT TANGENT: the 5 mm forward chord of the host subclavian from O_h (the prototype used 5 points = 8.7 / 7.1 mm).
  * SPLICE IN V3: splice on the donor within [C2, dura] (its landmarks; fallback [100, L - 10]), length-matched
    (donor length - TopBrain piece) and moved by v4's ladder inside that window.
  * SPLICE CALIBRE RULE: q = TopBrain join radius / donor radius 4 mm before the splice (both as graft_splice floors and
    caps them); pass 1 accepts max(q, 1/q) <= 1.25 only; pass 2 (only if pass 1 found nothing) <= 1.6, flagged.
  * TAKE-OFF RECORD: donor native vs placed take-off at 5 / 10 mm (VA chord from the ostium vs the 5 mm forward parent
    chord, host and donor alike), delta, takeoff_not_preserved (|delta| > 15 deg), elevations, parent-tangent mismatch.
  * Per-gate record of the BASE candidate (pose 0, splice length-matched, no trim) -- for pass rates per gate -- and of
    every rejected candidate's first failing gate.
  * --compat: the prototype's settings (old host chord, no new gates, splice [100, L - 10], no calibre rule, native
    handedness) -- the regression check against graft/dry_all_takeoffrec.json.

Modes (eve-training-fixed):
    graft_v5_reg.py --unit vmr0248 [--dry-dir DIR] [--workers 6] [--only A,B] [--force]   dry-run ONE donor unit
        against every anatomy (per-anatomy JSON in DIR/<unit>/<anatomy>.json; existing files are skipped)
    graft_v5_reg.py --plan PLAN.json --dry-dir DIR [--workers 6]       dry-run the per-anatomy plans (assignment output)
    graft_v5_reg.py --plan PLAN.json --out DIR [--only A]              WRITE anatomies (copies of v4 + new RVA / LVA
        curves + provenance "v5_va" + xform vaC_R / vaC_L) for bake_v5.py
A plan is {"anatomies": {name: {"right": {"donor_id", "donor_side", "mirror"} | null, "left": ... | null}}}; a null
side keeps v4's VA (flagged).
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, "/repo/carotid_tools")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/sp/v5_va")

import graft_three as G3  # noqa: E402
import graft_v4 as G4  # noqa: E402
import graft_v5_va as P  # noqa: E402   (prototype; imported unchanged)
from graft_siphon import arclength, pair_gaps, prep_siphon, read_curve, unit, write_curve  # noqa: E402

VERSION = "graft_v5_reg 1.1 (2026-10-02; v5_assign: calibre at the cut splice, min ostium advance, plan handedness check, assignment provenance)"
DONOR_ROOT = "/repo/carotid_data/va_sources/v5_reg/donors"
V4_ROOT = "/repo/carotid_data/anatomies_v4"
STEP = P.STEP
NEW_CLEAR_MM = P.NEW_CLEAR_MM            # 0.6 (v4)
JOIN_MAX_KINK = P.JOIN_MAX_KINK          # 60 (v4)
EXIT_MAX_MM = P.EXIT_MAX_MM              # 12 (prototype)
VA_MIN_R = P.VA_MIN_R
VA_TAIL_MM = P.VA_TAIL_MM
VA_MIN_LEN_MM = P.VA_MIN_LEN_MM
TAKEOFF_TANGENT_MM = 5.0
TAKEOFF_TOL_DEG = 15.0                   # |native - placed| above this -> takeoff_not_preserved
# new gates (report, "Fixes before scaling")
ORIGIN_TURN_MAX_DEG = 60.0
COURSE_RMS_MAX_MM = 20.0
TIP_MOVE_MAX_MM = 60.0
TAKEOFF_BAND_DEG = (20.0, 120.0)
RISE_MIN_DEG = 15.0
CALIBRE_PREF = 1.25
CALIBRE_MAX = 1.6
SPLICE_FALLBACK_MIN_MM = 100.0           # prototype SPLICE_MIN_MM, used when the donor has no C2 landmark
UP = np.array([0.0, 0.0, 1.0])
RVA, LVA = G4.RVA, G4.LVA
MIRROR = np.array([-1.0, 1.0, 1.0])
GATES = ("rotation_proper", "origin_turn", "course_rms", "takeoff_band", "rise", "splice_window", "calibre",
         "join_kink", "tip_move", "clearance")

CFG = {"compat": False, "hand": "follow", "donor_root": DONOR_ROOT}


class Unresolvable(Exception):
    pass


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def elevation(v):
    return 90.0 - angle(v, UP)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ donors ----
_DONOR_CACHE = {}


def load_side(donor_id, side, mirror):
    """(points, radii, meta) of one donor VA side, ostium -> distal, 1 mm (prep_siphon, as the prototype), mirrored
    in x when `mirror` (follow handedness)."""
    key = (donor_id, side, bool(mirror))
    if key in _DONOR_CACHE:
        return _DONOR_CACHE[key]
    f = os.path.join(CFG["donor_root"], donor_id, "%s.json" % side)
    rec = jload(f)
    c = rec["centerline"]
    pts = np.asarray(c["points"], float)
    if rec["real_surface"]:
        rad, rkind = np.asarray(c["radii_misr"], float), c["radius_kind"]
    else:
        rad = np.asarray(c["radii_calibre"] if c.get("radii_calibre") is not None else c["radii_misr"], float)
        rkind = c.get("calibre_kind") or c["radius_kind"]
    p, r = prep_siphon(pts, rad, STEP)
    lm = rec.get("landmarks") or {}
    s_raw = arclength(pts)
    s_p = arclength(p)
    # landmarks are mm along the record's raw centerline; map onto the prepped line by nearest point
    lmap = {}
    for k in ("C6", "C2", "C1", "dura"):
        v = lm.get(k)
        if v is None:
            lmap[k] = None
            continue
        q = np.array([np.interp(float(v), s_raw, pts[:, i]) for i in range(3)])
        lmap[k] = float(s_p[int(np.argmin(np.linalg.norm(p - q, axis=1)))])
    M = MIRROR if mirror else np.ones(3)
    sub = rec["subclavian"]
    meta = {"donor_id": donor_id, "side": side, "mirror": M.tolist(), "record": f, "source": rec["source"],
            "licence": rec["licence"], "licence_file": rec.get("licence_file"), "wall_class": rec["wall_class"],
            "real_surface": bool(rec["real_surface"]), "real_from_mm": float(rec.get("real_from_mm") or 0.0),
            "why_tube": rec.get("why_tube"), "surface": (rec.get("surface") or {}).get("path"),
            "surface_md5": (rec.get("surface") or {}).get("md5"), "radius_kind": rkind,
            "O_v": (np.asarray(rec["ostium"]["O_v"], float) * M).tolist(),
            "t_sub": (np.asarray(sub["t_central"], float) * M).tolist(),
            "t_sub_fwd": (np.asarray(sub["t_fwd_5mm"], float) * M).tolist(),
            "takeoff_native": {"5": float(rec["takeoff_native"]["5"]), "10": float(rec["takeoff_native"]["10"])},
            "subclavian_elevation_native": float(sub["elevation_fwd5_deg"]),
            "landmarks_prepped_mm": lmap, "len_mm": float(s_p[-1]), "qc_verdict": rec["qc"]["verdict"],
            "coarseness": rec["coarseness"].get("score"), "contamination_zones": rec.get("contamination_zones", []),
            "surface_genus": (rec.get("surface") or {}).get("genus_measure_json"),
            "patient_key": rec.get("patient_key"), "built": rec["builder"]["built"],
            # v5_assign (1.1): reserve condition (ia172_172 L: start >= 3 mm past the arch-origin tissue jump) and the
            # final eligible list's facts for provenance
            "min_ostium_advance_mm": float(rec.get("min_ostium_advance_mm") or 0.0),
            "eligible": {k: (rec.get("eligible") or {}).get(k) for k in (
                "class", "rank", "patient_key", "dataset_ref", "licence", "non_commercial", "arterial_derived",
                "attributions", "scanner", "acquisition_voxel_mm", "provenance_flags", "side_flags", "qc_status",
                "provisional_only", "wall_truncated", "v1_rescue_used", "flag_level")} if rec.get("eligible") else None,
            "coarseness_detail": {k: rec["coarseness"].get(k) for k in ("score", "score_def", "tally_contact_free_u03",
                                                                         "tally_all_bins_u03", "builder_score")},
            "surface_seg_tag": (rec.get("surface") or {}).get("seg_tag")}
    if meta["eligible"] is not None:
        meta["eligible"]["eye"] = [{k: e.get(k) for k in ("verdict", "by", "when")}
                                   for e in ((rec.get("eligible") or {}).get("eye") or [])]
    out = (p * M, r, meta)
    _DONOR_CACHE[key] = out
    return out


def side_for(anat_side, anatomy_mirrored, hand):
    """Which donor side feeds this host side, and mirrored or not (the follow rule)."""
    if hand == "follow" and anatomy_mirrored:
        return ("L" if anat_side == "right" else "R"), True
    return ("R" if anat_side == "right" else "L"), False


# --------------------------------------------------------------- per side ----
def splice_window(meta):
    L_d = meta["len_mm"]
    if CFG["compat"]:
        return P.SPLICE_MIN_MM, L_d - VA_TAIL_MM, "prototype [100, L - 10]"
    lm = meta["landmarks_prepped_mm"]
    lo = lm.get("C2") if lm.get("C2") is not None else SPLICE_FALLBACK_MIN_MM
    hi = lm.get("dura") if lm.get("dura") is not None else L_d - VA_TAIL_MM
    hi = min(hi, L_d - VA_TAIL_MM)
    how = "[C2, dura]" if lm.get("C2") is not None and lm.get("dura") is not None else "fallback (landmark missing)"
    return float(lo), float(hi), how


def calibre_ratio(dp, dr, splice, tb_r, tb_p):
    """max(q, 1/q), q = TopBrain join radius (graft_splice: r at 4 mm, floored, capped 1.3 x median) / donor radius
    4 mm before the splice (floored)."""
    ds = arclength(dp)
    r_keep = max(float(np.interp(max(splice - P.VA_ANCHOR_MM, 0.0), ds, dr)), VA_MIN_R)
    ts = arclength(tb_p)
    tr = np.maximum(np.asarray(tb_r, float), VA_MIN_R)
    cap = P.VA_BLOB_FACTOR * float(np.median(tr))
    r_join = min(float(np.interp(min(P.VA_ANCHOR_MM, ts[-1]), ts, tr)), cap)
    q = r_join / r_keep
    return max(q, 1.0 / q), q


def origin_turns(p, st, ostium_at):
    s_oh = float(arclength(st["sub_p"])[-1])
    return {"at_O_h": G4.join_kink(p, s_oh), "at_ostium": G4.join_kink(p, ostium_at)}


def pose_checks(R, st, dp, meta, ext, corridor):
    """Pose-level gates (independent of splice and TopBrain trim). Returns (dict of gate -> (pass, value), extras)."""
    O_v = np.asarray(meta["O_v"])
    placed = (dp - O_v) @ R.T + st["O_h"]
    d10 = P.d_chord(placed, 10.0)
    tk10 = angle(d10, st["t_sub"])
    rms = P.corridor_misfit(R, ext, corridor, weighted=False)
    det = float(np.linalg.det(R))
    g = {"rotation_proper": (abs(det - 1.0) < 1e-6, det),
         "course_rms": (rms <= COURSE_RMS_MAX_MM, rms),
         "takeoff_band": (TAKEOFF_BAND_DEG[0] <= tk10 <= TAKEOFF_BAND_DEG[1], tk10),
         "rise": (elevation(d10) >= RISE_MIN_DEG, elevation(d10))}
    if CFG["compat"]:
        g = {k: ((True, v[1]) if k != "rotation_proper" else v) for k, v in g.items()}
    return g, placed


def first_fail(g):
    for k in GATES:
        if k in g and not g[k][0]:
            return k
    return None


OSTIUM_ADVANCE_STEPS_MM = G4.DONOR_TRIM_STEPS_MM     # (0..5 mm): v4's "a donor that opens on a turn: advance its start"


def advanced(dp, dr, meta, a):
    """The donor VA started `a` mm later (1 mm samples): the stub then runs from O_h to the new start. Landmarks,
    length, real_from and the native take-off are re-referenced to the new start."""
    if a == 0:
        return dp, dr, meta
    m = dict(meta)
    dp_a, dr_a = dp[a:], dr[a:]
    m["len_mm"] = float(arclength(dp_a)[-1])
    m["landmarks_prepped_mm"] = {k: (None if v is None else v - a) for k, v in meta["landmarks_prepped_mm"].items()}
    m["real_from_mm"] = max(0.0, meta["real_from_mm"] - a)
    t = np.asarray(meta["t_sub_fwd"])
    m["takeoff_native"] = {"5": angle(P.d_chord(dp_a, 5.0), t), "10": angle(P.d_chord(dp_a, 10.0), t)}
    return dp_a, dr_a, m


def place_side(side, st, dp, dr, meta, tb, v4tip, par, others, record_base=True):
    """_search, repeated with the donor start advanced 1..5 mm ONLY while the origin turn fails AT THE OSTIUM (that
    turn is the donor's own: a rigid pose cannot change it -- e.g. AortaSeg Dissec_01 L reads 68-70 deg in every pose)."""
    fails_all, tried_all, base0, n_all = Counter(), [], None, 0
    a_min = float(meta.get("min_ostium_advance_mm") or 0.0)
    steps = (0,) if CFG["compat"] else tuple(a for a in OSTIUM_ADVANCE_STEPS_MM if a >= a_min)
    if not steps:
        steps = (int(np.ceil(a_min)),)
    for a in steps:
        dp_a, dr_a, m_a = advanced(dp, dr, meta, a)
        res = _search(side, st, dp_a, dr_a, m_a, tb, v4tip, par, others, record_base=(base0 is None and record_base))
        base0 = base0 if base0 is not None else res.get("base")
        fails_all.update(res["fails"])
        tried_all += ["adv %d: %s" % (a, t) for t in res["tried"][:4]]
        n_all += res["n_eval"]
        if res["ok"]:
            res.update(advance=a, meta=m_a, dp=dp_a, dr=dr_a, base=base0, fails=fails_all, tried=tried_all, n_eval=n_all)
            return res
        if not res["fails"].get("origin_turn_ostium"):
            break
    res.update(advance=None, base=base0, fails=fails_all, tried=tried_all, n_eval=n_all)
    return res


def _search(side, st, dp, dr, meta, tb, v4tip, par, others, record_base=True):
    """Search pose x splice x TopBrain trim (v4's repair ladder) for one VA. Returns a dict with the chosen candidate
    (or none), the base candidate's per-gate record and the counts of first failing gates."""
    tbp0, tbr0, tmeta = tb
    tds0 = arclength(tbp0)
    corridor = st["v4_va"][0][st["i_o"]:]
    ext = np.vstack([np.asarray(meta["O_v"])[None, :], dp])
    R0 = P.base_pose(ext, corridor)
    L_d = float(arclength(dp)[-1])
    lo, hi, how = splice_window(meta)
    fails = Counter()
    tried = []
    base = None
    n_eval = 0
    pose_cache = {}
    origin_failed = set()
    for pass_ in ((1,) if CFG["compat"] else (1, 2)):
        for ang, ax, R in P.poses(R0):
            pk = (ang, ax)
            if pk not in pose_cache:
                pose_cache[pk] = pose_checks(R, st, dp, meta, ext, corridor)
            pg, placed = pose_cache[pk]
            if base is None and ang == 0.0 and record_base:
                base = base_record(st, R, dp, dr, meta, tb, v4tip, par, others, lo, hi, pg)
            f = first_fail(pg)
            if f is not None:
                if pass_ == 1:
                    fails[f] += 1
                    tried.append("pose %+.0f%s: %s %.1f" % (ang, ax or "", f, pg[f][1]))
                continue
            if pk in origin_failed:
                continue
            pose_origin_failed = False
            for dc in P.VA_CUT_STEPS:
                for tr in P.DONOR_TRIM_STEPS_MM:
                    tb_p, tb_r = tbp0[tds0 >= tr], tbr0[tds0 >= tr]
                    Ltb = float(arclength(tb_p)[-1])
                    if Ltb < VA_MIN_LEN_MM:
                        break
                    splice = float(np.clip(L_d - Ltb, lo, hi)) + dc
                    if splice < lo or splice > hi:
                        if pass_ == 1:
                            fails["splice_window"] += 1
                        break
                    cq, q = calibre_ratio(dp, dr, splice, tb_r, tb_p)
                    if not CFG["compat"]:
                        if pass_ == 1 and cq > CALIBRE_PREF:
                            fails["calibre"] += 1
                            continue                       # a TopBrain trim changes the join radius: next trim
                        if pass_ == 2 and (cq <= CALIBRE_PREF or cq > CALIBRE_MAX):
                            continue                       # pass-1 candidates were already evaluated
                    p, r, rec = P.compose(st, R, dp, dr, meta, splice, tb_p, tb_r)
                    n_eval += 1
                    if not CFG["compat"]:
                        # v5_assign (1.1; verify_graft_reg item 4): compose cuts at the next 1 mm donor sample, so the
                        # calibre rule is re-applied at the splice actually cut
                        cq, q = calibre_ratio(dp, dr, rec["splice_on_donor_mm"], tb_r, tb_p)
                        if (pass_ == 1 and cq > CALIBRE_PREF) or (pass_ == 2 and cq > CALIBRE_MAX):
                            if pass_ == 1:
                                fails["calibre"] += 1
                            continue
                    ot = origin_turns(p, st, rec["ostium_at_mm"])
                    if not CFG["compat"] and max(ot.values()) > ORIGIN_TURN_MAX_DEG:
                        fails["origin_turn"] += 1
                        if ot["at_ostium"] > ORIGIN_TURN_MAX_DEG:
                            fails["origin_turn_ostium"] += 1
                        tried.append("pose %+.0f%s: origin turn %.0f" % (ang, ax or "", max(ot.values())))
                        pose_origin_failed = True
                        origin_failed.add(pk)
                        break
                    kink = G4.join_kink(p, rec["cut_at_mm"])
                    if kink > JOIN_MAX_KINK:
                        fails["join_kink"] += 1
                        tried.append("pose %+.0f%s splice %+.0f trim %.0f: join %.0f" % (ang, ax or "", dc, tr, kink))
                        continue
                    tipmv = float(np.linalg.norm(p[-1] - v4tip))
                    if not CFG["compat"] and tipmv > TIP_MOVE_MAX_MM:
                        fails["tip_move"] += 1
                        tried.append("pose %+.0f%s splice %+.0f: tip moved %.0f" % (ang, ax or "", dc, tipmv))
                        break
                    g = P.gate_va(p, r, st["i_o"], par, others)
                    if g["ok"]:
                        return {"ok": True, "p": p, "r": r, "rec": rec, "gate": g, "kink": kink, "pose": (ang, ax),
                                "R": R, "splice_move": dc, "trim": tr, "Ltb": Ltb, "calibre": (cq, q, pass_),
                                "origin_turns": ot, "tip_move": tipmv, "pose_gates": pg, "n_eval": n_eval,
                                "fails": fails, "tried": tried, "base": base, "window": (lo, hi, how), "R0": R0}
                    fails["clearance"] += 1
                    tried.append("pose %+.0f%s splice %+.0f trim %.0f: %s" % (ang, ax or "", dc, tr, g.get("why")))
                    break
                if pose_origin_failed:
                    break
    return {"ok": False, "fails": fails, "tried": tried, "base": base, "n_eval": n_eval, "window": (lo, hi, how), "R0": R0}


def base_record(st, R, dp, dr, meta, tb, v4tip, par, others, lo, hi, pg, cand=None):
    """Every gate evaluated on the BASE candidate (pose 0, length-matched splice, no trim), whatever fails first."""
    tbp0, tbr0, _ = tb
    out = {k: {"pass": bool(v[0]), "value": float(v[1])} for k, v in pg.items()}
    L_d = float(arclength(dp)[-1])
    if cand is None:
        Ltb = float(arclength(tbp0)[-1])
        splice = float(np.clip(L_d - Ltb, lo, hi))
        cq, q = calibre_ratio(dp, dr, splice, tbr0, tbp0)
        p, r, rec = P.compose(st, R, dp, dr, meta, splice, tbp0, tbr0)
        if not CFG["compat"]:
            cq, q = calibre_ratio(dp, dr, rec["splice_on_donor_mm"], tbr0, tbp0)     # at the cut (1.1)
    else:
        p, r, rec, cq = cand
    ot = origin_turns(p, st, rec["ostium_at_mm"])
    out["origin_turn"] = {"pass": max(ot.values()) <= ORIGIN_TURN_MAX_DEG, "value": max(ot.values()), "detail": ot}
    out["splice_window"] = {"pass": True, "value": rec["splice_on_donor_mm"]}
    out["calibre"] = {"pass": cq <= CALIBRE_MAX, "value": cq, "preferred": cq <= CALIBRE_PREF}
    k = G4.join_kink(p, rec["cut_at_mm"])
    out["join_kink"] = {"pass": k <= JOIN_MAX_KINK, "value": k}
    tm = float(np.linalg.norm(p[-1] - v4tip))
    out["tip_move"] = {"pass": tm <= TIP_MOVE_MAX_MM, "value": tm}
    g = P.gate_va(p, r, st["i_o"], par, others)
    out["clearance"] = {"pass": bool(g["ok"]), "value": float(g.get("clear", -1.0)) if g.get("clear") is not None else None,
                        "why": g.get("why"), "exit_mm": g.get("exit_mm")}
    out["_geometry"] = {"p": p[::2].round(2).tolist(), "splice_on_donor_mm": rec["splice_on_donor_mm"],
                        "cut_at_mm": rec["cut_at_mm"], "ostium_at_mm": rec["ostium_at_mm"]}
    out["all_pass"] = all(v["pass"] for kk, v in out.items() if not kk.startswith("_"))
    return out


# ------------------------------------------------------------------ anatomy ----
def read_anatomy(name):
    v4 = os.path.join(V4_ROOT, name)
    prov4 = jload(os.path.join(v4, "provenance.json"))
    cdir = os.path.join(v4, "Centrelines_comb")
    br = {}
    for f in sorted(os.listdir(cdir)):
        if f.endswith(".json"):
            br[f] = read_curve(os.path.join(cdir, f))
    return prov4, br


def host_setup(side, br):
    st = P.side_setup(side, br, None, None, None)
    if not CFG["compat"]:
        sp = br[P.SUB_R][1] if side == "right" else br[P.SUB_L_OUT][1]
        st["t_sub_proto"] = st["t_sub"]
        st["t_sub"] = P.forward_chord(sp, st["O_h"], TAKEOFF_TANGENT_MM)
    return st


def build(name, plan, dry=True):
    """plan = {"right": {"donor_id", "donor_side", "mirror"} | None, "left": ...}. Returns (out, placed, br, prov4)."""
    t0 = time.time()
    prov4, br = read_anatomy(name)
    mirrored = prov4["handedness"] == "mirrored"
    pid = prov4["patient"]
    fixed = [(G4.short(f), p, r) for f, (_, p, r) in br.items() if f not in (RVA, LVA) and r is not None]
    out = {"name": name, "patient": pid, "handedness": prov4["handedness"], "version": VERSION,
           "compat": CFG["compat"], "hand": CFG["hand"], "sides": {}}
    placed = {}
    v4curves = {"right": br[RVA][1:], "left": br[LVA][1:]}
    for side in ("right", "left"):
        spec = (plan or {}).get(side)
        st = host_setup(side, br)
        vp4, vr4 = st["v4_va"]
        if not spec:
            out["sides"][side] = {"replaced": False, "why": "no donor planned for this side: v4 VA kept (flagged)"}
            placed[side] = None
            continue
        if not CFG["compat"]:
            # v5_assign (1.1; verify_graft_reg item 7): a plan must follow the handedness rule
            want_side, want_mir = side_for(side, mirrored, CFG["hand"])
            if spec["donor_side"] != want_side or bool(spec.get("mirror", False)) != want_mir:
                raise ValueError("plan for %s %s breaks the %s handedness rule: %s %s mirror %s (want %s mirror %s)" % (
                    name, side, CFG["hand"], spec["donor_id"], spec["donor_side"], spec.get("mirror"), want_side, want_mir))
        dp, dr, meta = load_side(spec["donor_id"], spec["donor_side"], spec.get("mirror", False))
        par = P.Neighbours(st["parents"])
        if side == "right":
            # the right VA is placed first: against every branch but both VAs, plus v4's LVA if the left stays v4
            extra = None if (plan or {}).get("left") else (G4.short(LVA), *v4curves["left"])
        else:
            rp = placed.get("right")
            extra = (G4.short(RVA), rp["p"], rp["r"]) if rp else (G4.short(RVA), *v4curves["right"])
        others = P.Neighbours(P.others_for(side, fixed, extra))
        tbp0, tbr0, tmeta = G4.tb_part("va", side, pid, mirrored, "/repo")
        if tbp0 is None:
            out["sides"][side] = {"replaced": False, "why": "no TopBrain piece (%s)" % tmeta.get("why")}
            placed[side] = None
            continue
        res = place_side(side, st, dp, dr, meta, (tbp0, tbr0, tmeta), vp4[-1], par, others)
        rec_side = {"donor_id": spec["donor_id"], "donor_side": spec["donor_side"], "donor_mirror": meta["mirror"],
                    "wall_class": meta["wall_class"], "real_surface": meta["real_surface"],
                    "donor_qc": meta["qc_verdict"], "donor_coarseness": meta["coarseness"],
                    "splice_window_mm": list(res["window"][:2]), "splice_window_rule": res["window"][2],
                    "n_candidates_evaluated": res["n_eval"], "first_fail_counts": dict(res["fails"]),
                    "first_rejections": res["tried"][:8], "base": res["base"],
                    "host_subclavian_elevation_deg": elevation(st["t_sub"]),
                    "donor_subclavian_elevation_deg_native": meta["subclavian_elevation_native"],
                    "tb_patient_side": tmeta["patient_side"], "tb_mirror": tmeta["mirror"], "tb_len_mm": tmeta["len_mm"],
                    "donor_source": meta["source"], "donor_licence": meta["licence"],
                    "donor_eligible": meta.get("eligible"), "donor_coarseness_detail": meta.get("coarseness_detail"),
                    "donor_surface_seg_tag": meta.get("surface_seg_tag"), "donor_surface_genus": meta.get("surface_genus"),
                    "min_ostium_advance_mm": meta.get("min_ostium_advance_mm", 0.0)}
        if not res["ok"]:
            rec_side.update(replaced=False, why="no pose / splice passes every gate (%d evaluated; first fails %s)"
                            % (res["n_eval"], dict(res["fails"])))
            out["sides"][side] = rec_side
            placed[side] = None
            continue
        p, r, rec = res["p"], res["r"], res["rec"]
        dp, dr, meta = res["dp"], res["dr"], res["meta"]
        s = arclength(p)
        io = int(np.searchsorted(s, rec["ostium_at_mm"]))
        i5 = int(np.searchsorted(s, rec["ostium_at_mm"] + 5.0))
        i10 = int(np.searchsorted(s, rec["ostium_at_mm"] + 10.0))
        tk10 = angle(p[i10] - p[io], st["t_sub"])
        nat10 = meta["takeoff_native"]["10"]
        R = np.asarray(rec["R"])
        ext = np.vstack([np.asarray(meta["O_v"])[None, :], dp])
        corridor = vp4[st["i_o"]:]
        placed_full = (dp - np.asarray(meta["O_v"])) @ R.T + st["O_h"]
        cq, q, cpass = res["calibre"]
        rec_side.update({
            "replaced": True, "pose_deg": res["pose"][0], "pose_axis": res["pose"][1], "splice_move_mm": res["splice_move"],
            "tb_start_trim_mm": res["trim"], "splice_on_donor_mm": rec["splice_on_donor_mm"], "cut_at_mm": rec["cut_at_mm"],
            "ostium_at_mm": rec["ostium_at_mm"], "stub_mm": rec["stub_mm"], "join_kink_deg": res["kink"],
            "origin_turn_deg": res["origin_turns"], "tb_len_used_mm": res["Ltb"], "ramp": rec["ramp"],
            "tb_placement": rec["tb_xform"].get("placement"), "gate": res["gate"], "total_mm": float(s[-1]),
            "splice_calibre_ratio": cq, "splice_calibre_q_tb_over_donor": q, "splice_calibre_preferred": bool(cq <= CALIBRE_PREF),
            "splice_calibre_pass": cpass,
            "takeoff_5mm_deg": angle(p[i5] - p[io], st["t_sub"]), "takeoff_10mm_deg": tk10,
            "donor_takeoff_10mm_deg_native": nat10, "donor_takeoff_5mm_deg_native": meta["takeoff_native"]["5"],
            "takeoff_delta_10mm_deg": tk10 - nat10, "takeoff_not_preserved": bool(abs(tk10 - nat10) > TAKEOFF_TOL_DEG),
            "parent_tangent_mismatch_deg": angle(R @ np.asarray(meta["t_sub_fwd"]), st["t_sub"]),
            "va_10mm_elevation_deg": elevation(p[i10] - p[io]),
            "course_rms_to_corridor_mm": P.corridor_misfit(R, ext, corridor, weighted=False),
            "tip_moved_vs_v4_mm": res["tip_move"], "tip_to_donor_confluence_mm": float(np.linalg.norm(p[-1] - placed_full[-1])),
            "rotation_det": float(np.linalg.det(R)), "pose_gates": {k: [bool(v[0]), float(v[1])] for k, v in res["pose_gates"].items()},
            "real_from_mm_past_ostium": meta["real_from_mm"], "ostium_advance_mm": res["advance"],
        })
        out["sides"][side] = rec_side
        placed[side] = {"p": p, "r": r, "rec": rec, "st": st, "meta": meta, "tmeta": tmeta, "trim": res["trim"]}
    # final gates on everything as it will be written (each placed VA against the other VA as finally written)
    finals = {"right": placed["right"], "left": placed["left"]}
    for side in ("right", "left"):
        pl = finals[side]
        if pl is None:
            continue
        oth = "left" if side == "right" else "right"
        op = (finals[oth]["p"], finals[oth]["r"]) if finals[oth] else v4curves[oth]
        other_items = P.others_for(side, fixed, (G4.short(LVA if side == "right" else RVA), *op))
        gf = P.gate_va(pl["p"], pl["r"], pl["st"]["i_o"], P.Neighbours(pl["st"]["parents"]), P.Neighbours(other_items))
        allp = np.vstack([x[1] for x in other_items])
        allr = np.concatenate([x[2] for x in other_items])
        s = arclength(pl["p"])
        e = int(np.searchsorted(s - s[pl["st"]["i_o"]], gf.get("exit_mm", 0.0)))
        exact = float(pair_gaps(pl["p"][e:], pl["r"][e:], allp, allr).min()) if e < len(pl["p"]) else np.inf
        fin = {k_: gf.get(k_) for k_ in ("ok", "clear", "who", "exit_mm", "self_mm", "clear_others_mm", "clear_others_who")}
        fin["exact_all_pairs_from_exit_mm"] = exact
        fin["ok"] = bool(gf.get("ok")) and exact >= NEW_CLEAR_MM - 1e-9
        out["sides"][side]["final"] = fin
        if not fin["ok"]:
            out["sides"][side]["final_fail"] = "final clearance %.2f mm with %s" % (min(gf.get("clear", 9), exact), gf.get("who"))
    tips = {sd: (finals[sd]["p"][-1] if finals[sd] else v4curves[sd][0][-1]) for sd in ("right", "left")}
    out["tip_to_tip_mm"] = float(np.linalg.norm(tips["right"] - tips["left"]))
    out["v4_tip_to_tip_mm"] = float(np.linalg.norm(v4curves["right"][0][-1] - v4curves["left"][0][-1]))
    for side in ("right", "left"):
        sd = out["sides"][side]
        sd["pass"] = bool(sd.get("replaced") and sd.get("final", {}).get("ok"))
    out["seconds"] = round(time.time() - t0, 2)
    return out, placed, br, prov4


# ------------------------------------------------------------------- write ----
def write(name, out_root, out, placed, br, prov4):
    src = os.path.join(V4_ROOT, name)
    dst = os.path.join(out_root, name)
    if os.path.abspath(dst).startswith(os.path.abspath(V4_ROOT)):
        raise SystemExit("refusing to write into anatomies_v4")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "Centrelines_comb"))
    for f in os.listdir(os.path.join(src, "Centrelines_comb")):
        if f not in (RVA, LVA):
            shutil.copy2(os.path.join(src, "Centrelines_comb", f), os.path.join(dst, "Centrelines_comb", f))
    prov = json.loads(json.dumps(prov4))
    xform = prov["xform"]
    v5 = {"version": VERSION, "strategy": "B: donor cervical VA (origin -> splice in V3) + v4's TopBrain distal piece",
          "tb_placement": P.TB_PLACE, "ramp_rule": P.RAMP_RULE, "handedness_rule": CFG["hand"],
          "new_graft_clearance_mm": NEW_CLEAR_MM, "exit_max_mm": EXIT_MAX_MM,
          "gates": {"origin_turn_max_deg": ORIGIN_TURN_MAX_DEG, "course_rms_max_mm": COURSE_RMS_MAX_MM,
                    "tip_move_max_mm": TIP_MOVE_MAX_MM, "takeoff_band_deg": TAKEOFF_BAND_DEG, "rise_min_deg": RISE_MIN_DEG,
                    "calibre_pref": CALIBRE_PREF, "calibre_max": CALIBRE_MAX, "join_max_deg": JOIN_MAX_KINK},
          "seconds": out["seconds"], "licences": {}}
    for side, fname, key, ckey in (("right", RVA, "va_R", "vaC_R"), ("left", LVA, "va_L", "vaC_L")):
        pl = placed[side]
        if pl is None or not out["sides"][side].get("pass"):
            shutil.copy2(os.path.join(src, "Centrelines_comb", fname), os.path.join(dst, "Centrelines_comb", fname))
            v5[side] = dict(out["sides"][side], v4_va_kept=True)
            continue
        write_curve(br[fname][0], os.path.join(dst, "Centrelines_comb", fname), pl["p"], pl["r"])
        rec, meta, tmeta = pl["rec"], pl["meta"], pl["tmeta"]
        old = xform.get(key, {})
        xform[key] = dict(rec["tb_xform"], kind="topbrain", centerline_src=tmeta["centerline_src"], mirror=tmeta["mirror"],
                          surface=tmeta["surface"], cut_mm=rec["cut_at_mm"], donor_start_trim_mm=float(pl["trim"]),
                          v4_cut_mm=old.get("cut_mm"))
        lo = rec["ostium_at_mm"] + meta["real_from_mm"]
        xc = {"R": np.asarray(rec["R"]).tolist(), "origin": list(meta["O_v"]), "anchor": pl["st"]["O_h"].tolist(),
              "mirror": meta["mirror"], "surface": meta["surface"], "surface_md5": meta["surface_md5"],
              "donor_record": meta["record"], "donor_id": meta["donor_id"], "donor_side": meta["side"],
              "kind": "v5 registry donor (%s, %s)" % (meta["source"], meta["wall_class"]), "lo_mm": lo,
              "hi_mm": rec["cut_at_mm"], "branch": "RVA" if side == "right" else "LVA",
              "real_from_mm_past_ostium": meta["real_from_mm"], "real_surface": meta["real_surface"]}
        if not meta["real_surface"]:
            xc["why_tube"] = meta.get("why_tube") or "tube donor"
        xform[ckey] = xc
        v5[side] = out["sides"][side]
        v5["licences"][meta["donor_id"]] = meta["licence"]
        if meta.get("licence_file") and os.path.isfile(meta["licence_file"]):
            shutil.copy2(meta["licence_file"], os.path.join(dst, os.path.basename(meta["licence_file"])))
    # v5_assign (1.1): assignment-level provenance (the plan entry's "v5_assign" block) and a per-anatomy summary
    asg = (out.get("plan") or {}).get("v5_assign")
    if asg:
        v5["assignment"] = asg
    used = {sd: (v5[sd].get("donor_eligible") or {}) for sd in ("right", "left") if not v5[sd].get("v4_va_kept")}
    v5["summary"] = {
        "v4_va_kept": {sd: bool(v5[sd].get("v4_va_kept")) for sd in ("right", "left")},
        "replaced": {sd: not v5[sd].get("v4_va_kept") for sd in ("right", "left")},
        "donor": {sd: ({"donor_id": v5[sd].get("donor_id"), "donor_side": v5[sd].get("donor_side"),
                        "mirrored": v5[sd].get("donor_mirror", [1, 1, 1])[0] < 0, "source": v5[sd].get("donor_source"),
                        "wall_class": v5[sd].get("wall_class"), "real_surface": v5[sd].get("real_surface"),
                        "takeoff_10mm_native_deg": v5[sd].get("donor_takeoff_10mm_deg_native"),
                        "takeoff_10mm_placed_deg": v5[sd].get("takeoff_10mm_deg"),
                        "takeoff_not_preserved": v5[sd].get("takeoff_not_preserved"),
                        "flags": (v5[sd].get("donor_eligible") or {}).get("side_flags")}
                       if not v5[sd].get("v4_va_kept") else None) for sd in ("right", "left")},
        "non_commercial": any(bool(e.get("non_commercial")) for e in used.values()),
        "attributions": sorted({a for e in used.values() for a in (e.get("attributions") or [])}),
        "flags": list((out.get("plan") or {}).get("flags") or [])}
    prov["v5_va"] = v5
    with open(os.path.join(dst, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1, default=_conv)
    if v5["licences"]:
        with open(os.path.join(dst, "LICENSE_v5_donors.txt"), "w", encoding="utf-8") as fh:
            for k, v in v5["licences"].items():
                fh.write("%s: %s\n" % (k, v))
            if v5["summary"]["attributions"]:
                fh.write("\nAttribution required:\n")
                for a_ in v5["summary"]["attributions"]:
                    fh.write("  - %s\n" % a_)
            if v5["summary"]["non_commercial"]:
                fh.write("\nNON-COMMERCIAL: the donor VA was derived through Arterial (CC BY-NC 4.0 weights, PolyForm-NC code).\n")


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# ------------------------------------------------------------------ driver ----
def anatomy_names(only=None):
    names = sorted(d for d in os.listdir(V4_ROOT) if os.path.isdir(os.path.join(V4_ROOT, d, "Centrelines_comb")))
    if only:
        want = set(only.split(","))
        names = [n for n in names if n in want]
    return names


def unit_plan(unit, name, prov):
    """The plan a donor UNIT implies for one anatomy under the handedness rule (sides it lacks stay v4)."""
    rec = jload(os.path.join(CFG["donor_root"], unit, "record.json"))
    mirrored = prov["handedness"] == "mirrored"
    plan = {}
    for side in ("right", "left"):
        ds, mir = side_for(side, mirrored, CFG["hand"])
        s = rec["sides"].get(ds) or {}
        plan[side] = {"donor_id": unit, "donor_side": ds, "mirror": mir} if s.get("built") and s.get("usable") else None
    return plan


def job(args):
    name, plan, outp, write_dir = args
    if outp and os.path.isfile(outp):
        return name, "skip", None
    t0 = time.time()
    try:
        if plan is None:
            prov = jload(os.path.join(V4_ROOT, name, "provenance.json"))
            plan = unit_plan(CFG["unit"], name, prov)
        out, placed, br, prov4 = build(name, plan, dry=write_dir is None)
        out["plan"] = plan
        if write_dir:
            write(name, write_dir, out, placed, br, prov4)
        status = "ok"
    except Exception as e:  # noqa: BLE001
        import traceback
        out = {"name": name, "error": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()[-1500:]}
        status = "error"
    out["wall_s"] = round(time.time() - t0, 2)
    if outp:
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        tmp = outp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1, default=_conv)
        os.replace(tmp, outp)
    return name, status, out


def _init(cfg):
    CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default=None, help="dry-run this donor unit against every anatomy")
    ap.add_argument("--plan", default=None, help="per-anatomy plans (graft_v5_assign.py output)")
    ap.add_argument("--dry-dir", default="/repo/carotid_data/va_sources/v5_reg/dry")
    ap.add_argument("--tag", default=None, help="sub-folder of --dry-dir (default: the unit, or 'plan')")
    ap.add_argument("--out", default=None, help="WRITE anatomies here (with --plan)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compat", action="store_true", help="prototype settings (regression check)")
    ap.add_argument("--hand", default=None, choices=("follow", "native"))
    ap.add_argument("--donor-root", default=DONOR_ROOT)
    a = ap.parse_args()
    cfg = {"compat": a.compat, "hand": a.hand or ("native" if a.compat else "follow"), "donor_root": a.donor_root,
           "unit": a.unit}
    CFG.update(cfg)
    names = anatomy_names(a.only)
    plans = None
    if a.plan:
        pj = jload(a.plan)
        plans = pj["anatomies"]
        names = [n for n in names if n in plans]
    tag = a.tag or (a.unit if a.unit else "plan")
    ddir = os.path.join(a.dry_dir, tag) if not a.out else None
    jobs = []
    for n in names:
        outp = os.path.join(ddir, n + ".json") if ddir else None
        if outp and a.force and os.path.isfile(outp):
            os.remove(outp)
        jobs.append((n, plans[n] if plans else None, outp, a.out))
    t0 = time.time()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
            rows = pool.map(job, jobs, chunksize=1)
    else:
        rows = [job(j) for j in jobs]
    st = Counter(s for _, s, _ in rows)
    npass = Counter()
    for n, s, o in rows:
        if o and "sides" in o:
            for side in ("right", "left"):
                npass[(side, bool(o["sides"][side].get("pass")))] += 1
        if s == "error":
            print("ERROR", n, o.get("error"), flush=True)
    print("%s: %d anatomies, %s, pass R %d / L %d (this run), %.0f s" % (
        tag, len(rows), dict(st), npass[("right", True)], npass[("left", True)], time.time() - t0), flush=True)
    if ddir:
        with open(os.path.join(ddir, "_done.json"), "w") as f:
            json.dump({"tag": tag, "n": len(rows), "status": dict(st), "seconds": round(time.time() - t0, 1),
                       "compat": a.compat, "hand": cfg["hand"], "version": VERSION,
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
