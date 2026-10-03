#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\graft_v5_reg_sep.py (= /sp/v5_va/graft_v5_reg_sep.py in the build containers)
"""graft_v5_reg_sep.py -- graft_v5_reg 1.1 + a SEPARATION OBJECTIVE for the placement (v5_route13, 2026-10-02).

Why. All non-VA inputs of v5 are byte-identical to v4, yet 13 anatomies read right_route_detail < 0.99 of v4: the
donor VA (or its ramp / the re-fitted TopBrain piece) runs inside the ownership reach of the RCCA's REAL surface (the
Zenodo bulb / the TopBrain siphon), so the owned union (sdf_union_v4) refuses real RCCA wall there and the RCCA falls
back to its tube floor on the side facing the VA. User decision 2026-10-02 22:50: MOVE THE VA first (in a real neck
the VA runs in the foramina behind the carotid; the closeness comes from donor / host geometry mismatch).

The objective, from the union's own rule (sdf_union_v4.ownership_margin). A voxel x of an RCCA real section that v4
kept (f_sec > cur, owned at v4's margin m, not suppressed: route13/diag13.py writes the table <name>.rcca_v4.npz)
stays owned in v5 iff, for both VAs,
        f_local(x) - f_VA(x) >= m,   f_VA(x) = max over the VA's K_OTHER nearest 0.25 mm samples of (r_j - |x - c_j|)
(every other term of the rule is unchanged: the RCCA and every non-VA branch are byte-identical to v4). So the VA
wall must keep a clearance of (-f_local(x) + m) from x: the RCCA real wall's stand-out beyond its tube plus the
ownership margin. LOST(candidate) = number of v4-kept RCCA real voxels (lower_R + siphon_R) the candidate VA, with
the other VA as written in anatomies_v5, takes. Exact (the diag's count on the v5 VA reproduces).

Stated diffs against graft_v5_reg 1.1 (every gate, constant and helper is graft_v5_reg's / the prototype's own):
 S1  _search no longer returns the FIRST candidate passing every gate: it ENUMERATES them (per pose, per splice move,
     the first TopBrain trim passing every gate -- the original's inner order), in the original order (pose nearest
     first, splice move nearest first), and returns the one with the smallest LOST, ties broken by the original
     order. Pass 2 of the calibre rule (1.25 < ratio <= 1.6, flagged) is still used only if pass 1 found nothing,
     and the ostium advance still only when the origin turn fails at the ostium (place_side unchanged).
 S2  only the TARGET side(s) use S1 (--target right|left|both). The other side first tries its anatomies_v5
     placement (same ostium advance, pose, splice move, trim -> the same curve): kept if it passes every gate against
     the target as placed; only otherwise graft_v5_reg's own search (first passing candidate). Among target
     candidates with equal LOST, one that leaves the other VA's anatomies_v5 placement valid is preferred. In the
     objective the other VA is the one placed in this build when it is already placed, else anatomies_v5's.
 S3  EXTENDED POSES (--extended): if the best of the 31 standard poses (0, +-3..15 deg about x / y / z) still loses
     voxels, the 300 two-axis compositions rot(b, j) @ rot(a, i) @ R0 (i, j in +-3..15 deg, axis pairs xy, xz, yz)
     are enumerated after them (original order first, so a standard pose wins any tie).
 S4  TOPBRAIN FIT SHIFT (--tb-shift): the re-fitted TopBrain distal piece may be turned about its start (the splice
     anchor) by rot(ax, a) @ R_fit, a in +-3..15 deg about x / y / z, within its own gates (join <= 60 deg, tip move
     <= 60 mm, clearance); enumerated after the unshifted fit (tie -> unshifted).
 S6  --objective net (added after the first bakes: w_012_left mr_021_L read fork 0.9887 with LOST 4.5k because the
     new course also dropped wall v5 had GAINED where v4's host VA had been): per route segment (fork s < 130 mm,
     siphon s >= 130 mm, right_route_detail's own split) net deficit = max(0, lost - gained) / v4-kept voxels there,
     gained = real RCCA voxels v4's VA refused (field would take them, every non-VA term owns them, not in a v4
     suppressed stretch) that the candidate leaves owned; key = (largest segment deficit, LOST, other VA must move,
     original order).
 S7  --pin-pose SIDE: a target side keeps its anatomies_v5 donor placement (ostium advance, pose, splice move); the
     enumeration then varies only the TopBrain fit shift (S4) and trim -- for a VA whose donor stretch is fine for
     the route but whose TopBrain piece competes with the siphon.
 S5  records: provenance v5_va.<side>.separation = {objective, margin, lost (chosen / as in anatomies_v5 /
     original first-passing candidate), candidates evaluated, chosen order index, pose, closest approach to the RCCA
     over the contested stretch (centre / wall), rule clearance slack}; VERSION string.

    python3 graft_v5_reg_sep.py --plan PLAN.json --out DIR --only NAME --target right --npz-dir D [--extended]
        [--tb-shift] [--dry-dir DIR]        (eve-training-fixed; same donor plan entries as plan_v5.json)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/repo/carotid_tools")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/sp/v5_va")

import graft_v5_reg as GR  # noqa: E402
from graft_siphon import arclength  # noqa: E402
P, G4 = GR.P, GR.G4
VERSION = "graft_v5_reg_sep 1.1 (2026-10-03; v5_route13: separation objective S1-S7 on graft_v5_reg 1.1)"
A5 = "/repo/carotid_data/anatomies_v5"
K_OTHER = 16            # sdf_union_v4.K_OTHER
SAMPLE_MM = 0.25        # sdf_union_v4.SAMPLE_MM
SEP = {"obj": None, "target": set(), "extended": False, "tb_shift": False, "other": {}, "log": {}, "objective": "lost",
       "pin_sides": set(), "pin": {}}
_TB_EXTRA = {"R": None}


def dense(p, r, step=SAMPLE_MM):
    """sdf_union_v4._dense."""
    p, r = np.asarray(p, float), np.asarray(r, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < step:
        return p, r, s
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, p[:, i]) for i in range(3)], 1), np.interp(q, s, r), q


class SepObjective:
    """The v4-kept RCCA real voxels and the rule's threshold t(x) = f_local(x) - m (lost iff f_VA(x) > t(x))."""

    def __init__(self, npz, secs=("lower_R", "siphon_R"), sup4=None):
        """sup4: v4's suppressed stretches {section: [(lo, hi)]} (RCCA arclength) -- voxels there can never be
        gained. Besides the v4-kept voxels (is4) the table holds the GAIN candidates: real RCCA voxels the field
        would take in v4 and every non-VA term owns (A >= m), refused in v4 only by v4's VA."""
        z = np.load(npz)
        self.m = float(z["margin"][0])
        X, T, S, D, sec, I4 = [], [], [], [], [], []
        for nm in secs:
            if nm + "__coords" not in z.files:
                continue
            u = z[nm + "__use4"].astype(bool)
            s_ = z[nm + "__s_star"].astype(float)
            sp = np.zeros(len(u), bool)
            for lo, hi in (sup4 or {}).get(nm, []):
                sp |= (s_ >= lo) & (s_ <= hi)
            pot = ~u & (z[nm + "__A"].astype(float) >= self.m) & ~sp
            keep = u | pot
            X.append(z[nm + "__coords"][keep].astype(float))
            T.append(z[nm + "__f_local"][keep].astype(float) - self.m)
            S.append(s_[keep])
            D.append(z[nm + "__depth"][keep].astype(float))
            I4.append(u[keep])
            sec += [nm] * int(keep.sum())
        self.X, self.t, self.s, self.depth = np.vstack(X), np.concatenate(T), np.concatenate(S), np.concatenate(D)
        self.is4 = np.concatenate(I4)
        self.sec = np.array(sec)
        self.seg = np.where(self.s < 130.0, "fork", "siphon")          # right_route_detail's own split
        self.n4 = {g: int((self.is4 & (self.seg == g)).sum()) for g in ("fork", "siphon")}
        self.reach = float(max(0.0, -self.t.min()))        # largest stand-out + m
        self.n = len(self.X)
        self.f_other = np.full(self.n, -np.inf)
        self.vtree = cKDTree(self.X)

    def f_va(self, p, r):
        """f_VA at every voxel: the rule's K_OTHER nearest dense samples of one VA branch. Exact where it can matter:
        f_VA <= r_max - d_nearest, so only voxels within r_max + reach of a sample can have f_VA > t (the others get
        -inf, which is below every t)."""
        dp, dr, ds = dense(p, r)
        rad = float(dr.max()) + self.reach + 0.5
        sub = dp[::2]                                         # 0.5 mm: +0.25 mm covers the skipped samples
        cnt = self.vtree.query_ball_point(sub, rad + 0.26, return_length=True)
        f = np.full(self.n, -np.inf)
        s_at = np.full(self.n, np.nan)
        hit = np.nonzero(cnt)[0]
        if not len(hit):
            return f, s_at
        lists = self.vtree.query_ball_point(sub[hit], rad + 0.26)
        cand = np.unique(np.concatenate([np.asarray(l_, dtype=np.int64) for l_ in lists]))
        tree = cKDTree(dp)
        k = min(K_OTHER, len(dp))
        d, nn = tree.query(self.X[cand], k=k)
        if d.ndim == 1:
            d, nn = d[:, None], nn[:, None]
        v = dr[nn] - d
        j = np.argmax(v, axis=1)
        f[cand] = v[np.arange(len(cand)), j]
        s_at[cand] = ds[nn[np.arange(len(cand)), j]]
        return f, s_at

    def set_other(self, p, r):
        self.f_other, _ = self.f_va(p, r)

    def lost(self, p, r):
        f, s_at = self.f_va(p, r)
        taken = (np.maximum(f, self.f_other) > self.t)
        L = taken & self.is4
        G = ~taken & ~self.is4
        mine = L & (f >= self.f_other)
        slack = np.where(self.is4, self.t - f, np.inf)       # >= 0 everywhere = nothing taken by this VA
        out = {"lost": int(L.sum()), "lost_by_this_va": int(mine.sum()), "gained": int(G.sum()),
               "lost_relief_gt_0.3mm": int((L & (self.depth > 0.3)).sum()),
               "lost_by_section": {nm: int((L & (self.sec == nm)).sum()) for nm in np.unique(self.sec)},
               "min_slack_mm": round(float(slack.min()), 3) if self.n else None}
        # net deficit per route segment (fork s < 130, siphon s >= 130) as a share of v4's kept voxels there
        dfc = {}
        for g in ("fork", "siphon"):
            lg = int((L & (self.seg == g)).sum())
            gg = int((G & (self.seg == g)).sum())
            dfc[g] = round(max(0, lg - gg) / max(self.n4[g], 1), 5)
        out["net_deficit"] = dfc
        out["net_deficit_max"] = max(dfc.values())
        if mine.any():
            out["rcca_s_mm"] = [round(float(self.s[mine].min()), 1), round(float(self.s[mine].max()), 1)]
            out["va_s_mm"] = [round(float(np.nanmin(s_at[mine])), 1), round(float(np.nanmax(s_at[mine])), 1)]
        return out


def closest_to_rcca(p, r, rc, s_rng=None):
    """Closest approach of the VA polyline to the RCCA centreline (dense; over an RCCA arclength range)."""
    rdp, rdr, rds = dense(*rc)
    if s_rng:
        m = (rds >= s_rng[0] - 5) & (rds <= s_rng[1] + 5)
        rdp, rdr, rds = rdp[m], rdr[m], rds[m]
    vdp, vdr, vds = dense(p, r)
    d, j = cKDTree(vdp).query(rdp)
    i = int(np.argmin(d))
    w = d - rdr - vdr[j]
    k = int(np.argmin(w))
    return {"centre_mm": round(float(d[i]), 2), "at_rcca_s": round(float(rds[i]), 1), "at_va_s": round(float(vds[j[i]]), 1),
            "wall_mm": round(float(w[k]), 2), "wall_at_rcca_s": round(float(rds[k]), 1)}


# ---- S4: the TopBrain fit, optionally turned about its start --------------------------------------------------------
_ORIG_TB_ROTATION = P.tb_rotation


def tb_rotation_shift(donor_p, keep_p, target):
    R, w = _ORIG_TB_ROTATION(donor_p, keep_p, target)
    if _TB_EXTRA["R"] is not None:
        R = _TB_EXTRA["R"] @ R
    return R, w


P.tb_rotation = tb_rotation_shift


def tb_shifts():
    yield 0.0, None, None
    for d in P.POSE_STEPS_DEG:
        for ax_name, ax in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
            for sgn in (1.0, -1.0):
                yield sgn * d, ax_name, P.rot(ax, sgn * d)


def poses_ext(R0):
    """S3: the 31 standard poses, then the two-axis compositions."""
    for t in P.poses(R0):
        yield t + ("std",)
    if not SEP["extended"]:
        return
    axes = (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))
    for ia in range(3):
        for ib in range(ia + 1, 3):
            for da in P.POSE_STEPS_DEG:
                for sa in (1.0, -1.0):
                    for db in P.POSE_STEPS_DEG:
                        for sb in (1.0, -1.0):
                            R = P.rot(axes[ib][1], sb * db) @ P.rot(axes[ia][1], sa * da) @ R0
                            yield (sa * da, "%s%+.0f%s" % (axes[ia][0], sb * db, axes[ib][0]), R, "ext")


# ---- S1: enumerate every candidate passing every gate ----------------------------------------------------------------
def _key(c):
    """Candidate order: objective 'lost' = (LOST, other VA must move, original order); objective 'net' (S6) =
    (largest per-segment net deficit, LOST, other VA must move, original order)."""
    if SEP["objective"] == "net":
        return (c["sep"]["net_deficit_max"], c["sep"]["lost"], c["other_breaks"], c["order"])
    return (c["sep"]["lost"], c["other_breaks"], c["order"])


def _other_breaks(side, p, r):
    """S2: does the other VA, as written in anatomies_v5, still pass graft_v5_reg's clearance gate against this
    candidate? (1 = it would have to be re-placed)."""
    oc = SEP.get("other_check")
    if not oc or oc["side"] == side:
        return 0
    tag = G4.short(GR.RVA if side == "right" else GR.LVA)
    g = P.gate_va(oc["p"], oc["r"], oc["st"]["i_o"], P.Neighbours(oc["st"]["parents"]),
                  P.Neighbours(P.others_for(oc["side"], oc["fixed"], (tag, p, r))))
    return 0 if g["ok"] else 1


def _enumerate(side, st, dp, dr, meta, tb, v4tip, par, others, record_base=True, objective=True, only=None):
    """S1: every candidate passing every gate (per pose, per splice move [, per TopBrain shift]: the first trim
    passing, the original inner order). only = {"pose": (ang, ax), "dc": dc} restricts to one pose / splice move."""
    obj = SEP["obj"] if objective else None
    if obj is not None:
        # the other VA in the objective: as placed in this build if it is among `others`, else anatomies_v5's
        oname = G4.short(GR.LVA if side == "right" else GR.RVA)
        it = [x for x in others.items if x[0] == oname]
        if it:
            obj.set_other(it[0][1], it[0][2])
        elif SEP.get("other_v5", {}).get(side) is not None:
            obj.set_other(*SEP["other_v5"][side])
    tbp0, tbr0, tmeta = tb
    tds0 = arclength(tbp0)
    corridor = st["v4_va"][0][st["i_o"]:]
    ext = np.vstack([np.asarray(meta["O_v"])[None, :], dp])
    R0 = P.base_pose(ext, corridor)
    L_d = float(arclength(dp)[-1])
    lo, hi, how = GR.splice_window(meta)
    fails, tried, base, n_eval = GR.Counter(), [], None, 0
    pose_cache, origin_failed = {}, set()
    cands = []
    order = 0
    shifts = list(tb_shifts()) if (SEP["tb_shift"] and objective) else [(0.0, None, None)]
    for pass_ in (1, 2):
        if pass_ == 2 and cands:
            break                                   # calibre pass 2 only if pass 1 found nothing (unchanged rule)
        pose_iter = poses_ext(R0) if objective else [t + ("std",) for t in P.poses(R0)]
        for ang, ax, R, fam in pose_iter:
            pk = (ang, ax)
            if only is not None and (abs(ang - only["pose"][0]) > 1e-9 or ax != only["pose"][1]):
                continue
            if pk not in pose_cache:
                pose_cache[pk] = GR.pose_checks(R, st, dp, meta, ext, corridor)
            pg, placed = pose_cache[pk]
            if base is None and ang == 0.0 and record_base:
                base = GR.base_record(st, R, dp, dr, meta, tb, v4tip, par, others, lo, hi, pg)
            f = GR.first_fail(pg)
            if f is not None:
                if pass_ == 1:
                    fails[f] += 1
                continue
            if pk in origin_failed:
                continue
            pose_origin_failed = False
            for dc in P.VA_CUT_STEPS:
                if only is not None and dc != only["dc"]:
                    continue
                for tsh_a, tsh_ax, tshR in (shifts if fam == "std" else shifts[:1]):
                    _TB_EXTRA["R"] = tshR
                    got = None
                    for tr in P.DONOR_TRIM_STEPS_MM:
                        tb_p, tb_r = tbp0[tds0 >= tr], tbr0[tds0 >= tr]
                        Ltb = float(arclength(tb_p)[-1])
                        if Ltb < GR.VA_MIN_LEN_MM:
                            break
                        splice = float(np.clip(L_d - Ltb, lo, hi)) + dc
                        if splice < lo or splice > hi:
                            if pass_ == 1:
                                fails["splice_window"] += 1
                            break
                        cq, q = GR.calibre_ratio(dp, dr, splice, tb_r, tb_p)
                        if pass_ == 1 and cq > GR.CALIBRE_PREF:
                            fails["calibre"] += 1
                            continue
                        if pass_ == 2 and (cq <= GR.CALIBRE_PREF or cq > GR.CALIBRE_MAX):
                            continue
                        p, r, rec = P.compose(st, R, dp, dr, meta, splice, tb_p, tb_r)
                        n_eval += 1
                        cq, q = GR.calibre_ratio(dp, dr, rec["splice_on_donor_mm"], tb_r, tb_p)
                        if (pass_ == 1 and cq > GR.CALIBRE_PREF) or (pass_ == 2 and cq > GR.CALIBRE_MAX):
                            if pass_ == 1:
                                fails["calibre"] += 1
                            continue
                        ot = GR.origin_turns(p, st, rec["ostium_at_mm"])
                        if max(ot.values()) > GR.ORIGIN_TURN_MAX_DEG:
                            fails["origin_turn"] += 1
                            if ot["at_ostium"] > GR.ORIGIN_TURN_MAX_DEG:
                                fails["origin_turn_ostium"] += 1
                            pose_origin_failed = True
                            origin_failed.add(pk)
                            break
                        kink = G4.join_kink(p, rec["cut_at_mm"])
                        if kink > GR.JOIN_MAX_KINK:
                            fails["join_kink"] += 1
                            continue
                        tipmv = float(np.linalg.norm(p[-1] - v4tip))
                        if tipmv > GR.TIP_MOVE_MAX_MM:
                            fails["tip_move"] += 1
                            break
                        g = P.gate_va(p, r, st["i_o"], par, others)
                        if g["ok"]:
                            got = {"ok": True, "p": p, "r": r, "rec": rec, "gate": g, "kink": kink, "pose": (ang, ax),
                                   "R": R, "splice_move": dc, "trim": tr, "Ltb": Ltb, "calibre": (cq, q, pass_),
                                   "origin_turns": ot, "tip_move": tipmv, "pose_gates": pg, "window": (lo, hi, how),
                                   "R0": R0, "pose_family": fam, "tb_shift": (tsh_a, tsh_ax),
                                   "tb_extra_R": None if tshR is None else tshR.tolist()}
                        else:
                            fails["clearance"] += 1
                        break
                    if pose_origin_failed:
                        break
                    if got is not None:
                        got["order"] = order
                        if obj is not None:
                            got["sep"] = obj.lost(got["p"], got["r"])
                            got["other_breaks"] = _other_breaks(side, got["p"], got["r"])
                        cands.append(got)
                        if obj is None:
                            break
                    order += 1
                _TB_EXTRA["R"] = None
                if pose_origin_failed or (obj is None and cands):
                    break
            if obj is None and cands:
                break
            # S3: the extended poses only if no standard candidate loses nothing
            if obj is not None and fam == "std" and ang == -P.POSE_STEPS_DEG[-1] and ax == "z" and cands and \
                    min(_key(c)[0] for c in cands) == 0:
                break
    _TB_EXTRA["R"] = None
    if not cands:
        return {"ok": False, "fails": fails, "tried": tried, "base": base, "n_eval": n_eval, "window": (lo, hi, how),
                "R0": R0}
    if obj is None:
        best = cands[0]
        best.update(n_eval=n_eval, fails=fails, tried=tried, base=base)
        return best
    best = min(cands, key=_key)
    first = min(cands, key=lambda c: c["order"])
    best.update(n_eval=n_eval, fails=fails, tried=tried, base=base)
    best["sep_summary"] = {"objective": "v4-kept RCCA real voxels (lower_R + siphon_R) refused under sdf_union_v4's "
                                        "ownership rule (f_local - f_VA < m) by this VA and the other VA (as placed in "
                                        "this build, else as in anatomies_v5); then the other VA kept; then the original order",
                           "margin_mm": obj.m, "rcca_voxels_v4_kept": obj.n, "reach_mm": round(obj.reach, 2),
                           "candidates_passing_every_gate": len(cands), "chosen_order": best["order"],
                           "chosen": best["sep"], "chosen_other_va_must_move": bool(best["other_breaks"]),
                           "original_first_passing": first["sep"],
                           "original_first_pose": [first["pose"][0], first["pose"][1]],
                           "original_first_splice_move": first["splice_move"],
                           "pose_family": best["pose_family"], "tb_shift_deg_axis": list(best["tb_shift"]),
                           "n_candidates_lost_0": int(sum(1 for c in cands if c["sep"]["lost"] == 0)),
                           "n_candidates_lost_0_other_kept": int(sum(1 for c in cands if c["sep"]["lost"] == 0
                                                                     and not c["other_breaks"])),
                           "objective_mode": SEP["objective"], "net_deficit_chosen": best["sep"].get("net_deficit"),
                           "lost_distribution_lowest": sorted(int(c["sep"]["lost"]) for c in cands)[:20]}
    SEP["log"][side] = best["sep_summary"]
    return best


def _search_sep(side, st, dp, dr, meta, tb, v4tip, par, others, record_base=True):
    if side not in SEP["target"] or SEP["obj"] is None:
        return _ORIG_SEARCH(side, st, dp, dr, meta, tb, v4tip, par, others, record_base)
    return _enumerate(side, st, dp, dr, meta, tb, v4tip, par, others, record_base, objective=True)


def place_side_sep(side, st, dp, dr, meta, tb, v4tip, par, others, record_base=True):
    """S2: a NON-target side first tries its anatomies_v5 placement (same ostium advance, pose, splice move, trim):
    kept if it passes every gate against the target as placed; otherwise graft_v5_reg's own search."""
    pin = SEP.get("pin", {}).get(side)
    if side in SEP["target"] and pin is not None:
        # S7: the donor keeps its anatomies_v5 pose / splice move / ostium advance; only the TopBrain fit shifts (S4)
        a = int(pin["adv"] or 0)
        dp_a, dr_a, m_a = GR.advanced(dp, dr, meta, a)
        res = _enumerate(side, st, dp_a, dr_a, m_a, tb, v4tip, par, others, record_base, objective=True,
                         only={"pose": pin["pose"], "dc": pin["dc"]})
        if res.get("ok"):
            res.update(advance=a, meta=m_a, dp=dp_a, dr=dr_a)
        return res
    k = SEP.get("keep", {}).get(side)
    if side not in SEP["target"] and k is not None:
        a = int(k["adv"] or 0)
        dp_a, dr_a, m_a = GR.advanced(dp, dr, meta, a)
        res = _enumerate(side, st, dp_a, dr_a, m_a, tb, v4tip, par, others, record_base, objective=False,
                         only={"pose": k["pose"], "dc": k["dc"]})
        if res.get("ok") and res["trim"] == k["tr"]:
            res.update(advance=a, meta=m_a, dp=dp_a, dr=dr_a)
            SEP["kept_other"][side] = True
            return res
        SEP["kept_other"][side] = False
    return _ORIG_PLACE(side, st, dp, dr, meta, tb, v4tip, par, others, record_base)


_ORIG_SEARCH = GR._search
GR._search = _search_sep
_ORIG_PLACE = GR.place_side
GR.place_side = place_side_sep


def run(name, plan_entry, out_root, npz, targets, dry_out=None):
    from graft_siphon import read_curve
    SEP["target"] = set(targets)
    rep4 = json.load(open(os.path.join(GR.V4_ROOT, name, "mesh_v4.json"), encoding="utf-8"))
    sup4 = {k: [tuple(x) for x in v] for k, v in (rep4.get("suppressed_stretches_mm") or {}).items()}
    SEP["obj"] = SepObjective(npz, sup4=sup4)
    SEP["log"] = {}
    # the other VA as written in anatomies_v5 (the objective counts both VAs)
    v5c = os.path.join(A5, name, "Centrelines_comb")
    cur = {sd: read_curve(os.path.join(v5c, f)) for sd, f in (("right", GR.RVA), ("left", GR.LVA))}
    prov5 = json.load(open(os.path.join(A5, name, "provenance.json"), encoding="utf-8"))
    SEP["other_v5"] = {"right": (cur["left"][1], cur["left"][2]), "left": (cur["right"][1], cur["right"][2])}
    base_v5 = {}
    for sd in ("right", "left"):
        SEP["obj"].set_other(*SEP["other_v5"][sd])
        base_v5[sd] = SEP["obj"].lost(cur[sd][1], cur[sd][2])
    # S2: the non-target side's anatomies_v5 placement (kept if it passes against the moved target)
    SEP["keep"], SEP["kept_other"], SEP["other_check"], SEP["pin"] = {}, {}, None, {}
    _, br0 = GR.read_anatomy(name)
    fixed0 = [(G4.short(f), p_, r_) for f, (_, p_, r_) in br0.items() if f not in (GR.RVA, GR.LVA) and r_ is not None]
    for sd in ("right", "left"):
        s5 = prov5["v5_va"].get(sd) or {}
        if sd in SEP["pin_sides"] and "pose_deg" in s5:
            SEP["pin"][sd] = {"pose": (float(s5["pose_deg"]), s5["pose_axis"]), "dc": s5["splice_move_mm"],
                              "tr": s5["tb_start_trim_mm"], "adv": s5.get("ostium_advance_mm") or 0}
        if sd not in targets and s5.get("replaced", True) and not s5.get("v4_va_kept") and "pose_deg" in s5:
            SEP["keep"][sd] = {"pose": (float(s5["pose_deg"]), s5["pose_axis"]), "dc": s5["splice_move_mm"],
                               "tr": s5["tb_start_trim_mm"], "adv": s5.get("ostium_advance_mm") or 0}
            SEP["other_check"] = {"side": sd, "p": cur[sd][1], "r": cur[sd][2], "st": GR.host_setup(sd, br0),
                                  "fixed": fixed0}
    plan = {"right": plan_entry["right"], "left": plan_entry["left"]}
    out, placed, br, prov4 = GR.build(name, plan, dry=out_root is None)
    out["plan"] = plan_entry
    rc = br[[f for f in br if f.endswith("RCCA.mrk.json")][0]][1:]
    sepinfo = {}
    for sd in ("right", "left"):
        pl = placed.get(sd)
        if pl is None:
            continue
        same = bool(len(pl["p"]) == len(cur[sd][1]) and np.allclose(pl["p"], cur[sd][1], atol=1e-3)
                    and np.allclose(pl["r"], cur[sd][2], atol=1e-3))
        rec = {"target": sd in targets, "identical_to_anatomies_v5": same,
               "v5_placement_kept": SEP["kept_other"].get(sd)}
        if sd in targets:
            rec.update(SEP["log"].get(sd, {}))
            rec["lost_as_in_anatomies_v5"] = base_v5[sd]
            ch = rec.get("chosen", {})
            rec["closest_rcca_chosen"] = closest_to_rcca(pl["p"], pl["r"], rc, ch.get("rcca_s_mm") or base_v5[sd].get("rcca_s_mm"))
            rec["closest_rcca_v5"] = closest_to_rcca(cur[sd][1], cur[sd][2], rc, ch.get("rcca_s_mm") or base_v5[sd].get("rcca_s_mm"))
            # course change vs the v5 placement (same donor): RMS of matched normalised arclength points
            q = np.linspace(0, 1, 41)
            a5, b5 = P.at_frac(cur[sd][1], q), P.at_frac(pl["p"], q)
            rec["course_shift_vs_v5_rms_mm"] = round(float(np.sqrt(((a5 - b5) ** 2).sum(1).mean())), 2)
            rec["course_shift_vs_v5_max_mm"] = round(float(np.sqrt(((a5 - b5) ** 2).sum(1)).max()), 2)
        out["sides"][sd]["separation"] = rec
        sepinfo[sd] = rec
    out["sep_version"] = VERSION
    if out_root:
        GR.write(name, out_root, out, placed, br, prov4)
        pp = os.path.join(out_root, name, "provenance.json")
        prov = json.load(open(pp, encoding="utf-8"))
        prov["v5_va"]["route13"] = {"tool": VERSION, "targets": list(targets), "separation": sepinfo,
                                    "npz": npz, "extended_poses": SEP["extended"], "tb_shift": SEP["tb_shift"]}
        prov["v5_va"]["version"] = prov["v5_va"]["version"] + " | " + VERSION
        with open(pp + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(prov, fh, indent=1, default=GR._conv)
        os.replace(pp + ".tmp", pp)
    if dry_out:
        with open(dry_out + ".tmp", "w") as fh:
            json.dump({k: v for k, v in out.items()}, fh, indent=1, default=GR._conv)
        os.replace(dry_out + ".tmp", dry_out)
    return out, sepinfo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/repo/carotid_data/va_sources/v5_reg/plan_v5.json")
    ap.add_argument("--donor-root", default="/repo/carotid_data/va_sources/v5_reg/donors_v5")
    ap.add_argument("--only", required=True)
    ap.add_argument("--target", default="right", choices=("right", "left", "both"))
    ap.add_argument("--npz-dir", default="/sp/v5_va/route13/diag")
    ap.add_argument("--out", default=None, help="write the anatomy here (scratch); omit for a dry run")
    ap.add_argument("--dry-out", default=None, help="write the build record (JSON) here")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--tb-shift", action="store_true")
    ap.add_argument("--objective", default="lost", choices=("lost", "net"))
    ap.add_argument("--pin-pose", default="", help="S7: target sides (comma list) keeping their v5 donor pose; only the TopBrain fit shifts")
    a = ap.parse_args()
    GR.CFG.update({"compat": False, "hand": "follow", "donor_root": a.donor_root, "unit": None})
    SEP["extended"], SEP["tb_shift"], SEP["objective"] = a.extended, a.tb_shift, a.objective
    SEP["pin_sides"] = set(x for x in a.pin_pose.split(",") if x)
    if a.out and (os.path.abspath(a.out).startswith("/repo/carotid_data/anatomies_v")):
        raise SystemExit("refusing to write into anatomies_v4 / anatomies_v5")
    plans = json.load(open(a.plan))["anatomies"]
    targets = ["right", "left"] if a.target == "both" else [a.target]
    t0 = time.time()
    out, sep = run(a.only, plans[a.only], a.out, os.path.join(a.npz_dir, a.only + ".rcca_v4.npz"), targets, a.dry_out)
    for sd in ("right", "left"):
        s = out["sides"][sd]
        print("SEP %s %s: pass %s pose %s%s splice %s | %s" % (
            a.only, sd, s.get("pass"), s.get("pose_deg"), s.get("pose_axis") or "", s.get("splice_move_mm"),
            json.dumps({k: v for k, v in (s.get("separation") or {}).items() if k in (
                "identical_to_anatomies_v5", "chosen", "lost_as_in_anatomies_v5", "original_first_passing",
                "candidates_passing_every_gate", "chosen_order", "pose_family", "tb_shift_deg_axis",
                "closest_rcca_chosen", "closest_rcca_v5", "course_shift_vs_v5_rms_mm")}, default=GR._conv)), flush=True)
    print("done %.0f s" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
