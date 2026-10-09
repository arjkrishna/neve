#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\diag13.py (= /sp/v5_va/route13/diag13.py in the build containers)
"""v5_route13 step 1 (read-only on anatomies_v4 / anatomies_v5): WHICH VA piece takes WHICH RCCA real wall.

For one anatomy, v4's carotid sections are applied to v4's own tube field exactly as the owned union does
(bake_v5b.UnionCache._prep / _apply logic, v4 margin, v4 suppressed stretches, union order), and for every RCCA
real-surface voxel (sections on branch RCCA: lower_R = the Zenodo fork/bulb slice, siphon_R = the TopBrain siphon;
eca_R / ecaext_R on RECA reported beside) we record
    use4   v4 widened it (f_sec > cur and owned at v4's margin, not suppressed)
    A      f_local - max(f_far, f_other WITHOUT the two VAs)   (sdf_union_v4.ownership_margin's terms)
    f_local, s_star (RCCA arclength of the owner), f_RVA / f_LVA in v4 and in v5 (K_OTHER nearest, as the rule)
The ownership decision does not depend on the field, only on the voxel and the centrelines, so v5's decision at
the same voxels is exact: owned5 = A >= m and f_local - f_VA5 >= m for both VAs (and v5's suppressed stretches).
    lost   = use4 & ~owned5          (real RCCA wall v4 kept and v5 refuses)
    gained = would4 & ~owned4 & owned5   (v4's VA had refused it; v5 lets it in)
For the lost voxels: which VA wins (max tube value) and where on it (VA arclength -> piece: host subclavian kept /
stub / donor V1 tube / donor lower VA (real) / ramp / TopBrain distal), the RCCA arclength range, and the
closest approach of that VA piece to the RCCA centreline there (centre-to-centre and wall-to-wall, declared radii),
v5 and v4. Saves the voxel table (for graft_v5_reg_sep's separation objective) as <out>/<name>.rcca_v4.npz.

    python3 diag13.py <name> [<name> ...]          (neve-build-meshlab; out /sp/v5_va/route13/diag)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
import sdf_union_v4 as U  # noqa: E402

A4 = "/repo/carotid_data/anatomies_v4"
A5 = os.environ.get("A5_DIR", "/repo/carotid_data/anatomies_v5")
OUT = os.environ.get("DIAG_OUT", "/sp/v5_va/route13/diag")
RCCA_SECS = ("lower_R", "siphon_R")
RECA_SECS = ("eca_R", "ecaext_R")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dense(p, r):
    return U._dense(p, r)


def va_tube(coords, p, r):
    """Max tube value r - |x - c| over the K_OTHER nearest 0.25 mm samples of one branch (the rule's f_other term)
    and the arclength of the winning sample."""
    dp, dr, ds = dense(p, r)
    f, nn = U._tube(coords, dp, dr, U.K_OTHER)
    k = np.argmax(f, axis=1)
    return f[np.arange(len(coords)), k], ds[nn[np.arange(len(coords)), k]]


def pieces(root, side):
    """VA arclength boundaries of the pieces of one VA as written in <root> (v5) -> [(name, lo, hi)]."""
    prov = jload(os.path.join(root, "provenance.json"))
    x = prov["xform"]
    tag = "RVA" if side == "right" else "LVA"
    ck, tk = ("vaC_R", "va_R") if side == "right" else ("vaC_L", "va_L")
    if ck not in x:
        cut = float(x[tk]["cut_mm"])
        return [("host VA tube (v4 kept)", 0.0, cut), ("TopBrain distal", cut, 1e9)], {"v4_kept": True, "cut": cut}
    sd = prov["v5_va"][side]
    from eve_bench.dualdevicenav import load_branches
    b = B.branch(load_branches(os.path.join(root, "Centrelines_comb")), tag)
    p = np.asarray(b.coordinates, float)
    s = B.arclen(p)
    o_h = 0.0 if side == "left" else float(s[int(np.argmin(np.linalg.norm(p - np.asarray(x[ck]["anchor"]), axis=1)))])
    ost, lo, cut = float(sd["ostium_at_mm"]), float(x[ck]["lo_mm"]), float(x[tk]["cut_mm"])
    ramp = sd["ramp"]
    blend = -float(ramp["span_mm"][0]) if ramp["side"] == "donor" else 0.0
    out = [("host subclavian (kept)", 0.0, o_h), ("stub (in parent)", o_h, ost)]
    if lo > ost + 1e-6:
        out.append(("donor V1 tube", ost, lo))
    out += [("donor lower VA (real)", lo, cut - blend), ("ramp (donor side)", cut - blend, cut),
            ("TopBrain distal (re-fitted)", cut, 1e9)]
    if ramp["side"] != "donor":
        out[-1] = ("TopBrain distal (re-fitted; ramp on TopBrain side)", cut, 1e9)
    return out, {"o_h": o_h, "ostium": ost, "real_lo": lo, "cut": cut, "ramp_side": ramp["side"], "blend": blend,
                 "donor": "%s %s" % (x[ck]["donor_id"], x[ck]["donor_side"])}


def piece_of(s, pcs):
    for nm, lo, hi in pcs:
        if lo - 1e-6 <= s < hi:
            return nm
    return pcs[-1][0]


def closest(pa, ra, pb, rb):
    """Closest approach of two polylines (dense samples): centre-to-centre, wall-to-wall (declared radii), indices."""
    t = cKDTree(pb)
    d, j = t.query(pa)
    i = int(np.argmin(d))
    w = d - ra - rb[j]
    k = int(np.argmin(w))
    return {"centre_mm": round(float(d[i]), 2), "wall_mm": round(float(w[k]), 2), "ia": i, "jb": int(j[i]),
            "wall_ia": k, "wall_jb": int(j[k])}


def one(name):
    from eve_bench.dualdevicenav import load_branches
    t0 = time.time()
    r4, r5 = os.path.join(A4, name), os.path.join(A5, name)
    rep4 = jload(os.path.join(r4, B.REPORT))
    rep5 = jload(os.path.join(r5, V.REPORT_V5))
    m = rep4["margin_mm"]
    br4 = load_branches(os.path.join(r4, "Centrelines_comb"))
    br5 = load_branches(os.path.join(r5, "Centrelines_comb"))
    s4, s5 = U.Samples(br4), U.Samples(br5)
    secs4, _ = B.sections_for(r4, br4)
    tube4 = set(rep4.get("sections_tube_only") or [])
    car = [s for s in secs4 if not V.is_va(s) and s.name not in tube4]
    sup4 = {k: [tuple(x) for x in v] for k, v in (rep4.get("suppressed_stretches_mm") or {}).items()}
    sup5 = {k: [tuple(x) for x in v] for k, v in (rep5.get("suppressed_stretches_mm") or {}).items()}
    cache = V.UnionCache(br4, s4, car, B.SPACING_MM)
    f = cache.field
    rec = {}
    for sec in car:                        # union order, exactly as _apply
        P = cache.pre.get(sec.name)
        if P is None:
            continue
        g = cache._g(P)
        cur = f.values[g]
        would = P["f_sec"] > cur
        coords = cache._coords(P, np.ones(P["n"], bool))
        keep = sec.name in RCCA_SECS + RECA_SECS
        if keep:
            # every term of ownership_margin at ALL would voxels (the VA terms kept apart)
            w = np.nonzero(would)[0]
            cw = coords[w]
            bi = s4.index_of(sec.branch)
            own = s4.b == bi
            f_o, nn = U._tube(cw, s4.p[own], s4.r[own], U.K_OWN)
            s_nn = s4.s[own][nn]
            best = np.argmax(f_o, axis=1)
            s_star = s_nn[np.arange(len(cw)), best]
            near = np.abs(s_nn - s_star[:, None]) <= U.WINDOW_MM
            f_local = np.where(near, f_o, -np.inf).max(axis=1)
            f_far = np.where(~near, f_o, -np.inf).max(axis=1)
            f_oth = np.full(len(cw), -np.inf)
            va_ids = {s4.index_of("RVA"), s4.index_of("LVA")}
            for j in np.unique(s4.b[~own]):
                if j in va_ids or any(sb in s4.names[j] for sb in sec.siblings):
                    continue
                mm = s4.b == j
                fx, _ = U._tube(cw, s4.p[mm], s4.r[mm], U.K_OTHER)
                f_oth = np.maximum(f_oth, fx.max(axis=1))
            inside = (s_star >= sec.s0 - U.WINDOW_MM) & (s_star <= sec.s0 + U._length(sec.points) + U.WINDOW_MM)
            A_ = np.where(inside, f_local - np.maximum(f_far, f_oth), -np.inf)
            fv4 = {}
            for tag in ("RVA", "LVA"):
                b4 = B.branch(br4, tag)
                fv4[tag] = va_tube(cw, np.asarray(b4.coordinates, float), np.asarray(b4.radii, float))
            fv5 = {}
            for tag in ("RVA", "LVA"):
                b5 = B.branch(br5, tag)
                fv5[tag] = va_tube(cw, np.asarray(b5.coordinates, float), np.asarray(b5.radii, float))
            m4 = np.minimum(A_, f_local - np.maximum(fv4["RVA"][0], fv4["LVA"][0]))
            m5 = np.minimum(A_, f_local - np.maximum(fv5["RVA"][0], fv5["LVA"][0]))
            # check against the rule itself (v4): must be the same margin
            chk = U.ownership_margin(cw[:4000], sec, s4)
            fin = np.isfinite(chk)
            chk_diff = float(np.max(np.abs(chk[fin] - m4[:4000][fin]))) if fin.any() else 0.0

            def supp(sp_, s_):
                o = np.zeros(len(s_), bool)
                for lo, hi in sp_.get(sec.name, []):
                    o |= (s_ >= lo) & (s_ <= hi)
                return o
            own4 = (m4 >= m) & ~supp(sup4, s_star)
            own5 = (m5 >= m) & ~supp(sup5, s_star)
            rec[sec.name] = {"coords": cw.astype(np.float32), "f_local": f_local.astype(np.float32),
                             "A": A_.astype(np.float32), "s_star": s_star.astype(np.float32),
                             "use4": own4, "own5": own5, "fv5": fv5, "fv4": fv4, "m4": m4, "m5": m5,
                             "sup5_only": supp(sup5, s_star) & ~supp(sup4, s_star), "chk_diff": chk_diff,
                             "depth": (P["f_sec"][w] - cur[w]).astype(np.float32), "f_sec": P["f_sec"][w]}
        # apply exactly as _apply (so the next section sees v4's field)
        mm_ = np.full(P["n"], np.nan)
        wv = np.nonzero(would)[0]
        if len(wv):
            mv, _, sv = U.ownership_margin(coords[wv], sec, s4, return_anchor=True)
            owned = np.ones(P["n"], bool)
            owned[wv] = mv >= m
            for lo, hi in sup4.get(sec.name, []):
                owned[wv[(sv >= lo) & (sv <= hi)]] = False
            use = would & owned
            vals = cur.copy()
            vals[use] = P["f_sec"][use]
            f.values[g] = vals
            f.region[g] = True
            if sec.name in rec:
                rec[sec.name]["use4_rule"] = int(use.sum())
    del cache, f
    prov5 = jload(os.path.join(r5, "provenance.json"))
    prov4 = jload(os.path.join(r4, "provenance.json"))
    pcs = {sd: pieces(r5, sd) for sd in ("right", "left")}
    out = {"name": name, "margin_mm": m, "rung_v5": rep5["ladder_v5b"]["rung"], "flags_v5": rep5["ladder_v5b"]["flags"],
           "rrd_v4": rep4["right_route_detail"], "rrd_v5": rep5["right_route_detail"],
           "ratio": {k: {t: round(rep5["right_route_detail"][k][t] / rep4["right_route_detail"][k][t], 4)
                         if rep4["right_route_detail"][k][t] else None for t in rep4["right_route_detail"][k]}
                     for k in rep4["right_route_detail"]},
           "lumen_RCCA_v5": rep5["routes"]["RCCA"].get("lumen_min_mm"),
           "pieces": {sd: {"bounds": [[n_, round(a_, 1), round(min(b_, 1e4), 1)] for n_, a_, b_ in pcs[sd][0]],
                           **pcs[sd][1]} for sd in pcs},
           "donors": {sd: (prov5["v5_va"][sd].get("donor_id"), prov5["v5_va"][sd].get("donor_side"))
                      for sd in ("right", "left")},
           "sections": {}}
    rc5 = B.branch(br5, "RCCA")
    rp, rr = np.asarray(rc5.coordinates, float), np.asarray(rc5.radii, float)
    rdp, rdr, rds = dense(rp, rr)
    save = {}
    for nm, R in rec.items():
        bsec = B.branch(br5, "RCCA" if nm in RCCA_SECS else "RECA")      # closest approach to the section's branch
        rdp, rdr, rds = dense(np.asarray(bsec.coordinates, float), np.asarray(bsec.radii, float))
        use4, own5 = R["use4"], R["own5"]
        lost = use4 & ~own5
        gained = ~R["use4"] & (R["m4"] < m) & own5 & (R["A"] >= m)
        lost_supp = lost & R["sup5_only"]
        lost_own = lost & ~R["sup5_only"]
        sec_out = {"would_voxels": int(len(use4)), "use4": int(use4.sum()), "use4_rule_check": R.get("use4_rule"),
                   "margin_formula_vs_rule_maxdiff": R["chk_diff"],
                   "lost_v4_wall_voxels": int(lost.sum()), "lost_by_ownership": int(lost_own.sum()),
                   "lost_by_v5_suppression": int(lost_supp.sum()), "gained_voxels": int(gained.sum()),
                   "lost_relief_gt_0.3mm": int((lost & (R["depth"] > 0.3)).sum())}
        if lost_own.any():
            w = np.nonzero(lost_own)[0]
            fr, sr = R["fv5"]["RVA"]
            fl, sl = R["fv5"]["LVA"]
            who = np.where(fr[w] >= fl[w], "RVA", "LVA")
            vs = np.where(who == "RVA", sr[w], sl[w])
            pn_all = np.array(["%s: %s" % (who[k_], piece_of(float(vs[k_]), pcs["right" if who[k_] == "RVA" else "left"][0]))
                               for k_ in range(len(w))])
            comp = {}
            for pn in sorted(set(pn_all)):
                sel = pn_all == pn
                ii = w[sel]
                a = np.stack([R["s_star"][ii], vs[sel]], 1)
                lst = a
                comp[pn] = {"voxels": len(lst), "rcca_s_mm": [round(float(a[:, 0].min()), 1), round(float(a[:, 0].max()), 1)],
                            "va_s_mm": [round(float(a[:, 1].min()), 1), round(float(a[:, 1].max()), 1)],
                            "rcca_s_median": round(float(np.median(a[:, 0])), 1)}
                sdk = "right" if pn.startswith("RVA") else "left"
                if "ostium" in pcs[sdk][1]:
                    comp[pn]["va_s_from_ostium_mm"] = [round(float(a[:, 1].min() - pcs[sdk][1]["ostium"]), 1),
                                                       round(float(a[:, 1].max() - pcs[sdk][1]["ostium"]), 1)]
                # closest approach of that VA piece to the RCCA centreline over the contested RCCA stretch
                tag, side = pn.split(":")[0], ("right" if pn.startswith("RVA") else "left")
                for ver, brx in (("v5", br5), ("v4", br4)):
                    bv = B.branch(brx, tag)
                    vdp, vdr, vds = dense(np.asarray(bv.coordinates, float), np.asarray(bv.radii, float))
                    if ver == "v5":
                        msk = (vds >= a[:, 1].min() - 5) & (vds <= a[:, 1].max() + 5)
                    else:
                        msk = np.ones(len(vds), bool)
                    rm = (rds >= a[:, 0].min() - 5) & (rds <= a[:, 0].max() + 5)
                    if msk.sum() < 2 or rm.sum() < 2:
                        continue
                    c = closest(rdp[rm], rdr[rm], vdp[msk], vdr[msk])
                    comp[pn]["closest_" + ver] = {"centre_mm": c["centre_mm"], "wall_mm": c["wall_mm"],
                                                  "at_rcca_s": round(float(rds[rm][c["ia"]]), 1),
                                                  "at_va_s": round(float(vds[msk][c["jb"]]), 1),
                                                  "r_rcca": round(float(rdr[rm][c["ia"]]), 2),
                                                  "r_va": round(float(vdr[msk][c["jb"]]), 2)}
                # the rule's reach: how far v4's real RCCA wall stands out of the RCCA tube there (-f_local), and the
                # VA-wall clearance the rule needs (-f_local + m) vs what it has (-f_VA)
                fva = (fr if tag == "RVA" else fl)[ii]
                comp[pn]["wall_standout_mm_max"] = round(float((-R["f_local"][ii]).max()), 2)
                comp[pn]["needed_va_wall_clearance_mm_max"] = round(float((-R["f_local"][ii] + m).max()), 2)
                comp[pn]["had_va_wall_clearance_mm_min"] = round(float((-fva).min()), 2)
            sec_out["competitors_v5"] = comp
            # 5 mm bins of RCCA arclength
            bins = {}
            for i_ in np.nonzero(lost)[0]:
                b_ = int(np.floor(R["s_star"][i_] / 5.0) * 5)
                bins.setdefault(b_, [0, 0])[0] += 1
            for i_ in np.nonzero(gained)[0]:
                b_ = int(np.floor(R["s_star"][i_] / 5.0) * 5)
                bins.setdefault(b_, [0, 0])[1] += 1
            sec_out["bins_lost_gained"] = {str(k): v for k, v in sorted(bins.items())}
        if gained.any():
            w = np.nonzero(gained)[0]
            sec_out["gained_rcca_s_mm"] = [round(float(R["s_star"][w].min()), 1), round(float(R["s_star"][w].max()), 1)]
        out["sections"][nm] = sec_out
        save[nm] = R
    # the voxel table for the separation objective (RCCA sections only; RECA beside)
    np.savez_compressed(os.path.join(OUT, name + ".rcca_v4.npz"),
                        **{"%s__%s" % (nm, k): v for nm, R in save.items()
                           for k, v in (("coords", R["coords"]), ("f_local", R["f_local"]), ("A", R["A"]),
                                        ("s_star", R["s_star"]), ("use4", R["use4"]), ("depth", R["depth"]),
                                        ("sup5_only", R["sup5_only"]))},
                        margin=np.array([m]), sections=np.array(list(save)))
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("name", "margin_mm", "ratio", "donors")}), flush=True)
    for nm, so in out["sections"].items():
        print("  %s: %s" % (nm, json.dumps({k: v for k, v in so.items() if k not in ("bins_lost_gained",)})), flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:]:
        if os.path.exists(os.path.join(OUT, name + ".json")) and not os.environ.get("FORCE"):
            print("done already", name, flush=True)
            continue
        try:
            one(name)
        except Exception:                                                # noqa: BLE001
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
