#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\wd_local.py (= /sp/v6/wall_local/wd_local.py in the build containers)
"""wd_local.py (s2_harden_wall_bake, D15 report): wall detail per donor side, v5 / stage-1 plateau / local, split by
the local artefact zones -- the gate "outside the zones the local wall equals v5".

Inputs: wd_proxy.py's per-donor-side 1 mm bins (/v6w/wd/<v5|plateau|local>/<unit>_<side>.json: the donor union
built as the bake builds it -- real surface + MISR tube floored at 1.0 mm, 0.45 mm grid -- measured with v5's exact
ownership, share of wall > 0.3 mm off the tube); each variant's own line (v5 record line / plateau VMTK line / local
spliced line, prep_siphon at 1 mm as wd_proxy); the local zones (/v6w/seg/local/<unit>/zones.json, w per v5 station).
Each bin's centre is placed on its own line, mapped to v5's line arclength (projection) and to the zone weight of its
nearest v5 station. Classes: ZONE (w > 0: core + 2 mm blend), GUARD (w == 0 but within GUARD_MM of a zone bin: the
union's own reach -- the 0.45 mm field and the capsule see the neighbouring zone), OUTSIDE (the rest).
Per v5 donor VA (wd_collect's used stretch [ostium advance + 2, splice - 25] in v5 donor-line mm): frac_0.3 whole /
zone / guard / outside for v5, plateau, local; floored mm (MISR < 1.0) whole / zone / outside. The identity gate: per
OUTSIDE bin, local vs v5 (bins matched by v5 arclength), |a03 - a03_v5| and |area - area_v5|.
    python3 wd_local.py -> /v6w/local/wd_local.json + printed per-donor-side table      (neve-build-meshlab)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from graft_siphon import prep_siphon  # noqa: E402

sys.path.insert(0, "/sp/v6/wall_local")
import local_seg as LS  # noqa: E402

A5 = "/repo/carotid_data/anatomies_v5"
TAGS = ("v5", "plateau", "local", "v5trim")   # v5trim: the control (wd_ctrl.py), v5 with its line's far end trimmed 0.3 mm
GUARD_MM = 3.0
OUT = "/v6w/local/wd_local.json"
WC1 = "/v6w/wd/wd_collect.json"   # stage 1 (read only): v5 measured per donor side


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def line_of(tag, unit, sd):
    rec = json.load(open(os.path.join(LS.R5, unit, "%s.json" % sd)))
    o5 = np.asarray(rec["centerline"]["points"][0], float)
    if tag in ("v5", "v5trim"):
        P, R = np.asarray(rec["centerline"]["points"], float), np.asarray(rec["centerline"]["radii_misr"], float)
        if tag == "v5trim":
            sys.path.insert(0, "/sp/v6/wall_local")
            import wd_ctrl as WC
            P, R = WC.trimmed(P, R)
    else:
        v = "rva" if sd == "R" else "lva"
        vj = json.load(open("/v6w/seg/%s/%s/%s_stageA_vmtk.json" % (tag, unit, v)))
        P, R = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
        if np.linalg.norm(P[-1] - o5) < np.linalg.norm(P[0] - o5):
            P, R = P[::-1].copy(), R[::-1].copy()
    p, r = prep_siphon(P, R, 1.0)
    return p


def side_bins(unit, sd):
    z = json.load(open("/v6w/seg/local/%s/zones.json" % unit)).get(sd)
    if z is None:
        return None
    v = "rva" if sd == "R" else "lva"
    st = json.load(open(os.path.join(LS.D[unit]["v5_seg_dir"], "stations_rays.json")))[v]
    w = np.asarray(z["w_station"], float)
    Ps = np.asarray(st["P"], float)[:len(w)]
    tree = cKDTree(Ps)
    p5 = line_of("v5", unit, sd)
    s5 = arclen(p5)
    out = {"zones": {k: z[k] for k in ("core_runs_s_mm", "core_mm", "weighted_mm", "length_mm")}}
    for t in TAGS:
        f = "/v6w/wd/%s/%s_%s.json" % (t, unit, sd)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        b = d["bins_1mm"]
        p = line_of(t, unit, sd)
        s = arclen(p)
        nb = len(b["area"])
        cen = np.minimum(np.arange(nb) + 0.5, s[-1])
        C = np.stack([np.interp(cen, s, p[:, k]) for k in range(3)], 1)
        wb = w[tree.query(C)[1]]
        # v5 arclength of each bin centre (projection on v5's prepped line, dense)
        sd5 = np.arange(0, s5[-1], 0.05)
        D5 = np.stack([np.interp(sd5, s5, p5[:, k]) for k in range(3)], 1)
        sig = sd5[cKDTree(D5).query(C)[1]]
        out[t] = {"area": np.asarray(b["area"]), "a03": np.asarray(b["a03"]), "fl": np.asarray(b["floored_frac"]), "lum": np.asarray([np.nan if v_ is None else v_ for v_ in b.get("lumen_min_union", [None] * len(b["area"]))], float),
                  "w": wb, "sig": sig}
    # classes on the v5 bins (the reference grid)
    if "v5" in out:
        w5 = out["v5"]["w"]
        zone5 = w5 > 0
        gi = int(round(GUARD_MM))
        near = np.zeros_like(zone5)
        for i in np.nonzero(zone5)[0]:
            near[max(0, i - gi):i + gi + 1] = True
        out["cls5"] = np.where(zone5, 0, np.where(near, 1, 2))   # 0 zone, 1 guard, 2 outside
    return out


def agg(x, m):
    a = x["area"][m].sum()
    return float(x["a03"][m].sum() / a) if a > 0 else None


def main():
    wc1 = json.load(open(WC1)) if os.path.exists(WC1) else {}
    vas = wc1.get("vas") or []
    cache = {}
    rows = []
    ident = []
    for r in vas:
        unit, sd = r["donor"].split(":")
        if (unit, sd) not in cache:
            try:
                cache[(unit, sd)] = side_bins(unit, sd)
            except Exception as e:  # noqa: BLE001
                print("skip", unit, sd, repr(e)[:200], flush=True)
                cache[(unit, sd)] = None
        sb = cache[(unit, sd)]
        if not sb or "v5" not in sb:
            continue
        a, b = r["from"], r["to"]
        row = {"name": r["name"], "va": r["va"], "donor": r["donor"], "from": a, "to": b,
               "v5_measured": r.get("v5_measured_donor_real")}
        for t in TAGS:
            if t not in sb:
                continue
            x = sb[t]
            used = (x["sig"] >= a) & (x["sig"] < b)
            # class of each bin: the class of the nearest v5 bin (by v5 arclength)
            j = np.clip(np.round(x["sig"] - 0.5).astype(int), 0, len(sb["cls5"]) - 1)
            cls = sb["cls5"][j]
            row[t] = agg(x, used)
            row[t + "_zone"] = agg(x, used & (cls == 0))
            row[t + "_guard"] = agg(x, used & (cls == 1))
            row[t + "_outside"] = agg(x, used & (cls == 2))
            row[t + "_floored_mm"] = float(x["fl"][used].sum())
            row[t + "_floored_mm_zone"] = float(x["fl"][used & (cls == 0)].sum())
            row[t + "_floored_mm_outside"] = float(x["fl"][used & (cls == 2)].sum())
            row[t + "_area_outside"] = float(x["area"][used & (cls == 2)].sum())
            lu = x["lum"][used]
            row[t + "_lumen_min_union"] = float(np.nanmin(lu)) if np.isfinite(lu).any() else None
        rows.append(row)
    # per-bin identity outside the zones (each donor side once, over the whole line)
    per_side = {}
    for (unit, sd), sb in cache.items():
        if not sb or "v5" not in sb or "local" not in sb:
            continue
        x5, xl = sb["v5"], sb["local"]
        out5 = np.nonzero(sb["cls5"] == 2)[0]
        jl = cKDTree(xl["sig"][:, None]).query(x5["sig"][out5][:, None])[1]
        f5 = x5["a03"][out5] / np.maximum(x5["area"][out5], 1e-9)
        fl = xl["a03"][jl] / np.maximum(xl["area"][jl], 1e-9)
        da = np.abs(xl["area"][jl] - x5["area"][out5])
        d3 = np.abs(xl["a03"][jl] - x5["a03"][out5])
        cls_l = None
        ident.append(float(np.abs(fl - f5).max()) if len(out5) else 0.0)
        per_side["%s:%s" % (unit, sd)] = {
            "zones": sb["zones"], "bins": len(sb["cls5"]), "bins_zone": int((sb["cls5"] == 0).sum()),
            "bins_guard": int((sb["cls5"] == 1).sum()), "bins_outside": int(len(out5)),
            "outside_bin_frac_absdiff_max": float(np.abs(fl - f5).max()) if len(out5) else 0.0,
            "outside_bin_frac_absdiff_p99": float(np.percentile(np.abs(fl - f5), 99)) if len(out5) else 0.0,
            "outside_bin_area_absdiff_max_mm2": float(da.max()) if len(out5) else 0.0,
            "outside_bins_identical_frac": float(np.mean((d3 < 1e-6) & (da < 1e-6))) if len(out5) else 1.0,
            "outside_frac_v5": float(x5["a03"][out5].sum() / max(x5["area"][out5].sum(), 1e-9)) if len(out5) else None,
            "outside_frac_local": float(xl["a03"][jl].sum() / max(xl["area"][jl].sum(), 1e-9)) if len(out5) else None}
    # per donor side table (median over the v5 VAs that use it)
    by = {}
    for r in rows:
        by.setdefault(r["donor"], []).append(r)

    def med(rr, k):
        v = [r[k] for r in rr if r.get(k) is not None]
        return round(float(np.median(v)), 4) if v else None
    table = {}
    for k, rr in sorted(by.items()):
        table[k] = {"vas": len(rr), "v5_measured": med(rr, "v5_measured"),
                    **{"%s%s" % (t, suf): med(rr, t + suf) for t in TAGS for suf in ("", "_zone", "_outside")},
                    **{"%s_floored_mm" % t: med(rr, t + "_floored_mm") for t in TAGS},
                    **{"%s_floored_mm_zone" % t: med(rr, t + "_floored_mm_zone") for t in TAGS},
                    "local_eq_v5_outside_med_absdiff": med([{"x": abs(r["local_outside"] - r["v5_outside"])} for r in rr
                                                            if r.get("local_outside") is not None and r.get("v5_outside") is not None], "x")}

    def q(v):
        v = np.asarray([x for x in v if x is not None], float)
        return [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] + [round(float(v.min()), 4), round(float(v.max()), 4)] if len(v) else None
    summ = {"vas": len(rows), "guard_mm": GUARD_MM}
    for t in TAGS:
        summ[t] = {"whole": q([r.get(t) for r in rows]), "zone": q([r.get(t + "_zone") for r in rows]),
                   "outside": q([r.get(t + "_outside") for r in rows]),
                   "floored_mm_per_va_med": float(np.median([r.get(t + "_floored_mm", np.nan) for r in rows if t + "_floored_mm" in r])),
                   "vas_using_floor": sum(1 for r in rows if r.get(t + "_floored_mm", 0) > 0),
                   "floored_mm_total": float(sum(r.get(t + "_floored_mm", 0) for r in rows)),
                   "floored_mm_outside_total": float(sum(r.get(t + "_floored_mm_outside", 0) for r in rows)),
                   "lumen_min_union": q([r.get(t + "_lumen_min_union") for r in rows]),
                   "lumen_min_union_lt_0.9": sum(1 for r in rows if (r.get(t + "_lumen_min_union") or 9) < 0.9)}
    od = [abs(r["local_outside"] - r["v5_outside"]) for r in rows if r.get("local_outside") is not None and r.get("v5_outside") is not None]
    oc = [abs(r["v5trim_outside"] - r["v5_outside"]) for r in rows if r.get("v5trim_outside") is not None and r.get("v5_outside") is not None]
    summ["control_v5trim_vs_v5_outside"] = {"per_va_frac_absdiff_max": float(max(oc)) if oc else None,
                                            "per_va_frac_absdiff_med": float(np.median(oc)) if oc else None, "vas": len(oc),
                                            "whole_per_va_absdiff_max": float(max(abs(r["v5trim"] - r["v5"]) for r in rows if r.get("v5trim") is not None)) if oc else None}
    summ["identity_outside"] = {"per_va_frac_absdiff_max": float(max(od)) if od else None,
                                "per_va_frac_absdiff_med": float(np.median(od)) if od else None,
                                "per_bin_frac_absdiff_max_over_sides": float(max(ident)) if ident else None,
                                "sides": len(per_side)}
    summ["sides_local_ge_0.8_of_v5"] = sum(1 for v in table.values() if v.get("local") and v.get("v5_measured")
                                           and v["local"] >= 0.8 * v["v5_measured"])
    summ["sides_local_le_0.51_of_v5"] = sum(1 for v in table.values() if v.get("local") and v.get("v5_measured")
                                            and v["local"] <= 0.51 * v["v5_measured"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summ, "per_donor_side": table, "per_side_identity": per_side, "vas": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("%-26s %3s %7s | %7s %7s %7s | zone %6s %6s %6s | out %6s %6s | fl v5/loc" % (
        "donor side", "VAs", "v5meas", "v5prx", "plat", "local", "v5", "plat", "loc", "v5", "loc"))
    for k, v in table.items():
        f = lambda x: "%.3f" % x if x is not None else "  -  "  # noqa: E731
        print("%-26s %3d %7s | %7s %7s %7s | zone %6s %6s %6s | out %6s %6s | %s/%s" % (
            k, v["vas"], f(v["v5_measured"]), f(v["v5"]), f(v["plateau"]), f(v["local"]), f(v["v5_zone"]),
            f(v["plateau_zone"]), f(v["local_zone"]), f(v["v5_outside"]), f(v["local_outside"]),
            v["v5_floored_mm"], v["local_floored_mm"]))


if __name__ == "__main__":
    main()
