#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\wd_sub_local.py (= /sp/v6/wall_local/wd_sub_local.py in the build containers)
"""wd_sub_local.py (s2_harden_wall_bake, D15 for item e): the donor subclavian stretches' wall detail, v5 rule /
stage-1 plateau / local, and the identity outside the zones.

wd_proxy bins (/v6w/wd/<subclavian_v5rule | subclavian | subclavian_local>/<unit>_<side>.json; each along its own
VMTK line, prep_siphon 1 mm), 'core' as stage 1's wd_sub_collect (without the first / last 8 mm and +-6 mm about the
ostium); zones = local_sub's subclavian zones (w per v5-rule subclavian station). Bins are mapped to the v5-rule line
by position; a bin is OUTSIDE when w == 0 at its centre and at every v5-rule bin within 3 mm.
Also the junction QC per side (bottleneck ratio v5-rule / plateau / local) and the surface identity checks
(local_qc.json).    python3 wd_sub_local.py -> /v6w/local/wd_sub_local.json    (neve-build-stagea)"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from graft_siphon import prep_siphon  # noqa: E402

TAGS = {"v5rule": ("subclavian_v5rule", "/v6w/subclavian_v5rule"), "plateau": ("subclavian", "/v6w/subclavian"),
        "local": ("subclavian_local", "/v6w/subclavian_local")}
GUARD = 3


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def side(u, sd):
    v = "rsub" if sd == "R" else "lsub"
    zf = "/v6w/subclavian_local/%s/sub_zones.json" % u
    if not os.path.exists(zf):
        return None
    z = json.load(open(zf)).get(sd)
    st = json.load(open("/v6w/subclavian_v5rule/%s/stations_rays.json" % u))[v]
    w = np.asarray(z["w_sub_station"], float)
    Ps = np.asarray(st["P"], float)[:len(w)]
    tw = cKDTree(Ps)
    req = [e for e in json.load(open("/v6w/subclavian/req/%s.json" % u)) if e["side"] == sd][0]
    C = np.asarray(req["centerline_ras_mm"], float)
    O = np.array([np.interp(float(req["ostium_s_mm"]), arclen(C), C[:, k]) for k in range(3)])
    out = {"zones": z["sub"]}
    lines = {}
    for t, (tag, root) in TAGS.items():
        f = "/v6w/wd/%s/%s_%s.json" % (tag, u, sd)
        vf = os.path.join(root, u, "%s_stageA_vmtk.json" % v)
        if not (os.path.exists(f) and os.path.exists(vf)):
            continue
        d = json.load(open(f))
        vj = json.load(open(vf))
        p, _ = prep_siphon(np.asarray(vj["points"], float), np.asarray(vj["radii"], float), 1.0)
        s = arclen(p)
        ar, a3 = np.asarray(d["bins_1mm"]["area"], float), np.asarray(d["bins_1mm"]["a03"], float)
        nb = len(ar)
        cen = np.minimum(np.arange(nb) + 0.5, s[-1])
        Cb = np.stack([np.interp(cen, s, p[:, k]) for k in range(3)], 1)
        s_o = float(s[int(np.argmin(np.linalg.norm(p - O, axis=1)))])
        core = (cen >= 8.0) & (cen <= s[-1] - 8.0) & (np.abs(cen - s_o) > 6.0)
        wb = w[tw.query(Cb)[1]]
        lines[t] = (Cb, ar, a3, core, wb)
        out[t] = {"whole": float(a3.sum() / max(ar.sum(), 1e-9)), "core": float(a3[core].sum() / max(ar[core].sum(), 1e-9)),
                  "zone_core": float(a3[core & (wb > 0)].sum() / ar[core & (wb > 0)].sum()) if (core & (wb > 0)).any() else None}
    if "v5rule" in lines and "local" in lines:
        C5, a5, b5, core5, w5 = lines["v5rule"]
        near = np.zeros(len(w5), bool)
        for i in np.nonzero(w5 > 0)[0]:
            near[max(0, i - GUARD):i + GUARD + 1] = True
        outside = ~near
        Cl, al, bl, corel, wl = lines["local"]
        dd, j = cKDTree(Cl).query(C5[outside])
        f5 = b5[outside] / np.maximum(a5[outside], 1e-9)
        fl = bl[j] / np.maximum(al[j], 1e-9)
        out["identity_outside"] = {"bins_outside": int(outside.sum()), "bins": len(w5),
                                   "bin_frac_absdiff_max": float(np.abs(fl - f5).max()) if outside.any() else 0.0,
                                   "bin_frac_absdiff_med": float(np.median(np.abs(fl - f5))) if outside.any() else 0.0,
                                   "bins_identical_frac": float(np.mean((np.abs(bl[j] - b5[outside]) < 1e-6) & (np.abs(al[j] - a5[outside]) < 1e-6))) if outside.any() else 1.0,
                                   "frac_outside_v5rule": float(b5[outside].sum() / max(a5[outside].sum(), 1e-9)),
                                   "frac_outside_local": float(bl[j].sum() / max(al[j].sum(), 1e-9))}
    q = json.load(open("/v6w/subclavian_local/%s/local_qc.json" % u))["sides"].get(sd, {})
    x = "r" if sd == "R" else "l"
    out["surface_identity"] = {t: {k: (q.get(t) or {}).get(k) for k in ("v5rule_recompute_equals_file", "vertices_outside_zones",
                                                                        "outside_vertices_equal_v5rule_file", "triangles_outside",
                                                                        "triangles_outside_equal_v5rule", "largest_choice_differs_outside_vox",
                                                                        "topology")}
                               for t in (v, "%sjunction" % x)}
    jb = {}
    for t, f in (("v5rule", "/v6w/subclavian_v5rule/%s/seg_qc.json" % u), ("plateau", "/v6w/subclavian/%s/seg_qc.json" % u),
                 ("local", "/v6w/subclavian_local/%s/seg_qc.json" % u)):
        if os.path.exists(f):
            a = ((json.load(open(f)).get("v6") or {}).get("junction_bridge") or {}).get("%sjunction" % x) or {}
            jb[t] = (a.get("after") or {}).get("bottleneck_ratio")
    out["junction_bottleneck_ratio"] = jb
    return out


def main():
    rows = {}
    for u in sorted(os.listdir("/v6w/subclavian_local")):
        if not os.path.exists("/v6w/subclavian_local/%s/.seg_done" % u):
            continue
        for sd in ("R", "L"):
            try:
                r = side(u, sd)
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)[:300]}
            if r:
                rows["%s:%s" % (u, sd)] = r
                print(u, sd, json.dumps({k: r.get(k) for k in ("v5rule", "plateau", "local", "identity_outside", "junction_bottleneck_ratio")}),
                      flush=True)

    def q(v):
        v = np.asarray([x for x in v if x is not None], float)
        return [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] + [round(float(v.min()), 4), round(float(v.max()), 4)] if len(v) else None
    summ = {"sides": len(rows)}
    for t in TAGS:
        summ[t + "_core"] = q([(r.get(t) or {}).get("core") for r in rows.values()])
        summ[t + "_zone_core"] = q([(r.get(t) or {}).get("zone_core") for r in rows.values()])
    io = [r["identity_outside"] for r in rows.values() if "identity_outside" in r]
    summ["identity_outside"] = {"sides": len(io), "bin_frac_absdiff_max": max(x["bin_frac_absdiff_max"] for x in io) if io else None,
                                "frac_outside_absdiff_max": max(abs(x["frac_outside_local"] - x["frac_outside_v5rule"]) for x in io) if io else None}
    summ["sub_zone_core_mm"] = q([r["zones"]["core_mm"] for r in rows.values() if "zones" in r])
    summ["junction_bottleneck_ratio_local_lt_0.5"] = [k for k, r in rows.items() if (r.get("junction_bottleneck_ratio") or {}).get("local") is not None
                                                      and r["junction_bottleneck_ratio"]["local"] < 0.5]
    os.makedirs("/v6w/local", exist_ok=True)
    json.dump({"summary": summ, "sides": rows}, open("/v6w/local/wd_sub_local.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
