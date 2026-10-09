#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tbo\va_narrow\bone_scan.py (= /sp/v5_va/tbo/va_narrow/bone_scan.py in the build containers)
"""va_narrow: is the w_025 / w_027 artefact systematic? (neve-build-stagea; detached, per-donor done files)

For every donor side, every MISR < 1.0 stretch >= 2 mm inside the donor range some v5 VA uses (floor_scan.json), at its
min-MISR point, and at a reference point of the same side (MISR >= 1.3, >= 5 mm from any floored stretch, nearest
within 30 mm): an independent CT lumen radius on the shipped ct.nii.gz -- 64 rays in the normal plane from the
brightest point within 1 mm, a ray ends at the first r where HU < (I_c + I_ring)/2 (soft tissue) or HU > I_c + 150
(a brighter neighbour: bone / dense vein) -> polygon r_eq, and the share of rays ended by the bright rule -- against the
donor stage-A surface's section r_eq and the MISR. CT r_eq >= 1.0 where MISR < 1.0 means the CT lumen is wider than the
graft's floor, i.e. the surface (not the artery) is narrow.
    python3 bone_scan.py   -> tbo/va_narrow/bone_scan/<unit>.json  (skips units already done), then bone_scan.json
"""
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
import pyvista as pv

sys.path.insert(0, "/sp/v5_va/tbo/va_narrow")
import ct_check as C  # noqa: E402

OUT = "/sp/v5_va/tbo/va_narrow/bone_scan"
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"


def ct_req(ct, c, t):
    e1, e2 = C.frame(t)
    u, G = C.plane_grid(c, e1, e2, half=1.0, h=0.1)
    hu = np.nan_to_num(ct(G), nan=-1000)
    from scipy import ndimage
    k = np.unravel_index(np.argmax(ndimage.gaussian_filter(hu, 2.0)), hu.shape)
    c2 = G[k]
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rr = np.arange(0.0, 6.0001, 0.05)
    dirs = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    prof = ct(c2 + rr[None, :, None] * dirs[:, None, :])
    Ic = float(np.nanmedian(prof[:, rr <= 0.5]))
    ring = prof[:, (rr >= 3.0) & (rr <= 5.0)]
    ring = ring[(ring > -200) & (ring < 500)]
    Ib = float(np.nanmedian(ring)) if ring.size else 0.0
    half = 0.5 * (Ic + Ib)
    r_e, bright = np.full(64, rr[-1]), np.zeros(64, bool)
    for j in range(64):
        p = prof[j]
        lo = np.where((p < half) & (rr > 0))[0]
        hi = np.where((p > Ic + 150) & (rr > 0))[0]
        a = lo[0] if len(lo) else 10 ** 6
        b = hi[0] if len(hi) else 10 ** 6
        if min(a, b) < 10 ** 6:
            m = min(a, b)
            r_e[j] = rr[m]
            bright[j] = b < a
    return {"ct_req": C.poly_req(r_e), "bright_ray_frac": float(bright.mean()), "Ic": Ic, "Ib": Ib,
            "centre_shift": float(np.linalg.norm(c2 - c))}


def unit_ct(rec):
    src = rec["surface"]["src"]
    if "/derived/thin_slice/" in src:
        tail = src.split("/derived/thin_slice/")[1].split("/")
        p = "/repo/carotid_data/va_sources/thin_slice/%s/%s/ct.nii.gz" % (tail[0], tail[1])
        return p if os.path.exists(p) else None
    return None


def main():
    fs = json.load(open("/sp/v5_va/tbo/va_narrow/floor_scan.json"))
    maxsp = {}
    for v in fs["vas"]:
        maxsp[v["donor"]] = max(maxsp.get(v["donor"], 0.0), v["splice_on_donor_mm"])
    os.makedirs(OUT, exist_ok=True)
    units = sorted({k.split(":")[0] for k in fs["donors"]})
    for u in units:
        of = os.path.join(OUT, "%s.json" % u)
        if os.path.exists(of):
            continue
        res = {"unit": u, "sides": {}}
        recs = {sd: json.load(open(os.path.join(R5, u, "%s.json" % sd))) for sd in ("R", "L")
                if os.path.exists(os.path.join(R5, u, "%s.json" % sd))}
        ctp = unit_ct(next(iter(recs.values())))
        if ctp is None:
            res["skipped"] = "no thin_slice ct.nii.gz for %s" % next(iter(recs.values()))["surface"]["src"]
            json.dump(res, open(of, "w"), indent=1)
            print(u, res["skipped"], flush=True)
            continue
        im = nib.load(ctp)
        allp = np.vstack([np.asarray(r["centerline"]["points"], float) for r in recs.values()])
        ct = C.Vol(im.dataobj, im.affine, allp.min(0), allp.max(0), pad=10.0)
        for sd, rec in recs.items():
            key = "%s:%s" % (u, sd)
            if key not in maxsp:
                continue
            P = np.asarray(rec["centerline"]["points"], float)
            m = np.asarray(rec["centerline"]["radii_misr"], float)
            s = C.TA.arclen(P)
            surf = pv.read(rec["surface"]["path"]).triangulate()
            floored = m < 1.0
            rows = []
            for st in fs["donors"][key]["stretches"]:
                if st["len_mm"] < 2.0 or st["s_from"] > maxsp[key]:
                    continue
                idx = np.where((s >= st["s_from"] - 1e-6) & (s <= st["s_to"] + 1e-6))[0]
                i = int(idx[np.argmin(m[idx])])
                # reference: MISR >= 1.3 and >= 5 mm from any floored point, nearest within 30 mm
                fl_s = s[floored]
                cand = np.where((m >= 1.3) & (np.abs(s - s[i]) <= 30.0))[0]
                cand = [j for j in cand if np.min(np.abs(fl_s - s[j])) >= 5.0]
                pts = [("floored", i)] + ([("reference", int(min(cand, key=lambda j: abs(s[j] - s[i]))))] if cand else [])
                row = {"s_from": st["s_from"], "s_to": st["s_to"], "len_mm": st["len_mm"],
                       "bright_rays_station_mean": st.get("bright_rays_mean"), "fix8_frac": st.get("fix8_frac")}
                for tag, j in pts:
                    a, b = max(j - 2, 0), min(j + 2, len(P) - 1)
                    t = C.unit(P[b] - P[a])
                    e1, e2 = C.frame(t)
                    r = ct_req(ct, P[j], t)
                    r.update({"s": float(s[j]), "misr": float(m[j]),
                              "stageA_req": C.loop_req(C.section_xy(surf, P[j], t, e1, e2))})
                    row[tag] = r
                rows.append(row)
            res["sides"][sd] = rows
        json.dump(res, open(of, "w"), indent=1)
        print(u, {sd: len(v) for sd, v in res["sides"].items()}, flush=True)
    # collect
    F, Rf = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        d = json.load(open(f))
        for sd, rows in d.get("sides", {}).items():
            for r in rows:
                if "floored" in r:
                    F.append(dict(r["floored"], unit=d["unit"], side=sd, len_mm=r["len_mm"]))
                if "reference" in r:
                    Rf.append(dict(r["reference"], unit=d["unit"], side=sd))

    def q(a):
        a = np.asarray([x for x in a if x is not None], float)
        return [round(float(v), 3) for v in np.percentile(a, [10, 50, 90])] if len(a) else None
    summ = {"floored_points": len(F), "reference_points": len(Rf),
            "floored": {"misr_p10_50_90": q([x["misr"] for x in F]), "stageA_req": q([x["stageA_req"] for x in F]),
                        "ct_req": q([x["ct_req"] for x in F]), "bright_ray_frac": q([x["bright_ray_frac"] for x in F]),
                        "ct_req_ge_1.0": sum(1 for x in F if x["ct_req"] >= 1.0),
                        "ct_over_stageA_ge_1.25": sum(1 for x in F if x["stageA_req"] and x["ct_req"] / x["stageA_req"] >= 1.25),
                        "ct_req_lt_1.0_and_bright_lt_0.1": sum(1 for x in F if x["ct_req"] < 1.0 and x["bright_ray_frac"] < 0.1)},
            "reference": {"misr": q([x["misr"] for x in Rf]), "stageA_req": q([x["stageA_req"] for x in Rf]),
                          "ct_req": q([x["ct_req"] for x in Rf]),
                          "ct_over_stageA": q([x["ct_req"] / x["stageA_req"] for x in Rf if x["stageA_req"]]),
                          "bright_ray_frac": q([x["bright_ray_frac"] for x in Rf])}}
    json.dump({"summary": summ, "floored": F, "reference": Rf}, open("/sp/v5_va/tbo/va_narrow/bone_scan.json", "w"),
              indent=1)
    print(json.dumps(summ, indent=1))
    open(os.path.join(OUT, "ALL_DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
