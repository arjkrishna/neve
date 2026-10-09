#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\sub_ct_gate.py (= /sp/v6/wall_d17/sub_ct_gate.py in the build containers)
"""sub_ct_gate.py (s2_harden_wall_bake, D17): is the artefact gone in the SUBCLAVIAN zones of the D17 local surfaces,
and is the surface v5-rule's outside them?  (neve-build-stagea)

Per donor side with a subclavian zone (d17/local/<unit>/sub_zones.json, core runs): points every 2 mm inside each
core run ("zone"), and reference points every 10 mm on stations with w = 0 at >= 5 mm from any w > 0 station
("outside"). At each point, in the plane normal to the station path (tangent over +-2 mm):
  ct_req   an independent CT lumen read = bone_scan.py's rule (64 rays from the brightest point within 1 mm, a ray ends
           where HU < (I_c + I_ring) / 2 or HU > I_c + 150 (bone / dense neighbour) -> polygon r_eq), with the ring and
           ray length SCALED to the station radius R (ring [R + 1.5, R + 3.5] mm, rays to R + 5 mm; bone_scan's fixed
           3-5 mm ring sits inside a 4-5 mm subclavian lumen) -- recorded also: bright_ray_frac;
  surf_req the section r_eq of the v5-rule / plateau / local subclavian surfaces (ct_check.section_xy / loop_req);
  ratio    ct_req / surf_req (the audit's quantity; ~1.0 = the surface follows the CT lumen).
    python3 sub_ct_gate.py   -> /v6w/d17/sub_ct_gate.json"""
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
import pyvista as pv

sys.path.insert(0, "/sp/v5_va/tbo/va_narrow")
import ct_check as C  # noqa: E402
import vtk  # noqa: E402

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
D17 = "/v6w/d17"
TAGS = ("v5rule", "plateau", "local")


def ct_req(ct, c, t, R):
    e1, e2 = C.frame(t)
    _, G = C.plane_grid(c, e1, e2, half=1.0, h=0.1)
    hu = np.nan_to_num(ct(G), nan=-1000)
    from scipy import ndimage
    k = np.unravel_index(np.argmax(ndimage.gaussian_filter(hu, 2.0)), hu.shape)
    c2 = G[k]
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rr = np.arange(0.0, R + 5.0001, 0.05)
    dirs = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    prof = ct(c2 + rr[None, :, None] * dirs[:, None, :])
    Ic = float(np.nanmedian(prof[:, rr <= 0.5]))
    ring = prof[:, (rr >= R + 1.5) & (rr <= R + 3.5)]
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
    return {"ct_req": C.poly_req(r_e), "bright_ray_frac": float(bright.mean()), "Ic": Ic, "Ib": Ib}


def unit_ct(unit):
    for sd in ("R", "L"):
        f = os.path.join(R5, unit, "%s.json" % sd)
        if os.path.exists(f):
            src = json.load(open(f))["surface"]["src"]
            if "/derived/thin_slice/" in src:
                tail = src.split("/derived/thin_slice/")[1].split("/")
                p = "/repo/carotid_data/va_sources/thin_slice/%s/%s/ct.nii.gz" % (tail[0], tail[1])
                return p if os.path.exists(p) else None
    return None


def q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return {"p10": round(float(np.percentile(a, 10)), 3), "median": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "n": int(len(a))} if len(a) else None


def main():
    rows, skipped = [], {}
    for ud in sorted(glob.glob(os.path.join(D17, "local", "*"))):
        unit = os.path.basename(ud)
        zf, sf = os.path.join(ud, "sub_zones.json"), os.path.join(ud, "subclavian_stations.json")
        if not (os.path.exists(zf) and os.path.exists(sf)):
            continue
        Z, S = json.load(open(zf)), json.load(open(sf))
        sides = [sd for sd in ("R", "L") if sd in Z and (Z[sd]["sub"]["core_runs_s_mm"] or [])]
        if not sides:
            continue
        ctp = unit_ct(unit)
        if ctp is None:
            skipped[unit] = "no thin-slice CT (sides %s)" % sides
            continue
        im = nib.load(ctp)
        allp = np.vstack([np.asarray(S["%ssub" % sd.lower()]["P"], float) for sd in sides])
        ct = C.Vol(im.dataobj, im.affine, allp.min(0), allp.max(0), pad=14.0)
        for sd in sides:
            st = S["%ssub" % sd.lower()]
            s, P, R = (np.asarray(st[k], float) for k in ("s", "P", "R"))
            w = np.asarray(Z[sd]["w_sub_station"], float)
            if len(w) != len(s):
                skipped["%s:%s" % (unit, sd)] = "w / station length mismatch %d / %d" % (len(w), len(s))
                continue
            surfs = {}
            for tg in TAGS:
                f = os.path.join(D17, tg, unit, "%ssub_stageA.vtp" % sd.lower())
                surfs[tg] = pv.read(f).triangulate() if os.path.exists(f) else None
            pts = []
            for a, b in Z[sd]["sub"]["core_runs_s_mm"]:
                for x in np.arange(a, b + 1e-6, 2.0):
                    pts.append(("zone", int(np.argmin(np.abs(s - x)))))
            pos = np.where(w > 0)[0]
            ok = [j for j in range(len(s)) if w[j] == 0 and (not len(pos) or np.min(np.abs(s[pos] - s[j])) >= 5.0)]
            last = -1e9
            for j in ok:
                if s[j] - last >= 10.0 and 8.0 <= s[j] <= s[-1] - 8.0:
                    pts.append(("outside", j))
                    last = s[j]
            for cls, j in pts:
                a_, b_ = max(j - 8, 0), min(j + 8, len(P) - 1)
                t = C.unit(P[b_] - P[a_])
                e1, e2 = C.frame(t)
                r = ct_req(ct, P[j], t, float(R[j]))
                r.update({"unit": unit, "side": sd, "class": cls, "s": float(s[j]), "R_station": float(R[j]),
                          "w": float(w[j])})
                for tg in TAGS:
                    m = surfs[tg]
                    r["surf_" + tg] = C.loop_req(C.section_xy(m, P[j], t, e1, e2)) if m is not None else None
                    r["ratio_" + tg] = (r["ct_req"] / r["surf_" + tg]) if r["surf_" + tg] else None
                rows.append(r)
        print(unit, len(rows), flush=True)
    summ = {}
    for cls in ("zone", "outside"):
        R_ = [r for r in rows if r["class"] == cls]
        summ[cls] = {"points": len(R_), "sides": len({(r["unit"], r["side"]) for r in R_})}
        for tg in TAGS:
            rat = [r["ratio_" + tg] for r in R_]
            summ[cls]["ratio_" + tg] = q(rat)
            summ[cls]["ratio_ge_1.25_" + tg] = int(sum(1 for x in rat if x is not None and x >= 1.25))
            summ[cls]["ratio_le_0.8_" + tg] = int(sum(1 for x in rat if x is not None and x <= 0.8))
        summ[cls]["bright_ray_frac"] = q([r["bright_ray_frac"] for r in R_])
        summ[cls]["local_eq_v5rule_surf_req_abs_diff_max"] = (max(abs(r["surf_local"] - r["surf_v5rule"]) for r in R_
                                                                   if r["surf_local"] and r["surf_v5rule"]) if R_ else None)
    out = {"script": "sub_ct_gate 1.0 (s2_harden_wall_bake, D17 subclavian zones)", "summary": summ, "skipped": skipped,
           "rows": rows}
    json.dump(out, open(os.path.join(D17, "sub_ct_gate.json"), "w"), indent=1)
    print(json.dumps({"summary": summ, "skipped": skipped}, indent=1))


if __name__ == "__main__":
    main()
