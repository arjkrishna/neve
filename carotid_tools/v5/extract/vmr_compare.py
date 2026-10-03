#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_compare.py (= /sp/v5_va/vmr_compare.py in the build containers)
"""VMR 0248, CT-derived vs lofted: (1) CT edge sharpness (erf fit to lumen edges: what wall scales the CT
can resolve), (2) where the tight bends are, (3) radius profiles (CT VMTK MISR vs lofted VMTK MISR vs
expert contour r_eq) on one arclength axis, (4) the cervical pieces a v5 graft would actually use, measured
and graded alone (va_metrics / va_standard unchanged). eve-training-fixed.
Writes results/vmr/vmr_compare.json (+ the plot results/vmr/radius_profiles.png)."""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402
import va_standard as vs  # noqa: E402
from vmr_common import CT, DERIVED, WORK, jdump, load_vmr, rmf  # noqa: E402

TAGS = ("ct_seg3", "ct_seg4")
VOXEL = (0.4004, 0.4004, 0.7)


def edge_model(r, A, B, r0, sig):
    return B + (A - B) * 0.5 * erfc((r - r0) / (np.sqrt(2) * sig))


def psf_fit(ct, st, names, every=8, rays=(0, 8, 16, 24, 32, 40, 48, 56)):
    out = {}
    for nm in names:
        d = st[nm]
        P, re, kind = np.asarray(d["P"]), np.asarray(d["r_e"]), np.asarray(d["kind"])
        _, U, W = rmf(P)
        sig = []
        for i in range(5, len(P) - 5, every):
            for a in rays:
                if kind[i, a] != 0:
                    continue
                th = 2 * np.pi * a / re.shape[1]
                dvec = np.cos(th) * U[i] + np.sin(th) * W[i]
                r = np.arange(max(0.0, re[i, a] - 2.0), re[i, a] + 2.0, 0.05)
                p = ct.sample(P[i] + r[:, None] * dvec)
                try:
                    popt, _ = curve_fit(edge_model, r, p, p0=[p[0], p[-1], re[i, a], 0.35],
                                        bounds=([0, -400, r[0], 0.05], [1500, 600, r[-1], 2.0]), maxfev=4000)
                    if popt[0] - popt[1] > 150:
                        sig.append(popt[3])
                except Exception:  # noqa: BLE001
                    pass
        sig = np.asarray(sig)
        out[nm] = {"n_rays": int(len(sig)), "sigma_median_mm": float(np.median(sig)) if len(sig) else None,
                   "sigma_p25_p75_mm": [float(np.percentile(sig, 25)), float(np.percentile(sig, 75))] if len(sig) else None,
                   "edge_FWHM_mm": float(2.3548 * np.median(sig)) if len(sig) else None}
    return out


def bends(P, R, sigma):
    Pr, Rr, s = vm.resample(P, R)
    t, kappa, _, _ = vm.frenet(Pr, sigma)
    core = (s >= vm.END_EXCL_MM) & (s <= s[-1] - vm.END_EXCL_MM)
    rb = 1.0 / np.maximum(kappa, 1e-9)
    tight = core & (rb < 2.5)
    runs = []
    i = 0
    while i < len(s):
        if tight[i]:
            j = i
            while j + 1 < len(s) and tight[j + 1]:
                j += 1
            k = i + int(np.argmin(rb[i:j + 1]))
            runs.append({"from_mm": round(float(s[i]), 1), "to_mm": round(float(s[j]), 1),
                         "min_bend_radius_mm": round(float(rb[k]), 2), "at_mm": round(float(s[k]), 1),
                         "r_misr_there_mm": round(float(Rr[k]), 2), "z_mm": round(float(Pr[k, 2]), 1)})
            i = j + 1
        else:
            i += 1
    k = int(np.argmin(np.where(core, rb, np.inf)))
    return {"sigma_mm": sigma, "min_bend_radius_mm": round(float(rb[k]), 2), "at_mm": round(float(s[k]), 1),
            "z_mm": round(float(Pr[k, 2]), 1), "stretches_below_2.5mm": runs}


def project_s(Pref, sref, Q):
    from scipy.spatial import cKDTree
    _, i = cKDTree(Pref).query(Q)
    return sref[i]


def piece_job(args):
    tag, variant, side, lo, hi, label = args
    import pyvista as pv
    v = {"right": "rva", "left": "lva"}[side]
    d = json.load(open(os.path.join(DERIVED, tag, "%s_%s_vmtk.json" % (v, variant))))
    P, R, s = vm.resample(np.asarray(d["points"]), np.asarray(d["radii"]))
    m = (s >= lo) & (s <= hi)
    surf = pv.read(os.path.join(DERIVED, tag, "%s_%s.vtp" % (v, variant)))
    res = vm.measure_va(P[m], R[m], surf, voxel_mm=VOXEL,
                        source_type="CT segmentation (%s, %s) piece %s" % (tag, variant, label),
                        radius_kind="VMTK MISR (vmtkCenterlines pointlist, AppendEndPoints=1; neve-build-vmtk)",
                        starts_at_origin=False, name="VMR 0248 %s VA %s %s/%s" % (side, label, tag, variant),
                        extra={"piece_mm_on_full_centerline": [round(lo, 2), round(hi, 2)]})
    return tag, variant, side, label, res


def main():
    t0 = time.time()
    vmr = load_vmr(with_ct=True)
    ct = CT(vmr["ct"])
    out = {"what": __doc__}
    # 1. edge sharpness
    st = json.load(open(os.path.join(DERIVED, "ct_seg4", "stations_rays.json")))
    out["edge_sharpness"] = psf_fit(ct, st, ("rsub", "lsub", "rva", "lva", "ba"))
    print("edge", json.dumps(out["edge_sharpness"]), flush=True)
    # 2. bends + 3. radius profiles
    met = {t: json.load(open(os.path.join(WORK, "vmr0248_metrics_%s.json" % t))) for t in TAGS}
    out["bends"], out["radius_profiles"] = {}, {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(16, 10))
    for ax, side in zip(axs, ("right", "left")):
        v = {"right": "rva", "left": "lva"}[side]
        ref = json.load(open(os.path.join(DERIVED, "ct_seg4", "%s_stageA_vmtk.json" % v)))
        Pref, Rref, sref = vm.resample(np.asarray(ref["points"]), np.asarray(ref["radii"]))
        prof = {"ct_seg4_stageA": (sref, Rref)}
        for nm, path in (("ct_seg4_iso", os.path.join(DERIVED, "ct_seg4", "%s_iso_vmtk.json" % v)),
                         ("ct_seg3_stageA", os.path.join(DERIVED, "ct_seg3", "%s_stageA_vmtk.json" % v)),
                         ("lofted_vmtk", os.path.join(DERIVED, "lofted", "%s_vmtk.json" % side))):
            d = json.load(open(path))
            P, R, s = vm.resample(np.asarray(d["points"]), np.asarray(d["radii"]))
            sp = project_s(Pref, sref, P)
            prof[nm] = (sp, R)
        cons = vmr["contours"][side + "_vertebral"]
        cc = np.array([c["center"] for c in cons])
        cr = np.array([c["r_eq"] for c in cons])
        csp = project_s(Pref, sref, cc)
        keep = (csp > 1.0) & (csp < sref[-1] - 1.0)
        prof["expert_contour_req"] = (csp[keep], cr[keep])
        bins = np.arange(0, sref[-1], 1.0)
        tab = {}
        for nm, (sp, R) in prof.items():
            idx = np.digitize(sp, bins)
            tab[nm] = np.array([np.median(R[idx == k]) if (idx == k).any() else np.nan for k in range(1, len(bins) + 1)])
        lm = met["ct_seg4"]["variants"]["ct_stageA"][side]["segments_wall"]["ranges_mm"]
        segs = {k: lm[k] for k in ("V1", "V2", "V3", "V4")}
        comp = {}
        for nm in tab:
            if nm == "ct_seg4_stageA":
                continue
            row = {}
            for sg, (a, b) in list(segs.items()) + [("whole", (0, sref[-1]))]:
                mm = (bins >= a) & (bins < b) & np.isfinite(tab[nm]) & np.isfinite(tab["ct_seg4_stageA"])
                if nm == "expert_contour_req":
                    sel = (prof[nm][0] >= a) & (prof[nm][0] < b)
                    Rc = np.interp(prof[nm][0][sel], sref, Rref)
                    row[sg] = {"n": int(sel.sum()), "median_ratio_to_ct": round(float(np.median(prof[nm][1][sel] / Rc)), 3)
                               if sel.any() else None}
                    continue
                if mm.sum() < 5:
                    continue
                x, y = tab["ct_seg4_stageA"][mm], tab[nm][mm]
                row[sg] = {"n_mm": int(mm.sum()), "median_ratio_to_ct": round(float(np.median(y / x)), 3),
                           "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3)}
            comp[nm] = row
        segr = {}
        for sg, (a, b) in segs.items():
            mm = (sref >= a) & (sref < b)
            segr[sg] = round(float(np.median(Rref[mm])), 3)
        out["radius_profiles"][side] = {"ct_seg4_stageA_misr_median_per_segment": segr, "vs_ct_seg4_stageA": comp,
                                        "profiles_1mm": {nm: np.round(t, 3).tolist() for nm, t in tab.items()}}
        for nm, (sp, R) in prof.items():
            if nm == "expert_contour_req":
                ax.plot(sp, R, "k.", ms=4, label=nm)
            else:
                o = np.argsort(sp)
                ax.plot(sp[o], R[o], lw=0.9, label=nm)
        for sg, (a, b) in segs.items():
            ax.axvline(a, color="0.6", ls=":")
            ax.text(0.5 * (a + b), 0.55, sg, ha="center")
        ax.set_ylim(0.5, 2.8)
        ax.set_xlabel("s along the CT (seg4 stage-A) centerline, mm from the origin")
        ax.set_ylabel("radius mm")
        ax.set_title("VMR 0248 %s VA: VMTK MISR radius by representation, expert contour r_eq" % side)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        ob = {}
        for nm, path in (("ct_seg4_stageA", os.path.join(DERIVED, "ct_seg4", "%s_stageA_vmtk.json" % v)),
                         ("lofted_vmtk", os.path.join(DERIVED, "lofted", "%s_vmtk.json" % side))):
            d = json.load(open(path))
            ob[nm] = {"sigma3": bends(np.asarray(d["points"]), np.asarray(d["radii"]), 3.0),
                      "sigma1.5": bends(np.asarray(d["points"]), np.asarray(d["radii"]), 1.5)}
        out["bends"][side] = ob
        print(side, json.dumps(ob)[:600], flush=True)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK, "radius_profiles.png"), dpi=70)
    # 4. the cervical pieces
    jobs = []
    for tag in TAGS:
        for side in ("right", "left"):
            lmk = met[tag]["variants"]["ct_stageA"][side]["segments_wall"]["ranges_mm"]
            C6, C2, C1 = lmk["V2"][0], lmk["V2"][1], lmk["V3_C1_to_dura"][0]
            v1_bad_end = 40.0 if side == "left" else 0.0   # left V1 s 9.75-39.5: dense venous contrast contact
            for variant in ("stageA", "iso"):
                jobs.append((tag, variant, side, 0.0, C1, "origin_to_C1"))
                jobs.append((tag, variant, side, max(C6, v1_bad_end), C1, "V2_V3low_clean"))
    with Pool(8) as pool:
        res = pool.map(piece_job, jobs, chunksize=1)
    pieces = {}
    for tag, variant, side, label, r in res:
        pieces.setdefault(tag, {}).setdefault(variant, {}).setdefault(label, {})[side] = r
    grades = {}
    for tag, a in pieces.items():
        for variant, b in a.items():
            for label, c in b.items():
                g = vs.grade_set([c["right"], c["left"]])
                grades["%s/%s/%s" % (tag, variant, label)] = g
                print(tag, variant, label, "union R %.3f L %.3f" % (c["right"]["wall_union"]["core"]["frac_0.3mm"],
                                                                     c["left"]["wall_union"]["core"]["frac_0.3mm"]),
                      json.dumps({k: (v["median"], v["result"]) for k, v in g["set_rules"].items()}), flush=True)
    out["pieces"] = pieces
    out["piece_grades"] = grades
    out["seconds"] = round(time.time() - t0, 1)
    jdump(out, os.path.join(WORK, "vmr_compare.json"))
    print("done %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
