#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_edge.py (= /sp/v5_va/pilot_proc_song_edge.py in the build containers)
# SONG COPY (pilot_proc_song, 2026-10-01 21:20): pilot_proc_ia172_edge.py with the song paths (thin_slice/song), pilot_proc_song_ imports and the Song meta keys; diffs beyond that are stated in the docstring / marked 'song'.
"""pilot_proc_song_edge.py = pilot_proc_aortaseg_edge.py (the AortaSeg pilot agent's script, read-only) for a 172-series
pilot donor; diff: paths (ia172 derived dir; TS total_v3 masks on the CTA grid = <der>/totalseg), the CT loader
(pilot_proc_song_ctseg_common.load_ct_crop: native RAS ct.nii.gz), the landmarks (<der>/ct_seg/<tag>/landmarks.json
sides[*].landmarks_used = ts_qc C2 / C1 / dura + the C6 level's own run, as tag_ctwall's TS stretches), output
<der>/ct_seg/<tag>/edge.json. Everything else verbatim.
--- docstring of pilot_proc_aortaseg_edge.py follows ---
What the AortaSeg pilot CT resolves at the VA lumen edge, and what touches the VA,
with the functions of tag_ctwall_edge.py imported UNCHANGED (so the numbers compare with TaG-Net A and VMR 0248).
neve-build-stagea; env PILOT_CASE, PILOT_SEG_TAGS (first tag used, default ct_seg4), PILOT_WALL_SIDES.

Per side with a CT wall (ct_seg/<tag>):
  1. edge FWHM: (a) the vmr agent's erf fit (tag_ctwall_edge.psf_rays) on the segmentation's own soft-tissue rays, every
     8th station x 8 rays ("default") and every 2nd station x 32 rays ("dense", split by stretch / ray inclination /
     radius); (b) the verifier's fit (verifier_rays: VMTK line + MISR, every 1 mm, 24 rays, soft-tissue exterior, no
     brighter neighbour, residual <= 12 % of contrast) -- the tag_ctwall.md headline method;
  2. image-domain relief (image_relief): share of half-max edge rays > 0.3 mm (and > 0.15 r) off the station median;
  3. lumen noise and CNR (lumen_noise);
  4. contacts (contacts): the bright-neighbour (kind 3) and dip-before-bright (kind 1) rays, the neighbour's peak HU and
     its class by the TS total_v3 masks on the CT grid (bone <= 1 mm: C1-C7, T1-T2, skull, clavicles, ribs 1-2, sternum;
     vein: brachiocephalic veins, SVC; artery: TS arteries or another vessel of our own label; else "unlabelled"),
     protrusion rays (kind 0, r_e > station median + 1 mm), label voxels touching >= 700 HU, per 10 mm.
Stretches from landmarks.json ("ts_C6level" on total_v3). Writes <der>/edge.json.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/sp/v5_va")
from vmr_common import CT  # noqa: E402
import tag_ctwall_edge as TE  # noqa: E402  (functions imported unchanged; its main() is not run)
from pilot_proc_song_ctseg_common import load_ct_crop, jdump  # noqa: E402  (ia172 loader)

CASE = os.environ["PILOT_CASE"]
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % CASE
TAG = [t for t in os.environ.get("PILOT_SEG_TAGS", "ct_seg4").split(",") if t][0]
TSV3 = os.path.join(DER, "totalseg")   # ia172: total_v3 masks on the CTA grid


def ts_classifier(lo, hi):
    """tag_ctwall_edge.ts_classifier with the masks directory as a parameter (it hard-codes TaG-Net's)."""
    import nibabel as nib
    groups = {"bone": ["vertebrae_C%d" % k for k in range(1, 8)] + ["vertebrae_T1", "vertebrae_T2", "skull", "clavicula_left",
                                                                    "clavicula_right", "rib_left_1", "rib_right_1",
                                                                    "rib_left_2", "rib_right_2", "sternum"],
              "vein": ["brachiocephalic_vein_left", "brachiocephalic_vein_right", "superior_vena_cava"],
              "artery_ts": ["subclavian_artery_left", "subclavian_artery_right", "common_carotid_artery_left",
                            "common_carotid_artery_right", "aorta", "brachiocephalic_trunk"]}
    seg = os.path.join(TSV3, "segmentations")
    im = nib.load(os.path.join(seg, "vertebrae_C1.nii.gz"))
    A = im.affine
    Ai = np.linalg.inv(A)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = corners @ Ai[:3, :3].T + Ai[:3, 3]
    a = np.maximum(np.floor(ijk.min(0)).astype(int), 0)
    b = np.minimum(np.ceil(ijk.max(0)).astype(int), np.array(im.shape) - 1)
    sp = np.abs(np.diag(A)[:3])
    dist = {}
    shape = tuple(b - a + 1)
    for g, names in groups.items():
        acc = np.zeros(shape, bool)
        for nm in names:
            p = os.path.join(seg, nm + ".nii.gz")
            if os.path.exists(p):
                acc |= np.asarray(nib.load(p).dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]) > 0
        dist[g] = ndimage.distance_transform_edt(~acc, sampling=sp) if acc.any() else None

    def classify(W):
        I = np.round(W @ Ai[:3, :3].T + Ai[:3, 3]).astype(int) - a
        I = np.clip(I, 0, np.array(shape) - 1)
        return {g: (np.full(len(W), np.inf) if dd is None else dd[I[:, 0], I[:, 1], I[:, 2]]) for g, dd in dist.items()}
    return classify


def main():
    t0 = time.time()
    D4 = os.path.join(DER, "ct_seg", TAG)
    st = json.load(open(os.path.join(D4, "stations_rays.json")))
    allp = np.vstack([np.asarray(st[k]["P"]) for k in st])
    ct = CT(load_ct_crop(allp.min(0), allp.max(0), margin_mm=12.0))
    hu = lambda X: ct.sample(np.asarray(X, float))  # noqa: E731
    LMJ = json.load(open(os.path.join(D4, "landmarks.json")))["sides"]   # ia172 landmarks.json
    z4 = np.load(os.path.join(D4, "labels_kji.npz"))
    vv = [k for k in ("rva", "lva") if k in st]
    pts = np.vstack([np.asarray(st[k]["P"]) for k in vv])
    classify = ts_classifier(pts.min(0) - 10, pts.max(0) + 10)
    wall_sides = [s for s in os.environ.get("PILOT_WALL_SIDES", "R,L").split(",") if s]
    out = {"what": __doc__.split("\n\n")[0], "case": CASE, "tag": TAG, "ct_spacing_mm": ct.sp.tolist(), "sides": {},
           "erf_fit_bias_on_blurred_discs": TE.erf_bias_table()}
    for side, v, k in (("R", "rva", 1), ("L", "lva", 2)):
        if v not in st or side not in wall_sides:
            continue
        clp = os.path.join(D4, "%s_stageA_vmtk.json" % v)
        if not os.path.isfile(clp):
            continue
        Q, s, RR = TE.resample_cl(clp)
        L = float(s[-1])
        sd_ = "right" if side == "R" else "left"
        lm = {kk: LMJ[sd_]["landmarks_used"].get(kk) for kk in ("C6", "C2", "C1", "dura")}
        for kk in ("C6", "C2", "C1", "dura"):
            if lm.get(kk) is None:
                lm[kk] = {"C6": 0.0, "C2": 0.0, "C1": 0.0, "dura": L}[kk] if kk != "dura" else L
        rec = {"length_mm": round(L, 2), "landmarks_used": lm}
        d = st[v]
        Pst = np.asarray(d["P"])
        sv = TE.map_s(Pst, Q, s)
        re = np.asarray(d["r_e"])
        req = np.sqrt(np.mean(re ** 2, axis=1))
        dflt = TE.psf_rays(ct, d, 8, (0, 8, 16, 24, 32, 40, 48, 56))
        dens = TE.psf_rays(ct, d, 2, tuple(range(0, 64, 2)))
        rec["edge_vmr_method_default"] = TE.summ_sig(dflt[:, 2]) if len(dflt) else {"n": 0}
        ii = dens[:, 0].astype(int)
        rec["edge_vmr_method_dense"] = TE.splits(sv[ii], dens[:, 2], dens[:, 4], req[ii], lm, L) if len(dens) else {"n": 0}
        vr = TE.verifier_rays(hu, Q, s, RR)
        rec["edge_verifier_method"] = TE.splits(vr[:, 0], vr[:, 1], vr[:, 3], vr[:, 4], lm, L) if len(vr) else {"all": {"n": 0}}
        if len(vr):
            rec["edge_verifier_method"]["edge_R_minus_misr_median_mm"] = round(float(np.median(vr[:, 2])), 3)
        print(side, "FWHM vmr-default", rec["edge_vmr_method_default"].get("FWHM_mm"),
              "verifier", rec["edge_verifier_method"]["all"].get("FWHM_mm"), "n", rec["edge_verifier_method"]["all"].get("n"),
              "dense", (rec["edge_vmr_method_dense"].get("all") or {}).get("FWHM_mm"), flush=True)
        Dr = TE.image_relief(hu, Q, s, RR)
        sm = TE.stretch_of(Dr[:, 0], lm, L) if len(Dr) else {}
        rec["image_edge_dev"] = {kk: {"frac_rays_dev_gt_0.3mm": round(float(Dr[m, 1].mean()), 4) if m.any() else None,
                                      "frac_rays_dev_gt_0.15r": round(float(Dr[m, 2].mean()), 4) if m.any() else None,
                                      "edge_r_median": round(float(np.median(Dr[m, 3])), 3) if m.any() else None}
                                 for kk, m in sm.items()}
        N = TE.lumen_noise(ct, Q, s, RR)
        Ic, Ib = np.median(np.asarray(d["Ic"])), np.median(np.asarray(d["Ib"]))
        rec["noise"] = {"lumen_voxel_std_HU_median": round(float(np.median(N[:, 1])), 1) if len(N) else None,
                        "lumen_voxel_std_HU_p25_p75": np.percentile(N[:, 1], [25, 75]).round(1).tolist() if len(N) else None,
                        "n_stations": int(len(N)), "I_core_median": round(float(Ic), 1), "I_bg_median": round(float(Ib), 1),
                        "CNR": round(float((Ic - Ib) / np.median(N[:, 1])), 1) if len(N) else None}
        rows, cls = TE.contacts(ct, st, v, sv, (z4["labels"], z4["origin_mm"], z4["spacing_mm"]), classify, k)
        kind = np.asarray(d["kind"])
        med = np.median(re, axis=1)
        prot = (kind == 0) & (re > med[:, None] + 1.0)
        Lk = z4["labels"] == k
        brow = []
        for a0 in np.arange(0.0, L, 10.0):
            a1 = min(a0 + 10.0, L)
            ms = (sv >= a0) & (sv < a1)
            mr = (rows[:, 0] >= a0) & (rows[:, 0] < a1) if len(rows) else np.zeros(0, bool)
            cl_here = [cls[j] for j in np.nonzero(mr)[0]]
            nr = max(int(ms.sum()) * kind.shape[1], 1)
            brow.append({"from_mm": round(float(a0), 1), "to_mm": round(float(a1), 1),
                         "frac_rays_upper_threshold_kind3": round(float((kind[ms] == 3).sum() / nr), 4),
                         "frac_rays_dip_before_bright_kind1": round(float((kind[ms] == 1).sum() / nr), 4),
                         "frac_rays_protrusion": round(float(prot[ms].sum() / nr), 4),
                         "contact_classes": {c: cl_here.count(c) for c in sorted(set(cl_here))},
                         "contact_peak_HU_median": round(float(np.median(rows[mr, 2])), 0) if mr.any() else None,
                         "contact_peak_HU_max": round(float(np.max(rows[mr, 2])), 0) if mr.any() else None,
                         "image_edge_dev_0.3": round(float(Dr[(Dr[:, 0] >= a0) & (Dr[:, 0] < a1), 1].mean()), 4)
                         if len(Dr) and ((Dr[:, 0] >= a0) & (Dr[:, 0] < a1)).any() else None})
        rec["per_10mm"] = brow
        rec["contacts_total"] = {"n_rays": int(len(rows)), "by_class": {c: cls.count(c) for c in sorted(set(cls))},
                                 "frac_rays_kind3_whole": round(float((kind == 3).mean()), 4),
                                 "frac_rays_kind1_whole": round(float((kind == 1).mean()), 4),
                                 "frac_rays_protrusion_whole": round(float(prot.mean()), 4),
                                 "peak_HU_by_class": {c: [round(float(np.percentile(rows[[j for j in range(len(rows)) if cls[j] == c], 2], q)), 0)
                                                          for q in (10, 50, 90)] for c in sorted(set(cls))}}
        rec["label_voxels"] = int(Lk.sum())
        out["sides"][side] = rec
        print(side, "contacts", json.dumps(rec["contacts_total"]), "noise", json.dumps(rec["noise"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    jdump(out, os.path.join(D4, "edge.json"))
    print("done %.0f s" % out["seconds"])


if __name__ == "__main__":
    main()
