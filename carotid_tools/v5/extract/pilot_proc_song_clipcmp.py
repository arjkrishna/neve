#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_clipcmp.py (= /sp/v5_va/pilot_proc_song_clipcmp.py in the build containers)
"""pilot_proc_song_clipcmp.py <case> [<case> ...] -- does the Song 0-HU clipping move the lumen edge? (eve-training-fixed or
neve-build-stagea; json + numpy only; pilot_proc_song, 2026-10-01)

Compares, per VA, the ct_seg4 run on the SHIPPED CT (<der>/ct_seg/ct_seg4: every voxel <= 50 HU stored as 0) with the
identical run on the re-windowed CT (<der>/ct_seg/ct_seg4_negzero: PILOT_CT=negzero = ct_negzero-100, -100 HU on every
originally negative voxel, i.e. fat / air at -100 instead of 0; soft tissue 0..50 HU stays 0 in both). Same seeds (the
Arterial VMTK line), same code and parameters.
  * half-max edge: stations_rays.json r_e (ray edge radius at T_loc = the local half-max between the lumen core and the
    background) on stations matched by arclength s (nearest within 0.13 mm) and the same ray index; per-ray difference
    negzero - shipped: median, p5 / p95, mean |d|, fraction |d| > 0.1 / 0.2 mm, by ray kind of the shipped run;
  * background / threshold: Ib (soft-tissue background), T_loc (threshold), Ic (core) medians;
  * label: seg_qc r_eq_rays_median, r_volume_equiv_median, voxels;
  * wall detail (measure.json, whole VA, va_metrics unchanged): union > 0.3 / > 0.5 mm / > 0.15 r;
  * edge FWHM (edge.json, verifier erf method; each run on its own CT and its own label).
Writes /sp/v5_va/results/pilot_proc_song_clipcmp.json (merged per case).
"""
import json
import os
import sys

import numpy as np

ROOT = "/repo/carotid_data/va_sources/derived/thin_slice/song"
OUT = "/sp/v5_va/results/pilot_proc_song_clipcmp.json"


def L(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def rr(x, n=4):
    return None if x is None else round(float(x), n)


res = L(OUT) or {"about": __doc__.split("\n\n")[0], "cases": {}}
for case in sys.argv[1:]:
    A = os.path.join(ROOT, case, "ct_seg", "ct_seg4")
    B = os.path.join(ROOT, case, "ct_seg", "ct_seg4_negzero")
    sa, sb = L(os.path.join(A, "stations_rays.json")), L(os.path.join(B, "stations_rays.json"))
    qa, qb = L(os.path.join(A, "seg_qc.json")), L(os.path.join(B, "seg_qc.json"))
    ma, mb = L(os.path.join(A, "measure.json")), L(os.path.join(B, "measure.json"))
    ea, eb = L(os.path.join(A, "edge.json")), L(os.path.join(B, "edge.json"))
    crec = {}
    for v, sd, S in (("rva", "right", "R"), ("lva", "left", "L")):
        if not (sa and sb and v in sa and v in sb):
            continue
        a, b = sa[v], sb[v]
        s_a, s_b = np.asarray(a["s"], float), np.asarray(b["s"], float)
        ra, rb = np.asarray(a["r_e"], float), np.asarray(b["r_e"], float)
        ka = np.asarray(a.get("kind"), object) if a.get("kind") is not None else None
        j = np.searchsorted(s_b, s_a).clip(1, len(s_b) - 1)
        j = np.where(np.abs(s_b[j - 1] - s_a) < np.abs(s_b[j] - s_a), j - 1, j)
        ok = np.abs(s_b[j] - s_a) <= 0.13
        d = (rb[j[ok]] - ra[ok])
        fin = np.isfinite(d)
        dd = d[fin]
        rec = {"stations_matched": int(ok.sum()), "stations_shipped": int(len(s_a)), "rays": int(fin.sum()),
               "d_r_e_median_mm": rr(np.median(dd)), "d_r_e_p5_p95_mm": [rr(np.percentile(dd, 5)), rr(np.percentile(dd, 95))],
               "d_r_e_mean_abs_mm": rr(np.mean(np.abs(dd))), "frac_abs_gt_0.1mm": rr(np.mean(np.abs(dd) > 0.1), 3),
               "frac_abs_gt_0.2mm": rr(np.mean(np.abs(dd) > 0.2), 3),
               "r_e_median_shipped_mm": rr(np.nanmedian(ra[ok])), "r_e_median_negzero_mm": rr(np.nanmedian(rb[j[ok]]))}
        if ka is not None and ka.shape == ra.shape:
            kk = ka[ok][fin]
            rec["by_kind_shipped"] = {str(k): {"n": int((kk == k).sum()), "d_median_mm": rr(np.median(dd[kk == k]))}
                                      for k in sorted(set(kk.tolist()), key=str)}
        for key in ("Ib", "T_loc", "Ic"):
            xa, xb = np.asarray(a.get(key), float), np.asarray(b.get(key), float)
            rec[key + "_median_shipped"] = rr(np.nanmedian(xa), 1)
            rec[key + "_median_negzero"] = rr(np.nanmedian(xb), 1)
        for key in ("r_eq_rays_median", "r_volume_equiv_median", "voxels", "I_bg_median", "T_loc_median"):
            rec["seg_qc_" + key] = [rr(g(qa, "vessels", v, key), 4), rr(g(qb, "vessels", v, key), 4)]
        ua, ub = g(ma, "full", sd, "wall_union"), g(mb, "full", sd, "wall_union")
        if ua and ub:
            ca, cb = ua.get("core", ua), ub.get("core", ub)
            rec["union_whole_shipped"] = [rr(ca.get("frac_0.3mm")), rr(ca.get("frac_0.5mm")), rr(ca.get("frac_rel_0.15"))]
            rec["union_whole_negzero"] = [rr(cb.get("frac_0.3mm")), rr(cb.get("frac_0.5mm")), rr(cb.get("frac_rel_0.15"))]
        rec["FWHM_verifier_mm"] = [g(ea, "sides", S, "edge_verifier_method", "all", "FWHM_mm"),
                                   g(eb, "sides", S, "edge_verifier_method", "all", "FWHM_mm")]
        rec["FWHM_vmr_default_mm"] = [g(ea, "sides", S, "edge_vmr_method_default", "FWHM_mm"),
                                      g(eb, "sides", S, "edge_vmr_method_default", "FWHM_mm")]
        crec[S] = rec
        print(case, S, json.dumps(rec), flush=True)
    res["cases"][case] = crec
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
