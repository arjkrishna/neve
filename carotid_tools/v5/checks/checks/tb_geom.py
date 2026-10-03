#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\tb_geom.py (= /sp/v5_va/checks/tb_geom.py in the build containers)
"""v5_checks item 2, TopBrain piece unchanged (geometry): for every grafted VA, the TopBrain distal piece of v5 (cut ->
end) against v4's (cut -> end), aligned from the distal end on a 0.5 mm grid: rigid (Kabsch) RMS of the centreline and
max |radius difference|, over the common length and over the common length minus the 25 mm seam zone when the calibre
ramp sits on the TopBrain side; plus the length difference and the start trim.  -> /sp/v5_va/checks/tb_geom.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/sp/v4_build/audit")
import topo_audit as TA  # noqa: E402

A5, V4 = "/repo/carotid_data/anatomies_v5", "/repo/carotid_data/anatomies_v4"


def kabsch(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return float(np.sqrt(np.mean(np.sum((a @ (U @ D @ Vt) - b) ** 2, axis=1))))


def piece(p, r, cut):
    s = TA.arclen(p)
    m = s >= cut - 1e-9
    q, rr, ss = p[m], r[m], s[m]
    return q, rr, ss[-1] - ss


out = []
for n in sorted(x for x in os.listdir(A5) if x.startswith("case_")):
    pr5 = json.load(open(os.path.join(A5, n, "provenance.json")))
    pr4 = json.load(open(os.path.join(V4, n, "provenance.json")))
    b5 = {t: (p, r) for t, p, r in TA.load_branches(os.path.join(A5, n, "Centrelines_comb"))}
    b4 = {t: (p, r) for t, p, r in TA.load_branches(os.path.join(V4, n, "Centrelines_comb"))}
    for side, tag, ck, tk in (("right", "RVA", "vaC_R", "va_R"), ("left", "LVA", "vaC_L", "va_L")):
        if ck not in pr5["xform"]:
            continue
        sd = pr5["v5_va"][side]
        q5, r5, d5 = piece(*b5[tag], float(pr5["xform"][tk]["cut_mm"]))
        q4, r4, d4 = piece(*b4[tag], float(pr4["xform"][tk]["cut_mm"]))
        L = min(d5.max(), d4.max())
        row = {"name": n, "side": side, "ramp_side": sd["ramp"]["side"], "len_v5": round(float(d5.max()), 2),
               "len_v4": round(float(d4.max()), 2), "tb_start_trim_mm": sd.get("tb_start_trim_mm")}
        for lab, top in (("common", L), ("minus_seam", L - (25.0 if sd["ramp"]["side"] != "donor" else 0.0)), ("core_minus_2mm", L - 2.0), ("core_minus_5mm", L - 5.0), ("core_minus_10mm", L - 10.0)):
            g = np.arange(0.0, max(top, 0.5), 0.5)
            A = np.stack([np.interp(g, d5[::-1], q5[::-1, k]) for k in range(3)], 1)
            B = np.stack([np.interp(g, d4[::-1], q4[::-1, k]) for k in range(3)], 1)
            ra, rb = np.interp(g, d5[::-1], r5[::-1]), np.interp(g, d4[::-1], r4[::-1])
            row[lab] = {"len_mm": round(float(g[-1]), 1), "kabsch_rms_mm": round(kabsch(A, B), 4),
                        "radius_max_abs_diff_mm": round(float(np.abs(ra - rb).max()), 4),
                        "radius_max_diff_at_from_distal_mm": round(float(g[int(np.argmax(np.abs(ra - rb)))]), 1),
                        "radius_median_abs_diff_mm": round(float(np.median(np.abs(ra - rb))), 4)}
        out.append(row)
json.dump(out, open("/sp/v5_va/checks/tb_geom.json", "w"), indent=1)
for lab in ("common", "minus_seam", "core_minus_2mm", "core_minus_5mm", "core_minus_10mm"):
    k = np.array([r[lab]["kabsch_rms_mm"] for r in out]); rd = np.array([r[lab]["radius_max_abs_diff_mm"] for r in out])
    print(lab, "n", len(out), "kabsch max %.4f p99 %.4f" % (k.max(), np.percentile(k, 99)), "radius diff max %.4f p99 %.4f" % (rd.max(), np.percentile(rd, 99)))
print("ramp sides", {s: sum(r["ramp_side"] == s for r in out) for s in set(r["ramp_side"] for r in out)})
dl = np.array([r["len_v5"] - r["len_v4"] for r in out]); print("len v5-v4 min/median/max", dl.min().round(2), np.median(dl).round(2), dl.max().round(2))
bad = [r for r in out if r["core_minus_2mm"]["radius_max_abs_diff_mm"] > 0.05 or r["core_minus_2mm"]["kabsch_rms_mm"] > 0.05]
print("over 0.05 mm (core minus 2 mm):", len(bad), [(r["name"], r["side"], r["common"]) for r in bad[:6]])
import collections; print("where max (common, from distal):", collections.Counter(int(r["common"]["radius_max_diff_at_from_distal_mm"] // 10 * 10) for r in out if r["common"]["radius_max_abs_diff_mm"] > 0.05))
print("median abs radius diff max", max(r["common"]["radius_median_abs_diff_mm"] for r in out))
