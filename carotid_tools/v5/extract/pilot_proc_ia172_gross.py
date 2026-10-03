#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_gross.py (= /sp/v5_va/pilot_proc_ia172_gross.py in the build containers)
"""pilot_proc_ia172_gross.py -- gross shape of every traced VA side of one 172-series pilot case (PILOT_CASE), on the VMTK
MISR line of Arterial's cut-out surface (<der>/arterial/tsba/<v>_stageA_vmtk.json), sigma 3 mm. neve-build-stagea (pyvista + nibabel).

Same measures as the calibration's gross ranges (tag_calibration_gross.py / tag_labels_gross.py), imported UNCHANGED:
  * va_metrics.measure_va(points, radii, surface=None, do_union=False, starts_at_origin=True, parent_dir = SCA fwd5 tangent
    at the ostium foot, up = +z, landmarks): length, chord, DM, shape@3.0 (min bend radius, loops, turning), take-off;
  * tag_labels_gross.takeoffs (fwd5 / central7 / ref3_4 take-off at 5 and 10 mm, SCA fwd5 elevation at the origin, VA
    10 mm chord elevation) with the SCA centreline from Arterial's RSA / LSA label, else the TS subclavian mask
    (pilot_proc_ia172_ctseg_common._sa_path), oriented medial -> lateral;
  * tag_calibration_gross.bend_window: the C2-loop bend radius = va_metrics.frenet (sigma 3) maximum curvature in
    [C2 - 20, C2 + 15] mm, and the path-only window s in [0.40, 0.65] L;
  * V1-V4 lengths and the entry level from the QC landmarks (qc.json: C6 = Erec, C2 = ts_qc, C1 = C1f, dura = ts_qc) mapped
    from the trace to the VMTK line by nearest point.
A side whose course above C1 failed QC (distal_ok false) is measured on its CERVICAL part only (VMTK line cut at the dura
landmark): length / DM are then "origin -> dura", flagged. Radius: MISR median origin -> C1 x 1.10 (tag_calibration item 3,
native factor); L/R calibre ratio from those. Writes <der>/qc/gross.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402  (unchanged)
from tag_labels_gross import takeoffs, foot_on, at_s, unit  # noqa: E402  (unchanged)
from tag_calibration_gross import bend_window  # noqa: E402  (unchanged)
import pilot_proc_ia172_ctseg_common as CC  # noqa: E402  (SCA path helper; reads PILOT_CASE)

CASE = os.environ["PILOT_CASE"]
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172/%s" % CASE
UP = np.array([0.0, 0.0, 1.0])
F = 1.10


def rr(x, n=2):
    return None if x is None else round(float(x), n)


qc = json.load(open(os.path.join(DER, "qc/qc.json")))
ex = json.load(open(os.path.join(DER, "arterial/tsba/va_extract.json")))
out = {"case": CASE, "what": __doc__, "sides": {}}
rmed = {}
for side, v, sd in (("R", "rva", "right"), ("L", "lva", "left")):
    q = qc["sides"][side]
    vp = os.path.join(DER, "arterial/tsba/%s_stageA_vmtk.json" % v)
    if not q.get("trace_ok") or not os.path.isfile(vp):
        out["sides"][side] = {"measured": False, "why": "no trace or no VMTK line"}
        continue
    vj = json.load(open(vp))
    Q, R = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
    T = np.asarray(ex["sides"][side]["trace"]["points"], float)
    if np.linalg.norm(Q[-1] - T[0]) < np.linalg.norm(Q[0] - T[0]):
        Q, R = Q[::-1], R[::-1]
    sQ = vm.arclen(Q)
    sT = vm.arclen(T)

    def tmap(s_t):
        if s_t is None:
            return None
        p = np.array([np.interp(s_t, sT, T[:, i]) for i in range(3)])
        return float(sQ[int(np.argmin(np.linalg.norm(Q - p, axis=1)))])
    lmT = q.get("landmarks_used_mm") or {}
    lm = {k: tmap(lmT.get(k)) for k in ("C6", "C2", "C1", "dura")}
    cut = None
    if not q.get("distal_ok") and lm.get("dura") is not None:
        cut = lm["dura"]
        k = int(np.searchsorted(sQ, cut)) + 1
        Q, R, sQ = Q[:k], R[:k], sQ[:k]
    try:
        sca, _, how = CC._sa_path(side, Q[0])
        sca = np.asarray(sca, float)
        sg = 1.0 if side == "R" else -1.0
        if sg * (sca[-1, 0] - sca[0, 0]) < 0:
            sca = sca[::-1]
        tk, t5 = takeoffs(Q, sca)
    except Exception as e:  # noqa: BLE001
        tk, t5, how = {"error": repr(e)}, None, None
    lmv = {k: lm[k] for k in ("C6", "C2", "C1", "dura") if lm.get(k) is not None and lm[k] < sQ[-1]}
    m = vm.measure_va(Q, R, surface=None, voxel_mm=qc["acquisition_voxel_mm"],
                      source_type="Arterial binary mask, art_extract cut-out stage-A surface, VMTK MISR (172-series)",
                      radius_kind="VMTK MISR (art_vmtk_one)", landmarks=lmv or None, starts_at_origin=True,
                      parent_dir=t5, up=UP, name="%s %s VA (Arterial VMTK)" % (CASE, sd), do_union=False)
    sh = m["gross"]["shape@3.0"]
    c1 = lm.get("C1")
    sel = (sQ >= 3.0) & (sQ <= (c1 if c1 is not None else sQ[-1] - 3.0))
    r_c = float(np.median(R[sel])) if sel.any() else float(np.median(R))
    rmed[side] = F * r_c
    V = None
    if all(lm.get(k) is not None for k in ("C6", "C2", "dura")) and cut is None:
        V = [rr(lm["C6"], 1), rr(lm["C2"] - lm["C6"], 1), rr(lm["dura"] - lm["C2"], 1), rr(sQ[-1] - lm["dura"], 1)]
    elif all(lm.get(k) is not None for k in ("C6", "C2", "dura")):
        V = [rr(lm["C6"], 1), rr(lm["C2"] - lm["C6"], 1), rr(lm["dura"] - lm["C2"], 1), None]
    rec = {"measured": True, "centerline": vp, "cut_at_dura_mm": rr(cut, 1),
           "extent": "origin -> confluence (whole VA)" if cut is None else "origin -> dura only (course above C1 failed QC)",
           "length_mm": rr(m["gross"]["length_mm"], 1), "chord_mm": rr(m["gross"]["chord_mm"], 1), "DM": rr(m["gross"]["DM"], 3),
           "min_bend_radius_s3_mm": sh["min_bend_radius_mm"], "loops_360_60mm": sh["n_loops_360_60mm"],
           "uturns_150_30mm": sh["n_uturns_150_30mm"], "turn_max_60mm_deg": sh["turn_max_60mm_deg"],
           "kappa_p50_s3": sh["kappa_p50"], "takeoff_va_metrics": m["gross"].get("takeoff"),
           "takeoff_variants": tk, "takeoff_10mm_fwd5_deg": tk.get("takeoff_10mm_fwd5"),
           "subclavian_elevation_fwd5_deg": tk.get("sca_fwd5_elevation_deg"), "sca_source": how,
           "c2_loop": {"landmark": bend_window(Q, lm["C2"] - 20.0, lm["C2"] + 15.0) if lm.get("C2") is not None else None,
                       "pathonly": bend_window(Q, 0.40 * sQ[-1], 0.65 * sQ[-1]) if cut is None else None},
           "landmarks_on_vmtk_mm": {k: rr(x, 1) for k, x in lm.items()}, "V1_V4_mm": V,
           "entry_level_Erec": q.get("entry_level_Erec"), "entry_level_ts_qc": q.get("entry_level_ts_qc"),
           "r_misr_median_origin_to_C1_mm": rr(r_c, 3), "r_corrected_x1.10_mm": rr(F * r_c, 3),
           "calibre": m.get("calibre"), "qc_status": q.get("status"), "passes_as_v5_donor": q.get("passes_as_v5_donor")}
    out["sides"][side] = rec
    print(CASE, side, json.dumps({k: rec[k] for k in ("extent", "length_mm", "DM", "min_bend_radius_s3_mm", "takeoff_10mm_fwd5_deg",
                                                      "subclavian_elevation_fwd5_deg", "V1_V4_mm", "entry_level_Erec",
                                                      "r_corrected_x1.10_mm")}),
          "C2 bend", (rec["c2_loop"]["landmark"] or {}).get("bend_radius_mm"), flush=True)
if "R" in rmed and "L" in rmed:
    out["L_over_R_calibre"] = rr(rmed["L"] / rmed["R"], 3)
    out["dominance"] = vm.dominance(rmed["L"], rmed["R"])
json.dump(out, open(os.path.join(DER, "qc/gross.json"), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
print("L/R", out.get("L_over_R_calibre"), out.get("dominance"))
