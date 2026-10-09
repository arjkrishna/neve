#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\graft_v5_reg_donor.py (= /sp/v5_va/graft_v5_reg_donor.py in the build containers)
"""graft_v5_reg_donor.py -- build the v5 REGISTRY donor record of one donor patient (one "unit", 1-2 VA sides).

A donor record is what graft_v5_reg.py needs to graft a donor's cervical VA under strategy B (host subclavian kept
byte for byte; donor VA origin -> splice in V3; v4's TopBrain distal piece above), and what graft_v5_assign.py needs to
rank donors. It is built from what the pilot / study agents left on disk; nothing is re-segmented here.

Output (gitignored scratch data):
    /repo/carotid_data/va_sources/v5_reg/donors/<donor_id>/record.json      the UNIT (patient-level) record
    /repo/carotid_data/va_sources/v5_reg/donors/<donor_id>/<R|L>.json       one SIDE record (arrays inline)
    /repo/carotid_data/va_sources/v5_reg/donors/<donor_id>/<R|L>_surface.vtp a COPY of the side's closed lumen surface
                                                                              (real-wall donors only; md5 recorded)
Side record (schema "v5_reg_donor_side/1"):
    donor_id, side, source, source_ref, licence, case_id, patient_id, patient_key (de-duplication key)
    wall_class "real_surface" | "tube"; real_surface bool; real_from_mm (tube below this, past the ostium)
    surface {path, md5, kind, genus, n_points, src}                                   (real-wall donors)
    centerline {points [ostium -> distal, RAS mm], radii_misr, radii_calibre, radius_kind, calibre_kind, src,
                length_mm, cut_at_dura_mm}
    ostium {point, O_v (foot on the donor's OWN subclavian axis), ostium_to_axis_mm}
    subclavian {points (medial -> lateral, ~+-40 mm about O_v), radii, src, t_fwd_5mm (unit, 5 mm forward chord from
                O_v, lateral), t_central (prototype's +-5-point tangent), elevation_fwd5_deg}
    landmarks {C6, C2, C1, dura, end (mm along centerline.points), set, src, all_sets}
    takeoff_native {5, 10 (deg, VA chord from the ostium vs t_fwd_5mm), 10_central_tangent, va_10mm_elevation_deg}
    qc {verdict pass|review|fail, status, passes_as_v5_donor, distal_ok, failed, review, src}
    coarseness {whole_union_0.3, C6_C1_union_0.3, origin_C1_union_0.3, plain_tube_bin_frac, clean_union_0.3,
                FWHM_mm, score (what the assignment ranks by), src}
    contamination_zones [{s_from, s_to, kind, evidence}]
    frame {"name": "RAS mm", "chirality_sign": +1/-1/None, how}
    builder {version, built, inputs {path: mtime}}
Unit record: donor_id, source, case_id, patient_id, licence, sides {R|L: verdict, wall_class, usable}, pair_complete.

Sources (--source):
    ia172 | song | aortaseg   thin-slice pilot case (derived/thin_slice/<source>/<case>): ct_seg4 CT-edge stage-A surface,
                              its VMTK MISR line, ct_seg4 landmarks / measure / edge / seg_qc / stations, pilot QC record
                              (results/pilot_proc_<source>.json)                        -> real_surface
    vmr0248                   VMR 0248 ct_seg4 (derived/vmr0248/ct_seg4), expert subclavian path (sources.load_vmr0248),
                              landmarks / wall from results/vmr0248_metrics.json; left V1 s 10-40 mm vein contact -> tube
                              below 40 mm (prototype rule)                             -> real_surface
    ts                        1.5 mm TotalSegmentator CTA traced by Arterial (derived/arterial/ts_<sid>): VMTK MISR of the
                              cut-out surface x 1.10 (no CT half-max at 1.5 mm), QC = recalibrated Stage-2 gate list in
                              results/tag_calibration.json                             -> tube (real_surface false)
TaG-Net is NEVER a donor (no data licence): refused here.

    python3 /sp/v5_va/graft_v5_reg_donor.py --source ia172 --case ia172_141 [--sides R,L]
    python3 /sp/v5_va/graft_v5_reg_donor.py --source vmr0248
    python3 /sp/v5_va/graft_v5_reg_donor.py --source ts --case ts_s0960
    python3 /sp/v5_va/graft_v5_reg_donor.py --testing          # the graft_reg test set (VMR + the passing pilot sides + TS pairs)
Runs in eve-training-fixed (numpy, scipy, vtk). Re-run safe: a unit's files are rewritten; inputs' mtimes are recorded
so a stale record (pilot output changed afterwards) can be detected (graft_v5_reg.py warns).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/repo/carotid_tools")
sys.path.insert(0, "/sp/v5_va")

from graft_siphon import arclength, prep_siphon, unit  # noqa: E402

VERSION = "graft_v5_reg_donor 1.1 (2026-10-02; v5_assign: --out-root, --eligible per-side wall / QC / trace files)"
SCHEMA = "v5_reg_donor_side/1"
OUT_ROOT = "/repo/carotid_data/va_sources/v5_reg/donors"
CFG = {"out_root": OUT_ROOT}
# v5_assign extension (2026-10-02): the reserve conditions of results/v5_eligible_final.json (v5_eye172 14:30)
RESERVE_CONDITIONS = {
    ("ia172_172", "L"): {"min_ostium_advance_mm": 3.0,
                         "flag": "arch-origin donor VA (left VA arises from the aortic arch; the trace's first ~2 mm is a "
                                 "tissue jump from the subclavian): start >= 3 mm past the jump; the junction to the host "
                                 "subclavian is synthetic (v5_eye172)"},
    ("AHMU1218138", "R"): {"flag": "real wall origin -> C2 only (right wall stops at C2, 131 of 209 mm; 9.5 mm straight "
                                   "tissue jump above C1 on the trace) (v5_eye172)"},
}
DER = "/repo/carotid_data/va_sources/derived"
RES = "/sp/v5_va/results"
STEP = 1.0
TAKEOFF_TANGENT_MM = 5.0
SUB_KEEP_MM = 40.0
TS_RADIUS_FACTOR = 1.10          # tag_calibration item 3, protocol-matched native factor (no CT half-max at 1.5 mm)
# contamination zone rule on ct_seg4 edge.json per-10 mm bins: non-bone contact rays (other artery / vein / unlabelled
# bright) >= CONTACT_RAYS_MIN, or rays with a dip before a brighter neighbour >= DIP_FRAC_MIN
CONTACT_RAYS_MIN = 15
DIP_FRAC_MIN = 0.05
PLAIN_TUBE_LT = 0.01

LICENCES = {
    "ia172": "Head-CTA 172-series (Blagosklonova et al., Zenodo 15697196, doi 10.5281/zenodo.15697196), CC BY 4.0 image; "
             "segmentation derived through Arterial (CC BY-NC 4.0 weights, PolyForm-NC code): NON-COMMERCIAL",
    "song": "Song et al. 2024 head-and-neck CTA (AHMU), CC BY 4.0 image; derived through Arterial: NON-COMMERCIAL",
    "aortaseg": "AortaSeg-60 (CC BY 4.0, treated as such); derived through Arterial: NON-COMMERCIAL",
    "vmr0248": "VMR 0248 research and development use; keep LICENSE_VMR_0248.txt; cite Wilson et al. 2013 "
               "(doi 10.1115/1.4025983), NLM R01LM013120 / NHLBI HHSN268201100035C",
    "ts": "TotalSegmentator v3 CTA (CC BY 4.0, Wasserthal et al. 2023, doi 10.1148/ryai.230024); trace derived through "
          "Arterial: NON-COMMERCIAL",
}
SOURCE_REF = {
    "ia172": "Zenodo 15697196 (172-series head CTA)", "song": "Song 2024 AHMU head-and-neck CTA",
    "aortaseg": "AortaSeg-60", "vmr0248": "Vascular Model Repository 0248_H_AOCERE_CAS",
    "ts": "TotalSegmentator v3 dataset",
}
SIDE_NAME = {"R": "right", "L": "left"}
VPRE = {"R": "rva", "L": "lva"}

# the graft_reg test set: VMR 0248 + pilot sides passing so far (task brief) + the 4 TotalSegmentator pairs that pass
# the recalibrated Stage-2 gate (tube fallback class)
TESTING = [("vmr0248", None, "R,L"), ("ia172", "ia172_058", "R"), ("ia172", "ia172_061", "R"), ("ia172", "ia172_141", "L"),
           ("ia172", "ia172_015", "L"), ("aortaseg", "Dissec_01", "L"),
           ("ts", "ts_s0610", "R,L"), ("ts", "ts_s0777", "R,L"), ("ts", "ts_s0960", "R,L"), ("ts", "ts_s1786", "R,L")]


# ------------------------------------------------------------------ helpers ----
def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)
    os.replace(tmp, path)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rr(x, n=3):
    return None if x is None else round(float(x), n)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def elevation(v):
    return 90.0 - angle(v, np.array([0.0, 0.0, 1.0]))


def foot_on(sp, q):
    """(foot point, distance, arclength of the foot) of q on the polyline sp (projection on the 2 segments at the
    nearest vertex -- the prototype's rule)."""
    sp = np.asarray(sp, float)
    s = arclength(sp)
    k = int(np.argmin(np.linalg.norm(sp - q, axis=1)))
    best = (np.inf, sp[k], s[k])
    for a, b in ((max(k - 1, 0), k), (k, min(k + 1, len(sp) - 1))):
        if a == b:
            continue
        ab = sp[b] - sp[a]
        t = float(np.clip((q - sp[a]) @ ab / (ab @ ab), 0.0, 1.0))
        f = sp[a] + t * ab
        d = float(np.linalg.norm(f - q))
        if d < best[0]:
            best = (d, f, s[a] + t * (s[b] - s[a]))
    return best[1], best[0], float(best[2])


def forward_chord(sp, O, L):
    """graft_v5_va.forward_chord: unit chord from O (on sp, index increasing distally) to the point L mm further."""
    sp = np.asarray(sp, float)
    s = arclength(sp)
    _, _, s0 = foot_on(sp, np.asarray(O, float))
    q = np.array([np.interp(s0 + L, s, sp[:, i]) for i in range(3)])
    return unit(q - np.asarray(O, float))


def d_chord(p, d):
    s = arclength(p)
    return p[min(int(np.searchsorted(s, d)), len(p) - 1)] - p[0]


def orient_lateral(sp, sr, side):
    """Subclavian medial -> lateral (RAS: the right one runs to +x, the left one to -x)."""
    sg = 1.0 if side == "R" else -1.0
    if sg * (sp[-1, 0] - sp[0, 0]) < 0:
        return sp[::-1].copy(), (None if sr is None else sr[::-1].copy())
    return sp, sr


def crop_about(sp, sr, s0, half):
    s = arclength(sp)
    m = np.abs(s - s0) <= half
    return sp[m], (None if sr is None else np.asarray(sr)[m])


def surface_info(path):
    """points, Euler genus of the largest component (vtk), closedness."""
    try:
        import vtk
        rd = vtk.vtkXMLPolyDataReader()
        rd.SetFileName(path)
        rd.Update()
        pd = rd.GetOutput()
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(pd)
        fe.BoundaryEdgesOn()
        fe.NonManifoldEdgesOn()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()
        npts, ncell = pd.GetNumberOfPoints(), pd.GetNumberOfCells()
        ext = vtk.vtkExtractEdges()
        ext.SetInputData(pd)
        ext.Update()
        ne = ext.GetOutput().GetNumberOfLines()
        chi = npts - ne + ncell
        b = pd.GetBounds()
        return {"n_points": npts, "n_cells": ncell, "open_or_nonmanifold_edges": fe.GetOutput().GetNumberOfLines(),
                "euler_chi": chi, "genus_if_one_component": rr((2 - chi) / 2.0, 1), "bounds": [rr(x, 2) for x in b]}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def mtimes(paths):
    out = {}
    for p in paths:
        if p and os.path.exists(p):
            out[p] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
    return out


def chirality(right_pts, left_pts, dy_v2_sign):
    """sign of ((R - L) x anterior) . up, from the two VA traces at a common height (RAS-like frame -> +1).
    anterior = +y if the VA lies anterior of the cord in V2 (dy_V2 > 0, as on every GT patient); up = mean VA chord."""
    try:
        R, L = np.asarray(right_pts, float), np.asarray(left_pts, float)
        up = unit((R[-1] - R[0]) + (L[-1] - L[0]))
        z = min(R[:, 2].max(), L[:, 2].max(), max(R[0, 2], L[0, 2]) + 40.0)
        pr = R[int(np.argmin(np.abs(R[:, 2] - z)))]
        pl = L[int(np.argmin(np.abs(L[:, 2] - z)))]
        right = pr - pl
        ant = np.array([0.0, float(np.sign(dy_v2_sign) or 1.0), 0.0])
        return float(np.sign(np.cross(right, ant) @ up))
    except Exception:  # noqa: BLE001
        return None


def native_takeoff(points, radii, O_v, t_fwd, t_central):
    p, _ = prep_siphon(np.asarray(points, float), np.asarray(radii, float), STEP)
    d10 = d_chord(p, 10.0)
    return {"5": angle(d_chord(p, 5.0), t_fwd), "10": angle(d10, t_fwd),
            "10_central_tangent": angle(d10, t_central) if t_central is not None else None,
            "va_10mm_elevation_deg": elevation(d10), "subclavian_elevation_fwd5_deg": elevation(t_fwd),
            "measured_on": "prep_siphon 1 mm line (as graft_v5_reg places it); VA chord ostium -> first point >= d mm; "
                           "parent = 5 mm forward chord of the donor's own subclavian from O_v"}


def coarseness_from_bins(bins10, zones):
    """whole union from 10 mm bins (area-weighted when areas are given), plain-tube fraction, and the union with the
    contaminated bins left out (clean)."""
    if not bins10:
        return {}
    f = np.array([b.get("frac_0.3mm", np.nan) for b in bins10], float)
    a = np.array([b.get("area_mm2", 1.0) or 1.0 for b in bins10], float)
    ok = np.isfinite(f)
    bad = np.zeros(len(bins10), bool)
    for i, b in enumerate(bins10):
        for z in zones:
            if z["s_from"] < b["to_mm"] and z["s_to"] > b["from_mm"]:
                bad[i] = True
    out = {"bins10_n": int(ok.sum()), "plain_tube_bin_frac": rr(np.mean(f[ok] < PLAIN_TUBE_LT), 3),
           "bins10_union_0.3_areaweighted": rr((f[ok] * a[ok]).sum() / a[ok].sum(), 4)}
    m = ok & ~bad
    out["clean_union_0.3"] = rr((f[m] * a[m]).sum() / a[m].sum(), 4) if m.any() else None
    out["n_bins_contaminated"] = int(bad.sum())
    return out


def zones_from_edge(per10):
    zones = []
    for b in per10 or []:
        cc = b.get("contact_classes") or {}
        nonbone = {k: v for k, v in cc.items() if not k.startswith("bone")}
        n_nb = int(sum(nonbone.values()))
        dip = float(b.get("frac_rays_dip_before_bright_kind1") or 0.0)
        if n_nb >= CONTACT_RAYS_MIN or dip >= DIP_FRAC_MIN:
            kinds = sorted(nonbone) or ["bright neighbour (dip before bright)"]
            zones.append({"s_from": float(b["from_mm"]), "s_to": float(b["to_mm"]), "kind": "; ".join(kinds),
                          "evidence": "edge.json per_10mm: %d non-bone contact rays %s, dip-before-bright %.3f, peak HU %s"
                                      % (n_nb, json.dumps(nonbone), dip, b.get("contact_peak_HU_max"))})
    # merge touching bins
    merged = []
    for z in zones:
        if merged and abs(merged[-1]["s_to"] - z["s_from"]) < 1e-6:
            merged[-1]["s_to"] = z["s_to"]
            merged[-1]["evidence"] += " | " + z["evidence"]
            if z["kind"] not in merged[-1]["kind"]:
                merged[-1]["kind"] += "; " + z["kind"]
        else:
            merged.append(dict(z))
    return merged


def pilot_record(source, case, side):
    p = os.path.join(RES, "pilot_proc_%s.json" % source)
    if not os.path.isfile(p):
        return None, p
    for r in jload(p).get("records", []):
        if r.get("case") == case and r.get("side") == side:
            return r, p
    return None, p


def decide_lookup(source, case, side):
    """attach any entry of a later pilot decide file (format unknown when this was written: generic search)."""
    p = os.path.join(RES, "pilot_decide.json")
    if not os.path.isfile(p):
        return None
    hits = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("case") == case and o.get("side") in (side, SIDE_NAME[side]):
                hits.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    try:
        walk(jload(p))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return {"file": p, "entries": hits[:3]}


# --------------------------------------------------------------- thin-slice ----
def build_thin(source, case, side, out_dir, ov=None):
    """ov (v5_assign extension): one side of a pair in results/v5_eligible_final.json -- the wall files
    (wall_files.surface_vtp / vmtk_centerline_json / measure_json; the seg dir is the surface's folder), the QC file
    (qc_file: its sides.<side> record replaces the pilot record) and the trace file (trace_file: chirality) that the
    final tally graded. Without ov the 1.0 behaviour is unchanged."""
    der = os.path.join(DER, "thin_slice", source, case)
    if not os.path.isdir(der):
        raise FileNotFoundError(der)
    seg = None
    if ov:
        wf = ov["wall_files"]
        sd = os.path.dirname(wf["surface_vtp"])
        seg = (ov.get("wall_tag") or os.path.basename(sd), sd)
        vj_p, sf_p = wf["vmtk_centerline_json"], wf["surface_vtp"]
        for p_ in (vj_p, sf_p):
            if not os.path.isfile(p_):
                raise FileNotFoundError(p_)
    else:
        for tag in ("ct_seg4", "ct_seg4_nofix"):
            d = os.path.join(der, "ct_seg", tag)
            if os.path.isfile(os.path.join(d, "%s_stageA_vmtk.json" % VPRE[side])) and \
                    os.path.isfile(os.path.join(d, "%s_stageA.vtp" % VPRE[side])):
                seg = (tag, d)
                break
        if seg is None:
            raise FileNotFoundError("no ct_seg4 surface + VMTK line for %s %s %s" % (source, case, side))
    tag, sd = seg
    if not ov:
        vj_p = os.path.join(sd, "%s_stageA_vmtk.json" % VPRE[side])
        sf_p = os.path.join(sd, "%s_stageA.vtp" % VPRE[side])
    vj = jload(vj_p)
    P, Rm = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
    # subclavian: the Arterial RSA / LSA label segment nearest the VA ostium (the pilot's gross.json uses the same path,
    # so the native take-off compares); fallback: the ct_seg4 stations (+-30 mm, 0.25 mm stations: locally wiggly at
    # the branch, e.g. 058 R fwd-5 elevation -10.6 deg vs 17.6 on the label)
    st_p = os.path.join(sd, "stations_rays.json")
    sub_key = "rsub" if side == "R" else "lsub"
    sub, sub_r, sub_src = None, None, None
    import glob
    best, bd, bfile = None, 1e9, None
    for lcp in sorted(glob.glob(os.path.join(der, "arterial", "runs", "*", "labelled_centerlines.json"))):
        for s_ in jload(lcp)["segments"]:
            if s_["class"] != side + "SA":
                continue
            Q = np.asarray(s_["points_ras_mm"], float)
            d = min(np.linalg.norm(Q - P[0], axis=1).min(), np.linalg.norm(Q - P[-1], axis=1).min())
            if d < bd:
                best, bd, bfile = s_, d, lcp
    if best is not None and bd <= 6.0:
        sub = np.asarray(best["points_ras_mm"], float)
        sub_r = np.asarray(best["radius_mm"], float)
        sub_src = "%s %sSA label (nearest segment %.1f mm from the VA ostium)" % (bfile, side, bd)
    if sub is None and os.path.isfile(st_p):
        st = jload(st_p)
        if sub_key in st:
            sub = np.asarray(st[sub_key]["P"], float)
            re_ = st[sub_key].get("r_e")
            sub_r = (np.sqrt(np.nanmean(np.asarray(re_, float) ** 2, axis=1)) if re_ is not None and np.ndim(re_) == 2
                     else np.asarray(re_ if re_ is not None else st[sub_key]["R"], float))
            sub_src = "%s %s (ct_seg4 station path, +-30 mm about the VA origin; r = CT half-max r_e)" % (st_p, sub_key)
    if sub is None:
        raise RuntimeError("no subclavian path for %s %s %s" % (source, case, side))
    # VA orientation: the ostium is the end nearer the subclavian
    if np.linalg.norm(sub - P[-1], axis=1).min() < np.linalg.norm(sub - P[0], axis=1).min():
        P, Rm = P[::-1].copy(), Rm[::-1].copy()
    sub, sub_r = orient_lateral(sub, sub_r, side)
    O_v, d_ax, s_foot = foot_on(sub, P[0])
    t_fwd = forward_chord(sub, O_v, TAKEOFF_TANGENT_MM)
    k = int(np.argmin(np.linalg.norm(sub - O_v, axis=1)))
    t_c = unit(sub[min(k + 5, len(sub) - 1)] - sub[max(k - 5, 0)])
    s_va = arclength(P)

    # CT half-max calibre (seg_qc r_volume_equiv_per_mm along the station path, mapped by normalised arclength)
    rcal, cal_kind = None, None
    sq_p = os.path.join(sd, "seg_qc.json")
    seg_qc = jload(sq_p) if os.path.isfile(sq_p) else {}
    rv = ((seg_qc.get("vessels") or {}).get(VPRE[side]) or {}).get("r_volume_equiv_per_mm")
    if rv:
        rv = np.asarray(rv, float)
        x = np.linspace(0.0, 1.0, len(rv))
        rcal = np.interp(s_va / s_va[-1], x, rv)
        cal_kind = "CT half-max equal-volume radius per mm (ct_seg4 seg_qc r_volume_equiv_per_mm), mapped by normalised arclength"

    # landmarks
    lm, lm_set, lm_src, lm_all = None, None, None, {}
    lmp = os.path.join(sd, "landmarks.json")
    if os.path.isfile(lmp):
        L = jload(lmp)["sides"].get(SIDE_NAME[side]) or {}
        for key in ("landmarks_rec", "landmarks_used", "landmarks_ct"):
            if L.get(key):
                lm_all[key] = L[key]
        for key in ("landmarks_rec", "landmarks_used"):
            if L.get(key):
                lm, lm_set, lm_src = L[key], key, lmp
                break
    rec_p, rec_file = pilot_record(source, case, side)
    pilot_status = rec_p.get("status") if isinstance(rec_p, dict) else None
    if ov and ov.get("qc_file"):
        # the graded QC record (final tally) replaces the pilot record (pilot_proc_<src>.json may predate pipeline v2)
        rec_p, rec_file = (jload(ov["qc_file"]).get("sides") or {}).get(side), ov["qc_file"]
        if rec_p is None:
            raise RuntimeError("no sides.%s in %s" % (side, ov["qc_file"]))
    if lm is None and rec_p and isinstance(rec_p.get("landmarks"), dict):
        cs = rec_p["landmarks"].get("ct_seg4_stageA") or {}
        for key in ("Erec", "foramen", "ts"):
            if cs.get(key):
                lm_all["ct_seg4_stageA/" + key] = cs[key]
        if cs.get("Erec"):
            lm, lm_set, lm_src = cs["Erec"], "ct_seg4_stageA/Erec", rec_file
    landmarks = {"set": lm_set, "src": lm_src, "all_sets": lm_all}
    for k_ in ("C6", "C2", "C1", "dura", "end"):
        landmarks[k_] = None if not lm or lm.get(k_) is None else float(lm[k_])
    if landmarks["end"] is None:
        landmarks["end"] = float(s_va[-1])

    # measure (wall)
    meas, meas_src = None, None
    mp = os.path.join(sd, "measure.json")
    if os.path.isfile(mp):
        m = jload(mp)
        meas, meas_src = (m.get("full") or {}).get(SIDE_NAME[side]), mp
    if meas is None and os.path.isfile(os.path.join(der, "measure.json")):
        m = jload(os.path.join(der, "measure.json"))
        f = ((m.get("full") or {}).get("ct_seg4_stageA") or {}).get("k64") or {}
        meas, meas_src = f.get(side), os.path.join(der, "measure.json")
    # edge / contamination
    edge, edge_src = None, None
    for ep in (os.path.join(sd, "edge.json"), os.path.join(der, "edge.json")):
        if os.path.isfile(ep):
            e = jload(ep)
            if (e.get("sides") or {}).get(side):
                edge, edge_src = e["sides"][side], ep
                break
    zones = zones_from_edge(edge.get("per_10mm") if edge else None)
    for z in zones:
        z["src"] = edge_src
    # aortaseg attribution (bright-neighbour share of the relief, by stretch)
    att_p = os.path.join(der, "attrib.json")
    attrib = None
    if os.path.isfile(att_p) and side == "L":
        a = jload(att_p)
        attrib = {kk: {"s_mm": v.get("s_mm"), "frac_dep_gt_0.3": v.get("frac_dep_gt_0.3"),
                      "bright_neighbour_share_of_relief": ((v.get("zones") or {}).get("bright_neighbour") or {}).get("share_of_dep_gt_0.3_area")}
                  for kk, v in (a.get("stretches") or {}).items()}
        for kk, v in (a.get("stretches") or {}).items():
            bn = ((v.get("zones") or {}).get("bright_neighbour") or {})
            # elementary stretches only (the composite C6 -> C1 and whole would swallow everything)
            if kk != "whole" and not kk.startswith("C6_to_C1") and (bn.get("share_of_dep_gt_0.3_area") or 0) >= 0.3 \
                    and v.get("s_mm"):
                zones.append({"s_from": float(v["s_mm"][0]), "s_to": float(v["s_mm"][1]), "kind": "bright neighbour (attrib)",
                              "evidence": "attrib.json %s: %.0f%% of the > 0.3 mm relief sits on bright-neighbour contacts"
                                          % (kk, 100 * bn["share_of_dep_gt_0.3_area"]), "src": att_p})

    # coarseness
    co = {"src": meas_src}
    if meas:
        wu = (meas.get("wall_union") or {})
        co["whole_union_0.3"] = rr(((wu.get("core") or {}).get("frac_0.3mm")), 4)
        stu = ((meas.get("stretches") or {}).get("union") or {})
        for nm, key in (("C6_C1_union_0.3", "C6_to_C1"), ("origin_C1_union_0.3", "origin_to_C1")):
            v = stu.get(key) or {}
            co[nm] = rr(((v.get("core") or v.get("all") or {}).get("frac_0.3mm")), 4)
        co.update(coarseness_from_bins(meas.get("bins10"), zones))
        topo = meas.get("surface_topology") or {}
        surf_genus = topo.get("largest_genus")
    else:
        surf_genus = None
    if rec_p and isinstance(rec_p.get("wall"), dict):
        w = rec_p["wall"]
        if co.get("whole_union_0.3") is None:
            co["whole_union_0.3"] = rr(((w.get("whole") or {}).get("0.3mm")) or
                                       ((w.get("union_whole_core") or {}).get("frac_0.3mm")), 4)
        stt = (w.get("stretches") or {}).get("ts") or {}
        if co.get("C6_C1_union_0.3") is None and stt.get("C6_to_C1"):
            co["C6_C1_union_0.3"] = rr(stt["C6_to_C1"].get("0.3mm"), 4)
        if co.get("origin_C1_union_0.3") is None and stt.get("origin_to_C1"):
            co["origin_C1_union_0.3"] = rr(stt["origin_to_C1"].get("0.3mm"), 4)
    if edge:
        co["FWHM_mm"] = rr(((edge.get("edge_verifier_method") or {}).get("all") or {}).get("FWHM_mm"), 3)
    # the score the assignment ranks by: the grafted cervical stretch (origin -> C1) with contaminated bins out when
    # available; never the relief that sits on contamination
    co["score"] = co.get("clean_union_0.3") if co.get("clean_union_0.3") is not None else co.get("origin_C1_union_0.3")
    co["score_def"] = "union > 0.3 mm on the CT-edge wall, 10 mm bins with contamination removed (else origin -> C1)"

    # QC verdict
    qc = {"src": rec_file}
    if rec_p is None:
        qp = os.path.join(der, "qc", "qc.json")
        q = (jload(qp).get("sides") or {}).get(side) if os.path.isfile(qp) else None
        rec_p, qc["src"] = q, qp
    if rec_p is not None:
        if "status" in rec_p:
            st_ = rec_p.get("status")
            qc.update(status=st_, passes_as_v5_donor=rec_p.get("passes_as_v5_donor"), distal_ok=rec_p.get("distal_ok"),
                      failed=rec_p.get("gate_fail_reasons") or rec_p.get("recalibrated_failed"), review=rec_p.get("review"))
            qc["verdict"] = {"donor": "pass", "donor_review": "review"}.get(st_, "fail")
        elif isinstance(rec_p.get("qc"), dict):
            q = rec_p["qc"]
            v = str(q.get("cervical_verdict", ""))
            qc.update(status=v, passes_as_v5_donor=q.get("passes_as_v5_donor"),
                      distal_ok=(q.get("distal") or {}).get("distal_clean"), failed=q.get("failed_cervical"),
                      review=q.get("review_flags"))
            qc["verdict"] = "fail" if not q.get("passes_as_v5_donor") else ("review" if "review" in v.lower() else "pass")
            if rec_p.get("visual_check"):
                qc["visual_check"] = rec_p["visual_check"].get("verdict")
    else:
        qc["verdict"] = "unknown"
    if source == "song":
        selp = os.path.join(der, "qc", "select.json")
        if os.path.isfile(selp):
            qc["song_select"] = (jload(selp).get("sides") or {}).get(side)
    dec = decide_lookup(source, case, side)
    if dec:
        qc["pilot_decide"] = dec

    # distal cut (the seed was cut at the dura when the course above C1 failed QC)
    cut = None
    seed_info = (((seg_qc.get("adapter") or {}).get("seed_info") or {}).get(SIDE_NAME[side] + "_vertebral") or {})
    if isinstance(seed_info, dict) and seed_info.get("cut_at_dura_mm") is not None:
        cut = float(seed_info["cut_at_dura_mm"])

    # chirality from the two traces of the case
    chir, how = None, None
    vx_list = [os.path.join(der, "arterial", ext, "va_extract.json") for ext in ("tsba", "ts")]
    if ov and ov.get("trace_file"):
        vx_list = [ov["trace_file"]] + [v_ for v_ in vx_list if v_ != ov["trace_file"]]
    for vx in vx_list:
        if os.path.isfile(vx):
            x = jload(vx)
            try:
                Rt = x["sides"]["R"]["trace"]["points"]
                Lt = x["sides"]["L"]["trace"]["points"]
                dy = ((rec_p.get("v2_band") or {}).get("dy_med", 1.0)) if isinstance(rec_p, dict) else 1.0
                chir = chirality(Rt, Lt, dy if dy is not None else 1.0)
                how = "%s traces R/L at a common height, anterior from the V2 cord offset" % vx
                break
            except Exception:  # noqa: BLE001
                continue

    os.makedirs(out_dir, exist_ok=True)
    surf_copy = os.path.join(out_dir, "%s_surface.vtp" % side)
    shutil.copy2(sf_p, surf_copy)
    sinfo = surface_info(surf_copy)
    sub_c, subr_c = crop_about(sub, sub_r, foot_on(sub, O_v)[2], SUB_KEEP_MM)
    tk = native_takeoff(P, Rm, O_v, t_fwd, t_c)
    rec = {
        "schema": SCHEMA, "donor_id": unit_id(source, case), "side": side, "source": source,
        "source_ref": SOURCE_REF[source], "licence": LICENCES[source], "case_id": case,
        "patient_id": case, "patient_key": "%s:%s" % (source, case),
        "wall_class": "real_surface", "real_surface": True, "real_from_mm": 0.0,
        "surface": {"path": surf_copy, "md5": md5(surf_copy), "src": sf_p, "seg_tag": tag,
                    "kind": "%s CT-edge segmentation %s, stage-A surface (marching cubes + windowed sinc 20/0.1)" % (source, tag),
                    "genus_measure_json": surf_genus, **sinfo},
        "centerline": {"points": P, "radii_misr": Rm, "radii_calibre": rcal, "radius_kind": "VMTK MISR of the stage-A surface",
                       "calibre_kind": cal_kind, "src": vj_p, "length_mm": float(s_va[-1]), "cut_at_dura_mm": cut,
                       "r_misr_median": rr(np.median(Rm)), "r_calibre_median": rr(np.median(rcal)) if rcal is not None else None},
        "ostium": {"point": P[0], "O_v": O_v, "ostium_to_axis_mm": d_ax},
        "subclavian": {"points": sub_c, "radii": subr_c, "src": sub_src, "t_fwd_5mm": t_fwd, "t_central": t_c,
                       "elevation_fwd5_deg": elevation(t_fwd), "r_at_foot": rr(np.asarray(sub_r)[k]) if sub_r is not None else None},
        "landmarks": landmarks, "takeoff_native": tk, "qc": qc, "coarseness": co,
        "contamination_zones": zones, "attrib": attrib,
        "frame": {"name": "RAS mm (NIfTI positive diagonal affine)", "chirality_sign": chir, "how": how},
        "acquisition_voxel_mm": (rec_p or {}).get("acquisition_voxel_mm") if isinstance(rec_p, dict) else None,
        "builder": {"version": VERSION, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "inputs": mtimes([vj_p, sf_p, st_p, sq_p, lmp, mp, edge_src, rec_file, att_p])},
    }
    if ov:
        rec["builder"]["inputs"].update(mtimes([ov.get("qc_file"), ov.get("trace_file")]))
        rec["qc"]["pilot_record_status"] = pilot_status
        rec["eligible_side"] = {"wall_tag": tag, "trace_set": ov.get("trace_set"), "qc_file": ov.get("qc_file"),
                                "trace_file": ov.get("trace_file"), "wall_files": ov.get("wall_files"),
                                "how": "v5_assign: files the final tally graded (results/v5_eligible_final.json)"}
    elif tag != "ct_seg4":
        rec["qc"].setdefault("notes", []).append("surface from %s (ct_seg4 not complete when built): rebuild when it is" % tag)
    return rec


def attach_eligible(rec, pair, side):
    """v5_assign extension: the final eligible list's per-pair / per-side facts go into the side record (class,
    licence and attributions, NC flag, scanner, flags, eye verdicts, tally coarseness). The assignment ranks by the
    tally's contact-free donor-stretch union (decision 03:50 item 4); the builder's own clean union is kept beside."""
    sd = pair["sides"][side]
    co = rec["coarseness"]
    co["builder_score"] = co.get("score")
    cf = sd.get("donor_u03_contact_free")
    if cf is None:
        i = "RL".index(side)
        cf = (pair["coarseness"].get("R_L_contact_free") or [None, None])[i]
        ab = (pair["coarseness"].get("R_L_all_bins") or [None, None])[i]
    else:
        ab = sd.get("donor_u03_all_bins")
    co["tally_contact_free_u03"] = cf
    co["tally_all_bins_u03"] = ab
    co["score"] = cf
    co["score_def"] = ("final tally (scaleup_tally.json): donor-stretch union > 0.3 mm on the CT-edge wall, contact-free "
                       "10 mm bins (validated ranking rule, decision 03:50 item 4)")
    side_flags = [f for f in pair.get("flags", []) if f.startswith(side + " ")]
    level = 0
    if sd.get("qc_status") == "donor_review" or sd.get("review"):
        level = 1
    if sd.get("provisional_only") or sd.get("wall_truncated") or sd.get("v1_rescue_used") or pair.get("provenance_flags"):
        level = 2
    rc = RESERVE_CONDITIONS.get((pair["case"], side)) or {}
    rec["eligible"] = {
        "file": "/sp/v5_va/results/v5_eligible_final.json", "class": pair["class"],
        "rank": pair.get("primary_rank", pair.get("reserve_rank")), "tally_rank": pair.get("tally_rank"),
        "patient_key": pair.get("patient_key"), "dataset_ref": pair.get("dataset_ref"), "licence": pair.get("licence"),
        "non_commercial": pair.get("non_commercial"), "arterial_derived": pair.get("arterial_derived"),
        "attributions": pair.get("attributions"), "scanner": pair.get("scanner"),
        "acquisition_voxel_mm": pair.get("acquisition_voxel_mm"), "provenance_flags": pair.get("provenance_flags") or [],
        "pair_flags": pair.get("flags", []), "side_flags": side_flags + ([rc["flag"]] if rc.get("flag") else []),
        "qc_status": sd.get("qc_status"), "provisional_only": sd.get("provisional_only"),
        "wall_truncated": sd.get("wall_truncated"), "partial_truncated_wall": sd.get("partial_truncated_wall"),
        "v1_rescue_used": sd.get("v1_rescue_used"), "pinch_fix_triggered": sd.get("pinch_fix_triggered"),
        "eye": sd.get("eye"), "flag_level": level,
        "flag_level_def": "0 QC 'donor' and nothing else; 1 review flags (donor_review); 2 provisional-limit pass, "
                          "truncated wall, V1 rescue or a provenance flag (Siemens 1.0 mm)",
        "min_ostium_advance_mm": rc.get("min_ostium_advance_mm", 0.0),
        "surface_md5_expected": (sd.get("wall_files") or {}).get("surface_md5")}
    if pair.get("licence"):
        rec["licence"] = pair["licence"]
    if rc.get("min_ostium_advance_mm"):
        rec["min_ostium_advance_mm"] = rc["min_ostium_advance_mm"]
    return rec


# --------------------------------------------------------------------- VMR ----
def build_vmr(side, out_dir):
    from sources import load_vmr0248
    v = load_vmr0248(with_ct=False, with_model=False, with_walls=False)
    root = os.path.join(DER, "vmr0248")
    vj_p = os.path.join(root, "ct_seg4", "%s_stageA_vmtk.json" % VPRE[side])
    sf_p = os.path.join(root, "ct_seg4", "%s_stageA.vtp" % VPRE[side])
    vj = jload(vj_p)
    P, Rm = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
    sp = np.asarray(v["vessels"][SIDE_NAME[side] + "_subclavian"]["points"], float)
    sr = np.asarray(v["vessels"][SIDE_NAME[side] + "_subclavian"]["radius_mm"], float)
    # prototype rule (graft_v5_va.load_donor): foot of the FIRST VMTK point on the expert path, as stored
    pp, _ = prep_siphon(P, Rm, STEP)
    O_v, d_ax, s_foot = foot_on(sp, pp[0])
    k = int(np.argmin(np.linalg.norm(sp - pp[0], axis=1)))
    t_c = unit(sp[min(k + 5, len(sp) - 1)] - sp[max(k - 5, 0)])
    t_fwd = forward_chord(sp, O_v, TAKEOFF_TANGENT_MM)
    met = jload(os.path.join(RES, "vmr0248_metrics.json"))
    lmc = met["landmarks_ct"][SIDE_NAME[side]]
    s_va = arclength(P)
    landmarks = {k_: (float(lmc[k_]) if lmc.get(k_) is not None else None) for k_ in ("C6", "C2", "C1", "dura")}
    landmarks.update(end=float(s_va[-1]), set="landmarks_ct (vmr_landmarks.py)", src=os.path.join(RES, "vmr0248_metrics.json"),
                     all_sets={})
    if side == "L":
        landmarks["note"] = "left C6 disputed by the verifier: s ~ 31-40 (likely 37), not 27.5 (report, 'What the verification changed')"
    sm = met["summary_full_va"]["ct_seg4_stageA"][SIDE_NAME[side]]
    bins = (met.get("union_detail_per_10mm") or {}).get("ct_seg4_stageA", {}).get(SIDE_NAME[side])
    zones = []
    real_from = 0.0
    if side == "L":
        zones.append({"s_from": 10.0, "s_to": 40.0, "kind": "vein (undiluted venous contrast against the VA)",
                      "evidence": "notes/vmr.md: left VA against a vein full of undiluted contrast for s ~ 10-40 mm; the "
                                  "prototype bakes the first 40 mm as tube", "src": "/sp/v5_va/notes/vmr.md"})
        real_from = 40.0
    co = {"src": os.path.join(RES, "vmr0248_metrics.json"), "whole_union_0.3": rr(sm["union_core"]["frac_0.3mm"], 4),
          "C6_C1_union_0.3": {"R": 0.076, "L": 0.126}[side], "origin_C1_union_0.3": None,
          "C6_C1_note": "report: VMR neck stretch C6 -> C1 0.076 / 0.126 (R / L)", "FWHM_mm": {"R": 1.39, "L": 1.44}[side]}
    co.update(coarseness_from_bins(bins, zones))
    co["score"] = co.get("clean_union_0.3")
    co["score_def"] = "union > 0.3 mm on the CT-edge wall, 10 mm bins with contamination removed"
    os.makedirs(out_dir, exist_ok=True)
    surf_copy = os.path.join(out_dir, "%s_surface.vtp" % side)
    shutil.copy2(sf_p, surf_copy)
    sinfo = surface_info(surf_copy)
    sub_c, subr_c = crop_about(sp, sr, s_foot, SUB_KEEP_MM)
    rec = {
        "schema": SCHEMA, "donor_id": "vmr0248", "side": side, "source": "vmr0248", "source_ref": SOURCE_REF["vmr0248"],
        "licence": LICENCES["vmr0248"], "licence_file": os.path.join(root, "LICENSE_VMR_0248.txt"),
        "case_id": "0248_H_AOCERE_CAS", "patient_id": "VMR 0248", "patient_key": "vmr:0248",
        "wall_class": "real_surface", "real_surface": True, "real_from_mm": real_from,
        "surface": {"path": surf_copy, "md5": md5(surf_copy), "src": sf_p, "seg_tag": "ct_seg4",
                    "kind": "VMR 0248 CT segmentation ct_seg4, stage-A surface (marching cubes + windowed sinc 20/0.1)",
                    "genus_measure_json": sm.get("union_genus"), **sinfo},
        "centerline": {"points": P, "radii_misr": Rm, "radii_calibre": None, "radius_kind": "VMTK MISR of the stage-A surface",
                       "calibre_kind": None, "src": vj_p, "length_mm": float(s_va[-1]), "cut_at_dura_mm": None,
                       "r_misr_median": rr(np.median(Rm)), "r_calibre_median": None},
        "ostium": {"point": pp[0], "O_v": O_v, "ostium_to_axis_mm": d_ax, "rule": "prototype: foot of the prep_siphon first point"},
        "subclavian": {"points": sub_c, "radii": subr_c, "src": "sources.load_vmr0248 vessels %s_subclavian (expert SimVascular path)"
                       % SIDE_NAME[side], "t_fwd_5mm": t_fwd, "t_central": t_c, "elevation_fwd5_deg": elevation(t_fwd),
                       "r_at_foot": rr(sr[k])},
        "landmarks": landmarks, "takeoff_native": native_takeoff(P, Rm, O_v, t_fwd, t_c),
        "qc": {"verdict": "pass", "status": "reference donor (expert-traced CT; prototype donor)", "passes_as_v5_donor": True,
               "distal_ok": True, "failed": [], "review": [], "src": "report, VMR 0248 section"},
        "coarseness": co, "contamination_zones": zones,
        "frame": {"name": "RAS mm (SimVascular cm x 10)", "chirality_sign": 1.0,
                  "how": "verify_graft_mesh_chiral.py: VMR native sign +1, same as the host"},
        "acquisition_voxel_mm": [0.4004, 0.4004, 0.7],
        "builder": {"version": VERSION, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "inputs": mtimes([vj_p, sf_p, os.path.join(RES, "vmr0248_metrics.json")])},
    }
    return rec


# ---------------------------------------------------------------------- TS ----
def ts_gate_lists():
    t = jload(os.path.join(RES, "tag_calibration.json"))
    rc = t["answers"]["4_draft_gate"]["TS_counts"]["recalibrated"]
    return set(rc["pass_with_additions_list"]), set(rc["of_which_flagged_review"])


def ts_mask_subclavian(case, side, origin):
    """Fallback SCA path (pilot_proc_ia172_ctseg_common._sa_path's rule): the TS subclavian mask voxels within 40 mm
    of the VA origin, 2 mm bins along their principal axis, bin centroids. Needs nibabel (neve-build-stagea)."""
    try:
        import nibabel as nib
        from scipy import ndimage
    except ImportError:
        return None, None, "nibabel not available in this container (run in neve-build-stagea)"
    nm = "subclavian_artery_right" if side == "R" else "subclavian_artery_left"
    p = os.path.join("/repo/carotid_data/va_sources/totalseg_v3", case.replace("ts_", ""), "segmentations", nm + ".nii.gz")
    if not os.path.isfile(p):
        return None, None, "no %s" % p
    im = nib.load(p)
    m = np.asanyarray(im.dataobj) > 0
    A = im.affine
    edt = ndimage.distance_transform_edt(m, sampling=np.sqrt((A[:3, :3] ** 2).sum(0)))
    ii = np.argwhere(m)
    W = (A @ np.c_[ii, np.ones(len(ii))].T)[:3].T
    near = np.linalg.norm(W - origin, axis=1) <= 40.0
    if near.sum() < 20:
        return None, None, "TS mask has < 20 voxels within 40 mm of the origin"
    W, e = W[near], edt[tuple(ii[near].T)]
    c = W.mean(0)
    u = np.linalg.svd(W - c, full_matrices=False)[2][0]
    t = (W - c) @ u
    rows = []
    for a0 in np.arange(t.min(), t.max(), 2.0):
        mm = (t >= a0) & (t < a0 + 2.0)
        if mm.sum() >= 3:
            rows.append((W[mm].mean(0), e[mm].max()))
    return (np.array([r_[0] for r_ in rows]), np.array([r_[1] for r_ in rows]),
            "%s: 2 mm bins along the principal axis within 40 mm of the VA origin (no Arterial label)" % p)


def build_ts(case, side, out_dir):
    der = os.path.join(DER, "arterial", case)
    vj_p = os.path.join(der, "%s_stageA_vmtk.json" % VPRE[side])
    vx_p = os.path.join(der, "va_extract.json")
    vj, vx = jload(vj_p), jload(vx_p)
    P, Rm = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
    tr = vx["sides"][side]["trace"]
    T = np.asarray(tr["points"], float)
    if np.linalg.norm(P[-1] - T[0]) < np.linalg.norm(P[0] - T[0]):
        P, Rm = P[::-1].copy(), Rm[::-1].copy()
    lcp = os.path.join(DER, "arterial", "runs", case, "labelled_centerlines.json")
    sub, sub_r, sub_src = None, None, None
    if os.path.isfile(lcp):
        segs = [s for s in jload(lcp)["segments"] if s["class"] == side + "SA"]
        best, bd = None, 1e9
        for s_ in segs:
            Q = np.asarray(s_["points_ras_mm"], float)
            d = np.linalg.norm(Q - P[0], axis=1).min()
            if d < bd:
                best, bd = s_, d
        if best is not None and bd <= 8.0:
            sub = np.asarray(best["points_ras_mm"], float)
            sub_r = np.asarray(best["radius_mm"], float)
            sub_src = "%s %sSA label (nearest segment %.1f mm from the VA origin)" % (lcp, side, bd)
    if sub is None:
        sub, sub_r, sub_src = ts_mask_subclavian(case, side, P[0])
    if sub is None:
        raise RuntimeError("no Arterial %sSA label within 8 mm of the %s VA origin of %s, and no TS mask path (%s)"
                           % (side, side, case, sub_src))
    sub, sub_r = orient_lateral(sub, sub_r, side)
    O_v, d_ax, s_foot = foot_on(sub, P[0])
    t_fwd = forward_chord(sub, O_v, TAKEOFF_TANGENT_MM)
    k = int(np.argmin(np.linalg.norm(sub - O_v, axis=1)))
    t_c = unit(sub[min(k + 5, len(sub) - 1)] - sub[max(k - 5, 0)])
    # landmarks: ts_qc on the trace, mapped to the VMTK line by nearest point
    sP, sT = arclength(P), arclength(T)
    lmT = (vx["sides"][side].get("ts_qc") or {}).get("landmarks_mm") or {}

    def tmap(s_t):
        if s_t is None:
            return None
        q = np.array([np.interp(s_t, sT, T[:, i]) for i in range(3)])
        return float(sP[int(np.argmin(np.linalg.norm(P - q, axis=1)))])
    landmarks = {k_: tmap(lmT.get(k_)) for k_ in ("C6", "C2", "C1", "dura")}
    landmarks.update(end=float(sP[-1]), set="ts_qc (trace) mapped to the VMTK line", src=vx_p, all_sets={"ts_qc_trace": lmT},
                     note="ts_qc's C2 (7-12 mm late) and C1 (26-40 mm late) are uncertain (report); C6 over-called as C7")
    passed, review = ts_gate_lists()
    key = "%s %s" % (case.replace("ts_", ""), side)
    verdict = "fail"
    if key in passed:
        verdict = "review" if key in review else "pass"
    chir = chirality(vx["sides"]["R"]["trace"]["points"], vx["sides"]["L"]["trace"]["points"], 1.0) \
        if vx["sides"].get("R", {}).get("trace", {}).get("points") and vx["sides"].get("L", {}).get("trace", {}).get("points") else None
    rec = {
        "schema": SCHEMA, "donor_id": unit_id("ts", case), "side": side, "source": "ts", "source_ref": SOURCE_REF["ts"],
        "licence": LICENCES["ts"], "case_id": case, "patient_id": case.replace("ts_", ""), "patient_key": "ts:%s" % case,
        "wall_class": "tube", "real_surface": False, "real_from_mm": None,
        "why_tube": "1.5 mm source: Arterial / CT wall detail at 1.5 mm is voxel artefact (report); real course only",
        "surface": None,
        "centerline": {"points": P, "radii_misr": Rm, "radii_calibre": Rm * TS_RADIUS_FACTOR,
                       "radius_kind": "VMTK MISR of Arterial's cut-out stage-A surface (1.5 mm grid)",
                       "calibre_kind": "MISR x %.2f (tag_calibration item 3, TS-skull protocol; NO CT half-max at 1.5 mm)" % TS_RADIUS_FACTOR,
                       "src": vj_p, "length_mm": float(sP[-1]), "cut_at_dura_mm": None,
                       "r_misr_median": rr(np.median(Rm)), "r_calibre_median": rr(np.median(Rm) * TS_RADIUS_FACTOR)},
        "ostium": {"point": P[0], "O_v": O_v, "ostium_to_axis_mm": d_ax},
        "subclavian": {"points": crop_about(sub, sub_r, foot_on(sub, O_v)[2], SUB_KEEP_MM)[0],
                       "radii": crop_about(sub, sub_r, foot_on(sub, O_v)[2], SUB_KEEP_MM)[1], "src": sub_src,
                       "t_fwd_5mm": t_fwd, "t_central": t_c, "elevation_fwd5_deg": elevation(t_fwd),
                       "r_at_foot": rr(sub_r[k])},
        "landmarks": landmarks, "takeoff_native": native_takeoff(P, Rm, O_v, t_fwd, t_c),
        "qc": {"verdict": verdict, "status": "recalibrated Stage-2 gate (tag_calibration)", "passes_as_v5_donor": verdict != "fail",
               "distal_ok": None, "failed": [], "review": ["recalibrated gate: flagged for review"] if verdict == "review" else [],
               "src": os.path.join(RES, "tag_calibration.json"),
               "note": "recalibrated gate is in-sample (verify_tag_calibration); V3-loop gate included (rule 7)"},
        "coarseness": {"score": 0.0, "score_def": "tube donor: no wall detail", "whole_union_0.3": None},
        "contamination_zones": [],
        "frame": {"name": "RAS mm (TotalSegmentator NIfTI via Arterial)", "chirality_sign": chir,
                  "how": "va_extract traces R/L at a common height (anterior assumed +y)"},
        "acquisition_voxel_mm": vx.get("source_spacing_mm"),
        "builder": {"version": VERSION, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "inputs": mtimes([vj_p, vx_p, lcp, os.path.join(RES, "tag_calibration.json")])},
    }
    return rec


# ------------------------------------------------------------------- driver ----
def unit_id(source, case):
    if source == "vmr0248":
        return "vmr0248"
    if source == "ts":
        return case
    return "%s_%s" % (source, case) if not case.startswith(source) else case


def build_unit(source, case, sides, pair=None):
    if source == "tagnet" or (case and "tagnet" in case.lower()):
        raise SystemExit("TaG-Net is never a donor (no data licence)")
    did = unit_id(source, case)
    out_dir = os.path.join(CFG["out_root"], did)
    os.makedirs(out_dir, exist_ok=True)
    unit = {"schema": "v5_reg_donor_unit/1", "donor_id": did, "source": source, "case_id": case or "0248_H_AOCERE_CAS",
            "licence": LICENCES[source], "sides": {}, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION}
    if pair is not None:
        unit.update(eligible_class=pair["class"], eligible_rank=pair.get("primary_rank", pair.get("reserve_rank")),
                    patient_key=pair.get("patient_key"), licence=pair.get("licence") or unit["licence"],
                    non_commercial=pair.get("non_commercial"), attributions=pair.get("attributions"),
                    provenance_flags=pair.get("provenance_flags") or [], pair_flags=pair.get("flags", []),
                    coarseness_pair=pair.get("coarseness"))
    for side in sides:
        try:
            if source == "vmr0248":
                rec = build_vmr(side, out_dir)
            elif source == "ts":
                rec = build_ts(case, side, out_dir)
            else:
                ov = None
                if pair is not None and (pair["sides"][side].get("wall_files") or {}).get("vmtk_centerline_json"):
                    ov = pair["sides"][side]
                rec = build_thin(source, case, side, out_dir, ov=ov)
            if pair is not None:
                rec = attach_eligible(rec, pair, side)
        except Exception as e:  # noqa: BLE001
            unit["sides"][side] = {"built": False, "why": "%s: %s" % (type(e).__name__, e)}
            print("FAIL %s %s: %s" % (did, side, e), flush=True)
            continue
        jdump(rec, os.path.join(out_dir, "%s.json" % side))
        usable = rec["qc"]["verdict"] in ("pass", "review") and rec["frame"]["chirality_sign"] in (1.0, None)
        unit["sides"][side] = {"built": True, "file": os.path.join(out_dir, "%s.json" % side), "verdict": rec["qc"]["verdict"],
                               "wall_class": rec["wall_class"], "usable": bool(usable),
                               "coarseness_score": rec["coarseness"].get("score"),
                               "takeoff_10_native": rr(rec["takeoff_native"]["10"], 1),
                               "subclavian_elevation_deg": rr(rec["subclavian"]["elevation_fwd5_deg"], 1),
                               "length_mm": rr(rec["centerline"]["length_mm"], 1), "landmarks": {k: rec["landmarks"].get(k) for k in ("C6", "C2", "C1", "dura", "end")},
                               "surface_genus": (rec.get("surface") or {}).get("genus_measure_json"),
                               "chirality": rec["frame"]["chirality_sign"],
                               "flag_level": (rec.get("eligible") or {}).get("flag_level"),
                               "eligible_class": (rec.get("eligible") or {}).get("class"),
                               "surface_md5": (rec.get("surface") or {}).get("md5"),
                               "min_ostium_advance_mm": rec.get("min_ostium_advance_mm", 0.0)}
        print("OK   %-22s %s verdict %-6s %s len %.0f C2 %s dura %s takeoff10 %.1f subcl elev %.1f score %s genus %s chir %s" % (
            did, side, rec["qc"]["verdict"], rec["wall_class"], rec["centerline"]["length_mm"], rec["landmarks"].get("C2"),
            rec["landmarks"].get("dura"), rec["takeoff_native"]["10"], rec["subclavian"]["elevation_fwd5_deg"],
            rec["coarseness"].get("score"), (rec.get("surface") or {}).get("genus_measure_json"), rec["frame"]["chirality_sign"]), flush=True)
    old = os.path.join(out_dir, "record.json")
    if os.path.isfile(old):                       # keep sides built earlier and not rebuilt now
        prev = jload(old).get("sides", {})
        for s, v in prev.items():
            unit["sides"].setdefault(s, v)
    unit["pair_complete"] = all(unit["sides"].get(s, {}).get("usable") for s in ("R", "L"))
    jdump(unit, old)
    return unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("ia172", "song", "aortaseg", "vmr0248", "ts"))
    ap.add_argument("--case", default=None)
    ap.add_argument("--sides", default="R,L")
    ap.add_argument("--testing", action="store_true")
    ap.add_argument("--out-root", default=OUT_ROOT, help="registry folder (v5_assign: v5_reg/donors_v5)")
    ap.add_argument("--eligible", default=None, help="build every pair of this final eligible list (primary + reserve)")
    ap.add_argument("--classes", default="primary,reserve")
    ap.add_argument("--only", default=None, help="with --eligible: comma list of cases")
    a = ap.parse_args()
    CFG["out_root"] = a.out_root
    if a.eligible:
        el = jload(a.eligible)
        pairs = [p for c in a.classes.split(",") for p in el.get(c, [])]
        for p in pairs:
            if a.only and p["case"] not in a.only.split(","):
                continue
            src = p["source"]
            case = None if src == "vmr0248" else p["case"]
            build_unit(src, case, ["R", "L"], pair=p)
        return 0
    jobs = TESTING if a.testing else [(a.source, a.case, a.sides)]
    for src, case, sides in jobs:
        build_unit(src, case, [s.strip() for s in sides.split(",") if s.strip()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
