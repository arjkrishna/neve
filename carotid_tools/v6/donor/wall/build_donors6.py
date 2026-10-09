#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\build_donors6.py (= /sp/v6/wall/build_donors6.py in the build containers)
"""build_donors6.py (v6_wall): the v6 donor records ("records like donors_v5") from a ctseg6 variant.

v5's own builder is imported UNCHANGED (/sp/v5_va/graft_v5_reg_donor.py, build_unit with the final eligible list's pair,
as v5_assign ran it) with three inputs swapped per side:
  * wall_files -> the variant's stage-A surface and its VMTK MISR line (/v6w/seg/<variant>/<unit>/<v>_stageA[_vmtk]...),
    wall_tag "ct_seg6_<variant>", surface_md5 = the new surface's md5;
  * the seg folder's landmarks.json = v5's file with every landmark REMAPPED onto the new centreline (the v5 landmark
    point, by arclength on the v5 record line, projected to the nearest point of the new line; the landmark DEFINITIONS
    are vertebra levels, unchanged by a wall fix), edge.json and measure.json = v5's copies (contamination zones and the
    v5 coarseness the assignment ranked by; the assignment is not re-run);
  * out_root /v6w/donors/<variant>; afterwards every "/v6w/" path in the records is rewritten to its repo path
    /repo/carotid_data/v6_work/wall/... (the graft containers mount the repo, not /v6w).
Each side record gains "v6": variant, rule, the seg folder, v5's record / surface md5, floored length (MISR < 1.0) before
and after, the landmark remap shifts. VMR 0248 is built separately (vmrseg6.py) or kept as v5 (see the notes).
    python3 build_donors6.py <variant> [unit,unit,...]     (eve-training-fixed)
"""
import copy
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import graft_v5_reg_donor as GD  # noqa: E402

VAR = sys.argv[1]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
D6 = json.load(open("/sp/v6/wall/donors.json"))
EL = json.load(open("/sp/v5_va/results/v5_eligible_final.json"))
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
SEG = "/v6w/seg/%s" % VAR
OUT = "/v6w/donors/%s" % VAR
REPO_V6W = "/repo/carotid_data/v6_work/wall"
VP = {"R": "rva", "L": "lva"}
SN = {"R": "right", "L": "left"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def project_s(P, X):
    """arclength of the point of polyline P nearest X."""
    s = arclen(P)
    best, bs = np.inf, 0.0
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip((X - a) @ ab / L2, 0, 1))
        d = float(np.linalg.norm(a + t * ab - X))
        if d < best:
            best, bs = d, s[i] + t * np.sqrt(L2)
    return bs, best


def remap_landmarks(unit, v5dir, vdir, sides):
    L5 = json.load(open(os.path.join(v5dir, "landmarks.json")))
    L6 = copy.deepcopy(L5)
    shifts = {}
    for sd in sides:
        nm = SN[sd]
        if nm not in L5.get("sides", {}):
            continue
        r5 = json.load(open(os.path.join(R5, unit, "%s.json" % sd)))
        P5 = np.asarray(r5["centerline"]["points"], float)
        s5 = arclen(P5)
        vj = json.load(open(os.path.join(vdir, "%s_stageA_vmtk.json" % VP[sd])))
        P6 = np.asarray(vj["points"], float)
        if np.linalg.norm(P6[-1] - P5[0]) < np.linalg.norm(P6[0] - P5[0]):
            P6 = P6[::-1]
        L6s = L6["sides"][nm]
        sh = {}
        for key in ("landmarks_rec", "landmarks_used", "landmarks_ct", "ts_qc_landmarks_mm"):
            lm = L6s.get(key)
            if not isinstance(lm, dict):
                continue
            for k, v in list(lm.items()):
                if not isinstance(v, (int, float)) or v is None:
                    continue
                if k == "end":
                    lm[k] = float(arclen(P6)[-1])
                    continue
                X = np.array([np.interp(float(v), s5, P5[:, i]) for i in range(3)])
                sn, dd = project_s(P6, X)
                lm[k] = round(float(sn), 2)
                if key == "landmarks_rec":
                    sh[k] = {"v5": float(v), "v6": round(float(sn), 2), "dist_to_new_line_mm": round(dd, 2)}
        L6s["v6_remap"] = "landmarks remapped from v5's line by nearest point (build_donors6.py); definitions unchanged"
        shifts[sd] = sh
    L6["v6_note"] = "v5 landmarks.json (%s) remapped onto the ct_seg6 %s centrelines" % (v5dir, VAR)
    json.dump(L6, open(os.path.join(vdir, "landmarks.json"), "w"), indent=1)
    return shifts


def floored_mm(P, R):
    s = arclen(P)
    ds = np.r_[np.diff(s), 0.0]
    return float(ds[R < 1.0].sum())


def rewrite(o):
    if isinstance(o, dict):
        return {k: rewrite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rewrite(v) for v in o]
    if isinstance(o, str) and o.startswith("/v6w/"):
        return REPO_V6W + o[4:]
    return o


def build_vmr6():
    """VMR 0248: v5's build_vmr (imported unchanged) pointed at the variant's vmrseg6 output through a stand-in
    derived root (/v6w/vmrroot/vmr0248/ct_seg4 = copies of the variant's surfaces / VMTK lines); landmarks (v5's
    vmr0248_metrics landmarks_ct, mm along v5's line) remapped onto the new line afterwards."""
    vdir = os.path.join(SEG, "vmr0248")
    if not os.path.exists(os.path.join(vdir, ".seg_done")):
        print("SKIP vmr0248: no seg", flush=True)
        return
    fake = "/v6w/vmrroot/%s/vmr0248/ct_seg4" % VAR
    os.makedirs(fake, exist_ok=True)
    for f in os.listdir(vdir):
        if f.endswith((".vtp", ".json")):
            shutil.copy2(os.path.join(vdir, f), os.path.join(fake, f))
    old_der = GD.DER
    GD.DER = "/v6w/vmrroot/%s" % VAR
    try:
        GD.build_unit("vmr0248", None, ["R", "L"], pair=next(p for c in ("primary", "reserve") for p in EL.get(c, [])
                                                          if p["source"] == "vmr0248"))
    finally:
        GD.DER = old_der
    for sd in ("R", "L"):
        f = os.path.join(OUT, "vmr0248", "%s.json" % sd)
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        r5 = json.load(open(os.path.join(R5, "vmr0248", "%s.json" % sd)))
        P5 = np.asarray(r5["centerline"]["points"], float)
        s5 = arclen(P5)
        P6 = np.asarray(r["centerline"]["points"], float)
        sh = {}
        for k in ("C6", "C2", "C1", "dura"):
            v = r5["landmarks"].get(k)
            if v is None:
                continue
            X = np.array([np.interp(float(v), s5, P5[:, i]) for i in range(3)])
            sn, dd = project_s(P6, X)
            r["landmarks"][k] = round(float(sn), 2)
            sh[k] = {"v5": float(v), "v6": round(float(sn), 2), "dist_to_new_line_mm": round(dd, 2)}
        r["landmarks"]["end"] = float(arclen(P6)[-1])
        r["licence_file"] = r5.get("licence_file")
        M6, M5 = np.asarray(r["centerline"]["radii_misr"], float), np.asarray(r5["centerline"]["radii_misr"], float)
        r["v6"] = {"variant": VAR, "script": "/sp/v6/wall/vmrseg6.py", "seg_dir": vdir,
                   "seg_qc_v6": json.load(open(os.path.join(vdir, "seg_qc.json"))).get("v6"),
                   "v5_record": os.path.join(R5, "vmr0248", "%s.json" % sd), "v5_surface_md5": r5["surface"]["md5"],
                   "floored_mm_misr_lt_1": {"v5": round(floored_mm(P5, M5), 1), "v6": round(floored_mm(P6, M6), 1)},
                   "misr_median": {"v5": round(float(np.median(M5)), 3), "v6": round(float(np.median(M6)), 3)},
                   "length_mm": {"v5": round(float(s5[-1]), 1), "v6": round(float(arclen(P6)[-1]), 1)},
                   "landmark_remap": sh, "stand_in_root": "/v6w/vmrroot/%s" % VAR,
                   "coarseness_note": "coarseness.* = v5's vmr0248_metrics (pre-fix surface)"}
        json.dump(rewrite(r), open(f, "w"), indent=1)
    uf = os.path.join(OUT, "vmr0248", "record.json")
    json.dump(rewrite(json.load(open(uf))), open(uf, "w"), indent=1)
    print("BUILT vmr0248", flush=True)


def main():
    if ONLY == ["vmr0248"]:
        GD.CFG["out_root"] = OUT
        build_vmr6()
        return
    pairs = {(p["case"] if p["case"].startswith("ia172") else "song_" + p["case"]): (c, p)
             for c in ("primary", "reserve") for p in EL.get(c, []) if p["source"] != "vmr0248"}
    GD.CFG["out_root"] = OUT
    done = []
    for unit, cfg in D6.items():
        if unit == "vmr0248" or (ONLY and unit not in ONLY):
            continue
        vdir = os.path.join(SEG, unit)
        if not os.path.exists(os.path.join(vdir, ".seg_done")):
            print("SKIP %s: no seg" % unit, flush=True)
            continue
        cls, pair = pairs[unit]
        pair = copy.deepcopy(pair)
        sides = [s for s in ("R", "L") if os.path.exists(os.path.join(vdir, "%s_stageA_vmtk.json" % VP[s]))]
        v5dir = cfg["v5_seg_dir"]
        for f in ("edge.json", "measure.json"):
            if os.path.exists(os.path.join(v5dir, f)):
                shutil.copy2(os.path.join(v5dir, f), os.path.join(vdir, f))
        shifts = remap_landmarks(unit, v5dir, vdir, sides)
        for sd in ("R", "L"):
            wf = pair["sides"][sd].get("wall_files") or {}
            if sd in sides:
                sv = os.path.join(vdir, "%s_stageA.vtp" % VP[sd])
                pair["sides"][sd]["wall_files"] = {"surface_vtp": sv, "surface_md5": md5(sv),
                                                   "vmtk_centerline_json": os.path.join(vdir, "%s_stageA_vmtk.json" % VP[sd]),
                                                   "measure_json": os.path.join(vdir, "measure.json"),
                                                   "v5_wall_files": wf}
                pair["sides"][sd]["wall_tag"] = "ct_seg6_%s" % VAR
            else:
                pair["sides"][sd]["wall_files"] = {}
        unit_rec = GD.build_unit(cfg["source"], cfg["case"], ["R", "L"], pair=pair)
        # v6 block + path rewrite
        for sd in ("R", "L"):
            f = os.path.join(OUT, unit, "%s.json" % sd)
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            r5 = json.load(open(os.path.join(R5, unit, "%s.json" % sd)))
            P6, M6 = np.asarray(r["centerline"]["points"], float), np.asarray(r["centerline"]["radii_misr"], float)
            P5, M5 = np.asarray(r5["centerline"]["points"], float), np.asarray(r5["centerline"]["radii_misr"], float)
            r["v6"] = {"variant": VAR, "script": "/sp/v6/wall/ctseg6.py", "seg_dir": vdir,
                       "seg_qc_v6": json.load(open(os.path.join(vdir, "seg_qc.json"))).get("v6"),
                       "v5_record": os.path.join(R5, unit, "%s.json" % sd), "v5_surface_md5": r5["surface"]["md5"],
                       "floored_mm_misr_lt_1": {"v5": round(floored_mm(P5, M5), 1), "v6": round(floored_mm(P6, M6), 1)},
                       "misr_median": {"v5": round(float(np.median(M5)), 3), "v6": round(float(np.median(M6)), 3)},
                       "length_mm": {"v5": round(float(arclen(P5)[-1]), 1), "v6": round(float(arclen(P6)[-1]), 1)},
                       "landmark_remap": shifts.get(sd),
                       "vmtk_note": json.load(open(os.path.join(vdir, "%s_stageA_vmtk.json" % VP[sd]))).get("note"),
                       "coarseness_note": "coarseness.* = v5's measure.json / tally (pre-fix surface; the assignment ranked "
                                          "by them and is not re-run); the v6 wall detail is in wall/wd (wd_proxy.py)"}
            json.dump(rewrite(r), open(f, "w"), indent=1)
        uf = os.path.join(OUT, unit, "record.json")
        json.dump(rewrite(json.load(open(uf))), open(uf, "w"), indent=1)
        done.append(unit)
        print("BUILT %s sides %s" % (unit, sides), flush=True)
    if not ONLY or "vmr0248" in ONLY:
        build_vmr6()
    print("built", len(done))


if __name__ == "__main__":
    main()
