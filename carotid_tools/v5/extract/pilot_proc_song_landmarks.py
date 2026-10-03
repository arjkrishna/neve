#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_landmarks.py (= /sp/v5_va/pilot_proc_song_landmarks.py in the build containers)
# SONG COPY (pilot_proc_song, 2026-10-01 21:20): pilot_proc_ia172_landmarks.py with the song paths (thin_slice/song), pilot_proc_song_ imports and the Song meta keys; diffs beyond that are stated in the docstring / marked 'song'.
"""pilot_proc_song_landmarks.py -- V1..V4 landmarks on the CT-segmented (ct_seg4) VA centrelines of one 172-series pilot
donor (PILOT_CASE). Same definitions as tag_ctwall_landmarks.py (TaG-Net A), so the two are graded alike.

Primary ("ts"): art_extract.load_ts + art_extract.ts_qc, UNCHANGED (imported), on each ct_seg4 VMTK centreline
(<der>/ct_seg/<tag>/<v>_stageA_vmtk.json), against the TotalSegmentator total_v3 masks of the case (1.5 mm RAS copy
<der>/totalseg_1p5): C6 = the start of the C6 level's own enclosed run (tag_ctwall's fix of ts_qc's C7 over-call; ts_qc's
raw entry is kept), C2 = end of the C2 enclosed run, C1 = end of the C1-mask enclosure (it continues along the groove on
the posterior arch of C1), dura = first s after C1 with z >= z_FM = C1 top + 6 mm. V3-loop numbers: tag_totalseg_qc.v3loop
(unchanged, imported). Entry level also by tag_calibration_entry.entry "Erec" (unchanged, imported; reported only).
Secondary ("ct", VMR's method, the verifier's own code verify_tag_ctwall_image.enclosure / runs_of, imported unchanged):
every 0.5 mm 48 normal-plane rays, a ray is enclosed if it meets HU >= 400 within r + 0.3 .. 7 mm; runs of enclosure >= 0.5
(merged over <= 1 mm gaps, >= 1 mm long), each run assigned to a vertebra by the TS total_v3 masks at the ray hit points
(majority, +-1 voxel tolerance; the verifier's ts_level_lookup re-pointed to this case's masks). C6_ct = start of the first
run at the entry level (lowest of C7..C3; C7 only if a C6 run does not start within 2 mm after it ends), C2_ct = end of the
last C2 run, C1_ct = end of the FIRST C1 run (= the exit from the C1 transverse foramen, VMR's convention), dura = the ts
dura. Third ("rec", added by the third instance of this agent, 2026-10-01 20:40): tag_calibration's recommended set from
tag_calibration_paths.one + tag_calibration.augment (both unchanged, imported) on the same centreline: C6 = the Erec
entry landmark, C2 = ts_qc, C1 = C1f (exit of the first C1 run after C2), dura = ts_qc; plus the robust-cord V3 loop.
Writes <der>/ct_seg/<tag>/landmarks.json.  neve-build-stagea.
"""
import json
import os
import sys
import time

import numpy as np
import nibabel as nib

sys.path.insert(0, "/sp/v5_va")
import art_extract as AE  # noqa: E402  (unchanged)
import va_metrics as vm  # noqa: E402  (unchanged)
from tag_totalseg_qc import v3loop  # noqa: E402  (unchanged)
from tag_calibration_entry import entry as erec_entry  # noqa: E402  (unchanged)
import verify_tag_ctwall_image as VI  # noqa: E402  (unchanged: enclosure, runs_of, Vol)
import tag_calibration_paths as TP  # noqa: E402  (unchanged; "rec" set)
import tag_calibration as TC  # noqa: E402  (unchanged; "rec" set)

CASE = os.environ["PILOT_CASE"]
SRC = "/repo/carotid_data/va_sources/thin_slice/song/%s" % CASE
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % CASE
TS15 = os.path.join(DER, "totalseg_1p5")
TAGS = [t for t in os.environ.get("PILOT_SEG_TAGS", "ct_seg4").split(",") if t]


def level_lookup():
    """verify_tag_ctwall_image.ts_level_lookup with this case's masks (logic copied)."""
    lev, A = None, None
    for k, nm in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"], start=1):
        p = "%s/segmentations/vertebrae_%s.nii.gz" % (TS15, nm)
        if not os.path.isfile(p):
            continue
        im = nib.load(p)
        m = np.asanyarray(im.dataobj) > 0
        if lev is None:
            lev = np.zeros(m.shape, np.uint8)
            A = im.affine
        lev[m & (lev == 0)] = k
    Ai = np.linalg.inv(A)
    names = ["none", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"]

    def look(P):
        if not len(P):
            return []
        q = np.round(P @ Ai[:3, :3].T + Ai[:3, 3]).astype(int)
        ok = np.all((q >= 0) & (q < np.array(lev.shape)), 1)
        out = np.zeros(len(P), int)
        out[ok] = lev[q[ok, 0], q[ok, 1], q[ok, 2]]
        miss = np.nonzero(ok & (out == 0))[0]
        for t in miss:
            a = np.maximum(q[t] - 1, 0)
            b = np.minimum(q[t] + 2, np.array(lev.shape))
            nb = lev[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
            nz = nb[nb > 0]
            if len(nz):
                out[t] = np.bincount(nz).argmax()
        return [names[v] for v in out]
    return look


def ct_volume(lo, hi):
    """float32 crop of the native CT around [lo, hi] + 15 mm (the verifier's Vol over it)."""
    im = nib.load(os.path.join(SRC, "ct.nii.gz"))
    A = im.affine
    Ai = np.linalg.inv(A)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = c @ Ai[:3, :3].T + Ai[:3, 3]
    a = np.maximum(np.floor(ijk.min(0)).astype(int) - 40, 0)
    b = np.minimum(np.ceil(ijk.max(0)).astype(int) + 40, np.array(im.shape) - 1)
    blk = np.asarray(im.dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).astype(np.float32)

    def w2i(P):
        return P @ Ai[:3, :3].T + Ai[:3, 3] - a
    return VI.Vol(blk, w2i)
def ct_landmarks(vol, look, P, R, dura):
    q = np.arange(0.0, vm.arclen(P)[-1] + 1e-9, 0.25)
    s0 = vm.arclen(P)
    Q = np.stack([np.interp(q, s0, P[:, i]) for i in range(3)], 1)
    RR = np.interp(q, s0, R)
    prof, hits = VI.enclosure(vol, q, Q, RR)
    runs = []
    for a, b in VI.runs_of(prof):
        lv = [x for h in hits[a:b + 1] for x in look(h)]
        lv = [x for x in lv if x != "none"]
        level = max(set(lv), key=lv.count) if lv else "none"
        runs.append({"s": [float(prof[a, 0]), float(prof[b, 0])], "z": [round(float(prof[a, 2]), 1), round(float(prof[b, 2]), 1)],
                     "mean_enc": round(float(prof[a:b + 1, 1].mean()), 2), "level": level,
                     "level_share": round(lv.count(level) / max(len(lv), 1), 2) if lv else 0})
    lm = {}
    by = {}
    for r in runs:
        by.setdefault(r["level"], []).append(r)
    entry = None
    for L in ("C7", "C6", "C5", "C4", "C3"):
        if L in by:
            if L == "C7" and "C6" in by:
                c6 = [r for r in by["C6"] if r["s"][0] > by["C7"][0]["s"][0]]
                if c6 and c6[0]["s"][0] - by["C7"][0]["s"][1] < 2.0:
                    continue
            entry = L
            break
    if entry:
        lm["C6"] = by[entry][0]["s"][0]
    if "C2" in by:
        lm["C2"] = by["C2"][-1]["s"][1]
    if "C1" in by:
        c1 = [r for r in by["C1"] if r["s"][0] > lm.get("C2", -1) - 5]
        if c1:
            lm["C1"] = c1[0]["s"][1]
    lm["dura"] = dura
    return {"runs": runs, "entry_level": entry, "landmarks": lm}


def main():
    t0 = time.time()
    ts = AE.load_ts(TS15, None, None)
    z_fm = ts["z_c1_top"] + 6.0
    look = level_lookup()
    allP = []
    for tag in TAGS:
        for v in ("rva", "lva"):
            p = os.path.join(DER, "ct_seg", tag, "%s_stageA_vmtk.json" % v)
            if os.path.exists(p):
                allP.append(np.asarray(json.load(open(p))["points"], float))
    if not allP:
        print("no ct_seg centrelines"); return
    allP = np.vstack(allP)
    vol = ct_volume(allP.min(0), allP.max(0))
    vz = {}
    for k in range(1, 8):
        m, a = ts["masks"].get("vertebrae_C%d" % k, (None, None))
        if m is not None:
            W = AE.i2w(a, np.argwhere(m)[::3].astype(float))
            vz["C%d" % k] = (float(W[:, 2].min()), float(W[:, 2].max()))
    for tag in TAGS:
        D = os.path.join(DER, "ct_seg", tag)
        res = {"what": __doc__, "case": CASE, "z_fm_mm": round(z_fm, 2), "sides": {}}
        for side, v, sd in (("R", "rva", "right"), ("L", "lva", "left")):
            p = os.path.join(D, "%s_stageA_vmtk.json" % v)
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
            Pr, Rr, s = vm.resample(P, R)
            q = AE.ts_qc({"P": P, "start_on_sa_world": P[0]}, ts, side, z_fm)
            lv = q["vertebra_levels"]
            lm = dict(q.get("landmarks_mm", {}))
            c6 = (lv.get("C6") or {}).get("s_enclosed_mm")
            used = {"C6": None if not c6 else c6[0], "C2": lm.get("C2"), "C1": lm.get("C1"), "dura": lm.get("dura"),
                    "end": round(float(s[-1]), 2)}
            rec = {"centerline": p, "length_mm": round(float(s[-1]), 2), "ts_qc_entry_level": q.get("foramen_entry_level"),
                   "ts_qc_landmarks_mm": lm, "landmarks_used": used, "landmarks_complete": q.get("landmarks_complete"),
                   "foramen_levels_passed": q.get("foramen_levels_passed"), "n_C6_to_C1_passed": q.get("n_C6_to_C1_passed"),
                   "enclosed_runs_per_level_mm": {k: vv.get("s_enclosed_mm") for k, vv in lv.items()},
                   "max_enclosure": {k: vv.get("max_enclosure") for k, vv in lv.items()},
                   "median_dist_to_vertebra_mm": {k: vv.get("median_dist_to_vertebra_mm") for k, vv in lv.items()},
                   "origin_to_TS_SA_mask_mm": q.get("origin_to_TS_SA_mask_mm")}
            try:
                rec["v3loop"] = v3loop(Pr, ts, z_fm, lm.get("dura"))
            except Exception as e:  # noqa: BLE001
                rec["v3loop"] = "failed: %s" % e
            # Erec entry
            prof = {"s": [], "z": [], "d": {}}
            P1, _, s1 = AE.resample(P, 1.0)
            prof["s"], prof["z"] = s1.tolist(), P1[:, 2].tolist()
            from scipy import ndimage
            from scipy.ndimage import map_coordinates
            for k in range(1, 8):
                L = "C%d" % k
                if "vertebrae_" + L not in ts["masks"]:
                    continue
                m, a = ts["masks"]["vertebrae_" + L]
                dm = ndimage.distance_transform_edt(~m, sampling=ts["sp"])
                I = (np.linalg.inv(a) @ np.c_[P1, np.ones(len(P1))].T)[:3]
                prof["d"][L] = map_coordinates(dm, I, order=1, mode="nearest").tolist()
            el, es = erec_entry(prof, 0.5, 5.0, 2.0, vz)
            rec["entry_Erec"] = {"level": el, "s_mm": None if es is None else round(float(es), 1)}
            # recommended set (tag_calibration): Erec C6, ts_qc C2, C1f, ts_qc dura
            vrec = TP.one("song", P, P[0], side, "total_v3", TS15)
            TC.augment(vrec)
            le = vrec["entry"].get("Erec_landmarks_mm") or {}
            rec["landmarks_rec"] = {"C6": le.get("entry"), "C2": le.get("C2"), "C1": le.get("C1f"), "dura": lm.get("dura"),
                                    "end": round(float(s[-1]), 2)}
            rec["entry_level_rec"] = vrec["entry"].get("Erec_entry")
            rec["v3loop_robust"] = vrec["v3loop_robust"]
            # CT method
            rec["ct_method"] = ct_landmarks(vol, look, Pr, Rr, lm.get("dura"))
            lmc = rec["ct_method"]["landmarks"]
            rec["landmarks_ct"] = {"C6": lmc.get("C6"), "C2": lmc.get("C2"), "C1": lmc.get("C1"), "dura": lmc.get("dura"),
                                   "end": round(float(s[-1]), 2)}
            rec["landmark_z_mm"] = {k: (None if val is None else round(float(Pr[int(np.argmin(np.abs(s - val))), 2]), 1))
                                    for k, val in used.items()}
            res["sides"][sd] = rec
            print(CASE, tag, sd, "ts", json.dumps(used), "entry", rec["ts_qc_entry_level"], "Erec", json.dumps(rec["entry_Erec"]),
                  "| ct", json.dumps(rec["landmarks_ct"]), rec["ct_method"]["entry_level"], "| v3loop", json.dumps(rec["v3loop"]), flush=True)
        res["seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(os.path.join(D, "landmarks.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
