# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_prearch.py (= /sp/v5_va/scaleup_ia172_prearch.py in the build containers)
"""scaleup_ia172_prearch: PRE-FETCH arch check for the 172-series scale-up (scaleup_lane_ia172).

The pilot's lesson (pilot_fetch_ia172.md section 1, "Lesson for bulk selection"; 166 / 090 / 118): z >= 300-350 mm
does not guarantee the arch, and Arterial's start point needs the bottom slice to cut the arch into TWO limbs
(ascending + descending). pilot_ia172_archcut.py checks this AFTER the fetch with the quick-look TotalSegmentator aorta
mask; this script does it BEFORE the fetch, mask-free, from a few range-read DICOM members (~200 KB each): the
lowest-z slice and the slices 5 and 10 mm above it.

Per slice: HU > 150 inside the body, opened with a 1.5 mm disk (splits a limb touching the vertebral body), components
with area >= 150 mm^2; per component the area, the fill of its max-radius circle, and the median / IQR (std recorded) of
its eroded interior.
  descending limb = round (fill >= 0.55), area 250-1500 mm^2, homogeneous (interior IQR <= 90 HU: blood pool; robust to
                    calcification and contrast streaks, which broke a std rule on 062 / 139; trabecular bone is wider), median 180-800 HU (dense undiluted contrast and cortex excluded);
  anterior partner = area >= 400 mm^2, (IQR <= 90 HU or fill >= 0.6), median 180-800 HU, >= 25 mm anterior of the descending limb and
                    >= 30 mm from it (the ascending aorta; when the slice lies below the arch it may be merged with the
                    pulmonary trunk / heart, which is still "arch fully in view").
TWO LIMBS = a descending limb with an anterior partner (spread = min(IQR, std)). PASS = two limbs on the lowest slice, or on the slice 5 or 10 mm
above it (a slice cut at the arch APEX shows one elongated component and nothing round behind it, and nothing at all
above the apex, so +5 / +10 mm do not admit an apex-only cut; they only rescue a limb merged with bone or heart at
the bottom: limbs at +10 mm imply limbs at the bottom). A FAIL is probed at +20 ... +80 mm to record why (apex / above
the arch, or an unenhanced bottom), and every FAIL PNG is read by eye (the lane notes record the verdict).
Calibrated on the 12 pilot cases (10 two-limb, 090 / 118 apex-only per pilot_ia172_archcut.json), run with --local on
their stored DICOM. Rows increase posteriorly (HFS, IOP [1,0,0,0,1,0], all 172-series series).
A PNG per case (the three slices, components outlined, green = limb / partner) is written for the eye.

usage: python3 scaleup_ia172_prearch.py [--local] <sid> [<sid> ...]     (sid = zip CaseID, e.g. 062)
  --local: read the members from thin_slice/ia172/ia172_<sid>/dicom/ instead of the zip (calibration)
out: /sp/v5_va/results/scaleup_ia172_prearch.json (merged per sid), /sp/v5_va/results/scaleup_ia172_prearch/<sid>.png
Container: neve-build-stagea, PYTHONPATH=/sp/va_pool/pylib (pydicom + pylibjpeg).
"""
import io
import json
import os
import sys
import time

import numpy as np
import pydicom
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/sp/v5_va")
from pilot_ia172_zip import build_index, get_member  # noqa: E402

HDR = json.load(open("/sp/v5_va/results/pilot_ia172_headers.json"))
OUT = "/sp/v5_va/results/scaleup_ia172_prearch.json"
PNGD = "/sp/v5_va/results/scaleup_ia172_prearch"
IMG = "/repo/carotid_data/va_sources/thin_slice/ia172"
RULE = {"hu_thr": 150, "open_mm": 1.5, "min_area_mm2": 150,
        "desc": {"area_mm2": [250, 1500], "fill_min": 0.55, "iqr_max": 90, "median_hu": [180, 800]},
        "partner": {"area_min_mm2": 400, "iqr_max": 90, "or_fill_min": 0.6, "median_hu": [180, 800], "anterior_min_mm": 25,
                    "dist_min_mm": 30},
        "spread": "min(IQR, std) of the eroded interior",
        "pass": "two limbs on the lowest slice or on the slice 5 or 10 mm above it",
        "probe_if_fail_mm": [20, 30, 40, 60, 80]}


def slice_files(sid):
    h = HDR[sid]
    n = h["n_files"]
    z0 = h["first"]["ImagePositionPatient"][2]
    z1 = h["last"]["ImagePositionPatient"][2]
    f0 = int(h["files"]["first"].split(".")[0])
    f1 = int(h["files"]["last"].split(".")[0])
    dz = abs(z1 - z0) / max(n - 1, 1)
    if z0 <= z1:
        low, step = f0, (1 if f1 > f0 else -1)
    else:
        low, step = f1, (-1 if f1 > f0 else 1)
    return [(up, "%d.dcm" % (low + step * int(round(up / dz)))) for up in (0.0, 5.0, 10.0)], dz, min(z0, z1)


def probe_file(sid, up):
    h = HDR[sid]
    n = h["n_files"]
    z0, z1 = h["first"]["ImagePositionPatient"][2], h["last"]["ImagePositionPatient"][2]
    f0, f1 = int(h["files"]["first"].split(".")[0]), int(h["files"]["last"].split(".")[0])
    dz = abs(z1 - z0) / max(n - 1, 1)
    low, step = (f0, (1 if f1 > f0 else -1)) if z0 <= z1 else (f1, (-1 if f1 > f0 else 1))
    return "%d.dcm" % (low + step * int(round(up / dz)))


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def analyse(ds):
    a = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    ps = [float(x) for x in ds.PixelSpacing]
    px = ps[0] * ps[1]
    # body = everything above -300 HU, holes filled (NOT the largest component: a chest wall cut open by the FOV edge
    # leaves the mediastinum as a separate component, 136)
    body = ndimage.binary_fill_holes(a > -300)
    m = (a > RULE["hu_thr"]) & body
    m = ndimage.binary_opening(m, structure=disk(max(1, int(round(RULE["open_mm"] / ps[0])))))
    lab, n = ndimage.label(m)
    comps = []
    if n:
        areas = np.bincount(lab.ravel()) * px
        for i in range(1, n + 1):
            if areas[i] < RULE["min_area_mm2"]:
                continue
            mm = lab == i
            ij = np.argwhere(mm)
            c = ij.mean(axis=0)
            r = np.sqrt((((ij - c) * ps) ** 2).sum(axis=1)).max()
            er = ndimage.binary_erosion(mm, iterations=3)
            vals = a[er] if er.sum() >= 20 else a[mm]
            comps.append({"label": int(i), "area_mm2": round(float(areas[i]), 0),
                          "centre_rc_mm": [round(float(c[0] * ps[0]), 1), round(float(c[1] * ps[1]), 1)],
                          "fill_of_circle": round(float(areas[i] / (np.pi * r * r)), 2),
                          "median_hu": round(float(np.median(vals)), 0), "std_hu": round(float(np.std(vals)), 0),
                          "iqr_hu": round(float(np.subtract(*np.percentile(vals, [75, 25]))), 0)})
    comps.sort(key=lambda d: -d["area_mm2"])
    return a, lab, comps


def is_desc(c):
    R = RULE["desc"]
    return (R["area_mm2"][0] <= c["area_mm2"] <= R["area_mm2"][1] and c["fill_of_circle"] >= R["fill_min"]
            and min(c["iqr_hu"], c["std_hu"]) <= R["iqr_max"] and R["median_hu"][0] <= c["median_hu"] <= R["median_hu"][1])


def is_partner(c, d):
    R = RULE["partner"]
    if not (c["area_mm2"] >= R["area_min_mm2"] and (min(c["iqr_hu"], c["std_hu"]) <= R["iqr_max"] or c["fill_of_circle"] >= R["or_fill_min"])
            and R["median_hu"][0] <= c["median_hu"] <= R["median_hu"][1]):
        return False
    ant = d["centre_rc_mm"][0] - c["centre_rc_mm"][0]
    dist = float(np.hypot(ant, d["centre_rc_mm"][1] - c["centre_rc_mm"][1]))
    return ant >= R["anterior_min_mm"] and dist >= R["dist_min_mm"]


def verdict(comps):
    best = None
    for d in comps:
        if not is_desc(d):
            continue
        for c in comps:
            if c is d or not is_partner(c, d):
                continue
            score = d["area_mm2"] + c["area_mm2"]
            if best is None or score > best["score"]:
                best = {"score": score, "desc": d["label"], "partner": c["label"],
                        "dist_mm": round(float(np.hypot(d["centre_rc_mm"][0] - c["centre_rc_mm"][0],
                                                        d["centre_rc_mm"][1] - c["centre_rc_mm"][1])), 1)}
    return best


def main():
    args = sys.argv[1:]
    local = "--local" in args
    sids = [s for s in args if s != "--local"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    idx = None if local else build_index()
    os.makedirs(PNGD, exist_ok=True)
    for sid in sids:
        t0 = time.time()
        files, dz, zlow = slice_files(sid)
        rec = {"sid": sid, "mode": "local" if local else "remote", "interval_mm": round(dz, 3), "z_lowest": zlow,
               "rule": RULE, "slices": {}}
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.3))
        for k, (up, fn) in enumerate(files):
            if local:
                raw = open(os.path.join(IMG, "ia172_" + sid, "dicom", fn), "rb").read()
            else:
                raw = get_member(idx, "Studies/%s/%s" % (sid, fn))
            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            a, lab, comps = analyse(ds)
            best = verdict(comps)
            rec["slices"]["%dmm" % up] = {"file": fn, "ipp_z": float(ds.ImagePositionPatient[2]),
                                          "two_limbs": best is not None, "pair": best, "components": comps[:8]}
            ax = axs[k]
            ax.imshow(np.clip(a, -200, 600), cmap="gray")
            sel = set() if best is None else {best["desc"], best["partner"]}
            for c in comps[:8]:
                ax.contour(lab == c["label"], levels=[0.5], colors=["lime" if c["label"] in sel else "red"],
                           linewidths=0.8)
                ax.text(c["centre_rc_mm"][1] / float(ds.PixelSpacing[1]), c["centre_rc_mm"][0] / float(ds.PixelSpacing[0]),
                        "%d|%.2f|%d" % (c["median_hu"], c["fill_of_circle"], c["iqr_hu"]), color="yellow", fontsize=7)
            ax.set_title("%s +%d mm (z %.1f): %s" % (sid, up, float(ds.ImagePositionPatient[2]),
                         "two limbs, d %.0f mm" % best["dist_mm"] if best else "no two limbs"), fontsize=9)
            ax.axis("off")
        rec["two_limbs_bottom"] = rec["slices"]["0mm"]["two_limbs"]
        rec["two_limbs_5mm"] = rec["slices"]["5mm"]["two_limbs"]
        rec["two_limbs_10mm"] = rec["slices"]["10mm"]["two_limbs"]
        rec["pass"] = bool(rec["two_limbs_bottom"] or rec["two_limbs_5mm"] or rec["two_limbs_10mm"])
        if not rec["pass"]:   # record WHY: the first level with two enhanced limbs (none = above / at the apex)
            rec["probe"] = {}
            for up in RULE["probe_if_fail_mm"]:
                fn = probe_file(sid, up)
                if local:
                    raw = open(os.path.join(IMG, "ia172_" + sid, "dicom", fn), "rb").read()
                else:
                    raw = get_member(idx, "Studies/%s/%s" % (sid, fn))
                ds = pydicom.dcmread(io.BytesIO(raw), force=True)
                best = verdict(analyse(ds)[2])
                rec["probe"]["%dmm" % up] = {"file": fn, "ipp_z": float(ds.ImagePositionPatient[2]), "two_limbs": best is not None}
                if best is not None:
                    break
            lv = [k for k, v in rec["probe"].items() if v["two_limbs"]]
            rec["first_two_limb_level"] = lv[0] if lv else None
        rec["seconds"] = round(time.time() - t0, 1)
        rec["date"] = time.strftime("%Y-%m-%d %H:%M")
        fig.suptitle("%s prearch: %s (green = descending limb + anterior partner; label = median HU | fill | IQR)"
                     % (sid, "PASS" if rec["pass"] else "FAIL"))
        fig.tight_layout()
        fig.savefig(os.path.join(PNGD, "%s.png" % sid), dpi=70)
        plt.close(fig)
        res[sid] = rec
        json.dump(res, open(OUT, "w"), indent=1)
        print(sid, "PASS" if rec["pass"] else "FAIL", "two limbs at 0 / +5 / +10 mm:", rec["two_limbs_bottom"],
              rec["two_limbs_5mm"], rec["two_limbs_10mm"], "| %.1f s" % rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
