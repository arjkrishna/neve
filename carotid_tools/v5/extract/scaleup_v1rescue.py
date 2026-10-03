#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_v1rescue.py (= /sp/v5_va/scaleup_v1rescue.py in the build containers)
"""scaleup_v1rescue.py <source> <case> -- PIPELINE v2 fix 1: the V1 MASK-BREAK RESCUE (scaleup_fixes, 2026-10-02).

Problem (pilot): Arterial's binary mask sometimes breaks in V1 although the lumen is enhanced; art_extract then bridges the
break (8 mm cost per mm outside the mask) and the recalibrated gate fails the side on rule 3a (V1 bridge > 3 mm) or passes it
only through the provisional V1 <= 3 mm limit (172_061 L 5.6 mm at 310 HU, Song 141 R 4.6 mm at 587 HU, Song 044 R 9.1 mm at
480 HU; 172_001 R 2.1 mm / Song 109 R 2.0 mm pass provisionally). A bridge through fat / soft tissue (172_041, 073 L: 177 /
128 / -25 HU) is a real non-vessel shortcut and must stay failed.

APPLIES to a side (of the trace set selected by the base selection, qc/<SCALEUP_SELECT> or tsba) when ALL hold:
  (a) it has >= 1 bridge in segment "V1" (the qc file's bridge rows: origin -> the Erec C6 landmark);
  (b) EVERY V1 bridge runs through enhanced lumen: its native-CT HU median along the bridge >= 0.6 x the native CCA median
      (qc cca_median_HU_native) -- the same 0.6 x CCA as the gate's rule 5b;
  (c) its recalibrated-gate failures (if any) are bridge failures only: subset of {1_origin_le_2mm, 3a_no_V1_bridge,
      3d_total_le_25}. (A side that passes, but only through the provisional V1 <= 3 mm limit, also applies.)
RESCUE per V1 bridge (Arterial mask grid, native CT sampled trilinear at the mask voxel centres):
  span    = the bridge's s-range on the trace [s, s + length] widened by 2 mm, united with the out-of-mask run(s) of the
            trace points that overlap it (a trace point is out of the mask when no mask voxel lies within 0.6 voxel
            diagonals);
  capsule = voxels within r_c of the trace polyline over the span, r_c = clip(1.6 r_VA + 0.5, 2.5, 4.0) mm, r_VA = median
            EDT radius of the trace within 10 mm of the span outside it;
  grow    = capsule & (mask | (T_lo <= HU <= T_hi)), T_lo = 0.5 x CCA (art_extract's "vessel-like" bound; fat and soft
            tissue < 150 HU never pass), T_hi = 1.5 x CCA (cortical bone / undiluted venous contrast excluded);
  connectivity check: one 26-connected component of grow must hold BOTH ends -- mask voxels within r_c of the trace 0.5-3 mm
            below the span (at s = 0: within r_c of the trace start, the subclavian junction) and 0.5-3 mm above it;
  lumen check: the added voxels (component & ~mask) have a native HU median >= 0.6 x CCA, and along the span (0.5 mm
            stations, slab +-0.5 mm normal to the trace) the component's cross-section never falls below an equivalent
            radius max(0.5 mm, 0.35 r_VA) (no thread of partial-volume voxels);
  -> the added voxels join the mask. A side is RESCUED only if every V1 bridge of it passes; otherwise its mask is untouched.
OUTPUT (only when >= 1 side is rescued): arterial/runs/native_v1r/ = a copy of runs/native's inputs to art_extract
  (run_report.json, prep.json, skull_mask.nii.gz, labelled_centerlines.json if present) + extracranial_vessels/
  segmentation.nii.gz = the patched mask (same grid, header and dtype; added voxels = 1), + v1rescue.json there.
  Always: qc/v1rescue.json (applicability, per bridge numbers, verdict) and, when rescued, qc/v1rescue_<case>.png (MIPs of
  the CT slab with the mask and the added voxels around each bridge).
The driver (scaleup_v2_post.sh stage "rescue") then re-traces runs/native_v1r with the SAME tool as the base trace set
  (tsba: pilot_art_extract.py; ts: art_extract.py) into arterial/<ext>_v1r, runs VMTK and the unchanged pilot QC on it
  (qc/qc_<ext>_v1r.json), and scaleup_select.py decides per side (see there).
neve-build-stagea (nibabel, scipy, matplotlib). Licence: derived through Arterial -> non-commercial.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

SRCN, CASE = sys.argv[1], sys.argv[2]
assert SRCN in ("ia172", "song")
SRC = "/repo/carotid_data/va_sources/thin_slice/%s/%s" % (SRCN, CASE)
DER = "/repo/carotid_data/va_sources/derived/thin_slice/%s/%s" % (SRCN, CASE)
SEL = os.environ.get("SCALEUP_SELECT", "select.json")
BRIDGE_FAILS = {"1_origin_le_2mm", "3a_no_V1_bridge", "3d_total_le_25"}
K_ENH, K_LO, K_HI = 0.6, 0.5, 1.5


def rr(x, n=2):
    return None if x is None else round(float(x), n)


def arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def L(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def base_ext(side):
    s = L(os.path.join(DER, "qc", SEL))
    if s is None and SEL != "select.json":
        s = L(os.path.join(DER, "qc", "select.json"))
    e = ((s or {}).get("sides") or {}).get(side, {}).get("ext") or "tsba"
    return e[:-4] if e.endswith("_v1r") else e     # the rescue always starts from the base (un-rescued) trace


def qc_of(ext):
    return L(os.path.join(DER, "qc", "qc.json" if ext == "tsba" else "qc_%s.json" % ext))


t0 = time.time()
out = {"what": __doc__.split("\n\n")[0], "case": CASE, "source": SRCN, "select_file": SEL,
       "rule": {"enhanced_bridge": "HU median along the bridge >= %.1f x CCA (native)" % K_ENH,
                "allowed_failures": sorted(BRIDGE_FAILS), "grow_HU": "[%.1f, %.1f] x CCA" % (K_LO, K_HI)},
       "sides": {}}
todo = {}
for side in ("R", "L"):
    ext = base_ext(side)
    q = qc_of(ext) or {}
    r = (q.get("sides") or {}).get(side) or {}
    cca = q.get("cca_median_HU_native")
    rec = {"base_trace_set": ext, "status": r.get("status"), "fails": r.get("gate_fail_reasons"), "cca_HU": cca}
    out["sides"][side] = rec
    if not r.get("trace_ok") or cca is None:
        rec.update(applies=False, why="no trace / no CCA HU")
        continue
    v1 = [b for b in (r.get("bridges") or []) if b.get("segment") == "V1"]
    rec["v1_bridges"] = [dict(b, ratio_to_CCA=rr((b.get("HU_median_native") or np.nan) / cca, 3)) for b in v1]
    fails = set(r.get("gate_fail_reasons") or [])
    if not v1:
        rec.update(applies=False, why="no V1 bridge")
        continue
    bad = [b for b in v1 if b.get("HU_median_native") is None or not np.isfinite(b["HU_median_native"])
           or b["HU_median_native"] < K_ENH * cca]
    if bad:
        rec.update(applies=False, why="V1 bridge(s) not through enhanced lumen (< %.1f x CCA %.0f HU): %s" % (
            K_ENH, cca, ", ".join("%.1f mm at s %.1f, %s HU" % (b["length_mm"], b["s_mm"], b["HU_median_native"]) for b in bad)))
        continue
    other = sorted(fails - BRIDGE_FAILS)
    if other:
        rec.update(applies=False, why="other gate failures: %s" % other)
        continue
    rec["applies"] = True
    rec["why"] = ("fails only on V1 bridges through enhanced lumen" if fails else
                  "passes only with V1 bridge(s) <= 3 mm (provisional limit); bridges through enhanced lumen")
    todo[side] = (ext, r, cca, v1)

rescued = {}
if todo:
    mp = os.path.join(DER, "arterial/runs/native/extracranial_vessels/segmentation.nii.gz")
    mim = nib.load(mp)
    mraw = np.asanyarray(mim.dataobj)
    M = mraw > 0
    A = mim.affine
    Ai = np.linalg.inv(A)
    vsp = np.sqrt((A[:3, :3] ** 2).sum(0))
    cim = nib.load(os.path.join(SRC, "ct.nii.gz"))
    CA, CAi = cim.affine, np.linalg.inv(cim.affine)
    added_all = np.zeros(M.shape, bool)
    figs = []
    for side, (ext, r, cca, v1) in todo.items():
        ex = L(os.path.join(DER, "arterial/%s/va_extract.json" % ext))
        tr = ex["sides"][side]["trace"]
        P = np.asarray(tr["points"], float)
        s = arclen(P)
        redt = np.asarray(tr.get("r_edt_mm") or np.full(len(P), 1.5), float)
        # out-of-mask flags along the trace (no mask voxel within 0.6 voxel diagonals)
        Ip = (Ai[:3, :3] @ P.T + Ai[:3, 3:4]).T
        lo_all = np.maximum(np.floor(Ip.min(0)).astype(int) - 12, 0)
        hi_all = np.minimum(np.ceil(Ip.max(0)).astype(int) + 13, np.array(M.shape))
        srec = out["sides"][side]
        srec["bridges_rescue"] = []
        ok_side = True
        side_add = np.zeros(M.shape, bool)
        for b in v1:
            a0, a1 = float(b["s_mm"]), float(b["s_mm"]) + float(b["length_mm"])
            br = {"s_mm": rr(a0, 1), "length_mm": b["length_mm"], "HU_median_native": b["HU_median_native"],
                  "ratio_to_CCA": rr(b["HU_median_native"] / cca, 3)}
            srec["bridges_rescue"].append(br)
            win = (s >= a0 - 6) & (s <= a1 + 6)
            # local crop around the bridge
            Iw = Ip[win]
            lo = np.maximum(np.floor(Iw.min(0)).astype(int) - int(np.ceil(8 / vsp.min())), 0)
            hi = np.minimum(np.ceil(Iw.max(0)).astype(int) + int(np.ceil(8 / vsp.min())) + 1, np.array(M.shape))
            sub = M[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            dout = ndimage.distance_transform_edt(~sub, sampling=vsp)
            ii = Ip - lo
            dP = map_coordinates(dout, ii.T, order=1, mode="nearest")
            outm = (dP > 0.6 * float(np.linalg.norm(vsp))) & win
            # out-of-mask runs overlapping [a0 - 3, a1 + 3]
            g0, g1 = a0, a1
            runs = []
            i = 0
            while i < len(s):
                if outm[i]:
                    j = i
                    while j + 1 < len(s) and outm[j + 1]:
                        j += 1
                    runs.append((float(s[i]), float(s[j])))
                    i = j + 1
                else:
                    i += 1
            for u0, u1 in runs:
                if u1 >= a0 - 3 and u0 <= a1 + 3:
                    g0, g1 = min(g0, u0), max(g1, u1)
            br["out_of_mask_runs_mm"] = [[rr(u0, 1), rr(u1, 1)] for u0, u1 in runs]
            br["span_mm"] = [rr(g0, 1), rr(g1, 1)]
            near = ((s >= g0 - 10) & (s < g0 - 0.5)) | ((s > g1 + 0.5) & (s <= g1 + 10))
            r_va = float(np.median(redt[near])) if near.sum() >= 3 else float(np.median(redt))
            rc = float(np.clip(1.6 * r_va + 0.5, 2.5, 4.0))
            br.update(r_VA_mm=rr(r_va), capsule_radius_mm=rr(rc))
            sp0, sp1 = max(0.0, g0 - 2.0), min(float(s[-1]), g1 + 2.0)
            seg = (s >= sp0) & (s <= sp1)
            # crop voxel world coordinates
            gi = np.indices(sub.shape).reshape(3, -1).T + lo
            Wv = (A[:3, :3] @ gi.T + A[:3, 3:4]).T
            kd = cKDTree(P[seg])
            dcap, jn = kd.query(Wv, distance_upper_bound=rc + 1e-6)
            cap = np.isfinite(dcap).reshape(sub.shape)
            Ic = (CAi[:3, :3] @ Wv.T + CAi[:3, 3:4])
            hu = np.full(len(Wv), -1024.0, np.float32)
            capf = cap.ravel()
            # read the CT block that covers the capsule only
            Icap = Ic[:, capf]
            clo = np.maximum(np.floor(Icap.min(1)).astype(int) - 2, 0)
            chi = np.minimum(np.ceil(Icap.max(1)).astype(int) + 3, np.array(cim.shape[:3]))
            blk = np.asarray(cim.dataobj[clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]], dtype=np.float32)
            hu[capf] = map_coordinates(blk, Icap - clo[:, None], order=1, mode="nearest")
            hu = hu.reshape(sub.shape)
            T_lo, T_hi = K_LO * cca, K_HI * cca
            grow = cap & (sub | ((hu >= T_lo) & (hu <= T_hi)))
            lab, nl = ndimage.label(grow, structure=np.ones((3, 3, 3)))
            sv = s[seg]
            jn2 = np.where(np.isfinite(dcap), jn, 0).reshape(sub.shape)
            s_vox = np.where(cap, sv[np.clip(jn2, 0, len(sv) - 1)], np.nan)
            if g0 <= 0.6:
                lo_end = cap & sub & (np.linalg.norm((Wv - P[0]).reshape(sub.shape + (3,)), axis=-1) <= rc)
                br["lower_end"] = "mask voxels within r_c of the trace start (subclavian junction)"
            else:
                lo_end = cap & sub & (s_vox >= g0 - 3.0) & (s_vox <= g0 - 0.5)
                br["lower_end"] = "mask voxels 0.5-3 mm below the span"
            hi_end = cap & sub & (s_vox >= g1 + 0.5) & (s_vox <= g1 + 3.0)
            la, lb = set(np.unique(lab[lo_end])) - {0}, set(np.unique(lab[hi_end])) - {0}
            both = sorted(la & lb)
            br.update(T_lo_HU=rr(T_lo, 0), T_hi_HU=rr(T_hi, 0), lower_end_voxels=int(lo_end.sum()),
                      upper_end_voxels=int(hi_end.sum()), connected=bool(both))
            if not both:
                br["verdict"] = "NOT rescued: the enhanced region in the capsule does not connect the two mask ends"
                ok_side = False
                continue
            comp = lab == both[0]
            add = comp & ~sub
            hu_add = hu[add]
            br["added_voxels"] = int(add.sum())
            br["added_volume_mm3"] = rr(add.sum() * float(np.prod(vsp)), 1)
            br["added_HU_median"] = rr(np.median(hu_add), 0) if add.any() else None
            # cross-section along the span (0.5 mm stations, +-0.5 mm slab normal to the trace)
            T = np.gradient(P, axis=0)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
            Wc = Wv.reshape(sub.shape + (3,))[comp]
            req = []
            for k in np.nonzero((s >= g0) & (s <= g1))[0]:
                dv = Wc - P[k]
                ax = dv @ T[k]
                rad = np.linalg.norm(dv - ax[:, None] * T[k], axis=1)
                n = int(np.sum((np.abs(ax) <= 0.5) & (rad <= rc)))
                area = n * float(np.prod(vsp)) / 1.0     # voxels in a 1 mm thick slab -> mm^2
                req.append(float(np.sqrt(area / np.pi)))
            br["cross_section_r_eq_min_mm"] = rr(min(req), 2) if req else None
            br["cross_section_r_eq_median_mm"] = rr(np.median(req), 2) if req else None
            rmin = max(0.5, 0.35 * r_va)
            why = []
            if add.any() and np.median(hu_add) < K_ENH * cca:
                why.append("added voxels HU median %.0f < %.1f x CCA" % (np.median(hu_add), K_ENH))
            if req and min(req) < rmin:
                why.append("cross-section r_eq min %.2f mm < %.2f mm (thread)" % (min(req), rmin))
            if not add.any():
                why.append("no voxel added (the ends already connect inside the capsule?)")
            if why:
                br["verdict"] = "NOT rescued: " + "; ".join(why)
                ok_side = False
                continue
            br["verdict"] = "rescued"
            full = np.zeros(M.shape, bool)
            full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = add
            side_add |= full
            figs.append((side, b, lo, hi, sub, add, hu, P[win], Ai))
        srec["rescued"] = bool(ok_side and side_add.any())
        if srec["rescued"]:
            rescued[side] = ext
            added_all |= side_add
            srec["added_voxels_total"] = int(side_add.sum())
    if rescued:
        nd = os.path.join(DER, "arterial/runs/native_v1r")
        os.makedirs(os.path.join(nd, "extracranial_vessels"), exist_ok=True)
        for f in ("run_report.json", "prep.json", "skull_mask.nii.gz", "labelled_centerlines.json"):
            p = os.path.join(DER, "arterial/runs/native", f)
            if os.path.isfile(p):
                shutil.copyfile(p, os.path.join(nd, f))
        newm = mraw.copy()
        newm[added_all & ~M] = 1
        nim = nib.Nifti1Image(newm, A, mim.header)
        nib.save(nim, os.path.join(nd, "extracranial_vessels/segmentation.nii.gz"))
        out["native_v1r"] = {"dir": nd, "added_voxels": int((added_all & ~M).sum()),
                             "mask_voxels_before": int(M.sum()), "mask_voxels_after": int((newm > 0).sum()),
                             "copied": [f for f in ("run_report.json", "prep.json", "skull_mask.nii.gz", "labelled_centerlines.json")
                                        if os.path.isfile(os.path.join(nd, f))]}
        json.dump(out, open(os.path.join(nd, "v1rescue.json"), "w"), indent=1, default=str)
        # figure: per rescued bridge, 3 MIPs of the CT block (0..1.6 CCA) with the mask (cyan) and the added voxels (red)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(figs), 3, figsize=(12, 4 * len(figs)), squeeze=False)
            for row, (side, b, lo, hi, sub, add, hu, Pw, Ai_) in enumerate(figs):
                cca = out["sides"][side]["cca_HU"]
                for ax_i, ax in enumerate((0, 1, 2)):
                    a = axs[row, ax_i]
                    a.imshow(np.max(np.where(np.isfinite(hu), hu, -1024), axis=ax).T, cmap="gray", vmin=0, vmax=1.6 * cca,
                             origin="lower")
                    a.contour(np.max(sub, axis=ax).T.astype(float), levels=[0.5], colors="c", linewidths=0.7)
                    if add.any():
                        a.contour(np.max(add, axis=ax).T.astype(float), levels=[0.5], colors="r", linewidths=0.9)
                    Iw = (Ai_[:3, :3] @ Pw.T + Ai_[:3, 3:4]).T - lo
                    keep = [k for k in range(3) if k != ax]
                    a.plot(Iw[:, keep[0]], Iw[:, keep[1]], "y-", lw=0.6)
                    a.set_title("%s %s bridge s %.1f (%.1f mm, %s HU): MIP over voxel axis %d" % (
                        CASE, side, b["s_mm"], b["length_mm"], b["HU_median_native"], ax), fontsize=8)
                    a.set_xticks([]); a.set_yticks([])
            fig.suptitle("V1 rescue: CT MIP (capsule only, 0..1.6 CCA), cyan = Arterial mask, red = added voxels, "
                         "yellow = base trace", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(DER, "qc/v1rescue_%s.png" % CASE), dpi=90)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            out["figure_error"] = str(e)
out["rescued_sides"] = rescued
out["seconds"] = round(time.time() - t0, 1)
os.makedirs(os.path.join(DER, "qc"), exist_ok=True)
json.dump(out, open(os.path.join(DER, "qc/v1rescue.json"), "w"), indent=1, default=str)
for side in ("R", "L"):
    sr = out["sides"][side]
    print("%s %s: applies %s (%s) rescued %s %s" % (CASE, side, sr.get("applies"), sr.get("why"), sr.get("rescued"),
          json.dumps([{k: b.get(k) for k in ("s_mm", "length_mm", "span_mm", "capsule_radius_mm", "connected", "added_voxels",
                                            "added_HU_median", "cross_section_r_eq_min_mm", "verdict")}
                      for b in sr.get("bridges_rescue", [])])), flush=True)
print("rescued:", json.dumps(rescued), "%.0f s" % out["seconds"])
