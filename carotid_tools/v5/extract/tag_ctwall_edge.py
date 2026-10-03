#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_ctwall_edge.py (= /sp/v5_va/tag_ctwall_edge.py in the build containers)
"""tag_ctwall_edge.py -- what the TaG-Net A CTA can resolve at the VA lumen edge, and what touches the VA,
side by side with VMR 0248 (same code on both CTs).

    bash /sp/v5_va/tag_ctwall_run.sh neve-build-stagea tag_ctwall_edge.py

1. Edge FWHM, two methods (both copied verbatim from the study, per-ray records kept so they can be split):
   a. the vmr agent's (vmr_compare.psf_fit): erf fit r_e +- 2 mm on the segmentation's own rays with a soft-tissue
      exterior (kind 0), every 8th station (2 mm), 8 of 64 rays -> "default"; the same fit on every 2nd station and
      every 2nd ray -> "dense" (for the splits);
   b. the verifier's (verify_sources_vmr1.py edge block): VMTK centerline + MISR, every 1 mm from 8 mm, 24 rays,
      0..5 mm, kept only with a soft-tissue exterior, no brighter neighbour and an RMS residual <= 12% of contrast.
   Splits: by stretch (TS landmarks for TaG-Net, the study's landmarks for VMR), by ray inclination |d_z| (0 =
   axial-plane ray; VMR's slices are 0.7 mm, TaG-Net's 0.4 mm) and by station radius.
   Calibration: the same erf fit on noiseless profiles of a disc of radius r blurred by an isotropic Gaussian of
   sigma_true (2-D, along a radius), to show the fit's bias at VA radii.
2. Image-domain relief (verify_sources_vmr2.py part b, copied): per station every 0.5 mm on the VMTK centerline,
   48 rays, the local half-max edge radius; share of rays > 0.3 mm (and > 0.15 r) off the station median.
3. Noise: std of HU over the voxels (nearest) inside 0.5 r of the VMTK centerline (+-1 mm along it), stations with
   r >= 1.2 mm, every 2 mm; CNR = (I_core - I_bg) / noise.
4. Contacts (as VMR's left V1 vein): on the ct_seg4 rays, the bright-neighbour rays (kind 3, upper threshold) and
   the dip-before-bright rays (kind 1); for each, the neighbour's peak (position, HU) re-sampled along the ray and
   classified by the TotalSegmentator total_v3 masks on the CTA grid (bone: vertebrae, skull, clavicles, first
   ribs, sternum, <= 1 mm; vein: brachiocephalic veins, SVC; artery: TS arteries or another vessel of our own
   label) -> else "unlabelled (likely venous plexus)". Plus the voxels the 0.8 mm-rule opening removed
   (ct_seg3 - ct_seg4), the label voxels touching >= 700 HU, and protrusion rays (kind 0, r_e > station median
   + 1 mm: a lumen-bright continuation = branch stub or touching vessel). All per 10 mm on the VMTK arclength.
Writes results/tag_ctwall_edge.json.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
from vmr_common import CT, rmf as rmf_vmr  # noqa: E402
from tag_ctwall_common import RES, SEG_ROOT, TSV3, jdump, load_tag  # noqa: E402

VMR_DER = "/repo/carotid_data/va_sources/derived/vmr0248"
NO_UPPER = 1.0e4
BRIGHT_FACTOR, BRIGHT_PLUS = 1.3, 200.0


# --------------------------------------------------------------------------- 1a vmr agent's erf fit (copied)
def edge_model(r, A, B, r0, sig):
    return B + (A - B) * 0.5 * erfc((r - r0) / (np.sqrt(2) * sig))


def psf_rays(ct, d, every, rays):
    """vmr_compare.psf_fit's loop, returning one record per accepted ray."""
    P, re, kind = np.asarray(d["P"]), np.asarray(d["r_e"]), np.asarray(d["kind"])
    _, U, W = rmf_vmr(P)
    out = []
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
                    out.append((i, a, popt[3], popt[2], abs(dvec[2]), popt[0] - popt[1]))
            except Exception:  # noqa: BLE001
                pass
    return np.array(out) if out else np.zeros((0, 6))


# --------------------------------------------------------------------------- 1b verifier's erf fit (copied)
def rmf_ver(Q):
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    a = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    U = np.cross(T[0], a)
    U /= np.linalg.norm(U)
    Us = [U]
    for i in range(1, len(Q)):
        U = Us[-1] - (Us[-1] @ T[i]) * T[i]
        U /= np.linalg.norm(U)
        Us.append(U)
    Us = np.array(Us)
    return T, Us, np.cross(T, Us)


def ver_model(rho, Ib, Ic, R0, sg):
    return Ib + (Ic - Ib) * 0.5 * erfc((rho - R0) / (np.sqrt(2) * sg))


def resample_cl(path):
    d = json.load(open(path))
    P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.arange(0, s[-1] + 1e-9, 0.25)
    Q = np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)
    return Q, q, np.interp(q, s, R)


def verifier_rays(hu, Q, s, RR):
    T, U, V = rmf_ver(ndimage.gaussian_filter1d(Q, 4, axis=0, mode="nearest"))
    out = []
    rho = np.arange(0, 5.0, 0.05)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for i in range(int(8 / 0.25), len(Q) - int(8 / 0.25), 4):   # every 1 mm
        for a in th:
            d = np.cos(a) * U[i] + np.sin(a) * V[i]
            pts = Q[i] + rho[:, None] * d
            I = hu(pts)
            ext = I[(rho > RR[i] + 1.8) & (rho < RR[i] + 3.2)]
            core = I[rho < 0.4]
            if not (ext.size and -150 < ext.mean() < 160 and I.max() < core.mean() + 120 and ext.max() < 220):
                continue
            m = rho < RR[i] + 3.0
            try:
                p, _ = curve_fit(ver_model, rho[m], I[m], p0=[ext.mean(), core.mean(), RR[i], 0.5],
                                 bounds=([-300, 100, 0.3, 0.05], [300, 1500, 4.0, 2.0]), maxfev=4000)
            except Exception:  # noqa: BLE001
                continue
            resid = I[m] - ver_model(rho[m], *p)
            if np.sqrt(np.mean(resid ** 2)) > 0.12 * (p[1] - p[0]):
                continue
            out.append((s[i], p[3], p[2] - RR[i], abs(d[2]), RR[i], p[1] - p[0]))
    return np.array(out) if out else np.zeros((0, 6))


# --------------------------------------------------------------------------- 2 image-domain relief (copied)
def image_relief(hu, Q, s, R):
    T, U, V = rmf_ver(ndimage.gaussian_filter1d(Q, 4, axis=0, mode="nearest"))
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rho = np.arange(0, 5.0, 0.05)
    rows = []
    for i in range(int(3 / 0.25), len(Q) - int(3 / 0.25), 2):
        Ic = float(np.median(hu(Q[i] + 0.3 * (np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]))))
        dirs = np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]
        P = Q[i] + rho[None, :, None] * dirs[:, None, :]
        I = hu(P.reshape(-1, 3)).reshape(len(th), len(rho))
        er = []
        for a in range(len(th)):
            prof = I[a]
            out = prof[(rho > R[i] + 1.5) & (rho < R[i] + 2.5)]
            bg = float(np.min(out)) if out.size else 0.0
            half = 0.5 * (Ic + bg)
            b = np.nonzero(prof < half)[0]
            if not len(b) or b[0] == 0:
                er.append(np.nan)
                continue
            k = b[0]
            er.append(rho[k - 1] + (prof[k - 1] - half) / (prof[k - 1] - prof[k] + 1e-9) * 0.05)
        er = np.array(er)
        ok = np.isfinite(er)
        if ok.sum() < 30:
            continue
        med = np.median(er[ok])
        dev = np.abs(er[ok] - med)
        rows.append([s[i], float((dev > 0.3).mean()), float((dev > 0.15 * med).mean()), med])
    return np.array(rows)


# --------------------------------------------------------------------------- 3 noise
def lumen_noise(ct, Q, s, R):
    _, U, V = rmf_ver(ndimage.gaussian_filter1d(Q, 4, axis=0, mode="nearest"))
    T = np.gradient(ndimage.gaussian_filter1d(Q, 4, axis=0, mode="nearest"), axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    out = []
    g = np.linspace(-1, 1, 9)
    for i in range(int(4 / 0.25), len(Q) - int(4 / 0.25), 8):
        if R[i] < 1.2:
            continue
        rr = np.linspace(0, 0.5 * R[i], 6)
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = (Q[i][None, None, None, :] + g[:, None, None, None] * T[i] +
               (rr[None, :, None, None] * (np.cos(th)[None, None, :, None] * U[i] + np.sin(th)[None, None, :, None] * V[i])))
        q = np.round((pts.reshape(-1, 3) - ct.org) / ct.sp).astype(int)
        q = np.unique(q, axis=0)
        v = ct.arr[q[:, 2], q[:, 1], q[:, 0]].astype(float)
        if len(v) >= 10:
            out.append((s[i], float(np.std(v)), float(np.median(v)), len(v)))
    return np.array(out)


# --------------------------------------------------------------------------- calibration of the erf fit
def erf_bias_table():
    rows = []
    x = np.arange(-8, 8.0001, 0.02)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for r in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        disc = (X ** 2 + Y ** 2 <= r * r).astype(float)
        for sg in (0.4, 0.5, 0.6, 0.7):
            b = ndimage.gaussian_filter(disc, sg / 0.02)
            prof = b[x.size // 2:, x.size // 2]
            rr = x[x.size // 2:]
            # the fit as in psf_fit: window r0 +- 2 mm around the half-max crossing, A=lumen, B=bg
            half = 0.5 * prof.max()
            k = int(np.nonzero(prof < half)[0][0])
            re = rr[k]
            m = (rr >= max(0, re - 2)) & (rr < re + 2)
            try:
                popt, _ = curve_fit(edge_model, rr[m], prof[m], p0=[prof[m][0], 0, re, 0.35],
                                    bounds=([0, -1, rr[m][0], 0.05], [2, 1, rr[m][-1], 2.0]), maxfev=4000)
                rows.append({"r_mm": r, "sigma_true": sg, "sigma_fit": round(float(popt[3]), 3),
                             "fit_over_true": round(float(popt[3] / sg), 3), "peak_frac": round(float(prof.max()), 3),
                             "halfmax_edge_minus_r": round(float(re - r), 3)})
            except Exception:  # noqa: BLE001
                pass
    return rows


# --------------------------------------------------------------------------- helpers
def stretch_of(sv, lm, L):
    segs = {"V1": (0.0, lm["C6"]), "V2": (lm["C6"], lm["C2"]), "V3_C2_to_C1": (lm["C2"], lm["C1"]),
            "V3_C1_to_dura": (lm["C1"], lm["dura"]), "V4": (lm["dura"], L), "C6_to_C1": (lm["C6"], lm["C1"]),
            "whole": (0.0, L), "distal_0_40": (L - 40, L)}
    return {k: (sv >= a) & (sv < b) for k, (a, b) in segs.items()}


def fwhm(sig):
    return None if len(sig) == 0 else round(float(2.3548 * np.median(sig)), 3)


def summ_sig(sig):
    if len(sig) == 0:
        return {"n": 0}
    return {"n": int(len(sig)), "sigma_median_mm": round(float(np.median(sig)), 3),
            "sigma_p25_p75_mm": np.percentile(sig, [25, 75]).round(3).tolist(), "FWHM_mm": fwhm(sig)}


def splits(sv, sig, dz, rad, lm, L):
    out = {"all": summ_sig(sig)}
    out["by_stretch"] = {k: summ_sig(sig[m]) for k, m in stretch_of(sv, lm, L).items()}
    out["by_ray_inclination_absdz"] = {}
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = (dz >= lo) & (dz < hi)
        out["by_ray_inclination_absdz"]["%.2f-%.2f" % (lo, hi)] = summ_sig(sig[m])
    out["by_radius_mm"] = {}
    for lo, hi in ((0.0, 1.0), (1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 9.0)):
        m = (rad >= lo) & (rad < hi)
        out["by_radius_mm"]["%.1f-%.1f" % (lo, hi)] = summ_sig(sig[m])
    # same radius band (1.0-1.6 mm) and axial-ish rays only: the like-for-like number
    m = (rad >= 1.0) & (rad < 1.6) & (dz < 0.5)
    out["r1.0-1.6_and_absdz<0.5"] = summ_sig(sig[m])
    return out


def map_s(Pst, Q, s):
    _, j = cKDTree(Q).query(Pst)
    return s[j]


# --------------------------------------------------------------------------- TS masks (bone / vein / artery)
def ts_classifier(lo, hi):
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
    for g, names in groups.items():
        acc = np.zeros(tuple(b - a + 1), bool)
        for nm in names:
            p = os.path.join(seg, nm + ".nii.gz")
            if os.path.exists(p):
                acc |= np.asarray(nib.load(p).dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]) > 0
        dist[g] = ndimage.distance_transform_edt(~acc, sampling=sp) if acc.any() else None

    def classify(W):
        I = np.round(W @ Ai[:3, :3].T + Ai[:3, 3]).astype(int) - a
        I = np.clip(I, 0, np.array(acc.shape) - 1)
        out = {}
        for g, dd in dist.items():
            out[g] = np.full(len(W), np.inf) if dd is None else dd[I[:, 0], I[:, 1], I[:, 2]]
        return out
    return classify


def contacts(ct, st, nm, sv, labels, classify, own_k):
    """Re-sample the kind-1 / kind-3 rays of one vessel; locate the brighter neighbour's peak and classify it."""
    d = st[nm]
    P, R = np.asarray(d["P"]), np.asarray(d["R"])
    Ic = np.asarray(d["Ic"])
    kind = np.asarray(d["kind"])
    _, U, W = rmf_vmr(P)
    NR = kind.shape[1]
    TH = np.linspace(0, 2 * np.pi, NR, endpoint=False)
    DR = 0.05
    rows = []
    zl, lorg, lsp = labels
    for i, a in zip(*np.nonzero((kind == 1) | (kind == 3))):
        dv = np.cos(TH[a]) * U[i] + np.sin(TH[a]) * W[i]
        r = np.arange(0, R[i] + 3.0 + 1e-9, DR)
        q = ct.sample(P[i] + r[:, None] * dv)
        i0 = int(np.ceil(0.3 * R[i] / DR))
        ilim = min(len(r) - 1, int((R[i] + 3.0) / DR))
        thr_b = max(BRIGHT_FACTOR * Ic[i], Ic[i] + BRIGHT_PLUS)
        if kind[i, a] == 3:
            hb = np.nonzero(q[i0:ilim] > thr_b)[0]
            if not len(hb):
                continue
            j0 = i0 + hb[0]
        else:
            # dip: first local minimum < 0.8 Ic followed by a >= 40 HU rise within 1 mm (as cast())
            seg = q[i0:ilim]
            mins = np.nonzero((seg[1:-1] <= seg[:-2]) & (seg[1:-1] <= seg[2:]) & (seg[1:-1] < 0.8 * Ic[i]))[0] + 1
            j0 = None
            for jm in mins:
                if q[i0 + jm:i0 + jm + int(1.0 / DR)].max() - seg[jm] >= 40:
                    j0 = i0 + jm
                    break
            if j0 is None:
                continue
        j1 = min(len(q), j0 + int(1.0 / DR) + 1)
        jp = j0 + int(np.argmax(q[j0:j1]))
        Xp = P[i] + r[jp] * dv
        li = np.round((Xp - lorg) / lsp).astype(int)
        lab_here = int(zl[li[2], li[1], li[0]]) if np.all((li >= 0) & (li < np.array(zl.shape[::-1]))) else 0
        rows.append((sv[i], int(kind[i, a]), float(q[jp]), float(r[jp]), lab_here, Xp[0], Xp[1], Xp[2], float(Ic[i])))
    rows = np.array(rows) if rows else np.zeros((0, 9))
    cls = []
    if len(rows):
        c = classify(rows[:, 5:8]) if classify is not None else None
        for k in range(len(rows)):
            if rows[k, 4] > 0 and rows[k, 4] != own_k:
                cls.append("artery (own label: other vessel)")
            elif c is not None and c["bone"][k] <= 1.0:
                cls.append("bone (TS mask <= 1 mm)")
            elif c is not None and c["artery_ts"][k] <= 1.0:
                cls.append("artery (TS mask)")
            elif c is not None and c["vein"][k] <= 1.0:
                cls.append("vein (TS mask)")
            else:
                cls.append("unlabelled" if c is not None else "no masks (peak HU only)")
    return rows, cls


# --------------------------------------------------------------------------- main
def dataset(name):
    """returns ct (CT), st (stations_rays), cl paths, landmarks per side, labels (seg4, seg3)"""
    if name == "TaG-A":
        D4, D3 = os.path.join(SEG_ROOT, "ct_seg4"), os.path.join(SEG_ROOT, "ct_seg3")
        st = json.load(open(os.path.join(D4, "stations_rays.json")))
        allp = np.vstack([np.asarray(st[k]["P"]) for k in st])
        from tag_ctwall_common import load_ct_crop
        ctd = load_ct_crop(allp.min(0), allp.max(0), margin_mm=12.0, check=False)
        ct = CT(ctd)
        LMJ = json.load(open(os.path.join(RES, "tag_ctwall_landmarks.json")))
        lms = {sd: LMJ["tags"]["ct_seg4"]["stageA"][sd]["total_v3_1p5"]["landmarks_used"] for sd in ("right", "left")}
        lms_ct = {sd: LMJ["tags"]["ct_seg4"]["stageA"][sd]["ct_method"] for sd in ("right", "left")}
    else:
        D4, D3 = os.path.join(VMR_DER, "ct_seg4"), os.path.join(VMR_DER, "ct_seg3")
        st = json.load(open(os.path.join(D4, "stations_rays.json")))
        from vmr_common import load_vmr
        ct = CT(load_vmr(with_ct=True)["ct"])
        m = json.load(open(os.path.join(RES, "vmr0248_metrics.json")))["full_results"]["ct_seg4"]["ct_stageA"]
        lms = {}
        for sd in ("right", "left"):
            rg = m[sd]["segments_wall"]["ranges_mm"]
            lms[sd] = {"C6": rg["V2"][0], "C2": rg["V2"][1], "C1": rg["V3_C1_to_dura"][0], "dura": rg["V4"][0]}
        lms_ct = lms
    z4 = np.load(os.path.join(D4, "labels_kji.npz"))
    z3 = np.load(os.path.join(D3, "labels_kji.npz"))
    return ct, st, D4, D3, lms, lms_ct, z4, z3


def bins10(L):
    return np.arange(0.0, L, 10.0)


def main():
    t0 = time.time()
    res = {"what": __doc__, "erf_fit_bias_on_blurred_discs": erf_bias_table()}
    print("erf bias", json.dumps(res["erf_fit_bias_on_blurred_discs"][:6]), flush=True)
    for name in ("TaG-A", "VMR"):
        ct, st, D4, D3, lms, lms_ct, z4, z3 = dataset(name)
        hu = lambda X: ct.sample(np.asarray(X, float))  # noqa: E731
        out = {"ct_spacing_mm": ct.sp.tolist(), "landmarks_used": lms, "sides": {}}
        classify = None
        if name == "TaG-A":
            allp = np.vstack([np.asarray(st[k]["P"]) for k in ("rva", "lva")])
            classify = ts_classifier(allp.min(0) - 10, allp.max(0) + 10)
        assert z4["labels"].shape == z3["labels"].shape and np.allclose(z4["origin_mm"], z3["origin_mm"])
        for side, v, k in (("right", "rva", 1), ("left", "lva", 2)):
            Q, s, RR = resample_cl(os.path.join(D4, "%s_stageA_vmtk.json" % v))
            L = float(s[-1])
            lm = lms[side]
            rec = {"length_mm": round(L, 2)}
            # 1a vmr agent's fit
            d = st[v]
            Pst = np.asarray(d["P"])
            sv = map_s(Pst, Q, s)
            req = None
            # station radius for the split: equal-area radius of the ray polygon (as vmr_ct_seg3 recentre r_eq)
            re = np.asarray(d["r_e"])
            req = np.sqrt(np.mean(re ** 2, axis=1))
            dflt = psf_rays(ct, d, 8, (0, 8, 16, 24, 32, 40, 48, 56))
            dens = psf_rays(ct, d, 2, tuple(range(0, 64, 2)))
            rec["edge_vmr_method_default"] = summ_sig(dflt[:, 2]) if len(dflt) else {"n": 0}
            ii = dens[:, 0].astype(int)
            rec["edge_vmr_method_dense"] = splits(sv[ii], dens[:, 2], dens[:, 4], req[ii], lm, L)
            # 1b verifier's fit
            vr = verifier_rays(hu, Q, s, RR)
            rec["edge_verifier_method"] = splits(vr[:, 0], vr[:, 1], vr[:, 3], vr[:, 4], lm, L)
            rec["edge_verifier_method"]["edge_R_minus_misr_median_mm"] = round(float(np.median(vr[:, 2])), 3)
            print(name, side, "FWHM vmr-default", rec["edge_vmr_method_default"].get("FWHM_mm"),
                  "verifier", rec["edge_verifier_method"]["all"].get("FWHM_mm"),
                  "dense", rec["edge_vmr_method_dense"]["all"].get("FWHM_mm"), flush=True)
            # 2 image relief
            D = image_relief(hu, Q, s, RR)
            sm = stretch_of(D[:, 0], lm, L)
            rec["image_edge_dev"] = {kk: {"frac_rays_dev_gt_0.3mm": round(float(D[m, 1].mean()), 4) if m.any() else None,
                                          "frac_rays_dev_gt_0.15r": round(float(D[m, 2].mean()), 4) if m.any() else None,
                                          "edge_r_median": round(float(np.median(D[m, 3])), 3) if m.any() else None}
                                     for kk, m in sm.items()}
            # 3 noise
            N = lumen_noise(ct, Q, s, RR)
            Ic, Ib = np.median(np.asarray(d["Ic"])), np.median(np.asarray(d["Ib"]))
            rec["noise"] = {"lumen_voxel_std_HU_median": round(float(np.median(N[:, 1])), 1) if len(N) else None,
                            "lumen_voxel_std_HU_p25_p75": np.percentile(N[:, 1], [25, 75]).round(1).tolist() if len(N) else None,
                            "n_stations": int(len(N)), "I_core_median": round(float(Ic), 1), "I_bg_median": round(float(Ib), 1),
                            "CNR": round(float((Ic - Ib) / np.median(N[:, 1])), 1) if len(N) else None}
            # 4 contacts
            rows, cls = contacts(ct, st, v, sv, (z4["labels"], z4["origin_mm"], z4["spacing_mm"]), classify, k)
            kind = np.asarray(d["kind"])
            nst = kind.shape[0]
            # opened voxels and bone-touch, per voxel -> s
            L4, L3 = z4["labels"] == k, z3["labels"] == k
            removed = np.argwhere(L3 & ~L4)
            lorg, lsp = z4["origin_mm"], z4["spacing_mm"]
            s_removed = map_s(lorg + removed[:, ::-1] * lsp, Q, s) if len(removed) else np.zeros(0)
            # protrusion rays (kind 0, r_e > station median + 1 mm)
            med = np.median(re, axis=1)
            prot = (kind == 0) & (re > med[:, None] + 1.0)
            B = bins10(L)
            brow = []
            for a0 in B:
                a1 = min(a0 + 10.0, L)
                ms = (sv >= a0) & (sv < a1)
                mr = (rows[:, 0] >= a0) & (rows[:, 0] < a1) if len(rows) else np.zeros(0, bool)
                cl_here = [cls[j] for j in np.nonzero(mr)[0]]
                nr = max(int(ms.sum()) * kind.shape[1], 1)
                brow.append({"from_mm": round(float(a0), 1), "to_mm": round(float(a1), 1),
                             "frac_rays_upper_threshold_kind3": round(float((kind[ms] == 3).sum() / nr), 4),
                             "frac_rays_dip_before_bright_kind1": round(float((kind[ms] == 1).sum() / nr), 4),
                             "frac_rays_protrusion": round(float(prot[ms].sum() / nr), 4),
                             "opened_voxels_removed": int(((s_removed >= a0) & (s_removed < a1)).sum()),
                             "contact_classes": {c: cl_here.count(c) for c in sorted(set(cl_here))},
                             "contact_peak_HU_median": round(float(np.median(rows[mr, 2])), 0) if mr.any() else None,
                             "contact_peak_HU_max": round(float(np.max(rows[mr, 2])), 0) if mr.any() else None,
                             "image_edge_dev_0.3": round(float(D[(D[:, 0] >= a0) & (D[:, 0] < a1), 1].mean()), 4)
                             if ((D[:, 0] >= a0) & (D[:, 0] < a1)).any() else None})
            rec["per_10mm"] = brow
            rec["contacts_total"] = {"n_rays": int(len(rows)), "by_class": {c: cls.count(c) for c in sorted(set(cls))},
                                     "frac_rays_kind3_whole": round(float((kind == 3).mean()), 4),
                                     "frac_rays_kind1_whole": round(float((kind == 1).mean()), 4),
                                     "opened_voxels_removed_total": int(len(removed)),
                                     "peak_HU_by_class": {c: [round(float(np.percentile(rows[[j for j in range(len(rows)) if cls[j] == c], 2], q)), 0)
                                                              for q in (10, 50, 90)] for c in sorted(set(cls))}}
            rec["contact_rays"] = [{"s": round(float(r_[0]), 2), "kind": int(r_[1]), "peak_HU": round(float(r_[2]), 0),
                                    "peak_r_mm": round(float(r_[3]), 2), "class": c} for r_, c in zip(rows, cls)][:4000]
            out["sides"][side] = rec
            print(name, side, "contacts", json.dumps(rec["contacts_total"]), "noise", json.dumps(rec["noise"]), flush=True)
        res[name] = out
        del ct
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, os.path.join(RES, "tag_ctwall_edge.json"))
    print("done %.0f s" % res["seconds"])


if __name__ == "__main__":
    main()
