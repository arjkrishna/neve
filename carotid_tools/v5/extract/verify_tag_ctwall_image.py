#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\verify_tag_ctwall_image.py (= /sp/v5_va/verify_tag_ctwall_image.py in the build containers)
"""verify_tag_ctwall_image.py -- adversarial verifier (prefix verify_tag_ctwall): image-domain checks of the
tag_ctwall claim with MY OWN code (no vmr_common / tag_ctwall_* / verify_sources_* import; VMR's CT is read with
sources.load_vmr0248_ct, a plain loader).

    bash /sp/v5_va/verify_tag_ctwall_run.sh neve-build-stagea verify_tag_ctwall_image.py

A. Frame: TaG-Net CTA sampled straight through the NIfTI affine (no flip); the claim's label grid must land on
   integer NIfTI voxels; sides from the GT labels.
B. Edge FWHM (own): stations every 1 mm on each ct_seg4 stage-A VMTK centreline (8 mm off the ends), 24 rays in the
   normal plane, trilinear HU every 0.05 mm to r + 4 mm. Ray kept only if the exterior r+1.5..r+3.5 mm is soft tissue
   (median -100..150 HU), nothing >= 250 HU in r+1..r+4 mm, the profile never exceeds core + 100 HU, contrast >= 150.
   Two estimators: (i) erf fit (4 parameters) on rho <= r + 3 mm, RMS residual <= 10% of contrast; (ii) model-free
   20-80 % rise distance / 1.683 (Gaussian ESF). FWHM = 2.3548 x median sigma. Splits: |d_z| < 0.25 (axial-plane
   rays) and > 0.75 (rays along z), radius 1.0-1.6 mm, stretches. TaG-Net is also re-run with white noise added to
   VMR's lumen noise (noise-matched check).
C. Noise (own): HU std inside the eroded subclavian labels (labels 4/5 of ct_seg4), per 2 mm cube, median.
D. Contamination (own): HU in a 1.5 mm shell outside each VA label (other labels excluded), per 10 mm of the VMTK
   arclength; label voxels with a 26-neighbour >= 700 HU.
E. Landmarks (VMR's enclosure rule, own code): every 0.5 mm, 48 normal-plane rays, a ray is enclosed if it meets
   HU >= 400 within r + 0.3 .. 7 mm; runs >= 0.5. TaG-Net runs are assigned to vertebrae by the TotalSegmentator
   total_v3 1.5 mm masks at the ray hit points; VMR runs by the study's z levels.
Writes carotid_data/va_sources/derived/tagnet/verify_ctwall/verify_tag_ctwall_image.json.
"""
import json
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
TAGROOT = "/repo/carotid_data/va_sources/tagnet/SampleData"
CTA = TAGROOT + "/002/CTA.nii.gz"
GTN = TAGROOT + "/001/labeled_cl_gt.nii.gz"
TSEG = "/repo/carotid_data/va_sources/derived/tagnet/ct_seg/ct_seg4"
VSEG = "/repo/carotid_data/va_sources/derived/vmr0248/ct_seg4"
TS15 = "/repo/carotid_data/va_sources/derived/tagnet/totalseg_v3_1p5/segmentations"
OUT = "/repo/carotid_data/va_sources/derived/tagnet/verify_ctwall/verify_tag_ctwall_image.json"
RNG = np.random.default_rng(20261001)

TAG_LM = {"right": {"ts": {"C6": 35.0, "C2": 118.0, "C1": 169.0, "dura": 186.0},
                    "ct": {"C6": 36.5, "C2": 110.5, "C1": 137.25, "dura": 180.5}},
          "left": {"ts": {"C6": 40.0, "C2": 139.0, "C1": 200.0, "dura": 215.0},
                   "ct": {"C6": 41.25, "C2": 128.0, "C1": 171.75, "dura": 210.0}}}
VMR_LM = {"right": {"study": {"C6": 44.75, "C2": 112.0, "C1": 143.5, "dura": 183.25}},
          "left": {"study": {"C6": 27.5, "C2": 115.0, "C1": 142.5, "dura": 188.75}}}
VMR_LEVELS = [("C2", -180.0, -142.0), ("C3", -197.0, -182.0), ("C4", -212.0, -200.0), ("C5", -228.0, -215.0),
              ("C6", -244.0, -231.0), ("C7", -260.0, -247.0)]


class Vol:
    """HU volume + world(RAS mm) -> fractional index map (index order of `arr`)."""

    def __init__(self, arr, world_to_idx):
        self.arr = arr
        self.w2i = world_to_idx

    def __call__(self, P, order=1):
        P = np.asarray(P, float)
        sh = P.shape[:-1]
        q = self.w2i(P.reshape(-1, 3))
        v = map_coordinates(self.arr, q.T, order=order, mode="constant", cval=-1024.0)
        return v.reshape(sh)


def tag_volume(lo, hi):
    import nibabel as nib
    im = nib.load(CTA)
    A = im.affine
    Ai = np.linalg.inv(A)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = c @ Ai[:3, :3].T + Ai[:3, 3]
    a = np.maximum(np.floor(ijk.min(0)).astype(int) - 2, 0)
    b = np.minimum(np.ceil(ijk.max(0)).astype(int) + 2, np.array(im.shape) - 1)
    blk = np.asarray(im.dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).astype(np.float32)

    def w2i(P):
        return P @ Ai[:3, :3].T + Ai[:3, 3] - a
    return Vol(blk, w2i), A, a


def vmr_volume():
    from sources import load_vmr0248_ct
    ct = load_vmr0248_ct()
    org, sp = np.asarray(ct["origin_mm"], float), np.asarray(ct["spacing_mm"], float)
    arr = ct["array"].astype(np.float32)          # [k, j, i]

    def w2i(P):
        q = (P - org) / sp
        return q[:, ::-1]                            # -> (k, j, i)
    return Vol(arr, w2i), org, sp


def load_cl(path):
    d = json.load(open(path))
    P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.arange(0.0, s[-1] + 1e-9, 0.25)
    return q, np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1), np.interp(q, s, R)


def normal_frame(Q):
    Qs = ndimage.gaussian_filter1d(Q, 4.0, axis=0, mode="nearest")   # sigma 1 mm
    T = np.gradient(Qs, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    U = np.zeros_like(T)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    U[0] = np.cross(T[0], ref)
    U[0] /= np.linalg.norm(U[0])
    for i in range(1, len(T)):
        u = U[i - 1] - (U[i - 1] @ T[i]) * T[i]
        n = np.linalg.norm(u)
        U[i] = u / n if n > 1e-9 else U[i - 1]
    return T, U, np.cross(T, U)


def erf_edge(rho, Ib, Ic, R0, sg):
    return Ib + (Ic - Ib) * 0.5 * erfc((rho - R0) / (np.sqrt(2.0) * sg))


def edge_rays(vol, s, Q, RR):
    T, U, V = normal_frame(Q)
    rho = np.arange(0.0, 6.5, 0.05)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rows = []
    for i in range(32, len(Q) - 32, 4):
        r = RR[i]
        D = np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]          # 24 x 3
        pts = Q[i][None, None, :] + rho[None, :, None] * D[:, None, :]       # 24 x n x 3
        I = vol(pts)
        for k in range(len(th)):
            p = I[k]
            core = np.median(p[rho <= 0.3])
            ext = p[(rho >= r + 1.5) & (rho <= r + 3.5)]
            near = p[(rho >= r + 1.0) & (rho <= r + 4.0)]
            if not ext.size:
                continue
            Ib = np.median(ext)
            if not (-100 < Ib < 150) or near.max() >= 250 or p.max() > core + 100 or core - Ib < 150:
                continue
            # model-free 20-80
            l80, l20 = Ib + 0.8 * (core - Ib), Ib + 0.2 * (core - Ib)
            j80 = np.nonzero(p < l80)[0]
            if not len(j80) or j80[0] == 0:
                continue
            j80 = j80[0]
            j20 = np.nonzero(p[j80:] < l20)[0]
            if not len(j20):
                continue
            j20 = j20[0] + j80

            def cross(j, lev):
                x0, x1, y0, y1 = rho[j - 1], rho[j], p[j - 1], p[j]
                return x0 + (lev - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
            d2080 = cross(j20, l20) - cross(j80, l80)
            m = rho <= r + 3.0
            try:
                par, _ = curve_fit(erf_edge, rho[m], p[m], p0=[Ib, core, r, 0.5],
                                   bounds=([-300, 50, 0.2, 0.05], [400, 2000, 5.0, 2.5]), maxfev=5000)
                res = p[m] - erf_edge(rho[m], *par)
                ok = np.sqrt(np.mean(res ** 2)) <= 0.10 * (par[1] - par[0]) and 0.5 * r <= par[2] <= 2 * r + 0.5
                sg = par[3] if ok else np.nan
            except Exception:  # noqa: BLE001
                sg = np.nan
            rows.append((s[i], sg, d2080 / 1.6832, abs(D[k][2]), r))
    return np.array(rows)


def summarise(rows, lm, L):
    out = {}
    if not len(rows):
        return out
    s, sg, s28, dz, r = rows.T

    def f(m):
        a, b = sg[m & np.isfinite(sg)], s28[m]
        return {"n_fit": int(len(a)), "FWHM_erf": round(2.3548 * float(np.median(a)), 3) if len(a) else None,
                "n_2080": int(len(b)), "FWHM_2080": round(2.3548 * float(np.median(b)), 3) if len(b) else None}
    allm = np.ones(len(s), bool)
    out["all"] = f(allm)
    out["axial_dz<0.25"] = f(dz < 0.25)
    out["along_z_dz>0.75"] = f(dz > 0.75)
    out["r1.0-1.6_dz<0.5"] = f((r >= 1.0) & (r < 1.6) & (dz < 0.5))
    out["r1.0-1.6_dz>0.75"] = f((r >= 1.0) & (r < 1.6) & (dz > 0.75))
    if lm:
        segs = {"V1": (0, lm["C6"]), "V2": (lm["C6"], lm["C2"]), "C2_C1": (lm["C2"], lm["C1"]),
                "C1_dura": (lm["C1"], lm["dura"]), "V4": (lm["dura"], L)}
        out["stretch"] = {k: f((s >= a) & (s < b)) for k, (a, b) in segs.items()}
    return out


def label_grid_hu(npz, vol_idx_fn, arr):
    """HU of arr on the label grid (nearest), with the index residual (frame check)."""
    z = np.load(npz)
    lab = z["labels"]
    org, sp = np.asarray(z["origin_mm"], float), np.asarray(z["spacing_mm"], float)
    K, J, I = lab.shape
    xi = org[0] + sp[0] * np.arange(I)
    yj = org[1] + sp[1] * np.arange(J)
    zk = org[2] + sp[2] * np.arange(K)
    # separable (axis-aligned) frames: index along each output axis
    fx = vol_idx_fn(np.c_[xi, np.full(I, yj[0]), np.full(I, zk[0])])
    fy = vol_idx_fn(np.c_[np.full(J, xi[0]), yj, np.full(J, zk[0])])
    fz = vol_idx_fn(np.c_[np.full(K, xi[0]), np.full(K, yj[0]), zk])
    resid = max(np.abs(f - np.round(f)).max() for f in (fx, fy, fz))
    # axis-aligned frames only: x -> index 0, y -> index 1, z -> index 2 (both CTs are diagonal)
    assert np.ptp(fx[:, 1]) < 1e-6 and np.ptp(fx[:, 2]) < 1e-6 and np.ptp(fz[:, 0]) < 1e-6
    ix, iy, iz = fx[:, 0], fy[:, 1], fz[:, 2]
    return lab, org, sp, (np.round(ix).astype(int), np.round(iy).astype(int), np.round(iz).astype(int)), float(resid)


def contamination(lab, sp, hu, which, s, Q):
    M = lab == which
    other = (lab > 0) & ~M
    dist = ndimage.distance_transform_edt(~M, sampling=(sp[2], sp[1], sp[0]))
    shell = (dist > 0) & (dist <= 1.5) & ~other
    sv = np.argwhere(shell)
    lv = np.argwhere(M)
    org0 = np.zeros(3)
    out = {}
    h = hu[shell]
    out["shell_vox"] = int(shell.sum())
    out["shell_frac_ge700"] = round(float((h >= 700).mean()), 5)
    out["shell_frac_ge1000"] = round(float((h >= 1000).mean()), 5)
    out["shell_n_ge1800"] = int((h >= 1800).sum())
    out["shell_max_HU"] = float(h.max())
    out["shell_p999_HU"] = float(np.percentile(h, 99.9))
    b700 = ndimage.binary_dilation(hu >= 700, structure=np.ones((3, 3, 3), bool))
    out["label_frac_touching700"] = round(float(b700[M].mean()), 5)
    return out, sv, h


def per_s(sv, h, lab_org, sp, s, Q, edges):
    W = lab_org + sv[:, ::-1] * sp                      # [k,j,i] -> world x,y,z
    _, ii = cKDTree(Q).query(W, k=1)
    ss = s[ii]
    rows = []
    for a in edges:
        m = (ss >= a) & (ss < a + 10)
        if m.sum() < 5:
            continue
        rows.append([float(a), int(m.sum()), float(h[m].max()), round(float((h[m] >= 700).mean()), 4),
                     int((h[m] >= 1800).sum())])
    return rows


def enclosure(vol, s, Q, RR):
    T, U, V = normal_frame(Q)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    out = []
    hits_all = []
    for i in range(0, len(Q), 2):
        rho = np.arange(RR[i] + 0.3, 7.0 + 1e-9, 0.1)
        D = np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]
        pts = Q[i][None, None, :] + rho[None, :, None] * D[:, None, :]
        I = vol(pts)
        hit = (I >= 400).any(1)
        first = np.argmax(I >= 400, axis=1)
        hp = pts[np.arange(len(th)), first][hit]
        out.append((s[i], hit.mean(), Q[i][2]))
        hits_all.append(hp)
    return np.array(out), hits_all


def runs_of(prof, thr=0.5, gap=1.0, minlen=1.0):
    s, e, z = prof.T
    on = e >= thr
    runs = []
    i = 0
    while i < len(s):
        if on[i]:
            j = i
            while j + 1 < len(s) and on[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and s[a] - s[merged[-1][1]] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if s[b] - s[a] >= minlen]


def ts_level_lookup():
    import nibabel as nib
    lev = None
    A = None
    for k, nm in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"], start=1):
        im = nib.load("%s/vertebrae_%s.nii.gz" % (TS15, nm))
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
        # tolerate 1 voxel (1.5 mm): nearest nonzero in a 3x3x3 neighbourhood
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


def main():
    t0 = time.time()
    res = {"what": __doc__}
    # ---------------- TaG volume (crop around both VA centrelines)
    cls = {}
    for side, v in (("right", "rva"), ("left", "lva")):
        cls[("TaG", side)] = load_cl("%s/%s_stageA_vmtk.json" % (TSEG, v))
        cls[("VMR", side)] = load_cl("%s/%s_stageA_vmtk.json" % (VSEG, v))
    allp = np.vstack([cls[("TaG", sd)][1] for sd in ("right", "left")])
    zl = np.load(TSEG + "/labels_kji.npz")
    lo_l = np.asarray(zl["origin_mm"], float)
    hi_l = lo_l + np.asarray(zl["spacing_mm"], float) * (np.array(zl["labels"].shape[::-1]) - 1)
    lo = np.minimum(allp.min(0) - 12, lo_l)
    hi = np.maximum(allp.max(0) + 12, hi_l)
    tvol, A, a0 = tag_volume(lo, hi)
    print("TaG block", tvol.arr.shape, "%.0f s" % (time.time() - t0), flush=True)
    vvol, vorg, vsp = vmr_volume()
    print("VMR volume", vvol.arr.shape, "%.0f s" % (time.time() - t0), flush=True)

    # ---------------- A. frame: GT label sides and HU on the GT path
    import nibabel as nib
    gt = nib.load(GTN)
    G = np.asanyarray(gt.dataobj)
    side_chk = {}
    for lab, nm in ((8, "R-VA"), (13, "L-VA"), (14, "R-SCA"), (15, "L-SCA"), (3, "BCT (txt 2)"), (2, "arch (txt 1)")):
        idx = np.argwhere(G == lab)
        W = idx @ gt.affine[:3, :3].T + gt.affine[:3, 3]
        side_chk[nm] = {"n": int(len(idx)), "mean_world_RAS": W.mean(0).round(2).tolist(),
                        "HU_median_at_voxels": float(np.median(tvol(W, order=0)))}
    res["frame"] = {"cta_affine": A.tolist(), "gt_labels": side_chk,
                    "note": "RAS world (nibabel); R-VA at +x, BCT at +x of the arch: patient right = +x"}

    # ---------------- B. edge FWHM
    res["edge"] = {}
    rows_keep = {}
    for (src, side), (s, Q, RR) in cls.items():
        vol = tvol if src == "TaG" else vvol
        rows = edge_rays(vol, s, Q, RR)
        rows_keep[(src, side)] = rows
        lm = (TAG_LM[side]["ts"] if src == "TaG" else VMR_LM[side]["study"])
        res["edge"]["%s_%s" % (src, side)] = summarise(rows, lm, float(s[-1]))
        e = res["edge"]["%s_%s" % (src, side)]
        print("edge %s %s all %s | axial %s | z %s | like %s (%.0f s)" % (
            src, side, e["all"], e["axial_dz<0.25"], e["along_z_dz>0.75"], e["r1.0-1.6_dz<0.5"], time.time() - t0),
            flush=True)

    # ---------------- C. noise inside eroded subclavian labels
    def idx_fn_tag(P):
        return P @ np.linalg.inv(A)[:3, :3].T + np.linalg.inv(A)[:3, 3]

    def idx_fn_vmr(P):
        return (P - vorg) / vsp

    noise = {}
    hu_grids = {}
    for src, npz, fn in (("TaG", TSEG + "/labels_kji.npz", idx_fn_tag), ("VMR", VSEG + "/labels_kji.npz", idx_fn_vmr)):
        lab, lorg, lsp, (ix, iy, iz), resid = label_grid_hu(npz, fn, None)
        if src == "TaG":
            # NIfTI index (i, j, k) of the block: arr[i - a0, j - a0, k - a0]
            hu = tvol.arr[np.ix_(ix - a0[0], iy - a0[1], iz - a0[2])].transpose(2, 1, 0)   # -> [k, j, i]
        else:
            hu = vvol.arr[np.ix_(iz, iy, ix)]                                              # [k, j, i]
        hu_grids[src] = (lab, lorg, lsp, hu)
        out = {"label_grid_index_residual": resid}
        for which, nm in ((4, "rsub"), (5, "lsub")):
            M = ndimage.binary_erosion(lab == which, iterations=3)
            vals = []
            cz = np.argwhere(M)
            if len(cz) < 50:
                continue
            cell = np.floor(cz * lsp[::-1] / 2.0).astype(int)
            keys, inv = np.unique(cell, axis=0, return_inverse=True)
            h = hu[M]
            for k in range(len(keys)):
                hh = h[inv == k]
                if len(hh) >= 20:
                    vals.append(hh.std())
            out[nm] = {"n_cells": len(vals), "std_median_HU": round(float(np.median(vals)), 1),
                       "mean_HU": round(float(h.mean()), 1)}
        noise[src] = out
        print("noise", src, out, flush=True)
    res["noise"] = noise

    # ---------------- B2. noise-matched TaG edge
    nT = np.mean([noise["TaG"][k]["std_median_HU"] for k in ("rsub", "lsub") if k in noise["TaG"]])
    nV = np.mean([noise["VMR"][k]["std_median_HU"] for k in ("rsub", "lsub") if k in noise["VMR"]])
    add = float(np.sqrt(max(nV ** 2 - nT ** 2, 0.0)))
    noisy = Vol(tvol.arr + RNG.normal(0.0, add, tvol.arr.shape).astype(np.float32), tvol.w2i)
    res["edge_noise_matched"] = {"added_white_noise_HU": round(add, 1)}
    for side in ("right", "left"):
        s, Q, RR = cls[("TaG", side)]
        rows = edge_rays(noisy, s, Q, RR)
        res["edge_noise_matched"]["TaG_%s" % side] = summarise(rows, TAG_LM[side]["ts"], float(s[-1]))
        e = res["edge_noise_matched"]["TaG_%s" % side]
        print("edge noisy TaG %s all %s | axial %s | z %s" % (side, e["all"], e["axial_dz<0.25"], e["along_z_dz>0.75"]),
              flush=True)
    del noisy

    # ---------------- D. contamination
    res["contamination"] = {}
    for src in ("TaG", "VMR"):
        lab, lorg, lsp, hu = hu_grids[src]
        for which, side in ((1, "right"), (2, "left")):
            s, Q, RR = cls[(src, side)]
            out, sv, h = contamination(lab, lsp, hu, which, s, Q)
            out["per10mm_[s,n,max,frac700,n1800]"] = per_s(sv, h, lorg, lsp, s, Q, np.arange(0, s[-1], 10.0))
            res["contamination"]["%s_%s" % (src, side)] = out
            print("contam %s %s %s" % (src, side, {k: v for k, v in out.items() if not k.startswith("per")}), flush=True)

    # ---------------- E. enclosure landmarks
    look = ts_level_lookup()
    res["enclosure"] = {}
    for (src, side), (s, Q, RR) in cls.items():
        vol = tvol if src == "TaG" else vvol
        prof, hits = enclosure(vol, s, Q, RR)
        runs = []
        for a, b in runs_of(prof):
            if src == "TaG":
                lv = [x for h in hits[a:b + 1] for x in look(h)]
                lv = [x for x in lv if x != "none"]
                level = max(set(lv), key=lv.count) if lv else "none"
                share = round(lv.count(level) / max(len(lv), 1), 2) if lv else 0
            else:
                zc = float(np.median(prof[a:b + 1, 2]))
                level = next((nm for nm, z0, z1 in VMR_LEVELS if z0 <= zc <= z1), "other")
                share = None
            runs.append({"s": [float(prof[a, 0]), float(prof[b, 0])], "z": [round(float(prof[a, 2]), 1),
                         round(float(prof[b, 2]), 1)], "mean_enc": round(float(prof[a:b + 1, 1].mean()), 2),
                         "level": level, "level_share": share})
        res["enclosure"]["%s_%s" % (src, side)] = runs
        print("enclosure %s %s" % (src, side), flush=True)
        for r_ in runs:
            print("    s %6.1f-%6.1f z %7.1f..%7.1f enc %.2f %s %s" % (r_["s"][0], r_["s"][1], r_["z"][0], r_["z"][1],
                  r_["mean_enc"], r_["level"], r_["level_share"]), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
