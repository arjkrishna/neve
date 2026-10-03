#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ct_seg.py (= /sp/v5_va/scaleup_ct_seg.py in the build containers)
# scaleup_ct_seg.py (scaleup_fixes, 2026-10-02): ct_seg4 of PIPELINE v2, ONE script for both sources. = pilot_proc_song_ct_seg.py
# (byte-identical algorithm to pilot_proc_ia172_ct_seg.py; the two pilot copies differ only in their adapter import) with:
#   v2-a  adapter by source: env PILOT_SOURCE = ia172 | song -> scaleup_ctseg_common_<source>.py (the pilot adapters + the
#         per-side SELECTED trace set, qc/<SCALEUP_SELECT>, so a V1-rescue trace seeds the wall). PILOT_SOURCE is required.
#   v2-b  the C2-EXIT PINCH FIX (block "v2 pinch"; env SCALEUP_PINCH_FIX=0 disables it = the pilot method exactly): after
#         every ray cast (pre-pass, pass 1, pass 2) a station is FLAGGED when its centre is not lumen -- centre HU (trilinear)
#         < the station's median local threshold T_loc, OR >= 50 % of its rays end at the 0.3 R start floor (the first
#         sample is already outside [T_lo, T_hi]; 094 R: centre 344-396 HU vs T_loc 341-350 at the lumen edge, r_e 0.30-0.45).
#         A flagged station is moved, in its own normal plane, to the DEEPEST point (2-D distance-to-edge maximum) of the
#         lumen-bright region (T_loc_med <= HU <= max(1.3 I_c, I_c + 200)) connected to the lumen point nearest the old
#         centre, searched within min(R + 1, 3) mm; then the vessel is cast again from the moved stations (tangent frame
#         recomputed). No flagged station -> no change at all (bit-identical to the pilot). Recorded in seg_qc.json
#         "pinch_fix" (per vessel and pass: stations flagged, s-runs, moved, max move, flagged after the re-cast).
#   Everything else (CAP, rays, rules, recentring, diff 5 pre-pass, diff 7 seeds, diff 8 keep-connected opening) unchanged.
# Run: docker ... -e VMR_OPEN_BRIGHT=1 -e PILOT_SOURCE=<src> -e PILOT_CASE=<case> [-e PILOT_SEG_TAG=ct_seg4]
#      [-e SCALEUP_SELECT=select.json] neve-build-stagea python3 /sp/v5_va/scaleup_ct_seg.py
# --- pilot header follows ---
# SONG COPY (pilot_proc_song, re-copied 2026-10-01 21:55 incl. ia172's diff 8): pilot_proc_ia172_ct_seg.py with the song paths and pilot_proc_song_ imports only (the Song adapter differences live in pilot_proc_song_ctseg_common.py).
"""pilot_proc_song_ct_seg.py = /sp/v5_va/tag_ct_seg.py (tag_ctwall's copy of vmr_ct_seg3.py, the study's ct_seg3 /
ct_seg4 CT-edge segmentation) for a 172-series pilot donor. The algorithm and every parameter are unchanged (CAP 0.5 mm,
64 rays, 0.05 mm steps, bright-neighbour rule 1.3 I_c / I_c + 200 HU, dip rule, threshold-field smoothing, 2-pass
recentring, 0.8 mm-rule opening with VMR_OPEN_BRIGHT=1 = ct_seg4). Diff against tag_ct_seg.py:
  1. data: tag_ctwall_common.load_tag / build_stations -> pilot_proc_song_ctseg_common.load_donor / build_stations (same
     structures). Seeds = the VMTK MISR centreline of Arterial's cut-out VA surface (origin -> end, or -> dura when the
     side's distal course failed QC), the SCA from Arterial's RSA / LSA label (+-30 mm around the VA origin; TS-mask
     fallback), the BA from the two traces' shared stretch when both run to the confluence. Only donor-gate sides.
     CT = the native RAS ct.nii.gz (positive diagonal affine: no flip).
  2. output: carotid_data/va_sources/derived/thin_slice/song/<case>/ct_seg/<tag> (tag = PILOT_SEG_TAG, else ct_seg4).
  3. TAG_SE_R kept as in tag_ct_seg.py; UNSET (the primary run) = the original expression in the original 3 x 5 x 5 array.
     On these grids (0.39-0.51 mm in-plane x 0.25-0.5 mm z) it is a 3 x 3 x 3 cube (in-plane +-1 voxel, z +-1 voxel =
     +-0.25..0.5 mm): the in-plane +-2 voxels (0.78-1.01 mm) fall outside 0.8 mm except at 0.39 mm pixels. Recorded in
     seg_qc.json "structuring_element".
  4. vessels may be absent (a side that fails the gate, no BA): names = the present ones; the label value stays
     ALL.index + 1 (1 rva, 2 lva, 3 ba, 4 rsub, 5 lsub) so every reader of labels_kji.npz works unchanged.
  5. a prior-radius pre-pass (block "diff 5"): R = the image half-max r_eq of the unchanged rays (pass 2), measured from
     the seed with R = MISR first. tag_ctwall's prior was the same quantity (tag_labels' port of these rays).
  6. seg_qc.json "adapter" records the seed sources and the frame note; the seeds note names this script.
  7. (third instance, 21:10) seeds: where the first / last 1.5 mm of stations holds no label voxel, the seed is the
     centroid of the label's OWN first / last 1.5 mm (tag_ctwall_seeds.py's repair, built in); seg_qc.json "seeds_fix".
     Run in neve-build-stagea (eve-training-fixed has no nibabel).
  8. (third instance, 21:25) connectivity-preserving opening (block "diff 8"): where the 0.8 mm-rule opening would cut
     the VA label and "keep the largest piece" drop a piece of >= 50 voxels spanning >= 5 mm of stations beyond the kept
     piece (058 R: the whole 52 mm V1, cut at the C6 bone contact), the flagged station runs between them are left
     un-opened. Elsewhere identical. seg_qc.json open_bright.<v>.keep_connected_fix8. PILOT_KEEP_CONNECTED=0 = plain ct_seg4.
Run: docker ... -e VMR_OPEN_BRIGHT=1 -e PILOT_CASE=<case> eve-training-fixed python3 pilot_proc_song_ct_seg.py
     (pilot_proc_ia172.sh stage ctseg).
LICENCE: CC BY 4.0 image; derived through Arterial: non-commercial.

--- docstring of tag_ct_seg.py follows ---
tag_ct_seg.py = /sp/v5_va/vmr_ct_seg3.py (the study's ct_seg3 / ct_seg4 CT segmentation), COPIED for TaG-Net
patient A by the tag_ctwall agent. The algorithm and every parameter are unchanged. Diff against vmr_ct_seg3.py:
  1. data: vmr_common.load_vmr / build_stations -> tag_ctwall_common.load_tag / build_stations (same structures:
     the TaG-Net GT VA paths ostium -> merge, SCA, BA, with the image half-max r_eq as the prior radius R, times
     TAG_R_SCALE (default 1.0); the CTA crop flipped to positive spacing, frame-checked);
  2. output: carotid_data/va_sources/derived/tagnet/ct_seg/<tag> (tag = TAG_SEG_TAG, else ct_seg4 / ct_seg3);
  3. optional TAG_SE_R (mm): build the opening ellipsoid with that radius in an array big enough to hold it.
     UNSET (default) = the original expression in the original 3 x 5 x 5 array, which on the 0.405 x 0.405 x
     0.4 mm grid is a 3 x 3 x 3 cube (in-plane +-2 voxels = 0.81 mm > 0.8 mm), not VMR's 15-voxel 0.8 / 0.7 mm
     ellipsoid. TAG_SE_R=0.82 adds the in-plane +-2 and z +-2 voxels (sensitivity run);
  4. seg_qc.json also records the frame check, R scale and the structuring element; the seeds note names TaG-Net.
Run: bash /sp/v5_va/tag_ctwall_run.sh neve-build-stagea tag_ct_seg.py   (VMR_OPEN_BRIGHT=1 -> ct_seg4).
LICENCE: TaG-Net has no data licence: internal benchmarking only.

--- original docstring of vmr_ct_seg3.py follows ---
v3 (this file) = v2 + an UPPER threshold where a BRIGHTER neighbour touches the lumen before it has
fallen off (dense venous contrast in the left vertebral / subclavian vein, cortical bone): on that ray
T_hi = (I_c + neighbour peak within 1 mm)/2 (neighbour = HU > max(1.3 I_c, I_c + 200)), min-filtered over
5 stations x 3 rays; lumen = T_lo <= HU <= T_hi, F = min(HU - T_lo, T_hi - HU). v2 leaked into the left-V1
venous contrast (left V1 union 0.60 > 0.3 mm, circularity 0.73, union genus 1).

VMR 0248: both vertebral arteries (+ basilar) segmented from the CT (p12full.vti,
0.4 x 0.4 x 0.7 mm) with a LOCAL full-width-at-half-maximum edge (v2 of vmr_ct_seg.py).
VMR_OPEN_BRIGHT=1 (-> ct_seg4): also open the label (0.8 mm ellipsoid) at stations within 2 mm of a
bright-neighbour ray, removing the neighbour's partial-volume halo spurs; keep the largest piece.

Why v2: a single half-maximum threshold per station (lumen vs soft tissue, ~270 HU) leaked into
the enhancing vertebral venous plexus / cancellous bone around V2 (20-24 % of the label boundary
ran out to the ROI limit; results/vmr/zoom_leaks.png).

Method
  stations every 0.25 mm on RVA / LVA (origin -> merge), BA (merge -> right-PCA origin) and the
  subclavians (+-30 mm around each VA origin), radius R = expert contour r_eq.
  Per station, 64 rays in the normal plane, HU profile p(r) (trilinear, 0.05 mm) out to R + 3 mm:
    I_c   = lumen core (median HU within 0.4 R), smoothed sigma 2 mm along s
    I_bg  = soft tissue (-150..250 HU) median at R+1.2..R+2.8 mm, smoothed sigma 2 mm
    r_x   = first r >= 0.3 R where p < (I_c + I_bg)/2, or an earlier local minimum below 0.8 I_c
            that is followed by a >= 40 HU rise (a dip before a bright neighbour: vein / bone)
    I_ext = min p over [r_x, r_x + 0.8 mm]   (the immediate exterior on that ray)
    T_loc = (I_c + I_ext) / 2                 local half maximum on that ray
    r_e   = first outward crossing of T_loc   (sub-sample)
  T_loc smoothed over sigma 2 rays (11 deg) x 2 stations (0.5 mm) -- a threshold field, NOT the wall.
  Pass 1 rays start on the SimVascular path; the stations are then moved to the centroid of their
  edge polygon (smoothed sigma 1 mm) and pass 2 casts again from that medial line.
  3D: every crop voxel is assigned to the vessel with the smallest d/R; VA/BA ends are planar cuts
  normal to the path (origin = where the expert path leaves the subclavian lumen; merge).
  F(x) = HU(x) - T_loc(station(x), theta(x)); label = {F >= 0} & {rho <= r_e + CAP} in its own
  vessel, 6-connected to the stations, holes filled. CAP = 0.5 mm (only stops leaks; the wall is
  F = 0 in 3D, not the per-ray r_e).
  Surfaces: (A) stage-A protocol of v4/TopBrain: binary MC 0.5 + windowed sinc 20 / 0.1
  (topbrain_tools/mask_to_surface.write_vtp, unchanged); (B) sub-voxel isosurface F = 0 within the
  label's region, no smoothing.
Writes carotid_data/va_sources/derived/vmr0248/ct_seg2/. eve-training-fixed.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from vmr_common import CT, STEP, arclen, jdump, resample, rmf, smooth  # noqa: F401
import importlib  # noqa: E402
PSRC = os.environ["PILOT_SOURCE"]   # v2-a
assert PSRC in ("ia172", "song"), PSRC
_CC = importlib.import_module("scaleup_ctseg_common_%s" % PSRC)
R_SCALE, SEG_ROOT, build_stations, load_donor = _CC.R_SCALE, _CC.SEG_ROOT, _CC.build_stations, _CC.load_donor  # adapter (diff 1)
PINCH = os.environ.get("SCALEUP_PINCH_FIX", "1") == "1"   # v2-b
# v2-c (08:55): per-vessel FALLBACK to the pilot method. <SEG_ROOT>/pinch_skip_<tag>.json {"vessels": ["rva"], "why": ...}
# (written by scaleup_ctseg_fallback.sh when the fixed wall of that VA gives no usable VMTK centreline, e.g. Song 138 R: the
# fix extended the label 131 -> 154.5 mm but vmtkCenterlines returned 0.7 mm) -> the listed vessels are cast without the fix.
_PS_F = os.path.join(SEG_ROOT, "pinch_skip_%s.json" % os.environ.get(
    "PILOT_SEG_TAG", "ct_seg4" if os.environ.get("VMR_OPEN_BRIGHT", "0") == "1" else "ct_seg3"))
try:
    PINCH_SKIP = set(json.load(open(_PS_F)).get("vessels") or []) if os.path.isfile(_PS_F) else set()
except Exception:  # noqa: BLE001
    PINCH_SKIP = set()
PINCH_LOG = []
_CUR = {"vessel": None, "pass": None}

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from mask_to_surface import write_vtp  # noqa: E402

CAP = float(os.environ.get("VMR_CAP", 0.5))
NRAY = 64
DR = 0.05
BRIGHT_FACTOR, BRIGHT_PLUS = 1.3, 200.0   # a neighbour brighter than max(1.3 I_c, I_c + 200 HU)
NO_UPPER = 1.0e4
OPEN_BRIGHT = os.environ.get("VMR_OPEN_BRIGHT", "0") == "1"
KEEP_CONNECTED = os.environ.get("PILOT_KEEP_CONNECTED", "1") == "1"   # diff 8
OUT = os.path.join(SEG_ROOT, os.environ.get("PILOT_SEG_TAG", "ct_seg4" if OPEN_BRIGHT else "ct_seg3"))  # diff 2
SE_R = os.environ.get("TAG_SE_R")  # diff 3 (None = original structuring element)


def _runs_idx(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

vmr = load_donor(with_ct=True)  # diff 1
ct = CT(vmr["ct"])
ves = build_stations(vmr)
ALL = ["rva", "lva", "ba", "rsub", "lsub"]   # diff 4: label value = ALL.index + 1 whatever is present
names = [n for n in ALL if n in ves]
VAB = [n for n in names if n in ("rva", "lva", "ba")]


def KN(lst):
    return [(ALL.index(n), n) for n in lst]
TH = np.linspace(0, 2 * np.pi, NRAY, endpoint=False)


def core_and_bg(P, R, U, W):
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    Il, Ib = np.zeros(len(P)), np.zeros(len(P))
    for i in range(len(P)):
        rr = 0.4 * R[i] * np.sqrt(np.linspace(0, 1, 6))[:, None]
        G = P[i] + (rr * np.cos(th))[..., None] * U[i] + (rr * np.sin(th))[..., None] * W[i]
        Il[i] = np.median(ct.sample(G.reshape(-1, 3)))
        rr = R[i] + np.linspace(1.2, 2.8, 5)[:, None]
        G = P[i] + (rr * np.cos(th))[..., None] * U[i] + (rr * np.sin(th))[..., None] * W[i]
        h = ct.sample(G.reshape(-1, 3))
        h = h[(h > -150) & (h < 250)]
        Ib[i] = np.median(h) if len(h) > 10 else np.nan
    good = np.isfinite(Ib)
    Ib = np.interp(np.arange(len(P)), np.nonzero(good)[0], Ib[good]) if good.any() else np.full(len(P), 40.0)
    sg = 2.0 / STEP
    return ndimage.gaussian_filter1d(Il, sg, mode="nearest"), ndimage.gaussian_filter1d(Ib, sg, mode="nearest")


def cast(d):
    """v2-b wrapper: the unchanged ray cast (_cast_once), then the C2-exit pinch fix (see the header)."""
    d = _cast_once(d)
    if not PINCH or _CUR["vessel"] not in ("rva", "lva"):   # VAs only (not the SCA / BA stations)
        return d
    if _CUR["vessel"] in PINCH_SKIP:   # v2-c fallback: this VA is cast with the pilot method
        return d
    flag, hc, Tm, ffl = _pinch_flags(d)
    if not flag.any():
        return d
    rec = {"vessel": _CUR["vessel"], "pass": _CUR["pass"], "stations_flagged": int(flag.sum()),
           "flagged_s_runs_mm": [[float(d["s"][a]), float(d["s"][b])] for a, b in _runs_idx(flag)],
           "flag_by_centre_HU": int((hc < Tm)[flag].sum()), "flag_by_ray_floor": int((ffl >= 0.5)[flag].sum())}
    moved, mv = _pinch_move(d, flag, Tm)
    rec.update({"moved": int(moved.sum()), "move_mm_max": float(mv.max()) if moved.any() else 0.0,
                "move_mm_median": float(np.median(mv[moved])) if moved.any() else 0.0,
                "not_moved_no_lumen_found": int((flag & ~moved).sum())})
    if moved.any():
        T_, _, _ = rmf(d["P"])
        d["T"] = T_
        d = _cast_once(d)
        f2, _, _, _ = _pinch_flags(d)
        rec["flagged_after_recast"] = int(f2.sum())
        rec["flagged_after_s_runs_mm"] = [[float(d["s"][a]), float(d["s"][b])] for a, b in _runs_idx(f2)][:20]
    PINCH_LOG.append(rec)
    print("pinch_fix", json.dumps(rec), flush=True)
    return d


def _pinch_flags(d):
    """stations whose centre is not lumen: centre HU < median T_loc, or >= 50 % of the rays end at the 0.3 R start floor"""
    P, R, r_e = d["P"], d["R"], d["r_e"]
    hc = ct.sample(P)
    Tm = np.median(d["T_loc"], axis=1)
    r_floor = np.ceil(0.3 * R / DR) * DR
    ffl = np.mean(r_e <= r_floor[:, None] + 0.5 * DR, axis=1)
    return (hc < Tm) | (ffl >= 0.5), hc, Tm, ffl


def _pinch_move(d, flag, Tm):
    """move each flagged station, in its normal plane, to the deepest point of the lumen-bright region connected to the
    lumen point nearest its centre (search radius min(R + 1, 3) mm, 0.1 mm grid). Returns (moved mask, move distances)."""
    P, R, U, W, Ic = d["P"], d["R"], d["U"], d["W"], d["Ic"]
    moved = np.zeros(len(P), bool)
    mv = np.zeros(len(P))
    newP = P.copy()
    for i in np.nonzero(flag)[0]:
        rmax = min(float(R[i]) + 1.0, 3.0)
        g = np.arange(-rmax, rmax + 1e-9, 0.1)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        disc = gx ** 2 + gy ** 2 <= rmax ** 2
        G = P[i][None, None, :] + gx[..., None] * U[i] + gy[..., None] * W[i]
        h = ct.sample(G.reshape(-1, 3)).reshape(gx.shape)
        upper = max(BRIGHT_FACTOR * Ic[i], Ic[i] + BRIGHT_PLUS)
        lum = disc & (h >= Tm[i]) & (h <= upper)
        if not lum.any():
            continue
        labl, nl = ndimage.label(lum, structure=np.ones((3, 3)))
        rr2 = np.where(lum, gx ** 2 + gy ** 2, np.inf)
        k0 = np.unravel_index(int(np.argmin(rr2)), rr2.shape)
        comp = labl == labl[k0]
        dep = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
        dep = np.where(comp, dep - 1e-3 * np.sqrt(gx ** 2 + gy ** 2), -1.0)   # ties -> nearer the old centre
        k1 = np.unravel_index(int(np.argmax(dep)), dep.shape)
        newP[i] = P[i] + gx[k1] * U[i] + gy[k1] * W[i]
        mv[i] = float(np.hypot(gx[k1], gy[k1]))
        moved[i] = mv[i] > 1e-6
    d["P"] = newP
    return moved, mv


def _cast_once(d):
    """Rays from d['P'] in the (U, W) plane. Returns T_loc, r_e (n, NRAY), stats."""
    P, R = d["P"], d["R"]
    _, U, W = rmf(P)
    d["U"], d["W"] = U, W
    Ic, Ib = core_and_bg(P, R, U, W)
    rmax = R.max() + 3.0
    r = np.arange(0, rmax + 1e-9, DR)
    dirs = np.cos(TH)[None, :, None] * U[:, None, :] + np.sin(TH)[None, :, None] * W[:, None, :]  # n,NRAY,3
    n = len(P)
    T_loc = np.zeros((n, NRAY))
    T_hi = np.full((n, NRAY), NO_UPPER)
    kind = np.zeros((n, NRAY), np.int8)  # 0 soft-tissue crossing, 1 dip rule, 2 steepest descent, 3 bright neighbour
    for i in range(n):
        G = P[i][None, None, :] + r[None, :, None] * dirs[i][:, None, :]
        p = ct.sample(G.reshape(-1, 3)).reshape(NRAY, len(r))
        Ts = 0.5 * (Ic[i] + Ib[i])
        i0 = int(np.ceil(0.3 * R[i] / DR))
        ilim = min(len(r) - 1, int((R[i] + 3.0) / DR))
        for a in range(NRAY):
            q = p[a]
            below = np.nonzero(q[i0:ilim] < Ts)[0]
            ix = i0 + below[0] if len(below) else None
            # earlier dip: local min < 0.8 Ic followed by a >= 40 HU rise (within 1 mm)
            seg = q[i0:(ix if ix is not None else ilim)]
            jd = None
            if len(seg) > 3:
                mins = np.nonzero((seg[1:-1] <= seg[:-2]) & (seg[1:-1] <= seg[2:]) & (seg[1:-1] < 0.8 * Ic[i]))[0] + 1
                for jm in mins:
                    after = q[i0 + jm:i0 + jm + int(1.0 / DR)]
                    if after.max() - seg[jm] >= 40:
                        jd = i0 + jm
                        break
            if jd is not None:
                rx, k = jd, 1
            elif ix is not None:
                rx, k = ix, 0
            else:
                dq = np.gradient(q[i0:ilim])
                rx, k = i0 + int(np.argmin(dq)), 2
            I_ext = q[rx:rx + int(0.8 / DR) + 1].min()
            if k == 1:
                I_ext = q[rx]
            T_loc[i, a] = 0.5 * (Ic[i] + I_ext)
            kind[i, a] = k
            # v3: a BRIGHTER neighbour (dense venous contrast, cortical bone) reached before the lumen
            # has fallen off: upper threshold half way between the lumen and that neighbour's peak
            thr_b = max(BRIGHT_FACTOR * Ic[i], Ic[i] + BRIGHT_PLUS)
            hb = np.nonzero(q[i0:ilim] > thr_b)[0]
            if len(hb):
                jb = i0 + hb[0]
                if k == 2 or jb < rx:
                    Ipk = q[jb:jb + int(1.0 / DR) + 1].max()
                    T_hi[i, a] = 0.5 * (Ic[i] + Ipk)
                    kind[i, a] = 3
    # smooth the threshold field over rays (periodic) and stations
    Tp = ndimage.gaussian_filter1d(T_loc, 2.0, axis=1, mode="wrap")
    Tp = ndimage.gaussian_filter1d(Tp, 2.0, axis=0, mode="nearest")
    # upper threshold: spread to the neighbouring rays / stations (min filter), no blur with the "none" value
    Th = ndimage.minimum_filter(T_hi, size=(5, 3), mode=("nearest", "wrap"))
    # edge radius on each ray: first exit from [T_lo, T_hi]
    r_e = np.zeros((n, NRAY))
    for i in range(n):
        G = P[i][None, None, :] + r[None, :, None] * dirs[i][:, None, :]
        p = ct.sample(G.reshape(-1, 3)).reshape(NRAY, len(r))
        i0 = int(np.ceil(0.3 * R[i] / DR))
        for a in range(NRAY):
            q = p[a, i0:]
            b = np.nonzero((q < Tp[i, a]) | (q > Th[i, a]))[0]
            if len(b) == 0:
                r_e[i, a] = r[-1]
                continue
            j = b[0]
            if j == 0:
                r_e[i, a] = r[i0]
                continue
            q0, q1 = q[j - 1], q[j]
            lev = Tp[i, a] if q1 < Tp[i, a] else Th[i, a]
            f = (q0 - lev) / (q0 - q1) if abs(q0 - q1) > 1e-6 else 0.5
            r_e[i, a] = r[i0 + j - 1] + min(max(f, 0.0), 1.0) * DR
    d.update({"Ic": Ic, "Ib": Ib, "T_loc": Tp, "T_raw": T_loc, "T_hi": Th, "kind": kind, "r_e": r_e, "dirs": dirs})
    return d


def recentre(d):
    """Move each station to the centroid of its edge polygon; smooth sigma 1 mm; re-resample."""
    P, r_e, dirs = d["P"], d["r_e"], d["dirs"]
    pts = P[:, None, :] + r_e[..., None] * dirs
    # polygon area centroid in-plane
    U, W = d["U"], d["W"]
    x = np.einsum("naj,nj->na", pts - P[:, None, :], U)
    y = np.einsum("naj,nj->na", pts - P[:, None, :], W)
    x1, y1 = np.roll(x, -1, 1), np.roll(y, -1, 1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum(1)
    cx = ((x + x1) * cr).sum(1) / (6 * A)
    cy = ((y + y1) * cr).sum(1) / (6 * A)
    newP = P + cx[:, None] * U + cy[:, None] * W
    shift = np.hypot(cx, cy)
    newP = smooth(newP, 1.0 / STEP)
    s_old = d["s"]
    q, Pn, Rn = resample(newP, STEP, d["R"])
    T, _, _ = rmf(Pn)
    return {"s": q, "P": Pn, "R": Rn, "T": T, "s_scale": q[-1] / max(s_old[-1], 1e-9)}, shift, np.sqrt(np.abs(A) / np.pi)


qc = {"cap_mm": CAP, "n_rays": NRAY, "vessels": {}}
# ---------------------------------------------------------------- diff 5: prior radius pre-pass
# tag_ctwall's prior R was the image half-max equal-area radius of the ct_seg3 rays, pass 2 (VMR's was the expert contour
# r_eq). A donor has neither, so the same quantity is measured here first: the unchanged cast / recentre / cast / recentre
# from the VMTK seed with R = its MISR; the pass-2 polygon r_eq (smoothed sigma 1 mm, clipped 0.5-5 mm) becomes R, mapped
# back onto the seed stations by nearest point. The main passes below then run exactly as in vmr_ct_seg3.py.
qc["prior_R_prepass"] = {}
if os.environ.get("PILOT_NO_PREPASS", "0") != "1":
    for nm in names:
        d0 = {kk_: (vv_.copy() if hasattr(vv_, "copy") else vv_) for kk_, vv_ in ves[nm].items()}
        _CUR.update(vessel=nm, **{"pass": "prepass_1"})
        d0 = cast(d0)
        d1, _, _ = recentre(d0)
        _CUR.update(vessel=nm, **{"pass": "prepass_2"})
        d1 = cast(d1)
        _, _, req_p = recentre(d1)
        _, jn = cKDTree(d1["P"]).query(ves[nm]["P"])
        Rn = np.clip(ndimage.gaussian_filter1d(np.asarray(req_p, float), 1.0 / STEP, mode="nearest")[jn], 0.5, 5.0)
        qc["prior_R_prepass"][nm] = {"R_seed_median": float(np.median(ves[nm]["R"])), "R_prior_median": float(np.median(Rn)),
                                     "ratio_median": float(np.median(Rn / np.maximum(ves[nm]["R"], 1e-3)))}
        print("prepass", nm, qc["prior_R_prepass"][nm], flush=True)
        ves[nm]["R"] = Rn
for nm in names:
    _CUR.update(vessel=nm, **{"pass": "pass_1"})
    d = cast(ves[nm])
    d2, shift, req = recentre(d)
    _CUR.update(vessel=nm, **{"pass": "pass_2"})
    d2 = cast(d2)
    _, shift2, req2 = recentre(d2)
    ves[nm] = d2
    kinds = np.bincount(d2["kind"].ravel(), minlength=4) / d2["kind"].size
    qc["vessels"][nm] = {"pass1_shift_median_mm": float(np.median(shift)), "pass1_shift_p90_mm": float(np.percentile(shift, 90)),
                         "pass2_shift_median_mm": float(np.median(shift2)),
                         "r_eq_rays_median": float(np.median(req2)), "r_contour_median": float(np.median(d2["R"])),
                         "I_core_median": float(np.median(d2["Ic"])), "I_bg_median": float(np.median(d2["Ib"])),
                         "T_loc_median": float(np.median(d2["T_loc"])),
                         "T_loc_p10_p90": [float(np.percentile(d2["T_loc"], 10)), float(np.percentile(d2["T_loc"], 90))],
                         "ray_rule_frac": {"soft_tissue_halfmax": float(kinds[0]), "dip_before_bright": float(kinds[1]),
                                           "steepest_descent": float(kinds[2]), "bright_neighbour_upper": float(kinds[3])},
                         "T_hi_rays_frac": float((d2["T_hi"] < NO_UPPER).mean())}
    print(nm, qc["vessels"][nm], flush=True)

# ---------------------------------------------------------------- 3D field
allP = np.vstack([ves[n]["P"] for n in names])
Rmax = max(ves[n]["R"].max() for n in names)
lo = allP.min(0) - (Rmax + 4)
hi = allP.max(0) + (Rmax + 4)
i0 = np.maximum(np.floor((lo - ct.org) / ct.sp).astype(int), 0)
i1 = np.minimum(np.ceil((hi - ct.org) / ct.sp).astype(int), ct.dims - 1)
sub = ct.arr[i0[2]:i1[2] + 1, i0[1]:i1[1] + 1, i0[0]:i1[0] + 1].astype(np.float32)
corg = ct.org + i0 * ct.sp
shp = sub.shape
kk, jj, ii = np.meshgrid(np.arange(shp[0]), np.arange(shp[1]), np.arange(shp[2]), indexing="ij")
X = corg + np.stack([ii, jj, kk], -1).reshape(-1, 3) * ct.sp
del kk, jj, ii
best = np.full(len(X), np.inf)
vid = np.full(len(X), -1, np.int16)
sid = np.zeros(len(X), np.int32)
for k, nm in KN(names):
    d = ves[nm]
    rmx = float(np.max(d["r_e"])) + CAP + 1.0
    dd, ix = cKDTree(d["P"]).query(X, distance_upper_bound=rmx)
    ok = np.isfinite(dd)
    nd = np.full(len(X), np.inf)
    nd[ok] = dd[ok] / d["R"][ix[ok]]
    better = nd < best
    best[better] = nd[better]
    vid[better] = k
    sid[better] = ix[better]
for k, nm in KN(VAB):
    d = ves[nm]
    for end, j in (("start", 0), ("end", len(d["P"]) - 1)):
        m = (vid == k) & (sid == j)
        proj = (X[m] - d["P"][j]) @ d["T"][j]
        bad = proj < 0 if end == "start" else proj > 0
        vid[np.nonzero(m)[0][bad]] = -2
Tx = np.full(len(X), 1e4, np.float32)
Txh = np.full(len(X), NO_UPPER, np.float32)
cap = np.zeros(len(X), bool)
rho = np.zeros(len(X), np.float32)
for k, nm in KN(names):
    m = np.nonzero(vid == k)[0]
    d = ves[nm]
    j = sid[m]
    v = X[m] - d["P"][j]
    v -= np.einsum("ij,ij->i", v, d["T"][j])[:, None] * d["T"][j]
    xx = np.einsum("ij,ij->i", v, d["U"][j])
    yy = np.einsum("ij,ij->i", v, d["W"][j])
    ang = np.mod(np.arctan2(yy, xx), 2 * np.pi) / (2 * np.pi) * NRAY
    a0 = np.floor(ang).astype(int) % NRAY
    a1 = (a0 + 1) % NRAY
    f = ang - np.floor(ang)
    Tl = (1 - f) * d["T_loc"][j, a0] + f * d["T_loc"][j, a1]
    re = (1 - f) * d["r_e"][j, a0] + f * d["r_e"][j, a1]
    rh = np.hypot(xx, yy)
    Tx[m] = Tl
    Txh[m] = np.minimum(d["T_hi"][j, a0], d["T_hi"][j, a1])
    rho[m] = rh
    cap[m] = rh <= re + CAP
hu = sub.reshape(-1)
F = np.minimum(hu - Tx, Txh - hu).astype(np.float32)
cand = (vid >= 0) & cap & (F >= 0)
lab, nlab = ndimage.label(cand.reshape(shp))
seeds = []
for nm in names:
    q = np.clip(np.round((ves[nm]["P"] - corg) / ct.sp).astype(int), 0, np.array(shp[::-1]) - 1)
    seeds.append(lab[q[:, 2], q[:, 1], q[:, 0]])
seeds = np.concatenate(seeds)
keep = np.unique(seeds[seeds > 0])
lumen = ndimage.binary_fill_holes(np.isin(lab, keep)).reshape(-1)
qc.update({"crop_shape_kji": shp, "crop_origin_mm": corg, "spacing_mm": ct.sp, "n_candidate_components": int(nlab),
           "n_seeded_components": int(len(keep)), "seed_coverage": float((seeds > 0).mean())})
print("components %d seeded %d coverage %.4f" % (nlab, len(keep), (seeds > 0).mean()), flush=True)

labels = np.zeros(len(X), np.uint8)
bone = ndimage.binary_dilation(sub >= 700, iterations=1).reshape(-1)
for k, nm in KN(names):
    m = lumen & (vid == k)
    labels[m] = k + 1
L3 = labels.reshape(shp)
if OPEN_BRIGHT:
    # v4 option: where a ray met a brighter neighbour (+-2 mm of station), the neighbour's partial-volume
    # halo (T_lo <= HU <= T_hi) can still hang on the lumen as thin spurs: open the VA label there with an
    # ellipsoid of 0.8 mm radius (2 x 2 x 1 voxels), keep the part connected to the rest of the label.
    if SE_R is None:
        se = np.zeros((3, 5, 5), bool)
        kk_, jj_, ii_ = np.mgrid[-1:2, -2:3, -2:3]
        se[(kk_ * ct.sp[2]) ** 2 + (jj_ * ct.sp[1]) ** 2 + (ii_ * ct.sp[0]) ** 2 <= 0.8 ** 2 + 1e-6] = True
    else:  # diff 3: same formula, radius SE_R, array sized to hold it
        rse = float(SE_R)
        nk, nj, ni = (int(np.floor(rse / ct.sp[2] + 1e-9)), int(np.floor(rse / ct.sp[1] + 1e-9)),
                      int(np.floor(rse / ct.sp[0] + 1e-9)))
        kk_, jj_, ii_ = np.mgrid[-nk:nk + 1, -nj:nj + 1, -ni:ni + 1]
        se = (kk_ * ct.sp[2]) ** 2 + (jj_ * ct.sp[1]) ** 2 + (ii_ * ct.sp[0]) ** 2 <= rse ** 2 + 1e-6
    qc["structuring_element"] = {"TAG_SE_R": SE_R, "shape_kji": list(se.shape), "n_voxels": int(se.sum()),
                                 "extent_mm_kji": [float(np.abs(kk_[se]).max() * ct.sp[2]),
                                                   float(np.abs(jj_[se]).max() * ct.sp[1]),
                                                   float(np.abs(ii_[se]).max() * ct.sp[0])]}
    print("structuring element", qc["structuring_element"], flush=True)
    qc["open_bright"] = {}
    for k, nm in KN(VAB):
        d = ves[nm]
        flag_st = (d["T_hi"] < NO_UPPER).any(1)
        w = int(round(2.0 / STEP))
        flag_st = ndimage.binary_dilation(flag_st, iterations=w) if flag_st.any() else flag_st
        Lk = L3 == k + 1
        reg = np.zeros(len(X), bool)
        mk = np.nonzero(vid == k)[0]
        reg[mk] = flag_st[sid[mk]]
        reg = reg.reshape(shp)
        if not reg.any():
            qc["open_bright"][nm] = {"stations_flagged": 0}
            continue
        opened = ndimage.binary_opening(Lk, structure=se)

        def _keep_largest(reg_):
            nl = np.where(reg_, opened, Lk)
            labc_, nc_ = ndimage.label(nl)
            if nc_ > 1:
                sizes = ndimage.sum(nl, labc_, range(1, nc_ + 1))
                big_ = int(np.argmax(sizes)) + 1
                return labc_ == big_, labc_, big_, nc_
            return nl, labc_, 1, nc_
        newL, labc, big, nc = _keep_largest(reg)
        # diff 8 (third instance, 21:25): connectivity-preserving opening. On ia172_058 R the opening at a bone contact
        # (C6 foramen, s 42-55) cut the label in two and "keep the largest piece" dropped the whole V1 (52 mm). Where a
        # dropped piece holds >= 50 voxels and spans >= 5 mm of the vessel's stations outside the kept piece, the flagged
        # station runs between them are left un-opened (the ct_seg3 label there) and the largest piece is taken again.
        # Everything else (the opening elsewhere, its element, the flags) is unchanged. PILOT_KEEP_CONNECTED=0 disables it.
        fix8 = {"enabled": KEEP_CONNECTED, "runs_not_opened_s_mm": [], "pieces": []}
        if KEEP_CONNECTED:
            newL0 = newL.copy()
            vk = (vid == k).reshape(shp)
            sidk = sid.reshape(shp)
            runs8 = _runs_idx(flag_st)
            done8 = set()
            for _it in range(8):
                if nc <= 1:
                    break
                kept_s = d["s"][sidk[newL & vk]]
                if len(kept_s) == 0:
                    break
                kmin, kmax = float(kept_s.min()), float(kept_s.max())
                worst = None
                for c_ in range(1, nc + 1):
                    if c_ == big:
                        continue
                    pm = (labc == c_) & vk
                    if pm.sum() < 50:
                        continue
                    ps = d["s"][sidk[pm]]
                    span = float(ps.max() - ps.min())
                    if span < 5.0:
                        continue
                    if ps.max() <= kmin + 1.0:
                        gap = (float(ps.max()), kmin)
                    elif ps.min() >= kmax - 1.0:
                        gap = (kmax, float(ps.min()))
                    else:
                        continue
                    if worst is None or span > worst[0]:
                        worst = (span, gap, float(ps.min()), float(ps.max()), int(pm.sum()))
                if worst is None:
                    break
                g0, g1 = worst[1]
                cand = [(a, b) for a, b in runs8 if d["s"][b] >= g0 - 3.0 and d["s"][a] <= g1 + 3.0 and (a, b) not in done8]
                if not cand:
                    break
                for a, b in cand:
                    reg[(sidk >= a) & (sidk <= b) & vk] = False
                    done8.add((a, b))
                    fix8["runs_not_opened_s_mm"].append([float(d["s"][a]), float(d["s"][b])])
                fix8["pieces"].append({"s_range_mm": [worst[2], worst[3]], "voxels": worst[4], "gap_s_mm": [g0, g1]})
                newL, labc, big, nc = _keep_largest(reg)
            fix8["voxels_kept_by_fix"] = int((newL & ~newL0).sum())
        removed = int((Lk & ~newL).sum())
        L3[Lk & ~newL] = 0
        qc["open_bright"][nm] = {"stations_flagged": int(flag_st.sum()), "flagged_s_ranges_mm": [
            [float(d["s"][a]), float(d["s"][b])] for a, b in _runs_idx(flag_st)], "voxels_removed": removed,
            "keep_connected_fix8": fix8}
        print("open_bright", nm, qc["open_bright"][nm], flush=True)
    labels = L3.reshape(-1)
for k, nm in KN(VAB):
    m = labels == k + 1
    d = ves[nm]
    Lk = L3 == k + 1
    bd = (Lk & ~ndimage.binary_erosion(Lk)).reshape(-1)
    j = sid[m]
    re_v = np.zeros(m.sum())
    # boundary voxels at the cap
    mm = np.nonzero(m)[0]
    v = X[mm] - d["P"][j]
    v -= np.einsum("ij,ij->i", v, d["T"][j])[:, None] * d["T"][j]
    xx = np.einsum("ij,ij->i", v, d["U"][j]); yy = np.einsum("ij,ij->i", v, d["W"][j])
    ang = np.mod(np.arctan2(yy, xx), 2 * np.pi) / (2 * np.pi) * NRAY
    a0 = np.floor(ang).astype(int) % NRAY
    re_v = d["r_e"][j, a0]
    at_cap = bd[mm] & (np.hypot(xx, yy) > re_v + CAP - 0.3)
    s_vox = d["s"][j]
    nb = int(np.ceil(d["s"][-1]))
    hist = np.bincount(np.minimum(s_vox.astype(int), nb - 1), minlength=nb) * np.prod(ct.sp)
    r_eqv = np.sqrt(hist / np.pi)
    qc["vessels"][nm].update({
        "voxels": int(m.sum()), "volume_mm3": float(m.sum() * np.prod(ct.sp)), "length_mm": float(d["s"][-1]),
        "r_volume_equiv_median": float(np.median(r_eqv[2:-2])),
        "frac_boundary_at_cap": float(at_cap.sum() / max(bd[mm].sum(), 1)),
        "frac_vox_touching_bone700": float(bone[m].mean()), "r_volume_equiv_per_mm": np.round(r_eqv, 3)})
    print("%-4s vox %d  r_vol %.2f  r_rays %.2f  r_contour %.2f  at-cap %.4f bone-touch %.4f" % (
        nm, m.sum(), qc["vessels"][nm]["r_volume_equiv_median"], qc["vessels"][nm]["r_eq_rays_median"],
        np.median(d["R"]), qc["vessels"][nm]["frac_boundary_at_cap"], bone[m].mean()), flush=True)
    # diff 7 (third instance, 21:10): tag_ctwall_seeds.py's repair built in. Where the label does not reach a station end
    # (its first / last 1.5 mm empty: the VA origin voxels go to the subclavian label, the merge end to the BA or the
    # 0.8 mm-rule opening), the seed is the centroid of the label's OWN first / last 1.5 mm instead of NaN.
    Xm = X[m]
    fix = {"label_s_range_mm": [float(s_vox.min()), float(s_vox.max())], "station_length_mm": float(d["s"][-1]),
           "missing_at_start_mm": float(s_vox.min()), "missing_at_end_mm": float(d["s"][-1] - s_vox.max())}
    a = Xm[s_vox <= 1.5].mean(0) if (s_vox <= 1.5).any() else None
    b = Xm[s_vox >= d["s"][-1] - 1.5].mean(0) if (s_vox >= d["s"][-1] - 1.5).any() else None
    note = "centroids of the label's first / last 1.5 mm (%s donor CT seg, scaleup_ct_seg.py = pipeline v2)" % PSRC
    if a is None:
        a = Xm[s_vox <= s_vox.min() + 1.5].mean(0)
        fix["source_replaced"] = "centroid of the label's own first 1.5 mm (s %.2f-%.2f)" % (s_vox.min(), s_vox.min() + 1.5)
    if b is None:
        b = Xm[s_vox >= s_vox.max() - 1.5].mean(0)
        fix["target_replaced"] = "centroid of the label's own last 1.5 mm (s %.2f-%.2f)" % (s_vox.max() - 1.5, s_vox.max())
    if "source_replaced" in fix or "target_replaced" in fix:
        note += "; repaired as tag_ctwall_seeds.py (label does not reach the station end)"
    qc.setdefault("seeds_fix", {})[nm] = fix
    print("seeds", nm, fix, flush=True)
    jdump({"source": a, "target": b, "note": note}, os.path.join(OUT, "%s_seeds.json" % nm))

np.savez_compressed(os.path.join(OUT, "labels_kji.npz"), labels=L3, origin_mm=corg, spacing_mm=ct.sp,
                    names=np.array(["bg"] + ALL), present=np.array(names), note="1 rva, 2 lva, 3 ba, 4 rsub, 5 lsub; array [k,j,i]")
jdump({nm: {"s": ves[nm]["s"], "P": ves[nm]["P"], "R": ves[nm]["R"], "T_loc": ves[nm]["T_loc"],
            "T_hi": ves[nm]["T_hi"], "r_e": ves[nm]["r_e"], "Ic": ves[nm]["Ic"], "Ib": ves[nm]["Ib"], "kind": ves[nm]["kind"]}
       for nm in names}, os.path.join(OUT, "stations_rays.json"))

from skimage import measure  # noqa: E402
import pyvista as pv  # noqa: E402

F3 = F.reshape(shp)
V3 = vid.reshape(shp)


def open_edges(s):
    return int(s.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                       non_manifold_edges=False).n_cells)


surf_info = {}
for k, nm in KN(VAB):
    m = L3 == k + 1
    vol = np.pad(m.transpose(2, 1, 0).astype(np.uint8), 1)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=tuple(ct.sp))
    pa = os.path.join(OUT, "%s_stageA.vtp" % nm)
    write_vtp(corg + verts - ct.sp, faces, pa)
    sA = pv.read(pa)
    reg = ndimage.binary_dilation(m, iterations=1) & (V3 == k)
    G = np.where(reg, F3, -50.0)
    G = np.where(reg & ~m, np.minimum(G, -1.0), G)
    Gi = np.pad(G.transpose(2, 1, 0), 1, constant_values=-50.0)
    verts, faces, _, _ = measure.marching_cubes(Gi, level=0.0, spacing=tuple(ct.sp))
    sB = pv.PolyData(corg + verts - ct.sp, np.c_[np.full(len(faces), 3), faces].ravel()).clean()
    sB = sB.connectivity(extraction_mode="largest").extract_surface().triangulate().clean()
    pb = os.path.join(OUT, "%s_iso.vtp" % nm)
    sB.save(pb)
    surf_info[nm] = {"stageA": {"file": pa, "points": int(sA.n_points), "open_edges": open_edges(sA)},
                     "iso": {"file": pb, "points": int(sB.n_points), "open_edges": open_edges(sB)}}
    print(nm, surf_info[nm], flush=True)
m = (L3 >= 1) & (L3 <= 3)
vol = np.pad(m.transpose(2, 1, 0).astype(np.uint8), 1)
verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=tuple(ct.sp))
write_vtp(corg + verts - ct.sp, faces, os.path.join(OUT, "posterior_tree_stageA.vtp"))
qc["surfaces"] = surf_info
qc["adapter"] = {"phantom": vmr.get("phantom"), "frame_check": vmr["ct"].get("frame_check"), "nifti_block_ijk": vmr["ct"].get("nifti_block_ijk"), "seed_info": vmr["seed_info"], "vessels_present": names,
                     "R_scale": R_SCALE, "open_bright": OPEN_BRIGHT, "TAG_SE_R": SE_R, "licence": vmr["licence"]}
qc["pinch_fix"] = {"enabled": PINCH, "rule": "station centre HU < median T_loc OR >= 50 % of rays at the 0.3 R floor -> moved in-plane to the deepest lumen-bright point (<= min(R + 1, 3) mm), re-cast; VAs only", "events": PINCH_LOG,
                   "triggered": bool(PINCH_LOG), "skipped_vessels_fallback": sorted(PINCH_SKIP),
                   "skip_file": _PS_F if PINCH_SKIP else None}
qc["pipeline"] = {"version": "v2", "script": "scaleup_ct_seg.py", "source": PSRC, "select_file": os.environ.get("SCALEUP_SELECT", "select.json")}
qc["seconds"] = round(time.time() - t0, 1)
jdump(qc, os.path.join(OUT, "seg_qc.json"))
print("done %.0f s" % (time.time() - t0))
