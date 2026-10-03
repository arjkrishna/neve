#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_eye172_origin.py (= /sp/v5_va/v5_eye172_origin.py in the build containers)
"""v5_eye172_origin.py <source> <case> <side> [ext] -- (v5_eye172, 2026-10-02) read-only by-eye look at a donor VA ORIGIN:
does the traced VA leave the subclavian through an enhanced lumen, or does it start elsewhere (arch origin, tissue gap)?
Native CT, window -150 .. 1.6 x CCA. Row 1: 6 mm thin-slab MIPs (coronal, sagittal, axial) +-35 mm around the trace point
at s = 3 mm. Rows 2-3: 12 axial slices from 12 mm below to 21 mm above the trace start (3 mm steps), +-30 mm.
Overlays: TS 1.5 mm aorta (red), subclavian (magenta), CCA (green), brachiocephalic trunk (blue), BCV/SVC (orange);
Arterial native mask (cyan); trace 0-40 mm (yellow), 0-5 mm (red); Arterial's own VA-labelled centreline of that side
(white dashes, points_src_world_mm). Writes /sp/v5_va/results/v5_eye172/origin_<case>_<side>.png + .json (HU along the
first 15 mm of the trace and the max HU on short straight links from the trace start to the TS subclavian)."""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRCN, CASE, SIDE = sys.argv[1], sys.argv[2], sys.argv[3]
EXT = sys.argv[4] if len(sys.argv) > 4 else "tsba"
SRC = "/repo/carotid_data/va_sources/thin_slice/%s/%s" % (SRCN, CASE)
DER = "/repo/carotid_data/va_sources/derived/thin_slice/%s/%s" % (SRCN, CASE)
OUT = "/sp/v5_va/results/v5_eye172"
Q = json.load(open(os.path.join(DER, "qc", "qc.json")))
cca = float(Q.get("cca_median_HU_native") or 400.0)
P = np.asarray(json.load(open(os.path.join(DER, "arterial/%s/va_extract.json" % EXT)))["sides"][SIDE]["trace"]["points"], float)
s0 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
ct = nib.load(os.path.join(SRC, "ct.nii.gz"))
CT = np.asanyarray(ct.dataobj).astype(np.float32)
Ainv = np.linalg.inv(ct.affine)


def samp(vol, Ai, W, order=1):
    sh = W.shape[:-1]
    W2 = W.reshape(-1, 3)
    I = (Ai[:3, :3] @ W2.T + Ai[:3, 3:4])
    if order == 0:
        I = np.round(I).astype(int)
        ok = np.all((I >= 0) & (I < np.array(vol.shape)[:, None]), axis=0)
        out = np.zeros(W2.shape[0], bool)
        out[ok] = vol[I[0, ok], I[1, ok], I[2, ok]]
        return out.reshape(sh)
    return map_coordinates(vol, I, order=1, mode="nearest", output=np.float32).reshape(sh)


TSD = os.path.join(DER, "totalseg_1p5/segmentations")
sd = "left" if SIDE == "L" else "right"
groups = [("aorta", ["aorta"], "red"), ("SCA", ["subclavian_artery_left", "subclavian_artery_right"], "magenta"),
          ("CCA", ["common_carotid_artery_left", "common_carotid_artery_right"], "lime"),
          ("BT", ["brachiocephalic_trunk"], "deepskyblue"),
          ("BCV/SVC", ["brachiocephalic_vein_left", "brachiocephalic_vein_right", "superior_vena_cava"], "orange")]
masks = []
mp = os.path.join(DER, "arterial/runs/native/extracranial_vessels/segmentation.nii.gz")
if os.path.isfile(mp):
    m = nib.load(mp)
    masks.append(("Arterial", np.asanyarray(m.dataobj) > 0, np.linalg.inv(m.affine), "cyan"))
own_sca = None
for gname, names, col in groups:
    acc, Ai = None, None
    for n in names:
        p = os.path.join(TSD, n + ".nii.gz")
        if os.path.isfile(p):
            im = nib.load(p)
            v = np.asanyarray(im.dataobj) > 0
            acc = v if acc is None else (acc | v)
            Ai = np.linalg.inv(im.affine)
            if n == "subclavian_artery_%s" % sd:
                own_sca = (v, im.affine)
    if acc is not None:
        masks.append((gname, acc, Ai, col))
lab = []
lcp = os.path.join(DER, "arterial/runs/native/labelled_centerlines.json")
if os.path.isfile(lcp):
    for sg in json.load(open(lcp)).get("segments", []):
        if sg.get("class") == ("LVA" if SIDE == "L" else "RVA"):
            lab.append(np.asarray(sg.get("points_src_world_mm") or sg["points_ras_mm"], float))

p3 = P[np.searchsorted(s0, 3.0)]
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 6)
R = 35.0
g = np.arange(-R, R + 0.01, 0.4)
slab = np.arange(-3.0, 3.01, 0.5)
views = [("coronal (x-z), image left = patient right", 0, 2, 1), ("sagittal (y-z), anterior right", 1, 2, 0),
         ("axial (x-y), image left = patient right, anterior up", 0, 1, 2)]
for vi, (title, a, b, c) in enumerate(views):
    ax = fig.add_subplot(gs[0, 2 * vi:2 * vi + 2])
    A, Bm = np.meshgrid(g, g)
    acc = None
    for t in slab:
        W = np.zeros(A.shape + (3,))
        W[..., a] = p3[a] + (-A if a == 0 else A)
        W[..., b] = p3[b] + Bm
        W[..., c] = p3[c] + t
        im = samp(CT, Ainv, W)
        acc = im if acc is None else np.maximum(acc, im)
    ax.imshow(acc, cmap="gray", vmin=-150, vmax=1.6 * cca, origin="lower", extent=[-R, R, -R, R])
    W = np.zeros(A.shape + (3,))
    W[..., a] = p3[a] + (-A if a == 0 else A)
    W[..., b] = p3[b] + Bm
    W[..., c] = p3[c]
    for (mn, vol, Ai, col) in masks:
        mk = samp(vol, Ai, W, order=0).astype(float)
        if mk.any():
            ax.contour(g, g, mk, levels=[0.5], colors=col, linewidths=0.7)

    def proj(X):
        u = (X[:, a] - p3[a]) * (-1 if a == 0 else 1)
        return u, X[:, b] - p3[b]
    u, v = proj(P[s0 <= 40])
    ax.plot(u, v, "-", color="yellow", lw=1)
    u, v = proj(P[s0 <= 5])
    ax.plot(u, v, "-", color="red", lw=2)
    for L_ in lab:
        u, v = proj(L_)
        ok = (np.abs(u) < R) & (np.abs(v) < R)
        ax.plot(u[ok], v[ok], "--", color="white", lw=0.8)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_title("6 mm MIP " + title, fontsize=9)
ZLO = float(os.environ.get("ORIGIN_ZLO", "-12"))
ZST = float(os.environ.get("ORIGIN_ZSTEP", "3"))
TAG = os.environ.get("ORIGIN_TAG", "")
zs = P[0, 2] + ZLO + np.arange(12) * ZST
R2 = 30.0
g2 = np.arange(-R2, R2 + 0.01, 0.35)
for i, z in enumerate(zs[:12]):
    ax = fig.add_subplot(gs[1 + i // 6, i % 6])
    j = np.argmin(np.abs(P[:, 2] - z)) if np.abs(P[:, 2] - z).min() < 2 else 0
    cx, cy = P[0, 0], P[0, 1]
    XX, YY = np.meshgrid(-g2, g2)
    W = np.stack([cx + XX, cy + YY, np.full_like(XX, z)], -1)
    ax.imshow(samp(CT, Ainv, W), cmap="gray", vmin=-150, vmax=1.6 * cca, origin="lower", extent=[-R2, R2, -R2, R2])
    for (mn, vol, Ai, col) in masks:
        mk = samp(vol, Ai, W, order=0).astype(float)
        if mk.any():
            ax.contour(g2, g2, mk, levels=[0.5], colors=col, linewidths=0.6)
    near = np.abs(P[:, 2] - z) < 1.5
    if near.any():
        ax.plot(-(P[near, 0] - cx), P[near, 1] - cy, ".", color="yellow", ms=3)
    for L_ in lab:
        nr = np.abs(L_[:, 2] - z) < 1.5
        if nr.any():
            ax.plot(-(L_[nr, 0] - cx), L_[nr, 1] - cy, "x", color="white", ms=4)
    ax.plot([0], [0], "+", color="red", ms=8)
    ax.set_title("z %.1f (trace start %+.0f mm)" % (z, z - P[0, 2]), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
ss = np.arange(0, 15.01, 0.5)
Ps = np.stack([np.interp(ss, s0, P[:, k]) for k in range(3)], 1)
hu = samp(CT, Ainv, Ps)
info = {"case": CASE, "side": SIDE, "ext": EXT, "cca_HU": cca, "trace_start": P[0].tolist(),
        "hu_first_15mm": [[float(a), round(float(b))] for a, b in zip(ss, hu)]}
if own_sca is not None:
    v, aff = own_sca
    ij = np.argwhere(v)
    Wv = (aff[:3, :3] @ ij.T + aff[:3, 3:4]).T
    d = np.linalg.norm(Wv - P[0], axis=1)
    info["dist_trace_start_to_TS_own_SCA_mm"] = round(float(d.min()), 2)
    best = []
    for st in (0.0, 1.0, 2.0, 3.0):
        q = Ps[int(st / 0.5)]
        dd = np.linalg.norm(Wv - q, axis=1)
        for k in np.argsort(dd)[:400:20]:
            tgt = Wv[k]
            n = max(2, int(np.linalg.norm(tgt - q) / 0.25))
            line = q[None] + np.linspace(0, 1, n)[:, None] * (tgt - q)[None]
            h = samp(CT, Ainv, line)
            best.append((float(h.min()), float(np.linalg.norm(tgt - q)), st))
    best.sort(key=lambda t: -t[0])
    info["best_straight_link_to_own_SCA"] = {"min_HU_on_link": round(best[0][0]), "over_CCA": round(best[0][0] / cca, 3),
                                             "length_mm": round(best[0][1], 2), "from_trace_s_mm": best[0][2]}
fig.suptitle("%s %s %s VA origin (%s): CCA %.0f HU; overlays cyan Arterial mask, red TS aorta, magenta TS SCA, lime TS CCA, "
             "blue TS BT, orange BCV/SVC; trace yellow (0-40 mm) / red (0-5 mm); white = Arterial's own %sVA label" %
             (SRCN, CASE, SIDE, EXT, cca, SIDE), fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(OUT, "origin_%s_%s%s.png" % (CASE, SIDE, TAG))
fig.savefig(png, dpi=70)
json.dump(info, open(os.path.join(OUT, "origin_%s_%s%s.json" % (CASE, SIDE, TAG)), "w"), indent=1)
print(png, json.dumps({k: v for k, v in info.items() if k != "hu_first_15mm"}))
print("HU first 15 mm:", [h for _, h in info["hu_first_15mm"]])
