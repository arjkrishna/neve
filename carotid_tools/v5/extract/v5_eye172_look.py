#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_eye172_look.py (= /sp/v5_va/v5_eye172_look.py in the build containers)
"""v5_eye172_look.py <source> <case> <side> [ext] [qc_file] -- (v5_eye172, 2026-10-02) read-only by-eye look at one donor VA
trace: is it a VA from the subclavian origin through the C6..C1 foramina with the V3 loop, without carotid / vein shortcuts?
Native CT (trilinear), window -150 .. 1.6 x CCA. Left: two straightened CPRs along the trace (rotation-minimising frame,
+-14 mm), s up, landmarks from qc (C6 = Erec entry, C2, C1 = C1f, dura), overlays: Arterial native mask (cyan), TS 1.5 mm
vertebrae C1-C7 (yellow), TS CCA / SCA / BT (magenta), TS BCV / SVC (orange); below them the native HU / CCA along the
trace. Right: up to 18 AXIAL crops (+-24 mm, image left = patient right, anterior up) at origin, V1, C6 and evenly to C2,
C1, the loop and dura; trace point red +, trace within +-1.5 mm of the slice yellow, same overlays.
Writes /sp/v5_va/results/v5_eye172/look_<case>_<side>.png and .json (HU stats per segment). neve-build-stagea."""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRCN, CASE, SIDE = sys.argv[1], sys.argv[2], sys.argv[3]
EXT = sys.argv[4] if len(sys.argv) > 4 else "tsba"
QCF = sys.argv[5] if len(sys.argv) > 5 else "qc.json"
TAG = os.environ.get("LOOK_TAG", "")
SRC = "/repo/carotid_data/va_sources/thin_slice/%s/%s" % (SRCN, CASE)
DER = "/repo/carotid_data/va_sources/derived/thin_slice/%s/%s" % (SRCN, CASE)
OUT = "/sp/v5_va/results/v5_eye172"
os.makedirs(OUT, exist_ok=True)
Q = json.load(open(os.path.join(DER, "qc", QCF)))
qs = Q["sides"][SIDE]
cca = float(Q.get("cca_median_HU_native") or 400.0)
lm = qs.get("landmarks_used_mm") or {}
P = np.asarray(json.load(open(os.path.join(DER, "arterial/%s/va_extract.json" % EXT)))["sides"][SIDE]["trace"]["points"], float)
s0 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
ds = 0.5
S = np.arange(0.0, s0[-1], ds)
Pr = np.stack([np.interp(S, s0, P[:, k]) for k in range(3)], 1)
k = 6
T = np.gradient(Pr, axis=0)
Tk = np.copy(T)
for i in range(len(T)):
    a, b = max(0, i - k), min(len(T), i + k + 1)
    Tk[i] = T[a:b].mean(0)
T = Tk / np.maximum(np.linalg.norm(Tk, axis=1, keepdims=True), 1e-9)
N = np.zeros_like(T)
ref = np.array([1.0, 0.0, 0.0])
n0 = ref - T[0] * (ref @ T[0])
N[0] = n0 / np.linalg.norm(n0)
for i in range(1, len(T)):
    n = N[i - 1] - T[i] * (N[i - 1] @ T[i])
    nn = np.linalg.norm(n)
    N[i] = n / nn if nn > 1e-9 else N[i - 1]
B = np.cross(T, N)

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


masks = []
mp = os.path.join(DER, "arterial/runs/native/extracranial_vessels/segmentation.nii.gz")
if os.path.isfile(mp):
    m = nib.load(mp)
    masks.append(("Arterial mask", np.asanyarray(m.dataobj) > 0, np.linalg.inv(m.affine), "cyan"))
TSD = os.path.join(DER, "totalseg_1p5/segmentations")
groups = {"TS vertebrae": (["vertebrae_C%d" % i for i in range(1, 8)], "yellow"),
          "TS CCA/SCA/BT": (["common_carotid_artery_left", "common_carotid_artery_right", "subclavian_artery_left",
                             "subclavian_artery_right", "brachiocephalic_trunk"], "magenta"),
          "TS BCV/SVC": (["brachiocephalic_vein_left", "brachiocephalic_vein_right", "superior_vena_cava"], "orange")}
for gname, (names, col) in groups.items():
    acc, Ai = None, None
    for n in names:
        p = os.path.join(TSD, n + ".nii.gz")
        if os.path.isfile(p):
            im = nib.load(p)
            v = np.asanyarray(im.dataobj) > 0
            acc = v if acc is None else (acc | v)
            Ai = np.linalg.inv(im.affine)
    if acc is not None:
        masks.append((gname, acc, Ai, col))

W = 14.0
U = np.arange(-W, W + 0.01, 0.4)
lo, hi = -150.0, 1.6 * cca
smax = min(S[-1], (lm.get("dura") or S[-1]) + 15.0)
sel = S <= smax
fig = plt.figure(figsize=(26, 15))
gs = fig.add_gridspec(4, 8, width_ratios=[1.1, 1.1, 1, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 0.55])
for j, (nm, Vv) in enumerate((("N (lateral-ish)", N), ("B (AP-ish)", B))):
    G = Pr[sel][:, None, :] + U[None, :, None] * Vv[sel][:, None, :]
    img = samp(CT, Ainv, G)
    ax = fig.add_subplot(gs[0:3, j])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[U[0], U[-1], 0, S[sel][-1]], aspect="auto")
    for (mn, vol, Ai, col) in masks:
        mk = samp(vol, Ai, G, order=0).astype(float)
        if mk.any():
            ax.contour(U, S[sel], mk, levels=[0.5], colors=col, linewidths=0.7)
    for name, val in lm.items():
        if name in ("C6", "C2", "C1", "dura") and val is not None and val <= smax:
            ax.axhline(val, color="red", lw=0.6, ls="--")
            ax.text(U[0] + 0.5, val + 1, name, color="red", fontsize=8)
    ax.axvline(0, color="red", lw=0.3, alpha=0.4)
    ax.set_title("CPR along %s VA, plane %s" % (SIDE, nm), fontsize=9)
    ax.set_xlabel("mm off the trace")
    if j == 0:
        ax.set_ylabel("s from the trace start (mm)")
hu = samp(CT, Ainv, Pr[sel])
axh = fig.add_subplot(gs[3, 0:2])
axh.plot(S[sel], hu / cca, lw=0.8)
axh.axhline(1.0, color="k", lw=0.5)
axh.axhline(0.6, color="r", lw=0.5, ls=":")
for name, val in lm.items():
    if name in ("C6", "C2", "C1", "dura") and val is not None and val <= smax:
        axh.axvline(val, color="red", lw=0.6, ls="--")
axh.set_ylim(0, 2.2)
axh.set_xlabel("s (mm)")
axh.set_ylabel("HU / CCA (centreline)")
c6, c2, c1, du = (lm.get(kk) for kk in ("C6", "C2", "C1", "dura"))
lv = [1.0, 4.0, (c6 or 30) * 0.5, c6 or 30]
if c6 and c2:
    lv += list(np.linspace(c6, c2, 7)[1:-1])
lv += [c2 or 120]
if c2 and c1:
    lv += [c2 + 0.33 * (c1 - c2), c2 + 0.66 * (c1 - c2)]
lv += [c1 or 150]
if c1 and du:
    lv += [c1 + 0.33 * (du - c1), c1 + 0.66 * (du - c1)]
lv += [du or S[-1] - 5]
lv = [min(max(v, 0.5), S[-1] - 0.5) for v in lv][:18]
HW = 24.0
g1 = np.arange(-HW, HW + 0.01, 0.35)
for i, sv in enumerate(lv):
    r, c = divmod(i, 6)
    if r > 2:
        break
    ax = fig.add_subplot(gs[r, 2 + c])
    p = Pr[int(round(sv / ds))]
    XX, YY = np.meshgrid(-g1, g1)
    Gw = np.stack([p[0] + XX, p[1] + YY, np.full_like(XX, p[2])], -1)
    img = samp(CT, Ainv, Gw)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[-HW, HW, -HW, HW])
    for (mn, vol, Ai, col) in masks:
        mk = samp(vol, Ai, Gw, order=0).astype(float)
        if mk.any():
            ax.contour(g1, g1, mk, levels=[0.5], colors=col, linewidths=0.6)
    near = np.abs(Pr[:, 2] - p[2]) < 1.5
    if near.any():
        ax.plot(-(Pr[near, 0] - p[0]), Pr[near, 1] - p[1], ".", ms=1.5, color="yellow")
    ax.plot([0], [0], "+", color="red", ms=10, mew=1.2)
    ax.set_title("s %.0f mm, z %.0f, HU %.0f" % (sv, p[2], samp(CT, Ainv, p[None])[0]), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
info = {"case": CASE, "source": SRCN, "side": SIDE, "ext": EXT, "qc_file": QCF, "cca_HU": cca, "landmarks": lm,
        "length_mm": float(S[-1]), "levels_s": [float(v) for v in lv]}
seg = {}
for nm, a, b in (("origin->C6", 0, c6), ("C6->C2", c6, c2), ("C2->C1", c2, c1), ("C1->dura", c1, du)):
    if a is None or b is None:
        continue
    mm = (S[sel] >= a) & (S[sel] < b)
    if mm.any():
        h = hu[mm] / cca
        seg[nm] = {"median_HU_over_CCA": round(float(np.median(h)), 3), "p10": round(float(np.percentile(h, 10)), 3),
                   "min": round(float(h.min()), 3), "frac_lt_0.6": round(float((h < 0.6).mean()), 3)}
info["hu_segments"] = seg
fig.suptitle("%s %s %s VA (%s): CPRs + axial crops (image left = patient right, anterior up); cyan Arterial mask, yellow TS "
             "C1-C7, magenta TS CCA/SCA/BT, orange TS BCV/SVC; red + trace point; CCA %.0f HU; landmarks %s" %
             (SRCN, CASE, SIDE, EXT, cca, {kk: v for kk, v in lm.items()}), fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(OUT, "look_%s_%s%s.png" % (CASE, SIDE, TAG))
fig.savefig(png, dpi=72)
json.dump(info, open(os.path.join(OUT, "look_%s_%s%s.json" % (CASE, SIDE, TAG)), "w"), indent=1)
print(png, json.dumps(seg))
