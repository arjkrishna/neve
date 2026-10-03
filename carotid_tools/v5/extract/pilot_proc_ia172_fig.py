#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_fig.py (= /sp/v5_va/pilot_proc_ia172_fig.py in the build containers)
"""pilot_proc_ia172_fig.py <case> -- figures for one 172-series pilot case (neve-build-stagea: nibabel + matplotlib).

1. <der>/qc/qc_<case>.png: the QC picture of both VA traces.
   - coronal and sagittal MIPs of Arterial's binary vessel mask (native grid; grey) with the two tsba traces on top,
     coloured by QC status (green = passes as a v5 donor, orange = passes with review flags, red = fails); the stretch
     above the C1 landmark is drawn thin, bridged gaps as crosses, landmarks C6 / C2 / C1 / dura as dots;
   - a coronal CT MIP (HU window 0..700) of the posterior neck slab (y between the traces' V2 extent +-8 mm), with the same
     traces, so foramina and veins can be seen beside the trace.
2. <der>/qc/<case>_wall3d_<side>.png for every side with a ct_seg4 wall: the render of pilot_proc_aortaseg_fig.render
   (the AortaSeg pilot agent's, logic copied): the stage-A wall coloured by dep = |x - c| - r (c, r = the owning VMTK MISR
   sample among the 64 nearest, the v4 union's ownership rule), three views + the unrolled map dep(s, theta), landmarks
   from ct_seg landmarks.json ("rec" set, else "ts").
3. <der>/qc/<case>_detail_along_va.png: union > 0.3 mm per 10 mm bin (measure.json bins10) against VMR 0248 and TaG-Net A
   in the same harness (results/tag_ctwall_measure.json, ct_seg4 stage-A, exact ownership).
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else os.environ["PILOT_CASE"]
SRC = "/repo/carotid_data/va_sources/thin_slice/ia172/%s" % CASE
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172/%s" % CASE
TAG = "ct_seg4"
OUT = os.path.join(DER, "qc")
os.makedirs(OUT, exist_ok=True)
COL = {"donor": "tab:green", "donor_review": "tab:orange", "fail": "tab:red"}


def w2i(A, P):
    return (np.linalg.inv(A) @ np.c_[P, np.ones(len(P))].T)[:3].T


def qc_png():
    qc = json.load(open(os.path.join(DER, "qc/qc.json")))
    ex = json.load(open(os.path.join(DER, "arterial/tsba/va_extract.json")))
    segp = os.path.join(DER, "arterial/runs/native/extracranial_vessels/segmentation.nii.gz")
    im = nib.load(segp)
    A = im.affine
    m = np.asanyarray(im.dataobj) > 0
    sd = np.diag(A)[:3]          # signed: Arterial writes LAS (x index runs to -x)
    org = A[:3, 3]
    ext_x = [org[0] - 0.5 * sd[0], org[0] + sd[0] * (m.shape[0] - 0.5)]
    ext_y = [org[1] - 0.5 * sd[1], org[1] + sd[1] * (m.shape[1] - 0.5)]
    ext_z = [org[2] - 0.5 * sd[2], org[2] + sd[2] * (m.shape[2] - 0.5)]
    cor = m.max(1).T.astype(float)     # [z, x]
    sag = m.max(0).T.astype(float)     # [z, y]
    traces = {}
    for side in ("R", "L"):
        tr = ex["sides"][side].get("trace", {})
        if tr.get("ok"):
            traces[side] = np.asarray(tr["points"], float)
    # CT slab (posterior neck): y range of the traces below the C1 landmark
    ct = nib.load(os.path.join(SRC, "ct.nii.gz"))
    Ac = ct.affine
    ys, zs = [], []
    for side, P in traces.items():
        lm = qc["sides"][side].get("landmarks_used_mm") or {}
        s = vm.arclen(P)
        c1 = lm.get("C1") or s[-1]
        ys.append(P[s <= c1, 1])
        zs.append(P[:, 2])
    fig, axs = plt.subplots(1, 3, figsize=(21, 10))
    axs[0].imshow(cor, cmap="Greys", origin="lower", extent=[ext_x[0], ext_x[1], ext_z[0], ext_z[1]], aspect="equal", alpha=0.6)
    axs[1].imshow(sag, cmap="Greys", origin="lower", extent=[ext_y[0], ext_y[1], ext_z[0], ext_z[1]], aspect="equal", alpha=0.6)
    axs[0].set_xlim(min(ext_x), max(ext_x))      # ascending x in every panel (Arterial's LAS grid would flip it)
    axs[1].set_xlim(min(ext_y), max(ext_y))
    if ys:
        y0, y1 = np.min(np.concatenate(ys)) - 8, np.max(np.concatenate(ys)) + 8
        j0 = int(max(0, np.floor((y0 - Ac[1, 3]) / Ac[1, 1])))
        j1 = int(min(ct.shape[1] - 1, np.ceil((y1 - Ac[1, 3]) / Ac[1, 1])))
        z0 = min(np.min(np.concatenate(zs)) - 10, ext_z[1])
        k0 = int(max(0, np.floor((z0 - Ac[2, 3]) / Ac[2, 2])))
        slab = np.asarray(ct.dataobj[:, j0:j1 + 1, k0:]).astype(np.float32)
        cm = np.clip(slab, 0, 700).max(1).T
        axs[2].imshow(cm, cmap="gray", origin="lower", aspect="equal",
                      extent=[Ac[0, 3], Ac[0, 3] + Ac[0, 0] * ct.shape[0], Ac[2, 3] + Ac[2, 2] * k0,
                              Ac[2, 3] + Ac[2, 2] * ct.shape[2]])
        axs[2].set_title("CT MIP, HU 0..700, posterior slab y %.0f..%.0f mm" % (y0, y1), fontsize=10)
    for side, P in traces.items():
        q = qc["sides"][side]
        st = q.get("status", "fail")
        c = COL.get(st, "tab:red")
        lm = q.get("landmarks_used_mm") or {}
        s = vm.arclen(P)
        c1 = lm.get("C1") or s[-1]
        lo, hi = s <= c1, s >= c1
        for ax, (i, j) in zip(axs, ((0, 2), (1, 2), (0, 2))):
            ax.plot(P[lo, i], P[lo, j], color=c, lw=2.2)
            ax.plot(P[hi, i], P[hi, j], color=c, lw=0.9, ls="--")
            for k, key in enumerate(("C6", "C2", "C1", "dura")):
                if lm.get(key) is not None:
                    t = int(np.argmin(np.abs(s - lm[key])))
                    ax.plot(P[t, i], P[t, j], "o", color="k", ms=4)
                    if ax is axs[0]:
                        ax.text(P[t, i] + (4 if side == "R" else -14), P[t, j], key, fontsize=7)
            for g in q.get("bridges", []):
                t = int(np.argmin(np.abs(s - g["s_mm"])))
                ax.plot(P[t, i], P[t, j], "x", color="m", ms=8, mew=2)
        lab = "%s: %s%s" % (side, st, "" if q.get("passes_as_v5_donor") else " (%s)" % "; ".join(q.get("gate_fail_reasons", []))[:90])
        axs[0].plot([], [], color=c, lw=2.2, label=lab)
    axs[0].legend(fontsize=8, loc="upper center")
    axs[0].set_title("coronal MIP of Arterial's vessel mask (x: RAS, + = right)", fontsize=10)
    axs[1].set_title("sagittal MIP of the mask (y: + = anterior)", fontsize=10)
    for ax in axs:
        if zs:
            ax.set_ylim(min(np.min(np.concatenate(zs)) - 15, ext_z[1]), max(np.max(np.concatenate(zs)) + 15, ext_z[0]))
    rows = []
    for side in ("R", "L"):
        q = qc["sides"][side]
        if not q.get("trace_ok"):
            rows.append("%s no trace" % side)
            continue
        v3 = q.get("v3loop_robust") or {}
        rows.append("%s: L %.0f mm, entry %s (ts_qc %s), C6..C1 %s, origin %.1f mm, V3 robust %s / %s mm, HU/CCA %s, review: %s" % (
            side, q.get("length_trace_mm") or -1, q.get("entry_level_Erec"), q.get("entry_level_ts_qc"),
            q.get("n_C6_to_C1_passed"), -1 if q.get("junction_to_TS_SA_mm") is None else q["junction_to_TS_SA_mm"], v3.get("posterior_excursion_mm"),
            v3.get("d_C1_posterior_half_mm"), (q.get("trace_HU_native_origin_to_C1") or {}).get("ratio_to_CCA"),
            "; ".join(q.get("review", []))[:120]))
    fig.suptitle("%s: VA traces (art_extract + basilar fix on Arterial's native mask) coloured by QC status; thick = origin -> C1, "
                 "dashed = above C1; x = bridged gap\n%s" % (CASE, "\n".join(rows)), fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "qc_%s.png" % CASE)
    fig.savefig(p, dpi=80)
    plt.close(fig)
    print("wrote", p, flush=True)


def dep_on_surface(P, R, pts):
    """pilot_proc_aortaseg_fig.dep_on_surface, verbatim."""
    Pr, Rr, s = vm.resample(P, R)
    dd, ii = cKDTree(Pr).query(pts, k=min(64, len(Pr)))
    best = ii[np.arange(len(ii)), np.argmax(Rr[ii] - dd, axis=1)]
    v = pts - Pr[best]
    dep = np.linalg.norm(v, axis=1) - Rr[best]
    T = np.gradient(Pr, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    a = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    U = [np.cross(T[0], a) / np.linalg.norm(np.cross(T[0], a))]
    for i in range(1, len(Pr)):
        u = U[-1] - (U[-1] @ T[i]) * T[i]
        U.append(u / np.linalg.norm(u))
    U = np.array(U)
    W = np.cross(T, U)
    th = np.degrees(np.arctan2(np.einsum("ij,ij->i", v, W[best]), np.einsum("ij,ij->i", v, U[best])))
    return dep, s[best], th, Pr


def render(side, v, voxel):
    """pilot_proc_aortaseg_fig.render, logic copied (paths, landmarks and the title's voxel adapted)."""
    import pyvista as pv
    D = os.path.join(DER, "ct_seg", TAG)
    sf, cl = os.path.join(D, "%s_stageA.vtp" % v), os.path.join(D, "%s_stageA_vmtk.json" % v)
    if not (os.path.isfile(sf) and os.path.isfile(cl)):
        return None
    surf = pv.read(sf).triangulate()
    d = json.load(open(cl))
    P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
    pts = np.asarray(surf.points)
    dep, sv, th, Pr = dep_on_surface(P, R, pts)
    faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
    fd = dep[faces].mean(1)
    lm = {}
    lp = os.path.join(D, "landmarks.json")
    if os.path.isfile(lp):
        r = json.load(open(lp))["sides"].get("right" if side == "R" else "left", {})
        lm = r.get("landmarks_rec") or r.get("landmarks_used") or {}
    fig = plt.figure(figsize=(20, 11))
    cmap = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    cols = cmap(norm(fd))
    ctr = pts.mean(0)
    span = (pts.max(0) - pts.min(0)).max() / 2
    for k, (title, elev, azim) in enumerate((("anterior", 0, -90), ("right lateral", 0, 0), ("posterior", 0, 90))):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        pc = Poly3DCollection(pts[faces], facecolors=cols, edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title("%s VA wall, %s view" % (side, title), fontsize=9)
        for key, c in (("C6", "k"), ("C2", "k"), ("C1", "k"), ("dura", "g")):
            if lm.get(key) is not None:
                j = int(np.argmin(np.abs(vm.arclen(Pr) - lm[key])))
                ax.text(Pr[j, 0], Pr[j, 1], Pr[j, 2], "  " + key, color=c, fontsize=8)
    ax = fig.add_subplot(1, 4, 4)
    sc = ax.scatter(th, sv, c=dep, cmap=cmap, norm=norm, s=1.5, linewidths=0)
    for key in ("C6", "C2", "C1", "dura"):
        if lm.get(key) is not None:
            ax.axhline(lm[key], color="k", lw=0.8, ls="--")
            ax.text(-180, lm[key], key, fontsize=8, va="bottom")
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, sv.max())
    ax.set_xlabel("angle around the centerline (deg, RMF frame)")
    ax.set_ylabel("s from origin (mm)")
    ax.set_title("unrolled wall: dep = |x - c| - r", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05)
    cb.set_label("|x - c| - r (mm); red = outward of the MISR tube")
    fig.suptitle("%s %s VA: CT wall (%s stage-A, acquisition voxel %.3f x %.3f x %.2f mm), coloured by |x - c| - r "
                 "(c, r = owning VMTK MISR sample, k = 64). %d triangles; dep p5/p50/p95 = %.2f / %.2f / %.2f mm; "
                 "share |dep| > 0.3 mm = %.3f" % (CASE, side, TAG, voxel[0], voxel[1], voxel[2], len(faces),
                                                  *np.percentile(dep, [5, 50, 95]), float(np.mean(np.abs(dep) > 0.3))),
                 fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "%s_wall3d_%s.png" % (CASE, side))
    fig.savefig(p, dpi=80)
    plt.close(fig)
    print("wrote", p, flush=True)
    return p


def along(voxel):
    mp = os.path.join(DER, "ct_seg", TAG, "measure.json")
    if not os.path.isfile(mp):
        return None
    M = json.load(open(mp))
    full = M.get("full", {})
    ref = json.load(open("/sp/v5_va/results/tag_ctwall_measure.json"))["full"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    for ax, (side, sd) in zip(axs, (("R", "right"), ("L", "left"))):
        for src, rep, c, lab in (("TaG-A", "ct_seg4_stageA", "tab:blue", "TaG-Net A (0.405 mm)"),
                                 ("VMR", "ct_seg4_stageA", "tab:green", "VMR 0248 (0.40 x 0.70 mm)")):
            r = ref.get(src, {}).get(rep, {}).get("k64", {}).get(sd)
            if r:
                b = r["bins10"]
                ax.step([x["from_mm"] for x in b] + [b[-1]["to_mm"]], [x["frac_0.3mm"] for x in b] + [b[-1]["frac_0.3mm"]],
                        where="post", color=c, lw=1.0, alpha=0.8, label=lab)
        r = full.get(sd)
        if r:
            b = r["bins10"]
            ax.step([x["from_mm"] for x in b] + [b[-1]["to_mm"]], [x["frac_0.3mm"] for x in b] + [b[-1]["frac_0.3mm"]],
                    where="post", color="tab:red", lw=2.0, label="%s (%.2f x %.2f mm)" % (CASE, voxel[0], voxel[2]))
            lm = r["provenance"].get("landmarks_primary") or {}
            for k in ("C6", "C2", "C1", "dura"):
                if lm.get(k) is not None:
                    ax.axvline(lm[k], color="tab:red", lw=0.7, ls="--")
                    ax.text(lm[k], 0.62, k, color="tab:red", fontsize=8)
        ax.axhline(0.0966, color="k", lw=0.8, ls=":")
        ax.text(1, 0.1, "W1 set bar 0.0966", fontsize=7)
        ax.axhline(0.01, color="grey", lw=0.6, ls=":")
        ax.set_title("%s VA: union > 0.3 mm per 10 mm (exact ownership)" % side)
        ax.set_xlabel("s from the origin (mm)")
        ax.set_ylim(0, 0.65)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("union area fraction > 0.3 mm off the MISR tube")
    fig.tight_layout()
    p = os.path.join(OUT, "%s_detail_along_va.png" % CASE)
    fig.savefig(p, dpi=80)
    plt.close(fig)
    print("wrote", p)
    return p


if __name__ == "__main__":
    voxel = json.load(open(os.path.join(SRC, "meta.json")))["acquisition_voxel_mm"]
    try:
        qc_png()
    except Exception as e:  # noqa: BLE001
        print("qc png failed:", repr(e))
    for side, v in (("R", "rva"), ("L", "lva")):
        try:
            render(side, v, voxel)
        except Exception as e:  # noqa: BLE001
            print("render", side, "failed:", repr(e))
    try:
        along(voxel)
    except Exception as e:  # noqa: BLE001
        print("along failed:", repr(e))
