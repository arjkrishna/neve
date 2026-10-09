#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tbo\va_narrow\ct_check.py (= /sp/v5_va/tbo/va_narrow/ct_check.py in the build containers)
"""va_narrow: CT evidence at the two donor narrow points (neve-build-stagea: nibabel + pyvista + matplotlib).

For w_025_right__topcow_mr_021_L LVA (Song AHMU1218113 R, x-mirrored) and w_027_right__topcow_mr_022 RVA (ia172_192 R):
 1. the anatomy's VA centreline (TA frame) within +-15 mm of the OBJ minimum is mapped back into the donor's native RAS
    frame by the inverse of the bake's transform (sdf_union.load_surface: x = (p*m - origin) @ R.T + anchor);
 2. per point (1 mm): the donor's own records -- ct_seg4 station (nearest; r_e of its 64 rays as polygon r_eq, min / max,
    ray rule kinds, I_c, I_bg, open_bright flagged run, fix-8 run), the donor MISR / CT calibre radii;
 3. an INDEPENDENT CT read on the shipped ct.nii.gz: 64 rays in the normal plane from the brightest point within 1 mm,
    half-max between the core (median within 0.5 mm) and the soft-tissue ring (median 3-5 mm, bone > 500 HU and air
    excluded) -> FWHM polygon r_eq, min ray, peak HU, rays with bone (> 700 HU) within 1.5 mm outside the edge;
 4. cross-section r_eq of: the ct_seg4 label, Arterial's tsba VA mask, Arterial's full segmentation, the donor stage-A
    surface (native), and the shipped OBJ / iso union mapped into the donor frame;
 5. a figure: radius profiles, 5 CT cross-sections (s0 -8, -4, 0, +4, +8 mm) with contours, two straightened views.
    python3 ct_check.py  -> /sp/v5_va/tbo/va_narrow/ct_<key>.json, ct_<key>.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import vtk  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, "/sp/v4_build/audit")
import topo_audit as TA  # noqa: E402

OUT = "/sp/v5_va/tbo/va_narrow"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/"
SRC = "/repo/carotid_data/va_sources/thin_slice/"
CASES = [
    {"key": "w025_LVA", "name": "case_w_025_right__topcow_mr_021_L", "side": "LVA", "xf": "vaC_L",
     "donor": "song/AHMU1218113", "dside": "R", "vessel": "rva", "label": 1},
    {"key": "w027_RVA", "name": "case_w_027_right__topcow_mr_022", "side": "RVA", "xf": "vaC_R",
     "donor": "ia172/ia172_192", "dside": "R", "vessel": "rva", "label": 1},
]
HALF = 15.0


def unit(v):
    return v / np.linalg.norm(v)


def frame(t):
    e1 = unit(np.cross(t, [0, 0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0, 0]))
    e2 = np.cross(t, e1)
    return e1, e2


def inv_xf(X, xf):
    R, o, a, m = (np.asarray(xf[k], float) for k in ("R", "origin", "anchor", "mirror"))
    return (((np.asarray(X, float) - a) @ R) + o) * m


class Vol:
    """Trilinear / nearest sampling of a NIfTI-like array in world (RAS) mm."""

    def __init__(self, arr, A, lo=None, hi=None, pad=12.0, post=None):
        self.A = np.asarray(A, float)
        Ai = np.linalg.inv(self.A)
        if lo is not None:
            cs = np.array([[x, y, z] for x in (lo[0] - pad, hi[0] + pad) for y in (lo[1] - pad, hi[1] + pad)
                           for z in (lo[2] - pad, hi[2] + pad)])
            ijk = cs @ Ai[:3, :3].T + Ai[:3, 3]
            a = np.maximum(np.floor(ijk.min(0)).astype(int), 0)
            b = np.minimum(np.ceil(ijk.max(0)).astype(int) + 1, np.array(arr.shape[:3]))
        else:
            a, b = np.zeros(3, int), np.array(arr.shape[:3])
        self.a = a
        blk = np.asarray(arr[a[0]:b[0], a[1]:b[1], a[2]:b[2]])
        self.arr = np.asarray(post(blk) if post else blk, np.float32)
        self.Ai = Ai

    def __call__(self, P, order=1):
        P = np.asarray(P, float)
        sh = P.shape[:-1]
        ijk = P.reshape(-1, 3) @ self.Ai[:3, :3].T + self.Ai[:3, 3] - self.a
        v = ndimage.map_coordinates(self.arr, ijk.T, order=order, mode="constant", cval=np.nan)
        return v.reshape(sh)


def kji_vol(arr_kji, origin, spacing, lo, hi):
    A = np.diag(list(spacing) + [1.0])
    A[:3, 3] = origin
    return Vol(np.ascontiguousarray(np.transpose(arr_kji, (2, 1, 0))), A, lo, hi)


def poly_req(r, n=64):
    th = 2 * np.pi / n
    return float(np.sqrt(max(0.5 * np.sum(r * np.roll(r, -1)) * np.sin(th), 0) / np.pi))


def plane_grid(c, e1, e2, half=7.0, h=0.1):
    u = np.arange(-half, half + 1e-9, h)
    U, V = np.meshgrid(u, u, indexing="xy")
    return u, c + U[..., None] * e1 + V[..., None] * e2


def comp_req(mask2d, h):
    """r_eq of the connected component at (or nearest to) the grid centre."""
    if not mask2d.any():
        return 0.0
    lab, n = ndimage.label(mask2d)
    cy, cx = mask2d.shape[0] // 2, mask2d.shape[1] // 2
    k = lab[cy, cx]
    if k == 0:
        ii = np.argwhere(lab > 0)
        j = np.argmin(((ii - [cy, cx]) ** 2).sum(1))
        if np.sqrt(((ii[j] - [cy, cx]) ** 2).sum()) * h > 1.5:
            return 0.0
        k = lab[tuple(ii[j])]
    return float(np.sqrt((lab == k).sum() * h * h / np.pi))


def section_xy(mesh, c, t, e1, e2):
    pl = vtk.vtkPlane()
    pl.SetOrigin(*c)
    pl.SetNormal(*t)
    cut = vtk.vtkCutter()
    cut.SetCutFunction(pl)
    cut.SetInputData(mesh)
    st = vtk.vtkStripper()
    st.SetInputConnection(cut.GetOutputPort())
    st.JoinContiguousSegmentsOn()
    st.Update()
    out = pv.wrap(st.GetOutput())
    loops = []
    if out.n_points == 0:
        return loops
    pts = np.asarray(out.points)
    L = np.asarray(out.lines)
    i = 0
    while i < len(L):
        n = L[i]
        q = pts[L[i + 1:i + 1 + n]] - c
        i += n + 1
        xy = np.c_[q @ e1, q @ e2]
        if len(xy) >= 4 and np.linalg.norm(xy, axis=1).min() < 6.0:
            loops.append(xy)
    return loops


def loop_req(loops):
    """r_eq of the loop containing the origin (else nearest)."""
    best = None
    for xy in loops:
        x, y = xy[:, 0], xy[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = ((y > 0) != (y2 > 0)) & (0 < (x2 - x) * (0 - y) / np.where(y2 - y == 0, 1e-30, y2 - y) + x)
        inside = bool(cr.sum() % 2)
        area = 0.5 * abs(np.sum(x * y2 - x2 * y))
        key = (0 if inside else 1, float(np.linalg.norm(xy, axis=1).min()))
        if best is None or key < best[0]:
            best = (key, float(np.sqrt(area / np.pi)))
    return best[1] if best else None


def own_ct(ct, c, t, e1, e2):
    """Independent half-max read of the CT at c (normal plane)."""
    # brightest point within 1 mm (gaussian-smoothed in plane)
    u, G = plane_grid(c, e1, e2, half=1.0, h=0.1)
    hu = ct(G)
    hu_s = ndimage.gaussian_filter(np.nan_to_num(hu, nan=-1000), 2.0)
    k = np.unravel_index(np.argmax(hu_s), hu_s.shape)
    c2 = G[k]
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rr = np.arange(0.0, 6.0001, 0.05)
    dirs = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    P = c2 + rr[None, :, None] * dirs[:, None, :]
    prof = ct(P)                                   # 64 x len(rr)
    core = prof[:, rr <= 0.5]
    Ic = float(np.nanmedian(core))
    ring = prof[:, (rr >= 3.0) & (rr <= 5.0)]
    ring = ring[(ring > -200) & (ring < 500)]
    Ib = float(np.nanmedian(ring)) if ring.size else 0.0
    half = 0.5 * (Ic + Ib)
    r_e = np.full(64, np.nan)
    bone = np.zeros(64, bool)
    for j in range(64):
        p = prof[j]
        below = np.where(p < half)[0]
        below = below[below > 0]
        if len(below):
            m = below[0]
            f = (p[m - 1] - half) / max(p[m - 1] - p[m], 1e-6)
            r_e[j] = rr[m - 1] + f * 0.05
            w = (rr > r_e[j]) & (rr <= r_e[j] + 1.5)
            bone[j] = bool(np.nanmax(p[w]) > 700) if w.any() else False
        else:
            r_e[j] = rr[-1]
    return {"centre_shift_mm": float(np.linalg.norm(c2 - c)), "Ic": Ic, "Ib": Ib, "peak_HU": float(np.nanmax(core)),
            "r_eq_fwhm": poly_req(r_e), "r_min": float(np.nanmin(r_e)), "r_max": float(np.nanmax(r_e)),
            "d_min": float(np.nanmin(r_e + np.roll(r_e, 32))), "bone_rays": int(bone.sum())}


def in_runs(s, runs):
    return any(a <= s <= b for a, b in runs)


def run(case):
    key, name = case["key"], case["name"]
    A5 = "/repo/carotid_data/anatomies_v5/" + name
    prov = json.load(open(A5 + "/provenance.json"))
    xf = prov["xform"][case["xf"]]
    prof = json.load(open(os.path.join(OUT, "profile_%s_%s_v5.json" % (name, case["side"]))))
    br = TA.load_branches(A5 + "/Centrelines_comb")
    p = [b[1] for b in br if b[0] == case["side"]][0]
    s = TA.arclen(p)
    s0 = prof["min_obj"]["s"]
    sel = np.where(np.abs(s - s0) <= HALF)[0]
    q = inv_xf(p[sel], xf)                       # donor native RAS
    # donor record
    D = DER + case["donor"]
    rec = json.load(open("/repo/carotid_data/va_sources/v5_reg/donors_v5/%s/%s.json" % (
        case["donor"].split("/")[-1] if "ia172" in case["donor"] else "song_" + case["donor"].split("/")[-1],
        case["dside"])))
    dc = np.asarray(rec["centerline"]["points"], float)
    ds = TA.arclen(dc)
    misr = np.asarray(rec["centerline"]["radii_misr"], float)
    cal = np.asarray(rec["centerline"]["radii_calibre"], float) if rec["centerline"].get("radii_calibre") else None
    # stations
    St = json.load(open(D + "/ct_seg/ct_seg4/stations_rays.json"))[case["vessel"]]
    sP, sS = np.asarray(St["P"], float), np.asarray(St["s"], float)
    sq = json.load(open(D + "/ct_seg/ct_seg4/seg_qc.json"))
    ob = sq["open_bright"][case["vessel"]]
    flagged = ob.get("flagged_s_ranges_mm", [])
    fix8 = (ob.get("keep_connected_fix8") or {}).get("runs_not_opened_s_mm", [])
    lo, hi = q.min(0), q.max(0)
    # volumes
    ctp = SRC + case["donor"] + "/ct.nii.gz"
    im = nib.load(ctp)
    ct = Vol(im.dataobj, im.affine, lo, hi)
    z = np.load(D + "/ct_seg/ct_seg4/labels_kji.npz")
    labv = kji_vol((z["labels"] == case["label"]).astype(np.float32), z["origin_mm"], z["spacing_mm"], lo, hi)
    m = np.load(D + "/arterial/tsba/%s_mask.npz" % case["vessel"])
    artv = Vol(m["mask"].astype(np.float32), m["affine"], lo, hi)
    sim = nib.load(D + "/arterial/runs/native/extracranial_vessels/segmentation.nii.gz")
    segv = Vol(sim.dataobj, sim.affine, lo, hi, post=lambda x: x > 0)
    surf = pv.read(rec["surface"]["path"]).triangulate()
    obj_V, obj_F, _ = TA.read_obj(A5 + "/vessel_architecture_collision.obj")
    objm = pv.PolyData(inv_xf(obj_V, xf), np.c_[np.full(len(obj_F), 3), obj_F])
    iso = pv.read("/repo/carotid_data/va_sources/v5_reg/iso_v5/%s.vtp" % name)
    iso = iso.extract_points(np.linalg.norm(np.asarray(iso.points) - p[sel].mean(0), axis=1) < 40.0,
                             adjacent_cells=True).extract_surface()
    isom = pv.PolyData(inv_xf(np.asarray(iso.points), xf), iso.faces)
    rows = []
    for j, i in enumerate(sel):
        c = q[j]
        a, b = max(j - 2, 0), min(j + 2, len(q) - 1)
        t = unit(q[b] - q[a])
        e1, e2 = frame(t)
        k = int(np.argmin(np.linalg.norm(sP - c, axis=1)))
        kd = int(np.argmin(np.linalg.norm(dc - c, axis=1)))
        re = np.asarray(St["r_e"][k], float)
        kinds = np.asarray(St["kind"][k], int)
        u, G = plane_grid(c, e1, e2, half=5.0, h=0.1)
        row = {"s": float(s[i]), "s_from_min": float(s[i] - s0), "r_cl": float(prof["r_cl"][i]),
               "d_obj": float(prof["d_obj"][i]), "d_full": float(prof["d_full"][i]), "d_iso": float(prof["d_iso"][i]),
               "donor_s": float(ds[kd]), "donor_cl_dist": float(np.linalg.norm(dc[kd] - c)),
               "misr": float(misr[kd]), "calibre": float(cal[kd]) if cal is not None else None,
               "station_s": float(sS[k]), "station_dist": float(np.linalg.norm(sP[k] - c)),
               "st_r_eq": poly_req(re), "st_r_min": float(re.min()), "st_d_min": float((re + np.roll(re, 32)).min()),
               "st_r_max": float(re.max()), "st_kinds": {str(v): int((kinds == v).sum()) for v in np.unique(kinds)},
               "st_Ic": float(St["Ic"][k]), "st_Ib": float(St["Ib"][k]),
               "open_bright_flagged": in_runs(float(sS[k]), flagged), "fix8_not_opened": in_runs(float(sS[k]), fix8),
               "own": own_ct(ct, c, t, e1, e2),
               "label_r_eq": comp_req(labv(G, 0) > 0.5, 0.1), "arterial_va_r_eq": comp_req(artv(G, 0) > 0.5, 0.1),
               "arterial_seg_r_eq": comp_req(segv(G, 0) > 0.5, 0.1),
               "stageA_r_eq": loop_req(section_xy(surf, c, t, e1, e2)),
               "obj_r_eq": loop_req(section_xy(objm, c, t, e1, e2)), "iso_r_eq": loop_req(section_xy(isom, c, t, e1, e2))}
        rows.append(row)
    res = {"case": case, "s0": s0, "donor_ct": ctp, "ct_affine_diag": np.diag(im.affine)[:3].tolist(),
           "flagged_runs": flagged, "fix8_runs": fix8, "rows": rows,
           "donor_landmarks": rec.get("landmarks"), "contamination_zones": rec.get("contamination_zones")}
    json.dump(res, open(os.path.join(OUT, "ct_%s.json" % key), "w"), indent=1)
    figure(key, name, case, rows, q, s[sel], s0, ct, labv, artv, surf, objm, isom)
    r0 = min(rows, key=lambda r: abs(r["s_from_min"]))
    print(key, json.dumps(r0)[:2000], flush=True)


def figure(key, name, case, rows, q, ss, s0, ct, labv, artv, surf, objm, isom):
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.15, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, :])
    S = np.array([r["s_from_min"] for r in rows])

    def col(k, sub=None):
        return np.array([(r[k] if sub is None else r[k][sub]) if (r[k] if sub is None else r[k][sub]) is not None
                         else np.nan for r in rows], float)
    ax.plot(S, col("own", "r_eq_fwhm"), "k-", lw=2, label="CT half-max r_eq (own read, shipped CT)")
    ax.plot(S, col("st_r_eq"), "k--", lw=1.2, label="ct_seg4 station rays r_eq")
    ax.plot(S, col("stageA_r_eq"), "-", color="tab:red", lw=1.5, label="donor stage-A surface section r_eq")
    ax.plot(S, col("misr"), ":", color="tab:red", lw=1.5, label="donor MISR (VMTK)")
    ax.plot(S, col("label_r_eq"), "-", color="goldenrod", lw=1, label="ct_seg4 label section r_eq")
    ax.plot(S, col("arterial_va_r_eq"), "-", color="tab:cyan", lw=1, label="Arterial VA mask section r_eq")
    ax.plot(S, col("r_cl"), "-", color="tab:purple", lw=1, label="v5 centreline r (MISR floored 1.0)")
    ax.plot(S, col("iso_r_eq"), "-", color="tab:green", lw=1, alpha=0.6, label="iso_v5 union section r_eq")
    ax.plot(S, col("obj_r_eq"), "-", color="tab:green", lw=2, label="OBJ section r_eq")
    ax.plot(S, col("d_obj"), "o-", color="tab:green", ms=3, lw=0.8, label="OBJ wall clearance at centreline")
    ax.axhline(0.65, color="tab:green", ls=":", lw=1)
    ax.axhline(1.0, color="tab:purple", ls=":", lw=1)
    for r in rows:
        if r["open_bright_flagged"]:
            ax.axvspan(r["s_from_min"] - 0.5, r["s_from_min"] + 0.5, color="0.85", lw=0, zorder=0)
        if r["fix8_not_opened"]:
            ax.axvspan(r["s_from_min"] - 0.5, r["s_from_min"] + 0.5, color="orange", alpha=0.25, lw=0, zorder=0)
    ax.set_xlabel("mm from the OBJ minimum (anatomy s0 = %.1f mm; donor s = %.1f mm)" % (
        s0, min(rows, key=lambda r: abs(r["s_from_min"]))["donor_s"]))
    ax.set_ylabel("radius / clearance (mm)")
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_title("%s %s -- donor %s %s (grey: open-bright flagged stations, orange: fix-8 not opened)" % (
        name, case["side"], case["donor"], case["dside"]), fontsize=10)
    # cross-sections
    for col_i, off in enumerate((-8, -4, 0, 4, 8)):
        j = int(np.argmin(np.abs(ss - (s0 + off))))
        c = q[j]
        a, b = max(j - 2, 0), min(j + 2, len(q) - 1)
        t = unit(q[b] - q[a])
        e1, e2 = frame(t)
        u, G = plane_grid(c, e1, e2, half=6.0, h=0.05)
        axc = fig.add_subplot(gs[1, col_i])
        axc.imshow(ct(G), origin="lower", extent=[u[0], u[-1], u[0], u[-1]], cmap="gray", vmin=-100, vmax=800)
        axc.contour(u, u, labv(G, 0), [0.5], colors="gold", linewidths=1)
        axc.contour(u, u, artv(G, 0), [0.5], colors="cyan", linewidths=0.8, linestyles="dashed")
        for xy in section_xy(surf, c, t, e1, e2):
            axc.plot(*np.vstack([xy, xy[:1]]).T, "-", color="red", lw=1)
        for xy in section_xy(objm, c, t, e1, e2):
            axc.plot(*np.vstack([xy, xy[:1]]).T, "-", color="lime", lw=1.2)
        th = np.linspace(0, 2 * np.pi, 100)
        axc.plot(np.cos(th), np.sin(th), ":", color="violet", lw=1)
        axc.plot(0, 0, "+", color="violet")
        axc.set_title("%+d mm (s %.1f)" % (off, ss[j]), fontsize=9)
        axc.set_xlim(-6, 6)
        axc.set_ylim(-6, 6)
        axc.tick_params(labelsize=6)
    # straightened views
    j0 = int(np.argmin(np.abs(ss - s0)))
    t0 = unit(q[min(j0 + 2, len(q) - 1)] - q[max(j0 - 2, 0)])
    E1, E2 = frame(t0)
    w = np.arange(-6, 6.0001, 0.1)
    for vi, E in enumerate((E1, E2)):
        img, lab, art = [], [], []
        for j in range(len(q)):
            a, b = max(j - 2, 0), min(j + 2, len(q) - 1)
            t = unit(q[b] - q[a])
            e = unit(E - (E @ t) * t)
            P = q[j] + w[:, None] * e
            img.append(ct(P))
            lab.append(labv(P, 0))
            art.append(artv(P, 0))
        img, lab, art = np.array(img), np.array(lab), np.array(art)
        axs = fig.add_subplot(gs[2, vi * 2:vi * 2 + 2])
        ext = [w[0], w[-1], ss[0] - s0, ss[-1] - s0]
        axs.imshow(img, origin="lower", extent=ext, aspect="auto", cmap="gray", vmin=-100, vmax=800)
        axs.contour(w, ss - s0, lab, [0.5], colors="gold", linewidths=1)
        axs.contour(w, ss - s0, art, [0.5], colors="cyan", linewidths=0.8, linestyles="dashed")
        axs.set_xlabel("mm across (straightened, view %d)" % (vi + 1), fontsize=8)
        axs.set_ylabel("mm along from s0", fontsize=8)
        axs.tick_params(labelsize=7)
    axn = fig.add_subplot(gs[2, 4])
    axn.axis("off")
    axn.text(0, 0.95, "CT window -100..800 HU\nred: donor stage-A surface\ngold: ct_seg4 label\ncyan dashed: Arterial "
             "VA mask\nlime: shipped OBJ (mapped back)\nviolet dotted: 1.0 mm tube floor\n\nstraightened views: the VA "
             "runs\nbottom -> top (1 mm rows,\n1 point per anatomy mm)", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ct_%s.png" % key), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    for c in CASES:
        if len(sys.argv) > 1 and c["key"] not in sys.argv[1:]:
            continue
        run(c)
