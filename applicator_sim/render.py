"""HOST figures (local only; never uploaded).  Units mm.

    python render.py --tag R1_P2_S0
figs/<tag>_slices.png : axial / sagittal / coronal slices of the BT MRI through the BT flange with contours of
    the BT labels, B0 (undeformed preBT at the same pose, mapped by E_app) and the simulation.  The UTERUS
    contours (sealed) are drawn only once frozen/S1.json exists; before that only HR-CTV and the applicator.
figs/<tag>_body.png   : rest vs deformed body surface (preBT world mm), rod and canal, coloured by displacement.
"""
import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
import evaluate as ev  # noqa: E402

P = config.paths()


# ===================================================================================== stage 2 validation figures
# python render.py --validate BONE:V1_PB_M1_S0,V2_PB_M2_S0 HR:R1_P2_S0,V3_P2_M2_S0
#   figs/val_<FRAME>_slices.png : BT MRI resampled on two oblique planes through the REAL BT tandem (tandem-sagittal =
#       span(tandem axis, y_app); tandem-coronal = span(tandem axis, x_app)); solid = BT labels, dashed = model
#       (column 1 rigid-only baseline, then one column per simulation); cyan uterus, red HR-CTV, green vagina,
#       yellow = BT applicator label.  Uterus metrics in the titles come from validation/metrics_M1.json.
#   figs/val_3d.png : uterus surfaces in the applicator frame (tandem vertical): BT (grey), baseline, simulation.
STRUCT_COL = (("uterus", "uterus", "cyan"), ("HR-CTV", "hrctv", "red"), ("vagina", "vagina", "lime"))


def _model_masks(bt, frame, tag=None):
    al = json.load(open(P["out"] + "/validation/alignment.json"))["frames"][frame]
    R, t = np.array(al["R_BT_to_pre"]), np.array(al["t_BT_to_pre"])
    out = {}
    for nm, obj, _ in STRUCT_COL:
        p = (os.path.join(P["runs"], tag, "surf_%s.obj" % obj) if tag else None)
        if p is None or not os.path.exists(p):
            p = P["inputs"] + "/pre_%s.obj" % obj          # baseline, or a structure the run does not simulate
        V, F = geom.read_obj(p)
        out[nm] = ev.voxelize((V - t) @ R, F, bt.shape, bt.aff)
    return out


def _plane(bt, horiz, s_half=55.0, v_lo=-70.0, v_hi=90.0, px=0.5):
    s = np.arange(-s_half, s_half + 1e-9, px); v = np.arange(v_lo, v_hi + 1e-9, px)
    S, Vv = np.meshgrid(s, v)
    X = bt.F + S[..., None] * horiz + Vv[..., None] * bt.a
    ijk = ((X.reshape(-1, 3) - bt.aff[:3, 3]) @ np.linalg.inv(bt.aff[:3, :3]).T).T
    from scipy import ndimage as ndi

    def samp(vol, order):
        return ndi.map_coordinates(np.asarray(vol, np.float32), ijk, order=order, mode="constant").reshape(S.shape)
    return S, Vv, samp


def validate_figs(specs):
    from matplotlib.lines import Line2D
    os.makedirs(P["figs"], exist_ok=True)
    bt = ev.BT(); img = np.asarray(nib.load(P["data"] + "/BT_MRI.nii").dataobj).astype(np.float32)
    met = json.load(open(P["out"] + "/validation/metrics_M1.json"))["runs"]
    for spec in specs:
        frame, tags = spec.split(":"); tags = [t for t in tags.split(",") if t]
        cols = [("rigid-only baseline", None)] + [(t, t) for t in tags]
        masks = [_model_masks(bt, frame, t) for _, t in cols]
        planes = [("tandem-sagittal (axis, y_app)", bt.R[1]), ("tandem-coronal (axis, x_app)", bt.R[0])]
        fig, axs = plt.subplots(2, len(cols), figsize=(4.6 * len(cols), 11.5), squeeze=False)
        for r, (pname, hz) in enumerate(planes):
            S, Vv, samp = _plane(bt, hz); im = samp(img, 1)
            lo, hi = np.percentile(im[im > 0], [1, 99.5]) if (im > 0).any() else (0, 1)
            ref = {nm: samp(bt.lab[nm], 0) for nm, _, _ in STRUCT_COL}; appl = samp(bt.lab["applicator"], 0)
            for c, ((cname, tag), mk) in enumerate(zip(cols, masks)):
                ax = axs[r, c]
                ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[S.min(), S.max(), Vv.min(), Vv.max()])
                ax.contour(S, Vv, appl, [0.5], colors="yellow", linewidths=0.7)
                for nm, _, col in STRUCT_COL:
                    ax.contour(S, Vv, ref[nm], [0.5], colors=col, linewidths=1.3)
                    ax.contour(S, Vv, samp(mk[nm], 0), [0.5], colors=col, linewidths=1.3, linestyles="dashed")
                key = "%s@%s" % (tag, frame) if tag else "%s@%s" % (tags[0], frame)
                u = met.get(key, {}).get("structures", {}).get("uterus", {})
                uu = u.get("sim" if tag else "baseline")
                ttl = "%s\n%s" % (cname if tag else "rigid-only baseline (%s frame)" % frame, pname)
                if uu:
                    ttl += "\nuterus Dice %.2f  MSD %.1f  HD95 %.1f mm" % (uu["dice"], uu["msd"], uu["hd95"])
                if tag and key in met:
                    n = met[key]["numerics"]; ttl += "\n%s%s" % (n["status"], ", INADMISSIBLE" if n["admissibility_flag"] else "")
                ax.set_title(ttl, fontsize=8); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
                ax.set_xlabel("in-plane mm from BT flange", fontsize=7)
                if c == 0:
                    ax.set_ylabel("mm along BT tandem axis from flange", fontsize=7)
        hs = [Line2D([], [], color=col, lw=1.3, label=nm) for nm, _, col in STRUCT_COL]
        hs += [Line2D([], [], color="k", lw=1.3, label="solid = BT label"), Line2D([], [], color="k", lw=1.3, ls="--", label="dashed = model"),
               Line2D([], [], color="yellow", lw=0.8, label="BT applicator")]
        fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=8)
        fig.suptitle("%s frame: BT MRI with the BT applicator in place; model = preBT anatomy (rigid-only or simulated) "
                     "mapped by the %s rigid transform" % (frame, frame), fontsize=10)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fn = os.path.join(P["figs"], "val_%s_slices.png" % frame); fig.savefig(fn, dpi=100); plt.close(fig)
        print("wrote", fn)
    _fig3d(bt, specs)


def _fig3d(bt, specs):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    m = np.pad(bt.lab["uterus"], 1).astype(np.uint8)
    vv, ff, _, _ = measure.marching_cubes(m, 0.5, step_size=2)
    Vbt = (vv - 1.0) @ bt.aff[:3, :3].T + bt.aff[:3, 3]
    app = lambda X: (np.asarray(X) - bt.F) @ bt.R.T  # noqa: E731   BT world -> applicator frame (x, y, z=axis)
    panels = []
    for spec in specs:
        frame, tags = spec.split(":"); tags = [t for t in tags.split(",") if t]
        al = json.load(open(P["out"] + "/validation/alignment.json"))["frames"][frame]
        R, t = np.array(al["R_BT_to_pre"]), np.array(al["t_BT_to_pre"])
        V0, F0 = geom.read_obj(P["inputs"] + "/pre_uterus.obj")
        panels.append(("%s frame: rigid-only baseline" % frame, (V0 - t) @ R, F0, "tab:red"))
        for tg in tags:
            Vd, Fd = geom.read_obj(os.path.join(P["runs"], tg, "surf_uterus.obj"))
            panels.append(("%s frame: %s" % (frame, tg), (Vd - t) @ R, Fd, "tab:blue"))
    sph_c = np.array([q["center_app_mm"] for q in bt.sph]); sph_r = np.array([q["r_mm"] for q in bt.sph])
    nc = min(3, len(panels)); nr = int(np.ceil(len(panels) / nc))
    fig = plt.figure(figsize=(5.2 * nc, 5.2 * nr))
    uu, vv_ = np.mgrid[0:2 * np.pi:12j, 0:np.pi:7j]
    for i, (ttl, V, F, col) in enumerate(panels):
        ax = fig.add_subplot(nr, nc, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(app(Vbt)[ff], facecolor="0.6", edgecolor="none", alpha=0.25))
        ax.add_collection3d(Poly3DCollection(app(V)[F], facecolor=col, edgecolor="none", alpha=0.35))
        ax.plot([0, 0], [0, 0], [0, bt.L], color="k", lw=3)
        for c_, r_ in zip(sph_c, sph_r):
            ax.plot_wireframe(c_[0] + r_ * np.cos(uu) * np.sin(vv_), c_[1] + r_ * np.sin(uu) * np.sin(vv_),
                              c_[2] + r_ * np.cos(vv_), color="goldenrod", lw=0.3)
        ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_zlim(-30, 90); ax.set_box_aspect((90, 90, 120))
        ax.view_init(elev=8, azim=0); ax.set_title(ttl + "\ngrey = BT uterus, black = tandem, gold = ovoid spheres", fontsize=8)
        ax.set_xlabel("x_app"); ax.set_ylabel("y_app"); ax.set_zlabel("tandem axis (mm)")
    fig.tight_layout(); fn = os.path.join(P["figs"], "val_3d.png"); fig.savefig(fn, dpi=100); plt.close(fig)
    print("wrote", fn)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag"); ap.add_argument("--validate", nargs="*")
    a = ap.parse_args()
    if a.validate:
        validate_figs(a.validate)
        return
    os.makedirs(P["figs"], exist_ok=True)
    rd = os.path.join(P["runs"], a.tag); z = np.load(rd + "/final.npz")
    v = json.load(open(os.path.join(P["eval"], a.tag, "visible.json")))
    Re, te = np.array(v["E_app"]["R"]), np.array(v["E_app"]["t"])
    bt = ev.BT(); img = np.asarray(nib.load(P["data"] + "/BT_MRI.nii").dataobj).astype(float)
    unsealed = os.path.exists(P["frozen"] + "/S1.json")
    names = [("HR-CTV", "hrctv")] + ([("uterus", "uterus")] if unsealed else [])
    sim, b0 = {}, {}
    for nm, obj in names:
        Vd, Fc = geom.read_obj(rd + "/surf_%s.obj" % obj); V0, _ = geom.read_obj(P["inputs"] + "/pre_%s.obj" % obj)
        sim[nm] = ev.voxelize(ev.transform(Vd, Re, te), Fc, bt.shape, bt.aff)
        b0[nm] = ev.voxelize(ev.transform(V0, Re, te), Fc, bt.shape, bt.aff)
    c = np.rint((bt.F - bt.aff[:3, 3]) @ np.linalg.inv(bt.aff[:3, :3]).T).astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    views = [("axial k=%d" % c[2], lambda m: m[:, :, c[2]].T), ("coronal j=%d" % c[1], lambda m: m[:, c[1], :].T),
             ("sagittal i=%d" % c[0], lambda m: m[c[0], :, :].T)]
    asp = [bt.sp[1] / bt.sp[0], bt.sp[2] / bt.sp[0], bt.sp[2] / bt.sp[1]]
    for ax, (ttl, sl), asp_ in zip(axs, views, asp):
        ax.imshow(sl(img), cmap="gray", origin="lower", aspect=asp_)
        ax.contour(sl(bt.lab["applicator"]).astype(float), [0.5], colors="yellow", linewidths=0.8)
        for nm, col in (("HR-CTV", "red"), ("uterus", "cyan")):
            if nm in sim:
                ax.contour(sl(bt.lab[nm]).astype(float), [0.5], colors=col, linewidths=1.2)
                ax.contour(sl(b0[nm]).astype(float), [0.5], colors=col, linewidths=0.8, linestyles="dotted")
                ax.contour(sl(sim[nm]).astype(float), [0.5], colors=col, linewidths=1.0, linestyles="dashed")
        ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("%s on BT MRI through the BT flange: solid = BT label, dotted = B0 (undeformed, E_app), dashed = sim; "
                 "yellow = applicator; %s" % (a.tag, "uterus shown (S1 frozen)" if unsealed else "uterus SEALED (not drawn)"),
                 fontsize=9)
    fig.savefig(os.path.join(P["figs"], a.tag + "_slices.png"), dpi=110); plt.close(fig)
    # rest vs deformed body
    V0, Fb = geom.read_obj(P["inputs"] + "/pre_body.obj"); V1, _ = geom.read_obj(rd + "/surf_body.obj")
    u = np.linalg.norm(V1 - V0, axis=1); F, al = z["F"], geom.unit(z["a"]); rod = F + np.outer([0, float(z["L_iu"])], al)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (i, j, lab) in zip(axs, [(0, 2, "coronal (x-z)"), (1, 2, "sagittal (y-z)"), (0, 1, "axial (x-y)")]):
        ax.scatter(V0[:, i], V0[:, j], s=1, c="0.8")
        sc = ax.scatter(V1[:, i], V1[:, j], s=1.5, c=u, cmap="viridis", vmin=0, vmax=max(1e-6, u.max()))
        ax.plot(z["canal_rest"][:, i], z["canal_rest"][:, j], "r.-", ms=2, lw=0.8, label="canal (rest)")
        ax.plot(z["canal_final"][:, i], z["canal_final"][:, j], "m.-", ms=2, lw=0.8, label="canal (final)")
        ax.plot(rod[:, i], rod[:, j], "k-", lw=2.5, label="tandem (rigid, flange->tip)")
        ax.set_aspect("equal"); ax.set_title(lab + " [preBT world mm]")
    axs[0].legend(fontsize=7, loc="lower left"); fig.colorbar(sc, ax=axs, shrink=0.8, label="surface displacement (mm)")
    fig.suptitle("%s: body surface rest (grey) vs deformed (colour)" % a.tag)
    fig.savefig(os.path.join(P["figs"], a.tag + "_body.png"), dpi=110); plt.close(fig)
    print("wrote", os.path.join(P["figs"], a.tag + "_slices.png"), os.path.join(P["figs"], a.tag + "_body.png"))


if __name__ == "__main__":
    main()
