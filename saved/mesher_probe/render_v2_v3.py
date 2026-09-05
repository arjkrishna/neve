"""Side by side: v2 tube mesh vs v3 real-surface mesh, zoomed on the graft."""
import os, sys, numpy as np, pyvista as pv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches
OUT = "/opt/eve_training/saved/mesher_probe"
for name, r2, r3, lo in (("topcow_mr_001", "/opt/eve_training/topbrain_data/anatomies_v2/topcow_mr_001", "/opt/eve_training/topbrain_data/anatomies_v3/topcow_mr_001", 125.0),
                         ("case_k_004_left__topcow_mr_010", "/opt/eve_training/carotid_data/anatomies_v2/case_k_004_left__topcow_mr_010", "/opt/eve_training/carotid_data/anatomies_v3/case_k_004_left__topcow_mr_010", 40.0)):
    br = load_branches(r3 + "/Centrelines_comb"); rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
    route = np.asarray(rcca.coordinates, float); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    seg = route[s >= lo]; c = (seg.min(0) + seg.max(0)) / 2; R = float(np.max(seg.max(0) - seg.min(0))) / 2 * 1.08
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    for k, (lab, root) in enumerate((("v2: SDF tubes", r2), ("v3: tubes + real segmented surfaces", r3))):
        m = pv.read(root + "/vessel_architecture_collision.obj").extract_surface().triangulate()
        v = np.asarray(m.points); f = np.asarray(m.faces).reshape(-1, 4)[:, 1:]
        cen = v[f].mean(1); keep = np.all(np.abs(cen - c) < R * 1.15, axis=1)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d"); ax.set_facecolor("white")
        ax.add_collection3d(Poly3DCollection(v[f[keep]], facecolor=(0.84, 0.15, 0.16, 0.32), edgecolor=(0.25, 0.25, 0.3, 0.12), linewidth=0.15))
        ax.plot(*seg.T, color="#1f77b4", lw=1.2)
        for i in range(3):
            getattr(ax, "set_%slim" % "xyz"[i])(c[i] - R, c[i] + R)
        ax.view_init(elev=12, azim=-60); ax.set_axis_off()
        ax.set_title("%s  (%d triangles)" % (lab, m.n_cells), fontsize=12)
    fig.suptitle(name + " -- collision mesh from %.0f mm along the route" % lo, fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.0)
    fig.savefig(os.path.join(OUT, "v2_vs_v3_%s.png" % name), dpi=150); plt.close(fig); print("wrote", name, flush=True)
