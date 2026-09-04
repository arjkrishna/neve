import sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from eve_bench.dualdevicenav import load_branches
import sdf_mesher as S
root = "/opt/eve_training/topbrain_data/anatomies_v2_test/topcow_mr_001"
br = load_branches(root + "/Centrelines_comb")
def report(tag, m):
    e = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    print("%-28s tris %7d  open edges %4d" % (tag, m.n_cells, e.n_cells), flush=True)
    if e.n_cells:
        c = e.cell_centers().points
        from scipy.cluster.vq import kmeans2
        k = min(6, len(c)); cent, lab = kmeans2(c, k, seed=1, minit="++")
        for i in range(k):
            pts = c[lab == i]
            if not len(pts): continue
            # nearest branch and its radius there
            best = None
            for b in br:
                bc = np.asarray(b.coordinates); d = np.linalg.norm(bc - pts.mean(0), axis=1); j = d.argmin()
                if best is None or d[j] < best[0]: best = (d[j], str(b.name)[-22:], float(np.asarray(b.radii)[j]), j, len(bc))
            print("     cluster %d: %3d edges near %s  dist %.1f mm  r %.2f  (pt %d/%d)" % (i, len(pts), best[1], best[0], best[2], best[3], best[4]))
for tag, kw in (("boxed + coarse (new)", {}), ("boxed, band 6", {"band": 6})):
    f = S.tube_field(br, **kw); m = S.iso_surface(f); report(tag + " iso", m); report(tag + " 20k", S.decimate_to(m, 20000))
S._coarse = lambda c, r, sp: (c, r)
f = S.tube_field(br); m = S.iso_surface(f); report("boxed, dense marking iso", m); report("boxed, dense 20k", S.decimate_to(m, 20000))
