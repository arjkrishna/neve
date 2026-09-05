import json, os, sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from eve_bench.dualdevicenav import load_branches
from sdf_union import load_surface
from scipy.spatial import cKDTree
root = "/opt/eve_training/topbrain_data/anatomies_v3/topcow_mr_001"
x = json.load(open(root + "/graft_xform.json"))["sections"][0]
raw = pv.read(x["surface"]); print("raw surface: pts %d open %d" % (raw.n_points, raw.n_open_edges))
surf = load_surface(x["surface"], x["mirror"], x["R"], x["origin"], x["anchor"])
print("after load_surface: pts %d cells %d open %d  manifold %s  bounds z %.0f..%.0f" % (surf.n_points, surf.n_cells, surf.n_open_edges, surf.is_manifold, surf.bounds[4], surf.bounds[5]))
# sign sanity: route points inside the siphon section should be INSIDE the surface (negative implicit distance)
br = load_branches(root + "/Centrelines_comb"); rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
route = np.asarray(rcca.coordinates, float); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
sec = route[(s > 140) & (s < s[-1] - 5)]
d = np.asarray(pv.PolyData(sec).compute_implicit_distance(surf)["implicit_distance"])
print("route pts in siphon section: %d, implicit distance median %.2f (negative = inside), frac inside %.2f" % (len(sec), np.median(d), (d < 0).mean()))
m = pv.read(root + "/vessel_architecture_collision.obj")
e = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("obj open edges:", e.n_cells)
if e.n_cells:
    cen = e.cell_centers().points
    for c in cen[:14]:
        best = None
        for b in br:
            bc = np.asarray(b.coordinates); dd = np.linalg.norm(bc - c, axis=1); j = dd.argmin()
            if best is None or dd[j] < best[0]: best = (dd[j], str(b.name)[-20:], j, float(np.asarray(b.radii)[j]))
        arc = s[np.linalg.norm(route - c, axis=1).argmin()]
        print("   edge at %s  nearest %-20s dist %.1f r %.2f | RCCA arclength %.0f mm" % (np.round(c, 1), best[1], best[0], best[3], arc))
full = pv.read(root + "/collision_full.vtp"); print("full vtp open edges:", full.n_open_edges)
