import json, os, sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from eve_bench.dualdevicenav import load_branches
from sdf_mesher import tube_field, iso_surface, mesh_stats
from sdf_union import load_surface, add_real_sections, Section, _capsule_radii
import bake_meshes_v3 as B
root = "/opt/eve_training/topbrain_data/anatomies_v3/topcow_mr_001"
br = load_branches(root + "/Centrelines_comb"); rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
field = tube_field(br, spacing=0.45)
secs, kinds = B.sections_for(root, br, route, rad)
add_real_sections(field, secs, 0.45)
iso = iso_surface(field); print("iso surface:", mesh_stats(iso))
e = iso.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
if e.n_cells:
    c = e.cell_centers().points; print("iso open-edge centres (first 5):", np.round(c[:5], 1))
# what is the real surface doing near (-15, 15.5, 588.5)?
q = np.array([[-15.0, 15.5, 588.5]])
surf = secs[0].surface
d = pv.PolyData(q).compute_implicit_distance(surf)["implicit_distance"][0]
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
j = np.linalg.norm(route - q, axis=1).argmin()
print("probe: implicit dist to real surface %.2f (neg=inside) | nearest route pt at %.0f mm, r %.2f, dist %.2f" % (d, s[j], rad[j], np.linalg.norm(route[j] - q)))
# real surface extent vs route: how far from the route does the real surface reach, along the section?
sec = secs[0]; sp = sec.points
from scipy.spatial import cKDTree
tree = cKDTree(sp); dd, jj = tree.query(np.asarray(surf.points))
capr = _capsule_radii(sec.points, sec.radii, True, False)
over = dd > capr[jj]
print("real surface points beyond the capsule: %d of %d (%.1f%%); where (route mm): %s" % (over.sum(), len(dd), 100 * over.mean(), np.round(np.percentile(np.r_[0, np.cumsum(np.linalg.norm(np.diff(sp, axis=0), axis=1))][jj[over]], [5, 50, 95]) + kinds["siphon"][0], 0) if over.any() else "-"))
