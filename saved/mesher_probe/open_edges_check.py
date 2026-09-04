import sys, time, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from eve_bench.dualdevicenav import load_branches
import sdf_mesher as S
root = "/opt/eve_training/topbrain_data/anatomies_v2_test/topcow_mr_001"
br = load_branches(root + "/Centrelines_comb")
t0 = time.time(); f = S.tube_field(br, verbose=True); m = S.iso_surface(f); t1 = time.time() - t0
m20 = S.decimate_to(m, 20000)
rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
d, ins, body, s = S.route_lumen(m20, np.asarray(rcca.coordinates, float)); ok = ins & body
print("iso %d tris open %d | 20k open %d comps %d | lumen_min %.2f deficit %.2f | field+iso %.1f s" % (
    m.n_cells, m.n_open_edges, m20.n_open_edges, S.mesh_stats(m20)["comps"], d[ok].min(), np.median(np.asarray(rcca.radii)[ok] - d[ok]), t1))
