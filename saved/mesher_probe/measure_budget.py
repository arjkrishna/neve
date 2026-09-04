import os, sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from sdf_mesher import route_lumen, mesh_stats
from eve_bench.dualdevicenav import load_branches
T = "/opt/eve_training/topbrain_data/anatomies_v2_test"; P = "/opt/eve_training/saved/mesher_probe/sofa_cost"
br = load_branches(T + "/topcow_mr_001/Centrelines_comb"); rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
full = pv.read(T + "/topcow_mr_001/collision_full.vtp")
cands = [("quadric 6k", P + "/sdfq_6k/vessel_architecture_collision.obj"), ("quadric 9k", P + "/sdfq_9k/vessel_architecture_collision.obj"),
         ("quadric 12k", P + "/sdfq_12k/vessel_architecture_collision.obj"), ("quadric 20k", T + "/topcow_mr_001/vessel_architecture_collision.obj"),
         ("vmtk weighted k6 11.7k", T + "/mr001_k6/vessel_architecture_collision.obj"), ("vmtk weighted k8 12.8k", T + "/mr001_k8/vessel_architecture_collision.obj")]
print("%-24s %7s %6s %6s %9s %9s %8s" % ("mesh", "tris", "comps", "open", "lumen_min", "deficit", "pts_out"))
for lab, f in cands:
    m = pv.read(f).extract_surface().triangulate()
    d, ins, body, s = route_lumen(m, route); ok = ins & body; st = mesh_stats(m)
    print("%-24s %7d %6d %6d %9.2f %9.2f %8d" % (lab, st["tris"], st["comps"], st["open_edges"], d[ok].min(), np.median(rad[ok] - d[ok]), int((~ins & body).sum())), flush=True)
