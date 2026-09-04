"""Meshed lumen along the RCCA route for every v1 anatomy (the shipped .obj),
so v1 and v2 can be compared distribution against distribution."""
import glob, json, os, sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from sdf_mesher import route_lumen, mesh_stats
from eve_bench.dualdevicenav import load_branches
out = {}
for root in sorted(glob.glob("/opt/eve_training/topbrain_data/anatomies/*") + glob.glob("/opt/eve_training/carotid_data/anatomies/*")):
    cl = os.path.join(root, "Centrelines_comb"); obj = os.path.join(root, "vessel_architecture_collision.obj")
    if not (os.path.isdir(cl) and os.path.exists(obj)):
        continue
    br = load_branches(cl); rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
    route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
    m = pv.read(obj); d, ins, body, s = route_lumen(m, route); ok = ins & body
    rec = mesh_stats(m)
    if ok.any():
        i = int(np.argmin(np.where(ok, d, np.inf)))
        rec.update(lumen_min_mm=round(float(d[i]), 3), lumen_min_at_mm=round(float(s[i]), 1), declared_there_mm=round(float(rad[i]), 3),
                   median_deficit_mm=round(float(np.median(rad[ok] - d[ok])), 3), route_pts_outside=int((~ins & body).sum()),
                   navigable=bool(d[ok].min() - 0.3 >= 0.35))
    out[os.path.basename(root)] = rec
    print("%-46s lumen_min %.2f deficit %.2f comps %d %s" % (os.path.basename(root)[:46], rec.get("lumen_min_mm", -1), rec.get("median_deficit_mm", -1), rec["comps"], "" if rec.get("navigable") else "NOT NAVIGABLE"), flush=True)
json.dump(out, open("/opt/eve_training/saved/mesher_probe/lumen_v1.json", "w"), indent=1)
nav = sum(1 for r in out.values() if r.get("navigable")); print("\n%d anatomies, %d navigable by the meshed-lumen test" % (len(out), nav))
