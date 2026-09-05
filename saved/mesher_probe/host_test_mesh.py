"""Shipped host TEST mesh measured in the env's own frame: build DualDeviceNav,
take vessel_tree.mesh_path (the mesh after the env's rotation/scale) and
vessel_tree.branches (same frame), then lumen along the RCCA vs MISR radii."""
import sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench")
import eve_bench.dualdevicenav as dd
cls = getattr(dd, "DualDeviceNav")
iv = cls()
vt = iv.vessel_tree
rcca = [b for b in vt.branches if "RCCA" in str(b.name).upper()][0]
route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]; body = (s > 3) & (s < s[-1] - 3)
m = pv.read(vt.mesh_path).extract_surface().triangulate()
d = np.abs(np.asarray(pv.PolyData(route).compute_implicit_distance(m)["implicit_distance"]))
dd_ = rad[body] - d[body]
print("host TEST mesh (env frame)  tris %d  open %d | lumen min %.2f at %.0f mm | deficit median %+.2f  p10 %+.2f  p90 %+.2f | declared min %.2f" % (
    m.n_cells, m.n_open_edges, d[body].min(), s[body][np.argmin(d[body])], np.median(dd_), np.percentile(dd_, 10), np.percentile(dd_, 90), rad[body].min()), flush=True)
# how non-circular is the real test lumen? distance to wall in 8 directions round the centerline at a few stations
from scipy.spatial import cKDTree
pts = np.asarray(m.points); tree = cKDTree(pts)
ratios = []
for i in np.linspace(int(0.15 * len(route)), int(0.85 * len(route)), 12).astype(int):
    t = route[min(i + 1, len(route) - 1)] - route[max(i - 1, 0)]; t /= np.linalg.norm(t)
    a = np.cross(t, [1, 0, 0]); a = a if np.linalg.norm(a) > 0.1 else np.cross(t, [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(t, a)
    rr = []
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        u = np.cos(th) * a + np.sin(th) * b
        # march along u until the nearest wall point lies ahead: crude ray-cast via nearest-point distance
        best = None
        for step in np.arange(0.2, 8.0, 0.1):
            q = route[i] + u * step
            dist, _ = tree.query(q)
            if dist < 0.35: best = step; break
        rr.append(best if best else np.nan)
    rr = np.array(rr); rr = rr[~np.isnan(rr)]
    if len(rr) >= 8: ratios.append((rr.min() / rr.max(), s[i], rad[i], rr.min(), rr.max()))
print("real test lumen cross-sections (wall distance in 16 directions): min/max ratio median %.2f, worst %.2f; e.g. at %.0f mm declared MISR %.2f, wall %.1f-%.1f mm" % (
    np.median([r[0] for r in ratios]), min(r[0] for r in ratios), ratios[0][1], ratios[0][2], ratios[0][3], ratios[0][4]))
