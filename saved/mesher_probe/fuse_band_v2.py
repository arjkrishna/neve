"""Re-derive the fusing band for the v2 mesher: two parallel 1.6 mm tubes at
increasing wall gap; at what gap do they come out as one lumen?"""
import sys, types, numpy as np
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
import pyvista as pv
from sdf_mesher import tube_field, iso_surface

def tube(r, x0, L=40.0):
    n = int(L / 0.5) + 1
    p = np.zeros((n, 3)); p[:, 2] = np.linspace(0, L, n); p[:, 0] = x0; p += 10.0
    return types.SimpleNamespace(coordinates=p, radii=np.full(n, r), name="t")

for sp in (0.45, 0.35):
    print("spacing %.2f mm" % sp)
    for gap in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        r = 1.6
        b = [tube(r, 0.0), tube(r, 2 * r + gap)]
        m = iso_surface(tube_field(b, spacing=sp))
        comps = int(m.connectivity().cell_data["RegionId"].max()) + 1
        # a probe point midway between the two axes, mid-height: lumen or wall?
        mid = pv.PolyData(np.array([[10.0 + r + gap / 2, 10.0, 30.0]]))
        inside = bool(mid.select_enclosed_points(m, check_surface=False)["SelectedPoints"][0])
        print("   gap %.2f mm  components %d  midpoint %s" % (gap, comps, "LUMEN (fused)" if inside else "wall"), flush=True)
