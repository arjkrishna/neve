import sys, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from eve_bench.dualdevicenav import load_branches
from sdf_mesher import tube_field, iso_surface, mesh_stats
from sdf_union import add_real_sections
from skimage import measure
from scipy import ndimage
import bake_meshes_v3 as B
root = "/opt/eve_training/topbrain_data/anatomies_v3/topcow_mr_001"
br = load_branches(root + "/Centrelines_comb"); rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
field = tube_field(br, spacing=0.45); secs, kinds = B.sections_for(root, br, route, rad); add_real_sections(field, secs, 0.45)
def mc(mask):
    v, f, _, _ = measure.marching_cubes(field.values, 0.0, spacing=(0.45,) * 3, gradient_direction="descent", mask=mask, allow_degenerate=False)
    m = pv.PolyData(v, np.c_[np.full(len(f), 3), f]); m.translate(field.offset, inplace=True); return m
for lab, mask in (("eroded region (current)", ndimage.binary_erosion(field.region, iterations=1)),
                  ("region, no erosion", field.region),
                  ("region dilated 2", ndimage.binary_dilation(field.region, iterations=2)),
                  ("no mask", None)):
    m = mc(mask); st = mesh_stats(m)
    conn = m.connectivity(); rid = np.asarray(conn.cell_data["RegionId"]); sizes = np.bincount(rid)
    print("%-26s tris %7d comps %d open %4d  component sizes %s" % (lab, st["tris"], st["comps"], st["open_edges"], sorted(sizes.tolist(), reverse=True)[:4]), flush=True)
