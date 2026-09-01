"""CHECK 9 stratification instrument ONLY: per-anatomy RCCA clearance median/min
so the SOFA subset can be picked at the true extremes. Native station spacing
(selection, not a defect measurement -- HANDOFF 11.3 densification is for minima).
Control anatomy in the same script per HANDOFF 11.2."""
import glob, os, sys, json
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

def probe(d):
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    hit = [k for k in brs if "RCCA" in k.upper()]
    c = np.asarray(brs[hit[0]].coordinates, float)
    sd = np.array([f.EvaluateFunction(p) for p in c])
    return len(c), float(np.median(sd)), float(sd.min()), int((sd < 0).sum()), int(m.n_cells)

CTRL = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
n, med, mn, no, nc = probe(CTRL)
print("CONTROL topcow_mr_001: n=%d median %.3f min %.3f outside %d cells %d" % (n, med, mn, no, nc))
if med < 0:
    print("SIGN INVERTED -- abort"); sys.exit(1)
print("sign confirmed positive-inside\n")

rows = {}
for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")):
    if not os.path.isdir(d): continue
    nm = os.path.basename(d)
    try:
        rows[nm] = probe(d)
    except Exception as e:
        print("FAIL %s %r" % (nm, e))
print("done %d" % len(rows))
json.dump(rows, open("/tmp/rank.json", "w"))
print("JSONSTART"); print(json.dumps(rows)); print("JSONEND")
