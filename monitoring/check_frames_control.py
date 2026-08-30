"""Frame check on the 24 mirrored-LEFT anatomies only.

The trap: the host's raw .obj is in the MESH frame while load_branches() returns the
BRANCH frame; comparing them gives ~400 mm nonsense. The 25 right-ICA cohort meshes were
verified frame-consistent with their own centerlines. The 24 mirrored-left grafts are new
and the mirror (sp * [-1,1,1] in RAS) is exactly the kind of operation that can be applied
to one and not the other. A frame mismatch shows up as clearance in the hundreds of mm; a
correct one sits at roughly the lumen radius.
"""
import glob, os, sys
import numpy as np
import pyvista as pv
import vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/results_topbrain/anatomies"

def exact_signed(mesh, pts):
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(mesh)
    return np.array([f.EvaluateFunction(p) for p in pts])

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "topcow_mr_00[1-8]"))+glob.glob(os.path.join(ROOT,"topcow_mr_00[1-4]_L"))):
    n = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = [b for b in load_branches(os.path.join(d, "Centrelines_comb"))
          if "RCCA" in str(b.name).upper()][0]
    c = np.asarray(br.coordinates, float)
    sd = exact_signed(m, c)                 # negative = inside for this filter
    clr = -sd                               # distance to wall, positive inside
    rows.append((n, len(c), float(np.median(clr)), float(clr.min()), float(clr.max()),
                 int((clr < 0).sum())))

print(f"{'anatomy':18s} {'pts':>5} {'clr med':>9} {'clr min':>9} {'clr max':>9} {'outside':>8}")
print("-"*66)
bad = 0
for n, k, med, mn, mx, out in rows:
    flag = ""
    if med > 20 or mx > 50: flag = "  <-- FRAME MISMATCH"; bad += 1
    elif out: flag = f"  <-- {out} pts outside"
    print(f"{n:18s} {k:5d} {med:9.3f} {mn:9.3f} {mx:9.3f} {out:8d}{flag}")
print(f"\nframe-mismatched: {bad}/{len(rows)}")
med_all = np.array([r[2] for r in rows])
print(f"median clearance across the 24: min {med_all.min():.3f}  med {np.median(med_all):.3f}  max {med_all.max():.3f} mm")
print("(a mesh-vs-branch frame mismatch reads in the hundreds of mm; ~1-2 mm is correct)")
