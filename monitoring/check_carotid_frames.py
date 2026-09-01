"""Intake frame/sign/enclosure check on the 216 three-source anatomies (HANDOFF 12.1).

Frame mismatch reads in the hundreds of mm; correct reads at lumen scale.
Sign is validated against a known-good TopBrain anatomy run through identical code --
vtkImplicitPolyDataDistance takes its sign from normals and these meshes are not watertight.
"""
import glob, os, sys
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

def probe(d, names):
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    out = {}
    for want in names:
        hit = [k for k in brs if want in k.upper()]
        if not hit: out[want] = None; continue
        c = np.asarray(brs[hit[0]].coordinates, float)
        sd = np.array([f.EvaluateFunction(p) for p in c])   # POSITIVE = inside, this build
        out[want] = (len(c), float(np.median(sd)), float(sd.min()), int((sd < 0).sum()))
    return out

CTRL = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
c = probe(CTRL, ["RCCA"])["RCCA"]
print(f"CONTROL topcow_mr_001 (known good): n={c[0]} median {c[1]:.3f} min {c[2]:.3f} outside {c[3]}")
print("  -> sign convention confirmed positive-inside\n" if c[1] > 0 else "  -> SIGN INVERTED, abort\n")

rows = []
for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")):
    if not os.path.isdir(d): continue
    try: rows.append((os.path.basename(d), probe(d, ["RCCA", "RECA"])))
    except Exception as e: print(f"  FAIL {os.path.basename(d)}: {type(e).__name__}: {e}")

for br in ("RCCA", "RECA"):
    v = [(n, r[br]) for n, r in rows if r.get(br)]
    med = np.array([x[1][1] for x in v]); mn = np.array([x[1][2] for x in v])
    out = np.array([x[1][3] for x in v])
    frame_bad = [n for n, t in v if abs(t[1]) > 20]
    print(f"{br}: {len(v)}/{len(rows)} anatomies have this branch")
    print(f"   median clearance  min {med.min():.3f}  med {np.median(med):.3f}  max {med.max():.3f} mm")
    print(f"   FRAME MISMATCH (|median| > 20 mm): {len(frame_bad)}  {frame_bad[:5]}")
    print(f"   anatomies with >=1 point outside: {(out > 0).sum()} / {len(v)}"
          f"   (worst {out.max()} pts, deepest {mn.min():.3f} mm)")
    bad = sorted([(n, t[3], t[2]) for n, t in v if t[3] > 0], key=lambda x: -x[1])[:8]
    for n, k, d_ in bad: print(f"      {n:44s} {k:3d} pts, worst {d_:.2f} mm")
