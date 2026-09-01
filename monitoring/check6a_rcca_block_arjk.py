"""CHECK 6a -- RCCA blockages, 216 three-source anatomies (HANDOFF 11.3 / 12.2 #6).

Exact vtkImplicitPolyDataDistance, centerline densified to <=0.25 mm.
Sign: POSITIVE = inside on this build. Validated in-script against topcow_mr_001.
Blocked station := signed distance < threshold (so points outside the wall count too).
"""
import glob, os, sys
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

THR = [("gw", 0.18), ("sofa", 0.30), ("cath", 0.35)]
DS = 0.25
TERM = 8.0

def densify(c, ds=DS):
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    n = int(np.ceil(L / ds)) + 1
    sq = np.linspace(0.0, L, n)
    out = np.column_stack([np.interp(sq, s, c[:, k]) for k in range(3)])
    return out, sq, L

def runs(mask, sq):
    """Return list of (i0, i1, s0, s1, nstations)."""
    r = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            r.append((i, j, float(sq[i]), float(sq[j]), j - i + 1))
            i = j + 1
        else:
            i += 1
    return r

def measure(d, branch="RCCA"):
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    hit = [k for k in brs if branch in k.upper()]
    if not hit:
        return None
    c = np.asarray(brs[hit[0]].coordinates, float)
    dc, sq, L = densify(c)
    sd = np.array([f.EvaluateFunction(p) for p in dc])
    return dict(n_native=len(c), n=len(sd), L=L, sq=sq, sd=sd)

# ---- CONTROL -------------------------------------------------------------
ctl = measure("/opt/eve_training/results_topbrain/anatomies/topcow_mr_001")
print("CONTROL topcow_mr_001 RCCA: n=%d L=%.2f median=%.3f min=%.3f blocked@0.35=%d"
      % (ctl["n"], ctl["L"], float(np.median(ctl["sd"])), float(ctl["sd"].min()),
         int((ctl["sd"] < 0.35).sum())))
print("SIGN: %s" % ("positive-inside CONFIRMED" if np.median(ctl["sd"]) > 0 else "INVERTED -- ABORT"))
if np.median(ctl["sd"]) <= 0:
    sys.exit(1)
for tag, t in THR:
    rr = runs(ctl["sd"] < t, ctl["sq"])
    print("  CTRL %-5s thr=%.2f nblk=%d nruns=%d %s"
          % (tag, t, int((ctl["sd"] < t).sum()), len(rr),
             ";".join("%.1f-%.1f" % (a, b) for _, _, a, b, _ in rr[:6])))
print("")

# ---- COHORT --------------------------------------------------------------
dirs = sorted(d for d in glob.glob("/opt/eve_training/carotid/anatomies/*") if os.path.isdir(d))
print("NDIRS,%d" % len(dirs))
for d in dirs:
    nm = os.path.basename(d)
    try:
        r = measure(d)
    except Exception as e:
        print("FAIL,%s,%s: %s" % (nm, type(e).__name__, e)); continue
    if r is None:
        print("FAIL,%s,no RCCA" % nm); continue
    sd, sq, L = r["sd"], r["sq"], r["L"]
    parts = ["ROW", nm, "%.3f" % L, "%d" % r["n"], "%.4f" % sd.min(), "%.2f" % sq[int(sd.argmin())]]
    for tag, t in THR:
        mask = sd < t
        rr = runs(mask, sq)
        nblk = int(mask.sum())
        if nblk == 0:
            parts.append("%s|0|-|-|-" % tag)
        else:
            prox = rr[0][2]
            rs = "+".join("%.2f@%.2f-%.2f#%d" % (b - a, a, b, k) for _, _, a, b, k in rr)
            termonly = all(a >= L - TERM for _, _, a, b, k in rr)
            parts.append("%s|%d|%.2f|%s|%d" % (tag, nblk, prox, rs, int(termonly)))
    print(",".join(parts))
print("DONE")
