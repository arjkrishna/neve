"""CHECK 6b -- blockages on the WRONG-BRANCH route (RECA), all 216 three-source anatomies.

Exact vtkImplicitPolyDataDistance, centerline densified to <=0.25 mm (HANDOFF 11.3/11.4).
Sign POSITIVE=INSIDE, validated in-script against topcow_mr_001 RCCA before anything runs.
RCCA measured alongside RECA by identical code as the paired comparison.

A run of stations with clearance < device radius is TERMINAL if it reaches the branch tip
(natural endcap taper -- the wire stops there anyway), PROXIMAL if it touches s=0, else
MID-VESSEL (a real wedge that arrests the wire short of the tip).
"""
import glob, os, sys
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

STEP = 0.25
THR = [0.18, 0.30, 0.35, 0.48, 0.65]   # wire r | SOFA contactDistance | cath r | wire+cd | cath+cd
TOL = 0.26                              # one densification step

def densify(c, step=STEP):
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    sn = np.concatenate([[0.0], np.cumsum(seg)]); L = sn[-1]
    s = np.linspace(0.0, L, int(np.ceil(L / step)) + 1)
    out = np.empty((len(s), 3))
    for k in range(3): out[:, k] = np.interp(s, sn, c[:, k])
    return out, s, L

def branch(brs, want):
    hit = [k for k in brs if want in k.upper()]
    return np.asarray(brs[hit[0]].coordinates, float) if hit else None

def runs_below(sd, s, t):
    m = sd < t
    if not m.any(): return []
    idx = np.flatnonzero(m)
    return [(float(s[g[0]]), float(s[g[-1]]), float(sd[g].min()))
            for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)]

def classify(r, L):
    st, en, mn = r
    if en >= L - TOL: return "T"
    if st <= TOL:     return "P"
    return "M"

# ---------- control ----------
CTRL = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
mc = pv.read(os.path.join(CTRL, "vessel_architecture_collision.obj")).triangulate().clean()
fc = vtk.vtkImplicitPolyDataDistance(); fc.SetInput(mc)
cb = branch({str(b.name): b for b in load_branches(os.path.join(CTRL, "Centrelines_comb"))}, "RCCA")
dc, sc, Lc = densify(cb)
sdc = np.array([fc.EvaluateFunction(p) for p in dc])
print("CONTROL topcow_mr_001 RCCA densified: n=%d L=%.1f med %.3f min %.3f outside %d"
      % (len(dc), Lc, np.median(sdc), sdc.min(), (sdc < 0).sum()))
if np.median(sdc) <= 0: print("SIGN INVERTED -- abort"); sys.exit(1)
print("sign confirmed POSITIVE=INSIDE; magnitude at lumen scale (frame ok)\n")

hdr = ["name", "n", "L", "s_div", "min", "med", "tip_sd", "prox_sd"]
for t in THR: hdr += ["d%.2f" % t, "dm%.2f" % t, "cls%.2f" % t]
print("HDR " + " ".join(hdr))

MAXN = int(os.environ.get("MAXN", "9999"))
nrow = 0
for dpath in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*"))[:MAXN]:
    if not os.path.isdir(dpath): continue
    name = os.path.basename(dpath)
    try:
        mesh = pv.read(os.path.join(dpath, "vessel_architecture_collision.obj")).triangulate().clean()
        fn = vtk.vtkImplicitPolyDataDistance(); fn.SetInput(mesh)
        brs = {str(b.name): b for b in load_branches(os.path.join(dpath, "Centrelines_comb"))}
        out_lines = []
        cr = branch(brs, "RCCA"); dr, sr, Lr = densify(cr)
        for tag, want in (("RECA", "RECA"), ("RCCA", "RCCA")):
            c = branch(brs, want)
            if c is None: print("MISSING %s %s" % (name, want)); continue
            d, s, L = densify(c)
            sd = np.array([fn.EvaluateFunction(p) for p in d])
            if tag == "RECA":
                dmin = np.empty(len(d))
                for a in range(0, len(d), 200):
                    b = min(a + 200, len(d))
                    dmin[a:b] = np.sqrt(((d[a:b, None, :] - dr[None, :, :]) ** 2).sum(2)).min(1)
                hit = np.flatnonzero(dmin > 0.5)
                s_div = float(s[hit[0]]) if len(hit) else -1.0
            else:
                s_div = -1.0
            fields = [name, "%d" % len(d), "%.3f" % L, "%.3f" % s_div, "%.4f" % sd.min(),
                      "%.4f" % np.median(sd), "%.4f" % sd[-1], "%.4f" % sd[0]]
            for t in THR:
                rr = runs_below(sd, s, t)
                cls = "".join(classify(x, L) for x in rr) or "-"
                depth = rr[0][0] if rr else L                       # first arrest, any cause
                mid = [x for x in rr if classify(x, L) == "M"]
                depth_m = mid[0][0] if mid else L                   # first MID-VESSEL arrest
                fields += ["%.3f" % depth, "%.3f" % depth_m, cls]
            # mid-vessel minimum: exclude the terminal 1.0 mm endcap taper and s=0
            keep = (s > 0.0) & (s < L - 1.0)
            fields.append("%.4f" % (sd[keep].min() if keep.any() else float("nan")))
            fields.append("%.4f" % (s[keep][int(np.argmin(sd[keep]))] if keep.any() else float("nan")))
            for t in (0.18, 0.35):
                rr = runs_below(sd, s, t)
                fields.append("R%.2f=" % t + (";".join("%.2f:%.2f:%.3f:%s" % (a, b, c, classify((a, b, c), L)) for a, b, c in rr) or "none"))
            print(tag + " " + " ".join(fields))
        nrow += 1
    except Exception as e:
        print("FAIL %s: %s: %s" % (name, type(e).__name__, e))
print("\nDONE N=%d" % nrow)
