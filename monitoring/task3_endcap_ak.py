"""TASK 3 - terminal end-cap artifact quantification.

1. exact signed clearance of last 20 mm of RCCA centerline, 0.25 mm spacing, per anatomy + HOST
2. terminal centerline points OUTSIDE the mesh
3. re-score the 220-episode run under distal target-pool trims
4. step-budget check on the >240 mm band
"""
import sys, os, csv, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
import pyvista as pv
import vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

ROOT = "/opt/eve_training/results_topbrain/anatomies"
RUN = ("/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/"
       "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
WIRE, CONTACT, CATH = 0.18, 0.30, 0.35
OFFSET = 33.314          # path_len = s_RCCA + OFFSET
RETAIN = ["001","002","003","004","005","006","007","008","010","011","012","016",
          "017","018","020","021","022","023","024","025","026","027"]


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


def enclosed(mesh, pts):
    ps = pv.PolyData(np.asarray(pts, float))
    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(ps)
    sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6)
    sel.CheckSurfaceOff()
    sel.Update()
    o = sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(len(pts))]) > 0.5


def load(name):
    if name == "HOST":
        vt = DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    d0 = os.path.join(ROOT, name)
    return (pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0, "Centrelines_comb")))


def densify(C, step=0.25):
    S = arclen(C)
    g = np.arange(0.0, S[-1], step)
    g = np.append(g, S[-1])
    P = np.stack([np.interp(g, S, C[:, i]) for i in range(3)], 1)
    return P, g


NAMES = ["HOST"] + ["topcow_mr_" + n for n in RETAIN]
G = {}
print("=" * 168)
print("PART 1/2 -- LAST 20 mm OF RCCA CENTERLINE, exact signed distance, 0.25 mm spacing")
print("  onset_X = mm back from terminus at which the terminal contiguous run with d_eff < X begins")
print("  d_eff = |signed| if enclosed else 0.0 ; nOUT = stations outside the surface")
print("=" * 168)
hdr = ("{:>15} {:>7} {:>5} {:>6} {:>6} {:>6} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>9}"
       .format("anatomy", "L_RCCA", "npts", "n<.18", "n<.30", "n<.35", "on_.18", "on_.30",
               "on_.35", "nOUT", "maxOutD", "outSpan", "d@term"))
print(hdr)


def tail_onset(g, mask):
    """length (mm) of the contiguous True run that ends at the last sample; 0 if last is False"""
    if not mask[-1]:
        return 0.0
    i = len(mask) - 1
    while i - 1 >= 0 and mask[i - 1]:
        i -= 1
    return float(g[-1] - g[i])


for name in NAMES:
    mesh, brs = load(name)
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float)
    P, g = densify(C)
    sd = signed(mesh, P)
    if (sd < 0).mean() < 0.5:
        sd = -sd
    ins = enclosed(mesh, P)
    d = np.abs(sd)
    d_eff = np.where(ins, d, 0.0)
    L = float(g[-1])
    sel = g >= L - 20.0
    gs, ds, insS = g[sel], d_eff[sel], ins[sel]
    dOut = d[sel][~insS]
    outSpan = 0.0
    if (~insS).any():
        outSpan = float(gs[-1] - gs[~insS].min())
    row = ("{:>15} {:7.1f} {:5d} {:6d} {:6d} {:6d} {:7.2f} {:7.2f} {:7.2f} {:8d} {:8.3f} {:8.2f} {:9.3f}"
           .format(name, L, len(gs), int((ds < WIRE).sum()), int((ds < CONTACT).sum()),
                   int((ds < CATH).sum()),
                   tail_onset(gs, ds < WIRE), tail_onset(gs, ds < CONTACT),
                   tail_onset(gs, ds < CATH), int((~insS).sum()),
                   float(dOut.max()) if len(dOut) else 0.0, outSpan, float(ds[-1])))
    print(row)
    G[name] = dict(L=L,
                   n18=int((ds < WIRE).sum()), n30=int((ds < CONTACT).sum()),
                   n35=int((ds < CATH).sum()),
                   on18=tail_onset(gs, ds < WIRE), on30=tail_onset(gs, ds < CONTACT),
                   on35=tail_onset(gs, ds < CATH),
                   nout=int((~insS).sum()),
                   maxout=float(dOut.max()) if len(dOut) else 0.0,
                   outspan=outSpan, dterm=float(ds[-1]),
                   prof=[[float(a), float(b), bool(c)] for a, b, c in
                         zip(gs - L, ds, insS)])

# whole-centerline outside check (part 2, full extent, not just last 20)
print()
print("PART 2b -- OUTSIDE points over the WHOLE RCCA centerline (0.25 mm densified)")
print("{:>15} {:>7} {:>7} {:>9} {:>34}".format("anatomy", "nOUT", "maxDep", "firstOUT_s", "OUT runs (s_start-s_end, depth)"))
for name in NAMES:
    mesh, brs = load(name)
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float)
    P, g = densify(C)
    sd = signed(mesh, P)
    if (sd < 0).mean() < 0.5:
        sd = -sd
    ins = enclosed(mesh, P)
    d = np.abs(sd)
    m = ~ins
    runs = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            runs.append("{:.1f}-{:.1f}/{:.2f}".format(g[i], g[j], float(d[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    print("{:>15} {:7d} {:7.3f} {:9} {:>34}".format(
        name, int(m.sum()), float(d[m].max()) if m.any() else 0.0,
        "{:.1f}".format(g[m][0]) if m.any() else "-", str(runs)[:120]))
    G[name]["nout_full"] = int(m.sum())
    G[name]["maxdep_full"] = float(d[m].max()) if m.any() else 0.0

with open("/tmp/task3_geom.json", "w") as f:
    json.dump(G, f)

# ---------------- PART 3: re-score ----------------
rows = []
with open(os.path.join(RUN, "episodes.csv")) as f:
    for r in csv.DictReader(f):
        r["path_len_mm"] = float(r["path_len_mm"])
        r["steps"] = int(r["steps"])
        r["success"] = int(r["success"])
        rows.append(r)


def key(a):
    return "topcow_mr_" + a.replace("topcowmr", "")


for r in rows:
    k = key(r["anatomy"])
    r["L"] = G[k]["L"]
    r["s_rcca"] = r["path_len_mm"] - OFFSET
    r["dist_from_term"] = G[k]["L"] - r["s_rcca"]

print()
print("PART 3 -- target distance from its own anatomy terminus  (s_term = L_RCCA; "
      "target s_RCCA = path_len - {:.2f})".format(OFFSET))
dts = np.array([r["dist_from_term"] for r in rows])
print("  overall n={} min {:.2f}  p05 {:.2f}  median {:.2f}  max {:.2f}"
      .format(len(dts), dts.min(), np.percentile(dts, 5), np.median(dts), dts.max()))
print("  per-anatomy minimum dist_from_term (i.e. deepest target vs its terminus):")
per = {}
for r in rows:
    per.setdefault(r["anatomy"], []).append(r)
for a in sorted(per):
    v = [x["dist_from_term"] for x in per[a]]
    print("    {:>12}  L={:6.1f}  n={:2d}  min_dft={:6.2f}  max_plen={:6.1f}"
          .format(a, per[a][0]["L"], len(v), min(v), max(x["path_len_mm"] for x in per[a])))


def score(sub):
    n = len(sub)
    s = sum(x["success"] for x in sub)
    return n, s, (100.0 * s / n if n else float("nan"))


print()
print("PART 3 -- RE-SCORE UNDER DISTAL TARGET-POOL TRIM N (exclude targets within N mm of terminus)")
print("{:>4} {:>6} {:>18} {:>18} {:>18} {:>18}".format(
    "N", "drop", "OVERALL", "siphon", "plen>240", "plen 200-240"))
for N in [0, 2, 4, 8, 12]:
    keep = [r for r in rows if r["dist_from_term"] >= N]
    drop = len(rows) - len(keep)
    o = score(keep)
    si = score([r for r in keep if r["section"] == "siphon"])
    b240 = score([r for r in keep if r["path_len_mm"] > 240])
    b200 = score([r for r in keep if 200 <= r["path_len_mm"] <= 240])
    def f(t):
        return "{:3d}/{:3d}={:5.1f}%".format(t[1], t[0], t[2])
    print("{:>4} {:>6} {:>18} {:>18} {:>18} {:>18}".format(N, drop, f(o), f(si), f(b240), f(b200)))

# control: trim by ABSOLUTE path_len cap instead (does removing the deepest targets by
# path_len alone explain it?) -- and trim from the FRONT as a placebo
print()
print("PLACEBO -- drop the SAME NUMBER of episodes but the ones FARTHEST from the terminus")
srt = sorted(rows, key=lambda r: -r["dist_from_term"])
for N in [0, 2, 4, 8, 12]:
    ndrop = sum(1 for r in rows if r["dist_from_term"] < N)
    keep = srt[ndrop:]
    o = score(keep)
    b240 = score([r for r in keep if r["path_len_mm"] > 240])
    print("  N={:2d} drop={:3d}  overall {:3d}/{:3d}={:5.1f}%   >240 {:3d}/{:3d}={:5.1f}%"
          .format(N, ndrop, o[1], o[0], o[2], b240[1], b240[0], b240[2]))

# ---------------- PART 4: step budget ----------------
print()
print("PART 4 -- STEP BUDGET")
bands = [("<167", lambda p: p < 167), ("167-200", lambda p: 167 <= p < 200),
         ("200-240", lambda p: 200 <= p < 240), (">=240", lambda p: p >= 240)]
print("{:>9} {:>6} {:>6} {:>7} | successes: {:>5} {:>5} {:>5} {:>5} {:>5} | fails: {:>7} {:>7}"
      .format("band", "n", "succ", "rate", "med", "p75", "p90", "max", "n>=500", "n_600", "n_other"))
for nm, fn in bands:
    sub = [r for r in rows if fn(r["path_len_mm"])]
    su = [r["steps"] for r in sub if r["success"] == 1]
    fa = [r["steps"] for r in sub if r["success"] == 0]
    su_a = np.array(su) if su else np.array([0])
    print("{:>9} {:6d} {:6d} {:6.1f}% | {:16d} {:5d} {:5d} {:5d} {:5d} | {:14d} {:7d}"
          .format(nm, len(sub), len(su), 100.0 * len(su) / max(len(sub), 1),
                  int(np.median(su_a)), int(np.percentile(su_a, 75)),
                  int(np.percentile(su_a, 90)), int(su_a.max()),
                  int((su_a >= 500).sum()),
                  sum(1 for s in fa if s >= 600), sum(1 for s in fa if s < 600)))

print()
print("  >=240 band, every episode (steps, success, dist_from_term):")
for r in sorted([r for r in rows if r["path_len_mm"] >= 240], key=lambda r: r["path_len_mm"]):
    print("    {:>12} plen={:6.1f} dft={:6.2f} steps={:4d} succ={}"
          .format(r["anatomy"], r["path_len_mm"], r["dist_from_term"], r["steps"], r["success"]))

print()
print("  failure step-count distribution, ALL rows: ")
fa = [r["steps"] for r in rows if r["success"] == 0]
import collections
print("   ", collections.Counter(fa).most_common(10))
