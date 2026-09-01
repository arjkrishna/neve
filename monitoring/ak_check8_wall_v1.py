"""CHECK 8, mesh ground truth: is there WALL between the RCCA and RECA lumens?

The declared-radius test (ak_check8_fusion_v1) is a proxy; per HANDOFF 11.5 the
declared radius overstates true bore by ~1.1 mm on grafted geometry and RECA is
floored at 1.6 mm, so it is systematically PESSIMISTIC.  This script measures the
actual surface: at every RECA station distal to the bifurcation, take the segment
joining that station to its nearest RCCA station and measure how much of it lies
OUTSIDE the collision surface.  A contiguous outside stretch = wall.  Zero = one
merged cavity, i.e. the wire can cross between branches without a fork decision.

Sign convention (HANDOFF 11.2): vtkImplicitPolyDataDistance is POSITIVE INSIDE on
these meshes.  Two controls are run in this same script every time.
"""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
import pyvista as pv
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
STEP = 0.25
NSEG = 401          # samples along each connecting segment
FORK_EXCL = 8.0     # mm along RECA from the fork; beyond this the branches must be separate


def densify(C, R, step=STEP):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    S = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(int(np.ceil(S[-1] / step)) + 1, len(C))
    Sn = np.linspace(0.0, S[-1], n)
    Cn = np.stack([np.interp(Sn, S, C[:, k]) for k in range(3)], axis=1)
    return Cn, np.interp(Sn, S, R), Sn


def short(n):
    return str(n).replace("Centerline curve ", "").replace(".mrk", "")


def sd(mesh, pts):
    """signed distance, POSITIVE INSIDE on these meshes (verified by controls)."""
    p = pv.PolyData(np.ascontiguousarray(np.asarray(pts, float)))
    return np.asarray(p.compute_implicit_distance(mesh)["implicit_distance"], float)


names = sorted(os.listdir(ROOT))
names = [n for n in names if os.path.isdir(os.path.join(ROOT, n, "Centrelines_comb"))]
print("anatomies:", len(names))
print("device scale: guidewire r=0.18, catheter r=0.35, SOFA contactDistance=0.30 mm")
sys.stdout.flush()

rows = []
ctrl_fail = []
for ai, nm in enumerate(names):
    d0 = os.path.join(ROOT, nm)
    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    bd = {short(b.name): densify(np.asarray(b.coordinates, np.float64),
                                 np.asarray(b.radii, np.float64)) for b in brs}
    C, rC, sC = bd["- RCCA"]
    E, rE, sE = bd["- RECA"]

    # ---- CONTROLS, every anatomy -------------------------------------------
    ctrl_in = sd(mesh, C)                       # centerline points: inside by construction
    far = np.asarray(mesh.bounds).reshape(3, 2)[:, 1] + 200.0
    ctrl_out = sd(mesh, far[None, :])[0]        # a point 200 mm outside the bbox
    sign_ok = (np.median(ctrl_in) > 0) and (ctrl_out < 0)
    if not sign_ok:
        ctrl_fail.append(nm)

    # ---- wall between RCCA and RECA, station by station ---------------------
    keep = np.where(sE >= FORK_EXCL)[0]
    segs, meta = [], []
    for j in keep:
        k = int(np.argmin(np.linalg.norm(C - E[j], axis=1)))
        P, Q = C[k], E[j]
        t = np.linspace(0.0, 1.0, NSEG)
        segs.append(P[None, :] + t[:, None] * (Q - P)[None, :])
        meta.append((float(sE[j]), float(sC[k]), float(np.linalg.norm(Q - P))))
    allpts = np.concatenate(segs, axis=0)
    v = sd(mesh, allpts).reshape(len(keep), NSEG)
    outside = v < 0.0
    # contiguous outside run in the interior of the segment, in mm
    walls = []
    for m in range(len(keep)):
        L = meta[m][2]
        o = outside[m]
        best = 0
        run = 0
        for val in o:
            run = run + 1 if val else 0
            if run > best:
                best = run
        walls.append(best * L / (NSEG - 1))
    walls = np.array(walls)
    fo = outside.mean(axis=1)
    im = int(np.argmin(walls))
    r = {"name": nm, "sign_ok": bool(sign_ok),
         "ctrl_in_med": float(np.median(ctrl_in)), "ctrl_out": float(ctrl_out),
         "n_stations": int(len(keep)),
         "wall_min": float(walls[im]), "wall_min_sE": meta[im][0], "wall_min_sC": meta[im][1],
         "ctr_dist_at_min": meta[im][2], "fracout_at_min": float(fo[im]),
         "wall_med": float(np.median(walls)), "wall_max": float(walls.max()),
         "n_zero_wall": int((walls <= 1e-9).sum()),
         "n_wall_lt_030": int((walls < 0.30).sum()),
         "n_wall_lt_070": int((walls < 0.70).sum())}
    rows.append(r)
    if (ai + 1) % 25 == 0:
        print("  ...%d/%d" % (ai + 1, len(names)))
        sys.stdout.flush()

json.dump(rows, open("/tmp/out/check8_wall.json", "w"), indent=0, default=float)


def dist(vals, label, fmt="%.3f"):
    v = np.array([x for x in vals if x == x], float)
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    print(("%-40s n=%3d  min/p5/p25/med/p75/p95/max = " + " / ".join([fmt] * 7))
          % ((label, len(v)) + tuple(q)))


print("")
print("=" * 118)
print("CONTROLS")
print("=" * 118)
print("  sign control failed on: %s" % (ctrl_fail if ctrl_fail else "none (0/216)"))
dist([r["ctrl_in_med"] for r in rows], "median signed dist at RCCA centerline")
dist([r["ctrl_out"] for r in rows], "signed dist 200 mm outside bbox")

print("")
print("=" * 118)
print("MESH WALL between RCCA and RECA lumens, RECA s >= %.0f mm from fork" % FORK_EXCL)
print("=" * 118)
dist([r["wall_min"] for r in rows], "MIN wall thickness per anatomy (mm)")
dist([r["wall_med"] for r in rows], "median wall over stations (mm)")
dist([r["ctr_dist_at_min"] for r in rows], "centre distance at the min (mm)")
print("  anatomies with ANY zero-wall station (fused): %d"
      % sum(1 for r in rows if r["n_zero_wall"] > 0))
print("  anatomies with min wall < 0.30 mm (contactDistance): %d"
      % sum(1 for r in rows if r["wall_min"] < 0.30))
print("  anatomies with min wall < 0.70 mm (catheter diameter): %d"
      % sum(1 for r in rows if r["wall_min"] < 0.70))
print("")
print(" 15 thinnest walls:")
for r in sorted(rows, key=lambda r: r["wall_min"])[:15]:
    print("  %-42s wall=%6.3f at RECA s=%6.2f / RCCA s=%7.2f  ctr_d=%6.3f fracout=%.3f  nz=%d n<0.3=%d"
          % (r["name"], r["wall_min"], r["wall_min_sE"], r["wall_min_sC"],
             r["ctr_dist_at_min"], r["fracout_at_min"], r["n_zero_wall"], r["n_wall_lt_030"]))
print(" 5 thickest:")
for r in sorted(rows, key=lambda r: -r["wall_min"])[:5]:
    print("  %-42s wall=%6.3f" % (r["name"], r["wall_min"]))
