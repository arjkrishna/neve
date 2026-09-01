"""CHECK 8, mesh ground truth v2: full wall profile along RECA, carina located
from the mesh instead of assumed.

For every RECA station (0.25 mm), take the segment to the nearest RCCA station and
measure the longest contiguous stretch of it lying OUTSIDE the collision surface.
That stretch is the WALL between the two lumens.

  wall == 0 near the fork  -> the shared bifurcation lumen, correct
  wall  > 0 beyond it      -> two separate lumens, a fork decision exists
  a SECOND zero-wall interval further distal -> genuine fusion defect

Sign: vtkImplicitPolyDataDistance POSITIVE INSIDE (HANDOFF 11.2); two controls run
per anatomy.  Also spot-checks the tightest RCCA-RVA and the LVA/RVA confluence.
"""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
import pyvista as pv
from collections import Counter
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
STEP = 0.25
NSEG = 161


def densify(C, R, step=STEP):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    S = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(int(np.ceil(S[-1] / step)) + 1, len(C))
    Sn = np.linspace(0.0, S[-1], n)
    return (np.stack([np.interp(Sn, S, C[:, k]) for k in range(3)], axis=1),
            np.interp(Sn, S, R), Sn)


def short(n):
    return str(n).replace("Centerline curve ", "").replace(".mrk", "")


def sd(mesh, pts):
    p = pv.PolyData(np.ascontiguousarray(np.asarray(pts, float)))
    return np.asarray(p.compute_implicit_distance(mesh)["implicit_distance"], float)


def longest_runs(M):
    """longest contiguous True run per ROW of a boolean 2-D array, vectorised."""
    n = M.shape[1]
    pad = np.zeros((M.shape[0], 1), bool)
    P = np.concatenate([pad, M, pad], axis=1).astype(np.int8)
    d = np.diff(P, axis=1)
    out = np.zeros(M.shape[0], int)
    for i in range(M.shape[0]):
        st = np.flatnonzero(d[i] == 1)
        en = np.flatnonzero(d[i] == -1)
        out[i] = (en - st).max() if st.size else 0
    return out


def wall_profile(mesh, A, sA, B):
    """for each station of A, wall thickness on the segment to the nearest point of B"""
    k = np.array([int(np.argmin(np.linalg.norm(B - a, axis=1))) for a in A])
    P = B[k]
    t = np.linspace(0.0, 1.0, NSEG)
    segs = P[:, None, :] + t[None, :, None] * (A - P)[:, None, :]
    L = np.linalg.norm(A - P, axis=1)
    v = sd(mesh, segs.reshape(-1, 3)).reshape(len(A), NSEG)
    out = v < 0.0
    w = longest_runs(out) * L / (NSEG - 1)
    return w, L, k


def runs_of_zero(w, s, tol=1e-9):
    z = w <= tol
    out = []
    i = 0
    while i < len(z):
        if z[i]:
            j = i
            while j + 1 < len(z) and z[j + 1]:
                j += 1
            out.append((float(s[i]), float(s[j]), j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


names = sorted(os.listdir(ROOT))
names = [n for n in names if os.path.isdir(os.path.join(ROOT, n, "Centrelines_comb"))]
print("anatomies: %d" % len(names))
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
    ci = sd(mesh, C)
    far = np.asarray(mesh.bounds).reshape(3, 2)[:, 1] + 200.0
    co = sd(mesh, far[None, :])[0]
    if not (np.median(ci) > 0 and co < 0):
        ctrl_fail.append(nm)

    w, L, k = wall_profile(mesh, E, sE, C)
    zr = runs_of_zero(w, sE)
    r = {"name": nm, "L_eca": float(sE[-1]), "n_zero_runs": len(zr),
         "zero_runs": [(round(a, 2), round(b, 2), n) for a, b, n in zr],
         "ctrl_in_med": float(np.median(ci)), "ctrl_out": float(co)}
    if zr:
        first = zr[0]
        r["carina_s"] = first[1]              # end of the first (fork) zero-wall interval
        r["fork_zero_starts_at"] = first[0]
        extra = zr[1:]
    else:
        r["carina_s"] = 0.0
        r["fork_zero_starts_at"] = float("nan")
        extra = []
    r["extra_zero_runs"] = [(round(a, 2), round(b, 2), n) for a, b, n in extra]
    m = sE > r["carina_s"] + 1e-9
    if m.any():
        r["wall_min_distal"] = float(w[m].min())
        r["wall_min_distal_sE"] = float(sE[m][int(np.argmin(w[m]))])
        r["wall_med_distal"] = float(np.median(w[m]))
        r["wall_tip"] = float(w[-1])
        r["distal_len"] = float(sE[-1] - r["carina_s"])
        for thr, key in ((0.30, "n_lt030"), (0.70, "n_lt070")):
            r[key] = int((w[m] < thr).sum())
    else:
        for key in ("wall_min_distal", "wall_min_distal_sE", "wall_med_distal",
                    "wall_tip", "distal_len"):
            r[key] = float("nan")
        r["n_lt030"] = r["n_lt070"] = 0
    # margin-2mm version: min wall at least 2 mm beyond the carina
    m2 = sE > r["carina_s"] + 2.0
    r["wall_min_c2"] = float(w[m2].min()) if m2.any() else float("nan")
    r["wall_min_c2_sE"] = float(sE[m2][int(np.argmin(w[m2]))]) if m2.any() else float("nan")

    # RCCA vs RVA, whole length, mesh wall (RVA is the other branch off the RCCA origin)
    V, rV, sV = bd["- RVA"]
    selc = sC >= 20.0
    wv, Lv, _ = wall_profile(mesh, C[selc][::8], sC[selc][::8], V)
    zv = runs_of_zero(wv, sC[selc][::8])
    r["rcca_rva_zero_runs"] = [(round(a, 2), round(b, 2), n) for a, b, n in zv]
    r["rcca_rva_wall_min"] = float(wv.min())
    rows.append(r)
    if (ai + 1) % 25 == 0:
        print("  ...%d/%d" % (ai + 1, len(names)))
        sys.stdout.flush()

json.dump(rows, open("/tmp/out/check8_wall2.json", "w"), indent=0, default=float)


def dist(vals, label, fmt="%.3f"):
    v = np.array([x for x in vals if x == x], float)
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    print(("%-44s n=%3d  min/p5/p25/med/p75/p95/max = " + " / ".join([fmt] * 7))
          % ((label, len(v)) + tuple(q)))


print("")
print("=" * 122)
print("CONTROLS -- sign of vtkImplicitPolyDataDistance")
print("=" * 122)
print("  sign control failed on: %s" % (ctrl_fail if ctrl_fail else "none, 0/216"))
dist([r["ctrl_in_med"] for r in rows], "median signed dist at RCCA centerline")
dist([r["ctrl_out"] for r in rows], "signed dist 200 mm outside bbox", "%.1f")

print("")
print("=" * 122)
print("A. THE BIFURCATION -- where the shared lumen ends, measured on the mesh")
print("=" * 122)
dist([r["carina_s"] for r in rows], "carina: RECA s where wall first appears")
dist([r["L_eca"] for r in rows], "RECA total length", "%.2f")
dist([r["distal_len"] for r in rows], "RECA length distal to the carina", "%.2f")
print("  anatomies whose fork zero-wall run does NOT start at RECA s=0: %d"
      % sum(1 for r in rows if r["fork_zero_starts_at"] > 0.001))
print("  anatomies with NO zero-wall interval at all (no shared fork lumen): %d"
      % sum(1 for r in rows if r["n_zero_runs"] == 0))

print("")
print("=" * 122)
print("B. FUSION TEST -- wall thickness DISTAL to the carina")
print("=" * 122)
dist([r["wall_min_distal"] for r in rows], "min wall distal to carina (mm)")
dist([r["wall_min_c2"] for r in rows], "min wall, carina + 2 mm margin (mm)")
dist([r["wall_med_distal"] for r in rows], "median wall distal to carina (mm)")
dist([r["wall_tip"] for r in rows], "wall at the RECA distal tip (mm)")
print("  anatomies with a SECOND zero-wall interval (re-fusion): %d"
      % sum(1 for r in rows if r["extra_zero_runs"]))
for r in rows:
    if r["extra_zero_runs"]:
        print("     %-42s carina=%6.2f extra=%s L_eca=%5.2f"
              % (r["name"], r["carina_s"], r["extra_zero_runs"], r["L_eca"]))
print("  anatomies with min distal wall < 0.30 mm (SOFA contactDistance): %d"
      % sum(1 for r in rows if r["wall_min_distal"] < 0.30))
print("  anatomies with min distal wall < 0.70 mm (catheter OD): %d"
      % sum(1 for r in rows if r["wall_min_distal"] < 0.70))
print(" 15 thinnest distal walls:")
for r in sorted(rows, key=lambda r: r["wall_min_distal"])[:15]:
    print("  %-42s wall=%6.3f at RECA s=%6.2f | carina=%6.2f L_eca=%5.2f n<0.3=%d n<0.7=%d wall_c2=%6.3f"
          % (r["name"], r["wall_min_distal"], r["wall_min_distal_sE"], r["carina_s"],
             r["L_eca"], r["n_lt030"], r["n_lt070"], r["wall_min_c2"]))
print(" 5 thickest:")
for r in sorted(rows, key=lambda r: -r["wall_min_distal"])[:5]:
    print("  %-42s wall=%6.3f  carina=%6.2f" % (r["name"], r["wall_min_distal"], r["carina_s"]))

print("")
print("=" * 122)
print("C. RCCA vs RVA on the mesh (RCCA s>=20 mm, 1 mm stations)")
print("=" * 122)
dist([r["rcca_rva_wall_min"] for r in rows], "min RCCA-RVA wall (mm)")
print("  anatomies with a zero-wall RCCA-RVA station beyond s=20: %d"
      % sum(1 for r in rows if r["rcca_rva_zero_runs"]))
for r in sorted(rows, key=lambda r: r["rcca_rva_wall_min"])[:8]:
    print("  %-42s wall=%6.3f  zero_runs=%s"
          % (r["name"], r["rcca_rva_wall_min"], r["rcca_rva_zero_runs"][:3]))
