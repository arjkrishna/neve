"""Audit the TopCoW anatomies against the host patient.

Two independent questions, both of which have burned this project before:

  1. MESH QUALITY — are these decimated/reconstructed to the same extent as the
     host patient's collision surface? Triangle count and edge length relative to
     vessel calibre is what determines whether the surface is a faithful tube or
     a coarse polygon.
  2. PASSABILITY — can the 0.36 mm guidewire physically traverse each navigated
     branch? Clearance is measured to the nearest point ON THE TRIANGLE SURFACE,
     densely sampled. Nearest-VERTEX distance is NOT usable: on a ~6 mm-triangle
     mesh the vertices sit outside the facets and it reported 1% erosion where the
     true figure was ~45%.

Reference is the host patient: eve_bench/data/dualdevicenav/.

usage: audit_topbrain.py [branch]        branch default RCCA
"""
import glob
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches  # noqa: E402

WIRE_R = 0.18                      # guidewire radius, mm
CATH_R = 0.35                      # catheter radius, mm
HOST = "/opt/eve_training/eve_bench/data/dualdevicenav"
ROOT = "/opt/eve_training/results_topbrain/anatomies"
WANT = (sys.argv[1] if len(sys.argv) > 1 else "RCCA").upper()


def dense_surface(mesh, per_tri=20):
    """Points sampled across triangle FACES, not just vertices."""
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    rg = np.random.default_rng(0)
    u = rg.random((per_tri, len(f), 1))
    w = rg.random((per_tri, len(f), 1))
    k = u + w > 1
    u = np.where(k, 1 - u, u)
    w = np.where(k, 1 - w, w)
    return np.vstack([v, (a + u * (b - a) + w * (c - a)).reshape(-1, 3)])


def edge_stats(mesh):
    ed = mesh.extract_all_edges()
    L = ed.lines.reshape(-1, 3)
    e = np.linalg.norm(ed.points[L[:, 1]] - ed.points[L[:, 2]], axis=1)
    return np.median(e), np.percentile(e, 90), e.max()


def audit(obj_path, cl_dir, label):
    m = pv.read(obj_path).triangulate().clean()
    e_med, e_p90, e_max = edge_stats(m)
    branches = load_branches(cl_dir)
    names = [str(b.name) for b in branches]
    hit = [b for b in branches if WANT in str(b.name).upper()]
    if not hit:
        return dict(label=label, ok=False, why=f"no {WANT} branch", names=names,
                    cells=m.n_cells, e_med=e_med, e_p90=e_p90, e_max=e_max)
    br = hit[0]
    c = np.asarray(br.coordinates, float)
    r = np.asarray(br.radii, float)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1)))) \
        if len(c) > 1 else np.array([0.0])
    d, _ = cKDTree(dense_surface(m)).query(c)
    blocked = int((d < WIRE_R).sum())
    # edge length relative to the vessel it represents -- the quantity that
    # decides whether the surface is a tube or a polygon
    dist_r = float(np.median(r[s > 0.5 * s[-1]])) if len(r) > 2 else float(np.median(r))
    return dict(
        label=label, ok=True, cells=m.n_cells, pts=m.n_points,
        e_med=e_med, e_p90=e_p90, e_max=e_max,
        n_cl=len(c), length=float(s[-1]),
        r_med=float(np.median(r)), r_distal=dist_r,
        cl_med=float(np.median(d)), cl_p05=float(np.percentile(d, 5)),
        cl_min=float(d.min()), blocked=blocked,
        first_block=float(s[d < WIRE_R][0]) if blocked else None,
        edge_per_diam=e_med / (2 * dist_r) if dist_r > 0 else float("nan"),
        cl_over_r=float(np.median(d)) / float(np.median(r)) if np.median(r) > 0 else float("nan"),
        e_irreg=e_max / e_med if e_med > 0 else float("nan"),
        names=names,
    )


print("=" * 118)
print(f"TOPBRAIN ANATOMY AUDIT — navigated branch = {WANT}")
print("=" * 118)

from eve_bench.dualdevicenav import DualDeviceNav  # noqa: E402
_host_iv = DualDeviceNav()   # FromMesh writes the .obj rotated INTO the branch frame
ref = audit(_host_iv.vessel_tree.mesh_path,
            os.path.join(HOST, "Centrelines_comb"), "HOST PATIENT")
print(f"\nREFERENCE (host patient):")
print(f"  mesh   {ref['cells']:6d} cells   edge med {ref['e_med']:5.2f}  p90 {ref['e_p90']:5.2f}  max {ref['e_max']:6.2f} mm")
print(f"  branch {ref['n_cl']:4d} pts  {ref['length']:6.1f} mm   r_med {ref['r_med']:4.2f}  r_distal {ref['r_distal']:4.2f} mm")
print(f"  clear  med {ref['cl_med']:5.2f}  p05 {ref['cl_p05']:5.2f}  min {ref['cl_min']:5.2f} mm   blocked {ref['blocked']}")
print(f"  edge/diameter ratio {ref['edge_per_diam']:5.2f}   (>1 means triangles exceed the vessel width)")

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(d):
        continue
    obj = os.path.join(d, "vessel_architecture_collision.obj")
    cl = os.path.join(d, "Centrelines_comb")
    if not (os.path.exists(obj) and os.path.isdir(cl)):
        print(f"  [SKIP] {os.path.basename(d)} — missing obj or centerlines")
        continue
    try:
        rows.append(audit(obj, cl, os.path.basename(d)))
    except Exception as ex:
        print(f"  [FAIL] {os.path.basename(d)} — {type(ex).__name__}: {ex}")

good = [r for r in rows if r["ok"]]
print(f"\n{'anatomy':16s} {'cells':>6} {'edge':>5} {'e/diam':>6} {'len':>6} "
      f"{'e_p90':>6} {'e_max':>6} {'irreg':>6} "
      f"{'r_med':>6} {'cl_med':>6} {'cl/r':>6} {'cl_p05':>6} {'cl_min':>6} {'blk':>4}")
print("-" * 118)
print(f"{'HOST PATIENT':16s} {ref['cells']:6d} {ref['e_med']:5.2f} {ref['edge_per_diam']:6.2f} "
      f"{ref['length']:6.1f} {ref['e_p90']:6.2f} {ref['e_max']:6.2f} {ref['e_irreg']:6.2f} "
      f"{ref['r_med']:6.2f} {ref['cl_med']:6.2f} {ref['cl_over_r']:6.2f} {ref['cl_p05']:6.2f} "
      f"{ref['cl_min']:6.2f} {ref['blocked']:4d}")
print("-" * 118)
for r in good:
    flag = ""
    if r["blocked"]:
        flag = f"  <-- BLOCKED @{r['first_block']:.0f}mm"
    elif r["cl_min"] < CATH_R:
        flag = "  <-- catheter marginal"
    print(f"{r['label']:16s} {r['cells']:6d} {r['e_med']:5.2f} {r['edge_per_diam']:6.2f} "
          f"{r['length']:6.1f} {r['e_p90']:6.2f} {r['e_max']:6.2f} {r['e_irreg']:6.2f} "
          f"{r['r_med']:6.2f} {r['cl_med']:6.2f} {r['cl_over_r']:6.2f} {r['cl_p05']:6.2f} "
          f"{r['cl_min']:6.2f} {r['blocked']:4d}{flag}")

bad = [r for r in rows if not r["ok"]]
for r in bad:
    print(f"{r['label']:16s} {r['cells']:6d} {r['e_med']:5.2f}    --     --     --  "
          f"    --     --     --   --  <-- {r['why']}")

if good:
    def col(k):
        return np.array([r[k] for r in good], float)
    print("\n" + "=" * 118)
    print("SUMMARY vs HOST PATIENT")
    print("=" * 118)
    for k, name, hostv in (("cells", "mesh cells", ref["cells"]),
                           ("e_med", "edge median (mm)", ref["e_med"]),
                           ("edge_per_diam", "edge/diameter", ref["edge_per_diam"]),
                           ("cl_med", "clearance median (mm)", ref["cl_med"]),
                           ("cl_p05", "clearance p05 (mm)", ref["cl_p05"]),
                           ("cl_min", "clearance min (mm)", ref["cl_min"]),
                           ("length", "branch length (mm)", ref["length"]),
                           ("e_max", "edge max (mm)", ref["e_max"]),
                           ("e_irreg", "edge max/median", ref["e_irreg"]),
                           ("cl_over_r", "clearance / stated r", ref["cl_over_r"])):
        v = col(k)
        print(f"  {name:24s} host {hostv:8.2f}   cohort  min {v.min():7.2f}  "
              f"median {np.median(v):7.2f}  max {v.max():7.2f}")
    nb = sum(1 for r in good if r["blocked"])
    print(f"\n  PASSABLE (0 blocked stations): {len(good) - nb}/{len(good)}")
    if nb:
        print(f"  BLOCKED: " + ", ".join(f"{r['label']}(@{r['first_block']:.0f}mm)"
                                         for r in good if r["blocked"]))
    tight = [r for r in good if r["cl_min"] < CATH_R and not r["blocked"]]
    if tight:
        print(f"  guidewire fits but CATHETER marginal (min < {CATH_R} mm): "
              + ", ".join(r["label"] for r in tight))
    print(f"\n  Host clearance median {ref['cl_med']:.2f} mm. Cohort anatomies BELOW it "
          f"(i.e. tighter than the host): "
          f"{sum(1 for r in good if r['cl_med'] < ref['cl_med'])}/{len(good)}")
    if bad:
        print(f"\n  NO {WANT} BRANCH: " + ", ".join(r["label"] for r in bad))
        print(f"    branch names in the first such case: {bad[0]['names']}")
