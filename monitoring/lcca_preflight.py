"""Pre-flight for the LCCA different-vessel transfer experiment.

V0 (geometry, no SOFA) + V1 (surface clearance against the PINNED original mesh).
Both must pass before any episode is run. Every assertion here corresponds to a
failure mode that otherwise produces a run which looks perfect and measures the
wrong vessel.

usage:  python3 lcca_preflight.py [insert_idx]     (default 2)
"""
import sys
import numpy as np

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")

from eve_bench.dualdevicenav import DualDeviceNav

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
RCCA = "Centerline curve - RCCA.mrk"
LCCA = "Centerline curve - LCCA.mrk"
BRIDGE = "(11)"
SHORTEST_RCCA_TASK_MM = 73.8      # RCCA path_len floor under min_arclength=40
GW_R, CATH_R = 0.18, 0.35

ok = True


def check(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        ok = False


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


print("=" * 78)
print("V0 — GEOMETRY (no SOFA)")
print("=" * 78)

iv = DualDeviceNav()
vt = iv.vessel_tree
B = {str(b.name): b for b in vt.branches}
bridge_name = next(n for n in B if BRIDGE in n)

lc = np.asarray(B[LCCA].coordinates, float)
lr = np.asarray(B[LCCA].radii, float)
ls = arclen(lc)
rc = np.asarray(B[RCCA].coordinates, float)
rr = np.asarray(B[RCCA].radii, float)
rs = arclen(rc)
bc = np.asarray(B[bridge_name].coordinates, float)

# -- the premise the index choice rests on -------------------------------------
d0 = float(np.linalg.norm(lc[0] - bc[0]))
check("LCCA[0] is bit-identical to (11)[0] (shared arch junction)", d0 < 1e-6,
      f"||diff|| = {d0:.6f} mm")

# -- the chosen insertion ------------------------------------------------------
s_ins, r_ins = float(ls[IDX]), float(lr[IDX])
print(f"\n  LCCA insertion candidate: idx={IDX}  s={s_ins:.2f} mm  r={r_ins:.2f} mm")
print(f"    position {lc[IDX].round(2)}")
d = lc[IDX + 1] - lc[IDX]
print(f"    direction {(d / np.linalg.norm(d)).round(4)}")
check("insertion is past the aortic arch (r < 8 mm)", r_ins < 8.0, f"r = {r_ins:.2f} mm")
check("insertion radius is in the trained band (r < 12 mm rail)", r_ins < 12.0,
      "localguidance clips local_radius at 12 mm; above it features 47/48 rail at 1.0")
check("insertion is not index 0", IDX >= 1,
      "LCCA[0] ties with the arch branch and would plan the path from the aorta")

# -- target pool ---------------------------------------------------------------
min_arc = s_ins + SHORTEST_RCCA_TASK_MM
keep = ls >= min_arc
pl = ls[keep] - s_ins
print(f"\n  min_arclength_from_start = {min_arc:.1f} mm  (= s_ins + {SHORTEST_RCCA_TASK_MM})")
check("target pool is non-empty", keep.sum() > 0, f"{int(keep.sum())} targets")
check("every target is ahead of the insertion", pl.min() > 10.0,
      f"path_len {pl.min():.1f}..{pl.max():.1f} mm")
check("shortest task matches the RCCA experiment", abs(pl.min() - SHORTEST_RCCA_TASK_MM) < 1.0,
      f"{pl.min():.1f} vs {SHORTEST_RCCA_TASK_MM} mm")

# -- the control arm -----------------------------------------------------------
k_r = int(np.searchsorted(rs, s_ins))
k_r = min(max(k_r, 1), len(rc) - 2)
s_r, r_r = float(rs[k_r]), float(rr[k_r])
min_arc_r = s_r + SHORTEST_RCCA_TASK_MM
keep_r = rs >= min_arc_r
pl_r = rs[keep_r] - s_r
print(f"\n  CONTROL ARM (RCCA-internal, matches the LCCA offset):")
print(f"    idx={k_r}  s={s_r:.2f} mm  r={r_r:.2f} mm  min_arc={min_arc_r:.1f} mm")
print(f"    {int(keep_r.sum())} targets, path_len {pl_r.min():.1f}..{pl_r.max():.1f} mm")
lo, hi = max(pl.min(), pl_r.min()), min(pl.max(), pl_r.max())
print(f"    COMMON BAND for the primary comparison: [{lo:.0f}, {hi:.0f}] mm")
check("control-arm offset matches the LCCA arm", abs(s_r - s_ins) < 1.5,
      f"{s_r:.2f} vs {s_ins:.2f} mm")

print(f"\n  radii side by side: LCCA[{IDX}] r={r_ins:.2f}   RCCA[{k_r}] r={r_r:.2f}")
if abs(r_ins - r_r) > 1.5:
    print(f"    NOTE: radii differ by {abs(r_ins - r_r):.2f} mm. Report it; do NOT "
          f"retune k to match, that would break the offset match.")

# -- V1: clearance against the PINNED surface ----------------------------------
print()
print("=" * 78)
print("V1 — CLEARANCE against the pinned original surface")
print("=" * 78)
try:
    import pyvista as pv
    from scipy.spatial import cKDTree

    m = pv.read(vt.mesh_path).triangulate().clean()
    f = m.faces.reshape(-1, 4)[:, 1:]
    v = m.points
    a_, b_, c_ = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    rg = np.random.default_rng(0)
    u = rg.random((24, len(f), 1))
    w = rg.random((24, len(f), 1))
    kk = u + w > 1
    u = np.where(kk, 1 - u, u)
    w = np.where(kk, 1 - w, w)
    surf = np.vstack([v, (a_ + u * (b_ - a_) + w * (c_ - a_)).reshape(-1, 3)])
    tree = cKDTree(surf)

    cl_ins = float(tree.query(lc[IDX])[0])
    cl_rcca_ref = float(tree.query(bc[2])[0])
    print(f"  clearance at LCCA[{IDX}]      : {cl_ins:.2f} mm")
    print(f"  clearance at (11)[2] (RCCA ref): {cl_rcca_ref:.2f} mm")
    check("insertion has clearance for the catheter", cl_ins > max(1.0, CATH_R * 2),
          f"{cl_ins:.2f} mm vs catheter radius {CATH_R}")

    d_all, _ = tree.query(lc)
    blocked = d_all < GW_R
    print(f"\n  LCCA clearance profile: median {np.median(d_all):.2f}  "
          f"p05 {np.percentile(d_all, 5):.2f}  min {d_all.min():.2f} mm")
    check("LCCA is passable end-to-end", blocked.sum() == 0,
          f"{int(blocked.sum())} station(s) below the guidewire radius"
          + (f", first at s={ls[blocked][0]:.1f} mm" if blocked.any() else ""))
    tight = np.where(d_all < 2 * GW_R)[0]
    if len(tight):
        bands, run = [], [tight[0]]
        for i in tight[1:]:
            (run.append(i) if i == run[-1] + 1 else (bands.append(run), run := [i]))
        bands.append(run)
        print("  tight bands (< 2x wire radius), arclength mm: " +
              "; ".join(f"{ls[b[0]]:.0f}-{ls[b[-1]]:.0f}" for b in bands[:8]))
    np.save("/opt/eve_training/results/lcca_clearance_profile.npy",
            np.vstack([ls, d_all]))
    print("  saved clearance profile -> results/lcca_clearance_profile.npy "
          "(bucket arrests against this, not against RCCA depth bands)")
except Exception as e:
    print(f"  SKIPPED — {type(e).__name__}: {e}")
    ok = False

print()
print("=" * 78)
print(f"PREFLIGHT {'PASSED' if ok else 'FAILED'}")
print("=" * 78)
if ok:
    print(f"\n  ARM 1 (LCCA)         --insert_inside_branch LCCA "
          f"--insert_point_idx {IDX} --target_min_arclength_mm {min_arc:.1f}")
    print(f"  ARM 2 (RCCA-internal)--insert_inside_branch RCCA "
          f"--insert_point_idx {k_r} --target_min_arclength_mm {min_arc_r:.1f}")
    print(f"  ARM 3 (RCCA baseline) (no --insert_inside_branch)")
    print(f"\n  Primary comparison: ARM 1 vs ARM 2 on path_len in [{lo:.0f}, {hi:.0f}] mm.")
    print("  NOT against the published RCCA-from-(11) number — that comparison")
    print("  confounds 'different vessel' with 'planned path collapsed to one branch'.")
sys.exit(0 if ok else 1)
