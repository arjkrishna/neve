"""CHECK 8 adjudication of every mesh-flagged fusion.

The segment test can read FUSED falsely if the straight line to the neighbour
happens to pass through a THIRD vessel.  For each flagged station this prints:
  - the exact location and centre distance
  - the declared radii and declared separation
  - the signed-distance profile along the segment (min, and where)
  - the nearest OTHER branch to the segment midpoint, centre distance minus its
    radius: negative => the midpoint sits inside a third lumen => ARTIFACT
"""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
import pyvista as pv
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
STEP = 0.25
NSEG = 401

RVA_FLAG = ["case_k_005_right__topcow_mr_023_L", "case_k_011_left__topcow_mr_012_L",
            "case_k_011_left__topcow_mr_023_L", "case_m_030_right__topcow_mr_016",
            "case_w_007_left__topcow_mr_005", "case_w_007_right__topcow_mr_023",
            "case_w_008_right__topcow_mr_023_L", "case_w_012_left__topcow_mr_020",
            "case_w_017_left__topcow_mr_010", "case_w_017_left__topcow_mr_014_L",
            "case_w_022_right__topcow_mr_011", "case_w_022_right__topcow_mr_024_L",
            "case_w_025_right__topcow_mr_020", "case_w_025_right__topcow_mr_021_L",
            "case_w_027_left__topcow_mr_026", "case_w_036_right__topcow_mr_005",
            "case_w_047_left__topcow_mr_026_L", "case_w_048_left__topcow_mr_008",
            "case_w_048_left__topcow_mr_010_L", "case_w_051_right__topcow_mr_017_L",
            "case_w_052_left__topcow_mr_027_L"]
ECA_FLAG = ["case_m_024_left__topcow_mr_001", "case_m_024_left__topcow_mr_006",
            "case_m_024_left__topcow_mr_013_L", "case_m_024_left__topcow_mr_023_L",
            "case_m_024_left__topcow_mr_016_L", "case_w_029_right__topcow_mr_010_L",
            "case_w_050_left__topcow_mr_004_L", "case_w_050_left__topcow_mr_005",
            "case_w_050_left__topcow_mr_020_L", "case_w_050_left__topcow_mr_021_L",
            "case_w_050_left__topcow_mr_026_L"]
CONTROL = ["case_k_004_left__topcow_mr_010", "case_w_027_right__topcow_mr_008"]


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


def report(nm, probe, target, lo, hi, tag):
    d0 = os.path.join(ROOT, nm)
    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    bd = {short(b.name): densify(np.asarray(b.coordinates, np.float64),
                                 np.asarray(b.radii, np.float64)) for b in brs}
    A, ra, sa = bd[probe]
    B, rb, sb = bd[target]
    sel = np.where((sa >= lo) & (sa <= hi))[0]
    if not len(sel):
        print("  %s: no stations in [%.1f, %.1f]" % (nm, lo, hi))
        return
    for i in sel[::max(1, len(sel) // 4)]:
        k = int(np.argmin(np.linalg.norm(B - A[i], axis=1)))
        P, Q = B[k], A[i]
        Lseg = float(np.linalg.norm(Q - P))
        t = np.linspace(0.0, 1.0, NSEG)
        segpts = P[None, :] + t[:, None] * (Q - P)[None, :]
        v = sd(mesh, segpts)
        mid = 0.5 * (P + Q)
        third = []
        for k2, (X, rx, sx) in bd.items():
            if k2 in (probe, target):
                continue
            d = np.linalg.norm(X - mid, axis=1)
            j = int(np.argmin(d))
            third.append((float(d[j] - rx[j]), k2, float(sx[j]), float(d[j])))
        third.sort()
        gdecl = Lseg - float(ra[i]) - float(rb[k])
        print("  %-40s %-6s s=%7.2f -> %-6s s=%7.2f | ctr_d=%6.3f rA=%.2f rB=%.2f "
              "g_decl=%7.3f | sd_min=%7.3f at t=%.2f frac_out=%.3f | 3rd: %-6s d-r=%7.3f  [%s]"
              % (nm, probe, sa[i], target, sb[k], Lseg, ra[i], rb[k], gdecl,
                 float(v.min()), float(t[int(np.argmin(v))]), float((v < 0).mean()),
                 third[0][1], third[0][0], tag))


print("=" * 150)
print("RCCA vs RVA -- mesh reported zero wall")
print("=" * 150)
wall2 = json.load(open("/tmp/out/check8_wall2.json"))
byname = {r["name"]: r for r in wall2}
for nm in RVA_FLAG:
    runs = byname[nm]["rcca_rva_zero_runs"]
    for a, b, n in runs[:1]:
        report(nm, "- RCCA", "- RVA", a - 0.01, b + 0.01, "flagged")

print("")
print("=" * 150)
print("RECA vs RCCA -- mesh reported a SECOND zero-wall interval distal to the carina")
print("=" * 150)
for nm in ECA_FLAG:
    r = byname[nm]
    runs = r["extra_zero_runs"]
    if runs:
        a, b, n = runs[-1]
        report(nm, "- RECA", "- RCCA", a - 0.01, b + 0.01, "refusion")
    else:
        s = r["wall_min_distal_sE"]
        report(nm, "- RECA", "- RCCA", s - 0.01, s + 0.01, "near-zero")

print("")
print("=" * 150)
print("CONTROLS -- anatomies the scan reported as CLEAN, same code path")
print("=" * 150)
for nm in CONTROL:
    r = byname[nm]
    s = r["wall_min_distal_sE"]
    report(nm, "- RECA", "- RCCA", s - 0.01, s + 0.01, "clean-ctrl")
    report(nm, "- RCCA", "- RVA", 170.0, 172.0, "clean-ctrl")
