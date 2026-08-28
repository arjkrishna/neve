import os, json, sys
import numpy as np

RCCA = "Centerline curve - RCCA.mrk"
EXCL = ["topcow_mr_013", "topcow_mr_014", "topcow_mr_015"]
MIN_ARC = 40.0

from eve.intervention.target.centerlinerandom import CenterlineRandom
from eve.pathfinder.fixedpath import FixedPathfinder
from eve.intervention.vesseltree import find_nearest_branch_to_point


class Shim:
    def __init__(self, vt):
        self.vessel_tree = vt
        self.fluoroscopy = None
        self.target = None


def cum(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def derive_insertion(branches, rcca_name=RCCA):
    rcca = None
    for b in branches:
        if b.name == rcca_name or "RCCA" in str(b.name).upper():
            rcca = b; break
    ost = np.asarray(rcca.coordinates[0], dtype=np.float64)
    bridge = None
    for b in branches:
        if "(11)" in str(getattr(b, "name", "")):
            bridge = np.asarray(b.coordinates, dtype=np.float64); break
    if bridge is not None and len(bridge) >= 3:
        if np.linalg.norm(bridge[0]-ost) < np.linalg.norm(bridge[-1]-ost):
            bridge = bridge[::-1]
        k = int(min(max(1, 2), len(bridge)-2))
        return bridge[k]
    return ost


def pool_for(vt, min_arc):
    """Exactly reproduce CenterlineRandom's pool, and return (pool, rcca_idx)."""
    t = CenterlineRandom(vessel_tree=vt, fluoroscopy=None, threshold=5,
                         branches=[RCCA], min_arclength_from_start=min_arc)
    t._init_centerline_point_cloud()
    pool = np.asarray(t._potential_targets, dtype=np.float64)
    # reconstruct index mask ourselves and cross-check
    rc = np.asarray(vt[RCCA].coordinates, dtype=np.float64)
    s = cum(rc)
    arc_ok = s >= min_arc
    from eve.intervention.vesseltree.util.branch import BranchWithRadii
    excl = [b for b in vt.branches if str(b.name) != RCCA]
    inb = np.zeros(len(rc), dtype=bool)
    for b in excl:
        if isinstance(b, BranchWithRadii):
            inb |= np.asarray(b.in_branch(rc), dtype=bool)
    mask = arc_ok & (~inb)
    assert mask.sum() == len(pool), (mask.sum(), len(pool))
    assert np.allclose(rc[mask], pool)
    return pool, np.where(mask)[0], s, arc_ok, inb


def path_lengths(vt, ins, pts):
    pf = FixedPathfinder(Shim(vt))
    pf._init_vessel_tree()
    pf._root_branch = vt.branches[0]
    sb = find_nearest_branch_to_point(ins, vt)
    out = []
    for p in pts:
        tb = find_nearest_branch_to_point(p, vt)
        _, L, _, _ = pf._get_shortest_path_dijkstra(sb, tb, ins, p)
        out.append(float(L))
    return np.asarray(out), str(sb.name)


def signed_dist(mesh_path, pts):
    """+ inside, - outside. vtkImplicitPolyDataDistance is -inside for outward
    normals; sign is fixed empirically using a deep interior probe."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkOBJReader(); r.SetFileName(mesh_path); r.Update()
    poly = r.GetOutput()
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(poly)
    d = np.array([imp.EvaluateFunction(*map(float, p)) for p in pts])
    return d, poly.GetNumberOfPoints(), poly.GetNumberOfCells()


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


res = {}

# ---------------- HOST ----------------
from eve_bench.dualdevicenav import DualDeviceNav
host = DualDeviceNav()
hvt = host.vessel_tree
host_mesh = hvt.mesh_path
host_visu = getattr(hvt, "visu_mesh_path", None)
hbr = list(hvt.branches)
hins = derive_insertion(hbr)
hrc = np.asarray(hvt[RCCA].coordinates, dtype=np.float64)
print("HOST branches:", [str(b.name) for b in hbr])
print("HOST RCCA n=%d total_s=%.2f  spacing med=%.3f" % (
    len(hrc), cum(hrc)[-1], np.median(np.diff(cum(hrc)))))
print("HOST insertion", hins, "mesh", host_mesh, "visu", host_visu)

hpool, hidx, hs, harc, hinb = pool_for(hvt, MIN_ARC)
hpl, hsb = path_lengths(hvt, hins, hpool)
hd, hnp, hnc = signed_dist(host_mesh, hpool)
hdv, _, _ = signed_dist(host_visu, hpool) if host_visu and os.path.exists(host_visu) else (np.array([]), 0, 0)

res["host"] = dict(n_rcca=len(hrc), s_total=float(hs[-1]),
                   n_after_arc=int(harc.sum()), n_excl_by_branch=int((harc & hinb).sum()),
                   n_pool=len(hpool), s_pool=hs[hidx].tolist(),
                   path_len=hpl.tolist(), sd=hd.tolist(),
                   sd_visu=hdv.tolist(), start_branch=hsb,
                   mesh_cells=hnc)
print("HOST pool=%d  s %.1f..%.1f  path_len %.1f..%.1f  sd_min %.3f" % (
    len(hpool), hs[hidx].min(), hs[hidx].max(), hpl.min(), hpl.max(), hd.min()))
sys.stdout.flush()

# ---------------- COHORT ----------------
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
iv = DualDeviceNavTopBrain(anatomy_dir="/opt/eve_training/results_topbrain/anatomies",
                           seed=42, episodes_between_change=1, exclude=EXCL)
vt = iv.vessel_tree
names = list(vt.anatomy_names)
print("cohort n=%d" % len(names), names)
ins_c = None
for i, nm in enumerate(names):
    vt._select(i)
    if ins_c is None:
        ins_c = np.asarray(vt.insertion.position, dtype=np.float64)
        print("cohort insertion", ins_c, "host insertion delta",
              float(np.linalg.norm(ins_c - hins)))
    rc = np.asarray(vt[RCCA].coordinates, dtype=np.float64)
    s = cum(rc)
    # divergence from host RCCA
    n = min(len(rc), len(hrc))
    dif = np.linalg.norm(rc[:n] - hrc[:n], axis=1)
    j = np.argmax(dif > 1e-9) if np.any(dif > 1e-9) else n
    s_div = float(s[j]) if j < len(s) else float(s[-1])
    pool, idx, s_all, arc_ok, inb = pool_for(vt, MIN_ARC)
    pl, sb = path_lengths(vt, ins_c, pool)
    d, npnt, ncell = signed_dist(vt.mesh_path, pool)
    res[nm] = dict(n_rcca=len(rc), s_total=float(s[-1]), s_div=s_div, div_idx=int(j),
                   n_after_arc=int(arc_ok.sum()),
                   n_excl_by_branch=int((arc_ok & inb).sum()),
                   n_pool=len(pool), s_pool=s_all[idx].tolist(),
                   path_len=pl.tolist(), sd=d.tolist(), start_branch=sb,
                   mesh_cells=ncell)
    sp = s_all[idx]
    print("%-14s pool=%3d s %.1f..%.1f  s_div=%.1f  frac<130=%.3f frac<s_div=%.3f  "
          "pl %.1f..%.1f  sd min %.3f  n_out=%d" % (
        nm, len(pool), sp.min(), sp.max(), s_div,
        float((sp < 130.0).mean()), float((sp < s_div).mean()),
        pl.min(), pl.max(), d.min(), int((d > 0).sum() if d.mean() < 0 else (d < 0).sum())))
    sys.stdout.flush()

with open("/scratch/tgt_dist.json", "w") as f:
    json.dump(res, f)
print("WROTE json")
