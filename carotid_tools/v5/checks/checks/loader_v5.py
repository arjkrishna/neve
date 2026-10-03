# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\loader_v5.py (= /sp/v5_va/checks/loader_v5.py in the build containers)
"""v5_checks item 4a: the training loader over ALL 223 v5 anatomies, no SOFA (v4 audit sim_loader.py, pointed at
anatomies_v5; stated diffs: V4 -> anatomies_v5; RCCA byte identity checked against v4 AND v3; every non-VA centreline
file byte-identical to v4; RVA / LVA lengths recorded; output /sp/v5_va/checks/loader_v5.json).

For every anatomy, through eve_bench.dualdevicenavtopbrain (repo version):
  * branch names seen by the loader (expect 18: 12 host-numbered + RCCA RECA RVA LCCA LECA LVA)
  * per branch: finite coords, len(radii) == len(coords), radii > 0
  * insertion derived per anatomy == the set's insertion == host's derived insertion;
    host-numbered branches and RCCA[0] identical to the host's
  * the insertion point and every centerline point enclosed by vessel_architecture_collision.obj
    (vtkSelectEnclosedPoints), and the insertion point's clearance to the wall
  * the RCCA target pool beyond 133 mm is non-empty
  * RCCA centerline file byte-identical to v3
Writes /sp/v5_va/checks/loader_v5.json.
"""
import sys, os, json, hashlib, time
import numpy as np
sys.path[:0] = ["/repo/eve", "/repo/eve_bench"]
import eve_bench, eve
print("eve_bench from", eve_bench.__file__, "| eve from", eve.__file__)
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain, find_anatomies
from eve_bench.dualdevicenav import load_branches
from eve.intervention.vesseltree.topbrainanatomyset import TopBrainAnatomySet
import vtk
from vtk.util.numpy_support import vtk_to_numpy

V4 = "/repo/carotid_data/anatomies_v5"
V4REF = "/repo/carotid_data/anatomies_v4"
V3 = "/repo/carotid_data/anatomies_v3"
HOST = "/repo/eve_bench/data/dualdevicenav/Centrelines_comb"
EXPECT = {"LCCA", "LECA", "LVA", "RCCA", "RECA", "RVA"}

t0 = time.time()
iv = DualDeviceNavTopBrain(anatomy_dir=V4, seed=1, episodes_between_change=1,
                           target_min_arclength_mm=133.0)
vt = iv.vessel_tree
print("loader built over %d anatomies in %.1f s" % (vt.anatomy_count, time.time() - t0))
names = list(vt._names)

host = load_branches(HOST)
host_by = {str(b.name): b for b in host}
shim = type("S", (), {})()
shim.rcca_name = vt.rcca_name
shim.start_point_offset = vt.start_point_offset
host_ins = TopBrainAnatomySet._derive_insertion(shim, host)
set_ins = vt._insertion
print("host-derived insertion", np.round(host_ins.position, 4).tolist(), np.round(host_ins.direction, 4).tolist())
print("set insertion         ", np.round(np.asarray(set_ins.position), 4).tolist(), np.round(np.asarray(set_ins.direction), 4).tolist())


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read_obj(p):
    r = vtk.vtkOBJReader(); r.SetFileName(p); r.Update()
    return r.GetOutput()


def wall_distance(poly, pts):
    d = vtk.vtkImplicitPolyDataDistance(); d.SetInput(poly)
    return np.array([d.EvaluateFunction(*map(float, p)) for p in pts])


def enclosed(poly, pts):
    vp = vtk.vtkPoints()
    for p in pts:
        vp.InsertNextPoint(*map(float, p))
    pd = vtk.vtkPolyData(); pd.SetPoints(vp)
    s = vtk.vtkSelectEnclosedPoints(); s.SetInputData(pd); s.SetSurfaceData(poly)
    s.CheckSurfaceOn(); s.SetTolerance(1e-7); s.Update()
    return vtk_to_numpy(s.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


res = {}
bad = []
branch_count_hist = {}
for i, n in enumerate(names):
    vt._select(i)
    br = list(vt.branches)
    bn = [str(b.name) for b in br]
    short = sorted(x.split("- ")[-1].replace(".mrk", "") for x in bn if " - " in x)
    numbered = sorted(x for x in bn if " - " not in x)
    branch_count_hist[len(bn)] = branch_count_hist.get(len(bn), 0) + 1
    prob = []
    if set(short) != EXPECT:
        prob.append("named branches %s" % short)
    if len(bn) != 18:
        prob.append("%d branches" % len(bn))
    for b in br:
        c = np.asarray(b.coordinates); r = np.asarray(b.radii)
        if len(c) != len(r):
            prob.append("%s: %d coords vs %d radii" % (b.name, len(c), len(r)))
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)):
            prob.append("%s: non-finite" % b.name)
        if len(r) and r.min() <= 0:
            prob.append("%s: radius <= 0 (min %.3f)" % (b.name, r.min()))
        if len(c) < 3:
            prob.append("%s: only %d points" % (b.name, len(c)))
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1) if len(c) > 1 else np.zeros(1)
        if seg.max() > 5.0:
            prob.append("%s: point gap %.1f mm" % (b.name, seg.max()))
        if seg.min() == 0.0:
            prob.append("%s: %d duplicate consecutive points" % (b.name, int((seg == 0).sum())))
    # insertion
    ins = TopBrainAnatomySet._derive_insertion(shim, br)
    dpos = float(np.linalg.norm(np.asarray(ins.position) - np.asarray(host_ins.position)))
    ddir = float(np.linalg.norm(np.asarray(ins.direction) - np.asarray(host_ins.direction)))
    if dpos > 1e-6 or ddir > 1e-6:
        prob.append("insertion differs from host by %.4f mm / dir %.4f" % (dpos, ddir))
    # numbered host branches identical to host
    diffs = []
    by = {str(b.name): b for b in br}
    for k in numbered:
        if k not in host_by:
            diffs.append("%s not in host" % k); continue
        a, h = np.asarray(by[k].coordinates), np.asarray(host_by[k].coordinates)
        if a.shape != h.shape or not np.array_equal(a, h) or not np.array_equal(np.asarray(by[k].radii), np.asarray(host_by[k].radii)):
            diffs.append(k)
    if diffs:
        prob.append("numbered branches differ from host: %s" % diffs)
    rcca = by["Centerline curve - RCCA.mrk"]
    if not np.array_equal(np.asarray(rcca.coordinates[0]), np.asarray(host_by["Centerline curve - RCCA.mrk"].coordinates[0])):
        prob.append("RCCA[0] differs from host")
    # RCCA byte identity with v3
    f = "Centrelines_comb/Centerline curve - RCCA.mrk.json"
    rcca_same = md5(os.path.join(V4, n, f)) == md5(os.path.join(V3, n, f))
    if not rcca_same:
        prob.append("RCCA json not byte-identical to v3")
    # diff (v5): every non-VA centreline file byte-identical to v4; the VA files as recorded in provenance
    cdir5, cdir4 = os.path.join(V4, n, "Centrelines_comb"), os.path.join(V4REF, n, "Centrelines_comb")
    f5, f4 = sorted(os.listdir(cdir5)), sorted(os.listdir(cdir4))
    nonva_diff = [x for x in f4 if not (x.endswith("RVA.mrk.json") or x.endswith("LVA.mrk.json"))
                  and (x not in f5 or md5(os.path.join(cdir5, x)) != md5(os.path.join(cdir4, x)))]
    extra5 = [x for x in f5 if x not in f4]
    if nonva_diff or extra5:
        prob.append("non-VA centreline files differ from v4: %s; extra %s" % (nonva_diff, extra5))
    va_same_v4 = {t: md5(os.path.join(cdir5, "Centerline curve - %s.mrk.json" % t)) == md5(os.path.join(cdir4, "Centerline curve - %s.mrk.json" % t))
                  for t in ("RVA", "LVA")}
    pv5 = json.load(open(os.path.join(V4, n, "provenance.json")))["v5_va"]
    for side, t in (("right", "RVA"), ("left", "LVA")):
        repl = bool((pv5.get(side) or {}).get("replaced"))
        if repl == va_same_v4[t]:
            prob.append("%s: replaced=%s but file identical to v4=%s" % (t, repl, va_same_v4[t]))
    # mesh
    poly = read_obj(os.path.join(V4, n, "vessel_architecture_collision.obj"))
    allpts = np.concatenate([np.asarray(b.coordinates) for b in br])
    inside = enclosed(poly, allpts)
    ins_in = bool(enclosed(poly, [ins.position])[0])
    ins_wall = float(abs(wall_distance(poly, [ins.position])[0]))
    if not ins_in:
        prob.append("insertion point OUTSIDE the obj")
    per_branch_out = {}
    k0 = 0
    for b in br:
        m = len(b.coordinates)
        o = int((~inside[k0:k0 + m]).sum())
        if o:
            per_branch_out[str(b.name).split("- ")[-1]] = o
        k0 += m
    if per_branch_out:
        prob.append("centerline pts outside obj: %s" % per_branch_out)
    # targets
    iv.target._branches_initialized = None
    iv.target._init_centerline_point_cloud()
    ntgt = len(iv.target._potential_targets)
    if ntgt == 0:
        prob.append("no RCCA targets beyond 133 mm")
    lens = {x.split("- ")[-1].replace(".mrk", ""): round(float(np.sum(np.linalg.norm(np.diff(np.asarray(by[x].coordinates), axis=0), axis=1))), 1)
            for x in bn if " - " in x}
    res[n] = dict(n_branches=len(bn), named=short, numbered=numbered, insertion_dpos=dpos, rcca_byte_identical_v3=rcca_same,
                  obj_points=poly.GetNumberOfPoints(), obj_tris=poly.GetNumberOfCells(), centerline_pts=int(len(allpts)),
                  centerline_outside=per_branch_out, insertion_inside=ins_in, insertion_wall_mm=ins_wall,
                  n_targets=ntgt, lengths=lens, va_file_identical_v4=va_same_v4, nonva_identical_v4=not (nonva_diff or extra5), problems=prob)
    if prob:
        bad.append(n)
        print("PROBLEM", n, prob)
    if i % 25 == 0:
        print("%3d/%d %s ok=%s ins_wall=%.2f targets=%d lens=%s (%.0f s)" % (i, len(names), n, not prob, ins_wall, ntgt, lens, time.time() - t0), flush=True)

summary = dict(n=len(names), branch_count_hist=branch_count_hist, bad=bad,
               host_insertion=[np.asarray(host_ins.position).tolist(), np.asarray(host_ins.direction).tolist()],
               set_insertion=[np.asarray(set_ins.position).tolist(), np.asarray(set_ins.direction).tolist()],
               insertion_wall_mm_minmax=[min(r["insertion_wall_mm"] for r in res.values()), max(r["insertion_wall_mm"] for r in res.values())],
               targets_min=min(r["n_targets"] for r in res.values()),
               centerline_pts_total=sum(r["centerline_pts"] for r in res.values()))
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, per_anatomy=res), open("/sp/v5_va/checks/loader_v5.json", "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))
