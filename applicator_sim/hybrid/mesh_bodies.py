"""Stage 1 MESHING for the hybrid pelvis simulator (CONTRACT.md section 1).

preBT labels -> six voxel-exclusive bodies (priority corpus > cervix > vagina > bladder > rectum > sigmoid, largest
6-connected component of each) -> closed, smoothed surfaces (marching cubes in preBT world RAS mm + windowed-sinc,
the label->surface code of prep_inputs.py, then an isotropic ACVD remesh at the target edge instead of quadric
decimation) -> TetGen tetrahedra (+ a local sliver fix so that every dihedral > 10 deg) -> named node sets ->
    <out>/hybrid/meshes/<body>/{surface.obj, tets.vtk, meta.json}   and   <out>/hybrid/meshes/bodies.json

Host commands (details in README.md, section MESHING):
    python   hybrid/mesh_bodies.py build --pk <dir holding the tetgen+pyacvd wheels>  # py3.13: nibabel/scipy/skimage/vtk
    py -3.11 hybrid/mesh_bodies.py render                                            # pyvista off-screen -> hybrid/figs
    bash run_docker.sh MESHCHK hybrid/mesh_bodies.py sofacheck                       # optional: SOFA MeshVTKLoader reads
                                                                                     # every tets.vtk (format check)
Frame: preBT world RAS mm (nibabel affine of the preBT labels; x=R, y=A, z=S). Units mm, cc for volumes.
No patient-derived coordinates live in this file: the canal polyline, the internal os and L_end are read from
MRI_GYN_sim/inputs/canal.npz (written by prep_inputs.py) and every body axis is computed from the labels.
The tet route is TetGen (pyvista's `tetgen` wheel, installed with `pip install --target <dir> tetgen pyacvd` and added
to sys.path); the SOFA MeshTetraStuffing fallback of the contract was not needed (see README).
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True                  # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import config  # noqa: E402
import geom  # noqa: E402

BODIES = ["corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"]         # priority order (1 = highest)
SOURCE = dict(corpus="uterus", cervix="HR-CTV", vagina="vagina", bladder="bladder", rectum="rectum", sigmoid="sigmoid")
ROLE = dict(corpus="rigid, kinematic (attached to the tandem by the pose rule)",
            cervix="deformable (HR-CTV minus uterus; the tandem passes through it along the preBT canal)",
            vagina="deformable (collapsed tube; lower 15 mm fixed)",
            bladder="deformable, near-incompressible (filled)",
            rectum="deformable (cut ends fixed)", sigmoid="deformable (cut ends fixed)")
COLOR = dict(corpus=(0.85, 0.30, 0.30), cervix=(0.95, 0.60, 0.20), vagina=(0.75, 0.45, 0.80),
             bladder=(0.95, 0.85, 0.30), rectum=(0.35, 0.55, 0.85), sigmoid=(0.35, 0.75, 0.55))
FRAME = "preBT world RAS mm (nibabel affine of preBT_MRI_label_*.nii; x=R, y=A, z=S)"

# Every choice not fixed by the contract lives here (README, MESHING > decisions).
CFG = dict(
    connectivity=6,                                   # largest component: 6-connected (drops diagonal-only strays)
    edge_mm=dict(corpus=3.5, cervix=3.5, vagina=2.0, bladder=4.5, rectum=3.5, sigmoid=3.5),   # target edge per body
    vol_tol_pct=2.0,                                  # surface volume vs label (contract: 2 %)
    surface_iters=3,                                  # remesh retries (cluster count +12 %) before the thin-sheet fallback
    thin_fallback="opening",                          # if the raw remesh is never manifold: 6-connected radius-1 opening of
                                                      # the body mask (drops sheets <= 2 voxels thick), largest component,
                                                      # then surface_iters more tries ("none" = fail instead)
    volume_offset_cap_mm=0.3,                         # max uniform normal offset restoring the (pre-opening) label volume
    tet=dict(minratio=1.4, mindihedral=10.0, nobisect=True, maxvolume_factor=1.5,       # -pq1.4/10 -Y -a(1.5 * V_regular)
             opt_scheme=7, opt_iterations=10, optmaxdihedral=165.0,                     # -O7/10 optimiser, slivers targeted
             sliver_target_deg=11.0, sliver_max_boundary_move_mm=0.3, sliver_max_passes=30),   # local fix after TetGen
    sets=dict(interface_corpus_mm=1.5, canal_mm=3.0, os_level_mm=6.0, os_lateral_min_mm=10.0,
              vagina_fixed_inferior_mm=15.0, vagina_apex_mm=3.0, fixed_ends_mm=4.0, end_slab_mm=12.0, junction_max_mm=15.0,
              support_fraction=1.0 / 3.0, bladder_support_dir=[0.0, 1.0, 1.0],          # bladder: cap = outer third along +y+z
              rectum_support_dir=[0.0, -1.0, 0.0], support_cone_deg=60.0),               # rectum: wall = normals within 60 deg of -y
)


def paths():
    P = config.paths()
    hyb = P["out"] + "/hybrid"
    return dict(data=P["data"], inputs=P["inputs"], hybrid=hyb, meshes=hyb + "/meshes", figs=hyb + "/figs",
                logs=hyb + "/logs")


def v2w(A, ijk):
    return np.atleast_2d(ijk).astype(float) @ A[:3, :3].T + A[:3, 3]


# ============================================================================ tet-mesh helpers (numpy only)
# outward faces of a positively oriented tet (a,b,c,d): (a,c,b) (a,b,d) (a,d,c) (b,c,d)
OUT_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
DIHEDRAL_PAIRS = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def tet_volumes(P, T):
    X = P[T]
    return np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0]) / 6.0


def orient_tets(P, T):
    """Flip tets with negative signed volume (TetGen normally returns positive ones). Returns T, n_flipped."""
    v = tet_volumes(P, T)
    neg = v < 0
    T = T.copy()
    T[neg] = T[neg][:, [0, 2, 1, 3]]
    return T, int(neg.sum())


def dihedral_angles(P, T):
    """(m, 6) interior dihedral angles in degrees."""
    return _dihedrals(P[T])


def _dihedrals(X):
    """X (m,4,3) tet corner coordinates -> (m, 6) interior dihedral angles in degrees."""
    out = []
    for a, b, c, d in DIHEDRAL_PAIRS:
        n1 = np.cross(X[:, b] - X[:, a], X[:, c] - X[:, a])
        n2 = np.cross(X[:, b] - X[:, a], X[:, d] - X[:, a])
        n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-300)
        n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-300)
        out.append(np.degrees(np.arccos(np.clip(-(n1 * n2).sum(1), -1.0, 1.0))))
    return np.stack(out, 1)


def _min_dihedral_and_volume(X):
    """X (m,4,3) -> (min dihedral per tet in deg, signed volume per tet)."""
    vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0]) / 6.0
    return _dihedrals(X).min(1), vol


# 14 trial directions (axes + cube diagonals) for the local sliver fix
_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                  [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], float)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def fix_slivers(P, T, target_deg=11.0, max_boundary_move_mm=0.3, max_passes=30):
    """Raise the minimum dihedral of every tet above target_deg by greedy local vertex relocation.

    TetGen's -q.../mindihedral is a request, not a guarantee: a few boundary slivers (~8 deg) always remain.  For each
    offending tet, its nodes (interior first) are moved to the trial position (14 directions + the opposite-face normal,
    5 step sizes relative to the local edge length) that maximises the minimum dihedral of the node's whole star, while
    keeping every star tet positive.  Boundary nodes may move at most max_boundary_move_mm from their input position
    (the surface stays well inside one voxel).  A tet that still fails and has >= 2 boundary faces is deleted (a
    surface diagonal flip / cap removal that keeps the boundary closed).  Returns P, T, log."""
    P = P.copy(); T = T.copy(); P0 = P.copy()
    log = dict(target_deg=target_deg, passes=0, moved_interior=0, moved_boundary=0, deleted_tets=0, dropped_nodes=0)
    for pas in range(max_passes):
        BF = boundary_faces(T)
        is_bnd = np.zeros(len(P), bool); is_bnd[np.unique(BF)] = True
        bkey = {tuple(sorted(f)) for f in BF}
        inc = [[] for _ in range(len(P))]
        for k, t in enumerate(T):
            for v in t:
                inc[v].append(k)
        dmin, _ = _min_dihedral_and_volume(P[T])
        if pas == 0:
            log["start_min_dihedral_deg"] = round(float(dmin.min()), 3); log["start_tets_below_target"] = int((dmin < target_deg).sum())
        bad = np.nonzero(dmin < target_deg)[0]
        if len(bad) == 0:
            break
        log["passes"] = pas + 1
        changed = False
        for k in bad:
            t = T[k]
            if _min_dihedral_and_volume(P[T[[k]]])[0][0] >= target_deg:
                continue                                              # already fixed by an earlier move in this pass
            done = False
            for v in sorted(t, key=lambda u: (bool(is_bnd[u]), -len(inc[u]))):
                star = np.array(inc[v]); Ts = T[star]; hit = (Ts == v)
                ell = float(np.mean([np.linalg.norm(P[Ts[:, i]] - P[Ts[:, j]], axis=1).mean() for i, j in EDGES]))
                X0 = P[Ts]; q0, v0 = _min_dihedral_and_volume(X0)
                best_q, best_p = float(q0.min()), None
                others = [u for u in t if u != v]
                nrm = np.cross(P[others[1]] - P[others[0]], P[others[2]] - P[others[0]])
                nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-12)
                for frac in (0.05, 0.1, 0.2, 0.35, 0.5):
                    for d in np.vstack([_DIRS, nrm, -nrm]):
                        p = P[v] + frac * ell * d
                        if is_bnd[v] and np.linalg.norm(p - P0[v]) > max_boundary_move_mm:
                            continue
                        X = X0.copy(); X[hit] = p
                        q, vv = _min_dihedral_and_volume(X)
                        if vv.min() <= 0.0 or vv.min() <= 0.05 * v0.min():
                            continue
                        if q.min() > best_q + 1e-3:
                            best_q, best_p = float(q.min()), p
                if best_p is not None:
                    P[v] = best_p; changed = True
                    log["moved_boundary" if is_bnd[v] else "moved_interior"] += 1
                    if best_q >= target_deg:
                        done = True
                        break
            if not done:
                faces = [tuple(sorted(t[f])) for f in OUT_FACES]
                onb = [f in bkey for f in faces]
                if sum(onb) >= 2:
                    ok = True
                    if sum(onb) == 2:                                 # the new boundary edge must not already be one
                        inner = [f for f, o in zip(faces, onb) if not o]
                        e_new = set(inner[0]) & set(inner[1])
                        ok = not any(e_new <= set(f) for f in bkey)
                    if ok:
                        T[k] = -1; changed = True; log["deleted_tets"] += 1
        if (T < 0).any():
            T = T[(T >= 0).all(1)]
            used = np.zeros(len(P), bool); used[np.unique(T)] = True
            if not used.all():
                remap = -np.ones(len(P), np.int64); remap[used] = np.arange(int(used.sum()))
                log["dropped_nodes"] += int((~used).sum())
                P = P[used]; P0 = P0[used]; T = remap[T]
        if not changed:
            break
    dmin, _ = _min_dihedral_and_volume(P[T])
    log["end_min_dihedral_deg"] = round(float(dmin.min()), 3); log["end_tets_below_target"] = int((dmin < target_deg).sum())
    log["max_node_move_mm"] = round(float(np.linalg.norm(P - P0, axis=1).max()), 4)
    log["ok"] = bool(log["end_tets_below_target"] == 0)
    return P, T, log


def tet_edge_lengths(P, T):
    return np.concatenate([np.linalg.norm(P[T[:, i]] - P[T[:, j]], axis=1) for i, j in EDGES])


def tri_edge_lengths(V, F):
    return np.concatenate([np.linalg.norm(V[F[:, i]] - V[F[:, j]], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))])


def tri_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def boundary_faces(T):
    """Outward-oriented boundary triangles of a positively oriented tet mesh (faces used by exactly one tet)."""
    faces = np.concatenate([T[:, f] for f in OUT_FACES])                 # (4m, 3) outward
    key = np.sort(faces, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[cnt == 1]]


def vertex_normals(P, F):
    """Area-weighted outward vertex normals from oriented triangles (dict node -> unit normal as an (n,3) array)."""
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])     # 2*area*normal
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(n, 1e-300)


def stats(x, digits=3):
    x = np.asarray(x, float)
    return dict(mean=round(float(x.mean()), digits), min=round(float(x.min()), digits), max=round(float(x.max()), digits),
                p05=round(float(np.percentile(x, 5)), digits), p95=round(float(np.percentile(x, 95)), digits))


def dist_to_polyline(X, C):
    """Euclidean distance from points X (n,3) to the polyline C (k,3)."""
    X = np.atleast_2d(X)
    d = np.full(len(X), np.inf)
    for a, b in zip(C[:-1], C[1:]):
        ab = b - a
        l2 = float(ab @ ab)
        t = np.clip(((X - a) @ ab) / l2, 0.0, 1.0) if l2 > 0 else np.zeros(len(X))
        d = np.minimum(d, np.linalg.norm(X - a - t[:, None] * ab, axis=1))
    return d


# ============================================================================ file I/O (legacy VTK, OBJ)
def write_vtk_legacy(path, P, T, title="tets"):
    """Legacy ASCII VTK unstructured grid, classic CELLS layout (readable by SOFA MeshVTKLoader and VTK)."""
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n%s\nASCII\nDATASET UNSTRUCTURED_GRID\n" % title)
        fh.write("POINTS %d float\n" % len(P))
        fh.write("\n".join("%.6f %.6f %.6f" % tuple(p) for p in P) + "\n")
        fh.write("CELLS %d %d\n" % (len(T), 5 * len(T)))
        fh.write("\n".join("4 %d %d %d %d" % tuple(t) for t in T) + "\n")
        fh.write("CELL_TYPES %d\n" % len(T))
        fh.write("\n".join(["10"] * len(T)) + "\n")


def read_vtk_legacy(path):
    """Minimal reader for the files written above (numpy only; used by render under py -3.11)."""
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    P = np.array(tok[i + 3:i + 3 + 3 * n], float).reshape(n, 3)
    j = tok.index("CELLS", i)
    m = int(tok[j + 1])
    T = np.array(tok[j + 3:j + 3 + 5 * m], int).reshape(m, 5)[:, 1:]
    return P, T


def write_obj(path, V, F, header):
    with open(path, "w") as fh:
        for h in header.splitlines():
            fh.write("# %s\n" % h)
        fh.write("\n".join("v %.6f %.6f %.6f" % tuple(v) for v in V) + "\n")
        fh.write("\n".join("f %d %d %d" % (f[0] + 1, f[1] + 1, f[2] + 1) for f in F) + "\n")


# ============================================================================ build (python 3.13 host)
def load_labels(data):
    import nibabel as nib
    masks, aff = {}, None
    for name in sorted(set(SOURCE.values())):
        im = nib.load("%s/preBT_MRI_label_%s.nii" % (data, name))
        masks[name] = np.asarray(im.dataobj) > 0
        if aff is None:
            aff = im.affine.copy()
        elif not np.allclose(aff, im.affine):
            raise RuntimeError("label %s has a different affine" % name)
    return masks, aff


def assign_bodies(masks, aff, connectivity):
    """One body per voxel by priority, then the largest connected component per body. Returns masks, volumes."""
    from scipy import ndimage as ndi
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    vv = float(np.prod(sp)) / 1000.0
    region = np.zeros(next(iter(masks.values())).shape, np.uint8)
    for code in range(len(BODIES), 0, -1):                       # lowest priority first, higher overwrite
        region[masks[SOURCE[BODIES[code - 1]]]] = code
    st = ndi.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    out, vols = {}, {}
    for code, b in enumerate(BODIES, 1):
        m = region == code
        lab, k = ndi.label(m, st)
        sizes = ndi.sum(m, lab, range(1, k + 1)) if k else np.zeros(0)
        big = (lab == (int(np.argmax(sizes)) + 1)) if k > 1 else m
        raw = int(masks[SOURCE[b]].sum())
        vols[b] = dict(label_raw=round(raw * vv, 3), after_priority=round(int(m.sum()) * vv, 3),
                       after_largest_component=round(int(big.sum()) * vv, 3), components=int(k),
                       component_sizes_vox=sorted([int(s) for s in sizes], reverse=True)[:6],
                       removed_by_priority=round((raw - int(m.sum())) * vv, 3),
                       removed_stray=round((int(m.sum()) - int(big.sum())) * vv, 3))
        out[b] = big
    tot = sum(int(m.sum()) for m in out.values())
    if int((np.sum([out[b].astype(np.uint8) for b in BODIES], 0) > 0).sum()) != tot:
        raise RuntimeError("bodies are not voxel-exclusive")
    return out, vols, sp


def polydata(V, F):
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=1))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(F), ns.numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].astype(np.int64).ravel(), deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    return poly


def poly_arrays(poly):
    from vtk.util import numpy_support as ns
    V = ns.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    off = ns.vtk_to_numpy(poly.GetPolys().GetOffsetsArray())
    if len(off) > 1 and not np.all(np.diff(off) == 3):
        raise RuntimeError("non-triangular faces")
    F = ns.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    return V, F


def orient_outward(V, F):
    """Consistent triangle orientation (vtkPolyDataNormals, auto-orient) with positive signed volume."""
    import vtk
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(polydata(V, F))
    nrm.AutoOrientNormalsOn()
    nrm.ConsistencyOn()
    nrm.SplittingOff()
    nrm.ComputePointNormalsOff()
    nrm.Update()
    V2, F2 = poly_arrays(nrm.GetOutput())
    if geom.mesh_volume(V2, F2) < 0:
        F2 = F2[:, ::-1]
    return V2, F2


def manifold_check(V, F):
    """(open edges, non-manifold edges, non-manifold vertices) of a triangle mesh: every edge in exactly two
    triangles and every vertex link a single cycle."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    n_open, n_nonman = int((cnt == 1).sum()), int((cnt > 2).sum())
    # vertex link: union-find over the opposite edges of the incident triangles
    n_bad_vertex = 0
    inc = [[] for _ in range(len(V))]
    for k, f in enumerate(F):
        for j in range(3):
            inc[f[j]].append((f[(j + 1) % 3], f[(j + 2) % 3]))
    for v, link in enumerate(inc):
        if not link:
            n_bad_vertex += 1
            continue
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in link:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        roots = {find(x) for e in link for x in e}
        if len(roots) != 1:
            n_bad_vertex += 1
    return n_open, n_nonman, n_bad_vertex


def fine_surface(mask, aff):
    """prep_inputs.surface without decimation: marching cubes in world mm + windowed-sinc smoothing (closed, oriented)."""
    import prep_inputs as pi
    V, F, st = pi.surface(mask, aff, n_tris=10 ** 8)
    return V, F, st


def remesh_isotropic(V, F, edge_mm, n_clusters=None, min_ratio=20):
    """pyacvd (approximated centroidal Voronoi) uniform remesh with ~n_clusters vertices (edge ~ edge_mm)."""
    import pyvista as pv
    import pyacvd
    area = float(tri_areas(V, F).sum())
    if n_clusters is None:
        n_clusters = max(100, int(round(area / (np.sqrt(3.0) / 2.0 * edge_mm ** 2))))
    clus = pyacvd.Clustering(pv.PolyData(np.ascontiguousarray(V), np.c_[np.full(len(F), 3), F].ravel()))
    nsub = 0
    while clus.mesh.n_points < min_ratio * n_clusters and nsub < 3:
        clus.subdivide(1)
        nsub += 1
    clus.cluster(int(n_clusters))
    out = clus.create_mesh().clean().triangulate()
    V2 = np.asarray(out.points, float)
    F2 = np.asarray(out.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    return V2, F2, dict(n_clusters=int(n_clusters), subdivisions=nsub, input_points_after_subdivision=int(clus.mesh.n_points))


def volume_offset(V, F, vol_target_mm3, cap_mm=0.3, iters=3):
    """Move vertices along their outward normals by a uniform delta so that the closed volume equals the target
    (compensates the systematic shrinkage of smoothing + remeshing). Returns V, total delta (mm)."""
    total = 0.0
    for _ in range(iters):
        vol = geom.mesh_volume(V, F)
        d = (vol_target_mm3 - vol) / float(tri_areas(V, F).sum())
        if abs(d) < 1e-4:
            break
        d = float(np.clip(d, -cap_mm - total, cap_mm - total))
        V = V + d * vertex_normals(V, F)
        total += d
        if abs(total) >= cap_mm:
            break
    return V, total


def open_thin(mask, connectivity=6):
    """6-connected radius-1 morphological opening (drops sheets/spikes <= 2 voxels thick), largest component."""
    from scipy import ndimage as ndi
    st = ndi.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    op = ndi.binary_opening(mask, st)
    lab, k = ndi.label(op, st)
    if k > 1:
        sizes = ndi.sum(op, lab, range(1, k + 1))
        op = lab == (int(np.argmax(sizes)) + 1)
    return op


def build_surface(mask, aff, edge_mm, vol_tol_pct, iters, offset_cap_mm, thin_fallback="opening"):
    """Label -> fine MC surface (prep_inputs, undecimated) -> isotropic ACVD remesh at edge_mm -> checks (closed,
    manifold, outward, volume within vol_tol_pct of the label after a capped volume-restoring normal offset).
    If the raw body is never manifold after `iters` tries (sheets thinner than the edge collapse in the remesh), the
    mask is opened (open_thin) and the tries repeat; the volume target stays the un-opened label volume."""
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    vol_lab = float(mask.sum()) * float(np.prod(sp))
    hist, best = [], None
    stages = [("raw", mask)]
    if thin_fallback == "opening":
        stages.append(("opened", None))
    ref = {}
    for stage, m in stages:
        if m is None:
            m = open_thin(mask)
            ref["opening_removed_cc"] = round((float(mask.sum()) - float(m.sum())) * float(np.prod(sp)) / 1000.0, 3)
        V0, F0, st0 = fine_surface(m, aff)
        ref[stage] = dict(tris=int(len(F0)), verts=int(len(V0)), open_or_nonmanifold_edges=st0["open_or_nonmanifold_edges"],
                          vol_cc=st0["mesh_vol_cc"], mask_vol_cc=st0["label_vol_cc"])
        n_clus = None
        for it in range(iters):
            V, F, rinfo = remesh_isotropic(V0, F0, edge_mm, n_clus)
            V, F = orient_outward(V, F)
            n_open, n_nonman, n_badv = manifold_check(V, F)
            vol_raw = geom.mesh_volume(V, F)
            V, delta = volume_offset(V, F, vol_lab, cap_mm=offset_cap_mm)
            vol = geom.mesh_volume(V, F)
            e = tri_edge_lengths(V, F)
            rec = dict(stage=stage, n_clusters=rinfo["n_clusters"], tris=int(len(F)), verts=int(len(V)), open_edges=n_open,
                       nonmanifold_edges=n_nonman, nonmanifold_vertices=n_badv, edge_mean_mm=round(float(e.mean()), 3),
                       vol_remesh_pct=round(100 * (vol_raw - vol_lab) / vol_lab, 2), offset_mm=round(delta, 4),
                       vol_err_pct=round(100 * (vol - vol_lab) / vol_lab, 2))
            rec["manifold"] = bool(n_open == 0 and n_nonman == 0 and n_badv == 0 and vol > 0)
            rec["ok"] = bool(rec["manifold"] and abs(rec["vol_err_pct"]) <= vol_tol_pct)
            hist.append(rec)
            # keep the best manifold candidate: volume gate first, then closeness to the edge target
            score = (0 if rec["ok"] else 1, abs(float(e.mean()) - edge_mm))
            if rec["manifold"] and (best is None or score < best[3]):
                st = dict(tris=int(len(F)), verts=int(len(V)), open_or_nonmanifold_edges=n_open + n_nonman, nonmanifold_vertices=n_badv,
                          mesh_vol_cc=round(vol / 1000.0, 3), label_vol_cc=round(vol_lab / 1000.0, 3), vol_err_pct=rec["vol_err_pct"],
                          vol_remesh_pct=rec["vol_remesh_pct"], volume_offset_mm=rec["offset_mm"], signed_vol_cc=round(vol / 1000.0, 3),
                          edge_mm=stats(e), area_mm2=round(float(tri_areas(V, F).sum()), 1), remesh=rinfo, stage=stage,
                          gate_ok=rec["ok"])
                best = (V, F, st, score)
            if rec["ok"] and abs(float(e.mean()) - edge_mm) / edge_mm < 0.06:
                break
            # not manifold: nudge the cluster count; edge off: rescale it (vertices ~ 1/edge^2)
            n_clus = int(rinfo["n_clusters"] * (1.12 if not rec["manifold"] else (float(e.mean()) / edge_mm) ** 2))
        if best is not None and best[2]["gate_ok"]:
            break
    if best is None:
        raise RuntimeError("surface gate failed (no manifold remesh): %s" % json.dumps(hist, default=json_default))
    V, F, st, _ = best
    st["target_edge_mm"] = edge_mm
    st["reference"] = ref
    st["history"] = hist
    return V, F, st


def tetrahedralize(V, F, edge_mm, tcfg, nobisect):
    """TetGen -pq{minratio}/{mindihedral} [-Y] -a{maxvolume} -O{scheme}/{iterations}; then fix_slivers."""
    import tetgen
    vreg = edge_mm ** 3 / (6.0 * np.sqrt(2.0))                    # regular tet of edge edge_mm
    kw = dict(order=1, quality=True, minratio=float(tcfg["minratio"]), mindihedral=float(tcfg["mindihedral"]),
              fixedvolume=True, maxvolume=float(tcfg["maxvolume_factor"]) * vreg, nobisect=bool(nobisect),
              opt_scheme=int(tcfg["opt_scheme"]), opt_iterations=int(tcfg["opt_iterations"]),
              optmaxdihedral=float(tcfg["optmaxdihedral"]), quiet=True, nowarning=True)
    tg = tetgen.TetGen(np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64))
    P, T, _, _ = tg.tetrahedralize(**kw)
    P = np.asarray(P, float)
    T = np.asarray(T, np.int64)
    T, nflip = orient_tets(P, T)
    q_raw = tet_quality(P, T)
    BF = boundary_faces(T)
    preserved = bool(len(BF) == len(F) and len(P) >= len(V) and np.allclose(P[:len(V)], V, atol=1e-9))
    P, T, fix = fix_slivers(P, T, tcfg["sliver_target_deg"], tcfg["sliver_max_boundary_move_mm"], tcfg["sliver_max_passes"])
    sw = "pq%g/%g%sa%.4gO%d/%d" % (kw["minratio"], kw["mindihedral"], "Y" if kw["nobisect"] else "", kw["maxvolume"],
                                   kw["opt_scheme"], kw["opt_iterations"])
    return P, T, dict(route="tetgen %s (host, python %s) + local sliver fix" % (getattr(tetgen, "__version__", "?"), sys.version.split()[0]),
                      switches_equivalent=sw, kwargs={k: (v if not isinstance(v, float) else round(v, 4)) for k, v in kw.items()},
                      maxvolume_mm3=round(kw["maxvolume"], 3), flipped=nflip, input_surface_preserved=preserved,
                      tetgen_raw=dict(nodes=q_raw["nodes"], tets=q_raw["tets"], min_dihedral_deg=q_raw["min_dihedral_deg"],
                                      n_tets_min_dihedral_lt_10deg=q_raw["n_tets_min_dihedral_lt_10deg"]),
                      sliver_fix=fix)


def tetrahedralize_best(V, F, edge_mm, tcfg):
    """TetGen with the input surface preserved (-Y); if the dihedral bound is still missed after the local sliver fix,
    allow boundary Steiner points (no -Y) and keep the better of the two."""
    tried = []
    best = None
    for nobisect in ([True, False] if tcfg["nobisect"] else [False]):
        P, T, info = tetrahedralize(V, F, edge_mm, tcfg, nobisect)
        q = tet_quality(P, T)
        tried.append(dict(nobisect=nobisect, min_dihedral_deg=q["min_dihedral_deg"], nodes=q["nodes"], tets=q["tets"],
                          tetgen_min_dihedral_deg=info["tetgen_raw"]["min_dihedral_deg"]))
        if best is None or q["min_dihedral_deg"] > best[2]["min_dihedral_deg"]:
            best = (P, T, q, info)
        if q["min_dihedral_deg"] >= tcfg["mindihedral"] and q["inverted_or_flat"] == 0:
            break
    P, T, q, info = best
    info["attempts"] = tried
    return P, T, q, info


def json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % type(o))


def tet_quality(P, T):
    vol = tet_volumes(P, T)
    dih = dihedral_angles(P, T)
    dmin = dih.min(1)
    e = tet_edge_lengths(P, T)
    return dict(nodes=int(len(P)), tets=int(len(T)), volume_cc=round(float(vol.sum()) / 1000.0, 3),
                min_dihedral_deg=round(float(dmin.min()), 2), max_dihedral_deg=round(float(dih.max()), 2),
                frac_tets_min_dihedral_lt_10deg=round(float((dmin < 10.0).mean()), 5),
                frac_tets_min_dihedral_lt_15deg=round(float((dmin < 15.0).mean()), 5),
                n_tets_min_dihedral_lt_10deg=int((dmin < 10.0).sum()),
                edge_mm=stats(e), tet_volume_mm3=stats(vol), inverted_or_flat=int((vol <= 1e-9).sum()))


def signed_distance_fn(V, F):
    """Signed distance to a closed oriented triangle surface (vtkImplicitPolyDataDistance; negative inside)."""
    import vtk
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(polydata(V, F))

    def f(X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.array([ipd.EvaluateFunction([float(x[0]), float(x[1]), float(x[2])]) for x in X])
    return f


def label_axis(mask, aff, orient=None):
    """Principal axis of the body's voxels (world mm): centroid, unit axis, min/max projection of voxel centres."""
    X = v2w(aff, np.argwhere(mask))
    c = X.mean(0)
    _, s, vt = np.linalg.svd(X - c, full_matrices=False)
    ax = vt[0].copy()
    if orient is not None and abs(float(ax @ orient)) > 1e-9:
        ax *= np.sign(float(ax @ orient))
    pr = (X - c) @ ax
    return dict(centroid=c.round(3).tolist(), axis=ax.round(5).tolist(), proj_min=round(float(pr.min()), 3),
                proj_max=round(float(pr.max()), 3), sigma_mm=(s / np.sqrt(len(X))).round(2).tolist()), c, ax, float(pr.min()), float(pr.max())


def support_third(P, S, N, d, frac):
    """Surface nodes S in the outermost `frac` of the body along unit d whose outward normal faces d (a blob's cap)."""
    d = geom.unit(d)
    pr = (P[S] - P[S].mean(0)) @ d
    thr = np.percentile(pr, 100.0 * (1.0 - frac))
    return S[(pr >= thr) & (N[S] @ d > 0.0)]


def support_wall(S, N, d, cone_deg):
    """Surface nodes S whose outward normal lies within cone_deg of unit d (a tube's wall facing d, along its whole
    length; 60 deg = one third of the circumference)."""
    d = geom.unit(d)
    return S[N[S] @ d >= np.cos(np.radians(cone_deg))]


def tube_ends(Xvox, c, ax, pmin, pmax, slab_mm=12.0):
    """The two cut ends of a tubular label.  For each extreme of the principal axis, the local tube direction is the
    unit vector from the centroid of the second end slab (slab_mm..2*slab_mm from the extreme) to the centroid of the
    first (0..slab_mm), pointing outwards; the end plane passes through the extreme voxel centre along that direction.
    Returns {end_min|end_max: dict(centroid, dir, t_ext)} with t_ext the extreme projection about the first centroid."""
    pr = (Xvox - c) @ ax
    ends = {}
    for name, pe in (("end_min", pmin), ("end_max", pmax)):
        dist = np.abs(pr - pe)
        X1 = Xvox[dist <= slab_mm]
        X2 = Xvox[(dist > slab_mm) & (dist <= 2.0 * slab_mm)]
        c1 = X1.mean(0)
        d = geom.unit(c1 - X2.mean(0)) if len(X2) else geom.unit(ax * np.sign(pe - pr.mean()))
        ends[name] = dict(centroid=c1, dir=d, t_ext=float(((X1 - c1) @ d).max()), n_vox=int(len(X1)))
    return ends


def compute_node_sets(body, P, T, S, N, ctx):
    """Named node sets of CONTRACT section 1 (indices into the tets.vtk points). ctx carries the other bodies'
    signed-distance functions, the canal polyline and the label axes."""
    cs = ctx["cfg"]["sets"]
    sets, defs, extra = {}, {}, {}
    sets["surface_nodes"] = S
    defs["surface_nodes"] = "nodes on the boundary of the tet mesh (faces used by one tet)"
    if body == "cervix":
        d = np.abs(ctx["sdf"]["corpus"](P[S]))
        sets["interface_corpus"] = S[d <= cs["interface_corpus_mm"]]
        defs["interface_corpus"] = "cervix surface nodes within %g mm of the corpus surface" % cs["interface_corpus_mm"]
        canal = ctx["canal_below_os"]
        sets["canal"] = np.nonzero(dist_to_polyline(P, canal) <= cs["canal_mm"])[0]
        defs["canal"] = "all cervix nodes within %g mm of the preBT canal polyline from O_pre to the internal os (canal.npz pts[:i_internal_os+1])" % cs["canal_mm"]
        os_pt = ctx["internal_os"]
        lvl = np.abs(P[S, 2] - os_pt[2]) <= cs["os_level_mm"]
        lat = np.abs(P[S, 0] - os_pt[0]) > cs["os_lateral_min_mm"]
        sets["lateral_os_level"] = S[lvl & lat]
        defs["lateral_os_level"] = ("cervix surface nodes with |z - z_os| <= %g mm (internal-os level, S-I) and |x - x_os| > %g mm"
                                    % (cs["os_level_mm"], cs["os_lateral_min_mm"]))
        extra["lateral_os_level_sides"] = dict(left_x_lt_os=int((P[S[lvl & lat], 0] < os_pt[0]).sum()),
                                               right_x_gt_os=int((P[S[lvl & lat], 0] > os_pt[0]).sum()))
        extra["internal_os_mm"] = np.round(os_pt, 3).tolist()
    if body == "corpus":
        d = np.abs(ctx["sdf"]["cervix"](P[S]))
        sets["interface_cervix"] = S[d <= cs["interface_corpus_mm"]]
        defs["interface_cervix"] = "(extra) corpus surface nodes within %g mm of the cervix surface" % cs["interface_corpus_mm"]
    if body == "vagina":
        ax = ctx["axes"]["vagina"]
        pr = (P - np.asarray(ax[0]["centroid"])) @ np.asarray(ax[0]["axis"])
        sets["fixed_inferior"] = np.nonzero(pr <= ax[0]["proj_min"] + cs["vagina_fixed_inferior_mm"])[0]
        defs["fixed_inferior"] = ("all vagina nodes within %g mm of the inferior end of the vagina label along the vagina's principal axis (oriented +S)"
                                  % cs["vagina_fixed_inferior_mm"])
        d = np.abs(ctx["sdf"]["cervix"](P[S]))
        sets["apex"] = S[d <= cs["vagina_apex_mm"]]
        defs["apex"] = "vagina surface nodes within %g mm of the cervix surface (fornices)" % cs["vagina_apex_mm"]
    if body == "bladder":
        sets["anterior_support"] = support_third(P, S, N, cs["bladder_support_dir"], cs["support_fraction"])
        defs["anterior_support"] = ("bladder surface nodes in the anterior-superior third (largest projection on (0,1,1)/sqrt2 about the "
                                    "surface centroid, top %.0f %%) whose outward normal faces +y/+z" % (100 * cs["support_fraction"]))
    if body in ("rectum", "sigmoid"):
        ax = ctx["axes"][body]
        ends = tube_ends(ctx["voxels"][body], np.asarray(ax[0]["centroid"]), np.asarray(ax[0]["axis"]), ax[0]["proj_min"], ax[0]["proj_max"],
                         cs["end_slab_mm"])
        # the rectum and sigmoid labels meet at an oblique, tapered rectosigmoid junction (no clean slice there): the end
        # nearest the other body is taken as the junction (nodes within fixed_ends_mm of the other body's surface)
        other = "sigmoid" if body == "rectum" else "rectum"
        jname = None
        if other in ctx["sdf"]:
            dj = {name: float(ctx["sdf"][other](e["centroid"][None])[0]) for name, e in ends.items()}
            jname = min(dj, key=dj.get) if min(dj.values()) <= cs["junction_max_mm"] else None
            extra["end_distance_to_%s_mm" % other] = {k: round(v, 2) for k, v in dj.items()}
        sel, split = np.zeros(len(P), bool), {}
        for name, e in ends.items():
            if name == jname:
                m = ctx["sdf"][other](P) <= cs["fixed_ends_mm"]
                sets["junction_" + other] = np.nonzero(m)[0]
                defs["junction_" + other] = "(extra) all nodes within %g mm of the %s surface (the rectosigmoid junction end)" % (cs["fixed_ends_mm"], other)
                kind = "junction with " + other
            else:
                m = (P - e["centroid"]) @ e["dir"] >= e["t_ext"] - cs["fixed_ends_mm"]
                kind = "slab"
            sel |= m
            split[name] = int(m.sum())
            extra.setdefault("ends", {})[name] = dict(kind=kind, centroid=np.round(e["centroid"], 3).tolist(), outward_dir=np.round(e["dir"], 5).tolist(),
                                                     extreme_offset_mm=round(e["t_ext"], 3), n_nodes=int(m.sum()))
        sets["fixed_ends"] = np.nonzero(sel)[0]
        defs["fixed_ends"] = ("both cut ends: all nodes within %g mm of the label's extreme slice measured along the local tube direction of "
                              "that end (centroid of the last %g mm minus centroid of the previous %g mm); the end within %g mm of the %s is "
                              "the rectosigmoid junction and is taken as the nodes within %g mm of the %s surface instead"
                              % (cs["fixed_ends_mm"], cs["end_slab_mm"], cs["end_slab_mm"], cs["junction_max_mm"], other, cs["fixed_ends_mm"], other))
        extra["fixed_ends_split"] = split
        if body == "rectum":
            sets["posterior_support"] = support_wall(S, N, cs["rectum_support_dir"], cs["support_cone_deg"])
            defs["posterior_support"] = ("rectum surface nodes whose outward normal lies within %g deg of -y (the posterior third of the "
                                         "circumference along the whole tube)" % cs["support_cone_deg"])
    sets = {k: np.unique(np.asarray(v, np.int64)) for k, v in sets.items()}
    return sets, defs, extra


def build(args):
    t0 = time.time()
    PT = paths()
    for k in ("meshes", "figs", "logs"):
        os.makedirs(PT[k], exist_ok=True)
    cfg = json.loads(json.dumps(CFG))
    for kv in args.edge or []:
        b, v = kv.split("=")
        cfg["edge_mm"][b] = float(v)
    if args.pk:
        for d in args.pk:
            sys.path.append(d)                         # appended: the interpreter's own numpy stays first
    try:
        import tetgen  # noqa: F401
    except ImportError:
        sys.exit("tetgen not importable: py -3.13/-3.11 -m pip install --target <dir> tetgen ; then --pk <dir>")
    bodies = args.bodies or BODIES
    log = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=cfg, frame=FRAME, units="mm; volumes cc",
               interpreter=sys.version.split()[0], bodies={})
    print("[mesh_bodies] loading labels from %s" % PT["data"], flush=True)
    masks, aff = load_labels(PT["data"])
    bm, vols, sp = assign_bodies(masks, aff, cfg["connectivity"])
    for b in BODIES:
        v = vols[b]
        print("  %-8s label %7.2f cc -> priority %7.2f cc -> largest %7.2f cc (%d comps %s)"
              % (b, v["label_raw"], v["after_priority"], v["after_largest_component"], v["components"], v["component_sizes_vox"][:4]), flush=True)
    # named inputs (prep_inputs.py): canal polyline, internal os
    cz = np.load(PT["inputs"] + "/canal.npz")
    i_os = int(cz["i_internal_os"])
    canal = np.asarray(cz["pts"], float)
    ctx = dict(cfg=cfg, sdf={}, axes={}, voxels={}, canal_below_os=canal[:i_os + 1], internal_os=canal[i_os], L_end=np.asarray(cz["L_end"], float))
    # body axes from the labels (vagina oriented +S; rectum/sigmoid: both ends are used)
    for b, orient in (("vagina", [0, 0, 1]), ("rectum", [0, 0, 1]), ("sigmoid", None)):
        ctx["axes"][b] = label_axis(bm[b], aff, orient)
        ctx["voxels"][b] = v2w(aff, np.argwhere(bm[b]))
    # ---- surfaces + tets (all bodies first: the node sets need the neighbours' surfaces)
    mesh = {}
    for b in bodies:
        tb = time.time()
        print("[mesh_bodies] %s: surface (edge %.1f mm)" % (b, cfg["edge_mm"][b]), flush=True)
        V, F, sst = build_surface(bm[b], aff, cfg["edge_mm"][b], cfg["vol_tol_pct"], cfg["surface_iters"], cfg["volume_offset_cap_mm"],
                                  cfg["thin_fallback"])
        print("   surface [%s%s] tris %d verts %d open/nonmanifold %d vol %.2f cc (label %.2f: remesh %+.2f %%, offset %+.3f mm -> %+.2f %%%s) "
              "edge %.2f mm [%.2f..%.2f] (%d remesh tries)"
              % (sst["stage"], (", opening removed %.2f cc" % sst["reference"]["opening_removed_cc"]) if sst["stage"] == "opened" else "",
                 sst["tris"], sst["verts"], sst["open_or_nonmanifold_edges"], sst["mesh_vol_cc"], sst["label_vol_cc"], sst["vol_remesh_pct"],
                 sst["volume_offset_mm"], sst["vol_err_pct"], "" if sst["gate_ok"] else " VOLUME GATE FAILED", sst["edge_mm"]["mean"],
                 sst["edge_mm"]["min"], sst["edge_mm"]["max"], len(sst["history"])), flush=True)
        P, T, q, tinfo = tetrahedralize_best(V, F, cfg["edge_mm"][b], cfg["tet"])
        if q["inverted_or_flat"]:
            raise RuntimeError("%s: %d inverted/flat tets" % (b, q["inverted_or_flat"]))
        BF = boundary_faces(T)
        S = np.unique(BF)
        # surface.obj = the boundary of the tet mesh (the TetGen input when preserved, up to the sliver-fix node moves)
        remap = -np.ones(len(P), np.int64)
        remap[S] = np.arange(len(S))
        Vs, Fs = P[S], remap[BF]
        if geom.mesh_volume(Vs, Fs) < 0:
            Fs = Fs[:, ::-1]
        N = vertex_normals(P, BF)
        vol_tet = q["volume_cc"]
        fx = tinfo["sliver_fix"]
        print("   tets: nodes %d tets %d (surface nodes %d, input surface preserved %s) min dihedral %.2f deg (tetgen %.2f -> sliver fix: "
              "%d interior + %d boundary moves, %d deleted, max move %.2f mm), frac<10 %.4f, edge %.2f mm [%.2f..%.2f], vol %.2f cc "
              "(%+.2f %% vs label), %.1f s"
              % (q["nodes"], q["tets"], len(S), tinfo["input_surface_preserved"], q["min_dihedral_deg"], tinfo["tetgen_raw"]["min_dihedral_deg"],
                 fx["moved_interior"], fx["moved_boundary"], fx["deleted_tets"], fx["max_node_move_mm"], q["frac_tets_min_dihedral_lt_10deg"],
                 q["edge_mm"]["mean"], q["edge_mm"]["min"], q["edge_mm"]["max"], vol_tet,
                 100 * (vol_tet - sst["label_vol_cc"]) / sst["label_vol_cc"], time.time() - tb), flush=True)
        mesh[b] = dict(P=P, T=T, S=S, BF=BF, N=N, Vs=Vs, Fs=Fs, sst=sst, q=q, tinfo=tinfo)
        ctx["sdf"][b] = signed_distance_fn(Vs, Fs)
    # ---- node sets, files, meta
    index = dict(frame=FRAME, units="mm; volumes cc; angles deg", voxel_mm=np.round(sp, 4).tolist(), affine=np.round(aff, 6).tolist(),
                 priority=BODIES, connectivity=cfg["connectivity"], source_labels=SOURCE, cfg=cfg, tet_route=None,
                 generated=time.strftime("%Y-%m-%d %H:%M:%S"), inputs=dict(canal="inputs/canal.npz", i_internal_os=i_os),
                 bodies={}, pairs={}, commands=dict(build="python hybrid/mesh_bodies.py build --pk <dir with tetgen>",
                                                    render="py -3.11 hybrid/mesh_bodies.py render",
                                                    sofacheck="bash run_docker.sh MESHCHK hybrid/mesh_bodies.py sofacheck"))
    for b in bodies:
        m = mesh[b]
        sets, defs, extra = compute_node_sets(b, m["P"], m["T"], m["S"], m["N"], ctx)
        bnd_open, bnd_nonman, bnd_badv = manifold_check(m["Vs"], m["Fs"])
        d = "%s/%s" % (PT["meshes"], b)
        os.makedirs(d, exist_ok=True)
        write_obj(d + "/surface.obj", m["Vs"], m["Fs"],
                  "%s surface: boundary of tets.vtk; vertex k == tets.vtk point meta.surface_obj_vertex_to_tet_node[k]\n%s\nunits mm; derived from labels (local only)" % (b, FRAME))
        write_vtk_legacy(d + "/tets.vtk", m["P"], m["T"], "%s tetrahedra, %s" % (b, FRAME))
        lab_vol = m["sst"]["label_vol_cc"]
        meta = dict(body=b, priority=BODIES.index(b) + 1, source_label=SOURCE[b], role=ROLE[b], frame=FRAME, units="mm; volumes cc; angles deg",
                    voxel_mm=np.round(sp, 4).tolist(), files=dict(surface="surface.obj", tets="tets.vtk", meta="meta.json"),
                    volumes_cc=dict(label_raw=vols[b]["label_raw"], after_priority=vols[b]["after_priority"],
                                    after_largest_component=vols[b]["after_largest_component"], surface_mesh=m["sst"]["mesh_vol_cc"],
                                    tet_mesh=m["q"]["volume_cc"], surface_vs_label_pct=m["sst"]["vol_err_pct"],
                                    tet_vs_label_pct=round(100 * (m["q"]["volume_cc"] - lab_vol) / lab_vol, 2),
                                    components=vols[b]["components"], removed_by_priority=vols[b]["removed_by_priority"],
                                    removed_stray=vols[b]["removed_stray"]),
                    surface=dict(tris=int(len(m["Fs"])), verts=int(len(m["Vs"])), open_edges=int(bnd_open), nonmanifold_edges=int(bnd_nonman),
                                 nonmanifold_vertices=int(bnd_badv), target_edge_mm=m["sst"]["target_edge_mm"],
                                 edge_mm=stats(tri_edge_lengths(m["Vs"], m["Fs"])), area_mm2=round(float(tri_areas(m["Vs"], m["Fs"]).sum()), 1),
                                 gate_ok=bool(m["sst"]["gate_ok"] and bnd_open == 0 and bnd_nonman == 0),
                                 mask_stage=m["sst"]["stage"], thin_sheet_opening_applied=bool(m["sst"]["stage"] == "opened"),
                                 opening_removed_cc=m["sst"]["reference"].get("opening_removed_cc", 0.0),
                                 volume_offset_mm=m["sst"]["volume_offset_mm"], vol_remesh_pct=m["sst"]["vol_remesh_pct"],
                                 remesh=m["sst"]["remesh"], remesh_history=m["sst"]["history"], reference=m["sst"]["reference"],
                                 method="marching cubes (world mm) -> vtkWindowedSincPolyDataFilter 20 it passband 0.1 (prep_inputs.surface, "
                                        "undecimated) -> pyacvd isotropic remesh -> outward orientation -> uniform normal offset to the label "
                                        "volume; the written surface is the boundary of tets.vtk"),
                    tets=dict(nodes=m["q"]["nodes"], tets=m["q"]["tets"], surface_nodes=int(len(m["S"])), target_edge_mm=cfg["edge_mm"][b],
                              min_dihedral_deg=m["q"]["min_dihedral_deg"], max_dihedral_deg=m["q"]["max_dihedral_deg"],
                              sliver_frac_min_dihedral_lt_10deg=m["q"]["frac_tets_min_dihedral_lt_10deg"],
                              frac_min_dihedral_lt_15deg=m["q"]["frac_tets_min_dihedral_lt_15deg"],
                              n_tets_min_dihedral_lt_10deg=m["q"]["n_tets_min_dihedral_lt_10deg"], edge_mm=m["q"]["edge_mm"],
                              tet_volume_mm3=m["q"]["tet_volume_mm3"], inverted_or_flat=m["q"]["inverted_or_flat"],
                              gate_ok=bool(m["q"]["min_dihedral_deg"] > 10.0 and m["q"]["inverted_or_flat"] == 0), **m["tinfo"]),
                    axis=ctx["axes"][b][0] if b in ctx["axes"] else None,
                    node_set_defs=defs, node_set_extra=extra,
                    node_set_sizes={k: int(len(v)) for k, v in sets.items()},
                    surface_obj_vertex_to_tet_node=m["S"].tolist(),
                    node_sets={k: v.tolist() for k, v in sets.items()})
        json.dump(meta, open(d + "/meta.json", "w"), indent=1, default=json_default)
        empty = [k for k, v in sets.items() if len(v) == 0]
        print("[mesh_bodies] %-8s node sets: %s%s" % (b, {k: int(len(v)) for k, v in sets.items()}, ("  EMPTY: %s" % empty) if empty else ""), flush=True)
        index["bodies"][b] = dict(dir="meshes/%s" % b, files=dict(surface="meshes/%s/surface.obj" % b, tets="meshes/%s/tets.vtk" % b, meta="meshes/%s/meta.json" % b),
                                  priority=meta["priority"], source_label=SOURCE[b], role=ROLE[b], volumes_cc=meta["volumes_cc"],
                                  counts=dict(surface_tris=meta["surface"]["tris"], surface_verts=meta["surface"]["verts"], nodes=meta["tets"]["nodes"],
                                              tets=meta["tets"]["tets"], surface_nodes=meta["tets"]["surface_nodes"]),
                                  quality=dict(open_edges=meta["surface"]["open_edges"], min_dihedral_deg=meta["tets"]["min_dihedral_deg"],
                                               sliver_frac_lt_10deg=meta["tets"]["sliver_frac_min_dihedral_lt_10deg"],
                                               surface_edge_mean_mm=meta["surface"]["edge_mm"]["mean"], tet_edge_mean_mm=meta["tets"]["edge_mm"]["mean"],
                                               target_edge_mm=cfg["edge_mm"][b], surface_vs_label_pct=meta["volumes_cc"]["surface_vs_label_pct"]),
                                  node_set_sizes=meta["node_set_sizes"], color=COLOR[b])
        index["tet_route"] = meta["tets"]["route"]
        log["bodies"][b] = dict(meta["volumes_cc"], surface=meta["surface"], tets={k: v for k, v in meta["tets"].items()}, node_set_sizes=meta["node_set_sizes"])
    # ---- pairwise surface proximity (rest-state contact / interpenetration report for the scene module)
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            da = ctx["sdf"][b](mesh[a]["Vs"])            # signed distance of a's surface nodes to b (negative = inside b)
            db = ctx["sdf"][a](mesh[b]["Vs"])
            index["pairs"]["%s|%s" % (a, b)] = dict(min_signed_dist_mm=round(float(min(da.min(), db.min())), 3),
                                                     n_nodes_inside_gt_0p5mm=int((da < -0.5).sum() + (db < -0.5).sum()),
                                                     n_nodes_within_1mm=int((np.abs(da) < 1.0).sum() + (np.abs(db) < 1.0).sum()))
    json.dump(index, open(PT["meshes"] + "/bodies.json", "w"), indent=1, default=json_default)
    log["pairs"] = index["pairs"]
    log["wall_s"] = round(time.time() - t0, 1)
    json.dump(log, open(PT["logs"] + "/mesh_bodies.json", "w"), indent=1, default=json_default)
    print("[mesh_bodies] pairs (min signed distance mm, nodes inside > 0.5 mm):")
    for k, v in index["pairs"].items():
        if v["n_nodes_within_1mm"]:
            print("   %-18s %6.2f  %4d  (within 1 mm: %d)" % (k, v["min_signed_dist_mm"], v["n_nodes_inside_gt_0p5mm"], v["n_nodes_within_1mm"]))
    print("[mesh_bodies] wrote %s/bodies.json  (%.1f s)" % (PT["meshes"], time.time() - t0), flush=True)


# ============================================================================ render (py -3.11, pyvista off-screen)
def _pv_body(P, T, BF):
    import pyvista as pv
    faces = np.c_[np.full(len(BF), 3), BF].ravel()
    return pv.PolyData(P, faces)


def render(args):
    import pyvista as pv
    pv.OFF_SCREEN = True
    PT = paths()
    os.makedirs(PT["figs"], exist_ok=True)
    idx = json.load(open(PT["meshes"] + "/bodies.json"))
    bodies = list(idx["bodies"])
    M = {}
    for b in bodies:
        P, T = read_vtk_legacy("%s/%s/tets.vtk" % (PT["meshes"], b))
        meta = json.load(open("%s/%s/meta.json" % (PT["meshes"], b)))
        M[b] = dict(P=P, T=T, BF=boundary_faces(T), meta=meta, poly=_pv_body(P, T, boundary_faces(T)))
    cz = np.load(PT["inputs"] + "/canal.npz")
    canal = np.asarray(cz["pts"], float)
    i_os = int(cz["i_internal_os"])
    views = dict(v1=dict(title="anterior-right-superior oblique", vec=(1.0, 1.6, 0.9)),
                 v2=dict(title="left lateral (sagittal, from -x)", vec=(-1.0, 0.0, 0.0)))
    for tag, vw in views.items():
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1050))
        pl.set_background("white")
        for b in bodies:
            pl.add_mesh(M[b]["poly"], color=COLOR[b], show_edges=True, edge_color=(0.25, 0.25, 0.25), line_width=0.5,
                        opacity=1.0, smooth_shading=False, label="%s (%d tets)" % (b, M[b]["meta"]["tets"]["tets"]))
        pl.add_mesh(pv.Spline(canal, 200).tube(radius=0.8), color="black", label="preBT canal polyline")
        pl.add_legend(bcolor="white", face=None, size=(0.22, 0.22), loc="upper left")
        pl.add_axes(xlabel="x=R", ylabel="y=A", zlabel="z=S")
        pl.add_text("Stage 1 bodies: tet-mesh boundaries (%s)" % vw["title"], font_size=11, position="lower_left")
        c = np.vstack([M[b]["P"] for b in bodies])
        ctr = 0.5 * (c.min(0) + c.max(0))
        vec = np.asarray(vw["vec"], float) / np.linalg.norm(vw["vec"])
        pl.camera_position = [(ctr + 330 * vec).tolist(), ctr.tolist(), (0, 0, 1)]
        pl.camera.zoom(1.25)
        out = "%s/mesh_all_%s.png" % (PT["figs"], tag)
        pl.screenshot(out)
        pl.close()
        print("[mesh_bodies] wrote %s" % out)
    # node sets: 2 x 3 panels
    # (body, node sets, ghost bodies, camera direction, camera up, body opacity, view name)
    panels = [("cervix", ["interface_corpus", "canal", "lateral_os_level"], ["corpus"], (0.15, 1.0, 0.55), (0, 0, 1), 0.35, "from anterior-superior, body translucent"),
              ("vagina", ["fixed_inferior", "apex"], ["cervix"], (1.0, 1.2, 0.5), (0, 0, 1), 1.0, "from anterior-right"),
              ("bladder", ["anterior_support"], [], (0.0, 1.0, 1.2), (0, 0, 1), 1.0, "from anterior-superior"),
              ("rectum", ["posterior_support", "fixed_ends"], [], (-0.7, -1.0, 0.45), (0, 0, 1), 1.0, "from posterior-left-superior"),
              ("sigmoid", ["fixed_ends"], ["rectum"], (0.05, 0.1, 1.0), (0, 1, 0), 1.0, "from superior (plan view)"),
              ("corpus", ["interface_cervix"], ["cervix"], (1.0, 1.4, 0.8), (0, 0, 1), 1.0, "from anterior-right-superior")]
    setcol = dict(interface_corpus=(0.85, 0.1, 0.1), canal=(0.1, 0.2, 0.9), lateral_os_level=(0.1, 0.65, 0.1), fixed_inferior=(0.85, 0.1, 0.1),
                  apex=(0.1, 0.2, 0.9), anterior_support=(0.85, 0.1, 0.1), posterior_support=(0.85, 0.1, 0.1), fixed_ends=(0.1, 0.2, 0.9),
                  interface_cervix=(0.85, 0.1, 0.1))
    pl = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(2100, 1300), border=True)
    pl.set_background("white")
    for k, (b, names, ghosts, vec, up, opac, vname) in enumerate(panels):
        pl.subplot(k // 3, k % 3)
        m = M[b]
        pl.add_mesh(m["poly"], color=(0.82, 0.82, 0.82), show_edges=True, edge_color=(0.55, 0.55, 0.55), line_width=0.4, opacity=opac)
        for g in ghosts:
            pl.add_mesh(M[g]["poly"], color=COLOR[g], opacity=0.25, show_edges=False)
        lines = []
        for nm in names:
            ids = np.asarray(m["meta"]["node_sets"].get(nm, []), int)
            lines.append("%s: %d%s" % (nm, len(ids), "" if nm != "fixed_ends" else " (%s)" % m["meta"]["node_set_extra"].get("fixed_ends_split")))
            if len(ids):
                pl.add_points(pv.PolyData(m["P"][ids]), color=setcol[nm], point_size=8, render_points_as_spheres=True)
        if b in ("cervix", "corpus"):
            pl.add_mesh(pv.Spline(canal, 200).tube(radius=0.6), color="black")
            pl.add_points(pv.PolyData(canal[i_os][None]), color="magenta", point_size=16, render_points_as_spheres=True)
        pl.add_text("%s (%s)\n%s" % (b, vname, "\n".join(lines)), font_size=9, position="upper_left")
        ctr = 0.5 * (m["P"].min(0) + m["P"].max(0))
        ext = float(np.linalg.norm(m["P"].max(0) - m["P"].min(0)))
        v = np.asarray(vec, float) / np.linalg.norm(vec)
        pl.camera_position = [(ctr + 2.4 * ext * v).tolist(), ctr.tolist(), tuple(up)]
        pl.camera.zoom(1.5)
        pl.add_axes(xlabel="x=R", ylabel="y=A", zlabel="z=S")
    out = "%s/mesh_nodesets.png" % PT["figs"]
    pl.screenshot(out)
    pl.close()
    print("[mesh_bodies] wrote %s" % out)


# ============================================================================ sofacheck (inside the container)
def sofacheck(args):
    """Read every tets.vtk with SOFA's MeshVTKLoader and compare the counts with meta.json (format check)."""
    import Sofa.Core
    import Sofa.Simulation
    PT = paths()
    idx = json.load(open(PT["meshes"] + "/bodies.json"))
    root = Sofa.Core.Node("root")
    root.addObject("RequiredPlugin", pluginName="Sofa.Component")
    root.addObject("DefaultAnimationLoop")
    ok = True
    res = {}
    for b in idx["bodies"]:
        meta = json.load(open("%s/%s/meta.json" % (PT["meshes"], b)))
        n = root.addChild(b)
        ld = n.addObject("MeshVTKLoader", name="loader", filename="%s/%s/tets.vtk" % (PT["meshes"], b))
        n.addObject("TetrahedronSetTopologyContainer", name="topo", src="@loader")
        n.addObject("MechanicalObject", name="dofs", src="@loader")
    Sofa.Simulation.init(root)
    for b in idx["bodies"]:
        meta = json.load(open("%s/%s/meta.json" % (PT["meshes"], b)))
        n = root.getChild(b)
        pos = np.asarray(n.dofs.position.value)
        tets = np.asarray(n.topo.tetrahedra.value)
        good = bool(len(pos) == meta["tets"]["nodes"] and len(tets) == meta["tets"]["tets"])
        vol = tet_volumes(pos, tets)
        res[b] = dict(nodes=int(len(pos)), tets=int(len(tets)), expected=[meta["tets"]["nodes"], meta["tets"]["tets"]],
                      vol_cc=round(float(vol.sum()) / 1000, 3), inverted=int((vol <= 0).sum()), ok=good)
        ok &= good
        print("[sofacheck] %-8s MeshVTKLoader nodes %d tets %d (meta %d/%d) vol %.2f cc inverted %d -> %s"
              % (b, len(pos), len(tets), meta["tets"]["nodes"], meta["tets"]["tets"], vol.sum() / 1000, (vol <= 0).sum(), "OK" if good else "MISMATCH"), flush=True)
    os.makedirs(PT["logs"], exist_ok=True)
    json.dump(dict(ok=ok, bodies=res, when=time.strftime("%Y-%m-%d %H:%M:%S")), open(PT["logs"] + "/mesh_sofacheck.json", "w"), indent=1, default=json_default)
    print("[sofacheck] %s" % ("ALL OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--pk", action="append", default=[os.environ["APPSIM_PK"]] if os.environ.get("APPSIM_PK") else None,
                   help="directory holding the tetgen wheel (pip install --target DIR tetgen); appended to sys.path")
    b.add_argument("--edge", action="append", help="override a target edge, e.g. --edge bladder=5")
    b.add_argument("--bodies", nargs="+", choices=BODIES, help="subset (default: all six)")
    sub.add_parser("render")
    sub.add_parser("sofacheck")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "render":
        render(args)
    elif args.cmd == "sofacheck":
        sofacheck(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
