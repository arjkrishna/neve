"""Stage 2a MESHING: the vagina as an expanding hollow WALL around a lumen (VAGINA_WALL.md; extends CONTRACT.md).

Stage 1 meshed the vagina as a SOLID rod from the collapsed preBT label, so the applicator cannot pass through it.
This module replaces that body by a hollow annular WALL: a structured tube of tissue around a lumen of radius
`lumen_r0_mm`, built so that the device travels up the lumen and the wall parts and stretches around it.

    <out>/hybrid/meshes/vagina_wall[_<variant>]/{surface.obj, tets.vtk, meta.json}

written in exactly the format `mesh_bodies.py` produces, so `scene_hybrid.py` loads it like any other body.

REFERENCE (STRESS-FREE) STATE -- A MODELLING CHOICE, NOT A MEASUREMENT.  The real vaginal wall is folded into
rugae; opening it is mostly UNFOLDING, which costs almost no stress.  A continuum mesh cannot unfold, so the
stress-free reference here is an ALREADY-UNFOLDED tube of lumen radius `lumen_r0_mm` (default 5 mm, swept), i.e.
larger than the anatomical slit.  Wall tissue volume is conserved from the label per axial station:
    r_out(s) = sqrt(lumen_r0^2 + A(s)/pi)        A(s) = measured cross-sectional area of the vagina body at s.

Frame: preBT world RAS mm.  Units mm, cc for volumes, kPa for moduli.  No patient-derived coordinate is hard-coded:
the axis comes from `applicator/pose.json` (device_final.shaft_axis = the vagina body's principal axis by
construction) and every cross-section from the preBT labels.

Commands (host):
    python   hybrid/vagina_wall.py build                      # py3.13: labels -> wall mesh + node sets (default r0)
    python   hybrid/vagina_wall.py build --sweep              # r0 = 2 / 3.5 / 5 / 7 mm into vagina_wall_r*
    python   hybrid/vagina_wall.py report --tag V1            # lumen opening + circumferential stretch of a run
    python   hybrid/vagina_wall.py axis   --tag V1            # per-step lumen centreline vs the device travel axis
    py -3.11 hybrid/vagina_wall.py render --tag V1            # pyvista off-screen -> hybrid/figs/wall_<tag>_*.png
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True                  # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config  # noqa: E402
import geom  # noqa: E402

FRAME = "preBT world RAS mm (nibabel affine of preBT_MRI_label_*.nii; x=R, y=A, z=S)"
BODY = "vagina"                                  # the wall REPLACES the Stage-1 vagina body, under the same name
DEFAULT_DIR = "vagina_wall"

# --------------------------------------------------------------------------- configuration
# Every choice not fixed by VAGINA_WALL.md lives here; it is copied verbatim into meta.json["wall"]["cfg"].
CFG = dict(
    # --- the reference state (MODELLING CHOICE, VAGINA_WALL.md "Reference (stress-free) state")
    lumen_r0_mm=5.0,                 # mm  stress-free lumen radius = the unfolded tube.  NOT a measurement.
    sweep_r0_mm=[2.0, 3.5, 5.0, 7.0],  # mm  the sweep of VAGINA_WALL.md
    # --- discretisation (NUMERICAL)
    n_theta=20,                      # -   nodes per ring (circumferential); VAGINA_WALL.md default 20
    n_radial=1,                      # -   radial CELLS; n_radial + 1 = 2 node layers across the wall (default)
    axial_step_mm=2.0,               # mm  target axial station spacing (VAGINA_WALL.md 1.5-2.5)
    # --- how the label is turned into A(s) and a centreline (NUMERICAL)
    area_slab_mm=1.6,                # mm  slab thickness for A(s) = (voxels in slab) * voxel_volume / slab
    area_smooth_mm=3.0,              # mm  Gaussian sigma smoothing A(s) along the axis
    centre_smooth_mm=8.0,            # mm  Gaussian sigma smoothing the per-station label centroid
    centre_dev_max_mm=2.5,           # mm  the centreline may deviate this far from the axis LINE (clamped), so the
                                     #     lumen stays a function of the axis and the device travels straight up it
    min_thickness_mm=0.8,            # mm  floor on r_out - r_in: the label tapers to ~0 area at both ends and a
                                     #     zero-thickness ring has no tets.  Costs volume; reported.
    # --- node sets
    apex_mm=4.0,                     # mm  top rings within this of the superior end = `apex` (attached to cervix)
    fixed_inferior_mm=2.5,           # mm  bottom rings within this of the inferior end = `fixed_inferior` (sprung).
                                     #     VAGINA_WALL.md says "bottom ring(s)", NOT the Stage-1 lowest 15 mm: the
                                     #     device passes through the introitus, so a 15 mm pinned skirt would forbid
                                     #     the lumen from opening exactly where it must.
    # --- material (ASSUMED; VAGINA_WALL.md)
    E_vagina_kPa=10.0,               # kPa VAGINA_WALL.md default 10 (Stage 1 used 15 for the solid body)
    nu_vagina=0.45,                  # -
)


def paths():
    P = config.paths()
    hyb = P["out"] + "/hybrid"
    return dict(data=P["data"], inputs=P["inputs"], hybrid=hyb, meshes=hyb + "/meshes",
                applicator=hyb + "/applicator", runs=hyb + "/runs", figs=hyb + "/figs", logs=hyb + "/logs")


def r0_tag(r0):
    return ("r%g" % r0).replace(".", "p")


def variant_dir(r0, args):
    """Where a build lands: an explicit --variant, else one directory per swept r0, else the canonical name."""
    if getattr(args, "variant", None):
        return "vagina_wall_" + args.variant
    if getattr(args, "sweep", False):
        return "vagina_wall_" + r0_tag(r0)
    return DEFAULT_DIR


# --------------------------------------------------------------------------- small numpy helpers
# outward faces of a positively oriented tet (a,b,c,d) -- same convention as mesh_bodies.py
OUT_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
DIHEDRAL_PAIRS = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
# Kuhn (Freudenthal) subdivision of a hexahedron into 6 tets, corner code = 4*a + 2*b + c for offsets (a,b,c).
# Every tet contains the main diagonal 000-111, so neighbouring cells that share the SAME index frame split their
# common face along the same diagonal: the mesh is conforming, including across the theta seam.
KUHN = np.array([[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7], [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]])


def tet_volumes(P, T):
    X = P[T]
    return np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0]) / 6.0


def dihedral_angles(P, T):
    X = P[T]
    out = []
    for a, b, c, d in DIHEDRAL_PAIRS:
        n1 = np.cross(X[:, b] - X[:, a], X[:, c] - X[:, a])
        n2 = np.cross(X[:, b] - X[:, a], X[:, d] - X[:, a])
        n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-300)
        n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-300)
        out.append(np.degrees(np.arccos(np.clip(-(n1 * n2).sum(1), -1.0, 1.0))))
    return np.stack(out, 1)


def boundary_faces(T):
    faces = np.concatenate([T[:, f] for f in OUT_FACES])
    key = np.sort(faces, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[cnt == 1]]


def tet_edge_lengths(P, T):
    return np.concatenate([np.linalg.norm(P[T[:, i]] - P[T[:, j]], axis=1) for i, j in EDGES])


def tri_edge_lengths(V, F):
    return np.concatenate([np.linalg.norm(V[F[:, i]] - V[F[:, j]], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))])


def tri_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def stats(x, digits=3):
    x = np.asarray(x, float)
    return dict(mean=round(float(x.mean()), digits), min=round(float(x.min()), digits),
                max=round(float(x.max()), digits), p05=round(float(np.percentile(x, 5)), digits),
                p95=round(float(np.percentile(x, 95)), digits))


def manifold_check(V, F):
    """(open edges, non-manifold edges) -- every edge of a closed surface is used by exactly two triangles."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return int((cnt == 1).sum()), int((cnt > 2).sum())


def gsmooth(y, sigma_pts):
    """1-D Gaussian smoothing with reflected padding (numpy only: py -3.11 has no scipy)."""
    y = np.asarray(y, float)
    if sigma_pts <= 0 or len(y) < 3:
        return y.copy()
    r = int(min(max(1, round(3 * sigma_pts)), len(y) - 1))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / float(sigma_pts)) ** 2)
    k /= k.sum()
    yp = np.r_[y[r:0:-1], y, y[len(y) - 2:len(y) - r - 2:-1]] if r > 0 else y
    return np.convolve(yp, k, mode="same")[r:r + len(y)]


def write_vtk_legacy(path, P, T, title="tets"):
    """Legacy ASCII VTK unstructured grid, classic CELLS layout (SOFA MeshVTKLoader + VTK)."""
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n%s\nASCII\nDATASET UNSTRUCTURED_GRID\n" % title)
        fh.write("POINTS %d float\n" % len(P))
        fh.write("\n".join("%.6f %.6f %.6f" % tuple(p) for p in P) + "\n")
        fh.write("CELLS %d %d\n" % (len(T), 5 * len(T)))
        fh.write("\n".join("4 %d %d %d %d" % tuple(t) for t in T) + "\n")
        fh.write("CELL_TYPES %d\n" % len(T))
        fh.write("\n".join(["10"] * len(T)) + "\n")


def read_vtk_legacy(path):
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


def json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % type(o))


# --------------------------------------------------------------------------- the wall geometry
def label_sections(X, a, c, cfg):
    """Per-axial-station cross-sectional area and lateral centroid of the vagina body's voxels.

    X (n,3) voxel centres in world mm, `a` the axis, `c` the centroid.  Returns the station table and the
    orthonormal (e1, e2) used for the lateral coordinates."""
    e1 = geom.unit(np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0]))
    e2 = np.cross(a, e1)
    s = (X - c) @ a
    lat = np.c_[(X - c) @ e1, (X - c) @ e2]
    s_lo, s_hi = float(s.min()), float(s.max())
    n_ax = max(4, int(round((s_hi - s_lo) / float(cfg["axial_step_mm"]))) + 1)
    sk = np.linspace(s_lo, s_hi, n_ax)
    half = 0.5 * float(cfg["area_slab_mm"])
    vox = float(cfg["_voxel_mm3"])               # passed in by the caller (voxel volume, mm^3)
    A = np.zeros(n_ax)
    cxy = np.zeros((n_ax, 2))
    nv = np.zeros(n_ax, int)
    for k, sv in enumerate(sk):
        m = np.abs(s - sv) <= half
        if m.sum() == 0:                         # never happens for a 1.6 mm slab on a 1.6 mm voxel grid, but be safe
            m = np.abs(s - sv) <= 2.0 * half
        nv[k] = int(m.sum())
        A[k] = m.sum() * vox / (2.0 * half)
        cxy[k] = lat[m].mean(0)
    return dict(s=sk, A_raw=A, c_raw=cxy, n_vox=nv, s_lo=s_lo, s_hi=s_hi), e1, e2


def build_wall(sec, a, c, e1, e2, cfg):
    """Structured annular mesh: n_theta x (n_radial+1) x n_axial nodes on parallel-transported frames,
    each hexahedral cell split into 6 tets by the Kuhn rule (conforming, including across the theta seam)."""
    step = float(np.mean(np.diff(sec["s"])))
    A = gsmooth(sec["A_raw"], float(cfg["area_smooth_mm"]) / step)
    cxy = np.stack([gsmooth(sec["c_raw"][:, 0], float(cfg["centre_smooth_mm"]) / step),
                    gsmooth(sec["c_raw"][:, 1], float(cfg["centre_smooth_mm"]) / step)], 1)
    dev_raw = float(np.linalg.norm(sec["c_raw"], axis=1).max())
    dev_sm = float(np.linalg.norm(cxy, axis=1).max())
    d = np.linalg.norm(cxy, axis=1)
    lim = float(cfg["centre_dev_max_mm"])
    scale = np.where(d > lim, lim / np.maximum(d, 1e-9), 1.0)
    cxy = cxy * scale[:, None]
    n_clamped = int((scale < 1.0).sum())

    r_in = float(cfg["lumen_r0_mm"])
    r_out = np.sqrt(r_in ** 2 + A / np.pi)
    th_min = float(cfg["min_thickness_mm"])
    thin = (r_out - r_in) < th_min
    r_out = np.maximum(r_out, r_in + th_min)

    C = c[None, :] + sec["s"][:, None] * a[None, :] + cxy[:, 0:1] * e1[None, :] + cxy[:, 1:2] * e2[None, :]
    # rotation-minimising (parallel transport) frames along the centreline
    T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    U = np.zeros_like(C)
    V = np.zeros_like(C)
    U[0] = geom.ortho(e1, T[0])
    V[0] = np.cross(T[0], U[0])
    for k in range(1, len(C)):
        R = geom.rot_between(T[k - 1], T[k])
        U[k] = geom.ortho(R @ U[k - 1], T[k])
        V[k] = np.cross(T[k], U[k])

    n_ax, n_th, n_rd = len(C), int(cfg["n_theta"]), int(cfg["n_radial"])
    th = 2.0 * np.pi * np.arange(n_th) / n_th

    def nid(k, m, j):
        return (k * (n_rd + 1) + m) * n_th + (j % n_th)

    P = np.zeros((n_ax * (n_rd + 1) * n_th, 3))
    gidx = np.zeros((len(P), 3), int)            # (axial, radial layer, theta) of every node
    for k in range(n_ax):
        for m in range(n_rd + 1):
            r = r_in + (r_out[k] - r_in) * m / float(n_rd)
            ring = C[k][None, :] + r * (np.cos(th)[:, None] * U[k][None, :] + np.sin(th)[:, None] * V[k][None, :])
            for j in range(n_th):
                P[nid(k, m, j)] = ring[j]
                gidx[nid(k, m, j)] = (k, m, j)
    tets = []
    for k in range(n_ax - 1):
        for m in range(n_rd):
            for j in range(n_th):
                cor = [nid(k + (q >> 2 & 1), m + (q >> 1 & 1), j + (q & 1)) for q in range(8)]
                for t in KUHN:
                    tets.append([cor[t[0]], cor[t[1]], cor[t[2]], cor[t[3]]])
    T4 = np.array(tets, int)
    v = tet_volumes(P, T4)
    flipped = int((v < 0).sum())
    T4[v < 0] = T4[v < 0][:, [0, 2, 1, 3]]
    return dict(P=P, T=T4, C=C, U=U, V=V, Tg=T, grid=gidx, s=sec["s"], A=A, r_in=r_in, r_out=r_out,
                n_ax=n_ax, n_th=n_th, n_rd=n_rd, flipped=flipped, cxy=cxy,
                centre_dev=dict(raw_max_mm=round(dev_raw, 3), smoothed_max_mm=round(dev_sm, 3),
                                clamp_mm=lim, n_stations_clamped=n_clamped,
                                final_max_mm=round(float(np.linalg.norm(cxy, axis=1).max()), 3)),
                thin=dict(n_stations_below_min_thickness=int(thin.sum()), min_thickness_mm=th_min,
                          stations=[round(float(x), 2) for x in sec["s"][thin]]))


def node_sets(W, sec, cfg, sdf=None):
    """inner_surface / outer_surface / apex / fixed_inferior / surface_nodes (indices into tets.vtk points)."""
    P, T, g = W["P"], W["T"], W["grid"]
    BF = boundary_faces(T)
    S = np.unique(BF)
    m = g[:, 1]
    inner = np.nonzero(m == 0)[0]
    outer = np.nonzero(m == W["n_rd"])[0]
    kk = g[:, 0]
    s_lo, s_hi = sec["s"][0], sec["s"][-1]
    apex = np.nonzero(sec["s"][kk] >= s_hi - float(cfg["apex_mm"]))[0]
    inf = np.nonzero(sec["s"][kk] <= s_lo + float(cfg["fixed_inferior_mm"]))[0]
    # boundary triangles by side: an inner face has all three nodes in the innermost layer, an outer face all
    # three in the outermost; the rest are the two end-cap annuli.
    fin = np.all(m[BF] == 0, axis=1)
    fout = np.all(m[BF] == W["n_rd"], axis=1)
    sets = dict(surface_nodes=S, inner_surface=inner, outer_surface=outer, apex=apex, fixed_inferior=inf)
    defs = dict(
        surface_nodes="nodes on the boundary of the tet mesh (faces used by one tet)",
        inner_surface="nodes of the innermost radial layer: the LUMEN wall, which contacts the device",
        outer_surface="nodes of the outermost radial layer: contacts bladder / rectum / cervix",
        apex="nodes within %g mm of the superior end of the wall along the vaginal axis (the fornices; attached "
             "to the cervix as the Stage-1 vagina.apex was)" % cfg["apex_mm"],
        fixed_inferior="nodes within %g mm of the inferior end of the wall along the vaginal axis (the introitus "
                       "ring; sprung as in Stage 1).  NOT the Stage-1 lowest-15 mm skirt: the device passes "
                       "through the introitus and a pinned skirt would forbid the lumen from opening there."
                       % cfg["fixed_inferior_mm"])
    extra = dict(inner_triangles=BF[fin].tolist(), outer_triangles=BF[fout].tolist(),
                 n_inner_triangles=int(fin.sum()), n_outer_triangles=int(fout.sum()),
                 n_cap_triangles=int((~fin & ~fout).sum()),
                 apex_rings=int(len(np.unique(g[apex, 0]))), inferior_rings=int(len(np.unique(g[inf, 0]))))
    if sdf is not None:
        for nm, f in sdf.items():
            d = f(P[outer])
            extra["outer_signed_dist_to_%s_mm" % nm] = dict(min=round(float(d.min()), 3),
                                                            n_inside=int((d < 0).sum()))
        d = sdf["cervix"](P[apex])
        extra["apex_signed_dist_to_cervix_mm"] = dict(min=round(float(d.min()), 3), max=round(float(d.max()), 3),
                                                      mean=round(float(d.mean()), 3))
    return sets, defs, extra, BF, S


# --------------------------------------------------------------------------- build (py3.13 host)
def build_one(r0, args, shared):
    cfg = json.loads(json.dumps(CFG))
    cfg["lumen_r0_mm"] = float(r0)
    for kv in args.set or []:
        k, v = kv.split("=")
        cfg[k] = json.loads(v)
    cfg["_voxel_mm3"] = shared["vox"]
    PT = shared["PT"]
    X, a, c = shared["X"], shared["a"], shared["c"]
    t0 = time.time()
    sec, e1, e2 = label_sections(X, a, c, cfg)
    W = build_wall(sec, a, c, e1, e2, cfg)
    sets, defs, extra, BF, S = node_sets(W, sec, cfg, shared.get("sdf"))
    P, T = W["P"], W["T"]

    vol = tet_volumes(P, T)
    dih = dihedral_angles(P, T)
    dmin = dih.min(1)
    remap = -np.ones(len(P), np.int64)
    remap[S] = np.arange(len(S))
    Vs, Fs = P[S], remap[BF]
    surf_vol = geom.mesh_volume(Vs, Fs)
    if surf_vol < 0:                              # keep the closed boundary outward-oriented (positive volume)
        Fs = Fs[:, ::-1]
        surf_vol = -surf_vol
    n_open, n_nonman = manifold_check(Vs, Fs)
    # inner-surface normals must point INTO the lumen (see the note in meta.json)
    fin = np.all(W["grid"][:, 1][BF] == 0, axis=1)
    nrm = np.cross(P[BF[fin, 1]] - P[BF[fin, 0]], P[BF[fin, 2]] - P[BF[fin, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    ctri = P[BF[fin]].mean(1)
    kk = W["grid"][BF[fin, 0], 0]
    radial = ctri - W["C"][kk]
    radial -= (np.einsum("ij,ij->i", radial, W["Tg"][kk]))[:, None] * W["Tg"][kk]
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    inward_frac = float((np.einsum("ij,ij->i", nrm, radial) < 0).mean())

    lab_cc = shared["label_cc"]
    wall_cc = float(vol.sum()) / 1000.0
    d = "%s/%s" % (PT["meshes"], variant_dir(r0, args))
    os.makedirs(d, exist_ok=True)
    write_obj(d + "/surface.obj", Vs, Fs,
              "vagina WALL surface: boundary of tets.vtk; vertex k == tets.vtk point "
              "meta.surface_obj_vertex_to_tet_node[k]\n%s\nunits mm; derived from labels (local only)" % FRAME)
    write_vtk_legacy(d + "/tets.vtk", P, T, "vagina wall tetrahedra, %s" % FRAME)

    thick = W["r_out"] - W["r_in"]
    meta = dict(
        body=BODY, priority=3, source_label="vagina", frame=FRAME, units="mm; volumes cc; angles deg",
        role="deformable HOLLOW WALL around a lumen (Stage 2a): the device travels up the lumen and the wall "
             "parts and stretches around it",
        voxel_mm=shared["voxel_mm"], files=dict(surface="surface.obj", tets="tets.vtk", meta="meta.json"),
        volumes_cc=dict(label_raw=shared["label_raw_cc"], after_priority=shared["after_priority_cc"],
                        after_largest_component=lab_cc, surface_mesh=round(surf_vol / 1000.0, 4),
                        tet_mesh=round(wall_cc, 4),
                        surface_vs_label_pct=round(100 * (surf_vol / 1000.0 - lab_cc) / lab_cc, 2),
                        tet_vs_label_pct=round(100 * (wall_cc - lab_cc) / lab_cc, 2),
                        note="wall tissue volume is conserved from the label per axial station "
                             "(r_out = sqrt(r0^2 + A(s)/pi)); the deficit is the min-thickness floor at the "
                             "tapered ends plus the round-annulus idealisation of a flat slit"),
        surface=dict(tris=int(len(Fs)), verts=int(len(Vs)), open_edges=n_open, nonmanifold_edges=n_nonman,
                     nonmanifold_vertices=0, target_edge_mm=cfg["axial_step_mm"],
                     edge_mm=stats(tri_edge_lengths(Vs, Fs)), area_mm2=round(float(tri_areas(Vs, Fs).sum()), 1),
                     gate_ok=bool(n_open == 0 and n_nonman == 0 and surf_vol > 0),
                     inner_normals_point_into_lumen_frac=round(inward_frac, 4),
                     method="structured annulus: per-station label area A(s) -> r_out = sqrt(r0^2 + A/pi) on a "
                            "smoothed, clamped centreline with parallel-transported frames; hexahedral cells "
                            "split into 6 tets by the Kuhn rule; the written surface is the boundary of tets.vtk"),
        tets=dict(nodes=int(len(P)), tets=int(len(T)), surface_nodes=int(len(S)),
                  target_edge_mm=cfg["axial_step_mm"], min_dihedral_deg=round(float(dmin.min()), 2),
                  max_dihedral_deg=round(float(dih.max()), 2),
                  dihedral_convention="min_dihedral_deg / max_dihedral_deg are computed with the SAME formula as "
                                      "mesh_bodies.py, which returns the SUPPLEMENT (180 - theta) of the interior "
                                      "dihedral angle -- a regular tetrahedron reports 109.47 deg, not 70.53, and "
                                      "a unit-cube Kuhn tet reports 90-135 deg where the true range is 45-90. "
                                      "They are kept for comparability with meshes/<body>/meta.json; the TRUE "
                                      "interior angles are interior_dihedral_{min,max}_deg below, and the gate "
                                      "below uses the true minimum.",
                  interior_dihedral_min_deg=round(float(180.0 - dih.max()), 2),
                  interior_dihedral_max_deg=round(float(180.0 - dmin.min()), 2),
                  n_tets_interior_dihedral_lt_10deg=int(((180.0 - dih.max(1)) < 10.0).sum()),
                  n_tets_interior_dihedral_lt_20deg=int(((180.0 - dih.max(1)) < 20.0).sum()),
                  sliver_frac_min_dihedral_lt_10deg=round(float((dmin < 10.0).mean()), 5),
                  frac_min_dihedral_lt_15deg=round(float((dmin < 15.0).mean()), 5),
                  n_tets_min_dihedral_lt_10deg=int((dmin < 10.0).sum()), edge_mm=stats(tet_edge_lengths(P, T)),
                  tet_volume_mm3=stats(vol), inverted_or_flat=int((vol <= 1e-9).sum()), flipped=W["flipped"],
                  gate_ok=bool(180.0 - dih.max() > 10.0 and (vol <= 1e-9).sum() == 0),
                  route="structured annular grid + Kuhn 6-tet split (vagina_wall.py, python %s)"
                        % sys.version.split()[0]),
        axis=dict(centroid=np.round(c, 3).tolist(), axis=np.round(a, 5).tolist(),
                  proj_min=round(float(sec["s_lo"]), 3), proj_max=round(float(sec["s_hi"]), 3),
                  source="pose.json device_final.shaft_axis (= the vagina body's principal axis, rule v2)"),
        node_set_defs=defs, node_set_extra=extra,
        node_set_sizes={k: int(len(v)) for k, v in sets.items()},
        surface_obj_vertex_to_tet_node=S.tolist(),
        node_sets={k: np.asarray(v, int).tolist() for k, v in sets.items()},
        wall=dict(
            reference_state="MODELLING CHOICE, not a measured geometry: the stress-free state is an ALREADY "
                            "UNFOLDED tube of lumen radius %g mm.  The real wall is folded into rugae and opens "
                            "mostly by unfolding, which a continuum mesh cannot represent; a mesh built at the "
                            "true collapsed slit would have to stretch ~10x circumferentially and would invert. "
                            "lumen_r0_mm is a parameter to sweep, not a fitted value." % cfg["lumen_r0_mm"],
            cfg={k: v for k, v in cfg.items() if not k.startswith("_")},
            n_axial=W["n_ax"], n_theta=W["n_th"], n_radial=W["n_rd"], axial_step_mm=round(
                float(np.mean(np.diff(sec["s"]))), 4),
            lumen_r0_mm=cfg["lumen_r0_mm"], E_vagina_kPa=cfg["E_vagina_kPa"], nu_vagina=cfg["nu_vagina"],
            s=np.round(sec["s"], 4).tolist(), area_mm2=np.round(W["A"], 4).tolist(),
            area_raw_mm2=np.round(sec["A_raw"], 4).tolist(), n_vox_per_station=sec["n_vox"].tolist(),
            r_out_mm=np.round(W["r_out"], 4).tolist(), thickness_mm=np.round(thick, 4).tolist(),
            thickness_stats=stats(thick), centreline=np.round(W["C"], 4).tolist(),
            frame_u=np.round(W["U"], 6).tolist(), frame_v=np.round(W["V"], 6).tolist(),
            tangent=np.round(W["Tg"], 6).tolist(),
            grid_index=W["grid"].tolist(),
            grid_index_note="per node (axial station, radial layer 0=lumen .. n_radial=outer, theta index)",
            centre_deviation=W["centre_dev"], min_thickness=W["thin"],
            ref_inner_perimeter_mm=round(2.0 * np.pi * cfg["lumen_r0_mm"], 4),
            collision_note="the closed boundary is oriented OUTWARD from the tissue (positive enclosed volume), "
                           "which on the inner cylinder points INTO the lumen -- what SOFA's TriangleCollisionModel "
                           "expects for a closed body (the normal points away from the material)."))
    json.dump(meta, open(d + "/meta.json", "w"), indent=1, default=json_default)
    row = dict(r0_mm=cfg["lumen_r0_mm"], dir=os.path.basename(d), nodes=int(len(P)), tets=int(len(T)),
               n_axial=W["n_ax"], n_theta=W["n_th"], n_radial=W["n_rd"],
               vol_cc=round(wall_cc, 3), label_cc=lab_cc, vol_vs_label_pct=meta["volumes_cc"]["tet_vs_label_pct"],
               thickness_mm=[round(float(thick.min()), 2), round(float(np.median(thick)), 2),
                             round(float(thick.max()), 2)],
               interior_dihedral_deg=[meta["tets"]["interior_dihedral_min_deg"],
                                      meta["tets"]["interior_dihedral_max_deg"]],
               min_dihedral=meta["tets"]["min_dihedral_deg"], inverted=meta["tets"]["inverted_or_flat"],
               open_edges=n_open, nonmanifold=n_nonman, inner_normals_inward=round(inward_frac, 3),
               sets={k: int(len(v)) for k, v in sets.items()}, wall_s=time.time() - t0)
    print("[vagina_wall] r0 %4.1f mm -> %-18s nodes %4d tets %5d  vol %.3f cc (%+.1f %% vs label %.2f)  "
          "thickness %.2f/%.2f/%.2f mm  interior dihedral %.1f-%.1f deg  inverted %d  open %d  "
          "inner-normals-inward %.0f %%"
          % (cfg["lumen_r0_mm"], os.path.basename(d), len(P), len(T), wall_cc,
             meta["volumes_cc"]["tet_vs_label_pct"], lab_cc, thick.min(), np.median(thick), thick.max(),
             180.0 - dih.max(), 180.0 - dmin.min(), (vol <= 1e-9).sum(), n_open, 100 * inward_frac), flush=True)
    return row, meta, W, sec


def build(args):
    import mesh_bodies as MB                     # host only (nibabel / scipy / vtk)
    PT = paths()
    for k in ("meshes", "figs", "logs"):
        os.makedirs(PT[k], exist_ok=True)
    pose = json.load(open(PT["applicator"] + "/pose.json"))
    a = geom.unit(np.asarray(pose["device_final"]["shaft_axis"], float))
    print("[vagina_wall] loading labels from %s" % PT["data"], flush=True)
    masks, aff = MB.load_labels(PT["data"])
    bm, vols, sp = MB.assign_bodies(masks, aff, MB.CFG["connectivity"])
    X = MB.v2w(aff, np.argwhere(bm["vagina"]))
    c = X.mean(0)
    vox = float(np.prod(sp))
    shared = dict(PT=PT, X=X, a=a, c=c, vox=vox, voxel_mm=np.round(sp, 4).tolist(),
                  label_raw_cc=vols["vagina"]["label_raw"], after_priority_cc=vols["vagina"]["after_priority"],
                  label_cc=vols["vagina"]["after_largest_component"])
    print("[vagina_wall] vagina body %d voxels, %.3f cc; axis %s (pose.json shaft_axis); centroid %s"
          % (len(X), shared["label_cc"], np.round(a, 5).tolist(), np.round(c, 3).tolist()), flush=True)
    # neighbour surfaces, for the rest-state proximity report
    sdf = {}
    for b in ("cervix", "rectum", "bladder", "corpus"):
        V, F = geom.read_obj("%s/%s/surface.obj" % (PT["meshes"], b))
        sdf[b] = MB.signed_distance_fn(V, F)
    shared["sdf"] = sdf
    r0s = list(CFG["sweep_r0_mm"]) if args.sweep else [float(args.r0 if args.r0 is not None else
                                                             CFG["lumen_r0_mm"])]
    rows = []
    for r0 in r0s:
        row, meta, W, sec = build_one(r0, args, shared)
        rows.append(row)
    if args.sweep:                               # the default r0 is also written to the canonical directory
        src = "%s/vagina_wall_%s" % (PT["meshes"], r0_tag(CFG["lumen_r0_mm"]))
        dst = "%s/%s" % (PT["meshes"], DEFAULT_DIR)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for fn in ("surface.obj", "tets.vtk", "meta.json"):
                shutil.copy2(src + "/" + fn, dst + "/" + fn)
            print("[vagina_wall] default r0 %g mm copied %s -> %s" % (CFG["lumen_r0_mm"], os.path.basename(src),
                                                                      DEFAULT_DIR), flush=True)
    log = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), frame=FRAME, units="mm; volumes cc; angles deg",
               cfg={k: v for k, v in CFG.items()}, interpreter=sys.version.split()[0], variants=rows)
    json.dump(log, open(PT["logs"] + "/vagina_wall.json", "w"), indent=1, default=json_default)
    print("[vagina_wall] wrote %s/vagina_wall.json" % PT["logs"], flush=True)


# --------------------------------------------------------------------------- scene support (container, py3.8)
# Several containers of one batch (run_docker_par.sh) build the SAME shadow root at the same instant when they
# share a wall variant, and a plain open(path, "w") is visible to the others as an EMPTY file: MEASURED (run
# WB_a5) as `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` on bodies.json while its three siblings
# proceeded normally.  Every file here is therefore written to a private temp name and os.replace()d into place,
# which is atomic.
def _atomic_copy(src, dst):
    tmp = "%s.tmp.%d" % (dst, os.getpid())
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _atomic_json(obj, path):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=json_default)
    os.replace(tmp, path)


def scene_mesh_root(meshes_dir, wall_dir):
    """Return a directory that looks exactly like `meshes/` but whose `vagina` body IS the wall.

    `run_hybrid.py` (which this module must not edit) builds every per-body path from a single
    `inp["P"]["meshes"]`, so the wall has to be presented under the name `vagina` inside such a directory.
    The five unchanged bodies are copied once; the sixth is the wall."""
    root = "%s/_scene_%s" % (meshes_dir, wall_dir)
    src_wall = "%s/%s" % (meshes_dir, wall_dir)
    if not os.path.isdir(src_wall):
        raise SystemExit("no wall mesh at %s (run: python hybrid/vagina_wall.py build)" % src_wall)
    os.makedirs(root, exist_ok=True)
    for b in ("corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"):
        src = src_wall if b == "vagina" else "%s/%s" % (meshes_dir, b)
        dst = "%s/%s" % (root, b)
        os.makedirs(dst, exist_ok=True)
        for fn in ("surface.obj", "tets.vtk", "meta.json"):
            s, d = src + "/" + fn, dst + "/" + fn
            if (not os.path.exists(d)) or os.path.getmtime(s) > os.path.getmtime(d):
                _atomic_copy(s, d)
    idx = json.load(open(meshes_dir + "/bodies.json"))
    wm = json.load(open(root + "/vagina/meta.json"))
    e = idx["bodies"]["vagina"]
    e["dir"] = "meshes/%s" % wall_dir
    e["role"] = wm["role"]
    e["volumes_cc"] = wm["volumes_cc"]
    e["counts"] = dict(surface_tris=wm["surface"]["tris"], surface_verts=wm["surface"]["verts"],
                       nodes=wm["tets"]["nodes"], tets=wm["tets"]["tets"],
                       surface_nodes=wm["tets"]["surface_nodes"])
    e["quality"] = dict(open_edges=wm["surface"]["open_edges"], min_dihedral_deg=wm["tets"]["min_dihedral_deg"],
                        sliver_frac_lt_10deg=wm["tets"]["sliver_frac_min_dihedral_lt_10deg"],
                        surface_edge_mean_mm=wm["surface"]["edge_mm"]["mean"],
                        tet_edge_mean_mm=wm["tets"]["edge_mm"]["mean"])
    e["node_set_sizes"] = wm["node_set_sizes"]
    idx["vagina_model"] = "wall (%s); the other five bodies are copies of meshes/<body>" % wall_dir
    _atomic_json(idx, root + "/bodies.json")
    return root


# --------------------------------------------------------------------------- the lumen-vs-device-axis curve
def _perim(R):
    return float(np.linalg.norm(np.diff(np.r_[R, R[:1]], axis=0), axis=1).sum())


def wall_mesh_dir(PT, tag):
    """The mesh directory a run actually used for the vagina (the scene shadow root, else the variant itself)."""
    cfgj = json.load(open("%s/%s/cfg.json" % (PT["runs"], tag)))
    wd = cfgj.get("vagina_wall_dir", DEFAULT_DIR)
    d = "%s/_scene_%s/vagina" % (PT["meshes"], wd)
    if not os.path.exists(d + "/meta.json"):
        d = "%s/%s" % (PT["meshes"], wd)
    return d, cfgj


def axis(args):
    """Per STEP: how far the lumen centreline has slid off the device's straight travel axis.

    THE diagnostic for the Stage-2a failure mode.  The wall's apex follows the cervix, which rides the KINEMATIC
    corpus, while the device translates along a fixed straight line.  Once the lumen centre is further off that line
    than (lumen radius - tandem radius), the shaft is no longer inside the lumen: it presses THROUGH the wall
    sideways instead of opening it, and the elements at that patch invert.

    Reconstructed on the HOST from `runs/<tag>/frames/` -- every inner-ring node is a surface node, so a frame OBJ
    carries it exactly -- so it works for runs made before the same metric was added to the container-side log
    (`log.jsonl["wall"]`), which is used when no frames were exported."""
    PT = paths()
    md, cfgj = wall_mesh_dir(PT, args.tag)
    meta = json.load(open(md + "/meta.json"))
    W = meta["wall"]
    P0 = read_vtk_legacy(md + "/tets.vtk")[0]
    s2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
    g = np.asarray(W["grid_index"], int)
    rings = [np.nonzero((g[:, 0] == k) & (g[:, 1] == 0))[0] for k in range(int(W["n_axial"]))]
    s = np.asarray(W["s"], float)
    perim0 = np.array([_perim(P0[q]) for q in rings])
    rad0 = np.array([float(np.linalg.norm(P0[q] - P0[q].mean(0), axis=1).mean()) for q in rings])
    fd = "%s/%s/frames" % (PT["runs"], args.tag)
    rows, src, r_tandem = [], None, None
    if os.path.exists(fd + "/index.json"):
        src = "frames/ (host reconstruction of the inner ring from the exported surfaces)"
        for f in json.load(open(fd + "/index.json"))["frames"]:
            if "vagina" not in f.get("surfaces", {}):
                continue
            V, _ = geom.read_obj("%s/%s" % (fd, f["surfaces"]["vagina"]))
            P = P0.copy()
            P[s2n] = V                       # boundary nodes only; every layer-0 (lumen) node is one of them
            dev = json.load(open("%s/%s" % (fd, f["device"])))
            r_tandem = float(dev.get("r_tandem_mm") or 0.0)
            F = np.asarray(dev["flange_mm"], float)
            a = geom.unit(np.asarray(dev.get("path_axis", dev["tube_axis"]), float))
            C = np.array([P[q].mean(0) for q in rings])
            qv = C - F
            t = qv @ a
            off = np.linalg.norm(qv - np.outer(t, a), axis=1)
            rad = np.array([float(np.linalg.norm(P[q] - C[k], axis=1).mean()) for k, q in enumerate(rings)])
            st = np.array([_perim(P[q]) for q in rings]) / perim0
            j = int(np.argmin(np.abs(t)))
            rows.append(dict(step=int(dev["step"]), phase=dev["phase"], u=float(dev["u"]),
                             advance_mm=float(dev.get("advance_mm", 0.0)), off_max=float(off.max()),
                             off_mean=float(off.mean()), off_at_flange=float(off[j]), station_at_flange=j,
                             r_mean=float(rad.mean()), r_max=float(rad.max()), stretch_max=float(st.max()),
                             n_contacts=dev.get("n_contacts"), min_vol_ratio=dev.get("min_vol_ratio"),
                             off_mm=[round(float(v), 2) for v in off], r_mm=[round(float(v), 2) for v in rad]))
    else:                                    # runs with the container-side metric but no frames
        src = "log.jsonl[wall] (container-side)"
        for ln in open("%s/%s/log.jsonl" % (PT["runs"], args.tag)):
            r = json.loads(ln)
            w = r.get("wall")
            if not w:
                continue
            rows.append(dict(step=r["step"], phase=r["phase"], u=r["u"], advance_mm=float("nan"),
                             off_max=w["lumen_axis_off_mm"]["max"], off_mean=w["lumen_axis_off_mm"]["mean"],
                             off_at_flange=w["lumen_axis_off_mm"]["at_flange"],
                             station_at_flange=w["lumen_axis_off_mm"]["station_at_flange"],
                             r_mean=w["lumen_r_mm"]["mean"], r_max=w["lumen_r_mm"]["max"],
                             stretch_max=w["circ_stretch"]["max"], n_contacts=r.get("n_contacts"),
                             min_vol_ratio=r.get("min_vol_ratio"), off_mm=w["per_station"]["off_mm"],
                             r_mm=w["per_station"]["r_mm"]))
    if not rows:
        raise SystemExit("run %s exported neither frames nor a wall diagnostic" % args.tag)
    rt = r_tandem if r_tandem else float(json.load(open(PT["applicator"] + "/applicator.json"))
                                         ["params"]["r_tandem_mm"]["value"])
    r0 = float(W["lumen_r0_mm"])
    print("\n%s: lumen centreline vs the device travel axis (r0 = %g mm, tandem radius %.2f mm, %d stations)"
          % (args.tag, r0, rt, len(rings)))
    print("  source: %s" % src)
    print("  off = perpendicular distance from a station's lumen centre to the line through the CURRENT flange "
          "along the travel axis")
    print("  step ph      u  advance |   off max   mean  at-flange (station) | lumen r mean  max | stretch | "
          "cont   minV | shaft in lumen?")
    for w in rows:
        clear = w["r_mean"] - rt                     # how far off-axis the centre may be before the shaft is out
        print("  %4d %2s %6.3f %7.1f | %8.2f %6.2f %10.2f (%2d)   | %11.2f %5.2f | %7.3f | %4s %6.3f | %s"
              % (w["step"], w["phase"], w["u"], w["advance_mm"], w["off_max"], w["off_mean"], w["off_at_flange"],
                 w["station_at_flange"], w["r_mean"], w["r_max"], w["stretch_max"], w["n_contacts"],
                 w["min_vol_ratio"] if w["min_vol_ratio"] is not None else float("nan"),
                 "yes" if w["off_at_flange"] < clear else "NO  (off > r - r_tandem = %.2f)" % clear))
    o = np.array([w["off_max"] for w in rows])
    of = np.array([w["off_at_flange"] for w in rows])
    print("\n  lumen centreline offset from the device axis: start %.2f mm -> end %.2f mm, max %.2f mm at step %d"
          % (o[0], o[-1], o.max(), rows[int(o.argmax())]["step"]))
    print("  at the flange station: start %.2f -> end %.2f mm, max %.2f mm" % (of[0], of[-1], of.max()))
    out = dict(tag=args.tag, units="mm", source=src, lumen_r0_mm=r0, r_tandem_mm=rt,
               wall_dir=cfgj.get("vagina_wall_dir", DEFAULT_DIR), n_radial=W["n_radial"], n_theta=W["n_theta"],
               s_mm=[round(float(v), 2) for v in s], r_ref_mm=[round(float(v), 3) for v in rad0],
               definition="off = |(C_k - F) - ((C_k - F).a) a| with C_k the mean of the lumen (layer 0) ring at "
                          "station k, F the current flange and a the travel axis (pose.json insertion_path_alt for "
                          "a wall run); the shaft leaves the lumen when off > lumen radius - tandem radius",
               steps=rows)
    os.makedirs(PT["logs"], exist_ok=True)
    json.dump(out, open("%s/wall_%s_axis.json" % (PT["logs"], args.tag), "w"), indent=1, default=json_default)
    print("  wrote %s/wall_%s_axis.json" % (PT["logs"], args.tag))


# --------------------------------------------------------------------------- report: did the lumen open?
def ring_metrics(P, meta):
    """Per-axial-station lumen radius and inner-ring perimeter of a (possibly deformed) wall state."""
    g = np.asarray(meta["wall"]["grid_index"], int)
    n_ax, n_th, n_rd = meta["wall"]["n_axial"], meta["wall"]["n_theta"], meta["wall"]["n_radial"]
    out = []
    for k in range(n_ax):
        idx_in = np.nonzero((g[:, 0] == k) & (g[:, 1] == 0))[0]
        idx_in = idx_in[np.argsort(g[idx_in, 2])]
        idx_out = np.nonzero((g[:, 0] == k) & (g[:, 1] == n_rd))[0]
        ring = P[idx_in]
        ctr = ring.mean(0)
        per = float(np.linalg.norm(np.diff(np.r_[ring, ring[:1]], axis=0), axis=1).sum())
        r = np.linalg.norm(ring - ctr, axis=1)
        ro = np.linalg.norm(P[idx_out] - ctr, axis=1)     # outer ring about the SAME centre -> wall thickness
        ecc = float(np.linalg.norm(P[idx_out].mean(0) - ctr))   # how far the wall has slid off its own lumen
        q = ring - ctr                                    # vector area of the (possibly non-planar) ring polygon
        area = 0.5 * float(np.linalg.norm(np.cross(q, np.roll(q, -1, axis=0)).sum(0)))
        out.append(dict(k=k, s=meta["wall"]["s"][k], centre=ctr, perimeter_mm=per,
                        r_mean=float(r.mean()), r_min=float(r.min()), r_max=float(r.max()),
                        area_mm2=area, r_outer_mean=float(ro.mean()),
                        thickness_mm=float(ro.mean() - r.mean()), ecc_mm=ecc))
    return out


def report(args):
    PT = paths()
    rd = "%s/%s" % (PT["runs"], args.tag)
    cfgj = json.load(open(rd + "/cfg.json"))
    wall_dir = cfgj.get("vagina_wall_dir", DEFAULT_DIR)
    md = "%s/_scene_%s/vagina/meta.json" % (PT["meshes"], wall_dir)
    if not os.path.exists(md):
        md = "%s/%s/meta.json" % (PT["meshes"], wall_dir)
    meta = json.load(open(md))
    P0 = np.asarray(read_vtk_legacy(os.path.dirname(md) + "/tets.vtk")[0], float)
    u = np.load("%s/final/vagina_u.npy" % rd)
    if len(u) != len(P0):
        raise SystemExit("run %s has %d vagina nodes, the wall mesh has %d" % (args.tag, len(u), len(P0)))
    R0 = ring_metrics(P0, meta)
    R1 = ring_metrics(P0 + u, meta)
    r0 = float(meta["wall"]["lumen_r0_mm"])
    print("\n%s: lumen opening (reference lumen radius r0 = %g mm, %d stations, %d theta)"
          % (args.tag, r0, meta["wall"]["n_axial"], meta["wall"]["n_theta"]))
    print("  s(mm)  ref r  final r_mean  r_min  r_max | perim ref -> final  circ. stretch | thickness  ecc")
    rows = []
    for A, B in zip(R0, R1):
        st = B["perimeter_mm"] / A["perimeter_mm"]
        rows.append(dict(s=A["s"], r_ref=A["r_mean"], r_mean=B["r_mean"], r_min=B["r_min"], r_max=B["r_max"],
                         perim_ref=A["perimeter_mm"], perim=B["perimeter_mm"], stretch=st,
                         thickness_ref_mm=A["thickness_mm"], thickness_mm=B["thickness_mm"], ecc_mm=B["ecc_mm"]))
        print("%7.2f %6.2f %12.2f %6.2f %6.2f | %7.2f -> %7.2f %8.3f      | %5.2f (%4.2f) %5.2f"
              % (A["s"], A["r_mean"], B["r_mean"], B["r_min"], B["r_max"], A["perimeter_mm"],
                 B["perimeter_mm"], st, B["thickness_mm"], A["thickness_mm"], B["ecc_mm"]))
    st = np.array([r["stretch"] for r in rows])
    rmax = np.array([r["r_max"] for r in rows])
    print("\n  max circumferential stretch %.3f at s = %.2f mm;  mean %.3f;  stations with stretch > 1.1: %d of %d"
          % (st.max(), rows[int(st.argmax())]["s"], st.mean(), int((st > 1.1).sum()), len(rows)))
    print("  largest achieved lumen radius %.2f mm at s = %.2f mm (reference %.2f mm)"
          % (rmax.max(), rows[int(rmax.argmax())]["s"], r0))
    summ = json.load(open(rd + "/summary.json")) if os.path.exists(rd + "/summary.json") else {}
    out = dict(tag=args.tag, units="mm", lumen_r0_mm=r0, wall_dir=wall_dir,
               status=summ.get("status"), converged=summ.get("converged"), n_steps=summ.get("n_steps"),
               max_stretch=round(float(st.max()), 4), mean_stretch=round(float(st.mean()), 4),
               max_lumen_radius_mm=round(float(rmax.max()), 3),
               n_stations_stretch_gt_1p1=int((st > 1.1).sum()), stations=rows)
    os.makedirs(PT["logs"], exist_ok=True)
    json.dump(out, open("%s/wall_%s_lumen.json" % (PT["logs"], args.tag), "w"), indent=1, default=json_default)
    print("  wrote %s/wall_%s_lumen.json" % (PT["logs"], args.tag))


# --------------------------------------------------------------------------- render (py -3.11, pyvista)
def _dev_world(PT, origin, R_rows, parts):
    out = {}
    for p in parts:
        V, F = geom.read_obj("%s/%s.obj" % (PT["applicator"], p))
        out[p] = (np.asarray(origin, float) + V @ np.asarray(R_rows, float), F)
    return out


def render(args):
    import pyvista as pv
    pv.OFF_SCREEN = True
    PT = paths()
    os.makedirs(PT["figs"], exist_ok=True)
    rd = "%s/%s" % (PT["runs"], args.tag)
    cfgj = json.load(open(rd + "/cfg.json"))
    wall_dir = cfgj.get("vagina_wall_dir", DEFAULT_DIR)
    root = "%s/_scene_%s" % (PT["meshes"], wall_dir)
    if not os.path.isdir(root):
        root = PT["meshes"]
    meta = json.load(open("%s/vagina/meta.json" % root))
    dev = json.load(open(rd + "/device_final.json"))
    R_rows = np.array([dev["x_app"], dev["y_app"], dev["tube_axis"]], float)
    Vw, Fw = geom.read_obj("%s/vagina/surface.obj" % root)
    s2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
    P0 = read_vtk_legacy("%s/vagina/tets.vtk" % root)[0]
    other = [b for b in ("cervix", "bladder", "rectum") if os.path.exists("%s/%s/surface.obj" % (root, b))]
    idx = json.load(open("%s/bodies.json" % root))

    # per-step states exported by run_hybrid.write_frame (cfg frame_every > 0), read through its index.json:
    # a frame OBJ has the same vertex order as meshes/<body>/surface.obj, i.e. meta.surface_obj_vertex_to_tet_node.
    fd = rd + "/frames"

    def _state(lbl, surfaces, devj):
        V, _ = geom.read_obj("%s/%s" % (fd, surfaces["vagina"]))
        Pf = P0.copy()
        Pf[s2n] = V
        oth = dict((b, geom.read_obj("%s/%s" % (fd, surfaces[b]))) for b in other if b in surfaces)
        return dict(lbl=lbl, P=Pf, F=np.asarray(devj["flange_mm"], float),
                    OV=np.asarray(devj["ovoid_origin_mm"], float), u=float(devj["u"]), others=oth)

    states = []
    fi = fd + "/index.json"
    if os.path.exists(fi):
        fr = [f for f in json.load(open(fi))["frames"] if "vagina" in f.get("surfaces", {})]
        for lbl, i in zip(("initial", "mid-insertion", "final"),
                          sorted(set([0, len(fr) // 2, len(fr) - 1]))):
            states.append(_state("%s (step %d)" % (lbl, fr[i]["step"]), fr[i]["surfaces"],
                                 json.load(open("%s/%s" % (fd, fr[i]["device"])))))
    if not states:                                        # no frames exported: the converged state alone
        u = np.load(rd + "/final/vagina_u.npy")
        states.append(dict(lbl="final", P=P0 + u, F=np.asarray(dev["flange_mm"], float),
                           OV=np.asarray(dev["ovoid_origin_mm"], float), u=float(dev["path_final_u"]),
                           others=dict((b, geom.read_obj("%s/%s/surface.obj" % (root, b))) for b in other)))

    # frame on the wall AND the device path, or the view crops to the tube alone and shows no context
    allp = np.vstack([s["P"] for s in states] + [P0]
                     + [np.asarray(s["F"], float).reshape(1, 3) for s in states]
                     + [np.asarray(s["OV"], float).reshape(1, 3) for s in states])
    ctr = 0.5 * (allp.min(0) + allp.max(0))
    ext = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    pl = pv.Plotter(off_screen=True, shape=(2, len(states)), window_size=(700 * len(states), 1200), border=True)
    pl.set_background("white")
    for col, st in enumerate(states):
        lbl, P, F, OV, uu = st["lbl"], st["P"], st["F"], st["OV"], st["u"]
        dv = _dev_world(PT, F, R_rows, ["tube", "shaft"])
        dv.update(_dev_world(PT, OV, R_rows, ["ovoid_L", "ovoid_R"]))
        for row in (0, 1):
            pl.subplot(row, col)
            if row == 0:                                  # the REST wall as a faint wireframe: makes motion visible
                pl.add_mesh(pv.PolyData(P0[s2n], np.c_[np.full(len(Fw), 3), Fw].ravel()), style="wireframe",
                            color=(0.45, 0.30, 0.50), opacity=0.30, line_width=1)
            wall = pv.PolyData(P[s2n], np.c_[np.full(len(Fw), 3), Fw].ravel())
            if row == 1:                                  # cut-away: clip the wall on the mid-sagittal plane
                wall = wall.clip(normal=(1, 0, 0), origin=ctr, invert=True)
            if wall.n_points:                             # a clip can empty a mesh (pyvista refuses to plot those)
                pl.add_mesh(wall, color=idx["bodies"]["vagina"]["color"], opacity=1.0, show_edges=(row == 1),
                            edge_color=(0.3, 0.3, 0.3), line_width=0.4, smooth_shading=(row == 0))
            if row == 0:
                for b, (V, Fb) in st["others"].items():
                    pl.add_mesh(pv.PolyData(V, np.c_[np.full(len(Fb), 3), Fb].ravel()),
                                color=idx["bodies"][b]["color"], opacity=0.18, smooth_shading=True)
            for p, (V, Fd) in dv.items():
                m = pv.PolyData(V, np.c_[np.full(len(Fd), 3), Fd].ravel())
                if row == 1:
                    m = m.clip(normal=(1, 0, 0), origin=ctr, invert=True)
                if not m.n_points:                        # a device part entirely on the far side of the cut
                    continue
                pl.add_mesh(m, color=(0.10, 0.10, 0.14) if p in ("tube", "shaft") else (0.45, 0.45, 0.52),
                            smooth_shading=True)
            pl.add_axes(xlabel="x=R", ylabel="y=A", zlabel="z=S")
            pl.add_text("%s -- %s  u=%.2f\n%s" % (args.tag, lbl, uu,
                                                  "left lateral" if row == 0 else "mid-sagittal cut-away"),
                        font_size=9, position="lower_left")
            v = np.array([-1.0, 0.0, 0.12])
            v /= np.linalg.norm(v)
            pl.enable_parallel_projection()
            pl.camera_position = [(ctr + 2.2 * ext * v).tolist(), ctr.tolist(), (0, 0, 1)]
            # Under a PARALLEL projection the camera distance does not set the view size -- parallel_scale does
            # (half the view height, in world mm), and zoom() only rescales it.  Setting zoom(1.0) instead put
            # the camera inside the wall and rendered a flat purple field.
            pl.camera.parallel_scale = (0.55 if row == 0 else 0.40) * ext
    fn = "%s/wall_%s_states.png" % (PT["figs"], args.tag)
    pl.screenshot(fn)
    pl.close()
    print("[vagina_wall] wrote %s" % fn)

    # axial cross-sections through the lumen at three stations, reference vs final
    g = np.asarray(meta["wall"]["grid_index"], int)
    n_ax = meta["wall"]["n_axial"]
    # show the station that actually opened most (that is the whole point of the figure), plus two references
    _R0, _R1 = ring_metrics(P0, meta), ring_metrics(states[-1]["P"], meta)
    _st = np.array([b["perimeter_mm"] / a["perimeter_mm"] for a, b in zip(_R0, _R1)])
    ks = sorted(set([int(_st.argmax()), int(n_ax * 0.5), int(n_ax * 0.75)]))
    U = np.asarray(meta["wall"]["frame_u"], float)
    V2 = np.asarray(meta["wall"]["frame_v"], float)
    C = np.asarray(meta["wall"]["centreline"], float)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.4))
    ax = np.atleast_1d(ax)
    lbl_last, P_last = states[-1]["lbl"], states[-1]["P"]
    for q, k in enumerate(ks):
        A = ax[q]
        for lay, sty in ((0, "-"), (meta["wall"]["n_radial"], "--")):
            i = np.nonzero((g[:, 0] == k) & (g[:, 1] == lay))[0]
            i = i[np.argsort(g[i, 2])]
            for Pst, col, nm in ((P0, "0.55", "reference"), (P_last, "tab:purple", lbl_last)):
                q2 = Pst[i] - C[k]
                xy = np.c_[q2 @ U[k], q2 @ V2[k]]
                xy = np.r_[xy, xy[:1]]
                A.plot(xy[:, 0], xy[:, 1], sty, color=col, lw=1.6,
                       label=("%s %s" % (nm, "lumen" if lay == 0 else "outer")) if True else None)
        A.set_aspect("equal")
        A.grid(alpha=0.3)
        A.set_title("station %d,  s = %.1f mm,  circ. stretch %.3f" % (k, meta["wall"]["s"][k], _st[k]),
                    fontsize=10)
        A.set_xlabel("mm")
        if q == 0:
            A.set_ylabel("mm")
            A.legend(fontsize=7)
    fig.suptitle("%s: vaginal wall cross-sections, reference (grey) vs %s (purple); r0 = %g mm"
                 % (args.tag, lbl_last, meta["wall"]["lumen_r0_mm"]), fontsize=11)
    fig.tight_layout()
    fn = "%s/wall_%s_sections.png" % (PT["figs"], args.tag)
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    print("[vagina_wall] wrote %s" % fn)


def straight_shaft(args):
    """Write `applicator/shaft_straight.obj`: the vaginal shaft as a STRAIGHT rod along the SHAFT (vaginal) axis.

    WHY THIS EXISTS -- a geometry error that invalidated four batches of wall runs.  The Stage-1 shaft is a 30 mm
    ARC that leaves the flange tangent to the TUBE axis and bends anteriorly by `angle_deg` (24 deg) to meet the
    shaft axis at its far end: `applicator.json landmarks.shaft_end = [0, 6.192, -29.13]`, i.e. a SAGITTA of
    6.192 mm from its own chord.  The vaginal lumen has radius 4.2-5.0 mm.  A rod that bows 6.19 mm sideways
    CANNOT lie inside a 4.2-5.0 mm tube -- it is geometrically impossible, independently of any mechanics.  The
    measured device-to-wall gaps of 0.1-0.6 mm with `in_contact: true` were therefore the rod pressed against the
    wall from OUTSIDE and sliding along it, not a device inside the lumen failing to dilate it.

    A real tandem-and-ovoid applicator is straight in the vagina and takes its angle at the flange/junction, and
    this patient's own BT label measured a straight vaginal shaft (see the APPLICATOR section of the README).
    This rod runs from the flange along -shaft_axis, i.e. exactly down the lumen centreline: deviation 0 by
    construction, against the arc's 6.192 mm.

    The arc remains the DEFAULT device geometry; the scene opts in with cfg `device_part_files`
    ({"shaft": "shaft_straight"}).  `applicator_venezia.py` owns the device and is not modified.
    """
    PT = paths()
    app = json.load(open(PT["applicator"] + "/applicator.json"))
    pose = json.load(open(PT["applicator"] + "/pose.json"))
    R = np.asarray(pose["device_final"]["R_rows"], float)
    sa = geom.unit(np.asarray(pose["device_final"]["shaft_axis"], float))
    d = geom.unit(-(sa @ R.T))                       # applicator-frame direction from the flange INTO the vagina
    r = float(app["params"]["r_shaft_mm"]["value"])
    L = float(args.length) if args.length else float(app["params"]["shaft_arc_len_mm"]["value"])
    nth = int(args.n_theta)
    nax = max(2, int(round(L / 2.0)) + 1)
    e1 = geom.unit(np.cross(d, [1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.cross(d, [0.0, 1.0, 0.0]))
    e2 = np.cross(d, e1)
    th = 2.0 * np.pi * np.arange(nth) / nth
    ring = np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]
    zs = np.linspace(0.0, L, nax)
    V = np.vstack([z * d[None, :] + r * ring for z in zs])
    c0, c1 = len(V), len(V) + 1
    V = np.vstack([V, (0.0 * d)[None, :], (L * d)[None, :]])

    def vid(k, j):
        return k * nth + (j % nth)

    F = []
    for k in range(nax - 1):
        for j in range(nth):
            F += [[vid(k, j), vid(k, j + 1), vid(k + 1, j + 1)], [vid(k, j), vid(k + 1, j + 1), vid(k + 1, j)]]
    for j in range(nth):                              # caps (flat fans); global orientation fixed below
        F += [[c0, vid(0, j + 1), vid(0, j)]]
        F += [[c1, vid(nax - 1, j), vid(nax - 1, j + 1)]]
    F = np.asarray(F, int)
    vol = geom.mesh_volume(V, F)
    if vol < 0:                                       # keep the closed surface outward-oriented (positive volume)
        F = F[:, ::-1]
        vol = -vol
    dev_arc = float(np.linalg.norm(np.asarray(app["landmarks"]["shaft_end"], float)
                                   - np.dot(np.asarray(app["landmarks"]["shaft_end"], float), d) * d))
    out = "%s/shaft_straight.obj" % PT["applicator"]
    write_obj(out, V, F,
              "STRAIGHT vaginal shaft (Stage 2a): rod of radius %.3f mm, length %.2f mm, from the flange along "
              "-shaft_axis (the vaginal axis) in the APPLICATOR frame\n"
              "replaces the %.1f mm arc whose sagitta (%.3f mm) exceeds the lumen radius; see vagina_wall.py "
              "straight_shaft.__doc__\nunits mm; derived from applicator.json + pose.json (local only)"
              % (r, L, float(app["params"]["shaft_arc_len_mm"]["value"]), 6.192))
    rep = dict(file=out, r_shaft_mm=r, length_mm=L, n_theta=nth, n_axial=nax,
               verts=int(len(V)), tris=int(len(F)), volume_mm3=round(float(vol), 3),
               cylinder_volume_mm3=round(float(np.pi * r * r * L), 3),
               dir_app_frame=np.round(d, 6).tolist(),
               deviation_from_shaft_axis_line_mm=0.0,
               arc_shaft_end_perp_to_this_line_mm=round(dev_arc, 3),
               note="deviation of THIS rod from the shaft-axis line through the flange is 0 by construction; the "
                    "arc's far end lies %.3f mm off that same line" % dev_arc)
    print("[vagina_wall] straight shaft -> %s  r %.2f mm  L %.1f mm  %d verts %d tris  vol %.1f mm3 "
          "(cylinder %.1f)  arc end was %.2f mm off the lumen axis"
          % (out, r, L, len(V), len(F), vol, np.pi * r * r * L, dev_arc), flush=True)
    os.makedirs(PT["logs"], exist_ok=True)
    json.dump(rep, open("%s/shaft_straight.json" % PT["logs"], "w"), indent=1, default=json_default)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--r0", type=float, default=None, help="lumen radius mm (default CFG lumen_r0_mm)")
    b.add_argument("--sweep", action="store_true", help="build every r0 of CFG sweep_r0_mm")
    b.add_argument("--variant", default=None, help="write to meshes/vagina_wall_<variant>")
    b.add_argument("--set", action="append", help="override a CFG key, e.g. --set n_theta=24")
    sh = sub.add_parser("shaft", help="write applicator/shaft_straight.obj (straight vaginal shaft)")
    sh.add_argument("--length", type=float, default=None, help="mm (default: applicator.json shaft_arc_len_mm)")
    sh.add_argument("--n-theta", dest="n_theta", type=int, default=24)
    for nm in ("report", "render", "axis"):
        p = sub.add_parser(nm)
        p.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "report":
        report(args)
    elif args.cmd == "axis":
        axis(args)
    elif args.cmd == "shaft":
        straight_shaft(args)
    elif args.cmd == "render":
        render(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
