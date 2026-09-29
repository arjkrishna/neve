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
    # --- the FORNIX (Stage 2b): wrap the top of the wall AROUND the cervix instead of through it
    fornix=True,                     # -   True: above the portio the lumen is an ANNULUS around the cervix, so the
                                     #     vault is a recess encircling the protruding portio and the ovoids have a
                                     #     real pocket to seat in.  False: the Stage-2a straight round tube, whose
                                     #     top stations sit INSIDE the cervix (MEASURED: the cervix label starts at
                                     #     s = +18.9 mm while the wall ends at s = +27.7 mm, so the top ~9 mm of the
                                     #     tube is buried in the portio -- which is why `wall_outer_exclude` had to
                                     #     drop cervix, and why the angled intrauterine tube has no route out.
    fornix_clearance_mm=0.5,         # mm  gap left between the lumen surface and the cervix surface
    fornix_scan_max_mm=28.0,         # mm  how far out along each ray to look for the cervix
    fornix_scan_step_mm=0.5,         # mm  ray-march sampling step (cost = n_axial * n_theta * range/step SDF calls)
    fornix_smooth_theta=1.2,         # rays      Gaussian sigma, circumferential smoothing of the lumen radius
    fornix_smooth_axial=0.8,         # stations  ... and along the axis
    fornix_smooth_iters=4,           # -   passes of max(smooth(r_in), required); only pushes OUTWARD, so it
                                     #     converges and the lumen provably never re-enters the cervix
    # --- the TAPERED ELLIPTICAL reference lumen (Stage 2b): a stress-free lumen the seated ring actually fits inside
    lumen_profile="uniform",         # "uniform": r_in = lumen_r0_mm everywhere (Stage 2a/2b default, unchanged).
                                     # "device":  r_in is an ELLIPSE per station whose semi-axes taper from a slit at
                                     #   the introitus to a near-circular vault sized to the RING ITSELF (read from
                                     #   `lumen_ovoid_files`, so it auto-fits d22/d26/d30).  Why: MEASURED, a 13 mm
                                     #   ring cannot enter a 6-7 mm lumen, so it slides around the OUTSIDE of the
                                     #   tube (run G10: "wall INSIDE the ovoid solid") and nothing ever opens the
                                     #   lumen.  Elliptical rather than round because the real section is a
                                     #   transverse slit; a round lumen of the same area bulges into rectum/bladder
                                     #   (the r18 build put 125 bladder + 124 rectum nodes inside the wall).
                                     #   Area is still conserved EXACTLY: r_out = sqrt(r_in^2 + A/pi) integrates to
                                     #   A for any r_in(theta).
    lumen_ovoid_files=["ovoid_L_d26s", "ovoid_R_d26s"],  # caps the vault is sized to (applicator/<name>.obj);
                                     #     d26s = 26 mm caps aligned to the STRAIGHT vaginal rod (ring_radius_mm)
    lumen_about="tube",              # axis the parts' radius is taken about: "tube" (the applicator z; right for caps
                                     #     aligned to the tube travelling along the tube) | "rod": the vaginal rod line
                                     #     through the flange (Stage 3: the ring is perpendicular to the TUBE but
                                     #     travels up the vagina along the RODS, sweeping 17.5 mm about them for the
                                     #     26 mm caps, and the cap rods reach 19.7 mm)
    applicator_dir="applicator",     # hybrid/<dir> the parts and applicator.json are read from ("applicator_v3")
    lumen_vault_clear_mm=1.0,        # mm  clearance added to the ring's own radius at the vault
    lumen_low_a_mm=8.0,              # mm  LR semi-axis at the introitus (the slit's long axis)
    lumen_low_b_mm=4.5,              # mm  AP semi-axis at the introitus (the slit's short axis)
    lumen_taper_frac=0.45,           # -   fraction of the vaginal length, measured from the APEX, over which the
                                     #     section widens from the slit to the vault (smoothstep)
    fornix_extend_stations=0,        # -   stations added ABOVE the label's superior end.  Every Stage-2b failure is
                                     #     the TERMINAL ring, which is at once the thinnest, the most flared, and the
                                     #     free end with nothing above it to brace against; extending removes the
                                     #     "terminal" property from the ring that actually carries the ovoid load.
                                     #     The added tissue is EXTRAPOLATED, not measured: A(s) tapers by
                                     #     `fornix_extend_taper` toward the apex.  Recorded in meta as
                                     #     wall.apex_extension, and it costs label fidelity (volumes_cc).
    fornix_extend_taper=0.25,        # -   A(s) at the topmost added station, as a fraction of the last measured A
    fornix_min_thickness_mm=1.2,     # mm  minimum wall thickness at the FORNIX rays only.  Conserving area exactly
                                     #     while the circumference triples thins the vault onto the 0.8 mm global
                                     #     floor, which slivers (MEASURED: 0.57 deg interior dihedral, 98 of 108
                                     #     sub-10-deg tets at stations 24-27).  Costs a little volume, reported in
                                     #     volumes_cc; still inside the Stage-2a thickness range 1.05-2.35 mm.
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


def build_wall(sec, a, c, e1, e2, cfg, sdf_cervix=None):
    """Structured annular mesh: n_theta x (n_radial+1) x n_axial nodes on parallel-transported frames,
    each hexahedral cell split into 6 tets by the Kuhn rule (conforming, including across the theta seam).

    The lumen radius is per (station, theta), not a single number, so the vault can be a FORNIX: above the portio
    every ray is pushed out to the cervix surface (`sdf_cervix`) plus `fornix_clearance_mm`, which makes the top of
    the wall a recess ENCIRCLING the cervix instead of a straight tube driven through it.  Wall tissue area is still
    conserved exactly per station, because r_out = sqrt(r_in^2 + A/pi) gives 1/2 * closed-integral (r_out^2 - r_in^2)
    dtheta = A for ANY r_in(theta)."""
    # ---- optional APEX EXTENSION above the label (Stage 2b).  The label stops at the fornix, so the topmost ring
    # is a free edge; adding stations gives the load-carrying ring material above it.  The cervix SDF is defined
    # above the label, so the extended LUMEN is still measured -- only the wall AREA is extrapolated.
    n_ext = int(cfg.get("fornix_extend_stations", 0) or 0)
    ext = dict(stations=0)
    if n_ext > 0:
        ds = float(np.mean(np.diff(sec["s"])))
        taper = np.linspace(1.0, float(cfg["fornix_extend_taper"]), n_ext + 1)[1:]
        sec = dict(sec)                                  # never mutate the caller's station table
        sec["s"] = np.r_[sec["s"], sec["s"][-1] + ds * np.arange(1, n_ext + 1)]
        sec["A_raw"] = np.r_[sec["A_raw"], float(sec["A_raw"][-1]) * taper]
        sec["c_raw"] = np.vstack([sec["c_raw"], np.tile(sec["c_raw"][-1], (n_ext, 1))])
        sec["n_vox"] = np.r_[sec["n_vox"], np.zeros(n_ext, int)]
        ext = dict(stations=n_ext, axial_step_mm=round(ds, 4), taper=float(cfg["fornix_extend_taper"]),
                   s_added_mm=[round(float(v), 2) for v in sec["s"][-n_ext:]],
                   note="EXTRAPOLATED, not measured: these stations lie above the vagina label's superior end. "
                        "Their lumen still follows the cervix SDF, but their wall area is the last measured A(s) "
                        "tapered by `fornix_extend_taper`. They exist so the ovoid-bearing ring is not a free edge.")

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
    cs, sn = np.cos(th), np.sin(th)

    # ---- the lumen radius, per (station, theta)
    r0 = float(cfg["lumen_r0_mm"])
    r_in = np.full((n_ax, n_th), r0)
    prof = dict(enabled=False)
    if cfg.get("lumen_profile", "uniform") == "device":
        # ellipse per station: semi-axis A_lr along the patient LR direction projected into this station's frame,
        # B_ap perpendicular to it, tapering from the introitus slit to a vault that holds the ring.
        R = float(cfg.get("_ring_r_mm") or 0.0) + float(cfg["lumen_vault_clear_mm"])
        a_lo, b_lo = float(cfg["lumen_low_a_mm"]), float(cfg["lumen_low_b_mm"])
        frac = float(np.clip(cfg["lumen_taper_frac"], 1e-3, 1.0))
        t = (sec["s"] - sec["s"][0]) / max(1e-9, sec["s"][-1] - sec["s"][0])      # 0 = introitus, 1 = apex
        wgt = np.clip((t - (1.0 - frac)) / frac, 0.0, 1.0)
        wgt = wgt * wgt * (3.0 - 2.0 * wgt)                                       # smoothstep
        A_lr = a_lo + (R - a_lo) * wgt
        B_ap = b_lo + (R - b_lo) * wgt
        xhat = np.array([1.0, 0.0, 0.0])                                          # patient RIGHT
        phis = np.zeros(n_ax)
        for k in range(n_ax):
            e = xhat - (xhat @ T[k]) * T[k]
            nrm = np.linalg.norm(e)
            phis[k] = 0.0 if nrm < 1e-9 else np.arctan2((e / nrm) @ V[k], (e / nrm) @ U[k])
            d = th - phis[k]
            r_in[k] = (A_lr[k] * B_ap[k]) / np.sqrt((B_ap[k] * np.cos(d)) ** 2 + (A_lr[k] * np.sin(d)) ** 2)
        prof = dict(enabled=True, ring_radius_mm=round(float(cfg.get("_ring_r_mm") or 0.0), 3),
                    ovoid_files=list(cfg["lumen_ovoid_files"]), vault_semi_axis_mm=round(R, 3),
                    clearance_mm=float(cfg["lumen_vault_clear_mm"]), taper_frac=frac,
                    low_semi_axes_mm=[a_lo, b_lo],
                    a_lr_mm=[round(float(v), 3) for v in A_lr], b_ap_mm=[round(float(v), 3) for v in B_ap],
                    lr_angle_in_frame_deg=[round(float(np.degrees(v)), 2) for v in phis],
                    note="stress-free lumen = an ellipse per station, semi-axes tapering (smoothstep over the top "
                         "`lumen_taper_frac` of the length) from the introitus slit to a vault sized to the ring's "
                         "own radius + clearance.  A MODELLING CHOICE, like lumen_r0_mm: the real wall opens by "
                         "unfolding its rugae, which a continuum mesh cannot do, so the stress-free state is the "
                         "opened section.  Supported by the BT scan, where the vault reaches r 27-30 mm.")
    base_in = r_in.copy()                        # the profile is the floor the fornix may only push further out
    fx = dict(enabled=False, n_rays_on_cervix=0, n_stations_on_cervix=0, first_station=None,
              clearance_mm=float(cfg["fornix_clearance_mm"]), r_in_max_mm=float(r_in.max()))
    if cfg.get("fornix", False) and sdf_cervix is not None:
        # For every ray out of the centreline, the lumen starts where the CERVIX ends: march outward and take the
        # OUTERMOST crossing of the cervix surface (the portio can be entered and left again on one ray).  Rays that
        # never meet the cervix keep r0, so the tube below the vault is unchanged and the join is continuous -- the
        # portio tapers to a point, so its exit radius grows through r0 rather than jumping past it.
        rs = np.arange(0.0, float(cfg["fornix_scan_max_mm"]) + 1e-9, float(cfg["fornix_scan_step_mm"]))
        clear = float(cfg["fornix_clearance_mm"])
        req = base_in.copy()                       # the HARD constraint: the lumen may not lie inside the cervix
        for k in range(n_ax):
            D = cs[:, None] * U[k][None, :] + sn[:, None] * V[k][None, :]          # (n_th, 3) ray directions
            pts = C[k][None, None, :] + rs[None, :, None] * D[:, None, :]          # (n_th, n_r, 3)
            inside = sdf_cervix(pts.reshape(-1, 3)).reshape(n_th, len(rs)) < 0.0
            hit = inside.any(1)
            if not hit.any():
                continue
            r_exit = rs[np.where(inside, np.arange(len(rs))[None, :], -1).max(1)] + clear
            upd = hit & (r_exit > req[k])
            req[k][upd] = r_exit[upd]
            if upd.any():
                fx["n_rays_on_cervix"] += int(upd.sum())
                fx["n_stations_on_cervix"] += 1
                if fx["first_station"] is None:
                    fx["first_station"] = int(k)

        def ring_smooth(M, sig):                   # circumferential: tile x3 so the kernel wraps the seam
            return np.array([gsmooth(np.r_[v, v, v], sig)[n_th:2 * n_th] for v in M])

        # A radial ray-cast is DISCONTINUOUS at the cervix silhouette: MEASURED on this case, r_in stepped by up to
        # 13 mm between adjacent rays only 3 mm apart, and that is what produced 0.57 deg slivers (108 tets below
        # 10 deg, 98 of them at the fornix stations).  Smooth the field, but re-impose the constraint every pass:
        # max(smoothed, required) moves nodes OUTWARD only, so the iteration converges and the lumen can never
        # re-enter the cervix.  The step becomes a ramp; it does not vanish, because the constraint itself is steep.
        step0 = float(np.abs(np.diff(np.c_[req, req[:, :1]], axis=1)).max())
        r_in = req.copy()
        s_th, s_ax = float(cfg["fornix_smooth_theta"]), float(cfg["fornix_smooth_axial"])
        for _ in range(int(cfg["fornix_smooth_iters"])):
            sm = ring_smooth(r_in, s_th) if s_th > 0 else r_in
            if s_ax > 0:
                sm = np.array([gsmooth(v, s_ax) for v in sm.T]).T
            r_in = np.maximum(sm, req)
        fx["enabled"] = True
        fx["r_in_max_mm"] = round(float(r_in.max()), 3)
        fx["max_adjacent_ray_step_mm"] = dict(before=round(step0, 3),
                                              after=round(float(np.abs(np.diff(np.c_[r_in, r_in[:, :1]],
                                                                              axis=1)).max()), 3))
        fx["smoothing"] = dict(sigma_theta_rays=s_th, sigma_axial_stations=s_ax,
                               iters=int(cfg["fornix_smooth_iters"]),
                               note="r_in = max(smooth(r_in), required) per pass: outward-only, so the lumen stays "
                                    "outside the cervix by construction (verified by outer/apex SDF below)")

    r_out = np.sqrt(r_in ** 2 + (A / np.pi)[:, None])
    th_min = np.full_like(r_in, float(cfg["min_thickness_mm"]))
    if fx["enabled"]:                            # the vault is thinned by its own circumference: floor it harder
        th_min[r_in > base_in + 1e-6] = float(cfg["fornix_min_thickness_mm"])
    thin_th = (r_out - r_in) < th_min
    thin = thin_th.any(1)                        # per-station flag, as before
    fx["n_rays_at_thickness_floor"] = int(thin_th.sum())
    r_out = np.maximum(r_out, r_in + th_min)

    def nid(k, m, j):
        return (k * (n_rd + 1) + m) * n_th + (j % n_th)

    P = np.zeros((n_ax * (n_rd + 1) * n_th, 3))
    gidx = np.zeros((len(P), 3), int)            # (axial, radial layer, theta) of every node
    for k in range(n_ax):
        for m in range(n_rd + 1):
            r = r_in[k] + (r_out[k] - r_in[k]) * m / float(n_rd)                   # (n_th,)
            ring = C[k][None, :] + r[:, None] * (cs[:, None] * U[k][None, :] + sn[:, None] * V[k][None, :])
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
    return dict(P=P, T=T4, C=C, U=U, V=V, Tg=T, grid=gidx, s=sec["s"], A=A, r_in=r_in, r_out=r_out, r0=r0, fornix=fx,
                apex_extension=ext, sec=sec, lumen_profile=prof,
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
def ring_radius_mm(PT, files, about="tube"):
    """Radius of the device parts about the axis they TRAVEL along, so the lumen auto-fits whichever parts the run
    loads.  about="tube": max hypot(x, y) about the applicator z (caps built about the tube and travelling along it);
    caps aligned to the straight rod (d26s) use their record's radius_about_shaft_axis_mm.  about="rod": the distance
    from the vaginal rod line through the flange, direction (0, sin th, -cos th) with th = applicator.json angle_deg
    -- Stage 3, where the tube-perpendicular ring and the cap rods travel up the vagina along the rods."""
    rr = 0.0
    d_rod = None
    if about == "rod":
        th = np.radians(float(json.load(open(PT["applicator"] + "/applicator.json"))["params"]["angle_deg"]["value"]))
        d_rod = np.array([0.0, np.sin(th), -np.cos(th)])
    for nm in files:
        tag = nm.split("_")[-1] if nm.startswith("ovoid_") else None
        rec = "%s/ovoids_%s.json" % (PT["applicator"], tag) if tag else None
        if about == "tube" and rec and os.path.exists(rec):
            j = json.load(open(rec))
            if j.get("aligned_to", {}).get("radius_about_shaft_axis_mm") is not None:
                rr = max(rr, float(j["aligned_to"]["radius_about_shaft_axis_mm"]))
                continue
        Vo = np.asarray(geom.read_obj("%s/%s.obj" % (PT["applicator"], nm))[0], float)
        if d_rod is None:
            rr = max(rr, float(np.hypot(Vo[:, 0], Vo[:, 1]).max()))
        else:
            t = Vo @ d_rod
            rr = max(rr, float(np.linalg.norm(Vo - np.outer(t, d_rod), axis=1).max()))
    return rr


def build_one(r0, args, shared):
    cfg = json.loads(json.dumps(CFG))
    cfg["lumen_r0_mm"] = float(r0)
    for kv in args.set or []:
        k, v = kv.split("=")
        cfg[k] = json.loads(v)
    cfg["_voxel_mm3"] = shared["vox"]
    PT = shared["PT"]
    if cfg.get("lumen_profile", "uniform") == "device":
        PTa = dict(PT, applicator=PT["hybrid"] + "/" + cfg.get("applicator_dir", "applicator"))
        cfg["_ring_r_mm"] = ring_radius_mm(PTa, cfg["lumen_ovoid_files"], cfg.get("lumen_about", "tube"))
    X, a, c = shared["X"], shared["a"], shared["c"]
    t0 = time.time()
    sec, e1, e2 = label_sections(X, a, c, cfg)
    W = build_wall(sec, a, c, e1, e2, cfg, (shared.get("sdf") or {}).get("cervix"))
    sec = W["sec"]                               # the apex extension adds stations, so take the table the grid uses
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

    thick = (W["r_out"] - W["r_in"]).ravel()     # r_in/r_out are (n_axial, n_theta) since the fornix landed
    r_in_st, r_out_st = W["r_in"].mean(1), W["r_out"].mean(1)      # per-station means keep meta's 1-D schema
    thick_st = r_out_st - r_in_st
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
            r_out_mm=np.round(r_out_st, 4).tolist(), thickness_mm=np.round(thick_st, 4).tolist(),
            thickness_stats=stats(thick), centreline=np.round(W["C"], 4).tolist(),
            r_in_theta_mm=np.round(W["r_in"], 4).tolist(),        # (n_axial, n_theta): the FORNIX lumen, per ray
            r_out_theta_mm=np.round(W["r_out"], 4).tolist(),
            r_in_note="r_out_mm / thickness_mm are per-station MEANS over theta (1-D, unchanged schema); the full "
                      "per-ray radii are r_in_theta_mm / r_out_theta_mm.  Wall AREA is conserved exactly per "
                      "station for any r_in(theta) because r_out = sqrt(r_in^2 + A/pi).",
            fornix=W["fornix"], apex_extension=W["apex_extension"], lumen_profile=W["lumen_profile"],
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


# =========================================================================== WALL v5 (fix plan S7a), host only
# lumen_profile "packed" (built by vagina_wall_tet.py; nothing here runs for any other profile, and none of these
# keys enters CFG, so every earlier build reproduces byte for byte).
#
#   * The vagina ENDS at the cervix (physician, Q1): the wall is built only from the vagina label BELOW the HR-CTV
#     (a label slab is used while < v5_interface_frac of the FULL label slab lies inside the HR-CTV); the label drawn
#     inside the HR-CTV (to connect the tandem path) is tissue, not vagina.  The top is ATTACHED to the portio: in the START
#     shape every generator of the tube ends on the cervix surface (signed distance v5_start_top_gap_mm), and the
#     top ring is paired with the cervix surface (nearest node + closest-point triangle) for the scene's tie.
#   * STRESS-FREE (rest) shape = the SEATED shape around the applicator_v5 device at pose.json device_final: vault
#     stations = the ring's outline in each station plane + v5_vault_clear_mm, the top station on the ring's top
#     face (no cup above the ring); below the ring the convex hull of the device section + v5_margin_mm
#     (v5_margin_ant_mm anteriorly), and for v5_section "cal" at least the BT vagina section (pelvis frame, minus
#     the wall thickness; CALIBRATED, in-sample).  "pred" uses the device only (PREDICTED; packing volume 0 until
#     the protocol volume is known).
#   * START (collapsed) shape = the label, station by station (smoothed centroid, PCA ellipse of the label's area,
#     angle), keeping its curve; the lumen is a slit (half-height v5_slit_b_mm) inside it.  Rest station k maps to
#     the start at the same fraction f_k of the rest inner-sheet meridian length along each generator.
#   * Wall area per station from the label, thickness floor v5_min_thickness_mm (the plan's "A(s) from the label,
#     1.2 mm floor"); centreline smoothing v5_centre_smooth_mm; r_in < v5_curv_ratio_max x the local radius of
#     curvature (checked, the lower guide is lengthened automatically when violated).
CFG_V5 = dict(
    v5_applicator_dir="applicator_v5",  # hybrid/<dir> holding the device the rest shape is built around (device_final)
    v5_section="cal",                   # "cal" | "pred" (see above)
    v5_vault_clear_mm=1.0,              # vault lumen = the ring outline + this (fix plan S7a)
    v5_margin_mm=3.0,                   # below the ring: device hull + this all round ...
    v5_margin_ant_mm=4.5,               # ... and this anteriorly (contact alarm 3.0 + 1.5)
    v5_top_dz_mm=0.0,                   # top station height above the ring's top face (applicator z, mm); 0 = on the face
    v5_vault_blend_mm=6.0,              # the margin blends from the vault clearance to the lower margins over this length
                                        #   below the ring's bottom face (along the ring axis)
    v5_parallel_below_mm=6.0,           # station planes are exactly parallel to the ring faces from this far below the
                                        #   ring's bottom face upward; below, the normals turn to the vaginal axis
    v5_drape_deg=45.0,                  # under the ring the lumen closes in at most as a cone of this half-angle
    v5_hinge_clear_mm=8.0,              # the bend's hinge line stays this far outside every lumen (outer sheet)
    v5_curv_ratio_max=0.8,              # r_in < this x the local radius of curvature (the G15 fold): checked, reported
    v5_centre_smooth_mm=8.0,            # Gaussian sigma of the label centroids (start) and the device-hull centroids (rest)
    v5_lab_step_mm=1.0,                 # label table spacing along the vaginal axis (slab = area_slab_mm)
    v5_cov_smooth_mm=4.0,               # Gaussian sigma of the per-slab label covariance (PCA ellipse)
    v5_area_smooth_mm=3.0,              # Gaussian sigma of the per-slab label area
    v5_start_section="full",            # start tube sections from the FULL label, cut by the cervix surface | "below"
    v5_interface_frac=0.05,             # a slab is below the HR-CTV while < this fraction of the FULL label slab is inside it
    v5_extrap_fit_mm=8.0,               # above the last clean slab the centroid is extrapolated linearly (fit over this) ...
    v5_extrap_mm=16.0,                  # ... this far, the section held at the mean of the last v5_extrap_hold_mm
    v5_extrap_hold_mm=4.0,
    v5_slit_b_mm=0.5,                   # start: collapsed lumen = a slit of this half-height inside the label section
    v5_min_thickness_mm=1.2,            # rest: r_out >= r_in + this
    v5_start_top_gap_mm=0.2,            # start: a generator ends where the cervix signed distance first reaches this
    v5_ray_step_mm=0.25,                # ray sampling of the device signed distance (rest sections)
    v5_ray_max_mm=40.0,
    v5_guide_rays=24,                   # rays of the provisional sections (guide + meridian spacing)
    v5_meridian_step_mm=1.0,            # provisional section spacing along the centreline for the meridian length
    v5_r_smooth_theta=1.0,              # rays (of 72): circumferential Gaussian sigma of r_in, outward only
    v5_r_smooth_axial=1.0,              # stations: axial Gaussian sigma of r_in, outward only
    v5_r_smooth_iters=3,
    v5_cal_slab_mm=1.6,                 # "cal": BT section = BT (vagina | applicator | ovoid) voxels within this slab
    v5_cal_frame="BONE",                # "cal": BT labels mapped by validation/alignment.json frames[<this>] (pelvis frame)
    v5_gate_mm=0.5,                     # S7c(1): 0 start nodes deeper than this inside v5_gate_bodies
    v5_gate_bodies=["bladder", "rectum", "sigmoid", "cervix"],
    v5_clamp=True,                      # start nodes inside a gate body are pulled toward their generator axis until
    v5_clamp_clear_mm=0.1,              #   the signed distance is >= this (partner node scaled alike); reported
    v5_pack_cc=0.0,                     # "pred": posterior packing volume (fix plan Q5 unanswered: 0 = device-only lumen)
)


def v5_cfg(cfg):
    """The v5 defaults under `cfg` (explicit keys win).  Only for lumen_profile "packed"."""
    out = json.loads(json.dumps(CFG_V5))
    out.update(cfg)
    if out.get("fornix_extend_stations", 0):
        raise SystemExit("lumen_profile 'packed': fornix_extend_stations must be 0 (the vagina ends at the cervix)")
    if out["v5_section"] not in ("cal", "pred"):
        raise SystemExit("v5_section must be 'cal' or 'pred', not %r" % out["v5_section"])
    if float(out["v5_pack_cc"]) != 0.0:
        raise SystemExit("v5_pack_cc > 0 is not implemented (fix plan Q5 unanswered)")
    return out


def sdf_fast(V, F, bbox_pad=None):
    """Signed distance to a closed oriented triangle surface (vtkImplicitPolyDataDistance, negative inside), evaluated
    on arrays.  With bbox_pad, points farther than bbox_pad outside the surface's bounding box get +bbox_pad without a
    vtk call (callers that only test `d <= m` with m < bbox_pad lose nothing)."""
    import vtk
    from vtk.util import numpy_support as ns
    import mesh_bodies as MB
    V = np.asarray(V, float)
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(MB.polydata(V, np.asarray(F, int)))
    lo, hi = V.min(0), V.max(0)

    def f(X):
        X = np.ascontiguousarray(np.atleast_2d(np.asarray(X, float)))
        out = np.empty(len(X))
        sel = np.ones(len(X), bool)
        if bbox_pad is not None:
            sel = np.all((X >= lo - bbox_pad) & (X <= hi + bbox_pad), axis=1)
            out[~sel] = float(bbox_pad)
        if sel.any():
            arr = ns.numpy_to_vtk(np.ascontiguousarray(X[sel]), deep=1)
            res = vtk.vtkDoubleArray()
            ipd.FunctionValue(arr, res)
            out[sel] = ns.vtk_to_numpy(res)
        return out
    return f


def _wsmooth(x, w, sig):
    """Weighted Gaussian smoothing along axis 0 (x (n, ...), weights (n,)), reflected ends."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    sh = x.shape
    X = x.reshape(len(x), -1)
    den = gsmooth(w, sig)
    out = np.stack([gsmooth(w * X[:, j], sig) for j in range(X.shape[1])], 1) / np.maximum(den, 1e-12)[:, None]
    return out.reshape(sh)


def _ring_smooth(r, sig):
    """Circular Gaussian smoothing of r (..., n_theta) along the last axis."""
    if sig <= 0:
        return np.asarray(r, float).copy()
    n = r.shape[-1]
    return np.array([gsmooth(np.r_[v, v, v], sig)[n:2 * n] for v in np.atleast_2d(r)]).reshape(r.shape)


def _slab_stats(S, L, sk, half, vox):
    n = len(sk)
    nv, A, cen, cov = np.zeros(n), np.zeros(n), np.zeros((n, 2)), np.zeros((n, 2, 2))
    for k, s in enumerate(sk):
        m = np.abs(S - s) <= half
        nv[k] = float(m.sum())
        A[k] = nv[k] * vox / (2.0 * half)
        if nv[k] >= 1:
            cen[k] = L[m].mean(0)
            d = L[m] - cen[k]
            cov[k] = d.T @ d / nv[k]
    return nv, A, cen, cov


def v5_label_table(Xb, Xf, inH, a, c, cfg, vox):
    """The START tube from the label, per slab along the vaginal axis a (s from the label centroid c).

    v5_start_section "full" (default): section centroid / PCA ellipse / area from the FULL vagina label, a continuous
    tube through the vagina-HR-CTV interface; the cervix surface then cuts every generator (v5_start_tops), so the
    start keeps the label's part below the HR-CTV -- an oblique interface ends each side at its own height -- and
    ends on the portio.  "below": the label after priority (outside the
    HR-CTV) up to the last clean slab, then extrapolated (centroid linear, section held).
    Either way the WALL area (rest thickness) comes from the label outside the HR-CTV: slabs up to the last with
    < v5_interface_frac of the full label slab inside the HR-CTV, held above.  Lateral coordinates are in the slab
    plane: (U0, V0) = (patient right, a x right ~ anterior)."""
    U0 = geom.ortho(np.array([1.0, 0.0, 0.0]), a)
    V0 = np.cross(a, U0)
    sb = (Xb - c) @ a
    lb = np.c_[(Xb - c) @ U0, (Xb - c) @ V0]
    sf = (Xf - c) @ a
    lf = np.c_[(Xf - c) @ U0, (Xf - c) @ V0]
    half = 0.5 * float(cfg["area_slab_mm"])
    step = float(cfg["v5_lab_step_mm"])
    full = cfg.get("v5_start_section", "full") == "full"
    sk = np.arange(float(sb.min()), float((sf if full else sb).max()) + 1e-9, step)
    n = len(sk)
    nv, A, cen, cov = _slab_stats(sb, lb, sk, half, vox)
    nf, Af, cenf, covf = _slab_stats(sf, lf, sk, half, vox)
    fin = np.array([float(inH[np.abs(sf - s) <= half].mean()) if (np.abs(sf - s) <= half).any() else 0.0 for s in sk])
    thr = float(cfg["v5_interface_frac"])
    k_mid = int(np.argmin(np.abs(sk - 0.5 * (sk[0] + float(sb.max())))))
    bad = np.nonzero(fin[k_mid:] >= thr)[0]
    kc = int(k_mid + bad[0] - 1) if len(bad) else int(np.nonzero(nv > 0)[0][-1])
    s_c = sk[:kc + 1]
    hold = s_c >= s_c[-1] - float(cfg["v5_extrap_hold_mm"])
    A_w = gsmooth(A[:kc + 1], float(cfg["v5_area_smooth_mm"]) / step)
    wh = nv[:kc + 1][hold] / max(nv[:kc + 1][hold].sum(), 1e-12)
    if full:
        w = nf
        cen_all = _wsmooth(cenf, w, float(cfg["v5_centre_smooth_mm"]) / step)
        cov_all = _wsmooth(covf, w, float(cfg["v5_cov_smooth_mm"]) / step)
        A_sec = gsmooth(Af, float(cfg["v5_area_smooth_mm"]) / step)
        s_all = sk
        A_all = np.r_[A_w, np.full(n - kc - 1, float((A_w[hold] * wh).sum()))]
    else:
        w = nv[:kc + 1]
        cen_s = _wsmooth(cen[:kc + 1], w, float(cfg["v5_centre_smooth_mm"]) / step)
        cov_s = _wsmooth(cov[:kc + 1], w, float(cfg["v5_cov_smooth_mm"]) / step)
        n_ext = int(round(float(cfg["v5_extrap_mm"]) / step))
        fit = s_c >= s_c[-1] - float(cfg["v5_extrap_fit_mm"])
        s_e = s_c[-1] + step * np.arange(1, n_ext + 1)
        cen_e = np.stack([np.polyval(np.polyfit(s_c[fit], cen_s[fit, j], 1, w=np.sqrt(w[fit] + 1e-9)), s_e)
                          for j in range(2)], 1)
        cov_e = np.tile((cov_s[hold] * wh[:, None, None]).sum(0), (n_ext, 1, 1))
        s_all = np.r_[s_c, s_e]
        cen_all = np.vstack([cen_s, cen_e])
        cov_all = np.concatenate([cov_s, cov_e])
        A_all = np.r_[A_w, np.full(n_ext, float((A_w[hold] * wh).sum()))]
        A_sec = A_all
    # ---- PCA ellipse of the section's area per slab; the major-axis angle unwrapped mod pi (no 180 deg flips)
    ev, evec = np.linalg.eigh(cov_all)
    lam1 = np.maximum(ev[:, 1], 1e-6)
    lam2 = np.maximum(ev[:, 0], 1e-6)
    psi = np.arctan2(evec[:, 1, 1], evec[:, 0, 1])
    for k in range(1, len(psi)):
        while psi[k] - psi[k - 1] > np.pi / 2:
            psi[k] -= np.pi
        while psi[k] - psi[k - 1] < -np.pi / 2:
            psi[k] += np.pi
    a0, b0 = 2.0 * np.sqrt(lam1), 2.0 * np.sqrt(lam2)
    q = np.sqrt(np.maximum(A_sec, 1e-6) / (np.pi * a0 * b0))
    a_o, b_o = q * a0, q * b0
    bi = float(cfg["v5_slit_b_mm"])
    tmin = float(cfg["v5_min_thickness_mm"])
    b_o = np.maximum(b_o, bi + tmin)
    a_o = np.maximum(a_o, b_o)
    a_i = np.maximum(a_o - (b_o - bi), 1.0)
    P = c[None, :] + s_all[:, None] * a[None, :] + cen_all[:, 0:1] * U0[None, :] + cen_all[:, 1:2] * V0[None, :]
    sig = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return dict(U0=U0, V0=V0, a=a, c=c, step=step, s=s_all, sigma=sig, P=P, cen=cen_all, cov=cov_all, A=A_all,
                A_section=A_sec, a_o=a_o, b_o=b_o, a_i=a_i, b_i=np.full(len(s_all), bi), psi=psi, n_clean=kc + 1,
                s_clean_top=float(s_c[-1]), s_below_top=float(sb.max()), start_section="full" if full else "below",
                raw=dict(s=sk, n_vox_below_hrctv=nv, area_below_hrctv_mm2=A, n_vox_full=nf, area_full_mm2=Af,
                         frac_full_label_in_hrctv=fin, centroid_below=cen, centroid_full=cenf))


def v5_start_points(tab, sig, th, sheet):
    """Start-shape points: generator (theta, sheet) at start arclength sig (arrays broadcast together).
    Parametric ellipse per slab: p = a cos(th - psi) e_psi + b sin(th - psi) e_psi_perp in the slab plane, so a rest
    node at polar angle th (from patient right toward anterior) keeps its angular order in the start."""
    sig = np.asarray(sig, float)
    th = np.asarray(th, float) + np.zeros_like(sig)
    S = tab["sigma"]
    pick = (lambda v: np.interp(sig, S, v))
    C = np.stack([pick(tab["P"][:, j]) for j in range(3)], -1)
    psi = pick(tab["psi"])
    if sheet == 0:
        A, B = pick(tab["a_i"]), pick(tab["b_i"])
    else:
        A, B = pick(tab["a_o"]), pick(tab["b_o"])
    t = th - psi
    ep = np.cos(psi)[..., None] * tab["U0"] + np.sin(psi)[..., None] * tab["V0"]
    eq = -np.sin(psi)[..., None] * tab["U0"] + np.cos(psi)[..., None] * tab["V0"]
    return C + (A * np.cos(t))[..., None] * ep + (B * np.sin(t))[..., None] * eq


def v5_start_tops(tab, sdf_cervix, cfg, n_th=144, dsig=0.2):
    """Start arclength at which each generator (theta grid, sheet 0/2) and the centre column first reach the cervix
    surface (signed distance <= v5_start_top_gap_mm), marching up from the introitus; linear refinement."""
    gap = float(cfg["v5_start_top_gap_mm"])
    S = np.arange(0.0, float(tab["sigma"][-1]) + 1e-9, dsig)
    th = 2.0 * np.pi * np.arange(n_th) / n_th
    out = {}
    for sheet in (0, 2, -1):
        if sheet == -1:
            pts = np.stack([np.interp(S, tab["sigma"], tab["P"][:, j]) for j in range(3)], -1)[None]
        else:
            pts = v5_start_points(tab, S[None, :], th[:, None], sheet)
        d = sdf_cervix(pts.reshape(-1, 3)).reshape(pts.shape[:2]) - gap
        top = np.full(len(d), np.nan)
        for i, row in enumerate(d):
            hit = np.nonzero(row <= 0.0)[0]
            if not len(hit):
                continue
            j = int(hit[0])
            top[i] = S[j] if j == 0 else S[j - 1] + dsig * row[j - 1] / max(row[j - 1] - row[j], 1e-12)
        out[sheet] = top
    if np.isnan(out[0]).any() or np.isnan(out[2]).any() or np.isnan(out[-1]).any():
        raise SystemExit("v5 start: %d generators never reach the cervix within the extrapolated label (raise "
                         "v5_extrap_mm)" % int(np.isnan(out[0]).sum() + np.isnan(out[2]).sum() + np.isnan(out[-1]).sum()))
    return dict(theta=th, inner=out[0], outer=out[2], centre=float(out[-1][0]), dsig=dsig, gap_mm=gap)


def _polar_hull(P2, th):
    """Polar radius at angles th of the convex hull of 2-D points P2 about the origin (inside the hull), and the
    hull's area centroid."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(P2)
    Vh = P2[h.vertices]
    Bh = np.roll(Vh, -1, axis=0)
    E = Bh - Vh
    d = np.c_[np.cos(th), np.sin(th)]
    den = d[:, 0, None] * E[None, :, 1] - d[:, 1, None] * E[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (Vh[None, :, 0] * E[None, :, 1] - Vh[None, :, 1] * E[None, :, 0]) / den
        u = (Vh[None, :, 0] * d[:, 1, None] - Vh[None, :, 1] * d[:, 0, None]) / den
    ok = np.isfinite(t) & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
    r = np.where(ok, t, np.inf).min(1)
    cr = Vh[:, 0] * Bh[:, 1] - Bh[:, 0] * Vh[:, 1]
    ar = 0.5 * cr.sum()
    cxy = ((Vh + Bh) * cr[:, None]).sum(0) / (6.0 * ar)
    return r, cxy


def _polar_poly_centroid(r, th):
    x, y = r * np.cos(th), r * np.sin(th)
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    ar = 0.5 * cr.sum()
    return np.array([((x + x2) * cr).sum(), ((y + y2) * cr).sum()]) / (6.0 * ar)


def v5_frame(n):
    """Station basis: U = patient right projected into the plane normal to n, V = n x U (anterior for n superior)."""
    U = geom.ortho(np.array([1.0, 0.0, 0.0]), n)
    return U, np.cross(n, U)


def _device_hits(C, U, V, th, sdf_parts, margin, cfg):
    """Along each ray (angle th in the plane (U, V) through C) the OUTERMOST point whose device signed distance is
    <= margin(th): points of the margin-offset device surface.  Returns r (n_th,) and a hit mask."""
    step, rmax = float(cfg["v5_ray_step_mm"]), float(cfg["v5_ray_max_mm"])
    rs = np.arange(0.0, rmax + 1e-9, step)
    D = np.cos(th)[:, None] * U[None, :] + np.sin(th)[:, None] * V[None, :]
    X = (C[None, None, :] + rs[None, :, None] * D[:, None, :]).reshape(-1, 3)
    d = np.full(len(X), np.inf)
    for f in sdf_parts:
        d = np.minimum(d, f(X))
    d = d.reshape(len(th), len(rs)) - np.asarray(margin, float)[:, None]
    idx = np.where(d <= 0.0, np.arange(len(rs))[None, :], -1).max(1)
    has = idx >= 0
    j = np.clip(idx, 0, len(rs) - 2)
    d0, d1 = d[np.arange(len(th)), j], d[np.arange(len(th)), j + 1]
    r = np.zeros(len(th))
    r[has] = rs[j[has]] + step * np.clip(-d0[has] / np.maximum(d1[has] - d0[has], 1e-12), 0.0, 1.0)
    return r, has


def v5_section(C, n, sdf_parts, margin, th, cfg, bt=None, area=0.0, recentre=2, bt_weight=1.0):
    """Lumen of one rest station: the plane through C with normal n, polar about an in-plane centre.

    device: the convex hull of the margin-offset device section (_device_hits) + a 2 mm disc about the centre (so
    the centre is always inside).  bt (optional, BT points already restricted to a slab about the plane): the polar
    outline of the BT section minus the wall thickness, sqrt(r_BT^2 - area/pi), as a second lower bound ("cal").
    The polar centre is moved IN-PLANE to the lumen's area centroid `recentre` times (the plane itself never moves).
    Returns the centre, r_in, r_dev, r_bt_in (or None), the frame and the hit count."""
    U, V = v5_frame(n)
    C = np.asarray(C, float).copy()
    disc = 2.0 * np.c_[np.cos(th), np.sin(th)]
    for it in range(int(recentre) + 1):
        r_hit, has = _device_hits(C, U, V, th, sdf_parts, margin, cfg)
        P2 = np.c_[r_hit * np.cos(th), r_hit * np.sin(th)][has]
        r_dev, _ = _polar_hull(np.vstack([P2, disc]), th)
        r_in, r_bt = r_dev.copy(), None
        if bt is not None and len(bt) >= 10:
            q = bt - C[None, :]
            x, y = q @ U, q @ V
            rho, ang = np.hypot(x, y), np.arctan2(y, x) % (2.0 * np.pi)
            nb = len(th)
            b = (np.round(ang / (2.0 * np.pi) * nb).astype(int)) % nb
            rb = np.full(nb, np.nan)
            for i in range(nb):
                m = b == i
                if m.any():
                    rb[i] = rho[m].max()
            if np.isfinite(rb).sum() >= nb // 3:
                ok = np.isfinite(rb)
                xi = np.nonzero(ok)[0]
                rb = np.interp(np.arange(nb), np.r_[xi - nb, xi, xi + nb], np.r_[rb[ok], rb[ok], rb[ok]])
                rb = _ring_smooth(rb, 1.0)
                r_bt = np.sqrt(np.maximum(rb ** 2 - float(area) / np.pi, 0.0))
                r_in = r_dev + float(bt_weight) * np.maximum(r_bt - r_dev, 0.0)   # blends into the vault
        cen = _polar_poly_centroid(r_in, th)
        if it < int(recentre) and np.hypot(*cen) > 0.25:
            C = C + cen[0] * U + cen[1] * V
            continue
        break
    return dict(C=C, r_in=r_in, r_dev=r_dev, r_bt_in=r_bt, centroid=cen, n_hit=int(has.sum()), U=U, V=V)


def v5_margin(th, w_vault, cfg):
    """Device clearance per ray: the vault clearance where w_vault = 1, the lower margins (anterior larger) where 0."""
    m_lo = float(cfg["v5_margin_mm"]) + (float(cfg["v5_margin_ant_mm"]) - float(cfg["v5_margin_mm"])) \
        * np.maximum(0.0, np.sin(th)) ** 2
    return w_vault * float(cfg["v5_vault_clear_mm"]) + (1.0 - w_vault) * m_lo


def _repolar(r, th, dc):
    """Polar radius at angles th, about the point dc (2-D, same plane), of the closed polygon r(th) given about the
    origin: the OUTERMOST crossing of each new ray with the polygon (star-shaped about dc for the lumens here)."""
    Vh = np.c_[r * np.cos(th), r * np.sin(th)] - np.asarray(dc, float)[None, :]
    Bh = np.roll(Vh, -1, axis=0)
    E = Bh - Vh
    d = np.c_[np.cos(th), np.sin(th)]
    den = d[:, 0, None] * E[None, :, 1] - d[:, 1, None] * E[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (Vh[None, :, 0] * E[None, :, 1] - Vh[None, :, 1] * E[None, :, 0]) / den
        u = (Vh[None, :, 0] * d[:, 1, None] - Vh[None, :, 1] * d[:, 0, None]) / den
    ok = np.isfinite(t) & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
    out = np.where(ok, t, -np.inf).max(1)
    if not np.isfinite(out).all():
        raise SystemExit("v5: a lumen polygon is not star-shaped about its smoothed centre")
    return out


def _resample(P, step):
    sig = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keep = np.r_[True, np.diff(sig) > 1e-9]
    P, sig = P[keep], sig[keep]
    S = np.linspace(0.0, sig[-1], max(2, int(np.ceil(sig[-1] / step)) + 1))
    return np.stack([np.interp(S, sig, P[:, j]) for j in range(3)], 1), S


def v5_ring_geometry(appj, F, R, a, c, dev_world, cfg):
    """Ring frame at device_final from applicator.json landmarks (all in the applicator frame): centre, axis (= the
    tube axis), top / bottom face heights, and the lowest point of the ring DISC along the vaginal axis."""
    lm = appj["landmarks"]
    rc_app = np.asarray(lm["ring_centre"], float)
    z_top = float(lm["ring_top_face_z"]) + float(cfg["v5_top_dz_mm"])
    z_bot = float(lm["ring_bottom_face_z"])
    zr = geom.unit(R[2])
    rc = F + rc_app @ R
    top = F + np.array([rc_app[0], rc_app[1], z_top]) @ R
    disc = []
    for p in ("ovoid_L", "ovoid_R"):
        Vw, Va = dev_world[p][0], dev_world[p][2]
        disc.append(Vw[Va[:, 2] >= z_bot - 0.5])
    disc = np.vstack(disc)
    return dict(centre_app=rc_app, centre_w=rc, top_w=top, axis=zr, z_top_app=z_top, z_bot_app=z_bot,
                z_centre_app=float(rc_app[2]), outer_r_mm=float(lm.get("ring_outer_r", np.nan)),
                s_disc_min=float(((disc - c) @ a).min()), s_top=float((top - c) @ a))


def v5_rest_sweep(tab, ring, sdf_parts, cfg, bt_pts=None):
    """The rest station planes as ONE parameter u (mm):
      lower   u in [0, L1]          planes normal to the vaginal axis a at s = s_lo + u (introitus = the start's plane);
      bend    u in [L1, L1 + L2]    planes rotating about ONE hinge line H (direction a x z_ring), from normal a to
                                    normal z_ring; H lies on the concave (anterior) side OUTSIDE every lumen by
                                    v5_hinge_clear_mm, so two planes can only meet on H, never inside the wall;
      upper   u in [.., L1+L2+L3]   planes parallel to the ring's faces (normal z_ring) from v5_parallel_below_mm
                                    below the ring's bottom face up to the top station on its top face.
    H is fixed by the two joins (it lies in the last lower plane and the first upper plane); its clearance from the
    lumen sets how low the bend starts.  Each plane carries a reference point (lumen centroid below, the arc of
    radius R_c about H in the bend, the ring axis above); the section is re-centred in-plane from there."""
    a, c, U0, V0 = tab["a"], tab["c"], tab["U0"], tab["V0"]
    zr, P_rc = ring["axis"], ring["centre_w"]
    k = geom.unit(np.cross(a, zr))
    b = np.cross(k, a)                                          # in the bend plane, normal to a, toward z_ring's lean
    al_max = float(np.arccos(np.clip(a @ zr, -1.0, 1.0)))
    h_par = (ring["z_bot_app"] - ring["z_centre_app"]) - float(cfg["v5_parallel_below_mm"])
    h_top = ring["z_top_app"] - ring["z_centre_app"]
    P_par = P_rc + h_par * zr
    s_par = float((P_par - c) @ a)
    s_lo = float(tab["raw"]["s"][0])
    # ---- lower pass: lumen centroid and anterior (b) reach in planes normal to a, introitus -> s_par
    ss = np.arange(s_lo, s_par + 1e-9, 1.0)
    thg = 2.0 * np.pi * np.arange(int(cfg["v5_guide_rays"])) / int(cfg["v5_guide_rays"])
    marg = v5_margin(thg, 0.0, cfg)
    cal = cfg["v5_section"] == "cal" and bt_pts is not None
    lat = np.zeros((len(ss), 2))
    reach_b = np.zeros(len(ss))
    for i, s in enumerate(ss):
        C0 = c + s * a + np.interp(s, tab["s"], tab["cen"][:, 0]) * U0 + np.interp(s, tab["s"], tab["cen"][:, 1]) * V0
        bt = bt_pts[np.abs((bt_pts - C0) @ a) <= 0.5 * float(cfg["v5_cal_slab_mm"])] if cal else None
        A = float(np.interp(s, tab["s"], tab["A"]))
        sec = v5_section(C0, a, sdf_parts, marg, thg, cfg, bt=bt, area=A)
        q = sec["C"] - c
        lat[i] = [q @ U0, q @ V0]
        r_o = np.sqrt(sec["r_in"] ** 2 + A / np.pi) + float(cfg["v5_min_thickness_mm"])
        X = sec["C"][None, :] + r_o[:, None] * (np.cos(thg)[:, None] * sec["U"] + np.sin(thg)[:, None] * sec["V"])
        reach_b[i] = float(((X - c) @ b).max())
    lat = np.stack([gsmooth(lat[:, j], float(cfg["v5_centre_smooth_mm"])) for j in range(2)], 1)
    # ---- the hinge: H = P_par + p a + q b with p = s1 - s_par, q = -p cot(al_max); lower s1 until H clears the lumen
    clear = float(cfg["v5_hinge_clear_mm"])
    s1 = s_par - 2.0
    while True:
        p = s1 - s_par
        qh = -p / np.tan(al_max)
        H = P_par + p * a + qh * b
        need = float(reach_b[ss >= s1 - 1e-9].max() - (H - c) @ b) + clear
        if need <= 0.0 or s1 - 1.0 < s_lo + 4.0:
            break
        s1 -= 1.0
    C1 = c + s1 * a + np.interp(s1, ss, lat[:, 0]) * U0 + np.interp(s1, ss, lat[:, 1]) * V0
    e0 = C1 - H
    e0 -= (e0 @ k) * k + (e0 @ a) * a
    R_c = float(np.linalg.norm(e0))
    e0 /= R_c
    kx = float((C1 - H) @ k)
    L1, L2, L3 = s1 - s_lo, al_max * R_c, h_top - h_par

    def plane(u):
        if u <= L1:
            s = s_lo + u
            return c + s * a + np.interp(s, ss, lat[:, 0]) * U0 + np.interp(s, ss, lat[:, 1]) * V0, a.copy(), "lower"
        if u <= L1 + L2:
            al = (u - L1) / R_c
            Rk = geom.rodrigues(k, np.degrees(al))
            return H + kx * k + R_c * (Rk @ e0), geom.unit(Rk @ a), "bend"
        h = h_par + (u - L1 - L2)
        return P_rc + h * zr, zr.copy(), "upper"

    return dict(plane=plane, L=L1 + L2 + L3, L1=L1, L2=L2, L3=L3, H=H, k=k, e0=e0, R_c=R_c, s1=float(s1),
                s_par=s_par, h_par=h_par, h_top=h_top, alpha_max_deg=float(np.degrees(al_max)),
                hinge_clear_needed_mm=float(need), lower_s=ss, lower_lat=lat, reach_b=reach_b)


def _drape(r, C, n, below, slope):
    """Top-down: a station below the ring may close in by at most slope x (plane spacing) per station (a cone of
    half-angle atan(slope) hanging from the ring's underside instead of a horizontal shelf).  Outward only."""
    r = r.copy()
    for k in range(len(r) - 2, -1, -1):
        if not below[k]:
            continue
        dh = max(float((C[k + 1] - C[k]) @ n[k + 1]), 0.0)
        r[k] = np.maximum(r[k], r[k + 1] - slope * dh)
    return r


def _smooth_out(r, cfg, axial_scale, fixed=None):
    """Outward-only smoothing of r (n_st, n_th): circular (v5_r_smooth_theta rays of 72) and axial (v5_r_smooth_axial
    x axial_scale stations), re-imposing r >= the input each pass -- never into the device.  Stations flagged in
    `fixed` (the ring span: lumen = ring outline + clearance exactly) are smoothed circumferentially only."""
    bound = r.copy()
    fixed = np.zeros(len(r), bool) if fixed is None else np.asarray(fixed, bool)
    for _ in range(int(cfg["v5_r_smooth_iters"])):
        sm = _ring_smooth(r, float(cfg["v5_r_smooth_theta"]) * r.shape[1] / 72.0)
        if float(cfg["v5_r_smooth_axial"]) > 0:
            ax = np.array([gsmooth(v, float(cfg["v5_r_smooth_axial"]) * axial_scale) for v in sm.T]).T
            sm = np.where(fixed[:, None], sm, ax)
        r = np.maximum(sm, bound)
    return r


def v5_rest_profile(tab, tops, dev, ring, sdf_parts, cfg, bt_pts=None):
    """Rest stations and the analytic rest lumen.

    Planes: v5_rest_sweep -- normal to the vaginal axis from the introitus (the start's plane there), rotating about
    one hinge line outside the lumen, and EXACTLY parallel to the ring's faces from v5_parallel_below_mm below the
    ring upward (no plane slices the ring's flat faces obliquely).  Stations at uniform inner-sheet meridian distance
    (axial_step_mm).  Lumen per station = v5_section with the vault clearance inside the ring span, blending to the
    lower margins below it; under the ring the lumen may close in only as a v5_drape_deg cone; outward-only smoothing.
    Returns the W dict vagina_wall_tet.build_surface consumes plus the v5 tables and the fold checks."""
    n_th = int(cfg["n_theta"])
    th = 2.0 * np.pi * np.arange(n_th) / n_th
    thg = 2.0 * np.pi * np.arange(int(cfg["v5_guide_rays"])) / int(cfg["v5_guide_rays"])
    a, zr, P_rc = tab["a"], ring["axis"], ring["centre_w"]
    z_bot_rel = ring["z_bot_app"] - ring["z_centre_app"]
    blend = float(cfg["v5_vault_blend_mm"])
    slope = 1.0 / np.tan(np.radians(float(cfg["v5_drape_deg"])))
    cal = cfg["v5_section"] == "cal" and bt_pts is not None
    sp = v5_rest_sweep(tab, ring, sdf_parts, cfg, bt_pts)
    S = np.array([0.0, sp["L"]])

    def normal(sv):
        return sp["plane"](sv)[1]

    def point(sv):
        return sp["plane"](sv)[0]

    def w_vault(C):
        h = float((C - P_rc) @ zr)
        return float(np.clip((h - (z_bot_rel - blend)) / blend, 0.0, 1.0))

    def sections(Svals, thv, recentre):
        out = []
        for sv in Svals:
            C0, n = point(sv), normal(sv)
            wv = w_vault(C0)
            f = float(np.clip(sv / S[-1], 0.0, 1.0))
            A = float(np.interp(f * tops["centre"], tab["sigma"], tab["A"]))
            bt = None
            if cal and wv < 1.0:
                bt = bt_pts[np.abs((bt_pts - C0) @ n) <= 0.5 * float(cfg["v5_cal_slab_mm"])]
            sec = v5_section(C0, n, sdf_parts, v5_margin(thv, wv, cfg), thv, cfg, bt=bt, area=A, recentre=recentre,
                             bt_weight=1.0 - wv)
            sec.update(n=n, w_vault=wv, area=A, below=bool(float((C0 - P_rc) @ zr) < z_bot_rel))
            out.append(sec)
        return out

    # ---- provisional sections -> meridian distance -> stations
    Sp = np.arange(0.0, S[-1] + 1e-9, float(cfg["v5_meridian_step_mm"]))
    if Sp[-1] < S[-1] - 1e-6:
        Sp = np.r_[Sp, S[-1]]
    pv = sections(Sp, thg, 2)
    Cp, Np = np.array([q["C"] for q in pv]), np.array([q["n"] for q in pv])
    rp = _smooth_out(_drape(np.array([q["r_in"] for q in pv]), Cp, Np, np.array([q["below"] for q in pv]), slope), cfg,
                     float(cfg["axial_step_mm"]) / float(cfg["v5_meridian_step_mm"]),
                     fixed=np.array([not q["below"] for q in pv]))
    # meridian distance between provisional sections: plane gap (along the upper normal) and mean-radius change --
    # blind to the in-plane re-centring, whose jitter would otherwise bunch the stations
    dhp = np.array([max(float((Cp[k + 1] - Cp[k]) @ Np[k + 1]), 0.0) for k in range(len(pv) - 1)])
    dm = np.hypot(dhp, np.diff(rp.mean(1)))
    Sm = np.r_[0.0, np.cumsum(dm)]
    n_ax = max(8, int(round(Sm[-1] / float(cfg["axial_step_mm"]))) + 1)
    Sk = np.interp(np.linspace(0.0, Sm[-1], n_ax), Sm, Sp)
    fk = np.linspace(0.0, 1.0, n_ax)                        # fraction of the inner meridian (rest -> start map)
    # ---- final sections
    fs = sections(Sk, th, 2)
    C = np.array([q["C"] for q in fs])
    Tn = np.array([q["n"] for q in fs])
    U = np.array([q["U"] for q in fs])
    V = np.array([q["V"] for q in fs])
    r_dev = np.array([q["r_dev"] for q in fs])
    r_bt = np.array([q["r_bt_in"] if q["r_bt_in"] is not None else np.full(n_th, np.nan) for q in fs])
    below = np.array([q["below"] for q in fs])
    wv_k = np.array([q["w_vault"] for q in fs])
    A_k = np.array([float(np.interp(fk[k] * tops["centre"], tab["sigma"], tab["A"])) for k in range(n_ax)])
    r_in = np.array([q["r_in"] for q in fs])
    # ---- smooth centre curve: ring axis inside the ring span, lumen centroids below; Gaussian over the stations'
    #      meridian spacing (v5_centre_smooth_mm), each point put back INTO its own plane, r re-sampled about it
    tgt = C.copy()
    hk = (C - P_rc) @ zr
    inring = hk >= z_bot_rel
    tgt[inring] = P_rc[None, :] + hk[inring, None] * zr[None, :]
    sig_st = float(cfg["v5_centre_smooth_mm"]) / float(np.mean(np.diff(np.linspace(0.0, Sm[-1], n_ax))))
    sm = np.stack([gsmooth(tgt[:, j], sig_st) for j in range(3)], 1)
    wp = np.clip((hk - z_bot_rel) / max(-z_bot_rel, 1e-9), 0.0, 1.0)
    wp = (wp * wp * (3.0 - 2.0 * wp))[:, None]                    # 0 at the ring's bottom face, 1 from its centre up:
    sm = wp * tgt + (1.0 - wp) * sm                               # the upper vault stays exactly coaxial with the ring
    for k in range(n_ax):
        q = sm[k] - C[k]
        q -= (q @ Tn[k]) * Tn[k]
        dc = np.array([q @ U[k], q @ V[k]])
        if np.hypot(*dc) > 1e-9:
            r_in[k] = _repolar(r_in[k], th, dc)
            r_dev[k] = _repolar(r_dev[k], th, dc)
            if np.isfinite(r_bt[k]).all():
                r_bt[k] = _repolar(r_bt[k], th, dc)
            C[k] = C[k] + q
    r_in = _smooth_out(_drape(r_in, C, Tn, below, slope), cfg, 1.0, fixed=~below)
    tmin = float(cfg["v5_min_thickness_mm"])
    r_out = np.maximum(np.sqrt(r_in ** 2 + (A_k / np.pi)[:, None]), r_in + tmin)
    # ---- fold checks: adjacent station planes must not meet inside the wall
    Xi = C[:, None, :] + r_in[:, :, None] * (np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :])
    Xo = C[:, None, :] + r_out[:, :, None] * (np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :])
    up = np.array([min(float(((Xi[k + 1] - C[k]) @ Tn[k]).min()), float(((Xo[k + 1] - C[k]) @ Tn[k]).min()))
                   for k in range(n_ax - 1)])
    dn = np.array([max(float(((Xi[k] - C[k + 1]) @ Tn[k + 1]).max()), float(((Xo[k] - C[k + 1]) @ Tn[k + 1]).max()))
                   for k in range(n_ax - 1)])
    kind = np.array([sp["plane"](v)[2] for v in Sk])
    hinge_clear = np.inf
    for kk in np.nonzero(kind == "bend")[0]:
        e = C[kk] - sp["H"]
        e -= (e @ sp["k"]) * sp["k"]
        e /= np.linalg.norm(e)
        hinge_clear = min(hinge_clear, float(((Xi[kk] - sp["H"]) @ e).min()), float(((Xo[kk] - sp["H"]) @ e).min()))
    dal = np.array([np.arccos(np.clip(Tn[k] @ Tn[k + 1], -1.0, 1.0)) for k in range(n_ax - 1)])
    dh = np.array([float((C[k + 1] - C[k]) @ Tn[k + 1]) for k in range(n_ax - 1)])
    R_pl = np.where(dal > 1e-9, np.abs(dh) / np.maximum(np.tan(dal), 1e-12), np.inf)
    R_pl = np.r_[R_pl, np.inf]
    ratio = r_in.max(1) / R_pl
    # centreline curvature (station centres, the plan's r_in < 0.8 R): circumradius of consecutive triples
    Rc = np.full(n_ax, np.inf)
    for k in range(1, n_ax - 1):
        p0, p1, p2 = C[k - 1], C[k], C[k + 1]
        a_, b_, c_ = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p2 - p0)
        ar2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if ar2 > 1e-9:
            Rc[k] = a_ * b_ * c_ / (2.0 * ar2)
    return dict(W=dict(C=C, U=U, V=V, Tg=Tn, r_in=r_in, r_out=r_out, n_ax=n_ax, n_th=n_th),
                th=th, Sk=Sk, Sm=Sm, Sp=Sp, fk=fk, A=A_k, w_vault=wv_k, below_ring=below, r_dev=r_dev, r_bt_in=r_bt,
                n_hit=np.array([q["n_hit"] for q in fs]), R_planes=R_pl, curv_ratio=ratio, R_centre=Rc,
                centre_curv_ratio=r_in.max(1) / Rc, fold=dict(min_next_above_plane_mm=round(float(up.min()), 4),
                                                              max_prev_above_next_plane_mm=round(float(dn.max()), 4),
                                                              n_pairs_crossing=int(((up <= 0) | (dn >= 0)).sum())),
                sweep=dict(L_lower_mm=round(sp["L1"], 3), L_bend_mm=round(sp["L2"], 3), L_upper_mm=round(sp["L3"], 3),
                           s_bend_start_mm=round(sp["s1"], 3), s_bend_end_mm=round(sp["s_par"], 3),
                           bend_deg=round(sp["alpha_max_deg"], 3), R_centre_mm=round(sp["R_c"], 3),
                           hinge_point=np.round(sp["H"], 4).tolist(), hinge_dir=np.round(sp["k"], 6).tolist(),
                           hinge_clear_min_mm=round(hinge_clear, 3), stations_bend=[int(v) for v in np.nonzero(kind == "bend")[0]]),
                kind=kind)


# --------------------------------------------------------------------------- scene support (container, py3.8)
# Several containers of one batch (run_docker_par.sh) build the SAME shadow root at the same instant when they
# share a wall variant, and a plain open(path, "w") is visible to the others as an EMPTY file: MEASURED (run
# WB_a5) as `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` on bodies.json while its three siblings
# proceeded normally.  Every file here is therefore written to a private temp name and os.replace()d into place,
# which is atomic.
WALL_V5_EXTRA_FILES = ("start.vtk", "start_surface.obj")


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
        fns = ["surface.obj", "tets.vtk", "meta.json"]
        if b == "vagina":                         # wall v5 also carries its START shape (absent for every older wall)
            fns += [fn for fn in WALL_V5_EXTRA_FILES if os.path.exists(src + "/" + fn)]
        for fn in fns:
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
