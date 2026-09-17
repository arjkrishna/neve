"""Stage 2b MESHING: the vaginal wall as an UNSTRUCTURED tetrahedral mesh (TetGen) around the tapered elliptical lumen.

WHY.  vagina_wall.py builds the wall as a STRUCTURED annulus: n_theta rays per station, one hexahedral cell per
(station, theta, layer), each split into 6 tets by the Kuhn rule.  That grid carries a UNIFORM lumen well (18.5 deg
interior dihedral for the r7 fornix build) but cannot carry a VARYING cross-section: every tapered-lumen build,
whatever the smoothing, thickness floor, ray count or station spacing, collapsed to <= 0.89 deg at stations 23-25
(ten builds, seven hypotheses refuted -- README, Stage 2b).  The cell there is ~1.2 mm thick, ~4 mm wide and sheared
by the vault flare, and a Kuhn split of a sheared thin cell is a sliver by construction.

HERE the wall is defined by the SAME analytic profile -- centreline C(s), parallel-transported frames (U, V),
lumen radius r_in(s, theta) (introitus slit -> vault ellipse sized to the ring, fornix around the cervix) and the
area-conserving r_out(s, theta) -- but the tissue between the two sheets is meshed by TetGen: each ring is sampled
with as many nodes as ITS OWN perimeter needs at a uniform target edge, adjacent rings are zipped into triangles,
the two end annuli close the surface, TetGen preserves that surface (-Y) and fills the interior under a dihedral
bound, then mesh_bodies.fix_slivers repairs what is left.  Nothing about the lumen geometry changes; only the
elements do.

Output: <out>/hybrid/meshes/vagina_wall_<variant>/{surface.obj, tets.vtk, meta.json}, the SAME schema as
vagina_wall.py, so scene_hybrid.py loads it unchanged.  The structured (station, layer, theta) `grid_index` is kept
as a SHIM: every node carries its nearest station, its sheet (0 = lumen, 2 = outer, 1 = interior TetGen node) and
its rank in angle about that station's centre, so wall_metrics / animate_lumen keep working.  Ring node counts now
differ per station (wall.n_theta_per_station); no consumer may assume a rectangular grid any more.

Commands (host, py3.13 with the tetgen wheel dir of mesh_bodies.py):
    python hybrid/vagina_wall_tet.py build --variant tet26 --pk <dir with tetgen> --set 'lumen_profile="device"'
    python hybrid/vagina_wall_tet.py build --variant tet26 --pk <dir> --set edge_mm=1.8   # tet CFG keys work too
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
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402
import vagina_wall as VW  # noqa: E402

CFG_TET = dict(
    method="prism",                  # "prism":  every zipper triangle of the lumen sheet and its matched outer
                                     #           triangle bound a PRISM; each prism is split into 3 tets by the
                                     #           Dompierre rule (quad diagonals through the quad's smallest node id,
                                     #           so neighbours conform).  No Delaunay step at all; the quality is
                                     #           set by the prism shape alone (base ~edge_mm, height = thickness).
                                     # "tetgen": the closed surface is handed to TetGen (-Y, surface preserved).
                                     #           MEASURED: a thin curved shell defeats it -- flat tets with all four
                                     #           nodes on ONE sheet at the vault (stations 25-27, 1.1-1.3 mm thick)
                                     #           and slivers on its own interior points, best 8.2 deg over 20 knob
                                     #           settings (edge 1.5-2.5, mindihedral 10/15, node moves to 0.8 mm,
                                     #           40 optimiser passes, thickness floors 0.8-1.5).
    offset="normal",                 # how the OUTER sheet is placed from the lumen sheet, at the profile's own
                                     #   thickness t(s, theta) = r_out - r_in:
                                     # "normal": along the lumen sheet's node normal.  On the fornix SHELF -- where
                                     #   the lumen widens ~12 mm within one 2 mm station as the rays clear the
                                     #   cervix -- the sheet is nearly perpendicular to the axis, so a radial offset
                                     #   lays the outer sheet almost IN the inner one and every prism there is
                                     #   sheared flat: MEASURED (tetx_p, radial), 28 of the 30 tets under 10 deg
                                     #   sat at station 25, the shelf, where ring counts jump 32 -> 69.  Along the
                                     #   normal the shelf becomes a roof of thickness t and the prisms stay upright.
                                     # "radial": the vagina_wall.py rule (outer = centre + r_out * ray), kept for
                                     #   comparison.
    edge_mm=2.0,                     # mm  target triangle edge on both sheets and the end annuli
    n_theta_min=16,                  # -   floor on nodes per ring (the introitus slit is ~40 mm around)
    profile_n_theta=72,              # -   theta samples of the ANALYTIC profile (fornix ray cast at this resolution);
                                     #     the ring nodes are interpolated from it, so the vault's cervix recess is
                                     #     resolved at 5 deg whatever the ring's own node count
    stagger=True,                    # -   alternate stations start half a step round, so the strip triangles are
                                     #     near-equilateral instead of right-angled
    match_sheets=True,               # -   the outer ring of a station uses the SAME node count and angles as its
                                     #     lumen ring (count from the larger of the two perimeters), so the tissue
                                     #     between the sheets is a stack of prisms and TetGen only picks their
                                     #     diagonals.  MEASURED (tet26, sheets sampled independently): all 15 tets
                                     #     under 10 deg were "crossed" ones with two nodes on each sheet whose
                                     #     sheet edges ran offset and nearly parallel -- a Delaunay sliver between
                                     #     two parallel surfaces, not a thickness limit (12 of the 15 sat where the
                                     #     wall is 1.8-2.6 mm thick).
    wall_overrides=dict(             # vagina_wall.CFG keys this builder changes (a --set still wins):
        lumen_profile="device",      #   the tapered elliptical lumen sized to the ring is the point of this mesh
        min_thickness_mm=1.2,        #   floors raised from 0.8 / 1.2: a 0.8 mm sheet pair under 2 mm triangles
        fornix_min_thickness_mm=1.5),  # is the one place the prism split still slivers (MEASURED: 5.7 vs 11.0 deg)
    fornix_smooth_in_rays_of=72,     # -   vagina_wall's fornix_smooth_theta is a sigma in RAYS (1.2, tuned on the
                                     #     20-ray structured profile = 21.6 deg); the sigma is scaled by
                                     #     profile_n_theta / this, so 72 = the 72-ray profile keeps 1.2 rays = 6 deg.
                                     #     MEASURED: rescaling to the old 21.6 deg (value 20) FOLDS the normal-offset
                                     #     outer sheet at stations 23-24, theta 340-360 deg (7 inverted prisms, min
                                     #     dihedral 2.6 deg); at 6 deg there is no inversion and the mesh passes the
                                     #     10 deg gate.  The wide smoothing was only ever there to save the
                                     #     structured grid from its ray-cast steps, which this mesh does not suffer.
    tet=dict(minratio=1.5, mindihedral=10.0, nobisect=True, maxvolume_factor=3.0,   # TetGen -pq1.5/10 -Y -a
             opt_scheme=7, opt_iterations=10, optmaxdihedral=165.0,
             sliver_target_deg=11.0, sliver_max_boundary_move_mm=0.5, sliver_max_passes=60),
    quality_gate_deg=10.0,           # deg minimum TRUE interior dihedral for tets.gate_ok
)


# --------------------------------------------------------------------------- surface triangulation
def periodic_interp(th, th_s, r_s):
    """r(theta) by linear interpolation of periodic samples r_s taken at th_s = 2 pi j / n."""
    n = len(th_s)
    x = (np.asarray(th, float) % (2.0 * np.pi)) / (2.0 * np.pi) * n
    i0 = np.floor(x).astype(int) % n
    f = x - np.floor(x)
    return r_s[i0] * (1.0 - f) + r_s[(i0 + 1) % n] * f


def zipper(ia, ta, ib, tb):
    """Triangulate the band between two closed rings A and B given as node ids and angles (increasing, [0, 2 pi)).

    Greedy by angle: at each step the ring whose NEXT node comes first in angle is advanced, so the band is a
    single strip of len(A) + len(B) triangles whatever the two node counts.  Winding is (A_i, A_i+1, B_j) and
    (A_i, B_j+1, B_j), which for B 'above' A (along the tangent) and theta counter-clockwise about the tangent gives
    every triangle the +RADIAL normal; the caller flips whole patches where the outward normal is the other way."""
    ia, ib = np.asarray(ia, int), np.asarray(ib, int)
    ta, tb = np.asarray(ta, float), np.asarray(tb, float)
    na, nb = len(ia), len(ib)
    d = (tb - ta[0] + np.pi) % (2.0 * np.pi) - np.pi
    j0 = int(np.argmin(np.abs(d)))
    ib, tb = np.roll(ib, -j0), np.roll(tb, -j0)
    ta_u = np.r_[ta - ta[0], 2.0 * np.pi]                      # unwrapped, starting at 0
    tb_u = d[j0] + np.r_[0.0, np.cumsum(np.diff(tb) % (2.0 * np.pi))]
    tb_u = np.r_[tb_u, tb_u[0] + 2.0 * np.pi]
    F = []
    i = j = 0
    while i < na or j < nb:
        adv_a = (j >= nb) or (i < na and ta_u[i + 1] <= tb_u[j + 1])
        if adv_a:
            F.append([ia[i % na], ia[(i + 1) % na], ib[j % nb]])
            i += 1
        else:
            F.append([ia[i % na], ib[(j + 1) % nb], ib[j % nb]])
            j += 1
    return np.asarray(F, int)


def _perim(pts):
    return float(np.linalg.norm(np.diff(np.r_[pts, pts[:1]], axis=0), axis=1).sum())


def build_surface(W, tcfg):
    """Closed, outward-oriented triangle surface of the wall from the analytic profile W (vagina_wall.build_wall).

    Returns V (n, 3), F (m, 3), per-node (station, layer, theta rank, theta) and per-face patch labels."""
    C, U, Vf, T = W["C"], W["U"], W["V"], W["Tg"]
    n_ax, n_th = int(W["n_ax"]), int(W["n_th"])
    th_s = 2.0 * np.pi * np.arange(n_th) / n_th
    edge, n_min = float(tcfg["edge_mm"]), int(tcfg["n_theta_min"])
    if tcfg.get("offset", "normal") == "normal" and not tcfg.get("match_sheets", True):
        raise SystemExit("offset=normal needs match_sheets=true (the outer sheet is built node by node from the lumen)")
    match = bool(tcfg.get("match_sheets", True))
    counts = dict(inner=[], outer=[])
    # ---- 1. the LUMEN sheet: one ring per station, node count from the larger of the two profile perimeters
    in_pos, in_info, in_rings, in_thick = [], [], [], []
    for k in range(n_ax):
        pers = [_perim(C[k][None, :] + rs[:, None] * (np.cos(th_s)[:, None] * U[k][None, :]
                                                       + np.sin(th_s)[:, None] * Vf[k][None, :]))
                for rs in (W["r_in"][k], W["r_out"][k])]
        n_k = max(n_min, int(round((max(pers) if match else pers[0]) / edge)))
        off = 0.5 if (tcfg["stagger"] and (k % 2 == 1)) else 0.0
        th = 2.0 * np.pi * (np.arange(n_k) + off) / n_k
        r = periodic_interp(th, th_s, W["r_in"][k])
        P = C[k][None, :] + r[:, None] * (np.cos(th)[:, None] * U[k][None, :] + np.sin(th)[:, None] * Vf[k][None, :])
        ids = np.arange(len(in_pos), len(in_pos) + n_k)
        in_pos.extend(P.tolist())
        in_info.extend([(k, 0, j, float(th[j])) for j in range(n_k)])
        in_rings.append((ids, th))
        in_thick.extend(periodic_interp(th, th_s, W["r_out"][k] - W["r_in"][k]).tolist())
        counts["inner"].append(n_k)
    Vin = np.asarray(in_pos, float)
    n_in = len(Vin)
    strip_in = [zipper(in_rings[k][0], in_rings[k][1], in_rings[k + 1][0], in_rings[k + 1][1])
                for k in range(n_ax - 1)]              # lumen strip faces as zipped (before any winding flip)
    # ---- 2. the OUTER sheet
    out_pos, out_info, out_rings = [], [], []
    if tcfg.get("offset", "normal") == "normal":
        # node normals of the lumen sheet, pointing INTO the tissue (the zipper winding gives +radial, see zipper)
        Fi = np.vstack(strip_in)
        fn = np.cross(Vin[Fi[:, 1]] - Vin[Fi[:, 0]], Vin[Fi[:, 2]] - Vin[Fi[:, 0]])      # area-weighted
        nn = np.zeros((n_in, 3))
        for col in range(3):
            np.add.at(nn, Fi[:, col], fn)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        # a normal must never point back into the lumen: check against the radial direction, and flip if so
        rad = np.zeros((n_in, 3))
        for k, (ids, th) in enumerate(in_rings):
            q = Vin[ids] - C[k]
            q -= np.outer(q @ T[k], T[k])
            rad[ids] = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        flip = np.einsum("ij,ij->i", nn, rad) < 0.0
        nn[flip] *= -1.0
        Vout = Vin + np.asarray(in_thick, float)[:, None] * nn
        for k, (ids, th) in enumerate(in_rings):
            out_rings.append((ids + n_in, th))
            out_info.extend([(k, 2, j, float(th[j])) for j in range(len(ids))])
            counts["outer"].append(len(ids))
        out_pos = Vout.tolist()
        normal_info = dict(n_flipped_normals=int(flip.sum()),
                           normal_vs_radial_deg=dict(max=round(float(np.degrees(np.arccos(np.clip(
                               np.einsum("ij,ij->i", nn, rad), -1, 1))).max()), 2)))
    else:
        for k, (ids, th) in enumerate(in_rings):
            if match:
                r = periodic_interp(th, th_s, W["r_out"][k])
                th_o = th
            else:
                prof = C[k][None, :] + W["r_out"][k][:, None] * (np.cos(th_s)[:, None] * U[k][None, :]
                                                                  + np.sin(th_s)[:, None] * Vf[k][None, :])
                n_k = max(n_min, int(round(_perim(prof) / edge)))
                off = 0.5 if (tcfg["stagger"] and (k % 2 == 1)) else 0.0
                th_o = 2.0 * np.pi * (np.arange(n_k) + off) / n_k
                r = periodic_interp(th_o, th_s, W["r_out"][k])
            P = C[k][None, :] + r[:, None] * (np.cos(th_o)[:, None] * U[k][None, :]
                                              + np.sin(th_o)[:, None] * Vf[k][None, :])
            ids_o = np.arange(n_in + len(out_pos), n_in + len(out_pos) + len(th_o))
            out_pos.extend(P.tolist())
            out_info.extend([(k, 2, j, float(th_o[j])) for j in range(len(th_o))])
            out_rings.append((ids_o, th_o))
            counts["outer"].append(len(th_o))
        normal_info = None
    V = np.vstack([Vin, np.asarray(out_pos, float)])
    ninfo = in_info + out_info
    partner = np.r_[np.arange(n_in) + n_in, -np.ones(len(out_pos), int)] if match else -np.ones(len(V), int)
    rings = dict(inner=in_rings, outer=out_rings)
    patches = []                                       # (faces, label, expected outward direction per face)

    def radial_dir(c, k):
        q = c - C[k]
        q = q - (q @ T[k]) * T[k]
        return q / max(1e-12, np.linalg.norm(q))

    for k in range(n_ax - 1):
        for name, sign in (("inner", -1.0), ("outer", +1.0)):
            (ia, ta), (ib, tb) = rings[name][k], rings[name][k + 1]
            F = strip_in[k] if name == "inner" else zipper(ia, ta, ib, tb)
            exp = np.array([sign * radial_dir(V[f].mean(0), k) for f in F])
            patches.append((F, name, exp))
    (ia, ta), (ib, tb) = rings["inner"][0], rings["outer"][0]
    F = zipper(ia, ta, ib, tb)
    patches.append((F, "cap_bottom", np.tile(-T[0], (len(F), 1))))
    (ia, ta), (ib, tb) = rings["inner"][-1], rings["outer"][-1]
    F = zipper(ia, ta, ib, tb)
    patches.append((F, "cap_top", np.tile(T[-1], (len(F), 1))))
    faces, labels, flips = [], [], {}
    for F, name, exp in patches:
        nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        agree = float((np.einsum("ij,ij->i", nrm, exp) > 0).mean())
        if agree < 0.5:                                # the whole patch is wound the other way: flip it
            F = F[:, ::-1]
            agree = 1.0 - agree
        flips.setdefault(name, []).append(agree)
        faces.append(F)
        labels.extend([name] * len(F))
    F = np.vstack(faces)
    labels = np.asarray(labels)
    agree = {k: round(float(min(v)), 4) for k, v in flips.items()}   # worst per-patch agreement after flipping
    prisms = None
    if match:
        Fi = np.vstack(strip_in)
        prisms = np.c_[Fi, partner[Fi]]                # (a, b, c, A, B, C): A outside a through the thickness, etc.
        assert (prisms >= 0).all()
    return V, F, ninfo, labels, dict(n_theta_per_station=counts, patch_orientation_agreement=agree, prisms=prisms,
                                     offset=tcfg.get("offset", "normal"), normal_offset=normal_info)


def split_prisms(prisms):
    """3 tets per prism (a, b, c, A, B, C) by the Dompierre et al. (1999) rule: bring the smallest node id to local
    vertex 0 (a top/bottom swap keeps the vertical pairing, then a rotation), then
        min(I1, I5) < min(I2, I4)  ->  (0,1,2,5) (0,1,5,4) (0,4,5,3)     else  (0,1,2,4) (0,4,2,5) (0,4,5,3)
    Every quad face's diagonal then passes through that quad's smallest node, so the two prisms sharing it split
    it the same way: the mesh is conforming.  Orientation is fixed afterwards by signed volume."""
    out = []
    for p in np.asarray(prisms, int):
        v = p.copy()
        if int(np.argmin(v)) >= 3:                     # smallest id on the top face: swap the faces
            v = np.r_[v[3:], v[:3]]
        r = int(np.argmin(v[:3]))
        v = np.r_[np.roll(v[:3], -r), np.roll(v[3:], -r)]
        if min(v[1], v[5]) < min(v[2], v[4]):
            out += [[v[0], v[1], v[2], v[5]], [v[0], v[1], v[5], v[4]], [v[0], v[4], v[5], v[3]]]
        else:
            out += [[v[0], v[1], v[2], v[4]], [v[0], v[4], v[2], v[5]], [v[0], v[4], v[5], v[3]]]
    return np.asarray(out, np.int64)


# --------------------------------------------------------------------------- the build
def build_one(args, shared, MB):
    cfg = json.loads(json.dumps(VW.CFG))
    tcfg = json.loads(json.dumps(CFG_TET))
    cfg.update(tcfg["wall_overrides"])
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        if k in tcfg:
            tcfg[k] = json.loads(v)
        elif k in tcfg["tet"]:
            tcfg["tet"][k] = json.loads(v)
        else:
            cfg[k] = json.loads(v)
    cfg["n_theta"] = int(tcfg["profile_n_theta"])   # the PROFILE's theta resolution; ring node counts are chosen below
    cfg["fornix_smooth_theta"] = float(cfg["fornix_smooth_theta"]) * cfg["n_theta"] / float(tcfg["fornix_smooth_in_rays_of"])
    cfg["_voxel_mm3"] = shared["vox"]
    PT = shared["PT"]
    if cfg.get("lumen_profile", "uniform") == "device":
        cfg["_ring_r_mm"] = VW.ring_radius_mm(PT, cfg["lumen_ovoid_files"])
    X, a, c = shared["X"], shared["a"], shared["c"]
    t0 = time.time()
    sec, e1, e2 = VW.label_sections(X, a, c, cfg)
    W = VW.build_wall(sec, a, c, e1, e2, cfg, (shared.get("sdf") or {}).get("cervix"))
    sec = W["sec"]
    n_ax = int(W["n_ax"])

    # ---- closed surface at a uniform edge
    Vs, Fs, ninfo, flab, sinfo = build_surface(W, tcfg)
    n_open, n_nonman = VW.manifold_check(Vs, Fs)
    surf_vol = geom.mesh_volume(Vs, Fs)
    if n_open or n_nonman or surf_vol <= 0:
        raise SystemExit("wall surface is not a closed outward-oriented manifold: open %d nonmanifold %d volume %.1f"
                         % (n_open, n_nonman, surf_vol))
    t_surf = time.time() - t0

    # ---- the tets: prism split (default) or TetGen with the surface preserved; then the local sliver fix
    if tcfg["method"] == "prism":
        if sinfo["prisms"] is None:
            raise SystemExit("method=prism needs match_sheets=true")
        T = split_prisms(sinfo["prisms"])
        P = Vs.copy()
        v0 = VW.tet_volumes(P, T)
        n_inv = int(min((v0 < 0).sum(), (v0 > 0).sum()))   # a consistent handedness flips all or none: the rest are
        T, nflip = MB.orient_tets(P, T)                     # genuinely inverted prisms (normal-offset fold-over)
        q_raw = MB.tet_quality(P, T)
        P, T, fix = MB.fix_slivers(P, T, tcfg["tet"]["sliver_target_deg"], tcfg["tet"]["sliver_max_boundary_move_mm"],
                                   tcfg["tet"]["sliver_max_passes"])
        tinfo = dict(route="prism split (Dompierre) + mesh_bodies.fix_slivers", input_surface_preserved=True,
                     flipped=int(nflip), n_prisms=int(len(sinfo["prisms"])), n_inverted_before_orient=n_inv,
                     offset=sinfo["offset"], normal_offset=sinfo["normal_offset"],
                     tetgen_raw=dict(nodes=q_raw["nodes"], tets=q_raw["tets"], min_dihedral_deg=q_raw["min_dihedral_deg"],
                                     n_tets_min_dihedral_lt_10deg=q_raw["n_tets_min_dihedral_lt_10deg"],
                                     note="raw = the prism split before the sliver fix"),
                     sliver_fix=fix)
    else:
        P, T, tinfo = MB.tetrahedralize(Vs, Fs, float(tcfg["edge_mm"]), tcfg["tet"], bool(tcfg["tet"]["nobisect"]))
    P, T = np.asarray(P, float), np.asarray(T, np.int64)
    if len(VW.boundary_faces(T)) != len(Fs):
        print("[vagina_wall_tet] WARNING boundary of the tet mesh (%d faces) != input surface (%d faces)"
              % (len(VW.boundary_faces(T)), len(Fs)), flush=True)
    q = MB.tet_quality(P, T)
    vol = VW.tet_volumes(P, T)
    dih = VW.dihedral_angles(P, T)
    dmin = dih.min(1)
    BF = VW.boundary_faces(T)
    S = np.unique(BF)
    t_tet = time.time() - t0 - t_surf

    # ---- classify every boundary face: the input faces by their own label, any TetGen-added boundary face
    #      (nobisect fallback only) by where its centroid sits between the two sheets
    key = {tuple(sorted(f)): l for f, l in zip(Fs.tolist(), flab)}
    s_node = (P - c) @ a
    st_node = np.abs(s_node[:, None] - np.asarray(sec["s"], float)[None, :]).argmin(1)
    th_s = 2.0 * np.pi * np.arange(int(W["n_th"])) / int(W["n_th"])

    def frac_between_sheets(pt, k):
        qv = pt - W["C"][k]
        ax = qv @ W["Tg"][k]
        lat = qv - ax * W["Tg"][k]
        rho = np.linalg.norm(lat)
        th = np.arctan2(lat @ W["V"][k], lat @ W["U"][k])
        ri = float(periodic_interp([th], th_s, W["r_in"][k])[0])
        ro = float(periodic_interp([th], th_s, W["r_out"][k])[0])
        return (rho - ri) / max(1e-9, ro - ri), th, rho

    # sheet of every GENERATED node (0 lumen, 2 outer; -1 = a node TetGen added).  A boundary face whose three
    # nodes are all on one sheet IS that sheet; a mixed face is an end annulus.  This is exact for the prism mesh
    # and for a preserved TetGen surface -- the end caps in particular, whose diagonals the prism split chooses
    # itself so they never match the input cap faces.  MEASURED (G11 first attempt): sent through the geometric
    # rule below instead, 2 of those cap faces were called "inner", the inner node set then lacked 2 vertices of
    # its own triangles, and SOFA's TriangleCollisionModel segfaulted in updateNormals() at init.
    sheet = -np.ones(len(P), int)
    sheet[:min(len(ninfo), len(P))] = [l for (_, l, _, _) in ninfo][:len(P)]
    blab = []
    n_geom = n_sheet = 0
    for f in BF:
        l = key.get(tuple(sorted(f.tolist())))
        if l is None and (sheet[f] >= 0).all():
            n_sheet += 1
            l = ("inner" if (sheet[f] == 0).all() else "outer" if (sheet[f] == 2).all()
                 else ("cap_bottom" if st_node[f].min() < n_ax // 2 else "cap_top"))
        if l is None:
            n_geom += 1
            cen = P[f].mean(0)
            k = int(np.abs(((cen - c) @ a) - np.asarray(sec["s"], float)).argmin())
            fr, _, _ = frac_between_sheets(cen, k)
            nrm = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
            nrm /= max(1e-12, np.linalg.norm(nrm))
            if k in (0, n_ax - 1) and abs(nrm @ W["Tg"][k]) > 0.6:
                l = "cap_bottom" if k == 0 else "cap_top"
            else:
                l = "inner" if fr < 0.5 else "outer"
        blab.append(l)
    blab = np.asarray(blab)
    fin, fout = blab == "inner", blab == "outer"
    inner = np.unique(BF[fin])
    outer = np.unique(BF[fout])
    both = np.intersect1d(inner, outer)               # a rim node belongs to ONE sheet: decide by its own radius
    if len(both):
        keep_in = []
        for i in both:
            fr, _, _ = frac_between_sheets(P[i], int(st_node[i]))
            keep_in.append(fr < 0.5)
        keep_in = np.asarray(keep_in, bool)
        inner = np.setdiff1d(inner, both[~keep_in])
        outer = np.setdiff1d(outer, both[keep_in])
    for nm, fm, ns in (("inner", fin, inner), ("outer", fout, outer)):   # every collision triangle must map
        missing = np.setdiff1d(np.unique(BF[fm]), ns)
        if len(missing):
            raise SystemExit("%s_triangles use %d nodes that are not in %s_surface: %s" % (nm, len(missing), nm, missing[:8]))
    s_lo, s_hi = float(sec["s"][0]), float(sec["s"][-1])
    apex = np.nonzero(s_node >= s_hi - float(cfg["apex_mm"]))[0]
    inf = np.nonzero(s_node <= s_lo + float(cfg["fixed_inferior_mm"]))[0]

    # ---- grid_index shim: (nearest station, layer 0/1/2, rank in angle about that station's centre)
    layer = np.ones(len(P), int)
    layer[inner] = 0
    layer[outer] = 2
    gidx = np.zeros((len(P), 3), int)
    gidx[:, 0] = st_node
    gidx[:, 1] = layer
    for k in range(n_ax):
        for m in (0, 1, 2):
            idx = np.nonzero((st_node == k) & (layer == m))[0]
            if not len(idx):
                continue
            qv = P[idx] - W["C"][k]
            th = np.arctan2(qv @ W["V"][k], qv @ W["U"][k]) % (2.0 * np.pi)
            gidx[idx[np.argsort(th)], 2] = np.arange(len(idx))
    # every generated ring node keeps its own station / rank (the s-nearest rule reproduces it; assert, don't trust)
    own = np.array([(k, l, j) for (k, l, j, _) in ninfo], int)
    n_pres = int(min(len(own), len(P)))
    same_station = int((gidx[:n_pres, 0] == own[:, 0]).sum())
    same_layer = int((gidx[:n_pres, 1] == own[:, 1]).sum())

    # ---- per-station lumen / outer radii of the MESH (what the scene will see), for the 1-D meta schema
    r_in_st = np.zeros(n_ax)
    r_out_st = np.zeros(n_ax)
    n_th_in = np.zeros(n_ax, int)
    n_th_out = np.zeros(n_ax, int)
    for k in range(n_ax):
        ii = np.nonzero((gidx[:, 0] == k) & (gidx[:, 1] == 0))[0]
        oo = np.nonzero((gidx[:, 0] == k) & (gidx[:, 1] == 2))[0]
        n_th_in[k], n_th_out[k] = len(ii), len(oo)
        ctr = P[ii].mean(0) if len(ii) else W["C"][k]
        r_in_st[k] = np.linalg.norm(P[ii] - ctr, axis=1).mean() if len(ii) else np.nan
        r_out_st[k] = np.linalg.norm(P[oo] - ctr, axis=1).mean() if len(oo) else np.nan
    thick = (W["r_out"] - W["r_in"]).ravel()

    # ---- surface.obj = boundary of the tet mesh
    remap = -np.ones(len(P), np.int64)
    remap[S] = np.arange(len(S))
    Vb, Fb = P[S], remap[BF]
    bvol = geom.mesh_volume(Vb, Fb)
    if bvol < 0:
        Fb = Fb[:, ::-1]
        bvol = -bvol
    b_open, b_nonman = VW.manifold_check(Vb, Fb)
    nrm = np.cross(P[BF[fin, 1]] - P[BF[fin, 0]], P[BF[fin, 2]] - P[BF[fin, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    ctri = P[BF[fin]].mean(1)
    kk = st_node[BF[fin, 0]]
    radial = ctri - W["C"][kk]
    radial -= (np.einsum("ij,ij->i", radial, W["Tg"][kk]))[:, None] * W["Tg"][kk]
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    # boundary faces of a positively oriented tet point OUT of the tissue: on the lumen sheet that is INTO the lumen
    inward_frac = float((np.einsum("ij,ij->i", nrm, radial) < 0).mean()) if fin.any() else float("nan")
    if inward_frac < 0.5:                             # BF winding is the tet's; if it reads outward, flip for the check
        inward_frac = 1.0 - inward_frac

    extra = {}
    if shared.get("sdf"):
        for nm, f in shared["sdf"].items():
            d = f(P[outer])
            extra["outer_signed_dist_to_%s_mm" % nm] = dict(min=round(float(d.min()), 3), n_inside=int((d < 0).sum()))
        d = shared["sdf"]["cervix"](P[apex])
        extra["apex_signed_dist_to_cervix_mm"] = dict(min=round(float(d.min()), 3), max=round(float(d.max()), 3),
                                                      mean=round(float(d.mean()), 3))
    sets = dict(surface_nodes=S, inner_surface=inner, outer_surface=outer, apex=apex, fixed_inferior=inf)
    defs = dict(
        surface_nodes="nodes on the boundary of the tet mesh (faces used by one tet)",
        inner_surface="nodes of the LUMEN sheet (contacts the device); unstructured: one sheet, not a node layer",
        outer_surface="nodes of the OUTER sheet: contacts bladder / rectum / cervix",
        apex="nodes within %g mm of the superior end along the vaginal axis (fornices; follow the cervix)"
             % cfg["apex_mm"],
        fixed_inferior="nodes within %g mm of the inferior end along the vaginal axis (introitus ring; sprung)"
                       % cfg["fixed_inferior_mm"])
    extra.update(inner_triangles=BF[fin].tolist(), outer_triangles=BF[fout].tolist(),
                 n_inner_triangles=int(fin.sum()), n_outer_triangles=int(fout.sum()),
                 n_cap_triangles=int((~fin & ~fout).sum()), n_boundary_faces_classified_by_sheet=n_sheet,
                 n_boundary_faces_classified_geometrically=n_geom,
                 apex_rings=int(len(np.unique(gidx[apex, 0]))), inferior_rings=int(len(np.unique(gidx[inf, 0]))))

    lab_cc = shared["label_cc"]
    wall_cc = float(vol.sum()) / 1000.0
    d = "%s/vagina_wall_%s" % (PT["meshes"], args.variant)
    os.makedirs(d, exist_ok=True)
    VW.write_obj(d + "/surface.obj", Vb, Fb,
                 "vagina WALL surface (UNSTRUCTURED, TetGen): boundary of tets.vtk; vertex k == tets.vtk point "
                 "meta.surface_obj_vertex_to_tet_node[k]\n%s\nunits mm; derived from labels (local only)" % VW.FRAME)
    VW.write_vtk_legacy(d + "/tets.vtk", P, T, "vagina wall tetrahedra (TetGen), %s" % VW.FRAME)
    gate_tets = bool(180.0 - dih.max() > float(tcfg["quality_gate_deg"]) and (vol <= 1e-9).sum() == 0)
    meta = dict(
        body=VW.BODY, priority=3, source_label="vagina", frame=VW.FRAME, units="mm; volumes cc; angles deg",
        role="deformable HOLLOW WALL around a lumen (Stage 2b, UNSTRUCTURED TetGen mesh): the device travels up the "
             "lumen and the wall parts and stretches around it",
        voxel_mm=shared["voxel_mm"], files=dict(surface="surface.obj", tets="tets.vtk", meta="meta.json"),
        volumes_cc=dict(label_raw=shared["label_raw_cc"], after_priority=shared["after_priority_cc"],
                        after_largest_component=lab_cc, surface_mesh=round(bvol / 1000.0, 4),
                        tet_mesh=round(wall_cc, 4),
                        surface_vs_label_pct=round(100 * (bvol / 1000.0 - lab_cc) / lab_cc, 2),
                        tet_vs_label_pct=round(100 * (wall_cc - lab_cc) / lab_cc, 2),
                        note="wall tissue volume is conserved from the label per axial station "
                             "(r_out = sqrt(r_in^2 + A/pi)); the surplus is the thickness floors and the "
                             "unfolded-lumen idealisation, exactly as for the structured wall"),
        surface=dict(tris=int(len(Fb)), verts=int(len(Vb)), open_edges=b_open, nonmanifold_edges=b_nonman,
                     nonmanifold_vertices=0, target_edge_mm=tcfg["edge_mm"],
                     edge_mm=VW.stats(VW.tri_edge_lengths(Vb, Fb)),
                     area_mm2=round(float(VW.tri_areas(Vb, Fb).sum()), 1),
                     gate_ok=bool(b_open == 0 and b_nonman == 0 and bvol > 0),
                     inner_normals_point_into_lumen_frac=round(inward_frac, 4),
                     input_surface=dict(verts=int(len(Vs)), tris=int(len(Fs)), open_edges=n_open,
                                        nonmanifold_edges=n_nonman, volume_cc=round(surf_vol / 1000.0, 4),
                                        preserved_by_tetgen=bool(tinfo["input_surface_preserved"]),
                                        patch_orientation_agreement=sinfo["patch_orientation_agreement"]),
                     method="analytic profile (vagina_wall.build_wall: centreline, parallel-transported frames, "
                            "r_in(s,theta) slit->vault ellipse + fornix, area-conserving r_out) sampled per ring at "
                            "a uniform edge, rings zipped by angle, end annuli closed; the written surface is the "
                            "boundary of tets.vtk"),
        tets=dict(nodes=int(len(P)), tets=int(len(T)), surface_nodes=int(len(S)),
                  target_edge_mm=tcfg["edge_mm"], min_dihedral_deg=round(float(dmin.min()), 2),
                  max_dihedral_deg=round(float(dih.max()), 2),
                  dihedral_convention="min_dihedral_deg / max_dihedral_deg use the mesh_bodies.py formula, which "
                                      "returns the SUPPLEMENT (180 - theta) of the interior dihedral; the TRUE "
                                      "interior angles are interior_dihedral_{min,max}_deg and the gate uses them.",
                  interior_dihedral_min_deg=round(float(180.0 - dih.max()), 2),
                  interior_dihedral_max_deg=round(float(180.0 - dmin.min()), 2),
                  n_tets_interior_dihedral_lt_10deg=int(((180.0 - dih.max(1)) < 10.0).sum()),
                  n_tets_interior_dihedral_lt_20deg=int(((180.0 - dih.max(1)) < 20.0).sum()),
                  sliver_frac_min_dihedral_lt_10deg=round(float((dmin < 10.0).mean()), 5),
                  frac_min_dihedral_lt_15deg=round(float((dmin < 15.0).mean()), 5),
                  n_tets_min_dihedral_lt_10deg=int((dmin < 10.0).sum()),
                  edge_mm=VW.stats(VW.tet_edge_lengths(P, T)), tet_volume_mm3=VW.stats(vol),
                  inverted_or_flat=int((vol <= 1e-9).sum()), flipped=int(tinfo.get("flipped", 0)),
                  gate_ok=gate_tets, mesh_bodies_quality=q, tetgen=tinfo,
                  route="analytic wall surface + TetGen (vagina_wall_tet.py, python %s)" % sys.version.split()[0]),
        axis=dict(centroid=np.round(c, 3).tolist(), axis=np.round(a, 5).tolist(),
                  proj_min=round(float(sec["s_lo"]), 3), proj_max=round(float(sec["s_hi"]), 3),
                  source="pose.json device_final.shaft_axis (= the vagina body's principal axis, rule v2)"),
        node_set_defs=defs, node_set_extra=extra,
        node_set_sizes={k: int(len(v)) for k, v in sets.items()},
        surface_obj_vertex_to_tet_node=S.tolist(),
        node_sets={k: np.asarray(v, int).tolist() for k, v in sets.items()},
        wall=dict(
            reference_state="MODELLING CHOICE, not a measured geometry: the stress-free state is the OPENED lumen "
                            "(lumen_profile) -- the real wall opens by unfolding its rugae, which a continuum mesh "
                            "cannot do.",
            mesh="unstructured (TetGen); grid_index is a SHIM (nearest station, sheet 0/1/2, rank in angle) and "
                 "ring node counts differ per station: see n_theta_per_station",
            cfg={k: v for k, v in cfg.items() if not k.startswith("_")}, tet_cfg=tcfg,
            n_axial=n_ax, n_theta=int(max(n_th_in.max(), n_th_out.max())), n_radial=2,
            n_theta_per_station=dict(inner=n_th_in.tolist(), outer=n_th_out.tolist()),
            axial_step_mm=round(float(np.mean(np.diff(sec["s"]))), 4),
            lumen_r0_mm=cfg["lumen_r0_mm"], E_vagina_kPa=cfg["E_vagina_kPa"], nu_vagina=cfg["nu_vagina"],
            s=np.round(sec["s"], 4).tolist(), area_mm2=np.round(W["A"], 4).tolist(),
            area_raw_mm2=np.round(sec["A_raw"], 4).tolist(), n_vox_per_station=sec["n_vox"].tolist(),
            r_in_mm=np.round(r_in_st, 4).tolist(), r_out_mm=np.round(r_out_st, 4).tolist(),
            thickness_mm=np.round(r_out_st - r_in_st, 4).tolist(),
            thickness_stats=VW.stats(thick), centreline=np.round(W["C"], 4).tolist(),
            profile_theta_deg=np.round(np.degrees(th_s), 3).tolist(),
            r_in_theta_mm=np.round(W["r_in"], 4).tolist(),     # (n_axial, profile_n_theta): the analytic lumen
            r_out_theta_mm=np.round(W["r_out"], 4).tolist(),
            r_in_note="r_in_mm / r_out_mm are per-station MEAN radii of the MESH rings about the lumen ring "
                      "centre; r_in_theta_mm / r_out_theta_mm are the analytic profile at profile_theta_deg",
            fornix=W["fornix"], apex_extension=W["apex_extension"], lumen_profile=W["lumen_profile"],
            frame_u=np.round(W["U"], 6).tolist(), frame_v=np.round(W["V"], 6).tolist(),
            tangent=np.round(W["Tg"], 6).tolist(),
            grid_index=gidx.tolist(),
            grid_index_note="per node (nearest axial station, sheet: 0 = lumen, 1 = interior, 2 = outer, rank in "
                            "angle about the station centre).  Generated ring nodes come first in tets.vtk in "
                            "theta order; TetGen's interior nodes follow.",
            grid_index_check=dict(generated_nodes=n_pres, same_station=same_station, same_layer=same_layer),
            centre_deviation=W["centre_dev"], min_thickness=W["thin"],
            ref_inner_perimeter_mm=round(2.0 * np.pi * cfg["lumen_r0_mm"], 4),
            collision_note="the closed boundary is oriented OUTWARD from the tissue (positive enclosed volume), "
                           "which on the lumen sheet points INTO the lumen -- what SOFA's TriangleCollisionModel "
                           "expects for a closed body."))
    json.dump(meta, open(d + "/meta.json", "w"), indent=1, default=VW.json_default)
    row = dict(variant=args.variant, dir=os.path.basename(d), nodes=int(len(P)), tets=int(len(T)),
               surface_nodes=int(len(S)), n_axial=n_ax, n_theta_per_station=meta["wall"]["n_theta_per_station"],
               vol_cc=round(wall_cc, 3), label_cc=lab_cc, vol_vs_label_pct=meta["volumes_cc"]["tet_vs_label_pct"],
               thickness_mm=[round(float(thick.min()), 2), round(float(np.median(thick)), 2),
                             round(float(thick.max()), 2)],
               interior_dihedral_deg=[meta["tets"]["interior_dihedral_min_deg"],
                                      meta["tets"]["interior_dihedral_max_deg"]],
               n_tets_lt_10deg=meta["tets"]["n_tets_interior_dihedral_lt_10deg"],
               tetgen_min_dihedral_deg=tinfo["tetgen_raw"]["min_dihedral_deg"],
               preserved=bool(tinfo["input_surface_preserved"]), inverted=meta["tets"]["inverted_or_flat"],
               inverted_before_orient=tinfo.get("n_inverted_before_orient"), offset=sinfo["offset"],
               open_edges=b_open, nonmanifold=b_nonman, inner_normals_inward=round(inward_frac, 3),
               sets={k: int(len(v)) for k, v in sets.items()}, gate_ok=gate_tets,
               surface_s=round(t_surf, 2), tetgen_s=round(t_tet, 2), total_s=round(time.time() - t0, 2))
    print("[vagina_wall_tet] %-10s nodes %5d tets %6d (surface %d, preserved %s)  vol %.3f cc (%+.1f %% vs label "
          "%.2f)  thickness %.2f/%.2f/%.2f mm  interior dihedral %.2f-%.2f deg (tetgen raw %.2f)  <10 deg: %d  "
          "inverted %d (pre-orient %s)  open %d  inner-normals-inward %.0f %%  rings/station %d-%d  %.1f s"
          % (args.variant, len(P), len(T), len(S), tinfo["input_surface_preserved"], wall_cc,
             meta["volumes_cc"]["tet_vs_label_pct"], lab_cc, thick.min(), np.median(thick), thick.max(),
             180.0 - dih.max(), 180.0 - dmin.min(), tinfo["tetgen_raw"]["min_dihedral_deg"],
             meta["tets"]["n_tets_interior_dihedral_lt_10deg"], (vol <= 1e-9).sum(),
             tinfo.get("n_inverted_before_orient", "-"), b_open, 100 * inward_frac,
             n_th_in.min(), n_th_out.max(), time.time() - t0), flush=True)
    return row, meta


def build(args):
    pk = args.pk or os.environ.get("APPSIM_PK")
    if pk and pk not in sys.path:
        sys.path.append(pk)
    want_tetgen = CFG_TET["method"] == "tetgen"
    for kv in args.set or []:
        if kv.startswith("method="):
            want_tetgen = "tetgen" in kv
    if want_tetgen:
        try:
            import tetgen  # noqa: F401
        except ImportError:
            sys.exit("tetgen not importable: pip install --target <dir> tetgen ; then --pk <dir> (or APPSIM_PK)")
    import mesh_bodies as MB                     # host only (nibabel / scipy / vtk)
    PT = VW.paths()
    for k in ("meshes", "figs", "logs"):
        os.makedirs(PT[k], exist_ok=True)
    pose = json.load(open(PT["applicator"] + "/pose.json"))
    a = geom.unit(np.asarray(pose["device_final"]["shaft_axis"], float))
    print("[vagina_wall_tet] loading labels from %s" % PT["data"], flush=True)
    masks, aff = MB.load_labels(PT["data"])
    bm, vols, sp = MB.assign_bodies(masks, aff, MB.CFG["connectivity"])
    X = MB.v2w(aff, np.argwhere(bm["vagina"]))
    c = X.mean(0)
    shared = dict(PT=PT, X=X, a=a, c=c, vox=float(np.prod(sp)), voxel_mm=np.round(sp, 4).tolist(),
                  label_raw_cc=vols["vagina"]["label_raw"], after_priority_cc=vols["vagina"]["after_priority"],
                  label_cc=vols["vagina"]["after_largest_component"])
    sdf = {}
    for b in ("cervix", "rectum", "bladder", "corpus"):
        V, F = geom.read_obj("%s/%s/surface.obj" % (PT["meshes"], b))
        sdf[b] = MB.signed_distance_fn(V, F)
    shared["sdf"] = sdf
    row, meta = build_one(args, shared, MB)
    log = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), frame=VW.FRAME, units="mm; volumes cc; angles deg",
               interpreter=sys.version.split()[0], variant=row, cfg=meta["wall"]["cfg"],
               tet_cfg=meta["wall"]["tet_cfg"], node_set_extra={k: v for k, v in meta["node_set_extra"].items()
                                                                  if not k.endswith("_triangles")})
    out = "%s/vagina_wall_tet_%s.json" % (PT["logs"], args.variant)
    json.dump(log, open(out, "w"), indent=1, default=VW.json_default)
    print("[vagina_wall_tet] wrote %s" % out, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--variant", required=True, help="write to meshes/vagina_wall_<variant>")
    b.add_argument("--pk", default=None, help="dir holding the tetgen wheel (pip install --target DIR tetgen)")
    b.add_argument("--set", action="append",
                   help="override a key of vagina_wall.CFG, CFG_TET or CFG_TET['tet'] (json value), e.g. "
                        "--set 'lumen_profile=\"device\"' --set edge_mm=1.8 --set mindihedral=12")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
