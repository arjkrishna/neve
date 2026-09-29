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
    if cfg.get("lumen_profile") == "packed":         # wall v5 (fix plan S7a): its own builder; nothing below changes
        cfg["n_theta"] = int(tcfg["profile_n_theta"])
        cfg["_voxel_mm3"] = shared["vox"]
        return build_one_v5(args, shared, MB, cfg, tcfg)
    cfg["n_theta"] = int(tcfg["profile_n_theta"])   # the PROFILE's theta resolution; ring node counts are chosen below
    cfg["fornix_smooth_theta"] = float(cfg["fornix_smooth_theta"]) * cfg["n_theta"] / float(tcfg["fornix_smooth_in_rays_of"])
    cfg["_voxel_mm3"] = shared["vox"]
    PT = shared["PT"]
    if cfg.get("lumen_profile", "uniform") == "device":
        PTa = dict(PT, applicator=PT["hybrid"] + "/" + cfg.get("applicator_dir", "applicator"))
        cfg["_ring_r_mm"] = VW.ring_radius_mm(PTa, cfg["lumen_ovoid_files"], cfg.get("lumen_about", "tube"))
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


# --------------------------------------------------------------------------- wall v5 (fix plan S7a): lumen_profile "packed"
def _is_packed(args):
    return any(kv.split("=", 1)[0] == "lumen_profile" and "packed" in kv.split("=", 1)[1] for kv in (args.set or []))


def _v5_bt_points(PT, cfg):
    """BT (vagina | updated applicator | ovoid) voxel centres mapped into the preBT world by validation/alignment.json
    frames[v5_cal_frame] (y_pre = R x_BT + t: the pelvis frame).  The updated applicator label is read from
    APPSIM_APP_LABEL, else <data>/../BT_MRI_label_applicator.nii (as applicator_venezia.py)."""
    import nibabel as nib
    app_lab = os.environ.get("APPSIM_APP_LABEL") or os.path.dirname(PT["data"].rstrip("/")) + "/BT_MRI_label_applicator.nii"
    files = [PT["data"] + "/BT_MRI_label_vagina.nii", app_lab, PT["data"] + "/BT_MRI_label_ovoid.nii"]
    m, aff = None, None
    for fn in files:
        im = nib.load(fn)
        x = np.asarray(im.dataobj) > 0
        if m is None:
            m, aff = x, im.affine.copy()
        else:
            if x.shape != m.shape or not np.allclose(im.affine, aff):
                raise SystemExit("v5 cal: %s is not on the BT vagina label's grid" % fn)
            m |= x
    ijk = np.argwhere(m)
    Xbt = ijk @ aff[:3, :3].T + aff[:3, 3]
    al = json.load(open(os.path.dirname(PT["hybrid"]) + "/validation/alignment.json"))
    f = al["frames"][cfg["v5_cal_frame"]]
    Rb, tb = np.asarray(f["R_BT_to_pre"], float), np.asarray(f["t_BT_to_pre"], float)
    return Xbt @ Rb.T + tb, dict(files=files, frame=cfg["v5_cal_frame"], n_vox=int(len(ijk)),
                                 volume_cc=round(float(len(ijk) * abs(np.linalg.det(aff[:3, :3]))) / 1000.0, 3))


def _periodic(th_grid, vals, th):
    n = len(th_grid)
    x = (np.asarray(th, float) % (2.0 * np.pi)) / (2.0 * np.pi) * n
    i0 = np.floor(x).astype(int) % n
    f = x - np.floor(x)
    return vals[i0] * (1.0 - f) + vals[(i0 + 1) % n] * f


def _closed_len(P):
    return float(np.linalg.norm(np.diff(np.r_[P, P[:1]], axis=0), axis=1).sum())


def build_one_v5(args, shared, MB, cfg, tcfg):
    """lumen_profile "packed": rest (seated) shape around applicator_v5 + a collapsed START shape on the same mesh
    (start.vtk), node sets and the cervix pairing of the vault top (vagina_wall.py, WALL v5)."""
    PT = shared["PT"]
    cfg = VW.v5_cfg(cfg)
    if cfg.get("fornix", False):
        cfg["fornix"] = False                        # fix plan S7a: no ray-cast to the rest cervix (fornix_mode 'ring')
    t0 = time.time()
    a, c = shared["a"], shared["c"]
    AD = PT["hybrid"] + "/" + cfg["v5_applicator_dir"]
    appj = json.load(open(AD + "/applicator.json"))
    pose = json.load(open(AD + "/pose.json"))
    F = np.asarray(pose["device_final"]["flange"], float)
    R = np.asarray(pose["device_final"]["R_rows"], float)
    parts = list(appj.get("tandem_parts", ["tube", "shaft"])) + list(appj.get("ring_parts", []))
    if not appj.get("ring_parts"):
        raise SystemExit("lumen_profile 'packed' needs a ring device (applicator.json ring_parts): %s" % AD)
    pad = max(float(cfg["v5_margin_mm"]), float(cfg["v5_margin_ant_mm"]), float(cfg["v5_vault_clear_mm"])) + 1.0
    dev, sdf_parts = {}, []
    for p in parts:
        Va, Fa = geom.read_obj("%s/%s.obj" % (AD, p))
        Va, Fa = np.asarray(Va, float), np.asarray(Fa, int)
        Vw = F + Va @ R
        dev[p] = (Vw, Fa, Va)
        sdf_parts.append(VW.sdf_fast(Vw, Fa, bbox_pad=pad))
    sdf5 = {}
    for b in ("cervix", "bladder", "rectum", "sigmoid"):
        Vb, Fb = geom.read_obj("%s/%s/surface.obj" % (PT["meshes"], b))
        sdf5[b] = VW.sdf_fast(np.asarray(Vb, float), np.asarray(Fb, int))
    # ---- START: the label below the HR-CTV, station by station; every generator ends on the cervix surface
    tab = VW.v5_label_table(shared["X"], shared["Xf"], shared["inH"], a, c, cfg, shared["vox"])
    tops = VW.v5_start_tops(tab, sdf5["cervix"], cfg)
    # ---- REST: the seated shape around the device at device_final
    ring = VW.v5_ring_geometry(appj, F, R, a, c, dev, cfg)
    bt_pts = shared.get("bt_pts") if cfg["v5_section"] == "cal" else None
    prof = VW.v5_rest_profile(tab, tops, dev, ring, sdf_parts, cfg, bt_pts=bt_pts)
    W = prof["W"]
    n_ax = int(W["n_ax"])
    t_prof = time.time() - t0

    # ---- surface + prism tets on the REST shape (the stress-free state the scene loads as tets.vtk)
    Vs, Fs, ninfo, flab, sinfo = build_surface(W, tcfg)
    n_open, n_nonman = VW.manifold_check(Vs, Fs)
    surf_vol = geom.mesh_volume(Vs, Fs)
    if n_open or n_nonman or surf_vol <= 0:
        raise SystemExit("v5 wall surface is not a closed outward-oriented manifold: open %d nonmanifold %d volume %.1f"
                         % (n_open, n_nonman, surf_vol))
    if tcfg["method"] != "prism" or sinfo["prisms"] is None:
        raise SystemExit("lumen_profile 'packed' needs method=prism (the start shape is mapped node by node)")
    T = split_prisms(sinfo["prisms"])
    P = Vs.copy()
    v0 = VW.tet_volumes(P, T)
    n_inv = int(min((v0 < 0).sum(), (v0 > 0).sum()))
    T, nflip = MB.orient_tets(P, T)
    q_raw = MB.tet_quality(P, T)
    P, T, fix = MB.fix_slivers(P, T, tcfg["tet"]["sliver_target_deg"], tcfg["tet"]["sliver_max_boundary_move_mm"],
                               tcfg["tet"]["sliver_max_passes"])
    P, T = np.asarray(P, float), np.asarray(T, np.int64)
    if len(P) != len(ninfo) or fix.get("dropped_nodes", 0):
        raise SystemExit("v5: the sliver fix dropped nodes (%d of %d left); the start map needs every generated node"
                         % (len(P), len(ninfo)))
    info = np.array([(k, l, j) for (k, l, j, _) in ninfo], int)
    th_node = np.array([t for (_, _, _, t) in ninfo], float)
    st, layer = info[:, 0], info[:, 1]
    n_in = int((layer == 0).sum())
    partner = np.r_[np.arange(n_in) + n_in, np.arange(n_in)]

    # ---- START positions: node (station k, sheet, theta) -> generator (theta, sheet) at f_k of its length
    fk = prof["fk"]
    top_n = np.where(layer == 0, _periodic(tops["theta"], tops["inner"], th_node),
                     _periodic(tops["theta"], tops["outer"], th_node))
    sig_n = fk[st] * top_n
    Pst = np.zeros_like(P)
    Cst = np.zeros_like(P)
    for sh in (0, 2):
        m = layer == sh
        Pst[m] = VW.v5_start_points(tab, sig_n[m], th_node[m], sh)
    Cst[:] = np.stack([np.interp(sig_n, tab["sigma"], tab["P"][:, j]) for j in range(3)], 1)
    # ---- S7c(1) gate on the start, with the clamp (pull toward the generator axis, partner alike)
    gate_mm, clear = float(cfg["v5_gate_mm"]), float(cfg["v5_clamp_clear_mm"])
    gb = list(cfg["v5_gate_bodies"])

    def sd_all(X):
        return {b: sdf5[b](X) for b in gb}

    d0 = sd_all(Pst)
    gate_before = {b: dict(n_deeper_than_gate=int((d0[b] < -gate_mm).sum()), n_inside=int((d0[b] < 0).sum()),
                           min_mm=round(float(d0[b].min()), 3)) for b in gb}
    clamp = dict(enabled=bool(cfg["v5_clamp"]), passes=0, n_nodes=0, max_move_mm=0.0, clear_mm=clear)
    if cfg["v5_clamp"]:
        for _ in range(4):
            dd = sd_all(Pst)
            dmin = np.min(np.stack([dd[b] for b in gb]), 0)
            bad = np.nonzero(dmin < clear)[0]
            if not len(bad):
                break
            clamp["passes"] += 1
            lam = np.ones(len(P))
            for i in bad:
                lo, hi = 0.0, 1.0
                for _b in range(30):
                    mid = 0.5 * (lo + hi)
                    X = Cst[i] + mid * (Pst[i] - Cst[i])
                    if min(float(sdf5[b](X[None])[0]) for b in gb) >= clear:
                        lo = mid
                    else:
                        hi = mid
                lam[i] = min(lam[i], lo)
                lam[partner[i]] = min(lam[partner[i]], lo)
            mv = np.nonzero(lam < 1.0)[0]
            Pn = Cst[mv] + lam[mv, None] * (Pst[mv] - Cst[mv])
            clamp["max_move_mm"] = max(clamp["max_move_mm"], float(np.linalg.norm(Pn - Pst[mv], axis=1).max()))
            Pst[mv] = Pn
            clamp["n_nodes"] += int(len(mv))
        clamp["max_move_mm"] = round(clamp["max_move_mm"], 3)
    d1 = sd_all(Pst)
    gate_after = {b: dict(n_deeper_than_gate=int((d1[b] < -gate_mm).sum()), n_inside=int((d1[b] < 0).sum()),
                          min_mm=round(float(d1[b].min()), 3)) for b in gb}
    gate_ok = all(v["n_deeper_than_gate"] == 0 for v in gate_after.values())

    # ---- quality of both shapes
    q = MB.tet_quality(P, T)
    vol = VW.tet_volumes(P, T)
    dih = VW.dihedral_angles(P, T)
    dmin = dih.min(1)
    vol_st = VW.tet_volumes(Pst, T)
    dih_st = VW.dihedral_angles(Pst, T)
    start_q = dict(inverted_or_flat=int((vol_st <= 1e-9).sum()), interior_dihedral_min_deg=round(float(180.0 - dih_st.max()), 2),
                   n_tets_interior_dihedral_lt_5deg=int(((180.0 - dih_st.max(1)) < 5.0).sum()),
                   n_tets_interior_dihedral_lt_10deg=int(((180.0 - dih_st.max(1)) < 10.0).sum()),
                   volume_ratio_start_over_rest=VW.stats(vol_st / np.maximum(vol, 1e-12)),
                   wall_volume_cc=round(float(vol_st.sum()) / 1000.0, 4))

    # ---- boundary faces and node sets (exact: every node is a generated ring node with its own sheet / station)
    BF = VW.boundary_faces(T)
    S = np.unique(BF)
    lf = layer[BF]
    fin, fout = np.all(lf == 0, axis=1), np.all(lf == 2, axis=1)
    inner, outer = np.unique(BF[fin]), np.unique(BF[fout])
    top = np.nonzero(st == n_ax - 1)[0]
    Sk = prof["Sk"]
    inf = np.nonzero(Sk[st] <= Sk[0] + float(cfg["fixed_inferior_mm"]))[0]
    vault_st = np.nonzero(prof["w_vault"] >= 1.0 - 1e-9)[0]
    vault = np.nonzero(np.isin(st, vault_st))[0]
    sets = dict(surface_nodes=S, inner_surface=inner, outer_surface=outer, apex=top, fixed_inferior=inf,
                vault=vault, vault_top=top, vault_top_inner=top[layer[top] == 0], vault_top_outer=top[layer[top] == 2])
    defs = dict(
        surface_nodes="nodes on the boundary of the tet mesh (faces used by one tet)",
        inner_surface="nodes of the LUMEN sheet (contacts the device)",
        outer_surface="nodes of the OUTER sheet: contacts bladder / rectum / cervix",
        apex="= vault_top (the vagina-cervix junction ring); kept under this name for the scene's apex options",
        fixed_inferior="nodes of the rest stations within %g mm (rest centreline arclength) of the introitus station"
                       % cfg["fixed_inferior_mm"],
        vault="nodes of the rest stations inside the ring span (station centre at or above the ring's bottom face along "
              "the ring axis): lumen = ring outline + v5_vault_clear_mm",
        vault_top="both sheets of the TOP station: at rest on the ring's top face, at the START on the cervix surface "
                  "(signed distance v5_start_top_gap_mm); paired with the cervix in wall.v5.pairing",
        vault_top_inner="vault_top, lumen sheet", vault_top_outer="vault_top, outer sheet")

    # ---- the vault-top <-> cervix pairing (START shape vs the preBT cervix mesh)
    from scipy.spatial import cKDTree
    import vtk
    cm = json.load(open("%s/cervix/meta.json" % PT["meshes"]))
    s2n_c = np.asarray(cm["surface_obj_vertex_to_tet_node"], int)
    Vc, Fc = geom.read_obj("%s/cervix/surface.obj" % PT["meshes"])
    Vc, Fc = np.asarray(Vc, float), np.asarray(Fc, int)
    dn, jn = cKDTree(Vc).query(Pst[top])
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(MB.polydata(Vc, Fc))
    loc.BuildLocator()
    tri_n, bary, cp = [], [], []
    ref = getattr(vtk, "reference", None) or vtk.mutable
    for x in Pst[top]:
        cpt, cid, sid, d2 = [0.0, 0.0, 0.0], ref(0), ref(0), ref(0.0)
        loc.FindClosestPoint([float(v) for v in x], cpt, cid, sid, d2)
        f = Fc[int(cid)]
        M = np.c_[Vc[f[0]] - Vc[f[2]], Vc[f[1]] - Vc[f[2]]]
        l12 = np.linalg.lstsq(M, np.asarray(cpt) - Vc[f[2]], rcond=None)[0]
        tri_n.append(s2n_c[f].tolist())
        bary.append([float(l12[0]), float(l12[1]), float(1.0 - l12.sum())])
        cp.append(cpt)
    cp = np.asarray(cp, float)
    sd_top = sdf5["cervix"](Pst[top])
    pairing = dict(
        vault_top=top.tolist(), cervix_node=s2n_c[jn].tolist(), cervix_node_dist_mm=np.round(dn, 4).tolist(),
        cervix_tri_nodes=tri_n, cervix_tri_bary=np.round(bary, 6).tolist(),
        offset_start_mm=np.round(Pst[top] - cp, 4).tolist(), sdf_start_mm=np.round(sd_top, 4).tolist(),
        rest_app_mm=np.round((P[top] - F) @ R.T, 4).tolist(),
        note="per vault_top node (wall tets.vtk index): its nearest cervix SURFACE node (cervix tets.vtk index) and the "
             "closest point on the cervix surface as a triangle of cervix tets.vtk nodes + barycentric weights, "
             "measured at the START shape (start.vtk) against the preBT cervix (meshes/cervix); offset_start_mm = "
             "wall node - closest point (zero-force offset of a tie at the start).  rest_app_mm = the node's REST "
             "(seated) position in the %s applicator frame: world = flange + p_app @ R_rows of the current device pose."
             % cfg["v5_applicator_dir"])

    # ---- lengths, stretch and the vault-to-portio geometry
    th_p = prof["th"]
    Rin, C_r, U_r, V_r = W["r_in"], W["C"], W["U"], W["V"]
    gen_rest = np.array([np.linalg.norm(np.diff(C_r + Rin[:, j:j + 1] * (np.cos(t) * U_r + np.sin(t) * V_r), axis=0),
                                        axis=1).sum() for j, t in enumerate(th_p)])
    tops_in = _periodic(tops["theta"], tops["inner"], th_p)
    gen_start = np.array([np.linalg.norm(np.diff(VW.v5_start_points(tab, fk * tops_in[j], np.full(n_ax, t), 0), axis=0),
                                         axis=1).sum() for j, t in enumerate(th_p)])
    centre_rest = float(np.linalg.norm(np.diff(C_r, axis=0), axis=1).sum())
    stretch = gen_rest / gen_start
    zr = ring["axis"]
    Ptop_r = P[top]
    h_top = (Ptop_r - (F + np.array([0.0, 0.0, ring["z_top_app"] - float(cfg["v5_top_dz_mm"])]) @ R)) @ zr
    q_rad = Ptop_r - ring["centre_w"]
    rad_top = np.linalg.norm(q_rad - np.outer(q_rad @ zr, zr), axis=1)
    ti, to_ = top[layer[top] == 0], top[layer[top] == 2]
    oi = ti[np.argsort(th_node[ti])]
    oo = to_[np.argsort(th_node[to_])]
    per = dict(rest_inner=_closed_len(P[oi]), rest_outer=_closed_len(P[oo]), start_inner=_closed_len(Pst[oi]),
               start_outer=_closed_len(Pst[oo]))
    pc = np.asarray(MB.read_vtk_legacy("%s/cervix/tets.vtk" % PT["meshes"])[0], float)[s2n_c[jn]]
    trav = P[top] - pc
    report = dict(
        lengths_mm=dict(rest_centreline=round(centre_rest, 2), start_centre_column=round(float(tops["centre"]), 2),
                        centre_stretch=round(centre_rest / float(tops["centre"]), 4),
                        rest_inner_generators=VW.stats(gen_rest, 2), start_inner_generators=VW.stats(gen_start, 2),
                        generator_stretch=VW.stats(stretch, 4),
                        start_top_s_mm=VW.stats(np.interp(tops["inner"], tab["sigma"], tab["s"]), 2),
                        label_below_hrctv_s_mm=[round(float(tab["raw"]["s"][0]), 2), round(float(tab["s_clean_top"]), 2),
                                                round(float(tab["s_below_top"]), 2)],
                        label_below_hrctv_note="[introitus, last slab fully below the HR-CTV, highest label voxel outside "
                                               "the HR-CTV] along the vaginal axis",
                        start_section=tab["start_section"],
                        rest_top_s_mm=round(ring["s_top"], 2),
                        plan_reference="fix plan S7a: rest 77.5 mm (introitus -> seated ring top along the preBT "
                                       "axis) vs the label outside the HR-CTV 53.8 mm: implied stretch 1.44",
                        note="generator = one polar angle of the lumen sheet, introitus -> top station; rest length "
                             "follows the lumen round the ring (meridian), start length along the collapsed label"),
        volumes_cc=dict(rest_wall=round(float(vol.sum()) / 1000.0, 4), start_wall=start_q["wall_volume_cc"],
                        start_over_rest=round(float(vol_st.sum() / vol.sum()), 4),
                        label_below_hrctv=shared["label_cc"]),
        vault_to_portio=dict(
            start_top_sdf_cervix_mm=VW.stats(sd_top, 3),
            start_top_nearest_cervix_node_mm=VW.stats(dn, 3),
            rest_top_height_above_ring_top_face_mm=VW.stats(h_top, 3),
            rest_top_radius_about_ring_axis_mm=VW.stats(rad_top, 3),
            top_ring_perimeter_mm={k: round(v, 2) for k, v in per.items()},
            top_ring_circumferential_stretch=dict(inner=round(per["rest_inner"] / per["start_inner"], 4),
                                                  outer=round(per["rest_outer"] / per["start_outer"], 4)),
            paired_cervix_node_to_rest_top_mm=VW.stats(np.linalg.norm(trav, axis=1), 3),
            paired_cervix_node_to_rest_top_mean_vec_mm=np.round(trav.mean(0), 3).tolist(),
            paired_cervix_node_to_rest_top_along_ring_axis_mm=VW.stats(trav @ zr, 3),
            note="start: the top ring lies on the preBT cervix surface; rest: on the ring's top face at device_final. "
                 "paired_cervix_node_to_rest_top = how far each paired preBT cervix surface node is from its wall node's "
                 "seated position: the lift + spread the drive / ring must give the portio for the junction to stay "
                 "closed at hand-over"),
        curvature=dict(max_r_in_over_R_planes=round(float(prof["curv_ratio"].max()), 4),
                       max_r_in_over_R_centreline=round(float(prof["centre_curv_ratio"].max()), 4),
                       limit=float(cfg["v5_curv_ratio_max"]),
                       ok=bool(prof["curv_ratio"].max() < float(cfg["v5_curv_ratio_max"])
                               and prof["fold"]["n_pairs_crossing"] == 0),
                       stations_over_limit_planes=[int(k) for k in np.nonzero(prof["curv_ratio"] >= float(cfg["v5_curv_ratio_max"]))[0]],
                       stations_over_limit_centreline=[int(k) for k in np.nonzero(prof["centre_curv_ratio"]
                                                                                  >= float(cfg["v5_curv_ratio_max"]))[0]],
                       fold=prof["fold"], sweep=prof["sweep"],
                       note="the G15 fold is two station planes meeting inside the wall.  R_planes = distance from a "
                            "station centre to the line where its plane meets the next (the fold radius of the sweep; "
                            "the verdict uses it and the direct crossing test `fold`).  R_centreline = circumradius of "
                            "consecutive station CENTRES: where the planes are parallel (the ring span and "
                            "v5_parallel_below_mm below it) a small value is an in-plane shift of the lumen centre "
                            "(shear), which cannot fold the mesh; reported for the plan's r_in < 0.8 R wording"),
        gate_S7c1=dict(ok=gate_ok, gate_mm=gate_mm, bodies=gb, before_clamp=gate_before, after_clamp=gate_after,
                       clamp=clamp))
    t_all = time.time() - t0

    # ---- write
    lab_cc = shared["label_cc"]
    wall_cc = float(vol.sum()) / 1000.0
    d = "%s/vagina_wall_%s" % (PT["meshes"], args.variant)
    os.makedirs(d, exist_ok=True)
    remap = -np.ones(len(P), np.int64)
    remap[S] = np.arange(len(S))
    Vb, Fb = P[S], remap[BF]
    bvol = geom.mesh_volume(Vb, Fb)
    if bvol < 0:
        Fb = Fb[:, ::-1]
        bvol = -bvol
    b_open, b_nonman = VW.manifold_check(Vb, Fb)
    VW.write_obj(d + "/surface.obj", Vb, Fb,
                 "vagina WALL v5 surface (REST = seated around %s at device_final): boundary of tets.vtk; vertex k == "
                 "tets.vtk point meta.surface_obj_vertex_to_tet_node[k]\n%s\nunits mm; derived from labels (local only)"
                 % (cfg["v5_applicator_dir"], VW.FRAME))
    VW.write_vtk_legacy(d + "/tets.vtk", P, T, "vagina wall v5 tetrahedra, REST (seated), %s" % VW.FRAME)
    VW.write_vtk_legacy(d + "/start.vtk", Pst, T, "vagina wall v5 tetrahedra, START (collapsed label), %s" % VW.FRAME)
    VW.write_obj(d + "/start_surface.obj", Pst[S], Fb,
                 "vagina WALL v5 START surface (collapsed, from the label below the HR-CTV): same vertices / faces as "
                 "surface.obj\n%s\nunits mm; derived from labels (local only)" % VW.FRAME)
    nrm = np.cross(P[BF[fin, 1]] - P[BF[fin, 0]], P[BF[fin, 2]] - P[BF[fin, 0]])
    radial = P[BF[fin]].mean(1) - C_r[st[BF[fin, 0]]]
    radial -= np.einsum("ij,ij->i", radial, W["Tg"][st[BF[fin, 0]]])[:, None] * W["Tg"][st[BF[fin, 0]]]
    inward_frac = float((np.einsum("ij,ij->i", nrm, radial) < 0).mean())
    if inward_frac < 0.5:
        inward_frac = 1.0 - inward_frac
    gidx = np.c_[st, layer, info[:, 2]]
    thick = (W["r_out"] - W["r_in"]).ravel()
    extra = dict(inner_triangles=BF[fin].tolist(), outer_triangles=BF[fout].tolist(), n_inner_triangles=int(fin.sum()),
                 n_outer_triangles=int(fout.sum()), n_cap_triangles=int((~fin & ~fout).sum()),
                 apex_rings=1, inferior_rings=int(len(np.unique(st[inf]))))
    for nm in ("cervix", "bladder", "rectum", "sigmoid"):
        dd = sdf5[nm](P[outer])
        extra["outer_signed_dist_to_%s_mm" % nm] = dict(min=round(float(dd.min()), 3), n_inside=int((dd < 0).sum()),
                                                        note="REST (seated) shape vs the preBT organs: overlaps by design "
                                                             "(fix plan S7c(1) gates the START only)")
    gate_tets = bool(180.0 - dih.max() > float(tcfg["quality_gate_deg"]) and (vol <= 1e-9).sum() == 0)
    r_in_st = np.array([np.linalg.norm(P[(st == k) & (layer == 0)] - P[(st == k) & (layer == 0)].mean(0), axis=1).mean()
                        for k in range(n_ax)])
    r_out_st = np.array([np.linalg.norm(P[(st == k) & (layer == 2)] - P[(st == k) & (layer == 0)].mean(0), axis=1).mean()
                         for k in range(n_ax)])
    n_th_in = np.array([int(((st == k) & (layer == 0)).sum()) for k in range(n_ax)])
    n_th_out = np.array([int(((st == k) & (layer == 2)).sum()) for k in range(n_ax)])
    tagv = "CALIBRATED (in-sample: mid / lower sections from this patient's BT vagina label)" if cfg["v5_section"] == "cal" \
        else "PREDICTED / SCENARIO (device hull + margins; no BT information; packing volume 0)"
    sta = dict(
        rest=dict(sigma_mm=np.round(Sk, 4).tolist(), f=np.round(fk, 6).tolist(), centre=np.round(C_r, 4).tolist(),
                  normal=np.round(W["Tg"], 6).tolist(), w_vault=np.round(prof["w_vault"], 4).tolist(),
                  area_mm2=np.round(prof["A"], 4).tolist(), n_device_hits=prof["n_hit"].tolist(),
                  r_device_mm=np.round(prof["r_dev"], 3).tolist(),
                  r_bt_inner_mm=(np.round(np.nan_to_num(prof["r_bt_in"], nan=-1.0), 3).tolist()
                                 if cfg["v5_section"] == "cal" else None),
                  radius_planes_mm=np.round(np.minimum(prof["R_planes"], 1e6), 3).tolist(),
                  radius_centreline_mm=np.round(np.minimum(prof["R_centre"], 1e6), 3).tolist(),
                  r_in_max_over_R_planes=np.round(prof["curv_ratio"], 4).tolist(),
                  r_in_max_over_R_centreline=np.round(prof["centre_curv_ratio"], 4).tolist(),
                  below_ring=prof["below_ring"].astype(int).tolist(),
                  kind=prof["kind"].tolist(), sweep=prof["sweep"],
                  vault_stations=vault_st.tolist(),
                  note="stations at uniform inner-sheet meridian length (axial_step_mm) along the rest centreline; "
                       "station k's plane: centre, normal (= 'tangent'), basis frame_u (patient right projected) / "
                       "frame_v = normal x frame_u; lumen r_in_theta_mm about the centre"),
        start=dict(sigma_top_inner=np.round(tops["inner"], 4).tolist(), sigma_top_outer=np.round(tops["outer"], 4).tolist(),
                   theta=np.round(tops["theta"], 6).tolist(), sigma_top_centre=round(float(tops["centre"]), 4),
                   label_s_mm=np.round(tab["s"], 3).tolist(), label_sigma_mm=np.round(tab["sigma"], 4).tolist(),
                   label_centre=np.round(tab["P"], 4).tolist(), ellipse_a_outer=np.round(tab["a_o"], 4).tolist(),
                   ellipse_b_outer=np.round(tab["b_o"], 4).tolist(), ellipse_a_inner=np.round(tab["a_i"], 4).tolist(),
                   ellipse_b_inner=np.round(tab["b_i"], 4).tolist(), ellipse_psi_rad=np.round(tab["psi"], 6).tolist(),
                   area_mm2=np.round(tab["A"], 4).tolist(), n_clean_slabs=int(tab["n_clean"]),
                   slab_basis_u=np.round(tab["U0"], 6).tolist(), slab_basis_v=np.round(tab["V0"], 6).tolist(),
                   raw=dict((k, np.round(v, 4).tolist()) for k, v in tab["raw"].items()),
                   note="start node (station k, sheet, theta) = generator (theta, sheet) at start arclength f_k x its "
                        "top; generator point = label_centre(sigma) + a cos(theta - psi) e_psi + b sin(theta - psi) "
                        "e_psi_perp in the slab plane (basis slab_basis_u / _v), sheet 0 = (a_inner, b_inner), 2 = "
                        "(a_outer, b_outer); the top = the first contact with the cervix surface (sigma_top_*)"))
    v5 = dict(
        version="S7a wall v5 (lumen_profile packed)", section=cfg["v5_section"], tag=tagv,
        applicator_dir=cfg["v5_applicator_dir"], device_final=dict(flange=F.tolist(), R_rows=R.tolist()),
        ring=dict((k, (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v)) for k, v in ring.items()),
        start_file="start.vtk", start_surface_file="start_surface.obj",
        start_note="start.vtk = the same tets at the collapsed START positions; the drive (S7b) moves every node from "
                   "start to rest (tets.vtk), which is stress-free: at hand-over the wall is at rest",
        node_station=st.tolist(), node_sheet=layer.tolist(), node_theta_rad=np.round(th_node, 6).tolist(),
        node_f=np.round(fk[st], 6).tolist(), node_start_sigma_mm=np.round(sig_n, 4).tolist(),
        stations=sta, pairing=pairing, report=report, start_quality=start_q,
        bt=shared.get("bt_info") if cfg["v5_section"] == "cal" else None,
        profile_s=round(t_prof, 2))
    meta = dict(
        body=VW.BODY, priority=3, source_label="vagina", frame=VW.FRAME, units="mm; volumes cc; angles deg",
        role="deformable HOLLOW WALL v5 (fix plan S7a): stress-free = seated around the ring device; start = the "
             "collapsed label below the HR-CTV (start.vtk); the top ring is the vagina-cervix junction",
        voxel_mm=shared["voxel_mm"],
        files=dict(surface="surface.obj", tets="tets.vtk", meta="meta.json", start="start.vtk",
                   start_surface="start_surface.obj"),
        volumes_cc=dict(label_raw=shared["label_raw_cc"], after_priority=shared["after_priority_cc"],
                        after_largest_component=lab_cc, surface_mesh=round(bvol / 1000.0, 4), tet_mesh=round(wall_cc, 4),
                        start_tet_mesh=start_q["wall_volume_cc"],
                        surface_vs_label_pct=round(100 * (bvol / 1000.0 - lab_cc) / lab_cc, 2),
                        tet_vs_label_pct=round(100 * (wall_cc - lab_cc) / lab_cc, 2),
                        note="rest wall area per station = the label's (at the mapped start station) with the thickness "
                             "floor; the rest is ~1.5x longer than the start, so its volume is larger (start_over_rest)"),
        surface=dict(tris=int(len(Fb)), verts=int(len(Vb)), open_edges=b_open, nonmanifold_edges=b_nonman,
                     nonmanifold_vertices=0, target_edge_mm=tcfg["edge_mm"], edge_mm=VW.stats(VW.tri_edge_lengths(Vb, Fb)),
                     area_mm2=round(float(VW.tri_areas(Vb, Fb).sum()), 1),
                     gate_ok=bool(b_open == 0 and b_nonman == 0 and bvol > 0),
                     inner_normals_point_into_lumen_frac=round(inward_frac, 4),
                     input_surface=dict(verts=int(len(Vs)), tris=int(len(Fs)), open_edges=n_open, nonmanifold_edges=n_nonman,
                                        volume_cc=round(surf_vol / 1000.0, 4),
                                        patch_orientation_agreement=sinfo["patch_orientation_agreement"]),
                     method="v5 analytic rest profile (vagina_wall.v5_rest_profile) sampled per ring at a uniform edge, "
                            "rings zipped by angle, outer sheet along the lumen normal; boundary of tets.vtk"),
        tets=dict(nodes=int(len(P)), tets=int(len(T)), surface_nodes=int(len(S)), target_edge_mm=tcfg["edge_mm"],
                  min_dihedral_deg=round(float(dmin.min()), 2), max_dihedral_deg=round(float(dih.max()), 2),
                  dihedral_convention="min_dihedral_deg / max_dihedral_deg use the mesh_bodies.py formula (the "
                                      "SUPPLEMENT of the interior dihedral); the TRUE interior angles are "
                                      "interior_dihedral_{min,max}_deg and the gate uses them.",
                  interior_dihedral_min_deg=round(float(180.0 - dih.max()), 2),
                  interior_dihedral_max_deg=round(float(180.0 - dmin.min()), 2),
                  n_tets_interior_dihedral_lt_10deg=int(((180.0 - dih.max(1)) < 10.0).sum()),
                  n_tets_interior_dihedral_lt_20deg=int(((180.0 - dih.max(1)) < 20.0).sum()),
                  sliver_frac_min_dihedral_lt_10deg=round(float((dmin < 10.0).mean()), 5),
                  frac_min_dihedral_lt_15deg=round(float((dmin < 15.0).mean()), 5),
                  n_tets_min_dihedral_lt_10deg=int((dmin < 10.0).sum()),
                  edge_mm=VW.stats(VW.tet_edge_lengths(P, T)), tet_volume_mm3=VW.stats(vol),
                  inverted_or_flat=int((vol <= 1e-9).sum()), flipped=int(nflip), gate_ok=gate_tets,
                  mesh_bodies_quality=q,
                  tetgen=dict(route="prism split (Dompierre) + mesh_bodies.fix_slivers", input_surface_preserved=True,
                              flipped=int(nflip), n_prisms=int(len(sinfo["prisms"])), n_inverted_before_orient=n_inv,
                              offset=sinfo["offset"], normal_offset=sinfo["normal_offset"],
                              tetgen_raw=dict(nodes=q_raw["nodes"], tets=q_raw["tets"],
                                              min_dihedral_deg=q_raw["min_dihedral_deg"],
                                              n_tets_min_dihedral_lt_10deg=q_raw["n_tets_min_dihedral_lt_10deg"],
                                              note="raw = the prism split before the sliver fix"),
                              sliver_fix=fix),
                  start=start_q,
                  route="v5 profile + prism split (vagina_wall_tet.py, python %s)" % sys.version.split()[0]),
        axis=dict(centroid=np.round(c, 3).tolist(), axis=np.round(a, 5).tolist(),
                  proj_min=round(float(tab["raw"]["s"][0]), 3), proj_max=round(float(tab["s_clean_top"]), 3),
                  source="pose.json device_final.shaft_axis (= the vagina body's principal axis, rule v2); s range = "
                         "the label slabs below the HR-CTV"),
        node_set_defs=defs, node_set_extra=extra, node_set_sizes={k: int(len(v)) for k, v in sets.items()},
        surface_obj_vertex_to_tet_node=S.tolist(), node_sets={k: np.asarray(v, int).tolist() for k, v in sets.items()},
        wall=dict(
            reference_state="MODELLING CHOICE: the stress-free state is the SEATED shape around the applicator (fix plan "
                            "S7a); the start is the collapsed label.  The real wall reaches the seated shape by "
                            "unfolding its rugae and stretching; the drive (S7b) carries it there kinematically, so axial "
                            "wall tension is not modelled (implied stretch in v5.report.lengths_mm).",
            mesh="prism (Dompierre) mesh of the v5 profile; grid_index = (station, sheet 0 lumen / 2 outer, rank in angle)",
            cfg={k: v for k, v in cfg.items() if not k.startswith("_")}, tet_cfg=tcfg,
            n_axial=n_ax, n_theta=int(max(n_th_in.max(), n_th_out.max())), n_radial=2,
            n_theta_per_station=dict(inner=n_th_in.tolist(), outer=n_th_out.tolist()),
            axial_step_mm=round(float(np.mean(np.diff(Sk))), 4), lumen_r0_mm=cfg["lumen_r0_mm"],
            E_vagina_kPa=cfg["E_vagina_kPa"], nu_vagina=cfg["nu_vagina"],
            s=np.round(Sk, 4).tolist(), s_note="v5: s = rest centreline arclength of each station from the introitus",
            area_mm2=np.round(prof["A"], 4).tolist(), area_raw_mm2=np.round(prof["A"], 4).tolist(),
            n_vox_per_station=[0] * n_ax,
            r_in_mm=np.round(r_in_st, 4).tolist(), r_out_mm=np.round(r_out_st, 4).tolist(),
            thickness_mm=np.round(r_out_st - r_in_st, 4).tolist(), thickness_stats=VW.stats(thick),
            centreline=np.round(C_r, 4).tolist(), profile_theta_deg=np.round(np.degrees(th_p), 3).tolist(),
            r_in_theta_mm=np.round(W["r_in"], 4).tolist(), r_out_theta_mm=np.round(W["r_out"], 4).tolist(),
            r_in_note="r_in_mm / r_out_mm = per-station mean radii of the mesh rings; r_in_theta_mm / r_out_theta_mm = the "
                      "analytic rest profile at profile_theta_deg (theta from patient right toward frame_v)",
            fornix=dict(enabled=False, mode="ring", note="v5: no ray-cast to the rest cervix; the vault is the ring outline"),
            apex_extension=dict(stations=0),
            lumen_profile=dict(enabled=True, profile="packed", lr_angle_in_frame_deg=[0.0] * n_ax,
                               note="frame_u IS patient right projected into each station plane, so the LR angle is 0"),
            frame_u=np.round(U_r, 6).tolist(), frame_v=np.round(V_r, 6).tolist(), tangent=np.round(W["Tg"], 6).tolist(),
            grid_index=gidx.tolist(),
            grid_index_note="per node (station, sheet: 0 = lumen, 2 = outer, rank in angle within the station ring); "
                            "every node is a generated ring node (no interior nodes)",
            grid_index_check=dict(generated_nodes=int(len(ninfo)), same_station=int(len(ninfo)), same_layer=int(len(ninfo))),
            centre_deviation=dict(note="v5: the rest centreline follows the device (see v5.stations.rest)"),
            min_thickness=dict(min_thickness_mm=float(cfg["v5_min_thickness_mm"])),
            ref_inner_perimeter_mm=round(2.0 * np.pi * cfg["lumen_r0_mm"], 4),
            collision_note="the closed boundary is oriented OUTWARD from the tissue (positive enclosed volume), which on "
                           "the lumen sheet points INTO the lumen -- what SOFA's TriangleCollisionModel expects.",
            v5=v5))
    json.dump(meta, open(d + "/meta.json", "w"), indent=1, default=VW.json_default)
    root = VW.scene_mesh_root(PT["meshes"], "vagina_wall_%s" % args.variant)
    row = dict(variant=args.variant, dir=os.path.basename(d), section=cfg["v5_section"], nodes=int(len(P)), tets=int(len(T)),
               surface_nodes=int(len(S)), n_axial=n_ax, n_theta_per_station=meta["wall"]["n_theta_per_station"],
               vol_cc=round(wall_cc, 3), start_vol_cc=start_q["wall_volume_cc"], label_cc=lab_cc,
               thickness_mm=[round(float(thick.min()), 2), round(float(np.median(thick)), 2), round(float(thick.max()), 2)],
               interior_dihedral_deg=[meta["tets"]["interior_dihedral_min_deg"], meta["tets"]["interior_dihedral_max_deg"]],
               n_tets_lt_10deg=meta["tets"]["n_tets_interior_dihedral_lt_10deg"], inverted=meta["tets"]["inverted_or_flat"],
               start_inverted=start_q["inverted_or_flat"], start_interior_dihedral_min_deg=start_q["interior_dihedral_min_deg"],
               gate_S7c1_ok=gate_ok, gate_after=gate_after, clamp=clamp, open_edges=b_open, nonmanifold=b_nonman,
               inner_normals_inward=round(inward_frac, 3), sets={k: int(len(v)) for k, v in sets.items()}, gate_ok=gate_tets,
               lengths=report["lengths_mm"], vault_to_portio=report["vault_to_portio"], curvature=report["curvature"],
               scene_root=root, total_s=round(t_all, 2),
               tetgen_min_dihedral_deg=q_raw["min_dihedral_deg"], preserved=True)
    print("[vagina_wall_tet] %-10s v5 %s  nodes %5d tets %6d  rest vol %.3f cc  start vol %.3f cc  interior dihedral "
          "%.2f deg (start %.2f, inverted %d)  gate S7c(1) %s  stretch centre %.3f generators %.3f-%.3f  fold "
          "r_in/R_planes max %.3f  %.1f s"
          % (args.variant, cfg["v5_section"], len(P), len(T), wall_cc, start_q["wall_volume_cc"], 180.0 - dih.max(),
             start_q["interior_dihedral_min_deg"], start_q["inverted_or_flat"], "PASS" if gate_ok else "FAIL",
             report["lengths_mm"]["centre_stretch"], stretch.min(), stretch.max(),
             prof["curv_ratio"].max(),
             time.time() - t0), flush=True)
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
    if _is_packed(args):                         # wall v5: the FULL vagina label (inside-HR-CTV flag) and the BT labels
        ijk = np.argwhere(masks["vagina"])
        shared["Xf"] = MB.v2w(aff, ijk)
        shared["inH"] = masks["HR-CTV"][tuple(ijk.T)]
        c5 = VW.v5_cfg({k: json.loads(v) for k, v in (kv.split("=", 1) for kv in (args.set or [])) if k.startswith("v5_")})
        shared["bt_pts"], shared["bt_info"] = _v5_bt_points(PT, c5)
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
