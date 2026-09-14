"""SOFA v22.12 scene (CONTAINER, SofaPython3, py3.8): preBT uterus/cervix body + applicator coupling.

UNITS mm / kg / s  ->  force mN, stress kPa, stiffness mN/mm, Winkler modulus mN/mm^3.
Graph (headless; gravity 0; DefaultAnimationLoop; one quasi-static solve per animate step):
  root
   /targets           solver-less MOs written by the controller each step
      tie_tgt         node-level targets of the distributed node ties (DNT)
      con_tgt         (M2) contact targets for candidate nodes
      found_tgt       (found_anchor='device_rigid') moving anchors of the supports
   /body              StaticSolver + CGLinearSolver (default; the Python tie field is matrix-free)
      SparseGridTopology(pre_body.obj, cellWidth[, nbVirtualFinerLevels]) + MechanicalObject
      NonUniformHexahedronFEMForceFieldAndMass(large) (default: fill-weighted stiffness condensed from finer virtual
          levels) or HexahedronFEMForceField(large) (fe_impl='hexa', stage 1-3: every boundary cell fully stiff)
      RestShapeSpringsForceField 'found'  Winkler + paracervical supports (rest shape or moving device-rigid anchors)
      NodeTieFF 'tie' (ties.py, default)  anisotropic distributed node ties + (M2) surface contact, on FEM nodes;
          alternative RestShapeSpringsForceField 'tie' -> /targets/tie_tgt (tie_impl='rssff', direct solvers)
      RestShapeSpringsForceField 'con'    (M2, contact_impl='rssff_nodes') sphere-SDF penalty on FEM nodes
      read-only mapped children (BarycentricMapping, NO force fields): surf_body, surf_uterus, surf_hrctv, canal
Tie points: with tie_group='cell' the non-tip canal points sharing a rest cell are tied as one point (their mean).
Fact F1 (measured earlier): force fields on MAPPED nodes diverge with SparseLDLSolver in v22.12, so every
force-producing component acts on the FEM MechanicalObject; mapped children are for read-out/export only.

runSofa: set env APPSIM_CFG=/out/runs/<tag>/cfg.json and load this file (module-level createScene(rootNode)).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402

import Sofa  # noqa: E402
import Sofa.Core  # noqa: E402
import Sofa.Simulation  # noqa: E402

HEX_FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


# ----------------------------------------------------------------------------- inputs
def load_inputs(inp=None):
    inp = inp or config.paths()["inputs"]
    d = dict(inp=inp)
    d["named"] = json.load(open(os.path.join(inp, "named.json")))
    d["poses"] = json.load(open(os.path.join(inp, "poses.json")))
    pa = os.path.join(inp, "poses_align.json")          # stage 2 (align.py): BT device mapped by the bone frame etc.
    if os.path.exists(pa):
        for k, v in json.load(open(pa)).items():
            if isinstance(v, dict) and "F" in v and k not in d["poses"]:
                d["poses"][k] = v
    d["app"] = json.load(open(os.path.join(inp, "applicator.json")))
    z = np.load(os.path.join(inp, "canal.npz"))
    d["canal"] = {k: z[k] for k in z.files}
    r = np.load(os.path.join(inp, "pre_labels_roi.npz"))
    d["roi"] = dict(region=r["region"], affine=r["affine"])
    return d


def select_pose(cfg, inp):
    """Final pose (F, a, x) in preBT world mm for cfg['pose'] (P1 | P2 | P2u | SWEEP with d_F_mm)."""
    P = inp["poses"]; a0 = np.array(P["a0"]); L_end = np.array(P["L_end"])
    name = cfg["pose"]
    if name in ("P1", "SWEEP"):
        d = P["d_F_pred_mm"] if cfg.get("d_F_mm") is None else float(cfg["d_F_mm"])
        pose = dict(F=L_end - d * a0, a=a0, x=geom.ortho([1, 0, 0], a0), d_F_mm=d)
    elif name in P and isinstance(P[name], dict) and "F" in P[name]:     # P2, P2u, PB (bone frame), PBg, ...
        q = P[name]
        pose = dict(F=np.array(q["F"]), a=np.array(q["a"]), x=np.array(q["x"]),
                    d_F_mm=float((L_end - np.array(q["F"])) @ a0))
    else:
        raise ValueError("unknown pose %s" % name)
    pose["name"] = name
    return pose


def fillet_canal(pts, s0, i_c, half_mm):
    """Cosine-blended Gaussian smoothing (sigma = half/2) of the points within +/- half_mm arclength of point i_c
    (the junction of the synthetic straight O_pre->L_end segment with the centre line).  Arclength labels s0 kept."""
    if half_mm <= 0:
        return pts
    out = pts.copy(); sig = 0.5 * half_mm
    for i in range(len(pts)):
        ds = abs(s0[i] - s0[i_c])
        if ds >= half_mm:
            continue
        w = np.exp(-0.5 * ((s0 - s0[i]) / sig) ** 2); w[np.abs(s0 - s0[i]) > 2 * sig] = 0.0
        sm = (w[:, None] * pts).sum(0) / w.sum()
        b = 0.5 * (1 + np.cos(np.pi * ds / half_mm))                    # 1 at the junction, 0 at +/- half
        out[i] = (1 - b) * pts[i] + b * sm
    return out


def extended_canal(cfg, inp):
    c = inp["canal"]; pts = np.array(c["pts"], float); s0 = np.array(c["s0"], float)
    dlt = min(float(cfg.get("delta_fund_mm") or 0.0), float(c["delta_fund_max_mm"]))
    if dlt > 0:
        n = int(np.ceil(dlt))
        ext = pts[-1] + np.outer(np.linspace(dlt / n, dlt, n), np.array(c["fund_dir"]))
        pts = np.vstack([pts, ext]); s0 = np.r_[s0, s0[-1] + np.linspace(dlt / n, dlt, n)]
    pts = fillet_canal(pts, s0, int(c["i_L_end"]), float(cfg.get("canal_fillet_mm") or 0.0))
    return pts, s0, dlt


# ----------------------------------------------------------------------------- grid probe (pass 1)
def _sparse_grid(node, body_obj, cfg, name="topo", n=None):
    """With nbVirtualFinerLevels, SparseGridTopology in v22.12 IGNORES cellWidth (MEASURED: it collapsed to a single
    2x2x2 cell), so the non-uniform grid is given the node counts n of the pass-1 cellWidth grid instead."""
    if cfg.get("fe_impl", "hexa") == "nonuniform" and n is not None:
        return node.addObject("SparseGridTopology", name=name, n=[int(v) for v in n], fileTopology=body_obj,
                              nbVirtualFinerLevels=int(cfg["nb_virtual_finer_levels"]))
    return node.addObject("SparseGridTopology", name=name, cellWidth=float(cfg["cell_mm"]), fileTopology=body_obj)


def probe_grid(body_obj, cfg):
    """Pass 1: rest node positions and hexahedra of the SparseGridTopology (deterministic), plus the voxel-hull volume
    of the coarse grid and of the finest virtual level (the volume the non-uniform stiffness integrates), in cc."""
    r = Sofa.Core.Node("probe")
    r.addObject("RequiredPlugin", pluginName="Sofa.Component")
    b = r.addChild("b")
    _sparse_grid(b, body_obj, cfg)                  # always the plain cellWidth grid (defines the nodes)
    b.addObject("MechanicalObject", name="dofs", template="Vec3d")
    L = int(cfg.get("nb_virtual_finer_levels") or 0) if cfg.get("fe_impl", "hexa") == "nonuniform" else 0
    Sofa.Simulation.init(r)
    X0 = np.array(b.dofs.position.value, dtype=float, copy=True)
    hexa = np.array(b.topo.hexahedra.value, dtype=int, copy=True)
    n = [int(v) for v in b.topo.n.value]
    Sofa.Simulation.unload(r)
    cell = np.abs(X0[hexa[:, 6]] - X0[hexa[:, 0]]).mean(0)
    vols = dict(grid_hull_cc=round(float(len(hexa) * np.prod(cell)) / 1000.0, 3), fine_level=L)
    if L > 0:
        r2 = Sofa.Core.Node("probe2")
        r2.addObject("RequiredPlugin", pluginName="Sofa.Component")
        f = r2.addChild("f")                     # the finest virtual level, approximated as its own sparse grid
        f.addObject("SparseGridTopology", name="topo", cellWidth=float(cfg["cell_mm"]) / 2 ** L, fileTopology=body_obj)
        f.addObject("MechanicalObject", name="dofs", template="Vec3d")
        Sofa.Simulation.init(r2)
        hf = np.array(f.topo.hexahedra.value, dtype=int, copy=True); Xf = np.array(f.dofs.position.value, float)
        cf = np.abs(Xf[hf[:, 6]] - Xf[hf[:, 0]]).mean(0)
        vols.update(fine_hull_cc=round(float(len(hf) * np.prod(cf)) / 1000.0, 3), fine_cell_mm=np.round(cf, 3).tolist(),
                    n_fine_hexa=int(len(hf)))
        Sofa.Simulation.unload(r2)
    return X0, hexa, n, vols


def group_ties(canal, cells, cnodes, W, n_tip):
    """Tie points: the non-tip canal points sharing a rest cell become ONE tie point (mean position, averaged trilinear
    weights over the identical 8 nodes); the last n_tip canal points stay individual (tip push).  Returns the tie
    points, their cnodes/W, the group index of every canal point, and the member count per tie point."""
    n = len(canal); grp = -np.ones(n, int); rows = []
    first = {}
    for i in range(n - n_tip):
        k = int(cells[i])
        if k not in first:
            first[k] = len(rows); rows.append([i])
        else:
            rows[first[k]].append(i)
    rows = sorted(rows, key=lambda r: float(np.mean(r)))
    rows += [[i] for i in range(n - n_tip, n)]
    for g, r in enumerate(rows):
        grp[r] = g
    tp = np.array([canal[r].mean(0) for r in rows]); tn = np.array([cnodes[r[0]] for r in rows])
    tw = np.array([W[r].mean(0) for r in rows]); cnt = np.array([len(r) for r in rows])
    return tp, tn, tw, grp, cnt


def node_sets(X0, hexa, cfg, inp):
    """Boundary nodes, exposed area, region label, height/radius wrt (L_end, a0), support stiffness."""
    h_mm = float(cfg["cell_mm"])
    nn = len(X0)
    cnt = np.bincount(hexa.ravel(), minlength=nn)
    faces = np.sort(hexa[:, HEX_FACES].reshape(-1, 4), axis=1)
    _, inv, fc = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    exposed = faces[fc[inv.ravel()] == 1]
    cell = np.abs(X0[hexa[:, 6]] - X0[hexa[:, 0]]).mean(0)            # real cell size (mm), may differ from cfg
    face_area = float(np.mean([cell[0] * cell[1], cell[1] * cell[2], cell[0] * cell[2]]))
    nexp = np.bincount(exposed.ravel(), minlength=nn)
    A_j = nexp * face_area / 4.0                                        # mm^2 per node
    boundary = np.nonzero(cnt < 8)[0]
    # region label: nearest non-zero ROI label within 3 mm
    from scipy import ndimage as ndi
    reg = inp["roi"]["region"]; Ar = inp["roi"]["affine"]; sp = np.abs(np.diag(Ar)[:3])
    dist, idx = ndi.distance_transform_edt(reg == 0, sampling=sp, return_indices=True)
    ijk = np.rint((X0 - Ar[:3, 3]) @ np.linalg.inv(Ar[:3, :3]).T).astype(int)
    ok = np.all((ijk >= 0) & (ijk < reg.shape), 1)
    region = np.zeros(nn, int)
    ii = ijk[ok]
    nearest = reg[idx[0][tuple(ii.T)], idx[1][tuple(ii.T)], idx[2][tuple(ii.T)]]
    dd = dist[tuple(ii.T)]
    region[ok] = np.where(dd <= 3.0, nearest, 0)
    L_end = np.array(inp["poses"]["L_end"]); a0 = np.array(inp["poses"]["a0"])
    r = X0 - L_end; hgt = r @ a0; rad = np.linalg.norm(r - np.outer(hgt, a0), axis=1)
    k_f = cfg["kappa_f_mN_per_mm3"] * A_j
    lig = (np.isin(region, [1, 2]) & (hgt >= cfg["lig_h_min_mm"]) & (hgt <= cfg["lig_h_max_mm"])
           & (rad > cfg["lig_radial_min_mm"]) & (A_j > 0))
    k_sup = k_f + np.where(lig, cfg["kappa_lig_mN_per_mm3"] * A_j, 0.0)
    # M2b: levator / perineal support of the lower vagina (only when the vagina is part of the body)
    # (Only with body_mesh='pre_bodyvag': with pre_body the portio nodes nearest the vagina label also carry region 4,
    # and an earlier version wrongly anchored the portio tip with this floor support -- runs *_vfloorbug.)
    vfloor = np.zeros(nn, bool)
    if cfg.get("body_mesh", "pre_body") == "pre_bodyvag" and (region == 4).any():
        h_bot = float(hgt[region == 4].min())
        vfloor = (region == 4) & (hgt < h_bot + float(cfg["vag_floor_h_mm"])) & (A_j > 0)
        k_sup = k_sup + np.where(vfloor, float(cfg["kappa_vag_floor_mN_per_mm3"]) * A_j, 0.0)
    sup_nodes = np.nonzero(k_sup > 0)[0]
    return dict(boundary=boundary, A_j=A_j, region=region, h=hgt, rad=rad, lig=np.nonzero(lig)[0], vfloor=np.nonzero(vfloor)[0],
                sup_nodes=sup_nodes, k_sup=k_sup[sup_nodes], cell_real_mm=cell, n_per_node=cnt,
                total_found_mN_per_mm=float(k_f.sum()), total_lig_mN_per_mm=float((k_sup - k_f).sum()),
                exposed_area_mm2=float(A_j.sum()), h_mm=h_mm)


# ----------------------------------------------------------------------------- scene
def surface_names(cfg):
    """Read-only surfaces carried by the body ('body' is the body mesh itself, whichever it is)."""
    names = ["body", "uterus", "hrctv"]
    if cfg.get("body_mesh", "pre_body") == "pre_bodyvag":
        names.append("vagina")
    return names


def add_linear_solver(node, cfg, name="ls"):
    ls = cfg["linear_solver"]
    if ls == "SparseLDLSolver":
        return node.addObject("SparseLDLSolver", name=name, template="CompressedRowSparseMatrixMat3x3d")
    if ls == "EigenSimplicialLDLT":
        return node.addObject("EigenSimplicialLDLT", name=name, template="CompressedRowSparseMatrixMat3x3d")
    if ls == "CGLinearSolver":
        return node.addObject("CGLinearSolver", name=name, iterations=int(cfg["cg_iterations"]),
                              tolerance=float(cfg["cg_tolerance"]), threshold=float(cfg["cg_threshold"]))
    raise ValueError(ls)


def add_static_solver(node, cfg):
    return node.addObject("StaticSolver", name="ode", newton_iterations=int(cfg["newton_iterations"]),
                          absolute_correction_tolerance_threshold=float(cfg["abs_corr_tol_mm"]),
                          relative_correction_tolerance_threshold=float(cfg["rel_corr_tol"]),
                          absolute_residual_tolerance_threshold=float(cfg["abs_res_tol_mN"]),
                          relative_residual_tolerance_threshold=float(cfg["rel_res_tol"]),
                          should_diverge_when_residual_is_growing=True)


def build_scene(root, cfg=None, inp=None, tie_mode="dnt"):
    """Build the graph. Returns ctx (handles + node sets + canal weights). tie_mode 'mapped' is the G0.3
    reference (RSSFF on the mapped canal; valid with CG only)."""
    cfg = cfg if cfg is not None else config.load_cfg(os.environ.get("APPSIM_CFG"))
    inp = inp or load_inputs()
    rs = float(cfg.get("sph_r_scale") or 1.0); dz = float(cfg.get("sph_dz_mm") or 0.0)
    if rs != 1.0 or dz != 0.0:          # device size / position tolerance (calibration); the scoring uses the MEASURED pack
        import copy as _copy
        inp = dict(inp); app = _copy.deepcopy(inp["app"])
        for s_ in app["spheres"]:
            c_ = list(s_["center_app_mm"]); c_[2] = float(c_[2]) + dz; s_["center_app_mm"] = c_; s_["r_mm"] = float(s_["r_mm"]) * rs
        inp["app"] = app
    body_obj = os.path.join(inp["inp"], "%s.obj" % cfg.get("body_mesh", "pre_body"))
    X0, hexa, grid_n, vols = probe_grid(body_obj, cfg)
    Vbody, Fbody = geom.read_obj(body_obj)
    vols["body_mesh_cc"] = round(geom.mesh_volume(Vbody, Fbody) / 1000.0, 3)
    ns = node_sets(X0, hexa, cfg, inp)
    canal_full, s0, dlt = extended_canal(cfg, inp)
    cells_f, cnodes_f, W_f, clamped_f = geom.trilinear_weights(canal_full, X0, hexa)
    n_tip = int(cfg["tip_push_n_pts"])
    if cfg.get("tie_group", "point") == "cell":
        canal, cnodes, W, grp, grp_n = group_ties(canal_full, cells_f, cnodes_f, W_f, n_tip)
        cells = cells_f[[int(np.nonzero(grp == g)[0][0]) for g in range(len(canal))]]
        clamped = np.array([bool(clamped_f[grp == g].any()) for g in range(len(canal))])
    else:
        canal, cnodes, W, cells, clamped = canal_full, cnodes_f, W_f, cells_f, clamped_f
        grp = np.arange(len(canal)); grp_n = np.ones(len(canal), int)
    c_off_full = canal_full - np.einsum("ij,ijk->ik", W_f, X0[cnodes_f])
    # Optional tip cap patch (NUMERICAL regularisation, default off = spec): the last tip_push_n_pts canal points
    # read out / are pushed through a smooth kernel (1-(d/rho)^2)^2 over all body nodes within rho of their rest
    # position, instead of the 8 nodes of one cell -> the rigid tip loads a blunt patch, not a point.  A constant
    # rest offset keeps the read-out equal to the canal point at rest (linear map, same DNT algebra).
    rho = float(cfg.get("tip_patch_radius_mm") or 0.0); n_tip = int(cfg["tip_push_n_pts"])
    is_patch = np.zeros(len(canal), bool)
    rows_n, rows_w = [list(r) for r in cnodes], [list(w) for w in W]
    if rho > 0:
        for i in range(len(canal) - n_tip, len(canal)):
            d = np.linalg.norm(X0 - canal[i], axis=1); sel = np.nonzero(d < rho)[0]
            if len(sel) >= 8:
                w = (1 - (d[sel] / rho) ** 2) ** 2
                rows_n[i], rows_w[i] = list(sel), list(w / w.sum()); is_patch[i] = True
    M = max(len(r) for r in rows_n)
    cnodes = np.array([r + [r[0]] * (M - len(r)) for r in rows_n], int)
    W = np.array([w + [0.0] * (M - len(w)) for w in rows_w], float)
    c_off = canal - np.einsum("ij,ijk->ik", W, X0[cnodes])
    tie_nodes = np.unique(cnodes.ravel())
    loc = -np.ones(len(X0), int); loc[tie_nodes] = np.arange(len(tie_nodes))
    pose = select_pose(cfg, inp)
    csurf = None
    if cfg["m2"] and cfg.get("contact_impl", "rssff_nodes") == "pyff_surface":
        if cfg.get("tie_impl") != "pyff":
            raise ValueError("contact_impl='pyff_surface' lives in the NodeTieFF linearisation: needs tie_impl='pyff'")
        Vs, _ = geom.read_obj(body_obj)                  # the TRUE tissue boundary (grid nodes lie up to 1 cell outside)
        _, sn, sw, scl = geom.trilinear_weights(Vs, X0, hexa)
        csurf = dict(pts0=Vs, nodes=sn, W=sw, off=Vs - np.einsum("ij,ijk->ik", sw, X0[sn]), clamped=scl)
    if not cfg["m2"] or csurf is not None:
        cand = np.zeros(0, int)
    elif cfg.get("contact_candidates", "radius") == "all":
        cand = np.arange(len(X0))
    else:
        cand = np.nonzero(np.linalg.norm(X0 - pose["F"], axis=1) <= cfg["contact_candidate_radius_mm"])[0]

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = 1.0
    root.addObject("RequiredPlugin", pluginName="Sofa.Component")
    if cfg.get("fe_impl", "hexa") == "nonuniform":     # not part of the Sofa.Component bundle in v22.12 (MEASURED)
        root.addObject("RequiredPlugin", pluginName="Sofa.Component.SolidMechanics.FEM.NonUniform")
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields showVisual")
    tg = root.addChild("targets")
    tie_tgt = tg.addObject("MechanicalObject", name="tie_tgt", template="Vec3d", position=X0[tie_nodes].tolist())
    con_tgt = None
    if len(cand):
        con_tgt = tg.addObject("MechanicalObject", name="con_tgt", template="Vec3d", position=X0[cand].tolist())
    ref_tgt = None
    if tie_mode == "mapped":
        ref_tgt = tg.addObject("MechanicalObject", name="ref_tgt", template="Vec3d", position=canal.tolist())

    # device-rigid support anchors (found_anchor='device_rigid'): the rigid map that puts the rest canal on the final
    # rod (a0 -> a by the minimal rotation about L_end; L_end at s_Lend along the rod so the canal end sits at the
    # tip-push target L_iu + margin), ramped with the insertion progress by the controller
    anchor = None
    found_tgt = None
    if cfg.get("found_anchor", "rest") == "device_rigid":
        i_le = int(inp["canal"]["i_L_end"]); L_end = np.array(inp["poses"]["L_end"], float)
        a0 = geom.unit(inp["poses"]["a0"]); a_f = geom.unit(pose["a"])
        s_Lend = float(inp["app"]["L_iu_mm"]) + float(cfg["tip_push_margin_mm"]) - float(s0[-1] - s0[i_le])
        Rm = geom.rot_between(a0, a_f)
        ang = geom.angle_deg(a0, a_f); axis = geom.unit(np.cross(a0, a_f)) if ang > 1e-9 else np.array([0.0, 0.0, 1.0])
        anchor = dict(pivot=L_end, axis=axis, angle_deg=ang, R=Rm, shift=np.asarray(pose["F"], float) + s_Lend * a_f - L_end,
                      s_Lend_mm=s_Lend)
        found_tgt = tg.addObject("MechanicalObject", name="found_tgt", template="Vec3d", position=X0[ns["sup_nodes"]].tolist())

    body = root.addChild("body")
    ode = add_static_solver(body, cfg)
    lsol = add_linear_solver(body, cfg)
    _sparse_grid(body, body_obj, cfg, n=grid_n)
    dofs = body.addObject("MechanicalObject", name="dofs", template="Vec3d")
    if cfg.get("fe_impl", "hexa") == "nonuniform":
        body.addObject("NonUniformHexahedronFEMForceFieldAndMass", name="fem", youngModulus=float(cfg["E_kPa"]),
                       poissonRatio=float(cfg["nu"]), method="large", nbVirtualFinerLevels=int(cfg["nb_virtual_finer_levels"]),
                       density=1e-6)            # kg/mm^3 ~ water; mass is never used (StaticSolver, gravity 0)
    else:
        body.addObject("HexahedronFEMForceField", name="fem", youngModulus=float(cfg["E_kPa"]),
                       poissonRatio=float(cfg["nu"]), method="large")
    if found_tgt is not None:
        found = body.addObject("RestShapeSpringsForceField", name="found", points=ns["sup_nodes"].tolist(),
                               stiffness=ns["k_sup"].tolist(), external_rest_shape="@/targets/found_tgt",
                               external_points=list(range(len(ns["sup_nodes"]))))
    else:
        found = body.addObject("RestShapeSpringsForceField", name="found", points=ns["sup_nodes"].tolist(),
                               stiffness=ns["k_sup"].tolist())
    tie = None
    tie_impl = cfg.get("tie_impl", "rssff")
    if tie_mode == "dnt" and tie_impl == "pyff":
        # anisotropic DNT (frictionless sliding inside the solve); matrix-free -> CG only (see ties.py)
        if cfg["linear_solver"] != "CGLinearSolver":
            raise ValueError("tie_impl='pyff' is matrix-free and needs CGLinearSolver; use tie_impl='rssff' with %s"
                             % cfg["linear_solver"])
        from ties import NodeTieFF
        tie = body.addObject(NodeTieFF(name="tie"))
    elif tie_mode == "dnt":
        tie = body.addObject("RestShapeSpringsForceField", name="tie", points=tie_nodes.tolist(),
                             external_rest_shape="@/targets/tie_tgt", external_points=list(range(len(tie_nodes))),
                             stiffness=[0.0] * len(tie_nodes))
    con = None
    if len(cand):
        con = body.addObject("RestShapeSpringsForceField", name="con", points=cand.tolist(),
                             external_rest_shape="@/targets/con_tgt", external_points=list(range(len(cand))),
                             stiffness=[0.0] * len(cand))
    surfs = {}
    if cfg["mapped_children"] or tie_mode == "mapped":
        for nm in surface_names(cfg):
            s = body.addChild("surf_" + nm)
            fn = cfg.get("body_mesh", "pre_body") if nm == "body" else "pre_%s" % nm   # 'body' = the body mesh itself
            s.addObject("MeshObjLoader", name="loader", filename=os.path.join(inp["inp"], "%s.obj" % fn))
            s.addObject("MechanicalObject", name="mo", src="@loader")
            s.addObject("BarycentricMapping", input="@../dofs", output="@mo")
            surfs[nm] = s
    cn = body.addChild("canal")                  # read-out of the FULL canal (every 1 mm point), never tied here
    cmo = cn.addObject("MechanicalObject", name="cdofs", template="Vec3d", position=canal_full.tolist())
    cn.addObject("BarycentricMapping", input="@../dofs", output="@cdofs")
    ref = None
    if tie_mode == "mapped":   # G0.3 reference only: springs on MAPPED points (CG handles mapped K)
        if len(canal_full) != len(canal):
            raise ValueError("tie_mode='mapped' needs tie_group='point'")
        ref = cn.addObject("RestShapeSpringsForceField", name="ref_tie", points=list(range(len(canal))),
                           external_rest_shape="@/targets/ref_tgt", external_points=list(range(len(canal))),
                           stiffness=[0.0] * len(canal))
    ctx = dict(root=root, body=body, ode=ode, ls=lsol, dofs=dofs, found=found, tie=tie, con=con, tie_tgt=tie_tgt, con_tgt=con_tgt,
               ref=ref, ref_tgt=ref_tgt, surfs=surfs, tie_impl=(tie_impl if tie_mode == "dnt" else "mapped"), canal_mo=cmo, X0=X0, hexa=hexa, grid_n=grid_n, ns=ns,
               canal=canal, s0=s0, delta_fund_mm=dlt, cells=cells, cnodes=cnodes, W=W, clamped=clamped,
               c_off=c_off, is_patch=is_patch, tip_patch_radius_mm=rho, contact_surface=csurf,
               tie_nodes=tie_nodes, loc=loc, cand=cand, pose=pose, inp=inp, cfg=cfg, tie_mode=tie_mode,
               body_obj=body_obj, vols=vols,
               canal_full=canal_full, cnodes_full=cnodes_f, W_full=W_f, c_off_full=c_off_full, grp=grp, grp_n=grp_n,
               anchor=anchor, found_tgt=found_tgt)
    return ctx


def scene_summary(ctx):
    ns = ctx["ns"]
    return dict(n_nodes=int(len(ctx["X0"])), n_hexa=int(len(ctx["hexa"])), grid_n=ctx["grid_n"],
                cell_real_mm=np.round(ns["cell_real_mm"], 3).tolist(), n_boundary=int(len(ns["boundary"])),
                n_support_nodes=int(len(ns["sup_nodes"])), n_lig_nodes=int(len(ns["lig"])),
                n_vag_floor_nodes=int(len(ns.get("vfloor", []))), body_mesh=ctx["cfg"].get("body_mesh", "pre_body"),
                exposed_area_mm2=round(ns["exposed_area_mm2"], 1),
                total_found_mN_per_mm=round(ns["total_found_mN_per_mm"], 3),
                total_lig_mN_per_mm=round(ns["total_lig_mN_per_mm"], 3),
                n_canal_pts=int(len(ctx["canal"])), n_canal_clamped=int(ctx["clamped"].sum()),
                n_tie_nodes=int(len(ctx["tie_nodes"])), n_contact_candidates=int(len(ctx["cand"])),
                n_contact_surface_pts=int(len(ctx["contact_surface"]["pts0"])) if ctx.get("contact_surface") else 0,
                n_contact_surface_clamped=int(ctx["contact_surface"]["clamped"].sum()) if ctx.get("contact_surface") else 0,
                tip_patch_radius_mm=ctx["tip_patch_radius_mm"], n_patch_pts=int(ctx["is_patch"].sum()),
                patch_nodes_per_pt=int(ctx["W"].shape[1]),
                delta_fund_mm=ctx["delta_fund_mm"], pose=ctx["pose"]["name"],
                d_F_equiv_mm=round(float(ctx["pose"]["d_F_mm"]), 3),
                fe_impl=ctx["cfg"].get("fe_impl", "hexa"), volumes_cc=ctx.get("vols"),
                tie_group=ctx["cfg"].get("tie_group", "point"), n_canal_full=int(len(ctx["canal_full"])),
                n_tie_pts=int(len(ctx["canal"])), tie_pts_members_max=int(ctx["grp_n"].max()),
                canal_fillet_mm=float(ctx["cfg"].get("canal_fillet_mm") or 0.0),
                found_anchor=ctx["cfg"].get("found_anchor", "rest"),
                anchor_s_Lend_mm=(round(ctx["anchor"]["s_Lend_mm"], 3) if ctx.get("anchor") else None),
                anchor_rot_deg=(round(ctx["anchor"]["angle_deg"], 3) if ctx.get("anchor") else None))


def createScene(rootNode):
    """runSofa entry point: reads the run config from env APPSIM_CFG (defaults if unset), adds the controller."""
    cfg = config.load_cfg(os.environ.get("APPSIM_CFG"))
    ctx = build_scene(rootNode, cfg)
    from controller import ApplicatorController, build_schedule
    rootNode.addObject(ApplicatorController(name="applicator", ctx=ctx, schedule=build_schedule(ctx), out_dir=None))
    return rootNode
