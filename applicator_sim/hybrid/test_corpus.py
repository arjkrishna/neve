"""Host unit tests (python 3.13, numpy) of the S9 elastic-corpus helpers in scene_hybrid.py (cfg corpus_model "fem").
SOFA is stubbed exactly as in hybrid/test_ties.py (scene_hybrid needs only Sofa.Core.Controller as a base class).

    python -P -B hybrid/test_corpus.py          # runs every test_* below (pytest-style; pytest is not on this host)

What is tested:
  pose    corpus_pose_targets with no stretch (and lam 1) is, bit for bit, the rigid corpus's placement X0 @ R^T + t;
          the calibrated stretch keeps every tet's volume, holds the anchor level, moves the top by (lam - 1)(h - h0)
          and scales lateral offsets by 1 / sqrt(lam); corpus_stretch_at ramps lam with w_r (or s).
  nodes   corpus_pose_nodes: the serosa / surface / all / shell sets, the bottom springs, the canal exclusion.
  follow  follow_surface: the OARs' copy equals the target when the corpus is on its previous target.
  ties    clip_canal_s, corpus_tie_params + canal_tie_step (depth engagement of clipped nodes, "centre" both ways),
          axial_tie_targets (along the tube only, bounded, active nodes only).
  bary    bary_weights / bary_apply: exact at rest, weights sum to 1, inside points non-negative, and an AFFINE motion
          of the tets carries inside AND extrapolated outside points exactly (SOFA's extrapolation rule); read_vtk_tets
          and the real corpus / cervix interface when this machine has the meshes.
  ctrl    a HybridController on a fake elastic-corpus ctx: _begin writes the pose target (= the rigid placement), the
          follow copy and the predicted attach target, the corpus ties engage by depth; _corpus_log reads it back; a
          rigid ctx keeps body_order == DEFORMABLE.
  eu2     corpus_tie_follow (review HIGH-2): with the tube moving WITH the corpus and the tie nodes on the tube, the
          follow targets equal the end-of-step one-body positions (zero spring force) while the default targets hold
          the start-of-step nodes (force k x the carried motion, above the k x cap bound) and equal canal_tie_step
          without carry; an off-target corpus gets its lateral misfit corrected and its axial offset kept; with a
          calibrated stretch (corpus_pose_stretch_lam != 1) the carry is the stretched target's change and the targets
          are canal_tie_step at the new stretched target (its axial elongation kept, not pulled back).
All geometry is synthetic (no patient or case-derived numbers); test_real_corpus_interface_and_canal_set checks this
machine's meshes against quantities derived from the data themselves (meta.json sizes, the schedule's tip_s reach).
"""
import json
import os
import sys
import types

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)


class _Controller(object):                              # stands in for Sofa.Core.Controller
    def __init__(self, *a, **k):
        pass


def _stub_sofa():
    for name in ("Sofa", "Sofa.Core", "Sofa.Simulation", "SofaRuntime"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["Sofa"].Core = sys.modules["Sofa.Core"]
    sys.modules["Sofa"].Simulation = sys.modules["Sofa.Simulation"]
    if not hasattr(sys.modules["Sofa.Core"], "Controller"):
        sys.modules["Sofa.Core"].Controller = _Controller
    for p in (HERE, PARENT):
        if p not in sys.path:
            sys.path.insert(0, p)


_stub_sofa()
import geom  # noqa: E402
import scene_hybrid as S  # noqa: E402

R_TUBE, L_IU = 2.5, 60.0                                # a synthetic tandem: radius, length above the flange (mm)


# ----------------------------------------------------------------------------- synthetic geometry
def _cube_tets(n=4, h=5.0, origin=(0.0, 0.0, 0.0)):
    """A structured n^3 cube of side n*h mm split into 6 tets per cell (positive orientation)."""
    g = np.arange(n + 1) * h
    P = np.array([[x, y, z] for z in g for y in g for x in g], float) + np.asarray(origin, float)

    def v(i, j, k):
        return i + (n + 1) * (j + (n + 1) * k)
    T = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k), v(i, j + 1, k),
                     v(i, j, k + 1), v(i + 1, j, k + 1), v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)]
                for a, b in ((1, 2), (2, 3), (3, 7), (7, 4), (4, 5), (5, 1)):
                    T.append([c[0], c[a], c[b], c[6]])
    T = np.array(T)
    vol = S._tet_vol(P, T)
    T[vol < 0] = T[vol < 0][:, [0, 2, 1, 3]]
    return P, T


def _rigid(seed=0, deg=17.0):
    rng = np.random.default_rng(seed)
    T = np.eye(4)
    T[:3, :3] = S.rodrigues(rng.normal(size=3), deg)
    T[:3, 3] = rng.normal(0, 20, 3)
    return T


def _real_corpus():
    """(X0, tets, meta, cervix X0, cervix meta) of this machine's meshes, or None."""
    try:
        P = S.paths()
        X0, T = S.read_vtk_tets(P["meshes"] + "/corpus/tets.vtk")
        meta = json.load(open(P["meshes"] + "/corpus/meta.json"))
        Xc = S.read_vtk_points(P["meshes"] + "/cervix/tets.vtk")
        mc = json.load(open(P["meshes"] + "/cervix/meta.json"))
        return X0, T, meta, Xc, mc
    except Exception:
        return None


# ----------------------------------------------------------------------------- pose targets
def test_pose_targets_rigid_bit_identical():
    """No stretch (None, or lam 1.0): exactly the rigid corpus's placement X0 @ T[:3, :3].T + T[:3, 3] (run_hybrid,
    write_frame), so a stiff pose set reproduces the rigid corpus's target bit for bit."""
    P, _ = _cube_tets()
    T = _rigid(1)
    ref = P @ T[:3, :3].T + T[:3, 3]
    assert np.array_equal(S.corpus_pose_targets(P, T), ref)
    assert np.array_equal(S.corpus_pose_targets(P, T, dict(lam=1.0, axis=[0, 0, 1])), ref)
    idx = np.array([5, 1, 30])
    assert np.array_equal(S.corpus_pose_targets(P, T, None, idx), ref[idx])
    assert S.corpus_stretch_at(1.0, dict(w_r=0.7, s=0.3, tube_axis=[0, 0, 1])) is None


def test_stretch_volume_anchor_top_lateral():
    """The calibrated stretch: every tet keeps its volume (det 1), the anchor level h0 (0.5 percentile of h about the
    centroid) stays, the top rises by (lam - 1)(h_top - h0), lateral offsets from the axis line shrink by sqrt(lam)."""
    P, Tt = _cube_tets(5, 4.0, (10.0, -3.0, 40.0))
    T = _rigid(2, 11.0)
    a = geom.unit([0.45, 0.3, 0.84])                     # any unit axis (synthetic)
    lam = 1.05
    X = S.corpus_pose_targets(P, T)
    Y = S.corpus_pose_targets(P, T, dict(lam=lam, axis=a))
    v0, v1 = S._tet_vol(X, Tt), S._tet_vol(Y, Tt)
    assert np.allclose(v1, v0, rtol=1e-10, atol=1e-12), np.abs(v1 / v0 - 1).max()
    c = X.mean(0)
    h, hy = (X - c) @ a, (Y - c) @ a
    h0 = np.percentile(h, 0.5)
    assert np.allclose(hy - h0, lam * (h - h0), atol=1e-9)
    top = int(np.argmax(h))
    assert abs((hy[top] - h[top]) - (lam - 1.0) * (h[top] - h0)) < 1e-9
    lat_x = (X - c) - np.outer(h, a)
    lat_y = (Y - c) - np.outer(hy, a)
    assert np.allclose(lat_y, lat_x / np.sqrt(lam), atol=1e-9)
    return dict(top_rise_mm=round(float(hy[top] - h[top]), 4), max_vol_change=float(np.abs(v1 / v0 - 1).max()))


def test_stretch_at_ramps_with_w():
    """lam_row = 1 + (lam - 1) w with w = w_r (tandem-first rows) else s; the axis is the row's tube axis."""
    r = dict(w_r=0.4, s=0.9, tube_axis=[0.0, 0.0, 1.0])
    st = S.corpus_stretch_at(1.05, r)
    assert abs(st["lam"] - 1.02) < 1e-12 and st["w"] == 0.4 and np.allclose(st["axis"], [0, 0, 1])
    st = S.corpus_stretch_at(1.10, dict(s=0.5, tube_axis=[1.0, 0.0, 0.0]))
    assert abs(st["lam"] - 1.05) < 1e-12
    assert abs(S.corpus_stretch_at(1.05, dict(w_r=0.0, s=0.0, tube_axis=[0, 0, 1]))["lam"] - 1.0) < 1e-15


# ----------------------------------------------------------------------------- pose node sets
def _meta(n=60):
    surf = list(range(0, 40))
    bot = list(range(30, 45))                          # 10 on the surface, 5 interior (as the real interface is not)
    return dict(node_sets=dict(surface_nodes=surf, interface_cervix=bot)), n


def test_pose_nodes_sets_and_stiffness():
    meta, n = _meta()
    cfg = S.load_cfg({})
    sel = S.corpus_pose_nodes(meta, cfg, n)
    ser = np.setdiff1d(range(40), range(30, 45))
    assert np.array_equal(sel["idx"], ser) and np.all(sel["k"] == cfg["k_corpus_pose_mN_per_mm"])
    sel = S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_set="shell")), n)
    assert np.array_equal(sel["idx"], ser) and np.all(sel["k"] == cfg["k_fixed_mN_per_mm"])
    sel = S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_set="surface")), n)
    assert np.array_equal(sel["idx"], np.arange(40))
    sel = S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_set="all")), n)
    assert np.array_equal(sel["idx"], np.arange(n))
    # bottom springs: every interface node, at k_bottom, also those already in the set
    sel = S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_set="surface", k_corpus_bottom_mN_per_mm=50.0)), n)
    assert np.array_equal(sel["idx"], np.arange(45))
    kb = dict(zip(sel["idx"].tolist(), sel["k"].tolist()))
    assert all(kb[i] == 50.0 for i in range(30, 45)) and all(kb[i] == 5.0 for i in range(30))
    assert sel["info"]["n_bottom_springs"] == 15
    try:
        S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_set="nope")), n)
        raise AssertionError("an unknown pose set must be refused")
    except ValueError:
        pass


def test_pose_nodes_canal_exclusion():
    """corpus_pose_exclude_canal_mm drops the pose nodes within R of the canal polyline (and only those)."""
    meta, n = _meta()
    X0 = np.zeros((n, 3))
    X0[:, 0] = np.arange(n) * 1.0                     # nodes on the x axis, 1 mm apart
    canal = np.array([[0.0, 2.0, 0.0], [5.0, 2.0, 0.0]])      # 2 mm above nodes 0..5
    sel = S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_exclude_canal_mm=2.5)), n, X0, canal)
    d = S._polyline_dist(X0, canal)
    ser = np.setdiff1d(range(40), range(30, 45))
    assert np.array_equal(sel["idx"], ser[d[ser] > 2.5]) and sel["info"]["n_excluded_near_canal"] == 7   # 0..6
    try:
        S.corpus_pose_nodes(meta, S.load_cfg(dict(corpus_pose_exclude_canal_mm=2.5)), n)
        raise AssertionError("the exclusion needs the rest nodes and the canal")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- follow copy, ties
def test_follow_surface_predictor():
    """A corpus sitting on its previous target gives the new target exactly; otherwise its offset is carried along."""
    rng = np.random.default_rng(3)
    X0 = rng.normal(0, 10, (50, 3))
    T0, T1 = _rigid(4, 5.0), _rigid(4, 6.0)
    t0, t1 = S.corpus_pose_targets(X0, T0), S.corpus_pose_targets(X0, T1)
    assert np.allclose(S.follow_surface(t0, t1, t0), t1, atol=1e-12)
    e = rng.normal(0, 0.3, X0.shape)
    assert np.allclose(S.follow_surface(t0 + e, t1, t0), t1 + e, atol=1e-12)


def test_clip_canal_s():
    """Nodes above the deepest tip engage at tip_s_max - 0.05; the others keep their canal_s (synthetic values)."""
    cs = np.array([5.0, 18.0, 29.9, 30.4, 33.0])
    c = S.clip_canal_s(cs, 30.0)
    assert np.allclose(c, [5.0, 18.0, 29.9, 29.95, 29.95])
    assert np.all(c <= 30.0) and np.all(c <= cs)


def test_corpus_ties_depth_engagement_clipped():
    """canal_tie_step with corpus_tie_params: "centre" both ways, depth engagement on the CLIPPED canal_s -- a node
    whose canal_s the tip never reaches (above the deepest tip_s) engages at the row where the tip is deepest."""
    cfg = S.load_cfg(dict(corpus_model="fem"))
    p = S.corpus_tie_params(cfg, R_TUBE, L_IU)
    assert p["centre"] and p["engage"] == "depth" and p["below"] == 0.0
    assert p["cap"] == cfg["corpus_canal_max_offset_mm"] and p["k"] == cfg["k_corpus_canal_mN_per_mm"]
    a = geom.unit([-0.35, 0.25, 0.9])                    # any unit axis (synthetic)
    F = np.zeros(3)
    tip_max = 40.0
    cs = np.array([10.0, 25.0, 40.6, 42.5])                                 # the last two beyond the deepest tip
    e1 = geom.ortho([1.0, 0.0, 0.0], a)
    X = F + np.outer(20.0 + cs, a) + np.outer([0.5, 4.0, 1.0, 6.0], e1)      # inside, outside the tube radius
    tips = list(np.linspace(0.0, tip_max, 40)) + [tip_max] * 5
    eng = np.full(len(cs), -1, int)
    cs_eng = S.clip_canal_s(cs, max(tips))
    for k, ts in enumerate(tips):
        res = S.canal_tie_step(X, F, a, a, eng, k, p, tip_s=ts, canal_s=cs_eng)
    first = [next(k for k, t in enumerate(tips) if t >= c) for c in cs_eng]
    assert list(eng) == first, (eng, first)
    assert eng[2] == eng[3] == 39                      # the two clipped nodes: at the deepest row
    d, h = _line(X, F, a)
    inside = (h >= 0) & (h <= p["L_iu"])               # engaged nodes above the tube's tip are not tied (inactive)
    assert list(inside) == [True, True, False, False] and list(res["act"]) == list(inside)
    off = np.linalg.norm(res["tgt"] - X, axis=1)
    assert np.allclose(off, np.where(inside, np.minimum(np.abs(p["rad"] - d), p["cap"]), 0.0))
    dT, _ = _line(res["tgt"], F, a)
    assert dT[0] > d[0] and dT[1] < d[1]               # pushed out from 0.5 mm, pulled in from 4 mm: both ways
    # without the clip the two top nodes never engage
    eng2 = np.full(len(cs), -1, int)
    for k, ts in enumerate(tips):
        S.canal_tie_step(X, F, a, a, eng2, k, p, tip_s=ts, canal_s=cs)
    assert eng2[2] == eng2[3] == -1


def _line(P, F, a):
    q = np.asarray(P, float) - F
    s = q @ a
    return np.linalg.norm(q - np.outer(s, a), axis=1), s


def test_axial_tie_targets():
    """arclength: active targets also move along the tube toward h* = L_iu - (tip_s - canal_s), bounded by cap;
    the lateral part of canal_tie_step's target is unchanged; inactive nodes keep their target."""
    a = geom.unit([0.3, -0.4, 0.87])                     # any unit axis (synthetic)
    e = geom.ortho([1.0, 0.0, 0.0], a)
    F = np.array([1.0, 2.0, 3.0])
    X = F + np.outer([20.0, 30.0, 40.0], a) + np.outer([1.0, 0.0, 3.0], e)
    res = dict(tgt=X + np.outer([0.3, 0.0, 0.2], e), act=np.array([True, True, False]))
    L_iu, tip_s, cs, cap = L_IU, 40.0, np.array([5.0, 10.3, 30.0]), 0.5
    tgt, dh = S.axial_tie_targets(res, X, F, a, tip_s, cs, L_iu, cap)
    want = np.clip(L_iu - (tip_s - cs) - (X - F) @ a, -cap, cap) * res["act"]
    assert np.allclose(dh, want) and dh[2] == 0.0
    assert np.allclose(tgt - res["tgt"], np.outer(dh, a))
    assert np.allclose(tgt[2], res["tgt"][2])
    assert abs(dh[1] - 0.3) < 1e-12 and abs(dh[0]) == cap


# ----------------------------------------------------------------------------- barycentric map (SOFA's rule)
def test_bary_weights_affine_exact():
    """Rest points reproduced exactly; weights sum to 1; inside points non-negative; an AFFINE motion of the mesh
    carries inside and extrapolated outside points exactly (so a rigid corpus maps the interface rigidly)."""
    P, T = _cube_tets(3, 5.0)
    rng = np.random.default_rng(5)
    Qin = rng.uniform(0.5, 14.5, (40, 3))
    Qout = np.array([[-1.5, 7.0, 7.0], [16.2, 3.0, 9.0], [7.0, -2.0, 15.8], [15.5, 15.5, 15.5]])
    Q = np.vstack([Qin, Qout])
    idx, W = S.bary_weights(P, T, Q)
    assert np.allclose(W.sum(1), 1.0)
    assert W[:len(Qin)].min() >= -1e-9 and W[len(Qin):].min() < 0.0          # outside: extrapolated
    assert np.abs(S.bary_apply(P, idx, W) - Q).max() < 1e-12
    A = np.array([[1.05, 0.1, -0.02], [0.03, 0.97, 0.05], [0.0, -0.04, 1.01]])
    b = np.array([3.0, -4.0, 5.0])
    assert np.abs(S.bary_apply(P @ A.T + b, idx, W) - (Q @ A.T + b)).max() < 1e-10
    Tr = _rigid(6, 25.0)
    assert np.abs(S.bary_apply(S.corpus_pose_targets(P, Tr), idx, W) - S.corpus_pose_targets(Q, Tr)).max() < 1e-10


def test_real_corpus_interface_and_canal_set():
    """This machine's meshes (skipped otherwise): read_vtk_tets gives the corpus with the node / tet counts its own
    meta.json records; the cervix interface_corpus nodes map exactly at rest (some extrapolated); the corpus canal_path
    set (tandem_path.py corpus_nodesets) exists with its arclengths, ordered, reaching the deepest tip_s of the
    tandem-first schedule (tandem_path.tip_s_reach, when that record exists) to within corpus_set_reach_tol_mm."""
    r = _real_corpus()
    if r is None:
        return "skipped (no meshes on this machine)"
    X0, T, meta, Xc, mc = r
    assert X0.shape == (int(meta["tets"]["nodes"]), 3) and T.shape == (int(meta["tets"]["tets"]), 4)
    assert T.min() >= 0 and T.max() < len(X0) and np.all(S._tet_vol(X0, T) > 0.0)
    assert np.allclose(X0, S.read_vtk_points(S.paths()["meshes"] + "/corpus/tets.vtk"))
    iface = np.asarray(mc["node_sets"]["interface_corpus"], int)
    idx, W = S.bary_weights(X0, T, Xc[iface])
    rest = float(np.abs(S.bary_apply(X0, idx, W) - Xc[iface]).max())
    assert rest < 1e-9, rest
    n_out = int((W.min(1) < -1e-9).sum())
    ns = meta["node_sets"]
    out = dict(rest_err_mm=rest, n_iface=len(iface), n_extrapolated=n_out, min_weight=round(float(W.min()), 3))
    if "canal_path" in ns:
        import tandem_path as TP
        cs = np.asarray(ns["canal_path_s"], float)
        assert len(cs) == len(ns["canal_path"]) >= TP.ACCEPT["set_min_nodes"] and np.all(np.diff(cs) >= 0)
        reach, _ = TP.tip_s_reach()
        if reach is not None:
            assert cs.max() >= reach - TP.ACCEPT["corpus_set_reach_tol_mm"], (cs.max(), reach)
        out.update(n_canal_path=len(cs), reach_checked=reach is not None)
    sel = S.corpus_pose_nodes(meta, S.load_cfg({}), len(X0))
    out["n_serosa"] = int(len(sel["idx"]))
    return out


# ----------------------------------------------------------------------------- the controller on a fake fem ctx
class _Data(object):
    def __init__(self, v):
        self.value = v


def _obj(**data):
    return types.SimpleNamespace(**{k: _Data(v) for k, v in data.items()})


def _fem_ctx(cfg_over=None):
    """A minimal elastic-corpus ctx: a cube corpus (+ its serosa / bottom sets), a one-node-per-row cervix interface,
    3 corpus tie nodes, rows turning and lifting the corpus with tip_s growing."""
    Pk, Tk = _cube_tets(3, 5.0)
    n = len(Pk)
    surf = np.nonzero((Pk.min(1) <= 0) | (Pk.max(1) >= 15))[0]
    bot = np.nonzero(Pk[:, 2] <= 0)[0]
    meta = dict(node_sets=dict(surface_nodes=surf.tolist(), interface_cervix=bot.tolist()),
                surface_obj_vertex_to_tet_node=surf.tolist())
    cfg = S.load_cfg(dict(dict(corpus_model="fem", attach_target="predicted"), **(cfg_over or {})))
    Xc = np.array([[2.0, 3.0, -0.5], [7.0, 7.0, -1.0], [12.0, 2.0, 0.5]])     # cervix interface (synthetic, 2 outside)
    iface = np.arange(3)
    a = geom.unit([0.0, 0.0, 1.0])
    rows = []
    for k in range(12):
        T = np.eye(4)
        T[:3, :3] = S.rodrigues([1.0, 0.0, 0.0], 1.5 * k)
        T[:3, 3] = [0.0, 0.0, 0.8 * k]
        rows.append(dict(phase="C", u=k / 11.0, s=k / 11.0, w_r=k / 11.0, F=np.array([7.5, 7.5, -10.0]),
                         tube_axis=a, R_rows=geom.frame_from(a, [1.0, 0.0, 0.0]), T_corpus=T, tip_s=4.0 * k, ov_lag=0.0))
    sel = S.corpus_pose_nodes(meta, cfg, n)
    tie = np.array([13, 25, 38])
    cs = np.array([6.0, 20.0, 50.0])
    cs_eng = S.clip_canal_s(cs, max(r["tip_s"] for r in rows))
    ctx = dict(cfg=cfg, sched=rows, tgt=dict(tube_axis=a, axis=a, R_rows=rows[0]["R_rows"]),
               inp=dict(app=dict(params=dict(L_iu_mm=dict(value=L_IU), r_tandem_mm=dict(value=R_TUBE)), landmarks={}),
                        meta=dict(corpus=meta)),
               X0=dict(cervix=Xc.copy(), corpus=Pk.copy()), deformable=["cervix", "corpus"], iface=iface,
               nodes=dict(cervix=types.SimpleNamespace(dofs=_obj(position=Xc.tolist(), velocity=np.zeros_like(Xc).tolist())),
                          corpus=types.SimpleNamespace(dofs=_obj(position=Pk.tolist(), velocity=np.zeros_like(Pk).tolist()))),
               corpus=None, tandem=types.SimpleNamespace(rig=_obj(position=None)), ovoids=None,
               corpus_fem=object(), corpus_iface_mo=_obj(position=Xc.tolist()),
               corpus_iface_bary=S.bary_weights(Pk, Tk, Xc[iface]),
               corpus_iface_tgt=_obj(position=Xc.tolist()),
               corpus_pose=dict(idx=sel["idx"], k=sel["k"], serosa=sel["serosa"], bottom=sel["bottom"],
                                tgt_mo=_obj(position=Pk[sel["idx"]].tolist()), lam=float(cfg["corpus_pose_stretch_lam"])),
               corpus_follow=dict(mo=_obj(position=Pk[surf].tolist()), s2n=surf),
               corpus_canal=tie, corpus_canal_s=cs, corpus_canal_s_eng=cs_eng,
               corpus_canal_tgt=_obj(position=Pk[tie].tolist()), corpus_canal_ff=_obj(stiffness=[0.0] * 3),
               tets=dict(corpus=Tk), vol0=dict(corpus=S._tet_vol(Pk, Tk)))
    return ctx, rows


def test_controller_fem_begin_and_log():
    ctx, rows = _fem_ctx()
    ctrl = S.HybridController(name="h", ctx=ctx, out_dir=None)
    assert ctrl.corpus_fem and ctrl.body_order == S.DEFORMABLE + ["corpus"]
    Pk = ctx["X0"]["corpus"]
    eng_seen = []
    for k, r in enumerate(rows):
        ctrl.k = k
        ctrl._begin()
        T = np.asarray(r["T_corpus"], float)
        rigid = Pk @ T[:3, :3].T + T[:3, 3]
        cp = ctx["corpus_pose"]
        assert np.array_equal(np.asarray(cp["tgt_mo"].position.value), rigid[cp["idx"]])      # the rigid target
        Y = np.asarray(ctx["corpus_follow"]["mo"].position.value)
        Zi = np.asarray(ctx["corpus_iface_tgt"].position.value)
        want_i = S.corpus_pose_targets(ctx["X0"]["cervix"][ctx["iface"]], T)
        assert np.abs(Y - rigid[ctx["corpus_follow"]["s2n"]]).max() < 1e-9                   # corpus on its target
        assert np.abs(Zi - want_i).max() < 1e-9                                               # interface: exact
        eng_seen.append(ctrl.eng_step_k.copy())
        # the step "solves": the corpus lands exactly on this row's target
        ctx["nodes"]["corpus"].dofs.position.value = rigid.tolist()
        ctx["nodes"]["cervix"].dofs.position.value = want_i.tolist()
    first = [next(k for k, r in enumerate(rows) if r["tip_s"] >= c) for c in ctx["corpus_canal_s_eng"]]
    assert list(ctrl.eng_step_k) == first, (ctrl.eng_step_k, first)                    # depth engagement, clipped
    assert np.all(np.asarray(ctx["corpus_canal_ff"].stiffness.value) > 0)
    X = np.asarray(ctx["nodes"]["corpus"].dofs.position.value)
    lg = ctrl._corpus_log(X)
    assert lg["pose_max_mm"] < 1e-9 and lg["nonrigid_serosa_max_mm"] < 1e-9 and abs(lg["vol_ratio_total"] - 1) < 1e-9
    assert lg["attach_residual_mm"] < 1e-9 and lg["n_ties_active"] == 3
    return dict(engage_steps=[int(v) for v in ctrl.eng_step_k], log_keys=sorted(lg))


def test_controller_rigid_unchanged_order():
    """A rigid ctx (no corpus_fem): body_order is DEFORMABLE and nothing of the elastic corpus is initialised."""
    ctx, rows = _fem_ctx()
    ctx["corpus_fem"] = None
    ctx["corpus"] = types.SimpleNamespace(rig=_obj(position=None))
    ctrl = S.HybridController(name="h", ctx=ctx, out_dir=None)
    assert not ctrl.corpus_fem and ctrl.body_order == S.DEFORMABLE and not hasattr(ctrl, "eng_step_k")
    # attach_target "predicted" with the rigid corpus: this row's interface pose, exactly; the rig placed as always
    ctrl.k = 7
    ctrl._begin()
    T = np.asarray(rows[7]["T_corpus"], float)
    want = ctx["X0"]["cervix"][ctx["iface"]] @ T[:3, :3].T + T[:3, 3]
    assert np.array_equal(np.asarray(ctx["corpus_iface_tgt"].position.value), want)
    assert ctx["corpus"].rig.position.value == S.rigid_pose(T[:3, :3], T[:3, 3])


def _fem_ctx_one_body(cfg_over=None):
    """_fem_ctx with the tube moving WITH the corpus (as in L: one body) and 4 corpus tie nodes exactly on the tube
    surface at rest: the grid line (5, 5, z) of the cube, the tube axis parallel to it at the tube radius."""
    ctx, _ = _fem_ctx(cfg_over)
    rad = R_TUBE + float(ctx["cfg"]["canal_slack_mm"])
    a0 = np.array([0.0, 0.0, 1.0])
    F0 = np.array([5.0 + rad, 5.0, -10.0])
    R0 = geom.frame_from(a0, [1.0, 0.0, 0.0])
    rows = []
    for k in range(10):
        T = np.eye(4)
        T[:3, :3] = S.rodrigues([1.0, 0.3, 0.0], 1.5 * k)
        T[:3, 3] = [0.2 * k, -0.1 * k, 0.8 * k]
        Q = T[:3, :3]
        rows.append(dict(phase="L", u=1.0, s=1.0, w_r=1.0, F=F0 @ Q.T + T[:3, 3], tube_axis=Q @ a0, R_rows=R0 @ Q.T,
                         T_corpus=T, tip_s=4.0 * k, ov_lag=0.0))
    tie = np.array([5, 21, 37, 53])
    assert np.allclose(np.linalg.norm((ctx["X0"]["corpus"][tie] - F0)[:, :2], axis=1), rad)
    cs = np.array([1.0, 2.0, 3.0, 4.0])
    ctx.update(sched=rows, corpus_canal=tie, corpus_canal_s=cs, corpus_canal_s_eng=cs.copy(),
               corpus_canal_tgt=_obj(position=ctx["X0"]["corpus"][tie].tolist()), corpus_canal_ff=_obj(stiffness=[0.0] * 4))
    return ctx, rows


def test_corpus_tie_follow_one_body_zero_force():
    """corpus_tie_follow true: a corpus that sits on its previous pose target, with its canal nodes on the tube and
    the tube moving with it, gets tie targets EQUAL to this row's target (spring force 0); the default (false) holds
    the nodes at their start-of-step positions + the lateral correction, i.e. pulls with k x the carried motion (the
    rate artifact of review HIGH-2), above the per-node bound k x cap.  The default targets equal canal_tie_step
    without carry (test_ties: bit-identical to the pre-eu2 code)."""
    ctxN, rows = _fem_ctx_one_body(dict(corpus_tie_follow=True))
    ctxO, _ = _fem_ctx_one_body()
    cN = S.HybridController(name="h", ctx=ctxN, out_dir=None)
    cO = S.HybridController(name="h", ctx=ctxO, out_dir=None)
    assert cN.corpus_tie_follow and not cO.corpus_tie_follow
    Pk = ctxN["X0"]["corpus"]
    tie = ctxN["corpus_canal"]
    p = S.corpus_tie_params(ctxO["cfg"], R_TUBE, L_IU)
    eng_ref = np.full(4, -1, int)
    worst_new, worst_old = 0.0, 0.0
    for k, r in enumerate(rows):
        X = S.corpus_pose_targets(Pk, rows[k - 1]["T_corpus"]) if k else Pk.copy()   # on the previous target
        for ctx in (ctxN, ctxO):
            ctx["nodes"]["corpus"].dofs.position.value = X.tolist()
        cN.k = cO.k = k
        cN._begin()
        cO._begin()
        X_end = S.corpus_pose_targets(Pk, r["T_corpus"])[tie]
        kN = np.asarray(ctxN["corpus_canal_ff"].stiffness.value)
        tN = np.asarray(ctxN["corpus_canal_tgt"].position.value)
        tO = np.asarray(ctxO["corpus_canal_tgt"].position.value)
        at = geom.unit(r["tube_axis"])
        ref = S.canal_tie_step(X[tie], np.asarray(r["F"], float), at, at, eng_ref, k, p, tip_s=r["tip_s"],
                               canal_s=ctxO["corpus_canal_s_eng"])
        assert np.array_equal(tO, ref["tgt"]) and np.array_equal(np.asarray(ctxO["corpus_canal_ff"].stiffness.value), ref["k"])
        if k >= 1:
            assert (kN > 0).all()
            worst_new = max(worst_new, float(np.abs(tN - X_end).max()))
            worst_old = max(worst_old, float((np.linalg.norm(tO - X_end, axis=1) * kN).max()) / 1000.0)
    assert worst_new < 1e-9, worst_new
    assert worst_old > p["k"] * p["cap"] / 1000.0, worst_old
    # the log: the follow carry and a zero net tie force when the corpus lands on its target
    X = S.corpus_pose_targets(Pk, rows[-1]["T_corpus"])
    lg = cN._corpus_log(X)
    assert lg["tie_follow"] and lg["tie_net_force_N"] < 1e-9 and lg["tie_carry_max_mm"] > 0.5
    assert "tie_follow" not in cO._corpus_log(X)
    return dict(follow_target_vs_end_max_mm=worst_new, default_max_node_force_N=round(worst_old, 4))


def test_corpus_tie_follow_misfit_lateral_only():
    """corpus_tie_follow true with the corpus OFF its previous target by e (0.3 mm along the tube + 0.2 mm radially
    out): the target is the carried node + e's lateral part pulled back onto the tube (capped), and e's AXIAL part is
    kept -- the spring re-anchors each step and never pulls an axial offset back (no axial ratchet)."""
    ctx, rows = _fem_ctx_one_body(dict(corpus_tie_follow=True))
    c = S.HybridController(name="h", ctx=ctx, out_dir=None)
    Pk = ctx["X0"]["corpus"]
    tie = ctx["corpus_canal"]
    K = 3
    for k in range(K + 1):
        X = S.corpus_pose_targets(Pk, rows[k - 1]["T_corpus"]) if k else Pk.copy()
        if k == K:                                    # the misfit, in the frame of THIS row's tube line
            a = geom.unit(rows[k]["tube_axis"])
            X_end = S.corpus_pose_targets(Pk, rows[k]["T_corpus"])[tie]
            q = X_end - np.asarray(rows[k]["F"], float)
            radial = q - np.outer(q @ a, a)
            radial /= np.linalg.norm(radial, axis=1)[:, None]
            e = 0.3 * a + 0.2 * radial
            X[tie] += e
        ctx["nodes"]["corpus"].dofs.position.value = X.tolist()
        c.k = k
        c._begin()
    t = np.asarray(ctx["corpus_canal_tgt"].position.value)
    dev = t - X_end                                                   # what the spring would hold on top of the one body
    assert np.abs(dev - 0.3 * a).max() < 1e-9, np.abs(dev - 0.3 * a).max()   # axial kept, radial 0.2 mm corrected
    return dict(axial_kept_mm=round(float((dev @ a).mean()), 4))


def test_corpus_tie_follow_with_stretch():
    """corpus_tie_follow true with a calibrated stretch (corpus_pose_stretch_lam 1.05, ramped by w_r): the carry is the
    change of the STRETCHED pose target at the tie nodes (not the rigid increment), and a corpus sitting on its previous
    stretched target gets tie targets = canal_tie_step at this row's stretched target -- the elongation along the tube
    is carried (kept), only the lateral misfit to the tube is corrected (capped)."""
    ctx, rows = _fem_ctx_one_body(dict(corpus_tie_follow=True, corpus_pose_stretch_lam=1.05))
    for i, r in enumerate(rows):
        r["w_r"] = i / float(len(rows) - 1)                        # the stretch grows 1.0 -> 1.05 over the rows
    c = S.HybridController(name="h", ctx=ctx, out_dir=None)
    Pk = ctx["X0"]["corpus"]
    tie = ctx["corpus_canal"]
    p = S.corpus_tie_params(ctx["cfg"], R_TUBE, L_IU)
    eng = np.full(len(tie), -1, int)
    tg = [S.corpus_pose_targets(Pk, r["T_corpus"], S.corpus_stretch_at(1.05, r)) for r in rows]
    worst, stretch_part = 0.0, 0.0
    for k, r in enumerate(rows):
        X = tg[k - 1] if k else Pk.copy()                          # on the previous (stretched) target
        ctx["nodes"]["corpus"].dofs.position.value = X.tolist()
        c.k = k
        c._begin()
        want = tg[k][tie] - (tg[k - 1] if k else Pk)[tie]
        carry = c.corpus_tie["carry"]
        assert np.array_equal(carry, want), k
        if k:
            rigid = S.rigid_step_increment(X[tie], rows[k - 1]["T_corpus"], r["T_corpus"])
            stretch_part = max(stretch_part, float(np.abs(carry - rigid).max()))
        at = geom.unit(r["tube_axis"])
        ref = S.canal_tie_step(tg[k][tie], np.asarray(r["F"], float), at, at, eng, k, p, tip_s=r["tip_s"],
                               canal_s=ctx["corpus_canal_s_eng"])
        t = np.asarray(ctx["corpus_canal_tgt"].position.value)
        worst = max(worst, float(np.abs(t - ref["tgt"]).max()))
        assert np.abs((t - tg[k][tie]) @ at).max() < 1e-9            # nothing pulled back along the tube
    assert worst < 1e-9, worst
    assert stretch_part > 0.01, stretch_part                          # the stretch really is in the carry
    return dict(target_vs_ref_max_mm=worst, stretch_part_of_carry_max_mm=round(stretch_part, 4))


def test_corpus_model_values():
    assert S.corpus_model({}) == "rigid" and S.corpus_model(dict(corpus_model="fem")) == "fem"
    try:
        S.corpus_model(dict(corpus_model="elastic"))
        raise AssertionError("an unknown corpus_model must be refused")
    except ValueError:
        pass
    d = S.load_cfg({})
    assert d["corpus_model"] == "rigid" and d["attach_target"] == "mapped" and d["corpus_pose_stretch_lam"] == 1.0


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            out = fn()
            print("PASS %-50s %s" % (name, "" if out is None else out))
        except Exception as e:                           # report every test, then fail
            bad += 1
            import traceback
            print("FAIL %-50s %r" % (name, e))
            traceback.print_exc()
    print("%d/%d passed" % (len(tests) - bad, len(tests)))
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
