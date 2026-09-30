"""Host unit tests (python 3.13, numpy; scipy for the sheet test) of the S1 canal tie and the S2 probe helpers in
scene_hybrid.py.  SOFA is not importable on the host, so it is stubbed: scene_hybrid needs only Sofa.Core.Controller
as a base class, and the controller is driven here with stand-in Data objects (`.value`), exactly as SOFA would.

    python -P -B hybrid/test_ties.py          # runs every test_* below (pytest-style; pytest is not on this host)

What is tested (fix plan logs/audit_G32/fix_plan_G32_audit.md, S1 "Accept"):
  S1a  on a synthetic row whose tube axis is 30 deg from the final axis, every active tie target lies on THAT row's
       tube line (at the tube radius, same axial level) within 0.01 mm, and canal_d_mm is the distance to that line;
       the below-flange ties aim at the row's rod line.  The same row under the pre-S1 "final" axis is off by > 10 mm
       (the bug the fix removes).
  S1b  depth engagement: a node engages at the first step whose row carries tip_s >= its canal_s -- within one step
       of the tip passing it, never before -- and its spring then ramps over tie_ramp_steps.
  G32  the default cfg reproduces the pre-S1 tie block bit for bit (the block is kept verbatim below as
       _pre_s1_tie_block), over random states in every mode G16-G32 used.
  S2   subdivide_tris / apply_subdiv keep the surface (midpoints, area, orientation); sheet_signed_distance on a
       cylinder of known radius.
  eu2  canal_tie_follow (review HIGH-2): carry None / zero reproduces the pre-eu2 canal_tie_step (kept verbatim below
       as _pre_eu2_canal_tie_step) bit for bit; tie nodes on the tube carried rigidly with the tube and the corpus get
       zero spring force with "corpus" and "tube" and k x the carried motion without; the increments ("tube" about the
       row's own axis, also under canal_tie_axis "final"); the controller.
  eu3  (review eu2 HIGH-1, LOW-6, MEDIUM-5): the cfg guard (a follow needs canal_tie and the lag-free springs +
       predicted attach, "tube" no below-flange ties, bools are bools, no silent no-ops; build_scene refuses before
       touching the graph); why: an interface one or two rows behind the carried ties shears the junction; the carry
       over B / V / C with a rate jump / L / H rows (exact increment at every boundary, zero at rest); carry=c is
       canal_tie_step at X + c in every mode, dilate never pulls in; tie_force_log and the end-of-step force row.
  TF4  the rectum's lateral-only support (cfg k_rectum_lateral_mN_per_mm): the target rule (x_rest, y_now, z_now),
       the key's values, the node set (surface minus junction), the force log, _add_supports building NOTHING with
       the key None / 0 and only the target MO + one spring field with it on, and the controller's per-step write.
All geometry is synthetic (no patient or case-derived coordinates, axes or device dimensions); only
test_s1b_depth_engagement reads this machine's canal_path_s when the meshes exist.
"""
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

TOL = 0.01                                              # mm, the plan's tolerance


# ----------------------------------------------------------------------------- the pre-S1 code, verbatim
def _pre_s1_tie_block(X, F, tgt_tube_axis, tgt_axis, eng_step, k, cfg, r_tube, L_iu):
    """HybridController._begin section (3) as it was before S1 (git c2922e2, lines 1463-1511), with self.* made
    arguments.  Kept verbatim as the reference the default path must reproduce."""
    at = tgt_tube_axis
    rad = r_tube + float(cfg["canal_slack_mm"])
    q = X - F
    s = q @ at
    lat = q - np.outer(s, at)
    d = np.linalg.norm(lat, axis=1)
    inside = (s >= 0.0) & (s <= L_iu)
    centre = cfg.get("canal_tie_mode", "dilate") == "centre"
    below = float(cfg.get("canal_tie_below_mm", 0.0) or 0.0)
    if centre and below > 0.0:
        ar = np.asarray(tgt_axis, float)
        sr = q @ ar
        latr = q - np.outer(sr, ar)
        dr = np.linalg.norm(latr, axis=1)
        low = (sr < 0.0) & (sr >= -below)
        inside = inside | low
        lat = np.where(low[:, None], latr, lat)
        d = np.where(low, dr, d)
        s = np.where(low, sr, s)
    if centre:
        new = inside & (d < rad + float(cfg.get("canal_engage_mm", 3.0))) & (eng_step < 0)
    else:
        new = inside & (eng_step < 0)
    eng_step[new] = k
    act = (inside & (eng_step >= 0)) if centre else (inside & (d < rad))
    u = np.where(d[:, None] > 1e-9, lat / np.maximum(d, 1e-9)[:, None], S._perp(at)[None, :])
    cap = float(cfg["canal_max_offset_mm"])
    off = np.clip(rad - d, -cap, cap) if centre else np.maximum(np.minimum(rad - d, cap), 0.0)
    tgt = np.where(act[:, None], X + off[:, None] * u, X)
    ramp = np.clip((k - eng_step + 1) / float(cfg["tie_ramp_steps"]), 0.0, 1.0)
    kk = np.where(act, float(cfg["k_canal_mN_per_mm"]) * ramp, 0.0)
    canal_d = (dict(n_in_span=int(inside.sum()), min=round(float(d[inside].min()), 3),
                    median=round(float(np.median(d[inside])), 3), max=round(float(d[inside].max()), 3))
               if inside.any() else dict(n_in_span=0))
    return tgt, kk, int(act.sum()), canal_d


# ----------------------------------------------------------------------------- a controller on stand-in Data
class _Data(object):
    def __init__(self, v):
        self.value = v


def _obj(**data):
    return types.SimpleNamespace(**{k: _Data(v) for k, v in data.items()})


# Neutral synthetic device and directions (no patient- or case-derived numbers): a tube of radius 2.5 mm and 60 mm
# above the flange, its vaginal rod 30 deg off the tube; generic unit axes (the "final" and "path" axes 30 deg apart).
R_TUBE, L_IU = 2.5, 60.0
APP = dict(params=dict(L_iu_mm=dict(value=L_IU), r_tandem_mm=dict(value=R_TUBE)),
           landmarks=dict(shaft_end_dir=[0.0, 0.5, -0.8660254037844386]))


def _frame(a, xref=(1.0, 0.0, 0.0)):
    return geom.frame_from(a, xref)                     # rows x, y, z; z = a


def _rot(axis, deg):
    return S.rodrigues(axis, deg)


def make_controller(cfg_over, rows, Xc, canal_idx, canal_s=None, tgt=None):
    """A HybridController on a fake ctx holding only what _begin needs for the canal tie: kinematic nodes with a
    `rig`, the cervix dofs, the tie's target MO and spring field."""
    cfg = S.load_cfg(dict(cfg_over))
    n = len(canal_idx)
    if tgt is None:                                      # the final pose = the rows' own axis (no turn)
        a = geom.unit(rows[-1]["tube_axis"])
        tgt = dict(tube_axis=a, axis=a, R_rows=_frame(a))
    ctx = dict(cfg=cfg, sched=rows, tgt=tgt, inp=dict(app=APP), X0=dict(cervix=Xc.copy()), deformable=["cervix"],
               canal=np.asarray(canal_idx, int), canal_s=(None if canal_s is None else np.asarray(canal_s, float)),
               nodes=dict(cervix=types.SimpleNamespace(dofs=_obj(position=Xc.tolist(),
                                                                 velocity=np.zeros_like(Xc).tolist()))),
               corpus=types.SimpleNamespace(rig=_obj(position=None)),
               tandem=types.SimpleNamespace(rig=_obj(position=None)),
               ovoids=types.SimpleNamespace(rig=_obj(position=None)),
               canal_tgt=_obj(position=Xc[canal_idx].tolist()), canal_ff=_obj(stiffness=[0.0] * n))
    return S.HybridController(name="h", ctx=ctx, out_dir=None), ctx


def _row(F, a, phase="T", tip_s=None, x_ref=(1.0, 0.0, 0.0)):
    R = _frame(a, x_ref)
    r = dict(phase=phase, u=0.5, s=0.0, F=np.asarray(F, float), tube_axis=np.asarray(a, float), R_rows=R,
             T_corpus=np.eye(4), ov_pos=np.asarray(F, float) - 100.0 * np.asarray(a, float), ov_lag=100.0)
    if tip_s is not None:
        r["tip_s"] = float(tip_s)
    return r


def _line_dist(P, F, a):
    q = np.asarray(P, float) - F
    s = q @ a
    return np.linalg.norm(q - np.outer(s, a), axis=1), s


# ----------------------------------------------------------------------------- S1a
def _axes():
    """(a_fin, a_path): two generic unit axes 30 deg apart (synthetic)."""
    a_fin = geom.unit([0.3, -0.2, 0.93])
    return a_fin, _rot(geom.ortho([1.0, 1.0, 0.0], a_fin), 30.0) @ a_fin


def _s1a_setup(seed=0):
    rng = np.random.default_rng(seed)
    a_fin, a_path = _axes()                              # a generic tube axis (any unit vector would do)
    a_row = _rot(geom.ortho([1.0, 0.0, 0.0], a_fin), 30.0) @ a_fin
    F = np.array([10.0, 20.0, 0.0])       # synthetic origin (not a patient coordinate)
    n = 40
    s = rng.uniform(1.0, 60.0, n)
    e1 = geom.ortho([1.0, 0.0, 0.0], a_row)
    e2 = np.cross(a_row, e1)
    th = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.2, 5.0, n)
    X = F + np.outer(s, a_row) + (r * np.cos(th))[:, None] * e1 + (r * np.sin(th))[:, None] * e2
    tgt = dict(tube_axis=a_fin, axis=a_path, R_rows=_frame(a_fin))
    return a_fin, a_row, F, X, tgt


def test_s1a_targets_on_row_tube_line():
    """Every active tie target on the row's tube line (tube radius, same level) within 0.01 mm; canal_d = distance to
    that line.  cap 100 mm, so a target is the full radial move (no per-step bound)."""
    a_fin, a_row, F, X, tgt = _s1a_setup()
    assert abs(geom.angle_deg(a_fin, a_row) - 30.0) < 1e-9
    rows = [_row(F, a_row) for _ in range(4)]
    ctrl, ctx = make_controller(dict(canal_tie_axis="row", canal_tie_mode="centre", canal_engage_mm=100.0,
                                     canal_max_offset_mm=100.0), rows, X, np.arange(len(X)), tgt=tgt)
    for k in range(4):                                   # 3-step ramp: full stiffness from the 3rd step
        ctrl.k = k
        ctrl._begin()
    T = np.array(ctx["canal_tgt"].position.value)
    kk = np.array(ctx["canal_ff"].stiffness.value)
    act = kk > 0
    assert act.all(), "every node is in the tube's span and engaged"
    dT, sT = _line_dist(T, F, a_row)
    dX, sX = _line_dist(X, F, a_row)
    assert np.abs(dT[act] - R_TUBE).max() <= TOL, np.abs(dT[act] - R_TUBE).max()
    assert np.abs(sT - sX).max() <= TOL                  # lateral only: the node's axial level is kept
    assert np.allclose(kk, float(ctx["cfg"]["k_canal_mN_per_mm"]))
    cd = ctrl.canal_d
    assert cd["n_in_span"] == len(X)
    for key, v in (("min", dX.min()), ("median", np.median(dX)), ("max", dX.max())):
        assert abs(cd[key] - round(float(v), 3)) < 1e-9, (key, cd[key], v)
    return dict(max_target_off_line_mm=float(np.abs(dT - R_TUBE).max()))


def test_s1a_default_cap_moves_radially_about_row_line():
    """With the default 0.5 mm cap the target is a radial step (about the ROW line) of at most 0.5 mm."""
    a_fin, a_row, F, X, tgt = _s1a_setup(1)
    rows = [_row(F, a_row)]
    ctrl, ctx = make_controller(dict(canal_tie_axis="row", canal_tie_mode="centre", canal_engage_mm=100.0),
                                rows, X, np.arange(len(X)), tgt=tgt)
    ctrl._begin()
    T = np.array(ctx["canal_tgt"].position.value)
    D = T - X
    dX, sX = _line_dist(X, F, a_row)
    radial = (X - F) - np.outer(sX, a_row)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    along = np.einsum("ij,ij->i", D, radial)
    assert np.linalg.norm(D - along[:, None] * radial, axis=1).max() <= TOL
    assert np.abs(along).max() <= 0.5 + 1e-9
    assert np.all(np.sign(along[np.abs(dX - R_TUBE) > 1e-6]) == np.sign((R_TUBE - dX)[np.abs(dX - R_TUBE) > 1e-6]))


def test_s1a_final_axis_is_the_bug():
    """The same row under canal_tie_axis "final" (the pre-S1 code) aims > 10 mm off the row's tube line."""
    a_fin, a_row, F, X, tgt = _s1a_setup()
    rows = [_row(F, a_row) for _ in range(4)]
    ctrl, ctx = make_controller(dict(canal_tie_axis="final", canal_tie_mode="centre", canal_engage_mm=100.0,
                                     canal_max_offset_mm=100.0), rows, X, np.arange(len(X)), tgt=tgt)
    for k in range(4):
        ctrl.k = k
        ctrl._begin()
    T = np.array(ctx["canal_tgt"].position.value)
    act = np.array(ctx["canal_ff"].stiffness.value) > 0
    dT, _ = _line_dist(T[act], F, a_row)
    err = float(np.abs(dT - R_TUBE).max())
    assert err > 10.0, err
    return dict(final_axis_target_error_max_mm=err, n_active=int(act.sum()))


def test_s1a_below_flange_row_rod_line():
    """canal_tie_below_mm > 0 with "row": nodes below the flange are tied radially about the ROW's rod line
    (applicator rod direction carried by the row's R_rows), not about tgt axis."""
    a_fin, a_row, F, _, tgt = _s1a_setup(2)
    R_row = _frame(a_row)
    ar = geom.unit(S.rod_dir_app(APP) @ R_row)
    rng = np.random.default_rng(3)
    n = 12
    s = -rng.uniform(0.5, 9.5, n)                        # below the flange, within canal_tie_below_mm = 10
    e1 = geom.ortho([0.0, 1.0, 0.0], ar)
    e2 = np.cross(ar, e1)
    th = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.3, 4.0, n)
    X = F + np.outer(s, ar) + (r * np.cos(th))[:, None] * e1 + (r * np.sin(th))[:, None] * e2
    row = _row(F, a_row)
    row["R_rows"] = R_row
    ctrl, ctx = make_controller(dict(canal_tie_axis="row", canal_tie_mode="centre", canal_engage_mm=100.0,
                                     canal_max_offset_mm=100.0, canal_tie_below_mm=10.0), [row] * 4, X,
                                np.arange(n), tgt=tgt)
    for k in range(4):
        ctrl.k = k
        ctrl._begin()
    T = np.array(ctx["canal_tgt"].position.value)
    act = np.array(ctx["canal_ff"].stiffness.value) > 0
    dT, sT = _line_dist(T, F, ar)
    _, sX = _line_dist(X, F, ar)
    assert act.all()
    assert np.abs(dT - R_TUBE).max() <= TOL and np.abs(sT - sX).max() <= TOL
    at2, ar2 = S.tie_axes(ctx["cfg"], tgt, row, S.rod_dir_app(APP))
    assert np.allclose(at2, a_row) and np.allclose(ar2, ar)


# ----------------------------------------------------------------------------- S1b
def _depth_setup(canal_s):
    a = geom.unit([0.3, -0.25, 0.92])                                 # any unit axis (synthetic)
    F = np.array([0.0, 0.0, 0.0])
    n = len(canal_s)
    e1 = geom.ortho([1.0, 0.0, 0.0], a)
    X = F + np.outer(20.0 + np.asarray(canal_s), a) + 1.0 * e1       # inside the tube's span, 1 mm off its axis
    return a, F, X


def _canal_path_s():
    """The real canal_path_s from the cervix meta when this machine has it (S3), else a synthetic spread."""
    try:
        P = S.paths()
        import json
        ns = json.load(open(P["meshes"] + "/cervix/meta.json"))["node_sets"]
        return np.asarray(ns["canal_path_s"], float), "meshes/cervix/meta.json canal_path_s"
    except Exception:
        return np.linspace(-15.0, 10.0, 39), "synthetic"


def test_s1b_depth_engagement():
    """Node i engages at the FIRST step whose row carries tip_s >= canal_s[i] (never before, never later), rows
    without tip_s (B/P) engage nothing, and the spring ramps 1/3, 2/3, 1 of k_canal from the engagement step."""
    canal_s, src = _canal_path_s()
    a, F, X = _depth_setup(canal_s)
    rows = [_row(F, a, phase="B") for _ in range(5)]                          # no tip_s: never engages
    tip = -30.0 + 0.8 * np.arange(120)                                         # tandem-first V/C tip travel
    rows += [_row(F, a, phase="V" if t < 0 else "C", tip_s=t) for t in tip]
    ctrl, ctx = make_controller(dict(canal_tie_axis="row", canal_tie_mode="centre", canal_engage="depth",
                                     canal_tie_set="canal_path"), rows, X, np.arange(len(X)), canal_s=canal_s)
    kc = float(ctx["cfg"]["k_canal_mN_per_mm"])
    tips = [r.get("tip_s") for r in rows]
    K = []
    for k in range(len(rows)):
        ctrl.k = k
        ctrl._begin()
        K.append(np.array(ctx["canal_ff"].stiffness.value))
    K = np.array(K)
    for i, cs in enumerate(canal_s):
        first = next(k for k, t in enumerate(tips) if t is not None and t >= cs)
        assert ctrl.eng_step[i] == first, (i, cs, ctrl.eng_step[i], first)
        assert np.all(K[:first, i] == 0.0), "engaged before the tip passed"
        want = kc * np.clip((np.arange(first, len(rows)) - first + 1) / 3.0, 0.0, 1.0)
        assert np.allclose(K[first:, i], want)
    assert np.all(K[:5] == 0.0)
    lag = [ctrl.eng_step[i] - next(k for k, t in enumerate(tips) if t is not None and t >= cs)
           for i, cs in enumerate(canal_s)]
    return dict(canal_s_source=src, n=len(canal_s), max_engage_lag_steps=int(max(lag)),
                first_engage_step=int(ctrl.eng_step.min()), last_engage_step=int(ctrl.eng_step.max()))


def test_s1b_radius_would_engage_early():
    """Contrast: the same nodes under radius engagement all engage at the first step the tube is there (they sit
    1 mm off its axis) -- regardless of where the tip is.  That is what depth engagement replaces."""
    canal_s, _ = _canal_path_s()
    a, F, X = _depth_setup(canal_s)
    rows = [_row(F, a, tip_s=-30.0)]
    ctrl, _ = make_controller(dict(canal_tie_mode="centre", canal_engage_mm=12.0), rows, X, np.arange(len(X)))
    ctrl._begin()
    assert np.all(ctrl.eng_step == 0)


# ----------------------------------------------------------------------------- the default path == pre-S1 code
def test_default_path_bit_identical_to_pre_s1():
    """canal_tie_step with the default cfg (via tie_axes) reproduces _pre_s1_tie_block exactly -- targets,
    stiffnesses, engagement steps, active count and canal_d -- over random multi-step states in every tie mode the
    G16-G32 cfgs used (dilate; centre; centre + canal_tie_below_mm 10 + canal_engage_mm 12 = G32), on rows whose
    axis turns (so the "final" axis differs from the row's, as in G32's canal schedule)."""
    rng = np.random.default_rng(7)
    a_fin, a_path = _axes()
    tgt = dict(tube_axis=a_fin, axis=a_path)
    modes = [dict(), dict(canal_tie_mode="centre"),
             dict(canal_tie_mode="centre", canal_engage_mm=12.0, canal_tie_below_mm=10.0)]
    n_cmp = 0
    for m in modes:
        cfg = S.load_cfg(m)
        p = S.canal_tie_params(cfg, R_TUBE, L_IU)
        eng_new = np.full(38, -1, int)
        eng_old = np.full(38, -1, int)
        for k in range(60):
            w = k / 59.0
            a_row = geom.slerp(a_path, a_fin, w)
            F = np.array([5.0, -5.0, 10.0]) - (1 - w) * 80.0 * a_path   # synthetic
            row = dict(tube_axis=a_row, R_rows=_frame(a_row))
            X = F + np.outer(rng.uniform(-15, 70, 38), a_row) + rng.normal(0, 6.0 * (1 - w) + 1.0, (38, 3))
            at, ar = S.tie_axes(cfg, tgt, row, S.rod_dir_app(APP))
            res = S.canal_tie_step(X, F, at, ar, eng_new, k, p)
            t_old, k_old, n_old, cd_old = _pre_s1_tie_block(X, F, tgt["tube_axis"], tgt["axis"], eng_old, k, cfg,
                                                            R_TUBE, L_IU)
            assert np.array_equal(res["tgt"], t_old) and np.array_equal(res["k"], k_old)
            assert np.array_equal(eng_new, eng_old) and int(res["act"].sum()) == n_old and res["canal_d"] == cd_old
            n_cmp += 1
    return dict(steps_compared=n_cmp)


# ----------------------------------------------------------------------------- eu2: the tie target carried with the tissue
def _pre_eu2_canal_tie_step(X, F, at, ar, eng_step, k, p, tip_s=None, canal_s=None):
    """scene_hybrid.canal_tie_step as it was before eu2 (working tree of 2026-09-27, after b3745b1 + S9), verbatim:
    the reference the default (carry None) path must reproduce bit for bit."""
    q = X - F
    s = q @ at
    lat = q - np.outer(s, at)
    d = np.linalg.norm(lat, axis=1)
    inside = (s >= 0.0) & (s <= p["L_iu"])
    centre, below = p["centre"], p["below"]
    if centre and below > 0.0:
        sr = q @ ar
        latr = q - np.outer(sr, ar)
        dr = np.linalg.norm(latr, axis=1)
        low = (sr < 0.0) & (sr >= -below)
        inside = inside | low
        lat = np.where(low[:, None], latr, lat)
        d = np.where(low, dr, d)
        s = np.where(low, sr, s)
    if centre and p["engage"] == "depth":
        reached = (np.asarray(canal_s, float) <= float(tip_s)) if tip_s is not None else np.zeros(len(X), bool)
        new = reached & (eng_step < 0)
    elif centre:
        new = inside & (d < p["rad"] + p["engage_mm"]) & (eng_step < 0)
    else:
        new = inside & (eng_step < 0)
    eng_step[new] = k
    act = (inside & (eng_step >= 0)) if centre else (inside & (d < p["rad"]))
    u = np.where(d[:, None] > 1e-9, lat / np.maximum(d, 1e-9)[:, None], S._perp(at)[None, :])
    cap = p["cap"]
    off = np.clip(p["rad"] - d, -cap, cap) if centre else np.maximum(np.minimum(p["rad"] - d, cap), 0.0)
    tgt = np.where(act[:, None], X + off[:, None] * u, X)
    ramp = np.clip((k - eng_step + 1) / p["ramp"], 0.0, 1.0)
    kk = np.where(act, p["k"] * ramp, 0.0)
    canal_d = (dict(n_in_span=int(inside.sum()), min=round(float(d[inside].min()), 3),
                    median=round(float(np.median(d[inside])), 3), max=round(float(d[inside].max()), 3))
               if inside.any() else dict(n_in_span=0))
    return dict(tgt=tgt, k=kk, act=act, inside=inside, d=d, canal_d=canal_d)


def test_eu2_follow_off_bit_identical():
    """canal_tie_step with carry None -- and with an all-zero carry -- reproduces the pre-eu2 function bit for bit
    (targets, stiffnesses, act / inside / d, canal_d, engagement) over random multi-step states in dilate, centre,
    centre + below-flange (G32) and centre + depth engagement (TF0c)."""
    rng = np.random.default_rng(23)
    a_fin, a_path = _axes()
    modes = [dict(), dict(canal_tie_mode="centre"),
             dict(canal_tie_mode="centre", canal_engage_mm=12.0, canal_tie_below_mm=10.0),
             dict(canal_tie_mode="centre", canal_engage="depth", k_canal_mN_per_mm=400.0)]
    n_cmp = 0
    for m in modes:
        p = S.canal_tie_params(S.load_cfg(m), R_TUBE, L_IU)
        cs = rng.uniform(-15.0, 10.0, 38)
        engs = [np.full(38, -1, int) for _ in range(3)]
        for k in range(50):
            w = k / 49.0
            at = geom.slerp(a_path, a_fin, w)
            F = np.array([5.0, -5.0, 10.0]) - (1 - w) * 80.0 * a_path     # synthetic
            X = F + np.outer(rng.uniform(-15, 70, 38), at) + rng.normal(0, 4.0, (38, 3))
            tip = -20.0 + 35.0 * w
            ref = _pre_eu2_canal_tie_step(X, F, at, a_path, engs[0], k, p, tip_s=tip, canal_s=cs)
            for eng, carry in ((engs[1], None), (engs[2], np.zeros_like(X))):
                res = S.canal_tie_step(X, F, at, a_path, eng, k, p, tip_s=tip, canal_s=cs, carry=carry)
                for key in ("tgt", "k", "act", "inside", "d"):
                    assert np.array_equal(res[key], ref[key]), (m, k, key)
                assert res["canal_d"] == ref["canal_d"] and np.array_equal(eng, engs[0])
                n_cmp += 1
    return dict(calls_compared=n_cmp)


def _eu2_rigid_rows(seed, n_steps=5, deg=0.8, mm=0.5):
    """Rows whose corpus AND device move together by a small rigid motion each step (as the L-phase lift): T_corpus
    = Q_k x + t_k; the device frame R_rows_k = R0 @ Q_k^T, F_k = F0 @ Q_k^T + t_k (the rows convention x = x_app @ R +
    F), tube_axis = R_rows_k[2].  Returns rows, the rest tube (F0, a0) and the per-step transforms."""
    rng = np.random.default_rng(seed)
    a0 = geom.unit([-0.2, 0.3, 0.93])                                 # synthetic
    R0 = _frame(a0)
    F0 = np.array([4.0, -7.0, 2.0])                                   # synthetic
    rows, Ts = [], []
    Q, t = np.eye(3), np.zeros(3)
    for _ in range(n_steps):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Q, t
        Rk = R0 @ Q.T
        rows.append(dict(phase="L", u=1.0, s=1.0, F=F0 @ Q.T + t, tube_axis=Rk[2], R_rows=Rk, T_corpus=T.copy(),
                         ov_pos=F0 - 100.0 * a0, ov_lag=100.0, tip_s=40.0))
        Ts.append(T.copy())
        Q = _rot(rng.normal(size=3), deg) @ Q
        t = t @ _rot(rng.normal(size=3), deg).T + mm * geom.unit(rng.normal(size=3)) + 0.4 * a0
    return rows, F0, a0, Ts


def _on_tube(F0, a0, n=24, seed=5):
    rng = np.random.default_rng(seed)
    e1 = geom.ortho([1.0, 0.0, 0.0], a0)
    e2 = np.cross(a0, e1)
    th = rng.uniform(0, 2 * np.pi, n)
    s = rng.uniform(5.0, 55.0, n)
    return F0 + np.outer(s, a0) + R_TUBE * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


FOLLOW_OK = dict(attach_impl="springs", attach_target="predicted")   # the lag-free attach canal_tie_follow requires


def test_eu2_follow_rigid_motion_zero_force():
    """Tie nodes EXACTLY on the tube surface, carried rigidly with the tube and the corpus (one body, as in L): with
    canal_tie_follow "corpus" or "tube" every target equals the node's end-of-step position (spring force 0 to 1e-9
    N); the old start-of-step target (carry None) pulls with k x the step's carried motion -- the rate artifact of
    review HIGH-2 -- above the cap bound k x cap when the motion per step exceeds the cap."""
    cfg = S.load_cfg(dict(canal_tie_mode="centre", k_canal_mN_per_mm=400.0))
    p = S.canal_tie_params(cfg, R_TUBE, L_IU)
    rows, F0, a0, _ = _eu2_rigid_rows(3)
    Xr = _on_tube(F0, a0)
    out = {}
    for mode in (None, "corpus", "tube"):
        eng = np.full(len(Xr), -1, int)
        fmax = 0.0
        for k in range(1, len(rows)):
            ra, rb = rows[k - 1], rows[k]
            X = S.corpus_pose_targets(Xr, ra["T_corpus"])              # end of the previous step: carried rigidly
            X_end = S.corpus_pose_targets(Xr, rb["T_corpus"])          # where the one body puts them this step
            at = geom.unit(rb["tube_axis"])
            carry = S.canal_tie_carry(mode, X, ra, rb)
            res = S.canal_tie_step(X, np.asarray(rb["F"], float), at, at, eng, k, p, carry=carry)
            assert res["act"].all()
            f = res["k"][:, None] * (res["tgt"] - X_end)               # mN
            fmax = max(fmax, float(np.linalg.norm(f, axis=1).max()) / 1000.0)
            if mode is not None:
                assert np.abs(res["tgt"] - X_end).max() < 1e-9, (mode, k, np.abs(res["tgt"] - X_end).max())
                assert np.abs(res["d"] - p["rad"]).max() < 1e-9               # measured at the carried position
        out[str(mode)] = round(fmax, 6)
    assert out["corpus"] < 1e-9 and out["tube"] < 1e-9
    assert out["None"] > p["k"] * p["cap"] / 1000.0, out            # old: above the k x cap per-node bound
    return dict(max_node_force_N=out)


def test_eu2_carry_increments():
    """rigid_step_increment is T_now o T_prev^-1 (exactly 0 for equal placements); tube_lateral_increment has no
    component along the line and is exactly 0 for an unchanged device; canal_tie_carry: None when off, 0 on the first
    step, "tube" = the tube's lateral motion about the ROW's tube axis + the corpus's part along it -- also when the
    tie itself aims at the "final" axis (review LOW-6: the slide removed is the tube's own, not the tie line's)."""
    rng = np.random.default_rng(9)
    X = rng.normal(0, 20, (30, 3))
    rows, _, _, Ts = _eu2_rigid_rows(4, n_steps=3, deg=3.0, mm=2.0)
    D = S.rigid_step_increment(X, Ts[1], Ts[2])
    X0 = (X - Ts[1][:3, 3]) @ Ts[1][:3, :3]                           # X = X0 @ Q1^T + t1
    assert np.abs(X + D - S.corpus_pose_targets(X0, Ts[2])).max() < 1e-9
    assert np.array_equal(S.rigid_step_increment(X, Ts[2], Ts[2].copy()), np.zeros_like(X))
    a = geom.unit(rows[2]["tube_axis"])
    Dt = S.tube_lateral_increment(X, rows[1]["R_rows"], rows[1]["F"], rows[2]["R_rows"], rows[2]["F"], a)
    assert np.abs(Dt @ a).max() < 1e-9
    assert np.array_equal(S.tube_lateral_increment(X, rows[2]["R_rows"], rows[2]["F"], rows[2]["R_rows"],
                                                   rows[2]["F"], a), np.zeros_like(X))
    # a device moving differently from the corpus (C: the tube turns in the uterus): "tube" != "corpus"
    Rb = _rot([1.0, 0.0, 0.0], 2.0) @ rows[2]["R_rows"]
    rb = dict(rows[2], R_rows=Rb, tube_axis=Rb[2])
    ab = geom.unit(rb["tube_axis"])
    ct = S.canal_tie_carry("tube", X, rows[1], rb)
    cc = S.canal_tie_carry("corpus", X, rows[1], rb)
    Dt2 = S.tube_lateral_increment(X, rows[1]["R_rows"], rows[1]["F"], rb["R_rows"], rb["F"], ab)
    assert np.abs(ct - (Dt2 + np.outer(cc @ ab, ab))).max() < 1e-12 and np.abs(ct - cc).max() > 0.1
    assert np.abs(ct @ ab - cc @ ab).max() < 1e-9                      # along the ROW axis: the corpus's part only
    assert S.canal_tie_carry(None, X, rows[1], rows[2]) is None
    assert np.array_equal(S.canal_tie_carry("corpus", X, None, rows[2]), np.zeros_like(X))
    # the controller under canal_tie_axis "final" (tie line = tgt tube_axis, 30 deg off the rows): the "tube" carry is
    # still the one about the row's own axis
    a_fin, _ = _axes()
    Xr = _on_tube(rows[0]["F"], geom.unit(rows[0]["tube_axis"]), n=12, seed=4)
    tgt = dict(tube_axis=a_fin, axis=a_fin, R_rows=_frame(a_fin))
    rws = [rows[0], rows[1], rb]
    ctrl, ctx = make_controller(dict(FOLLOW_OK, canal_tie_follow="tube", canal_tie_mode="centre"), rws, Xr,
                                np.arange(len(Xr)), tgt=tgt)
    for k in range(3):
        Xk = S.corpus_pose_targets(Xr, rws[max(k - 1, 0)]["T_corpus"])
        ctx["nodes"]["cervix"].dofs.position.value = Xk.tolist()
        ctrl.k = k
        ctrl._begin()
        want = S.canal_tie_carry("tube", Xk, rws[k - 1] if k else None, rws[k])
        assert np.array_equal(ctrl.canal_tie_last["carry"], want)
    return dict(tube_minus_corpus_max_mm=round(float(np.abs(ct - cc).max()), 4))


def test_eu3_follow_cfg_guard():
    """Review eu2 HIGH-1 / LOW-6: canal_tie_follow is refused unless the attach is lag-free (springs + predicted) and
    canal_tie is on; "tube" is refused with below-flange ties; the off values stay off whatever the attach; non-bool
    corpus_tie_follow / log_canal_tie_force, a corpus_tie_follow on a rigid corpus (or without corpus ties) and a
    tie-force log without ties are refused.  build_scene refuses BEFORE touching the graph; so does the controller."""
    def refused(fn, *a):
        try:
            fn(*a)
        except ValueError:
            return True
        return False

    L = S.load_cfg
    for off in (False, None, "off"):
        assert S.canal_tie_follow_mode(L(dict(canal_tie_follow=off))) is None                 # default attach: fine
    for m in ("corpus", "tube"):
        assert S.canal_tie_follow_mode(L(dict(FOLLOW_OK, canal_tie_follow=m))) == m
        for att in (dict(), dict(attach_target="predicted"), dict(attach_impl="springs"),
                    dict(attach_impl="bilateral", corpus_model="fem")):
            assert refused(S.canal_tie_follow_mode, L(dict(att, canal_tie_follow=m))), (m, att)
        assert refused(S.canal_tie_follow_mode, L(dict(FOLLOW_OK, canal_tie_follow=m, canal_tie=False)))
    for bad in ("rigid", "Corpus", True, 1, 0, 0.0):
        assert refused(S.canal_tie_follow_mode, L(dict(FOLLOW_OK, canal_tie_follow=bad))), bad
    below = dict(FOLLOW_OK, canal_tie_mode="centre", canal_tie_below_mm=10.0)
    assert refused(S.canal_tie_follow_mode, L(dict(below, canal_tie_follow="tube")))
    assert S.canal_tie_follow_mode(L(dict(below, canal_tie_follow="corpus"))) == "corpus"
    assert S.canal_tie_follow_mode(L(dict(FOLLOW_OK, canal_tie_follow="tube", canal_tie_below_mm=10.0))) == "tube"
    # corpus_tie_follow: a bool, and only where there are corpus ties
    assert S.corpus_tie_follow_on(L(dict(corpus_model="fem", corpus_tie_follow=True))) is True
    assert S.corpus_tie_follow_on(L(dict(corpus_tie_follow=None))) is False
    assert refused(S.corpus_tie_follow_on, L(dict(corpus_tie_follow=True)))                   # rigid corpus
    assert refused(S.corpus_tie_follow_on, L(dict(corpus_model="fem", corpus_tie_follow=True, corpus_canal_tie=False)))
    for bad in (1, 0, "true"):
        assert refused(S.corpus_tie_follow_on, L(dict(corpus_model="fem", corpus_tie_follow=bad))), bad
    # log_canal_tie_force
    assert S.check_follow_cfg(L(dict(log_canal_tie_force=True))) == (None, False)
    assert refused(S.check_follow_cfg, L(dict(log_canal_tie_force=1)))
    assert refused(S.check_follow_cfg, L(dict(log_canal_tie_force=True, canal_tie=False)))

    class _Untouched(object):                            # a scene root that must never be used
        def __getattr__(self, name):
            raise AssertionError("the graph was touched (%s) before the cfg check" % name)

        def __setattr__(self, name, value):
            raise AssertionError("the graph was touched (%s) before the cfg check" % name)

    assert refused(S.build_scene, _Untouched(), L(dict(canal_tie_follow="corpus")), {"never": "read"})
    assert refused(S.build_scene, _Untouched(), L(dict(corpus_tie_follow=True)), {"never": "read"})
    rows, F0, a0, _ = _eu2_rigid_rows(2, n_steps=2)
    Xr = _on_tube(F0, a0, n=4)
    assert refused(make_controller, dict(canal_tie_follow="corpus"), rows, Xr, np.arange(4))


def _phased_rows(F0, a0, pivot):
    """A synthetic tandem-first-like schedule with every kind of boundary the carry meets: B (3 rows) and V (4 rows,
    the tube sliding in along its axis) with the corpus at rest; C (8 rows) turning corpus AND device about `pivot`,
    0.5 deg / step then 2.0 deg / step (a rate jump at row 11); L (4 rows) translating both 0.8 mm / step along the
    tube.  Rows past the end are the controller's H (copies of the last row).  Returns rows, the jump row index."""
    R0 = _frame(a0)
    rot_ax = geom.ortho([1.0, 0.0, 0.0], a0)
    rows = []

    def row(phase, T, slide):
        Q, t = T[:3, :3], T[:3, 3]
        R = R0 @ Q.T
        F = (F0 + slide * a0) @ Q.T + t
        return dict(phase=phase, u=0.5, s=0.5, F=F, tube_axis=R[2], R_rows=R, T_corpus=T.copy(),
                    ov_pos=F - 100.0 * R[2], ov_lag=100.0)

    T = np.eye(4)
    for _ in range(3):
        rows.append(row("B", T, -20.0))
    for i in range(4):
        rows.append(row("V", T, -20.0 + 5.0 * (i + 1)))
    ang = 0.0
    for i in range(8):
        ang += 0.5 if i < 4 else 2.0
        Q = _rot(rot_ax, ang)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Q, pivot - Q @ pivot                   # x -> Q (x - pivot) + pivot
        rows.append(row("C", T, 0.0))
    for _ in range(4):
        T = T.copy()
        T[:3, 3] = T[:3, 3] + 0.8 * rows[-1]["tube_axis"]
        rows.append(row("L", T, 0.0))
    return rows, 3 + 4 + 4


def test_eu3_carry_rate_jumps_and_phase_boundaries():
    """canal_tie_follow "corpus" through the controller over B / V / C (with a 4x rate jump) / L / H rows, the tie nodes
    on the tube and the tissue left each step on the previous row's pose: at EVERY step the carry is exactly
    rigid_step_increment(row k-1 -> k) -- zero on the first step, in B, V and H (bit for bit), and it takes the new rate
    AT the jump row (no smoothing, no lag); the written targets are exactly canal_tie_step at the carried position and
    equal the one-body end positions (zero force); the end-of-step force row reads 0 N with the step's carry."""
    a0 = geom.unit([0.5, -0.3, 0.81])
    F0 = np.array([2.0, 3.0, -4.0])
    rows, jump = _phased_rows(F0, a0, pivot=F0 + 25.0 * a0)
    Xr = _on_tube(F0, a0, n=16, seed=6)
    cfg = dict(FOLLOW_OK, canal_tie_follow="corpus", canal_tie_mode="centre", canal_tie_axis="row",
               k_canal_mN_per_mm=400.0)
    ctrl, ctx = make_controller(cfg, rows, Xr, np.arange(len(Xr)))
    p = S.canal_tie_params(ctx["cfg"], R_TUBE, L_IU)
    eng = np.full(len(Xr), -1, int)
    n_steps = len(rows) + 3
    cmax, prev = [], None
    for k in range(n_steps):
        rp = rows[min(k - 1, len(rows) - 1)] if k else None
        Xk = S.corpus_pose_targets(Xr, rp["T_corpus"]) if rp is not None else Xr.copy()
        ctx["nodes"]["cervix"].dofs.position.value = Xk.tolist()
        ctrl.k = k
        ctrl._begin()
        r = ctrl.cur                                                     # rows[k], or the controller's H copy
        carry = ctrl.canal_tie_last["carry"]
        want = np.zeros_like(Xk) if rp is None else S.rigid_step_increment(Xk, rp["T_corpus"], r["T_corpus"])
        assert np.array_equal(carry, want), k
        if k == 0 or r["phase"] in ("B", "V", "H"):
            assert not np.any(carry), (k, r["phase"])
        assert ctrl.row_prev is ctrl.cur and (prev is None or ctrl.cur is not prev)
        prev = ctrl.cur
        at = geom.unit(r["tube_axis"])
        ref = S.canal_tie_step(Xk + want, np.asarray(r["F"], float), at, at, eng, k, p)
        tgt = np.asarray(ctx["canal_tgt"].position.value)
        assert np.array_equal(tgt, ref["tgt"]) and np.array_equal(np.asarray(ctx["canal_ff"].stiffness.value), ref["k"])
        X_end = S.corpus_pose_targets(Xr, r["T_corpus"])
        assert np.abs(tgt - X_end).max() < 1e-9, (k, np.abs(tgt - X_end).max())
        row = ctrl._canal_tie_force_row(X_end)
        assert row["net_N"] == 0.0 and row["max_node_N"] == 0.0 and row["follow"] == "corpus"
        assert row["carry_max_mm"] == round(float(np.linalg.norm(want, axis=1).max()), 4)
        cmax.append(float(np.linalg.norm(carry, axis=1).max()))
    assert cmax[jump] > 3.0 * cmax[jump - 1] > 0.0, (cmax[jump - 1], cmax[jump])      # 0.5 -> 2.0 deg at the jump row
    assert cmax[7] > 0.0 and cmax[6] == 0.0                              # V -> C: the first C row moves the corpus
    return dict(carry_mm_before_after_jump=[round(cmax[jump - 1], 4), round(cmax[jump], 4)],
                carry_mm_L=round(cmax[len(rows) - 1], 4), carry_mm_H=cmax[-1])


def test_eu3_lagged_interface_shears_junction():
    """Why canal_tie_follow needs the lag-free attach (review eu2 HIGH-1).  Through the controller on a rigid corpus
    with attach_target "predicted": the carried tie targets and the written interface target sit on the SAME row-k pose
    at every step (tie nodes + interface nodes fit rest rigidly to 1e-9 mm: no shear).  An interface held one or two
    rows behind (what AttachConstraint + "predicted" / the default "mapped" deliver) against the same carried ties is
    sheared by the corpus's motion over those rows -- and those attach settings are refused."""
    a0 = geom.unit([0.5, -0.3, 0.81])
    F0 = np.array([2.0, 3.0, -4.0])
    rows, jump = _phased_rows(F0, a0, pivot=F0 + 25.0 * a0)
    Xt = _on_tube(F0, a0, n=12, seed=8)                                      # canal tie nodes
    e1 = geom.ortho([1.0, 0.0, 0.0], a0)
    Xi = F0 + np.outer(np.linspace(-4.0, 4.0, 9), a0) + 9.0 * e1             # interface nodes, off the tube
    Xc = np.vstack([Xt, Xi])
    tie, iface = np.arange(len(Xt)), np.arange(len(Xt), len(Xc))
    cfg = dict(FOLLOW_OK, canal_tie_follow="corpus", canal_tie_mode="centre", canal_tie_axis="row")
    ctrl, ctx = make_controller(cfg, rows, Xc, tie)
    ctx["iface"] = iface
    ctx["corpus_iface_tgt"] = _obj(position=Xc[iface].tolist())
    shear = {0: [], 1: [], 2: []}
    X0 = np.vstack([Xt, Xi])
    for k in range(len(rows)):
        Xk = S.corpus_pose_targets(Xc, rows[max(k - 1, 0)]["T_corpus"])
        ctx["nodes"]["cervix"].dofs.position.value = Xk.tolist()
        ctrl.k = k
        ctrl._begin()
        tt = np.asarray(ctx["canal_tgt"].position.value)
        zi = np.asarray(ctx["corpus_iface_tgt"].position.value)
        assert np.abs(zi - S.corpus_pose_targets(Xi, rows[k]["T_corpus"])).max() < 1e-12   # "predicted": row k
        for lag in (0, 1, 2):
            z = zi if lag == 0 else S.corpus_pose_targets(Xi, rows[max(k - lag, 0)]["T_corpus"])
            Y = np.vstack([tt, z])
            R, t = geom.kabsch(X0, Y)
            shear[lag].append(float(np.linalg.norm(X0 @ R.T + t - Y, axis=1).max()))
    assert max(shear[0]) < 1e-9, max(shear[0])
    moving = [k for k in range(1, len(rows)) if rows[k]["phase"] in ("C", "L")]
    assert all(shear[2][k] > shear[1][k] > 1e-3 for k in moving[1:]), (shear[1], shear[2])
    assert shear[2][jump] > 1.8 * shear[2][jump - 1]                           # the rate jump shears harder
    for lagged in (dict(), dict(attach_target="predicted"), dict(attach_impl="springs")):
        try:
            make_controller(dict(lagged, canal_tie_follow="corpus"), rows, Xc, tie)
            raise AssertionError("a lagged attach with canal_tie_follow must be refused: %r" % (lagged,))
        except ValueError:
            pass
    return dict(max_shear_mm_lag0_1_2=[round(max(shear[j]), 6) for j in (0, 1, 2)])


def test_eu3_dilate_and_modes_with_carry():
    """canal_tie_step(X, carry=c) IS canal_tie_step(X + c) -- bit for bit, every output and the engagement -- in dilate,
    centre, centre + below-flange and centre + depth engagement.  Dilate with a rigid carry (tube + corpus one body):
    nodes inside the tube radius get the carried position + the outward correction (<= cap), nodes outside it are
    never pulled in: their target IS the carried position (zero force), where the start-of-step target pulled with k x
    the step's motion."""
    rng = np.random.default_rng(31)
    a_fin, a_path = _axes()
    modes = [dict(), dict(canal_tie_mode="centre"), dict(canal_tie_mode="centre", canal_engage_mm=12.0,
                                                         canal_tie_below_mm=10.0),
             dict(canal_tie_mode="centre", canal_engage="depth")]
    n = 0
    for m in modes:
        p = S.canal_tie_params(S.load_cfg(m), R_TUBE, L_IU)
        cs = rng.uniform(-10.0, 30.0, 30)
        e1, e2 = np.full(30, -1, int), np.full(30, -1, int)
        for k in range(20):
            F = np.array([1.0, 2.0, 3.0]) + k * 0.7 * a_fin
            X = F + np.outer(rng.uniform(-12, 65, 30), a_fin) + rng.normal(0, 3.0, (30, 3))
            c = rng.normal(0, 0.6, (30, 3))
            A = S.canal_tie_step(X, F, a_fin, a_path, e1, k, p, tip_s=-5.0 + 2.0 * k, canal_s=cs, carry=c)
            B = S.canal_tie_step(X + c, F, a_fin, a_path, e2, k, p, tip_s=-5.0 + 2.0 * k, canal_s=cs)
            for key in ("tgt", "k", "act", "inside", "d"):
                assert np.array_equal(A[key], B[key]), (m, k, key)
            assert A["canal_d"] == B["canal_d"] and np.array_equal(e1, e2)
            n += 1
    # dilate, one body
    p = S.canal_tie_params(S.load_cfg(dict(k_canal_mN_per_mm=400.0)), R_TUBE, L_IU)
    rows, F0, a0, _ = _eu2_rigid_rows(12, n_steps=4, deg=1.0, mm=0.6)
    e1v = geom.ortho([1.0, 0.0, 0.0], a0)
    rr = np.array([1.0, 1.5, 4.0, 5.0])                                 # 2 inside the radius, 2 outside
    Xr = F0 + np.outer([10.0, 20.0, 30.0, 40.0], a0) + rr[:, None] * e1v
    for k in range(1, len(rows)):
        ra, rb = rows[k - 1], rows[k]
        X = S.corpus_pose_targets(Xr, ra["T_corpus"])
        X_end = S.corpus_pose_targets(Xr, rb["T_corpus"])
        at = geom.unit(rb["tube_axis"])
        Fb = np.asarray(rb["F"], float)
        res = S.canal_tie_step(X, Fb, at, at, np.full(4, -1, int), k, p, carry=S.canal_tie_carry("corpus", X, ra, rb))
        old = S.canal_tie_step(X, Fb, at, at, np.full(4, -1, int), k, p)
        d_end, _ = _line_dist(X_end, Fb, at)
        assert np.abs(d_end - rr).max() < 1e-9
        inside = rr < p["rad"]
        assert list(res["act"]) == list(inside)
        off = np.linalg.norm(res["tgt"] - X_end, axis=1)
        assert np.allclose(off[inside], np.minimum(p["rad"] - rr[inside], p["cap"])) and off[~inside].max() < 1e-9
        dT, _ = _line_dist(res["tgt"][inside], Fb, at)
        assert np.all(dT > rr[inside])                                   # moved OUT only
        assert np.array_equal(old["tgt"][~inside], X[~inside])          # the old target: the start-of-step node
    return dict(calls_compared=n)


def test_eu3_tie_force_log_values():
    """tie_force_log on hand-computed springs (k mN/mm x (target - node) mm): net 0.2828 N, max per node 0.2 N, summed
    0.4 N, 2 loaded, largest carry 3 mm (no carry key without a carry); and the controller's end-of-step row
    (_canal_tie_force_row) is exactly tie_force_log of the targets / stiffnesses _begin wrote, with the follow mode --
    or "off" and no carry under log_canal_tie_force alone."""
    tgt = np.array([[0.5, 0.0, 0.0], [10.0, 9.0, 10.0], [5.0, 5.0, 5.0]])
    X = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
    k = np.array([400.0, 200.0, 0.0])
    carry = np.array([[0.3, 0.4, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    f = S.tie_force_log(tgt, k, X, carry)
    assert f == dict(net_N=0.2828, max_node_N=0.2, sum_abs_N=0.4, n_loaded=2, carry_max_mm=3.0), f
    f0 = S.tie_force_log(tgt, k, X)
    assert "carry_max_mm" not in f0 and f0["net_N"] == 0.2828
    e = S.tie_force_log(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))
    assert e == dict(net_N=0.0, max_node_N=0.0, sum_abs_N=0.0, n_loaded=0, carry_max_mm=0.0), e
    # the controller row, with an off-target end state
    rows, F0, a0, _ = _eu2_rigid_rows(14, n_steps=4)
    Xr = _on_tube(F0, a0, n=10, seed=3)
    Xc = np.vstack([Xr, Xr + 20.0])
    idx = np.arange(10)
    rng = np.random.default_rng(2)
    for cfg, follow in ((dict(FOLLOW_OK, canal_tie_follow="corpus"), "corpus"), (dict(log_canal_tie_force=True), "off")):
        ctrl, ctx = make_controller(dict(cfg, canal_tie_mode="centre", canal_tie_axis="row"), rows, Xc, idx)
        assert ctrl.tie_force_on
        for kk in range(len(rows)):
            ctx["nodes"]["cervix"].dofs.position.value = S.corpus_pose_targets(Xc, rows[max(kk - 1, 0)]["T_corpus"]).tolist()
            ctrl.k = kk
            ctrl._begin()
        X_end = S.corpus_pose_targets(Xc, rows[-1]["T_corpus"]) + rng.normal(0, 0.2, Xc.shape)
        row = ctrl._canal_tie_force_row(X_end)
        want = S.tie_force_log(np.asarray(ctx["canal_tgt"].position.value), np.asarray(ctx["canal_ff"].stiffness.value),
                               X_end[idx], ctrl.canal_tie_last["carry"])
        assert row == dict(want, follow=follow), (row, want)
        assert ("carry_max_mm" in row) == (follow != "off") and row["net_N"] > 0.0
    ctrl, _ = make_controller(dict(canal_tie_mode="centre"), rows, Xc, idx)
    assert not ctrl.tie_force_on                                       # the default logs nothing new
    return dict(hand=f)


def test_eu2_controller_follow_and_default():
    """The controller: canal_tie_follow "corpus" writes targets equal to the one-body end-of-step positions (zero
    force), and logs nothing new in _begin; the default cfg writes exactly the pre-eu2 targets on the same states."""
    rows, F0, a0, Ts = _eu2_rigid_rows(8, n_steps=6)
    Xr = _on_tube(F0, a0, n=20, seed=2)
    Xc = np.vstack([Xr, Xr + 30.0])                                   # 20 tie nodes + 20 untied cervix nodes
    idx = np.arange(20)
    tgt = dict(tube_axis=geom.unit(rows[-1]["tube_axis"]), axis=a0, R_rows=rows[-1]["R_rows"])
    base = dict(canal_tie_mode="centre", canal_tie_axis="row", canal_rod_axis="tube", k_canal_mN_per_mm=400.0)
    cN, ctxN = make_controller(dict(base, canal_tie_follow="corpus", **FOLLOW_OK), rows, Xc, idx, tgt=tgt)
    cO, ctxO = make_controller(base, rows, Xc, idx, tgt=tgt)
    p = S.canal_tie_params(S.load_cfg(base), R_TUBE, L_IU)
    eng_ref = np.full(20, -1, int)
    worst = 0.0
    for k in range(len(rows)):
        Xk = S.corpus_pose_targets(Xc, rows[max(k - 1, 0)]["T_corpus"])     # end of the previous step
        for ctx in (ctxN, ctxO):
            ctx["nodes"]["cervix"].dofs.position.value = Xk.tolist()
        cN.k = cO.k = k
        cN._begin()
        cO._begin()
        at = geom.unit(rows[k]["tube_axis"])
        ref = _pre_eu2_canal_tie_step(Xk[idx], np.asarray(rows[k]["F"], float), at, at, eng_ref, k, p)
        assert np.array_equal(np.asarray(ctxO["canal_tgt"].position.value), ref["tgt"])
        assert np.array_equal(np.asarray(ctxO["canal_ff"].stiffness.value), ref["k"])
        X_end = S.corpus_pose_targets(Xc, rows[k]["T_corpus"])[idx]
        worst = max(worst, float(np.abs(np.asarray(ctxN["canal_tgt"].position.value) - X_end).max()))
        assert cN.row_prev is rows[k] and cN.canal_tie_last is not None and cO.canal_tie_last is None
    assert worst < 1e-9, worst
    return dict(follow_target_vs_end_max_mm=worst)


# ----------------------------------------------------------------------------- TF4: the rectum's lateral-only support
def test_tf4_rectum_lateral_target_rule():
    """rectum_lateral_targets: x = the rest x, y and z = the current position, bit for bit; the input is not modified;
    a spring k (target - x) at the start of the step then has zero y / z components and k (x_rest - x) in x."""
    rng = np.random.default_rng(41)
    X_rest = rng.normal(0.0, 20.0, (50, 3))                          # synthetic
    X_now = X_rest + rng.normal(0.0, 5.0, X_rest.shape)
    keep = X_now.copy()
    T = S.rectum_lateral_targets(X_now, X_rest[:, 0])
    assert np.array_equal(X_now, keep)                                # no side effect
    assert np.array_equal(T[:, 0], X_rest[:, 0]) and np.array_equal(T[:, 1:], X_now[:, 1:])
    k = 0.5
    f = k * (T - X_now)
    assert np.array_equal(f[:, 1:], np.zeros((50, 2)))
    assert np.allclose(f[:, 0], k * (X_rest[:, 0] - X_now[:, 0]), atol=0.0, rtol=0.0)
    T0 = S.rectum_lateral_targets(X_rest, X_rest[:, 0])               # at rest: target = the node, zero force
    assert np.array_equal(T0, X_rest)
    Xs = X_rest + np.array([0.0, -7.0, 3.0])                          # pushed straight back / up: still zero force
    assert np.array_equal(S.rectum_lateral_targets(Xs, X_rest[:, 0]), Xs)
    return dict(n=len(T))


def test_tf4_rectum_lateral_k_and_nodes():
    """The cfg key: absent / None / 0 -> 0.0 (nothing is built), a number -> float, anything else refused; the node set
    = surface nodes minus the junction set, sorted."""
    assert S.CFG["k_rectum_lateral_mN_per_mm"] is None
    assert S.rectum_lateral_k({}) == 0.0 and S.rectum_lateral_k(S.load_cfg({})) == 0.0
    assert S.rectum_lateral_k(dict(k_rectum_lateral_mN_per_mm=0)) == 0.0
    assert S.rectum_lateral_k(dict(k_rectum_lateral_mN_per_mm=0.5)) == 0.5
    assert S.rectum_lateral_k(dict(k_rectum_lateral_mN_per_mm=1)) == 1.0
    for bad in (-0.1, True, "0.5", float("nan"), float("inf"), [0.5]):
        try:
            S.rectum_lateral_k(dict(k_rectum_lateral_mN_per_mm=bad))
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (bad,))
    ns = dict(surface_nodes=[9, 1, 4, 7, 3, 12], junction_sigmoid=[7, 12, 20], fixed_ends=[1, 7, 12, 20])
    assert S.rectum_lateral_nodes(ns).tolist() == [1, 3, 4, 9]      # lower cut end (1) kept, junction (7, 12) out


def test_tf4_rectum_lateral_log_values():
    """rectum_lateral_log on known states: 10 nodes 2 mm toward +x at k 0.5 -> fx -1 mN each (net -0.01 N), one node
    3 mm toward -x -> +1.5 mN; y / z drag = k x the step's motion; units N."""
    x_rest = np.zeros(11)
    X_end = np.zeros((11, 3))
    X_end[:10, 0] = 2.0
    X_end[10, 0] = -3.0
    tgt = np.zeros((11, 3))
    tgt[:, 0] = x_rest
    X_end[:, 1] = -0.2                                                # every node moved 0.2 mm in -y over the step
    out = S.rectum_lateral_log(tgt, 0.5, X_end, x_rest)
    assert abs(out["net_fx_N"] - (-0.010 + 0.0015)) < 1e-9, out
    assert abs(out["sum_abs_fx_N"] - 0.0115) < 1e-9, out
    assert abs(out["max_node_fx_N"] - 0.0015) < 1e-9, out
    assert abs(out["dx_mean_mm"] - round(17.0 / 11.0, 4)) < 1e-9 and out["dx_absmax_mm"] == 3.0, out
    assert abs(out["drag_yz_net_N"] - 11 * 0.5 * 0.2 / 1000.0) < 1e-9 and out["n"] == 11, out
    return out


class _RecNode(object):
    """A scene-graph node that records addObject / addChild (a stand-in for Sofa.Core.Node)."""

    def __init__(self, path, log):
        self.path, self.log = path, log

    def addObject(self, typ, name=None, **kw):
        o = _obj(**kw)
        o.typ, o.name = typ, name
        self.log.append((self.path, typ, name, kw))
        return o


def _supports_ctx(cfg_over):
    """A fake ctx holding only what _add_supports reads: synthetic meshes (no patient data) and recording nodes."""
    rng = np.random.default_rng(3)
    X0 = {b: rng.normal(0.0, 10.0, (30, 3)) for b in S.BODIES}
    ns = {b: dict() for b in S.BODIES}
    ns["cervix"]["lateral_os_level"] = []                            # no cardinal ligaments in this fake
    ns["vagina"]["fixed_inferior"] = [0, 1]
    ns["bladder"]["anterior_support"] = [2, 3]
    ns["rectum"].update(posterior_support=[4, 5, 6], fixed_ends=[0, 1, 28, 29], junction_sigmoid=[28, 29],
                        surface_nodes=list(range(0, 30, 2)) + [29])
    ns["sigmoid"]["fixed_ends"] = [0]
    log = []
    ctx = dict(cfg=S.load_cfg(dict(cfg_over)), inp=dict(meta={b: dict(node_sets=ns[b]) for b in S.BODIES}), X0=X0,
               nodes={b: _RecNode("/" + b, log) for b in S.BODIES}, targets=_RecNode("/targets", log),
               supports=_RecNode("/supports", log), extra={}, static=[])
    return ctx, log


def test_tf4_rectum_lateral_build_off_and_on():
    """_add_supports: key None (default) and 0 build exactly the objects of the key-less cfg (same calls, same args,
    no ctx["rect_lat"], no extra); 0.5 adds ONLY the target MO (rest positions of surface minus junction nodes) and one
    RestShapeSpringsForceField on the rectum pointing at it, after every existing object; a static rectum is refused."""
    base = dict(k_rectum_support_mN_per_mm=0.1, k_rectum_ends_mN_per_mm=0.1)
    ctxA, logA = _supports_ctx(base)
    ctxA["cfg"].pop("k_rectum_lateral_mN_per_mm")                    # the pre-TF4 cfg (no key at all)
    S._add_supports(ctxA)
    for v in (None, 0, 0.0):
        ctxB, logB = _supports_ctx(dict(base, k_rectum_lateral_mN_per_mm=v))
        S._add_supports(ctxB)
        assert repr(logB) == repr(logA) and "rect_lat" not in ctxB and ctxB["extra"] == ctxA["extra"], v
    ctxC, logC = _supports_ctx(dict(base, k_rectum_lateral_mN_per_mm=0.5))
    S._add_supports(ctxC)
    assert repr(logC[:len(logA)]) == repr(logA) and len(logC) == len(logA) + 2
    (p1, t1, n1, kw1), (p2, t2, n2, kw2) = logC[len(logA):]
    li = sorted(set(list(range(0, 30, 2)) + [29]) - {28, 29})
    assert (p1, t1, n1) == ("/targets", "MechanicalObject", "rect_lat_tgt")
    assert np.array_equal(np.asarray(kw1["position"]), ctxC["X0"]["rectum"][li])
    assert (p2, t2, n2) == ("/rectum", "RestShapeSpringsForceField", "lateral_support")
    assert kw2["points"] == li and kw2["stiffness"] == [0.5] * len(li)
    assert kw2["external_rest_shape"] == "@/targets/rect_lat_tgt" and kw2["external_points"] == list(range(len(li)))
    rl = ctxC["rect_lat"]
    assert rl["idx"].tolist() == li and np.array_equal(rl["x_rest"], ctxC["X0"]["rectum"][li, 0]) and rl["tgt"] is None
    assert ctxC["extra"]["rectum_lateral"]["n_nodes"] == len(li)
    ctxD, _ = _supports_ctx(dict(base, k_rectum_lateral_mN_per_mm=0.5))
    ctxD["static"] = ["rectum"]
    try:
        S._add_supports(ctxD)
    except ValueError:
        return dict(n_nodes=len(li))
    raise AssertionError("a static rectum with the lateral support was accepted")


def test_tf4_controller_writes_lateral_target():
    """HybridController._begin (3c): with ctx["rect_lat"] the target MO holds (x_rest, y_now, z_now) of the rectum's
    start-of-step state at every step; without it nothing is touched (the default path)."""
    rows, F0, a0, Ts = _eu2_rigid_rows(8, n_steps=4)
    Xr = _on_tube(F0, a0, n=20, seed=2)
    idx = np.arange(20)
    tgt = dict(tube_axis=geom.unit(rows[-1]["tube_axis"]), axis=a0, R_rows=rows[-1]["R_rows"])
    base = dict(canal_tie_mode="centre", canal_tie_axis="row", canal_rod_axis="tube", k_canal_mN_per_mm=400.0)
    rng = np.random.default_rng(5)
    R0 = rng.normal(0.0, 15.0, (40, 3))                               # a synthetic "rectum"
    li = np.arange(0, 40, 3)
    cL, ctxL = make_controller(dict(base, k_rectum_lateral_mN_per_mm=0.5), rows, Xr, idx, tgt=tgt)
    ctxL["nodes"]["rectum"] = types.SimpleNamespace(dofs=_obj(position=R0.tolist(), velocity=np.zeros_like(R0).tolist()))
    ctxL["rect_lat"] = dict(idx=li, x_rest=R0[li, 0].copy(), k=0.5, tgt_mo=_obj(position=R0[li].tolist()), tgt=None)
    cO, ctxO = make_controller(base, rows, Xr, idx, tgt=tgt)
    for k in range(len(rows)):
        Rk = R0 + rng.normal(0.0, 3.0, R0.shape)
        ctxL["nodes"]["rectum"].dofs.position.value = Rk.tolist()
        cL.k = cO.k = k
        cL._begin()
        cO._begin()
        T = np.asarray(ctxL["rect_lat"]["tgt_mo"].position.value)
        assert np.array_equal(T[:, 0], R0[li, 0]) and np.array_equal(T[:, 1:], Rk[li, 1:])
        assert np.array_equal(ctxL["rect_lat"]["tgt"], T)
        assert np.array_equal(np.asarray(ctxL["canal_tgt"].position.value), np.asarray(ctxO["canal_tgt"].position.value))
    assert "rect_lat" not in ctxO
    return dict(steps=len(rows))


# ----------------------------------------------------------------------------- S2 helpers
def test_subdivide_keeps_surface():
    rng = np.random.default_rng(11)
    V = rng.normal(0, 10, (30, 3))
    T = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [5, 6, 7]])
    for lv in (1, 2):
        T2, rounds = S.subdivide_tris(T, lv, n_vertices=len(V))
        V2 = S.apply_subdiv(V, rounds)
        assert np.array_equal(V2[:len(V)], V)
        assert len(T2) == len(T) * 4 ** lv

        def area_vec(P, F):
            return 0.5 * np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]).sum(0)

        assert np.allclose(area_vec(V, T), area_vec(V2, T2))            # same oriented area: shape and winding kept
    T1, r1 = S.subdivide_tris(T, 1, n_vertices=len(V))
    V1 = S.apply_subdiv(V, r1)
    e0, e1 = r1[0]
    assert np.allclose(V1[len(V):], 0.5 * (V[e0] + V[e1]))


def test_sheet_signed_distance_cylinder():
    """A closed-ring cylinder sheet of radius 10 mm along z: points at r 12 read +2, at r 8 read -2 (behind the
    sheet), points beyond an end ring are not measured."""
    n_ax, n_th, R = 21, 60, 10.0
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    z = np.linspace(0.0, 40.0, n_ax)
    Xs = np.array([[R * np.cos(t), R * np.sin(t), zz] for zz in z for t in th])
    st = np.repeat(np.arange(n_ax), n_th)
    tri = []
    for i in range(n_ax - 1):
        for j in range(n_th):
            a, b = i * n_th + j, i * n_th + (j + 1) % n_th
            c, d = a + n_th, b + n_th
            tri += [[a, b, d], [a, d, c]]
    tri = np.array(tri)
    Nv = S.vertex_normals(Xs, tri)
    sign = 1.0 if np.mean(np.einsum("ij,ij->i", Nv, Xs - np.c_[np.zeros((len(Xs), 2)), Xs[:, 2]])) >= 0 else -1.0
    Nv = sign * Nv
    q = np.array([[12.0, 0.0, 20.0], [0.0, 8.0, 21.0], [-12.0 / np.sqrt(2), 12.0 / np.sqrt(2), 5.0],
                  [11.0, 0.0, 45.0], [9.0, 0.0, -3.0]])
    sd, j, ok = S.sheet_signed_distance(q, Xs, Nv, st, n_ax, [0, 0, 1], [0, 0, 1])
    assert ok[:3].all() and not ok[3:].any(), ok
    assert abs(sd[0] - 2.0) < 0.05 and abs(sd[1] + 2.0) < 0.05 and abs(sd[2] - 2.0) < 0.05, sd


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
