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


R_TUBE, L_IU = 2.18, 61.8
APP = dict(params=dict(L_iu_mm=dict(value=L_IU), r_tandem_mm=dict(value=R_TUBE)),
           landmarks=dict(shaft_end_dir=[0.0, 0.48328, -0.87546]))       # applicator_v3's straight rod


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
def _s1a_setup(seed=0):
    rng = np.random.default_rng(seed)
    a_fin = geom.unit([0.0, 0.25, 0.97])                 # a generic tube axis (any unit vector would do)
    a_row = _rot(geom.ortho([1.0, 0.0, 0.0], a_fin), 30.0) @ a_fin
    F = np.array([10.0, 20.0, 0.0])       # synthetic origin (not a patient coordinate)
    n = 40
    s = rng.uniform(1.0, 60.0, n)
    e1 = geom.ortho([1.0, 0.0, 0.0], a_row)
    e2 = np.cross(a_row, e1)
    th = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.2, 5.0, n)
    X = F + np.outer(s, a_row) + (r * np.cos(th))[:, None] * e1 + (r * np.sin(th))[:, None] * e2
    tgt = dict(tube_axis=a_fin, axis=geom.unit([0.00296, -0.26649, 0.96383]), R_rows=_frame(a_fin))
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
    a = geom.unit([0.1, 0.2, 0.97])
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
        return np.linspace(-17.29, 11.46, 39), "synthetic"


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
    a_fin = geom.unit([0.0, 0.25, 0.97])
    a_path = geom.unit([0.00296, -0.26649, 0.96383])
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
            F = np.array([10.0, 18.0, 3.8]) - (1 - w) * 80.0 * a_path   # synthetic
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
