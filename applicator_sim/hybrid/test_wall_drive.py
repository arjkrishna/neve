"""Host unit tests (python 3.13, numpy) of hybrid/wall_drive.py (the S7b device-driven v5 wall) and of the S7b / S7f
cfg checks and schedule rows in scene_hybrid.py (SOFA stubbed as in hybrid/test_ties.py).

    python -P -B hybrid/test_wall_drive.py      # runs every test_* below (pytest-style; pytest is not on this host)

What is tested (all geometry synthetic: a straight collapsed tube along +z, no patient or case-derived numbers):
  tables    node_tables: the start node = its slab centre + an in-plane offset (exact), lumen partners by (station,
            rank), depth below the top ring, the glue pairing checked against vault_top.
  geometry  periodic_weights (exact at the reference angles, wrap-around); ellipse_exit (a rod along the slab normal
            gives a circle of the dilated radius; a tilted rod stretches the section by 1 / cos; off-centre rays; misses);
            device_crossing (centre on the line, the nose shrinking to nothing lead_mm ahead of the tip, no section below
            the rod); approach_disp (the click-in shape); half_inside (half-disc side, slot, outer radius, fillet corner).
  drive     no device -> the start shape exactly (no pre-opening); a rod up the axis opens every lumen node to >= r + c
            (minus the chord of the node spacing) and leaves the slit's far wings alone; the push-out changes by at most
            rate_mm and the nodes move at most speed_mm per step; the slot closes behind a rod that leaves, at the rate;
            straightening is monotone; the top ring follows a rigidly moving portio exactly (and nodes below with the
            glue weight); K2 at 1 gives the rest shape bit for bit; a ring half pushes the lumen out to its outline.
  scene     check_drive_cfg refuses the inconsistent combinations and accepts the defaults; ring_phase_rows gives
            V, C, R_L, R_R, K1, K2 in order, each half from its start distance to 0 at ring_step_mm, only its own
            half live, the lift rows carrying drive_kappa = w_l, u ending at 1.
"""
import os
import sys
import types

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _Controller(object):
    def __init__(self, *a, **k):
        pass


for _n in ("Sofa", "Sofa.Core", "Sofa.Simulation", "SofaRuntime"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["Sofa"].Core = sys.modules["Sofa.Core"]
sys.modules["Sofa"].Simulation = sys.modules["Sofa.Simulation"]
sys.modules["Sofa.Core"].Controller = _Controller

import wall_drive as WD  # noqa: E402
import scene_hybrid as S  # noqa: E402

# ============================================================================================ synthetic wall
K, NTH = 12, 24                     # stations, nodes per ring and sheet
A_IN, B_IN, A_OUT, B_OUT = 6.0, 0.5, 7.5, 2.0
DZ_S, DZ_R, R_REST_IN, R_REST_OUT = 2.0, 3.0, 10.0, 11.5


def synth():
    """A straight collapsed tube along +z (start: slit ellipses; rest: circles, 1.5x longer), a flat 'cervix' plate
    over its top ring (three cervix nodes per top node, the glue point 0.2 mm above the node), and the meta a v5 wall
    carries."""
    th = 2.0 * np.pi * np.arange(NTH) / NTH
    Xs, Xr, sheet, station, theta, gi = [], [], [], [], [], []
    for k in range(K):
        for sh, (a_, b_, rr) in ((0, (A_IN, B_IN, R_REST_IN)), (2, (A_OUT, B_OUT, R_REST_OUT))):
            for j, t in enumerate(th):
                Xs.append([a_ * np.cos(t), b_ * np.sin(t), k * DZ_S])
                Xr.append([rr * np.cos(t), rr * np.sin(t), k * DZ_R])
                sheet.append(sh)
                station.append(k)
                theta.append(t)
                gi.append([k, sh, j])
    Xs, Xr = np.array(Xs), np.array(Xr)
    n = len(Xs)
    station = np.array(station)
    f = station / float(K - 1)
    sigma = station * DZ_S
    top = np.nonzero(station == K - 1)[0]
    # cervix plate: three nodes per top node, the triangle's centroid 0.2 mm above it
    Xc, tri, bary, off = [], [], [], []
    for i in top:
        base = len(Xc)
        for d in ((0.3, 0.0), (-0.15, 0.26), (-0.15, -0.26)):
            Xc.append(Xs[i] + [d[0], d[1], 0.2])
        tri.append([base, base + 1, base + 2])
        bary.append([1 / 3.0, 1 / 3.0, 1 / 3.0])
        off.append([0.0, 0.0, -0.2])
    meta = dict(
        axis=dict(axis=[0.0, 0.0, 1.0], centroid=[0.0, 0.0, 0.0]),
        node_sets=dict(vault_top=[int(i) for i in top], inner_surface=[int(i) for i in np.nonzero(np.array(sheet) == 0)[0]],
                       outer_surface=[int(i) for i in np.nonzero(np.array(sheet) == 2)[0]]),
        wall=dict(grid_index=gi, s=[float(k * DZ_R) for k in range(K)],
                  v5=dict(node_sheet=sheet, node_station=[int(v) for v in station], node_theta_rad=theta,
                          node_start_sigma_mm=[float(v) for v in sigma], node_f=[float(v) for v in f],
                          stations=dict(start=dict(label_sigma_mm=[float(k * DZ_S) for k in range(K)],
                                                   label_centre=[[0.0, 0.0, k * DZ_S] for k in range(K)])),
                          pairing=dict(vault_top=[int(i) for i in top], cervix_tri_nodes=tri, cervix_tri_bary=bary,
                                       offset_start_mm=off))))
    return meta, Xr, Xs, np.array(Xc)


def app_rod(L=40.0, r=2.0, r_shaft=3.0, bottom=-60.0):
    """A straight tandem along the applicator z: tube r down to z -10, then a 'shaft' centreline to `bottom`."""
    zs = np.linspace(-11.0, bottom, 8)
    return dict(params=dict(L_iu_mm=dict(value=L), r_tandem_mm=dict(value=r), r_shaft_mm=dict(value=r_shaft)),
                landmarks=dict(tube_bottom=[0.0, 0.0, -10.0], shaft_centreline=np.c_[np.zeros(8), np.zeros(8), zs].tolist()))


def drive(pr=None, rg=None):
    meta, Xr, Xs, Xc = synth()
    tab = WD.node_tables(meta, Xr, Xs)
    return WD.WallDrive(tab, WD.tandem_geometry(app_rod()), rg, WD.drive_cfg(pr or {})), tab, Xc


def row_tip(z_tip, L=40.0, x=0.0):
    """A schedule row whose rod runs up +z at lateral x with its tip at height z_tip."""
    return dict(F=[x, 0.0, z_tip - L], R_rows=np.eye(3).tolist(), tube_axis=[0.0, 0.0, 1.0])


# ============================================================================================ tests: tables + geometry
def test_node_tables_exact_start_and_partners():
    meta, Xr, Xs, _ = synth()
    t = WD.node_tables(meta, Xr, Xs)
    assert np.abs(t["Cs"] + t["q"] - Xs).max() < 1e-12
    assert np.abs(t["q"] @ t["a"]).max() < 1e-12
    gi = np.asarray(meta["wall"]["grid_index"])
    p = t["partner"]
    assert (t["sheet"][p] == 0).all() and (gi[p, 0] == gi[:, 0]).all() and (gi[p, 2] == gi[:, 2]).all()
    top = t["top"]
    assert np.abs(t["depth"][top]).max() < 1e-9 and t["depth"][t["station"] == 0].min() > 1e6
    k = 5
    assert abs(t["depth"][t["station"] == k][0] - (K - 1 - k) * DZ_S) < 1e-9
    bad = dict(meta)
    bad["wall"] = dict(meta["wall"])
    bad["wall"]["v5"] = dict(meta["wall"]["v5"], pairing=dict(meta["wall"]["v5"]["pairing"], vault_top=top[::-1].tolist()))
    try:
        WD.node_tables(bad, Xr, Xs)
        raise AssertionError("a pairing in another order than vault_top must be refused")
    except ValueError:
        pass


def test_periodic_weights():
    ref = np.array([0.1, 1.0, 2.5, 4.0, 5.9])
    v = np.sin(ref) + 2.0
    j0, j1, w = WD.periodic_weights(ref, ref)
    assert np.abs((1 - w) * v[j0] + w * v[j1] - v).max() < 1e-12
    j0, j1, w = WD.periodic_weights([6.2, 0.0], ref)           # between 5.9 and 0.1 (+2 pi)
    assert set([int(j0[0]), int(j1[0])]) == {4, 0} and 0.0 < w[0] < 1.0 and 0.0 < w[1] < 1.0


def test_ellipse_exit_circle_tilt_offcentre_miss():
    a = np.array([0.0, 0.0, 1.0])
    E = np.array([[1, 0, 0], [0, 1, 0], [np.sqrt(0.5), np.sqrt(0.5), 0]], float)
    O = np.zeros((3, 3))
    r = WD.ellipse_exit(O, E, O, np.tile(a, (3, 1)), np.full(3, 3.0), a, 0.3)
    assert np.abs(r - 3.0).max() < 1e-12
    d = np.tile([np.sin(np.radians(40)), 0.0, np.cos(np.radians(40))], (3, 1))    # tilted 40 deg in x-z
    r = WD.ellipse_exit(O, E, O, d, np.full(3, 3.0), a, 0.3)
    assert abs(r[0] - 3.0 / np.cos(np.radians(40))) < 1e-9 and abs(r[1] - 3.0) < 1e-9
    Oo = np.array([[-5.0, 0.0, 0.0]])                           # ray from outside through the circle: far exit at 8
    r = WD.ellipse_exit(Oo, np.array([[1.0, 0, 0]]), np.zeros((1, 3)), a[None], np.array([3.0]), a, 0.3)
    assert abs(r[0] - 8.0) < 1e-9
    r = WD.ellipse_exit(Oo, np.array([[0.0, 1, 0]]), np.zeros((1, 3)), a[None], np.array([3.0]), a, 0.3)
    assert r[0] == -np.inf


def test_device_crossing_nose_and_bottom():
    geo = WD.tandem_geometry(app_rod())
    CLw = WD.device_world(geo, [0.0, 0.0, 0.0], np.eye(3))
    a = np.array([0.0, 0.0, 1.0])
    h = np.array([-30.0, 0.0, 37.9, 38.0 + 2.25 + 5.0, 38.0 + 2.25 + 5.0 + 0.5, -80.0])
    D, d, rA, ok, s_n = WD.device_crossing(h, CLw, geo["rad"], a, a, np.zeros(3), 0.25, 5.0)
    assert np.abs(D[:, :2]).max() < 1e-12 and np.abs(D[:, 2] - h).max() < 1e-9
    assert ok[0] and ok[1] and abs(rA[0] - 3.25) < 1e-9 and abs(rA[1] - 2.25) < 1e-9
    assert ok[2] and 0.0 < rA[2] <= 2.25 + 1e-12            # just above the hemisphere centre: the nose
    assert not ok[4] and not ok[5]                          # beyond the nose; below the rod
    assert rA[3] < 0.5                                      # the nose closes at its end


def test_approach_disp_click_shape():
    h = dict(e=np.array([0.0, 0.3, -1.0]) / np.linalg.norm([0, 0.3, -1]), n_out_ap=np.array([1.0, 0, 0]), offset=5.0,
             click=20.0, p=4.0, D=100.0)
    assert np.abs(WD.approach_disp(h, 0.0)).max() < 1e-12
    v = WD.approach_disp(h, 50.0)
    assert np.abs(v - (50.0 * h["e"] + 5.0 * h["n_out_ap"])).max() < 1e-12
    o = [float(WD.approach_disp(h, d) @ h["n_out_ap"] - d * (h["e"] @ h["n_out_ap"])) for d in (0.0, 5.0, 10.0, 19.9, 20.0)]
    assert all(b >= a_ - 1e-12 for a_, b in zip(o, o[1:])) and abs(o[-1] - 5.0) < 1e-12


def ring_synth():
    """A ring (ring frame = applicator frame, origin 0): faces z -20 .. 0, outer radius 20, fillets 4 / 3, slot 0.5;
    halves R (+x) and L (-x) with a short boss and a rod going down."""
    halves = {}
    for side, sx in (("R", 1.0), ("L", -1.0)):
        halves[side] = dict(part="ovoid_" + side, n_out=np.array([sx, 0.0, 0.0]),
                            boss=np.array([[sx * 5.0, 15.0, -15.0], [sx * 5.0, 20.0, -25.0]]),
                            rod=np.array([[sx * 5.0, 20.0, -25.0], [sx * 5.0, 30.0, -60.0]]),
                            e=WD.unit(np.array([0.0, 0.3, -1.0])), n_out_ap=np.array([sx, 0.0, 0.0]), D=80.0,
                            offset=5.0, click=15.0, p=4.0, order=1 if side == "L" else 2)
    return dict(origin=np.zeros(3), R=np.eye(3), z_top=0.0, z_bot=-20.0, f_top=4.0, f_bot=3.0, R_out=20.0, slot=0.5,
                r_boss=4.0, r_rod=3.0, halves=halves, order=["L", "R"])


def test_half_inside():
    rg = ring_synth()
    P = np.array([[10.0, 0.0, -10.0],        # R half, mid-thickness
                  [-10.0, 0.0, -10.0],       # L side
                  [0.1, 0.0, -10.0],         # in the slot
                  [21.5, 0.0, -10.0],        # beyond R_out + clear
                  [19.5, 0.0, -0.5],         # the top outer corner: rounded off by the fillet
                  [5.0, 25.0, -40.0]])       # on the R rod
    ins = WD.half_inside(P, rg, "R", 0.0)
    assert list(ins) == [True, False, False, False, False, True]
    assert WD.half_inside(P[3:4], rg, "R", 2.0)[0]                 # within the dilation


# ============================================================================================ tests: the drive
def test_no_device_is_the_start_exactly():
    drv, tab, Xc = drive()
    P, st, dg = drv.step(row_tip(-200.0), Xc, drv.init_state())
    assert np.abs(P - tab["X_start"]).max() < 1e-12 and dg["n_pushed"] == 0 and dg["push_max_mm"] == 0.0


def test_rod_opens_lumen_to_r_plus_c_and_leaves_wings():
    pr = dict(speed_mm=None, rate_mm=100.0)
    drv, tab, Xc = drive(pr)
    st = drv.init_state()
    for z in np.arange(-60.0, 30.1, 1.0):                   # the tip passes every station (heights 0 .. 22)
        P, st, dg = drv.step(row_tip(z), Xc, st)
    inn = tab["inner"]
    below = tab["hs"][inn] < 30.0 - 2.0
    rr = np.hypot(P[inn, 0], P[inn, 1])
    # every station lies on the TUBE here (the tip at 30, the tube down to 30 - 40 - 10 = -20): r 2 + c
    need = 2.0 + WD.DRIVE_DEFAULTS["clear_mm"]
    ok = rr[below & (tab["hs"][inn] < 30.0 - 12.0)]
    assert ok.min() >= need - 1e-9, ok.min()
    wings = tab["rho_s"][inn] > need + 0.5                  # slit nodes beyond the rod: untouched radially
    assert np.abs(rr[wings] - tab["rho_s"][inn][wings]).max() < 1e-9


def test_rate_and_speed_limits():
    drv, tab, Xc = drive(dict(rate_mm=0.3, speed_mm=0.5))
    st = drv.init_state()
    Pp, pp = tab["X_start"].copy(), np.zeros(tab["n"])
    for z in np.arange(-40.0, 30.1, 2.0):                   # a fast tip: 2 mm per step
        P, st, dg = drv.step(row_tip(z), Xc, st)
        assert np.linalg.norm(P - Pp, axis=1).max() <= 0.5 + 1e-9
        assert np.abs(st["push"] - pp).max() <= 0.3 + 1e-9
        Pp, pp = P, st["push"].copy()


def test_slot_closes_behind_at_rate():
    drv, tab, Xc = drive(dict(speed_mm=None, rate_mm=0.7))
    st = drv.init_state()
    for z in np.arange(-60.0, 30.1, 1.0):
        P, st, _ = drv.step(row_tip(z), Xc, st)
    p0 = st["push"].max()
    assert p0 > 2.0
    prev = st["push"].copy()
    for k in range(40):                                     # the rod leaves (pulled far down)
        P, st, _ = drv.step(row_tip(-300.0), Xc, st)
        assert (prev - st["push"]).max() <= 0.7 + 1e-9 and (st["push"] >= -1e-12).all()
        prev = st["push"].copy()
    assert st["push"].max() < 1e-12
    # the straightening is monotone: omega from the highest tip reached, not the current one
    assert drv.omega(st["tip_h_max"]).min() >= drv.omega(-300.0).min()


def test_glue_follows_rigid_portio():
    drv, tab, Xc = drive(dict(speed_mm=None))
    v = np.array([1.5, -0.8, 2.0])
    P, st, _ = drv.step(row_tip(-200.0), Xc + v, drv.init_state())
    top = tab["top"]
    assert np.abs(P[top] - (tab["X_start"][top] + v)).max() < 1e-12
    g = drv.g
    mid = np.nonzero((g > 0.05) & (g < 0.95))[0]
    assert len(mid) and np.abs(P[mid] - (tab["X_start"][mid] + g[mid, None] * v)).max() < 1e-9
    far = np.nonzero(g == 0.0)[0]
    assert np.abs(P[far] - tab["X_start"][far]).max() < 1e-12


def test_k2_end_is_rest_bit_for_bit():
    """k2_vault "rest" (the DEFAULT, TF2): K2 ends on the rest shape bit for bit.  This is the behaviour that detaches the
    vault from a portio that is not where the rest shape expects it (test_k2_live_keeps_the_top_ring_glued)."""
    drv, tab, Xc = drive(dict(speed_mm=None))
    row = dict(row_tip(20.0), drive_kappa=1.0, drive_k2=1.0)
    P, st, dg = drv.step(row, Xc, drv.init_state())
    assert np.array_equal(P, tab["X_rest"]) and dg["lam_mean"] == 1.0
    row = dict(row_tip(20.0), drive_kappa=1.0, drive_k2=0.4)
    P, st, dg = drv.step(row, Xc, drv.init_state())
    assert 0.0 < dg["lam_mean"] < 1.0


def test_ring_half_pushes_to_outline():
    rg = ring_synth()
    drv, tab, Xc = drive(dict(speed_mm=None, rate_mm=100.0, ring_clear_mm=1.0, ring_ray_mm=40.0), rg=rg)
    st = drv.init_state()
    for z in np.arange(-60.0, 30.1, 1.0):
        P, st, _ = drv.step(row_tip(z), Xc, st)
    # the R half seated with its top face at height 16 (applicator origin 16 above world 0), rod pointing down
    org, R = np.array([0.0, 0.0, 16.0]), np.eye(3)
    row = dict(row_tip(30.0), ring={"R": (org, R)}, ring_lead={})
    P, st, dg = drv.step(row, Xc, st)
    inn = tab["inner"]
    zc = tab["hs"][inn]
    sel = (zc > 16.0 - 20.0 + 5.0) & (zc < 16.0 - 5.0) & (tab["E"][inn][:, 0] > 0.9)   # mid-thickness, R side
    rr = np.hypot(P[inn][sel, 0], P[inn][sel, 1])
    assert len(rr) and rr.min() >= 20.0 + 1.0 - 0.3, rr.min()
    lsel = (zc > 16.0 - 15.0) & (zc < 16.0 - 5.0) & (tab["E"][inn][:, 0] < -0.9)       # the L side: collapsed
    rl = np.hypot(P[inn][lsel, 0], P[inn][lsel, 1])
    assert rl.max() < 8.0, rl.max()
    assert dg["ring_push"]["R"] > 0


# ============================================================================================ tests: scene helpers
def test_check_drive_cfg():
    S.check_drive_cfg(S.load_cfg({}))
    ok = dict(vagina_model="wall", wall_drive="device", n_balloon=0, apex_attach="off", insertion_path="tandem_first")
    S.check_drive_cfg(dict(S.load_cfg({}), **ok))
    for bad in (dict(ok, n_balloon=60), dict(ok, apex_attach="lift"), dict(ok, insertion_path="canal"),
                dict(ok, wall_drive="balloon"), dict(ok, wall_drive_params=dict(nope=1)),
                dict(wall_handover=True), dict(n_k2=10), dict(ring_phases=True, insertion_path="canal"),
                dict(settle_rayleigh_stiffness=0.01), dict(warm_start=dict(tag="X"))):
        try:
            S.check_drive_cfg(dict(S.load_cfg({}), **bad))
            raise AssertionError("must be refused: %r" % bad)
        except ValueError:
            pass
    d = S.load_cfg({})
    assert d["wall_drive"] is None and d["ring_phases"] is False and d["settle_rayleigh_stiffness"] is None \
        and d["warm_start"] is None and d["n_k2"] == 0 and d["wall_handover"] is False


def test_ring_phase_rows_order_and_distances():
    def pose_of(r):
        return dict(F=np.asarray(r["F"], float), tube_axis=np.array([0.0, 0.0, 1.0]), R_rows=np.eye(3),
                    T_corpus=np.eye(4), tip_s=float(r["tip_s"]), d_mm=float(r["tip_s"]), w_r=float(r["w_r"]),
                    w_l=float(r["w_l"]))
    rows = [dict(phase="V", F=[0, 0, i], tip_s=i - 10.0, w_r=0.0, w_l=0.0) for i in range(5)] + \
           [dict(phase="C", F=[0, 0, 5 + i], tip_s=float(i), w_r=i / 4.0, w_l=0.0) for i in range(5)] + \
           [dict(phase="L", F=[0, 0, 10 + i], tip_s=4.0, w_r=1.0, w_l=(i + 1) / 3.0) for i in range(3)]
    ring = dict(approach=dict(halves=dict(L=dict(order=1, D_mm=10.0), R=dict(order=2, D_mm=10.0))))
    cfg = dict(ring_step_mm=3.0, ring_start_mm=None, ring_hold_steps=1, n_k2=4)
    out = S.ring_phase_rows(cfg, dict(ring_app=ring), rows, pose_of)
    ph = [r["phase"] for r in out]
    order = []
    for p in ph:
        if not order or order[-1] != p:
            order.append(p)
    assert order == ["V", "C", "R_L", "R_R", "K1", "K2"], order
    rl = [r for r in out if r["phase"] == "R_L"]
    assert [r["ring_d"]["L"] for r in rl] == [10.0, 7.0, 4.0, 1.0, 0.0, 0.0] and all(r["ring_d"]["R"] == 10.0 for r in rl)
    assert all(r["ring_on"] == ["L"] for r in rl)
    rr = [r for r in out if r["phase"] == "R_R"]
    assert all(r["ring_d"]["L"] == 0.0 for r in rr) and rr[-1]["ring_d"]["R"] == 0.0 and rr[0]["ring_on"] == ["L", "R"]
    k1 = [r for r in out if r["phase"] == "K1"]
    assert [r["drive_kappa"] for r in k1] == [1 / 3.0, 2 / 3.0, 1.0]
    k2 = [r for r in out if r["phase"] == "K2"]
    assert [r["drive_k2"] for r in k2] == [0.25, 0.5, 0.75, 1.0] and abs(out[-1]["u"] - 1.0) < 1e-12
    assert all(r.get("ring_on") == [] for r in out if r["phase"] in ("V", "C"))


def test_ring_half_does_not_push_the_glued_vault():
    rg = ring_synth()
    drv, tab, Xc = drive(dict(speed_mm=None, rate_mm=100.0, ring_clear_mm=1.0, ring_ray_mm=40.0), rg=rg)
    st = drv.init_state()
    for z in np.arange(-60.0, 30.1, 1.0):
        P, st, _ = drv.step(row_tip(z), Xc, st)
    top = tab["top"]
    P0 = P[top].copy()
    # the R half overlapping the top station (height 22): its face 26 .. 6
    row = dict(row_tip(30.0), ring={"R": (np.array([0.0, 0.0, 26.0]), np.eye(3))}, ring_lead={})
    P, st, dg = drv.step(row, Xc, st)
    assert np.abs(P[top] - P0).max() < 1e-12            # the glued top ring stays on the portio
    assert dg["ring_push"]["R"] > 0                     # ... while the nodes below it are pushed


def test_portio_tie_engages_once_the_tube_is_there():
    pt = dict(k=100.0, eng=np.full(3, -1, int))
    X = np.array([[0.0, 1.0, 50.0], [0.0, 8.0, 30.0], [0.0, 40.0, 30.0]])   # on the line; beside it; far aside
    L = 40.0
    for k, z_tip in enumerate((20.0, 34.9, 36.0, 60.0)):     # the tip rises along +z (flange = tip - L)
        kk, eng = S.portio_tie_step(X, dict(F=[0.0, 0.0, z_tip - L], R_rows=np.eye(3)), pt, k, L, 3.0)
        if k <= 1:
            assert not eng.any() and not kk.any()               # the tip below / less than 5 mm past every node
    assert list(pt["eng"]) == [3, 2, -1]                        # node 1 at step 2 (tip 36 >= 30 + 5), node 0 at step 3
    assert abs(kk[1] - 100.0 * 2 / 3.0) < 1e-9 and abs(kk[0] - 100.0 / 3.0) < 1e-9 and kk[2] == 0.0


def test_portio_anchor_offsets():
    pt = dict(k=100.0, eng=np.full(2, -1, int))
    X = np.array([[3.0, 0.0, 10.0], [0.0, 4.0, 12.0]])
    row0 = dict(F=[0.0, 0.0, -20.0], R_rows=np.eye(3))
    kk, eng = S.portio_tie_step(X, row0, pt, 0, 40.0, 1.0)             # tip at 20: both nodes >= 5 mm below it
    assert eng.all() and np.allclose(pt["anchor_app"], X - [0.0, 0.0, -20.0])
    assert np.abs(S.portio_anchor_offsets(X, row0, pt, eng, 0.5)).max() < 1e-12    # on the anchor: no force
    row1 = dict(F=[2.0, 0.0, -15.0], R_rows=np.eye(3))                  # the tube moved 2 mm sideways, 5 mm up
    d = S.portio_anchor_offsets(X, row1, pt, eng, 0.5)
    assert np.allclose(d, [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])            # lateral only, capped at 0.5
    d = S.portio_anchor_offsets(X + [1.8, 0.0, 3.0], row1, pt, eng, 0.5)
    assert np.allclose(d, [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])            # the axial offset is not corrected


def test_grid_smooth_contains_and_wraps():
    nS, nT = 8, 12
    si, ti = [a.ravel() for a in np.meshgrid(np.arange(nS), np.arange(nT), indexing="ij")]
    keep = np.ones(len(si), bool)
    keep[[5, 17, 40]] = False                                       # missing grid cells are ignored
    g = dict(si=si[keep], ti=ti[keep], nS=nS, nT=nT)
    v = np.zeros(int(keep.sum()))
    j0 = int(np.nonzero((g["si"] == 3) & (g["ti"] == 0))[0][0])
    v[j0] = 6.0                                                     # one spike at angle cell 0
    o = WD.grid_smooth(v, g, 1, 1)
    assert np.all(o >= v - 1e-12) and o[j0] >= 6.0 - 1e-12          # never below the push it smooths
    jw = int(np.nonzero((g["si"] == 3) & (g["ti"] == nT - 1))[0][0])
    assert o[jw] > 0.0                                              # the angle wraps round
    far = (np.abs(g["si"] - 3) > 2)
    assert np.all(o[far] == 0.0)                                    # nothing beyond twice the window
    assert np.array_equal(WD.grid_smooth(v, g, 0, 0), v)            # off = unchanged


def test_ring_phase_rows_joint_seat():
    def pose_of(r):
        return dict(F=np.asarray(r["F"], float), tube_axis=np.array([0.0, 0.0, 1.0]), R_rows=np.eye(3),
                    T_corpus=np.eye(4), tip_s=float(r["tip_s"]), d_mm=float(r["tip_s"]), w_r=float(r["w_r"]),
                    w_l=float(r["w_l"]))
    rows = [dict(phase="V", F=[0, 0, i], tip_s=i - 10.0, w_r=0.0, w_l=0.0) for i in range(3)] + \
           [dict(phase="C", F=[0, 0, 5 + i], tip_s=float(i), w_r=i / 2.0, w_l=0.0) for i in range(3)] + \
           [dict(phase="L", F=[0, 0, 10 + i], tip_s=4.0, w_r=1.0, w_l=(i + 1) / 2.0) for i in range(2)]
    h = dict(e_app=[0.0, 0.0, -1.0], n_out_app=[1.0, 0.0, 0.0], offset_mm=2.0, click_mm=4.0, click_shape_p=2.0)
    ring = dict(approach=dict(halves=dict(L=dict(h, order=1, D_mm=20.0), R=dict(h, order=2, D_mm=20.0,
                                                                                n_out_app=[-1.0, 0.0, 0.0]))))
    cfg = dict(ring_seat="joint", ring_step_mm=2.0, ring_step_near_mm=0.5, ring_near_mm=6.0, ring_standoff_mm=6.0,
               ring_start_mm=12.0, ring_hold_steps=2, n_k2=0)
    out = S.ring_phase_rows(cfg, dict(ring_app=ring), rows, pose_of)
    order = []
    for p in (r["phase"] for r in out):
        if not order or order[-1] != p:
            order.append(p)
    assert order == ["V", "C", "R_L", "R_R", "R_S", "K1"], order
    rl = [r for r in out if r["phase"] == "R_L"]
    assert [round(r["ring_d"]["L"], 6) for r in rl] == [10.0, 8.0, 6.0] and all(r["ring_d"]["R"] == 20.0 for r in rl)
    rr = [r for r in out if r["phase"] == "R_R"]
    assert all(r["ring_d"]["L"] == 6.0 for r in rr) and rr[-1]["ring_d"]["R"] == 6.0 and rr[0]["ring_on"] == ["L", "R"]
    rs = [r for r in out if r["phase"] == "R_S"]
    assert all(r["ring_d"]["L"] == r["ring_d"]["R"] for r in rs) and rs[-1]["ring_d"]["L"] == 0.0
    assert sum(1 for r in rs if r["ring_d"]["L"] == 0.0) == 3              # the seat row + 2 holds
    dd = [6.0] + [r["ring_d"]["L"] for r in rs]
    P = S.ring_disp_app(h, dd)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    assert steps.max() <= 0.5 + 0.02 and steps[steps > 0].min() > 0.0     # equal path steps in the near zone
    assert all(r["drive_kappa"] > 0 for r in out if r["phase"] == "K1")


def test_ring_phase_rows_after_lift():
    def pose_of(r):
        return dict(F=np.asarray(r["F"], float), tube_axis=np.array([0.0, 0.0, 1.0]), R_rows=np.eye(3),
                    T_corpus=np.eye(4), tip_s=float(r["tip_s"]), d_mm=float(r["tip_s"]), w_r=float(r["w_r"]),
                    w_l=float(r["w_l"]))
    rows = [dict(phase="V", F=[0, 0, i], tip_s=i - 10.0, w_r=0.0, w_l=0.0) for i in range(2)] + \
           [dict(phase="C", F=[0, 0, 5 + i], tip_s=float(i), w_r=i / 1.0, w_l=0.0) for i in range(2)] + \
           [dict(phase="L", F=[0, 0, 10 + i], tip_s=4.0, w_r=1.0, w_l=(i + 1) / 3.0) for i in range(3)]
    h = dict(e_app=[0.0, 0.0, -1.0], n_out_app=[1.0, 0.0, 0.0], offset_mm=2.0, click_mm=4.0, click_shape_p=2.0)
    ring = dict(approach=dict(halves=dict(L=dict(h, order=1, D_mm=20.0), R=dict(h, order=2, D_mm=20.0))))
    cfg = dict(ring_seat="joint", ring_after_lift=True, ring_step_mm=4.0, ring_step_near_mm=1.0, ring_near_mm=4.0,
               ring_standoff_mm=4.0, ring_start_mm=12.0, ring_hold_steps=1, n_k2=2)
    out = S.ring_phase_rows(cfg, dict(ring_app=ring), rows, pose_of)
    order = []
    for p in (r["phase"] for r in out):
        if not order or order[-1] != p:
            order.append(p)
    assert order == ["V", "C", "K1", "R_L", "R_R", "R_S", "K2"], order
    k1 = [r for r in out if r["phase"] == "K1"]
    assert [r["drive_kappa"] for r in k1] == [1 / 3.0, 2 / 3.0, 1.0] and all(r["ring_on"] == [] for r in k1)
    rr = [r for r in out if r["phase"] in ("R_L", "R_R", "R_S")]
    assert all(np.array_equal(r["F"], np.array([0.0, 0.0, 12.0])) for r in rr)      # the ring rides the LIFTED pose
    assert all(r["drive_kappa"] == 1.0 for r in rr)                                 # stations stay at rest height
    assert all(r["ring_d"] == dict(L=20.0, R=20.0) for r in k1)                     # both halves parked in K1


def test_check_ring_seat_and_portio_phases():
    base = dict(S.load_cfg({}), ring_phases=True, insertion_path="tandem_first", ovoid_mode="none")
    S.check_drive_cfg(dict(base, ring_seat="joint", ring_step_near_mm=0.5))
    for bad in (dict(base, ring_seat="together"), dict(S.load_cfg({}), ring_seat="joint"),
                dict(base, ring_seat="joint", ring_standoff_mm=30.0, ring_start_mm=20.0),
                dict(base, ring_seat="joint", ring_step_near_mm=0.0), dict(base, ring_step_near_mm=0.5),
                dict(base, portio_tie_phases="C")):
        try:
            S.check_drive_cfg(bad)
            raise AssertionError("must be refused: %r" % {k: bad[k] for k in ("ring_seat", "ring_step_near_mm")})
        except ValueError:
            pass
    dev = dict(S.load_cfg({}), vagina_model="wall", wall_drive="device", n_balloon=0, apex_attach="off",
               insertion_path="tandem_first")
    S.check_drive_cfg(dict(dev, vault_tie_phase="H", vault_tie_k_mN_per_mm=20.0))
    for bad in (dict(dev, vault_tie_phase="settle", vault_tie_k_mN_per_mm=20.0),
                dict(dev, vault_tie_phase="H", vault_tie_k_mN_per_mm=0.0),
                dict(dev, vault_tie_phase="H", vault_tie_k_mN_per_mm=20.0, wall_handover=True)):
        try:
            S.check_drive_cfg(bad)
            raise AssertionError("must be refused: vault_tie_phase %r" % bad["vault_tie_phase"])
        except ValueError:
            pass
    d = S.load_cfg({})
    assert d["vault_tie_phase"] == "handover"
    assert d["ring_seat"] == "sequential" and d["ring_step_near_mm"] is None and d["portio_tie_phases"] is None
    assert WD.DRIVE_DEFAULTS["ring_smooth_st"] == 0 and WD.DRIVE_DEFAULTS["ring_smooth_th"] == 0


# ============================================================================================ TF3: the live vault in K2
def portio_cylinder(R=14.0, z0=22.2, z1=60.0, n=48):
    """A closed triangulated cylinder (radius R, flat bottom face at z0 -- the synthetic start's glue plane -- top face
    at z1) as the 'cervix' the live vault meets: nodes and triangles (outward)."""
    th = 2.0 * np.pi * np.arange(n) / n
    bot = np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z0)]
    topr = np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z1)]
    V = np.vstack([bot, topr, [[0.0, 0.0, z0]], [[0.0, 0.0, z1]]])
    cb, ct = 2 * n, 2 * n + 1
    T = []
    for i in range(n):
        j = (i + 1) % n
        T += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    T = np.array(T, int)
    return V, WD.orient_outward(V, T)


def live_setup(pr=None, tilt_deg=11.0, lift_mm=10.8):
    """The synthetic wall + a cervix made of the start glue plate (synth) and a closed cylinder whose bottom face holds
    the plate at the start; for K2 the whole cervix is lifted by lift_mm and tilted by tilt_deg about y through the
    bottom face centre, so the REST top ring (z 33, r 10 / 11.5) lies up to ~2.2 mm inside it on one side and up to
    ~2.2 mm below it on the other."""
    meta, Xr, Xs, Xp = synth()
    Vc, Tc = portio_cylinder()
    Xc = np.vstack([Xp, Vc])
    tri = Tc + len(Xp)
    tab = WD.node_tables(meta, Xr, Xs)
    base = dict(speed_mm=None, k2_vault="live", vault_band_mm=15.0, k2_blend_mm=15.0)
    base.update(pr or {})
    drv = WD.WallDrive(tab, WD.tandem_geometry(app_rod()), None, WD.drive_cfg(base), cervix_tri=tri)
    c = np.radians(tilt_deg)
    Ry = np.array([[np.cos(c), 0.0, np.sin(c)], [0.0, 1.0, 0.0], [-np.sin(c), 0.0, np.cos(c)]])
    o = np.array([0.0, 0.0, 22.2])
    Xc2 = (Xc - o) @ Ry.T + o + [0.0, 0.0, lift_mm]
    return drv, tab, Xc2, tri


def test_k2_live_keeps_the_top_ring_glued():
    """k2_vault "live" (TF3, defect A of the TF2 review): through every K2 row and the settle rows the top ring stays ON
    the (moved, tilted) portio -- normal gap to its junction <= 1 mm and signed distance to the closed cervix within
    [-0.5, 1] mm -- no wall node ends inside the cervix, and the wall below the vault band ends exactly at rest.  The
    DEFAULT "rest" K2 on the same portio leaves top-ring nodes > 1 mm off it (the TF2 detachment)."""
    drv, tab, Xc, tri = live_setup()
    T = WD.orient_outward(Xc, tri)
    top = tab["top"]
    st = drv.init_state()
    row0 = dict(row_tip(-200.0), drive_kappa=1.0)
    for _ in range(3):                                  # the portio moved before K2: the glue carries the top ring
        P, st, _ = drv.step(row0, Xc, st)
    n_k2 = 40
    worst_gap, worst_sd = 0.0, (np.inf, -np.inf)
    for k in range(1, n_k2 + 8):                        # K2 rows, then settle rows (k2 = 1)
        row = dict(row0, drive_k2=min(1.0, k / float(n_k2)))
        P, st, dg = drv.step(row, Xc, st)
        J, nJ, ref = drv.junction(Xc, st)
        assert ref == "live"
        gap = np.einsum("ij,ij->i", P[top] - (J - drv.p["vault_offset_mm"] * nJ), nJ)
        Pc, _, _, dd = WD.closest_on_tris(P[top], Xc, T)
        sd = np.where(WD.inside_closed(P[top], Xc, T), -dd, dd)
        worst_gap = max(worst_gap, float(np.abs(gap).max()))
        worst_sd = (min(worst_sd[0], float(sd.min())), max(worst_sd[1], float(sd.max())))
    assert worst_gap <= 1.0, worst_gap
    assert worst_sd[0] >= -0.5 and worst_sd[1] <= 1.0, worst_sd
    assert not WD.inside_closed(P, Xc, T).any()
    below = ~np.isin(np.arange(tab["n"]), drv.vb["nodes"])
    assert np.array_equal(P[below], tab["X_rest"][below])
    assert dg["vault_live"]["n_fallback"] == 0 and dg["vault_live"]["n_shortened"] > 0 and dg["vault_live"]["n_extended"] > 0
    # the portio keeps moving in the settle: the top ring follows it (kinematically)
    Xc3 = Xc + [0.8, -0.5, 1.2]
    for _ in range(4):
        P, st, dg = drv.step(dict(row0, drive_k2=1.0), Xc3, st)
    J, nJ, _ = drv.junction(Xc3, st)
    assert np.abs(P[top] - J).max() < 1e-9
    # the DEFAULT on the same portio: the rest top ring, off it
    drv_r, _, _, _ = live_setup(dict(k2_vault="rest"))
    P_r, _, _ = drv_r.step(dict(row0, drive_k2=1.0), Xc, drv_r.init_state())
    Pc, _, _, dd = WD.closest_on_tris(P_r[top], Xc, T)
    sd_r = np.where(WD.inside_closed(P_r[top], Xc, T), -dd, dd)
    assert sd_r.max() > 1.0 and sd_r.min() < -0.5, (sd_r.min(), sd_r.max())
    return dict(live_gap_max_mm=round(worst_gap, 3), live_sd=np.round(worst_sd, 3).tolist(),
                rest_sd=[round(float(sd_r.min()), 2), round(float(sd_r.max()), 2)])


def test_geometry_helpers():
    V, T = portio_cylinder()
    Q = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 20.0], [20.0, 0.0, 30.0], [5.0, 3.0, 59.0]])
    assert list(WD.inside_closed(Q, V, T)) == [True, False, False, True]
    Pc, j, b, d = WD.closest_on_tris(Q, V, T)
    assert abs(d[1] - 2.2) < 1e-9 and abs(Pc[1, 2] - 22.2) < 1e-9          # below the bottom face
    assert abs(d[3] - 1.0) < 1e-9                                          # 1 mm under the top face
    assert np.abs(np.einsum("ij,ijk->ik", b, V[T[j]]) - Pc).max() < 1e-9
    t, j, b = WD.seg_first_hit(np.array([[1.0, 1.0, 0.0], [30.0, 0.0, 0.0]]), np.array([[1.0, 1.0, 40.0], [30.0, 0.0, 40.0]]),
                               V, T)
    assert abs(t[0] - 22.2 / 40.0) < 1e-12 and j[1] == -1 and not np.isfinite(t[1])
    assert np.abs(np.einsum("j,jk->k", b[0], V[T[j[0]]]) - [1.0, 1.0, 22.2]).max() < 1e-9
    n = WD.tri_normals(V, T)
    c = V[T].mean(1)
    assert (np.einsum("ij,ij->i", n, c - [0.0, 0.0, 41.1]) > 0).all()      # outward


def test_lead_poses_and_the_slope_cone():
    """lead_poses gives the old lead poses (+ how much further each is); with ring_lead_slope the lumen ahead of an
    approaching half opens only within (push needed) / slope of the half's own surface, where the default opens it to
    the half's full outline ring_lead_mm ahead (TF2: 41 mm)."""
    rg = ring_synth()
    h = rg["halves"]["R"]
    pr = WD.drive_cfg(dict(ring_lead_mm=30.0, ring_lead_samples=6))
    F, Rr = np.zeros(3), np.eye(3)
    lp = WD.lead_poses(F, Rr, h, 25.0, pr)
    old = [WD.half_pose(F, Rr, h, max(0.0, 25.0 - 30.0 * j / 6)) for j in range(1, 7)]
    assert all(np.array_equal(a_[0], b_[0]) and np.array_equal(a_[1], b_[1]) for a_, b_ in zip(lp, old))
    assert [round(x[2], 9) for x in lp] == [5.0, 10.0, 15.0, 20.0, 25.0, 25.0]
    assert WD.lead_poses(F, Rr, h, 0.0, pr) == []

    def ring_push(slope, d):
        """Per station height: the largest push-out the half (with its leads) adds; and the half's top-face height."""
        p = dict(speed_mm=None, rate_mm=1e9, ring_clear_mm=1.0, ring_ray_mm=40.0, ring_lead_mm=30.0,
                 ring_lead_samples=15, ring_lead_slope=slope)
        drv, tab, Xc = drive(p, rg=rg)
        st = drv.init_state()
        for z in np.arange(-60.0, 30.1, 1.0):
            P, st, _ = drv.step(row_tip(z), Xc, st)
        push0 = st["push"].copy()
        org = np.array([0.0, 0.0, 16.0])                # the seat: top face at height 16
        row = dict(row_tip(30.0), ring={"R": WD.half_pose(org, Rr, h, d)},
                   ring_lead={"R": WD.lead_poses(org, Rr, h, d, drv.p)})
        P, st, _ = drv.step(row, Xc, st)
        inn = tab["inner"]
        dp = st["push"][inn] - push0[inn]
        hs = tab["hs"][inn]
        zs = np.unique(hs)
        return zs, np.array([dp[hs == z].max() for z in zs]), 16.0 + float(WD.approach_disp(h, d)[2])
    d, s = 20.0, 1.0
    zs, full, z_face = ring_push(None, d)
    _, cone, _ = ring_push(s, d)
    ez = -float(h["e"][2])                               # height gained per mm of approach
    assert (cone <= full + 1e-9).all()                   # the cone never opens more than the full outline
    far = zs >= z_face + 10.0
    assert far.any() and (full[far] > 12.0).any()        # the default opens the lumen wide 10+ mm ahead of the half ...
    assert (cone[far] <= np.maximum(full[far] - 5.0, 0.0) + 1e-9).all(), (cone[far], full[far])    # ... the cone does not
    ah = (zs > z_face) & (cone > 0.5)                    # ahead of the face: falls at >= ~ s per mm of approach
    dz = np.diff(zs[ah])
    assert ah.sum() >= 3 and (-np.diff(cone[ah]) >= 0.9 * s * dz / ez).all(), cone[ah]
    return dict(z_face=round(z_face, 1), full=np.round(full[zs > z_face], 1).tolist(),
                cone=np.round(cone[zs > z_face], 1).tolist())


def test_check_live_vault_cfg():
    dev = dict(S.load_cfg({}), vagina_model="wall", wall_drive="device", n_balloon=0, apex_attach="off",
               insertion_path="tandem_first", ring_phases=True, ovoid_mode="none", n_k2=10)
    S.check_drive_cfg(dict(dev, wall_drive_params=dict(k2_vault="live", ring_lead_slope=0.8)))
    for bad in (dict(dev, wall_drive_params=dict(k2_vault="live"), wall_handover=True),
                dict(dev, wall_drive_params=dict(k2_vault="live"), vault_tie_k_mN_per_mm=20.0),
                dict(dev, wall_drive_params=dict(k2_vault="live"), n_k2=0),
                dict(dev, wall_drive_params=dict(k2_vault="packed")),
                dict(dev, wall_drive_params=dict(ring_lead_slope=0.0))):
        try:
            S.check_drive_cfg(bad)
            raise AssertionError("must be refused: %r" % bad["wall_drive_params"])
        except ValueError:
            pass
    assert WD.DRIVE_DEFAULTS["k2_vault"] == "rest" and WD.DRIVE_DEFAULTS["ring_lead_slope"] is None
    pr = WD.drive_cfg({})
    assert set(WD.params_record(pr)) == set(pr) - set(WD.TF3_KEYS)      # a TF2 cfg records what it recorded
    assert WD.params_record(WD.drive_cfg(dict(k2_vault="live")))["k2_vault"] == "live"


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
