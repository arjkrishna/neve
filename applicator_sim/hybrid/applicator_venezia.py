"""HOST: parametric Venezia-type applicator and the kinematic pose rule v1 (CONTRACT.md sections 2 and 3).

    cd applicator_sim/hybrid        (set MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1)
    python applicator_venezia.py                    # meshes + ovoid fit + pose rule + validation + figures + README
    python applicator_venezia.py --no-fit           # reuse the ovoid fit stored in applicator/applicator.json
    python applicator_venezia.py --angle 24 --n-steps 80 --path-axis shaft
    py -3.11 applicator_venezia.py --render3d       # pyvista render of the device meshes -> figs/applicator_3d.png
    python -P applicator_venezia.py --variant v4 --tandem-only [--pose-from v3] [--app-label <updated label>]
        # fix plan S4a/S4c: the tandem body alone (tube + vaginal tandem, one rigid body) measured on the UPDATED BT
        # applicator label, with v3's validated tube pose; writes applicator_v4/ (see main_tandem_only).  A rebuild
        # keeps pose.json insertion_path_tandem_first while the body is unchanged, else stops ([--force]: stale_records)
    python -P applicator_venezia.py --variant v5 --ring [--tandem-from v4] [--ring-outer-r 20] [--app-label <label>]
        # fix plan S4b/S4c: v4's tandem body (copied, checked) + two lunar ring halves (ring OUTER ~40 mm, each half
        # one closed manifold with its socket boss and its own rod into the handle bundle), measured on the updated BT
        # applicator label + the BT ovoid label; T_seat, per-half approach paths (host-checked clearance), label_check;
        # pose.json insertion_path_tandem_first carried from v4 and verified (see main_ring)

Outputs under <APPSIM_OUT>/hybrid/ (default ~/Downloads/MRI_GYN_sim/hybrid):
    applicator/{tube.obj, shaft.obj, ovoid_L.obj, ovoid_R.obj}   closed, outward-oriented surfaces in the APPLICATOR frame
    applicator/applicator.json    every parameter (value, unit, provenance), needle-channel axes, mesh stats, ovoid fit
    applicator/pose.json          pose rule v1 in the preBT world frame: shaft/tube axes, flange, corpus rigid transform
                                  (4x4 + screw decomposition), insertion-path keyframes, validation against the BT tandem
    figs/{ovoid_fit.png, pose_rule_BT.png, pose_rule_preBT.png, pose_rule_path.png, applicator_3d.png}
    logs/applicator_venezia.json

Applicator frame (CONTRACT 2): origin = flange centre (where the tube leaves the ovoid caps), z = intrauterine tube axis
flange->tip, y = anterior in the sagittal plane, x = y cross z (patient right).  Units mm, deg.
Nothing patient-specific is hard-coded: case measurements come from <out>/inputs/*.json, inputs/canal.npz,
final/bdev.json, validation/alignment.json and the READ-ONLY label images.  Device-shape defaults are the contract's.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import config  # noqa: E402
import geom  # noqa: E402

P = config.paths()
HY = P["out"] + "/hybrid"
APP = HY + "/applicator"
FIGS = HY + "/figs"
LOGS = HY + "/logs"
FRAME_APP = ("applicator frame: origin = flange centre (tube exit from the ovoid caps), z = intrauterine tube axis "
             "flange->tip, y = anterior in the sagittal plane, x = y cross z (patient right); mm")
FRAME_PRE = "preBT world RAS mm (nibabel affine of preBT_MRI_label_*.nii; x=R, y=A, z=S)"
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])
README_MARK = "## APPLICATOR + POSE RULE"


# ============================================================================ small helpers
def jz(x, nd=5):
    """numpy -> JSON-able (rounded)."""
    if isinstance(x, dict):
        return {k: jz(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jz(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return jz(x.tolist(), nd)
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def val(prm, k):
    return prm[k]["value"]


def _load_json_or_empty(fn):
    try:
        return json.load(open(fn))
    except (OSError, ValueError):
        return {}


def _find_key(obj, key):
    """First value of `key` anywhere in a nested JSON object (None if absent)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rodrigues(k, deg):
    k = geom.unit(k)
    K = geom.skew(k)
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def app_frame_rows(a_tube):
    """Applicator frame rows (x, y, z) for a tube axis: y = anterior in the sagittal plane, x = y cross z."""
    z = geom.unit(a_tube)
    y = geom.ortho(EY, z)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def to_world(F, R, p_app):
    return np.asarray(F, float) + np.atleast_2d(p_app) @ np.asarray(R, float)


def to_app(F, R, p_w):
    return (np.atleast_2d(p_w) - np.asarray(F, float)) @ np.asarray(R, float).T


# ============================================================================ parameters
def default_params(app_in, bdev):
    """Every parameter with value, unit and provenance (MEASURED / CONTRACT / ASSUMED / FITTED / NUMERICAL)."""
    p = {}

    def add(k, v, unit, source, note=""):
        p[k] = dict(value=v, unit=unit, source=source, **({"note": note} if note else {}))

    add("L_iu_mm", float(app_in["L_iu_mm"]), "mm", "MEASURED (BT applicator label, flange->tip; inputs/applicator.json)")
    add("r_tandem_mm", float(app_in["r_tandem_mm"]), "mm", "MEASURED (BT rod FWHM 4.36 mm / 2; inputs/applicator.json)")
    add("tip_hemispherical", True, "-", "ASSUMED (rounded tandem tip)")
    add("shaft_arc_len_mm", 30.0, "mm", "CONTRACT 2 (30 mm arc below the flange)",
        "inputs/applicator.json measured a 26 mm straight shaft in the BT label; the arc is the contract's device model")
    add("angle_deg", 24.0, "deg", "MEASURED 22-25 deg tube/shaft junction angle (CONTRACT 2), default 24")
    add("r_shaft_mm", float(app_in["r_tandem_mm"]), "mm", "ASSUMED = r_tandem (one continuous tube through the caps)")
    # ---- Stage 3 (2026-09-21): the VAGINAL PART as the updated BT applicator label shows it (Downloads/BT_MRI_label_
    #      applicator.nii, hybrid/logs/applicator_label_new_geometry.json): the tandem's own rod continues STRAIGHT from
    #      the flange down the vaginal axis, and two further rods hang from the caps, all three through the vagina and
    #      out of the body.  The ring/caps stay perpendicular to the TUBE (the user's device knowledge; the label's
    #      cap minor axis is 14 deg from the tube and 16 deg from the rods, not decisive).
    add("shaft_style", "arc", "-", "'arc' = the 30 mm bent shaft of the Stage-1 model; 'straight' = the tandem rod straight down "
                                   "the vaginal (shaft) axis from the flange, as in the BT label (rods 2.7 deg from the preBT vaginal axis)")
    add("rod_r_mm", 3.11, "mm", "MEASURED BT applicator label: three rods of median cross-section 30.4 mm^2 (partial volume "
                                "may add ~0.3 mm); the tube reads 2.37 mm on the same label vs 2.18 by FWHM")
    add("rod_len_mm", 80.0, "mm", "MEASURED BT applicator label: rods reach 79 mm below the ring centre (out of the body)")
    add("ovoid_rod_offsets_mm", [[-4.3, 16.0], [4.3, 16.0]], "mm",
        "MEASURED BT applicator label at z = -16.8 (BT): the two cap rods sit 16 mm ANTERIOR of the tandem rod and 8.6 mm apart "
        "LR; (x, y') in the plane normal to the rods through the flange, y' = anterior. Symmetrised about x = 0")
    add("rods", False, "-", "write rod_L.obj / rod_R.obj (the cap rods) and use shaft_style 'straight' for the tandem rod")
    # ---- the vaginal PACKING (Stage 3, 2026-09-21): a kinematic cylinder about the vaginal rod, riding with the ring,
    #      that the wall rests on from inside.  MEASURED at BT: the vagina label at the ring's level IS the device +
    #      packing (its section is centred on the ring to 1-3 mm, r_mean 13, r_max 22); the vault is held by the ring
    #      and the packing, not by tissue springs.  In the model the wall (10 kPa, 1.2 mm) crushed between the cervix
    #      bulk pushing it and any spring holding the ring (G25); on a kinematic packing the cervix yields instead.
    add("pack_r_mm", 17.0, "mm", "ASSUMED: lumen reference 20.7 mm minus 3.7 mm = outside the 3.0 mm contact alarm distance when coaxial, so the packing carries contacts only where the wall is pushed onto it (the vault). MEASURED G27: at 18.5 mm (2.2 mm clearance) every lumen vertex along the packing was a proximity contact: 1739 contacts, 136 s/step")
    add("pack_top_mm", 0.0, "mm", "ASSUMED: packing from the flange level (the caps' zone) ...")
    add("pack_len_mm", 70.0, "mm", "ASSUMED: ... 70 mm down the rods (the introitus is ~80 mm below the final flange)")
    add("pack_top_round_mm", 10.0, "mm", "ASSUMED: height of the elliptical shoulder that domes the packing's top (0 = the flat top of G26)")
    add("pack_top_r_mm", 12.0, "mm", "ASSUMED: the packing's radius at its top when domed (the caps' apex zone emerges from it)")
    add("ovoid_diam_mm", 40.0, "mm", "FITTED to the BT ovoid label (AP = LR diameter of the cap assembly; initial 40)")
    add("ovoid_height_mm", 22.2, "mm", "MEASURED assembly extent along z (CONTRACT 2)")
    add("ovoid_dome_mm", 8.0, "mm", "FITTED dome height of each cap (= height -> pure half-ellipsoid cap; initial 8)")
    add("ovoid_offset_x_mm", 0.0, "mm", "FITTED lateral offset of each cap's centre from the tube axis (0 = the two halves of one body)")
    sph_dz = _find_key(_load_json_or_empty(P["out"] + "/calibration/best.json"), "sph_dz_mm")
    add("ovoid_dz_mm", 0.0, "mm", "FITTED cap apex level relative to the flange along z",
        "the frame origin is the tube exit from the ovoid label (h_ovtop %.2f mm); the earlier sphere-pack calibration "
        "shifted the pack by sph_dz = %s mm (calibration/best.json)" % (float(app_in.get("h_ovtop_mm", 0.0)),
                                                                        "n/a" if sph_dz is None else "%.2f" % float(sph_dz)))
    add("ovoid_tilt_deg", 0.0, "deg", "FITTED AP tilt of the cap assembly about x (positive = apex tilted anteriorly)")
    add("ovoid_gap_mm", 0.5, "mm", "ASSUMED slot between the two caps at the mid-sagittal plane (the tube passes through the caps)")
    add("channel_radius_mm", 17.0, "mm", "CONTRACT 2 (parallel needle channels on a 17 mm radius)")
    add("n_parallel_per_ovoid", 5, "-", "CONTRACT 2")
    add("n_oblique_per_ovoid", 2, "-", "CONTRACT 2")
    add("oblique_deg", 15.0, "deg", "CONTRACT 2 (oblique channels at 15 deg to the tube axis, tilted laterally outward)")
    add("oblique_radius_mm", 12.0, "mm", "ASSUMED entry radius of the oblique channels at the cap bottom")
    add("needle_r_mm", 1.0, "mm", "CONTRACT 2")
    add("parallel_azimuth_deg", [20.0, 55.0, 90.0, 125.0, 160.0], "deg", "ASSUMED layout: azimuth from +y (anterior) toward the cap's side")
    add("oblique_azimuth_deg", [45.0, 135.0], "deg", "ASSUMED layout")
    add("mesh_n_theta", 24, "-", "NUMERICAL (vertices per ring of the tube/shaft sweeps)")
    add("mesh_target_tris", 1500, "-", "NUMERICAL (~1-2k triangles per part)")
    add("mesh_voxel_mm", 0.3, "mm", "NUMERICAL (implicit-function grid for the cap surfaces)")
    add("tip_below_os_mm", 4.0, "mm", "CONTRACT 3.4 (path start: tube tip 4 mm below the external os)")
    add("n_steps", 80, "-", "CONTRACT 3.4 (N_steps default 80)")
    add("path_axis", "shaft", "-", "CONTRACT 3.4 (device translates along the shaft axis; alternative 'tube' also written)")
    add("corpus_ease", "smoothstep", "-", "DECISION: screw-motion parameter s(u) = smoothstep((u - u_ios)/(1 - u_ios)); 'linear' available")
    add("tilt_axis", "LR", "-", "CONTRACT 3.2: tube axis = shaft axis rotated anteriorly about the patient LR axis (variant 'sagittal' reported)")
    add("d_F_margin_mm", 0.5, "mm", "CONTRACT 3.3 / final/bdev.json: d_F = L_iu + 0.5 - canal_above_L_end")
    add("vagina_fixed_inferior_mm", 15.0, "mm", "CONTRACT 1 (vagina.fixed_inferior = lowest 15 mm along the vaginal axis)")
    # ---- rule v2 (2026-09-11). v1 (vaginal-chord shaft axis, flange pinned at O_pre) is REJECTED on this case: see
    #      pose.json[validation][rule_v1]. v2 = shaft axis from the vagina body's principal axis + the seating shift Delta.
    add("shaft_axis_mode", "pca", "-", "RULE v2: 'pca' = principal axis of the vagina body (oriented +S); 'chord' = v1 (fixed point -> O_pre)")
    add("shaft_line_through", "O_pre", "-", "RULE v2: shaft line through 'O_pre', or 'vagina_axis' (the vagina body's own axis line; O_pre projected onto it)")
    add("flange_shift_mm", 0.0, "mm", "RULE v2 SCENARIO parameter Delta = seating push of the ovoids/packing: flange = base + Delta along shift_axis. "
                                       "Not predictable from preBT; the default is the in-sample best of the sweep (pose.json default_flange_shift_mm)")
    add("shift_axis", "shaft", "-", "RULE v2: direction of the seating shift ('shaft' = insertion direction; 'tube' reported as a variant)")
    add("flange_shift_sweep_mm", [0.0, 10.0, 20.0, 30.0, 35.0, 40.0], "mm", "RULE v2: Delta values scored against the BT tandem (validation only)")
    return p


FIT_KEYS = ("ovoid_diam_mm", "ovoid_offset_x_mm", "ovoid_dz_mm", "ovoid_tilt_deg", "ovoid_dome_mm")


# ============================================================================ implicit device geometry (applicator frame)
def ovoid_inside(Q, prm, side):
    """Membership of points Q (n,3, applicator frame) in the lunar cap of one side (+1 = patient right, -1 = left).
    Cap = half of a dome-on-cylinder body (dome = upper half-ellipsoid of height ovoid_dome_mm on a cylinder), the whole
    assembly tilted by ovoid_tilt_deg about x; medial flat face at |x| = ovoid_gap_mm / 2."""
    Q = np.atleast_2d(np.asarray(Q, float))
    t = np.radians(val(prm, "ovoid_tilt_deg")); c, s = np.cos(t), np.sin(t)
    x = Q[:, 0]; y = c * Q[:, 1] - s * Q[:, 2]; z = s * Q[:, 1] + c * Q[:, 2]        # assembly frame (axis tilted anteriorly by t)
    a = 0.5 * val(prm, "ovoid_diam_mm"); H = val(prm, "ovoid_height_mm"); cd = min(val(prm, "ovoid_dome_mm"), H)
    ztop = val(prm, "ovoid_dz_mm"); zeq = ztop - cd; zbot = ztop - H
    xs = side * x - val(prm, "ovoid_offset_x_mm")
    rr = (xs / a) ** 2 + (y / a) ** 2
    cyl = (z >= zbot) & (z <= zeq) & (rr <= 1.0)
    dome = (z > zeq) & (rr + ((z - zeq) / cd) ** 2 <= 1.0)
    return (cyl | dome) & (side * x >= 0.5 * val(prm, "ovoid_gap_mm"))


def ovoid_bbox(prm, side, margin=1.5):
    a = 0.5 * val(prm, "ovoid_diam_mm"); xc = val(prm, "ovoid_offset_x_mm"); H = val(prm, "ovoid_height_mm")
    ztop = val(prm, "ovoid_dz_mm"); tilt = abs(val(prm, "ovoid_tilt_deg"))
    ex = a + xc + margin; ey = a + margin + (H + a) * np.sin(np.radians(tilt))
    lo = np.array([0.0 if side > 0 else -ex, -ey, ztop - H - margin - ey * np.sin(np.radians(tilt))])
    hi = np.array([ex if side > 0 else 0.0, ey, ztop + margin + ey * np.sin(np.radians(tilt))])
    return lo, hi


def shaft_arc(prm, n=31):
    """Centreline C (n,3) and unit tangents T of the shaft arc below the flange: starts at the origin tangent to -z,
    bends anteriorly (+y) by angle_deg over shaft_arc_len_mm."""
    th = np.radians(val(prm, "angle_deg")); L = val(prm, "shaft_arc_len_mm")
    if th < 1e-6:
        s = np.linspace(0, L, n)
        return np.stack([np.zeros(n), np.zeros(n), -s], 1), np.tile([0.0, 0.0, -1.0], (n, 1))
    rho = L / th; al = np.linspace(0, th, n)
    C = np.stack([np.zeros(n), rho * (1 - np.cos(al)), -rho * np.sin(al)], 1)
    T = np.stack([np.zeros(n), np.sin(al), -np.cos(al)], 1)
    return C, T


def rod_frame(prm):
    """Direction DOWN the vaginal rod and the unit vector normal to it in the sagittal plane (toward anterior), both in
    the applicator frame: the rods are the tube axis rotated back by angle_deg about x, pointing down."""
    th = np.radians(val(prm, "angle_deg"))
    d_rod = np.array([0.0, np.sin(th), -np.cos(th)])
    y_rod = np.array([0.0, np.cos(th), np.sin(th)])
    return d_rod, y_rod


def shaft_inside(Q, prm):
    """Points within r_shaft of the arc centreline."""
    Q = np.atleast_2d(np.asarray(Q, float)); r = val(prm, "r_shaft_mm")
    th = np.radians(val(prm, "angle_deg")); L = val(prm, "shaft_arc_len_mm")
    C, _ = shaft_arc(prm, 2)
    if th < 1e-6:
        s = np.clip(-Q[:, 2], 0, L)
        return np.hypot(Q[:, 0], Q[:, 1]) ** 2 + (-Q[:, 2] - s) ** 2 <= r * r
    rho = L / th
    vy = Q[:, 1] - rho; vz = Q[:, 2]
    al = np.arctan2(-vz, -vy)
    d_circ = np.sqrt(Q[:, 0] ** 2 + (np.hypot(vy, vz) - rho) ** 2)
    on = (al >= 0) & (al <= th)
    d_end = np.minimum(np.linalg.norm(Q - C[0], axis=1), np.linalg.norm(Q - C[-1], axis=1))
    return np.where(on, d_circ, d_end) <= r


def device_inside(Q, prm, with_tube=False):
    m = ovoid_inside(Q, prm, +1) | ovoid_inside(Q, prm, -1) | shaft_inside(Q, prm)
    if with_tube:
        Q = np.atleast_2d(Q); L = val(prm, "L_iu_mm"); r = val(prm, "r_tandem_mm")
        s = np.clip(Q[:, 2], 0, L - r)
        m |= (Q[:, 0] ** 2 + Q[:, 1] ** 2 + (Q[:, 2] - s) ** 2) <= r * r
    return m


def needle_channels(prm):
    """Needle-channel axes (Stage 2 geometry; bodies not built): point on the axis at the cap bottom plane, direction."""
    ztop = val(prm, "ovoid_dz_mm"); zbot = ztop - val(prm, "ovoid_height_mm")
    Rt = rot_x(-val(prm, "ovoid_tilt_deg"))              # assembly axis = Rt @ z (anterior tilt)
    ch = []
    for side, name in ((-1, "L"), (+1, "R")):
        for k, az in enumerate(val(prm, "parallel_azimuth_deg")):
            rad = np.array([side * np.sin(np.radians(az)), np.cos(np.radians(az)), 0.0])
            p = Rt @ (val(prm, "channel_radius_mm") * rad + zbot * EZ)
            ch.append(dict(id="%s_par%d" % (name, k + 1), ovoid=name, type="parallel", azimuth_deg=float(az),
                           axis_point_mm=jz(p), dir=jz(Rt @ EZ), radius_mm=val(prm, "needle_r_mm"),
                           length_mm=val(prm, "ovoid_height_mm")))
        for k, az in enumerate(val(prm, "oblique_azimuth_deg")):
            rad = np.array([side * np.sin(np.radians(az)), np.cos(np.radians(az)), 0.0])
            ob = np.radians(val(prm, "oblique_deg"))
            d = Rt @ geom.unit(np.sin(ob) * rad + np.cos(ob) * EZ)
            p = Rt @ (val(prm, "oblique_radius_mm") * rad + zbot * EZ)
            ch.append(dict(id="%s_obl%d" % (name, k + 1), ovoid=name, type="oblique", azimuth_deg=float(az),
                           axis_point_mm=jz(p), dir=jz(d), radius_mm=val(prm, "needle_r_mm"),
                           length_mm=val(prm, "ovoid_height_mm") / np.cos(ob)))
    return ch


# ============================================================================ meshes
def sweep(rings, bottom, top, n_th):
    """Closed tube of revolution-like sweep: rings = [(centre, e1, e2, radius)], fan caps to `bottom` and `top`."""
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    V = [np.asarray(bottom, float)]; starts = []
    for c, e1, e2, r in rings:
        starts.append(len(V))
        V.extend(np.asarray(c, float) + r * np.cos(th)[:, None] * e1 + r * np.sin(th)[:, None] * e2)
    V.append(np.asarray(top, float)); it = len(V) - 1
    F = []
    b0 = starts[0]
    for j in range(n_th):
        F.append([0, b0 + (j + 1) % n_th, b0 + j])
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        for j in range(n_th):
            j1 = (j + 1) % n_th
            F.append([a + j, a + j1, b + j1]); F.append([a + j, b + j1, b + j])
    bl = starts[-1]
    for j in range(n_th):
        F.append([it, bl + j, bl + (j + 1) % n_th])
    V = np.array(V, float); F = np.array(F, int)
    if geom.mesh_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def mesh_stats(V, F, analytic_vol=None):
    """Closedness / orientation from the edge structure (pure numpy) + volumes."""
    F = np.asarray(F, int)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = e[:, 0] * (len(V) + 1) + e[:, 1]
    und = np.sort(e, 1); ukey = und[:, 0] * (len(V) + 1) + und[:, 1]
    _, cnt = np.unique(ukey, return_counts=True)
    _, dcnt = np.unique(key, return_counts=True)
    vol = geom.mesh_volume(V, F)
    st = dict(tris=int(len(F)), verts=int(len(V)), open_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
              inconsistent_orientation_edges=int((dcnt > 1).sum()), signed_volume_mm3=round(vol, 2),
              bbox_min=jz(V.min(0), 3), bbox_max=jz(V.max(0), 3))
    if analytic_vol is not None:
        st["reference_volume_mm3"] = round(float(analytic_vol), 2)
        st["vol_err_pct"] = round(100.0 * (vol - analytic_vol) / analytic_vol, 2)
    st["closed_oriented"] = bool(st["open_edges"] == 0 and st["nonmanifold_edges"] == 0 and st["inconsistent_orientation_edges"] == 0 and vol > 0)
    return st


def ovoid_mesh(prm, side):
    """Lunar cap surface: implicit membership on a fine grid -> marching cubes -> sinc smoothing -> quadric decimation
    (prep_inputs.surface, the label->surface route of the package)."""
    import prep_inputs as pi
    h = val(prm, "mesh_voxel_mm")
    lo, hi = ovoid_bbox(prm, side)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    g = np.stack(np.meshgrid(*[lo[i] + h * np.arange(n[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    mask = ovoid_inside(g, prm, side).reshape(tuple(n))
    aff = np.diag([h, h, h, 1.0]); aff[:3, 3] = lo
    V, F, st0 = pi.surface(mask, aff, n_tris=val(prm, "mesh_target_tris"))
    st = mesh_stats(V, F, analytic_vol=float(mask.sum()) * h ** 3)
    st.update(mc_grid=[int(v) for v in n], mc_voxel_mm=h, pre_decimation_gate=st0["gate_ok"])
    return V, F, st


def build_meshes(prm):
    os.makedirs(APP, exist_ok=True)
    n_th = int(val(prm, "mesh_n_theta")); L = val(prm, "L_iu_mm"); r = val(prm, "r_tandem_mm")
    parts = {}
    # intrauterine tube: cylinder 0..L-r, hemispherical tip to L, flat disc at the flange
    rings = [((0.0, 0.0, z), EX, EY, r) for z in np.linspace(0.0, L - r, 18)]
    rings += [((0.0, 0.0, L - r + r * np.sin(ph)), EX, EY, r * np.cos(ph)) for ph in np.linspace(0, np.pi / 2, 8)[1:-1]]
    V, F = sweep(rings, (0.0, 0.0, 0.0), (0.0, 0.0, L), n_th)
    parts["tube"] = (V, F, mesh_stats(V, F, np.pi * r * r * (L - r) + 2.0 / 3.0 * np.pi * r ** 3))
    if val(prm, "shaft_style") == "straight":
        # the tandem rod: straight from the flange down the vaginal (shaft) axis, i.e. the tube axis rotated back by
        # angle_deg (the rod goes DOWN: d_rod = (0, sin th, -cos th) in the applicator frame)
        d_rod, y_rod = rod_frame(prm); rs = val(prm, "rod_r_mm"); Lr = val(prm, "rod_len_mm")
        rings = [(t * d_rod, EX, y_rod, rs) for t in np.linspace(0.0, Lr, 9)]
        V, F = sweep(rings, np.zeros(3), Lr * d_rod, n_th)
        parts["shaft"] = (V, F, mesh_stats(V, F, np.pi * rs * rs * Lr))
        if val(prm, "rods"):
            for (ox, oy), name in zip(val(prm, "ovoid_rod_offsets_mm"), ("rod_L", "rod_R")):
                p0 = ox * EX + oy * y_rod
                rings = [(p0 + t * d_rod, EX, y_rod, rs) for t in np.linspace(0.0, Lr, 9)]
                V, F = sweep(rings, p0, p0 + Lr * d_rod, n_th)
                parts[name] = (V, F, mesh_stats(V, F, np.pi * rs * rs * Lr))
            rp, t0, Lp = val(prm, "pack_r_mm"), val(prm, "pack_top_mm"), val(prm, "pack_len_mm")
            rnd, r_top = float(val(prm, "pack_top_round_mm")), float(val(prm, "pack_top_r_mm"))
            ts = list(np.linspace(t0, t0 + Lp, 15))
            if rnd > 0:                                    # extra stations through the shoulder
                ts += list(t0 + rnd * (1.0 - np.cos(np.linspace(0.0, np.pi / 2, 9))))
            ts = sorted(set(round(float(t), 6) for t in ts))

            def r_at(t):                                   # elliptical shoulder r_top -> rp over the top `rnd` mm
                if rnd <= 0 or t - t0 >= rnd:
                    return rp
                return r_top + (rp - r_top) * np.sqrt(max(0.0, 1.0 - ((rnd - (t - t0)) / rnd) ** 2))
            rings = [(t * d_rod, EX, y_rod, r_at(t)) for t in ts]
            V, F = sweep(rings, t0 * d_rod, (t0 + Lp) * d_rod, 48)
            parts["packing"] = (V, F, mesh_stats(V, F, np.pi * rp * rp * Lp))
    else:
        # curved shaft below the flange
        C, T = shaft_arc(prm, 30); rs = val(prm, "r_shaft_mm")
        rings = [(C[i], EX, np.cross(T[i], EX), rs) for i in range(len(C))]
        V, F = sweep(rings, C[0], C[-1], n_th)
        parts["shaft"] = (V, F, mesh_stats(V, F, np.pi * rs * rs * val(prm, "shaft_arc_len_mm")))
    for side, name in ((-1, "ovoid_L"), (+1, "ovoid_R")):
        parts[name] = ovoid_mesh(prm, side)
    out = {}
    for name, (V, F, st) in parts.items():
        fn = os.path.join(APP, name + ".obj")
        geom.write_obj(fn, V, F, header="%s of the Venezia-type applicator model; %s" % (name, FRAME_APP))
        st["file"] = os.path.basename(fn); out[name] = st
        print("[applicator] %-8s tris %5d verts %5d open %d nonmanifold %d badorient %d vol %8.1f mm3 (ref %8.1f, %+.2f %%) closed=%s"
              % (name, st["tris"], st["verts"], st["open_edges"], st["nonmanifold_edges"], st["inconsistent_orientation_edges"],
                 st["signed_volume_mm3"], st["reference_volume_mm3"], st["vol_err_pct"], st["closed_oriented"]), flush=True)
    return out


# ============================================================================ BT data and the ovoid fit
def load_bt():
    import evaluate as ev
    import nibabel as nib
    bt = ev.BT()
    bt.img = np.asarray(nib.load(P["data"] + "/BT_MRI.nii").dataobj).astype(np.float32)
    return bt


def fit_ovoids(prm, bt, do_fit=True, n_grid_report=6):
    """Fit (ovoid_diam, lateral offset, dz, AP tilt, dome height) of the two-cap rigid model to the BT ovoid label by
    voxel Dice on the BT grid; the pose (flange, axis) is the measured BT tandem.  The label also contains the vaginal
    packing (gauze), which the rigid model cannot represent: the leftover volume is reported as packing."""
    from scipy.optimize import minimize
    ov = bt.lab["ovoid"]; idx = np.argwhere(ov)
    lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 13, bt.shape)
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    W = g @ bt.aff[:3, :3].T + bt.aff[:3, 3]
    Q = to_app(bt.F, bt.R, W)
    lab = ov[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].ravel()
    n_lab = int(lab.sum())
    bounds = dict(ovoid_diam_mm=(30.0, 46.0), ovoid_offset_x_mm=(0.0, 12.0), ovoid_dz_mm=(-5.0, 3.0),
                  ovoid_tilt_deg=(-12.0, 12.0), ovoid_dome_mm=(3.0, val(prm, "ovoid_height_mm")))
    work = json.loads(json.dumps(prm))

    def set_x(x):
        for k, v in zip(FIT_KEYS, x):
            work[k]["value"] = float(v)

    def dice_x(x):
        set_x(x)
        m = device_inside(Q, work)
        return 2.0 * float((m & lab).sum()) / max(1, int(m.sum()) + n_lab)

    def obj(x):
        pen = 0.0
        for k, v in zip(FIT_KEYS, x):
            b = bounds[k]; pen += max(0.0, b[0] - v) + max(0.0, v - b[1])
        return -dice_x(np.clip(x, [bounds[k][0] for k in FIT_KEYS], [bounds[k][1] for k in FIT_KEYS])) + 0.05 * pen

    x0 = np.array([val(prm, k) for k in FIT_KEYS], float)
    rec = dict(method="voxel Dice on the BT grid, coarse grid search then Nelder-Mead; bounds", bounds=bounds,
               label_vox=n_lab, label_vol_cc=round(n_lab * bt.vv / 1000.0, 3), initial=dict(zip(FIT_KEYS, jz(x0))),
               initial_dice=round(dice_x(x0), 4))
    if do_fit:
        t0 = time.time()
        grid = dict(ovoid_diam_mm=[36.0, 38.0, 40.0, 42.0, 44.0], ovoid_offset_x_mm=[0.0, 2.0, 4.0, 6.0],
                    ovoid_dz_mm=[-3.0, -1.5, 0.0, 1.5], ovoid_tilt_deg=[-8.0, -4.0, 0.0, 4.0, 8.0],
                    ovoid_dome_mm=[5.0, 8.0, 11.0, 15.0, val(prm, "ovoid_height_mm")])
        best = (-1.0, x0); n_eval = 0; top = []
        for D in grid["ovoid_diam_mm"]:
            for xc in grid["ovoid_offset_x_mm"]:
                for dz in grid["ovoid_dz_mm"]:
                    for tl in grid["ovoid_tilt_deg"]:
                        for dm in grid["ovoid_dome_mm"]:
                            x = np.array([D, xc, dz, tl, dm]); d = dice_x(x); n_eval += 1
                            top.append((d, x))
                            if d > best[0]:
                                best = (d, x)
        top.sort(key=lambda t: -t[0])
        print("[ovoid fit] grid: %d evaluations in %.1f s, best Dice %.4f at %s" % (n_eval, time.time() - t0, best[0], jz(best[1], 2)), flush=True)
        res = minimize(obj, best[1], method="Nelder-Mead", options=dict(maxiter=600, xatol=0.02, fatol=1e-5, initial_simplex=None))
        xf = np.clip(res.x, [bounds[k][0] for k in FIT_KEYS], [bounds[k][1] for k in FIT_KEYS])
        rec.update(grid=grid, grid_evaluations=n_eval, grid_best=dict(dice=round(best[0], 4), x=dict(zip(FIT_KEYS, jz(best[1], 3)))),
                   grid_top=[dict(dice=round(d, 4), x=dict(zip(FIT_KEYS, jz(x, 3)))) for d, x in top[:n_grid_report]],
                   nelder_mead=dict(iterations=int(res.nit), evaluations=int(res.nfev), success=bool(res.success)),
                   wall_s=round(time.time() - t0, 1))
    else:
        xf = x0
        rec["note"] = "fit skipped (--no-fit): parameters taken from the stored applicator.json"
    xf = np.round(xf, 3); xf[np.abs(xf) < 1e-3] = 0.0
    set_x(xf)
    for k, v in zip(FIT_KEYS, xf):
        prm[k]["value"] = float(v)
    m = device_inside(Q, prm)
    dice = 2.0 * float((m & lab).sum()) / (int(m.sum()) + n_lab)
    ztop = val(prm, "ovoid_dz_mm"); zbot = ztop - val(prm, "ovoid_height_mm")
    zone = Q[:, 2] >= zbot - 3.0
    dz_zone = 2.0 * float((m & lab & zone).sum()) / max(1, int((m & zone).sum()) + int((lab & zone).sum()))
    left = lab & ~m; outside = m & ~lab
    Ql = Q[left]
    rec.update(fitted=dict(zip(FIT_KEYS, jz(xf, 3))), dice=round(dice, 4), dice_ovoid_zone=round(dz_zone, 4),
               ovoid_zone="label voxels with z_app >= cap bottom - 3 mm",
               model_vol_cc=round(int(m.sum()) * bt.vv / 1000.0, 3), overlap_cc=round(int((m & lab).sum()) * bt.vv / 1000.0, 3),
               leftover_label_minus_model_cc=round(int(left.sum()) * bt.vv / 1000.0, 3),
               leftover_below_cap_bottom_cc=round(int((left & (Q[:, 2] < zbot)).sum()) * bt.vv / 1000.0, 3),
               leftover_centroid_app_mm=jz(Ql.mean(0), 2) if len(Ql) else None,
               leftover_z_app_range_mm=jz([Ql[:, 2].min(), Ql[:, 2].max()], 1) if len(Ql) else None,
               model_outside_label_cc=round(int(outside.sum()) * bt.vv / 1000.0, 3),
               cap_apex_z_mm=round(ztop, 3), cap_bottom_z_mm=round(zbot, 3),
               assembly_extent_LR_mm=round(val(prm, "ovoid_diam_mm") + 2 * val(prm, "ovoid_offset_x_mm"), 2),
               assembly_extent_AP_mm=round(val(prm, "ovoid_diam_mm"), 2), assembly_extent_z_mm=round(val(prm, "ovoid_height_mm"), 2),
               interpretation="leftover = label volume not covered by the rigid two-cap model = vaginal packing (gauze) and "
                              "label margin; it is NOT part of the rigid device")
    print("[ovoid fit] Dice %.4f (ovoid zone %.4f); model %.2f cc, label %.2f cc, leftover %.2f cc (below the caps %.2f cc), "
          "model outside label %.2f cc; fitted %s" % (dice, dz_zone, rec["model_vol_cc"], rec["label_vol_cc"],
                                                     rec["leftover_label_minus_model_cc"], rec["leftover_below_cap_bottom_cc"],
                                                     rec["model_outside_label_cc"], rec["fitted"]), flush=True)
    return rec


# ============================================================================ plane sampling / figures
def plane_sampler(vol, aff, order):
    from scipy import ndimage as ndi
    inv = np.linalg.inv(aff); v = np.asarray(vol, np.float32)

    def f(X):
        X = np.asarray(X, float)
        ijk = ((X.reshape(-1, 3) - aff[:3, 3]) @ inv[:3, :3].T).T
        return ndi.map_coordinates(v, ijk, order=order, mode="constant").reshape(X.shape[:-1])
    return f


def plane_grid(kind, c, ur, vr, px=0.5):
    """World-coordinate plane through c: 'sag' (u=y, v=z), 'cor' (u=x, v=z), 'ax' (u=x, v=y)."""
    u = np.arange(ur[0], ur[1] + 1e-9, px); v = np.arange(vr[0], vr[1] + 1e-9, px)
    U, Vv = np.meshgrid(u, v)
    if kind == "sag":
        X = np.stack([np.full(U.shape, c[0]), U, Vv], -1)
    elif kind == "cor":
        X = np.stack([U, np.full(U.shape, c[1]), Vv], -1)
    else:
        X = np.stack([U, Vv, np.full(U.shape, c[2])], -1)
    return U, Vv, X


def app_plane(F, R, kind, c_app, ur, vr, px=0.5):
    """Plane in the applicator frame (x,y,z app coords) -> world points."""
    u = np.arange(ur[0], ur[1] + 1e-9, px); v = np.arange(vr[0], vr[1] + 1e-9, px)
    U, Vv = np.meshgrid(u, v)
    if kind == "ax":
        Qp = np.stack([U, Vv, np.full(U.shape, c_app)], -1)
    elif kind == "sag":
        Qp = np.stack([np.full(U.shape, c_app), U, Vv], -1)
    else:
        Qp = np.stack([U, np.full(U.shape, c_app), Vv], -1)
    return U, Vv, Qp, to_world(F, R, Qp.reshape(-1, 3)).reshape(Qp.shape)


def fig_ovoid_fit(prm, bt, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(FIGS, exist_ok=True)
    simg = plane_sampler(bt.img, bt.aff, 1); sov = plane_sampler(bt.lab["ovoid"], bt.aff, 0)
    srod = plane_sampler(bt.lab["applicator"] & ~bt.lab["ovoid"], bt.aff, 0); shr = plane_sampler(bt.lab["HR-CTV"], bt.aff, 0)
    sut = plane_sampler(bt.lab["uterus"], bt.aff, 0)
    zb = fit["cap_bottom_z_mm"]; zt = fit["cap_apex_z_mm"]
    panels = [("ax", zt - 3.0), ("ax", 0.5 * (zt + zb)), ("ax", zb + 3.0), ("ax", zb - 4.0),
              ("sag", 0.0), ("sag", 0.5 * (val(prm, "ovoid_diam_mm") / 2 + val(prm, "ovoid_offset_x_mm"))), ("cor", 0.0), ("cor", 12.0)]
    fig, axs = plt.subplots(2, 4, figsize=(19, 10.5))
    for ax, (kind, cc) in zip(axs.ravel(), panels):
        if kind == "ax":
            U, Vv, Qp, X = app_plane(bt.F, bt.R, "ax", cc, (-35, 35), (-35, 35)); xl, yl = "x_app (mm, patient right)", "y_app (mm, anterior)"
        elif kind == "sag":
            U, Vv, Qp, X = app_plane(bt.F, bt.R, "sag", cc, (-35, 35), (-45, 70)); xl, yl = "y_app (mm, anterior)", "z_app (mm, along the tube)"
        else:
            U, Vv, Qp, X = app_plane(bt.F, bt.R, "cor", cc, (-35, 35), (-45, 70)); xl, yl = "x_app (mm, patient right)", "z_app (mm)"
        im = simg(X); lo, hi = np.percentile(im[im > 0], [1, 99.5]) if (im > 0).any() else (0, 1)
        ext = [U.min(), U.max(), Vv.min(), Vv.max()]
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=ext)
        ax.contour(U, Vv, sov(X), [0.5], colors="orange", linewidths=1.4)
        ax.contour(U, Vv, srod(X), [0.5], colors="yellow", linewidths=0.8)
        ax.contour(U, Vv, shr(X), [0.5], colors="red", linewidths=0.7)
        ax.contour(U, Vv, sut(X), [0.5], colors="cyan", linewidths=0.7)
        mdl = device_inside(Qp.reshape(-1, 3), prm, with_tube=True).reshape(U.shape).astype(float)
        ax.contour(U, Vv, mdl, [0.5], colors="magenta", linewidths=1.6, linestyles="dashed")
        ax.set_title("%s plane at %s_app = %.1f mm" % ({"ax": "axial", "sag": "sagittal", "cor": "coronal"}[kind],
                                                        {"ax": "z", "sag": "x", "cor": "y"}[kind], cc), fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.set_aspect("equal"); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    hs = [Line2D([], [], color="orange", lw=1.4, label="BT ovoid label (device + packing)"),
          Line2D([], [], color="magenta", lw=1.6, ls="--", label="fitted rigid model (2 lunar caps + tube + shaft)"),
          Line2D([], [], color="yellow", lw=0.8, label="BT rod label"), Line2D([], [], color="red", lw=0.7, label="BT HR-CTV"),
          Line2D([], [], color="cyan", lw=0.7, label="BT uterus")]
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=9)
    f = fit["fitted"]
    fig.suptitle("Two-cap ovoid model fitted to the BT ovoid label (planes of the BT applicator frame): Dice %.3f (ovoid zone %.3f); "
                 "diam %.1f, lateral offset %.1f, dz %.1f mm, tilt %.1f deg, dome %.1f mm; leftover (packing) %.1f cc of %.1f cc"
                 % (fit["dice"], fit["dice_ovoid_zone"], f["ovoid_diam_mm"], f["ovoid_offset_x_mm"], f["ovoid_dz_mm"], f["ovoid_tilt_deg"],
                    f["ovoid_dome_mm"], fit["leftover_label_minus_model_cc"], fit["label_vol_cc"]), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fn = os.path.join(FIGS, "ovoid_fit.png"); fig.savefig(fn, dpi=85); plt.close(fig)
    print("[fig] wrote", fn, flush=True)
    return fn


# ============================================================================ preBT inputs and the pose rule
def load_pre():
    import nibabel as nib
    from scipy import ndimage as ndi
    D = P["data"]
    lab = {}
    aff = None
    for n in ("vagina", "uterus", "HR-CTV", "IUcanal", "bladder", "rectum", "sigmoid"):
        im = nib.load("%s/preBT_MRI_label_%s.nii" % (D, n)); lab[n] = np.asarray(im.dataobj) > 0
        aff = im.affine.copy() if aff is None else aff
    img = np.asarray(nib.load(D + "/preBT_MRI.nii").dataobj).astype(np.float32)
    po = json.load(open(P["inputs"] + "/poses.json")); cz = np.load(P["inputs"] + "/canal.npz")
    bdev = json.load(open(P["out"] + "/final/bdev.json"))
    canal = np.asarray(cz["pts"], float); s0 = np.asarray(cz["s0"], float)
    i_le, i_os = int(cz["i_L_end"]), int(cz["i_internal_os"])
    # vagina body as the meshing module: priority corpus > cervix > vagina, largest 6-connected component
    vag = lab["vagina"] & ~lab["uterus"] & ~lab["HR-CTV"]
    lm, k = ndi.label(vag, ndi.generate_binary_structure(3, 1))
    if k > 1:
        sizes = ndi.sum(vag, lm, range(1, k + 1)); vag = lm == (int(np.argmax(sizes)) + 1)
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0)); vv = float(np.prod(sp))
    Xv = np.argwhere(vag) @ aff[:3, :3].T + aff[:3, 3]
    c = Xv.mean(0); _, s, vt = np.linalg.svd(Xv - c, full_matrices=False)
    ax = vt[0] * (1.0 if vt[0] @ EZ >= 0 else -1.0)
    pr = (Xv - c) @ ax
    return dict(lab=lab, aff=aff, img=img, shape=lab["vagina"].shape, sp=sp, vv=vv,
                O_pre=np.asarray(po["O_pre"], float), L_end=np.asarray(po["L_end"], float), a0=geom.unit(po["a0"]),
                canal=canal, s0=s0, i_L_end=i_le, i_internal_os=i_os, internal_os=canal[i_os],
                canal_above_L_end_mm=float(s0[-1] - s0[i_le]), bdev=bdev,
                vagina=dict(mask=vag, X=Xv, centroid=c, axis=ax, proj_min=float(pr.min()), proj_max=float(pr.max()),
                            sigma_mm=(s / np.sqrt(len(Xv))), vol_cc=float(vag.sum() * vv / 1000.0)),
                uterus_X=np.argwhere(lab["uterus"]) @ aff[:3, :3].T + aff[:3, 3])


def vagina_fixed_point(pre, h_mm):
    v = pre["vagina"]; pr = (v["X"] - v["centroid"]) @ v["axis"]
    sel = pr <= v["proj_min"] + h_mm
    return v["X"][sel].mean(0), int(sel.sum())


def screw_decompose(R, t):
    """(R, t) -> screw axis (unit k, point p0), angle (deg), translation d along k:  T(p) = R (p - p0) + p0 + d k."""
    th = np.radians(geom.rot_angle_deg(R))
    if th < 1e-4:            # < 0.006 deg: treated as a pure translation (the neglected rotation moves a 50 mm lever by < 0.005 mm)
        return dict(pure_translation=True, axis=geom.unit(t), point=np.zeros(3), angle_deg=0.0, pitch_mm=float(np.linalg.norm(t)))
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    k = geom.unit(k)
    d = float(t @ k); tp = t - d * k
    p0 = 0.5 * (tp + (np.cross(k, tp)) / np.tan(th / 2))
    return dict(pure_translation=False, axis=k, point=p0, angle_deg=float(np.degrees(th)), pitch_mm=d)


def screw_interp(sc, s):
    """Rigid transform at fraction s of the screw motion (4x4)."""
    T = np.eye(4)
    if sc["pure_translation"]:
        T[:3, 3] = s * sc["pitch_mm"] * np.asarray(sc["axis"], float); return T
    k = np.asarray(sc["axis"], float); p0 = np.asarray(sc["point"], float)
    Rs = rodrigues(k, s * sc["angle_deg"])
    T[:3, :3] = Rs; T[:3, 3] = p0 - Rs @ p0 + s * sc["pitch_mm"] * k
    return T


def pose_rule(inputs, params):
    """Kinematic pose rule (CONTRACT 3.1-3.4) from preBT geometry + the chosen device.  Returns everything in the preBT
    world frame.  inputs: O_pre, L_end, a0, internal_os, canal_above_L_end_mm, vagina_fixed_point (3,), vagina_axis (3,),
    vagina_centroid (3,), [corpus_centroid].  params: the parameter dict of default_params.
    v1: shaft_axis_mode 'chord' (fixed point -> O_pre), flange at O_pre (flange_shift_mm 0).
    v2: shaft_axis_mode 'pca' (vagina principal axis) through O_pre or the vagina axis line, flange = base + Delta."""
    O = np.asarray(inputs["O_pre"], float); Pf = np.asarray(inputs["vagina_fixed_point"], float)
    L_end = np.asarray(inputs["L_end"], float); a0 = geom.unit(inputs["a0"]); ios = np.asarray(inputs["internal_os"], float)
    L = val(params, "L_iu_mm"); th = val(params, "angle_deg")
    # 3.1 shaft axis: v2 'pca' = principal axis of the vagina body (+S); v1 'chord' = fixed point -> external os
    mode = val(params, "shaft_axis_mode"); through = val(params, "shaft_line_through")
    a_shaft = geom.unit(inputs["vagina_axis"]) if mode == "pca" else geom.unit(O - Pf)
    if mode == "pca" and through == "vagina_axis":
        c = np.asarray(inputs["vagina_centroid"], float)
        base = c + float((O - c) @ a_shaft) * a_shaft          # external os projected onto the vagina's own axis line
    else:
        base = O.copy()
    # 3.2 tube axis: shaft axis rotated anteriorly by angle_deg
    n_ant = geom.ortho(EY, a_shaft)
    a_sag = np.cos(np.radians(th)) * a_shaft + np.sin(np.radians(th)) * n_ant          # in the plane (shaft axis, +y)
    a_lr = rot_x(-th) @ a_shaft if (rot_x(-th) @ a_shaft) @ EY >= (rot_x(th) @ a_shaft) @ EY else rot_x(th) @ a_shaft
    a_tube = geom.unit(a_lr if val(params, "tilt_axis") == "LR" else a_sag)
    if inputs.get("tube_axis_override") is not None:      # diagnostic variants only (e.g. tube axis = a0): shaft = tube tilted back
        a_tube = geom.unit(inputs["tube_axis_override"])
        a_shaft = geom.unit(rot_x(th) @ a_tube if (rot_x(th) @ a_tube) @ EY <= (rot_x(-th) @ a_tube) @ EY else rot_x(-th) @ a_tube)
        a_sag = a_lr = a_tube
    # 3.1b flange = base point + the seating shift Delta (scenario parameter) along the insertion (shaft) or the tube axis
    dz = float(val(params, "flange_shift_mm"))
    F = base + dz * (a_shaft if val(params, "shift_axis") == "shaft" else a_tube)
    R = app_frame_rows(a_tube); tip = F + L * a_tube
    # 3.3 corpus placement: a0 (through L_end) -> tube axis, L_end at d_F above the flange, minimal rotation
    d_F = L + val(params, "d_F_margin_mm") - float(inputs["canal_above_L_end_mm"])
    L_end_t = F + d_F * a_tube
    Rc = geom.rot_between(a0, a_tube); tc = L_end_t - Rc @ L_end
    T = np.eye(4); T[:3, :3] = Rc; T[:3, 3] = tc
    sc = screw_decompose(Rc, tc)
    assert np.allclose(screw_interp(sc, 1.0), T, atol=1e-6) and np.allclose(screw_interp(sc, 0.0), np.eye(4))
    # 3.4 insertion path
    axis_name = val(params, "path_axis"); n = int(val(params, "n_steps")); below = val(params, "tip_below_os_mm")
    paths = {}
    for pa in ("shaft", "tube"):
        ad = a_shaft if pa == "shaft" else a_tube
        D = float((tip - O) @ ad) + below                      # travel so that at u=0 the tip is `below` mm below the os along ad
        F0 = F - D * ad; tip0 = F0 + L * a_tube
        u_ios = float(np.clip(((ios - tip0) @ ad) / D, 0.0, 1.0))
        keys = []
        for k in range(n + 1):
            u = k / n
            Fu = F - (1 - u) * D * ad
            s_lin = 0.0 if u <= u_ios else (u - u_ios) / max(1e-9, 1 - u_ios)
            s = geom.smoothstep(s_lin) if val(params, "corpus_ease") == "smoothstep" else float(np.clip(s_lin, 0, 1))
            Tu = screw_interp(sc, s)
            keys.append(dict(u=round(u, 5), phase="approach" if u < u_ios else "insertion", F=jz(Fu, 4), a=jz(a_tube), x=jz(R[0]),
                             tip=jz(Fu + L * a_tube, 4), corpus_s=round(s, 5), corpus_T=jz(Tu, 6)))
        paths[pa] = dict(translation_axis=pa, axis=jz(ad), travel_mm=round(D, 3), u_tip_at_internal_os=round(u_ios, 4),
                         tip_offset_from_os_line_at_u0_mm=round(float(np.linalg.norm((tip0 - O) - ((tip0 - O) @ ad) * ad)), 2),
                         device_at_u0=dict(F=jz(F0, 4), tip=jz(tip0, 4)), keyframes=keys)
    C_arc, T_arc = shaft_arc(params, 31)
    out = dict(shaft_axis=a_shaft, shaft_base=base, flange=F, flange_shift_mm=dz, shaft_axis_mode=mode, shaft_line_through=through,
               shift_axis=val(params, "shift_axis"), tube_axis=a_tube, x_app=R[0], y_app=R[1], R_rows=R, tip=tip,
               tube_axis_sagittal_variant=a_sag, tilt_variants_diff_deg=geom.angle_deg(a_lr, a_sag),
               angle_shaft_tube_deg=geom.angle_deg(a_shaft, a_tube), shaft_end=to_world(F, R, C_arc[-1])[0],
               canal_above_L_end_mm=float(inputs["canal_above_L_end_mm"]),
               shaft_end_dir=(T_arc[-1] @ R), d_F_mm=d_F, L_end_target=L_end_t, corpus_R=Rc, corpus_t=tc, corpus_T=T,
               corpus_rotation_deg=geom.rot_angle_deg(Rc), screw=sc, insertion_path=paths[axis_name], insertion_path_alt=paths["tube" if axis_name == "shaft" else "shaft"])
    if "corpus_centroid" in inputs:
        c = np.asarray(inputs["corpus_centroid"], float); out["corpus_centroid_pre"] = c; out["corpus_centroid_target"] = Rc @ c + tc
    return out


# ============================================================================ validation against the real BT tandem
def canal_path_record(rule, pre, prm):
    """Stage 3 insertion path (geom.canal_path): tube up the VAGINAL axis to the vault, then a tip-pinned swing onto the
    final tube axis while the corpus is drawn from rest onto the rule's target for the current tube pose.  The record
    holds the parameters the scene re-evaluates, plus keyframes for the figure / audit."""
    vg = pre["vagina"]
    a_v = geom.unit(rule["shaft_axis"]); base = np.asarray(rule["shaft_base"], float)
    h_intro = float(vg["proj_min"] - float((base - vg["centroid"]) @ a_v))       # introitus height about base (< 0)
    L = val(prm, "L_iu_mm"); below = val(prm, "tip_below_os_mm")
    pars = dict(base=jz(base, 4), a_v=jz(a_v), F_fin=jz(rule["flange"], 4), a_fin=jz(rule["tube_axis"]), x_fin=jz(rule["x_app"]),
                L_iu_mm=L, h_intro_mm=round(h_intro, 3), below_mm=below, a0=jz(pre["a0"]), L_end=jz(pre["L_end"], 4), d_F_mm=round(rule["d_F_mm"], 3))
    n = int(val(prm, "n_steps")); keys = []
    for k in range(n + 1):
        u = k / n
        q = geom.canal_path(u, base, a_v, rule["flange"], rule["tube_axis"], L, h_intro, rule["x_app"], below)
        Tc = screw_interp(rule["screw"], q["w"])            # corpus: the final screw, weighted by the S2 progress
        keys.append(dict(u=round(u, 5), stage=q["stage"], w=round(q["w"], 5), F=jz(q["F"], 4), a=jz(q["a"]), x=jz(q["x"]), tip=jz(q["tip"], 4), corpus_T=jz(Tc, 6)))
    q1 = geom.canal_path(0.0, base, a_v, rule["flange"], rule["tube_axis"], L, h_intro, rule["x_app"], below)
    return dict(pars, u1=q1["u1"],
                definition="S1 (u < u1): tube along the vaginal axis up the axis line, tip from below the introitus to the vault, corpus at rest.  "
                           "S2: tip = lerp(vault, tip_final, w), axis = slerp(a_v, a_final, w), w = smoothstep; corpus = fraction w of the FINAL "
                           "screw (rest -> corpus.T_preBT_to_target).  End state = device_final and corpus.T_preBT_to_target exactly.",
                why="the preBT os is %.1f mm off the vaginal axis line and the lower canal %.1f deg from it: no straight tube lies in the vagina "
                    "and along the canal at once; the tandem goes up the vagina and the cervix is drawn onto it" % (
                        float(np.linalg.norm((pre["O_pre"] - base) - float((pre["O_pre"] - base) @ a_v) * a_v)), geom.angle_deg(pre["a0"], a_v)),
                keyframes=keys)


def load_frames():
    al = json.load(open(P["out"] + "/validation/alignment.json"))
    fr = {k: dict(R=np.array(v["R_BT_to_pre"]), t=np.array(v["t_BT_to_pre"]), role=v["role"], uses_labels=v["uses_labels"])
          for k, v in al["frames"].items() if k in ("BONE", "GLOBAL", "HR")}
    b = al["frames"]["BONE"]["BT_device_in_preBT"]
    fr["BONE"]["context"] = dict(real_flange_above_O_pre_along_a0_mm=b["flange_above_O_pre_along_a0_mm"],
                                 real_flange_above_L_end_along_a0_mm=b["flange_above_L_end_along_a0_mm"],
                                 real_flange_lateral_from_a0_line_mm=b["flange_lateral_from_a0_line_mm"],
                                 real_axis_angle_to_a0_deg=b["axis_angle_to_a0_deg"],
                                 uterus_centroid_shift_preBT_to_BT_mm=al["frames"]["BONE"]["preBT_centroid_to_BT_centroid_mm"]["uterus"],
                                 source="validation/alignment.json (BONE frame): the real BT device carried into preBT coordinates")
    return fr


def axis_error_components(ap, ab):
    """Signed sagittal (about LR; + = predicted more anterior) and coronal (about AP; + = predicted more to the patient's
    right) components of the angle between the predicted axis ap and the real axis ab (both in the BT frame), deg."""
    sag = np.degrees(np.arctan2(ap[1], ap[2]) - np.arctan2(ab[1], ab[2]))
    cor = np.degrees(np.arctan2(ap[0], ap[2]) - np.arctan2(ab[0], ab[2]))
    return dict(sagittal_deg=round(float(sag), 3), coronal_deg=round(float(cor), 3))


def compare_to_bt(rule, fr, bt, L):
    """Predicted device (preBT) -> BT frame by x_BT = R^T (y - t); compared with the real BT tandem."""
    R, t = fr["R"], fr["t"]
    to_bt = lambda y: (np.asarray(y, float) - t) @ R  # noqa: E731
    Fp = to_bt(rule["flange"]); ap = geom.unit(R.T @ rule["tube_axis"]); tp = Fp + L * ap
    d = Fp - bt.F; along = float(d @ bt.a); lat = float(np.linalg.norm(d - along * bt.a))
    dt = tp - bt.tip; talong = float(dt @ bt.a); tlat = float(np.linalg.norm(dt - talong * bt.a))
    return dict(axis_angle_deg=round(geom.angle_deg(ap, bt.a), 3), axis_error_components=axis_error_components(ap, bt.a),
                flange_offset_mm=dict(total=round(float(np.linalg.norm(d)), 3), along_BT_axis=round(along, 3), lateral=round(lat, 3),
                                      vector_BT=jz(d, 3)),
                tip_offset_mm=dict(total=round(float(np.linalg.norm(dt)), 3), along_BT_axis=round(talong, 3), lateral=round(tlat, 3)),
                predicted_flange_BT=jz(Fp, 3), predicted_axis_BT=jz(ap), predicted_tip_BT=jz(tp, 3))


def corpus_k0_metrics(rule, fr, bt, V0, F0, Xut_pre):
    """K0 corpus (preBT uterus moved rigidly by the rule) vs the BT uterus label, in the frame fr."""
    import evaluate as ev
    R, t = fr["R"], fr["t"]
    Vt = V0 @ rule["corpus_R"].T + rule["corpus_t"]
    m = ev.voxelize((Vt - t) @ R, F0, bt.shape, bt.aff)
    c_bt = (np.argwhere(bt.lab["uterus"]) @ bt.aff[:3, :3].T + bt.aff[:3, 3]).mean(0)
    c_k0 = ((Xut_pre @ rule["corpus_R"].T + rule["corpus_t"]) - t) @ R
    c_k0 = c_k0.mean(0)
    c_none = ((Xut_pre - t) @ R).mean(0)
    return dict(uterus_dice_vs_BT=round(ev.dice(bt.lab["uterus"], m), 4), centroid_err_mm=round(float(np.linalg.norm(c_k0 - c_bt)), 2),
                centroid_err_NONE_mm=round(float(np.linalg.norm(c_none - c_bt)), 2), vol_cc=round(float(m.sum() * bt.vv / 1000.0), 2)), m


def validate(prm, pre, bt, frames, rule_inputs, rule):
    L = val(prm, "L_iu_mm")
    V0, F0 = geom.read_obj(P["inputs"] + "/pre_uterus.obj")
    out = dict(frame="peri-organ MI frame 'BONE' of align.py (validation/alignment.json; no organ label used); "
                     "x_BT = R^T (y_pre - t); GLOBAL (whole-image MI) and HR (HR-CTV ICP, not held out) as frame sensitivity",
               real_BT_tandem=dict(flange_BT=jz(bt.F, 3), axis_BT=jz(bt.a), tip_BT=jz(bt.tip, 3), L_iu_mm=L),
               metrics="axis angle (deg) between the predicted tube axis and the BT tandem; flange-to-flange offset (mm) split "
                       "into the component along the BT axis (+ = predicted flange deeper/superior) and the lateral component; "
                       "same for the tip; corpus K0 = preBT uterus moved rigidly by the rule vs the BT uterus label (informational)")
    fb = frames["BONE"]
    main = compare_to_bt(rule, fb, bt, L)
    k0, m_k0 = corpus_k0_metrics(rule, fb, bt, V0, F0, pre["uterus_X"])
    main["corpus_K0"] = k0
    main["tilt_axis_variants_diff_deg"] = round(rule["tilt_variants_diff_deg"], 3)
    out["rule_v1"] = dict(angle_deg=val(prm, "angle_deg"), **main)
    out["frame_sensitivity"] = {k: compare_to_bt(rule, frames[k], bt, L) for k in ("GLOBAL", "HR")}
    out["between_session_context"] = fb["context"]
    # alternatives: device angle 0 and 30 deg (contract), plus diagnostics for a rule v2 (all preBT-only geometry except the oracle)
    alts = {}

    def score(r2, extra=None):
        a2 = compare_to_bt(r2, fb, bt, L); a2["corpus_K0"], _ = corpus_k0_metrics(r2, fb, bt, V0, F0, pre["uterus_X"])
        a2["corpus_centroid_shift_vs_rule_mm"] = round(float(np.linalg.norm(r2["corpus_centroid_target"] - rule["corpus_centroid_target"])), 2)
        a2["tube_axis_preBT"] = jz(r2["tube_axis"]); a2["shaft_axis_preBT"] = jz(r2["shaft_axis"])
        if extra:
            a2.update(extra)
        return a2

    for ang in (0.0, 30.0):
        p2 = json.loads(json.dumps(prm)); p2["angle_deg"]["value"] = ang
        alts["angle_%g" % ang] = score(pose_rule(rule_inputs, p2), dict(description="contract rule with the device angle set to %g deg" % ang))
    # oracle device angle (IN-SAMPLE diagnostic: the angle that minimises the BT axis error with the same shaft axis)
    best = None
    for ang in np.arange(0.0, 60.01, 1.0):
        p2 = json.loads(json.dumps(prm)); p2["angle_deg"]["value"] = float(ang)
        e = compare_to_bt(pose_rule(rule_inputs, p2), fb, bt, L)["axis_angle_deg"]
        if best is None or e < best[1]:
            best = (float(ang), e)
    p2 = json.loads(json.dumps(prm)); p2["angle_deg"]["value"] = best[0]
    alts["angle_oracle"] = score(pose_rule(rule_inputs, p2), dict(description="IN-SAMPLE diagnostic: device angle minimising the BT axis error "
                                                                              "(same vaginal-chord shaft axis); the residual is the coronal (LR) error",
                                                                  angle_deg=best[0]))
    # shaft axis = the vagina body's principal axis through O_pre (preBT-only; replaces the chord to the fixed point)
    vg = pre["vagina"]; chord = float(np.linalg.norm(rule_inputs["O_pre"] - np.asarray(rule_inputs["vagina_fixed_point"])))
    inp = dict(rule_inputs); inp["vagina_fixed_point"] = np.asarray(rule_inputs["O_pre"], float) - chord * vg["axis"]
    alts["shaft_axis_vagina_pca"] = score(pose_rule(inp, prm), dict(description="preBT-only variant: shaft axis = principal axis of the vagina body "
                                                                            "(oriented +S) through O_pre instead of the chord to the fixed point",
                                                                    vagina_axis=jz(vg["axis"]),
                                                                    chord_vs_pca_axis_deg=round(geom.angle_deg(rule["shaft_axis"], vg["axis"]), 3),
                                                                    O_pre_lateral_from_vagina_axis_line_mm=round(float(np.linalg.norm(
                                                                        (rule_inputs["O_pre"] - vg["centroid"]) - ((rule_inputs["O_pre"] - vg["centroid"]) @ vg["axis"]) * vg["axis"])), 2)))
    # tube axis = the preBT cervical canal axis a0 through O_pre (preBT-only)
    inp = dict(rule_inputs); inp["tube_axis_override"] = np.asarray(rule_inputs["a0"], float)
    alts["tube_axis_a0"] = score(pose_rule(inp, prm), dict(description="preBT-only variant: tube axis = lower cervical canal axis a0 through O_pre "
                                                                     "(the device angle then only sets the shaft/insertion direction)"))
    out["alternatives"] = alts
    # sensitivity: vaginal fixed point +/- 5 mm along x, y, z
    sens = []
    for axn, e in (("x", EX), ("y", EY), ("z", EZ)):
        for sgn in (+5.0, -5.0):
            inp = dict(rule_inputs); inp["vagina_fixed_point"] = np.asarray(rule_inputs["vagina_fixed_point"], float) + sgn * e
            r2 = pose_rule(inp, prm); a2 = compare_to_bt(r2, fb, bt, L)
            sens.append(dict(delta_mm={axn: sgn}, shaft_axis_change_deg=round(geom.angle_deg(r2["shaft_axis"], rule["shaft_axis"]), 3),
                             tube_axis_change_deg=round(geom.angle_deg(r2["tube_axis"], rule["tube_axis"]), 3),
                             BT_axis_angle_deg=a2["axis_angle_deg"], BT_flange_offset_total_mm=a2["flange_offset_mm"]["total"],
                             corpus_rotation_change_deg=round(geom.rot_angle_deg(r2["corpus_R"] @ rule["corpus_R"].T), 3),
                             corpus_centroid_shift_mm=round(float(np.linalg.norm(r2["corpus_centroid_target"] - rule["corpus_centroid_target"])), 3),
                             tip_shift_mm=round(float(np.linalg.norm(r2["tip"] - rule["tip"])), 3)))
    out["sensitivity_vaginal_fixed_point_pm5mm"] = dict(
        perturbations=sens, max_tube_axis_change_deg=max(s["tube_axis_change_deg"] for s in sens),
        max_corpus_centroid_shift_mm=max(s["corpus_centroid_shift_mm"] for s in sens),
        max_tip_shift_mm=max(s["tip_shift_mm"] for s in sens),
        BT_axis_angle_range_deg=[min(s["BT_axis_angle_deg"] for s in sens), max(s["BT_axis_angle_deg"] for s in sens)],
        note="the flange is pinned to the external os, so the fixed point only steers the axes (lever arm = vaginal chord length)")
    out["vaginal_chord_length_mm"] = round(float(np.linalg.norm(rule_inputs["O_pre"] - np.asarray(rule_inputs["vagina_fixed_point"]))), 2)
    print("[validate] BONE: axis angle %.2f deg, flange offset %.2f mm (along %.2f, lateral %.2f), tip offset %.2f mm; corpus K0 Dice %.3f "
          "(centroid err %.1f mm; NONE %.1f mm)" % (main["axis_angle_deg"], main["flange_offset_mm"]["total"], main["flange_offset_mm"]["along_BT_axis"],
                                                   main["flange_offset_mm"]["lateral"], main["tip_offset_mm"]["total"], k0["uterus_dice_vs_BT"],
                                                   k0["centroid_err_mm"], k0["centroid_err_NONE_mm"]), flush=True)
    for k, a in alts.items():
        print("[validate] %-22s axis angle %.2f deg (sag %+.1f, cor %+.1f), flange offset %.2f (along %.2f, lat %.2f), K0 Dice %.3f centroid err %.1f" % (
            k, a["axis_angle_deg"], a["axis_error_components"]["sagittal_deg"], a["axis_error_components"]["coronal_deg"], a["flange_offset_mm"]["total"],
            a["flange_offset_mm"]["along_BT_axis"], a["flange_offset_mm"]["lateral"], a["corpus_K0"]["uterus_dice_vs_BT"], a["corpus_K0"]["centroid_err_mm"]), flush=True)
    print("[validate] fixed point +/-5 mm: tube axis change <= %.2f deg, corpus centroid shift <= %.2f mm, BT angle range %s" % (
        out["sensitivity_vaginal_fixed_point_pm5mm"]["max_tube_axis_change_deg"], out["sensitivity_vaginal_fixed_point_pm5mm"]["max_corpus_centroid_shift_mm"],
        out["sensitivity_vaginal_fixed_point_pm5mm"]["BT_axis_angle_range_deg"]), flush=True)
    return out, m_k0


# ============================================================================ rule v2: sweep and score
def rule_v2_score(prm, pre, bt, frames, rule_inputs, through, shift_axis, dz, V0, F0):
    """One v2 variant at one Delta: rule -> BT-frame score (compare_to_bt) + K0 corpus metrics."""
    p2 = json.loads(json.dumps(prm))
    p2["shaft_axis_mode"]["value"] = "pca"; p2["shaft_line_through"]["value"] = through
    p2["shift_axis"]["value"] = shift_axis; p2["flange_shift_mm"]["value"] = float(dz)
    r2 = pose_rule(rule_inputs, p2)
    sc = compare_to_bt(r2, frames["BONE"], bt, val(prm, "L_iu_mm"))
    sc["corpus_K0"], m = corpus_k0_metrics(r2, frames["BONE"], bt, V0, F0, pre["uterus_X"])
    sc.update(flange_shift_mm=float(dz), corpus_rotation_deg=round(r2["corpus_rotation_deg"], 3),
              corpus_centroid_shift_mm=round(float(np.linalg.norm(r2["corpus_centroid_target"] - r2["corpus_centroid_pre"])), 2),
              shaft_axis_preBT=jz(r2["shaft_axis"]), tube_axis_preBT=jz(r2["tube_axis"]), flange_preBT=jz(r2["flange"], 3))
    return sc, r2, m


def validate_v2(prm, pre, bt, frames, rule_inputs, valid_v1):
    """Rule v2 scored against the real BT tandem (BONE frame): (line placement) x (shift axis) x Delta sweep, then a fine
    Delta scan for the chosen variant.  Delta is a SCENARIO parameter: its best value here is in-sample."""
    V0, F0 = geom.read_obj(P["inputs"] + "/pre_uterus.obj")
    sweep = [float(d) for d in val(prm, "flange_shift_sweep_mm")]
    variants = {}
    for through in ("O_pre", "vagina_axis"):
        for shift_axis in ("shaft", "tube"):
            rows = [rule_v2_score(prm, pre, bt, frames, rule_inputs, through, shift_axis, dz, V0, F0)[0] for dz in sweep]
            best = min(rows, key=lambda s: s["corpus_K0"]["centroid_err_mm"])
            variants["pca_%s_%s" % (through, shift_axis)] = dict(shaft_line_through=through, shift_axis=shift_axis, rows=rows, best=best)
            print("[rule v2] line through %-11s shift along %-5s: axis angle %.2f deg (sag %+.1f, cor %+.1f); best Delta %.0f mm -> flange offset "
                  "%.1f mm (along %+.1f, lateral %.1f), K0 corpus Dice %.3f, centroid err %.1f mm" % (
                      through, shift_axis, best["axis_angle_deg"], best["axis_error_components"]["sagittal_deg"], best["axis_error_components"]["coronal_deg"],
                      best["flange_shift_mm"], best["flange_offset_mm"]["total"], best["flange_offset_mm"]["along_BT_axis"], best["flange_offset_mm"]["lateral"],
                      best["corpus_K0"]["uterus_dice_vs_BT"], best["corpus_K0"]["centroid_err_mm"]), flush=True)
    name = min(variants, key=lambda k: (variants[k]["best"]["corpus_K0"]["centroid_err_mm"], variants[k]["best"]["flange_offset_mm"]["lateral"]))
    ch = variants[name]
    fine_dz = [float(d) for d in np.arange(0.0, 45.01, 2.5)]
    fine = [rule_v2_score(prm, pre, bt, frames, rule_inputs, ch["shaft_line_through"], ch["shift_axis"], dz, V0, F0)[0] for dz in fine_dz]
    fbest = min(fine, key=lambda s: s["corpus_K0"]["centroid_err_mm"])
    fine_tab = [dict(flange_shift_mm=s["flange_shift_mm"], K0_dice=s["corpus_K0"]["uterus_dice_vs_BT"], K0_centroid_err_mm=s["corpus_K0"]["centroid_err_mm"],
                     flange_along_mm=s["flange_offset_mm"]["along_BT_axis"], flange_lateral_mm=s["flange_offset_mm"]["lateral"],
                     tip_offset_mm=s["tip_offset_mm"]["total"]) for s in fine]
    v1 = valid_v1["rule_v1"]
    out = dict(definition="shaft axis = principal axis of the preBT vagina body (+S); tube axis = shaft axis rotated angle_deg anteriorly about LR; "
                          "flange = base point (O_pre, or O_pre projected onto the vagina axis line) + Delta along the shift axis; corpus: a0 -> tube "
                          "axis, L_end at d_F above the flange (unchanged). Delta = seating shift of the ovoids/packing = a SCENARIO parameter",
               frame=valid_v1["frame"], sweep_mm=sweep, variants=variants,
               chosen=dict(variant=name, shaft_line_through=ch["shaft_line_through"], shift_axis=ch["shift_axis"],
                           selection="IN-SAMPLE: variant and Delta minimising the K0 corpus centroid error on this case (ties: smaller lateral flange error)",
                           coarse_best_flange_shift_mm=ch["best"]["flange_shift_mm"], fine_best_flange_shift_mm=fbest["flange_shift_mm"],
                           fine_step_mm=2.5, fine_table=fine_tab, at_best=fbest),
               v1_reference=dict(axis_angle_deg=v1["axis_angle_deg"], flange_offset_mm=v1["flange_offset_mm"], corpus_K0=v1["corpus_K0"]),
               residual_at_best=dict(axis_angle_deg=fbest["axis_angle_deg"], axis_error_components=fbest["axis_error_components"],
                                     flange_lateral_mm=fbest["flange_offset_mm"]["lateral"], flange_along_mm=fbest["flange_offset_mm"]["along_BT_axis"],
                                     tip_offset_mm=fbest["tip_offset_mm"], K0=fbest["corpus_K0"],
                                     reading="what remains once the seating shift is right: the axis error (deg) and the perpendicular line offset (mm)"))
    print("[rule v2] chosen %s, fine best Delta %.1f mm: axis angle %.2f deg, flange along %+.1f / lateral %.1f mm, tip offset %.1f mm, K0 Dice %.3f, "
          "centroid err %.1f mm (v1: %.1f mm, NONE %.1f mm)" % (
              name, fbest["flange_shift_mm"], fbest["axis_angle_deg"], fbest["flange_offset_mm"]["along_BT_axis"], fbest["flange_offset_mm"]["lateral"],
              fbest["tip_offset_mm"]["total"], fbest["corpus_K0"]["uterus_dice_vs_BT"], fbest["corpus_K0"]["centroid_err_mm"],
              v1["corpus_K0"]["centroid_err_mm"], v1["corpus_K0"]["centroid_err_NONE_mm"]), flush=True)
    return out


# ============================================================================ pose-rule figures
def _device_mask_on_plane(X, F, R, prm, with_tube=True):
    Qp = to_app(F, R, X.reshape(-1, 3))
    return device_inside(Qp, prm, with_tube=with_tube).reshape(X.shape[:-1]).astype(float)


def _line(ax, a, b, i, j, **kw):
    ax.plot([a[i], b[i]], [a[j], b[j]], **kw)


ALT_STYLE = {"angle_0": ("white", "dotted", "rule with angle 0"), "angle_30": ("white", "dashdot", "rule with angle 30"),
             "vagina_pca": ("springgreen", "dashed", "diagnostic: shaft axis = vagina principal axis"),
             "tube_a0": ("deepskyblue", "dashed", "diagnostic: tube axis = canal axis a0"),
             "v1": ("white", "dotted", "rule v1 (rejected): vaginal chord, flange at O_pre")}


def fig_pose_rule_bt(prm, pre, bt, frames, rule, valid, m_k0, alt_rules, score=None, label="rule v1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import evaluate as ev
    fb = frames["BONE"]; R, t = fb["R"], fb["t"]
    to_bt = lambda y: (np.asarray(y, float) - t) @ R  # noqa: E731
    L = val(prm, "L_iu_mm")
    Fp = to_bt(rule["flange"]); ap = geom.unit(R.T @ rule["tube_axis"])
    xp = geom.ortho(R.T @ rule["x_app"], ap); Rp = np.stack([xp, np.cross(ap, xp), ap])
    V0u, F0u = geom.read_obj(P["inputs"] + "/pre_uterus.obj"); V0h, F0h = geom.read_obj(P["inputs"] + "/pre_hrctv.obj")
    m_none_ut = ev.voxelize(to_bt(V0u), F0u, bt.shape, bt.aff); m_none_hr = ev.voxelize(to_bt(V0h), F0h, bt.shape, bt.aff)
    simg = plane_sampler(bt.img, bt.aff, 1)
    S = {n: plane_sampler(bt.lab[n], bt.aff, 0) for n in ("uterus", "HR-CTV", "vagina", "applicator", "ovoid")}
    s_k0 = plane_sampler(m_k0, bt.aff, 0); s_nu = plane_sampler(m_none_ut, bt.aff, 0); s_nh = plane_sampler(m_none_hr, bt.aff, 0)
    c = bt.F
    fig, axs = plt.subplots(1, 2, figsize=(17, 9.5))
    for ax, (kind, i, j, ur, vr, xl) in zip(axs, (("sag", 1, 2, (c[1] - 65, c[1] + 65), (c[2] - 75, c[2] + 80), "y (mm, anterior ->)"),
                                                  ("cor", 0, 2, (c[0] - 65, c[0] + 65), (c[2] - 75, c[2] + 80), "x (mm, patient right ->)"))):
        U, Vv, X = plane_grid(kind, c, ur, vr)
        im = simg(X); lo, hi = np.percentile(im[im > 0], [1, 99.5])
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[U.min(), U.max(), Vv.min(), Vv.max()])
        for n, col, lw in (("uterus", "cyan", 1.3), ("HR-CTV", "red", 1.1), ("vagina", "lime", 0.6), ("applicator", "yellow", 0.9), ("ovoid", "orange", 0.9)):
            ax.contour(U, Vv, S[n](X), [0.5], colors=col, linewidths=lw)
        ax.contour(U, Vv, s_nu(X), [0.5], colors="cyan", linewidths=0.7, linestyles="dotted")
        ax.contour(U, Vv, s_nh(X), [0.5], colors="red", linewidths=0.7, linestyles="dotted")
        ax.contour(U, Vv, s_k0(X), [0.5], colors="cyan", linewidths=2.0, linestyles="dashed")
        ax.contour(U, Vv, _device_mask_on_plane(X, Fp, Rp, prm), [0.5], colors="magenta", linewidths=1.2, linestyles="dashed")
        ax.contour(U, Vv, _device_mask_on_plane(X, bt.F, bt.R, prm), [0.5], colors="orange", linewidths=0.9, linestyles="dashed")
        _line(ax, bt.F, bt.tip, i, j, color="yellow", lw=3.0, label="real BT tandem")
        _line(ax, Fp, Fp + L * ap, i, j, color="magenta", lw=2.4, label="predicted (%s)" % label)
        for nm, r2 in alt_rules.items():
            F2 = to_bt(r2["flange"]); a2 = geom.unit(R.T @ r2["tube_axis"])
            _line(ax, F2, F2 + L * a2, i, j, color=ALT_STYLE[nm][0], lw=1.3, ls=ALT_STYLE[nm][1])
        ax.plot(bt.F[i], bt.F[j], "o", color="yellow", ms=6); ax.plot(Fp[i], Fp[j], "o", color="magenta", ms=6)
        ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel("z (mm, superior ->)")
        ax.set_title("BT %s slice through the real flange (projected lines)" % {"sag": "sagittal", "cor": "coronal"}[kind], fontsize=10)
        ax.tick_params(labelsize=8)
    hs = [Line2D([], [], color="yellow", lw=3, label="real BT tandem (flange->tip)"),
          Line2D([], [], color="magenta", lw=2.4, label="predicted tandem, %s" % label),
          *[Line2D([], [], color=ALT_STYLE[nm][0], lw=1.3, ls=ALT_STYLE[nm][1], label=ALT_STYLE[nm][2]) for nm in alt_rules],
          Line2D([], [], color="magenta", lw=1.2, ls="--", label="predicted device (caps+shaft)"),
          Line2D([], [], color="orange", lw=0.9, ls="--", label="fitted caps at the real pose"),
          Line2D([], [], color="cyan", lw=1.3, label="BT uterus"), Line2D([], [], color="cyan", lw=1.6, ls="--", label="K0 corpus (rule)"),
          Line2D([], [], color="cyan", lw=0.9, ls=":", label="NONE uterus (preBT, BONE)"),
          Line2D([], [], color="red", lw=1.1, label="BT HR-CTV"), Line2D([], [], color="red", lw=0.9, ls=":", label="NONE HR-CTV"),
          Line2D([], [], color="orange", lw=0.9, label="BT ovoid label"), Line2D([], [], color="lime", lw=0.6, label="BT vagina")]
    fig.legend(handles=hs, loc="lower center", ncol=7, fontsize=8)
    v = score if score is not None else valid["rule_v1"]
    fig.suptitle("Pose %s vs the real BT tandem in the peri-organ MI frame (BONE): axis angle %.1f deg; flange offset %.1f mm "
                 "(%.1f along the BT axis, %.1f lateral); tip offset %.1f mm; K0 corpus Dice %.3f (NONE centroid err %.1f mm -> K0 %.1f mm)"
                 % (label, v["axis_angle_deg"], v["flange_offset_mm"]["total"], v["flange_offset_mm"]["along_BT_axis"], v["flange_offset_mm"]["lateral"],
                    v["tip_offset_mm"]["total"], v["corpus_K0"]["uterus_dice_vs_BT"], v["corpus_K0"]["centroid_err_NONE_mm"], v["corpus_K0"]["centroid_err_mm"]),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fn = os.path.join(FIGS, "pose_rule_BT.png"); fig.savefig(fn, dpi=85); plt.close(fig)
    print("[fig] wrote", fn, flush=True)


def fig_pose_rule_pre(prm, pre, rule, rule_inputs, m_corpus_target, label="rule v1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    simg = plane_sampler(pre["img"], pre["aff"], 1)
    S = {n: plane_sampler(pre["lab"][n], pre["aff"], 0) for n in ("uterus", "HR-CTV", "vagina", "bladder", "rectum", "sigmoid")}
    s_vag = plane_sampler(pre["vagina"]["mask"], pre["aff"], 0); s_ct = plane_sampler(m_corpus_target, pre["aff"], 0)
    O = rule["flange"]; Pf = np.asarray(rule_inputs["vagina_fixed_point"], float); L = val(prm, "L_iu_mm")
    F, R = rule["flange"], rule["R_rows"]
    p0 = rule["insertion_path"]["device_at_u0"]; F0 = np.asarray(p0["F"], float)
    C_arc, _ = shaft_arc(prm, 31); arc_w = to_world(F, R, C_arc)
    fig, axs = plt.subplots(1, 2, figsize=(17, 9.5))
    for ax, (kind, i, j, ur, vr, xl) in zip(axs, (("sag", 1, 2, (O[1] - 70, O[1] + 70), (O[2] - 75, O[2] + 85), "y (mm, anterior ->)"),
                                                  ("cor", 0, 2, (O[0] - 70, O[0] + 70), (O[2] - 75, O[2] + 85), "x (mm, patient right ->)"))):
        U, Vv, X = plane_grid(kind, O, ur, vr)
        im = simg(X); lo, hi = np.percentile(im[im > 0], [1, 99.5])
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[U.min(), U.max(), Vv.min(), Vv.max()])
        for n, col, lw in (("uterus", "cyan", 1.3), ("HR-CTV", "red", 1.1), ("bladder", "dodgerblue", 0.7), ("rectum", "sienna", 0.7), ("sigmoid", "grey", 0.6)):
            ax.contour(U, Vv, S[n](X), [0.5], colors=col, linewidths=lw)
        ax.contour(U, Vv, s_vag(X), [0.5], colors="lime", linewidths=1.0)
        ax.contour(U, Vv, s_ct(X), [0.5], colors="cyan", linewidths=1.6, linestyles="dashed")
        ax.contour(U, Vv, _device_mask_on_plane(X, F, R, prm), [0.5], colors="magenta", linewidths=1.3, linestyles="dashed")
        ax.contour(U, Vv, _device_mask_on_plane(X, F0, R, prm), [0.5], colors="magenta", linewidths=0.7, linestyles="dotted")
        sb = np.asarray(rule["shaft_base"], float)
        _line(ax, sb - 60.0 * rule["shaft_axis"], sb, i, j, color="lime", lw=1.6, ls="--")
        ax.plot(sb[i], sb[j], "D", color="lime", ms=5)
        _line(ax, F, rule["tip"], i, j, color="magenta", lw=2.6)
        _line(ax, F0, F0 + L * rule["tube_axis"], i, j, color="magenta", lw=1.0, ls=":")
        ax.plot(arc_w[:, i], arc_w[:, j], color="magenta", lw=2.0)
        ax.plot(Pf[i], Pf[j], "*", color="lime", ms=12); ax.plot(O[i], O[j], "o", color="magenta", ms=6)
        ax.plot(rule["L_end_target"][i], rule["L_end_target"][j], "s", color="cyan", ms=5)
        le = np.asarray(rule_inputs["L_end"]); ios = np.asarray(rule_inputs["internal_os"])
        ax.plot(le[i], le[j], "^", color="white", ms=6); ax.plot(ios[i], ios[j], "v", color="white", ms=6)
        ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel("z (mm, superior ->)")
        ax.set_title("preBT %s slice through the external os O_pre (projected lines)" % {"sag": "sagittal", "cor": "coronal"}[kind], fontsize=10)
        ax.tick_params(labelsize=8)
    hs = [Line2D([], [], color="lime", lw=1.6, ls="--", label="shaft axis line (mode '%s', base point D; fixed point *, O_pre o)" % rule["shaft_axis_mode"]),
          Line2D([], [], color="magenta", lw=2.6, label="predicted tube (flange = base + Delta %.1f mm -> tip), shaft arc" % rule["flange_shift_mm"]),
          Line2D([], [], color="magenta", lw=1.3, ls="--", label="device caps at the final pose"),
          Line2D([], [], color="magenta", lw=0.8, ls=":", label="device at u = 0 (path start)"),
          Line2D([], [], color="cyan", lw=1.3, label="preBT uterus (corpus)"), Line2D([], [], color="cyan", lw=1.6, ls="--", label="target corpus pose (rule)"),
          Line2D([], [], color="red", lw=1.1, label="preBT HR-CTV"), Line2D([], [], color="lime", lw=1.0, label="vagina body"),
          Line2D([], [], color="dodgerblue", lw=0.7, label="bladder"), Line2D([], [], color="sienna", lw=0.7, label="rectum"),
          Line2D([], [], color="white", marker="^", ls="", label="L_end"), Line2D([], [], color="white", marker="v", ls="", label="internal os"),
          Line2D([], [], color="cyan", marker="s", ls="", label="L_end target")]
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Pose %s in the preBT frame: shaft axis mode '%s' (line through %s), seating shift Delta %.1f mm along the %s axis, tube rotated "
                 "%g deg anteriorly, corpus moved rigidly (a0 -> tube axis, L_end %.1f mm above the flange, rotation %.1f deg, centroid shift %.1f mm)"
                 % (label, rule["shaft_axis_mode"], rule["shaft_line_through"], rule["flange_shift_mm"], rule["shift_axis"], val(prm, "angle_deg"),
                    rule["d_F_mm"], rule["corpus_rotation_deg"], float(np.linalg.norm(rule["corpus_centroid_target"] - rule["corpus_centroid_pre"]))),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fn = os.path.join(FIGS, "pose_rule_preBT.png"); fig.savefig(fn, dpi=85); plt.close(fig)
    print("[fig] wrote", fn, flush=True)


def fig_pose_rule_path(prm, pre, rule, rule_inputs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import evaluate as ev
    path = rule["insertion_path"]; keys = path["keyframes"]; u_ios = path["u_tip_at_internal_os"]
    us = sorted(set([0.0, 0.25, round(u_ios, 4), round(0.5 * (u_ios + 1), 4), 0.9, 1.0]))
    V0, F0 = geom.read_obj(P["inputs"] + "/pre_uterus.obj")
    simg = plane_sampler(pre["img"], pre["aff"], 1)
    S = {n: plane_sampler(pre["lab"][n], pre["aff"], 0) for n in ("uterus", "HR-CTV", "bladder", "rectum")}
    s_vag = plane_sampler(pre["vagina"]["mask"], pre["aff"], 0)
    O = rule["flange"]; L = val(prm, "L_iu_mm"); R = rule["R_rows"]
    U, Vv, X = plane_grid("sag", O, (O[1] - 60, O[1] + 60), (O[2] - 70, O[2] + 85))
    im = simg(X); lo, hi = np.percentile(im[im > 0], [1, 99.5])
    fig, axs = plt.subplots(1, len(us), figsize=(3.6 * len(us), 7.5))
    for ax, u in zip(axs, us):
        k = min(keys, key=lambda q: abs(q["u"] - u))
        T = np.asarray(k["corpus_T"], float); Fu = np.asarray(k["F"], float)
        m = ev.voxelize(V0 @ T[:3, :3].T + T[:3, 3], F0, pre["shape"], pre["aff"])
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[U.min(), U.max(), Vv.min(), Vv.max()])
        for n, col, lw in (("uterus", "cyan", 0.7), ("HR-CTV", "red", 0.9), ("bladder", "dodgerblue", 0.5), ("rectum", "sienna", 0.5)):
            ax.contour(U, Vv, S[n](X), [0.5], colors=col, linewidths=lw, linestyles="dotted" if n == "uterus" else "solid")
        ax.contour(U, Vv, s_vag(X), [0.5], colors="lime", linewidths=0.8)
        ax.contour(U, Vv, plane_sampler(m, pre["aff"], 0)(X), [0.5], colors="cyan", linewidths=1.5)
        ax.contour(U, Vv, _device_mask_on_plane(X, Fu, R, prm), [0.5], colors="magenta", linewidths=1.2)
        _line(ax, Fu, Fu + L * rule["tube_axis"], 1, 2, color="magenta", lw=2.2)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        ax.set_title("u = %.3f (%s)\ncorpus s = %.2f" % (k["u"], k["phase"], k["corpus_s"]), fontsize=9)
    fig.suptitle("Insertion path (translation along the %s axis, %d keyframes): device (magenta) and corpus (cyan; dotted = preBT rest) on the preBT "
                 "sagittal slice through O_pre; corpus screw motion starts when the tip reaches the internal-os level (u = %.3f)"
                 % (path["translation_axis"], len(keys), u_ios), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fn = os.path.join(FIGS, "pose_rule_path.png"); fig.savefig(fn, dpi=85); plt.close(fig)
    print("[fig] wrote", fn, flush=True)


def fig_delta_sweep(v2, label):
    """K0 corpus error, flange offsets and axis angle vs the seating shift Delta, per v2 variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    cols = {"pca_O_pre_shaft": "tab:red", "pca_O_pre_tube": "tab:orange", "pca_vagina_axis_shaft": "tab:blue", "pca_vagina_axis_tube": "tab:cyan"}
    for name, vv in v2["variants"].items():
        rows = vv["rows"]; dz = [r["flange_shift_mm"] for r in rows]
        axs[0].plot(dz, [r["corpus_K0"]["centroid_err_mm"] for r in rows], "o-", color=cols.get(name, "k"), label=name)
        axs[1].plot(dz, [r["flange_offset_mm"]["along_BT_axis"] for r in rows], "o-", color=cols.get(name, "k"), label=name + " along")
        axs[1].plot(dz, [r["flange_offset_mm"]["lateral"] for r in rows], "s--", color=cols.get(name, "k"), label=name + " lateral")
        axs[2].plot(dz, [r["corpus_K0"]["uterus_dice_vs_BT"] for r in rows], "o-", color=cols.get(name, "k"), label=name)
    ft = v2["chosen"]["fine_table"]
    axs[0].plot([r["flange_shift_mm"] for r in ft], [r["K0_centroid_err_mm"] for r in ft], "-", color="k", lw=0.8, label="fine scan, chosen variant")
    axs[2].plot([r["flange_shift_mm"] for r in ft], [r["K0_dice"] for r in ft], "-", color="k", lw=0.8)
    b = v2["chosen"]["fine_best_flange_shift_mm"]
    for ax in axs:
        ax.axvline(b, color="k", ls=":", lw=0.8)
    axs[0].axhline(v2["v1_reference"]["corpus_K0"]["centroid_err_NONE_mm"], color="grey", ls="--", lw=0.8, label="NONE (unmoved)")
    axs[0].axhline(v2["v1_reference"]["corpus_K0"]["centroid_err_mm"], color="grey", ls="-.", lw=0.8, label="rule v1")
    axs[0].set_ylabel("K0 corpus centroid error vs BT (mm)"); axs[1].set_ylabel("flange offset vs BT (mm): along axis (o), lateral (s)")
    axs[2].set_ylabel("K0 corpus Dice vs BT uterus")
    for ax in axs:
        ax.set_xlabel("seating shift Delta (mm)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Rule v2 sweep of the seating shift Delta (in-sample best %.1f mm, dotted); %s" % (b, label), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fn = os.path.join(FIGS, "pose_rule_delta_sweep.png"); fig.savefig(fn, dpi=85); plt.close(fig)
    print("[fig] wrote", fn, flush=True)


def render3d():
    """py -3.11: pyvista render of the device meshes and channel axes -> figs/applicator_3d.png."""
    import pyvista as pv
    os.makedirs(FIGS, exist_ok=True)
    app = json.load(open(APP + "/applicator.json"))
    cols = dict(tube=(0.85, 0.85, 0.9), shaft=(0.6, 0.6, 0.7), ovoid_L=(0.9, 0.55, 0.2), ovoid_R=(0.95, 0.75, 0.3))
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 700), border=True)
    views = [("side (looking along -x: y right, z up)", "yz"), ("front (looking along y)", "xz"), ("top (looking down the tube)", "xy")]
    for k, (ttl, cam) in enumerate(views):
        pl.subplot(0, k); pl.set_background("white")
        for name, col in cols.items():
            m = pv.read(APP + "/%s.obj" % name)
            pl.add_mesh(m, color=col, smooth_shading=True, opacity=0.95 if k < 2 else 0.6, show_edges=(k == 1), edge_color="grey", line_width=0.3)
        for ch in app["needle_channels"]:
            p = np.asarray(ch["axis_point_mm"]); d = np.asarray(ch["dir"]); L = ch["length_mm"] + 6.0
            pl.add_mesh(pv.Line(p - 3.0 * d, p + L * d), color="blue" if ch["type"] == "parallel" else "green", line_width=3)
        pl.add_mesh(pv.Line((0, 0, -35), (0, 0, 70)), color="black", line_width=1)
        pl.add_text(ttl, font_size=10, color="black")
        pl.camera_position = cam
        pl.reset_camera()
    fn = os.path.join(FIGS, "applicator_3d.png"); pl.screenshot(fn); pl.close()
    print("[fig] wrote", fn, flush=True)


# ============================================================================ README
def write_readme(prm, mesh, fit, rule, valid, cmds, v2=None, score=None, label="rule v1"):
    fn = os.path.join(HERE, "README.md")
    v = valid["rule_v1"]; a0v = valid["alternatives"]["angle_0"]; a30 = valid["alternatives"]["angle_30"]; sv = valid["sensitivity_vaginal_fixed_point_pm5mm"]
    f = fit["fitted"]
    L = []
    L.append(README_MARK)
    L.append("")
    L.append("Module `applicator_venezia.py` (host, python 3.13; `py -3.11` only for the pyvista render). Implements CONTRACT sections 2 and 3: "
             "the parametric Venezia-type applicator (surfaces in the applicator frame, needle-channel axes), the fit of the two-cap ovoid "
             "model to the BT ovoid label, the kinematic pose rule v1 with its insertion path, and the rule's score against the real BT tandem.")
    L.append("")
    L.append("Frames/units: applicator frame = origin at the flange centre (tube exit from the caps), z = tube axis flange->tip, y = anterior "
             "in the sagittal plane, x = y x z (patient right); pose.json in preBT world RAS mm. All JSON state frame and units.")
    L.append("")
    L.append("### Commands")
    L.append("```")
    L.extend(cmds)
    L.append("```")
    L.append("")
    L.append("### Outputs")
    L.append("`MRI_GYN_sim/hybrid/applicator/{tube.obj, shaft.obj, ovoid_L.obj, ovoid_R.obj, applicator.json, pose.json}`, "
             "`MRI_GYN_sim/hybrid/figs/{ovoid_fit.png, pose_rule_BT.png, pose_rule_preBT.png, pose_rule_path.png, applicator_3d.png}`, "
             "`MRI_GYN_sim/hybrid/logs/applicator_venezia.json`. Surfaces: " +
             "; ".join("%s %d tris, closed=%s, vol %.0f mm3 (%+.1f %% vs reference)" % (k, s["tris"], s["closed_oriented"], s["signed_volume_mm3"], s["vol_err_pct"]) for k, s in mesh.items()) + ".")
    L.append("")
    L.append("### Parameters (applicator.json `params`)")
    L.append("")
    L.append("| parameter | value | unit | provenance |")
    L.append("|---|---|---|---|")
    for k, d in prm.items():
        vv = d["value"]
        vs = ("%.3g" % vv) if isinstance(vv, float) else str(vv)
        L.append("| `%s` | %s | %s | %s |" % (k, vs, d["unit"], d["source"]))
    L.append("")
    L.append("Ovoid shape (decision): each ovoid is a lunar cap = the half (patient left / right of the mid-sagittal slot `ovoid_gap_mm`) of a "
             "dome-on-cylinder body of diameter `ovoid_diam_mm` and height `ovoid_height_mm`, the dome being the upper half of an ellipsoid "
             "of height `ovoid_dome_mm` (dome = height recovers a pure half-ellipsoid cap). Its centre may be offset laterally "
             "(`ovoid_offset_x_mm`, 0 = one body split in two) and the assembly may be tilted about x (`ovoid_tilt_deg`). The BT ovoid label is a "
             "solid ~40 mm disc-like signal void with a domed top at the flange and a flat bottom ~20 mm below, plus a packing tail; this family "
             "matches it, the contract's 'two ovoids at x = +/-11 mm' corresponds to the centroids of the two halves (+/-%.1f mm here)." % (3.0 / 8.0 * f["ovoid_diam_mm"] / 2 + f["ovoid_offset_x_mm"]))
    L.append("")
    L.append("### Ovoid fit (rigid two-cap model vs the BT ovoid label, pose = measured BT tandem)")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append("| fitted diameter / lateral offset / dz / AP tilt / dome | %.1f mm / %.1f mm / %.1f mm / %.1f deg / %.1f mm |" % (
        f["ovoid_diam_mm"], f["ovoid_offset_x_mm"], f["ovoid_dz_mm"], f["ovoid_tilt_deg"], f["ovoid_dome_mm"]))
    L.append("| assembly extents LR x AP x z | %.1f x %.1f x %.1f mm (contract: 41.5 x 33.7 x 22.2) |" % (fit["assembly_extent_LR_mm"], fit["assembly_extent_AP_mm"], fit["assembly_extent_z_mm"]))
    L.append("| Dice model vs label (whole label / ovoid zone) | %.3f / %.3f |" % (fit["dice"], fit["dice_ovoid_zone"]))
    L.append("| volumes: label / model / overlap | %.2f / %.2f / %.2f cc |" % (fit["label_vol_cc"], fit["model_vol_cc"], fit["overlap_cc"]))
    L.append("| leftover label - model (= packing, not device) | %.2f cc (%.2f cc below the cap bottom; z_app range %s mm) |" % (
        fit["leftover_label_minus_model_cc"], fit["leftover_below_cap_bottom_cc"], fit["leftover_z_app_range_mm"]))
    L.append("| model outside the label | %.2f cc |" % fit["model_outside_label_cc"])
    L.append("")
    if v2 is not None:
        ch = v2["chosen"]; rb = v2["residual_at_best"]; ab = ch["at_best"]
        L.append("### Pose rule v2 (pose.json default): vaginal principal axis + seating shift Delta")
        L.append("")
        L.append("Shaft axis = principal axis of the preBT vagina body (oriented +S), line through **%s**; tube axis = shaft axis rotated %g deg "
                 "anteriorly about LR; flange = base + Delta along the **%s** axis. **Delta is the seating shift of the ovoids/packing: a SCENARIO "
                 "parameter, not a prediction** (nothing in the preBT scan gives it; the in-sample best on this case is %.1f mm, fine scan step 2.5 mm). "
                 "Corpus placement is unchanged (a0 -> tube axis, L_end %.1f mm above the flange). `pose.json` carries the corpus target for every "
                 "Delta of the sweep (`corpus.by_flange_shift_mm`) so the scene module sweeps Delta from its cfg; the device translates along the "
                 "%s axis (the tube enters the canal first)." % (ch["shaft_line_through"], val(prm, "angle_deg"), ch["shift_axis"], ch["fine_best_flange_shift_mm"],
                                                                 rule["d_F_mm"], rule["insertion_path"]["translation_axis"]))
        L.append("")
        L.append("| v2 variant (line through, Delta along) | axis angle to BT (deg): total (sag / cor) | best Delta of the coarse sweep (mm) | "
                 "flange along / lateral (mm) | tip offset (mm) | K0 corpus: Dice, centroid err (mm) |")
        L.append("|---|---|---|---|---|---|")
        for nm, vv in v2["variants"].items():
            b = vv["best"]; e = b["axis_error_components"]
            L.append("| %s%s | %.1f (%+.1f / %+.1f) | %.0f | %+.1f / %.1f | %.1f | %.3f, %.1f |" % (
                nm, " **(chosen)**" if nm == ch["variant"] else "", b["axis_angle_deg"], e["sagittal_deg"], e["coronal_deg"], b["flange_shift_mm"],
                b["flange_offset_mm"]["along_BT_axis"], b["flange_offset_mm"]["lateral"], b["tip_offset_mm"]["total"],
                b["corpus_K0"]["uterus_dice_vs_BT"], b["corpus_K0"]["centroid_err_mm"]))
        L.append("")
        L.append("Fine Delta scan of the chosen variant (K0 corpus vs the BT uterus; NONE = unmoved preBT corpus, centroid error %.1f mm; rule v1 %.1f mm):"
                 % (v2["v1_reference"]["corpus_K0"]["centroid_err_NONE_mm"], v2["v1_reference"]["corpus_K0"]["centroid_err_mm"]))
        L.append("")
        L.append("| Delta (mm) | K0 Dice | K0 centroid err (mm) | flange along / lateral (mm) | tip offset (mm) |")
        L.append("|---|---|---|---|---|")
        for r in ch["fine_table"]:
            L.append("| %.1f%s | %.3f | %.1f | %+.1f / %.1f | %.1f |" % (r["flange_shift_mm"], " **best**" if r["flange_shift_mm"] == ch["fine_best_flange_shift_mm"] else "",
                                                                  r["K0_dice"], r["K0_centroid_err_mm"], r["flange_along_mm"], r["flange_lateral_mm"], r["tip_offset_mm"]))
        L.append("")
        L.append("Residual at the best Delta: axis angle %.1f deg (sagittal %+.1f, coronal %+.1f), flange lateral offset %.1f mm (along %+.1f), tip offset "
                 "%.1f mm; K0 corpus Dice %.3f, centroid error %.1f mm. What remains is the axis error and the perpendicular offset of the shaft line; "
                 "the along-axis error is absorbed by Delta by construction (in-sample). Selection rule: %s." % (
                     rb["axis_angle_deg"], rb["axis_error_components"]["sagittal_deg"], rb["axis_error_components"]["coronal_deg"], rb["flange_lateral_mm"],
                     rb["flange_along_mm"], rb["tip_offset_mm"]["total"], ab["corpus_K0"]["uterus_dice_vs_BT"], ab["corpus_K0"]["centroid_err_mm"], ch["selection"]))
        L.append("")
    L.append("### Pose rule v1 (REJECTED on this case; kept as the reference in `validation.rule_v1`)")
    L.append("")
    L.append("Shaft axis = chord from the vaginal fixed point (centroid of the lowest 15 mm of the vagina body along its principal axis, "
             "computed from the preBT labels exactly as the meshing module defines `vagina.fixed_inferior`) to the external os `O_pre`; "
             "flange at `O_pre`; tube axis = shaft axis rotated %g deg anteriorly about the patient LR axis (the in-plane variant differs by "
             "%.2f deg); corpus: a0 -> tube axis by the minimal rotation with L_end %.1f mm above the flange (d_F = L_iu + 0.5 - %.1f; "
             "bdev.json d_F_tip = %.1f); insertion path: translation along the %s axis over %.1f mm (tip 4 mm below the os at u = 0), corpus "
             "screw motion (smoothstep) from u = %.3f (tip at the internal-os level) to u = 1, then settle." % (
                 val(prm, "angle_deg"), rule["tilt_variants_diff_deg"], rule["d_F_mm"], rule["canal_above_L_end_mm"],
                 float(json.load(open(P["out"] + "/final/bdev.json"))["picks"]["d_F_tip"]), rule["insertion_path"]["translation_axis"],
                 rule["insertion_path"]["travel_mm"], rule["insertion_path"]["u_tip_at_internal_os"]))
    L.append("")
    al = valid["alternatives"]
    L.append("| variant | axis angle to the BT tandem (deg): total (sagittal / coronal) | flange offset total / along BT axis / lateral (mm) | tip offset (mm) | K0 corpus: Dice vs BT uterus, centroid err (mm) |")
    L.append("|---|---|---|---|---|")
    rows = [("**rule v1**, angle %g (BONE frame)" % val(prm, "angle_deg"), v), ("contract alternative: angle 0", a0v), ("contract alternative: angle 30", a30),
            ("diagnostic: oracle angle %g (in-sample)" % al["angle_oracle"]["angle_deg"], al["angle_oracle"]),
            ("diagnostic: shaft axis = vagina principal axis (preBT-only)", al["shaft_axis_vagina_pca"]),
            ("diagnostic: tube axis = canal axis a0 (preBT-only)", al["tube_axis_a0"])]
    for nm, r in rows:
        e = r["axis_error_components"]
        L.append("| %s | %.1f (%+.1f / %+.1f) | %.1f / %+.1f / %.1f | %.1f | %.3f, %.1f |" % (
            nm, r["axis_angle_deg"], e["sagittal_deg"], e["coronal_deg"], r["flange_offset_mm"]["total"], r["flange_offset_mm"]["along_BT_axis"],
            r["flange_offset_mm"]["lateral"], r["tip_offset_mm"]["total"], r["corpus_K0"]["uterus_dice_vs_BT"], r["corpus_K0"]["centroid_err_mm"]))
    for k, r in valid["frame_sensitivity"].items():
        e = r["axis_error_components"]
        L.append("| rule v1 in the %s frame | %.1f (%+.1f / %+.1f) | %.1f / %+.1f / %.1f | %.1f | - |" % (
            k, r["axis_angle_deg"], e["sagittal_deg"], e["coronal_deg"], r["flange_offset_mm"]["total"], r["flange_offset_mm"]["along_BT_axis"],
            r["flange_offset_mm"]["lateral"], r["tip_offset_mm"]["total"]))
    L.append("")
    L.append("NONE = preBT anatomy unmoved (centroid error %.1f mm in the BONE frame). Sign convention: along-axis offset < 0 = predicted flange inferior "
             "to the real one along the BT tandem; sagittal component > 0 = predicted axis more anterior, coronal > 0 = more to the patient's right."
             % v["corpus_K0"]["centroid_err_NONE_mm"])
    L.append("")
    ctx = valid["between_session_context"]; pca = al["shaft_axis_vagina_pca"]
    L.append("Why rule v1 fails on this case (numbers in `pose.json[validation]`): (i) the real flange sits %.1f mm above the preBT external os along a0 "
             "(the cervix was pushed cranially by the device and packing; uterus centroid shift %.1f mm in the BONE frame), so a flange pinned at O_pre "
             "is ~%.0f mm too inferior whatever the axis; (ii) the vaginal chord from the fixed point to O_pre tilts %.1f deg away from the vagina body's "
             "own principal axis because the external os lies %.1f mm lateral of that axis line - the chord inherits a %.0f deg leftward tilt that the real "
             "(centred) tandem does not have; (iii) the preBT vagina is ~%.0f deg more posterior than the inserted shaft. Replacing the chord by the vaginal "
             "principal axis brings the axis error from %.1f to %.1f deg and the K0 corpus centroid error from %.1f to %.1f mm without using any BT information; "
             "the canal-axis variant gives %.1f deg / %.1f mm." % (
                 ctx["real_flange_above_O_pre_along_a0_mm"], ctx["uterus_centroid_shift_preBT_to_BT_mm"], abs(v["flange_offset_mm"]["along_BT_axis"]),
                 pca["chord_vs_pca_axis_deg"], pca["O_pre_lateral_from_vagina_axis_line_mm"], abs(v["axis_error_components"]["coronal_deg"]),
                 abs(al["angle_oracle"]["angle_deg"] - val(prm, "angle_deg")), v["axis_angle_deg"], pca["axis_angle_deg"], v["corpus_K0"]["centroid_err_mm"],
                 pca["corpus_K0"]["centroid_err_mm"], al["tube_axis_a0"]["axis_angle_deg"], al["tube_axis_a0"]["corpus_K0"]["centroid_err_mm"]))
    L.append("")
    L.append("Sensitivity to the vaginal fixed point (+/-5 mm along x, y, z; chord length %.1f mm): tube axis changes by <= %.2f deg, the corpus "
             "centroid moves by <= %.2f mm, the tip by <= %.2f mm; the BT-frame axis error ranges %.1f-%.1f deg. The flange is pinned to O_pre, "
             "so the fixed point only steers the axes." % (valid["vaginal_chord_length_mm"], sv["max_tube_axis_change_deg"], sv["max_corpus_centroid_shift_mm"],
                                                          sv["max_tip_shift_mm"], sv["BT_axis_angle_range_deg"][0], sv["BT_axis_angle_range_deg"][1]))
    L.append("")
    L.append("Reading: the along-axis flange offset is the between-session cranial shift of the cervix by the applicator + packing that a rule with "
             "the flange at the preBT external os cannot produce (the real flange sits %.1f mm above L_end along a0 in the BONE frame, poses_align.json); "
             "the lateral offset and the axis angle are what the vaginal-chord construction gets wrong on its own. NONE -> K0 corpus centroid error: "
             "%.1f -> %.1f mm." % (ctx["real_flange_above_L_end_along_a0_mm"], v["corpus_K0"]["centroid_err_NONE_mm"], v["corpus_K0"]["centroid_err_mm"]))
    L.append("")
    L.append("### Caveats")
    L.append("- Device identity is unconfirmed until the RTPLAN / vendor geometry is available: tube radius, L_iu, junction angle and the cap shape "
             "are measured from the BT labels (MRI signal voids include partial volume and susceptibility, the ovoid label includes packing); "
             "the 30 mm curved shaft, the channel layout (azimuths, oblique entry radius) and the 0.5 mm slot are assumed.")
    L.append("- The BT ovoid label is not the device: %.1f cc of it is packing; the fit optimises Dice against the whole label, bounded so the caps stay "
             "at the flange (dz in [-5, 3] mm) - the reported Dice is a lower bound on the device fit." % fit["leftover_label_minus_model_cc"])
    L.append("- Insertion path: translation along the %s axis (tip %.1f mm off the os line at u = 0; the other axis is written as `insertion_path_alt`). "
             "The cranial cervix shift that rule v1 could not produce is, in v2, the seating shift Delta: an explicit scenario parameter to sweep, "
             "not a prediction; a cohort or an intra-procedural measurement is what would give it a prior."
             % (rule["insertion_path"]["translation_axis"], rule["insertion_path"]["tip_offset_from_os_line_at_u0_mm"]))
    L.append("- Roll about the tube axis is unobservable from the labels (the caps are near-axisymmetric): the applicator x axis is defined by the "
             "patient LR direction in both the rule and the BT frame.")
    L.append("")
    txt = "\n".join(L)
    if os.path.exists(fn):
        old = open(fn, encoding="utf-8").read()
        if README_MARK in old:
            i = old.index(README_MARK); j = old.find("\n## ", i + len(README_MARK))
            new = old[:i] + txt + ("\n" if j < 0 else old[j:])
        else:
            new = old.rstrip("\n") + "\n\n" + txt + "\n"
    else:
        new = "# Hybrid pelvis simulator - Stage 1 modules\n\nSee CONTRACT.md. Each module appends its own section.\n\n" + txt + "\n"
    open(fn, "w", encoding="utf-8").write(new)
    print("[readme] wrote section '%s' into %s" % (README_MARK, fn), flush=True)


# ============================================================================ realistic cap size
def shaft_tilt_deg():
    """Tilt about x (deg) that makes the cap assembly's axis parallel to the STRAIGHT vaginal rod.

    The caps are built about the applicator z axis = the intrauterine tube, i.e. the bent UPPER rod.  Clinically the
    ring sits on the vaginal (lower) rod, which leaves the flange at the junction angle.  Measured from
    shaft_straight.obj rather than assumed: its principal axis is 24.00 deg from +z, leaning posteriorly going up,
    so the tilt is negative in the `ovoid_tilt_deg` convention (positive = apex anterior)."""
    V, _ = geom.read_obj(APP + "/shaft_straight.obj")
    V = np.asarray(V, float)
    c = V.mean(0)
    _, _, vt = np.linalg.svd(V - c, full_matrices=False)
    d = geom.unit(vt[0])
    if d[2] < 0:
        d = -d                                        # up-pointing, toward the flange
    return float(np.degrees(np.arctan2(d[1], d[2]))), d


def scaled_ovoids(diams, align_shaft=False):
    """Write the lunar caps at a REALISTIC assembly diameter, as extra files next to the fitted ones.

    align_shaft=True additionally tilts the assembly so its axis is parallel to the straight vaginal rod (tag suffix
    "s"), which is where the ring sits clinically; the fitted -4 deg tilt was to the TUBE axis.

    The fitted `ovoid_diam_mm` (39.4 mm) is not a device size.  It was fitted to the BT ovoid label, which is the MRI
    dark region = device + vaginal packing + susceptibility bloom, and the fit was BOUNDED BELOW AT 30 mm
    (fit_ovoids bounds), so it could never return a clinical size; only 4.1 of the 23.4 cc label was attributed to
    packing.  A 39 mm body cannot travel up a vagina whose lumen is ~12-14 mm across, and the resulting simulations
    had the caps closing around the OUTSIDE of the vaginal tube instead of entering it.

    Each variant multiplies diameter, height, dome height and lateral offset by k = D / D_fit about the cap apex
    (`ovoid_dz_mm` is kept, so the tube still exits the caps at the flange), and keeps the tilt and the 0.5 mm
    mid-sagittal slot: the fitted SHAPE is preserved and only its size changes.  Output, in <out>/hybrid/applicator/:
        ovoid_L_d<D>.obj, ovoid_R_d<D>.obj, ovoids_d<D>.json
    Select them in a run with cfg device_part_files {"ovoid_L": "ovoid_L_d26", "ovoid_R": "ovoid_R_d26"}.
    The fitted ovoid_L.obj / ovoid_R.obj are never overwritten, so every earlier run stays reproducible."""
    stored = json.load(open(APP + "/applicator.json"))
    tilt, d_shaft = (shaft_tilt_deg() if align_shaft else (None, None))
    rows = []
    for D in diams:
        prm = json.loads(json.dumps(stored["params"]))
        D0 = float(val(prm, "ovoid_diam_mm"))
        k = float(D) / D0
        prm["ovoid_diam_mm"]["value"] = float(D)
        for key in ("ovoid_height_mm", "ovoid_dome_mm", "ovoid_offset_x_mm"):
            prm[key]["value"] = float(val(prm, key)) * k
        if align_shaft:
            prm["ovoid_tilt_deg"]["value"] = tilt
        tag = ("d%g" % D).replace(".", "p") + ("s" if align_shaft else "")
        rec = dict(tag=tag, units="mm; " + FRAME_APP, assembly_diam_mm=float(D), fitted_diam_mm=D0, scale=round(k, 5),
                   params={key: round(float(val(prm, key)), 4) for key in
                           ("ovoid_diam_mm", "ovoid_height_mm", "ovoid_dome_mm", "ovoid_offset_x_mm", "ovoid_dz_mm",
                            "ovoid_tilt_deg", "ovoid_gap_mm")},
                   source="ASSUMED clinical assembly diameter (not measured on this patient: device identity and size "
                          "are unconfirmed until the RTPLAN / vendor geometry is available); shape = the BT fit, "
                          "scaled by k = D / D_fit about the cap apex",
                   select_with='device_part_files {"ovoid_L": "ovoid_L_%s", "ovoid_R": "ovoid_R_%s"}' % (tag, tag))
        allV = []
        for side, name in ((-1, "ovoid_L"), (+1, "ovoid_R")):
            V, F, st = ovoid_mesh(prm, side)
            fn = "%s/%s_%s.obj" % (APP, name, tag)
            geom.write_obj(fn, V, F, header="%s of the Venezia-type applicator model, SCALED to a %g mm assembly "
                                            "(fitted %.2f mm); %s" % (name, D, D0, FRAME_APP))
            st["file"] = os.path.basename(fn)
            st["vertex_mean_mm"] = np.round(V.mean(0), 3).tolist()
            rec[name] = st
            allV.append(V)
        A = np.vstack(allV)
        rec["assembly_extent_mm"] = dict(LR=round(float(np.ptp(A[:, 0])), 2), AP=round(float(np.ptp(A[:, 1])), 2),
                                         z=round(float(np.ptp(A[:, 2])), 2),
                                         z_range=[round(float(A[:, 2].min()), 2), round(float(A[:, 2].max()), 2)])
        if align_shaft:
            ax = (A @ d_shaft)
            rad = np.linalg.norm(A - np.outer(ax, d_shaft), axis=1)
            rec["aligned_to"] = dict(axis="straight vaginal rod (shaft_straight.obj principal axis)",
                                     tilt_deg=round(tilt, 3), axis_app=np.round(d_shaft, 5).tolist(),
                                     radius_about_shaft_axis_mm=round(float(rad.max()), 3),
                                     extent_along_shaft_mm=[round(float(ax.min()), 2), round(float(ax.max()), 2)])
        rec["volume_cc"] = round(sum(rec[n]["signed_volume_mm3"] for n in ("ovoid_L", "ovoid_R")) / 1000.0, 3)
        json.dump(rec, open("%s/ovoids_%s.json" % (APP, tag), "w"), indent=1, default=float)
        rows.append(rec)
        e = rec["assembly_extent_mm"]
        print("[ovoids] %-5s  assembly LR %5.1f x AP %5.1f x z %5.1f mm  volume %6.2f cc  (fitted %.1f mm, k %.3f)  "
              "closed %s/%s" % (tag, e["LR"], e["AP"], e["z"], rec["volume_cc"], D0, k,
                                rec["ovoid_L"]["closed_oriented"], rec["ovoid_R"]["closed_oriented"]), flush=True)
    return rows


# ============================================================================ v4: the tandem body from the updated BT label
# Fix plan of the G32 audit (hybrid/logs/audit_G32/fix_plan_G32_audit.md), S4a (tandem body) and S4c (label
# measurement and validation).  S4b (the ring halves with their rods) waits on the ring-size decision U1, so nothing
# below builds an ovoid body: a v4 directory holds the tandem alone (tube.obj + shaft.obj = ONE rigid body in the scene).
# What the updated label shows and v3 got wrong (audit R1): the real tube runs STRAIGHT through the ring to z_app
# -22..-24; below that the vaginal tandem curves forward (y_app ~0 -> ~21 mm over z_app -24..-50) into a three-rod
# bundle that leaves the body ~25-28 deg from the tube.  v3 bent 28.9 deg AT the flange and hung its cap rods 16 mm in
# front of the tandem rod (8.8-17 mm anterior of the real rods in the pelvis frame).
FRAME_APP_V4 = ("applicator frame v4 = the BT LABEL frame: origin = flange (the inputs/applicator.json BT flange projected "
                "onto the label tube axis), z = label tube axis flange->tip (slab-centroid line of the updated applicator "
                "label minus the ovoid label, 15-55 mm above the flange), x = BT world x orthogonalised against z, "
                "y = z cross x (anterior); mm")
TANDEM_V4_PARTS = ("tube", "shaft")
POSE_RULE_KEYS = ("L_iu_mm", "angle_deg", "tip_below_os_mm", "n_steps", "path_axis", "corpus_ease", "tilt_axis",
                  "d_F_margin_mm", "vagina_fixed_inferior_mm", "shaft_axis_mode", "shaft_line_through", "flange_shift_mm",
                  "shift_axis", "flange_shift_sweep_mm")
V4_DROPPED_PREFIXES = ("ovoid_", "pack_", "rod_")
V4_DROPPED_KEYS = ("shaft_arc_len_mm", "channel_radius_mm", "n_parallel_per_ovoid", "n_oblique_per_ovoid", "oblique_deg",
                   "oblique_radius_mm", "needle_r_mm", "parallel_azimuth_deg", "oblique_azimuth_deg", "mesh_voxel_mm")


def v4_params(prm):
    """The tandem-body (v4) parameters, added to / overriding default_params; the ovoid-body, packing, rod and needle
    parameters are dropped (no ovoid body exists in a tandem-only variant).  Returns the dropped keys."""
    dropped = [k for k in list(prm) if k.startswith(V4_DROPPED_PREFIXES) or k in V4_DROPPED_KEYS]
    for k in dropped:
        del prm[k]

    def add(k, v, unit, source, note=""):
        prm[k] = dict(value=v, unit=unit, source=source, **({"note": note} if note else {}))

    add("tandem_only", True, "-", "fix plan S4a/S6: v4 = the tandem body alone (tube + vaginal tandem, one rigid body); the "
        "ring halves (S4b) wait on the ring-size decision U1")
    add("r_tandem_mm", 2.35, "mm", "MEASURED (fix plan S4a): MRI FWHM 4.5-5.2 mm of the tube on the updated BT applicator "
        "label; the scene reads this key as the TUBE radius.  label_geometry.tube.r_eq_mm = this build's voxel-count radius")
    add("r_shaft_mm", 3.15, "mm", "MEASURED (fix plan S4a): FWHM 6.30 mm of the vaginal tandem rod on the updated label.  "
        "label_geometry.tandem.r_label_mm = this build's tilt-corrected section radius (partial volume adds ~0.2 mm)")
    add("shaft_style", "label_sweep", "-", "v4: the vaginal tandem = a tube of r_shaft_mm swept (sweep_polyline, parallel-"
        "transport frames) along the tandem centreline measured on the updated label, then straight along the measured "
        "bundle axis past the introitus.  One rigid body with the tube: there is no bend AT the flange any more")
    add("rods", False, "-", "v4 tandem-only: no ovoid rods (S4b)")
    add("label_axis_s_mm", [15.0, 55.0], "mm", "DEFINITION (fix plan 'Frames'): the label tube axis is the slab-centroid line "
        "of (updated applicator minus ovoid) over this height range above the json flange, along the json axis")
    add("label_axis_slab_mm", 2.0, "mm", "NUMERICAL: slab height of that line fit")
    add("label_axis_lat_mm", 6.0, "mm", "NUMERICAL: voxels within this distance of the json axis enter the line fit (8 mm "
        "gives the same line to 0.001 deg); also the radius that defines the ring's on-axis extent")
    add("label_station_mm", 2.0, "mm", "NUMERICAL: spacing of the label-frame z planes the rods are traced on (voxels are "
        "1.125 x 1.125 x 1.6 mm)")
    add("label_plane_px_mm", 0.5, "mm", "NUMERICAL: in-plane resampling of those planes (trilinear, >= 0.5 = inside)")
    add("rod_min_px", 6, "-", "NUMERICAL: plane components under 6 px (1.5 mm2) are ignored (ring-edge specks)")
    add("tandem_start_max_off_mm", 2.5, "mm", "NUMERICAL: the first tandem section below the ring must lie this close to "
        "the tube axis (the junk just under the ring sits >= 4 mm off it)")
    add("tandem_min_req_mm", 2.0, "mm", "NUMERICAL: minimum equivalent radius of the first tandem section")
    add("track_min_req_mm", 1.5, "mm", "NUMERICAL: minimum equivalent radius while tracking the tandem downward")
    add("track_max_jump_mm", 3.5, "mm", "NUMERICAL: largest in-plane step between consecutive tandem sections (after linear "
        "extrapolation of the last two)")
    add("rod_merge_req_mm", 5.0, "mm", "NUMERICAL: a section above this equivalent radius holds more than one rod (one rod "
        "reads 2.9-4.3 mm at tilts 0-50 deg, the merged bundle 6.3-7.8 mm)")
    add("bundle_gap_max", 2, "-", "NUMERICAL: planes allowed between the last separable tandem section and the bundle")
    add("bundle_full_frac", 0.9, "-", "NUMERICAL: bundle sections with r_eq >= this x the median (the plateau) define the "
        "bundle axis.  MEASURED: r_eq 6.3, 6.6 at the merge, 7.0-7.8 on the plateau, 6.5 where the label ends; the tapering "
        "end alone turns a line fit by ~2 deg")
    add("tandem_select_margin_mm", 0.75, "mm", "NUMERICAL: label voxels within (section semi-axis + this) of the traced "
        "tandem centre, and nearer to it than to any other rod, form the label tandem rod scored by S4c")
    add("shaft_overlap_mm", 1.0, "mm", "NUMERICAL: the shaft's top ring sits this far inside the tube, so the one rigid body "
        "has no gap at the junction")
    add("shaft_ring_step_mm", 2.5, "mm", "NUMERICAL: ring spacing on the straight bundle-axis extension (the traced part keeps "
        "its 2 mm stations)")
    add("shaft_past_introitus_mm", 20.0, "mm", "ASSUMED: the straight extension ends this far beyond the preBT introitus "
        "plane (vaginal principal axis) with the device at device_final; at the unlifted end of C it is ~22 mm further out")
    return dropped


def label_applicator_file(arg=None):
    """Path of the UPDATED BT applicator label (tube, ring, tandem rod, the two ovoid rods and their bundle).  It was drawn
    after the original label set and sits beside the read-only data folder, not in it; --app-label or APPSIM_APP_LABEL
    override.  Local patient data: the path is recorded, the file is never copied."""
    cands = [arg, os.environ.get("APPSIM_APP_LABEL"),
             os.path.dirname(P["data"].rstrip("/")) + "/BT_MRI_label_applicator.nii"]
    for c in cands:
        if c and os.path.exists(c):
            return c.replace("\\", "/")
    raise FileNotFoundError("updated BT applicator label not found (tried %s); pass --app-label" % [c for c in cands if c])


def load_label_new(fn):
    """The updated applicator label and the BT ovoid label (same grid, checked)."""
    import nibabel as nib
    im = nib.load(fn)
    ov = nib.load(P["data"] + "/BT_MRI_label_ovoid.nii")
    if im.shape != ov.shape or not np.allclose(im.affine, ov.affine, atol=1e-4):
        raise ValueError("%s is not on the BT label grid" % fn)
    aff = im.affine.copy()
    return dict(app=np.asarray(im.dataobj) > 0, ovoid=np.asarray(ov.dataobj) > 0, aff=aff, shape=tuple(im.shape),
                vv=float(abs(np.linalg.det(aff[:3, :3]))), file=fn)


def _vox_world(mask, aff):
    return np.argwhere(mask).astype(float) @ aff[:3, :3].T + aff[:3, 3]


def label_frame(lab, app_in, prm):
    """The BT label frame (fix plan, 'Frames').  Axis = the slab-centroid line of (updated applicator minus ovoid) over
    label_axis_s_mm above the inputs/applicator.json flange (heights along the json axis); origin = the json flange
    projected onto that line; roll = the json convention (x = BT world x orthogonalised against z, y = z cross x).  That
    roll is the model's own: device_final's x_app carried into BT by the BONE frame lies on it (validate_device_vs_label
    reports the angle)."""
    Fj = np.asarray(app_in["origin_BT_world"], float)
    zj = geom.unit(np.asarray(app_in["R_rows_BT_world"], float)[2])
    W = _vox_world(lab["app"] & ~lab["ovoid"], lab["aff"])
    q = W - Fj
    s = q @ zj
    lat = np.linalg.norm(q - np.outer(s, zj), axis=1)
    s_lo, s_hi = [float(v) for v in val(prm, "label_axis_s_mm")]
    h, lat_max = float(val(prm, "label_axis_slab_mm")), float(val(prm, "label_axis_lat_mm"))
    cen = []
    for s0 in np.arange(s_lo, s_hi - 1e-9, h):
        m = (s >= s0) & (s < s0 + h) & (lat < lat_max)
        if m.sum() >= 4:
            cen.append(W[m].mean(0))
    cen = np.array(cen)
    if len(cen) < 5:
        raise ValueError("label tube: only %d slabs with voxels in s %g..%g" % (len(cen), s_lo, s_hi))
    c = cen.mean(0)
    _, _, vt = np.linalg.svd(cen - c, full_matrices=False)
    z = geom.unit(vt[0] if vt[0] @ zj >= 0 else -vt[0])
    O = c + float((Fj - c) @ z) * z
    r = cen - c
    res = np.linalg.norm(r - np.outer(r @ z, z), axis=1)
    return dict(origin=O, R=geom.frame_from(z), axis=z, n_slabs=int(len(cen)), slab_line_rms_mm=float(np.sqrt((res ** 2).mean())),
                angle_to_json_axis_deg=geom.angle_deg(z, zj), origin_from_json_flange_mm=float(np.linalg.norm(O - Fj)),
                json_flange=Fj, json_axis=zj)


class LabelPlanes:
    """z_app = const planes of the label frame, resampled at px mm (trilinear on the 0/1 masks, >= 0.5 = inside).
    components(z) = the connected pieces of (applicator minus ovoid) in that plane, with centroid and equivalent radius."""

    def __init__(self, lab, O, R, px, xr=(-40.0, 40.0), yr=(-40.0, 75.0)):
        self.O, self.R = np.asarray(O, float), np.asarray(R, float)
        self.inv = np.linalg.inv(lab["aff"])
        self.app, self.ov = lab["app"].astype(np.float32), lab["ovoid"].astype(np.float32)
        self.px = float(px)
        self.xs = np.arange(xr[0], xr[1] + 1e-9, px)
        self.ys = np.arange(yr[0], yr[1] + 1e-9, px)
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")

    def _sample(self, vol, z):
        from scipy import ndimage as ndi
        Pp = np.stack([self.X, self.Y, np.full_like(self.X, z)], -1).reshape(-1, 3)
        ijk = (self.O + Pp @ self.R) @ self.inv[:3, :3].T + self.inv[:3, 3]
        return (ndi.map_coordinates(vol, ijk.T, order=1, mode="constant", cval=0.0) >= 0.5).reshape(self.X.shape)

    def components(self, z, min_px):
        from scipy import ndimage as ndi
        lab_, n = ndi.label(self._sample(self.app, z) & ~self._sample(self.ov, z))
        out = []
        for k in range(1, n + 1):
            pix = np.argwhere(lab_ == k)
            if len(pix) < min_px:
                continue
            x, y = self.xs[pix[:, 0]], self.ys[pix[:, 1]]
            out.append(dict(x=float(x.mean()), y=float(y.mean()), r_eq=float(np.sqrt(len(pix) * self.px ** 2 / np.pi)),
                            n_px=int(len(pix)), px_x=x, px_y=y))
        return out


def _station_tangents(P):
    """Unit tangents of a station polyline (central differences, one-sided at the ends)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.tile([0.0, 0.0, -1.0], (len(P), 1))
    T = np.empty_like(P)
    T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    T[1:-1] = P[2:] - P[:-2]
    return T / np.linalg.norm(T, axis=1, keepdims=True)


def measure_label_geometry(lab, app_in, prm):
    """S4c: every tandem-body dimension measured on the updated BT applicator label, in the BT label frame (label_frame).
    Nothing is taken from an analysis script: the rods are traced plane by plane here.
      tube      applicator-minus-ovoid voxels above the ovoid label (with the 0-12 mm 'collar' of label around the tube)
      tandem    the rod that leaves the ring bore on the axis, traced downward (nearest section to the extrapolated
                position) until its section merges with the ovoid rods (r_eq > rod_merge_req_mm)
      bundle    the merged three-rod section below; its axis = the line through the full sections' centroids
      tandem in the bundle: the posterior lobe of each bundle section (the tandem is the posterior rod: its centre is one
                label-rod semi-axis in front of the section's posterior edge)
    Returns (geo, vox): geo is JSON-able; vox holds label-frame voxel centres of the label parts that
    validate_device_vs_label scores against (all, ovoid, tube, tandem, tandem_in_bundle)."""
    fr = label_frame(lab, app_in, prm)
    O, R = fr["origin"], fr["R"]

    def to_lf(W):
        return (np.asarray(W, float) - O) @ R.T

    idx_a = np.argwhere(lab["app"] & ~lab["ovoid"])                 # grid indices, same order as Qa
    Qa = to_lf(idx_a.astype(float) @ lab["aff"][:3, :3].T + lab["aff"][:3, 3])
    Qo = to_lf(_vox_world(lab["ovoid"], lab["aff"]))
    lat_max = float(val(prm, "label_axis_lat_mm"))
    ro = np.hypot(Qo[:, 0], Qo[:, 1])
    ring = dict(z_top_mm=float(Qo[:, 2].max()), z_bottom_mm=float(Qo[:, 2].min()),
                z_top_on_axis_mm=float(Qo[ro < lat_max, 2].max()), z_bottom_on_axis_mm=float(Qo[ro < lat_max, 2].min()),
                on_axis_radius_mm=lat_max, vol_cc=float(len(Qo) * lab["vv"] / 1000.0),
                note="ovoid label (the ring; its rod sockets hang below it off the axis).  Measured only: the ring is S4b")
    # ---- tube
    Qt = Qa[Qa[:, 2] > ring["z_top_mm"]]
    rt = np.hypot(Qt[:, 0], Qt[:, 1])
    core = rt < 4.0
    z_tip = float(Qt[core, 2].max())
    z_lo_c, z_hi_c = ring["z_top_mm"] + 14.0, z_tip - 3.0            # above the collar, below the rounded tip
    cz = core & (Qt[:, 2] > z_lo_c) & (Qt[:, 2] < z_hi_c)
    tube = dict(z_tip_label_mm=z_tip, n_vox=int(len(Qt)), n_vox_core=int(core.sum()),
                r_eq_mm=float(np.sqrt(cz.sum() * lab["vv"] / (np.pi * max(1e-6, z_hi_c - z_lo_c)))),
                r_eq_zone_mm=[z_lo_c, z_hi_c], lateral_rms_core_mm=float(np.sqrt((rt[core] ** 2).mean())),
                z_range_mm=[float(Qt[:, 2].min()), float(Qt[:, 2].max())],
                definition="applicator-minus-ovoid voxels above the ovoid label top; core = within 4 mm of the label axis; "
                           "r_eq = voxel volume of the core between ring top + 14 mm (above the collar) and tip - 3 mm over "
                           "that length")
    # ---- trace the rods on z planes below the ring's on-axis bottom
    st = float(val(prm, "label_station_mm"))
    pl = LabelPlanes(lab, O, R, float(val(prm, "label_plane_px_mm")))
    min_px, r_merge = int(val(prm, "rod_min_px")), float(val(prm, "rod_merge_req_mm"))
    z0 = -st * np.ceil(-ring["z_bottom_on_axis_mm"] / st - 1e-9)
    zs = np.arange(z0, float(Qa[:, 2].min()) - st, -st)
    comps = [pl.components(float(z), min_px) for z in zs]
    start_off, jump = float(val(prm, "tandem_start_max_off_mm")), float(val(prm, "track_max_jump_mm"))
    T = []
    for i in range(min(6, len(zs))):
        cand = [c for c in comps[i] if float(val(prm, "tandem_min_req_mm")) <= c["r_eq"] <= r_merge
                and np.hypot(c["x"], c["y"]) <= start_off]
        if cand:
            T.append((i, min(cand, key=lambda c: np.hypot(c["x"], c["y"]))))
            break
    if not T:
        raise ValueError("no tandem section within %.1f mm of the tube axis in the 6 planes below the ring" % start_off)
    for i in range(T[0][0] + 1, len(zs)):
        p = np.array([T[-1][1]["x"], T[-1][1]["y"]])
        if len(T) >= 2:
            p = 2.0 * p - np.array([T[-2][1]["x"], T[-2][1]["y"]])
        cand = [c for c in comps[i] if c["r_eq"] >= float(val(prm, "track_min_req_mm"))
                and np.hypot(c["x"] - p[0], c["y"] - p[1]) <= jump]
        if not cand:
            break
        c = min(cand, key=lambda c: np.hypot(c["x"] - p[0], c["y"] - p[1]))
        if c["r_eq"] > r_merge:
            break
        T.append((i, c))
    Tp = np.array([[c["x"], c["y"], zs[i]] for i, c in T])
    tT = _station_tangents(Tp)
    cosT = np.abs(tT[:, 2])
    r_eqT = np.array([c["r_eq"] for _, c in T])
    r_lab = float(np.median(r_eqT * np.sqrt(cosT)))       # a z-plane cuts a rod tilted by t in an ellipse of area pi r^2 / cos t
    # ---- the bundle below the last separable tandem section
    B, gap = [], 0
    for i in range(T[-1][0] + 1, len(zs)):
        c = max(comps[i], key=lambda c: c["n_px"]) if comps[i] else None
        if c is None or c["r_eq"] < r_merge:
            if B:
                break
            gap += 1
            if gap > int(val(prm, "bundle_gap_max")):
                break
            continue
        B.append((i, c))
    if len(B) < 3:
        raise ValueError("bundle: only %d merged sections below the tandem" % len(B))
    rB = np.array([c["r_eq"] for _, c in B])
    full = rB >= float(val(prm, "bundle_full_frac")) * float(np.median(rB))
    Bc = np.array([[c["x"], c["y"], zs[i]] for i, c in B])
    cb = Bc[full].mean(0)
    _, _, vt = np.linalg.svd(Bc[full] - cb, full_matrices=False)
    d_b = geom.unit(vt[0] if vt[0][2] < 0 else -vt[0])              # bundle axis, pointing DOWN (out of the body)
    cos_b = abs(float(d_b[2]))
    lobe = []
    for i, c in B:                                                  # the tandem = the bundle's posterior rod
        y_min = float(c["px_y"].min())
        band = c["px_y"] <= y_min + r_lab / cos_b
        lobe.append([float(c["px_x"][band].mean()), y_min + r_lab / cos_b, float(zs[i])])
    lobe = np.array(lobe)
    # ---- label voxel sets scored in S4c
    margin = float(val(prm, "tandem_select_margin_mm"))
    by_i = {i: c for i, c in T}
    sel_T = np.zeros(len(Qa), bool)
    zT_inc = Tp[::-1, 2]
    for i, _ in T:
        m = np.abs(Qa[:, 2] - zs[i]) <= 0.5 * st + 1e-9
        if not m.any():
            continue
        q = Qa[m]
        cx, cy = np.interp(q[:, 2], zT_inc, Tp[::-1, 0]), np.interp(q[:, 2], zT_inc, Tp[::-1, 1])
        semi = np.interp(q[:, 2], zT_inc, (r_lab / cosT)[::-1]) + margin
        dT = np.hypot(q[:, 0] - cx, q[:, 1] - cy)
        others = [c for c in comps[i] if c is not by_i[i]]
        dO = np.min([np.hypot(q[:, 0] - c["x"], q[:, 1] - c["y"]) for c in others], axis=0) if others else np.full(len(q), np.inf)
        sel_T[np.nonzero(m)[0][(dT <= semi) & (dT < dO)]] = True
    sel_L = np.zeros(len(Qa), bool)
    zL_inc = lobe[::-1, 2]
    mL = (Qa[:, 2] <= lobe[0, 2] + 0.5 * st) & (Qa[:, 2] >= lobe[-1, 2] - 0.5 * st)
    q = Qa[mL]
    lx, ly = np.interp(q[:, 2], zL_inc, lobe[::-1, 0]), np.interp(q[:, 2], zL_inc, lobe[::-1, 1])
    sel_L[np.nonzero(mL)[0][(np.hypot(q[:, 0] - lx, q[:, 1] - ly) <= r_lab / cos_b + margin) & (q[:, 1] <= ly + 0.5)]] = True
    geo = dict(
        label_file=lab["file"], ovoid_label_file=P["data"] + "/BT_MRI_label_ovoid.nii",
        frame=dict(origin_BT_world=O, R_rows_BT_world=R, axis_BT_world=fr["axis"], n_slabs=fr["n_slabs"],
                   slab_line_rms_mm=fr["slab_line_rms_mm"], angle_to_json_axis_deg=fr["angle_to_json_axis_deg"],
                   origin_from_json_flange_mm=fr["origin_from_json_flange_mm"],
                   definition="axis = slab-centroid line (%g mm slabs, voxels < %g mm from the json axis) of applicator minus "
                              "ovoid over s %s mm above the json flange; origin = the json flange projected onto it; x = BT "
                              "world x orthogonalised, y = z cross x" % (val(prm, "label_axis_slab_mm"), lat_max,
                                                                        val(prm, "label_axis_s_mm"))),
        ring=ring, tube=tube,
        tandem=dict(stations_xyz_req=np.c_[Tp, r_eqT], tilt_deg=np.degrees(np.arccos(np.clip(cosT, -1, 1))),
                    z_first_mm=float(Tp[0, 2]), z_last_mm=float(Tp[-1, 2]), n_stations=int(len(Tp)), r_label_mm=r_lab,
                    r_eq_raw_median_mm=float(np.median(r_eqT)), n_label_vox=int(sel_T.sum()),
                    definition="posterior rod leaving the ring bore, traced on %g mm z planes from the first section within "
                               "%g mm of the axis below the ring's on-axis bottom until it merges into the bundle; "
                               "r_label = median of r_eq * sqrt(cos tilt)" % (st, start_off)),
        bundle=dict(stations_xyz_req=np.c_[Bc, rB], full=full, axis_app_down=d_b,
                    angle_to_tube_deg=geom.angle_deg(d_b, -EZ), median_r_eq_mm=float(np.median(rB)),
                    z_range_mm=[float(Bc[:, 2].max()), float(Bc[:, 2].min())], label_end_z_mm=float(Qa[:, 2].min()),
                    chord_angle_to_tube_deg=geom.angle_deg(geom.unit(Bc[full][-1] - Bc[full][0]), -EZ),
                    chord_all_angle_to_tube_deg=geom.angle_deg(geom.unit(Bc[-1] - Bc[0]), -EZ),
                    tandem_lobe_xyz=lobe, n_lobe_label_vox=int(sel_L.sum()),
                    lobe_line_angle_to_tube_deg=geom.angle_deg(geom.unit(lobe[-1] - lobe[0]), -EZ),
                    definition="largest section below the tandem with r_eq > rod_merge_req_mm; axis = PCA line through the "
                               "centroids of the plateau sections (r_eq >= bundle_full_frac x median); chord angles = first -> "
                               "last plateau / any section (the fix plan's 27.7 deg is such a chord, json frame); tandem lobe "
                               "= (mean x of the posterior band, posterior edge + r_label / cos(bundle angle))"),
        stations=[dict(z=float(z), comps=[[c["x"], c["y"], c["r_eq"]] for c in cs]) for z, cs in zip(zs, comps)],
        stations_note="every applicator-minus-ovoid component (x, y, r_eq) per z plane below the ring: the rod data S4b needs")
    # the tube part as the a3 / a14 analysis split it (the reference the S4c tube threshold was set on): applicator minus
    # ovoid in the IMAGE slices above the ovoid label's highest slice.  It leaves out the part of the 0-12 mm 'collar' of
    # label around the tube (fix plan Q8: air in the fornix or device?) that shares slices with the tilted ring.
    k_top = int(np.nonzero(lab["ovoid"].any((0, 1)))[0].max())
    m3 = lab["app"] & ~lab["ovoid"]
    m3[:, :, :k_top + 1] = False
    vox = dict(all=Qa, ovoid=Qo, tube=Qt, tube_a3=to_lf(_vox_world(m3, lab["aff"])), tandem=Qa[sel_T], tandem_in_bundle=Qa[sel_L],
               idx_tube=idx_a[Qa[:, 2] > ring["z_top_mm"]], idx_tandem=idx_a[sel_T], idx_tandem_in_bundle=idx_a[sel_L])
    geo["tube"]["a3_split"] = dict(ovoid_top_image_slice=k_top, n_vox=int(len(vox["tube_a3"])),
                                   definition="applicator minus ovoid in the image slices above the ovoid label's top slice "
                                              "(scratch wf_applicator/a3_parts.py, the split of the a14 reference 1.07 / 4.33 mm)")
    print("[label v4] frame: %d slabs, rms %.2f mm, %.2f deg from the json axis, origin %.2f mm from the json flange; ring on "
          "axis z %.1f..%.1f; tube tip z %.1f, r_eq %.2f; tandem z %.1f..%.1f (%d sections, r_label %.2f); bundle z %.1f..%.1f, "
          "axis %.1f deg from the tube" % (fr["n_slabs"], fr["slab_line_rms_mm"], fr["angle_to_json_axis_deg"],
                                           fr["origin_from_json_flange_mm"], ring["z_bottom_on_axis_mm"], ring["z_top_on_axis_mm"],
                                           z_tip, tube["r_eq_mm"], Tp[0, 2], Tp[-1, 2], len(Tp), r_lab, Bc[0, 2], Bc[-1, 2],
                                           geo["bundle"]["angle_to_tube_deg"]), flush=True)
    return geo, vox


def sweep_polyline(C, r, n_th, e1_ref=EX):
    """Closed tube of radius r along the polyline C (n, 3), each ring oriented by PARALLEL-TRANSPORT frames: the first
    ring's e1 = e1_ref orthogonalised against the first tangent, every next e1 = the previous one turned by the minimal
    rotation between consecutive tangents (no twist about the curve; Frenet frames flip at inflections and on straight
    runs).  Tangent at an interior vertex = the bisector of its two segments.  Flat fan caps at both ends (sweep).
    Returns (V, F, check): check.min_spacing_ratio = min over bends of (shorter adjacent segment) / (r tan(bend / 2));
    below 1 the rings of the inner side of a bend could cross."""
    C = np.asarray(C, float)
    seg = np.diff(C, axis=0)
    Ls = np.linalg.norm(seg, axis=1)
    if len(C) < 2 or (Ls < 1e-9).any():
        raise ValueError("sweep_polyline: needs >= 2 distinct consecutive points")
    u = seg / Ls[:, None]
    mid = [u[i - 1] + u[i] for i in range(1, len(u))]
    if any(np.linalg.norm(m) < 1e-6 for m in mid):
        raise ValueError("sweep_polyline: the polyline folds back on itself")
    T = np.vstack([u[:1]] + [geom.unit(m)[None] for m in mid] + [u[-1:]])
    e1 = geom.ortho(e1_ref, T[0])
    rings = []
    for i in range(len(C)):
        if i > 0:
            e1 = geom.ortho(geom.rot_between(T[i - 1], T[i]) @ e1, T[i])
        rings.append((C[i], e1, np.cross(T[i], e1), r))
    V, F = sweep(rings, C[0], C[-1], n_th)
    ratios = [min(Ls[i - 1], Ls[i]) / (r * np.tan(0.5 * np.radians(geom.angle_deg(u[i - 1], u[i]))))
              for i in range(1, len(u)) if geom.angle_deg(u[i - 1], u[i]) > 1e-3]
    bends = [geom.angle_deg(u[i - 1], u[i]) for i in range(1, len(u))]
    return V, F, dict(min_spacing_ratio=float(min(ratios)) if ratios else None, max_bend_deg=float(max(bends)) if bends else 0.0,
                      n_rings=int(len(C)), arclength_mm=float(Ls.sum()))


def tandem_centreline_v4(geo, prm, F_fin, R_fin, P_intro, a_v):
    """Centreline (applicator frame) of the vaginal tandem: from shaft_overlap_mm inside the tube's bottom, through the
    traced label tandem and its lobe in the bundle (one 1-2-1 smoothing pass; the tube's bottom point is kept), then
    straight along the measured bundle axis until the end lies shaft_past_introitus_mm beyond the preBT introitus plane
    with the device at device_final (F_fin, R_fin rows).  The tube's bottom = one station above the first label section
    of the tandem: there the label first shows the rod itself below the ring (fix plan: straight to z_app ~ -22)."""
    st = float(val(prm, "label_station_mm"))
    T = np.asarray(geo["tandem"]["stations_xyz_req"], float)[:, :3]
    Lb = np.asarray(geo["bundle"]["tandem_lobe_xyz"], float)
    z_j = float(T[0, 2]) + st
    S = np.vstack([[0.0, 0.0, z_j], T, Lb])
    Sm = S.copy()
    Sm[1:-1] = 0.25 * S[:-2] + 0.5 * S[1:-1] + 0.25 * S[2:]
    C = np.vstack([[0.0, 0.0, z_j + float(val(prm, "shaft_overlap_mm"))], Sm])
    d = geom.unit(np.asarray(geo["bundle"]["axis_app_down"], float))
    R_fin = np.asarray(R_fin, float)
    a_v = geom.unit(a_v)
    k = float((d @ R_fin) @ a_v)
    if k > -0.3:
        raise ValueError("bundle axis does not point out of the vagina at device_final (cos %.2f to -a_v)" % -k)
    h0 = float((np.asarray(F_fin, float) + C[-1] @ R_fin - np.asarray(P_intro, float)) @ a_v)
    t_ext = max(0.0, (-float(val(prm, "shaft_past_introitus_mm")) - h0) / k)
    n = max(1, int(np.ceil(t_ext / float(val(prm, "shaft_ring_step_mm")))))
    C = np.vstack([C, C[-1] + np.outer(np.linspace(0.0, t_ext, n + 1)[1:], d)])
    end_h = float((np.asarray(F_fin, float) + C[-1] @ R_fin - np.asarray(P_intro, float)) @ a_v)
    info = dict(z_tube_bottom_mm=z_j, i_first_label=2, n_tandem=int(len(T)), n_lobe=int(len(Lb)),
                i_tandem=[2, 2 + len(T)], i_lobe=[2 + len(T), 2 + len(T) + len(Lb)],
                label_end_height_above_introitus_mm=h0, extension_mm=t_ext, end_beyond_introitus_mm=-end_h,
                end_app=C[-1], end_dir_app=d, arclength_mm=float(geom.arclength(C)[-1]),
                definition="C[0] = (0, 0, tube bottom + overlap), C[1] = the tube's bottom on the axis, C[i_tandem] = the "
                           "smoothed label tandem sections, C[i_lobe] = the tandem lobe in the bundle, then the straight "
                           "extension along the bundle axis")
    return C, info


def write_meshes_v4(VF):
    """Write the v4 tandem-body parts (tube.obj, shaft.obj) into APP."""
    os.makedirs(APP, exist_ok=True)
    for name in TANDEM_V4_PARTS:
        V, F = VF[name]
        geom.write_obj(os.path.join(APP, name + ".obj"), V, F,
                       header="%s of the v4 tandem body (tube + vaginal tandem = one rigid body), built from the updated BT "
                              "applicator label; %s" % (name, FRAME_APP_V4))


def build_meshes_v4(prm, C, z_tube_bottom, write=True):
    """tube.obj (r_tandem_mm, straight on the axis from z_tube_bottom to the hemispherical tip at L_iu) and shaft.obj
    (r_shaft_mm swept along C) in the v4 applicator frame.  Returns (stats per part, {part: (V, F)}).  write=False
    builds in memory only (main_tandem_only checks the pose.json records it must keep before it writes anything)."""
    n_th = int(val(prm, "mesh_n_theta"))
    L, r, rs = val(prm, "L_iu_mm"), val(prm, "r_tandem_mm"), val(prm, "r_shaft_mm")
    zb = float(z_tube_bottom)
    parts = {}
    n_cyl = int(np.ceil((L - r - zb) / 3.5)) + 1                  # ~3.5 mm ring spacing, as the v1-v3 tube
    rings = [((0.0, 0.0, z), EX, EY, r) for z in np.linspace(zb, L - r, n_cyl)]
    rings += [((0.0, 0.0, L - r + r * np.sin(ph)), EX, EY, r * np.cos(ph)) for ph in np.linspace(0, np.pi / 2, 8)[1:-1]]
    V, F = sweep(rings, (0.0, 0.0, zb), (0.0, 0.0, L), n_th)
    parts["tube"] = (V, F, mesh_stats(V, F, np.pi * r * r * (L - r - zb) + 2.0 / 3.0 * np.pi * r ** 3))
    V, F, chk = sweep_polyline(C, rs, n_th)
    st = mesh_stats(V, F, np.pi * rs * rs * chk["arclength_mm"])
    st["sweep_check"] = chk
    parts["shaft"] = (V, F, st)
    out, VF = {}, {}
    for name, (V, F, st) in parts.items():
        st["file"] = name + ".obj"
        out[name], VF[name] = st, (V, F)
        print("[applicator v4] %-6s tris %5d verts %5d open %d nonmanifold %d badorient %d vol %8.1f mm3 (ref %8.1f, %+.2f %%) "
              "closed=%s" % (name, st["tris"], st["verts"], st["open_edges"], st["nonmanifold_edges"],
                             st["inconsistent_orientation_edges"], st["signed_volume_mm3"], st["reference_volume_mm3"],
                             st["vol_err_pct"], st["closed_oriented"]), flush=True)
    if write:
        write_meshes_v4(VF)
    return out, VF


def _sdf(V, F):
    """Signed distance to a closed, outward-oriented triangle mesh (negative inside): vtkImplicitPolyDataDistance."""
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=True))
    conn = np.ascontiguousarray(np.asarray(F, np.int64).ravel())
    offs = np.arange(0, conn.size + 1, 3, dtype=np.int64)
    ca = vtk.vtkCellArray()
    ca.SetData(ns.numpy_to_vtkIdTypeArray(offs, deep=True), ns.numpy_to_vtkIdTypeArray(conn, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(pd)

    def ev(X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.array([f.EvaluateFunction(float(x[0]), float(x[1]), float(x[2])) for x in X])
    return ev


def _dist_stats(d):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    return dict(n=int(len(d)), mean_mm=float(d.mean()), p95_mm=float(np.percentile(d, 95)), max_mm=float(d.max()),
                frac_le_1p5mm=float((d <= 1.5).mean()))


def _mask_isosurface(mask, aff, level=0.5):
    """Triangle surface (BT world mm) of a 0/1 voxel mask at `level`: vtkMarchingCubes on the index grid (voxel centres at
    integer indices, as the nibabel affine), the mask cropped to its bounding box and zero-padded by one voxel so the
    surface closes, then carried to world by the affine.  Returns (V, F)."""
    import vtk
    from vtk.util import numpy_support as ns
    ijk = np.argwhere(mask)
    lo = np.maximum(ijk.min(0) - 1, 0)
    hi = np.minimum(ijk.max(0) + 2, np.asarray(mask.shape))
    sub = np.pad(np.asarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32), 1)
    img = vtk.vtkImageData()
    img.SetDimensions(*[int(n) for n in sub.shape])
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    img.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(sub.ravel(order="F")), deep=True))
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img)
    mc.SetValue(0, float(level))
    mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.Update()
    pd = mc.GetOutput()
    V = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float) + (lo - 1)
    F = ns.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
    return V @ np.asarray(aff, float)[:3, :3].T + np.asarray(aff, float)[:3, 3], F


def _surface_samples(V, F, density, seed):
    """Points uniform by area on a triangle surface (density per mm2, fixed seed: reproducible)."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    rng = np.random.default_rng(seed)
    n = int(np.ceil(A.sum() * float(density)))
    t = rng.choice(len(F), size=n, p=A / A.sum())
    s, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return ((1.0 - s)[:, None] * V[F[t, 0]] + (s * (1.0 - r2))[:, None] * V[F[t, 1]] + (s * r2)[:, None] * V[F[t, 2]])


def _polyline_param(C, Q):
    """Parameter u = j + t of each point's nearest point on the polyline C (segment j, fraction t): u = i at station i."""
    C, Q = np.asarray(C, float), np.asarray(Q, float)
    a, d = C[:-1], np.diff(C, axis=0)
    t = np.clip(np.einsum("nmk,mk->nm", Q[:, None, :] - a[None], d) / (d * d).sum(1)[None], 0.0, 1.0)
    dist = np.linalg.norm(Q[:, None, :] - (a[None] + t[..., None] * d[None]), axis=2)
    j = dist.argmin(1)
    return j + t[np.arange(len(Q)), j]


def model_to_label(VF, lab, geo, C, cl, prm, density=4.0, seed=0):
    """The MODEL-to-label side of the S4c distance (review L1: label-to-model = max(0, signed distance) over label voxels
    reads ~0 for any model that CONTAINS the label, so an oversized model passes it; this side grows with the excess).
    Points uniform on the tandem-body surface (`density` per mm2, fixed seed) in the label frame -> distance to the label:
    0 inside it (trilinear 0/1 mask >= 0.5), else the distance to the mask's 0.5 iso-surface (_mask_isosurface, carried
    into the label frame).  Label = updated applicator label U ovoid label (every device voxel: the model tube runs through
    the ring, the vaginal tandem through the three-rod bundle).  Scored only where the label shows the part:
      tube_above_ring    tube surface with z_app > ring top (includes the tip: a model tube longer than the label's shows)
      tube_above_collar  z_app > ring top + 12 mm (clear of the collar, Q8)
      tandem_rod         shaft surface whose nearest centreline point lies between the first traced tandem section and
                         the first bundle lobe (cl i_tandem .. i_lobe)
      tandem_in_bundle   between the first and the last bundle lobe (inside the bundle an oversize is hidden up to the
                         bundle's own width: r_eq ~7 mm around three rods)
    The tube inside the ring slab, the shaft's junction stations and its straight extension past the label end are not
    scored (no label to compare with)."""
    from scipy import ndimage as ndi
    O = np.asarray(geo["frame"]["origin_BT_world"], float)
    R = np.asarray(geo["frame"]["R_rows_BT_world"], float)
    ring = geo["ring"]
    U = lab["app"] | lab["ovoid"]
    Vs, Fs = _mask_isosurface(U, lab["aff"])
    dist = _sdf((Vs - O) @ R.T, Fs)                         # label frame; |value| = distance to the iso-surface
    inv = np.linalg.inv(lab["aff"])
    Uf = U.astype(np.float32)

    def m2l(Q):
        ijk = (O + Q @ R) @ inv[:3, :3].T + inv[:3, 3]
        out_ = ndi.map_coordinates(Uf, ijk.T, order=1, mode="constant", cval=0.0) < 0.5
        d = np.zeros(len(Q))
        if out_.any():
            d[out_] = np.abs(dist(Q[out_]))
        return d

    def stats(d):
        s = _dist_stats(d)
        if len(d):
            s["frac_outside_label"] = float((d > 0).mean())
        return s
    Pt = _surface_samples(*VF["tube"], density, seed)
    Ps = _surface_samples(*VF["shaft"], density, seed + 1)
    dt, ds = m2l(Pt), m2l(Ps)
    u = _polyline_param(C, Ps)
    i_t, i_l = cl["i_tandem"], cl["i_lobe"]
    sel = dict(tube_above_ring=(dt, Pt[:, 2] > ring["z_top_mm"]), tube_above_collar=(dt, Pt[:, 2] > ring["z_top_mm"] + 12.0),
               tandem_rod=(ds, (u >= i_t[0]) & (u < i_l[0])), tandem_in_bundle=(ds, (u >= i_l[0]) & (u <= i_l[1] - 1)))
    out = {k: stats(d[m]) for k, (d, m) in sel.items()}
    out["all_scored"] = stats(np.concatenate([dt[sel["tube_above_ring"][1]], ds[sel["tandem_rod"][1] | sel["tandem_in_bundle"][1]]]))
    out["tube_tip_model_minus_label_mm"] = float(val(prm, "L_iu_mm")) - float(geo["tube"]["z_tip_label_mm"])
    out["sampling"] = dict(density_per_mm2=float(density), seed=int(seed), n_tube=int(len(Pt)), n_shaft=int(len(Ps)),
                           label_surface_tris=int(len(Fs)), label_surface_verts=int(len(Vs)))
    out["definition"] = ("model surface point (uniform by area) -> distance OUTSIDE the label (applicator U ovoid; 0 inside "
                         "by the trilinear mask >= 0.5, else the distance to the mask's 0.5 iso-surface).  The other side "
                         "of label_to_model_mm: an oversized model shows here, not there.  tube_tip_model_minus_label_mm = "
                         "L_iu - the label tube's tip (z_tip_label_mm)")
    return out


def validate_device_vs_label(VF, geo, vox, lab, prm, C, cl, rule, frames, app_in, src_pose, src_name):
    """S4c: the v4 tandem body against the updated BT label, and its pose.  In-sample geometry fit (the parts were
    measured on this label): a consistency ceiling, not validation (fix plan S4).
      at the BT pose (the label frame is the part frame, so the parts sit where the label is):
        Dice(tandem body, applicator minus ovoid) on the BT grid (also outside the ring slab, and over tube + separable
        tandem only: the model has no ring and no ovoid rods, the label has both); label-to-model distance = max(0,
        signed distance to the body) of the label voxels of the tube, the tandem rod and the tandem lobe in the bundle;
        model-to-label distance (model_to_label: surface points of the body outside the label) -- the label-to-model
        side alone cannot see a model that is too LARGE (review L1)
      tube pose: device_final (this pose.json, re-derived) against the source variant's, and against both BT axes
      pelvis frame: at the device_final (= G32) tube pose the model's vaginal tandem stations against the label's, carried
        BT -> preBT by frames.BONE (y = x R^T + t)"""
    import evaluate as ev
    g = geo
    O = np.asarray(g["frame"]["origin_BT_world"], float)
    R = np.asarray(g["frame"]["R_rows_BT_world"], float)
    zL = R[2]
    L = float(val(prm, "L_iu_mm"))
    out = dict(frame="BT label frame at the BT pose (applicator_v4 parts placed with origin_BT_world / R_rows_BT_world)",
               note="IN-SAMPLE: the parts were measured on this label (fix plan S4: a consistency ceiling, not validation)")
    # ---- Dice on the BT grid
    Lm = lab["app"] & ~lab["ovoid"]
    M = np.zeros(lab["shape"], bool)
    for p in TANDEM_V4_PARTS:
        V, F = VF[p]
        M |= ev.voxelize(O + np.asarray(V, float) @ R, F, lab["shape"], lab["aff"])
    idx = np.argwhere(M | Lm)
    zq = (idx.astype(float) @ lab["aff"][:3, :3].T + lab["aff"][:3, 3] - O) @ zL
    ring = g["ring"]
    out_ring = (zq > ring["z_top_mm"]) | (zq < ring["z_bottom_on_axis_mm"])
    tz = out_ring & (zq >= g["tandem"]["z_last_mm"] - 0.5 * float(val(prm, "label_station_mm")))
    # the label's OWN tandem parts (tube with its collar, tandem rod, tandem lobe in the bundle), without the ovoid rods
    Lt = np.zeros(lab["shape"], bool)
    for k in ("idx_tube", "idx_tandem", "idx_tandem_in_bundle"):
        Lt[tuple(np.asarray(vox[k]).T)] = True
    no_collar = out_ring & ((zq < ring["z_bottom_on_axis_mm"]) | (zq > ring["z_top_mm"] + 12.0))

    def dice(sel, ref=None):
        ii = tuple(idx[sel].T)
        m, l_ = M[ii], (Lm if ref is None else ref)[ii]
        return float(2.0 * (m & l_).sum() / max(1, m.sum() + l_.sum()))
    out["dice"] = dict(vs_app_minus_ovoid=dice(np.ones(len(idx), bool)), outside_ring_slab=dice(out_ring),
                       tube_and_separable_tandem=dice(tz), vs_label_tandem_parts_outside_ring=dice(out_ring, Lt),
                       vs_label_tandem_parts_outside_ring_and_collar=dice(no_collar, Lt),
                       model_cc=float(M.sum() * lab["vv"] / 1000.0),
                       label_cc=float(Lm.sum() * lab["vv"] / 1000.0),
                       note="the label also holds the two ovoid rods and their bundle (below the ring) and the model tube "
                            "runs through the ring slab where the label is 'ovoid': those parts cap the first two numbers; "
                            "the third keeps only z_app above the ring and down to the last separable tandem section (it "
                            "still holds both ovoid rods).  vs_label_tandem_parts = against the label's own tandem parts "
                            "(tube with collar, tandem rod, tandem lobe in the bundle; label_check vox sets) outside the ring "
                            "slab, and also without the 12 mm collar zone above the ring (Q8).  Model voxels beyond the "
                            "image field (the shaft's extension) do not count")
    # ---- label -> model distances (label frame = part frame)
    sd = {p: _sdf(*VF[p]) for p in TANDEM_V4_PARTS}
    probe = np.array([[0.0, 0.0, 0.5 * L], C[len(C) // 2], [30.0, 30.0, 0.0]])
    s_t, s_s = float(sd["tube"](probe[:1])[0]), float(sd["shaft"](probe[1:2])[0])
    s_far = min(float(sd["tube"](probe[2:])[0]), float(sd["shaft"](probe[2:])[0]))
    if not (s_t < 0 and s_s < 0 and s_far > 0):
        raise RuntimeError("signed-distance sign check failed (inside %.2f / %.2f, far %.2f)" % (s_t, s_s, s_far))

    def l2m(Q):
        return np.maximum(0.0, np.minimum(sd["tube"](Q), sd["shaft"](Q)))
    Qt = vox["tube"]
    core = np.hypot(Qt[:, 0], Qt[:, 1]) < 4.0
    dT = l2m(Qt)
    above = Qt[:, 2] > ring["z_top_mm"] + 12.0
    out["label_to_model_mm"] = dict(tube=_dist_stats(l2m(vox["tube_a3"])), tube_incl_collar=_dist_stats(dT),
                                    tube_core_r_lt_4mm=_dist_stats(dT[core]), tube_above_collar=_dist_stats(dT[above]),
                                    tandem_rod=_dist_stats(l2m(vox["tandem"])),
                                    tandem_in_bundle=_dist_stats(l2m(vox["tandem_in_bundle"])),
                                    definition="label voxel centre -> max(0, signed distance to tube U shaft).  tube (GATED) = "
                                               "the a3 / a14 split (label_geometry.tube.a3_split), the reference the S4c "
                                               "threshold was set on; tube_incl_collar = every applicator-minus-ovoid voxel "
                                               "above the ovoid label in the label frame, with the 0-12 mm collar (Q8); "
                                               "tube_core = those within 4 mm of the axis; tube_above_collar = z_app > ring "
                                               "top + 12 mm.  ONE-SIDED: a model larger than the label reads ~0 here; "
                                               "see model_to_label_mm")
    # ---- model -> label distances (review L1: the other side; an oversized model shows here)
    m2l = model_to_label(VF, lab, g, C, cl, prm)
    lm_ = out["label_to_model_mm"]
    m2l["vs_label_to_model_thresholds"] = dict(
        note="REPORTED, not gated: the S4c label-to-model thresholds applied to this side for reference",
        tandem_rod_mean_le_1mm=bool(m2l["tandem_rod"]["mean_mm"] <= 1.0), tandem_rod_p95_le_2mm=bool(m2l["tandem_rod"]["p95_mm"] <= 2.0),
        tube_above_ring_mean_le_1p5mm=bool(m2l["tube_above_ring"]["mean_mm"] <= 1.5))
    out["model_to_label_mm"] = m2l
    for k, k2 in (("tube_above_ring", "tube"), ("tube_above_collar", "tube_above_collar"), ("tandem_rod", "tandem_rod"),
                  ("tandem_in_bundle", "tandem_in_bundle")):
        print("[S4c] INFO %-17s model-to-label mean %.3f / P95 %.3f / max %.3f mm (%.1f %% of %d points outside);  "
              "label-to-model (%s) mean %.3f / P95 %.3f mm" % (k, m2l[k]["mean_mm"], m2l[k]["p95_mm"], m2l[k]["max_mm"],
                                                                100.0 * m2l[k]["frac_outside_label"], m2l[k]["n"], k2,
                                                                lm_[k2]["mean_mm"], lm_[k2]["p95_mm"]), flush=True)
    # ---- ring bore (for S4b: the ring halves will have a central bore of r 4.0 mm)
    Vall = np.vstack([VF[p][0] for p in TANDEM_V4_PARTS])
    inr = (Vall[:, 2] <= ring["z_top_on_axis_mm"]) & (Vall[:, 2] >= ring["z_bottom_on_axis_mm"])
    out["ring_bore"] = dict(max_radius_in_ring_span_mm=float(np.hypot(Vall[inr, 0], Vall[inr, 1]).max()) if inr.any() else None,
                            ring_span_z_mm=[ring["z_bottom_on_axis_mm"], ring["z_top_on_axis_mm"]],
                            note="largest distance of a tandem-body vertex from the axis within the ring's on-axis span (S4b bore r 4.0)")
    # ---- tube pose: re-derived device_final vs the source variant, and the tube error in BT
    sdf_ = src_pose["device_final"]
    F_fin, a_fin, x_fin = (np.asarray(rule["flange"], float), geom.unit(rule["tube_axis"]), geom.unit(rule["x_app"]))
    pose = dict(source="applicator_%s/pose.json device_final" % src_name,
                flange_diff_mm=float(np.linalg.norm(F_fin - np.asarray(sdf_["flange"], float))),
                tube_axis_diff_deg=geom.angle_deg(a_fin, sdf_["tube_axis"]), roll_x_diff_deg=geom.angle_deg(x_fin, sdf_["x_app"]))
    fb = frames["BONE"]
    Rb, tb = np.asarray(fb["R"], float), np.asarray(fb["t"], float)
    F_bt = (F_fin - tb) @ Rb                                    # x_BT = R^T (y - t)
    a_bt = geom.unit(Rb.T @ a_fin)
    tip_bt = F_bt + L * a_bt
    zj = geom.unit(np.asarray(app_in["R_rows_BT_world"], float)[2])
    dO = F_bt - O
    pose["tube_error_BT"] = dict(
        angle_to_json_axis_deg=geom.angle_deg(a_bt, zj), angle_to_label_axis_deg=geom.angle_deg(a_bt, zL),
        tip_to_json_tip_mm=float(np.linalg.norm(tip_bt - np.asarray(app_in["tip_BT_world"], float))),
        tip_to_label_tube_tip_mm=float(np.linalg.norm(tip_bt - (O + g["tube"]["z_tip_label_mm"] * zL))),
        tip_to_label_axis_at_L_iu_mm=float(np.linalg.norm(tip_bt - (O + L * zL))),
        flange_to_label_origin_along_mm=float(dO @ zL), flange_to_label_origin_lateral_mm=float(np.linalg.norm(dO - (dO @ zL) * zL)),
        roll_model_x_vs_label_x_deg=geom.angle_deg(geom.ortho(Rb.T @ x_fin, zL), R[0]),
        frame="frames.BONE (validation/alignment.json), x_BT = R^T (y_pre - t)")
    out["tube_pose"] = pose
    # ---- pelvis frame: the vaginal tandem at the device_final tube pose vs the mapped label tandem
    R_fin = np.asarray(rule["R_rows"], float)

    def to_pre_model(p):
        return F_fin + np.atleast_2d(p) @ R_fin

    def to_pre_label(p):
        return (O + np.atleast_2d(p) @ R) @ Rb.T + tb
    pel = {}
    for key, raw, (i0, i1) in (("tandem_rod", np.asarray(g["tandem"]["stations_xyz_req"], float)[:, :3], cl["i_tandem"]),
                               ("tandem_in_bundle", np.asarray(g["bundle"]["tandem_lobe_xyz"], float), cl["i_lobe"])):
        d = to_pre_model(C[i0:i1]) - to_pre_label(raw)
        n = np.linalg.norm(d, axis=1)
        pel[key] = dict(n=int(len(n)), mean_mm=float(n.mean()), max_mm=float(n.max()), mean_dx_mm=float(d[:, 0].mean()),
                        mean_dy_mm=float(d[:, 1].mean()), mean_dz_mm=float(d[:, 2].mean()),
                        z_app_range_mm=[float(raw[0, 2]), float(raw[-1, 2])])
    # the same physical label sections, but the tandem geometry referred to the JSON frame (v1-v3, and the scratch
    # analysis vaf/v3_pelvis.py behind the fix plan's 3.1 mm): separates the frame choice from the geometry
    Fj = np.asarray(app_in["origin_BT_world"], float)
    Rj = np.asarray(app_in["R_rows_BT_world"], float)
    raw = np.asarray(g["tandem"]["stations_xyz_req"], float)[:, :3]
    Wlab = O + raw @ R
    dj = (F_fin + ((Wlab - Fj) @ Rj.T) @ R_fin) - (Wlab @ Rb.T + tb)
    nj = np.linalg.norm(dj, axis=1)
    F_L_pre = O @ Rb.T + tb
    pel["tandem_rod_if_referred_to_json_frame"] = dict(mean_mm=float(nj.mean()), max_mm=float(nj.max()),
                                                        mean_dx_mm=float(dj[:, 0].mean()), mean_dy_mm=float(dj[:, 1].mean()),
                                                        mean_dz_mm=float(dj[:, 2].mean()))
    pel["pose_decomposition"] = dict(flange_minus_label_origin_pre_mm=F_fin - F_L_pre,
                                     flange_offset_mm=float(np.linalg.norm(F_fin - F_L_pre)),
                                     tube_axis_vs_label_axis_deg=geom.angle_deg(a_fin, Rb @ zL),
                                     note="d(p) = (F_fin - BONE(origin)) + p (R_fin - R_label carried by BONE): the flange "
                                          "offset plus the 2.4 deg axis error times the lever arm; neither is geometry")
    g32 = _load_json_or_empty(P["out"] + "/hybrid/runs/G32/device_final.json")
    if g32:
        pel["G32_device_final_vs_pose"] = dict(flange_mm=float(np.linalg.norm(np.asarray(g32["flange_mm"], float) - F_fin)),
                                               tube_axis_deg=geom.angle_deg(g32["tube_axis"], a_fin),
                                               x_app_deg=geom.angle_deg(g32["x_app"], x_fin))
    pel["definition"] = ("model = the built centreline stations placed at device_final (F + p R_rows); label = the raw traced "
                         "sections placed at the BT label pose and carried BT -> preBT by frames.BONE; d = model - label "
                         "(preBT RAS: +x right, +y anterior, +z superior)")
    out["pelvis_frame"] = pel
    # ---- acceptance (fix plan S4a / S4c items that apply to the tandem body)
    lm = out["label_to_model_mm"]
    acc = [
        dict(test="tandem rod: label-to-model mean <= 1.0 mm", measured=lm["tandem_rod"]["mean_mm"], pass_=lm["tandem_rod"]["mean_mm"] <= 1.0),
        dict(test="tandem rod: label-to-model P95 <= 2.0 mm", measured=lm["tandem_rod"]["p95_mm"], pass_=lm["tandem_rod"]["p95_mm"] <= 2.0),
        dict(test="tube (a3/a14 split): label-to-model mean <= 1.5 mm (P95 reported; with the whole collar see tube_incl_collar)",
             measured=lm["tube"]["mean_mm"], p95_mm=lm["tube"]["p95_mm"], incl_collar_mean_mm=lm["tube_incl_collar"]["mean_mm"],
             pass_=lm["tube"]["mean_mm"] <= 1.5),
        dict(test="tube pose = source device_final within 0.1 mm / 0.05 deg (flange, axis, roll)",
             measured=[pose["flange_diff_mm"], pose["tube_axis_diff_deg"], pose["roll_x_diff_deg"]],
             pass_=pose["flange_diff_mm"] <= 0.1 and pose["tube_axis_diff_deg"] <= 0.05 and pose["roll_x_diff_deg"] <= 0.05),
        dict(test="pelvis frame, G32 tube pose: vaginal tandem within 3.5 mm of the mapped label tandem (max over the sections)",
             measured=pel["tandem_rod"]["max_mm"], mean_mm=pel["tandem_rod"]["mean_mm"], mean_dy_mm=pel["tandem_rod"]["mean_dy_mm"],
             pass_=pel["tandem_rod"]["max_mm"] <= 3.5),
        dict(test="Dice(tandem body, applicator minus ovoid): reported, no floor", measured=out["dice"]["vs_app_minus_ovoid"], pass_=True),
    ]
    for a_ in acc:
        a_["pass"] = bool(a_.pop("pass_"))
    out["acceptance"] = acc
    out["pass_all"] = all(a_["pass"] for a_ in acc)
    for a_ in acc:
        print("[S4c] %-4s %s: %s" % ("PASS" if a_["pass"] else "FAIL", a_["test"], jz(a_["measured"], 3)), flush=True)
    return out


def fig_v4_label_check(geo, vox, VF, C, check, fn):
    """Label frame, sagittal (y) and coronal (x) projections: the label parts against the v4 tandem body."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 11))
    for ax, (i, xl) in zip(axs, ((1, "y_app (mm, anterior ->)"), (0, "x_app (mm, patient right ->)"))):
        for key, col, s, lbl in (("ovoid", "navajowhite", 1, "ovoid label (ring; not built, S4b)"),
                                 ("all", "0.72", 1, "applicator minus ovoid (incl. the ovoid rods)"),
                                 ("tube", "tab:blue", 2, "label tube"), ("tandem", "tab:green", 2, "label tandem rod"),
                                 ("tandem_in_bundle", "tab:olive", 2, "label tandem in the bundle")):
            Q = vox[key]
            ax.scatter(Q[:, i], Q[:, 2], s=s, c=col, label=lbl, rasterized=True)
        for p, col in (("tube", "m"), ("shaft", "darkmagenta")):
            V = VF[p][0]
            ax.scatter(V[:, i], V[:, 2], s=0.4, c=col, alpha=0.5, rasterized=True)
        ax.plot(C[:, i], C[:, 2], "-", color="darkmagenta", lw=1.4, label="v4 vaginal tandem centreline (mesh dots around)")
        ax.plot([0, 0], [C[1, 2], float(np.max(VF["tube"][0][:, 2]))], "--", color="m", lw=1.2, label="v4 tube axis")
        ax.set_xlabel(xl)
        ax.set_ylabel("z_app (mm, along the tube; 0 = flange)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axs[0].legend(loc="lower left", fontsize=8, markerscale=4)
    lm, d, ml = check["label_to_model_mm"], check["dice"], check["model_to_label_mm"]
    fig.suptitle("applicator v4 tandem body vs the updated BT applicator label (BT label frame; IN-SAMPLE fit)\n"
                 "label-to-model: tandem rod mean %.2f / P95 %.2f mm, tube (a3 split) mean %.2f / P95 %.2f mm (with the whole "
                 "collar %.2f mm)\nmodel-to-label: tandem rod mean %.2f / P95 %.2f mm, tube above the ring mean %.2f / P95 "
                 "%.2f mm, in the bundle mean %.2f / P95 %.2f mm\nDice vs app-minus-ovoid %.3f, vs the label's tandem parts "
                 "outside the ring %.3f;  bundle axis %.1f deg from the tube" % (
                     lm["tandem_rod"]["mean_mm"], lm["tandem_rod"]["p95_mm"], lm["tube"]["mean_mm"], lm["tube"]["p95_mm"],
                     lm["tube_incl_collar"]["mean_mm"], ml["tandem_rod"]["mean_mm"], ml["tandem_rod"]["p95_mm"],
                     ml["tube_above_ring"]["mean_mm"], ml["tube_above_ring"]["p95_mm"], ml["tandem_in_bundle"]["mean_mm"],
                     ml["tandem_in_bundle"]["p95_mm"], d["vs_app_minus_ovoid"], d["vs_label_tandem_parts_outside_ring"],
                     geo["bundle"]["angle_to_tube_deg"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    fig.savefig(fn, dpi=90)
    plt.close(fig)
    print("[fig] wrote", fn, flush=True)


def _max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# pose.json records gated against THIS variant's tandem body (hybrid/tf_gate.py): written by tandem_first_record, never
# inherited from --pose-from, kept across a --tandem-only rebuild only while the body they were gated against is unchanged
BODY_PATH_RECORDS = ("insertion_path_tandem_first",)


def tandem_body_depends(params, pose, VF):
    """What a tandem-first record was gated against: the inputs hybrid/tf_gate.Geo reads from the applicator dir (the
    tube.obj / shaft.obj geometry, L_iu_mm, r_tandem_mm, the default Delta, its corpus entry and device_final's axis and
    rows).  mesh_sha256 = sha256 of each part's OBJ body exactly as geom.write_obj writes it (v lines at %.4f, f lines;
    no header), so an in-memory build and the file on disk hash alike.  params: applicator.json params; pose: pose.json;
    VF: {part: (V, F)}."""
    import hashlib
    sh = {}
    for p in TANDEM_V4_PARTS:
        V, F = VF[p]
        h = hashlib.sha256()
        h.update("".join("v %.4f %.4f %.4f\n" % tuple(v) for v in np.asarray(V, float)).encode("ascii"))
        h.update("".join("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1) for f in np.asarray(F, int)).encode("ascii"))
        sh[p] = h.hexdigest()
    delta = float(pose["default_flange_shift_mm"])
    by = pose["corpus"]["by_flange_shift_mm"]
    e = by[[k for k in by if abs(float(k) - delta) < 1e-9][0]]
    df = pose["device_final"]
    return dict(mesh_sha256=sh, L_iu_mm=float(params["L_iu_mm"]["value"]), r_tandem_mm=float(params["r_tandem_mm"]["value"]),
                default_flange_shift_mm=delta, flange=[float(x) for x in e["flange"]],
                T_preBT_to_target=np.asarray(e["T_preBT_to_target"], float).tolist(),
                tube_axis=[float(x) for x in df["tube_axis"]], R_rows=np.asarray(df["R_rows"], float).tolist(),
                definition="tandem_body_depends (applicator_venezia): the tandem body and pose a tandem-first record was "
                           "gated against; a --tandem-only rebuild keeps the record only while these are unchanged")


def _depends_on_disk(app_dir):
    """tandem_body_depends of what an applicator dir holds now, or (None, why) when it is incomplete."""
    fa, fp = os.path.join(app_dir, "applicator.json"), os.path.join(app_dir, "pose.json")
    missing = [f for f in [fa, fp] + [os.path.join(app_dir, p + ".obj") for p in TANDEM_V4_PARTS] if not os.path.exists(f)]
    if missing:
        return None, "missing %s" % [os.path.basename(f) for f in missing]
    try:
        VF = {p: geom.read_obj(os.path.join(app_dir, p + ".obj")) for p in TANDEM_V4_PARTS}
        return tandem_body_depends(json.load(open(fa))["params"], json.load(open(fp)), VF), None
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return None, "unreadable (%s: %s)" % (type(e).__name__, e)


def _depends_diff(ref, new, tol=1e-6):
    """Differences between two tandem_body_depends dicts (mesh hashes exact, numbers within tol)."""
    out = []
    for k in sorted((set(ref) | set(new)) - {"definition"}):
        if k not in ref or k not in new:
            out.append("%s: present on one side only" % k)
        elif k == "mesh_sha256":
            out += ["%s.obj geometry changed" % p for p in sorted(set(ref[k]) | set(new[k])) if ref[k].get(p) != new[k].get(p)]
        else:
            try:
                d = _max_abs_diff(ref[k], new[k])
            except ValueError:
                d = np.inf
            if not d <= tol:
                out.append("%s differs by %.3g" % (k, d))
    return out


def _tf_record_problems(rec, pose_new, deps_new, deps_old, why_old):
    """Why a kept insertion_path_tandem_first record would no longer match the rebuilt variant ([] = it still does).
    Same final-pose test as tandem_first_record, plus the body it was gated against: the record's depends_on, or for a
    record written before depends_on existed, the body on disk before this rebuild (tf_gate read it from there)."""
    probs = []
    try:
        prm = rec["params"]
        dF = _max_abs_diff(prm["F_fin"], pose_new["device_final"]["flange"])
        da = geom.angle_deg(prm["a_fin"], pose_new["device_final"]["tube_axis"])
    except (KeyError, TypeError, ValueError) as e:
        return ["record params unreadable (%s: %s)" % (type(e).__name__, e)]
    if dF > 1e-3 or da > 1e-3:
        probs.append("final pose moved: params F_fin / a_fin are %.3g mm / %.3g deg from the rebuilt device_final" % (dF, da))
    fs = rec.get("flange_shift_mm")
    if fs is None or abs(float(fs) - float(pose_new["default_flange_shift_mm"])) > 1e-9:
        probs.append("flange_shift_mm %s != the rebuilt default_flange_shift_mm %s" % (fs, pose_new["default_flange_shift_mm"]))
    ref, src = rec.get("depends_on"), "the record's depends_on"
    if ref is None:
        ref, src = deps_old, "the body on disk before this rebuild (record without depends_on)"
    if ref is None:
        probs.append("cannot verify the body the record was gated against: no depends_on in the record and the variant "
                     "dir was incomplete before this rebuild (%s)" % why_old)
    else:
        probs += ["%s (vs %s)" % (d, src) for d in _depends_diff(ref, deps_new)]
    return probs


def carry_path_records(old_pose, pose_new, deps_new, deps_old, why_old, force=False):
    """Review M8: a --tandem-only rebuild must not drop the insertion_path_* records other steps wrote into the variant's
    pose.json (tandem_first_record's insertion_path_tandem_first).  Every insertion_path_* key of the OLD pose.json that
    the rebuilt one does not carry is returned VERBATIM (full precision: the scene regenerates the rows from its params):
      BODY_PATH_RECORDS  kept when _tf_record_problems finds nothing; otherwise the rebuild is REFUSED (SystemExit before
                         anything is written) unless force, which moves the record to pose.json stale_records[key] (kept for
                         the audit, invisible to the scene: re-run hybrid/tf_gate.py search --write)
      any other key      kept, unchecked (reported)
    old pose.json stale_records are carried too.  Returns (keep, stale_records, messages)."""
    keep, stale, msgs = {}, dict(old_pose.get("stale_records") or {}), []
    for k, rec in old_pose.items():
        if not k.startswith("insertion_path") or k in pose_new:
            continue
        if k not in BODY_PATH_RECORDS:
            keep[k] = rec
            msgs.append("%s kept verbatim (not a body-gated record: unchecked)" % k)
            continue
        probs = _tf_record_problems(rec, pose_new, deps_new, deps_old, why_old)
        if not probs:
            keep[k] = rec
            msgs.append("%s kept verbatim (written %s; final pose, Delta and tandem body unchanged, checked against %s)"
                        % (k, rec.get("written"), "its depends_on" if rec.get("depends_on") else "the body on disk"))
        elif force:
            stale[k] = dict(record=rec, moved=time.strftime("%Y-%m-%d %H:%M:%S"), why=probs,
                            note="moved here by a --tandem-only --force rebuild: the record no longer matches this variant's "
                                 "body / pose; the scene does not read it (re-run hybrid/tf_gate.py search --write)")
            msgs.append("%s NO LONGER MATCHES the rebuilt variant (%s): --force moved it to stale_records" % (k, "; ".join(probs)))
        else:
            raise SystemExit("refusing to rebuild: pose.json %s no longer matches the rebuilt variant:\n  %s\nNothing was "
                             "written.  Re-run hybrid/tf_gate.py search --write after the rebuild, or pass --force to move the "
                             "record to pose.json stale_records." % (k, "\n  ".join(probs)))
    return keep, stale, msgs


def _dump_json_atomic(obj, fn):
    tmp = fn + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, fn)


def main_tandem_only(a):
    """v4 (fix plan S4a + S4c): the tandem body alone, measured on the updated BT applicator label, carrying the SAME
    validated tube pose as the source variant (--pose-from, default v3).  pose.json is the source's, copied, and checked
    field by field against a re-derivation by the same rule (pose_rule with the source's rule parameters); only the
    device-geometry fields (shaft end, cap centres) are replaced.  Writes applicator_<variant>/{tube.obj, shaft.obj,
    applicator.json, pose.json} and logs/applicator_venezia_<variant>.json; no ovoid, rod or packing file.
    A REBUILD keeps the insertion_path_* records other steps wrote into this variant's pose.json (tandem_first_record's
    insertion_path_tandem_first, review M8) verbatim, so the build and the record can run in either order; a body-gated
    record that no longer matches the rebuilt body / pose stops the rebuild before anything is written, unless --force
    (carry_path_records).  Everything is built and checked in memory first; pose.json is replaced atomically."""
    import copy
    t0 = time.time()
    if not a.variant:
        raise SystemExit("--tandem-only builds a NEW applicator variant: pass --variant (e.g. v4)")
    src = HY + "/applicator_" + a.pose_from
    if os.path.abspath(src) == os.path.abspath(APP):
        raise SystemExit("--pose-from must name another variant than --variant")
    old = _load_json_or_empty(APP + "/applicator.json")
    if old and not old.get("params", {}).get("tandem_only", {}).get("value", False):
        raise SystemExit("%s holds an applicator that is not tandem-only: refusing to overwrite it (pick a new --variant)" % APP)
    if os.path.isdir(APP):
        stray = sorted(f for f in os.listdir(APP) if f.endswith(".obj") and os.path.splitext(f)[0] not in TANDEM_V4_PARTS)
        if stray:
            raise SystemExit("%s already holds non-tandem parts %s: a tandem-only variant must not" % (APP, stray))
    # the records a rebuild must keep (M8), read before anything is written
    old_pose = {}
    if os.path.exists(APP + "/pose.json"):
        old_pose = json.load(open(APP + "/pose.json"))           # unreadable -> stop: it may hold a record to keep
    deps_old, why_old = _depends_on_disk(APP) if old_pose else (None, "no pose.json")
    for d in (LOGS, FIGS):
        os.makedirs(d, exist_ok=True)
    app_in = json.load(open(P["inputs"] + "/applicator.json"))
    bdev = json.load(open(P["out"] + "/final/bdev.json"))
    src_app, src_pose = json.load(open(src + "/applicator.json")), json.load(open(src + "/pose.json"))
    if src_pose.get("rule_version") != "v2":
        raise SystemExit("the source pose (%s) is not rule v2" % src)
    prm = default_params(app_in, bdev)
    dropped = v4_params(prm)
    # ---- the pose rule with the source's parameters (the validated tube pose is kept, fix plan S4a 'Pose rule')
    for k in POSE_RULE_KEYS:
        prm[k]["value"] = src_app["params"][k]["value"]
        prm[k]["source"] = "copied from applicator_%s (%s)" % (a.pose_from, src_app["params"][k]["source"])
    prm["angle_deg"]["source"] = ("POSE RULE, copied from applicator_%s: the anatomical angle between the preBT vaginal principal "
                                  "axis and the tube (fix plan S4a: 'an anatomical angle, not a device bend').  The v4 geometry "
                                  "does not use it" % a.pose_from)
    prm_rule = json.loads(json.dumps(src_app["params"]))     # the source's own parameter set: the SAME rule (its arc
    #                                                          shaft_end output is replaced by the v4 shaft end below)
    pre = load_pre()
    Pf, n_fix = vagina_fixed_point(pre, val(prm_rule, "vagina_fixed_inferior_mm"))
    rule_inputs = dict(O_pre=pre["O_pre"], L_end=pre["L_end"], a0=pre["a0"], internal_os=pre["internal_os"],
                       canal_above_L_end_mm=pre["canal_above_L_end_mm"], vagina_fixed_point=Pf, corpus_centroid=pre["uterus_X"].mean(0),
                       vagina_axis=pre["vagina"]["axis"], vagina_centroid=pre["vagina"]["centroid"])
    rule = pose_rule(rule_inputs, prm_rule)
    df_src = src_pose["device_final"]
    rederive = dict(flange_mm=_max_abs_diff(rule["flange"], df_src["flange"]), tube_axis=_max_abs_diff(rule["tube_axis"], df_src["tube_axis"]),
                    x_app=_max_abs_diff(rule["x_app"], df_src["x_app"]), R_rows=_max_abs_diff(rule["R_rows"], df_src["R_rows"]),
                    tip_mm=_max_abs_diff(rule["tip"], df_src["tip"]),
                    corpus_T=_max_abs_diff(rule["corpus_T"], src_pose["corpus"]["T_preBT_to_target"]),
                    insertion_path_F_mm=max(_max_abs_diff(q["F"], s["F"]) for q, s in zip(rule["insertion_path"]["keyframes"],
                                                                                          src_pose["insertion_path"]["keyframes"])))
    by = src_pose["corpus"]["by_flange_shift_mm"]
    for key in by:
        p2 = json.loads(json.dumps(prm_rule))
        p2["flange_shift_mm"]["value"] = float(key)
        r2 = pose_rule(rule_inputs, p2)
        rederive["by_flange_shift_%s" % key] = max(_max_abs_diff(r2["flange"], by[key]["flange"]),
                                                   _max_abs_diff(r2["corpus_T"], by[key]["T_preBT_to_target"]))
    cp = canal_path_record(rule, pre, prm_rule)
    rederive["insertion_path_canal_F_mm"] = max(_max_abs_diff(q["F"], s["F"]) for q, s in zip(cp["keyframes"],
                                                                                             src_pose["insertion_path_canal"]["keyframes"]))
    worst = max(rederive.values())
    print("[pose v4] pose rule re-derived with applicator_%s's parameters: max |difference| to its pose.json %.2e "
          "(flange %.2e mm)" % (a.pose_from, worst, rederive["flange_mm"]), flush=True)
    if worst > 1e-3:
        raise SystemExit("the re-derived pose differs from applicator_%s/pose.json by %.4g: not the same rule" % (a.pose_from, worst))
    # ---- the label, the centreline and the meshes
    lab = load_label_new(label_applicator_file(a.app_label))
    geo, vox = measure_label_geometry(lab, app_in, prm)
    vg = pre["vagina"]
    P_intro = vg["centroid"] + vg["proj_min"] * vg["axis"]
    C, cl = tandem_centreline_v4(geo, prm, rule["flange"], rule["R_rows"], P_intro, vg["axis"])
    mesh, VF = build_meshes_v4(prm, C, cl["z_tube_bottom_mm"], write=False)
    frames = load_frames()
    check = validate_device_vs_label(VF, geo, vox, lab, prm, C, cl, rule, frames, app_in, src_pose, a.pose_from)
    check["pose_rederivation_max_abs_diff"] = rederive
    # ---- applicator.json
    L = val(prm, "L_iu_mm")
    lm_ = dict(flange=[0.0, 0.0, 0.0],
               flange_note="applicator-frame origin = the flange: the inputs/applicator.json BT flange projected onto the label "
                           "tube axis (fix plan 'Frames'); the device pose (pose.json device_final) places this point",
               tip=[0.0, 0.0, L], tube_bottom=[0.0, 0.0, cl["z_tube_bottom_mm"]], shaft_start=C[0], junction=C[1],
               tandem_label_first=C[cl["i_tandem"][0]], tandem_label_last=C[cl["i_tandem"][1] - 1], bundle_lobe_last=C[cl["i_lobe"][1] - 1],
               shaft_end=C[-1], shaft_end_dir=cl["end_dir_app"], shaft_centreline=C, shaft_centreline_info=cl,
               ring_top_on_axis_z=geo["ring"]["z_top_on_axis_mm"], ring_bottom_on_axis_z=geo["ring"]["z_bottom_on_axis_mm"],
               cap_centres=[], cap_centres_note="no ovoid body in a tandem-only variant (S4b blocked on U1): the list is empty")
    app_out = dict(units="mm, deg", frame=FRAME_APP_V4, variant=a.variant,
                   device="Venezia-type tandem body v4: tube + vaginal tandem, ONE rigid body (the scene loads tube.obj + "
                          "shaft.obj as the tandem), built from the updated BT applicator label (fix plan S4a); the ring halves "
                          "with their rods are pending (S4b, decision U1).  Identity unconfirmed: RTPLAN / vendor geometry not available",
                   written=time.strftime("%Y-%m-%d %H:%M:%S"), params=prm, dropped_params=dropped,
                   dropped_params_note="ovoid-body, packing, rod and needle parameters of default_params: no such body in v4",
                   parts=mesh, tandem_parts=list(TANDEM_V4_PARTS), landmarks=lm_, label_geometry=geo, label_check=check,
                   BT_pose=dict(frame="BT world (nibabel affine of BT_MRI_label_*.nii)", origin_BT_world=geo["frame"]["origin_BT_world"],
                                R_rows_BT_world=geo["frame"]["R_rows_BT_world"],
                                note="the BT LABEL frame (fix plan 'Frames') = this variant's part frame at the BT pose: p_BT = "
                                     "origin + p_app @ R_rows.  v1-v3 used the json pose (BT_pose_json)"),
                   BT_pose_json=dict(origin_BT_world=app_in["origin_BT_world"], R_rows_BT_world=app_in["R_rows_BT_world"],
                                     note="inputs/applicator.json BT tandem pose (v1-v3 frame)"),
                   pose_from=dict(variant=a.pose_from, applicator_json=src + "/applicator.json", pose_json=src + "/pose.json"))
    app_json = jz(app_out)
    # ---- pose.json: the source's, with the device-geometry fields of v4 (never the source's body-gated records)
    pose_out = copy.deepcopy(src_pose)
    for k in BODY_PATH_RECORDS + ("stale_records",):
        pose_out.pop(k, None)
    R_fin, F_fin = np.asarray(rule["R_rows"], float), np.asarray(rule["flange"], float)
    pose_out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pose_out["device_final"]["shaft_end"] = jz(F_fin + C[-1] @ R_fin, 4)
    pose_out["device_final"]["shaft_end_dir"] = jz(cl["end_dir_app"] @ R_fin)
    pose_out["device_final"]["cap_centres_world"] = []
    pose_out["device_final"]["v4_note"] = ("tube pose identical to applicator_%s (copied; re-derived by the same rule, max |diff| "
                                           "%.1e); shaft_end = the far end of the v4 vaginal tandem, %.1f mm beyond the preBT "
                                           "introitus plane; no ovoid body" % (a.pose_from, worst, cl["end_beyond_introitus_mm"]))
    pose_out["v4"] = dict(pose_from=a.pose_from, pose_rederivation_max_abs_diff=rederive,
                          tandem_body=dict(parts=list(TANDEM_V4_PARTS), shaft_end_world=F_fin + C[-1] @ R_fin,
                                           tube_bottom_world=F_fin + np.array([0.0, 0.0, cl["z_tube_bottom_mm"]]) @ R_fin,
                                           introitus_point_preBT=P_intro, introitus_normal_preBT=vg["axis"],
                                           end_beyond_introitus_mm=cl["end_beyond_introitus_mm"]),
                          note="insertion_path_tandem_first (fix plan S5) is written by tandem_first_record (hybrid/tf_gate.py "
                               "search --write), not here; a rebuild keeps it verbatim while the body it was gated against "
                               "is unchanged (carry_path_records)")
    pose_json = jz(pose_out)
    keep, stale, msgs = carry_path_records(old_pose, pose_json, tandem_body_depends(app_json["params"], pose_json, VF),
                                           deps_old, why_old, force=getattr(a, "force", False))
    pose_json.update(keep)                                     # verbatim: NOT through jz (full-precision params)
    if stale:
        pose_json["stale_records"] = stale
    for m in msgs:
        print("[pose v4] %s" % m, flush=True)
    # ---- write (every check above passed): meshes, applicator.json, pose.json (atomic)
    os.makedirs(APP, exist_ok=True)
    write_meshes_v4(VF)
    json.dump(app_json, open(APP + "/applicator.json", "w"), indent=1)
    _dump_json_atomic(pose_json, APP + "/pose.json")
    print("[pose v4] wrote %s/{applicator.json, pose.json, tube.obj, shaft.obj}; kept %s" % (APP, sorted(keep) or "no record"),
          flush=True)
    if not a.no_figs:
        fig_v4_label_check(geo, vox, VF, C, check, os.path.join(FIGS, "applicator_%s_label_check.png" % a.variant))
    files = sorted(os.listdir(APP))
    log = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(a), wall_s=round(time.time() - t0, 1), files=files,
               label_check=check, label_geometry=dict(frame=geo["frame"], tube=geo["tube"], ring=geo["ring"],
                                                       tandem={k: v for k, v in geo["tandem"].items() if k != "stations_xyz_req"},
                                                       bundle={k: v for k, v in geo["bundle"].items() if k not in ("stations_xyz_req", "tandem_lobe_xyz", "full")}),
               centreline=cl, mesh=mesh)
    json.dump(jz(log), open(LOGS + "/applicator_venezia_%s.json" % a.variant, "w"), indent=1)
    print("[done v4] %s: files %s; S4c pass_all %s; wall %.1f s" % (APP, files, check["pass_all"], time.time() - t0), flush=True)


# ============================================================================ fix plan S5: the tandem-first record
TF_DEFINITION = (
    "Tandem-first insertion (G32 fix plan S5).  The validated corpus transform is split as T_final = Trans(L) o "
    "Rot(theta about p), p = O_true (the physician's external os, inputs/tandem_path.npz), L = T_final(p) - p.  "
    "V: the tip runs up the vaginal part of the physician's path to O_true, tube axis = the chord of the path over the "
    "tube length behind the tip, corpus at rest.  C: depth d 0 -> d_fin = L_iu - d_F, tip = T_C(w_r) tau(d) + w_r delta, "
    "T_C(w) = Trans(w L_C) o Rot(w theta about p); tau = the labelled canal for follow_mm, then straight along a_lc; "
    "tube axis = the line through the tip fitted to the traversed carried path, reached from the previous row by the "
    "least motion that keeps frac_min of the traversed lower canal within frac_tol_mm, blended onto a_fin at the end; "
    "w_r(d) = the smallest monotone weight that keeps the tandem body in the wall's rest lumen (hybrid/tf_gate.py); "
    "the last C row = the tandem at Trans(L_C - L) device_final.  L: tandem and corpus translate together by "
    "w_l (L - L_C), the last row = device_final and corpus.T_preBT_to_target.  L_C = 0 is the plan's design; a "
    "non-zero L_C is a flagged fallback (caudal traction, or part of the lift started in C).  Rows are regenerated "
    "from params by geom.tandem_first_path (scene_hybrid.build_schedule_tandem_first).")


def _tf_json(o):
    """numpy -> JSON at FULL precision (the scene regenerates the rows from params and compares them with `rows`)."""
    if isinstance(o, dict):
        return {k: _tf_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_tf_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return _tf_json(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def tandem_first_record(app_dir, params, info=None, wall=None):
    """Write pose.json["insertion_path_tandem_first"] of a TANDEM-ONLY applicator (applicator_v4, --tandem-only).

    params: the geom.tandem_first_path parameter dict found by hybrid/tf_gate.py search (tau polyline, corpus split,
    L_C, delta, final pose, V / C / L knots, axis-rule constants).  They are stored at full precision; the rows are
    regenerated from the STORED params (so the scene's own regeneration reproduces them bit for bit) and stored too,
    rounded to 1e-9, for the audit, for tf_metrics (tau_pts / tau_s / delta_mm / w_r_table) and for the scene's
    staleness check.  info: provenance (mode, fallback, lift_in_C, gate summary, search) merged into the record.
    Refuses a directory that is neither tandem-only nor a ring variant (params.ring_body: applicator_v5, whose tandem
    body is the same tube.obj / shaft.obj), and params whose final pose is not this pose.json's device_final.
    Only this one key of pose.json is written (atomically: temp file + os.replace); every other key, and whatever a
    --tandem-only build wrote, is left as it is.  depends_on = tandem_body_depends of the dir's tube.obj / shaft.obj,
    applicator.json and pose.json as they are now (what tf_gate read): a later --tandem-only rebuild keeps the record
    verbatim only while those are unchanged (carry_path_records), so the two steps can run in either order."""
    app = json.load(open(os.path.join(app_dir, "applicator.json")))
    prm_a = app.get("params", {})
    if not (prm_a.get("tandem_only", {}).get("value", False) or prm_a.get("ring_body", {}).get("value", False)):
        raise SystemExit("tandem_first_record: %s is neither a tandem-only applicator (applicator.json params.tandem_only) "
                         "nor a ring variant (params.ring_body)" % app_dir)
    fn = os.path.join(app_dir, "pose.json")
    pose = json.load(open(fn))
    missing = [p + ".obj" for p in TANDEM_V4_PARTS if not os.path.exists(os.path.join(app_dir, p + ".obj"))]
    if missing:
        raise SystemExit("tandem_first_record: %s has no %s (the record is gated against the tandem body)" % (app_dir, missing))
    deps = tandem_body_depends(app["params"], pose, {p: geom.read_obj(os.path.join(app_dir, p + ".obj")) for p in TANDEM_V4_PARTS})
    df = pose["device_final"]
    prm = json.loads(json.dumps(_tf_json(params)))                     # exactly what the scene will read
    dF = float(np.abs(np.asarray(prm["F_fin"], float) - np.asarray(df["flange"], float)).max())
    da = geom.angle_deg(prm["a_fin"], df["tube_axis"])
    if dF > 1e-3 or da > 1e-3:
        raise SystemExit("tandem_first_record: params' final pose is %.2e mm / %.2e deg from device_final" % (dF, da))
    rows = geom.tandem_first_path(prm)

    def rr(x):
        return np.round(np.asarray(x, float), 9).tolist()
    rec = dict(definition=TF_DEFINITION, frame=FRAME_PRE, units="mm, deg", version="tf_gate/1.0 (fix plan S5 pass 1)",
               written=time.strftime("%Y-%m-%d %H:%M:%S"), flange_shift_mm=float(pose["default_flange_shift_mm"]),
               wall=wall, params=prm,
               # tf_metrics (S0) reads these top-level keys: the planned path, delta and w_r(d)
               tau_pts=prm["tau_pts"], tau_s=prm["tau_s"], tau_follow_mm=prm["follow_mm"], delta_mm=prm["delta"],
               w_r_table=dict(d_mm=[k[0] for k in prm["C_knots"]], w_r=[k[1] for k in prm["C_knots"]]),
               n_rows=dict(V=len(prm["V_tip_s"]), C=len(prm["C_knots"]), L=len(prm["L_w"])),
               rows=[dict(i=i, phase=r["phase"], tip_s=round(r["tip_s"], 9), d_mm=round(r["d_mm"], 9),
                          w_r=round(r["w_r"], 9), w_l=round(r["w_l"], 9), F=rr(r["F"]), tube_axis=rr(r["tube_axis"]),
                          x_app=rr(r["R_rows"][0]), tip=rr(r["tip"]), T_corpus=rr(r["T_corpus"]))
                     for i, r in enumerate(rows)])
    if info:
        rec.update(_tf_json(info))
    rec["depends_on"] = deps
    pose["insertion_path_tandem_first"] = rec
    _dump_json_atomic(pose, fn)
    return rec


# ============================================================================ fix plan S4b / S4c: applicator_v5, the ring halves
# What the updated label and the BT ovoid label show below and around the ring (audit R1; the physician on G32: "the
# segmented ovoids outer diameter is about 40 mm, but your model has another ovoids rod outside the ovoids"): ONE flat
# disc about 40 mm across and about 20 mm thick, split at the device's sagittal plane into two lunar halves that clamp
# around the tube, a socket boss hanging under each half's anterior rim, and each half's own rod running down beside
# the vaginal tandem into the handle bundle.  v3 hung two 26 mm caps with free rods 16 mm in front of the tandem rod
# whose axis never entered a cap (+6.7 mm outside).  In v5 each half is ONE closed manifold (half-disc + boss + rod),
# the zero set of a signed-distance composite meshed by flying edges, so no rod exists outside the ovoids.
RING_V5_PARTS = ("ovoid_L", "ovoid_R")
RING_SIDES = (("L", "ovoid_L", -1.0), ("R", "ovoid_R", 1.0))    # (side, part, sign of the outward medial normal on x_r)
FRAME_APP_V5 = FRAME_APP_V4 + ("; the ring halves (ovoid_L.obj, ovoid_R.obj) are stored at their SEATED pose in this "
                               "frame: the scene places them with the tandem's pose (applicator.json ring.T_seat)")


def v5_params(prm, ring_outer_r=None):
    """The ring-half (v5) parameters, added to a copy of the tandem variant's (v4) applicator.json params, which stay as
    they are (the tandem body is v4's, byte for byte).  Returns the list of added keys."""
    added = []

    def add(k, v, unit, source, note=""):
        prm[k] = dict(value=v, unit=unit, source=source, **({"note": note} if note else {}))
        added.append(k)

    add("tandem_only", False, "-", "v5 (fix plan S4b): the v4 tandem body (tube + vaginal tandem, copied unchanged) PLUS the two "
        "lunar ring halves; the scene loads tube.obj + shaft.obj as the tandem and ovoid_L.obj + ovoid_R.obj as the ovoid body")
    add("ring_body", True, "-", "v5: two lunar ovoid halves, each ONE closed manifold = half-disc + socket boss + its own rod "
        "(no rod outside the ovoids; fix plan S4b, the physician's G32 comment)")
    add("ring_outer_r_mm", 20.0 if ring_outer_r is None else float(ring_outer_r), "mm",
        "DECISION U1 = label (user-confirmed): ring OUTER radius, the physician's 'segmented ovoids outer diameter is about "
        "40 mm'.  MEASURED beside it (label_geometry.ring_halves.outer_radius): the ovoid label's area-equivalent radius on "
        "its plateau, its outline circle fit and the BT MRI half-way edge (--ring-outer-r overrides)")
    add("ring_channel_diam_mm", 30.0, "mm", "VENDOR SPEC (the physician: 'ovoids have 30 mm diameter'; Venezia literature: "
        "22 / 26 / 30 mm = the diameter of the ring of SOURCE CHANNELS inside the lunar ovoids, not the outer diameter).  "
        "Recorded, not used by the geometry: applicator.json vendor_vs_label explains the correspondence")
    add("tandem_length_mm", 70.0, "mm", "VENDOR SPEC (the physician: 'the tandem used in the current case is 70 mm length').  "
        "Recorded, not used: the model tube keeps the MEASURED L_iu_mm (the validated pose); vendor_vs_label gives the "
        "label's tip-to-reference lengths")
    add("ring_bore_r_mm", 4.0, "mm", "DESIGN (fix plan S4b): central bore of the ring about the TUBE axis, so the tandem body "
        "never overlaps the ring at the seat (tube r_tandem_mm, vaginal tandem r_shaft_mm start inside it)")
    add("ring_slot_mm", 0.5, "mm", "DESIGN (fix plan S4b): medial slot between the two halves (the device's sagittal plane "
        "through the tube axis, azimuth measured from the rod pair)")
    add("ring_boss_r_mm", 4.0, "mm", "fix plan S4b: radius of the socket boss that joins each half's rod to its rim; its axis is "
        "MEASURED (rod top -> the median of the half's socket voxels)")
    add("r_ovoid_rod_mm", 3.12, "mm", "MEASURED (fix plan S4b: MRI FWHM of the ovoid rods).  label_geometry.ovoid_rods.<side>."
        "r_label_mm = this build's tilt-corrected section radius on the label")
    add("ring_profile_step_mm", 0.5, "mm", "NUMERICAL: spacing of the label-frame z planes on which the ovoid label's section "
        "(area, outline circle) is measured")
    add("ring_plateau_frac", 0.97, "-", "NUMERICAL: planes whose area-equivalent radius is >= this x the largest form the "
        "ring's full-size plateau (centre and radius are medians over it)")
    add("ring_column_grid_mm", 0.5, "mm", "NUMERICAL: xy spacing of the columns along the label z axis whose top / bottom give "
        "the ring's radial profile")
    add("ring_column_dz_mm", 0.25, "mm", "NUMERICAL: sampling step along each column (trilinear on the 0/1 mask, >= 0.5 = inside)")
    add("ring_face_rho_min_mm", 6.0, "mm", "NUMERICAL: the face / fillet fit uses radii from here to ring_outer_r_mm - 0.25 "
        "(inside this radius the label has a hub around the tube: reported as hub_*, not modelled)")
    add("ring_socket_sector_deg", 45.0, "deg", "NUMERICAL: half-width of the anterior sector (about the rod pair's azimuth) that "
        "holds the socket bosses: left out of the disc fits, searched for the boss voxels")
    add("ring_socket_out_mm", 1.0, "mm", "NUMERICAL: ovoid-label voxels more than this outside the fitted disc are socket "
        "voxels (the boss axis goes through their median)")
    add("ovoid_rod_min_off_mm", 10.0, "mm", "NUMERICAL: an ovoid-rod seed section lies at least this far from the tube axis "
        "(the tandem and the specks under the ring are nearer)")
    add("ovoid_rod_merge_ratio", 1.25, "-", "NUMERICAL: tracking stops when a section's r_eq exceeds this x the rod's median "
        "(the rods merge into the handle bundle) or both rods take the same section")
    add("ovoid_rod_full_tol", 0.15, "-", "NUMERICAL: sections within this fraction of the median r_eq are full sections (the "
        "rod line and radius use only them)")
    add("ring_mesh_h_mm", 0.3, "mm", "NUMERICAL: grid of the signed-distance composite each half is meshed from (flying edges)")
    add("ring_mesh_target_tris", 5000, "-", "NUMERICAL: triangles per half after quadric decimation (the tube / shaft have 1.5-1.8k)")
    add("ring_approach_D_mm", 100.0, "mm", "fix plan S7f: each half starts D down its own rod direction from its seat")
    add("ring_approach_dir", "rod_sagittal", "-", "direction each half backs out along (fix plan S7f: its rod direction): "
        "'rod_sagittal' (default) = its measured upper rod direction without the component along its medial normal, so "
        "the lateral offset stays the offset (MEASURED on this case: one rod converges medially, and backing its half out "
        "along it drifts the half into the vaginal tandem, so 'rod' needs about twice the offset; applicator.json "
        "ring.approach); 'rod' = the measured direction itself; 'handle' = the bundle axis")
    add("ring_approach_offset_min_mm", 5.0, "mm", "fix plan S4b / S7c(iii): the lateral offset along the half's outward medial "
        "normal is at least this; the smallest offset (0.5 mm steps) that keeps ring_clearance_min_mm on every row is kept")
    add("ring_approach_offset_max_mm", 12.0, "mm", "NUMERICAL: largest lateral offset tried")
    add("ring_click_mm", [15.0, 20.0], "mm", "fix plan S4b: each half clicks in (its lateral offset goes to 0) over the last "
        "15-20 mm of its approach (1 mm steps tried)")
    add("ring_click_shapes", [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], "-", "NUMERICAL: click-in profiles tried, offset(d) = o "
        "(1 - (1 - d / click)^p) for d < click (p = 1 linear; a larger p keeps the offset until the half is nearly at its "
        "seat height, then closes it medially.  MEASURED on this case: a half's bore clears the tube and the top of the "
        "vaginal tandem only in the last few mm of its rise, so the linear profiles fail at the minimum offset)")
    add("ring_approach_step_mm", 0.5, "mm", "NUMERICAL: spacing of the host-checked approach rows along d")
    add("ring_clearance_min_mm", 0.5, "mm", "fix plan S4b / S7c(iii): device-device clearance between the tandem body and each "
        "half on every approach row")
    add("ring_sample_density_per_mm2", 2.0, "1/mm2", "NUMERICAL: surface samples (plus the mesh vertices) per mm2 for the "
        "approach clearance")
    add("mri_edge_bands_mm", [[8.0, 14.0], [23.0, 32.0]], "mm", "NUMERICAL (report only): radial bands of the BT MRI whose "
        "median (inside, the ring's signal void) and 75th percentile (outside, tissue) set the half-way edge level")
    return added


def _circle_fit(p):
    """Least-squares circle through 2-D points: (cx, cy, r, rms)."""
    p = np.asarray(p, float)
    A = np.c_[2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))]
    c, *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
    r = float(np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2))
    res = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - r
    return float(c[0]), float(c[1]), r, float(np.sqrt((res ** 2).mean()))


def _azimuth_deg(x, y):
    """Azimuth in the label-frame xy plane: 0 = anterior (+y), positive toward the patient's right (+x)."""
    return np.degrees(np.arctan2(x, y))


def _ang_diff(a, b):
    return (np.asarray(a, float) - b + 180.0) % 360.0 - 180.0


def rounded_disc_sdf(rho, z, R, z_top, z_bot, f_top, f_bot):
    """Exact signed distance in the meridian plane (rho, z) of a disc of radius R between z_bot and z_top whose top and
    bottom outer edges are rounded with radii f_top / f_bot (a 2-D box with per-corner radii, revolved)."""
    zm, H = 0.5 * (z_top + z_bot), 0.5 * (z_top - z_bot)
    pz = np.asarray(z, float) - zm
    f = np.where(pz > 0.0, f_top, f_bot)
    qx = np.asarray(rho, float) - R + f
    qz = np.abs(pz) - H + f
    return np.minimum(np.maximum(qx, qz), 0.0) + np.hypot(np.maximum(qx, 0.0), np.maximum(qz, 0.0)) - f


def disc_profile(rho, R, z_top, z_bot, f_top, f_bot):
    """(top, bottom) surface heights of that disc at radius rho <= R."""
    rho = np.asarray(rho, float)
    top = np.where(rho <= R - f_top, z_top, z_top - f_top + np.sqrt(np.maximum(0.0, f_top ** 2 - (rho - R + f_top) ** 2)))
    bot = np.where(rho <= R - f_bot, z_bot, z_bot + f_bot - np.sqrt(np.maximum(0.0, f_bot ** 2 - (rho - R + f_bot) ** 2)))
    return top, bot


def _seg_dist(Q, a, b):
    """Distance of points Q (n, 3) to the segment a-b."""
    Q = np.atleast_2d(np.asarray(Q, float))
    a = np.asarray(a, float)
    ab = np.asarray(b, float) - a
    t = np.clip((Q - a) @ ab / max(1e-12, float(ab @ ab)), 0.0, 1.0)
    return np.linalg.norm(Q - a - t[:, None] * ab, axis=1)


def _polyline_dist(Q, C):
    """Distance of points Q to the polyline C (a chain of segments)."""
    C = np.asarray(C, float)
    out = np.full(len(np.atleast_2d(Q)), np.inf)
    for i in range(len(C) - 1):
        out = np.minimum(out, _seg_dist(Q, C[i], C[i + 1]))
    return out


def _line_z(c, d, z):
    """Point of the line c + t d at height z (d_z != 0)."""
    c, d = np.asarray(c, float), np.asarray(d, float)
    return c + ((float(z) - c[2]) / d[2]) * d


def _trace_ovoid_rods(Qa, pl, geo, prm):
    """The two ovoid rods on the tandem's own z planes below the ring (label frame).  Seed: the first plane with two
    sections other than the tandem, r_eq in [tandem_min_req_mm, rod_merge_req_mm], >= ovoid_rod_min_off_mm from the
    tube axis (L = the one at smaller x); then each is tracked downward like the tandem (nearest section to the linearly
    extrapolated position within track_max_jump_mm) until the two take the same section or a section outgrows
    ovoid_rod_merge_ratio x its median: the handle bundle.  Returns {side: dict(stations (x, y, z, r_eq), plane index
    list, comps)} and the plane list."""
    st = float(val(prm, "label_station_mm"))
    min_px, r_merge = int(val(prm, "rod_min_px")), float(val(prm, "rod_merge_req_mm"))
    ring0 = geo["ring"]
    z0 = -st * np.ceil(-ring0["z_bottom_on_axis_mm"] / st - 1e-9)
    zs = np.arange(z0, float(Qa[:, 2].min()) - st, -st)
    comps = [pl.components(float(z), min_px) for z in zs]
    Tp = np.asarray(geo["tandem"]["stations_xyz_req"], float)

    def t_comp(i):
        k = np.nonzero(np.abs(Tp[:, 2] - zs[i]) < 1e-6)[0]
        if not len(k):
            return None
        cand = [c for c in comps[i] if np.hypot(c["x"] - Tp[k[0], 0], c["y"] - Tp[k[0], 1]) < 1.0]
        return min(cand, key=lambda c: np.hypot(c["x"] - Tp[k[0], 0], c["y"] - Tp[k[0], 1])) if cand else None
    r_seed, r_track = float(val(prm, "tandem_min_req_mm")), float(val(prm, "track_min_req_mm"))
    off_min, jump = float(val(prm, "ovoid_rod_min_off_mm")), float(val(prm, "track_max_jump_mm"))
    ratio = float(val(prm, "ovoid_rod_merge_ratio"))

    def track(i0, pair):
        tracks = {"L": [(i0, pair[0])], "R": [(i0, pair[1])]}
        alive, stop = {"L": True, "R": True}, {}
        for i in range(i0 + 1, len(zs)):
            tc = t_comp(i)
            pick = {}
            for s in ("L", "R"):
                if not alive[s]:
                    continue
                tr = tracks[s]
                p = np.array([tr[-1][1]["x"], tr[-1][1]["y"]])
                if len(tr) >= 2:
                    p = 2.0 * p - np.array([tr[-2][1]["x"], tr[-2][1]["y"]])
                cand = [c for c in comps[i] if c is not tc and c["r_eq"] >= r_track
                        and np.hypot(c["x"] - p[0], c["y"] - p[1]) <= jump]
                if not cand:
                    alive[s], stop[s] = False, "no section within %.1f mm at z %.1f" % (jump, zs[i])
                    continue
                pick[s] = min(cand, key=lambda c: np.hypot(c["x"] - p[0], c["y"] - p[1]))
            if "L" in pick and "R" in pick and pick["L"] is pick["R"]:
                for s in ("L", "R"):
                    alive[s], stop[s] = False, "both rods take one section at z %.1f (the handle bundle)" % zs[i]
                break
            for s, c in pick.items():
                if len(tracks[s]) >= 3 and c["r_eq"] > ratio * float(np.median([t[1]["r_eq"] for t in tracks[s]])):
                    alive[s], stop[s] = False, "section r_eq %.2f > %.2f x the median at z %.1f (bundle)" % (c["r_eq"], ratio, zs[i])
                    continue
                tracks[s].append((i, c))
            if not (alive["L"] or alive["R"]):
                break
        return tracks, stop
    # seed = the first plane whose two rod-like sections off the axis are BOTH tracked over >= 4 planes (the specks
    # under the ring's rim pass the size test in single planes but do not continue)
    tracks, stop, seeds_tried = None, {}, []
    for i in range(len(zs)):
        tc = t_comp(i)
        cand = [c for c in comps[i] if c is not tc and r_seed <= c["r_eq"] <= r_merge and np.hypot(c["x"], c["y"]) >= off_min]
        if len(cand) < 2:
            continue
        cand = sorted(cand, key=lambda c: -c["r_eq"])[:2]
        if np.hypot(cand[0]["x"] - cand[1]["x"], cand[0]["y"] - cand[1]["y"]) < 2.0 * r_seed:
            continue
        trk, stp = track(i, sorted(cand, key=lambda c: c["x"]))
        seeds_tried.append(float(zs[i]))
        if min(len(trk["L"]), len(trk["R"])) >= 4:
            tracks, stop = trk, stp
            break
    if tracks is None:
        raise ValueError("ovoid rods: no plane below the ring starts two rod tracks of >= 4 sections (seeds tried at z %s)"
                         % seeds_tried)
    for s in ("L", "R"):
        stop[s] = "%s; seed planes tried at z %s" % (stop.get(s, "label end"), seeds_tried)
    out = {}
    for s in ("L", "R"):
        S = np.array([[c["x"], c["y"], zs[i], c["r_eq"]] for i, c in tracks[s]])
        if len(S) < 4:
            raise ValueError("ovoid rod %s: only %d sections traced" % (s, len(S)))
        out[s] = dict(stations=S, planes=[i for i, _ in tracks[s]], comps=[c for _, c in tracks[s]], stop=stop.get(s, "label end"))
    return out, zs, comps


def measure_ring_geometry(lab, geo, prm, C, cl):
    """S4b/S4c: the ring halves measured on the updated BT applicator label + the BT ovoid label, in the BT label frame
    (= the v4 part frame).  Nothing is hard-coded from an analysis script.
      ovoid rods   traced plane by plane (_trace_ovoid_rods); line = PCA of the full sections, pointing down; r_label =
                   median r_eq x sqrt(cos tilt); rod top = that line one plane above the first full section (where the
                   rod leaves the ovoid label's socket); the rod runs straight along its line to the plane through the
                   tandem shaft's end normal to the bundle axis (the handle bundle ends together)
      slot         the device's sagittal plane: through the tube axis toward the mean midpoint of the paired rod sections
      section      per z plane of the ovoid label (largest component): area-equivalent radius and an outline circle fit
                   (socket sector left out); plateau = r_eq >= ring_plateau_frac x the largest; centre = plateau median
      profile      columns along z: radial medians of the ovoid label's top / bottom outside the socket sector; the flat
                   faces and the two edge fillets are the least-squares fit of the rounded disc to them (R =
                   ring_outer_r_mm)
      bosses       ovoid-label voxels > ring_socket_out_mm outside that disc in the socket sector below the equator, split
                   at the slot: the boss axis runs from the rod top through their median, up to their 95th-percentile z
      MRI edge     (report only) the BT MRI's half-way edge on the plateau planes
    Returns (rg, rvox): rg is JSON-able (after jz); rvox holds label-frame voxel sets scored in S4c."""
    from scipy import ndimage as ndi
    from scipy.optimize import least_squares
    O = np.asarray(geo["frame"]["origin_BT_world"], float)
    R = np.asarray(geo["frame"]["R_rows_BT_world"], float)

    def to_lf(W):
        return (np.asarray(W, float) - O) @ R.T
    Qa = to_lf(_vox_world(lab["app"] & ~lab["ovoid"], lab["aff"]))
    Qo = to_lf(_vox_world(lab["ovoid"], lab["aff"]))
    ring0 = geo["ring"]
    st = float(val(prm, "label_station_mm"))
    pl = LabelPlanes(lab, O, R, float(val(prm, "label_plane_px_mm")))
    R_out = float(val(prm, "ring_outer_r_mm"))
    # ---- (a) the ovoid rods
    tr, zs, comps = _trace_ovoid_rods(Qa, pl, geo, prm)
    tol, margin = float(val(prm, "ovoid_rod_full_tol")), float(val(prm, "tandem_select_margin_mm"))
    d_b = geom.unit(np.asarray(geo["bundle"]["axis_app_down"], float))
    S_end = np.asarray(C[-1], float)
    rods, rvox = {}, dict(all=Qa, ovoid=Qo)
    for s in ("L", "R"):
        S = tr[s]["stations"]
        med = float(np.median(S[:, 3]))
        full = np.abs(S[:, 3] / med - 1.0) <= tol
        if full.sum() < 3:
            raise ValueError("ovoid rod %s: fewer than 3 full sections" % s)
        c0 = S[full, :3].mean(0)
        _, _, vt = np.linalg.svd(S[full, :3] - c0, full_matrices=False)
        d = geom.unit(vt[0] if vt[0][2] < 0 else -vt[0])
        cos_t = abs(float(d[2]))
        r_lab = float(np.median(S[full, 3]) * np.sqrt(cos_t))
        res = S[:, :3] - c0
        lat = np.linalg.norm(res - np.outer(res @ d, d), axis=1)
        top = _line_z(c0, d, S[0, 2] + st)
        knee = _line_z(c0, d, S[-1, 2] - 0.5 * st)
        t_end = float((S_end - knee) @ d_b)
        end = knee + max(0.0, t_end) * d_b
        # label voxels of this rod (scored in S4c): the tandem's selection rule on the rod's own planes
        sel = np.zeros(len(Qa), bool)
        zi = S[::-1, 2]
        for i, c in zip(tr[s]["planes"], tr[s]["comps"]):
            m = np.abs(Qa[:, 2] - zs[i]) <= 0.5 * st + 1e-9
            if not m.any():
                continue
            q = Qa[m]
            cx, cy = np.interp(q[:, 2], zi, S[::-1, 0]), np.interp(q[:, 2], zi, S[::-1, 1])
            dT = np.hypot(q[:, 0] - cx, q[:, 1] - cy)
            others = [cc for cc in comps[i] if cc is not c]
            dO = (np.min([np.hypot(q[:, 0] - cc["x"], q[:, 1] - cc["y"]) for cc in others], axis=0) if others
                  else np.full(len(q), np.inf))
            sel[np.nonzero(m)[0][(dT <= r_lab / cos_t + margin) & (dT < dO)]] = True
        rvox["rod_" + s] = Qa[sel]
        rods[s] = dict(stations_xyz_req=S, full=full, n_stations=int(len(S)), n_full=int(full.sum()),
                       z_first_mm=float(S[0, 2]), z_last_mm=float(S[-1, 2]), stop=tr[s]["stop"],
                       line_point=c0, dir_app_down=d, tilt_to_tube_deg=geom.angle_deg(d, -EZ),
                       line_rms_mm=float(np.sqrt((lat[full] ** 2).mean())), line_max_dev_mm=float(lat.max()),
                       r_label_mm=r_lab, r_eq_median_mm=med, n_label_vox=int(sel.sum()),
                       top=top, knee=knee, end=end, knee_angle_deg=geom.angle_deg(d, d_b),
                       length_mm=float(np.linalg.norm(knee - top) + np.linalg.norm(end - knee)),
                       top_from_tube_axis_mm=float(np.hypot(top[0], top[1])),
                       definition="top = the rod line one plane above the first traced section (it leaves the socket "
                                  "there); knee = the line half a plane below the last separable section (the rods enter "
                                  "the handle bundle, where the label no longer separates them); below the knee the rod "
                                  "runs parallel to the measured bundle axis, as the tandem does, to the plane through "
                                  "the tandem shaft's end normal to that axis")
    pair = [i for i in tr["L"]["planes"] if i in tr["R"]["planes"]]
    SL = {i: p for i, p in zip(tr["L"]["planes"], tr["L"]["stations"])}
    SR = {i: p for i, p in zip(tr["R"]["planes"], tr["R"]["stations"])}
    mid = np.array([0.5 * (SL[i][:2] + SR[i][:2]) for i in pair])
    sep = np.array([np.linalg.norm(SL[i][:2] - SR[i][:2]) for i in pair])
    phi = float(_azimuth_deg(mid[:, 0].mean(), mid[:, 1].mean()))
    y_r = np.array([np.sin(np.radians(phi)), np.cos(np.radians(phi)), 0.0])
    x_r = np.cross(y_r, EZ)
    sector = float(val(prm, "ring_socket_sector_deg"))
    # ---- (b) sections of the ovoid label: area-equivalent radius and outline circle, per z plane
    h_p = float(val(prm, "ring_profile_step_mm"))
    prof = []
    for z in np.arange(ring0["z_top_mm"], ring0["z_bottom_mm"] - 1e-9, -h_p):
        m = pl._sample(pl.ov, float(z))
        lab_, n = ndi.label(m)
        if n == 0:
            continue
        k = int(np.argmax(np.bincount(lab_.ravel())[1:])) + 1
        mk = lab_ == k
        area = float(mk.sum()) * pl.px ** 2
        bd = mk & ~ndi.binary_erosion(mk)
        bx, by = pl.X[bd], pl.Y[bd]
        keep = np.abs(_ang_diff(_azimuth_deg(bx - pl.X[mk].mean(), by - pl.Y[mk].mean()), phi)) > sector
        fit = _circle_fit(np.c_[bx[keep], by[keep]]) if keep.sum() >= 20 else (np.nan,) * 4
        prof.append([float(z), area, float(np.sqrt(area / np.pi)), float(n), *fit])
    prof = np.array(prof)
    pla = prof[:, 2] >= float(val(prm, "ring_plateau_frac")) * prof[:, 2].max()
    pla &= np.isfinite(prof[:, 4])
    xc, yc = float(np.median(prof[pla, 4])), float(np.median(prof[pla, 5]))
    # ---- (c) radial profile of the label's top / bottom (columns along z), fit of the rounded disc
    hg, dz = float(val(prm, "ring_column_grid_mm")), float(val(prm, "ring_column_dz_mm"))
    g = np.arange(-(R_out + 3.0), R_out + 3.0 + 1e-9, hg)
    zc = np.arange(ring0["z_bottom_mm"] - 1.0, ring0["z_top_mm"] + 1.0 + 1e-9, dz)
    GX, GY, GZ = np.meshgrid(xc + g, yc + g, zc, indexing="ij")
    Qg = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    inv = np.linalg.inv(lab["aff"])
    vin = (ndi.map_coordinates(pl.ov, ((O + Qg @ R) @ inv[:3, :3].T + inv[:3, 3]).T, order=1, mode="constant", cval=0.0)
           >= 0.5).reshape(GX.shape)
    has = vin.any(2)
    ztop = np.where(has, zc[vin.shape[2] - 1 - np.argmax(vin[:, :, ::-1], axis=2)], np.nan)
    zbot = np.where(has, zc[np.argmax(vin, axis=2)], np.nan)
    CX, CY = GX[:, :, 0], GY[:, :, 0]
    rho = np.hypot(CX - xc, CY - yc)
    free = np.abs(_ang_diff(_azimuth_deg(CX - xc, CY - yc), phi)) > sector
    bins = np.arange(0.0, R_out + 3.0, 0.5)
    rp = []
    for b0 in bins:
        m = free & (rho >= b0) & (rho < b0 + 0.5)
        if m.sum() < 5:
            continue
        pres = float(has[m].mean())
        rp.append([b0 + 0.25, float(np.nanmedian(ztop[m])) if has[m].any() else np.nan,
                   float(np.nanmedian(zbot[m])) if has[m].any() else np.nan, pres, float(m.sum())])
    rp = np.array(rp)
    rho_min = float(val(prm, "ring_face_rho_min_mm"))
    fm = (rp[:, 0] >= rho_min) & (rp[:, 0] <= R_out - 0.25) & (rp[:, 3] >= 0.5) & np.isfinite(rp[:, 1])
    rb, tmed, bmed = rp[fm, 0], rp[fm, 1], rp[fm, 2]
    inner = rb < R_out - 8.0
    p0 = np.array([np.median(tmed[inner]), np.median(bmed[inner]), 4.0, 4.0])

    def resid(p):
        t_, b_ = disc_profile(rb, R_out, *p)
        return np.r_[t_ - tmed, b_ - bmed, 10.0 * max(0.0, p[2] + p[3] - (p[0] - p[1]) + 0.5)]
    lo = [p0[0] - 10.0, p0[1] - 10.0, 0.5, 0.5]
    hi = [p0[0] + 10.0, p0[1] + 10.0, 12.0, 12.0]
    sol = least_squares(resid, p0, bounds=(lo, hi))
    z_t, z_b, f_t, f_b = [float(v) for v in sol.x]
    r_fit = resid(sol.x)[:-1]
    free_r = free & has
    # ---- (d) sockets and bosses
    ds = rounded_disc_sdf(np.hypot(Qo[:, 0] - xc, Qo[:, 1] - yc), Qo[:, 2], R_out, z_t, z_b, f_t, f_b)
    zm = 0.5 * (z_t + z_b)
    sock = (ds > float(val(prm, "ring_socket_out_mm"))) & (Qo[:, 2] < zm) & \
        (np.abs(_ang_diff(_azimuth_deg(Qo[:, 0], Qo[:, 1]), phi)) <= sector)
    r_boss = float(val(prm, "ring_boss_r_mm"))
    halves = {}
    for s, part, sg in RING_SIDES:
        n_out = sg * x_r
        Sv = Qo[sock & ((Qo[:, :2] @ n_out[:2]) > 0.0)]
        rd = rods[s]
        top = np.asarray(rd["top"], float)
        if len(Sv) < 10:
            raise ValueError("ring half %s: only %d socket voxels" % (s, len(Sv)))
        Mv = np.median(Sv, axis=0)
        z_hi = float(np.percentile(Sv[:, 2], 95))
        u = Mv - top
        u = geom.unit(u) if u[2] > 1.0 else EZ.copy()
        B0 = top + ((z_hi - top[2]) / u[2]) * u
        # the boss must reach into the disc (fused): walk its top up along u until the capsule overlaps the disc by 1 mm
        n_walk = 0
        while rounded_disc_sdf(np.hypot(B0[0] - xc, B0[1] - yc), B0[2], R_out, z_t, z_b, f_t, f_b) > r_boss - 1.0 and n_walk < 40:
            B0 = B0 + 0.25 * u
            n_walk += 1
        e_rod = np.asarray(rd["dir_app_down"], float)
        mode = val(prm, "ring_approach_dir")
        if mode == "rod":
            e_app = e_rod
        elif mode == "rod_sagittal":
            e_app = geom.unit(e_rod - (e_rod @ x_r) * x_r)
        elif mode == "handle":
            e_app = d_b
        else:
            raise ValueError("ring_approach_dir must be 'rod', 'rod_sagittal' or 'handle' (got %r)" % mode)
        halves[s] = dict(part=part, n_out=n_out, boss_top=B0, boss_bottom=top, rod_top=top, rod_knee=rd["knee"],
                         rod_end=rd["end"], rod_pts=[top, rd["knee"], rd["end"]], rod_dir_down=e_rod, handle_dir_down=d_b,
                         approach_dir_down=e_app, approach_dir_mode=mode,
                         approach_dir_to_rod_deg=geom.angle_deg(e_app, e_rod),
                         n_socket_vox=int(len(Sv)), socket_vol_cc=float(len(Sv) * lab["vv"] / 1000.0),
                         socket_median=Mv, socket_z_range_mm=[float(Sv[:, 2].min()), float(Sv[:, 2].max())],
                         boss_len_mm=float(np.linalg.norm(B0 - top)), boss_tilt_to_tube_deg=geom.angle_deg(u, EZ),
                         boss_top_from_tube_axis_mm=float(np.hypot(B0[0], B0[1])),
                         rod_top_from_tube_axis_mm=float(np.hypot(top[0], top[1])),
                         boss_down_mm=float(B0[2] - top[2]),
                         boss_outward_mm=float(np.hypot(top[0], top[1]) - np.hypot(B0[0], B0[1])),
                         boss_top_walk_mm=0.25 * n_walk)
        rvox["socket_" + s] = Sv
    # ---- (e) MRI edge (report only)
    mri = _mri_ring_edge(O, R, (xc, yc), prof[pla, 0], prm)
    zs_pla = prof[pla, 0]
    outer = dict(value_mm=R_out, source=prm["ring_outer_r_mm"]["source"],
                 label_req_plateau_mm=dict(median=float(np.median(prof[pla, 2])), min=float(prof[pla, 2].min()),
                                           max=float(prof[pla, 2].max())),
                 label_outline_circle_mm=dict(median=float(np.median(prof[pla, 6])), min=float(prof[pla, 6].min()),
                                              max=float(prof[pla, 6].max()), rms_median=float(np.median(prof[pla, 7]))),
                 label_column_presence_half_mm=float(np.interp(0.5, rp[::-1, 3], rp[::-1, 0])) if (rp[:, 3] < 0.5).any() else None,
                 mri_half_way_edge_mm=mri,
                 definition="label_req = area-equivalent radius of the ovoid label's largest section on the plateau planes; "
                            "label_outline_circle = least-squares circle through that section's outline outside the socket "
                            "sector; label_column_presence_half = the radius at which half of the columns (outside the "
                            "socket sector) hold ovoid label; mri_half_way_edge = BT MRI, see _mri_ring_edge")
    rg = dict(
        frame="BT label frame = the v4 applicator frame (mm): origin = flange, z = label tube axis, x = BT world x "
              "orthogonalised, y = anterior",
        centre_xy=[xc, yc], centre_from_tube_axis_mm=float(np.hypot(xc, yc)),
        R_out=R_out, z_top=z_t, z_bottom=z_b, fillet_top=f_t, fillet_bottom=f_b, z_mid=zm, thickness_mm=z_t - z_b,
        r_bore=float(val(prm, "ring_bore_r_mm")), slot=float(val(prm, "ring_slot_mm")), r_boss=r_boss,
        r_rod=float(val(prm, "r_ovoid_rod_mm")),
        slot_azimuth_deg=phi, x_r=x_r, y_r=y_r,
        z_last_separable_mm=float(min(rods["L"]["z_last_mm"], rods["R"]["z_last_mm"])),
        rod_pair=dict(n_paired=int(len(pair)), separation_mm=dict(mean=float(sep.mean()), min=float(sep.min()), max=float(sep.max())),
                      midpoint_mean_xy=mid.mean(0), slot_azimuth_deg=phi,
                      definition="paired sections = the planes where both rods were traced; the slot plane contains the "
                                 "tube axis and points to the mean midpoint (azimuth from +y toward +x)"),
        halves=halves, outer_radius=outer,
        profile_fit=dict(z_top_mm=z_t, z_bottom_mm=z_b, fillet_top_mm=f_t, fillet_bottom_mm=f_b, R_mm=R_out,
                         rms_mm=float(np.sqrt((r_fit ** 2).mean())), n_bins=int(fm.sum()), rho_range_mm=[float(rb.min()), float(rb.max())],
                         radial_median_table=rp,
                         radial_median_table_columns="rho_mm, median top z, median bottom z, fraction of columns with label, n columns",
                         hub=dict(top_median_mm=float(np.nanmedian(ztop[free_r & (rho < rho_min)])),
                                  bottom_median_mm=float(np.nanmedian(zbot[free_r & (rho < rho_min)])),
                                  note="inside ring_face_rho_min_mm (around the tube) the label's top rises above the "
                                       "fitted face and its bottom sits above it: reported, not modelled"),
                         definition="least-squares fit of the rounded disc's top / bottom heights (flat faces z_top / "
                                    "z_bottom, circular edge fillets) to the radial medians of the ovoid label's column "
                                    "tops / bottoms outside the socket sector, rho in [ring_face_rho_min_mm, R - 0.25]"),
        sections=dict(table=prof, columns="z_mm, area_mm2, r_eq_mm, n_components, circle cx, cy, r, rms (socket sector left out)",
                      plateau_z_mm=[float(zs_pla.max()), float(zs_pla.min())], n_plateau=int(pla.sum()),
                      centre_spread_mm=[float(np.ptp(prof[pla, 4])), float(np.ptp(prof[pla, 5]))]),
        ovoid_label=dict(z_top_mm=ring0["z_top_mm"], z_bottom_mm=ring0["z_bottom_mm"], vol_cc=ring0["vol_cc"],
                         n_socket_vox=int(sock.sum()), socket_vol_cc=float(sock.sum() * lab["vv"] / 1000.0)))
    return rg, rods, rvox


def _mri_ring_edge(O, R, centre, zs, prm):
    """BT MRI half-way edge of the ring (its signal void) on label-frame planes zs: rays every 7.5 deg from the centre;
    level = half-way between the median of the inside band and the 75th percentile of the outside band
    (mri_edge_bands_mm); first crossing beyond the inside band; circle fit with a 2.5 mm outlier cut, re-centred 3x.
    Report only (ring_outer_r_mm is the U1 decision).  None when the image is missing."""
    import nibabel as nib
    from scipy import ndimage as ndi
    fn = P["data"] + "/BT_MRI.nii"
    if not os.path.exists(fn):
        return None
    im = nib.load(fn)
    img = np.asarray(im.dataobj).astype(np.float32)
    inv = np.linalg.inv(im.affine)
    (i0, i1), (o0, o1) = [tuple(float(v) for v in b) for b in val(prm, "mri_edge_bands_mm")]
    rs = np.arange(0.0, o1 + 3.0, 0.2)
    out = []
    for z in np.arange(float(max(zs)), float(min(zs)) - 1e-9, -1.0):
        c = np.array(centre, float)
        fit = None
        for _ in range(3):
            pts = []
            for a in np.radians(np.arange(0.0, 360.0, 7.5)):
                Pp = np.stack([c[0] + rs * np.sin(a), c[1] + rs * np.cos(a), np.full(rs.size, z)], 1)
                v = ndi.map_coordinates(img, ((O + Pp @ R) @ inv[:3, :3].T + inv[:3, 3]).T, order=1, mode="nearest")
                lev = 0.5 * (np.median(v[(rs > i0) & (rs < i1)]) + np.percentile(v[(rs > o0) & (rs < o1)], 75))
                j = np.nonzero((rs > i1 - 2.0) & (v > lev))[0]
                if len(j):
                    pts.append(Pp[j[0], :2])
            if len(pts) < 12:
                break
            pts = np.array(pts)
            cx, cy, r, _ = _circle_fit(pts)
            k = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 2.5
            if k.sum() < 12:
                break
            fit = _circle_fit(pts[k])
            c = np.array(fit[:2])
        if fit is not None:
            out.append([float(z), *fit])
    if not out:
        return None
    out = np.array(out)
    return dict(median=float(np.median(out[:, 3])), min=float(out[:, 3].min()), max=float(out[:, 3].max()),
                centre_median_xy=[float(np.median(out[:, 1])), float(np.median(out[:, 2]))], n_planes=int(len(out)),
                table=out, columns="z_mm, cx, cy, r, rms", image=fn)


def ring_half_sdf(Q, rg, side):
    """Signed-distance composite (negative inside; exact for the disc, capsule and rod pieces, a lower bound on the
    distance where the bore and the slot cut it) of ONE ring half in the applicator frame:
      disc   rounded_disc_sdf about the ring centre (flat faces z_bottom / z_top, fillets fillet_bottom / fillet_top)
      minus  the bore (r_bore about the TUBE axis) and everything on the far side of the slot plane (half the slot
             width from the plane through the tube axis along y_r)
      plus   the socket boss (capsule boss_top -> boss_bottom, r_boss; also cut at the slot) and the rod (capsule
             chain rod_top -> rod_knee -> rod_end, r_rod), so half, boss and rod are ONE solid."""
    Q = np.atleast_2d(np.asarray(Q, float))
    h = rg["halves"][side]
    c = rg["centre_xy"]
    d = rounded_disc_sdf(np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]), Q[:, 2], rg["R_out"], rg["z_top"], rg["z_bottom"],
                         rg["fillet_top"], rg["fillet_bottom"])
    d = np.maximum(d, rg["r_bore"] - np.hypot(Q[:, 0], Q[:, 1]))
    slot = 0.5 * rg["slot"] - Q[:, :2] @ np.asarray(h["n_out"], float)[:2]
    d = np.maximum(d, slot)
    db = np.maximum(_seg_dist(Q, h["boss_top"], h["boss_bottom"]) - rg["r_boss"], slot)
    dr = _polyline_dist(Q, h["rod_pts"]) - rg["r_rod"]
    return np.minimum(np.minimum(d, db), dr)


def _half_bbox(rg, side, pad):
    h = rg["halves"][side]
    c, Rr = np.asarray(rg["centre_xy"], float), rg["R_out"]
    lo = np.array([c[0] - Rr, c[1] - Rr, rg["z_bottom"]])
    hi = np.array([c[0] + Rr, c[1] + Rr, rg["z_top"]])
    for pts, r in (([h["boss_top"], h["boss_bottom"]], rg["r_boss"]), (h["rod_pts"], rg["r_rod"])):
        pts = np.asarray(pts, float)
        lo = np.minimum(lo, pts.min(0) - r)
        hi = np.maximum(hi, pts.max(0) + r)
    return lo - pad, hi + pad


def sdf_grid(fn, lo, hi, h, chunk=400000):
    """fn sampled on the regular grid lo + h * (i, j, k) covering [lo, hi]: (lo, h, A[i, j, k]) (float32)."""
    lo = np.asarray(lo, float)
    n = np.ceil((np.asarray(hi, float) - lo) / h).astype(int) + 1
    xs, ys, zs = [lo[i] + h * np.arange(n[i]) for i in range(3)]
    A = np.empty(tuple(int(v) for v in n), np.float32)
    per = max(1, chunk // int(n[0] * n[1]))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for k0 in range(0, int(n[2]), per):
        k1 = min(int(n[2]), k0 + per)
        m = k1 - k0
        Q = np.stack([np.repeat(X[..., None], m, 2).ravel(), np.repeat(Y[..., None], m, 2).ravel(),
                      np.broadcast_to(zs[k0:k1], (n[0], n[1], m)).ravel()], 1)
        A[:, :, k0:k1] = fn(Q).reshape(n[0], n[1], m)
    return lo, float(h), A


def grid_eval(grid, Q, cval):
    """Trilinear interpolation of an sdf_grid at points Q; cval outside the grid."""
    from scipy import ndimage as ndi
    lo, h, A = grid
    return ndi.map_coordinates(A, ((np.atleast_2d(Q) - lo) / h).T, order=1, mode="constant", cval=float(cval))


def _mesh_from_grid(grid, n_tris):
    """Zero level set of an sdf_grid (vtkFlyingEdges3D) -> quadric decimation to ~n_tris -> clean.  Returns (V, F,
    info) with outward orientation (signed volume > 0)."""
    import vtk
    from vtk.util import numpy_support as ns
    lo, h, A = grid
    img = vtk.vtkImageData()
    img.SetDimensions(*[int(v) for v in A.shape])
    img.SetSpacing(h, h, h)
    img.SetOrigin(*[float(v) for v in lo])
    img.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(A.ravel(order="F")), deep=True))
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.ComputeScalarsOff()
    fe.Update()
    n0 = fe.GetOutput().GetNumberOfCells()
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(fe.GetOutput())
    cl.Update()
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(cl.GetOutput())
    dec.SetTargetReduction(max(0.0, 1.0 - float(n_tris) / max(1, cl.GetOutput().GetNumberOfCells())))
    dec.VolumePreservationOn()
    dec.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(dec.GetOutput())
    tri.Update()
    cl2 = vtk.vtkCleanPolyData()
    cl2.SetInputData(tri.GetOutput())
    cl2.Update()
    out = cl2.GetOutput()
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(out)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    V = ns.vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    F = ns.vtk_to_numpy(out.GetPolys().GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
    if geom.mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F, dict(n_components=int(conn.GetNumberOfExtractedRegions()), mc_tris=int(n0),
                      mc_grid=[int(v) for v in A.shape], mc_h_mm=float(h))


def build_ring_meshes(rg, prm):
    """ovoid_L / ovoid_R (applicator frame, seated): each half's signed-distance composite (ring_half_sdf) on a
    ring_mesh_h_mm grid, meshed at 0 (flying edges) and decimated to ring_mesh_target_tris.  Checks per half: closed,
    oriented, ONE component; |composite sdf| at the mesh vertices (the decimation's deviation from the design).
    Returns (stats per part, {part: (V, F)}, {side: sdf grid})."""
    h = float(val(prm, "ring_mesh_h_mm"))
    stats, VF, grids = {}, {}, {}
    for s, part, _ in RING_SIDES:
        lo, hi = _half_bbox(rg, s, 3.0 * h + 1.0)
        grid = sdf_grid(lambda Q, s=s: ring_half_sdf(Q, rg, s), lo, hi, h)
        V, F, info = _mesh_from_grid(grid, int(val(prm, "ring_mesh_target_tris")))
        dv = np.abs(ring_half_sdf(V, rg, s))
        stv = mesh_stats(V, F, float((grid[2] <= 0).sum()) * h ** 3)
        stv.update(info)
        stv["vertex_abs_sdf_mm"] = dict(mean=float(dv.mean()), p95=float(np.percentile(dv, 95)), max=float(dv.max()))
        stv["reference_volume_note"] = "grid-cell count of the composite <= 0 (the design solid on the meshing grid)"
        stv["file"] = part + ".obj"
        stv["one_component"] = bool(info["n_components"] == 1)
        stats[part], VF[part], grids[s] = stv, (V, F), grid
        print("[applicator v5] %-7s tris %5d verts %5d open %d nonmanifold %d badorient %d components %d vol %8.1f mm3 "
              "(grid %8.1f, %+.2f %%) |sdf| at vertices mean %.3f max %.3f mm closed=%s" % (
                  part, stv["tris"], stv["verts"], stv["open_edges"], stv["nonmanifold_edges"],
                  stv["inconsistent_orientation_edges"], info["n_components"], stv["signed_volume_mm3"],
                  stv["reference_volume_mm3"], stv["vol_err_pct"], dv.mean(), dv.max(), stv["closed_oriented"]), flush=True)
    return stats, VF, grids


def tandem_body_sdf(C, prm, z_tube_bottom):
    """Signed distance (negative inside) of the tandem body as capsules: the tube (r_tandem_mm, axis from z_tube_bottom
    to the tip centre L_iu - r) and the vaginal tandem (r_shaft_mm about the centreline C).  The meshes' flat ends are
    inside these capsules, so a clearance measured against them is conservative."""
    L, r, rs = float(val(prm, "L_iu_mm")), float(val(prm, "r_tandem_mm")), float(val(prm, "r_shaft_mm"))
    a, b = np.array([0.0, 0.0, float(z_tube_bottom)]), np.array([0.0, 0.0, L - r])
    C = np.asarray(C, float)

    def f(Q):
        return np.minimum(_seg_dist(Q, a, b) - r, _polyline_dist(Q, C) - rs)
    return f


def _surface_points(V, F, density, seed):
    return np.vstack([np.asarray(V, float), _surface_samples(V, F, density, seed)])


def ring_approach(rg, VF, grids, C, prm, z_tube_bottom):
    """Fix plan S4b / S7c(iii) on the host: the per-half approach paths.  Half i (L first, then R) at approach distance
    d is displaced in the tandem frame by  disp_i(d) = d e_i + o_i(d) n_i,  e_i = its rod direction (down, out of the
    body), n_i = its outward medial normal (+/- x_r), o_i(d) = o for d >= click, o (1 - (1 - d / click)^p) below
    (the half clicks in over the last `click` mm); d runs from D (ring_approach_D_mm) to 0 = the seat.  Clearance of a
    row = min(tandem sdf at the displaced half's surface points, half sdf at the tandem's surface points) -- both
    signed-distance grids, the tandem as capsules (conservative), the half as its composite (a lower bound); the second
    half is also checked against the first, seated.  Kept per half: the smallest o >= ring_approach_offset_min_mm
    (0.5 mm steps), then the largest click in ring_click_mm, then the smallest p, with >= ring_clearance_min_mm to the
    tandem on every row (ring_approach_step_mm apart) and >= slot - 0.1 mm to the other half."""
    h = float(val(prm, "ring_mesh_h_mm"))
    dens = float(val(prm, "ring_sample_density_per_mm2"))
    D, step = float(val(prm, "ring_approach_D_mm")), float(val(prm, "ring_approach_step_mm"))
    thr = float(val(prm, "ring_clearance_min_mm"))
    thr_hh = rg["slot"] - 0.1
    ck_lo, ck_hi = [float(v) for v in val(prm, "ring_click_mm")]
    shapes = [float(v) for v in val(prm, "ring_click_shapes")]
    offs = np.arange(float(val(prm, "ring_approach_offset_min_mm")), float(val(prm, "ring_approach_offset_max_mm")) + 1e-9, 0.5)
    tsdf = tandem_body_sdf(C, prm, z_tube_bottom)
    PT = np.vstack([_surface_points(*VF[p], dens, 11 + k) for k, p in enumerate(TANDEM_V4_PARTS)])
    lo_t = np.minimum(PT.min(0), [-3.0, -3.0, z_tube_bottom]) - 4.0
    hi_t = PT.max(0) + 4.0
    hi_t[2] = min(hi_t[2], rg["z_top"] + 6.0)                     # a half never rises above its seat
    tgrid = sdf_grid(tsdf, lo_t, hi_t, h)
    PH = {s: _surface_points(*VF[part], dens, 21 + k) for k, (s, part, _) in enumerate(RING_SIDES)}
    ds = np.round(np.arange(0.0, D + 1e-9, step), 6)
    order = [s for s, _, _ in RING_SIDES]
    halves = rg["halves"]
    pad = 3.0 * h + 1.0

    def disp(s, d, o):
        return d * np.asarray(halves[s]["approach_dir_down"], float) + o * np.asarray(halves[s]["n_out"], float)

    def clear(s, v, other):
        c_t = min(float(grid_eval(tgrid, PH[s] + v, 4.0).min()), float(grid_eval(grids[s], PT - v, pad).min()))
        if other is None:
            return c_t, None
        c_h = min(float(grid_eval(grids[other], PH[s] + v, pad).min()), float(grid_eval(grids[s], PH[other] - v, pad).min()))
        return c_t, c_h

    def offset(d, o, click, p):
        return o if d >= click else o * (1.0 - (1.0 - d / click) ** p)
    out = {}
    for k, s in enumerate(order):
        other = order[0] if k == 1 else None
        memo = {}

        def row(d, o):
            key = (round(float(d), 6), round(float(o), 6))
            if key not in memo:
                memo[key] = clear(s, disp(s, d, o), other)
            return memo[key]

        def ok(c):
            return c[0] >= thr and (c[1] is None or c[1] >= thr_hh)
        chosen, best = None, None
        for o in offs:
            if not all(ok(row(d, o)) for d in ds if d >= ck_hi):
                continue
            for click in np.arange(ck_hi, ck_lo - 1e-9, -1.0):
                if not all(ok(row(d, o)) for d in ds if d >= click):
                    continue
                for p in shapes:
                    cs = [row(d, offset(d, o, click, p)) for d in ds]
                    worst = min(c[0] for c in cs)
                    if best is None or worst > best[0]:
                        best = (worst, float(o), float(click), p)
                    if all(ok(c) for c in cs):
                        chosen = (float(o), float(click), p)
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen is None and best is None:          # nothing even clears the far rows: keep the largest offset
            best = (None, float(offs[-1]), ck_hi, shapes[0])
        o, click, p = chosen if chosen else best[1:]
        rows_ = []
        for d in ds[::-1]:
            oo = offset(d, o, click, p)
            c = row(d, oo)
            rows_.append(dict(d_mm=float(d), offset_mm=float(oo), disp_app=disp(s, d, oo), clearance_tandem_mm=c[0],
                              clearance_other_half_mm=c[1]))
        ct = np.array([r_["clearance_tandem_mm"] for r_ in rows_])
        ch = np.array([r_["clearance_other_half_mm"] for r_ in rows_ if r_["clearance_other_half_mm"] is not None])
        keep = rows_[::2] if (len(rows_) - 1) % 2 == 0 else rows_[::2] + [rows_[-1]]
        out[s] = dict(order=k + 1, part=halves[s]["part"], e_app=halves[s]["approach_dir_down"],
                      e_mode=halves[s]["approach_dir_mode"], e_rod_down_app=halves[s]["rod_dir_down"], n_out_app=halves[s]["n_out"],
                      D_mm=D, offset_mm=o, click_mm=click, click_shape_p=p, pass_=bool(chosen is not None),
                      min_clearance_tandem_mm=float(ct.min()), d_at_min_mm=float(rows_[int(np.argmin(ct))]["d_mm"]),
                      min_clearance_other_half_mm=float(ch.min()) if len(ch) else None,
                      seat_clearance_tandem_mm=float(ct[-1]), n_rows=int(len(rows_)), rows=keep,
                      rows_note="every other checked row (every %.1f mm of d), the seat (d = 0) last" % (2 * step),
                      n_rows_checked=int(len(memo)))
        print("[approach v5] %s (%d.): offset %.1f mm, click-in over the last %.0f mm (p %.1f): min clearance to the tandem "
              "%.2f mm at d %.1f (seat %.2f)%s; %s" % (
                  s, k + 1, o, click, p, ct.min(), out[s]["d_at_min_mm"], ct[-1],
                  "" if other is None else ", to the seated %s half %.2f mm" % (other, ch.min()),
                  "PASS" if chosen else "NO schedule passes (best kept)"), flush=True)
    return dict(
        definition="half i follows pose_i(s) = tandem pose . Trans(disp_i((1 - s) D)) with disp_i(d) = d e_i + o_i(d) n_i "
                   "(tandem / applicator frame; world = F + (p_app + disp) @ R_rows); e_i = e_app (down, out of the body; "
                   "e_mode = params ring_approach_dir, from its rod direction e_rod_down_app), n_i = n_out_app (its "
                   "outward medial normal), o_i(d) = offset_mm for d >= click_mm, offset_mm "
                   "(1 - (1 - d / click_mm)^click_shape_p) below; L goes first (plan S7f: R_L then R_R).  The ring halves' "
                   "OBJs are at their seat, so T_seat = identity for them: the scene adds only disp_i",
        order=order, halves=out, clearance_min_mm=thr, half_half_min_mm=thr_hh,
        check="host (applicator_venezia.ring_approach): signed-distance grids at %.2f mm, %.1f surface samples / mm2 + "
              "mesh vertices, rows every %.1f mm of d" % (h, dens, step),
        pass_all=bool(all(v["pass_"] for v in out.values())))


def _ovoid_axis_at_z(h, z):
    """The ovoid half's axis point at height z: on the boss (between its bottom and top) or on the rod polyline."""
    b0, b1 = np.asarray(h["boss_top"], float), np.asarray(h["boss_bottom"], float)
    if z >= b1[2]:
        t = (float(z) - b1[2]) / (b0[2] - b1[2])
        return b1 + min(max(t, 0.0), 1.0) * (b0 - b1)
    pts = np.asarray(h["rod_pts"], float)
    for a, b in zip(pts[:-1], pts[1:]):
        if b[2] <= z <= a[2]:
            return a + ((float(z) - a[2]) / (b[2] - a[2])) * (b - a)
    return pts[-1]


def _tandem_axis_at_z(C, z_tube_bottom, z):
    """The tandem body's axis point at height z: the tube axis above its bottom, the vaginal tandem's centreline below."""
    if z >= z_tube_bottom:
        return np.array([0.0, 0.0, float(z)])
    C = np.asarray(C, float)
    zi = C[::-1, 2]
    return np.array([np.interp(z, zi, C[::-1, 0]), np.interp(z, zi, C[::-1, 1]), float(z)])


def validate_ring_vs_label(VF, rg, rods, rvox, lab, geo, prm, pose, frames, tandem_check, C, cl):
    """S4c for v5: the whole device (v4 tandem body + ring halves) against the updated BT applicator label U the BT ovoid
    label, at the BT pose (the label frame is the part frame).  IN-SAMPLE (the parts were measured on these labels):
    a consistency ceiling, not validation (fix plan S4).  Every distance is signed (vtkImplicitPolyDataDistance on the
    exported meshes, negative inside).
      dice             device vs applicator U ovoid on the BT grid (GATED >= 0.80, provisional); also the ring halves vs
                       the ovoid label, and both without the 12 mm collar zone above the ring (Q8)
      label_to_model   label voxel -> max(0, signed distance to the part): the whole ovoid label (ring, GATED mean <= 1),
                       its disc and socket voxels, each traced ovoid rod (GATED mean <= 1, P95 <= 2); the tandem rod and
                       tube come from the v4 check (tandem_body)
      model_to_label   surface samples of each half outside the label: disc, boss, rod along the traced sections, rod in
                       the handle bundle (the label's blob), rod beyond the label (not scored)
      connection       each rod's axis over its top 5 mm inside its half (signed distance < -2 mm)
      seat             tandem-body vertices inside a half and half vertices inside the tandem body (0 each), and the
                       smallest mesh-to-mesh distance
      axes             tandem axis to each ovoid axis per height (plan: 22 +/- 2 mm at the ring bottom, the rod line
                       extended; <= 8 mm in the bundle)
      pelvis_frame     at device_final (the G32 tube pose) the model's ovoid-rod line at the traced heights vs the label
                       sections carried BT -> preBT by frames.BONE (plan: within 3 mm in y)"""
    import evaluate as ev
    O = np.asarray(geo["frame"]["origin_BT_world"], float)
    R = np.asarray(geo["frame"]["R_rows_BT_world"], float)
    zL = R[2]
    out = dict(frame="BT label frame at the BT pose (applicator_v5 parts placed with origin_BT_world / R_rows_BT_world)",
               note="IN-SAMPLE: the parts were measured on these labels (fix plan S4: a consistency ceiling, not validation)")
    # ---- Dice on the BT grid
    U = lab["app"] | lab["ovoid"]
    Mp = {p: ev.voxelize(O + np.asarray(V, float) @ R, F, lab["shape"], lab["aff"]) for p, (V, F) in VF.items()}
    M = np.zeros(lab["shape"], bool)
    for m in Mp.values():
        M |= m
    Mr = Mp["ovoid_L"] | Mp["ovoid_R"]
    idx = np.argwhere(M | U)
    zq = (idx.astype(float) @ lab["aff"][:3, :3].T + lab["aff"][:3, 3] - O) @ zL
    no_collar = ~((zq > rg["z_top"]) & (zq <= rg["z_top"] + 12.0))
    ii = tuple(idx.T)

    def dice(a, b, sel=None):
        s = np.ones(len(idx), bool) if sel is None else sel
        aa, bb = a[ii][s], b[ii][s]
        return float(2.0 * (aa & bb).sum() / max(1, aa.sum() + bb.sum()))
    out["dice"] = dict(device_vs_app_or_ovoid=dice(M, U), device_vs_app_or_ovoid_without_collar=dice(M, U, no_collar),
                       ring_halves_vs_ovoid=dice(Mr, lab["ovoid"]), device_cc=float(M.sum() * lab["vv"] / 1000.0),
                       label_cc=float(U.sum() * lab["vv"] / 1000.0), ring_halves_cc=float(Mr.sum() * lab["vv"] / 1000.0),
                       ovoid_label_cc=float(lab["ovoid"].sum() * lab["vv"] / 1000.0),
                       note="device = tube + shaft + ovoid_L + ovoid_R voxelised at the BT pose; the collar zone = z_app in "
                            "(ring top face, + 12 mm] (Q8: air in the fornix or device?); model voxels beyond the image "
                            "field (the handle's extension) do not count")
    # ---- signed distances to the exported meshes
    sd = {p: _sdf(*VF[p]) for p in VF}
    probe_far = np.array([[60.0, 60.0, 0.0]])
    if not all(float(sd[p](probe_far)[0]) > 0 for p in VF):
        raise RuntimeError("signed-distance sign check failed (a far point reads inside)")
    for s, part, _ in RING_SIDES:
        inside_pt = np.asarray(rg["halves"][s]["rod_knee"], float)[None]
        if not float(sd[part](inside_pt)[0]) < 0:
            raise RuntimeError("signed-distance sign check failed (%s: its rod axis reads outside)" % part)

    def sd_ring(Q):
        return np.minimum(sd["ovoid_L"](Q), sd["ovoid_R"](Q))
    Qo = rvox["ovoid"]
    do = np.maximum(0.0, sd_ring(Qo))
    ds_disc = rounded_disc_sdf(np.hypot(Qo[:, 0] - rg["centre_xy"][0], Qo[:, 1] - rg["centre_xy"][1]), Qo[:, 2], rg["R_out"],
                               rg["z_top"], rg["z_bottom"], rg["fillet_top"], rg["fillet_bottom"])
    sock = ds_disc > float(val(prm, "ring_socket_out_mm"))
    l2m = dict(ring=_dist_stats(do), ring_disc_voxels=_dist_stats(do[~sock]), ring_outside_disc_voxels=_dist_stats(do[sock]),
               rod_L=_dist_stats(np.maximum(0.0, sd["ovoid_L"](rvox["rod_L"]))),
               rod_R=_dist_stats(np.maximum(0.0, sd["ovoid_R"](rvox["rod_R"]))),
               tandem_rod=tandem_check["label_to_model_mm"]["tandem_rod"], tube=tandem_check["label_to_model_mm"]["tube"],
               definition="label voxel centre -> max(0, signed distance to the part mesh): ring = every ovoid-label voxel to "
                          "ovoid_L U ovoid_R (disc voxels / voxels > ring_socket_out_mm outside the fitted disc: the "
                          "sockets and the label's hub); rod_<side> = the traced ovoid rod's own label voxels (the "
                          "tandem's selection rule) to its half; tandem_rod / tube = the v4 check (label_check.tandem_body)")
    out["label_to_model_mm"] = l2m
    # ---- model -> label (surface samples outside U)
    Vs, Fs = _mask_isosurface(U, lab["aff"])
    dist_U = _sdf((Vs - O) @ R.T, Fs)
    inv = np.linalg.inv(lab["aff"])
    Uf = U.astype(np.float32)
    from scipy import ndimage as ndi

    def m2l(Q):
        ijk = (O + Q @ R) @ inv[:3, :3].T + inv[:3, 3]
        o_ = ndi.map_coordinates(Uf, ijk.T, order=1, mode="constant", cval=0.0) < 0.5
        d = np.zeros(len(Q))
        if o_.any():
            d[o_] = np.abs(dist_U(Q[o_]))
        return d
    z_lab_end = float(geo["bundle"]["label_end_z_mm"])
    m2l_out = {}
    for k, (s, part, _) in enumerate(RING_SIDES):
        h = rg["halves"][s]
        Ps = _surface_samples(*VF[part], 2.0, 31 + k)
        d = m2l(Ps)
        d_boss = _seg_dist(Ps, h["boss_top"], h["boss_bottom"]) - rg["r_boss"]
        d_rod = _polyline_dist(Ps, h["rod_pts"]) - rg["r_rod"]
        on_boss = (np.abs(d_boss) < 0.3) & (d_boss <= d_rod)
        on_rod = (np.abs(d_rod) < 0.3) & ~on_boss
        disc = ~on_boss & ~on_rod
        z_knee = float(h["rod_knee"][2])
        sel = dict(disc=disc, boss=on_boss, rod_traced=on_rod & (Ps[:, 2] >= z_knee),
                   rod_in_bundle=on_rod & (Ps[:, 2] < z_knee) & (Ps[:, 2] >= z_lab_end))
        m2l_out[part] = {k2: dict(_dist_stats(d[m]), frac_outside_label=float((d[m] > 0).mean()) if m.any() else None)
                         for k2, m in sel.items()}
        m2l_out[part]["not_scored_beyond_label_n"] = int((on_rod & (Ps[:, 2] < z_lab_end)).sum())
    m2l_out["definition"] = ("surface point (uniform by area, 2 / mm2) of each half -> distance OUTSIDE applicator U ovoid (0 "
                             "inside by the trilinear mask >= 0.5, else the distance to its 0.5 iso-surface); parts by the "
                             "nearest design piece; rod_in_bundle = below the knee down to the label's end (the label's "
                             "handle blob is larger than the rods); rod beyond the label not scored")
    out["model_to_label_mm"] = m2l_out
    # ---- rod connection: the rod axis over its top 5 mm is inside its half
    conn = {}
    for s, part, _ in RING_SIDES:
        h = rg["halves"][s]
        a, b = np.asarray(h["rod_top"], float), np.asarray(h["rod_knee"], float)
        u = geom.unit(b - a)
        Pa = a + np.outer(np.linspace(0.0, 5.0, 11), u)
        v = sd[part](Pa)
        conn[part] = dict(max_signed_distance_mm=float(v.max()), mean_mm=float(v.mean()), pass_=bool(v.max() < -2.0),
                          rod_top_app=a, boss_top_app=h["boss_top"])
    out["rod_connection"] = dict(conn, definition="signed distance of the rod axis to its own half mesh at 11 points over "
                                                  "the rod's top 5 mm (plan: < -2 mm; G32 +6.74 mm, never inside)")
    # ---- seat: the tandem body and the halves at the seat
    VT = np.vstack([VF[p][0] for p in TANDEM_V4_PARTS])
    VR = np.vstack([VF[p][0] for p in RING_V5_PARTS])
    sdT_at_R = np.minimum(sd["tube"](VR), sd["shaft"](VR))
    sdR_at_T = sd_ring(VT)
    lh = sd["ovoid_R"](VF["ovoid_L"][0])
    rh = sd["ovoid_L"](VF["ovoid_R"][0])
    out["seat"] = dict(tandem_vertices_inside_ring=int((sdR_at_T < 0).sum()), ring_vertices_inside_tandem=int((sdT_at_R < 0).sum()),
                       min_tandem_vertex_to_ring_mm=float(sdR_at_T.min()), min_ring_vertex_to_tandem_mm=float(sdT_at_R.min()),
                       half_to_half_min_mm=float(min(lh.min(), rh.min())),
                       definition="signed distances between the exported meshes' vertices and the other part's mesh at the "
                                  "seat (G32: the shaft 2.86-2.96 mm into the caps); half_to_half = the slot")
    # ---- axis distances
    ztb = float(cl["z_tube_bottom_mm"])
    b0 = np.asarray(geo["bundle"]["stations_xyz_req"], float)
    rows = []
    zlist = [("ring bottom face (rod line extended)", rg["z_bottom"], "rod_line"),
             ("ring bottom face (boss axis)", rg["z_bottom"], "axis")]
    for s in ("L", "R"):
        zlist.append(("rod top %s" % s, float(rg["halves"][s]["rod_top"][2]), "axis"))
    z_st = sorted(set(float(z) for s in ("L", "R") for z in np.asarray(rods[s]["stations_xyz_req"], float)[:, 2]), reverse=True)
    zlist += [("z %.0f" % z, z, "axis") for z in z_st]
    zlist += [("rod knee (enters the handle bundle)", float(rg["halves"]["L"]["rod_knee"][2]), "axis"),
              ("first bundle section", float(b0[0, 2]), "axis"), ("label end", z_lab_end, "axis")]
    for name, z, how in zlist:
        t = _tandem_axis_at_z(C, ztb, z)
        r_ = dict(where=name, z_mm=float(z))
        for s in ("L", "R"):
            if how == "rod_line":
                p = _line_z(rods[s]["line_point"], rods[s]["dir_app_down"], z)
            else:
                p = _ovoid_axis_at_z(rg["halves"][s], z)
            r_[s] = float(np.linalg.norm(p[:2] - t[:2]))
        r_["L_R"] = float(np.linalg.norm(_ovoid_axis_at_z(rg["halves"]["L"], z)[:2] - _ovoid_axis_at_z(rg["halves"]["R"], z)[:2]))
        rows.append(r_)
    out["axis_distances_mm"] = dict(rows=rows, definition="in-plane (z_app = const) distance between the tandem body's axis "
                                                          "(tube axis, then the vaginal tandem's centreline) and each ovoid "
                                                          "axis (boss, then rod); L_R = between the two ovoid axes")
    # ---- pelvis frame at device_final
    df = pose["device_final"]
    F_fin, R_fin = np.asarray(df["flange"], float), np.asarray(df["R_rows"], float)
    Rb, tb = np.asarray(frames["BONE"]["R"], float), np.asarray(frames["BONE"]["t"], float)
    pel = {}
    for s in ("L", "R"):
        S = np.asarray(rods[s]["stations_xyz_req"], float)[:, :3]
        mdl = np.array([_line_z(rods[s]["line_point"], rods[s]["dir_app_down"], z) for z in S[:, 2]])
        d = (F_fin + mdl @ R_fin) - ((O + S @ R) @ Rb.T + tb)
        n = np.linalg.norm(d, axis=1)
        pel["rod_" + s] = dict(n=int(len(n)), mean_mm=float(n.mean()), max_mm=float(n.max()), mean_dx_mm=float(d[:, 0].mean()),
                               mean_dy_mm=float(d[:, 1].mean()), mean_dz_mm=float(d[:, 2].mean()),
                               max_abs_dy_mm=float(np.abs(d[:, 1]).max()))
    cr = np.array([rg["centre_xy"][0], rg["centre_xy"][1], rg["z_mid"]])
    d = (F_fin + cr @ R_fin) - ((O + cr @ R) @ Rb.T + tb)
    pel["ring_centre"] = dict(d_mm=d, norm_mm=float(np.linalg.norm(d)))
    pel["definition"] = ("model = the rod line at the traced heights placed at device_final (F + p R_rows, the G32 tube "
                         "pose); label = the raw traced sections at the BT label pose carried BT -> preBT by frames.BONE; "
                         "d = model - label (preBT RAS: +x right, +y anterior, +z superior).  As for the vaginal tandem, "
                         "d = the flange offset + the 2.4 deg axis error x the lever arm (label_check.tandem_body."
                         "pelvis_frame.pose_decomposition)")
    out["pelvis_frame"] = pel
    return out


def ring_acceptance(check, tandem_check, ap, stats, rg, carry):
    """The S4c acceptance list of v5 (fix plan S4 'Accept' + the task's gates).  Each item: test, measured, pass."""
    lm, tl = check["label_to_model_mm"], tandem_check["label_to_model_mm"]
    ax = {r["where"]: r for r in check["axis_distances_mm"]["rows"]}
    rb = ax["ring bottom face (rod line extended)"]
    bun = ax["first bundle section"]
    last_key = "z %.0f" % rg["z_last_separable_mm"]
    pose_t = [a for a in tandem_check["acceptance"] if a["test"].startswith("tube pose")][0]
    pel_t = [a for a in tandem_check["acceptance"] if a["test"].startswith("pelvis frame")][0]
    acc = [
        dict(test="Dice(device, applicator U ovoid label) >= 0.80 at the BT pose (in-sample geometry fit, provisional floor)",
             measured=check["dice"]["device_vs_app_or_ovoid"], without_collar=check["dice"]["device_vs_app_or_ovoid_without_collar"],
             ring_vs_ovoid=check["dice"]["ring_halves_vs_ovoid"], pass_=check["dice"]["device_vs_app_or_ovoid"] >= 0.80),
        dict(test="tandem rod: label-to-model mean <= 1.0 mm and P95 <= 2.0 mm", measured=[tl["tandem_rod"]["mean_mm"], tl["tandem_rod"]["p95_mm"]],
             pass_=tl["tandem_rod"]["mean_mm"] <= 1.0 and tl["tandem_rod"]["p95_mm"] <= 2.0),
        dict(test="OR (right ovoid rod): label-to-model mean <= 1.0 mm and P95 <= 2.0 mm", measured=[lm["rod_R"]["mean_mm"], lm["rod_R"]["p95_mm"]],
             pass_=lm["rod_R"]["mean_mm"] <= 1.0 and lm["rod_R"]["p95_mm"] <= 2.0),
        dict(test="OL (left ovoid rod): label-to-model mean <= 1.0 mm and P95 <= 2.0 mm", measured=[lm["rod_L"]["mean_mm"], lm["rod_L"]["p95_mm"]],
             pass_=lm["rod_L"]["mean_mm"] <= 1.0 and lm["rod_L"]["p95_mm"] <= 2.0),
        dict(test="ring (every ovoid-label voxel): label-to-model mean <= 1.0 mm", measured=lm["ring"]["mean_mm"],
             p95_mm=lm["ring"]["p95_mm"], pass_=lm["ring"]["mean_mm"] <= 1.0),
        dict(test="tube (a3/a14 split): label-to-model mean <= 1.5 mm (P95 reported)", measured=tl["tube"]["mean_mm"],
             p95_mm=tl["tube"]["p95_mm"], pass_=tl["tube"]["mean_mm"] <= 1.5),
        dict(test="each half ONE closed, oriented component (half-disc + boss + rod)",
             measured={p: [stats[p]["closed_oriented"], stats[p]["n_components"]] for p in RING_V5_PARTS},
             pass_=all(stats[p]["closed_oriented"] and stats[p]["one_component"] for p in RING_V5_PARTS)),
        dict(test="rod connection: each rod axis inside its half by more than 2 mm over its top 5 mm",
             measured={p: check["rod_connection"][p]["max_signed_distance_mm"] for p in RING_V5_PARTS},
             pass_=all(check["rod_connection"][p]["pass_"] for p in RING_V5_PARTS)),
        dict(test="seat: 0 tandem-body vertices inside the ovoid body (and 0 ring vertices inside the tandem body)",
             measured=[check["seat"]["tandem_vertices_inside_ring"], check["seat"]["ring_vertices_inside_tandem"]],
             min_distance_mm=[check["seat"]["min_tandem_vertex_to_ring_mm"], check["seat"]["min_ring_vertex_to_tandem_mm"]],
             pass_=check["seat"]["tandem_vertices_inside_ring"] == 0 and check["seat"]["ring_vertices_inside_tandem"] == 0),
        dict(test="axis distance tandem to ovoid rod at the ring bottom 22 +/- 2 mm (rod line extended to the bottom face)",
             measured=[rb["L"], rb["R"]], pass_=all(abs(rb[s] - 22.0) <= 2.0 for s in ("L", "R"))),
        dict(test="axis distance tandem to ovoid rod <= 8 mm in the bundle (first bundle section)",
             measured=[bun["L"], bun["R"]], label_last_separable=[ax[last_key]["L"], ax[last_key]["R"]],
             note="MEASURED on the label: at its last separable section (z %.0f) the ovoid rods are %.1f / %.1f mm from the "
                  "tandem; below that the label is one blob (the handle / clamp, wider than three rods), so the model rods "
                  "run on parallel to the bundle axis from there" % (rg["z_last_separable_mm"], ax[last_key]["L"], ax[last_key]["R"]),
             pass_=all(bun[s] <= 8.0 for s in ("L", "R"))),
        dict(test="pelvis frame, G32 tube pose: ovoid rods within 3 mm in y of the mapped label rods (mean dy)",
             measured=[check["pelvis_frame"]["rod_L"]["mean_dy_mm"], check["pelvis_frame"]["rod_R"]["mean_dy_mm"]],
             max_abs_dy_mm=[check["pelvis_frame"]["rod_L"]["max_abs_dy_mm"], check["pelvis_frame"]["rod_R"]["max_abs_dy_mm"]],
             pass_=all(abs(check["pelvis_frame"]["rod_" + s]["mean_dy_mm"]) <= 3.0 for s in ("L", "R"))),
        dict(test=pel_t["test"] + " (tandem body = v4's)", measured=pel_t["measured"], pass_=pel_t["pass"]),
        dict(test=pose_t["test"] + " (tandem body = v4's)", measured=pose_t["measured"], pass_=pose_t["pass"]),
        dict(test="approach: each half offset >= 5 mm, clearance to the tandem body >= 0.5 mm on every row, the second "
                  "half >= slot - 0.1 mm to the first (host)",
             measured={s: [h["offset_mm"], h["min_clearance_tandem_mm"], h["min_clearance_other_half_mm"]] for s, h in ap["halves"].items()},
             pass_=ap["pass_all"] and all(h["offset_mm"] >= 5.0 for h in ap["halves"].values())),
        dict(test="tandem-first record carried from the tandem variant: tandem body identical, rows identical",
             measured=carry, pass_=bool(carry.get("pass"))),
    ]
    for a_ in acc:
        a_["pass"] = bool(a_.pop("pass_"))
    return acc


def _mesh_cut(V, F, p0, u, v):
    """Segments (k, 2, 2) of a triangle mesh cut by the plane through p0 spanned by the unit vectors u, v, in (u, v)
    coordinates."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    n = np.cross(u, v)
    s = (V - p0) @ n
    sF = s[F]
    X = F[(sF.min(1) < 0.0) & (sF.max(1) > 0.0)]
    if not len(X):
        return np.zeros((0, 2, 2))
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        si, sj = s[X[:, i]], s[X[:, j]]
        m = (si * sj) < 0.0
        t = np.where(m, si / np.where(m, si - sj, 1.0), 0.0)
        P_ = V[X[:, i]] + t[:, None] * (V[X[:, j]] - V[X[:, i]])
        pts.append((m, P_))
    seg = []
    for k in range(len(X)):
        q = [P_[k] for m, P_ in pts if m[k]]
        if len(q) == 2:
            seg.append([[(q[0] - p0) @ u, (q[0] - p0) @ v], [(q[1] - p0) @ u, (q[1] - p0) @ v]])
    return np.array(seg)


def fig_v5_label_check(geo, rg, rods, lab, VF, check, acc, fn):
    """The v5 device against the updated BT applicator label and the BT ovoid label (BT label frame = part frame, the BT
    pose): the label resampled on six planes (ovoid label orange, applicator minus ovoid blue) with the exported meshes'
    cut outlines (tandem black, left half green, right half magenta)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy import ndimage as ndi
    O = np.asarray(geo["frame"]["origin_BT_world"], float)
    R = np.asarray(geo["frame"]["R_rows_BT_world"], float)
    inv = np.linalg.inv(lab["aff"])
    apf, ovf = lab["app"].astype(np.float32), lab["ovoid"].astype(np.float32)
    xr, yr = np.asarray(rg["x_r"], float), np.asarray(rg["y_r"], float)
    xo = {s: float(np.median(np.asarray(rods[s]["stations_xyz_req"], float)[:, :2] @ xr[:2])) for s in ("L", "R")}
    z_rod = float(np.median(np.asarray(rods["L"]["stations_xyz_req"], float)[:, 2]))
    zm = float(rg["z_mid"])
    panels = [
        ("device sagittal plane, 1 mm right of the slot (x' 1.0 mm)", 1.0 * xr, yr, EZ, (-30, 55), (-95, 68)),
        ("parallel plane through the LEFT ovoid rod (x' %.1f mm)" % xo["L"], xo["L"] * xr, yr, EZ, (-30, 55), (-95, 12)),
        ("parallel plane through the RIGHT ovoid rod (x' %.1f mm)" % xo["R"], xo["R"] * xr, yr, EZ, (-30, 55), (-95, 12)),
        ("device coronal plane (through the tube axis, across the slot)", np.zeros(3), xr, EZ, (-30, 30), (-40, 30)),
        ("axial plane at the ring's mid-thickness (z_app %.1f)" % zm, np.array([0.0, 0.0, zm]), EX, EY, (-28, 28), (-28, 32)),
        ("axial plane through the rods (z_app %.0f)" % z_rod, np.array([0.0, 0.0, z_rod]), EX, EY, (-16, 20), (-8, 40)),
    ]
    col = dict(tube="k", shaft="k", ovoid_L="tab:green", ovoid_R="m")
    fig, axs = plt.subplots(2, 3, figsize=(19, 15))
    for ax, (title, p0, u, v, ur, vr) in zip(axs.ravel(), panels):
        h = 0.25
        A_, B_ = np.meshgrid(np.arange(ur[0], ur[1] + 1e-9, h), np.arange(vr[0], vr[1] + 1e-9, h), indexing="ij")
        Q = p0 + A_.ravel()[:, None] * u + B_.ravel()[:, None] * v
        ijk = ((O + Q @ R) @ inv[:3, :3].T + inv[:3, 3]).T
        a_ = (ndi.map_coordinates(apf, ijk, order=1, mode="constant", cval=0.0) >= 0.5).reshape(A_.shape)
        o_ = (ndi.map_coordinates(ovf, ijk, order=1, mode="constant", cval=0.0) >= 0.5).reshape(A_.shape)
        rgb = np.ones(A_.shape + (3,))
        rgb[a_ & ~o_] = [0.55, 0.75, 1.0]
        rgb[o_] = [1.0, 0.78, 0.45]
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=(ur[0], ur[1], vr[0], vr[1]), interpolation="nearest")
        for p, (V, F) in VF.items():
            seg = _mesh_cut(V, F, p0, u, v)
            if len(seg):
                ax.add_collection(LineCollection(seg, colors=col[p], linewidths=1.3))
        ax.set_xlim(*ur)
        ax.set_ylim(*vr)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    for ax, xl in zip(axs.ravel(), ("y' (mm, anterior ->)", "y' (mm)", "y' (mm)", "x' (mm, patient right ->)",
                                     "x_app (mm, right ->)", "x_app (mm, right ->)")):
        ax.set_xlabel(xl)
    for ax, yl in zip(axs.ravel(), ("z_app (mm, 0 = flange)",) * 4 + ("y_app (mm, anterior ->)",) * 2):
        ax.set_ylabel(yl)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axs[0, 0].legend(handles=[Patch(color=[1.0, 0.78, 0.45], label="BT ovoid label"),
                              Patch(color=[0.55, 0.75, 1.0], label="updated applicator label minus ovoid"),
                              Line2D([], [], color="k", label="v5 tandem body (= v4)"),
                              Line2D([], [], color="tab:green", label="v5 left half (disc + boss + rod)"),
                              Line2D([], [], color="m", label="v5 right half")], loc="lower left", fontsize=8)
    lm, d = check["label_to_model_mm"], check["dice"]
    npass = sum(a_["pass"] for a_ in acc)
    fail = [a_["test"].split(":")[0][:48] for a_ in acc if not a_["pass"]]
    fig.suptitle("applicator v5 vs the BT labels (BT label frame, IN-SAMPLE fit; x' / y' = the slot frame, azimuth %.1f deg)\n"
                 "ring outer %.1f mm (label r_eq %.2f, MRI edge %.2f), faces z %.1f / %.1f, fillets %.1f / %.1f mm;  Dice device "
                 "vs applicator U ovoid %.3f (ring vs ovoid %.3f)\nlabel-to-model mean / P95: ring %.2f / %.2f, OL %.2f / %.2f, "
                 "OR %.2f / %.2f, tandem rod %.2f / %.2f mm;  acceptance %d / %d%s" % (
                     rg["slot_azimuth_deg"], 2.0 * rg["R_out"], rg["outer_radius"]["label_req_plateau_mm"]["median"],
                     (rg["outer_radius"]["mri_half_way_edge_mm"] or {}).get("median", float("nan")), rg["z_top"], rg["z_bottom"],
                     rg["fillet_top"], rg["fillet_bottom"], d["device_vs_app_or_ovoid"], d["ring_halves_vs_ovoid"],
                     lm["ring"]["mean_mm"], lm["ring"]["p95_mm"], lm["rod_L"]["mean_mm"], lm["rod_L"]["p95_mm"],
                     lm["rod_R"]["mean_mm"], lm["rod_R"]["p95_mm"], lm["tandem_rod"]["mean_mm"], lm["tandem_rod"]["p95_mm"],
                     npass, len(acc), (" (fail: %s)" % "; ".join(fail)) if fail else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    fig.savefig(fn, dpi=80)
    plt.close(fig)
    print("[fig] wrote", fn, flush=True)


def _vendor_vs_label(prm, geo, rg, cl):
    """The physician's vendor sizes next to what the label measures, and how they correspond."""
    z_tip = float(geo["tube"]["z_tip_label_mm"])
    ring0 = geo["ring"]
    zt, zb, zm = rg["z_top"], rg["z_bottom"], rg["z_mid"]
    L_v = float(val(prm, "tandem_length_mm"))
    z_ref = z_tip - L_v
    tab = np.asarray(rg["profile_fit"]["radial_median_table"], float)
    ok = np.isfinite(tab[:, 1]) & np.isfinite(tab[:, 2])
    ch = float(val(prm, "ring_channel_diam_mm"))
    orad = rg["outer_radius"]
    mri = orad["mri_half_way_edge_mm"] or {}
    return dict(
        tandem_length=dict(
            vendor_mm=L_v, model_L_iu_mm=float(val(prm, "L_iu_mm")), label_tip_z_app_mm=z_tip,
            label_tip_to_mm=dict(flange=z_tip, ring_top_on_axis=z_tip - float(ring0["z_top_on_axis_mm"]), ring_top_face=z_tip - zt,
                                 ring_mid_thickness=z_tip - zm, ring_bottom_face=z_tip - zb,
                                 tube_bottom=z_tip - float(cl["z_tube_bottom_mm"])),
            vendor_reference_z_app_mm=z_ref, vendor_reference_below_top_face_mm=zt - z_ref,
            vendor_reference_above_mid_thickness_mm=z_ref - zm,
            note="The label's tube tip is %.1f mm above the flange (the model keeps its validated L_iu %.1f mm); %.0f mm below "
                 "the tip lands %.1f mm below the ring's top face, %.1f mm above its mid-thickness, i.e. INSIDE the ring.  "
                 "The vendor's intrauterine length is therefore not measured from the ring's cervical surface the label "
                 "shows: it corresponds to a reference inside the ovoids (the ring / channel plane), and the label's tip, a "
                 "signal void on 1.6 mm slices, may also end short of the physical tip (the physician: the length 'might be "
                 "larger depending on which part you measure')" % (z_tip, float(val(prm, "L_iu_mm")), L_v, zt - z_ref,
                                                                    z_ref - zm)),
        ring_diameter=dict(
            vendor_channel_diam_mm=ch, channel_radius_mm=0.5 * ch, model_outer_diam_mm=2.0 * rg["R_out"],
            label_outer_diam_req_mm=2.0 * orad["label_req_plateau_mm"]["median"],
            label_outer_diam_outline_mm=2.0 * orad["label_outline_circle_mm"]["median"],
            label_outer_diam_column_half_mm=None if orad["label_column_presence_half_mm"] is None else 2.0 * orad["label_column_presence_half_mm"],
            mri_edge_diam_mm=None if not mri else 2.0 * mri["median"],
            channel_to_outer_surface_mm=rg["R_out"] - 0.5 * ch,
            thickness_faces_mm=zt - zb, label_thickness_max_mm=float((tab[ok, 1] - tab[ok, 2]).max()),
            note="30 mm is the diameter of the circle of SOURCE CHANNELS (vendor: 22 / 26 / 30 mm sets); the label's ovoids "
                 "measure %.1f mm across (area-equivalent, %.1f at the MRI edge), so the channel circle runs %.1f mm inside the "
                 "outer surface of the %.0f mm model ring: 'the ovoids diameter in the spec is likely somewhere inside the "
                 "ovoids, not the outer diameter' (the physician).  The model uses the OUTER size (U1 = label) and records "
                 "the channel size only" % (2.0 * orad["label_req_plateau_mm"]["median"], 2.0 * mri.get("median", float("nan")),
                                            rg["R_out"] - 0.5 * ch, 2.0 * rg["R_out"])))


def main_ring(a):
    """v5 (fix plan S4b + S4c): the v4 tandem body (copied byte for byte, its label geometry re-measured and checked)
    PLUS the two lunar ring halves measured on the updated BT applicator label and the BT ovoid label.  Writes
    applicator_<variant>/{tube.obj, shaft.obj, ovoid_L.obj, ovoid_R.obj, applicator.json, pose.json},
    logs/applicator_venezia_<variant>.json and figs/applicator_<variant>_label_check.png.  pose.json = the tandem
    variant's (same device_final, corpus, insertion paths) plus the ring's seat and approach in the world at
    device_final; its insertion_path_tandem_first is carried verbatim (this variant's own record on a rebuild, else the
    tandem variant's), checked against the body it was gated against (_tf_record_problems), and its rows are
    regenerated and compared.  Everything is built and checked in memory first; pose.json is replaced atomically."""
    import copy
    import shutil
    t0 = time.time()
    if not a.variant:
        raise SystemExit("--ring builds a NEW applicator variant: pass --variant (e.g. v5)")
    src = HY + "/applicator_" + a.tandem_from
    if os.path.abspath(src) == os.path.abspath(APP):
        raise SystemExit("--tandem-from must name another variant than --variant")
    old = _load_json_or_empty(APP + "/applicator.json")
    if old and not old.get("params", {}).get("ring_body", {}).get("value", False):
        raise SystemExit("%s holds an applicator that is not a ring variant (params.ring_body): refusing to overwrite it "
                         "(pick a new --variant)" % APP)
    if os.path.isdir(APP):
        stray = sorted(f for f in os.listdir(APP) if f.endswith(".obj")
                       and os.path.splitext(f)[0] not in TANDEM_V4_PARTS + RING_V5_PARTS)
        if stray:
            raise SystemExit("%s already holds other parts %s: a ring variant must not" % (APP, stray))
    old_pose = {}
    if os.path.exists(APP + "/pose.json"):
        old_pose = json.load(open(APP + "/pose.json"))
    deps_old, why_old = _depends_on_disk(APP) if old_pose else (None, "no pose.json")
    src_app, src_pose = json.load(open(src + "/applicator.json")), json.load(open(src + "/pose.json"))
    if not src_app.get("params", {}).get("tandem_only", {}).get("value", False):
        raise SystemExit("--tandem-from %s is not a tandem-only variant (params.tandem_only)" % src)
    deps_src, why_src = _depends_on_disk(src)
    if deps_src is None:
        raise SystemExit("the tandem variant %s is incomplete: %s" % (src, why_src))
    for d in (LOGS, FIGS):
        os.makedirs(d, exist_ok=True)
    VF = {p: geom.read_obj(os.path.join(src, p + ".obj")) for p in TANDEM_V4_PARTS}
    prm = copy.deepcopy(src_app["params"])
    added = v5_params(prm, a.ring_outer_r)
    if a.ring_approach_dir:
        prm["ring_approach_dir"]["value"] = a.ring_approach_dir
    app_in = json.load(open(P["inputs"] + "/applicator.json"))
    lab = load_label_new(label_applicator_file(a.app_label))
    geo, vox = measure_label_geometry(lab, app_in, prm)
    g4 = src_app["label_geometry"]
    same_geo = dict(frame_origin=_max_abs_diff(geo["frame"]["origin_BT_world"], g4["frame"]["origin_BT_world"]),
                    frame_R=_max_abs_diff(geo["frame"]["R_rows_BT_world"], g4["frame"]["R_rows_BT_world"]),
                    tandem_stations=_max_abs_diff(geo["tandem"]["stations_xyz_req"], g4["tandem"]["stations_xyz_req"]),
                    bundle_lobe=_max_abs_diff(geo["bundle"]["tandem_lobe_xyz"], g4["bundle"]["tandem_lobe_xyz"]),
                    bundle_axis=_max_abs_diff(geo["bundle"]["axis_app_down"], g4["bundle"]["axis_app_down"]))
    if max(same_geo.values()) > 1e-4:
        raise SystemExit("the label geometry re-measured now differs from applicator_%s's by %s: not the same label / code "
                         "(rebuild the tandem variant first)" % (a.tandem_from, jz(same_geo, 6)))
    C = np.asarray(src_app["landmarks"]["shaft_centreline"], float)
    cl = src_app["landmarks"]["shaft_centreline_info"]
    # ---- the ring halves
    rg, rods, rvox = measure_ring_geometry(lab, geo, prm, C, cl)
    ring_stats, VFr, grids = build_ring_meshes(rg, prm)
    VFall = dict(VF)
    VFall.update(VFr)
    ap = ring_approach(rg, VFall, grids, C, prm, float(cl["z_tube_bottom_mm"]))
    frames = load_frames()
    df = src_pose["device_final"]
    rule = dict(flange=np.asarray(df["flange"], float), tube_axis=np.asarray(df["tube_axis"], float),
                x_app=np.asarray(df["x_app"], float), R_rows=np.asarray(df["R_rows"], float))
    tandem_check = validate_device_vs_label(VF, geo, vox, lab, prm, C, cl, rule, frames, app_in, src_pose, a.tandem_from)
    check = validate_ring_vs_label(VFall, rg, rods, rvox, lab, geo, prm, src_pose, frames, tandem_check, C, cl)
    # ---- T_seat, landmarks, the vendor sizes
    c_ring = np.array([rg["centre_xy"][0], rg["centre_xy"][1], rg["z_mid"]])
    R_seat = np.stack([np.asarray(rg["x_r"], float), np.asarray(rg["y_r"], float), EZ])
    T_seat = np.eye(4)
    T_seat[:3, :3], T_seat[:3, 3] = R_seat.T, c_ring
    k_c = 4.0 * rg["R_out"] / (3.0 * np.pi)
    cap_c = [c_ring + sg * k_c * np.asarray(rg["x_r"], float) for _, _, sg in RING_SIDES]
    lm_ = copy.deepcopy(src_app["landmarks"])
    lm_.update(cap_centres=cap_c, cap_centres_note="the half-disc centroids of the two ring halves (L, R): ring centre +/- "
                                                   "4 R / (3 pi) along x_r at the mid-thickness (reference points only)",
               ring_centre=c_ring, ring_top_face_z=rg["z_top"], ring_bottom_face_z=rg["z_bottom"],
               ring_axis=EZ, ring_slot_dir=rg["y_r"], ring_outer_r=rg["R_out"],
               ovoid_rods={s: dict(top=rg["halves"][s]["rod_top"], knee=rg["halves"][s]["rod_knee"], end=rg["halves"][s]["rod_end"],
                                   boss_top=rg["halves"][s]["boss_top"], dir_down=rg["halves"][s]["rod_dir_down"])
                           for s in ("L", "R")})
    vendor = _vendor_vs_label(prm, geo, rg, cl)
    # ---- pose.json: the tandem variant's, plus the ring at device_final; the tandem-first record carried
    R_fin, F_fin = np.asarray(df["R_rows"], float), np.asarray(df["flange"], float)
    pose_out = copy.deepcopy(src_pose)
    for k in BODY_PATH_RECORDS + ("stale_records",):
        pose_out.pop(k, None)
    pose_out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pose_out["device_final"]["cap_centres_world"] = jz([F_fin + c @ R_fin for c in cap_c], 4)
    pose_out["device_final"]["v5_note"] = ("tube pose, shaft end and corpus = applicator_%s's (copied: the tandem body is its "
                                           "own); cap_centres_world = the ring halves' half-disc centroids at the seat" % a.tandem_from)
    pose_out["v5"] = jz(dict(
        tandem_from=a.tandem_from,
        ring_seat_world=dict(origin=F_fin + c_ring @ R_fin, R_rows=R_seat @ R_fin,
                             convention="p_world = origin + p_ring @ R_rows (the ring frame at the seat, device_final)"),
        halves={s: dict(part=h["part"], approach_dir_world=np.asarray(h["e_app"], float) @ R_fin,
                        n_out_world=np.asarray(h["n_out_app"], float) @ R_fin, offset_mm=h["offset_mm"], click_mm=h["click_mm"],
                        click_shape_p=h["click_shape_p"], D_mm=h["D_mm"], order=h["order"])
                for s, h in ap["halves"].items()},
        note="the ring halves ride on the tandem: at any tandem pose (F, R_rows) half i sits at F + (p_app + disp_i(d)) @ "
             "R_rows (applicator.json ring.approach); the world vectors here are at device_final only"))
    pose_json = pose_out
    app_params_json = jz(prm)
    deps_new = tandem_body_depends(app_params_json, pose_json, VF)
    keep, stale, msgs = {}, dict(old_pose.get("stale_records") or {}), []
    for k in BODY_PATH_RECORDS:
        cands = []
        if k in old_pose:
            cands.append(("applicator_%s's own (before this rebuild)" % a.variant, old_pose[k], deps_old, why_old))
        if k in src_pose:
            cands.append(("applicator_%s's (the tandem variant)" % a.tandem_from, src_pose[k], deps_src, why_src))
        chosen = None
        for name, rec, dref, why in cands:
            probs = _tf_record_problems(rec, pose_json, deps_new, dref, why)
            if not probs:
                chosen = (name, rec)
                break
            msgs.append("%s %s does not match: %s" % (k, name, "; ".join(probs)))
        if chosen:
            keep[k] = chosen[1]
            msgs.append("%s carried verbatim from %s (written %s)" % (k, chosen[0], chosen[1].get("written")))
        elif cands:
            if not getattr(a, "force", False):
                raise SystemExit("refusing to build: no %s record matches this variant's tandem body:\n  %s\nNothing was "
                                 "written (--force moves it to pose.json stale_records)" % (k, "\n  ".join(msgs)))
            stale[k] = dict(record=cands[0][1], moved=time.strftime("%Y-%m-%d %H:%M:%S"), why=msgs,
                            note="moved here by a --ring --force build: the scene does not read it")
    pose_json.update(keep)
    if stale:
        pose_json["stale_records"] = stale
    carry = dict(pass_=False)
    k = "insertion_path_tandem_first"
    if k in keep and k in src_pose:
        rows_new = geom.tandem_first_path(keep[k]["params"])
        ref = src_pose[k]["rows"]
        dF = max(float(np.abs(np.asarray(q["F"], float) - r_["F"]).max()) for q, r_ in zip(ref, rows_new))
        da = max(float(np.abs(np.asarray(q["tube_axis"], float) - r_["tube_axis"]).max()) for q, r_ in zip(ref, rows_new))
        carry = dict(record_identical_to_tandem_variant=bool(json.dumps(keep[k], sort_keys=True) == json.dumps(src_pose[k], sort_keys=True)),
                     n_rows=[len(ref), len(rows_new)], regenerated_rows_max_abs_diff_F_mm=dF,
                     regenerated_rows_max_abs_diff_axis=da, tandem_body_diff=_depends_diff(deps_src, deps_new),
                     source=[m_ for m_ in msgs if "carried" in m_])
        carry["pass_"] = bool(carry["record_identical_to_tandem_variant"] and len(ref) == len(rows_new) and dF <= 1e-6
                              and da <= 1e-9 and not carry["tandem_body_diff"])
    elif k not in src_pose:
        carry = dict(pass_=True, note="the tandem variant holds no %s record" % k)
    carry["pass"] = bool(carry.pop("pass_"))
    for m_ in msgs:
        print("[pose v5] %s" % m_, flush=True)
    acc = ring_acceptance(check, tandem_check, ap, ring_stats, rg, carry)
    for a_ in acc:
        print("[S4c v5] %-4s %s: %s" % ("PASS" if a_["pass"] else "FAIL", a_["test"], jz(a_["measured"], 3)), flush=True)
    label_check = dict(check, tandem_body=tandem_check, acceptance=acc, pass_all=all(a_["pass"] for a_ in acc),
                       note="IN-SAMPLE (fix plan S4c): the parts were measured on these labels; a consistency ceiling, not validation")
    # ---- applicator.json
    parts = copy.deepcopy(src_app["parts"])
    parts.update(ring_stats)
    rg_json = {kk: vv for kk, vv in rg.items()}
    app_out = copy.deepcopy(src_app)
    app_out.update(
        frame=FRAME_APP_V5, variant=a.variant,
        device="Venezia-type applicator v5 (fix plan S4b): the v4 tandem body (tube + vaginal tandem, ONE rigid body; the "
               "scene loads tube.obj + shaft.obj as the tandem) PLUS two lunar ring halves (ovoid_L.obj, ovoid_R.obj: the "
               "scene's ovoid body), each ONE closed manifold = half-disc + socket boss + its own rod into the handle "
               "bundle; ring OUTER diameter %.0f mm (U1 = label), all shapes measured on the updated BT applicator label and "
               "the BT ovoid label.  Identity unconfirmed: RTPLAN / vendor geometry not available" % (2.0 * rg["R_out"]),
        written=time.strftime("%Y-%m-%d %H:%M:%S"), params=prm, added_params=added,
        added_params_note="ring-half keys added to applicator_%s's params (every other key is its, unchanged)" % a.tandem_from,
        parts=parts, tandem_parts=list(TANDEM_V4_PARTS), ring_parts=list(RING_V5_PARTS), landmarks=lm_,
        label_geometry=dict(geo, ring_halves=rg_json, ovoid_rods=rods), label_check=label_check,
        ring=dict(T_seat=dict(origin_app=c_ring, R_rows_app=R_seat, matrix_app_from_ring=T_seat,
                              convention="p_app = origin_app + p_ring @ R_rows_app; rows = x_r (the right half's outward medial "
                                         "normal), y_r (anterior along the slot), z (the ring axis = the tube axis).  The OBJs "
                                         "are already at this seat in the applicator frame",
                              definition="origin = the ring centre (label plateau median) at the mid-thickness between the "
                                         "fitted faces; the ring is perpendicular to the tube (device design; the label's "
                                         "face tilts are within its voxel size, label_geometry.ring_halves.profile_fit)"),
                  halves={s: dict(part=h["part"], n_out_app=h["n_out"], rod_dir_down_app=h["rod_dir_down"],
                                  approach_dir_down_app=h["approach_dir_down"], rod_pts_app=h["rod_pts"],
                                  boss=[h["boss_top"], h["boss_bottom"]]) for s, h in rg["halves"].items()},
                  approach=ap),
        vendor_vs_label=vendor,
        tandem_from=dict(variant=a.tandem_from, applicator_json=src + "/applicator.json", pose_json=src + "/pose.json",
                         mesh_sha256=deps_src["mesh_sha256"], label_geometry_max_abs_diff=same_geo,
                         note="tube.obj / shaft.obj copied byte for byte; the label geometry they were built from re-measured "
                              "identical"))
    app_json = jz(app_out)
    # ---- write (every check above ran): meshes, applicator.json, pose.json (atomic)
    os.makedirs(APP, exist_ok=True)
    for p in TANDEM_V4_PARTS:
        shutil.copyfile(os.path.join(src, p + ".obj"), os.path.join(APP, p + ".obj"))
    for p in RING_V5_PARTS:
        V, F = VFr[p]
        geom.write_obj(os.path.join(APP, p + ".obj"), V, F,
                       header="%s: one lunar ring half of applicator v5 (half-disc + socket boss + rod, ONE closed manifold) at "
                              "its seat, built from the updated BT applicator label and the BT ovoid label; %s" % (p, FRAME_APP_V5))
    json.dump(app_json, open(APP + "/applicator.json", "w"), indent=1)
    _dump_json_atomic(pose_json, APP + "/pose.json")
    deps_disk, why_disk = _depends_on_disk(APP)
    if deps_disk is None or _depends_diff(deps_src, deps_disk):
        raise SystemExit("written %s: its tandem body differs from applicator_%s's (%s)" % (APP, a.tandem_from,
                                                                                            why_disk or _depends_diff(deps_src, deps_disk)))
    print("[pose v5] wrote %s/{applicator.json, pose.json, tube.obj, shaft.obj, ovoid_L.obj, ovoid_R.obj}; kept %s" % (
        APP, sorted(keep) or "no record"), flush=True)
    if not a.no_figs:
        fig_v5_label_check(geo, rg, rods, lab, VFall, check, acc, os.path.join(FIGS, "applicator_%s_label_check.png" % a.variant))
    files = sorted(os.listdir(APP))
    log = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(a), wall_s=round(time.time() - t0, 1), files=files,
               acceptance=acc, label_check={kk: vv for kk, vv in label_check.items() if kk != "tandem_body"},
               ring={kk: vv for kk, vv in rg.items() if kk not in ("sections", "profile_fit")}, approach=ap, vendor_vs_label=vendor,
               mesh=ring_stats, carry=carry)
    json.dump(jz(log), open(LOGS + "/applicator_venezia_%s.json" % a.variant, "w"), indent=1)
    print("[done v5] %s: files %s; S4c pass %d / %d; wall %.1f s" % (APP, files, sum(a_["pass"] for a_ in acc), len(acc),
                                                                     time.time() - t0), flush=True)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true", help="reuse the ovoid fit stored in applicator.json")
    ap.add_argument("--no-meshes", action="store_true"); ap.add_argument("--no-figs", action="store_true"); ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--angle", type=float, default=None); ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--path-axis", choices=["shaft", "tube"], default=None)
    ap.add_argument("--rule", choices=["v1", "v2"], default="v2", help="v2 (default): vaginal principal axis + seating shift Delta; v1: rejected reference")
    ap.add_argument("--flange-shift", type=float, default=None, help="rule v2 Delta in mm (default: in-sample best of the sweep)")
    ap.add_argument("--shaft-line-through", choices=["O_pre", "vagina_axis"], default=None, help="rule v2 line placement (default: best variant)")
    ap.add_argument("--shift-axis", choices=["shaft", "tube"], default=None, help="rule v2 direction of Delta (default: best variant)")
    ap.add_argument("--render3d", action="store_true", help="pyvista render of the meshes (py -3.11)")
    ap.add_argument("--scaled-ovoids", type=float, nargs="+", default=None, metavar="D",
                    help="write lunar caps at realistic assembly diameters D mm (e.g. 22 26 30) as ovoid_{L,R}_d<D>.obj; "
                         "the fitted caps are left untouched")
    ap.add_argument("--variant", default=None, help="write everything to hybrid/applicator_<variant>/ (e.g. v3) instead of hybrid/applicator/")
    ap.add_argument("--rods", action="store_true", help="Stage 3: straight tandem rod + the two cap rods (rod_L/rod_R.obj), see params rod_*")
    ap.add_argument("--angle-source", default=None, help="provenance note recorded for --angle")
    ap.add_argument("--align-shaft", action="store_true",
                    help="with --scaled-ovoids: tilt the caps so their axis follows the STRAIGHT vaginal rod instead of "
                         "the intrauterine tube (tag suffix 's', e.g. ovoid_L_d26s.obj)")
    ap.add_argument("--tandem-only", action="store_true",
                    help="fix plan S4a/S4c: build the tandem body alone (tube + vaginal tandem swept along the label "
                         "centreline) from the UPDATED BT applicator label into applicator_<variant>/ (needs --variant); "
                         "pose.json = the --pose-from variant's, re-derived by the same rule and checked; no ovoid body")
    ap.add_argument("--pose-from", default="v3", help="with --tandem-only: the variant whose validated tube pose is kept (v3)")
    ap.add_argument("--force", action="store_true",
                    help="with --tandem-only: rebuild even when the variant's pose.json insertion_path_tandem_first no longer "
                         "matches the rebuilt body / pose (it is moved to pose.json stale_records; without --force the "
                         "rebuild stops before writing anything)")
    ap.add_argument("--app-label", default=None, help="with --tandem-only / --ring: the updated BT applicator label (default "
                                                      "APPSIM_APP_LABEL, else <data>/../BT_MRI_label_applicator.nii)")
    ap.add_argument("--ring", action="store_true",
                    help="fix plan S4b/S4c: the --tandem-from variant's tandem body (copied, checked) PLUS two lunar ring "
                         "halves, each one closed manifold with its socket boss and rod, measured on the updated BT "
                         "applicator label and the BT ovoid label, into applicator_<variant>/ (needs --variant, a new name)")
    ap.add_argument("--tandem-from", default="v4", help="with --ring: the tandem-only variant whose tandem body, pose and "
                                                        "tandem-first record are carried (v4)")
    ap.add_argument("--ring-outer-r", type=float, default=None,
                    help="with --ring: ring OUTER radius in mm (default params ring_outer_r_mm: 20, U1 = label)")
    ap.add_argument("--ring-approach-dir", choices=["rod_sagittal", "rod", "handle"], default=None,
                    help="with --ring: the direction each half backs out along (default params ring_approach_dir)")
    a = ap.parse_args()
    if a.ring and a.tandem_only:
        raise SystemExit("--ring and --tandem-only build different variants: pass one")
    global APP
    base_app = APP
    if a.variant:
        APP = HY + "/applicator_" + a.variant
    if a.render3d:
        render3d(); return
    if a.scaled_ovoids:
        scaled_ovoids(a.scaled_ovoids, align_shaft=a.align_shaft); return
    if a.tandem_only:
        main_tandem_only(a); return
    if a.ring:
        main_ring(a); return
    t0 = time.time()
    for d in (APP, FIGS, LOGS):
        os.makedirs(d, exist_ok=True)
    app_in = json.load(open(P["inputs"] + "/applicator.json")); bdev = json.load(open(P["out"] + "/final/bdev.json"))
    prm = default_params(app_in, bdev)
    if a.angle is not None:
        prm["angle_deg"]["value"] = float(a.angle)
        if a.angle_source:
            prm["angle_deg"]["source"] = a.angle_source
    if a.rods:
        prm["rods"]["value"] = True
        prm["shaft_style"]["value"] = "straight"
    if a.n_steps is not None:
        prm["n_steps"]["value"] = int(a.n_steps)
    if a.path_axis is not None:
        prm["path_axis"]["value"] = a.path_axis
    stored = APP + "/applicator.json"
    if a.no_fit and not os.path.exists(stored) and os.path.exists(base_app + "/applicator.json"):
        stored = base_app + "/applicator.json"           # a fresh variant dir reuses the base fit
    if a.no_fit and os.path.exists(stored):
        old = json.load(open(stored))["params"]
        for k in FIT_KEYS:
            prm[k]["value"] = old[k]["value"]
    log = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(a))
    # ---- ovoid fit (needs the BT labels)
    bt = load_bt()
    fit = fit_ovoids(prm, bt, do_fit=not a.no_fit)
    if not a.no_figs:
        fig_ovoid_fit(prm, bt, fit)
    # ---- meshes
    mesh = None
    if not a.no_meshes:
        mesh = build_meshes(prm)
    elif os.path.exists(stored):
        mesh = json.load(open(stored)).get("parts")
    # ---- pose rule
    pre = load_pre()
    Pf, n_fix = vagina_fixed_point(pre, val(prm, "vagina_fixed_inferior_mm"))
    d_F_bdev = float(bdev["picks"]["d_F_tip"])
    rule_inputs = dict(O_pre=pre["O_pre"], L_end=pre["L_end"], a0=pre["a0"], internal_os=pre["internal_os"],
                       canal_above_L_end_mm=pre["canal_above_L_end_mm"], vagina_fixed_point=Pf, corpus_centroid=pre["uterus_X"].mean(0),
                       vagina_axis=pre["vagina"]["axis"], vagina_centroid=pre["vagina"]["centroid"])
    # ---- rule v1 (rejected; kept as the reference) and its validation, then rule v2
    frames = load_frames()
    p1 = json.loads(json.dumps(prm))
    p1["shaft_axis_mode"]["value"] = "chord"; p1["shaft_line_through"]["value"] = "O_pre"; p1["flange_shift_mm"]["value"] = 0.0
    p1["path_axis"]["value"] = "shaft"
    rule_v1 = pose_rule(rule_inputs, p1)
    valid, m_k0 = validate(p1, pre, bt, frames, rule_inputs, rule_v1)
    v2 = None; score = valid["rule_v1"]; label = "rule v1 (%g deg)" % val(prm, "angle_deg")
    if a.rule == "v2":
        v2 = validate_v2(prm, pre, bt, frames, rule_inputs, valid)
        ch = v2["chosen"]
        prm["shaft_axis_mode"]["value"] = "pca"
        prm["shaft_line_through"]["value"] = a.shaft_line_through or ch["shaft_line_through"]
        prm["shift_axis"]["value"] = a.shift_axis or ch["shift_axis"]
        prm["flange_shift_mm"]["value"] = float(a.flange_shift) if a.flange_shift is not None else float(ch["fine_best_flange_shift_mm"])
        if a.path_axis is None:      # v2: the tube enters the canal first, so the device translates along the tube axis (shaft path = alt)
            prm["path_axis"]["value"] = "tube"
        rule = pose_rule(rule_inputs, prm)
        V0u, F0u = geom.read_obj(P["inputs"] + "/pre_uterus.obj")
        score = compare_to_bt(rule, frames["BONE"], bt, val(prm, "L_iu_mm"))
        score["corpus_K0"], m_k0 = corpus_k0_metrics(rule, frames["BONE"], bt, V0u, F0u, pre["uterus_X"])
        score.update(flange_shift_mm=val(prm, "flange_shift_mm"), shaft_line_through=val(prm, "shaft_line_through"), shift_axis=val(prm, "shift_axis"))
        label = "rule v2 (vagina axis through %s, Delta %.1f mm along %s)" % (val(prm, "shaft_line_through"), val(prm, "flange_shift_mm"), val(prm, "shift_axis"))
    else:
        prm = p1; rule = rule_v1
    if abs(rule["d_F_mm"] - d_F_bdev) > 0.2:
        print("WARNING: d_F %.2f differs from bdev.json d_F_tip %.2f" % (rule["d_F_mm"], d_F_bdev))
    print("[pose] fixed point %s (%d vox), chord %.1f mm; shaft axis %s, tube axis %s (angle to a0 %.1f deg); d_F %.2f mm; corpus rotation %.1f deg, "
          "centroid shift %.1f mm; path travel %.1f mm, u_ios %.3f" % (jz(Pf, 1), n_fix, float(np.linalg.norm(pre["O_pre"] - Pf)), jz(rule["shaft_axis"], 3),
                                                                       jz(rule["tube_axis"], 3), geom.angle_deg(rule["tube_axis"], pre["a0"]), rule["d_F_mm"],
                                                                       rule["corpus_rotation_deg"], float(np.linalg.norm(rule["corpus_centroid_target"] - rule["corpus_centroid_pre"])),
                                                                       rule["insertion_path"]["travel_mm"], rule["insertion_path"]["u_tip_at_internal_os"]), flush=True)
    # ---- corpus targets per Delta (the scene module picks Delta from its cfg)
    by_dz = {}
    if v2 is not None:
        for dz in sorted(set([float(d) for d in val(prm, "flange_shift_sweep_mm")] + [float(val(prm, "flange_shift_mm"))])):
            p2 = json.loads(json.dumps(prm)); p2["flange_shift_mm"]["value"] = dz
            r2 = pose_rule(rule_inputs, p2)
            by_dz["%g" % dz] = dict(flange=jz(r2["flange"], 4), tip=jz(r2["tip"], 4), T_preBT_to_target=jz(r2["corpus_T"], 6), R=jz(r2["corpus_R"], 6),
                                    t=jz(r2["corpus_t"], 4), screw=jz(r2["screw"]), L_end_target=jz(r2["L_end_target"], 4),
                                    corpus_centroid_target=jz(r2["corpus_centroid_target"], 3), rotation_deg=round(r2["corpus_rotation_deg"], 3),
                                    u_tip_at_internal_os=r2["insertion_path"]["u_tip_at_internal_os"], travel_mm=r2["insertion_path"]["travel_mm"])
    # ---- write applicator.json
    ch = needle_channels(prm)
    C_arc, T_arc = shaft_arc(prm, 31)
    if val(prm, "shaft_style") == "straight":
        d_rod, _ = rod_frame(prm)
        C_arc = np.stack([np.zeros(3), val(prm, "rod_len_mm") * d_rod]); T_arc = np.stack([d_rod, d_rod])
    app_out = dict(units="mm, deg", frame=FRAME_APP, device="Venezia-type hybrid applicator, parametric model (identity unconfirmed: RTPLAN / vendor "
                   "geometry not available); surfaces are closed, outward-oriented triangle meshes for SOFA collision + visual models",
                   written=time.strftime("%Y-%m-%d %H:%M:%S"), params=prm, parts=mesh,
                   landmarks=dict(flange=[0.0, 0.0, 0.0], tip=[0.0, 0.0, val(prm, "L_iu_mm")], shaft_end=jz(C_arc[-1], 3), shaft_end_dir=jz(T_arc[-1]),
                                  cap_apex_z=round(val(prm, "ovoid_dz_mm"), 3), cap_bottom_z=round(val(prm, "ovoid_dz_mm") - val(prm, "ovoid_height_mm"), 3),
                                  cap_centres=[jz([s * (val(prm, "ovoid_offset_x_mm") + 3.0 / 8.0 * val(prm, "ovoid_diam_mm") / 2), 0.0,
                                                   val(prm, "ovoid_dz_mm") - 0.5 * val(prm, "ovoid_height_mm")], 3) for s in (-1, 1)],
                                  cap_centres_note="centroid-like reference points of the two caps (L, R), not used by the geometry"),
                   needle_channels=ch, ovoid_fit=fit,
                   BT_pose=dict(frame="BT world (nibabel affine of BT_MRI_label_*.nii)", origin_BT_world=app_in["origin_BT_world"],
                                R_rows_BT_world=app_in["R_rows_BT_world"], note="measured BT tandem flange/axis (inputs/applicator.json); the fit used this pose"))
    json.dump(jz(app_out), open(APP + "/applicator.json", "w"), indent=1)
    # ---- write pose.json
    vg = pre["vagina"]
    pose_out = dict(
        units="mm, deg",
        frame=FRAME_PRE,
        rule=("kinematic pose rule v2 (CONTRACT 3, amended 2026-09-11): shaft axis = vagina principal axis, flange = base + Delta (seating shift, "
              "SCENARIO parameter); the real BT pose enters only the validation" if v2 is not None else
              "kinematic pose rule v1 (CONTRACT 3.1-3.4 original; REJECTED on this case); the real BT pose enters only the validation"),
        rule_version=a.rule,
        rule_params=dict(shaft_axis_mode=val(prm, "shaft_axis_mode"), shaft_line_through=val(prm, "shaft_line_through"), shift_axis=val(prm, "shift_axis"),
                         flange_shift_mm=val(prm, "flange_shift_mm"), path_axis=val(prm, "path_axis"), angle_deg=val(prm, "angle_deg")),
        default_flange_shift_mm=val(prm, "flange_shift_mm"),
        default_flange_shift_note="IN-SAMPLE best of the Delta sweep on this case (validation.rule_v2.chosen); Delta is a scenario parameter, "
                                  "not a prediction - sweep it",
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        inputs=dict(O_pre=dict(value=jz(pre["O_pre"], 4), source="MEASURED (inputs/poses.json: external os on the preBT canal axis)"),
                    L_end=dict(value=jz(pre["L_end"], 4), source="MEASURED (inputs/poses.json)"),
                    a0=dict(value=jz(pre["a0"]), source="MEASURED (inputs/poses.json: lower-canal PCA axis)"),
                    internal_os=dict(value=jz(pre["internal_os"], 4), source="MEASURED (inputs/canal.npz i_internal_os)"),
                    canal_above_L_end_mm=dict(value=round(pre["canal_above_L_end_mm"], 3), source="MEASURED (inputs/canal.npz arclength; bdev.json 44.8)"),
                    vagina_fixed_point=dict(value=jz(Pf, 4), n_vox=n_fix, source="MEASURED: centroid of the lowest %g mm of the vagina body "
                                            "(vagina & ~uterus & ~HR-CTV, largest 6-connected component) along its principal axis oriented +S" % val(prm, "vagina_fixed_inferior_mm"),
                                            vagina_axis=jz(vg["axis"]), vagina_centroid=jz(vg["centroid"], 3), vagina_length_along_axis_mm=round(vg["proj_max"] - vg["proj_min"], 2),
                                            vagina_body_vol_cc=round(vg["vol_cc"], 3)),
                    L_iu_mm=dict(value=val(prm, "L_iu_mm"), source=prm["L_iu_mm"]["source"]),
                    angle_deg=dict(value=val(prm, "angle_deg"), source=prm["angle_deg"]["source"]),
                    d_F_mm=dict(value=round(rule["d_F_mm"], 3), rule="L_iu + %g - canal_above_L_end" % val(prm, "d_F_margin_mm"), bdev_d_F_tip=d_F_bdev,
                                source="CONTRACT 3.3 / final/bdev.json (geometry only)"),
                    tilt_axis=val(prm, "tilt_axis"), path_axis=val(prm, "path_axis"), n_steps=val(prm, "n_steps"), corpus_ease=val(prm, "corpus_ease")),
        device_final=dict(flange=jz(rule["flange"], 4), shaft_axis=jz(rule["shaft_axis"]), tube_axis=jz(rule["tube_axis"]), x_app=jz(rule["x_app"]), y_app=jz(rule["y_app"]),
                          R_rows=jz(rule["R_rows"]), tip=jz(rule["tip"], 4), shaft_end=jz(rule["shaft_end"], 4), shaft_end_dir=jz(rule["shaft_end_dir"]),
                          angle_shaft_tube_deg=round(rule["angle_shaft_tube_deg"], 3), tube_axis_angle_to_a0_deg=round(geom.angle_deg(rule["tube_axis"], pre["a0"]), 3),
                          tube_axis_sagittal_variant=jz(rule["tube_axis_sagittal_variant"]), tilt_variants_diff_deg=round(rule["tilt_variants_diff_deg"], 3),
                          cap_centres_world=[jz(to_world(rule["flange"], rule["R_rows"], c)[0], 3) for c in app_out["landmarks"]["cap_centres"]],
                          convention="p_world = flange + p_app @ R_rows (rows = applicator x, y, z axes in preBT world)"),
        corpus=dict(T_preBT_to_target=jz(rule["corpus_T"], 6), R=jz(rule["corpus_R"], 6), t=jz(rule["corpus_t"], 4), rotation_deg=round(rule["corpus_rotation_deg"], 3),
                    L_end_target=jz(rule["L_end_target"], 4), d_F_mm=round(rule["d_F_mm"], 3),
                    screw=jz(rule["screw"]), screw_convention="T(s) p = R(s) (p - point) + point + s * pitch * axis, R(s) = rotation by s*angle about axis; s in [0,1]",
                    corpus_centroid_pre=jz(rule["corpus_centroid_pre"], 3), corpus_centroid_target=jz(rule["corpus_centroid_target"], 3),
                    centroid_shift_mm=round(float(np.linalg.norm(rule["corpus_centroid_target"] - rule["corpus_centroid_pre"])), 3),
                    definition="rigid map of the preBT corpus (uterus label) to its target: a0 -> tube axis by the minimal rotation (roll preserved), "
                               "L_end -> flange + d_F * tube axis",
                    by_flange_shift_mm=by_dz,
                    by_flange_shift_note="rule v2 corpus targets per seating shift Delta (keys = Delta in mm as strings); the scene picks one from cfg"),
        insertion_path=dict(rule["insertion_path"], phases=dict(approach="u < u_tip_at_internal_os: device translates, corpus at rest",
                                                                insertion="u >= u_tip_at_internal_os: corpus screw motion s(u) to the target at u = 1",
                                                                settle="after u = 1 until convergence (CONTRACT 5)"),
                            keyframe_fields="u, phase, F (flange), a (tube axis), x (x_app), tip, corpus_s, corpus_T (4x4 preBT -> current)"),
        insertion_path_alt=dict(rule["insertion_path_alt"], note="alternative translation axis (not the default)"),
        insertion_path_canal=canal_path_record(rule, pre, prm),
        validation=dict(rule_v1=valid, rule_v2=v2, default_rule_score=score))
    json.dump(jz(pose_out), open(APP + "/pose.json", "w"), indent=1)
    print("[pose] wrote", APP + "/pose.json", "and", APP + "/applicator.json", flush=True)
    # ---- figures
    if not a.no_figs:
        import evaluate as ev
        alt_rules = {"v1": rule_v1} if v2 is not None else {}
        for ang in (0.0, 30.0):
            p2 = json.loads(json.dumps(prm)); p2["angle_deg"]["value"] = ang; alt_rules["angle_%g" % ang] = pose_rule(rule_inputs, p2)
        if v2 is None:
            inp = dict(rule_inputs); inp["vagina_fixed_point"] = pre["O_pre"] - float(np.linalg.norm(pre["O_pre"] - Pf)) * pre["vagina"]["axis"]
            alt_rules["vagina_pca"] = pose_rule(inp, prm)
        inp = dict(rule_inputs); inp["tube_axis_override"] = pre["a0"]; alt_rules["tube_a0"] = pose_rule(inp, prm)
        fig_pose_rule_bt(prm, pre, bt, frames, rule, valid, m_k0, alt_rules, score=score, label=label)
        V0, F0 = geom.read_obj(P["inputs"] + "/pre_uterus.obj")
        m_ct = ev.voxelize(V0 @ rule["corpus_R"].T + rule["corpus_t"], F0, pre["shape"], pre["aff"])
        fig_pose_rule_pre(prm, pre, rule, rule_inputs, m_ct, label=label)
        fig_pose_rule_path(prm, pre, rule, rule_inputs)
        if v2 is not None:
            fig_delta_sweep(v2, label)
    cmds = ["cd %s" % HERE.replace("\\", "/"), "set MPLBACKEND=Agg                 (PowerShell: $env:MPLBACKEND='Agg')",
            "set PYTHONDONTWRITEBYTECODE=1      (PowerShell: $env:PYTHONDONTWRITEBYTECODE='1')",
            "python applicator_venezia.py                 # meshes + ovoid fit + pose rule + validation + figures + README section",
            "python applicator_venezia.py --no-fit        # reuse the stored ovoid fit", "py -3.11 applicator_venezia.py --render3d    # figs/applicator_3d.png"]
    if not a.no_readme:
        write_readme(prm, mesh or {}, fit, rule, valid, cmds, v2=v2, score=score, label=label)
    log.update(wall_s=round(time.time() - t0, 1), fit=fit, mesh=mesh,
               validation_summary=dict(rule_v1=valid["rule_v1"], alternatives=valid["alternatives"],
                                       rule_v2_chosen=None if v2 is None else dict(v2["chosen"], fine_table=None), default_rule_score=score))
    json.dump(jz(log), open(LOGS + "/applicator_venezia.json", "w"), indent=1)
    print("[done] wall %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
