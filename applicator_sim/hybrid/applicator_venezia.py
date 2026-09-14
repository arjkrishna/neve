"""HOST: parametric Venezia-type applicator and the kinematic pose rule v1 (CONTRACT.md sections 2 and 3).

    cd applicator_sim/hybrid        (set MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1)
    python applicator_venezia.py                    # meshes + ovoid fit + pose rule + validation + figures + README
    python applicator_venezia.py --no-fit           # reuse the ovoid fit stored in applicator/applicator.json
    python applicator_venezia.py --angle 24 --n-steps 80 --path-axis shaft
    py -3.11 applicator_venezia.py --render3d       # pyvista render of the device meshes -> figs/applicator_3d.png

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
    a = ap.parse_args()
    if a.render3d:
        render3d(); return
    t0 = time.time()
    for d in (APP, FIGS, LOGS):
        os.makedirs(d, exist_ok=True)
    app_in = json.load(open(P["inputs"] + "/applicator.json")); bdev = json.load(open(P["out"] + "/final/bdev.json"))
    prm = default_params(app_in, bdev)
    if a.angle is not None:
        prm["angle_deg"]["value"] = float(a.angle)
    if a.n_steps is not None:
        prm["n_steps"]["value"] = int(a.n_steps)
    if a.path_axis is not None:
        prm["path_axis"]["value"] = a.path_axis
    stored = APP + "/applicator.json"
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
