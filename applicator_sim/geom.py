"""Pure-numpy geometry shared by host (py3.13) and container (py3.8). Units: mm, deg.

Conventions
- An applicator frame is a 3x3 matrix R whose ROWS are the frame axes (x, y, z) in world
  coordinates; z is the tandem axis pointing from the flange to the tip. p_world = F + R.T @ p_app.
- A pose is a dict(F=flange (3,), a=unit axis (3,), x=unit x_app (3,)).
"""
import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def ortho(x, z):
    """x orthogonalised against unit z, normalised."""
    z = unit(z)
    x = np.asarray(x, float) - np.dot(x, z) * z
    return unit(x)


def frame_from(z, xref=(1.0, 0.0, 0.0)):
    z = unit(z)
    x = ortho(xref, z)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def pose_frame(pose):
    z = unit(pose["a"]); x = ortho(pose["x"], z); y = np.cross(z, x)
    return np.stack([x, y, z])


def app_to_world(pose, p_app):
    R = pose_frame(pose)
    return np.asarray(pose["F"], float) + np.atleast_2d(p_app) @ R


def world_to_app(pose, p):
    R = pose_frame(pose)
    return (np.atleast_2d(p) - np.asarray(pose["F"], float)) @ R.T


# ----------------------------------------------------------------------------- rotations
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def rot_between(a, b):
    """Minimal rotation taking unit a to unit b (Rodrigues)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -1 + 1e-12:  # antiparallel: rotate pi about any perpendicular axis
        p = ortho([1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0], a)
        return 2 * np.outer(p, p) - np.eye(3)
    K = skew(v)
    return np.eye(3) + K + K @ K / (1 + c)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def slerp(a, b, t):
    a, b = unit(a), unit(b)
    om = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if om < 1e-9:
        return a.copy()
    return unit((np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om))


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def lerp(p, q, t):
    return (1 - t) * np.asarray(p, float) + t * np.asarray(q, float)


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| (rows are points). Proper rotation."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


# ----------------------------------------------------------------------------- insertion path
def pose_at(u, O, a0, F_fin, a_fin, L_iu, onset=0.3):
    """Phase-T kinematics.  P(u)=lerp(O,F_fin,u); a(u)=slerp(a0,a_fin,smoothstep((u-onset)/(1-onset)));
    tip(u)=P(u)+L_iu*u*a(u); F(u)=tip(u)-L_iu*a(u).  Returns (F, a, tip)."""
    P = lerp(O, F_fin, u)
    a = slerp(a0, a_fin, smoothstep((u - onset) / (1 - onset)))
    tip = P + L_iu * u * a
    return tip - L_iu * a, a, tip


def pose_path(O, a0, x0, F_fin, a_fin, L_iu, step_mm=1.0, max_rot_deg=0.5, onset=0.3):
    """Adaptive u schedule: tip travel <= step_mm and axis rotation <= max_rot_deg per step.
    x_app is parallel-transported by the minimal rotation between successive axes."""
    out = []
    u = 0.0
    F, a, tip = pose_at(0.0, O, a0, F_fin, a_fin, L_iu, onset)
    x = ortho(x0, a)
    out.append(dict(u=0.0, F=F, a=a, x=x, tip=tip))
    while u < 1.0 - 1e-12:
        du = 1.0 - u
        for _ in range(60):
            F1, a1, tip1 = pose_at(u + du, O, a0, F_fin, a_fin, L_iu, onset)
            if np.linalg.norm(tip1 - tip) <= step_mm + 1e-9 and angle_deg(a, a1) <= max_rot_deg + 1e-9:
                break
            du *= 0.5
        # grow back toward the limit (bisection between du and 2du)
        lo, hi = du, min(2 * du, 1.0 - u)
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            F1, a1, tip1 = pose_at(u + mid, O, a0, F_fin, a_fin, L_iu, onset)
            if np.linalg.norm(tip1 - tip) <= step_mm + 1e-9 and angle_deg(a, a1) <= max_rot_deg + 1e-9:
                lo = mid
            else:
                hi = mid
        du = lo
        u = min(1.0, u + du)
        F1, a1, tip1 = pose_at(u, O, a0, F_fin, a_fin, L_iu, onset)
        x = ortho(rot_between(a, a1) @ x, a1)
        F, a, tip = F1, a1, tip1
        out.append(dict(u=float(u), F=F, a=a, x=x, tip=tip))
    return out


# ----------------------------------------------------------------------------- screw motions and the canal path
def rodrigues(k, deg):
    """Rotation by `deg` about the unit axis k."""
    k = unit(k)
    K = skew(k)
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def screw_decompose(R, t):
    """(R, t) -> screw: unit axis k, point p0, angle (deg), translation d along k:  T(p) = R (p - p0) + p0 + d k."""
    th = np.radians(rot_angle_deg(R))
    if th < 1e-4:
        return dict(pure_translation=True, axis=unit(t) if np.linalg.norm(t) > 0 else np.array([0.0, 0.0, 1.0]),
                    point=np.zeros(3), angle_deg=0.0, pitch_mm=float(np.linalg.norm(t)))
    k = unit(np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)))
    d = float(t @ k)
    tp = t - d * k
    p0 = 0.5 * (tp + np.cross(k, tp) / np.tan(th / 2))
    return dict(pure_translation=False, axis=k, point=p0, angle_deg=float(np.degrees(th)), pitch_mm=d)


def screw_interp(sc, s):
    """Rigid transform (4x4) at fraction s of a screw motion from screw_decompose."""
    T = np.eye(4)
    k = np.asarray(sc["axis"], float)
    if sc.get("pure_translation"):
        T[:3, 3] = s * float(sc["pitch_mm"]) * k
        return T
    p0 = np.asarray(sc["point"], float)
    Rs = rodrigues(k, s * float(sc["angle_deg"]))
    T[:3, :3] = Rs
    T[:3, 3] = p0 - Rs @ p0 + s * float(sc["pitch_mm"]) * k
    return T


def corpus_rule_T(a0, L_end, d_F, F, a):
    """The pose rule's corpus placement for a tube at flange F with axis a (4x4, preBT -> target): the minimal
    rotation a0 -> a, and the canal landmark L_end carried to F + d_F a.  applicator_venezia.pose_rule, 3.3."""
    R = rot_between(a0, a)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (np.asarray(F, float) + float(d_F) * unit(a)) - R @ np.asarray(L_end, float)
    return T


def canal_path(u, base, a_v, F_fin, a_fin, L_iu, h_intro, x_fin, below=4.0, u1=None):
    """Tandem placement THROUGH the vagina and INTO the canal, in two stages (Stage 3, 2026-09-21).

    Why: the pre-implant external os sits 16 mm off the vaginal axis line and the lower canal is 24 deg from
    it, so no straight tube can lie in the vagina and point along the canal at once; a line through the os
    along the canal passes ~37 mm outside the introitus.  Clinically the tandem goes up the vagina and the
    cervix is DRAWN onto it (traction), then the uterus straightens onto the tube.  Hence:
      S1 (u < u1): tube along the vaginal axis a_v, translating up the axis line through `base` (the os
          projected on the line); the tip rises from `below` mm under the introitus (height h_intro < 0 about
          `base`) to `base` (the vault).  Corpus at rest.
      S2 (u >= u1): tip-pinned swing: tip = lerp(base, tip_fin, w), a = slerp(a_v, a_fin, w), F = tip - L a,
          w = smoothstep((u - u1) / (1 - u1)); the caller blends the corpus from rest onto the rule's target
          for the CURRENT tube pose with the same w (no jump), so at u = 1 the pose and the corpus are the
          validated final ones.
    u1 defaults to the S1 share of the total tip travel (uniform tip speed).  x is the applicator x axis,
    parallel-transported from x_fin so the final frame is bit-identical to the rule's R_rows.
    Returns dict(F, a, x, tip, w, stage)."""
    base, a_v, F_fin, a_fin, x_fin = (np.asarray(v, float) for v in (base, a_v, F_fin, a_fin, x_fin))
    a_v, a_fin = unit(a_v), unit(a_fin)
    tip_fin = F_fin + L_iu * a_fin
    h0 = float(h_intro) - float(below)                    # tip height (about base, along a_v) at u = 0
    s1 = -h0                                              # S1 tip travel (up to the vault at height 0)
    s2 = float(np.linalg.norm(tip_fin - base))
    if u1 is None:
        u1 = s1 / (s1 + s2)
    u = float(np.clip(u, 0.0, 1.0))
    if u < u1:
        tip = base + (h0 + (0.0 - h0) * (u / u1)) * a_v
        a = a_v
        w = 0.0
        stage = "S1"
    else:
        w = smoothstep((u - u1) / max(1e-9, 1.0 - u1))
        tip = lerp(base, tip_fin, w)
        a = slerp(a_v, a_fin, w)
        stage = "S2"
    x = ortho(rot_between(a_fin, a) @ x_fin, a)
    return dict(F=tip - L_iu * a, a=a, x=x, tip=tip, w=w, stage=stage, u1=float(u1))


# ----------------------------------------------------------------------------- tandem-first kinematics (G32 fix plan S5)
# The tandem goes in FIRST and its tip follows the physician's tandem path (hybrid/tandem_path.py, S3): phase V up the
# vaginal slit to the os O_true, phase C through the labelled lower canal while the corpus ROTATES about the os, phase
# L tandem + corpus lifted together onto the validated final pose.  The pose rule's corpus transform is split for that:
#     T_final = Trans(L) o Rot(theta about p),   p = O_true,   L = T_final(p) - p,
# so the os does not move in C (the vault stays on the portio) and the cranial lift comes last, with the device.
# A part L_C of the lift may be started in C (the plan's fallbacks: caudal traction, part of the lift; flagged), in
# which case the corpus in C is Trans(w L_C) o Rot(w theta about p) and L lifts by L - L_C.
# Pure numpy, py3.8-safe: the host (hybrid/tf_gate.py) searches the rotation weight w_r(d) against the wall and
# applicator_venezia.tandem_first_record stores the knots in pose.json; the scene regenerates the SAME rows from them
# with tandem_first_path (scene_hybrid.build_schedule_tandem_first).
def rigid4(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t, float)
    return T


def corpus_split(T, p):
    """T (4x4) = Trans(L) o Rot(theta about p): the unit rotation axis k, theta (deg), L = T(p) - p and the pivot."""
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    p = np.asarray(p, float)
    k = unit(np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))
    return dict(pivot=p, axis=k, theta_deg=rot_angle_deg(R), L=R @ p + t - p)


def tf_corpus_T(pivot, axis, theta_deg, w, L_C=None):
    """Corpus transform at rotation weight w: Trans(w L_C) o Rot(w theta about the pivot) (4x4)."""
    p = np.asarray(pivot, float)
    Rw = rodrigues(axis, float(w) * float(theta_deg))
    t = p - Rw @ p
    if L_C is not None:
        t = t + float(w) * np.asarray(L_C, float)
    return rigid4(Rw, t)


def polyline_at(P, sv, q):
    """Points at arclengths q on the polyline P (vertex arclengths sv, increasing), extrapolated linearly beyond
    both ends along the end segments."""
    P = np.asarray(P, float)
    sv = np.asarray(sv, float)
    q = np.atleast_1d(np.asarray(q, float))
    out = np.stack([np.interp(q, sv, P[:, k]) for k in range(3)], 1)
    lo, hi = q < sv[0], q > sv[-1]
    if lo.any():
        d0 = (P[1] - P[0]) / (sv[1] - sv[0])
        out[lo] = P[0] + np.outer(q[lo] - sv[0], d0)
    if hi.any():
        d1 = (P[-1] - P[-2]) / (sv[-1] - sv[-2])
        out[hi] = P[-1] + np.outer(q[hi] - sv[-1], d1)
    return out


def tf_axis_fit(tip, Q, wq):
    """Direction of the line through `tip` that best fits the points Q (weights wq, least squares on the
    perpendicular distances): the top eigenvector of the weighted scatter about the tip, oriented towards the tip."""
    D = np.asarray(Q, float) - np.asarray(tip, float)
    M = (D * np.asarray(wq, float)[:, None]).T @ D
    _, U = np.linalg.eigh(M)
    a = U[:, -1]
    return unit(-a if a @ (np.asarray(tip, float) - np.asarray(Q, float)[0]) < 0 else a)


def seg_within_frac(P, F, a, L, tol):
    """Fraction of the points P within `tol` of the segment F .. F + L a (the tube line of the canal-in-tube tests)."""
    P = np.atleast_2d(np.asarray(P, float))
    if not len(P):
        return 1.0
    q = P - np.asarray(F, float)
    h = np.clip(q @ a, 0.0, float(L))
    return float(np.mean(np.linalg.norm(q - np.outer(h, a), axis=1) <= float(tol)))


def tf_least_motion_axis(a_prev, a_goal, frac_fn, frac_min, n_bisect=40):
    """The axis that moves LEAST from a_prev towards a_goal (along the great circle) while frac_fn(axis) >= frac_min:
    a_prev itself if it already satisfies it, else the first satisfying point found by bisection, else a_goal.
    Returns (axis, alpha in [0, 1] = how far along a_prev -> a_goal)."""
    a_prev, a_goal = unit(a_prev), unit(a_goal)
    if frac_fn(a_prev) >= frac_min or angle_deg(a_prev, a_goal) < 1e-9:
        return a_prev, 0.0
    if frac_fn(a_goal) < frac_min:
        return a_goal, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        if frac_fn(slerp(a_prev, a_goal, mid)) >= frac_min:
            hi = mid
        else:
            lo = mid
    return unit(slerp(a_prev, a_goal, hi)), hi


def tf_frame(a, a_fin, x_fin):
    """Applicator rows (x, y, z) for tube axis a: x_fin carried by the minimal rotation a_fin -> a (as canal_path),
    so the frame is exactly the final one at a = a_fin and never spins about the tube."""
    a = unit(a)
    x = ortho(rot_between(a_fin, a) @ np.asarray(x_fin, float), a)
    return np.array([x, np.cross(a, x), a])


def tf_row_V(prm, q):
    """Phase V row: tip at path arclength q (< 0 in the vagina), tube axis = the chord of the path over the tube
    length behind the tip, corpus at rest."""
    tau_pts, tau_s, L = prm["tau_pts"], prm["tau_s"], float(prm["L_iu"])
    tip = polyline_at(tau_pts, tau_s, [q])[0]
    a = unit(tip - polyline_at(tau_pts, tau_s, [q - L])[0])
    return dict(phase="V", tip_s=float(q), d_mm=float(q), w_r=0.0, w_l=0.0, tip=tip, F=tip - L * a, tube_axis=a,
                R_rows=tf_frame(a, prm["a_fin"], prm["x_fin"]), T_corpus=np.eye(4), axis_alpha=None, axis_beta=0.0)


def tf_row_C(prm, d, w, a_prev):
    """Phase C row at depth d and rotation weight w (the plan's formula):
        tip = T_C(w) tau(d) + w delta,   T_C(w) = Trans(w L_C) o Rot(w theta about p),
    tau = the path for d <= follow_mm (the labelled lower canal), then straight along a_lc.  Tube axis: the line
    through the tip fitted to the traversed carried path (carried O_true weighted fit_w_os), reached from the previous
    row's axis by the LEAST motion that keeps frac_min of the traversed lower canal (0 <= s <= min(d, follow_mm))
    within frac_tol_mm of the tube, then blended onto a_fin by beta = smoothstep((sigma - sigma0) / (1 - sigma0)),
    sigma = (d / d_fin + w) / 2, so the row at (d_fin, 1) is exactly the tandem at Trans(L_C - L) device_final."""
    tau_pts, tau_s = np.asarray(prm["tau_pts"], float), np.asarray(prm["tau_s"], float)
    L, d_fin, fol = float(prm["L_iu"]), float(prm["d_fin"]), float(prm["follow_mm"])
    Tc = tf_corpus_T(prm["pivot"], prm["rot_axis"], prm["theta_deg"], w, prm["L_C"])
    Rc, tc = Tc[:3, :3], Tc[:3, 3]
    tip = Rc @ polyline_at(tau_pts, tau_s, [d])[0] + tc + float(w) * np.asarray(prm["delta"], float)
    if d >= float(prm["fit_min_mm"]):
        ss = np.linspace(0.0, d, int(np.ceil(d)) + 1)
        Q = polyline_at(tau_pts, tau_s, ss) @ Rc.T + tc
        wq = np.ones(len(Q))
        wq[0] = float(prm["fit_w_os"])
        a_goal = tf_axis_fit(tip, Q, wq)
    else:
        a_goal = unit(a_prev)
    m = (tau_s >= 0.0) & (tau_s <= min(d, fol))
    canal = tau_pts[m] @ Rc.T + tc
    tol = float(prm["frac_tol_mm"])

    def frac_fn(a):
        return seg_within_frac(canal, tip - L * a, a, L, tol)
    a_lm, alpha = tf_least_motion_axis(a_prev, a_goal, frac_fn, float(prm["frac_min"]))
    s0 = float(prm["blend_sigma0"])
    sigma = 0.5 * (max(float(d), 0.0) / d_fin + float(w))
    beta = smoothstep((sigma - s0) / (1.0 - s0))
    a = unit(slerp(a_lm, prm["a_fin"], beta)) if beta > 0.0 else a_lm
    return dict(phase="C", tip_s=float(d), d_mm=float(d), w_r=float(w), w_l=0.0, tip=tip, F=tip - L * a, tube_axis=a,
                R_rows=tf_frame(a, prm["a_fin"], prm["x_fin"]), T_corpus=Tc, axis_alpha=float(alpha),
                axis_beta=float(beta), axis_goal_deg=angle_deg(a, a_goal), frac_plan=frac_fn(a))


def tf_row_L(prm, wl):
    """Phase L row: tandem and corpus translate together by w_l (L - L_C) from the end of C."""
    L_rem = np.asarray(prm["L"], float) - np.asarray(prm["L_C"], float)
    Tc = tf_corpus_T(prm["pivot"], prm["rot_axis"], prm["theta_deg"], 1.0, prm["L_C"])
    Tc[:3, 3] += float(wl) * L_rem
    a = unit(prm["a_fin"])
    F = np.asarray(prm["F_fin"], float) - (1.0 - float(wl)) * L_rem
    return dict(phase="L", tip_s=float(prm["d_fin"]), d_mm=float(prm["d_fin"]), w_r=1.0, w_l=float(wl),
                tip=F + float(prm["L_iu"]) * a, F=F, tube_axis=a, R_rows=tf_frame(a, prm["a_fin"], prm["x_fin"]),
                T_corpus=Tc, axis_alpha=None, axis_beta=1.0)


def tandem_first_path(prm):
    """Every V, C and L row of the tandem-first schedule from its parameters (pose.json
    insertion_path_tandem_first["params"], written by applicator_venezia.tandem_first_record): V at the knots
    V_tip_s, C at the knots C_knots [[d, w_r], ...] (the host's containment search, hybrid/tf_gate.py), L at L_w.
    The C axis depends on the previous row (least motion), so the rows are generated in order.  Rows: phase, tip_s
    (the tip's arclength along the planned path from O_true = the depth d in C; S1b depth engagement), d_mm, w_r,
    w_l, tip, F, tube_axis, R_rows (rows = applicator x, y, z), T_corpus (4x4 preBT -> current)."""
    rows = [tf_row_V(prm, q) for q in prm["V_tip_s"]]
    a_prev = rows[-1]["tube_axis"] if rows else unit(prm["a_start"])
    for d, w in prm["C_knots"]:
        r = tf_row_C(prm, float(d), float(w), a_prev)
        rows.append(r)
        a_prev = r["tube_axis"]
    rows += [tf_row_L(prm, wl) for wl in prm["L_w"]]
    return rows


# ----------------------------------------------------------------------------- rod / spheres
def rod_project(p, F, a, s_lo, s_hi):
    """Closest point on the segment F + a*s, s in [s_lo, s_hi]; returns (q, s, dist)."""
    p = np.atleast_2d(p); a = unit(a)
    s = np.clip((p - F) @ a, s_lo, s_hi)
    q = F + s[:, None] * a
    return q, s, np.linalg.norm(p - q, axis=1)


def sphere_sdf(p, centers, radii):
    """phi (n, k) = |p - c_k| - r_k."""
    p = np.atleast_2d(p)
    d = np.linalg.norm(p[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
    return d - np.asarray(radii)[None, :]


def smooth_union_sdf(p, centers, radii, k):
    """Soft-min (log-sum-exp, blend k mm) SDF of a union of spheres and its unit gradient (outward normal).
    phi <= min_i(|p-c_i| - r_i); the smooth surface lies outside the hard union by at most k*ln(#overlapping spheres)
    (~0.7 k at a two-sphere crease) and has no creases, so a penalty along n cannot trap or tunnel a point."""
    p = np.atleast_2d(p); C = np.asarray(centers, float); r = np.asarray(radii, float)
    v = p[:, None, :] - C[None]; d = np.maximum(np.linalg.norm(v, axis=2), 1e-12)
    ph = d - r[None]; m = ph.min(1, keepdims=True)
    e = np.exp(-(ph - m) / k); s = e.sum(1, keepdims=True)
    phi = m[:, 0] - k * np.log(s[:, 0])
    g = ((e / s)[:, :, None] * v / d[:, :, None]).sum(1)
    n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return phi, n


def sphere_project(p, c, r):
    """Project points onto the surface of sphere (c, r)."""
    p = np.atleast_2d(p); v = p - c
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1e-12
    return c + v / n * r


# ----------------------------------------------------------------------------- polylines
def arclength(P):
    P = np.asarray(P, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def resample(P, step=1.0):
    """Resample a polyline at uniform arclength spacing (end point always kept)."""
    P = np.asarray(P, float); s = arclength(P)
    n = max(2, int(np.round(s[-1] / step)) + 1)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)


def max_dev_from_chord(P):
    d = unit(P[-1] - P[0]); r = P - P[0]
    return float(np.linalg.norm(r - np.outer(r @ d, d), axis=1).max())


# ----------------------------------------------------------------------------- hexahedra
# 5-tet split of a hexahedron with SOFA's node order
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
HEX5 = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 6, 7, 4], [1, 3, 4, 6]])


def tet_signed_vol(A, B, C, D):
    return np.einsum("...i,...i->...", np.cross(B - A, C - A), D - A) / 6.0


def hexa_volumes(X, hexa):
    """Signed volume per hexa (sum of 5 tets) and the per-hexa minimum tet volume. X (n,3), hexa (m,8)."""
    H = X[np.asarray(hexa)]
    tv = np.stack([tet_signed_vol(H[:, t[0]], H[:, t[1]], H[:, t[2]], H[:, t[3]]) for t in HEX5], 1)
    return tv.sum(1), tv


def hexa_corner_signs(X0, hexa):
    """xi_j in {-1,+1}^3 of each hexa node at rest (axis-aligned cells)."""
    H = X0[np.asarray(hexa)]
    c = H.mean(1, keepdims=True)
    return np.sign(H - c), (H.max(1) - H.min(1))  # (m,8,3), cell size (m,3)


def hexa_center_F(X, X0, hexa, signs=None, size=None):
    """Deformation gradient at each hexa centre (trilinear). dN_j/dX = xi_j / (2 h) * 1/4 per axis."""
    if signs is None:
        signs, size = hexa_corner_signs(X0, hexa)
    H = X[np.asarray(hexa)]                          # (m,8,3)
    G = signs / (4.0 * size[:, None, :])             # (m,8,3) dN/dX  (1/8 * 2/h)
    return np.einsum("mji,mjk->mik", H, G)           # F_ik = sum_j x_ji dN_j/dX_k


def principal_stretches(F):
    C = np.einsum("mki,mkj->mij", F, F)
    ev = np.linalg.eigvalsh(C)
    return np.sqrt(np.clip(ev, 0, None))            # ascending (m,3)


def trilinear_weights(P, X0, hexa, tol=1e-6):
    """For each point, the containing rest hexa (axis-aligned) and its 8 trilinear weights.
    Points outside every cell use the nearest cell with clamped local coords (flag clamped=True).
    Returns cell (n,), nodes (n,8), w (n,8), clamped (n,)."""
    P = np.atleast_2d(P); hexa = np.asarray(hexa)
    H = X0[hexa]; lo = H.min(1); hi = H.max(1)
    cells = np.zeros(len(P), int); W = np.zeros((len(P), 8)); clamped = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        inside = np.all((p >= lo - tol) & (p <= hi + tol), 1)
        if inside.any():
            k = int(np.nonzero(inside)[0][0])
        else:
            d = np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi)), axis=1)
            k = int(np.argmin(d)); clamped[i] = True
        loc = np.clip((p - lo[k]) / (hi[k] - lo[k]), 0, 1)
        corner = (H[k] > (lo[k] + hi[k]) / 2).astype(float)     # (8,3) 0/1 per axis
        W[i] = np.prod(np.where(corner > 0.5, loc[None, :], 1 - loc[None, :]), axis=1)
        cells[i] = k
    return cells, hexa[cells], W, clamped


def mesh_volume(V, F):
    V = np.asarray(V, float); F = np.asarray(F, int)
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def read_obj(p):
    V, F = [], []
    with open(p) as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in ln.split()[1:4]])
    return np.array(V, float), np.array(F, int)


def write_obj(p, V, F, header=None):
    with open(p, "w") as fh:
        if header:
            for h in header.splitlines():
                fh.write("# %s\n" % h)
        for v in V:
            fh.write("v %.4f %.4f %.4f\n" % tuple(v))
        for f in F:
            fh.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))
