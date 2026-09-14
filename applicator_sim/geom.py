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
