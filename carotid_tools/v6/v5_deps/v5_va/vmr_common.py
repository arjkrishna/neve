# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_common.py (= /sp/v5_va/vmr_common.py in the build containers)
"""Shared helpers for the vmr agent (VMR 0248 VA from the CT and from the lofted walls).

RAS mm throughout (sources.load_vmr0248 frame). CT array is [k,j,i] int16 HU.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

sys.path.insert(0, "/sp/v5_va")

DERIVED = "/repo/carotid_data/va_sources/derived/vmr0248"
RES = "/sp/v5_va/results"
WORK = "/sp/v5_va/results/vmr"
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def arclen(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample(P, step, *arrs):
    s = arclen(P)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    out = [np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)]
    for a in arrs:
        out.append(np.interp(q, s, a))
    return (q,) + tuple(out)


def smooth(P, sigma_pts):
    return ndimage.gaussian_filter1d(np.asarray(P, float), sigma_pts, axis=0, mode="nearest")


def rmf(P):
    """Tangents + rotation-minimising frame (double reflection)."""
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    a = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    U = np.zeros_like(P)
    U[0] = np.cross(T[0], a)
    U[0] /= np.linalg.norm(U[0])
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-12:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2 / c1) * (v1 @ U[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        U[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-12 else rL
        U[i + 1] -= (U[i + 1] @ T[i + 1]) * T[i + 1]
        U[i + 1] /= np.linalg.norm(U[i + 1])
    V = np.cross(T, U)
    return T, U, V


class CT:
    def __init__(self, ct):
        self.arr = ct["array"]
        self.sp = np.asarray(ct["spacing_mm"], float)
        self.org = np.asarray(ct["origin_mm"], float)
        self.dims = np.asarray(ct["dims_ijk"])

    def ijk(self, P):
        return (np.asarray(P, float) - self.org) / self.sp

    def sample(self, P, order=1, cval=-1024, arr=None):
        a = self.arr if arr is None else arr
        q = self.ijk(P)
        sh = q.shape[:-1]
        q = q.reshape(-1, 3)
        v = map_coordinates(a, [q[:, 2], q[:, 1], q[:, 0]], order=order, mode="constant", cval=cval)
        return v.reshape(sh)

    def world(self, kji):
        kji = np.asarray(kji, float)
        return self.org + kji[..., ::-1] * self.sp


def load_vmr(with_ct=True):
    from sources import load_vmr0248
    return load_vmr0248(with_ct=with_ct, with_model=True, with_walls=True)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def write_poly(mesh, path):
    mesh.save(path)


STEP = 0.25


def _stations(points, radii, sigma_mm=1.0):
    q, P, R = resample(np.asarray(points, float), STEP, np.asarray(radii, float))
    P = smooth(P, sigma_mm / STEP)
    T, _, _ = rmf(P)
    return {"s": q, "P": P, "R": R, "T": T}


def build_stations(vmr):
    """rva / lva (origin -> merge), ba (merge -> right-PCA origin), rsub / lsub (+-30 mm around the VA origin)."""
    V = vmr["vessels"]
    ves = {"rva": _stations(V["right_vertebral"]["points"], V["right_vertebral"]["radius_mm"]),
           "lva": _stations(V["left_vertebral"]["points"], V["left_vertebral"]["radius_mm"]),
           "ba": _stations(V["basilar"]["points"], V["basilar"]["radius_mm"])}
    for side, key in (("right", "rsub"), ("left", "lsub")):
        sub = V[side + "_subclavian"]
        sp, sr = np.asarray(sub["points"]), np.asarray(sub["radius_mm"])
        o = ves[side[0] + "va"]["P"][0]
        k = int(np.argmin(np.linalg.norm(sp - o, axis=1)))
        ss = arclen(sp)
        m = np.abs(ss - ss[k]) <= 30.0
        ves[key] = _stations(sp[m], sr[m])
    return ves
