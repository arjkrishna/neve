import json, math, os, sys, glob
import numpy as np

def load_cl(path):
    d = json.load(open(path))
    m = d["markups"][0]
    P = np.array([c["position"] for c in m["controlPoints"]], float)
    r = None
    for meas in m.get("measurements", []):
        if meas.get("name") == "Radius" and meas.get("controlPointValues"):
            r = np.array(meas["controlPointValues"], float)
    return P, r

def arclen(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def resample(P, ds=0.5):
    s = arclen(P); tot = s[-1]
    q = np.arange(0, tot, ds)
    out = np.stack([np.interp(q, s, P[:, i]) for i in range(3)], axis=1)
    return q, out

def curvature(P, ds=0.5, win=2.0):
    """Discrete curvature via circumscribed circle over +/- win mm."""
    q, R = resample(P, ds)
    k = int(round(win / ds))
    kap = np.full(len(q), np.nan)
    for i in range(k, len(q) - k):
        a, b, c = R[i - k], R[i], R[i + k]
        v1 = a - b; v2 = c - b
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2); n3 = np.linalg.norm(a - c)
        if n1 < 1e-9 or n2 < 1e-9 or n3 < 1e-9: continue
        # area via cross
        A = 0.5 * np.linalg.norm(np.cross(v1, v2))
        if A < 1e-12: kap[i] = 0.0; continue
        Rc = (n1 * n2 * n3) / (4 * A)
        kap[i] = 1.0 / Rc
    return q, kap

def tangent_turn(P, ds=1.0, base=5.0):
    """Angle in deg between chords of length `base` centred at each station."""
    q, R = resample(P, ds)
    k = int(round(base / ds))
    ang = np.full(len(q), np.nan)
    for i in range(k, len(q) - k):
        t1 = R[i] - R[i - k]; t2 = R[i + k] - R[i]
        n1 = np.linalg.norm(t1); n2 = np.linalg.norm(t2)
        if n1 < 1e-9 or n2 < 1e-9: continue
        c = np.clip(np.dot(t1, t2) / (n1 * n2), -1, 1)
        ang[i] = math.degrees(math.acos(c))
    return q, ang
