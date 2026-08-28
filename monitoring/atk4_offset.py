import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from atk4_geom import load_cl, arclen

def kabsch_rmsd(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Ar = (R @ (A - ca).T).T + cb
    return float(np.sqrt(((Ar - B) ** 2).sum(1).mean()))

def solve_offset(cl_path, targets, pls, lo=0.0, hi=90.0):
    P, _ = load_cl(cl_path)
    s = arclen(P)
    best = (1e9, None)
    for c in np.arange(lo, hi, 0.25):
        q = np.array(pls) - c
        if q.min() < 0 or q.max() > s[-1]: continue
        B = np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)
        r = kabsch_rmsd(np.array(targets), B)
        if r < best[0]: best = (r, c)
    # refine
    if best[1] is not None:
        for c in np.arange(best[1] - 0.5, best[1] + 0.5, 0.02):
            q = np.array(pls) - c
            if q.min() < 0 or q.max() > s[-1]: continue
            B = np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)
            r = kabsch_rmsd(np.array(targets), B)
            if r < best[0]: best = (r, c)
    return best[1], best[0], s[-1]
