"""CELL 1: geometric derivation of the HOST offset  OFF = path_len - s_RCCA(target).
Reuses monitoring/attack1_host.py's centerline loader/rotation verbatim.
For the TOPBRAIN cohort the same construction is known to give 33.31; here we
recompute it independently for the HOST run.
"""
import os, re, glob, json
import numpy as np
ROOT = r"D:/Arjun/workspace/neve"
HOSTCL = os.path.join(ROOT, "eve_bench/data/dualdevicenav/Centrelines_comb")

def load(p):
    d = json.load(open(p)); pts = []
    for m in d["markups"]:
        if m["type"] != "Curve": continue
        for cp in m["controlPoints"]:
            x, y, z = cp["position"]; pts.append((y, -z, -x))
    return np.array(pts, float)

def arc(p): return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])

def rot(zx=(20., 5.)):
    rz = -zx[0]*np.pi/180; rx = -zx[1]*np.pi/180
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    return Rz @ Rx
M = rot()
poly = load(os.path.join(HOSTCL, "Centerline curve - RCCA.mrk.json")); cum = arc(poly)

def proj1(p):
    A = poly[:-1]; B = poly[1:]; AB = B - A; L2 = (AB*AB).sum(1)
    t = np.clip(((p-A)*AB).sum(1)/np.maximum(L2, 1e-12), 0, 1)
    dd = np.linalg.norm(A + t[:, None]*AB - p, axis=1); k = int(np.argmin(dd))
    return cum[k] + t[k]*float(np.sqrt(L2[k])), float(dd[k])

ES = re.compile(r"EPISODE_START .*?pid=(\d+) \| target=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\).*?seed=(\d+)")
PJ = re.compile(r"path_len=([0-9.]+)")

def run(tag, L):
    rows = []
    for f in sorted(glob.glob(os.path.join(L, "worker_*.log"))):
        live = {}
        for line in open(f, errors="replace"):
            m = ES.search(line)
            if m:
                live[m.group(1)] = dict(seed=int(m.group(5)),
                    tgt=np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))]), pl=None)
                continue
            if " STEP |" not in line: continue
            i = line.find("pid="); pid = line[i+4:].split(" ")[0].strip()
            st = live.get(pid)
            if st is None or st["pl"] is not None: continue
            q = PJ.search(line)
            if q:
                st["pl"] = float(q.group(1))
                s, d = proj1((M.T @ st["tgt"].T).T)
                rows.append((st["seed"], s, d, st["pl"]))
    off = np.array([r[3]-r[1] for r in rows]); dd = np.array([r[2] for r in rows])
    print("%s n=%d" % (tag, len(rows)))
    print("   target->RCCA-centerline residual dist: med=%.2f mm p90=%.2f max=%.2f" % (np.median(dd), np.percentile(dd, 90), dd.max()))
    print("   OFF = path_len - s_RCCA(target): med=%.3f mean=%.3f sd=%.3f min=%.3f max=%.3f" % (
        np.median(off), off.mean(), off.std(), off.min(), off.max()))
    print("   target s_RCCA: min=%.1f med=%.1f max=%.1f ; path_len med=%.1f" % (
        min(r[1] for r in rows), np.median([r[1] for r in rows]), max(r[1] for r in rows),
        np.median([r[3] for r in rows])))

if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        t, L = a.split("=", 1); run(t, L)
