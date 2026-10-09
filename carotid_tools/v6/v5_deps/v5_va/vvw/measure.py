#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vvw\measure.py (= /sp/v5_va/vvw/measure.py in the build containers)
"""verify_v5_wall: OWN implementation of right_route_detail's definition, applied per VA stretch and to the RCCA
route, on the UNDECIMATED union (iso_v5/<name>.vtp), with k=24 ownership (as v4's function) and EXACT ownership
(argmax over ALL samples of r - |x - c|, no k truncation). Also the OBJ / collision_full (v5) and v4's OBJ (report).

Written from the definition (bake_meshes_v4.right_route_detail docstring), not importing va_metrics / bake code:
  * own markups reader ((y, -z, -x), float32 as the loader), own 0.25 mm resampling of every branch;
  * triangle centroids / areas from the raw faces;
  * owner = argmax_j (r_j - |x - c_j|); dep = | |x - c*| - r* |; frac = area(dep > t) / area.
Segments per donor VA (branch arclength; 'core' trims 2 mm at both ends, as the bake's measure):
  donor_stretch [lo, cut] core; donor_real = donor_stretch minus the 25 mm seam zone [cut-25, cut] (study / v4
  ramp rule: ramp on the donor side) minus the tube windows (mesh_v5 suppressed_stretches_mm[vaC_*]);
  seam [cut-25, cut] (no trim); windows (no trim); topbrain [cut, L] core; whole VA from the ostium raw and
  conservative (seam dep counted 0, the study's convention); whole VA from the origin (branch point of (17)/(19)) raw.
Extra (adversarial): signed deviation, share of large protrusions (> 1.0 / 1.5 mm), and relief against a FITTED
tube (area-weighted median |x-c| per 1 mm bin, smoothed sigma 5 mm) -- calibre offsets do not count there.

    python3 measure.py <name>     (neve-build-meshlab; writes /sp/v5_va/vvw/out/<name>.json, skip if present)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

A5 = "/repo/carotid_data/anatomies_v5"
V4 = "/repo/carotid_data/anatomies_v4"
ISO = "/repo/carotid_data/va_sources/v5_reg/iso_v5"
OUT = "/sp/v5_va/vvw/out"
STEP, SEAM, TRIM = 0.25, 25.0, 2.0


# ------------------------------------------------------------------ inputs (own readers)
def read_branches(folder):
    out = {}
    for f in sorted(os.listdir(folder)):
        if not (f.startswith("Centerline curve") and f.endswith(".json")):
            continue
        d = json.load(open(os.path.join(folder, f), encoding="utf-8"))
        P, R = [], []
        for mk in d["markups"]:
            if mk["type"] != "Curve":
                continue
            for cp in mk["controlPoints"]:
                x, y, z = (float(v) for v in cp["position"])
                P.append((y, -z, -x))
            for ms in mk.get("measurements", []):
                if ms["name"] == "Radius":
                    R.extend(ms["controlPointValues"])
        tag = f.replace("Centerline curve", "").replace(".mrk.json", "").strip(" -")
        out[tag] = (np.asarray(np.asarray(P, np.float32), float), np.asarray(np.asarray(R, np.float32), float))
    return out


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


class Samp:
    def __init__(self, br):
        P, R, B, S = [], [], [], []
        self.tags = []
        for i, (tag, (p, r)) in enumerate(br.items()):
            s = arclen(p)
            if s[-1] >= STEP:
                q = np.arange(0.0, s[-1] + 1e-9, STEP)
                p = np.stack([np.interp(q, s, p[:, k]) for k in range(3)], 1)
                r = np.interp(q, s, r)
                s = q
            P.append(p); R.append(r); B.append(np.full(len(p), i)); S.append(s)
            self.tags.append(tag)
        self.p, self.r = np.vstack(P), np.concatenate(R)
        self.b, self.s = np.concatenate(B), np.concatenate(S)
        self.tree = cKDTree(self.p)
        self.rmax = float(self.r.max())

    def idx(self, tag):
        return self.tags.index(tag)


def owners(samp, X, mode, chunk=150000):
    """best sample per point: 'k24' = argmax over the 24 nearest (v4's function); 'exact' = over all samples."""
    best = np.empty(len(X), np.int64)
    for a in range(0, len(X), chunk):
        x = X[a:a + chunk]
        d, i = samp.tree.query(x, k=24)
        f = samp.r[i] - d
        j = np.argmax(f, axis=1)
        bb = i[np.arange(len(x)), j]
        if mode == "exact":
            fb = f[np.arange(len(x)), j]
            todo = np.nonzero(fb < samp.rmax - d[:, -1])[0]
            K = 24
            while len(todo):
                K = min(K * 4, len(samp.p))
                d2, i2 = samp.tree.query(x[todo], k=K)
                f2 = samp.r[i2] - d2
                j2 = np.argmax(f2, axis=1)
                bb[todo] = i2[np.arange(len(todo)), j2]
                fb2 = f2[np.arange(len(todo)), j2]
                if K >= len(samp.p):
                    break
                todo = todo[fb2 < samp.rmax - d2[:, -1]]
        best[a:a + chunk] = bb
    return best


def tri_geom(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return (a + b + c) / 3.0, 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def read_vtp(path):
    import pyvista as pv
    m = pv.read(path)
    fa = np.asarray(m.faces)
    if not (len(fa) % 4 == 0 and (fa.reshape(-1, 4)[:, 0] == 3).all()):
        fa = np.asarray(m.triangulate().faces)
    return np.asarray(m.points, float), fa.reshape(-1, 4)[:, 1:].astype(np.int64)


def read_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, np.int64)


# ------------------------------------------------------------------ measure
class Own:
    def __init__(self, samp, X, A, mode):
        self.samp, self.A = samp, A
        best = owners(samp, X, mode)
        self.best = best
        self.b, self.s, self.r = samp.b[best], samp.s[best], samp.r[best]
        self.rad = np.linalg.norm(X - samp.p[best], axis=1)
        self.sdep = self.rad - self.r
        self.dep = np.abs(self.sdep)

    def mask(self, bi, lo, hi, core=True, excl=()):
        Lb = float(self.samp.s[self.samp.b == bi].max())
        e = TRIM if core else 0.0
        m = (self.b == bi) & (self.s >= lo + e) & (self.s <= min(hi, Lb) - e)
        for a, z in excl:
            m &= ~((self.s >= a) & (self.s <= z))
        return m

    def frac(self, m, dep=None, extra=False):
        if not m.any():
            return None
        dep = self.dep if dep is None else dep
        a, x, r = self.A[m], dep[m], self.r[m]
        tot = float(a.sum())
        d = {"area_mm2": round(tot, 2), "cells": int(m.sum()),
             "frac_0.3mm": float(a[x > 0.3].sum()) / tot, "frac_0.5mm": float(a[x > 0.5].sum()) / tot,
             "frac_rel_0.15": float(a[x > 0.15 * r].sum()) / tot}
        if extra:
            sd = self.sdep[m]
            o = np.argsort(sd)
            cw = np.cumsum(a[o])
            d["signed_dep_wmedian_mm"] = float(sd[o][np.searchsorted(cw, 0.5 * cw[-1])])
            d["frac_out_0.3mm"] = float(a[sd > 0.3].sum()) / tot
            d["frac_in_0.3mm"] = float(a[sd < -0.3].sum()) / tot
            d["frac_1.0mm"] = float(a[x > 1.0].sum()) / tot
            d["frac_1.5mm"] = float(a[x > 1.5].sum()) / tot
            d["dep_p99_mm"] = float(np.percentile(x, 99))
            # fitted plain tube: area-weighted median |x - c| per 1 mm, smoothed sigma 5 mm
            s = self.s[m]
            rad = self.rad[m]
            s0 = np.floor(s.min())
            k = np.floor(s - s0).astype(int)
            nb = k.max() + 1
            rf = np.full(nb, np.nan)
            for q in range(nb):
                w = k == q
                if w.sum() >= 3:
                    oo = np.argsort(rad[w])
                    cc = np.cumsum(a[w][oo])
                    rf[q] = rad[w][oo][np.searchsorted(cc, 0.5 * cc[-1])]
            ok = ~np.isnan(rf)
            if ok.sum() >= 2:
                rf = np.interp(np.arange(nb), np.nonzero(ok)[0], rf[ok])
                rf = gaussian_filter1d(rf, 5.0, mode="nearest")
                df = np.abs(rad - rf[k])
                d["fit_frac_0.3mm"] = float(a[df > 0.3].sum()) / tot
                d["fit_frac_0.5mm"] = float(a[df > 0.5].sum()) / tot
        return d

    def seg(self, bi, lo, hi, core=True, excl=(), extra=False, dep=None):
        if hi - lo < (2 * TRIM + 1.0 if core else 0.5):
            return None
        r = self.frac(self.mask(bi, lo, hi, core, excl), dep=dep, extra=extra)
        if r:
            r["range_mm"] = [round(lo, 2), round(hi, 2)]
        return r


def rrd(O, samp, cut):
    rc = samp.idx("RCCA")
    own = (O.b == rc) & (O.s >= cut)
    out = {}
    for seg, m in (("fork", own & (O.s < 130.0)), ("siphon", own & (O.s >= 130.0))):
        a, x = O.A[m], O.dep[m]
        tot = float(a.sum()) or 1.0
        out[seg] = {"frac_0.3mm": float(a[x > 0.3].sum()) / tot, "frac_0.5mm": float(a[x > 0.5].sum()) / tot,
                    "area_mm2": float(a.sum()), "cells": int(m.sum())}
    return out


def r4(d):
    return {g: {t: round(d[g][t], 4) for t in ("frac_0.3mm", "frac_0.5mm")} for g in d}


def origin_s(br, tag, sub):
    P, _ = br[tag]
    q = br[sub][0][0]
    return float(arclen(P)[int(np.argmin(np.linalg.norm(P - q, axis=1)))])


def va_sides(O, samp, br, prov, rep, extra=True, modes=("all",)):
    x, v5 = prov["xform"], prov.get("v5_va") or {}
    sup = rep.get("suppressed_stretches_mm") or {}
    out = {}
    for side, tag, sub, ck, tk in (("right", "RVA", "(17)", "vaC_R", "va_R"), ("left", "LVA", "(19)", "vaC_L", "va_L")):
        bi = samp.idx(tag)
        L = float(arclen(br[tag][0])[-1])
        if ck not in x:
            out[side] = {"donor": None}
            continue
        lo, hi = float(x[ck]["lo_mm"]), float(x[ck]["hi_mm"])
        cut = float(x[tk]["cut_mm"])
        ost = float((v5.get(side) or {}).get("ostium_at_mm", lo))
        ramp = ((v5.get(side) or {}).get("ramp") or {}).get("side")
        seam = (cut - SEAM, cut) if ramp == "donor" else (cut, min(cut + SEAM, L))
        win = [tuple(w) for w in sup.get(ck, [])]
        tbw = [tuple(w) for w in sup.get(tk, [])]
        o_s = origin_s(br, tag, sub)
        seam_m = O.mask(bi, seam[0], seam[1], core=False)
        dep_cons = np.where(seam_m, 0.0, O.dep)
        d = {"donor": "%s %s" % (x[ck]["donor_id"], x[ck]["donor_side"]), "lo": lo, "hi": hi, "cut": cut, "ost": ost,
             "L": L, "origin_s": o_s, "ramp": ramp, "seam": list(seam), "windows": win, "tb_windows": tbw,
             "donor_stretch": O.seg(bi, lo, cut),
             "donor_real": O.seg(bi, lo, cut, excl=[seam] + win, extra=extra),
             "seam_zone": O.seg(bi, seam[0], seam[1], core=False),
             "windows_only": O.frac(np.logical_or.reduce([O.mask(bi, a, z, core=False) for a, z in win])) if win else None,
             "topbrain": O.seg(bi, cut, L, extra=extra),
             "topbrain_real": O.seg(bi, seam[1] if ramp != "donor" else cut, L, excl=tbw),
             "whole_ostium_raw": O.seg(bi, ost, L),
             "whole_ostium_cons": O.seg(bi, ost, L, dep=dep_cons),
             "whole_origin_raw": O.seg(bi, o_s, L),
             "junction_stub": O.seg(bi, o_s, ost, core=False)}
        out[side] = d
    return out


def v4_sides(O, samp, br, prov4, rep4):
    """v4 OBJ: host tube [origin, cut4 - 25] and TopBrain [cut4, L] per VA (report)."""
    x = prov4["xform"]
    out = {}
    for side, tag, sub, tk in (("right", "RVA", "(17)", "va_R"), ("left", "LVA", "(19)", "va_L")):
        bi = samp.idx(tag)
        L = float(arclen(br[tag][0])[-1])
        cut = float(x[tk]["cut_mm"])
        o_s = origin_s(br, tag, sub)
        out[side] = {"cut": cut, "origin_s": o_s, "host_tube": O.seg(bi, o_s, cut - SEAM),
                     "topbrain": O.seg(bi, cut, L)}
    return out


def vertex_area(V, F, A):
    w = np.zeros(len(V))
    for k in range(3):
        np.add.at(w, F[:, k], A / 3.0)
    return w


def mesh_block(V, F, samp, br, prov, rep, v4=False):
    C, A = tri_geom(V, F)
    O = Own(samp, C, A, "exact")
    res = {"tris": int(len(F)),
           "cent": (v4_sides(O, samp, br, prov, rep) if v4 else va_sides(O, samp, br, prov, rep, extra=False))}
    Ov = Own(samp, V, vertex_area(V, F, A), "exact")
    res["vert"] = v4_sides(Ov, samp, br, prov, rep) if v4 else va_sides(Ov, samp, br, prov, rep, extra=False)
    return res


def main():
    name = sys.argv[1]
    dst = os.path.join(OUT, name + ".json")
    if os.path.exists(dst) and "--force" not in sys.argv:
        print("done already", dst)
        return 0
    t0 = time.time()
    root = os.path.join(A5, name)
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    rep = json.load(open(os.path.join(root, "mesh_v5.json"), encoding="utf-8"))
    rep4 = json.load(open(os.path.join(V4, name, "mesh_v4.json"), encoding="utf-8"))
    prov4 = json.load(open(os.path.join(V4, name, "provenance.json"), encoding="utf-8"))
    br = read_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samp(br)
    V, F = read_vtp(os.path.join(ISO, name + ".vtp"))
    C, A = tri_geom(V, F)
    rec = {"name": name, "union_tris": int(len(F)), "union_area_mm2": float(A.sum())}
    cut = float(prov.get("host_cut_mm", 0.0))
    owns = {}
    for mode in ("k24", "exact"):
        t = time.time()
        O = Own(samp, C, A, mode)
        owns[mode] = O
        rr = rrd(O, samp, cut)
        rec[mode] = {"rrd": rr, "rrd_r4": r4(rr), "va": va_sides(O, samp, br, prov, rep), "seconds": round(time.time() - t, 1)}
    k, e = owns["k24"], owns["exact"]
    rec["rrd_k24_equals_mesh_v5"] = rec["k24"]["rrd_r4"] == rep["right_route_detail"]
    rec["rrd_k24_equals_mesh_v4"] = rec["k24"]["rrd_r4"] == rep4["right_route_detail"]
    rec["rrd_exact_equals_mesh_v4"] = rec["exact"]["rrd_r4"] == rep4["right_route_detail"]
    rec["rrd_mesh_v5"], rec["rrd_mesh_v4"] = rep["right_route_detail"], rep4["right_route_detail"]
    # ownership changes k24 -> exact, per VA branch
    ch = {}
    for tag in ("RVA", "LVA", "RCCA"):
        bi = samp.idx(tag)
        a_k, a_e = k.b == bi, e.b == bi
        ch[tag] = {"area_k24_only": float(A[a_k & ~a_e].sum()), "area_exact_only": float(A[a_e & ~a_k].sum()),
                   "area_k24": float(A[a_k].sum()), "lost_to": {samp.tags[j]: round(float(A[a_k & ~a_e & (e.b == j)].sum()), 2)
                                                               for j in np.unique(e.b[a_k & ~a_e])}}
    rec["ownership_k24_vs_exact"] = ch
    del owns, k, e, O, C, A, V, F
    # meshes (report): v5 OBJ, v5 collision_full, v4 OBJ
    t = time.time()
    Vo, Fo = read_obj(os.path.join(root, "vessel_architecture_collision.obj"))
    rec["obj"] = mesh_block(Vo, Fo, samp, br, prov, rep)
    Vf, Ff = read_vtp(os.path.join(root, "collision_full.vtp"))
    rec["full"] = mesh_block(Vf, Ff, samp, br, prov, rep)
    br4 = read_branches(os.path.join(V4, name, "Centrelines_comb"))
    samp4 = Samp(br4)
    V4o, F4o = read_obj(os.path.join(V4, name, "vessel_architecture_collision.obj"))
    rec["v4_obj"] = mesh_block(V4o, F4o, samp4, br4, prov4, rep4, v4=True)
    rec["meshes_seconds"] = round(time.time() - t, 1)
    rec["seconds"] = round(time.time() - t0, 1)
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            rec["peak_rss_gb"] = round(int(line.split()[1]) / 1048576.0, 3)
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(tmp, dst)
    print("DONE %s %.0f s, peak %s GB, rrd k24==v5rec %s, ==v4rec %s" % (
        name, rec["seconds"], rec.get("peak_rss_gb"), rec["rrd_k24_equals_mesh_v5"], rec["rrd_k24_equals_mesh_v4"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
