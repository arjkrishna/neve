#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_v5c.py (= /sp/v5_va/bake_v5c.py in the build containers)
"""bake_v5c: bake_v5b + ROUTE-PRIORITY OWNERSHIP (v5_route13 fallback, 2026-10-02).

Where moving the VA cannot give the right route back its v4 wall, the RCCA route sections win the contested voxels
against the VA sections. Installed at run time (bake_meshes_v4, sdf_union_v4 and bake_v5b are NOT edited): the
ownership function bake_v5b's UnionCache calls (bake_v5b.ownership_margin) is replaced by own_rp below, and
bake_meshes_v4.bake_one by a wrapper that tells it the anatomy's carotid margin.

Stated diffs against sdf_union_v4.ownership_margin (margin = f_local - max(f_far, f_other); owned iff >= m):
 R1  RCCA ROUTE SECTIONS (sections on branch RCCA: lower_R = the fork / bulb slice, siphon_R = the TopBrain siphon):
     the two VA branches (RVA, LVA) are taken out of f_other, and replaced by a TUBE-FLOOR term:
         margin_R = min(f_local - max(f_far, f_other without the VAs),  -f_VA)        f_VA = max(f_RVA, f_LVA)
     i.e. the RCCA real wall may grow up to the VA's tube wall less the margin (-f_VA = |x - c| - r >= m), never into
     it: the VA's tube floor stays, with the same >= m band between the two surfaces the rule always leaves.
 R2  VA SECTIONS (va_R / va_L TopBrain pieces, vaC_R / vaC_L donors): the RCCA route real wall competes as if it
     were the RCCA's tube:
         margin_VA = min(the rule's margin,  f_local - f_Rreal(x))
     f_Rreal(x) = f_sec of an RCCA route section at x where R1 owns x (static: before suppression and before the
     field comparison, so it never shrinks below what the union applies and the cached VA margins stay valid);
     -inf elsewhere. The VA's real wall is therefore refused where the RCCA real wall is nearer (the VA tube floor
     stays: it is the tube field, untouched).
 R3  every other section (RECA, the left side) and every other rule, ladder step, gate, decimation and record: bake_v5b
     / bake_meshes_v4 unchanged. mesh_v5.json / provenance bake_v5 carry "ownership": "route_priority (bake_v5c R1-R3)".
--selftest <root>: own_rp with the route priority switched OFF equals sdf_union_v4.ownership_margin exactly.

    python3 /sp/v5_va/bake_v5c.py --anatomies <dir> --only NAME --dump-iso <dir>        (neve-build-meshlab)
    (bake_full/bake_full_one.py and measure_full_one.py work unchanged with BAKE_MODULE=bake_v5c, see route13 run)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
import sdf_union_v4 as U  # noqa: E402

VERSION = "bake_v5c 1.0 (2026-10-02; v5_route13 route-priority ownership R1-R3 on bake_v5b)"
ROUTE_BRANCH = "RCCA"
VA_BRANCHES = ("RVA", "LVA")
CTX = {"on": True, "m_car": None, "rreal": None, "offset": None, "shape": None, "spacing": None, "stats": {}}


def _parts(coords, sec, samp):
    """Every term of sdf_union_v4.ownership_margin, the two VAs kept apart (copied; same K, WINDOW, siblings)."""
    bi = samp.index_of(sec.branch)
    own = samp.b == bi
    op, orad, os_ = samp.p[own], samp.r[own], samp.s[own]
    f_o, nn = U._tube(coords, op, orad, U.K_OWN)
    s_nn = os_[nn]
    best = np.argmax(f_o, axis=1)
    s_star = s_nn[np.arange(len(coords)), best]
    near = np.abs(s_nn - s_star[:, None]) <= U.WINDOW_MM
    f_loc_all = np.where(near, f_o, -np.inf)
    f_local = f_loc_all.max(axis=1)
    anchor = op[nn[np.arange(len(coords)), np.argmax(f_loc_all, axis=1)]]
    f_far = np.where(~near, f_o, -np.inf).max(axis=1)
    va_ids = set()
    for t in VA_BRANCHES:
        try:
            va_ids.add(samp.index_of(t))
        except KeyError:
            pass
    f_other = np.full(len(coords), -np.inf)
    f_va = np.full(len(coords), -np.inf)
    for j in np.unique(samp.b[~own]):
        if any(sb in samp.names[j] for sb in getattr(sec, "siblings", ())):
            continue
        m = samp.b == j
        f_x, _ = U._tube(coords, samp.p[m], samp.r[m], U.K_OTHER)
        if j in va_ids:
            f_va = np.maximum(f_va, f_x.max(axis=1))
        else:
            f_other = np.maximum(f_other, f_x.max(axis=1))
    inside = (s_star >= sec.s0 - U.WINDOW_MM) & (s_star <= sec.s0 + U._length(sec.points) + U.WINDOW_MM)
    return f_local, f_far, f_other, f_va, anchor, s_star, inside


def _rreal_at(coords):
    R = CTX["rreal"]
    out = np.full(len(coords), -np.inf)
    if R is None or not len(R[0]):
        return out
    g = np.round((coords - CTX["offset"]) / CTX["spacing"]).astype(np.int64)
    ok = np.all((g >= 0) & (g < np.array(CTX["shape"])), axis=1)
    flat = np.full(len(coords), -1, np.int64)
    flat[ok] = np.ravel_multi_index(g[ok].T, CTX["shape"])
    k = np.searchsorted(R[0], flat)
    k = np.clip(k, 0, len(R[0]) - 1)
    hit = ok & (R[0][k] == flat)
    out[hit] = R[1][k[hit]]
    return out


def own_rp(coords, sec, samp, return_anchor=False):
    f_local, f_far, f_other, f_va, anchor, s_star, inside = _parts(coords, sec, samp)
    std = f_local - np.maximum(f_far, np.maximum(f_other, f_va))
    if not CTX["on"]:
        margin = std
    elif sec.branch == ROUTE_BRANCH:                                   # R1
        margin = np.minimum(f_local - np.maximum(f_far, f_other), -f_va)
    elif sec.branch in VA_BRANCHES:                                     # R2
        margin = np.minimum(std, f_local - _rreal_at(coords))
    else:                                                               # R3
        margin = std
    margin = np.where(inside, margin, -np.inf)
    return (margin, anchor, s_star) if return_anchor else margin


# ---- install: the cache computes the static RCCA real wall (R2) once per bake_one ------------------------------------
_ORIG_INIT = V.UnionCache.__init__


def _init_rp(self, br, samp, secs, spacing, verbose=False):
    _ORIG_INIT(self, br, samp, secs, spacing, verbose)
    CTX.update(offset=np.asarray(self.field.offset, float), shape=tuple(self.field.values.shape), spacing=spacing,
               rreal=None)
    if not CTX["on"]:
        return
    m = CTX["m_car"] if CTX["m_car"] is not None else U.MARGIN_MM
    flats, vals = [], []
    for sec in secs:
        if sec.branch != ROUTE_BRANCH:
            continue
        P = self.pre.get(sec.name)
        if P is None:
            continue
        coords = self._coords(P, np.ones(P["n"], bool))
        mg = own_rp(coords, sec, samp)
        ok = mg >= m
        flats.append(np.ravel_multi_index(P["gidx"][ok].astype(np.int64).T, CTX["shape"]))
        vals.append(P["f_sec"][ok].astype(float))
        CTX["stats"][sec.name] = {"voxels": int(P["n"]), "route_owned": int(ok.sum())}
    if flats:
        f = np.concatenate(flats)
        v = np.concatenate(vals)
        o = np.lexsort((-v, f))                                         # per voxel keep the largest f_sec
        f, v = f[o], v[o]
        first = np.r_[True, f[1:] != f[:-1]]
        CTX["rreal"] = (f[first], v[first])


V.UnionCache.__init__ = _init_rp
V.ownership_margin = own_rp

_BAKE_ONE_V5B = V.bake_one_v5b


def bake_one_v5c(root, spacing, full_tris, obj_tris, verbose, dump_iso=None, far_clamp=False, far_clamp_why=None):
    name = os.path.basename(os.path.normpath(root))
    rep4 = json.load(open(os.path.join(V.V4_DIR, name, B.REPORT), encoding="utf-8"))
    CTX["m_car"] = rep4["margin_mm"]
    CTX["stats"] = {}
    rep = _BAKE_ONE_V5B(root, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp, far_clamp_why)
    rep["ownership"] = "route_priority (bake_v5c R1-R3)" if CTX["on"] else "standard"
    rep["bake_v5c"] = {"version": VERSION, "route_priority": CTX["on"], "carotid_margin_mm": CTX["m_car"],
                       "route_sections_static": CTX["stats"]}
    with open(os.path.join(root, V.REPORT_V5), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


B.bake_one = bake_one_v5c
_REC_PROV = V.record_provenance


def record_provenance_v5c(root, rep):
    _REC_PROV(root, rep)
    p = os.path.join(root, "provenance.json")
    prov = json.load(open(p, encoding="utf-8"))
    prov["bake_v5"]["script"] = "v5_va/bake_v5c.py (bake_v5b + route-priority ownership R1-R3)"
    prov["bake_v5"]["ownership"] = rep.get("ownership")
    prov["bake_v5"]["bake_v5c"] = rep.get("bake_v5c")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    os.replace(p + ".tmp", p)


V.record_provenance = record_provenance_v5c


def selftest(root):
    """own_rp with the priority OFF == sdf_union_v4.ownership_margin, on every section's would-be voxels (sampled)."""
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = U.Samples(br)
    secs, _ = B.sections_for(root, br)
    rng = np.random.default_rng(0)
    out = []
    CTX["on"] = False
    for sec in secs:
        c = sec.points[rng.integers(0, len(sec.points), 3000)] + rng.normal(0, 3.0, (3000, 3))
        a = U.ownership_margin(c, sec, samp, return_anchor=True)
        b = own_rp(c, sec, samp, return_anchor=True)
        same = all(np.array_equal(x, y) for x, y in zip(a, b))
        out.append({"section": sec.name, "identical": bool(same)})
        print("SELFTEST %s identical %s" % (sec.name, same), flush=True)
    CTX["on"] = True
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        r = selftest(sys.argv[2])
        sys.exit(0 if all(x["identical"] for x in r) else 1)
    sys.exit(V.main())
