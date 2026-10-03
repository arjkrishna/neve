#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_v5b.py (= /sp/v5_va/bake_v5b.py in the build containers)
"""bake_v5b: bake_v5 (bake_meshes_v4 UNCHANGED + the donor vaC_R / vaC_L sections) with the USER HANDLE RULE
(2026-10-02 16:20): a handle is removed LOCALLY -- only a short stretch of donor wall around it becomes tube, the
rest of the donor keeps its real wall. Whole-section tube only if local windows cannot reach genus 0 (flagged).

Why a new ladder: v4's ladder (localise -> window search -> prune -> shrink -> whole section -> tubes) re-chooses
EVERY section's margin and suppression. On v5 that (a) re-opens decisions v4 already made for the carotid sections
(probe case_w_012_left__topcow_mr_003_L: localised at margin 1.0 where v4 baked at 0.5, so v4's own fork handle --
which v4 needed ecaext_R as a whole tube for -- comes back, its 10 mm bins cannot see it, no single-section or
section-pair probe makes a handle, and the ladder falls to its whole-section rung), and (b) gives up on a donor
after 5 rounds of 10 mm bins.

Stated diffs against bake_meshes_v4.bake_one / bake_anatomy (every other step is v4's own code, called from B):

 D1  CAROTID SECTIONS PINNED TO v4. The sections on RCCA / RECA / LCCA / LECA (unchanged centrelines and surfaces
     in v5) are baked exactly as v4 baked them: v4's margin (mesh_v4.json margin_mm), v4's suppressed stretches,
     v4's whole-section tubes, v4's far clamp first. The VA sections (va_R / va_L TopBrain distal pieces, re-
     registered in v5, and the donors vaC_R / vaC_L) are the only ones searched.
 D2  VA MARGIN. The VA sections get their own ownership margin, 0.5 then 1.0 (v4's MARGINS, v4's rule: localise at
     the margin with fewer handles, 0.5 on a tie). The VA sections come after every carotid section in the union,
     so their margin cannot change what the carotid sections add.
 D3  VA LOCALISATION (rung L), up to 25 rounds (v4: 5): bins along each VA section's branch, from WINDOW_MM before
     its start to WINDOW_MM past its end (its ownership reach), at 10 mm as v4, then 20 / 40 / 80 mm only when the
     smaller bins find nothing new -- a handle longer than a bin (a vein channel beside the VA touching it at both
     ends) is invisible to 10 mm bins. Pinch-split genus (B.piece_genus), padding B.BIN_PAD_MM, as v4.
 D4  VA BISECTION (rung W) when a round finds nothing new: suppress a VA section's whole reach; if the genus drops,
     halve it while a half still lowers the genus (straddling handle: the middle window), down to 12 mm. Then back
     to D3. Replaces v4's single-section probe + 12 mm slide (which needs a handle made by ONE section alone).
 D5  PRUNE / SPLIT / SHRINK on the VA stretches only (v4's pinned stretches stay as v4 had them): prune as v4; a
     stretch > 24 mm is split by restoring its middle while genus 0 holds (recursively); each end shrunk by bisection
     to SHRINK_TOL_MM (2 mm, v4's).
 D6  FALLBACKS, each flagged: C) D3-D5 over every section, carotid included ("carotid_changed": the right route may
     move); D) a donor section whole as tube ("donor_tube", real_surface false, course kept); E) v4's rung 5 / 6.
     ("keep that VA from v4" needs a re-assembled anatomy -- outside the bake; reported if E is reached.)
 D7  SPEED, bit-identical: the tube field is built once per bake_one and each section's per-voxel inputs (real SDF,
     capsule, ownership margin and owning arclength) once; an attempt undoes only the sections whose (margin,
     suppressed stretches) changed and re-applies them exactly as add_real_sections_owned does (same `would`,
     same ownership test, same suppression, same order). --selftest compares the union against v4's own path.
 D8  The report is written as mesh_v5.json (v4's keys + "ladder_v5b": pinned state, VA margin, rung, flags, the
     tubed length per VA section and its share of the donor stretch).

    python3 /sp/v5_va/bake_v5b.py --anatomies /repo/carotid_data/anatomies_v5 [--only A,B] [--shard i/n] [-v]
    python3 /sp/v5_va/bake_v5b.py --selftest <anatomy root>
Runs in neve-build-meshlab.
"""
import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import bake_v5  # noqa: E402  (installs sections_for_v5 into bake_meshes_v4)

B = bake_v5.B
from scipy.spatial import cKDTree  # noqa: E402
from sdf_mesher import iso_surface, mesh_stats, tube_field  # noqa: E402
from sdf_union import BAND_MM, _capsule_radii  # noqa: E402
from sdf_union_v4 import WINDOW_MM, Samples, genus, ownership_margin  # noqa: E402

V4_DIR = os.environ.get("V4_DIR", "/repo/carotid_data/anatomies_v4")
REPORT_V5 = "mesh_v5.json"
CAROTID = ("RCCA", "RECA", "LCCA", "LECA")
VA_MARGINS = B.MARGINS
VA_ROUNDS = 25
VA_BINS_MM = (10.0, 20.0, 40.0, 80.0)
VA_MIN_WINDOW_MM = 12.0
VA_SPLIT_ABOVE_MM = 24.0
MAX_BISECTIONS = 12
FAR = -1e3
# test hooks (probe validation of the deeper rungs only; recorded in ladder_v5b.test_overrides):
#   V5B_BINS="40,80" -> other bin sizes; V5B_BINS="" -> no bins (bisection only); V5B_NO_LOCAL=1 -> no local search
TEST = {}
if os.environ.get("V5B_BINS") is not None:
    VA_BINS_MM = tuple(float(v) for v in os.environ["V5B_BINS"].split(",") if v.strip())
    TEST["bins_mm"] = list(VA_BINS_MM)
NO_LOCAL = os.environ.get("V5B_NO_LOCAL") == "1"
if NO_LOCAL:
    TEST["no_local"] = True


def is_va(sec):
    return sec.branch in ("RVA", "LVA")


# ---- D7: the union, cached ------------------------------------------------------------------------------------------
class UnionCache:
    """One tube field per bake_one; sections applied / undone in order, exactly as add_real_sections_owned."""

    def __init__(self, br, samp, secs, spacing, verbose=False):
        self.field = tube_field(br, spacing=spacing, verbose=verbose)
        self.samp, self.spacing, self.verbose = samp, spacing, verbose
        self.pre = {sec.name: self._prep(sec) for sec in secs}
        self.applied = []          # [(key, undo)], key = (name, margin, spans)

    def _prep(self, sec):
        import pyvista as pv
        f, sp = self.field, self.spacing
        cap_r = _capsule_radii(sec.points, sec.radii, sec.taper_start, sec.taper_end)
        cap = tube_field([SimpleNamespace(coordinates=sec.points, radii=cap_r)],
                         spacing=sp, band=int(np.ceil(BAND_MM / sp)) + 1)
        off = np.round((cap.offset - f.offset) / sp).astype(int)
        idx = np.argwhere(cap.region & (cap.values > -BAND_MM))
        gidx = idx + off
        ok = np.all((gidx >= 0) & (gidx < np.array(f.values.shape)), axis=1)
        idx, gidx = idx[ok], gidx[ok]
        if not len(idx):
            return None
        coords = gidx * sp + f.offset
        d = np.asarray(pv.PolyData(coords).compute_implicit_distance(sec.surface)["implicit_distance"])
        f_real = (-d).astype(np.float32)
        f_sec = np.minimum(f_real, cap.values[idx[:, 0], idx[:, 1], idx[:, 2]])
        n = len(idx)
        return {"sec": sec, "gidx": gidx.astype(np.int32), "f_sec": f_sec, "m": np.full(n, np.nan),
                "s": np.full(n, np.nan), "n": n, "ntri": int(sec.surface.n_cells)}

    def _g(self, P):
        g = P["gidx"]
        return g[:, 0], g[:, 1], g[:, 2]

    def _coords(self, P, mask):
        return P["gidx"][mask].astype(np.int64) * self.spacing + self.field.offset

    def _apply(self, name, margin, spans):
        P = self.pre.get(name)
        if P is None:
            return {"voxels": 0}, None
        v, r = self.field.values, self.field.region
        g = self._g(P)
        cur = v[g]
        undo = (cur.copy(), r[g])
        would = P["f_sec"] > cur
        need = would & np.isnan(P["m"])
        if need.any():
            m_, _, s_ = ownership_margin(self._coords(P, need), P["sec"], self.samp, return_anchor=True)
            P["m"][need] = m_
            P["s"][need] = s_
        owned = np.ones(P["n"], bool)
        w = np.nonzero(would)[0]
        owned[w] = P["m"][w] >= margin
        sw = P["s"][w]
        for lo, hi in spans:
            owned[w[(sw >= lo) & (sw <= hi)]] = False
        use = would & owned
        vals = cur.copy()
        vals[use] = P["f_sec"][use]
        v[g] = vals
        r[g] = True
        st = {"voxels": int(P["n"]), "voxels_widened": int(use.sum()), "voxels_refused": int((would & ~owned).sum()),
              "surface_tris": P["ntri"]}
        if self.verbose:
            print("   real %-9s %7d vox, %6d widened, %5d refused (not owned)"
                  % (name, P["n"], st["voxels_widened"], st["voxels_refused"]), flush=True)
        return st, undo

    def build(self, seq):
        """seq: [(name, margin, spans tuple)] in union order. Returns {name: stats}."""
        k = 0
        while k < min(len(seq), len(self.applied)) and self.applied[k][0] == seq[k]:
            k += 1
        for key, undo, _ in reversed(self.applied[k:]):
            if undo is not None:
                g = self._g(self.pre[key[0]])
                self.field.values[g] = undo[0]
                self.field.region[g] = undo[1]
        self.applied = self.applied[:k]
        for key in seq[k:]:
            st, undo = self._apply(*key)
            self.applied.append((key, undo, st))
        return {key[0]: st for key, _, st in self.applied}

    def iso(self, far_clamp):
        f = self.field
        far = None
        if far_clamp:
            far = f.values < B.FAR_BELOW
            # the restore below writes FAR back: every clamped value must BE the placeholder (one extra bool grid)
            odd = np.greater(f.values, FAR)
            np.logical_and(odd, far, out=odd)
            assert not odd.any(), "a value below FAR_BELOW that is not the FAR placeholder"
            del odd
            f.values[far] = -2.0 * self.spacing
        try:
            return B.keep_largest(iso_surface(f))
        finally:
            if far is not None:
                f.values[far] = FAR


# ---- D3: VA localisation ---------------------------------------------------------------------------------------------
def reach(br, sec):
    """A VA section's ownership reach on its branch: [s0 - WINDOW_MM, s0 + L + WINDOW_MM] (branch arclength)."""
    b = B.branch(br, sec.branch)
    S = B.arclen(np.asarray(b.coordinates, float))
    L = float(B.arclen(sec.points)[-1])
    return max(sec.s0 - WINDOW_MM, 0.0), min(sec.s0 + L + WINDOW_MM, float(S[-1]))


def localise_bins(mesh, br, pool, bin_mm):
    """{section: [(lo, hi)]} of bins of bin_mm (stride bin_mm / 2) along each section's reach whose patch carries a
    handle (pinch vertices split) -- v4's localise_handles on the branch arclength, any bin size."""
    cen = np.asarray(mesh.cell_centers().points)
    out = {}
    for sec in pool:
        b = B.branch(br, sec.branch)
        P, R = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        S = B.arclen(P)
        a, z = reach(br, sec) if is_va(sec) else (sec.s0, sec.s0 + float(B.arclen(sec.points)[-1]))
        sel = (S >= a - 1e-6) & (S <= z + 1e-6)
        p, r, s = P[sel], R[sel], S[sel]
        if len(p) < 3:
            continue
        d, i = cKDTree(p).query(cen)
        near = d < (1.8 * r[i] + 2.0)
        si = s[i]
        for lo in np.arange(s[0], max(s[-1] - 1e-6, s[0]) + 1e-9, bin_mm / 2.0):
            m = near & (si >= lo) & (si < lo + bin_mm)
            if m.sum() < 50:
                continue
            piece = mesh.extract_cells(np.nonzero(m)[0]).extract_surface().clean().triangulate()
            if not any(c["genus"] > 0.25 for c in genus(piece) if c["cells"] > 50):
                continue
            faces = np.asarray(piece.faces).reshape(-1, 4)[:, 1:]
            if any(g >= 0.75 for g, _, _ in B.piece_genus(faces)):
                out.setdefault(sec.name, []).append(
                    (max(lo - B.BIN_PAD_MM, s[0]), min(lo + bin_mm + B.BIN_PAD_MM, s[-1])))
    return out


def tubed_mm(spans, lo, hi):
    return float(sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in spans))


# ---- the ladder ------------------------------------------------------------------------------------------------------
def bake_one_v5b(root, spacing, full_tris, obj_tris, verbose, dump_iso=None, far_clamp=False, far_clamp_why=None):
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    t0 = time.time()
    name = os.path.basename(os.path.normpath(root))
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs_all, spans = B.sections_for(root, br)
    rep4 = json.load(open(os.path.join(V4_DIR, name, B.REPORT), encoding="utf-8"))
    v4_margin = rep4["margin_mm"]
    v4_tube = list(rep4.get("sections_tube_only") or [])
    car_names = {s.name for s in secs_all if s.branch in CAROTID}
    pin_sup = {k: [tuple(x) for x in v] for k, v in (rep4.get("suppressed_stretches_mm") or {}).items()
               if k in car_names}
    v4_tubes_only = str(rep4.get("union", "")).startswith("TUBES")
    drop = set(n for n in v4_tube if n in car_names)
    if v4_tubes_only or v4_margin is None:
        drop |= car_names
    for n in drop:
        spans[n] = "tube only (as in v4)"
    secs = [s for s in secs_all if s.name not in drop]
    carotid = [s for s in secs if not is_va(s)]
    vas = [s for s in secs if is_va(s)]
    flags = []
    lad = {"pinned": {"carotid_margin_mm": v4_margin, "suppressed": {k: [list(x) for x in v] for k, v in pin_sup.items()},
                      "tube_only": sorted(drop), "v4_far_clamp": bool((rep4.get("far_clamp") or {}).get("used"))},
           "rounds": [], "bisections": [], "flags": flags, "test_overrides": TEST}

    t_c = time.time()
    cache = UnionCache(br, samp, secs, spacing, verbose)
    lad["cache_seconds"] = round(time.time() - t_c, 1)
    attempts = []

    def fmt(sup):
        return {k: [[round(a, 2), round(b, 2)] for a, b in v] for k, v in sup.items() if v}

    def attempt(sections, va_margin, tag, sup):
        ta = time.time()
        seq = [(s.name, (va_margin if is_va(s) else v4_margin), tuple(tuple(x) for x in sup.get(s.name, [])))
               for s in sections]
        st = cache.build(seq)
        iso, st_iso, dropped = cache.iso(far_clamp)
        g = genus(iso)
        gmax = max(c["genus"] for c in g) if g else -1
        nme = B.edge_defects(iso)
        if gmax == 0 and nme:
            gmax = 0.5
        attempts.append(dict(tag, genus=gmax, nonmanifold_edges=nme, components_before_keep=st_iso["comps"],
                             dropped_fragments=dropped["cells"][:10], dropped_volume_mm3=round(dropped["volume_mm3"], 2),
                             seconds=round(time.time() - ta, 1)))
        if verbose:
            print("   %s -> genus %s, %d comps, %.0f s" % (tag, gmax, st_iso["comps"], time.time() - ta), flush=True)
        return iso, st, gmax

    def with_span(sup, nm, span):
        out = {k: list(v) for k, v in sup.items()}
        out[nm] = B.merge_spans(out.get(nm, []) + [span])
        return out

    def replace_span(sup, nm, old, new):
        out = {k: list(v) for k, v in sup.items()}
        out[nm] = B.merge_spans([x for x in out[nm] if x != old] + list(new))
        return {k: v for k, v in out.items() if v}

    def without_span(sup, nm, span):
        out = {k: [s for s in v if s != span] for k, v in sup.items()}
        return {k: v for k, v in out.items() if v}

    suppress = {k: list(v) for k, v in pin_sup.items()}
    full, sec_stats, va_margin, rung, tube_only, mode = None, None, None, None, [], "owned"

    # 1. pinned carotid sections, VA sections whole: VA margin 0.5, then 1.0
    isos, gen = {}, {}
    for vm in VA_MARGINS:
        iso, st, g = attempt(secs, vm, {"margin_mm": v4_margin, "va_margin_mm": vm}, suppress)
        isos[vm], gen[vm] = (iso, st), g
        if g == 0:
            full, sec_stats, va_margin, rung = iso, st, vm, "1 pinned, VA whole (va margin %s)" % vm
            break
    if full is None:
        tubes = attempt([], None, {"margin_mm": None, "tubes_check": True}, {})
        if tubes[2] != 0:
            raise RuntimeError("even the tube-only mesh has a handle: a centerline-level fusion")
        tubes = tubes[:2]
        _, _, g_car = attempt(carotid, None, {"margin_mm": v4_margin, "carotid_check": True}, suppress)
        lad["carotid_check_genus"] = g_car
        va_margin = min(VA_MARGINS, key=lambda x: (gen[x], x))
        (iso, st), g = isos[va_margin], gen[va_margin]
        isos.clear()
        protected = {k: list(v) for k, v in pin_sup.items()}     # never pruned / shrunk

        def search(pool, iso, st, g, suppress, label):
            """D3 + D4 over the sections of pool until genus 0 or no progress. Returns (iso, st, g, suppress)."""
            stalls = 0
            for rnd in range(1, VA_ROUNDS + 1):
                if g == 0:
                    break
                new, used_bin = False, None
                for bin_mm in VA_BINS_MM:
                    found = localise_bins(iso, br, pool, bin_mm)
                    for nm, sp_ in found.items():
                        cur = suppress.get(nm, [])
                        for lo, hi in sp_:
                            if not any(a <= lo and hi <= b for a, b in cur):
                                cur = B.merge_spans(cur + [(lo, hi)])
                                new = True
                        suppress[nm] = cur
                    if new:
                        used_bin = bin_mm
                        break
                if new:
                    iso, st, g = attempt(secs, va_margin, {"va_margin_mm": va_margin, "round": rnd, "pool": label,
                                                           "bin_mm": used_bin, "suppressed": fmt(suppress)}, suppress)
                    lad["rounds"].append({"round": rnd, "pool": label, "bin_mm": used_bin, "genus": g})
                    continue
                # D4: bisection along the pool's reaches
                g0 = g
                hit = None
                for sec in pool:
                    a, z = reach(br, sec) if is_va(sec) else (sec.s0, sec.s0 + float(B.arclen(sec.points)[-1]))
                    trial = with_span(suppress, sec.name, (a, z))
                    iso_t, st_t, g_t = attempt(secs, va_margin, {"bisect": [sec.name, round(a, 2), round(z, 2)],
                                                                 "pool": label}, trial)
                    if g_t < g0:
                        hit = (sec, a, z, iso_t, st_t, g_t)
                        break
                if hit is None:
                    stalls += 1
                    break
                sec, a, z, iso_b, st_b, g_b = hit
                steps = 0
                while z - a > VA_MIN_WINDOW_MM and steps < MAX_BISECTIONS:
                    steps += 1
                    mid = 0.5 * (a + z)
                    moved = False
                    for span in ((a, mid), (mid, z),
                                 (max(a, mid - VA_MIN_WINDOW_MM / 2), min(z, mid + VA_MIN_WINDOW_MM / 2))):
                        trial = with_span(suppress, sec.name, span)
                        iso_t, st_t, g_t = attempt(secs, va_margin, {"bisect": [sec.name, round(span[0], 2),
                                                                                round(span[1], 2)], "pool": label}, trial)
                        if g_t < g0:
                            a, z, iso_b, st_b, g_b, moved = span[0], span[1], iso_t, st_t, g_t, True
                            break
                    if not moved:
                        break
                suppress = with_span(suppress, sec.name, (a, z))
                iso, st, g = iso_b, st_b, g_b
                lad["bisections"].append({"pool": label, "section": sec.name, "window": [round(a, 2), round(z, 2)],
                                          "genus_before": g0, "genus_after": g})
            return iso, st, g, suppress

        # L / W on the VA sections (only if the pinned carotid part is itself a tree)
        if g_car == 0 and not NO_LOCAL:
            iso, st, g, suppress = search(vas, iso, st, g, suppress, "va")
            if g == 0:
                rung = "2 VA-local windows"
        if g != 0 and not NO_LOCAL:
            # C. every section, carotid included (flagged)
            flags.append("carotid_changed" if g_car != 0 else "carotid_changed (VA-local search stalled)")
            iso, st, g, suppress = search(secs, iso, st, g, suppress, "all")
            if g == 0:
                rung = "C local windows, carotid sections included"
        def refine(sections, suppress, full, sec_stats):
            """D5: prune / split / shrink every stretch v4 did not have (v4's pinned stretches stay)."""
            sec_names = {s.name for s in sections}

            def new_spans():
                return [(nm, sp_) for nm, v in suppress.items() for sp_ in v
                        if nm in sec_names and sp_ not in protected.get(nm, [])]
            for nm, span in new_spans():
                trial = without_span(suppress, nm, span)
                iso_p, st_p, g_p = attempt(sections, va_margin, {"prune": [nm, round(span[0], 2), round(span[1], 2)]},
                                           trial)
                if g_p == 0:
                    suppress, full, sec_stats = trial, iso_p, st_p
            work = new_spans()
            while work:
                nm, span = work.pop(0)
                a, b = span
                if b - a > VA_SPLIT_ABOVE_MM:
                    h = (b - a) / 4.0
                    parts = [(a, a + h), (b - h, b)]
                    trial = replace_span(suppress, nm, span, parts)
                    iso_s, st_s, g_s = attempt(sections, va_margin, {"split": [nm, round(a, 2), round(b, 2)]}, trial)
                    if g_s == 0:
                        suppress, full, sec_stats = trial, iso_s, st_s
                        work = [(nm, p) for p in parts] + work
                        continue
                cur = span
                for end in ("lo", "hi"):
                    ok_, bad_ = (a, b) if end == "lo" else (b, a)
                    while abs(bad_ - ok_) > SHRINK_TOL:
                        mid = 0.5 * (ok_ + bad_)
                        newsp = (mid, b) if end == "lo" else (a, mid)
                        trial = replace_span(suppress, nm, cur, [newsp])
                        iso_s, st_s, g_s = attempt(sections, va_margin, {"shrink": [nm, round(newsp[0], 2),
                                                                                    round(newsp[1], 2)]}, trial)
                        if g_s == 0:
                            ok_, suppress, full, sec_stats, cur = mid, trial, iso_s, st_s, newsp
                        else:
                            bad_ = mid
                    if end == "lo":
                        a = ok_
                    else:
                        b = ok_
            return suppress, full, sec_stats

        if g == 0:
            suppress, full, sec_stats = refine(secs, suppress, iso, st)
        else:
            # D. a donor section whole as tube (flagged), the one with the most stretches first, then both
            dons = sorted([s for s in vas if s.name.startswith("vaC")],
                          key=lambda s_: -tubed_mm(suppress.get(s_.name, []), -1e9, 1e9))
            tries = [[d] for d in dons] + ([dons] if len(dons) > 1 else [])
            for grp in tries:
                keep = [s for s in secs if s not in grp]
                sup_ = {k: v for k, v in suppress.items() if k not in {d.name for d in grp}}
                iso, st, g = attempt(keep, va_margin, {"without": [d.name for d in grp]}, sup_)
                if g == 0:
                    tube_only = [d.name for d in grp]
                    suppress, full, sec_stats = refine(keep, sup_, iso, st)
                    for d in grp:
                        spans[d.name] = "tube only (no local fix found; course kept)"
                    flags.append("donor_tube")
                    rung = "D donor section(s) whole as tube: %s" % ",".join(tube_only)
                    break
            if full is None:
                # E. v4's rung 5 over the carotid sections, then tubes only
                flags.append("v4_rung5")
                order = sorted(carotid, key=lambda s_: (st.get(s_.name, {}).get("voxels_widened", 0), s_.name))
                for sec in order:
                    keep = [s for s in secs if s is not sec]
                    iso, st2, g = attempt(keep, va_margin, {"without": sec.name}, suppress)
                    if g == 0:
                        tube_only = [sec.name]
                        suppress, full, sec_stats = refine(keep, suppress, iso, st2)
                        spans[sec.name] = "tube only (no local fix found)"
                        rung = "E carotid section whole as tube: %s" % sec.name
                        break
            if full is None:
                flags.append("tubes_only")
                full, sec_stats = tubes
                mode, rung = "tubes", "E tubes only"
    del cache
    lad["rung"] = rung
    lad["va_margin_mm"] = va_margin
    # tubed length per VA section, and its share of the donor stretch
    with open(os.path.join(root, "provenance.json"), encoding="utf-8") as f:
        x = json.load(f)["xform"]
    tub = {}
    for sec in vas:
        sp_ = suppress.get(sec.name, [])
        L = float(B.arclen(sec.points)[-1])
        lo, hi = (float(x[sec.name]["lo_mm"]), float(x[sec.name]["hi_mm"])) if sec.name in x and "lo_mm" in x[sec.name] \
            else (sec.s0, sec.s0 + L)
        t_in = tubed_mm(sp_, lo, hi)
        tub[sec.name] = {"stretch_mm": [round(lo, 2), round(hi, 2)], "windows_mm": [[round(a, 2), round(b, 2)] for a, b in sp_],
                         "n_windows": len(sp_), "tubed_mm": round(t_in, 2),
                         "tubed_frac": round(t_in / (hi - lo), 4) if hi > lo else None,
                         "tubed_beyond_mm": round(tubed_mm(sp_, -1e9, 1e9) - t_in, 2),
                         "whole_tube": sec.name in tube_only}
    lad["va_tubed"] = tub
    iso_full = full
    detail = B.right_route_detail(iso_full, samp, root)
    if dump_iso:
        os.makedirs(dump_iso, exist_ok=True)
        iso_full.save(os.path.join(dump_iso, name + ".vtp"))
    # v4's tail, unchanged
    full, full_path, full_used = None, None, None
    for fb in (full_tris, full_tris * 4 // 3, full_tris * 5 // 3, full_tris * 2, full_tris * 3):
        full, full_path = B.clean_decimate(iso_full, fb, raw_iso=True)
        if full is not None:
            full_used = fb
            break
    if full is None:
        raise B.NoCleanDecimation("no decimation of the genus-0 union to %d-%d tris is a clean tree"
                                  % (full_tris, full_tris * 3))
    full.save(os.path.join(root, B.FULL_NAME))
    obj, obj_used, obj_path, tried = B.obj_ladder(full, br, obj_tris)
    pv.save_meshio(os.path.join(root, B.MESH_NAME), obj)
    rr_obj, ok_obj = B.route_report(obj, br)
    go = genus(obj)
    out_n, tot_n = B.enclosure(obj, br)
    union = {"owned": "owned (sdf_union_v4)" if not tube_only else "owned, %s as tube" % ",".join(tube_only),
             "tubes": "TUBES ONLY (no owned union passed the topology gate)"}[mode]
    rep = {"spacing_mm": spacing, "seconds": round(time.time() - t0, 1), "union": union,
           "sections_tube_only": tube_only + sorted(drop),
           "suppressed_stretches_mm": fmt(suppress),
           "sections_tube_in_effect": B.tube_in_effect(secs, suppress),
           "localise_margin_mm": va_margin,
           "far_clamp": {"used": bool(far_clamp), "why": far_clamp_why},
           "right_route_detail": detail,
           "margin_mm": v4_margin, "va_margin_mm": va_margin, "attempts": attempts, "sections": sec_stats,
           "spans": spans, "full": dict(mesh_stats(full), genus=genus(full)), "obj": dict(mesh_stats(obj), genus=go),
           "obj_tris_budget": obj_used, "routes": rr_obj,
           "decimation": {"full": full_path, "full_budget": full_used, "obj": obj_path, "obj_budgets_rejected": tried},
           "centerline_pts_outside": out_n, "centerline_pts_checked": tot_n,
           "ladder_v5b": lad,
           "pass": bool(ok_obj and B.is_clean(obj) and out_n == 0)}
    with open(os.path.join(root, REPORT_V5), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


SHRINK_TOL = B.SHRINK_TOL_MM


def lost_of(rep):
    """Real surface lost beyond v4's: whole VA sections as tube, carotid sections changed, tubes only."""
    fl = rep.get("ladder_v5b", {}).get("flags", [])
    return (100 if "tubes_only" in fl else 0) + (10 if "v4_rung5" in fl else 0) + \
        sum(1 for v in rep.get("ladder_v5b", {}).get("va_tubed", {}).values() if v.get("whole_tube"))


def bake_anatomy_v5b(root, spacing, full_tris, obj_tris, verbose, dump_iso=None):
    """v4's bake_anatomy, starting from v4's own far-clamp choice (D1)."""
    name = os.path.basename(os.path.normpath(root))
    rep4 = json.load(open(os.path.join(V4_DIR, name, B.REPORT), encoding="utf-8"))
    fc0 = bool((rep4.get("far_clamp") or {}).get("used"))
    why0 = "as v4 (%s)" % (rep4.get("far_clamp") or {}).get("why") if fc0 else None
    try:
        first = B.bake_one(root, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp=fc0, far_clamp_why=why0)
    except B.NoCleanDecimation as e:
        return B.bake_one(root, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp=not fc0,
                          far_clamp_why="flipped from v4's choice: %s" % e)
    lost = lost_of(first)
    if not lost:
        return first
    keep = {f: open(os.path.join(root, f), "rb").read() for f in (B.FULL_NAME, B.MESH_NAME, REPORT_V5)}
    # the second call dumps its union aside, so the dumped union is always the KEPT call's
    flip_dump = os.path.join(dump_iso, "_flip") if dump_iso else None
    try:
        second = B.bake_one(root, spacing, full_tris, obj_tris, verbose, flip_dump, far_clamp=not fc0,
                            far_clamp_why="flipped from v4's choice: the first union lost %s" % first["ladder_v5b"]["flags"])
        if second["pass"] and lost_of(second) < lost:
            if flip_dump:
                os.replace(os.path.join(flip_dump, name + ".vtp"), os.path.join(dump_iso, name + ".vtp"))
            return second
    except B.NoCleanDecimation:
        pass
    for f, b in keep.items():
        with open(os.path.join(root, f), "wb") as fh:
            fh.write(b)
    return first


# install: bake_meshes_v4.bake_anatomy -> bake_anatomy_v5b -> B.bake_one (looked up at call time) = bake_one_v5b
B.bake_one = bake_one_v5b
B.bake_anatomy = bake_anatomy_v5b
B.REPORT_V5 = REPORT_V5


def selftest(root, configs=None):
    """D7 check: the cached union equals add_real_sections_owned's, bit for bit, through apply / undo sequences."""
    from eve_bench.dualdevicenav import load_branches
    from sdf_union_v4 import add_real_sections_owned
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs, _ = B.sections_for(root, br)
    vas = [s for s in secs if is_va(s)]
    cache = UnionCache(br, samp, secs, B.SPACING_MM)
    d0 = vas[-1]
    a, z = reach(br, d0)
    configs = configs or [
        ("all 0.5", secs, 0.5, {}),
        ("all 1.0 + donor window", secs, 1.0, {d0.name: [(a + 20.0, a + 36.0)]}),
        ("carotid only", [s for s in secs if not is_va(s)], 0.5, {}),
        ("all 0.5 again", secs, 0.5, {}),
        ("all 0.5, two windows", secs, 0.5, {d0.name: [(a + 5.0, a + 15.0), (a + 40.0, a + 60.0)],
                                             vas[0].name: [(vas[0].s0, vas[0].s0 + 8.0)]}),
    ]
    out = []
    for tag, ss, m, sup in configs:
        cache.build([(s.name, m, tuple(sup.get(s.name, []))) for s in ss])
        v_c = cache.field.values.copy()
        r_c = cache.field.region.copy()
        field = tube_field(br, spacing=B.SPACING_MM)
        add_real_sections_owned(field, ss, samp, B.SPACING_MM, m, suppress=sup)
        ok = bool(np.array_equal(v_c, field.values) and np.array_equal(r_c, field.region))
        nd = int((v_c != field.values).sum())
        rec = {"config": tag, "identical": ok, "values_differing": nd}
        if tag == "all 0.5":
            # the far clamp (v4's attempt: clamp, then iso) and its restore (the configs after this one)
            iso_c = cache.iso(True)[0]
            field.values[field.values < B.FAR_BELOW] = -2.0 * B.SPACING_MM
            iso_v = B.keep_largest(iso_surface(field))[0]
            same = bool(np.array_equal(np.asarray(iso_c.points), np.asarray(iso_v.points))
                        and np.array_equal(np.asarray(iso_c.faces), np.asarray(iso_v.faces)))
            rec["far_clamp_iso_identical"] = same
            rec["identical"] = ok and same
            print("SELFTEST far clamp iso identical %s" % same, flush=True)
            del iso_c, iso_v
        out.append(rec)
        print("SELFTEST %s: identical %s (%d values differ)" % (tag, ok, nd), flush=True)
        del field, v_c, r_c
    return out


def record_provenance(root, rep):
    """Add the bake record to provenance.json (every other key, the licences included, kept as is)."""
    p = os.path.join(root, "provenance.json")
    with open(p, encoding="utf-8") as f:
        prov = json.load(f)
    lad = rep["ladder_v5b"]
    prov["bake_v5"] = {
        "script": "v5_va/bake_v5b.py (bake_meshes_v4 unchanged + bake_v5 sections + the handle rule, D1-D8)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "report": REPORT_V5, "pass": rep["pass"],
        "files": [B.MESH_NAME, B.FULL_NAME, REPORT_V5], "spacing_mm": rep["spacing_mm"],
        "obj_tris": rep["obj"]["tris"], "obj_tris_budget": rep["obj_tris_budget"], "full_tris": rep["full"]["tris"],
        "union": rep["union"], "rung": lad["rung"], "flags": lad["flags"], "carotid_margin_mm": rep["margin_mm"],
        "va_margin_mm": lad["va_margin_mm"], "far_clamp": rep["far_clamp"],
        "sections_tube_only": rep["sections_tube_only"], "suppressed_stretches_mm": rep["suppressed_stretches_mm"],
        "va_tubed": lad["va_tubed"], "right_route_detail": rep["right_route_detail"],
        "seconds": rep["seconds"]}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", default="/repo/carotid_data/anatomies_v5")
    ap.add_argument("--only", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--spacing", type=float, default=B.SPACING_MM)
    ap.add_argument("--full-tris", type=int, default=60000)
    ap.add_argument("--obj-tris", type=int, default=23500)
    ap.add_argument("--dump-iso", default="/repo/carotid_data/va_sources/v5_reg/iso_v5",
                    help="the undecimated gate-passing unions (for the VA wall-detail check); kept OUT of anatomies_v5")
    ap.add_argument("--force", action="store_true", help="re-bake anatomies that already have mesh_v5.json")
    ap.add_argument("--no-provenance", action="store_true", help="do not add the bake record to provenance.json")
    ap.add_argument("--selftest", default=None, help="anatomy root: check D7 against add_real_sections_owned")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        res = selftest(a.selftest)
        return 0 if all(r["identical"] for r in res) else 1
    dump = a.dump_iso
    folders = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    if a.only:
        want = set(a.only.split(","))
        folders = [f for f in folders if os.path.basename(os.path.dirname(f)) in want]
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        folders = folders[i::n]
    bad = 0
    for f in folders:
        root = os.path.dirname(f)
        name = os.path.basename(root)
        if os.path.exists(os.path.join(root, REPORT_V5)) and not a.force:
            print("%s kept (mesh_v5.json exists)" % name, flush=True)
            continue
        t0 = time.time()
        print("== %s" % name, flush=True)
        try:
            rep = B.bake_anatomy(root, a.spacing, a.full_tris, a.obj_tris, a.verbose, dump)
        except Exception as e:                                           # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print("FAILED %s: %s" % (type(e).__name__, e), flush=True)
            continue
        if not a.no_provenance:
            record_provenance(root, rep)
        lad = rep["ladder_v5b"]
        print("%s: pass %s, rung %s, flags %s, tubed %s, %.0f s" % (
            name, rep["pass"], lad["rung"], lad["flags"],
            {k: (v["tubed_mm"], v["tubed_frac"]) for k, v in lad["va_tubed"].items()}, time.time() - t0), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
