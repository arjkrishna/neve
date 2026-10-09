#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\checks\checks_v6_1p6_dev.py (= /sp/v6/checks/checks_v6_1p6_dev.py in the build containers)
"""checks_v6.py -- the v6 post-bake check suite for ONE baked anatomy (stage 2; s2_harden_wall_bake, 2026-10-04).

    python3 checks_v6.py <name> [--tree DIR] [--iso DIR] [--out DIR] [--parts item1,route,lumen,va,orient,loader,geom]
                         [--force]                    (neve-build-meshlab; /repo read-only)
    -> OUT/<name>.json (parts are skipped when already present, unless --force); "gates" = every invariant with
       pass / fail, "pass" = all blocking gates pass.

Parts (the measuring code is the v4 / v5 verifiers', imported unchanged unless stated):
 item1  STRICT MESH AUDIT -- /sp/v5_va/checks/chk_one.item1 (v4's topo_audit: pymeshlab topology raw / welded,
        self-intersecting faces on OBJ / 60k / union, Euler / orientation / duplicates, budgets, both routes
        navigable, every centreline point inside by winding number). Gate: strict_pass.
 route  RIGHT ROUTE vs v4 under EXACT ownership (vvw/measure.py Own 'exact' and 'k24'; v4's union rebuilt from
        mesh_v4.json as rrd_v4.rebuild_v4), fork and siphon frac_0.3mm ratios v6 / v4.
        D13 baseline: where the v6 RECA TopBrain tip (xform ecaext_R) is re-posed or skipped against v4's, v4 is ALSO
        rebuilt WITHOUT its tip (RECA truncated at v4's ecaext_R join_at_mm, the ecaext_R section dropped; = the
        pilot's v4_no_tip.py, + v4's far clamp) and that is the reference when it differs from v4 as shipped (= v4's
        tip owned RCCA wall, i.e. it ran along the ICA). A provenance override (route_baseline / d13 keys written by
        the carotid tool) is honoured. Gate: fork and siphon ratio >= 0.99 against the baseline used.
        1.1 (user decisions D19 / D20: the right fork is re-posed and/or x-mirrored, the RCCA training route moves):
        the gate is PER REAL SECTION -- each section's own slice of the RCCA (fork = lower_R [host_cut, host_cut +
        cca + min(ica, ica_real)], siphon = siphon_R [host_cut + cca + ica, end], from each tree's own provenance, the
        slices the bake uses), exact ownership, 2 mm trims, v6 against the SAME section in the baseline (v4 or v4
        without its tip): the moved surface against itself. 1.0's positional split (RCCA-owned wall < / >= 130 mm) is
        kept as information (gate_ratio_legacy_130). New gate rcca_prefix_unchanged: v6's RCCA points and radii up to
        min(host_cut v6, v4) - 0.5 mm are v4's bit for bit (the prefix arch -> seam 1 stays host); against the
        host's own RCCA as information.
 lumen  VA LUMEN ON THE POLYLINE -- wall/va_lumen_poly.py (0.05 mm, exact point-triangle distance, winding; 3 mm
        caps out) on the OBJ, collision_full and the union. Gate: OBJ minimum >= 0.65 mm on both VAs.
 va     WALL DETAIL (D5 / D15), exact ownership on the UNDECIMATED unions (v6 ISO, v5 iso_v5): per VA side the donor
        stretch, donor_real (minus the 25 mm seam zone and tube windows), the donor subclavian stretch, the TopBrain
        top, the whole VA; v5 the same (measure.va_sides). D15 split: the donor_real cells are classed by the donor's
        local zones (wall/seg/local/<donor>/zones.json, w > 0 = zone + blend) through the donor stations placed with
        the section's own xform, for v6 AND v5 (same zones, v5's own pose) -> frac_0.3mm inside / outside the zones.
        Info (no gate here; the identity gate is the donor-level one, wall_local/wd_local.py).
 orient ORIENTATION GATES -- gates/orient_gates.py measure_tree on this anatomy (tier A must be 0); v6-blocking flags
        with D8 applied (va_target <= 20 pass, 20-30 accepted + flag, > 30 blocking unless the VA tool recorded the
        D9 rung 3 "v5 top as shipped"); fork flags informational (D11).
 loader TRAINING LOADER -- the pilot's loader_v6p checks on this one anatomy (DualDeviceNavTopBrain over a temp dir
        holding only it): 18 branches, finite, radii > 0, insertion = host's, numbered host branches = host's except
        the donor subclavian branches, RCCA identical to v5 up to the siphon seam, every centreline point inside the
        OBJ, RCCA targets > 0. Known host point gaps (> 5 mm, identical in v5's record) are reported, not failed.
 1.3    (s2_integrate2, 2026-10-06; USER decisions D22 / D23, orchestrator D25; frozen 1.2 = checks_v6_v1_2.py):
        D22 RIGHT-ROUTE GATE = "SURFACE PRESERVED": for the right fork (lower_R) and the right siphon (siphon_R), the
        baked union's cells exactly owned by the RCCA in the piece's span beyond its seam zone ([cut, cut + max(blend
        v6, v4) + 2] for the fork; [seam 2, seam 2 + max(siphon blend v6, v4) + 2] for the siphon) and >= 2 mm before
        the span end, minus the bake's handle windows (mesh_v6 suppressed stretches of the section, and v4's mapped
        onto v6), must lie within 0.3 mm of v4's union (the D13 baseline union where it applies) CARRIED BY THE PIECE'S
        OWN TRANSFORM (x6 = ((x4 - a4) R4 + o4) M4 M6 - o6) R6^T + a6; mirror included) for >= 99 % of their area
        (exact point-to-triangle distance). REFERENCE VALIDITY (as D13 for v4's tip): cells whose v4 wall was occupied
        by ANOTHER branch (in v4's world another branch's tube value within v4's ownership margin + 0.3 mm of the
        RCCA's at the inverse-carried point; the fork's own ECA excepted) are left out of the gated area -- there v4's
        union never showed the piece's own surface (k_011_left mr_012_L: v4's RVA top fused on the right siphon's bulge,
        39 mm2 at 0.3-1.5 mm, which v6's re-placed VA no longer touches) -- and reported (v4_occupied, and the value
        with them: value_strict / strict_flag). Gates route_fork_d22_ge_0.99 / route_siphon_d22_ge_0.99 REPLACE
        route_fork_ge_0.99 / route_siphon_ge_0.99; the relief ratio (gate_ratio) stays as information. Reported, not
        gated: the seam zone and the handle windows (area, share within 0.3, max deviation) and COVERAGE (v4's own
        RCCA-owned cells of the same stretch, carried, within 0.3 mm of the v6 union: sees wall a new neighbour now
        owns). rcca_prefix_unchanged unchanged.
        D23: orient's tier A is accepted (and the blocking flags of that piece and the VA pair flags it causes) only
        where provenance v6_build.d23_accepted lists the piece (w_017_left mr_002_L RVA_top: D9 rung 3 v5 top as
        shipped, user decision); gate orient_tierA_0 reads "0 tier A except flagged D23".
        geom: geom_v6 1.1 (1 mm turns at every sampling phase, D25).
 1.4    (s2_build, 2026-10-07; USER decision D26 "A"; frozen 1.3 = checks_v6_v1_3.py; ONLY part route / route_d22
        changed -- every other part is code-identical to 1.3): D22 REFERENCE WHERE v4 HID PART OF THE PIECE'S REAL
        SURFACE. The gated area is now EVERY RCCA-owned cell of the piece beyond its seam zone (2 mm end trim, minus the
        handle windows) -- 1.3's v4-occupied exclusion (the integrator's reference-validity rule, never confirmed) is
        replaced by D26 and kept as information. A cell is preserved when it lies within 0.3 mm of v4's (baseline) union
        carried by the piece's own transform (D22), or -- where v4's mesh did not expose the piece's own scanned surface
        there -- within 0.3 mm of the piece's OWN scanned (source) surface carried by its v6 transform (the bake's rigid
        placement, sdf_union.load_surface: mirror, R, origin, anchor). Per piece: value_v4 (= value_strict, v4 only),
        value_own (v4 or own surface), d22_reference = "v4" when value_v4 >= 0.99; else "own_surface" (FLAGGED,
        d26_flag) when value_own >= 0.99 AND no real wall is lost (coverage: v4's own exposed RCCA-owned cells of the
        same stretch, carried, within 0.3 mm of the v6 union for >= 99 % of their area); else "none" (fails). value =
        the value under the reference used. Every set (gate_strict, the 1.3 gate set, v4_occupied, zone, windows) also
        reports within_0.3_own / area_off_v4_on_own_mm2 / area_off_both_mm2. Each record carries part_versions (the
        checks_v6 version that computed each part; parts reused from a 1.3 record say so).
        Part route_d22 (USER, 2026-10-07 15:15, "re-run ONLY the changed route-check part (D22/D26)" on anatomies whose
        geometry and bake are unchanged): re-runs ONLY rec.route.d22 -- the same route_d22() on the same inputs part
        route feeds it (exact ownership of the v6 union and of the D13 baseline union) -- and carries the rest of the
        route record (RCCA prefix, D13 baseline decision, relief information) from the record's earlier route pass. The
        carried D13 decision's provenance inputs (RECA tip status, declared baseline) are recomputed and must match, else
        the part errors (the caller then runs the full part route). part_versions.route keeps the version of the
        carried pass; part_versions.route_d22 = this version; route.d22_only_rerun says so.
 1.5    (s2_build, 2026-10-07; USER rule D27 "just do the recommended and then report/flag it"; frozen 1.4 =
        checks_v6_v1_4.py; ONLY route_d22() changed, and ONLY for a piece whose 1.4 reference is "none" -- every piece
        that passes under 1.4 takes exactly 1.4's code path and values): BAKE REPAIR WINDOW. D22 reports, and does not
        gate, the bake's windows (seam blend zone, handle windows). For a piece that fails under 1.4, the cells off
        BOTH references (v4 and own surface, > 0.3 mm) at which the bake's own section field is the section CAPSULE
        rather than the real wall (sdf_union._capsule_radii 1.8 r + 1 mm round the section's own centreline, evaluated
        as sdf_mesher.tube_field does, on bake_v6's own sections -- sections_for_morph -- of this tree: capsule value <
        f_real, i.e. the bake clipped the scanned lumen there and the union shows its smooth capsule) are the bake's
        REPAIR WINDOW: reported (d22_repair_window: cells, area, arclength range, how far the scanned lumen reaches
        beyond the capsule, the same test in v4's world for information) and left out of the gated area. The piece's
        reference is then re-decided exactly as 1.4 does on the remaining cells (v4, else own_surface with
        no_wall_lost); value_before_d27 keeps 1.4's value. d27_flag marks such a piece. Applied by D27 to
        k_011_left mr_012_L's siphon (1.4: own 0.9898, 17.79 mm2 off both, all on the capsule at s 206.75-207.5) and to
        any piece with the same condition.
 geom   (1.2) WHOLE-TREE CENTRELINE GEOMETRY -- geom_v6.measure (needs only the tree; also runs before the bake):
        clearance d - r_a - r_b >= 0.6 mm between curves of different placement groups (CR {RCCA, RECA}, CL {LCCA,
        LECA}, VR {RVA, (17)}, VL {LVA, (18), (19)}, every other curve its own) over point pairs with at least one
        point that is not v5's (joins left out over the child's fused run + 15 mm), and the 1 mm turn <= 60 deg at
        every new point (or no sharper than the same v5 curve). Catches contacts made ACROSS the composers (e.g. a
        re-posed right fork against a VA). Gates geom_clearance_new_ge_0.6, geom_kink_new_le_60.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BK  # noqa: E402  (B.branch fallback; sections_for_v6 for v6 trees)

B = BK.B
sys.path.insert(0, "/sp/v5_va/checks")
import chk_one as CK  # noqa: E402
sys.path.insert(0, "/sp/v5_va/vvw")
sys.path.insert(0, "/sp/v5_va/swap13")
import measure as M  # noqa: E402
import rrdx as RX  # noqa: E402

VERSION = "checks_v6 1.6 (2026-10-08; s3_fix: 1.6 = part coverage: the right-route pieces against the v4 / v5 SHIPPED wall (real scanned wall lost < 1 % of the piece, D27; >= 0.1 % flagged); 1.5 = D27 bake repair window (capsule-clipped cells off both references) reported not gated, only for pieces failing under 1.4; 1.4 = D26 own scanned surface as the D22 reference where v4 hid it, every cell gated, + part_versions; 1.3 2026-10-06 s2_integrate2: 1.3 =D22 surface-preserved right-route gate (fork / siphon) replaces the relief-ratio gate (ratio kept as information) + coverage report, D23 tier-A acceptance, geom_v6 1.1; 1.2 2026-10-04 s2_harden_wall_bake; 1.1: D19/D20 route gate per real section on the moved surface + RCCA prefix gate; loader_one 1.1; 1.2: part geom = geom_v6 whole-tree clearance + 1 mm kinks)"
D22_MIN = 0.99
D22_DEV_MM = 0.3
D22_PAD_MM = 2.0
D22_END_TRIM_MM = 2.0
D22_WIN_PAD_MM = 0.5
ISO5 = "/repo/carotid_data/va_sources/v5_reg/iso_v5"
A5 = "/repo/carotid_data/anatomies_v5"
V4 = "/repo/carotid_data/anatomies_v4"
ZONES = os.environ.get("CHECKS_V6_ZONES", "/repo/carotid_data/v6_work/wall/seg/local")
HOST = "/repo/eve_bench/data/dualdevicenav/Centrelines_comb"
NEED = 0.65
ROUTE_MIN = 0.99
PARTS = ("item1", "route", "coverage", "folds", "lumen", "va", "orient", "loader", "geom")


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ------------------------------------------------------------------ route (D13)
def v4_union(name, drop_tip=False):
    """v4's union as rrd_v4.rebuild_v4 (mesh_v4.json: sections minus tube-only, margin, suppressed stretches, far
    clamp); drop_tip: RECA truncated at v4's ecaext_R join_at_mm and the ecaext_R section dropped."""
    import dataclasses
    from eve_bench.dualdevicenav import load_branches
    from sdf_mesher import SPACING_MM, iso_surface, tube_field
    from sdf_union_v4 import Samples, add_real_sections_owned
    root = os.path.join(V4, name)
    rep = jl(os.path.join(root, "mesh_v4.json"))
    prov = jl(os.path.join(root, "provenance.json"))
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    secs, _ = B.sections_for(root, br)
    j = None
    if drop_tip:
        j = float(prov["xform"]["ecaext_R"]["join_at_mm"])
        br2 = []
        for b in br:
            if str(b.name).endswith("RECA.mrk"):
                c = np.asarray(b.coordinates, float)
                k = B.arclen(c) <= j + 1e-6
                b = dataclasses.replace(b, coordinates=c[k], radii=np.asarray(b.radii)[k])
            br2.append(b)
        br = br2
    samp = Samples(br)
    tube_only = rep.get("sections_tube_only") or []
    secs = [s for s in secs if s.name not in tube_only and not (drop_tip and s.name == "ecaext_R")]
    spacing = rep.get("spacing_mm", SPACING_MM)
    margin = rep.get("margin_mm")
    sup = {k: [tuple(x) for x in v] for k, v in (rep.get("suppressed_stretches_mm") or {}).items()}
    field = tube_field(br, spacing=spacing)
    if margin is not None:
        add_real_sections_owned(field, secs, samp, spacing, margin, suppress=sup)
    if (rep.get("far_clamp") or {}).get("used"):
        far = field.values < B.FAR_BELOW
        field.values[far] = -2.0 * spacing
    iso, _, _ = B.keep_largest(iso_surface(field))
    V = np.asarray(iso.points, float)
    F = np.asarray(iso.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    return V, F, j


def reca_tip_status(prov6, prov4):
    """'same' | 'reposed' | 'skipped' | 'absent_v4' -- the v6 RECA TopBrain tip against v4's."""
    a, b = (prov6.get("xform") or {}).get("ecaext_R"), (prov4.get("xform") or {}).get("ecaext_R")
    if not b:
        return "absent_v4"
    if not a:
        return "skipped"
    for k in ("R", "origin", "anchor", "mirror"):
        x, y = np.asarray(a.get(k, [0.0]), float), np.asarray(b.get(k, [0.0]), float)
        if x.shape != y.shape or not np.allclose(x, y, atol=1e-9):
            return "reposed"
    if os.path.basename(str(a.get("surface"))) != os.path.basename(str(b.get("surface"))):
        return "reposed"
    if abs(float(a.get("join_at_mm", 0)) - float(b.get("join_at_mm", 0))) > 1e-6:
        return "reposed"
    return "same"


def declared_baseline(prov6):
    # the hardened carotid tool (s2_harden_carotid, route_baseline.py 1.0): provenance v6_carotid.summary.route_baseline
    v = ((prov6.get("v6_carotid") or {}).get("summary") or {}).get("route_baseline")
    if isinstance(v, dict):
        v = v.get("baseline") or v.get("used")
    if isinstance(v, str):
        return v
    for k in ("route_baseline", "d13"):
        v = prov6.get(k)
        if isinstance(v, dict):
            v = v.get("baseline") or v.get("used")
        if isinstance(v, str):
            return v
    for blk in ("v6_carotid", "carotid_v6", "v6"):
        x = prov6.get(blk)
        if isinstance(x, dict):
            for k in ("route_baseline", "d13_baseline", "baseline"):
                if isinstance(x.get(k), str):
                    return x[k]
    return None


def route_spans(prov):
    """the right route's real sections on the RCCA, as bake_meshes_v4.sections_for slices them:
    fork = lower_R [host_cut, host_cut + cca + min(ica, ica_real)], siphon = siphon_R [host_cut + cca + ica, end]."""
    lo, cca, ica = float(prov["host_cut_mm"]), float(prov["cca_mm"]), float(prov["ica_mm"])
    lx = prov["xform"]["lower"]
    return {"fork": (lo, lo + cca + min(ica, float(lx.get("ica_real_mm", ica)))), "siphon": (lo + cca + ica, 1.0e9)}


def _f3(v):
    return None if v is None else {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()}


def rcca_prefix(tree, name, cut6, cut4):
    """D19: the RCCA prefix (arch -> seam 1) stays host: v6's RCCA points / radii up to min(cut6, cut4) - 0.5 mm must be
    v4's (raw markup points); also against the host's own RCCA up to cut6 - 0.5 (information)."""
    from eve_bench.dualdevicenav import load_branches
    out = {}

    def rc(folder):
        for b in load_branches(folder):
            if str(b.name).endswith("RCCA.mrk"):
                return np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        return None, None
    p6, r6 = rc(os.path.join(tree, name, "Centrelines_comb"))
    p4, r4 = rc(os.path.join(V4, name, "Centrelines_comb"))
    s6 = B.arclen(p6)
    k = int(np.searchsorted(s6, min(cut6, cut4) - 0.5))
    same = len(p4) >= k and np.array_equal(p6[:k], p4[:k]) and np.array_equal(r6[:k], r4[:k])
    out["points_compared_v4"] = k
    out["identical_v4"] = bool(same)
    if not same and len(p4) >= k:
        out["max_diff_v4_mm"] = float(np.abs(p6[:k] - p4[:k]).max())
    try:
        ph, rh = rc(HOST)
        kh = int(np.searchsorted(s6, cut6 - 0.5))
        out["points_compared_host"] = kh
        out["identical_host"] = bool(len(ph) >= kh and np.array_equal(p6[:kh], ph[:kh]))
        if not out["identical_host"] and len(ph) >= kh:
            out["max_diff_host_mm"] = float(np.abs(p6[:kh] - ph[:kh]).max())
    except Exception as e:  # noqa: BLE001
        out["host_error"] = "%s: %s" % (type(e).__name__, e)
    return out


# ------------------------------------------------------------------ route D22 (1.3)
def _pt_tri(P, T):
    """distance from points P (n,3) to triangles T (n,3,3) (Ericson's closest point, vectorised)."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac, ap = b - a, c - a, P - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = P - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = P - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    Q = a + ab * v[:, None] + ac * w[:, None]
    reg = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (d6 >= 0) & (d5 <= d6)]
    Q = np.where(reg[0][:, None], a, Q)
    Q = np.where(reg[1][:, None], b, Q)
    Q = np.where(reg[2][:, None], c, Q)
    e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~reg[0] & ~reg[1]
    t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    Q = np.where(e_ab[:, None], a + ab * t[:, None], Q)
    e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~reg[0] & ~reg[2]
    t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    Q = np.where(e_ac[:, None], a + ac * t[:, None], Q)
    e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~reg[1] & ~reg[2]
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
    Q = np.where(e_bc[:, None], b + (c - b) * t[:, None], Q)
    return np.linalg.norm(P - Q, axis=1)


class SurfDist:
    """exact distance from points to a triangle mesh (candidates: the k nearest triangle centroids)."""

    def __init__(self, V, F, k=12):
        self.V, self.F = np.asarray(V, float), np.asarray(F, np.int64)
        C = self.V[self.F].mean(axis=1)
        self.tree, self.k = cKDTree(C), min(k, len(C))

    def __call__(self, X):
        X = np.asarray(X, float)
        if not len(X):
            return np.zeros(0)
        _, J = self.tree.query(X, k=self.k)
        if J.ndim == 1:
            J = J[:, None]
        best = np.full(len(X), np.inf)
        for c in range(J.shape[1]):
            best = np.minimum(best, _pt_tri(X, self.V[self.F[J[:, c]]]))
        return best


def piece_map(xf4, xf6):
    """v4 world -> v6 world through the piece's source frame (load_surface: x = (src * M - o) R^T + a)."""
    R4, o4, a4 = (np.asarray(xf4[k], float) for k in ("R", "origin", "anchor"))
    R6, o6, a6 = (np.asarray(xf6[k], float) for k in ("R", "origin", "anchor"))
    M4 = np.asarray(xf4.get("mirror", [1, 1, 1]), float)
    M6 = np.asarray(xf6.get("mirror", [1, 1, 1]), float)
    return lambda X: (((np.asarray(X, float) - a4) @ R4 + o4) * M4 * M6 - o6) @ R6.T + a6


def v4_occupied(X4, samp4, rc4, sib, margin4):
    """D22 reference validity (1.3): True where, in v4's world, ANOTHER branch (siblings -- the fork's own ECA -- left out)
    comes within the union's ownership margin (+ D22_DEV_MM) of the RCCA at the point X4: v4's union did not show the
    piece's own wall there (e.g. v4's RVA top fused on k_011_left mr_012_L's right siphon), so v4 is no reference for it
    (as D13 for v4's tip). Tube values on the dense samples (measure.Samp), K nearest per branch."""
    X4 = np.asarray(X4, float)
    out = np.zeros(len(X4), bool)
    if not len(X4):
        return out, np.zeros(0), np.zeros(0)

    def fmax(bi, k):
        m = samp4.b == bi
        P, R = samp4.p[m], samp4.r[m]
        kk = min(k, len(P))
        d, i = cKDTree(P).query(X4, k=kk)
        if d.ndim == 1:
            d, i = d[:, None], i[:, None]
        return (R[i] - d).max(axis=1)
    f_rc = fmax(rc4, 48)
    f_ot = np.full(len(X4), -np.inf)
    for bi in np.unique(samp4.b):
        if bi == rc4 or samp4.tags[bi] in sib:
            continue
        f_ot = np.maximum(f_ot, fmax(bi, 16))
    return f_ot > f_rc - margin4 - D22_DEV_MM, f_rc, f_ot


def _in_windows(s, wins, pad=D22_WIN_PAD_MM):
    m = np.zeros(len(s), bool)
    for a, b in wins:
        m |= (s >= float(a) - pad) & (s <= float(b) + pad)
    return m


D27_SEC = {"fork": "lower_R", "siphon": "siphon_R"}


def bake_capsule_clip(root, sec_name, X):
    """1.5 (D27): the bake's own section field of `sec_name` in anatomy folder `root` (bake_v6's sections as baked:
    B.sections_for = sections_for_morph) at points X of that folder's world: f_real (> 0 inside the scanned lumen, mm),
    the capsule value (sdf_union._capsule_radii round the section's own centreline, resampled and k-NN as
    sdf_mesher.tube_field evaluates it) and clip = capsule < f_real (the bake's field there is the capsule, not the real
    wall: f_sec = min(f_real, capsule))."""
    import pyvista as pv
    import sdf_mesher as SM
    from sdf_union import _capsule_radii
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    secs, _ = B.sections_for(root, br)
    sec = [s for s in secs if s.name == sec_name]
    if not sec:
        raise RuntimeError("bake section %s not found in %s" % (sec_name, root))
    sec = sec[0]
    cap_r = _capsule_radii(sec.points, sec.radii, sec.taper_start, sec.taper_end)
    c, r = SM._resample(np.asarray(sec.points, float), np.asarray(cap_r, float), SM.SAMPLE_MM)
    dist, nn = cKDTree(c).query(X, k=min(SM.K_NN, len(c)))
    if dist.ndim == 1:
        dist, nn = dist[:, None], nn[:, None]
    capv = (r[nn] - dist).max(axis=1)
    f_real = -np.asarray(pv.PolyData(np.asarray(X, float)).compute_implicit_distance(sec.surface)["implicit_distance"])
    return f_real, capv, capv < f_real


def _st(v):
    v = np.asarray(v, float)
    return None if not len(v) else {"min": round(float(v.min()), 3), "median": round(float(np.median(v)), 3),
                                     "max": round(float(v.max()), 3)}


def route_d22(prov6, prov4, rep6, rep4, s6, s4b, O6, O4b, C6, A6, C4b, A4b, V4b, F4b, V6, F6, base, root6=None, root4=None):
    """D22 per piece: the baked RCCA-owned cells beyond the seam zone vs v4's (baseline) union carried by the piece's
    own transform. s6 / s4b: measure.Samp of v6 / the baseline; O6 / O4b: exact owners; C*, A*: cell centres and
    areas; V4b / F4b: the baseline union; V6 / F6: the v6 union."""
    out = {"baseline": base, "min": D22_MIN, "dev_mm": D22_DEV_MM}
    rc6, rc4 = s6.idx("RCCA"), s4b.idx("RCCA")
    sp6, sp4 = route_spans(prov6), route_spans(prov4)
    P6 = s6.p[s6.b == rc6]
    S6 = s6.s[s6.b == rc6]
    L6 = float(S6.max())
    P4 = s4b.p[s4b.b == rc4]
    S4 = s4b.s[s4b.b == rc4]
    L4 = float(S4.max())
    t6 = cKDTree(P6)
    sd6 = SurfDist(V6, F6)
    sup6 = rep6.get("suppressed_stretches_mm") or {}
    sup4 = rep4.get("suppressed_stretches_mm") or {}
    for piece, key, sec in (("fork", "lower", "lower_R"), ("siphon", "siphon", "siphon_R")):
        r = {}
        try:
            xf6, xf4 = prov6["xform"][key], prov4["xform"][key]
            r["surface_v6"], r["surface_v4"] = os.path.basename(str(xf6.get("surface"))), os.path.basename(str(xf4.get("surface")))
            r["same_surface"] = r["surface_v6"] == r["surface_v4"]
            r["mirror_v6"], r["mirror_v4"] = xf6.get("mirror", [1, 1, 1]), xf4.get("mirror", [1, 1, 1])
            T = piece_map(xf4, xf6)
            lo6, hi6 = sp6[piece][0], min(sp6[piece][1], L6)
            lo4, hi4 = sp4[piece][0], min(sp4[piece][1], L4)
            z = max(float(xf6.get("blend_mm") or 0.0), float(xf4.get("blend_mm") or 0.0)) + D22_PAD_MM
            g6 = (lo6 + z, hi6 - D22_END_TRIM_MM)
            zone6 = (lo6, lo6 + z)
            # arclength offset of the rigid part (v4 RCCA points beyond the zone carried onto the v6 RCCA)
            k4 = (S4 >= max(lo4 + z + 3.0, hi4 - 40.0 if piece == "fork" else lo4 + z + 3.0)) & (S4 <= hi4 - 3.0)
            dd, jj = t6.query(T(P4[k4]))
            off = float(np.median(S6[jj] - S4[k4])) if k4.any() else float(hi6 - hi4)
            r.update(span_v6=[round(lo6, 3), round(hi6, 3)], span_v4=[round(lo4, 3), round(hi4, 3)], zone_mm=round(z, 3),
                     gate_v6=[round(g6[0], 3), round(g6[1], 3)], arclength_offset_mm=round(off, 4),
                     rcca_carried_max_mm=round(float(dd.max()), 4) if len(dd) else None,
                     rcca_carried_p95_mm=round(float(np.percentile(dd, 95)), 4) if len(dd) else None)
            win6 = [list(w) for w in (sup6.get(sec) or [])]
            win4 = [[float(a) + off, float(b) + off] for a, b in (sup4.get(sec) or [])]
            wins = win6 + win4
            r["handle_windows_v6"], r["handle_windows_v4_mapped"] = win6, [[round(a, 2), round(b, 2)] for a, b in win4]
            V4c = T(V4b)
            sd4 = SurfDist(V4c, F4b)
            own = O6.b == rc6
            m_all = own & (O6.s >= g6[0]) & (O6.s <= g6[1]) & ~_in_windows(O6.s, wins)
            # D22 reference validity: cells whose v4 wall was occupied by another branch are left out (reported)
            m4 = float(rep4.get("margin_mm") or 0.5)
            Ti = piece_map(xf6, xf4)
            occ = np.zeros(len(O6.s), bool)
            ia = np.nonzero(m_all)[0]
            occ_a, _, _ = v4_occupied(Ti(C6[ia]), s4b, rc4, ("RECA",) if piece == "fork" else (), m4)
            occ[ia] = occ_a
            m_gate = m_all & ~occ
            m_occ = m_all & occ
            m_zone = own & (O6.s >= zone6[0]) & (O6.s < zone6[1])
            m_win = own & (O6.s >= lo6) & (O6.s <= hi6) & _in_windows(O6.s, wins)
            # 1.4 (D26): the piece's OWN scanned (source) surface carried by its v6 transform (the bake's rigid placement)
            srcm = BK.load_surface(xf6["surface"], xf6.get("mirror", [1, 1, 1]), xf6["R"], xf6["origin"], xf6["anchor"])
            Vs = np.asarray(srcm.points, float)
            Fs = np.asarray(srcm.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
            sds = SurfDist(Vs, Fs)
            r["own_surface"] = {"path": str(xf6["surface"]), "mirror": xf6.get("mirror", [1, 1, 1]), "tris": int(len(Fs))}
            gsc = None
            for nm, m in (("gate", m_gate), ("gate_strict", m_all), ("v4_occupied", m_occ), ("zone", m_zone),
                          ("windows", m_win)):
                if not m.any():
                    r[nm] = None
                    continue
                d = sd4(C6[m])
                do = sds(C6[m])
                a = A6[m]
                ok4, oko = d <= D22_DEV_MM, do <= D22_DEV_MM
                if nm == "gate_strict":            # 1.5 (D27): kept for the repair-window test
                    gsc = (np.nonzero(m)[0], a, ok4, oko)
                r[nm] = {"area_mm2": round(float(a.sum()), 2), "cells": int(m.sum()),
                         "within_0.3": round(float(a[ok4].sum() / a.sum()), 6),
                         "p50_mm": round(float(np.median(d)), 4), "p95_mm": round(float(np.percentile(d, 95)), 4),
                         "max_mm": round(float(d.max()), 3),
                         "area_gt_0.3_mm2": round(float(a[~ok4].sum()), 2),
                         "within_0.3_own": round(float(a[ok4 | oko].sum() / a.sum()), 6),
                         "area_off_v4_on_own_mm2": round(float(a[~ok4 & oko].sum()), 2),
                         "area_off_both_mm2": round(float(a[~ok4 & ~oko].sum()), 2),
                         "max_off_both_own_mm": round(float(do[~ok4 & ~oko].max()), 3) if (~ok4 & ~oko).any() else None}
            # coverage (reported): v4's own RCCA-owned cells of the same stretch, carried, vs the v6 union
            own4 = O4b.b == rc4
            g4 = (g6[0] - off, g6[1] - off)
            wins4 = [[a - off, b - off] for a, b in wins]
            mc = own4 & (O4b.s >= g4[0]) & (O4b.s <= g4[1]) & ~_in_windows(O4b.s, wins4)
            ic = np.nonzero(mc)[0]
            if len(ic):
                oc_, _, _ = v4_occupied(C4b[ic], s4b, rc4, ("RECA",) if piece == "fork" else (), m4)
                mc[ic[oc_]] = False
            if mc.any():
                d = sd6(T(C4b[mc]))
                a = A4b[mc]
                r["coverage"] = {"area_mm2": round(float(a.sum()), 2), "within_0.3": round(float(a[d <= D22_DEV_MM].sum() / a.sum()), 6),
                                 "p95_mm": round(float(np.percentile(d, 95)), 4), "max_mm": round(float(d.max()), 3),
                                 "area_gt_0.3_mm2": round(float(a[d > D22_DEV_MM].sum()), 2)}
            # 1.4 (D26): every cell gated; reference v4, else (FLAGGED) the piece's own scanned surface where v4 hid it,
            # provided no real wall is lost (coverage >= 0.99)
            gs = r.get("gate_strict") or {}
            r["value_1p3"] = (r.get("gate") or {}).get("within_0.3")
            r["value_v4"] = gs.get("within_0.3")
            r["value_own"] = gs.get("within_0.3_own")
            r["value_strict"] = r["value_v4"]
            cov = (r.get("coverage") or {}).get("within_0.3")
            r["no_wall_lost"] = bool(cov is None or cov >= D22_MIN)
            if r["value_v4"] is not None and r["value_v4"] >= D22_MIN:
                ref = "v4"
            elif r["value_own"] is not None and r["value_own"] >= D22_MIN and r["no_wall_lost"]:
                ref = "own_surface"
            else:
                ref = "none"
            r["d22_reference"] = ref
            r["d26_flag"] = ref == "own_surface"
            r["value"] = r["value_v4"] if ref == "v4" else r["value_own"]
            r["ok"] = bool(ref != "none" and r["value"] is not None)
            r["strict_flag"] = bool(r["value_strict"] is not None and r["value_strict"] < D22_MIN)
            r["coverage_flag"] = bool((r.get("coverage") or {}).get("within_0.3", 1.0) < D22_MIN)
            # 1.5 (D27): ONLY for a piece failing under 1.4 -- the bake's repair window (capsule-clipped cells off both
            # references) is reported and left out of the gated area; the reference is re-decided as in 1.4
            r["d27_flag"] = False
            if ref == "none" and root6 is not None and gsc is not None:
                ig, a_, ok4_, oko_ = gsc
                offb = ~ok4_ & ~oko_
                rw = np.zeros(len(ig), bool)
                rec27 = {"rule": "D27 (USER 2026-10-07 16:25): bake repair window reported, not gated", "section": D27_SEC[piece],
                         "off_both_cells": int(offb.sum()), "off_both_area_mm2": round(float(a_[offb].sum()), 2)}
                if offb.any():
                    X = C6[ig[offb]]
                    f_real, capv, clip = bake_capsule_clip(root6, D27_SEC[piece], X)
                    rw[np.nonzero(offb)[0][clip]] = True
                    rec27.update(cells=int(rw.sum()), area_mm2=round(float(a_[rw].sum()), 2),
                                 share_of_off_both_area=round(float(a_[rw].sum() / a_[offb].sum()), 4),
                                 s_range_mm=[round(float(O6.s[ig[rw]].min()), 2), round(float(O6.s[ig[rw]].max()), 2)] if rw.any() else None,
                                 lumen_beyond_capsule_mm=_st((f_real - capv)[clip]), f_real_mm=_st(f_real[clip]),
                                 capsule_value_mm=_st(capv[clip]), dist_own_surface_mm=_st(sds(X[clip])))
                    if root4 is not None:          # information: the same cells carried into v4's world, v4's bake section
                        try:
                            _, _, clip4 = bake_capsule_clip(root4, D27_SEC[piece], Ti(X))
                            rec27["clipped_in_v4_world_frac"] = round(float(clip4.mean()), 4)
                        except Exception as e4:  # noqa: BLE001
                            rec27["clipped_in_v4_world_error"] = "%s: %s" % (type(e4).__name__, e4)
                keep = ~rw
                ak = a_[keep]
                v4k = round(float(ak[ok4_[keep]].sum() / ak.sum()), 6) if ak.sum() > 0 else None
                vok = round(float(ak[(ok4_ | oko_)[keep]].sum() / ak.sum()), 6) if ak.sum() > 0 else None
                rec27.update(value_v4_without_window=v4k, value_own_without_window=vok)
                if rw.any():
                    if v4k is not None and v4k >= D22_MIN:
                        ref27 = "v4"
                    elif vok is not None and vok >= D22_MIN and r["no_wall_lost"]:
                        ref27 = "own_surface"
                    else:
                        ref27 = "none"
                    rec27["reference_after"] = ref27
                    if ref27 != "none":
                        r["value_before_d27"] = r["value_own"]
                        ref = ref27
                        r["d22_reference"] = ref
                        r["d26_flag"] = ref == "own_surface"
                        r["value"] = v4k if ref == "v4" else vok
                        r["ok"] = True
                        r["d27_flag"] = True
                r["d22_repair_window"] = rec27
        except Exception as e:  # noqa: BLE001
            r = {"error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc()[-1500:], "ok": False, "value": None}
        out[piece] = r
    out["reference"] = {p: (out.get(p) or {}).get("d22_reference") for p in ("fork", "siphon")}
    out["d26_flag"] = any((out.get(p) or {}).get("d26_flag") for p in ("fork", "siphon"))
    out["d27_flag"] = any((out.get(p) or {}).get("d27_flag") for p in ("fork", "siphon"))   # 1.5 (D27)
    return out


def part_route(name, tree, iso6, rep6):
    prov6 = jl(os.path.join(tree, name, "provenance.json"))
    prov4 = jl(os.path.join(V4, name, "provenance.json"))
    rep4 = jl(os.path.join(V4, name, "mesh_v4.json"))
    cut6, cut4 = float(prov6.get("host_cut_mm", 0)), float(prov4.get("host_cut_mm", 0))
    s6 = M.Samp(M.read_branches(os.path.join(tree, name, "Centrelines_comb")))
    br4 = M.read_branches(os.path.join(V4, name, "Centrelines_comb"))
    s4 = M.Samp(br4)
    V4_, F4_, _ = v4_union(name)
    V6, F6 = M.read_vtp(os.path.join(iso6, name + ".vtp"))
    C6, A6_ = M.tri_geom(V6, F6)
    C4, A4_ = M.tri_geom(V4_, F4_)
    va = np.vstack([s.p[(s.b == s.idx("RVA")) | (s.b == s.idx("LVA"))] for s in (s4, s6)])
    vt = cKDTree(va)
    rec = {"cut6": cut6, "cut4": cut4, "mesh_v6_record_k24": rep6.get("right_route_detail"),
           "mesh_v4_record": rep4.get("right_route_detail")}
    sp6, sp4 = route_spans(prov6), route_spans(prov4)
    rec["sections"] = {"spans_v6": sp6, "spans_v4": sp4,
                       "tube_only_v6": [k for k in ("lower_R", "siphon_R") if k in (rep6.get("sections_tube_only") or [])],
                       "tube_only_v4": [k for k in ("lower_R", "siphon_R") if k in (rep4.get("sections_tube_only") or [])]}
    rec["rcca_prefix"] = rcca_prefix(tree, name, cut6, cut4)
    sec6x = None
    for mode in ("k24", "exact"):
        O6 = M.Own(s6, C6, A6_, mode)
        O4 = M.Own(s4, C4, A4_, mode)
        sec6 = {g: O6.seg(s6.idx("RCCA"), lo, hi) for g, (lo, hi) in sp6.items()}
        sec4 = {g: O4.seg(s4.idx("RCCA"), lo, hi) for g, (lo, hi) in sp4.items()}
        rec["sections"]["v6_" + mode] = {g: _f3(v) for g, v in sec6.items()}
        rec["sections"]["v4_" + mode] = {g: _f3(v) for g, v in sec4.items()}
        rec["sections"]["ratio_" + mode] = {g: (sec6[g]["frac_0.3mm"] / sec4[g]["frac_0.3mm"]
                                                if sec6.get(g) and sec4.get(g) and sec4[g]["frac_0.3mm"] else None)
                                            for g in ("fork", "siphon")}
        if mode == "exact":
            sec6x = sec6
        r6, r4 = M.rrd(O6, s6, cut6), M.rrd(O4, s4, cut4)
        _, st6, _ = RX.route_sets(O6, s6, cut6, C6, A6_, vt)
        _, st4, _ = RX.route_sets(O4, s4, cut4, C4, A4_, vt)
        ratio = {g: {t: (r6[g][t] / r4[g][t] if r4[g][t] else None) for t in ("frac_0.3mm", "frac_0.5mm")} for g in r6}
        rec[mode] = {"rrd_v6": M.r4(r6), "rrd_v4": M.r4(r4), "ratio": ratio,
                     "beyond_8mm_of_VAs": {g: {"v6": st6["%s_beyond_8" % g]["frac_0.3mm"], "v4": st4["%s_beyond_8" % g]["frac_0.3mm"]}
                                           for g in ("fork", "siphon")},
                     "area_mm2": {g: {"v6": r6[g]["area_mm2"], "v4": r4[g]["area_mm2"]} for g in r6}}
        if mode == "exact":
            r6_exact = r6
            O6x, O4x = O6, O4
    # (1.3: the v4 union is kept for D22 below)
    tip = reca_tip_status(prov6, prov4)
    d13 = {"reca_tip_v6_vs_v4": tip, "declared": declared_baseline(prov6)}
    base = "v4"
    if tip in ("reposed", "skipped") or d13["declared"] in ("v4_no_tip", "v4-without-tip", "no_tip"):
        Vn, Fn, j = v4_union(name, drop_tip=True)
        brn = dict(br4)
        p, r = brn["RECA"]
        k = M.arclen(p) <= j + 1e-6
        brn["RECA"] = (p[k], r[k])
        sn = M.Samp(brn)
        Cn, An = M.tri_geom(Vn, Fn)
        On = M.Own(sn, Cn, An, "exact")
        rn = M.rrd(On, sn, cut4)
        d13["rrd_v4_without_tip"] = M.r4(rn)
        d13["join_at_mm_v4"] = j
        d13["ratio_vs_v4_without_tip"] = {g: (r6_exact[g]["frac_0.3mm"] / rn[g]["frac_0.3mm"] if rn[g]["frac_0.3mm"] else None)
                                          for g in ("fork", "siphon")}
        r4x = rec["exact"]["rrd_v4"]
        differs = any(abs(r4x[g]["frac_0.3mm"] - rn[g]["frac_0.3mm"]) > 5e-4 for g in ("fork", "siphon"))
        d13["v4_tip_owned_rcca_wall"] = bool(differs)
        # 1.1: the per-section measure on v4 without its tip too
        secn = {g: On.seg(sn.idx("RCCA"), lo, hi) for g, (lo, hi) in sp4.items()}
        d13["sections_v4_without_tip"] = {g: _f3(v) for g, v in secn.items()}
        d13["ratio_sections_vs_v4_without_tip"] = {
            g: (sec6x[g]["frac_0.3mm"] / secn[g]["frac_0.3mm"] if sec6x.get(g) and secn.get(g) and secn[g]["frac_0.3mm"] else None)
            for g in ("fork", "siphon")}
        if d13["declared"] in ("v4", "v4_shipped"):
            base = "v4"
        elif differs or d13["declared"] in ("v4_no_tip", "v4-without-tip", "no_tip"):
            base = "v4_no_tip"
    d13["baseline_used"] = base
    rec["d13"] = d13
    used = rec["exact"]["ratio"] if base == "v4" else {g: {"frac_0.3mm": d13["ratio_vs_v4_without_tip"][g]} for g in ("fork", "siphon")}
    # 1.0's gate (RCCA-owned wall split at 130 mm, positions) kept as information
    rec["gate_ratio_legacy_130"] = {g: used[g]["frac_0.3mm"] for g in ("fork", "siphon")}
    # 1.1 (D19 / D20): the gate is each real section's OWN wall detail (its own slice of the RCCA, exact ownership,
    # 2 mm trims) against the same section in the baseline -- the moved (re-posed / mirrored) fork is compared with
    # itself, not with v4's positions
    rec["gate_ratio"] = (rec["sections"]["ratio_exact"] if base == "v4" else d13["ratio_sections_vs_v4_without_tip"])
    rec["gate_basis"] = "per section (fork = lower_R slice [host_cut, host_cut + cca + min(ica, ica_real)], siphon = siphon_R slice), exact ownership, baseline %s" % base
    p5 = "/sp/v5_va/swap13/rrdx/%s.json" % name
    if os.path.exists(p5):
        rec["v5_exact_ratio"] = jl(p5)["exact"]["ratio"]
    # 1.3 (D22): the gate = surface preserved, against the baseline union (v4, or v4 without its tip: D13)
    if base == "v4_no_tip":
        b_ = (sn, On, Cn, An, Vn, Fn)
    else:
        b_ = (s4, O4x, C4, A4_, V4_, F4_)
    rec["d22"] = route_d22(prov6, prov4, rep6, rep4, s6, b_[0], O6x, b_[1], C6, A6_, b_[2], b_[3], b_[4], b_[5], V6, F6, base,
                           root6=os.path.join(tree, name), root4=os.path.join(V4, name))
    rec["gate_ratio_note"] = "information since checks_v6 1.3 (the gate is D22: rec.d22)"
    return rec


# ------------------------------------------------------------------ coverage against the v4 / v5 SHIPPED wall (1.6)
COV_REFS = (("v4", V4), ("v5", A5))
COV_LOSS_MAX = 0.01            # D27: a real-wall deficit >= 1 % of the piece fails
COV_FLAG_MIN = 0.001           # a deficit >= 0.1 % of the piece (and < 1 %) is accepted with a flag (D27)


def _runs(s, step=0.5):
    s = np.sort(np.asarray(s, float))
    if not len(s):
        return []
    out = [[s[0], s[0]]]
    for v in s[1:]:
        if v - out[-1][1] <= step:
            out[-1][1] = v
        else:
            out.append([v, v])
    return [[round(float(a), 2), round(float(b), 2)] for a, b in out]


def part_coverage(name, tree, iso6, rep6):
    """1.6 (s3_fix; final verifier's must-fix 1, USER rule D27): COVERAGE AGAINST THE v4 / v5 SHIPPED WALL.
    D22 gates one direction (every v6 RCCA cell lies on v4's surface or on the piece's own scan); this part gates the
    other. Per right-route piece (fork = lower_R slice, siphon = siphon_R slice) and per reference (v4's and v5's SHIPPED
    collision_full.vtp, each in its own world): the reference's cells exactly owned by ITS OWN RCCA in its gated stretch
    (span start + max(blend v6, blend ref) + 2 mm .. span end - 2 mm), carried to v6 by the piece's own transforms
    (piece_map: mirror included), kept where their v6 RCCA arclength lies in v6's gated stretch and outside v6's bake
    handle windows (+- 0.5 mm; D22 reports them, does not gate them). A carried cell is COVERED when the v6 union lies
    within 0.3 mm of it (exact point-to-triangle distance). An uncovered cell is LOST REAL WALL when it lies on the piece's
    OWN scanned surface (<= 0.3 mm of the source surface placed by the v6 xform) and outside the v6 union (an uncovered
    cell inside the union is a neighbour's wall the reference showed -- D26's situation -- not wall lost).
    deficit = lost area / gated reference area. Gate route_<piece>_coverage_loss_lt_1pct: deficit < 1 % on both
    references; a deficit >= 0.1 % (and < 1 %) is accepted and flagged (D27). Reported: coverage, uncovered and lost
    area, their v6 RCCA arclength runs, depth (distance to the v6 union)."""
    import pyvista as pv
    prov6 = jl(os.path.join(tree, name, "provenance.json"))
    s6 = M.Samp(M.read_branches(os.path.join(tree, name, "Centrelines_comb")))
    rc6 = s6.idx("RCCA")
    P6, S6 = s6.p[s6.b == rc6], s6.s[s6.b == rc6]
    t6 = cKDTree(P6)
    V6, F6 = M.read_vtp(os.path.join(iso6, name + ".vtp"))
    sd6 = SurfDist(V6, F6)
    U6 = pv.PolyData(np.asarray(V6, float), np.c_[np.full(len(F6), 3), F6].ravel())
    sup6 = rep6.get("suppressed_stretches_mm") or {}
    sp6 = route_spans(prov6)
    L6 = float(S6.max())
    out = {"dev_mm": D22_DEV_MM, "loss_max": COV_LOSS_MAX, "flag_min": COV_FLAG_MIN, "union": os.path.join(iso6, name + ".vtp")}
    own_sd = {}
    for ref, troot in COV_REFS:
        root = os.path.join(troot, name)
        rr = {}
        try:
            provR = jl(os.path.join(root, "provenance.json"))
            sR = M.Samp(M.read_branches(os.path.join(root, "Centrelines_comb")))
            rcR = sR.idx("RCCA")
            LR = float(sR.s[sR.b == rcR].max())
            VR, FR = M.read_vtp(os.path.join(root, "collision_full.vtp"))
            CR, AR = M.tri_geom(VR, FR)
            OR = M.Own(sR, CR, AR, "exact")
            spR = route_spans(provR)
            for piece, key, sec in (("fork", "lower", "lower_R"), ("siphon", "siphon", "siphon_R")):
                r = {}
                xf6, xfR = prov6["xform"][key], provR["xform"][key]
                r["surface_v6"], r["surface_ref"] = os.path.basename(str(xf6.get("surface"))), os.path.basename(str(xfR.get("surface")))
                if r["surface_v6"] != r["surface_ref"]:
                    r.update(skipped="the reference piece is a different scanned surface", deficit=None, ok=True)
                    rr[piece] = r
                    continue
                T = piece_map(xfR, xf6)
                z = max(float(xf6.get("blend_mm") or 0.0), float(xfR.get("blend_mm") or 0.0)) + D22_PAD_MM
                g6 = (sp6[piece][0] + z, min(sp6[piece][1], L6) - D22_END_TRIM_MM)
                gR = (spR[piece][0] + z, min(spR[piece][1], LR) - D22_END_TRIM_MM)
                wins = [list(w) for w in (sup6.get(sec) or [])]
                m = (OR.b == rcR) & (OR.s >= gR[0]) & (OR.s <= gR[1])
                X, A = T(CR[m]), AR[m]
                _, j = t6.query(X)
                sc = S6[j]
                keep = (sc >= g6[0]) & (sc <= g6[1]) & ~_in_windows(sc, wins)
                X, A, sc = X[keep], A[keep], sc[keep]
                r.update(gate_v6=[round(g6[0], 3), round(g6[1], 3)], gate_ref=[round(gR[0], 3), round(gR[1], 3)],
                         handle_windows_v6=wins, mirror_v6=xf6.get("mirror", [1, 1, 1]), mirror_ref=xfR.get("mirror", [1, 1, 1]),
                         cells=int(len(X)), area_mm2=round(float(A.sum()), 2))
                if not len(X):
                    r.update(deficit=None, ok=True, note="no reference cells in the gated stretch")
                    rr[piece] = r
                    continue
                d = sd6(X)
                unc = d > D22_DEV_MM
                r["coverage"] = round(float(A[~unc].sum() / A.sum()), 6)
                r["uncovered_mm2"] = round(float(A[unc].sum()), 2)
                lost = np.zeros(len(X), bool)
                if unc.any():
                    if sec not in own_sd:
                        srcm = BK.load_surface(xf6["surface"], xf6.get("mirror", [1, 1, 1]), xf6["R"], xf6["origin"], xf6["anchor"])
                        own_sd[sec] = SurfDist(np.asarray(srcm.points, float),
                                               np.asarray(srcm.faces).reshape(-1, 4)[:, 1:].astype(np.int64))
                    iu = np.nonzero(unc)[0]
                    on = own_sd[sec](X[iu]) <= D22_DEV_MM
                    ins = np.asarray(pv.PolyData(X[iu]).select_enclosed_points(U6, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
                    lost[iu[on & ~ins]] = True
                    r["uncovered"] = {"on_own_scan_mm2": round(float(A[iu][on].sum()), 2),
                                      "inside_v6_union_mm2": round(float(A[iu][ins].sum()), 2),
                                      "s_runs_v6": _runs(sc[iu]), "depth_mm": _st(d[iu])}
                r["lost_mm2"] = round(float(A[lost].sum()), 2)
                r["deficit"] = round(float(A[lost].sum() / A.sum()), 6)
                if lost.any():
                    r["lost"] = {"s_runs_v6": _runs(sc[lost]), "depth_mm": _st(d[lost]),
                                 "near_window_2mm_mm2": round(float(A[lost & _in_windows(sc, wins, 2.0)].sum()), 2)}
                r["ok"] = bool(r["deficit"] < COV_LOSS_MAX)
                r["flag"] = bool(COV_FLAG_MIN <= r["deficit"] < COV_LOSS_MAX)
                rr[piece] = r
        except Exception as e:  # noqa: BLE001
            rr = {"error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc()[-1500:]}
        out[ref] = rr
    for piece in ("fork", "siphon"):
        ds = [(out[ref].get(piece) or {}).get("deficit") for ref, _ in COV_REFS]
        errs = [ref for ref, _ in COV_REFS if "error" in out[ref]]
        okp = not errs and all((out[ref].get(piece) or {}).get("ok", False) for ref, _ in COV_REFS)
        dd = [x for x in ds if x is not None]
        out[piece] = {"deficit_max": max(dd) if dd else None, "ok": bool(okp),
                      "flag": bool(okp and dd and max(dd) >= COV_FLAG_MIN),
                      "lost_mm2_max": max([(out[ref].get(piece) or {}).get("lost_mm2") or 0.0 for ref, _ in COV_REFS])}
    return out


FOLD_COS = float(np.cos(np.radians(1.0)))


def fold_pairs(V, F):
    """1.6: fold-over pairs = EDGE-ADJACENT triangles whose normals are opposite within 1 deg (the two faces lie on each
    other; MeshLab's self-intersection filter skips edge-sharing faces, so item1 cannot see them)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fi = np.tile(np.arange(len(F)), 3)
    key = np.sort(E, 1)
    o = np.lexsort((key[:, 1], key[:, 0]))
    key, fi = key[o], fi[o]
    i = np.nonzero(np.all(key[1:] == key[:-1], 1))[0]
    f1, f2 = fi[i], fi[i + 1]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    k = (n[f1] * n[f2]).sum(1) < -FOLD_COS
    return [{"faces": [int(a), int(b)], "at": np.round(V[list(key[i][j])].mean(0), 2).tolist()}
            for a, b, j in zip(f1[k], f2[k], np.nonzero(k)[0])]


def part_folds(name, tree):
    """1.6 (s3_fix; final verifier's should-fix 3): fold-over pairs on the training OBJ (gate obj_no_fold_over: 0) and on
    collision_full (information: v4 / v5 / v6 each carry one there on some anatomy)."""
    import pyvista as pv
    root = os.path.join(tree, name)
    out = {}
    for tag, fn in (("obj", B.MESH_NAME), ("full", B.FULL_NAME)):
        m = pv.read(os.path.join(root, fn))
        fp = fold_pairs(np.asarray(m.points, float), np.asarray(m.faces).reshape(-1, 4)[:, 1:])
        out[tag] = {"fold_pairs": len(fp), "pairs": fp[:10]}
    return out


def part_route_d22(name, tree, iso6, rep6, rt):
    """1.4: ONLY the D22 / D26 part of part route (see the 1.4 note), on an anatomy whose geometry and bake are unchanged
    since rt (its earlier route record) was computed. Same inputs and code path as part_route's route_d22 call."""
    d13 = rt.get("d13") or {}
    base = d13.get("baseline_used")
    if base not in ("v4", "v4_no_tip"):
        raise RuntimeError("carried route record has no D13 baseline decision")
    prov6 = jl(os.path.join(tree, name, "provenance.json"))
    prov4 = jl(os.path.join(V4, name, "provenance.json"))
    rep4 = jl(os.path.join(V4, name, "mesh_v4.json"))
    tip, decl = reca_tip_status(prov6, prov4), declared_baseline(prov6)
    if tip != d13.get("reca_tip_v6_vs_v4") or decl != d13.get("declared"):
        raise RuntimeError("D13 inputs differ from the carried route record: tip %s/%s declared %s/%s"
                           % (tip, d13.get("reca_tip_v6_vs_v4"), decl, d13.get("declared")))
    s6 = M.Samp(M.read_branches(os.path.join(tree, name, "Centrelines_comb")))
    br4 = M.read_branches(os.path.join(V4, name, "Centrelines_comb"))
    V6, F6 = M.read_vtp(os.path.join(iso6, name + ".vtp"))
    C6, A6_ = M.tri_geom(V6, F6)
    O6x = M.Own(s6, C6, A6_, "exact")
    if base == "v4_no_tip":
        Vb, Fb, j = v4_union(name, drop_tip=True)
        brn = dict(br4)
        p, r = brn["RECA"]
        k = M.arclen(p) <= j + 1e-6
        brn["RECA"] = (p[k], r[k])
        sb = M.Samp(brn)
    else:
        Vb, Fb, _ = v4_union(name)
        sb = M.Samp(br4)
    Cb, Ab = M.tri_geom(Vb, Fb)
    Ob = M.Own(sb, Cb, Ab, "exact")
    return route_d22(prov6, prov4, rep6, rep4, s6, sb, O6x, Ob, C6, A6_, Cb, Ab, Vb, Fb, V6, F6, base,
                     root6=os.path.join(tree, name), root4=os.path.join(V4, name))


# ------------------------------------------------------------------ lumen
def part_lumen(name, tree, iso6):
    sys.path.insert(0, "/sp/v6/wall")
    import va_lumen_poly as LP
    LP.ARGS = argparse.Namespace(root=tree, step=0.05, need=NEED, meshes="obj,full", union_dir=iso6)
    r = LP.one(name)
    out = {"raw": r, "min_mm": {}}
    for t in ("RVA", "LVA"):
        if t in r:
            out["min_mm"][t] = {m: (r[t][m]["lumen_min_mm"] if isinstance(r[t].get(m), dict) else None) for m in ("obj", "full", "union")}
    return out


# ------------------------------------------------------------------ va wall detail (+ D15 split)
def zone_weights(donor_id, dside):
    f = os.path.join(ZONES, str(donor_id), "zones.json")
    if not os.path.exists(f):
        return None, None
    z = jl(f).get(dside)
    if not z:
        return None, None
    st = jl(os.path.join(jl("/sp/v6/wall/donors.json")[donor_id]["v5_seg_dir"], "stations_rays.json"))
    P = np.asarray(st["rva" if dside == "R" else "lva"]["P"], float)
    w = np.asarray(z["w_station"], float)
    n = min(len(P), len(w))
    return P[:n], w[:n]


def d15_split(O, samp, bi, lo, hi, excl, xf):
    """frac_0.3mm of the donor_real cells inside (w > 0) / outside (w == 0) the donor's local zones."""
    P, w = zone_weights(xf.get("donor_id"), xf.get("donor_side"))
    if P is None:
        return None
    X = (P * np.asarray(xf.get("mirror", [1, 1, 1]), float) - np.asarray(xf["origin"], float)) @ np.asarray(xf["R"], float).T + \
        np.asarray(xf["anchor"], float)
    m = O.mask(bi, lo, hi, True, excl)
    if not m.any():
        return None
    pos = samp.p[O.best[m]]
    dd, ix = cKDTree(X).query(pos)
    ok = dd <= 8.0
    inz = ok & (w[ix] > 0)
    outz = ok & (w[ix] == 0)
    res = {}
    for k, mm in (("zone", inz), ("outside", outz)):
        idx = np.nonzero(m)[0][mm]
        if len(idx):
            a, x = O.A[idx], O.dep[idx]
            res[k] = {"area_mm2": round(float(a.sum()), 2), "frac_0.3mm": float(a[x > 0.3].sum() / a.sum())}
    res["unmatched_area_mm2"] = round(float(O.A[np.nonzero(m)[0][~ok]].sum()), 2)
    return res


def segs(O, samp, br, prov, rep, with_sub=True):
    x = prov["xform"]
    sup = rep.get("suppressed_stretches_mm") or {}
    out = {}
    for side, tag, sub, ck, tk, sc, sd, prox, dist in (
            ("right", "RVA", "(17)", "vaC_R", "va_R", "scC_R", "scD_R", "RVA", "(17)"),
            ("left", "LVA", "(19)", "vaC_L", "va_L", "scC_L", "scD_L", "(18)", "(19)")):
        bi = samp.idx(tag)
        L = float(M.arclen(br[tag][0])[-1])
        if tk not in x:
            continue
        cut = float(x[tk]["cut_mm"])
        d = {"cut": cut, "L": L, "tb_windows": sup.get(tk, [])}
        d["topbrain"] = O.seg(bi, cut, L)
        try:
            o_s = M.origin_s(br, tag, sub)
        except Exception:  # noqa: BLE001
            o_s = 0.0
        d["origin_s"] = o_s
        d["whole_from_origin"] = O.seg(bi, o_s, L)
        if ck in x:
            lo = float(x[ck]["lo_mm"])
            win = [tuple(w) for w in sup.get(ck, [])]
            excl = [(cut - 25.0, cut)] + win
            d.update(donor="%s %s" % (x[ck].get("donor_id"), x[ck].get("donor_side")), lo=lo, windows=win,
                     donor_stretch=O.seg(bi, lo, cut), donor_real=O.seg(bi, lo, cut, excl=excl, extra=True),
                     seam_zone=O.seg(bi, cut - 25.0, cut, core=False), tubed_mm=round(sum(b - a for a, b in win), 2))
            try:
                d["d15_split"] = d15_split(O, samp, bi, lo, cut, excl, x[ck])
            except Exception as e:  # noqa: BLE001
                d["d15_split"] = {"error": "%s: %s" % (type(e).__name__, e)}
        else:
            d["donor"] = None
        if with_sub and sc in x:
            rows = {}
            for key, bname in ((sc, prox), (sd, dist)) + ((("scJ_L", "LVA"),) if side == "left" else ()):
                if key not in x:
                    continue
                xs = x[key]
                win = [tuple(w) for w in sup.get(key, [])]
                bj = samp.idx(bname)
                rows[key] = {"branch": bname, "range": [float(xs["lo_mm"]), float(xs["hi_mm"])], "windows": win,
                             "real_surface": xs.get("real_surface"),
                             "all": O.seg(bj, float(xs["lo_mm"]), float(xs["hi_mm"]), core=False),
                             "real": O.seg(bj, float(xs["lo_mm"]), float(xs["hi_mm"]), core=False, excl=win)}
            rr = [v["real"] for k, v in rows.items() if k in (sc, sd) and v["real"]]
            if rr:
                a = sum(r_["area_mm2"] for r_ in rr)
                rows["subclavian_real_combined"] = {"area_mm2": round(a, 2),
                                                    "frac_0.3mm": sum(r_["frac_0.3mm"] * r_["area_mm2"] for r_ in rr) / a}
            d["subclavian"] = rows
        out[side] = d
    return out


def part_va(name, tree, iso6, rep6):
    prov6 = jl(os.path.join(tree, name, "provenance.json"))
    br6 = M.read_branches(os.path.join(tree, name, "Centrelines_comb"))
    s6 = M.Samp(br6)
    V6, F6 = M.read_vtp(os.path.join(iso6, name + ".vtp"))
    C6, A6_ = M.tri_geom(V6, F6)
    O6 = M.Own(s6, C6, A6_, "exact")
    res = {"v6": segs(O6, s6, br6, prov6, rep6)}
    del O6, C6, A6_, V6, F6
    if os.path.exists(os.path.join(A5, name)) and os.path.exists(os.path.join(ISO5, name + ".vtp")):
        prov5 = jl(os.path.join(A5, name, "provenance.json"))
        rep5 = jl(os.path.join(A5, name, "mesh_v5.json"))
        br5 = M.read_branches(os.path.join(A5, name, "Centrelines_comb"))
        s5 = M.Samp(br5)
        V5, F5 = M.read_vtp(os.path.join(ISO5, name + ".vtp"))
        C5, A5_ = M.tri_geom(V5, F5)
        O5 = M.Own(s5, C5, A5_, "exact")
        res["v5"] = segs(O5, s5, br5, prov5, rep5, with_sub=False)
    return res


# ------------------------------------------------------------------ orientation
def part_orient(name, tree, out_dir):
    sys.path.insert(0, "/sp/v6/gates")
    import orient_gates as OG
    res, errs = OG.measure_tree(tree, OG.TB_ROOT, [name], False)
    if name not in res:
        return {"error": errs.get(name), "version": OG.VERSION}
    r = OG.strip(res[name])
    prov = jl(os.path.join(tree, name, "provenance.json"))
    va = (r.get("pairs") or {}).get("va") or {}
    blocking = [list(b) for b in r.get("blocking", [])]
    after, flags = [], []
    for p, f in blocking:
        if f == "va_target":
            side = "R" if p.startswith("R") else "L"
            tm = va.get("tip_move_anat_%s_mm" % side)
            rung = None
            for k in ("va_R" if side == "R" else "va_L",):
                xf = (prov.get("xform") or {}).get(k) or {}
                rung = xf.get("va_rung") or xf.get("rung") or (xf.get("v6") or {}).get("rung")
            if tm is not None and tm <= 30.0:
                flags.append("va_target_20_30:%s" % p)
                continue
            if str(rung) in ("3", "v5_top", "rung3"):
                flags.append("va_target_gt30_rung3:%s" % p)
                continue
        after.append([p, f])
    # 1.3 (D23, user 2026-10-06): a piece listed in provenance v6_build.d23_accepted (w_017_left mr_002_L RVA_top: D9
    # rung 3, v5's top as shipped) keeps its tier A and its own blocking flags, accepted and flagged; the VA pair flags
    # it causes on the other top too (the pair is measured against that top)
    d23 = ((prov.get("v6_build") or {}).get("d23_accepted") or {})
    acc = set(d23.get("pieces") or [])
    pair_ok = set(d23.get("pair_flags") or [])
    tier_a = list(r.get("tierA_pieces") or [])
    if acc:
        after2 = []
        for p, f in after:
            if p in acc or (f in pair_ok and p.endswith("VA_top")):
                flags.append("D23:%s:%s" % (p, f))
                continue
            after2.append([p, f])
        after = after2
    tier_a_after = [p for p in tier_a if p not in acc]
    return {"version": OG.VERSION, "tierA_pieces": r.get("tierA_pieces"), "primary_pieces": r.get("primary_pieces"),
            "tierA_pieces_after_D23": tier_a_after, "d23_accepted": sorted(acc & set(tier_a)),
            "v6_blocking": blocking, "v6_blocking_after_D8": after, "accepted_flags": flags,
            "va_pair": {k: va.get(k) for k in ("tip_move_anat_R_mm", "tip_move_anat_L_mm", "lumen_gap_mm", "end_sep_mm",
                                                "end_sep_native_mm", "end_du_mm")},
            "forks": {k: {kk: vv for kk, vv in (v or {}).items() if kk in ("flags", "yaw_deg", "tilt_deg", "bend_deg")}
                      for k, v in (r.get("forks") or {}).items()},
            "classes": r.get("classes")}


# ------------------------------------------------------------------ loader
def part_loader(name, tree):
    """the training loader in a clean interpreter (repo eve / eve_bench first): loader_one.py."""
    import subprocess
    r = subprocess.run(["python3", "/sp/v6/checks/loader_one.py", tree, name], capture_output=True, text=True, timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith("LOADER_JSON "):
            return json.loads(line[len("LOADER_JSON "):])
    raise RuntimeError("loader_one.py rc %d: %s" % (r.returncode, (r.stderr or r.stdout)[-1500:]))


# ------------------------------------------------------------------ geometry (1.2)
def part_geom(name, tree):
    sys.path.insert(0, "/sp/v6/checks")
    import geom_v6 as GV
    r = GV.measure(tree, name, A5)
    keep = ("version", "v5_root", "changed_curves", "new_points", "pairs_checked", "clear_min_new_mm", "clear_fails",
            "clear_inherited_lt_0.6", "kink_fails", "kink", "gates")
    out = {k: r[k] for k in keep}
    out["clear_new_per_pair"] = {k: v["new"] for k, v in r["clear"].items() if v["new"] is not None}
    return out


# ------------------------------------------------------------------ gates
def gates(rec):
    g = {}
    it = rec.get("item1")
    if it:
        g["strict_audit"] = bool(it["gates"]["strict_pass"])
    rt = rec.get("route")
    if rt:
        d22 = rt.get("d22")
        if d22 is None:                                   # a record written by checks_v6 < 1.3: re-run part route
            g["route_d22_measured"] = False
        else:
            g["route_fork_d22_ge_0.99"] = bool((d22.get("fork") or {}).get("ok"))
            g["route_siphon_d22_ge_0.99"] = bool((d22.get("siphon") or {}).get("ok"))
        if "rcca_prefix" in rt:
            g["rcca_prefix_unchanged"] = bool(rt["rcca_prefix"].get("identical_v4"))
    cv = rec.get("coverage")                              # 1.6 (D27): coverage against the v4 / v5 shipped wall
    if cv:
        g["route_fork_coverage_loss_lt_1pct"] = bool((cv.get("fork") or {}).get("ok"))
        g["route_siphon_coverage_loss_lt_1pct"] = bool((cv.get("siphon") or {}).get("ok"))
    fo = rec.get("folds")                                 # 1.6: fold-over pairs on the training OBJ
    if fo:
        g["obj_no_fold_over"] = (fo.get("obj") or {}).get("fold_pairs", 1) == 0
    lu = rec.get("lumen")
    if lu:
        objs = [v.get("obj") for v in lu["min_mm"].values()]
        g["va_lumen_obj_ge_0.65"] = bool(objs) and all(x is not None and x >= NEED for x in objs)
    orr = rec.get("orient")
    if orr and "error" not in orr:
        # 1.3 (D23): "0 tier A except flagged D23"
        g["orient_tierA_0"] = not orr.get("tierA_pieces_after_D23", orr["tierA_pieces"])
        g["orient_v6_blocking_after_D8_0"] = not orr["v6_blocking_after_D8"]
    ld = rec.get("loader")
    if ld:
        g["loader_ok"] = bool(ld["ok"])
    ge = rec.get("geom")
    if ge:
        g.update({k: bool(v) for k, v in ge["gates"].items()})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tree", default=os.environ.get("A6_DIR", "/bw/tree"))
    ap.add_argument("--iso", default=os.environ.get("ISO_DIR", "/bw/iso"))
    ap.add_argument("--out", default=os.environ.get("CHECKS_DIR", "/bw/checks"))
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    name = a.name
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, name + ".json")
    rec = jl(dst) if os.path.exists(dst) else {"name": name}
    # 1.4: the checks_v6 version that computed each part (parts already in a record written by an older version say so)
    pv = rec.setdefault("part_versions", {})
    for p_ in PARTS:
        if p_ in rec and p_ not in pv:
            pv[p_] = str(rec.get("script") or "?").split(" (")[0]
    rec["script"] = VERSION
    root = os.path.join(a.tree, name)
    rep6 = jl(os.path.join(root, "mesh_v6.json"))
    for part in a.parts.split(","):
        if part in rec and not a.force:
            continue
        t = time.time()
        try:
            if part == "item1":
                CK.ISO = a.iso
                out, _ = CK.item1(name, root, rep6)
                rec["item1"] = out
                print("item1 strict_pass", out["gates"]["strict_pass"], flush=True)
            elif part == "route":
                rec["route"] = part_route(name, a.tree, a.iso, rep6)
                d_ = rec["route"]["d22"]
                print("route", rec["route"]["d13"]["baseline_used"], "d22", d_.get("reference"),
                      {p: [(d_.get(p) or {}).get(k) for k in ("value_v4", "value_own", "value_1p3")] for p in ("fork", "siphon")},
                      flush=True)
                rec.pop("route_d22_error", None)
                rec.pop("route_d22_tb", None)
                rec["route"].pop("d22_only_rerun", None)
            elif part == "route_d22":
                rt = rec.get("route")
                if not rt or "route_error" in rec:
                    raise RuntimeError("part route_d22 needs a complete earlier route record")
                d_ = part_route_d22(name, a.tree, a.iso, rep6, rt)
                rt["d22"] = d_
                rt["d22_only_rerun"] = {"d22_by": VERSION.split(" (")[0], "rest_of_route_from": pv.get("route"),
                                        "time": time.strftime("%F %T"),
                                        "why": "geometry and bake unchanged since the carried route pass; D26 (1.4) and D27 (1.5) changed only "
                                               "route.d22 (USER 2026-10-07 15:15 and 16:25)"}
                print("route_d22", rt["d13"]["baseline_used"], "d22", d_.get("reference"),
                      {p: [(d_.get(p) or {}).get(k) for k in ("value_v4", "value_own", "value_1p3", "value")] for p in ("fork", "siphon")},
                      flush=True)
            elif part == "coverage":
                rec["coverage"] = part_coverage(name, a.tree, a.iso, rep6)
                print("coverage", {p: rec["coverage"][p] for p in ("fork", "siphon")}, flush=True)
            elif part == "folds":
                rec["folds"] = part_folds(name, a.tree)
                print("folds", {k: v["fold_pairs"] for k, v in rec["folds"].items()}, flush=True)
            elif part == "lumen":
                rec["lumen"] = part_lumen(name, a.tree, a.iso)
                print("lumen", rec["lumen"]["min_mm"], flush=True)
            elif part == "va":
                rec["va"] = part_va(name, a.tree, a.iso, rep6)
                print("va done", flush=True)
            elif part == "orient":
                rec["orient"] = part_orient(name, a.tree, a.out)
                print("orient tierA", rec["orient"].get("tierA_pieces"), "blocking", rec["orient"].get("v6_blocking_after_D8"),
                      flush=True)
            elif part == "loader":
                rec["loader"] = part_loader(name, a.tree)
                print("loader", rec["loader"]["ok"], rec["loader"]["problems"], flush=True)
            elif part == "geom":
                rec["geom"] = part_geom(name, a.tree)
                print("geom", rec["geom"]["gates"], "min_new", rec["geom"]["clear_min_new_mm"], rec["geom"]["clear_fails"],
                      rec["geom"]["kink_fails"], flush=True)
            rec.setdefault("seconds", {})[part] = round(time.time() - t, 1)
            pv[part] = VERSION.split(" (")[0]
            rec.pop(part + "_error", None)
        except Exception as e:  # noqa: BLE001
            rec[part + "_error"] = "%s: %s" % (type(e).__name__, e)
            rec[part + "_tb"] = traceback.format_exc()[-3000:]
            print(rec[part + "_tb"], flush=True)
        rec["gates"] = gates(rec)
        rec["pass"] = bool(rec["gates"]) and all(rec["gates"].values()) and not any(k.endswith("_error") for k in rec)
        tmp = dst + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, default=conv)
        os.replace(tmp, dst)
    print("GATES", json.dumps(rec["gates"]), "PASS" if rec["pass"] else "FAIL", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
