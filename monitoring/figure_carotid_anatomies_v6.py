#!/usr/bin/env python3
"""Visual QC of the v6 anatomies: a FRONT and a SIDE view of every anatomy, one image per anatomy.

Successor of figure_carotid_anatomies_v5.py (house colours, provenance-driven seams, one shared window for the whole
set so every image is at the same scale) and of the v6 pilot figures (front + side projections). The single oblique
view of v5 hid heading errors: a fork or siphon that runs posterior looks right from the front. Here both views are
orthographic projections of the branch frame (+x patient left, +y posterior, +z superior; the collision OBJ is in the
same frame):

    FRONT  x vs z seen from anterior: the patient's right on the image left; dotted line = host midline x 14.8
    SIDE   y vs z seen from the patient's right: anterior on the image RIGHT (y axis reversed); the patient's left
           side (the far side) is drawn faded; the ICA termini carry their AP offset (provenance
           v6_build.carotid.ica_ap_post_mm, flagged > 40 mm)

Colours (v5 house style; the donor colour change, V6_ITEMS item f, is deferred):

    RCCA / LCCA  host grey | CCA blue | ICA cyan | siphon orange, a white dot per seam                     (as v5)
                 ICA extension (xform.lower*.extend_mm, the copied distal shape at the ICA tip): white dashes
    RECA / LECA  fork ECA magenta, TopBrain ECA tip purple (dot at the join)                              (as v5)
    RVA / LVA    host subclavian grey | DONOR subclavian stretch olive (v6: xform scC_* / scJ_*, the donor's own
                 subclavian + junction surface; under the donor VA where they overlap) | donor VA dark teal |
                 TopBrain VA top brown; a VA with no donor: host VA green | TopBrain top brown         (as v5 + olive)
    (17) / (19)  on a donor-junction side the donor's OWN distal subclavian toward the arm (D17, xform scD_*): olive
                 to its cut end (x); its short D17 tube part dashed; elsewhere the host's distal subclavian, grey
    junction     marker at the VA ostium by junction type: olive hexagon = the donor's own junction (D9 rung 1);
                 yellow hexagon = the same with a tube-bridged pinch (D16); grey diamond = v5's junction (rung 2/3);
                 green triangle = host VA, no donor
    windows      the bake's tube windows (bake_v6.va_tubed windows_mm, any section): yellow squares; a D27 repair
                 window (v6_checks.d22.<piece>.d22_repair_window): yellow square with a red edge
    mesh         the baked collision OBJ as a faint shadow, shaded by the piece of its nearest centreline
    v5           v5's six named centrelines (anatomies_v5, same name) as thin dotted grey lines, for comparison
    other        host branches thin red; host VA confluence (D2 target centre) black star

The FRONT view carries call-outs for what v6 changed: each fork (re-posed by the Zenodo map / v5's pose, rung,
mirrored D20, substituted D1, host-cut shift, seam-1 bend, ICA extension), each siphon (re-posed anatomically,
deviation), each VA (D9 rung, junction type, donor, D16 / D17 tubes). The title adds the same per side, the D22 route
check reference (v4 / own surface D26 / repair window D27) and every flag of v6_build.flags + v6_checks.flags
(va_<side>_rung1 left out: it is the default and is in the VA line).

    docker run --rm <mounts> eve-training-fixed
      python3 monitoring/figure_carotid_anatomies_v6.py <anatomies_v6> <out> [--v5 <anatomies_v5>] [--only a,b,...]
             [--shard i/n] [--skip-existing] [--claims <dir>] [--records <dir>] [--no-v5]
"""
import argparse
import glob
import json
import os
import socket
import sys
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

VERSION = ("figure_carotid_anatomies_v6 1.2 (2026-10-07, s2_images; 1.1 = host-VA sides grey before the ostium, ICA terminus "
           "marker + lateral offset from the midline; 1.2 = build-record flags (--build-json), fork-near-midline warning, "
           "side-view label spacing)")

SEG = [("host", "#7f7f7f"), ("CCA", "#1f77b4"), ("ICA", "#17becf"), ("siphon", "#ff7f0e")]
ECA_C, ECAEXT_C = "#e377c2", "#9467bd"
VA_C, VAEXT_C = "#2ca02c", "#8c564b"          # host VA kept (no donor) green, TopBrain VA top brown
HOST_C = "#7f7f7f"                            # host subclavian
DONOR_C = "#00332e"                           # donor lower VA (thin-slice CTA), dark teal (unchanged, item f deferred)
DSUB_C = "#8a8a00"                            # donor subclavian + junction + own distal stretch (v6), olive
WIN_C = "#ffd400"                             # tube-window marker
OTHER_C = "#d62728"
GHOST_C = "#8c8c8c"
MESH = {"right": ("#1f77b4", 0.05), "left": ("#17becf", 0.05), "donor": (DONOR_C, 0.06), "dsub": (DSUB_C, 0.06),
        "tbva": (VAEXT_C, 0.05), "v4va": (VA_C, 0.05), "other": (OTHER_C, 0.035)}
SRC = {"ia172": "Head-CTA 172", "song": "Song 2024", "vmr0248": "VMR 0248"}
SIDES = (("right", "R"), ("left", "L"))
NAMED = ("RCCA", "LCCA", "RECA", "LECA", "RVA", "LVA")
SUBCL = ("(17)", "(18)", "(19)")
SUBCL_SIDE = {"R": ("(17)",), "L": ("(18)", "(19)")}
BRANCH_SIDE = {"RCCA": "R", "RECA": "R", "RVA": "R", "(17)": "R", "LCCA": "L", "LECA": "L", "LVA": "L", "(18)": "L",
               "(19)": "L"}
FADE = 0.4                                    # the far side (patient's left) in the SIDE view
C_HOST = (14.8, 12.5, 572.0)                  # host VA confluence (D2 target centre), branch frame
MIDLINE_X = 14.8
VIEWS = (("FRONT  (from anterior; image left = patient's right)", 0),
         ("SIDE  (from the patient's right; image right = anterior)", 1))
JUNC = {"own": dict(marker="h", ms=9, mfc=DSUB_C, mec="k", mew=0.9),
        "own_tube": dict(marker="h", ms=9.5, mfc=WIN_C, mec=DSUB_C, mew=2.0),
        "v5": dict(marker="D", ms=7, mfc="#bdbdbd", mec="k", mew=0.9),
        "host": dict(marker="^", ms=8, mfc=VA_C, mec="k", mew=0.9)}


# ----------------------------------------------------------------------------------------------------------- data
def read_curve(path):
    """Slicer markups curve -> branch frame (y, -z, -x of the JSON), as eve_bench load_points_from_json."""
    d = json.load(open(path, encoding="utf-8"))
    for m in d["markups"]:
        if m.get("type", "Curve") == "Curve":
            a = np.array([cp["position"] for cp in m["controlPoints"]], float)
            return np.stack([a[:, 1], -a[:, 2], -a[:, 0]], axis=-1)
    raise ValueError("no curve in %s" % path)


def load_curves(root, keys=None):
    out = {}
    cdir = os.path.join(root, "Centrelines_comb")
    for f in sorted(os.listdir(cdir)):
        if not f.endswith(".mrk.json"):
            continue
        k = f[len("Centerline curve"):-len(".mrk.json")].strip(" -")
        if keys is None or k in keys:
            out[k] = read_curve(os.path.join(cdir, f))
    return out


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def mirrored(m):
    return m is not None and any(float(x) < 0 for x in m)


# ---------------------------------------------------------------------------------------------------- the layout
def va_info(prov, side, S):
    """Everything drawn / written about one VA side."""
    X = prov["xform"]
    rec = g(prov, "v6_build", "va", side, default={}) or {}
    vac = X.get("vaC_" + S)
    cut = float(X["va_" + S]["cut_mm"])
    jt = rec.get("junction_type") or ("donor" if vac else "host_va_no_donor")
    if jt.startswith("donor_own_junction"):
        jkind = "own_tube" if ("d16" in jt or "tube_bridge" in jt) else "own"
    elif jt.startswith("v5_"):
        jkind = "v5"
    else:
        jkind = "host"
    sc = {k: X[k] for k in X if k.startswith("sc") and k.endswith("_" + S) and isinstance(X[k], dict)}
    d = {"cut": cut, "vac": vac, "rec": rec, "jt": jt, "jkind": jkind, "sc": sc, "rung": rec.get("rung")}
    if vac:
        vside = g(prov, "v6_va", "record", "sides", side, default={}) or {}
        did = vac.get("donor_id") or rec.get("donor_id") or "?"
        src = vside.get("donor_source") or did.split("_")[0]
        d.update(donor=did, donor_side=vac.get("donor_side") or rec.get("donor_side") or "?",
                 donor_src=SRC.get(src, src), donor_mir=mirrored(vac.get("mirror") or vside.get("donor_mirror")),
                 lo=float(vac["lo_mm"]), hi=float(vac["hi_mm"]))
    return d


def section_branch(key, X):
    """Branch (centreline key) of a bake section key of bake_v6.va_tubed."""
    b = g(X, key, "branch")
    if b:
        return b
    base, _, S = key.rpartition("_")
    return {"lower": S + "CCA", "siphon": S + "CCA", "eca": S + "ECA", "ecaext": S + "ECA", "va": S + "VA",
            "vaC": S + "VA"}.get(base)


def group_labels(key, s, prov, V):
    """Mesh group of every point of one centreline (for the shaded shadow)."""
    lab = np.full(len(s), "other", dtype=object)
    X = prov["xform"]
    if key in ("RCCA", "RECA"):
        lab[:] = "right"
    elif key in ("LCCA", "LECA"):
        lab[:] = "left"
    elif key in ("RVA", "LVA"):
        v = V[key[0]]
        if v["vac"]:
            lab[:] = "other"
            lab[(s >= v["lo"]) & (s <= v["cut"])] = "donor"
            lab[s > v["cut"]] = "tbva"
        else:
            lab[s <= v["cut"]] = "v4va"
            lab[s > v["cut"]] = "tbva"
            lab[s < v.get("host_to", 0.0)] = "other"           # 1.1: the host subclavian before the ostium
    for k, xs in X.items():
        if k.startswith("sc") and isinstance(xs, dict) and xs.get("branch") == key:
            m = (s >= float(xs["lo_mm"])) & (s <= float(xs["hi_mm"]))
            if key in ("RVA", "LVA"):
                v = V[key[0]]
                m &= (s < v["lo"]) if v["vac"] else m
            lab[m] = "dsub"
    return lab


# ------------------------------------------------------------------------------------------------------- drawing
def seg(ax, p, s, lo, hi, i, **kw):
    """The stretch lo <= s <= hi, end points snapped to the nearest samples so adjacent pieces join."""
    a = int(np.argmin(np.abs(s - lo)))
    b = int(np.argmin(np.abs(s - hi)))
    if b > a:
        ax.plot(p[a:b + 1, i], p[a:b + 1, 2], solid_capstyle="round", **kw)


def at(p, s, x):
    return p[int(np.argmin(np.abs(s - x)))]


def dot(ax, q, i, z=9):
    ax.plot(q[i], q[2], "o", ms=4.6, mfc="white", mec="k", mew=0.9, zorder=z)


def mesh_shadow(ax, tri2d, i):
    for k, (polys) in tri2d.items():
        if len(polys):
            col, a = MESH[k]
            ax.add_collection(PolyCollection(polys[:, :, [i, 2]], facecolor=(*mcolors.to_rgb(col), a),
                                             edgecolor="none", zorder=1))


def load_mesh(root, C, prov, V, target=12000):
    import pyvista as pv
    from scipy.spatial import cKDTree
    path = os.path.join(root, "vessel_architecture_collision.obj")
    if not os.path.isfile(path):
        return {}, None
    m = pv.read(path).extract_surface().triangulate()
    n0 = m.n_cells
    if m.n_cells > target:
        m = m.decimate(1.0 - target / float(m.n_cells))
    v = np.asarray(m.points, np.float64)
    f = np.asarray(m.faces, np.int64).reshape(-1, 4)[:, 1:]
    pts, labs = [], []
    for k, p in C.items():
        pts.append(p)
        labs.append(group_labels(k, arclen(p), prov, V))
    dist, j = cKDTree(np.concatenate(pts)).query(v[f].mean(axis=1))
    lab = np.concatenate(labs)[j]
    tri = v[f]
    return {k: tri[lab == k] for k in MESH}, {"obj_tris": int(n0), "drawn_tris": int(len(f)),
                                                "tri_to_centreline_mm_p99": float(np.percentile(dist, 99))}


def draw_route(ax, p, cut, cca, ica, ext, i):
    s = arclen(p)
    e = [0.0, cut, cut + cca, cut + cca + ica, s[-1]]
    for (lab, col), lo, hi in zip(SEG, e[:-1], e[1:]):
        seg(ax, p, s, lo, hi, i, color=col, lw=2.2, zorder=7)
    if ext and ext > 0.05:
        a, b = e[3] - ext, e[3]
        k0, k1 = int(np.argmin(np.abs(s - a))), int(np.argmin(np.abs(s - b)))
        if k1 > k0:
            ax.plot(p[k0:k1 + 1, i], p[k0:k1 + 1, 2], color="white", lw=1.1, ls=(0, (2.0, 1.6)), zorder=7.5)
    for x in e[1:4]:
        dot(ax, at(p, s, x), i)
    ax.plot(p[-1, i], p[-1, 2], "^", ms=7, mfc=SEG[3][1], mec="k", mew=0.9, zorder=9.5)   # 1.1: ICA terminus
    return s


def draw_va(ax, C, key, V, X, i):
    """The VA centreline in its pieces + the sc* sections on any branch of this side."""
    S = key[0]
    v = V[S]
    p = C[key]
    s = arclen(p)
    if v["vac"]:
        sc_va = [xs for xs in v["sc"].values() if xs.get("branch") == key]
        host_to = min([float(xs["lo_mm"]) for xs in sc_va] + [v["lo"]])
        seg(ax, p, s, 0.0, host_to, i, color=HOST_C, lw=1.8, zorder=6)
        for xs in sc_va:     # olive: below the VA where the junction section overlaps the donor VA
            seg(ax, p, s, float(xs["lo_mm"]), float(xs["hi_mm"]), i, color=DSUB_C, lw=4.0, zorder=5.5)
        seg(ax, p, s, v["lo"], v["cut"], i, color=DONOR_C, lw=2.2, zorder=7)
        if v["lo"] > 0.5:
            dot(ax, at(p, s, v["lo"]), i)
    else:                    # 1.1: a host VA (no donor) drawn green from its ostium, the subclavian before it grey
        h = float(v.get("host_to", 0.0))
        if h > 0.5:
            seg(ax, p, s, 0.0, h, i, color=HOST_C, lw=1.8, zorder=6)
        seg(ax, p, s, h, v["cut"], i, color=VA_C, lw=1.9, zorder=7)
    seg(ax, p, s, v["cut"], s[-1], i, color=VAEXT_C, lw=1.9, zorder=7)
    dot(ax, at(p, s, v["cut"]), i)
    ax.plot(p[-1, i], p[-1, 2], "v", ms=4.5, color=VAEXT_C, mec="k", mew=0.5, zorder=9)
    return s


def draw_subclavians(ax, C, V, i, S):
    """(17) right / (18), (19) left: host grey; donor sc* sections olive; a D17 cut end x, its tube part dashed."""
    for b in SUBCL_SIDE[S]:
        if b in C:
            p = C[b]
            ax.plot(p[:, i], p[:, 2], color=HOST_C, lw=1.3, zorder=4, solid_capstyle="round")
    ends = []
    if True:
        for k, xs in V[S]["sc"].items():
            b = xs.get("branch")
            if b not in SUBCL or b not in C:
                continue
            p = C[b]
            s = arclen(p)
            lo, hi = float(xs["lo_mm"]), float(xs["hi_mm"])
            d = xs.get("d17") or {}
            t0, tm = d.get("tube_from_mm"), float(d.get("tube_mm") or 0.0)
            if t0 is not None and tm > 0.05:
                seg(ax, p, s, lo, float(t0), i, color=DSUB_C, lw=2.6, zorder=6.5)
                a = int(np.argmin(np.abs(s - float(t0))))
                b2 = int(np.argmin(np.abs(s - hi)))
                ax.plot(p[a:b2 + 1, i], p[a:b2 + 1, 2], color=DSUB_C, lw=2.6, ls=(0, (2.2, 1.6)), zorder=6.5)
            else:
                seg(ax, p, s, lo, hi, i, color=DSUB_C, lw=2.6, zorder=6.5)
            if d.get("cut_end"):
                q = at(p, s, float(d.get("end_mm") or hi))
                ax.plot(q[i], q[2], "x", ms=7, mew=2.0, color="k", zorder=9.5)
                ends.append(q)
    return ends


def draw_eca(ax, C, X, S, i):
    p = C[S + "ECA"]
    s = arclen(p)
    ext = X.get("ecaext_" + S)
    if ext:
        j = float(ext["join_at_mm"])
        seg(ax, p, s, 0.0, j, i, color=ECA_C, lw=2.0, zorder=7)
        seg(ax, p, s, j, s[-1], i, color=ECAEXT_C, lw=2.0, zorder=7)
        dot(ax, at(p, s, j), i)
    else:
        ax.plot(p[:, i], p[:, 2], color=ECA_C, lw=2.0, zorder=7, solid_capstyle="round")


def junction_point(C, S):
    """The VA ostium: the start of (17) on the right (the subclavian's distal branch leaves the RVA there), the
    start of the LVA on the left."""
    if S == "R":
        return C["(17)"][0] if "(17)" in C else None
    return C["LVA"][0] if "LVA" in C else None


def windows(prov):
    """[(branch, lo, hi, kind)] for the bake's tube windows and any D27 repair window."""
    X = prov["xform"]
    out = []
    for key, r in (g(prov, "bake_v6", "va_tubed", default={}) or {}).items():
        b = section_branch(key, X)
        for w in (r or {}).get("windows_mm") or []:
            out.append((b, float(w[0]), float(w[1]), "tube", key))
    for piece, b in (("fork", "RCCA"), ("siphon", "RCCA")):
        rw = g(prov, "v6_checks", "d22", piece, "d22_repair_window")
        if rw and rw.get("s_range_mm"):
            out.append((b, float(rw["s_range_mm"][0]), float(rw["s_range_mm"][1]), "d27", piece))
    return out


def draw_windows(ax, C, W, i):
    for b, lo, hi, kind, _ in W:
        if b not in C:
            continue
        p = C[b]
        s = arclen(p)
        a, c = int(np.argmin(np.abs(s - lo))), int(np.argmin(np.abs(s - hi)))
        if c > a:
            ln, = ax.plot(p[a:c + 1, i], p[a:c + 1, 2], color="white", lw=1.0, zorder=7.6)
            ln.set_path_effects([pe.Stroke(linewidth=3.4, foreground="k"), pe.Normal()])
        q = at(p, s, 0.5 * (lo + hi))
        ax.plot(q[i], q[2], "s", ms=6.5 if kind == "d27" else 5.5, mfc=WIN_C, mec="#d62728" if kind == "d27" else "k",
                mew=1.8 if kind == "d27" else 0.9, zorder=9.6)


# -------------------------------------------------------------------------------------------------------- text
def short(n):
    return (n or "?").replace("case_", "")


def fork_text(prov, side, near=None):
    """(call-out lines, title text) for one fork. near (1.2): (min lateral offset of the CCA + ICA from the host midline,
    the bifurcation's), mm, + = on its own side; a fork closer than 5 mm gets a warning line."""
    lines, t = _fork_text(prov, side)
    if near is not None and near[0] < 5.0:
        w = "cervical ICA %s the midline (min %+.0f mm; bifurcation %+.0f mm)" % (
            "ACROSS" if near[0] < 0 else "within 5 mm of", near[0], near[1])
        lines = lines + [w]
        t = t + " | " + w
    return lines, t


def _fork_text(prov, side):
    car = g(prov, "v6_build", "carotid", default={}) or {}
    X = prov["xform"]
    if side == "right":
        rf = car.get("right_fork") or {}
        xl = X.get("lower") or {}
        lower = prov.get("lower")
        rung, pose = rf.get("right_fork_rung"), rf.get("right_fork_pose")
        mir = rf.get("right_fork_mirrored")
        shift = float(rf.get("host_cut_shift_mm") or 0.0)
        bend = rf.get("right_seam1_bend_deg")
        reposed = rf.get("right_fork_placement", xl.get("placement")) not in (None, "v5")
        what = ("re-posed (%s)" % pose) if reposed else "v5 pose (R5, flagged)"
        l1 = "R fork %s: R%s %s" % (short(lower), rung, what)
        extra = []
        if mir:
            extra.append("MIRRORED (D20)")
        if shift > 0.05:
            extra.append("host cut +%.1f mm" % shift)
        if bend is not None:
            extra.append("seam-1 bend %.0f%s" % (float(bend), "°" + (" >45" if float(bend) > 45 else "")))
        ext = float(xl.get("extend_mm") or 0.0)
        if ext > 0.05:
            extra.append("ICA ext %.1f mm" % ext)
        dev = rf.get("right_dev_deg")
        if dev:
            extra.append("dev %.0f°" % float(dev))
        return [l1, ", ".join(extra)] if extra else [l1], "RIGHT fork %s R%s %s%s" % (
            short(lower), rung, what, (" | " + ", ".join(extra)) if extra else "")
    lf = car.get("left_fork") or {}
    xl = X.get("lower_L") or {}
    dec, lower = lf.get("left_fork_decision"), lf.get("left_fork_lower") or g(prov, "left", "lower")
    pose = lf.get("left_fork_pose")
    rung = lf.get("fork_rung")
    sub = dec == "substituted"
    l1 = "L fork %s: %s rung %s" % (short(lower), "SUBSTITUTED (D1)" if sub else "own", rung)
    extra = ["pose %s%s" % (pose, " (flag)" if lf.get("left_fork_pose_flag") else "")]
    if lf.get("left_fork_mirrored"):
        extra.append("MIRRORED (D20)")
    if lf.get("left_seam1_bend_deg") is not None:
        extra.append("seam-1 bend %.0f°" % float(lf["left_seam1_bend_deg"]))
    dev = g(xl, "v6", "dev_deg")
    if dev:
        extra.append("dev %.0f°" % float(dev))
    ext = float(xl.get("extend_mm") or 0.0)
    if ext > 0.05:
        extra.append("ICA ext %.1f mm" % ext)
    tub = float(lf.get("tubed_ica_mm") or 0.0)
    if tub > 0.05:
        extra.append("ICA tube %.1f mm" % tub)
    t = "LEFT fork %s %s rung %s%s" % (short(lower), "SUBSTITUTED (D1; v5 %s)" % short(g(prov, "left", "v5_lower"))
                                        if sub else "own", rung, " | " + ", ".join(extra))
    return [l1, ", ".join(extra)], t


def siphon_text(prov, S, lat=None):
    """lat (1.1): the ICA terminus's lateral offset from the host midline x 14.8 (mm, + = on its own side)."""
    X = prov["xform"]
    xs = X.get("siphon" if S == "R" else "siphon_L") or {}
    pl = xs.get("placement")
    dev = g(xs, "v6", "dev_deg")
    bend = g(xs, "v6", "seam5_deg")
    src = os.path.basename(str(xs.get("surface") or "")).replace(".vtp", "")
    mir = mirrored(xs.get("mirror"))
    what = "re-posed (anatomical)" if pl == "v6_anat" else "pose %s" % pl
    l2 = "%s%sdev %.0f°, seam bend %.0f°" % (src, " mirrored, " if mir else ", ", float(dev or 0), float(bend or 0))
    lt = ""
    if lat is not None:
        lt = ("terminus %.0f mm lateral of the midline" % lat) if lat >= 0 else \
             ("terminus %.0f mm ACROSS the midline" % -lat)
    return (["%s siphon %s" % (S, what), l2] + ([lt] if lt else []),
            "siphon %s %s (%s%s)" % (what, src + (" mirrored" if mir else ""), "dev %.0f°" % float(dev or 0),
                                     ", " + lt if lt else ""))


def va_text(prov, side, S, v):
    rec = v["rec"]
    rung = v["rung"]
    jt = v["jt"]
    jname = {"donor_own_junction": "donor's own junction", "donor_own_junction+d16_tube": "donor's own junction + D16 tube",
             "donor_own_junction+tube_bridge": "donor's own junction + tube bridge", "v5_junction": "v5's junction",
             "v5_top_as_shipped": "v5's junction + v5's top", "host_va_no_donor": "host VA, no donor"}.get(jt, jt)
    lines = ["%sVA rung %s: %s" % (S, rung, jname)]
    t = "%sVA rung %s %s" % (S, rung, jname)
    if v["vac"]:
        dl = "%s %s%s (%s) %.0f mm" % (v["donor"].replace("song_", ""), v["donor_side"], " mirrored" if v["donor_mir"] else "",
                                       v["donor_src"], v["hi"] - v["lo"])
        lines.append(dl)
        t += ": " + dl
    tubes = []
    if rec.get("d16_tube_mm"):
        tubes.append("D16 tube %.1f mm" % float(rec["d16_tube_mm"]))
    elif rec.get("tube_bridge_mm"):
        tubes.append("tube bridge %.1f mm" % float(rec["tube_bridge_mm"]))
    d17 = rec.get("d17") or {}
    if d17:
        tubes.append("own distal %.0f mm past ostium%s" % (float(d17.get("past_ostium_mm") or 0),
                                                           (", D17 tube %.1f mm" % float(d17["tube_mm"]))
                                                           if float(d17.get("tube_mm") or 0) > 0.05 else ""))
    if tubes:
        lines.append(", ".join(tubes))
        t += " | " + ", ".join(tubes)
    if rec.get("va_target_mm") is not None:
        t += " | top %.0f mm from target" % float(rec["va_target_mm"])
    if rec and rec.get("pass_all") is False:
        lines.append("(accepted with flags)")
        t += " (VA gates: accepted with flags)"
    return lines, t


def d22_text(prov):
    d = g(prov, "v6_checks", "d22", default={}) or {}
    out = []
    for piece in ("fork", "siphon"):
        r = d.get(piece) or {}
        if not r:
            continue
        txt = "%s %s %.4f" % (piece, "own surface (D26)" if r.get("reference") == "own_surface" else r.get("reference"),
                              float(r.get("value") or 0))
        if r.get("d27_flag"):
            rw = r.get("d22_repair_window") or {}
            txt += " [D27 repair window %.2f mm² at s %s; %.4f before]" % (
                float(rw.get("area_mm2") or 0), "-".join("%.2f" % x for x in rw.get("s_range_mm") or []),
                float(r.get("value_before_d27") or 0))
        if r.get("coverage") is not None and float(r["coverage"]) < 0.99:
            txt += " (v4 wall coverage %.3f)" % float(r["coverage"])
        out.append(txt)
    ok = g(prov, "v6_checks", "pass")
    return "v6 checks %s (%s) | D22 right route: %s" % ("PASS" if ok else "NOT PASS" if ok is False else "?",
                                                        g(prov, "v6_checks", "checks_v6", default="?"), "; ".join(out))


def flag_list(prov, extra=None):
    """v6_build.flags + v6_checks.flags + (1.2) the build record's per-anatomy flags (--build-json: e.g. orient:*,
    left_pose_map_bounded_dev, which provenance does not carry)."""
    fl = list(g(prov, "v6_build", "flags", default=[]) or [])
    for f in list(g(prov, "v6_checks", "flags", default=[]) or []) + list(extra or []):
        if f not in fl and not any(x.startswith(f + "_") for x in fl):
            fl.append(f)
    return [f for f in fl if f not in ("va_right_rung1", "va_left_rung1")]


def eca_text(prov, side):
    e = g(prov, "eca_extension", side, default={}) or {}
    if e.get("extended"):
        return "ECA tip %s" % (e.get("v6_pose") or "kept")
    return "ECA tip skipped (%s)" % e["why"] if e.get("why") else "no ECA tip"


def callouts(ax, items, where):
    """Boxed call-outs in the upper margin of the FRONT view with leader lines to their anchors."""
    x0, x1 = sorted(ax.get_xlim())
    z0, z1 = ax.get_ylim()
    tx = x0 + 0.012 * (x1 - x0) if where == "left" else x1 - 0.012 * (x1 - x0)
    z = z1 - 0.012 * (z1 - z0)
    for q, lines, col in items:
        if q is None:
            continue
        ax.annotate("\n".join(lines), xy=(q[0], q[1]), xytext=(tx, z), textcoords="data",
                    ha="left" if where == "left" else "right", va="top", fontsize=6.6, zorder=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.0, alpha=0.88),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.75, ls=(0, (1.2, 1.8)), shrinkA=0,
                                    shrinkB=4))
        z -= (len(lines) * 4.3 + 4.5) * (z1 - z0) / 260.0


def title_lines(prov, name, V, W, T=None, NEAR=None, extra_flags=None):
    hand = prov.get("handedness")
    lines = ["%s   (%s; patient %s)   %s" % (name, hand, prov.get("patient"), d22_text(prov))]
    ap = g(prov, "v6_build", "carotid", "ica_ap_post_mm", default={}) or {}
    for side, S in SIDES:
        ft = fork_text(prov, side, (NEAR or {}).get(S))[1]
        st = siphon_text(prov, S, (T or {}).get(S))[1]
        lines.append("%s | %s | ICA end AP %s mm | %s" % (ft, st, ap.get(S + "ICA"), eca_text(prov, side)))
    lines.append("   ".join(va_text(prov, side, S, V[S])[1] for side, S in SIDES))
    wt = {}
    for b, lo, hi, kind, key in W:
        if kind == "tube":
            wt[key] = wt.get(key, 0.0) + hi - lo
    if wt:
        lines.append("bake tube windows: " + ", ".join("%s %.1f mm" % (k, x) for k, x in sorted(wt.items())))
    fl = flag_list(prov, extra_flags)
    lines.append("flags: " + (", ".join(fl) if fl else "none"))
    out = []
    for ln in lines:                       # every line wrapped to the figure width
        out += textwrap.wrap(ln, 228, subsequent_indent="      ") or [""]
    return out


# ------------------------------------------------------------------------------------------------------ one image
def render(root, v5root, lims, out_png, ghost=True, extra_flags=None):
    t0 = time.time()
    name = os.path.basename(root)
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    C = load_curves(root)
    X = prov["xform"]
    L = prov["left"]
    V = {S: va_info(prov, side, S) for side, S in SIDES}
    for S in "RL":                  # 1.1: a host VA (no donor) starts at its ostium; the stretch before it is subclavian
        jq = junction_point(C, S)
        if not V[S]["vac"] and jq is not None:
            p = C[S + "VA"]
            d = np.linalg.norm(p - jq, axis=1)
            V[S]["host_to"] = float(arclen(p)[int(np.argmin(d))]) if d.min() < 3.0 else 0.0
    T = {"R": float(MIDLINE_X - C["RCCA"][-1][0]), "L": float(C["LCCA"][-1][0] - MIDLINE_X)}
    NEAR = {}                       # 1.2: the fork's (CCA + ICA) closest approach to the midline, and its bifurcation's
    for S, key, cut, cca, ica in (("R", "RCCA", prov["host_cut_mm"], prov["cca_mm"], prov["ica_mm"]),
                                  ("L", "LCCA", L["host_cut_mm"], L["cca_mm"], L["ica_mm"])):
        c = C[key]
        s = arclen(c)
        lat = (MIDLINE_X - c[:, 0]) if S == "R" else (c[:, 0] - MIDLINE_X)
        m = (s >= cut) & (s <= cut + cca + ica)
        NEAR[S] = (float(lat[m].min()) if m.any() else float("nan"), float(lat[int(np.argmin(np.abs(s - cut - cca)))]))
    tri, mrec = load_mesh(root, C, prov, V)
    G = load_curves(v5root, NAMED) if (ghost and v5root and os.path.isdir(v5root)) else {}
    W = windows(prov)

    (fx, fy, fz) = lims
    wF, wS = fx[1] - fx[0], fy[1] - fy[0]
    fig = plt.figure(figsize=(16.0, 11.6))
    tlines = title_lines(prov, name, V, W, T, NEAR, extra_flags)
    top = 0.993 - 0.0128 * len(tlines) - 0.035
    gs = fig.add_gridspec(1, 2, width_ratios=[wF, wS], left=0.035, right=0.995, top=top, bottom=0.115, wspace=0.07)
    rec = {"name": name, "version": VERSION, "mesh": mrec, "checks": {}}
    for col, (vlab, i) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, col])
        if tri:
            mesh_shadow(ax, tri, i)
        for k, p in G.items():
            ax.plot(p[:, i], p[:, 2], color=GHOST_C, lw=0.9, ls=(0, (1.0, 1.6)), zorder=2)
        for k, p in C.items():
            if k not in NAMED and k not in SUBCL:
                ax.plot(p[:, i], p[:, 2], color=OTHER_C, lw=0.7, alpha=0.45, zorder=3)
        ends, sVA, sRL = [], {}, {}
        for S in (("L", "R") if i == 1 else ("R", "L")):        # SIDE view: the far side first, faded
            n0 = len(ax.lines)
            ends += draw_subclavians(ax, C, V, i, S)
            if S == "R":
                sRL[S] = draw_route(ax, C["RCCA"], prov["host_cut_mm"], prov["cca_mm"], prov["ica_mm"],
                                    float(g(X, "lower", "extend_mm", default=0) or 0), i)
            else:
                sRL[S] = draw_route(ax, C["LCCA"], L["host_cut_mm"], L["cca_mm"], L["ica_mm"],
                                    float(g(X, "lower_L", "extend_mm", default=0) or 0), i)
            draw_eca(ax, C, X, S, i)
            sVA[S] = draw_va(ax, C, S + "VA", V, X, i)
            q = junction_point(C, S)
            if q is not None:
                ax.plot(q[i], q[2], linestyle="none", zorder=10, **JUNC[V[S]["jkind"]])
            draw_windows(ax, C, [w for w in W if BRANCH_SIDE.get(w[0]) == S], i)
            if i == 1 and S == "L":
                for ln in ax.lines[n0:]:
                    ln.set_alpha(FADE)
        sR, sL = sRL["R"], sRL["L"]
        ax.plot(C_HOST[i], C_HOST[2], "*", ms=10, color="k", zorder=10)
        if i == 0:
            ax.axvline(MIDLINE_X, color="k", ls=":", lw=0.7, zorder=0.5)
        ax.set_xlim(*(fx if i == 0 else (fy[1], fy[0])))
        ax.set_ylim(*fz)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm, + = patient left)" if i == 0 else "y (mm, + = posterior; reversed)", fontsize=8)
        ax.set_ylabel("z (mm, + = superior)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(vlab, fontsize=9.5, loc="left")
        if i == 0:
            seamR = prov["host_cut_mm"] + prov["cca_mm"] + prov["ica_mm"]
            seamL = L["host_cut_mm"] + L["cca_mm"] + L["ica_mm"]
            bR = at(C["RCCA"], sR, prov["host_cut_mm"] + prov["cca_mm"])
            bL = at(C["LCCA"], sL, L["host_cut_mm"] + L["cca_mm"])
            qR, qL = at(C["RCCA"], sR, seamR + 8), at(C["LCCA"], sL, seamL + 8)
            jR, jL = junction_point(C, "R"), junction_point(C, "L")
            items = {"R": [], "L": []}
            for S, sq, bq, jq in (("R", qR, bR, jR), ("L", qL, bL, jL)):
                side = "right" if S == "R" else "left"
                items[S].append(((sq[0], sq[2]), siphon_text(prov, S, T[S])[0], SEG[3][1]))
                items[S].append(((bq[0], bq[2]), fork_text(prov, side, NEAR[S])[0], SEG[1][1]))
                items[S].append(((jq[0], jq[2]) if jq is not None else None, va_text(prov, side, S, V[S])[0], DONOR_C))
            callouts(ax, items["R"], "left")
            callouts(ax, items["L"], "right")
        else:
            ap = g(prov, "v6_build", "carotid", "ica_ap_post_mm", default={}) or {}
            # each label above its own terminus, the more posterior one (image left) shifted left: no crossing leaders
            yR, yL = C["RCCA"][-1][1], C["LCCA"][-1][1]
            post = "R" if yR >= yL else "L"
            tys = {}
            for S, y in (("R", yR), ("L", yL)):
                t = y + 0.13 * wS if S == post else y - 0.13 * wS
                tys[S] = min(max(t, fy[0] + 0.15 * wS), fy[1] - 0.15 * wS)
            ant = "L" if post == "R" else "R"              # 1.2: keep the two labels >= 0.34 wS apart
            if tys[post] - tys[ant] < 0.34 * wS:
                mid = min(max(0.5 * (tys[post] + tys[ant]), fy[0] + 0.32 * wS), fy[1] - 0.32 * wS)
                tys[post], tys[ant] = mid + 0.17 * wS, mid - 0.17 * wS
            for key, S in (("RCCA", "R"), ("LCCA", "L")):
                ty = tys[S]
                q = C[key][-1]
                val = ap.get(S + "ICA")
                if val is None:
                    continue
                flag = float(val) > 40
                ax.annotate("%sICA end: AP %+.0f mm%s" % (S, float(val), " (> 40, flagged)" if flag else ""),
                            xy=(q[1], q[2]), xytext=(ty, fz[1] - 0.025 * (fz[1] - fz[0])), fontsize=6.8, zorder=12,
                            color="#b00000" if flag else "k", ha="center", va="top",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.6, alpha=0.88),
                            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7, ls=(0, (1.2, 1.8)), shrinkB=3))
            ax.text(0.99, 0.01, "patient's left side faded (far side)", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7, color="0.35")

    # drawn vs provenance: route lengths and VA cuts inside their curves (mm)
    rec["checks"] = {"RCCA_minus_total": round(float(sR[-1] - float(prov.get("total_mm") or np.nan)), 3),
                     "LCCA_minus_total": round(float(sL[-1] - float(L.get("total_mm") or np.nan)), 3),
                     "RVA_len_minus_cut": round(float(sVA["R"][-1] - V["R"]["cut"]), 3),
                     "LVA_len_minus_cut": round(float(sVA["L"][-1] - V["L"]["cut"]), 3),
                     "d17_cut_ends": len(ends), "windows": len(W),
                     "ica_terminus_lat_mm": {k: round(x, 1) for k, x in T.items()},
                     "host_va_ostium_s_mm": {S: round(V[S]["host_to"], 1) for S in "RL" if "host_to" in V[S]},
                     "fork_min_lat_mm": {S: round(NEAR[S][0], 1) for S in "RL"},
                     "bif_lat_mm": {S: round(NEAR[S][1], 1) for S in "RL"}}

    fig.suptitle("\n".join(tlines), fontsize=8.2, y=0.993, va="top", x=0.005, ha="left")

    h = [Line2D([], [], color=c, lw=3, label=lab) for lab, c in SEG]
    h += [Line2D([], [], color="#17becf", lw=3, ls=(0, (2, 1.5)), label="ICA extension (white dashes)"),
          Line2D([], [], color=ECA_C, lw=3, label="fork ECA"),
          Line2D([], [], color=ECAEXT_C, lw=3, label="TopBrain ECA tip"),
          Line2D([], [], color=HOST_C, lw=3, label="host subclavian"),
          Line2D([], [], color=DSUB_C, lw=3, label="donor subclavian + junction + own distal stretch"),
          Line2D([], [], color=DSUB_C, lw=3, ls=(0, (2, 1.5)), label="D17 tube at the distal end"),
          Line2D([], [], color="k", marker="x", ls="none", ms=7, mew=2, label="donor's own cut end (D17)"),
          Line2D([], [], color=DONOR_C, lw=3, label="donor lower VA"),
          Line2D([], [], color=VAEXT_C, lw=3, label="TopBrain VA top"),
          Line2D([], [], color=VA_C, lw=3, label="host VA (no donor)"),
          Line2D([], [], ls="none", label="junction: donor's own", **JUNC["own"]),
          Line2D([], [], ls="none", label="junction: own + D16 tube", **JUNC["own_tube"]),
          Line2D([], [], ls="none", label="junction: v5's (rung 2/3)", **JUNC["v5"]),
          Line2D([], [], ls="none", label="host VA ostium", **JUNC["host"]),
          Line2D([], [], color="none", marker="o", ms=6, mfc="white", mec="k", label="seams / joins / VA origin + splice"),
          Line2D([], [], color="none", marker="s", ms=6, mfc=WIN_C, mec="k", label="bake tube window"),
          Line2D([], [], color="none", marker="s", ms=6, mfc=WIN_C, mec="#d62728", mew=1.8, label="D27 repair window"),
          Line2D([], [], color=OTHER_C, lw=2, alpha=0.6, label="other host branches"),
          Line2D([], [], color=GHOST_C, lw=1.2, ls=(0, (1, 1.6)), label="v5 centrelines (comparison)"),
          Line2D([], [], color="k", marker="*", ls="none", ms=9, label="host VA confluence"),
          Line2D([], [], color="none", marker="^", ms=7, mfc=SEG[3][1], mec="k", label="ICA terminus")]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=7.0, frameon=False, bbox_to_anchor=(0.5, 0.013),
               columnspacing=1.2, handlelength=2.2)
    fig.text(0.997, 0.001, "%s | anatomies_v6/%s" % (VERSION, name), ha="right", va="bottom", fontsize=5.5,
             color="0.45")
    tmp = out_png[:-4] + ".part.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, out_png)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["png"] = os.path.basename(out_png)
    return rec


# ------------------------------------------------------------------------------------------------------------ main
def shared_window(roots, v5dir):
    """One window for the whole set (the named centrelines and the subclavians of every v6 anatomy, plus v5's named
    centrelines for the comparison), so every image is at the same scale whatever subset or worker made it."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for r in roots:
        for p in load_curves(r, set(NAMED) | set(SUBCL)).values():
            lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
        r5 = os.path.join(v5dir, os.path.basename(r)) if v5dir else None
        if r5 and os.path.isdir(r5):
            for p in load_curves(r5, set(NAMED)).values():
                lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
    m = 6.0
    return [(float(np.floor(lo[k] - m)), float(np.ceil(hi[k] + m))) for k in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anatomies")
    ap.add_argument("out")
    ap.add_argument("--v5", default=None, help="anatomies_v5 (default: next to <anatomies>)")
    ap.add_argument("--no-v5", action="store_true", help="no v5 comparison lines (the window still includes v5)")
    ap.add_argument("--only", default=None, help="comma list of anatomy names")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--claims", default=None, help="claim dir (atomic mkdir per anatomy) for parallel workers")
    ap.add_argument("--records", default=None, help="write <name>.json per image (drawn-vs-provenance checks)")
    ap.add_argument("--build-json", default=None, help="build record with anatomies.<name>.flags (default: "
                    "<anatomies>/BUILD_v6.json if present); its flags are added to the title")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v5dir = a.v5 or os.path.join(os.path.dirname(os.path.abspath(a.anatomies)), "anatomies_v5")
    if not os.path.isdir(v5dir):
        v5dir = None
    roots = [os.path.dirname(r) for r in sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))]
    lims = shared_window(roots, v5dir)
    print("%s: %d anatomies, shared window x %s y %s z %s (v5 %s)" % (VERSION, len(roots), lims[0], lims[1], lims[2],
                                                                    v5dir), flush=True)
    bj = a.build_json or os.path.join(a.anatomies, "BUILD_v6.json")
    bflags = {}
    if os.path.isfile(bj):
        try:
            bflags = {n: list(r.get("flags") or []) for n, r in
                      (json.load(open(bj, encoding="utf-8")).get("anatomies") or {}).items() if isinstance(r, dict)}
        except (ValueError, AttributeError) as e:
            print("build record %s unreadable (%r): no extra flags" % (bj, e), flush=True)
    print("build-record flags: %s (%d anatomies)" % (bj if bflags else "none", len(bflags)), flush=True)
    if a.only:
        by = {os.path.basename(r): r for r in roots}
        roots = [by[n] for n in a.only.split(",")]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        roots = roots[i::n]
    if a.records:
        os.makedirs(a.records, exist_ok=True)
    if a.claims:
        os.makedirs(a.claims, exist_ok=True)
    nfail = 0
    for root in roots:
        name = os.path.basename(root)
        png = os.path.join(a.out, "carotid_v6_%s.png" % name)
        if a.skip_existing and os.path.isfile(png):
            print("skip %s" % name, flush=True)
            continue
        if a.claims:
            cdir = os.path.join(a.claims, name)
            try:
                os.mkdir(cdir)
                open(os.path.join(cdir, "by"), "w").write("%s %s\n" % (socket.gethostname(), time.ctime()))
            except FileExistsError:
                if time.time() - os.path.getmtime(cdir) < 1800:
                    print("claimed elsewhere %s" % name, flush=True)
                    continue
                os.utime(cdir)
        try:
            r5 = os.path.join(v5dir, name) if v5dir else None
            rec = render(root, r5, lims, png, ghost=not a.no_v5, extra_flags=bflags.get(name))
            rec["window"] = lims
            if a.records:
                json.dump(rec, open(os.path.join(a.records, name + ".json"), "w"), indent=1)
            print("wrote %s  %.1fs  checks %s" % (os.path.basename(png), rec["seconds"], rec["checks"]), flush=True)
        except Exception as e:  # keep going; the log names the anatomy
            nfail += 1
            import traceback
            traceback.print_exc()
            print("FAILED %s: %r" % (name, e), flush=True)
    print("done (%d failed)" % nfail, flush=True)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
