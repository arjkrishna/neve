#!/usr/bin/env python3
"""Visual QC of the v5 anatomies (v4 + real donor lower vertebral arteries), two per image.

The house style of figure_carotid_anatomies_v4.py: same view, same colours for the carotid routes and ECAs, one cube
shared by the whole set so every image is at the same scale, and segment boundaries taken from provenance.json, so
the seams shown are the ones the graft used. v5 replaces the shared host LOWER vertebral artery with a real donor VA
from thin-slice CTA (strategy B: host subclavian kept, donor VA from its origin to a splice in V3, v4's TopBrain
distal piece above), so each VA is drawn in three pieces:

    RCCA / LCCA  host grey | CCA blue | ICA cyan | siphon orange, a dot per seam            (as v4)
    RECA / LECA  fork ECA magenta, TopBrain ECA tip purple (dot at the join)               (as v4)
    RVA / LVA    host subclavian grey | DONOR lower VA dark teal | TopBrain VA brown,
                 a dot at the donor's origin seam (xform.vaC_*.lo_mm) and at the splice (hi_mm)
                 a VA kept from v4 (no gate-passing donor pose): host VA green | TopBrain VA brown, as v4,
                 and flagged in the title
                 handle-rule tube windows (bake_v5.va_tubed): a hollow stretch with a yellow square
    other        host branches, thin red                                                  (as v4)

The baked collision mesh (vessel_architecture_collision.obj) is drawn under the centerlines, shaded by the piece of
its nearest centerline: right carotid, left carotid, donor VA, TopBrain VA, kept v4 VA, host (the subclavian stretch
of each VA counts as host). The title adds, per side, the donor (id, side, source, mirrored or not, length from
origin to splice), whether the VA was moved by the route13 fix (provenance v5_va.route13, variant from BUILD_v5.json),
and the tube-window length if any. The donor colour was checked against every house colour it sits next to
(OKLab dE x 100 >= 16 under protan / deutan / tritan simulation; teal and dark orange both collide with tab10).

    docker run --rm <mounts> eve-training-fixed
      python3 monitoring/figure_carotid_anatomies_v5.py <anatomies_v5> <out> [--only a,b,...] [--shard i/n]
                                                         [--skip-existing]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# the repo's eve + eve_bench when /repo is mounted (eve_bench imports the repo's eve), else the image's copy
if os.path.isdir("/repo/eve_bench"):
    sys.path[:0] = ["/repo/eve", "/repo/eve_bench"]
else:
    sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

ELEV, AZIM = 16, -68
SEG = [("host", "#7f7f7f"), ("CCA", "#1f77b4"), ("ICA", "#17becf"), ("siphon", "#ff7f0e")]
ECA_C, ECAEXT_C = "#e377c2", "#9467bd"
VA_C, VAEXT_C = "#2ca02c", "#8c564b"          # v4: host VA green (now only a VA kept from v4), TopBrain VA brown
HOST_C = "#7f7f7f"                            # host subclavian stretch of each VA
DONOR_C = "#00332e"                           # donor lower VA (thin-slice CTA), dark teal
WIN_C = "#ffd400"                             # tube-window marker
OTHER_C = "#d62728"
MESH = {"right": ("#1f77b4", 0.13), "left": ("#17becf", 0.13), "donor": (DONOR_C, 0.15),
        "tbva": (VAEXT_C, 0.12), "v4va": (VA_C, 0.12), "other": ("#d62728", 0.10)}
SRC = {"ia172": "Head-CTA 172", "song": "Song 2024", "vmr0248": "VMR 0248"}
SIDES = (("right", "R"), ("left", "L"))


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def load_case(root):
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    g = {"other": []}
    for b in br:
        n = str(getattr(b, "name", "")).upper()
        c = np.asarray(b.coordinates, float)
        for key in ("RCCA", "LCCA", "RECA", "LECA", "RVA", "LVA"):
            if key in n:
                g[key] = c
                break
        else:
            g["other"].append(c)
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    return g, prov


def va_pieces(prov, S):
    """(kind, lo, hi) of one VA: kind 'donor' with the origin seam lo and the splice hi, or 'v4' with hi = the
    TopBrain cut (lo None)."""
    X = prov["xform"]
    D = X.get("vaC_" + S)
    if D is not None:
        return "donor", float(D["lo_mm"]), float(D["hi_mm"])
    return "v4", None, float(X["va_" + S]["cut_mm"])


def mesh_groups(root, g, prov, target=12000):
    """The baked mesh, decimated for drawing, split by the piece of its nearest centerline."""
    import pyvista as pv
    from scipy.spatial import cKDTree
    m = pv.read(os.path.join(root, "vessel_architecture_collision.obj")).extract_surface().triangulate()
    if m.n_cells > target:
        m = m.decimate(1.0 - target / float(m.n_cells), volume_preservation=True)
    v = np.asarray(m.points, np.float32)
    f = np.asarray(m.faces, np.int64).reshape(-1, 4)[:, 1:]
    keys, pts = [], []
    for grp, names in (("right", ("RCCA", "RECA")), ("left", ("LCCA", "LECA"))):
        for n in names:
            keys += [grp] * len(g[n]); pts.append(g[n])
    for _, S in SIDES:
        c = g[S + "VA"]
        s = arclen(c)
        kind, lo, hi = va_pieces(prov, S)
        if kind == "donor":
            k = np.where(s < lo, "other", np.where(s <= hi, "donor", "tbva"))
        else:
            k = np.where(s <= hi, "v4va", "tbva")
        keys += list(k); pts.append(c)
    for c in g["other"]:
        keys += ["other"] * len(c); pts.append(c)
    _, j = cKDTree(np.concatenate(pts)).query(v[f].mean(axis=1))
    keys = np.asarray(keys)
    return {k: (v, f[keys[j] == k]) for k in MESH}


def dot(ax, p):
    ax.scatter(*p, s=26, facecolor="white", edgecolor="k", lw=1.0, zorder=9)


def piece(ax, c, s, lo, hi, col, lw, z=7):
    """The stretch lo <= s <= hi of a centerline, the end points snapped to the nearest samples so that adjacent
    pieces share a point and draw without a gap."""
    a = int(np.argmin(np.abs(s - lo))); b = int(np.argmin(np.abs(s - hi)))
    if b > a:
        ax.plot(c[a:b + 1, 0], c[a:b + 1, 1], c[a:b + 1, 2], color=col, lw=lw, zorder=z)


def route(ax, c, cut, cca, ica):
    """A carotid route in its four segments, with the seams the graft used."""
    s = arclen(c)
    edges = [0.0, cut, cut + cca, cut + cca + ica, s[-1] + 1]
    for (lab, col), lo, hi in zip(SEG, edges[:-1], edges[1:]):
        m = (s >= lo - 1e-6) & (s <= hi)
        if m.sum() > 1:
            ax.plot(c[m, 0], c[m, 1], c[m, 2], color=col, lw=2.3, zorder=8)
    for e in edges[1:4]:
        dot(ax, c[int(np.argmin(np.abs(s - e)))])
    return s[-1] - edges[3]


def split_at(ax, c, at_mm, col_a, col_b, lw):
    """A branch drawn in two colours, split at an arclength, with a dot there."""
    s = arclen(c)
    k = int(np.argmin(np.abs(s - at_mm)))
    ax.plot(c[:k + 1, 0], c[:k + 1, 1], c[:k + 1, 2], color=col_a, lw=lw, zorder=7)
    if k < len(c) - 1:
        ax.plot(c[k:, 0], c[k:, 1], c[k:, 2], color=col_b, lw=lw, zorder=7)
        dot(ax, c[k])


def windows(prov, S):
    """Handle-rule tube windows on one VA: [(lo, hi, 'donor'|'TopBrain'|'v4')], and whole-section tubes."""
    t = prov.get("bake_v5", {}).get("va_tubed", {})
    out, whole = [], []
    for key, lab in (("vaC_" + S, "donor"), ("va_" + S, "TopBrain")):
        r = t.get(key) or {}
        for w in r.get("windows_mm") or []:
            out.append((float(w[0]), float(w[1]), lab))
        if r.get("whole_tube"):
            whole.append(lab)
    return out, whole


def draw_va(ax, c, prov, S):
    s = arclen(c)
    kind, lo, hi = va_pieces(prov, S)
    if kind == "donor":
        piece(ax, c, s, 0.0, lo, HOST_C, 1.8)
        piece(ax, c, s, lo, hi, DONOR_C, 2.1, z=7.2)
        piece(ax, c, s, hi, s[-1], VAEXT_C, 1.8)
        dot(ax, c[int(np.argmin(np.abs(s - lo)))])
        dot(ax, c[int(np.argmin(np.abs(s - hi)))])
    else:
        split_at(ax, c, hi, VA_C, VAEXT_C, 1.8)
    ws, _ = windows(prov, S)
    for a, b, _ in ws:
        i = int(np.argmin(np.abs(s - a))); j = int(np.argmin(np.abs(s - b)))
        if j > i:
            ln, = ax.plot(c[i:j + 1, 0], c[i:j + 1, 1], c[i:j + 1, 2], color="white", lw=1.0, zorder=7.6)
            ln.set_path_effects([pe.Stroke(linewidth=3.4, foreground="k"), pe.Normal()])
        m = c[int(np.argmin(np.abs(s - 0.5 * (a + b))))]
        ax.scatter(*m, s=30, marker="s", facecolor=WIN_C, edgecolor="k", lw=0.9, zorder=9.5)
    return s[-1], s[-1] - hi          # whole VA length, TopBrain piece length as drawn


def va_label(prov, side, S, kept_by):
    V = prov["v5_va"]
    kind, lo, hi = va_pieces(prov, S)
    if kind == "v4":
        return "%sVA: KEPT FROM v4%s" % (S, " (at %s)" % kept_by if kept_by else "")
    d = V.get(side, {})
    did = prov["xform"]["vaC_" + S].get("donor_id", "?")
    src = d.get("donor_source") or did.split("_")[0]
    mir = any(float(x) < 0 for x in (d.get("donor_mirror") or [1, 1, 1]))
    return "%sVA: %s %s%s (%s) %.0f mm" % (S, did.replace("song_", ""), prov["xform"]["vaC_" + S].get("donor_side", "?"),
                                           " mirrored" if mir else "", SRC.get(src, src), hi - lo)


def flag_line(prov, name, build):
    """Route13 move, tube windows, kept sides' reason, carotid-section windows: '' if none."""
    V, B = prov["v5_va"], prov.get("bake_v5", {})
    parts = []
    r13 = V.get("route13")
    if r13:
        var = ((build.get("route13") or {}).get("anatomies") or {}).get(name, {}).get("variant")
        parts.append("route13: %s moved%s%s" % ("+".join(s[0].upper() + "VA" for s in r13.get("targets", [])),
                                                 " (%s)" % var if var else "",
                                                 ", TopBrain piece shifted" if r13.get("tb_shift") else ""))
    for side, S in SIDES:
        ws, whole = windows(prov, S)
        if ws:
            by = {}
            for a, b, lab in ws:
                by[lab] = by.get(lab, 0.0) + (b - a)
            parts.append("%sVA tube %s" % (S, " + ".join("%.1f mm %s" % (v, k) for k, v in by.items())))
        for lab in whole:
            parts.append("%sVA whole %s section tube" % (S, lab))
    if "carotid_changed" in (B.get("flags") or []):
        sup = B.get("suppressed_stretches_mm") or {}
        # rung C put carotid sections in the handle windows; the record lists the suppressed stretches together
        # with the ones v4's own bake already had, so they are shown as recorded, not as v5's alone
        parts.append("carotid_changed (rung C: carotid sections in the handle windows; suppressed, v4's included: %s mm)"
                     % ", ".join("%s %.1f-%.1f" % (k, w[0], w[1]) for k, ws in sup.items() for w in ws))
    for f in B.get("flags") or []:
        if f != "carotid_changed":
            parts.append(f)
    return "   ".join(parts)


def draw(ax, root, case, lims, legend, build, kept):
    g, prov = case
    name = os.path.basename(root)
    # draw in zorder, not mplot3d's per-artist depth estimate: the translucent
    # mesh would otherwise land on top of the centerlines and turn the orange
    # siphon brown under a blue shell
    ax.computed_zorder = False
    for k, (v, f) in mesh_groups(root, g, prov).items():
        if f.size:
            col, a = MESH[k]
            ax.add_collection3d(Poly3DCollection(v[f], facecolor=(*mcolors.to_rgb(col), a),
                                                 edgecolor=(0.25, 0.25, 0.3, 0.04), linewidth=0.1, zorder=1))
    for c in g["other"]:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=OTHER_C, lw=0.8, alpha=0.45, zorder=4)

    L, X, VT = prov["left"], prov["xform"], prov["vertebrals"]
    sip_r = route(ax, g["RCCA"], prov["host_cut_mm"], prov["cca_mm"], prov["ica_mm"])
    sip_l = route(ax, g["LCCA"], L["host_cut_mm"], L["cca_mm"], L["ica_mm"])
    for side, key, ext in (("right", "RECA", "ecaext_R"), ("left", "LECA", "ecaext_L")):
        if ext in X:
            split_at(ax, g[key], X[ext]["join_at_mm"], ECA_C, ECAEXT_C, 2.2)
        else:
            c = g[key]
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=ECA_C, lw=2.2, zorder=7)
    check, tbl = [], {}
    for side, S in SIDES:
        tot, tbl[side] = draw_va(ax, g[S + "VA"], prov, S)
        ref = prov["v5_va"].get(side, {}).get("total_mm") if va_pieces(prov, S)[0] == "donor" else VT[side].get("total_mm")
        if ref is not None:
            check.append("%sVA %.2f" % (S, tot - float(ref)))
        tref = prov["v5_va"].get(side, {}).get("tb_len_used_mm") or VT[side].get("donor_len_mm")
        if tref is not None:
            check.append("%sTB %.2f" % (S, tbl[side] - float(tref)))

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    for i, nm in enumerate("xyz"):
        getattr(ax, "set_%slabel" % nm)("%s (vessel-CS, mm)" % nm, fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    tips = [s[0].upper() for s in ("right", "left") if prov["eca_extension"][s].get("extended")]
    lines = ["%s  (%s)" % (name, prov["handedness"]),
             "left fork %s%s   siphons R %.0f / L %.0f mm   ECA tips %s   TopBrain VA %.0f / %.0f mm"
             % (L["lower"], " (substitute)" if L.get("lower_fallback") else "", sip_r, sip_l,
                " + ".join(tips) or "none", tbl["right"], tbl["left"]),
             "   ".join(va_label(prov, side, S, kept.get((name, side))) for side, S in SIDES)]
    fl = flag_line(prov, name, build)
    if fl:
        lines.append(fl)
    ax.set_title("\n".join(lines), fontsize=8)
    if legend:
        h = [Line2D([], [], color=col, lw=4, label=lab) for lab, col in SEG]
        h += [Line2D([], [], color=ECA_C, lw=4, label="fork ECA"),
              Line2D([], [], color=ECAEXT_C, lw=4, label="TopBrain ECA tip"),
              Line2D([], [], color=HOST_C, lw=3, label="host subclavian (VA parent)"),
              Line2D([], [], color=DONOR_C, lw=3, label="donor lower VA (thin-slice CTA)"),
              Line2D([], [], color=VAEXT_C, lw=3, label="TopBrain VA"),
              Line2D([], [], color=VA_C, lw=3, label="host VA kept from v4 (flagged)"),
              Line2D([], [], color=OTHER_C, lw=3, alpha=0.6, label="other host branches"),
              Line2D([], [], color="none", marker="o", markersize=7,
                     markerfacecolor="white", markeredgecolor="k", label="seams / joins / cuts / VA origin + splice"),
              Line2D([], [], color="none", marker="s", markersize=6,
                     markerfacecolor=WIN_C, markeredgecolor="k", label="tube window (handle rule)")]
        ax.legend(handles=h, loc="upper left", fontsize=7)
    return check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anatomies")
    ap.add_argument("out")
    ap.add_argument("--only", default=None, help="comma list, paired in the order given")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    roots = [os.path.dirname(r) for r in sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))]
    # One shared cube across the WHOLE set, computed before selecting, so every
    # image is at identical scale no matter which subset or worker made it.
    P = []
    for r in roots:
        g, _ = load_case(r)
        P += [g[k] for k in ("RCCA", "LCCA", "RECA", "LECA", "RVA", "LVA")]
    P = np.concatenate(P, axis=0)
    ctr = (P.min(0) + P.max(0)) / 2.0
    rad = float(np.max(P.max(0) - P.min(0))) / 2.0 * 1.05
    lims = [(ctr[i] - rad, ctr[i] + rad) for i in range(3)]
    print("%d anatomies, shared cube half-size %.1f mm, centre %s" % (len(roots), rad, np.round(ctr, 1).tolist()),
          flush=True)

    build = {}
    bp = os.path.join(a.anatomies, "BUILD_v5.json")
    if os.path.isfile(bp):
        build = json.load(open(bp, encoding="utf-8"))
    kept = {}
    for row in (build.get("v4_va_kept_sides") or {}).get("sides", []):
        kept[(row["anatomy"], row["side"])] = row.get("by")

    if a.only:
        by = {os.path.basename(r): r for r in roots}
        roots = [by[n] for n in a.only.split(",")]
    pairs = [roots[i:i + 2] for i in range(0, len(roots), 2)]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        pairs = pairs[i::n]

    for grp in pairs:
        nm = "__".join(os.path.basename(r) for r in grp)
        path = os.path.join(a.out, "carotid_v5_%s.png" % nm[:110])
        if a.skip_existing and os.path.isfile(path):
            print("skip %s" % os.path.basename(path), flush=True)
            continue
        fig = plt.figure(figsize=(8.5 * len(grp), 9.5))
        checks = []
        for j, root in enumerate(grp):
            ax = fig.add_subplot(1, len(grp), j + 1, projection="3d")
            checks.append(draw(ax, root, load_case(root), lims, legend=(j == 0), build=build, kept=kept))
        fig.tight_layout()
        tmp = path[:-4] + ".part.png"
        fig.savefig(tmp, dpi=130)
        plt.close(fig)
        os.replace(tmp, path)
        print("wrote %s   drawn - provenance (mm; VA whole, TB piece): %s" % (os.path.basename(path), checks), flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
