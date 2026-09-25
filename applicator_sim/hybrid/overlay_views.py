"""Overlay stills: the dataset's BT (post-insertion) segmentations on the two views of a run's seated state, to show
what is misaligned, where, and the likely cause.

    python hybrid/bt_overlay.py prep --tag G32                                 # host py3.13, once per run
    APPSIM_OUT=<render tree> py -3.11 hybrid/overlay_views.py --tag G32        # -> figs/labeled/<tag>_step<k>_overlay_*.png

Sagittal: on the device's sagittal plane, the simulation's section outlines (solid) and the scan's (dashed), per
organ in the organ's colour; the device in black and the intrauterine canal in cyan, the same way.  Oblique: the
simulation drawn see-through as in the labelled view, the scan's structures as outlines.  Each callout sits where
that structure's scan contour or surface is furthest from the simulation, with an arrow from the simulation to
the scan, the measured offset and the likely cause.  Every number comes from figs/overlay_prep/<tag>_PELVIS/
stats.json (bt_overlay.py) or is computed here from the run's files.  Patient-derived: local only."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402
import animate_hybrid as AH  # noqa: E402
import label_views as LV  # noqa: E402

P = AH.P
ORG = ["corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"]
NAMES = dict(corpus="uterus (corpus)", cervix="HR-CTV / cervix", vagina="vagina", bladder="bladder", rectum="rectum",
             sigmoid="sigmoid colon", device="applicator", canal="intrauterine canal")
COL_BT_CANAL = (0.0, 0.38, 0.55)
COL_DEV = (0.08, 0.08, 0.10)


def load_prep(tag):
    d = "%s/overlay_prep/%s_PELVIS" % (P["figs"], tag)
    if not os.path.exists(d + "/stats.json"):
        raise SystemExit("run `python hybrid/bt_overlay.py prep --tag %s` (host py3.13) first" % tag)
    stats = AH.load_json(d + "/stats.json")
    bt = {n: tuple(np.asarray(x) for x in geom.read_obj("%s/%s.obj" % (d, n)))
          for n in ORG + ["applicator", "ovoid"]}
    return stats, bt, np.load(d + "/IUcanal_centreline_bt.npy")


def line_col(c, f=0.70):
    return tuple(f * np.asarray(c, float))


def polylines(pd):
    """Ordered polylines (3-D points) of a pyvista line PolyData."""
    if pd.n_points == 0:
        return []
    try:
        pd = pd.strip(join=True)
    except Exception:
        pass
    L, out, i = np.asarray(pd.lines), [], 0
    pts = np.asarray(pd.points)
    while i < len(L):
        n = int(L[i])
        out.append(pts[L[i + 1:i + 1 + n]])
        i += n + 1
    return out


def dirw(v, thr=3.0):
    """'11 mm right, 4 mm anterior' for a preBT-world RAS vector."""
    names = (("right", "left"), ("anterior", "posterior"), ("superior", "inferior"))
    v = np.asarray(v, float)
    parts = ["%.0f mm %s" % (abs(v[k]), names[k][0] if v[k] > 0 else names[k][1]) for k in np.argsort(-np.abs(v))
             if abs(v[k]) >= thr]
    return ", ".join(parts) if parts else "within %.0f mm" % thr


def findings(stats, canal_div):
    """Callout text per structure: the measured misalignment, then the likely cause with its evidence."""
    S, M, sc, pe = stats["structures"], stats["measurements"], stats["scores_PELVIS"], stats.get("pose_rule_error") or {}
    res = {n: S[n]["residual_final_minus_bt_mm"] for n in S}
    w = M["vagina_width_lr_ap_mm"].get("+0", {})
    hs = M["hrctv_lr_split_cc"]
    ro = float(np.linalg.norm(M["ring_offset_mm"]))
    f = dict(
        corpus=["Dice %.2f; model within %.0f mm of the scan" % (sc["corpus"]["dice"], S["corpus"]["residual_len"]),
                "aligned: the pose rule places the rigid uterus"],
        cervix=["Dice %.2f; model %s of the scan" % (sc["cervix"]["dice"], dirw(res["cervix"])),
                "lifted correctly (%.0f mm up, scan %.0f), but the tandem" % (S["cervix"]["model_displacement_mm"][2],
                                                                            S["cervix"]["true_displacement_mm"][2]),
                "passes on the wrong side of the tumour: the model's",
                "lower %.0f mm of canal is extrapolated; scan halves" % M["canal_label_starts_above_os_mm"],
                "right %.0f→%.0f cc, left %.0f→%.0f cc (not just shrinkage)" % (hs["pre_right"], hs["bt_right"],
                                                                                 hs["pre_left"], hs["bt_left"])],
        vagina=["Dice %.2f (filled); model %s of the scan" % (sc.get("vagina_filled", sc["vagina"])["dice"],
                                                             dirw(res["vagina"])),
                "vault %.0f mm wide (L-R) vs %.0f in the scan: fornix flare;" % (w.get("model", [np.nan])[0],
                                                                                w.get("BT", [np.nan])[0]),
                "real distension is posterior (packing behind the ring)"],
        bladder=["Dice %.2f; model %s of the scan" % (sc["bladder"]["dice"], dirw(res["bladder"])),
                 "%+.0f cc filling change between the scans (not modelled)" % (S["bladder"]["vol_bt_label_cc"]
                                                                              - S["bladder"]["vol_pre_label_cc"])],
        rectum=["Dice %.2f; model %s of the scan" % (sc["rectum"]["dice"], dirw(res["rectum"])),
                "real move: %s; model: %s" % (dirw(S["rectum"]["true_displacement_mm"], 2.0),
                                             dirw(S["rectum"]["model_displacement_mm"], 2.0)),
                "no posterior packing or rectovaginal coupling in the model"],
        sigmoid=["Dice %.2f; model %s of the scan" % (sc["sigmoid"]["dice"], dirw(res["sigmoid"])),
                 "bowel moved %.0f mm, volume %+.0f %%: not applicator-driven" % (
                     S["sigmoid"]["true_displacement_len"],
                     100.0 * (S["sigmoid"]["vol_bt_label_cc"] / S["sigmoid"]["vol_pre_label_cc"] - 1.0))],
        device=["tandem %.1f° / %.1f mm from the real one; ring %.1f mm" % (
                    pe.get("axis_angle_deg", np.nan), (pe.get("flange_offset_mm") or {}).get("total", np.nan), ro),
                "from the scan's ovoids: placement is not the problem"],
        canal=["scan canal within %.1f mm of the real tandem;" % M["bt_canal_vs_real_tandem_max_mm"],
               "model's rigid uterus keeps its curve: %.0f mm off" % canal_div])
    return f


def gap_anchor(A, B):
    """The point of A (scan) at the 95th percentile of its distance to B (simulation), and B's nearest point."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = np.array([np.linalg.norm(B - a, axis=1).min() for a in A])
    i = int(np.argsort(d)[int(0.95 * (len(d) - 1))])
    j = int(np.argmin(np.linalg.norm(B - A[i], axis=1)))
    return A[i], B[j], float(d[i])


def canal_divergence(st, cl):
    """Largest distance of the model's carried canal (cavity part) from the scan's canal centreline, with the pair."""
    C = st.canal[st.i_ios:]
    d = np.array([np.linalg.norm(cl - c, axis=1).min() for c in C])
    k = int(np.argmax(d))
    return float(d[k]), C[k], cl[int(np.argmin(np.linalg.norm(cl - C[k], axis=1)))]


# ------------------------------------------------------------------------------------------------ views
def build(st, view, Sw, Sh, prep):
    stats, bt, cl = prep
    pv = AH._pv()
    cut = view == "sagittal"

    def add_actors(pl, pv_, ctx):
        if cut:
            return
        for n in ORG:
            m = AH.poly(pv_, *bt[n]).smooth_taubin(n_iter=80, pass_band=0.02)
            pl.add_silhouette(m, color=line_col(st.col[n], 0.62), line_width=3.2)
        for n in ("applicator", "ovoid"):
            pl.add_silhouette(AH.poly(pv_, *bt[n]).smooth_taubin(n_iter=40, pass_band=0.05), color=COL_DEV,
                              line_width=2.4)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))]
        for lo in np.arange(0.0, s[-1], 5.0):                  # a dashed tube: 3 mm on, 2 mm off
            m = (s >= lo) & (s <= lo + 3.0)
            if m.sum() >= 2:
                pl.add_mesh(pv_.lines_from_points(cl[m]).tube(radius=0.9), color=COL_BT_CANAL)

    def after(proj, info):
        out = dict(lines=[], callouts={})
        x_cut = info["x_cut"]
        div, c_m, c_b = canal_divergence(st, cl)
        out["canal_div"] = div
        if cut:
            for n in ORG:
                Lm = polylines(AH.poly(pv, *st.surf[n]).slice(normal="x", origin=(x_cut, 0, 0)))
                Lb = polylines(AH.poly(pv, *bt[n]).slice(normal="x", origin=(x_cut, 0, 0)))
                for L in Lm:
                    out["lines"].append(("model", n, np.array([proj(q) for q in L])))
                for L in Lb:
                    out["lines"].append(("bt", n, np.array([proj(q) for q in L])))
                if Lm and Lb:
                    pb, pm, d = gap_anchor(np.vstack(Lb), np.vstack(Lm))
                    out["callouts"][n] = (proj(pb), proj(pm), d)
            for p_, (V, F) in st.parts.items():
                for L in polylines(AH.poly(pv, V, F).slice(normal="x", origin=(x_cut, 0, 0))):
                    out["lines"].append(("model", "packing" if p_ == "packing" else "device",
                                         np.array([proj(q) for q in L])))
            Lbd = []
            for n in ("applicator", "ovoid"):
                for L in polylines(AH.poly(pv, *bt[n]).slice(normal="x", origin=(x_cut, 0, 0))):
                    out["lines"].append(("bt", "device", np.array([proj(q) for q in L])))
                    Lbd.append(L)
            oc = np.asarray(st.dev["ovoid_centres_mm"], float).mean(0)
            if Lbd:
                B = np.vstack(Lbd)
                hb = (B - oc) @ st.a
                sel = np.abs(hb) < 12.0
                q = B[sel][np.argmin(np.linalg.norm(B[sel] - oc, axis=1))] if sel.any() else B[0]
                out["callouts"]["device"] = (proj(q), proj(q), 0.0)
        else:
            cam = info["cam"]
            for n in ORG:
                Vb, Vm = bt[n][0], st.surf[n][0]
                Pb = np.array([proj(q) for q in Vb])
                inside = (Pb[:, 0] > 20) & (Pb[:, 0] < Sw - 20) & (Pb[:, 1] > 20) & (Pb[:, 1] < Sh - 20)
                sub = Vb[inside][:: max(1, int(inside.sum() // 2500))]
                if len(sub) == 0:
                    continue
                pb, pm, d = gap_anchor(sub, Vm[:: max(1, len(Vm) // 4000)])
                out["callouts"][n] = (proj(pb), proj(pm), d)
            Vo = bt["ovoid"][0]
            q = Vo[np.argmin(np.linalg.norm(Vo - cam, axis=1))]
            out["callouts"]["device"] = (proj(q), proj(q), 0.0)
        out["callouts"]["canal"] = (proj(c_b), proj(c_m), div)
        out["model_canal"] = np.array([proj(q) for q in st.canal])
        out["bt_canal"] = np.array([proj(q) for q in cl])
        return out

    img, _, info, ov = LV.render(st, view, Sw, Sh, add_actors=add_actors, after=after, labels=False)
    return img, info, ov


# ------------------------------------------------------------------------------------------------ compose
def compose(img, info, ov, view, st, stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    Sh, Sw = img.shape[:2]
    mL = mR = 600
    top, bot = 120, 330
    W, H = mL + Sw + mR, top + Sh + bot
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    base = img.astype(float)
    if view == "sagittal":                          # fade the render so the outlines carry the comparison
        base = 0.50 * base + 0.50 * 255.0
    ax.imshow(base.astype(np.uint8), extent=(mL, mL + Sw, top + Sh, top), aspect="auto", zorder=0)
    off = np.array([mL, top])
    halo = [pe.withStroke(linewidth=4.2, foreground="white")]
    for kind, n, L in ov["lines"]:
        c = COL_DEV if n == "device" else ((0.62, 0.50, 0.30) if n == "packing" else line_col(st.col[n]))
        ls = "-" if kind == "model" else (0, (5, 3))
        lw = 1.6 if n == "packing" else (2.3 if kind == "model" else 2.6)
        ax.plot(L[:, 0] + mL, L[:, 1] + top, color=c, lw=lw, ls=ls, zorder=3, path_effects=halo,
                solid_capstyle="round")
    if view == "sagittal":
        mc, bc = ov["model_canal"] + off, ov["bt_canal"] + off
        ax.plot(mc[:, 0], mc[:, 1], color=LV.COL_CANAL, lw=2.6, zorder=4, path_effects=halo)
        ax.plot(bc[:, 0], bc[:, 1], color=COL_BT_CANAL, lw=2.6, ls=(0, (5, 3)), zorder=4, path_effects=halo)
    F = findings(stats, ov["canal_div"])
    labels = []
    for key, (pb, pm, d) in ov["callouts"].items():
        col = COL_DEV if key == "device" else (COL_BT_CANAL if key == "canal" else line_col(st.col[key], 0.72))
        name = NAMES[key] + ("" if d < 1.0 else "  (gap here %.0f mm)" % d)
        labels.append(dict(name=name, lines=F[key], col=col, pb=np.asarray(pb) + off, pm=np.asarray(pm) + off))
    for Lb in labels:
        Lb["side"] = "L" if Lb["pb"][0] < mL + 0.5 * Sw else "R"
        Lb["h"] = 24.0 + 17.0 * len(Lb["lines"])
    for side in ("L", "R"):
        grp = sorted([Lb for Lb in labels if Lb["side"] == side], key=lambda Lb: Lb["pb"][1])
        ys = LV.place([Lb["pb"][1] for Lb in grp], [Lb["h"] for Lb in grp], top + 6, top + Sh - 6, gap=12.0)
        for Lb, y in zip(grp, ys):
            xt = mL - 22 if side == "L" else mL + Sw + 22
            ha = "right" if side == "L" else "left"
            yn = y - 0.5 * Lb["h"] + 12
            ax.text(xt, yn, Lb["name"], ha=ha, va="center", fontsize=13.5, fontweight="bold", color=Lb["col"], zorder=8)
            for i, s in enumerate(Lb["lines"]):
                ax.text(xt, yn + 21 + 17 * i, s, ha=ha, va="center", fontsize=10.8, color=(0.25, 0.25, 0.28), zorder=8)
            x0 = mL - 14 if side == "L" else mL + Sw + 14
            ax.plot([x0, Lb["pb"][0]], [yn, Lb["pb"][1]], color=Lb["col"], lw=1.1, zorder=6,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
            if np.linalg.norm(Lb["pb"] - Lb["pm"]) > 9.0:        # arrow: from the simulation to the scan
                ax.annotate("", xy=tuple(Lb["pb"]), xytext=tuple(Lb["pm"]), zorder=9,
                            arrowprops=dict(arrowstyle="-|>", color=Lb["col"], lw=2.2, mutation_scale=16,
                                            path_effects=[pe.withStroke(linewidth=4.0, foreground="white")]))
            ax.plot([Lb["pb"][0]], [Lb["pb"][1]], marker="o", ms=7, mfc=Lb["col"], mec="white", mew=1.3, zorder=10)
    # ---- titles and orientation
    ttl = dict(sagittal="Simulation vs BT scan: sagittal section through the device",
               oblique="Simulation vs BT scan: oblique 3-D view")[view]
    sub = dict(sagittal="Solid outlines: the simulation (run %s, seated).  Dashed outlines: the dataset's post-insertion (BT) "
                        "segmentation, registered by the pelvis.  Anterior left, superior up." % st.tag,
               oblique="See-through surfaces: the simulation (run %s, seated).  Outlines: the dataset's post-insertion (BT) "
                       "segmentation, registered by the pelvis." % st.tag)[view]
    ax.text(W / 2, 40, ttl, ha="center", va="center", fontsize=21, fontweight="bold")
    ax.text(W / 2, 80, sub, ha="center", va="center", fontsize=12.5, color=(0.25, 0.25, 0.28))
    if view == "sagittal":
        ax.text(mL + 16, top + 22, "◀ anterior", ha="left", va="center", fontsize=12, color=(0.3, 0.3, 0.3))
        ax.text(mL + Sw - 16, top + 22, "posterior ▶", ha="right", va="center", fontsize=12, color=(0.3, 0.3, 0.3))
        ppm = info["px_per_mm"]
        x1, yb = mL + Sw - 40, top + Sh - 30
        ax.plot([x1 - 20 * ppm, x1], [yb, yb], color="black", lw=3)
        ax.text(x1 - 10 * ppm, yb - 14, "20 mm", ha="center", va="center", fontsize=11.5)
    # ---- key
    ky = top + Sh + 32
    x = mL - 60
    if view == "sagittal":
        ax.plot([x, x + 46], [ky, ky], color=(0.3, 0.3, 0.3), lw=2.3)
        ax.text(x + 56, ky, "simulation", va="center", fontsize=12)
        x += 200
        ax.plot([x, x + 46], [ky, ky], color=(0.3, 0.3, 0.3), lw=2.6, ls=(0, (5, 3)))
        ax.text(x + 56, ky, "BT scan segmentation", va="center", fontsize=12)
        x += 320
        ax.plot([x, x + 46], [ky, ky], color=(0.62, 0.50, 0.30), lw=1.6)
        ax.text(x + 56, ky, "model packing (no counterpart in the scan labels)", va="center", fontsize=12)
    else:
        ax.text(x, ky, "see-through fill = simulation;   thick outline = BT scan segmentation;   red dots = wall nodes "
                "inside the cervix", va="center", fontsize=12)
    ky2, x = ky + 30, mL - 60
    ax.annotate("", xy=(x + 46, ky2), xytext=(x, ky2), arrowprops=dict(arrowstyle="-|>", color=(0.3, 0.3, 0.3), lw=2.2))
    ax.text(x + 56, ky2, "from the simulation to the scan, at the largest local gap", va="center", fontsize=12)
    x += 560
    ax.plot([x, x + 30], [ky2, ky2], color=LV.COL_CANAL, lw=4)
    ax.plot([x + 36, x + 66], [ky2, ky2], color=COL_BT_CANAL, lw=4, ls=(0, (4, 2)))
    ax.text(x + 76, ky2, "intrauterine canal: simulation (carried pre-insertion canal) / scan", va="center", fontsize=12)
    # ---- table
    S, sc = stats["structures"], stats["scores_PELVIS"]
    cols = [("structure", 0), ("Dice", 250), ("mean surface\ndistance", 340), ("volume before → after\ninsertion (scans)", 490),
            ("real centroid\nmove", 720), ("simulated\nmove", 870), ("angle between\nthe moves", 1010), ("simulation centroid\nvs scan", 1160)]
    tx0, ty0 = mL - 120, top + Sh + 110
    for h_, dx in cols:
        ax.text(tx0 + dx, ty0, h_, fontsize=11, fontweight="bold", va="center", color=(0.2, 0.2, 0.2))
    for r_, n in enumerate(ORG):
        s_ = S[n]
        key = "vagina_filled" if (n == "vagina" and "vagina_filled" in sc) else n
        y = ty0 + 38 + 22 * r_
        cells = [NAMES[n] + (" (filled)" if n == "vagina" else ""), "%.2f" % sc[key]["dice"], "%.1f mm" % sc[key]["msd"],
                 "%.0f → %.0f cc" % (s_["vol_pre_label_cc"], s_["vol_bt_label_cc"]),
                 "%.1f mm" % s_["true_displacement_len"],
                 "n/a *" if n == "vagina" else "%.1f mm" % s_["model_displacement_len"],
                 "n/a *" if n == "vagina" else "%.0f°" % s_["direction_agreement_deg"], "%.1f mm" % s_["residual_len"]]
        for (h_, dx), c_ in zip(cols, cells):
            ax.text(tx0 + dx, y, c_, fontsize=11, va="center", color=line_col(st.col[n], 0.72) if dx == 0 else (0.15, 0.15, 0.15))
    ax.text(tx0, ty0 + 38 + 22 * len(ORG) + 8, "Dice and surface distance: the evaluator's pelvis-registered scores (vagina: the "
            "filled wall vs the scan's vagina + applicator).  Volumes from the segmentations; the simulation conserves "
            "each organ's volume.", fontsize=10, color=(0.35, 0.35, 0.38), va="center")
    ax.text(tx0, ty0 + 38 + 22 * len(ORG) + 28, "* the simulated vagina starts from a distended reference (%.0f cc), not the "
            "collapsed pre-insertion label (%.0f cc), so its move is not comparable.  Patient-derived figure: keep local."
            % (S["vagina"]["vol_model_rest_cc"], S["vagina"]["vol_pre_label_cc"]),
            fontsize=10, color=(0.35, 0.35, 0.38), va="center")
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--views", default="sagittal,oblique")
    a = ap.parse_args()
    AH.run_extra_parts(a.tag)
    st = LV.State(a.tag)
    st.fx = st.facts()
    prep = load_prep(a.tag)
    od = P["figs"] + "/labeled"
    os.makedirs(od, exist_ok=True)
    for view in a.views.split(","):
        Sh = a.size if view == "sagittal" else int(a.size * 1.15)
        Sw = a.size if view == "sagittal" else int(a.size * 0.93)
        img, info, ov = build(st, view, Sw, Sh, prep)
        out = "%s/%s_step%04d_overlay_%s.png" % (od, a.tag, st.step, view)
        compose(img, info, ov, view, st, prep[0], out)
        print("wrote", out, "| callouts:", {k: round(v[2], 1) for k, v in ov["callouts"].items()},
              "| canal divergence %.1f mm" % ov["canal_div"])


if __name__ == "__main__":
    main()
