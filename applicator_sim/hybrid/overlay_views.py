"""Overlay stills: the dataset's BT (post-insertion) segmentations on the two views of a run's seated state, to show
what is misaligned, where, and the likely cause; and the CANAL view: the labelled intrauterine canal carried by the
tissue against the tandem, with the upper-canal residual marked and numbered.

    python hybrid/bt_overlay.py prep --tag G32                                 # host py3.13, once per run
    APPSIM_OUT=<render tree> py -3.11 hybrid/overlay_views.py --tag G32        # -> figs/labeled/<tag>_step<k>_overlay_*.png
    APPSIM_OUT=<render tree> py -3.11 hybrid/overlay_views.py --tag TF0 --views canal --steps end:C,final
                                                                               # -> figs/labeled/<tag>_step<k>_canal.png

Sagittal: on the device's sagittal plane, the simulation's section outlines (solid) and the scan's (dashed), per
organ in the organ's colour; the device in black and the intrauterine canal in cyan, the same way.  Oblique: the
simulation drawn see-through as in the labelled view, the scan's structures as outlines.  Each callout sits where
that structure's scan contour or surface is furthest from the simulation, with an arrow from the simulation to
the scan, the measured offset and the likely cause.  Every number comes from figs/overlay_prep/<tag>_PELVIS/
stats.json (bt_overlay.py) or is computed here from the run's files.  The BT overlays compare the SEATED state with
the post-insertion scan, so they are drawn for the last frame only.

Canal view (fix plan S6c, "a canal overlay"): two orthographic views in the device's own frame -- coronal (from
anterior, the patient's right on the image's left, the tube vertical) and sagittal (from the patient's left,
anterior on the left) -- with the uterus, cervix and vagina see-through, the device opaque, the physician's labelled
canal (tandem_path.npz, s >= 0) carried by the tissue as label_views.State carries it (cyan for 0..20 mm above the
external os, the part the tandem must follow; orange above, the part it does not follow), the same canal before
insertion dotted, and numbered points every 5 mm of the upper canal joined to their nearest tube point.  A third
panel plots every canal point's distance from the tube segment (flange..tip) against its arclength, with the
r_tube + 1 mm in-tube band, the 20 mm limit and the tandem depth.  The numbers are fix plan S0's definitions
(lower_canal_in_tube_frac, traversed_canal_in_tube_frac, upper_canal_residual), measured on this carry -- exact
where the frame saved its cervix nodal displacement (u_npy; label_views.State: TF0c step 192 lower 0.62), named on
the figure; when S0's eval/<tag>/tf_metrics.json has the step, its values are printed beside them with the carry
tf_metrics recorded for that frame (a tf_metrics 1.0 file: the harmonic extension).  --steps as in
label_views.py.  --canal legacy (canal.npz) applies to the BT overlays only and reproduces the earlier figures.

Device: the run's own parts; a tandem-only run (applicator_v4, TF0) has no ring, and the ring / packing callouts
and keys are left out.  A ring_phases run (S7f, device json ring_halves; label_views.State draws each half at its own
logged pose) is captioned for what it is (findings / ring_extras): the wall variant (CAL = rest shape calibrated on
this BT, in-sample; PRED = device hull + margins, predictive), the vagina Dice against the updated BT reference
(eval/<tag>/tf_metrics.json) and, with --trim-json (a trimref.py JSON), against the trimmed reference, the halves'
state and the scan's ring label to the model's halves, the in-sample pose rule, no packing, and the gap of each callout
measured to the simulated SURFACE in 3-D (gap_anchor_surf).  Other runs are captioned as before.
Patient-derived: local only."""
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
CANAL_MARK_S = (25.0, 30.0, 35.0, 40.0, 45.0)      # canal view: numbered points of the upper canal (canal_s, mm)


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


def fmt(v, f="%.2f"):
    """f % v, or 'n/a' for a number the prep does not have (a run eval_hybrid has not scored: no scores_PELVIS)."""
    try:
        return f % v if v is not None and np.isfinite(float(v)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def findings(stats, canal_div, cfg=None, ring=None):
    """Callout text per structure: the measured misalignment, then the likely cause with its evidence.  The cervix
    cause depends on what the run's canal ties followed (cfg canal_tie_set): G16-G32 tied canal.npz, whose lower
    part is extrapolated along a0; a canal_path run (TF0 on) ties the physician's path.  The ring line is left out
    for a run without a ring.  A ring_phases run (ring = ring_extras(...)) is worded for what it is: the wall variant
    (CAL / PRED) and the reference of the vagina Dice, the ring halves (bt_overlay's ring_offset_mm reads cap centres,
    which a ring run has none of), no packing (none was used), the in-sample pose rule, and the rectum's moves without
    a cause the figure cannot measure."""
    cfg = cfg or {}
    S, M, sc, pe = stats["structures"], stats["measurements"], stats.get("scores_PELVIS") or {}, \
        stats.get("pose_rule_error") or {}
    dice = lambda n: fmt((sc.get(n) or {}).get("dice"))                                    # noqa: E731
    res = {n: S[n]["residual_final_minus_bt_mm"] for n in S}
    w = M["vagina_width_lr_ap_mm"].get("+0", {})
    hs = M["hrctv_lr_split_cc"]
    ro = float(np.linalg.norm(M["ring_offset_mm"])) if M.get("ring_offset_mm") is not None else None
    ref = "physician's path" if hs.get("reference") == "tandem_path" else "canal line"
    ut = "elastic" if cfg.get("corpus_model", "rigid") == "fem" else "rigid"     # S9: cfg corpus_model
    if cfg.get("canal_tie_set", "canal") == "canal_path":
        lift = "lifted %.0f mm up (scan %.0f); the tandem's ties follow" % (S["cervix"]["model_displacement_mm"][2],
                                                                            S["cervix"]["true_displacement_mm"][2])
        why = ["the physician's canal path; HR-CTV halves",
               "(pre: of the %s; BT: of the real tandem)" % ref]
    else:
        lift = "lifted correctly (%.0f mm up, scan %.0f), but the tandem" % (S["cervix"]["model_displacement_mm"][2],
                                                                           S["cervix"]["true_displacement_mm"][2])
        why = ["passes on the wrong side of the tumour: the model's",
               "lower %.0f mm of canal is extrapolated; scan halves" % M["canal_label_starts_above_os_mm"]]
    f = dict(
        corpus=["Dice %s; model within %.0f mm of the scan" % (dice("corpus"), S["corpus"]["residual_len"]),
                "aligned: the pose rule places the %s uterus" % ut],
        cervix=["Dice %s; model %s of the scan" % (dice("cervix"), dirw(res["cervix"])), lift] + why +
               ["right %.0f→%.0f cc, left %.0f→%.0f cc (not just shrinkage)" % (hs["pre_right"], hs["bt_right"],
                                                                                 hs["pre_left"], hs["bt_left"])],
        vagina=["Dice %s (filled); model %s of the scan" % (dice("vagina_filled" if "vagina_filled" in sc else "vagina"),
                                                           dirw(res["vagina"])),
                "vault %.0f mm wide (L-R) vs %.0f in the scan: fornix flare;" % (w.get("model", [np.nan])[0],
                                                                                w.get("BT", [np.nan])[0]),
                "real distension is posterior (packing behind the ring)"],
        bladder=["Dice %s; model %s of the scan" % (dice("bladder"), dirw(res["bladder"])),
                 "%+.0f cc filling change between the scans (not modelled)" % (S["bladder"]["vol_bt_label_cc"]
                                                                              - S["bladder"]["vol_pre_label_cc"])],
        rectum=["Dice %s; model %s of the scan" % (dice("rectum"), dirw(res["rectum"])),
                "real move: %s; model: %s" % (dirw(S["rectum"]["true_displacement_mm"], 2.0),
                                             dirw(S["rectum"]["model_displacement_mm"], 2.0)),
                "no posterior packing or rectovaginal coupling in the model"],
        sigmoid=["Dice %s; model %s of the scan" % (dice("sigmoid"), dirw(res["sigmoid"])),
                 "bowel moved %.0f mm, volume %+.0f %%: not applicator-driven" % (
                     S["sigmoid"]["true_displacement_len"],
                     100.0 * (S["sigmoid"]["vol_bt_label_cc"] / S["sigmoid"]["vol_pre_label_cc"] - 1.0))],
        device=(["tandem %s° / %s mm from the real one; ring %.1f mm" % (
                    fmt(pe.get("axis_angle_deg"), "%.1f"), fmt((pe.get("flange_offset_mm") or {}).get("total"), "%.1f"),
                    ro),
                 "from the scan's ovoids: placement is not the problem"] if ro is not None else
                ["tandem %s° / %s mm from the real one;" % (
                    fmt(pe.get("axis_angle_deg"), "%.1f"), fmt((pe.get("flange_offset_mm") or {}).get("total"), "%.1f")),
                 "no ring in this run (tandem only)"]),
        canal=["scan canal within %.1f mm of the real tandem;" % M["bt_canal_vs_real_tandem_max_mm"],
               "model's %s uterus keeps its curve: %.0f mm off" % (ut, canal_div)])
    if ring is not None:                 # S7f ring run (ring_extras): no packing was used; the pose rule is in-sample
        f["corpus"] = [f["corpus"][0], "placed by the tandem pose rule (%s uterus);" % ut,
                       "the rule was tuned on this BT: in-sample"]
        tr = ring.get("trim")
        # (lines <= ~60 characters: a callout's text column is 580 px wide on either side)
        f["vagina"] = ([ring["variant"]] if ring.get("variant") else []) + [
            "Dice %s (filled) %s" % (fmt(ring.get("dice"), "%.3f"), ring["ref_short"])] + (
            ["trimmed reference, %s cc: Dice %s%s" % (
                fmt(tr["cc"], "%.0f"), fmt(tr["dice"], "%.3f"),
                "" if tr["spec"].find("@") < 0 else " (M121, step %s)" % tr["spec"].split("@")[1]),
             "(BT label kept ≤ %s mm behind the device)" % fmt(tr["t_mm"], "%.1f")] if tr else
            ["(the BT contour also takes in tissue behind the vagina)"]) + [
            "model %s of the scan;" % dirw(res["vagina"]),
            "vault %.0f mm wide (L-R) vs %.0f in the scan" % (w.get("model", [np.nan])[0], w.get("BT", [np.nan])[0])]
        f["device"] = ["tandem %s° / %s mm from the real one" % (
                           fmt(pe.get("axis_angle_deg"), "%.1f"), fmt((pe.get("flange_offset_mm") or {}).get("total"), "%.1f")),
                       "(tandem pose rule tuned on this BT: in-sample)",
                       "ring halves: %s" % ring["status"]] + \
            (["scan's ring label to the model's halves: median %.1f mm," % ring["ring_mm"][0],
              "95th percentile %.1f mm" % ring["ring_mm"][1]] if ring.get("ring_mm") else [])
        f["rectum"] = [f["rectum"][0], "real move: %s;" % dirw(S["rectum"]["true_displacement_mm"], 2.0),
                       "model move: %s" % dirw(S["rectum"]["model_displacement_mm"], 2.0)]
    return f


def gap_anchor(A, B):
    """The point of A (scan) at the 95th percentile of its distance to B (simulation), and B's nearest point."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = np.array([np.linalg.norm(B - a, axis=1).min() for a in A])
    i = int(np.argsort(d)[int(0.95 * (len(d) - 1))])
    j = int(np.argmin(np.linalg.norm(B - A[i], axis=1)))
    return A[i], B[j], float(d[i])


def gap_anchor_surf(A, V, F):
    """As gap_anchor, but the distance is to the simulated SURFACE (V, F) in 3-D, not to the simulation's outline in
    the section plane: an organ that moved out of the plane (TF3c's rectum, 13 mm to the right) has no outline near the
    scan's, and the in-plane distance (72 mm there) is not its gap.  Returns the scan point, the surface's nearest
    vertex to it and the exact point-to-surface distance."""
    A, V = np.asarray(A, float), np.asarray(V, float)
    d = np.abs(LV.implicit_distance(V, F)(A))
    i = int(np.argsort(d)[int(0.95 * (len(d) - 1))])
    j = int(np.argmin(np.linalg.norm(V - A[i], axis=1)))
    return A[i], V[j], float(d[i])


def trim_entry(fn, tag):
    """The run's trimmed-reference vagina Dice from a trimref.py JSON (keys t_mm, trim_cc, and per run spec 'TAG' or
    'TAG@k' -> dice_trim): the final state (key TAG) if scored, else its latest M121 average (TAG@k).  None when the
    file does not score this run."""
    J = AH.load_json(fn)
    keys = [k for k in J if isinstance(J[k], dict) and k.split("@")[0] == tag]
    if not keys:
        print("[overlay] %s does not score %s: no trimmed-reference Dice on the figure" % (fn, tag))
        return None
    key = tag if tag in keys else sorted(keys, key=lambda k: int(k.split("@")[1]))[-1]
    return dict(spec=key, dice=J[key].get("dice_trim"), t_mm=J.get("t_mm"), cc=J.get("trim_cc"))


def ring_extras(st, prep, trim_json=None):
    """What the captions of a ring_phases run (S7f, st.ring_halves) say, computed from the run's files:
    wall variant (meta wall.v5.section: 'cal' = rest shape from this BT, in-sample; 'pred' = device hull + margins),
    the vagina Dice against the UPDATED BT reference (eval/<tag>/tf_metrics.json final.vagina.filled_dice_BT_grid:
    vagina | updated applicator | ovoid; else the evaluator's scores_PELVIS, older applicator label, named so), the
    trimmed reference (--trim-json, trimref.py) if given, each half's state, and the scan's ring label (prep
    ovoid.obj) to the model's ring halves: point-to-surface distance of its vertices, median / 95th percentile."""
    stats, bt, _ = prep
    v5 = ((st.meta["vagina"].get("wall") or {}).get("v5") or {})
    sec = v5.get("section")
    r = dict(section=sec, variant={"cal": "CAL: seated shape from this BT (calibrated, in-sample)",
                                   "pred": "PRED: seated shape = device hull + margins (predictive)"}.get(sec),
             short={"cal": "CAL", "pred": "PRED"}.get(sec))
    fn = "%s/eval/%s/tf_metrics.json" % (P["hybrid"], st.tag)
    fd = ((AH.load_json(fn).get("final") or {}).get("vagina") or {}).get("filled_dice_BT_grid") or {} \
        if os.path.exists(fn) else {}
    if fd.get("updated_reference") is not None:
        cc = fmt(fd.get("updated_reference_cc"), "%.0f")
        r.update(dice=fd["updated_reference"], ref="vs BT vagina + device (updated label, %s cc)" % cc,
                 ref_short="vs BT vagina + device, %s cc" % cc)
    else:
        sc = (stats.get("scores_PELVIS") or {}).get("vagina_filled") or {}
        r.update(dice=sc.get("dice"), ref="vs BT vagina + applicator (older label)",
                 ref_short="vs BT vagina + applicator (older label)")
    r["trim"] = trim_entry(trim_json, st.tag) if trim_json else None
    r["status"] = LV.ring_status(st)
    halves = [st.parts[p] for p in sorted(st.ring_halves) if p in st.parts]
    r["ring_mm"] = None
    if halves:
        V = np.vstack([h[0] for h in halves])
        F = np.vstack([h[1] + sum(len(g[0]) for g in halves[:i]) for i, h in enumerate(halves)])
        d = np.abs(LV.implicit_distance(V, F)(bt["ovoid"][0]))
        r["ring_mm"] = (float(np.median(d)), float(np.percentile(d, 95)))
    return r


def canal_divergence(st, cl):
    """Largest distance of the model's carried canal (cavity part) from the scan's canal centreline, with the pair."""
    C = st.canal[st.i_ios:]
    d = np.array([np.linalg.norm(cl - c, axis=1).min() for c in C])
    k = int(np.argmax(d))
    return float(d[k]), C[k], cl[int(np.argmin(np.linalg.norm(cl - C[k], axis=1)))]


def canal_split(st, pts2d):
    """The projected canal as (points, colour) pieces: one cyan piece (legacy), or the followed lower canal in cyan
    and the upper canal (canal_s > 20 mm) in orange, sharing the joint point."""
    if st.canal_s is None:
        return [(pts2d, LV.COL_CANAL)]
    k = int(np.nonzero(st.canal_s <= LV.LOWER_CANAL_MM)[0].max())
    return [(pts2d[:k + 1], LV.COL_CANAL), (pts2d[k:], LV.COL_RESID)]


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
                if Lm and Lb:                   # ring runs: the gap to the simulated SURFACE (gap_anchor_surf)
                    pb, pm, d = gap_anchor_surf(np.vstack(Lb), *st.surf[n]) if st.ring_halves else \
                        gap_anchor(np.vstack(Lb), np.vstack(Lm))
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
            ocs = AH.ovoid_centres(st.dev)                      # no caps (tandem only): the ring plane at the flange
            oc = ocs.mean(0) if len(ocs) else st.F
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
                pb, pm, d = gap_anchor_surf(sub, *st.surf[n]) if st.ring_halves else \
                    gap_anchor(sub, Vm[:: max(1, len(Vm) // 4000)])
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
def compose(img, info, ov, view, st, stats, out, ring=None):
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
        for piece, col in canal_split(st, mc):
            ax.plot(piece[:, 0], piece[:, 1], color=col, lw=2.6, zorder=4, path_effects=halo)
        ax.plot(bc[:, 0], bc[:, 1], color=COL_BT_CANAL, lw=2.6, ls=(0, (5, 3)), zorder=4, path_effects=halo)
    F = findings(stats, ov["canal_div"], st.cfg, ring)
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
    run = st.tag + ((", %s vaginal wall" % ring["short"]) if ring and ring.get("short") else "")
    sub = dict(sagittal="Solid outlines: the simulation (run %s, seated).  Dashed outlines: the dataset's post-insertion (BT) "
                        "segmentation, registered by the pelvis.  Anterior left, superior up." % run,
               oblique="See-through surfaces: the simulation (run %s, seated).  Outlines: the dataset's post-insertion (BT) "
                       "segmentation, registered by the pelvis." % run)[view]
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
        if "packing" in st.parts:
            ax.plot([x, x + 46], [ky, ky], color=(0.62, 0.50, 0.30), lw=1.6)
            ax.text(x + 56, ky, "model packing (no counterpart in the scan labels)", va="center", fontsize=12)
    else:
        ax.text(x, ky, "see-through fill = simulation;   thick outline = BT scan segmentation;   red dots = wall nodes "
                "inside the cervix", va="center", fontsize=12)
    ky2, x = ky + 30, mL - 60
    ax.annotate("", xy=(x + 46, ky2), xytext=(x, ky2), arrowprops=dict(arrowstyle="-|>", color=(0.3, 0.3, 0.3), lw=2.2))
    ax.text(x + 56, ky2, "from the simulation to the scan, at the largest local gap" +
            (" (3-D, to the simulated surface)" if ring else ""), va="center", fontsize=12)
    x += 560 + (250 if ring else 0)
    if st.canal_s is None:
        ax.plot([x, x + 30], [ky2, ky2], color=LV.COL_CANAL, lw=4)
        ax.plot([x + 36, x + 66], [ky2, ky2], color=COL_BT_CANAL, lw=4, ls=(0, (4, 2)))
        ax.text(x + 76, ky2, "intrauterine canal: simulation (carried pre-insertion canal) / scan", va="center",
                fontsize=12)
    else:
        ax.plot([x, x + 15], [ky2, ky2], color=LV.COL_CANAL, lw=4)
        ax.plot([x + 15, x + 30], [ky2, ky2], color=LV.COL_RESID, lw=4)
        ax.plot([x + 36, x + 66], [ky2, ky2], color=COL_BT_CANAL, lw=4, ls=(0, (4, 2)))
        ax.text(x + 76, ky2, "intrauterine canal: simulation (carried labelled canal; orange > %.0f mm above the os) "
                "/ scan" % LV.LOWER_CANAL_MM, va="center", fontsize=12)
    # ---- table
    S, sc = stats["structures"], stats.get("scores_PELVIS") or {}
    cols = [("structure", 0), ("Dice", 250), ("mean surface\ndistance", 340), ("volume before → after\ninsertion (scans)", 490),
            ("real centroid\nmove", 720), ("simulated\nmove", 870), ("angle between\nthe moves", 1010), ("simulation centroid\nvs scan", 1160)]
    tx0, ty0 = mL - 120, top + Sh + 110
    for h_, dx in cols:
        ax.text(tx0 + dx, ty0, h_, fontsize=11, fontweight="bold", va="center", color=(0.2, 0.2, 0.2))
    for r_, n in enumerate(ORG):
        s_ = S[n]
        key = "vagina_filled" if (n == "vagina" and "vagina_filled" in sc) else n
        y = ty0 + 38 + 22 * r_
        vd = (sc.get(key) or {}).get("dice")
        if n == "vagina" and ring:           # ring run: the updated-reference Dice of the callout, the variant named
            vd = ring.get("dice")
        cells = [NAMES[n] + ((" (filled%s)" % ((", " + ring["short"]) if ring and ring.get("short") else ""))
                             if n == "vagina" else ""), fmt(vd),
                 fmt((sc.get(key) or {}).get("msd"), "%.1f mm"),
                 "%.0f → %.0f cc" % (s_["vol_pre_label_cc"], s_["vol_bt_label_cc"]),
                 "%.1f mm" % s_["true_displacement_len"],
                 "n/a *" if n == "vagina" else "%.1f mm" % s_["model_displacement_len"],
                 "n/a *" if n == "vagina" else "%.0f°" % s_["direction_agreement_deg"], "%.1f mm" % s_["residual_len"]]
        for (h_, dx), c_ in zip(cols, cells):
            ax.text(tx0 + dx, y, c_, fontsize=11, va="center", color=line_col(st.col[n], 0.72) if dx == 0 else (0.15, 0.15, 0.15))
    if ring:
        n1 = ("Dice and surface distance: the evaluator's pelvis-registered scores; vagina Dice: the filled wall %s%s.  "
              "Volumes from the segmentations; the simulation conserves each organ's volume."
              % (ring["ref"], ", its surface distance vs the older label" if "updated" in ring["ref"] else ""))
    else:
        n1 = "Dice and surface distance: the evaluator's pelvis-registered scores (vagina: the filled wall vs the scan's " \
             "vagina + applicator).  Volumes from the segmentations; the simulation conserves each organ's volume."
    ax.text(tx0, ty0 + 38 + 22 * len(ORG) + 8, n1, fontsize=10, color=(0.35, 0.35, 0.38), va="center")
    if ring and st.cfg.get("wall_drive") == "device":     # S7b: the driven v5 wall starts collapsed, ends prescribed
        n2 = ("* the driven wall starts collapsed (built from the pre-insertion label, %.0f cc) and ends on a prescribed "
              "seated shape (%s), so its centroid move is not a simulation result.  Patient-derived figure: keep local."
              % (S["vagina"]["vol_pre_label_cc"], {"cal": "CAL: from this BT, in-sample",
                                                   "pred": "PRED: device hull + margins"}.get(ring.get("section"),
                                                                                             "the wall's rest shape")))
    else:
        n2 = ("* the simulated vagina starts from a distended reference (%.0f cc), not the collapsed pre-insertion label "
              "(%.0f cc), so its move is not comparable.  Patient-derived figure: keep local."
              % (S["vagina"]["vol_model_rest_cc"], S["vagina"]["vol_pre_label_cc"]))
    ax.text(tx0, ty0 + 38 + 22 * len(ORG) + 28, n2, fontsize=10, color=(0.35, 0.35, 0.38), va="center")
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ canal view
def tf_carry_words(fm):
    """The carry tf_metrics used for one frame: its per-frame "carry" record (tf_metrics >= 1.1: exact u_npy or the
    harmonic extension), else "harmonic carry" (a 1.0 file has no record and always used the harmonic extension)."""
    c = fm.get("carry", (fm.get("insertion") or {}).get("carry"))
    if isinstance(c, dict):
        c = c.get("source") or c.get("cervix") or c.get("kind") or ", ".join(
            "%s %s" % (k, v) for k, v in c.items() if isinstance(v, (str, int, float)))
    return ("carry: %s" % c) if c else "harmonic carry"


def canal_panel(st, kind, Sw, Sh, box):
    """One orthographic render in the device frame: kind 'coronal' (camera anterior, looking along -y_app: the
    patient's right on the image's left) or 'sagittal' (camera on the patient's left, looking along +x_app: anterior
    on the left), up = the tube axis.  Uterus / cervix / vagina see-through, device opaque.  Returns (img, proj,
    px_per_mm)."""
    pv = AH._pv()
    xa, ya, za = st.R_t
    view_dir = -ya if kind == "coronal" else xa                 # direction of projection
    right = geom.unit(np.cross(view_dir, za))
    c = 0.5 * (box.min(0) + box.max(0))
    q = box - c
    half_h, half_w = 0.5 * np.ptp(q @ za), 0.5 * np.ptp(q @ right)
    scale = 1.08 * max(half_h, half_w * Sh / Sw)
    pl = pv.Plotter(off_screen=True, window_size=[Sw, Sh])
    pl.set_background("white")
    try:
        pl.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)
    except Exception:
        pass
    for b, op in (("vagina", 0.10), ("cervix", 0.28), ("corpus", 0.20)):
        pl.add_mesh(AH.poly(pv, *st.surf[b]), color=st.col[b], opacity=op, smooth_shading=True, specular=0.1)
    for p, (V, Fc) in st.parts.items():
        opq = p in ("tube", "shaft")
        pl.add_mesh(AH.poly(pv, V, Fc), color=AH.COL_TANDEM if opq else AH.COL_OVOID, opacity=1.0 if opq else 0.25,
                    smooth_shading=True)
    pl.camera_position = [tuple(c - 600.0 * view_dir), tuple(c), tuple(za)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = scale
    pl.renderer.reset_camera_clipping_range()
    pl.show(auto_close=False)
    pl.render()
    img = pl.screenshot(return_img=True)
    ren = pl.renderer
    ww, wh = pl.ren_win.GetSize()

    def proj(p):
        ren.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        return np.array([x * img.shape[1] / ww, img.shape[0] - y * img.shape[0] / wh])
    # A parallel projection is affine, so three projected points define it exactly; capture it BEFORE closing (a
    # closed plotter's renderer no longer holds the camera: every point then lands on the same pixel).
    o2, u2, r2 = proj(c), proj(c + za), proj(c + right)
    pl.close()

    def proj_affine(p):
        q = np.asarray(p, float) - c
        return o2 + (q @ za) * (u2 - o2) + (q @ right) * (r2 - o2)
    # the world direction that points to the image's right (screen x grows with it when rx > 0)
    rx = float((r2 - o2)[0])
    return img, proj_affine, float(np.linalg.norm(u2 - o2)), (right if rx > 0 else -right)


def canal_view(st, out, size=1040):
    """The canal overlay figure (module docstring).  Returns the numbers printed on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    if st.canal_s is None:
        raise SystemExit("the canal view needs the physician's path (--canal path)")
    cs, C, C0 = st.canal_s, st.canal, st.canal_rest
    m = st.canal_metrics()
    dseg, h = m["dseg"], m["h"]
    L = float(st.dev["L_iu_mm"])
    near = st.F + np.outer(np.clip(h, 0.0, L), st.a)          # nearest tube-axis point of each canal point
    marks = [float(s) for s in CANAL_MARK_S if s <= cs[-1]]
    if m.get("upper_residual_s") is not None and min(abs(m["upper_residual_s"] - s) for s in marks) > 2.0:
        marks.append(m["upper_residual_s"])
    mi = [int(np.argmin(np.abs(cs - s))) for s in marks]
    # ---- the framing box: the canal (now and before insertion), the tube, the tissue around the canal
    Vt = np.vstack([st.surf["corpus"][0], st.surf["cervix"][0]])
    hh = (Vt - st.F) @ st.a
    Vt = Vt[(hh > -25.0) & (hh < L + 25.0)]
    box = np.vstack([C, C0, Vt, st.F[None], st.tip[None], (st.F - 20.0 * st.a)[None]])
    Sw, Sh = int(size * 0.72), size
    panels = {k: canal_panel(st, k, Sw, Sh, box) for k in ("coronal", "sagittal")}
    # ---- figure (pixel canvas)
    W, H = 30 + 2 * (Sw + 20) + 820, 190 + Sh + 150
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    halo = [pe.withStroke(linewidth=4.0, foreground="white")]
    top = 190
    for j, kind in enumerate(("coronal", "sagittal")):
        img, proj, ppm, rvec = panels[kind]
        x0 = 30 + j * (Sw + 20)
        ax.imshow(img, extent=(x0, x0 + Sw, top + Sh, top), aspect="auto", zorder=0)
        ax.add_patch(plt.Rectangle((x0, top), Sw, Sh, fill=False, ec=(0.6, 0.6, 0.6), lw=1.0, zorder=1))
        off = np.array([x0, top])
        pr = lambda X: np.array([proj(q) for q in np.atleast_2d(X)]) + off    # noqa: E731
        R0 = pr(C0)
        ax.plot(R0[:, 0], R0[:, 1], color=(0.35, 0.35, 0.38), lw=1.6, ls=(0, (1.5, 2.5)), zorder=3)
        for piece, col in canal_split(st, pr(C)):
            ax.plot(piece[:, 0], piece[:, 1], color=col, lw=3.2, zorder=4, path_effects=halo, solid_capstyle="round")
        for n_, i in enumerate(mi):
            a2, b2 = pr(C[i])[0], pr(near[i])[0]
            ax.plot([a2[0], b2[0]], [a2[1], b2[1]], color=LV.COL_RESID, lw=1.3, ls=(0, (3, 2)), zorder=5)
            ax.plot([a2[0]], [a2[1]], marker="o", ms=17, mfc=LV.COL_RESID, mec="white", mew=1.5, zorder=6)
            ax.text(a2[0], a2[1], "%d" % (n_ + 1), ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white", zorder=7)
        for q, nm in ((C[0], "external os (O_true)"), (C[st.i_ios], "internal os")):
            p2 = pr(q)[0]
            ax.plot([p2[0]], [p2[1]], marker="D", ms=9, mfc="black", mec="white", mew=1.4, zorder=8)
            ax.text(p2[0] + 12, p2[1] + 4, nm, fontsize=10.5, va="center", zorder=8, path_effects=halo)
        k20 = int(np.argmin(np.abs(cs - LV.LOWER_CANAL_MM)))
        p2 = pr(C[k20])[0]
        ax.plot([p2[0]], [p2[1]], marker="o", ms=7, mfc="white", mec=LV.COL_RESID, mew=2.0, zorder=8)
        ax.text(p2[0] - 12, p2[1], "s = %.0f mm" % LV.LOWER_CANAL_MM, fontsize=10, ha="right", va="center", zorder=8,
                path_effects=halo)
        tp = pr(st.tip)[0]
        ax.text(tp[0], tp[1] - 14, "tip", fontsize=10.5, ha="center", va="bottom", zorder=8, path_effects=halo)
        # orientation words (from the world direction that points to the image's right) and a 10 mm bar
        if kind == "coronal":
            lr = ("patient left", "patient right") if rvec[0] > 0 else ("patient right", "patient left")
        else:
            lr = ("posterior", "anterior") if rvec[1] > 0 else ("anterior", "posterior")
        ax.text(x0 + 10, top + 16, "◀ " + lr[0], fontsize=11, va="center", color=(0.3, 0.3, 0.3))
        ax.text(x0 + Sw - 10, top + 16, lr[1] + " ▶", fontsize=11, ha="right", va="center", color=(0.3, 0.3, 0.3))
        ax.text(x0 + Sw / 2, top + 16, "▲ along the tube", fontsize=11, ha="center", va="center", color=(0.3, 0.3, 0.3))
        ax.plot([x0 + Sw - 20 - 10 * ppm, x0 + Sw - 20], [top + Sh - 22] * 2, color="black", lw=3)
        ax.text(x0 + Sw - 20 - 5 * ppm, top + Sh - 36, "10 mm", ha="center", fontsize=10.5)
        ax.text(x0 + Sw / 2, top + Sh + 22, dict(coronal="coronal, in the device frame (viewed from anterior)",
                                                  sagittal="sagittal, in the device frame (viewed from the patient's "
                                                           "left)")[kind],
                ha="center", va="center", fontsize=12, fontweight="bold", color=(0.25, 0.25, 0.28))
    # ---- distance plot and table
    xp = 30 + 2 * (Sw + 20) + 60
    pw, ph = 740, int(0.52 * Sh)
    axp = fig.add_axes([xp / W, 1.0 - (top + ph) / H, pw / W, ph / H])
    tol = st.r_tube + LV.TUBE_TOL_MM
    axp.axhspan(0.0, tol, color=(0.85, 0.93, 0.97), zorder=0)
    axp.text(0.5, tol + 0.3, "in the tube: ≤ r_tube + %.0f mm = %.1f mm" % (LV.TUBE_TOL_MM, tol), fontsize=9,
             color=(0.2, 0.4, 0.5), va="bottom")
    lo = cs <= LV.LOWER_CANAL_MM
    k = int(np.nonzero(lo)[0].max())
    axp.plot(cs[:k + 1], dseg[:k + 1], color=LV.COL_CANAL, lw=2.4, zorder=3, label="lower canal (followed)")
    axp.plot(cs[k:], dseg[k:], color=LV.COL_RESID, lw=2.4, zorder=3, label="upper canal (residual)")
    ymax = max(tol + 2.0, 1.12 * float(dseg.max()))
    axp.set_xlim(min(-1.0, cs[0] - 1.0), cs[-1] + 2.0)
    axp.set_ylim(0.0, ymax)
    axp.axvline(LV.LOWER_CANAL_MM, color=LV.COL_RESID, lw=1.0, ls="--")
    axp.axvline(cs[st.i_ios], color="0.3", lw=0.9, ls=":")
    axp.text(cs[st.i_ios] - 0.3, 0.5 * ymax, "internal os", fontsize=9, va="center", ha="right", rotation=90,
             color="0.3")
    x_lo = min(-1.0, cs[0] - 1.0)
    if st.depth is not None and x_lo <= st.depth <= cs[-1] + 2.0:
        axp.axvline(st.depth, color="black", lw=1.2)
        axp.text(st.depth - 0.3, 0.5 * ymax, "tip depth d = %.1f mm" % st.depth, fontsize=9, va="center",
                 ha="right", rotation=90)
    elif st.depth is not None:                                  # the tip is below the os or above the canal's end
        axp.text(0.99 if st.depth > 0 else 0.01, 0.5, "tip depth d = %.1f mm\n(%s)" % (
            st.depth, "past the canal's end" if st.depth > 0 else "below the external os"), transform=axp.transAxes,
                 fontsize=9, va="center", ha="right" if st.depth > 0 else "left")
    for n_, i in enumerate(mi):
        axp.plot([cs[i]], [dseg[i]], marker="o", ms=15, mfc=LV.COL_RESID, mec="white", mew=1.3, zorder=5)
        axp.text(cs[i], dseg[i], "%d" % (n_ + 1), ha="center", va="center", fontsize=9, fontweight="bold",
                 color="white", zorder=6)
    axp.set_xlabel("canal arclength s above the external os (mm)", fontsize=10.5)
    axp.set_ylabel("distance from the tube segment (mm)", fontsize=10.5)
    axp.grid(alpha=0.3)
    axp.tick_params(labelsize=9)
    axp.legend(loc="upper left", fontsize=9, framealpha=0.9)
    axp.set_title("every labelled-canal point against the tube", fontsize=11.5, fontweight="bold")
    ty = top + ph + 70
    ax.text(xp, ty, "numbered points (upper canal)", fontsize=11.5, fontweight="bold")
    ax.text(xp, ty + 26, "  #      s (mm)    off the tube    direction from the tube (patient)", fontsize=10.5,
            family="monospace", color=(0.2, 0.2, 0.2))
    for n_, i in enumerate(mi):
        ax.text(xp, ty + 50 + 22 * n_, "%3d    %6.1f    %6.1f mm      %s" % (n_ + 1, cs[i], dseg[i],
                                                                           dirw(C[i] - near[i], 1.0)),
                fontsize=10.5, family="monospace", color=(0.15, 0.15, 0.15))
    by = {b: st.canal_by.count(b) for b in ("cervix", "corpus", "cervix~", "corpus~")}
    kc = "the rigid corpus"                    # S9: an elastic corpus is carried barycentrically when its nodes are exact
    if getattr(st, "corpus_fem", False):
        kc = "the elastic corpus (%s)" % ("barycentric" if st.X_src["corpus"].startswith("exact") else "Kabsch, approx.")
    ax.text(xp, ty + 60 + 22 * len(mi), "carried: %d points in cervix tets (barycentric), %d with %s, "
            "%d outside both (nearest-node)" % (by["cervix"], by["corpus"], kc, by["cervix~"] + by["corpus~"]),
            fontsize=9.5,
            color=(0.35, 0.35, 0.38))
    ax.text(xp, ty + 80 + 22 * len(mi), "cervix nodal displacement: %s" % st.X_src["cervix"], fontsize=9.5,
            color=(0.35, 0.35, 0.38))
    # ---- title and summary
    ax.text(W / 2, 36, "Labelled intrauterine canal vs the tandem", ha="center", va="center", fontsize=21,
            fontweight="bold")
    ax.text(W / 2, 72, "run %s, step %d (%s).  The physician's canal (tandem_path.npz, from the external os to the "
            "fundal end) moved with the tissue." % (st.tag, st.step, LV.state_words(st)), ha="center", va="center",
            fontsize=12.5, color=(0.25, 0.25, 0.28))
    fr = lambda v, f_="%.2f": "n/a" if v is None else f_ % v      # noqa: E731
    s1 = ("lower canal (0 ≤ s ≤ min(d, %.0f) mm, %d points) within %.1f mm of the tube: %s  (max %.1f mm over s 0-%.0f);"
          "   traversed (in the tube's span): %s;   upper-canal residual (s > %.0f mm): %s mm at s %s mm"
          % (LV.LOWER_CANAL_MM, m["n_lower"], tol, fr(m["lower_in_tube_frac"]), m["lower_max_mm"], LV.LOWER_CANAL_MM,
             fr(m["traversed_in_tube_frac"]), LV.LOWER_CANAL_MM, fr(m["upper_residual_mm"], "%.1f"),
             fr(m["upper_residual_s"], "%.1f")))
    ax.text(W / 2, 108, s1, ha="center", va="center", fontsize=11.5)
    fm = AH.tf_metrics(st.tag).get(st.step, {})
    tm = fm.get("insertion")
    s2 = ("S0 tf_metrics for this step (%s): lower %s, traversed %s, upper residual %s mm, tip depth %s mm"
          % (tf_carry_words(fm), fr(tm.get("lower_canal_in_tube_frac")), fr(tm.get("traversed_canal_in_tube_frac")),
             fr(tm.get("upper_canal_residual_mm"), "%.1f"), fr(tm.get("d_used_mm"), "%.1f"))) if tm else \
        "S0 tf_metrics: this step not scored (python -P hybrid/tf_metrics.py score --tag %s)" % st.tag
    ax.text(W / 2, 136, s2 + (";   tip depth here: %s" % ("%.1f mm, %s" % (st.depth, st.depth_src)
                                                          if st.depth is not None else "n/a")),
            ha="center", va="center", fontsize=10.5, color=(0.35, 0.35, 0.38))
    ky = top + Sh + 62
    for x, col, ls, txt in ((40, LV.COL_CANAL, "-", "labelled canal 0-%.0f mm above the os (to be followed)"
                             % LV.LOWER_CANAL_MM),
                            (40 + 0.33 * W, LV.COL_RESID, "-", "above %.0f mm: upper canal, not followed (numbered; "
                             "dashed = to the nearest tube point)" % LV.LOWER_CANAL_MM),
                            (40 + 0.72 * W, (0.35, 0.35, 0.38), (0, (1.5, 2.5)), "the same canal before insertion")):
        ax.plot([x, x + 36], [ky, ky], color=col, lw=4 if ls == "-" else 1.6, ls=ls)
        ax.text(x + 46, ky, txt, va="center", fontsize=11.5)
    if st.ring_halves:                   # S7f: each half at its own logged pose; a half not yet inserted is noted
        dv = "Tandem black, ring halves light grey; uterus, cervix and vagina see-through.  %s" % (
            (LV.ring_parked_note(st) + "  ") if LV.ring_parked_note(st) else "")
    else:
        dv = "Device black%s; uterus, cervix and vagina see-through.  " % ("" if st.has_ring else " (tandem only: no ring)")
    ax.text(W / 2, ky + 40, dv + "Distances are to the tube SEGMENT (flange to tip), fix plan S0's definition.  "
            "Patient-derived figure: keep local.", ha="center", va="center", fontsize=10.5, color=(0.35, 0.35, 0.38))
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return dict(step=st.step, lower_in_tube_frac=m["lower_in_tube_frac"], lower_max_mm=round(m["lower_max_mm"], 2),
                traversed_in_tube_frac=m["traversed_in_tube_frac"], upper_residual_mm=m["upper_residual_mm"],
                upper_residual_s=m["upper_residual_s"], depth=st.depth, depth_src=st.depth_src,
                marks={"%d" % (n_ + 1): [round(float(cs[i]), 1), round(float(dseg[i]), 2)] for n_, i in enumerate(mi)},
                carried_by=by)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--views", default="sagittal,oblique", help="any of sagittal, oblique (BT overlays), canal")
    ap.add_argument("--steps", default=None, help="canal view: step numbers and/or final, end:<phase>, d:<mm> "
                                                  "(label_views.py); the BT overlays are drawn for the last frame")
    ap.add_argument("--canal", choices=["path", "legacy"], default="path",
                    help="the model canal on the BT overlays: path (tandem_path.npz, default) or legacy (canal.npz)")
    ap.add_argument("--trim-json", default=None,
                    help="ring runs: a trimref.py JSON (t_mm, trim_cc, <TAG>[@k].dice_trim); its trimmed-reference "
                         "vagina Dice is printed beside the full-reference one (default: the reference is only named)")
    ap.add_argument("--ring-parked", choices=["omit", "draw"], default=LV.RING_PARKED,
                    help="ring runs: a ring half not yet inserted is left out and noted (omit) or drawn where parked")
    a = ap.parse_args()
    LV.set_ring_parked(a.ring_parked)
    AH.run_extra_parts(a.tag)
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    od = P["figs"] + "/labeled"
    os.makedirs(od, exist_ok=True)
    bt_views = [v for v in views if v in ("sagittal", "oblique")]
    if bt_views:
        st = LV.State(a.tag, canal=a.canal)
        st.fx = st.facts()
        prep = load_prep(a.tag)
        ring = ring_extras(st, prep, a.trim_json) if st.ring_halves else None
        if ring:
            print("ring run: wall %s | vagina Dice %s %s | trimmed %s | halves: %s | scan ring to model halves %s mm"
                  % (ring["short"], fmt(ring.get("dice"), "%.4f"), ring["ref"], ring["trim"], ring["status"],
                     None if ring["ring_mm"] is None else tuple(round(v, 2) for v in ring["ring_mm"])))
        for view in bt_views:
            Sh = a.size if view == "sagittal" else int(a.size * 1.15)
            Sw = a.size if view == "sagittal" else int(a.size * 0.93)
            img, info, ov = build(st, view, Sw, Sh, prep)
            out = "%s/%s_step%04d_overlay_%s%s.png" % (od, a.tag, st.step, view,
                                                       "_canalnpz" if a.canal == "legacy" else "")
            compose(img, info, ov, view, st, prep[0], out, ring=ring)
            print("wrote", out, "| callouts:", {k: round(v[2], 1) for k, v in ov["callouts"].items()},
                  "| canal divergence %.1f mm" % ov["canal_div"])
    if "canal" in views:
        for step in (LV.select_steps(a.tag, a.steps) if a.steps else [None]):
            st = LV.State(a.tag, step, canal="path")
            out = "%s/%s_step%04d_canal.png" % (od, a.tag, st.step)
            r = canal_view(st, out, size=int(a.size * 0.75))
            print("wrote", out, "|", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
