"""HOST (python 3.13): finisher report -> MRI_GYN_sim/final/verdict.json, final/report.md, figs/final_*.png.
Reads only files written by final_eval.py (bdev.json, scores/, boot.json, calib_rescore.json), the stage-2/3 outputs
(validation/metrics_M1.json, calibration/best.json, heldout.json) and final/A1_repro.json.  Local only.  Units mm.

    python final_eval.py report      (or: python final_report.py)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

P = config.paths(); FIN = P["out"] + "/final"; VAL = P["out"] + "/validation"; CAL = P["out"] + "/calibration"
ROLES = [("M1_fixed", "F0_R1_P2_S0_fixed", "R1 settings (P2, S0, tandem only), fixed code"),
         ("best", "F1_best", "calibrated best (C3_L19 parameters), BONE frame, fixed code, delta_fund 7 mm"),
         ("best_margin", "F1m_best_margin", "as F1 but the original tip_push_margin -6.72 mm (no canal extension)"),
         ("best_anchor", "F1a_best_anchor", "as F1 + supports anchored to the device-rigid placement"),
         ("best_HR", "F2_best_HR", "calibrated parameters in the HR frame (pose P2)"),
         ("best_P1", "F3_best_P1", "calibrated parameters at pose P1 (preBT axis a0, fornix-rule d_F)"),
         ("mesh5", "F1_c5", "F1 at 5 mm cells"), ("mesh3", "F1_c3", "F1 at 3 mm cells")]
# the pre-registered criteria, verbatim from the design spec section 8.5 (hashed into verdict.json)
CRITERIA = """NUMERIC (every run): Phase A control passes; hold converges; min hexa volume ratio > 0.2; no NaN; tie and penetration criteria met; piercing = 0.
M1 VALIDATION (P2-S0, tandem only): the sealed uterus MSD < 2.07 AND HD95 < 4.66 AND Dice > 0.816 (beats B_HR); tip-to-serosa > 0 (tip inside the uterus); uterus volume change < 3%.
M2 VALIDATION (S1-frozen): P2-S1 beats B_HR on all 3 uterus metrics. P1-S1 beats its B0 on uterus MSD and HD95. P1-S1 tip-to-serosa within +/-3 mm of BT. Mesh convergence: |dMSD_uterus| between 3 and 4 mm cells < 0.3 mm."""
B_HR = dict(dice=0.816, msd=2.07, hd95=4.66)
STRUCTS = (("uterus", "uterus (in the calibration objective)"),
           ("HR-CTV", "HR-CTV (not in the objective; analysts saw bone-frame HR-CTV numbers in stage 2)"),
           ("HR-CTV-minus-uterus", "HR-CTV minus uterus (less uterus-coupled check)"))


def load(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def f3(m):
    return "%.3f / %.2f / %.2f / %.2f" % (m["dice"], m["msd"], m["hd95"], m.get("hd95max", float("nan")))


def prereg_table(sc, m1):
    rows = []
    r1 = (m1 or {}).get("preregistered_M1_validation_R1_P2_S0")
    if r1:
        u = m1["runs"]["R1_P2_S0@HR"]["structures"]["uterus"]["sim"]
        rows.append(("M1 (R1_P2_S0, frozen stage-1 code)", "FAIL" if not r1["PASS"] else "PASS",
                     "uterus %.3f / %.2f / %.2f vs B_HR 0.816 / 2.07 / 4.66; tip inside: %s; volume < 3%%: %s"
                     % (u["dice"], u["msd"], u["hd95"], r1["tip_to_serosa_gt_0"], r1["uterus_volume_change_lt_3pct"])))
    f0 = sc.get("M1_fixed")
    if f0:
        u = f0["models"]["SIM"]["uterus"]; ok = u["dice"] > 0.816 and u["msd"] < 2.07 and u["hd95"] < 4.66
        rows.append(("M1 re-run with the fixed code (F0, post hoc)", "PASS" if ok else "FAIL",
                     "uterus %.3f / %.2f / %.2f; status %s" % (u["dice"], u["msd"], u["hd95"], f0["status"])))
    hr = sc.get("best_HR")
    if hr:
        u = hr["models"]["SIM"]["uterus"]; ok = u["dice"] > 0.816 and u["msd"] < 2.07 and u["hd95"] < 4.66
        rows.append(("M2-1 P2-S1 beats B_HR on all 3 uterus metrics (S1 = exploratory stage-3 calibration, redefined "
                     "after unblinding)", "PASS" if ok else "FAIL",
                     "F2: uterus %.3f / %.2f / %.2f (stage-3 C3_K_HR: 0.752 / 2.91 / 6.28)" % (u["dice"], u["msd"], u["hd95"])))
    p1 = sc.get("best_P1")
    if p1:
        u = p1["models"]["SIM"]["uterus"]; b = p1["models"]["RIGID_FRAME"]["uterus"]
        ok = u["msd"] < b["msd"] and u["hd95"] < b["hd95"]
        rows.append(("M2-2 P1-S1 beats its B0 on uterus MSD and HD95", ("PASS" if ok else "FAIL") + ("" if p1["gates"]["pass_preregistered"] else " (run fails numeric gates)"),
                     "F3 (E_APP frame): sim %.2f / %.2f vs B0 %.2f / %.2f" % (u["msd"], u["hd95"], b["msd"], b["hd95"])))
        e = p1["models"]["SIM"]["landing"]["tip_to_serosa_err_mm"]
        rows.append(("M2-3 P1-S1 tip-to-serosa within +/-3 mm of BT", "PASS" if abs(e) <= 3 else "FAIL", "error %+.2f mm (sub-voxel)" % e))
    b4, b3 = sc.get("best"), sc.get("mesh3")
    if b4 and b3:
        d = b3["models"]["SIM"]["uterus"]["msd"] - b4["models"]["SIM"]["uterus"]["msd"]
        both = b4["status"] == "converged" and b3["status"] == "converged"
        res = ("PASS" if abs(d) < 0.3 else "FAIL") if both else "INCONCLUSIVE"
        rows.append(("M2-4 mesh convergence |dMSD_uterus| 3 vs 4 mm < 0.3 mm", res,
                     "F1_c3 - F1: %+.3f mm; statuses %s / %s (the criterion needs two converged states; the metric is also "
                     "insensitive to the force errors, see mesh study)" % (d, b3["status"], b4["status"])))
    return rows


def mesh_study(sc):
    lv = [(k, sc[k]) for k in ("mesh5", "best", "mesh3") if sc.get(k)]
    out = []
    for k, r in lv:
        n = r["numerics"]
        out.append(dict(run=r["tag"], cell_mm=r["cfg"]["cell_mm"], status=r["status"], uterus_msd=r["models"]["SIM"]["uterus"]["msd"],
                        uterus_dice=r["models"]["SIM"]["uterus"]["dice"], F_axial_mN=n.get("final_F_axial_mN"),
                        F_ovoid_axial_mN=n.get("F_ovoid_axial_mN"), F_lat_sum_mN=n.get("final_F_lat_sum_mN"),
                        lam_max_mm=n.get("final_lam_max_mm"), volumes=n.get("volumes_cc"), n_nodes=n.get("n_nodes"),
                        ms_T=n.get("ms_per_step_median_T"), ms_H=n.get("ms_per_step_median_H")))
    rich = None
    if len(out) == 3:
        q = {}
        for key in ("uterus_msd", "F_axial_mN", "F_ovoid_axial_mN"):
            f1, f2, f3_ = [o[key] for o in out]            # coarse -> fine (5, 4, 3 mm; ratio ~1.25-1.33)
            if None in (f1, f2, f3_) or f2 == f3_ or (f1 - f2) * (f2 - f3_) <= 0:
                q[key] = dict(monotone=False, values=[f1, f2, f3_]); continue
            r_ = np.mean([out[0]["cell_mm"] / out[1]["cell_mm"], out[1]["cell_mm"] / out[2]["cell_mm"]])
            p = float(np.log(abs(f1 - f2) / abs(f2 - f3_)) / np.log(r_))
            q[key] = dict(monotone=True, values=[f1, f2, f3_], observed_order=round(p, 2),
                          fine_minus_mid_rel=round(abs(f3_ - f2) / max(abs(f3_), 1e-9), 3))
        rich = q
    return out, rich


def fig_bdev(bd, sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    C = ["#2a78d6", "#eb6834", "#1baf7a"]
    d = [r["d_F_mm"] for r in bd["curve"]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6)); fig.patch.set_facecolor(SURF)
    for ax, q, lab in zip(axs, ("msd", "dice"), ("mean surface distance to BT (mm, lower is better)", "Dice with BT (higher is better)")):
        ax.set_facecolor(SURF)
        for (nm, _), col in zip(STRUCTS, C):
            y = [r[nm][q] for r in bd["curve"]]
            ax.plot(d, y, color=col, lw=2, zorder=3)
            ax.text(d[-1] + 0.3, y[-1], nm, color=INK2, fontsize=8, va="center")
        for k, ls in (("d_F_tip", "-"), ("d_F_hr", "--")):
            ax.axvline(bd["picks"][k], color=AX, lw=1, ls=ls, zorder=1)
            ax.text(bd["picks"][k], ax.get_ylim()[1], " %s = %.1f" % (k, bd["picks"][k]), fontsize=7.5, color=INK2, va="top",
                    rotation=90)
        s = sc.get("best")
        if s:
            ax.axhline(s["models"]["SIM"]["uterus"][q], color=C[0], lw=1, ls=":", zorder=2)
            ax.text(0.2, s["models"]["SIM"]["uterus"][q], "calibrated sim F1, uterus", fontsize=7.5, color=INK2, va="bottom")
        ax.set_xlabel("flange depth d_F below the preBT canal-label end (mm)", fontsize=9, color=INK2)
        ax.set_ylabel(lab, fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(AX)
        ax.tick_params(colors=INK2, labelsize=8, length=0); ax.set_xlim(-0.5, 31)
    fig.suptitle("B_dev: undeformed preBT anatomy placed rigidly on the real BT tandem (no simulation), vs flange depth",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fn = P["figs"] + "/final_bdev_curve.png"; fig.savefig(fn, dpi=110, facecolor=SURF); plt.close(fig)
    return fn


def fig_summary(sc, boot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    runs = [(k, sc[k]) for k in ("best", "best_anchor", "best_margin", "best_HR", "M1_fixed") if sc.get(k)]
    comps = [("RIGID_FRAME", "rigid, registration frame", "#898781"), ("B_DEV_TIP", "B_dev (canal end at tip), no run", "#2a78d6"),
             ("RIGID_CANAL_eps0.01", "rigid canal-on-tandem (from run)", "#eb6834"), ("SIM", "simulation", "#1baf7a")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.55 * len(runs) + 2.2), sharey=True); fig.patch.set_facecolor(SURF)
    y = np.arange(len(runs))[::-1]
    for ax, nm in zip(axs, ("uterus", "HR-CTV")):
        ax.set_facecolor(SURF)
        for yi, (k, r) in zip(y, runs):
            vals = [r["models"][c][nm]["msd"] for c, _, _ in comps]
            ax.plot([min(vals), max(vals)], [yi, yi], color="#e1e0d9", lw=2, zorder=1)
            for (c, lab, col), v in zip(comps, vals):
                ax.scatter(v, yi, s=70, color=col, edgecolor=SURF, linewidth=2, zorder=3, label=lab if yi == y[0] else None)
            bb = (boot or {}).get(r["tag"], {}).get("deltas_sim_minus", {}).get("B_DEV_TIP", {}).get("%s_msd" % nm)
            if bb:
                ax.text(max(vals) + 0.15, yi, "sim - B_dev: %+.2f [%+.2f, %+.2f]" % (bb["median"], bb["p5"], bb["p95"]),
                        fontsize=7.5, color=INK2, va="center")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(AX); ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_xlabel("%s: mean surface distance to BT (mm, lower is better)" % nm, fontsize=9, color=INK2)
        lo, hi = ax.get_xlim(); ax.set_xlim(max(0, lo), hi + 3.2)
    axs[0].set_yticks(y); axs[0].set_yticklabels(["%s (%s frame)" % (r["tag"], r["frame"]) for _, r in runs], fontsize=8.5, color=INK)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.01, 0.01, "Brackets: median and 5-95 %% range of (sim - B_dev) MSD over %d rigid frame perturbations "
             "(sd 2 deg/axis, 1.35 mm/axis) applied to both." % (list((boot or {"x": {"n_draws": 0}}).values())[0].get("n_draws", 0)),
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fn = P["figs"] + "/final_summary.png"; fig.savefig(fn, dpi=110, facecolor=SURF); plt.close(fig)
    return fn


def fig_slices(sc, bd):
    """Oblique planes through the REAL BT tandem: BT labels (solid) vs model (dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import geom
    import evaluate as ev
    import render
    import final_eval as fe
    cx = fe.Ctx(); bt = cx.bt; img = np.asarray(nib.load(P["data"] + "/BT_MRI.nii").dataobj).astype(np.float32)
    cols = [("rigid BONE frame", cx.surf, cx.to_bt_frame("BONE")),
            ("B_dev d_F %.1f (no run)" % bd["picks"]["d_F_tip"], cx.surf, cx.bdev(bd["picks"]["d_F_tip"]))]
    for k in ("best", "best_anchor"):
        r = sc.get(k)
        if r:
            rd = os.path.join(P["runs"], r["tag"]); surfs = {nm: geom.read_obj(rd + "/surf_%s.obj" % ob) for nm, ob in (("uterus", "uterus"), ("HR-CTV", "hrctv"))}
            cols.append(("sim %s" % r["tag"], surfs, cx.to_bt_frame(r["frame"])))
    masks = [{nm: ev.voxelize(fm(V), F, bt.shape, bt.aff) for nm, (V, F) in s.items()} for _, s, fm in cols]
    planes = [("tandem-sagittal", bt.R[1]), ("tandem-coronal", bt.R[0])]
    fig, axs = plt.subplots(2, len(cols), figsize=(4.4 * len(cols), 10.5), squeeze=False)
    for ri, (pn, hz) in enumerate(planes):
        S, Vv, samp = render._plane(bt, hz, s_half=50, v_lo=-40, v_hi=85); im = samp(img, 1)
        lo, hi = np.percentile(im[im > 0], [1, 99.5])
        ref = {nm: samp(bt.lab[nm], 0) for nm in ("uterus", "HR-CTV")}; ap = samp(bt.lab["applicator"], 0)
        for ci, ((name, _, _), mk) in enumerate(zip(cols, masks)):
            ax = axs[ri, ci]; ext = [S.min(), S.max(), Vv.min(), Vv.max()]
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=ext)
            ax.contour(S, Vv, ap, [0.5], colors="yellow", linewidths=0.7)
            for nm, col in (("uterus", "cyan"), ("HR-CTV", "red")):
                ax.contour(S, Vv, ref[nm], [0.5], colors=col, linewidths=1.3)
                ax.contour(S, Vv, samp(mk[nm], 0), [0.5], colors=col, linewidths=1.3, linestyles="dashed")
            m = fe.metrics_ext(mk["uterus"], bt.lab["uterus"], bt.sp, bt.vv)
            ax.set_title("%s\n%s\nuterus Dice %.2f  MSD %.2f mm" % (name, pn, m["dice"], m["msd"]), fontsize=8)
            ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    fig.suptitle("BT MRI, planes through the real tandem. solid = BT label, dashed = model; cyan uterus, red HR-CTV, "
                 "yellow BT applicator", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fn = P["figs"] + "/final_slices.png"; fig.savefig(fn, dpi=100); plt.close(fig)
    return fn


def main():
    bd = load(FIN + "/bdev.json"); boot = load(FIN + "/boot.json", {}); cr = load(FIN + "/calib_rescore.json")
    m1 = load(VAL + "/metrics_M1.json"); best = load(CAL + "/best.json"); ho = load(CAL + "/heldout.json")
    rep = load(FIN + "/A1_repro.json")
    sc = {}
    for role, tag, _ in ROLES:
        p = FIN + "/scores/%s.json" % tag
        if os.path.exists(p):
            sc[role] = json.load(open(p))
    pre = prereg_table(sc, m1); ms, rich = mesh_study(sc)
    verdict = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), units="mm; Dice [-]; mN",
                   criteria_text_sha256=hashlib.sha256(CRITERIA.encode()).hexdigest(), criteria_text=CRITERIA,
                   preregistration_status=[dict(criterion=a, result=b, detail=c) for a, b, c in pre],
                   bdev_picks=bd["picks"] if bd else None, runs={r: sc[r]["tag"] for r in sc}, mesh_study=ms, mesh_richardson=rich,
                   calibration_rescore_argmin=cr["argmin"] if cr else None, reproducibility=rep)
    # headline comparison
    head = {}
    for role in ("best", "best_anchor", "best_HR", "M1_fixed", "best_margin", "best_P1"):
        r = sc.get(role)
        if r:
            head[r["tag"]] = {m: {nm: r["models"][m][nm] for nm, _ in STRUCTS} for m in
                              ("SIM", "RIGID_FRAME", "B_DEV_TIP", "B_DEV_HR", "B_DEV_ORACLE", "RIGID_CANAL_eps0", "RIGID_CANAL_eps0.01", "RIGID_CANAL_eps0.1")
                              if m in r["models"]}
    verdict["headline"] = head; verdict["noise_floor"] = boot
    os.makedirs(P["figs"], exist_ok=True)
    figs = []
    if bd:
        figs.append(fig_bdev(bd, sc))
    if sc:
        figs.append(fig_summary(sc, boot))
        try:
            figs.append(fig_slices(sc, bd))
        except Exception as e:  # noqa: BLE001
            print("slices figure failed:", e)
    verdict["figures"] = figs
    json.dump(verdict, open(FIN + "/verdict.json", "w"), indent=1)
    # ---------------------------------------------------------------- report.md
    L = ["# applicator_sim: finisher report (local, derived data)", "",
         "Written %s. Units mm, Dice [-], mN. One patient. Everything below the pre-registration table is post hoc "
         "(the uterus was unblinded in stage 2)." % verdict["written"], "", "## Pre-registration status", "",
         "| criterion | result | detail |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % row for row in pre]
    L += ["", "Criteria text sha256: `%s`." % verdict["criteria_text_sha256"], "",
          "## Headline: simulation vs rigid comparators", "",
          "Dice / MSD / HD95 (pooled) / HD95max (max directed), mm. B_DEV uses no simulation run; RIGID_CANAL is fitted to "
          "the run's final canal.", ""]
    for tag, mods in head.items():
        r = [v for v in sc.values() if v["tag"] == tag][0]
        g = r["gates"]
        L += ["### %s (%s frame; status %s; gates: pre-registered %s, added %s, physics %s)" % (
            tag, r["frame"], r["status"], g["pass_preregistered"], g["pass_added_v2"], g["pass_physics"]), "",
              "| model | " + " | ".join(n for n, _ in STRUCTS) + " |", "|---|" + "---|" * len(STRUCTS)]
        for m, v in mods.items():
            L.append("| %s | %s |" % (m, " | ".join(f3(v[nm]) for nm, _ in STRUCTS)))
        la = r["models"]["SIM"]["landing"]
        L += ["", "Landing (sub-voxel): sim tip-to-serosa %.2f mm (BT %.2f), flange-to-corpus %.2f mm (BT %.2f)." % (
            la["tip_to_serosa_mm"], la["BT_tip_to_serosa_mm"], la["flange_to_corpus_mm"], la["BT_flange_to_corpus_mm"]), ""]
        bb = boot.get(tag)
        if bb:
            L += ["Noise floor (%d frame perturbations, same draw for sim and comparator; sim - comparator MSD, median [p5, p95]):" % bb["n_draws"], ""]
            for m, d in bb["deltas_sim_minus"].items():
                L.append("- %s: uterus %+.2f [%+.2f, %+.2f] -> %s; HR-CTV %+.2f [%+.2f, %+.2f] -> %s" % (
                    m, d["uterus_msd"]["median"], d["uterus_msd"]["p5"], d["uterus_msd"]["p95"], d["uterus_verdict"],
                    d["HR-CTV_msd"]["median"], d["HR-CTV_msd"]["p5"], d["HR-CTV_msd"]["p95"], d["HR-CTV_verdict"]))
            L.append("")
    L += ["## Numerics of the finisher runs", "", "| run | status | steps (hold) | ms/step T / H | CG its T median (at cap) / H median | "
          "f_res mN | drift mm | lam max mm | rod beyond canal mm | min vol ratio | max stretch | neg corner tets | vol out of [0.85,1.15] | "
          "mech. volume cc (grid hull) | tip / ovoid axial force mN |", "|" + "---|" * 15]
    for role, tag, _ in ROLES:
        r = sc.get(role)
        if not r:
            continue
        n = r["numerics"]; cT = n.get("cg_its_T") or {}; cH = n.get("cg_its_H") or {}; v = n.get("volumes_cc") or {}
        L.append("| %s | %s | %s (%s) | %s / %s | %s (%s) / %s | %s | %s | %s | %s | %.3f | %.2f | %s | %s | %s (%s) | %s / %s |" % (
            tag, r["status"], n.get("n_steps"), n.get("n_hold_steps"), n.get("ms_per_step_median_T"), n.get("ms_per_step_median_H"),
            cT.get("median"), cT.get("frac_at_cap"), cH.get("median"),
            None if n.get("final_f_res_max_mN") is None else round(n["final_f_res_max_mN"], 3),
            None if n.get("final_drift_est_mm") is None else round(n["final_drift_est_mm"], 4),
            None if n.get("final_lam_max_mm") is None else round(n["final_lam_max_mm"], 2),
            None if n.get("final_rod_beyond_canal_end_mm") is None else round(n["final_rod_beyond_canal_end_mm"], 2),
            n.get("min_vol_ratio_run") or 0, n.get("max_stretch_run") or 0, n.get("n_neg_corner_tet_run_max"), n.get("n_vol_out_0p85_1p15"),
            v.get("fine_hull_cc"), v.get("grid_hull_cc"),
            None if n.get("final_F_axial_mN") is None else round(n["final_F_axial_mN"]), None if n.get("F_ovoid_axial_mN") is None else round(n["F_ovoid_axial_mN"])))
    L += ["", "Gates per run (pre-registered / added after review / physics):", ""]
    for role, tag, desc in ROLES:
        r = sc.get(role)
        if r:
            g = r["gates"]
            fails = [k for grp in ("preregistered", "added_v2", "physics") for k, v in g[grp].items() if not v]
            L.append("- %s (%s): failing = %s" % (tag, desc, fails or "none"))
    L += ["", "## Mesh study (5 / 4 / 3 mm)", ""]
    for o in ms:
        L.append("- %s (%s mm, %s): uterus MSD %.3f, Dice %.3f; tip force %s mN; ovoid axial %s mN; lateral tie sum %s mN; lam max %s; "
                 "mechanical volume %s cc (grid hull %s cc)" % (o["run"], o["cell_mm"], o["status"], o["uterus_msd"], o["uterus_dice"],
                                                               None if o["F_axial_mN"] is None else round(o["F_axial_mN"]),
                                                               None if o["F_ovoid_axial_mN"] is None else round(o["F_ovoid_axial_mN"]),
                                                               None if o["F_lat_sum_mN"] is None else round(o["F_lat_sum_mN"]),
                                                               None if o["lam_max_mm"] is None else round(o["lam_max_mm"], 2),
                                                               (o["volumes"] or {}).get("fine_hull_cc"), (o["volumes"] or {}).get("grid_hull_cc")))
    if rich:
        L += ["", "Observed convergence (Richardson, where monotone): `%s`" % json.dumps(rich)]
    if bd:
        L += ["", "## B_dev curve (no simulation)", "", "Picks: %s. Uterus / HR-CTV at the picks (BONE roll):" % json.dumps(bd["picks"]), ""]
        for k, v in bd["selected"].items():
            L.append("- %s: uterus %s; HR-CTV %s" % (k, f3(v["uterus"]), f3(v["HR-CTV"])))
    if cr:
        L += ["", "## Calibration argmin under sub-voxel landing", "", "`%s` (frozen best: %s)" % (json.dumps(cr["argmin"]), best and best["best_tag"])]
    if ho:
        L += ["", "Stage-3 held-out file (read-only here): calibrated C3_L19 HR-CTV %s; its verdict compared against the "
              "bone-frame rigid baseline, which any device-informed placement beats (not evidence for the mechanics)."
              % f3(dict(ho["runs"][best["best_tag"]]["HR_CTV"], hd95max=float("nan")))]
    if rep:
        L += ["", "## Reproducibility", "", "- " + "; ".join("%s: %s" % kv for kv in rep["result"].items())]
    L += ["", "Figures: " + ", ".join(os.path.basename(f) for f in figs)]
    open(FIN + "/report.md", "w").write("\n".join(L) + "\n")
    print("wrote", FIN + "/verdict.json", FIN + "/report.md", figs)


if __name__ == "__main__":
    main()
