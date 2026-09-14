"""HOST (python 3.13): POST-HOC controls and diagnostics for the stage-2 validation.  Units mm, cc, deg.

    python posthoc.py     # -> MRI_GYN_sim/validation/posthoc_M1.json, and a 'post_hoc' section in metrics_M1.json

Declared in MRI_GYN_sim/validation/posthoc_plan.json AFTER the uterus metrics of the frozen runs were read, so
nothing here can change the pre-registered verdict.  It lives in its own file so that evaluate.py (hashed in
validation/freeze.json) stays byte-identical.  It answers two questions:
  1. Does the MECHANICS beat rigid motion, or only the device information?  Two rigid controls use the same run:
     RIGID_CANAL = Kabsch of the preBT canal points (L_end -> end, engaged, not released) onto their final simulated
     positions (the device information the simulation uses, without mechanics), applied to the UNDEFORMED surfaces;
     RIGID_PART = Kabsch of all FEM nodes rest -> final (the rigid component of the simulated motion).
     Both go through the identical route: rigid map into BT world -> vtkPolyDataToImageStencil -> dice/surf_dists.
  2. Why does the simulation lose in the HR frame?  Diagnostic runs X1-X4 (tip push off, fundal canal extension,
     nu 0.49) are scored exactly like the frozen runs (evaluate.validate_run), plus data facts (uterus/HR-CTV label
     overlap, cervix length preBT vs BT, fundal wall beyond the labelled canal end) and a frame-invariance check.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
import evaluate as ev  # noqa: E402

P = config.paths(); VAL = P["out"] + "/validation"; INP = P["inputs"]
X_TAGS = ("X1_P2_latonly", "X2_P2_dfund7", "X3_P2_nu049", "X4_PB_M2soft_latonly")


def frame_rt(frame, z, bt, al):
    """(R, t) with y_pre = R x_BT + t, for a named frame or the per-run device-to-device E_APP."""
    if frame == "E_APP":
        Rs_ = geom.pose_frame(dict(F=z["F"], a=z["a"], x=z["x"]))
        Re, te = geom.kabsch(ev.app_landmarks(z["F"], Rs_, bt.sph), ev.app_landmarks(bt.F, bt.R, bt.sph))
        return Re.T, -Re.T @ te
    fr = al["frames"][frame]
    return np.array(fr["R_BT_to_pre"]), np.array(fr["t_BT_to_pre"])


def kabsch_reg(src, dst, anchor, eps_frac=0.01):
    """Rigid (R, t) with dst ~ R src + t, translation exact at the centroids.  The rotation adds a weak identity
    prior eps_frac * (anchor spread) to the cross-covariance: once the canal has been straightened onto the rod
    the roll about the rod axis is not determined by the canal points (plain Kabsch then returns arbitrary rolls,
    e.g. 138 deg); the prior resolves only that null direction and barely moves the well-determined tilt."""
    pc, qc = src.mean(0), dst.mean(0); P0, Q0 = src - pc, dst - qc; A0 = anchor - anchor.mean(0)
    lam = eps_frac * np.trace(P0.T @ P0) / np.trace(A0.T @ A0)
    U, _, Vt = np.linalg.svd(P0.T @ Q0 + lam * (A0.T @ A0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rigid_controls(tag, frame, bt, al, named):
    z = np.load(os.path.join(P["runs"], tag, "final.npz"))
    R, t = frame_rt(frame, z, bt, al)
    to_bt = lambda Y: (np.asarray(Y) - t) @ R  # noqa: E731
    n = len(z["canal_rest"]); i_le = int(named["i_L_end"])
    sel = (np.arange(n) >= i_le) & z["engaged"].astype(bool) & ~z["released"].astype(bool)
    Vu0, _ = geom.read_obj(INP + "/pre_uterus.obj"); cu = Vu0.mean(0)
    out = dict(tag=tag, frame=frame, controls={})
    fits = {"RIGID_CANAL": (z["canal_rest"][sel], z["canal_final"][sel]), "RIGID_PART": (z["X0"], z["X"])}
    for cname, (src, dst) in fits.items():
        R_plain, _ = geom.kabsch(src, dst)
        if cname == "RIGID_CANAL":                                      # roll about the rod is a null direction
            Rk, tk = kabsch_reg(src, dst, z["X0"])
        else:
            Rk, tk = geom.kabsch(src, dst)                              # dst ~ Rk src + tk  (preBT world)
        res = np.linalg.norm(src @ Rk.T + tk - dst, axis=1)
        rec = dict(n_points=int(len(src)), fit_rms_mm=round(float(np.sqrt((res ** 2).mean())), 3),
                   rotation_plain_kabsch_deg=round(geom.rot_angle_deg(R_plain), 2),
                   fit_max_mm=round(float(res.max()), 3), rotation_deg=round(geom.rot_angle_deg(Rk), 2),
                   uterus_centroid_shift_mm=round(float(np.linalg.norm(Rk @ cu + tk - cu)), 2))
        for nm, obj in (("uterus", "uterus"), ("HR-CTV", "hrctv")):
            V0, Fc = geom.read_obj(INP + "/pre_%s.obj" % obj)
            m = ev.voxelize(to_bt(V0 @ Rk.T + tk), Fc, bt.shape, bt.aff)
            rec[nm] = ev.seg_metrics(m, bt.lab[nm], bt.sp, bt.vv)
        out["controls"][cname] = rec
    Rp, tp = geom.kabsch(z["X0"], z["X"])
    nr = np.linalg.norm(z["X"] - (z["X0"] @ Rp.T + tp), axis=1); tot = np.linalg.norm(z["X"] - z["X0"], axis=1)
    out["node_displacement_mm"] = dict(total_mean=round(float(tot.mean()), 2), total_max=round(float(tot.max()), 2),
                                       nonrigid_mean=round(float(nr.mean()), 2), nonrigid_p95=round(float(np.percentile(nr, 95)), 2),
                                       nonrigid_max=round(float(nr.max()), 2))
    return out


def sim_mask(tag, frame, bt, al, obj="uterus"):
    z = np.load(os.path.join(P["runs"], tag, "final.npz")); R, t = frame_rt(frame, z, bt, al)
    V, F = geom.read_obj(os.path.join(P["runs"], tag, "surf_%s.obj" % obj))
    return ev.voxelize((V - t) @ R, F, bt.shape, bt.aff)


def base_mask(frame, bt, al, obj="uterus"):
    fr = al["frames"][frame]; R, t = np.array(fr["R_BT_to_pre"]), np.array(fr["t_BT_to_pre"])
    V, F = geom.read_obj(INP + "/pre_%s.obj" % obj)
    return ev.voxelize((V - t) @ R, F, bt.shape, bt.aff)


def data_facts(bt, named):
    lab = lambda s, n: np.asarray(nib.load("%s/%s_MRI_label_%s.nii" % (P["data"], s, n)).dataobj) > 0  # noqa: E731
    A = nib.load("%s/preBT_MRI_label_uterus.nii" % P["data"]).affine
    vv_pre = abs(float(np.prod(np.diag(A)[:3]))) / 1000.0; vv_bt = bt.vv / 1000.0
    ut, hr = lab("preBT", "uterus"), lab("preBT", "HR-CTV"); utb, hrb = bt.lab["uterus"], bt.lab["HR-CTV"]
    c = np.load(INP + "/canal.npz"); pts, s0 = c["pts"], c["s0"]
    i_le, i_ios = int(named["i_L_end"]), int(named["i_internal_os"])
    bf = ev.below_flange_extent(hrb | utb, bt.aff, bt.F, bt.a)
    f2c = ev.entry_dist(utb, bt.aff, bt.F, bt.a)
    wall_pre = ev.exit_dist(ut, A, pts[-1], geom.unit(np.array(c["fund_dir"], float)))
    wall_pre_a0 = ev.exit_dist(ut, A, pts[-1], geom.unit(np.array(named["a0"], float)))
    cr = json.load(open(INP + "/canal_resolution.json")) if os.path.exists(INP + "/canal_resolution.json") else {}
    return dict(
        volumes_cc=dict(preBT_uterus=round(ut.sum() * vv_pre, 2), BT_uterus=round(utb.sum() * vv_bt, 2),
                        preBT_HRCTV=round(hr.sum() * vv_pre, 2), BT_HRCTV=round(hrb.sum() * vv_bt, 2),
                        HRCTV_change_frac=round(float(hrb.sum() * vv_bt / (hr.sum() * vv_pre) - 1.0), 3)),
        label_overlap=dict(preBT_frac_of_uterus_inside_HRCTV=round(float((ut & hr).sum() / ut.sum()), 3),
                           BT_frac_of_uterus_inside_HRCTV=round(float((utb & hrb).sum() / utb.sum()), 3),
                           note="the HR frame (HR-CTV-only ICP) therefore carries partial uterus information"),
        cervix_length_mm=dict(
            preBT_O_pre_to_internal_os_along_canal=round(float(s0[i_ios] - s0[0]), 2),
            preBT_portio_below_L_end=round(float(s0[i_le] - s0[0]), 2),
            BT_below_flange_on_axis=round(bf[1], 2), BT_below_flange_25mm=round(bf[0], 2),
            BT_flange_to_corpus_entry=round(f2c, 2), BT_on_axis_total=round(bf[1] + f2c, 2),
            note="BT cervix = tissue below the flange on the tandem axis + flange to uterus-label entry"),
        fundal_wall_mm=dict(preBT_canal_end_to_serosa_along_last10mm_dir=round(wall_pre, 2),
                            preBT_canal_end_to_serosa_along_a0=round(wall_pre_a0, 2),
                            BT_tip_to_serosa_along_tandem=round(ev.exit_dist(utb, bt.aff, bt.tip, bt.a), 2),
                            note="the tip push puts the labelled canal end 0.5-2.5 mm beyond the tandem tip, so the "
                                 "simulated tip-to-serosa is about the preBT wall beyond the canal end"),
        canal_resolution_corpus_invariants=cr.get("iv_corpus_invariants", cr.get("corpus_invariants")))


def fig_summary(metrics, post):
    """figs/val_summary.png: uterus MSD and Dice per run; rigid-only baseline vs rigid component of the simulated
    motion vs simulation (all through the identical voxelisation route).  Palette slots 1-3 of the dataviz reference
    palette (validated all-pairs, light mode).  Local only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"rigid-only baseline": "#2a78d6", "rigid part of sim (post hoc)": "#eb6834", "simulation": "#1baf7a"}
    rows = [("R1_P2_S0@HR", "R1  P2 tandem"), ("R8_P2_S0_cell3@HR", "R8  P2 tandem 3 mm"), ("V3_P2_M2_S0@HR", "V3  P2 +ovoids"),
            ("V5_P2_M2_soft@HR", "V5  P2 +ovoids soft*"), ("X1_P2_latonly@HR", "X1  P2 no tip push (post hoc)"),
            ("X2_P2_dfund7@HR", "X2  P2 fundal ext. 7 mm (post hoc)"), ("X3_P2_nu049@HR", "X3  P2 nu 0.49 (post hoc)"),
            ("V1_PB_M1_S0@BONE", "V1  bone tandem (INADM.)"), ("V2_PB_M2_S0@BONE", "V2  bone +ovoids (INADM.)"),
            ("V4_PB_M2_soft@BONE", "V4  bone +ovoids soft*"), ("X4_PB_M2soft_latonly@BONE", "X4  bone no tip push (post hoc)")]
    allruns = dict(metrics["runs"]); allruns.update(post["diagnostic_runs"])
    rows = [r for r in rows if r[0] in allruns and r[0] in post["controls"]]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(rows) + 1.9), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    y = np.arange(len(rows))[::-1]
    for ax, (q, lab) in zip(axs, (("msd", "uterus mean surface distance to BT (mm, lower is better)"),
                                  ("dice", "uterus Dice with BT (higher is better)"))):
        ax.set_facecolor("#fcfcfb")
        for yi, (key, _) in zip(y, rows):
            r = allruns[key]["structures"]["uterus"]; c = post["controls"][key]["controls"]["RIGID_PART"]["uterus"]
            vals = [r["baseline"][q], c[q], r["sim"][q]]
            ax.plot([min(vals), max(vals)], [yi, yi], color="#d9d8d2", lw=2, zorder=1, solid_capstyle="round")
            for (nm, cc), v in zip(col.items(), vals):
                ax.scatter(v, yi, s=64, color=cc, edgecolor="#fcfcfb", linewidth=2, zorder=3, label=nm if yi == y[0] else None)
        ax.axhline(y[len([r for r in rows if r[0].endswith("@HR")]) - 1] - 0.5, color="#52514e", lw=0.8)
        ax.grid(axis="x", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#b5b4ad"); ax.tick_params(colors="#52514e", labelsize=8, length=0)
        ax.set_xlabel(lab, color="#52514e", fontsize=9)
    axs[0].set_yticks(y); axs[0].set_yticklabels([r[1] for r in rows], fontsize=8.5, color="#0b0b0b")
    nhr = len([r for r in rows if r[0].endswith("@HR")])
    axs[0].text(axs[0].get_xlim()[1], y[0] + 0.45, "HR frame (HR-CTV-only ICP)", ha="right", va="bottom", fontsize=8, color="#52514e")
    axs[0].text(axs[0].get_xlim()[1], y[nhr] + 0.45, "BONE frame (bony-pelvis MI, no organ label)", ha="right", va="bottom", fontsize=8, color="#52514e")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "* soft = supports at the lower bounds of the prior ranges (pre-declared sensitivity). INADM. = failed the "
             "numeric/admissibility gates. Rigid part = Kabsch fit of all FEM nodes rest -> final applied to the undeformed "
             "surfaces.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fn = os.path.join(P["figs"], "val_summary.png"); fig.savefig(fn, dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
    print("wrote", fn)


def main():
    fz = json.load(open(VAL + "/freeze.json")); al = json.load(open(VAL + "/alignment.json"))
    named = json.load(open(INP + "/named.json")); bt = ev.BT(); cache = {}
    out = dict(units="mm, cc, deg", status="POST HOC (declared in validation/posthoc_plan.json after the uterus "
               "metrics were read); does not change the pre-registered verdict", controls={}, diagnostic_runs={})
    for tag in list(fz["runs"]) + [t for t in X_TAGS if os.path.exists(os.path.join(P["runs"], t, "final.npz"))]:
        s = json.load(open(os.path.join(P["runs"], tag, "summary.json")))
        frame = ev.FRAME_OF_POSE.get(s["cfg"]["pose"], "E_APP")
        rc = rigid_controls(tag, frame, bt, al, named)
        out["controls"]["%s@%s" % (tag, frame)] = rc
        if tag in X_TAGS:
            r = ev.validate_run(tag, frame, bt, al, named, cache)
            out["diagnostic_runs"]["%s@%s" % (tag, frame)] = r
        c = rc["controls"]
        print("%-22s %-6s canal-fit rms %.2f | uterus RIGID_CANAL %.3f/%.2f/%.2f RIGID_PART %.3f/%.2f/%.2f | "
              "HR-CTV RC %.3f/%.2f RP %.3f/%.2f | nonrigid mean %.1f mm"
              % (tag, frame, c["RIGID_CANAL"]["fit_rms_mm"], *(c["RIGID_CANAL"]["uterus"][k] for k in ("dice", "msd", "hd95")),
                 *(c["RIGID_PART"]["uterus"][k] for k in ("dice", "msd", "hd95")),
                 c["RIGID_CANAL"]["HR-CTV"]["dice"], c["RIGID_CANAL"]["HR-CTV"]["msd"],
                 c["RIGID_PART"]["HR-CTV"]["dice"], c["RIGID_PART"]["HR-CTV"]["msd"], rc["node_displacement_mm"]["nonrigid_mean"]), flush=True)
        if tag in X_TAGS:
            su = r["structures"]; g = r["uterus_geometry"]["sim"]
            print("    X-run sim: %s | uterus %.3f/%.2f/%.2f | HR-CTV %.3f/%.2f/%.2f | tip2ser %.1f f2corp %.1f flex %.1f"
                  % (r["numerics"]["status"], su["uterus"]["sim"]["dice"], su["uterus"]["sim"]["msd"], su["uterus"]["sim"]["hd95"],
                     su["HR-CTV"]["sim"]["dice"], su["HR-CTV"]["sim"]["msd"], su["HR-CTV"]["sim"]["hd95"],
                     g["tip_to_serosa_along_axis_mm"], g["flange_to_corpus_entry_mm"], g["flexion_angle_deg"]), flush=True)
    # frame invariance: the same model settings started from two different rigid frames, compared in BT space
    inv = {}
    for a_, b_ in (("V5_P2_M2_soft@HR", "V4_PB_M2_soft@BONE"), ("V3_P2_M2_S0@HR", "V2_PB_M2_S0@BONE"), ("R1_P2_S0@HR", "V1_PB_M1_S0@BONE")):
        (ta, fa), (tb, fb) = a_.split("@"), b_.split("@")
        ma, mb = sim_mask(ta, fa, bt, al), sim_mask(tb, fb, bt, al)
        inv["%s_vs_%s" % (a_, b_)] = ev.metrics(ma, mb, bt.sp)
    inv["baseline_HR_vs_baseline_BONE"] = ev.metrics(base_mask("HR", bt, al), base_mask("BONE", bt, al), bt.sp)
    out["frame_invariance_uterus"] = dict(metrics=inv, note="Dice/MSD/HD95 between two SIMULATED uteri in BT space; "
                                          "the device pins the final shape, largely independent of the starting frame")
    out["data_facts"] = data_facts(bt, named)
    m = json.load(open(VAL + "/metrics_M1.json"))
    allruns = dict(m["runs"]); allruns.update(out["diagnostic_runs"])
    dl = {}
    for key, rc in out["controls"].items():
        if key not in allruns:
            continue
        rec = {}
        for nm in ("uterus", "HR-CTV"):
            s = allruns[key]["structures"][nm]; b, sm = s["baseline"], s["sim"]
            rp, rcn = rc["controls"]["RIGID_PART"][nm], rc["controls"]["RIGID_CANAL"][nm]
            rec[nm] = dict(sim_minus_baseline={q: round(sm[q] - b[q], 3) for q in ("dice", "msd", "hd95")},
                           sim_minus_rigid_part={q: round(sm[q] - rp[q], 3) for q in ("dice", "msd", "hd95")},
                           sim_minus_rigid_canal={q: round(sm[q] - rcn[q], 3) for q in ("dice", "msd", "hd95")})
        rec["numeric_gates_pass"] = allruns[key]["numerics"]["numeric_gates_pass"]
        dl[key] = rec
    out["deltas"] = dict(note="negative msd/hd95 and positive dice = simulation better", runs=dl)
    json.dump(out, open(VAL + "/posthoc_M1.json", "w"), indent=1)
    m["post_hoc"] = out
    json.dump(m, open(VAL + "/metrics_M1.json", "w"), indent=1)
    fig_summary(m, out)
    print(json.dumps(out["frame_invariance_uterus"], indent=1)); print(json.dumps(out["data_facts"], indent=1))
    print("wrote", VAL + "/posthoc_M1.json", "and metrics_M1.json['post_hoc']")


if __name__ == "__main__":
    main()
