"""HOST report (spec 8.5 / 11).  Refuses to read any sealed_heldout.json unless frozen/S1.json exists (the S1
calibration has been frozen by calibrate.py).  Writes eval/report.md and eval/summary.json (derived data, local).

    python report.py                    # visible-only report (allowed any time)
    python report.py --unseal           # requires frozen/S1.json; applies the pre-registered criteria
Sections: E0; G0 gate; canal-length resolution; every run (timing, convergence, residuals, forces, flags) with the
pre-registered NUMERIC gates; visible metrics vs B0; sealed metrics only after the S1 freeze.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

P = config.paths()
B = dict(B_HR=dict(dice=0.816, msd=2.07, hd95=4.66), B1=dict(dice=0.883, msd=1.38, hd95=3.38),
         B2=dict(dice=0.91, msd=1.0, hd95=2.8))
SUPERSEDED = ("_v1eqAL", "_v2hold1n", "_v3capbug", "_vfloorbug", "_ks300", "_nodecontact", "R1x_")   # superseded/diagnostic
# STAGE-1 REPORT ONLY.  The stage-1 unsealing mechanism (frozen/S1.json) was never used: stage 2 unsealed the uterus
# through validation/freeze.json, stage 3 froze calibration/best.json, and the finisher report is final/report.md
# (final_report.py).  eval/report.md carries a STALE banner.


def load(p):
    with open(p) as fh:
        return json.load(fh)


def frozen_ok():
    f = P["frozen"] + "/S1.json"; h = P["frozen"] + "/S1.sha256"
    if not (os.path.exists(f) and os.path.exists(h)):
        return False
    return hashlib.sha256(open(f).read().encode()).hexdigest() == open(h).read().split()[0]


def numeric_gate(s, v=None):
    """Pre-registered NUMERIC criteria (README): Phase A control, hold converged, min hexa volume ratio > 0.2,
    no NaN, tie (mean < 0.3, max < 0.6 mm) and penetration (< 0.5 mm) criteria, piercing = 0."""
    fin = s.get("final", {})
    c = dict(phaseA=bool(s.get("phaseA_control_ok")), converged=bool(s.get("converged")),
             vol_ratio=bool((s.get("min_vol_ratio_run") or 0) > 0.2), finite=not s.get("nonfinite", True),
             tie=bool(fin.get("tie_err_mean", 9) < 0.3 and fin.get("tie_err_max", 9) < 0.6),
             penetration=bool((fin.get("pen_max") or 0) < 0.5))
    if v is not None:
        c["piercing"] = bool(v["piercing"]["frac_rod_outside_body_beyond_entry"] == 0)
    return c


def f(x, fmt="%.2f"):
    return "-" if x is None else (fmt % x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    if a.unseal and not frozen_ok():
        sys.exit("REFUSED: frozen/S1.json (with a matching sha256) does not exist; sealed metrics stay sealed.")
    L = ["# applicator_sim report (local, derived data)", "",
         "Units: mm, cc, deg, mN, ms. Forces are penalty-tie forces at an ASSUMED E; UNVALIDATED. Timings are medians "
         "on a shared host whose CPUs are saturated by another training container (docker --cpus 2).", ""]
    e0 = load(P["eval"] + "/E0.json") if os.path.exists(P["eval"] + "/E0.json") else None
    L.append("## E0 evaluator gate: %s" % ("PASS" if e0 and e0.get("pass") else ("FAIL" if e0 else "not run")))
    if e0:
        for k in ("B_HR", "B1"):
            for nm, r in e0[k].items():
                g = r["got"]; L.append("- %s %s: Dice %.3f, MSD %.2f, HD95 %.2f (expected %s)" % (k, nm, g["dice"], g["msd"], g["hd95"], r["expected"]))
    L.append("")
    g0p = P["logs"] + "/G0.json"
    if os.path.exists(g0p):
        g0 = load(g0p); L += ["## G0 gate (logs/G0.json)", ""]
        g1 = g0.get("G0.1", {}); g2 = g0.get("G0.2", {}); g3 = g0.get("G0.3", {}); g4 = g0.get("G0.4", {}); g6 = g0.get("G0.6", {})
        L.append("- G0.1 components: StaticSolver data %s, cellWidth honoured %s, RSSFF runtime rewrite %s, vagina membrane %s"
                 % (g1.get("StaticSolver_data_ok"), g1.get("cellWidth_honoured"), g1.get("rssff_rewrite_ok"), g1.get("vagina_membrane_ok")))
        for cw, cnt in (g1.get("sparse_grid_counts") or {}).items():
            L.append("  - cell %s mm: %d nodes, %d hexa, real cell %s mm" % (cw, cnt["n_nodes"], cnt["n_hexa"], cnt["cell_real_mm"]))
        L.append("- G0.2 DNT one step vs CG(1000): %s" % {k: "%.1e mm" % v for k, v in g2.items() if k.startswith("maxdiff")})
        L.append("- G0.3 DNT+LDL (20 steps) vs mapped springs + CG: canal max diff %s mm, pass %s" % (f(g3.get("canal_maxdiff_mm"), "%.1e"), g3.get("pass")))
        L.append("- G0.4 zero-load control: umax %s mm, pass %s" % (g4.get("umax_mm"), g4.get("pass_")))
        L.append("- G0.6 anisotropic-tie ForceField (P = I) vs RSSFF DNT on the body: max diff %s mm, pass %s; default config "
                 "phase-T median %s ms/step" % (f(g6.get("maxdiff_pyffI_vs_rssff_mm"), "%.1e"), g6.get("pass_equivalence"), g6.get("default_ms_median_T")))
        for k, v in (g0.get("G0.5") or {}).items():
            if isinstance(v, dict):
                L.append("  - G0.5 %s: %s ms/step (phase T median), init %s s" % (k, v.get("ms_median_T"), v.get("t_init_s")))
        L.append("")
    cr = P["inputs"] + "/canal_resolution.json"
    if os.path.exists(cr):
        c = load(cr); L += ["## Canal-length resolution (inputs/canal_resolution.json)", "", c.get("conclusion", ""), "",
                            "Insertion depth definition: %s" % json.dumps(c.get("tandem_insertion_depth_definition")), ""]
    L += ["## Runs (runs/<tag>/run.json)", "",
          "| run | status | steps (hold) | init s | ms/step med (T) | run s | umax / umean mm | tie err mean / max mm | F_axial mN | max stretch | min V ratio | tie impl / solver / cell | numeric gates |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for sp in sorted(glob.glob(P["runs"] + "/*/summary.json")):
        s = load(sp); t = os.path.basename(os.path.dirname(sp)); c = s.get("cfg", {}); fin = s.get("final", {})
        vp = os.path.join(P["eval"], t, "visible.json"); v = load(vp) if os.path.exists(vp) else None
        gate = numeric_gate(s, v); failed = [k for k, ok in gate.items() if not ok]
        sup = any(t.endswith(x) for x in SUPERSEDED)
        L.append("| %s%s | %s | %d (%d) | %s | %s (%s) | %s | %s / %s | %s / %s | %s | %s | %s | %s / %s / %s | %s |" % (
            t, " (superseded)" if sup else "", s["status"], s["n_steps"], s.get("n_hold_steps", 0), f(s.get("t_init_s")),
            f(s.get("ms_per_step_median"), "%.0f"), f(s.get("ms_per_step_median_T"), "%.0f"), f(s.get("run_s"), "%.0f"),
            f(s.get("final_umax_mm")), f(s.get("final_umean_mm")), f(fin.get("tie_err_mean"), "%.3f"), f(fin.get("tie_err_max"), "%.3f"),
            f(fin.get("F_axial_mN"), "%.0f"), f(s.get("max_stretch_run")), f(s.get("min_vol_ratio_run"), "%.3f"),
            s.get("tie_impl") or "rssff", c.get("linear_solver"), c.get("cell_mm"),
            "PASS" if not failed else "FAIL: " + ",".join(failed) + ("" if v is not None else " (not evaluated)")))
        summ[t] = dict(run=dict(status=s["status"], converged=s["converged"], numeric_gates=gate, superseded=sup))
    # ---- canal hypotheses (spec 9, stage 2): sim-internal mechanics per flange depth; tip-to-serosa and flange-to-
    # corpus compare against the BT uterus and therefore stay SEALED until the S1 freeze.
    import numpy as np
    named = load(P["inputs"] + "/named.json") if os.path.exists(P["inputs"] + "/named.json") else {}
    i_ios = int(named.get("i_internal_os", 0))
    sw = [t for t in sorted(summ) if t.split("_")[0] in ("R1", "R2", "R3", "R4", "R5", "R6", "R7") and not summ[t]["run"]["superseded"]]
    if sw:
        L += ["", "## d_F sweep and P2 (spec 9 stage 2; sim-internal quantities; the landing metrics are SEALED)", "",
              "| run | d_F equiv mm | status | canal-end (fundal) displ. along rod mm | O_pre displ. along rod mm | cervical canal stretch (O_pre -> internal os) | max stretch | min V ratio | F_axial mN | below-flange sim / BT mm | admissible |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in sw:
            rd = P["runs"] + "/" + t
            if not os.path.exists(rd + "/final.npz"):
                continue
            z = np.load(rd + "/final.npz"); s = load(rd + "/summary.json")
            a_ = z["a"] / np.linalg.norm(z["a"]); dc = z["canal_final"] - z["canal_rest"]
            arc = lambda Q: float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())  # noqa: E731
            cx = arc(z["canal_final"][:i_ios + 1]) / arc(z["canal_rest"][:i_ios + 1]) if i_ios > 0 else float("nan")
            vp = os.path.join(P["eval"], t, "visible.json"); bf = load(vp)["below_flange_extent_mm"] if os.path.exists(vp) else None
            L.append("| %s | %.1f | %s | %.1f | %.1f | %.3f | %.2f | %.3f | %.0f | %s | %s |" % (
                t, s.get("d_F_equiv_mm", float("nan")), s["status"], float(dc[-1] @ a_), float(dc[0] @ a_), cx,
                s["max_stretch_run"], s["min_vol_ratio_run"], s["final"].get("F_axial_mN", float("nan")),
                ("%.1f / %.1f" % (bf["sim"], bf["BT"])) if bf else "-", not s["admissibility_flag"]))
    L += ["", "## Visible metrics (eval/<tag>/visible.json; E_app frame)", "",
          "| run | status | HR-CTV Dice/MSD/HD95 (sim) | HR-CTV (B0) | HR-CTV vol sim / BT cc | below-flange sim / BT | piercing | E_pelvic flange mm / axis deg |",
          "|---|---|---|---|---|---|---|---|"]
    for fp in sorted(glob.glob(P["eval"] + "/*/visible.json")):
        v = load(fp); t = v["tag"]; h = v["HR-CTV"]; n = v["numerics"]; bf = v["below_flange_extent_mm"]; ep = v.get("E_pelvic", {})
        L.append("| %s | %s | %.3f / %.2f / %.2f | %.3f / %.2f / %.2f | %.1f / %.1f | %.1f / %.1f | %.2f | %s / %s |" % (
            t, n["status"], h["sim"]["dice"], h["sim"]["msd"], h["sim"]["hd95"], h["B0"]["dice"], h["B0"]["msd"], h["B0"]["hd95"],
            h["vol_sim_cc"], h["vol_BT_cc"], bf["sim"], bf["BT"], v["piercing"]["frac_rod_outside_body_beyond_entry"],
            f(ep.get("flange_offset_mm"), "%.1f"), f(ep.get("axis_angle_deg"), "%.1f")))
        summ.setdefault(t, {})["visible"] = v
        if a.unseal:
            summ[t]["sealed"] = load(os.path.join(os.path.dirname(fp), "sealed_heldout.json"))
    if a.unseal:
        L += ["", "## Sealed (unsealed after S1 freeze)", "", "| run | uterus Dice/MSD/HD95 | B0 | tip-to-serosa sim/BT | beats B_HR |", "|---|---|---|---|---|"]
        for t, d in summ.items():
            if "sealed" not in d:
                continue
            s = d["sealed"]; u = s["uterus"]["sim"]; b0 = s["uterus"]["B0"]
            beats = u["dice"] > B["B_HR"]["dice"] and u["msd"] < B["B_HR"]["msd"] and u["hd95"] < B["B_HR"]["hd95"]
            L.append("| %s | %.3f / %.2f / %.2f | %.3f / %.2f / %.2f | %.1f / %.1f | %s |" % (
                t, u["dice"], u["msd"], u["hd95"], b0["dice"], b0["msd"], b0["hd95"], s["tip_to_serosa_mm"],
                s["BT_reference"]["tip_to_serosa_mm"], beats))
    else:
        L += ["", "Sealed held-out metrics (uterus, tip/canal landing) are NOT shown: frozen/S1.json does not exist yet."]
    with open(P["eval"] + "/report.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(P["eval"] + "/summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
