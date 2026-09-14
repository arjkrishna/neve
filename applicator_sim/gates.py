"""Numeric gates of a run, in ONE place (py3.8 and py3.13; no numpy needed).  Used by run_insertion (container),
evaluate / calibrate / report / final_eval (host).  Units mm, mN.

Two groups:
  preregistered  the numeric criteria copied from the design spec before any unsealing (README):
                 phase-A control, converged hold, min hexa volume ratio > 0.2, finite, tie residual mean < 0.3 and
                 max < 0.6 mm, penetration < 0.5 mm, piercing = 0 (+ the stretch/volume admissibility flag)
  added_v2       added by the finisher after the physics review (reported separately; a run that passes the
                 pre-registered gates but fails these is NOT a clean result):
                   lam_not_saturated   no augmented-Lagrangian multiplier >= lam_sat_frac * clip (over-constrained ties)
                   force_residual      max nodal force residual < hold_force_tol_mN at the end (v2 hold rule)
                   no_corner_inversion no negative corner tetrahedron of any hexa at the end of the run
                   rod_in_lumen        rod length beyond the tied canal end (inside tissue, with no contact) < 1 mm
  physics        strict admissibility for near-incompressible tissue: every hexa volume ratio in [0.85, 1.15]
                 (reported; the linear compressible law is known to fail it where the ovoids compress the portio)
"""


def _lt(x, tol):
    return x is not None and x < tol


def numeric_gates(s, pierce=None, cfg=None):
    """s: summary.json dict of a run; pierce: fraction of rod samples outside the deformed body beyond its entry
    (host-side, evaluate.enclosed) or None if not computed.  Returns dict(preregistered={..}, added_v2={..},
    physics={..}, pass_preregistered, pass_all)."""
    cfg = cfg or s.get("cfg", {})
    fin = s.get("final", {}) or {}
    pre = dict(
        phaseA=bool(s.get("phaseA_control_ok")),
        converged=bool(s.get("converged")),
        vol_ratio_gt_0p2=_lt(0.2, s.get("min_vol_ratio_run") if s.get("min_vol_ratio_run") is not None else -1),
        finite=not s.get("nonfinite", True),
        tie=_lt(fin.get("tie_err_mean"), 0.3) and _lt(fin.get("tie_err_max"), 0.6),
        penetration=_lt(fin.get("pen_max") if fin.get("pen_max") is not None else 0.0, 0.5),
        admissible=not s.get("admissibility_flag", True),
    )
    if pierce is not None:
        pre["piercing_0"] = bool(pierce == 0.0)
    clip = float(cfg.get("tie_al_clip_mm", 6.0)); frac = float(cfg.get("lam_sat_frac", 0.9))
    lam = fin.get("lam_max_mm")
    add = dict(
        lam_not_saturated=(lam is not None and lam < frac * clip),
        force_residual=_lt(fin.get("f_res_max_mN"), float(cfg.get("hold_force_tol_mN", 1.0))),
        no_corner_inversion=(fin.get("n_neg_corner_tet") == 0),
        rod_in_lumen=_lt(fin.get("rod_beyond_canal_end_mm"), 1.0),
    )
    phys = dict(vol_ratio_in_0p85_1p15=(fin.get("n_vol_out_0p85_1p15") == 0))
    return dict(preregistered=pre, added_v2=add, physics=phys,
                pass_preregistered=bool(all(pre.values())),
                pass_added_v2=bool(all(add.values())),
                pass_physics=bool(all(phys.values())),
                pass_all=bool(all(pre.values()) and all(add.values())))
