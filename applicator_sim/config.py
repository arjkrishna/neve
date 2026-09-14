"""applicator_sim: single source of defaults and paths (py3.8+ and py3.13; numpy only).

UNITS: length mm, mass kg, time s  =>  force mN, stress kPa, spring stiffness mN/mm,
Winkler (foundation) modulus mN/mm^3 (= kPa/mm).  Every key carries its unit.
Every physical parameter is tagged in PARAM_SOURCE as MEASURED (from the images),
ASSUMED (literature / modelling choice) or NUMERICAL (a solver device, not physics).
"""
import copy
import json
import os

# ----------------------------------------------------------------------------- paths
# Host paths come from environment variables; the fallback is ~/Downloads/{MRI_GYN, MRI_GYN_sim} (no machine-specific
# path is written into the code).
#   APPSIM_DATA    read-only image + label folder (preBT_MRI.nii, BT_MRI.nii, *_label_*.nii)
#   APPSIM_OUT     derived-data root (inputs/, runs/, eval/, validation/, calibration/, final/, figs/, logs/, prior/)
#   APPSIM_PRIOR   snapshot of the earlier workflow's outputs (default <APPSIM_OUT>/prior, see snapshot_prior.py)
#   APPSIM_SCRATCH temporary scratch (optional)
IN_CONTAINER = os.name != "nt" and os.path.isdir("/app") and os.path.isdir("/out")
_FALLBACK = dict(data=os.path.expanduser("~/Downloads/MRI_GYN"), out=os.path.expanduser("~/Downloads/MRI_GYN_sim"))


def _host():
    out = os.environ.get("APPSIM_OUT", _FALLBACK["out"]).replace("\\", "/")
    return dict(app=os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"),
                data=os.environ.get("APPSIM_DATA", _FALLBACK["data"]).replace("\\", "/"),     # READ-ONLY
                out=out,
                scratch=os.environ.get("APPSIM_SCRATCH", out + "/scratch").replace("\\", "/"),
                prior=os.environ.get("APPSIM_PRIOR", out + "/prior").replace("\\", "/"))


CONTAINER = dict(app="/app", data="/data", out="/out", scratch="/scratch", prior="/out/prior")
DOCKER_IMAGE = "eve-training-fixed"


def paths():
    p = dict(CONTAINER if IN_CONTAINER else _host())
    for k in ("inputs", "runs", "eval", "figs", "logs", "frozen"):
        p[k] = p["out"] + "/" + k
    return p


# ----------------------------------------------------------------------------- defaults
DEFAULTS = dict(
    # --- tissue (ASSUMED)
    E_kPa=30.0,                       # Young's modulus, literature 1-100 kPa for uterus/cervix
    nu=0.45,                          # Poisson ratio; sensitivity 0.40 / 0.49
    # --- discretisation (NUMERICAL)
    cell_mm=4.0,                      # SparseGridTopology cellWidth; 5 for calibration, 3 for convergence
    mapped_children=True,             # BarycentricMapping read-only children (surfaces, canal) inside SOFA
    fe_impl="hexa",                   # "hexa": HexahedronFEMForceField -- every boundary cell fully stiff: the mechanical
                                      # body is the voxel hull (MEASURED 119 cc at 4 mm, 111 cc at 3 mm, for an 83 cc
                                      # tissue surface; logged per run as volumes_cc) | "nonuniform" (EXPERIMENTAL, not
                                      # used for results): NonUniformHexahedronFEMForceFieldAndMass, stiffness condensed
                                      # from nb_virtual_finer_levels finer levels (~92 cc at 4 mm).  MEASURED to fail
                                      # (runs F0 first attempt, D2, D3): a nearly empty fundal boundary cell collapses
                                      # (volume ratio -> 0.19) because the Winkler supports still act on grid nodes up to
                                      # one cell outside the tissue; it needs supports on the true surface (not built)
    nb_virtual_finer_levels=2,        # 4 mm cells -> stiffness integrated on ~1 mm virtual sub-cells
    # --- canal tie discretisation (NUMERICAL; finisher fix of the over-constrained ties)
    tie_group="cell",                 # "cell": the non-tip canal points that share a rest cell are tied as ONE point (their
                                      # mean; same 8 nodes, averaged trilinear weights) -> at most one 2-D lateral
                                      # constraint per cell | "point": every canal point tied (stage 1-3 behaviour)
    canal_fillet_mm=5.0,              # smooth the kink where the synthetic straight O_pre->L_end segment meets the
                                      # centre line: points within +/- this arclength of L_end are cosine-blended with a
                                      # Gaussian average (sigma = fillet/2); 0 = off (stage 1-3 behaviour)
    lam_sat_frac=0.9,                 # numeric gate: any AL multiplier >= this fraction of tie_al_clip_mm fails the run
    # --- supports (ASSUMED)
    kappa_f_mN_per_mm3=0.01,          # Winkler foundation on all boundary nodes (LHS [0.002, 0.05] log)
    kappa_lig_mN_per_mm3=0.1,         # paracervical band (LHS [0.02, 2.0] log)
    lig_h_min_mm=0.0, lig_h_max_mm=20.0,   # band height along a0 above L_end
    lig_radial_min_mm=8.0,            # band only outside this radius from the a0 line
    found_anchor="rest",              # "rest": supports anchored at the preBT rest positions (stage 1-3) |
                                      # "device_rigid": anchors carried by the rigid placement that puts the rest canal on
                                      # the final tandem axis with its end at the tip-push target (a0 -> a by the minimal
                                      # rotation about L_end), ramped with the insertion progress u.  The supports then
                                      # model local restraint around a device-placed organ instead of fixing it to its
                                      # preBT position in the registration frame (finisher fix, physics review)
    # --- applicator coupling (NUMERICAL penalties)
    k_tie_mN_per_mm=500.0,            # per engaged canal point
    tie_impl="pyff",                  # "pyff": anisotropic DNT ForceField (lateral ties truly frictionless inside the
                                      # implicit solve; matrix-free -> CG only; ties.py)  | "rssff": isotropic
                                      # RestShapeSprings DNT (LDL-safe; axial slide released only by re-projection)
    tie_ramp_steps=3,                 # stiffness 1/3, 2/3, 1 after engagement
    lateral_cap_mm_per_step=1.0,
    axial_cap_mm_per_step=1.0,
    tip_push_n_pts=3,
    tip_push_margin_mm=0.5,           # canal end kept this far above the rod tip
    tip_engage_tol_mm=0.5,
    shaft_len_mm=26.0,                # MEASURED straight shaft below the flange
    tie_al="hold",                    # augmented-Lagrangian target correction: "off" | "hold" | "always"
    tie_al_tip=True,                  # AL also on the tip push (unilateral, multiplier >= 0); False = penalty tip
    tie_al_clip_mm=6.0,
    tie_al_step_mm=0.5,               # max multiplier change per step (mm) -- no kick at hold entry
    tie_al_relax=1.0,                 # AL update relaxation factor (1 = plain AL)
    tip_patch_radius_mm=0.0,          # NUMERICAL: >0 spreads the tip push over body nodes within this radius
                                      # (blunt-indenter regularisation of the point load); 0 = spec (one cell)
    k_s_mN_per_mm=300.0,              # ovoid-sphere penalty (M2)
    contact_omega=0.7,
    contact_candidate_radius_mm=35.0,
    contact_candidates="radius",      # "radius": body nodes within contact_candidate_radius_mm of the final flange;
                                      # "all": every body node (needed when the spheres travel with the device)
    contact_impl="rssff_nodes",       # "rssff_nodes" (spec 5.3: RSSFF on grid nodes, target = omega-relaxed sphere
                                      # projection; measured in stage 2 to tunnel / invert elements next to the
                                      # ~5 mm top spheres at 4 mm cells) | "pyff_surface" (stage 2: frictionless
                                      # penalty on the TRUE body-surface vertices, force k (-phi) n along the normal
                                      # of a smooth sphere-union SDF, distributed to the 8 cell nodes with the DNT
                                      # trilinear algebra inside the NodeTieFF linearisation; needs tie_impl='pyff')
    contact_blend_mm=0.75,            # NUMERICAL soft-min blend of the sphere-union SDF (pyff_surface)
    contact_depth_cap_mm=2.0,         # NUMERICAL: penetration used in the penalty force is capped per step
    # --- geometry choices
    pose="P2",                        # P1 | P2 | P2u | SWEEP | any key of inputs/poses_align.json (PB bone frame, PBg)
    d_F_mm=None,                      # flange depth below L_end for P1/SWEEP (None -> d_F_pred)
    delta_fund_mm=0.0,                # fundal canal extension (sensitivity 10)
    sph_r_scale=1.0,                  # ovoid/cap sphere radius factor (device size tolerance; 1 = MEASURED pack)
    sph_dz_mm=0.0,                    # ovoid/cap sphere shift along the applicator z axis (mm; 0 = MEASURED pack)
    m2=False,                         # ovoid/cap contact (M2a)
    m2_mode="inflate",                # "inflate" (spec: radii ramp over phaseD_steps at the final pose) |
                                      # "travel" (stage 2: full-size spheres move WITH the device along the whole
                                      # insertion path and push the portio/fornices ahead of them; no phase D).
                                      # In "travel" a canal point is released from its tie once it lies inside a
                                      # sphere (the device body occupies that space).
    body_mesh="pre_body",             # "pre_body" (uterus|HR-CTV|IUcanal) | "pre_bodyvag" (+ vagina, M2b)
    kappa_vag_floor_mN_per_mm3=1.0,   # ASSUMED levator / perineal support: Winkler on vagina boundary nodes below
    vag_floor_h_mm=25.0,              # (vaginal bottom + this, along a0) -- only with body_mesh='pre_bodyvag'
    # --- loading protocol
    step_mm=1.0,                      # max tip travel per phase-T step
    max_rot_deg_per_step=0.5,
    phaseA_steps=5, phaseA_start_mm=5.0,
    phaseD_steps=10,
    hold_max_steps=60,
    hold_dx_tol_mm=0.05,
    hold_rule="v2",                   # "v2" (finisher): converged only when, for hold_consec_steps consecutive steps,
                                      # max |dx| < hold_dx_tol_mm, the max nodal force residual < hold_force_tol_mN, the AL
                                      # multipliers moved < al_dlam_tol_mm, the tie/penetration criteria hold, AND the
                                      # geometric-series drift estimate dx*rho/(1-rho) < hold_drift_tol_mm |
                                      # "v1": first hold step with dx < tol (stage 1-3; a stopping-rule artefact)
    hold_consec_steps=3,
    hold_force_tol_mN=1.0,            # max |nodal force residual| (dofs.force after the solve; mN)
    hold_drift_tol_mm=0.05,
    al_dlam_tol_mm=0.01,
    al_freeze_after_hold_steps=25,    # AL multipliers frozen after this many hold steps (targets stop moving)
    hold_cg_iterations=800,           # CG cap and tolerance during the hold (the insertion keeps cg_iterations/tolerance)
    hold_cg_tolerance=1e-10,
    tie_err_mean_tol_mm=0.3, tie_err_max_tol_mm=0.6,
    pen_tol_mm=0.5,
    # --- solver (NUMERICAL)
    # The pyff ties are matrix-free, so CG is required.  During the insertion CG is capped at cg_iterations (it hits
    # the cap: MEASURED by the physics review); every step re-solves the full residual, and the hold (v2 rule) then
    # runs CG with hold_cg_iterations / hold_cg_tolerance until the force residual criterion is met.  The CG
    # iteration count of the last Newton solve is logged per step (cg_its, parsed from the CGLinearSolver 'graph').
    linear_solver="CGLinearSolver",   # SparseLDLSolver | EigenSimplicialLDLT | CGLinearSolver
    cg_iterations=100, cg_tolerance=1e-6, cg_threshold=1e-12,
    newton_iterations=1,              # per insertion step (phases A/T/D)
    hold_newton_iterations=3,         # per hold step (phase H): full nonlinear solve while the ties settle
    abs_corr_tol_mm=1e-3, rel_corr_tol=1e-4,
    abs_res_tol_mN=1e-3, rel_res_tol=1e-5,
    # --- guards
    min_vol_ratio_abort=0.2,
    residual_growth_abort_steps=3,
    max_steps=260,
    wall_limit_s=540.0,
    export_every=0,                   # >0: also dump dofs every N steps
    frame_every=0,                    # >0: every N steps (and at the last step) write the deformed mapped surfaces and the
                                      # applicator state to runs/<tag>/frames/ (step_XXXX_{body,uterus,hrctv}.obj,
                                      # step_XXXX_device.json, index.json) for the host renderer animate.py; 0 = off
)

PARAM_SOURCE = dict(
    E_kPa="ASSUMED", nu="ASSUMED", kappa_f_mN_per_mm3="ASSUMED", kappa_lig_mN_per_mm3="ASSUMED",
    lig_h_min_mm="ASSUMED", lig_h_max_mm="ASSUMED", lig_radial_min_mm="ASSUMED",
    k_tie_mN_per_mm="NUMERICAL", k_s_mN_per_mm="NUMERICAL", contact_omega="NUMERICAL", cell_mm="NUMERICAL",
    shaft_len_mm="MEASURED (BT applicator label)", L_iu_mm="MEASURED (BT applicator label)",
    r_tandem_mm="MEASURED (BT applicator label)", ovoid_spheres="MEASURED (BT ovoid label sphere pack)",
    d_F_mm="ASSUMED / swept (not identifiable from preBT alone)", delta_fund_mm="ASSUMED (0; sensitivity 10)",
)


# Key values that reproduce the stage 1-3 model (code frozen in validation/freeze.json).  A stage 1-3 cfg.json lacks
# the finisher keys, so loading it with plain DEFAULTS would silently switch the model: use {"preset": "legacy_v1"}.
# (Bit-identical reproduction of the frozen runs needs the frozen code itself; see MRI_GYN_sim/final/A1_repro.json.)
PRESETS = dict(
    legacy_v1=dict(fe_impl="hexa", tie_group="point", canal_fillet_mm=0.0, found_anchor="rest", hold_rule="v1",
                   hold_max_steps=25, hold_cg_iterations=None, hold_cg_tolerance=None, phaseA_null_ties=False),
)
DEFAULTS["phaseA_null_ties"] = True    # phase A engages every tie with target = current canal (non-zero stiffness, zero
                                       # force): umax must stay < 1e-3 mm (a real null test of the coupling)


def load_cfg(overrides=None):
    """DEFAULTS (optionally a preset: overrides['preset']) updated by a dict or a JSON file path.
    Unknown keys are kept but reported."""
    cfg = copy.deepcopy(DEFAULTS)
    if overrides is None:
        return cfg
    if isinstance(overrides, str):
        with open(overrides) as fh:
            overrides = json.load(fh)
    pre = overrides.get("preset")
    if pre:
        cfg.update(PRESETS[pre])
    unknown = [k for k in overrides if k not in DEFAULTS and not k.startswith("_") and k not in ("tag", "preset")]
    cfg.update(overrides)
    cfg["_unknown_keys"] = unknown
    return cfg
