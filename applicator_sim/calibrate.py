"""Stage-3 CALIBRATION (HOST, python 3.13).  Units: mm, kg, s -> kPa, mN, mN/mm, mN/mm^3; Dice [-]; deg.

Serial parameter sweep of the M2a scene (tandem + ovoids travelling with the device, frictionless surface contact)
in the BONE frame (pose PB: the BT device mapped into preBT by the bony-pelvis MI registration, which uses NO organ
label).  The objective uses ONLY uterus + vagina + canal/landing metrics; the HR-CTV is HELD OUT: it is never
computed during the sweep and is scored only after best.json has been written and hashed (`heldout`).

    python calibrate.py plan            # pre-declare: calibration/plan.json (+ .sha256) and the LHS job list
    python calibrate.py run lhs         # run the LHS jobs serially (run_docker.sh + calib_run.py), then score them
    python calibrate.py refine          # plan the refinement box around the best LHS sample (rule fixed in plan.json)
    python calibrate.py run refine
    python calibrate.py freeze          # best.json (+ best.sha256): argmin J over gate-passing LHS+refine runs
    python calibrate.py run controls    # controls at the frozen best (scaling identity, penalties, mesh, HR frame)
    python calibrate.py heldout         # AFTER the freeze: HR-CTV for every run + rigid baselines/controls
    python calibrate.py analyze         # sensitivity (SRRC with bootstrap CIs), sweep.csv, sensitivity.png
    python calibrate.py score TAG ...   # (re)score runs (uterus/vagina/canal only)

Objective (pre-declared in plan.json, all terms in mm):
    J = MSD_uterus + 0.5 * MSD_vagina+device(ovoid slab) + 0.5 * C_canal
    C_canal = 0.5 * ( |tip_to_serosa_sim - tip_to_serosa_BT| + |internal_os_landing_sim - flange_to_corpus_BT| )
  - MSD = symmetric mean surface distance (evaluate.surf_dists, verbatim registration/common.py), deformed preBT
    surface -> BONE rigid map -> vtkPolyDataToImageStencil on the BT grid (the E0-validated route).
  - vagina: the preBT vagina is NOT an FEM body (M2b was dropped in stage 2).  It is carried by a pre-declared
    KINEMATIC follower of the body's displacement field: inverse-distance (power 2) average of the 8 nearest FEM node
    displacements, tapered linearly along a0 from 1 at the lowest body point to 0 at the vaginal bottom (pelvic-floor
    anchoring, ASSUMED).  It is scored in the ovoid slab 0-18 mm below the BT vaginal apex (deeper = packing) with the
    MEASURED device sphere pack added to both masks, because the BT vagina label contains the ovoids.
  - landing: tip-to-serosa along the BT tandem (sim uterus mask) and the deformed internal-os canal point's axial
    position vs the BT flange-to-uterus-entry distance.
Candidates must pass the stage-2 numeric gates (phase-A control, converged hold, min hexa volume ratio > 0.2, finite,
tie residual mean < 0.3 / max < 0.6 mm, penetration < 0.5 mm, rod never outside the body, stretch admissible).

E (Young's modulus) is NOT swept: with every load displacement-controlled (kinematic device, AL-enforced ties,
penalty contact) and force-type supports, the deformed shape depends only on kappa/E, k/E and nu.  The control
C3_K_scale3 (E, kappa_f, kappa_lig, k_tie, k_s all x3) tests this identity directly.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402
import geom  # noqa: E402

P = config.paths()
CAL = P["out"] + "/calibration"
SCORES = CAL + "/scores"
INP = P["inputs"]
VAL = P["out"] + "/validation"
PREFIX = "C3_"

# ---------------------------------------------------------------------------------------------- pre-declared design
BASE_CFG = dict(pose="PB", m2=True, m2_mode="travel", contact_impl="pyff_surface", k_s_mN_per_mm=600.0,
                hold_max_steps=40, cell_mm=4.0, E_kPa=30.0)      # = the stage-2 V4 scene (only admissible bone run)
# name, cfg key, lo, hi, scale, unit, meaning
PARAMS = [
    ("kappa_f", "kappa_f_mN_per_mm3", 5e-4, 2e-2, "log", "mN/mm^3",
     "Winkler foundation on all boundary nodes (bladder/bowel/peritoneum lumped; boundary stiffness)"),
    ("kappa_lig", "kappa_lig_mN_per_mm3", 2e-3, 0.5, "log", "mN/mm^3",
     "extra paracervical (cardinal/uterosacral) band stiffness"),
    ("lig_h_max", "lig_h_max_mm", 10.0, 35.0, "lin", "mm", "height of the paracervical band above L_end (fixed-region extent)"),
    ("nu", "nu", 0.40, 0.49, "lin", "-", "Poisson ratio (uterus/cervix, one material)"),
    ("tip_margin", "tip_push_margin_mm", -8.0, 0.5, "lin", "mm",
     "canal end held at L_iu + margin: +0.5 = tip pushes the labelled canal end; negative = rod tip may overrun the "
     "labelled canal end by |margin| before pushing (fundal under-labelling / own track); <= -4 ~ no push"),
    ("sph_r_scale", "sph_r_scale", 0.85, 1.10, "lin", "-", "ovoid/cap sphere radius factor (size tolerance)"),
    ("sph_dz", "sph_dz_mm", -3.0, 3.0, "lin", "mm", "ovoid/cap sphere shift along the tandem axis (position tolerance)"),
]
W_VAG, W_CANAL = 0.5, 0.5
N_LHS, N_REF, REF_HALFWIDTH = 20, 5, 0.2
IDENT_TOL_MM = 0.3
CAL_CODE = ("calibrate.py", "calib_run.py")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    return np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(d)], 1)


def maximin_lhs(n, d, seed, tries=200):
    best, bd = None, -1.0
    for k in range(tries):
        u = lhs(n, d, seed * 100003 + k)
        dm = np.linalg.norm(u[:, None] - u[None], axis=2)[np.triu_indices(n, 1)].min()
        if dm > bd:
            best, bd = u, dm
    return best, float(bd)


def u_to_val(u):
    out = {}
    for (nm, key, lo, hi, sc, _, _), ui in zip(PARAMS, u):
        v = 10 ** (np.log10(lo) + ui * (np.log10(hi) - np.log10(lo))) if sc == "log" else lo + ui * (hi - lo)
        out[key] = float(round(v, 6))
    return out


def val_to_u(cfg):
    u = []
    for nm, key, lo, hi, sc, _, _ in PARAMS:
        v = float(cfg[key])
        u.append((np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if sc == "log" else (v - lo) / (hi - lo))
    return np.array(u)


def job(tag, pv, phase, extra=None):
    cfg = dict(BASE_CFG); cfg.update(pv); cfg.update(extra or {})
    return dict(tag=PREFIX + tag, cfg=cfg, phase=phase)


# ---------------------------------------------------------------------------------------------- plan
def cmd_plan(force=False):
    os.makedirs(SCORES, exist_ok=True)
    fn = CAL + "/plan.json"
    if os.path.exists(fn) and not force:
        sys.exit("plan.json exists (pre-declared); refusing to overwrite without --force")
    u, dmin = maximin_lhs(N_LHS, len(PARAMS), 0)
    jobs = [job("L%02d" % i, u_to_val(ui), "lhs") for i, ui in enumerate(u)]
    plan = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"), stage="3 (calibration)",
        units="mm, kg, s -> kPa, mN, mN/mm, mN/mm^3; Dice [-]",
        honesty=("Written BEFORE any calibration run. Uterus metrics of S0 runs were already read in stage 2, so the "
                 "uterus is in-sample here by design. HR-CTV is HELD OUT: calibrate.py never computes it before "
                 "best.json + best.sha256 exist; `heldout` refuses otherwise."),
        scene=dict(base_cfg=BASE_CFG, frame="BONE (pose PB; bony-pelvis MI, no organ label)",
                   why_bone=("HR-CTV can only be a held-out check in a frame that does not use it; the HR frame is the "
                             "HR-CTV-only ICP optimum, so HR-CTV is not held out there.")),
        params=[dict(name=n, cfg_key=k, lo=lo, hi=hi, scale=s, unit=un, meaning=m) for n, k, lo, hi, s, un, m in PARAMS],
        fixed=dict(E_kPa="30 (ASSUMED; not identifiable from shape, see scaling control)", k_tie_mN_per_mm=500.0,
                   k_s_mN_per_mm=600.0, cell_mm=4.0, delta_fund_mm=0.0, step_mm=1.0),
        not_swept={
            "E_kPa (uterus/cervix)": "not identifiable from geometry with displacement-driven loading (only kappa/E, k/E, nu "
                                     "enter the shape); tested by control C3_K_scale3",
            "E vagina": "the vagina is not an FEM body in this model (M2b dropped in stage 2): no vaginal modulus exists",
            "E cervix vs corpus": "single material; HexahedronFEMForceField takes one modulus per force field (M4 item)",
            "k_tie_mN_per_mm, k_s_mN_per_mm": "numerical penalties: one-at-a-time controls at the best point",
            "delta_fund_mm": "covered by the continuous tip_margin parameter (overrun of the labelled canal end)",
        },
        design=dict(lhs=dict(n=N_LHS, seed=0, maximin_tries=200, min_pair_dist_unit_cube=round(dmin, 4)),
                    refinement=dict(n=N_REF, seed=1, rule="maximin LHS in the unit-cube box best_LHS_u +/- %.2f (clipped), "
                                                          "best = gate-passing LHS run with minimal J" % REF_HALFWIDTH),
                    controls=["scale3: E, kappa_f, kappa_lig, k_tie, k_s all x3 (shape-invariance identity)",
                              "ktie250: k_tie x0.5", "ks1200: k_s x2", "cell3: 3 mm cells (mesh)",
                              "HR: best parameters in the HR frame (pose P2; uterus transfer check; HR-CTV NOT held out there)"],
                    budget="<= 30 SOFA runs, one container at a time, --cpus 2; <= ~90 min wall"),
        objective=dict(
            J="MSD_uterus + %.1f * MSD_vag+device_slab + %.1f * C_canal  [mm]" % (W_VAG, W_CANAL),
            C_canal="0.5*(|tip_to_serosa_sim - BT| + |internal_os_landing_sim - flange_to_corpus_BT|) [mm]",
            weights_rationale=("uterus MSD is the primary term; the vagina slab score is dominated by the device and the "
                               "vagina is only a kinematic follower (weight 0.5); both landing terms are also uterus-derived "
                               "(partly redundant with MSD_uterus; weight 0.5)"),
            vagina_follower=("IDW (power 2) of the 8 nearest FEM-node displacements, times a linear taper along a0: 1 at "
                             "and above the lowest pre_body point, 0 at the vaginal bottom (ASSUMED pelvic-floor anchor)"),
            gates="stage-2 numeric gates; failing runs are reported but never selected",
            identifiability="runs with J <= J_best + %.1f mm form the identifiability set" % IDENT_TOL_MM,
            references_not_candidates=["V4_PB_M2_soft (stage-2 S0-soft)", "X4_PB_M2soft_latonly (stage-2 post hoc)",
                                       "rigid BONE baseline (undeformed preBT, same frame)"]),
        lhs_jobs=jobs)
    txt = json.dumps(plan, indent=1)
    open(fn, "w").write(txt)
    open(CAL + "/plan.sha256", "w").write(hashlib.sha256(txt.encode()).hexdigest() + "  plan.json\n")
    json.dump(jobs, open(CAL + "/jobs_lhs.json", "w"), indent=1)
    print("plan: %d LHS jobs (maximin min pair distance %.3f in the unit cube) -> %s" % (len(jobs), dmin, fn))
    for j in jobs:
        print(" ", j["tag"], {k: j["cfg"][k] for _, k, *_ in PARAMS})


# ---------------------------------------------------------------------------------------------- running
def _bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if os.path.exists(c):
            return c
    return shutil.which("bash")


def run_done(tag):
    p = os.path.join(P["runs"], tag, "summary.json")
    return os.path.exists(p)


def run_jobs(jobs, name, chunk=8):
    pending = [j for j in jobs if not run_done(j["tag"])]
    timing = json.load(open(CAL + "/timing.json")) if os.path.exists(CAL + "/timing.json") else []
    k = 0
    while pending:
        bn = "batch_%s%s_%d.json" % (PREFIX, name, k)
        json.dump([dict(tag=j["tag"], cfg=j["cfg"]) for j in pending[:chunk]], open(os.path.join(P["runs"], bn), "w"), indent=1)
        t0 = time.time()
        rc = subprocess.call([_bash(), os.path.join(HERE, "run_docker.sh"), "%s%s_%d" % (PREFIX, name, k),
                              "calib_run.py", "--batch", "/out/runs/" + bn])
        wall = time.time() - t0
        before = len(pending)
        pending = [j for j in pending if not run_done(j["tag"])]
        timing.append(dict(batch=bn, rc=rc, wall_s=round(wall, 1), n_done=before - len(pending),
                           finished=time.strftime("%H:%M:%S")))
        json.dump(timing, open(CAL + "/timing.json", "w"), indent=1)
        print("[calibrate] %s rc=%s wall %.0f s, %d done, %d pending" % (bn, rc, wall, before - len(pending), len(pending)), flush=True)
        if before == len(pending):
            print("[calibrate] no progress; stopping (pending: %s)" % [j["tag"] for j in pending]); break
        k += 1
    return pending


# ---------------------------------------------------------------------------------------------- scoring (NO HR-CTV)
class Scorer:
    def __init__(self):
        import evaluate as ev
        self.ev = ev
        self.bt = ev.BT()        # loads the BT labels; the HR-CTV label is NOT used by any scoring below
        self.al = json.load(open(VAL + "/alignment.json"))
        self.named = json.load(open(INP + "/named.json"))
        po = json.load(open(INP + "/poses.json"))
        self.L_end, self.a0 = np.array(po["L_end"]), geom.unit(np.array(po["a0"]))
        self.Vv0, self.Fv = geom.read_obj(INP + "/pre_vagina.obj")
        self.Vu0, self.Fu = geom.read_obj(INP + "/pre_uterus.obj")
        Vb0, _ = geom.read_obj(INP + "/pre_body.obj")
        hv = (self.Vv0 - self.L_end) @ self.a0
        self.h_vbot, self.h_bbot = float(hv.min()), float(((Vb0 - self.L_end) @ self.a0).min())
        self.taper = np.clip((hv - self.h_vbot) / (self.h_bbot - self.h_vbot), 0.0, 1.0)
        bt = self.bt
        self.ov = ev.spheres_mask(bt.F + np.array([q["center_app_mm"] for q in bt.sph]) @ bt.R,
                                  [q["r_mm"] for q in bt.sph], bt.shape, bt.aff)      # MEASURED device, always
        zb = (ev.mask_world(bt.lab["vagina"], bt.aff) - bt.F) @ bt.a
        self.apex_BT = float(zb.max())
        g = np.indices(bt.shape).reshape(3, -1).T @ bt.aff[:3, :3].T + bt.aff[:3, 3]
        zz = ((g - bt.F) @ bt.a).reshape(bt.shape)
        self.slab = (zz <= self.apex_BT) & (zz >= self.apex_BT - 18.0)
        self.t2s_BT = ev.exit_dist(bt.lab["uterus"], bt.aff, bt.tip, bt.a)
        self.f2c_BT = ev.entry_dist(bt.lab["uterus"], bt.aff, bt.F, bt.a)
        self.cache = {}

    def frame(self, pose):
        fr = self.ev.FRAME_OF_POSE.get(pose)
        f = self.al["frames"][fr]
        return fr, np.array(f["R_BT_to_pre"]), np.array(f["t_BT_to_pre"])

    def follower(self, z):
        from scipy.spatial import cKDTree
        X0, U = z["X0"], z["X"] - z["X0"]
        d, idx = cKDTree(X0).query(self.Vv0, k=8)
        w = 1.0 / np.maximum(d, 1e-6) ** 2; w /= w.sum(1, keepdims=True)
        return self.Vv0 + self.taper[:, None] * np.einsum("nk,nkj->nj", w, U[idx])

    def score_masks(self, mu, mv, canal_bt):
        ev, bt = self.ev, self.bt
        ut = ev.seg_metrics(mu, bt.lab["uterus"], bt.sp, bt.vv)
        ref_v = bt.lab["vagina"]
        vds = ev.seg_metrics((mv | self.ov) & self.slab, ref_v & self.slab, bt.sp, bt.vv)
        vs = ev.seg_metrics(mv & self.slab, ref_v & self.slab, bt.sp, bt.vv)
        apex = float(((ev.mask_world(mv, bt.aff) - bt.F) @ bt.a).max()) if mv.any() else float("nan")
        i_le, i_ios = int(self.named["i_L_end"]), int(self.named["i_internal_os"])
        cm = ev.canal_metrics(canal_bt, bt, i_le, i_ios)
        t2s = ev.exit_dist(mu, bt.aff, bt.tip, bt.a); f2c = ev.entry_dist(mu, bt.aff, bt.F, bt.a)
        ios = cm["internal_os_along_BT_axis_mm"]
        C = 0.5 * (abs(t2s - self.t2s_BT) + abs(ios - self.f2c_BT))
        J = ut["msd"] + W_VAG * vds["msd"] + W_CANAL * C
        return dict(J=round(J, 4), C_canal=round(C, 4), uterus=ut, vagina_device_slab=vds, vagina_slab=vs,
                    vagina_apex_along_BT_axis_mm=round(apex, 2), vagina_apex_err_mm=round(apex - self.apex_BT, 2),
                    tip_to_serosa_mm=round(t2s, 2), tip_to_serosa_err_mm=round(t2s - self.t2s_BT, 2),
                    flange_to_corpus_mm=round(f2c, 2), flange_to_corpus_err_mm=round(f2c - self.f2c_BT, 2),
                    internal_os_landing_mm=round(ios, 2), internal_os_landing_err_mm=round(ios - self.f2c_BT, 2),
                    canal=cm)

    def baseline(self, frame_name):
        if frame_name in self.cache:
            return self.cache[frame_name]
        ev, bt = self.ev, self.bt
        f = self.al["frames"][frame_name]; R, t = np.array(f["R_BT_to_pre"]), np.array(f["t_BT_to_pre"])
        to_bt = lambda Y: (np.asarray(Y) - t) @ R  # noqa: E731
        mu = ev.voxelize(to_bt(self.Vu0), self.Fu, bt.shape, bt.aff)
        mv = ev.voxelize(to_bt(self.Vv0), self.Fv, bt.shape, bt.aff)
        canal = np.load(INP + "/canal.npz")["pts"]
        rec = self.score_masks(mu, mv, to_bt(canal)); rec.update(tag="rigid_%s" % frame_name, phase="baseline", frame=frame_name)
        self.cache[frame_name] = rec
        return rec

    def score(self, tag, phase=None):
        ev, bt = self.ev, self.bt
        rd = os.path.join(P["runs"], tag); s = json.load(open(rd + "/summary.json"))
        cfg = s["cfg"]; fr, R, t = self.frame(cfg["pose"])
        to_bt = lambda Y: (np.asarray(Y) - t) @ R  # noqa: E731
        rec = dict(tag=tag, phase=phase, frame=fr, cfg={k: cfg.get(k) for k in
                   ["pose", "E_kPa", "k_tie_mN_per_mm", "k_s_mN_per_mm", "cell_mm"] + [p[1] for p in PARAMS]},
                   status=s["status"], run_s=s.get("total_s"), ms_per_step_median=s.get("ms_per_step_median"),
                   n_steps=s.get("n_steps"), n_nodes=s.get("n_nodes"))
        fin = s.get("final", {})
        rec["numerics"] = dict(max_stretch=s.get("max_stretch_run"), min_vol_ratio=s.get("min_vol_ratio_run"),
                               tie_err_mean=fin.get("tie_err_mean"), tie_err_max=fin.get("tie_err_max"),
                               pen_max=fin.get("pen_max"), F_axial_mN=fin.get("F_axial_mN"),
                               F_ovoid_axial_mN=fin.get("F_ovoid_axial_mN"), final_umax_mm=s.get("final_umax_mm"))
        if not os.path.exists(rd + "/final.npz") or not os.path.exists(rd + "/surf_uterus.obj"):
            rec.update(gates_pass=False, J=None, note="no final state exported")
            return rec
        z = np.load(rd + "/final.npz")
        Fs, a_s = z["F"], geom.unit(z["a"])
        Vb, Fb = geom.read_obj(rd + "/surf_body.obj")
        ins = ev.enclosed(Fs + np.outer(np.arange(0.0, float(z["L_iu"]) + 1e-9, 1.0), a_s), Vb, Fb)
        first = int(np.argmax(ins)) if ins.any() else len(ins)
        pierce = float((~ins[first:]).mean()) if first < len(ins) else 1.0
        import gates as gates_mod        # the single gate definition (gates.py); stage-3 scores used an inline copy
        gg = gates_mod.numeric_gates(s, pierce=pierce)
        gates = gg["preregistered"]
        rec.update(gates=gates, gates_pass=gg["pass_preregistered"], gates_added_v2=gg["added_v2"], piercing_frac=pierce)
        Vd, Fd = geom.read_obj(rd + "/surf_uterus.obj")
        mu = ev.voxelize(to_bt(Vd), Fd, bt.shape, bt.aff)
        Vv = self.follower(z)
        mv = ev.voxelize(to_bt(Vv), self.Fv, bt.shape, bt.aff)
        rec.update(self.score_masks(mu, mv, to_bt(z["canal_final"])))
        rec["uterus_mesh_volume_change_frac"] = round(geom.mesh_volume(Vd, Fd) / geom.mesh_volume(self.Vu0, self.Fu) - 1.0, 4)
        rec["baseline_same_frame"] = {k: self.baseline(fr)[k] for k in ("J", "uterus", "vagina_device_slab", "C_canal")}
        return rec


def phase_of(tag):
    t = tag[len(PREFIX):] if tag.startswith(PREFIX) else tag
    return {"L": "lhs", "R": "refine", "K": "control"}.get(t[:1], "reference") if tag.startswith(PREFIX) else "reference"


def cmd_score(tags, sc=None):
    sc = sc or Scorer()
    os.makedirs(SCORES, exist_ok=True)
    for tag in tags:
        if not run_done(tag):
            print("[score] %s: no run" % tag); continue
        t0 = time.time()
        rec = sc.score(tag, phase_of(tag))
        json.dump(rec, open(os.path.join(SCORES, tag + ".json"), "w"), indent=1)
        if rec.get("J") is None:
            print("[score] %-14s %-22s no final state" % (tag, rec["status"])); continue
        u = rec["uterus"]
        print("[score] %-14s %-20s gates %-5s J %6.3f | uterus %.3f/%.2f/%.2f | vag+dev slab msd %.2f | t2s %+.1f ios %+.1f "
              "(C %.2f) | %.0f s" % (tag, rec["status"], rec["gates_pass"], rec["J"], u["dice"], u["msd"], u["hd95"],
                                    rec["vagina_device_slab"]["msd"], rec["tip_to_serosa_err_mm"],
                                    rec["internal_os_landing_err_mm"], rec["C_canal"], time.time() - t0), flush=True)
    return sc


def load_scores(phases=None):
    out = []
    for fn in sorted(os.listdir(SCORES)) if os.path.isdir(SCORES) else []:
        r = json.load(open(os.path.join(SCORES, fn)))
        if phases is None or r.get("phase") in phases:
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------- refinement / freeze
def best_of(recs):
    ok = [r for r in recs if r.get("gates_pass") and r.get("J") is not None]
    return min(ok, key=lambda r: r["J"]) if ok else None


def cmd_refine():
    if os.path.exists(CAL + "/jobs_refine.json"):
        sys.exit("jobs_refine.json exists; refusing to re-plan")
    b = best_of(load_scores(["lhs"]))
    if b is None:
        sys.exit("no gate-passing LHS run")
    u0 = val_to_u(b["cfg"])
    u, _ = maximin_lhs(N_REF, len(PARAMS), 1)
    lo = np.clip(u0 - REF_HALFWIDTH, 0, 1); hi = np.clip(u0 + REF_HALFWIDTH, 0, 1)
    jobs = [job("R%d" % i, u_to_val(lo + ui * (hi - lo)), "refine") for i, ui in enumerate(u)]
    json.dump(dict(best_lhs=b["tag"], J=b["J"], u0=u0.round(4).tolist(), box_lo=lo.round(4).tolist(),
                   box_hi=hi.round(4).tolist(), jobs=jobs), open(CAL + "/jobs_refine.json", "w"), indent=1)
    print("refinement around %s (J %.3f): %d jobs" % (b["tag"], b["J"], len(jobs)))
    for j in jobs:
        print(" ", j["tag"], {k: j["cfg"][k] for _, k, *_ in PARAMS})


def cmd_freeze():
    if os.path.exists(CAL + "/best.json"):
        sys.exit("best.json exists (frozen); refusing to overwrite")
    recs = load_scores(["lhs", "refine"])
    b = best_of(recs)
    ok = sorted([r for r in recs if r.get("gates_pass") and r.get("J") is not None], key=lambda r: r["J"])
    ident = [r for r in ok if r["J"] <= b["J"] + IDENT_TOL_MM]
    spread = {}
    for _, key, lo, hi, sc, un, _ in PARAMS:
        v = [r["cfg"][key] for r in ident]
        spread[key] = dict(min=min(v), max=max(v), unit=un, prior_range=[lo, hi])
    rd = os.path.join(P["runs"], b["tag"])
    best = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        statement="Frozen BEFORE any HR-CTV metric of any calibration run was computed. Selection: argmin J over "
                  "gate-passing LHS + refinement runs (plan.json).",
        best_tag=b["tag"], J=b["J"], objective_terms=dict(uterus=b["uterus"], vagina_device_slab=b["vagina_device_slab"],
                                                          C_canal=b["C_canal"], tip_to_serosa_err_mm=b["tip_to_serosa_err_mm"],
                                                          internal_os_landing_err_mm=b["internal_os_landing_err_mm"],
                                                          vagina_apex_err_mm=b["vagina_apex_err_mm"]),
        params={key: b["cfg"][key] for _, key, *_ in PARAMS},
        fixed=dict(E_kPa=b["cfg"]["E_kPa"], k_tie_mN_per_mm=b["cfg"]["k_tie_mN_per_mm"], k_s_mN_per_mm=b["cfg"]["k_s_mN_per_mm"],
                   cell_mm=b["cfg"]["cell_mm"], frame="BONE (pose PB)"),
        identifiable_ratios=dict(kappa_f_over_E_per_mm=b["cfg"]["kappa_f_mN_per_mm3"] / b["cfg"]["E_kPa"],
                                 kappa_lig_over_E_per_mm=b["cfg"]["kappa_lig_mN_per_mm3"] / b["cfg"]["E_kPa"],
                                 note="mN/mm^3 over kPa (= mN/mm^2) -> 1/mm; with displacement-driven loading only these "
                                      "ratios (and nu) set the shape"),
        full_cfg=json.load(open(rd + "/cfg.json")),
        gates=b["gates"], numerics=b["numerics"],
        identifiability_set=dict(tol_mm=IDENT_TOL_MM, tags=[r["tag"] for r in ident], J=[r["J"] for r in ident],
                                 param_spread=spread),
        n_candidates=len(recs), n_gate_pass=len(ok),
        hashes=dict(plan=sha256_file(CAL + "/plan.json"), code={f: sha256_file(os.path.join(HERE, f)) for f in CAL_CODE},
                    best_run={f: sha256_file(os.path.join(rd, f)) for f in ("cfg.json", "summary.json", "final.npz")},
                    scores={r["tag"]: sha256_file(os.path.join(SCORES, r["tag"] + ".json")) for r in recs}))
    txt = json.dumps(best, indent=1)
    open(CAL + "/best.json", "w").write(txt)
    open(CAL + "/best.sha256", "w").write(hashlib.sha256(txt.encode()).hexdigest() + "  best.json\n")
    print("FROZEN best %s  J %.3f  %s" % (b["tag"], b["J"], json.dumps(best["params"])))
    print("identifiability set (J <= best + %.1f mm): %s" % (IDENT_TOL_MM, best["identifiability_set"]["tags"]))


def check_frozen():
    """best.json must match best.sha256, and every calibration code file must match the hash frozen in best.json OR a
    hash listed with a reason in calibration/amendments.json (post-freeze edits are recorded, never silent)."""
    fn = CAL + "/best.json"
    if not os.path.exists(fn):
        sys.exit("REFUSED: best.json does not exist (freeze first)")
    h = open(CAL + "/best.sha256").read().split()[0]
    # the digest is of the JSON text with LF newlines (as generated); on Windows the text-mode write stores CRLF, so
    # compare against the universal-newline text (this detects any content change, not newline conversion)
    if hashlib.sha256(open(fn, encoding="utf-8").read().encode()).hexdigest() != h:
        sys.exit("REFUSED: best.json does not match best.sha256")
    best = json.load(open(fn))
    amend = json.load(open(CAL + "/amendments.json")) if os.path.exists(CAL + "/amendments.json") else {"entries": []}
    ok_hash = {f: {v} for f, v in best["hashes"]["code"].items()}
    for e in amend["entries"]:
        ok_hash.setdefault(e["file"], set()).add(e["new_sha256"])
    bad = [f for f in CAL_CODE if sha256_file(os.path.join(HERE, f)) not in ok_hash.get(f, set())]
    if bad:
        sys.exit("REFUSED: calibration code changed after the freeze without an amendment entry: %s" % bad)
    return best


def control_jobs(best):
    pv = best["params"]; E = best["fixed"]["E_kPa"]
    s3 = dict(E_kPa=3 * E, kappa_f_mN_per_mm3=3 * pv["kappa_f_mN_per_mm3"], kappa_lig_mN_per_mm3=3 * pv["kappa_lig_mN_per_mm3"],
              k_tie_mN_per_mm=3 * best["fixed"]["k_tie_mN_per_mm"], k_s_mN_per_mm=3 * best["fixed"]["k_s_mN_per_mm"])
    return [job("K_scale3", pv, "control", s3), job("K_ktie250", pv, "control", dict(k_tie_mN_per_mm=250.0)),
            job("K_ks1200", pv, "control", dict(k_s_mN_per_mm=1200.0)), job("K_cell3", pv, "control", dict(cell_mm=3.0)),
            job("K_HR", pv, "control", dict(pose="P2"))]


def cmd_run(phase):
    if phase == "lhs":
        jobs = json.load(open(CAL + "/jobs_lhs.json"))
    elif phase == "refine":
        jobs = json.load(open(CAL + "/jobs_refine.json"))["jobs"]
    elif phase == "controls":
        jobs = control_jobs(check_frozen())
        json.dump(jobs, open(CAL + "/jobs_controls.json", "w"), indent=1)
    else:
        sys.exit("phase lhs|refine|controls")
    pending = run_jobs(jobs, phase)
    cmd_score([j["tag"] for j in jobs])
    if pending:
        print("NOT RUN:", [j["tag"] for j in pending])


# ---------------------------------------------------------------------------------------------- held-out HR-CTV
def cmd_heldout():
    best = check_frozen()
    import evaluate as ev
    import posthoc
    bt = ev.BT(); al = json.load(open(VAL + "/alignment.json")); named = json.load(open(INP + "/named.json"))
    V0, Fc = geom.read_obj(INP + "/pre_hrctv.obj")
    out = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), best_sha256=open(CAL + "/best.sha256").read().split()[0],
               code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("calibrate.py", "posthoc.py", "evaluate.py", "gates.py")},
               statement="HR-CTV computed only now, after best.json was frozen; it played no role in any selection.",
               units="Dice [-], MSD/HD95 mm, cc", BT_HRCTV_cc=round(float(bt.lab["HR-CTV"].sum() * bt.vv / 1000), 2), runs={})
    for fr in ("BONE", "HR"):
        f = al["frames"][fr]; R, t = np.array(f["R_BT_to_pre"]), np.array(f["t_BT_to_pre"])
        m = ev.voxelize((V0 - t) @ R, Fc, bt.shape, bt.aff)
        out["rigid_baseline_" + fr] = ev.seg_metrics(m, bt.lab["HR-CTV"], bt.sp, bt.vv)
    recs = load_scores()
    for r in recs:
        tag = r["tag"]; rd = os.path.join(P["runs"], tag)
        if r.get("J") is None:
            continue
        fr = r["frame"]; f = al["frames"][fr]; R, t = np.array(f["R_BT_to_pre"]), np.array(f["t_BT_to_pre"])
        Vd, Fd = geom.read_obj(rd + "/surf_hrctv.obj")
        m = ev.voxelize((Vd - t) @ R, Fd, bt.shape, bt.aff)
        out["runs"][tag] = dict(phase=r["phase"], frame=fr, J=r["J"], gates_pass=r["gates_pass"],
                                held_out=(fr == "BONE"), HR_CTV=ev.seg_metrics(m, bt.lab["HR-CTV"], bt.sp, bt.vv))
    # device-informed rigid controls (stage-2 posthoc definitions) for the best run and the S0-soft reference
    for tag in (best["best_tag"], "V4_PB_M2_soft"):
        rc = posthoc.rigid_controls(tag, "BONE", bt, al, named)
        out.setdefault("rigid_controls", {})[tag] = {k: dict(uterus=v["uterus"], HR_CTV=v["HR-CTV"], fit_rms_mm=v["fit_rms_mm"])
                                                     for k, v in rc["controls"].items()}
    b = out["runs"][best["best_tag"]]["HR_CTV"]; base = out["rigid_baseline_BONE"]
    rcn = out["rigid_controls"][best["best_tag"]]["RIGID_CANAL"]["HR_CTV"]
    out["verdict"] = dict(
        best_vs_rigid_canal_control=dict(dice=round(b["dice"] - rcn["dice"], 4), msd=round(b["msd"] - rcn["msd"], 3),
                                         hd95=round(b["hd95"] - rcn["hd95"], 3)),
        best_vs_rigid_frame_baseline_device_position_informative_only=dict(
            dice=round(b["dice"] - base["dice"], 4), msd=round(b["msd"] - base["msd"], 3), hd95=round(b["hd95"] - base["hd95"], 3)),
        note=("negative msd/hd95 and positive dice = calibrated simulation better.  The rigid frame baseline leaves the "
              "organs ~24 mm from their BT position, so beating it shows only that the device position is informative, "
              "not that the mechanics adds anything; the mechanics test is against device-informed rigid placement "
              "(RIGID_CANAL here; the run-independent B_dev in final_eval.py).  HR-CTV: not in the objective, but the "
              "analysts saw bone-frame HR-CTV numbers in stage 2, and it overlaps the in-sample uterus by 20-30 %%."))
    ids = best["identifiability_set"]["tags"]
    band = [out["runs"][t]["HR_CTV"] for t in ids if t in out["runs"]]
    out["identifiability_band_HR_CTV"] = {q: [round(min(x[q] for x in band), 4), round(max(x[q] for x in band), 4)]
                                          for q in ("dice", "msd", "hd95")}
    cand = [(r["J"], out["runs"][r["tag"]]["HR_CTV"]["msd"]) for r in recs
            if r["tag"] in out["runs"] and r["phase"] in ("lhs", "refine") and r["gates_pass"]]
    if len(cand) > 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr([c[0] for c in cand], [c[1] for c in cand])
        out["J_vs_HRCTV_msd_spearman"] = dict(rho=round(float(rho), 3), p=round(float(p), 4), n=len(cand))
    json.dump(out, open(CAL + "/heldout.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("rigid_baseline_BONE", "verdict", "identifiability_band_HR_CTV")}, indent=1))
    print("best HR-CTV:", b, "\nrigid-canal control:", rcn, "\nJ vs HR-CTV MSD:", out.get("J_vs_HRCTV_msd_spearman"))


# ---------------------------------------------------------------------------------------------- analysis & figure
def srrc(U, y, nboot=2000, seed=0):
    """Standardised rank regression coefficients of y on the columns of U, with bootstrap 90 % intervals."""
    from scipy.stats import rankdata

    def fit(Ui, yi):
        R = np.column_stack([rankdata(c) for c in Ui.T]); r = rankdata(yi)
        R = (R - R.mean(0)) / (R.std(0) + 1e-12); r = (r - r.mean()) / (r.std() + 1e-12)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(r)), R]), r, rcond=None)
        pred = np.column_stack([np.ones(len(r)), R]) @ beta
        return beta[1:], 1 - ((r - pred) ** 2).sum() / (r ** 2).sum()
    b, r2 = fit(U, y)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(i)) > U.shape[1] + 2:
            bs.append(fit(U[i], y[i])[0])
    bs = np.array(bs)
    return b, np.percentile(bs, 5, 0), np.percentile(bs, 95, 0), r2


def spearman_boot(U, y, nboot=2000, seed=0):
    """Marginal Spearman rank correlation of y with each column of U, bootstrap 90 % intervals.  Bounded in [-1, 1]
    and stable at small n (unlike SRRC, which is near-saturated when n ~ 2 x number of parameters)."""
    from scipy.stats import spearmanr

    def rho(Ui, yi):
        return np.array([spearmanr(Ui[:, j], yi)[0] if np.ptp(Ui[:, j]) > 0 and np.ptp(yi) > 0 else 0.0
                         for j in range(Ui.shape[1])])
    r = rho(U, y); rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(i)) > 4:
            bs.append(rho(U[i], y[i]))
    bs = np.nan_to_num(np.array(bs))
    return r, np.percentile(bs, 5, 0), np.percentile(bs, 95, 0)


def cmd_analyze():
    recs = load_scores()
    best = json.load(open(CAL + "/best.json")) if os.path.exists(CAL + "/best.json") else None
    sc = None
    base = None
    try:
        sc = Scorer(); base = sc.baseline("BONE"); base_hr = sc.baseline("HR")
    except Exception as e:  # noqa: BLE001
        print("baseline scoring failed:", e); base_hr = None
    # ---- sweep.csv (no HR-CTV)
    cols = (["tag", "phase", "frame", "status", "gates_pass"] + [p[1] for p in PARAMS] +
            ["E_kPa", "k_tie_mN_per_mm", "k_s_mN_per_mm", "cell_mm", "J_mm", "uterus_msd_mm", "uterus_dice", "uterus_hd95_mm",
             "vag_dev_slab_msd_mm", "vag_dev_slab_dice", "vag_slab_dice", "vag_apex_err_mm", "tip_to_serosa_err_mm",
             "internal_os_landing_err_mm", "flange_to_corpus_err_mm", "C_canal_mm", "uterus_vol_change_frac", "max_stretch",
             "min_vol_ratio", "tie_err_max_mm", "pen_max_mm", "F_axial_mN", "F_ovoid_axial_mN", "run_s", "ms_per_step_median"])
    rows = []
    for r in ([base, base_hr] if base else []) + recs:
        if r is None:
            continue
        c = r.get("cfg", {}); n = r.get("numerics", {}); has = r.get("J") is not None
        row = dict(tag=r["tag"], phase=r.get("phase"), frame=r.get("frame"), status=r.get("status", "rigid"),
                   gates_pass=r.get("gates_pass", ""))
        for _, key, *_ in PARAMS + [(None, k) for k in ("E_kPa", "k_tie_mN_per_mm", "k_s_mN_per_mm", "cell_mm")]:
            row[key] = c.get(key, "")
        if has:
            row.update(J_mm=r["J"], uterus_msd_mm=r["uterus"]["msd"], uterus_dice=r["uterus"]["dice"], uterus_hd95_mm=r["uterus"]["hd95"],
                       vag_dev_slab_msd_mm=r["vagina_device_slab"]["msd"], vag_dev_slab_dice=r["vagina_device_slab"]["dice"],
                       vag_slab_dice=r["vagina_slab"]["dice"], vag_apex_err_mm=r["vagina_apex_err_mm"],
                       tip_to_serosa_err_mm=r["tip_to_serosa_err_mm"], internal_os_landing_err_mm=r["internal_os_landing_err_mm"],
                       flange_to_corpus_err_mm=r["flange_to_corpus_err_mm"], C_canal_mm=r["C_canal"],
                       uterus_vol_change_frac=r.get("uterus_mesh_volume_change_frac", ""))
        row.update(max_stretch=n.get("max_stretch", ""), min_vol_ratio=n.get("min_vol_ratio", ""), tie_err_max_mm=n.get("tie_err_max", ""),
                   pen_max_mm=n.get("pen_max", ""), F_axial_mN=n.get("F_axial_mN", ""), F_ovoid_axial_mN=n.get("F_ovoid_axial_mN", ""),
                   run_s=r.get("run_s", ""), ms_per_step_median=r.get("ms_per_step_median", ""))
        rows.append(row)
    with open(CAL + "/sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    print("wrote", CAL + "/sweep.csv", len(rows), "rows")
    # ---- sensitivity on the gate-passing LHS + refinement runs
    cand = [r for r in recs if r.get("phase") in ("lhs", "refine") and r.get("gates_pass") and r.get("J") is not None]
    allc = [r for r in recs if r.get("phase") in ("lhs", "refine")]
    U = np.array([val_to_u(r["cfg"]) for r in cand])
    outs = {"J": np.array([r["J"] for r in cand]), "uterus MSD": np.array([r["uterus"]["msd"] for r in cand]),
            "vagina+device slab MSD": np.array([r["vagina_device_slab"]["msd"] for r in cand]),
            "canal/landing C": np.array([r["C_canal"] for r in cand])}
    saturated = len(cand) < 2 * (len(PARAMS) + 1)
    sens = dict(n=len(cand), n_all=len(allc),
                method=("PRIMARY: marginal Spearman rank correlation of each output with each parameter (unit-cube inputs), "
                        "bootstrap 90 %% intervals (2000 resamples). SECONDARY: standardised rank regression coefficients "
                        "(SRRC, joint fit)%s." % (" -- NEAR-SATURATED (n < 2 x (parameters + 1)); treat as indicative only"
                                                  if saturated else "")),
                srrc_saturated=bool(saturated), outputs={})
    for k, y in outs.items():
        b, lo, hi, r2 = srrc(U, y)
        rs, rlo, rhi = spearman_boot(U, y)
        sens["outputs"][k] = dict(
            spearman={p[0]: dict(value=round(float(v), 3), ci90=[round(float(l), 3), round(float(h), 3)])
                      for p, v, l, h in zip(PARAMS, rs, rlo, rhi)},
            R2_rank=round(float(r2), 3),
            srrc={p[0]: dict(value=round(float(bi), 3), ci90=[round(float(l), 3), round(float(h), 3)])
                  for p, bi, l, h in zip(PARAMS, b, lo, hi)})
    # gate failures vs parameters (which parameters make the model inadmissible): Spearman of each parameter with the
    # failure indicator over ALL LHS + refinement runs (= rank-biserial association; + = larger value, more failures)
    Ua = np.array([val_to_u(r["cfg"]) for r in allc]); fail = np.array([0.0 if r.get("gates_pass") else 1.0 for r in allc])
    fs, flo, fhi = spearman_boot(Ua, fail)
    sens["gate_failure_association"] = dict(
        n_fail=int(fail.sum()), n_all=len(allc),
        spearman={p[0]: dict(value=round(float(v), 3), ci90=[round(float(l), 3), round(float(h), 3)])
                  for p, v, l, h in zip(PARAMS, fs, flo, fhi)})
    fails = [r for r in allc if not r.get("gates_pass")]
    sens["gate_failures"] = [dict(tag=r["tag"], status=r["status"], gates={k: v for k, v in (r.get("gates") or {}).items() if not v},
                                  params={p[1]: r["cfg"][p[1]] for p in PARAMS}) for r in fails]
    # controls relative to best
    if best:
        bz = np.load(os.path.join(P["runs"], best["best_tag"], "final.npz")); bs = json.load(open(os.path.join(SCORES, best["best_tag"] + ".json")))
        ctrl = {}
        for r in [r for r in recs if r.get("phase") == "control" and r.get("J") is not None]:
            z = np.load(os.path.join(P["runs"], r["tag"], "final.npz"))
            d = dict(J=r["J"], dJ=round(r["J"] - best["J"], 4), gates_pass=r["gates_pass"],
                     uterus=r["uterus"], d_uterus_msd=round(r["uterus"]["msd"] - bs["uterus"]["msd"], 4),
                     F_axial_mN=r["numerics"]["F_axial_mN"], F_ovoid_axial_mN=r["numerics"]["F_ovoid_axial_mN"])
            if z["X"].shape == bz["X"].shape and r["frame"] == "BONE":
                d["max_node_diff_mm"] = round(float(np.linalg.norm(z["X"] - bz["X"], axis=1).max()), 4)
                d["mean_node_diff_mm"] = round(float(np.linalg.norm(z["X"] - bz["X"], axis=1).mean()), 4)
            if r["frame"] == "HR" and base_hr:
                d["HR_frame_rigid_baseline_uterus"] = base_hr["uterus"]; d["HR_frame_rigid_baseline_J"] = base_hr["J"]
            ctrl[r["tag"]] = d
        sens["controls_vs_best"] = ctrl
        ident = best["identifiability_set"]["tags"]
        sens["identifiability_set"] = ident
    sens["baseline_BONE"] = base and {k: base[k] for k in ("J", "uterus", "vagina_device_slab", "C_canal",
                                                           "tip_to_serosa_err_mm", "internal_os_landing_err_mm")}
    refs = {r["tag"]: {k: r.get(k) for k in ("J", "uterus", "vagina_device_slab", "C_canal", "gates_pass")} for r in recs if r.get("phase") == "reference"}
    sens["references"] = refs
    json.dump(sens, open(CAL + "/sensitivity.json", "w"), indent=1)
    for k, v in sens["outputs"].items():
        print("%-24s spearman  " % k + "  ".join("%s %+.2f" % (p, d["value"]) for p, d in v["spearman"].items()))
    print("%-24s spearman  " % "gate failure" + "  ".join("%s %+.2f" % (p, d["value"])
                                                         for p, d in sens["gate_failure_association"]["spearman"].items()))
    fig_sensitivity(cand, allc, sens, best, base)


def fig_sensitivity(cand, allc, sens, best, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    SURF, INK, INK2, GRID, AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e2", "#b5b4ad"
    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    fig = plt.figure(figsize=(14.5, 10.2)); fig.patch.set_facecolor(SURF)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.28)
    btag = best["best_tag"] if best else None
    ident = set(best["identifiability_set"]["tags"]) if best else set()
    fails = [r for r in allc if not r.get("gates_pass") and r.get("J") is not None]
    for i, (nm, key, lo, hi, scl, un, _) in enumerate(PARAMS):
        ax = fig.add_subplot(gs[i // 4, i % 4]); ax.set_facecolor(SURF)
        x = [r["cfg"][key] for r in cand]; y = [r["J"] for r in cand]
        ax.scatter(x, y, s=46, color=C1, edgecolor=SURF, linewidth=1.5, zorder=3, label="gate-passing run")
        if fails:
            ax.scatter([r["cfg"][key] for r in fails], [r["J"] for r in fails], s=46, facecolor="none", edgecolor=AX,
                       linewidth=1.4, zorder=2, label="failed numeric gates")
        seen = False
        for r in cand:
            if r["tag"] in ident and r["tag"] != btag:
                ax.scatter(r["cfg"][key], r["J"], s=120, facecolor="none", edgecolor=C3, linewidth=1.6, zorder=4,
                           label=None if seen else "within 0.3 mm of best")
                seen = True
        if btag:
            rb = [r for r in cand if r["tag"] == btag][0]
            ax.scatter(rb["cfg"][key], rb["J"], s=150, facecolor="none", edgecolor=C2, linewidth=2.2, zorder=5, label="best (frozen)")
        if scl == "log":
            ax.set_xscale("log")
        ax.set_xlim(lo / 1.15 if scl == "log" else lo - 0.04 * (hi - lo), hi * 1.15 if scl == "log" else hi + 0.04 * (hi - lo))
        ax.set_xlabel("%s (%s)" % (key, un), fontsize=8.5, color=INK2)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(AX)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        v = sens["outputs"]["J"]["spearman"][nm]
        ax.set_title("Spearman rho with J %+.2f [%+.2f, %+.2f]" % (v["value"], v["ci90"][0], v["ci90"][1]),
                     fontsize=8.5, color=INK, loc="left")
        if i % 4 == 0:
            ax.set_ylabel("objective J (mm, lower is better)", fontsize=8.5, color=INK2)
    # legend panel
    axl = fig.add_subplot(gs[1, 3]); axl.axis("off")
    h, l = [], []
    for ax in fig.axes[:-1]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    axl.legend(h, l, loc="upper left", frameon=False, fontsize=9)
    txt = ["J = MSD_uterus + 0.5 MSD_vag+device slab + 0.5 C_canal  (mm)", "BONE frame (no organ label); HR-CTV held out",
           "n = %d gate-passing of %d LHS+refinement runs" % (len(cand), len(allc))]
    if base:
        txt.append("rigid BONE baseline: J = %.2f mm (off scale)" % base["J"])
    if best:
        txt.append("best %s: J = %.2f mm" % (best["best_tag"], best["J"]))
    hf = CAL + "/heldout.json"
    if best and os.path.exists(hf):                     # written only after the freeze
        h = json.load(open(hf)); hb = h["runs"][best["best_tag"]]["HR_CTV"]; hr = h["rigid_baseline_BONE"]
        rc = h["rigid_controls"][best["best_tag"]]["RIGID_CANAL"]["HR_CTV"]
        txt += ["", "HELD-OUT HR-CTV (Dice / MSD / HD95 mm), after freeze:",
                "  best sim            %.3f / %.2f / %.1f" % (hb["dice"], hb["msd"], hb["hd95"]),
                "  rigid BONE baseline %.3f / %.2f / %.1f" % (hr["dice"], hr["msd"], hr["hd95"]),
                "  rigid canal-on-tandem control %.3f / %.2f / %.1f" % (rc["dice"], rc["msd"], rc["hd95"])]
    axl.text(0.0, 0.42, "\n".join(txt), fontsize=8, color=INK2, va="top", transform=axl.transAxes)
    # Spearman bars (which parameters matter for each objective term)
    def _style(ax):
        ax.axhline(0, color=AX, lw=1)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right", "bottom"):
            ax.spines[s_].set_visible(False)
        ax.spines["left"].set_color(AX); ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_ylim(-1.05, 1.05)
    ax = fig.add_subplot(gs[2, :3]); ax.set_facecolor(SURF)
    outs = [("J", C1), ("uterus MSD", C2), ("vagina+device slab MSD", C3), ("canal/landing C", C4)]
    npar = len(PARAMS); wbar = 0.19; xg = np.arange(npar)
    for j, (k, col) in enumerate(outs):
        d = sens["outputs"][k]["spearman"]
        vals = np.array([d[p[0]]["value"] for p in PARAMS])
        lo_ = np.array([d[p[0]]["ci90"][0] for p in PARAMS]); hi_ = np.array([d[p[0]]["ci90"][1] for p in PARAMS])
        xs = xg + (j - 1.5) * (wbar + 0.02)
        ax.bar(xs, vals, width=wbar, color=col, edgecolor=SURF, linewidth=1.5, zorder=3, label=k)
        ax.errorbar(xs, vals, yerr=[np.maximum(vals - lo_, 0), np.maximum(hi_ - vals, 0)], fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2, zorder=4)
    _style(ax)
    ax.set_xticks(xg); ax.set_xticklabels(["%s\n%s" % (p[0], p[5]) for p in PARAMS], fontsize=8.5, color=INK)
    ax.set_ylabel("Spearman rho (gate-passing runs)", fontsize=9, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=8.5)
    ax.set_title("Which parameters matter (n = %d): rank correlation with each error term, 90 %% bootstrap intervals; "
                 "+ = larger value, larger error" % sens["n"], fontsize=9.5, color=INK, loc="left")
    # gate failures (admissibility) vs parameters
    ax = fig.add_subplot(gs[2, 3]); ax.set_facecolor(SURF)
    g = sens["gate_failure_association"]
    vals = np.array([g["spearman"][p[0]]["value"] for p in PARAMS])
    lo_ = np.array([g["spearman"][p[0]]["ci90"][0] for p in PARAMS]); hi_ = np.array([g["spearman"][p[0]]["ci90"][1] for p in PARAMS])
    ax.bar(xg, vals, width=0.6, color=INK2, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.errorbar(xg, vals, yerr=[np.maximum(vals - lo_, 0), np.maximum(hi_ - vals, 0)], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=2, zorder=4)
    _style(ax)
    ax.set_xticks(xg); ax.set_xticklabels([p[0] for p in PARAMS], fontsize=7.5, color=INK, rotation=45, ha="right")
    ax.set_title("Gate failures (%d of %d runs):\nrank correlation with failing" % (g["n_fail"], g["n_all"]),
                 fontsize=9, color=INK, loc="left")
    fig.suptitle("Stage-3 calibration landscape: tandem + ovoid insertion, preBT -> BT, one patient", fontsize=12, color=INK,
                 x=0.01, ha="left", y=0.995)
    fn = CAL + "/sensitivity.png"
    fig.savefig(fn, dpi=110, facecolor=SURF, bbox_inches="tight"); plt.close(fig)
    print("wrote", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "refine", "freeze", "heldout", "analyze", "score"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.force)
    elif a.cmd == "run":
        cmd_run(a.args[0])
    elif a.cmd == "refine":
        cmd_refine()
    elif a.cmd == "freeze":
        cmd_freeze()
    elif a.cmd == "heldout":
        cmd_heldout()
    elif a.cmd == "analyze":
        cmd_analyze()
    elif a.cmd == "score":
        cmd_score(a.args)


if __name__ == "__main__":
    main()
