"""HOST (python 3.13): FINISHER evaluation.  Units: mm, cc, deg; Dice [-].

    python final_eval.py bdev                 # run-independent device-informed rigid baseline B_dev(d_F) curve
    python final_eval.py rescore_calib        # re-score the stage-3 runs with sub-voxel landing terms (argmin check)
    python final_eval.py score TAG [TAG ...]  # score finisher runs against every comparator (+ gates, landing)
    python final_eval.py boot TAG [TAG ...]   # frame-perturbation noise floor for sim-minus-comparator deltas
    python final_eval.py report               # final/verdict.json + final/report.md + figs/final_*.png
Everything is POST HOC relative to the stage-2 unblinding of the uterus (and of bone-frame HR-CTV numbers); nothing
here is a pre-registered test except the pre-registration status table, which only READS the declared criteria.

Comparators (all through the identical route: surface -> rigid map into BT world -> vtkPolyDataToImageStencil on the
BT grid -> dice / surf_dists copied verbatim from registration/common.py):
  RIGID_FRAME   undeformed preBT anatomy under the run's registration frame (BONE = peri-organ MI, HR = HR-CTV ICP)
  B_DEV(d_F)    undeformed preBT anatomy placed on the REAL BT tandem without any run: the preBT canal axis a0 onto
                the tandem axis (minimal rotation; roll from the BONE frame, which uses no organ label) and the canal
                label end L_end at d_F mm above the BT flange.  Picks: d_F_tip = L_iu + 0.5 - (canal length above L_end)
                (geometry-only rule: canal end at the rod tip, no BT label), d_F_hr = argmin HR-CTV MSD, d_F_oracle =
                argmin uterus MSD (in-sample ceiling)
  RIGID_CANAL   Kabsch of the preBT canal onto the run's final canal (run-dependent; eps_frac 0 / 0.01 / 0.1 roll prior)
Metrics: Dice, MSD, HD95 (pooled symmetric, 6-connected surface voxels = the registration-stage definition) and
HD95max (max of the two directed 95th percentiles, the more common definition); HR-CTV-minus-uterus (masks
HR-CTV & ~uterus in both sessions) as the less uterus-coupled HR-CTV check.  Landing distances are sub-voxel:
first crossing of the trilinearly interpolated occupancy through 0.5 along the ray.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
import gates  # noqa: E402
import evaluate as ev  # noqa: E402

P = config.paths(); INP = P["inputs"]; VAL = P["out"] + "/validation"; CAL = P["out"] + "/calibration"
FIN = P["out"] + "/final"
D_GRID = np.round(np.arange(0.0, 27.51, 1.0), 2)
EQUIV_MM = 0.5                     # declared equivalence margin on MSD (mm) for "indistinguishable"
N_BOOT = 30; ROT_SD_DEG = 2.0; TR_SD_MM = 2.34 / np.sqrt(3.0)   # frame perturbation: ICP residual 2.34 mm rms


def _fmt(m):
    return "%.3f / %.2f / %.2f" % (m["dice"], m["msd"], m["hd95"])


# ------------------------------------------------------------------------------------------------ metrics
def metrics_ext(sim, ref, sp, vv):
    r = dict(dice=ev.dice(ref, sim)); r.update(ev.surf_dists(ref, sim, sp))     # identical to evaluate.metrics
    if sim.sum() and ref.sum():
        st = ndi.generate_binary_structure(3, 1)
        sa = ref & ~ndi.binary_erosion(ref, st); sb = sim & ~ndi.binary_erosion(sim, st)
        idx = np.argwhere(sa | sb); lo = np.maximum(idx.min(0) - 5, 0); hi = idx.max(0) + 6
        sl = tuple(slice(a, b) for a, b in zip(lo, hi)); sa, sb = sa[sl], sb[sl]
        da = ndi.distance_transform_edt(~sa, sampling=sp); db = ndi.distance_transform_edt(~sb, sampling=sp)
        r["hd95max"] = float(max(np.percentile(db[sa], 95), np.percentile(da[sb], 95)))
    r["vol_cc"] = float(sim.sum() * vv / 1000.0)
    return {k: round(float(v), 4) for k, v in r.items()}


def occ_along(mask, aff, p, d, t):
    inv = np.linalg.inv(aff); X = np.asarray(p)[None] + t[:, None] * np.asarray(d)[None]
    ijk = (X @ inv[:3, :3].T + inv[:3, 3]).T
    return ndi.map_coordinates(mask.astype(np.float32), ijk, order=1, mode="constant")


def _cross(t, o, inside_first):
    k = np.nonzero(o < 0.5)[0] if inside_first else np.nonzero(o >= 0.5)[0]
    if not len(k) or k[0] == 0:
        return float("nan") if not len(k) else float(t[0])
    k = k[0]; o0, o1 = o[k - 1], o[k]
    return float(t[k - 1] + (o0 - 0.5) / (o0 - o1) * (t[k] - t[k - 1]))


def exit_sub(mask, aff, p, d, step=0.05, maxd=100.0):
    """Sub-voxel distance along d from p to the exit of mask; negative = p outside (distance back to mask)."""
    t = np.arange(0, maxd, step); o = occ_along(mask, aff, p, d, t)
    if o[0] >= 0.5:
        return _cross(t, o, True)
    ob = occ_along(mask, aff, p, -np.asarray(d), t)
    return -_cross(t, ob, False)


def entry_sub(mask, aff, p, d, step=0.05, maxd=100.0):
    t = np.arange(0, maxd, step); o = occ_along(mask, aff, p, d, t)
    return _cross(t, o, False) if o[0] < 0.5 else 0.0


# ------------------------------------------------------------------------------------------------ context
class Ctx:
    def __init__(self):
        self.bt = ev.BT(); self.al = json.load(open(VAL + "/alignment.json"))
        self.named = json.load(open(INP + "/named.json")); po = json.load(open(INP + "/poses.json"))
        self.L_end = np.array(po["L_end"]); self.a0 = geom.unit(po["a0"])
        c = np.load(INP + "/canal.npz"); self.canal = c["pts"]; self.s0 = c["s0"]; self.i_le = int(c["i_L_end"])
        self.surf = {nm: geom.read_obj(INP + "/pre_%s.obj" % ob) for nm, ob in (("uterus", "uterus"), ("HR-CTV", "hrctv"))}
        bt = self.bt
        self.ref = dict(uterus=bt.lab["uterus"], **{"HR-CTV": bt.lab["HR-CTV"]})
        self.ref["HR-CTV-minus-uterus"] = bt.lab["HR-CTV"] & ~bt.lab["uterus"]
        self.canal_above_L_end = float(self.s0[-1] - self.s0[self.i_le])
        self.d_tip = float(bt.L) + 0.5 - self.canal_above_L_end
        self.t2s_BT = exit_sub(bt.lab["uterus"], bt.aff, bt.tip, bt.a)
        self.f2c_BT = entry_sub(bt.lab["uterus"], bt.aff, bt.F, bt.a)

    def frame(self, name):
        f = self.al["frames"][name]
        return np.array(f["R_BT_to_pre"]), np.array(f["t_BT_to_pre"])

    def to_bt_frame(self, name):
        R, t = self.frame(name)
        return lambda Y: (np.asarray(Y) - t) @ R            # x_BT = R^T (y - t)

    def bdev(self, dF, roll_frame="BONE"):
        R, t = self.frame(roll_frame); to_bt = self.to_bt_frame(roll_frame)
        Lb = to_bt(self.L_end[None])[0]; a0b = self.a0 @ R
        Rm = geom.rot_between(a0b, self.bt.a); tgt = self.bt.F + dF * self.bt.a
        return lambda Y: (to_bt(Y) - Lb) @ Rm.T + tgt

    def masks(self, surfs, fmap):
        """surfs: {name: (V, F)} in preBT world; fmap: preBT -> BT world.  Returns BT-grid masks incl. HR-minus-U."""
        bt = self.bt
        m = {nm: ev.voxelize(fmap(V), F, bt.shape, bt.aff) for nm, (V, F) in surfs.items()}
        m["HR-CTV-minus-uterus"] = m["HR-CTV"] & ~m["uterus"]
        return m

    def score(self, m, canal_bt=None, tip_bt=None, F_bt=None, a_bt=None):
        bt = self.bt
        out = {nm: metrics_ext(m[nm], self.ref[nm], bt.sp, bt.vv) for nm in self.ref}
        tip = bt.tip if tip_bt is None else tip_bt; F = bt.F if F_bt is None else F_bt; a = bt.a if a_bt is None else a_bt
        out["landing"] = dict(tip_to_serosa_mm=round(exit_sub(m["uterus"], bt.aff, tip, a), 2),
                              flange_to_corpus_mm=round(entry_sub(m["uterus"], bt.aff, F, a), 2),
                              BT_tip_to_serosa_mm=round(self.t2s_BT, 2), BT_flange_to_corpus_mm=round(self.f2c_BT, 2))
        out["landing"]["tip_to_serosa_err_mm"] = round(out["landing"]["tip_to_serosa_mm"] - self.t2s_BT, 2)
        out["landing"]["flange_to_corpus_err_mm"] = round(out["landing"]["flange_to_corpus_mm"] - self.f2c_BT, 2)
        if canal_bt is not None:
            i_ios = int(self.named["i_internal_os"])
            ios = float((canal_bt[i_ios] - bt.F) @ bt.a)
            out["landing"].update(internal_os_along_BT_axis_mm=round(ios, 2), internal_os_err_mm=round(ios - self.f2c_BT, 2))
        return out


# ------------------------------------------------------------------------------------------------ B_dev curve
def cmd_bdev(cx=None):
    cx = cx or Ctx(); os.makedirs(FIN, exist_ok=True)
    curve = []
    for d in D_GRID:
        m = cx.masks(cx.surf, cx.bdev(d))
        canal_b = cx.bdev(d)(cx.canal)
        r = cx.score(m, canal_b); r["d_F_mm"] = float(d); curve.append(r)
        print("B_dev d_F %5.1f | uterus %s | HR-CTV %s | HR-U %s" % (d, _fmt(r["uterus"]), _fmt(r["HR-CTV"]),
                                                                      _fmt(r["HR-CTV-minus-uterus"])), flush=True)
    picks = dict(d_F_tip=round(cx.d_tip, 2),
                 d_F_hr=float(min(curve, key=lambda r: r["HR-CTV"]["msd"])["d_F_mm"]),
                 d_F_oracle=float(min(curve, key=lambda r: r["uterus"]["msd"])["d_F_mm"]))
    sel = {}
    for k, d in picks.items():
        for roll in ("BONE", "HR"):
            m = cx.masks(cx.surf, cx.bdev(d, roll)); r = cx.score(m, cx.bdev(d, roll)(cx.canal)); r["d_F_mm"] = d
            sel["%s@roll_%s" % (k, roll)] = r
    out = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), units="mm; Dice [-]",
               definition=("undeformed preBT anatomy, canal axis a0 on the REAL BT tandem axis (minimal rotation; roll from "
                           "the BONE frame, which uses no organ label; HR-frame roll as sensitivity), L_end at d_F above the "
                           "BT flange; no simulation run involved"),
               canal_above_L_end_mm=round(cx.canal_above_L_end, 2), L_iu_mm=float(cx.bt.L),
               picks=picks, pick_rules=dict(
                   d_F_tip="L_iu + 0.5 - canal length above L_end (canal end at the rod tip; geometry only, no BT label)",
                   d_F_hr="argmin HR-CTV MSD over the grid (uses the BT HR-CTV; uterus stays out of the choice)",
                   d_F_oracle="argmin uterus MSD (IN-SAMPLE ceiling; 1 degree of freedom)"),
               selected=sel, curve=curve,
               honesty="post hoc: the physics/methodology review (after stage-2 unblinding) pointed to this comparator")
    json.dump(out, open(FIN + "/bdev.json", "w"), indent=1)
    print(json.dumps(picks), "->", FIN + "/bdev.json")
    return out


# ------------------------------------------------------------------------------------------------ calibration re-score
def cmd_rescore_calib(cx=None):
    """Sub-voxel landing terms for the stage-3 runs; argmin under the frozen objective form and alternatives."""
    cx = cx or Ctx(); bt = cx.bt; to_bt = cx.to_bt_frame("BONE")
    recs = []
    for fn in sorted(os.listdir(CAL + "/scores")):
        r = json.load(open(os.path.join(CAL, "scores", fn)))
        if r.get("J") is None or r.get("frame") != "BONE" or r.get("phase") not in ("lhs", "refine"):
            continue
        rd = os.path.join(P["runs"], r["tag"]); z = np.load(rd + "/final.npz")
        Vd, Fd = geom.read_obj(rd + "/surf_uterus.obj"); mu = ev.voxelize(to_bt(Vd), Fd, bt.shape, bt.aff)
        t2s = exit_sub(mu, bt.aff, bt.tip, bt.a)
        ios = float((to_bt(z["canal_final"])[int(cx.named["i_internal_os"])] - bt.F) @ bt.a)
        C_sub = 0.5 * (abs(t2s - cx.t2s_BT) + abs(ios - cx.f2c_BT))
        J_sub = r["uterus"]["msd"] + 0.5 * r["vagina_device_slab"]["msd"] + 0.5 * C_sub
        recs.append(dict(tag=r["tag"], gates_pass=r["gates_pass"], J=r["J"], J_subvoxel=round(J_sub, 4),
                         J_no_vagina=round(r["uterus"]["msd"] + 0.5 * C_sub, 4), uterus_msd=r["uterus"]["msd"],
                         t2s_err_voxel=r["tip_to_serosa_err_mm"], t2s_err_subvoxel=round(t2s - cx.t2s_BT, 2),
                         C_voxel=r["C_canal"], C_subvoxel=round(C_sub, 3)))
    ok = [r for r in recs if r["gates_pass"]]
    arg = {k: min(ok, key=lambda r: r[k])["tag"] for k in ("J", "J_subvoxel", "J_no_vagina", "uterus_msd")}
    out = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(recs), n_gate_pass=len(ok), argmin=arg,
               BT_tip_to_serosa_subvoxel_mm=round(cx.t2s_BT, 2), BT_flange_to_corpus_subvoxel_mm=round(cx.f2c_BT, 2),
               runs=sorted(recs, key=lambda r: r["J_subvoxel"]),
               note=("frozen selection used voxel-quantised tip-to-serosa (steps of ~1.8 mm along the tandem); the "
                     "frozen best.json is NOT changed; this only reports how the argmin moves"))
    json.dump(out, open(FIN + "/calib_rescore.json", "w"), indent=1)
    print("argmin:", arg)
    return out


# ------------------------------------------------------------------------------------------------ run scoring
def frame_of(tag):
    s = json.load(open(os.path.join(P["runs"], tag, "summary.json")))
    return s, ev.FRAME_OF_POSE.get(s["cfg"]["pose"], "E_APP")


def run_map(cx, z, frame):
    if frame == "E_APP":
        Rs = geom.pose_frame(dict(F=z["F"], a=z["a"], x=z["x"]))
        Re, te = geom.kabsch(ev.app_landmarks(z["F"], Rs, cx.bt.sph), ev.app_landmarks(cx.bt.F, cx.bt.R, cx.bt.sph))
        return lambda Y: np.asarray(Y) @ Re.T + te
    return cx.to_bt_frame(frame)


def kabsch_eps(src, dst, anchor, eps):
    if eps <= 0:
        return geom.kabsch(src, dst)
    import posthoc
    return posthoc.kabsch_reg(src, dst, anchor, eps)


def rigid_canal_map(cx, z, fmap, eps):
    n = len(z["canal_rest"]); i_le = cx.i_le
    sel = (np.arange(n) >= i_le) & z["engaged"].astype(bool) & ~z["released"].astype(bool)
    Rk, tk = kabsch_eps(z["canal_rest"][sel], z["canal_final"][sel], z["X0"], eps)
    return (lambda Y: fmap(np.asarray(Y) @ Rk.T + tk)), geom.rot_angle_deg(Rk)


def pierce_frac(rd, z):
    Vb, Fb = geom.read_obj(rd + "/surf_body.obj"); a = geom.unit(z["a"])
    rod = z["F"] + np.outer(np.arange(0.0, float(z["L_iu"]) + 1e-9, 1.0), a)
    ins = ev.enclosed(rod, Vb, Fb); first = int(np.argmax(ins)) if ins.any() else len(ins)
    return float((~ins[first:]).mean()) if first < len(ins) else 1.0


def cmd_score(tags, cx=None, bd=None):
    cx = cx or Ctx(); bt = cx.bt
    bd = bd or json.load(open(FIN + "/bdev.json"))
    os.makedirs(FIN + "/scores", exist_ok=True)
    for tag in tags:
        rd = os.path.join(P["runs"], tag)
        if not os.path.exists(rd + "/final.npz"):
            print("[score] %s: no final state" % tag); continue
        s, frame = frame_of(tag); z = np.load(rd + "/final.npz"); fmap = run_map(cx, z, frame)
        sim_surf = {nm: geom.read_obj(rd + "/surf_%s.obj" % ob) for nm, ob in (("uterus", "uterus"), ("HR-CTV", "hrctv"))}
        a_s = geom.unit(z["a"]); F_b = fmap(z["F"][None])[0]; tip_b = fmap((z["F"] + float(z["L_iu"]) * a_s)[None])[0]
        a_b = geom.unit(tip_b - F_b)
        rec = dict(tag=tag, frame=frame, pose=s["cfg"]["pose"], status=s["status"],
                   device_landing=dict(flange_err_mm=round(float(np.linalg.norm(F_b - bt.F)), 3),
                                       axis_err_deg=round(geom.angle_deg(a_b, bt.a), 3)),
                   cfg={k: s["cfg"].get(k) for k in ("fe_impl", "tie_group", "canal_fillet_mm", "found_anchor", "hold_rule",
                                                      "cell_mm", "nu", "kappa_f_mN_per_mm3", "kappa_lig_mN_per_mm3", "lig_h_max_mm",
                                                      "tip_push_margin_mm", "delta_fund_mm", "sph_r_scale", "sph_dz_mm", "m2")},
                   models={})
        pier = pierce_frac(rd, z)
        rec["gates"] = gates.numeric_gates(s, pierce=pier); rec["piercing_frac"] = pier
        rec["numerics"] = {k: s.get(k) for k in ("n_steps", "n_hold_steps", "t_init_s", "run_s", "ms_per_step_median",
                                                  "ms_per_step_median_T", "ms_per_step_median_H", "cg_its_T", "cg_its_H",
                                                  "final_f_res_max_mN", "final_drift_est_mm", "final_lam_max_mm",
                                                  "final_rod_beyond_canal_end_mm", "min_vol_ratio_run", "max_stretch_run",
                                                  "min_tet_ratio_run", "n_neg_corner_tet_run_max", "volumes_cc", "n_nodes", "n_hexa",
                                                  "n_tie_pts", "final_F_axial_mN", "final_F_lat_sum_mN")}
        fin = s.get("final", {})
        rec["numerics"].update(F_ovoid_axial_mN=fin.get("F_ovoid_axial_mN"), n_vol_out_0p85_1p15=fin.get("n_vol_out_0p85_1p15"),
                               max_vol_ratio=fin.get("max_vol_ratio"), min_vol_ratio=fin.get("min_vol_ratio"),
                               n_lam_sat=fin.get("n_lam_sat"), pen_max=fin.get("pen_max"))
        canal_sim_b = fmap(z["canal_final"])
        rec["models"]["SIM"] = cx.score(cx.masks(sim_surf, fmap), canal_sim_b, tip_b, F_b, a_b)
        rec["models"]["RIGID_FRAME"] = cx.score(cx.masks(cx.surf, fmap), fmap(z["canal_rest"]), tip_b, F_b, a_b)
        for k in ("d_F_tip", "d_F_hr", "d_F_oracle"):
            rec["models"]["B_DEV_%s" % k[4:].upper()] = bd["selected"]["%s@roll_BONE" % k]
        dlt = float(s["cfg"].get("delta_fund_mm") or 0.0)
        if dlt > 0:          # B_dev matched to the run's canal hypothesis (canal extended by delta_fund)
            dm = cx.d_tip - dlt; mm = cx.masks(cx.surf, cx.bdev(dm)); r = cx.score(mm, cx.bdev(dm)(cx.canal)); r["d_F_mm"] = dm
            rec["models"]["B_DEV_TIP_MATCHED"] = r
        for eps in (0.0, 0.01, 0.1):
            fm, ang = rigid_canal_map(cx, z, fmap, eps)
            r = cx.score(cx.masks(cx.surf, fm)); r["rotation_deg"] = round(ang, 2)
            rec["models"]["RIGID_CANAL_eps%g" % eps] = r
        dl = {}
        for mname, mr in rec["models"].items():
            if mname == "SIM":
                continue
            dl[mname] = {nm: {q: round(rec["models"]["SIM"][nm][q] - mr[nm][q], 4) for q in ("dice", "msd", "hd95", "hd95max")}
                         for nm in cx.ref}
        rec["delta_sim_minus"] = dl
        json.dump(rec, open(FIN + "/scores/%s.json" % tag, "w"), indent=1)
        su = rec["models"]["SIM"]
        print("[score] %-24s %-6s %-18s gates pre %s v2 %s | uterus %s | HR-CTV %s | HR-U %s | t2s %+.1f"
              % (tag, frame, s["status"], rec["gates"]["pass_preregistered"], rec["gates"]["pass_added_v2"], _fmt(su["uterus"]),
                 _fmt(su["HR-CTV"]), _fmt(su["HR-CTV-minus-uterus"]), su["landing"]["tip_to_serosa_err_mm"]), flush=True)
        for mname in ("RIGID_FRAME", "B_DEV_TIP", "B_DEV_HR", "RIGID_CANAL_eps0.01"):
            mr = rec["models"][mname]
            print("        %-20s uterus %s | HR-CTV %s" % (mname, _fmt(mr["uterus"]), _fmt(mr["HR-CTV"])), flush=True)


# ------------------------------------------------------------------------------------------------ noise floor
def _perturb(rng, c):
    w = np.radians(ROT_SD_DEG) * rng.standard_normal(3); th = np.linalg.norm(w)
    K = geom.skew(w / th) if th > 0 else np.zeros((3, 3))
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    t = TR_SD_MM * rng.standard_normal(3)
    return lambda X: (np.asarray(X) - c) @ R.T + c + t


def cmd_boot(tags, cx=None):
    """The SAME random rigid perturbation (rotation sd 2 deg/axis about the BT uterus centroid, translation sd
    2.34/sqrt(3) mm/axis ~ the HR-CTV ICP residual) is applied to the simulation and to each comparator; the spread of
    the sim-minus-comparator deltas over N_BOOT draws is the frame-choice noise floor of that comparison."""
    cx = cx or Ctx(); bt = cx.bt; bd = json.load(open(FIN + "/bdev.json"))
    c = ev.mask_world(bt.lab["uterus"], bt.aff).mean(0)
    out = json.load(open(FIN + "/boot.json")) if os.path.exists(FIN + "/boot.json") else {}
    for tag in tags:
        s, frame = frame_of(tag); rd = os.path.join(P["runs"], tag); z = np.load(rd + "/final.npz"); fmap = run_map(cx, z, frame)
        sim = {nm: geom.read_obj(rd + "/surf_%s.obj" % ob) for nm, ob in (("uterus", "uterus"), ("HR-CTV", "hrctv"))}
        models = {"SIM": (sim, fmap), "RIGID_FRAME": (cx.surf, fmap),
                  "B_DEV_TIP": (cx.surf, cx.bdev(bd["picks"]["d_F_tip"])), "B_DEV_HR": (cx.surf, cx.bdev(bd["picks"]["d_F_hr"])),
                  "RIGID_CANAL_eps0.01": (cx.surf, rigid_canal_map(cx, z, fmap, 0.01)[0])}
        rng = np.random.default_rng(0); draws = {m: [] for m in models}
        for b in range(N_BOOT):
            pt = _perturb(rng, c)
            for mname, (surfs, fm) in models.items():
                mk = {nm: ev.voxelize(pt(fm(V)), F, bt.shape, bt.aff) for nm, (V, F) in surfs.items()}
                draws[mname].append({nm: ev.metrics(mk[nm], cx.ref[nm], bt.sp) for nm in ("uterus", "HR-CTV")})
        res = {}
        for mname in models:
            if mname == "SIM":
                continue
            res[mname] = {}
            for nm in ("uterus", "HR-CTV"):
                for q in ("dice", "msd", "hd95"):
                    d = np.array([draws["SIM"][i][nm][q] - draws[mname][i][nm][q] for i in range(N_BOOT)])
                    lo, med, hi = np.percentile(d, [5, 50, 95])
                    res[mname]["%s_%s" % (nm, q)] = dict(median=round(float(med), 3), p5=round(float(lo), 3), p95=round(float(hi), 3))
                d = res[mname]["%s_msd" % nm]
                better = "sim better" if d["p95"] < 0 else ("comparator better" if d["p5"] > 0 else "interval covers 0")
                equiv = d["p5"] > -EQUIV_MM and d["p95"] < EQUIV_MM
                res[mname]["%s_verdict" % nm] = "%s%s" % (better, "; within +/-%.1f mm equivalence" % EQUIV_MM if equiv else "")
        out[tag] = dict(frame=frame, n_draws=N_BOOT, rot_sd_deg=ROT_SD_DEG, trans_sd_mm_per_axis=round(TR_SD_MM, 3),
                        equivalence_margin_msd_mm=EQUIV_MM, deltas_sim_minus=res)
        print(tag, json.dumps({m: {k: v for k, v in r.items() if k.endswith("verdict") or k.endswith("_msd")} for m, r in res.items()}, indent=1), flush=True)
    json.dump(out, open(FIN + "/boot.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]; args = sys.argv[2:]
    if cmd == "bdev":
        cmd_bdev()
    elif cmd == "rescore_calib":
        cmd_rescore_calib()
    elif cmd == "score":
        cmd_score(args)
    elif cmd == "boot":
        cmd_boot(args)
    elif cmd == "report":
        import final_report
        final_report.main()
