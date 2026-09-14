"""Run one insertion (or a batch) headless (CONTAINER).  Units mm, mN, s.

    python3 /app/run_insertion.py --tag R1_P2_S0 --cfg '{"pose": "P2"}'
    python3 /app/run_insertion.py --tag R1_P2_S0 --cfg /out/runs/R1_P2_S0/cfg_in.json
    python3 /app/run_insertion.py --batch /out/runs/batch_m1a.json      # [{"tag":..., "cfg": {...}}, ...]
Outputs /out/runs/<tag>/: cfg.json, log.jsonl, final.npz, surf_{body,uterus,hrctv}.obj (deformed, preBT world
mm), summary.json and run.json (same content: timings, convergence, residuals, flags).
With cfg frame_every > 0 also frames/step_XXXX_{body,uterus,hrctv}.obj + step_XXXX_device.json every N steps and at
the last step (index.json lists them): the input of the host renderer animate.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
import scene  # noqa: E402
from controller import ApplicatorController, build_schedule  # noqa: E402

import Sofa.Core  # noqa: E402
import Sofa.Simulation  # noqa: E402


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def deformed_surface(ctx, nm, X, cache=None):
    # 'body' is the body mesh itself (pre_body or pre_bodyvag); the others are the organ surfaces.
    # cache (dict, optional): rest mesh and trilinear weights per name, for repeated per-frame calls.
    fn = ctx["cfg"].get("body_mesh", "pre_body") if nm == "body" else "pre_%s" % nm
    key = ("rest", nm)
    if cache is not None and key in cache:
        V0, Fc = cache[key]
    else:
        V0, Fc = geom.read_obj(os.path.join(ctx["inp"]["inp"], "%s.obj" % fn))
        if cache is not None:
            cache[key] = (V0, Fc)
    if nm in ctx["surfs"]:
        return np.array(ctx["surfs"][nm].mo.position.value, dtype=float, copy=True), Fc, V0
    key = ("weights", nm)
    if cache is not None and key in cache:
        nodes, W = cache[key]
    else:
        _, nodes, W, _ = geom.trilinear_weights(V0, ctx["X0"], ctx["hexa"])
        if cache is not None:
            cache[key] = (nodes, W)
    return np.einsum("ij,ijk->ik", W, X[nodes]), Fc, V0


PHASE_NAMES = dict(A="approach", T="insertion", D="ovoid inflation", H="hold", RL="rl action")


def write_frame(ctx, ctrl, out, row, fc):
    """cfg frame_every > 0: dump the deformed mapped surfaces and the applicator state of the step just solved to
    runs/<tag>/frames/ (step_XXXX_{body,uterus,hrctv}.obj + step_XXXX_device.json; index.json lists the frames).
    Read-out only (mapped children / trilinear weights); it never touches the solve.  fc: per-run cache/index dict."""
    cfg = ctx["cfg"]; fd = os.path.join(out, "frames"); os.makedirs(fd, exist_ok=True)
    k = int(row["step"]); ph = row["phase"]; pose = ctrl.pose
    X = np.array(ctx["dofs"].position.value, dtype=float, copy=True)
    files = {}
    for nm in scene.surface_names(cfg):
        V, Fc, _ = deformed_surface(ctx, nm, X, cache=fc)
        fn = "step_%04d_%s.obj" % (k, nm)
        geom.write_obj(os.path.join(fd, fn), V, Fc,
                       header="step %d phase %s: deformed pre_%s, preBT world RAS mm (local only)" % (k, ph, nm))
        files[nm] = fn
    F = np.asarray(pose["F"], float); a = geom.unit(pose["a"]); Rm = geom.pose_frame(pose)
    u = float(pose.get("u", 0.0)); rs = float(pose.get("r_scale", 0.0)); L = float(ctrl.L)
    O = np.asarray(ctx["inp"]["poses"]["O_pre"], float); tip = F + L * a
    dev = dict(step=k, phase=ph, phase_name=PHASE_NAMES.get(ph, ph), u=round(u, 6),
               hold_step=int(ctrl.hold) if ph == "H" else 0, units="mm, mN; preBT world RAS",
               flange_mm=F.round(4).tolist(), axis=a.round(6).tolist(), x_app=Rm[0].round(6).tolist(),
               tip_mm=tip.round(4).tolist(),
               rod_length_mm=L,                              # rigid tandem, flange -> tip (what is drawn)
               depth_mm=round(L * u, 4),                     # inserted depth of the spec kinematics tip = P(u) + L_iu u a
               tip_beyond_O_pre_mm=round(float((tip - O) @ a), 4),   # tip past the rest external-os point, along the axis
               tandem_r_mm=float(ctrl.r_rod), shaft_len_mm=float(cfg["shaft_len_mm"]),
               sphere_r_scale=rs, spheres=[],
               canal_mm=np.round(ctrl.canal_full(X), 4).tolist(),          # current canal centre line (read-out)
               n_engaged=row["n_engaged"], n_released=row["n_released"], n_contacts=row["n_contacts"],
               umax_mm=row["umax"], umean_mm=row["umean"], dx_max_mm=row["dx_max"], pen_max_mm=row["pen_max"],
               F_axial_mN=row["F_axial_mN"], F_ovoid_axial_mN=row["F_ovoid_axial_mN"])
    if rs > 0:                                                # ovoid/cap sphere pack (travel: from the first T step)
        cc, rr = ctrl.spheres_world(pose, rs)
        dev["spheres"] = [dict(center_mm=c.round(4).tolist(), r_mm=round(float(r), 4)) for c, r in zip(cc, rr)]
    if ctrl.anchor is not None:                               # device-rigid support anchors (controller._begin)
        an = ctrl.anchor; w = 0.0 if ph == "A" else (u if ph == "T" else 1.0)
        dev["support_anchor"] = dict(mode="device_rigid", progress_w=w, pivot_mm=np.asarray(an["pivot"]).round(4).tolist(),
                                     rot_axis=np.asarray(an["axis"]).round(6).tolist(), rot_deg=round(float(an["angle_deg"]) * w, 4),
                                     shift_mm=(w * np.asarray(an["shift"])).round(4).tolist(),
                                     n_support_nodes=int(len(ctx["ns"]["sup_nodes"])))
    else:
        dev["support_anchor"] = dict(mode=cfg.get("found_anchor", "rest"), n_support_nodes=int(len(ctx["ns"]["sup_nodes"])))
    dj = "step_%04d_device.json" % k
    with open(os.path.join(fd, dj), "w") as fh:
        json.dump(dev, fh, default=_jsonable)
    idx = fc.setdefault("index", [])
    idx.append(dict(step=k, phase=ph, phase_name=PHASE_NAMES.get(ph, ph), u=round(u, 6), device=dj, surfaces=files))
    with open(os.path.join(fd, "index.json"), "w") as fh:
        json.dump(dict(tag=cfg.get("tag"), frame_every=int(cfg.get("frame_every") or 0), units="mm; preBT world RAS",
                       frames=idx), fh, indent=1)
    fc["last_step"] = k


CODE_FILES = ("config.py", "geom.py", "scene.py", "controller.py", "ties.py", "run_insertion.py", "gates.py", "calib_run.py")


def provenance():
    """sha256 of the container-side code, the docker image id (env APPSIM_IMAGE_ID, set by run_docker.sh) and versions."""
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    code = {}
    for f in CODE_FILES:
        p = os.path.join(here, f)
        if os.path.exists(p):
            code[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    try:
        import Sofa as _S
        sofa_v = getattr(_S, "__version__", None)
    except Exception:
        sofa_v = None
    return dict(code_sha256=code, image_id=os.environ.get("APPSIM_IMAGE_ID"), numpy=np.__version__,
                python=sys.version.split()[0], sofapython3=sofa_v, sofa="v22.12 (image eve-training-fixed)")


def run_one(cfg, tag, wall_budget_s=None):
    P = config.paths()
    out = os.path.join(P["runs"], tag); os.makedirs(out, exist_ok=True)
    cfg = dict(cfg); cfg["tag"] = tag
    prov = provenance()
    open(os.path.join(out, "RUNNING"), "w").write("started %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    for fn in ("summary.json", "run.json"):                  # a stale summary must never pass for this run's result
        if os.path.exists(os.path.join(out, fn)):
            os.remove(os.path.join(out, fn))
    with open(os.path.join(out, "cfg.json"), "w") as fh:
        json.dump(dict(cfg, _provenance=prov), fh, indent=1, default=_jsonable)
    t0 = time.perf_counter()
    root = Sofa.Core.Node("root")
    ctx = scene.build_scene(root, cfg)
    sched = build_schedule(ctx)
    ctrl = root.addObject(ApplicatorController(name="applicator", ctx=ctx, schedule=sched, out_dir=out))
    Sofa.Simulation.init(root)
    t_init = time.perf_counter() - t0
    X = np.array(ctx["dofs"].position.value, dtype=float, copy=True)
    same = bool(np.allclose(X, ctx["X0"]))
    info = scene.scene_summary(ctx)
    info.update(tag=tag, t_init_s=round(t_init, 2), pass2_nodes_equal_pass1=same, n_sched=len(sched),
                n_phaseA=sum(1 for s in sched if s["phase"] == "A"), n_phaseT=sum(1 for s in sched if s["phase"] == "T"),
                n_phaseD=sum(1 for s in sched if s["phase"] == "D"))
    print("INIT", json.dumps(info, default=_jsonable), flush=True)
    if not same:
        raise RuntimeError("pass-2 FEM nodes differ from pass 1")
    # the mapped (full) canal must equal the trilinear read-out
    cm = np.array(ctx["canal_mo"].position.value, dtype=float)
    info["canal_map_vs_trilinear_mm"] = float(np.abs(cm - ctrl.canal_full(X)).max())
    wall_limit = float(cfg["wall_limit_s"]) if wall_budget_s is None else float(wall_budget_s)
    fe = int(cfg.get("frame_every") or 0); fc = {}                   # per-frame export (animate.py), default off
    t_run0 = time.perf_counter(); walls = []
    while not ctrl.done and ctrl.k < int(cfg["max_steps"]):
        if time.perf_counter() - t0 > wall_limit:
            ctrl.status = "abort_wall_time"; break
        ts = time.perf_counter()
        Sofa.Simulation.animate(root, root.dt.value)
        walls.append(1000 * (time.perf_counter() - ts))
        ctrl.rows[-1]["wall_ms"] = round(walls[-1], 2)
        r = ctrl.rows[-1]
        if r["step"] % 10 == 0 or ctrl.done or r["phase"] == "H":
            print("STEP %3d %s u=%.3f wall=%.0fms umax=%.2f dx=%.3f tie=%.3f/%.3f (lat %.3f tip %.3f) eng=%d "
                  "Fax=%.0fmN minV=%.3f stretch=%.3f"
                  % (r["step"], r["phase"], r["u"], r["wall_ms"], r["umax"], r["dx_max"], r["tie_err_mean"],
                     r["tie_err_max"], r["tie_err_lat_max"], r["tie_err_tip_max"], r["n_engaged"], r["F_axial_mN"],
                     r["min_vol_ratio"], r["max_stretch"]), flush=True)
        if fe > 0 and r.get("finite", False) and (r["step"] % fe == 0 or ctrl.done):
            write_frame(ctx, ctrl, out, r, fc)
    if not ctrl.done and ctrl.status == "running":
        ctrl.status = "max_steps"
    if fe > 0 and ctrl.rows and ctrl.rows[-1].get("finite", False) and fc.get("last_step") != ctrl.rows[-1]["step"]:
        write_frame(ctx, ctrl, out, ctrl.rows[-1], fc)                 # last step of a wall-time / max_steps exit
    t_run = time.perf_counter() - t_run0
    ctrl.close()
    rows = ctrl.rows
    X = np.array(ctx["dofs"].position.value, dtype=float, copy=True)
    fin = rows[-1] if rows else {}
    wall = np.array([r.get("wall_ms", np.nan) for r in rows]); ph = np.array([r["phase"] for r in rows])
    rowsA = [r for r in rows if r["phase"] == "A"]
    phaseA_ok = bool(rowsA and max(r["umax"] for r in rowsA) < 1e-3 and all(r["n_engaged"] == 0 for r in rowsA))
    rowsH = [r for r in rows if r["phase"] == "H"]; rowsT = [r for r in rows if r["phase"] == "T"]
    cgT = [r["cg_its"] for r in rowsT if r.get("cg_its") is not None]
    cgH = [r["cg_its"] for r in rowsH if r.get("cg_its") is not None]
    summary = dict(info)
    summary.update(provenance=prov, hold_rule=cfg.get("hold_rule", "v1"),
                   phaseA_null_ties=int(rowsA[0].get("n_null_ties", 0)) if rowsA else 0,
                   cg_its_T=dict(median=float(np.median(cgT)), frac_at_cap=float(np.mean([r["cg_its"] >= r["cg_cap"] for r in rowsT if r.get("cg_its") is not None])))
                   if cgT else None,
                   cg_its_H=dict(median=float(np.median(cgH)), max=int(max(cgH)),
                                 frac_at_cap=float(np.mean([r["cg_its"] >= r["cg_cap"] for r in rowsH if r.get("cg_its") is not None])))
                   if cgH else None,
                   ms_per_step_median_H=round(float(np.nanmedian([r["wall_ms"] for r in rowsH])), 1) if rowsH else None,
                   n_steps_T=len(rowsT),
                   final_f_res_max_mN=fin.get("f_res_max_mN"), final_drift_est_mm=fin.get("drift_est_mm"),
                   final_lam_max_mm=fin.get("lam_max_mm"), final_rod_beyond_canal_end_mm=fin.get("rod_beyond_canal_end_mm"),
                   n_neg_corner_tet_run_max=int(max(r.get("n_neg_corner_tet", 0) for r in rows)) if rows else None,
                   min_tet_ratio_run=float(min(r["min_tet_ratio"] for r in rows)) if rows else None)
    summary.update(
        status=ctrl.status, converged=ctrl.status == "converged", n_steps=len(rows),
        steps_to_convergence=len(rows) if ctrl.status == "converged" else None,
        n_hold_steps=int((ph == "H").sum()), run_s=round(t_run, 2), total_s=round(time.perf_counter() - t0, 2),
        ms_per_step_median=round(float(np.nanmedian(wall)), 1) if len(wall) else None,
        ms_per_step_mean=round(float(np.nanmean(wall)), 1) if len(wall) else None,
        ms_per_step_median_T=round(float(np.nanmedian(wall[ph == "T"])), 1) if (ph == "T").any() else None,
        solve_ms_median=round(float(np.median([r["solve_ms"] for r in rows])), 1) if rows else None,
        ctrl_ms_median=round(float(np.median([r["ctrl_ms"] for r in rows])), 2) if rows else None,
        phaseA_control_ok=phaseA_ok, nonfinite=not fin.get("finite", False),
        final=fin, min_vol_ratio_run=float(min(r["min_vol_ratio"] for r in rows)) if rows else None,
        max_stretch_run=float(max(r["max_stretch"] for r in rows)) if rows else None,
        admissibility_flag=bool(rows and (min(r["min_vol_ratio"] for r in rows) < 0.2 or max(r["max_stretch"] for r in rows) > 2.0)),
        final_umax_mm=fin.get("umax"), final_umean_mm=fin.get("umean"),
        final_tie_err_mean_mm=fin.get("tie_err_mean"), final_tie_err_max_mm=fin.get("tie_err_max"),
        final_F_axial_mN=fin.get("F_axial_mN"), final_F_lat_sum_mN=fin.get("F_lat_sum_mN"),
        tie_impl=ctx.get("tie_impl"),
        pyff_calls=(dict(addForce=ctx["tie"].n_addforce, addDForce=ctx["tie"].n_adddforce)
                    if ctx.get("tie_impl") == "pyff" else None),
        forces_note="penalty-tie forces at ASSUMED E=%g kPa; UNVALIDATED" % cfg["E_kPa"],
        contention_note="median wall ms on a shared host (another training container may use most CPUs), docker --cpus %s"
                        % os.environ.get("APPSIM_CPUS", "2"),
        frames=(dict(n=len(fc.get("index", [])), every=fe, dir="frames") if fe > 0 else None),
        cfg={k: v for k, v in cfg.items() if not k.startswith("_")})
    if fin.get("finite", False):
        for nm in scene.surface_names(cfg):
            V, Fc, V0 = deformed_surface(ctx, nm, X)
            geom.write_obj(os.path.join(out, "surf_%s.obj" % nm), V, Fc,
                           header="deformed pre_%s, preBT world RAS mm, run %s (local only)" % (nm, tag))
            du = np.linalg.norm(V - V0, axis=1)
            summary["surf_%s_umax_mm" % nm] = round(float(du.max()), 3)
            summary["surf_%s_umean_mm" % nm] = round(float(du.mean()), 3)
        pose = ctx["pose"]; R = geom.pose_frame(pose)
        sc = np.asarray(pose["F"]) + ctrl.sph_c_app @ R
        g = ctx["grp"]                                   # tie-point index of every (full) canal point
        np.savez(os.path.join(out, "final.npz"), X=X, X0=ctx["X0"], hexa=ctx["hexa"],
                 canal_rest=ctx["canal_full"], canal_final=ctrl.canal_full(X), s0=ctx["s0"], engaged=ctrl.engaged[g],
                 released=ctrl.released[g], is_tip=ctrl.is_tip[g], lam_lat=ctrl.lam_lat[g], lam_ax=ctrl.lam_ax[g],
                 tie_rest=ctx["canal"], tie_final=ctrl.canal_pos(X), tie_engaged=ctrl.engaged, tie_lam_lat=ctrl.lam_lat,
                 tie_lam_ax=ctrl.lam_ax, tie_grp=g, tie_members=ctx["grp_n"],
                 is_patch=ctx["is_patch"], F=np.asarray(pose["F"]), a=np.asarray(pose["a"]),
                 x=np.asarray(pose["x"]), L_iu=ctrl.L, r_rod=ctrl.r_rod, sphere_centers=sc, sphere_r=ctrl.sph_r,
                 sup_nodes=ctx["ns"]["sup_nodes"], k_sup=ctx["ns"]["k_sup"], lig=ctx["ns"]["lig"],
                 boundary=ctx["ns"]["boundary"], tie_nodes=ctx["tie_nodes"], cand=ctx["cand"],
                 region=ctx["ns"]["region"])
    import gates
    summary["numeric_gates_container"] = gates.numeric_gates(summary, pierce=None, cfg=cfg)   # piercing: host side
    for fn in ("summary.json", "run.json"):
        with open(os.path.join(out, fn), "w") as fh:
            json.dump(summary, fh, indent=1, default=_jsonable)
    if os.path.exists(os.path.join(out, "RUNNING")):
        os.remove(os.path.join(out, "RUNNING"))
    print("DONE", json.dumps({k: summary[k] for k in ("tag", "status", "n_steps", "t_init_s", "ms_per_step_median",
                                                      "final_umax_mm", "final_umean_mm", "final_tie_err_mean_mm",
                                                      "final_tie_err_max_mm", "final_F_axial_mN", "run_s")},
                             default=_jsonable), flush=True)
    Sofa.Simulation.unload(root)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--cfg", default=None, help="JSON file or inline JSON dict of overrides")
    ap.add_argument("--batch", default=None, help="JSON list of {tag, cfg}")
    a = ap.parse_args()
    if a.batch:
        jobs = json.load(open(a.batch))
        budget = float(os.environ.get("APPSIM_BATCH_BUDGET_S", 540))   # < the container's `timeout 570`
        t0 = time.perf_counter(); done, times = [], []
        # status is run output: always next to the runs (never into the code folder, e.g. /app/runs_def)
        stat = os.path.join(config.paths()["runs"], os.path.splitext(os.path.basename(a.batch))[0] + "_status.json")

        def write_status():
            rep = dict(batch=a.batch, done=done, remaining=[jb["tag"] for jb in jobs if jb["tag"] not in done], run_s=times)
            with open(stat, "w") as fh:
                json.dump(rep, fh, indent=1)
            return rep
        write_status()
        for j, job in enumerate(jobs):
            el = time.perf_counter() - t0
            med = float(np.median(times)) if times else 0.0
            if el + 1.3 * med > budget or budget - el < 45.0:
                break
            s = run_one(config.load_cfg(job.get("cfg", {})), job["tag"], wall_budget_s=budget - el)
            times.append(s["total_s"]); done.append(job["tag"])
            write_status()                                # after every job: a killed batch still leaves the list
        print("BATCH", json.dumps(write_status()))
        return
    ov = None
    if a.cfg:
        ov = json.load(open(a.cfg)) if os.path.isfile(a.cfg) else json.loads(a.cfg)
    run_one(config.load_cfg(ov), a.tag or "run")


if __name__ == "__main__":
    main()
    # SofaPython3 v22.12 segfaults in interpreter teardown once a Python ForceField has existed (ties.py);
    # every output is already closed, so leave without the Python finaliser.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
