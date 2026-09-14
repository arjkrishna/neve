"""G0 gate (CONTAINER, < 8 min): in-context component checks, DNT solver equivalence and fidelity, zero-load
control and timing.  Writes /out/logs/G0.json.  Units mm, mN, ms.

  G0.1 components: StaticSolver data names; SparseGridTopology cellWidth counts at 3/4/5 mm; RSSFF per-point
       stiffness + runtime rewrite of stiffness and external target MO; TriangularFEMForceField + FixedConstraint
       on pre_vagina.obj; EigenSimplicialLDLT template.
  G0.2 one fixed-target DNT straightening step: SparseLDL vs EigenSimplicialLDLT vs CG(1000, 1e-12) (and CG(200)):
       max nodal difference < 1e-3 mm.
  G0.3 DNT+LDL (20 steps) vs springs on MAPPED canal points + CG (the known-correct physics): canal diff < 0.3 mm.
  G0.4 zero-load control: 10 steps without loading, umax < 1e-3 mm.
  G0.5 timing: 15 phase-T steps of P2 at cell 4 and 5 mm per linear solver; median ms/step.
    python3 /app/gate_tests.py [--quick]
"""
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import geom  # noqa: E402
import scene  # noqa: E402
from controller import ApplicatorController, build_schedule  # noqa: E402

import Sofa.Core  # noqa: E402
import Sofa.Simulation  # noqa: E402

T0 = time.perf_counter()
RES = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"))
INP = scene.load_inputs()
OUT = os.path.join(config.paths()["logs"], "G0.json")


def save():
    RES["elapsed_s"] = round(time.perf_counter() - T0, 1)
    json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def cfg_(**kw):
    # G0 baseline: direct solver, 3 Newton iterations (independent of the run defaults chosen by G0)
    c = config.load_cfg(); c.update(linear_solver="SparseLDLSolver", newton_iterations=3, tie_impl="rssff"); c.update(kw)
    return c


def X_of(ctx):
    return np.array(ctx["dofs"].position.value, dtype=float, copy=True)


def build(cfg, tie_mode="dnt"):
    root = Sofa.Core.Node("root")
    t = time.perf_counter()
    ctx = scene.build_scene(root, cfg, INP, tie_mode)
    Sofa.Simulation.init(root)
    ctx["t_init_s"] = time.perf_counter() - t
    return root, ctx


def straighten_targets(ctx):
    """Fixed targets: label-canal points (above L_end) projected on the L_end -> C_top chord."""
    c = ctx["canal"]; i0 = int(INP["canal"]["i_L_end"])
    L_end = c[i0]; d = geom.unit(c[-1] - L_end)
    t = L_end + np.outer((c - L_end) @ d, d)
    k = np.zeros(len(c)); k[i0:] = cfg_()["k_tie_mN_per_mm"]
    return t, k


def dnt_write(ctx, t, k):
    X = X_of(ctx)
    cp = canal_of(ctx, X)
    li = ctx["loc"][ctx["cnodes"]]; kw = k[:, None] * ctx["W"]
    kj = np.zeros(len(ctx["tie_nodes"])); np.add.at(kj, li.ravel(), kw.ravel())
    num = np.zeros((len(ctx["tie_nodes"]), 3)); np.add.at(num, li.ravel(), (kw[:, :, None] * (t - cp)[:, None, :]).reshape(-1, 3))
    tgt = X[ctx["tie_nodes"]] + np.where(kj[:, None] > 0, num / np.maximum(kj, 1e-300)[:, None], 0.0)
    ctx["tie_tgt"].position.value = tgt.tolist(); ctx["tie"].stiffness.value = kj.tolist()


def canal_of(ctx, X):
    return np.einsum("ij,ijk->ik", ctx["W"], X[ctx["cnodes"]]) + ctx["c_off"]


def g01():
    r = {}
    root = Sofa.Core.Node("r"); root.addObject("RequiredPlugin", pluginName="Sofa.Component")
    ss = root.addChild("b").addObject("StaticSolver", name="ss")
    names = set(d.getName() for d in ss.getDataFields())
    need = ["newton_iterations", "absolute_correction_tolerance_threshold", "relative_correction_tolerance_threshold",
            "absolute_residual_tolerance_threshold", "relative_residual_tolerance_threshold",
            "should_diverge_when_residual_is_growing"]
    r["StaticSolver_data_ok"] = all(n in names for n in need)
    r["StaticSolver_missing"] = [n for n in need if n not in names]
    counts = {}
    for cw in (3.0, 4.0, 5.0):
        X0, hexa, n = scene.probe_grid(os.path.join(INP["inp"], "pre_body.obj"), cw)
        cell = np.abs(X0[hexa[:, 6]] - X0[hexa[:, 0]]).mean(0)
        counts["%g" % cw] = dict(n_nodes=int(len(X0)), n_hexa=int(len(hexa)), grid_n=n, cell_real_mm=np.round(cell, 3).tolist(),
                                 hexa_vol_cc=round(float(len(hexa) * np.prod(cell) / 1000), 2))
    r["sparse_grid_counts"] = counts
    r["cellWidth_honoured"] = bool(abs(counts["4"]["cell_real_mm"][0] - 4.0) < 0.5 and counts["3"]["n_nodes"] > counts["4"]["n_nodes"] > counts["5"]["n_nodes"])
    # RSSFF runtime rewrite of stiffness and of the external target MO
    root, ctx = build(cfg_(cell_mm=5.0, mapped_children=False))
    X0 = X_of(ctx); tn = ctx["tie_nodes"]
    ctx["tie_tgt"].position.value = (X0[tn] + [1.0, 0, 0]).tolist(); ctx["tie"].stiffness.value = [50.0] * len(tn)
    Sofa.Simulation.animate(root, 1.0); X1 = X_of(ctx)
    ctx["tie_tgt"].position.value = (X0[tn] + [0, 1.0, 0]).tolist()
    Sofa.Simulation.animate(root, 1.0); X2 = X_of(ctx)
    ctx["tie"].stiffness.value = [0.0] * len(tn)
    for _ in range(2):
        Sofa.Simulation.animate(root, 1.0)
    X3 = X_of(ctx)
    r["rssff_rewrite"] = dict(dx_tie_after_x_target=float((X1 - X0)[tn, 0].mean()), dy_tie_after_y_target=float((X2 - X0)[tn, 1].mean()),
                              dx_after_y_retarget=float((X2 - X0)[tn, 0].mean()),
                              umax_after_k0=float(np.linalg.norm(X3 - X0, axis=1).max()))
    rr = r["rssff_rewrite"]
    r["rssff_rewrite_ok"] = bool(rr["dx_tie_after_x_target"] > 0.2 and rr["dy_tie_after_y_target"] > 0.2 and abs(rr["dx_after_y_retarget"]) < 0.5 * rr["dx_tie_after_x_target"] and rr["umax_after_k0"] < 1e-3)
    Sofa.Simulation.unload(root)
    # vagina membrane
    try:
        V, Fc = geom.read_obj(os.path.join(INP["inp"], "pre_vagina.obj"))
        root = Sofa.Core.Node("v"); root.addObject("RequiredPlugin", pluginName="Sofa.Component"); root.addObject("DefaultAnimationLoop")
        root.gravity = [0, 0, 0]
        v = root.addChild("vag"); scene.add_static_solver(v, cfg_()); v.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
        v.addObject("MeshObjLoader", name="loader", filename=os.path.join(INP["inp"], "pre_vagina.obj"))
        v.addObject("TriangleSetTopologyContainer", name="topo", src="@loader")
        mo = v.addObject("MechanicalObject", name="mo", template="Vec3d", src="@loader")
        v.addObject("TriangularFEMForceField", name="fem", youngModulus=150.0, poissonRatio=0.45, method="large")
        ax = geom.unit(np.linalg.svd(V - V.mean(0), full_matrices=False)[2][0]); h = (V - V.mean(0)) @ ax
        low = np.nonzero(h < h.min() + 25.0)[0]
        v.addObject("FixedConstraint", indices=low.tolist())
        top = int(np.argmax(h))
        v.addObject("RestShapeSpringsForceField", name="found", points=list(range(len(V))), stiffness=[0.05])
        v.addObject("RestShapeSpringsForceField", name="pull", points=[top], stiffness=[50.0], external_rest_shape="@/tg/t", external_points=[0])
        tg = root.addChild("tg"); tg.addObject("MechanicalObject", name="t", template="Vec3d", position=[(V[top] + 2.0 * ax).tolist()])
        Sofa.Simulation.init(root)
        for _ in range(3):
            Sofa.Simulation.animate(root, 1.0)
        Xv = np.array(mo.position.value)
        r["vagina_membrane"] = dict(n_verts=int(len(V)), n_fixed=int(len(low)), finite=bool(np.all(np.isfinite(Xv))),
                                    top_disp_mm=float(np.linalg.norm(Xv[top] - V[top])), fixed_disp_max=float(np.linalg.norm(Xv[low] - V[low], axis=1).max()))
        r["vagina_membrane_ok"] = bool(r["vagina_membrane"]["finite"] and r["vagina_membrane"]["top_disp_mm"] > 0.1 and r["vagina_membrane"]["fixed_disp_max"] < 1e-9)
        Sofa.Simulation.unload(root)
    except Exception as e:
        r["vagina_membrane_ok"] = False; r["vagina_membrane_err"] = repr(e)[:400]
    return r


def g02():
    r = {}; Xs = {}
    for nm, kw in (("SparseLDLSolver", {}), ("EigenSimplicialLDLT", {}),
                   ("CG1000", dict(linear_solver="CGLinearSolver", cg_iterations=1000, cg_tolerance=1e-12, cg_threshold=1e-15)),
                   ("CG200", dict(linear_solver="CGLinearSolver", cg_iterations=200, cg_tolerance=1e-9, cg_threshold=1e-12))):
        cfg = cfg_(cell_mm=5.0, mapped_children=False, **kw)
        if nm in ("SparseLDLSolver", "EigenSimplicialLDLT"):
            cfg["linear_solver"] = nm
        root, ctx = build(cfg)
        t, k = straighten_targets(ctx)
        dnt_write(ctx, t, k)
        ts = time.perf_counter(); Sofa.Simulation.animate(root, 1.0); ms = 1000 * (time.perf_counter() - ts)
        Xs[nm] = X_of(ctx); r[nm + "_ms"] = round(ms, 1)
        r[nm + "_umax"] = float(np.linalg.norm(Xs[nm] - ctx["X0"], axis=1).max())
        Sofa.Simulation.unload(root)
    ref = Xs["CG1000"]
    for nm in ("SparseLDLSolver", "EigenSimplicialLDLT", "CG200"):
        r["maxdiff_%s_vs_CG1000_mm" % nm] = float(np.linalg.norm(Xs[nm] - ref, axis=1).max())
    r["maxdiff_LDL_vs_Eigen_mm"] = float(np.linalg.norm(Xs["SparseLDLSolver"] - Xs["EigenSimplicialLDLT"], axis=1).max())
    r["pass"] = {nm: bool(r["maxdiff_%s_vs_CG1000_mm" % nm] < 1e-3) for nm in ("SparseLDLSolver", "EigenSimplicialLDLT", "CG200")}
    return r


def g03(nsteps=20):
    r = {}
    root, ctx = build(cfg_(cell_mm=5.0, mapped_children=False))
    t, k = straighten_targets(ctx)
    hist = []
    for s in range(nsteps):
        dnt_write(ctx, t, k); Sofa.Simulation.animate(root, 1.0)
        cp = canal_of(ctx, X_of(ctx)); hist.append(float(np.linalg.norm(cp - t, axis=1)[k > 0].max()))
    C_dnt = canal_of(ctx, X_of(ctx)); X_dnt = X_of(ctx)
    Sofa.Simulation.unload(root)
    cfg = cfg_(cell_mm=5.0, mapped_children=False, linear_solver="CGLinearSolver", cg_iterations=3000, cg_tolerance=1e-14,
               cg_threshold=1e-18, newton_iterations=5)
    root, ctx = build(cfg, tie_mode="mapped")
    ctx["ref_tgt"].position.value = t.tolist(); ctx["ref"].stiffness.value = k.tolist()
    ts = time.perf_counter(); hist_m = []
    for s in range(3):
        Sofa.Simulation.animate(root, 1.0)
        hist_m.append(float(np.linalg.norm(np.array(ctx["canal_mo"].position.value) - t, axis=1)[k > 0].max()))
    r["mapped_cg_ms_per_step"] = round(1000 * (time.perf_counter() - ts) / 3, 1)
    C_map = np.array(ctx["canal_mo"].position.value); X_map = X_of(ctx)
    Sofa.Simulation.unload(root)
    r.update(dnt_tie_err_max_by_step=np.round(hist, 4).tolist(), mapped_tie_err_max_by_step=np.round(hist_m, 4).tolist(),
             canal_maxdiff_mm=float(np.linalg.norm(C_dnt - C_map, axis=1).max()),
             canal_meandiff_mm=float(np.linalg.norm(C_dnt - C_map, axis=1).mean()),
             node_maxdiff_mm=float(np.linalg.norm(X_dnt - X_map, axis=1).max()),
             umax_dnt=float(np.linalg.norm(X_dnt - ctx["X0"], axis=1).max()))
    r["pass"] = bool(r["canal_maxdiff_mm"] < 0.3)
    return r


def g04():
    root, ctx = build(cfg_(cell_mm=4.0, mapped_children=True))
    for _ in range(10):
        Sofa.Simulation.animate(root, 1.0)
    u = float(np.linalg.norm(X_of(ctx) - ctx["X0"], axis=1).max())
    Sofa.Simulation.unload(root)
    return dict(umax_mm=u, pass_=bool(u < 1e-3))


def g05(budget_s):
    r = {}
    combos = [(4.0, "SparseLDLSolver", False, 1), (4.0, "EigenSimplicialLDLT", False, 1), (4.0, "SparseLDLSolver", True, 1),
              (4.0, "SparseLDLSolver", False, 3), (5.0, "SparseLDLSolver", False, 1), (5.0, "EigenSimplicialLDLT", False, 1),
              (4.0, "EigenSimplicialLDLT", False, 3), (5.0, "CGLinearSolver", False, 1)]
    for cw, ls, mc, nit in combos:
        key = "cell%g_%s_%s_newton%d" % (cw, ls, "mapped" if mc else "nomap", nit)
        if time.perf_counter() - T0 > budget_s:
            r[key] = "skipped (time budget)"; continue
        try:
            cfg = cfg_(cell_mm=cw, linear_solver=ls, mapped_children=mc, pose="P2", newton_iterations=nit)
            root = Sofa.Core.Node("root"); t = time.perf_counter()
            ctx = scene.build_scene(root, cfg, INP)
            sched = build_schedule(ctx)
            ctrl = root.addObject(ApplicatorController(name="applicator", ctx=ctx, schedule=sched, out_dir=None))
            Sofa.Simulation.init(root); t_init = time.perf_counter() - t
            walls = []
            while ctrl.k < cfg["phaseA_steps"] + 15:
                ts = time.perf_counter(); Sofa.Simulation.animate(root, 1.0); walls.append(1000 * (time.perf_counter() - ts))
            wT = walls[cfg["phaseA_steps"]:]
            r[key] = dict(t_init_s=round(t_init, 2), n_nodes=int(len(ctx["X0"])), n_hexa=int(len(ctx["hexa"])),
                          ms_median_T=round(float(np.median(wT)), 1), ms_mean_T=round(float(np.mean(wT)), 1),
                          solve_ms_median=round(float(np.median([q["solve_ms"] for q in ctrl.rows[cfg["phaseA_steps"]:]])), 1),
                          ctrl_ms_median=round(float(np.median([q["ctrl_ms"] for q in ctrl.rows[cfg["phaseA_steps"]:]])), 2),
                          umax_after=round(ctrl.rows[-1]["umax"], 3), tie_err_max=round(ctrl.rows[-1]["tie_err_max"], 3),
                          phaseA_umax=float(max(q["umax"] for q in ctrl.rows[:cfg["phaseA_steps"]])))
            Sofa.Simulation.unload(root)
        except Exception as e:
            r[key] = "ERROR " + repr(e)[:300]
        save_part("G0.5", r)
    return r


def save_part(k, v):
    RES[k] = v; save()


def pyff_write(ctx, t, k, P):
    """Fixed-target DNT through the NodeTieFF (ties.py) with one projector P for every canal point."""
    X = X_of(ctx); cp = canal_of(ctx, X)
    li = ctx["loc"][ctx["cnodes"]]; W = ctx["W"]; kw = k[:, None] * W; m = len(ctx["tie_nodes"])
    fi = k[:, None] * ((t - cp) @ P.T)
    Kb = np.zeros((m, 3, 3)); np.add.at(Kb, li.ravel(), (kw[:, :, None, None] * P[None, None]).reshape(-1, 3, 3))
    F0 = np.zeros((m, 3)); np.add.at(F0, li.ravel(), (W[:, :, None] * fi[:, None, :]).reshape(-1, 3))
    ctx["tie"].set_linearisation(ctx["tie_nodes"], X[ctx["tie_nodes"]], F0, Kb)


def g06():
    """G0.6 (added with ties.py): the Python NodeTieFF with P = I must reproduce the RSSFF DNT on the real body
    (one fixed-target straightening step, cell 4 mm, StaticSolver 3 Newton + CG(1000, 1e-12)); max nodal
    difference < 1e-3 mm.  Then the timing of the run default (anisotropic pyff, CG(200)) over 15 phase-T steps of P2."""
    r = {}; Xs = {}
    for impl in ("rssff", "pyff"):
        cfg = cfg_(cell_mm=4.0, mapped_children=False, linear_solver="CGLinearSolver", cg_iterations=1000,
                   cg_tolerance=1e-12, cg_threshold=1e-15, tie_impl=impl)
        root, ctx = build(cfg)
        t, k = straighten_targets(ctx)
        if impl == "rssff":
            dnt_write(ctx, t, k)
        else:
            pyff_write(ctx, t, k, np.eye(3))
        ts = time.perf_counter(); Sofa.Simulation.animate(root, 1.0); r[impl + "_one_step_ms"] = round(1000 * (time.perf_counter() - ts), 1)
        Xs[impl] = X_of(ctx); r[impl + "_umax_mm"] = float(np.linalg.norm(Xs[impl] - ctx["X0"], axis=1).max())
        Sofa.Simulation.unload(root)
    r["maxdiff_pyffI_vs_rssff_mm"] = float(np.linalg.norm(Xs["pyff"] - Xs["rssff"], axis=1).max())
    r["pass_equivalence"] = bool(r["maxdiff_pyffI_vs_rssff_mm"] < 1e-3)
    cfg = config.load_cfg(); cfg.update(pose="P2")
    root = Sofa.Core.Node("root"); ctx = scene.build_scene(root, cfg, INP)
    sched = build_schedule(ctx); ctrl = root.addObject(ApplicatorController(name="applicator", ctx=ctx, schedule=sched, out_dir=None))
    Sofa.Simulation.init(root); walls = []
    while ctrl.k < cfg["phaseA_steps"] + 15:
        ts = time.perf_counter(); Sofa.Simulation.animate(root, 1.0); walls.append(1000 * (time.perf_counter() - ts))
    r["default_cfg"] = dict(tie_impl=cfg["tie_impl"], linear_solver=cfg["linear_solver"], cell_mm=cfg["cell_mm"],
                            newton_iterations=cfg["newton_iterations"])
    r["default_ms_median_T"] = round(float(np.median(walls[cfg["phaseA_steps"]:])), 1)
    r["default_phaseA_umax_mm"] = float(max(q["umax"] for q in ctrl.rows[:cfg["phaseA_steps"]]))
    Sofa.Simulation.unload(root)
    return r


def main():
    quick = "--quick" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    if only and os.path.exists(OUT):           # keep the earlier gate results, update only the requested tests
        prev = json.load(open(OUT)); prev.update(RES); RES.clear(); RES.update(prev)
    for name, fn in (("G0.1", g01), ("G0.2", g02), ("G0.3", g03), ("G0.4", g04), ("G0.6", g06)):
        if (only and name not in only) or (quick and name in ("G0.3",)):
            continue
        t = time.perf_counter()
        try:
            RES[name] = fn()
        except Exception:
            RES[name] = dict(error=traceback.format_exc()[-1500:])
        RES[name + "_s"] = round(time.perf_counter() - t, 1)
        print(name, json.dumps(RES[name], default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:1500], flush=True)
        save()
    if only and "G0.5" not in only:
        save(); print("ONLY", only, "elapsed", round(time.perf_counter() - T0, 1), flush=True)
        return
    RES["G0.5"] = g05(budget_s=430.0 if not quick else 200.0)
    print("G0.5", json.dumps(RES["G0.5"])[:3000], flush=True)
    # pick the fastest linear solver that passed G0.2 at the chosen grid
    g2 = RES.get("G0.2", {}).get("pass", {})
    cand = []
    for key, v in RES["G0.5"].items():
        if isinstance(v, dict) and "nomap" in key:
            ls = key.split("_")[1]
            ok = g2.get(ls if ls != "CGLinearSolver" else "CG200", False)
            if ok:
                cand.append((v["ms_median_T"], key))
    RES["chosen"] = sorted(cand)[0][1] if cand else None
    save()
    print("CHOSEN", RES["chosen"], "elapsed", round(time.perf_counter() - T0, 1))


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # avoid the SofaPython3 teardown segfault after a Python ForceField existed (ties.py)
