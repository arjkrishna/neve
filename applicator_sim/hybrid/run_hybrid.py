"""Headless driver for the Stage 1 hybrid pelvis scene (CONTAINER).  CONTRACT.md section 4 outputs.

    bash applicator_sim/run_docker.sh <tag> hybrid/run_hybrid.py --tag <tag> [--cfg '<json>|<path>']

Writes `<out>/hybrid/runs/<tag>/`:
    cfg.json          the full resolved configuration + provenance (code hashes, image id, versions)
    log.jsonl         one row per step: u, phase, per-body max/mean displacement + dx, contact count,
                      constraint residual/iterations, min tet volume ratio, step wall time
    final/<body>.obj  deformed surfaces, preBT world RAS mm (corpus: the rigid pose-rule placement)
    final/<body>_u.npy  per-node displacement (mm), SAME point order as meshes/<body>/tets.vtk
    device_final.json flange, tube axis, ovoid pose, Delta (preBT world RAS mm)
    summary.json      steps, ms/step, convergence per CONTRACT 5, forces, gate flags
With cfg `frame_every` > 0 also, every `frame_every` steps AND at the final step, `frames/`:
    step_XXXX_<body>.obj    the deformed surface of every body, preBT world RAS mm
    step_XXXX_device.json   phase, u, flange, tube axis, ovoid poses, per-body max/mean |u|, contacts
    index.json              the frame list + the constants (R_rows, colours' source, travel, Delta)
i.e. the input of the host renderer `hybrid/animate_hybrid.py`.  Default 0 = off (behaviour unchanged).
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402

# SOFA (container, py3.8) and scene_hybrid are imported lazily inside run_one, so that the host interpreter
# (`py -3.11 hybrid/run_hybrid.py render --tag <tag>`) can draw the figures without SofaPython3.
S = None

CODE_FILES =("hybrid/scene_hybrid.py", "hybrid/run_hybrid.py", "hybrid/mesh_bodies.py", "hybrid/applicator_venezia.py",
              "geom.py", "config.py")


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def provenance():
    code = {}
    for f in CODE_FILES:
        p = os.path.join(PARENT, f)
        if os.path.exists(p):
            code[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    try:
        import Sofa as _S
        sofa_v = getattr(_S, "__version__", None)
    except Exception:
        sofa_v = None
    return dict(code_sha256_16=code, image_id=os.environ.get("APPSIM_IMAGE_ID"), cpus=os.environ.get("APPSIM_CPUS"),
                numpy=np.__version__, python=sys.version.split()[0], sofapython3=sofa_v,
                sofa="v22.12 (image eve-training-fixed)")


def write_outputs(ctx, ctrl, out, summary):
    """final/<body>.obj + final/<body>_u.npy (tets.vtk order) + device_final.json."""
    P = ctx["inp"]["P"]
    fin = os.path.join(out, "final")
    os.makedirs(fin, exist_ok=True)
    tgt = ctx["tgt"]
    r = ctrl.rows[-1] if ctrl.rows else dict(phase="P", s=0.0)
    disp = {}
    for b in S.BODIES:
        meta = ctx["inp"]["meta"][b]
        V0, F = geom.read_obj("%s/%s/surface.obj" % (P["meshes"], b))
        s2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
        X0 = ctx["X0"][b]
        if b == "corpus":                                   # rigid: the pose-rule placement reached at the last step
            T = np.asarray(ctx["sched"][min(ctrl.k, len(ctx["sched"])) - 1]["T_corpus"], float) \
                if ctrl.k else np.eye(4)
            X = X0 @ T[:3, :3].T + T[:3, 3]
        else:
            X = ctrl.X(b)
        u = X - X0
        np.save(os.path.join(fin, "%s_u.npy" % b), u.astype(np.float64))
        geom.write_obj(os.path.join(fin, "%s.obj" % b), X[s2n], F,
                       header="%s deformed surface, preBT world RAS mm, run %s (local only)" % (b, summary["tag"]))
        du = np.linalg.norm(u, axis=1)
        disp[b] = dict(umax_mm=round(float(du.max()), 4), umean_mm=round(float(du.mean()), 4),
                       nodes=int(len(X0)), surface_verts=int(len(V0)))
    # device
    def rig(node):
        q = np.array(node.rig.position.value, dtype=float)[0]
        return q[:3], q[3:]
    Ft, qt = rig(ctx["tandem"])
    Fo, qo = rig(ctx["ovoids"])
    Fc, qc = rig(ctx["corpus"])
    a = tgt["tube_axis"]
    dev = dict(units="mm; preBT world RAS (x=R, y=A, z=S)", tag=summary["tag"],
               flange_shift_mm=tgt["delta_mm"], flange_shift_note="SCENARIO parameter Delta (pose.json rule v2)",
               flange_mm=Ft.round(4).tolist(), tube_axis=a.round(6).tolist(),
               x_app=tgt["R_rows"][0].round(6).tolist(), y_app=tgt["R_rows"][1].round(6).tolist(),
               tip_mm=(Ft + float(S._param(ctx["inp"]["app"], "L_iu_mm")) * a).round(4).tolist(),
               tandem_quat_xyzw=qt.round(6).tolist(),
               ovoid_origin_mm=Fo.round(4).tolist(), ovoid_quat_xyzw=qo.round(6).tolist(),
               ovoid_lag_mm=round(float((Ft - Fo) @ a), 4), ovoid_mode=ctx["cfg"]["ovoid_mode"],
               ovoid_centres_mm=[(Fo + np.asarray(c, float) @ tgt["R_rows"]).round(4).tolist()
                                 for c in ctx["inp"]["app"]["landmarks"]["cap_centres"]],
               corpus_translation_mm=Fc.round(4).tolist(), corpus_quat_xyzw=qc.round(6).tolist(),
               corpus_T_preBT_to_final=np.asarray(
                   ctx["sched"][min(ctrl.k, len(ctx["sched"])) - 1]["T_corpus"] if ctrl.k else np.eye(4)).round(6).tolist(),
               path_final_u=round(float(r.get("u", 0.0)), 5), corpus_s=round(float(r.get("corpus_s", 0.0)), 5),
               pose_rule="v2 (pose.json): shaft axis = vagina principal axis, flange = base + Delta, "
                         "corpus a0 -> tube axis with L_end d_F above the flange")
    with open(os.path.join(out, "device_final.json"), "w") as fh:
        json.dump(dev, fh, indent=1, default=_json)
    return disp, dev


# ----------------------------------------------------------------------------- per-step frames (animate_hybrid.py)
PHASE_NAMES = dict(P="pre-settle", A="approach", T="insertion", D="ovoid seating", H="settle")


def _sched_at(ctx, k):
    """The schedule row that drove step k (a step past the end of the schedule is a settle step at the final pose)."""
    sched = ctx["sched"]
    return dict(sched[k]) if k < len(sched) else dict(sched[-1], phase="H")


def frame_cache(ctx):
    """Read ONCE per run: every body's surface faces (pre-formatted OBJ text) and its surface-vertex -> tet-node map,
    so a frame costs one gather + one write per body.  A body whose stored surface does not index its tet mesh is
    skipped and the reason recorded in index.json, so a scene variant that rebuilds one body (e.g. a hollow vagina
    wall) still exports the other five instead of aborting the run."""
    P = ctx["inp"]["P"]
    fc = dict(bodies={}, skipped={}, index=[], last_step=None, ms=[])
    for b in S.BODIES:
        try:
            V0, F = geom.read_obj("%s/%s/surface.obj" % (P["meshes"], b))
            s2n = np.asarray(ctx["inp"]["meta"][b]["surface_obj_vertex_to_tet_node"], int)
            n = len(ctx["X0"][b])
            if len(s2n) != len(V0) or int(s2n.max()) >= n:
                raise ValueError("surface.obj (%d verts, max node %d) does not index tets.vtk (%d nodes)"
                                 % (len(V0), int(s2n.max()), n))
            if b != "corpus" and b not in ctx["nodes"]:
                raise ValueError("no mechanical node in the scene")
            fc["bodies"][b] = dict(s2n=s2n, n_tris=int(len(F)),
                                   faces="".join("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1) for f in F))
        except Exception as e:                       # never let the animation export stop a simulation
            fc["skipped"][b] = str(e)
    return fc


def _write_obj_frame(path, V, faces, header):
    """OBJ at 1 um precision (a frame is for drawing, not for evaluation: final/<body>.obj keeps the 0.1 um form)."""
    with open(path, "w") as fh:
        fh.write("# %s\n" % header)
        fh.write("".join("v %.3f %.3f %.3f\n" % (p[0], p[1], p[2]) for p in V))
        fh.write(faces)


def write_frame(ctx, ctrl, out, row, fc):
    """cfg frame_every > 0: dump the deformed surface of every body and the device state of the step just solved.
    Read-out only (the mechanical states are copied, never written), ~0.1 s against a ~5.8 s step."""
    t0 = time.perf_counter()
    fd = os.path.join(out, "frames")
    os.makedirs(fd, exist_ok=True)
    k = int(row["step"])
    r = _sched_at(ctx, k)
    tgt, app = ctx["tgt"], ctx["inp"]["app"]
    a_path, a_tube = tgt["axis"], tgt["tube_axis"]           # path translation axis / intrauterine tube axis
    L_iu = float(S._param(app, "L_iu_mm"))
    F = np.asarray(r["F"], float)
    Fo = F - float(r["ov_lag"]) * a_path
    tip = F + L_iu * a_tube
    # ---- corpus: rigid, placed by the pose rule (it has no mechanical DOFs to read)
    Tc = np.asarray(r["T_corpus"], float)
    X0c = ctx["X0"]["corpus"]
    Xc = X0c @ Tc[:3, :3].T + Tc[:3, 3]
    duc = np.linalg.norm(Xc - X0c, axis=1)
    disp = {b: dict(umax_mm=d["umax"], umean_mm=d["umean"]) for b, d in row["disp"].items()}
    disp["corpus"] = dict(umax_mm=round(float(duc.max()), 4), umean_mm=round(float(duc.mean()), 4))
    # ---- deformed surfaces
    files = {}
    for b, e in fc["bodies"].items():
        X = Xc if b == "corpus" else ctrl.X(b)
        fn = "step_%04d_%s.obj" % (k, b)
        _write_obj_frame(os.path.join(fd, fn), X[e["s2n"]], e["faces"],
                         "step %d phase %s: deformed %s surface, preBT world RAS mm, run %s (local only)"
                         % (k, row["phase"], b, ctx["cfg"].get("tag")))
        files[b] = fn
    # ---- device state (every coordinate read from pose.json / applicator.json, none hard-coded)
    ins = ctx["inp"]["pose"]["inputs"]
    O_pre = np.asarray(ins["O_pre"]["value"], float)
    i_os = np.asarray(ins["internal_os"]["value"], float)
    u = float(r["u"])
    dev = dict(step=k, phase=row["phase"], phase_name=PHASE_NAMES.get(row["phase"], row["phase"]),
               u=round(u, 6), corpus_s=round(float(r["s"]), 6),
               units="mm; preBT world RAS (x=R, y=A, z=S)", tag=ctx["cfg"].get("tag"),
               flange_mm=F.round(4).tolist(), tube_axis=a_tube.round(6).tolist(),
               path_axis=a_path.round(6).tolist(), x_app=tgt["R_rows"][0].round(6).tolist(),
               y_app=tgt["R_rows"][1].round(6).tolist(), tip_mm=tip.round(4).tolist(), L_iu_mm=L_iu,
               r_tandem_mm=float(S._param(app, "r_tandem_mm")),
               ovoid_origin_mm=Fo.round(4).tolist(), ovoid_lag_mm=round(float(r["ov_lag"]), 4),
               ovoid_centres_mm=[(Fo + np.asarray(c, float) @ tgt["R_rows"]).round(4).tolist()
                                 for c in app["landmarks"]["cap_centres"]],
               # inserted depth, three honest readings of the same motion
               advance_mm=round(u * float(tgt["travel_mm"]), 3),           # travelled along the path so far
               remaining_mm=round((1.0 - u) * float(tgt["travel_mm"]), 3),
               tip_beyond_O_pre_mm=round(float((tip - O_pre) @ a_tube), 3),      # past the preBT external os
               tip_beyond_internal_os_mm=round(float((tip - i_os) @ a_tube), 3),
               flange_shift_mm=tgt["delta_mm"], disp=disp,
               n_contacts=row.get("n_contacts"), n_constraint_rows=row.get("n_constraint_rows"),
               constraint_err_per_contact=row.get("constraint_err_per_contact"),
               n_canal_ties=row.get("n_canal_ties"), dx_max_mm=row.get("dx_max_mm"),
               min_vol_ratio=row.get("min_vol_ratio"), wall_ms=row.get("wall_ms"),
               settle_step=row.get("settle_step"))
    dj = "step_%04d_device.json" % k
    with open(os.path.join(fd, dj), "w") as fh:
        json.dump(dev, fh, default=_json)
    fc["index"].append(dict(step=k, phase=row["phase"], phase_name=dev["phase_name"], u=dev["u"],
                            device=dj, surfaces=files))
    fc["ms"].append(1000.0 * (time.perf_counter() - t0))
    with open(os.path.join(fd, "index.json"), "w") as fh:
        json.dump(dict(tag=ctx["cfg"].get("tag"), frame_every=int(ctx["cfg"].get("frame_every") or 0),
                       units="mm; preBT world RAS (x=R, y=A, z=S)", bodies=sorted(fc["bodies"]),
                       skipped_bodies=fc["skipped"], flange_shift_mm=tgt["delta_mm"],
                       travel_mm=float(tgt["travel_mm"]), u_tip_at_internal_os=float(tgt["u_ios"]),
                       R_rows=tgt["R_rows"].round(6).tolist(),
                       R_rows_note="device surfaces: p_world = origin + p_app @ R_rows (applicator/<part>.obj); "
                                   "origin = flange_mm for tube/shaft, ovoid_origin_mm for ovoid_L/R",
                       rest_meshes="meshes/<body>/surface.obj (same faces, same vertex order)",
                       frames=fc["index"]), fh, indent=1, default=_json)
    fc["last_step"] = k


def run_one(cfg, tag):
    global S
    import Sofa.Core
    import Sofa.Simulation
    if S is None:
        import scene_hybrid as S  # noqa: F401
    P = S.paths()
    out = os.path.join(P["runs"], tag)
    os.makedirs(out, exist_ok=True)
    for fn in ("summary.json",):                       # a stale summary must never pass for this run's result
        if os.path.exists(os.path.join(out, fn)):
            os.remove(os.path.join(out, fn))
    cfg = dict(cfg)
    cfg["tag"] = tag
    prov = provenance()
    with open(os.path.join(out, "cfg.json"), "w") as fh:
        json.dump(dict(cfg, _provenance=prov), fh, indent=1, default=_json)
    t0 = time.perf_counter()
    root = Sofa.Core.Node("root")
    inp = S.load_inputs(cfg)
    ctx = S.build_scene(root, cfg, inp)
    ctrl = root.addObject(S.HybridController(name="hybrid", ctx=ctx, out_dir=out))
    Sofa.Simulation.init(root)
    S.post_init(ctx)
    t_init = time.perf_counter() - t0
    info = S.scene_summary(ctx)
    info.update(tag=tag, t_init_s=round(t_init, 2))
    # self-check: the reconstructed device path must reproduce pose.json's stored keyframes (default Delta only)
    if ctx["tgt"]["is_default"]:
        kf = inp["pose"]["insertion_path"]["keyframes"]
        err = 0.0
        for q in kf:
            Fq = ctx["tgt"]["flange"] - (1.0 - float(q["u"])) * ctx["tgt"]["travel_mm"] * ctx["tgt"]["axis"]
            err = max(err, float(np.abs(Fq - np.asarray(q["F"], float)).max()))
        info["path_vs_pose_json_max_mm"] = round(err, 6)
    print("INIT " + json.dumps(info, default=_json), flush=True)
    n_max = int(cfg["max_steps"])
    fe = int(cfg.get("frame_every") or 0)                 # per-step export for animate_hybrid.py; 0 = off (default)
    fc = frame_cache(ctx) if fe > 0 else None
    if fe > 0:
        print("FRAMES every %d step(s): %s%s" % (fe, sorted(fc["bodies"]),
                                                 (" SKIPPED %s" % fc["skipped"]) if fc["skipped"] else ""), flush=True)
    while not ctrl.done and ctrl.k < n_max:
        Sofa.Simulation.animate(root, root.dt.value)
        r = ctrl.rows[-1]
        if fe > 0 and r.get("finite", False) and (r["step"] % fe == 0 or ctrl.done):
            write_frame(ctx, ctrl, out, r, fc)
        if r["step"] % int(cfg["log_every"]) == 0 or ctrl.done:
            print("STEP %3d %s u=%.3f s=%.3f lag=%5.1f wall=%5.0fms dx=%.4f cont=%4d err=%.2e it=%3d minV=%.3f "
                  "umax(cvx/vag/bld/rec/sig)=%.2f/%.2f/%.2f/%.2f/%.2f"
                  % (r["step"], r["phase"], r["u"], r["corpus_s"], r["ov_lag_mm"], r["wall_ms"], r["dx_max_mm"],
                     r["n_contacts"], r["constraint_err"], r["constraint_it"], r["min_vol_ratio"],
                     r["disp"]["cervix"]["umax"], r["disp"]["vagina"]["umax"], r["disp"]["bladder"]["umax"],
                     r["disp"]["rectum"]["umax"], r["disp"]["sigmoid"]["umax"]), flush=True)
    if not ctrl.done and ctrl.status == "running":
        ctrl.status = "max_steps"
    if fe > 0 and ctrl.rows and ctrl.rows[-1].get("finite", False) and fc["last_step"] != ctrl.rows[-1]["step"]:
        write_frame(ctx, ctrl, out, ctrl.rows[-1], fc)     # final step of a max_steps / wall-time exit
    ctrl.close()
    rows = ctrl.rows
    wall = np.array([r["wall_ms"] for r in rows], float) if rows else np.zeros(1)
    fin = rows[-1] if rows else {}
    summary = dict(info)
    summary.update(
        tag=tag, status=ctrl.status, converged=bool(ctrl.status == "converged"), n_steps=len(rows),
        n_settle_steps=int(sum(1 for r in rows if r["phase"] == "H")),
        ms_per_step_median=round(float(np.median(wall)), 1), ms_per_step_mean=round(float(wall.mean()), 1),
        ms_per_step_by_phase={p: round(float(np.median([r["wall_ms"] for r in rows if r["phase"] == p])), 1)
                              for p in "PATDH" if any(r["phase"] == p for r in rows)},
        total_s=round(time.perf_counter() - t0, 1), t_init_s=round(t_init, 2),
        convergence=dict(rule="CONTRACT 5: %d consecutive settle steps with max nodal change < %g mm AND constraint "
                              "residual per contact < %g AND the constraint solver within its iteration cap"
                              % (int(cfg["conv_steps"]), float(cfg["conv_dx_mm"]),
                                 float(cfg["conv_constraint_err_per_contact"])),
                         dx_max_final_mm=fin.get("dx_max_mm"), dx_ratio=fin.get("dx_ratio"),
                         projected_remaining_drift_mm=fin.get("drift_est_mm"),
                         constraint_err_final=fin.get("constraint_err"),
                         constraint_err_per_contact_final=fin.get("constraint_err_per_contact"),
                         constraint_it_final=fin.get("constraint_it"),
                         constraint_solver_converged=fin.get("constraint_converged"),
                         n_consecutive_ok=int(sum(1 for v in ctrl.ok_hist[-int(cfg["conv_steps"]):] if v))),
        final=dict(n_contacts=fin.get("n_contacts"), min_vol_ratio=fin.get("min_vol_ratio"),
                   n_canal_ties=fin.get("n_canal_ties"), u=fin.get("u"), phase=fin.get("phase")),
        gates=dict(nan=bool(ctrl.status == "abort_nan"),
                   inverted_tets=bool(ctrl.status == "abort_inverted_tets"),
                   wall_time=bool(ctrl.status == "abort_wall_time"),
                   min_vol_ratio_run=round(float(min([r["min_vol_ratio"] for r in rows] or [1.0])), 4),
                   min_vol_ratio_ok=bool(min([r["min_vol_ratio"] for r in rows] or [1.0])
                                         >= float(cfg["min_vol_ratio_abort"])),
                   all_finite=bool(all(r["finite"] for r in rows)),
                   reached_final_pose=bool(fin.get("u", 0.0) >= 0.999 and fin.get("corpus_s", 0.0) >= 0.999),
                   constraint_solver_converged_frac=round(
                       float(np.mean([1.0 if r["constraint_converged"] else 0.0 for r in rows])), 3) if rows else None),
        frames=(dict(n=len(fc["index"]), every=fe, dir="frames", bodies=sorted(fc["bodies"]),
                     skipped_bodies=fc["skipped"], ms_per_frame_median=round(float(np.median(fc["ms"])), 1),
                     ms_per_frame_total=round(float(np.sum(fc["ms"])), 1)) if fe > 0 and fc["ms"] else None),
        cfg={k: v for k, v in cfg.items() if not k.startswith("_")}, provenance=prov)
    if fin.get("finite", False):
        disp, dev = write_outputs(ctx, ctrl, out, summary)
        summary["displacement"] = disp
        summary["device_final"] = {k: dev[k] for k in ("flange_mm", "tube_axis", "tip_mm", "ovoid_origin_mm",
                                                       "ovoid_lag_mm", "flange_shift_mm")}
    else:
        summary["displacement"] = None
        print("NON-FINITE state: final/ outputs not written", flush=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json)
    print("DONE " + json.dumps(dict(tag=tag, status=summary["status"], n_steps=summary["n_steps"],
                                    ms_per_step_median=summary["ms_per_step_median"], total_s=summary["total_s"],
                                    convergence=summary["convergence"], gates=summary["gates"]), default=_json),
          flush=True)
    return summary


# ----------------------------------------------------------------------------- host-side verification figures
BODIES = ["corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"]
DEV_PARTS = ["tube", "shaft", "ovoid_L", "ovoid_R"]


def _paths_host():
    import config
    out = config.paths()["out"] + "/hybrid"
    return dict(hybrid=out, meshes=out + "/meshes", applicator=out + "/applicator", runs=out + "/runs",
                figs=out + "/figs", logs=out + "/logs")


def _dev_world(P, origin, R_rows, parts=DEV_PARTS):
    """Device surfaces carried from the applicator frame into preBT world: p = origin + p_app @ R_rows."""
    out = {}
    for p in parts:
        V, F = geom.read_obj("%s/%s.obj" % (P["applicator"], p))
        out[p] = (np.asarray(origin, float) + V @ np.asarray(R_rows, float), F)
    return out


def _min_dist(Va, Fa, Vb, Fb):
    """Minimum signed distance of surface A's vertices to closed surface B (negative = A inside B)."""
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(Vb, float), deep=1))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(Fb), ns.numpy_to_vtkIdTypeArray(
        np.c_[np.full(len(Fb), 3), Fb].astype(np.int64).ravel(), deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(poly)
    return float(min(f.EvaluateFunction([float(v[0]), float(v[1]), float(v[2])]) for v in Va))


def render(tag):
    """py -3.11: off-screen pyvista figures of the rest state and of the converged state (CONTRACT 7.1 check)."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    P = _paths_host()
    os.makedirs(P["figs"], exist_ok=True)
    idx = json.load(open(P["meshes"] + "/bodies.json"))
    pose = json.load(open(P["applicator"] + "/pose.json"))
    dev = json.load(open("%s/%s/device_final.json" % (P["runs"], tag)))
    R_rows = np.array([dev["x_app"], dev["y_app"], dev["tube_axis"]], float)
    states = {}
    rest = {b: geom.read_obj("%s/%s/surface.obj" % (P["meshes"], b)) for b in BODIES}
    F0 = np.array(pose["insertion_path"]["device_at_u0"]["F"], float)
    states["initial"] = dict(bodies=rest, dev=_dev_world(P, F0, R_rows), note="rest state (preBT), device at u = 0")
    fin = {b: geom.read_obj("%s/%s/final/%s.obj" % (P["runs"], tag, b)) for b in BODIES}
    dv = _dev_world(P, dev["flange_mm"], R_rows, ["tube", "shaft"])
    dv.update(_dev_world(P, dev["ovoid_origin_mm"], R_rows, ["ovoid_L", "ovoid_R"]))
    states["final"] = dict(bodies=fin, dev=dv, note="converged state, Delta = %g mm" % dev["flange_shift_mm"])
    # ---- interpenetration report (the enabled contact pairs must not end up inside each other)
    rep = {}
    for i, a in enumerate(BODIES):
        for b in BODIES[i + 1:]:
            rep["%s|%s" % (a, b)] = dict(
                rest=round(min(_min_dist(rest[a][0], rest[a][1], rest[b][0], rest[b][1]),
                               _min_dist(rest[b][0], rest[b][1], rest[a][0], rest[a][1])), 3),
                final=round(min(_min_dist(fin[a][0], fin[a][1], fin[b][0], fin[b][1]),
                                _min_dist(fin[b][0], fin[b][1], fin[a][0], fin[a][1])), 3))
    for p in ("ovoid_L", "ovoid_R", "tube"):
        for b in ("cervix", "vagina", "rectum", "bladder"):
            rep["%s|%s" % (p, b)] = dict(final=round(_min_dist(fin[b][0], fin[b][1], dv[p][0], dv[p][1]), 3))
    with open("%s/scene_%s_contacts.json" % (P["logs"], tag), "w") as fh:
        json.dump(dict(tag=tag, units="mm; min signed distance between surfaces, negative = interpenetration",
                       pairs=rep), fh, indent=1)
    print("[render] min signed distance (mm), negative = interpenetration:")
    for k, v in sorted(rep.items()):
        print("   %-18s rest %8s  final %8s" % (k, v.get("rest", "-"), v["final"]))
    # ---- figures
    views = [((1.0, 1.6, 0.9), "anterior-right-superior oblique"), ((-1.0, 0.0, 0.12), "left lateral (sagittal)")]
    allpts = np.vstack([rest[b][0] for b in BODIES] + [fin[b][0] for b in BODIES])
    ctr = 0.5 * (allpts.min(0) + allpts.max(0))
    ext = float(np.linalg.norm(allpts.max(0) - allpts.min(0)))
    for name, st in states.items():
        pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000), border=True)
        pl.set_background("white")
        for k, (vec, vname) in enumerate(views):
            pl.subplot(0, k)
            for b in BODIES:
                V, F = st["bodies"][b]
                pl.add_mesh(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()), color=idx["bodies"][b]["color"],
                            opacity=0.45 if b in ("bladder", "sigmoid") else 0.95, show_edges=False,
                            smooth_shading=True, label=b if k == 0 else None)
            for p, (V, F) in st["dev"].items():
                pl.add_mesh(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()),
                            color=(0.10, 0.10, 0.14) if p in ("tube", "shaft") else (0.45, 0.45, 0.52),
                            smooth_shading=True, label=p if k == 0 else None)
            if k == 0:
                pl.add_legend(bcolor="white", face=None, size=(0.17, 0.30), loc="upper left")
            pl.add_axes(xlabel="x=R", ylabel="y=A", zlabel="z=S")
            pl.add_text("%s -- %s\n%s" % (tag, name, vname), font_size=10, position="lower_left")
            v = np.asarray(vec, float) / np.linalg.norm(vec)
            pl.camera_position = [(ctr + 2.1 * ext * v).tolist(), ctr.tolist(), (0, 0, 1)]
            pl.camera.zoom(1.35)
        fn = "%s/scene_%s_%s.png" % (P["figs"], tag, name)
        pl.screenshot(fn)
        pl.close()
        print("[render] wrote", fn)
    # ---- rest (wireframe) vs converged (solid), sagittal + oblique
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000), border=True)
    pl.set_background("white")
    for k, (vec, vname) in enumerate(views):
        pl.subplot(0, k)
        for b in BODIES:
            V, F = rest[b]
            pl.add_mesh(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()), style="wireframe",
                        color=idx["bodies"][b]["color"], opacity=0.35, line_width=1)
            V, F = fin[b]
            pl.add_mesh(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()), color=idx["bodies"][b]["color"],
                        opacity=0.85, smooth_shading=True)
        for p, (V, F) in states["final"]["dev"].items():
            pl.add_mesh(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()), color=(0.10, 0.10, 0.14),
                        smooth_shading=True)
        pl.add_axes(xlabel="x=R", ylabel="y=A", zlabel="z=S")
        pl.add_text("%s -- rest (wireframe) vs converged (solid)\n%s" % (tag, vname), font_size=10,
                    position="lower_left")
        v = np.asarray(vec, float) / np.linalg.norm(vec)
        pl.camera_position = [(ctr + 2.1 * ext * v).tolist(), ctr.tolist(), (0, 0, 1)]
        pl.camera.zoom(1.35)
    fn = "%s/scene_%s_compare.png" % (P["figs"], tag)
    pl.screenshot(fn)
    pl.close()
    print("[render] wrote", fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "render"],
                    help="run = the SOFA scene (container); render = verification figures (host, py -3.11)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--cfg", default=None, help="inline JSON dict of overrides, or a path to a JSON file")
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.tag)
        return
    import scene_hybrid as _S
    run_one(_S.load_cfg(a.cfg), a.tag)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                      # SofaPython3 v22.12 can segfault in interpreter teardown; outputs are closed
