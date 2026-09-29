"""Cervix-region-alone probe: the FEM cervix (HR-CTV minus uterus) and the corpus under the applicator_v5 device --
tandem first, then each lunar ring half along its approach onto its seat, then the lift -- with no vagina and no organs
at risk.  Built for the investigator's request ("model the deformation of the cervix region alone first, see how it fits
the actual data, then include the effects of the tandem on the uterus") and to choose the cervix material for TF1
(fix plan S7f; S2 probe P6 "cervix against a kinematic ring").

    # CONTAINER (SOFA v22.12), from the repo dir:
    APPSIM_TIMEOUT=3600 APPSIM_CPUS=2 bash run_docker_par.sh <TAG> hybrid/cervix_probe.py run --tag <TAG> \
        --cfg /out/hybrid/runs/_cfg/<TAG>.json
    # HOST (python -P, 3.13: numpy / scipy / nibabel / vtk):
    python -P -B hybrid/cervix_probe.py selftest                        # the pure schedule / monitor arithmetic
    python -P -B hybrid/cervix_probe.py score --tag <TAG> [--control <TAG0>] [--m121 <k*>] [--rigid-carry]
                                              [--regions-from <applicator dir>] [--out <json>] [--quiet]
        -> eval/<TAG>/cervix_probe.json (patient-derived: local only).  Works on any run with final/cervix_u.npy (e.g.
        TF1v with --m121 294 --regions-from applicator_v5); --rigid-carry scores the REST cervix carried by the run's
        final corpus target instead (the K0 comparator).

THE SCENE.  scene_hybrid.build_scene with an ISOLATED cfg, which is CHECKED here and refused otherwise (never silently
rewritten): vagina_model "solid" whose collision groups share an id with the cervix (no contact), bladder / rectum /
sigmoid in static_bodies and likewise non-colliding, apex_attach "off", n_balloon 0, insertion_path "tandem_first",
ovoid_mode "none" -- the EUISO_* / EU3_* isolated scene of Stage 3c.  The solid vagina is still integrated (inert: no
load reaches it); nothing else touches the cervix but the corpus attach, the canal ties, the cardinal cables and the
ring.  Added here: two kinematic Rigid3d bodies ring_L / ring_R carrying the applicator's ring halves
(<applicator_dir>/applicator.json ring.approach.halves[side].part, OBJs already at their seat in the applicator
frame), collision groups probe.ring_groups, which must be DISJOINT from the cervix's and share an id with every other
model (tandem, corpus, the non-colliding bodies, each other): the ring meets the cervix and nothing else.  The tandem
still meets no tissue (the canal ties carry it, as in every Stage-3 run).

THE SCHEDULE.  The scene's own tandem-first rows (V, C, L from pose.json insertion_path_tandem_first) with the ring
phases inserted between C and L, in applicator.json ring.approach.order (L first, then R: plan S7f).  Each half follows
its approach path d (disp(d) = d e_app + o(d) n_out_app, o(d) = offset for d >= click_mm, offset (1 - (1 - d /
click_mm)^p) below; world = F + (p_app + disp(d)) @ R_rows), the tandem and corpus held at the last C row:
  probe.seat "joint" (DEFAULT):
    RL  the left half from D0_mm to the stand-off d = standoff_mm (clear of the portio);
    RR  the right half likewise, the left one waiting at the stand-off;
    RS  both halves together from the stand-off onto the seat (the assembled ring pushed onto the portio), n_hold rows;
  probe.seat "sequential": RL the left half all the way onto its seat (+ n_hold), then RR the right one;
  L   the record's lift rows: tandem, both halves (at the seat, riding the tandem) and the corpus translate together;
  H   the settle (probe.drain: the cervix switches to its drained material at the first H row, see PROBE).
The steps along an approach are equal in PATH LENGTH of disp (the largest over the halves that move): step_far_mm while
d > d_near_mm, step_near_mm below (the contact zone; the lateral click-in is part of the path, so no step moves a half
more than that).  Before its own phase a half is parked at d = D0_mm riding the tandem (MEASURED on EU3_C's V / C
frames: >= 19 mm from the cervix at D0 30).  probe.ring false = the tandem-only control: the same rows (R rows are
holds), no ring bodies.  MEASURED (sequential, CP0): the left half alone lifts its side of the hanging portio while the
right side stays down, the tets across the slot edge shear to J 0.08 and the run aborts; the joint seat completes.

READ-OUTS (per step, log.jsonl row "probe"): each half's d, a nearest-vertex signed-distance monitor of the cervix
surface nodes against each half (count < 0 / < -0.5 mm, minimum; approximate, the host re-measures the saved states
exactly), the cervix total volume ratio and per-tet volume ratio percentiles (J).  Frames (cfg frame_every, plus the
last row of every phase and the final step) also get frames/step_<k>_ring.json.  Final: run_hybrid.write_outputs
(final/<body>.obj, final/<body>_u.npy, device_final.json) + final/ring.json.

SCORE (host).  In the PELVIS frame (validation/alignment.json frames.BONE) and the device frame E_app (the run's final
tube superimposed on the real BT tandem; eval_hybrid.Ctx.eapp_map): cervix Dice / MSD / HD95 against BT HR-CTV minus
uterus (eval_hybrid's reference) and the tf_metrics Dice triple (minus vagina, minus vagina | applicator | ovoid); the
HR-CTV\\U left-right offset about the model tube (tf_metrics.lr_side: tet-volume-weighted x offset, the definition of
TF1v's +3.95 mm) with its BT references; cervix volume (tets) and per-tet J; the label volumes preBT / BT; and by
REGION of the final own-flange frame (over the ring = within the ring's outer radius of its centre and below ring top
+ REGION_LOWER_H_MM; beside the ring = outside that radius, same heights; upper = above): the non-rigid displacement
(final minus the corpus target's rigid carry of the rest cervix), the change against a --control run, the volume ratio,
and the per-region surface distances to the BT HR-CTV\\U surface (model vertices -> BT, BT surface voxels -> model).

No patient-derived constant is written in this file: every coordinate comes from the meshes, the applicator dir, the
labels and the run.  Outputs are patient-derived: local only.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True                      # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402

VERSION = "cervix_probe/1.0"

# probe keys (cfg "probe": {...}); everything else in the cfg is a scene_hybrid key
PROBE = dict(
    ring=True,              # False: the tandem-only control (same rows; the R rows are holds; no ring bodies)
    seat="joint",           # "joint" (DEFAULT): each half in order travels its approach to the stand-off d = standoff_mm
                            #   (phases RL, RR), then BOTH travel together from there onto the seat (phase RS): the
                            #   assembled ring is pushed onto the portio.  "sequential": each half all the way to its seat
                            #   in turn (RL, RR).  MEASURED (CP0, sequential): the left half alone pushes its side of the
                            #   portio ~8-10 mm up while the right side still hangs below the ring top, the tets across the
                            #   slot edge (x_app ~ 0, at the ring top) shear to J 0.08 and the run aborts at the start of RR;
                            #   the halves never come closer than the 0.5 mm slot when moved together (host check).
    standoff_mm=14.0,       # "joint": the stand-off d (MEASURED on EU3_C's end-of-C cervix: first contact at d ~ 10-13)
    D0_mm=30.0,             # approach start = park distance along each half's path (disp(d), applicator.json ring.approach)
    step_far_mm=2.0,        # path length of disp per step while d > d_near_mm
    step_near_mm=0.5,       # ... and while d <= d_near_mm (the contact zone; G2's safe rate <= 1.4 mm/step, P6 0.7-1.4)
    d_near_mm=14.0,
    n_hold=5,               # rows held at the seat after each half ("sequential") or after the joint seat
    ring_groups=[9, 12],    # the ring bodies' collision groups (two models collide iff their sets are DISJOINT)
    monitor_reach_mm=6.0,   # penetration monitor: only cervix surface nodes this close to a ring vertex are measured
    drain=None,             # None, or {"E_kPa": E_d, "nu": nu_d}: at the first settle row (H) the cervix's corotational FEM
                            #   switches to this (drained) material (Data youngModulus / poissonRatio + reinit(), which
                            #   rebuilds the per-tet stiffness from the REST positions) and the settle re-equilibrates under
                            #   the same device, ties and attach: the long-time (drained) limit of a poroelastic cervix whose
                            #   insertion ran undrained (the investigator: incompressible at first, compressible once the
                            #   fluid has left the loaded regions, minutes later).  Linear elasticity is path-independent, so
                            #   the drained end state does not need the drained material during the insertion -- where the
                            #   low bulk modulus lets the canal ties' point loads collapse tets (MEASURED: CP2, E 12 / nu
                            #   0.25 from the start, aborted in C at J 0.17).
)
ISOLATED_OARS = ("bladder", "rectum", "sigmoid")
RING_PHASE = dict(L="RL", R="RR")
PHASE_NAMES = dict(RL="left ring half along its approach", RR="right ring half along its approach",
                   RS="the assembled ring onto its seat (both halves together)")


# ============================================================================================ pure (host-testable)
def probe_cfg(d):
    """PROBE updated by the cfg's "probe" dict; unknown keys and wrong types are refused."""
    p = json.loads(json.dumps(PROBE))
    d = dict(d or {})
    bad = sorted(k for k in d if k not in PROBE)
    if bad:
        raise ValueError("unknown probe keys %s (have %s)" % (bad, sorted(PROBE)))
    p.update(d)
    if not isinstance(p["ring"], bool):
        raise ValueError("probe.ring must be true or false (got %r)" % (p["ring"],))
    for k in ("D0_mm", "step_far_mm", "step_near_mm", "d_near_mm", "monitor_reach_mm"):
        p[k] = float(p[k])
        if not p[k] > 0.0:
            raise ValueError("probe.%s must be > 0 (got %r)" % (k, p[k]))
    p["n_hold"] = int(p["n_hold"])
    if p["seat"] not in ("joint", "sequential"):
        raise ValueError("probe.seat must be 'joint' or 'sequential' (got %r)" % (p["seat"],))
    p["standoff_mm"] = float(p["standoff_mm"])
    if p["seat"] == "joint" and not 0.0 < p["standoff_mm"] < p["D0_mm"]:
        raise ValueError("probe.standoff_mm must lie in (0, D0_mm) (got %r)" % p["standoff_mm"])
    if p["drain"] is not None:
        dr = dict(p["drain"])
        if sorted(dr) != ["E_kPa", "nu"]:
            raise ValueError("probe.drain must be null or {'E_kPa': ..., 'nu': ...} (got %r)" % (p["drain"],))
        dr = dict(E_kPa=float(dr["E_kPa"]), nu=float(dr["nu"]))
        if not (dr["E_kPa"] > 0.0 and -1.0 < dr["nu"] < 0.5):
            raise ValueError("probe.drain: E_kPa > 0 and -1 < nu < 0.5 (got %r)" % (dr,))
        p["drain"] = dr
    p["ring_groups"] = [int(g) for g in p["ring_groups"]]
    if not p["ring_groups"]:
        raise ValueError("probe.ring_groups must not be empty (an empty group collides with everything)")
    return p


def ring_disp(h, d):
    """A ring half's approach displacement in the applicator frame at distance d (mm) from its seat:
    d e_app + o(d) n_out_app, o(d) = offset_mm for d >= click_mm, offset_mm (1 - (1 - d / click_mm)^click_shape_p) below
    (applicator.json ring.approach.definition).  d scalar -> (3,), d (m,) -> (m, 3)."""
    d = np.asarray(d, float)
    e = np.asarray(h["e_app"], float)
    n = np.asarray(h["n_out_app"], float)
    off, ck, p = float(h["offset_mm"]), float(h["click_mm"]), float(h["click_shape_p"])
    if not ck > 0.0:
        raise ValueError("ring approach click_mm must be > 0 (got %r)" % ck)
    t = np.clip(d / ck, 0.0, 1.0)
    o = off * (1.0 - (1.0 - t) ** p)
    return d[..., None] * e + o[..., None] * n


def approach_d(hs, d_hi, d_lo, step_far, step_near, d_near, res=0.005):
    """The d values of the approach rows of one or more halves moving together (hs: a list of approach dicts), from
    d_hi (excluded: where they are) down to d_lo (included), equally spaced in PATH LENGTH: a new row whenever the largest
    ring_disp path since the last row reaches step_far (d > d_near) or step_near (d <= d_near), and one row exactly where
    the near zone starts.  Returns (d list, per-row path steps, per-row start d)."""
    d_hi, d_lo = float(d_hi), float(d_lo)
    n = max(2, int(np.ceil((d_hi - d_lo) / float(res))))
    dd = np.linspace(d_hi, d_lo, n + 1)
    seg = np.max([np.linalg.norm(np.diff(ring_disp(h, dd), axis=0), axis=1) for h in hs], axis=0)
    out, steps, start, acc, d0 = [], [], [], 0.0, d_hi
    for i in range(1, len(dd)):
        acc += float(seg[i - 1])
        st = float(step_near) if dd[i] <= float(d_near) else float(step_far)
        cross = dd[i] <= float(d_near) < dd[i - 1]      # a row exactly where the near zone starts
        if acc >= st - 1e-12 or cross or i == len(dd) - 1:
            out.append(float(dd[i]))
            steps.append(acc)
            start.append(d0)
            acc, d0 = 0.0, float(dd[i])
    out[-1] = d_lo
    return out, steps, start


def probe_schedule(sched, halves, order, pcfg):
    """The scene's tandem-first rows with the ring phases inserted after the last C row (see the module doc).  Every row
    gets ring_d = {side: d}.  Rows are shallow copies (their arrays are shared, never written).  Returns (rows, info)."""
    ph = [r["phase"] for r in sched]
    if "C" not in ph or "L" not in ph:
        raise ValueError("the probe needs the tandem-first phases C and L (got %s)" % sorted(set(ph)))
    iC = max(i for i, p in enumerate(ph) if p == "C")
    if any(p != "L" for p in ph[iC + 1:]):
        raise ValueError("the probe expects the L rows right after the last C row (got %s)" % sorted(set(ph[iC + 1:])))
    D0 = float(pcfg["D0_mm"])
    joint = pcfg.get("seat", "joint") == "joint"
    d_stop = float(pcfg["standoff_mm"]) if joint else 0.0
    park = {s: D0 for s in order}
    rows = []
    for r in sched[:iC + 1]:
        rows.append(dict(r, ring_d=dict(park)))
    last = sched[iC]
    info = dict(D0_mm=D0, seat=("joint" if joint else "sequential"), standoff_mm=(d_stop if joint else None),
                order=list(order), halves={}, insert_after_row=int(iC))

    def add(sides, ds, phase, n_hold):
        n0 = len(rows)
        for d in list(ds) + [ds[-1]] * int(n_hold):
            c = dict(rows[-1]["ring_d"])
            for s_ in sides:
                c[s_] = float(d)
            rows.append(dict(last, phase=phase, stage=phase, ring_d=c))
        return [n0, len(rows) - 1]

    def stats(st, start):
        near = [x for x, d in zip(st, start) if d <= pcfg["d_near_mm"]]
        return dict(max_step_mm=round(float(max(st)), 4), max_step_near_mm=round(float(max(near or [0.0])), 4))

    for s in order:
        ds, st, start = approach_d([halves[s]], D0, d_stop, pcfg["step_far_mm"], pcfg["step_near_mm"], pcfg["d_near_mm"])
        rr = add([s], ds, RING_PHASE[s], 0 if joint else pcfg["n_hold"])
        info["halves"][s] = dict(phase=RING_PHASE[s], part=halves[s].get("part"), rows=rr, n_approach=len(ds),
                                 d_end=ds[-1], **stats(st, start))
    if joint:
        ds, st, start = approach_d([halves[s] for s in order], d_stop, 0.0, pcfg["step_far_mm"], pcfg["step_near_mm"],
                                   pcfg["d_near_mm"])
        rr = add(list(order), ds, "RS", pcfg["n_hold"])
        info["seat_together"] = dict(phase="RS", rows=rr, n_approach=len(ds), n_hold=int(pcfg["n_hold"]), **stats(st, start))
    seated = {s: 0.0 for s in order}
    for r in sched[iC + 1:]:
        rows.append(dict(r, ring_d=dict(seated)))
    info["n_rows"] = len(rows)
    info["phases"] = {p: sum(1 for r in rows if r["phase"] == p) for p in sorted(set(r["phase"] for r in rows))}
    return rows, info


def ring_origin(row, h, d):
    """World origin of a ring half's rigid body (its OBJ is in the applicator frame at the seat):
    F + disp(d) @ R_rows, so that p_world = origin + p_app @ R_rows."""
    return np.asarray(row["F"], float) + ring_disp(h, float(d)) @ np.asarray(row["R_rows"], float)


def check_isolated(cfg, groups, ring_groups):
    """Refuse a cfg that is not the isolated cervix scene (see the module doc).  groups = scene_hybrid.groups_for(cfg)."""
    err = []
    if cfg.get("vagina_model", "solid") != "solid":
        err.append("vagina_model must be 'solid' (got %r)" % cfg.get("vagina_model"))
    if cfg.get("apex_attach") != "off":
        err.append("apex_attach must be 'off' (got %r): the vagina must not load the cervix" % cfg.get("apex_attach"))
    if int(cfg.get("n_balloon") or 0) != 0:
        err.append("n_balloon must be 0 (got %r)" % cfg.get("n_balloon"))
    miss = [b for b in ISOLATED_OARS if b not in (cfg.get("static_bodies") or [])]
    if miss:
        err.append("static_bodies must hold %s (missing %s)" % (list(ISOLATED_OARS), miss))
    if cfg.get("insertion_path") != "tandem_first":
        err.append("insertion_path must be 'tandem_first' (got %r)" % cfg.get("insertion_path"))
    if cfg.get("ovoid_mode") != "none":
        err.append("ovoid_mode must be 'none' (got %r): the ring halves are the probe's own bodies" % cfg.get("ovoid_mode"))
    cg = set(groups["cervix"])
    for b in ("vagina",) + ISOLATED_OARS:
        if not set(groups[b]) & cg:
            err.append("%s (groups %s) would collide with the cervix (%s): share an id" % (b, groups[b], sorted(cg)))
    rg = set(int(g) for g in ring_groups)
    if rg & cg:
        err.append("ring_groups %s share an id with the cervix %s: the ring would not touch it" % (sorted(rg), sorted(cg)))
    for m, g in sorted(groups.items()):
        if m != "cervix" and not (set(g) & rg):
            err.append("ring_groups %s are disjoint from %s %s: the ring would collide with it" % (sorted(rg), m, g))
    if err:
        raise ValueError("cervix_probe: not the isolated cervix scene:\n  - " + "\n  - ".join(err))


def vertex_normals(V, F):
    """Area-weighted unit vertex normals (orientation = the triangles' winding)."""
    V = np.asarray(V, float)
    T = np.asarray(F, np.int64)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for j in range(3):
        np.add.at(N, T[:, j], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def mesh_signed_volume(V, F):
    V = np.asarray(V, float)
    T = np.asarray(F, np.int64)
    return float(np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6.0)


def approx_sd(Pq, V, N, tree, reach):
    """Nearest-vertex signed distance of the points Pq (the half's LOCAL frame) to a closed surface (V, outward N):
    (p - v_j) . n_j, NaN beyond `reach`.  A monitor only (error up to ~half an edge tangentially)."""
    dist, j = tree.query(np.asarray(Pq, float), distance_upper_bound=float(reach))
    sd = np.full(len(Pq), np.nan)
    ok = np.isfinite(dist)
    if ok.any():
        jj = j[ok]
        sd[ok] = np.einsum("ij,ij->i", np.asarray(Pq, float)[ok] - V[jj], N[jj])
    return sd


def tet_vol(X, T):
    A = np.asarray(X, float)[np.asarray(T, np.int64)]
    return np.einsum("ij,ij->i", np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), A[:, 3] - A[:, 0]) / 6.0


def j_stats(v, v0):
    """Per-tet volume ratio J summary (percentiles, fractions) and the total volume ratio."""
    J = np.asarray(v, float) / np.asarray(v0, float)
    return dict(total=round(float(np.sum(v) / np.sum(v0)), 5),
                p1=round(float(np.percentile(J, 1)), 4), p5=round(float(np.percentile(J, 5)), 4),
                p50=round(float(np.percentile(J, 50)), 4), p95=round(float(np.percentile(J, 95)), 4),
                min=round(float(J.min()), 4), frac_lt_0p8=round(float(np.mean(J < 0.8)), 5),
                frac_lt_0p5=round(float(np.mean(J < 0.5)), 5))


# ============================================================================================ SOFA (container)
def build_probe(root, cfg, pcfg, S):
    """scene_hybrid.build_scene + the ring bodies + the probe schedule.  Returns ctx (ctx["probe"] holds the rest)."""
    check_isolated(cfg, S.groups_for(cfg), pcfg["ring_groups"])
    inp = S.load_inputs(cfg)
    appr = ((inp["app"].get("ring") or {}).get("approach") or {})
    if not appr.get("halves"):
        raise ValueError("applicator %r has no ring.approach (build it: python -P hybrid/applicator_venezia.py --variant v5 "
                         "--ring)" % cfg.get("applicator_dir"))
    order = list(appr.get("order") or sorted(appr["halves"]))
    halves = {s: appr["halves"][s] for s in order}
    ctx = S.build_scene(root, cfg, inp)
    sched, info = probe_schedule(ctx["sched"], halves, order, pcfg)
    ctx["sched"] = sched
    bodies, mon = {}, {}
    if pcfg["ring"]:
        from scipy.spatial import cKDTree
        r0 = sched[0]
        R0 = np.asarray(r0["R_rows"], float)
        for s in order:
            part = halves[s]["part"]
            f = "%s/%s.obj" % (inp["P"]["applicator"], part)
            bodies[s] = S.add_rigid_parts(root, "ring_" + s, [(part, f)], {part: list(pcfg["ring_groups"])},
                                          {part: [0.35, 0.35, 0.42, 1.0]},
                                          S.rigid_pose(R0.T, ring_origin(r0, halves[s], r0["ring_d"][s])))
            V, F = geom.read_obj(f)
            V = np.asarray(V, float)
            sv = mesh_signed_volume(V, F)
            N = vertex_normals(V, F) * (1.0 if sv > 0 else -1.0)      # outward
            mon[s] = dict(V=V, N=N, tree=cKDTree(V), part=part, signed_volume_mm3=round(sv, 2))
        info["ring_bodies"] = {s: dict(part=mon[s]["part"], n_vertices=int(len(mon[s]["V"])),
                                       signed_volume_mm3=mon[s]["signed_volume_mm3"]) for s in order}
    info["ring"] = bool(pcfg["ring"])
    info["ring_groups"] = list(pcfg["ring_groups"])
    info["cfg"] = dict(pcfg)
    surf = np.asarray(inp["meta"]["cervix"]["node_sets"]["surface_nodes"], int)
    ctx["probe"] = dict(cfg=pcfg, halves=halves, order=order, bodies=bodies, mon=mon, surf=surf, info=info)
    ctx["extra"]["cervix_probe"] = info
    return ctx


def make_controller(S):
    """scene_hybrid.HybridController + the ring poses (after its own kinematics) and the probe read-outs, appended to
    each log row before it is written."""
    class ProbeController(S.HybridController):
        def __init__(self, *args, **kw):
            S.HybridController.__init__(self, *args, **kw)
            self._plog, self._log = self._log, None     # the row is written here, after the probe block is added
            self.drained = None

        def _drain(self):
            """probe.drain: the cervix FEM to the drained material, once, at the first settle row (see PROBE)."""
            pr = self.ctx["probe"]
            dr = pr["cfg"]["drain"]
            ff = self.ctx["nodes"]["cervix"].fem
            before = dict(E_kPa=float(ff.youngModulus.value), nu=float(ff.poissonRatio.value))
            ff.youngModulus.value = float(dr["E_kPa"])
            ff.poissonRatio.value = float(dr["nu"])
            ff.reinit()
            self.drained = dict(step=int(self.k), before=before,
                                after=dict(E_kPa=float(ff.youngModulus.value), nu=float(ff.poissonRatio.value)))
            print("[cervix_probe] DRAIN at step %d: %s" % (self.k, json.dumps(self.drained)), flush=True)

        def _begin(self):
            pr = self.ctx["probe"]
            if pr["cfg"]["drain"] is not None and self.drained is None and self.row_now()["phase"] == "H":
                self._drain()                           # before the step's solve: the settle re-equilibrates drained
            S.HybridController._begin(self)
            if pr["bodies"]:
                r = self.cur
                Rr = np.asarray(r["R_rows"], float)
                for s, nd in pr["bodies"].items():
                    self._set_rigid(nd, Rr.T, ring_origin(r, pr["halves"][s], r["ring_d"][s]))

        def _end(self):
            S.HybridController._end(self)
            row = self.rows[-1]
            try:
                row["probe"] = self._probe_row()
            except Exception as ex:                     # a read-out: never let it stop the run
                row["probe"] = dict(error="%s: %s" % (type(ex).__name__, ex))
            if self._plog:
                self._plog.write(json.dumps(row) + "\n")
                self._plog.flush()

        def _probe_row(self):
            c, pr = self.ctx, self.ctx["probe"]
            r = self.cur
            out = dict(ring_d={s: round(float(r["ring_d"][s]), 4) for s in pr["order"]})
            if self.drained is not None:
                out["drained"] = self.drained["after"]
            Xc = self.X_prev["cervix"]                  # the state just solved (HybridController._end copied it)
            if not np.all(np.isfinite(Xc)):
                return out
            if c["tets"].get("cervix") is not None:
                out["cervix_J"] = j_stats(tet_vol(Xc, c["tets"]["cervix"]), c["vol0"]["cervix"])
            if pr["bodies"]:
                Xs = Xc[pr["surf"]]
                Rr = np.asarray(r["R_rows"], float)
                reach = float(pr["cfg"]["monitor_reach_mm"])
                mon = {}
                for s in pr["order"]:
                    m = pr["mon"][s]
                    o = ring_origin(r, pr["halves"][s], r["ring_d"][s])
                    sd = approx_sd((Xs - o) @ Rr.T, m["V"], m["N"], m["tree"], reach)
                    v = sd[np.isfinite(sd)]
                    mon[s] = dict(n_near=int(len(v)), n_lt_0=int((v < 0.0).sum()), n_lt_m0p5=int((v < -0.5).sum()),
                                  sd_min=(round(float(v.min()), 3) if len(v) else None),
                                  n_within_alarm=int((v < float(self.cfg["alarm_mm"])).sum()))
                out["ring_monitor"] = mon
            return out

        def close(self):
            if self._plog:
                self._plog.close()
                self._plog = None
            S.HybridController.close(self)

    return ProbeController


def _ring_state(ctx, row):
    pr = ctx["probe"]
    R = np.asarray(row["R_rows"], float)
    return {s: dict(part=pr["halves"][s]["part"], d_mm=round(float(row["ring_d"][s]), 4),
                    origin_mm=np.round(ring_origin(row, pr["halves"][s], row["ring_d"][s]), 5).tolist(),
                    R_rows=np.round(R, 7).tolist())
            for s in pr["order"]}


def _sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def run(cfg_in, tag, force=False):
    import Sofa.Core
    import Sofa.Simulation
    import scene_hybrid as S
    import run_hybrid as RH
    RH.S = S                                            # run_hybrid's helpers read their scene module from here
    cfg = S.load_cfg(cfg_in)
    pcfg = probe_cfg(cfg.pop("probe", None))
    # keys read with cfg.get() outside scene_hybrid.CFG: frame_every (run_hybrid.write_frame) and collision_groups
    # (scene_hybrid.groups_for); anything else unknown is a typo that would silently do nothing
    unknown = [k for k in cfg.get("_unknown_keys", []) if k not in ("probe", "frame_every", "collision_groups")]
    if unknown:
        raise ValueError("unknown scene keys %s (a typo would silently do nothing)" % unknown)
    cfg["_unknown_keys"] = []
    P = S.paths()
    out = os.path.join(P["runs"], tag)
    if os.path.isdir(out) and os.listdir(out) and not force:
        raise SystemExit("runs/%s exists and is not empty: never overwrite a run (new tag, or --force for a probe you own)"
                         % tag)
    os.makedirs(out, exist_ok=True)
    cfg["tag"] = tag
    prov = RH.provenance()
    prov["code_sha256_16"]["hybrid/cervix_probe.py"] = _sha16(os.path.abspath(__file__))
    prov["cervix_probe"] = VERSION
    with open(os.path.join(out, "cfg.json"), "w") as fh:
        json.dump(dict(cfg, probe=pcfg, _provenance=prov), fh, indent=1, default=RH._json)
    t0 = time.perf_counter()
    root = Sofa.Core.Node("root")
    ctx = build_probe(root, cfg, pcfg, S)
    ctrl = root.addObject(make_controller(S)(name="hybrid", ctx=ctx, out_dir=out))
    Sofa.Simulation.init(root)
    S.post_init(ctx)
    t_init = time.perf_counter() - t0
    info = S.scene_summary(ctx)
    info.update(tag=tag, t_init_s=round(t_init, 2))
    print("INIT " + json.dumps(info, default=RH._json), flush=True)
    sched = ctx["sched"]
    n_max = int(cfg["max_steps"])
    fe = int(cfg.get("frame_every") or 0)
    fc = RH.frame_cache(ctx) if fe > 0 else None
    fd = os.path.join(out, "frames")

    def frame(r):
        RH.write_frame(ctx, ctrl, out, r, fc)
        k = int(r["step"])
        with open(os.path.join(fd, "step_%04d_ring.json" % k), "w") as fh:
            json.dump(dict(step=k, phase=r["phase"], ring=_ring_state(ctx, RH._sched_at(ctx, k)),
                           probe=r.get("probe")), fh, default=RH._json)

    while not ctrl.done and ctrl.k < n_max:
        Sofa.Simulation.animate(root, root.dt.value)
        r = ctrl.rows[-1]
        k = int(r["step"])
        nxt = sched[k + 1]["phase"] if k + 1 < len(sched) else "H"
        if fe > 0 and r.get("finite", False) and (k % fe == 0 or ctrl.done or nxt != r["phase"]):
            frame(r)
        pb = r.get("probe") or {}
        rm = pb.get("ring_monitor") or {}
        cf = r.get("corpus_fem")
        print("STEP %3d %-2s d=%s wall=%5.0fms dx=%.4f cont=%4d err=%.2e it=%3d minV=%.3f cvx umax=%.2f J(p1/tot)=%s/%s "
              "ring sd_min=%s n<-0.5=%s%s"
              % (k, r["phase"], "/".join("%.2f" % v for v in (pb.get("ring_d") or {}).values()), r["wall_ms"],
                 r["dx_max_mm"], r["n_contacts"], r["constraint_err"], r["constraint_it"], r["min_vol_ratio"],
                 r["disp"]["cervix"]["umax"], (pb.get("cervix_J") or {}).get("p1"), (pb.get("cervix_J") or {}).get("total"),
                 "/".join(str(v.get("sd_min")) for v in rm.values()), "/".join(str(v.get("n_lt_m0p5")) for v in rm.values()),
                 "" if cf is None else " corpus minV=%s attach=%s" % (r["disp"]["corpus"].get("min_vol_ratio"),
                                                                    cf.get("attach_residual_mm"))), flush=True)
    if not ctrl.done and ctrl.status == "running":
        ctrl.status = "max_steps"
    if fe > 0 and ctrl.rows and ctrl.rows[-1].get("finite", False) and fc["last_step"] != ctrl.rows[-1]["step"]:
        frame(ctrl.rows[-1])
    ctrl.close()
    rows = ctrl.rows
    wall = np.array([r["wall_ms"] for r in rows], float) if rows else np.zeros(1)
    fin = rows[-1] if rows else {}
    phases = []
    for r in rows:
        if r["phase"] not in phases:
            phases.append(r["phase"])
    per_phase = {}
    for p in phases:
        rr = [r for r in rows if r["phase"] == p]
        per_phase[p] = dict(steps=len(rr), ms_median=round(float(np.median([r["wall_ms"] for r in rr])), 1),
                            contacts_max=int(max(r["n_contacts"] for r in rr)),
                            cervix_min_vol_ratio=round(float(min(r["disp"]["cervix"].get("min_vol_ratio", 1.0)
                                                                 for r in rr)), 4),
                            min_vol_ratio=round(float(min(r["min_vol_ratio"] for r in rr)), 4))
        mons = [(r.get("probe") or {}).get("ring_monitor") for r in rr]
        mons = [m for m in mons if m]
        if mons:
            per_phase[p]["ring_monitor"] = {
                s: dict(sd_min=min([m[s]["sd_min"] for m in mons if m[s]["sd_min"] is not None] or [None]),
                        n_lt_m0p5_max=int(max(m[s]["n_lt_m0p5"] for m in mons)))
                for s in ctx["probe"]["order"]}
    first_contact = {}
    for s in ctx["probe"]["order"]:
        hit = [r for r in rows if r["phase"] in ("RL", "RR", "RS") and ((r.get("probe") or {}).get("ring_monitor") or {}).get(
            s, {}).get("n_within_alarm", 0) > 0]
        first_contact[s] = dict(step=int(hit[0]["step"]), phase=hit[0]["phase"], d_mm=hit[0]["probe"]["ring_d"][s],
                                note="first ring-phase row with a cervix surface node within alarm_mm of this half "
                                     "(monitor): d at first contact ~ how far the ring pushes the portio after it") \
            if hit else None
    summary = dict(info)
    summary.update(
        tag=tag, version=VERSION, status=ctrl.status, converged=bool(ctrl.status == "converged"), n_steps=len(rows),
        n_settle_steps=int(sum(1 for r in rows if r["phase"] == "H")),
        ms_per_step_median=round(float(np.median(wall)), 1), total_s=round(time.perf_counter() - t0, 1),
        t_init_s=round(t_init, 2), per_phase=per_phase, ring_first_contact=first_contact, drained=ctrl.drained,
        convergence=dict(dx_max_final_mm=fin.get("dx_max_mm"), dx_ratio=fin.get("dx_ratio"),
                         projected_remaining_drift_mm=fin.get("drift_est_mm"),
                         constraint_err_per_contact_final=fin.get("constraint_err_per_contact"),
                         n_consecutive_ok=int(sum(1 for v in ctrl.ok_hist[-int(cfg["conv_steps"]):] if v))),
        final=dict(n_contacts=fin.get("n_contacts"), min_vol_ratio=fin.get("min_vol_ratio"), phase=fin.get("phase"),
                   probe=fin.get("probe")),
        gates=dict(nan=bool(ctrl.status == "abort_nan"), inverted_tets=bool(ctrl.status == "abort_inverted_tets"),
                   wall_time=bool(ctrl.status == "abort_wall_time"),
                   min_vol_ratio_run=round(float(min([r["min_vol_ratio"] for r in rows] or [1.0])), 4),
                   all_finite=bool(all(r["finite"] for r in rows)),
                   reached_final_pose=bool(len(rows) >= len(sched) and all(r["finite"] for r in rows)),
                   constraint_solver_converged_frac=round(
                       float(np.mean([1.0 if r["constraint_converged"] else 0.0 for r in rows])), 3) if rows else None),
        frames=(dict(n=len(fc["index"]), every=fe, dir="frames") if fe > 0 and fc else None),
        cfg={k: v for k, v in cfg.items() if not k.startswith("_")}, probe_cfg=pcfg, provenance=prov)
    if fin.get("finite", False):
        disp, dev = RH.write_outputs(ctx, ctrl, out, summary)
        summary["displacement"] = disp
        with open(os.path.join(out, "final", "ring.json"), "w") as fh:
            last = RH._sched_at(ctx, int(fin["step"]))
            json.dump(dict(tag=tag, ring=bool(pcfg["ring"]), halves=_ring_state(ctx, last),
                           convention="p_world = origin_mm + p_app @ R_rows (the half's OBJ in <applicator_dir>)",
                           applicator_dir=cfg.get("applicator_dir")), fh, indent=1, default=RH._json)
    else:
        summary["displacement"] = None
        print("NON-FINITE state: final/ outputs not written", flush=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=RH._json)
    print("DONE " + json.dumps(dict(tag=tag, status=summary["status"], n_steps=summary["n_steps"],
                                    ms_per_step_median=summary["ms_per_step_median"], total_s=summary["total_s"],
                                    gates=summary["gates"], ring_first_contact=first_contact), default=RH._json),
          flush=True)
    return summary


# ============================================================================================ HOST: score
REGION_LOWER_H_MM = 15.0        # "lower" cervix = below ring top + this (the part the ring can load directly)


def _host_ctx():
    import eval_hybrid as EH
    import tf_metrics as TM
    return EH, TM


def _sdf_mask(m, sp):
    """Signed distance (mm) on the mask's grid, < 0 inside: EDT to the outside minus EDT to the inside (voxel centres)."""
    from scipy import ndimage as ndi
    return ndi.distance_transform_edt(~m, sampling=sp) - ndi.distance_transform_edt(m, sampling=sp)


def _sample(field, aff, X, order=1):
    from scipy import ndimage as ndi
    inv = np.linalg.inv(aff)
    ijk = (np.asarray(X, float) @ inv[:3, :3].T + inv[:3, 3]).T
    return ndi.map_coordinates(field.astype(np.float32), ijk, order=order, mode="nearest")


def _surface_pts(m, aff):
    from scipy import ndimage as ndi
    s = m & ~ndi.binary_erosion(m, ndi.generate_binary_structure(3, 1))
    return np.argwhere(s) @ aff[:3, :3].T + aff[:3, 3]


def _regions(L, ring):
    """Region labels of points L in the own-flange frame: 0 over the ring (within its outer radius of its centre, below
    ring top + REGION_LOWER_H_MM), 1 beside it (outside that radius, same heights), 2 upper."""
    rho = np.hypot(L[:, 0] - ring["cx"], L[:, 1] - ring["cy"])
    low = L[:, 2] < ring["top"] + REGION_LOWER_H_MM
    return np.where(low & (rho <= ring["R"]), 0, np.where(low, 1, 2))


REGION_NAMES = ("over_ring", "beside_ring", "upper")


def _load_state(rd, m121=None):
    """(cervix node displacement, device json, ring json or None, source) of a run: final/, or the 1-2-1 average of the
    frames k-6, k-3, k (README STAGE 3c M121@k*) when m121 = k."""
    if m121 is None:
        U = np.load(rd + "/final/cervix_u.npy")
        dev = json.load(open(rd + "/device_final.json"))
        ring = json.load(open(rd + "/final/ring.json")) if os.path.exists(rd + "/final/ring.json") else None
        return U, dev, ring, "final"
    k = int(m121)
    Us = [np.load(rd + "/frames/step_%04d_cervix_u.npy" % j).astype(float) for j in (k - 6, k - 3, k)]
    U = (Us[0] + 2.0 * Us[1] + Us[2]) / 4.0
    fdev = json.load(open(rd + "/frames/step_%04d_device.json" % k))
    dev = json.load(open(rd + "/device_final.json"))
    if np.abs(np.asarray(fdev["flange_mm"], float) - np.asarray(dev["flange_mm"], float)).max() > 1e-3:
        raise SystemExit("M121@%d: the frame's flange differs from device_final (not a settle frame)" % k)
    rp = rd + "/frames/step_%04d_ring.json" % k
    ring = dict(halves=json.load(open(rp))["ring"]) if os.path.exists(rp) else None
    return U, dev, ring, "M121@%d (frames %d, %d, %d; 1-2-1)" % (k, k - 6, k - 3, k)


def cmd_score(tag, control=None, m121=None, control_m121=None, quiet=False, out_path=None, regions_from=None,
              rigid_carry=False):
    import nibabel as nib
    EH, TM = _host_ctx()
    import evaluate as ev
    import final_eval as fe
    rd = "%s/%s" % (EH.RUNS, tag)
    cfg = json.load(open(rd + "/cfg.json"))
    summ = json.load(open(rd + "/summary.json")) if os.path.exists(rd + "/summary.json") else {}
    appd = EH.HYB + "/" + cfg.get("applicator_dir", "applicator")
    appj = json.load(open(appd + "/applicator.json"))
    lm = appj["landmarks"]
    if "ring_centre" not in lm:                         # a tandem-only run (v4): the regions need a ring's geometry
        if not regions_from:
            raise SystemExit("%s has no ring landmarks: pass --regions-from <applicator dir with a ring> (the same "
                             "tandem body, e.g. applicator_v5)" % os.path.basename(appd))
        lm = json.load(open(EH.HYB + "/" + regions_from + "/applicator.json"))["landmarks"]
    cx = EH.Ctx()
    bt = cx.bt
    # ---- the run's final cervix
    import vagina_wall as VW
    X0, T = VW.read_vtk_legacy(EH.MESHES + "/cervix/tets.vtk")[:2]
    X0 = np.asarray(X0, float)
    T = np.asarray(T, np.int64)
    U, dev_j, ring_j, src = _load_state(rd, m121)
    if rigid_carry:                                     # the K0-style comparator: the REST cervix carried rigidly by the
        Tk = np.asarray(dev_j["corpus_T_preBT_to_final"], float)     # run's final corpus target, no deformation
        U = X0 @ Tk[:3, :3].T + Tk[:3, 3] - X0
        src = "rigid carry of the rest cervix by the final corpus target of %s (no deformation)" % tag
    X = X0 + U
    meta = json.load(open(EH.MESHES + "/cervix/meta.json"))
    s2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
    _, Fs = geom.read_obj(EH.MESHES + "/cervix/surface.obj")
    Vs = X[s2n]
    F = np.asarray(dev_j["flange_mm"], float)
    Rr = np.array([dev_j["x_app"], dev_j["y_app"], dev_j["tube_axis"]], float)
    a = Rr[2]
    dev = EH.read_device_final(rd + "/device_final.json", bt.L)
    # ---- volumes and J
    v0, v1 = tet_vol(X0, T), tet_vol(X, T)
    sg = np.sign(v0.sum())
    vols = dict(model_rest_cc=round(float(sg * v0.sum() / 1000.0), 3), model_final_cc=round(float(sg * v1.sum() / 1000.0), 3),
                J=j_stats(v1, v0))
    vols["model_delta_cc"] = round(vols["model_final_cc"] - vols["model_rest_cc"], 3)
    # label volumes (preBT / BT)
    D = EH.DATA
    lp = {n: np.asarray(nib.load("%s/preBT_MRI_label_%s.nii" % (D, n)).dataobj) > 0 for n in ("HR-CTV", "uterus", "vagina")}
    vpre = float(abs(np.linalg.det(nib.load("%s/preBT_MRI_label_HR-CTV.nii" % D).affine[:3, :3]))) / 1000.0
    app_new = np.asarray(nib.load(TM.APP_LABEL_NEW).dataobj) > 0.5
    dev_bt = app_new | bt.lab["applicator"] | bt.lab["ovoid"]
    cerv_bt = bt.lab["HR-CTV"] & ~bt.lab["uterus"]
    vv = bt.vv / 1000.0
    lab = dict(preBT_HRCTV_cc=round(float(lp["HR-CTV"].sum() * vpre), 3),
               preBT_HRCTV_minus_U_cc=round(float((lp["HR-CTV"] & ~lp["uterus"]).sum() * vpre), 3),
               preBT_HRCTV_minus_U_minus_V_cc=round(float((lp["HR-CTV"] & ~lp["uterus"] & ~lp["vagina"]).sum() * vpre), 3),
               BT_HRCTV_cc=round(float(bt.lab["HR-CTV"].sum() * vv), 3),
               BT_HRCTV_minus_U_cc=round(float(cerv_bt.sum() * vv), 3),
               BT_HRCTV_minus_U_minus_device_cc=round(float((cerv_bt & ~dev_bt).sum() * vv), 3),
               BT_HRCTV_minus_U_minus_V_minus_device_cc=round(float((cerv_bt & ~dev_bt & ~bt.lab["vagina"]).sum() * vv), 3))
    dBT = lab["BT_HRCTV_minus_U_minus_device_cc"] - lab["preBT_HRCTV_minus_U_cc"]
    vols["labels"] = lab
    vols["BT_delta_cc_tissue"] = round(dBT, 3)
    vols["model_share_of_BT_delta"] = round(vols["model_delta_cc"] / dBT, 4) if abs(dBT) > 1e-9 else None
    vols["note"] = ("model = the cervix body (HR-CTV minus uterus, incl. the vagina-label tract inside it); BT tissue = BT "
                    "HR-CTV minus uterus minus device voxels (applicator labels | ovoid).  The BT change mixes mechanics "
                    "with tumour regression and contouring between the scans: a share is what a material COULD account "
                    "for, not evidence that it does.")
    # ---- Dice / MSD / HD95, PELVIS and E_app
    refs = dict(HRCTV_minus_U=cerv_bt, HRCTV_minus_U_minus_V=cerv_bt & ~bt.lab["vagina"],
                HRCTV_minus_U_minus_V_or_A=cerv_bt & ~(bt.lab["vagina"] | app_new | bt.lab["ovoid"]))
    fmaps = dict(PELVIS=cx.pelvis_map()[0], E_app=cx.eapp_map(dev)[0])
    fit = {}
    masks = {}
    for fr, fm in fmaps.items():
        m = ev.voxelize(fm(Vs), Fs, bt.shape, bt.aff)
        masks[fr] = m
        fit[fr] = {k: fe.metrics_ext(m, r, bt.sp, bt.vv) for k, r in refs.items()}
    # ---- HR-CTV\U left-right offset about the model tube (tf_metrics.final_hrctv's definition)
    cen = X[T].mean(1)
    vol = np.abs(v1) / 1000.0
    x, s, tot = TM.lr_side(cen, F, a, None, vol)
    lev = []
    for s0 in np.arange(-30.0, 45.0, 5.0):
        mm = (s >= s0) & (s < s0 + 5)
        if vol[mm].sum() > 0.1:
            lev.append(dict(h=[float(s0), float(s0 + 5)], cc=round(float(vol[mm].sum()), 2),
                            mean_x_mm=round(float((x * vol)[mm].sum() / vol[mm].sum()), 2)))
    m10 = (s >= 0) & (s < 10)
    ng = json.load(open(TM.NEW_GEOM_JSON))
    Pb = TM.mask_pts(cerv_bt, bt.aff)
    _, _, r_json = TM.lr_side(Pb, bt.F, bt.a, vv)
    _, _, r_new = TM.lr_side(Pb, np.asarray(ng["tandem"]["tip"], float), np.asarray(ng["tandem"]["axis"], float), vv)
    hr = dict(model_vs_model_tube=tot, model_per_level=lev,
              model_lowest_10mm_above_flange_mean_x_mm=(round(float((x * vol)[m10].sum() / vol[m10].sum()), 2)
                                                        if vol[m10].sum() > 0 else None),
              BT_vs_json_tandem=r_json, BT_vs_updated_label_tandem=r_new,
              definition="tf_metrics.final_hrctv: final cervix tets, tet-volume weighted x (patient right +) of the offset "
                         "normal to the model tube; BT: HR-CTV minus uterus voxels about the BT tandem")
    # ---- regions (final own-flange frame) and per-region fit
    ring = dict(cx=float(lm["ring_centre"][0]), cy=float(lm["ring_centre"][1]), top=float(lm["ring_top_face_z"]),
                R=float(lm["ring_outer_r"]))
    Lnode = (X - F) @ Rr.T
    reg_n = _regions(Lnode, ring)
    Tc = np.asarray(dev_j.get("corpus_T_preBT_to_final"), float)
    Xr = X0 @ Tc[:3, :3].T + Tc[:3, 3]                  # the rest cervix carried rigidly by the final corpus target
    Unr = X - Xr
    rad = (Lnode[:, :2] - np.array([ring["cx"], ring["cy"]])) / np.maximum(
        np.hypot(Lnode[:, 0] - ring["cx"], Lnode[:, 1] - ring["cy"]), 1e-9)[:, None]
    Unr_L = Unr @ Rr.T
    ctrl_X = None
    if control:
        crd = "%s/%s" % (EH.RUNS, control)
        Uc = _load_state(crd, control_m121)[0]
        ctrl_X = X0 + Uc
    Jt = v1 / v0
    reg_t = _regions((cen - F) @ Rr.T, ring)
    # surface distances by region (PELVIS): model surface vertices -> BT signed distance; BT surface voxels -> model
    sdf_bt = _sdf_mask(cerv_bt, bt.sp)
    fmP = fmaps["PELVIS"]
    d_m2b = _sample(sdf_bt, bt.aff, fmP(Vs))            # > 0: the model surface lies OUTSIDE BT (too big there)
    reg_v = _regions((Vs - F) @ Rr.T, ring)
    sdf_m = _sdf_mask(masks["PELVIS"], bt.sp)
    Pbs = _surface_pts(cerv_bt, bt.aff)
    d_b2m = _sample(sdf_m, bt.aff, Pbs)                 # > 0: the BT surface lies OUTSIDE the model (model too small)
    Rp, tp = cx.pelvis_map()[1:]
    Pbs_pre = Pbs @ Rp.T + tp                           # BT -> preBT world (y = R x + t)
    reg_b = _regions((Pbs_pre - F) @ Rr.T, ring)
    regions = {}
    for i, nm in enumerate(REGION_NAMES):
        mn, mt, mv, mb = reg_n == i, reg_t == i, reg_v == i, reg_b == i
        e = dict(n_nodes=int(mn.sum()), cc_final=round(float(vol[mt].sum()), 3),
                 vol_ratio=round(float(v1[mt].sum() / v0[mt].sum()), 4) if mt.any() else None)
        if mn.any():
            e["nonrigid_mm"] = dict(mean=round(float(np.linalg.norm(Unr[mn], axis=1).mean()), 3),
                                    p95=round(float(np.percentile(np.linalg.norm(Unr[mn], axis=1), 95)), 3),
                                    axial_mean=round(float(Unr_L[mn, 2].mean()), 3),
                                    radial_out_mean=round(float(np.einsum("ij,ij->i", Unr_L[mn, :2], rad[mn]).mean()), 3))
            if ctrl_X is not None:
                dc = X[mn] - ctrl_X[mn]
                dcl = dc @ Rr.T
                e["vs_control_mm"] = dict(mean=round(float(np.linalg.norm(dc, axis=1).mean()), 3),
                                          max=round(float(np.linalg.norm(dc, axis=1).max()), 3),
                                          axial_mean=round(float(dcl[:, 2].mean()), 3),
                                          radial_out_mean=round(float(np.einsum("ij,ij->i", dcl[:, :2], rad[mn]).mean()), 3))
        if mv.any():
            e["model_to_BT_mm"] = dict(n=int(mv.sum()), signed_mean=round(float(d_m2b[mv].mean()), 3),
                                       abs_mean=round(float(np.abs(d_m2b[mv]).mean()), 3))
        if mb.any():
            e["BT_to_model_mm"] = dict(n=int(mb.sum()), signed_mean=round(float(d_b2m[mb].mean()), 3),
                                       abs_mean=round(float(np.abs(d_b2m[mb]).mean()), 3))
        if mv.any() and mb.any():
            e["region_msd_mm"] = round(float(np.concatenate([np.abs(d_m2b[mv]), np.abs(d_b2m[mb])]).mean()), 3)
        regions[nm] = e
    regions["definition"] = (
        "own-flange frame of the scored state (rows x_app, y_app, tube_axis about the flange); ring = applicator.json "
        "landmarks ring_centre / ring_outer_r / ring_top_face_z.  over_ring: within ring_outer_r of the ring centre and "
        "below ring top + %.0f mm; beside_ring: outside that radius, same heights; upper: above.  nonrigid = final minus "
        "the rest cervix carried by the final corpus target (radial_out = away from the ring axis, axial = along the "
        "tube); model_to_BT = BT HR-CTV\\U signed distance at the model's surface vertices (PELVIS, > 0 model outside "
        "BT); BT_to_model = the model mask's signed distance at the BT surface voxels (> 0 BT outside the model)"
        % REGION_LOWER_H_MM)
    # ---- ring: the scored state's cervix against each half (exact signed distance)
    ring_pen = None
    if ring_j and ring_j.get("halves"):
        import vtk
        ring_pen = {}
        for s, h in ring_j["halves"].items():
            V, Fh = geom.read_obj("%s/%s.obj" % (appd, h["part"]))
            Vw = np.asarray(h["origin_mm"], float) + np.asarray(V, float) @ np.asarray(h["R_rows"], float)
            f = vtk.vtkImplicitPolyDataDistance()
            f.SetInput(ev._poly(Vw, Fh))
            sd = np.array([f.EvaluateFunction([float(p[0]), float(p[1]), float(p[2])]) for p in Vs])
            ring_pen[s] = dict(d_mm=h.get("d_mm"), n_inside=int((sd < 0).sum()), n_lt_m0p5=int((sd < -0.5).sum()),
                               sd_min=round(float(sd.min()), 3))
    # ---- the cervix bottom against the ring top (own-flange frame)
    hv = (Vs - F) @ Rr[2]
    bottom = dict(min_h=round(float(hv.min()), 2), p1_h=round(float(np.percentile(hv, 1)), 2),
                  ring_top_h=round(ring["top"], 2), n_surface_below_ring_top=int((hv < ring["top"]).sum()))
    res = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION, tag=tag, state=src, control=control,
               material=dict(E_kPa=cfg.get("E_kPa", {}).get("cervix"), nu=cfg.get("nu", {}).get("cervix"),
                             law=(cfg.get("material_by_body") or {}).get("cervix", cfg.get("material", "corotational")),
                             corpus_model=cfg.get("corpus_model", "rigid"),
                             ring=(cfg.get("probe") or {}).get("ring"), seat=(cfg.get("probe") or {}).get("seat"),
                             drain=(cfg.get("probe") or {}).get("drain"),
                             note="E_kPa / nu = the insertion material; drain = the material the settle switched to"),
               status=dict(status=summ.get("status"), n_steps=summ.get("n_steps"), gates=summ.get("gates"),
                           per_phase=summ.get("per_phase"), ring_first_contact=summ.get("ring_first_contact")),
               fit=fit, hrctv_offset=hr, volumes=vols, regions=regions, ring_penetration=ring_pen, cervix_bottom=bottom,
               tags=dict(fit="PREDICTED given a CALIBRATED device pose (Delta in-sample); material rows are a sensitivity",
                         BT="REFERENCE (labels; HR-CTV changed between the scans: fix plan R6)"))
    p = out_path or "%s/%s/cervix_probe%s.json" % (EH.EVALD, tag, "" if m121 is None else "_m121_%d" % int(m121))
    EH.wjson(p, res)
    if not quiet:
        f0 = fit["PELVIS"]["HRCTV_minus_U"]
        f1 = fit["E_app"]["HRCTV_minus_U"]
        print("%s [%s]  E %s nu %s  cervix Dice PELVIS %.4f (MSD %.3f)  E_app %.4f (MSD %.3f)  HR-CTV x %+.2f mm  "
              "vol %.4f (J p1 %.3f)" % (tag, src, res["material"]["E_kPa"], res["material"]["nu"], f0["dice"], f0["msd"],
                                         f1["dice"], f1["msd"], tot["mean_x_mm"], vols["J"]["total"], vols["J"]["p1"]))
        print("wrote", p, flush=True)
    return res


# ============================================================================================ HOST: selftest
def cmd_selftest():
    """The pure arithmetic on synthetic values (no patient data)."""
    ok = []
    h = dict(e_app=[0.0, 0.6, -0.8], n_out_app=[1.0, 0.0, 0.0], offset_mm=5.0, click_mm=20.0, click_shape_p=4.0)
    # 1 disp: seat, click boundary, far
    ok.append(("disp(0) = 0", np.allclose(ring_disp(h, 0.0), 0.0)))
    ok.append(("disp(click) = click e + offset n", np.allclose(ring_disp(h, 20.0), [5.0, 12.0, -16.0])))
    ok.append(("disp(50) = 50 e + offset n", np.allclose(ring_disp(h, 50.0), [5.0, 30.0, -40.0])))
    ok.append(("disp vectorised", ring_disp(h, np.array([0.0, 20.0])).shape == (2, 3)))
    # 2 approach sampling: ends at 0, path steps within the limits, near steps small
    ds, st, start = approach_d([h], 30.0, 0.0, 2.0, 0.5, 14.0)
    P = ring_disp(h, np.array([30.0] + ds))
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    ok.append(("approach ends at the seat", ds[-1] == 0.0 and all(a > b for a, b in zip(ds, ds[1:]))))
    ok.append(("start d = previous row", start == [30.0] + ds[:-1]))
    ok.append(("chord per step <= limit (+res)", bool(np.all(step <= np.where(np.array(start) <= 14.0, 0.5, 2.0) + 0.02))))
    ok.append(("near rows exist", sum(1 for d in ds if d <= 14.0) >= 14.0 / 0.5 * 0.9))
    ok.append(("a row where the near zone starts", 14.0 in [round(d, 6) for d in ds]))
    h2 = dict(h, n_out_app=[-1.0, 0.0, 0.0], e_app=[0.0, 0.8, -0.6])
    dj, sj, _ = approach_d([h, h2], 14.0, 0.0, 2.0, 0.5, 14.0)
    PA, PB = ring_disp(h, np.array([14.0] + dj)), ring_disp(h2, np.array([14.0] + dj))
    ok.append(("joint steps bound both halves", bool(np.all(np.maximum(np.linalg.norm(np.diff(PA, axis=0), axis=1),
                                                                        np.linalg.norm(np.diff(PB, axis=0), axis=1))
                                                             <= 0.52))))
    # 3 schedule: phases in order, park / stand-off / seat values, tandem rows untouched (both seat modes)
    base = []
    for ph, n in (("P", 2), ("V", 3), ("C", 4), ("L", 3)):
        for i in range(n):
            base.append(dict(phase=ph, F=np.array([0.0, 0.0, float(len(base))]), R_rows=np.eye(3), tip_s=float(i),
                             T_corpus=np.eye(4), u=0.0, s=0.0, ov_lag=0.0, stage=ph))
    hh = dict(L=h, R=dict(h, n_out_app=[-1.0, 0.0, 0.0]))
    # sequential
    rows, info = probe_schedule(base, hh, ["L", "R"], probe_cfg(dict(n_hold=2, seat="sequential")))
    phs = [r["phase"] for r in rows]
    exp = ["P"] * 2 + ["V"] * 3 + ["C"] * 4 + ["RL"] * (info["halves"]["L"]["n_approach"] + 2) + \
        ["RR"] * (info["halves"]["R"]["n_approach"] + 2) + ["L"] * 3
    ok.append(("sequential: phase order", phs == exp))
    ok.append(("park before R", all(r["ring_d"] == dict(L=30.0, R=30.0) for r in rows[:9])))
    rl = [r for r in rows if r["phase"] == "RL"]
    ok.append(("sequential: R parked during RL, L seated at its end",
               all(r["ring_d"]["R"] == 30.0 for r in rl) and rl[-1]["ring_d"]["L"] == 0.0))
    rr = [r for r in rows if r["phase"] == "RR"]
    ok.append(("sequential: L seated during RR", all(r["ring_d"]["L"] == 0.0 for r in rr) and rr[-1]["ring_d"]["R"] == 0.0))
    ok.append(("tandem held in R (last C row)", all(np.array_equal(r["F"], base[8]["F"]) for r in rl + rr)))
    ok.append(("L rows seated and unchanged", [r["F"][2] for r in rows if r["phase"] == "L"] == [9.0, 10.0, 11.0]
               and all(r["ring_d"] == dict(L=0.0, R=0.0) for r in rows if r["phase"] == "L")))
    ok.append(("input rows not mutated", all("ring_d" not in r for r in base)))
    # joint (default)
    rows, info = probe_schedule(base, hh, ["L", "R"], probe_cfg(dict(n_hold=2)))
    rl = [r for r in rows if r["phase"] == "RL"]
    rr = [r for r in rows if r["phase"] == "RR"]
    rs = [r for r in rows if r["phase"] == "RS"]
    ok.append(("joint: phase order", [r["phase"] for r in rows] == ["P"] * 2 + ["V"] * 3 + ["C"] * 4 + ["RL"] * len(rl)
               + ["RR"] * len(rr) + ["RS"] * len(rs) + ["L"] * 3))
    ok.append(("joint: L to the stand-off, R parked", rl[-1]["ring_d"] == dict(L=14.0, R=30.0)
               and all(r["ring_d"]["R"] == 30.0 for r in rl)))
    ok.append(("joint: R to the stand-off, L waiting", rr[-1]["ring_d"] == dict(L=14.0, R=14.0)
               and all(r["ring_d"]["L"] == 14.0 for r in rr)))
    ok.append(("joint: both together onto the seat, then held", all(r["ring_d"]["L"] == r["ring_d"]["R"] for r in rs)
               and [r["ring_d"]["L"] for r in rs[-3:]] == [0.0, 0.0, 0.0] and info["seat_together"]["n_hold"] == 2))
    ok.append(("joint: tandem held", all(np.array_equal(r["F"], base[8]["F"]) for r in rl + rr + rs)))
    # drain cfg
    ok.append(("drain parsed", probe_cfg(dict(drain={"E_kPa": 25, "nu": 0.25}))["drain"] == dict(E_kPa=25.0, nu=0.25)))
    for name, bad in (("drain nu 0.5 refused", dict(drain={"E_kPa": 25, "nu": 0.5})),
                      ("drain extra key refused", dict(drain={"E_kPa": 25, "nu": 0.3, "at": "H"})),
                      ("seat typo refused", dict(seat="together")), ("stand-off >= D0 refused", dict(standoff_mm=30.0))):
        try:
            probe_cfg(bad)
            ok.append((name, False))
        except ValueError:
            ok.append((name, True))
    # 4 ring origin: world = origin + p_app @ R_rows reproduces F + (p + disp) @ R
    Rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    row = dict(F=np.array([1.0, 2.0, 3.0]), R_rows=Rz)
    p = np.array([2.0, -1.0, 4.0])
    ok.append(("ring origin convention", np.allclose(ring_origin(row, h, 7.0) + p @ Rz,
                                                     row["F"] + (p + ring_disp(h, 7.0)) @ Rz)))
    # 5 isolation check: accepts the isolated groups, refuses a colliding OAR and a ring touching the tandem
    G = dict(corpus=[1, 9], cervix=[1, 2, 11], vagina=[1, 2, 3, 4, 9, 10, 11, 12], bladder=[1, 2, 3, 4, 9, 10, 11, 12],
             rectum=[1, 2, 3, 4, 9, 10, 11, 12], sigmoid=[1, 2, 3, 4, 9, 10, 11, 12], tube=[9, 10, 11], shaft=[9, 10, 11])
    cfg = dict(vagina_model="solid", apex_attach="off", n_balloon=0, static_bodies=list(ISOLATED_OARS),
               insertion_path="tandem_first", ovoid_mode="none")
    try:
        check_isolated(cfg, G, [9, 12])
        ok.append(("isolated cfg accepted", True))
    except ValueError:
        ok.append(("isolated cfg accepted", False))
    for name, g2, rg in (("colliding bladder refused", dict(G, bladder=[4]), [9, 12]),
                         ("ring vs tube refused", dict(G, tube=[10, 11]), [9, 12]),
                         ("ring sharing the cervix id refused", G, [11, 12])):
        try:
            check_isolated(cfg, g2, rg)
            ok.append((name, False))
        except ValueError:
            ok.append((name, True))
    try:
        check_isolated(dict(cfg, apex_attach="lift"), G, [9, 12])
        ok.append(("apex lift refused", False))
    except ValueError:
        ok.append(("apex lift refused", True))
    try:
        probe_cfg(dict(D0=3))
        ok.append(("unknown probe key refused", False))
    except ValueError:
        ok.append(("unknown probe key refused", True))
    # 6 monitor: a unit cube, outward normals, points inside / outside
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 10.0
    Fq = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                   [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    ok.append(("cube outward", mesh_signed_volume(V, Fq) > 0))
    from scipy.spatial import cKDTree
    Nq = vertex_normals(V, Fq)
    sd = approx_sd(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [50.0, 50.0, 50.0]]), V, Nq, cKDTree(V), 6.0)
    ok.append(("monitor sign (out > 0, in < 0, far NaN)", sd[0] > 0 and sd[1] < 0 and np.isnan(sd[2])))
    # 7 J stats
    js = j_stats(np.array([1.0, 0.5, 2.0, 1.0]), np.ones(4))
    ok.append(("J stats", js["total"] == 1.125 and js["frac_lt_0p8"] == 0.25 and js["min"] == 0.5))
    n_ok = sum(1 for _, v in ok if v)
    for name, v in ok:
        print("%-4s %s" % ("ok" if v else "FAIL", name))
    print("%d / %d" % (n_ok, len(ok)))
    return n_ok == len(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "score", "selftest"])
    ap.add_argument("--tag")
    ap.add_argument("--cfg", default=None, help="inline JSON dict or a path to a JSON file (scene keys + 'probe')")
    ap.add_argument("--force", action="store_true", help="run: allow a non-empty runs/<tag> (a probe you own)")
    ap.add_argument("--control", default=None, help="score: the run the per-region change is measured against")
    ap.add_argument("--m121", type=int, default=None, help="score: M121@k* of the frames instead of final/")
    ap.add_argument("--control-m121", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true", help="score: no summary line")
    ap.add_argument("--rigid-carry", action="store_true", help="score: the REST cervix carried rigidly by the run's final "
                                                               "corpus target instead of the run's state (a K0 comparator)")
    ap.add_argument("--regions-from", default=None, help="score: applicator dir whose ring landmarks define the regions "
                                                         "when the run's own applicator has none")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if cmd_selftest() else 1)
    if not a.tag:
        ap.error("--tag is required")
    if a.cmd == "run":
        run(a.cfg, a.tag, force=a.force)
    else:
        cmd_score(a.tag, control=a.control, m121=a.m121, control_m121=a.control_m121, out_path=a.out, quiet=a.quiet,
                  regions_from=a.regions_from, rigid_carry=a.rigid_carry)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                      # SofaPython3 v22.12 can segfault in interpreter teardown; outputs are closed
