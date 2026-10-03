#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\sofa_rva.py (= /sp/v5_va/checks/sofa_rva.py in the build containers)
"""v5_checks item 4b: SOFA load test that EXERCISES THE NEW RVA, from the training insertion (eve-training-fixed).

    python3 sofa_rva.py <name> [steps=300] [anatomy_dir=/repo/carotid_data/anatomies_v5] [seed=777]

* The anatomy is loaded through the training loader (DualDeviceNavTopBrain, the training insertion inside the (11)
  bridge, devices / simulation / fluoroscopy exactly as training), only=[name].
* Frame fix (load_test_v5.py compared fluoroscopy.tracking3d, the TRACKING frame, with vessel-frame centrelines):
  every geometric test here uses simulation.dof_positions, which IS the vessel frame (the frame of the centrelines
  and of the collision OBJ); the target is converted with tracking3d_to_vessel_cs before it is located on the RCCA.
* Schedule (v5 = final; v1-v4 drafts in sofa/dev/): closed loop on the tip's vessel-frame position. Path = (11) from
  the insertion -> RVA (its first ~27 mm is the host's subclavian stretch, then the vertebral: junction stub, donor
  stretch, seam, TopBrain piece). Fork 1 = the (11) end (RVA vs RCCA); fork 2 = the VA ostium (VA vs the subclavian
  continuation (17); in v5 the donor ostium sits 4-10 mm past the (17) take-off, in v4 at it).
    advance   guidewire +8 mm/s; the catheter follows 16-22 mm behind the guidewire tip (12-18 mm inside the VA);
    align     fork 1, 12 mm before it: rotate the guidewire (1.5 rad/s, slow or no advance) until the J tip's sideways
              direction (off the parent's axis) points to the RVA's (cos > 0.6), at most 20 steps a pass, then push;
    approach  fork 2, 8 mm before the ostium: the same alignment toward the VA's first 8 mm, then push at 4 mm/s;
    hook      the wire ran on into (17): pull it back slowly with the J tip turned toward the VA ostium until the tip
              drops into the VA (then advance); after 25 steps, retract to before the fork, turn, retry;
    recover   tip > 6 mm into the RCCA (or curled back into the arch): retract to 8 mm before the fork, turn ~0.6 rad;
    unbuckle  12 mm more guidewire over 15 steps with < 3 mm of new path progress: retract 4 steps (10 mm/s) while
              turning (gentle: the run3 version retracted 19 mm and backed the tip out of the RVA);
    unstick   in the VA, no progress for 10 steps: alternately turn with a gentle push / retract and turn;
    exercise  past the donor stretch (or in the last 40 steps): back-and-forth with rotation in the VA.
  SOFA's route choice is stochastic: up to SOFA_EPISODES (3) episodes of `steps` steps; a new episode (reset, seed+1)
  only if the previous one did not get >= 20 mm into the donor stretch. Every step of every episode is checked.
* Per step: all device DOFs finite; the tip and every device point tested against the anatomy's collision OBJ
  (inside test + wall distance when outside); the tip's location (branch, arclength) by right_route_detail's
  ownership rule (largest r - |x - c| among the 8 nearest centreline samples).
Prints "SOFA RVA PASSED" (simulation sound: 18 branches incl. RVA/LVA, the loader's mesh is this anatomy's OBJ,
insertion inside, 0 exceptions, 0 non-finite steps, tip never > 0.5 mm outside the wall) or "SOFA RVA FAILED", the
separate verdict "RVA EXERCISED yes/no" (tip reached >= 20 mm into the donor stretch), a JSON line; writes
$SOFA_OUT/<name>.json; then os._exit(0) (SOFA segfaults at teardown, so the exit code means nothing).
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

sys.path[:0] = ["/repo/eve", "/repo/eve_bench"]
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain  # noqa: E402
from eve.util.coordtransform import tracking3d_to_vessel_cs  # noqa: E402
import pyvista as pv  # noqa: E402
import vtk  # noqa: E402

name = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ADIR = sys.argv[3] if len(sys.argv) > 3 else "/repo/carotid_data/anatomies_v5"
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 777
OUTD = os.environ.get("SOFA_OUT", "/sp/v5_va/checks/sofa/out")
os.makedirs(OUTD, exist_ok=True)
DT = 1 / 7.5
STEER = os.environ.get("STEER", "0") == "1"      # dynamic-tangent steering in the VA (off: it stalled more in tests)


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


res = {"name": name, "anatomy_dir": ADIR, "steps_budget": STEPS, "seed": SEED, "exception": None}
t_all = time.time()
try:
    root = os.path.join(ADIR, name)
    prov = json.load(open(os.path.join(root, "provenance.json")))
    mesh = pv.read(os.path.join(root, "vessel_architecture_collision.obj")).triangulate().clean()
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)

    def inside(P):
        P = np.atleast_2d(np.asarray(P, float))
        sel = pv.PolyData(P).select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
        ins = np.asarray(sel["SelectedPoints"], bool)
        d = np.zeros(len(P))
        for k in np.nonzero(~ins)[0]:
            d[k] = abs(imp.EvaluateFunction(*P[k]))   # magnitude: the meshes are inward-oriented (as v4)
        return ins, d

    iv = DualDeviceNavTopBrain(anatomy_dir=ADIR, only=[name], seed=4242, episodes_between_change=1,
                               target_min_arclength_mm=133.0)
    t0 = time.time()
    iv.reset(episode_number=0, seed=SEED)
    res["reset_s"] = round(time.time() - t0, 1)
    vt, sim, fl = iv.vessel_tree, iv.simulation, iv.fluoroscopy
    br = list(vt.branches)
    bn = [str(b.name) for b in br]
    res["branches"] = len(bn)
    res["has_RVA_LVA"] = [any(n.endswith("RVA.mrk") for n in bn), any(n.endswith("LVA.mrk") for n in bn)]
    res["loader_mesh_path"] = str(getattr(vt, "mesh_path", None))
    res["loader_mesh_is_this_obj"] = os.path.normpath(res["loader_mesh_path"]) == os.path.normpath(
        os.path.join(root, "vessel_architecture_collision.obj"))

    def B(tag):
        for b in br:
            if str(b.name).endswith(tag):
                return np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        raise KeyError(tag)
    # centreline samples for locating the tip
    P, R, Bi, S = [], [], [], []
    for k, b in enumerate(br):
        c, r = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        P.append(c); R.append(r); Bi.append(np.full(len(c), k)); S.append(arclen(c))
    P, R, Bi, S = np.vstack(P), np.concatenate(R), np.concatenate(Bi), np.concatenate(S)
    tree = cKDTree(P)
    short = [n.split("- ")[-1].replace(".mrk", "") if " - " in n else n.replace("Centerline curve ", "").replace(".mrk", "")
             for n in bn]
    iRVA, iRCCA, i11, i17 = short.index("RVA"), short.index("RCCA"), short.index("(11)"), short.index("(17)")

    def locate(x):
        d, ii = tree.query(x, k=8)
        j = ii[int(np.argmax(R[ii] - d))]
        return int(Bi[j]), float(S[j]), float(np.min(d))

    # path geometry
    p11, _ = B("(11).mrk")
    prva, rrva = B("RVA.mrk")
    srva = arclen(prva)
    ins_pos = np.asarray(vt.insertion.position, float)
    L11 = float(arclen(p11)[-1])
    rev11 = bool(np.linalg.norm(p11[0] - prva[0]) < np.linalg.norm(p11[-1] - prva[0]))
    if rev11:
        p11 = p11[::-1]
    s11 = arclen(p11)
    k_ins = int(np.argmin(np.linalg.norm(p11 - ins_pos, axis=1)))
    s_f1 = float(s11[-1] - s11[k_ins])                       # path arclength of fork 1 (RVA s = 0)
    p17, _ = B("(17).mrk")
    s17_arc = arclen(p17)
    s_rva17 = float(srva[int(np.argmin(np.linalg.norm(prva - p17[0], axis=1)))])
    s_f2 = s_f1 + s_rva17
    prcca, _ = B("RCCA.mrk")
    srcca = arclen(prcca)

    def dir_at(p, s, a, b_):
        i, j = int(np.argmin(np.abs(s - a))), int(np.argmin(np.abs(s - b_)))
        v = p[j] - p[i]
        return v / max(np.linalg.norm(v), 1e-9)
    forks = [  # (path s, desired dir, wrong dir, wrong branch index, parent tangent at the fork)
        (s_f1, dir_at(prva, srva, 0.0, 8.0), dir_at(prcca, srcca, 0.0, 8.0), iRCCA, dir_at(p11, s11, s11[-1] - 8.0, s11[-1])),
        (s_f2, dir_at(prva, srva, s_rva17, s_rva17 + 8.0), dir_at(p17, arclen(p17), 0.0, 8.0), i17,
         dir_at(prva, srva, max(s_rva17 - 8.0, 0.0), s_rva17))]

    def lat(v, t):
        w = v - (v @ t) * t
        n = float(np.linalg.norm(w))
        return w / n if n > 1e-6 else w, n
    v5 = prov.get("v5_va", {}).get("right", {}) or {}
    xr = prov["xform"].get("vaC_R")
    donor = xr is not None
    lo = float(xr["lo_mm"]) if donor else float(prov["xform"]["va_R"]["cut_mm"]) - 60.0
    cut = float(prov["xform"]["va_R"]["cut_mm"])
    ost = float(v5.get("ostium_at_mm", lo)) if donor else s_rva17
    s_o = max(ost, s_rva17)                              # the VA lumen starts at the ostium (past the junction stub)
    # fork 2 sits at the VA ostium: in v5 the donor ostium lies 4-10 mm past the (17) take-off (junction stub inside
    # the subclavian lumen); in v4 s_o = s(17)
    forks[1] = (s_f1 + s_o, dir_at(prva, srva, s_o, s_o + 8.0), forks[1][2], i17, dir_at(prva, srva, max(s_o - 8.0, 0.0), s_o))
    stub = s_o - s_rva17
    res["rva"] = {"donor": ("%s %s" % (xr.get("donor_id"), xr.get("donor_side"))) if donor else None,
                  "v4_va_kept": not donor, "length_mm": round(float(srva[-1]), 1), "s_17_takeoff_mm": round(s_rva17, 1),
                  "ostium_mm": round(ost, 1), "donor_lo_mm": round(lo, 1), "cut_mm": round(cut, 1),
                  "path_s_fork1_mm": round(s_f1, 1), "path_s_fork2_mm": round(s_f2, 1)}
    # target, converted to the vessel frame
    tgt_v = np.asarray(tracking3d_to_vessel_cs(np.asarray(iv.target.coordinates3d, float), fl.image_rot_zx, fl.image_center), float)
    kt = int(np.argmin(np.linalg.norm(prcca - tgt_v, axis=1)))
    res["target_vessel_frame"] = {"dist_to_rcca_mm": round(float(np.linalg.norm(prcca[kt] - tgt_v)), 4),
                                  "rcca_s_mm": round(float(srcca[kt]), 1)}
    res["insertion_inside"] = bool(inside(ins_pos)[0][0])

    def path_s(bi, s):
        if bi == i11:                                    # (11) arclength in path orientation, from the insertion
            return float((L11 - s if rev11 else s) - s11[k_ins])
        if bi == iRVA:
            return s_f1 + s
        return None

    def fork_index(bi, s):
        if bi == i17 or (bi == iRVA and s > 6.0):
            return 1
        return 0

    def inserted_part(dof, length):
        c = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dof, axis=0), axis=1))]
        return dof[c <= max(length - 0.5, 0.0)]

    def tip_dir(dof):
        seg = np.linalg.norm(np.diff(dof[:40], axis=0), axis=1)
        c = np.cumsum(seg)
        j = int(np.searchsorted(c, 3.0)) + 1
        v = dof[0] - dof[min(j, len(dof) - 1)]
        return v / max(np.linalg.norm(v), 1e-9)

    def episode(ep):
        if ep > 0:
            iv.reset(episode_number=ep, seed=SEED + ep)
        mode, hold, hold_act, attempts, best_ps, stuck, n_unstick = "advance", 0, (0.0, 0.0), [0, 0], -1e9, 0, 0
        align_n, hook_n, ex0, best_prog = 0, 0, None, -1e9
        hist = []                                            # (guidewire inserted length, path progress) for buckling
        rec = {"nonfinite_steps": 0, "tip_outside_steps": 0, "tip_outside_max_mm": 0.0, "tip_outside_gt_0.5mm_steps": 0,
               "dev_pts_checked": 0, "dev_pts_outside": 0, "dev_outside_max_mm": 0.0, "max_rva_s_mm": -1.0,
               "first_step_in_rva_past_17": None, "first_step_in_donor": None, "first_step_past_cut": None,
               "steps_in_donor_stretch": 0, "mode_counts": {}, "branch_visits": {}, "dev_out_gt_0.3mm_by_branch": {},
               "dev_out_gt_0.3mm_pts": 0, "dev_out_worst": None, "unbuckles": 0, "tip_out_sites": {}, "tip_out_worst": None}
        trace = []
        t1 = time.time()
        for i in range(STEPS):
            dof = np.asarray(sim.dof_positions, float)
            lens = np.asarray(sim.inserted_lengths, float)
            tip = dof[0]
            bi, s, dcl = locate(tip)
            ps = path_s(bi, s)
            on_path = ps is not None
            if bi == iRVA:
                rec["max_rva_s_mm"] = max(rec["max_rva_s_mm"], s)
                if s > s_o + 5 and rec["first_step_in_rva_past_17"] is None:
                    rec["first_step_in_rva_past_17"] = i
                if s >= lo + 20.0 and rec["first_step_in_donor"] is None:
                    rec["first_step_in_donor"] = i
                if s >= cut and rec["first_step_past_cut"] is None:
                    rec["first_step_past_cut"] = i
                if lo <= s <= cut:
                    rec["steps_in_donor_stretch"] += 1
            fk = fork_index(bi, s)                           # the fork that matters now (by where the tip is)
            f_s, d_good, d_bad, wrong_b, t_par = forks[fk]
            gw, gr = 0.0, 0.0
            # buckling: 12 mm more guidewire in over the last 15 steps with < 3 mm of path progress -> pull back, turn
            prog = ps if on_path else (s_f2 + s if bi == i17 else -1.0)
            best_prog = max(best_prog, prog)
            hist.append((float(lens[0]), float(best_prog)))
            buckled = False
            if mode in ("advance", "align", "approach") and len(hist) > 15:
                l0, p0 = hist[-16]
                buckled = (hist[-1][0] - l0 > 12.0) and (hist[-1][1] - p0 < 3.0)
            if mode == "exercise":
                phase = (i - ex0) % 24
                gw, gr = (-10.0, 1.0) if phase < 5 else (8.0, 0.6)
            elif hold > 0:
                hold -= 1
                gw, gr = hold_act
                if hold == 0:
                    mode, align_n = "advance", 0
            elif buckled:
                rec["unbuckles"] += 1
                hist.clear()
                mode, hold, hold_act = "unbuckle", 4, (-10.0, 1.5)
                gw, gr = hold_act
            elif mode == "recover":
                if (on_path and ps < f_s - 8.0) or lens[0] < 5.0:
                    mode, hold, hold_act = "turn", 3, (0.0, 1.5)
                    gw, gr = hold_act
                else:
                    gw = -20.0
            elif mode == "hook":
                # fork 2, hook technique: the wire ran on into the subclavian continuation (17); pull it back slowly with
                # the J tip turned toward the VA ostium until the tip drops into the VA (then advance); give up after
                # 25 steps (or if it slides 25 mm on): retract to before the fork, turn, retry
                hook_n += 1
                if bi == iRVA and s > s_o + 3.0:
                    mode, align_n = "advance", 0
                    gw = 5.0
                elif hook_n > 25 or (bi == i17 and s > 25.0) or (bi == iRVA and s < s_rva17 - 8.0) or bi not in (i17, iRVA):
                    mode = "recover"
                    gw = -20.0
                else:
                    t17 = dir_at(p17, s17_arc, max(s - 4.0, 0.0), s + 1.0) if bi == i17 else t_par
                    to_va = prva[int(np.argmin(np.abs(srva - (s_o + 3.0))))] - dof[0]
                    ltd, ntd = lat(tip_dir(dof), t17)
                    lva, _ = lat(to_va / max(np.linalg.norm(to_va), 1e-9), t17)
                    score = float(ltd @ lva) if ntd > 0.1 else -1.0
                    gw, gr = (-3.0, 0.0) if score > 0.6 else (-1.5, 1.5)
            else:
                if bi == i17 and s > stub + 4.0:             # fork 2 missed: hook back
                    attempts[1] += 1
                    mode, hook_n = "hook", 0
                    gw, gr = -1.5, 1.5
                elif (not on_path) and not ((bi == iRCCA and s <= 6.0) or (bi == i17 and s <= stub + 4.0)):
                    # off the path (RCCA past 6 mm, or curled back into the arch): retract to before the fork, turn
                    attempts[fk] += 1
                    mode = "recover"
                    gw = -20.0
                elif on_path and fk == 0 and f_s - 12.0 <= ps <= f_s + 3.0:
                    # fork 1: turn the J tip's sideways direction (off the parent's axis) toward the RVA's, at most 20
                    # steps per pass, then push
                    align_n += 1
                    ltd, ntd = lat(tip_dir(dof), t_par)
                    lgood, _ = lat(d_good, t_par)
                    if (ntd > 0.1 and float(ltd @ lgood) > 0.6) or align_n > 20:
                        gw = 5.0
                    else:
                        gw, gr = (0.0 if ps > f_s - 3.0 else 2.0), 1.5
                    mode = "align"
                elif on_path and fk == 1 and f_s - 8.0 <= ps <= f_s + 3.0:
                    # fork 2: as fork 1, toward the VA's first 8 mm past its ostium; a run-on into (17) is hooked back
                    align_n += 1
                    ltd, ntd = lat(tip_dir(dof), t_par)
                    lgood, _ = lat(d_good, t_par)
                    if (ntd > 0.1 and float(ltd @ lgood) > 0.6) or align_n > 20:
                        gw = 4.0
                    else:
                        gw, gr = (0.0 if ps > f_s - 3.0 else 2.0), 1.5
                    mode = "approach"
                else:
                    mode = "advance"
                    gw = 8.0
                    if bi == iRVA and s > s_o + 10.0:
                        if s > best_ps + 0.5:
                            best_ps, stuck = s, 0
                        else:
                            stuck += 1
                        if stuck >= 10:
                            n_unstick += 1
                            # alternate: turn in place with a gentle push, then retract-and-turn
                            hold_act = (2.0, 1.5) if n_unstick % 2 == 1 else (-10.0, 1.5)
                            stuck, mode, hold = 0, "unstick", (8 if n_unstick % 2 == 1 else 9)
                            gw, gr = hold_act
                        if s >= cut or i >= STEPS - 40:
                            mode, ex0 = "exercise", i
            # catheter follows the guidewire: 16-22 mm behind its tip (the 15 mm J tip free to steer), 12-18 mm in the VA
            gap = lens[0] - lens[1]
            ct = 0.0
            far_in = bi == iRVA and s > s_o + 10.0
            g_hi, g_lo = (18.0, 12.0) if far_in else (22.0, 16.0)
            if gap > g_hi and gw >= 0:
                ct = min(15.0, max(gw, 6.0))
            elif gap < g_lo:
                ct = -8.0
            rec["mode_counts"][mode] = rec["mode_counts"].get(mode, 0) + 1
            nm = short[bi]
            rec["branch_visits"][nm] = rec["branch_visits"].get(nm, 0) + 1
            iv.step(np.array([[gw, gr], [ct, 0.0]]))
            dof2 = np.asarray(sim.dof_positions, float)
            if not np.all(np.isfinite(dof2)):
                rec["nonfinite_steps"] += 1
                continue
            pts = inserted_part(dof2, float(max(np.asarray(sim.inserted_lengths))))
            if len(pts) == 0:
                pts = dof2[:1]
            ins_, dd = inside(pts)
            rec["dev_pts_checked"] += len(pts)
            rec["dev_pts_outside"] += int((~ins_).sum())
            rec["dev_outside_max_mm"] = max(rec["dev_outside_max_mm"], float(dd.max()) if len(dd) else 0.0)
            # where device points stand > 0.3 mm (SOFA's contact distance) outside the wall
            far_out = np.nonzero(dd > 0.3)[0]
            rec["dev_out_gt_0.3mm_pts"] += int(len(far_out))
            for k in far_out:
                b_k, s_k, _ = locate(pts[k])
                key = "%s@%d" % (short[b_k], int(s_k // 10 * 10))
                rec["dev_out_gt_0.3mm_by_branch"][key] = rec["dev_out_gt_0.3mm_by_branch"].get(key, 0) + 1
            if len(far_out):
                kw = int(far_out[np.argmax(dd[far_out])])
                if rec["dev_out_worst"] is None or dd[kw] > rec["dev_out_worst"]["mm"]:
                    b_k, s_k, dcl_k = locate(pts[kw])
                    rec["dev_out_worst"] = {"mm": round(float(dd[kw]), 3), "step": i, "branch": short[b_k], "s_mm": round(s_k, 1),
                                            "dof_index": kw, "dist_from_tip_mm": round(float(np.sum(np.linalg.norm(np.diff(pts[:kw + 1], axis=0), axis=1))), 1),
                                            "point": np.round(pts[kw], 2).tolist(), "d_centreline_mm": round(dcl_k, 2)}
            if not ins_[0]:
                rec["tip_outside_steps"] += 1
                rec["tip_outside_max_mm"] = max(rec["tip_outside_max_mm"], float(dd[0]))
                b_t, s_t, _ = locate(pts[0])
                kt = "%s@%d" % (short[b_t], int(s_t // 10 * 10))
                rec["tip_out_sites"][kt] = rec["tip_out_sites"].get(kt, 0) + 1
                if rec["tip_out_worst"] is None or dd[0] > rec["tip_out_worst"]["mm"]:
                    rec["tip_out_worst"] = {"mm": round(float(dd[0]), 3), "step": i, "branch": short[b_t], "s_mm": round(s_t, 1), "point": np.round(pts[0], 2).tolist(), "mode": mode}
                if dd[0] > 0.5:
                    rec["tip_outside_gt_0.5mm_steps"] += 1
            if i % 10 == 9 or i == STEPS - 1:
                trace.append([i, mode, nm, round(s, 1), round(float(lens[0]), 1), round(float(lens[1]), 1)])
                print("step %3d mode %-8s tip on %-6s s %6.1f  gw %.1f cath %.1f  attempts %s" % (
                    i, mode, nm, s, lens[0], lens[1], attempts), flush=True)
        dt = time.time() - t1
        bi, s, _ = locate(np.asarray(sim.dof_positions, float)[0])
        rec.update({"steps": STEPS, "ms_per_step": round(1000 * dt / STEPS, 1), "fork_attempts": attempts,
                    "final_tip": {"branch": short[bi], "s_mm": round(s, 1)},
                    "max_rva_s_mm": round(rec["max_rva_s_mm"], 1),
                    "donor_traversed_mm": round(max(0.0, min(rec["max_rva_s_mm"], cut) - lo), 1),
                    "reached_donor_20mm": rec["max_rva_s_mm"] >= lo + 20.0,
                    "reached_topbrain": rec["max_rva_s_mm"] >= cut})
        rec["trace"] = trace
        return rec

    # up to SOFA_EPISODES episodes (SOFA's route choice is stochastic): a new episode (reset, new seed) only if the
    # previous one did not get >= 20 mm into the donor stretch
    MAX_EP = int(os.environ.get("SOFA_EPISODES", "3"))
    eps = []
    for ep in range(MAX_EP):
        t0 = time.time()
        r_ep = episode(ep)
        r_ep["episode"] = ep
        r_ep["seconds"] = round(time.time() - t0, 1)
        eps.append(r_ep)
        print("EPISODE %d: max RVA s %.1f, donor traversed %.1f mm, attempts %s, non-finite %d, tip outside %d" % (
            ep, r_ep["max_rva_s_mm"], r_ep["donor_traversed_mm"], r_ep["fork_attempts"], r_ep["nonfinite_steps"],
            r_ep["tip_outside_steps"]), flush=True)
        if r_ep["reached_donor_20mm"]:
            break
    worst = [e["dev_out_worst"] for e in eps if e["dev_out_worst"]]
    res.update({
        "episodes": len(eps), "steps": sum(e["steps"] for e in eps), "steps_per_episode": STEPS,
        "exercised_in_episode": next((e["episode"] for e in eps if e["reached_donor_20mm"]), None),
        "nonfinite_steps": sum(e["nonfinite_steps"] for e in eps),
        "tip_outside_steps": sum(e["tip_outside_steps"] for e in eps),
        "tip_outside_max_mm": max(e["tip_outside_max_mm"] for e in eps),
        "tip_outside_gt_0.5mm_steps": sum(e["tip_outside_gt_0.5mm_steps"] for e in eps),
        "dev_pts_checked": sum(e["dev_pts_checked"] for e in eps), "dev_pts_outside": sum(e["dev_pts_outside"] for e in eps),
        "dev_out_gt_0.3mm_pts": sum(e["dev_out_gt_0.3mm_pts"] for e in eps),
        "dev_outside_max_mm": max(e["dev_outside_max_mm"] for e in eps),
        "dev_out_worst": max(worst, key=lambda w: w["mm"]) if worst else None,
        "tip_out_worst": max([e["tip_out_worst"] for e in eps if e["tip_out_worst"]], key=lambda w: w["mm"]) if any(e["tip_out_worst"] for e in eps) else None,
        "tip_out_sites": {k: sum(e["tip_out_sites"].get(k, 0) for e in eps) for k in sorted({k for e in eps for k in e["tip_out_sites"]})},
        "dev_out_gt_0.3mm_by_branch": {k: sum(e["dev_out_gt_0.3mm_by_branch"].get(k, 0) for e in eps)
                                       for k in sorted({k for e in eps for k in e["dev_out_gt_0.3mm_by_branch"]})},
        "max_rva_s_mm": max(e["max_rva_s_mm"] for e in eps),
        "donor_traversed_mm": max(e["donor_traversed_mm"] for e in eps),
        "reached_donor_20mm": any(e["reached_donor_20mm"] for e in eps),
        "reached_topbrain": any(e["reached_topbrain"] for e in eps),
        "steps_in_donor_stretch": sum(e["steps_in_donor_stretch"] for e in eps),
        "first_step_in_donor": next((e["first_step_in_donor"] for e in eps if e["first_step_in_donor"] is not None), None),
        "fork_attempts": [e["fork_attempts"] for e in eps], "unbuckles": sum(e.get("unbuckles", 0) for e in eps),
        "ms_per_step": round(float(np.mean([e["ms_per_step"] for e in eps])), 1),
        "final_tip": eps[-1]["final_tip"]})
    res["episode_records"] = eps
except Exception as e:  # noqa: BLE001
    res["exception"] = "%s: %s" % (type(e).__name__, e)
    res["tb"] = traceback.format_exc()
    print(res["tb"], flush=True)
res["seconds"] = round(time.time() - t_all, 1)
ok = (res.get("exception") is None and res.get("branches") == 18 and all(res.get("has_RVA_LVA", [False]))
      and res.get("loader_mesh_is_this_obj") and res.get("insertion_inside") and res.get("nonfinite_steps", 1) == 0
      and res.get("tip_outside_gt_0.5mm_steps", 1) == 0)
res["passed"] = bool(ok)
res["rva_exercised"] = bool(res.get("reached_donor_20mm"))
with open(os.path.join(OUTD, name + ".json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
print("SOFA RVA PASSED" if ok else "SOFA RVA FAILED", flush=True)
print("RVA EXERCISED %s (max RVA s %.1f mm; donor [%s, %s]; traversed %s mm)" % (
    "yes" if res["rva_exercised"] else "no", res.get("max_rva_s_mm", -1), res.get("rva", {}).get("donor_lo_mm"),
    res.get("rva", {}).get("cut_mm"), res.get("donor_traversed_mm")), flush=True)
print("JSON", json.dumps({k: v for k, v in res.items() if k not in ("trace", "tb", "episode_records")}, default=float), flush=True)
os._exit(0)
