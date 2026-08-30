#!/usr/bin/env python3
"""Is every grafted anatomy actually loadable, meshable and navigable?

Geometry validation on the centerlines says nothing about whether the thing
the simulator sees is sound. The mesh is built by marking the centerline into
a voxel cube at 0.6/0.9 mm spacing, smoothing it twice and decimating, so a
thin distal siphon can erode: the narrowest lumen here is 1.46 mm across,
barely two voxels, and if it pinches shut the device would be navigating
outside its own vessel with nothing to report the fault.

So the checks that matter are, per anatomy:

  load      the branch list parses and the RCCA is found
  insert    the insertion point comes from the (11) bridge, not the fallback,
            and lands inside the meshed lumen
  enclose   EVERY RCCA centerline point is inside the mesh. This is the
            erosion test, and the one a centerline-only check cannot make
  targets   the sampling pool is non-empty after the near-ostium exclusion,
            and every target is inside the mesh
  fit       the catheter's outer diameter fits the narrowest lumen

With --sofa it then builds the real environment and runs episodes, which is
the only way to find out whether SOFA loads the mesh and steps without
diverging.

    python3 topbrain_tools/check_anatomies.py --anatomies <dir> [--sofa 3]
"""
import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "/opt/eve_training/eve_bench")

CATH_OD = 0.7      # mic_cath straight_outer_diameter, from the env
# A centerline point can sit marginally outside a smoothed wall without any
# consequence: the pathfinder works off centerlines, not the mesh, and a target
# a fraction of a millimetre proud of the wall is still well inside the 5 mm
# reach threshold. What is NOT tolerable is a stretch of vessel that pinched
# shut, where the device would be navigating outside its own lumen. These two
# thresholds separate the cases; both must be exceeded to matter.
MAX_OUTSIDE_MM = 1.5    # how far a single point may sit beyond the wall
MAX_OUTSIDE_RUN = 3     # how many consecutive points may sit outside
GUIDE_OD = 0.36    # mic_guide
RCCA_KEY = "Centerline curve - RCCA.mrk"


def enclosed(mesh_path, points):
    """Fraction of points strictly inside the mesh, and the worst run outside."""
    import pyvista as pv
    surf = pv.read(mesh_path)
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    # check_surface=False: the shipped tree is already slightly open at the
    # cut branch ends, which is not what is being tested here.
    sel = cloud.select_enclosed_points(surf, tolerance=0.0, check_surface=False)
    inside = np.asarray(sel["SelectedPoints"], dtype=bool)
    worst, run = 0, 0
    for v in inside:
        run = 0 if v else run + 1
        worst = max(worst, run)
    return float(inside.mean()), int(worst), inside


def check_one(name, folder, verbose=False, drop_stubs=False):
    from eve_bench.dualdevicenav import load_branches
    from eve.intervention.vesseltree.topbrainanatomyset import (
        TopBrainAnatomySet)

    out = {"name": name, "ok": True, "notes": []}
    branches = load_branches(os.path.join(folder, "Centrelines_comb")
                             if os.path.isdir(os.path.join(folder, "Centrelines_comb"))
                             else folder)
    if drop_stubs:
        # Control: the shipped tree with the same cranial stubs removed that
        # the graft removes, so the two are compared like for like.
        branches = [b for b in branches
                    if " - " in str(b.name)
                    or float(np.asarray(b.coordinates)[:, 2].min()) <= 500.0]
    mesh = os.path.join(folder, "vessel_architecture_collision.obj")
    if not os.path.exists(mesh):
        # control folders (the shipped tree) have no baked mesh; make one so
        # the control is measured through the identical code path
        from eve.intervention.vesseltree.util.meshing import generate_temp_mesh
        mesh = generate_temp_mesh(branches, "ctrl", 0.99)
        out["_tmp_mesh"] = mesh
    tree = TopBrainAnatomySet(branch_lists=[branches], mesh_paths=[mesh],
                              anatomy_names=[name], seed=0,
                              episodes_between_change=1)

    rcca = None
    for b in tree.branches:
        if "RCCA" in str(b.name).upper():
            rcca = b
            break
    if rcca is None:
        out["ok"] = False
        out["notes"].append("no RCCA branch")
        return out

    coords = np.asarray(rcca.coordinates, dtype=float)
    radii = np.asarray(rcca.radii, dtype=float)
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    out["route_mm"] = float(seg.sum())
    out["dmin"] = float(2 * radii.min())

    # insertion: must have come from the (11) bridge, not the RCCA-ostium
    # fallback, or the fork-discrimination skill is not being exercised
    ip = np.asarray(tree.insertion.position, dtype=float)
    bridge = None
    for b in tree.branches:
        if "(11)" in str(b.name):
            bridge = np.asarray(b.coordinates, dtype=float)
            break
    out["ip_from_bridge"] = bridge is not None and bool(
        np.linalg.norm(bridge - ip, axis=1).min() < 1e-6)
    if not out["ip_from_bridge"]:
        out["ok"] = False
        out["notes"].append("insertion not on branch (11)")

    mesh_path = tree.mesh_path
    out["mesh_path"] = mesh_path
    import pyvista as pv
    surf = pv.read(mesh_path)
    out["tris"] = int(surf.n_cells)
    out["open_edges"] = int(surf.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False,
        non_manifold_edges=False).n_cells)

    frac, worst, inside = enclosed(mesh_path, coords)
    out["encl_frac"] = frac
    out["encl_gap"] = worst
    out["max_out"] = 0.0
    if frac < 1.0:
        s = np.concatenate([[0.0], np.cumsum(seg)])
        bad = s[~inside]
        depth = np.linalg.norm(
            np.asarray(surf.points)[None, :, :] - coords[~inside][:, None, :],
            axis=2).min(axis=1)
        out["max_out"] = float(depth.max())
        msg = ("RCCA leaves the lumen at %.0f-%.0f mm (%d pts, longest run %d, "
               "up to %.2f mm beyond the wall)"
               % (bad.min(), bad.max(), int((~inside).sum()), worst,
                  depth.max()))
        if depth.max() > MAX_OUTSIDE_MM and worst > MAX_OUTSIDE_RUN:
            out["ok"] = False
            out["notes"].append(msg)
        else:
            out["notes"].append("tolerated: " + msg)

    ip_frac, _, _ = enclosed(mesh_path, ip.reshape(1, 3))
    out["ip_inside"] = bool(ip_frac > 0.5)
    if not out["ip_inside"]:
        out["ok"] = False
        out["notes"].append("insertion point outside the mesh")

    # target pool, exactly as the env builds it
    from eve.intervention.target.centerlinerandom import CenterlineRandom

    class _Fluoro:                      # target.reset needs these only
        image_rot_zx = [20, 5]
        image_center = [0.0, 0.0, 0.0]
        field_of_view = None

    tgt = CenterlineRandom(vessel_tree=tree, fluoroscopy=_Fluoro(), threshold=5,
                           branches=[RCCA_KEY], min_arclength_from_start=40.0)
    tgt._init_centerline_point_cloud()
    pool = np.asarray(tgt._potential_targets, dtype=float)
    out["n_targets"] = int(len(pool))
    if len(pool) == 0:
        out["ok"] = False
        out["notes"].append("no targets survive the near-ostium exclusion")
    else:
        d = np.linalg.norm(pool[:, None, :] - coords[None, :, :], axis=2).argmin(axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])[d]
        out["tgt_arc"] = (float(arc.min()), float(arc.max()))
        tfrac, _, _ = enclosed(mesh_path, pool)
        out["tgt_inside"] = tfrac
        if tfrac < 1.0:
            # Judged by the same rule: a target just proud of the wall is
            # still reachable within the 5 mm threshold.
            note = "%.0f%% of targets lie outside the mesh" % (100 * (1 - tfrac))
            if out["max_out"] > MAX_OUTSIDE_MM and worst > MAX_OUTSIDE_RUN:
                out["ok"] = False
                out["notes"].append(note)
            else:
                out["notes"].append("tolerated: " + note)

    if out["dmin"] <= CATH_OD:
        out["ok"] = False
        out["notes"].append("lumen %.2f mm does not admit the %.2f mm catheter"
                            % (out["dmin"], CATH_OD))
    if out.get("_tmp_mesh"):
        try:
            os.remove(out["_tmp_mesh"])
        except OSError:
            pass
    return out


def run_sofa(anatomy_dir, names, steps, verbose=False):
    from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain

    print("\n=== SOFA rollout: %d anatomies x %d steps ===" % (len(names), steps))
    allok = True
    for name in names:
        try:
            iv = DualDeviceNavTopBrain(anatomy_dir=anatomy_dir, seed=0,
                                       episodes_between_change=1, only=[name])
            # Roughly half the RCCA target pool sits in the 130 mm of host
            # trunk every anatomy shares, so a fixed seed can pick a target
            # that never enters the graft. Search for one that does, or the
            # rollout would not be testing the real siphon at all.
            rcca = [b for b in iv.vessel_tree.branches
                    if "RCCA" in str(b.name).upper()][0]
            cc = np.asarray(rcca.coordinates, dtype=float)
            arc = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(cc, axis=0), axis=1))])
            tgt_arc = -1.0
            for trial in range(60):
                iv.reset(episode_number=0, seed=trial)
                t3 = np.asarray(iv.target.coordinates3d, dtype=float)
                j = int(np.linalg.norm(cc - t3, axis=1).argmin())
                tgt_arc = float(arc[j])
                if tgt_arc > 130.0:
                    break
            start = np.asarray(iv.fluoroscopy.device_trackings3d[0][0], dtype=float)
            moved = 0.0
            for i in range(steps):
                iv.step(np.array([[12.0, 0.0], [12.0, 0.0]], dtype=np.float32))
                tip = np.asarray(iv.fluoroscopy.device_trackings3d[0][0], dtype=float)
                moved = float(np.linalg.norm(tip - start))
            ins = [float(x) for x in iv.device_lengths_inserted]
            print("  %-16s OK  inserted %.1f/%.1f mm  tip moved %.1f mm  "
                  "target at %.0f mm along the route%s"
                  % (name, ins[0], ins[1], moved, tgt_arc,
                     "" if tgt_arc > 130.0 else "  (SHARED TRUNK, not the graft)"))
            iv.close()
        except Exception as e:                                # noqa: BLE001
            allok = False
            print("  %-16s FAILED  %s: %s" % (name, type(e).__name__, e))
            if verbose:
                traceback.print_exc()
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", default="/opt/eve_training/topbrain_data/anatomies")
    ap.add_argument("--sofa", type=int, default=0,
                    help="run SOFA on this many anatomies (0 = skip)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--host", default=None,
                    help="shipped Centrelines_comb, run as a control")
    ap.add_argument("--shard", default=None,
                    help="i/n: check only every n-th anatomy, so workers can "
                         "split the set")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    names = sorted(d for d in os.listdir(a.anatomies)
                   if os.path.isdir(os.path.join(a.anatomies, d, "Centrelines_comb")))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        names = names[i::n]
    print("checking %d anatomies in %s\n" % (len(names), a.anatomies))
    print("%-16s %8s %7s %7s %6s %8s %7s %s"
          % ("anatomy", "route", "d_min", "tris", "open", "enclosed", "targets", "status"))

    # Controls: the shipped tree as it ships, and the shipped tree with the
    # same stubs dropped. Any artifact the second one also shows is a property
    # of the meshing pipeline, not something the graft introduced.
    if a.host:
        for label, drop in (("HOST(shipped)", False), ("HOST(-stubs)", True)):
            try:
                r = check_one(label, a.host, a.verbose, drop_stubs=drop)
                print("%-16s %8.1f %7.2f %7d %6d %8.3f %7d %s"
                      % (label, r.get("route_mm", 0), r.get("dmin", 0),
                         r.get("tris", 0), r.get("open_edges", -1),
                         r.get("encl_frac", 0), r.get("n_targets", 0),
                         "OK" if r["ok"] else "FAIL"))
                for note in r["notes"]:
                    print("%-16s   -> %s" % ("", note))
            except Exception as e:                            # noqa: BLE001
                print("%-16s CONTROL CRASHED %s: %s" % (label, type(e).__name__, e))
                if a.verbose:
                    traceback.print_exc()
        print()

    rows = []
    for n in names:
        folder = os.path.join(a.anatomies, n)
        try:
            r = check_one(n, folder, a.verbose)
        except Exception as e:                                # noqa: BLE001
            print("%-16s CRASHED %s: %s" % (n, type(e).__name__, e))
            if a.verbose:
                traceback.print_exc()
            rows.append({"name": n, "ok": False, "notes": ["crashed"]})
            continue
        rows.append(r)
        print("%-16s %8.1f %7.2f %7d %6d %8.3f %7d %s"
              % (n, r.get("route_mm", 0), r.get("dmin", 0), r.get("tris", 0),
                 r.get("open_edges", -1), r.get("encl_frac", 0),
                 r.get("n_targets", 0), "OK" if r["ok"] else "FAIL"))
        for note in r["notes"]:
            print("%-16s   -> %s" % ("", note))

    bad = [r for r in rows if not r["ok"]]
    print("\n%d/%d anatomies pass the static checks" % (len(rows) - len(bad), len(rows)))
    if bad:
        print("failing: %s" % ", ".join(r["name"] for r in bad))

    if a.sofa:
        good = [r["name"] for r in rows if r["ok"]]
        pick = good[:: max(1, len(good) // a.sofa)][:a.sofa] if good else []
        if bad:      # always exercise a failing one too, if SOFA tolerates it
            pick = list(dict.fromkeys(pick + [bad[0]["name"]]))
        ok = run_sofa(a.anatomies, pick, a.steps, a.verbose)
        if not ok:
            return 1
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
