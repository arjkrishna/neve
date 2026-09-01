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
  route     the WHOLE insertion-to-target route lies inside the ONE connected
            mesh component the device is inserted into. 'enclose' cannot make
            this test: select_enclosed_points answers "inside any closed sheet
            of this surface", so a vessel the mesher severed into two closed
            tubes still scores ~100% enclosed - every point is inside one tube
            or the other - and the device hits a wall halfway up
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

# --- navigable-route check ---------------------------------------------------
# select_enclosed_points is a point-in-solid test against the WHOLE surface: it
# returns true if the point is inside ANY closed sheet of it. That makes the
# 'enclose' score above blind to the one failure that matters most. Measured on
# a synthetic 1.5 mm-radius tube cut in half by a 0.5 mm gap: the severed pair
# scores enclosed=0.9950 with a longest-outside-run of 1 point, against 0.9975
# and 1 for the intact tube. The two are indistinguishable, and the severed one
# is unnavigable.
#
# What the device actually needs is that the entire route from the insertion
# point to the target lies inside the SINGLE connected component it is inserted
# into, and that that component is closed around it.
ROUTE_STEP_MM = 0.25        # route sampling
# How much contiguous route may sit outside the hosting component before it is
# a hole rather than a discretisation artefact. The mesher marks the centerline
# into a 0.6/0.6/0.9 mm cube and smooths it twice with sigma = 1 voxel, so the
# iso-surface can legitimately fall a voxel or two inside a sharply curving
# centerline. Measured over the 49 reference anatomies of set A: the largest
# contiguous gap in any anatomy the validator accepts is 2.75 mm, and the next
# value in the whole 278-anatomy corpus is 7.0 mm - the histogram is bimodal
# with nothing in between, so the threshold is not delicate.
MAX_ROUTE_GAP_MM = 3.0
MIN_COMPONENT_CELLS = 4     # fewer triangles cannot enclose anything
# Holes in the hosting component, i.e. edges used by one triangle. Measured max
# over all 278 anatomies of A and B is 5; anything much larger is a torn wall.
MAX_HOST_OPEN_EDGES = 12


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


def _resample(pts, step):
    """Route resampled at a fixed arclength step, with its arclength."""
    pts = np.asarray(pts, dtype=float)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], d > 1e-9])]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    n = max(int(s[-1] / step) + 1, 2)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, pts[:, i]) for i in range(3)], axis=1), q


def _inside(surf, pts):
    import pyvista as pv
    sel = pv.PolyData(np.asarray(pts, dtype=float)).select_enclosed_points(
        surf, tolerance=0.0, check_surface=False)
    return np.asarray(sel["SelectedPoints"], dtype=bool)


def _open_edges(surf):
    return int(surf.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False,
        non_manifold_edges=False).n_cells)


def navigable_route(mesh_path, route, insertion):
    """Can the device travel the whole route without leaving the lumen?

    The route must lie inside ONE connected component of the mesh - the one the
    insertion point is in - and that component must be closed around it. Split
    the surface into connected components, test enclosure against each one
    separately, and report where the hosting component stops containing the
    route. Whether some OTHER component picks the route up again is the
    difference between a vessel severed into two tubes and a lumen eroded shut;
    both are unnavigable, but they have different causes.
    """
    import pyvista as pv
    surf = mesh_path if isinstance(mesh_path, pv.DataSet) else pv.read(mesh_path)
    surf = surf.extract_surface().triangulate()

    pts, s = _resample(route, ROUTE_STEP_MM)
    # No welding pass: the mesher writes shared vertices, and welding at 1e-6 mm
    # leaves the component count unchanged on all 278 meshes of A and B.
    conn = surf.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"], dtype=int)
    ncomp = int(rid.max()) + 1

    memb = np.zeros((ncomp, len(pts)), dtype=bool)
    subs = {}
    holds_ip = []
    ip = np.asarray(insertion, dtype=float).reshape(1, 3)
    for c in range(ncomp):
        sub = conn.extract_cells(np.nonzero(rid == c)[0]
                                 ).extract_surface().triangulate()
        if sub.n_cells < MIN_COMPONENT_CELLS:
            continue
        subs[c] = sub
        memb[c] = _inside(sub, pts)
        if bool(_inside(sub, ip)[0]):
            holds_ip.append(c)

    out = {"n_components": ncomp, "route_len": float(s[-1]),
           "n_route_pts": int(len(pts))}
    if holds_ip:
        host = max(holds_ip, key=lambda c: int(memb[c].sum()))
        out["ip_hosted"] = True
    elif subs:
        host = max(subs, key=lambda c: int(memb[c].sum()))
        out["ip_hosted"] = False
    else:
        out["ip_hosted"] = False
        out["host"] = -1
        out["cover_frac"] = 0.0
        out["gap_mm"] = float(s[-1])
        out["gap_at_mm"] = 0.0
        out["gap_on_other_comp"] = 0.0
        out["handoff_mm"] = 0.0
        out["host_cells"] = 0
        out["host_open_edges"] = -1
        out["host_surf"] = None
        out["inside_host"] = np.zeros(len(pts), dtype=bool)
        return out

    cover = memb[host]
    # longest contiguous stretch of route the hosting component does not hold
    worst = worst_i = worst_j = 0
    i = 0
    while i < len(cover):
        if not cover[i]:
            j = i
            while j + 1 < len(cover) and not cover[j + 1]:
                j += 1
            if j - i + 1 > worst:
                worst, worst_i, worst_j = j - i + 1, i, j
            i = j + 1
        else:
            i += 1
    other = memb.copy()
    other[host] = False
    on_other = (float(other[:, worst_i:worst_j + 1].any(axis=0).mean())
                if worst else 0.0)
    # Route the hosting component has let go of and some OTHER component has
    # picked up: the device would have to cross a wall to follow it. Measured
    # over the 244 anatomies of A and B that pass the gap rule, this is 0.00 mm
    # in every single one, so any non-zero value is a severance, however short.
    handoff = float(((~cover) & other.any(axis=0)).sum() * ROUTE_STEP_MM)

    out.update(host=int(host), host_cells=int(subs[host].n_cells),
               host_open_edges=_open_edges(subs[host]),
               host_surf=subs[host],
               cover_frac=float(cover.mean()),
               gap_mm=float(worst * ROUTE_STEP_MM),
               gap_at_mm=float(s[worst_i]) if worst else float("nan"),
               gap_on_other_comp=on_other, handoff_mm=handoff,
               inside_host=cover)
    return out


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

    # --- route: one connected component, from the insertion point onward -----
    # The device enters part-way along the (11) bridge and travels the rest of
    # the bridge and then the whole RCCA, so that is the route, not the RCCA
    # alone.
    if bridge is not None and len(bridge) >= 3:
        j = int(np.linalg.norm(bridge - ip, axis=1).argmin())
        lead = (bridge[j:]
                if np.linalg.norm(bridge[-1] - coords[0])
                <= np.linalg.norm(bridge[0] - coords[0])
                else bridge[:j + 1][::-1])
        route = np.concatenate([lead, coords], axis=0)
    else:
        route = coords
    nav = navigable_route(mesh_path, route, ip)
    out["n_components"] = nav["n_components"]
    out["route_cover"] = nav["cover_frac"]
    out["route_gap_mm"] = nav["gap_mm"]
    out["route_handoff_mm"] = nav["handoff_mm"]
    out["host_open_edges"] = nav["host_open_edges"]
    if not nav["ip_hosted"]:
        out["ok"] = False
        out["notes"].append("insertion point is inside no connected component "
                            "of the mesh")
    if nav["gap_mm"] > MAX_ROUTE_GAP_MM or nav["handoff_mm"] > 0.0:
        out["ok"] = False
        kind = ("SEVERED: the route continues on a detached mesh component"
                if nav["handoff_mm"] > 0.0 else
                "PINCHED: the lumen closes and the route leaves the mesh")
        out["notes"].append(
            "%s - the component holding the insertion point stops containing "
            "the route at %.0f mm and does not hold the next %.1f mm of it "
            "(%.0f%% of the route is unreachable, %.1f mm of it sealed inside "
            "another component; %d mesh components)"
            % (kind, nav["gap_at_mm"], nav["gap_mm"],
               100 * (1 - nav["cover_frac"]), nav["handoff_mm"],
               nav["n_components"]))
    elif nav["gap_mm"] > 0:
        out["notes"].append("tolerated: %.1f mm of route outside the hosting "
                            "component" % nav["gap_mm"])
    if nav["host_open_edges"] > MAX_HOST_OPEN_EDGES:
        out["ok"] = False
        out["notes"].append("the hosting component is not closed: %d boundary "
                            "edges" % nav["host_open_edges"])

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
        # ...and again against the hosting component alone. The whole-surface
        # number counts a target sealed inside a detached island as "inside the
        # mesh"; only this one says whether the device can reach it.
        out["tgt_reachable"] = (float(_inside(nav["host_surf"], pool).mean())
                                if nav.get("host_surf") is not None else 0.0)
        if out["tgt_reachable"] < tfrac - 1e-9:
            out["notes"].append(
                "targets: %.0f%% lie inside the mesh but only %.0f%% inside the "
                "component the device is inserted into"
                % (100 * tfrac, 100 * out["tgt_reachable"]))
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


def selftest():
    """Construct the case the 'enclose' check cannot see, with the real mesher.

    A straight 100 mm vessel of radius 1.5 mm with a 4 mm neck at mid-length.
    The mesher marks the centerline into a 0.6/0.6/0.9 mm cube and smooths it
    twice with sigma = 1 voxel, so below a neck radius of about 0.8 mm the neck
    erodes away and the surface comes out as two closed tubes.

    At a 0.6 mm neck the vessel is severed and 'enclose' still scores 0.9950
    with a longest-outside-run of one point, 0.06 mm beyond the wall - it is
    testing "inside ANY closed sheet of the surface", and every route point is
    inside one tube or the other. The route check splits the surface into
    connected components first and finds that the tube the device is inserted
    into stops holding the route at mid-length.
    """
    import pyvista as pv
    from eve.intervention.vesseltree.util.branch import BranchWithRadii
    from eve.intervention.vesseltree.util.meshing import generate_temp_mesh

    z = np.arange(0.0, 100.25, 0.5)
    coords = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    ip = np.array([0.0, 0.0, 1.0])
    print("%7s | %6s | %9s %5s %7s %7s | %8s %9s %7s"
          % ("neck_r", "comps", "enclosed", "run", "out_mm", "enclose",
             "gap_mm", "handoff", "route"))
    ok = True
    for neck, want_severed in ((1.5, False), (1.0, False), (0.8, False),
                               (0.6, True), (0.4, True)):
        radii = np.full_like(z, 1.5)
        radii[np.abs(z - 50.0) <= 2.0] = neck
        path = generate_temp_mesh(
            [BranchWithRadii(name="synthetic", coordinates=coords, radii=radii)],
            "check_anatomies_selftest", 0.99)
        try:
            surf = pv.read(path).extract_surface().triangulate()
            ncomp = int(np.asarray(
                surf.connectivity().cell_data["RegionId"]).max()) + 1
            frac, worst, ins = enclosed(path, coords)
            depth = 0.0
            if (~ins).any():
                depth = float(np.linalg.norm(
                    np.asarray(surf.points)[None, :, :]
                    - coords[~ins][:, None, :], axis=2).min(axis=1).max())
            old_fail = bool(depth > MAX_OUTSIDE_MM and worst > MAX_OUTSIDE_RUN)
            nav = navigable_route(surf, coords, ip)
            new_fail = bool(nav["gap_mm"] > MAX_ROUTE_GAP_MM
                            or nav["handoff_mm"] > 0.0)
            print("%7.1f | %6d | %9.4f %5d %7.2f %7s | %8.2f %9.2f %7s"
                  % (neck, ncomp, frac, worst, depth,
                     "FAIL" if old_fail else "pass", nav["gap_mm"],
                     nav["handoff_mm"], "FAIL" if new_fail else "pass"))
            if new_fail != want_severed:
                ok = False
                print("        ^ the route check should have said %s"
                      % ("FAIL" if want_severed else "pass"))
            if want_severed and ncomp < 2:
                ok = False
                print("        ^ the mesher did not sever this one; the "
                      "construction no longer tests what it claims to")
            if neck == 0.6 and old_fail:
                print("        (note: 'enclose' now catches the 0.6 mm neck "
                      "too; the blindness demo needs a thinner neck)")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    print("selftest %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


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
    ap.add_argument("--selftest", action="store_true",
                    help="run the route check on a constructed severed vessel "
                         "and exit")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    names = sorted(d for d in os.listdir(a.anatomies)
                   if os.path.isdir(os.path.join(a.anatomies, d, "Centrelines_comb")))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        names = names[i::n]
    print("checking %d anatomies in %s\n" % (len(names), a.anatomies))
    print("%-16s %8s %7s %7s %6s %8s %6s %7s %7s %s"
          % ("anatomy", "route", "d_min", "tris", "open", "enclosed", "comps",
             "gap_mm", "targets", "status"))

    # Controls: the shipped tree as it ships, and the shipped tree with the
    # same stubs dropped. Any artifact the second one also shows is a property
    # of the meshing pipeline, not something the graft introduced.
    if a.host:
        for label, drop in (("HOST(shipped)", False), ("HOST(-stubs)", True)):
            try:
                r = check_one(label, a.host, a.verbose, drop_stubs=drop)
                print("%-16s %8.1f %7.2f %7d %6d %8.3f %6d %7.2f %7d %s"
                      % (label, r.get("route_mm", 0), r.get("dmin", 0),
                         r.get("tris", 0), r.get("open_edges", -1),
                         r.get("encl_frac", 0), r.get("n_components", -1),
                         r.get("route_gap_mm", -1), r.get("n_targets", 0),
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
        print("%-16s %8.1f %7.2f %7d %6d %8.3f %6d %7.2f %7d %s"
              % (n, r.get("route_mm", 0), r.get("dmin", 0), r.get("tris", 0),
                 r.get("open_edges", -1), r.get("encl_frac", 0),
                 r.get("n_components", -1), r.get("route_gap_mm", -1),
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
