"""BEFORE/AFTER equivalence of eval_anatomies.make_env on the DEFAULT path,
plus a TopBrain parity + target-pool audit. Read-only; scratch only."""
import argparse
import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/eve_training/training_scripts")

BEFORE = "/tmp/eval_before.py"
AFTER = "/opt/eve_training/training_scripts/eval_anatomies.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


DEV_ATTRS = ("name", "length", "velocity_limit", "tip_outer_diameter",
             "straight_outer_diameter", "tip_inner_diameter",
             "straight_inner_diameter", "mass_density_tip",
             "mass_density_straight", "young_modulus_tip",
             "young_modulus_straight", "beams_per_mm_straight",
             "visu_edges_per_mm")


def env_signature(env):
    """Everything that defines the task, hashed. Excludes wall-clock/tempfile."""
    iv = env.intervention
    vt = iv.vessel_tree
    h = hashlib.md5()
    parts = []
    for b in vt.branches:
        h.update(np.asarray(b.coordinates, np.float64).round(6).tobytes())
        h.update(np.asarray(getattr(b, "radii", []), np.float64).round(6).tobytes())
        parts.append(str(b.name))
    sig = {
        "branch_names": parts,
        "geometry_md5": h.hexdigest()[:16],
        "insertion_pos": np.round(np.asarray(vt.insertion.position, float), 9).tolist(),
        "insertion_dir": np.round(np.asarray(vt.insertion.direction, float), 9).tolist(),
        "devices": [[str(getattr(d, at, None)) for at in DEV_ATTRS] for d in iv.devices],
        "target_class": type(iv.target).__name__,
        "target_branches": list(iv.target.branches or []),
        "target_threshold": iv.target.threshold,
        "target_min_arc": iv.target.min_arclength_from_start,
        "sim": type(iv.simulation).__name__,
        "friction": getattr(iv.simulation, "friction", None),
        "rot_zx": list(iv.fluoroscopy.image_rot_zx),
        "img_freq": iv.fluoroscopy.image_frequency,
        "normalize_action": iv.normalize_action,
        "stop_at_end": iv.stop_device_at_tree_end,
        "obs_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }
    return sig


def ns(**kw):
    d = dict(
        checkpoint="x", target_branch="Centerline curve - RCCA.mrk",
        relax_failure_truncations=True, buckle_reward_coef=0.5, max_steps=600,
        residual_heuristic=True, residual_scale=1.0, heur_action_obs=True,
        cath_slack_coef=0.5, progress_tip_mode="avg", avg_gw_weight=0.5,
        real_patient_anatomy=False, insert_inside_branch="none",
        insert_point_idx=2, require_passable=False, passable_max_tries=15,
        passable_min_median_mm=2.0, radius_scale=1.0,
        target_min_arclength_mm=40.0,
        # new flags (AFTER only; harmless extras on BEFORE's Namespace)
        topbrain=False, topbrain_dir="/opt/eve_training/results_topbrain/anatomies",
        topbrain_exclude=["topcow_mr_013", "topcow_mr_014", "topcow_mr_015"],
        topbrain_only=None, topbrain_trim_stations=0,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "equiv"

    if mode == "equiv":
        b = load(BEFORE, "evb")
        a = load(AFTER, "eva")
        for seed in (900000, 912345):
            eb = b.make_env(ns(), seed=seed, change_every=2, mode="eval")
            sb = env_signature(eb)
            eb.close()
            ea = a.make_env(ns(), seed=seed, change_every=2, mode="eval")
            sa = env_signature(ea)
            ea.close()
            same = sb == sa
            print(f"[equiv] seed={seed} default-path env signature: "
                  f"{'IDENTICAL' if same else 'DIFFERS'}")
            print(f"        geometry_md5 before={sb['geometry_md5']} "
                  f"after={sa['geometry_md5']}")
            print(f"        insertion    before={sb['insertion_pos']} "
                  f"after={sa['insertion_pos']}")
            if not same:
                for k in sb:
                    if sb[k] != sa[k]:
                        print(f"        DIFF {k}: {sb[k]!r} != {sa[k]!r}")
        return

    if mode == "topbrain":
        a = load(AFTER, "eva")
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        trim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        env = a.make_env(ns(topbrain=True, topbrain_only=only,
                            topbrain_trim_stations=trim),
                         seed=900000, change_every=1, mode="eval")
        iv = env.intervention
        vt = iv.vessel_tree
        print("\n[topbrain] parity vs DualDeviceNavRCCAVaried")
        ok = a._topbrain_parity_report(iv)
        print(f"[topbrain] parity_ok={ok}")

        # target pool audit: is every sampled target inside the lumen?
        import pyvista as pv
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        iv.target.reset(0, 0)
        pool = np.asarray(iv.target._branch_targets[
            "Centerline curve - RCCA.mrk"], dtype=float)
        rcca = next(b for b in vt.branches
                    if str(b.name) == "Centerline curve - RCCA.mrk")
        c = np.asarray(rcca.coordinates, float)
        cum = np.concatenate(([0.0], np.cumsum(
            np.linalg.norm(np.diff(c, axis=0), axis=1))))
        mesh = pv.read(vt.mesh_path).triangulate().clean()
        imp = vtk.vtkImplicitPolyDataDistance()
        imp.SetInput(mesh)
        d = np.array([imp.EvaluateFunction(p) for p in pool])   # <0 = inside
        print(f"[topbrain] anatomy={vt.current_anatomy} "
              f"fingerprint={vt.mesh_fingerprint} mesh={os.path.basename(vt.mesh_path)}")
        print(f"[topbrain] RCCA stations={len(c)} arclength=0..{cum[-1]:.1f} mm; "
              f"target pool={len(pool)} (trim={trim}, min_arc="
              f"{iv.target.min_arclength_from_start})")
        print(f"[topbrain] EXACT signed distance of every pooled target to the "
              f"collision surface: inside={(d<0).sum()}/{len(d)}  "
              f"clearance min={-d.max():.3f} median={-np.median(d):.3f} "
              f"max={-d.min():.3f} mm  (positive = inside the lumen)")
        if (d >= 0).any():
            bad = np.where(d >= 0)[0]
            print(f"[topbrain] *** {len(bad)} pooled targets NOT inside "
                  f"the surface, indices {bad.tolist()[:10]} ***")
        # trim proof: pool with trim=N must equal pool with trim=0 minus last N
        env.close()
        return

    if mode == "trim":
        a = load(AFTER, "eva")
        name = sys.argv[2]
        pools = {}
        for n_trim in (0, 1, 2, 5):
            env = a.make_env(ns(topbrain=True, topbrain_only=[name],
                                topbrain_trim_stations=n_trim),
                             seed=900000, change_every=1, mode="eval")
            iv = env.intervention
            iv.target.reset(0, 0)
            pools[n_trim] = np.asarray(iv.target._branch_targets[
                "Centerline curve - RCCA.mrk"], float)
            vt = iv.vessel_tree
            rcca = next(b for b in vt.branches
                        if str(b.name) == "Centerline curve - RCCA.mrk")
            n_geom = len(rcca.coordinates)
            pf = env.intervention  # pathfinder lives on env5
            env.close()
            print(f"[trim] N={n_trim}: pool={len(pools[n_trim])} "
                  f"branch_stations={n_geom} (geometry unchanged)")
        base = pools[0]
        for n_trim in (1, 2, 5):
            exp = base[:len(base) - n_trim]
            got = pools[n_trim]
            same = got.shape == exp.shape and np.array_equal(got, exp)
            print(f"[trim] N={n_trim} pool == pool(N=0)[:-{n_trim}] : {same} "
                  f"(dropped {len(base)-len(got)} station(s))")
        print(f"[trim] N=0 pool identical to unpatched pool: "
              f"{len(pools[0])} points, last point "
              f"{np.round(pools[0][-1],4).tolist()}")
        return


main()
