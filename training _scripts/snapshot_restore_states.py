"""Render SOFA fluoroscopy snapshots of restore-state checkpoints.

Builds the BenchEnv5 SOFA env exactly as DualDeviceNav_train does, then for
each pre-bif(11) checkpoint .npz: restores the SOFA state via
env.reset(options={"restore_checkpoint": ...}) and writes one snapshot in
the SAME format as the in-run episode snapshots (util.snapshot.save_snapshot
— device wires + centerlines/mesh + planned path + target).

NO training / explore / eval is run. One env, sequential restores.

Usage (inside the SOFA container):
    python3 snapshot_restore_states.py \
        --checkpoint_dir /opt/eve_training/results/rcca_pre_bif11_checkpoints \
        --out_dir       /opt/eve_training/results/restore_state_snapshots_sofa \
        --good pre_bif11_pid20043_ep0003_step0038.npz,pre_bif11_pid3285_ep0046_step0132.npz \
        --n_early 148 --mode centerlines
"""
import argparse
import glob
import os

import numpy as np

from util.env5 import BenchEnv5
from util.snapshot import save_snapshot
from eve_bench import DualDeviceNav


def render_group(env, files, ckdir, out_group, target_branch, mode):
    os.makedirs(out_group, exist_ok=True)
    n_ok = 0
    for i, fn in enumerate(files):
        path = os.path.join(ckdir, fn)
        try:
            with np.load(path) as data:
                ckpt = {k: np.array(data[k]) for k in data.files}
        except Exception as e:
            print(f"  [skip] {fn}: load failed: {e}", flush=True)
            continue
        try:
            env.reset(
                seed=1000 + i,
                options={"restore_checkpoint": ckpt, "target_branch": target_branch},
            )
        except Exception as e:
            print(f"  [skip] {fn}: reset/restore failed: {e}", flush=True)
            continue
        try:
            sim = env.intervention.simulation
            dofp = np.asarray(sim.dof_positions)
            ins = np.asarray(sim.inserted_lengths)
            stored_tip = np.asarray(ckpt["tracking3d"])[0]
            print(
                f"  DBG {fn}: live_dof_z=[{dofp[:,2].min():.1f},{dofp[:,2].max():.1f}] "
                f"live_tip={dofp[0].round(1)} inserted={ins.round(1)} "
                f"stored_tip_z={stored_tip[2]:.1f}",
                flush=True,
            )
        except Exception as e:
            print(f"  DBG {fn}: probe failed: {e}", flush=True)
        p = save_snapshot(
            env,
            episode=i,
            ep_step=0,
            reason="state",
            reward=0.0,
            phase="restore_state",
            base_dir_override=os.path.join(out_group, "_tmp"),
            mode_override=mode,
        )
        if p and os.path.isfile(p):
            dst = os.path.join(out_group, fn.replace(".npz", ".png"))
            os.replace(p, dst)
            n_ok += 1
            if n_ok <= 5 or n_ok % 25 == 0:
                print(f"  [{n_ok}] {os.path.basename(dst)}", flush=True)
        else:
            print(f"  [warn] {fn}: snapshot returned None", flush=True)
    # clean tmp
    tmp = os.path.join(out_group, "_tmp")
    if os.path.isdir(tmp):
        for r, _, fnames in os.walk(tmp, topdown=False):
            for f in fnames:
                os.remove(os.path.join(r, f))
            os.rmdir(r)
    return n_ok


def main(a):
    target_branch = a.target_branch
    print(f"Building env (insertion_z={a.insertion_z}, target={target_branch}, mode={a.mode})", flush=True)
    intervention = DualDeviceNav(insertion_z=a.insertion_z)
    env = BenchEnv5(
        intervention=intervention,
        mode="train",
        visualisation=False,
        default_target_branch=target_branch,
    )

    all_files = sorted(glob.glob(os.path.join(a.checkpoint_dir, "*.npz")), key=os.path.getmtime)
    all_names = [os.path.basename(f) for f in all_files]
    print(f"pool: {len(all_names)} checkpoints in {a.checkpoint_dir}", flush=True)

    good = [g.strip() for g in a.good.split(",") if g.strip()] if a.good else []
    early = all_names[: a.n_early]

    # SOFA first-reset quirk: the very first restore-RENDER after scene build
    # is malformed (fluoroscopy transform/device-tracking not fully
    # initialized) even though the DOF restore itself is numerically correct.
    # Do one throwaway warm-up restore (no snapshot) so every real snapshot
    # below comes from a settled (>=2nd) reset.
    warm = (good[:1] or early[:1])
    if warm:
        try:
            with np.load(os.path.join(a.checkpoint_dir, warm[0])) as data:
                wck = {k: np.array(data[k]) for k in data.files}
            env.reset(seed=999, options={"restore_checkpoint": wck, "target_branch": target_branch})
            print(f"warm-up reset done (absorbs SOFA first-episode quirk): {warm[0]}", flush=True)
        except Exception as e:
            print(f"warm-up reset failed: {e}", flush=True)

    if good:
        print(f"\n=== GOOD ({len(good)}) ===", flush=True)
        ng = render_group(env, good, a.checkpoint_dir, os.path.join(a.out_dir, "good"),
                          target_branch, a.mode)
        print(f"good done: {ng}/{len(good)}", flush=True)

    print(f"\n=== EARLY-{a.n_early} (mostly bad) ===", flush=True)
    nb = render_group(env, early, a.checkpoint_dir,
                      os.path.join(a.out_dir, f"early{a.n_early}_mostly_bad"),
                      target_branch, a.mode)
    print(f"early done: {nb}/{len(early)}", flush=True)
    print(f"\nALL DONE -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--good", default="")
    p.add_argument("--n_early", type=int, default=148)
    p.add_argument("--mode", default="centerlines", choices=["centerlines", "mesh"])
    p.add_argument("--insertion_z", type=int, default=345)
    p.add_argument("--target_branch", default="Centerline curve - RCCA.mrk")
    main(p.parse_args())
