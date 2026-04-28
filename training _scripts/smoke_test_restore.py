"""Single-process smoke test for SOFA checkpoint restore.

Instantiates BenchEnv5, picks one .npz from --checkpoint_dir, calls
env.reset(options={"restore_checkpoint": state}), prints the resulting
insertion and tracking3d samples, then takes a few zero-velocity steps to
see if SOFA's internal settling pulls the wire back to 0mm.

The expected output if restore works:
    before reset: inserted_gw ≈ 0 (or stale from construction)
    after reset:  inserted_gw ≈ 370-385 mm (from the .npz)
    step 1:       inserted_gw ≈ 370-385 mm (zero velocity = no change)

If restore is broken, 'after reset' and step 1 will both show ~0 mm.

Usage (inside the SOFA container):
    python3 /opt/eve_training/training_scripts/smoke_test_restore.py \
        --checkpoint_dir /opt/eve_training/results/sofa_checkpoints/selected \
        --pick 0
"""

import argparse
import glob
import os
import sys

import numpy as np

from util.env5 import BenchEnv5
from eve_bench import DualDeviceNav


def _pp(label, env):
    try:
        dli = env.intervention.device_lengths_inserted
        ins = list(dli) if dli is not None else None
    except Exception as e:
        ins = f"err:{e}"
    try:
        tr = np.asarray(env.intervention.fluoroscopy.tracking3d)
        tip = tr[0].tolist() if len(tr) > 0 else None
        n_nonzero = int(np.sum(np.linalg.norm(tr, axis=1) > 1e-6)) if len(tr) > 0 else 0
        n_dofs = len(tr)
    except Exception as e:
        tip, n_nonzero, n_dofs = f"err:{e}", 0, 0
    print(
        f"[{label}] inserted={ins}  tip3d={tip}  "
        f"n_dofs={n_dofs}  n_nonzero_dofs={n_nonzero}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--pick", type=int, default=0, help="Index into sorted .npz list")
    ap.add_argument("--settle_steps", type=int, default=5,
                    help="Passed through to restore_checkpoint via settle_steps")
    ap.add_argument("--noop_steps", type=int, default=5,
                    help="Zero-velocity env.step calls after reset to observe drift")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.npz")))
    if not files:
        print(f"No .npz in {args.checkpoint_dir}", file=sys.stderr)
        return 1
    path = files[args.pick % len(files)]
    print(f"Using checkpoint: {os.path.basename(path)}", flush=True)

    with np.load(path) as d:
        state = {k: np.array(d[k]) for k in d.files}
    print(
        f"Saved xtip={state['xtip'].tolist()}  "
        f"index_first_node={int(state['index_first_node'])}  "
        f"dof_positions.shape={state['dof_positions'].shape}",
        flush=True,
    )

    intervention = DualDeviceNav()
    env = BenchEnv5(intervention=intervention, mode="train", visualisation=False)

    _pp("before-reset", env)

    obs, _ = env.reset(
        seed=42,
        options={"restore_checkpoint": {k: state[k] for k in state}, "heuristic_mode": True},
    )
    _pp("after-reset", env)

    # Zero-velocity action: normalized space → mapping to env action low/high.
    # With normalize_actions=True the mapping is (a+1)/2*(high-low)+low. For
    # a zero-commanded-translation/rotation, the normalized action is chosen
    # so the mapped action is zero. For symmetric high/low this is 0.0. With
    # asymmetric bounds we solve a = -(high+low)/(high-low). Simpler: disable
    # normalize_actions by passing raw zero here and letting the env clip.
    zero_raw = np.zeros(env.action_space.shape, dtype=np.float32)
    zero_normalized = np.zeros_like(zero_raw)
    for i in range(args.noop_steps):
        obs, reward, term, trunc, info = env.step(zero_normalized)
        _pp(f"step-{i+1}", env)
        if term or trunc:
            print(f"episode ended at step {i+1}: term={term} trunc={trunc}", flush=True)
            break

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)
