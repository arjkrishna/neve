#!/usr/bin/env python
"""Visualize the eval-only pid19116-best-state run in SOFA.

Combines `visualize_policy.py`'s SofaPygame rendering with the
single-state-restore wrapper used by `eval_policy_from_state.py`. Loads
the eval-#1 policy (32.67 % mixed-state / 85.7 % from pid19116) and runs
the first N seeds of EVAL_SEEDS from the pid19116 restore state, rendering
each step in a SOFA window.

REQUIRES X11 display forwarding in Docker:
  docker run -e DISPLAY=<host-ip>:0 ...
Or Linux host: -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix

Usage (inside docker, with X11 set up):
  python3 /opt/eve_training/training_scripts/visualize_eval_best_state.py \
    --episodes 5

Optional overrides:
  --checkpoint <path>          override default eval-#1 checkpoint
  --snapshot   <policy_NN.pt>  use snapshot weights with the checkpoint's arch
  --restore_dir <dir>          override pid19116 restore dir
  --seed_start_idx N           start at EVAL_SEEDS[N] instead of 0
  --max_steps N                override env truncation
  --stochastic                 sample actions instead of deterministic
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import glob
import numpy as np

import eve  # noqa: F401  (registers config classes)
import eve_rl
from eve.visualisation import SofaPygame
from eve_bench import DualDeviceNav

# Make `util.*` importable (matches DualDeviceNav_train launch pattern).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util.env5 import BenchEnv5  # noqa: E402


# The 98 EVAL_SEEDS from DualDeviceNav_train.py — same seeds used by eval-#1.
EVAL_SEEDS = [
    1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 27, 31, 34,
    35, 37, 39, 42, 43, 44, 47, 48, 50, 52, 55, 56, 58, 61, 62, 63, 68, 69, 70,
    71, 73, 79, 80, 81, 84, 89, 91, 92, 93, 95, 97, 102, 103, 108, 109, 110,
    115, 116, 117, 118, 120, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132,
    134, 136, 138, 139, 140, 141, 142, 143, 144, 147, 148, 149, 150, 151, 152,
    154, 155, 156, 158, 159, 161, 162, 167, 168, 171, 175,
]

# Defaults that match the eval-only run we already shipped.
DEFAULT_CHECKPOINT = (
    "/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/"
    "2026-05-25_031951_rcca_awac_v3/checkpoints/checkpoint250531.everl"
)
DEFAULT_RESTORE_DIR = "/opt/eve_training/results/rcca_best_state_pid19116"

# Named-color presets -> RGB float in [0,1]. These map straight onto the device
# `color` field that SOFA's OglModel renders (eve/intervention/device/jshaped.py).
# Any "r,g,b" string of floats in [0,1] is also accepted via parse_color().
_NAMED_COLORS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
}


def parse_color(s):
    """Parse a device color argument.

    Accepts either a named color (e.g. ``blue``, ``cyan``) or three
    comma-separated RGB floats in [0,1] (e.g. ``0,0,1``). Returns an
    (R,G,B) float tuple suitable for a device's ``.color`` attribute.
    """
    if s is None:
        return None
    key = str(s).strip().lower()
    if key in _NAMED_COLORS:
        return _NAMED_COLORS[key]
    parts = [p for p in key.replace(";", ",").split(",") if p != ""]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"color must be a name ({', '.join(sorted(_NAMED_COLORS))}) "
            f"or 'r,g,b' / 'r,g,b,a'; got {s!r}")
    try:
        rgb = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"could not parse RGB(A) floats from {s!r}")
    if any(c < 0.0 or c > 1.0 for c in rgb):
        raise argparse.ArgumentTypeError(f"RGB(A) channels must be in [0,1]; got {rgb}")
    return rgb


def load_algo(checkpoint_path: Path, snapshot_path: Optional[Path]):
    """Load eve_rl AlgoPlayOnly. If snapshot given, overlay its policy weights."""
    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(str(checkpoint_path))
    if snapshot_path is not None:
        snap = torch.load(str(snapshot_path), map_location="cpu")
        if "policy" not in snap:
            raise ValueError(f"snapshot does not contain 'policy' key: {list(snap.keys())}")
        algo.model.policy.load_state_dict(snap["policy"])
        print(f"  overrode policy weights from snapshot (explore_step="
              f"{snap.get('explore_step', '?')})")
    return algo


def restrict_target_to_siphon(env, depth_mm: float = 25.0):
    """Force the RCCA target sampler to draw only from the carotid SIPHON —
    the deep-ICA distal end of the RCCA centerline.

    The siphon is the deepest ``depth_mm`` of the RCCA branch, measured in the
    C-arm tracking frame where z grows with depth (in this dataset CCA ~z416,
    cervical-ICA ~z510, siphon ~z575-601). We patch CenterlineRandom's
    point-cloud builder so the restriction survives any pool re-init. Because
    the target is chosen during env.reset()'s target.reset() — BEFORE the
    pathfinder / observation / reward reset — the whole chain (planned path,
    guidance obs, reward shaping, and the rendered target sphere) is built
    consistently for the siphon target.
    """
    from eve.util.coordtransform import vessel_cs_to_tracking3d

    target = env.intervention.target
    rcca = "Centerline curve - RCCA.mrk"

    def _tracking_z(pt_vessel_cs):
        return float(
            vessel_cs_to_tracking3d(
                pt_vessel_cs,
                target.fluoroscopy.image_rot_zx,
                target.fluoroscopy.image_center,
                target.fluoroscopy.field_of_view,
            )[2]
        )

    def _filter():
        pools = target._branch_targets
        if not pools.get(rcca):
            print("  WARNING: RCCA target pool empty; cannot restrict to siphon")
            return
        zs = np.array([_tracking_z(p) for p in pools[rcca]])
        z_max = float(zs.max())
        z_cut = z_max - float(depth_mm)
        keep = [p for p, z in zip(pools[rcca], zs) if z >= z_cut]
        pools[rcca] = keep
        target._potential_targets = np.asarray(keep)
        print(f"  SIPHON target: restricted RCCA pool to {len(keep)} distal pts "
              f"(tracking-z in [{z_cut:.1f}, {z_max:.1f}] mm)")

    _orig_build = target._init_centerline_point_cloud

    def _build_then_filter():
        _orig_build()
        _filter()

    target._init_centerline_point_cloud = _build_then_filter
    # If the pool was already built by an earlier reset, filter it now too.
    if target._branch_targets:
        _filter()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="Path to .everl checkpoint (default: eval-#1 of rcca_awac_v3)")
    p.add_argument("--snapshot", type=str, default=None,
                   help="Optional policy_XXXXX.pt to override policy weights")
    p.add_argument("--restore_dir", type=str, default=DEFAULT_RESTORE_DIR,
                   help="Dir containing the single start-state .npz "
                        "(default: rcca_best_state_pid19116)")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes to visualize (default 5)")
    p.add_argument("--seed_start_idx", type=int, default=0,
                   help="Start at EVAL_SEEDS[N]; runs N to N+episodes-1")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override env truncation max_steps")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic actions instead of deterministic eval mean")
    p.add_argument("--gw_color", type=parse_color, default=None,
                   help="Guidewire (mic_guide) color: a name "
                        f"({', '.join(sorted(_NAMED_COLORS))}) or 'r,g,b' in [0,1] "
                        "(default: black from JShaped)")
    p.add_argument("--cath_color", type=parse_color, default=None,
                   help="Catheter (mic_cath) color: a name or 'r,g,b' in [0,1] "
                        "(default: red from DualDeviceNav)")
    p.add_argument("--gw_alpha", type=float, default=None,
                   help="Guidewire opacity in [0,1] (<1 = see-through)")
    p.add_argument("--cath_alpha", type=float, default=None,
                   help="Catheter opacity in [0,1]; e.g. 0.35 makes the cyan "
                        "catheter translucent so the guidewire inside shows "
                        "through (see both at once)")
    p.add_argument("--target_region", choices=["random", "siphon"],
                   default="random",
                   help="Where to place the RCCA target: 'random' (anywhere on "
                        "the RCCA centerline, default) or 'siphon' (deep-ICA "
                        "distal end of the RCCA centerline only)")
    p.add_argument("--siphon_depth_mm", type=float, default=25.0,
                   help="Siphon depth: keep the deepest N mm (by tracking-z) of "
                        "the RCCA centerline as the target pool (default 25)")
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    snapshot_path = Path(args.snapshot) if args.snapshot else None
    if snapshot_path and not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    restore_dir = Path(args.restore_dir)
    if not restore_dir.exists():
        raise FileNotFoundError(f"Restore dir not found: {restore_dir}")

    seeds = EVAL_SEEDS[args.seed_start_idx:args.seed_start_idx + args.episodes]
    if len(seeds) < args.episodes:
        print(f"  WARNING: only {len(seeds)} seeds available from index "
              f"{args.seed_start_idx} (asked for {args.episodes})")

    print(f"Loading policy: {checkpoint_path.name}")
    algo = load_algo(checkpoint_path, snapshot_path)

    print(f"Building eval env from scratch (DualDeviceNav insertion_z=345 + BenchEnv5)")
    intervention = DualDeviceNav(insertion_z=345)
    env = BenchEnv5(
        intervention=intervention,
        mode="eval",
        visualisation=False,
        default_target_branch="Centerline curve - RCCA.mrk",
    )
    env.intervention.make_non_mp()

    # Optional device recoloring / transparency. SOFA's OglModel reads each
    # device's `.color` when the scene is (re)built on the first env.reset()
    # below, so we set it here — before SofaPygame attaches and before any
    # reset. devices[0] is the guidewire (mic_guide), devices[1] is the
    # catheter (mic_cath). A 4th (alpha) channel < 1 makes that device
    # translucent: giving the catheter alpha ~0.35 lets the coaxial guidewire
    # inside it stay visible (see both at once). _add_devices in
    # sofabeamadapter.py propagates this alpha into the OglModel material.
    def _resolve_color(device, color, alpha):
        if color is None and alpha is None:
            return None  # leave device's default color untouched
        base = list(color) if color is not None else list(device.color)
        rgb = [float(base[0]), float(base[1]), float(base[2])]
        if alpha is not None:
            if not 0.0 <= alpha <= 1.0:
                raise SystemExit(f"alpha must be in [0,1]; got {alpha}")
            return tuple(rgb + [float(alpha)])
        return tuple(base)  # color as given (3- or 4-tuple), no alpha override

    devices = env.intervention.devices
    gw_color = _resolve_color(devices[0], args.gw_color, args.gw_alpha)
    cath_color = _resolve_color(devices[1], args.cath_color, args.cath_alpha)
    if gw_color is not None:
        devices[0].color = gw_color
        print(f"  guidewire (mic_guide) color -> "
              f"{tuple(round(c, 3) for c in gw_color)}")
    if cath_color is not None:
        devices[1].color = cath_color
        print(f"  catheter  (mic_cath)  color -> "
              f"{tuple(round(c, 3) for c in cath_color)}")

    # Optional: pin the target to the carotid siphon (deep-ICA distal end of
    # the RCCA centerline) instead of a random RCCA point. Installed before the
    # first reset so every episode's target.reset() draws from the siphon pool.
    if args.target_region == "siphon":
        print("Restricting target to the carotid siphon (deep ICA, distal RCCA)")
        restrict_target_to_siphon(env, args.siphon_depth_mm)

    # Load the single pid19116 restore checkpoint once at startup. We do NOT
    # wrap with CheckpointRestoreWrapper here (it seems to break the gym
    # step() chain when SofaPygame is attached); instead we manually inject
    # restore_checkpoint into env.reset(options=...) in the loop below.
    ckpt_files = sorted(glob.glob(os.path.join(str(restore_dir), "*.npz")))
    if not ckpt_files:
        raise SystemExit(f"No .npz files in {restore_dir}")
    restore_name = os.path.basename(ckpt_files[0])
    print(f"Restore checkpoint ({len(ckpt_files)} file): {restore_name}")
    if len(ckpt_files) > 1:
        print("  WARNING: restore_dir has >1 file; using only the first.")
    with np.load(ckpt_files[0]) as _data:
        restore_payload = {k: np.array(_data[k]) for k in _data.files}
    if args.max_steps is not None:
        env.truncation.max_steps = args.max_steps

    # Attach SofaPygame directly to the env (no wrapper).
    visu = SofaPygame(env.intervention)
    env.visualisation = visu

    # Mirror the action denormalization that eve_rl's Single agent applies
    # when normalize_actions=True (which is what BenchAgentSynchron uses).
    # Without this, raw tanh outputs [-1,+1] get fed into an env whose action
    # space is e.g. [-30,+30] and motion is ~30x slower than at training time.
    a_low = np.asarray(env.action_space.low, dtype=np.float32)
    a_high = np.asarray(env.action_space.high, dtype=np.float32)
    print(f"env action_space: low={a_low.tolist()}, high={a_high.tolist()}")

    def denormalize(act):
        return (np.asarray(act, dtype=np.float32) + 1.0) / 2.0 * (a_high - a_low) + a_low

    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"\nVisualizing {len(seeds)} episodes (seeds={seeds}) with {mode} actions")
    print(f"Restoring every episode from: {restore_name}")
    print("-" * 70)

    n_success = 0
    for i, seed in enumerate(seeds):
        algo.reset()
        # Inject restore_checkpoint directly via options — same payload the
        # CheckpointRestoreWrapper would build, but with no wrapper in between.
        reset_options = {
            "restore_checkpoint": restore_payload,
            "_restore_checkpoint_file": restore_name,
            "_restore_checkpoint_idx": 0,
        }
        reset_result = env.reset(seed=seed, options=reset_options)
        if isinstance(reset_result, tuple):
            obs, _info = reset_result
        else:
            obs = reset_result
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
        tgt = np.asarray(env.intervention.target.coordinates3d, dtype=float)
        print(f"  ep{i+1} target (tracking3d) = "
              f"[{tgt[0]:+.1f}, {tgt[1]:+.1f}, {tgt[2]:+.1f}]  (z grows with depth)")

        ep_reward, steps = 0.0, 0
        term = trunc = False
        while True:
            if args.stochastic:
                action = algo.get_exploration_action(obs_flat)
            else:
                action = algo.get_eval_action(obs_flat)
            env_action = denormalize(action.reshape(env.action_space.shape))
            obs, r, term, trunc, info = env.step(env_action)
            obs_flat, _ = eve_rl.util.flatten_obs(obs)
            ep_reward += r
            steps += 1
            env.render()
            # Per-step diagnostic so we can see if the sim is actually advancing
            try:
                ins = env.intervention.device_lengths_inserted
                t3 = env.intervention.fluoroscopy.tracking3d
                tip_z = float(t3[0][2]) if t3 is not None and len(t3) > 0 else float("nan")
                gw = float(ins[0]) if ins is not None else float("nan")
                cath = float(ins[1]) if ins is not None and len(ins) > 1 else float("nan")
            except Exception as _e:
                tip_z = gw = cath = float("nan")
            a = env_action.flatten() if hasattr(env_action, "flatten") else env_action
            print(f"  ep{i+1} step={steps:3d}  "
                  f"act=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]  "
                  f"r={r:+.3f}  cumR={ep_reward:+.3f}  "
                  f"gw={gw:.1f} cath={cath:.1f} tip_z={tip_z:.1f}  "
                  f"term={int(term)} trunc={int(trunc)}",
                  flush=True)
            if term or trunc:
                break

        success = bool(info.get("success", False))
        if success:
            n_success += 1
        status = "SUCCESS" if success else ("trunc " if trunc else "term  ")
        print(f"[{i + 1:2d}/{len(seeds)}] seed={seed:4d}  {status}  "
              f"R={ep_reward:+8.3f}  steps={steps:4d}  "
              f"term={int(term)} trunc={int(trunc)}")

    pct = 100.0 * n_success / len(seeds) if seeds else 0
    print("-" * 70)
    print(f"RESULT: {n_success}/{len(seeds)} success ({pct:.1f}%) from {restore_name}")

    try:
        algo.close()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
