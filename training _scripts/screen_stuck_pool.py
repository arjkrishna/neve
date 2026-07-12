"""Gen-4 #6 — escapability screener for the stuck-state restore pool.

Restores each stuck checkpoint, runs a scripted pure-retract for K steps,
and keeps only the states a retract can actually recover — so the recovery
curriculum never trains from mechanically-wedged states that would teach the
critic "recovery is impossible" (the audit's #6). The verdict logic lives in
util/escapability.py (unit-tested without SOFA); this file is the SOFA
orchestration and runs IN THE CONTAINER, same image as training.

Usage (inside the eve-training image, pool mounted at RESULTS):
    python3 /opt/eve_training/training_scripts/screen_stuck_pool.py \
        --pool_dir  /opt/eve_training/results/rcca_proc_stuck \
        --out_dir   /opt/eve_training/results/rcca_proc_stuck/escapable \
        --procedural --procedural_seed 12345 \
        --steps 40 --retract_mm_s 20

Fixed-mesh pool (DualDeviceNav): drop --procedural (and pass --insertion_z if
the pool was harvested from a non-default insertion). Escapable checkpoints
(+ their .json sidecars) are COPIED into --out_dir; point a recovery-
curriculum run's --checkpoint_dir at that subdir. A screen_report.json
summarises every verdict.

The screener restores through the SAME BenchEnv5.reset(restore_checkpoint=...)
path training uses, and for procedural pools pins the vessel tree to each
checkpoint's mesh fingerprint (parsed from the filename) BEFORE restore — so
the state lands in the exact siphon it was captured on (#3).
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.env5 import BenchEnv5  # noqa: E402
from util.escapability import escape_metrics, restore_faithful  # noqa: E402
from eve.intervention.vesseltree.rccavariedfrommesh import (  # noqa: E402
    RCCAVariedFromMesh,
)

_RCCA = "Centerline curve - RCCA.mrk"


def _build_env(args):
    """Build a single BenchEnv5 on the right intervention. Procedural pools
    use DualDeviceNavRCCAVaried (pinnable per checkpoint); fixed pools use
    DualDeviceNav."""
    if args.procedural:
        from eve_bench import DualDeviceNavRCCAVaried
        interv = DualDeviceNavRCCAVaried(
            seed=int(args.procedural_seed),
            # Never self-regenerate: the pin dictates the mesh per checkpoint.
            episodes_between_change=10 ** 9,
        )
        target_branch = _RCCA
    else:
        from eve_bench import DualDeviceNav
        interv = DualDeviceNav(insertion_z=args.insertion_z)
        target_branch = args.target_branch
    env = BenchEnv5(
        intervention=interv, mode="eval", visualisation=False,
        default_target_branch=target_branch,
    )
    return env, target_branch


def _fp_from_name(path):
    import re
    m = re.search(r"_mesh-([A-Za-z0-9]+)_", os.path.basename(path))
    return m.group(1) if m else None


def _load_ckpt(path):
    with np.load(path) as data:
        return {k: np.array(data[k]) for k in data.files}


def _meta_for(npz_path):
    """The .json sidecar as a dict (reason + capture references), {} if none."""
    side = os.path.splitext(npz_path)[0] + ".json"
    if os.path.isfile(side):
        try:
            with open(side) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _reason_from(meta, npz_path):
    r = str(meta.get("reason", "")).lower()
    if r:
        return r
    base = os.path.basename(npz_path).lower()
    if "fold" in base:
        return "fold"
    if "off" in base:
        return "offpath"
    return "unknown"


def _captured_refs(meta):
    """(captured_inserted_gw_mm, captured_slack_mm) from the sidecar, or
    (None, None) when the checkpoint predates the fidelity fields."""
    ins = meta.get("inserted_lengths")
    cap_ins = float(ins[0]) if ins else None
    cap_slack = meta.get("slack_at_capture")
    if cap_slack is None and cap_ins is not None and "proj_s_at_capture" in meta:
        cap_slack = cap_ins - float(meta["proj_s_at_capture"])
    return cap_ins, (float(cap_slack) if cap_slack is not None else None)


def _slack(env):
    inserted_gw = float(env.intervention.device_lengths_inserted[0])
    proj_s = float(env._path_context.get_projection().s)
    return inserted_gw, inserted_gw - proj_s


def _on_path(env):
    try:
        return bool(env._path_context.is_on_correct_path())
    except Exception:
        return False


def screen_one(env, target_branch, npz_path, args):
    """Restore one checkpoint, verify the restore is FAITHFUL, then (only if
    so) retract K steps and judge escapability. Returns the verdict dict."""
    meta = _meta_for(npz_path)
    reason = _reason_from(meta, npz_path)
    cap_ins, cap_slack = _captured_refs(meta)
    ckpt = _load_ckpt(npz_path)

    # #3 — pin the vessel tree to this checkpoint's mesh before restore.
    if args.procedural:
        fp = _fp_from_name(npz_path)
        tree = env.intervention.vessel_tree
        if fp and hasattr(tree, "pin_next"):
            tree.pin_next(fp)

    options = {"restore_checkpoint": ckpt, "target_branch": target_branch}
    env.reset(seed=int(args.seed), options=options)

    inserted_gw_0, slack_0 = _slack(env)

    # --- restore-fidelity gate (the user's point: a buckled state may spring
    # / pop / penetrate on restore) --------------------------------------
    # If the restored wire does not match the captured fed-length AND bow,
    # the restore did not reproduce the stuck state -> the checkpoint is
    # unusable, regardless of how the retract goes. Reported as its OWN
    # bucket so the fraction lost to restore infidelity is visible (the real
    # diagnostic for whether restore-based recovery is viable at all).
    faithful = True
    if cap_ins is not None and cap_slack is not None:
        faithful = restore_faithful(cap_ins, inserted_gw_0, cap_slack, slack_0)
    if not faithful:
        return {
            "file": os.path.basename(npz_path), "reason": reason,
            "restore_faithful": False, "escapable": False,
            "captured_inserted": cap_ins, "restored_inserted": round(inserted_gw_0, 2),
            "captured_slack": cap_slack, "restored_slack": round(slack_0, 2),
        }

    # Scripted pure retract: both devices full reverse translation, no
    # rotation. Physical units (2 devices x [trans, rot]); the env clips to
    # the per-device velocity limits and masks below-zero, so an
    # over-retract can't drive the wire past the ostium.
    retract = float(args.retract_mm_s)
    action = np.array([[-retract, 0.0], [-retract, 0.0]], dtype=np.float32)
    for _ in range(int(args.steps)):
        try:
            env.step(action)
        except Exception as e:  # noqa: BLE001
            print(f"  step error on {os.path.basename(npz_path)}: {e}")
            break

    inserted_gw_k, slack_k = _slack(env)
    m = escape_metrics(
        reason,
        slack_start=slack_0,
        slack_end=slack_k,
        retract_mm=inserted_gw_0 - inserted_gw_k,
        on_path_end=_on_path(env),
    )
    m["restore_faithful"] = True
    m["file"] = os.path.basename(npz_path)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="Escapable checkpoints copied here (default "
                         "<pool_dir>/escapable).")
    ap.add_argument("--procedural", action="store_true",
                    help="Pool harvested under --procedural_rcca: pin each "
                         "checkpoint's mesh from its fingerprint before restore.")
    ap.add_argument("--procedural_seed", type=int, default=12345)
    ap.add_argument("--insertion_z", type=float, default=None,
                    help="Fixed-mesh only: insertion depth of the harvest run.")
    ap.add_argument("--target_branch", default=_RCCA)
    ap.add_argument("--steps", type=int, default=40,
                    help="Retract steps per checkpoint (default 40).")
    ap.add_argument("--retract_mm_s", type=float, default=20.0,
                    help="Retract translation speed, mm/s (default 20).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Screen at most N checkpoints (0 = all).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report verdicts but copy nothing.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.pool_dir, "escapable")
    files = sorted(glob.glob(os.path.join(args.pool_dir, "stuck_*.npz")))
    if not files:
        # fall back to any npz (fixed pools may use other prefixes)
        files = sorted(
            f for f in glob.glob(os.path.join(args.pool_dir, "*.npz"))
            if os.path.basename(os.path.dirname(f)) != "escapable"
        )
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"No checkpoints found in {args.pool_dir}")
        return 1

    env, target_branch = _build_env(args)
    print(f"Screening {len(files)} checkpoints "
          f"({'procedural/pinned' if args.procedural else 'fixed'} mesh, "
          f"{args.steps} retract steps @ {args.retract_mm_s} mm/s)")

    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    report, n_keep = [], 0
    for i, path in enumerate(files):
        m = screen_one(env, target_branch, path, args)
        report.append(m)
        if not m.get("restore_faithful", True):
            print(f"[{i + 1}/{len(files)}] UNFAITHFUL {m['file']} "
                  f"reason={m['reason']} inserted "
                  f"{m.get('captured_inserted')}->{m.get('restored_inserted')} "
                  f"slack {m.get('captured_slack')}->{m.get('restored_slack')} "
                  f"(restore did not reproduce the state — dropped)")
            continue
        tag = "KEEP" if m["escapable"] else "drop"
        print(f"[{i + 1}/{len(files)}] {tag} {m['file']} "
              f"reason={m['reason']} slack {m['slack_start']}->{m['slack_end']} "
              f"retract={m['retract_mm']}")
        if m["escapable"] and not args.dry_run:
            shutil.copy2(path, os.path.join(out_dir, m["file"]))
            side = os.path.splitext(path)[0] + ".json"
            if os.path.isfile(side):
                shutil.copy2(side, os.path.join(out_dir, os.path.basename(side)))
            n_keep += 1

    n_unfaithful = sum(1 for m in report if not m.get("restore_faithful", True))
    summary = {
        "pool_dir": args.pool_dir,
        "out_dir": out_dir,
        "n_total": len(files),
        # Restore infidelity is reported FIRST and separately: if it is high,
        # the stuck states do not survive SOFA restore (the buckle springs /
        # pops) and restore-based recovery is not viable — lean on live
        # in-situ buckling (--relax_failure_truncations) instead.
        "n_restore_unfaithful": n_unfaithful,
        "n_escapable": sum(1 for m in report if m.get("escapable")),
        "n_kept_copied": n_keep,
        "steps": args.steps,
        "retract_mm_s": args.retract_mm_s,
        "procedural": bool(args.procedural),
        "verdicts": report,
    }
    report_path = os.path.join(
        out_dir if not args.dry_run else args.pool_dir, "screen_report.json"
    )
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"could not write report: {e}")
    print(f"Done: {n_unfaithful}/{summary['n_total']} dropped for restore "
          f"infidelity; {summary['n_escapable']}/{summary['n_total']} escapable "
          f"({'copied ' + str(n_keep) if not args.dry_run else 'dry-run'}). "
          f"Report: {report_path}")
    if summary["n_total"] and n_unfaithful / summary["n_total"] > 0.5:
        print("WARNING: >50% of stuck states did not survive SOFA restore. "
              "Restore-based recovery curriculum is unreliable for this pool; "
              "prefer live in-situ buckling (--relax_failure_truncations).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
