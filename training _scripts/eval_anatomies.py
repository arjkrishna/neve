#!/usr/bin/env python3
"""Standalone MULTI-ANATOMY evaluation of a trained checkpoint.

WHY THIS EXISTS
---------------
The training loop's built-in eval is single-anatomy. `DualDeviceNav_train.py`
builds the eval intervention as

    DualDeviceNavRCCAVaried(seed=procedural_seed-1, episodes_between_change=10**9)

and `RCCAVariedFromMesh.reset()` only regenerates the vessel tree when
``episode_nr % episodes_between_change == 0`` — passing a per-episode seed
merely re-seeds the RNG, it does NOT regenerate geometry. With the interval
set to 1e9 the tree is therefore generated once and FROZEN for the whole run:
every eval episode, in every eval, of every run so far, was navigated in ONE
anatomy. The 98 eval seeds vary the TARGET (and the device start rotation)
inside that fixed geometry — which is why eval path lengths span 74-284 mm.

Training, by contrast, runs 16 workers each regenerating its own tree every
10 episodes — hundreds of distinct anatomies. So the reported eval numbers
measure "navigate this one held-out anatomy to various depths", NOT
"generalize across anatomies". This script measures the latter.

WHAT IT DOES
------------
Rebuilds the agent exactly as training does (BenchAgentSynchron — the only
builder carrying critic_layernorm / privileged_obs_dim / aux plumbing, so
checkpoint state_dicts load strictly), loads a checkpoint, and evaluates it
with a PER-WORKER anatomy stream (env_eval_factory, mirroring training's
env_train_factory) regenerating every `--change_every` episodes.

ANATOMY ACCOUNTING: reset() regenerates only when
``episode_nr > 0 and episode_nr % change_every == 0`` — episode 0 never
regenerates, so each worker's first episode uses the tree its factory
constructed. Distinct anatomies ~= n_worker + (regenerations), NOT
n_episodes: at change_every=1 you get one per episode; at 2, roughly half.
The report prints the OBSERVED distinct count from the logged
``anatomy=<branch-hash>`` field — trust that, not the arithmetic.

Reports overall success with a Wilson interval, plus a depth-resolved split
(CCA / ICA-mid / siphon by planned-path length) using the same 146/210 mm cuts
as the prior single-anatomy analysis, so old and new numbers are comparable.

PREFLIGHT
---------
`--verify_variation` resets the eval env under a few seeds and hashes the
planned-path geometry, printing whether the anatomy actually changes. Run it
once against this script's env (expect DIFFER) and, if you want to reproduce
the bug, against `--frozen_anatomy` (expect IDENTICAL).

USAGE (inside the training container, same mounts as the launcher)
------------------------------------------------------------------
  python3 /opt/eve_training/training_scripts/eval_anatomies.py \
      --checkpoint /opt/eve_training/results/<run>/checkpoints/checkpoint757854.everl \
      --n_episodes 98 --change_every 2 --n_worker 16 \
      --residual_heuristic --heur_action_obs --privileged_actor \
      --critic_layernorm --relax_failure_truncations --buckle_reward_coef 0.5

The architecture flags MUST match the run that produced the checkpoint or the
strict state_dict load fails loudly (which is the intended behavior).
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/eve_training/training_scripts")


# ----------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval — correct for proportions near 0/1 and small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def section_of(path_len_mm: float, cuts=(146.0, 210.0)) -> str:
    """Depth section by planned-path length to target (same cuts as the
    2026-07 single-anatomy analysis, so numbers stay comparable)."""
    if path_len_mm < cuts[0]:
        return "CCA"
    if path_len_mm < cuts[1]:
        return "ICA-mid"
    return "siphon"


def build_env_kwargs(a):
    kw = {
        "default_target_branch": a.target_branch,
        "relax_failure_truncations": bool(a.relax_failure_truncations),
        "buckle_reward_coef": float(a.buckle_reward_coef),
        # Step budget (eve.truncation.MaxSteps). Training used 600; raising
        # it tests whether max_steps failures are "slow but correct" or
        # genuinely lost. NB: comparisons across different budgets are not
        # like-for-like — always report the budget with the number.
        "n_max_steps": int(a.max_steps),
    }
    if a.residual_heuristic:
        kw["residual_heuristic"] = True
        kw["residual_scale"] = float(a.residual_scale)
        kw["heur_action_obs"] = bool(a.heur_action_obs)
    if a.cath_slack_coef != 0.0:
        kw["cath_slack_coef"] = float(a.cath_slack_coef)
    if a.progress_tip_mode != "frontier":
        kw["progress_tip_mode"] = a.progress_tip_mode
        kw["avg_gw_weight"] = float(a.avg_gw_weight)
    return kw


def make_env(a, seed: int, change_every: int, mode: str):
    from eve_bench.dualdevicenavrccavaried import DualDeviceNavRCCAVaried
    from util.env5 import BenchEnv5

    if getattr(a, "real_patient_anatomy", False):
        # THE REAL PATIENT ANATOMY — RCCA *not* replaced.
        # RCCAVariedFromMesh.__init__ calls _generate() (line 282), which
        # swaps the loaded patient RCCA for a perturbed copy. So even the
        # "frozen" legacy eval tree (episodes_between_change=1e9) was ONE
        # GENERATED VARIANT, never the patient vessel — and its
        # mesh_fingerprint reads g0 because _generation resets to 0 on the
        # reset() re-seed, which is why this hid for so long.
        # perturb_rcca's displacement scales with base_amp_mm * tortuosity,
        # so zeroing both returns the loaded centerline exactly; radii are
        # scaled toward radius_scale, so pin it at 1.0. rva_amp_mm is NOT
        # exposed by the wrapper, so zero it and regenerate once — after
        # which every branch equals the loaded patient geometry.
        interv = DualDeviceNavRCCAVaried(
            seed=seed,
            episodes_between_change=10 ** 9,
            base_amp_mm=0.0,
            tortuosity_mean_sigma=(0.0, 0.0),
            tortuosity_clip=(0.0, 0.0),
            radius_scale_mean_sigma=(1.0, 0.0),
        )
        vt = interv.vessel_tree
        vt.rva_amp_mm = 0.0
        vt._generate()          # rebuild with ALL amplitudes at zero
    else:
        interv = DualDeviceNavRCCAVaried(
            seed=seed, episodes_between_change=change_every
        )
    return BenchEnv5(
        intervention=interv, mode=mode, visualisation=False,
        **build_env_kwargs(a)
    )


def geometry_hash(env) -> str:
    """Hash of the ACTUAL vessel branch geometry = anatomy identity.

    Do NOT use pathfinder.path_points_vessel_cs (the planned path moves with
    the TARGET, so it differs across seeds even in an identical tree), and do
    NOT trust vessel_tree.mesh_fingerprint (it is f"s{seed}g{gen}" and the
    seed is reassigned on every reset, so it changes even when nothing
    regenerates — verified 2026-07-27: under episodes_between_change=1e9 the
    fingerprints read s…g0, s…g0, s…g0 while the branch geometry was one
    constant hash).
    """
    import hashlib

    vt = env.intervention.vessel_tree
    parts = []
    for b in (getattr(vt, "branches", None) or []):
        parts.append(
            np.asarray(b.coordinates, dtype=np.float64).round(4).tobytes()
        )
    if not parts:
        return "NO-BRANCHES"
    return hashlib.md5(b"".join(parts)).hexdigest()[:12]


# ------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="*.everl to evaluate")
    p.add_argument("--n_episodes", type=int, default=98,
                   help="total eval episodes (default 98 = the standard "
                        "EVAL_SEEDS count, so numbers stay comparable)")
    p.add_argument("--change_every", type=int, default=2,
                   help="regenerate each worker's anatomy every N episodes. "
                        "Mirrors training (which uses 10). N=2 with 16 "
                        "workers over 98 episodes gives ~48 distinct trees "
                        "at ~48 scene rebuilds — every rebuild is expensive, "
                        "so this trades anatomy count against wall-clock.")
    p.add_argument("--snapshot_mode", default="centerlines",
                   choices=["centerlines", "mesh", "off"],
                   help="per-episode PNG snapshots (successes AND failures, "
                        "bucketed by outcome) for failure forensics")
    p.add_argument("--seed_base", type=int, default=900000,
                   help="held-out seed band; must not overlap training "
                        "(procedural_seed..+n_worker) or the legacy eval "
                        "anatomy (procedural_seed-1)")
    p.add_argument("--n_worker", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=None,
                   help="where STEP logs land (default: alongside checkpoint)")
    p.add_argument("--verify_variation", action="store_true",
                   help="preflight: prove the anatomy changes per episode")
    p.add_argument("--frozen_anatomy", action="store_true",
                   help="reproduce the LEGACY single-anatomy protocol: one "
                        "frozen tree shared by all workers (no per-worker "
                        "factory, episodes_between_change=1e9). Pair with "
                        "--anatomy_seed 12344 to get the exact tree every "
                        "prior eval in this program used (procedural_seed-1)")
    p.add_argument("--anatomy_seed", type=int, default=None,
                   help="seed of the frozen/master anatomy (default: "
                        "--seed_base). 12344 = the legacy eval tree.")
    p.add_argument("--max_steps", type=int, default=600,
                   help="per-episode step budget (training used 600)")
    p.add_argument("--real_patient_anatomy", action="store_true",
                   help="evaluate on the REAL PATIENT vessel with its "
                        "original RCCA (perturbation amplitudes zeroed, so "
                        "the loaded centerline is kept). NOTE this is NOT "
                        "the same as --frozen_anatomy: the constructor "
                        "always calls _generate(), so a 'frozen' tree is "
                        "one GENERATED variant. Self-check: with zero "
                        "amplitude the geometry is seed-independent, so "
                        "--verify_variation must report IDENTICAL.")
    # --- architecture flags: MUST match the checkpoint's run ---
    p.add_argument("--hidden", nargs="+", type=int, default=[256, 256])
    p.add_argument("--embedder_layers", type=int, default=0)
    p.add_argument("--embedder_nodes", type=int, default=256)
    p.add_argument("--privileged_actor", action="store_true")
    p.add_argument("--critic_layernorm", action="store_true")
    p.add_argument("--aux_coef", type=float, default=0.0)
    p.add_argument("--aux_labels", type=str, default="",
                   help='e.g. "2,3,5,6" — sets the policy aux-head width')
    p.add_argument("--log_std_min", type=float, default=-2.0)
    p.add_argument("--log_std_max", type=float, default=0.0)
    p.add_argument("--algo", default="sac")
    p.add_argument("--stochastic_eval", action="store_true",
                   help="sample tanh(N(mean, policy std)) instead of "
                        "tanh(mean) — evaluates the STOCHASTIC controller "
                        "(dither can break static-friction stalls). "
                        "Default off = deterministic, the headline protocol.")
    # --- env flags: MUST match the checkpoint's run (obs width!) ---
    p.add_argument("--residual_heuristic", action="store_true")
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--heur_action_obs", action="store_true")
    p.add_argument("--relax_failure_truncations", action="store_true")
    p.add_argument("--buckle_reward_coef", type=float, default=0.5)
    p.add_argument("--cath_slack_coef", type=float, default=0.0)
    p.add_argument("--progress_tip_mode", default="frontier",
                   choices=["frontier", "avg"])
    p.add_argument("--avg_gw_weight", type=float, default=0.5)
    p.add_argument("--target_branch", default="Centerline curve - RCCA.mrk")
    a = p.parse_args()

    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(a.checkpoint)),
        "eval_anatomies_" + os.path.basename(a.checkpoint).replace(".everl", ""),
    )
    # M2 — per-invocation subdir. env5's setup_step_logger opens
    # worker_<pid>.log in APPEND mode and analyze() globs *.log, so reusing
    # one dir silently merges a re-run into the previous run's counts.
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(out_dir, "logs", run_tag)
    os.makedirs(log_dir, exist_ok=True)
    # M3 — Synchron.evaluate's only stop condition here is the hard timeout
    # (seeds given, no step/episode limits => both inf). Default is 70 min,
    # on expiry it BREAKS and returns PARTIAL results; ~98 episodes with
    # SOFA scene rebuilds can exceed it.
    os.environ.setdefault("EVE_RL_EVAL_HARD_TIMEOUT_MIN", "600")
    # env5 writes per-episode STEP / EPISODE_OUTCOME lines here; the depth
    # split is parsed from them afterwards.
    os.environ["STEP_LOG_DIR"] = log_dir
    print(f"[eval-anat] STEP_LOG_DIR={log_dir}")

    change_every = (10 ** 9 if (a.frozen_anatomy or a.real_patient_anatomy)
                    else a.change_every)
    seeds = [a.seed_base + i for i in range(a.n_episodes)]

    # Snapshots: env5 reads these env vars at import/step time, and the
    # workers inherit os.environ on spawn — so they MUST be set before the
    # agent is constructed. save_snapshot fires at terminal/truncation and
    # buckets by outcome (success / max_steps / vessel_end), i.e. FAILURES
    # are captured too; filenames embed ep+pid+step+reward so the 16 worker
    # processes cannot collide.
    if a.snapshot_mode != "off":
        snap_dir = os.path.join(out_dir, "snapshots", run_tag)
        os.makedirs(snap_dir, exist_ok=True)
        os.environ["SNAPSHOT_MODE"] = a.snapshot_mode
        os.environ["SNAPSHOT_DIR"] = snap_dir
        os.environ["SNAPSHOT_EVERY"] = "1"     # every eval episode
        print(f"[eval-anat] SNAPSHOT_MODE={a.snapshot_mode} "
              f"SNAPSHOT_DIR={snap_dir} SNAPSHOT_EVERY=1")
    else:
        for k in ("SNAPSHOT_MODE", "SNAPSHOT_DIR", "SNAPSHOT_EVERY"):
            os.environ.pop(k, None)

    # ---------------- preflight: does the anatomy actually vary? ---------
    if a.verify_variation:
        print(f"[eval-anat] preflight (episodes_between_change={change_every}) …")
        env = make_env(a, seed=a.seed_base, change_every=change_every, mode="eval")
        hashes = []
        for i, s in enumerate(seeds[:4]):
            env.reset(seed=s)
            hashes.append(geometry_hash(env))
            print(f"  episode {i} seed={s} geometry={hashes[-1]}")
        env.close()
        uniq = len(set(hashes))
        single_expected = a.real_patient_anatomy or a.frozen_anatomy
        if single_expected:
            # One tree is the POINT here: real-patient (zero perturbation =>
            # seed-independent) or a legacy frozen variant.
            verdict = ("IDENTICAL (correct — single fixed anatomy)"
                       if uniq == 1 else
                       "VARYING (WRONG — this mode must not regenerate)")
            ok = uniq == 1
        else:
            verdict = ("VARYING (correct)" if uniq > 1 else
                       "IDENTICAL (single-anatomy — this is the bug)")
            ok = uniq > 1
        print(f"[eval-anat] distinct geometries: {uniq}/{len(hashes)} — {verdict}")
        if not ok:
            print("[eval-anat] ABORT: anatomy variation is not as expected; "
                  "do not trust results.")
            return 2
        return 0

    # ---------------- build agent exactly as training does ---------------
    from util.agent import BenchAgentSynchron
    from eve.observation.meshinvariant import PrivilegedState

    aux_rel = [int(t) for t in a.aux_labels.split(",") if t.strip()]
    priv_dim = 0 if a.privileged_actor else PrivilegedState.N_DIMS

    anat_seed = a.anatomy_seed if a.anatomy_seed is not None else a.seed_base
    env_train = make_env(a, seed=a.seed_base - 1000, change_every=change_every,
                         mode="train")          # sizing only; never stepped
    env_eval = make_env(a, seed=anat_seed, change_every=change_every,
                        mode="eval")            # master (sizing/config)

    if a.real_patient_anatomy:
        env_eval_factory = None      # one shared, unperturbed patient tree
        print("[eval-anat] REAL PATIENT ANATOMY: original RCCA kept "
              "(base_amp=0, tortuosity=0, radius_scale=1, rva_amp=0); "
              "no factory, no regeneration")
    elif a.frozen_anatomy:
        # LEGACY protocol reproduction: no factory, so every worker
        # deep-copies this one master tree and never regenerates — exactly
        # what DualDeviceNav_train did (and why all prior evals were
        # single-anatomy). Only the target + start rotation vary.
        env_eval_factory = None
        print(f"[eval-anat] FROZEN single anatomy (legacy protocol): "
              f"seed={anat_seed}, change_every={change_every}, no factory")
    else:
        # Per-worker EVAL anatomy streams — the point of the exercise.
        # Without this every worker deep-copies ONE env_eval, so all 16
        # start on the SAME tree and (at change_every=N) their first N
        # episodes share it. Mirrors training's env_train_factory.
        def env_eval_factory(worker_id: int):
            return make_env(a, seed=a.seed_base + 10000 * (worker_id + 1),
                            change_every=change_every, mode="eval")

        print(f"[eval-anat] anatomy regeneration: every {change_every} "
              f"episode(s), per-worker streams (seeds {a.seed_base + 10000}, "
              f"{a.seed_base + 20000}, … x{a.n_worker})")
    print(f"[eval-anat] step budget: {a.max_steps} "
          f"(training/eval standard = 600)")
    print(f"[eval-anat] action selection: "
          f"{'STOCHASTIC (tanh(N(mu,std)) sampled)' if a.stochastic_eval else 'deterministic tanh(mu)'}")
    print(f"[eval-anat] seeds: {seeds[0]}..{seeds[-1]}  (n={len(seeds)})")
    print(f"[eval-anat] privileged_obs_dim={priv_dim} "
          f"critic_layernorm={a.critic_layernorm} aux={aux_rel or None}")

    agent = BenchAgentSynchron(
        a.device, a.device,
        3e-4, 1.0, 1,                       # lr / lr_end_factor / lr_steps
        a.hidden, a.embedder_nodes, a.embedder_layers,
        0.99, 64, 1.0, 10000,               # gamma / batch / rew_scale / buffer
        env_train, env_eval,
        1,                                   # consecutive_action_steps
        a.n_worker,
        stochastic_eval=bool(a.stochastic_eval),   # default False = deterministic
        replay_mode="step",
        algo=a.algo,
        log_std_min=a.log_std_min, log_std_max=a.log_std_max,
        critic_layernorm=a.critic_layernorm,
        privileged_obs_dim=priv_dim,
        aux_coef=a.aux_coef,
        aux_label_rel_indices=aux_rel or None,
        env_eval_factory=env_eval_factory,
    )

    print(f"[eval-anat] loading checkpoint {a.checkpoint}")
    agent.load_checkpoint(a.checkpoint)

    print(f"[eval-anat] evaluating {len(seeds)} episodes on "
          f"{a.n_worker} workers …")
    episodes = agent.evaluate(seeds=seeds) or []
    agent.close()

    # ---- M1: headline = runner.eval()'s own metric, from the Episode
    # objects (runner.py computes quality as infos[-1][quality_info] with
    # quality_info="success"). The STEP-log parse below agrees by
    # construction (EPISODE_OUTCOME reason=success is snapshotted BEFORE
    # env5 rewrites terminated for truncated failures) but the log path
    # cannot see DROPPED SEEDS: Synchron.evaluate clears self._eval_seeds
    # before collecting, so a worker restart re-dispatches evaluate(
    # seeds=None), runs ONE unseeded episode, and silently abandons every
    # seed that worker had left — moving numerator AND denominator.
    k_off = sum(1 for ep in episodes
                if bool((ep.infos[-1] if ep.infos else {}).get("success", False)))
    n_off = len(episodes)
    got = sorted(int(ep.seed) for ep in episodes if getattr(ep, "seed", None) is not None)
    missing = sorted(set(seeds) - set(got))
    unseeded = sum(1 for ep in episodes if getattr(ep, "seed", None) is None)
    lo, hi = wilson(k_off, n_off) if n_off else (float("nan"), float("nan"))
    print("\n" + "=" * 68)
    print(f"OFFICIAL (infos[-1]['success']): {k_off}/{n_off} = "
          f"{100*k_off/max(1,n_off):.1f}%  "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    print("=" * 68)
    if n_off != len(seeds) or missing or unseeded:
        print(f"*** EVAL INTEGRITY FAILURE: requested {len(seeds)} seeds, got "
              f"{n_off} episodes; {len(missing)} never ran "
              f"{missing[:10]}{'…' if len(missing) > 10 else ''}; "
              f"{unseeded} unseeded (worker restart). DO NOT REPORT — re-run. ***")
    with open(os.path.join(out_dir, f"episodes_official_{run_tag}.jsonl"), "w") as fh:
        for ep in episodes:
            info = ep.infos[-1] if ep.infos else {}
            fh.write(json.dumps({
                "seed": getattr(ep, "seed", None),
                "success": bool(info.get("success", False)),
                "reward": float(ep.episode_reward),
                "steps": len(ep.actions),
                "final_branch_short": info.get("final_branch_short"),
                "grader_success": bool(info.get("grader_success", False)),
            }) + "\n")

    # ---------------- depth-resolved report from the STEP logs -----------
    analyze(log_dir, out_dir, n_expected=len(seeds), k_official=k_off)
    return 0


def analyze(log_dir: str, out_dir: str, n_expected: int = None,
            k_official: int = None):
    """Join EPISODE_START / STEP (path_len) / EPISODE_OUTCOME per episode.

    MUST key state by pid: `env_eval` is built in the main process and
    deep-copied to the workers, so every worker inherits pid-1's log handle
    and all 16 of them INTERLEAVE into a single worker_1.log (unlike
    training, where env_train_factory builds each worker's env in-process
    and each gets its own pid-named file). A single sequential state machine
    would splice episodes from different workers together and silently
    fabricate results — verified 2026-07-27, 16 pids in one file with
    consecutive lines alternating between them.
    """
    WT = re.compile(r"wall_time=([0-9.]+)")
    PL = re.compile(r"path_len=([0-9.]+)")
    PID = re.compile(r"pid=([0-9]+)")
    ANA = re.compile(r"anatomy=([0-9a-f]+)")
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        live = {}                      # pid -> in-progress episode state
        with open(path, errors="replace") as fh:
            for line in fh:
                mp_pid = PID.search(line)
                if not mp_pid:
                    continue
                pid = mp_pid.group(1)
                if "EPISODE_START" in line:
                    prev = live.get(pid)
                    if prev and prev["pl"] is not None:
                        rows.append(prev)
                    m = WT.search(line)
                    ma = ANA.search(line)
                    live[pid] = {"t": float(m.group(1)) if m else 0.0,
                                 "pl": None, "succ": False, "steps": 0,
                                 "pid": pid,
                                 "anat": ma.group(1) if ma else None}
                    continue
                st = live.get(pid)
                if st is None:
                    continue
                if " STEP |" in line:
                    st["steps"] += 1
                    if st["pl"] is None:
                        mpl = PL.search(line)
                        if mpl:
                            st["pl"] = float(mpl.group(1))
                    if "term=True" in line and "trunc=False" in line:
                        st["succ"] = True
                elif "EPISODE_OUTCOME" in line:
                    if "reason=success" in line:
                        st["succ"] = True
                    if st["pl"] is not None:
                        rows.append(st)
                    live.pop(pid, None)
        for st in live.values():       # trailing open episodes
            if st["pl"] is not None:
                rows.append(st)

    if not rows:
        print("[eval-anat] WARNING: no episodes parsed from STEP logs; "
              "depth split unavailable (check STEP_LOG_DIR).")
        return

    # M4 — rows are kept only when path_len parsed; env5 prints path_len=?
    # when get_projection() raises, which would drop an episode from BOTH
    # numerator and denominator with no trace. Audit it loudly.
    starts = 0
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        with open(path, errors="replace") as fh:
            starts += sum(1 for line in fh if "EPISODE_START" in line)
    dropped = starts - len(rows)
    print(f"[eval-anat] parse audit: EPISODE_START={starts} rows={len(rows)} "
          f"dropped(no path_len)={dropped}"
          + (f" expected={n_expected}" if n_expected else ""))
    if dropped or (n_expected and starts != n_expected):
        print("*** PARSER DROPPED EPISODES — the depth split below is "
              "biased; the OFFICIAL number above is authoritative ***")
    if k_official is not None:
        k_log = sum(1 for r in rows if r["succ"])
        if k_log != k_official:
            print(f"*** MISMATCH: log-derived successes={k_log} vs "
                  f"official={k_official} — investigate before reporting ***")

    n = len(rows)
    k = sum(1 for r in rows if r["succ"])
    lo, hi = wilson(k, n)
    print("\n" + "=" * 68)
    print(f"MULTI-ANATOMY RESULT: {k}/{n} = {100*k/n:.1f}%  "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    print("=" * 68)

    # --- anatomy diversity: VERIFIED, not assumed -----------------------
    anats = [r["anat"] for r in rows if r.get("anat")]
    if anats:
        uniq = sorted(set(anats))
        print(f"anatomies: {len(uniq)} distinct over {len(anats)} episodes "
              f"(episodes/anatomy: min {min(anats.count(x) for x in uniq)}, "
              f"max {max(anats.count(x) for x in uniq)})")
        if len(uniq) == 1:
            print("  *** ONE ANATOMY — this is the single-anatomy bug; "
                  "the number above is NOT a generalization measurement ***")
        else:
            per = [(x, anats.count(x),
                    sum(1 for r in rows if r.get("anat") == x and r["succ"]))
                   for x in uniq]
            rates = [100.0 * s / c for _, c, s in per if c >= 3]
            if rates:
                print(f"  per-anatomy success spread (n>=3): "
                      f"min {min(rates):.0f}%  median {np.median(rates):.0f}%  "
                      f"max {max(rates):.0f}%  "
                      f"(across-anatomy variance is the generalization story)")
    else:
        print("anatomies: NOT LOGGED (env5 lacks the anatomy= field — "
              "mount the RL_IMPROV_18 env5.py to verify diversity)")
    print(f"{'section':10s} {'succ':>10s} {'rate':>8s} {'95% CI':>16s} "
          f"{'path_len mm':>14s}")
    for s in ("CCA", "ICA-mid", "siphon"):
        d = [r for r in rows if section_of(r["pl"]) == s]
        if not d:
            continue
        ks = sum(1 for r in d if r["succ"])
        l2, h2 = wilson(ks, len(d))
        print(f"{s:10s} {ks:4d}/{len(d):-4d} {100*ks/len(d):7.1f}% "
              f"{100*l2:6.1f}-{100*h2:-5.1f}% "
              f"{min(r['pl'] for r in d):6.0f}-{max(r['pl'] for r in d):-6.0f}")
    succ_steps = [r["steps"] for r in rows if r["succ"]]
    if succ_steps:
        print(f"\nsteps-to-success: median {int(np.median(succ_steps))} "
              f"p10 {int(np.percentile(succ_steps,10))} "
              f"p90 {int(np.percentile(succ_steps,90))}")
    print(f"path_len distribution: min {min(r['pl'] for r in rows):.0f} "
          f"median {np.median([r['pl'] for r in rows]):.0f} "
          f"max {max(r['pl'] for r in rows):.0f} mm")

    csv_path = os.path.join(out_dir, "episodes.csv")
    with open(csv_path, "w") as fh:
        fh.write("wall_time,path_len_mm,section,steps,success\n")
        for r in sorted(rows, key=lambda r: r["t"]):
            fh.write(f"{r['t']:.3f},{r['pl']:.1f},{section_of(r['pl'])},"
                     f"{r['steps']},{int(r['succ'])}\n")
    print(f"\nper-episode CSV: {csv_path}")


if __name__ == "__main__":
    # M6 — match DualDeviceNav_train.py:1409. Under the Linux default
    # (fork) the parent builds every worker env, so setup_step_logger opens
    # worker_<parentpid>.log in the parent and all 16 children inherit that
    # ONE file descriptor (why both earlier runs produced a single
    # worker_1.log with interleaved pids) — and fork+CUDA is unsafe besides.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
