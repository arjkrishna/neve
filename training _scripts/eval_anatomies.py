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

TOPBRAIN COHORT (--topbrain, default off)
-----------------------------------------
`--topbrain` swaps the procedural env for `DualDeviceNavTopBrain`: the shipped
host tree carrying a REAL TopBrain-2025 patient right-ICA siphon grafted on at
130 mm, one per folder under `--topbrain_dir`. Devices, insertion pose and
target-sampler configuration are the same as `DualDeviceNavRCCAVaried` — which
is what makes the number comparable with every host result in the program, and
which `--verify_variation` DIFFS AND PRINTS rather than assuming.

  --topbrain_exclude        default drops topcow_mr_013/014/015 (the 2026-08
                            audit cut), leaving the retained 22
  --topbrain_only           held-out arm; authoritative (exclude not applied)
  --topbrain_trim_stations  drop the last N stations of the target branch from
                            the TARGET POOL only — never from the geometry, the
                            mesh or the pathfinder. N=0 is an exact no-op.

Attribution: env5 stamps `mesh_fp=<name-fingerprint>` on every EPISODE_START,
so the report prints a per-anatomy table, cross-checks name against geometry
hash 1:1, and writes `anatomy_success.csv`; `episodes.csv` gains anatomy /
geometry_hash / seed columns in this mode only.

Every one of these is inert without `--topbrain`; with none of them the script
behaves exactly as before.
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


def _install_pinned_surface_patch():
    """Make RCCAVariedFromMesh honour an instance attribute `_pinned_mesh`.

    Workers are SPAWNED, so anything reachable from the env must pickle. A
    closure bound to the instance, or a class defined inside a function, both
    fail (`Can't pickle local object`). Patching the CLASS method leaves the
    instance carrying only a plain string, which pickles fine; each process
    that builds an env calls this, so the patch exists wherever it is needed.

    Why it is needed at all: _generate() nulls _mesh_path, and reset() calls
    _generate() (episode_nr % episodes_between_change == 0 is true at episode
    0), so without this the re-meshed surface would come back on the first
    reset even after being overridden at construction.
    """
    from eve.intervention.vesseltree.rccavariedfrommesh import RCCAVariedFromMesh

    if getattr(RCCAVariedFromMesh, "_pinned_surface_patch", False):
        return
    _orig_generate = RCCAVariedFromMesh._generate

    def _generate(self):
        # `_require_passable`: reject-and-regenerate until the generated surface
        # actually admits the guidewire. The procedural mesher yields anatomies
        # that are geometrically impassable (measured: 4 of 6 sampled have
        # stations where the 0.36 mm device cannot fit; median clearance 1.1-1.3
        # mm vs the original segmentation's 2.11). Evaluating on those measures
        # the mesher, not the policy. See MESH_GENERATOR_FIX_PLAN.md.
        need = getattr(self, "_require_passable", False)
        tries = int(getattr(self, "_passable_max_tries", 15))
        for k in range(tries if need else 1):
            _orig_generate(self)
            pinned = getattr(self, "_pinned_mesh", None)
            if pinned:
                self._mesh_path = pinned
            if not need:
                return
            if _tree_is_passable(
                    self, min_median_mm=float(getattr(self, "_passable_min_median", 0.0))):
                self._passable_tries = k + 1
                return
        self._passable_tries = -1        # gave up; surface may be impassable

    RCCAVariedFromMesh._generate = _generate
    RCCAVariedFromMesh._pinned_surface_patch = True


def _install_target_pool_trim_patch():
    """Make CenterlineRandom honour an instance attribute `_trim_last_stations`.

    Drops the last N centerline stations of each sampled branch from the TARGET
    POOL. Nothing else: the branch coordinates, the mesh, the pathfinder and
    `at_tree_end` all keep the full station list, so only which points may be
    CHOSEN as a target changes.

    Why it is wanted: the terminal-station deficit is universal — the last one
    or two stations of a real centerline sit at/through the capped end of the
    surface, so a target there can be geometrically unreachable. Measured in
    the retained-22 cohort at wire radius 0.18 mm: mr_027 blocked at the
    terminal station only, and the HOST itself blocked at 3 / outside at 2 over
    s = 223.9-225.9 mm. Trimming makes the cohort and the host comparable on
    reachable targets instead of on how each one's centerline was clipped.

    `_arclength_from_start_mask` is the right seam: it is called ONLY from
    `_init_centerline_point_cloud`, once per branch with that branch's own
    ordered coordinates, and both the pooled and the per-branch target lists
    are built through it — so a False in the mask removes the station from the
    target pool and from nowhere else. `branches=[RCCA]` in both this env and
    DualDeviceNavRCCAVaried, so the only branch it ever sees is the navigated
    one (asserted at build time in make_env).

    Workers are SPAWNED, so this patches the CLASS (as
    `_install_pinned_surface_patch` does); the instance carries only a plain
    int, which pickles. N = 0 / attribute absent is an exact no-op: the
    original mask object is returned unmodified, not a copy.
    """
    from eve.intervention.target.centerlinerandom import CenterlineRandom

    if getattr(CenterlineRandom, "_target_pool_trim_patch", False):
        return
    _orig_mask = CenterlineRandom._arclength_from_start_mask

    def _arclength_from_start_mask(self, points):
        mask = _orig_mask(self, points)
        n_trim = int(getattr(self, "_trim_last_stations", 0) or 0)
        if n_trim <= 0 or len(mask) == 0:
            return mask
        mask = np.asarray(mask, dtype=bool).copy()
        mask[-min(n_trim, len(mask)):] = False
        return mask

    CenterlineRandom._arclength_from_start_mask = _arclength_from_start_mask
    CenterlineRandom._target_pool_trim_patch = True


# TopBrain cohort: the 3 anatomies cut by the 2026-08 audit. Name-based, so
# the exclusion survives any re-baking of the meshes.
_TOPBRAIN_DEFAULT_EXCLUDE = ["topcow_mr_013", "topcow_mr_014", "topcow_mr_015"]
_TOPBRAIN_DEFAULT_DIR = "/opt/eve_training/results_topbrain/anatomies"


def _topbrain_parity_report(interv) -> bool:
    """Prove the TopBrain env is task-identical to DualDeviceNavRCCAVaried.

    The whole point of running the cohort is COMPARABILITY with every host
    number already in the program, so "the docstring says they match" is not
    good enough — an insertion or target-pool difference would turn a
    difficulty probe into a different-task measurement (exactly the failure
    mode `_retarget_inside_branch` documents). This constructs the procedural
    env and diffs the three things that define the task: the device set, the
    insertion pose, and the target sampler's configuration.

    Returns True when everything matches. Preflight only (it builds a
    procedural tree, which costs a marching-cubes pass).
    """
    from eve_bench.dualdevicenavrccavaried import DualDeviceNavRCCAVaried

    ref = DualDeviceNavRCCAVaried(seed=12344, episodes_between_change=10 ** 9)
    ok = True

    def _cmp(label, got, exp):
        nonlocal ok
        same = got == exp
        ok = ok and same
        print(f"  [{'ok ' if same else 'DIFF'}] {label}: {got!r}"
              + ("" if same else f"   != RCCAVaried {exp!r}"))

    dev_attrs = ("name", "length", "velocity_limit", "tip_outer_diameter",
                 "straight_outer_diameter", "tip_inner_diameter",
                 "straight_inner_diameter", "mass_density_tip",
                 "mass_density_straight", "young_modulus_tip",
                 "young_modulus_straight", "beams_per_mm_straight",
                 "visu_edges_per_mm")
    print("[eval-anat] PARITY vs DualDeviceNavRCCAVaried — devices")
    _cmp("n_devices", len(interv.devices), len(ref.devices))
    for d_got, d_exp in zip(interv.devices, ref.devices):
        _cmp(f"device {getattr(d_exp,'name','?')} class",
             type(d_got).__name__, type(d_exp).__name__)
        for at in dev_attrs:
            _cmp(f"device {getattr(d_exp,'name','?')}.{at}",
                 getattr(d_got, at, None), getattr(d_exp, at, None))

    print("[eval-anat] PARITY — insertion (the wire's start pose)")
    p_got = np.asarray(interv.vessel_tree.insertion.position, dtype=float)
    p_exp = np.asarray(ref.vessel_tree.insertion.position, dtype=float)
    d_got = np.asarray(interv.vessel_tree.insertion.direction, dtype=float)
    d_exp = np.asarray(ref.vessel_tree.insertion.direction, dtype=float)
    dp = float(np.max(np.abs(p_got - p_exp)))
    dd = float(np.max(np.abs(d_got - d_exp)))
    print(f"  position  topbrain={np.round(p_got,6).tolist()} "
          f"rccavaried={np.round(p_exp,6).tolist()}  max|diff|={dp:.9f} mm")
    print(f"  direction topbrain={np.round(d_got,6).tolist()} "
          f"rccavaried={np.round(d_exp,6).tolist()}  max|diff|={dd:.9f}")
    if dp > 1e-9 or dd > 1e-9:
        ok = False
        print("  [DIFF] insertion differs — the wire does NOT start where "
              "every host number started")
    else:
        print("  [ok ] insertion byte-identical")

    print("[eval-anat] PARITY — target sampler")
    t_got, t_exp = interv.target, ref.target
    _cmp("target class", type(t_got).__name__, type(t_exp).__name__)
    for at in ("branches", "threshold", "min_arclength_from_start",
               "min_distance_between_possible_targets"):
        _cmp(f"target.{at}", getattr(t_got, at, None), getattr(t_exp, at, None))

    print("[eval-anat] PARITY — sim / view / env")
    _cmp("simulation class", type(interv.simulation).__name__,
         type(ref.simulation).__name__)
    _cmp("simulation.friction", getattr(interv.simulation, "friction", None),
         getattr(ref.simulation, "friction", None))
    _cmp("fluoroscopy class", type(interv.fluoroscopy).__name__,
         type(ref.fluoroscopy).__name__)
    _cmp("fluoroscopy.image_rot_zx",
         list(interv.fluoroscopy.image_rot_zx), list(ref.fluoroscopy.image_rot_zx))
    _cmp("fluoroscopy.image_frequency", interv.fluoroscopy.image_frequency,
         ref.fluoroscopy.image_frequency)
    _cmp("normalize_action", interv.normalize_action, ref.normalize_action)
    _cmp("stop_device_at_tree_end", interv.stop_device_at_tree_end,
         ref.stop_device_at_tree_end)

    print(f"[eval-anat] PARITY VERDICT: "
          f"{'MATCH — comparable with every host result' if ok else 'MISMATCH — DO NOT COMPARE'}")
    ref.close()
    return ok


_NAV_BRANCH_NAMES = {
    "RCCA": "Centerline curve - RCCA.mrk",
    "LCCA": "Centerline curve - LCCA.mrk",
}


def _retarget_inside_branch(interv, tag, insert_idx, min_arclength_mm):
    """Insert the wire INSIDE `tag`'s branch and repoint the target sampler at it.

    Used for the different-vessel transfer experiment: an RCCA-trained policy is
    evaluated on the LCCA of the same patient. The wire starts already inside the
    carotid rather than in the (11) bridge, because no bridge branch feeds the
    LCCA — its point[0] IS the aortic-arch junction (r = 14.7 mm), shared
    bit-identically with (11)[0].

    Three things must all be set or the run silently measures the wrong vessel:
      * `vt._insertion` — the PRIVATE one. `_generate()` re-asserts
        `self.insertion = self._insertion`, so assigning the public alias alone
        is reverted on the next reset.
      * `tgt.branches` AND `tgt._branches_initialized = None` — CenterlineRandom
        rebuilds its pool only when `_branches_initialized != vessel_tree.branches`.
        The pinned tree's branch tuple never changes again, so without the reset
        the RCCA pool built at construction persists and the run navigates to
        RCCA targets while being logged as LCCA.
      * `min_arclength_from_start` — measured from the branch's OWN point[0],
        which for the LCCA is the arch junction. Left at the inherited 40 mm it
        admits targets ~22 mm ahead of the insertion, and nothing anywhere
        compares a target against the insertion, so a target BEHIND it yields a
        well-formed reversed path into the arch.

    Pickling-safe: module level, mutates only plain attributes. Workers are
    spawned, so closures and locally-defined classes cannot be used here.
    Call AFTER `vt._generate()` and BEFORE `BenchEnv5` is constructed.
    """
    import numpy as np
    from eve.intervention.vesseltree.vesseltree import Insertion

    name = _NAV_BRANCH_NAMES[tag]
    vt = interv.vessel_tree
    br = next(b for b in vt.branches if str(b.name) == name)
    c = np.asarray(br.coordinates, dtype=np.float64)
    cum = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(c, axis=0), axis=1))))

    k = int(min(max(1, int(insert_idx)), len(c) - 2))
    d = c[k + 1] - c[k]
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    ins = Insertion(c[k], d)
    vt._insertion = ins
    vt.insertion = ins

    tgt = interv.target
    tgt.branches = [name]
    tgt.min_arclength_from_start = float(min_arclength_mm)
    tgt._branches_initialized = None

    keep = cum >= float(min_arclength_mm)
    radii = np.asarray(getattr(br, "radii", np.full(len(c), np.nan)), dtype=float)
    s = float(cum[k])
    return (s, float(radii[k]), int(keep.sum()),
            float(cum[keep].min() - s) if keep.any() else float("nan"),
            float(cum[keep].max() - s) if keep.any() else float("nan"))


def _tree_is_passable(vt, wire_radius_mm: float = 0.18, per_tri: int = 12,
                      min_median_mm: float = 0.0) -> bool:
    """Gate on the navigated branch's clearance profile.

    Two conditions:
      * min clearance >= wire radius  -> the device geometrically fits at all
      * median clearance >= min_median_mm -> the vessel is not systematically
        narrower than the real anatomy. The REAL patient surface measures
        median 2.14 mm / p05 1.20; the generator at radius_scale=1.0 gives
        median 1.26 / p05 0.46, i.e. it erodes ~37% of radius. Passing the
        first test alone would still leave a vessel far tighter than reality.

    Clearance is distance to the nearest point ON THE TRIANGLE SURFACE. Nearest
    -VERTEX distance is NOT usable here: on a ~6 mm-triangle mesh the vertices
    sit outside the facets and overstate clearance enough to call an impassable
    mesh passable (it reported 1% erosion where the true figure is ~45%).
    """
    import numpy as np
    import pyvista as pv
    from scipy.spatial import cKDTree

    try:
        m = pv.read(vt.mesh_path).triangulate().clean()
        f = m.faces.reshape(-1, 4)[:, 1:]
        v = m.points
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        rg = np.random.default_rng(0)
        u = rg.random((per_tri, len(f), 1))
        w = rg.random((per_tri, len(f), 1))
        k = u + w > 1
        u = np.where(k, 1 - u, u)
        w = np.where(k, 1 - w, w)
        surf = np.vstack([v, (a + u * (b - a) + w * (c - a)).reshape(-1, 3)])
        br = next(x for x in vt.branches if "RCCA" in str(x.name).upper())
        d, _ = cKDTree(surf).query(np.asarray(br.coordinates, dtype=float))
        return bool(d.min() >= wire_radius_mm
                    and np.median(d) >= min_median_mm)
    except Exception:
        return True      # never hard-fail an eval on the gate


def make_env(a, seed: int, change_every: int, mode: str):
    from eve_bench.dualdevicenavrccavaried import DualDeviceNavRCCAVaried
    from util.env5 import BenchEnv5

    if getattr(a, "topbrain", False):
        # THE TOPBRAIN COHORT — real patient siphons instead of procedural ones.
        #
        # Each anatomy is the shipped HOST tree (arch, trunk, cervical vessels)
        # with one real TopBrain-2025 patient right-ICA siphon grafted on at 130
        # mm arclength. Everything proximal to the graft — including branch (11),
        # which carries the insertion — is shared bit-identically across the set
        # and with the host, so the wire starts in the same place, the devices
        # are the same, and the target sampler is configured the same. That is
        # what makes an H0 number here comparable with every host number in the
        # program; run `--verify_variation` to have that PROVEN rather than
        # assumed (it diffs devices / insertion / target against
        # DualDeviceNavRCCAVaried and prints a verdict).
        #
        # Unlike RCCAVariedFromMesh there is nothing to generate: the collision
        # meshes are baked to disk, so no _pinned_mesh / _require_passable /
        # radius_scale machinery applies here (main() rejects those combinations).
        from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain

        only = list(a.topbrain_only) if getattr(a, "topbrain_only", None) else None
        # `only` is authoritative when given: find_anatomies applies only-then-
        # exclude, so silently intersecting it with the default 3-name cut would
        # turn "--topbrain_only topcow_mr_014" into an empty-set crash whose
        # cause is invisible in the log.
        exclude = None if only else list(getattr(a, "topbrain_exclude", None) or [])
        interv = DualDeviceNavTopBrain(
            anatomy_dir=a.topbrain_dir,
            seed=seed,
            episodes_between_change=change_every,
            only=only,
            exclude=exclude,
            target_min_arclength_mm=float(a.target_min_arclength_mm),
        )
        vt = interv.vessel_tree
        # ATTRIBUTION, printed by every process that builds an env: the roster
        # this worker can draw from, and the anatomy it opens on. env5 then
        # stamps `mesh_fp=<name-fingerprint>` on every EPISODE_START, and
        # analyze() turns that into the per-anatomy table. A prior experiment in
        # this program was nearly reported backwards because the arms could not
        # be told apart in the result files; this makes it checkable by eye.
        print(f"[eval-anat] TOPBRAIN cohort from {a.topbrain_dir}: "
              f"{vt.anatomy_count} anatomies "
              f"{'(only=' + ','.join(only) + ')' if only else '(exclude=' + ','.join(exclude or []) + ')'}")
        print(f"[eval-anat] TOPBRAIN roster: {','.join(vt.anatomy_names)}")
        print(f"[eval-anat] TOPBRAIN seed={seed} episodes_between_change="
              f"{change_every} opening_anatomy={vt.current_anatomy} "
              f"fingerprint={vt.mesh_fingerprint}")

        n_trim = int(getattr(a, "topbrain_trim_stations", 0) or 0)
        if n_trim > 0:
            # Guard the assumption the patch relies on: the mask hook fires per
            # branch and cannot see WHICH branch, so it is only equivalent to
            # "trim the navigated branch" while the sampler draws from exactly
            # that one branch.
            assert list(interv.target.branches or []) == [a.target_branch], (
                f"--topbrain_trim_stations assumes the target sampler draws "
                f"from exactly [{a.target_branch}], got "
                f"{list(interv.target.branches or [])}")
            _install_target_pool_trim_patch()
            interv.target._trim_last_stations = n_trim
            interv.target._branches_initialized = None   # force a pool rebuild
            rcca = next(b for b in vt.branches if str(b.name) == a.target_branch)
            print(f"[eval-anat] TOPBRAIN target-pool trim: dropping the last "
                  f"{n_trim} station(s) of {a.target_branch} from the TARGET "
                  f"POOL only (branch has {len(rcca.coordinates)} stations; "
                  f"geometry, mesh and pathfinder untouched)")
    elif getattr(a, "real_patient_anatomy", False):
        # THE REAL PATIENT ANATOMY — the ORIGINAL segmented surface.
        #
        # 2026-07-29 FIX. The previous implementation built
        # DualDeviceNavRCCAVaried with every perturbation amplitude zeroed.
        # That reproduces the patient CENTERLINES to zero floating-point
        # error (verified: max|diff| = 0.000000 mm) — which is exactly why
        # it passed every check — but RCCAVariedFromMesh ALWAYS re-meshes
        # the tree from those centerlines (voxel -> marching-cubes, see its
        # docstring L10; mesh_path -> generate_temp_mesh). The wire collides
        # with the SURFACE, not the centerline, and the regenerated surface
        # is NOT the patient's:
        #     original vessel_architecture_collision.obj : 3,584 cells,
        #         median clearance 2.11 mm, 0/235 stations blocked (PASSABLE)
        #     zeroed-amplitude regeneration              : 3,721 cells,
        #         median clearance 1.23 mm, 2/235 blocked, first block at
        #         raw arclength 120.4 mm -> proj_s ~154.0 mm
        # and three different controllers (v1b, v1bp, and the parameterless
        # heuristic H0) were measured arresting at proj_s 153.4 mm — 0.6 mm
        # from the geometric prediction. So the old "real patient" number
        # measured a reconstruction that is impassable at the mid-ICA.
        #
        # We do NOT simply construct DualDeviceNav: it also changes the wire
        # INSERTION POINT (femoral entry, threading the whole arch) and the
        # target sampler (4 branches, no minimum arclength). A smoke test of
        # that swap gave path_len 594-752 mm against ~103-156 mm for every
        # previous real-patient run — a different task, not a different mesh.
        #
        # Instead: keep DualDeviceNavRCCAVaried (correct insertion at the (11)
        # bridge, RCCA-only targets, same devices/frames) and swap ONLY the
        # collision SURFACE. DualDeviceNav's FromMesh writes the original .obj
        # already rotated into the branch frame, and the zeroed-amplitude
        # centerlines are identical to it (max|diff| = 0.000000 mm), so that
        # transformed mesh is exactly the right surface for these centerlines.
        # _generate() nulls _mesh_path (and reset() calls it), so wrap it to
        # re-pin the original surface every time.
        from eve_bench.dualdevicenav import DualDeviceNav

        _orig_mesh = DualDeviceNav().vessel_tree.mesh_path

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

        _install_pinned_surface_patch()
        vt._pinned_mesh = _orig_mesh
        vt._generate()
        print(f"[eval-anat] REAL PATIENT: collision surface pinned to the "
              f"ORIGINAL segmented mesh ({_orig_mesh}); centerlines, insertion, "
              f"targets, devices unchanged from every prior run")

        if getattr(a, "insert_inside_branch", "none") != "none":
            s, r, n_t, pl_min, pl_max = _retarget_inside_branch(
                interv, a.insert_inside_branch, int(a.insert_point_idx),
                float(a.target_min_arclength_mm))
            print(f"[eval-anat] NAV BRANCH={a.insert_inside_branch} "
                  f"insert idx={a.insert_point_idx} s={s:.2f}mm r={r:.2f}mm "
                  f"min_arc={a.target_min_arclength_mm}mm targets={n_t} "
                  f"path_len={pl_min:.1f}..{pl_max:.1f}mm")
            # Structural guard against the silent reversed-path failure: nothing
            # else in the stack compares a sampled target against the insertion,
            # and a target behind it yields a well-formed path into the arch.
            assert n_t > 0, "empty target pool — min_arclength exceeds the branch"
            assert pl_min > 10.0, (
                f"targets at or behind the insertion (pl_min={pl_min:.1f} mm); "
                f"raise --target_min_arclength_mm")
            print(f"[eval-anat] GUARD ok: {n_t} targets, all ahead of the tip")
    else:
        rs = float(getattr(a, "radius_scale", 1.0))
        interv = DualDeviceNavRCCAVaried(
            seed=seed, episodes_between_change=change_every,
            radius_scale_mean_sigma=(rs, 0.07 if rs == 1.0 else 0.05),
        )
        if getattr(a, "require_passable", False):
            _install_pinned_surface_patch()
            vt = interv.vessel_tree
            vt._require_passable = True
            vt._passable_max_tries = int(a.passable_max_tries)
            vt._passable_min_median = float(a.passable_min_median_mm)
            vt._generate()      # re-draw now under the gate
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
    # --- TopBrain cohort (DEFAULT OFF; none of these change anything unless
    #     --topbrain is given) ---
    p.add_argument("--topbrain", action="store_true",
                   help="evaluate on the TOPBRAIN COHORT: the shipped host "
                        "tree with a REAL TopBrain-2025 patient right-ICA "
                        "siphon grafted on at 130 mm, one anatomy per "
                        "folder under --topbrain_dir, instead of the "
                        "procedural RCCAVariedFromMesh env. Devices, "
                        "insertion and target semantics are the same as "
                        "DualDeviceNavRCCAVaried — run --verify_variation to "
                        "have that diffed and printed rather than assumed.")
    p.add_argument("--topbrain_dir", default=_TOPBRAIN_DEFAULT_DIR,
                   help="directory of <anatomy>/ folders, each holding "
                        "vessel_architecture_collision.obj + Centrelines_comb/")
    p.add_argument("--topbrain_exclude", nargs="+",
                   default=list(_TOPBRAIN_DEFAULT_EXCLUDE),
                   help="anatomy names to drop (default: the 3 cut by the "
                        "2026-08 audit, leaving the retained 22). Ignored "
                        "when --topbrain_only is given.")
    p.add_argument("--topbrain_only", nargs="+", default=None,
                   help="restrict to these anatomy names — the held-out "
                        "evaluation arm (train with --topbrain_exclude on the "
                        "same names). Authoritative: --topbrain_exclude is "
                        "not also applied.")
    p.add_argument("--topbrain_trim_stations", type=int, default=0,
                   help="drop the last N centerline stations of the target "
                        "branch from the TARGET POOL only — never from the "
                        "geometry, the mesh or the pathfinder. The terminal "
                        "station deficit is universal (measured: mr_027 "
                        "blocked at its last station, and the HOST blocked "
                        "at 3 / outside at 2 over s=223.9-225.9 mm), so N=1 "
                        "or 2 compares reachable targets instead of "
                        "centerline clipping. N=0 (default) is an exact "
                        "no-op.")
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
    p.add_argument("--insert_inside_branch", choices=["none", "RCCA", "LCCA"],
                   default="none",
                   help="DIFFERENT-VESSEL TRANSFER EXPERIMENT. Insert the wire "
                        "INSIDE the named branch instead of the (11) bridge, and "
                        "sample targets on that branch. 'none' (default) leaves "
                        "the (11)->RCCA behaviour byte-identical. Requires "
                        "--real_patient_anatomy; incompatible with "
                        "--require_passable.")
    p.add_argument("--insert_point_idx", type=int, default=2,
                   help="Centerline index within --insert_inside_branch. Index 0 "
                        "must be avoided: LCCA[0] and (11)[0] are the SAME point "
                        "and both resolve to the arch branch in a 3-way distance "
                        "tie, which would plan the path from the aorta. Ignored "
                        "when --insert_inside_branch none.")
    p.add_argument("--require_passable", action="store_true",
                   help="reject-and-regenerate each anatomy until the guidewire "
                        "geometrically fits everywhere along the navigated branch. "
                        "Without this, ~2/3 of generated anatomies are impassable "
                        "and the eval measures the mesher, not the policy.")
    p.add_argument("--passable_max_tries", type=int, default=15)
    p.add_argument("--passable_min_median_mm", type=float, default=2.00,
                   help="reject anatomies whose MEDIAN clearance is below this. "
                        "The real patient surface measures 2.14 mm; the raw "
                        "generator gives 1.26 (it erodes ~37%% of radius), so "
                        "fitting the wire is not sufficient — the vessel must "
                        "also not be systematically tighter than reality.")
    p.add_argument("--radius_scale", type=float, default=1.0,
                   help="compensates the mesher's erosion. Measured: 1.6 "
                        "reproduces the real patient's clearance (median 2.14 "
                        "vs 2.14, p05 1.13 vs 1.20 — i.e. equal on average and "
                        "marginally tighter in the narrow tail).")
    p.add_argument("--target_min_arclength_mm", type=float, default=40.0,
                   help="matches DualDeviceNavRCCAVaried's default so the "
                        "real-patient run differs from the generated runs "
                        "ONLY in the collision surface")
    a = p.parse_args()

    if a.topbrain:
        # Every one of these silently means "not the cohort": the TopBrain
        # meshes are BAKED, so anything that regenerates, re-pins or rescales a
        # procedural surface would either be ignored (and the run mislabelled)
        # or applied to the wrong class. Refuse instead of guessing.
        for flag, val, why in (
            ("--real_patient_anatomy", a.real_patient_anatomy,
             "both select a fixed anatomy; --topbrain already IS real patient "
             "geometry (25 of them), so combining the two is ambiguous"),
            ("--frozen_anatomy", a.frozen_anatomy,
             "the cohort's whole purpose is to vary anatomy; pin a single one "
             "with --topbrain_only <name> instead"),
            ("--require_passable", a.require_passable,
             "the passability gate reject-and-regenerates a PROCEDURAL surface; "
             "baked meshes cannot be redrawn (and the retained 22 were already "
             "screened by exact signed distance)"),
        ):
            if val:
                p.error(f"--topbrain is incompatible with {flag}: {why}")
        if a.insert_inside_branch != "none":
            p.error("--topbrain is incompatible with --insert_inside_branch: "
                    "that path requires --real_patient_anatomy and rewrites the "
                    "insertion, which would break comparability with the host")
        if float(a.radius_scale) != 1.0:
            p.error("--topbrain is incompatible with --radius_scale: it "
                    "compensates the procedural mesher's erosion, and the "
                    "baked cohort meshes come from the real segmentations")
    elif a.topbrain_trim_stations:
        p.error("--topbrain_trim_stations requires --topbrain")

    if a.insert_inside_branch != "none":
        # These two guards catch failures that otherwise produce a run which
        # looks perfect and measures the wrong thing.
        if not getattr(a, "real_patient_anatomy", False):
            p.error("--insert_inside_branch requires --real_patient_anatomy: "
                    "RCCAVariedFromMesh._generate perturbs only the RCCA and RVA, "
                    "so every 'distinct' generated anatomy would share one "
                    "identical LCCA.")
        if getattr(a, "require_passable", False):
            p.error("--insert_inside_branch is incompatible with "
                    "--require_passable: _tree_is_passable hardcodes "
                    "'RCCA' in the branch name behind a bare except, so it would "
                    "gate on the wrong vessel without raising.")
        forced = _NAV_BRANCH_NAMES[a.insert_inside_branch]
        if a.target_branch != forced:
            print(f"[eval-anat] forcing --target_branch {forced} to match "
                  f"--insert_inside_branch {a.insert_inside_branch}")
            a.target_branch = forced
        # Forcing target_branch BEFORE build_env_kwargs reads it keeps env5's
        # _target_branch_short in sync, which is what keeps the privileged
        # "in target daughter" bit (obs 14) from becoming "wrong daughter"
        # (obs 15) on every step of every episode. Under --privileged_actor the
        # ACTOR consumes that tail, so a mismatch corrupts the policy input.

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
        parity_ok = True
        if a.topbrain:
            # Comparability is the entire premise of the probe, so verify it
            # here rather than trusting TOPBRAIN_PIPELINE.md.
            parity_ok = _topbrain_parity_report(env.intervention)
        hashes = []
        names = []
        for i, s in enumerate(seeds[:4]):
            env.reset(seed=s)
            hashes.append(geometry_hash(env))
            fp = getattr(env.intervention.vessel_tree, "mesh_fingerprint", "?")
            names.append(str(fp))
            print(f"  episode {i} seed={s} geometry={hashes[-1]} "
                  f"mesh_fp={fp}")
        env.close()
        uniq = len(set(hashes))
        if a.topbrain:
            # The name-based fingerprint IS the anatomy identity here, so a
            # geometry hash and a fingerprint must partition the episodes the
            # same way. If they disagree, attribution in the result files is
            # not trustworthy and no per-anatomy number may be reported.
            pairs = sorted(set(zip(hashes, names)))
            print(f"[eval-anat] TOPBRAIN attribution: "
                  f"{len(set(hashes))} distinct geometries / "
                  f"{len(set(names))} distinct fingerprints; pairs={pairs}")
            if len(set(hashes)) != len(set(names)) or len(pairs) != len(set(hashes)):
                print("[eval-anat] ABORT: geometry hash and name fingerprint "
                      "do not agree — per-anatomy attribution would be wrong.")
                return 2
            if not parity_ok:
                print("[eval-anat] ABORT: the cohort env is NOT task-identical "
                      "to DualDeviceNavRCCAVaried; results would not be "
                      "comparable with any host number.")
                return 2
        # `--topbrain_only <one name>` is a legitimate single-anatomy run (the
        # exact-balance per-anatomy arm), so one geometry is correct there.
        single_expected = (a.real_patient_anatomy or a.frozen_anatomy
                           or (a.topbrain and a.topbrain_only is not None
                               and len(a.topbrain_only) == 1))
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
        if a.topbrain:
            # HOW THE COHORT IS SAMPLED — read this before quoting a
            # per-anatomy rate. TopBrainAnatomySet.reset() zeroes _generation
            # whenever it is handed a seed (it always is here), so from episode
            # 1 on the anatomy is _index_for(0) = a pure function of that
            # EPISODE's vessel seed, not a round-robin over the roster. The
            # pooled cohort rate is therefore unbiased, but coverage is
            # multinomial: with 22 anatomies some will draw few episodes. For an
            # exactly balanced per-anatomy number, run one arm per anatomy with
            # --topbrain_only <name>. The per-anatomy table in the report prints
            # the OBSERVED counts, so the imbalance is never hidden.
            print("[eval-anat] TOPBRAIN sampling: anatomy per episode is a "
                  "pure function of that episode's seed (not round-robin) — "
                  "coverage is multinomial; the report prints observed "
                  "per-anatomy counts. Use --topbrain_only <name> for an "
                  "exactly balanced single-anatomy arm.")
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
    analyze(log_dir, out_dir, n_expected=len(seeds), k_official=k_off,
            named_anatomies=bool(a.topbrain))
    return 0


def analyze(log_dir: str, out_dir: str, n_expected: int = None,
            k_official: int = None, named_anatomies: bool = False):
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
    # Widened from [0-9a-f]+ to alphanumerics. For the procedural env the field
    # is a 12-char md5 and both patterns match the identical substring, so this
    # is a no-op there; TopBrain's identity, however, is the NAME (topcowmr023),
    # which the hex-only pattern could not have matched.
    ANA = re.compile(r"anatomy=([0-9A-Za-z]+)")
    # `mesh_fp` is env5's echo of vessel_tree.mesh_fingerprint. For the
    # procedural env that is "s<seed>g<gen>" and is NOT an anatomy identity (the
    # seed is reassigned on every reset — this is exactly how the single-anatomy
    # eval bug hid). For TopBrainAnatomySet it IS the identity: the anatomy name
    # stripped to alphanumerics. Only used when `named_anatomies` says so.
    MFP = re.compile(r"mesh_fp=([0-9A-Za-z]+)")
    SEED = re.compile(r"seed=([0-9]+)")
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
                    mf = MFP.search(line)
                    ms = SEED.search(line)
                    live[pid] = {"t": float(m.group(1)) if m else 0.0,
                                 "pl": None, "succ": False, "steps": 0,
                                 "pid": pid,
                                 "anat": ma.group(1) if ma else None,
                                 "name": mf.group(1) if mf else None,
                                 "seed": ms.group(1) if ms else None}
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

    # --- NAMED per-anatomy attribution (TopBrain cohort only) ------------
    # Default-off: for the procedural env `mesh_fp` is s<seed>g<gen>, which is
    # not an anatomy identity, so printing a table keyed on it would be worse
    # than useless. Under --topbrain it is the loader's name fingerprint, and
    # this is the table the difficulty probe is for.
    if named_anatomies:
        named = [r for r in rows if r.get("name")]
        if not named:
            print("*** TOPBRAIN ATTRIBUTION MISSING: no mesh_fp= field in the "
                  "STEP logs. Per-anatomy rates CANNOT be reported — mount the "
                  "RL_IMPROV_18 env5.py. ***")
        else:
            if len(named) != len(rows):
                print(f"*** {len(rows)-len(named)} of {len(rows)} episodes "
                      f"carry no mesh_fp — per-anatomy table is incomplete ***")
            # Cross-check the two independent identities. They must partition
            # the episodes identically; if a name maps to two geometries (or
            # vice versa) the attribution is unsafe and must not be quoted.
            by_name = {}
            for r in named:
                by_name.setdefault(r["name"], set()).add(r.get("anat"))
            by_geom = {}
            for r in named:
                by_geom.setdefault(r.get("anat"), set()).add(r["name"])
            bad = ([f"{k}->{sorted(v)}" for k, v in by_name.items() if len(v) > 1]
                   + [f"{k}->{sorted(v)}" for k, v in by_geom.items() if len(v) > 1])
            if bad:
                print(f"*** ATTRIBUTION INCONSISTENT (name<->geometry is not "
                      f"1:1): {bad} — DO NOT report per-anatomy rates ***")
            else:
                print(f"attribution check: {len(by_name)} names <-> "
                      f"{len(by_geom)} geometry hashes, 1:1 — per-anatomy "
                      f"rates are trustworthy")
            print(f"\n{'anatomy':16s} {'succ':>9s} {'rate':>8s} {'95% CI':>16s} "
                  f"{'path_len mm':>14s}")
            per_rows = []
            for name in sorted(by_name):
                d = [r for r in named if r["name"] == name]
                ks = sum(1 for r in d if r["succ"])
                l2, h2 = wilson(ks, len(d))
                print(f"{name:16s} {ks:4d}/{len(d):-4d} "
                      f"{100*ks/len(d):7.1f}% "
                      f"{100*l2:6.1f}-{100*h2:-5.1f}% "
                      f"{min(r['pl'] for r in d):6.0f}-"
                      f"{max(r['pl'] for r in d):-6.0f}")
                per_rows.append((name, len(d), ks,
                                 sorted(set(r.get("anat") for r in d))[0],
                                 min(r["pl"] for r in d),
                                 max(r["pl"] for r in d)))
            counts = [c for _, c, _, _, _, _ in per_rows]
            print(f"coverage: {len(per_rows)} anatomies, episodes/anatomy "
                  f"min {min(counts)} median {int(np.median(counts))} "
                  f"max {max(counts)} (sampling is seed-driven, not "
                  f"round-robin — see the TOPBRAIN sampling note above)")
            ap = os.path.join(out_dir, "anatomy_success.csv")
            with open(ap, "w") as fh:
                fh.write("anatomy,geometry_hash,n_episodes,n_success,"
                         "success_rate,path_len_min_mm,path_len_max_mm\n")
                for nm, c, s, g, p0, p1 in per_rows:
                    fh.write(f"{nm},{g},{c},{s},{s/c:.4f},{p0:.1f},{p1:.1f}\n")
            print(f"per-anatomy CSV: {ap}")

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
        # Extra columns ONLY under --topbrain, appended at the end so any
        # header-aware reader of the legacy 5-column file keeps working; the
        # default run writes the identical bytes it always did.
        if named_anatomies:
            fh.write("wall_time,path_len_mm,section,steps,success,"
                     "anatomy,geometry_hash,seed\n")
            for r in sorted(rows, key=lambda r: r["t"]):
                fh.write(f"{r['t']:.3f},{r['pl']:.1f},{section_of(r['pl'])},"
                         f"{r['steps']},{int(r['succ'])},"
                         f"{r.get('name') or ''},{r.get('anat') or ''},"
                         f"{r.get('seed') or ''}\n")
        else:
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
