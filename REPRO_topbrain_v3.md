# Reproducing the TopBrain v3 run (mirror of rcca_topbrain_v2)

Written 2026-09-05 on the branch-18 machine, where `rcca_topbrain_v2` (SDF-remeshed
v2 anatomies, 49-anatomy roster) is currently training. This describes how to run
the identical configuration against `anatomies_v3` (real, non-remeshed surfaces)
on another machine. v3's roster is byte-identical to v2's: the same 49 directory
names, all 49 passed geometry intake (`topbrain_tools/check_anatomies.py`).

## Prereq

Branch `rl_improv_18_p2` at or after commit `a16750f` (anatomy sync). It contains
`topbrain_data/anatomies_v3/` and `launch_rcca_topbrain_v2.sh` (committed alongside
this doc).

## Recipe

Copy `launch_rcca_topbrain_v2.sh` -> `launch_rcca_topbrain_v3.sh` and make exactly
three kinds of edits:

1. Container/run name: `rcca_topbrain_v2` -> `rcca_topbrain_v3` (4 spots: the two
   Monitor/Stop comment lines, `docker rm`, `docker run --name`, and the `-n` flag).
2. Anatomy dir: `--topbrain_dir /opt/eve_training/topbrain_data/anatomies_v3`
3. Host mount prefix: the `-v` lines carry the branch-18 machine's repo root
   (`D:\neve\.claude\worktrees\rl_improv_18_p2`). Replace that prefix with your
   local repo root everywhere. Only the host side of each mount changes; container
   paths stay as-is.

Everything else must stay byte-identical. The full topbrain block, for checking:

```
--topbrain \
--topbrain_dir /opt/eve_training/topbrain_data/anatomies_v3 \
--topbrain_seed 12345 \
--topbrain_change_every 10 \
--topbrain_holdout topcow_mr_007 topcow_mr_007_L topcow_mr_008 topcow_mr_008_L topcow_mr_017 topcow_mr_017_L topcow_mr_022 topcow_mr_022_L \
--topbrain_exclude __none__ \
--topbrain_target_min_arclength 133.0 \
```

## Why `--topbrain_exclude __none__`

`DualDeviceNav_train.py` has an argparse default of
`["topcow_mr_013", "topcow_mr_014", "topcow_mr_015"]` for `--topbrain_exclude`
(leftover from the v1 25-anatomy set, where those meshes were unusable). Their v2/v3
rebuilds pass intake and must train. The exclude filter in
`eve_bench/eve_bench/dualdevicenavtopbrain.py` is a silent set-difference with no
name validation, so the non-existent sentinel `__none__` overrides the default with
a clean no-op. Do NOT simply omit the flag - that re-activates the default and
silently drops 3 right-side anatomies while keeping their _L twins.

## Resulting split (patient-level, no L/R leakage)

- TRAIN - 41 anatomies, 21 patients: both `_L` and right of
  topcow_mr_009 010 011 012 013 014 015 016 018 019 020 021 023 024 025 026 027
  028 029 030 (20 patients x 2), plus `topcow_mr_006` (right only - no `_L` twin
  exists for 006, so it is unpaired and cannot leak).
- HELD-OUT (validation eval, never trained) - 8 anatomies, 4 patients:
  007, 007_L, 008, 008_L, 017, 017_L, 022, 022_L.
- EXCLUDED - none in effect (only the `__none__` sentinel echoes).

## Verify at launch (abort if any check fails)

The startup roster echo must read exactly:

```
TRAIN 41
HELD-OUT (eval only, never trained) 8
EXCLUDED-unusable 1: ['__none__']
```

and no name containing 007/008/017/022 may appear in the TRAIN list. Both `[v3c]`
config lines must echo as in the v1/v2 runs.

## Unchanged from v1 (do not touch)

Seed 12345, change_every 10, holdout patient IDs, target_min_arclength 133.0, all
reward/obs/SAC/`--privileged_actor` flags, `--shm-size=30g`, worker count.

## Notes

- v3 = real surfaces; tri density differs from v2's ~20k-tri SDF meshes, so SOFA
  step cost will differ. Check throughput early.
- Per HANDOFF 10.2, validation numbers saturate; host TEST
  (`eval_anatomies.py --real_patient_anatomy`) is the only instrument with
  resolution. Branch-18 plan: host-test after >=6 checkpoints exist.
