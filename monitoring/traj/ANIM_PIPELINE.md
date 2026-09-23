# Animating validation episodes: record → type → render

Design for turning validation-eval episodes — especially the recovery successes that
carry the last ten points on carotid — into key-frame sets and short movies. Derived
from the trajectory atlas (`saved/traj/`, Sep 18 2026) and the recovery phase timing
measured on 362 closed stall events in 359 carotid recovery-type validation successes.

## 1. What a recovery looks like in time (the numbers the schedule is built on)

Carotid validation, recovery-type successes (light + deep recovery clusters), per closed
stall event (median, IQR):

| moment | step | note |
|---|---|---|
| episode length | 233 (130–368, p90 514) | |
| first stalled step (frontier stops) | 73 (38–153) | the frame you want *before* anything looks wrong |
| detector onset (12 consecutive stalled steps) | +11 after first (p90 +17) | the only point an *online* trigger can fire |
| deepest insertion = peak load | +24 after first (13–45, p90 85) | the buckle at its worst |
| shallowest insertion = deepest withdrawal | +0 after peak (p75 +8, p90 +28) | **53 % of events have no executed withdrawal at all** (grind) |
| frontier passed (escape) | +0 after withdrawal (p75 +7, p90 +29) | |
| whole event window | 38 (24–74, p90 138) | |
| escape → success | 62 (28–141) | |

Events per episode: one in 131 episodes, two in 52, three or more in 23. Withdrawal
depth: median 0 mm, p75 5 mm, p90 21 mm; slack at peak p75 12 mm, p90 23 mm.

Consequences:

- A recovery is **short and buried** — a 38-step window in a 233-step episode, with
  ≤ 23 mm of slack that disappears at whole-anatomy scale (the end-of-episode snapshot
  already shows this: a 39 mm buckle inside the siphon is invisible). Frames need a
  **fixed camera on the planned path** and a **tip inset** for the event frames.
- The informative moment (first stalled step) precedes the earliest possible online
  trigger by 11–17 steps, so live rendering must keep a **ring buffer** of ~20 steps.
- Half the "recoveries" are grinds (no withdrawal). For those, peak-load and
  withdrawal frames coincide; the schedule collapses to 3 frames per event.

## 2. Frame schedule

### Key frames (PNG, named by role — this is the "storyboard")

| id | when | why |
|---|---|---|
| `k0_start` | step at 25 % progress | approach; the step-1 frame is identical for every seed on the anatomy |
| `k1_stall_e<i>` | first stalled step of event *i* | frontier stops; nothing looks wrong yet |
| `k2_load_e<i>` | deepest insertion inside the event | max slack / buckle |
| `k3_withdraw_e<i>` | shallowest insertion after the peak — only if ≥ 1 mm was withdrawn | the actual retraction |
| `k4_escape_e<i>` | frontier passed (+1 mm) | the recovery |
| `k9_end` | terminal step | success / failure frame (what exists today) |

Per episode: 2 + 3–4 per event → **5–6 frames for the median single-event recovery,
8–10 for two events.** At ~45 KB per centerlines PNG that is ≈ 0.3 MB per episode.

### Sequence (movie)

Fixed cadence **every 5 steps** outside stall windows (tip moves ≤ 4 mm/step, so
≤ 20 mm between frames) and **every 2 steps inside** `[first − 5, escape + 5]`.
Frames per recovery episode: median ≈ 58, IQR ≈ 35–95, p90 ≈ 145. Render the sequence
straight to MP4 / GIF (0.3–0.6 MB at 6 fps) and keep only the key frames as PNGs;
`--dump-frames` writes the sequence PNGs when wanted (≈ 2.6 MB per episode).

### Frame layout

Two panels, fixed per episode: **left** the 3-D view with axis limits set once from the
planned path's bounding box (± 25 mm) so motion is visible frame to frame, devices in
the existing colours, target marker, faint on-path centerlines; **right** a progress
trace — `proj_s / path_len` and guidewire slack against step, with the stall windows
shaded and a cursor at the current step. Key frames add a **tip inset** (± 40 mm cube
around the guidewire tip). The trace is what makes a still legible as a recovery.

### Budget per validation block (98 episodes, carotid)

| policy | episodes rendered | key PNGs | movies | ≈ size |
|---|---|---|---|---|
| recoveries + all failures (default) | ~25–30 | ~180 | 25–30 | 25 MB |
| everything | 98 | ~500 | 98 | 90 MB |

Offline rendering at ~0.35 s/frame: ~20 s per recovery episode, ~10 min per block on
one core (parallel across episodes trivially).

## 3. Pipeline

### A. Record (in the eval worker, no rendering)

Per step, when `self.mode == "eval"` and `TRAJ_RECORD_DIR` is set: the two device
polylines from `intervention.fluoroscopy.device_trackings3d` → vessel CS, resampled to
64 nodes each by arclength, float16 (768 B); plus `proj_s, inserted[2], cmd[4], fold,
slack, cath_slack, tip3d, reward, term, trunc` (≈ 40 B). Per episode once: planned path,
target, `path_len`, seed, mesh fingerprint, target branch. Written at terminal/truncated
as one `.npz`; a 233-step episode ≈ 0.19 MB, a block ≈ 18 MB, ≈ 10 MB gzipped. Record
**every** eval episode — types are only known afterwards.

The block name (`eval_<kk>_<explore_steps>`) should come from the trainer, which knows
the eval index and spawns fresh eval workers per block (set `TRAJ_RECORD_BLOCK` in the
environment before spawning; workers inherit it). Fallback: name by first wall-clock
time and let the indexer cluster blocks the way `traj_load.eval_blocks` does.

### B. Type (offline, reuses the atlas)

`traj_extract.core_features` on the recorded series → nearest centre in
`saved/traj/cluster_model.npz` → one of the eleven types; the canonical stall detector
gives the events and the phase steps (first / peak / withdrawal / escape) for the key
frames. Same definitions as the atlas, so a rendered folder is directly comparable to
the tables.

### C. Render (offline, `traj_render.py`)

```
python monitoring/traj/traj_render.py <run_dir> [--blocks eval_03_1545187,...]
       [--types recovery,near_fight,all] [--cadence 5 --stall-cadence 2]
       [--movie mp4|gif|none] [--dump-frames] [--jobs 4]
```

### Online alternative (live rendering, no recording)

If recording is not wanted: an online copy of the canonical stall rule in `env5.step`
(`proj_s < running_max + 0.3 mm and |cmd_gw| > 2 mm/s`, 12 consecutive; note the
`abs()` — keep it for consistency with the atlas), a ring buffer of the last 20 steps'
polylines, and: at onset render `k1` from the buffer; while stuck keep copies of the
running deepest/shallowest-insertion states; at escape render `k2/k3/k4`; at terminal
render `k9`. Cadence frames render directly. Cost lands in the eval worker (~0.35 s per
frame → 5–10 % slower eval); no cross-block comparisons beyond what the frames show.
Recording is the better default: it is cheap, and every re-render (new camera, new
cadence, side-by-side seeds) is free.

## 4. Folder structure

```
<run_dir>/diagnostics/
  traj_records/                              # A. recorded state, validation only
    eval_03_1545187/                         # <block index>_<explore steps> — maps to the eval CSV row
      s044_casew013righttopcowmr013L.npz     # seed first: the same seed is comparable across blocks
      index.csv                              # seed, mesh, steps, success, reason, wt_start, pid, file
  anim/                                      # C. rendered; mirrors the record layout
    eval_03_1545187/
      3_light_recovery/                      # type folders carry the atlas cluster id, so they sort like the atlas
        s044_casew013…_142st_success/
          key/
            k0_start_s036.png
            k1_stall_e1_s059.png
            k2_load_e1_s102.png
            k3_withdraw_e1_s106.png
            k4_escape_e1_s121.png
            k9_end_s142.png
          episode.mp4
          trace.png                          # the 2-D progress / slack trace with event windows
          events.json                        # detector events, phase steps, frame → step map
      4_deep_recovery/
      6_near_fight/                          # failures that fought at the target (the strong runs' failure)
      7_midpath_thrash/ 10_prox_thrash/ 5_shove_fight/ 8_coil/ 9_shove_cap/
      0_clean/ 1_clean_fast/ 2_cathled/      # key frames only unless --types all
      index.csv                              # every episode of the block: type, steps, events, paths
    by_seed/                                 # generated index (no symlinks): one page per seed listing
      s044.html                              #   its episode in every block — fail-by-near-fight → recovery → clean
```

Naming rules: seed first, then mesh fingerprint, so `ls` groups the same task across
blocks; the step count and outcome in the episode folder name so the recovery episodes
are visible without opening anything; key frames prefixed `k0…k9` so they sort in
narrative order, with the event index and the step number in the name.

## 5. Implementation status (2026-09-19)

Built and smoke-tested end to end (4 replay episodes of the baseline's 94.9 % checkpoint
on the carotid validation anatomies: records → typed → key frames + trace + GIF +
indexes in 27 s):

| piece | file | notes |
|---|---|---|
| recorder (tier 2/3) | `training _scripts/util/traj_record.py` | `TrajRecordWrapper(env)`; records only while `TRAJ_RECORD_DIR` is set; both device polylines resampled to 64 nodes (float16) + tip, insertion, `proj_s`, cross-track, fold, off-branch, catheter slack, `d_tgt`, command, reward, term/trunc, wall time; planned path, target, centerlines (+ on-path flags) once per episode. ~1 KB per step. Never raises into the env. |
| replay hook | `training _scripts/eval_anatomies.py --traj_record` | wraps the master eval env and the per-worker factory; `--seed_list` replays explicit seeds (the trainer's `EVAL_SEEDS` → same seed→anatomy/target as the in-run validation blocks) |
| launcher | `launch_eval_carotid_traj.sh` | = `launch_eval_topbrain.sh` + `carotid_data` mount + `traj_record.py` mount + the 16 validation anatomies + `--traj_record`; `SEED_LIST=validation` for the 98 in-run seeds; passes `EVE_RL_ACTOR_ZERO_OBS` / `EVE_RL_CRITIC_ZERO_OBS` through (a masked checkpoint evaluated without them is silently wrong) |
| renderer | `monitoring/traj/traj_render.py` | types each record with `saved/traj/cluster_model.npz`, phases from the canonical detector, key frames `k0/k1/k2/k3/k4/k9`, 2-panel frames (fixed camera on the planned-path bbox + tip inset; progress and slack traces with stall windows), GIF via Pillow (no ffmpeg in the image or on the host), `index.csv` per block, `by_seed/*.html` |
| log-only tier (tier 1) | `traj_render.py --from-logs <tag> --scenery <records_dir>` | log episodes carry the guidewire tip only; the 3-D panel shows the tip track, and centerlines / planned path are borrowed from tier-2 records of the same anatomy. Needs series extracted after 2026-09-19 (they now store `tip`). |

Usage:

```bash
# replay a checkpoint on the validation seeds, recording every episode (~20-25 min / 98 episodes at 16 workers)
SEED_LIST=validation N_WORKER=16 ./launch_eval_carotid_traj.sh /opt/eve_training/results/.../checkpoints/checkpoint1545187.everl
# masked (ablation) checkpoints need the mask:
EVE_RL_ACTOR_ZERO_OBS=101,102,106,107,113,115,116,118,119,123 SEED_LIST=validation ./launch_eval_carotid_traj.sh .../checkpoint1005189.everl
# render recoveries + failures (default), movies as GIF, four processes
python monitoring/traj/traj_render.py <ckpt-dir>/eval_anatomies_<ckpt>/traj_records/<run_tag> <out_dir> --block ck1005189 --jobs 4
python monitoring/traj/traj_render.py --from-logs car_v3 --scenery <records_dir> <out_dir>      # tier 1
```

Measured on the first real replay (ablation checkpoint 1760615, the 13 seeds that failed
its latest in-run block, 3 workers alongside the live training run): 13 records in
~45 min (the 600-step failures dominate), ~1 KB per recorded step; the replay produced
2 light recoveries, 1 deep recovery, a shove-fight success, a near-target buckle fight,
a mid-path thrash and a 243 mm mega-coil — and 6 clean successes on seeds that had
failed in-run, the ORBIT noise floor at work. Rendering: ~30 s per episode at two
processes; key frames 5–13 per episode; GIFs 1.9–5.8 MB with the dense cadence limited
to ±15 steps around the phase points (a 230-step stall at 2-step cadence throughout was
a 7–12 MB GIF), before the 900 px downscale. The recorded tip agrees with the log's
`tip3d` to 0.06 mm after `tracking3d_to_vessel_cs((20, 5), (0, 0, 0))` — which is how
tier 1 places historical episodes on the scenery.

First result the pipeline produced — the ablation's peak checkpoint (1005189, 91.8 %
in-run) against its current one (1760615, 86.7 %), replayed on the 13 seeds that failed
its latest in-run block, rendered into
`2026-09-17_180209_rcca_carotid_v3_noprivactor/diagnostics/anim/` (`by_seed/` pages
show each seed under both checkpoints):

| seed | peak ck1005189 | current ck1760615 |
|---|---|---|
| 91 | clean, 70 steps | light recovery, 354 steps |
| 118 | clean, 58 steps | deep recovery, 427 steps |
| 120 | clean, 37 steps | light recovery, 456 steps |
| 122 | clean, 51 steps | shove fight (won), 208 steps |
| 93 | clean, 92 steps | mid-path thrash, failed |
| 161 | shove fight (won), 232 steps | mega-coil, failed |
| 55 | near-target buckle fight, failed | near-target buckle fight, failed |
| 8, 43, 44, 50, 80, 108 | clean (26–85 steps) | clean (34–71 steps) |

12/13 vs 10/13 succeeded, but 11 clean successes at the peak against 6 now: the seeds the
peak solved in ~60 steps are the ones the current checkpoint survives only by recovery or
loses to thrash and coil — the atlas' "seeds migrate from recovery to clean as a run
improves" running backwards as this run decays. n = 13 and one replay each; the ORBIT
twist is unseeded, so a seed's outcome is a draw, not a constant.

Not done: the env5 hook for recording the in-run validation blocks (tier 3) — `env5.py`
is bind-mounted into the live ablation container; wrap `env_eval` in
`DualDeviceNav_train.py` with `TrajRecordWrapper` after the run ends. Movies are GIF
only; an MP4 path needs ffmpeg in the image.

## 6. Constraints and open points

- The running container **bind-mounts** `training _scripts/util/env5.py`,
  `snapshot.py` and `DualDeviceNav_train.py` from this worktree. Editing them while
  the ablation run is live changes the code its next eval workers import. Implement the
  recorder as a new module and wire the two hook lines only after the run ends (or in a
  copy of the tree the container does not mount).
- Hook points: end-of-episode snapshot at `env5.py:1745–1766` (single-grader path) and
  the `MultiTargetEnv5` driver at `:2762–2810`; the eval phase is `self.mode == "eval"`
  in both. The per-step hook goes next to the `STEP |` log emission.
- Snapshot filenames use `pid` + `ep`, which collide across eval blocks; the record
  layout above avoids that by keying on seed + mesh inside a block folder.
- The recorded state also removes the need to parse 13 GB of worker logs for the next
  trajectory analysis: `traj_extract.core_features` can run on the `.npz` directly.
