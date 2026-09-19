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

## 5. Constraints and open points

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
