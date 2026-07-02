# Handoff — Relaunch `lcca_awac_v2` on a new machine

**Audience:** an agent (or engineer) setting up and running this training job on a
second computer. Read this top to bottom before launching. Author: the RL_IMPROV_8
worktree session, 2026-07-02. Repo branch: `worktree-rl_improv_8`, at commit `c7a33e3`
or later.

---

## 1. What this project is trying to do (context)

We are training a reinforcement-learning policy for **autonomous neurovascular
guidewire navigation** inside a **SOFA** physics simulation. A catheter + guidewire
(2 devices, 4 continuous action dims: `[gw_trans, gw_rot, cath_trans, cath_rot]`) is
driven from an aortic insertion point (**z = 345**, "Challenge‑1", the hard from‑the‑
start variant) and must thread the **LCCA** (left common carotid artery) — one of four
supra‑aortic daughter vessels (RCCA / LCCA / RVA / LVA).

- **Algorithm:** AWAC (advantage‑weighted actor‑critic), *step*-based replay with
  **PER** (prioritized replay) + a **balanced two‑stream sampler** (a "clean" lane of
  successful LCCA demos). Warm‑started with 10k pretrain updates on a **curated seed
  buffer**, then continues online SOFA exploration. This is the same stack that hit
  ~86% on the earlier from‑fork RCCA task.
- **Success metric:** deterministic **eval Quality** = fraction of 98 held‑out seeds
  where the policy (mean action, noise off) reaches the deep LCCA target
  (`TargetReached`). Reported every 250k explore steps in `main.log` as `Quality: …`.

### Why v2 exists (the story so far)

- **`lcca_awac_v1`** (the baseline, `launch_lcca_awac_v1.sh`) **soft‑collapsed**: eval
  plateaued at ~7% (`10.2 → 3.1 → 7.1 → 7.1`). A 6‑agent forensic of its logs found
  **two independent problems**:
  1. **Stability bug — a `log_std` CEILING explosion.** The policy's log‑std head blew
     UP to its `+2` ceiling on all 4 dims (std 1.0→2.0), so actions railed from sheer
     sampling noise. (NOT the floor collapse we first suspected; `target_entropy` /
     alpha was a red herring.)
  2. **Navigation defect — LVA‑confusion is STRUCTURAL.** ~38% of episodes end in **LVA
     (the wrong daughter)** instead of LCCA, and this was already ~38% *when entropy was
     healthy* (it is NOT caused by the collapse). Mechanism: at the "bif2" fork
     (world‑CS z≈400), entering LCCA needs a leftward "hook" deflecting the tip‑x from
     ~73→~50 mm. **Successes do this 100% of the time; LVA failures 6%** and instead go
     straight up into the LVA corridor. A second ~43% of episodes under‑reach and buckle
     in the trunk (`trunk/vessel_end`).

- **`lcca_awac_v2`** (`launch_lcca_awac_v2.sh`, what you are running) applies two
  changes vs v1, no reward/observation/terminal/dynamics change:
  - **EXP‑1:** `--log_std_max 0.0` (was default +2). Brackets std into a healthy
    `[0.135, 1.0]` band, structurally preventing the ceiling explosion.
  - **EXP‑2:** `--balanced_fraction 0.6` (was 0.3). The clean lane is 100% LCCA‑hook
    demos; doubling its weight puts more gradient mass on the missing hook behavior.

- **First v2 run result (partial):** **EXP‑1 worked** — entropy stayed flat ~2.65 and
  clamp_fraction ~0.5% through 470k steps (v1 had gone negative / 25% by then). But the
  run was **killed at 470k by a host interruption** (not a crash) with only **one early
  eval (3.06% @257k)**. Training‑rollout LCCA‑success (stochastic) rose 7.2→8.1→11.4%
  over the run — encouraging, but that is a *noisy‑rollout* stat, **not** the eval. The
  one deterministic eval (3%) was actually *below* the noisy rollouts, meaning the
  policy's *mean* hasn't locked onto the hook yet.

### The goal of THIS relaunch

Run v2 **to completion** (many evals, ~2M steps) with the **same 250k eval interval**,
to answer the open question: **does the deterministic eval Quality climb past ~7% and
does LVA% drop below ~38%?**
- **If yes:** EXP‑1+2 solve it; move on (other daughters).
- **If eval stays ~3–7% and LVA stays ~38%:** we've cleanly isolated it — stability is
  fixed but the LVA hook is genuinely structural, and **EXP‑3** is the next lever: a
  fork‑disambiguating **observation feature** or an **LCCA‑ostium waypoint reward**.
  EXP‑3 touches observation/reward, so it **requires the human's explicit approval**
  (standing rule: per RL iteration change only the variable under study + stabilization
  knobs; never touch reward/observation/terminal/dynamics unless asked).

---

## 2. Prerequisites on the target machine

| dependency | how to get it | notes |
|---|---|---|
| Docker image `eve-training-fixed` (~40.8 GB) | already present on target | verify: `docker images eve-training-fixed`. Contains SOFA + the eve/eve_rl/eve_bench base install + centerline/mesh data. If missing, rebuild from `./dockerfile` (slow) or `docker save`/`load` it from the source machine. |
| NVIDIA GPU + nvidia‑docker | host | launcher uses `--gpus all -d cuda:0`, `--shm-size=24g` |
| This git repo @ `worktree-rl_improv_8` (HEAD `c7a33e3`+) | `git clone` + `git checkout worktree-rl_improv_8` | all mounted CODE is committed |
| **`saved/lcca_awac_seed_v1.npz` (157 MB)** | **transfer OUT‑OF‑BAND** (see §3) | the curated seed buffer. NOT in git (exceeds GitHub's 100 MB limit). This is the "heatup data". |

---

## 3. Transferring the seed buffer (the only non‑git artifact)

`saved/lcca_awac_seed_v1.npz` is 157 MB → **cannot be pushed to plain GitHub.** Options:
- **scp / rsync** from the source machine (simplest if both are reachable).
- **Cloud drive** (Google Drive / Dropbox / S3) — upload once, download on target.
- **Git LFS** if you want it version‑controlled (`git lfs track "saved/*.npz"`), then it
  can live in the repo.
- **USB** if the machines are colocated.

Place it at `<repo>/saved/lcca_awac_seed_v1.npz`. The launcher mounts `saved/` →
`/opt/eve_training/results` and reads `--heatup_cache_file
/opt/eve_training/results/lcca_awac_seed_v1.npz`. No other file from `saved/` is read
(v2 is from z=345 with **no** `--checkpoint_dir` restore).

> Seed provenance (for reference): 8,738 episodes / 1.07M transitions — all LCCA
> successes + ostium near‑misses + wrong‑branch + reservoir‑thinned LVA/trunk failures.
> `is_clean=1012`, all of them LCCA (the clean lane is pure, verified). Rebuildable via
> `scratchpad/build_seed.py` from a heatup harvest, but just copy the .npz.

---

## 4. ⚠️ CRITICAL: fix the hardcoded host paths in the launcher

`launch_lcca_awac_v2.sh` mounts files with **absolute Windows paths**:
`-v "D:\neve\.claude\worktrees\rl_improv_8\...:/opt/..."`.

- **Same path on target (Windows, repo at `D:\neve\.claude\worktrees\rl_improv_8`):**
  run as‑is.
- **Different path or Linux:** you MUST rewrite the mount prefix (and, on Linux, the
  backslashes + the trailing `\` line‑continuation style is fine but the host side must
  be POSIX). Example prefix rewrite:
  ```bash
  # from repo root, adjust to your actual absolute repo path:
  REPO="$(pwd)"                      # e.g. /home/you/neve
  sed -i "s|D:\\\\neve\\\\.claude\\\\worktrees\\\\rl_improv_8|${REPO}|g; s|\\\\|/|g" launch_lcca_awac_v2.sh
  ```
  Verify every `-v` host path resolves to a real file before launching:
  `grep -oE '\-v "[^"]+"' launch_lcca_awac_v2.sh` and spot‑check.

---

## 5. Run it

```bash
cd <repo>
# (adjust launcher paths per §4 if needed)
# ensure saved/lcca_awac_seed_v1.npz is in place (§3)
MSYS_NO_PATHCONV=1 bash launch_lcca_awac_v2.sh      # Git Bash on Windows; drop the env var on Linux
```

The launcher: removes any old `lcca_awac_v2` container, then `docker run -d --name
lcca_awac_v2 …`. Config (do NOT change for this relaunch — "same eval interval" was the
instruction): AWAC λ=3.0, step PER, `balanced_fraction 0.6`, `grad_clip 1.0`,
`log_std_min -2`, **`log_std_max 0.0`**, MLP 256×256, lr 3e‑4, `pretrain 10000`, 11M
buffer, eval every **250k** steps (the trainer constant `EXPLORE_STEPS_BTW_EVAL=2.5e5`),
`-nw 16 -d cuda:0`.

Startup takes ~5–10 min (SOFA init + seed load + 10k pretrain). Expect the log line
`Loaded 8738 episodes (1065454 steps)` and `Warm-start: 10000 pretraining updates`.

---

## 6. Monitoring (what to watch, and the thresholds)

Run dir: `saved/eve_paper/neurovascular/full/mesh_ben/<timestamp>_lcca_awac_v2/`.

- **Live log:** `docker logs -f lcca_awac_v2`
- **Stability metrics** (`diagnostics/csv/losses_trainer_synchron.csv`, 1‑indexed cols):
  `col3`=explore_step, `col11`=**entropy_proxy** (want ~2.6, POSITIVE — EXP‑1 holding),
  `col20`=**clamp_fraction** (want <0.05; v1 relapse = climbing past 0.15), `col12`=q1_mean
  (bounded, ~‑3), `col23/24/25`=nonfinite counts (must be 0).
- **The key metric — eval Quality** (deterministic, every 250k):
  `grep 'Quality:' <rundir>/main.log`. **This is the number that matters** (not the
  training‑rollout success — see the v1/v2 lesson: noisy rollouts can beat the mean).
- **Navigation breakdown (LVA vs LCCA)** — tally the per‑episode outcome logs:
  ```bash
  LG=<rundir>/diagnostics/logs_subprocesses
  cat $LG/*.log | grep EPISODE_OUTCOME \
    | awk -F'|' '{for(i=1;i<=NF;i++){if($i~/final_branch=/){gsub(/.*final_branch=/,"",$i);gsub(/ /,"",$i);f=$i}}; print f}' \
    | sort | uniq -c | sort -rn
  ```
  Watch whether **`final_branch=LCCA | reason=success`** grows and **`LVA`** shrinks as
  training progresses (compare late third vs early third).

---

## 7. Decision criteria + next steps

- **EXP‑1 (stability):** already validated. Just confirm it holds — entropy stays
  positive (~2.6), clamp stays <5%, no NaN, no divergence. If it *does* relapse
  (entropy→negative, clamp→>15%), something regressed — check `log_std_max` actually
  reached the policy (no crash on `--log_std_max`).
- **EXP‑2 (the LVA lever) — the real question this run answers:** over several evals,
  does **deterministic eval Quality climb clearly above ~7%** AND does **LVA% drop below
  ~38%**?
  - **Yes →** EXP‑1+2 solved LCCA from z=345. Report it; consider replicating for
    RCCA/RVA/LVA.
  - **No (eval stuck ~3–7%, LVA ~38%) →** the hook is confirmed structural. **Do NOT
    keep throwing compute at it.** The next lever is **EXP‑3** (fork‑disambiguating obs
    feature and/or LCCA‑ostium waypoint reward). EXP‑3 changes observation/reward →
    **stop and get the human's explicit approval first** (minimal‑change rule).
  - **Yellow flag to report:** if deterministic eval stays *below* stochastic training
    LCCA‑success, the policy mean isn't learning the hook — that points to EXP‑3.

---

## 8. Quick reference — cheatsheet

```bash
docker images eve-training-fixed                 # confirm image present
ls -la saved/lcca_awac_seed_v1.npz               # confirm seed (157 MB) in place
# (fix launcher paths per §4 if repo isn't at D:\neve\.claude\worktrees\rl_improv_8)
MSYS_NO_PATHCONV=1 bash launch_lcca_awac_v2.sh   # launch
docker logs -f lcca_awac_v2                       # watch startup
grep 'Quality:' saved/eve_paper/*/*/*/*/*_lcca_awac_v2/main.log   # evals (every 250k)
docker stop lcca_awac_v2                          # stop
```

Related in‑repo docs: `RL_IMPROV_8_CHANGES.md`, the Plan v9–v12 history, and the
launcher header comments in `launch_lcca_awac_v1.sh` / `launch_lcca_awac_v2.sh`.
