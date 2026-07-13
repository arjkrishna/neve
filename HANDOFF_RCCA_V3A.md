# Handoff — Run `rcca_procedural_v3a` on a second machine

**Audience:** an agent or engineer launching the RL_IMPROV_16 Tier-A experiment on a
second computer while `rcca_procedural_v2` keeps running on machine 1. Read top to
bottom before launching. Branch `rl_improv_15` at commit `0cd073c` or later.
Companion docs: `RL_IMPROV_16_EXPERIMENTS.md` (the full experiment plan; v3a is Tier A),
`RL_IMPROV_15_CHANGES.md` (the fix package v3a builds on), `saved/monitor_rcca_procedural.md`
(the v1 collapse forensic + v2 live monitor log), `saved/v2_micro_investigation/` (the
investigation that motivated every v3a change).

---

## 1. What this is trying to do (context)

Training an RL policy for **autonomous neurovascular guidewire navigation** in a **SOFA**
physics sim. Catheter + guidewire, 4 continuous actions `[gw_trans, gw_rot, cath_trans,
cath_rot]`, inserted at the **RCCA ostium** and threading the **RCCA → carotid siphon**
across **16 per-worker procedurally-varied anatomies** (anti-memorization: each worker
sees a distinctly-perturbed RCCA/siphon, re-randomized every 10 episodes; eval is a
fixed held-out RCCA). Success = deterministic **eval Quality** = fraction of 98 held-out
seeds reaching the deep RCCA target, reported every ~250k explore steps.

**Stack:** AWAC + step-PER + balanced clean-lane sampler, mesh-invariant obs (121-flat:
policy sees 97, privileged critic sees all 121), asymmetric critic + aux distillation,
recovery training (failures don't truncate), anti-buckle potential shaping. Warm-started
with a 282k-transition harvested seed + 10k pretrain, then online SOFA exploration.

### The story so far (why v3a exists)

- **`rcca_procedural_v1`** learned to ~34% explore / 13.3% eval, then **froze**
  (deterministic policy collapsed to 0.95 mm/s) and IPC-deadlocked. A 5-agent forensic
  found: std pinned at its ceiling → alpha decayed to its floor then whipsawed →
  entropy term crushed the action mean → and AWAC weights ≈ 1.0 (inert). See
  `RL_IMPROV_15_CHANGES.md` Part E.
- **`rcca_procedural_v2`** (`launch_rcca_procedural_v2.sh`) applied the F1–F8 fix
  package (configurable alpha rails, awac_lambda 1.0, action_mean_penalty, clean-lane
  rail filter, IPC timeout guard, pretrain-baseline eval). It **worked**: eval
  trajectory **6.1% (pretrain baseline) → 30.6% → 49.0%** — 3.7× v1's all-time peak,
  first positive eval reward, ≈0 train/eval gap (generalizes across the procedural
  meshes). v2 is the **control** and keeps running on machine 1.
- **A live investigation of v2** (`saved/v2_micro_investigation/`) then found the ceiling
  on further gains:
  1. **Credit assignment too weak** — adv std ~0.09 with λ=1.0 gives AWAC weights
     [0.72, 1.25] ≈ uniform behavior-cloning; the critic *has* a correctly-signed
     retract-when-stuck preference but cannot transmit it to the policy.
  2. **Micro-recovery (stuck→retract→re-advance) is fading**, not learned — it was
     distilled from exploration noise that is drying up as entropy collapses; stuck
     states are only ~10%/1% of the buffer (gradient-starved).
  3. **The aux/privileged channel is half-dead** — the SOFA force labels are identically
     zero (SOFA `MechanicalObject.force` is a per-solve scratch buffer, zero at any
     post-step read), and the live contact labels are scale-starved (std ~1e-3 →
     aux gradient ~5e-8).

### What v3a changes (Tier A of RL_IMPROV_16 — bundle, one launch)

All default-off knobs; with them absent the code is byte-identical to v2.

| E | Flag (in `launch_rcca_procedural_v3a.sh`) | Attacks |
|---|---|---|
| **E1b** | `--awac_adv_norm_tau 2.0` — batch-normalized advantages `exp((adv/adv.std())/tau)` | weak credit assignment (finding 1); targets weight p99/p1 ≈ 10 |
| **E2** | `--aux_labels "0,1,5,6"` (off dead force dims → node velocities + contact) + `--aux_label_znorm` (loss-time z-scoring) | half-dead aux (finding 3) |
| **E3** | `--stuck_fraction 0.15` — third sampling lane over stuck states | gradient-starved stuck states (finding 2) |
| **E4-prep** | `-e STUCK_CHECKPOINT_DIR=…` — harvest a mesh-fingerprinted stuck pool | supplies the v3b restore curriculum (zero cost) |
| **E8** | `monitoring/monitor_pass_v3a.sh` + `probe_policy_v3.py` | measures every gate below |

### Goal of this run

Reach eval2 (~explore 510k, ~10h) and answer: **does the retract-when-stuck coupling
appear and steepen** (the direct micro-recovery-learning signal, flat/decaying in v2),
and **does eval Quality hold ≥ v2's curve** (30.6% @ eval1, 49% @ eval2)? If yes → v3b
adds the stuck-restore curriculum (E4). If a specific gate fails → peel that one flag
(see §6).

---

## 2. Prerequisites on the target machine

| dependency | status / how | notes |
|---|---|---|
| Docker image `eve-training-fixed` (~40 GB) | **already present on target** — verify `docker images eve-training-fixed` | contains SOFA + eve/eve_rl/eve_bench base install + centerline/mesh data. The launcher mounts individual updated source files OVER the image's older copies. |
| NVIDIA GPU + nvidia-docker | host | launcher uses `--gpus all -d cuda:0 --shm-size=24g --init` |
| Repo at branch `rl_improv_15` ≥ `0cd073c` | `git clone` / `git pull` | brings all v3a code + launcher + `monitoring/` + this handoff |
| **Seed buffer** `saved/rcca_proc_heatup/seed.npz` | **download from Google Drive** (the human is uploading it; ~68 MB, NOT in git) | after download, verify: 67,776,360 bytes, md5 `54fe108c45ec46978d8987c00b09e2b7`, 282,310 transitions / 480 episodes / `meta_buckle_coef=0.5`. Place at `<repo>/saved/rcca_proc_heatup/seed.npz`. The run pretrains from it; the reward-version guard (`EVE_RL_BUCKLE_COEF` vs the cache's `meta_buckle_coef`) fails fast on a mismatch, so a wrong/corrupt file aborts cleanly rather than training on bad data. |
| Host stays awake | disable sleep/suspend | the v1 deadlock trigger was a host suspend mid-IPC. The v3a IPC guard converts a hang into a loud crash but does not survive it. |

**Path fix.** The launcher's `-v` mounts hardcode `D:\Arjun\workspace\neve`. If the
checkout is elsewhere, rewrite the prefix:
```bash
sed -i 's|D:\\Arjun\\workspace\\neve|<your-abs-repo-path>|g' launch_rcca_procedural_v3a.sh
```
(On Linux the mount source syntax also becomes forward-slash; adapt if not on Windows/Git-Bash.)

---

## 3. Launch

```bash
cd <repo>
git checkout rl_improv_15 && git pull      # ensure >= 0cd073c
ls -l saved/rcca_proc_heatup/seed.npz      # confirm the seed is present (67,776,360 bytes)
bash launch_rcca_procedural_v3a.sh
docker logs -f rcca_procedural_v3a         # watch startup
```

**Expected startup (~5 min):** 16 `[Gen-4] varied-RCCA` worker seeds (12345..12360), seed
cache loaded (`buffer_len=282310, episodes_received=480`), then `Warm-start: 10000
pretraining updates`, then `Post-pretrain BASELINE eval` (~30 min, explore=0, banks
`checkpoint0`), then online exploration. Confirm the new flags in the process cmdline:
`--awac_adv_norm_tau 2.0 --aux_label_znorm --stuck_fraction 0.15 --aux_labels 0,1,5,6`
and env `STUCK_CHECKPOINT_DIR`, `EVE_CLEAN_RAIL_MAX=0.15`, `EVE_RL_MODEL_QUEUE_TIMEOUT_S=900`.

---

## 4. Monitor (every ~2h)

```bash
bash monitoring/monitor_pass_v3a.sh          # set CONTAINER=... if you renamed it
```
Reports: liveness, the v2 fix-package checks (alpha band, clamp, nonfinite, rail-filter,
IPC guard, recovery relax, diversity, outcomes) PLUS the Tier-A gates. The deterministic
probe (`probe_policy_v3.py`) runs inside the container CPU-only (safe alongside training)
and prints the freeze probe, retract-vs-slack coupling, aux R², and buffer stuck-share.

**Known-benign (do NOT flag):** SOFA `IRController "Case 1"` log spam; `can_sample`
toggling; `update_step` frozen during the first ~30-min explore cycle after pretrain
(the first 100-episode cycle runs 0 updates by design); STATUS-line `update_step` lags
(losses CSV is authoritative); trainer at ~1% CPU (GPU-bound); ~98-episode eval windows
interleaving into the step logs every ~250k explore steps; rare `vessel_end` at
`steps=600` (benign terminal label).

---

## 5. Gates (GO / peel-off / abort)

| Gate | GO | Trouble |
|---|---|---|
| **Eval trajectory** | eval1 (~287k) ≥ v2's 30.6%; eval2 (~510k) climbing; eval speed ≥ 3 mm/s | eval < 30.6% for 2 evals → peel (see §6) |
| **E1b weight spread** | `awac_weight_p99p1` ∈ [5, 20] within ~20k online updates | ~1.7 = still BC-degenerate (τ too high / adv-norm off); ≫20 or `awac_weight_saturation` rising = over-sharp → raise τ |
| **E2 aux** | `aux_loss` column O(0.1–2) (not ~0); contact R² ≥ 0.55 **and rising** past u≈130k; velocity R² ≥ 0.5 | `aux_loss` ~0 = z-norm not active; policy_loss noise ↑ >2× → halve `--aux_coef` |
| **E3 stuck lane** | probe "buffer stuck-share" ~0.10; success not dropping; `q1_mean` slope not steeper than v2's (−1.9 @ this point) | q1_mean diving faster than v2 = over-sampling failures biasing the critic pessimistic |
| **THE headline gate** | probe **retract-vs-slack**: the slack-tail bin's `P(retract)` minus the base bin should be **positive and GROWING** across snapshots (v2 was flat, decaying +13pp→+8pp) | flat/decaying = E1+E3 insufficient → escalate to v3b (E4 restore curriculum) |
| **Freeze probe** | mean\|a0\| ratio ≥ 2× the run's own pretrain baseline, not falling | ratio → 1 with eval speed falling = freeze (should be impossible with the v2 fix package intact) |
| **IPC guard** | `IPC TIMEOUT` count = 0 | any occurrence = a subprocess hung; the guard crashed it visibly — inspect, do not blind-restart |
| **Deadlock recovery (NEW)** | `Restarting Trainer because of … deadlock guard` = the trainer-result deadline fired and the run **self-recovered** (continues) — expected at eval3 if the v1/v2 race recurs. `WATCHDOG … Hard-exiting (os._exit 42)` = the catch-all fired: the container exits code 42 after a total stall; **restart it** (progress up to the last checkpoint is safe). | container exit 42 + a `WATCHDOG` line = a hang the trainer-restart didn't cover — capture the logged thread-wchan dump before relaunching |

---

## 6. Peel-off protocol (the bundle is intentional)

The three levers (E1/E2/E3) are legs of one mechanism (retract-when-stuck needs
audible credit + present stuck data + a sharp representation), so they launch together;
attribution comes from the **per-component metrics above**, not from separate runs. If a
gate fails, relaunch minus the implicated flag — each is default-off and the seed reuse
makes a relaunch ~25 min (startup + 7-min pretrain):

- E1b implicated (weights concentrated / saturating, success −10pts): drop
  `--awac_adv_norm_tau` (reverts to λ=1.0) or lower τ toward 1.0. **Highest-risk lever.**
- E3 implicated (q1_mean pessimism outpacing v2): drop `--stuck_fraction`. Moderate risk.
- E2 implicated (≈ never — its prior state was "trains on constants"): drop
  `--aux_label_znorm`. Near-zero risk.

Do NOT change two knobs on the same distribution in one run (e.g. `--awac_lambda` AND
`--awac_adv_norm_tau`) — it destroys attribution.

---

## 7. After v3a — the pipeline (see RL_IMPROV_16_EXPERIMENTS.md)

1. During v3a, `STUCK_CHECKPOINT_DIR` fills a mesh-fingerprinted stuck pool. After
   ~12h, screen it: `bash launch_screen_stuck.sh` (escapability + restore-fidelity
   filter) → screened pool.
2. **v3b** = v3a flags + `--checkpoint_dir <screened pool> --rl_start_mode sofa_restore`
   + a `--restore_prob 0.3` addition (NOTE: not yet implemented — the wrapper currently
   restores on every reset; add the Bernoulli gate first). Starts episodes *in* screened
   stuck states so retraction gets on-policy credit.
3. **v4** = obs surgery (cache-breaking; fresh harvest): prune ~21 dead/dup dims, add
   catheter-path + windup + stuck-duration, de-crutch `ep_step` from the policy prefix,
   optionally `computeConstraintForces=True` for real force labels.

**Standing rule:** reward/observation/terminal changes are frozen without explicit
human approval; never alter `is_on_correct_path()`. The one approval-gated item in the
plan is E7 (a retract-on-escape reward term), held as a fallback only if E1+E3+E4 leave
micro-recovery fading.

---

## 8. Comparability

Identical eval protocol as v2 (98 fixed held-out seeds, ostium starts,
`--eval_after_pretrain` baseline). v2's per-eval numbers at matched explore steps are the
control curve: pretrain 6.1% → eval1 30.6% → eval2 49.0%. Keep evals on ostium starts
even in v3b (restore only during explore) so the eval numbers stay comparable across all
runs.
