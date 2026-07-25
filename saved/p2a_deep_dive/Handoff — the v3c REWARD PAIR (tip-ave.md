# Handoff — the v3c REWARD PAIR (tip-average progress + catheter-slack channel)

**Audience:** machine 1, implementing the same reward fix on its checkout. Scope = the
reward pair ONLY (the later trainer levers — contact-gated anti-rail, Q-target floor,
per-mode adv-norm — are separate and not covered here, except one warning in §7).
**Origin:** machine 2, branch `rl_improv_16_resume`, developed 2026-07-21..22 on top of the
v3a code state. Everything below was unit-tested in-container (32-test suite,
`tests/test_v3c_reward.py`), adversarially reviewed (2 BLOCKERs found and fixed — both fixes
are REQUIRED, see §3.4 and §6), and validated in two training runs (§7).

---

## 1. Why — the measured diagnosis this fixes

Full forensics: `FORENSIC_RCCA_V3A.md`, `FORENSIC_V3A_PATH_FORWARD.md`. The load-bearing
measured facts (v3a family, 98-seed held-out eval):

1. **Success was a path-length cliff.** 49 always-solved seeds all have planned-path length
   ≤ 170 mm; 46 never-solved all ≥ 170 mm past a fixed r=2.0 mm choke at arc ~158 mm. Zero
   overlap across 7 evals.
2. **Deep failures never deploy the guidewire.** Median final gw insertion in eval failures
   0.1–1.3 mm; the catheter alone cannot pass the choke (it CAN reach it), so the policy
   shoves the catheter to its hard ~898 mm insertion cap — ~700 mm of slack coils/knots at
   the RVA ostium shelf (69 % of stuck-pool captures have far-arc self-contact < 2 mm;
   device self-collision is OFF in SOFA so the beam passes through itself).
3. **The reward built this behavior.** Three reward asymmetries, all verified in code and in
   ~640k-step log mining:
   - The progress term paid **only the frontier (leading) tip**: a parked guidewire cost
     nothing; catheter-only telescoping was a *full-pay* strategy.
   - The anti-buckle shaping (`buckle_reward.py`) priced **guidewire slack only** —
     gw-lead through resistance was *taxed*, gw-retract was *paid* — while **catheter slack
     appeared nowhere**: 700 mm coils formed at a measured net ≈ −0.00003/step. Loops were
     reward-*free*, and the −3 timeout penalty is one distant spike, behavior-invariant.
   - **The decisive tie:** the only eval episodes ever to pass the choke pushed the
     guidewire at a specific divergence point; failures retracted it at the same point —
     and the instantaneous rewards of the two actions were **identical** (+0.0305). AWAC
     (advantage-weighted BC, no Q-ascent) cannot propagate the +3 terminal back through a
     reward tie; the winning action was never preferred.
4. Physics check (measured): the catheter advances **better** with the guidewire inside
   (2.4–3.1 mm/step with gw > 50 mm inserted vs 1.25–1.46 without) — gw-retract was never
   mechanically useful; it was purely reward-selected.

**Design goal of the pair:** *pay the coordinated two-device gait, charge the slack* —
break the divergence tie toward deploying the wire, price the coil the tip-based terms
cannot see, keep every term a pure potential (telescoping, unfarmable).

---

## 2. The two changes, semantically

### A. Tip-average progress (`--progress_tip_mode avg`, `--avg_gw_weight 0.5`)

Replace the progress quantity: instead of Δ(frontier-tip arc), pay
`progress_factor × Δ( w·s_gw + (1−w)·s_cath )` — the weighted-average planned-path arc of
**both** device tips (w = 0.5 default).

Consequences (all verified by unit test + log reconstruction):
- A parked guidewire **halves** the pay rate → catheter-only banking stops being full-pay.
- Advancing the **trailing** device is paid (+w·Δ/mm) → at the measured divergence point,
  gw-push now out-earns gw-retract (~79 mm of paid progress available) — the tie is broken
  by a smooth, always-on gradient, not an event bonus.
- The telescoping gait (cath forward + gw retract) nets ≈ 0 even during the approach.
- Still a pure potential: `s_eff` is a state function ⇒ round trips net exactly 0.
- Empirical divergence from the legacy signal on identical trajectories: 99.6 % of steps
  differ; avg pays ~31 % of frontier on forward steps, ~15 % when the gw is parked.

Protocol details that matter (asymmetric by design — see §3.2 for why each is safe):
- The **leading** device keeps the exact legacy frontier projection incl. the on/off-path
  state machine. The **trailing** tip is located on the combined device polyline at its own
  inserted length, projected (windowed, 30 mm anchor), and clamped to `[0, s_frontier]`.
- Off-path: the whole avg branch suspends (legacy Δoff-arc penalty runs, frontier-based)
  and the avg tracker **freezes** (see the BLOCKER fix §3.4). Coaxial geometry means a
  trailing tip can only be off-course via a body loop — which channel B prices.
- Loop geometry measured safe: feeding the catheter *into* a prolapse loop makes its
  projection **retreat** (−4 mm per 10 mm fed in the synthetic test) — loop-deepening is
  charged, not paid, without any explicit loop detector in this term.

### B. Catheter-slack potential channel (`--cath_slack_coef 0.5`)

New shaping channel, identical construction rules to the existing gw buckle potential:
`reward += coef × (φ_c(t) − φ_c(t−1))` with
`cath_slack = inserted_cath − arc(catheter tip projected on the planned path)`.

- `cath_slack` ≈ 0 in clean over-wire tracking; > 50 mm only when the shaft stores
  redundant length. **Validated as a knot detector: precision 0.97 / recall 0.96 at
  > 50 mm** against far-arc self-contact geometry on 423 stuck-pool states.
- Dead-band 15 mm (normal catheter lag + projection jitter), cap 150 mm, φ_c ∈ [−1, 0].
  Potential/delta form ⇒ un-coiling refunds the penalty (recovery incentive), a restore
  into a coiled state re-baselines (delta 0 on step 1, positive on un-coil), closed cycles
  net exactly 0 (unfarmable).
- Per-mm price at coef 0.5: 0.00333/mm — deliberately below the 0.01/mm progress factor
  below the cap.
- **Use the CAPPED form.** Machine 2 later tried an uncapped linear tail (v3c2): it was
  verified **behaviorally irrelevant** (failure slack 710 vs 708 mm, identical q1
  trajectories) — don't bother; the capped run peaked higher. See §7 caution regardless.

**Separate coefficient** from `buckle_reward_coef` so the two channels attribute
independently in reward accounting.

---

## 3. Code changes, file by file

Repo-relative paths as on machine 2 (note `training _scripts` contains a space).

### 3.1 `eve/eve/util/polyline.py` — new helper

Locates a device tip on the DISTAL-FIRST combined `tracking3d` polyline at a given inserted
length, dropping the undeployed zero-length node pile at the proximal end:

```python
def point_at_inserted_length(
    device_polyline: np.ndarray, inserted_mm: float
) -> np.ndarray:
    """3-D point of a device tip that sits ``inserted_mm`` of arclength past
    the insertion end of a DISTAL-FIRST device polyline (tracking3d order:
    index 0 = leading tip, last index = insertion point).

    Undeployed nodes pile up as zero-length segments at the proximal end;
    they are dropped before measuring, so the walk runs over deployed
    geometry only. ``inserted_mm`` is clamped to [0, deployed length]: 0
    returns the insertion point, values past the deployed length return the
    leading tip. Returns None when fewer than 2 deployed points exist.
    """
    pts = np.asarray(device_polyline, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return None
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    keep = seg > 1e-9
    if not keep.any():
        return None
    # Keep point i when either adjacent segment is non-degenerate.
    keep_pts = np.concatenate([[keep[0]], keep[:-1] | keep[1:], [keep[-1]]]) \
        if len(seg) > 1 else np.array([True, True])
    pts = pts[keep_pts]
    if len(pts) < 2:
        return None
    # Arc from the PROXIMAL (insertion) end: reverse, then walk forward.
    rev = pts[::-1]
    cum = compute_cumulative_arclength(rev)
    target = float(np.clip(inserted_mm, 0.0, cum[-1]))
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    if idx >= len(rev) - 1:
        return rev[-1].copy()
    seg_len = cum[idx + 1] - cum[idx]
    t = 0.0 if seg_len <= 1e-9 else (target - cum[idx]) / seg_len
    return rev[idx] + t * (rev[idx + 1] - rev[idx])
```

Empirical note: `tracking3d[0]` is the frontier tip of the **combined** instrument; the
polyline arclength ≈ the *leading* device's inserted length (verified on stuck-pool states:
arclen matches cath_ins when the catheter leads, gw_ins when the wire leads).

### 3.2 `eve/eve/reward/arclengthprogress.py` — avg mode

Constructor gains `tip_mode: str = "frontier"` and `avg_gw_weight: float = 0.5`
(**defaults = byte-identical legacy**; validate `tip_mode ∈ {"frontier","avg"}`; store both
as same-named attributes — the ConfigHandler getattrs every `__init__` param). New internal
trackers `_prev_s_eff = None`, `_trail_prev_s = None` (underscore ⇒ not serialized).

**reset():** in avg mode, ALWAYS compute `self._polyline` / `self._cumlen` from
`pathfinder.path_points_vessel_cs` (even when a path_context serves the frontier
projection — avg mode projects the trailing tip itself), reset `_trail_prev_s = None`, and
compute the baseline **at reset**: `self._prev_s_eff = self._effective_avg_arc(result.s)`
(so step-1 motion is priced; a restore into a retracted state re-baselines there and
recovery nets positive).

**step(), inside the existing `if on_path:` branch** — after the legacy
`r_progress = progress_factor * delta_s` is computed, override it in avg mode:

```python
if self.tip_mode == "avg":
    s_eff = self._effective_avg_arc(result.s)
    if s_eff is not None:
        if self._prev_s_eff is not None:
            r_progress = self.progress_factor * (s_eff - self._prev_s_eff)
        else:
            r_progress = 0.0   # resync after a geometry-failure gap
        self._prev_s_eff = s_eff
    else:
        self._prev_s_eff = None   # fall back to legacy frontier delta this step
```

Any geometry failure degrades to the legacy frontier delta for that step (never a dropped
reward). The off-path branch and the lateral penalty are **unchanged** (frontier-based —
intended asymmetry; the trailing tip's own off-course case is a body loop, priced by
channel B and by the projection-retreat behavior above).

The trailing-tip locator (verbatim; note the windowed projection + the clamp):

```python
    def _effective_avg_arc(self, s_frontier: float):
        """Weighted-average planned-path arc of the two device tips.

        The LEADING device (larger inserted length) keeps the frontier
        projection ``s_frontier`` — identical to the legacy signal. The
        TRAILING tip is located on the combined device polyline at its own
        inserted length, projected onto the planned path, and clamped to
        [0, s_frontier] (a trailing tip cannot legitimately grade ahead of
        the frontier; projection noise in tight bends must not pay).
        Returns None when the geometry is unavailable (missing devices,
        degenerate polyline) — callers fall back to the frontier delta.
        """
        try:
            inserted = self.intervention.device_lengths_inserted
            if inserted is None or len(inserted) < 2:
                return None
            ins_gw = float(inserted[0])
            ins_cath = float(inserted[1])
            fluoro = self.intervention.fluoroscopy
            track = np.asarray(fluoro.tracking3d, dtype=float)
            trailing_ins = min(ins_gw, ins_cath)
            tip_3d = point_at_inserted_length(track, trailing_ins)
            if tip_3d is None:
                return None
            tip_vessel = tracking3d_to_vessel_cs(
                tip_3d, fluoro.image_rot_zx, fluoro.image_center
            )
            if len(self._polyline) < 2:
                return None
            proj = project_onto_polyline(
                tip_vessel,
                self._polyline,
                self._cumlen,
                prev_s=self._trail_prev_s,
                window_mm=30.0,
            )
            self._trail_prev_s = float(proj.s)
            s_trailing = float(np.clip(proj.s, 0.0, s_frontier))
            w = self.avg_gw_weight
            if ins_gw <= ins_cath:
                s_gw, s_cath = s_trailing, float(s_frontier)
            else:
                s_gw, s_cath = float(s_frontier), s_trailing
            return w * s_gw + (1.0 - w) * s_cath
        except Exception:
            return None
```

The windowed projection (`prev_s` + `window_mm=30`) reuses the existing hairpin guard in
`project_onto_polyline` — a bare full scan limb-flips at the siphon fold and injects
fold-arc-sized jumps; the anchor + the built-in `fallback_dist_mm` escape prevent that.

### 3.3 THE BLOCKER FIX you must not skip — freeze, don't rebaseline, off-path

At the end of `step()` (after the unconditional `self._prev_d_rem = d_rem_curr`) there must
be **NO rebaseline of `_prev_s_eff` during off-path steps** — the tracker simply freezes.
Machine 2's first implementation rebaselined it off-path; adversarial review proved with a
numeric counterexample against the real class that this creates a **farmable pump**:
retract the catheter while the frontier is off-path-classified (free — the off-arc channel
tracks only the frontier), re-advance it on-path (paid) → **+0.125 per 13-step closed
cycle, ~0.005/mm of catheter cycled, ceiling ~+5 per 600-step episode — more than a real
success** — the same exploit class as the removed RL_IMPROV_15 2× doubling. Frozen, the
rejoin step pays `s_eff(rejoin) − s_eff(last on-path)`, netting the excursion's trailing
motion exactly once; the counterexample then nets 0.0000 (regression test in the suite,
§6). If your port pays anything on that cycle, stop and fix before training.

### 3.4 `training _scripts/util/buckle_reward.py` — the new potential

```python
CATH_SLACK_DEADBAND_MM = 15.0
CATH_SLACK_CAP_MM = 150.0

def cath_slack_potential(cath_slack_mm: float) -> float:
    """Catheter-slack potential phi_c(cath_slack) in [-1, 0].

    Same construction rules as buckle_potential: monotonically
    non-increasing, flat inside the dead-band and beyond the cap, caps on
    the INPUT (never the delta) so loop-neutrality holds — forming a coil
    and pulling it back out nets exactly zero, and an episode restored
    INTO a coiled state that un-coils nets positive.
    """
    ex = min(
        max(float(cath_slack_mm) - CATH_SLACK_DEADBAND_MM, 0.0),
        CATH_SLACK_CAP_MM,
    )
    return -(ex / CATH_SLACK_CAP_MM)
```

(That is the **capped run-1 form — use this one**; see §2B/§7 on the uncapped variant.)

### 3.5 `training _scripts/util/env5.py` — wiring

- Guarded import: `from util.buckle_reward import buckle_potential, cath_slack_potential`
  (both fallback paths), `cath_slack_potential = None` on ImportError.
- `BenchEnv5.__init__` gains `cath_slack_coef=0.0`, `progress_tip_mode="frontier"`,
  `avg_gw_weight=0.5` (defaults = byte-identical). Store as attributes; raise ImportError
  if `cath_slack_coef != 0` and the potential is unimportable. New trackers:
  `_cath_phi_prev = None`, `_cath_slack_prev_raw = 0.0`, `_last_cath_slack_mm = 0.0`,
  `_cath_proj_prev_s = None` — ALL reset in the episode-reset block next to the existing
  `_buckle_phi_prev` reset (the None re-baseline is what makes restores-into-coiled-states
  net positive on recovery).
- Pass `tip_mode=self.progress_tip_mode, avg_gw_weight=self.avg_gw_weight` into the
  `eve.reward.ArcLengthProgress(...)` construction.
- In `step()`, immediately after the existing buckle-shaping block, the same delta pattern:

```python
if self.cath_slack_coef != 0.0:
    try:
        phi_c = cath_slack_potential(self._compute_cath_slack_mm())
        if self._cath_phi_prev is not None:
            reward += self.cath_slack_coef * (phi_c - self._cath_phi_prev)
        self._cath_phi_prev = phi_c
    except Exception as e:
        self._step_logger.warning(f"cath-slack reward failed: {e}")
```

- `_compute_cath_slack_mm()` (next to `_compute_buckle_potential`): computes
  `inserted_cath − proj(cath tip).s` with `point_at_inserted_length` + a **windowed**
  `project_onto_polyline` (`prev_s=self._cath_proj_prev_s, window_mm=30.0`, anchor updated
  each call), `pathfinder.path_points_vessel_cs` as the polyline. Fall back to the PREVIOUS
  raw value on any accessor failure (delta = 0 through the φ indirection — same failure
  semantics as the buckle channel). `inserted_cath ≤ 1e-6` short-circuits to 0.0.
  Per-step cost measured ~0.3 ms vs ~1.5 s median env step — no caching needed.
- STEP log line: append `f" | cath_slack={getattr(self, '_last_cath_slack_mm', 0.0):+.1f}"`
  right after the `buckle_phi=` fragment (name-keyed parsers are unaffected; +~2.5 % log
  volume; gives the monitors/forensics the signal for free).

### 3.6 Reward-version guard — REQUIRED, this is what makes old seeds fail fast

`eve_rl/eve_rl/util/experience_cache.py`:
- Three new env-var stamps mirroring `EVE_RL_BUCKLE_COEF`: `EVE_RL_CATH_SLACK_COEF`,
  `EVE_RL_PROGRESS_TIP_MODE`, `EVE_RL_AVG_GW_WEIGHT`, with `_current_*()` readers
  (defaults 0.0 / `"frontier"` / 0.5 ⇒ absent = legacy, pre-v3c caches stay valid for
  legacy-reward runs).
- `save_episodes_npz` stamps `meta_cath_slack_coef` (float64), `meta_progress_tip_mode`
  (np.str_), `meta_avg_gw_weight` (float64) next to `meta_buckle_coef`.
- New accessor `cache_reward_version(path) -> dict` returning all four fields with legacy
  defaults for absent keys.

`training _scripts/DualDeviceNav_train.py`:
- After arg parsing: resolve `_cath_slack_coef`, `_tip_mode`, `_avg_gw_weight`; put them in
  `env_kwargs` when non-legacy (with `[v3c] ...` startup prints so the monitor can confirm
  the flags took); export all three env vars **before any worker spawns** (workers inherit
  os.environ on spawn — their rolling flushes stamp correctly).
- **Both** cache-load guard sites (heatup + heuristic) compare the full 4-field tuple via
  `cache_reward_version` and raise on any mismatch (string equality for tip_mode, 1e-9 for
  floats). Loading the old frontier-reward seed into an avg-mode run then fails fast with
  the mismatch list — that is the intended behavior; **a fresh harvest is required** (§5).
- Argparse: `--cath_slack_coef` (float, 0.0), `--progress_tip_mode`
  (choices `frontier|avg`, default `frontier`), `--avg_gw_weight` (float, 0.5). ⚠ argparse
  help strings must escape every literal `%` as `%%` or `--help` crashes with
  `TypeError: %o format` (bit machine 2 twice).
- Optional but recommended (machine 2 did it): stamp the same 4 fields into the
  incremental-buffer `replay_state.npz` on save and guard `--resume` against a mismatch —
  otherwise a careless resume can silently mix two reward MDPs in one buffer.
- Guard `--multi_target_heatup` + the pair with a hard error (MultiTargetEnv5 does not
  receive the new kwargs; a multi-target harvest would be scored legacy while its stamps
  claim otherwise).

### 3.7 Launchers

Training launcher: add the three flags (everything else unchanged from your current run):

```
    --cath_slack_coef 0.5 \
    --progress_tip_mode avg \
    --avg_gw_weight 0.5 \
```

Harvest launcher: same three flags added to your existing harvest config (machine 2:
480 episodes, same procedural seeds) with a NEW output path (e.g.
`rcca_proc_heatup_v3c/seed.npz`) — plus a preflight `[ -f seed ] || exit 1` in the training
launcher (a missing `--heatup_cache_file` silently degrades to a random-heatup run).

---

## 4. Why the old seed cannot be reused (fresh harvest is mandatory)

The stored seed's rewards were computed under the frontier reward. Recomputing them offline
is **not possible**: the catheter-tip arc is not recoverable from the stored 121-dim obs
(only a chord-biased approximation via the privileged 3-D tip-offset dims — biased exactly
in the contact/tortuosity states that matter). Replays need per-episode RNG seeds that were
never stored. So: harvest a fresh seed under the new reward (~3 h, 480 episodes on machine
2's box), let the guard enforce the match. The new seed is statistically equivalent to the
old (random actions, same meshes): machine 2's came out 480 eps / 282,657 transitions,
26/480 grader successes (old: 282,310 / 27).

---

## 5. Verification you should replicate before training

`tests/test_v3c_reward.py` (runs inside the training image with the run's mounts; numpy
only). The critical cases:
- `cath_slack_potential`: bounds, dead-band, cap, monotone, **closed-cycle sums to 0**.
- `point_at_inserted_length`: insertion point at 0, midpoint, undeployed-pile ignored.
- Avg mode on synthetic geometry: cath-solo advance pays HALF; trailing-gw advance pays
  HALF; **round trip nets zero**; frontier mode pays the full legacy delta (byte-identical).
- **The pump regression (§3.3): on→off(retract trailing)→on→re-advance closed cycle must
  net exactly 0** — with a scripted path_context driving the on/off flags. If this test
  doesn't exist in your port, you have not ported the fix.
- Cache stamp round-trip via env vars.
- Argparse `--help` executes without traceback; launcher dry-run (docker shim) shows one
  invocation with all three flags.

Sanity check worth repeating on your first harvest logs: reconstruct both rewards on the
same trajectories (frontier from Δproj_s; avg from the logged reward minus the other
channels) — expect ~99 % of steps to differ and avg/frontier ≈ 0.3 on forward steps.

---

## 6. Results on machine 2 (what to expect)

| run | baseline (pretrain-only) | eval1 (~252k explore) | notes |
|---|---|---|---|
| v3a (control, frontier reward) | 50.0 % | 50.0 % | all-time SUSTAINED ~50–52 % over 15 evals |
| v3c run 1 (the pair, capped) | 29.6 % | **62.2 %** | host reboot killed it at explore 415k |
| v3c2 (pair, uncapped tail) | 35.7 % | 58.2 % | then declined — see §7 |

- The **baseline dip is expected** (25–36 % across three runs): the pair de-incentivizes the
  catheter-shove that solved shallow seeds, before online learning builds the replacement.
  Do not judge the pair on the baseline.
- Mechanism confirmation at baseline already: eval-failure median gw insertion 23 mm
  (was 0.1–13.5), catheter coil gone (111 mm vs 898 mm).
- At eval1: successes deploy 76–95 mm of guidewire; failures reach 177–190 mm median depth
  (past the old 158 mm wall); explore success 53–63 % vs v3a's flat ~45 %.

## 7. KNOWN RISK — read before running long

Both v3c runs **peaked at eval1 and then declined** (v3c2: 58→55→51→48→41 %…). The verified
cause (machine-2 forensic, self-checked against CSVs/chunks/step logs) is **NOT the reward
pair's shaping math** and NOT the slack cap — it is an **endogenous critic/TD-bootstrap
instability** that the pair's harder task exposes: the policy's contact-state mean rails →
its sampled next-actions go OOD → min-double-Q extrapolates negatively → the plain Bellman
target follows (q1 → −90 while honest returns bottom at ~−7) → advantage ranking degrades.
The pair changes *what* fails (v3a failed by actor mean-rail with a bounded critic), not
*whether* something fails. Machine 2 is currently testing trainer-side guards (Bellman-
target floor clip + contact-gated anti-rail) — outside this doc's scope, but if you run the
pair long, **watch `q1_mean` (losses CSV col 12): if it descends past ~−2.5 and keeps
going, the critic is diverging** and the run's peak is behind it. Short runs to eval1–2
reproduce the headline gain regardless.

---

## 8. Machine-1 practicalities

- **Mount every touched file** into the container (the image's baked copies are stale):
  `polyline.py`, `arclengthprogress.py`, `buckle_reward.py`, `env5.py`,
  `experience_cache.py`, `DualDeviceNav_train.py` (+ `pervanillastep.py` if you take the
  resume-guard stamps). All were already in the v2/v3a mount lists except none — verify
  each `-v` source exists; a missing source makes docker create an empty dir that shadows
  the module.
- The CRLF trap applies to any new/edited launcher: `core.autocrlf=true` materializes `.sh`
  as CRLF and bash then splits every `\`-continued line (single `docker run` becomes
  garbage commands; `bash -n` does NOT catch it). Keep launchers LF (`.gitattributes` with
  `*.sh text eol=lf`), and never rewrite mount paths with sed patterns containing `\n`.
- The guard makes misuse loud: old seed + new flags → immediate `reward version mismatch`
  listing the offending fields. That's working as intended.
