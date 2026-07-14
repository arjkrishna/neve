# AWAC Stability Evolution — history, mechanism, and the v2b one-more-chance config

**Question this answers** (2026-07-14): before the current experiments we had ~70%
accuracy at LCCA-family training. What was the AWAC formulation *before* w.r.t.
learning stability / entropy collapse / alpha / rails, why did we change what we
changed, what is happening *now* (v2), and — **without any Tier-A/B/C changes
(no E1b/E2/E3, no obs or reward edits)** — can pure-AWAC improvements make the
current configuration stable enough to reach eval3 without collapsing?

Sources: pre-review code at commit `c810959` (verified by direct inspection, cited
below), [RL_IMPROV_10_CHANGES.md](RL_IMPROV_10_CHANGES.md) §13–15/19,
[HANDOFF_LCCA_V2_RELAUNCH.md](HANDOFF_LCCA_V2_RELAUNCH.md), the deep review
(Part A of [RL_IMPROV_15_CHANGES.md](RL_IMPROV_15_CHANGES.md)), the v1 collapse
forensic + v2 monitor log (`saved/monitor_rcca_procedural.md`), and the v2 live
investigation (`saved/v2_micro_investigation/`).

---

## 1. The AWAC formulation BEFORE (the ~70%-era stack), and what was wrong with it

The policy update at `c810959` (verified in the old `sac.py`/`gaussianpolicy.py`):

```
w        = exp( (Q(s,a_buf) − Q(s,a_π)) / λ ).clamp(max=20),  λ = 3.0
loss     = −( w · log π(a_buf|s) )  +  α · log π(a_π|s)        (entropy term added late in this era)
log_std  = torch.clamp(log_std, −20, +2)                        # HARD clamp, defaults (−20, +2)
log_prob = log_prob.sum(−1).clamp(min=−20)                      # HARD BC floor
noise    : action += np.random.normal(0, σ_expl)                # ONE scalar, all 4 dims, unclipped
log_alpha: UNBOUNDED (no clamp anywhere in _update_alpha)
```

Four structural pathologies, one per axis the question names:

| Axis | What the old formulation did | Consequence (documented) |
|---|---|---|
| **Learning stability** | λ=3 vs small advantages → `w ≈ 1` for nearly all samples → the "advantage-weighted" update degenerates toward **uniform behavior cloning of whatever is in the buffer** — including the policy's own increasingly-extreme successes (a positive feedback loop with no counterweight). The hard `log_prob` floor (−20) additionally **zeroed the BC gradient on exactly the far/high-advantage demos** AWAC most needs (deep-review B1). | RCCA-v3: per-state success flat across 3 evals (RL_IMPROV_10 §14) — critic stable, policy plateaued; the learning signal was too undifferentiated to move it. |
| **Entropy collapse** | AWAC originally had **no entropy term at all**; the α·log π term was bolted on late in this era. Even with it, `entropy_proxy = Gaussian(σ) + Σlog(1−a²)` — the tanh-Jacobian half is **unbounded in the mean**, so entropy collapses via mean-rail even when σ is floored. | RL_IMPROV_10 §15: entropy −2.3 → −10.2 over 650k updates *despite* `log_std_min=−2`; clamp_fraction 0.06 → **0.53**. The floor protected σ, not the mean. |
| **Alpha** | `log_alpha` **unbounded** (verified: old `_update_alpha` has no clamp). With entropy persistently off-target the integrator winds up without limit in either direction. | Deep-review F1 (high risk). Realized later as procedural-v1's decay-to-−10 → 165k dead updates → whipsaw to 0.45 → **mean-crush freeze** (the eval 3.1% collapse). |
| **Rails** | Hard `torch.clamp` on log_std has **zero gradient outside the band** — a railed std head is a one-way ratchet (deep-review C1). Defaults (−20, +2) allowed *both* failure modes: σ→2e-9 collapse (Plan v10 era) and σ→2.0 **ceiling explosion** (LCCA v1: actions railed from sheer sampling noise → 7% soft-collapse, HANDOFF §1). Scalar unclipped noise (H4) correlated all 4 dims and pushed stored actions outside the tanh domain. | LCCA v1's log_std ceiling explosion; RCCA-v3's clamp_fraction 0.53 death-spiral; both documented before any Gen-4 work. |

**The through-line:** every generation's instability is the *same* variable —
**saturation of the squashed Gaussian** — escaping through whichever bound was
left loose. The BC term is an engine that clones whatever succeeded (including
the policy's own saturated actions); entropy/penalty terms are the brakes.
Before the review, three of the four brakes were missing or broken.

## 2. Why we made each change (the deep-review / RL_IMPROV_15 package)

| Change | Replaces | Why (which pathology above) |
|---|---|---|
| Soft tanh-rescale log_std band (−2, 0) | hard clamp (−20, +2) | C1: restores gradient at the bounds (rails become recoverable); ceiling 0 structurally prevents the LCCA-v1 σ-explosion; floor −2 prevents the σ-collapse. |
| `log_alpha` clamp rails (−5, −2.3) | unbounded | F1 + the v1 forensic: floor keeps the entropy term *alive* (no 165k dead-α decay), ceiling caps it *below* the mean-crush zone (v1 crushed at α≈0.28–0.45; healthy learning observed at ≤0.1). |
| Leaky log-prob floor (5% grad to −30) | hard −20 floor | B1: the far demos keep a BC gradient — advantage-weighting can actually act on them. |
| λ 3.0 → 1.0 | λ=3 | widen `w` spread so the update is less BC-degenerate (v2 measured: still only [0.72,1.25] — hence Tier-A E1b later, but that is out of scope here). |
| `action_mean_penalty` 0.005 (`+c·|atanh(tanh μ)|`) | nothing | the ONLY term that acts directly on the unbounded tanh-Jacobian half of the entropy — i.e., on mean-saturation itself. |
| Per-dim clipped exploration noise | scalar unclipped | H4: decorrelates exploration; keeps stored actions inside the tanh domain (protects the BC log-prob). |
| Clean-lane rail filter (`EVE_CLEAN_RAIL_MAX=0.15`) | nothing | breaks the self-cloning loop at the *data* level: bang-bang successes stay out of the amplified BC lane. |
| `target_entropy` +1.0 (native float) | −n_actions=−4 | the −4 default left the α controller a dead zone the policy railed straight through; +1.0 keeps it engaged in the healthy band. |

Validation that the package addressed the right things: v2's eval trajectory
**6.1% → 30.6% → 49.0%** (vs v1's peak 13.3% then 3.1% freeze), with the α
lift-off executing smoothly and zero rail-filter rejections.

## 3. What is happening NOW (v2's late-run regime, the eval3 regression)

From ~u165k the system entered a **deterministic-aggressive regime**:

- α reached and pinned at its **ceiling** (0.1003) — the controller at full
  authorized power; entropy_proxy then fell **through** target: 1.0 → 0 →
  **−2.2** (min −2.97). Since σ is at its cap the whole time, ALL of that fall
  is **mean-saturation** (tanh-Jacobian), not variance loss.
- clamp_fraction 2% → **13.8%**; grad-clip binding; q1_mean −1.0 → −2.3.
- Deterministic probe stayed healthy (0.15–0.21 = 4.7–6.7× the pretrain
  attractor) and explore success kept climbing to ~51% — this is **not** a
  freeze and not v1's collapse.
- But **eval3 regressed 49.0% → 30.6%** (speed still 8.9mm/s), with explore
  softening 51→47% in the same window: the BC engine cloning its own
  ever-more-aggressive successes finally outran the brakes
  (α capped at 0.1 + penalty at 0.005 were **too weak at eval3 scale**), and
  the aggressive push-forward style hit the stuck-grinder ceiling on hard
  held-out meshes (micro-recovery had faded — see
  `saved/v2_micro_investigation/`).

So the *current* instability is the mildest expression yet of the same
through-line: **mean-saturation growth under an under-powered brake** — no
longer a σ rail (capped), no longer an α runaway (capped), just the engine
(BC self-cloning) slightly stronger than the brakes over 300k+ updates.

## 4. Can pure-AWAC changes stabilize this configuration to eval3? — YES (with honest scope)

The levers that act on mean-saturation *within the AWAC loss*, using **flags
that already exist in the v2 code** (no new code, no Tier-A Es):

### v2b = v2 + exactly two knob changes

| Knob | v2 | v2b | Mechanism |
|---|---|---|---|
| `--action_mean_penalty` | 0.005 | **0.02** | The direct brake on the thing that actually destabilized: pre-tanh mean growth. 4× the restoring force, constant-strength (unlike the α term it does not saturate at a ceiling), still ~100× below dominating the BC loss at healthy means (at \|tanh μ\|=0.25 it contributes ~0.005 to a ~1.5 loss). |
| `--log_alpha_max` | −2.3 (α≤0.100) | **−2.0 (α≤0.135)** | +35% entropy-brake authority for the late run, still 2–3× below v1's mean-crush zone (0.28–0.45) — and unlike v1, the BC term is now live (λ=1) anchoring the mean to buffer actions, the mean penalty resists over-shrink from the other side, and the freeze probe watches the basin. |

Everything else **identical to v2**: λ=1.0, target_entropy 1.0, log_std (−2,0),
alpha floor −5, rail filter 0.15, buckle 0.5, aux 0.05 @ 2,3,5,6 (raw), PER +
balanced 0.3, seed + 10k pretrain, eval_after_pretrain. **None of the Tier-A
flags** (`awac_adv_norm_tau`, `aux_label_znorm`, `stuck_fraction` all absent =
those code paths byte-identical to v2). The run also inherits the two
deadlock guards (trainer-result deadline + progress watchdog, default-on) so
eval3 cannot kill it the way it killed v1/v2.

### Why this is the right-sized bet

- The failure signature to prevent is precisely measured: entropy through 0
  into −2 with clamp 14% by explore ~750k. Both knobs push directly and only
  on that axis.
- Both knobs preserve the v2 attribution chain: if v2b holds entropy ≥ ~0 and
  clamp ≤ ~8–10% through eval3 **and** eval3 ≥ ~45%, the regression is proven
  to be brake-strength, not something deeper.
- Risk is bounded and monitored: the failure mode of *over*-braking is drift
  toward the pretrain attractor — the deterministic freeze probe (ratio to
  0.032 baseline) catches that within one snapshot interval.

### Gates for v2b (stability-to-eval3 is the objective)

| Gate | Healthy | Abort/peel |
|---|---|---|
| entropy_proxy late-run | ≥ −0.5 sustained (v2: −2.2) | < −1.5 sustained → penalty under-sized; raise to 0.03 on relaunch |
| clamp_fraction | ≤ 10% (v2: 13.8%) | sustained > 12% → same |
| freeze probe | ratio ≥ 2× baseline and not falling on 2 consecutive passes | < 1.25× → α ceiling back to −2.3 (over-braked) |
| eval2 | ≥ 45% (v2: 49%) | < 40% → the knobs cost too much learning speed; revert |
| **eval3 (the target)** | **≥ 40–45% with speed ≥ 3mm/s** (v2: 30.6%) | — |

### What v2b will NOT do (honesty)

The stuck-grinder ceiling is a **capability** gap (retract-when-stuck faded;
credit assignment too uniform to teach it), not a stability gap. Two brake
knobs won't add that skill. Realistic expectation: v2b **holds ~45–50%
through eval3 instead of regressing to 30%** — i.e., it stabilizes the
plateau. Breaking past ~50% still needs the Tier-A/B recovery levers
(machine 2). If v2b achieves its gates, it also becomes the *better baseline*
those levers get measured against.

---

## 5. Run artifacts

- Launcher: `launch_rcca_procedural_v2b.sh` (v2 launcher + the two flags; no
  Tier-A flags; deadlock-guard envs explicit).
- Monitoring: same cadence/checks as v2 (`scratchpad monitor pass`), with the
  gate table above substituted for v2's watch-items.
- Control curves for comparison at matched explore: v2 = 6.1 / 30.6 / 49.0 /
  30.6; v1 = 6.1-equiv / 8.2 / 13.3 / 3.1.
