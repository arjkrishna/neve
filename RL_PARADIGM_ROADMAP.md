# RL Paradigm Roadmap — beyond AWAC fine-tuning (literature-grounded, 2026-07-14)

**Question answered:** we keep fine-tuning AWAC's brakes (the 0.005-vs-0.02
action-mean-penalty knife-edge, alpha rails, rail filters). Are there better
paradigms for our exact problem — follow a planned path by pushing/pulling
flexible coaxial devices, with buckling/stall recovery — given our constraints
(SOFA ≈1–2M steps/day, deterministic 7.5Hz deployable-obs deployment, mesh
generalization, existing assets: planned path, scripted heuristic, 280k demos,
privileged sim state, 16 workers, PER lanes)?

**Method:** three parallel literature agents with live web access (domain SOTA
/ algorithm alternatives / hierarchical-structured methods), ~30 sources
fetched and verified. Full reports preserved in `saved/rl_paradigm_research/`.

---

## 0. The headline findings (three independent agents converged)

1. **We are AT the published domain frontier.** The stEVE `DualDeviceNav`
   baseline (the exact framework we run; Karstensen et al.) is **40/100**;
   the best 2026 result (SAC+GAIL+expert-in-the-loop, arXiv:2602.20216) is
   **59%**; SplineFormer real-robot single-branch is 50%. Our v2 eval2 =
   **49%** on harder (procedural, siphon-depth, dual-device) conditions. No
   published system demonstrates high-success siphon-level autonomy with
   explicit slack/buckling recovery. **The stuck-grinder ceiling is the
   field's frontier, not our local bug** — and framing slack/buckling as the
   core learning problem puts us *ahead* of most of the endovascular-RL
   literature.
2. **The policy-constraint family (AWAC/TD3+BC/CRR/SPOT) is the wrong class
   for our regime — per the literature, not just our forensics.** The "Three
   Regimes of Offline-to-Online RL" study (arXiv:2510.01460) shows
   constraint-centric methods systematically underperform data-centric ones
   (RLPD/WSRL family) exactly when the pretrained policy is comparable/inferior
   to buffer quality — our regime. Our documented advantage-collapse-to-1.0
   is this family's known failure signature. **More knob-tuning fights the
   class, not the tuning.**
3. **The bimodal-recovery pathology is representational and the literature
   names it.** CHDP (arXiv:2601.05675) and the diffusion-policy line state
   directly that unimodal Gaussian/deterministic policies cannot represent
   multimodal/hybrid action optima; the best domain-adjacent systems deploy
   **multimodal, temporally-chunked policies** (ACT/CVAE, diffusion, B-spline
   transformers). The strongest recent guidewire result (soft-robotic
   guidewire IL, arXiv:2510.09497: **83% on unseen geometries**) used an
   action-chunking CVAE trained with **429 recovery demos vs 218 normal** —
   recovery as *labeled data*, not exploration noise.
4. **Privileged information belongs in the ACTOR, not just the critic.**
   "Privileged Sensing Scaffolds RL" (arXiv:2405.14853): critic-only
   privileged setups (ours) capture only a fraction of the achievable gap;
   routing privileged obs into the actor recovered ~79% of the gap at
   250K–5M samples. This **re-explains our advantage-collapse**: the
   deployable actor cannot SEE the stall trigger (slack/windup/contact), so
   the critic's correctly-signed retract preference is information the actor
   has no feature to condition on. Teacher(privileged actor) → DAgger-distill
   → deployable student is the proven recipe (Learning by Cheating; Lee et
   al. Science Robotics 2020; Miki et al. 2022).

---

## 1. Ranked candidates (fit × evidence ÷ effort)

| # | Approach | Key evidence | Attacks | Effort | Verdict |
|---|---|---|---|---|---|
| 1 | **RLPD** — SAC + 50/50 offline/online symmetric sampling + LayerNorm critics + small ensemble + high UTD, **no BC term** (Ball/Smith/Kostrikov/Levine, ICML 2023) | ~2.5× prior SOTA across 21 benchmarks, ~300k-step convergence on sparse Adroit; LayerNorm ablation is decisive | Deletes advantage-collapse *by construction* (no weights to collapse); LayerNorm+ensemble = the value-stability recipe; ends the brake saga class | **LOW** (config-level on our SAC trainer; our balanced lanes ≈ symmetric sampling) | **Phase 1 — do first** |
| 2 | **Privileged-ACTOR teacher → DAgger student** (Scaffolder 2024; Learning by Cheating 2019; Lee 2020) | ~79% of privileged gap at our sample scale; beats asymmetric-critic-only with 100M samples using 250K–5M | The stuck-grinder root: makes the stall trigger observable → retract-when-stuck becomes near-Markovian and directly learnable; student inherits recovery instead of rediscovering it from dying noise | MEDIUM (privileged-actor path + DAgger loop; privileged pipeline already built) | **Phase 2 — the unlock** |
| 3 | **Residual RL on the scripted heuristic** (Johannink ICRA 2019; CR-DAgger 2025: +64% base success on contact-rich; domain-matched ICRA-2026 catheter paper: base+correction converged in 123 episodes) | Kills the cold-start/freeze class (residual≈0 init = instant competence; no BC bootstrap needed); slashes exploration burden | MEDIUM-LOW | **Phase 2 co-design** (teacher trains as residual-on-heuristic). Caveat: residual bound must allow sign-reversal or pair with a mode head |
| 4 | **Multimodal action head** — GMM first, then QC-FQL (Q-chunking + one-step flow policy; Li/Zhou/Levine NeurIPS 2025) or DSRL (SAC in a frozen diffusion policy's latent space, CoRL 2025) | CHDP: diffusion beats P-DQN/HyAR/hybrid-SAC on hybrid multimodality; QC: chunked retract-then-readvance becomes ONE Q-valued macro-action; deploy = single deterministic forward pass | The bimodal averaging error itself; also sidesteps tanh-Gaussian σ/α dynamics entirely | MEDIUM (GMM) → HIGH (QC-FQL port) | **Phase 3 — if the ceiling persists after 1+2** |
| 5 | Recovery demos as labeled data — DAgger/expert-in-the-loop at stall states; scripted stall→retract→re-advance segments at the RCCA ostium | 2602.20216 (59%, −26% sample cost); 2510.09497 (83% unseen, 2:1 recovery:normal demos) | The fading-micro-recovery pathology at the data level | LOW | Fold into Phase 2 (the teacher generates them; the heuristic labels easy regions) |
| 6 | HER with arclength subgoals + formalized reverse curriculum (Florensa CoRL 2017) | Standard; uses our planned-path + restore machinery natively | Reward-shaping fragility; start-state difficulty | MEDIUM | Adjunct in Phase 2 |
| 7 | SAC-X-style "reduce-slack" auxiliary intention (Riedmiller ICML 2018) — *idea only, not the scheduler* | Named skill practice vs hoping noise finds it | Recovery skill acquisition | LOW (aux reward channel from privileged slack) | Adjunct in Phase 2 |
| 8 | TD-MPC2/MoDem world model (ICLR 2024/2023); domain: thrombectomy world model (MICCAI 2025) | Best-in-class data efficiency (104 tasks) | Sample budget | HIGH | Fallback tier — only if model-free plateaus; FEM buckling is worst-case for model error |

## 2. Anti-recommendations (looked right, skip them)

- **PPO + domain randomization at legged-robotics scale** — needs 2–3 orders
  of magnitude more env throughput than 16 SOFA workers (Rudin CoRL 2022:
  4096 Isaac envs). Revisit only with a learned surrogate sim.
- **Another constraint-family variant** (TD3+BC, SPOT, CRR, better-λ AWAC,
  penalty scheduling): same structural anchor, same regime disadvantage.
  This closes the v2c penalty-bracketing question: **don't**.
- **Full HRL rebuild** (option-critic, HIRO, HAC, Director): thin evidence
  outside toy/DeepMind-internal domains, sample-hungry, fragile. Take the
  auxiliary-intention idea; skip the machinery.
- **Pure IL** (vanilla Diffusion Policy/ACT/BeT as endpoint): inherits the
  heuristic's regional ceiling, no online improvement loop.
- **DreamerV3** specifically: no natural demo-buffer path; TD-MPC2 dominates
  it on continuous-control data efficiency.
- **ep_step/time features as stall memory**: our own saliency finding +
  POMDP literature agree — replace with state-based slack obs (teacher) or
  short history (student).

## 3. The phased migration (composable with everything already built)

**Phase 1 — RLPD-ize the trainer (days; deletes the brake saga).**
Keep: SOFA workers, PER buffer (uniform-within-halves first, per paper),
asymmetric critic, procedural meshes, recovery relax, all guards/monitoring,
seed buffer as the offline half. Change: drop the AWAC BC term; SAC objective
only; 50/50 offline/online batches; LayerNorm on critics; ensemble ~5–10 with
min-over-2; UTD as GPU allows; entropy backup off (sparse-task recipe).
Expected: v2-level (~49%) performance with the entire σ/α/penalty knife-edge
class removed. Gate: eval2 ≥ 45% with entropy/clamp untouched by hand.

**Phase 2 — privileged-actor teacher + residual base + recovery data (the
unlock; ~1 week).** Teacher = Phase-1 learner whose ACTOR consumes the full
121-dim obs (sees slack/windup/contact), trained as a residual on the
scripted heuristic, with a "reduce-slack" auxiliary reward from privileged
signals and the stuck-restore curriculum (built, mesh-safe) supplying stall
starts. Distill to a deployable student (obs-only, short history replacing
ep_step) via DAgger on on-policy student rollouts. Deploy = deterministic
student. Gate: student eval ≥ teacher−10pts AND stall-response probe shows
state-conditioned retraction (the P(retract|slack) gradient that never
materialized under AWAC).

**Phase 3 — multimodal head (only if the stall ceiling persists).** GMM head
(cheap, deterministic via mode-argmax) → QC-FQL port if GMM insufficient.
Domain evidence says chunked/multimodal is what the 83%-unseen system used;
h=3–5 chunks (~0.4–0.7s) to preserve reactivity in contact.

**What this retires:** v2c penalty bracketing (anti-rec #2); AWAC-specific
Tier-A E1b (advantage normalization — moot without advantage weights).
**What survives unchanged:** E2 aux-label repair (feeds Phase-2 teacher),
E3 stuck-lane (composes with symmetric sampling), E4 stuck-pool curriculum
(Phase-2 ingredient), all deadlock guards, incremental-save/resume branch,
monitoring/probes (the P(retract|slack) probe becomes the Phase-2 gate).

## 4. Machine allocation proposal

- **Machine 1 (here):** kill v2b (0% eval, frozen mean — gate-failed);
  implement Phase 1 (RLPD trainer) on a branch; launch as the new mainline.
- **Machine 2:** hold v3a (AWAC-based — superseded in spirit); either run it
  anyway as the AWAC control curve, or skip straight to a second Phase-1
  seat / Phase-2 teacher work once Phase 1 validates.

Sources: full agent reports with per-claim citations and caveats in
`saved/rl_paradigm_research/` (domain-sota, algo-alternatives,
hierarchical-structured). Key unverified items flagged there: QC-FQL chunk
defaults, diffusion inference latency at 7.5Hz (why GMM-first), Scaffolder
gains are manipulation/locomotion-benchmarked (mechanism fits, magnitude
unverified on FEM wire physics).
