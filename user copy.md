
# Snapshot Analysis

All episodes are represented by [pid, episode number].
MS = Max Steps Case
FS = Fold Stall Case
WB = Wrong Branch Case

## 1. RCCA/RVA:- 
### a. MS: In MS cases Passing all entries in the main aorta (going all the way to the -ve x side) and then curving onto the end of main aorta mesh and having high negative reward < -60 (ex: [312, 1], [502, 1], [122/578/616, 1]) which is worse than MS cases when the wire is stuck just before the entry point of LVA (and the target is in RCCA and RVA; similar pattern for LCCA targets too; ex: [274/160/198, 2], [388, 2]). In some MS cases it's better and about the same reward as when stuck before LVA entry point (reward < -30 but > -35; ex: [160/388/540, 3], [350/464/502, 3]). This should not be case as wire went way further but crossed the entry points of RCCA/RVA. Also why is wire travelling that far since as soon as it passes the entry point of RCCA/RVA it should be off_path and in retract mode with a newer planned path (different from initialized one).

### b. FS: FS only happens in the RCCA cases [160/72/426, 1] and they happend in the same end orientation as MS; again why isn't wire retracting way sooner and what's the different episodes evolution that's causing thgis difference


## 2 LCCA:-
### a. MS: Stuck right outside LCCA entry episode got the worst reward (-54; [350, 1]) as compared to suck right before LVA/RVA/RCCA entry episodes (-30s, [122/502/578, 2]) . Analyse and Debug

### b. FS: Catheter seems stuck at/near LCCA entry but guidewire is hanging out in the middle of the vessel (aorta); what's the problem ; why can't it move forward / retract 

## 3. LVA:
### a. FS. I think it does make it to LVA entry but gets stuck , why ? too much rotation ?

### b. MS: stuck at/near LCCA entry



-----


Item 8 — fold-stall _prev_d_corr_arc (minor residual)
Verified the fold-stall detector at [env5.py:518-538](training _scripts/util/env5.py#L518-L538) genuinely reads _prev_tip_s, _prev_inserted_gw, _prev_d_corr_arc. After Plan v5:

delta_s (tip arclength change) → recoverable from frame-stacked tracking.
delta_gw (insertion change) → ≈ last_action gw_trans.
_fold_stall_count itself → feature 24.
Residual: the d_corr_improving bypass compares against _prev_d_corr_arc (previous-step d_corr). The agent sees feature 14 at step t but not t-1, so it can't perfectly predict whether the next fold-counter increment will be cancelled by the bypass. Minor — shifts fold-truncation timing by a few steps; the counter itself is observable.
-----

check all options for DualDeviceNva_train
curriculum 
mesh all the time 
step wise - what all needs to be done (compare with episode code)
HER / PER
since we have redefined heuristic success for RCCA (threading RCCA); lets make it 40% for sac trainig 


why is --min_success_rate 0.15 in train.sh when definition of success is so lenient (I think now it should be 0.40 since rate of threading RCCA is very high as we learned previously from just running daughter specific heuristics ); I also notice there is no snapshot option in train command , I think I would need snapshots for all heuristic episodes that are fed into replay buffer for training and also 100 evaluation episodes that come in betweens , also for 10 best and 10 worst for every 100 training time episodes 

----------



The reward bug: _committed_forks is a single shared latch keyed only by j_arc (pathcontext.py:1094-1098). The first event at a fork — whichever sign — latches it, so the other sign can never fire. A wire that goes RVA (−1) then corrects into RCCA gets only the −1; one that goes RCCA (+1) then dips RVA gets only the +1.
→ Fix: latch per (j_arc, sign) instead of per j_arc, so each fork can emit one +1 and one −1 independently.

The success definition: seeding success = episode finishes with the wire inside the RCCA daughter — RVA detours along the way are fine. The current filter is reached_target_daughter AND NOT received_wrong_daughter:

reached_target_daughter is an ever-touched latch, not a final-state check.
NOT received_wrong_daughter wrongly disqualifies legitimate RVA-detour-then-RCCA episodes. → Fix: seeding success = wire's final _current_branch_idx == target_daughter at episode end. Drop the received_wrong_daughter exclusion.
These are two separate concerns: fix #1 repairs the ±1 reward signal; fix #2 repairs the seeding filter label.

This run's buffer is being built with both defects, so it should be restarted. Shall I stop the container now, implement both fixes, and relaunch? (Pausing the monitoring loop pending your go-ahead.)


-----------------



Straight answer: yes, PER can and does sample them — but it gives them no special protection, and that's worth understanding.

They are definitely in the buffer and the priority tree:

Buffer capacity is 1e6; buffer_len=132,148. The ~55k cache transitions sit at ring positions 0–55k, heatup ~55k–87k — nothing has been overwritten, all heuristic transitions are present.
PERVanillaStep.push() puts every pushed transition into the sum-tree at max_priority^α — cache-loaded ones go through the identical path. So they all entered with max priority → each is guaranteed ≥1 sample.
But PER does not privilege them. Once a heuristic transition has been sampled, its priority is reset to its actual (|TD|+ε)^α. PER then samples ∝ TD-error. Heuristic transitions from consistent threading behavior tend to be easy for the critic to fit → low TD → low priority → sampled less and less over time. PER deliberately shifts compute toward high-TD transitions, which increasingly are the novel explore transitions. The rising q1_loss is consistent with exactly that — PER is chasing the harder explore data.

This is a known gap: vanilla PER has no demonstration bonus. DQfD-style replay adds a per-demo priority bump (ε_D) so seeded demos keep getting replayed regardless of TD. Ours treats heuristic and explore transitions identically — so the heuristic demos get less replay as training proceeds, not a protected amount.

What I can't measure from the current logs: the exact heuristic-vs-explore sample split. batch_samples logs batch_idx (in-batch position), not the buffer/tree index, and actions are normalized for both sources so they're indistinguishable. Buffer composition is ~55k/132k ≈ 42% heuristic and shrinking as explore data grows.

If you want hard numbers on the split — or want the heuristic demos protected (DQfD-style bonus) — both need a small code change + restart. Want me to add tree-index-distribution logging so we can see the actual split, or is the conceptual picture enough for now?

--------------------


Yes — and what you're describing is a real, well-studied axis: what to prioritize when sampling. PER prioritizes by TD-error ("what the critic finds surprising"), which is theoretically efficient but, as we're seeing, amplifies instability during turbulent phases. The alternatives prioritize by outcome quality instead:

Options, roughly in order of effort:

Balanced two-stream sampling (cheap, high value) — keep a separate stream of "clean" episodes (reached the RCCA daughter / near-positive return) and draw a fixed fraction of every batch (e.g. 25–50%) from it, the rest from the general buffer. Directly does what you're asking, and it's a small change.
DQfD-style demo protection — give the heuristic-seeded transitions a permanent priority bonus so they don't get starved as PER drifts toward explore data (the gap I flagged earlier). Bolts onto the existing PER.
Return/outcome-prioritized replay — replace TD-priority with episode-return priority. More stable than PER, but can over-imitate.
AWAC (Advantage-Weighted Actor-Critic) (the principled fit) — weights the policy update by exp(advantage), so good actions are imitated more and bad ones ignored. It was designed for exactly this regime — warm-start from demonstrations, then fine-tune online — and is notably more stable than SAC+PER with a demo-seeded buffer. It needs no PER at all.
The one tradeoff to not get wrong: you can't train only on clean/positive episodes. The critic needs the failures to learn correct low values for bad states — otherwise it stays optimistic about regions it never sees and the policy chases mirages (that's part of what the q-recalibration turbulence is). So the fix is balanced sampling, not filtered sampling.

One caveat specific to us: "positive reward" is a noisy filter right now because of the deferred ±1 daughter-commit bug — episode return is partly corrupted. The robust quality signal is the reached_target_daughter flag (did it thread RCCA), not raw return. Any clean-episode stream should key off that.

My recommendation: the immediate diagnostic stays the PER-off baseline (cleanly tells us if PER is the amplifier). If we then want a better sampler, balanced two-stream sampling keyed on reached_target_daughter is the cheap high-value move; AWAC is the principled upgrade if we're willing to change the algorithm. Which direction do you want to explore — or should I let the current run keep going and we decide after the baseline comparison?


--------------



The step-RL stabilization suite is in, all six strategies gated by independent CLI flags. Every flag defaults off/neutral, so omitting them reproduces current step-SAC byte-for-byte.

Flag	What it does
--grad_clip <norm>	clip_grad_norm_ on critic+policy (the prime divergence fix — grad ran unbounded to 434k)
--update_per_explore_step <f>	overrides the update:explore ratio
--demo_priority_bonus <ε_D>	DQfD — heuristic transitions keep a permanent PER priority floor
--priority_mode {td,return,outcome}	PER priority source: TD-error / episode-return / clean-thread
--balanced_fraction <f>	two-stream sampling — a fixed batch fraction from "clean" episodes
--algo {sac,awac}	AWAC advantage-weighted policy update
Files modified (10): sac.py (grad clip + AWAC branch), gaussianpolicy.py (log_prob), replaybuffer.py (EpisodeReplay quality fields), pervanillastep.py + vanillastep.py (priority modes, balanced sampling, demo bonus), pervanillashared.py + vanillashared.py (param threading), experience_cache.py (metadata persistence), agent.py + DualDeviceNav_train.py (CLI args + wiring). launch_rcca_train.sh gets --grad_clip 1.0.

Verified: all 10 files compile; unit tests pass — experience-cache metadata round-trip + pre-v8 backward-compat, and PERVanillaStep outcome/return/demo-bonus priority ordering + balanced clean-stream draw.

Documented: RL_IMPROV_9_CHANGES.md §16 + File Index; plan file Plan v8 marked implemented.

One caveat carried forward: the existing rcca_heuristic_cache.npz predates the metadata format, so --balanced_fraction/--priority_mode on cache demos need the cache regenerated (the loader still blanket-tags is_demo=True, so --demo_priority_bonus works regardless). Phase-2 docker smoke tests (--grad_clip 1.0, --algo awac, --balanced_fraction 0.3) are the next validation step when you want to launch.


------------


# Plan(RAS2):

ok , now we have indentified a stable setup, now we need to capitalize on it and improv bigly;

First we indentified previously a couple of bugs in our reward setup ; we need to fix it and to improve it further we need to make a few more modifications ; right now all our RLs come with a standard step penalty that means they are getting 0.001 for every step at the beginning and also at RCCA daughter and (11) when they are much closer to the target which sometimes encourages agent to be slow at the start and remain close since there is not much cross_track penalty and   .001 penalty for 600 steps doesn't count to much so we need reformulation where they are getting 0.007 (closer to start) - 0.002 (closer to (11)) penalty, 0.007 penalty in (0) (past bifurcation), no penalty in (11) , 0.005 step reward in RCCA and 0.007 step penalty in RVA ; right now +1/-1 are awards for making right decision at every fork, for each juncton make it -0.05 (that they can get again and again, when threaded wrongly) / +1 (once per fork per episode) so that they are not afraid and encouraged to explore to go for +1 ; obviously we will be running new heusristics / heatup for our new reward setup, for heuristics we will be aiming for 100 successful (100% success in threading RCCA) episodes (so for 100 succ. it will be running probably upto 400/450 heusistic episodes) as AWAC closely mirrors heuristic and we will be starting a little before and outside (11) (SOFA restore), one question I had, we will be running heuristics from z=345 as we are now but during RL we would be using SOFA restore just outside (11), would we append the saved steps and reward from that episode and continue this episode forward and q1 mean would reflect expected reward from the start point and saved heuristic steps but optimized steps after the start point (within daughters) because this would be ideal.  The other question I had can we also implement one of these:
1. Balanced two-stream sampling (cheap, high value) 
2. DQfD-style demo protection 
3. Return/outcome-prioritized replay 
within AWAC to optimize success further faster , I think 1 is best within AWAC.

Read below all text with regards to what's written up to make the plan:

## Rewards

The reward bug: _committed_forks is a single shared latch keyed only by j_arc (pathcontext.py:1094-1098). The first event at a fork — whichever sign — latches it, so the other sign can never fire. A wire that goes RVA (−1) then corrects into RCCA gets only the −1; one that goes RCCA (+1) then dips RVA gets only the +1.
→ Fix: latch per (j_arc, sign) instead of per j_arc, so each fork can emit one +1 and one −1 independently.

The success definition: seeding success = episode finishes with the wire inside the RCCA daughter — RVA detours along the way are fine. The current filter is reached_target_daughter AND NOT received_wrong_daughter:

reached_target_daughter is an ever-touched latch, not a final-state check.
NOT received_wrong_daughter wrongly disqualifies legitimate RVA-detour-then-RCCA episodes. → Fix: seeding success = wire's final _current_branch_idx == target_daughter at episode end. Drop the received_wrong_daughter exclusion.
These are two separate concerns: fix #1 repairs the ±1 reward signal; fix #2 repairs the seeding filter label.

This run's buffer is being built with both defects, so it should be restarted. Shall I stop the container now, implement both fixes, and relaunch? (Pausing the monitoring loop pending your go-ahead.)

## SOFA RESTORE

Investigation results — verified directly inside the running container
SOFA Python API has NO native full-scene serialization. Confirmed from inside the container:


Sofa.Simulation methods: animate, animateNSteps, init, initTextures, initVisual,
                         load, print, reset, unload, updateVisual
No WriteState / ReadState / exportXML / serialize / snapshot. The compiled bindings just don't expose it. So a "save the whole scene like a video game" via SOFA's own API isn't available — would require a custom C++ plugin.

However, the current checkpoint is far thinner than it could be. A MechanicalObject (the wire's DOFs container) exposes these state DataFields:

field	currently saved?	meaning
position	✅ (as dof_positions)	xyz + quaternion per beam node
velocity	❌	per-node linear+angular velocity
force	❌	currently-applied force per node
free_position	❌	predicted position before constraint solve
free_velocity	❌	predicted velocity before constraints
derivX	❌	acceleration / dx-per-dt
externalForce	❌	externally-applied force vector
We're saving 1 of 7+ state fields per MechanicalObject. Then sofabeamadapter.py:136 calls Sofa.Simulation.reset(self.root) which explicitly zeroes velocity/force, and the 50-step "settle" loop tries to reconverge from rest. That settle re-derives a contact manifold from scratch — sometimes it converges to a different local minimum than the one that was saved. This is exactly the wrong-location bug.

Worse: sofabeamadapter.py:139-150 deliberately zeroes the saved rotation_instrument (per a prior RL_IMPROV_7 fix). So even the wire's J-tip orientation that committed it toward the right daughter is intentionally erased on restore.

What's also gone but unrecoverable via Python:

BeamAdapter plugin internals (per-beam interpolation frames, strain) — not in factory accessible scope
LCP solver cached lambdas / impulses
Broad/narrow-phase contact pairs (BruteForceBroadPhase rebuilds these every step anyway)
Practical proposal — "fuller save" without writing a C++ plugin:

Save 6 more arrays per checkpoint: velocity, force, externalForce, free_position, free_velocity, derivX from ic.DOFs.
On restore: skip Simulation.reset() if checkpoint has the new fields, OR call reset then assign all 7 fields directly (overwriting the zeros).
Restore rotation_instrument instead of zeroing it (revert RL_IMPROV_7 §7 Fix 6 — accept that this re-introduces the prior issue, since now we're testing a stronger restore).
Reduce settle_steps to 1-3 (or 0) — with velocities preserved the solver shouldn't need to reconverge from rest.
That should catch >80% of the missing physics state. The remaining gap (BeamAdapter internal state) would only matter for very long-lasting strain memory effects which probably aren't the dominant cause of the 36% wrong-location problem.

## Further optimization within AWAC

Balanced two-stream sampling (cheap, high value) — keep a separate stream of "clean" episodes (reached the RCCA daughter / near-positive return) and draw a fixed fraction of every batch (e.g. 25–50%) from it, the rest from the general buffer. Directly does what you're asking, and it's a small change.
DQfD-style demo protection — give the heuristic-seeded transitions a permanent priority bonus so they don't get starved as PER drifts toward explore data (the gap I flagged earlier). Bolts onto the existing PER.
Return/outcome-prioritized replay — replace TD-priority with episode-return priority. More stable than PER, but can over-imitate.

--------------


The experiment is launched and monitored. Status:

rcca_awac_v2 is running — single-run design as you requested. Sequence it will execute:

Heuristic seeding from z=345 (~333-500 episodes) → builds rcca_heuristic_cache_v2.npz + fills the pre-bif(11) checkpoint pool (one per RCCA-final success).
Heatup → builds rcca_heatup_cache_20_v2.npz (now runs from the restored fork, since the pool is filled by then and heatup isn't heuristic-mode).
10k warm-start pretrain updates on the filled buffer (the gate fix lets this fire even though caches are generated, not loaded).
Online explore + train from SOFA-restored pre-bif(11) states.
The three blockers I fixed to make the single run work:

Lazy CheckpointRestoreWrapper (confirmed in logs: 0 .npz files accepted at startup, no crash — it'll re-scan once seeding fills the pool).
Warm-start gate now accepts a seeding-generated buffer (not just disk-loaded caches).
Relaxed the --insertion_z / --checkpoint_dir mutual-exclusivity for sofa_restore mode.
A persistent monitor (b7676uxls) is watching for seeding-batch progress, cache saves, the 10k warm-start, eval Quality:, and any divergence/error/OOM. I'll report as milestones land — next expected is the first seeding batch result. Once seeding completes I'll verify the pool actually has ~100 checkpoints before trusting the restore phase.


-------> so all the explore updates and training will happen post SOFA restore steps; right?
--------> after heuris... finishes ; gather all 100 selected episode snapshots 
--------> (0) then should 0.007 step penanlty post (0) point where it needs to turn up; otherwise it could stay at same 0.002
-------> there should be +2 more reward at  commiting to (11) (and not going deeper into (0)) and post top junction (not getting stuck in LVA direction) that happene earlier in the trunk 

-----> ok, it's all fine ; let's do some deep analysis in the meantime ; for example can you analyse the logs of successes and failures and see if there's any way it can be trained faster ; for example most of the failures 
--balanced ratio 
--planning
--success sofa res. states 
---The dominant failure is overshooting straight into (0) instead of turning up into (11). RL can only turn at the right moment if the observation makes the turn visible — verify LocalGuidance's arc_to_next_daughter and direction-to-correct-entry are sharp and correct in the (0)-corridor approach. If the wire can't "see" the bridge coming, no reward shaping will fix the overshoot.
---so about your SOFA store seeding analysis 140 vs 1028 : I check the sofa start snapshots of both 140 and 1028 stages (based on timestamps of when they were saved) and found out why this huge divergence of trainig progress between 2 runs . Making the pool larger was a mistake since you are making sofa restore states based on existing sofa restore states with guidewire more pushed in further hence making the pooi of sofa res. states inconsistent (no consistent start point) which will inevitable cause it to overshoot , also making start states harder by pinning it to the entry, when instead we should have gone the other way; recognize the sable start points out of those 140 (maybe 10 states or so where guidewire is little ahead of cathetar but comfortably before (11) entry ) and only have done training on those 10 or fewer states (one isn't bad either to get the guidewire consistently thread (11)/RCCA). Now I have realized that you can actually calculate which of those 140 states got the most success in the 202750 run by actually looking at the success folder (about 1500) and find the matching start state in the start folder and reading the start point of that episode for cathetar and guidewors bucketing successes for the start states and then cross referencing them with the saved 140 start states  

------so it means that you can extract the success episodes and fail episodes of these list start states --> (success	n	gw	cath	gap	file
46.7%	7/15	109.0	101.0	8.0	pre_bif11_pid20043_ep0003_step0038.npz
27.3%	6/22	118.6	104.3	14.3	pre_bif11_pid4907_ep0176_step0054.npz
26.7%	4/15	114.2	99.0	15.1	pre_bif11_pid19116_ep0017_step0050.npz
25.0%	4/16	101.4	97.4	3.9	pre_bif11_pid20926_ep0002_step0115.npz
22.0%	9/41	115.0	100.7	14.4	pre_bif11_pid3285_ep0046_step0132.npz
20.0%	4/20	104.0	102.9	1.1	pre_bif11_pid10145_ep0075_step0060.npz
) --> which gives us 34 successful states and several states and that store can make a better heuristic seeding file ; we can also reduce the number of updates say ~ 500 /1000 to keep the entropy healthy but first can you make snapshots of all these listed states in a separate folder under saved folder ; make snapshots of these good states and also of the other 148 states that ar mostly bad
----new heusristic heatup reduced
---entropy regulation 
--- pretraining reduced



-----sofa restore seed 
     ----investigate shapshots 
     -----retaction behaviour
     -----heirrchial learning 
     ----- speed/tolerance lever 
-----episode RL
------update per explore

--------"The two levers that target this failure (both RL-stabilization knobs, not reward/obs/terminal changes) are: (a) raise awac_lambda 3 → 8-10 (softer advantage weighting keeps the policy near the broader buffer distribution instead of cloning only the few saturated high-A actions — slows the collapse), and (b) add a true entropy bonus to the AWAC policy loss (penalizes log_pi directly, including the tanh-Jacobian, so it counters the mean rail — the std floor can't). Per Plan v10's own decision tree, (b) is the deferred lever for "entropy collapse with floor insufficient."" ---> You mention some solutions to this , can it be that if we have tried a lower ratio of update per explore step which we increased from 1:20 to 1:1 it would have been better since policy would have the chance to make a lot more exploratory steps before locking in behaviours.

----------
harvester should also load varying mesh/ the whole mesh varying without extrenal dataset

now decresease the retraction steps penatly (during off_path) to 0.002 instead of 0.007 but make sure the wire is well 0ff_path (atleast 2-3 few steps inside the wrong path  ) before it begins retarction. 

what launch_screen_stuck.sh is for 

eve_rl_deep_review_combined.pdf, eve_rl_multimesh_generalization_combined.pdf ----> These are 2 pdfs in which we have done most of the fixes , First you scan these files to locate what all the fixes we have done ; you can easily look up previous chats in this session and the code to do that . Then every two hours scan the logs and values and losses to verify if these fixes wrt learning stabilty diverse ways and revovery schedules are holding ; do this  thoroughly every two hours , log your findings but don't make any changes ; let the run continue

------------

Q1: Are micro-recoveries actually happening in the successes?
Yes — 53% of explore successes contain at least one full stall→retract→resume cycle, and 35% contain your exact scenario (stuck while threading RCCA, ease slack, re-advance). One captured deterministic-eval success shows a textbook cycle: stall at step 337 with buckle_phi engaged, retract latency 6 steps, resumed by 349, succeeded.

But three uncomfortable truths behind that number:

The behavior is FADING, not being learned: cycle prevalence in successes falls 86% → 58% → 28% across training thirds; post-stall forward-push intent hardens +1.8 → +7.2 mm/s. The micro-recoveries were distilled from exploration noise (noise generates small retractions everywhere; AWAC clones the ones inside successes) — and with α at its ceiling and entropy collapsing, that demo source is drying up.
Retraction isn't the success discriminator — stall avoidance is. Failures actually retract-and-resume more than successes (3.8 vs 1.4 cycles/ep). Successes mostly don't stall, and when they do, buckle≈0; failure-stalls carry buckle −0.19.
The held-out eval gap (30.6% vs ~44% explore) is precisely a recovery gap — eval failures are stuck-grinders: 69% stall, post-stall response still +13mm forward net, strong pull-backs only 7%.
Q2: Did the privileged critic actually pass its knowledge to the policy?
Inference: partially yes. Behavior: no. The aux head genuinely learned to infer contact from deployable obs (r≈0.75 vs 0.002 for slack-alone) — the trunk carries the knowledge. But P(retract) is flat across contact quintiles, and the policy's correlation with its own contact inference is positive — it pushes harder when it senses contact. Three causes, in order of importance:

Credit assignment is the bottleneck (biggest lever): the critic has the right preference — AWAC weight 1.042 for retract vs 0.947 for advance in the buckled tail — but with advantage std 0.092 and λ=1.0, the weights span only [0.72, 1.25] ≈ uniform BC. The critic knows; the policy update can't hear it. λ needs ~0.1–0.3, or advantage normalization.
Silent bug: the SOFA force labels are dead — flat dims 99–101 are identically zero across all 569k buffer states (the dofs.force accessor exception-guards to 0). Half your aux capacity trains on constants.
The live contact labels are scale-starved — std ~1e-3 normalized → aux_coef·MSE ≈ 5e-8 — effectively zero shaping pressure. They need z-scoring or ~50× tighter normalizers.
Q3: Too much or too little observation?
Both, and mis-weighted is the bigger problem. The single most alarming finding: ep_step is the #1 saliency input for all four action means (and the aux and log_std heads), with the four last_action dims at ranks 2–5 — time + own-momentum = 17% of total saliency, feeding exactly the determinism drift we're watching. Meanwhile: ~21 prunable dims (frame t-1 body offsets are r=0.99 duplicates; in_wrong_branch is the exact negation of on_path; d_rem_log got ignored), at_ostium is dead at source on procedural meshes (another bug), and curv_ahead is variance-crushed by its /10 scaling. The good news: the Gen-4 wire-state channels genuinely work — slip is rank 5, slack 13, fork geometry and the action-mask flags all load-bearing. Missing: catheter-relative-to-path state (P4-O3), torsional windup (the privileged sin/cos rotations carry std 0.71 of state the policy can't see, despite two rotation actions), a stuck-duration integrator (slip is 1-step), signed cross-track.

The machine-2 setup (v3), prioritized
Stage A — no obs change (reuses the existing seed, launch fast):

T0 bugs (safe, pure fixes): repair/repoint the force accessor; z-score aux labels + retune aux_coef so the term is 1–10% of policy loss; fix at_ostium; fix br_trunk mis-binning.
T1 credit assignment: awac_lambda 1.0 → ~0.3 (or advantage-normalized exp(adv/std·τ)), targeting p99/p1 weight ratio 5–20×; plus a stuckness-balanced sampling lane (slack/contact-tail analog of balanced_fraction — stuck states are ~10%/1% of the buffer and gradient-starved).
T2 recovery source: the stuck-pool restore curriculum — the machinery is already built and mesh-safe (STUCK_CHECKPOINT_DIR harvest → screen_stuck_pool escapability filter → fingerprint-matched restore). Start episodes in screened stuck states so retraction gets on-policy credit instead of relying on dying exploration noise. Optionally scripted retract-jiggle demos at the RCCA ostium (82% of stalls happen there).
Stage B — obs surgery (breaks seed caches → fresh harvest on machine 2):

Add (~8 dims): cath along-path projection gap + cath cross-track, sin/cos cumulative commanded rotation per device (deployable windup proxy), stuck-duration EMA + steps-since-tip-moved, signed cross-track.
Prune (~21 dims): frame t-1 body offsets (keep tip rows or replace with a t-5 delta), in_wrong_branch, d_rem_log; fix curv/radius/target scalings.
De-crutch: drop ep_step from the policy prefix (keep it in the privileged tail for the critic), and feed executed last-action alongside commanded.
The one item needing your explicit approval (reward-frozen rule): a small retract-when-stuck reward term conditioned on the detectable stall signature (proj_s flat ≥10 steps + buckle ≤ −0.05). The buckle potential alone rewards slack reduction but evidently hasn't overcome BC inertia; T1+T2 may suffice without it, so I'd hold it as the Stage-A fallback. Everything's in memory + the saved reports; v2 keeps running untouched (its ~40% explore / eval2 due shortly remains a valid baseline for v3 comparisons).




-----


Written: RL_IMPROV_16_EXPERIMENTS.md — the machine-2 experiment plan, every experiment traced to a verified finding (F-A credit assignment / F-B fading micro-recovery / F-C half-dead aux + mis-weighted obs), with exact settings, files, gates, and cache-compatibility flagged throughout. The shape:

Tier A — v3a (launch first, reuses the existing seed, all default-off flags):

E1 — advantage discrimination: preferred arm is batch-normalized advantages (exp((adv/σ)/τ), τ=2.0 → weight p99/p1 ≈ 12) over a bare λ=0.3, because it self-calibrates as the critic sharpens. Success gate = the contact-bin P(retract) gradient reappearing in the probe.
E2 — aux repair: repoint labels "2,3,5,6" → "0,1,5,6" (off the dead force dims onto live velocities+contact) and z-score at loss time — the key trick that fixes the 5e-8 gradient without touching stored obs, keeping the seed cache valid. Plus: log the aux loss (it was invisible and silently ~0 all of v2).
E3 — stuckness-balanced sampling lane (--stuck_fraction 0.15, cloning the clean-lane SumTree pattern, thresholds from measured buffer stats).
E4-prep — free stuck-pool harvest via STUCK_CHECKPOINT_DIR during the run.
E8 — monitor upgrades so every gate is measurable (P(retract|stuck), aux R², weight p99/p1).
Tier B — v3b: the stuck-restore curriculum (E4) using the screened pool — with the one gap I found during verification: the restore wrapper currently fires on every reset, so it needs a small --restore_prob 0.3 addition. E7 (pay-on-escape bonus, designed unfarmable) is held as the approval-gated fallback if micro-recovery still fades.

Tier C — v4 (cache-breaking, bundled as one change-set): the obs surgery — prune 21 dims, add 8 (catheter-path state, commanded-rotation windup, stuck-duration integrator, signed cross-track), fix the scaling/dead-feature bugs, de-crutch ep_step out of the policy prefix, and optionally computeConstraintForces=True for real force labels (the only way, given the verified scratch-buffer root cause).

Tier D: E5 (entropy-regime A/B) is deliberately held pending v2's eval3 — if the deterministic-aggressive regime holds ≥49% there, leave it alone and let E4 supply recovery pressure; if it regresses, arm the α-ceiling raise or the per-dim entropy floor (that scaffold already exists as --entropy_beta_per_dim).

The sequencing table enforces one-tier-per-launch for attribution, identical eval protocol everywhere, and v2 as the standing control curve. Rough effort: v3a ≈ 1 day (three <50-line diffs in already-mounted files), v3b ≈ half-day, v4 ≈ 2–3 days + re-harvest. Want me to commit this doc, or start implementing the v3a diffs so they're ready when the second machine is?

d:\Arjun\workspace\neve\RL_IMPROV_15_CHANGES.md
are the tier A changes need to be done one by one like E1 the E2 etc.... are all at once is recommnded
Bundle them — that's deliberate in the plan — but with a defined peel-off protocol. Here's the reasoning:

Why all-at-once is right for this particular trio
1. They're three legs of one mechanism, not three independent bets. The target outcome (policy retracts when stuck) needs all three simultaneously:

E1 makes the critic's preference audible (weight spread),
E3 makes stuck states present in the batch (they're ~10%/1% of the buffer — E1 alone amplifies a preference the update rarely sees),
E2 makes the trunk represent stuckness sharply enough to condition on.
Run E1 alone and you'll likely measure "no behavioral change" — not because E1 failed, but because the stuck states it would act on appear in ~1 of 10 batches at tiny gradient share. A one-by-one ablation would produce three individually-weak results and a misleading conclusion.

2. Attribution is preserved by the metrics, not by the schedule. This is why E8 ships in the same launch — each component has an orthogonal, component-specific diagnostic that isolates it within the bundled run:

Component	Its own metric (independent of eval)	Its own abort trigger
E1	awac weight p99/p1 ∈ [5,20]; P(retract|contact-bin) gradient	success −10pts vs v2 at matched steps ×2 checks
E2	aux R² per label rising; aux-loss magnitude in CSV	policy-loss noise ↑ >2× → halve aux_coef
E3	stuck-lane hit-rate ≈ 15%; q1_mean slope vs v2's	q1_mean pessimism outpacing v2
If the bundle underperforms, these tell you which leg misbehaved without a separate run.

3. The economics. Sequential ablation = 3 runs × ~10h-to-eval2 each ≈ 3 days before the combined config even exists — on the machine whose whole purpose is to get past v2. A bundled launch reaches the same decision point in one run. And because every change is a default-off flag with seed reuse, peeling one component out costs a ~25-minute relaunch (startup + 7-min pretrain), not a code change.

The peel-off protocol (making the implicit explicit)
Launch v3a with E1b+E2+E3 (+E4-prep, E8 — those two are zero-interaction: pure data collection and monitoring).
First gate at eval1 (~287k explore, ~5-6h): if ≥ v2's 30.6% and component metrics green → continue to eval2.
If eval1 regresses: the component metrics name the suspect (in practice: E1 if weights concentrated/saturating, E3 if q1_mean diving, E2 almost never — it's the near-zero-risk one since its current state is "trains on constants"). Relaunch minus that one flag. One ablation run only happens if needed, targeted at the implicated component.
Risk ranking if you have to guess: E1 riskiest (weight concentration shrinks the effective batch), E3 moderate, E2 ≈ free.
One thing I would not bundle: the two E1 arms — pick E1b (adv-norm, τ=2.0) and don't also change λ; two knobs on the same weight distribution in one run genuinely does destroy attribution.

The only scenario where I'd flip to one-by-one: if machine 2 turns out to be much faster than machine 1 (evals every ~2h), sequential becomes cheap and the extra rigor is nearly free. Otherwise: bundle, gate at eval1, peel only on evidence.

----------

Before we did our current experiments we got about 70 acc at lcca training  , then you analysed RL speicfics and suggested improvements in RL Awac formuilation and obs/reward formulation , analsse what was RL AWAc before wrt learnoing stability, entropy collapse, aplha and rails and why we needed to change it, document these and explain why we made the changes that we did ; what's happening now and if we want to make learning more stable with just this configuration (and not with Tier A chmages ; all the Es that we shipped for the second machine ) but with inprovememnts to awac , can it be done , make it more stable , I want to give this one more change before we continuw with Tier A, B,C etc changes

-------

So . I keep thinking if the crtic was actually was able to use it's proviledged state and pass on it's hidden representational knowledge to the policy which may then decide to retract if stuck or buckled ; right now you are looking at recoveries differently which is wire going completely different way and then somehow turning to the right path; I was talking about micro-recoveries where the wire went into RVA or got stuck for a while (even if it threaded the RCCA) but got stuck , realized it's stuck and then retract to ease the slack and then move forward; are these actually happening in these successes and what else can we do in learning algorithms setup or obseravation tuple to make learning more robust wrt planned path and wire state ; are we putting too much observation in the environment or too less? Inverstigate that if I have another machine to run this setup, what's the next best iteration setup wrt above to use

----------

MACHINE-2 (v3a) STANDING MONITORING MANDATE — copy-paste this whole block to the machine-2 Claude session

Every 2 hours, run a READ-ONLY monitoring pass on the rcca_procedural_v3a run. Make NO changes to code, containers, or the run — only observe and log. Append each pass (numbers, deltas vs previous pass, verdict OK/WATCH/ALERT) to saved/monitor_rcca_v3a.md. Set up a recurring schedule for this if you can; otherwise run it when I paste this.

PART 1 — health metrics (script already in the repo):
    bash monitoring/monitor_pass_v3a.sh
It prints liveness, the latest losses-CSV row, gate scans, the deterministic probe, and the eval trajectory. Read the results against these healthy ranges (v3a = AWAC + Tier-A bundle):
  - alpha in [0.0067, 0.100] (rails -5.0/-2.3) and MOVING; pinned at either rail for >50k updates = ALERT.
  - entropy_proxy: should decline gently as the policy sharpens (v2 healthy phase: 2.7 -> 1.0); entropy pinned high ~2.5+ all run = mean not growing (v2b failure signature); cratering toward 0 = collapse precursor.
  - q1_mean / target_q_mean: bounded, tracking each other; q1_mean diving hard negative = suspect E3 (stuck lane oversampling pain states); monotone exponential growth = critic divergence (ALERT).
  - q1_loss/q2_loss O(0.001-0.1); grad norms bounded nonzero; nonfinite counts must stay 0,0,0; clamp_fraction < 5%.
  - awac_weight_p99p1 (col 29, E1b gate): want 5-20. ~1.5-1.7 = BC-degenerate (E1b not biting); >>20 = over-sharp (effective batch collapse — E1b suspect if eval regresses).
  - aux_loss (col 30, E2 gate): O(0.1-2) and slowly falling = aux heads learning; ~1e-7 = still dead (znorm not active — check flags).
  - CLEAN_RAIL_FILTER rejections: a few % of successes is healthy; >30% = policy railing (WATCH).
  - Deterministic probe (monitor script section A3): freeze ratio vs own PRETRAIN baseline — OK >= 2x, WATCH >= 1.25x, below that FREEZE-ALERT (v1 died this way: eval quality collapses while explore looks fine, because sigma=1 noise masks a dead mean; NEVER trust explore success alone).
  - Probe retract-vs-slack: tail-bin P(retract) minus base-bin should be POSITIVE and GROWING across passes (v2: +13pp decaying to +8pp — the decay is what E1b/E3 exist to fix).
  - Probe aux R^2 (E2, labels 97/98/102/103): contact pair >= 0.55 and rising; velocities joining later.
  - Eval trajectory reference (v2): 6.1 -> 30.6 -> 49.0 -> 30.6. GATE: eval1 >= ~30% and component metrics green -> continue; regression -> the component metrics name the suspect (E1 if weights concentrated, E3 if q1_mean diving, E2 almost never) — report, do NOT relaunch on your own.
  - Also each pass: RestartCount, OOM flag (docker inspect), post-eval stall (buffer save lines appear and CSV keeps advancing within ~30 min of an eval), GPU util + container mem (docker stats; mem creeping toward the cap = WATCH — machine-1 hit an in-container OOM kill at ~87%).

PART 2 — soft/hard recovery tracking per 400 episodes (NEW: save the script below as monitoring/recovery_tracker.py, then run it each pass):
    docker exec -i rcca_procedural_v3a python3 - < monitoring/recovery_tracker.py
What it measures (from the per-worker step logs): a stuck event = the wire is being pushed but path progress stalls for >= 12 consecutive steps. After a stuck event:
  - SOFT recovery (the behavior we WANT): retract a little (1-8 mm of guidewire), then re-advance PAST the stuck point. This is the micro-recovery: get stuck, ease the slack, go.
  - HARD recovery: same but with a big pullback (> 8 mm).
  - GRIND-THROUGH: passes the stuck point with < 1 mm retraction (brute force — the stuck-grinder ceiling behavior).
  - UNRECOVERED: never passes the stuck point before the episode ends.
It reports rates per 400-episode window plus success-vs-failure retract depth. What we want to accomplish: SOFT rate RISING across windows and higher in successes than failures. In v2 the equivalent cycles FADED across the run (~86% -> 58% -> 28% by thirds — noise-distilled away); v3a's E1b+E3 exist to reverse exactly that fade. Absolute rates depend on detector thresholds (tunable at the top of the script) — judge the TREND across windows, not the absolute number. If SOFT rate falls monotonically for 3+ consecutive windows while eval stagnates, log ALERT: distillation-fade recurring.

```python
#!/usr/bin/env python3
"""Soft/hard recovery-rate tracker (read-only). RL_IMPROV_16 E8-companion.

Parses per-worker step logs (diagnostics/logs_subprocesses/worker_*.log)
and reports, per 400-episode window (chronological across workers):
stuck-episode share, soft/hard/grind/unrecovered rates, and retract-depth
split by episode outcome. Run inside the training container:
    docker exec -i <container> python3 - < monitoring/recovery_tracker.py
"""
import glob
import os
import re

# ---- run location (edit GLOB for a different run) ----
BASE = "/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben"
GLOB = "2026-*_rcca_procedural_v3a"
# ---- detector thresholds (mm / steps; judge TRENDS, not absolutes) ----
STALL_EPS = 0.3     # progress gain below this counts as stalled
PUSH_MIN = 2.0      # commanded gw translation above this = "pushing"
STUCK_STEPS = 12    # consecutive stalled-while-pushing steps => stuck event
RETRACT_MIN = 1.0   # min gw retraction (mm) to count as a retract
SOFT_MAX = 8.0      # retraction <= this = SOFT recovery; above = HARD
PASS_EPS = 1.0      # must exceed pre-stuck max progress by this to recover
WINDOW = 400        # episodes per report window

runs = sorted(glob.glob(os.path.join(BASE, GLOB)))
assert runs, "no run dir matching " + GLOB
run = runs[-1]
logs = sorted(glob.glob(os.path.join(run, "diagnostics/logs_subprocesses/worker_*.log")))
assert logs, "no worker step logs in " + run

F = {k: re.compile(k + r"=([-0-9.a-z\[\],]+)") for k in
     ("ep", "ep_step", "wall_time", "proj_s", "term", "trunc")}
INS = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
CMD = re.compile(r"cmd_action=\[([-0-9.]+),")

episodes = []  # (start_wall, success, events list, had_steps)

def close_episode(st):
    if not st["n"]:
        return
    # close any open stuck event as unrecovered
    if st["stuck"]:
        st["events"].append(("unrecovered", st["retract"]))
    episodes.append((st["t0"], st["success"], st["events"]))

for path in logs:
    st = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "EPISODE_START" in line:
                if st:
                    close_episode(st)
                st = {"n": 0, "t0": None, "success": False, "events": [],
                      "maxp": -1e9, "stall": 0, "stuck": False,
                      "gw_peak": 0.0, "gw_min": 0.0, "retract": 0.0, "p0": 0.0}
                continue
            if st is None or " STEP | " not in line:
                continue
            m_ins, m_cmd = INS.search(line), CMD.search(line)
            m_prog = F["proj_s"].search(line)
            if not (m_ins and m_cmd and m_prog):
                continue
            try:
                gw = float(m_ins.group(1))
                cmd0 = float(m_cmd.group(1))
                prog = float(m_prog.group(1))
            except ValueError:
                continue
            st["n"] += 1
            if st["t0"] is None:
                m_t = F["wall_time"].search(line)
                st["t0"] = float(m_t.group(1)) if m_t else 0.0
            if st["stuck"]:
                # track deepest retraction, look for pass-through
                st["gw_min"] = min(st["gw_min"], gw)
                st["retract"] = max(st["retract"], st["gw_peak"] - st["gw_min"])
                if prog > st["p0"] + PASS_EPS:
                    r = st["retract"]
                    kind = ("grind" if r < RETRACT_MIN
                            else "soft" if r <= SOFT_MAX else "hard")
                    st["events"].append((kind, r))
                    st["stuck"] = False
                    st["stall"] = 0
            else:
                stalled = (prog < st["maxp"] + STALL_EPS) and (cmd0 > PUSH_MIN)
                # decay instead of hard reset: tolerates brief command
                # sign-flips (noise) inside an otherwise-stuck push phase
                st["stall"] = st["stall"] + 1 if stalled else max(0, st["stall"] - 2)
                if st["stall"] >= STUCK_STEPS:
                    st["stuck"] = True
                    st["p0"] = st["maxp"]
                    st["gw_peak"] = gw
                    st["gw_min"] = gw
                    st["retract"] = 0.0
            st["maxp"] = max(st["maxp"], prog)
            if "term=True" in line and "trunc=False" in line:
                st["success"] = True
    if st:
        close_episode(st)

episodes.sort(key=lambda e: e[0])
n = len(episodes)
print("RECOVERY TRACKER: run=%s episodes=%d (window=%d)" %
      (os.path.basename(run), n, WINDOW))
hdr = ("window       eps  stuck-eps  events  soft%  hard%  grind%  "
       "unrec%  succ%  ret-depth succ/fail (mm)")
print(hdr)
for b in range(0, n, WINDOW):
    w = episodes[b:b + WINDOW]
    ev = [e for (_, _, evs) in w for e in evs]
    stuck_eps = sum(1 for (_, _, evs) in w if evs)
    succ = sum(1 for (_, s, _) in w if s)
    tot = max(1, len(ev))
    cnt = {k: sum(1 for (kk, _) in ev if kk == k)
           for k in ("soft", "hard", "grind", "unrecovered")}
    dep_s = [r for (t, s, evs) in w if s for (_, r) in evs]
    dep_f = [r for (t, s, evs) in w if not s for (_, r) in evs]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("%5d-%-5d %5d  %6d     %5d  %5.1f  %5.1f  %6.1f  %6.1f  %5.1f"
          "   %.1f / %.1f" %
          (b + 1, b + len(w), len(w), stuck_eps, len(ev),
           100.0 * cnt["soft"] / tot, 100.0 * cnt["hard"] / tot,
           100.0 * cnt["grind"] / tot, 100.0 * cnt["unrecovered"] / tot,
           100.0 * succ / max(1, len(w)), avg(dep_s), avg(dep_f)))
print("GOAL: soft% RISING across windows and retract-depth succ < fail; "
      "v2's cycles faded ~86->58->28 across run thirds — that fade "
      "recurring (3+ windows monotone down + flat evals) = ALERT.")
```

Notes for the machine-2 session:
  - The step logs only exist if the run was launched with the v3a launcher (STEP_LOG_DIR is set there). If worker_*.log files are missing, say so in the pass log instead of failing.
  - Detector sensitivity: the stuck detector needs sustained pushing (cmd_action[0] > PUSH_MIN with stall persistence). On a PRETRAINED policy (v3a) this fires normally. If events-per-window come out near zero on a healthy directed policy, lower PUSH_MIN to 1.0 or STUCK_STEPS to 8 and note the change in the pass log so trends stay comparable — never compare absolute rates across different thresholds. (Validated on machine 1: parses real worker logs end-to-end; the RLPD run showed ~0 events precisely because its noise-dominated policy never pushes 12 steps straight.)
  - Success detection = a step with term=True and trunc=False (EPISODE_END lines carry no success flag).
  - Both scripts are read-only. Never edit files mounted by the running container; if a script needs a tweak, copy it to a scratch path first.
  - Log every pass to saved/monitor_rcca_v3a.md with a dated header and an explicit verdict; include the recovery table verbatim so trends are visible pass-to-pass.
-------


First we need to take a deeo dive into the trained model and our training specifics and health ; how come H0 was so effective and then the model basically learned nothing after that ; after looking at all the eval successes and failures it felt that about half of failures were pretty close to success but just ran out of steps ; other half got stuck at RCCA and RVA intersection ;  also the failures that ran out of steps almost always went into RVA first and then instead of retracting they use other fork within RVA (going elsewhere) to make a u-turn within RVA and then  come back and thread RCCA again ; why the speed never improves at straight sections; can something like separate stuck training help like that proposed in v3b (E series) models on other machine. Also , I think in all these RL models we are stuck between 2 modalities either go fast and can't make fine turns or go slow and get trapped in local optima since there are two modes to navigation where there is crunch and where it is not , hence a singular mode of training /model is not able to encode it; since we are using planned path we can better encode it in our obsevation about the 2 modes (and also include wrong pts in one of those modes) or use two different models (actors) for classified (easy with path planning) 2 different modes but getting trained in the same run (2 different buffers getting filled); what would be the best way to proceed because if we can unlock RCCA / RVA instersection with learned recoveries we can unlock huge gains. Investigate all this thoroughly

This deserves the full treatment — your two-modality hypothesis in particular maps directly onto the roadmap's "bimodal commit-vs-retract is unrepresentable by a unimodal tanh-Gaussian" finding, and the failure taxonomy you observed is quantifiable from the eval step logs. Launching a deep-dive workflow: eval forensics, training-health autopsy, mode-hypothesis testing, and a design synthesis:


Eval forensics — parses all five eval windows from the worker step logs to quantify your observations: the timeout-near-target vs stuck-at-junction split among failures, the RVA U-turn pattern (entered RVA → forked around inside it → re-threaded RCCA, with steps wasted counted), and whether success speed or straight-section speed moved at all between eval1 and eval5.
Training-health autopsy — the "why did nothing compound" question: alpha-cycle phase at each eval's timestamp (were the good evals just low-alpha phases?), residual magnitude evolution from the policy snapshots (is the residual stuck near zero, growing uselessly, or oscillating?), and — the key one — whether the critic prefers retraction at high-slack states while the actor ignores it, which would be the λ-transmission failure from the LCCA era recurring in SAC form.
Bimodality evidence — your two-mode hypothesis, tested quantitatively: does optimal |action| actually differ between crunch steps (contact/slack/junction-proximity) and free steps (open trunk), does the current policy's action distribution differentiate the modes at all, and how many steps would full-speed straight sections save — enough to convert the timeout half of failures?
Design survey — scores every candidate fix (mode-flag observation from the planned path, your two-actor dual-buffer MoE, GMM/QC-FQL bimodal heads from roadmap P3, the v3b/E4 stuck-restore curriculum, reduce-slack potential, alpha-ceiling, and step-budget/speed shaping for the timeout half) on which failure-half it addresses, cost, risk, and what composes cleanly into one relaunch.
Then a synthesis agent writes saved/p2a_deep_dive/DEEP_DIVE_P2A_V1.md with the causal story, the decision matrix, and a concrete recommended P2a-v2 bundle with expected-gain arithmetic — and an adversarial auditor checks every headline number and the arithmetic before it's final. I'll bring you the executive summary and the recommended bundle when it lands.

The deep dive is done — and it found something that reframes everything, including my own earlier analysis. Full report in saved/p2a_deep_dive/DEEP_DIVE_P2A_V1.md (adversarially audited, 6 agents).

The bombshell: the run never deployed a trained weight. All 142 policy snapshots and all 7 checkpoints hash byte-identical to initialization. Two independent infrastructure breaks: (a) the trainer→worker/eval weight sync never fired once in 1.45M explore steps; (b) my deadlock guard from e3bf215 misfires — the 1800s trainer-result deadline is shorter than this run's ~40-minute explore cycles, so it "restarted" a healthy trainer into fresh init 32 times, every ~23.4k updates. That cadence is exactly the "alpha limit cycle" I've been reporting across passes — it was a restart sawtooth (init α=1.0 → anneal → wipe → repeat), not an entropy pathology. The 4-eval "reward decline" was an aggregation artifact. Every eval executed the identical init policy (residual ≈ 0 = pure heuristic), so 45.9→48.0→49.0→46.9→45.9→45.9 is episode-mix noise around a pooled H₀ ≈ 46.9% (85 of 98 seeds are deterministic; repeat-eval SE ≈ ±1.8pp). I was wrong to narrate "learning under the surface" — the composition shifts were noise under a constant policy. P1's snapshots did evolve (distinct md5s), but its same-period "alpha cycle" now needs re-examination too.

The silver lining — ~590 episodes of clean H₀ characterization answering your questions:

Failure mass is NOT what it looked like: 58.3% is deep-RVA capture — enter RVA early, spend ~505/600 steps inside with 0.0 mm retraction (commit-without-retract, the untransmitted-retract pathology again); ~34% near-target grind; genuinely at-the-door only ~4%. U-turns are a red herring (16% of failures, median 4 steps, successes do it too), and RVA-before-RCCA-then-success never occurs (0/259).
Straight speed is frozen at the heuristic's compromise: ~5 mm/s commanded = 17% of max, identical in every eval. The counterfactual sweep from explore noise confirms your bimodal target: free-mode optimum ≈ 25–30 mm/s, crunch optimum ≈ 0. But speed alone converts only ~20% of timeouts — the dominant lever is breaking the RVA capture mode.
Ceiling arithmetic if the bundle works: ~62–73% vs H₀ 47%.
Recommended path (bundle in the doc): (1) infra repairs first — fix the spurious deadline restart + weight sync, with invariant alarms (checkpoint-hash-must-change, sync-counter) and a blocking smoke gate; (2) α-ceiling flag; (3) wrong-branch/path-context observation (~6 dims, attacks the 58% mode — your "encode the modes via planned path" proposal, adopted); (4) stuck-restore curriculum if machine-2's pool is ready; (5) success@900 secondary eval metric; (6) a scripted wrong-branch-escape ablation as the no-RL cost floor. Two-actor MoE rejected; GMM head is the v3 backup. Your call on killing the current (init-policy) run and starting the repairs.

--------Here's the full parked queue, organized by what gates each item:

Gated on v1b's eval3–5 (the active experiment decides):

Reduce-slack potential reward — telescoping potential on guidewire slack, same non-farmable delta form as the existing buckle potential (buckle_reward.py family), directly paying for ease-slack-then-advance.
Stuck-restore curriculum — episodes started from harvested stuck checkpoints (the E4 mechanism), so recovery states get dense practice instead of incidental visits.
Tighter alpha ceiling (e.g. --log_alpha_max −1.2 to −0.7) — added on this run's evidence, to damp the limit cycle that P1 and P2 both showed.


P2a-v2 bundle (DEEP_DIVE_P2A_V1.md): wrong-branch/path-context observation (~6 dims, attacks the deep-RVA-capture failure mode — your mode-encoding idea), E4/v3b stuck-restore curriculum (--restore_prob 0.3 from the harvested stuck-state pool, if machine-2's pool is ready), success@900 secondary eval metric, and the scripted wrong-branch-escape ablation as the no-RL cost floor.
GMM residual head (P3 backup) — activates only if the soft-recovery transition stalls, i.e. evidence that one Gaussian can't hold both modes. Two-actor MoE stays rejected unless conditioning and GMM both saturate.
Infra/protocol fixes queued for the next relaunch (whatever its reason):
3. P0 eval-state hard reset — the orbit bug: per-episode device-state reset so eval measures seeds, not sequence position. Restores cross-run comparability and a single meaningful H₀.
4. Launch seeding + policy_0.pt dump — reproducible inits.
5. Probe re-arm after trainer restart (secondary bug from the deep dive), and eval accounting from episode_summary.jsonl instead of worker logs (which drop each worker's final episode).

Needing your explicit approval (reward changes are frozen):
6. Reduce-slack potential reward — telescoping, non-farmable, pays directly for the ease-slack behavior. Notably: the policy is now discovering soft recovery without it, so this may have become an accelerant rather than a necessity — the eval3–5 trend tells us.

The larger roadmap, in order:
7. P2b — DAgger student distillation (teacher → deployable obs-only policy, short history replacing ep_step) — gates on the teacher beating targets stably.
8. Machine-2 v3a Tier-A run (E1b advantage-norm + E2 aux z-scoring + E3 stuck-lane, AWAC control arm) — handoff ready, its stuck-harvest also feeds item 1's restore pool.
9. Multi-mesh generalization (per-worker mesh families beyond procedural RCCA) and the obs-pruning list from the audit (~21 dims).
10. Paper: PAPER_PLAN_NEUROVASC_RL.md's [X%] gates plus two corrections we owe it — the C3 "diversity vs competence" numbers update, and the eval-protocol section rewritten for the orbit discovery (which is itself publishable as a benchmark-hygiene finding).

The near-term decision tree stays simple: eval3 recovering into the 60s with soft-share retained → let v1b run deep, then bundle 1+3+4 into a v2 relaunch at a natural checkpoint; eval3 declining → same bundle but sooner, with the alpha/replay investigation added. Nothing needs a decision from you until eval3 lands (~4–5h).


----------


Gated on v1b's eval3–5 (the active experiment decides):

P2a-v2 bundle (DEEP_DIVE_P2A_V1.md): wrong-branch/path-context observation (~6 dims, attacks the deep-RVA-capture failure mode — your mode-encoding idea), E4/v3b stuck-restore curriculum (--restore_prob 0.3 from the harvested stuck-state pool, if machine-2's pool is ready), success@900 secondary eval metric, and the scripted wrong-branch-escape ablation as the no-RL cost floor.
GMM residual head (P3 backup) — activates only if the soft-recovery transition stalls, i.e. evidence that one Gaussian can't hold both modes. Two-actor MoE stays rejected unless conditioning and GMM both saturate.
Infra/protocol fixes queued for the next relaunch (whatever its reason):
3. P0 eval-state hard reset — the orbit bug: per-episode device-state reset so eval measures seeds, not sequence position. Restores cross-run comparability and a single meaningful H₀.
4. Launch seeding + policy_0.pt dump — reproducible inits.
5. Probe re-arm after trainer restart (secondary bug from the deep dive), and eval accounting from episode_summary.jsonl instead of worker logs (which drop each worker's final episode).

Needing your explicit approval (reward changes are frozen):
6. Reduce-slack potential reward — telescoping, non-farmable, pays directly for the ease-slack behavior. Notably: the policy is now discovering soft recovery without it, so this may have become an accelerant rather than a necessity — the eval3–5 trend tells us.

The larger roadmap, in order:
7. P2b — DAgger student distillation (teacher → deployable obs-only policy, short history replacing ep_step) — gates on the teacher beating targets stably.
8. Machine-2 v3a Tier-A run (E1b advantage-norm + E2 aux z-scoring + E3 stuck-lane, AWAC control arm) — handoff ready, its stuck-harvest also feeds item 1's restore pool.
9. Multi-mesh generalization (per-worker mesh families beyond procedural RCCA) and the obs-pruning list from the audit (~21 dims).
10. Paper: PAPER_PLAN_NEUROVASC_RL.md's [X%] gates plus two corrections we owe it — the C3 "diversity vs competence" numbers update, and the eval-protocol section rewritten for the orbit discovery (which is itself publishable as a benchmark-hygiene finding).

The near-term decision tree stays simple: eval3 recovering into the 60s with soft-share retained → let v1b run deep, then bundle 1+3+4 into a v2 relaunch at a natural checkpoint; eval3 declining → same bundle but sooner, with the alpha/replay investigation added. Nothing needs a decision from you until eval3 lands (~4–5h)
-------

so we have learned a lot from machine 2 v3a experiments ; stuck lane files (v3b) do not work because the way it is programmed right now , when wires finally get stuck , wires are too colied up into knots and loops and they aren't recoverable at that point , you will have to retract 100s of steps before nay recovery is possible , so we are not extactly looking for truly stuck states but the states before where we recover from or never go into those states ,so  another idea emerged , we can classify the steps (actual crunch steps that lead to a successful passage via crunch easily based on nearby planned path parameters - as high as with 90% success rate; similar classifier can also recover priviledge teacher force/contact values that can be use om student's actor network) that we want to resample for this another mode. These steps generally use both cathetar and guidewire and both are getting used and are at close distance (their tips) with each other .  First we need to make such classifier 

--------


Which features matter (permutation importance on the transfer task)
Permutation importance — how much unseen-AUC drops when each feature is shuffled — gives a far cleaner ranking than univariate (which was drowned in redundant tracking dims):

dim	feature	Δauc when removed
89	gw_slack	+0.237 ← dominant
77	cath_offset_z (catheter−guidewire tip offset)	+0.123
58	arc_past_daughter	+0.061
78	gw_cath_gap (insertion difference)	+0.054
93	local_radius	+0.045
39, 59, 17…	a tail of tracking + guidance dims	+0.02–0.04 each
gw_slack is the contact predictor, by a wide margin — nearly 2× the next feature. That's physically exactly right: slack (inserted length minus tip progress) is accumulated buckling, which is what wall contact produces. Second and fourth are the two catheter-vs-guidewire relative-position features, then local radius. So the features that "really matter" are the buckling/relative-geometry ones — not the tip-tracking dims that topped the univariate list (those ranked high only because they're mutually redundant, not individually informative).

But feature selection didn't raise accuracy — it compressed it
feature set	hard-transfer AUC (train seed → test unseen online regime)
full-97 (baseline)	0.787
top-8 by importance	0.670 (too few — loses the distributed tail)
top-15 by importance	0.788
top-25	0.788
physics hand-picked (12)	0.774
------




Now our next big problem is to fix the mesh vessel (RCCA) generation that renders most RCCAs unpassable . Invesitage thoroughly how to fix this ; There are three critierias ; it should be vaied enough (which I think it is) ; it should be passable; it should be almost as hard as the real one (not significantly harder and not significantly easier ); Investigate thoroughly , implement and then verify (maybe run these evals with  generated mesh vessel identical to real one ; should give identical result)


-------


ok , coming back from this detour ; the last run we checked the correct real patient mesh vs v1b best model ; now run v1bp best model vs corrected real patient mesh; after that as we can see the eval rates were flawed because of one generated mesh bug, so find the actual best models (based on explore succ%) , as we can see from the report you just created , they come much later ; do both for v1b and v1bp; so three runs more total


---------


Now our next big problem is to fix the mesh vessel (RCCA) generation that renders most RCCAs unpassable . Invesitage thoroughly how to fix this ; There are three critierias ; it should be vaied enough (which I think it is) ; it should be passable; it should be almost as hard as the real one (slightly harder will do but not significantly easier one); Investigate thoroughly , implement and then verify (maybe run these evals with  generated mesh vessel identical to real one ; should give identical result)
