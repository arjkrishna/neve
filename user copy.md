
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

First we indentified previously a couple of bugs in our reward setup ; we need to fix it and to improve it further we need to make a few more modifications ; right now all our RLs come with a standard step penalty that means they are getting 0.001 for every step at the beginning and also at RCCA daughter and (11) when they are much closer to the target which sometimes encourages agent to be slow at the start and remain close since there is no much cross_track penalty and .001 penalty for 600 steps doesn't count to much so we need reformulation where they are getting 0.007 (closer to start) - 0.002 (closer to (11)) penalty, 0.007 penalty in (0) (past bifurcation), no penalty in (11) , 0.005 step reward in RCCA and 0.007 step penalty in RVA ; right now +1/-1 are awards at making right decision at every point, for each juncton make it -0.05 (that they can get again and again, when theaded wrongly) / +1 (once per fork per episode) so that they are not afraid and encouraged to explore to go for +1 ; obviously we will be running new heusristics / heatup for our new setup , for heuristics we will be aiming for 100 successful episodes (so for 100 succ. it will be running probably upto 400/450 heusistic episodes) as AWAC closely mirrors heuristic and we will be starting a little before and outside (11) (SOFA restore), one question I had, we will be running heuristics from z= 345 as we are now but during RL we would be using SOFA restore just outside (11), would we append the saved steps and reward from that episode and continue this episode forward and q1 mean would reflect expected reward from the start point and saved heuristic steps but optimized steps after the start point (within daughters) because this would be ideal.  The other question I had can we also implement one of these:
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