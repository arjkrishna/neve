This is a very comprehensive analysis. Now let me think about what I need to explain to the user:

Why early wrong branch timeouts happen - at the first bifurcation (21.8,-22.5,79.2) at ~70mm insertion

What targets are being used - the EPISODE_START lines don't show targets explicitly, but the terminated=True episodes show targets at (14.9,12.5,572.0) and (47.6,34.5,430.1) - both seem to be deep targets. The heuristic is navigating toward these but always takes the wrong branch at bifurcation (21.8,-22.5,79.2) early on.

The d_wrong == d_corr pattern - I'm noticing that over 95% of the time, the wrong and correct points are identical with matching distances. This suggests the cache is marking the same bifurcation point as both a wrong entry and a correct entry simultaneously. Since a bifurcation connects multiple branches, it makes geometric sense that one branch leads to the target while another is a dead end—so the same point serves both roles depending on which branch you take.

But here's the issue: if the points are truly identical, then the direction vectors computed from them should also be identical, which would make the wrong direction features match the correct direction features. That shouldn't happen if they're supposed to represent different navigation choices.

I think I've been confusing what these entries actually represent. The wrong_entry and correct_entry aren't supposed to be the same bifurcation point — they should point toward different branch corridors. But the cache is storing the branching points themselves, not the actual corridor directions. Let me look back at what PathProjectionCache is actually classifying.

So a branching point that connects to both correct and wrong branches gets listed in both categories, which means the distance features end up nearly identical for both. That's the real problem here — the observation can't distinguish between the correct and wrong branches because it's measuring distance to the same junction point. A better approach would be to use the centroid of the correct branch itself, somewhere inside that branch rather than at the bifurcation, so the direction features actually point toward different locations.

Why the heuristic enters wrong branches - The CenterlineFollowerHeuristic follows the path centerline, but at bifurcations it lacks sufficient look-ahead or rotation awareness, causing it to consistently take the wrong branch without explicit bifurcation handling logic.

Fixing early wrong branch timeouts - I need to check which targets are being sampled, since some might require more complex navigation early on. The heuristic's random retraction strategy fails because the catheter is still inserted about 70mm when it takes the wrong branch, and the retraction steps only move it back around 50mm maximum, so the catheter can't fully escape the wrong branch before running out of steps.

Looking at the abort logs more carefully, when episodes terminate at step 34, the heuristic enters the wrong branch around step 24 with the catheter inserted at roughly 70mm. But checking the actual abort data, the action values are still positive even after attempting retraction, which means the catheter is continuing to insert rather than retract as intended.

The core issue is that the retraction sampling uses a uniform distribution from -10 to 5mm, which gives a 33% chance of pushing forward instead of pulling back. Combined with a grace period that's too short for random retraction to reliably work, the agent gets stuck. Additionally, the direction features for wrong versus correct branches are degenerate since they both point to the same bifurcation junction, making it impossible for the agent to learn which way to go. is the entry point for BOTH the correct path branch AND wrong branches. So PathProjectionCache correctly classifies it into both wrong_branch_entries and correct_branch_entries. The distance to the nearest wrong entry == distance to nearest correct entry because they're the same physical point.

The direction features (wrong_dir, correct_dir) would then be identical too, making features 9-10 redundant with 12-13. This reduces the effective new information in the 14-dim observation to just: dist_wrong_entry (=dist_correct_entry), and a single direction vector. Not much information about WHICH branch to take.

Why early wrong branch timeouts specifically:
The first bifurcation is at ~60-80mm insertion depth. The heuristic reaches this with inserted=[70mm, 56mm]. On entering the wrong branch, the random retraction strategy tries uniform[-10, 5] per step. Since many samples are positive, the tip doesn't retract enough in 10 steps to clear the bifurcation. The abort fires at off_br_steps=10.

Proposed fixes:

I need to address the retraction strategy first—changing it to ensure negative values so the tip actually backs out. I should also increase the grace period for off-branch detection to give more time for recovery. The key insight is that the direction features are degenerate because they point to the same bifurcation point, so I need to compute directions toward distinct points deeper inside each branch corridor instead. Finally, I should verify that all four targets are being sampled during training.

----------------------------


Now I have a comprehensive picture. Let me analyze the key findings:

Only 3 unique wrong_pt coordinates appear:

(21.8,-22.5,79.2) — first bifurcation at ~70mm insertion (early failures)
(18.6,13.6,383.2) — appears as wrong_pt in the middle of long episodes (around step 100, ~260mm)
(46.3,34.2,390.3) — second/deeper bifurcation where long episodes fail (~420mm)
(47.6,34.5,430.1) and (14.9,12.5,572.0) — success targets for different episodes
The same wrong_pt (18.6,13.6,383.2) appears at step 100 in ALL long episodes, but they don't abort there — they continue and eventually either fail at (46.3,34.2,390.3) or succeed, indicating at least 2-3 bifurcation points along the path.

The heuristic's core issue: it enforces a minimum forward translation threshold that's preventing proper navigation through these critical points.

Why the wrong branch gets entered early: The heuristic encounters the first bifurcation at (21.8,-22.5,79.2) where the path tangent should guide toward the correct branch, but stochastic noise in rotation sometimes pushes the tip into the wrong branch instead. Once there, the retraction strategy attempts recovery but often fails.

Examining whether the wrong and correct branches have equivalent distances...

Actually, they do — at bifurcation points, both branches share the same entry coordinates, so the distance and direction features are identical. This is a core problem: the observation space provides no signal about which branch to take at these junctions.

Looking at target information: The logs don't record the target at episode start, but analyzing the wrong branch points reveals a pattern. Short episodes consistently show the same behavior regardless of target, while longer episodes diverge around step 100 when some trajectories reach different points. Successful runs end at one of two target locations.
Now I'm thinking about what could actually fix the early wrong branch problem. The fundamental issue is that the heuristic can't look ahead to distinguish between the branches at their shared entry point.

The heuristic itself doesn't handle bifurcations explicitly — it just follows the path tangent and applies cross-track corrections. When a bifurcation occurs, the tangent direction becomes ambiguous depending on which polyline segment is closest, and if the projected segment is past the bifurcation, the tangent might point the wrong way.

Beyond that, the rotation noise being added can push the tip into the wrong branch when precision is critical at these decision points. And the retraction strategy isn't optimal for recovering from these mistakes.

Looking at the retraction logic, the current approach has a 33% chance of moving forward when it should always be retracting. The magnitude is also insufficient — it needs about 70mm of retraction to clear the first bifurcation, but the current strategy only manages around 50mm over 10 steps. A more aggressive retraction of 10-15mm per step for 2-3 steps would actually clear the obstacle.

There's also a feature degeneracy problem: at bifurcations, the wrong branch point and correct branch point are identical, so the observation features that should distinguish them end up being the same. The agent can't learn which direction is correct when both options look identical in the feature space.

I'm proposing four fixes: first, change the retraction to always be negative (between -10 and -1) instead of allowing forward movement; second, increase the grace period for retraction from 10 to 20-30 steps to give more time; third, log the target information at episode start so we can track which targets are failing; and fourth, fix the branch entry point coordinates by using distinct points for wrong versus correct branches instead of reusing the bifurcation point.

I'm also considering a deeper fix where we pre-compute the required rotation before reaching a bifurcation and apply it proactively, though this would need access to the correct branch direction. There's something interesting happening at step 100 where corr_pt and wrong_pt diverge in certain cases.

Looking at the data, when the agent is 130-140mm away from a bifurcation, the nearest correct entry point can be 7mm farther than the wrong one, which actually suggests the PathProjectionCache is functioning as intended. The most impactful fix would be adjusting the retraction direction in the heuristic policy to always pull backward rather than allowing forward movement, which should significantly reduce early wrong-branch aborts.

Beyond that, I'd increase the grace period for off-branch detection to give the agent more time to navigate through bifurcations, and potentially store points slightly into each branch rather than at the junction itself so the correct and wrong directions are more clearly distinguished.

----------------------


Now I have a very clear picture. Let me summarize:

Episode stats (668 total):

554 wrong_branch_timeout
102 wire_fold_stall
9 success (term=True) + some no-abort (3 confirmed term=True + 6 more?)
Actually: 668 total. 554 + 102 + 9 = 665. The remaining 3 are the term=True successes.
Actually wait, the last command showed "9" for episodes without heur_abort.

So there's a discrepancy here - I found 9 episodes without heur_abort in the logs, but only 3 of those are confirmed term=True successes. The other 6 without heur_abort could be episodes that hit max_steps truncation instead of terminating successfully, which would also result in no heur_abort field in the EPISODE_END line.

Looking at the breakdown across all 668 episodes: wrong branch timeouts dominate at 554 total (83%), split between early failures at the first bifurcation (396 episodes, 59%) and later failures (158 episodes, 24%). Wire fold stalls account for 102 episodes (15%), leaving just 9 episodes (1.3%) with no abort recorded.

Only 3 episodes actually succeeded. The divergence between wrong and correct points appears in just 4 STEP lines across the entire worker log, which is surprisingly sparse. This divergence happens around 260mm insertion depth when approaching the second bifurcation, and it's actually the expected behavior since the nearest wrong branch at that point is genuinely a different direction. The path structure has exactly two major bifurcation points: the first around 70mm insertion depth at step 25, and the second much deeper in the insertion sequence.

The early failure rate of 59% traces back to that first bifurcation being extremely sensitive to the heuristic's rotation noise. The noise can easily push the agent down the wrong branch, and once committed, the forward movement probability makes recovery nearly impossible. The rotation noise at bifurcations is the primary culprit here.

When the catheter is already inserted deeply at around 70mm, attempting to retract becomes much harder because the stiff catheter resists backing out. Even with 10 grace steps, the retraction strategy of uniform(-10, 5) only gives a 33% chance of moving backward each step, resulting in an average net retraction of about -3mm per step—nowhere near enough to escape from a 70mm insertion depth.

The immediate fix is to change the retraction strategy to always move backward instead of having that forward movement probability.

For the longer-term improvements, I'd increase the grace steps buffer to give more time for retraction, add target information to the episode logs for better debugging, and fix the degenerate case where the wrong and correct directions look identical at the bifurcation junction by storing a point further along the branch centerline instead of just the junction itself.

There's also a potential look-ahead approach where I could compute the rotation needed to align with the correct branch before actually reaching the bifurcation, applying it proactively to reduce wrong-branch entries. But the quickest win here is really just fixing that retraction direction—that alone should make a huge difference.

-------------------------------------------


Here's the full picture across 668 completed episodes:

Episode breakdown:

Outcome	Count	%
wrong_branch_timeout, ≤50 steps (first bifurcation)	396	59%
wrong_branch_timeout, >50 steps (deep bifurcation)	158	24%
wire_fold_stall	102	15%
Success (term=True)	3	0.4%
Vessel structure (inferred from coordinates):

There are exactly 2 bifurcations:

Bifurcation	3D coords	Insertion depth	Step ~#
Bif 1	(21.8, -22.5, 79.2)	~70mm	step 25
Bif 2	(18.6, 13.6, 383.2) / (46.3, 34.2, 390.3)	~390mm	step 160
Two targets: (47.6, 34.5, 430.1) at ~430mm, and (14.9, 12.5, 572.0) at ~570mm. EPISODE_START doesn't log which target was assigned — we can't break down failure by target.

Why early wrong-branch (59% of all episodes):

The heuristic takes the wrong branch at bif 1 stochastically (rotation noise, noise_std_frac=0.1). Then the retraction strategy in heuristic_policy.py uses uniform(-10, 5) — which has a 33% chance of going forward on any step. 10 grace steps × ~−3mm average net retraction = only −30mm. The tip entered the wrong branch at ~70mm insertion, so it needs ~−20mm to clear the junction. But with 1-in-3 steps going forward, it often doesn't make it.

Why d_wrong == d_corr 95%+ of steps:

At both bifurcations, the same 3D junction point is simultaneously the entry to the correct branch AND the wrong branch. PathProjectionCache._build_entry_points() correctly classifies it into both lists — but since the coordinates are identical, wrong_dir == correct_dir (features 9-10 == features 12-13). The 6 new observation dims reduce to effectively just one scalar (distance to next junction). The agent can't learn which way to turn at the junction from these features alone.

What can be done:

Quick wins (1-line fixes):

Fix retraction direction — heuristic_policy.py: change uniform(-10.0, 5.0) → uniform(-10.0, -1.0). Guarantees retraction every step. Would cut early aborts substantially.

Increase grace steps — env5.py: OFF_BRANCH_GRACE_STEPS = 10 → 20. More time to actually clear the junction.

Log target in EPISODE_START — add target name/coords to the log line so future runs let us correlate per-target failure rates.

Observation fix (medium):

Branch interior points — In PathProjectionCache._build_entry_points(), instead of storing the bifurcation junction coordinate, store a point 10-15mm inside each branch along its centerline. Then wrong_pt ≠ corr_pt at the same bifurcation — the agent can see which direction is the wrong branch vs the correct one.

Heuristic improvement (harder):

Pre-bifurcation rotation — When d_corr < threshold (approaching a bifurcation), compute the rotation needed to align the device with the correct branch direction and apply it. The CenterlineFollowerHeuristic.get_action() already has cross-track + tangent alignment, but it uses the projected segment on the path — which at a bifurcation might be ambiguous depending on which side of the junction the projection falls on.

The highest-leverage fix is #1 (retraction direction) — it's one character and would eliminate a large fraction of those 396 early aborts. Want me to apply 1 + 2 + 3 now?


------------------------------------------------------



Confirmed — you're right to suspect it. In pid 578 ep 7:

242 of 539 off-branch debug steps (45%) had reward > -0.101 → arclength progress was strictly positive while on_br=0.
Several steps had positive net reward while off-branch (e.g. step 190: +0.024, step 191: +0.044) — the arc-progress term overcame the combined −0.1 wrong-branch + −0.001 step penalty.
And look at the off_br counter trajectory across the 14 off-branch INFO samples:


step:   200  250  300  350  400  450  500  550  600  650  700  800  850  900
off_br:  4    8   10   10    1    3    8    5    1    2    1    6    5    5
The counter keeps resetting and re-accumulating — meaning on_br is flipping back to 1 repeatedly, which is why OFF_BRANCH_GRACE_STEPS=20 never triggered. The tip is being alternately classified as on/off within tens of steps while insertion stays at ~490mm.

Two plausible explanations, both consistent with what you're seeing:

False-negative branch classification: PathProjectionCache.is_on_correct_branch() uses a KD-tree over all branch centerline points and picks the nearest branch. Near bif2, the two daughter branches (e.g. LCCA & RCCA proximal segments, or LVA & LCCA) can be within millimeters of each other. Tiny tip position jitter flips the nearest-branch winner. When the flip says "wrong," on_br=0 fires, but the tip is still geometrically close to the correct polyline → arclength progression continues normally.

Genuine parallel wrong-branch traversal: the wrong daughter branch is nearly parallel to the correct one near bif2, so even with true off-branch state, the projection onto the correct polyline slides forward.

Distinguishing them would mean logging cross_track_dist alongside on_br (when cross_track is small, flip is suspicious). The KD-tree flip-flopping shown above strongly points at #1. This matches the P3 concern in RL_IMPROV_4_CHANGES.md lines 458–481 ("branch interior points" fix — store points 10–15mm into each daughter branch to disambiguate junction-neighbor points).


Smoking gun. Look at the wrong_pt and corr_pt columns — they're literally identical at every sample:

Steps 200–250: both = (18.6, 13.6, 383.2) (bif2 junction)
Steps 300–1000: both = (-0.6, 24.5, 416.1) (bif2 distal daughter junction)
That's the P3 degenerate-junction problem documented in RL_IMPROV_4_CHANGES.md:458-481 — PathProjectionCache._build_entry_points() stores the same 3D junction coordinate in both _wrong_branch_entries and _correct_branch_entries (because a bifurcation connects to both correct AND wrong branches). So the direction features the agent sees are identical for "avoid this way" and "go this way."

But more telling: look at the on_br flip pattern while tip is stationary:

step	insert	on_br	off_br	d_corr
300	485.4	0	10	7.0
350	488.0	0	10	5.7
400	493.8	0	1	4.8
600	494.0	0	1	4.6
700	493.5	0	1	4.6
750	491.7	1	0	5.2
800	488.9	0	6	5.0
950	492.3	1	0	5.0
Tip is parked at ~490mm, 4–7mm from the junction, and on_br is toggling between 0 and 1 every few tens of steps despite essentially no position change. The off_br counter keeps resetting because the KD-tree classification flips — that's why OFF_BRANCH_GRACE_STEPS=20 never fires.

Mechanism: PathProjectionCache.is_on_correct_branch() does kdtree.query(tip) over all branch centerline points, takes the nearest, and asks if that branch is in path_branch_set. When the tip is near a junction, tiny vessel-CS coordinate jitter flips which daughter's nearest centerline point wins by micrometers.

So your assertion holds: the arclength reward should not pay out while genuinely off-branch — but the detector is falsely classifying bif2-adjacent states as off-branch, so the reward is legitimately accruing for a tip that's essentially still on the correct path. The flip-flop also explains why the wrong-branch timeout never fires on these deep-stuck episodes (counter keeps resetting).

Fix direction matches what RL_IMPROV_4_CHANGES.md §P3 already proposed: store interior points 10–15mm into each branch along its centerline, not the shared junction coordinate, so the KD-tree has disambiguated entries near junctions. That change would also fix the degenerate wrong_pt == corr_pt observation feature.

--------------------------------

the §4 fix did not actually address the KD-tree flip-flop.

Looking at the code I wrote:

_build_branch_index() at pathcontext.py:203-227 is unchanged. It still stuffs branch.coordinates (all centerline points, including the junction-proximal segments that are within mm of each other across branches) into _all_branch_coords → _branch_kdtree. This is the KD-tree that is_on_correct_branch() queries.
My §4 fix only touched _build_entry_points() (the list of "entry" points stored in _wrong_branch_entries / _correct_branch_entries). Those lists are used only for observation features 8–13 (dist_to_wrong_entry, wrong_entry_dir_{x,z}, etc.), never by the classifier.
So §4 fixed obs-feature degeneracy (wrong_pt == corr_pt), not KD-tree flip-flop. The hysteresis (§7 Fix 2) masks the flip-flop by debouncing the output but doesn't fix the underlying ambiguous classification — that's why pid 160 ep 2 still showed multiple WRONG_BRANCH_ENTRY_PENALTY hits during bif2 oscillation, and why the 1000-step oscillators persist (each off-branch stretch stays under 50 steps due to true raw flips at the junction).

The actual KD-tree fix is to disambiguate the tree itself. Cleanest option: in _build_branch_index(), exclude centerline points within ~15 mm of any branching_point.coordinates — then the KD-tree contains only unambiguous interior points, and tip.query() returns a branch that matches the tip's true corridor. Near the junction all candidates are ~15 mm away, so the nearest-neighbor still picks the branch whose interior vector the tip is aligned with.

Alternative: use per-branch polyline projection (perpendicular distance) instead of nearest-point-on-any-centerline — more robust but heavier refactor.

Want me to apply the exclusion-based fix? Hysteresis + the real KD-tree fix together should cut the 1000-step oscillators significantly.







Option 2 (per-branch polyline projection) is better. Option 1 is less code but has corner cases; Option 2 is correct by construction and reuses an existing utility.

Option 1 (exclusion) — the catches
Short branches break. Any branch < ~30 mm loses all its points from the KD-tree → effectively becomes invisible to the classifier. A daughter branch that's only 20 mm long before its next bifurcation has only ~5 mm of "interior" left after excluding 15 mm at each end. Brittle.
Tip at the junction — all remaining KD-tree points are ≥ 15 mm away, by definition. The nearest-neighbor is still ambiguous because the tip now sits equidistant from multiple branches' first non-excluded points. Option 1 pushes the ambiguity zone outward instead of eliminating it.
Magic number. The 15 mm threshold is a tuning knob. Too small → still ambiguous, too large → short branches vanish. No principled way to pick it.
Option 2 (perpendicular distance to each branch's polyline)
Geometrically correct: for each branch, project tip onto its centerline polyline, take the perpendicular (cross-track) distance, pick the branch with the smallest one.
At the junction point, the two daughter polylines diverge in different directions, so perpendicular distance to each is a well-defined, continuous function of tip position. Tiny tip jitter doesn't flip the winner.
Reuses existing code. eve/util/polyline.py already has vectorized project_onto_polyline(point, polyline, cumlen) that returns cross_track_dist. The PathProjectionCache already uses it for the correct-path projection.
Compute cost: ~10 branches × ~100 segments each = ~1000 vectorized segment projections per call. Well under a millisecond; negligible vs SOFA step time.
Sketch

def _build_branch_index(self) -> None:
    # Replace: self._branch_kdtree = cKDTree(all_coords)
    # With: precompute each branch's cumlen for fast polyline projection
    self._branch_polylines = [b.coordinates for b in vessel_tree.branches]
    self._branch_cumlens = [compute_cumulative_arclength(p) if len(p) >= 2 else np.zeros(1)
                            for p in self._branch_polylines]
    self._branches_tuple = vessel_tree.branches

def get_nearest_branch(self):
    if self._nearest_branch is None:
        tip = self.get_tip_vessel_cs()
        best_d, best_i = float("inf"), 0
        for i, (poly, cumlen) in enumerate(zip(self._branch_polylines, self._branch_cumlens)):
            if len(poly) < 2: continue
            r = project_onto_polyline(tip, poly, cumlen)
            if r.cross_track_dist < best_d:
                best_d, best_i = r.cross_track_dist, i
        self._nearest_branch = self._branches_tuple[best_i]
    return self._nearest_branch
About 20 lines of code total, drops the KD-tree and scipy dependency for this method, gives a mathematically sound answer, and stacks nicely with hysteresis (which now reacts to a cleaner raw signal).

Want me to apply this?


Pattern confirmed — the flip-flop is happening at BIF1, not bif2. Look at wrong_pt=(12.7,-26.0,67.3) and corr_pt=(30.3,-19.0,68.7) — both at z≈68mm which is bif1 coordinate range. Insertion bounces 55–75mm throughout (not ~490 like the bif2 oscillators we saw before).

What happened in this episode:

Restore placed wire at ~380mm (bif2 zone), but with target of a different branch than the checkpoint was captured for. So tip was already in the wrong daughter from step 1.
Heuristic retracted wire full speed (~6mm/step × 50 steps) from 380mm down to 64mm — all the way back through bif2 and the trunk, now sitting at bif1.
At bif1, same flip-flop pattern: off_br counter climbs to 22, 28, 30 then resets to 0 as stable flips back True. Never hits 50. Never fires timeout. Episode runs full 1000.
Evidence rows:


step  50: on_br=0  off_br=22   (approaching timeout)
step 100: on_br=1  off_br=0    (flipped back — counter reset)
step 150: on_br=0  off_br=12   (off again)
step 200: on_br=0  off_br=28   (approaching timeout)
step 250: on_br=0  off_br=10   (counter reset happened between 200-250)
...
Hysteresis helps but the raw KD-tree signal is still noisy enough at junctions to flip the stable state through the 5-step debouncer every ~100 steps. Applying Option 2 now.

------------------------------------


The formula
Let:

axis = normalize(tracking[0] - tracking[k]) — wire's long axis at the tip (spanning ~10 mm, using k ≈ 5 beam nodes of spacing)
curv_vec = (tracking[0] - tracking[k]) - (tracking[k] - tracking[2k]) — discrete second-derivative of the wire position: the J-tip's curvature direction. Perpendicular-ish to axis.
tangent — path tangent at the wire's projection onto the correct path (already computed in current code)
Then:

# Project onto plane perpendicular to the rotation axis
curv_perp = curv_vec - dot(curv_vec, axis) * axis
tangent_perp = tangent - dot(tangent, axis) * axis

# Fallback: when J-tip curvature is tiny (straight wire section), no
# rotation can steer anything. Return 0.
if |curv_perp| < 0.1 mm or |tangent_perp| < 1e-3:
    heading_error = 0.0
else:
    curv_perp /= |curv_perp|
    tangent_perp /= |tangent_perp|
    cos_ang = dot(curv_perp, tangent_perp)
    sin_ang = dot(axis, cross(curv_perp, tangent_perp))  # signed-about-axis
    heading_error = atan2(sin_ang, cos_ang)  # in [-π, π]
This is the angle (in radians) by which rotating the base around axis aligns the J-tip's curvature direction with where the path wants to go.

Apply the same projection to cross_track_signed:

# offset_vec: tip's lateral offset from correct centerline (already computed)
offset_perp = offset_vec - dot(offset_vec, axis) * axis
cross_track_signed = dot(axis, cross(curv_perp, offset_perp_hat))  # signed
— or keep its current gain small; the main lever is heading_error.


---------------------------------------------


Possible next experiments: (a) narrow Phase 2 clamp back to ±0.5 or ±0.3 for small heading errors (keep ±1.5 only when |heading_err|>0.5), (b) add a "commit lock" — once on correct branch at d_corr<15, disable rotation for N steps to let the wire finalize the entry.



---------------------------------------------------------


# Phase C — Full Design Option Space
Trigger condition
Single option (validated): activate when wire's projection arclength s ≥ rva_jn_arc (past the bridge → RVA junction).

rva_jn_arc cached at first call via _find_junction_arc(junctions, "(11)", "RVA") over pathcontext.get_path_junctions().

Target direction (the main design choice)
Option 1: Fixed +z (matches v2 simplicity).

Pro: degenerate vs wire's +z long-axis inside RVA → closed-loop returns ~0 → harmless
Pro: behaves like Phase A/B in v8
Con: provides no actual tip-orientation signal; only modulates gw_trans
Option 2: Dynamic planned-path tangent at s + lookahead.

target = pathcontext.get_planned_path_tangent_at(s, lookahead_mm)
Pro: tracks RVA's continuously-curving centerline (sweeps +y → +z over first 10 mm)
Con: closed-loop sign convention validated as right-hand-rule (+gw_rot = CCW around +t̂); needs original sign (NOT flipped). The flipped-sign experiments (v6/v7) confirm.
Con: small lookahead gives reactive control; large lookahead anticipates better but pulls J-curl ahead before body catches up.
Sub-option 2a: lookahead = 1 mm — track tangent very locally
Sub-option 2b: lookahead = 5 mm — what was used in v3
Sub-option 2c: lookahead = 10 mm — half of the daughter-entry-turn distance
Sub-option 2d: lookahead = 20 mm — past the entry turn entirely
Option 3: Position-vector toward target.

target = (target_pos - tip_pos) / norm
Pro: orients J-curl toward the goal directly; non-degenerate when wire and target differ in position
Pro: doesn't need path tangent at all — works regardless of pathfinder
Con: ignores path topology — could try to push tip toward target through a vessel wall
Con: assumes target_pos is reachable in straight-line direction from tip, which fails inside curved daughters
Option 4: Daughter-centerline normal (axis perpendicular to the daughter's local tangent, pointing into the lumen).

target = lumen_axis_at(s)
Pro: explicit "stay centered in the daughter" signal
Con: requires daughter-radius lookup; data exists (_branch_radii) but not implemented for path-aware targets
gw_trans speed
Option A: Slow (1.5 mm/s) — matches Phase B; gives torsion time to propagate; safest in narrow daughter

Option B: Medium (2.0 mm/s) — was used in v3

Option C: Default (5 mm/s) — gives the wire body more push to commit through the entry turn; risks fold

Option D: Variable — min(5, 0.1 * d_remaining_to_target) — slows down as target approaches (matches default heuristic regime)

Window extent
Option α: Full daughter (s ∈ [rva_jn_arc, total_length]) — Phase C active until end of episode

Option β: Entry-turn region only (s ∈ [rva_jn_arc, rva_jn_arc + 30]) — only the 30 mm where RVA's tangent rotates +y → +z; revert to default heuristic for the long straight portion afterward

Option γ: Entry + middle (s ∈ [rva_jn_arc, rva_jn_arc + 100]) — covers the entry plus most of the daughter's length, leaving only the final approach to default

Stagnation/recovery sub-policy
Independent of target/speed — for the 125 episodes (~25 %) that reach RVA but stall at entrance:

Sub-policy A: Off-path detection — if d_corr_routed > X mm after entering Phase C, treat as off-path and let default retract handler take over

Sub-policy B: Stagnation retract — if Δs < 1 mm over 20 consecutive steps, command gw_trans = -3 for 5 steps, then resume forward

Sub-policy C: Wiggle — if stalled, alternate gw_rot = ±1.5 for a few steps (sweep J-curl orientation) while keeping gw_trans minimal

Sub-policy D: None — let the wire stall to truncation

Sign convention (validated, not optional)
gw_rot = +sign((bend × target) · t̂) × angle — right-hand rule, ORIGINAL sign. Flipped sign caused v6/v7 regressions.

Combinatorial space
If we treat each axis as independent:

4 target options × 4 lookahead sub-options (Option 2 only) = effectively ~7 target variants
4 gw_trans options
3 window options
4 stagnation sub-policies
That's ~7 × 4 × 3 × 4 = 336 combinations. Most aren't worth testing. The interesting subset is small:

candidate	target	gw_trans	window	stagnation
C-fixed	Fixed +z (degenerate)	2 mm/s	full	none
C-dyn5	Dynamic, lookahead=5	2 mm/s	full	none
C-dyn1	Dynamic, lookahead=1	2 mm/s	full	none
C-dyn10	Dynamic, lookahead=10	2 mm/s	full	none
C-target	Position-vector to target	2 mm/s	full	none
C-recover	Dynamic, lookahead=5	variable D	full	sub-policy B
C-narrow	Dynamic, lookahead=5	2 mm/s	β (30 mm)	none
-----------------------
 
 What happened before Phase C existed (v8 / v2cfg-500 / v2cfg-500-inst):
 what all changed for Phase C apart from new run files? what file were we running for before for RVA with only Phase A and Phase B; want to run that same 500 sepsode episode and store episode as npz for heuristic training ; success counts as threading into RVA daughter (around 25 - 30 %)

 Analyse rewards 


 revert snapshot 


 -----------------------------------------------

# Can you explore SOFA code to explore if there is a better way to restore saved checkpoints; right now it feels like it tries to recreate the scene with save DOFs for the wires ; what if we can save the whole scene as it is ; like a saved video game checkpoint ; what if space is not an issue.

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

Want me to implement steps 1-4 and re-test? Note this only matters for future Strategy 1 runs — the currently-running paired Phase C grid (full traversal, no restore) doesn't depend on it.


----------------------------------

# ok out of this last run that ran 4500 episodes ; find the seeds that ran with max steps for atleast 6 out of 9 runs and were in RVA daughter ; analyse why the variants all failed then looking at the episodes log; give a deep dive looking at tll the details of the step log


Deep dive: why all 9 variants fail on seed=1455355984 (target deep RVA at z=496.9)
The "26.4 mm ceiling" is a metric reset, not a wall
Looking at C2 step 331→332 (the supposed peak):


s=331 | tip=(15.3, 77.1, 419.0) | arc_d=26.3 | d3=1.8
s=332 | tip=(15.4, 77.3, 419.2) | arc_d=0.0  | d3=inf  ← tip moved 0.2mm, metric reset
The wire's projection arclength crossed an internal RVA bifurcation (RVA splits at z≈422 into RVA-main and branch(17)). After the crossing, arc_past_d measures from the new junction → resets to 0; d_corr_3d returns inf because no further daughter entries exist on the planned path.

So the 26.4 mm value seen across ALL seeds and variants is just the arclength from bridge(11)→RVA junction to the first internal RVA junction. Wires DO go past it; the metric just zeros out.

The actual stall is at z≈420-424, the internal RVA bifurcation
Tracking C2 tip3d directly:

step 305: (14, 72.5, 419) — first at peak position
step 305-331: oscillates around (15, 73-77, 418-419) — only 4 mm of y-progress in 26 steps
step 332-349: drifts to (14.5, 80, 423.6) — now AT the internal bifurcation
step 350+: alternates between cur_branch=RVA.mrk and cur_branch=CC(17).mrk for the next 250 steps — wire deflects into branch(17) at every forward push, gets retracted, re-enters RVA, deflects again
This is the real failure: the wire is stuck at z≈420-424 oscillating between RVA-main and the (17) sister branch.

How each variant fails differently
C0 (no Phase C, default heuristic, R=-115): After bridge crossing, default centerline-follower takes over. Default uses heading-error + cross-track corrections that aren't tuned for the merged-cavity geometry. The wire spends 100+ steps off-path → ~115 of cumulative penalty. Worst variant.

C1 (fixed +y, R=-20.6, BEST): alternating ±1.5 gw_rot every 2-3 steps (saturated commands; the closed-loop rotation always wants to drive the J-tip toward +y). This gives a steady "snake-charm" wagging motion. Wire body advances slowly but consistently, spends minimal time off-path. Doesn't go further than C2 (max=25.7 vs C2's 26.3) but accumulates fewer penalties.

C2/C3/C4/C6/C7 (dynamic tangent, R=-64 to -74): target = path tangent at s+lookahead. Near the internal junction, the planned-path tangent swings wildly (it's interpolating across a junction). gw_rot output oscillates with high frequency:


C2 steps 308-313: gw_rot = +0.05, +0.11, +0.06, +0.03, +0.10, +0.05
Wire body can't follow high-frequency rotation commands; it drifts off-axis. More off-path episodes than C1.

C5 (position_to_target, R=-62): target = (target − tip).normalized(). Tip at (15, 75, 420), target at (28, 69, 497) → direction ≈ (+0.16, -0.07, +0.98). Mostly +z. But RVA's local centerline at this point curls in the -x, +y plane. The closed-loop tries to rotate J-tip toward +z (which would push wire through vessel wall). Wire fights wall → more off-path.

C8 (dynamic + stagnation retract, R=-7.6, but FAILS DIFFERENTLY):

Steps 305-325: same as C2, wire wedged near 26 mm
Stagnation detector (Δs<1mm over 20 steps) fires → commands gw_trans=-3 for 5 steps
This works as designed — wire retracts back into bridge(11)
BUT: wire can't re-thread the daughter. By step 400 tip is at (12, 56, 419), off_br=8 (off-path for 8 consecutive steps), cur=branch(17)
For remaining 200 steps, oscillates in/out of bridge(11) cavity
Lower magnitude reward only because it's not stuck on the wrong-branch retract loop — it's just stuck farther back
The retract IS achieving its design goal (escape stagnation) but Phase C has no machinery to recover the prior J-tip orientation on a re-entry attempt. The retract throws away progress without a recovery plan.

Why ALL variants share the same failure
The wire is stuck at the internal RVA bifurcation (z≈422). All 9 variants — including the C0 default — share these traits:

No phase logic for the internal junction. Phase C only knows about the bridge(11)→RVA junction (rva_jn_arc). After 26 mm in RVA, the wire is at a second daughter junction that no Phase recognizes. The variant config gets the same dynamic_tangent target it had 5 mm ago, but the geometric situation has changed (sister branch entry).

Routed-d_corr returns inf after the metric reset. The heuristic regime selector (default heuristic) loses its near-junction signal, falls back to plain centerline-following, which can't handle the bifurcation.

No coordinated J-tip reorientation. At a bifurcation, the wire body needs to be deliberately twisted to commit to one sub-branch. None of the variants are designed for this second commit. They all try to follow the centerline the same way they did in the proximal RVA segment.

Off-path retract handler thrashes. Once wire enters branch(17), is_on_correct_path() returns False → handler commands gw_trans = uniform(-10, -1). Wire pulls back ~5 mm into RVA, on-path again, heuristic pushes forward → deflects into (17) again. Repeats indefinitely.

Concrete diagnosis
The current implementation treats RVA as a single curve. The actual mesh has:

bridge(11) → "merged cavity" at z≈412-417 (Phase C handles this)
RVA proximal segment z=417→422 (~26mm of arclength, default-heuristic territory)
Internal RVA bifurcation at z≈422 (no logic exists)
RVA distal segment z=422→440+ (toward target at z=497)
To get past 26 mm, we need either:

Phase D: same machinery as Phase C but anchored at the internal junction's arclength — explicit J-tip reorientation while crossing it.
Path-context fix: include the internal junction in _path_daughter_arclengths so Phase C's daughter_gw_trans extends through it (not just the first 5mm past rva_jn_arc).
Routed-d_corr fix: don't return inf past the last daughter junction; return Euclidean distance to target instead, so the heuristic has a useful signal in the deep daughter.
The 14 seeds with ≥6 max-steps failures all share this same internal-bifurcation pattern. The closer the target is to the deep RVA (z≥440), the more likely all variants fail.

--------------------------------------


# Both points well taken — Euclidean shaping was the wrong call (LCCA/LVA/RCCA/RVA daughters all converge in z, so a wire deep in the wrong daughter is Euclideanly close with zero reachability), and the LVA reward numbers in the analysis predate the wbt-suppression + reward-grace patches (those landed during the LVA v2 500-ep run, not before it, so the run's mix of grace-on/grace-off steps is muddled).

Given path-following is the right reward axis, the concrete fixes that fit that frame:

Gate ArcLengthProgress on is_on_correct_path(). Currently the +0.01/mm fires off the polyline projection arclength, so an arch-wedged wire whose projection drifts forward still earns shaping. Zero out the progress term when is_on_correct_path()==False so only on-path forward motion is rewarded. Off-path lateral penalty stays on.

Boost the progress factor. 0.01/mm × ~120 mm of path = +1.2 max shaping across a successful episode. Step penalty (−0.001 × ~300 steps) = −0.3. That's a thin margin — explains why on-path successes can land near zero. Bumping progress_factor to 0.03–0.05 gives the path-following signal real weight versus the step penalty floor, without needing any new reward term.

Re-run LVA 500-ep clean. All three daughters now use the same env5 (grace + suppression). Re-running LVA 500-ep with the current code gives a fair reward distribution to compare with the post-fix RVA/RCCA numbers; only then can we decide whether the LVA reward shap

# SUCCESS band overlaps FAIL bands: best LVA NEAR-MISS could outscore worst LVA SUCCESS. Policy gradient noise.
Entry +1 rewards dominate the signal: an episode that just touched the (0)→(11) junction gets +1, then runs 500 steps in a wedge for −15 net. Compared to a SUCCESS at +4: differ by only ~19. Entry rewards are not aligned with "is this episode good?"
No progress-to-target shaping: distance-to-target is not in reward. A wire 5 mm from target for 100 steps earns nothing more than a wire 200 mm away.
RL applicability — step-level (TD / Actor-Critic) view
Per-step reward distribution:

Mostly −0.01 per step (base time penalty)
−0.1 per step when off-path (after 3-step grace)
+1.0 at daughter entry crossings (sparse, ~3 per success)
+3.0 at terminal (sparse, ~3-19% of episodes)
Small arc_progress reward proportional to forward motion on path
TD bootstrap problems:

The +1 entry signal is the dominant gradient. Agent will learn to cross entries — fine for shallow targets but doesn't help for navigating deep daughters past arc=25-40mm.
−0.1 off-path penalty is "noise" in LVA's (19) region where it's geometrically correct but classified wrong. Without the LVA-specific suppression flag, off-path penalty mis-trains the policy.
Long episodes (600 steps) with sparse signal: TD with γ=0.99 has effective horizon ~100 steps — terminal +3 propagates back to step 500 but barely reaches step 1. The agent can't link "reached target" to "exited trunk-top correctly" via Bellman.
# Recommended reward changes for RL training
Critical (must-have):
Dense distance-to-target shaping:

reward += k * (prev_dist_to_target - cur_dist_to_target)  # k≈0.5
This converts the sparse +3 terminal into a continuous gradient pointing at target. The agent gets +reward for closing in, even mid-trajectory. Standard potential-based shaping → policy-invariant per Ng et al. 1999.

Increase terminal success reward to +20 or +30:
Current +3 gets drowned by accumulated −50 step penalties. The terminal needs to dominate. Even in worst case (600 step episode in cavity), a true success should net positive return.

Remove or reduce off-path STEP penalty (keep entry/timeout):
The −0.1/step for being off-path creates a wrong signal in geometrically valid cases (LVA's (19) extension). Either:

Generalize the _heur_suppress_wrong_branch flag to env5 logic (no step penalty when projection arc is past target arclength)
Remove step penalty entirely; keep only the +/-1 entry penalty + timeout
# Useful (nice-to-have):
Per-step on-path bonus:

## if on_correct_path: reward += 0.02
Counterbalances time penalty when wire is correctly navigating.

Wider success radius (5 mm → 7-8 mm):
Wire's J-tip mechanics put a hard floor on tip-to-centerline distance (~4-5 mm). Tightening success below this threshold guarantees some near-misses can never succeed — that's an environment ceiling, not a policy failure.

## Wedge-mode penalty:
Detect "wire stationary for N steps" → small penalty (e.g., −0.05/step). Currently a wedged wire still gets the time penalty but no extra signal. With wedge detection, the policy can learn to recover.

## Curriculum on target depth:
Currently targets are uniformly sampled along the daughter's length. Easier-to-reach shallow targets get success while deep targets are unreachable. Sampling biased toward shallow early in training would build a stronger "navigate-to-daughter-mouth" foundation.

Specific to the heuristic
The +1 daughter-entry rewards are great signal for "did the wire choose right at this junction" but fire too easily (wire briefly touches LCCA daughter then deflects away → still gets +1). Should they require sustained cur_branch=daughter for N steps?
Phase A/B/C's success on RCCA/RVA correlates with junction crossing, not target reaching. Reward currently rewards both equally per step — RL would learn the cheaper option (cross more junctions).
Bottom-line for RL
Current reward structure is good for heuristic comparison (different policies on same env produce comparable returns) but weak for RL training:

Too sparse (only 4-5 reward events per success)
Variance within outcome categories overlaps inter-category differences
No directional shaping toward target
A SAC-trained agent on this reward would likely converge to "cross junctions and stop" — not "navigate to target". Adding distance-to-target shaping reward is the single most impactful change.


-----------------------

# then we should use the same state machine / code to formulate rewards ; also make heuristic behaviour with rewards consistent

Agreed — the heuristic and reward currently disagree on what counts as "wire is making valid progress", and that mismatch is exactly why LVA successes net negative. Concrete unification:

Single source of truth — extend path_branch_idx_set with a per-target path_extension_set. Branches that are geometrically/functionally the continuation of the planned path but aren't in the planner's polyline (the (19)-after-LVA case is the canonical example). Built at episode reset from a small per-daughter table:

LVA target → extension = {(19)} (deep LVA continuation)
RVA / RCCA / LCCA targets → extension = {} (no equivalent)
This set is the shared truth used by all three consumers:

is_on_correct_path() in pathcontext — returns True if _current_branch_idx ∈ (path_branch_idx_set ∪ path_extension_set). The state-machine logic stays as-is; only the membership-check union changes.
env5 wrong-branch penalty + timeout — falls out for free, since both penalty (if not on_correct_path) and timeout already key off is_on_correct_path(). The _heur_suppress_wrong_branch flag becomes deletable.
ArcLengthProgress reward — when _current_branch_idx is in the extension (not the strict planned path), project onto THAT branch's centerline and reward forward motion along it (instead of the saturated planned-path polyline projection that today gives 0). When in the strict planned path, behave as today.
What this gets us:

Heuristic stops needing the suppression hack — the wire is genuinely on-path by the new definition.
LVA-target wires advancing through (19) earn +0.01/mm of forward motion along (19)'s centerline, not 0. A wire that walks 30 mm of (19) earns +0.3 of shaping instead of accruing the (currently-being-fixed) penalty stream.
Wedged-off-path wires (different code, in some sister branch not on the route) still get no progress and still get penalized, because they're outside the union.
Heuristic phase selection becomes "look at _current_branch_idx and decide what to do" — same data the reward sees, no possibility of disagreement.
Implementation surface (small):

pathcontext: add path_extension_set field, lookup table indexed by target name, build at reset, union into is_on_correct_path(). Add helper get_progress_polyline_and_cumlen() returning either planned-path polyline or current-branch polyline based on state-machine.
arclengthprogress: replace direct get_projection() with the helper, so the projection axis follows the state machine.
env5: delete _heur_suppress_wrong_branch reads (both penalty block and _resolve_termination_reason()).
heuristic_policy_lva: delete the suppression-flag write and the off-path override branch; the _detect_phase lva_daughter regime now fires naturally via is_on_correct_path()==True when wire is in (19).
Want me to draft this as a plan doc for review before changing code, or just apply it directly?




-------


Critical (5)
A1 — LVA suppression silences penalties but doesn't generate reward (already known). [heuristic_policy_lva.py:357-369](training _scripts/util/heuristic_policy_lva.py#L357-L369) + [env5.py:538-548](training _scripts/util/env5.py#L538-L548) + arclengthprogress.py:99-109. Suppression makes (19) penalty-free but proj.s is pinned, so forward (19) motion still pays 0 progress. Unification plan fixes this.

A2 — +1 daughter-entry reward gate doesn't align with heuristic Phase-C trigger. [env5.py:566-573](training _scripts/util/env5.py#L566-L573) requires arc_past_last_daughter >= 10.0. Heuristic Phase C triggers at s >= jn_arc - 5 and "in-daughter" sub-regime at s > jn_arc + 5 ([heuristic_policy_rva.py:296-301](training _scripts/util/heuristic_policy_rva.py#L296-L301)). 10 mm dead zone: heuristic is doing the "right thing" with no reward signal.

B1 — Agent observation has no absolute arclength, no junction-arc, no count-of-bifurcations-passed. localguidance.py:179-180,318-326 only exposes normalized d_rem (feature 0) and a clipped dist_to_bifurc (feature 6). Two episodes with the same d_rem_norm=0.4 can be in totally different phases (LVA path ≠ RVA path length). Agent literally cannot replicate "fire Phase C at s >= rva_jn_arc - 5."

B2 — Heuristic computes J-tip bend from 3D tracking points; agent gets 2D Tracking2D. [heuristic_policy_rva.py:158-188](training _scripts/util/heuristic_policy_rva.py#L158-L188) uses bend = p0 + p2 - 2*p1 with full vessel-CS coords. [env5.py:114](training _scripts/util/env5.py#L114) constructs Tracking2D(n_points=5) — image-plane projection, y-axis dropped. The agent cannot reproduce the closed-loop rotation calculation from its observation.

B3 — None of _heur_rva_phase / _heur_*_jn_arc / _heur_suppress_wrong_branch are in the observation. They're stashed on base_env for STEP-log diagnostics only ([env5.py:677,864](training _scripts/util/env5.py#L677)). The single best summary of state at a bifurcation is invisible to the agent.

Moderate (6)
C1 — LCCA hard-codes Phase C parameters; ignores _phase_c_variant. [heuristic_policy_lcca.py:377-393](training _scripts/util/heuristic_policy_lcca.py#L377-L393) doesn't read the env's variant ID, has no PHASE_C_VARIANTS dict. RVA / RCCA / LVA all do. Factorial grid silently skips LCCA.

D1 — Heuristic and +1 reward use different "what counts as a junction" sets. Trunk-top crossing (2)→(0) has no off-path branches on the LVA route → filtered out of _path_daughter_arclengths (pathcontext.py:625-637) → no +1 reward there. But heuristic Phase-A milestone is exactly that crossing ([heuristic_policy_lva.py:294-296](training _scripts/util/heuristic_policy_lva.py#L294-L296)). Disagreement on what's a "meaningful event".

A4 — is_on_correct_path() applies the on-path-mask; ArcLengthProgress uses raw get_projection(). Same wire state, two different verdicts: heuristic sees off-path (stops steering), reward sees a valid projection (keeps shaping).

B4 — LCCA Phase B position-vector uses 3D target.coordinates3d; agent's Target2D is the 2D image projection. Partial signal access — y-axis component of the position-vector is dropped.

B5 — Lookahead tangent: agent gets tangent at s; heuristic uses s + 5 (C2 default). Tangent at curvature regions differs measurably; agent has no way to "look ahead" the way C2/C4 do.

D2 — COMMIT_HYSTERESIS_MM = 10.0 is fixed but bridges like (19) are short. ~13 mm bridges + 10 mm commit threshold ≈ wire commits to the next branch almost immediately. Per-junction adaptation would be safer.

Cosmetic (3)
C2 (mislabeled docstrings: RCCA/LVA say "RVA daughter"), C4 (LCCA/RCCA/LVA all write to _heur_rva_phase to fool the STEP-log writer), E2 (curriculum stage 2 multiplies cath_trans by 0.1 only when collecting demos — distorts heuristic-mode trajectories vs inference).

What this changes for the plan
The unification I sketched fixes A1, A4, D1 — the heuristic/reward/state-machine disagreement triad. But B1/B2/B3 are a separate, equally critical problem: even with reward signal aligned, the agent literally can't see the state variables the heuristic uses to decide its action. Imitation learning will plateau because the policy class isn't expressive enough given the input.

Proposed expanded scope:

Reward unification (path_extension_set + state-machine-driven progress projection). Fixes A1, A4, partial D1. ~80 LOC.
Observation enrichment in LocalGuidance — add features: proj.s/total_length, arc_to_next_daughter/100, arc_past_last_daughter/100, 3-or-4-dim one-hot phase indicator, plus the heuristic's bend_hat projected to 2D as 2 extra features. Fixes B1, B2, B3. ~30 LOC, expands obs from N to N+8 features (will require retraining from scratch).
Reward trigger alignment — move +1 daughter-entry from arc_past >= 10 to s >= jn_arc - 5 (matching Phase-C onset) OR change the heuristic. Fixes A2. ~5 LOC.
LCCA variant dispatch — copy RVA's PHASE_C_VARIANTS table into LCCA. Fixes C1. ~50 LOC.
Items 1+3+4 are local and safe. Item 2 is more invasive (obs shape change → retraining). Want me to draft a phased plan, or pick a subset to do first?


------------------------



# Assuming I will do all this unification as much as I can ; should I consider training separate RL model for each daughter or one for all ; what are the pros and cons ; elaborate in detail

Both options are viable; the right answer depends on what you optimize for. Concrete trade-offs grounded in this codebase:

Shared model (one policy, target-conditioned via Target2D obs)
Pros
1. Shared trunk-traversal experience. Looking at the heuristic logs, ~50-65% of every episode is trunk(2) ascent up to z≈380 — identical physics, identical "correct" action (push gw forward with minor rotation corrections) regardless of which daughter the target is in. A shared model accumulates 4× the data for this dominant sub-task. Per-daughter models each have to learn it independently from 1× data.

2. Cross-daughter signal generalizes. Bif2 geometry-knowledge ("the bridge entries are at these specific x-y positions in vessel-CS") transfers across daughters. An RVA failure pattern (wedge in cavity) is the same physical event as the LCCA/RCCA/LVA wedge cluster — a shared model can treat it as one "don't do this" pattern with 4× the negative examples.

3. Better curriculum tools available. With one buffer you can balance success-rate-by-target via weighted sampling, and you can use shared hindsight-experience-replay-style relabeling ("this trajectory hit LCCA, treat it as a positive example for the LCCA-target distribution").

4. Single artifact at deployment. New target locations within the same vessel tree become a runtime config change, not a retraining job.

5. Robustness to distribution shift. If the target sampler at deployment is broader than the train distribution (e.g., interim mid-daughter targets), a target-conditional model has a chance; per-daughter models silently break.

Cons
1. Class imbalance is severe. Empirical: heuristic produces 19% LVA / 2.9% RCCA / 1.4% LCCA / 0.6% RVA successes. Without reweighting, ~19/24 of successful demos are LVA. Naïve SAC training will pull the policy toward "behave like LVA-agent" for every target — the worst possible failure mode for the hard daughters.

2. Phase B / Phase C are genuinely different policies. RVA needs dynamic_tangent + RVA-aimed steering at bif2. LCCA needs position-vector at lcca_jn + 25 mm. LVA needs no active steering (let trunk-ascent physics deliver wire to (18)). RCCA mirrors RVA. A shared network has to learn 4 distinct conditional sub-policies given target_state as the discriminator. Whether your SAC actor has enough capacity to learn this is empirical — current network width may or may not suffice.

3. Reward magnitude varies 3-4× across daughters (LVA path ≈ 140 mm shaping max, LCCA path ≈ 60 mm). Critic has to learn a wider Q-value range. Solvable with reward normalization, but worth being aware of.

4. Per-daughter regression hard to detect. A training run that improves average reward by 20% might actually be: LVA +40%, RVA −60%. You only catch this with per-target metric logging — easy to forget.

5. Single point of failure during training. If RVA learning destabilizes the shared model, LVA inference also degrades. Per-daughter models isolate failures.

Per-daughter models (4 separate policies, each conditioned on a fixed target)
Pros
1. No class imbalance. Each model sees only its target's demos and only its target's training experiences. RVA model isn't pulled toward LVA behavior.

2. Dedicated capacity. The whole network is allocated to one steering strategy. RVA's Phase B can be learned more sharply because the model isn't allocating weights for LVA's "no steering" regime.

3. Per-daughter hyperparam tuning. RVA likely wants higher exploration noise during training (harder problem); LVA can have lower noise. Reward shaping factors, action curriculum stages, demo seeding ratios — all tunable per daughter without trade-offs.

4. Trivially parallelizable. 4 models × 4 workers on 1 GPU each = 4× wall-clock speedup if you have the hardware. Today your single training run uses 1 GPU.

5. Debugging is local. When RCCA stops converging, you don't have to wonder whether it was caused by an LVA-targeted reward shaping change.

Cons
1. 4× compute and storage. Independent of how you parallelize, the total compute is 4× — each model retrains the trunk traversal from scratch.

2. Per-daughter trunk-traversal sample inefficiency. The agent that knows "push up trunk" generalizes; 4 agents each have to discover this independently. You're discarding the obvious data-sharing opportunity.

3. Maintenance overhead. 4 training runs, 4 hyperparam configs, 4 deployments, 4 checkpoints, 4 evaluation pipelines.

4. No transfer to new targets. If a 5th target (or an intermediate target along the route) is needed later, you train from scratch.

5. Inconsistent action distributions. 4 policies trained with different randomness will produce slightly different "default" actions for indistinguishable trunk states. At deployment in scripted procedures (where the target might change mid-procedure), this could matter.

Hybrid — shared backbone, per-daughter head
Worth considering as a middle ground. Concrete shape for this codebase:

One feature extractor (taking tracking, last_action, guidance from obs) → shared embedding.
4 small policy heads, one per daughter. At inference, route based on target_state (categorical: which daughter is the target on).
Critic: also per-head, or one shared critic that consumes target as input.
Pros: shares trunk-traversal feature learning. Per-daughter head can specialize Phase B/C. Capacity isn't over-allocated to easy daughters.
Cons: implementation complexity in SAC; need to modify the eve_rl actor/critic. Routing requires reliable "which daughter is this target on" inference at every step (you have this — daughter identity is stable per episode).

Concrete recommendation for this codebase
Given:

Heuristic success rates are wildly imbalanced (30× spread).
Phase B/C policies differ substantively per daughter.
You have heuristic seeding infrastructure already per-daughter.
You don't yet have any trained RL model — the current state is pure heuristic.
Suggest starting with per-daughter models for the first generation of RL training, for three reasons:

Sample-efficiency from heuristic seeding is the dominant factor at this stage. Each per-daughter heuristic is producing genuine demonstrations of the right Phase B/C. A per-daughter agent ingests pure on-policy demos, no mixing. A shared agent ingests 19× more LVA demos than RVA demos and the seeding asymmetry compounds.

You'll have 4 baseline numbers. When you compare to a future shared-model experiment, you have ground-truth per-daughter ceilings. Without per-daughter baselines you can't tell if a "70% overall" shared model is actually 100% on LVA and 40% on the others.

The trunk-traversal share is overstated. Each daughter's heuristic injects ~250-400 trunk-traversal trajectory steps × 500 episodes = ~150k steps of trunk experience per model. That's plenty for the trunk sub-policy. The marginal benefit of 4× = 600k is small once you're past the data-density elbow.

After per-daughter models converge (or plateau), THEN run the shared-model experiment with the per-daughter buffers as initialization (multi-task fine-tuning). That experiment answers the interesting research question: "does the shared representation help generalize?", with per-daughter ceilings already in hand.

Decision-making heuristic
If by the end of training each per-daughter model achieves >40% success rate and the per-daughter heuristics share <30% of their Phase B/C action distributions (measurable from the logs), the case for keeping them separate is strong.

If success rates plateau much lower across all daughters AND the failure modes are similar (e.g., all of them fail because of bif2 cavity dynamics), the case for sharing flips — the underlying problem is structural and you're not getting separable benefit from separate training.

# "1. Class imbalance is severe. Empirical: heuristic produces 19% LVA / 2.9% RCCA / 1.4% LCCA / 0.6% RVA successes. Without reweighting, ~19/24 of successful demos are LVA. Naïve SAC training will pull the policy toward "behave like LVA-agent" for every target — the worst possible failure mode for the hard daughters. 2. Phase B / Phase C are genuinely different policies. RVA needs dynamic_tangent + RVA-aimed steering at bif2. LCCA needs position-vector at lcca_jn + 25 mm. LVA needs no active steering (let trunk-ascent physics deliver wire to (18)). RCCA mirrors RVA. A shared network has to learn 4 distinct conditional sub-policies given target_state as the discriminator. Whether your SAC actor has enough capacity to learn this is empirical — current network width may or may not suffice." ----> Can these be solved by HER / PER applied to replay buffer

Mostly yes for Con #1, no for Con #2. They're different problems and need different fixes.

PER vs class imbalance — strong yes
A successful RVA terminal transition has reward +3, but the critic has barely seen any RVA successes so its value estimate is near 0. TD-error for that transition is huge (~3 vs ~0). PER samples it ~5-10× more often than its 0.6% natural frequency. Effective gradient contribution from RVA successes becomes proportional to learning signal (TD-error), not episode frequency. That directly attacks the "policy behaves like LVA-agent for every target" failure mode — rare-but-informative transitions get the gradient mass they deserve.

Concretely for this codebase:

Without PER: 1 RVA success per ~160 LVA episodes in the buffer. Gradient is 99% LVA-flavored.
With PER (TD-error-prioritized): RVA terminal transitions sample ~5-10×. Now ~5% of the gradient comes from RVA successes. Still imbalanced but no longer catastrophically so.
Caveats:

PER amplifies high-TD-error failures too. Pre-unification this was bad (the LVA-(19) flicker created artificial high-TD spikes from spurious penalties). Post-unification (penalties suppressed when on extension, shaping aligned), the residual high-TD events are genuine learning signal — PER + clean rewards is a much stronger combination than PER alone.
PER has annealing-bias (β) and learning-rate-sensitivity considerations — not free, but well-understood.
HER vs class imbalance — partial yes, with a structural caveat
HER converts "failed to reach goal G" trajectories into "successfully reached achieved-state G'". This works for near-miss failures but not for structurally wrong-path failures.

For this codebase, two failure modes dominate:

Near-miss in daughter (wire reached daughter centerline, didn't go deep enough): HER relabels target to whatever centerline point the wire achieved at its deepest. Suddenly a 250-step "failure" becomes a 250-step success demo. Strong yes for these.
Bif2 trunk wedge (wire never reaches any daughter centerline, stuck at z≈395 in the cavity): HER would relabel target to the wedge position. But the wedge isn't on any daughter centerline, so relabeling teaches the policy "drive to bif2 cavity" — which is anti-pedagogical. Skip these.
So HER's utility depends on the failure-mode mix per daughter:

daughter	near-miss-in-daughter %	bif2-wedge %	HER yield
LVA	27% reach × ~60% near-miss within ≈ 16%	50%	medium
LCCA	91.7% reach × ~95% near-miss ≈ 87%	~5%	huge — almost every failure is HER-relabelable
RCCA	33.7% reach × ~50% near-miss ≈ 17%	55%	medium
RVA	33% reach × ~50% near-miss ≈ 16%	47%	medium
This is actually a fascinating asymmetry: HER would dramatically benefit LCCA (its 1.4% success becomes effectively 80%+ after relabeling). And it would help RVA/RCCA/LVA modestly. So HER doesn't equalize the imbalance — it amplifies easier-to-relabel daughters. Combined with PER it ends up roughly balanced anyway because PER then re-prioritizes the still-rare bif2-wedge experiences.

Implementation cost: HER requires (a) a goal-conditional critic — you already have it, target_state is in obs — and (b) replay-buffer relabeling logic. Need to check if eve_rl's vanillaepisode.py or vanillashared.py supports goal-relabeling. If not, ~100 LOC of wrapper.

A critical constraint for HER here: only relabel to achieved states that are on a daughter centerline (not any 3D point in the vessel). Otherwise you teach "drive to wedge cluster".

Neither solves Con #2 (capacity)
PER and HER are sample-efficiency techniques. They don't add network capacity. If the SAC actor is too narrow to encode 4 distinct conditional sub-policies, no amount of clever sampling fixes that.

But "capacity bottleneck" is probably not the actual binding constraint at standard hyperparams. A 3-layer MLP of width 256-512 can comfortably encode dozens of conditional sub-policies if the input adequately discriminates between them — and target_state (the 2D image-plane target position) is a clean discriminator across all 4 daughter targets.

The bigger threat from Con #2 is optimization not capacity: gradient averaging in mini-batches biases the policy toward the majority-target's "correct action". This is the same disease as Con #1, just framed as a gradient problem instead of a sampling problem. PER fixes both framings simultaneously because high-TD minority transitions become dominant in the gradient.

Cheaper alternative to PER+HER: per-target replay buffers
Worth mentioning because it requires zero TD-error machinery: maintain 4 replay buffers (LCCA/LVA/RCCA/RVA), sample equally from each per gradient step (16 transitions = 4 from each buffer). Guaranteed balance. No PER β tuning, no HER relabeling logic, no goal-conditioning complications.

Pros over PER: trivial implementation, no hyperparam sensitivity, no annealing-bias concerns.
Cons vs PER: doesn't upweight high-TD-error transitions within a target; just balances across targets. Within-target diversity (e.g., LVA's 92 successes vs 392 failures) is still natural-frequency-sampled.

Recommended stack for the shared-model path
If you do want to attempt one-shared-model, the strongest combination would be:

Per-target replay buffer balancing (4 buffers, equal sampling) — fixes cross-daughter imbalance trivially.
PER within each buffer — fixes intra-daughter (success-vs-failure) imbalance.
HER with daughter-centerline-restricted relabeling — multiplies near-miss episodes into synthetic successes, especially valuable for LCCA's 91% reach.
Conservative target_state encoding — make sure the network's input layer can disambiguate targets clearly (consider augmenting with a 4-dim one-hot daughter indicator alongside the 2D coordinates, so the network doesn't have to discover the discrete target identity from the continuous coordinates).
This stack neutralizes most of Con #1 and side-steps Con #2 entirely by making the daughter identity highly observable.

My standing recommendation, though, is still per-daughter models for the first training cycle — debugging is dramatically simpler, the engineering surface is smaller, and you get clean baselines for the shared-model experiment afterward. PER+HER+per-target-buffers is the right setup once you have a working per-daughter baseline to compare against.


--------



# Yes — and the most valuable use isn't resumability, it's that the buffer contains a far better demo set than we started with. Let me explore the angles, highest-value first.

1. The RL successes are superior demos to the heuristic (the big one)
Recall the tension we found: the heuristic demos are gentle (gw~2) and reliably thread RCCA, but they time out before the deep target — only ~12 heuristic episodes actually reach it. The RL policy, by contrast, has now produced ~1000+ episodes that genuinely reach the target (the explore/success set, fast, R+5 to +8).

Those 1000 successes are exactly the demos we wish we'd had:

They reach the actual target (heuristic demos mostly don't).
They're at the speed that works for target-reaching — so re-seeding AWAC with them sidesteps the whole "demos say gw~2 but the task needs speed" mismatch that the policy had to discover on its own.
There are ~5× more of them than the 209 seed demos.
Re-seed a fresh run's buffer + warm-start from these RL successes and you'd start from a much stronger prior than the heuristic gave — likely well above the current ~10-17%.

2. Iterated self-improvement loop (the compounding version of #1)
This generalizes into a DAgger/distillation-style loop:


run RL → harvest the successful episodes → re-seed the next run with them → run RL → harvest better successes → repeat
Each generation's successes become the next generation's demos. Generation 1 distilled the heuristic; generation 2 distills generation 1's target-reaching policy; etc. The demo quality (success count, strategy quality) ratchets up each cycle. This is probably the single most impactful thing the buffer enables — without it, each run starts cold from the heuristic.

3. Offline RL — sim-efficient learning
SOFA simulation is the bottleneck (the run takes ~37h for ~900k steps). The buffer is a 900k-transition offline dataset. You could run offline RL (AWAC/IQL/CQL) on it — many gradient epochs, zero new simulation — to extract a strong policy, then do only a short online fine-tune. Given how slow the sim is, squeezing more learning out of already-collected data is a large efficiency lever.

4. Curriculum restore-state mining
The successful trajectories pass through every depth of the path. Mine restore checkpoints from them at multiple depths (at the turn, mid-RCCA, deep-RCCA) → build the reverse-curriculum restore pools we discussed, instead of only the single pre-bif(11) pool. The buffer is the raw material for that.

5. Precise reward/threshold tuning
Full (state, action) pairs at the turn let you analyze exactly what separates a successful turn from an overshoot — e.g., the precise gw_trans / rotation that makes the (0)→(11) turn — which would let you set the gentle-turn-penalty threshold from data rather than guessing.

The catch, and what it implies
The current buffer is already unrecoverable (the save is broken — and the successes' full transitions exist only in the in-memory buffer; the STEP logs don't store the observation vectors, the snapshots are just images). So none of the above can be applied to this run's data.

But that's exactly why fixing the buffer save is worth more than just "resumability" — it unlocks the success-harvesting / iterated-improvement loop (#1, #2) for every future run. If you want #2 specifically and don't want to wait for a full buffer fix, a lighter alternative is a targeted hook: have env5 dump the full transition trajectory of each TargetReached episode to disk (like the pre-bif(11) checkpoints, but the whole episode) — that captures the superior demos directly, even without persisting the entire buffer.

## So my recommendation shifts: fixing the PER save isn't just bug-cleanup — pair it with a "harvest successful episodes → next-gen demos" step, and you get a compounding training loop. Want me to design that (buffer-save fix + success-harvest → re-seed pipeline) as the Plan v10 follow-up?

with aiwac and 2 stream balanced streaming isn't self iterating imrpovement already happening?
Sharp point — and largely yes, it is. The clean stream + AWAC already form a within-run self-improvement loop:


online success → pushed to buffer → flagged is_clean → enters the 30% clean stream
   → AWAC advantage-clones it → policy improves → more successes → ...
That loop is exactly what's driving the rising success rate (2.4% → 17%). The policy is bootstrapping off its own target-reaching episodes, not just the heuristic seed. So the in-run iterated improvement is real and working — you're right.

The cross-run harvest I described isn't a different engine; it's the same loop, but it addresses three limits the within-run version has:

1. The buffer is a sliding window, not an archive. Capacity is 1e6 and you're at ~900k. Once it's full it's a ring buffer — old successes get evicted as new data arrives. So the clean stream is "recent successes," not "all-time best successes." The first great episodes are gone. A harvest preserves them permanently.

2. It reinforces the current strategy — it can't easily switch basins. This is the important one. The clean stream is full of high-speed successes (gw~14), so AWAC keeps cloning high-speed. The loop improves within the high-speed regime (more high-speed successes), but the ~7% gentle successes are a minority that gets out-weighted. So the within-run loop is unlikely to spontaneously discover the gentle-reliable strategy even if it has a higher ceiling — it's self-improving inside a local optimum. Escaping it needs an external nudge (the gentle-turn penalty to bias exploration, and/or re-seeding with curated gentle successes).

3. No reset on degradation. If the run destabilizes (the action-rail collapse / critic strain we saw), the loop has no way to recover — a degraded policy pollutes the clean stream and the loop reinforces the degradation. A harvest-and-restart gives a clean slate from the best curated data.

So the framing: the within-run loop is the optimizer; the cross-run harvest is curation + basin-hopping + checkpointing the best demos. If the current high-speed loop keeps climbing to a satisfying rate, you may never need the harvest. You'd reach for it if (a) the rate plateaus below what gentle could achieve, (b) the run degrades, or (c) you change the reward/arch and want to carry the good demos forward.

Net: nothing urgent to add — the self-improvement you built is functioning. The harvest matters mainly as insurance against the ring-buffer eviction + the high-speed-local-optimum lock-in, which become relevant only if 17%-and-climbing stalls.


--------
# what are all the possible avenues you mentioned to improve this AWAC training for V10; can you reanalyse deeply and list all the possible avenues of improvement

Here's a deep reanalysis grouped by mechanism and effort. The diagnosis is unchanged: critic stable, but AWAC's policy loss has no force that resists the mean railing into tanh saturation → policy converges to advantage-weighted clone of the buffer's (mostly saturated) high-A demos. Every lever below attacks that failure from a different angle.

I'll mark each lever with [in-scope] (RL/AWAC-stabilization knob, no reward/obs/terminal change) or [out-of-scope] (would touch reward/obs/terminal — listed for completeness only, per your standing rule).

Tier 1 — Direct attacks on the policy-mean rail (highest impact, in-scope)
1.1 Entropy bonus on AWAC policy loss [in-scope] — the direct fix.
Add + β · log_pi to the policy objective (with β small, e.g. 0.01-0.05). log_pi = Gaussian term + tanh-Jacobian −Σ log(1−aᵢ²) — the Jacobian explodes as the mean saturates, so this term directly penalizes mean rail. Unlike log_std_min, it doesn't just floor variance. AWAC's literature occasionally calls this "soft AWAC."
Code change: ~5 lines in sac.py _update_policy AWAC branch. Plan v10's deferred lever for "entropy collapse with floor insufficient."

1.2 Raise awac_lambda 3 → 6/8/10 [in-scope]
Softer advantage weighting. AWAC's exp(A/λ) at λ=3 is sharply peaked on the few highest-A (saturated) actions; raising λ flattens the weighting toward uniform BC over the whole buffer. Slows the saturation feedback loop. Cheap to test (one CLI flag). Compose with 1.1.

1.3 Tanh-margin penalty on the policy mean [in-scope]
Penalize E[max(0, |tanh(μ)| − 0.9)²] or E[soft_plus(|tanh(μ)| − 0.95)] — a barrier that activates only when the mean approaches the rail. Cleaner than 1.1 (doesn't touch entropy semantics) and explicit. ~10 lines.

1.4 Asymmetric policy-vs-critic learning rate [in-scope]
Currently LR=3e-4 for both. Drop policy LR to 1e-4 while keeping critic at 3e-4. Standard actor-critic stabilization; smaller policy steps slow mean drift toward saturation without slowing critic convergence. Trivial change.

1.5 Spectral / layer norm on policy network [in-scope]
Add LayerNorm after each hidden layer in the policy MLP. Bounds activations, prevents the linear head from producing unbounded means. Architectural change but small (~3 lines). Composable.

Tier 2 — Update-schedule and replay-buffer levers (in-scope, composable)
2.1 Lower update_per_explore_step 1.0 → 0.25-0.5 [in-scope]
Already discussed — slows the collapse feedback loop and improves buffer diversity. Cost: lower updates/wallclock; at 0.5 you'd have ~half the updates in the same time. Combine with Tier 1, not alone.

2.2 Lower PER alpha 0.6 → 0.3, or disable PER [in-scope]
PER amplifies high-TD transitions; in this regime the high-TD ones tend to include the saturated-success transitions and the −5 terminal transitions, so PER is reinforcing the saturation. Uniform sampling on a balanced buffer might be calmer. Cheap A/B.

2.3 Raise balanced_fraction 0.3 → 0.5 [in-scope]
Force more clean-thread (reached_target_daughter=True) transitions per batch. Risk: if all clean threads are saturated, this accelerates saturation. Worth A/B in combination with the entropy bonus only.

2.4 PER demo priority bonus on heuristic-seeded transitions [in-scope]
The 10 heuristic demos are gentle (not yet saturated) trajectories. Bumping their priority floor keeps them sampled forever, so AWAC always sees some gentle-action targets in the BC mix. Plan v8 implemented demo_priority_bonus; just turn it on.

2.5 N-step returns (n=3 or 5) [in-scope]
Currently 1-step Bellman. N-step propagates terminal +3 / −5 rewards faster to upstream transitions, reducing critic variance on sparse terminals. Helps the critic give cleaner advantage signals to AWAC, which then sees more useful relative-action gradients. ~30 LOC in sac.py _get_expected_q.

2.6 More heuristic seed demos (10 → 50-100) [in-scope]
Currently only 10 RCCA-threaded heuristic demos seed the buffer. With balanced sampling at 0.3, the clean stream sees the same handful of demos repeatedly → AWAC overfits to them. More diverse demos broaden the BC anchor.

Tier 3 — Alternative algorithms (still in-scope as "AWAC stabilization replacement")
3.1 IQL (Implicit Q-Learning) [in-scope]
Replaces AWAC's exp(A/λ) weighting with expectile-regression on V, then weighted BC. Empirically more stable than AWAC on imbalanced offline buffers — doesn't suffer the same mean-rail concentration. The buffer + critic infra is reusable. Code change: ~150 LOC for a new algo/iql.py.

3.2 BC-warm + SAC finetune [in-scope]
Pure BC on the clean-thread buffer for 5-10k updates (no critic involved), then switch to SAC (--algo sac) for online improvement. SAC's mandatory entropy term (with alpha auto-tuned to target_entropy = −4) actively prevents mean saturation. The hard part (critic stability) is solved by the warm-started weights from BC. Best test of "what if AWAC's lack-of-entropy is the only issue."

3.3 CQL (Conservative Q-Learning) [in-scope]
Adds OOD-action penalty to Q. Probably overkill here since the issue isn't critic OOD overestimation. Listed for completeness.

3.4 Soft AWAC with auto-tuned entropy [in-scope]
Full SAC-style auto-tuned alpha on top of AWAC's advantage-weighted BC objective. Best-of-both: AWAC clones high-A actions but is constrained to hit a target entropy. ~20 LOC.

Tier 4 — Curriculum / start-state levers (in-scope, Plan v10's natural extension)
4.1 Narrow to 1-2 best states only [in-scope]
Drop pid10145 (10%) and pid20043 (25%). Train on pid19116 + pid3285 + pid4907 only — the three states where the policy already exceeds 19%. Fewer failure modes diluting the buffer; pid10145 alone is 16+9 = 25/200 transitions that are almost all failures. Plan v10's own decision-tree branch.

4.2 Reverse curriculum — pid19116 alone until >80%, then add others [in-scope]
Most aggressive narrowing. If the policy genuinely can't get pid19116 alone to 80%, no broader curriculum will. If it can, that becomes the proven start, and you add pid3285 next.

4.3 Resample pre-bif(11) checkpoint pool — find more "easy" states [in-scope]
The 5 states were curated from one run's heuristic-success episodes. Re-run heuristic from z=345 with stricter "catheter-forward + small gap" criteria to produce ~20 high-quality states with diverse-but-comparable difficulty.

4.4 Per-restore-state success-rate gating [in-scope]
Schedule weights restore selection so harder states (pid10145) are sampled less until easier ones are mastered. Plan v10 idea.

4.5 Restore-state jitter / domain randomization [in-scope]
Add small per-restore Gaussian noise to xtip/dof_positions before scene step. Currently every restore from pid19116 produces an identical start — the policy may be overfitting to one frozen pose. ~10 LOC in env5.reset.

Tier 5 — Run strategy
5.1 Continue from current checkpoint with new knobs [in-scope]
Save the current weights, restart with --algo sac --log_std_min -1 (or AWAC+entropy bonus). Cheaper than a fresh run; tests whether the new knobs can pull the policy off the rail after collapse. If yes, that's the recovery path. If no, fresh start needed.

5.2 Offline RL on the saved 750k buffer [in-scope]
After eval #3, take the buffer + apply IQL (or CQL) purely offline for 100-500k gradient steps with strong entropy regularization. Compare offline-final policy to current online one. Decouples policy improvement from explore generation.

5.3 Fresh restart with composed levers [in-scope]
The cleanest experiment. Drop all training, restart with: --algo awac + entropy-bonus patch + λ=8 + update_per_explore_step=0.5 + balanced_fraction=0.5 + demo_priority_bonus=on + n_step=3 + LR_pol=1e-4 + 50 seed demos + reverse-curriculum start = strongest in-scope combination. Expected eval ~50-60%.

Tier 6 — Out-of-scope by current rule (listed for completeness)
6.1 ⚠ Reward shaping — relax depth pressure
The deep-target reward currently rewards high-speed push (139× the high-speed policy hit it). Softer depth requirement (target zone not point) or capped progress reward would let gentler policies win. Plan v10 deferred lever.

6.2 ⚠ Saturation penalty in reward
r += −γ · max(0, |a| − 0.9)² directly penalizes near-rail actions through the reward signal. Reward change.

6.3 ⚠ Action history in observation
Add last 4 actions to obs (16 dims). Lets the policy see its own rotation drift → could break the rotation-rail cycle. Obs change.

6.4 ⚠ Truncation tuning — relax OFF_BRANCH_GRACE_STEPS 50 → 100
Gives the wire more time to recover from an off-path excursion. Terminal-condition change.

6.5 ⚠ Lower cross-track penalty weight
Plan v10's Phase-4 fallback. Reward weight change.

Recommended composition (in-scope only)
If you wanted the strongest single experiment, I'd compose:

Tier 1.1 (entropy bonus) — the direct mean-rail fix
Tier 1.2 (awac_lambda → 8) — softer cloning
Tier 1.4 (LR_pol = 1e-4) — slower policy drift
Tier 2.4 (PER demo priority bonus) — protect gentle demos
Tier 4.1 or 4.2 (narrow to 2-3 best states) — denser success signal
Six small code changes, no reward/obs/terminal touched. Expected outcome on a fresh run: policy entropy holds in [−4, −6] range (vs current −10), eval climbs to 45-60%.

The single best bisection test if you want to identify the dominant cause: Tier 3.2 (BC-warm + SAC finetune) — SAC's mandatory entropy makes the rail impossible, so if it works, you've confirmed entropy is the only issue. If SAC also plateaus, the problem is deeper (likely the reward structure itself, Tier 6).

