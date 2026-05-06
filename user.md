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


Phase C — Full Design Option Space
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
