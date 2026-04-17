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
