# verify_s2_gate -- adversarial verification of stage 2 (v6), GO / NO-GO for the full build

Agent verify_s2_gate, 2026-10-06. State: `state/verify_s2_gate.md`. Own code (numpy / scipy only, no project measuring code):
`scratchpad/v6/verify_s2_gate/` (vlib.py, audit14.py, route14.py, recount223.py, turns223.py, turns2mm.py, d17dir.py,
chirality.py, cl223.py, d15check.py, d15b.py, hostdrop.py, lowersrc.py; runner dr_vg.sh mounts /bld and /rp READ-ONLY).
Outputs: `carotid_data/v6_work/verify_s2_gate/` (/out). Containers vs2g_* only (all exited). No git writes, no repo-tracked
tree touched, no other agent's file edited.

## Answer so far (verdict: NO-GO)

The stage-2 tools mostly do what the decisions say, and most invariants re-measure clean with my own code. A full
build cannot be expected to pass every invariant yet. Three things block it, and two more should be fixed first:

1. **Right-route fork gate (>= 0.99 of v4, binding invariant): fails on 7 / 14 re-piloted anatomies.** This is
   confirmed, not a tool bug. My own test shows the moved fork's surface beyond the seam-1 blend IS v4's surface,
   rigidly carried. Mirrored forks fit with det -1 and native ones with det +1, residual 0. v6 vertices lie on the
   carried v4 surface with p95 0.054-0.079 mm, and 0-0.78 % of the area is more than 0.3 mm off. The only real changes
   are the D19 blend zone (25-40 % of its area moved > 0.3 mm on 8 forks; 4-5 % of the fork) and the bake's handle
   windows (w_035, w_051, w_025). The relief-share statistic is not invariant under a rigid / mirror move at the 1-10 %
   level. Examples: w_003 x2, w_018b and m_006 mr_025_L have surfaces identical to v4 within 0.1 mm on > 99.8 % of the
   fork, yet read 0.978-0.990 in checks_v6, and 0.907-0.951 with my own relief share on the 60k meshes. The gate as
   written will fail about half of the 223 anatomies. The siphon gate sits at the same noise floor (12 DRY siphons
   flagged 0.990-0.995). **Needs a user decision on the measurement** before stage 3, e.g. "same surface, moved":
   >= 99 % of the fork area beyond the blend within 0.3 mm of v4's carried surface. All 14 pass that. Otherwise the
   post-bake fallback ladder will push forks back to v5's pose and lose D19.
2. **Tier-A conflict: 1 / 223 (w_017_left mr_002_L, RVA top at D9 rung 3 "v5's top as shipped").** This breaks the
   "0 tier-A" invariant. Needs a rule: drop to a host VA with an anatomical top (as D21), or accept and flag.
3. **D17 wrong direction: ia172_069 R** (mirrored onto the left in m_006_left mr_025_L, a RE-PILOT anatomy, and
   w_051_right mr_025_L). Its "distal subclavian" (19) runs 38 mm MEDIAL and 87 mm ANTERIOR, ending at x 15.9 near the
   midline instead of going toward the arm. The other 427 D17 sides run 11-92 mm lateral. It is visible in the
   re-pilot figure (orange loop in front of the neck). No gate tests the D17 direction, and the same two anatomies have
   a 108-116 deg corner at the (18) seam. Fix the path or demote the side (D18), and add a direction gate (cut end
   lateral of the ostium, < 90 deg from the host's distal subclavian).
4. **(Should fix) 1 mm joins are gated on written vertices only (geom_v6, graft_v6_va T2), so a spread corner passes.**
   Resampled at 1 mm at any phase, these new seam turns exceed 60 deg:
   - (19) 61.1 deg on 8 mr_014_L anatomies (song_095 R's spread reversal; written 58.7 deg);
   - (18) 64.5 deg on the 2 mr_025_L anatomies.

   Net turns over 2-3 mm at these seams reach 95-116 deg on 18 anatomies (song_040 R (18) x8, ia172_069 R (18) x2,
   song_095 R (19) x8). Gate the 1 mm turn at every sampling phase (T1's rule) on all new seams.
5. **(Should fix) Records.** v6_build.va.<side>.donor_id is null on 13 rung-2/3 sides although xform.vaC_* holds the
   donor (BUILD_v6 "record == provenance"). stage2_dry.json says that both redone anatomies ran graft_v6_va 1.2.3, but
   w_018_right mr_022 ran 1.2.2 (substantively the same, since its targets are 9-11 mm). 221 anatomies ran 1.2, so
   stage 3 should run a single toolchain.

## Checks reproduced (no refutation)
- Read-only trees untouched: git status clean on anatomies_v2..v5, topbrain_tools and carotid_tools (ignored files
  included). The latest mtimes there are 10-03 17:26, before the v5 commit at 17:32. anatomies_v6, carotid_tools/v6 and
  CAROTID_V6.md do not exist yet. The tool md5 manifest matches.
- DRY 223 (recount223.py, lowersrc.py, cl223.py):
  - Left side: 0 host-LCCA fallbacks (LCCA != host, LECA and lower_L present on 223); left rung 1 201 / rung 2 22, no
    ICA extension. Lower names equal the xform sources on 223 / 223.
  - Right fork rungs R1 118 / R2 83 / R3 13 / R4 4 / R5 5. Contralateral == mirrored on both sides (113 / 101).
    Seam-1 bend <= 60 everywhere; left <= 45.
  - VA: donor never dropped or changed vs v5. Donors one-to-one (also in the split run: V: 25 / 25). VA rungs 416 / 12 /
    1, host 17. va_target bands 436 / 9 / 1.
  - D10: my own geometry gives 421 / 429, matching the records with 0 mismatches.
  - ICA AP recomputed from the files equals the records; > 40 mm on 3 anatomies.
  - The RCCA prefix (arch -> v5 seam 1) is bit-identical to v5 and v4 on 223 / 223.
  - Own centreline clearance (moved groups, new vs v5): 0 new contacts < 0.6.
- Split (own rerun of v5_split.py on /bld/dry223/final): exit 0, none of 45 lowered (`/out/split`).
- Re-pilot 14 (audit14.py, own code):
  - OBJ genus 0, closed, consistently oriented, 1 component, 0 self-intersections (own Moller-Trumbore on 23.5k OBJ
    and 60k full).
  - Every centreline point is inside (|winding| = 1). OBJ normals point inward, the same convention as v5.
  - VA polyline lumen min 0.664 mm (w_050).
  - RCCA prefix bit-identical. The OBJ near the prefix is within 0.18 mm of v5's (rebake noise).
  - No new centreline clearance < 0.6.
- D20: the 10 mirrored re-pilot forks are exact mirror images of v4's (det -1) and their baked surface matches, so the
  winding is right. Population test on 223: each fork keeps its source side's ECA-medial share through the mirror
  (L-donor forks 0.81-0.85, R-donor forks 0.49-0.54, on either route). ECA anterior 0.85-1.00 (v5 right native 0.24).
- D17 host distal dropped (hostdrop.py): no v5 host-distal path inside the v6 OBJ on any D17 side of the 14. Every cut
  end is free: 29-92 mm from other curves.
- D15 (d15check / d15b, my own station projection): changed vertices lie within 1.4-6.3 mm of the zone + blend edges.
  The surface moved > 0.3 mm outside them only at donor VA distal end caps (beyond the splice, unused) and on
  ia172_071 L (<= 2.25 mm, 1-4 mm past the blend). This is consistent with their voxel-based identity claim within the
  projection tolerance, so it is not refuted.
- SOFA RCCA re-pilot (their output files): "passed" 14 / 14 means soundness only. Reach (bifurcation / siphon / target)
  41 / 34 / 22 for v6 against 41 / 33 / 19 for v5. Per anatomy it dropped on w_025 (target 3 -> 1), w_047 (siphon
  1 -> 0) and w_018b (siphon 2 -> 1).

## Decision-implementation notes (code read: compose_v6 3.0.3, graft_v6_va 1.2.3, va_corridor 1.0)
- D1: the ladder has rungs 1-6 and no host path. D11 left: 215 posed by the map. The 8 that are not: 2 use v5's pose,
  and 6 use a frame match (4 substitutes, 2 mirrored kept forks). D11 said "else v5's pose"; the frame match is
  D20-consistent and flagged.
- D19: R4 (deviation within the fork gates; 4 anatomies) and R5 (v5's pose on 5 NATIVE forks: w_002 mr_013_L,
  w_002 mr_022, w_003_right mr_001, w_017_right mr_011, w_036_right mr_018_L) are tool additions beyond D19 as written.
  R5 forks are > 90 deg off the map (orient forks yaw_gt90 5), and w_003_right mr_001 is one of the 3 AP > 40. They are
  flagged; the user should acknowledge them.
- D8 / D9 / D10 / D16 / D17 / D18 match the addendum. D10 rung-2/3 inherited failures are flagged, not pushed further
  (8 sides). D21 is out of scope (17 host sides keep host VAs with anatomical tops).

## Files
`/out/recount223.json`, `/out/split/`, `/out/audit14/`, `/out/route14/`, `/out/turns223.json`, `/out/turns2mm.json`,
`/out/d17dir.json`, `/out/cl223.json`, `/out/d15check.json`, `/out/d15b.json` (/out = carotid_data/v6_work/verify_s2_gate).

Status: complete
