# verify_v6_final2: independent verifier of s3_fix on carotid_data/anatomies_v6

Agent verify_v6_final2, 2026-10-08. State file: `state/verify_v6_final2.md`.
Own code: `scratchpad/v6/verify_final2/` (numpy / scipy; vtk only to read .vtp / .stl). Library `vf2_lib.py` (exact
point-to-triangle distance with a certified KD search, pseudo-normal inside test, winding number, topology, own
triangle-triangle self-intersection test over disjoint and vertex-sharing pairs, edge-adjacent fold test), self-tested
by `test_lib2.py` (brute-force distance error 0.0; pseudo-normal vs winding inside agreement 1.0 on 2,000 points; sphere
genus 0 / 0 SI; torus genus 1; two interpenetrating spheres 228 intersecting pairs). Runner `dr.sh` (containers `vf2_*`,
/repo read-only, /w = v6_work read-only, at most 4 at once). Outputs `scratchpad/v6/verify_final2/out/` (= /out below).
No repo file written, no git writes, no other agent's file edited.

## Answer (final): s3_fix holds; NO-GO on one small record item (unflagged sub-0.1 % real-wall losses)

Every claim of s3_fix reproduces with my own code (coverage on all 223, strict audit, D22, loader, SOFA, split, hashes,
BUILD vs provenance). The 8 lost-wall pieces are restored (0.0 against v4 and v5), the 8 at 0.1-1 % are flagged, the
fold-over is gone. Must-fix: acceptance criterion (a) says every piece under 1 % with a loss is flagged; 10 pieces with a
measurable loss below s3_fix's 0.1 % flag line carry no flag, and one of them (w_034_left mr_008_L fork, 0.82 mm2 of the
carotid bulb 1.1-1.4 mm outside the v6 wall, on both references) is a real patch, not noise. Record-only fix: stamp
d27_coverage flags on the 10 (or at least w_034_left mr_008_L and state the noise floor with evidence), regenerate
BUILD_v6.json, update the doc. Should-fix: the stale doc numbers in section 4.

- **(a) Coverage, all 223 right forks and siphons, own code (`vf2_cov.py`, /out/cov).** Against v4's and v5's SHIPPED
  collision_full, measured on the v6 union (primary) and on the shipped v6 collision_full (secondary):
  * 0 pieces lose >= 1 %. The 8 re-baked pieces lose 0.0 on both references and both v6 walls.
  * 0.1-1 %: exactly the 8 flagged pieces, values equal to the flags (w_035_left mr_015_L S 0.77, w_012_left mr_001 F
    0.60, w_012_left mr_013_L F 0.56, w_025_right mr_021_L F 0.46, w_008_right mr_001 F 0.43, w_017_right mr_011 S 0.42,
    w_047_left mr_027 S 0.22, w_034_left mr_011 S 0.14 %); each carries its `d27_coverage_*_accepted` flag in provenance.
  * **10 pieces lose more than 0 but less than 0.1 % and carry no flag** (s3_fix's flag threshold is 0.1 %): w_046_left
    mr_026 F 0.097 % (0.124 % against the shipped collision_full), w_012_right mr_012 S 0.070 % (0.115 % on the shipped
    wall), m_024_left mr_023_L F 0.057, **w_034_left mr_008_L F 0.056 % (0.82 mm2 at RCCA s 69.0-69.75, 1.10-1.41 mm
    outside the v6 wall, on the own scan, on both references: a real patch, not noise)**, w_042_left mr_010_L S 0.045,
    m_024_left mr_001 F 0.045, w_051_right mr_025_L F 0.015, w_012_left mr_013_L S 0.013, m_006_left mr_022_L S 0.008,
    w_018b_left mr_002 S 0.005 %. 9 of the 10 are 1-2 reference cells lying 0.31-0.59 mm off (decimation-noise level).
  * Pre-fix reproduction (the 17 folders s3_fix replaced, own code): exactly s3_fix's before-list (8 at 1.0-4.7 %, the 8
    at 0.1-1 %), to 0.01 mm2.
  * No piece skipped (same source surface on all 446 x 2), frame check (RCCA points inside the union) 1.0 everywhere,
    pseudo-normal vs winding sign agreement 1.0, shipped collision_full within 0.26 mm of the union.
- **(b) Changed anatomies.** Strict audit (own, 9 geometry-changed: OBJ, collision_full, union): closed, edge- and
  vertex-manifold, oriented, 1 component, genus 0, 0 degenerate faces, 0 self-intersecting pairs, 0 edge-adjacent fold
  pairs below 1 deg on OBJ and collision_full; every centreline control point inside the OBJ; centrelines identical to
  the pre-fix folders (so clearance, 1 mm joins and orientation are unchanged from the verified s2 tree); VA polyline
  lumen min 0.779 mm (w_052 LVA). Fold test validated on the pre-fix m_022 OBJ (finds faces 15297 / 15302 at 0.80 deg,
  the first verifier's pair; 0 after the fix). Folds on all 223: OBJ 0 below 1 deg; collision_full only w_017_left
  mr_014_L (0.91 deg, information, as recorded). Loader (own call, 17): 17 / 17. SOFA: the build's sofa_rcca_v6 re-run on
  the shipped w_002 reproduces its record exactly; the 9 records ran on byte-identical OBJs.
- **(c) Records.** Own sha256 of 5,136 files == s3_fix manifest 024edcd2 (== the _records copy); RECORDS.sha256 -c ok;
  vs the s2 manifest exactly the 43 intended files changed. BUILD_v6.json == provenance on every recount (digest,
  per-anatomy sha, flag counts, per-anatomy flags 223 / 223, derived flags recomputed own 223 / 223, s3_fix counts,
  rungs, D22 references, coverage blocks). Toolchain manifest 2288f778 passes md5sum -c (97 entries). CAROTID_V6.md:
  105 claims checked, 96 consistent; the rest are stale numbers or wording (section 4).
- **(d) Split**: own run exit 0, 45 compared, none lowered, the same 4 raised quantities.
- **(e)** Read-only trees unchanged; user.md / "user copy.md" untouched since the first verifier.

## 1. Coverage method (vf2_cov.py)
Per piece and reference: the reference's cells owned by its own RCCA (tube-distance rule against every other curve's
samples off the RCCA) in its gate [span start + max(blend v6, blend ref) + 2, span end - 2]; carried by the piece's own
transforms (src = M_R (R_R^T (x - a_R) + o_R); x6 = R_6 (M_6 src - o_6) + a_6); kept where their v6 RCCA arclength is in
v6's gate and outside v6's handle windows (+- 0.5 mm); uncovered = exact distance to the v6 wall > 0.3 mm; LOST = also on
the piece's own scanned surface (<= 0.3 mm, placed by the v6 xform) and outside the v6 wall; loss = lost / kept area.
Also recorded: real-scan cells uncovered but INSIDE the v6 wall (not counted by the rule): > 1 mm2 only on
m_030_right mr_003_L S 1.75, w_002_right mr_013_L F 1.70, w_017_left mr_014_L S 1.16 mm2.

## 2. D22 on the 8 re-baked anatomies (own, v4 SHIPPED + own scan; /out/audit/d22)
k_011 F 0.99999 / S 0.98984 (the 17.79 mm2 D27 window, as before); w_002 F 0.9963 (the build's 0.988 uses the D13
baseline) / S 0.9944; w_012_left mr_003_L F 0.9993; w_027 x2, w_052 1.0; w_051 F 0.9947 (own surface, as pre-fix).
**w_008_right mr_003 fork: 0.9798** -- 39.7 mm2 off both references, 33.6 mm2 of it on the bake's capsule
(distance from the RCCA - (1.8 r + 1) median 0.018 mm), 7.1 mm2 next to the 0.5 mm bridge tubes; pre-fix 1.0. It passes
only through checks_v6 1.5's generalised D27 repair window (capsule cells out -> 0.9968 = the build). Flagged and
disclosed; a judgment item for the user, not a rule breach (D27 names the repair-window class; the rule was accepted at s2).

## 3. SOFA
Own driver `vf2_sofa.py` (loader + seeded random push/pull/rotate, 2 seeds x 300 steps, own inside test of every
inserted DOF point) on the 9 geometry-changed anatomies from anatomies_v6: 9 / 9 sound (0 exceptions, 0 non-finite,
600 / 600 steps, tip never outside the OBJ, loader mesh = the anatomy's OBJ, insertion inside). Body points up to ~3 mm
outside, as in the build's records for all 223 (dev_outside_max 1.4-4.5 mm, median 1.6): a property of the setup, not gated.
Reproduction of sofa_rcca_v6 on the shipped w_002 and w_008_right mr_003: identical to the records.
Reproduction of sofa_rcca_v6 on the shipped w_002: identical to `_records/sofa/rcca_v6_s3` (per seed and route). w_002
is no longer "RCCA exercised": seeds reach RCCA s 73.2 / 69.3 / 62.3 (ECA take-off 63.3; the past-bifurcation mark is
73.3): seed 777 entered the ICA 9.9 mm and buckled (final tip RCCA s 51), it did not turn into the ECA. The OBJ around
the bifurcation (s 40-90) differs from the pre-fix one by <= 0.5 mm, lumen unchanged; the change is the scripted
controller's sensitivity to the re-sampled mesh.

## 4. CAROTID_V6.md (out/doc/doc_check.json, 105 claims)
Consistent: 96. Not consistent with the final folder (all minor):
- "SOFA RCCA not past the bifurcation 4 (w_042_left)" -- 5 after s3_fix (+ w_002_right mr_013_L); the target line was updated, this one not.
- "handle windows (on the fork in 28 anatomies, on the siphon in 31)" -- the pre-fix counts; final folder 25 / 32.
- w_002's seed "now turns into the ECA at the bifurcation" -- it went 9.9 mm into the ICA and buckled.
- "the 8 s3_fix re-bakes 177-647 s" in a sentence citing provenance bake_v6.seconds -- provenance reads 174-645 s.
- "Undecimated unions are kept only in ... v6_work/full/iso/" (also BUILD records.undecimated_unions) -- the 8 re-baked
  unions are in v6_work/s3fix/iso/final; full/iso holds their superseded pre-fix unions.
- "These few mm2 of wall are the only synthetic surface on the moved right route" -- the D27 capsule repair windows
  (17.79 + 12.01 + 56.57 mm2) are bake capsule surface, not scan.
- "none of [the 25-set's] anatomies changed in s3_fix" -- 2 of them (w_008_right mr_001, w_047_left mr_027) were
  re-stamped (provenance only; geometry unchanged, so the SOFA numbers do stand).
(3 more flagged by my script were rounding only: 108.25 -> 108.3, 120.25 -> 120.3, 1.865 -> 1.87.)

## 5. Other observations (information)
- w_051_right mr_008_L's new (unshipped) union has one edge-adjacent fold pair (0.034 deg at (-0.47, 1.8, 581.2)); its
  pre-fix union had none; its OBJ and collision_full are clean.
- OBJ edge-adjacent pairs with an interior angle 1-5 deg (sharp creases, not overlapping): 9 unchanged anatomies, as in s2.
