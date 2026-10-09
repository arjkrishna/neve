# verify_tbo: adversarial check of tbo_history, tbo_geom, tbo_cause and va_narrow

Prefix `verify_tbo` (state: `state/verify_tbo.md`). All code and outputs are in `/sp/v5_va/tbo/verify/`. Repo, anatomies and code were read-only: containers mounted `/repo:ro`, and nothing was written outside scratch.

## Answer so far (final)

**Verdict: the orientation findings hold.** I wrote my own extraction from provenance `xform` and the shipped centrelines and reproduced every headline number and every flag. The documented rule (frame match plus `anchor_trim`) is in the code as cited. Re-running it on the build's own inputs gives the shipped poses, and the v5 tops are a plain Kabsch fit. The weak parts are some numbers in the candidate-fix tables, and one narrow-VA attribution (details below).

### (a) Own re-measure: `vmeasure.py` → `vmeas_v{3,4,5}.json`; `vcompare.py` → `vcompare.txt`
- **Agreement with tbo_geom.** Over 997 pieces in each of v5 and v4, and 223 in v3:
  - rise, up_tilt and droop agree to 0.05 or better (rounding);
  - native rise agrees to at most 1.1 mm. I took it from the source TopBrain curve, not by back-mapping; back-mapped points lie at most 1.35 mm from the source.
- **Flags.** Using the stated rules, my tier-A flags agree on 997/997 pieces (v5), 997/997 (v4) and 223/223 (v3), and the primary flags agree as well. Anatomy counts match: v5 141 (siphon 47, VA 115), primary 218; v4 171 / 189; v3 11 / 25.
- **30 random pieces tbo_geom left unflagged** (seed 20261003): I flag none of them either. No missed flags and no false ones.
- **Every key count reproduced:**
  - LICA (rise < 20 or tilt > 40): 50;
  - LICA rise < 20: 19; negative: 10; |error| > 20: 34;
  - LICA seam tangent below 10°: 35 (30 of them flagged); below 0°: 29;
  - minimum RICA seam elevation: 25.8°;
  - |yaw| > 90°: 111 RICA, 107 LICA;
  - VA ends crossed by more than 10 mm: 47; median separation 26.6 mm; worst crossing 34.8 mm;
  - fit tops with tilt > 40°: 99/211 right, 113/218 left.
- **The user's cases:**
  - k_011_left LICA rise: −38.9 / −38.2 / −30.1 / −2.2 / −1.1 mm against native 61.5 / 68.9 / 51.4 / 62.3 / 59.4, tilt 96–136°, seam −57.4 to −58.7°.
  - w_003_left: both siphons cross. The ICA termini sit at du −2.9 (mr_007_L) and −48.0 (mr_010) mm against native +36.8 and +25.6.
  - mr_025_L RVA: rise −6.4 against 33.0, tilt 66.4°.
  - mr_027_L: the RVA end lies 21.2 mm left of the LVA end (x 28.4 against 7.2).
- **Identity check.** For frame pieces, |up_tilt − |e_src − e_anchor|| has a median of 1.6°, p90 5.5° and max 11.7° (n 568). "Exact" holds up to tangent sampling.

### (b) Code paths: every cited line read
- **Present as cited:**
  - `graft_siphon.py`: UP :73, frame / frame_rotation :157-169, anchor_trim :172-184 (used :238, :257-260), mirror :485-494.
  - `graft_three.py`: anchor tangent deliberately untrimmed :307-310, place :390-409, head_trim :412-436 (used :467), seam 1 :509-511, seam 2 :530-531, anchor_trim added :661.
  - `graft_v4.py`: tb_part :184-208; lower_records :268-318 drops the 58 mm and 2 mm-tip rules (comment :294-301); left_lower_candidates :338-354; graft_va :366-385; extend_eca :388-407; compose :577 with gates :580-598 (kink, ECA, seam landing, clearance; no rise or tilt).
  - `carotid_tools/v5/graft/graft_v5_va.py`: TB_PLACE "fit" :106; R_frame computed :458 and discarded when the fit is used :465-466.
  - `graft_v5_reg.py`: the "rise" gate at :230 is the donor's 10 mm chord at the origin.
  - route13 turn: `graft_v5_reg_sep.py:197-212`.
  - The scratch originals sit one line earlier (−1).
- **No rise, tilt or heading gate anywhere** in graft_three / graft_v4 / graft_v5_* / validate_anatomies / check_anatomies. The same holds on origin branches: rl_improv_18_p2 and origin/master have no such gate, and nothing after a898eb2 adds one.
- **Docs and commit as cited:** a898eb2 (2026-09-01, ORIENTATION section), TOPBRAIN_PIPELINE C1/C2, CAROTID_THREE_SOURCE 100-134, V2_BUILD_PLAN 77-78, VERIFY_f292143 439/461 "not covered", CAROTID_V4 verification "1 443, all proper rotations".
- **Nits:**
  - (i) The v4 VA cut search (`graft_v4.py:678-700`) moves the cut for join > 60 as well as for clearance. tbo_history says "clearance only".
  - (ii) The `graft_splice` docstring (`graft_v5_va.py:403-404`) still says "frame-matched (tangent + superior)". That is stale: the rotation is `tb_rotation` "fit".
  - (iii) The `kabsch()` used by "fit" subtracts centroids with uniform weights. "Start pinned" therefore pins only the translation, not the rotation. My centroid-Kabsch re-run matches the shipped R to 0.000°.

### (c) One re-run per class: `vrepro.py` → `vrepro.txt`; join checks in `vdebug*.py`
- **Cause re-run against the shipped R:**

  | class | piece | re-run vs shipped |
  |---|---|---|
  | L_PITCH | k_011 mr_012_L | 0.85° |
  | L_PITCH | w_024_right mr_002 | 0.16° |
  | R_PITCH | w_035_left mr_024_L | 10.0° |
  | S_HEAD | w_003_left mr_010 | 0.48° / 0.23° |
  | VA4 | mr_025_L | 5.09° |
  | ECA | k_004_left mr_010 | 2.48° |
  | VA5 (fit) | mr_027_L ×2, m_022 mr_020, k_011 mr_012_L | 0.000° |

  C6: 168 pairs, median 110.1°, above 90° in 103. All confirmed.
- **Numbers that reproduce:**
  - L_PITCH bend60: rise 7.0, tilt 85.1; w_024 bend60: 37.0 / 71.2.
  - S_HEAD anat: termini du +4.0 (cause +4.7); RICA still 17.2 past the midline; LICA clearance −2.64 against the LVA.
  - VA4 cut 13 mm lower: rise 30.9, tilt 4.3.
  - VA5 anat: ends 3.9 mm apart; VA-to-VA gap 0.03–0.25 mm (touching).
  - ECA bend60: rise −12.6.
- **Numbers that do not reproduce, or that depend on the metric:**
  - The 1 mm join kink at a re-posed seam changes by up to 44° depending on how the piece is sampled. I compared the tb_part source with the back-mapped shipped piece:

    | case | source | back-mapped | cause |
    |---|---|---|---|
    | mr_025_L, cut 13 mm lower | 57.9 | 13.4 | 13.5 |
    | mr_027_L LVA, anat | 56.9 | 69.5 | 74.7 |
    | k_011, anat | 155.8 | – | 106 |

    The 5 mm chord angles are 93–156° for the clear failures and 55° for the mr_027_L LVA. So the mr_027_L LVA "anat fails the join" is borderline, not a clear failure.
  - **atrim depends on the tangent definition.** The cause searched with a CENTRED tangent and then placed on the backward one. With the backward tangent, which is the one `place` uses, the search lands elsewhere:
    - k_011: 8 mm back (rise 47.8, tilt 34). Clearance 0.06 mm to the LECA, so it still fails.
    - k_004 RECA: an anchor rising ≥ 30° does exist 12 mm back. The cause said there was none. It still fails: clearance −1.5 against the RCCA, and the fork ECA is only 17.9 mm long. **The conclusions stand; the "none within 30 mm" wording does not.**
  - **R_PITCH showcase w_035_left mr_024_L bend45** (cause: clearance 0.80, "pass"): I get 0.39 mm (source radii) or 0.56 mm (shipped radii) against the RVA. That fails the 0.6 gate, so it is borderline at best.
  - **v3 lower rule:** with match_sections' `DEFECTIVE_LOWERS`, it refuses **26 left lowers / 125 anatomies**. The cause's 23 / 113 left that list out. Coverage of tier-A L_PITCH is 34/35 either way.

### (d) Narrow VA, re-checked end to end on case_w_025_right__topcow_mr_021_L RVA (`vnarrow*.py`, `vpassage2.py`, `vascan2.py`)
- **Reproduced:**
  - At the centreline control points: 0.632 at s 219.7, in the TopBrain piece (fit, starts at s 185.94, TopBrain arclength 38.8).
  - collision_full: 0.924.
  - The TopBrain source MISR drops to 0.39 there, against a VA median of 1.94. The stretch is horizontal (z range 1.2 mm over 11 mm), 4.8 mm above the source's lower end.
  - v4 at the same TopBrain point: OBJ 0.847, full 0.907.
  - Best in-section wire path: 0.821 (va_narrow: 0.829).
- **Not reproduced: the RVA's deepest point is on the DONOR stretch.** Sampling the centreline polyline every 0.05–0.1 mm finds **0.547 mm at s 158.65** (Song 113 L, radius at the 1.0 mm floor; under 0.65 over s 158.2–159.0). That is lower than 0.632. va_narrow's "donor stretch min 0.665 at s 158.2" is a control-point artefact.
  - So CAROTID_V5.md's attribution of this VA to the donor floor is right for its deepest point, but its value is the TopBrain point. va_narrow's record correction is itself only half right: both stretches are below 0.65, and the deeper one belongs to the donor.
  - The best wire path at s 158.65 is still 0.708 mm, so the passage verdict holds.
- **Fine scan of all 446 VAs** (3 mm end caps excluded, as in `checks/va_lumen.py`):
  - Control points: 3 VAs below 0.65, median 0.834. This matches the v5 record.
  - Polyline: **4 VAs** below 0.65 (median 0.819). The extra one is **case_w_047_left__topcow_mr_023 RVA, 0.637 at s 152.8 (donor)**.
  - The other polyline minima: w_025 LVA 0.567 (control points 0.578), w_027 RVA 0.630 (0.646).
- The CT, bone-scan and SOFA parts were not re-run.

### (e) Nothing changed
- **Tracked files.** `git status` is clean for carotid_data/anatomies_*, topbrain_tools and carotid_tools.
- **Files modified after 17:20.** They are 27 `anatomies_v5/_records/*` record files, `carotid_tools/v5/{FILES.tsv, MANIFEST.md, graft/route13/graft_v5_reg_sep.v2.py}`, `CAROTID_V5.md` and `.gitignore`. All were written between 17:25 and 17:28 and committed in 5c2cdc0 / 113c89a at 17:32. They are the orchestrator's v5 commit; no geometry file changed.
- **Older trees.** anatomies, v2, v3 and v4 have no file newer than 2026-09-27.
- **User notes.** `user.md` was last written on 2026-09-11. `user copy.md` was written at 17:06:59, before any tbo agent started (17:22). It was already modified when the session began, and nothing touched it afterwards.
- The tbo containers mounted /repo read-only.

## Corrected statements
1. tbo_history: "w_003 … rise −12 / −19 mm" are rise ERRORS. The rises are 56.0 / 51.8 against native 67.9 / 70.7.
2. tbo_history: the v4 VA cut is moved for join kink and clearance, not for clearance only.
3. The "fit" Kabsch subtracts centroids. "Start pinned" applies to the translation only.
4. tbo_cause candidate fixes:
   - The join values for re-posed pieces carry ±20–45° of sampling uncertainty. mr_027_L LVA anat is about 55–75°, borderline rather than a clear failure. mr_025_L cut 13 mm lower is 13–58°, under 60° either way.
   - R_PITCH w_035_left mr_024_L bend45 has 0.39–0.56 mm clearance by my measure (cause: 0.80), so it does not clearly pass.
   - The k_004 RECA has an anchor rising ≥ 30° 12 mm back, but it fails clearance (−1.5 mm) and would cut the 17.9 mm fork ECA. "Nothing passes" stands.
   - The v3 lower rule refuses 26 lowers / 125 anatomies, not 23 / 113.
5. va_narrow and CAROTID_V5:
   - The w_025 RVA minimum is 0.547 mm at s 158.65 on the donor stretch (best path 0.708). 0.632 is the TopBrain point. Both are below 0.65.
   - On the OBJ polyline, 4 VAs (not 3) fall below 0.65; the fourth is w_047_left mr_023 RVA at 0.637.
   - The w_025 LVA and w_027 RVA minima are 0.567 and 0.630.

Status: complete
