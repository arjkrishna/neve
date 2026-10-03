# verify_graft_reg: adversarial check of the v5 registry graft tool and the split checker

Agent prefix `verify_graft_reg`. State `v5_va/state/verify_graft_reg.md`. Own code only: `/sp/v5_va/verify_graft_reg_*.py`;
small outputs in `/sp/v5_va/verify_graft_reg_out/`; written anatomies in
`carotid_data/va_sources/v5_reg/verify_graft_reg/` (scratch, gitignored). No git, no GPU, one container
(`verify_graft_reg_w`, eve-training-fixed, `--cpus 6`). Nothing of graft_reg's, v5_split's, the prototype's, v4's or
the repo's was modified.

## Answer so far

**Verdict: holds, with four corrections.** Every key number reproduces with my own code: the dry-run tallies, the
assignments, the compat regression, the v4 / v3 split numbers (all 45 quantities, including div6x6) and the
shared-donor effects. No key number failed. The corrections are enforcement gaps. None of them changes a reported
plan or number today:
- (a) The splice-calibre gate is evaluated at the requested splice, not at the 1 mm sample actually cut. 7 of 2224
  dry passes exceed 1.6 where the cut is made; none is in a plan.
- (b) `graft_v5_assign` does not enforce one donor **patient** per TopBrain patient across unit ids (`patient_key`).
- (c) `build()` and `--plan` trust the plan: handedness, QC and sharing are not checked there. No graft gate bounds
  the synthetic stub or the absolute calibre.
- (d) `v5_split.py` reports a shared donor that lowers no number only inside its JSON: exit 0, nothing on stdout.

(Items below are in the order the results arrived.)

1. **Dry-run numbers reproduce** from the per-anatomy JSONs with my own tally (`verify_graft_reg_tally.py` ->
   `verify_graft_reg_out/tally.json`): 3345 unit-anatomy-sides, 2224 pass (0.6649), every per-unit R / L pair, every
   base per-gate rate. Every passing side's recorded gate values are inside every limit (0 violations).
2. **Plans are one-to-one** and follow the handedness rule: strict 10 units -> 10 patients, 89 / 102 VAs, 53 / 85 / 85
   anatomies; composite 10 -> 8, 92 / 102, 80 / 34 / 109; real-wall strict 53 / 54, 16 / 75 / 132. My own Hungarian on
   my own score matrix gives the same assignment; 191 is also the maximum number of VAs any one-to-one assignment can
   reach with these dry runs.
3. **Geometry re-check** (`verify_graft_reg_geom.py` -> `geom.json`): 22 anatomies (2 per unit, native + mirrored,
   plus 2 composite-plan anatomies), 34 new VAs, written by the tool's own build/write and then checked with my own
   code: exact all-pairs clearance >= 0.6 mm to every other branch, the parents after exit and itself; exit <= 12 mm,
   no re-entry; 1 mm step turns <= 60 deg at O_h, the ostium and the splice cut; det +1 (vaC and TopBrain xforms);
   follow handedness in the written xforms; every non-VA file byte-identical to v4 and the RVA's subclavian prefix
   identical; the written donor stretch lies on the donor record's raw centreline transformed by the written xform
   (<= 0.48 mm) and the TopBrain stretch on its source (<= 0.42 mm); the transformed donor SURFACE sits at the MISR
   radius from the written centreline (median |d - r| 0.03 mm), while flipping the mirror moves it 13-240 mm off.
   All 34 pass. Rebuilds are deterministic (identical to the dry JSONs).
4. **Defect: the splice-calibre gate is evaluated at the requested splice**, but `compose` cuts at the next 1 mm donor
   sample. Re-evaluating the tool's own `calibre_ratio` at the splice actually used: 7 of 2224 dry passes exceed the
   1.6 cap (ia172_015 x2, ia172_141 x5; none is in any plan) and 117 sides flagged "preferred" are above 1.25
   (`calib_sweep.json`). Small, but the gate is not exactly enforced on what is written.
5. The left origin-seam turn "at O_h" is vacuous by construction (the LVA curve starts at O_h): only the ostium turn
   and the take-off band bound the left seam, as the report already framed it.
6. **Compat regression reproduces** with a fresh `--compat` run of my own (`verify_graft_reg_compat.py`): 446 / 446
   sides pass as in the prototype; over 22,300 shared leaves the only difference is the label `tb_len_mm` on the 78
   sides with a TopBrain trim (the registry records the untrimmed length; its `tb_len_used_mm` equals the prototype's
   value). Right origin turn at O_h > 60 deg in 22 / 223, max 97.98, median 44.3, p90 56.9 (the report's numbers).
7. **Absurd inputs** (`verify_graft_reg_absurd.py` -> `absurd.json`; VMR records deformed, 12 anatomies x 2 sides,
   full search):
   - rejected 0 / 24 passing: long (x1.7), short (x0.5), U-turn start, self-crossing loop, bent 90 deg at 60 mm;
   - absurd poses of the real VMR donor: tilt 60 deg 24 / 24 rejected (17 accepted by v4's gates alone), the dropped
     Stage-6 frame match 24 / 24 rejected (8 accepted by v4's gates alone); a 90 deg yaw about the vertical 23 / 24;
   - **not rejected**: VA shifted 30 mm off its subclavian ("detached", a 34 mm synthetic straight stub) passes
     16 / 24: no gate bounds the stub. Real stubs are 3.1-7.4 mm (`stub.json`), so this is latent. Radius x3 ("fat")
     passes 1 / 24: only the splice ratio bounds calibre. A zigzag start passes 24 / 24 because prep_siphon trims up
     to 10 mm of a kinked start and the stub bridges it (no real donor is trimmed > 2 mm today).
   - **plans are not checked by build()**: a wrong-handed spec (donor R unmirrored on a mirrored anatomy) is grafted
     and passes on 5 / 6; a QC-fail side passes when named explicitly; one donor on two TopBrain patients is grafted
     on both. Handedness, QC and one-to-one are enforced only where plans are generated (`unit_plan`,
     `graft_v5_assign`), and after the fact by `v5_split.py`.
8. **v4 split recomputed independently** (`verify_graft_reg_split.py`, own extraction + own MILP): v3 definition
   153 / 16 / 54 (169) on v4 and on v3; max 183 / 16 and 181 / 17; every donor counted: roster 127 / 16 / 80, max
   174 / 16 (190) and 172 / 17 (189); v3 set with Zenodo per patient: 144 / 16 / 63, max 180 / 16, 177 / 17. One
   component. The block holds 16 anatomies, so the user's "153 / 17" is 153 / 16.
10. **Defect: `graft_v5_assign` does not enforce one donor PATIENT per TopBrain patient.** It reads each unit's
    `patient_key` and lists duplicates in the summary, but neither solver nor its split assertions use it. With
    ia172_058 present under two unit ids (same `patient_key`), strict and composite both assign it to topcow_mr_020
    and topcow_mr_005 and the module's assertions pass (`verify_graft_reg_dupkey.py` -> `dupkey.json`). Today's ten
    units have distinct keys, so the reported plans are unaffected; the gap opens as soon as the pool holds the same
    patient from two datasets (172-series / Large IA / RSNA, which the split notes already flag).
11. Take-off numbers reproduce (`takeoff.json`): preserved 63.9% when the subclavian elevation gap <= 15 deg
    (1092 VAs) vs 37.6% above; Spearman 0.384 over 30 groups; VMR median |delta| 22-44 deg. Plan verification changed
    nothing in any of the three variants (one round, 0 changed). No splice window fell back to [100, L - 10].
9. **Inputs moved on**: `pilot_proc_ia172.json` changed after the records were built; 015 R, 058 L and 141 R are now
   donor_review (pairs possible). The dry runs match the records as built, not the pilot as it stands.
12. **The split checker on 11 synthetic v5 sets**: v4 provenances plus `xform.vaC_*` donor ids, written by my own
    writer (`verify_graft_reg_split.py sets`, `verify_graft_reg_split_pairs.py`). Each set was run through both my
    checker and `v5_split.py --baseline results/split_v4.json` (`split_summary.json`). The two checkers agree on every
    number:

    | set | v3 def. roster / max k16 / k17 | strict roster / max k16 / k17 | v5_split exit, nesting |
    |---|---|---|---|
    | strict plan (10 units) | 153/16, 183/16, 181/17 | 127/16, 174/16, 172/17 | 0, nested |
    | composite plan | same | same | 0, nested |
    | 25 distinct donors | same | same | 0, nested |
    | vmr0248 on 020 + 003 (holdout) | **139**/16, 183, 181 | **113**/16, 174, 172 | 3, spanning |
    | same donor patient, 2 ids (patient_key) on 003 + 020 | **139**/16 | **113**/16 | 3, merged + spanning |
    | shared 003 + 026 | **138**/16 | **112**/16 | 3 (claim 153 -> 138, 127 -> 112) |
    | shared 011 + 016 | 153/16, 183, 181 | 127/16, **170/16, 168/17** | 3 (claim 190 -> 186, 189 -> 185) |
    | shared 001 + 002 | 153/16, **181/16, 179/17** | unchanged | 3 (claim v3 max 199 -> 197) |
    | vmr0248 on 020 + 027 | 153/16, **181/16, 179/17** | unchanged | 3 |
    | **vmr0248 on 020 + 001** | unchanged | unchanged | **0**; stdout silent; JSON `one_to_one_nested: false` |
    | TaG-Net id on 005 | unchanged | unchanged | 2 |

    One-to-one plans leave every quantity unchanged, and every shared donor is listed in the JSON. **But the exit code
    and the console only react when a number drops.** A shared donor that happens not to lower a number (001 + 020)
    passes silently. The user's one-to-one rule is still broken there, and the "harmless" verdict depends on which
    optimum HiGHS returns. A driver must read `defs.*.extra_donors.one_to_one_nested`, not only the exit code.
    TaG-Net is flagged by the id string only. The registry builder refuses TaG-Net inputs, which is the real guard.

## Corrections to the reported claims

| claim | verdict | evidence |
|---|---|---|
| tool enforces every v4 gate + origin seam, plausibility, calibre | **mostly**: v4 gates, origin seam (both seams), course / tip / take-off / rise / window / rotations / handedness enforced on every passing side (0 violations in 2224; 34 VAs re-checked from written geometry). **Calibre is checked at the requested splice**: at the splice actually cut, 7 / 2224 dry passes exceed 1.6 (015 x2, 141 x5; max 1.727) and 117 "preferred" flags are wrong | tally.json, geom.json, calib_sweep.json |
| absurd placements are rejected | long / short / U-turn / loop / bent donors and the Stage-6 / 60 deg poses: all rejected. **Not rejected**: VA 30 mm off its subclavian (16 / 24 pass; stub unbounded), radius x3 (1 / 24) | absurd.json, stub.json |
| assignment one-to-one with the handedness rule | true for every reported plan; my Hungarian = theirs. **Not enforced at donor-patient level** (patient_key), and `build()` / `--plan` accept wrong-handed, QC-fail and shared specs | tally.json, dupkey.json, absurd.json |
| dry-run results | reproduce exactly (3345 / 2224 / 0.665, per unit, per gate, plans 89/102 etc., take-off 64/38%, compat 22/223, max 98.0) and are deterministic; inputs have since moved (pilot) | tally.json, compat.json, takeoff.json, stale.json |
| split checker reproduces v4 and catches a shared donor | reproduces every number independently (153/16/54; 127/16/80; 183/16, 181/17; 174/16, 172/17; div 181/180, 172/170, 168/167; v3 set 144/16/63, 180/16, 177/17). Catches shared donors in the JSON in every case; **exit code 0 when nothing drops** | split_v4.json, split_v3.json, split_div.json, split_summary.json |

## Files

- Code: `/sp/v5_va/verify_graft_reg_{tally,geom,calib_dbg,calib_sweep,absurd,stub,stale,compat,compat_keys,takeoff,
  dupkey,split,split_pairs,split_div,split_summary}.py`, `verify_graft_reg_split_run.sh`.
- Outputs: `/sp/v5_va/verify_graft_reg_out/` (`tally, geom, calib_sweep, absurd, stub, stale, compat, takeoff, dupkey,
  split_v4, split_v3, split_div, split_check_<set>, v5split_<set>, split_summary` .json, plus logs).
- Scratch data (deletable): `carotid_data/va_sources/v5_reg/verify_graft_reg/` (`written/` 22 anatomies,
  `donors_absurd/`, `compat/`, `reg_dupkey/`, `split_sets/` 13 sets of 223 provenance files).
