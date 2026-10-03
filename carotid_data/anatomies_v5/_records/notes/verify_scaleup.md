# verify_scaleup -- adversarial check of the v5 VA scale-up (fixes + Song lane + 172-series lane)

State: `/sp/v5_va/state/verify_scaleup.md`. Own code only (no GPU): `verify_scaleup_{tally,rescue,sides,origin,originfig,wall,ctseg_cmp,genus}.py`.
Figures / JSON lines (not record files) in the scratchpad's `verify_scaleup_work/` (outside `/sp/v5_va`).

## Answer so far (updated 2026-10-02 ~14:25)

**The tally holds (12:59 and the final 13:55); the Song lane's 11:15 report overstates its pairs; one eligible pair
(ia172_172) has a tissue-level origin gap; the stop rule was not met at the cap (27 of 30).**

1. **Tally reproduces.** `verify_scaleup_tally.py` re-derives every per-side input from the raw collect records and the files
   on disk (qc/select.json -> qc_<ext>.json; ct_seg/<tag>/measure.json gates, genus, area, donor stretch; edge.json contact
   rays), without the grader or the tally script. Result = `results/scaleup_tally.json` (12:59) exactly: **26 eligible**
   (ia172 14, Song 11, VMR 1), **25** with all-bins donor-stretch union > 0.027 on both sides, **18** with contact-free
   >= 0.027 on both; 0 record-vs-disk mismatches, 0 pair / value differences (tolerance 6e-4). Stop rule not met (short 4).
   ia172 lane snapshots (11:29 19/18/11, 12:28 23/22/15, 12:59 26/25/18), its 9 lane pairs and scores, the 50 records
   (25 donor_review / 12 donor / 13 fail) and the fixes agent's numbers (13 pairs, 8 contact-free-both, 141 / 015 scores)
   all reproduce.
2. **Final tally (13:55, both lanes at the 25-case cap) also reproduces exactly:** 27 eligible (ia172 14, Song 12, VMR 1),
   26 coarse on both sides (all-bins > 0.027), 19 contact-free >= 0.027 on both; 0 differences, 0 record-vs-disk mismatches.
   Song's last four: 108 = new pair; 128 R wall genus 1, 013 R fails QC (no R VA below C2 in the mask), 012 fails (R QC,
   L genus 5). **Stop rule NOT met: 3 pairs short of 30 at the cap** (the 25-pair need is met; the 25-coarse condition is
   met with 26).
3. **Song lane report overstates its pairs.** Its "9 Song pairs pass QC on both sides with walls" counts walls that FAIL gate
   V4: 030 R (genus 2), 111 L (3), 117 R (1), 120 R (1), 003 R (2). Under the build's rule only **4** Song lane pairs are
   eligible (095, 002, 040, 138), and 138 is not coarse (all-bins 0.0154 / 0.0065) and has a truncated R wall (C1 not
   reached). Its "002 unflagged" is also wrong: both 002 sides pass only via provisional limits (kept by eye 13:17 by
   instance 4).
4. **V1 rescue (all 9 rescued sides / 10 bridges re-checked).** Added-voxel counts match the records exactly; bridge HU
   reproduces within 15 HU; the patched mask joins the two ends inside the capsule and the original does not (9/10; Song
   007 L was already connected -- a trace excursion, harmless). The three rescued sides inside eligible pairs (Song 109 R,
   141 R, 138 R) are VA lumen continuations on straightened CPRs. **But the rescue does not check vessel identity:** Song
   044 R's "bridge" lies inside the subclavian (94 % of added voxels in the TS SCA label, lumen r_eq 5.8 mm); only gate
   rule 1 kept it out. 061 L (ineligible, genus 1) adds a 5 mm-wide blob, ambiguous.
5. **Per-side QC (10 newly passing + 4 failing sides).** Every newly passing side runs through the C6-C1 foramina, at
   0.93-1.09 x CCA, on the correct side of the cord, never inside TS CCA / BCV / SVC labels. Rule-6 sides 069 L / 149 L:
   Arterial's own VA label is off the foramina (Arterial wrong, trace right). Failing sides fail for real reasons (138 R
   V3 11 mm from the C1 posterior arch; 130 L flat V3; 060 R no enhanced origin link), with two conservative misses
   (060 R's 5b on the 1.5 mm copy vs 0.82 native; Song 137 R V1 ridge >= 288 HU = enhanced lumen within 1 mm).
6. **Origin bridges.** Enhanced (>= 0.45-0.8 x CCA) SCA->VA link within 6 mm of the trace for 071 L, 164 L, 136 L, Song 002
   L, 109 R, 138 R. **None for ia172_172 L** (eligible lane pair, rank 11): best link 114 HU = 0.30 x CCA within 3 and
   6 mm; trace 95-167 HU over s 0.5-1.5 mm. It passes only through the provisional V1 <= 3 mm limit (1.9 mm bridge). Either
   a stenosed ostium or an off-path origin -- a tissue-level gap admitted into an eligible pair.
7. **Walls (2 per lane).** Own exact-ownership measure on the protocol's SDF union reproduces measure.json for ia172 071 L,
   172 L and Song 095 R, 040 L to 4 decimals (donor stretch and all 10 mm bins); exact vs k24/k64 ownership identical;
   all-bins and contact-free aggregation = tally. 071 L's two highest-relief bins (0-20 mm, 0.34) border a dense
   (> 1600 HU) vein but stay under the 5 % contact-ray bar, so they count as contact-free.
8. **Pinch fix.** On the pilot walls it does what was claimed (untriggered walls identical or Dice >= 0.9996; Song 094 R
   122.75 -> 241.75 mm). On the lanes, walls with a triggered pinch fix fail V4 more often (12/31 = 39 %) than untriggered
   ones (8/37 = 22 %); lane wall invalidity overall 20/68 vs 1/30 in the pilot. Handle localisation (20 mm windows,
   `verify_scaleup_genus.py`): in 8 of 12 pinch-triggered invalid walls the handle window contains a pinch-fix station
   (ia172 065 R, 067 L, 155 R/L; Song 004 R, 020 L, 030 R, 065 R); but untriggered invalid walls (ia172 124 R, Song 117 R)
   carry handles at the same s (70-110 mm, the C2-C3 bends), so this is co-location, not proof the fix builds them. The
   fix only ever costs pairs (gate V4), never admits one: on all 14 pinch-triggered walls inside eligible pairs the wall
   centreline stays within 2.9 mm of the QC trace at every flagged station (HU 0.40-1.51 x CCA; the extremes are 172 L's
   ostial gap and 071 L's edge of the dense vein).
9. **Handles above C1 cost pairs:** Song 120 R and 004 L fail V4 only above C1 (origin -> C1 piece genus 0); 120 would be an
   eligible pair under a donor-stretch genus rule (the build's rule uses the whole VA).

10. **Song 108 (the last new pair):** both sides through the C6-C1 foramina, 0.95 / 0.98 x CCA, no gaps, enhanced origin
    links (0.7 / 0.8 x CCA within 6 mm); R's rule-6 flag is Arterial's label 20 mm away and off the foramina (trace right).

## Lost work (the user's question)
Nothing of this verification was lost: no verify_scaleup state existed, so it ran from scratch after the reset; every lane
result it checks is on disk and the Song lane finished its cap at 13:57 (state "complete").

## What would change the verdict
- If ia172_172 L is judged a tissue-bridged origin (no enhanced SCA->VA link within 6 mm), the eligible count drops to 26.
- If walls that do not reach C1 - 25 mm are excluded (Song 138 R), it drops by one more; 138 is not coarse anyway.

## Details
(see state file Findings; per-item JSON / PNG in the scratchpad's verify_scaleup_work/)
