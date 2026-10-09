# v5 carotid anatomies: real cervical vertebral arteries

`carotid_data/anatomies_v5/` holds 223 anatomies, one for each v4 anatomy under the same name. v4 grafted real
patient geometry onto all six neck vessels, but below each vertebral artery's (VA's) TopBrain piece it still used the
shipped host's VA: one shared geometry in every anatomy, a plain tube with no wall relief. v5 replaces that lower VA
with a real donor VA segmented from thin-slice CT angiography: the donor's own lumen wall from its subclavian ostium
to a splice in V3, with v4's TopBrain piece above the splice. The carotids, ECAs, siphons and both routes are v4's,
and the collision mesh is baked by v4's baker with one added rule for the donor sections. 25 donor patients serve
the 25 TopBrain patients one to one, so the train/test split is unchanged.

    carotid_tools/v5/extract/                 donor VAs from thin-slice CTA: Arterial, traces, QC gate, CT-edge walls, tally
    carotid_tools/v5/graft/graft_v5_reg.py    place a donor VA under every v4 gate (strategy B)
    carotid_tools/v5/graft/graft_v5_assign.py one donor pair per TopBrain patient (exact lexicographic MILP)
    carotid_tools/v5/graft/graft_v5_reg_sep.py re-place a VA away from the right route's wall (the 13-anatomy move)
    carotid_tools/v5/bake/bake_v5b.py         bake_meshes_v4 unchanged + the donor sections + the handle rule
    carotid_tools/v5/checks/v5_split.py       donor-overlap train/test split checker
    carotid_tools/v5/MANIFEST.md              every copied file, the run that used it, how to re-run
    carotid_data/anatomies_v5/BUILD_v5.json   flags, sources, counts, fallbacks, final checks
    carotid_data/anatomies_v5/_records/       the results and notes this file cites

Every number below comes from a file in `carotid_data/anatomies_v5/_records/` (cited by its path there) or from
`BUILD_v5.json`. The final record is `results/v5_final_summary.json`, written after the 13-anatomy move.

## At a glance

| | v4 | v5 |
|---|---|---|
| anatomies | 223 | **223**, same names, none dropped |
| RCCA route (host → fork → siphon) | v3's, byte for byte | **unchanged**: RCCA file byte-identical to v3 in 223 / 223 |
| other non-VA centrelines | v4's | byte-identical to v4 in 223 / 223 |
| lower VA (subclavian → V3) | the host's tube, the same in every anatomy | **a real donor VA** in 429 of 446 VAs (RVA 211, LVA 218); v4's VA kept on 17 sides |
| VA above the splice | TopBrain piece | the same TopBrain piece, moved rigidly with the donor |
| donor patients | — | 25: Head-CTA 172-series 13, Song 2024 11, VMR 0248 1; one per TopBrain patient |
| VA wall > 0.3 mm off the tube, whole VA (median) | 0.018 | **0.128**; donor stretch 0.134 against v4's host tube 0.000 |
| collision mesh | one closed genus-0 two-manifold, no self-intersecting face | the same, 223 / 223 |
| collision mesh size | 23.5 k | 23,300–23,500 triangles (median 23,498) |
| right-route wall detail against v4 (exact ownership) | — | ≥ 0.99 of v4 in 222 / 223; fork and siphon median 1.000 |
| overlap-free train/test split | 45 quantities | all 45 unchanged |
| licence | — | 216 anatomies non-commercial (Arterial-derived donor); 7 VMR 0248 (research use) |

Sources: `results/v5_final_summary.json` (`counts`, `gates`, `right_route_vs_v4`, `va_wall_detail`, `loader`,
`split`, `licences`). The v4 whole-VA figure uses v4's 24-neighbour ownership; the v5 figures use exact ownership
with the seam zone counted as 0 (see *Wall detail*).

## Sources and licences

The dataset search came first (reports: *Public cervical vertebral artery datasets*, *Chinese cervical vertebral
artery datasets*). No public dataset labels the cervical VA (V1–V3) for more than a handful of subjects. VMR 0248 is
the one open origin-to-basilar pair; the SPL atlas is a coarser second; TopBrain's VA labels are cropped to the
braincase; TotalSegmentator and the other models have no VA class; no China-hosted platform holds labelled cervical
VAs, and the TaG-Net sample (origin-to-basilar VA labels for two patients, one with its CTA) has no data licence, so
it was used only as a calibration reference and never as a donor. The feasibility study (report *Cervical vertebral
artery grafting feasibility (v5)*) then showed that the graft mechanics work (strategy B), that 1.5 mm
TotalSegmentator CTAs give VA course but no gradeable wall, and that only thin-slice CT could give a real cervical
wall. The pilot (report *Thin-slice cervical VA pilot (v5)*) tested three open thin-slice sets; two carried the build
and the third was exhausted. All four reports are in `reports/`.

| source | what | licence | used |
|---|---|---|---|
| Head-CTA 172-series (Blagosklonova et al., Zenodo 15697196; doi 10.3390/data11040074) | head-and-neck CTA, Toshiba FC43, 0.5 mm slices | CC BY 4.0, attribution | 13 donor patients, 124 anatomies |
| Song et al. 2024 (CMHA aneurysm CTA, figshare 26965450; doi 10.1038/s41597-024-04056-8) | head-and-neck CTA, Philips 0.45 mm, GE 0.625 mm, Siemens 1.0 mm slices on a 0.7 mm grid | CC BY 4.0, attribution | 11 donor patients, 92 anatomies |
| VMR 0248_H_AOCERE_CAS (Wilson et al. 2013, doi 10.1115/1.4025983) | one CT patient, VA re-segmented from its CT | research and development use; keep LICENSE_VMR_0248.txt; acknowledge NLM R01LM013120 / NHLBI HHSN268201100035C | 1 donor patient, 7 anatomies |
| AortaSeg-60 (Zenodo 18147026) | aortic CTA | treated as CC BY 4.0 | none: 1 of 28 head-first images contains the head, and that case gave no pair |
| Arterial (FLOWCAT-CV/arterial @3ff58ab, weights Zenodo 22694951) | vessel mask and labelled centrelines on head-and-neck CTA | weights CC BY-NC 4.0, code PolyForm Noncommercial | every Song and 172-series donor |
| TotalSegmentator (Wasserthal et al. 2023, doi 10.1148/ryai.230024) | vertebra, subclavian and carotid masks for landmarks and QC | CC BY 4.0, attribution | every Song and 172-series donor |

Every donor that went through Arterial makes its anatomy non-commercial: 216 anatomies. The 7 VMR anatomies are not
NC, but carry the VMR terms. `LICENSE_v5_donors.txt` is in all 223 folders and names each anatomy's donor, licence,
attributions and the NC notice where it applies; the 7 VMR anatomies also carry `LICENSE_VMR_0248.txt`
(`results/v5_final_summary.json` → `licences`; `results/v5_assignment.json`).

## Donor pipeline

Each thin-slice case went through the same chain. Pilot cases ran it first (`notes/pilot_proc_ia172.md`,
`notes/pilot_proc_song.md`); the scale-up ran the pilot's chain with two fixes added ("pipeline v2",
`notes/SCALEUP_PIPELINE.md`).

1. **Fetch.** Single cases were range-read out of the archives (the 172-series DICOM series converted to NIfTI), and
   each was looked at by eye. 172-series cases first passed a pre-fetch arch check: Arterial needs the bottom slice to
   cut the aortic arch into two limbs.
2. **TotalSegmentator** total_v3 with `--ml` on the GPU, then split per class on the CPU (205–294 s per 172-series
   case against 695–1112 s before), plus a 1.5 mm copy.
3. **Arterial on the GPU** (image neve-arterial-gpu, an RTX A5000): nnU-Net vessel mask, then labelled centrelines.
   - The 172-series input is the CT clipped to [−1024, 3071] HU.
   - The Song input is re-windowed. The shipped Song files store every voxel ≤ 50 HU as 0, and Arterial's neck model,
     which z-scores the whole image, then loses the neck: on case 100 the CCA appears on 0.065 / 0.000 of C6–C2 slices.
     `ct_negzero_air1000` sets every voxel outside the body to −1000 and restores 0.72 / 0.62. This is the Arterial
     input only; QC and the wall read the shipped CT, and the clipping moves the lumen edge by a median 0.
   - A watchdog stops runs whose centreline stage hangs (2 of 12 pilot 172-series cases); those masks are traced
     without labels.
4. **Two traces per side** on Arterial's mask: tsba (the protocol trace: `art_extract.py` with a basilar-seed fix and
   a mask-plausibility down-weight) and ts (`art_extract.py` unchanged). Both are graded, and the better one per side
   is kept.
5. **QC gate** (next section) on both traces.
6. **CT-edge wall** (`ct_seg4`), seeded from the selected trace. Stations every 0.25 mm; 64 rays per station; a local
   half-maximum edge per ray; an upper threshold where a brighter neighbour (dense venous contrast, cortical bone)
   touches the lumen; a 0.8 mm opening near those contacts that keeps the lumen connected (pilot fix 8). The label
   becomes a stage-A surface (marching cubes and windowed sinc, TopBrain's surface protocol) with a VMTK centreline.
   The wall counts only if it passes the validity gates V1–V6 (centreline eccentricity ≤ 0.15 mm, centreline outside
   its wall ≤ 0.10, open sections ≤ 0.31, genus 0, the acquisition voxel (≤ 0.65 mm for native thresholds), VMTK
   radii) and was measured (`notes/vmr.md`, `notes/ref.md`).
7. **Measures**: vertebral landmarks (C6 … C1, dura), the wall relief per 10 mm bin, contact rays (bone, vein,
   artery), protrusion, figures.

### The QC gate: recalibrated, and what stays provisional

The gate is the donor-pool plan's Stage-2 QC with the values recalibrated on two TaG-Net ground-truth patients
(`notes/tag_calibration.md`, `notes/verify_tag_calibration.md`):
- rule 6 (Arterial label agreement) is a review flag, not a rejection;
- V3 loop: robust-cord excursion ≥ 12 mm, ≤ 6 mm from the posterior arch of C1, with C1 threaded;
- V1 mask breaks up to 3 mm are allowed;
- the V2-band IQR term is dropped;
- the 25 mm bridge total counts only below the splice;
- the Erec entry rule, and review lengths 260–290 mm.

Rule 6 as a review flag later held out of sample: all 14 rule-6 sides of the pilot were correct VA courses on the
pilot verifier's independent CT test (`notes/verify_pilot_qc.md`). The other changes are **provisional**: they rest
on the calibration sample only (V1 ≤ 3 mm, no IQR term, the 12 / 6 mm V3 values). A side that passes only through a
provisional limit is flagged and was checked by eye before use. 8 of the 27 eligible pairs carry such a side
(`results/scaleup_tally.json` → `counts`).

### The two pipeline fixes

- **V1 mask-break rescue** (`scaleup_v1rescue.py`). Where Arterial's mask breaks in V1 but the bridge runs through
  enhanced lumen (median HU ≥ 0.6 × CCA), the mask is grown inside a capsule along the break, the side is traced
  again, and the rescued trace replaces the first only if it passes and the first fails or passes only
  provisionally. Rescued sides inside eligible pairs: Song 109 R, 141 R and 138 R. The rescue does not check vessel
  identity: on Song 044 R the "bridge" lay inside the subclavian, and only the origin rule kept it out
  (`notes/verify_scaleup.md`).
- **C2-exit pinch fix** (`scaleup_ct_seg.py`). At the tight bend where the VA leaves C2 the station centre can fall
  out of the lumen and truncate the wall; flagged stations are moved back onto the lumen and cast again. Song 094 R
  went from 122.75 mm to 241.75 mm. Where the fix leaves a wall VMTK cannot trace, a second step re-casts that VA
  without the fix (`scaleup_ctseg_fallback.sh`). Lane walls with the fix triggered fail gate V4 more often than
  untriggered ones (12 / 31 against 8 / 37), so the fix only ever costs pairs; on all 14 pinch-triggered walls inside
  eligible pairs the wall centreline stays within 2.9 mm of the QC trace (`notes/verify_scaleup.md`).

The pilot cases a fix could act on were re-run under pipeline v2 (walls `ct_seg4v2`;
`results/scaleup_fixes_before_after.json`). The other pilot cases keep their pilot walls: no fix touched them.

### Yield

The pilot processed 23 cases (12 172-series, 10 Song, 1 AortaSeg). The scale-up then ran two lanes of 25 new cases
each, in the order of `results/pilot_next_cases.json`. Its stop rule (30 eligible pairs, at least 25 of them coarser
than the dense host mesh on both sides) was not met at the 25-case caps: 27 eligible, 3 short. The need of 25 pairs
was met with 2 spares, so the scale-up stopped there (`notes/V5_BUILD_LOG.md`, 2026-10-02 14:00).

| source | cases graded | eligible pairs | primary (used) | reserve | not eligible: a side fails QC | not eligible: QC passes, a wall fails V4 (genus) |
|---|---:|---:|---:|---:|---:|---:|
| Head-CTA 172-series | 37 | 14 | 13 | 1 (ia172_172) | 17 | 6 |
| Song 2024 | 35 | 12 | 11 | 1 (Song 138) | 15 | 8 |
| VMR 0248 | reference | 1 | 1 | 0 | | |
| **total** | **72** | **27** | **25** | **2** | **32** | **14** |

Source: `results/scaleup_tally.json` (`counts`, `eligible_pairs`, `not_eligible`), `results/v5_eligible_final.json`.
26 of the 27 eligible pairs have a donor-stretch wall above the dense host RVA mesh (0.027, all bins) on both sides;
19 stay above it with contact relief excluded.

AortaSeg-60 was exhausted at the pilot. 27 of its 28 head-first images start at the lower neck, so only Dissec_01
contained the head; its right side failed QC and its left wall failed validity gate V1.

## Eligibility, ranking and the 25 + 2 reserves

- **Eligible:** both sides of one donor patient pass the QC gate, and both have a valid, measured CT-edge wall. A
  single passing side cannot be used: every v4 anatomy uses both VA pieces of its TopBrain patient, so each patient
  needs a complete pair.
- **Ranking:** the mean over the two sides of the contact-free donor-stretch union > 0.3 mm (only 10 mm bins with fewer
  than 5% of edge rays meeting bone, a vein or another artery). It orders assignment priority; it does not decide
  eligibility.
- **By eye:** every flagged side (rule 6, provisional limit, rescue, bridge) was looked at before use; none was
  rejected (`results/scaleup_eye_rejections.json`; `notes/v5_eye172.md`).
- **Reserves:** ia172_172, whose left VA arises from the aortic arch and joins the subclavian only through a tissue
  link (it passes only through the provisional V1 ≤ 3 mm limit), and Song 138, which is not coarse (all-bins
  0.0154 / 0.0065) and whose right wall stops at C2 (`notes/verify_scaleup.md`, `notes/v5_eye172.md`). Reserves were to
  be used only for a TopBrain patient that no primary could serve. None was used.

All 25 primary pairs are above the dense host mesh on both sides over all bins; 18 are also above it with contact
relief excluded (`results/v5_eligible_final.json` → `counts`).

## Composition

### Strategy B

- The host subclavian is kept byte for byte: every replaced RVA keeps v4's 28-point subclavian prefix, and every
  replaced LVA starts at the first point of host branch (19) (`notes/verify_v5_assign.md`).
- The donor VA runs from its ostium to a splice in V3, between C2 and the dura. Its real lumen wall comes with it.
  The donor ostium joins the parent subclavian through a short synthetic junction stub (median 5.28 mm, max 10.49 mm;
  ungated).
- Above the splice, v4's TopBrain distal piece is re-fitted onto the donor and moved rigidly (Kabsch residual
  ≤ 0.034 mm, radii unchanged). A 25 mm calibre ramp sits on the donor side below the splice.
- Gates, all v4's plus new ones: 0.6 mm clearance to every other branch, to the parent after the exit and to itself
  (ramps included); every join ≤ 60°, the origin seam included; proper rotations; course RMS to the v4 VA corridor
  ≤ 20 mm; blind end moved ≤ 60 mm; take-off at 10 mm within 20–120°; rise ≥ 15°; the splice in V3; the calibre
  rule at the cut splice (ratios 1.25–1.6 pass as a flagged second pass).

Each donor starts from a weighted fit onto v4's VA corridor. The search then tries rotations about the origin,
splice moves and an ostium advance of up to 5 mm, nearest first, and keeps the first candidate that passes every
gate. Ostium advance: 0 mm on 421 of the 431 assigned VAs, 1 mm on 8, 2 mm on 2 (`results/v5_assignment.json` →
`counts`).

### One donor pair per TopBrain patient

Every eligible pair was dry-run against every anatomy: 27 units × 223 anatomies, 12,042 unit-anatomy-sides, of which
8,688 pass every gate (72.2%), with 0 errors (`results/v5_assign_dry_stats.json`). The assignment
(`graft_v5_assign.py` 1.1) is an exact lexicographic MILP, one unit per patient and one patient per unit. It
maximises, in this order (`BUILD_v5.json` → `assignment.objective_order`):

0. patients served by a primary pair;
1. anatomies with both VAs passing every gate;
2. VAs passing;
3. RVAs passing (the side training can reach);
4. the sum, over passing VAs, of the donor side's contact-free coarseness (coarser pairs to the patients with more
   anatomies);
5. unflagged before flagged;
6. take-off preserved.

Putting coarseness before the gate terms would have raised the coarseness sum from 33.28 to 39.37 but cost 24
anatomies on both sides and 24 VAs (`results/v5_assign_alt_orders.json`). A variant that put the reserve ia172_172 on
mr_015 gained one both-sided anatomy but used a reserve where a primary could serve; it was not taken
(`results/v5_assign_alt_all_served.json`).

| TopBrain patient | anatomies (native / mirrored) | donor | contact-free R / L | RVA | LVA | both / one side |
|---|---|---|---|---:|---:|---|
| topcow_mr_005 | 16 (8 / 8) | Song AHMU1218108 | 0.062 / 0.119 | 16 | 16 | 16 / 0 |
| topcow_mr_011 | 16 (8 / 8) | ia172_164 | 0.127 / 0.090 | 16 | 16 | 16 / 0 |
| topcow_mr_016 | 16 (8 / 8) | ia172_015 | 0.117 / 0.039 | 16 | 16 | 16 / 0 |
| topcow_mr_020 | 16 (8 / 8) | Song AHMU1218002 | 0.052 / 0.024 | 16 | 16 | 16 / 0 |
| topcow_mr_010 | 15 (7 / 8) | ia172_159 | 0.090 / 0.051 | 10 | 14 | 9 / 6 |
| topcow_mr_023 | 15 (7 / 8) | ia172_190 | 0.056 / 0.030 | 15 | 15 | 15 / 0 |
| topcow_mr_024 | 15 (7 / 8) | Song AHMU1218109 | 0.037 / 0.009 | 15 | 15 | 15 / 0 |
| topcow_mr_026 | 15 (7 / 8) | ia172_149 | 0.116 / 0.063 | 13 | 15 | 13 / 2 |
| topcow_mr_001 | 12 (8 / 4) | ia172_165 | 0.136 / 0.106 | 11 | 12 | 11 / 1 |
| topcow_mr_022 | 10 (7 / 3) | ia172_192 | 0.058 / 0.156 | 10 | 9 | 9 / 1 |
| topcow_mr_027 | 9 (6 / 3) | ia172_058 | 0.072 / 0.276 | 9 | 8 | 8 / 1 |
| topcow_mr_013 | 8 (0 / 8) | Song AHMU1218040 | 0.015 / 0.001 | 8 | 8 | 8 / 0 |
| topcow_mr_014 | 8 (0 / 8) | Song AHMU1218095 | 0.077 / 0.143 | 8 | 8 | 8 / 0 |
| topcow_mr_021 | 8 (1 / 7) | Song AHMU1218113 | 0.076 / 0.133 | 8 | 8 | 8 / 0 |
| topcow_mr_007 | 7 (3 / 4) | VMR 0248 | 0.035 / 0.076 | 7 | 6 | 6 / 1 |
| topcow_mr_008 | 6 (3 / 3) | Song AHMU1218100 | 0.129 / 0.076 | 6 | 6 | 6 / 0 |
| topcow_mr_002 | 5 (3 / 2) | ia172_136 | 0.095 / 0.079 | 5 | 4 | 4 / 1 |
| topcow_mr_012 | 5 (3 / 2) | Song AHMU1218008 | 0.010 / 0.080 | 5 | 5 | 5 / 0 |
| topcow_mr_003 | 4 (2 / 2) | Song AHMU1218094 | 0.061 / 0.083 | 4 | 4 | 4 / 0 |
| topcow_mr_004 | 4 (2 / 2) | ia172_071 | 0.063 / 0.200 | 2 | 4 | 2 / 2 |
| topcow_mr_017 | 4 (1 / 3) | Song AHMU1218141 | 0.002 / 0.019 | 4 | 4 | 4 / 0 |
| topcow_mr_018 | 4 (2 / 2) | ia172_141 | 0.032 / 0.012 | 4 | 4 | 4 / 0 |
| topcow_mr_006 | 2 (2 / 0) | Song AHMU1218032 | 0.021 / 0.045 | 2 | 2 | 2 / 0 |
| topcow_mr_025 | 2 (0 / 2) | ia172_069 | 0.142 / 0.123 | 0 | 2 | 0 / 2 |
| topcow_mr_015 | 1 (0 / 1) | ia172_039 | 0.078 / 0.106 | 1 | 1 | 1 / 0 |
| **total** | **223 (103 / 120)** | 25 | | **211** | **218** | **206 / 17** |

Coarseness from `results/v5_assignment.json` → `table`; VA counts from each anatomy's `provenance.json`
(`v5_va.summary.replaced`), after the bake; totals equal `results/v5_final_summary.json` → `counts.replaced`.
(`BUILD_v5.json` → `donors.table` still lists the assignment-time counts, 14 RVAs for ia172_149 and 7 LVAs for VMR
0248.)

### Handedness

v5 follows v4's handedness. In a native anatomy the donor's right VA becomes the RVA and its left the LVA. In a
mirrored anatomy (v3 siphon `*_L`) the donor's left VA, x-mirrored, becomes the RVA, and its right, mirrored, the LVA,
so the whole posterior circulation stays one mirror image. Every rotation is proper. The assignment verifier found the
correctly-handed donor fits the host's v4 corridor better than its mirror image for 419 of 431 VAs
(`notes/verify_v5_assign.md`).

### What the gates do not keep

- **Take-off.** The pose is registered to v4's VA corridor, so the donor's own take-off angle at 10 mm is not kept:
  251 of the 429 donor VAs differ from their native take-off by more than 15°, and the median difference over the 431
  assigned VAs was 17.8°. It is recorded per VA (`takeoff_not_preserved`), not gated.
- **Calibre.** 51 donor VAs pass the calibre rule only on its flagged second pass (ratio 1.25–1.6).
- **Flags.** 324 donor VAs carry a donor-side review flag (rule 6, provisional limit, rescue, bridge, Siemens slices).

Source: `results/v5_final_summary.json` → `flags`; `results/v5_assignment.json` → `counts`.

### VAs kept from v4: 17 sides, 17 one-sided anatomies

A side keeps v4's VA, its xform and its curve byte for byte when its assigned donor cannot be placed there; the side
is flagged with the reason in `provenance.json` (`v5_va.summary`, `v5_va.<side>`). 206 anatomies have both VAs
replaced and 17 have one (`BUILD_v5.json` → `v4_va_kept_sides`).

| anatomy | v4 VA kept | why |
|---|---|---|
| case_k_008_right__topcow_mr_010_L | RVA | no gate-passing pose (ia172_159 L: course RMS) |
| case_k_011_left__topcow_mr_026_L | RVA | ia172_149 L: calibre / splice window |
| case_m_006_left__topcow_mr_025_L | RVA | ia172_069 L: course RMS (fails every host) |
| case_w_003_left__topcow_mr_010 | LVA | ia172_159 L: take-off band |
| case_w_003_right__topcow_mr_001 | RVA | ia172_165 R: calibre / join / splice window |
| case_w_007_right__topcow_mr_010_L | RVA | ia172_159 L: mixed |
| case_w_007_right__topcow_mr_027 | LVA | ia172_058 L: course RMS |
| case_w_018_right__topcow_mr_022 | LVA | ia172_192 L: course RMS |
| case_w_024_right__topcow_mr_002 | LVA | ia172_136 L: course RMS |
| case_w_027_left__topcow_mr_004 | RVA | ia172_071 R: calibre |
| case_w_029_right__topcow_mr_004 | RVA | ia172_071 R: calibre |
| case_w_029_right__topcow_mr_010_L | RVA | ia172_159 L: mixed |
| case_w_037_right__topcow_mr_010_L | RVA | ia172_159 L: mixed |
| case_w_042_left__topcow_mr_010_L | RVA | ia172_159 L: course RMS |
| case_w_051_right__topcow_mr_025_L | RVA | ia172_069 L: course RMS |
| case_w_033_left__topcow_mr_007 | LVA | bake: the donor's tube-only union has a handle at its ostium junction (0.83 mm parent clearance) |
| case_w_050_left__topcow_mr_026_L | RVA | bake: the same, at the RVA ostium (0.70 mm parent clearance) |

The first 15 come from the assignment (`notes/v5_assign.md`, section 1), the last 2 from the bake (next section).
Weak donor sides explain most of them: ia172_069 L runs nearly horizontal for its first 30 mm and passes nowhere,
and ia172_159 L (native take-off 149°) passes only 12 of 223 candidates (`results/v5_assign_dry_stats.json`).

## Collision mesh

### The handle rule (`bake_v5b.py`)

`topbrain_tools/bake_meshes_v4.py` is used unchanged. The donor VA joins the owned union as two more real sections
(`vaC_R`, `vaC_L`), and many donor surfaces have handles of their own (stage-A genus up to 47 on Song 094 L).
v4's ladder re-chose every section's margin and, on one probe, fell to its whole-section rung, which skips the shrink
step and left 21 mm of the donor's wall as tube. The rule for v5 (user decision, `notes/V5_BUILD_LOG.md` 2026-10-02
16:02): tube only the short stretch around each handle and keep the rest of the donor's wall. `bake_v5b.py` adds
stated diffs D1–D8 to the v4 baker (`notes/v5_bake_probe.md`):

- **D1** the carotid sections are baked exactly as v4 baked them (margin, suppressed stretches, tubes, far clamp);
- **D2** the VA sections get their own margin (0.5, then 1.0 mm) and come after every carotid section in the union;
- **D3–D5** handle localisation on the VA sections only: up to 25 rounds of 10 mm bins (then 20, 40, 80 mm), bisection
  when a round finds nothing new, then prune, split and shrink each window to 2 mm, as v4 does;
- **D6** flagged fallbacks in order: local windows with the carotid sections included (`carotid_changed`), the whole
  donor as tube (`donor_tube`), v4's last rungs, and last the v4 VA kept;
- **D7** a cached union (bit-identical, about 13 s per attempt instead of 60–100 s);
- **D8** the report, `mesh_v5.json` (`ladder_v5b`).

### Result

| rung | anatomies |
|---|---:|
| 1: carotid pinned, VA whole, VA margin 0.5 mm | 193 |
| 1: carotid pinned, VA whole, VA margin 1.0 mm | 3 |
| 2: VA-local windows | 25 |
| C: local windows with the carotid sections included (flagged `carotid_changed`) | 2 |
| D / E / tubes only | 0 |

Source: `BUILD_v5.json` → `bake.rungs`.

- **Donor wall kept.** All 429 donor VAs keep their real wall; none became a whole tube. 404 needed no window; 25 have
  one window each, median 1.31 mm (1.15–9.14 mm), 0.55–6.2% of the donor stretch. The windows sit 46–125 mm from
  the ostium, 19 of them 83–125 mm, in the foraminal stretch (`BUILD_v5.json` → `handles`). The largest, 9.14 mm
  on case_w_008_right__topcow_mr_020 (Song 002 R), holds two handles that could not be split.
- **Detail kept in situ.** The donor stretch keeps a median 0.999 (0.81–1.27) of the relief it has alone
  (`BUILD_v5.json` → `va_detail_undecimated_union`).
- **Carotid windows.** The 2 rung-C anatomies, case_w_016_left__topcow_mr_011_L and case_w_016_right__topcow_mr_011_L,
  carry one v4-style siphon_L tube window each (1.6 and 1.3 mm). The donor's course changed which vessel owns some
  real carotid wall, and that wall closed a handle with a neighbour. A third, case_m_030_right__topcow_mr_013_L, had
  one on the RCCA lower stretch until the 13-anatomy move cleared it (`notes/v5_bake_full.md`, `notes/v5_route13.md`).
- **v4 VA kept by the bake (2 sides).** On case_w_050_left__topcow_mr_026_L (RVA) and case_w_033_left__topcow_mr_007
  (LVA) even the tube-only union has a handle where the donor leaves the parent subclavian (clearance 0.70 and 0.83 mm
  against the 0.6 mm gate): a centreline-level fusion that no tube window can remove. The anatomy was re-assembled
  with v4's VA on that side (`keep_v4_va.py`) and baked at rung 1.
- **OBJ budget (3 anatomies).** v4's `obj_ladder` found no clean 23.5 k OBJ for case_m_006_left__topcow_mr_007_L,
  case_w_018b_left__topcow_mr_007_L and case_w_048_left__topcow_mr_008 and stepped up to 35,250 triangles.
  `obj_fix.py` tried other targets and sources with v4's own decimation and acceptance; all three are clean within
  budget (two from the undecimated union at 23,500, one from the 60 k mesh at 23,300).
- **Margins.** Carotid margin 0.5 mm on 201 anatomies and 1.0 mm on 22, as in v4; VA margin 0.5 mm on 218, 1.0 mm on 5.

**Gates, all 223** (`results/v5_final_summary.json` → `gates`):

| gate | result |
|---|---|
| union, collision_full and OBJ each one closed genus-0 two-manifold | 223 / 223 |
| self-intersecting faces (union / full / OBJ), raw and welded | 0 / 0 / 0 |
| OBJ triangles | 23,300–23,500, median 23,498 (budget 23,500) |
| collision_full triangles | 59,998–60,000 |
| centreline points outside OBJ or collision_full | 0 of 421,265 |
| RCCA and LCCA navigable on both meshes | 223 / 223 |
| OBJ route lumen minimum | RCCA 0.701–2.094 mm (median 1.433), LCCA 0.690–1.822 mm (median 1.449) |
| collision_full / OBJ vertices to the union, p99 | ≤ 0.069 / ≤ 0.176 mm |

The strict audit is v4's own `topo_audit.py`, independent of the baker's checks; it passes 223 / 223 on every
sub-gate. The bake took a median 172.7 s per anatomy (121.6–496.5 s) and 11.9 CPU-hours in all, with peak RSS
1.80–2.16 GB, on 8 then 10 worker containers (`BUILD_v5.json` → `timings`).

## Wall detail

### The donor VA against v4's host tube

The measure is v4's `right_route_detail` applied per VA: the share of wall area standing more than 0.3 mm off the plain
tube (`|x − c| − r`), on the undecimated union. Two conventions matter:
- **Ownership is exact**: each wall cell goes to the branch with the largest `r − |x − c|` over all centreline
  samples. v4's code takes only the 24 nearest samples (k24), which can hand parent-subclavian or carotid-bulb wall
  to a VA; the feasibility study already named that artefact.
- **The seam zone** (the 25 mm calibre ramp on the donor side of the splice) is reported apart and counted as 0 in
  the whole-VA figure: there the donor's real wall stands off a ramped centreline radius, which is calibre, not relief.

| stretch (429 donor VAs) | > 0.3 mm, median | p10–p90 | min–max |
|---|---:|---|---|
| **donor stretch** (seam zone and tube windows excluded) | **0.134** | 0.056–0.254 | 0.029–0.336 |
| donor stretch, > 0.5 mm | 0.074 | 0.019–0.176 | 0.003–0.248 |
| donor stretch, > 0.15 r | 0.187 | 0.104–0.309 | 0.053–0.382 |
| donor stretch against a fitted tube | 0.108 | 0.046–0.247 | 0.023–0.315 |
| seam zone (reported apart) | 0.112 | 0.000–0.461 | 0–0.917 |
| TopBrain piece | 0.103 | 0.016–0.264 | 0–0.407 |
| junction stub | 0.157 | 0.000–0.647 | 0–0.744 |
| **whole VA**, ostium → end, seam = 0 | **0.128** | 0.054–0.231 | 0.028–0.267 |
| v4, host core on the same side | 0.000 | 0.000–0.001 | 0–0.020 |
| v4, whole VA on the same side (k24) | 0.018 | 0.003–0.097 | 0–0.155 |

Source: `results/v5_final_summary.json` → `va_wall_detail`. Every donor stretch beats v4's host core on the same side,
and every whole VA beats v4's whole VA: 429 / 429. The lowest donors, ia172_141 L (0.029–0.030 on four anatomies)
and ia172_190 L (0.031), still sit above the host core's maximum. These figures include relief at bone and vein
contacts. The user counts contact dents as anatomy (`notes/V5_BUILD_LOG.md`, 2026-10-02 03:50, item 3); a vein's
halo does not count, and the CT-edge wall's upper threshold reduces it without excluding it
(`notes/verify_pilot_wall.md`). The ranking used contact-free bins only.

For scale, the natural-anatomy standard asks for a set median ≥ 0.097 at > 0.3 mm on a VA measured alone
(`notes/ref.md`, W1). Measured that way on the pilot walls, the 172-series passed W1–W3 as a set and Song failed W1
and W3. The collision OBJ keeps a median 0.81 of the union's relief (p10 0.63, min 0.29); decimation smooths the
rest.

### The right route against v4

The right route must keep v4's wall. Every non-VA centreline is byte-identical to v4, and the carotid sections are
baked exactly as v4 baked them, but the donor VA's new course can change which vessel owns carotid wall near the
bulb or siphon. The final comparison rebuilds v4's union cell for cell and compares the RCCA-owned wall
(`results/v5_final_summary.json` → `right_route_vs_v4`):

| | exact ownership | k24 (v4's code) |
|---|---:|---:|
| identical to v4, cell for cell | 106 | 104 |
| identical beyond 4 / 6 / 8 mm of every VA centreline | 80 / 23 / 8 | 78 / 26 / 8 |
| not identical even beyond 8 mm | 6 | 7 |
| fork, v5 / v4 (> 0.3 mm): min, median, max | 0.956, 1.000, 2.387 | 0.991, 1.000, 2.387 |
| siphon, v5 / v4: min, median, max | 0.998, 1.000, 1.037 | 0.997, 1.000, 1.037 |
| anatomies ≥ 0.99 of v4 on both | 222 / 223 | 223 / 223 |
| RCCA median v5 / v4: fork, siphon | 0.1968 / 0.1961, 0.3218 / 0.3218 | 0.1968 / 0.1958, 0.3218 / 0.3218 |

Gains above 1 are real wall that v4's host RVA had refused (the case_w_033_left forks at 2.30–2.39, except mr_016 at 1.12). The v4 rebuild
reproduces `mesh_v4.json` and the v5 k24 reading reproduces `mesh_v5.json` on 223 / 223.

**The one anatomy below 0.99 under exact ownership** is case_w_018b_left__topcow_mr_002, fork 0.956. Its k24 reading
equals v4, which is why the k24-based selection below did not catch it. The loss is 121 v4 wall cells (7.85 mm²)
near the bifurcation, RCCA s 92–103 mm, that k24 gives to the RECA and exact ownership to the RCCA; in v5 that wall is
reshaped where the donor VA's centreline (ia172_136) passes 1.6–3.9 mm from it, while v4's host VAs ran 13.5–16 mm
away. The RCCA's own k24 route wall is identical to v4. It was not moved
(`swap13/diag_case_w_018b_left__topcow_mr_002.json`).

### The 13-anatomy move

After the full bake, 13 anatomies read below 0.99 of v4 on the right route under k24; the worst was
case_w_050_left__topcow_mr_004_L, fork 0.819. In each, a VA piece (the donor's lower VA, its V3, the calibre ramp or
the re-fitted TopBrain piece) ran within 0.66–4.45 mm of the RCCA tube, tube wall to tube wall, inside the reach of
the RCCA's real wall, so the owned union gave that wall up (`notes/v5_route13.md`, section 1). The user's rule
(`notes/V5_BUILD_LOG.md`, 2026-10-02 22:50): move the VA first, and change ownership only where moving cannot
restore ≥ 0.99.

- `graft_v5_reg_sep.py` is graft_v5_reg 1.1 with one change of objective: among the poses that pass every gate, take
  the one that refuses the fewest v4-kept RCCA wall voxels under the union's own ownership rule. Same donor, sides
  and assignment.
- All 13 were fixed by the move: 12 with the extended pose search, 1 (case_w_012_left__topcow_mr_021_L) with its right
  donor kept in place, its right TopBrain piece turned and its left VA moved. Route priority (`bake_v5c.py`) was built
  and run on 7, and never chosen.
- After the swap into `anatomies_v5` every fork and siphon ratio of the 13 is ≥ 0.9994 of v4 under exact ownership; 7
  are identical everywhere. Route lumen on the undecimated unions is unchanged (RCCA 0.959–1.886 mm). Every graft
  gate holds: clearance ≥ 0.71 mm, joins ≤ 59.3°, take-off ≤ 118.4° on every moved VA, course RMS 14.8–19.9 mm (gate 20), tip move
  ≤ 38 mm.
- What it cost: the moved donor stretches shift 2.8–13.9 mm RMS from their first placement and sit nearer the edge of
  the v4 corridor; their wall detail changes by −0.015 to +0.006; case_m_030_right__topcow_mr_013_L lost its carotid
  window (rung C → 1) and also its SOFA VA entry (below).

Sources: `results/v5_route13.json`, `results/v5_final_summary.json` → `route13`, `notes/v5_swap13.md`.

## Verification

Each stage was checked by an independent adversarial verifier with its own code; each item below says what it
changed. The final swap re-ran the check stage's own tools instead.

- **Final verifier** (`notes/verify_v5_final.md`), run after this record was first written: with its own code it
  reproduced that `anatomies_v5` is the verified 210 plus the 13 moved picks (by hash); every gate on the 13 moved
  anatomies, 10 random others and case_w_018b_left__topcow_mr_002; the split check; more than 120 numbers in this file
  against their sources; the images; and `anatomies_v4` and the tracked code untouched. One gap: the toolchain copy
  lacked graft_v5_reg_sep 1.0, which made 12 of the 13 moves. It was added (`carotid_tools/v5/MANIFEST.md`, "The
  route13 tool version") after a dry re-run on the pre-move inputs reproduced all 13 decision records
  (`checks/route13_tool_v1_0/cmp.txt`). Four wording slips it found here are corrected.

- **QC calibration** (`notes/verify_tag_calibration.md`): rule 6 → review needed an out-of-sample test (done in the
  pilot, holds); the V1 ≤ 3 mm, no-IQR and 12 / 6 mm V3 values were called unsupported, so they stay provisional and
  every side they admit is flagged and checked by eye.
- **Pilot QC and walls** (`notes/verify_pilot_qc.md`, `notes/verify_pilot_wall.md`): Song's best-of-two trace choice was
  undisclosed (12 pairs with it, 10 without); it was kept and disclosed. 140 R's course leaves the C1 loop. Contact
  relief is reduced by the bin rule, not excluded. The matched-noise floor is about 0.018, so pairs below about 0.03
  cannot be ordered. The Siemens Song series are 1.0 mm slices, not 0.7 mm.
- **Graft tool** (`notes/verify_graft_reg.md`): four enforcement gaps. The calibre gate was evaluated at the requested
  splice instead of the 1 mm sample cut (fixed in 1.1); one donor patient per TopBrain patient was not enforced across
  unit ids (enforced in 1.1); `build()` trusted the plan's handedness (checked in 1.1); `v5_split.py` was silent about
  a harmless shared donor (now prints the nesting and exits 4). The junction stub remains ungated.
- **Scale-up** (`notes/verify_scaleup.md`): the tally reproduces exactly. The Song lane had counted walls that fail
  gate V4 (its "9 pairs" at 11:15 were 4 under the build's rule). ia172_172 L has a tissue-level origin gap and
  Song 138 is not coarse: both became reserves. The rescue does not check vessel identity (Song 044 R).
- **Assignment** (`notes/verify_v5_assign.md`): holds; the split, the MILP optimum and every gate on all 431 written VAs
  reproduce. Qualifications: take-off is gated on the placed donor line but recorded on the written curve, so two
  VAs record 127.5° and 120.8°; 10 splices sit at the C2 edge; VMR's records understate its surface genus.
- **Mesh** (`notes/verify_v5_mesh.md`): holds; every mesh gate reproduces with its own topology, self-intersection and
  winding code. Notes: one zero-length edge in one union welds clean; one 0.02° sliver fold in one 60 k mesh, not in
  the OBJ.
- **Wall** (`notes/verify_v5_wall.md`): the substance holds, two numbers did not. The check stage's identity-class
  table was 4 anatomies too favourable, and its "whole VA 0.150" holds only with k24 ownership and the seam counted;
  the study's convention gives 0.128. It also found that 4 of the 13 route losses were k24 artefacts. This record
  uses the study's conventions (exact ownership, seam counted as 0), and the exact-ownership route comparison was
  then run on all 223, which found case_w_018b_left__topcow_mr_002.
- **Route move** (`notes/verify_v5_route13.md`): holds; every pick reproduces. Corrections: route priority's two drops
  came from widened RCCA tube windows, not attribution; 4 deficits were k24 artefacts, so under exact ownership those
  moves were harmless but not needed; one moved RVA (w_007) comes within 1.66 mm of the RCCA wall.
- **After the swap** (`notes/v5_swap13.md`): the 13 were re-checked with the check stage's own tools (strict audit,
  route, wall, lumen, loader, SOFA) and `BUILD_v5.json` was regenerated.

## Known limitations

- **SOFA VA entry reflects real geometry, not a defect.** SOFA passes on every anatomy tested: 36 / 36 v5, with 0
  exceptions and 0 non-finite steps (v4 controls 36 / 36; `results/v5_final_summary.json` → `sofa`). Under a
  closed-loop schedule that steers into the RVA, the wire went at least 20 mm past the ostium in 10 of the 25-anatomy
  set (8 of its 23 donor RVAs) against 24 / 25 in v4; before the move it was 11 / 25. Read this as diversity:
  - v4's VA entry is the host's in essentially every anatomy. Its RVA's first 60 mm lies within 0.05 mm of the host in
    213 / 223 (the other 10 are v4 repairs up to 3.3 mm); its LVA's first 35 mm deviates a median 0.046 mm, which is
    resampling (`checks/va_entry_geom.txt`).
  - v5's entries come from the donors: the RVA's first 60 mm deviates from the host a median 8.6 mm (up to 23.8 mm)
    and differs in 211 / 223; the LVA's first 35 mm a median 9.3 mm, differing in 219 / 223. The 23 donor RVAs of the
    SOFA set are 23 different real junctions.
  - So the v4 rate measures one host junction under a schedule tuned on it. The donor ostium sits 3–10 mm past the
    subclavian branch point (median 5), behind the junction stub, at 35–127° to the subclavian continuation, with
    0.95–1.69 mm of wall clearance in the 23 tested, against the 0.65 mm a wire needs; v4's host T-junction is 88° with
    2.2–2.3 mm (`notes/v5_checks.md`, `checks/entry_geom.json`, `checks/ostium_lumen.json`; measured before the move).
  - Training targets the RCCA, which is unchanged. The user decided against a junction fix
    (`notes/V5_BUILD_LOG.md`, 2026-10-03 14:05). `BUILD_v5.json`'s SOFA note still calls this question pending; it
    predates that decision.
- **Narrow VA points in the OBJ** (not a v4 gate). 3 VAs have points with less than 0.65 mm of wall clearance in the
  collision OBJ, where the donor radius sits at its 1.0 mm floor: case_w_025_right__topcow_mr_021_L LVA 0.578 mm and
  RVA 0.632 mm (Song 113), case_w_027_right__topcow_mr_022 RVA 0.646 mm (ia172_192). The 60 k mesh keeps ≥ 0.818 mm
  everywhere (`notes/verify_v5_mesh.md`, before the move, which fixed two more such points on
  case_m_022_left__topcow_mr_021_L). Median VA lumen minimum over all 446 VAs: 0.834 mm in v5, 0.978 mm in v4
  (`results/v5_final_summary.json` → `lumen.va_lumen_obj`).
  **Corrected by the orientation audit** (`reports/TopBrain piece orientation audit (v5).md`), which sampled the
  centreline polyline at 0.05–0.1 mm instead of the control points: 4 VAs fall below 0.65 mm, the fourth being
  case_w_047_left__topcow_mr_023 RVA (0.637, donor). The w_025 RVA has two such points: 0.547 mm on the donor stretch
  (the deepest) and the quoted 0.632 mm, which lies in the TopBrain piece (a pinch in the topcow_mr_021 left-VA label,
  the same point as in v4). None is a narrowing the CT supports: on the donor stretches the CT-edge wall stops short on
  the bone side, the 1.0 mm floor holds the union at about 0.93 mm and decimation takes the OBJ below 0.65. Every OBJ
  keeps a wire path with at least 0.71 mm of clearance. The artefact is systematic (377 of 429 donor VAs use floored
  stretches); its fix is a v6 item.
- **Smooth donors are near-tube on the OBJ.** 17 VAs from the two smoothest donor sides, ia172_190 L and ia172_141 L
  (union 0.029–0.037), read 0.010–0.019 on the collision OBJ (`notes/verify_v5_wall.md`, section 4; measured before
  the move).
- **Siemens 1.0 mm Song cases.** Song 094, 109 and 113 (27 anatomies: topcow_mr_003, mr_024, mr_021) are 1.0 mm slices
  on a 0.7 mm grid. They were accepted at native thresholds by convention and are flagged in provenance; any effect of
  the thicker slices on their relief is untested.
- **The 24-neighbour ownership artefact.** `right_route_detail` in `mesh_v5.json` uses v4's k24 ownership, like
  `mesh_v4.json`. k24 can hand carotid or subclavian wall to a VA, so 4 of the 13 route "losses" were artefacts, and
  case_w_018b_left__topcow_mr_002 is the reverse case (k24 equal to v4, exact 0.956). This record quotes exact
  ownership; per-anatomy readings in both conventions are in `per_anatomy/route_vs_v4_exact_final/`.
- **Carotid tube windows.** 2 anatomies carry a 1.3–1.6 mm siphon_L tube window that v4 did not have
  (case_w_016_left / right__topcow_mr_011_L, flagged `carotid_changed`); a third (case_m_030_right__topcow_mr_013_L)
  had one on the RCCA until the move.
- **The v4 split roster leaks** (pre-existing, not touched). The v3 launcher roster (153 train / 16 test) is
  overlap-free only under v3's donor definition. Counting v4's left forks, 26 of its train anatomies share a Zenodo
  donor with the test block, so the overlap-free roster is 127 / 16; the best overlap-free split v4 (and v5) allows is
  174 / 16 or 172 / 17 (`notes/split.md`, `results/split_v5_final.json`). v5's one-to-one VA donors change none of
  these numbers.
- **The junction stub is ungated** (max 10.49 mm, ia172_136 L) and **take-off is not preserved** on 251 VAs.
- **The left side is not reachable from the training insertion**, as in v4.

## Follow-ups

**Done in v6** (`CAROTID_V6.md`, `carotid_data/anatomies_v6`): every v6 item below, except the 17 VA sides with no
donor (left for later) and the donor colour in the figures (later).

- **v6: the donor's own subclavian junction** (user, `notes/V5_BUILD_LOG.md` 2026-10-03 14:02, first noted as
  v5.1, named v6 by the user later the same day). Graft a short
  stretch of the donor's own subclavian around the ostium, from about 10 mm after the subclavian origin (keeping the
  brachiocephalic split and the route start untouched) to about 20 mm past the VA ostium, so the VA leaves at its
  native take-off and the catheter meets the donor's real subclavian → VA junction. Cost: one more seam on the
  subclavian and a per-anatomy subclavian course. It addresses three v5 limitations: the synthetic junction stub
  (median 5.28 mm, max 10.49 mm, ungated), the take-off not preserved on 251 VAs (median 17.8° off native), and the
  SOFA wire reaching ≥ 20 mm into the VA in 10 of 25 anatomies against 24 of 25 in v4.
- **v6: case_w_018b_left__topcow_mr_002** — move its VA with the route13 tool (user: yes, 2026-10-03).
- **v6: regenerate the build record** with the two stale `BUILD_v5.json` fields corrected (`donors.table` counts are
  from assignment time; `checks_final.sofa.note` still says pending) (user: yes).
- **v6: TopBrain piece orientation.** The user saw TopBrain pieces that do not rise as they should (left ICA siphons in
  case_k_011_left mr_012_L / 013_L / 016_L / 023_L and case_w_003_left mr_007_L / 010; the TopBrain VA top in
  case_m_006_left mr_025_L / 027_L) — the mistake the frame-matching fix (tangent and superior, `TOPBRAIN_PIPELINE.md`,
  `CAROTID_THREE_SOURCE.md`) removed in the earlier sets. The audit (`reports/TopBrain piece orientation audit
  (v5).md`) found that the frame match pins only the roll, so tilt and heading follow an anchor tangent nothing checks,
  and the rise check was never a gate. 141 of 223 v5 anatomies have a visibly wrong piece (v4 171, v3 11); the siphons
  and ECA tips are inherited from v4, the VA tops come from v5's fit, which has no superior term. Its fixes go into v6.
- **v6: the narrow VAs and the donor wall's bone side** — see the corrected limitation above.
- Later: the donor VA colour in the figures (dark teal, near-black on the page).

## Records

- `provenance.json` keeps v4's record and adds:
  - `xform.vaC_R` / `vaC_L` for each replaced side: rotation, origin, anchor, mirror, donor surface path and md5,
    donor record, `lo_mm` / `hi_mm` (real wall from the ostium to the splice cut), `real_surface`;
  - `xform.va_R` / `va_L` updated for the re-fitted TopBrain piece;
  - `v5_va`: the gates, the full graft record per side (pose, splice, calibre ratio, origin turns, joins, course RMS,
    tip move, stub, ostium advance, take-off native and placed, flags, donor eligibility, licence, attributions), the
    assignment, a summary, and for the 13 moved anatomies `route13` (tool, targets, separation record);
  - `bake_v5`: rung, flags, margins, windows and tubed millimetres per VA, gates, VA detail.
- `mesh_v5.json`: v4's keys plus `ladder_v5b`, `v5_gates` and `va_detail`.
- `LICENSE_v5_donors.txt` in every folder; `LICENSE_VMR_0248.txt` in the 7 VMR anatomies.
- `BUILD_v5.json`: flags, sources, counts, fallbacks, handles, timings, route13, the 17 kept sides and `checks_final`
  (sha256 907b771759a31a6e8f5ea26b231329709a0d8ddc7cecafb34fdbfbc1acc2048a).
- `_records/`: `results/` (final summary, bake collections, checks, route13, assignment, eligible list, tallies,
  splits, pilot decision), `per_anatomy/` (route against v4 and VA wall per anatomy), `sofa/`, `checks/`, `swap13/`,
  and `notes/` (the build log and plan, every stage's notes and every verifier's notes). `_records/README.md` lists
  them.
- The undecimated unions are in `carotid_data/va_sources/v5_reg/iso_v5/`, the donor records and surfaces in
  `carotid_data/va_sources/v5_reg/donors_v5/`, and every derived donor case in `carotid_data/va_sources/derived/`
  (all gitignored).

## Reproduce

The toolchain ran from a session scratchpad and is copied to `carotid_tools/v5/`, one folder per stage
(`arterial/`, `extract/`, `graft/`, `bake/`, `checks/`). `carotid_tools/v5/MANIFEST.md` maps every file to its stage
and the run that used it and lists the hard-coded scratchpad paths a re-run must change.
`carotid_tools/v5/restore_layout.sh <dir>/v5_va` rebuilds the build-time layout byte for byte (the scripts call each
other as `/sp/v5_va/<path>`); mount `<dir>` at `/sp` and the repo at `/repo`.

Images: neve-arterial-gpu (Arterial; its Dockerfiles and pins are in `arterial/`), neve-totalseg-gpu
(TotalSegmentator), neve-build-stagea (QC, walls, measures), neve-build-vmtk (VMTK), eve-training-fixed (graft,
assignment, tally, loader, SOFA), neve-build-meshlab (bake and checks), as in CAROTID_V4.md for the last four.

The build order, in outline (the scripts are named as in the restored layout):

    # 1. donors (per case; Git Bash on the host)
    bash scaleup_song_chain.sh  <fetch|ts|art|trace|grade> <case ...>
    bash scaleup_ia172_chain.sh <fetch|ts|tscpu|gpu|post|grade> <case ...>
    bash scaleup_v2_post.sh <ia172|song> post2 <case ...>; bash scaleup_ctseg_fallback.sh <ia172|song> <case ...>
    python3 scaleup_tally.py; python3 v5_eligible_final.py                       # eve-training-fixed
    # 2. records, dry run, assignment, write
    python3 graft_v5_reg_donor.py --eligible /sp/v5_va/results/v5_eligible_final.json --out-root /repo/carotid_data/va_sources/v5_reg/donors_v5
    DONORS=donors_v5 DRY=dry_v5 bash graft_v5_reg_dryall.sh
    python3 graft_v5_assign.py --tag v5 --solver strict --score v5 --donor-root .../donors_v5 --dry .../dry_v5 --reserve-units ia172_172,song_AHMU1218138
    python3 graft_v5_reg.py --plan /repo/carotid_data/va_sources/v5_reg/plan_v5.json --donor-root .../donors_v5 --out /repo/carotid_data/anatomies_v5
    python3 v5_split.py /repo/carotid_data/anatomies_v5 --baseline results/split_v4.json
    # 3. bake and check (neve-build-meshlab)
    bash bake_full/launch.sh 10       # each worker: bake_full_one.py, then measure_full_one.py
    python3 bake_full/collect.py --build
    bash checks/launch.sh 8           # + checks/sofa_worker.sh in eve-training-fixed, then checks/collect.py
    # 4. the 13-anatomy move, swap and final record
    bash route13/graft13.sh <jobs>; VARIANT=moved_ext OWNERSHIP=standard bash route13/worker13.sh <list>
    bash swap13/swap.sh backup|swap|verify; bash swap13/run_a.sh 5; python3 swap13/final_collect.py

The exact arguments of each run are in its notes under `_records/notes/` and in `carotid_tools/v5/MANIFEST.md`.

## Loading

`DualDeviceNavTopBrain(anatomy_dir="carotid_data/anatomies_v5", ...)` loads v5 anatomies exactly as it loads v4 ones.
The training loader was run over all 223: 18 branches each, the host's insertion inside the mesh (5.87–6.00 mm from
the wall), every centreline point inside, RCCA targets beyond 133 mm on every anatomy (at least 74 each), and no
problem that v4's own audit does not also show (`results/v5_final_summary.json` → `loader`). Targets are drawn on the
RCCA as before.
