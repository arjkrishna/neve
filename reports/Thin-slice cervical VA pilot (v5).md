# Thin-slice cervical VA pilot (v5)

Date: 2026-10-02. Scope: v5 replaces the shared host lower vertebral artery (VA) in the 223 v4 carotid anatomies with
real donor VAs. Each donor runs from its subclavian origin to a splice in V3, and v4's TopBrain piece carries on above
the splice (strategy B). This pilot ran 23 open thin-slice CTAs through the donor pipeline to answer three questions:
- Can these sources supply the roughly 25 donor pairs the build needs, one pair per TopBrain patient?
- Are their lower-VA walls coarse and irregular like real anatomy, and coarse enough against the host?
- Which pairs are the coarsest once relief from vein, carotid and bone contacts is excluded?

The sources are Song et al. 2024, the Head-CTA 172-series set and AortaSeg-60. TaG-Net is a reference only and is never
a donor.

Every key number below was measured by a pilot agent and then re-derived by one of two independent verifiers with
their own code: one for QC, counts, need and the next-case list, one for walls and ranking. The compute cost and the
pool sizes were not re-derived. Where a verifier corrected or disputed a result, this report gives the corrected value
or both values. [What the verification changed](#what-the-verification-changed) lists every change.

Grading conventions throughout:
- native thresholds only: W1 0.0966, W2 0.0204, W3 0.1416, with W3 always read together with W1;
- the acquisition voxel;
- exact ownership (k ≥ 64);
- the 25 mm seam zone reported separately;
- W7 computed by hand.

QC verdicts are given under the recalibrated gate (primary) and under the study draft gate, plus the two-patient-backed
variant where it matters.

Working notes, results and scripts are in the session scratchpad under `v5_va\` (paths at the end). Derived data is in
`carotid_data\va_sources\`, which is gitignored. Apart from this report, nothing in the repo was changed.

Session note: session limits cut the pilot off twice (about 18:20 and 23:32–23:38 EDT on 1 October), and each agent
resumed from its checkpoint state file without redoing or losing finished work.

## Answer

1. **Go for the scale-up, under the recalibrated QC gate. The donor supply is partial: 12 eligible pairs against 25
   needed.** The verdict depends on which gate is used.

   | gate | 172-series sides | Song sides | complete pairs | verdict (≥ 60 % of sides) |
   |---|---|---|---|---|
   | recalibrated (primary) | 15 / 24 (62.5 %) | 15 / 20 (75 %) | 11 | **go** on both sources |
   | two-patient-backed changes only | 14 / 24 (58 %); 13 / 24 (54 %) without 140 R † | 13 / 20 (65 %); 11 / 20 (55 %) on the protocol trace alone ‡ | 9 | partial: Song go as run, 172-series no |
   | study draft + additions | 7 / 24 (29 %) | 5 / 20 (25 %) | 1 | **no-go** |

   † The verifier found that 140 R's course leaves the C1 loop on CT.
   ‡ See point 2.

   - The 172-series go has no margin. Dropping either side that passes only through a provisional limit (001 R or
     140 L) gives 58 %.
   - AortaSeg-60 is exhausted. Only 1 of its 28 head-first images contains the head, and that case gave no pair.
2. **Donor pairs: 12 eligible now, against 25 needed (30 with 5 spares).**
   - The 12 are 5 from the 172-series (015, 039, 058, 141, 190), 6 from Song (008, 032, 094, 100, 109, 113) and VMR 0248.
   - Every v4 anatomy uses both VA pieces of its TopBrain patient, so the need is 25 complete pairs, one per patient.
     A single passing side cannot be used.
   - Two Song pairs, 008 and 100, rely on keeping the better of two traces per side. On the protocol trace alone the
     count is 10, not 12. The other trace was checked on CT and is a correct VA, so the count stands as run, but the
     choice must be kept at scale.
   - 8 of the 12 pairs are coarse enough on both sides once contact relief is excluded (≥ 0.027, the dense host mesh).
3. **The lower-VA walls are coarse and irregular, far above the host. But most of the relief sits at bone and vein
   contacts, and it is spread unevenly.**
   - **Whole VA.** Wall more than 0.3 mm off the tube has a median of 0.143 (172-series 0.160, Song 0.086). The host
     tube reads 0.0007 and the dense host RVA mesh 0.027; 28 of 29 walls beat the mesh.
   - **References.** VMR 0248 reads 0.045 / 0.138, TaG-Net A 0.045 / 0.111, and TopBrain 0.121 (gate-valid,
     use-weighted 0.102). The 172-series passes W1–W3 as a set. Song fails W1 and W3.
   - **With contact relief excluded** (the user's rule), the donor stretch has a median of 0.072 (172-series 0.092, Song
     0.062, range 0.009–0.318). That is VMR's left side (0.076) and TaG-Net A's left side (0.061), and below TopBrain.
   - **The relief is real image structure.** Featureless-tube phantoms read 0.0008–0.018 against 0.21–0.24 real. The
     floor at matched noise is about 0.018, so scores below about 0.03 cannot be ordered.
   - **Evenness is mixed.** Few 10 mm bins are plain tube (0.17; 172-series 0.10, about TopBrain's 0.13–0.14), against
     0.23–0.76 for VMR and TaG-Net. But along the VA the relief sits in V1–V2 and the C2 loop, with V4 smooth. Between
     the two sides of a pair it can differ ×4 (058: 0.072 / 0.276), and between cases ×35.
4. **Ranking** (mean over R and L of the contact-free donor stretch).
   - 172-series: 058 (0.174), 039, 015, 190, 141 (0.022).
   - Song: 113 (0.104), 100, 094, 008, 032, 109 (0.022).
   - It reproduces exactly under its stated rule. The 172-series top is robust. The Song order depends on how contacts
     are excluded: per triangle instead of per 10 mm bin, 008 moves from fourth to first.
5. **What failed.**
   - **Sides.** 15 of 46 sides fail the recalibrated gate:
     - 7 at V1 or origin breaks, of which 3–4 are Arterial mask breaks through enhanced lumen and can be rescued;
     - 6 whose course leaves the VA above C2;
     - 2 on the wrong vessel.
   - **Pipeline.**
     - Song's 0-HU-clipped CT breaks Arterial's neck model. A re-windowed copy (air1000) fixes it.
     - ct_seg4 truncates walls at a tight C2-exit bend (2 of 15 Song walls).
     - The plain 0.8 mm opening loses most of the V1 on 172-series walls, so fix 8 is required.
     - Arterial's centreline stage hung on 2 of 12 cases.
6. **Scale-up.**
   - An ordered list of 197 new cases is ready.
   - Reaching 30 pairs needs about 36 more cases (central estimate), 44 for 80 % confidence and 50 for 90 %. That is
     about 12, 14.5 and 16.5 h on two lanes.
   - Two cheap fixes come first: the V1 mask-break rescue (about +2 pairs from cases already on disk) and the ct_seg4
     C2-exit fix.
   - If only pairs coarse enough on both sides after contact exclusion count, about 70–106 more cases are needed.
7. **Decisions for you** are listed in [Decisions only you can make](#decisions-only-you-can-make). The main ones:
   - the gate;
   - what counts as "coarse enough";
   - the contamination rule that orders Song;
   - the size of batch 1;
   - the Siemens Song cases, which are 1.0 mm slices.

## Terms used

- **Recalibrated gate** (primary). The donor-pool plan's Stage-2 QC with the two-patient calibration's values:
  - rule 6 (Arterial label agreement) becomes a review flag;
  - V3 loop: robust-cord excursion ≥ 12 mm and ≤ 6 mm from the posterior arch of C1, with C1 threaded;
  - V1 mask breaks ≤ 3 mm are allowed;
  - the V2-band IQR term is dropped;
  - the 25 mm bridge total counts only below the splice;
  - the Erec entry rule;
  - review lengths 260–290 mm.
- **Study draft gate.** The plan's draft plus the feasibility report's additions: V3 loop ≥ 15 mm and ≤ 5 mm, and C1
  threading.
- **Backed-only.** The draft with only the four changes that two patients support:
  - rule 6 to review;
  - a V3 floor below 15 mm (12 mm used);
  - the C7 over-call fix;
  - calibre from the CT half-max.
- **Donor stretch.** Origin → C1f − 25 mm, where C1f is the exit from the C1 transverse foramen. This is the piece
  strategy B grafts. The 25 mm seam zone below the splice is reported separately.
- **Union > 0.3 mm.** The share of the VA's wall area that stands more than 0.3 mm off the plain tube. It is measured
  on the ct_seg4 stage-A wall with the undecimated owned union.
- **Contact-free.** Only the 10 mm bins in which fewer than 5 % of the half-max edge rays meet a brighter neighbour (bone
  of the foramina, an enhanced vein, another artery) or a dip before one.
- **Plain-tube bin.** A 10 mm bin with union < 0.01.
- **Pair, eligible.** Both sides of one donor patient pass QC, and both have a valid ct_seg4 wall.
- **tsba / ts.**
  - tsba is the protocol trace: art_extract with the basilar-seed fix and a mask-plausibility down-weight.
  - ts is the unchanged art_extract.

## Donor need: 25 pairs, one per TopBrain patient

- **The patients.** The 223 v4 anatomies use 25 distinct TopBrain patients: 103 anatomies native and 120 mirrored,
  which matches `BUILD_v4.json`.
- **Both pieces are used.** Every anatomy uses both VA pieces of its patient.
  - In a native anatomy, the donor's R goes to the RVA and its L to the LVA.
  - In a mirrored anatomy, the donor's L goes x-mirrored to the RVA and its R to the LVA (the follow rule).
  - So each patient needs a complete pair. That includes mr_006 (used only native) and mr_013, 014, 015 and 025 (used
    only mirrored).
- **Need: 25 pairs one-to-one, 30 with 5 spares.** The spares cover graft-gate losses: the registry dry run passes
  66.5 % of unit-anatomy-sides.
- **Anatomies per patient** run from 1 to 16: 16 × 4 patients (mr_005, 011, 016, 020), 15 × 4 (mr_010, 023, 024, 026),
  then 12, 10, 9, 8 × 3, 7, 6, 5 × 2, 4 × 4, 2 × 2 and 1.
- **Assignment.** Give the coarsest pairs to the patients with the most anatomies. The training-reachable RVA takes the
  donor's R in native anatomies and its L in mirrored ones. The 8 patients with 15–16 anatomies are split 7–8 native
  against 8 mirrored, so for them each donor side reaches the RVA in about half the anatomies.

The verifier recounted all of this from the 223 `provenance.json` files and `BUILD_v4.json`, and it reproduced exactly.

## Per source

| | 172-series (Zenodo 15697196) | Song 2024 | AortaSeg-60 | total |
|---|---|---|---|---|
| licence | CC BY 4.0 | CC BY 4.0 | treated as CC BY 4.0 | all Arterial-derived output is non-commercial |
| scanners, slices | Toshiba Aquilion / PRIME, FC43; 0.5 mm slices, 0.39–0.62 mm in-plane (grids at 0.25–0.4 mm z are overlapping reconstructions; graded at 0.5 mm) | Philips 0.45 mm, GE 0.625 mm, Siemens 0.7 mm grid (the Siemens series are 1.0 mm slices; see Walls) | 0.78 × 0.78 × 0.625 mm | |
| cases fetched / processed | 13 / 12. 166 was rejected before processing (arch cut); 2 alternates (190, 015) were added for arch-apex cases | 10 / 10; none rejected | only Dissec_01 of 28 head-first images contains the head; the other 27 start at the lower neck | 23 processed (25 were planned) |
| input handling | Arterial input clipped to [−1024, 3071]; a pre-fetch arch check is needed (1 of 13 fetched failed) | **air1000 re-window needed**: Arterial loses the neck vessels on the shipped 0-HU-clipped CT | the release drops the −1024 HU offset and stores 32 volumes head-end first under a +z header; fixed per case | |
| Arterial ran | 12; the centreline stage hung on 061 and 090, so their masks were traced without labels | 10, on air1000 | 1; it labelled every class "L*" | 23 |
| sides traced | 24 | 20 | 2 | 46 |
| sides passing, recal / draft / backed | 15 / 7 / 14 (62.5 / 29 / 58 %) | 15 / 5 / 13 (75 / 25 / 65 %) | 1 / 0 / 1 | 31 / 12 / 28 (67 / 26 / 61 %) |
| go (≥ 60 %), recal / draft / backed | go / no / no | go / no / go as run (55 %, no, on the protocol trace alone) | n = 1, not informative | |
| complete pairs, recal / draft / backed | 5 (015 039 058 141 190) / 0 / 5 | 6 (008 032 094 100 109 113) / 1 (032) / 4 (008 032 094 113) | 0 | 11 / 1 / 9 |
| valid walls | 14 of 15 passing sides (140 L has a genus-1 handle) | 15 of 15 (094 R and 106 L truncated at the C2 bend) | 0 (L fails gate V1: offset 0.189 mm) | 29 |
| whole-VA union > 0.3 mm, median | 0.160 | 0.086 | 0.387 raw, not gradeable | 0.143 |
| contact-free donor stretch, median | 0.092 | 0.062 | – | 0.072 |
| edge FWHM | 1.12–1.41 mm | 0.95–1.35 mm | 1.69 mm | |
| cost per case, serial, fetch included | about 40 min | about 27 min | about 30 min | |

**Song's re-window.**
- **What the shipped files are.** Every voxel ≤ 50 HU is stored as 0.
- **Why Arterial fails on them.** Arterial's neck model z-scores the whole image, and on these files 59–75 % of the
  voxels (the air outside the body) read 0 instead of −1000. On the shipped CT of case 100, the CCA appears on
  0.065 / 0.000 of C6–C2 slices (R / L); on 109, 0.28 / 0.33.
- **What fixes it.** `ct_bg-100` and `negzero` fail the same way. `ct_negzero_air1000` sets every voxel outside the body
  to −1000 and restores 0.72 / 0.62.
- **It is the Arterial input only.** QC HU and the wall read the shipped CT.
- **The clipping does not move the wall.** The median edge shift is 0, the union changes by +0.002, and FWHM by
  +0.03–0.06 mm. The verifier confirmed this on 113 R / L and 100 R.

**AortaSeg-60.**
- **Only one usable case.** The earlier survey's "28 cases from the vertex" was a misreading of neck sections, so the
  open pool loses about 27 subjects. The plan's ≈ 40 sides / 16 pairs from AortaSeg does not exist.
- **Dissec_01.**
  - L passes QC (recalibrated and backed-only) as a course donor. Its wall fails validity gate V1, and its lumpy relief
    (raw 0.387, FWHM 1.69 mm, late mixed phase) is not a validated wall.
  - R fails: a 10 mm origin bridge, and the trace climbs the right ICA because Arterial's labels did not block the
    carotids.

## QC: where the gates differ, and why sides fail

**Sides the recalibrated and draft gates disagree on** (8 on the 172-series, 10 on Song):
- **Rule 6 alone** (Arterial label agreement):
  - 172-series 015 R, 058 L, 118 L, 141 R, 190 R;
  - Song 008 L, 044 L, 094 L, 100 L, 113 R, 113 L.
- **Rule 6 plus the study V3 floor of 15 mm:** 172 039 L (13.1 mm); Song 008 R (12.8 mm), 106 L (14.3 mm).
- **Provisional, in-sample limits only:**
  - 172 001 R: V1 bridge ≤ 3 mm, plus rule 6;
  - 172 140 L: the IQR term dropped;
  - Song 100 R: the IQR term dropped, plus rule 6;
  - Song 109 R: V1 bridge ≤ 3 mm.

  These 4 of the 31 passing sides rest on limits the calibration verifier called unsupported. Song pairs 100 and 109
  depend on them.
- **Backed-only also admits 172 140 R** (study cord 12.9 / 3.8), which the recalibrated gate rejects. The verifier's own
  cord reads 12.4 / 13.4: the trace leaves the C1 loop through an 8.2 mm bridge at 48 HU above C1.

**Rule 6 → review now holds out of sample.**
- All 14 rule-6 sides are correct VA courses on the verifier's independent CT test: origin on the own-side subclavian,
  foramina threaded, a cord-relative V3 loop on the C1 arch, lumen HU, no carotid contact, correct side.
- That is the out-of-sample check `verify_tag_calibration` asked for. The other provisional limits (V1 ≤ 3 mm, no IQR
  term, the 12 / 6 V3 values) were neither confirmed nor refuted: they admit only the sides listed above.

**Song's trace choice.**
- `qc/select.json` grades both traces, the protocol trace (tsba) and the unchanged tool's (ts), and keeps the better one
  per side. ts is kept on 008 R / L and 100 R / L.
- On tsba, 008 L (V3 0.7 / 21.0, C1 not threaded) and 100 L (a 21.8 mm bridge low in V3) fail.
- On 008 R and 100 R both traces pass, but only ts reaches the whole VA.
- The ts traces are correct VAs on CT.
- On the 172-series, ts equals tsba below the splice on every failing side, so no such rescue exists there.

**Why the 15 sides fail** (recalibrated gate; confirmed on CT by the verifier):

| cause | sides | note |
|---|---|---|
| V1 mask break through enhanced lumen (rescuable) | 172 061 L (5.6 mm at 310 HU), Song 141 R (4.6 mm at 587 HU), Song 044 R (9.1 mm at 480 HU; also origin 3.25 mm) | Arterial's mask breaks, the lumen is there. The verifier's course test passes all three |
| V1 bridge, borderline | 172 041 R (4.0 mm at 177 HU = 0.47 × CCA) | reported as "not lumen"; the verifier calls it borderline and a possible third side rescue |
| V1 bridge, not lumen | 172 041 L (8.0 + 8.3 mm, 102–128 HU), 073 L (3.9 mm through fat at −25 HU), Song 106 R (5.1 + 8.3 mm at 11 / 92 HU) | real breaks or shortcuts |
| course leaves the VA above C2 or skips the V3 loop | 172 073 R, 090 L, 118 R, 140 R; Song 090 R / L (head rotated at C1–C2) | Song 090 R also fails the origin (4.5 mm) and a 31 mm non-lumen V1 bridge |
| wrong vessel from the start | 172 001 L (Arterial's mask lost the left VA over > 40 mm); AortaSeg R (every Arterial class "L*", trace up the right ICA) | |

No wrong-vessel, venous or wrong-side trace is counted as a donor. There was no cherry-picking: every processed case is
in the denominator.

## Walls: coarse and irregular, mostly at contacts

Per passing side: ct_seg4 stage-A wall seeded from the selected trace, exact ownership k 64, acquisition voxel, native
bars. The per-side table is in `notes\pilot_decide.md` and `results\pilot_decide_tables.json`. The wall verifier
re-implemented the v4 union measure from its definition. It reproduced every walled side's whole-VA, C6 → C1, donor and
contact-free values to within 0.007 (Spearman 1.0), and the FWHM to within 0.036 mm.

| set | whole VA > 0.3 mm, median (range) | C6 → C1 > 0.3 mm | contact-free donor stretch, median (range) | plain-tube bins, whole VA | edge FWHM (mm) | noise (HU) |
|---|---|---|---|---|---|---|
| 172-series, 14 walls | **0.160** (0.026–0.262) | 0.217 | **0.092** (0.012–0.318) | 0.10 | 1.12–1.41 | 18–31 |
| Song, 15 walls | **0.086** (0.045–0.258) | 0.127 | **0.062** (0.009–0.134) | 0.25 | 0.95–1.35 | 13–24 |
| all pilot, 29 walls | 0.143 | 0.194 | 0.072 | 0.17 | 0.95–1.41 | |
| VMR 0248, R / L (same harness) | 0.045 / 0.138 | 0.076 / 0.129 | 0.035 / 0.076 | 0.55 / 0.35 | 1.39 / 1.44 | 28 |
| TaG-Net A, R / L (reference only) | 0.045 / 0.111 | 0.071 / 0.152 | 0.038 / 0.061 | 0.76 / 0.23 | 1.30 / 1.29 | 15 |
| TopBrain VA pieces (MRA) | 0.121 median; 0.102 gate-valid, use-weighted | – | no contacts in MRA | 0.13–0.14 | – | – |
| host | tube 0.0007; dense RVA mesh 0.027 | | | | | |
| native bar | W1 0.0966 | W1 0.0966 | | | | |

**Against the host.**
- 28 of 29 whole VAs beat the dense host RVA mesh (0.027). The exception is 172 190 L at 0.026.
- The set median is ×5 the dense mesh and ×200 the host tube.
- On the donor stretch counted over all bins, every side of every eligible pair beats 0.027 (lowest 0.038). 094 R has no
  donor stretch, because its wall is truncated.

**Against the standard (whole VA, set medians at native bars).**
- 172-series: 0.160 / 0.076 / 0.227 passes W1, W2 and W3.
- Song: 0.086 / 0.053 / 0.130 fails W1 and W3 and passes W2.
- C6 → C1 passes W1 and W3 on both sources.
- W7 fails on 30 of 30 walls: V4 and the upper V3 are smooth in CT. This does not matter for strategy B, which replaces
  V4 with the TopBrain piece.

**Compared with what the feasibility report expected.** It predicted thin-slice walls "at best VMR-like". Over the
whole VA, the 172-series is clearly coarser than VMR (0.160 against 0.045 / 0.138), and Song is about VMR's level. With
contact relief excluded, both are VMR-like.

**The relief is real image structure, not noise.**
- Featureless-tube phantoms of 172 061 R and 140 L were made with the real edge blur and noise and run through the same
  method. They read 0.0008 and 0.005, against 0.244 and 0.213 real.
- Phantom plain-tube bins are 0.62–0.95, against 0.09–0.10 real.
- The verifier found the ×1 phantom under-noised at the scale that matters (lumen voxel std 8.5 HU, against 13.8 real).
  The phantom at twice the noise (14.9 HU) reads 0.018, so **the matched-noise floor is about 0.018**.
- The coarse sides sit ×13 above it. The lowest contact-free scores are at the floor: 172 141 L 0.012, Song 109 L
  0.009, 008 R 0.010, 141 L 0.019, 032 R 0.022.
- Song had no phantom run. Its noise and FWHM are no worse than the 172-series', so its floor should be no higher. That
  is inferred, not measured.

**Most of the relief sits at contacts.**
- Per 10 mm bin (the pilot's attribution), 69 % (172-series) and 76 % (Song) of the relief area lies in contact bins.
  Union is 0.237 / 0.217 there, against 0.080 / 0.046 elsewhere.
- The contacts are bone in the transverse foramina and enhanced veins.
- Two Song walls are dominated by them:
  - 094 L's V1 runs against a ≥ 1500 HU structure, most likely a contrast-filled vertebral vein. It was checked by eye.
  - 008 R's V1 runs beside a paired vessel of similar brightness.
- The verifier's per-triangle detector puts the bright-adjacent share of the whole wall's relief at 54 % / 55 %.
- **Contamination is reduced, not excluded.**
  - Inside the pilot's "contact-free" bins, a median 23 % (172-series) and 16 % (Song) of the remaining relief is still
    next to a structure brighter than half-max, and a further 18 % / 10 % next to dim structure.
  - The bin rule also drops nearly every V2 bin, so the contact-free score is mostly V1.
  - It is conservative too: real foramen wall is excluded along with the bone contact.

**How evenly the relief is spread.**
- **Across bins: more evenly than VMR or TaG-Net.** The plain-tube bin fraction is 0.17 overall and 0.10 on the
  172-series, which is TopBrain's level (0.13–0.14). VMR reads 0.55 / 0.35 and TaG-Net A 0.76 / 0.23.
- **Along the VA: unevenly.** The relief sits in V1–V2 and at the C2 loop. C1 → dura and V4 are smooth on every wall.
- **Between the two sides of a pair: unevenly.** 058 reads 0.072 / 0.276 contact-free, and 015 0.131 / 0.036.
- **Between cases: widely.** Contact-free donor values run from 0.009 to 0.318 (×35). The 172-series has a coarse tail:
  090 R 0.318, 058 L 0.276, 118 L 0.249, 061 R 0.244, 140 L 0.209. Song tops out at 0.134. 5 of 30 donor sides fall
  below 0.027 contact-free.

**The acquisition voxel.**
- The 172-series is graded at its 0.5 mm slice thickness, not its finer overlapping grid.
- Song recorded the grid as the acquisition voxel. Its Siemens series (094, 109, 113: "CarotidAngio 1.0 Bv40",
  "DE_CarotidAngio 1.0 Qr40") are 1.0 mm slices on a 0.7 mm grid. The measured z blur confirms it: the z / in-plane
  FWHM ratio is 1.09–1.25, against 0.92–1.08 on Philips and GE.
- Union values do not change. But native bars apply to these 3 of 10 Song cases only by convention: even the recorded
  0.7 mm is above `va_standard`'s 0.65 mm native limit. They include Song's #1 and #3 pairs, and any resolution effect
  on their coarseness is untested.

## Donor ranking

**Score.** The mean over R and L of the contact-free donor-stretch union > 0.3 mm (`results\pilot_donor_ranking.json`).

**Per-triangle column.** The verifier's alternative removes bright-adjacent triangles everywhere instead of dropping
whole 10 mm bins. Its detector is the verifier's own heuristic and is not validated: it shows how sensitive the ranking
is, not which rule is right.

| source | rank | pair | contact-free donor R / L | score (min side) | per-triangle score (verifier) | donor, all bins, R / L | QC and flags |
|---|---|---|---|---|---|---|---|
| 172-series | 1 | **058** | 0.072 / 0.276 | **0.174** (0.072) | 0.165 | 0.102 / 0.330 | R donor; L review (rule 6) |
| | 2 | 039 | 0.078 / 0.106 | 0.092 (0.078) | 0.102 | 0.092 / 0.172 | both review: rule 6 on L, L V3 13.1 mm (fails the study floor), bridges above the splice |
| | 3 | 015 | 0.131 / 0.036 | 0.084 (0.036) | 0.113 | 0.212 / 0.213 | R review: rule 6, Erec entry C7 to check by eye; G2 < 2.5 mm on R |
| | 4 | 190 | 0.056 / 0.030 | 0.043 (0.030) | 0.048 | 0.072 / 0.038 | both review: rule 6 on R, a bridge above the splice |
| | 5 | 141 | 0.032 / 0.012 | 0.022 (0.012) | 0.040 | 0.149 / 0.038 | R review (rule 6); L at the noise floor, below the dense host mesh |
| Song | 1 | **113** | 0.076 / 0.133 | **0.104** (0.076) | 0.140 | 0.194 / 0.232 | both review (rule 6); Siemens 1.0 mm slices |
| | 2 | 100 | 0.129 / 0.071 | 0.100 (0.072) | 0.097 | 0.115 / 0.073 | ts trace on both sides; rule 6 on both; R also passes only without the IQR term |
| | 3 | 094 | 0.062 (partial) / 0.083 | 0.073 (0.062) | not computable | – / 0.396 | R wall truncated at the C2 exit; L rule 6, and its V1 runs against a contrast-filled vein (look before use); Siemens 1.0 mm |
| | 4 | 008 | 0.010 / 0.079 | 0.044 (0.010) | **0.187** | 0.244 / 0.288 | ts trace on both sides; rule 6 on both (R also V3 12.8 mm); contact-free on only 6 / 14 % of the stretch |
| | 5 | 032 | 0.021 / 0.050 | 0.036 (0.021) | 0.036 | 0.046 / 0.076 | the only pair that passes the draft gate |
| | 6 | 109 | 0.035 / 0.009 | 0.022 (0.009) | 0.045 | 0.065 / 0.103 | R passes only with the V1 bridge ≤ 3 mm; Siemens 1.0 mm |
| reference | – | VMR 0248 (eligible) | 0.035 / 0.076 | 0.055 (0.035) | – | 0.060 / 0.226 | its left V1 (vein contact) was baked as tube in the prototype |
| reference | – | TaG-Net A (not eligible) | 0.038 / 0.061 | 0.049 | – | 0.057 / 0.135 | no data licence; never a donor |

**Robustness.**
- **What reproduces.** Under the stated rule the ranking and the scores reproduce exactly from the per-case files.
- **172-series.** Under the per-triangle rule the order becomes 058, 015, 039, 190, 141 (Spearman 0.9). 058 is first
  under every rule tried.
- **Song.** The order becomes 008, 113, 100, 109, 032 (Spearman 0.6).
  - 008 rises from 0.044 to 0.187, because the bin rule keeps only 6 % / 14 % of its stretch.
  - The pilot's figures say the excluded relief on 008 R is bone and a merged neighbour vessel. So the question is
    whether that relief should count.
- **The bottom pairs.** 141, 109 and, by the bin rule, 008 and 032 each have a side at the noise floor (0.009–0.022),
  so their order among themselves is not meaningful.
- **The cited rank agreements are weak support.**
  - Agreement with the plain-tube fraction (Spearman 0.98 / 0.94) is computed on the same bins as the score, so it is
    near-circular.
  - The 172-series' contact-free C6 → C1 agrees with the score at only 0.3.
- **Kernel.** Every 172-series case is Toshiba FC43, so the within-source order is not a kernel effect. The gap between
  the two sources cannot separate kernel from anatomy.

**Merged build order.** A round-robin over the two sources, with VMR inserted by its score: 058, S113, S100, 039, 015,
S094, VMR, S008, 190, S032, 141, S109.
- With 12 eligible pairs against a need of 25, all 12 are used now. The order sets assignment priority, and later
  decides which pairs are displaced once the pool passes 30.
- **Coarse enough on both sides** (contact-free ≥ 0.027): 058, S113, S100, 039, 015, S094, VMR and 190, so 8 pairs.
  Under the per-triangle rule, S008 and S109 join. S094 cannot be scored that way, and 141 and S032 stay out.

**Passing single sides that cannot pair yet:**
- 172-series: 090 R 0.318, 118 L 0.249, 061 R 0.244, 140 L 0.209 (genus-1 handle), 001 R 0.153.
- Song: 106 L 0.134 (partial wall), 044 L 0.040, 141 L 0.019.
- AortaSeg: Dissec_01 L, course only.

Two of these become pairs if the V1 mask-break rescue works (next section): 061 (061 R plus a rescued 061 L) and S141
(S141 L plus a rescued S141 R). 041 R might be rescued as a side, but 041 L's bridges are not lumen, so 041 would still
not pair.

## What failed, and the fixes

| problem | where | fix | status and gain |
|---|---|---|---|
| Arterial mask breaks in V1 through enhanced lumen | 172 061 L, Song 141 R, Song 044 R; possibly 172 041 R | re-join the mask across a break whose bridge HU is lumen-like | **untested.** 061 and S141 become pairs (+2; 061 R is the fourth-coarsest single side). 041 R might pass as a side, but 041 L fails on non-lumen bridges. S044 R also fails the origin |
| ct_seg4 truncates the wall at a tight C2-exit bend (the station centre leaves the lumen) | Song 094 R (loses 118 mm above s 122.75), 106 L (78 mm above s 164) | re-centre stations whose centre HU is below the local threshold | **untested.** The C2 loop is the tightest bend on most VAs, so this will recur at scale |
| trace deviates only above the splice | 172 140 R | accept origin → C1f donors when the trace is correct up to the splice | untested; 140 would also need its L handle repaired |
| carotids not blocked when Arterial's labels are wrong | AortaSeg R; 172 001 L | block the carotids from the TotalSegmentator masks, not from Arterial's labels | untested |
| contamination excluded per 10 mm bin | all walls | exclude per ray or per triangle | untested; it changes the Song order |
| the 0-HU-clipped Song CT breaks Arterial's neck model | all Song cases | `ct_negzero_air1000` as the Arterial input | **applied** to 10 / 10 |
| ct_seg4's plain 0.8 mm opening keeps only 77–129 mm of a 200–250 mm VA, or drops V1 | 172 015 R / L, 039 L, 140 L, 141 L, 190 L, 058 R | fix 8, a connectivity-preserving opening | **applied** (with fix 7 for NaN VMTK seeds) |
| Arterial's centreline stage hangs | 172 061, 090 | a watchdog; trace without labels | **applied**. Rule 6 cannot be applied there |
| TotalSegmentator's per-class writer is single-threaded | all | `--ml` plus a CPU split | **applied**: 695–1112 s → 205–294 s per 172-series case |
| TotalSegmentator vertebra masks fill the transverse foramina | all | count bone contact from the edge rays only | applied |
| arch cut at the bottom of the scan | 172 166 (rejected); 090 and 118 cut only the apex | the pre-fetch arch check | applied |
| AortaSeg HU offset and slice direction | Dissec_01 | a per-case fix, recorded in `prep.json` | applied |

**Keep for the scale-up:**
- the air1000 re-window and the [−1024, 3071] clip;
- TotalSegmentator `--ml` with the CPU split;
- the Arterial watchdog;
- the arch pre-check;
- fixes 7 and 8;
- **both traces, with the per-side selection.** The yield estimates assume it.

## Scale-up: the next cases and their cost

**Pairs per processed case** (95 % Wilson intervals):

| gate | 172-series | Song | both |
|---|---|---|---|
| recalibrated | 5 / 12 = 0.42 (0.19–0.68) | 6 / 10 = 0.60 (0.31–0.83) | **11 / 22 = 0.50 (0.31–0.69)** |
| backed-only | 5 / 12 | 4 / 10 | 9 / 22 = 0.41 (0.23–0.61) |
| draft | 0 / 12 | 1 / 10 | 1 / 22 |
| coarse enough on both sides | 4 / 12 | 3 / 10 | 7 / 22 = 0.32 (0.16–0.53) |

**More cases needed to reach 30 eligible pairs.** Cases alternate 1:1 between Song and the 172-series; the estimates use
a Jeffreys beta-binomial predictive per source.

| scenario | eligible now | still needed | central | 80 % | 90 % |
|---|---|---|---|---|---|
| **recalibrated (primary)** | 12 | 18 | **36** (18 + 18) | **44** | **50** |
| recalibrated + V1 rescue (untested) | 14 | 16 | 27 | 32 | 36 |
| backed-only | 10 | 20 | 49 | 62 | 72 |
| coarse-enough pairs only | 8 | 22 | 70 | 90 | 106 |
| draft | 2 | 28 | not reachable within the pools | | |

These counts cover only the QC and the wall. They do not include graft-gate losses, which the 5 spares are meant to
absorb.

**Pools left** (distinct patients, z ≥ 300 mm):
- **Song: 122.** Siemens 0.7 mm 4, GE 0.625 mm 84, Philips 0.45 mm 34.
- **172-series: 75.** Removed before that:
  - 52 that are not Head+Neck or have z < 300 mm;
  - 19 with slices thicker than 0.75 mm;
  - 6 posterior-circulation aneurysms;
  - 7 likely duplicates.

  The duplicate screen is a heuristic. Some same-sex, same-age, same-model groups remain (125 / 127, 146 / 180,
  064 / 066, 156 / 152), so check them when fetched.

**Selection list** (`results\pilot_next_cases.json → selection`; 197 cases, ordered, each with its reason). The verifier
confirmed all 197 are distinct and that none overlaps the pilot.
- Song and the 172-series alternate, Song first.
- **Song order:** Siemens first (3 of 3 pilot cases gave pairs), then GE and Philips interleaved 5 : 2.
- **172-series order:**
  1. the 8 named reserves;
  2. 0.5 mm FC43 controls;
  3. 0.5 mm anterior-aneurysm cases;
  4. FC04;
  5. 0.6–0.75 mm series.
- **Age < 75 first** in every tier: the pilot gave pairs in 1 of 5 cases aged 75 or over, against 10 of 17 younger ones.
  That is weak evidence.
- Every 172-series case needs the pre-fetch arch check.
- **The first ten:** S111, 172_062, S138, 172_119, S096, 172_060, S120, 172_067, S020, 172_160.

**Batch 1: 36 or 44.** The decision text and the build log say 36 (orders 1–36, the central estimate, about an even
chance of reaching 30). `pilot_next_cases.json → batch_plan.first_batch` says 44 (the 80 % point, P = 0.81). Either
way, re-rank after the batch and stop once eligible pairs reach 30.

**Cost on this machine** (TotalSegmentator `--ml`; from the pilot's timing logs):

| | Song (10 cases) | 172-series (3 `--ml` cases) |
|---|---|---|
| GPU, median (max) | 356 s (720): TotalSegmentator 72–187 s, Arterial 167–620 s | 577 s (1521): TotalSegmentator 205–294 s, Arterial 265–1293 s |
| CPU, median (max) | 1192 s (1985) | 1643 s (1880) |
| fetch | 20–71 s range-read, 60–109 MB | 31–56 s + 33–100 s to convert, 0.3–0.6 GB |
| serial per case, fetch included | about 27 min | about 40 min |

CPU time covers the split, the 1.5 mm copy, traces, VMTK, QC, ct_seg4, landmarks, measure, edge and figures.

**Throughput.** Two lanes (one per source, each running one heavy CPU container) share the GPU lock:

| cases | wall time | GPU busy |
|---|---|---|
| 36 | about 12 h | 4.7 h |
| 44 | about 14.5 h | |
| 50 | about 16.5 h | |

The 172-series lane is the bottleneck. A single lane takes about 20–28 h. Derived data is 0.4–0.8 GB per case, and D:
has 505 GB free.

**Before the bulk run**, do the two cheap fixes: the V1 mask-break rescue (+2 pairs from disk) and the ct_seg4 C2-exit
fix. With the rescue, the central need drops to 27 cases.

**The working rule in `V5_BUILD_LOG.md`.** Stop when eligible pairs reach 30 and at least 25 pairs beat the dense host
mesh on both sides (donor stretch, all bins), with a cap of 50 new cases. All 12 current pairs already beat 0.027 on
all bins, so that rule reduces to "30 eligible pairs". The user's contamination-excluded bar is much stricter: 8 of 12
qualify now, and 70–106 more cases would be needed.

## What the verification changed

| item | as first reported (pilot_decide) | after verification | effect |
|---|---|---|---|
| Song pairs; backed-only go | 6 pairs; backed-only 13 / 20 (65 %, go) | Song's counts rest on a per-side best-of-two choice between tsba and ts (008 R / L, 100 R / L). The decision did not disclose it. On the protocol trace alone: recalibrated 13 / 20 (65 %) and 4 pairs; backed-only 11 / 20 (55 %, no-go) and 3 pairs. The ts traces are correct VAs on CT | the counts stand as run; eligible pairs are 12, or 10 without the choice. The scale-up must keep both traces |
| 172-series backed-only | 14 / 24 | includes 140 R, whose course leaves the C1 loop on CT. Without it: 13 / 24 (54 %) | "no" either way |
| 041 R | V1 bridge "not lumen" | 177 HU = 0.47 × CCA, and it passes the verifier's course test: borderline | a possible third side rescue (041 L still fails, so no pair) |
| Song 090 R | "course leaves the VA above C2" | it also fails the origin (4.5 mm) and a 31 mm non-lumen V1 bridge | none |
| batch 1 | 36 cases | the json plan says 44 | decision below |
| contamination | "relief from contamination excluded" | reduced, not excluded: 23 % / 16 % of the relief in contact-free bins is still bright-adjacent, plus 18 % / 10 % dim-adjacent | the scores are upper bounds on clean relief |
| ranking robustness | the ranking agrees with plain-tube at ρ 0.98 / 0.94 | near-circular (same bins). Under a per-triangle exclusion: 172-series ρ 0.9 (058 stays first), Song ρ 0.6 (008 first). C6 → C1 agreement on the 172-series is 0.3 | the 172-series top is robust; the Song order depends on the method |
| noise floor | phantoms 0.0008 / 0.005 | the ×1 phantom is under-noised locally; the matched-noise floor is about 0.018 | pairs with a side below about 0.03 cannot be ordered |
| Song acquisition voxel | grid = acquisition voxel | the Siemens 094, 109 and 113 are 1.0 mm slices on a 0.7 mm grid (series names and z blur) | native bars apply to them by convention only; any resolution effect is untested |
| relief at contacts | 69 % / 76 % (per bin) | 54 % / 55 % of relief area bright-adjacent (per triangle) | conclusion unchanged |
| minor wall values | 140 L donor 0.320 (from bins) | exact stretch 0.272; 001 R and 140 L C6 → C1 come from ts bins; Song 106 L contact-free 0.141 against 0.134 | none of these sides is in a ranked pair |

**What reproduced exactly:**
- the donor need and per-patient counts;
- the per-side QC verdicts, re-implemented from the raw QC values: 44 of 44 thin-slice sides under the recalibrated
  and draft gates, and the backed-only counts;
- the side and pair counts (31 / 12 / 28 sides, 11 / 1 / 9 pairs);
- the gates-differ lists;
- the ranking arithmetic, the merged order and the Wilson intervals;
- the next-case list.

**What independent checks confirmed:**
- **The independent CT course test.** Every one of the 31 passing sides is a VA course on CT. 11 of the 15 failing sides
  fail it for the stated reasons. The other 4 are exactly the V1 mask-break sides.
- **The wall numbers.** Every wall number was re-implemented independently and agrees to within 0.007, along with the
  FWHM, the plain-tube fraction and the phantom values.
- **Frames and sides.** On all 11 thin-slice pairs, the RVA lies on the subject's right.
- **Song's re-window and clipping.** They do not change the wall.

## Decisions only you can make

1. **The QC gate for the build.**
   - Recommended: the recalibrated gate, with rule 6 as a review flag, now backed out of sample.
   - The other provisional limits (V1 ≤ 3 mm, no IQR term, the V3 12 / 6 values) still rest on the calibration's own
     sample. They admit 4 of the 31 passing sides (172 001 R, 140 L; Song 100 R, 109 R).
   - Refusing them costs Song pairs 100 and 109 and raises the central need from 36 to about 49 cases.
   - Middle course: keep those sides, flagged for review by eye.
2. **Song's best-of-two trace choice.** Keep it (recommended). It adds 2 pairs (008, 100), and both chosen traces are
   correct VAs on CT. Without it, Song's backed-only rate falls below 60 %.
3. **What "coarse enough" means.**
   - (a) Any QC-passing pair with valid CT walls. All 12 beat the dense host mesh over all bins; this needs about 36
     more cases.
   - (b) Both sides ≥ 0.027 after contact exclusion. 8 pairs qualify now; this needs about 70–106 more cases.
   - (c) A higher bar.

   Linked to this is the feasibility report's open decision 1: does relief at bone and vein contacts count as nooks?
   Most of the relief these walls have sits there.
4. **The contamination rule that orders the pairs.** Per 10 mm bin (the pilot's), or per ray / per triangle (the
   verifier's alternative). This decides whether Song 008 ranks first or fourth, and whether 008 and 109 count as coarse
   enough. The per-triangle rule needs validating before it is adopted.
5. **Batch 1 size.** 36 cases (about an even chance of reaching 30 pairs, about 12 h) or 44 (80 %, about 14.5 h).
   Either way, re-rank after the batch.
6. **The Siemens Song cases** (1.0 mm slices). Accept them at native bars by convention (they hold Song's #1 and #3
   pairs), test the resolution effect first, or demote them in the selection list. The list puts the 4 remaining
   Siemens cases first.
7. **The gap while the pool is short.** For TopBrain patients without a donor pair:
   - keep v4's VAs (the registry tool flags these anatomies);
   - fill with tube-class TotalSegmentator pairs (course only);
   - or wait for the scale-up.
8. **Licence.**
   - Every donor that went through Arterial makes its v5 anatomies non-commercial.
   - Song 2024, the 172-series and AortaSeg-60 need CC BY attribution.
   - TaG-Net stays internal.

## Open risks

- **The yield rests on 22 cases.** The Wilson interval on pairs per case is 0.31–0.69. The age and scanner orderings are
  weak evidence. The plan asked for 25 pilot cases; AortaSeg could supply only 1, so 23 were run.
- **In-sample thresholds still decide 4 sides**, and the 172-series go has no margin.
- **The contact rule is coarse.**
  - Dim venous plexus merged at half-max stays in.
  - Several scores rest on less than 20 % of the stretch: Song 008 R / L and 094 L.
  - Real foramen wall is excluded together with the bone.
- **Individual donors need a look before baking.**
  - Song 094: its R wall is truncated, and its L V1 runs against a vein.
  - 172 015 R: the Erec entry at C7.
  - 172 039 and 190: bridges above the splice.
  - 172 140 L: a genus-1 handle.
  - G2 < 2.5 mm on 172 015 R, 061 R and 140 L.
- **The ct_seg4 C2-exit pinch** will recur at scale until it is fixed.
- **Graft compatibility filters further.**
  - The registry dry run passes 66.5 % of unit-anatomy-sides.
  - With today's 10 test units, a strict one-to-one assignment replaces 191 of 446 VAs.
  - The case counts above do not include these losses.
- **Duplicate patients** across the 172-series are screened only heuristically.
- **The V4 is smooth in CT on every wall** (W7 fails 30 of 30). This is harmless under strategy B, but rules out
  whole-VA donors from these sources.

## Files

Scratchpad root: `C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\`.

- **Notes** (`v5_va\notes\`):
  - fetch: `pilot_fetch_song.md`, `pilot_fetch_ia172.md`, `pilot_fetch_aortaseg.md`;
  - processing: `pilot_proc_ia172.md`, `pilot_proc_song.md`, `pilot_proc_aortaseg.md`;
  - decision: `pilot_decide.md`;
  - verifiers: `verify_pilot_qc.md`, `verify_pilot_wall.md`;
  - calibration background: `tag_calibration.md`, `verify_tag_calibration.md`.
- **Results** (`v5_va\results\`):
  - `pilot_decide.json`, `pilot_decide_tables.json`, `pilot_decide_need_rows.json`;
  - `pilot_donor_ranking.json`, `pilot_next_cases.json`;
  - `pilot_proc_{ia172,song}{,_grade,_summary}.json`, `pilot_proc_aortaseg.json`, `pilot_proc_song_clipcmp.json`.
- **Verifier outputs:** `v5_va\verify_pilot_qc_out\` (`final_sides.json`, per-case CT and V3 checks),
  `v5_va\verify_pilot_qc_out_recount.json`, and `v5_va\verify_pilot_wall_out\` (`collect.json`, `tables.json`).
- **State files:** `v5_va\state\pilot_*.md`, `verify_pilot_*.md`. All read "complete", and no pilot process is running.
- **Data** (`D:\Arjun\neve\carotid_data\va_sources\`, gitignored):
  - `thin_slice\{song,ia172,aortaseg}\<case>\` holds the CT and meta;
  - `derived\thin_slice\{song,ia172,aortaseg}\<case>\` holds TotalSegmentator, Arterial, traces, `qc\` (QC PNGs, 3-D
    wall renders, detail-along-VA plots, `qc.json`, `select.json`) and `ct_seg\ct_seg4\` (walls, `measure.json`,
    `edge.json`).

Licence notices to keep with any derived anatomy:
- Song et al. 2024 (CC BY 4.0).
- The Head-CTA 172-series set, Zenodo 15697196 (CC BY 4.0).
- AortaSeg-60, Zenodo 18147026 (CC BY 4.0 as treated here).
- TotalSegmentator: Wasserthal et al. 2023 (CC BY 4.0).
- Arterial: weights CC BY-NC 4.0, code PolyForm Noncommercial.
- VMR 0248: Wilson et al. 2013.
