# pilot_decide: the thin-slice pilot's decision and the donor ranking for the v5 build

Agent prefix `pilot_decide`, first instance, 02:11–02:45 EDT 2026-10-02. State file `/sp/v5_va/state/pilot_decide.md`.
No git; no tracked file touched; no GPU; every input was only read (python:3 container, one at a time). Nothing else
was started. Licences: Song 2024 and the 172-series are CC BY 4.0, AortaSeg-60 is treated as CC BY 4.0, and everything
that went through Arterial is non-commercial. TaG-Net is only a reference here and is never a donor.

**Diff of this file:** the 02:40 "Answer so far" (4 lines) is replaced by the final answer below.

## Answer

Status: complete (02:45).

### 1. Donor need: 25 pairs, 30 with spares

- The 223 v4 anatomies use **25 distinct TopBrain patients** (`provenance.json → patient`). 103 anatomies are native and
  120 mirrored, which matches `BUILD_v4.json` (`graft.native` / `graft.mirrored`).
- **Every anatomy uses both VA pieces of its patient.**
  - In a native anatomy, the donor's R goes to the RVA and its L to the LVA.
  - In a mirrored anatomy (v3 siphon `*_L`), the donor's L goes x-mirrored to the RVA and its R to the LVA (the follow
    rule).
  - So each patient needs a complete donor pair. That includes mr_006, used only native, and mr_013 / 014 / 015 / 025,
    used only mirrored.
- **Need: 25 donor pairs, one-to-one; 30 with the 5 spares.**
- **Anatomies per patient run from 1 to 16.**
  - 16 each: mr_005, 011, 016, 020.
  - 15 each: mr_010, 023, 024, 026.
  - Then 12, 10, 9, 8 × 3, 7, 6, 5 × 2, 4 × 4, 2 × 2 and 1.
  - Assignment hint: give the coarsest pairs to the patients with the most anatomies. The training-reachable RVA takes
    the donor's R in native anatomies and its L in mirrored ones (per patient in `pilot_decide.json → need.per_patient`).

### 2. Per source (recalibrated gate = primary; study draft + additions; only the two-patient-backed changes)

| source | cases | Arterial ran | sides traced | sides passing: recal / draft / backed | ≥ 60 %? recal / draft / backed | complete pairs: recal / draft / backed |
|---|---|---|---|---|---|---|
| 172-series | 12 | 12 (centreline stage hung on 061 and 090; masks traced without labels) | 24 | 15 / 7 / 14 (62.5 / 29 / 58 %) | **go** / no / no (just short) | **5** (015 039 058 141 190) / 0 / 5 |
| Song 2024 | 10 | 10 (on the `ct_negzero_air1000` re-window) | 20 | 15 / 5 / 13 (75 / 25 / 65 %) | **go** / no / **go** | **6** (008 032 094 100 109 113) / 1 (032) / 4 (008 032 094 113) |
| AortaSeg-60 | 1 (exhausted) | 1 | 2 | 1 / 0 / 1 | (n = 1) | 0 / 0 / 0. The L side is a course donor only: its wall fails gate V1 (offset 0.189 mm) |
| **total** | 23 | 23 | 46 | **31 / 12 / 28** (67 / 26 / 61 %) | | **11 / 1 / 9** |

**Where the gates differ.** Recalibrated and draft disagree on 8 sides (172-series), 10 (Song) and 1 (AortaSeg).

- **Rule 6 alone** (Arterial label agreement): 172 015 R, 058 L, 118 L, 141 R, 190 R; Song 008 L, 044 L, 094 L, 100 L,
  113 R, 113 L.
- **Rule 6 plus the study V3 floor of 15 mm:** 172 039 L (13.1 mm); Song 008 R (12.8 mm), 106 L (14.3 mm).
- **Provisional, in-sample limits only:**
  - 172 001 R: a V1 bridge ≤ 3 mm, plus rule 6;
  - 172 140 L: the V2-band IQR term was dropped;
  - Song 100 R: the IQR term, plus rule 6;
  - Song 109 R: a V1 bridge ≤ 3 mm.
- **AortaSeg L:** the draft reads the ostium (2.23 mm), and rule 6 applies.

**Is rule 6 → review safe?** Yes, out of sample.
- All 6 rule-6 sides of the 172-series are correct VA courses (ia172 agent, by eye).
- I viewed the QC overview PNGs of the Song rule-6 pairs 008, 094, 100 and 113. All four trace VA courses with V3 loops
  behind C1, at overview level only. I did not view 044 L.

The backed-only gate loses Song pairs 100 and 109, which pass only through the provisional limits. Conversely, the
robust-cord d ≤ 6 rejects 172 140 R, which the study cord (12.9 / 3.8) would pass.

**Failure causes (15 sides failing the recalibrated gate).**
- **V1 / origin breaks: 7.**
  - Through enhanced lumen (Arterial mask breaks, rescuable): 172 061 L (310 HU) and Song 141 R (587 HU), plus Song
    044 R (480 HU), which also has an origin at 3.25 mm.
  - Not lumen: 172 041 R / L, 073 L (fat), Song 106 R.
- **The course leaves the VA above C2 or skips the V3 loop: 6.** 172 073 R, 090 L, 118 R, 140 R; Song 090 R / L
  (rotated head).
- **Wrong vessel from the start: 2.** 172 001 L; AortaSeg R (Arterial labelled every class "L*", so the trace went into
  the right ICA).

**Fixes, in order of value** (none of them tested here):
1. Re-join V1 mask breaks that run through enhanced lumen. This turns 172_061 and Song_141 into pairs, about 1 more
   pair per 10 cases.
2. Fix ct_seg4's C2-exit pinch (re-centre stations whose centre HU < T_loc). It truncated 2 of 15 Song walls.
3. Allow origin → C1f donors when the trace deviates only above the splice (140 R).
4. Block the carotids in the trace without using Arterial's labels.
5. Exclude contamination per ray or per triangle instead of per 10 mm bin.

Keep these from the pilot: Song's air1000 re-window; the 172-series clip to [−1024, 3071]; TotalSegmentator `--ml`
with the CPU split; the Arterial watchdog; and the 172-series pre-fetch arch check.

### 3. Wall: coarse and irregular, but mostly at contacts and unevenly spread

Per passing side: ct_seg4 stage-A, exact ownership k 64, acquisition voxel, native bars. "Contact-free" means the
10 mm bins in which fewer than 5 % of the edge rays meet a bright neighbour. The donor stretch runs origin → C1f − 25 mm,
with the seam zone kept separate. Source: `results/pilot_decide_tables.json`.

| source | case | side | QC recal / draft / backed | whole > 0.3 | C6→C1 > 0.3 | C6→C1 contact-free | donor stretch all / contact-free (share) | plain-tube bins whole / donor c-f | FWHM mm | noise HU | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ia172 | ia172_001 | R | donor_review / FAIL / FAIL | 0.227 | 0.239 | 0.132 | 0.248 / 0.153 (0.40) | 0.00 / 0.00 | 1.22 | 23.7 | C6→C1 on measure.json full ranges (ts) landmarks |
| ia172 | ia172_015 | L | donor / pass / pass | 0.171 | 0.262 | 0.069 | 0.213 / 0.036 (0.28) | 0.08 / 0.25 | 1.40 | 22.7 |  |
| ia172 | ia172_015 | R | donor_review / FAIL / pass | 0.171 | 0.253 | 0.092 | 0.212 / 0.131 (0.35) | 0.09 / 0.00 | 1.36 | 25.9 |  |
| ia172 | ia172_039 | L | donor_review / FAIL / pass | 0.148 | 0.194 | 0.096 | 0.172 / 0.106 (0.47) | 0.09 / 0.12 | 1.41 | 22.6 |  |
| ia172 | ia172_039 | R | donor_review / pass / pass | 0.094 | 0.084 | 0.061 | 0.092 / 0.078 (0.57) | 0.14 / 0.12 | 1.36 | 24.4 |  |
| ia172 | ia172_058 | L | donor_review / FAIL / pass | 0.250 | 0.340 | 0.229 | 0.330 / 0.276 (0.40) | 0.08 / 0.00 | 1.34 | 24.0 |  |
| ia172 | ia172_058 | R | donor / pass / pass | 0.075 | 0.091 | 0.033 | 0.102 / 0.072 (0.49) | 0.13 / 0.00 | 1.38 | 27.6 |  |
| ia172 | ia172_061 | R | donor / pass / pass | 0.245 | 0.306 | 0.133 | 0.318 / 0.244 (0.36) | 0.09 / 0.00 | 1.31 | 19.4 |  |
| ia172 | ia172_090 | R | donor_review / pass / pass | 0.262 | 0.312 | 0.163 | 0.354 / 0.318 (0.53) | 0.12 / 0.00 | 1.11 | 25.8 |  |
| ia172 | ia172_118 | L | donor_review / FAIL / pass | 0.247 | 0.302 | 0.176 | 0.294 / 0.249 (0.42) | 0.05 / 0.00 | 1.26 | 27.6 |  |
| ia172 | ia172_140 | L | donor / FAIL / FAIL | 0.214 | 0.237 | 0.033 | 0.320 / 0.209 (0.31) | 0.10 / 0.00 | 1.33 | 29.5 | wall gate fails (genus 1.0); C6→C1 on measure.json full ranges (ts) landmarks |
| ia172 | ia172_141 | L | donor / pass / pass | 0.037 | 0.065 | 0.010 | 0.038 / 0.012 (0.68) | 0.43 / 0.44 | 1.20 | 23.8 |  |
| ia172 | ia172_141 | R | donor_review / FAIL / pass | 0.125 | 0.193 | 0.156 | 0.149 / 0.032 (0.35) | 0.19 / 0.40 | 1.22 | 18.3 |  |
| ia172 | ia172_190 | L | donor_review / pass / pass | 0.026 | 0.028 | 0.014 | 0.038 / 0.030 (0.80) | 0.43 / 0.20 | 1.33 | 30.8 |  |
| ia172 | ia172_190 | R | donor_review / FAIL / pass | 0.055 | 0.063 | 0.032 | 0.072 / 0.056 (0.64) | 0.36 / 0.20 | 1.30 | 28.4 |  |
| song | S008 | L | donor_review / FAIL / pass | 0.255 | 0.288 | 0.139 | 0.288 / 0.079 (0.14) | 0.08 / 0.00 | 1.23 | 13.0 |  |
| song | S008 | R | donor_review / FAIL / pass | 0.176 | 0.198 | 0.037 | 0.244 / 0.010 (0.06) | 0.30 / 1.00 | 1.21 | 16.3 |  |
| song | S032 | L | donor / pass / pass | 0.065 | 0.074 | 0.033 | 0.076 / 0.050 (0.56) | 0.35 / 0.62 | 1.24 | 23.8 |  |
| song | S032 | R | donor / pass / pass | 0.064 | 0.085 | 0.042 | 0.046 / 0.021 (0.48) | 0.46 / 0.50 | 1.19 | 23.8 |  |
| song | S044 | L | donor_review / FAIL / pass | 0.054 | 0.062 | 0.042 | 0.059 / 0.040 (0.59) | 0.19 / 0.44 | 1.27 | 16.6 |  |
| song | S094 | L | donor_review / FAIL / pass | 0.258 | 0.353 | 0.090 | 0.396 / 0.083 (0.12) | 0.09 / 0.00 | 1.04 | 17.3 |  |
| song | S094 | R | donor / pass / pass | 0.143 | – | – | – / 0.062 (0.58) | 0.25 / 0.43 | 0.95 | 17.8 | wall truncated at C2 bend; C6→C1 on measure.json full ranges (ts) landmarks |
| song | S100 | L | donor_review / FAIL / pass | 0.061 | 0.083 | 0.081 | 0.073 / 0.071 (0.88) | 0.43 / 0.43 | 1.20 | 20.2 |  |
| song | S100 | R | donor_review / FAIL / FAIL | 0.086 | 0.103 | 0.110 | 0.115 / 0.129 (0.84) | 0.17 / 0.09 | 1.19 | 17.1 |  |
| song | S106 | L | donor_review / FAIL / pass | 0.220 | 0.257 | 0.051 | – / 0.134 (0.38) | 0.06 / 0.14 | 1.20 | 17.8 | wall truncated at C2 bend; C6→C1 on measure.json full ranges (ts) landmarks |
| song | S109 | L | donor / pass / pass | 0.077 | 0.118 | 0.032 | 0.103 / 0.009 (0.28) | 0.40 / 0.75 | 1.03 | 14.4 |  |
| song | S109 | R | donor / FAIL / FAIL | 0.045 | 0.048 | 0.007 | 0.065 / 0.035 (0.34) | 0.63 / 0.50 | 1.07 | 21.7 |  |
| song | S113 | L | donor_review / FAIL / pass | 0.162 | 0.232 | 0.051 | 0.232 / 0.133 (0.31) | 0.17 / 0.20 | 1.31 | 19.2 |  |
| song | S113 | R | donor_review / FAIL / pass | 0.156 | 0.258 | 0.184 | 0.194 / 0.076 (0.43) | 0.17 / 0.17 | 1.35 | 20.0 |  |
| song | S141 | L | donor / pass / pass | 0.073 | 0.136 | 0.031 | 0.095 / 0.019 (0.60) | 0.48 / 0.67 | 1.16 | 12.7 |  |

The C6 → C1 contact-free values and the bin-based C6 → C1 for the 4 walls without CT landmarks (001 R, 140 L, Song 094 R
and 106 L) are recomputed from the per-case `measure.json` bins10 and `edge.json` per_10mm with the sibling grades'
contact rule. The recomputed contact-free donor values equal the grades' values on 30 of 30 sides.

| set, valid walls | whole > 0.3 median (range) | C6 → C1 > 0.3 | donor contact-free median (range) | plain-tube bins whole | edge FWHM mm | noise HU |
|---|---|---|---|---|---|---|
| 172-series, 14 | **0.160** (0.026–0.262) | 0.217 | **0.092** (0.012–0.318) | 0.10 (0.00–0.43) | 1.12–1.41 | 18–31 |
| Song, 15 | **0.086** (0.045–0.258) | 0.127 | **0.062** (0.009–0.134) | 0.25 (0.06–0.63) | 0.95–1.35 | 13–24 |
| all pilot, 29 | 0.143 | 0.194 | 0.072 | 0.17 | 0.95–1.41 | |
| VMR 0248 R / L (same harness) | 0.045 / 0.138 | 0.076 / 0.129 | 0.035 / 0.076 | 0.55 / 0.35 | 1.39 / 1.44 | 28 |
| TaG-Net A R / L (reference only) | 0.045 / 0.111 | 0.071 / 0.152 | 0.038 / 0.061 | 0.76 / 0.23 | 1.30 / 1.29 | 15 |
| TopBrain pieces | 0.121 median; 0.102 gate-valid, use-weighted | – | (no contacts in MRA) | 0.14 pooled / 0.125 per piece | – | – |
| host | tube 0.0007; dense RVA mesh 0.027 | | | | | |

- **The relief is real image structure.** The noise-floor phantoms (172-series: featureless tube, real PSF and noise,
  same method) read 0.0008 on 061 R (0.019 at twice the noise) and 0.005 on 140 L. The real walls read 0.245 / 0.214.
  Phantom plain-tube bins are 0.62–0.95, against real 0.09–0.10.
  - Song had no phantom run. Its noise (13–24 HU) and FWHM (0.95–1.35 mm) are at or below the 172-series values, so
    its floor should be no higher. That is an inference, not a measurement.
- **Coarse versus the host.** 28 of the 29 valid whole VAs beat the dense host RVA mesh (0.027); ia172_190 L reads 0.026.
  On the set median that is ×5 the dense mesh and ×200 the host tube.
- **Irregular, and more evenly than VMR or TaG-Net.** The plain-tube bin fraction is 0.17, against 0.35–0.76 for
  VMR / TaG-Net and 0.62–0.95 for the phantoms. The 172-series (0.10) matches TopBrain (0.13–0.14).
- **Most of it sits at contacts.** 69 % (172-series) and 76 % (Song) of the relief area lies in contact bins: bone of
  the foramina, and enhanced veins. Union there is 0.237 / 0.217, against 0.080 / 0.046 elsewhere.
- **With contamination excluded (the user's rule):**
  - The donor stretch reads a median 0.072. That is the level of VMR's left side (0.076) and TaG-Net A's left
    (0.061), and below TopBrain (0.10–0.12).
  - The 172-series is coarser (0.092) and has a coarse tail: 090 R 0.318, 058 L 0.276, 118 L 0.249, 061 R 0.244,
    140 L 0.209. Song tops out at 0.134.
  - 5 of 30 donor sides fall below the dense host mesh on contact-free bins: 172 141 L, and Song 008 R, 109 L, 141 L,
    032 R.
- **Evenness is poor in three ways.**
  - Along the VA: V4 and C1 → dura are smooth on every wall, so W7 fails on 30 of 30. This does not matter for strategy
    B. The relief sits in V1–V2 and the C2 loop.
  - Between the two sides of a pair: 058 reads 0.072 / 0.276.
  - Between cases: a ×35 range.
- **Set grades at native bars.** The whole VA passes W1, W2 and W3 on the 172-series (0.160 / 0.076 / 0.227). On Song it
  fails W1 and W3 and passes W2 (0.086 / 0.053 / 0.130). C6 → C1 passes W1 and W3 on both sources.

### 4. Donor ranking (`results/pilot_donor_ranking.json`)

The score is the mean over R and L of the contact-free donor-stretch union > 0.3 mm. The donor stretch is the piece
strategy B grafts. Whole-VA and C6 → C1 unions (exact, all bins, contact-free) and the plain-tube fractions are given
beside the score.

Within each source, the ranking agrees with the plain-tube fraction (Spearman 0.98 for the 172-series, 0.94 for Song) and
with the contact-free whole VA (0.90 / 0.94). It agrees less with the all-bins whole VA (0.60 / 0.49), which is where
contact relief would have reordered it.

| rank | 172-series pair | score (min side) | Song pair | score (min side) |
|---|---|---|---|---|
| 1 | **058** | 0.174 (0.072) | **113** | 0.104 (0.076) |
| 2 | 039 | 0.092 (0.078) | 100 | 0.100 (0.072) |
| 3 | 015 | 0.084 (0.036) | 094 | 0.073 (0.062): R wall truncated; L V1 by a contrast-filled vein |
| 4 | 190 | 0.043 (0.030) | 008 | 0.044 (0.010): contact-free on 6 / 14 % of the stretch |
| 5 | 141 | 0.022 (0.012): L below the host mesh | 032 | 0.036 (0.021): the only draft-passing pair |
| 6 | | | 109 | 0.022 (0.009) |

- **VMR 0248** (eligible, reference): 0.055 (R 0.035, L 0.076; the left V1 vein bins are excluded by the contact rule).
- **TaG-Net A** (not eligible): 0.049.
- **Merged build order.** Round-robin over the two sources by within-source rank, with the coarser candidate of each
  round first; VMR is inserted by its score. The order is 058, S113, S100, 039, 015, S094, VMR, S008, 190, S032, 141,
  S109.
- **All 12 eligible pairs are used now** (need 25). The order sets assignment priority and decides which pairs are
  displaced once the pool passes 30.
- **Coarse enough on both sides** (contact-free ≥ 0.027): 058, S113, S100, 039, 015, S094, VMR and 190, so **8 pairs**.
- **Single passing sides** that cannot pair yet:
  - 172: 090 R 0.318, 118 L 0.249, 061 R 0.244, 140 L 0.209 (genus 1), 001 R 0.153.
  - Song: 106 L 0.134 (partial wall), 044 L 0.040, 141 L 0.019.
  - AortaSeg: Dissec_01 L, course only.

### 5. Scale-up (`results/pilot_next_cases.json`)

**Pairs per processed case** (95 % Wilson intervals):

| gate | 172-series | Song | both |
|---|---|---|---|
| recalibrated | 5 / 12 = 0.42 (0.19–0.68) | 6 / 10 = 0.60 (0.31–0.83) | **11 / 22 = 0.50 (0.31–0.69)** |
| backed-only | 5 / 12 (0.19–0.68) | 4 / 10 (0.17–0.69) | 9 / 22 = 0.41 (0.23–0.61) |
| draft | 0 / 12 | 1 / 10 | 1 / 22 = 0.05 (0.01–0.22) |
| coarse enough on both sides | 4 / 12 | 3 / 10 | 7 / 22 = 0.32 (0.16–0.53) |

**More cases needed to reach 30 pairs.** Cases alternate 1:1 between Song and the 172-series. The estimate uses a
Jeffreys beta-binomial predictive per source.

| scenario | eligible now (incl. VMR) | still needed | central | 80 % | 90 % |
|---|---|---|---|---|---|
| **recalibrated (primary)** | 12 | 18 | **36** (18 + 18) | **44** | **50** |
| backed-only gate | 10 | 20 | 49 | 62 | 72 |
| coarse-enough pairs only | 8 | 22 | 70 | 90 | 106 |
| recalibrated + V1 rescue (untested) | 14 | 16 | 27 | 32 | 36 |
| draft gate | 2 | 28 | not reachable within the pools | | |

**Pools left** (distinct patients, z ≥ 300 mm):
- **Song: 122** (Siemens 0.7 mm 4, GE 0.625 mm 84, Philips 0.45 mm 34). No identical sex / age / grid duplicates.
- **172-series: 75**, after these exclusions:
  - not Head+Neck or z < 300 mm: 52;
  - slices > 0.75 mm: 19;
  - posterior-circulation aneurysms (038, 049, 093, 095, 099, 100): 6;
  - one member kept per same-demographics component, 7 dropped (042, 098, 110, 147, 168, 173, 176).

**Selection list** (`pilot_next_cases.json → selection`, 197 cases, ordered):
- Song and the 172-series alternate, Song first.
- **Song order:** Siemens 0.7 mm first (pilot: 3 of 3 cases gave pairs), then GE 0.625 and Philips 0.45 interleaved
  5 : 2 by pool share. Within a stratum, age < 75 comes first, then the longer z.
- **172-series order:**
  1. the fetch agent's 8 reserves;
  2. 0.5 mm FC43 controls;
  3. 0.5 mm anterior-aneurysm cases;
  4. FC04;
  5. 0.6–0.75 mm series.

  Within a tier, age < 75 comes first, then the finer pixel, then the longer z. Every 172-series case needs the pre-fetch
  arch check (lowest slice: two aortic limbs).
- **Age < 75 first:** the pilot gave pairs in 1 of 5 cases aged ≥ 75, against 10 of 17 younger ones. That is weak
  evidence.
- **The first 10:** S111, 172_062, S138, 172_119, S096, 172_060, S120, 172_067, S020, 172_160.
- **Batches:**
  - Batch 1 = orders 1–36, the central estimate.
  - Re-rank after batch 1. Extend to 44 (80 %) or 50 (90 %) if needed.
  - Stop once eligible pairs (QC-passing, both walls valid) reach 30.

**Per-case cost on this machine with TotalSegmentator `--ml`:**

| | GPU (TotalSegmentator `--ml` + Arterial) | CPU (split, 1.5 mm copy, traces, VMTK, QC, ct_seg4, landmarks, measure, edge, figures) | fetch |
|---|---|---|---|
| Song (10 cases) | median 356 s (max 720); TotalSegmentator 72–187 s, Arterial 167–620 s | median 1192 s (max 1985) | 20–71 s range-read, 60–109 MB |
| 172-series (3 `--ml` cases) | median 577 s (max 1521); TotalSegmentator 205–294 s, Arterial 265–1293 s | median 1643 s (max 1880) | 31–56 s + 33–100 s convert, 0.3–0.6 GB |

- **Serial cost:** about 27 min per Song case and 40 min per 172-series case, fetch included.
- **With two lanes** (one per source, each with one heavy CPU container, sharing the GPU lock):
  - 36 cases ≈ 12 h (GPU busy 4.7 h);
  - 44 cases ≈ 14.5 h;
  - 50 cases ≈ 16.5 h.

  The 172-series lane is the bottleneck.
- **Disk:** about 0.4–0.8 GB of derived data per case. D: has 505 GB free.

### 6. Decision

**GO for the scale-up under the recalibrated gate, on both open sources.**
- The 172-series passes 62.5 % of sides (5 pairs from 12 cases) and Song 75 % (6 pairs from 10). AortaSeg is exhausted.
- The study draft fails both sources (29 % / 25 %). With only the backed changes, Song passes (65 %) and the 172-series
  falls just short (58 %).

**The supply falls short of the need.** 12 pairs are eligible now (11 thin-slice + VMR) against 25, or 30 with spares.
Process about 36 more cases, 18 + 18, first; 44–50 give 80–90 % confidence. That is about 12–17 h with two lanes.

**What the walls do and do not have.**
- They are real, coarse, irregular relief, far above the host.
- Away from contacts they are VMR-like, and the 172-series has the coarse tail.
- If only pairs whose contact-free walls beat the dense host mesh on both sides count, 8 qualify now, and about 70–106
  more cases would be needed.

**Before the bulk run:**
- Fix the ct_seg4 C2-exit pinch.
- Try the V1 mask-break rescue. It is cheap, and it gives +2 pairs from cases already on disk.

## Open issues

- **The recalibrated gate's provisional limits are untested out of sample.** V1 ≤ 3 mm and the dropped IQR term alone
  admit 4 of the 31 passing sides (172 001 R, 140 L; Song 100 R, 109 R).
- **The contamination exclusion is per 10 mm bin.**
  - Dim venous plexus merged at half-max stays in.
  - Every bone-contact bin is excluded, real foramen wall included, so the exclusion is conservative.
  - Some contact-free scores rest on less than 20 % of the stretch: Song 008 R / L, 094 L.
- **094 needs a look before use.** Its R wall is truncated, and its L V1 runs at a contrast-filled vein. 140 L has a
  genus-1 handle.
- **The yield rests on 22 cases.** The strata and age orderings are weak evidence. The 172-series also loses about 1 in
  13 fetched series to arch truncation.
- **Graft compatibility will filter further.** The registry dry run passes 66.5 % of unit-anatomy-sides (graft_reg), so
  the 5 spares are needed.
- **Rescues not attempted:** 061 L, 141 R, 044 R, 041 R, and 140 R origin → C1.
- **Licence.** Every Arterial-derived donor makes its v5 anatomies non-commercial. Song and the 172-series need CC BY
  attribution.

## Method and files

- `pilot_decide_need.py`: the provenance count, written to `results/pilot_decide_need_rows.json`.
- `pilot_decide_collect.py`: the per-side table, yield, wall summary and ranking, written to
  `results/pilot_decide_tables.json` and `results/pilot_donor_ranking.json`.
  - It recomputes the contact-free values over the whole VA, C6 → C1 and the donor stretch from bins10 + per_10mm, with
    the sibling grades' rule.
  - It computes VMR and TaG-Net A from `results/tag_ctwall_{measure,edge}.json`.
- `pilot_decide_next.py`: rates, Wilson intervals, the beta-binomial need, pools, the ordered selection, cost and
  throughput, written to `results/pilot_next_cases.json`.
- `pilot_decide_final.py`: assembles `results/pilot_decide.json`.
- `pilot_decide_mdtable.py`: the per-side table above.
- `pilot_decide_peek.py`: a JSON structure printer.
- Inputs read:
  - `results/pilot_proc_{ia172,song}_grade.json` and `_summary.json`, `results/pilot_proc_aortaseg.json`;
  - the per-case `ct_seg/ct_seg4/{measure,edge}.json`;
  - `results/pilot_fetch_song_table.json`, `results/pilot_ia172_candidates.json`;
  - `carotid_data/anatomies_v4/*/provenance.json` and `BUILD_v4.json`;
  - the feasibility report, `notes/verify_tag_calibration.md`, the sibling notes and state files.
