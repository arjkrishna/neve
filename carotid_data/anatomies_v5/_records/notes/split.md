# v5_split: donor-overlap train/test split checker

Date 2026-10-02. Tool `/sp/v5_va/v5_split.py`, tests `/sp/v5_va/v5_split_test.py`. Results `results/split_v4.json`,
`results/split_v3.json` (reference), `results/split_test.json` + `split_test_<set>.json`,
`results/split_check_{write_test,v5_proto}.json`, pair sweep `split_work/sweep/`. State `state/v5_split.md`. No repo file
touched; synthetic test sets in `carotid_data/va_sources/v5_reg/split_test/` (gitignored).

## Answer so far

1. **The user's "~170 (153 train / 17 test)" is the carotid v3 launcher split, not a maximum.** The four carotid
   v3 launchers (4eef6bb; GUIDE_stitched_datasets.md 7.2) hold out the block {6 Zenodo lowers} x {6 TopBrain
   patients} = 16 anatomies, exclude the 54 cross-terms and train on 153. `v5_split.py` reproduces it exactly,
   on v3 and on v4: **153 train / 16 test / 54 excluded = 169**, with donors = the RCCA-route Zenodo lower
   (case-side) + the TopBrain patient. Test is 16, not 17. No definition I tried gives 153 / 17 (closest: the
   same block with Zenodo donors at patient level gives 17 test but 144 train).
2. **It is not the maximum.** With the same donor definition the true maximum overlap-free split (MILP, proven
   optimal) is 183 / 16 (199) or 181 / 17 (198); even forcing the roster's breadth (>= 6 TopBrain patients and
   >= 6 Zenodo donors in test) gives 181 / 16 and 180 / 17.
3. **On v4 the v3 roster is no longer overlap-free.** v4 added the left carotid fork (the patient's own other
   side in 168 anatomies, a substitute in 55). 26 of the 153 roster-train anatomies share a Zenodo donor with the
   test block through a left fork, theirs or a test anatomy's (the GUIDE's 9 "other-side carotid" composites,
   w_007_right x5 and w_017_left x4, now hold the holdout side as their left fork and vice versa; the other 17 carry a
   holdout lower as their substitute left fork; checked). Counting every donor an anatomy contains, the same 16-anatomy test block leaves **127 train / 16 test / 80 excluded (143)**.
4. **v4's true maximum with every donor counted is 172 / 17 (189), 174 / 16 (190)** -- same at Zenodo
   case-side and patient level; 167 / 17 (184) with the 6 x 6 breadth constraint. All optimal.
5. **One-to-one VA donors cannot change any of these numbers** (proof below). Tested end to end on synthetic v5
   sets in both provenance conventions: 0 of 45 quantities change.
6. **A donor shared by two TopBrain patients lowers them when the two sit on opposite sides of the split.** Over
   all 300 pairs: the roster drops for 114 pairs, by 1-15 train anatomies (153 -> 138 at worst); the strict
   maximum drops for 58 pairs, by 1-4 (189 -> 185 at worst, pair 011 + 016); the rest are unchanged.

## Data: what each v4 anatomy contains (provenance.json, BUILD_v4.json)

| part | donor | identity used |
|---|---|---|
| RCCA route lower (CCA + bifurcation + cervical ICA, + RECA) | Zenodo CarotidAnalyzer case-side `lower` | `Z:case_w_007_left` (case-side) or `Z:case_w_007` (patient) |
| left fork (LCCA route + LECA) | `left.lower`: own other side 168, substitute 55 (30 absent, 15 too short, 10 failed gates) | as above |
| right siphon, left siphon, both VA distal pieces, ECA tips (52 R / 53 L extended) | one TopBrain patient (`patient`; every `topcow_mr_NNN` in `xform.*.surface/centerline_src` agrees, 0 mismatches) | `T:topcow_mr_016` |
| handedness | 120 mirrored (`_L`: the patient's left ICA mirrored; every other TopBrain part from the opposite side, x-mirrored), 103 native | same patient -> same donor |
| host (stEVE_bench DualDeviceNav) | shared by all 223 by design | not a donor node |

`case_w_018b` (left only) is "probably a second scan" of `case_w_018` (GUIDE 2); the strict definition merges them.
Counts: 25 TopBrain patients (1-16 anatomies each, median 8); 48 route lowers (case-side) from 39 Zenodo patients;
with left forks 74 case-sides / 43 patients (a substitute fork patient reaches 22 anatomies).
**Graph: one connected component of 223 anatomies under every definition** (v3, v4side, strict), so no split
exists without dropping anatomies; "whole components" cannot define a test set.

## Definitions and the algorithm

Donor definitions (`--defs`):
- `v3`: route lower (Zenodo case-side) + TopBrain patient (+ extra donors). The v3 launcher's own notion.
- `v4side`: every donor in the anatomy -- route lower and left fork (case-side), TopBrain patient, extra donors.
- `strict` (the guard): `v4side` with Zenodo donors at patient level and w_018b = w_018.

Quantities (each per definition):
- **roster_v3** -- the launcher split, generalised. Test = the fixed block (route lower in {m_030_right,
  w_007_left, w_013_right, w_017_right, w_025_right, w_033_left} and TopBrain patient in {003, 011, 013, 016, 017,
  021}; 16 anatomies). H = every donor of the test anatomies. Train = anatomies with no donor in H. Excluded = the
  rest. New donors can only move train anatomies to excluded.
- **max_k** -- max (train + test) over all assignments of donors to {test, train}, with exactly k test anatomies;
  an anatomy is test iff all its donors are test, train iff all are train, else dropped. MILP (scipy 1.10 /
  HiGHS): x_d, t_a, r_a binary; t_a <= x_d and r_a <= 1 - x_d for d in a; sum t = k; maximise sum t + sum r
  (+1e-3 per train anatomy as tie-break). Every returned split is re-checked for disjointness.
- **max_k_div6x6** -- max_k with u_d <= sum_{a contains d} t_a and >= 6 TopBrain and >= 6 Zenodo u_d = 1 (the
  roster's breadth).
- k = 16 and 17 by default (`--test-sizes`). 17 / 170 is the user's 10%.

## v4 numbers (`results/split_v4.json`; all MILPs optimal, 11-35 s each)

Reproduced by the final code: `results/split_v4_final.json` equals `split_v4.json` on all 45 quantities.

| definition | roster_v3 train / test / excl | max k=16 | max k=17 | k=16, 6x6 | k=17, 6x6 |
|---|---|---|---|---|---|
| v3 (route lower case-side + patient) | **153 / 16 / 54** (169) | 183 / 16 (199) | 181 / 17 (198) | 181 / 16 (197) | 180 / 17 (197) |
| v4side (all donors, case-side) | 127 / 16 / 80 (143) | 174 / 16 (190) | 172 / 17 (189) | 172 / 16 (188) | 170 / 17 (187) |
| strict (all donors, Zenodo patient) | 127 / 16 / 80 (143) | 174 / 16 (190) | 172 / 17 (189) | 168 / 16 (184) | 167 / 17 (184) |

v3 set (reference, `results/split_v3.json`; no left forks): the v3 and v4side rows equal the v4 `v3` row; strict
(Zenodo patient level) gives roster 144 / 16 / 63 and max 180 / 16, 177 / 17.

Optimal test sets are narrow: strict k=17 holds out 4 TopBrain patients (011, 014, 017, 018) and 5 Zenodo patients
(k_004, w_014, w_016, w_017, w_034); an equally good optimum swaps 017 for 015, so the optimal test set is not
unique. The 6 x 6 constraint costs 5 anatomies.

The 26 roster-train anatomies that overlap the test block in v4 (left forks): see
`results/split_v4.json` -> defs.strict.roster_v3.v3_rule_train_overlapping_test_names.

## One-to-one VA donors: why nothing can change

If every VA donor D appears only in anatomies of one TopBrain patient P (D may serve P's R and L, or P may have
two donors), then D's anatomy set is a subset of P's:
- max_k: any optimal split puts P wholly on one side (all P's anatomies contain T:P); give D that side -> the
  same split stays feasible and nothing new is gained, so the optimum is unchanged.
- roster_v3: D joins H only if P is a test patient, and then every anatomy containing D contains T:P, already in H.
The tool reports this as `extra_donors.one_to_one_nested` and lists any donor spanning more than one patient.

## Tests (`v5_split_test.py`; `results/split_test.json`, per-set `results/split_test_<set>.json`)

Synthetic v5 sets = the 223 v4 provenances plus VA donor fields, written under
`carotid_data/va_sources/v5_reg/split_test/<set>/` (gitignored scratch). Two conventions: graft_v5_reg.py's
(`xform.vaC_R / vaC_L` with `donor_id`, `donor_record`, surface under `v5_reg/donors/<id>/`) and the prototype's
(no `donor_id`; a `va_sources/derived/thin_slice/...` path). Each set was run through the CLI against
`results/split_v4.json` (45 quantities: 3 definitions x {roster, 4 max variants} x {train, test, kept}).

| set | VA donors | lowered vs v4 | notes |
|---|---|---|---|
| one_to_one (registry convention) | 25: one pair per TopBrain patient | **0 of 45** | every quantity identical; `one_to_one_nested` true |
| one_to_one_2_pathconv (prototype convention) | 50: R and L from different donors, each used by one patient | **0 of 45** | identical; nested |
| shared_mr_011__mr_016 (largest max drop) | 011's donor also on 016 | strict / v4side max k=17 **189 -> 185**, k=16 190 -> 186, 6x6 184 -> 182 | roster unchanged: both are holdout patients |
| shared_mr_003__mr_026 (largest roster drop) | 003's donor also on 026 | roster **153 -> 138** (v3 def), 127 -> 112 (strict) | max: only v3 k=16 6x6 197 -> 196 |
| shared_mr_001__mr_002 (no drop on the strict guard) | 001's donor also on 002 | strict unchanged; v3-definition max k=16 199 -> 197, k=17 198 -> 196 | a pair can pass one guard and fail another |

**Every pair (300; `split_work/sweep/sweep_summary.json`).** One donor shared by TopBrain patients A and B:
- roster_v3: it lowers train only when one of A, B is a holdout patient and the other a training patient (114
  pairs under the v3 definition = 6 x 19; 108 under strict). The drop is that training patient's whole roster-train
  count: 1-15 anatomies (v3 definition: 1 x6, 2 x12, 3 x12, 5 x18, 6, 7, 9, 10 x6 each, 12 x12, 13 x6, 14 x12,
  15 x12). The 186 other pairs: 0.
- max strict k=17 (172 / 17): pairs with both patients on one side of the optimum are provably unchanged (216);
  the 84 straddling pairs were re-solved (all optimal): **58 of 300 lower it, by 1 (22), 2 (32), 3 (2) or 4 (2)**;
  worst 011+016 and 014+024 (189 -> 185).
- max v3 k=16 (183 / 16): 100 of 300 lower it, by 1 (28), 2 (59) or 3 (13); worst x+025 (199 -> 196).
- No pair ever raises a quantity.

So the rule that guarantees "not lower than v4" is the one-to-one rule itself (each VA donor inside one TopBrain
patient). A shared donor is not always harmful, but which pairs are harmless depends on the definition and on
which optimal split one picks, so the checker should be run on the real v5 set rather than reasoned about.

**Real v5 provenances** (`--no-solve`): graft_reg's `v5_reg/write_test` (3 anatomies) -> 3 registry donors read
from `xform.vaC_*.donor_id`, nested (`results/split_check_write_test.json`); the 4 prototypes in
`va_sources/v5_proto` -> `V:vmr0248` spans 4 TopBrain patients and is flagged (`results/split_check_v5_proto.json`),
as expected for a single-donor prototype.

Run time: the CLI does 12 MILPs per set, 10-50 s each (about 3-6 min single-threaded). Note for batch use: a
multiprocessing pool forked after HiGHS has run in the parent spins forever; the test uses `spawn`.

## Caveats and decisions for the user

- **Which number is "the" v4 value.** The launcher roster (153 / 16 under the v3 definition) is what the user
  quoted. Counting the v4 left forks it is 127 / 16, and the best overlap-free split v4 allows is 172 / 17. v5 must
  keep all three; with one-to-one VA donors it does, by construction. Whether left forks count as overlap is a
  user decision: in RCCA training the left carotid is collision geometry, never the route, but it is the same
  lumen surface the test anatomy navigates.
- **The roster was never a maximum.** A larger donor-disjoint split exists (181 / 17 under the launcher's own
  definition), but it holds out only 4-5 TopBrain patients; the 6 x 6 constraint costs 1-5 anatomies.
- **VMR 0248** was "kept as a possible held-out case" (GUIDE 1). As a v5 VA donor it would sit in training
  anatomies, so it could no longer serve as a clean held-out patient.
- **Duplicate patients across open datasets** (172-series, Song, Large IA, RSNA) would hide overlap: two ids, one
  person. The registry's `patient_key` merges only identical keys; give known duplicates to `--donor-alias`.
- **TaG-Net** is never a donor; the tool exits 2 if one appears.

## Using it for v5

    docker run --rm --cpus 4 --entrypoint python3 -v "D:\\Arjun\\neve:/repo" \
      -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data" -v "<scratchpad>:/sp" eve-training-fixed \
      /sp/v5_va/v5_split.py /opt/eve_training/carotid_data/<anatomies_v5> \
      --baseline /sp/v5_va/results/split_v4.json --out /sp/v5_va/results/split_v5.json

Exit 3 if any quantity drops below v4's; exit 2 if a TaG-Net donor is present. VA donors are found automatically
from graft_v5_reg.py's `xform.vaC_R / vaC_L.donor_id` (registry ids such as `ia172_058`, `vmr0248`, `ts_s0610`),
merged by the registry `patient_key` when the donor record is readable (mount /repo), from
`va_sources/derived/...` or `va_sources/v5_reg/donors/<id>/` paths, or from `--extra-donor-field`.
`--donor-alias` merges ids known to be one patient (the report warns that series may be duplicated across the
172-series, Song, Large IA and RSNA sets).
