# verify_v5_assign: adversarial check of the v5 assignment and the unbaked anatomies_v5

Agent prefix `verify_v5_assign`. State `v5_va/state/verify_v5_assign.md`. Own code only:
`/sp/v5_va/verify_v5_assign_{struct,geom,dry,split,probe,extra,misc}.py` (+ `verify_v5_assign_split_run.sh`); outputs in
`/sp/v5_va/verify_v5_assign_out/`. One container of my own (`verify_v5_assign_w`, eve-training-fixed, `--cpus 8`) with
`/repo` mounted READ-ONLY, so nothing in the repo, anatomies_v4, anatomies_v5, the registry or the v5_assign tools could
be modified. No git. The geometry, split and assignment checks import none of graft_v5_reg / graft_v5_va / graft_v4 /
graft_v5_assign / v5_split (the one exception is check (a) in section 7, which calls graft_v5_reg.build on purpose to
test its handedness refusal).

## Answer so far

(final, 15:40)

**Verdict: the claims hold, with three small qualifications.** Every key number I recomputed reproduces exactly,
including the split. No gate fails on any written VA under my own code.

0. **Split.** Recomputed independently: 45 / 45 quantities are equal on v4 and v5 and equal their split_v4 /
   split_v5:
   - v3: 153 / 16 / 54, max 183 / 16 and 181 / 17;
   - strict: 127 / 16 / 80, max 174 / 16 and 172 / 17;
   - 25 V: donors, each nested in one patient.
   A shared-donor positive control drops the roster 153 → 138, so the check has teeth.

1. **One-to-one and handedness.** 25 units go to 25 TopBrain patients; each patient has exactly one unit and each unit
   exactly one patient. The donor patient keys are distinct, and the written assignment equals plan_v5.json. Every
   replaced side follows the follow rule in the written xform and in the v5_va record: native anatomies take the
   donor's R on the RVA, mirrored ones take its L with mirror [-1, 1, 1]. Both VAs of an anatomy come from the same
   unit.
   - Chirality check: for 419 of 431 VAs, the correctly-handed donor fits the host's v4 corridor better (proper
     Kabsch) than the mirror image does.
2. **Reserves.** ia172_172 and Song 138 appear in no written anatomy. Every TopBrain patient has 23–25 primaries
   that pass at least one VA there. My own exact lexicographic MILP gives the claimed assignment and reaches every
   optimum: served 25, both 208, VAs 431, RVAs 212, coarseness 33.2842, unflag points 364, take-off kept 179.
   - Variant A reproduces: 209 / 431 + 1 / 213, with reserve ia172_172 on mr_015 as 1 LVA.
   - Coarseness first reproduces: coarseness 39.37, both-sides anatomies 184, VAs 407, 17 units moved.
3. **Gates, own code, all 223 anatomies (431 replaced VAs, 0 failures).** The 25-anatomy random sample (seed
   20261002) has 48 replaced VAs, with 0 failures.
   - Clearance (exact all-pairs): exit 5.9–11.9 mm (≤ 12), no re-entry, parent after exit ≥ 0.61 mm, other branches
     ≥ 0.602 mm (the other VA as written included), self ≥ 0.85 mm.
   - Turns: within ±3 mm of O_h ≤ 59.9°, of the ostium ≤ 56.6°, of the splice cut ≤ 59.0°.
   - Rotations: det = +1 for every vaC and TopBrain rotation.
   - Course RMS ≤ 19.87 mm; tip move ≤ 48.5 mm.
   - Take-off on the placed donor line 31.6–119.99°; rise ≥ 15.2°.
   - The splice lies in [C2, dura] on every VA (10 sit at the C2 edge; see qualifications).
   - Calibre at the cut ≤ 1.558. All 52 VAs above 1.25 are flagged as pass 2, and no VA is flagged without needing it.
   - Radii ≥ 1.0 mm.
   - Mirror handling: the written donor stretch lies ≤ 0.17 mm from the record's centreline transformed with the
     written mirror and R, against 4.8–38 mm when the mirror is flipped. The donor SURFACE transformed the same way
     sits at the MISR radius (median |d − r| ≤ 0.036 mm), against 0.8–37 mm flipped.
   - The TopBrain piece past the cut is v4's own piece moved rigidly: Kabsch residual ≤ 0.034 mm, det +1, radii
     unchanged.
4. **Kept v4 VAs.** All 15 one-sided anatomies keep the other VA byte-identical to v4, with xform.va_* equal to v4's.
   Each is flagged (summary.flags, v5_va.<side>.v4_va_kept) and marked not served in the assignment block. Elsewhere:
   - every non-VA curve is byte-identical to v4;
   - the replaced RVAs' 28-point subclavian prefix is identical (212 / 212), and every LVA starts at (19)[0];
   - provenance equals v4's outside xform.va_* / vaC_* / v5_va.
5. **Written == dry run.** For every anatomy and side, written equals the assigned unit's dry pass, and 7 recorded
   numbers are identical. Re-verification therefore changed nothing.
6. **No TaG-Net.** The donor sources are ia172 (13 units), Song (11) and VMR (1); no provenance text mentions TaG-Net.
   The 54 donor surface copies' md5s equal the record, the eligible list and the graded source file.
7. **Licences.** 216 anatomies are NC, which matches "an Arterial-derived donor is used". Every anatomy has
   attributions, including TotalSegmentator for the 216 Song / 172 anatomies, and LICENSE_v5_donors.txt names every
   donor with the NC notice where it applies. The 7 VMR anatomies carry LICENSE_VMR_0248.txt, identical to the source.
8. **anatomies_v4 untouched.** Of its 4,907 files, none was modified on or after 2026-09-28 (the newest is
   BUILD_v4.json, 09-27). The obj triangle count equals mesh_v4.json in 223 / 223. The v4 VA arclengths equal
   provenance `vertebrals.*.total_mm` on 446 / 446 (max |Δ| 6e-14). An md5 manifest is saved for later compares.
9. **Key numbers reproduced.** Dry run: 12,042 / 8,688 / 0.7215 with 0 errors; per-gate base rates equal the claim
   exactly; per-unit R / L / both equal the notes table. Written set: RVA 212, LVA 219, both 208, one 15 (4 + 11),
   none 0. Per-patient table identical. Take-off not preserved 252 (median 17.77°). Calibre pass 2: 52. Stub median
   5.28 mm, max 10.49 mm.

**Qualifications (none changes a key number):**
- (Q1) **Take-off is gated on one chord and recorded on another.** The gate measures the take-off on the placed donor
  line, while the provenance `takeoff_10mm_deg` is measured on the written curve. Two written VAs have a recorded
  take-off outside [20, 120]°:
  - case_w_012_left__topcow_mr_021_L RVA (Song 113 L): 127.5° recorded, gate 119.99°;
  - case_w_002_right__topcow_mr_022 LVA (ia172_192 L): 120.8° recorded, gate 119.84°.
  Both donors are native > 120° (133°, 128°), and the pose search parked them at the band edge.
- (Q2) **10 splices sit at the C2 edge.** These are the ia172_165 R VAs on mr_001: their splice equals the window's
  lower bound (margin 0) on the prep line. Projected onto the raw centreline, they sit 0.23 mm below the raw C2
  landmark. This is a 1 mm sampling effect, not a placement outside V3.
- (Q3) **The VMR records understate surface genus.** They give 0 for both sides; my Euler count on the surfaces used
  gives R 2 and L 1. Every other assigned side's genus matches my count. Song 008 R and ia172_069 R have 2
  components. This is bake-risk information only.
- Notes:
  - The 64.9° step turn on 14 mr_026 VAs is 1 mm before the blind end of v4's own TopBrain piece; v4 has the same
    turn (65.0°).
  - VMR's patient_key is "vmr:0248" in the record but "vmr0248:VMR0248" in the eligible list. This is cosmetic: only
    one VMR unit exists.

## Checks

### 1. Structure (`verify_v5_assign_struct.py` → `out/struct.json`, `out/v4_manifest.json`)
All 223 anatomies have 0 issues. See Answer items 1, 4, 5, 6, 7, 8.

### 2. Geometry (`verify_v5_assign_geom.py` → `out/geom.json`; follow-ups `probe.py` → `out/probe.json`, `extra.py` → `out/extra.json`)
Gate definitions are re-implemented from the brief and notes/graft_reg.md:
- clearance is EXACT all-pairs, stricter than the tool's k = 8 nearest-centre profile for the parent;
- the parent set is (17) plus the RVA's own subclavian prefix, or (18) + (19) for the LVA;
- the others are every other curve file, the other VA as written included;
- self pairs must be > 12 mm apart in arclength.
On a 3-VA test the recomputed values equal the recorded ones: exit 9.985, parent 1.390, others 1.611 (RCCA), self
3.732, course RMS 17.414, take-off 59.60, calibre 1.12. On all 431 VAs, my located ostium and cut equal the
provenance `ostium_at_mm` / `cut_at_mm` exactly, and `lo_mm` / `hi_mm` equal ostium + real_from and the cut (|Δ| = 0).

### 3. Dry runs and assignment (`verify_v5_assign_dry.py` → `out/dry.json`)
See Answer items 2 and 9.
- Coarseness: the record's score equals the eligible list's per-side contact-free value (max |Δ| 0).
- Every one of the 25 assigned pairs passes at least 1 VA at its patient; ia172_039 passes both VAs at mr_015.

### 4. Tool extensions (`verify_v5_assign_misc.py` → `out/misc.json`)
- (a) `graft_v5_reg.build` refuses two wrong-handed plans (raises ValueError) and accepts the correct one.
- (b) All 111 ia172_172 L dry passes use an advance of ≥ 3 mm.
- (c) The 50 assigned donor sides all have chirality +1, QC pass / review and a real surface.
- (d) The stored compat regression shows 9,366 leaves and 0 differences.
- (e) v5_split.py has the exit-4 path.

### 5. Split (`verify_v5_assign_split.py`, run by `verify_v5_assign_split_run.sh` → `out/split.json`, `out/split.log`)
My own extraction and MILP cover the definitions v3, v4side and strict. Each definition gives the roster plus max
k = 16 and k = 17, with and without the 6 × 6 breadth constraint: 45 quantities in all. Every MILP is optimal and its
split re-checked as disjoint.

| def | roster train / test / excl | max k16 | max k17 | k16 6×6 | k17 6×6 |
|---|---|---|---|---|---|
| v3 | 153 / 16 / 54 | 183 / 16 | 181 / 17 | 181 / 16 | 180 / 17 |
| v4side | 127 / 16 / 80 | 174 / 16 | 172 / 17 | 172 / 16 | 170 / 17 |
| strict | 127 / 16 / 80 | 174 / 16 | 172 / 17 | 168 / 16 | 167 / 17 |

- The table is identical on anatomies_v4 and on anatomies_v5 (45 / 45 equal). It also equals their
  `results/split_v4.json` and `results/split_v5.json` on all 45 quantities.
- v5 has 25 V: donors in 25 TopBrain patients; none spans two patients, under all three definitions.
- **Positive control.** I gave mr_003 and mr_026 one shared V: donor. My checker then lowers the roster
  153 → 138 (v3) and 127 → 112 (v4side, strict), and v3 k16 6×6 197 → 196. These are exactly the drops notes/split.md
  reports for that pair, so the checker is not passing trivially.
