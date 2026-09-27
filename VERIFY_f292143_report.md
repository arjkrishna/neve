# Verification report: commit f292143, "Bring the dataset build records up to the v2 and v3 sets"

Scope: the five docs changed in f292143 (TOPBRAIN_PIPELINE.md, CAROTID_THREE_SOURCE.md, CAROTID_VERIFICATION_HISTORY.md, MESHING_PIPELINE_ANALYSIS.md, V2_BUILD_PLAN.md) and the commit message. Final verdicts come from the verifier and adversary rulings. The line numbers below are for HEAD f292143. Versions: v1 is the blur mesher; v2 is the signed-distance (SDF) tube mesher at 0.45 mm; v3 is the v2 tube unioned with the real surfaces.

---

## 1. Verdict summary

| Verdict | Count |
|---|---|
| confirmed | 46 |
| partially | 26 |
| refuted | 5 |
| unverifiable | 2 |
| **total** | **79** |

- **Disputed claims:** 16 (C4, C5, C6, C7, C8, S3, S4, S5, S7, S9, S10, R6, K6, K7, K9, L3).
- **Refuted:** S3 (30 of 223 at 50 % or more), S4 (47 at 40 % or more), S5 (the grade is measured on the shipped centerline), K9 ("geometry unchanged in v2/v3") and L3 (rise and kink statistics unchanged).
- **Unverifiable:** P2 and P4. Both are facts about the external TopBrain and TopCoW datasets, which are not on this machine.

**Overall reliability.** Most of f292143 is reliable. The following were all reproduced from shipped data and code:
- set counts (215, 223, 223; 49, 49, 49)
- the mesher descriptions for v1, v2 and v3
- the constants and floors per version
- the byte identity of v2 and v3 centerlines in both sets
- the navigability counts (A 22 / 49 / 49, B 71 / 223 / 223)
- the rebuild flags
- the loader behaviour

The errors are concentrated in three places:

1. **Stenosis figures.** The commit says the grade is measured "on the centerline, after the floor, the way build_manifest.py measures it". Its numbers (56 %, 30 at 50 % or more, 47 at 40 % or more) do not come from that. They come from a paper calculation on donor lower_manifest values, `1 − max(min r, 1.0)/donor distal r`.
   - Measured on the shipped v2/v3 ICA, the maximum is 60.4 % (own distal) or 56.0 % (donor distal).
   - The shipped count is 24–25 at 50 % or more and 33–35 at 40 % or more.
   - The commit-message correction "estimated ~60 % → measured 56 %" is therefore mis-framed: both numbers are measurements under different definitions, and ~60 % is the one that matches the doc's stated method.
2. **Causal explanations.** Several are wrong or too broad:
   - Why the eight anatomies were gained: one of the five readmitted severed anatomies was never floored, and the three `case_w_014_right` pairs were ECA re-entry rejections that the ECA floor fixed.
   - Why 14 pairs remain unbuilt: mostly outright overlaps, and the evidence is reconstruction only.
   - What keeps the Set A label necks open: the SDF mesher, not the floor.
   - What the SOFA samples prove: they never reach the seam.
3. **"Geometry unchanged in v2/v3" (Set A).** This is false. Removing the distal trim and adding the siphon floor change route length, rise, worst bend and minimum diameter.

A smaller but real gap: the docs report "223/223 on every check" without noting that the checker has no topology test. v3 adds handles, including one device-passable RCCA–RVA merge.

---

## 2. Errors and overstatements in f292143

Each entry gives what the doc says, what is true, the evidence, and replacement text. When one passage covers several claims, the replacement is given once and referenced.

### C4, C6, C7, C8: the eight anatomies gained and the 14 unbuilt (Set B v2/v3)

**Doc** (CAROTID_THREE_SOURCE.md:52-61, *Current state (v2 / v3)*):
- The five severed anatomies are kept open by "the signed-distance mesher and the 1.0 mm siphon floor".
- The three `case_w_014_right__*` pairs were "rejected for fusing. The lower floors put their clearance back above the fusing band."
- "The 14 pairs still not built are fusing rejections."

MESHING_PIPELINE_ANALYSIS.md:353-356 (§7) and the commit message say the same.

**True:**
- **C4 (partially).** Four of the five got the siphon floor: the two `topcow_mr_007_L` anatomies (raw minimum 0.549 mm) and the two `topcow_mr_003_L` anatomies (raw minimum 0.952 mm). `case_w_040_left__topcow_mr_023_L` has no `siphon_*` keys in its v2 provenance, and its raw siphon minimum is 1.317 mm. It is open in v2 with a lumen of 1.228 mm and no floor, so the SDF mesher alone keeps it open. The adversary adds, without a re-run: without the floor the SDF mesher still keeps all five connected, and on 007_L the floor buys navigability rather than openness.
- **C6 (partially).** v1 rejected the three pairs at the ECA re-entry (ring) gate, not the route-clearance gate. At the 1.6 mm ECA floor, the gap between the ECA and the route after the opening run is about −0.24 mm. The re-entry is at 16.14 mm, giving a cutoff of 15.64 mm, which is under ECA_MIN_MM = 17. The grafter would raise "ECA re-enters the route at 16 mm (need 17)".
- **C7 (partially).** Only the ECA floor matters (ECA_MESH_R_MM 1.6 → 1.0). The route floor has no effect: route_min_r_raw is 1.80 mm and route_floored_frac is 0. The quantity that goes back above the 0.35 mm band is the ECA-to-route gap, which moves to +0.36 mm. The recorded clearance_mm (0.76–0.82 mm) is not what moved.
- **C8 (partially).** 14 of 237 pairs are unbuilt, and no grafter log exists. A reconstruction puts 13 at the final clearance gate, whose raise text is "fuses below 0.35". Most of those are outright overlaps with the LVA or LCCA, at −0.2 to −4.1 mm. `case_w_016_right__topcow_mr_015_L` sits at 0.00 to +0.18 mm and `case_w_003_right__topcow_mr_017` at −0.08 to −0.46 mm. `case_w_026_left__topcow_mr_002` is indeterminate: −0.85 to +3.28 mm depending on the donor.

**Evidence:**
- provenance `siphon_*` keys read per anatomy
- `grep -l siphon_floor_mm anatomies_v2/*/provenance.json | wc -l` gives 22
- adj_c6.py replays `graft_three.eca_reentry`: at the 1.0 mm floor the gap is +0.367/+0.358/+0.361 mm with no re-entry; at 1.6 mm it is −0.233/−0.242/−0.239 mm with re-entry at 16.14 mm
- graft_three.py:712-717 and :760-762
- adj_c8.py reconstruction, validated on 6 built pairs to within about 0–0.7 mm

**Replace CAROTID_THREE_SOURCE.md:52-62 with:**

> v2 gains eight anatomies over v1 and loses none:
>
> - **five** that v1 excluded after baking as severed at the siphon terminus (`excluded_severed.json`). In v2/v3 all five are one mesh component and navigable. Four (the `topcow_mr_007_L` and `topcow_mr_003_L` pairs) were regrafted with the 1.0 mm siphon floor. `case_w_040_left__topcow_mr_023_L` was not floored in v2 (raw siphon minimum 1.32 mm), so for it the signed-distance mesher alone keeps the terminus open.
> - **three** `case_w_014_right__*` pairs (`topcow_mr_010`, `_011_L`, `_027`) that v1's grafter rejected at the ECA re-entry gate. With the 1.6 mm ECA floor the ECA came back within the 0.35 mm band of the route at 16 mm, which left the ECA shorter than `ECA_MIN_MM` = 17 mm. Lowering `ECA_MESH_R_MM` to 1.0 mm opens that gap to +0.36 mm and removes the re-entry. The route floor plays no part: these routes are never floored.
>
> The 14 pairs still not built were rejected by the grafter (`BUILD_v2.json`: "fusing below the band"). No grafter log survives. A reconstruction from shipped data puts 13 of them at the final clearance gate, most as outright overlaps (−0.2 to −4.1 mm) with the LVA or LCCA rather than near-contact. It cannot decide the 14th (`case_w_026_left__topcow_mr_002`). v3's centerlines are identical to v2's, so only the mesh differs. Each v3 `provenance.json` also records the transforms the graft applied (`xform`), which the union uses to carry the source surfaces into place.

**Replace MESHING_PIPELINE_ANALYSIS.md:353-356 with:**

> Set B built 223 of 237 pairs against v1's 215. Of the eight gained, five are the anatomies v1 excluded as severed at the siphon terminus (now open; four also carry the siphon floor, `case_w_040_left__topcow_mr_023_L` does not). The other three are `case_w_014_right` pairs that v1 rejected at the ECA re-entry gate, which the 1.0 mm ECA floor removes. The 14 not built failed the final clearance gate, mostly as true overlaps with the LVA or LCCA (reconstructed; no grafter log).

The commit message cannot be amended. It should be corrected in the follow-up commit's message.

### C5: the five severed anatomies (CAROTID_VERIFICATION_HISTORY.md:329-330, *What remains known and open*)

**Doc:** "*(v1; all five are back in v2/v3, where the signed-distance mesher and the 1.0 mm siphon floor keep the terminus open)*"

**True:** All five are back and navigable. The floor applies to four; w_040 was readmitted without it (see C4).

**Replace with:**

> *(v1; all five are back and navigable in v2/v3. Four were regrafted with the 1.0 mm siphon floor. `case_w_040_left__topcow_mr_023_L` was not, and the signed-distance mesher alone keeps its terminus open.)*

### S1, S2, S3, S4, S5, S6, S7, S10: stenosis grade (Set B)

**Doc:**
- CAROTID_THREE_SOURCE.md:50 table: "shipped stenosis grade, max | 36 % | 56 % | 56 %".
- :66-67: "The grade is the declared grade on the centerline, after the floor, measured the way `build_manifest.py` measures it: `1 − min ICA radius / distal ICA radius`."
- :268-273: "reaches 56 %: 30 of 223 ... at or above ... 50 %, and 47 are at or above 40 %. Donor grades above 56 % (the database goes to 74 %) are still capped".
- CAROTID_VERIFICATION_HISTORY.md:326-328: "grade reaches 56 % and 30 of 223 ... Grades above 56 % are still capped".
- V2_BUILD_PLAN.md:37 and the commit message: "shipped grade up to 56 %, 30 of 223 at ≥ 50 % (measured; v1 capped at 36 %)".

**True** (v2 and v3 are identical):

| Definition | v1 max | v2/v3 max | v2/v3 at 50 % or more | v2/v3 at 40 % or more |
|---|---|---|---|---|
| build_manifest formula on the shipped ICA (own distal), D1 | 37.42 % | 60.40 % | 24 | 35 |
| shipped ICA minimum / donor distal, D3 | 36.01 % | 56.04 % | 25 | 33 |
| donor manifest on paper, `1 − max(min r, floor)/donor distal`, D2 | 36.01 % | 56.04 % | 30 | 47 |
| raw donor grade | 73.65 % | 73.65 % | 35 | 47 |

- Only D2 reproduces 30/47, and it is not a measurement on the shipped anatomy. The five extra pairs at 50 % or more are `case_w_036_left__*`. That donor's minimum is its ICA tip, which the 25 mm seam-2 ramp overwrites; the shipped grade is 3–13 %.
- At 40 % or more, 14 pairs drop out: w_036_left ×5, w_042_left ×4 and w_029_right ×5.
- **S6:** there is no uniform 56 % cap. The floor is an absolute 1.0 mm radius, so each donor's cap is `1 − 1.0/distal`: w_047_left goes from 63.8 to 44.4 %, w_052_left from 73.6 to 54.7 %. On the shipped centerlines several capped pairs exceed 56 % (k_011_left__mr_012_L 60.4 %).
- **S7:** lower_manifest.json (138 models) spans 2.53–86.34 %. 74 % is the maximum only for the 48 paired lowers, or for non-tip minima. All 8 grades above 73.65 % are ICA-tip minima.
- **S10:** "largely recovered" is fair qualitatively (v1: 0 at 50 % or more; v2/v3: 24–25). The quoted figures are inflated.
- **S1:** "36 %" matches only D2/D3. Measured on the shipped v1 route, the maximum is 37.4 %, which the older docs' "~37 %" already said.

**Evidence:** adj.py over anatomies, anatomies_v2 and anatomies_v3 (outputs in the claim table evidence); s7.py over lower_manifest; build_manifest.py:76-78; graft_three.py:544 (ramp) and :550-553 (floor).

**Replace CAROTID_THREE_SOURCE.md:50** with these two rows:

> | stenosis grade on the shipped ICA, max (own distal / donor distal) | 37 % / 36 % | 60 % / 56 % | 60 % / 56 % |
> | anatomies at ≥ 50 % (own / donor distal) | 0 | 24 / 25 | 24 / 25 |

**Replace CAROTID_THREE_SOURCE.md:66-67 with:**

> The grade is `1 − min ICA radius / distal ICA radius` on the shipped route centerline (`Centerline curve - RCCA`). The ICA window is [host_cut + cca_mm, + ica_mm] from `provenance.json`. With the distal radius taken the way `build_manifest.py` takes it (median of the last third of that ICA), v2/v3 reach 60.4 %, with 24 of 223 at ≥ 50 % and 35 at ≥ 40 %. With the donor's manifest distal radius instead, they reach 56.0 %, with 25 at ≥ 50 % and 33 at ≥ 40 %. Donor grades computed on paper from `lower_manifest.json` (30 at ≥ 50 %, 47 at ≥ 40 %) overcount, because they include ICA-tip crop-face minima that the 25 mm seam-2 ramp overwrites.

**Replace CAROTID_THREE_SOURCE.md:268-273 with:**

> **In v2/v3 this is largely recovered.** At a 1.0 mm floor the shipped ICA grade reaches 56–60 % (donor-distal / own-distal reference). 24–25 of 223 anatomies are at or above the NASCET 50 % threshold, from five lowers (`k_011_left`, `m_022_left`, `m_030_right`, `w_025_right`, `w_052_left`), and 33–35 are at or above 40 %. v1 has none at either. The floor is an absolute 1.0 mm radius, so each donor is capped at its own `1 − 1.0/distal`, not at a common 56 %: `w_052_left` falls from 73.6 % to about 55 %, and `w_047_left` from 63.8 % to 44–59 %. Among the 48 paired lowers the donor maximum is 73.6 %. The full 138-model `lower_manifest.json` reaches 86.3 %, but its eight grades above 73.6 % are all ICA-tip crop-face minima. The v3 union does not change the declared grade, because v3 centerlines are identical to v2's; it can only widen the meshed lumen.

**Replace CAROTID_VERIFICATION_HISTORY.md:326-328 with:**

> *In v2/v3, at a 1.0 mm floor, the shipped grade reaches 56–60 % and 24–25 of 223 anatomies are at or above 50 %. Each donor is still capped at its own `1 − 1.0 mm / distal radius`.*

**Replace the V2_BUILD_PLAN.md:37 tail with:**

> ...; shipped grade up to 60 % (`build_manifest` formula on the shipped ICA; 56 % against the donor's distal radius), 24–25 of 223 at ≥ 50 % (v1: 37 %, none)

### S9: "~60 % was an estimate; measured, 56 %" (commit message; V2_BUILD_PLAN.md:37)

**True:** The row replacement is real, but the framing is backwards. The ~60 % figure (BUILD_v2.json:16, MESHING_PIPELINE_ANALYSIS.md:312) matches the build_manifest formula on the shipped ICA, which gives 60.40 %. The 56 % figure is the donor-distal definition. Both are measurements.

**Replace:** use the V2_BUILD_PLAN.md:37 text above. Leave BUILD_v2.json:16 and MESHING_PIPELINE_ANALYSIS.md:312 at ~60 %, and add "(build_manifest formula on the shipped ICA)".

### M8, M9: collision_full.vtp and recut_obj.py (TOPBRAIN_PIPELINE.md:578-580, *Transporting to another machine*)

**Doc:** "Their folders also hold a `collision_full.vtp` (60 k triangles) that `recut_obj.py --tris N` uses to change the budget. It is gitignored and not needed to train."

**True:**
- There are 0 collision_full.vtp files on disk and none in git. The file exists only where a bake ran.
- On a checkout, recut_obj.py silently re-cuts 0 meshes.
- It always writes mesh_v2.json, never mesh_v3.json.
- It skips v3's navigability step-up. `case_m_006_left__topcow_mr_007_L` needed 30 k.

**Replace with:**

> The v2 and v3 bakers also write a `collision_full.vtp` (≈60 k triangles) into each folder. It is gitignored, so a checkout does not have it, and nothing loads it to train. `recut_obj.py --anatomies <dir> --tris N` re-cuts every `.obj` from those files, so it only works after a local re-bake; on a checkout it re-cuts 0 meshes. It writes `mesh_v2.json` only and skips the v3 navigability step-up, so use it on v2 folders only. Re-bake v3 with `bake_meshes_v3.py --force` instead.

### F7: "Both rebuilds keep every graft fix; what changes is the mesher plus compensating constants"

**Doc:** TOPBRAIN_PIPELINE.md:25-27 and CAROTID_THREE_SOURCE.md:38-39.

**True:**
- **Set A:** v1 data (a0cae4c) predates the RVA per-point-baseline fix (a898eb2). v2 was grafted with the fixed code, so four RVA repairs shrank: mr_002_L 4.00→2.25 mm, mr_012_L 4.50→3.00, mr_018 4.00→2.25, mr_026 4.00→1.50. The Fix C4 table (TOPBRAIN_PIPELINE.md:280-283) is v1 only.
- **Set B:** mostly true. However, FUSE_BAND_MM, itself mesher-compensating, was kept, and the siphon floor is new.

**Replace TOPBRAIN_PIPELINE.md:25-27 (from "Both rebuilds") with:**

> Both rebuilds keep every graft fix recorded below. v2 was also grafted with the RVA per-point-baseline fix (`a898eb2`), which the v1 set predates, so four RVA repairs are smaller in v2 (`mr_002_L` 4.00→2.25 mm, `mr_012_L` 4.50→3.00, `mr_018` 4.00→2.25, `mr_026` 4.00→1.50; the Fix C4 table is v1). Otherwise what changes is the mesher, the distal trim (4 mm → none), and a new 1.0 mm siphon floor.

**Replace CAROTID_THREE_SOURCE.md:38-39 (from "rebuilds keep") with:**

> rebuilds keep every graft fix recorded below; what changes is the mesher and the floors that compensated for it (route and ECA 1.6 → 1.0 mm). The 0.35 mm fusing band is kept, and a 1.0 mm siphon floor is new.

### R4: Set A v3 recipe (TOPBRAIN_PIPELINE.md:663-664, *Reproducing / v2 and v3*)

**Doc:** "For v3, graft into anatomies_v3 with the same flags -- that run writes graft_xform.json -- then bake with bake_meshes_v3.py."

**True:** graft_siphon.py writes graft_xform.json on every run since 033009a. v2 lacks it only because it was grafted earlier. The bakers skip folders that already have a report unless `--force` is given.

**Replace with:**

> # stage D + E (container). For v3, graft into anatomies_v3 with the same flags, then bake with bake_meshes_v3.py, which reads the graft_xform.json that graft_siphon.py writes on every run (shipped v2 lacks it only because it predates 033009a). Both bakers skip folders that already have a report unless --force is given.

### R8: Set B v3 recipe (CAROTID_THREE_SOURCE.md:288, *Rebuilding*)

**Doc:** "# v3: graft into anatomies_v3 with the same flags (that run records provenance.json["xform"]), then"

**True:** graft_three.py records xform on every run since 033009a. The v3 bake needs the Zenodo STLs at absolute /opt/eve_training/... paths and the TopBrain surfaces. Neither is on this machine, and run_container.sh mounts a path that does not exist here.

**Replace with:**

> # v3: graft into anatomies_v3 with the same flags (graft_three.py records provenance.json["xform"] on every run since 033009a), then bake. The bake needs the Zenodo *_lumen.stl files at the paths in xform and topbrain_data/surfaces*.

### K2: "fit passed v1 anatomies whose mesh was sealed / 0.00 mm at the worst point"

**Doc:** TOPBRAIN_PIPELINE.md:487 and CAROTID_VERIFICATION_HISTORY.md:314.

**True:**
- 0.00 mm is the host control through the v1 mesher. The worst v1 anatomy is `case_w_008_right__topcow_mr_023_L` at 0.011 mm; in Set A it is `topcow_mr_023_L` at 0.06 mm.
- The check is evaluated on RCCA points only.

**Replace CAROTID_VERIFICATION_HISTORY.md:314 (from "whose mesh was 0.00 mm") with:**

> whose meshed lumen was as low as 0.01 mm (`case_w_008_right__topcow_mr_023_L`; the host control through the v1 mesher reads 0.00 mm);

**Replace TOPBRAIN_PIPELINE.md:487 ending with:**

> ...passed v1 anatomies whose meshed lumen was effectively sealed (0.01–0.06 mm at the worst RCCA point)

### K5: "13 anatomies whose siphons were unfloored" (CAROTID_VERIFICATION_HISTORY.md:315-316)

**True:** The mechanism holds, and 22 pairs were regrafted with the floor. The count of 13 cannot be reproduced: an approximate unfloored re-bake gives 7/22. The prose also contradicts itself. BUILD_v2.json lists mr_001, a right siphon, among the 13, while 6002fcc says "every one on a necked _L siphon".

**Replace with:**

> On the first v2 bake it flagged anatomies whose TopBrain siphons had been left unfloored (`BUILD_v2.json` records 13). The 22 pairs on `mr_001`, `mr_003_L`, `mr_007_L` and `mr_010_L` were regrafted with `--siphon-min-r 1.0`.

### K6, K7: "223/223 on every check; SOFA 9/9 and 10/10 with targets past the seam" (CAROTID_VERIFICATION_HISTORY.md:295-298, *Where it landed*)

**True:**
- The static checks do pass 223/223 in both versions. However, check_anatomies.py has no genus or handle test. v3 adds handles in about 65 Set B anatomies (about 9 in Set A), including a device-passable RCCA–RVA merge in `case_w_047_left__topcow_mr_018_L` v3: a 6.8 mm chord with a clear channel of about 2.6 mm.
- SOFA ran on samples only. Each rollout inserts about 64 mm and the tip moves 39–56 mm, so no device reaches the 130 mm seam.
- run_sofa prints OK even when no target past 130 mm is found.

**Replace with:**

> Both sets hold **223** anatomies and pass `check_anatomies.py`'s static checks 223 / 223, including the meshed-lumen check below. The checker has no topology (genus) test. The v3 union adds handles in about 65 anatomies that v2 lacks, one of them a device-passable RCCA–RVA merge in `case_w_047_left__topcow_mr_018_L`. SOFA was run on samples only: 9/9 on v2 (targets 140–177 mm along the route) and 10/10 on v3 (targets 136–206 mm). Each rollout inserts about 64 mm and the tip moves 39–56 mm, so this shows the scene loads and steps. It does not show that a device reaches the 130 mm seam.

### K9, L3: Set A statistics "unchanged from v1"

**Doc:**
- TOPBRAIN_PIPELINE.md:39-40: "The rise, kink and junction statistics are unchanged from v1."
- :501 heading: "Current numbers *(v1; geometry unchanged in v2/v3)*".

**True:** validate_anatomies.py gives:
- **v1:** route 201–263 mm, rise 154–187 mm, worst bend 18–45°, min diameter 1.05–4.02 mm.
- **v2 = v3:** route 206–268 mm, rise 156–188 mm, worst bend 21–45°, min diameter 2.00–3.84 mm.
- **Unchanged:** only the junction bend (7–38°, 1/49 over the host).

**Replace TOPBRAIN_PIPELINE.md:39-40 (from "The rise,") with:**

> Against v1, every v2/v3 route is the v1 route plus 4–5 mm at the terminus (no distal trim), and six siphons are lifted by the floor. Route length is 206–268 mm, rise 156–188 mm, worst bend 21–45° and minimum diameter 2.00–3.84 mm. Junction bends (7–38°, 1/49 over the host) are unchanged.

**Replace the :501 heading with:**

> ### Current numbers *(v1; v2/v3 differ, see Current state)*

MESHING_PIPELINE_ANALYSIS.md:358-359 has the same wrong clause ("rise, kink and junction statistics are unchanged"), in context that f292143 did not edit. Replace it with "junction statistics are unchanged; route length, rise, worst bend and minimum diameter change (no trim, siphon floor)".

### N2, N4, N7: Known failures preamble (TOPBRAIN_PIPELINE.md:522-526)

**Doc:** "Each is a *label neck*: a stretch where the segmentation is one or two voxels thin (... 0.84 mm for `mr_015`, 0.73 mm for `mr_003_L`). The v1 mesher sealed it shut; the 1.0 mm siphon floor keeps it open. The train/test rule at the end of this section still applies to every version."

**True:**
- Both vessels have neck voxels. So do 18 other vessels, some of which pass v1 (mr_022_lICA, mr_008_lICA), and the JSON has no locations.
- Radii at the failure sites are 0.8–1.4 mm, not 1–2 voxels.
- A no-floor counterfactual keeps both open in v2 (enclosed 1.000, 1 component). The floor only lifts mr_015's lumen from 0.61 to 0.74 mm, which is what makes it navigable. mr_003_L changes by two radii.
- The p05 values describe the whole vessel.
- The rule is not at the end of the section; a v1-only borderline paragraph follows it.

**Replace with:**

> Both failures below pass in v2 and v3. Both vessels contain label-neck voxels (`label_necks.py`, skeleton EDT < 0.6 mm: 4 in `mr_015_rICA`, 8 in `mr_003_lICA`; whole-vessel 5th-percentile radius 0.84 and 0.73 mm). So do 18 other vessels, some of which pass v1, and `label_necks.json` records no locations, so the neck is not shown to be the failure site. The v1 blur mesher sealed or severed both. In v2 the signed-distance mesher keeps both open even without the floor. The 1.0 mm siphon floor lifts `mr_015`'s narrowest meshed lumen from 0.61 to 0.74 mm, which makes it navigable, and changes `mr_003_L` only marginally. The train/test rule in this section (keep both ICAs of a patient together) applies to every version. The borderline paragraph after it is v1 only.

### N6: v1 exclusions (TOPBRAIN_PIPELINE.md:12-13, 540, 561)

**True:**
- Under the current checker only 22/49 v1 anatomies pass. 27 fail, 25 of them on the meshed-lumen test alone.
- The v1 usage example at :561 excludes only mr_015, which contradicts :540.
- No loader enforces the exclusion.

**Replace :12-13 with:**

> **Result (v1):** 49 anatomies at `topbrain_data/anatomies/`. Two (`topcow_mr_015`, `topcow_mr_003_L`) leave the mesh and must be excluded. Under the current checker, which adds the meshed-lumen test, only 22 of 49 are navigable (see [Known failures](#known-failures-v1-only)).

Change :561 to `exclude=["topcow_mr_015", "topcow_mr_003_L"] + HELD_OUT`.

### P8: BUILD records "in each folder" (CAROTID_THREE_SOURCE.md:69)

**True:** There is one BUILD file per set-and-version folder, and none for v1. Only Set B BUILD_v2.json records outcome counts.

**Replace with:**

> ..., `BUILD_v2.json` / `BUILD_v3.json` in each set's `anatomies_v2/` and `anatomies_v3/` folder (method and flags; only Set B's `BUILD_v2.json` records build counts), and per anatomy in `mesh_v2.json` / `mesh_v3.json`.

### L6: target_min_arclength_mm (TOPBRAIN_PIPELINE.md:618)

**True:** The doc line is correct. The implied "rename" happened only in the doc: the env argument has always been `target_min_arclength_mm`, and it is forwarded to eve's `CenterlineRandom(min_arclength_from_start=...)`.

**Replace with:**

> `target_min_arclength_mm` (a constructor argument of `DualDeviceNavTopBrain`, passed to eve's `CenterlineRandom` as `min_arclength_from_start`) defaults to 40 mm

---

## 3. Where the earlier guide was wrong

No ruling marks a guide position as fully wrong. Five mark it "partly" correct.

| Ruling | Guide position | Corrected fact |
|---|---|---|
| C7 | (b) the mechanism is "not clearance vs the fusing band" | The mechanism is the 1.0 mm ECA floor removing the ECA re-entry, as the guide said. But the re-entry test is itself `(d − rr − er) < FUSE_BAND_MM`: the ECA-to-route gap moves from −0.24 to +0.36 mm across the 0.35 mm band. So "back above the fusing band" is loosely right; only "clearance" (clearance_mm, which did not move) and "floors" (plural) are wrong. |
| C8 | (c) all 14 at the final clearance gate; only w_003_right__mr_017 inside the band (+0.33) | 13 of 14 reconstruct at the final clearance gate. w_003_right__mr_017 reconstructs at −0.08 to −0.46 mm (below the band, not +0.33). w_016_right__mr_015_L is the one inside the band (0.00 to +0.18 mm after an RVA fix). w_026_left__mr_002 is indeterminate (−0.85 to +3.28 mm, depending on the donor). |
| R6 | (f) "the carotid v2 rebuild recipe omits --siphon-min-r 1.0" | The f292143 recipe in CAROTID_THREE_SOURCE.md:286 includes `--siphon-min-r 1.0`, and all five flags match BUILD_v2.json. The recipe that omits it is V2_BUILD_PLAN.md:99, unchanged by f292143. The guide's corollary ("13 non-navigable return without the floor") was not reproduced: an approximate unfloored re-bake gives 7/22 (K5). |
| S9 | (a) BUILD_v2's ~60 % originates from the manifest method on the shipped ICA | The data are consistent with that (D1 max 60.40 %), but no record proves where ~60 % came from. |
| S10 | (a) recovery overstated | The figures (30, 47) are overstated. The qualitative "largely recovered" survives: 0 → 24–25 anatomies at 50 % or more, and 30 genuine donor grades at 50 % or more, of which 24–25 ship at 50 % or more. |

Guide positions not settled by these rulings:
- **(a) "17 tip-crop donors".** Not reproduced as a count. The verifiers found 14 pairs from 3 lowers (w_036_left, w_042_left, w_029_right) that lose 40 %-or-more status, and 5 pairs (w_036_left) that lose 50 %-or-more status.
- **(k) extend_mm and ica_real_mm.** Unresolved. M4 read extend_mm = 0 and ica_real_mm = ica_mm as "no extension". The guide says the field misrecords a real extension, overstating the real ICA by 0.41–9.41 mm in 114 anatomies. Neither side re-derived the geometry.

---

## 4. Full claim table

| id | file | claim (short) | final verdict | correct statement (short) |
|---|---|---|---|---|
| C1 | CTS | v1 = 215; v2/v3 from same 237-pair plan | confirmed | 215; pairing.json 237 pairs, one commit (430bc76); all versions are subsets |
| C2 | CTS | v2 = v3 = 223, same set | confirmed | Identical names; 3791 centerline files byte-identical |
| C3 | CTS, MPA, msg | 5 of 8 gained are excluded_severed | confirmed | Exactly the 5 in excluded_severed.json; v1 minus v2 is empty |
| C4 | CTS | mesher + siphon floor keep those 5 open | partially | Floor on 4 only; w_040_left__mr_023_L unfloored, open by mesher alone |
| C5 | CVH | 5 back in v2/v3 via mesher + floor | partially | All 5 back and navigable; floor on 4, not w_040 |
| C6 | CTS, MPA, msg | 3 w_014_right pairs rejected for fusing | partially | Rejected at ECA re-entry gate (ECA 15.6 mm < 17) |
| C7 | CTS | lower floors restore their clearance above band | partially | Only the ECA floor; the ECA-route gap −0.24 → +0.36 mm, not clearance_mm |
| C8 | CTS, MPA | 14 unbuilt are fusing rejections | partially | 13 reconstruct at final clearance gate, mostly overlaps; 1 indeterminate; no log |
| C9 | CVH | back up to 223 | confirmed | 216 → 215 → 223 from git; 231/229/220 are prose only |
| C10 | CVH | built 2026-09-03 | confirmed | BUILD files and commit 6002fcc date |
| S1 | CTS, V2BP, msg | v1 max 36 %, 0 at 50 % or more | partially | 0 holds; 36.0 % only D2/D3; shipped D1 37.4 % |
| S2 | CTS, CVH, V2BP | v2/v3 max 56 % | partially | 56.0 % donor-distal only; 60.4 % build_manifest on shipped |
| S3 | CTS, CVH, V2BP, msg | 30/223 at 50 % or more | refuted | 25 (donor distal) or 24 (own distal); 30 is a paper calculation |
| S4 | CTS | 47 at 40 % or more | refuted | 33 (donor distal) or 35 (own distal) |
| S5 | CTS | grade measured on centerline per build_manifest | refuted | Formula quoted right, but figures are not from it; on shipped: 60.4 %, 24, 35 |
| S6 | CTS, CVH | grades above 56 % capped | partially | Per-donor cap `1 − 1.0/distal`, not 56 %; shipped values reach 60 % |
| S7 | CTS | database goes to 74 % | partially | Manifest 2.53–86.34 %; 73.65 % only for paired lowers or non-tip minima |
| S8 | CTS | v3 union does not change the grade | confirmed | Centerlines identical; union can only widen |
| S9 | msg, V2BP | ~60 % was estimate, 56 % measured | partially | Both measurements; ~60 % matches the doc's stated method |
| S10 | CTS, CVH | loss largely recovered in v2/v3 | partially | Qualitatively yes; quoted counts inflated |
| M1 | TBP, CTS | v1 blurred tube, 3.7 k tris | confirmed | Re-bake reproduces exactly; A 3654–3737, B 3677–3783 |
| M2 | TBP, CTS | v2 SDF 0.45 mm, 20 k | confirmed | A all 20000; B 219×20000, 4 within 2 |
| M3 | TBP | A v3 = v2 ∪ label surface | confirmed | Field max with capsule-clipped real surface |
| M4 | CTS | B v3 = v2 ∪ Zenodo + siphon | confirmed | lower/eca/siphon in all 223; one at 30 k budget |
| M5 | CTS, TBP | bakers and layout | confirmed | As stated; graft_three has no --shard (side issue) |
| M6 | TBP | A v1 59 MB | confirmed | 58.8 MB, 1.20 MB/anatomy |
| M7 | TBP | A v2/v3 96 MB | confirmed | 95.8 / 95.9 MB; growth is .obj |
| M8 | TBP | folders hold collision_full.vtp | partially | Written by bakers, but 0 on disk / in git |
| M9 | TBP | recut_obj --tris N changes budget | partially | Needs local .vtp; writes mesh_v2.json only; skips v3 step-up |
| M10 | TBP | SDF caps terminus; no trim; all enclosed | confirmed | 11,542 points enclosed in v2 and v3 |
| F1 | TBP | A trim 4/none/none | confirmed | v1 routes end 4.00–5.00 mm short |
| F2 | TBP | A siphon floor none/1.0/1.0 | confirmed | Six siphons lifted |
| F3 | CTS | B route/ECA floors 1.6 / 1.0 / 1.0 | confirmed | Provenance and ECA minima agree |
| F4 | CTS | B siphon floor none/1.0/1.0 | confirmed | Effective; only 22 v2 provenance files record it |
| F5 | CTS, CVH | FUSE_BAND 0.35 in all | confirmed | Introduced in ba3715b (shipped v1) |
| F6 | CTS | Radius floors section is v1 | confirmed | "55 floored" should be 54 |
| F7 | TBP, CTS | rebuilds keep every fix; only mesher constants change | partially | A: RVA fix a898eb2 changes 4 repairs; B: FUSE_BAND kept, siphon floor new |
| F8 | TBP, CTS | navigable definition | confirmed | lumen − 0.3 ≥ 0.35; all flags agree |
| F9 | CVH | v2/v3 replaced mesher, floors → 1.0 | confirmed | Siphon floor added, not lowered |
| R1 | TBP | label_necks.py recipe | confirmed | Run from repo root; advisory only |
| R2 | TBP | A v2 stage C flags | confirmed | Match BUILD_v2; rerun now also writes graft_xform.json |
| R3 | TBP | A v2 left ICA flags | confirmed | 24 _L mirrored |
| R4 | TBP | A v3: that run writes graft_xform.json | partially | Written on every run since 033009a |
| R5 | TBP | bake/check recipe | confirmed | --shard 0-based; --force needed to re-bake |
| R6 | CTS | B v2 graft flags incl. --siphon-min-r | confirmed | Matches BUILD_v2; V2_BUILD_PLAN:99 omits it |
| R7 | CTS | B v2 bake recipe | confirmed | Valid |
| R8 | CTS | B v3: that run records xform | partially | Recorded on every run; bake needs absent STLs |
| R9 | CTS | B v3 check recipe | confirmed | No SOFA by default; no v2 check line |
| R10 | TBP | v3 needs raw surfaces | confirmed | Baker needs them; grafter does not |
| K1 | TBP | route connectivity check | confirmed | Added a898eb2; "~100 %" holds at 0.6 mm neck |
| K2 | TBP, CVH | meshed-lumen check; v1 sealed / 0.00 mm | partially | Rule right; worst v1 is 0.011 mm; 0.00 is host control |
| K3 | TBP | A navigable 22/49/49 | confirmed | As stated |
| K4 | CTS, CVH | B navigable 71/223/223 | confirmed | As stated; CTS:3 "215 navigable" contradicts |
| K5 | CVH | 13 unfloored caught, added --siphon-min-r | partially | Mechanism right; 13 unreproducible (≈7/22) |
| K6 | CVH | 223/223 on every check | partially | Static yes; no genus test; v3 handles incl. RCCA–RVA merge |
| K7 | CVH | SOFA 9/9, 10/10, targets past seam | partially | Counts right; tips never reach seam |
| K8 | TBP | A v2/v3 need no exclusions | confirmed | 49/49, single component |
| K9 | TBP | Current numbers v1, unchanged in v2/v3 | refuted | v2/v3 route 206–268, rise 156–188, bend 21–45, min diam 2.00–3.84 |
| N1 | TBP | mr_015, mr_003_L pass v2/v3 | confirmed | Lumen 0.74/0.90 (v2), 0.78/1.05 (v3) |
| N2 | TBP | each failure is a label neck 1–2 voxels | partially | Necks present but not specific or located; failure radii 0.8–1.4 mm |
| N3 | TBP | p05 0.84 / 0.73 mm | confirmed | Whole-vessel statistic; does not discriminate |
| N4 | TBP | v1 sealed; floor keeps it open | partially | Mesher keeps it open; floor only makes mr_015 navigable |
| N5 | TBP | six siphons lifted | confirmed | 001, 003_L, 007_L, 010_L, 013, 015 |
| N6 | TBP | mr_015 excluded from v1 only | partially | Holds; not enforced in code; :561 omits mr_003_L; 27/49 fail v1 |
| N7 | TBP | train/test rule applies to every version | partially | Applies; not at section end; not enforced |
| P1 | CTS | siphon source is TopBrain | confirmed | 223 kind "topbrain" |
| P2 | CTS | topcow_mr_NNN because labels drawn on TopCoW MR | unverifiable | Names are mask basenames; dataset fact absent |
| P3 | CTS | TopCoW labels not used | confirmed | No builder reads cow_seg_labelsTr |
| P4 | CTS | TopCoW ICA stops about 18 mm | unverifiable | Prose only (18.4 vs 67.1 mm) |
| P5 | CTS | v3 provenance xform | confirmed | 223/223; 0 in v1/v2 |
| P6 | TBP | A v3 graft_xform.json | confirmed | 49/49; also origin and mirror |
| P7 | TBP | title 49 meshes | confirmed | 25 R + 24 L in each version |
| P8 | CTS | BUILD_v*.json in each folder | partially | One per set-and-version folder; only B v2 has outcome counts |
| L1 | TBP | A v3 centerlines = v2 | confirmed | 784/784 byte-identical |
| L2 | CTS | B v3 centerlines = v2 | confirmed | 3791/3791 byte-identical |
| L3 | TBP | rise/kink/junction unchanged from v1 | refuted | Only junction unchanged |
| L4 | TBP | v2/v3 need no exclude | confirmed | 49/49 static |
| L5 | TBP | loader ignores BUILD_v*.json | confirmed | 50 entries, 49 loaded |
| L6 | TBP | target_min_arclength_mm, renamed | partially | Argument correct; no code rename occurred |

File key: CTS = CAROTID_THREE_SOURCE.md, CVH = CAROTID_VERIFICATION_HISTORY.md, TBP = TOPBRAIN_PIPELINE.md, MPA = MESHING_PIPELINE_ANALYSIS.md, V2BP = V2_BUILD_PLAN.md, msg = commit message.

---

## 5. Cross-check of the other machine's pre-commit summary

Source labels:
- **[V: id]** verified ruling above
- **[G: letter]** guide position
- **[Q]** a quick check I ran for this report
- **[NC]** not covered by any source here

### Sources

| Claim | Status | Correct value / note |
|---|---|---|
| Host is the shipped eve_bench/data/dualdevicenav patient; supplies arch, trunk, other branches, RCCA from the ostium | not covered [NC] | Consistent with the task context; not checked |
| TopBrain 2025: 25 MR label masks, 50 ICAs | agrees [V: P7, R1] | 25 patients; label_necks.json has 50 vessels |
| Label 4 = R-ICA, label 6 = L-ICA; "8 is a different vessel" | agrees [V: R1, P3] | Code uses 4 and 6. The mask_to_surface.py docstring still says "label 8" (stale docstring) |
| mr_006_L is a 19.5 mm fragment, rejected, leaving 49 | agrees; 19.5 mm unverifiable [V: R1] | Excluded by graft_siphon's MIN_SIPHON_MM = 60 skip; also the only label_necks reject (longest run 3.01 mm) |
| TopCoW rejected: ICA 18 mm vs 67 mm | unverifiable [V: P4] | Doc prose only (18.4 vs 67.1 mm); no data or script here |
| remote_zip.py pulled 47 MB of 10.5 GB | partly covered [V: P3] | remote_zip.py fetched only the label masks (250); size not checked |
| Zenodo database: v1 62 cases, v2 79 | not covered [NC] | Raw database absent |
| 138 carotids with lumen STL + VMTK centerlines | agrees [V: S7] | lower_manifest.json has 138 models |
| One tree won't split (case_w_030_right) | agrees [Q] | lower_manifest.json `unparsed`: "case_w_030_right_lumen_centerlines.vtp", "tree would not split" |
| Donors are stenosis patients, 3–74 % | wrong as a database statement [V: S7; G: g] | Manifest spans 2.53–86.34 % (median 35.85). 73.65 % is the maximum only for the 48 paired lowers or non-tip minima |

### Set A

| Claim | Status | Correct value / note |
|---|---|---|
| mask_to_surface → vmtk_centerline → graft_siphon | not covered [NC] | Not re-run; raw inputs absent |
| Cut at about 130 mm | agrees [V: R3, P6] | route_from_mm = 130.0 in all 49 v3 graft_xform.json; "237.5 − 106" not checked |
| Frame match, host radius ramp, anchor_trim fixes mr_021, cranial stubs dropped | not covered [NC] | — |
| RVA collisions repaired per anatomy (4 right, 2 left) | count not covered; stale for v2 [V: F7] | v2 was grafted with the a898eb2 per-point-baseline fix, so mr_002_L, mr_012_L, mr_018 and mr_026 have smaller deflections than the v1 table |
| Left ICAs mirrored | agrees [V: R3, P6] | 24 _L with mirror [-1,1,1] |
| Torsion +1.95 / −2.00 | not covered [NC] | — |
| Splits keep both sides of a patient together | agrees [V: N7] | 24 patients contribute both ICAs in every version; not enforced in code |

### Set B

| Claim | Status | Correct value / note |
|---|---|---|
| host 0 → host_cut (15–54 mm), donor, siphon; seam 2 pinned at 130 mm | seam agrees [V: K7]; host_cut range not covered | Seam at 129.5–131.1 mm |
| Pipeline order analyze_bifurcations → build_manifest → extend_ica → match_sections → graft_three | not covered [NC] | Consistent with the CTS *Rebuilding* block |
| 34 short ICAs get 10 mm of extension | count consistent (extended/ holds 34); extension disputed [V: M4; G: k] | Provenance records extend_mm = 0 in all 223 v3 anatomies. The guide says ica_real_mm overstates the real ICA by 0.41–9.41 mm for 114. Unresolved |
| Usage caps: lowers at most 5×, siphons at most 8× | consistent [V: S3, F4] | 5 pairs per lower seen; mr_001 and mr_010_L used 8× each. Caps in code not checked |
| 73 donors too short to reach the seam | not covered [NC] | — |
| Bifurcation donors not mirrored; 3 of 8 same-patient pairs have opposite handedness | not covered [NC] | — |

### Bug classes

| Claim | Status | Correct value / note |
|---|---|---|
| Thresholds at 0 where resolution is 0.35 mm led to FUSE_BAND_MM | agrees [V: F5] | ba3715b changed `< 0.0` to `< FUSE_BAND_MM` in four places |
| Unconstrained DOF bugs (tangent-only roll, anchor_trim never called, chord as up axis, clearance from 130 mm) | not covered [NC] | Documented as CTS §1–4; not re-verified |

### Meshing

| Claim | Status | Correct value / note |
|---|---|---|
| v1 blurred binary tubes; vessels sealed shut | agrees [V: M1, K2] | v1 meshed lumen as low as 0.011 mm; 144/215 (B) and 27/49 (A) fail the lumen test |
| Constant floors compensated | mostly agrees [V: F7] | FUSE_BAND (also compensating) was kept; the siphon floor is new, not compensating |
| v2 SDF at 0.45 mm, 20 k triangles | agrees [V: M2] | Set B: 219 at 20000, 4 at 19998–19999 |
| Floors drop to 1.0 mm; siphons get a floor | agrees [V: F3, F4] | Only 22 v2 provenance files record the siphon floor; equivalent in result |
| A 49/49 and B 223/223 navigable, up from 22 and 71 | agrees [V: K3, K4] | — |
| v3: same centerlines, unioned with real surfaces using recorded transforms | agrees with caveats [V: L1, L2, M3, M4, P5, P6, K6] | v3 adds topological handles (about 65 B, 9 A), including a device-passable RCCA–RVA merge in `case_w_047_left__topcow_mr_018_L`. One B anatomy was baked at a 30 k budget |

### Stale docs

| Claim | Status | Correct value / note |
|---|---|---|
| TOPBRAIN_PIPELINE.md describes v1 only | stale [V: K3, K9, N6] | f292143 added *Current state (v2 / v3)*. But "47 are usable" (TBP:12) is wrong even for v1 under the current checker (22/49), and "geometry unchanged in v2/v3" is wrong |
| DISTAL_TRIM 4 mm | correct for v1 [V: F1] | v2/v3 have none |
| "No --topbrain flag" | not covered [NC] | — |
| Carotid docs say 215 | stale [V: C2, K4] | Now documented as 223 in v2/v3. CTS:3 still calls the v1 set "215 navigable anatomies"; 71/215 are navigable |
| v2/v3 have 223 because the lower ECA floor removed eight fusing rejections | wrong [V: C3, C4, C6] | 5 are v1 post-bake severed exclusions (SDF mesher, plus the siphon floor on 4). 3 are case_w_014_right pairs rejected at the ECA re-entry gate, fixed by the ECA floor |
| 224 entries = 223 + BUILD json | agrees [Q] | `ls carotid_data/anatomies_v2 \| wc -l` gives 224; same for v3 |
| 50 = 49 + BUILD json | agrees [V: L5] | — |
| "No clinically significant stenosis" is v1 | agrees [V: S1, S10] | v1 has 0 at 50 % or more under every definition |
| At the 1.0 mm floor stenosis restored to about 60 % | agrees under the build_manifest definition [V: S2, S9] | 60.40 % on the shipped ICA with its own distal; 56.04 % with the donor distal. 24–25 at 50 % or more |
| COSTA is documented nowhere | agrees [Q] | A case-insensitive search of *.md and *.py finds no mention |

---

## 6. Recommended follow-ups (prioritised)

**P1: correctness of numbers people will cite**

1. **Fix the stenosis figures everywhere they appear:** CTS:50, 66-67, 268-273; CVH:326-328; V2_BUILD_PLAN.md:37. Use the replacement text in §2. Pick one named definition and state the other alongside it. Commit the measuring script (adj.py logic: shipped RCCA radii over the provenance ICA window, own distal and donor distal) so the numbers can be regenerated. Keep BUILD_v2.json:16 and MESHING_PIPELINE_ANALYSIS.md:312 at ~60 %, with the definition named. The next commit message should say that f292143's "estimated ~60 % → measured 56 %, 30 at 50 % or more" was a definitional mix-up.
2. **Correct "geometry unchanged in v2/v3" for Set A:** TBP:39-40, the TBP:501 heading, and MPA:358-359. Add the v2/v3 validate_anatomies numbers.
3. **Add a topology (genus/handle) test to check_anatomies.py.** Investigate the v3 RCCA–RVA merge in `case_w_047_left__topcow_mr_018_L`, a device-passable channel of about 2.6 mm, before training on v3 Set B. Then list which of the about 65 B / 9 A v3 handles touch the route. Until then, qualify "223/223 on every check" (CVH:295-296).

**P2: causal statements and evidence strength**

4. Replace the gained-eight / unbuilt-14 explanation in CTS:52-62, MPA:353-356 and CVH:329-330 (§2).
5. **Make the SOFA check meaningful.**
   - Have run_sofa return FAIL (not OK with a note) when no target past 130 mm is drawn.
   - Add a mode that inserts past the seam: more steps, or report tip arc length against the seam.
   - Restate CVH:296-297, V2_BUILD_PLAN.md:102, and MPA:368 and 459 accordingly.
6. **Reconcile v1 usage guidance in TBP.**
   - :12-13 says "47 usable"; the checker gives 22/49.
   - :540 and :561 disagree about excluding mr_003_L.
   - Mark the RVA Fix C4 table (:280-283) as v1 and add the v2 deflections.
   - Correct the "graft fix" sentence (:25-27) and CTS:38-39.
   - Fix CTS:3, which says "215 navigable".
7. **Rewrite the Known failures preamble (TBP:522-526).** Necks are present but not shown to be the failure site. The SDF mesher, not the floor, keeps both open.

**P3: tooling and records**

8. **Persist grafter rejection reasons.** Record pair, gate and value in BUILD_v*.json or a committed log, so that C6/C8-type claims become checkable without reconstruction. Separate "excluded after bake" from "rejected by grafter": BUILD_v2's "v1: 22 rejected" merges them.
9. **Fix recut_obj.py.**
   - Refuse on v3 folders, or write mesh_v3.json and apply the navigability step-up.
   - Warn when no collision_full.vtp is found.
   - Update TBP:578-580, V2_BUILD_PLAN.md:61-62 and MPA:384-385, which say the file is "kept".
10. **Recipe fixes.**
    - Remove "graft_three.py ... take --shard i/n" (CTS:292-293); graft_three shards with `--only LO:HI`.
    - Add `--siphon-min-r 1.0` to V2_BUILD_PLAN.md:99.
    - Note `--force` for re-bakes.
    - Fix carotid_tools/run_container.sh, which mounts a worktree path that does not exist here.
11. **Provenance consistency.**
    - Backfill siphon_floor_mm into the 201 v2 provenance files, or note why it is absent.
    - Resolve guide (k) by re-deriving the real ICA length against ica_real_mm and extend_mm for the 114 flagged anatomies.
12. **Minor text fixes.**
    - K2 "0.00 mm" wording (CVH:314, TBP:487).
    - K5's 13 count vs. 6002fcc's "every one on a necked _L siphon" (CVH:315-316).
    - F6 "55 anatomies floored" should be 54 (CTS:189).
    - CVH:130 "stenosis_pct 3–74%" should be 2.5–86 % for the full manifest.
    - The mask_to_surface.py docstring says label 8; the code uses 4.
    - P8 "in each folder" (CTS:69).
    - L6: clarify env argument vs. eve keyword (TBP:618).

**P4: nice to have**

13. Add outcome counts (navigable, lumen range, budget exceptions such as the 30 k anatomy) to both BUILD_v3.json files.
14. Record the source statistic for P4's TopCoW extent (mean or median, over which patients) with a script. Record the TopBrain-on-TopCoW-scans provenance (P2) with a citation.