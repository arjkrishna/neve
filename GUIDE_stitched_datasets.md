# The stitched anatomy datasets on `rl_improv_16_resume`: builders, versions and consumers

**Scope and how to read this guide**

- **What it covers.** This guide covers the two families of stitched anatomies built on branch `rl_improv_16_resume` (HEAD `1597777`), the code that builds them, and the code on `origin/rl_improv_18_p2` that consumes them.
- **Sources.** Every number comes from a verified agent report (A1–A9, G1–G10) or from the reconciler's resolutions. Where the reconciler resolved a contradiction, this guide uses its resolution.
- **Versions.** Each claim is tagged with the version it applies to:
  - v1: blur mesher.
  - v2: SDF tube mesher.
  - v3: SDF tubes unioned with the patients' real surfaces.
- **Labels.** Items that were not verified are marked **UNVERIFIED**. Inferences are marked **(inference)**.
- **Revision (after `f292143`).** Local HEAD is now `f292143` (2026‑09‑24), which adds a v2/v3 "Current state" to the build‑record docs. A second, independent verification (21 agents) checked its 79 claims: 46 confirmed, 26 partly right, 5 refuted, 2 unverifiable. Its rulings are folded in below and summarised in §13; the full report is `VERIFY_f292143_report.md` beside this file. The data trees did not change.
- **Since then.** The report's §2 replacement texts have been applied to the five build-record docs (TOPBRAIN_PIPELINE, CAROTID_THREE_SOURCE, CAROTID_VERIFICATION_HISTORY, MESHING_PIPELINE_ANALYSIS, V2_BUILD_PLAN) and to Set B's `BUILD_v2.json`, so the quotes below describe `f292143` as committed, not the docs as they now read. A **v4** Set B (`carotid_data/anatomies_v4/`, the whole neck grafted, merge-free) was added after this guide; see `CAROTID_V4.md`.

---

## 1. Big picture

**The host patient.** There is one host patient: `eve_bench/data/dualdevicenav`. It is the stEVE_bench DualDeviceNav composite, a CTA aorta plus an MRA of the cerebral vessels, 25 centerline branches. Its right common carotid route (branch "RCCA", 237.50 mm) is a single continuous CCA→ICA branch with no bifurcation. The RL task in this repo is navigating that route, and the procedural generator (`RCCAVariedFromMesh`) only perturbs this one patient. ANATOMY_GENERATION_RATIONALE.md §3 on 18_p2 lists the resulting limitations:
- the generator samples "a neighbourhood of one patient, not a population";
- its bends have ~100 mm radii, against the 4–5 mm cavernous genua;
- so the siphon is left "essentially as it was".

**Why the two sets exist.** No public dataset covers the full route from CCA ostium to siphon. The one VMR candidate, 0248_H_AOCERE_CAS, is a single patient with 30 synthetic stenosis variants, and it was kept as a possible held-out case. Both sets therefore graft real vessel segments onto the host:

| | **Set A — "TopBrain grafts"** (`topbrain_data/`) | **Set B — "three-source carotid composites"** (`carotid_data/`) |
|---|---|---|
| Host kept | arch, trunk, BCT (11), RCCA 0→~130 mm, RVA (sometimes repaired), LCCA, LVA, 12 unnamed branches | same, but the RCCA only 0→`host_cut` (15.0–59.3 mm) |
| Donor 1 | real TopBrain/TopCoW ICA **siphon**, from ~130 mm to the terminus | real **CCA + ICA/ECA bifurcation + cervical ICA** from the CarotidAnalyzer Zenodo DB ("lower") |
| Donor 2 | — | real TopBrain **siphon**, pinned at 130 mm composed arclength |
| New branch | none (16 centerline files) | `Centerline curve - RECA.mrk.json`, the ECA as a wrong-turn decoy (17 files) |
| Count (v1/v2/v3) | 49 / 49 / 49 | 215 / 223 / 223 |
| Identity | `topcow_mr_NNN[_L]` | `<case_[kmw]_NNN[b]_(left|right)>__topcow_mr_NNN[_L]` |

**How the sets relate to each other.**
- Both draw their siphons from the same 25 TopBrain MR patients. Left ICAs are x‑mirrored and carry the suffix `_L`.
- Both pin the seam into the donor siphon at a nominal 130 mm of RCCA arclength. For Set B this is deliberate: seam 2 is placed at 130 mm to match Set A's graft point.
- Set B's pairing plan is frozen, and it permanently excludes right‑side siphons mr_013/014/015/025 because of v1 mesh defects. Set A v2/v3 *does* include those siphons (§10).
- For v2 and v3, the centerlines are byte‑identical within each set. Only the collision mesh and the build records differ. v1 centerlines differ from v2/v3 (§5).

**Where each branch stands.**
- **Branch 16** holds the builders, the data, the checks, and the v2/v3 grafter flags.
- **18_p2** holds identical copies of the data trees (synced in `a16750f`) and all the training/eval wiring: the `--topbrain` flag family and `eval_anatomies.py`. Branch 16's trainer has no `--topbrain` flag.
- 18_p2's `graft_three.py` is the older ba3715b version, so 18_p2 cannot regenerate Set B v2/v3 (§12).

---

## 2. Source datasets and provenance

### 2.1 TopBrain (the siphons in Sets A and B)
- **Release used.** Zenodo record **16878417**. This is v2 of concept 16623495 (concept DOI 10.5281/zenodo.16623495), published 2025‑08‑14.
  - File: `TopBrain_Data_Release_Batches1n2_081425.zip`, 1,958,849,592 B, MD5 `b703ea31…`.
  - 18_p2 SESSION_TRANSCRIPT.md:11239 records "Downloaded … Zenodo 16878417, MD5‑verified".
  - TOPBRAIN_PIPELINE.md:41 calls it "TopBrain_batches1n2.zip", which is imprecise. (G9 A7‑12)
- **Data used.** Only `labelsTr_topbrain_mr/`: 25 masks, `topcow_mr_{001..027}` minus 009 and 019, 8.68 MB, voxels 0.297×0.297×0.6 mm. The skipped IDs are missing from the release itself.
  - The ITK‑SNAP labelmap (42 classes) gives **4 = R‑ICA, 6 = L‑ICA**, 8 = R‑Pcom, 35/36 = R/L‑ECA. There is no CCA class.
  - The code uses `ICA_LABEL=4` (`mask_to_surface.py:33`) and `--label 6` for the left side. The earlier value 8 was the "Fix A1" bug. The docstring at lines 13–15 is still stale.
  - CT labels are unused.
- **Naming.** The name `topcow_mr_NNN` comes from the TopBrain file names. TopBrain re‑annotated the TopCoW images and kept TopCoW's naming.
- **Newer release trap.** The newer TopBrain v3 release (record **21972006**, 2026‑08‑17) has `labelsTr_topbrain_v1_mr`, byte‑identical to what was used (25/25). Its new `topaneu36class` masks redefine **4 = R‑ICA‑C6‑C7 only**. Re‑running `mask_to_surface.py` on those masks would silently extract a stub.
- **Licence** (License.txt in the zip): "Open use. Must provide the source. Use for commercial purposes requires permission of the data owner". The data owner is University Hospital Zurich. HANDOFF §9.1 on 18_p2 calls this "non‑commercial", which is imprecise.
- **Citations:** Yang, Shi et al., TopBrain challenge (medRxiv 2026), and Yang, Musio et al., TopCoW, NEJM AI 2026 (arXiv 2312.17670). **UNVERIFIED** in detail.

### 2.2 TopCoW (ruled out; no shipped geometry)
- Zenodo **15692630**, `TopCoW2024_Data_Release.zip`, 10,512,833,650 B.
- `topbrain_tools/remote_zip.py` reads the ZIP central directory through HTTP range requests: 1 MB blocks, 64‑block cache, `--max-mb 500`. Indexing the whole archive cost 1 request and 0.86 MB.
- It pulled the 250 `cow_seg_labelsTr` masks: 45.8 MB measured, against the doc's "47 MB".
- **Why it was ruled out** (doc‑only; no script committed): on the 25 shared patients, on an identical 508×585×189 grid, the R‑ICA spans 18.4 mm in TopCoW against 67.1 mm in TopBrain (TOPBRAIN_PIPELINE.md:340‑358; a0cae4c).

### 2.3 CarotidAnalyzer bifurcation DB (Set B "lowers")
- Zenodo **10695923**, **version 2.0.0**, published 2024‑02‑23. DOI 10.5281/zenodo.10695923; concept DOI 10.5281/zenodo.7634643. Licence **CC‑BY‑4.0**.
  - Authors: Eulzer, Richter, Probst, Hundertmark, Lawonn (FSU Jena and RPTU).
  - Related papers: CGF 42(3):25‑37 (2023), CMBBE 27(3):347‑364 (2024), CGF 43(3) (2024).
- **File used:** `carotid_bifurcation_database.zip`, 598,156,982 B, MD5 `9755c1b7…`. It contains:
  - 79 case folders (k=13, m=17, w=49);
  - 152 `*_lumen.stl`, of which only 142 are unique (see below);
  - 87 plaque STLs and 139 `*_lumen_centerlines.vtp`;
  - no README.
- **Duplicate trap.** `case_m_030_right_lumen.stl` sits in 10 other m‑folders, and `case_m_026` has no right lumen of its own. The build globs `*_centerlines.vtp` (`build_manifest.py:92`) and derives each STL path from it (`graft_three.py:781`), so this does not affect any set.
- **Version used.** The case_k/m/w names exist only in v2.0.0. v1.0.0 (record 7634644) uses case01..62, and the container path is `/opt/eve_training/carotid_data/v2/carotid_bifurcation_database/...`. **That "v2" is the DB release, not mesher v2.** The manifest was committed 09‑01, before the SDF mesher (09‑03).
- **k/m/w meaning.** Undocumented anywhere reachable. A CRC mapping (4 cases unmatched) gives v1 case01–13 → k, v1 case14–62 → w, and m = new in v2. Each prefix numbers independently with gaps, so they are probably source or batch cohorts (inference).
  - `case_w_018b` is v1 case29 (left only). It is geometrically unlike w_018_left (stenosis proxy 5.2% vs 67.3%). Probably a second scan (inference).
- All lowers are CTA‑derived. "Bifurcations with 100% stenosis were omitted" (record description).
- **Not on disk here.** The raw DB is not present in this checkout (`carotid_data/` holds only derived files), and `run_container.sh` mounts a nonexistent `D:/neve/.claude/worktrees/rl_improv_16_resume` path.

### 2.4 The host patient
- `eve_bench/data/dualdevicenav` is **unmodified upstream stEVE_bench data**. The local `.obj` blob SHAs (hashed with `--no-filters`) match upstream, and the `Centrelines_comb` tree hash `125327a` is identical.
  - Upstream history: added in `5b8a526` (2024‑04‑05), updated in `46660a0` (2024‑06‑24). Locally it arrives only in `905b58a` "initial baseline".
  - Licence: eve_bench MIT, © 2023 Lennart Karstensen.
- **Origin** (Karstensen et al., *Computers in Biology and Medicine* 196:110844, 2025; arXiv 2410.01956): "manually segmented from a computed tomography angiographic scan of the aorta and a magnetic resonance angiography scan of the cerebral vessels … manually assembled into a cohesive model". No dataset ID, institution or ethics statement is given.
- **It is not VMR "0105".** VMR legacy 0105_0001 is `0017_H_AO_COA` (aorta only), and the current 0105 is pulmonary. The mislabel comes from `vmr_download_tools/README.md:163` ("Likely base") and repeats in COORDINATE_SYSTEM_CONSISTENCY.md, CUSTOM_MODEL_SCRIPTS_SUMMARY.md, the `DualDeviceNavCustom` docstring (`dualdevicenav.py:244`) and training comments.
- **Contents** (see the table below):

| Item | Value |
|---|---|
| Centerlines | 25 Slicer `.mrk.json` (schema v1.0.3, tagged LPS, mm), per‑point `Radius` in `controlPointValues`; measurement `value` fields null |
| Loader frame | JSON (x,y,z) → vessel (y,−z,−x) (`dualdevicenav.py:191`); "superior" runs along −x in the JSON |
| Collision OBJ | Blender 3.4.0, object `ARCH2COW.001`, 1,786 v / 3,254 vn / 3,583 f = **3,582 triangles + 1 quad** (line 8481), 268,132 B |
| Visual OBJ | 22,437 v / 44,877 f |
| Unused mesh | `mechanical_thrombectomy_collision_FIX_REMESH.obj` (Blender 3.1.0, 7,411 v); nothing references it |
| RCCA | 235 points, 237.50 mm raw; 1‑mm resample 239 pts / 235.90 mm; ostium r 5.8121 mm; r(10)=2.491, r(100)=2.228, r(130)=1.431; raw r over 10–100 mm: 2.00–2.79, over 105–190 mm: 1.22–1.97 (narrowing), r(200)=1.68, r(237)=2.13 |
| Topology | (0) ascending aorta 57.9 mm; (2) descending aorta 373.1 mm; (11) BCT 48.1 mm and LCCA leave (0) at s=27.1; RCCA and RVA start at the end of (11); (17) leaves the RVA at 26.8; LVA starts at the end of (18); **RVA and LVA both end at the same point [14.8,12.5,572.0]** (vertebrobasilar confluence: gap −3.1122 mm, acknowledged at `graft_siphon.py:384‑387`); stubs (13)/(24) start at the RCCA terminus |
| Cranial stubs | (9),(10),(13),(15),(21),(22),(23),(24),(31): unnamed, min z ≥ 572 |

**How the host mesh is loaded (G8).**
- **Raw file.** Read directly, SOFA's `MeshObjLoader` sees 1,786 positions and does not split corners by normal. VTK splits the same file into 10,750 points and 10,750 boundary edges; after `clean()` there are 1,786 points and 0 boundary edges. MESHING_PIPELINE_ANALYSIS.md:32's "unwelded, 10,750 points" describes the VTK read.
- **What the env actually loads.** `FromMesh` reads the file with VTK, rotates it `[90,−90,0]` and re‑saves it through meshio to `/tmp/Test_*.obj` with 10,750 vertices. **SOFA collides with that triangle soup**: 10,746 open edges, and the quad is dropped, leaving a hole with 10.45/5.58 mm diagonals about 7 mm from the RCCA centerline at arc 87 mm.
- **Cost.** The soup costs +12–26% per step against the welded mesh.
- **Host tests use the soup.** 18_p2's host test (`eval_anatomies.py:548,562`) pins this soup.

### 2.5 VMR search (closed)
- **Catalogue.** The local catalogue lists 316 names (288 unique IDs). AOCERE appears exactly once, as `0248_H_AOCERE_CAS` (arch → CoW, CT, one patient, 30 synthetic NASCET 10–100% variants). IDs 0249–0288 are coronary.
- **Commit `19852bf`** closes the search: 316 vs 269 on the live site, and 531 name probes. The 269 and 531 figures are **UNVERIFIED** (no script).
- **0248 figure.** `figure_vmr_0248.py` rotates the raw OBJ by `[90,−90,0]`. The converter's `[−90,0,90]` is wrong for reading, which is the frame trap. `vmr_data/` is gitignored and absent.

### 2.6 Attribution gap (licence consequence)
- `arjkrishna/neve` is **public**. It tracks 2,649 files under `topbrain_data` and 13,044 under `carotid_data`, and v3 meshes embed donor real surfaces directly.
- On branch 16, no committed file names any dataset DOI or record ID (git grep: 0 hits). There is no NOTICE or data README.
- CC‑BY‑4.0 requires attribution, a licence link and a note of changes. The TopBrain/TopCoW terms require naming the source.

---

## 3. Set A — TopBrain grafts (pipeline)

The scripts are in `topbrain_tools/`. Intermediates are gitignored and absent on this machine: `topbrain_data/{surfaces,surfaces_left,centerlines,centerlines_left}`, the raw release, and `collision_full.vtp`.

### Stage 0 — neck screen (added 09‑03 for v2; branch 16 only)
- **Script:** `label_necks.py`, with `--thin 0.6` and `--reject-run 3.0` (lines 44‑45).
- **Method:** keep the largest 26‑connected component, compute the EDT, skeletonize, and flag skeleton voxels with EDT < 0.6 mm. A "run" is the bounding‑box diagonal of a connected thin run.
- **Output:** `topbrain_data/label_necks.json`, 50 entries (25 rICA + 25 lICA).
  - 20 entries have necks. There is 1 reject: `topcow_mr_006_lICA` (21 necks, longest 3.01 mm).
  - mr_015_rICA has 4 necks (longest 0.94) and is not rejected.
- **No build script reads this file.** Only `reports/make/figures.py` does. mr_006_L's absence comes from `MIN_SIPHON_MM=60`: it is a 19.5 mm fragment, and that 19.5 mm figure is from the docs only.
- `label_qa.txt`: every one of the 50 labels is one component with 0 holes, and MISR − EDT has median +0.00..+0.03 mm.
- **Necks are not shown to be the v1 failure site** (f292143 verification, N2–N4). mr_015 and mr_003_L do contain neck voxels (4 and 8; whole‑vessel 5th‑percentile radius 0.84 / 0.73 mm), but so do 18 other vessels, some of which pass v1, and `label_necks.json` records no locations. Radii at the v1 failure sites are 0.8–1.4 mm, not "one or two voxels". In a no‑floor counterfactual the v2 SDF mesher keeps **both** open (enclosed 1.000, one component); the 1.0 mm siphon floor only lifts mr_015's narrowest meshed lumen from 0.61 to 0.74 mm, which is what makes it navigable.

### Stage A — mask → surface (`mask_to_surface.py`, base env)
- Keep label 4 (or 6 with `--label 6 --suffix lICA`), take the largest 26‑connected component, and run skimage marching cubes at level 0.5 in world RAS mm via the NIfTI affine.
- Then `vtkCleanPolyData` and `vtkWindowedSincPolyDataFilter` (20 iterations, passband 0.1).
- Seeds come from a double BFS over the skeleton, ordered inferior→superior.
- Output: `surfaces[_left]/<stem>_[rl]ICA.vtp` plus seeds. This `.vtp` is also the v3 "real surface" for the siphon.

### Stage B — surface → centerline (`vmtk_centerline.py`, vmtk_env)
- Runs `vmtkCenterlines` with pointlist seeds and `AppendEndPoints=1`, and keeps `MaximumInscribedSphereRadius`.
- It walks the **longest cell by point ids** (Fix B1; the earlier zig‑zag bug gave tortuosity 3.7), then flips the polyline so z increases.
- Output: `centerlines[_left]/<stem>_ica.json`. The side suffix is dropped from the file name.
- Doc numbers, not re‑verifiable: right‑ICA lengths 80–144 mm (median 106), left 76–139.

### Stage C — graft (`graft_siphon.py`)

| Step | Detail (constants at `graft_siphon.py:63‑83`) |
|---|---|
| Host prep | Resample the host RCCA at 1 mm (239 pts, 235.90 mm) |
| Cut | `GRAFT_MM=130` = 237.5 − 106 (median donor) ≈ 131.5, rounded; lands on resampled sample **k=131, s=130.146**; anchor in branch frame **(−9.380, 11.828, 530.306)**, identical in all 49 `graft_xform.json` |
| Donor prep | (1) mirror if left: `sp*[-1,1,1]` in RAS, **before** anything else (:485‑499); (2) `prep_siphon`: dedup, 1‑mm resample, trim end kinks >40° up to 10 mm/end (Fix B2); (3) reject if <60 mm (`MIN_SIPHON_MM`); (4) `anchor_trim`: advance the start until its tangent is ≤70° from the chord to the end, max 30 mm, needs >30 pts left (fires on mr_021 idx 4 and mr_024_L idx 2); (5) optional radius floor `ROUTE_MIN_R` (0 in v1; **1.0 in v2/v3**; in Set A this flag floors the **siphon**) |
| Radius ramp | Host radius replaced over 100→130 mm by a smoothstep from r_host(100)=2.230 to the donor's first (floored) MISR. Junction radii b: right 1.43–2.91, left 1.70–3.09; b<a (ramps **down**) in 7/25 right and 9/24 left |
| Placement | Frame match, not just tangent: rows `[t, unit(UP−t(UP·t)), t×n]`, UP=+z, `R = frame(t_host)^T @ frame(t_siph)`, tangent span 5 samples; det(R)=+1 in all 49. Works because nibabel world is RAS (+z superior) and the loader's (y,−z,−x) maps the host's −x‑superior JSON to branch +z |
| Join | Concatenate, resample, distal trim `DISTAL_TRIM_MM` (4.0 in v1; **0 in v2/v3**; only if ≥30 points remain) |
| Stubs | Drop the 9 unnamed cranial stubs (`is_cranial_stub`: no " - " in the name and min z > `CRANIAL_Z=500`) → 16 files |
| Clearance | Lumen‑to‑lumen clearance of the grafted part (from 130 mm) against the kept neighbours. If clearance < `MIN_CLEARANCE_MM`=**0.0** **and** the nearest vessel is the RVA, repair by shortening first (≤45 mm), then deflecting (≤14 mm, 0.25‑mm grid × 400 Fibonacci directions), both to target 0.75 mm. Any other negative‑clearance neighbour → reject. **0 rejected.** Set A still uses a 0 threshold; it is latent, since the measured minimum is +0.482 mm (mr_011 vs RVA) in all versions |
| RVA repairs | Same 6 anatomies in every version: 002_L, 012_L, 018, 024 (shortened 20.7 mm; byte‑identical blob `a0c9df2` across versions), 025, 026. Deflection amplitudes **v1 4.00/4.50/4.00/5.25/4.00 → v2/v3 2.25/3.00/2.25/5.25/1.50** (002_L, 012_L, 018, 025, 026), because `a898eb2` changed `rva_deflect` to per‑point baselines after v1 was built |
| Outputs | `Centrelines_comb/` (16 `.mrk.json`) + `graft_xform.json` (**written only since 033009a; present in v3 only, 49/49**): keys `R, origin, anchor, anchor_trim_idx, route_from_mm=130, kind='topbrain', centerline_src, surface, mirror, name` |

**Build flags per version** (`topbrain_data/anatomies_v2/BUILD_v2.json`):
- v1: defaults (trim 4 mm, no floor).
- v2/v3: `--distal-trim 0.0 --route-min-r 1.0`, and `--mirror --name-suffix _L` for the left side.
- Five siphons land at exactly 2.00 mm minimum diameter in v2: 001, 007_L, 010_L, 013, 015. 003_L goes from 2.03 to 2.11; it is lifted only at a neck.

### Where the seam actually is (Set A; identical v1/v2/v3)

| Metric | min / median / max (mm) |
|---|---|
| Nominal cut | 130 (actual sample s=130.146) |
| **Nearest‑point distance to densified host RCCA > 1 mm** (basis of HANDOFF 133.6 and `--topbrain_target_min_arclength 133.0`) | **133.5 / 133.9 / 134.6** |
| Nearest‑point > 0.3 mm | 130.6 / 132.0 / 133.6 |
| Equal‑arclength vs raw host > 1 mm | 131.7 / 133.9 / 134.6 |
| Equal‑arclength vs 1‑mm‑resampled host > 1 mm | 132.7 / — / 134.8 |
| Radius departure (smoothstep from 100 mm) | 100.8–101.9 at \|Δr\|>0.01; 101.8–102.8 at >0.05; HANDOFF gives 102.5 |
| In insertion‑path coordinates | path_len ≈ s + 33.3 (seam ≈ 166.9); raw polyline of (11) from index 2 measures 33.47 |

The seam sits about 3.5 mm past the cut because the donor leaves along the host's own 5‑sample tangent.

### Route statistics
- Route length: v1 201.1–263.2 mm (median 227.9); v2/v3 206.0–268.2 (median 232.9), about 5 mm longer because there is no distal trim.
- Min diameter: v1 1.05–4.02; v2/v3 2.00–3.84.
- Rise: v1 153.9–186.5; v2/v3 156.4–187.7.
- Siphon length past the cut: v1 right 71.0–133.1, left 80.2–120.3; v2/v3 right 75.9–138.1, left 85.2–125.3.

### Stage D — meshes: see §5. Stage E — validation: see §6.

### Mirroring rationale
- A proper rotation cannot change handedness.
- The documented torsion table (TOPBRAIN_PIPELINE.md:294‑319): right mean +1.95 rad, left −2.00, 19/25 same‑patient opposite signs.
- The table's medians do not support "near‑exact mirrors" (right −0.89, left −1.06). No torsion script is committed. An independent estimator agrees in sign only. **Treat the torsion evidence as weaker than the docs present it.**

---

## 4. Set B — three-source composites (pipeline)

The scripts are in `carotid_tools/`. Container wrapper: `run_container.sh`, 76 `-v` mounts, **all read‑write**, all hard‑coded to `D:/neve/.claude/worktrees/rl_improv_16_resume`, which is absent on this machine.

### 4.1 Parse, measure, extend

**`analyze_bifurcations.py`** — reads VMTK root‑to‑tip polylines with MISR radii.
- `split_tree` takes as the **CCA** the prefix every path shares with `paths[0]` (tol 0.05 mm). Daughters are the remainders.
- Paths with ≤k+5 points are dropped, and tips within 1.0 mm are treated as duplicates.
- **ICA** is the daughter with the larger median radius over its first half; **ECA** is the next.
- The "bifurcation" is therefore the first divergence of *any* path. 41/48 used lowers have more than 2 daughters, and w_016_left and w_018b_left have bif_deg 0.0. Some forks may be side branches (inference).

**`build_manifest.py`** → `carotid_data/lower_manifest.json`:
- **138 models** (70 L / 68 R; w90/m28/k20; 77 patients, 61 with both sides) plus 1 unparsed (`case_w_030_right`, "tree would not split").
- `NEED_ICA_MM=58`, `MAX_EXTEND_MM=10`, derived from the host staying CCA‑calibre to 72 mm with the seam at 130.
- `extend_mm = max(0, 58 − ica_mm)`. Result: 31 ready, **34 extendable (≤10 mm)**, 73 too short.
- `stenosis_pct = 100(1 − min r / median distal‑third r)`. Across the manifest: 2.53 / median 35.85 / max 86.34%.
  - **20/48 paired donors have their "minimum" at the ICA crop face.** Manifest‑wide it is 95/138. See §4.5.
- The `superior` field is the CCA‑inlet→ICA‑tip **chord**, not an anatomical axis. Using it as "up" was defect #3.

**`extend_ica.py`** → `carotid_data/extended/<name>_ica_extended.json`:
- 34 files, exactly the extendable set. Keys: `name, source, donor, ica_mm_before, ica_mm_after, max_kink_deg, points (0.5 mm), radii`.
- It copies the distal 10 mm of a random long donor (`default_rng(20260830)`, 20 distinct donors) in a local frame `[tangent, +z orthogonalised, binormal]` (`SUPERIOR=[0,0,1]` since a898eb2). It appends `extend_mm + 0.5` mm.
- Added length: 0.50–9.95 mm over all 34 (median 5.97); **0.50–9.50 over the 25 used**. Final length 57.87–58.42.
- **Joins are not C1.** Median join kink is 20.1° (max 42.1°) against a native p95 of ~9.4–10°. The docstring's "C1" claim is false. Frames use a 12‑index (±6 mm) chord tangent (inference as the cause).

### 4.2 Pairing (`match_sections.py`) → `carotid_data/pairing.json`

This is a frozen blob `e2d2044`, committed only in 430bc76 and shared by v1/v2/v3 and 18_p2.

**Lower funnel** (`load_lowers`, :71‑97): 138 → 65 (extend ≤10) → 56 (−9 `DEFECTIVE_LOWERS`) → **48** (−8 with `tip_d < MIN_TIP_D=2.0`, defined :65, applied :95).
- `DEFECTIVE_LOWERS` (:52‑62, added in a898eb2) are 8 host‑colliders plus `case_w_014_left`: k_003_left, k_012_right, w_006_right, w_010_right, w_014_left, w_035_right, w_044_left, w_048_right, w_049_right.
- Dropped for tip < 2.0: k_003_right 1.575 (ext), **k_005_left 1.997 (0.003 mm under)**, k_010_right 0.590, k_013_right 0.897 (ext), m_023_right 1.335 (ext), w_031_left 1.143, w_040_right 1.452 (ext), w_045_left 1.827.
- The 48 lowers: 24 L / 24 R, 25 extended, 40 patients. For extended lowers, the record's `ica_mm` is the post‑extension length (`rec['ica_mm']=e['ica_mm_after']`, :87). There is no `extend_mm` key.

**Siphon pool:** 25 ids × 2 sides = 50, minus `DEFECTIVE_SIPHONS` {mr_013, 014, 015, 025} (right side only, :44‑45), minus mr_006_L (L<60) = **45** (21 right + 24 `_L`). Each passes `prep_siphon` + `anchor_trim`. len_mm 76.45–130.67; prox_d 3.41–6.16.

**Solver:** up to 5 rounds of strict one‑to‑one `linear_sum_assignment`.
- Cost is |tip_d − prox_d|. A pair is forbidden if the mismatch is above `MAX_MISMATCH_MM=3.0`, if the side differs, or if it was already used.
- `LOWER_CAP=5`; `SIPHON_CAP_PER_SIDE=4`, i.e. at most 8 uses per siphon.
- The module docstring (:7‑8) says 4 and 3 and was already stale at 5335a00.
- Rounds added 48+48+48+48+45 = **237 unique pairs**. Re‑running on the stored calibres reproduces them exactly.
- Mismatch 0.00–2.94 (median 0.70). 45 lowers are used 5 times, 3 are used 4 times (w_042_left, w_022_right, k_005_right). 22 siphons are at the 8‑use cap.

### 4.3 Composition (`graft_three.py`, branch‑16 HEAD, 816 lines)

**Seam placement.** `host_cut = 130 − (cca + ica)`, clamped to [15, 72] (`:68‑69`). If the result is below 15, the CCA is trimmed proximally.
- Measured on disk: **host_cut 15.000–59.319 mm (median 43.8) in all versions; 18 anatomies exceed 54 mm** because head_trim returns trimmed CCA to the host. Comments at :78 ("15.0–53.6 across 229 pairs"), :678 and CAROTID_THREE_SOURCE.md:69 ("15–54") are stale.
- Seam 2 is `host_cut + cca + ica` = **129.989–130.000** in every anatomy.
- `k = searchsorted(hs, host_cut)` (:487) overshoots. Geometric seam 1 sits 0.07–0.96 mm (median 0.51) past `host_cut_mm`, and geometric seam 2 sits −0.08..+1.11 (median 0.49) past the nominal value.

**`head_trim`** — advances the donor CCA inlet until its tangent is within `START_TC_THR=27°` of the inlet→ICA‑tip chord.
- `INLET_MAX_TRIM_MM` 25, `INLET_MIN_KEEP_MM` 12. If the budget runs out it returns the **best index**; the old bare `return 0` fall‑through affected 38 anatomies.
- It fires on 87 anatomies (0.96–24.80 mm) in all versions.

**Seam‑1 radii:**
- The join target is the CCA radius 4 mm in (`CCA_ANCHOR_MM`), and the first 4 mm are flattened to it.
- The host is ramped over `min(BLEND_MM=25, hs[k] − BLEND_ANCHOR_MM=10)`, so the ostium stays **5.8121 mm in every anatomy of every version**. Before the fix, 10 ostia were rewritten.
- There is no ramp across the bifurcation (the bulb is kept).

**Placement.** `place()` frame‑matches the CCA onto the host cut. The ICA and ECA ride the **same (R, origin)**.

**ECA:**
- Capped at `ECA_MAX_MM=30` and gated at ≥ `ECA_MIN_MM=17`.
- Radius floored at `ECA_MESH_R_MM` (1.6 in v1, **1.0** in v2/v3).
- On disk: ECA length 17.46–29.98 mm (identical v1/v3). RECA[0] lies 0.01–0.51 mm from the route.

**Siphon.** Placed at the ICA tip with the same frame match.
- Optional **siphon floor** `SIPHON_MIN_R` (default 0, **1.0 in v2/v3**; `:226, :540‑542`, applied before the ramp).
- Then the last 25 mm of the ICA are smoothstep‑ramped to `siphon_r[0]` (:544).

**Route floor.** `ROUTE_MIN_R` (1.60 in v1, **1.0** in v2/v3) applies to the **donor CCA + ramped ICA only** (:551‑554). It is recorded in provenance as `route_min_r_raw` (post‑ramp, pre‑floor; identical v1/v2/v3), `route_floor_mm` and `route_floored_frac`.
- Floored anatomies: 54 in v1 (12 donors), 25 in v2/v3 (5 donors).
- v1 siphons are unfloored, so the v1 RCCA radius reaches 0.494 mm.

**Assembly.** Concatenate, resample at 1 mm, apply `DISTAL_TRIM_MM` (4 in v1, 0 in v2/v3).

**Gates, in order:**
1. `--max-kink 60°`
2. ECA ≥ 17 mm
3. ECA clearance trim: cut at the first ECA point whose gap is < `FUSE_BAND_MM`, minus `CLEAR_MARGIN_MM 0.5`, and reject if the result is <17
4. `eca_reentry()`: a topological ring test in which any overlap after the opening run is cut out
5. `worst()`: route clearance from `host_cut+2` and ECA clearance from 0, against host neighbours
6. RVA repair when the worst neighbour is the RVA: deflect first, then shorten, with per‑point baselines and the ECA included
7. Final rejection if clearance < **`FUSE_BAND_MM = 0.35`** (:240)
- Route and ECA are never compared metrically; that is left to `eca_reentry`.

**Output** (see the folder contract in §8):
- Copies the 15 non‑cranial host branches.
- Writes the RCCA, RECA and possibly‑repaired RVA via `write_curve`.
- Writes `provenance.json`. Its `clearance_mm` is the worst of route/ECA and does not name the neighbour.

**v2/v3 flags** (BUILD_v2.json): `--route-min-r 1.0 --eca-mesh-r 1.0 --distal-trim 0.0 --fuse-band 0.35 --siphon-min-r 1.0`. These override the module globals via `globals().update` at :603‑605.
- The module defaults are still the v1 constants.
- **V2_BUILD_PLAN.md:99's recipe omits `--siphon-min-r`**, so following it verbatim reproduces the unfloored first bake.

**v3 provenance** (:775‑790, branch 16 only) adds `xform = {lower:{R, origin, anchor, kind:'zenodo', mirror:[1,1,1], surface:<Zenodo _lumen.stl>, ica_real_mm, extend_mm}, siphon:{R, origin, anchor, kind:'topbrain', centerline_src, mirror, surface}}`.
- **Defect:** `ica_real_mm = rec['ica_mm']` is the post‑extension length, and `extend_mm = rec.get('extend_mm', 0.0)` is **0.0 in all 223**. For the 114 anatomies on extended lowers, `ica_real_mm` overstates the real ICA by 0.41–9.41 mm. Its effect on the v3 mesh is described in §5.

### 4.4 Seams, repairs and set-level numbers

| Quantity | v1 (215) | v2 = v3 (223) |
|---|---|---|
| host_cut | 15.000–59.319 (med 43.8) | same |
| Seam 2 | 129.989–130.000 | same |
| Host course departure (nearest‑point >1 mm) | 20.0 / 50.1 / 66.1 | 20.0 / 50.0 / 66.1 |
| Nearest‑point >0.3 mm | 18.0 / 47.0 / 62.1 | 18.0 / 46.9 / 62.1 |
| Radius departure | 12–36 mm | same |
| RECA takeoff on RCCA | — | 37.9–73.1 (median 71.8–72.0) |
| Route length | 201.2–256.0 (med 227.1) | 206.1–261.0 (med 231.8) |
| Max kink / seam‑1 kink | 18.8–52.0 (28.8) / 3.63–29.09 (12.15) | — |
| clearance_mm min | 0.352 | 0.435 |
| Repairs | 95 RVA deflected (0.5–7.8), 9 ECA trimmed, 10 ECA re‑entry cuts, 0 shortened; 116 none | 100 RVA deflected (0.5–8.0), 10 ECA trimmed, 0 re‑entry; 122 none |
| RVA modified vs host | 95 | 100 |
| route floored | 54 | 25 |
| eca_floored_frac>0 | 167 | 123 |
| siphon floored | none | **22** (mr_001 ×8, mr_010_L ×8, mr_007_L ×4, mr_003_L ×2) |
| Distinct lowers / siphons | 47 (k4 m4 w39) / 44 | 48 (k4 m4 w40) / 45 |
| Lower side | 109 L / 106 R | 113 L / 110 R |
| Siphon mirror | 114 `_L` | 120 `[-1,1,1]` / 103 identity |

- v1 centerline differences vs v2 on the common 215: RCCA 215 (trim), RECA 167 (floor), RVA 1.
- **v2 = v3 centerlines byte‑identical** (3,791/3,791).

### 4.5 Stenosis actually carried (G5, re‑measured by the f292143 verification)

The grade is `1 − min ICA radius / distal ICA radius` on the shipped RCCA, over the provenance ICA window [host_cut + cca_mm, + ica_mm]. The answer depends on which distal radius is used, so state the definition with every number. v2 and v3 are identical (same centerlines).

| Definition | v1 max | v2/v3 max | v2/v3 ≥50% | v2/v3 ≥40% |
|---|---|---|---|---|
| D1: `build_manifest` formula on the shipped ICA (own distal = median of last third) | 37.4% | **60.4%** | **24** | **35** |
| D3: shipped ICA minimum / donor's manifest distal | 36.0% | **56.0%** | **25** | **33** |
| D2: donor manifest on paper, `1 − max(min r, 1.0)/donor distal` (not a measurement of the shipped anatomy) | 36.0% | 56.0% | 30 | 47 |
| raw donor grade | 73.7% | 73.7% | 35 | 47 |

- **v1:** 0 anatomies ≥40% under every definition. The docs' "~37% cap" is the D1 value.
- **v2/v3:** 24–25 anatomies ≥50%, from five lowers (`k_011_left`, `m_022_left`, `m_030_right`, `w_025_right`, `w_052_left`).
- **There is no uniform 56% cap.** The floor is an absolute 1.0 mm radius, so each donor caps at its own `1 − 1.0/distal`: w_052_left 73.6 → ~55%, w_047_left 63.8 → 44–59%. Under D1 several pairs exceed 56% (k_011_left__mr_012_L 60.4%).
- **f292143's "30 ≥50%, 47 ≥40%"** (all its docs and the commit message) is D2, a paper calculation. It overcounts crop‑face "stenoses" at the ICA tip that the 25 mm seam‑2 ramp overwrites: the 5 extra ≥50% pairs are all `case_w_036_left__*` (shipped 3–13%), and 14 pairs from w_036_left, w_042_left and w_029_right drop out at ≥40%. (An earlier count of "17 tip‑crop donors" is not reproduced.)
- **f292143's "~60% was an estimate, 56% measured" is mis‑framed:** both are measurements. ~60% is D1, which is the method the doc itself names; 56% is D3.
- The manifest spans 2.53–86.34%; 73.6% is the maximum only among the 48 paired lowers. All 8 manifest grades above 73.6% are ICA‑tip crop‑face minima.
- **Meshed (20k OBJ):** a floored 1.0 mm lesion meshes at 0.72–0.88 mm in v2 and 0.75–0.92 in v3, i.e. 54–66% meshed grade. v1 meshes the 1.60 floor at 0.18–0.96 mm; that is erosion.
- **Verdict:** the 6002fcc claim "real 40–74% grades survive" is false. Grades up to about 56% survive.

### 4.6 The eight defects fixed in `a898eb2` (plus later ones)

| # | Defect | Symptom | Fix |
|---|---|---|---|
| 1 | `anchor_trim` never called on the siphon | mr_021 rose 1.1 mm (native 65.4); 11 rose <20 mm, 4 negative | call `anchor_trim` after `prep_siphon` (:651‑662); also in match_sections |
| 2 | Clearance measured from 130 mm; ECA never checked | 27 routes and 32 ECAs interpenetrating | route from `host_cut+2`, ECA from 0, ECA trimmed first |
| 3 | `extend_ica` used the chord as "up" | roll of the copied 10 mm decided by noise (case_m_022_left \|n\|=0.047) | `SUPERIOR=[0,0,1]` |
| 4 | Stenotic donors sealed by the v1 mesher | 27/229 routes closed at 29–54% of path | `ROUTE_MIN_R=1.60` (donor section) |
| 5 | ECA missing from the mesh | 102–103/229 ECAs absent past 20 mm | `ECA_MESH_R_MM=1.6`, cap 30, gate 17 |
| 6 | Ramp window past s=0; target at the pinched crop face | ostium rewritten (5.00, 5.60 mm); waist at seam 1 | `BLEND_ANCHOR_MM=10`, `CCA_ANCHOR_MM=4` |
| 7 | RVA scalar baseline, blind to the ECA, shorten‑first | every deflection ≥4.00 mm; 17 shortenings removed a median 33 mm | per‑point baseline, ECA included, deflect before shorten |
| 8 | Validator blind to severing (`select_enclosed_points`) | reported 208/229; really 32/229 severed | component‑aware `navigable_route` + `--selftest` |

Also fixed in a898eb2: `head_trim` budget and fall‑through; `eca_reentry` for 8/220 false ICA‑ECA rings. Later:
- `ba3715b`: `FUSE_BAND_MM=0.35` on all four gates. `case_m_024_left` had a +0.057 mm centerline gap yet fused into a 1.60 mm ring in 4 of its 5 anatomies.
- `033009a` / `6002fcc`: the siphon floor.

### 4.7 Counts through time (reconciled)
- **231** at 5335a00 is a claim only: 264 pairs planned, 250 grafted, 231 passing, and no geometry committed (8 files: 7 code + .gitignore).
- The history doc's chain is 231 → 229 → 220 → **216** (430bc76 ships 216 folders, 4,104 files) → **215**. `ba3715b` deleted `case_w_003_right__topcow_mr_017`.
- **Why it was deleted:** it was a **fusing‑band rejection**. Its 430bc76 `clearance_mm` was 0.3091 < 0.35, against the LVA, which the RVA repair cannot fix. It was never in `excluded_severed.json`. 18_p2 `9b420ef`'s "fifth severed" and the doc's "six severed" are both wrong.
- `excluded_severed.json` has **5 names at every commit**: m_006_left__007_L, m_030_right__003_L, w_012_left__003_L, w_018b_left__007_L, w_040_left__023_L. It is a v1‑only list, and **no code reads it**.
- **v1's 22 absences** = 5 severed (built, then pruned after baking) + 17 grafter rejections. At 430bc76 it was 5 + 16, 16 = 5 w_014_right (ECA re‑entry at 16 mm < 17) + 11 route overlaps. That breakdown is from reconstruction.
- **v2/v3 = 223.** They are v1 + 8, with nothing dropped (confirmed by the f292143 verification, C3–C7):
  - the 5 severed are readmitted. In v2/v3 all five are one component and navigable. Four (the `mr_007_L` and `mr_003_L` pairs) were regrafted with the 1.0 mm siphon floor; `case_w_040_left__topcow_mr_023_L` was not (raw siphon minimum 1.32 mm), so for it the SDF mesher alone keeps the terminus open.
  - 3 `case_w_014_right` pairs (mr_010, 011_L, 027) are readmitted because the ECA floor at 1.0 removes the ECA re‑entry. At the v1 1.6 mm ECA floor the ECA‑to‑route gap after the opening run is −0.24 mm, the re‑entry falls at 16.14 mm, and the cut ECA (15.64 mm) fails `ECA_MIN_MM = 17`. At 1.0 mm the gap is +0.36 mm. The re‑entry test is itself `(d − rr − er) < FUSE_BAND_MM`, so "back above the fusing band" is loosely right; the recorded `clearance_mm` (0.76–0.82) is not what moved, and the route floor plays no part (these routes are never floored).
  - 6002fcc and BUILD_v2 ("223 built, 14 rejected for fusing below the band (v1: 215 built, 22 rejected)") misattribute all eight to the fusing band; f292143 corrects this to 5 + 3 but still calls the three "rejected for fusing".
- **The 14 unbuilt in v2/v3:** w_003_right__{mr_010, mr_017}; w_014_right__{021, 024}; w_016_right__015_L; w_026_left__{002, 023, 026}; w_034_right__021_L; w_037_right__{001_L, 002_L}; w_046_left__{022, 027}; w_048_left__027.
  - The ECA‑length and ECA‑clear gates are ruled out deterministically, because every involved lower has built siblings.
  - Reconstruction (validated on 6 built pairs to within ~0–0.7 mm) puts **13 of 14** at the final clearance gate, most as outright overlaps (−0.2 to −4.1 mm) with the LVA or LCCA. `w_016_right__015_L` is the one inside the band (0.00 to +0.18 mm after an RVA fix); `w_003_right__mr_017` is just below zero (−0.08 to −0.46; an earlier reconstruction gave +0.33). `w_026_left__mr_002` is indeterminate (−0.85 to +3.28 mm depending on the donor).
  - No grafter log exists, so these attributions are **reconstruction‑only**.

---

## 5. Collision meshes v1 / v2 / v3

The environment only ever loads `<anatomy>/vessel_architecture_collision.obj`, via SOFA `MeshObjLoader` → `MeshTopology` (triangles only) → static `TriangleCollisionModel` + `LineCollisionModel`, with `LocalMinDistance contactDistance 0.3, alarmDistance 0.5`.

**Navigability criterion, used everywhere:** meshed lumen − 0.3 contact ≥ 0.35 catheter radius, i.e. `lumen_min ≥ 0.65 mm`. `lumen_min_mm` is a **radius**, the 3‑mm end caps are excluded, and it is computed on the RCCA only.

### 5.1 v1 — `bake_meshes.py` → stEVE `meshing.generate_mesh`
- **Method:** voxel cube [0.6, 0.6, 0.9] mm with 5 padding layers; binary spheres; `gaussian_smooth(1)` twice (σ = √2 vox = 0.85 mm x/y, 1.27 mm z; anisotropic); marching cubes at `level=None` ((min+max)/2); `decimate(0.99)` quadric without volume preservation → ~3.7k triangles.
- **Faces:** A 3,654–3,737, B 3,677–3,783, about 163 KB each.
- **Straight‑tube erosion** (report.txt): r ≤ 1.0 absent; r=1.2 → 0.66 (z) / absent (x); r=2.0 → 1.62/1.41; r=4.0 → 3.69/3.64.
- **Set‑wide** (`lumen_v1.json`): median deficit **0.653 mm** (about half blur, half decimation).
  - Min‑lumen median A 0.560, B 0.534.
  - **Navigable: A 22/49, B 71/215 (93/264).**
- **Topology:** components A {3:43, 4:6}, B {3:171, 4:24, 2:19, 1:1}.
  - A 74‑face distal‑RVA fragment and a 6‑face RVA‑tip fragment sever the vertebrobasilar ring.
  - Boundary edges are present in 49/49 A and 186/215 B.
  - 2–4 duplicate vertices per file.
- **Mesher‑compensating constants** existed because of this mesher: Set B `ROUTE_MIN_R 1.60`, `ECA_MESH_R_MM 1.6`, `DISTAL_TRIM 4`, `FUSE_BAND 0.35`. Set A had only `DISTAL_TRIM 4`.
- **Bake dates:**
  - A: 25 right on 2026‑08‑27, 24 left on 08‑29.
  - B: first bake 2026‑08‑31T22:23 (430bc76 headers). The ba3715b re‑bake on 09‑01 changed only the header in 196 files and the geometry in 19.

### 5.2 v2 — `bake_meshes_v2.py` → `sdf_mesher.py`
- **Field:** `SPACING_MM 0.45` isotropic, `BAND_VOX 5`, `SAMPLE_MM 0.25`, `K_NN 8`, `PAD_VOX 6`, `FAR −1e3`.
  - f = max over branches of max over the 8 nearest 0.25‑mm samples of (r_j − |x−c_j|). This is a union of spheres, with radius lookup per branch.
  - The band is placed by coarse marking every max(spacing, 0.2r).
- **Surface:** marching cubes at **iso 0** with the mask dilated by 2 and `allow_degenerate=False`; no smoothing.
- **Decimation:** quadric with volume preservation → **60,000** (`collision_full.vtp`, gitignored and absent) and then **20,000** (`.obj`).
- **Result:**
  - Deficit median 0.118 (0.056 at 60k).
  - **49/49 and 223/223 navigable**, 1 component, 0 route points outside.
  - Min lumen A 0.740 (mr_015) / median 1.442; B 0.721 (case_k_011_left__topcow_mr_023_L) / median 1.380.
  - Open edges: A 0; B max 3 (2 anatomies).
- **Why 20k.** V2_BUILD_PLAN.md:43‑62 table: 3.7k 134/150 ms (free/contact), 20k 202/209 (**+51% / +39%**), 38k 219/234, 387k 2281/2243. The doc's "+55%" is its 12k figure. The baselines differ between docs (figures.py: 129/140), and the raw outputs are not on disk.
  - **G8 measured the shipped meshes** (seeded, first 40 trunk steps): A mr_001 v1 85 / v2 175 / v3 168 ms; B case_w_050 85 / 144 / 142. So v2/v3 cost **1.7–2.1× v1**, and v3 ≈ v2 at the same budget. The one 30k mesh costs +56%.
- **Not applied to any shipped set:** the VMTK route‑weighted remesh (`remesh_vmtk_v2.py`) and `recut_obj.py`. There are 0 `remesh`/`recut_to` keys.
- **`recut_obj.py` traps:** it does not re‑measure lumen, it always writes `mesh_v2.json` even in v3 folders, it would undo v3's adaptive budget, and it needs the absent `collision_full.vtp`.
- **The "1.0 mm tube meshes at 0.97" premise** comes from a probe (0.3‑mm grid, 250 triangles). On the shipped 20k OBJ, floored necks mesh at **0.72–0.93** (v2) and **0.75–0.96** (v3). The worst margin over 0.65 is 0.071 / 0.097 mm.
- **Fusing band on the SDF grid:** `fuse_band_v2.py` shows fusion at gaps ≤0.10 mm for one alignment. Over 6 grid alignments, a 0.35 mm wall fuses in 3/6 and 0.45 mm is the first gap that never fuses. So the worst case is about 1 voxel, above FUSE_BAND_MM. It stays at 0.35.

### 5.3 v3 — `bake_meshes_v3.py` + `sdf_union.py`
- **Field:** f = **max(f_tube, min(f_real, f_capsule))**.
  - `f_real` = −implicit distance (positive inside) to the patient's real surface, moved by `v' = R(m⊙v − o) + a`. It is evaluated only within `BAND_MM 2.0` of the capsule.
  - Capsule radius = `r + (1.8r+1 − r)·w`, where w ramps linearly over `TAPER_MM 8` from the flagged ends (`sdf_union.py:37‑40, 72‑81`) and r is the **declared (floored)** radius.
  - Because of the max(), the v3 lumen contains the v2 tube *field*. After decimation, the OBJ minimum lumen is lower than v2 in 9/49 A and 40/223 B, by up to 0.14 mm.
- **Real surfaces:**
  - A: TopBrain stage‑A `.vtp` for the siphon section, from `route_from_mm=130` to the end, tapered at the start only. 25 surfaces + 24 surfaces_left.
  - B: Zenodo `_lumen.stl` for the "lower" section [host_cut, host_cut+cca+min(ica, ica_real)] (both ends tapered) and for the ECA (tapered at the end); TopBrain surface for the siphon (tapered at the start).
  - Host arch, trunk and RVA remain tubes. **So in training the host arch is a tube, while in the host test it is the real (soup) surface.**
- **Iso‑surface → keep the largest component** (logs `dropped_fragments`) → 60k → **adaptive OBJ budget** 20k → 30k → 40k → 60k until navigable.
  - Only `case_m_006_left__topcow_mr_007_L` stepped up, to 30k (29,998 triangles; 0.63 mm at 20k → 0.892).
- **Result:**
  - Tube deficit 0.115; real deficit 0.041 (A) / 0.056 (B); overall obj deficit A 0.096, B 0.068.
  - Min lumen A 0.783 (mr_015 at 226.3 mm) / median 1.491; B 0.747 (case_w_025_right__topcow_mr_007 at 86 mm) / median 1.416.
  - **49/49 and 223/223 navigable.**
  - Shape (16 rays, 4‑mm stations; max/MISR): siphon A 1.255, B 1.237; lower 1.127; tube 1.022.
  - `voxels_widened` counts band voxels whose field rose. It is not lumen gained.
- **Topology** (OBJ): 1 component everywhere. Open edges (pyvista `n_open_edges` = boundary + non‑manifold): A max 5 (8 anatomies >0), B max 9 (68 >0).
- **Report provenance:** the v3 reports come from mixed revisions of `bake_meshes_v3.py`.
  - `dropped_fragments` is present in 148 B + 3 A reports (all baked ≥18:09:51). The 38 reports with iso comps >1 but no key were baked ≤18:34:32.
  - `obj_tris_budget` is present in 1 report.
  - 12 reports have `full.comps>1` (A mr_008/013; B case_w_024_right ×5, case_w_036_right ×5). All of them ship as one component.
  - **A re‑bake would not reproduce the shipped bytes.**
- **ica_real_mm mislabel effect** (G1; 114 anatomies, 25 lowers):
  - The min() guard at `:67` never engages (label − ica_mm = +0.047..+0.306 in all 223), so every lower capsule runs to seam 2.
  - Inside the synthetic extension the tube governs (v3−v2 ≤0.08–0.10 mm), so there is **no bulge**.
  - At the true real outlet, the capsule is still 3.0–5.0 mm wide, so the outlet is face‑cut rather than tapered. The equivalent radius steps up to 0.32 mm, and **one‑sided lips drop by more than 0.3 mm/mm in ~31–33/114** (worst 1.79 mm, case_k_005_right__topcow_mr_013_L), against 2/109 in controls.
  - 15 anatomies on w_012_left, w_017_right and w_025_right have an overrun below 0.6 mm and are unaffected.
  - `mesh_v3.json` counts the extension as "real".
  - The fix needs `ica_mm_before`, or the manifest `extend_mm`, plus the STLs, which are absent.
- **Real fusions introduced by v3** (G2 genus audit, G10 check8 re‑run):
  - Baseline: in the host tree, RVA and LVA meet at the vertebrobasilar confluence. That ring is anatomy. It gives **genus 1** to A v2 47/49 and B v2 184/223. The ring‑less ones are exactly those whose RVA tip the grafter moved.
  - v2 has **0 extra handles**.
  - v3 has extra handles in **A 9/49 and B 64/223 (70 handles)**:
    - siphon self‑contact at s 191–236 (A 7, B 35; donor‑driven: 001, 001_L, 008_L, 011, 018, 021_L are consistent);
    - route/siphon↔RVA (A 2, B 15);
    - ECA↔host/RVA through the CCA (B 10);
    - ICA↔ECA 8–35 mm distal to the fork (B 7: w_014_right ×3, w_050_left ×4).
  - G10's check8 wall metric on v3 finds 9 RCCA‑RVA merged runs. **`case_w_047_left__topcow_mr_018_L`** has a merge at s≈195–202 mm that keeps 1.1–1.48 mm clearance, above the 0.65 envelope. v2 has a wall there. This anatomy is in the carotid v3 **train** split.
  - Two carotid v3 holdouts carry thin merges: w_007_left__016_L and w_025_right__021_L.
  - Mechanism (inference): the capsule admits real lumen out to 1.8r+1, while the graft clearance gates use tube radii. `check_anatomies.py` has no genus or ring detector.
- **SOFA controlled reach** (G8; fixed rotation, pure push, 200 steps):
  - v3 mr_015 stalls at 194 mm and mr_008 at 196–197 mm. Neither reaches its target, although v2 does.
  - v3 case_w_050_left__topcow_mr_021_L lodges the wire in a real‑surface ECA pocket, 4.12 mm off the centerline (inside v3 only). Insertion then runs to 317 mm with DOFs 1.81 mm outside the wall.

### 5.4 Summary table

| | v1 | v2 | v3 |
|---|---|---|---|
| Mesher | binary spheres, Gaussian ×2, MC at half level, decimate 0.99 | SDF sphere union, 0.45 mm, iso 0, quadric VP | v2 field ∪ real surface ∩ capsule (1.8r+1, 8‑mm taper) |
| OBJ triangles | ~3.65–3.78k | 20k (B 19,998–20,000) | 19,998–20,000; one at 29,998 |
| Components | 1–4 (mostly 3) | 1 | 1 |
| Obj open edges | 0–12 (B) | A 0, B ≤3 | A ≤5, B ≤9 |
| Median deficit | 0.653 | 0.118 | tube 0.115 / real 0.04–0.06 |
| Navigable A / B | 22/49 / 71/215 | 49/49 / 223/223 | 49/49 / 223/223 |
| Min lumen A / B | (0.22–0.35 on samples) | 0.740 / 0.721 | 0.783 / 0.747 |
| Extra genus handles | ring severed | 0 | A 9, B 64 |
| Build dates | 08‑27, 08‑29 / 08‑31 → 09‑01 | 09‑03 16:21–17:04 | 09‑03 17:54–19:06 |
| Rebuildable here? | no (raw data absent) | from centerlines only | **no** (real surfaces and Zenodo STLs absent) |

---

## 6. Verification and QC

### 6.1 Graft-time gates
See §3 Stage C and §4.3. Set A still rejects at clearance 0.0. Set B uses 0.35 on four gates: ECA trim, re‑entry overlap, RVA trigger and final acceptance.

### 6.2 `validate_anatomies.py`
- Route length, z‑rise, max kink, junction kink (`kk[j-4:j+4]`), and min/terminal diameter against the host.
- **No pass/fail.** It returns 1 only if no anatomies are found.
- v2 and v3 give identical stats.

### 6.3 `check_anatomies.py` (649 lines; identical on 16 and 18_p2)

| Check | Rule |
|---|---|
| load / insert | RCCA present; insertion = (11) bridge point index 2, inside mesh |
| enclose | fails only if depth > `MAX_OUTSIDE_MM 1.5` **and** run > `MAX_OUTSIDE_RUN 3`; depth is nearest‑**vertex** (overstates: mr_015 7.19 vs exact 6.87) |
| route (a898eb2) | bridge+RCCA resampled at 0.25 mm, per connected component; fails if the insertion point is in no component, the gap exceeds `MAX_ROUTE_GAP_MM 3.0`, the handoff is >0, or host open edges >12 (boundary edges only) |
| targets | `CenterlineRandom(threshold 5, [RCCA], min_arclength 40)`; "reachable" is a note only; **40 mm, not the training 133** |
| declared fit | fails if 2·r_min ≤ 0.7 |
| meshed lumen (033009a) | `route_lumen` on RCCA only, minus 0.3, compared with 0.35. **Fails open** if the call throws (try/except only adds a note) or if no body point is inside |
| controls | `--host .../Centrelines_comb` meshes the host **centerlines through the v1 mesher**. Both controls fail the meshed lumen (0.00 mm at 121 mm) and do not affect the exit code. Passing the parent directory scores the raw OBJ in a different frame (a trap) |
| `--selftest` | v1 mesher only; a 0.6 mm neck gives 2 components, enclose passes (0.9950), route fails (gap 50.25) |
| `--sofa N` | see below |

**Full re-run (G3, no SOFA, read-only):**
- **A v2 49/49, A v3 49/49, B v2 223/223, B v3 223/223 pass**, with 0 notes, gap 0.00, 1 component, enclosed 1.000, targets 100% reachable.
- The meshed‑lumen number equals the bake report's to within 0.0005, because both call the same function. It is **not an independent check**.
- **No v2/v3 check log is committed.**

**`--sofa` harness problems (G8):**
1. The printed target arc mixes the tracking frame (rotated by `image_rot_zx [20,5]`) with the vessel frame. It is off by up to 73 mm, and the >130 mm gate misclassifies 0–2/60 seeds.
2. 40 steps insert 63.4 mm, so the tip ends ≤33 mm along the RCCA. **It never reaches the siphon seam at 130 mm.** For 19/223 Set B anatomies with host_cut < 32 it does enter the lower graft.
3. `SofaBeamAdapter._rng` is unseeded, so the initial device rotation, and with it the branch entered (RCCA/RVA/(17)), is random. SOFA itself is deterministic when seeded.
4. "OK" means no exception.

The docs' "A 6/6, B 9/9 and 10/10 SOFA rollouts at targets 140–244 mm" is therefore weak evidence of graft traversal. G8 samples re-ran OK.

### 6.4 Bake-time reports
`mesh_v2.json` and `mesh_v3.json` record `lumen_min`, `route_pts_outside` (0 everywhere) and `navigable`. Keys differ by baker revision (§5.3).

### 6.5 Controls and conventions
- `vtkImplicitPolyDataDistance` is **positive inside** on v1, v2 and v3 (checked).
- v1 non‑watertight meshes make the sign and `select_enclosed_points` disagree (mr_015: 15 vs 23 outside).
- HANDOFF 11.2's "all cohort meshes non‑watertight" is v1‑only.

### 6.6 What each check caught and missed

| Defect class | Caught by | Missed by |
|---|---|---|
| Roll DOF (siphon laid over) | figure rendering (by eye) | all automated checks |
| head_trim fall‑through | final audit | — |
| Clearance scope 130 vs host_cut | hand audit | — |
| Severing | statistical audit → route check | enclose check |
| ECA missing from mesh | root‑cause workflow | declared‑radius checks |
| Fusion at proximity | direct question → FUSE_BAND | zero‑threshold gates |
| Mesher erosion (v1) | meshed‑lumen check (033009a) | declared fit |
| Unfloored Set B siphon necks (v2 first bake) | meshed lumen at bake | — |
| v3 fusion rings / merges | **nothing in the pipeline** (G2/G10 found them post hoc) | check_anatomies (no genus test) |
| v3 outlet lips (ica_real_mm) | nothing (G1 post hoc) | all |
| mr_025 v1 mid‑siphon pinch | eval episodes (4/4 proximal ok, 5/5 distal fail); now the meshed lumen (0.31 mm at 167) | v1 static checks |

### 6.7 First v2 bake (13 vs 3) — settled by G5 reconstruction
- **Method:** unfloored siphon radii put back into the 22 regrafted pairs (18 index‑exact from v1, 4 aligned) and re‑baked with the shipped v2 mesher. A control re‑bake reproduced `mesh_v2.json` exactly.
- **Result:** **13 flagged at 0.302–0.628 mm**: all 8 mr_010_L pairs, all 4 mr_007_L pairs, and 1 mr_001 pair (w_016_right). This matches BUILD_v2's count, range and siphon list.
  - The 11 index‑exact rebuilds alone span 0.30–0.63.
  - The two mr_003_L pairs (0.86–0.87) were regrafted but never flagged.
- **Stale:** the "3 at 0.37–0.57, every one on a necked `_L` siphon" in `graft_three.py:220‑225` and V2_BUILD_PLAN.md:93. mr_001 is a right siphon.
- The origin of the "3" is a hypothesis: the first three flagged in sorted order.
- **Disputed count.** The f292143 verification (K5) ran a cruder, approximate unfloored re‑bake and got **7 of 22**, not 13. G5's reconstruction is the more careful of the two (index‑exact for 18 pairs, and a control re‑bake reproduced `mesh_v2.json` exactly), but the two independent attempts disagree. Treat "13" as BUILD_v2's record, supported by G5 and not by K5. Either way the mechanism holds: 22 pairs on mr_001, mr_003_L, mr_007_L and mr_010_L were regrafted with `--siphon-min-r 1.0`.

### 6.8 The 18_p2 QC instruments are all v1 (G10)
- **TopBrain audit (08‑27/28):** `audit_topbrain*`, `audit22_*`, `nav_quality_nq*`, `attack*`, `refute_*`, `t2/t3/t4*`. It read `results_topbrain/anatomies`, i.e. `topbrain_data/anatomies` **v1 25 right only**. Those 25 folders are byte‑identical at HEAD.
- **Carotid "216 sweep":** `arj_check5_*`, `ak_check8_*`, `check8b_*`, `check14_*`, `arj_reach_*`, chirality scripts. It audited the **430bc76** v1 state, before the ba3715b re‑bake.
  - Only check8 committed output. `check8_wall2_nseg401.json` is byte‑identical to the NSEG=161 file.
- **Re-measured with the audit estimator:**
  - TopBrain v1 defects reproduce on v1: mr_015 55 pts outside up to 6.86 mm; mr_003_L 21 up to 2.25; mr_014 11 up to 2.11; mr_024 min 0.013; mr_025 pinch 0.218.
  - v2/v3: 0 outside for all listed anatomies, minimum clearance ≥0.713. The mr_025 window is 1.230 (v2) and 1.296 (v3).
- **Conclusion:** TOPBRAIN_ANATOMY_AUDIT.md, HANDOFF 9.1/9.6/11.6 and nav_quality.json **do not apply to v2/v3**. REPRO_topbrain_v3.md says all 49 v3 anatomies passed intake.
- **check8 re-run** (re‑fusion / RCCA‑RVA zero‑wall): current v1 1/17; v2 5/0 (all w_029_right); v3 4/9. ba3715b's "zero rings" was incomplete for w_029_right.

### 6.9 Fork chirality
- **Claim** (4cc6b35, HANDOFF 12.6): left‑sourced forks 77/109 positive vs right 16/106. The same‑patient control shows **3/8** forks opposite vs 19/25 siphons, so forks are patient‑specific and left unmirrored. v3 still records the lower mirror as identity (`graft_three.py:780`).
- **Contradicting evidence (G10):**
  - The committed 18_p2 instruments `arj_handedness_ck13b/c/d`, re‑run on v1, give **6/8–7/8 opposite** for chord/dihedral measures. ck13d says "if LEFT mirrored +0.2°". Only CCA‑torsion measures give 2–3/8.
  - No committed script reproduces 77/109 and 16/106. ck13 gives 74/109 and 33/106.
- The task‑relevance argument still holds: the ECA is a decoy, and the route is CCA→ICA. **The empirical basis of the verdict is unresolved (§11).**

---

## 7. Runtime integration (18_p2 only)

### 7.1 Loader contract
Files: `eve_bench/dualdevicenavtopbrain.py` and `eve/.../topbrainanatomyset.py`, identical on both branches.

- `find_anatomies(anatomy_dir, only, exclude)`:
  - Lists subdirectories containing `Centrelines_comb/`, sorted. `BUILD_v*.json` is ignored.
  - Applies `only` first; unknown names **raise**, and an empty result raises.
  - Applies `exclude` second as a **silent** set difference. This is why `__none__` works; there is no code for it.
  - Requires `vessel_architecture_collision.obj`, or construction raises `FileNotFoundError`.
- Centerlines load via `load_branches` in the (y,−z,−x) frame. "(0)" and the paren‑less names share sort key 0, so the order depends on `listdir`. It is one order per set here, with RCCA at index 3.

| Tree | Branches | RCCA | RECA | RVA | (11) |
|---|---|---|---|---|---|
| Host | 25 | 3 | — | 4 | 13 |
| Set A | 16 | 3 | — | 4 | 11 |
| Set B | 17 | 3 | 4 | 5 | 12 |

- Baked meshes are already in the vessel frame and are used **unrotated**. The host is rotated [90,−90,0] by FromMesh.
- **Fingerprint** = name stripped to `[A-Za-z0-9]` (e.g. `topcowmr001L`). It matches `checkpoint_restore`'s `_mesh-([A-Za-z0-9]+)_`.
- **Selection** is stateless: generation g → `default_rng([seed & 0x7FFFFFFF, g//n]).permutation(n)[g%n]`. The constructor selects g=0 and sets `_generation=1`. `reset` switches only when `episode_nr>0 and episode_nr % N == 0`. A non‑None seed re‑seeds and zeroes the generation. Explore and heatup reset unseeded; eval resets seeded.
- **Insertion:** derived once from the (11) bridge at k=2. Pos `[17.7913, 14.554159, 398.16803]`, dir `[−0.064692, −0.008647, 0.997868]`, identical to the host for every anatomy in every version.
- **Targets:** `CenterlineRandom(threshold 5, branches=[RCCA], min_arclength_from_start)`. Every other branch, RECA included, is excluded. Default 40 mm; launchers use 133.
- **Anatomy switch:** changes `mesh_path` (or `vessel_visual_path`, insertion, bounds) and triggers a full SOFA rebuild.
- `TopBrainAnatomySet` is deliberately **not exported** from `vesseltree/__init__.py` (Fix D5). The image's installed eve lacks `topbrainanatomyset.py`, `rccavariedfrommesh.py` and `rccaprocedural.py`; launchers must bind‑mount them.
- **SOFA device rotation is unseeded** in training and eval as well (`sofabeamadapter.py:50,487,103`).

### 7.2 Trainer wiring (`DualDeviceNav_train.py` on 18_p2; added c7b60f6)
- `--topbrain` requires `--env_version 5` and is incompatible with `--procedural_rcca`.
- Training exclude = `sorted(set(exclude) | set(holdout))`.
- Worker i: `seed=12345+i`, `episodes_between_change=change_every`.
- Eval env: `seed=12344`, `episodes_between_change=1`, `only=holdout`, **no exclude**. The help text and the launcher comment wrongly say exclude also applies to eval.
- The 98 `EVAL_SEEDS` are the same list as `SEED_LIST=validation`.

**Argparse defaults, all v1‑era:**
- `--topbrain_dir /opt/eve_training/topbrain_data/anatomies` (v1)
- `--topbrain_exclude [topcow_mr_013, 014, 015]`
- `--topbrain_holdout [005, 012, 020, 026]`
- `--topbrain_change_every 10`
- `--topbrain_seed 12345`
- `--topbrain_target_min_arclength 40.0`

**Rosters:**

| Run | Dir | Holdout | Exclude | Train | Target | change_every |
|---|---|---|---|---|---|---|
| TopBrain v1 (`launch_rcca_topbrain_v1.sh`) | anatomies (25‑folder at the time) | 007 008 017 022 | 013 014 015 025 | 17 | 133 | 10 |
| TopBrain v2/v3 | anatomies_v2/_v3 | 007, 008, 017, 022 + `_L` twins (8) | `__none__` | **41** (21 patients) | 133 | 10 |
| Carotid v3 (4 launchers) | carotid_data/anatomies_v3 | **16** = the full 6×6 block | **54** cross‑terms | **153** | 133 | 10 (base), 3 (resume, noprivactor, nopriv_full) |

**Carotid v3 split structure** (4eef6bb; checked against env_train.yml):
- Holdout lowers: m_030_right, w_007_left, w_013_right, w_017_right, w_025_right, w_033_left.
- Holdout patients: mr_003, 011, 013, 016, 017, 021.
- The exclude list is exactly the 54 composites that share a lower or patient with the block. Train shares 0 lowers and 0 patients with the holdout.
- Caveat: 9 train composites use the **other‑side carotid** (w_007_right, w_017_left) of holdout cases.
- The Set A and Set B holdouts overlap by patient (017), and each set trains on the other's holdout patients. Cross‑set evaluation is not patient‑novel.
- The check15 annealing split on 18_p2 (reproduced 84/22/43/67 on the 216 set) is **exploratory**. No launcher uses it and its outputs were never saved.

**Task geometry vs target floor:**
- For Set A, 133 is the measured seam. At 40 mm, 41–57% of targets lie before the seam.
- **For Set B, 133 is not a seam.** The host is left at 20–66 mm and the RECA takes off at 38–73 mm. At 133 every target lies past the fork, at least 19 mm from the ECA.
- Pool sizes: A v3 @40 167–229, @133 74–136; B v3 @40 155–216, @133 74–129.

### 7.3 RECA semantics (G6)
- **`classify_physical_branch`** (pathcontext.py) has no ECA tag, so "RECA" → "other". Its 8‑mm named‑daughter snap then reports the proximal **7.98–29.80 mm** of every ECA (median 13.4) as **"RCCA"**. 22/223 are all‑RCCA and 42/223 are partly "RVA".
  - Knock‑on effects:
    - env5 Fix‑2 holds the off‑branch counter at 0 while the wire sits in the proximal ECA, so observation features read 0, the stuck trigger cannot fire, and the retract discount never applies;
    - OST cannot fire from the ECA;
    - the privileged one‑hot says "target daughter";
    - `EPISODE_OUTCOME` logs `final_branch=RCCA, is_clean=1`.
- **The state machine** adds an (RCCA, RECA) daughter entry but no junction‑sequence commit. `_pick_off_path_branch` uses the arclength‑nearest junction, so **any off‑path event past about 54–73 mm, including in the distal siphon, is labelled RECA** (in Set A, RVA). Logs: 135,294 distal steps on RECA, and **172 successes ended "on RECA" and are excluded from the PER clean lane**.
- **"final_branch" has three definitions:**
  - EPISODE_OUTCOME (classify);
  - single‑target `info['final_branch_short']` (state machine);
  - the PER lane (index equality).
  - The experience cache uses the state machine, contrary to the env5 comment.
- The heuristic has no hard‑coded names, but an extra daughter entry changes its regimes around the ECA fork ("near", then "default" for 10 mm).

### 7.4 Eval-side defaults and traps
- `eval_anatomies.py`:
  - default dir `/opt/eve_training/results_topbrain/anatomies` = v1 via the `launch_eval_topbrain.sh` mount;
  - default exclude = the v1 trio ("name‑based so it survives re‑baking");
  - target 40; `--change_every 2`; per‑worker seeds `900000+10000(i+1)`.
- `launch_eval_topbrain.sh` passes no `--topbrain_dir`, so it **evaluates v1**.
- `launch_eval_carotid_traj.sh` sets the v3 dir and the 16 holdouts but **no target flag (evaluates at 40)** and `CHANGE_EVERY=2`. So `SEED_LIST=validation` reproduces neither the anatomy nor the target mapping of the in‑run blocks.
- Under `--verify_variation`, a 133 eval prints MISMATCH against RCCAVaried's 40 and aborts.
- `launch_eval_anatomies.sh` forwards the obs masks but mounts no Set A/B data. The mask passthrough exists only on 18_p2 (HANDOFF 18.6).
- **Runtime effects confirmed in logs** (carotid v3 nopriv_full):
  - Episode 0 of a fresh eval env keeps the construction anatomy. All 157 ep‑1 evals ran on `case_m_030_right__topcow_mr_013_L`; for TopBrain v3 the computed construction anatomy is `topcow_mr_022`.
  - Worker respawns restart the schedule at g=0: 322 lifetimes = 16 + **306 respawns**, and exposure is 36–318 episodes per anatomy.
- A `--checkpoint_dir` pool with foreign fingerprints raises `ValueError` at reset, which triggers worker respawns rather than a clean fall‑through.
- The env5 anatomy hash covers centerlines only, so **it cannot tell v2 from v3**.

---

## 8. Inventory

### 8.1 Canonical counts

| Dir | Entries | Anatomy dirs | Extra | Files/anatomy | Tracked = on disk | Size |
|---|---|---|---|---|---|---|
| `topbrain_data/anatomies` (A v1) | 49 | 49 | — | 17 | 833 | 58.8 MB |
| `topbrain_data/anatomies_v2` | 50 | 49 | BUILD_v2.json | 18 | 883 | 95.8 MB |
| `topbrain_data/anatomies_v3` | 50 | 49 | BUILD_v3.json | 19 | 932 | 95.9 MB |
| `carotid_data/anatomies` (B v1) | 215 | 215 | — | 19 | 4,085 | 258.3 MB |
| `carotid_data/anatomies_v2` | 224 | 223 | BUILD_v2.json | 20 | 4,461 | 433.6 MB |
| `carotid_data/anatomies_v3` | 224 | 223 | BUILD_v3.json | 20 | 4,461 | 434.8 MB |
| `carotid_data/extended` | 34 files | — | — | — | 34 | 0.34 MB |

Other data files: `pairing.json` (237 pairs, 91,475 B), `lower_manifest.json` (138 models, 137,815 B), `excluded_severed.json` (5, 487 B), `label_necks.json` (50, 6,343 B, **branch 16 only**). In total ~1.38 GB of worktree data; no LFS; pack 548 MiB.

- Set A: 25 MR patients (001–027 minus 009, 019), 25 right + 24 `_L` (no 006_L).
- a2a1f26 shipped 25 right anatomies, 425 files.

### 8.2 Per-anatomy folder contract

| File | A v1 | A v2 | A v3 | B v1 | B v2 | B v3 | Read by env? |
|---|---|---|---|---|---|---|---|
| `Centrelines_comb/*.mrk.json` | 16 | 16 | 16 | 17 (+RECA) | 17 | 17 | **yes** |
| `vessel_architecture_collision.obj` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| `mesh_v2.json` | | ✓ | | | ✓ | | no |
| `mesh_v3.json` | | | ✓ | | | ✓ | no |
| `graft_xform.json` | | | ✓ | | | | no (v3 bake) |
| `provenance.json` | | | | ✓ (21 keys) | ✓ (201×21, 22×24 with siphon_*) | ✓ (25 keys: +siphon_*, +xform) | no |
| `collision_full.vtp` | | gitignored, absent | gitignored, absent | | absent | absent | no |

**Centerline content:**
- 14 of the host‑derived files parse identically to the host in every anatomy.
- The RCCA is always rewritten, as single‑line JSON with sequential labels.
- The RVA is modified in A 6/49 (all versions) and in B 95 (v1) / 100 (v2/v3).
- In v2/v3 the RCCA radius minimum is exactly 1.000 in both sets (v1: A 0.525, B 0.494).

**Naming:** Set B names match `^case_[kmw]_\d{3}b?_(left|right)__topcow_mr_\d{3}(_L)?$`; the only "b" is w_018b_left. Lowers are never mirrored.

### 8.3 Git and line endings
- `core.autocrlf=true` here (system gitconfig).
- `.gitattributes` marks all seven anatomy trees **`-text`**, so bytes are preserved. There is also `*.sh eol=lf`.
- The blobs themselves have mixed endings:
  - host‑copied centerlines are stored as **CRLF** blobs (copied by `shutil.copy2` from the Windows working copy);
  - **A v1's 25 right folders are LF** copies;
  - grafter‑written branches are single‑line;
  - OBJ and JSON reports are LF;
  - BUILD_v*.json is CRLF.
- The host dir and the 4 top‑level manifests (`label_necks`, `pairing`, `lower_manifest`, `excluded_severed`) have **no attribute**: LF blobs, CRLF in this worktree.
- **Byte identity to the host holds only after CR stripping** (0 content differences across all folders).
- Every grafted OBJ is triangle‑only with no vn/vt, a `meshio 5.3.5` header (timestamped, so every re‑bake changes the bytes), and is unique by md5.
- **Branch parity:** 18_p2 has identical data trees (synced in a16750f: trees 51bf813, fe01250, baabce8, ddaf0f3), apart from `label_necks.json`. Its `.gitignore` lacks the v2/v3 negations; the files stay tracked only because they are already indexed.

---

## 9. History (dates -0400)

| When | Commit | What |
|---|---|---|
| 07‑14 | 37df84b | merge base |
| 08‑27 14:08 | **a2a1f26** | Set A v1: 25 right‑ICA anatomies (425 files), loader, check/validate tools. Also silently carries **all** of branch 16's RL work (§12) and 18 figure scripts; .gitmodules deleted |
| 08‑27 16:21 | 569c644 | TOPBRAIN_PIPELINE.md (516 lines; never updated for v2/v3 at HEAD) |
| 08‑27 eve | f7d5b03, 80e599a | figure framing revert; raw‑label figures; retracts "no label dictionary" (label 4 = R‑ICA) |
| 08‑28 10:48 | [18_p2] c7b60f6 | imports the 25‑anatomy v1 set; adds the `--topbrain` flags, TOPBRAIN_ANATOMY_AUDIT.md, launch_rcca_topbrain_v1.sh |
| 08‑29 15:21 | **a0cae4c** | +24 mirrored left ICAs → 49 (47 usable; exclude mr_015 and 003_L); TopCoW ruled out; remote_zip.py |
| 08‑29 | 64140f8, b745af9, 19852bf 17:41 | per‑patient figures; VMR 0248 figures; VMR search closed |
| 08‑30 13:10 | **5335a00** | carotid_tools; "231 validated" (no geometry) |
| 08‑30 14:02 | 4e0b538 | figure_carotid_anatomies.py |
| 08‑30 15:28 | [18_p2] 3d130c7 | imports the 49‑anatomy Set A and 5335a00 tooling |
| 08‑31 22:23 | (bake) | first Set B v1 meshes (header timestamps) |
| 09‑01 11:11 | **a898eb2** | eight silent defects fixed; DEFECTIVE_LOWERS; component‑aware validator |
| 09‑01 11:14 | **430bc76** | Set B v1 geometry committed: 216 (4,104 files), extended/, pairing (237), manifest (138), excluded_severed (5) |
| 09‑01 12:34 | **ba3715b** | FUSE_BAND_MM 0.35; deletes w_003_right__mr_017 → 215; 19 geometry re‑bakes, 196 header‑only |
| 09‑01 12:46 / 13:08 | 0134069, 4cc6b35 | CAROTID_VERIFICATION_HISTORY.md; fork‑chirality record |
| 09‑01 13:07/13:11 | [18_p2] ceadf86, 9b420ef | imports Set B at the ba3715b state (+ a stale mr_017 folder), then removes it; 216‑sweep QC scripts |
| 09‑03 22:26 | **033009a** | SDF mesher (v2), real‑surface union (v3), label_necks, meshed‑lumen check, grafter flags + xform, siphon floor |
| 09‑03 22:27 | **3eb0d2a** | Set A v2 + v3 (49 each) |
| 09‑03 22:30 | **6002fcc** | Set B v2 + v3 (223 each) |
| 09‑05 | [18_p2] a16750f, b2444e7/6cb1294 | v2/v3 imported byte‑identically; REPRO_topbrain_v3.md; TopBrain v3 run (41/8) |
| 09‑10 18:15 | 1597777 | `presentation/` mount in run_container.sh (the folder is in no commit) |
| 09‑11 | [18_p2] 4eef6bb | carotid v3 run (153/16/54) |
| 09‑24 15:06 | **f292143** | docs only: adds "Current state (v2/v3)" to TOPBRAIN_PIPELINE, CAROTID_THREE_SOURCE, CAROTID_VERIFICATION_HISTORY; TopBrain‑only siphon source; 8‑extra = 5 severed + 3. Independently verified: 46/79 confirmed, 26 partly, 5 refuted (stenosis 30/47 and its stated method; "geometry/rise/kink unchanged") — §13 |

**Reversals:**
- DISTAL_TRIM 4 → 0.
- ROUTE_MIN_R 1.60 → 1.0.
- The v1 decision not to floor siphons (excluded_severed reason) was reversed by the v2 `--siphon-min-r`.
- Set A v1 exclusions (015, 003_L; later 013/014/015/025) were dropped in v2/v3.
- 20k triangles replaced the 3.7k budget.

---

## 10. Traps and gotchas

| # | Trap | Consequence | Avoid by |
|---|---|---|---|
| 1 | Applying v1 defect verdicts (TOPBRAIN_ANATOMY_AUDIT, HANDOFF 9.6/11.6, TOPBRAIN_PIPELINE "Known failures", excluded_severed.json, nav_quality.json) to v2/v3 | good anatomies dropped (all 49 v2/v3 A and all 223 B pass) | tag every verdict with its version |
| 2 | Omitting `--topbrain_exclude` on v2/v3 | right 013/014/015 silently dropped while their `_L` twins train | pass `__none__` or an explicit list |
| 3 | Eval defaults (`eval_anatomies.py`, `launch_eval_topbrain.sh`) | evaluates **v1 meshes**, minus the v1 trio, at 40 mm, change_every 2 | pass `--topbrain_dir`, `--topbrain_only`, `--target_min_arclength_mm 133`, `--change_every 1` |
| 4 | Carotid replay launcher omits the target flag | replays at 40 mm while training used 133 | add the flag to EXTRA_FLAGS |
| 5 | Treating 133 mm as a seam for Set B | Set B leaves the host at 20–66 mm; 133 only places targets past the ECA fork | — |
| 6 | `DEFECTIVE_SIPHONS` in the frozen pairing | right 013/014/015/025 absent from Set B in **all** versions, though Set A v2/v3 uses them | re‑solve the pairing if needed |
| 7 | "v2" in `carotid_data/v2/carotid_bifurcation_database` | it is the Zenodo DB release, not mesher v2 | — |
| 8 | 18_p2's `graft_three.py` (ba3715b version, 767 lines) | cannot regenerate Set B v2/v3 (no flags, siphon floor or xform) | use branch 16's (+52/−3) |
| 9 | V2_BUILD_PLAN.md:99 recipe | omits `--siphon-min-r`, so the unfloored first‑bake failures come back (13 per BUILD_v2/G5; 7/22 in a cruder re‑bake) | use BUILD_v2.json flags, or f292143's CAROTID_THREE_SOURCE.md recipe, which includes all five |
| 10 | `provenance.xform.lower.extend_mm` / `ica_real_mm` | always 0 / the extended length (114 affected) | join on `pairing.json` `extended` or `extended/*.json` `ica_mm_before` |
| 11 | Only v3 provenance has `xform`; 201/223 v2 lack siphon_* | `bake_meshes_v3.py` KeyError on v1/v2 folders | use v3 provenance (centerlines are identical) |
| 12 | Rebuilding v3 or re‑cutting budgets from this checkout | real surfaces, Zenodo STLs, raw TopBrain and `collision_full.vtp` are all absent | re‑download |
| 13 | Rebuilding siphons from TopBrain record 21972006 topaneu36class | label 4 = C6‑C7 only (silent stub) | use `labelsTr_topbrain_v1_mr` or record 16878417 |
| 14 | Citing the concept DOI 10.5281/zenodo.7634643 | resolves to the latest version; v1.0.0 has different names | cite 10.5281/zenodo.10695923 |
| 15 | Calling the host "VMR 0105" | wrong dataset and licence | cite Karstensen et al. 2025 / stEVE_bench |
| 16 | `run_container.sh` | hard‑coded foreign paths; mounts data read‑write | use a read‑only copy |
| 17 | Byte‑comparing centerlines across machines | CRLF/LF differences (A v1 right folders are LF) | compare parsed JSON |
| 18 | `mesh_v*.json` `open_edges` | = boundary + non‑manifold; check_anatomies counts boundary only | — |
| 19 | `lumen_min_mm` | a radius; threshold 0.65 | — |
| 20 | check_anatomies meshed‑lumen | same function as the bake (not independent); fails open on exception; RCCA only (no bridge, no ECA/LCCA/RVA) | — |
| 21 | `--sofa` harness | mixed‑frame target arc; 40 steps never reach the siphon; random rotation; "OK" = no exception | ≥110–150 steps; seed `_rng`; report tip branch and arc |
| 22 | Genus 1 read as a fusion | it is the host vertebrobasilar ring | subtract the VB ring |
| 23 | v3 assumed safer than v2 | v3 adds fusion handles (A 9, B 64), ECA pockets, outlet lips, and a device‑passable RCCA‑RVA merge (w_047_left__018_L, a TRAIN anatomy); v3 min lumen lower than v2 in 49 anatomies | gate v3 on genus and check8 wall |
| 24 | RECA classify snap | ECA‑band tips count as RCCA; 3 disagreeing `final_branch` definitions | fix classify / use state‑machine fields consistently |
| 25 | env5 anatomy hash | identical for v2 and v3 | log the dir |
| 26 | Fresh eval env, first episode | construction anatomy, not the seed's | — |
| 27 | Stale docs: CAROTID_THREE_SOURCE.md table (77 RVA deflections, 5 re‑entry, clearance +0.10, "216", "six severed", inlet trim "50 of 220"); match_sections docstring caps; graft_three host_cut comments; launcher header comments ("25 built, 22 retained, 18/4"); `mask_to_surface.py` docstring (label 8); REPRO_topbrain_v3 "non‑remeshed / different density"; report_versions "3,584 triangles"; MESHING "eight fusing rejections"; "40–74% survive" | wrong numbers quoted | regenerate from provenance/data |
| 28 | `label_necks.json` | read by no build script; branch 16 only (breaks `reports/make/figures.py` on 18_p2) | — |
| 29 | Figure scripts | `figure_topbrain_pairs.py` re‑meshes with the v1 mesher; `_v2` defaults to the v1 root at 0.6 mm; "(ICA extended)" tag fires on ~210/215 panels | pass roots explicitly |
| 30 | Quoting f292143's stenosis figures (56 %, 30 ≥50 %, 47 ≥40 %) | 30/47 are a paper calculation, not the shipped anatomy (§4.5) | quote 24–25 ≥50 % and 33–35 ≥40 %, with the definition named |
| 31 | f292143's "geometry unchanged in v2/v3" / "rise, kink and junction unchanged" (Set A) | false: only the junction bend is unchanged | use the v2/v3 numbers in §3 Route statistics |
| 32 | f292143's "223/223 on every check" and SOFA "9/9, 10/10 with targets past the seam" | static checks only (no genus test); SOFA tips move 39–56 mm and never reach the 130 mm seam | treat as "loads and steps", not traversal |
| 33 | `recut_obj.py` on a checkout | re‑cuts 0 meshes (no `collision_full.vtp`), writes `mesh_v2.json` only, skips v3's step‑up | re‑bake v3 with `bake_meshes_v3.py --force` |
| 34 | `graft_three.py --shard` (CAROTID_THREE_SOURCE.md recipe text) | no such flag; the grafter shards with `--only LO:HI` | — |

---

## 11. Open questions and unresolved inconsistencies

1. **Fork‑chirality evidence.** The cited 3/8 same‑patient control and the 77/109 / 16/106 counts come from uncommitted code. Committed ck13b/c/d give 6–7/8 opposite. Is leaving the lower unmirrored justified empirically?
2. **Siphon self‑contact handles in v3.** Real anatomy, or created by the capsule union and 0.45 mm grid? The source surfaces are absent.
3. **v3 fusion and pocket mechanism.** The capsule versus tube‑radius gates are inference. Do policies exploit the w_047_left__018_L merge or the w_050_left ECA pocket? Why do v3 mr_015 and mr_008 stall at ~194–197 mm?
4. **The 17 v1 and 14 v2/v3 grafter rejections.** Their reasons are reconstruction‑only; no log was saved.
5. **k/m/w cohort and "b" semantics.** Ask the depositors.
6. **Host patient.** Institution and ethics are not recorded. The data licence beyond eve_bench MIT is unclear.
7. **The v2/v3 SOFA rollout claims** (6/6, 9/9, 10/10) are weak evidence (§6.3). A proper traversal test has not been run on all anatomies.
8. **Stenosis restoration — now explained, docs still inconsistent.** BUILD_v2 "~60%" is definition D1 (60.4%, 24 ≥50%); f292143's 56% is D3 (25 ≥50%), but its 30/47 is the paper calculation D2 (§4.5). 6002fcc's "40–74% survive" is false. The docs need one named definition and a committed measuring script.
15. **The v2 first‑bake count (13).** BUILD_v2 and G5's careful reconstruction give 13; the f292143 verification's cruder re‑bake gives 7/22 (§6.7).
16. **`provenance.xform.lower.extend_mm` / `ica_real_mm` (guide (k)).** One verifier read `extend_mm = 0` as "no extension"; G1 says the fields misrecord a real extension for 114 anatomies. Neither re‑derived the geometry; unresolved.
9. **Why the host collision surface is a VTK‑split soup with a dropped quad in training host tests.** Is that intended?
10. **v2/v3 check_anatomies.** Now re‑run (all pass), but no committed log exists. v3 report schemas come from mixed baker revisions, so a re‑bake will not be byte‑identical.
11. **A v2 bake seconds** (median 94 vs 25.7 for B). Unexplained.
12. **HANDOFF 9.1 "ramp UP to 2.25–2.90"** vs measured 1.43–3.09 with ramps down in 16/49. Differing definition?
13. **Attribution.** There is no NOTICE for a public repo redistributing CC‑BY and USZ‑licensed derivatives.
14. **Documented numbers that could not be reproduced:**
    - mr_013 "5 pts / 1.37 mm" (raw re‑measure: 0.134 mm);
    - mr_015 "23 pts / 7.19" (15 / 6.87 exact);
    - the 102.5 mm radius departure;
    - "19/25" as a Set A pair count (only 24 `_L`).

---

## 12. Other branch-16 code (brief)

- **All non‑dataset RL work came in with a2a1f26**, and the commit message does not mention it. Docs date it 2026‑07‑16..24. Contents:
  - procedural‑RCCA experiments v3a → v3c → v3c2 → v3c3 (`launch_rcca_procedural_v3*.sh`, harvest v3c, `verify_seed.sh`);
  - forensics: FORENSIC_RCCA_V3A.md (plateau at 49–52/98; wedge at arc 158 mm, r=2.0); FORENSIC_V3A_PATH_FORWARD.md;
  - REWARD_PAIR_FIX_HANDOFF.md, V3C_PLAN.md (a reward plan, **not** mesh v3).
- **Reward pair:**
  - tip‑average progress: `--progress_tip_mode avg --avg_gw_weight 0.5` (`arclengthprogress.py`, `polyline.point_at_inserted_length`; off‑path tracker frozen to stop a +0.125/cycle pump);
  - catheter‑slack potential: `--cath_slack_coef 0.5` (`buckle_reward.cath_slack_potential`, fed by `env5._compute_cath_slack_mm`).
- **Branch 16 code is the uncapped v3c2 form:** dead band 15, slope 150, clip 1000, so phi ≥ −6.67. **18_p2 ships the capped form** (b3254d9, phi ∈ [−1, 0]).
  - Identical up to 165 mm of slack. At 708 mm: −2.31 vs −0.50 at coef 0.5.
  - The handoff doc says "use the capped form" (the snippet is at :260‑277). The argparse help is stale.
  - So `launch_rcca_procedural_v3c.sh` at HEAD reproduces v3c2, not run 1 (62.2% eval1). The reward‑version guard does not stamp this constant.
  - 18_p2's `tests/test_v3c_reward.py` would fail against branch 16.
- **Reward‑version stamps:** four fields (buckle, cath_slack, tip_mode, avg_gw) in the experience cache, `replay_state.npz` and a `--resume` guard. The last two exist on branch 16 only.
- **Trainer levers** (`sac.py`, branch 16 only):
  - `--contact_mean_penalty` (AWAC‑only; contact from privileged flat index 103);
  - `--q_target_floor` (any algo);
  - `--awac_mode_adv_norm` (inert unless tau > 0; off everywhere).
  - v3c3 = v3c2 + penalty 0.05 + floor −10. No result is recorded.
- **Hygiene:** `.gitattributes` (`*.sh eol=lf`, `-text` data trees); `.gitignore` data rules; stale `.gitmodules` removed (still on 18_p2). The branch‑16 trainer has **no `--topbrain`**, and dataset validation never runs BenchEnv5.
- **Figures and reports:**
  - 23 `monitoring/figure_*.py` (17 procedural, 6 dataset);
  - `reports/`: 19 PNGs, 13 of them from `reports/make/figures.py`; the two PDFs are gitignored and absent;
  - `saved/mesher_probe/`: probe scripts, report.txt, compare_v1_v2/v2_v3.txt, lumen_v1.json, label_qa.txt, v3 rebake name logs.
---

## 13. Verification of `f292143` (2026‑09‑24)

`f292143` brings TOPBRAIN_PIPELINE.md, CAROTID_THREE_SOURCE.md and CAROTID_VERIFICATION_HISTORY.md up to v2/v3, and touches MESHING_PIPELINE_ANALYSIS.md and V2_BUILD_PLAN.md. A separate workflow extracted its 79 added or changed claims. Each was checked by an independent verifier that never saw this guide's numbers. The 16 claims that conflicted with this guide also got an adversarial second verifier and an adjudicator that ran the decisive measurement. Full report, with replacement text for every wrong passage: `VERIFY_f292143_report.md`.

| Verdict | Count |
|---|---|
| confirmed | 46 |
| partially | 26 |
| refuted | 5 |
| unverifiable | 2 (TopBrain/TopCoW facts about external data absent here) |

**Reliable:** the set counts (A 49/49/49, B 215/223/223), the mesher descriptions, the constants and floors per version, v2 = v3 centerlines byte‑identical in both sets, navigability (A 22 → 49, B 71 → 223), every rebuild flag (including `--siphon-min-r 1.0` in the Set B recipe), and the loader behaviour.

**Refuted:**
- S3/S4/S5 — stenosis "30 ≥50 %, 47 ≥40 %, measured the way build_manifest measures it" (§4.5).
- K9/L3 — Set A "geometry unchanged in v2/v3" / "rise, kink and junction unchanged from v1". Only the junction bend is unchanged.

**Partly right, mostly causal overreach:** why the eight were gained and why 14 are unbuilt (§4.7); necks as the v1 failure site and the floor as what keeps them open (§3 stage 0); "223/223 on every check" (no genus test, §5.3); SOFA "targets past the seam" (tips never reach it, §6.3); "rebuilds keep every graft fix; only mesher constants change" (Set A v2 also got the a898eb2 RVA fix, shrinking four repairs; FUSE_BAND kept, siphon floor new); `collision_full.vtp` / `recut_obj.py` (absent on any checkout); v1 "47 usable" (22/49 under the current checker; 27 fail, 25 on the meshed‑lumen test alone; the worst v1 meshed lumen is 0.011 mm, `case_w_008_right__topcow_mr_023_L`).

**Where this guide was corrected** (5 rulings "partly", none "wrong"): the three w_014_right pairs (mechanism right; the re‑entry test is itself a fusing‑band test); which of the 14 unbuilt sits inside the band (w_016_right__015_L, not w_003_right__mr_017); the V2_BUILD_PLAN omission is in V2_BUILD_PLAN.md:99, not in f292143; the "13 return" corollary is disputed (§6.7); the origin of "~60 %" is consistent with D1 but not proven; "17 tip‑crop donors" is not reproduced (it is 5 pairs at ≥50 %, 14 at ≥40 %).

**Other machine's pre‑commit summary:** mostly right. Wrong or stale: "donors 3–74 %" (the manifest is 2.5–86 %; 74 % is the paired‑lower maximum), "eight fusing rejections" (5 severed + 3 ECA re‑entry), and "host_cut 15–54 mm" (15–59 mm). "~60 %" is right under the build_manifest definition. COSTA is confirmed undocumented.
