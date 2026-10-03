# arterial_infra: the Arterial pipeline, runnable here

Status 2026-10-01. **Ready for batch use on CPU.** One run takes about 10 minutes (hybrid
segmentation, 0.5 mm grid, 12 threads) with a peak of 9 GB, or 14 GB with landmarks. There is also
an optional GPU mode (see the GPU section). The pipeline runs end to end and writes Arterial's own
outputs plus my summaries.

**What it gives for the VA.** On TotalSegmentator v3 1.5 mm CTAs, a correctly prepared run yields
VA centerlines labelled RVA/LVA. They start exactly on the TotalSegmentator subclavian mask (0.0 mm)
and hug the cervical vertebrae, median 2-3 mm from the TotalSegmentator vertebra masks, so they are
real V1-V2 (V3) vertebral arteries.

**What it does not give.** In 0 of 4 prepared runs did a VA reach the BA as one labelled path. Either
the distal VA and the BA land in a separate mask island, or VMTK stops the RVA at C2-C3. One VA
side was missing in 3 of 4 runs, even though the binary mask holds vessel voxels beside the spine on
both sides. **The labels are also brittle.** The same CT segmented with mirroring TTA (run G) lost
the proximal LVA branch and labelled the distal LVA "LICA", where run D had called it "LVA". Use
Arterial as the **segmenter plus proximal-VA labeller**. Expect to close the V3/V4-to-BA stretch, and
any missing side, with your own tracing on its mask.

## 1. What Arterial is (read from code @3ff58ab, not from the paper)

- Stages: nnU-Net v2 binary vessel mask → VMTK surface/network/centerlines per island → 3D U-Net
  landmarks (ICA/MCA, not needed for the VA) → GATv2 GNN labels each *centerline segment* with one of
  14 classes (`other, AA, BT, RCCA, LCCA, RSA, LSA, RVA, LVA, RICA, LICA, RECA, LECA, BA`) →
  features/access prediction (not needed).
- **The mask is binary.** There are no per-voxel artery labels. "RVA/LVA" exist only as labelled
  centerline segments (points plus VMTK maximum-inscribed-sphere radius) in
  `extracranial_vessels/segments_graph_pred.pickle`. A VA wall surface has to be cut out of
  `segmentation.nii.gz` around those centerlines. Arterial's own `segmentation.vtk/stl` is
  windowed-sinc smoothed and decimated 80-90%, so it is useless for wall detail.
- Models (plans.json): neck `extracranial_vessels/3d_lowres` at 1.245×1.159×1.245 mm (patch
  112×192×112); head `intracranial_vessels/3d_fullres` at 0.43×0.40×0.43 mm; head/neck split by
  TotalSegmentator `craniofacial_structures` at 0.5 mm iso (class 3 skull). Training CTAs:
  512×(861)×512 at ~0.43×0.43×0.4 mm, foreground HU median 391.
- Hard-coded input assumptions:
  - orientation must be RAS, LAS or LPS, with axis 2 = I→S (`aff2axcodes` checks);
  - spacing is read from `diag(affine)`, so the affine must be non-oblique;
  - raw HU;
  - the vessel bounding box must be 2.5e6-3.5e7 mm³ and the vessel volume ≥ 3.5e4 mm³ (crop neck-to-leg CTAs);
  - the **bottom slice must cut the aortic arch so that the ascending and descending aorta are two
    blobs** (`aortic_arch_endpoint_check`);
  - the start point is the endpoint nearest voxel (350·0.43/sp, 0, 0) of the LAS grid. That is about
    40 mm left of the FOV centre at its posterior edge, which holds only for a ~220 mm FOV centred on
    the neck;
  - islands smaller than 591 mm³ are dropped.
- With no GPU, upstream turns nnU-Net mirroring TTA on (8 passes) and calls CPU use "debugging only".

## 2. Install (done)

| item | where |
|---|---|
| weights | `D:\Arjun\neve\carotid_data\va_sources\arterial\models\` (1.2 GB, from Zenodo 22694951 `arterial-models-v1.tar.gz`; SHA256 f3eeae79…90d5 matches the published file). The archive and `.sha256` are kept in `weights_dl\`. Licence and provenance are in `..\arterial\LICENSES_AND_PROVENANCE.md`. TotalSegmentator `LICENSE`/`NOTICE` stay in `models\segmentation\totalsegmentator_mandible\`. |
| code (read-only reference) | `...\arterial\repo\` (git clone @3ff58ab) |
| image `neve-arterial:latest` (57bc8d21fc6b, 5.8 GB) | micromamba env **`arterial`**: py3.11.8, vmtk 1.5.0, vtk-base 9.2.6, numpy 1.26.4, torch 2.6.0+cpu, torch_geometric 2.6.1 + pyg_lib/scatter/sparse/cluster/spline_conv (+pt26cpu), nnunetv2 2.6.2, monai 1.4.0, torchio 0.20.4, cc3d 3.22.0, arterial (no-deps, editable, /opt/arterial). Run python as `micromamba run -n arterial python`. Dockerfile: `<sp>/v5_va/docker_arterial/Dockerfile`; full freeze in `pip_freeze_py311.txt`. |
| image `neve-arterial:py310` (66003f1ef503, 3.9 GB) | same pins in the py3.10 base env of neve-build-vmtk (`/opt/conda/bin/python`; Arterial installed with `--ignore-requires-python`, since it uses no 3.11-only syntax). `Dockerfile.py310`. Smoke runs B, C and the py3.10 parts used it; results match. |
| `neve-arterial-gpu` (another agent's) | FROM the **py310** image + torch cu124. The built image (4af1dcf4c899) is on 66003f1ef503 (py310), CUDA + PyG ops tested on the A5000; its Dockerfile now says `FROM neve-arterial:py310`, so a rebuild stays on the right base. |

Build note: the py3.11 `micromamba create` took about 35 min on this machine (it sat on "Linking
qt-main" for 20+ min under load). Killing the Git Bash `docker build` client (TaskStop) **does not
cancel the BuildKit build**; it finished later and re-tagged `neve-arterial`. The same holds for
`docker run`: stop your own containers with `docker stop <name>`.

## 3. How to call it

```bash
# Git Bash on the host. Paths: D:\Arjun\neve\..., /d/Arjun/neve/..., /repo/..., scratchpad or /sp/...
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
$SP/v5_va/run_arterial.sh <ct.nii.gz> <outdir> [options]
```
- Options (forwarded to `arterial_cpu_run.py`):
  - `--stages core|full|seg,cl,lm,vl,fe,ap`. The default `core` (seg, cl, vl) is all the VA needs.
  - `--fast`: lowres on the whole volume, with no head split. It loses the BA and intracranial detail.
  - `--skull-mask <TS skull.nii.gz on the input grid>`: head/neck split without the 0.5 mm
    craniofacial model, which costs a 27× upsampling on a 1.5 mm CT.
  - `--slicer ts|laplacian|mask`.
  - `--crop-k k0:k1` / `--crop-mm z0:z1`: S slab of the input.
  - `--fov-mm 220 --fov-center x,y`: in-plane crop, RAS mm.
  - `--resample-mm 0.5`: trilinear isotropic resample before Arterial.
  - `--tta` / `--no-tta`.
  - `--threads N`.
- Environment variables:
  - `ARTERIAL_CPUS` (12), `ARTERIAL_MEM` (20g), `ARTERIAL_IMAGE`.
  - `ARTERIAL_GPU` **defaults to 1 since 02:40** (`--gpus all`, image `neve-arterial-gpu`; mirroring
    TTA then defaults on, as upstream). `ARTERIAL_GPU=0` restores the CPU path in `neve-arterial`.
    CUDA/NVIDIA lines are no longer filtered out of `arterial.log`, so a GPU out-of-memory error shows.
- **For a TotalSegmentator v3 subject**, get the arguments from
  `arterial_ts_args.py <ts_subject_dir>`. It chooses `--crop-k` inside the two-limb arch range up to
  the skull top, `--fov-mm 220` centred on the trachea when the FOV is wider than about 240 mm,
  `--skull-mask` and `--resample-mm 0.5`. It returns `ok:false` when the arch never shows two limbs.
  Example:
  `docker run --rm -v "D:\\Arjun\\neve:/repo:ro" -v "$SPW:/sp" neve-arterial:py310 /opt/conda/bin/python /sp/v5_va/arterial_ts_args.py /repo/carotid_data/va_sources/totalseg_v3/s0497`
  → `--crop-k 6:240 --fov-mm 220 --fov-center 6.4,151.6 --skull-mask /repo/.../skull.nii.gz --resample-mm 0.5`.
- One Arterial container at a time: the machine is shared.

**Outputs in `<outdir>`:**
- `cta.nii.gz`: the LAS, cropped, resampled CT that was actually processed, about 136 MB at 0.5 mm.
  You may delete it, because `prep.json` lets you rebuild it.
- `prep.json`:
  - `arterial_affine`;
  - `arterial_to_input_world`, the 4×4 back to the source CT world (identity unless the source was oblique);
  - the crops.
- `run_report.json`: per-stage seconds, per-stage and overall peak RAM (cgroup), errors and the file list.
- `extracranial_vessels/`, Arterial's own outputs:
  - `segmentation.nii.gz`;
  - `segmentations/`;
  - `centerlines/`;
  - `branch_model.vtk`;
  - `centerline_segments_array.npy`;
  - `segments_graph(_pred).pickle/.png`;
  - plus `landmarks.json`, `local_graph.pickle`, `supersegments/` and `access_prediction/` when
    `full` is run.
- `labelled_centerlines.json`: every labelled segment with `points_ras_mm` (world of `cta.nii.gz`),
  `points_src_world_mm` and `radius_mm`.
- `va_summary.json`:
  - per-class length, S range and median radius;
  - per-VA components, their neighbour classes, `touches_own_subclavian`, `touches_BA` and
    `continuous_SA_VA_BA_labelled_path`;
  - a frame check.
- `arterial.log`.

**Coordinates.** `centerline_segments_array` and the graph store points relative to the image's
min ("LPI") corner, so `world = p + lpi_corner`. The summary adds the corner back. The frame check
puts 99.4-100% of centerline points inside the mask, against 19-28% if x were mirrored, so the
frame is verified.

Helper scripts in `<sp>/v5_va/`:
- `arterial_ts_check.py <case> <ts_dir>`: VA origin distance to the TotalSegmentator subclavian
  mask, distance to the C1-C7 masks.
- `arterial_va_mask_probe.py <case> <ts_dir>`: is there mask beside the spine on each side, and is
  it in the main island.
- `arterial_dump_graph.py <case>`: segment table.
- `arterial_pick_ts.py`: two-limb-arch scan of all downloaded TotalSegmentator subjects, written to
  `results/arterial_ts_fov_scan.json`.
- `arterial_collect.py`: writes `results/arterial_smoke_summary.json`.

## 4. Smoke tests (CPU, 12 threads, mirroring off unless noted)

| run | input / prep | seg s | cl s | other s | total s | peak RAM | outcome |
|---|---|---|---|---|---|---|---|
| A s0152_fast | TS head CTA 233²×224 @1.5 mm, 350 mm FOV, `--fast` | 225 | 29 | – | 262 | 3.1 GB | **0 centerlines.** The arch is cut above its top (one blob), and the wide FOV made the start point pick a right-sided vessel. Unfixable by cropping. `arterial_ts_args` flags such cases (`ok:false`). |
| B s0289_fast | s0289 (145²×233 @1.5 mm, 217 mm FOV) native grid, `--fast` | 32 | 6 | vl 4 | 49 | 2.9 GB | 80 mask fragments. **No VA** labelled. |
| C s0289_fast_r05 | + `--resample-mm 0.5` (433²×697) | 69 | 29 | vl 1 | 127 | 6.1 GB | One island, 15 segments. **LVA 214 mm**, S −107→+39 (C1 top +23), origin 0.0 mm from TS left subclavian, median 3.0 mm from the vertebrae. No RVA, although the mask has a right-side tube at 100% of C levels in the main island. No BA (fast mode). |
| D s0289_hybrid_r05 | + hybrid, TS skull mask, `--stages full` | 393 | 42 | lm 126, fe 13, ap 2, prep 34 | 618 | 14.0 GB (lm). Core stages ≤ 8.9 GB | 29 segments in 3 islands. **RVA 120 mm** (RSA → S −3, about C2-C3); origin 0.0 mm from TS subclavian, median 2.1 mm from the vertebrae. **LVA 280 mm** (LSA → S +66), plus a separate-island LVA piece of 100 mm joined to a **BA** (3 segments, 101 mm, r 1.26 mm). The LVA end is about 10 mm from the BA island. Proximal LSA was mislabelled "BT". Landmarks: 3 of 6 ICA/MCA centerlines failed (irrelevant here). |
| E s0497_hybrid_r05_fov220 | TS head CTA 282 mm FOV → `arterial_ts_args`: `--crop-k 6:240 --fov-mm 220`, skull mask, 0.5 mm, core | 423 | 119 | vl 1, prep 40 | 595 | 12.0 GB | The first-model retry with a 5 mm bottom cut was needed. 6 islands. **LVA 146 mm** (LSA → S 248, C2), origin 0.0 mm from the subclavian, median 2.6 mm from the vertebrae. No RVA (right-side mask only 44% in the main island). No BA. |
| F s0289_fast_r05_tta | as C with `--tta`, seg only | 353 | – | – | – | 6.2 GB | Mirroring costs **5.1×** on CPU. Mask Dice vs C is **0.983**, so off is fine. |
| G s0289_hybrid_r05_gpu | as D on the **GPU** (`ARTERIAL_GPU=1`, mirroring on) | 141 | 73 | lm 284, vl 64, fe 90, ap 7 | 701 | 10.6 GB | Seg is 2.8× faster than CPU, and that is *with* 8× TTA. The CPU stages ran slower while other agents loaded the CPU. **RVA 121 mm** (RSA → S −3) again. **The proximal LVA is gone**: the LSA has no VA branch. The distal LVA plus BA island is present, but the distal LVA is labelled **LICA**. |

Note: A ran while other agents loaded the CPU, so its seg time is inflated; B's per-voxel rate is
about 7× faster.

**Answers to the brief's checks:**
- RVA/LVA labels are produced, with origin on RSA/LSA confirmed against the TotalSegmentator masks.
- Connection to the BA was *not* achieved as one labelled path in any run.
- CPU time is about 10 min per case in hybrid mode at 0.5 mm, or about 2 min in fast mode.
  120 TotalSegmentator CTAs is therefore about 20 h hybrid or about 4 h fast, single container.

## 5. GPU

`nvidia-smi` shows an **RTX A5000 (24 GB)**, and `docker run --gpus all neve-arterial-gpu` sees it
(`torch.cuda.is_available() == True`). This contradicts the "no GPU" assumption in the brief.
`run_arterial.sh` supports it via `ARTERIAL_GPU=1`; the wrapper then keeps upstream mirroring and
on-device prediction.

Measured in run G (s0289, hybrid, 0.5 mm, mirroring on):
- The segmentation stage takes 141 s on the GPU, against 393 s on 12 CPU threads without mirroring.
- Everything after segmentation is CPU-bound (VMTK) or tiny, so a full case is still about 5-12 min,
  depending on host load.
- The GPU is shared: about 5.9 GB was already in use by others during the run.

Use the GPU for segmentation-heavy batches. CPU alone is acceptable.

## 6. Recommendations for the TotalSegmentator batch / grafting agents

1. Pre-screen with `arterial_ts_args.py`. Head CTAs whose first slice is above the arch top (s0152,
   s0292, s0563 among the first downloads) and neck-only FOVs without a skull (for example s0065) are
   not worth running.
2. Always use `--resample-mm 0.5`. On the native 1.5 mm grid the mask shatters (80 components) and
   no VA survives. Use the TotalSegmentator skull mask for the split, and hybrid mode if the distal VA
   or BA matters.
3. Treat Arterial's VA labels as reliable for **origin and V1-V2**. For the V3/V4→BA stretch and for
   any missing side, trace on `segmentation.nii.gz` yourself. Seeds come from the labelled proximal VA
   and from the BA segment or the TotalSegmentator C1/foramen magnum. The mask usually contains both
   VAs beside the spine (`arterial_va_mask_probe.py`: 89-100% of C levels per side).
4. Natural-anatomy caveat: these masks are network outputs on a 1.5 mm CT, interpolated to 0.5 mm.
   Wall detail below about 1.5-3 mm is not resolved by the source (see `notes/ref.md`). Arterial's
   `segmentation.vtk` is smoothed and decimated, so never measure nooks on it.
5. Licences: everything derived from the Arterial weights is CC BY-NC (non-commercial). Keep
   `LICENSES_AND_PROVENANCE.md` with any derived data.
