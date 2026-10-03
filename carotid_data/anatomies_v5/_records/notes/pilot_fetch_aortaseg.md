# pilot_fetch_aortaseg: AortaSeg-60 cases for the thin-slice VA pilot

Date 2026-10-01.

**Task.** Pick 5 of the 28 unique head-first arterial AortaSeg-60 cases (Zenodo 18147026), preferring "Young" plus a mix
of the other groups. Fetch them, verify the cranial end, and crop each from about 20 mm below the aortic arch to the
vertex, keeping raw HU and the native voxel. For each case, write `ct.nii.gz`, `meta.json`, and neck MIPs that confirm
the arterial phase and both VAs. Reject and replace cases that are venous, non-contrast, truncated or motion-corrupted.

Machine-readable results: `results/pilot_fetch_aortaseg.json` and `results/pilot_fetch_aortaseg_screen.json` (+ `.png`).

## Answer

1. **AortaSeg-60 cannot supply 5 pilot cases. It supplies exactly one, Dissec_01.** The other 27 "head-first" arterial
   volumes start at the lower neck, not at the vertex.
   - **First-slice screen.** The first slices of all 28 were decompressed over HTTP and read with real HU (stored −
     1024). See `results/pilot_fetch_aortaseg_screen.png`.
     - 27 show a neck section: a cervical vertebra, with the trachea or larynx in most of them. Body area is 61–149 cm².
       The largest bone component encloses at most 21% of the body.
     - Only Dissec_01 shows a cranium: the skull ring encloses 87% of the section, with brain inside.
   - **Full downloads.** The full volumes of 4 head-first cases (Young_09, Young_10, Old_02, Anev_02) and of both 0.6 mm
     feet-first cases (Young_06, Dissec_06) were fetched and processed. In every one the scan ends 43–77 mm above
     the arch apex (table below). That is around C6–C7. It holds V1 at most, and no skull base.
   - **The rest.** The six other feet-first volumes have 1.25 mm slices, above the 1.15 mm gate, and were not fetched.
     Of the duplicates Anev_05/08/10 and Anev_07, none was fetched.
   - **Earlier survey and plan.** `va_pool/other_ctas.md` ("28 unique arterial cases start at the skull vertex") and
     `donor_pool_plan.md` (AortaSeg ≈ 40 sides / 16 pairs) are wrong. The survey's small round first slices were neck
     sections. On a −200..800 window over stored values with the +1024 offset, soft tissue saturates white, so the
     trachea looked like a "dark spot in the skull". The open pool loses about 27 subjects.
   - **Recommendation.** Move the 4 remaining AortaSeg pilot slots to Song 2024 or the 172-series set.

2. **Dissec_01 is prepared and accepted provisionally.** File:
   `D:\Arjun\neve\carotid_data\va_sources\thin_slice\aortaseg\Dissec_01\ct.nii.gz`.
   - **Grid and orientation.**
     - 512 × 512 × 415 at 0.779 × 0.779 × 0.625 mm, the native grid, int16 raw HU.
     - LAS, k increasing I→S, non-oblique.
   - **Crop.**
     - It runs 48.8 mm below the arch apex up to the top of the scan, 210 mm above the apex (259 mm in all).
     - The bottom slice cuts the aorta into two limbs, as Arterial needs.
     - The scan starts mid-cranium, so the vertex is not in it. The foramen magnum (C1 top +147.5 mm) and about 60 mm of
       posterior fossa are.
   - **Phase: arterial, late (mixed).**
     - Arteries: CCA R/L 273 / 256 HU (TotalSegmentator masks), 289 / 284 by hand ROI. Subclavian R/L 274 / 252.
       Aorta 226 HU, with a poorly opacified false lumen (p10 72 HU).
     - Injection-side veins are dense: right IJV 433, right BCV 348.
     - The contralateral left BCV reads 153 HU. That is under the plan's draft 200 HU gate, but far above a clean CTA
       (49 HU in the TotalSegmentator census). The presumed left IJV reads 202–223 HU.
     - Paired paraspinal vessels posterior to the transverse processes read 274–312 HU. These are presumably deep
       cervical veins or plexus. They are a trace-contamination risk.
   - **VAs: present on both sides on axial slices, but continuity is not confirmed.**
     - Lumen-like blobs and foramen contents read 240–400 HU at C6–C2 on both sides.
     - V2 cannot be isolated on MIPs. The lumen (~270 HU) lies inside the marrow range (200–350 HU) at 0.78 mm in-plane,
       and the neck is strongly flexed.
     - The Arterial + trace step has to settle it.
   - **Other flags.**
     - Aortic dissection. The branches show no low-attenuation false lumen (eroded-mask p10 191–220 HU), but a flap
       with both lumina opacified was not excluded.
     - The in-plane voxel is about 2× VMR's and TaG-Net's (0.40 mm), so expect less gradeable wall relief.
     - FOV 399 mm. ECG leads in the FOV.
     - No motion seen.

3. **Two traps in the release, handled per case and recorded in each `prep.json`.**
   - **HU.** Stored values have the −1024 rescale intercept dropped, with no `scl_inter`: air ≈ 0, fat ≈ 925, soft
     tissue ≈ 1060, padding −2000.
     - Any tool that reads these images as HU sees fat at +900.
     - `ct.nii.gz` holds stored − 1024. The padding is −3024.
   - **Slice direction.** Every arterial header has a +z step (+0.6 or +0.625 mm) with the origin at the k = 0
     slice.
     - In the 32 volumes stored head end first, k = 0 is the head end, so the header puts the head at the bottom. A
       naive RAS reorientation turns the patient upside down.
     - The release's own aorta masks are always stored feet end first, with a consistent +z affine. For the 5
       head-first downloads they run in the reverse k order to the image: the median HU under the mask is the aorta's
       (220–430 HU) on the k-reversed image and soft tissue (−31 to +58 HU) in the same order. For the 2 feet-first
       downloads, mask and image share the order and the image header is correct. Per-case values:
       `results/pilot_fetch_aortaseg.json` → `downloads.*.seg_alignment`.
     - Fix: reverse the data and take the step as +|dz|, keeping the header origin as the head-end slice. The result
       reproduces the mask affine exactly.
     - Left/right and anterior/posterior were then checked by anatomy:
       - Dissec_01: descending aorta 41.7 mm left of the trachea; abdominal soft tissue R/L 1.21; spine 66 mm posterior
         of the body centre; abdominal aorta 67 mm anterior of the spine.
       - The same checks pass on all 6 rejected downloads, including the 2 stored feet-first. On Young_10, a first
         version of the rule referenced the descending aorta to the body centre and failed. The final rule uses the
         trachea and passes it. Young_10's liver-side ratio, 1.11, is weak but does not contradict.

## Case table

| case | group | in-plane / slice (mm) | how checked | scan top above the arch apex | decision |
|---|---|---|---|---|---|
| **Dissec_01** | dissection | 0.779 / 0.625 | full download (CRC ok), prep, TotalSegmentator total_v3, by eye | **210 mm** (C1 top +147.5) | **accepted (provisional)** |
| Young_09 | young | 0.420 / 0.6 | full download, prep | 77 mm | rejected: truncated, no skull base |
| Young_10 | young | 0.449 / 0.6 | full download, prep | 43 mm | rejected: truncated |
| Old_02 | old | 0.486 / 0.6 | full download, prep | 51 mm | rejected: truncated |
| Anev_02 | aneurysm | 0.422 / 0.6 | full download, prep | 54 mm | rejected: truncated |
| Young_06 | young, stored feet-first | 0.393 / 0.6 | full download, prep | 72 mm | rejected: truncated |
| Dissec_06 | dissection, stored feet-first | 0.455 / 0.6 | full download, prep | 47 mm | rejected: truncated |
| the other 23 of the 28 (Dissec_08 included) | all | 0.42–0.82 / 0.6 | first slice over HTTP | neck section at the top slice | not usable: truncated |
| Young_05/08, Old_04/09, Dissec_07/09 | stored feet-first | ~0.44–0.58 / 1.25 | header | – | excluded: 1.25 mm slices (> 1.15 mm gate) |

The first choice was Young_09, Young_10, Old_02, Anev_02 and Dissec_08, the finest in-plane voxel per group. The
first 4 downloaded completely before the screen showed that they start at the neck. The download was stopped before
Dissec_08 began; its first slice is also a neck section.

## Method

- **Fetch** (`pilot_fetch_aortaseg.py fetch`).
  - zipfile over `va_pool/rzip.HttpFile`, which is unchanged. One member at a time, with CRC32 checked while reading.
  - About 1.1–2.3 MB/s from Zenodo, 2–4.5 min per 260–290 MB image.
  - Log: `thin_slice/aortaseg/_orig/fetch_log.jsonl`.
- **Screen** (`... screen`). Decompress the first 3 slices of each member; classify the middle one. About 1–2 MB per
  case. The automatic class in the JSON is crude (21 "air inside body", 7 "other"). The decision rests on the
  montage, read by eye, plus body area and skull-ring fill.
- **Prep** (`... prep`), in neve-build-stagea.
  - HU offset detected from the air peak (near 0 vs near −1000) and confirmed by the fat or soft-tissue mode.
  - Seg ↔ image k order taken from the HU under the release aorta mask.
  - Head end decided by which end of the aorta mask has the arch's two-limb zone. Body area is recorded but misleads
    on neck-start volumes.
  - L/R and A/P anatomy checks.
  - Crop: bottom = top of the two-limb zone − 20 mm; top = scan top + 5 mm.
  - Reject if no skull base (bone area > 15 cm² for 10 mm, ≥ 90 mm above the apex) or if the top is < 150 mm above
    the apex.
  - Figures.
  - An identical existing `ct.nii.gz` is kept, so its hard link survives a re-run.
- **TotalSegmentator.** total_v3 on the Dissec_01 crop, via the unchanged `tag_totalseg_run.sh`.
  - Run under the GPU lock (acquired, then released at exit); 210 s, 5.1 GB.
  - Masks: `derived/thin_slice/aortaseg/Dissec_01/totalseg_v3/segmentations/`.
  - `ct.nii.gz` is an NTFS hard link there, so `arterial_ts_args.py` and `art_extract --ts` can use the dir as a
    TotalSegmentator subject.
  - Following `tag_totalseg.md`, a 1.5 mm copy would make `art_extract --ts` faster. It has not been made.
- **HU.** Taken in eroded TotalSegmentator masks (`pilot_fetch_aortaseg_look.py tsstats`), plus hand ROIs (`roi`).
  - The automatic CCA / subclavian picks in `prep.json` (`quick_look_hu`) are **unverified**. On Dissec_01 the CCA
    picks landed on laryngeal cartilage. `meta.json` uses the TotalSegmentator values.
- **VA checks** (`look`, `blobs`, `vacheck`, `foramina`, `cordmip` in `pilot_fetch_aortaseg_look.py`): by eye on
  zoomed axials, HU in the foramina, and MIPs in a spinal-cord-following frame.

## For the processing step (Dissec_01)

- **Arterial.**
  - Use `--fov-mm 220 --fov-center=<x>,<y>` with the neck centre in `meta.json` (`neck_centre_for_arterial_fov`).
  - No resampling is needed (0.78 mm).
  - Skull mask: `derived/.../totalseg_v3/segmentations/skull.nii.gz`, on the input grid.
  - The double-lumen descending aorta may confuse the arch start-point heuristic. Check the start point.
- **Trace.** Expect V2 bridges or wall leakage into the vertebral marrow, since the lumen and marrow are iso-attenuating,
  and paraspinal venous candidates. Require the V3-loop gate and the foramen band.
- **Wall.** At 0.78 mm in-plane, `ct_seg4`'s upper bone threshold must sit near 350–400 HU. The lumen p90 is about
  300 HU, against 450+ for VMR. Check the bone/lumen separation before grading.

## Files

- **Scripts** (new):
  - `v5_va/pilot_fetch_aortaseg.py`: fetch, screen, prep, meta.
  - `v5_va/pilot_fetch_aortaseg_look.py`: look, blobs, roi, tsstats, vacheck, foramina, cordmip.
  - No study tool was modified.
- **Results:** `v5_va/results/pilot_fetch_aortaseg.json`, `pilot_fetch_aortaseg_screen.json` / `.png`.
- **Data** (D:, gitignored):
  - `carotid_data/va_sources/thin_slice/aortaseg/`:
    - `PROVENANCE.md`;
    - `Dissec_01/`: ct, meta, MIPs. `ct.nii.gz` md5 is 454a07f05035bb858a89ea270b89ed89;
    - `_rejected/<case>/`: prep.json + MIPs, run log `_rejected/prep.log`;
    - `_screen/`: a copy of the first-slice screen;
    - `_source_meta/`: the Zenodo record JSON and the README;
    - `_orig/`: the release files, CRC-checked, 1.9 GB. The rejected ones can be deleted.
  - `carotid_data/va_sources/derived/thin_slice/aortaseg/Dissec_01/`: release aorta mask on the crop grid,
    TotalSegmentator total_v3, `prep.json`, `ts_hu.json`, `roi_manual.json`, `assessment.json`, `va_foramina*.json`,
    QC figures.
