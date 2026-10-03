# TotalSegmentator and NVIDIA (MONAI / VISTA3D / NV-Segment / Clara): public data or labels for the cervical vertebral arteries

Research date: 2026-09-27. Versions checked: TotalSegmentator 2.18.0 (PyPI, uploaded 2026-08-12); TotalSegmentator CT dataset v3.0.0 (Zenodo, 2026-09-10); TotalSegmentator MRI dataset v2.0.0 (2025-01-21); MONAI `vista3d` bundle 0.5.11; NV-Segment-CT and NV-Segment-CTMR v0.1 (NV-Segment-CTMR Hugging Face repo last modified 2026-03-31); MONAI `maisi_ct_generative` 1.0.2; MONAI `wholeBody_ct_segmentation` 0.2.7.

**Bottom line:** none of these resources has a vertebral-artery (VA) class or a released VA label, in any segment (V0–V4), for CT or MR. The closest they come:
- the proximal parent vessels: brachiocephalic trunk, left/right subclavian, left/right common carotid;
- the internal carotid (TotalSegmentator `headneck_bones_vessels`);
- about 120 openly licensed (CC-BY-4.0) CT angiograms in the TotalSegmentator v3 dataset whose field of view runs from the aortic arch to the skull vertex. The VAs are visible in these images but unlabelled, and the images are resampled to 1.5 mm isotropic.

Only one released model can produce a VA mask: VISTA3D / NV-Segment-CT, through its zero-shot point-click branch.

## Q1. Which TotalSegmentator tasks and classes cover neck vessels, is there a vertebral-artery class, and what are the licence terms?

### Takeaway
No TotalSegmentator task (CT or MR, open or licensed) has a vertebral-artery class. Neck-vessel coverage is limited to two tasks, both open under Apache-2.0:
- `total`: brachiocephalic trunk, left/right subclavian and left/right common carotid arteries, plus brachiocephalic veins.
- `headneck_bones_vessels`: left/right internal carotid arteries and internal jugular veins.

`total_mr` has no neck arteries at all.

### Cited Findings
- The TotalSegmentator README lists as "Openly available for any usage (Apache-2.0 license)": `total` (117 main classes), `total_mr` (50 classes), `head_glands_cavities`, `head_muscles`, `headneck_bones_vessels`, `headneck_muscles`, `vertebrae_*` and others. A separate list "Available with a license (free licenses available for non-commercial usage ...)" includes `heartchambers_highres`, `coronary_arteries`, `brain_structures`, `aortic_sinuses` and others. No neck-vessel task is in the licensed list. — [TotalSegmentator README](https://github.com/wasserth/TotalSegmentator/blob/master/README.md)
- `headneck_bones_vessels` classes: larynx_air, thyroid_cartilage, hyoid, cricoid_cartilage, zygomatic_arch_left/right, styloid_process_left/right, **internal_carotid_artery_right/left, internal_jugular_vein_right/left**. Cited paper: Cancers 2024;16(2):415. — [TotalSegmentator README](https://github.com/wasserth/TotalSegmentator/blob/master/README.md)
- `total` class IDs 54–60: brachiocephalic_trunk, subclavian_artery_right, subclavian_artery_left, common_carotid_artery_right, common_carotid_artery_left, brachiocephalic_vein_left, brachiocephalic_vein_right. — [map_to_binary.py](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py)
- Searching the full class map (`map_to_binary.py`, all tasks) for "vertebral" returns only vertebrae, vertebral bodies and prevertebral muscles, never an artery. `total_mr`'s 50 classes contain no carotid, subclavian or vertebral artery; its only arteries are the aorta and iliac arteries. — [map_to_binary.py](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py)
- The brachiocephalic trunk, common carotid and subclavian arteries (with spinal_cord, sternum, skull and others) were **new in TotalSegmentator v2**. In v1 (104 classes) they did not exist. — [improvements_in_v2.md](https://github.com/wasserth/TotalSegmentator/blob/master/resources/improvements_in_v2.md)
- Other head/neck tasks, all without arteries:
  - `head_glands_cavities`: eyes, lenses, optic nerves, parotid/submandibular glands, naso/oro/hypopharynx, nasal cavities, auditory canals, palates.
  - `headneck_muscles`: sternocleidomastoid, pharyngeal constrictors, trapezius, platysma, levator scapulae, anterior/middle/posterior scalenes, sternothyroid, thyrohyoid, prevertebral left/right.
  - `craniofacial_structures`: mandible, skull, sinuses, teeth.

  — [TotalSegmentator README](https://github.com/wasserth/TotalSegmentator/blob/master/README.md)
- The `headneck_*` models come from Heidelberg (UKHD), from a paper segmenting 71 structures for head-and-neck cancer target volumes. Its label guideline defines:
  - Internal carotid: "Cranial boundary: entry into the skull; Caudal boundary: separation of common carotid".
  - Common carotid: "Cranial boundary: until artery bifurcation; Caudal boundary: branching from brachiocephalic trunk".
  - Subclavian: "Lateral boundary: up to cranial boundary of sternum".

  The vertebral artery is not among the 71 structures; the only "vertebral" items in the paper are vertebral bodies and prevertebral muscles. — [Cancers 2024;16(2):415, full text via Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11154560/fullTextXML) ([MDPI landing page](https://www.mdpi.com/2072-6694/16/2/415))
- Output formats: one NIfTI per class by default, `--ml` for a single multilabel NIfTI, and `--output_type dicom_seg` / `dicom_rtstruct`. `--roi_subset` restricts prediction to named classes. — [TotalSegmentator README](https://github.com/wasserth/TotalSegmentator/blob/master/README.md)
- Current release: 2.18.0 on PyPI (uploaded 2026-08-12). The code already defines a `total_v3` class map, identical to `total` except that ID 26 becomes vertebrae_L6. — [PyPI JSON](https://pypi.org/pypi/TotalSegmentator/json); [map_to_binary.py](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py)
- A search of TotalSegmentator GitHub issues for "vertebral artery" found no dedicated request or roadmap item. Issue #1 ("Tell us what you are using TotalSegmentator for", about 70 comments) does not mention vertebral arteries. — [GitHub issue #1](https://github.com/wasserth/TotalSegmentator/issues/1)

### Inferences
- Running TotalSegmentator on a head-and-neck CTA yields the VA's parent vessel (subclavian) and neighbouring anchors (CCA, ICA, C1–C7 vertebrae, anterior/middle scalenes, prevertebral/longus muscles), but never the VA itself. These anchors could constrain a separate VA extraction:
  - V0/V1: origin from the proximal subclavian, running between the longus colli and the anterior scalene.
  - V2: through the C6–C1 transverse foramina, which appear as holes in the per-vertebra masks.
  - V3: around C1 toward the foramen magnum.
- `headneck_bones_vessels` gives the ICA from the carotid bifurcation to the skull entry. This is the carotid counterpart of what we need, useful for alignment, but not the VA.
- Everything we would use (`total`, `headneck_*`, `vertebrae_*`) is Apache-2.0, so no licence request is needed.

### Gaps
- Could not confirm whether the TotalSegmentator `total` subclavian label is drawn with the same "lateral boundary at cranial sternum" convention as the Heidelberg guideline. The latter applies to the Heidelberg annotations, not necessarily to the `total` training labels. (The v3-dataset probe in Q2 shows `total` subclavian labels about 30–45 mm in craniocaudal extent, which covers the region where the VA arises.)
- The Radiology: AI 2023 TotalSegmentator paper is not open-access full text on PMC (only the abstract was retrievable). Per-class annotation conventions for the neck vessels could not be read from it.

## Q2. Which TotalSegmentator training datasets are publicly released, do their labels include the vertebral arteries, and what is their field of view?

### Takeaway
The CT dataset is public under CC-BY-4.0, now at **v3.0.0 (1,939 CTs, 117 structures, 37.4 GB, released 2026-09-10)**. Its neck-vessel labels are brachiocephalic trunk, subclavians and common carotids only, with no VA and no ICA.

The field of view varies by study type. The dataset does contain about 71 head/neck CT angiograms ("ct angiography head" 65, "ct angiography neck" 6) and 52 neck-to-leg angiograms. In the head/neck CTAs we probed, the field of view runs from the aortic-arch branches to the skull vertex, so the whole VA course is imaged but unlabelled. Images are stored at 1.5 mm isotropic.

The MRI dataset (616 MRIs, 50 classes) is CC-BY-NC-SA-2.0 and has no neck-artery labels.

### Cited Findings
- Zenodo concept record 6802613 has three versions:
  - **v3.0.0**: "Dataset with segmentations of 117 important anatomical structures in 1939 CT images". Published 2026-09-10, licence cc-by-4.0, open access. File `Totalsegmentator_dataset_v300.zip` (37,415,622,057 bytes), DOI 10.5281/zenodo.22688904. Changelog: "increases the number of images from 1228 to 1939. We added 291 pediatric CT images ... we refined the labels especially for bones structures ... and we fixed the vertebrae label mixups." — [Zenodo v3.0.0](https://zenodo.org/records/22688904)
  - **v2.0.1**: 1,228 CTs, 117 structures, cc-by-4.0, 23.6 GB, published 2023-10-27. DOI 10.5281/zenodo.10047292. — [Zenodo v2.0.1](https://zenodo.org/records/10047292)
  - A 102-subject subset, `Totalsegmentator_dataset_small_v201.zip` (3.2 GB), is on Zenodo record 10047263. — [Zenodo small subset](https://zenodo.org/records/10047263)
  - The README still links the concept DOI as "CT dataset (1228 subjects)" and the "MR dataset (616 subjects)". — [TotalSegmentator README](https://github.com/wasserth/TotalSegmentator/blob/master/README.md)
- **Direct verification of v3 contents.** We read the zip's central directory over HTTP range requests without downloading the archive:
  - 232,682 entries in 1,941 top-level items: 1,939 subject folders `sXXXX/`, `meta.csv` and one blank-named directory entry.
  - Each subject has `ct.nii.gz` and `segmentations/<class>.nii.gz`, with 118 distinct mask filenames across the dataset.
  - The only neck-vessel masks are `brachiocephalic_trunk`, `brachiocephalic_vein_left/right`, `common_carotid_artery_left/right` and `subclavian_artery_left/right`.
  - There is **no vertebral-artery, internal-carotid or jugular file**.

  — [Zenodo v3.0.0 archive](https://zenodo.org/records/22688904)
- **v3 `meta.csv`** (1,939 rows; 1,830 train / 109 test). Study-type counts relevant to the neck:

  | study_type | count |
  |---|---|
  | ct angiography head | 65 |
  | ct angiography neck | 6 |
  | ct angiography neck-thx-abd-pelvis-leg | 52 |
  | ct neck | 65 |
  | ct chest-neck | 79 |
  | ct neck-chest-abdomen-pelvis | 114 |
  | ct polytrauma | 125 |
  | ct polytrauma head | 2 |
  | ct head | 122 |
  | ct whole body | 6 |
  | (blank) | 326 |

  Columns: image_id, age, institute, study_type, split, manufacturer, scanner_model, kvp, convolution_kernel, slice_thickness, pathology, pathology_location, gender. — [Zenodo v3.0.0 archive, meta.csv](https://zenodo.org/records/22688904)
- **Field-of-view probe of four CTA subjects** (read remotely from the v3 zip; NIfTI headers and label z-extents):

  | subject | study type | original slice | stored volume | labels present (z-range in slices) |
  |---|---|---|---|---|
  | s0292 | ct angiography head | 0.75 mm | 148×148×233 at 1.5 mm isotropic | brachiocephalic trunk z 3–20; subclavians z 9–38; CCAs up to z 89; C1 z 112–131; skull to z 228 |
  | s0152 | ct angiography head | 0.75 mm | 233×233×224 at 1.5 mm | same pattern |
  | s1112 | ct angiography neck (test split) | 0.75 mm | 213×213×220 at 1.5 mm | same pattern |
  | s0744 | ct angiography neck | 2 mm | 109×92×111 | small head-only crop; no neck-vessel labels |

  — [Zenodo v3.0.0 archive](https://zenodo.org/records/22688904)
- The TotalSegmentator paper states: "The CT images were randomly sampled from routine clinical studies and thus represent a real-world dataset (different ages, abnormalities, scanners, body parts, sequences, and sites)." — [Radiology: AI 2023 abstract (PMC10546353)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10546353/)
- MRI dataset: "TotalSegmentator MRI dataset: 616 MRI images with segmentations for 50 anatomical regions", v2.0.0, published 2025-01-21, licence **cc-by-nc-sa-2.0**, 5.1 GB, DOI 10.5281/zenodo.14710732. Its 50 classes match `total_mr`, which has no neck arteries. — [Zenodo MRI dataset](https://zenodo.org/records/14710732); [map_to_binary.py](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py)
- The Heidelberg `headneck_*` training data (86 training planning CTs: 84 in-house UKHD plus 2 open TCIA HNC datasets; 18 test) is **not publicly released**. The paper says: "In-house pseudonymized data are available on request from the corresponding author ... The data are not publicly available due to missing explicit patient consent". The TCIA part is available "with the permission of TCIA". — [Cancers 2024;16(2):415 (Europe PMC full text)](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11154560/fullTextXML)

### Inferences
- No released TotalSegmentator label set contains VA geometry. The CC-BY-4.0 v3 archive does contain about 71 head/neck CT angiograms (plus 52 neck-to-leg runoff CTAs), several of which we confirmed span aortic arch to vertex. That makes them a legal, redistributable image source for our own VA annotation, already paired with subclavian, CCA, brachiocephalic-trunk, vertebra and skull masks.
- The main drawback is the **1.5 mm isotropic resampling**. The VA lumen is only a few millimetres wide, so it would be just 2–3 voxels across. That is enough for centreline and approximate-radius extraction, but coarse for wall geometry. Native-resolution originals are not in the archive.
- To find candidate subjects cheaply, filter `meta.csv` by study_type (`ct angiography head` / `neck`) and slice_thickness 0.75 mm. Then pull only those subjects' files from the zip with HTTP range requests, avoiding the 37 GB download. We did exactly this in the probe.

### Gaps
- Did not check how many of the 71 head/neck CTAs have full arch-to-skull-base coverage (1 of 4 probed was a head-only crop), or whether they are arterial-phase with good VA opacification.
- The 326 v3 rows with blank study_type (apparently including the new paediatric cases) have unknown field of view.
- Could not confirm from a primary document that all dataset images were deliberately resampled to 1.5 mm. This is observed on the 4 probed subjects only.

## Q3. Do the VISTA3D / NV-Segment-CT / NV-Segment-CTMR class lists include the vertebral, internal carotid, common carotid or subclavian arteries, and under what licence?

### Takeaway
None of NVIDIA's released CT/MR segmentation models has a vertebral-artery or internal-carotid class. All of them (VISTA3D 127 classes, NV-Segment-CT 132 labels, NV-Segment-CTMR 345) include the left/right common carotid, left/right subclavian and brachiocephalic trunk, all inherited from TotalSegmentator v2 pseudo-labels.

VISTA3D / NV-Segment-CT does support **zero-shot point-click segmentation of unlisted structures**. That is the only NVIDIA route to a VA mask. NV-Segment-CTMR has no point prompts.

### Cited Findings
- VISTA3D label dict: vessel entries are aorta (6), inferior vena cava (7), portal/splenic vein (17), hepatic vessel (25), left/right iliac artery and vein (58–61), **brachiocephalic trunk (109)**, left/right brachiocephalic vein (110/111), **left/right common carotid artery (112/113)**, pulmonary vein (119), **left/right subclavian artery (123/124)** and superior vena cava (125). There is no vertebral, internal-carotid or basilar entry. — [VISTA label_dict.json](https://github.com/Project-MONAI/VISTA/blob/main/vista3d/data/jsons/label_dict.json)
- The MONAI model-zoo `vista3d` bundle (v0.5.11, MONAI 1.4.0) lists the same vessel labels in `metadata.json`. — [model-zoo vista3d metadata.json](https://github.com/Project-MONAI/model-zoo/blob/dev/models/vista3d/configs/metadata.json)
- NV-Segment-CT class list (Hugging Face model card): includes 112 "left common carotid artery", 113 "right common carotid artery", 123 "left subclavian artery" and 124 "right subclavian artery", and no vertebral artery.
  - Zero-shot mechanism: "To specify a new class for zero-shot segmentation, set the `label_prompt` to a value between 133 and 254. Ensure that `points` and `point_labels` are also provided".
  - Input is NIfTI; output is a 3D NIfTI mask.
  - Licence: NVIDIA Open Model License Agreement.

  — [nvidia/NV-Segment-CT model card](https://huggingface.co/nvidia/NV-Segment-CT)
- The NV-Segment GitHub README summarises the two models:
  - "NV-Segment-CT ... trained systematically on 11,454 volumes encompassing 127 types of human anatomical structures", with 132 classes, automatic plus interactive (point-click), licence "Commercial Friendly" (NVIDIA Open Model License).
  - "NV-Segment-CTMR starts from NV-Segment-CT checkpoint and finetuned on over 30K CT and MRI scans, supporting over 300 classes", automatic only, licence "Non-Commercial".

  — [NVIDIA-Medtech/NV-Segment-CTMR README](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
- NV-Segment-CTMR `label_dict.json` (345 entries) gives the source dataset for each class:
  - Brachiocephalic trunk (109), left/right common carotid (112/113) and left/right subclavian (123/124) come only from "TotalSegmentatorV2".
  - The head-and-neck block (177–213) comes from SegRap1/SegRap2: chiasm, temporal lobes, eyes, lenses, optic nerves, middle ears, inner-ear structures, ethmoid, pituitary, oral cavity, mandible, glands, mastoid, TMJ, larynx, pharynx constrictor. It has no vessels.
  - MR brain parcellation (214–345) comes from LUMIR.
  - A search of all 345 names for artery, carotid, vertebral, jugular, basilar or Willis finds no vertebral or internal carotid artery.

  — [NV-Segment-CTMR label_dict.json](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CTMR/configs/label_dict.json)
- NV-Segment-CTMR model card:
  - Release: "Huggingface: 10/27/2025".
  - Model licence: "NVIDIA OneWay Non-Commercial License for academic research purposes".
  - Prompts: "NV-Segment-CTMR does not support point based interactive segmentation".
  - Training data: "~31k ... 32 datasets".
  - Head-and-neck coverage: "detailed facial structures, sensory organs, and cranial anatomy (172-213)".

  — [nvidia/NV-Segment-CTMR model card](https://huggingface.co/nvidia/NV-Segment-CTMR)
- The Hugging Face refactor `MONAI/VISTA3D-HF`: code under Apache-2.0; "The model weights ... are licensed under the NCLS v1 License". This is NVIDIA's non-commercial source-code licence, so this copy is more restrictive than NV-Segment-CT. — [MONAI/VISTA3D-HF model card](https://huggingface.co/MONAI/VISTA3D-HF)

### Inferences
- For our purpose, NVIDIA's automatic classes add nothing beyond TotalSegmentator `total`: they are the same five neck-vessel structures, learned from TotalSegmentator pseudo-labels.
- The distinctive NVIDIA capability is **VISTA3D / NV-Segment-CT zero-shot interactive segmentation**. You click a few foreground and background points on the VA in a CTA and assign a free label index from 133 to 254. This could produce VA masks segment by segment on a public CTA (TotalSegmentator v3 CTAs, TopCoW CTA, TCIA).
  - The interactive branch was trained on SAM supervoxels, not on vessels, so accuracy on a thin tortuous artery is unproven. Expect to click along V1–V3 and to QC the result.
  - Its NVIDIA Open Model License allows commercial use; VISTA3D-HF and NV-Segment-CTMR do not.

### Gaps
- No published accuracy for VISTA3D zero-shot segmentation on small vessels such as the vertebral artery was found.
- Did not verify whether the NVIDIA NIM hosted endpoint (build.nvidia.com/nvidia/vista-3d) exposes the zero-shot point prompts for novel label indices.

## Q4. What training data did NVIDIA use, is any of it released, and do other MONAI model-zoo bundles or Clara/NGC resources have head-neck vessel labels?

### Takeaway
NVIDIA's models were trained on public datasets plus TotalSegmentator pseudo-labels. No NVIDIA-curated label release with neck vessels was found, and no MONAI model-zoo bundle or NGC/Clara model segments the vertebral arteries.

The only zoo bundles that touch neck vessels are `vista3d` and `maisi_ct_generative` (its generative label set includes CCA and subclavian, not VA). `wholeBody_ct_segmentation` is trained on TotalSegmentator v1, which predates the v2 carotid and subclavian classes.

### Cited Findings
- The VISTA3D paper says NVIDIA "curated a collection of 11454 CT volumetric images obtained from in-house and publicly available data sources" and "generated pseudo-labels of 117 classes using TotalSegmentator ... and supervoxels using SAM for every scan". Table 1 of the paper lists the sources and case counts:
  - Annotated: TCIA Pancreas-CT 80, AbdomenCT-1K 1,050, AMOS22 300, MSD tasks 3/6/7/8/9/10 945, CT-ORG 136, TotalSegmentator 1,228, CRLM-CT 197, VerSe 374, AeroPath 27, in-house bone lesion 296.
  - Unannotated: LIDC-IDRI 470, COVID-19 524, TCIA Colonography 1,440, StonyBrook COVID19 CT 1,274, NLST 3,113.

  — [VISTA3D paper, arXiv:2406.05285](https://arxiv.org/html/2406.05285)
- In NV-Segment-CTMR's per-class provenance, the named source datasets include TotalSegmentatorV2, TotalSegmentatorMR, AutoPETAtlas, VerSe, SegRap1/2/3, AMOS22 (CT/MR), BTCV, FLARE22, LUMIR, Brats23 and MSD tasks. None contributes a vertebral-artery label. — [NV-Segment-CTMR label_dict.json](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CTMR/configs/label_dict.json)
- The NV-Segment `data_license.txt` lists only "Third Party Licenses ... Medical Segmentation Decathlon". `docs/data.md` only explains how users can build their own datalists; no curated data is distributed. — [NV-Segment-CTMR docs/data_license.txt](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CTMR/docs/data_license.txt); [docs/data.md](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CTMR/docs/data.md)
- Current MONAI model-zoo bundles (dev branch): brain_image_synthesis_latent_diffusion_model, brats_*, breast_density_classification, cxr_image_synthesis_latent_diffusion_model, endoscopic_*, lung_nodule_ct_detection, maisi_ct_generative, mednist_*, multi_organ_segmentation, pancreas_ct_dints_segmentation, pathology_*, pediatric_abdominal_ct_segmentation, prostate_mri_anatomy, renalStructures_*, retinalOCT_RPD_segmentation, spleen_*, swin_unetr_btcv_segmentation, valve_landmarks, ventricular_short_axis_3label, vista2d, vista3d, wholeBody_ct_segmentation and wholeBrainSeg_Large_UNEST_segmentation. None is a head/neck vessel model. — [MONAI model-zoo models/](https://github.com/Project-MONAI/model-zoo/tree/dev/models)
- `maisi_ct_generative` v1.0.2 (a synthetic CT generator conditioned on label maps) includes brachiocephalic trunk, left/right brachiocephalic vein, left/right common carotid artery and left/right subclavian artery, with no vertebral artery. — [model-zoo maisi_ct_generative metadata.json](https://github.com/Project-MONAI/model-zoo/blob/dev/models/maisi_ct_generative/configs/metadata.json)
- `wholeBody_ct_segmentation` v0.2.7: "segments 104 distinct anatomical structures ... trained on TotalSegmentator data" (the v1 class set). — [model-zoo wholeBody_ct_segmentation metadata.json](https://github.com/Project-MONAI/model-zoo/blob/dev/models/wholeBody_ct_segmentation/configs/metadata.json); [TotalSegmentator v2 new classes](https://github.com/wasserth/TotalSegmentator/blob/master/resources/improvements_in_v2.md)
- A web search of NGC for Clara head/neck or carotid vessel models found only unrelated or deprecated Clara items (e.g. "Clara Deploy AI Brain Tumor Segmentation [Deprecated]") and no neck-vessel model. — [NGC Clara Deploy brain tumor (deprecated)](https://ngc.nvidia.com/catalog/containers/nvidia:clara:ai-braintumor/); [Clara Train SDK model docs](https://docs.nvidia.com/clara/clara-train-sdk/pt/model.html)

### Inferences
- NVIDIA/MONAI provide no VA training data or VA labels. Their neck-vessel knowledge is a re-learning of TotalSegmentator v2 labels.
- MAISI could synthesise CTs from label maps, but without a VA label channel it cannot generate controlled VA geometry.

### Gaps
- The NGC catalog could not be browsed exhaustively: the Clara Train SDK model page did not render its model list for scripted retrieval. The "no Clara neck-vessel model" conclusion rests on the web search, plus the fact that Clara Train's MSD-style models are deprecated and superseded by the MONAI model zoo.
- The VISTA3D in-house bone-lesion data and NVIDIA's processed pseudo-labels do not appear to be released. No release was found, but none was explicitly ruled out either.

## Q5. Per-resource summary: URL, VA coverage, label format, access terms, and how we could use it

### Takeaway
No resource provides VA labels. There are three practical routes to cervical VA geometry:
- **(a)** Use the TotalSegmentator v3 CC-BY-4.0 head/neck CTAs (or TopCoW CT / TCIA CTAs) as images and extract the VA ourselves.
- **(b)** Seed or segment the VA with VISTA3D / NV-Segment-CT point prompts.
- **(c)** Constrain and QC the VA path with TotalSegmentator anchor masks: subclavian (V0 origin), C6–C1 vertebrae (V2 foramina), scalene and longus muscles (V1), skull (V3/V4).

### Cited Findings

TotalSegmentator:

| Resource | URL | VA coverage | Format | Access / licence |
|---|---|---|---|---|
| TotalSegmentator `total` task (v2.18.0) | [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | None. Brachiocephalic trunk, L/R subclavian, L/R CCA, C1–C7 vertebrae, skull. | NIfTI per class / multilabel; DICOM-SEG / RTSTRUCT | Apache-2.0, `pip install TotalSegmentator` |
| TotalSegmentator `headneck_bones_vessels` | same | None. L/R ICA (bifurcation to skull entry), L/R IJV. | NIfTI | Apache-2.0 |
| TotalSegmentator `headneck_muscles` | same | None. Anterior/middle/posterior scalenes, prevertebral muscles, SCM (V1 landmarks). | NIfTI | Apache-2.0 |
| TotalSegmentator `total_mr` / MRI dataset | [Zenodo 14710732](https://zenodo.org/records/14710732) | None (no neck arteries) | NIfTI | Model Apache-2.0; dataset CC-BY-NC-SA-2.0 |
| TotalSegmentator CT dataset v3.0.0 | [Zenodo 22688904](https://zenodo.org/records/22688904) | No VA labels. About 71 head/neck CTAs + 52 neck-to-leg CTAs with the VA imaged, 1.5 mm isotropic. | `sXXXX/ct.nii.gz` + `segmentations/<class>.nii.gz`; `meta.csv` | CC-BY-4.0, open download (37.4 GB) |

NVIDIA / MONAI:

| Resource | URL | VA coverage | Format | Access / licence |
|---|---|---|---|---|
| NV-Segment-CT (VISTA3D) | [huggingface.co/nvidia/NV-Segment-CT](https://huggingface.co/nvidia/NV-Segment-CT) | No VA class. L/R CCA, L/R subclavian, brachiocephalic trunk; zero-shot point prompts for new labels 133–254. | NIfTI in/out | NVIDIA Open Model License (commercial-friendly) |
| VISTA3D MONAI bundle / VISTA3D-HF | [model-zoo vista3d](https://github.com/Project-MONAI/model-zoo/tree/dev/models/vista3d); [MONAI/VISTA3D-HF](https://huggingface.co/MONAI/VISTA3D-HF) | Same as NV-Segment-CT | NIfTI | Code Apache-2.0; weights NCLS v1 (non-commercial) for the HF copy |
| NV-Segment-CTMR | [huggingface.co/nvidia/NV-Segment-CTMR](https://huggingface.co/nvidia/NV-Segment-CTMR); [GitHub](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR) | No VA. CCA, subclavian, brachiocephalic trunk; automatic only. | NIfTI | NVIDIA OneWay Non-Commercial |
| MONAI `maisi_ct_generative` 1.0.2 | [model-zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/maisi_ct_generative) | No VA label channel | Generated NIfTI | See bundle licence |
| MONAI `wholeBody_ct_segmentation` 0.2.7 | [model-zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation) | None (TotalSegmentator v1 classes, no carotid/subclavian) | NIfTI | See bundle licence |

- Sources for the table rows: [TotalSegmentator README](https://github.com/wasserth/TotalSegmentator/blob/master/README.md); [NV-Segment-CTMR README](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR); [NV-Segment-CT model card](https://huggingface.co/nvidia/NV-Segment-CT); [Zenodo v3.0.0](https://zenodo.org/records/22688904).

### Inferences
Suggested pipeline for grafting cervical VAs into the simulation anatomy, built from these resources:

1. **Pick CTAs.** From the TotalSegmentator v3 `meta.csv`, select `ct angiography head` / `ct angiography neck` subjects with slice_thickness 0.75 mm. Range-download only those subjects. They are CC-BY-4.0, so the derived meshes can be redistributed with attribution.
2. **Get the anchors.** Take `subclavian_artery_*`, `common_carotid_artery_*`, `brachiocephalic_trunk`, `vertebrae_C1`–`C7` and `skull` from the released labels, and ICA/IJV from `headneck_bones_vessels`. This fixes the VA origin, the V2 foramen corridor and the skull-base exit, and matches the carotid-side conventions we already use.
3. **Extract the VA** by one of two methods:
   - VISTA3D / NV-Segment-CT zero-shot point prompts, with a few clicks per segment.
   - Classic vesselness filtering plus centreline tracking (VMTK is already in our `neve-build-vmtk` image), seeded at the subclavian origin and at C1/foramen magnum.

   Then QC against the vertebra masks: V2 should pass through the C6–C1 transverse foramina.
4. **Mind the resolution.** At 1.5 mm isotropic the VA radius is under-resolved. Prefer TopCoW CTA or TCIA native-resolution CTAs if wall geometry matters. Where images are only available at native resolution elsewhere, TotalSegmentator can be re-run on them for the anchors.
5. **Connect to TopBrain.** The basilar/V4 end would still come from our TopBrain MR intracranial pieces, joined at the V3/V4 junction near the foramen magnum.

### Gaps
- The quality of any automated VA extraction on these data is untested: no benchmark of VISTA3D zero-shot or TotalSegmentator-anchored tracking on the VA was found.
- Whether TopCoW CT or specific TCIA collections contain the full cervical field of view (subclavian to skull base) was out of scope and not checked here. The TotalSegmentator v3 CTAs are confirmed to for 3 of 4 probed subjects.
