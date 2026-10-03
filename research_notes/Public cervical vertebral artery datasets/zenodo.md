# Zenodo records with vertebral artery (VA) geometry along its cervical course (V1–V3)

Scope and method: about 150 queries against the Zenodo REST API (`https://zenodo.org/api/records?q=...`). Unauthenticated calls are capped at 25 results per page and 30 requests per minute. The queries were plain, `+term` (required-term) and quoted-phrase forms, with and without `type=dataset`, and some were paged 4 deep. They covered every phrase in the brief (e.g. "vertebral artery segmentation", "head and neck CTA", "supra-aortic arteries", "cervical arteries segmentation", "neck vessel segmentation", "cerebrovascular CTA dataset", "aortic arch branches segmentation", "vessel centerline dataset neck", "VA CTA labels", "head neck angiography nnU-Net", "vascular tree head neck"). Additional queries covered vertebrobasilar, subclavian, brachiocephalic, craniocervical, carotid STL/mesh/centerline, phantoms, TOF-MRA, CE-MRA, TotalSegmentator, thrombectomy/tortuosity, named datasets (SMILE-UHURA, VesselVerse, CROWN, AneuX) and author searches (Canals, Yang K., Musio, Hilbert). Every candidate record was opened through `/api/records/<id>` to read its description, license, access right and file list. Where a record described its data only loosely, I checked the linked challenge proposal PDF, paper, README or GitHub.

## Q1. Which Zenodo records contain VA labels (masks, meshes or centerlines) covering the CERVICAL VA?

### Takeaway
No Zenodo record has subject-specific, manually annotated cervical VA labels (V1–V3), and none has a VA label that starts at the subclavian origin. Only two records come close:
- **Arterial model weights (Zenodo 22694951)**, an open, non-commercial tool. It segments head-and-neck CTA from the aortic arch up and labels the centerline segments RSA/LSA, RVA/LVA and BA, so it can *generate* origin-to-basilar VA geometry from any head-and-neck CTA you supply.
- **A PET-derived population-average cerebrovascular atlas (Zenodo 20307456)**. Its right and left VA labels run about 100 mm cranio-caudally through the neck. It is a single smoothed template in MNI space and does not reach the VA origin.

### Cited Findings

**A. Arterial: trained model weights (a tool, not a dataset). The closest match to "arch-to-basilar VA".**
- Title "Arterial: trained model weights". DOI 10.5281/zenodo.22694951 (concept 10.5281/zenodo.22694950), v1.0, published 2026-09-10, resource type "model", access **open**, license **CC BY-NC 4.0**. Files: `arterial-models-v1.tar.gz` (1,126.9 MB), a `.sha256` checksum and `README.md` — [Zenodo 22694951](https://zenodo.org/records/22694951)
- The record describes itself as trained weights for "vessel segmentation (nnU-Net v2), anatomical landmark detection (3D U-Net), vessel labelling (GNN) and catheter access prediction (ArterialGNet), for CT angiography of the supra-aortic region". It was developed at Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona, and the source code is under the PolyForm Noncommercial License 1.0.0 — [Zenodo 22694951](https://zenodo.org/records/22694951)
- README archive contents:
  - `segmentation/extracranial_vessels/`: "Vessel segmentation from the neck to the Circle of Willis", nnU-Net v2.
  - `segmentation/intracranial_vessels/`.
  - `vessel_labelling/`: "Anatomical naming of centerline segments (e.g. LCCA)", GNN.
  - `access_prediction/`.
  - A third-party TotalSegmentator `craniofacial_structures` model (Apache-2.0), used only to split head from neck.

  The README also states: "No account, no token and no licence gate" — [Arterial README on Zenodo](https://zenodo.org/records/22694951)
- The vessel-labelling module classifies centerline segments into 14 classes: 0 other, 1 AA (aortic arch), 2 BT, 3 RCCA, 4 LCCA, **5 RSA, 6 LSA, 7 RVA (right vertebral artery), 8 LVA (left vertebral artery)**, 9 RICA, 10 LICA, 11 RECA, 12 LECA, **13 BA**. It uses a GATv2 graph network with a 5-fold ensemble and 24 geometric features — [Arterial vessel_labelling docs](https://github.com/FLOWCAT-CV/arterial/blob/main/docs/modules/vessel_labelling.md)
- According to the GitHub README, the pipeline takes a "head-and-neck basal CTA" and outputs binary vessel masks (NIfTI), probability maps, surface meshes (VTK/STL) and centerline models with radii, plus labelled centerline graphs — [FLOWCAT-CV/arterial](https://github.com/FLOWCAT-CV/arterial)
- Limitations stated in the README:
  - The model was "Trained on CTA acquired in the context of acute ischaemic stroke workup".
  - The training cohort (size, centres, vendors, split) is still a "TODO" placeholder in the README.
  - The weights are for noncommercial research only.
  - The citation is Canals et al., *Computerized Medical Imaging and Graphics* 104:102170 (2023), doi:10.1016/j.compmedimag.2022.102170.

  — [Arterial README on Zenodo](https://zenodo.org/records/22694951)
- The **training images and labels are not released** on Zenodo. An author search for "Canals, Pere" returns only this model record — [Zenodo API search, creators.name:"Canals, Pere"](https://zenodo.org/api/records?q=creators.name:%22Canals,%20Pere%22)

**B. Pseudo-angiographic [18F]FDG PET template and cerebrovascular atlas. The only Zenodo dataset found with a VA label that extends into the neck.**
- DOI 10.5281/zenodo.20307456, published 2026-06-26, access **open**, license **CC BY 4.0**.
- Built from dynamic FDG PET in 42 healthy volunteers (Siemens Biograph Vision 600, Liège) and registered to MNI PET152 space. It is a group-level template.
- The coarse atlas has 11 labels, including the vertebral artery (VA). Everything is NIfTI at 1 mm isotropic.
- Files: `pseudo_angio_template.nii.gz` (52.5 MB), `atlas_cerebral_coarse.nii.gz` and `atlas_cerebral_fine.nii.gz` (0.3 MB each), plus two label `.tsv` files.

  — [Zenodo 20307456](https://zenodo.org/records/20307456)
- `labels_coarse.tsv`: 1/2 CCA R/L, 3/4 ICA R/L ("ICA full vessel or C1+C2+C3 fallback"), 5/6 MCA, 7 ACA, **8 VA right, 9 VA left**, 10/11 venous sinuses — [Zenodo 20307456 labels_coarse.tsv](https://zenodo.org/records/20307456/files/labels_coarse.tsv?download=1)
- My own measurement of `atlas_cerebral_coarse.nii.gz` (201×261×291 voxels, 1 mm, sform z = index − 170):

  | Label | Voxels | Slices (1 mm) | MNI z range |
  |---|---|---|---|
  | VA-R | 1,726 | 106 | −162 to −57 mm |
  | VA-L | 1,772 | 101 | −157 to −57 mm |
  | CCA | — | — | −162 to −141 mm |
  | ICA | — | — | about −102 to −36 mm |

  Derived from [Zenodo 20307456](https://zenodo.org/records/20307456).

**C. Records whose raw images include some neck, but with no VA labels.**
- **ISLES'24** (DOI 10.5281/zenodo.17652035):
  - 149 acute-stroke cases with NCCT, CTA, CTP and perfusion maps plus follow-up MRI.
  - Annotations: infarct masks, LVO masks from CTA, and a "multi-labeled Circle of Willis anatomy generated with an automatic algorithm over CTA".
  - Raw data are in original space and only defaced.
  - License CC BY-NC-SA 4.0, open, one `train.7z` of 92.2 GB.

  — [Zenodo 17652035](https://zenodo.org/records/17652035)

  The companion paper says: "The CTA scans are released after cropping the images around the head, thus removing the thoracic region when available" — [ISLES'24 arXiv 2408.10966](https://arxiv.org/html/2408.10966)
- **Open dataset of annotated DICOM images of head CTA for intracranial aneurysm detection** (DOI 10.5281/zenodo.15697196):
  - 172 CTA series (90 controls, 82 IA) from three centres, DICOM, defaced, 0.5–2 mm slices.
  - Annotations are aneurysm STL only.
  - The metadata include a "Series description (... inclusion of the carotid bifurcation area)" field.
  - CC BY 4.0, open, `Dataset.zip` 23.8 GB.

  — [Zenodo 15697196](https://zenodo.org/records/15697196)

### Inferences
- For grafting real cervical VA geometry, the only Zenodo-hosted route to arch-to-basilar VA centerlines, meshes and radii is to run **Arterial** on head-and-neck CTAs that include the aortic arch. Its label set (AA→RSA/LSA→RVA/LVA→BA) matches the "starts at the subclavian origin" priority. The masks would be model output, not expert ground truth, and non-commercial terms apply.
- Candidate open inputs for Arterial on Zenodo:
  - ISLES'24 raw CTA: the thorax was removed, so the VA origin is probably lost, but cervical V1/V2–V4 may survive.
  - The head-CTA IA dataset: only the series flagged as covering the carotid bifurcation, so probably V2 and above.

  Neither is verified to reach the subclavian origin, and both need a per-case FOV check.
- The PET atlas VA is useful only as a coarse prior for the average cervical VA path and length (≈100 mm from about the upper-CCA level to about the foramen magnum in MNI space):
  - It is a single template, not per-subject geometry.
  - Its lower end matches the lowest CCA slice, which suggests the atlas FOV cut it off rather than any anatomical landmark.
  - The label is voxel-blocky at 1 mm. From voxel count ÷ length, the cross-section is about 16 mm², roughly a 4.5 mm equivalent diameter — my derivation.
  - It does not reach V1 or the origin.

### Gaps
- None of the ~150 Zenodo queries found a dataset with manual VA masks, meshes or centerlines covering V1–V3, or starting at the subclavian origin. As far as Zenodo's own search can show, none exists. Zenodo full-text search is weak, however: it misses label lists inside zip READMEs.
- The Arterial training cohort (size, labels, FOV) and its per-class VA accuracy are not documented in the README. The CMIG 2023 paper was not retrieved here.
- The exact inferior extent of the ISLES'24 CTAs, and how many of the 172 head-CTA IA series include the neck, were not verified. The data would have to be downloaded, or the MDPI *Data* paper (doi:10.3390/data11040074) read; the MDPI page returned HTTP 403.
- The PET atlas paper was not located, and the anatomical definition of its VA label's lower end is not stated in the record.

## Q2. Which Zenodo records label the VA only intracranially (or not at all)? Exclusions

### Takeaway
All the brain-vessel datasets on Zenodo stop at the skull base, because they are braincase-cropped or TOF-MRA-limited. These include TopBrain, TopCoW and its derivatives, COSTA, BRAVE-NET labels, IXI-MRA annotations and the CoW centerline graphs. TopCoW has no VA class at all. TopBrain's VA (labels 23/24) is intracranial, and even the planned TopBrain 2.0 extends it only to "V3 to V4".

### Cited Findings

| Record | Why excluded |
|---|---|
| **TopBrain Challenge Data Release**: [Zenodo 16878417](https://zenodo.org/records/16878417) (2025-08-14, 1.82 GB) and newer version [Zenodo 21972006](https://zenodo.org/records/21972006) (2026-08-17, 1.83 GB, adds `labelsTr_topbrain_v2_topaneu36class`); same concept DOI 10.5281/zenodo.16623495 | 50 images (25 CTA+MRA pairs) taken from TopCoW. VA R/L = labels 23/24 ([TopBrain data page](https://topbrain2025.grand-challenge.org/data/)). Images are cropped to the braincase, so the VA is intracranial only (details in Q4). License: opendata.swiss "Open use. Must provide the source. Use for commercial purposes requires permission of the data owner." |
| **TopBrain 2025 challenge proposal**: [Zenodo 15084013](https://zenodo.org/records/15084013) | Proposal PDF only. Its label list includes "Vertebral artery (VA), left and right" with no segment qualifier. |
| **TopBrain 2.0 challenge proposal**: [Zenodo 19707577](https://zenodo.org/records/19707577) | The planned label list has "V3 to V4 segments of vertebral artery (VA), left and right". The stenosis task targets the "Intradural (V4) segment of vertebral artery". Images: "The defaced and cropped image includes only the brain region" ([proposal PDF](https://zenodo.org/records/19707577)). No V1–V2 planned. |
| **Best Performing TopBrain Segmentation Dockers**: [Zenodo 21959166](https://zenodo.org/records/21959166) (43.1 GB, CC BY 4.0) | Docker models trained on braincase-cropped TopBrain data (top 3 teams of 2025 plus the TA36 docker). Inference tools only, with no cervical VA class. |
| **TopCoW Training Data and External Testsets**: [Zenodo 15692630](https://zenodo.org/records/15692630) (13.4 GB) | 13 CoW classes only, with no VA class ([TopCoW paper](https://arxiv.org/html/2312.17670)). Images are "cropped to the braincase region". |
| **Circle of Willis Centerline Graphs and Morphometric Features**: [Zenodo 17358162](https://zenodo.org/records/17358162) (CC BY-NC 4.0, 355.8 MB) | Labelled CoW centerline graphs and meshes (.vtp) for 250 TopCoW images. CoW only. |
| **COSTA TOF-MRA database**: [Zenodo 11025761](https://zenodo.org/records/11025761) (restricted, no files) | 423 TOF-MRA volumes (IXI, ADAM, ...). Binary cerebrovascular masks with an intracranial FOV (e.g. IXI 512×512×100 at 0.8 mm slices ≈ 80 mm z-coverage). |
| **BRAVE-NET ground truth labels**: [Zenodo 3968844](https://zenodo.org/records/3968844) (MIT, 1 MB) | Voxel labels for 20 MIDAS TOF-MRA healthy volunteers. Intracranial, binary. |
| **Brain Vessel Annotated Dataset from IXI-MRA**: [Zenodo 17393202](https://zenodo.org/records/17393202) (CC BY 4.0, 6.3 MB) | 100 IXI TOF-MRA semi-automatic brain vessel masks. Intracranial FOV. |
| **Cerebral artery segmentation Challenge (CAS 2023)**: [Zenodo 7839970](https://zenodo.org/records/7839970) | Proposal only. Task is 3D TOF-MRA cerebral arteries in intracranial stenosis patients. |
| **CROWN (CoW classification and quantification challenge)**: [Zenodo 7844966](https://zenodo.org/records/7844966) | Proposal only. 600 TOF-MRA with CoW measurements. |
| **Intracranial Aneurysm & Stenosis challenge**: [Zenodo 10990482](https://zenodo.org/records/10990482); **TopAneu 2026**: [Zenodo 19848807](https://zenodo.org/records/19848807) | Proposals only. Intracranial lesion tasks. |
| **Large IA Segmentation dataset**: [Zenodo 6801398](https://zenodo.org/records/6801398) (restricted; request by institutional email; non-commercial) | 1,338 + 138 CTA with aneurysm masks only. No vessel labels. |
| **Dataset for modeling of a multiscale human cerebrovasculature**: [Zenodo 14638706](https://zenodo.org/records/14638706) | Large vessels rebuilt from head 4D-CTA of one subject. "The vasculature is modeled for the cerebrum, excepting ... brainstem and cerebellum" ([PLoS Comput Biol 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007943)). |
| **1000 angiographic cerebral vessel diameters**: [Zenodo 15222357](https://zenodo.org/records/15222357) | Tabular diameters of 14 basal cerebral arteries. No geometry. |
| **DIAS** ([Zenodo 11637181](https://zenodo.org/records/11637181)) and **DSCA** ([Zenodo 11255024](https://zenodo.org/records/11255024)) | 2D DSA sequences of intracranial arteries. Not 3D. |

### Inferences
- The user reports the TopBrain VA (labels 23/24) as 18–93 mm long. That fits the braincase crop: at most V4 plus a partial V3 at the foramen magnum or C1. No TopBrain/TopCoW-derived record can supply V1–V2.

### Gaps
- TopBrain 1.0 does not publish a written proximal boundary for the VA label (for example, whether it starts at the foramen magnum or at the dural entry). Neither the data page nor the medRxiv preprint excerpt gave it, and Table 1 was not fully retrievable ([medRxiv preprint](https://www.medrxiv.org/content/10.64898/2026.05.28.26354312v1.full)).

## Q3. Are there Zenodo records of whole supra-aortic or aortic-arch-to-brain vessel trees that include the VAs?

### Takeaway
No labelled data of this kind is on Zenodo. The aortic and arch datasets there either have no supra-aortic branches, or stop at the brachiocephalic, carotid and subclavian trunks without a VA. Only the Arterial model (Q1-A) produces an arch-to-BA labelled tree that includes the VA.

### Cited Findings

| Record | What it covers |
|---|---|
| **TotalSegmentator dataset v3.0.0**: [Zenodo 22688904](https://zenodo.org/records/22688904) (1,939 CT, 117 structures, CC BY 4.0, open, 34.85 GB) | Supra-aortic classes stop at the trunks. The "total" class map has `subclavian_artery_right/left` and `common_carotid_artery_right/left` (label IDs 55–58). The `headneck_bones_vessels` task has only `internal_carotid_artery_*` and `internal_jugular_vein_*`. No vertebral-artery class appears in the README class lists ([TotalSegmentator GitHub](https://github.com/wasserth/TotalSegmentator)). |
| **AortaSeg-60**: [Zenodo 18147026](https://zenodo.org/records/18147026) (CC0, 14.77 GB) | 60 CTA/NCCT scans with fully automatic TotalSegmentator segmentations. Therefore no VA. |
| **AortaSeg24 challenge**: [Zenodo 10991212](https://zenodo.org/records/10991212) (proposal PDF only) | 100 CTA with 23 aortic branches and SVS/STS zones. The listed branches are innominate, left common carotid, left subclavian, celiac, renals and internal iliacs, with no VA ([AortaSeg24 paper](https://arxiv.org/abs/2502.05330)). Data are distributed through grand-challenge via a DocuSign form, not Zenodo ([AortaSeg24 dataset page](https://aortaseg24.grand-challenge.org/dataset/)). |
| **SEG.A. 2023 aortic vessel tree challenge**: [Zenodo 7861733](https://zenodo.org/records/7861733) (proposal only) | The AVT dataset itself (56 CTA, binary aortic-vessel-tree masks including "the aortic arch and its branches into the head/neck area") is on figshare, not Zenodo ([AVT data article, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8760499/)). |
| **HUG-VAS dataset**: [Zenodo 18302984](https://zenodo.org/records/18302984) (CC BY 4.0, 27.5 MB) | Patient-specific aorta geometries with "major branches such as subclavian artery (RSA), right common carotid artery (RCCA), left subclavian artery (LSA), and left common carotid artery (LCCA)". No VA. |
| **Clinically-driven virtual patient cohort of 3000 thoracic aortas**: [Zenodo 10627490](https://zenodo.org/records/10627490) | "No supraaortic branches are present." |
| **Bovine aortic arch detection**: [Zenodo 7567792](https://zenodo.org/records/7567792) | Restricted with zero files. The underlying study segmented "the aortic arch and supra-aortic vessels" on chest CT, but no data are downloadable. |
| **Thoracic aorta TEVAR pipeline**: [Zenodo 5992287](https://zenodo.org/records/5992287) | Restricted, no files. Aorta and pulmonary arteries only. |
| **aortic dataset for DB-SNet**: [Zenodo 15025191](https://zenodo.org/records/15025191) | Thorax/abdomen CTA from six hospitals. |
| **HaN-Seg**: [Zenodo 7442914](https://zenodo.org/records/7442914) (CC BY-NC-ND 4.0, 4.6 GB) | 42 head-and-neck CT/MR with 30 organ-at-risk masks, including the carotid arteries (A_Carotid_L/R). No VA. |
| **Carotid bifurcation lumen/plaque models with centerline tree**: [Zenodo 10695923](https://zenodo.org/records/10695923) (CC BY 4.0) | 79 CTA cases, 152 carotid bifurcation lumen models. Carotid only. |
| **MICCAI 2022 carotid vessel wall challenge data**: [Zenodo 6481870](https://zenodo.org/records/6481870) | Carotid black-blood MRI. Carotid only. |
| **Head Neck CBCT CT datasets**: [Zenodo 13231287](https://zenodo.org/records/13231287) | Radiotherapy CT/CBCT with 5 structures. Not vascular. |

A curated catalogue of open cardiovascular imaging datasets lists no VA dataset and no Zenodo dataset with supra-aortic labels beyond TotalSegmentator ([open-cardiovasc-imaging-datasets](https://github.com/gfahrni/open-cardiovasc-imaging-datasets)). A web search summary of the associated review said supra-aortic trunks are "notably underrepresented ... with only three publicly open datasets identified" ([ScienceDirect scoping review](https://www.sciencedirect.com/science/article/pii/S3050577126000393)). That quote came from the search result snippet, not the full text.

### Inferences
- The TotalSegmentator v3, HUG-VAS and AVT arch/trunk labels could supply the arch and subclavian host geometry at the VA origin. Combining them with Arterial-generated VA centerlines on the same CTA would be the most practical Zenodo-based route to a real origin-to-skull-base VA.

### Gaps
- Web search results named private head-and-neck CTA datasets with VA labels: "HNCTA" (358 scans, 9 vessel classes including left and right VA) and "HNA" (800 CTA, cerebral, vertebral and carotid arteries) ([DeformCL, arXiv 2506.05820](https://arxiv.org/pdf/2506.05820)). Neither is on Zenodo, and I did not verify whether either is public. They sit outside this Zenodo-only scope and should be checked by the repositories/papers researcher.

## Q4. TopBrain (Zenodo 16878417) and TopCoW: VA field of view and label extent. Does any CT subset cover the neck?

### Takeaway
No. All released TopBrain and TopCoW images, CTA included, are cropped to the braincase, even though the original CTAs "usually cover neck and head regions". The TopBrain VA labels (23/24, shared by CTA and MRA) are therefore intracranial, V4 plus at most the uppermost V3. TopCoW has no VA label at all. The only uncropped CT in the TopCoW release sits in the external-testset `raw/` folders, and those carry CoW labels only.

### Cited Findings
- TopBrain data release contents:
  - 50 images (= 25 CTA/MRA pairs) from the TopCoW image data.
  - `labelsTr_topbrain_{ct|mr}` are multiclass NIfTI in LPS+ orientation. CTA has 40 labels, MRA 42, with 34 shared.
  - A single zip `TopBrain_Data_Release_Batches1n2_081425.zip` of 1,868.1 MB.
  - License per opendata.swiss: non-commercial use free, commercial use needs permission, source must be cited.

  — [Zenodo 16878417](https://zenodo.org/records/16878417)
- The August 2026 version adds `labelsTr_topbrain_v2_topaneu36class`: 36 modality-agnostic classes including new infraclinoid ICA classes C1–C5 vs C6–C7 — [Zenodo 21972006](https://zenodo.org/records/21972006)
- Label numbers: "Right/Left Vertebral Artery (VA): 23-24" are among the 34 labels shared by CTA and MRA. Training set is 25 patients with CTA+MRA pairs, and the test set of 20 is unreleased — [TopBrain data page](https://topbrain2025.grand-challenge.org/data/)
- The TopBrain 2025 proposal says:
  - "The original CTA scans usually cover neck and head regions, and sometimes just the head region. The original MRA scans typically cover the brain region."
  - Pre-processing: "Same data used as TopCoW challenge ... de-facing and cropping procedures ... The defaced image includes only the braincase region."

  — [TopBrain 2025 proposal, Zenodo 15084013](https://zenodo.org/records/15084013)
- TopCoW paper: "All TopCoW data were anonymized, defaced, re-oriented, and cropped to the braincase region." Its 13 classes are ICA, ACA, MCA, PCA (each L/R), Pcom L/R, Acom, BA and 3rd-A2, with no VA — [TopCoW arXiv 2312.17670](https://arxiv.org/html/2312.17670)
- The TopCoW Zenodo release (5 zips, 13.4 GB) contains:
  - `TopCoW2024_Data_Release.zip` (10.0 GB): the training data, same opendata.swiss license.
  - External testsets `CTA_ISLES2024_TUM.zip` (26 CTA, CC BY-NC), `CTA_LargeIA.zip` (20 CTA; images must be obtained from LargeIA), `MRA_Lausanne.zip` (20, CC0) and `MRA_IXI_HH.zip` (20, CC BY-SA 3.0).
  - External testsets include `raw/raw_imgs` (uncropped originals) plus CoW masks mapped back to raw space, and braincase ROI files used for cropping.

  — [Zenodo 15692630](https://zenodo.org/records/15692630)
- The TopBrain 2.0 proposal keeps brain-region cropping and extends the VA only to "V3 to V4 segments". Its planned training set includes 20 ISLES'24 CTA — [TopBrain 2.0 proposal, Zenodo 19707577](https://zenodo.org/records/19707577)

### Inferences
- The neck is present in the source CTAs (per the proposal) but was cropped away before release. No TopBrain or TopCoW record will ever give V1–V2 unless the organizers release uncropped images, which the 2.0 proposal does not plan.
- The TopCoW external-testset `raw/` CTAs (26 ISLES'24-TUM) may include some neck because ISLES'24 removed only the thorax. They could be run through Arterial to get cervical VA, but they carry no VA ground truth.

### Gaps
- The exact caudal end of the TopBrain VA label (foramen magnum, dural entry, or crop edge) is not documented in any retrieved source. It could be measured from the zip, as the user's 18–93 mm lengths already suggest.
- The per-case z-coverage of the TopCoW external-testset `raw/` CTAs was not verified.
