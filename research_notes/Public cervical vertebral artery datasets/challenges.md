# Challenge datasets that label the vertebral arteries in the neck (V1–V3)

Scope: public imaging challenges (Grand Challenge, MICCAI, ISBI, Kaggle/RSNA, CodaLab) with 3D head-and-neck angiography. For each one: does it label the vertebral artery (VA), over which segments, and can the data be obtained? The purpose is to graft real cervical VA geometry (ideally from the subclavian origin up to the basilar) into a catheter-navigation simulation. Research date: 2026-09-27.

## Q1. Does any public challenge label the cervical VA (V1–V3), ideally from the subclavian origin?

### Takeaway
No. None of the challenges checked releases voxel labels for the cervical VA (V1–V3), and none starts a VA label at the subclavian origin. There are three partial sources:
- **TopBrain 2025/2026** has left and right VA labels in both CTA and MRA, but its images come from the TopCoW/USZ stroke pipeline, which crops to the braincase. Its VA label is therefore almost certainly intracranial (V4 and perhaps the top of V3). I could not confirm the lower bound from the documentation.
- **AortaSeg24** labels the left subclavian artery (LSA) and the innominate artery up to their first bifurcation, which is about the VA origin.
- **AVT/SEG.A** gives one binary aortic-tree mask that includes the brachiocephalic trunk, left common carotid and LSA.

The only multi-class head-and-neck CTA vessel sets with cervical VA labels that I found (HNCTA, HNA, Fu et al.) are in-house or unverified, not challenge releases.

### Cited Findings

At-a-glance table. Details and extra sources are under Q2–Q5.

| Challenge | Modality | Cases (public / hidden) | Neck in FOV? | VA labelled? | Format | Access |
|---|---|---|---|---|---|---|
| TopCoW 2023/2024 | CTA + TOF-MRA pairs | 2024: 125 training patients (250 scans), 5 validation, 70 test hidden | No: "cropped to the braincase region" | **No** VA class; 13 Circle of Willis (CoW) classes | NIfTI (16-bit) | Zenodo; CC BY-NC; commercial use needs permission — [TopCoW24 data](https://topcow24.grand-challenge.org/data/), [TopCoW paper](https://arxiv.org/html/2312.17670v2) |
| TopBrain 2025 (+2026 v2) | CTA + MRA | 25 CTA + 25 MRA training; 20 + 20 test hidden | Not documented (inherits the TopCoW pipeline; see Q2) | **Yes**, R-VA/L-VA single labels (23/24); extent not documented | NIfTI (16-bit signed, LPS+) | Zenodo open; "commercial use requires permission" — [TopBrain data](https://topbrain2025.grand-challenge.org/data/), [Zenodo 16878417](https://zenodo.org/records/16878417) |
| RSNA Intracranial Aneurysm 2025 | CTA, MRA, MRI | Vessel masks for 178 cases | Not documented | No dedicated VA class; "other posterior circulation" | NIfTI (derivative) | Kaggle — [arXiv 2606.26706](https://arxiv.org/html/2606.26706) |
| AortaSeg24 | CTA (type B dissection) | 50 training; 10 validation + 40 test hidden | Aortic root to iliac bifurcation | **No**; LSA, LCCA and innominate end at their first bifurcation | Multi-class | DUA (DocuSign) + Dropbox — [arXiv 2502.05330](https://arxiv.org/html/2502.05330) |
| AVT / SEG.A 2023 | CTA | 56 public; SEG.A test (7 real + 150 synthetic) withheld | "arch and its branches into the head/neck area" | Not stated; brachiocephalic, LCCA, LSA are | Binary .seg.nrrd | figshare, CC BY — [AVT paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8760499/) |
| HaN-Seg 2023 | CT + T1 MR | 42 training; 14 test | Head and neck (radiotherapy) | **No**; carotid artery L/R only | Binary masks | Zenodo, CC BY-NC-ND 4.0 — [Zenodo 7442914](https://zenodo.org/records/7442914) |
| CAS 2023 | TOF-MRA | 100 (85 training / 15 test) | Intracranial | Binary "cerebral arteries" | — | CodaLab registration — [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/9804) |
| Carotid wall 2021 / COSMOS 2022 / CLS2009 | Black-blood MRI / CTA | see Q4 | Neck (carotid) | **No** (carotid only) | Contours / lumen | Zenodo / web — see Q4 |

- TopBrain is the only checked challenge with explicit VA classes: "Right VA" and "Left VA" are part of the 34 labels shared by CTA and MRA (values 23–24). The documentation "provides no specification regarding extent of VA coverage" — [TopBrain data page](https://topbrain2025.grand-challenge.org/data/).
- AortaSeg24 branches: "Each branch segmentation ended at the bifurcation point where it split into multiple branches." The innominate label ends where it splits into the right subclavian and right common carotid — [AortaSeg24 paper](https://arxiv.org/html/2502.05330).
- The in-house head-and-neck CTA sets with VA labels are HNCTA (358 CTAs, 9 classes including L/R VA, ICA, CCA, BA) — [DeformCL, arXiv 2506.05820](https://arxiv.org/html/2506.05820v1) — and HNA (800 CTAs; cerebral, vertebral and carotid arteries annotated) — [arXiv 2301.02393](https://arxiv.org/pdf/2301.02393). Neither is described as public (see Q5).

### Inferences
- For a subclavian-origin → basilar VA you would currently have to stitch sources: AortaSeg24 or AVT for the arch, subclavian and VA-origin region; unlabelled neck coverage from some other public CTA; TopBrain for the intracranial VA, basilar and branches. The cervical V1–V3 would still have to be annotated by you or taken from a non-challenge dataset (another researcher's scope).
- AortaSeg24's rule ("each branch ends at its first bifurcation") means the LSA label probably ends near the VA origin, since the VA is usually the first LSA branch. This is anatomical inference: the paper does not name the VA as the stopping point. It would give a clean subclavian-origin stub for grafting.

### Gaps
- How far down the TopBrain VA label extends (V4 only, or down to C1/C2?) is not stated in the grand-challenge page, the Zenodo record, or the parts of the medRxiv paper I could read. The paper's Table 1 "Label Definition" was not machine-readable in the fetched PDF, and no local PDF text extractor was available. The check is cheap: download the open 2 GB Zenodo archive (Zenodo 16878417 / 21972006) and look at the most caudal slice of labels 23/24 compared with the foramen magnum.

## Q2. Which challenges label the VA only intracranially, or not at all? (candidates to exclude)

### Takeaway
Exclude, or keep only for the intracranial end: TopCoW 2023/2024 (no VA class), TopBrain (VA present but, on the evidence, cropped to the braincase), RSNA Aneurysm 2025 (no VA class), CAS 2023, CROWN, SMILE-UHURA, IntrA and CADA. All are intracranial.

### Cited Findings

#### TopCoW 2023 / 2024 (MICCAI 2023, 2024) — https://topcow23.grand-challenge.org / https://topcow24.grand-challenge.org
- Modalities: paired CTA and TOF-MRA — [TopCoW24 data](https://topcow24.grand-challenge.org/data/).
- Cases:
  - 2023 edition: 90 training patients, 5 validation, 35 test, plus 20 MRAs from CROWN. Data are "Multi-site from Switzerland", from USZ Stroke Center patients (2018–2019) — [TopCoW paper](https://arxiv.org/html/2312.17670v2).
  - 2024 edition: 125 training patients (250 scans), 5 validation (images only), 70 test "not publicly released" — [TopCoW24 data](https://topcow24.grand-challenge.org/data/).
- Labels: BA, R/L-PCA, R/L-ICA, R/L-MCA, R/L-Pcom, Acom, R/L-ACA and a third A2. "The vertebral artery is not labeled" — [TopCoW paper](https://arxiv.org/html/2312.17670v2); [TopCoW24 data](https://topcow24.grand-challenge.org/data/).
- Field of view:
  - "CTA usually had the whole head scanned." All data were "anonymized, defaced and cropped to the braincase region". The CTA z-extent is 148.96 ± 8.87 mm; the MRA z-extent is 111.53 ± 11.07 mm — [TopCoW paper](https://arxiv.org/html/2312.17670v2).
  - The 2024 page also says facial regions are masked and ROI bounding boxes are provided — [TopCoW24 data](https://topcow24.grand-challenge.org/data/).
- Format: NIfTI (16-bit signed) images and masks, YAML edge lists, text ROI files — [TopCoW24 data](https://topcow24.grand-challenge.org/data/).
- Access:
  - Training and validation data released under CC BY-NC — [TopCoW paper](https://arxiv.org/html/2312.17670v2).
  - Permanent record at Zenodo 15692630 since June 2025; attribution required and commercial use needs the data owner's permission — [TopCoW24 data](https://topcow24.grand-challenge.org/data/).

#### TopBrain 2025 (MICCAI 2025) — https://topbrain2025.grand-challenge.org
- 90 annotated volumes over 48 landmark vessel classes (arterial and venous). 50 training volumes are public and 40 test volumes are hidden, split equally between CTA and MRA (25 + 25 training, 20 + 20 test) — [medRxiv TopBrain paper](https://www.medrxiv.org/content/10.64898/2026.05.28.26354312v1.full); [Zenodo 16878417](https://zenodo.org/records/16878417).
- CTA has 40 labels, MRA 42, with 34 shared. The shared labels include BA, PCA P1/P2 and P3/P4, ICA, MCA M1–M3, ACA A1–A3, Pcom, Acom, **R/L VA**, SCA, AICA, PICA, AChA and OA. CTA-only labels are the venous sinuses/veins; MRA-only labels are the ECA, STA, maxillary and middle meningeal arteries. Labels 1–12 and 15 are kept from TopCoW — [TopBrain data](https://topbrain2025.grand-challenge.org/data/).
- Images come from a "stroke-center origin"; the fetched text gives no field-of-view statement — [medRxiv TopBrain paper](https://www.medrxiv.org/content/10.64898/2026.05.28.26354312v1.full).
- Format: NIfTI, 16-bit signed, LPS+. Access: Zenodo 16878417, open. Licence: "Open use. Must provide the source. Use for commercial purposes requires permission" — [TopBrain data](https://topbrain2025.grand-challenge.org/data/); [Zenodo 16878417](https://zenodo.org/records/16878417).
- Timeline: training batches released 30 Jul and 14 Aug 2025; MICCAI event 23 Sep 2025; paper on medRxiv 30 May 2026 — [TopBrain overview](https://topbrain2025.grand-challenge.org/).
- **TopBrain v2 / TopBrain 2026:**
  - Zenodo 21972006 (v3 release, 17 Aug 2026) adds a 36-class, TopAneu-compatible label set, with new infraclinoid ICA classes separating C1–C5 from C6–C7 — [Zenodo 21972006](https://zenodo.org/records/21972006).
  - The TA36 track "excludes the extracarotid artery (ECA) classes". Vein and stenosis tasks run October–December 2026 — [TopBrain 2026](https://topbrain2026.grand-challenge.org/).
  - The V2 design document on Zenodo is CC BY 4.0 and dated 23 Apr 2026 — [Zenodo 19707577](https://zenodo.org/records/19707577).

#### RSNA Intracranial Aneurysm Detection 2025 (Kaggle) — https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
- 13 aneurysm locations: L/R infraclinoid ICA, L/R supraclinoid ICA, L/R MCA, L/R ACA, L/R Pcom, Acom, basilar tip, and "other posterior circulation". 13-class vessel segmentation masks are provided for 178 cases — [arXiv 2606.26706](https://arxiv.org/html/2606.26706).
- The RSNA-ICA dataset has over 4,000 scans, of which "about 200" studies carry AI-generated segmentations — [RSNA-ICA on AWS Open Data](https://registry.opendata.aws/rsna-intracranial-aneurysm-detection-dataset/). Conflicting counts: 178 (arXiv) vs "about 200" (AWS/RSNA).
- A 26-class community derivative (13 vessel + 13 aneurysm classes) is on Hugging Face as NIfTI uint8 under CC BY-NC 4.0; the DICOMs still have to come from Kaggle — [HF spc819/rsna2025-aneurysm-26class-seg](https://huggingface.co/datasets/spc819/rsna2025-aneurysm-26class-seg).

#### CAS 2023 — Cerebral Artery Segmentation (MICCAI 2023)
- TOF-MRA from 100 patients with symptomatic intracranial artery stenosis. Submissions are closed. Prizes were $4,000 / $1,500 / $500 (TS Imaging) — [CodaLab CAS2023](https://codalab.lisn.upsaclay.fr/competitions/9804); [Society for MRA](https://society4mra.org/index.php/2023/06/14/cerebral-artery-segmentation-challenge-cas2023/).
- A later paper reports 100 volumes (median grid 608×640×150), split 85 training / 15 test — via the search summary of [MDPI J. Imaging 11(6):202](https://www.mdpi.com/2313-433x/11/6/202). Secondary source; the voxel size quoted there (1 mm isotropic) looks doubtful for TOF.

#### CROWN (MICCAI 2023)
- Tasks: classify CoW anatomical variants, and predict CoW artery diameters and bifurcation angles. 300 TOF-MRA training + 300 test (as reported in the results summary) — [OpenReview CROWN results](https://openreview.net/forum?id=5R2cnoAOiR).
- TopCoW reused 20 CROWN MRAs — [TopCoW paper](https://arxiv.org/html/2312.17670v2).

#### SMILE-UHURA (ISBI 2023)
- 7T TOF-MRA at 300 µm isotropic, from StudyForrest. 14 training/validation MRAs + 4 held-out test; the second test set is secret. Focus is small (mesoscopic) intracranial vessels — [arXiv 2411.09593](https://arxiv.org/abs/2411.09593).

#### IntrA (CVPR 2020 dataset, not a challenge)
- 103 whole-cerebral-vasculature surface models reconstructed from MRA; 1,909 vessel segments (215 aneurysmal); 116 aneurysm segments annotated manually; OBJ/AD point formats — [IntrA GitHub](https://github.com/intra3d2019/IntrA); [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_IntrA_3D_Intracranial_Aneurysm_Dataset_for_Deep_Learning_CVPR_2020_paper.pdf).

#### CADA 2020 (3D rotational angiography, i.e. DSA-3D)
- 109 training + 22 test 3DRA volumes. The volume of interest was chosen by a neurosurgeon: ~220 slices of 256×256 at ~0.5 mm. Tasks: aneurysm detection, segmentation and rupture risk — [CADA dataset info](https://cada.grand-challenge.org/Dataset/); [CADA MedIA paper](https://pubmed.ncbi.nlm.nih.gov/34998111/).

### Inferences
- TopBrain's VA is very likely intracranial only. It reuses TopCoW labels and the same USZ stroke data pipeline, and TopCoW data were "cropped to the braincase region". A braincase crop would at most take in the V3 loop at C1, not V1/V2. This is inference, not a documented fact; see the Q1 gap for how to check it.
- In TopBrain v2, "C1–C5" is an ICA segmentation (Bouthillier), not a vertebral level. C1 is the cervical ICA, so the new infraclinoid class may take in some upper-neck ICA inside the crop. It says nothing about the VA.
- TopBrain is still useful for the intracranial end of the graft: the VA (V4) → basilar junction together with PICA, AICA and SCA.
- The RSNA vessel masks have no VA class. At most, "other posterior circulation" may cover the intracranial VA (unverified).

### Gaps
- The exact RSNA 13-vessel-class names, the field of view of the segmented series, and the Kaggle data licence were not verified: the RSNA wiki page was a "Coming soon" stub.
- The IntrA and CROWN licences and access procedures were not verified.

## Q3. Do the aortic challenges (AVT/SEG.A, AortaSeg24) include the supra-aortic branches, subclavians and VAs?

### Takeaway
Both include the arch branches but not the VA:
- **AortaSeg24** labels the innominate artery, left common carotid and left subclavian as separate classes, each ending at its first bifurcation. There is no VA and no right subclavian.
- **AVT/SEG.A** is a single binary aortic-tree mask that includes the brachiocephalic trunk, left common carotid and left subclavian. VAs are not mentioned, and the cranial extent of the branches is not specified.

### Cited Findings

#### AortaSeg24 (MICCAI 2024) — https://aortaseg24.grand-challenge.org
- 100 CTA volumes from 100 patients with uncomplicated type B aortic dissection (uTBAD): 50 annotated training, 10 hidden validation, 40 hidden test — [arXiv 2502.05330](https://arxiv.org/html/2502.05330).
- 23 classes:
  - 13 branches: innominate, left common carotid, left subclavian, celiac, SMA, L/R renal, L/R common iliac (Zone 10), L/R external iliac (Zone 11), L/R internal iliac.
  - 10 aortic zones (Zones 0–9).
  - No vertebral artery and no right subclavian class — [arXiv 2502.05330](https://arxiv.org/html/2502.05330).
- Branch extent: "Each branch segmentation ended at the bifurcation point where it split into multiple branches." Field of view: "From the aortic root to the bifurcation of the aorta into the left and right common iliac arteries" — [arXiv 2502.05330](https://arxiv.org/html/2502.05330).
- Access: a verified Grand Challenge account, a DocuSign data access agreement, a request to join the challenge, then a Dropbox access request; approval "may take up to 24 hours" — [AortaSeg24 access page](https://aortaseg24.grand-challenge.org/dataset-access-information/). The paper describes the data as publicly available under DUAs — [arXiv 2502.05330](https://arxiv.org/html/2502.05330).

#### AVT dataset / SEG.A 2023 (MICCAI 2023) — https://multicenteraorta.grand-challenge.org
- 56 CTAs: KiTS19 (20), Rider Lung CT (18), Dongyang Hospital (18). The SEG.A test set (7 real + 150 synthetic) is withheld for future editions — [SEG.A overview paper](https://arxiv.org/html/2510.24009).
- Coverage:
  - "Each CTA scan covers the ascending aorta, the aortic arch and its branches into the head/neck area, the thoracic aorta, the abdominal aorta and the lower abdominal aorta with the iliac arteries."
  - "All cases include the ascending aorta, the aortic arch, the brachiocephalic, the left common carotid, the left subclavian artery, the thoracic aorta, the abdominal aorta and the iliac arteries."
  - Sources: [AVT Data in Brief](https://pmc.ncbi.nlm.nih.gov/articles/PMC8760499/) and the [search summary of the same](https://www.sciencedirect.com/science/article/pii/S2352340922000130).
- Format: binary masks as .seg.nrrd. Licence: CC BY on figshare (doi:10.6084/m9.figshare.14806362) — [AVT paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8760499/); [figshare](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362).

### Inferences
- AVT is the only CC BY (commercial-friendly) option that gives arch → subclavian geometry, but the mask is binary. Separating out the subclavian, or a VA origin if present, would need post-processing such as centreline extraction and branch labelling.
- Pairing AVT's arch geometry with an intracranial source is a plausible hybrid route.
- There is a possible inconsistency: KiTS19 scans are kidney CTs, yet the paper says all cases include the arch and supra-aortic branches. Check the actual neck coverage per case before relying on it.

### Gaps
- Whether any AVT masks contain VA stubs, and how far cranially the branches run in each cohort, is not documented. It needs inspection of the figshare data.
- The AortaSeg24 data licence text (beyond the DUA) was not retrieved. The CC BY-NC-SA shown on arXiv covers the paper, not necessarily the data.

## Q4. Do the neck and carotid challenges (HaN-Seg, carotid vessel wall, COSMOS, CLS2009) label the VA?

### Takeaway
No. They label the carotid only (HaN-Seg carotid OAR, carotid vessel-wall MRI, carotid-bifurcation lumen CTA), so none gives VA geometry.

### Cited Findings

#### HaN-Seg 2023 (MICCAI 2023)
- 42 public patients with CT + T1 MR (radiotherapy planning), plus 14 test patients. 30 OARs as binary masks — [Zenodo 7442914](https://zenodo.org/records/7442914).
- OARs use AAPM TG-263 names and include the carotid artery (A_Carotid_L / A_Carotid_R). No vertebral artery appears in the listed OARs — [HaN-Seg search summary / Medical Physics paper](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16197).
- Licence: CC BY-NC-ND 4.0, open on Zenodo; cite Podobnik et al., Med Phys 2023 — [Zenodo 7442914](https://zenodo.org/records/7442914); [HaN-Seg challenge](https://han-seg2023.grand-challenge.org/).

#### Carotid Artery Vessel Wall Segmentation Challenge 2021 (MICCAI 2021)
- 3D-MERGE black-blood MRI, "large coverage of carotid arteries". Tasks: localisation, lumen and outer-wall contours. The Zenodo record is CC BY-ND 4.0 (published 3 Mar 2021) — [Zenodo 4575301](https://zenodo.org/records/4575301).

#### COSMOS 2022 (Carotid Vessel Wall Segmentation and Atherosclerosis Diagnosis, MICCAI 2022)
- 3D-VISTA MRI, 50 training/validation + 25 test cases. Labelled slices: 934 in training/validation, 268 in test — [search summary](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/COSMOS2022.md).
- Training data on Zenodo — [Zenodo 6481870](https://zenodo.org/records/6481870); [challenge](https://vessel-wall-segmentation-2022.grand-challenge.org/).

#### CLS2009 (MICCAI 2009 Carotid Lumen Segmentation and Stenosis Grading)
- 56 multi-centre CTAs (15 training, 41 test; test annotations not public), from three centres. Framework for "carotid bifurcation lumen segmentation and stenosis grading". Site: cls2009.bigr.nl — [Hameeteman et al., MedIA 2011](https://pubmed.ncbi.nlm.nih.gov/21419689/); [QIMS summary](https://qims.amegroups.org/article/view/50255/html).

### Inferences
- The raw neck CTAs in CLS2009 are head-and-neck CTAs, so they probably show the cervical VAs even though only the carotid is labelled. They could be re-annotated, depending on licence and field of view.

### Gaps
- How far the HaN-Seg carotid runs (CCA and/or ICA; caudal and cranial limits) was not retrieved.
- The CLS2009 data licence and the current state of its website (bigr.nl) were not verified.

## Q5. Is there a dedicated "head and neck CTA vessel segmentation" or "vertebral artery" challenge (including Chinese ones)?

### Takeaway
I found no public, verified challenge release dedicated to multi-class head-and-neck CTA arteries with the VA labelled. The candidates are in-house sets (HNCTA 358, HNA 800, Fu et al. 18,766) and an unverified Zhihu page titled "HeadNeckCTA2025 — 头颈CTA血管分割" ("head-and-neck CTA vessel segmentation"). That page appears to describe the Fu et al. multi-hospital data rather than an open challenge.

### Cited Findings
- **Fu et al., Nature Communications 2020:** 18,766 head-and-neck CTAs from 5 tertiary hospitals in China (June 2017 – November 2018) — [Nat Commun](https://www.nature.com/articles/s41467-020-18606-2); [PMC7518426](https://pmc.ncbi.nlm.nih.gov/articles/PMC7518426/). The data-availability statement could not be read: PMC returned a CAPTCHA and nature.com redirected to login.
- **Zhihu "HeadNeckCTA2025":**
  - The search snippet describes 18,766 contrast head-and-neck CTAs: Xuanwu 8,593; Beijing Friendship 4,407; Shanghai Sixth 2,584; Hebei 2,096; Shandong 1,086. Each scan covers the aorta, carotids and intracranial arteries, with about 561–967 z-slices.
  - The page itself returned HTTP 403 — [Zhihu](https://zhuanlan.zhihu.com/p/29607061020). Unverified.
- **HNCTA:** 358 head-and-neck CTAs with 9 vessel classes (L/R VA, ICA, CCA, PoCA, BA), split 242/58/58. Created in-house, not public — [DeformCL arXiv 2506.05820](https://arxiv.org/html/2506.05820v1).
- **HNA:** 800 in-house head-and-neck CTAs with cerebral, vertebral and carotid arteries annotated, split 640/80/80 — [arXiv 2301.02393](https://arxiv.org/pdf/2301.02393).
- **LIVE-Net:** reports a mean DSC of 90.03% for bilateral carotid and vertebral arteries on head-and-neck CTA (in-house data) — [Comput Biol Med 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523003517).
- **Chinese expert consensus (2024) on head-and-neck CTA data annotation and quality control:** its annotation targets include the bilateral VAs — [brainmed](https://www.brainmed.com/info/detail?id=47109) (search snippet only).
- **SemanticVessel:** a fine-grained brain-vessel CTA set (20 arterial classes); "data will be made available". VA extent not stated — [arXiv 2606.21756](https://arxiv.org/abs/2606.21756).
- **"VerSe"** is a vertebra (spine) segmentation challenge, not a vertebral-artery one. Searches for a "vertebral artery challenge" returned nothing relevant. This is a negative search result, not a sourced fact.

### Inferences
- Large Chinese head-and-neck CTA vessel sets with full-length VA labels do exist, but they sit with hospitals or companies. Access would mean contacting the authors, not downloading a public challenge release.
- ISLES'24 and RSNA-ICA CTA images are public and may contain the neck, but they carry no vessel labels for the neck. Using them would mean annotating yourself; their field of view is unverified.

### Gaps
- Whether "HeadNeckCTA2025" is a real, open challenge (organiser, URL, licence) could not be verified: Zhihu returned 403 and no other source mentioned it.
- The Fu et al. data-availability terms could not be read.
- A search in Chinese found no Tianchi or other Chinese challenge page for head-and-neck CTA vessels; results were limited.

## Q6. Which of the requested challenges are not relevant (no vessel labels, or wrong vascular bed)?

### Takeaway
ISLES, VALDO, Parse2022 and BraTS give no usable VA geometry: ISLES has stroke-lesion masks, VALDO has small-vessel-disease markers, Parse2022 has pulmonary arteries, and BraTS has tumours.

### Cited Findings
- **ISLES'24:**
  - Contents: NCCT, CTA and CTP (<24 h), post-interventional MRI 2–9 days later, and clinical data.
  - Labels: binary infarct masks only — [ISLES'24 Radiology: AI](https://doi.org/10.1148/ryai.250603); [Zenodo 10991145](https://zenodo.org/records/10991145); [isles-24.grand-challenge.org](https://isles-24.grand-challenge.org/).
  - ISLES 2022 is MRI stroke lesions — [ISLES 2022](https://arxiv.org/pdf/2206.06694).
- **VALDO (MICCAI 2021):** enlarged perivascular spaces, cerebral microbleeds and lacunes on MRI; 12 teams took part — [arXiv 2208.07167](https://arxiv.org/abs/2208.07167).
- **Parse2022 (MICCAI 2022):** 200 CTPAs from Harbin Medical University (100 training / 30 open validation / 70 closed test), pulmonary artery labels only — [PARSE paper](https://arxiv.org/pdf/2304.03708); [parse2022.grand-challenge.org](https://parse2022.grand-challenge.org/Dataset/).

### Inferences
- ISLES'24 CTAs are open image data that may show the cervical VAs. They are a possible raw source for your own annotation, but not labels.

### Gaps
- The ISLES'24 CTA field of view (head only, or aortic arch to vertex) was not checked.
- The "Contact-aware Path Planning for Autonomous Neuroendovascular Navigation" paper (arXiv 2601.07945) turned up alongside AVT in search. It may show how others built navigation phantoms from public data, but it was not read.
