# Verify 7: CAS2023 脑动脉分割挑战赛 (Cerebral Artery Segmentation Challenge, MICCAI 2023)

Verified 2026-10-01 by an adversarial check of the searcher's candidate. Earlier notes already list CAS2023 as an exclusion:
- `Public cervical vertebral artery datasets/challenges.md` (line 27, "100 (85 training / 15 test)");
- `Chinese cervical vertebral artery datasets/challenges.md` (line 23 and lines 94–96).

This note checks the claim against the host pages themselves.

## Verdict

**Exclude. CAS2023 holds no cervical vertebral artery.**

- **Coverage.** The images are intracranial ("head") TOF-MRA. The label is one binary class, "cerebral artery".
- **What "vertebral artery" means here.** The proposal mentions the VA only as one of the *intracranial* stenosis sites (V4), not as a labelled cervical segment.
- **Case count.** The searcher's "100 (85/15)" is wrong. The proposal and CodaLab give **150 cases: 100 training + 50 test**.
- **Licence.** The searcher's "CC BY-ND 4.0" applies only to the proposal PDF on Zenodo. The *data* terms are **"CC BY NC"**, plus a clause limiting use to the challenge.
- **Access.** The data was gated behind CodaLab registration. CodaLab Competitions was phased out at the end of 2025. I found no mirror.

| field | finding |
|---|---|
| Name | 脑动脉分割挑战赛 CAS2023 / Cerebral Artery Segmentation Challenge (CAS) 2023 |
| URLs | Proposal: https://zenodo.org/records/7839970 (DOI 10.5281/zenodo.7839970). Data: https://codalab.lisn.upsaclay.fr/competitions/9804 |
| Host | Tsinghua University Center for Biomedical Imaging Research + University of Washington; co-hosted with SMRA 2023; CodaLab (LISN) |
| Modality | 3D TOF-MRA, 3T Philips/Siemens, "8-channel phase-array head coil or 16-channel neurovascular coil" |
| Cases | 150 = 100 training + 50 test (5 Beijing/Hebei centres) |
| Cervical VA labelled? | **No.** One binary "cerebral artery" mask on intracranial TOF-MRA. There is no VA class and nothing from the origin or the neck. |
| Label format | nii.gz binary masks, plus an Excel sheet of stenosis positions ("pos_s" column) |
| Access | CodaLab account → Participate → Files → "Public Data" (training zip) / "Starting Kit" (50 test cases). Not verified as still downloadable in 2026. |
| Terms | CodaLab T&C: "CC BY NC." Requirement 1: use is "restricted solely to this challenge". Proposal: other uses need organiser approval; no commercial use. |
| Zenodo record contents | One file only: `CerebralarterysegmentationChallenge_04-18-2023_10-38-11.pdf`, 2,668,252 bytes, md5 8f39b088c1ef400e2059022ce825c4b7, licence cc-by-nd-4.0. **It contains no images or labels.** |

## Evidence: what I opened

### 1. Zenodo 7839970 (opened; API `https://zenodo.org/api/records/7839970`)
- The record's only file is the 13-page BIAS challenge proposal PDF. I downloaded it and checked its md5 against the API value.
- Licence `cc-by-nd-4.0`, `access_right: open`. These describe the PDF, not the data.
- Description: "the task is to segment cerebral arteries from 3D TOF-MRA images acquired from patients with symptomatic intracranial artery stenosis."

### 2. Proposal PDF (read in full via pdftotext)
**Coverage: intracranial or head only.**
- "3D TOF: All the **intracranial** TOF-MRA was acquired on 3T MR scanners by the routine imaging protocol used in each institution … Scanned in the axial direction."
- Data origin: "**Head** MR exams using 3D TOF-MRA sequence."
- Case definition: "Training and test cases both represent 3D TOF-MRA scans of the **brain**. Each slice in the data set will have manually annotated mask on cerebral artery."
- Algorithm target: "The cerebral arteries, especially the arteries near the stenosis region."

**The only VA mention.**
- It names stenosis sites, not a labelled cervical segment: "Each subject has at least one **intracranial** artery stenosis in the following artery segments: anterior cerebral artery, middle cerebral artery, posterior cerebral artery, internal carotid artery, vertebral artery or basilar artery."
- These are intracranial stenoses, so the VA meant is the intracranial V4.

**Labels.**
- "Our labeling target is all the visible cerebral blood vessels on TOF-MRA." The labels are binary: one vessel class, plus the stenosis region for scoring.
- Annotation was semi-automatic in MeVisLab: region-growing, then refinement by 5 junior reviewers and checking by 1 senior reviewer.
- "The MRI images and segmentation labels will be provided in nii.gz format."

**Cases.**
- "Training cases: 100, Closed test cases: 50."
- "Totally, TOF-MRA data of 150 cases from these institutions were pooled together and randomly divided into the training/testing set (100/50)."
- **This refutes the "85 training / 15 test" figure in the earlier notes.**

**Terms.**
- Data usage agreement: "CC BY NC."
- "Data used in this challenge are limited to the purpose of developing vessel segmentation algorithms. Other requests require approval from the organizers. After the challenge, new teams can still access the training data, and the test data will be released after the challenge."
- "No commercial uses are allowed."
- Test labels "will be released (Oct 1st, 2023) and can be used by the participants".

### 3. CodaLab competition 9804 (opened; raw HTML fetched with curl)
- Status "(Ended)", "Submission closed". "You must be logged in to participate in competitions."
- 2023/05/15 update: "The training dataset has been uploaded. Please go to Participate > Files > Public Data to download the zip file".
- Submission step 1: "Download the test set data (50 cases) from 'Participate > Files > Starting Kit.'"
- Terms and Conditions tab, verbatim: "CC BY NC."
- Requirement 1, verbatim: "Use of images will be restricted solely to this challenge, participants must agree not to use the images in any other capacity."
- The leaderboard has 21 teams, including non-Chinese ones (CREATIS-MYRIAD, UW, Davide Resta, Theresa Nolte). Registration was therefore open internationally, with no Chinese ID or phone verification.

### 4. CodaLab platform status (opened codalab.lisn.upsaclay.fr home page)
- Home-page notice: "CodaLab will be phased out in 2025 and will not allow new competitions after the end of September 2025."
- "Submissions will be disabled at the end of year (December 31, 2025)."
- The Codabench newsletter "CodaLab in 2025" says: "CodaLab Competitions and its main servers were finally phased out at end of 2025."
- On 2026-10-01 the competition page and the Sign Up page still return HTTP 200.
- **Not verified:** whether a new account can still reach Participate → Files and download. I did not create an account.

### 5. Mirrors (searched; none found)
- Hugging Face datasets API: searches for `CAS2023`, `CAS_2023`, `cas2023` returned `[]`.
- GitHub repo search for `CAS2023`: only an unrelated archival-science workshop repo.
- WebSearch, English and Chinese (CAS2023 数据集 下载 百度网盘 / huggingface / kaggle / opendatalab; 脑动脉分割挑战赛 CAS2023 清华): found no re-hosted copy.

### 6. Papers using the data (for image extent)
- PMC12194608 (MedSAM/MedSAM2 + nnUNet, 2025) says: "This study used the publicly available CAS2023 dataset, comprising TOF-MRA 3D imaging data (in NIIGZ format) from 100 stroke patients. During preprocessing, all 3D images were cropped along the Z-axis and converted into 2D slices, ultimately generating 14,189 2D images".
  - That is about 142 axial slices per case, which fits an intracranial slab.
- An ISMRM 2024 abstract (3023) says "The CAS2023 dataset, which comprises 100 cases with reference annotations". It gives no coverage details.
- **Gap:** I found no published voxel-spacing or FOV table for CAS2023. The intracranial-only conclusion rests on the organisers' own words ("intracranial TOF-MRA", "head MR exams", "scans of the brain").

### Could not open
- society4mra.org CAS2023 post: HTTP 404, now gone.
- CodaLab Participate tab: needs login, so the file list and zip names inside "Public Data" were not seen.

## Bottom line for the cervical-VA search
CAS2023 cannot supply cervical VA geometry.
- **Coverage:** the images are intracranial TOF-MRA. Any VA in them is the distal V4 near the basilar, at most.
- **Labels:** the mask is one undifferentiated "cerebral artery" class.
- **Terms:** use is limited to the challenge (vessel-segmentation development). Grafting geometry into a simulator would need written approval from the organisers (shz21@mails.tsinghua.edu.cn).

It is no better than TopBrain/TopCoW as an intracranial reference, so keep it in the exclusion list. The earlier notes should fix the case count to 150 (100 + 50) and the data licence to CC BY-NC with the challenge-only clause.
