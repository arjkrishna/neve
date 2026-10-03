# Chinese sources add one open vertebral-artery case, and China-hosted platforms add none

This report follows up the English-language search in [`Public cervical vertebral artery datasets.md`](Public%20cervical%20vertebral%20artery%20datasets.md), which named Chinese-hosted repositories as its main blind spot. It does not repeat that report's findings. Searched 2026-10-01, in Chinese and English.

**Short answer: no Chinese-hosted repository offers head-and-neck CTA or MRA with labelled cervical vertebral arteries (VAs, 椎动脉), whether by download or by application.** Four parallel searches covered:

- the native search APIs of 18 Chinese open-data platforms: OpenDataLab (浦源), ModelScope (魔搭), ScienceDB (科学数据银行), the national data centres NCMI and NBSDC (both through escience.org.cn, 中国科技资源共享网), Heywhale (和鲸), Gitee (码云), OpenI (启智), HyperAI (超神经) and PKU Dataverse, among others;
- all 577 Tianchi (天池) competitions, and Chinese-run MICCAI, ISBI, Grand Challenge and Codabench challenges;
- Chinese papers, journals and theses;
- Chinese community posts on WeChat, CSDN, Bilibili and Juejin.

Searching 椎动脉 ("vertebral artery") returned no imaging dataset on any native search that worked. Every Chinese multi-patient corpus that labels the VA is private in its authors' own words. A 2024 national annotation consensus explains why: it specifies whole-VA labels on arch-to-vertex CTAs, but for training and testing commercial AI products and for regulatory test sets, not for release.

**The Chinese-language pass did find one thing the English search missed, though it is hosted on Google Drive rather than in China: the TaG-Net sample data** (Yao et al., IEEE TMI 2023; authors from Shanghai Jiao Tong University and United Imaging Intelligence 联影智能, among others). It contains:

- **one head-and-neck CTA** at 0.405×0.405×0.4 mm;
- **18-class labelled centerlines** in which both VAs run from the subclavian label to the basilar label, about 146 mm vertically;
- **a second patient's** labelled centerlines, without the image.

That makes **three real people with ready-made origin-to-basilar VAs** (VMR 0248, the SPL atlas and TaG-Net) instead of two. It also gives the Arterial benchmark a second sub-millimetre case. Two caveats apply: the labels are 1-voxel centerlines with no lumen, and no data licence is stated.

"HeadNeckCTA2025" on Zhihu (知乎) is a blogger's tutorial about the private Fu et al. 2020 cohort. It is not a dataset, and its "data download address" (数据下载地址) is the PubMed page of that paper.

How each status was established:

- **Verified:** an independent check of the host pages (notes `verify_0`–`verify_7`).
- **Re-checked:** TaG-Net was not in that verification pass. While writing this report I listed its Drive folder again, downloaded both label files and re-measured them. Its "open" status rests on that re-check.
- **Searcher only:** everything else comes from the searchers' reading of primary sources.

The notes are in `research_notes/Chinese cervical vertebral artery datasets/`: `platforms.md`, `challenges.md`, `papers.md`, `community.md` and `verify_0.md`–`verify_7.md`.

## TaG-Net's sample is a third real origin-to-basilar VA pair

**The authors of "TaG-Net: Topology-Aware Graph Network for Centerline-Based Vessel Labeling" host sample data on a public Google Drive folder.** Their README says: "Sample data, the corresponding ground truth, and our result can be downloaded at Google Drive" ([PRESENT-Y/TaG-Net](https://github.com/PRESENT-Y/TaG-Net); [Drive folder](https://drive.google.com/drive/folders/1Q1GoRfvVZsSgxbia60jANJscOkKXetyx)). The files download with no login, and HTTP Range requests work. A range request on `CTA.nii.gz` returned `206` with `content-range: bytes 0-351/192162690` on 2026-10-01.

The code labels 18 head-and-neck segments, including both subclavians (SCA), both VAs, the basilar (BA), the arch, the brachiocephalic trunk, the carotids and the Circle of Willis. Its comments state the topology as "L-VA(12) -> L-SCA(14) & BA(8)" and "R-VA(7) -> R-SCA(13) & BA(8)" ([utils_multicl.py](https://github.com/PRESENT-Y/TaG-Net/blob/main/VesselCompletion/utils_multicl.py)).

The comments swap two names: what they call "BCT(1)" connects to the left SCA and left CCA, so it is the arch, and "AO(2)" is the brachiocephalic trunk. The geometry agrees: NIfTI label 2 sits lowest, at z 5.6–24.4 mm.

**Re-measured from the downloaded label volumes** (NIfTI value = txt label + 1; z is slice index × 0.4 mm):

| Case | Grid, spacing | R-VA | L-VA | R-SCA / L-SCA | BA |
|---|---|---|---|---|---|
| `SampleData/001/labeled_cl_gt.nii.gz` | 512×512×796, 0.405×0.405×0.4 mm | 506 voxels, z 70.4–216.8 mm | 580 voxels, z 67.6–216.4 mm | z 46.8–74.0 / 24.8–74.8 mm | z 216.8–235.6 mm |
| `SampleData/003/labeled_cl_gt.nii.gz` | 512×512×743, 0.366×0.366×0.4 mm | 499 voxels, z 57.2–202.8 mm | 507 voxels, z 56.4–201.2 mm | z 31.6–56.8 / 32.4–57.2 mm | z 201.6–228.8 mm |

These match the searcher's figures exactly. Each VA's lowest voxel sits inside its subclavian's z-range, and its highest voxel reaches the basilar's lowest slice. The searcher's 3D nearest-neighbour distances were 0.4–0.7 mm to the subclavian label and 0.5–1.0 mm to the basilar label. **These are cervical VAs from origin to confluence, about 145–149 mm vertically.**

**The CTA and the case-001 labels belong to one patient.** `SampleData/002/CTA.nii.gz` (int16, 192 MB) is on exactly the same 512×512×796 / 0.405 / 0.4 mm grid as `001/labeled_cl_gt.nii.gz`. Both headers carry the same qform: code 1, quaternion (0, 0, 1), origin (103.5, 137.9, −723) mm. The searcher also found that the labelled centerline points in folder 002 agree 99% with the 001 ground truth where they overlap. Case 003 has labels but no image.

`papers.md` says the headers store no orientation; that is wrong, because the qform is present.

Four limits:

- **Centerline only.** There is no lumen mask or radius. A graftable wall needs a lumen segmentation of the 0.4 mm CTA seeded from the labelled centerline, for example with VMTK level sets.
- **The scan starts at the top of the arch** (arch label at z 5.6–24.4 mm). The VA origins at z ≈ 68–70 mm are well inside it.
- **No data licence.** The code is GPL-3.0, and the data is described only as "sample data for testing". Ask the contact address, yaolinlin23@sjtu.edu.cn, before redistributing derived meshes.
- **Annotation method not stated.** Whether the ground truth is fully manual is not said in what could be read. The abstract says centerline labels were assigned to a refined segmentation. The full 401-subject set is not offered, and the trained models the README mentions are not in the Drive folder.

## China-hosted platforms hold images at best, never VA labels

Native searches in Chinese found head-and-neck angiography in only three places, and none carries VA labels:

- **Beijing Tiantan Hospital's 「心、脑、外周血管疾病医学影像数据集」 ("imaging dataset for heart, brain and peripheral vascular diseases"), on NBSDC (国家基础学科公共科学数据中心).** It was built for a national catheter-intervention robotics project (2018AAA0102600), the same goal as ours.
  - Its description file, downloaded anonymously, says 「本数据集包含以DICOM形式存储的原始医学影像」 ("raw medical images stored as DICOM") and 「脑血管影像模态为TOF-MRA」 ("the cerebrovascular modality is TOF-MRA").
  - The file tree (40 files, 207 GB) holds no mask, NIfTI or mesh file of any kind.
  - Data files return 403 until an application is approved. Registration requires a real name in Chinese characters and a mainland mobile number. The application form must be signed and stamped with the institution's official seal (加盖公章), then approved by both NBSDC and Tiantan.
  - NBSDC records 0 downloads. **Verified (verify_1): not public, no labels.**
- **NCMI's 「医学图像标准测试数据库-脑卒中影像与标注数据」 ("medical-image standard test database: stroke images and annotations"), on 国家人口健康科学数据中心.**
  - It includes 433 head-and-neck aneurysm CTAs from the PLA General Hospital (解放军总医院). Each comes with only 「医生勾画的感兴趣区域」 ("a physician-drawn region of interest").
  - Access is 有条件共享 / 线下共享 ("conditional sharing, offline") through a real-name PHDA application. Commercial use costs a fee, and the field of view is unverified.
  - **Verified (verify_2): not public, no vessel labels.**
- **CereVessMRA on ScienceDB, the one open download.** It is CC BY-NC 4.0, 271 TOF-MRA volumes.
  - The verifier pulled the archive by byte range. The labels are binary 0/1 masks, and every slab is 73.6–89.6 mm tall and ends at about the skull base.
  - **Verified (verify_4): public, but no cervical VA.**

Both gated sets would at best supply unlabelled images for Arterial. TotalSegmentator's 123 CC BY 4.0 neck CTAs already provide that without an application.

Everything else the platforms returned was intracranial, aneurysm-only, ultrasound, clinical tables or re-uploads of known sets; see the exclusion table below. The only 椎动脉 hits anywhere were an ultrasound teaching set, a traditional-Chinese-medicine text corpus and clinical tables.

## Every Chinese corpus with VA labels is private

**Fu et al. 2020 (宣武医院 Xuanwu Hospital with the company 数坤 Shukun) is closed, and it has no VA class.** The paper's data statement reads: "the source imaging data from five tertiary hospitals cannot be made publicly accessible due to privacy protection". Its Reporting Summary adds that CerebralDoc "is a commercial system and can not give people access to the source code".

The scans run from the aortic arch to the vertex, and the VA lumen is drawn from its origin. The paper discusses a "Left vertebral artery arising directly from the aorta". But the Peer Review File shows that the four ground-truth classes are body regions: "bone, aorta, carotid artery and intracranial artery". The cervical VA sits in the "carotid artery" neck class together with the subclavian, CCA and ICA.

What can be requested from Fan Fu (傅凡) is "software access and the remaining clinical evaluation images generated by technologists", which are rendered reconstructions. The view-only demo server is NXDOMAIN. **Verified (verify_0, verify_5).** This corrects the English report, which could not read the statement and listed Fu et al.'s VA labelling as "unclear" ([PMC7518426](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7518426/)).

**"HeadNeckCTA2025——头颈CTA血管分割" is a Zhihu tutorial by a blogger, not a challenge or a release.** Its dataset paragraph copies Fu et al.'s five-hospital counts, and its 「数据下载地址」 ("data download address") is PubMed 32973154. The blogger's own recipe uses 「二值化的Dice和CE」 ("binary Dice and CE"), so even his labels have no VA class.

One search-engine summary credits 「双侧椎动脉」 ("bilateral VAs") to this post. That text actually comes from the 2024 consensus, so it should not be cited as evidence for this post. **Verified (verify_5).**

**The other corpora say so themselves:**

| Corpus | Size | What its source says |
|---|---|---|
| HNCTA (Yizhun Medical AI + PKU) | 358 CTAs | "a private head and neck artery dataset"; left and right VA classes, extent unstated ([arXiv 2506.05820](https://arxiv.org/html/2506.05820)) |
| HNA (HKU, Deepwise 深睿, ZJU 2nd Affiliated Hospital) | 800 | "an in-house dataset"; the VA is inside one binary vessel mask ([arXiv 2301.02393](https://arxiv.org/pdf/2301.02393)) |
| Peking University Third Hospital (北医三院) set | 115 | "Not publicly available" ([BiSegMamba](https://github.com/bakhtzadaabshare/BiSegMamba); paper withdrawn on arXiv) |
| 3MA | 100 | "in-house" |
| TaG-Net full set | 401 | Not offered |
| LIVE-Net (东北大学 Northeastern University + 盛京医院 Shengjing Hospital) | — | "dataset from the clinic"; bilateral carotid and VA labels; its data statement could not be read (ScienceDirect 403) |

**The only written offer of VA-labelled Chinese data is MR, not CTA.** Zhang et al., npj Digit Med 2025 (仁济医院 Ren Ji Hospital SJTU with 联影 United Imaging), annotated 1,728 3D MR vessel-wall sequences with a VA class. Their data "are available from the corresponding author [Huilin Zhao] on reasonable request". The VA's neck extent is not stated ([PMC12304216](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304216/)).

**The 2024 consensus explains where the labels live.** Its title is 「头颈动脉CT血管成像数据标注与质量控制专家共识（2024）」 ("expert consensus on head-and-neck CTA data annotation and quality control"), 中华放射学杂志 2024;58(2):150-157, co-issued with the national device-testing institute NIFDC (中国食品药品检定研究院). It requires labels for:

> 「主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、双侧椎动脉…」
> ("aortic arch, brachiocephalic trunk, both subclavians, both CCAs, both ICAs, basilar, both VAs…")

on scans 「从颅顶至胸骨角水平」 ("from the vertex to the sternal angle"). Their stated uses are 「AI产品的训练集、调优集、性能独立测试集」 ("training, tuning and independent test sets for AI products").

The consensus announces a matching 「头颈动脉CTA影像标准数据库」 ("standard head-and-neck CTA database"), but no release or access route was found. The text never says "from the origin" explicitly, and its figures show coarse colour classes (intracranial / neck / aorta / bone). **Verified (verify_3, verify_6): a standard, not a dataset.** Chinese corpora built to this standard would contain exactly the origin-to-basilar lumens we need. They sit with hospitals, the vendors Shukun, United Imaging and Deepwise, and the regulator.

## All candidates

Status key:

| Status | Meaning |
|---|---|
| **Verified** | Checked independently against the host (verify note named) |
| **Re-checked** | Re-downloaded and re-measured while writing this report |
| **Source-read** | A searcher read the paper or repository text; not independently verified |
| **Unverified** | Search snippet or title only |

| # | Candidate (中文 / English) | Host | Modality | Cases | Cervical VA coverage | Access terms | Status | Link |
|---|---|---|---|---|---|---|---|---|
| 1 | TaG-Net 样例数据 / TaG-Net sample data (Yao et al., TMI 2023; SJTU, United Imaging Intelligence) | Google Drive + GitHub | CTA, 0.37–0.41 mm in-plane, 0.4 mm slices | 2 labelled patients, 1 with its CTA | **L+R VA from the subclavian to the basilar, ~146 mm**; 18 classes incl. SCA, arch, BCT, BA; 1-voxel centerlines, no lumen | **Open, no login**; code GPL-3.0; no data licence | **Re-checked: open** | [Drive](https://drive.google.com/drive/folders/1Q1GoRfvVZsSgxbia60jANJscOkKXetyx), [GitHub](https://github.com/PRESENT-Y/TaG-Net) |
| 2 | CereVessMRA (Guo et al., Nat Commun 2024) | ScienceDB 科学数据银行 | TOF-MRA | 271 | None: binary intracranial mask; slabs ≤ 89.6 mm end at the skull base | Open, CC BY-NC 4.0 | **Verified (verify_4): open, no cervical VA** | [ScienceDB](https://www.scidb.cn/detail?dataSetId=d3a67e63c0054652b75bfcc3cf15ae3f) |
| 3 | 头颈部MR管壁成像血管分割重建 / Zhang et al., npj Digit Med 2025 (仁济医院 + 联影) | Authors | 3D MR vessel-wall imaging | 1,728 annotated sequences, 4 centres | VA class; neck extent not stated | "available from the corresponding author [Huilin Zhao] on reasonable request" | Source-read: on request | [PMC12304216](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304216/) |
| 4 | 宣武医院/数坤 CerebralDoc / Fu et al., Nat Commun 2020 | Xuanwu + Shukun | CTA, arch to vertex, 0.4–0.6 mm | 18,766 scans (18,259 patients), 5 hospitals | VA drawn from origin but merged into region classes; **no VA class** | Private; software and rendered images only, on request to Fan Fu; demo NXDOMAIN | **Verified (verify_0, verify_5): not public** | [PMC7518426](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7518426/) |
| 5 | HeadNeckCTA2025——头颈CTA血管分割 (Zhihu post) | Zhihu / WeChat | — | Describes #4 | None offered; blogger's model is binary | Login wall; "download" = PubMed page | **Verified (verify_5): not a dataset** | [Zhihu](https://zhuanlan.zhihu.com/p/29607061020) |
| 6 | HNCTA (DeformCL; Yizhun Medical AI + PKU) | — | CTA | 358 | L/R VA among 9 classes; extent not stated | "private"; code public, data not | Source-read: private | [arXiv](https://arxiv.org/html/2506.05820), [code](https://github.com/barry664/DeformCL) |
| 7 | TaG-Net full set | — | CTA | 401 | 18 segments; the sample shows VA from subclavian to basilar | Not offered | Source-read: private | [doi](https://doi.org/10.1109/TMI.2023.3240825) |
| 8 | HNA (HKU, Deepwise 深睿, ZJU 2nd Affiliated) | — | CTA | 800 | VA inside one binary vessel mask | "in-house"; no statement | Source-read: private | [arXiv](https://arxiv.org/pdf/2301.02393) |
| 9 | 北医三院颈动脉CTA / PKU Third Hospital set (BiSegMamba, withdrawn) | — | CTA | 115 | Arch to ICA C6; VA not named; classes vessel/plaque | "Not publicly available" | Source-read: private | [arXiv v1](https://arxiv.org/pdf/2605.30972v1) |
| 10 | 3MA (PC-Net; NWPU, PKU, Deepwise) | — | CTA | 100 | "major arteries in the head and neck"; classes unstated | "in-house" | Source-read: private | [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.04428) |
| 11 | LIVE-Net (东北大学 + 盛京医院) | — | CTA | Not given | Bilateral carotid and VA; extent unknown | Statement unread (403) | Source-read: unknown, likely private | [doi](https://doi.org/10.1016/j.compbiomed.2023.106886) |
| 12 | 心、脑、外周血管疾病医学影像数据集 (天坛医院 Tiantan) | NBSDC | Raw DICOM: cardiac CTA, brain TOF-MRA 0.8 mm, peripheral CTA | 40 files, 207 GB; count unstated | **None: no labels** | Chinese real name, mainland mobile, signed form with 公章 (official seal), NBSDC + Tiantan approval | **Verified (verify_1): not public** | [NBSDC](https://www.nbsdc.cn/general/dataLinks/16666.11.nbsdc.UpfJ6qMG) |
| 13 | 医学图像标准测试数据库-脑卒中影像与标注数据 (NCMI) | NCMI / PHDA | Head-and-neck CTA (+ head CT) | 433 aneurysm CTAs | None: aneurysm ROIs only; field of view unverified | 有条件共享 / 线下共享 (conditional, offline); real-name account | **Verified (verify_2): not public** | [NCMI](https://www.ncmi.cn/phda/dataDetails.html?type=project_data&id=CSTR:17970.12.A00AL.202209.372.V1.0) |
| 14 | 头颈动脉CT血管成像数据标注与质量控制专家共识（2024） | 中华放射学杂志 58(2):150-157 | Annotation standard | 0 | Specifies both VAs + subclavians, vertex to sternal angle | Text free; announced "standard database" not found | **Verified (verify_3, verify_6): not a dataset** | [brainmed](http://www.brainmed.com/info/detail?id=47109) |
| 15 | 基于深度学习的CTA图像颈动脉和椎动脉血管分割 (华中科技大学 HUST thesis, 2020) | CNKI 知网 | CTA | Unknown | Carotid + VA; extent unknown | No statement seen | Unverified (CNKI blocked) | [CNKI](https://cdmd.cnki.com.cn/Article/CDMD-10487-1020353767.htm) |
| 16 | 深度学习在人脑CTA动脉血管分割中的应用 (HUST thesis, 2021) | CNKI | Head-and-neck CTA | 77 | Intracranial only | 「计划将其公开」 ("plans to make it public"); no release found | Unverified | [CNKI](https://cdmd.cnki.com.cn/Article/CDMD-10487-1021907175.htm) |
| 17 | 利用深度學習產生去骨電腦斷層掃描血管造影改進椎動脈分割 (NCU Taiwan thesis, 2020) | NCU ETD | CTA | Unknown | VA segmentation; extent unknown | Host DNS failure | Unverified (title only) | [NCU](https://etd.lib.ncu.edu.tw/detail/63d6e9f8791a744991983468328bb27f) |
| 18 | CAS2023 脑动脉分割挑战赛 (清华 Tsinghua + UW) | CodaLab 9804 | TOF-MRA | 150 (100 train / 50 test) | None: intracranial, one binary class | CodaLab login; data CC BY-NC, use limited to the challenge; CodaLab phased out end of 2025 | **Verified (verify_7): excluded** | [Zenodo](https://zenodo.org/records/7839970) |
| 19 | Duke VA-dissection CTA (medRxiv 2026; not Chinese, incidental) | Authors | CTA | 101 | "all discernible arterial branches"; V1–V4 reader study; per-class labels unverified | Data and weights "upon reasonable request" | Source-read: on request | [medRxiv](https://www.medrxiv.org/content/10.64898/2026.09.20.26363486v1.full) |

**Also checked and excluded** (no cervical VA; source-read):

| Item | Host | Reason |
|---|---|---|
| INSTED 2024 (Tsinghua) | Codabench 2139 | Intracranial TOF-MRA aneurysm and stenosis labels |
| SegRap2023 / SegRap2025 (电子科技大学 UESTC) | Grand Challenge | 45 radiotherapy organs-at-risk (OARs), or tumour/node targets; no vessel class |
| CSV 2026 | ISBI 2026 | Carotid plaque, ultrasound |
| Tianchi 天池, all 577 competitions | tianchi.aliyun.com | No title with 血管, CTA, 颈, 脑血管 or 动脉 |
| Parse2022, KiPA22, ATM22, GAVE/GAVE2, CARE 2026 | Various | Pulmonary, renal, airway, retinal, cardiac/liver |
| TopBrain 2025/2026 (Chinese co-organisers) | Zenodo | Braincase crop (see the English report) |
| 头颈CT血管造影后处理中医用图像配准的应用 ("image registration in head-and-neck CTA post-processing") | ScienceDB | One 4.2 MB statistics file, no images |
| Aneumo, COSTA mirror, royal_brisbane masks | ModelScope | Intracranial aneurysm or TOF-MRA |
| IntrA, SegRap | OpenDataLab | Intracranial aneurysm models; OARs |
| 解剖学（头颈部分）的CT分割数据 ("head-and-neck anatomy CT segmentation data") | Heywhale | Re-upload of PDDCA, no vessels |
| Cerebral-MRA annotations (Tsinghua IVG) | ivg.au.tsinghua.edu.cn | 109 TubeTK MRAs, Circle of Willis only |
| OMIX002680 「MRA or CTA data」 | NGDC/CNCB | A 16 KB spreadsheet of grades |
| 颈动脉及椎动脉疾病超声诊断数据集 ("carotid and VA disease ultrasound diagnosis dataset") | NCMI | Ultrasound teaching cases |
| CTA reports, stenosis clinical tables, HR-MRI and 7T sets | NCMI | Text, tables or intracranial MR |
| Tiantan intracranial aneurysm database (Front Neurol 2022) | Paper | Aneurysm-focused |
| 蚌埠 Bengbu vertebrobasilar 3D study (199 CTAs) | Paper | Intracranial focus; models not released |
| 高精度CT颅内骨骼、动静脉数据集 ("high-precision CT skull and intracranial artery/vein dataset") | CSDN vendor post | Intracranial; an advert with no terms |
| Chinese awesome-lists and round-ups (OpenMedLab, linhandev, M3D-Seg, SAT-DS, CSDN 150-dataset list, 中国图象图形学报 2024 review) | GitHub / CSDN / journal | Only already-known entries (CAS2023, COSMOS, CLS2009, SEG.A., TopBrain) |

## Corrections to the English report and its notes

- **Fu et al.**
  - The data statement is now read: the data are private.
  - The English report's "Fu et al. unclear" (table row 12) should read: **VA drawn from origin, but no VA class**. It is merged into a "carotid artery" neck-region mask.
  - `platforms.md` §1.1 implied VA-specific labels; that was wrong (verify_0).
- **Ready-made real cervical VAs now come from three people, not two:** VMR 0248, SPL and TaG-Net case 001/002. A fourth VA pair, TaG-Net 003, exists as a centerline only, with no image.
- **CAS2023 has 150 cases (100/50), not 100 (85/15).** Its data terms are CC BY-NC with use restricted to the challenge. The CC BY-ND 4.0 on Zenodo covers only the proposal PDF (verify_7).
- **The King's College London navigation work** now has "vasculature trees from 61 patients", still code-only (IJCARS 2026, [PMC13553664](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13553664/fullTextXML)).
- **TaG-Net headers do store orientation** (qform code 1, identical in the CTA and the case-001 labels), contrary to `papers.md`.

## What was searched, and what could not be opened

**Searched successfully:**

| Route | Coverage |
|---|---|
| Native platform APIs, Chinese terms sent as UTF-8 files or percent-encoded | OpenDataLab (~50 terms), ModelScope (~40), ScienceDB (25 terms × 3 pages, plus DataCite prefix 10.57760), the escience.org.cn national portal (26 terms; it indexes NCMI and NBSDC), NBSDC detail, file-tree and download endpoints, Heywhale (21), the Gitee widget search (26), OpenI (15), HyperAI (18), PKU Dataverse (13) |
| Challenge catalogues | All 577 Tianchi competitions (API), all 264 Grand Challenge challenges (API), Codabench keyword search, MICCAI 2023–2026 and ISBI 2024–2026 lists, CSIG (中国图象图形学学会) challenge site |
| Papers | Europe PMC full text (Fu et al., Zhang et al. and five others), arXiv, Semantic Scholar, GitHub, Fu et al.'s four supplementary files including the Peer Review File; CQVIP (维普) for Chinese journals (3 queries) |
| Community | Sogou WeChat search, then mp.weixin.qq.com full text (about 25 queries, 5 articles read); CSDN posts and its search API (after decoding its JavaScript cookie challenge); Bilibili, Juejin, Hugging Face and Heywhale APIs |
| General web | 360 Search (~20 queries before it blocked), Baidu (4 queries before a CAPTCHA), Sogou (1 query), and a US-based WebSearch with Chinese queries and site filters |

One pitfall cost a first pass: **Git Bash curl on this Windows host mangles Chinese characters passed on the command line**, which gave false zero-hit results. Every Chinese query was re-sent from a UTF-8 file or percent-encoded.

**Could not open:**

| Barrier | Where |
|---|---|
| Login required | Baidu AI Studio (飞桨) dataset search (`errorCode 8403 "未登录"`, "not logged in"); Tianchi dataset search ("api access denied!"); Zhihu (「请您登录后查看更多专业优质内容」, "please log in to see more"; API codes 10003 and 40362; no Wayback or archive.ph copy); the CodaLab Participate tab for CAS2023 |
| Identity, phone or seal requirements | NBSDC: a real name in Chinese characters, a mainland mobile number and a signed application stamped with an official seal (公章); NCMI/PHDA: real-name registration and an offline application |
| CAPTCHA or anti-bot | Baidu and Baidu Xueshu (百度学术) (「百度安全验证」, "Baidu security verification"); Sogou web and, on a later pass, Sogou WeChat (`/antispider/`); 360 Search (redirect loop after ~20 queries); Bing China (no usable results); HyperAI general search (Cloudflare); NCMI landing pages (JavaScript cookie challenge, not bypassed) |
| Blocked or unreachable | CNKI 知网 (HTTP 418, TLS mismatch, connection failures), so the HUST theses were never read; Wanfang 万方 (JavaScript shell); the Gitee v5 search API (`[]` for everything); BAAI DataHub (405); Huawei AI Gallery (403/404); InstDB and the Fudan and Wuhan University Dataverses (DNS); the NGDC OMIX search API; the NCU Taiwan thesis hosts (DNS); the Shukun demo server (NXDOMAIN); the NIFDC standards page (TLS mismatch, read with `-k`) |
| Paywalls | ScienceDirect (LIVE-Net data statement), IEEE Xplore (TaG-Net full text), RSNA (Radiology 2023), The Lancet (Lancet Digit Health 2024), the yiigle publisher copy of the consensus |
| Not searched | Douyin, Xiaohongshu, Baidu Tieba, Wenku and Pan (网盘), QQ and WeChat groups |

**Residual blind spot.** A cloud-drive (网盘) link posted only in a Zhihu comment, on Baidu Tieba or in a QQ group would have been missed. Data passed around that way would not be lawfully redistributable for a simulator anyway.

## Next steps

**1. Add TaG-Net to stage one and to the Arterial benchmark now.**
- Download `SampleData/002/CTA.nii.gz` (192 MB) and `SampleData/001/labeled_cl_gt.nii.gz`. Keep `SampleData/003/labeled_cl_gt.nii.gz` as a path-only VA pair.
- Segment the VA lumen on the 0.4 mm CTA, seeded from labels 8 (R-VA) and 13 (L-VA), and cut the graft at the subclavian labels 14 and 15.
- Score Arterial's RVA/LVA output against these centerlines exactly as against VMR 0248's paths. This gives a second sub-millimetre benchmark case, from a second patient and a Chinese hospital population.
- Write to Linlin Yao about data terms before publishing any derived mesh.

**2. Send these data requests, in order of expected value:**

| Priority | Who | Paper | Contact | Ask |
|---|---|---|---|---|
| 1 | Linlin Yao (TaG-Net group, SJTU / United Imaging Intelligence) | Yao et al., "TaG-Net", IEEE TMI 2023;42(11):3155–3166 | yaolinlin23@sjtu.edu.cn | Data licence for the sample; whether the ground truth is manual; whether more of the 401 labelled cases (already SCA → VA → BA per side) can be shared |
| 2 | DeformCL / HNCTA authors (Yizhun Medical AI + PKU) | Zhao et al., arXiv 2506.05820 | GitHub issue on [barry664/DeformCL](https://github.com/barry664/DeformCL); emails not captured | VA extent in HNCTA, and whether its masks can be released. They released the ImageCAS masks once "restrictions on releasing the new annotations" lifted |
| 3 | Huilin Zhao (Ren Ji Hospital SJTU) | Zhang et al., npj Digit Med 2025, doi 10.1038/s41746-025-01866-x | Corresponding author; email not captured | Whether the MR vessel-wall field of view reaches the VA origin; a sample of VA masks under the paper's "reasonable request" offer |
| 4 | Evan Calabrese (Duke; not Chinese) | Zaveri et al., medRxiv 2026, doi 10.64898/2026.09.20.26363486 | Via medRxiv | Whether the 101 CTA masks separate the VA by class and side; data "upon reasonable request" |
| 5 | Jinzhu Yang (Northeastern University) | Sun et al., LIVE-Net, Comput Biol Med 2023;157:106886 | yangjinzhu@cse.neu.edu.cn | Whether the bilateral carotid and VA labels can be shared, and their extent |
| 6 | Jie Lu 卢洁 / Fan Fu 傅凡 (Xuanwu) | Fu et al., Nat Commun 2020;11:4829 | imaginglu@hotmail.com | Collaboration only. The labels would need splitting into a VA class. Inference: China's personal-information export rules make a scan transfer unlikely without a formal agreement |

Not worth a request:
- Tiantan/NBSDC: no labels, and the application needs a Chinese identity and seal.
- NCMI: aneurysm ROIs only.
- HNA: one binary mask (Yizhou Yu, yizhouy@acm.org, could still be asked).
- CAS2023: intracranial only.
- The HUST theses: authors unidentified, and the 2021 set is intracranial.

**3. Optional checks for someone with Chinese accounts, about ten minutes, low expected yield:**
- Run 椎动脉 and 头颈CTA in AI Studio's and Tianchi's dataset search.
- Read the Zhihu HeadNeckCTA2025 post and its comments.
- Open the two HUST theses on CNKI for case counts and any release URL.

The conclusion of the English report stands, with one addition. Graft VMR 0248, SPL and now TaG-Net, then scale with Arterial on the TotalSegmentator CTAs, benchmarked against both VMR 0248 and TaG-Net.
