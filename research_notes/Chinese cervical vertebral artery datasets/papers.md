# Chinese papers on head-and-neck CTA vessel segmentation: what their data-availability statements actually say

Scope: papers by Chinese groups, in English and in Chinese journals, that segment or label the vertebral artery (椎动脉, VA) on head-and-neck angiography. For each paper, the question is whether any data can be obtained and on what terms. Searched 2026-10-01. Sibling notes in this folder cover the overlapping items in more depth: `challenges.md` (HeadNeckCTA2025, the 2024 annotation consensus, NIFDC) and `community.md` (the Zhihu blogger behind HeadNeckCTA2025). Here those items are kept to the exact paper statements.

How the claims were checked:
- Full texts came from the Europe PMC REST API (`/PMCxxxx/fullTextXML`), arXiv HTML/PDF and the arXiv API, Semantic Scholar and the GitHub API.
- Google Drive public folders were listed through `embeddedfolderview` and read with HTTP range requests.
- PDFs were converted to text with pypdf in a throwaway Docker container.
- Chinese-language searches went through CQVIP (维普), Sogou, Baidu (until it showed a CAPTCHA) and WebSearch with Chinese queries.
- What could not be opened is listed in the access log at the end.

## Headline

1. **One Chinese release is downloadable today, with no login: the TaG-Net sample data** (Yao et al., IEEE TMI 2023, head-and-neck vessel labelling). Its public Google Drive folder holds:
   - one real head-and-neck CTA, 512×512×796 at 0.405×0.405×0.4 mm (about 318 mm, aortic arch to vertex);
   - **expert-format 18-class labelled centerline volumes for two patients.**

   The label topology encodes `L-VA → L-SCA & BA` and `R-VA → R-SCA & BA`. Measured on the files, each VA label starts 0.4–0.7 mm from the subclavian label and ends 0.5–1.0 mm from the basilar label, over a ~146 mm vertical span. So these are **cervical VAs from origin to confluence**. It is two patients, not a dataset, but it is a third and fourth real cervical VA beside VMR 0248 and the SPL atlas, from a Chinese hospital population. No data licence is stated.
2. **Every multi-patient Chinese corpus with VA labels is private.** This was confirmed from the papers' own words for HNCTA (358), HNA (800), Fu et al. (18,766), the Peking University Third Hospital set (115) and the TaG-Net full set (401). LIVE-Net's statement could not be read, and its abstract says only "dataset from the clinic".
3. **Only one Chinese paper with VA labels offers its imaging data on request:** the npj Digital Medicine 2025 MR vessel-wall paper (Ren Ji Hospital SJTU and United Imaging, 1,728 annotated sequences). Its statement reads "available from the corresponding author [Huilin Zhao] on reasonable request". Fu et al. offers only its software and "the remaining clinical evaluation images", and the demo server it names no longer resolves.
4. **The Chinese-language journal literature has no data statements at all.** Head-and-neck CTA "AI" papers in 中国医学计算机成像杂志 and similar journals are clinical evaluations of commercial products: 数坤 Shukun, 深睿 Deepwise Dr.Wise, United Imaging, and GE/Canon deep-learning reconstruction. The labelled data behind them is held by companies.

---

## Q1. TaG-Net sample data: the only downloadable Chinese cervical-VA labels found

### Takeaway
The authors of "TaG-Net: Topology-Aware Graph Network for Centerline-Based Vessel Labeling" (IEEE TMI 2023;42(11):3155–3166) host sample data on Google Drive, publicly readable with no login. The labels are centerline point sets and volumes in 18 head-and-neck classes. They include both subclavians, both VAs and the basilar, and they connect VA to subclavian at the bottom and to basilar at the top. One of the two labelled patients comes with its full 0.4 mm CTA.

### Cited Findings
- **README:**
  - "We have provided sample data for testing."
  - "Sample data, the corresponding ground truth, and our result can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1Q1GoRfvVZsSgxbia60jANJscOkKXetyx?usp=sharing)."
  - "The code is released under GPL License." The GitHub licence is GPL-3.0, and no separate data licence is given.
  - Contact: yaolinlin23@sjtu.edu.cn.

  Source: [PRESENT-Y/TaG-Net README](https://github.com/PRESENT-Y/TaG-Net)
- **Paper abstract:** "Finally, the head and neck vessels of 18 segments are labeled by assigning centerline labels to the refined segmentation. We have conducted experiments on CTA images of 401 subjects". Authors: Yao L, Shi F, Wang S, Zhang X, Xue Z, Cao X, Zhan Y, Chen L, Chen Y, Song B, Wang Q, Shen D. Grants: NSFC 62131015, 81830056; National Key R&D 2018YFC0116400. — [Europe PMC record, PMID 37022246](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:37022246%20AND%20SRC:MED&resultType=core&format=json)
- **Label topology, quoted from the repository code** (`VesselCompletion/utils_multicl.py`):
  - "L-VA(12) -> L-SCA(14) & BA(8)"
  - "R-VA(7) -> R-SCA(13) & BA(8)"
  - The edge list includes `(7,13)`, `(7,8)`, `(12,14)`, `(8,12)` and `(1,12)`, the last being the "special" case of a left VA arising from the arch.
  - Full class list (txt labels 0–17): R-PCA 0, (arch) 1, (brachiocephalic trunk) 2, R-CCA 3, R-ICA 4, ACA 5, R-MCA 6, **R-VA 7**, **BA 8**, L-CCA 9, L-ICA 10, L-MCA 11, **L-VA 12**, **R-SCA 13**, **L-SCA 14**, L-ECA 15, R-ECA 16, L-PCA 17. `cfgs/config_train.yaml` sets `num_classes: 18`.

  Source: [utils_multicl.py](https://github.com/PRESENT-Y/TaG-Net/blob/main/VesselCompletion/utils_multicl.py)
- **Naming caveat, from the code itself:**
  - The comments call label 1 "BCT" and label 2 "AO". The edge list, however, joins label 1 to L-CCA, L-SCA and L-VA, and label 2 to R-CCA and R-SCA.
  - The geometry agrees: in the NIfTI ground truth, label 1+1 is a wide antero-posterior structure at the lowest z, and label 2+1 is a short right-sided vertical vessel.
  - So **label 1 is the aortic arch and label 2 is the brachiocephalic trunk**, and the two comment names are swapped.
- **Drive folder contents** (listed via `embeddedfolderview`, all modified 10/31/22 or 11/4/22):

  | Folder | Files | Notes |
  |---|---|---|
  | `SampleData/001` | `CenterlineGraph`, `cl.nii.gz`, `cl.txt`, `labeled_cl_gt.nii.gz` (231,849 B), `seg.nii.gz` (316,079 B) | |
  | `SampleData/002` | `CTA.nii.gz` (**192,162,690 B**), `labeled_cl.txt`, `labeled_cl_new.txt`, `CenterlineGraph*`, `connection_*` | |
  | `SampleData/003` | `labeled_cl_gt.nii.gz` (218,023 B), `labeled_cl*.txt`, `CenterlineGraph*` | No CTA |
  | `SegNet` | `README.md` only | Its text says "We have put our trained models at Google Drive", but no weights are in the folder |

  Source: [Drive root](https://drive.google.com/drive/folders/1Q1GoRfvVZsSgxbia60jANJscOkKXetyx)
- **Download behaviour:** `https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t` serves the files with no login and honours HTTP Range. Example: `CTA.nii.gz` id `10owNj4vZzokuo8VRRMk7-BUExpob-IJi` returned `206` with `content-range: bytes 0-65535/192162690`.
- **NIfTI headers, measured:**

  | File | Grid | Spacing (mm) | Type / extent | Labels |
  |---|---|---|---|---|
  | `002/CTA.nii.gz` | 512×512×796 | 0.405×0.405×0.4 | int16, 318 mm in z | — |
  | `001/labeled_cl_gt.nii.gz` | 512×512×796 | 0.405×0.405×0.4 | uint8 | values 1–18 |
  | `003/labeled_cl_gt.nii.gz` | 512×512×743 | 0.366×0.366×0.4 | uint8 | values 1–18 |
  | `001/seg.nii.gz` | 512×512×689 | 0.488×0.488×0.625 | binary vessel mask | — |

  All four files have sform/qform 0, i.e. no orientation stored. The NIfTI label value equals the txt label + 1.
- **Measured VA geometry in the ground-truth volumes** (1-voxel-wide centerlines, z up):

  | Case | Vessel | Voxels | z range (mm) | Lowest VA voxel to SCA label | Highest VA voxel to BA label |
  |---|---|---|---|---|---|
  | gt_001 | R-VA | 506 | 70–217 (146 vertical) | 0.4 mm (R-SCA) | 0.7 mm |
  | gt_001 | L-VA | 580 | 68–216 (149) | 0.7 mm (L-SCA) | 1.0 mm |
  | gt_003 | R-VA | 499 | 57–203 (146) | 0.5 mm | 0.5 mm |
  | gt_003 | L-VA | 507 | 56–201 (145) | 0.7 mm | 0.5 mm |

  - In gt_001 the R-SCA label spans z 47–74 mm, the L-SCA 25–75 mm and the BA 217–236 mm. The arch label sits at z 6–24 mm.
  - So the CTA field of view starts at the top of the arch.
- **Which ground truth belongs to the 002 CTA:**
  - `001/labeled_cl_gt.nii.gz` is on exactly the same 512×512×796 / 0.405 / 0.4 grid as `002/CTA.nii.gz`.
  - 48.7% of the points in `002/labeled_cl.txt` (columns z, y, x, label) fall on a gt_001 voxel, and where they do the label agrees 99.0% of the time.
  - `001/cl.txt` instead spans z up to 661, which fits the 689-slice `001/seg.nii.gz`.
  - The repository's `TaG-Net-Test/data/001`–`005/cl.txt` are five identical copies of one labelled case (identical per-label counts, 7,981 points each).

### Inferences
- Pairing `002/CTA.nii.gz` with `001/labeled_cl_gt.nii.gz` gives **one complete patient**: a sub-millimetre head-and-neck CTA plus labelled R-VA and L-VA centerlines from the subclavian to the basilar. This is the same kind of asset as VMR 0248, but from a Chinese cohort and in NIfTI. A second patient (gt_003) has labelled centerlines but no image.
- The ground truth is **centerline only, with no lumen mask or radius.** To graft geometry we would segment the lumen on the 0.4 mm CTA around the labelled VA centerline, for example with VMTK level sets seeded from the labels.
  - The same pair is also a second benchmark case for the Arterial-on-CTA check. Arterial's VA output can be scored against these labels exactly as against VMR 0248's paths.
- Whether `labeled_cl_gt` is fully manual is not stated in what could be read. The abstract says centerlines were extracted from a segmentation and then labelled.
- "Sample data … for testing", together with a GPL licence on code only, leaves reuse and redistribution of the data unclear. Writing to the contact address before publishing derived meshes is advisable.

### Gaps
- The full TMI text (dataset paragraph, hospital, annotation protocol) could not be read; the IEEE Xplore page returned empty content.
- One WebSearch summary described the TaG-Net dataset as "450 CTA scans … 380 … 20 … 50 … 17 categories". It probably comes from the authors' MICCAI 2020 GCN paper, but this is unverified. The TMI abstract says 401 subjects and 18 segments.
- The 192 MB CTA was read only to its header, not downloaded in full.

---

## Q2. The named Chinese corpora: exact availability statements

### Takeaway
None of these corpora is downloadable. Below are the statements as written, or the absence of one.

### Cited Findings

**Fu et al., Nat Commun 2020: "Rapid vessel segmentation and reconstruction of head and neck angiograms using 3D convolutional neural network" (中文译名: 使用3D卷积神经网络快速分割和重建头颈部血管造影)**
- Data availability, verbatim from the Europe PMC XML:

  > "Currently, the source imaging data from five tertiary hospitals cannot be made publicly accessible due to privacy protection. The CerebralDoc is accessible from the webserver created for this study at http://test.platform.shukun.net/login. User name: test, password: 123456. Request for software access and the remaining clinical evaluation images generated by technologists can be submitted to F.F. The source data underlying Supplementary Table 2, Figs. 3a, 7, and 8 are provided as a Source Data file with this paper."

  Source: [PMC7518426 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML)
  - The "supporting data" flag in Europe PMC refers to that figure Source Data file, not to images.
  - F.F. is Fan Fu. The corresponding author is Jie Lu, imaginglu@hotmail.com, as given in the XML.
- **The demo server is gone.** `http://test.platform.shukun.net/login` did not resolve: curl returned status 000 on 2026-10-01. No login was attempted.
- **The annotation has no separate VA class.**
  - English, from the XML: "prelabeled using ITKSNAP software with masks marking the areas of the bone, aorta, carotid artery, and intracranial artery. They were multiclass classification with four classes. We annotated the arteries including the aorta, subclavian artery, common carotid artery, vertebral artery, internal carotid artery, basilar artery, posterior cerebral artery, middle cerebral artery, and anterior cerebral artery."
  - The Chinese translation, extracted from the PDF: 「使用 ITKSNAP 软件进行预标记，并带有标记骨骼、主动脉、颈动脉和颅内动脉区域的掩码。它们是 4 类的多类分类。我们标注的动脉包括主动脉、锁骨下动脉、颈总动脉、椎动脉、颈内动脉、基底动脉、大脑后动脉、大脑中动脉和大脑前动脉。」 Translation: "prelabelled in ITK-SNAP with masks marking bone, aorta, carotid artery and intracranial artery regions; a four-class problem. The arteries we annotated include the aorta, subclavian, common carotid, vertebral, internal carotid, basilar, PCA, MCA and ACA."
  - Source: [Chinese translation PDF, cn-healthcare](http://docs.cn-healthcare.com/sharedoc/src_files/20210915/2748c31e711e43f4a574ed253cfe2b93.pdf). This translation does not include the data-availability section.
  - The same translation's failure-case caption reads 「(c) 左椎动脉直接起源于主动脉；CPR 和弯曲 MPR 失败。」 ("left VA arising directly from the aorta; CPR and curved MPR failed"). Origin variants were in the cohort, but they are not a label.
- **Chinese press wording**, Baidu snippet of 动脉网 (VCBeat): 「18259例头颈CTA数据集均采集于2017年6月至2018年11月间由5家国内顶尖三甲医院协作提供…整个实验过程已对超过一亿血管区域进行了AI勾画与分割」. Translation: "the 18,259-case head-and-neck CTA dataset was provided by five top tertiary hospitals between June 2017 and November 2018 … over 100 million vessel regions were AI-delineated and segmented". This describes the data; it does not release it.
- **HeadNeckCTA2025 (知乎 Zhihu p/29607061020)** is a catalogue post about this cohort; see `challenges.md` Q2 and `community.md` Q1.
  - The search-engine extract gives the "data download link" (数据下载链接) as `https://pubmed.ncbi.nlm.nih.gov/32973154/`, which is the PubMed record of Fu et al., not data.
  - One WebSearch summary in Chinese called this cohort 「这是该领域一项重要的公开资源」 ("an important public resource in this field"). **That is wrong:** the paper itself says the data "cannot be made publicly accessible".

**HNCTA, in DeformCL (arXiv 2506.05820): Zhao, Zhang, Liu, …, Wang (Yizhun Medical AI and Peking University)**
- Main text: "HNCTA is a private head and neck artery dataset with 358 CTA images."
- Appendix A: "clinical experts provide accurate segmentation labels for nine categories of vessels, including the left and right vertebral arteries (VA), internal carotid arteries (ICA), common carotid arteries (CCA), posterior communicating arteries (PoCA), and basilar artery (BA)."
- Split 242/58/58.
- Code: "We release the code in https://github.com/barry664/DeformCL". Its README offers only ImageCAS masks and HaN-Seg preprocessing; there is nothing for HNCTA.
- No VA extent (origin or V-segments), hospital or spacing is given.
- Sources: [arXiv HTML](https://arxiv.org/html/2506.05820); [DeformCL repo](https://github.com/barry664/DeformCL)
- Affiliation as read from the arXiv page: Yizhun Medical AI Co., Ltd; Peking University Center for Data Science and Center for Machine Learning Research; Pazhou Laboratory (Huangpu).

**HNA, in "Graph Convolution Based Cross-Network Multi-Scale Feature Fusion for Deep Vessel Segmentation" (arXiv 2301.02393; Zhao, Liang, Pan, Zhang, Wu, Hu, Yu)**
- Verbatim from the PDF text: "HNA Head and Neck Artery Dataset (HNA) is an in-house dataset which contains 800 CT angiography (CTA) images of head and neck. The images are annotated in the same way as ACA. Cerebral, vertebral and carotid arteries are annotated as the target vessel mask. The dataset is split into a training set of 640 images, a validation set of 80 images and a test set of 80 images. The original image resolution of the HNA dataset is also 512×512×N, where N is between 192 and 600."
- "Annotated in the same way as ACA" means "annotated by one expert annotator and verified by a second expert". The result is **one binary vessel mask, with no per-artery classes.**
- Affiliations: HKU (Yizhou Yu, corresponding), BUPT, Beihang, Deepwise AI Lab (深睿), and the Second Affiliated Hospital of Zhejiang University School of Medicine. Corresponding authors are Yizhou Yu (yizhouy@acm.org) and Xinyang Hu (hxy0507@zju.edu.cn).
- The PDF text contains **no data or code availability statement**; a grep for "availab", "release", "public", "github" and "request" hit only references.
- Source: [arXiv PDF](https://arxiv.org/pdf/2301.02393)

**LIVE-Net (Comput Biol Med 2023;157:106886; Sun, Yang, …, Huang)**
- Abstract: "LIVE-Net is trained and evaluated on two datasets: the MICCAI 2008 Coronary Artery Tracking Challenge (CAT08) dataset and head and neck CTA dataset from the clinic … For the segmentation of bilateral carotid and vertebral arteries, our method can … achieve … an average dice similarity coefficient (DSC) of 90.03%".
- Affiliations: Northeastern University (Shenyang) Key Laboratory of Intelligent Computing in Medical Image, and the Department of Radiology, ShengJing Hospital of China Medical University. Contact yangjinzhu@cse.neu.edu.cn.
- The full text and any data statement could not be read: ScienceDirect returned 403, and Semantic Scholar has no open-access PDF.
- Source: [Europe PMC search record, PMID 37062255](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:37062255&resultType=core&format=json)

**Peking University Third Hospital set, in BiSegMamba (arXiv 2605.30972, withdrawn; Zada, Tong, Su, Zhang; Beihang University)**
- v1 text: "We collected a carotid dataset consisting of 115 retrospectively acquired and fully anonymized head-and-neck CTA volumes from Peking University Third Hospital between January 2020 and June 2024 … 80 training, 18 validation, and 17 test volumes … axial spacing of 0.488/0.500/0.625 mm, and 312–945 slices per volume. Ground-truth masks were annotated in 3D Slicer under expert-doctor guidance … The annotation range covered major head-and-neck arterial structures from the lower cervical/aortic-arch region to the intracranial internal carotid artery C6 segment." The reported classes are vessel and plaque.
- Repository README: "Carotid CTA | … | Not publicly available | Private in-house clinical dataset; not distributed with this repository."
- The withdrawal note reads "This version has some technical issues."
- Sources: [arXiv v1 PDF](https://arxiv.org/pdf/2605.30972v1); [BiSegMamba repo](https://github.com/bakhtzadaabshare/BiSegMamba)

**TaG-Net full dataset (401 subjects, 18 segments)**
- The released sample is covered in Q1. The full set is not offered anywhere in the repository or on the Drive.

**3MA, in PC-Net (arXiv 2010.04428; Wu, Pan, Wang, Zhang, Xia, Yu; NWPU, Peking University, Deepwise)**
- "an in-house 3D major artery (3MA) database": "100 computed tomography angiography (CTA) scans" of "major arteries in the head and neck", "captured by TOSHIBA CT machine", labelled by a radiologist and verified by a senior radiologist.
- No availability statement, and the classes are not specified.
- Source: [ar5iv 2010.04428](https://ar5iv.labs.arxiv.org/html/2010.04428)

**Statements that could not be read** (head-and-neck CTA, Chinese multi-centre):
- Fu F et al., "Deep Learning for Head and Neck CT Angiography: Stenosis and Plaque Classification", Radiology 2023 (doi 10.1148/radiol.220996; Xuanwu and Shukun; four tertiary hospitals). pubs.rsna.org returned 403, and the ccmu.edu.cn PDF link redirected to the university homepage.
- Hu B et al., "A deep-learning model for intracranial aneurysm detection on CT angiography images in China", Lancet Digit Health 2024 (doi 10.1016/S2589-7500(23)00268-6). Its data come from "16,546 head and neck CTA examination images from 14,517 patients at eight Chinese hospitals", per a search snippet. The paper is gold open access (CC BY-NC-ND), but thelancet.com returned 403 for both the full text and the PDF.
- Yao L et al., "AASeg: Artery-Aware Global-to-Local Framework for Aneurysm Segmentation in Head and Neck CTA Images", IEEE TMI 2025, from the TaG-Net group. Its repository [PRESENT-Y/AneurysmSeg](https://github.com/PRESENT-Y/AneurysmSeg) holds only a README: "The complete codes will be released once the paper is published."

### Inferences
- The HNA labels are a single vessel mask, and Fu et al.'s merge the VA into a "carotid" or "intracranial" class. Even if obtained, neither gives separate VA geometry without relabelling.
- HNCTA and TaG-Net are the two Chinese corpora with **per-side VA classes**. TaG-Net's sample proves its VA class runs from the subclavian to the basilar. For HNCTA the VA extent is undocumented.

### Gaps
- LIVE-Net, Radiology 2023 and Lancet Digital Health 2024 data statements were not read; all three returned 403.

---

## Q3. Chinese papers that offer data "on request", and how strong each offer is

### Takeaway
There is one real "on reasonable request" offer with VA labels: the npj Digital Medicine 2025 MR vessel-wall paper. The others either offer something other than labelled volumes, or the data have no cervical VA labels.

### Cited Findings
- **Zhang J, Wang W, Dong J, et al., "Rapid vessel segmentation and reconstruction of head and neck angiograms from MR vessel wall images", npj Digit Med 2025, doi 10.1038/s41746-025-01866-x** (Ren Ji Hospital SJTU; United Imaging Healthcare; United Imaging Intelligence; ShanghaiTech; Ningbo Hangzhou Bay Hospital; South TaiHu Hospital):
  - Data availability: "Currently, the source imaging data from four centers cannot be made publicly accessible due to privacy protection but are available from the corresponding author [Huilin Zhao] on reasonable request."
  - Code availability: "The algorithm model developed in this study has been integrated into the post-processing software uWS Plaque Tool and uOmnispace MR Plaque Analysis, both by United Imaging Healthcare Co., Ltd."
  - Labels: "the 1728 original sequence datasets were delineated and annotated using a dedicated plaque analysis software (uWS PlaqueTool, United Imaging Healthcare) to segment various arteries, including the common carotid artery (CCA), internal carotid artery (ICA), external carotid artery (ECA), anterior cerebral artery (ACA), middle cerebral artery (MCA), vertebral artery (VA), basilar artery (BA), and posterior cerebral artery (PCA)". Centerline annotations were also derived.
  - Modality is 3D black-blood MR-VWI, not CTA. The VA coverage (origin or not) is not stated, and neither is the field of view.
  - Source: [PMC12304216 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12304216/fullTextXML). This paper is also listed in `challenges.md` Q6.
- **Fu et al. 2020.** The only on-request items are "software access and the remaining clinical evaluation images generated by technologists" (see Q2), i.e. reconstructions rather than source volumes or labels.
- **Beijing Tiantan intracranial aneurysm database** ("Protocol and Preliminary Results of the Establishment of Intracranial Aneurysm Database for Artificial Intelligence Application Based on CTA Images", Front Neurol 2022):
  - "The raw data supporting the conclusions of this article will be made available by the authors, without undue reservation." This is the Frontiers boilerplate.
  - Its scope is intracranial aneurysm (CTAs 2010–2021, Tiantan plus two centres). No VA labels are described.
  - Source: [PMC9343848 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9343848/fullTextXML)
- **Bengbu vertebrobasilar 3D reconstruction** (Medicine 2025, doi 10.1097/MD.0000000000041931; Second Affiliated Hospital of Bengbu Medical University):
  - "Individual 3-dimensional models of the VBA and skull were constructed … based on head and neck CTA data from 199 adults". Measurements cover the intracranial VA segments, the basilar and the atlas loop.
  - Statement: "All data generated or analyzed during this study are included in this published article [and its supplementary information files]." The models themselves are not in the article.
  - Corresponding author: Zhe Cheng (chengzhe@bbm…, truncated in the text extracted).
  - Source: [PMC11936587 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11936587/fullTextXML)
- **Guangdong Second Provincial General Hospital, C2 pedicle-screw deep-learning paper** (PLoS One 2026, doi 10.1371/journal.pone.0342349):
  - "not publicly available due to patient privacy concerns and institutional policies. Requests for access to the data can be submitted to the Medical Ethics Committee of Guangdong Second Provincial General Hospital … (Email: gd2hllwyh@163.com)".
  - It concerns C2 pedicle CT and mentions the VA only as a risk; there is no VA segmentation.
  - Source: [PMC12893610 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12893610/fullTextXML)

### Inferences
- If a data request is made, Huilin Zhao (Ren Ji) is the only Chinese author found who has committed in print to sharing VA-labelled head-and-neck data. The modality is MR vessel-wall imaging, which is closer to MRA than CTA. Neck coverage would have to be confirmed.
- Bengbu (199 head-and-neck CTAs with 3D VBA models) is a plausible informal request, but the models are intracranial-focused.

---

## Q4. Chinese-language journal literature (CNKI, Wanfang, CQVIP) and the annotation standard

### Takeaway
CNKI and Wanfang could not be queried, but CQVIP (维普) could. Head-and-neck CTA "deep learning" papers in Chinese journals are clinical evaluations of commercial software or of reconstruction kernels. None segments or releases VA geometry, and none carries a data statement. The 2024 annotation consensus requires full arch-to-vertex VA labels for AI-product datasets, but it releases no data; see `challenges.md` Q5.

### Cited Findings
- **CQVIP queries.** Three were run: 「椎动脉 深度学习 分割」 (VA, deep learning, segmentation), 「头颈 CTA 深度学习 血管分割」 (head-and-neck CTA, deep learning, vessel segmentation) and 「头颈CTA 数据集」 (head-and-neck CTA dataset). Of 30 filtered hits, none reports a VA segmentation dataset. Representative abstracts, verbatim:
  - 「目的:探讨数坤人工智能(AI)平台在头颈部动脉血管成像(CTA)检查中的诊断价值。」 Translation: "Objective: to evaluate the diagnostic value of the Shukun AI platform in head-and-neck CTA." The study had 103 patients with DSA as the reference, from 右江民族医学院学报 2022.
  - 「数据分别传输至深睿Dr.Wise头颈CTA智能辅助系统和Siemens Syngo View后处理工作站」 Translation: "data were sent to the Deepwise Dr.Wise head-and-neck CTA assistant and to a Siemens Syngo View workstation." This is a 108-patient AI-versus-manual post-processing comparison.
  - 「探讨基于深度学习的单序列三维空间血管及骨骼识别技术在头颈CT血管成像(CTA)中对去骨效果、血管显示质量的可行性研究。」 Translation: "feasibility of deep-learning single-series 3D vessel/bone recognition for bone removal in head-and-neck CTA." This is 100 patients, in 中国医学计算机成像杂志 2026 issue 4.
  - The remaining hits are DLIR, ClearInfinity and AiCE image-quality studies, 20–110 patients each.
  - Source: [CQVIP search](https://www.cqvip.com/search?k=%E5%A4%B4%E9%A2%88%20CTA%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%20%E8%A1%80%E7%AE%A1%E5%88%86%E5%89%B2). Titles render client-side; abstracts and journals are in the HTML.
- **2024 expert consensus**, 头颈动脉CT血管成像数据标注与质量控制专家共识（2024） ("expert consensus on data annotation and quality control for head-and-neck CTA"), 中华放射学杂志 2024;58(2):150-157, doi 10.3760/cma.j.cn112149-20230522-00358. It was co-issued with 中国食品药品检定研究院 (NIFDC). The following passages were extracted by the fetch tool from brainmed.com:
  - 「数据集图像特征中的血管包括主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、双侧椎动脉。」 Translation: "vessels in the dataset include the aortic arch, brachiocephalic trunk, bilateral subclavian, bilateral CCA, bilateral ICA, basilar and bilateral vertebral arteries."
  - 「每个病例的扫描范围从颅顶至胸骨角水平。」 Translation: "each case is scanned from the vertex to the level of the sternal angle."
  - 「标注边缘贴近血管内壁，标注范围不能超过血管管腔。」 Translation: "label edges hug the inner vessel wall and must not exceed the lumen."
  - 「数据集的用途包括AI产品的训练集、调优集、性能独立测试集、临床试验数据集、日常质控等。」 Translation: "datasets are used as AI-product training, tuning, independent performance-test and clinical-trial sets and for routine QC."
  - Source: [brainmed.com](https://www.brainmed.com/info/detail?id=47109)
- **NIFDC standard project page:** 「人工智能医疗器械 脑卒中CT影像辅助分析软件 算法性能测试方法」 ("AI medical devices: stroke CT image assisted-analysis software, algorithm performance test methods"). Drafted by 首都医科大学附属北京天坛医院 (Beijing Tiantan Hospital) with NIFDC as project owner. It says 「国内也有丰富的数据资源，为标准实施提供物质基础」 ("there are abundant domestic data resources to support implementing the standard"). The page offers only the draft standard and proposal PDFs, with no dataset. It was read with `curl -k` because of a TLS certificate mismatch. — [NIFDC page](https://app.nifdc.org.cn/biaogzx/dataGkZqyj.do?formAction=viewZqyj&id=2c9048d88a743f96018a87ff45b429d7)
- **A Chinese 2025 review** ("Beyond visual inspection: the deep learning revolution in quantitative cerebrovascular imaging", Front Neurosci) states: "Within head and neck vasculature imaging, deep learning (DL) applications have predominantly focused on intracranial arteries." — [PMC13422217 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13422217/fullTextXML)

### Inferences
- In China, the labels we need (whole VA from the arch, per side) are made to a national standard for commercial and regulatory datasets: Shukun, Deepwise, United Imaging and NIFDC test sets. They are not published. The academic English papers above are where any sharing would come from.

### Gaps
- CNKI (kns.cnki.net, search.cnki.com.cn, wap.cnki.net) refused the connection, and Wanfang returned only a JavaScript shell. Chinese-language theses (学位论文), where small labelled sets sometimes appear in appendices, were therefore not searched.

---

## Q5. Other Chinese papers checked and excluded (no cervical VA labels or no data)

| Paper | What it has | Why excluded |
|---|---|---|
| Shi G, Lu H, Hui H, Tian J, "Benefit from public unlabeled data: A Frangi filter-based pretraining network for 3D cerebrovascular segmentation", MedIA 2025 (CAS) | "the largest preprocessed unlabeled TOF-MRA dataset … 1510 subjects … manually annotated segmentation masks for 113 subjects" | Intracranial TOF-MRA |
| Kuang H et al., VANet LVO identification, MedIA 2025 (Central South University) | 2D MIP CTA classifier | No vessel labels |
| Anhui Sci Data 2024 IA CTA and 3D models (already in the English notes) | MCA aneurysm models | Intracranial |
| CSDN vendor post 「高精度CT颅内骨骼、动静脉数据集」 ("high-precision CT skull and intracranial artery/vein dataset"), [blog.csdn.net/weixin_48440952/…/146004187](https://blog.csdn.net/weixin_48440952/article/details/146004187), read via r.jina.ai | 「颅内骨骼、颅内动脉、颅内静脉 1:1 标注，stl支持二次修改标注，可输出stl和nii标签」 ("skull, intracranial arteries and veins labelled 1:1; STL editable; STL and NIfTI labels exportable") | Intracranial only; a data-vendor advert with no case count or terms. The page's "CC 4.0 BY-SA" notice covers the blog post, not data |
| VOF-Net (arXiv 2210.08868, Tsinghua) | "datasets of CTA and MRA" for cerebrovascular segmentation | Cerebrovascular; no data statement in the abstract |

---

## Q6. Incidental non-Chinese leads (outside this angle, recorded so they are not lost)

- **Duke University, "Does Deep Learning Vascular Segmentation on CTA Improve Vertebral Artery Dissection Detection?"**, medRxiv, posted 2026-09-20, doi 10.64898/2026.09.20.26363486 (Zaveri, Zhang, Castellino, Calabrese):
  - Data: "84 manually segmented CTAs plus 17 CTAs from the RSNA Intracranial Aneurysm Challenge (101 CTAs total)".
  - Protocol: "Segmentation followed a predefined protocol including all discernible arterial branches, lumen, wall, atherosclerotic plaque, and intra-arterial devices." The reader study covers V1–V4.
  - Availability: "Trained model weights are available upon request to the authors." "All data produced in the study are available upon reasonable request to the authors."
  - This is a plausible US source of cervical VA labels. The masks may be a single arterial class, which needs checking.
  - Sources: [medRxiv full text](https://www.medrxiv.org/content/10.64898/2026.09.20.26363486v1.full); [Europe PMC PPR1324986](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.64898/2026.09.20.26363486%22&resultType=core&format=json)
- **King's College London and National Taiwan University Hospital, "Vascular geometry characterization for AI-based endovascular navigation"**, IJCARS 2026:
  - Now "vasculature trees from 61 patients" (UK REC 24/LO/0057), segmented with 3D Slicer/VMTK.
  - Only code is released: github.com/SurgicalDataScienceKCL/AI-VascularGeometryCharacterisation. This updates the 12/10-CTA KCL entry in the English notes.
  - Source: [PMC13553664 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13553664/fullTextXML)

---

## Candidate table

| # | Name (中文 / English) | URL | Host | Modality | Cases | Cervical VA labelled? Segments | Label format | Access | Licence | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TaG-Net 样例数据 / TaG-Net sample data (Yao et al., TMI 2023) | [Drive](https://drive.google.com/drive/folders/1Q1GoRfvVZsSgxbia60jANJscOkKXetyx), [GitHub](https://github.com/PRESENT-Y/TaG-Net) | Google Drive + GitHub | CTA, 0.37–0.41 mm in-plane, 0.4 mm slices | 2 labelled patients (1 with CTA), plus 1 binary-mask scan | **Yes. L/R VA from the subclavian (0.4–0.7 mm) to the basilar (0.5–1.0 mm); ~146 mm vertical**; also SCA, arch, BCT, CCA/ICA/ECA, BA, CoW | 18-class 1-voxel centerline NIfTI + point-set txt; CTA NIfTI int16 | **Open, no login** | Code GPL-3.0; no data licence | **Downloadable** |
| 2 | 头颈部MR管壁成像血管重建 / VWI Assistant (npj Digit Med 2025) | [PMC12304216](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12304216/fullTextXML) | — | 3D MR-VWI | 1,728 annotated sequences, 4 centres | VA labelled; extent unstated | Masks + centerlines (uWS PlaqueTool) | "available from the corresponding author [Huilin Zhao] on reasonable request" | — | **On request** |
| 3 | 头颈CTA血管重建 CerebralDoc / Fu et al. Nat Commun 2020 (and Zhihu "HeadNeckCTA2025") | [PMC7518426](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML) | Shukun demo server (dead) | CTA, 0.4–0.6 mm | 18,766 scans / 18,259 patients | VA annotated but merged into 4 classes (bone/aorta/carotid/intracranial) | ITK-SNAP masks | "cannot be made publicly accessible"; software and evaluation images "can be submitted to F.F." | — | **Private** |
| 4 | HNCTA / DeformCL (Yizhun, PKU) | [arXiv 2506.05820](https://arxiv.org/html/2506.05820) | — | CTA | 358 | L/R VA classes (9 classes); extent unstated | Masks | "a private head and neck artery dataset" | — | **Private** |
| 5 | HNA (HKU, Deepwise, ZJU 2nd Affil.) | [arXiv 2301.02393](https://arxiv.org/pdf/2301.02393) | — | CTA, 192–600 slices | 800 | VA inside one binary vessel mask | Binary mask | "in-house"; no statement | — | **Private** |
| 6 | TaG-Net 全集 / TaG-Net full set | as #1 | — | CTA | 401 (TMI abstract) | 18 segments incl. L/R VA (origin to BA per sample) | Centerline labels + masks | Not offered | — | **Private** |
| 7 | LIVE-Net (NEU and Shengjing Hospital) | [PMID 37062255](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:37062255&resultType=core&format=json) | — | CTA | not given in abstract | "bilateral carotid and vertebral arteries" | Masks + centerlines | "dataset from the clinic"; statement unread (403) | — | **Unknown, likely private** |
| 8 | 北医三院颈动脉CTA / PKU Third Hospital (BiSegMamba) | [arXiv v1](https://arxiv.org/pdf/2605.30972v1) | — | CTA | 115 | Arch to ICA C6; VA not named; classes vessel/plaque | Slicer masks | "Not publicly available" | — | **Private** |
| 9 | 3MA (PC-Net, NWPU and Deepwise) | [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.04428) | — | CTA (Toshiba) | 100 | Head-neck "major arteries"; classes unstated | Masks | "in-house"; no statement | — | **Private** |
| 10 | 头颈动脉CTA数据标注共识 (2024) / annotation consensus | [brainmed](https://www.brainmed.com/info/detail?id=47109) | 中华放射学杂志 | Standard | — | Requires bilateral SCA + VA, vertex to sternal angle | Lumen masks | No dataset released | — | **Standard only** |
| 11 | 北京天坛 颅内动脉瘤AI数据库 / Tiantan IA database | [PMC9343848](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9343848/fullTextXML) | — | CTA | not counted here | Aneurysm-focused | — | "will be made available by the authors" | — | On request; out of scope |
| 12 | 椎基底动脉三维重建 / Bengbu VBA 3D study | [PMC11936587](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11936587/fullTextXML) | — | Head-neck CTA | 199 | 3D VBA models (intracranial VA + atlas loop measured) | 3D models (not released) | "included in this published article" (only measurements are) | CC BY-NC (article) | Not released |

---

## Access log: what could and could not be opened

| Resource | Result |
|---|---|
| Europe PMC REST (search + fullTextXML) | **Worked** for PMC7518426, PMC12304216, PMC9343848, PMC11936587, PMC12893610, PMC13422217 and PMC13553664 |
| arXiv HTML/PDF/API (2506.05820, 2301.02393, 2605.30972v1, 2010.04428) | **Worked**. arXiv API keyword search is sparse (it missed DeformCL and HNA) |
| GitHub (TaG-Net, DeformCL, BiSegMamba, AneurysmSeg) and Google Drive public folders/files | **Worked**; Drive serves files with no login and supports HTTP Range |
| Semantic Scholar API | Worked (abstracts, open-access PDF flags) |
| nature.com / RSNA (pubs.rsna.org) / thelancet.com / ScienceDirect / IEEE Xplore | **403** or empty content |
| ccmu.edu.cn Radiology 2023 PDF | Redirected to the university homepage |
| Zhihu p/29607061020 | **Blocked.** Direct: 403 JS challenge. Via r.jina.ai: 「请您登录后查看更多专业优质内容。」 ("log in to see more"). API `/api/v4/articles/…`: 「请求参数异常，请升级客户端后重试。」 ("request parameters abnormal, please upgrade the client"). API `api.zhihu.com/articles/…`: 「您当前请求存在异常，暂时限制本次访问。」 ("your request is abnormal; access temporarily restricted") |
| Baidu www (via WebFetch) | **Worked for 4 queries**, then redirected to the `wappass.baidu.com` CAPTCHA |
| Baidu mobile (curl) | Worked once, then 「百度安全验证」 (security verification) |
| Baidu Xueshu (百度学术) | CAPTCHA on the first request |
| Sogou web | Worked once (obfuscated result URLs), then `/antispider/` |
| 360 so.com | Redirect loop |
| Bing (cn and zh-CN parameters) | Redirected to www.bing.com and returned unrelated or garbled results |
| CNKI (kns.cnki.net, search.cnki.com.cn, wap.cnki.net) | **Connection failed** (curl status 000) |
| Wanfang (s.wanfangdata.com.cn) | JavaScript shell only, no results in HTML |
| CQVIP (www.cqvip.com/search) | **Worked**; abstracts in server HTML, titles rendered client-side |
| Shukun CerebralDoc demo `test.platform.shukun.net` | Does not resolve (status 000) |
| NIFDC standards page | TLS certificate mismatch; read with `curl -k` (GBK-encoded) |
| CSDN blog post | Direct 521 / timeout; read via r.jina.ai |
| Gitee search | JavaScript shell only |
| Science Data Bank (scidb.cn) query API | POST accepted but the query parameter was ignored (default listing returned) |
| HyperAI 超神经 | Dataset search for 「头颈」 ("head and neck") returned `resultsCount: 0`; general search behind a Cloudflare challenge |
| brainmed.com consensus page | Worked |
| cn-healthcare Chinese translation of Fu et al. (PDF) | Worked |
| medRxiv (Duke VAD preprint) | Worked |

---

## If data requests are pursued: who to write to (from the papers)

| Group | Contact | Why |
|---|---|---|
| TaG-Net (Linlin Yao) | yaolinlin23@sjtu.edu.cn, from the README | Reuse terms for the sample; any further labelled cases from the 401-subject set. Its labels already run SCA → VA → BA |
| Ren Ji Hospital SJTU (Huilin Zhao) | Corresponding author; email not captured | The only written "reasonable request" offer with VA labels (MR-VWI) |
| Xuanwu Hospital (Jie Lu; Fan Fu) | imaginglu@hotmail.com | The source data are refused, but software and evaluation images can be requested |
| HKU (Yizhou Yu) / ZJU (Xinyang Hu) for HNA | yizhouy@acm.org, hxy0507@zju.edu.cn | HNA has 800 CTAs, but its labels are a single binary mask |
| Northeastern University (Jinzhu Yang) for LIVE-Net | yangjinzhu@cse.neu.edu.cn | Bilateral carotid and VA labels |
| Duke (Evan Calabrese), non-Chinese | Via medRxiv | 101 CTAs with all arterial branches; data and weights "upon reasonable request" |
