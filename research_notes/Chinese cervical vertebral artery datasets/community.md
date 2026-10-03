# Chinese community posts on cervical vertebral artery (椎动脉) datasets: Zhihu, CSDN, WeChat, Bilibili, Juejin, GitHub/Gitee and the dataset hubs they link to

Scope note: this pass searched Chinese-language community posts and followed each data link to its host. The searches ran on 2026-10-01. Pages were opened as follows:

- **WeChat (微信公众号):** Sogou WeChat search (`weixin.sogou.com`, which worked), then the redirect to `mp.weixin.qq.com`, where the article text was read in full.
- **CSDN:** CSDN serves a "521" JavaScript cookie challenge. It was decoded locally with Windows `cscript`, after which posts were read in full. CSDN's own search API (`so.csdn.net/api/v3/search`) worked for blogs and for uploaded "资源" (resources).
- **Other site APIs:** Bilibili search API, Juejin search API, Heywhale dataset API (`Title=` filter), GitHub API, Hugging Face dataset API, Europe PMC full text.
- **Zhihu:** not readable at all (details under Gaps).

Translations are mine and are given after each Chinese quote.

## Q1. What is "HeadNeckCTA2025 头颈部CTA血管分割" on Zhihu?

### Takeaway
**It is not a dataset.** It is a tutorial write-up in a series by the blogger junqiangchen (WeChat account 最新医学影像技术, CSDN `qq_42886742`, GitHub `junqiangchen`). The blogger reproduces public challenges with a VNet3d. Its "dataset" section only summarises the private Fu et al., Nat Commun 2020 cohort of 18,766 head-and-neck CTAs, and its "数据下载地址" ("data download address") is just the PubMed page of that paper. Fu et al. state that the images "cannot be made publicly accessible due to privacy protection". The "reference project" link points to a generic training-code repository with no data.

### Cited Findings
- **Zhihu post `zhuanlan.zhihu.com/p/29607061020`** ("HeadNeckCTA2025——头颈CTA血管分割", "head-neck CTA vessel segmentation"). It could not be opened (see Gaps).
  - A search-engine extract of the page gives the dataset as the 18,766-image, five-hospital cohort.
  - The same extract gives "数据下载地址为：https://pubmed.ncbi.nlm.nih.gov/32973154/" ("the data download address is: [PubMed 32973154]"). PubMed 32973154 is Fu et al. 2020.
  - This was seen only through the WebSearch summary, not on the page itself — [Zhihu p/29607061020](https://zhuanlan.zhihu.com/p/29607061020)
- **WeChat version of the same series**, read in full: "HeadNeckCTA2025——头颈CTA血管血流动力学仿真评估与虚拟支架重建和贴合度评估" ("head-neck CTA haemodynamic simulation, virtual stent reconstruction and wall-apposition assessment").
  - Account: 最新医学影像技术. Published 2026-03-01 23:00 UTC.
  - Reached through Sogou WeChat search for "头颈CTA血管分割" and "HeadNeckCTA2025". The latter returned only this one article.
  - Opening sentence:
    - Quote: "今天将分享头颈CTA血管血流动力学仿真评估与虚拟支架重建和贴合度评估分割完整实现版本，为了方便大家学习理解整个流程，将整个流程步骤进行了整理…"
    - Translation: "Today I will share a complete implementation of … so that everyone can learn the whole workflow, I have organised the steps…"
  - Dataset section:
    - Quote: "使用的数据集包括回顾性随机收集的 18,766 张原始造影剂增强头颈部 CTA 图像，这些图像来自中国五家三级医院，其中 8593 张来自首都医科大学宣武医院，4407 张来自首都医科大学附属北京友谊医院，2584 张来自上海第六人民医院，2096 张来自河北省人民医院，1086 张来自山东省立医院。"
    - Translation: "The dataset comprises 18,766 retrospectively and randomly collected contrast-enhanced head-and-neck CTA images from five Chinese tertiary hospitals: 8,593 from Xuanwu Hospital, 4,407 from Beijing Friendship Hospital, 2,584 from Shanghai Sixth People's Hospital, 2,096 from Hebei Provincial People's Hospital and 1,086 from Shandong Provincial Hospital."
    - Quote: "每次扫描包括主动脉、颈动脉和颅内动脉，Z 轴上有 ~561-967 个切片。"
    - Translation: "Every scan includes the aorta, carotid and intracranial arteries, with ~561–967 slices along z."
    - Quote: "更多详细内容可以阅读《Fu F, Wei J, Zhang M, et al. Rapid vessel segmentation and reconstruction of head and neck angiograms using 3D convolutional neural network[J]. Nature communications, 2020, 11(1): 4829.》"
    - Translation: "For more detail read Fu et al., Nat Commun 2020."
  - Closing:
    - Quote: "点击阅读原文可以访问参考项目，如果大家觉得这个项目还不错，希望大家给个Star并Fork"
    - Translation: "Click 'Read original' to visit the reference project; if you like it please Star and Fork."
    - The "Read original" link (`msg_source_url`) is `https://github.com/junqiangchen/PytorchDeepLearing`.
  - Source: Sogou WeChat search → `mp.weixin.qq.com`, title as above.
- **`junqiangchen/PytorchDeepLearing`** is described as "Meidcal Image Segmentation Pytorch Version" (last updated 2026-09-26). Its file tree contains no head/neck/CTA/vessel data files. The README mentions only a "centerline dice loss for vessel segmentation" — [GitHub repo](https://github.com/junqiangchen/PytorchDeepLearing)
- **The blogger does link real downloads when they exist.** His same-template CSDN post "TopBrain2026——全脑血管解剖的分割挑战赛" (2026-09-12) gives "数据下载：https://zenodo.org/records/21972006" ("data download: Zenodo 21972006").
  - It says the anatomy is "扩展至大脑前/中/后动脉各分支、基底动脉、椎动脉…" ("extended to ACA/MCA/PCA branches, basilar, vertebral artery…").
  - It also says "公开训练集为 50 个扫描序列（25 对 CTA 与 MRA）" ("the public training set is 50 scans, 25 CTA–MRA pairs").
  - That record is already known to be braincase-cropped (see the English report) — [CSDN 165095878](https://blog.csdn.net/qq_42886742/article/details/165095878)
- **Fu et al. 2020 data-availability statement**, from the Europe PMC full-text XML of PMC7518426:
  - Quote: "Currently, the source imaging data from five tertiary hospitals cannot be made publicly accessible due to privacy protection. The CerebralDoc is accessible from the webserver created for this study at http://test.platform.shukun.net/login. User name: test, password: 123456. Request for software access and the remaining clinical evaluation images generated by technologists can be submitted to F.F."
  - Source: [Europe PMC PMC7518426](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML)
- **Fu et al. VA annotation**, from the same full text:
  - Quote: "prelabeled using ITKSNAP software with masks marking the areas of the bone, aorta, carotid artery, and intracranial artery. They were multiclass classification with four classes. We annotated the arteries including the aorta, subclavian artery, common carotid artery, vertebral artery, internal carotid artery, basilar artery, posterior cerebral artery, middle cerebral artery, and anterior cerebral artery … each CTA scan was annotated with four masks."
  - The paper's figures discuss "Left vertebral artery arising directly from the aorta" and "Vein artifact at the origin of the right vertebral artery". The VA is therefore traced from its origin, but it sits inside a four-class mask (bone / aorta / carotid / intracranial), not as a named VA class.
  - Source: [Europe PMC PMC7518426](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML)
- **The CerebralDoc demo server** `test.platform.shukun.net` no longer resolves in DNS (checked 2026-10-01, reachability only, no login attempted).

### Inferences
- HeadNeckCTA2025 adds nothing downloadable. The "unverified Zhihu page" in the brief is now verified: it is a re-description of Fu et al. Whatever images the blogger used for his own demo results are not offered anywhere he links to.
- The Fu et al./Xuanwu/Shukun (数坤) cohort does trace the VA from its origin, but it is closed. The only channel the paper offers is a request to the first author (F.F., Fan Fu), and it covers "software access and the remaining clinical evaluation images", not the 18,766 training scans.

### Gaps
- The Zhihu page itself was never rendered. The PubMed "download link" comes from a search-engine extract.

## Q2. Do Chinese dataset round-ups and awesome-lists point to any head-and-neck CTA with VA labels?

### Takeaway
**No.** Every Chinese round-up and awesome-list checked lists the same small set of head-and-neck vessel entries, and none has a cervical VA label. All of these are already covered in the English report:

- CAS2023 (intracranial TOF-MRA)
- COSMOS2022 and VWS2021 (carotid vessel-wall MRI)
- CLS2009 (carotid bifurcation CTA)
- SEG.A./AVT (aortic tree)
- TopCoW/TopBrain (braincase-cropped)

### Cited Findings
- **CSDN "医学影像数据集汇总（持续更新）150个"** ("medical imaging dataset round-up, 150 datasets, continuously updated"), 2024-03-12. Its stated source is the Zhihu index "医学数据集介绍文章汇总" (p/661132213).
  - The section "头颈部（29）" ("head and neck, 29") contains only one vessel item:
    - Quote: "CAS2023 (3D MRI, 100例, 1类脑动脉分割)"
    - Translation: "CAS2023 (3D MRI, 100 cases, 1 class, cerebral artery segmentation)"
  - The other head-and-neck entries are OAR, tumour, brain-tissue, teeth and thyroid sets: SegRap 2023, HaN-Seg, PDDCA, HECKTOR, BraTS…, ToothFairy, TN3K…
  - Source: [CSDN 136659841](https://blog.csdn.net/m0_52987303/article/details/136659841)
- **OpenMedLab "Awesome-Medical-Dataset"** (Shanghai AI Lab, GitHub, updated 2026-09-30, 2,158 stars). The "Head and Neck" section's only vessel rows are:
  - "COSMOS2022 | 3D MRI, 75 Cases, 1 Category of Carotid Artery Segmentation"
  - "SMILE-UHURA 2023 | 3D 7T … 1 Category of Cerebral Artery Segmentation"
  - "CAS2023 | 3D MRI, 100 Cases, 1 Category of Cerebral Artery Segmentation"
  - Source: [openmedlab/Awesome-Medical-Dataset](https://github.com/openmedlab/Awesome-Medical-Dataset)
- **linhandev/dataset "医学影像数据集集锦"** ("collection of medical imaging datasets", 3,624 stars). Its only vessel entries are "MSD肝脏血管分割" (MSD hepatic vessels) and "vessel12 肺部血管分割" (VESSEL12 lung vessels) — [linhandev/dataset](https://github.com/linhandev/dataset)
- **Project Imaging-X** (Hugging Face survey of 1,000+ open datasets). The carotid rows are "COSMOS 2022 … Carotid Artery", "VWS 2021 … Carotid Arteries" and "SHINY-ICARUS … 3D DSA … Internal Carotid Artery". There is no vertebral-artery row — [Project-Imaging-X README](https://huggingface.co/datasets/sunli1201/Project-Imaging-X/blob/main/README.md)
- **M3D-Seg** (ModelScope/HF, GoodBaiBai88) aggregates 25 public sets (CHAOS, HaN-Seg, AMOS22, … TotalSegmentator, VerSe, MSD, LUNA16). None has a VA class — [M3D-Seg README](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg/resolve/main/README.md)
- **SAT-DS** (SJTU, 72 datasets): a file-tree grep for vessel/artery hits only MSD hepatic vessel and TotalSegmentator vertebrae configs — [zhaoziheng/SAT-DS](https://github.com/zhaoziheng/SAT-DS)
- **WeChat 起源叙事 (DataBuddy), "医学影像学：头颈部-方寸之间的多模态之战"** ("Medical imaging: the head and neck, a multimodal battle in a small space"), 2026-09-26. Its dataset section:
  - Quote: "头颈部是医学影像公开数据集的富矿区，两条赛道最密集。甲状腺超声方向…头颈肿瘤方向…CLS 2009 收录 200 例颈动脉分叉 CTA，服务血管分析。"
  - Translation: "The head and neck is a rich seam of public datasets, densest in two tracks: thyroid ultrasound … head-and-neck tumours … CLS 2009 includes 200 carotid-bifurcation CTAs for vessel analysis."
  - CLS 2009 is carotid-only; the count of 200 is the post's figure and is not checked here.
  - Source: Sogou WeChat → `mp.weixin.qq.com`
- **Chinese review, 楼陆飞, 应俊杰, 蔡凯俊, 辛宇, "医学影像多血管和气道分割方法综述"** ("A review of multi-vessel and airway segmentation in medical images"), 中国图象图形学报 2024;29(9).
  - The abstract says it "收集了目前公开的各类血管和气道分割数据集" ("collects the currently public vessel and airway datasets").
  - 表3 (脑血管数据集, "brain-vessel datasets") lists only MIDAS21 (TOF-MRA, 109, public "是"), 1000PLUS, 7UP, SCAPIS and PEGASUS (all "否", not public).
  - No head-and-neck CTA set appears.
  - Source: [cjig.cn PDF](https://www.cjig.cn/rc-pub/front/front-article/download/70672131/lowqualitypdf/%E5%8C%BB%E5%AD%A6%E5%BD%B1%E5%83%8F%E5%A4%9A%E8%A1%80%E7%AE%A1%E5%92%8C%E6%B0%94%E9%81%93%E5%88%86%E5%89%B2%E6%96%B9%E6%B3%95%E7%BB%BC%E8%BF%B0.pdf)
- **OpenDataLab "收藏丨20个医疗场景经典、热门数据集资源汇总"** ("20 classic, popular medical datasets"), read via its Juejin mirror. Its "二、头颈部" ("2. head and neck") section contains DRIVE, RITE and CHASE_DB1, all retinal — [Juejin 7298161887882117174](https://juejin.cn/post/7298161887882117174)
- **junqiangchen "医学影像公开数据集（六）"** ("public medical imaging datasets, part 6"; WeChat 2023-11-06; also Zhihu p/665254825). Lists ToothFairy2023, LiTS2017, WORD2021, TotalSegmentator2022, FLARE22, AMOS22, ATLAS2023, HNAS2015, Seg.A.2023 and LNDb2020. The only head-and-neck item is:
  - Quote: "HNAS2015是CT图像头颈器官分割，一共九个结构：脑干、下颌骨、视交叉、左右视神经、左右腮腺和左右下颌下腺"
  - Translation: "HNAS2015 is head-neck organ segmentation on CT, nine structures: brainstem, mandible, chiasm, optic nerves, parotids, submandibular glands"
  - Source: Sogou WeChat → `mp.weixin.qq.com`
- **junqiangchen "OpenClaw-MedAgent——基于 OpenClaw 的医学文献AI助手"** (2026-02-23) embeds an AI-agent-generated file "头颈血管 CTA - 核心原理与数据资源" ("head-neck CTA: core principles and data resources", generated 2026-02-21).
  - Its "公开数据集汇总" ("public dataset summary") lists only generic hosts: TCIA ("HNSCC | 头颈鳞癌 | CT/CTA | 公开"), PhysioNet (MIMIC-IV, eICU), Grand Challenge, OpenNeuro.
  - No head-and-neck CTA vessel-label dataset is named.
  - Source: Sogou WeChat → `mp.weixin.qq.com`
- **Other CSDN, cnblogs and Juejin round-ups**, none with a head-and-neck vessel set:
  - CSDN 阿兵-AI医疗 "医学影像公开数据集": DRIVE, plus the carotid-plaque ultrasound set APS — [CSDN 120941187](https://blog.csdn.net/webzhuce/article/details/120941187)
  - CSDN "头部医学数据集-图像分割": TotalSegmentator only — [CSDN 139551052](https://blog.csdn.net/long119513/article/details/139551052)
  - cnblogs "医学数据集整理": 10 sets, no head-and-neck vessels — [cnblogs 17758930](https://www.cnblogs.com/Flashcc/p/17758930.html)
  - Juejin "2024 最新高质量 - 医学影像数据集" and "医疗影像研究现状（数据集汇总）": no head-and-neck vessel entries — [Juejin 7497890801349345343](https://juejin.cn/post/7497890801349345343); [Juejin 7029561131446304781](https://juejin.cn/post/7029561131446304781)

### Inferences
- The Chinese community indexes are downstream of the same international challenge catalogue. A Chinese head-and-neck CTA vessel dataset released through a challenge or an open repository would almost certainly appear in OpenMedLab's or junqiangchen's lists, and none does. Neither list shows any such set as of 2026-09-30.

## Q3. Platform searches: CSDN, CSDN resources, Bilibili, Juejin, GitHub, Gitee, Hugging Face, Heywhale, AI Studio, Tianchi, ModelScope, OpenDataLab

### Takeaway
No downloadable head-and-neck CTA with VA labels on any of these platforms. The only VA-relevant code release, DeformCL (which trained on the 358-case HNCTA), published its HaN-Seg pipeline and later its ImageCAS coronary masks, but not HNCTA.

### Cited Findings
- **CSDN search API**, blog and "all" scopes, queries "椎动脉 分割 数据集", "HeadNeckCTA2025" (0 hits), "头颈CTA血管分割", "颈动脉 分割挑战赛", "脑血管 分割挑战赛 CTA" and "椎动脉". There were no dataset hits. The only vessel-challenge post is the TopBrain2026 one above.
  - Searches of CSDN's uploaded "资源" (download scope) for "头颈 CTA 数据集", "椎动脉 CTA", "颈动脉 CTA 分割 数据集" and "头颈血管分割 数据" returned only clinical `.doc`/`.ppt`/`.pdf` documents. Examples: "头颈部CT血管成像扫描方案与注射方案专家共识.wps" and "CT血管造影及后处理方式.ppt". No image or label archives.
- **Bilibili search API**, queries "头颈CTA 血管分割", "椎动脉 分割", "头颈 血管 数据集" and "血管分割 数据集 CTA": clinical teaching videos only. Examples: "头颈CTA入门解读—解剖", "椎动脉解剖及分段、椎基底动脉主要分支", "3D Slicer 手动精细分割血管教程". No dataset links in the descriptions.
- **Juejin search API**: no head-and-neck vessel posts. The round-ups are listed in Q2.
- **GitHub repository search.**
  - Zero relevant repos for "头颈 CTA", "椎动脉", "头颈血管", "head neck CTA vessel", "cervical artery segmentation CTA" and "vertebral artery segmentation".
  - **DeformCL (official code, `barry664/DeformCL`):**
    - The README covers only HaN-Seg preparation.
    - Quote: "**2026-05-07**: We release the ImageCAS vessel masks used in our paper, including LAD, LCX, and RCA annotations."
    - There is no HNCTA release.
    - The maintainer answered an earlier request for the ImageCAS runner with: "Due to restrictions on releasing the new annotations, the runner for the ImageCAS dataset is currently unavailable" (issue #1, 2025-06-30). The masks were released later.
    - Source: [barry664/DeformCL](https://github.com/barry664/DeformCL); [issue #1](https://github.com/barry664/DeformCL/issues/1)
  - **AASeg** ("Artery-aware … Aneurysm Segmentation in Head and Neck CTA Images"): README only: "The complete codes will be released once the paper is published." — [PRESENT-Y/AneurysmSeg](https://github.com/PRESENT-Y/AneurysmSeg)
- **Gitee:** the public search API (`gitee.com/api/v5/search/repositories`) returned `[]` for every query, including a generic "medical segmentation" control, and Gitee web search renders client-side. Gitee could not be searched.
- **Hugging Face dataset API**, searches for vertebral, carotid, head-neck, headneck, HNCTA, "neck CTA", cerebrovascular, "CTA vessel" and "aortic arch": no head-and-neck CTA vessel dataset. The only carotid hit is `zeahub/zea-carotid-2023`, which is ultrasound.
- **Heywhale (和鲸) dataset API** (`/api/datasets?Title=`):
  - 血管 (vessel): 10 hits, all cardiovascular tabular sets, MSD hepatic vessel or CHASEDB1.
  - 头颈 (head-neck): 1 hit, "解剖学（头颈部分）的CT分割数据" ("anatomy, head-and-neck part, CT segmentation data"). It is PDDCA ("A Public Domain Database for Computational Anatomy", one 1.6 GB file), with no vessels.
  - 椎动脉, 颈动脉, 脑血管, vessel, artery: 0 relevant hits.
  - Source: [Heywhale PDDCA](https://www.heywhale.com/mw/dataset/6399a3afda3e2df846274483)
- **Baidu AI Studio, Tianchi, ModelScope, OpenDataLab:** their internal search APIs could not be reached (see Gaps). Site-restricted WebSearch for "头颈 血管 分割 数据集" and "椎动脉 颈动脉 CTA" returned only:
  - generic sets: AI Studio "医学图像2d语义分割数据集2" and "医学MNIST", Tianchi DRIVE and "Lung Vessel Segmentation", ModelScope M3D-Seg;
  - an AI Studio forum post "史上最全医学影像数据集列表（附下载）" ("the most complete list of medical imaging datasets, with downloads"), which returned 403/503 when opened.

## Q4. How Chinese head-and-neck CTA labels are defined, and the "standard database"

### Takeaway
China's national expert consensus on head-and-neck CTA annotation includes subclavian arteries and both VAs. It specifies a scan range from the vertex to the sternal angle, so Chinese labelled head-and-neck CTA corpora built to it would contain the full cervical VA from its origin. The consensus says a matching "standard database" is to be built with the national device-testing institute (中国食品药品检定研究院, NIFDC). No public download or application route for such a database was found.

### Cited Findings
- **【规范与共识】头颈动脉CT血管成像数据标注与质量控制专家共识** ("Expert consensus on data annotation and quality control for head-and-neck arterial CTA"), 中华放射学杂志 2024;58(2):150-157.
  - Reposted by the journal's WeChat account on 2024-04-07.
  - Authors: Chinese Society of Radiology AI working group, Beijing Radiology AI group and 中国食品药品检定研究院.
  - Vessel classes:
    - Quote: "数据集图像特征中的血管包括主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、双侧椎动脉、双侧大脑前动脉、双侧大脑中动脉、双侧大脑后动脉、前交通动脉和后交通动脉。"
    - Translation: "The vessels in the dataset comprise the aortic arch, brachiocephalic trunk, bilateral subclavian, common carotid and internal carotid arteries, basilar, bilateral vertebral, anterior/middle/posterior cerebral arteries, and anterior and posterior communicating arteries."
  - Scan range:
    - Quote: "每个病例的扫描范围从颅顶至胸骨角水平，采用螺旋扫描方式从足侧向头侧扫描。"
    - Translation: "Each case is scanned from the vertex to the sternal angle, helically, caudal to cranial."
  - Data quality:
    - Quote: "图像矩阵至少为512×512，增强图像的层厚小于2 mm … 不建议对原始影像数据做…图像裁剪等操作。"
    - Translation: "Matrix ≥512×512, contrast slice thickness <2 mm … cropping of raw images is not recommended."
  - Aim:
    - Quote: "本部分的主要目标为建立配套的头颈动脉CTA影像标准数据库"
    - Translation: "The main goal of this part is to build a matching standard head-and-neck CTA image database."
  - Lumen labelling:
    - Quote: "应标注所有对比剂充盈区域…标注边缘贴近血管内壁，标注范围不能超过血管管腔（图2~5）"
    - Translation: "All contrast-filled regions must be labelled … edges hugging the inner wall, never beyond the lumen (Figs 2–5)."
    - Fig. 2–3 are a straightened VA and its lumen label.
  - Label storage:
    - Quote: "必须以开放可读的格式（如DICOM）存储全部标注结果"
    - Translation: "All annotations must be stored in an open, readable format such as DICOM."
  - Source: Sogou WeChat → `mp.weixin.qq.com`, title as above (account 中华放射学杂志). Also reposted by 熊猫放射 and CT Clinical Evidence.
- **Searches for NIFDC access returned nothing on access terms.**
  - Sogou WeChat queries: "中检院 头颈 CTA 标准数据集", "人工智能医疗器械 标准数据库 头颈动脉", "头颈动脉CTA 标准数据库 建设".
  - WebSearch query: "中国食品药品检定研究院 人工智能医疗器械 标准数据集 头颈动脉CTA".
  - Results were limited to the consensus itself, device-registration news (e.g., Deepwise's 深睿医疗 head-neck AI product licences), and NIFDC standards work: the national standard on AI medical-device datasets and the "脑卒中CT影像辅助分析软件算法性能测试方法" draft.
  - Source: [NIFDC draft standard page](https://app.nifdc.org.cn/biaogzx/dataGkZqyj.do?formAction=viewZqyj&id=2c9048d88a743f96018a87ff45b429d7)

### Inferences
- If a Chinese head-and-neck CTA corpus labelled to this consensus were ever obtained, it would give origin-to-basilar VA lumens at ≤0.4 mm in-plane with arch and subclavian context. That is the ideal graft source. Everything found indicates these corpora stay inside hospitals, companies (数坤/Shukun, 联影/UII, 深睿/Deepwise, 东软/Neusoft) and the regulator's test bank.
- The NIFDC "standard databases" are, as far as the public record shows, used to test products for registration. No evidence of distribution to researchers was found. This is an inference from absence, not a confirmed policy.

## Q5. Candidate table

| # | Name (中文 / English) | URL | Host | Modality | Cases | Cervical VA labelled? | Label format | Access / licence | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HeadNeckCTA2025——头颈CTA血管分割 / "HeadNeckCTA2025 head-neck CTA vessel segmentation" (tutorial) | [Zhihu](https://zhuanlan.zhihu.com/p/29607061020); WeChat 最新医学影像技术 (2026-03-01); [code](https://github.com/junqiangchen/PytorchDeepLearing) | Zhihu / WeChat / GitHub | CTA | Describes Fu et al.'s 18,766 | Not offered | — | "数据下载地址" (data download address) = PubMed page only | Q1 quotes | **No availability**: a write-up, not a dataset |
| 2 | 宣武医院/数坤 CerebralDoc 头颈CTA数据 / Fu et al. Nat Commun 2020 | [PMC7518426](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7518426/) | Hospitals + Shukun | CTA (aorta to vertex) | 18,766 (18,259 patients) | VA traced from origin ("vertebral artery arising directly from the aorta"), inside four-class masks (bone/aorta/carotid/intracranial) | ITK-SNAP masks (internal) | "cannot be made publicly accessible due to privacy protection"; software and clinical-evaluation images by request to F.F.; demo server dead | Q1 quotes | **Not public**; partial request route |
| 3 | 头颈动脉CTA影像标准数据库 / national head-neck CTA standard database (consensus) | 中华放射学杂志 2024;58(2):150-157 | CSR AI group + NIFDC | CTA, vertex to sternal angle | Not stated | Specified: bilateral subclavian + bilateral VA + basilar | DICOM labels | No public route found | Q4 quotes | **No availability found** |
| 4 | HNCTA (DeformCL) | [barry664/DeformCL](https://github.com/barry664/DeformCL); [arXiv 2506.05820](https://arxiv.org/html/2506.05820) | GitHub (code) | CTA | 358 | L/R VA among nine classes (extent not stated) | — | Not released; the same authors later released ImageCAS masks once restrictions lifted | Q3 quotes | **Not public**; ask the authors |
| 5 | TopBrain2026 (CSDN write-up) | [CSDN 165095878](https://blog.csdn.net/qq_42886742/article/details/165095878) → [Zenodo 21972006](https://zenodo.org/records/21972006) | CSDN → Zenodo | CTA + MRA | 25 public pairs | VA labelled but braincase-cropped (V4 ± upper V3) | NIfTI | Already in the English report | Q1 quote | Downloadable, not cervical |
| 6 | CLS 2009 (via WeChat round-up) | WeChat 起源叙事 2026-09-26 | WeChat → MICCAI 2009 | CTA | Post says 200 | Carotid bifurcation only | — | Already excluded | Q2 quote | Not VA |
| 7 | 解剖学（头颈部分）的CT分割数据 = PDDCA | [Heywhale](https://www.heywhale.com/mw/dataset/6399a3afda3e2df846274483) | Heywhale | CT | 48 (PDDCA) | No vessels | — | Re-upload | Q3 | Not VA |
| 8 | 利用深度學習產生去骨電腦斷層掃描血管造影改進椎動脈分割 / "Bone-subtracted CTA for VA segmentation" (NCU Taiwan master's thesis, 2020) | [NCU ETD](https://etd.lib.ncu.edu.tw/detail/63d6e9f8791a744991983468328bb27f) | NCU (Taiwan) | CTA | Unknown | VA segmentation is the topic; extent unknown | Unknown | Could not open (DNS failure for `etd.lib.ncu.edu.tw` and `ir.lib.ncu.edu.tw`) | Search-engine title only | **Unverified**; likely in-house hospital data |

## Searched, and what could not be opened

- **Sogou WeChat search** (worked; top 10 per query, page 1):
  - 头颈CTA血管分割
  - 椎动脉 分割 数据集
  - 头颈部CTA 数据集
  - HeadNeckCTA2025
  - 头颈部CTA 数据集 公开
  - 头颈血管 分割 数据集
  - 颈部血管 分割 公开数据集
  - 椎动脉 CTA 数据集
  - 头颈动脉 CTA 数据集 下载
  - 主动脉弓 头颈 血管 分割 数据集
  - 头颈 CTA 数据集 网盘
  - 颈动脉 CTA 公开数据集 分割
  - 椎基底动脉 数据集 分割
  - 血管中心线 数据集 头颈
  - 脑血管 CTA 标注 数据集 开源
  - HNCTA 数据集
  - 中检院 头颈 CTA 标准数据集
  - 人工智能医疗器械 标准数据库 头颈动脉
  - 头颈动脉CTA 标准数据库 建设
  - 医学影像公开数据集 最新医学影像技术 (pages 1–2)
  - 血管分割 公开数据集 汇总
  - 椎动脉 分割 深度学习 数据
  - 颈部血管 数据集 开源 下载
  - 头颈动脉 分割 数据集 椎动脉 开源
  - Articles opened in full: HeadNeckCTA2025; 起源叙事 head-neck round-up; OpenClaw-MedAgent; consensus; 医学影像公开数据集（六）.
- **CSDN** (opened by decoding the 521 cookie challenge):
  - Posts read: 136659841, 165095878, 120941187, 139551052.
  - Search API: the blog/all and download queries listed in Q3.
- **Other platform APIs:**
  - Bilibili: 4 queries.
  - Juejin: 3 queries, 4 posts read.
  - GitHub: 10 repository queries; DeformCL issues read.
  - Hugging Face datasets: 10 queries.
  - Heywhale `Title=` filter: 9 queries.
- **WebSearch** (US engine) with Chinese queries:
  - HeadNeckCTA2025, with and without "数据下载地址 pubmed"
  - 头颈部CTA 血管分割 数据集 公开 下载
  - 椎动脉 分割 数据集 CTA 公开
  - 医学影像数据集 汇总 血管 头颈 颈动脉 椎动脉
  - 颈动脉 椎动脉 CTA 分割 开源 github 数据
  - 头颈CTA 挑战赛 血管分割 数据集
  - awesome 医学影像公开数据集汇总 github 中文 血管
  - 头颈动脉 分割 公开数据集 椎动脉 颈总动脉 锁骨下动脉 标注
  - site:zhihu.com 椎动脉 分割 数据集
  - site:cnblogs.com 血管分割 数据集 CTA 头颈
  - 头颈CTA 血管分割 数据集 申请 通讯作者 合理要求
  - Traditional-Chinese VA queries
  - 2025 头颈动脉 CTA 数据集 发布
  - Domain-restricted runs over aistudio/heywhale/modelscope/opendatalab/scidb/tianchi/gitee
- **Could not open:**
  - **Zhihu:** every page and API was blocked.
    - `zhuanlan.zhihu.com` returns 403 with a `zse-ck` JS challenge.
    - `www.zhihu.com/api/v4/articles/…` returns 403 "请求参数异常，请升级客户端后重试" ("abnormal request parameters, please upgrade the client and retry").
    - `api.zhihu.com/articles/…` returns 403, code 40362: "您当前请求存在异常，暂时限制本次访问" ("your request is abnormal; access is temporarily restricted").
    - The `r.jina.ai` reader returns "请您登录后查看更多专业优质内容" ("please log in to see more").
  - **Baidu search:** captcha ("百度安全验证", "Baidu security verification").
  - **Sogou general web search:** antispider.
  - **360 Search:** redirect loop.
  - **Bing CN via curl:** unrelated results.
  - **Baidu AI Studio:** search API not discoverable; forum post 959698 returned 503/403.
  - **Gitee search API:** empty for every query.
  - **Tianchi API:** 302. **OpenDataLab API:** 404.
  - **CSDN blog-list API:** captcha ("请进行安全验证").
  - **Taiwan NCU thesis hosts** (`etd.lib.ncu.edu.tw`, `ir.lib.ncu.edu.tw`): DNS failure.
  - **Shukun demo server** `test.platform.shukun.net`: DNS failure.
  - **cs.cnpharm.com NIFDC-related article:** 404.

## Gaps
- **Zhihu was never rendered.** Its round-ups were seen only through their CSDN/WeChat mirrors, which carry the same lists. A Zhihu-only answer or comment pointing to a 网盘 (cloud-drive) link would have been missed.
- **No usable Chinese general search engine** (Baidu, Sogou web and 360 were all blocked). Coverage of Baidu-indexed content (百度文库, 百度贴吧, 网盘 shares) is therefore weak.
- **Not searched:** Douyin, Xiaohongshu, QQ/WeChat groups and paid "数据集代下" ("dataset download on your behalf") sellers. Data offered there would not be legitimately redistributable anyway.
- **Out of scope for this angle:** the private-corpus papers (HNA 800, LIVE-Net, the 115-CTA set from Peking University Third Hospital) and Chinese repositories such as ScienceDB were left to the other angles, and so were not checked here.
