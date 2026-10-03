# Chinese challenges and competitions: any public set that labels the cervical vertebral artery?

Scope: Chinese-hosted challenges and competitions with 3D head-and-neck angiography (CTA, MRA or meshes). The question is whether any of them labels the vertebral artery (椎动脉, VA) in the neck (V1–V3, 颈段/横突孔段/寰椎段), ideally from the subclavian origin (锁骨下动脉) up to the basilar (基底动脉). This file builds on the English-language notes in `research_notes/Public cervical vertebral artery datasets/challenges.md` and the report `reports/Public cervical vertebral artery datasets.md`, and does not repeat what they already established (TopCoW, TopBrain crop, AortaSeg24, AVT, HaN-Seg, carotid wall). Research date: 2026-10-01.

## Q1. Does any Chinese-hosted challenge or competition release cervical VA labels?

### Takeaway
No. Every Chinese-run challenge and competition I could reach falls into one of four groups:
- intracranial only (Tsinghua's CAS2023 and INSTED 2024, TOF-MRA);
- carotid only (Tsinghua/UW carotid vessel wall 2021 and COSMOS 2022, MRI; CSV 2026, ultrasound);
- head-and-neck radiotherapy organs with no vessels (SegRap2023/2025);
- other vascular beds or non-imaging tasks (Parse2022, ATM22, KiPA22, GAVE, CARE, and every Tianchi contest).

**"HeadNeckCTA2025 — 头颈CTA血管分割" is not a challenge.** It is a Zhihu blog post that retells Fu et al. (Nat Commun 2020) and names the 18,766-scan cohort "HeadNeckCTA2025". That cohort's own data statement says the images "cannot be made publicly accessible due to privacy protection". Its ground truth also had no separate VA class: four masks per scan (bone, aorta, carotid artery, intracranial artery).

The only Chinese document that specifies full-length VA annotation on head-and-neck CTA is a 2024 expert consensus on how to annotate such data. It is a standard, and no dataset was released with it.

### At-a-glance

| Candidate (中文 / English) | Host / platform | Modality, cases | Cervical VA labelled? | Access | Verdict |
|---|---|---|---|---|---|
| HeadNeckCTA2025——头颈CTA血管分割 (head-and-neck CTA vessel segmentation) | Zhihu post [p/29607061020](https://zhuanlan.zhihu.com/p/29607061020) | Describes the 18,766 Fu et al. CTAs | Fu et al. GT = 4 masks (bone, aorta, carotid, intracranial); **no separate VA class** | Zhihu login wall; underlying data "cannot be made publicly accessible" | **No availability**; blog, not a challenge |
| 脑动脉分割挑战赛 CAS2023 (Cerebral Artery Segmentation) | Tsinghua + Univ. Washington; CodaLab; [Zenodo 7839970](https://zenodo.org/records/7839970) | TOF-MRA, 100 | No; intracranial binary arteries | CodaLab registration; proposal CC BY-ND 4.0 | Exclude |
| 颅内动脉瘤与颅内动脉狭窄检测分割挑战赛 INSTED 2024 | Tsinghua; [Codabench 2139](https://www.codabench.org/competitions/2139/); [Zenodo 10990482](https://zenodo.org/records/10990482) | TOF-MRA | No; aneurysm and intracranial stenosis | Codabench; proposal CC BY 4.0 | Exclude |
| 颈动脉管壁分割 (2021) / COSMOS 2022 | Tsinghua + UW; Grand Challenge | Black-blood MRI | No; carotid only (see the English notes) | Zenodo | Exclude |
| TopBrain 2025/2026 | Grand Challenge; co-organised by HIT Shenzhen, Zhongnan Hospital of Wuhan University, SJTU | CTA + MRA | VA labelled but braincase-cropped (see the English report) | Zenodo, open | Intracranial end only |
| SegRap2023 / SegRap2025 (鼻咽癌放疗靶区与危及器官分割) | UESTC / Sichuan Cancer Hospital; Grand Challenge + GitHub | CT + contrast CT, 200 | No; 45 OARs, no vessel class | End-user agreement; Google Drive / Baidu download | Exclude |
| CSV 2026 (颈动脉斑块分割与易损性评估) | ISBI 2026 | Ultrasound, 1,500 image pairs, 7 hospitals | No; carotid plaque | — | Exclude |
| Tianchi (天池), all 577 competitions | tianchi.aliyun.com | — | No vessel or CTA contest exists | — | Exclude |
| 头颈动脉CT血管成像数据标注与质量控制专家共识（2024） (expert consensus on head-and-neck CTA annotation and quality control) | 中华放射学杂志 (Chinese Journal of Radiology) 2024;58(2):150-157 | Standard, not data | Specifies **双侧椎动脉** (bilateral VA) plus arch, brachiocephalic trunk and subclavians; scan from vertex to sternal angle | Text readable (brainmed mirror) | Annotation spec only; **no dataset** |

## Q2. What is "HeadNeckCTA2025 头颈部CTA血管分割", and can its data be obtained?

### Takeaway
It is a Zhihu column post titled "HeadNeckCTA2025——头颈CTA血管分割". Its text, as visible through search-engine snippets, restates the Fu et al. cohort and cites the Fu et al. PubMed record. It names no organiser, challenge platform, licence or download, and readers are reported to have asked how to download the data. Fu et al.'s own data statement rules out public access. I could not read the post itself.

### Cited Findings
- **The page could not be opened.** Each attempt failed differently:

  | Attempt | Result | Translation |
  |---|---|---|
  | WebFetch | HTTP 403 | — |
  | curl with a browser user agent | HTTP 403 and Zhihu's `zse-ck` anti-bot shell | — |
  | Reader proxy (r.jina.ai) | Page titled "安全验证 - 知乎" with the text "请您登录后查看更多专业优质内容。" | "Security verification – Zhihu"; "Please log in to view more professional, high-quality content." |
  | Zhihu v4 API (`/api/v4/articles/29607061020`) | `{"message":"请求参数异常，请升级客户端后重试。","code":10003}` | "Request parameters abnormal; please upgrade the client and retry." |
  | api.zhihu.com | `"您当前请求存在异常，暂时限制本次访问。"` (code 40362) | "Your current request is abnormal; this access is temporarily restricted." |
  | Wayback Machine | No snapshot | — |

  I did not try to bypass the login wall, for example by spoofing a search-engine crawler.
- **What the search engines show of the post.** These are summaries generated by the search tool, not verbatim quotes:
  - It describes "回顾性随机收集的18,766张原始造影剂增强头颈部CTA图像，来自中国五家三级医院，包括首都医科大学宣武医院、首都医科大学附属北京友谊医院、上海第六人民医院、河北省人民医院和山东省立医院" ("18,766 original contrast-enhanced head-and-neck CTA images collected retrospectively at random from five Chinese tertiary hospitals: Xuanwu Hospital of Capital Medical University, Beijing Friendship Hospital of Capital Medical University, Shanghai Sixth People's Hospital, Hebei General Hospital and Shandong Provincial Hospital").
  - The per-hospital counts are 8,593 / 4,407 / 2,584 / 2,096 / 1,086.
  - The task is "头颈血管分割" (head-and-neck vessel segmentation).
  - The post points to PubMed 32973154, which is Fu et al., and readers asked how to download the dataset.

  Sources: WebSearch results for "HeadNeckCTA2025 知乎 数据集 标注 …" and "8593 4407 2584 头颈 CTA", both pointing to [Zhihu p/29607061020](https://zhuanlan.zhihu.com/p/29607061020).
- **The numbers match Fu et al. exactly.** The paper says "8593 … Xuanwu … 4407 … Beijing Friendship … 2584 from Shanghai Sixth People's Hospital, 2096 from Hebei General Hospital and 1086 from Shandong Provincial Hospital. … Each scan included the aorta, carotid artery, and intracranial artery with ~561–967 slices in the Z axis" — [Fu et al., Nat Commun 2020, PMC7518426](https://pmc.ncbi.nlm.nih.gov/articles/PMC7518426/), full text read through the [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML).
- **Fu et al. data availability**, read through Europe PMC; this was not readable in the English pass: "Currently, the source imaging data from five tertiary hospitals cannot be made publicly accessible due to privacy protection. The CerebralDoc is accessible from the webserver created for this study at http://test.platform.shukun.net/login … Request for software access and the remaining clinical evaluation images generated by technologists can be submitted to F.F." — [PMC7518426](https://pmc.ncbi.nlm.nih.gov/articles/PMC7518426/).
  - On 2026-10-01 the host `test.platform.shukun.net` returned NXDOMAIN (it no longer exists). I did not log in.
- **Fu et al. annotation: no VA class.**
  - "all 18,259 samples were … prelabeled using ITKSNAP software with masks marking the areas of the bone, aorta, carotid artery, and int[racranial artery] … each CTA scan was annotated with four masks to obtain the reliable ground truth" — [PMC7518426](https://pmc.ncbi.nlm.nih.gov/articles/PMC7518426/).
  - The paper does discuss the VA origin as a failure mode: "Left vertebral artery arising directly from the aorta; the CPR and curved-MPR failed", and "Bone remains at the origin of the left vertebral artery".
  - Severely anomalous origins "were not included in the training and validation sets".
- **Possible author.** The "XXX20XX——<task>" title format matches a blogger who posts challenge write-ups. On Tencent Cloud this is "医学处理分析专家" ("medical processing and analysis expert"; Tencent Cloud developer account 7498388), for example "Top_Cow 2023——…" and "AortaSeg2024——CTA主动脉相关区域子结构23类分割" ("23-class segmentation of aortic regions on CTA"). Those posts say "本文参与腾讯云自媒体同步曝光计划，分享自微信公众号。" ("This article is part of Tencent Cloud's self-media syndication programme, shared from a WeChat public account.") — [Tencent Cloud 2377559](https://cloud.tencent.com/developer/article/2377559).
  - The account's article list that I could fetch shows 2024 posts only, and has no HeadNeckCTA2025 copy.
  - A GitHub account with matching challenge repositories (`junqiangchen`, 368 repositories) has no HeadNeckCTA repository.
  - I could not confirm that the Zhihu author is this blogger.

### Inferences
- "HeadNeckCTA2025" is a blogger's label for a training experiment on, or a write-up of, the Fu et al. data. It is not an organised challenge with a public training set. No organiser, platform, registration or licence appears anywhere.
- Even if the Fu et al. data were obtained, its labels would not give separate VA geometry. The VA sits inside a "carotid artery" (neck vessel) mask or an "intracranial artery" mask together with the carotids. Separating the VAs would mean re-labelling, though the scans do cover the arch to the vertex.

### Gaps
- The Zhihu post body could not be read. If it did contain a download link (for example a Baidu Netdisk link), I have not seen it. The search snippets saying readers asked how to download suggest there is no clear link.

## Q3. Which Chinese-hosted international challenges (MICCAI, ISBI, Grand Challenge, Codabench) touch head-and-neck vessels, and do any label the cervical VA?

### Takeaway
None labels the cervical VA. I checked the full MICCAI challenge lists for 2023–2026, the ISBI lists for 2024–2026, all 264 Grand Challenge challenges (through its API), and a keyword sweep of public Codabench competitions. The Chinese-run vessel-related challenges are intracranial TOF-MRA (CAS2023, INSTED), carotid (vessel wall 2021, COSMOS 2022, CSV 2026), retinal (GAVE/GAVE2), or other beds (Parse2022 pulmonary, KiPA22 renal, ATM22 airway).

### Cited Findings
- **MICCAI 2026** — [MICCAI 2026 challenges](https://conferences.miccai.org/2026/en/challenges.asp):
  - Vessel-related: TopAneu (intracranial aneurysm; ~850 scans; organisers TU/e, UZH, ZHAW, CHUV, UMC Utrecht, Charité — not Chinese), TopBrain v2, CATHACTION (catheter segmentation), MVAA (mitral valve).
  - Chinese-run: CARE 2026 (Fudan; zmic.org.cn), CMRxRecon2026 (4D flow MRI), MRIxFields, GAVE2 (retinal vessels on Baidu AI Studio).
  - No head-and-neck CTA vessel challenge.
- **CARE 2026 tracks** are "CARE-Left Atrium, CARE-Liver, CARE-Myocardium, CARE-Whole Heart", with no vessel or angiography track — [CARE 2026](https://www.zmic.org.cn/care_2026/).
- **MICCAI 2025** — [MICCAI 2025 challenges](https://conferences.miccai.org/2025/en/challenges.asp):
  - Chinese-run: CARE 2025, GAVE (retinal), PDCADxFoundation, REG2025, SegRap2025.
  - Vessel: TopBrain 2025 only.
- **MICCAI 2024** — [MICCAI 2024 challenges](https://conferences.miccai.org/2024/en/challenges.asp):
  - Vessel-related: TopCoW, INSTED, AortaSeg24.
  - INSTED is Tsinghua's (see below).
- **MICCAI 2023** — [MICCAI 2023 challenges](https://conferences.miccai.org/2023/en/challenges.asp): vessel-related are ARCADE (coronary X-ray), SEG.A., the CoW benchmark and SegRap2023 (no vessels). CAS2023 was a MICCAI 2023 challenge per its proposal (below).
- **CAS2023 (Cerebral Artery Segmentation)** — [Zenodo 7839970](https://zenodo.org/records/7839970); [CodaLab 9804](https://codalab.lisn.upsaclay.fr/competitions/9804):
  - Organisers: Tsinghua University (Huijun Chen, Xihai Zhao and others) and University of Washington (Chun Yuan, Niranjan Balu).
  - Task: "segment cerebral arteries from 3D TOF-MRA images acquired from patients with symptomatic intracranial artery stenosis". The proposal is CC BY-ND 4.0.
- **INSTED 2024** — [Zenodo 10990482](https://zenodo.org/records/10990482); [Codabench 2139](https://www.codabench.org/competitions/2139/):
  - Organisers: Tsinghua University (Chen Huijun, Zhao Xihai, Li Rui and others).
  - Data: "3D TOF-MRA images from healthy volunteers and patients with intracranial pathologies"; labels are aneurysms and intracranial stenosis. The proposal is CC BY 4.0.
- **SegRap2023** — [SegRap2023 dataset](https://segrap2023.grand-challenge.org/dataset/):
  - Data: 200 nasopharyngeal-carcinoma patients with CT + contrast CT, 45 OARs. The list is brain, brainstem, chiasm, cochleae, oesophagus, Eustachian tubes, eyes, hippocampi, IACs, larynx (glottic/supraglottic), lenses, mandible, mastoids, middle ears, optic nerves, oral cavity, parotids, pharynx, pituitary, spinal cord, submandibulars, temporal lobes, thyroid, TMJs, trachea, tympanic cavities and vestibular canals.
  - No carotid, vertebral or jugular class.
  - Access needs a signed End User Agreement; downloads are password-protected on Google Drive or Baidu.
- **SegRap2025** covers GTV and lymph-node CTV tasks only — [arXiv 2601.20575](https://arxiv.org/abs/2601.20575v1).
- **ISBI**:
  - 2026 lists one vessel challenge, "CSV 2026: Carotid Plaque Segmentation and Vulnerability Assessment" (ultrasound; "1500 paired transverse–longitudinal images, collected from seven hospitals") — [ISBI 2026 challenges](https://biomedicalimaging.org/2026/challenges/).
  - ISBI 2025 has no vessel challenge — [ISBI 2025](https://biomedicalimaging.org/2025/challenges/).
  - ISBI 2024 has no vessel challenge — [ISBI 2024](https://biomedicalimaging.org/2024/?p=1938).
- **Grand Challenge API** (`/api/v1/challenges/?limit=300`; 264 challenges). Filtering titles and descriptions for vessel, artery, CTA, angio, carotid, vertebral, neck, cerebral, aorta, Willis, stenosis and aneurysm leaves only:
  - VESSEL12, CoronARe, ASOCA, CADA, the carotid vessel wall (2021/2022), KiPA22, Parse2022, ATM22, HECKTOR, HaN-Seg2023, SEG.A., TopCoW23/24, HNTSMRG24, AortaSeg24, TopBrain2025/2026, HANCOTHON25, HECKTOR26 and TopAneu-26 (plus retinal and VerSe spine entries).
  - None is a head-and-neck CTA vessel challenge — [Grand Challenge API](https://grand-challenge.org/api/v1/challenges/).
- **Codabench public competition API**: searches for "CTA", "vessel", "artery", "carotid", "vertebral", "head and neck", "cerebrovascular", "angiography", "aorta" and "stenosis" return only INSTED (2139) and a "Carotid Ultrasound Seg+Cls Challenge" (11798) among medical-vessel entries — [Codabench API](https://www.codabench.org/api/competitions/public/?search=artery).

### Inferences
- The Chinese groups that run cerebrovascular challenges (Tsinghua's Chen/Zhao group, the TopBrain Chinese co-organisers) publish intracranial TOF-MRA or braincase-cropped CTA. Head-and-neck CTA with neck vessels has not been released as a challenge by anyone, Chinese or otherwise, up to October 2026.

## Q4. Do the Chinese domestic competition platforms (Tianchi, AI Studio, DataFountain/CCF BDCI, iFLYTEK, Heywhale, CSIG, DiMTAIC) have a head-and-neck CTA or VA task?

### Takeaway
No. Tianchi was checked exhaustively: its public listing API returned all 577 competitions, and none is a CTA, vessel or neck-vessel contest. The other platforms were searched by keyword only. They turned up retinal-vessel, ultrasound-hemangioma, thyroid-ultrasound and non-imaging tasks, but nothing head-and-neck vascular.

### Cited Findings
- **Tianchi (天池).** The API `https://tianchi.aliyun.com/v3/proxy/competition/api/race/listBrief` answers without a login and reported `"total":577`. All 58 pages were downloaded. The medical-imaging contests by name are:
  - "天池医疗AI大赛[第一季]：肺部结节智能诊断" (lung nodules);
  - "'数字人体'视觉挑战赛——宫颈癌风险智能诊断" (cervical-cancer cytology);
  - "Spark'数字人体'AI挑战赛——脊柱疾病智能诊断大赛" (spine MRI);
  - "DiMTAIC 2024 数字医学技术及应用创新大赛-甲状腺B超静态及动态影像算法赛" (thyroid ultrasound);
  - the Asia-Pacific Academy of Ophthalmology (APAO) big-data contests (retina/OCT), the most vessel-related being "2023年亚太眼科学会大数据竞赛-使用人工智能自动生成眼底荧光血管摄影检查报告" (automatic reports from fundus fluorescein angiography);
  - "CSS体视学挑战赛：2026 CBCT金属伪影消除挑战赛" (CBCT metal-artifact removal).

  The rest are NLP or tabular. **No title contains 血管 (vessel), CTA, 颈 (neck), 脑血管 (cerebrovascular) or 动脉 (artery)** — [Tianchi competition API](https://tianchi.aliyun.com/v3/proxy/competition/api/race/listBrief?pageNum=1&pageSize=10).
- **Tianchi dataset search API**: `{"code":"TST-BIZ-ERROR","message":"api access denied!"}`, so Tianchi datasets could not be searched by script.
- **Baidu AI Studio (飞桨).** Site-restricted search returned GAMMA/STAGE/GAVE (ophthalmology), 3D-IRCADb, VerSe, ACDC, ARCADE (coronary X-ray) and MedMNIST. No head-and-neck CTA dataset or competition was found — e.g. [AI Studio ARCADE](https://aistudio.baidu.com/datasetdetail/240443) and [GAVE](https://aistudio.baidu.com/competition/detail/1315).
- **DataFountain (CCF BDCI)**: the only vessel contest found is "血管瘤超声图像分割" (hemangioma ultrasound segmentation) — [DataFountain 533](https://www.datafountain.cn/competitions/533).
- **iFLYTEK, Heywhale, Huawei Cloud**: site-restricted searches found no CTA or vessel task.
- **CSIG 图像图形技术挑战赛** (China Society of Image and Graphics challenge; five editions since 2020) — [challenge.csig.org.cn](http://challenge.csig.org.cn/). The pages I opened name tracks such as "短视频人脸解析竞赛" (short-video face parsing), "激光雷达里程计赛道" (LiDAR odometry) and an "AMD双赛道" (AMD twin tracks). No vessel track was found; not every edition's track list was read.
- **首届全国数字健康创新应用大赛** (first National Digital Health Innovation Application Competition, National Health Commission, 2023): a company product contest, with Shukun (数坤) winning awards. No public dataset — [Tencent News](https://news.qq.com/rain/a/20231116A07VUL00).
- **MICS (医学图像计算青年研讨会, Medical Imaging Computing Seminar) and ISICDM**: searches found only student video contests for MICS and nothing for ISICDM.

### Gaps
- AI Studio, DataFountain, Heywhale and iFLYTEK were searched only through the (US-based) search tool with site filters, not through their internal search, which needs login or is JavaScript-only.
- Domestic radiology-society AI contests (for example 中华医学会放射学分会, the Chinese Society of Radiology) may exist only as WeChat announcements, and WeChat search (Sogou) blocks bots.

## Q5. Is there any Chinese standard or institutional dataset that labels the whole VA on head-and-neck CTA?

### Takeaway
There is a national annotation standard that requires it, but no released dataset. The 2024 expert consensus, co-authored with the National Institutes for Food and Drug Control (中国食品药品检定研究院, NIFDC), defines head-and-neck CTA datasets as covering the aortic arch, brachiocephalic trunk, both subclavians, both CCAs and ICAs, the basilar, and **both VAs**, scanned from the vertex to the sternal angle. Such datasets are meant for training and independent testing of AI products in device registration, so any NIFDC test set would be institutional-only. I found no public NIFDC head-and-neck CTA database.

### Cited Findings
All quotes below are verbatim from the full text on brainmed.com, checked by grepping the fetched HTML — [brainmed 47109](http://www.brainmed.com/info/detail?id=47109).
- **Citation:** 中华医学会放射学分会医学影像人工智能工作组, 北京医学会放射学分会人工智能学组, 中国食品药品检定研究院. 头颈动脉CT血管成像数据标注与质量控制专家共识[J]. 中华放射学杂志, 2024, 58(2): 150-157. DOI: 10.3760/cma.j.cn112149-20230522-00358. Lead writer 卢洁 (Lu Jie, Xuanwu Hospital).
- **Annotation targets:**
  - "数据集的标注对象包括头颈动脉血管形态、骨结构、斑块成分、狭窄程度等信息" — "The annotation targets of the dataset include head-and-neck artery morphology, bone structures, plaque components and degree of stenosis."
  - "数据集图像特征中的血管包括主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、双侧椎动脉、双侧大脑前动脉、双侧大脑中动脉、双侧大脑后动脉、前交通动脉和后交通动脉。" — "The vessels among the dataset's image features include the aortic arch, brachiocephalic trunk, bilateral subclavian arteries, bilateral CCAs, bilateral ICAs, basilar artery, **bilateral vertebral arteries**, bilateral ACAs, MCAs and PCAs, and the anterior and posterior communicating arteries."
- **Scan range:** "每个病例的扫描范围从颅顶至胸骨角水平，采用螺旋扫描方式从足侧向头侧扫描。" — "Each case is scanned from the vertex to the level of the sternal angle, helically from feet to head." This range includes the VA origins.
- **Intracranial plaque rule:** "标注颅内动脉斑块时，仅对椎动脉颅内段、基底动脉、颈内动脉颅内段、大脑中动脉M1段、大脑前动脉A1段、大脑后动脉P1段的钙化斑块进行标注" — "When annotating intracranial plaques, only calcified plaques of the intracranial VA, basilar, intracranial ICA, MCA M1, ACA A1 and PCA P1 are annotated." The VA is thus split into intracranial and extracranial parts for plaque only; the vessel itself is annotated along its whole length.
- **Storage:** "必须以开放可读的格式（如DICOM）存储全部标注结果" — "All annotation results must be stored in an openly readable format (e.g. DICOM)."
- **Uses:** "数据集的用途包括AI产品的训练集、调优集、性能独立测试集、临床试验数据集、日常质控等。" — "Uses of the dataset include training, tuning and independent performance-test sets for AI products, clinical-trial datasets, routine quality control, etc." No dataset is released or linked.
- **NIFDC test databases**: searches found NIFDC standards for AI device testing (for example a stroke-CT test method) but no public head-and-neck CTA database. The NIFDC standards page failed TLS verification ("Hostname/IP does not match certificate's altnames: Host: app.nifdc.org.cn") — [NIFDC standard page](https://app.nifdc.org.cn/biaogzx/dataGkZqyj.do?formAction=viewZqyj&id=2c9048d88a743f96018a87ff45b429d7).

### Inferences
- The consensus shows that Chinese commercial head-and-neck CTA products (Shukun, United Imaging, Deepwise and others) are trained on full arch-to-vertex VA annotations of exactly the kind we need. Those labels stay with the companies, hospitals and NIFDC. The only route to them is collaboration or a data-use agreement with a hospital such as Xuanwu, whose radiology department leads both the consensus and Fu et al.

## Q6. Incidental Chinese leads outside challenges (for the repositories/papers researchers)

- **CADS (2025; 22,022 CTs, 167 structures):** Chinese dataset indexes list it as a head-and-neck resource, but it is not Chinese (UZH/TUM). Its only neck vessels are HaN-Seg-style "Carotid artery L/R" (`OAR_A_Carotid_L/R`); a code search for "vertebral" finds only vertebra code — [murong-xu/CADS labelmap](https://github.com/murong-xu/CADS/blob/main/resources/info/labelmap.md). No VA.
- **Zhang et al., npj Digit Med 2025, "Rapid vessel segmentation and reconstruction of head and neck angiograms from MR vessel wall images"** (Ren Ji Hospital SJTU + United Imaging) — [PMC12304216](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304216/):
  - 2,099 3D MR-VWI (black-blood) scans from four Chinese centres. Of 1,728 annotated sequences, the labels are "CCA, ICA, ECA, ACA, MCA, vertebral artery (VA), basilar artery (BA), and posterior cerebral artery (PCA)".
  - Data: "cannot be made publicly accessible due to privacy protection but are available from the corresponding author [Huilin Zhao] on reasonable request".
  - The VA extent (how far into the neck) is not stated. It is MR vessel-wall imaging, not CTA. This is a request-only lead.
- **BiSegMamba (arXiv 2605.30972)**, the source of the "Peking University Third Hospital 115 head-and-neck CTA" carotid dataset in the brief, **has been withdrawn on arXiv** ("This version of 2605.30972 has been withdrawn"; withdrawn by Bakht Zada). The v1 PDF still downloads, but its text could not be extracted here (CID fonts; no PDF tools on the host).
- **ModelScope (魔搭) dataset API** (`/api/v1/dolphin/datasets?Query=…`):
  - "CTA", "头颈" (head and neck), "TopCoW" and "HeadNeck" all return `TotalCount: 0`.
  - "椎动脉" and "颈动脉" (vertebral/carotid artery) match only a traditional-Chinese-medicine herbal-pillow text corpus ("本数据集为中医外用药枕经验词元库，主治各型颈椎病（颈型、神经根型、椎动脉型…）"; "a corpus of TCM herbal-pillow experience for all types of cervical spondylosis, including the vertebral-artery type").
- **Chinese dataset indexes** (linhandev/dataset, openmedlab/Awesome-Medical-Dataset with 1,449 files): no head-and-neck CTA vessel dataset.

## Access log: what could and could not be opened

| Target | Result |
|---|---|
| Zhihu p/29607061020 (WebFetch, curl, reader proxy, three Zhihu API endpoints, Wayback) | **Blocked**: 403, login wall "请您登录后查看更多专业优质内容", API code 10003 / 40362; no archive |
| Baidu web search (`/s?wd=`) | **Blocked**: redirected to "百度安全验证" CAPTCHA |
| Sogou web search | **Blocked**: redirected to `/antispider/` |
| Bing China (cn.bing.com) | Redirected to the bing.com homepage; no results |
| DuckDuckGo HTML (kl=cn-zh) | HTTP 202, no results |
| Tianchi competition API | **Opened**; all 577 competitions listed |
| Tianchi dataset search API | "api access denied!" |
| ModelScope dataset API | **Opened** |
| OpenDataLab search API | Returned empty placeholder objects; not usable |
| Tencent Cloud developer pages | **Opened** (articles and the author's 2024 list); user-article API routes 404 |
| brainmed.com consensus | **Opened** over http; the https fetch timed out |
| Europe PMC full text (Fu et al., Zhang et al.) | **Opened** |
| NIFDC standards page | TLS certificate mismatch |
| test.platform.shukun.net (CerebralDoc) | NXDOMAIN; not tried further |
| Codabench, Grand Challenge APIs, MICCAI/ISBI lists, Zenodo, CSIG site | **Opened** |
| arXiv 2605.30972 | Withdrawn; v1 PDF downloaded, text not extractable |

The WebSearch tool is US-based and returned mostly English results for Chinese queries. Site-restricted queries (zhihu.com, tianchi.aliyun.com, aistudio.baidu.com, csdn.net, cloud.tencent.com, mp.weixin.qq.com, datafountain.cn, heywhale.com and others) were used to compensate.
