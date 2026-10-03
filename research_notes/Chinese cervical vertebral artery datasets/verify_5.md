# Verify 5: "HeadNeckCTA2025——头颈CTA血管分割" (Zhihu p/29607061020) and the Fu et al. 18,766-CTA cohort

Adversarial check of the candidate's claim that it may provide public cervical vertebral artery (VA, 椎动脉) geometry. Checked 2026-10-01. This builds on `challenges.md` Q2, `community.md` Q1 and `papers.md`, which already cover this item. It repeats their checks only where an independent re-check was the point.

## Verdict

**Refuted. Not public, and no cervical VA label.**

- **It is not a dataset.** The Zhihu post is a blogger's tutorial write-up. Its "data download address" is the PubMed page of Fu et al. 2020.
- **The data is closed.** The underlying Fu et al. / Xuanwu / Shukun (数坤) cohort "cannot be made publicly accessible due to privacy protection".
- **There is no VA class at either level:**
  - Fu et al.'s ground truth is four masks: bone, aorta, carotid artery, intracranial artery.
  - The blogger's own experiment is a binary vessel/background segmentation.
- **The only public artefact is gone.** The paper's web demo showed 164 anonymised cases for viewing only, and its host no longer resolves.
- **A non-Chinese researcher has no route to the scans.**

## 1. The host page (Zhihu)

| Attempt (2026-10-01) | Result |
|---|---|
| WebFetch `https://zhuanlan.zhihu.com/p/29607061020` | HTTP 403 |
| curl, browser user agent | 403, `zh-zse-ck` anti-bot JS shell |
| r.jina.ai reader | Title 「安全验证 - 知乎」, body 「请您登录后查看更多专业优质内容。」 ("Security verification – Zhihu"; "Please log in to view more professional, high-quality content.") |
| Wayback `archive.org/wayback/available` | `"archived_snapshots": {}`: no snapshot |
| Wayback CDX | "Temporarily Offline" |
| archive.ph `/newest/` | 404: no snapshot |
| Bing / DuckDuckGo HTML | Bing returned unrelated results; DDG redirected (302) |

I did not log in, solve the challenge or spoof a search-engine crawler. **The post body was never rendered.** Everything below about its content comes from search-engine extracts, which are summaries rather than verbatim text.

### What the search-engine extracts attribute to the post
- **Dataset paragraph (= Fu et al.):** 「使用的数据集包括回顾性随机收集的18,766张原始造影剂增强头颈部CTA图像，这些图像来自中国五家三级医院，其中8593张来自首都医科大学宣武医院，4407张来自…北京友谊医院，2584张来自上海第六人民医院，2096张来自河北省人民医院，1086张来自山东省立医院。」
  - Translation: "18,766 retrospectively and randomly collected contrast head-and-neck CTAs from five Chinese tertiary hospitals: Xuanwu 8,593; Beijing Friendship 4,407; Shanghai Sixth 2,584; Hebei 2,096; Shandong 1,086."
- **"Download" line:** 「数据下载地址：https://pubmed.ncbi.nlm.nih.gov/32973154/」 ("data download address: [PubMed 32973154]"). That is the PubMed abstract page of Fu et al., not data.
- **New in this pass: the blogger's own training recipe.** It shows the experiment was a **binary** segmentation:
  - Extract: 「首先使用固定阈值和最大连通域提取方法来去除背景区域…将数据采样到固定spacing分辨率为0.4x0.4x0.4，并随机裁切出20个256x256x256大小的patch…搭建VNet3d网络，使用AdamW优化器，学习率为0.001，批量大小为1，训练100个epoch，损失函数采用二值化的Dice和CE。」
  - Translation: "remove background with a fixed threshold and largest connected component … resample to 0.4 mm isotropic, randomly crop 20 patches of 256³ … VNet3d, AdamW, lr 0.001, batch 1, 100 epochs, **loss = binary Dice + CE**."
  - So even the blogger's labels, wherever they came from, are a single vessel class with no VA.
  - Source: WebSearch for `"HeadNeckCTA2025" 数据下载 OR 数据集 OR 训练 OR Dice OR VNet3d`, result [Zhihu p/29607061020](https://zhuanlan.zhihu.com/p/29607061020).

### Trap: a VA vessel list that is NOT from this post
One WebSearch summary, for `"头颈CTA血管分割" 18766 数据集 下载 椎动脉`, put three things under a "HeadNeckCTA2025数据集" heading:
- a vessel list: 「主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、**双侧椎动脉**、双侧大脑前动脉…」 ("aortic arch, brachiocephalic trunk, bilateral subclavian, bilateral CCA, bilateral ICA, basilar, **bilateral VA**, bilateral ACA…");
- the sentence 「AI产品可基于这些数据实现和验证多种影像后处理功能…」 ("AI products can implement and validate many post-processing functions on these data…");
- the PubMed "download" line.

I fetched [brainmed.com/info/detail?id=47109](https://www.brainmed.com/info/detail?id=47109) and found the vessel list and the AI sentence there **verbatim**. That page is **头颈动脉CT血管成像数据标注与质量控制专家共识（2024）** (expert consensus on head-and-neck CTA annotation and quality control, 中华放射学杂志 2024;58(2)). Its scan range is 「从颅顶至胸骨角水平」 ("vertex to the sternal angle"), and it releases no dataset. The search tool blended the consensus with the Zhihu post, so that list is **not** evidence that HeadNeckCTA2025 or Fu et al. label the VA.

## 2. The underlying data: Fu et al., Nat Commun 2020 (PMC7518426)
Read on 2026-10-01 from the [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML).

- **Labels: four masks, no VA class.**
  - "prelabeled using ITKSNAP software with masks marking the areas of the bone, aorta, carotid artery, and intracranial artery. They were multiclass classification with four classes. We annotated the arteries including the aorta, subclavian artery, common carotid artery, vertebral artery, internal carotid artery, basilar artery, posterior cerebral artery, middle cerebral artery, and anterior cerebral artery … each CTA scan was annotated with four masks to obtain the reliable ground truth."
  - The VA is drawn, but it is merged into one of the four classes. The text never says which one, nor whether the cervical VA is split from the intracranial VA.
- **Coverage: from the origin, as imaged.**
  - "Each scan included the aorta, carotid artery, and intracranial artery with ~561–967 slices in the Z axis"; "The primary thickness ranged from 0.4 to 0.6 mm."
  - Failure-case captions show the VA origin is in the field of view: "Left vertebral artery arising directly from the aorta; the CPR and curved-MPR failed"; "Bone remains at the origin of the left vertebral artery"; "Vein artifact at the origin of the right vertebral artery".
  - The geometry would be there if the scans were obtainable. The labels would still need re-labelling to isolate the VA.
- **Availability:** "Currently, the source imaging data from five tertiary hospitals cannot be made publicly accessible due to privacy protection. The CerebralDoc is accessible from the webserver created for this study at http://test.platform.shukun.net/login. User name: test, password: 123456. Request for software access and the remaining clinical evaluation images generated by technologists can be submitted to F.F."
  - There is no Code availability section.
  - The on-request items are software access and technologist-made reconstructions (VR/MIP/CPR images), not source volumes or masks.
- **Supplements:** MOESM1–4 are the Peer Review File, Supplementary Information, Supplementary Software (`41467_2020_18606_MOESM3_ESM.docx`) and the Reporting Summary.
  - "Source Data" covers only Supplementary Table 2 and Figs. 3a, 7 and 8, i.e. numbers behind figures.
  - **No images or masks.**
- **New: what the Supplementary Software docx says.** I downloaded it from static-content.springer.com.
  - The demo was a web viewer: "click the 'Digital Brain' and 'CerebralDoc', so that you can see the demo cases processed by CerebralDoc. These demo data contain a total of 164 clinical routine cases of head and neck CTA scan examination. In order to ensure information security, we have anonymized all calculated cases."
  - The segmentation was displayed, not exported: "The system could automatedly mark the arteries segmentation results (red color) on the source image".
  - No download or export is mentioned anywhere. It was view-only.
- **Demo host is dead.** `nslookup test.platform.shukun.net` and `platform.shukun.net` both give "Non-existent domain". The company site `www.shukun.net` resolves (49.235.159.242) and returns 200, but it is a corporate site, not a data portal.
- **Follow-up paper:** Fu F et al., "Deep Learning for Head and Neck CT Angiography: Stenosis and Plaque Classification", Radiology 2023 (PMID 36880944). Europe PMC reports `isOpenAccess: N`, `hasData: N`, `hasSuppl: N`. No data route was found.

## 3. Mirrors and possible download channels

| Channel | Result |
|---|---|
| GitHub search API `HeadNeckCTA` | `total_count: 0` |
| GitHub user `junqiangchen` (368 repositories; suspected author per `community.md`) | Only head/vessel-named repositories are `HC18-Automated-measurement-of-fetal-head-circumference` and a `SimVascular` fork. No HeadNeckCTA repository and no data. |
| CSDN search API, blog scope: "HeadNeckCTA2025" | 0 results |
| CSDN search: "头颈CTA血管分割" | 12 generic results, no HeadNeckCTA2025 post, no dataset |
| CSDN user article-list API (`qq_42886742`) | Blocked: HTTP 521 from PowerShell; curl status 000 |
| Sogou WeChat search ("HeadNeckCTA2025", "头颈CTA血管分割") | 302 to `antispider` (blocked this pass; `community.md` read the sibling WeChat article earlier) |
| WebSearch: 宣武医院 / 数坤 头颈CTA 数据集 公开 开放 下载 | No release found; only Xuanwu/Shukun news and NMPA-certificate press |
| WebSearch: Shukun head-and-neck CTA dataset public release | Nothing; returns HNA / HNCTA (private) |

## 4. Can a non-Chinese researcher get it?
No route was found.
- The paper rules out public release.
- The only request channel is first author Fan Fu (F.F.). The corresponding author per the XML is Jie Lu (卢洁), imaginglu@hotmail.com. That channel covers software and evaluation images, not the 18,766 source scans.
- Inference, not verified here: sending identifiable Chinese hospital imaging abroad would also face China's personal-information and data-export rules, which makes a data-use agreement with a foreign group unlikely without a formal collaboration.

## 5. Corrections to the candidate record
- The "va_coverage" field is right that there is no separate VA class. The blogger's own experiment is also binary (new evidence above).
- The "access" field is right. Two additions:
  - the demo was a view-only web viewer of 164 cases;
  - `platform.shukun.net` is also NXDOMAIN.
- Do not reuse the search-engine vessel list that names 「双侧椎动脉」 under "HeadNeckCTA2025": it comes from the 2024 annotation consensus, not from this post or this cohort.
- Still unseen: the Zhihu post body and its comment section, behind a login wall. A cloud-drive (网盘) link in a comment cannot be ruled out. Data passed around that way would not be lawfully redistributable anyway.
