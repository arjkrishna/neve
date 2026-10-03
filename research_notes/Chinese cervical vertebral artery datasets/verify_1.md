# Verification 1: Tiantan "心、脑、外周血管疾病医学影像数据集" (NBSDC)

Adversarial check of the claim in `platforms.md` §1.2, run on 2026-10-01 from the host APIs (not from the searcher's saved files).

**Verdict: the claim holds.** The record is real and its metadata is public, but:

- It contains **no vessel labels of any kind**, so no cervical VA labels.
- The cerebrovascular part is **TOF-MRA**, not CTA.
- The data files are **application-only**.
- Registration effectively **requires a Chinese identity** (a Chinese-character real name and a mainland mobile number), and the application needs an **official institutional seal (公章)**.

public = false (not obtainable by us in practice); cervical_va_labelled = false.

## What I opened

| URL / endpoint | Result |
|---|---|
| `https://www.nbsdc.cn/general/dataLinks/16666.11.nbsdc.UpfJ6qMG` | 200; a JS-rendered shell. The CSTR resolver `cstr.cn/16666.11.nbsdc.UpfJ6qMG` 302-redirects here. |
| `/basic/modules/dataset/dataSetInfo.js` | Gives the API endpoints used below and the access logic. |
| `POST /api/general/searchDataDetail` `id=666072b5195d266d328f4fa4` | Full metadata (below). |
| `GET /api/general/v1/getFileTree?dataId=…&pid=0,1,11,12,13` | Complete file list (below). |
| `GET /api/general/v1/checkAndDownloadFile` for `数据说明文件.docx` (id `66508454c882f13b9722a9f8`) | 200, 20,056 bytes; downloaded anonymously and read. |
| `checkAndDownloadFile` for `b.脑血管医学影像.zip` (id `66508454c882f13b9722a9ef`), 1 KB range request | **403**: 「目前只支持部分数据集的说明文件下载，数据文件请通过数据申请获取。」 ("Currently only the description files of some datasets can be downloaded; please obtain data files by submitting a data application.") |
| `https://api.escience.org.cn/metadata/metadata/search/detail?cstrId=2ce818534b401c7fa2f2b28d816e5c73:CSTR:16666.11.NBSDC.UPFJ6QMG` | Full CSTR record, including `applicationProcess`. A bare `cstrId=CSTR:…` returns `data:null`; the org-id prefix is required. |
| `https://www.nbsdc.cn/nbs-auth/regPage` + `/nbs-auth/js/nbs/registerUs.js` | Registration form and its validation rules. |

## Metadata (searchDataDetail / escience)

- Title: 心、脑、外周血管疾病医学影像数据集 / "Medical imaging dataset for heart, brain, and peripheral vascular diseases"
- CSTR: 16666.11.nbsdc.UpfJ6qMG; producer 首都医科大学附属北京天坛医院 (Beijing Tiantan Hospital); author 高峰 (Gao Feng); generated 2024-06-04.
- Project: 2018AAA0102600, 「基于异构融合计算模型与架构的心脑血管诊疗人机协同介入技术智能平台」 ("an intelligent human–machine collaborative interventional platform for cardio-cerebrovascular diagnosis and treatment"). The project has 11 NBSDC releases. The escience search for 人机协同介入 ("human–machine collaborative intervention") shows that the others are test-result files and a knowledge graph, none of them geometry.
- Description: 「……记录了以往血管类疾病患者以及项目执行期间新入院患者的多模态影像信息，包括CTA、DSA、MRA和ASL等特殊模态数据，数据总量为约230GB。」 ("records multimodal imaging of past vascular-disease patients and patients newly admitted during the project, including CTA, DSA, MRA, ASL and other special modalities, about 230 GB in total.") **This conflicts with the files.** There are 207.09 GB in 40 files, and the description docx names only CTA, TOF-MRA and CTA.
- `dataStrategyMap`: shareRange 完全共享 ("fully shared"), strategyType 1, agreement 1. On escience: sharePathway 线上共享 ("shared online").
- `downloadCount: 0`, `viewingCount: 1467`. NBSDC records no downloads at all.
- Licence: the page JS shows a "cc-by.png" badge for strategyType 1, but the badge links to CC0 (`publicdomain/zero/1.0`). The site terms say datasets under CC BY 4.0 may not be used commercially (「商业目的不被允许」, "commercial purposes are not permitted"). **The licence is ambiguous**; no dataset-specific licence is stated.

## File tree (complete)

```
2018AAA0102601-001 心、脑、外周血管疾病医学影像数据集/
  a.心血管疾病医学影像数据集/   13 × 心血管疾病医学影像N.dcm.zip    (2.5–13.1 GB each)
  b.脑血管医学影像数据集/       b.脑血管医学影像.z01…z18 (18 × 268,435,456 B) + .zip (71,549,262 B)  ≈ 4.9 GB total
  c.外周血管医学影像数据集/     7 × 外周血管医学影像数据集N.dcm.zip  (6.8–11.8 GB each)
  数据说明文件.docx             20,056 B
```

There is no `.nii`, `.nrrd`, `.seg`, `.stl`, `.vtp`, `.json` or mask file of any kind. The only formats are `.dcm.zip`, split-zip parts and one `.docx`.

## Description docx (read in full)

- 「本数据集包含以DICOM形式存储的原始医学影像……数据均已通过匿名化处理，不包含患者基本属性信息，仅保留与成像参数相关的信息。」 ("This dataset contains raw medical images stored as DICOM … all anonymised, with no basic patient attributes, keeping only imaging-parameter information.")
- 「影像关键信息为成像灰度和成像参数，可支持图像分割、病灶识别、疾病分级等数字模型的构建与验证。」 ("The key information is image grey levels and imaging parameters; it can *support building and validating* segmentation, lesion-detection and disease-grading models.") This supports building segmentation models; the dataset itself supplies no segmentations.
- 「心、脑、外周血管影像均采集于项目执行周期内（2019.12-2023.12），其中：心血管影像模态为CTA……最小空间分辨率为0.5mm；**脑血管影像模态为TOF-MRA**，扫描最小空间分辨率为0.8mm；外周血管影像模态为CTA，扫描最小空间分辨率为0.5mm。」 ("All acquired during the project, Dec 2019 to Dec 2023. Cardiac: CTA, 0.5 mm. **Cerebrovascular: TOF-MRA**, 0.8 mm. Peripheral: CTA, 0.5 mm.")
- 「每个文件的命名包含序列编号和患者化名」 ("each file name contains the series number and a patient pseudonym"). The case count is **not stated** anywhere.
- 「本数据集完全共享。」 ("This dataset is fully shared.")
- Viewing software: 「Mimics和Radiant」 ("Mimics and RadiAnt").

### Anatomical coverage: what cannot be ruled out, and why it does not matter

- Neither the docx nor the metadata gives a scan range.
- A 0.8 mm TOF-MRA is typically an intracranial circle-of-Willis acquisition, which shows at most V4 of the vertebral artery.
- In Chinese clinical usage, 外周血管 ("peripheral vessels") *can* include the carotid, vertebral and subclavian arteries. The peripheral CTA might therefore include some neck CTAs, but this is unverified; the file names reveal nothing.
- Either way the images are **raw and unlabelled**. At best this set could be another source of input images for Arterial, which TotalSegmentator's ~123 CC BY 4.0 neck CTAs already provide without any application.

## Access, and whether a non-Chinese researcher can get it

- escience `applicationProcess` (verbatim): 「注册用户登录国家基础学科公共科学数据中心官网（www.nbsdc.cn），浏览并确定所需数据资源。完成在线数据申请表填写后，需下载该申请表的电子版，打印并由负责人签字，同时加盖公章后重新上传。数据申请经国家基础学科公共科学数据中心、数据生产方审批通过后，用户将获得数据下载链接。」 ("Registered users log in to nbsdc.cn and choose the data. After filling in the online application form, download it, print it, have the responsible person sign it and **affix the official seal**, then re-upload it. Once NBSDC **and the data producer** approve, the user receives a download link.")
- The page JS confirms the flow. The page offers a 「加入购物车」 ("add to cart") button that leads to 「工作台-个人中心-购物车」 ("workbench → personal centre → cart") to apply. A file click without login shows 「暂无权限，请先注册或登录」 ("no permission; please register or log in first").
- **Registration form (`registerUs.js`), new evidence not in `platforms.md`:**
  - Real name must match `/^[一-龥]+$/`: 「需填写中文真实姓名，不可包含英文或数字」 ("must be your real name in Chinese characters, with no English letters or digits").
  - Phone must match the mainland China mobile pattern `^((13[0-9])|(14[5,7])|(15[0-3,5-9])|(17[0,3,5-9])|(18[0-9])|166|193|198|199|191|(147))\d{8}$`. The SMS-code step is currently commented out.
  - Organisation and address must contain Chinese characters: 「请用中文填写真实单位信息」 ("enter your real organisation in Chinese").
  - The terms require registration information to be 「真实而准确」 ("true and accurate"), and the site 「有权自行决定拒绝或允许为任何个人或实体提供服务」 ("may at its discretion refuse or allow service to any person or entity").
- **Conclusion on access:** a foreign researcher cannot register truthfully without a Chinese name and a mainland mobile number. A US institution has no 公章, so the application would also need producer approval of a non-standard form. A Chinese collaborator at a Chinese institution could apply. NBSDC's own counter shows 0 downloads.

## Bottom line

| field | value |
|---|---|
| downloadable? | **No.** The description docx is open; the data files return 403 and need an application. |
| cervical VA labelled? | **No.** There are no labels of any kind; the brain part is TOF-MRA (likely intracranial), and the peripheral CTA scan range is unknown. |
| access | Registration (Chinese name, mainland mobile number, Chinese organisation and address) plus an application form signed and sealed with 公章, approved by NBSDC and Tiantan. |
| licence | Ambiguous: a CC-BY badge linked to CC0; the site terms forbid commercial use. |
| useful for VA geometry graft? | No. At most raw images, which TotalSegmentator already covers openly. |
