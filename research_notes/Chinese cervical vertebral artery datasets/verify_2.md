# Verify 2: NCMI "医学图像标准测试数据库-脑卒中影像与标注数据"

Date: 2026-10-01. Role: adversarial verifier. The searcher's claim under test: this NCMI/PHDA record may supply public cervical vertebral-artery (VA) geometry.

**Verdict: refuted. No VA labels. Not public: conditional, offline sharing through a real-name PHDA application.** It agrees with the earlier entry in `platforms.md` §1.3 and adds the full official metadata record, the PHDA account and sharing rules, and the reasons the NCMI page itself could not be opened.

## Identity

- 中文名: 医学图像标准测试数据库-脑卒中影像与标注数据
- English title (official `titleEn`): "Medical Image Standard Test Database - Stroke Imaging and Labeling Data"
- Identifier: CSTR:17970.12.A00AL.202209.372.V1.0
- Host: 国家人口健康科学数据中心 (NCMI / National Population Health Data Center); repository: 人口健康科学数据仓储 (PHDA)
- Landing page: <https://www.ncmi.cn/phda/dataDetails.html?type=project_data&id=CSTR:17970.12.A00AL.202209.372.V1.0>
- `generateDate` 2019-12-01; record created 2024-10-24; 15 views; 0 comments.
- Subject: 临床医学 (clinical medicine). Theme: 基本医疗与保健-医疗影像 (basic medical care: medical imaging).

## What I could and could not open

| Route | Result |
|---|---|
| WebFetch of the NCMI landing page | **HTTP 403** |
| curl with a browser User-Agent and Referer | HTTP 200, but every detail URL returns the same 46,955-byte shell, whatever the id: a loading spinner plus a JavaScript cookie challenge (`password_sjmm` cookie → `/phda/dataPasswordJy.do` → reload). **No dataset content is in the shell.** I did not get past the challenge (the attempt was blocked by policy), so **I never saw NCMI's own rendering: no file list, sample image, or data dictionary**. A normal browser passes the challenge automatically. |
| `cstr.cn/17970.12.A00AL.202209.372.V1.0` (CSTR resolver) | Page loads but contains `errCode = 404` (identifier not resolved) |
| escience public search `api.escience.org.cn/metadata/global-search/open/page?keyword=脑卒中影像与标注数据` | 200; the record is found |
| **escience public detail `api.escience.org.cn/metadata/metadata/search/detail?cstrId=774e74221ac511e980780242ac120006:CSTR:17970.12.A00AL.202209.372.V1.0`** (the org-id prefix is required; the bare CSTR returns `data:null`) | **200: the full official metadata record**, quoted below. Its `link` field points back to the NCMI landing page above. |
| NCMI `/phda/support.html` (help and policy page) | 200; the sharing and real-name-account rules quoted below |
| NCMI `/phda/register.html` | 200; account-type descriptions only; the form fields are loaded by JS and were not seen |

## Evidence (official metadata record, escience detail API)

- `descr`: 「本数据集共包含头部CT脑出血、头部CT脑缺血、CTA头颈动脉瘤三个子数据集，每个子数据集对应一个记录数据标注情况的Excel文件。数据集中的703例头部CT脑出血数据来源于吉林大学第一医院和四川大学华西医院，864例头部CT脑缺血数据来源于四川大学华西医院，433例CTA头颈动脉瘤数据来源于中国人民解放军总医院。每例影像数据包含患者的原始CT影像和医生勾画的感兴趣区域。每个子数据集对应的Excel文件记录数据的影像编号、数据来源、和数据标注内容。」
  - Translation: "The dataset has three sub-datasets: head-CT intracerebral haemorrhage, head-CT cerebral ischaemia, and CTA head-and-neck aneurysm. Each sub-dataset has an Excel file recording how the data were annotated. The 703 haemorrhage CTs come from the First Hospital of Jilin University and West China Hospital of Sichuan University; the 864 ischaemia CTs from West China Hospital; **the 433 CTA head-and-neck aneurysm cases from the Chinese PLA General Hospital. Each case contains the patient's raw CT images and a physician-drawn region of interest.** Each sub-dataset's Excel file records the image ID, data source, and annotation content."
- `keywords`: 「脑出血；脑缺血；头颈动脉瘤」 ("intracerebral haemorrhage; cerebral ischaemia; head-and-neck aneurysm"). There is no vessel, artery-segmentation, 椎动脉 or 血管分割 keyword.
- `shareScope`: **「有条件共享」** ("conditional sharing").
- `sharePathway`: **「线下共享」** ("offline sharing").
- `applicationProcess`: 「按照人口健康科学数据仓储（PHDA）服务信息中的申请流程申请使用」 ("apply following the procedure in the PHDA service information").
- Service contact: 国家人口健康科学数据中心, 北京市东城区东单三条九号, rkjkpt@126.com, (010)65105883.
- No licence field, no file list, no data volume, no format field. `extendFields` is empty.

## Evidence (PHDA policy page, ncmi.cn/phda/support.html)

- 「个人注册账号：面向个人用户，用于个人用户进行数据汇交和使用国家人口健康科学数据中心的数据服务，该账号需要实名注册」
  - Translation: "A personal account is for individuals to deposit data and **use the centre's data services; it requires real-name registration**."
- 「对于已经注册的全部数据……依据数据共享层级向社会公众提供共享服务，共享方式包括全社会开放共享、协议开放共享、领域共享等多种方式。」
  - Translation: "All registered data are shared according to their sharing level: open to all, open under agreement, domain-restricted sharing, and so on."
- 「政府部门、高校和科研院所等用户使用汇交数据进行政策研究、教学、科研等社会公益性活动时，可无偿使用汇交数据；利用汇交数据进行商业、经济等盈利活动的，应缴纳相应的费用」
  - Translation: "Government, universities and research institutes may use the data free for public-interest research; **commercial use requires a fee**."
- The privacy section says a data-service application collects 「姓名、工作单位、通讯地址、邮编、邮箱、联系电话」 ("name, employer, postal address, postcode, email, phone").
- The login form accepts 「手机号/邮箱/用户名」 ("phone / email / username") plus an SMS **or email** code, so email login exists. Whether real-name verification accepts a foreign passport is **not stated** anywhere I could read.

## Assessment against the goal (cervical VA from the subclavian origin)

1. **VA labels: none.** The only annotation is "医生勾画的感兴趣区域" ("physician-drawn ROI") per case, in a sub-dataset named for aneurysms. Nothing anywhere mentions vessel lumen masks, centrelines, artery classes, or 椎动脉. The Excel index records "annotation content", meaning the aneurysm description, not a label map. **cervical_va_labelled = false.**
2. **Field of view: unverified.** 「头颈动脉瘤」 / 「CTA头颈」 suggests the Chinese 头颈CTA protocol, which normally runs from the aortic arch to the vertex. But that is a naming inference: no slice range, z-extent, or sample image could be seen. Even if the full neck is imaged, these would be **unlabelled** scans, only Arterial input, like TotalSegmentator's 123 CC BY 4.0 CTAs, which are already in hand and need no application.
3. **Access: not public.**
   - Conditional, offline sharing. Not downloadable.
   - Real-name PHDA account, with an application that NCMI and presumably the data producer must approve.
   - Fee for commercial use.
   - No licence stated.
   - It is a 「标准测试数据库」 ("standard test database"). Sets of this kind in China serve as held-out benchmarks for evaluating AI medical devices, which argues against wide release. This is an inference from the name; no page I read says so.
   - Whether a non-Chinese researcher can get it is undocumented. Expect a Chinese real-name identity or a Chinese partner institution, plus personal-information export review for patient CTs.
4. **Distinction:** this is a **catalogued, application-only dataset** ("有条件共享 / 线下共享"). It is neither open download nor merely "available on request" in a paper. But the labels it would deliver are irrelevant to VA geometry.

## Searches run (no further provenance found)

- WebSearch: 「"医学图像标准测试数据库" 脑卒中 头颈动脉瘤 CTA」; 「中检院 人工智能医疗器械 标准数据库 头颈动脉瘤 CTA 解放军总医院 433例」; 「标准测试数据库 脑出血 脑缺血 头颈动脉瘤 中国食品药品检定研究院 人工智能 测评」; 「"433例" 动脉瘤 CTA 解放军总医院」; `"17970.12.A00AL"`; and an English title search.
  - None of them found a paper, a producer page, or a data descriptor for this database.
  - The producing project code (A00AL) and the lead institution remain unidentified.
- escience keyword 「医学图像标准测试数据库」 returns only this one record (no sibling 'standard test database' records). 「头颈动脉瘤」 also returns only this record.

## Bottom line

Discard it as a VA-geometry source. If unlabelled arch-to-vertex CTAs were ever needed beyond TotalSegmentator, this would be a slow, uncertain fallback: a Chinese real-name application for 433 aneurysm CTAs whose field of view is unverified.
