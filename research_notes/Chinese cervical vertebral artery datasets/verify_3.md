# Verify 3: 2024 head-and-neck CTA annotation consensus (头颈动脉CTA数据标注与质量控制专家共识)

Adversarial check, 2026-10-01, of the claim that this candidate "may provide public cervical vertebral-artery geometry".

## Verdict

**Refuted. This is an annotation standard, not a dataset.** It specifies that head-and-neck CTA datasets built for AI products should label both VAs, along with the subclavians and the arch, on scans from the vertex to the sternal angle. But it releases no images, no labels and no case count, and gives no download, application or contact route for data. The "matching standard database" it announces could not be found anywhere public.

- public = **false**
- cervical_va_labelled = **false**: there is no data. The standard only *specifies* VA labels.

## Identity

- 头颈动脉CT血管成像数据标注与质量控制专家共识 / "Expert consensus on the rule and quality control of CT angiography of head and carotid artery" (the PDF's own English title).
- 中华放射学杂志 (Chin J Radiol) 2024;58(2):150-157, DOI 10.3760/cma.j.cn112149-20230522-00358. Received 2023-05-22.
- Issued by 中华医学会放射学分会医学影像人工智能工作组, 北京医学会放射学分会人工智能学组 and 中国食品药品检定研究院 (NIFDC). Writers include 卢洁 (Xuanwu), 李小虎, 萧毅, 郑敏文, 张苗, and 王浩 (NIFDC device-testing institute).
- The PDF lists five corresponding authors: 刘士远, 陈敏, 金征宇, 卢洁 and 周俊林. The brainmed HTML shows only 卢洁.
- Funding: NSFC 82130058; Beijing NSF Z190014; Beijing S&T Z201100005620009.

## What I opened

| Target | Result |
|---|---|
| `http://www.brainmed.com/info/detail?id=47109` | **Opened** with curl; 195 KB of HTML containing the full text |
| PDF linked from that page ("PDF下载"): `https://medtion-image.medtion.com/uploads/1/file/public/202408/20240809151108_nmh9bon0uc.pdf` | **Downloaded**: 1.6 MB, journal pages 150-157; text extracted with pdftotext |
| DOI → `yiigle.com/LinkIn.do?...` (the publisher, 中华医学期刊全文数据库) | Page is JavaScript-only; WebFetch got only "loading" |
| WebSearch for a NIFDC 头颈动脉CTA 标准数据库, its access terms, or a dataset built to this consensus (4 queries, in Chinese) | Nothing. Results were only YY/T 1833 standard commentaries, the Tiantan aneurysm database (head-only, unrelated) and Fu et al. 2020 |

The PDF contains no 补充材料 (supplement), 数据可用性 (data-availability) or OSID statement. It ends with 利益冲突 (conflicts of interest) and the references.

## Evidence (verbatim from the PDF/HTML, with translation)

1. **Vessel list (the snippet quote is verified verbatim).** "数据集图像特征中的血管包括主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、双侧椎动脉、双侧大脑前动脉、双侧大脑中动脉、双侧大脑后动脉、前交通动脉和后交通动脉。"
   - "The vessels among the dataset image features include the aortic arch, brachiocephalic trunk, bilateral subclavian, bilateral common carotid and bilateral internal carotid arteries, the basilar, **bilateral vertebral arteries**, and the bilateral ACA, MCA and PCA plus the AComm and PComm."
2. **Scan range.** "每个病例的扫描范围从颅顶至胸骨角水平，采用螺旋扫描方式从足侧向头侧扫描。"
   - "Each case is scanned from the vertex to the level of the sternal angle, helically from feet to head."
   - This covers the VA origins, but the text never says "label the VA from its origin". "From the origin" is only implied by the rule to label every contrast-filled region: "应标注所有对比剂充盈区域…标注范围不能超过血管管腔（图2~5）" ("all contrast-filled regions must be labelled… not beyond the lumen (Figs 2-5)").
3. **The VA is used as the example lumen label.** "图2，3分别为椎动脉拉直图和血管管腔标注图" ("Figs 2 and 3 are a straightened VA and its lumen label").
4. **Cervical course is in scope.** The bone rule says "尤其要注意颈椎椎间孔和颅底部分有血管穿行的孔、裂区域" ("pay particular attention to the cervical-spine foramina and skull-base foramina/fissures through which vessels pass"). This is the V2/V3 course.
5. **The illustrated label classes are coarse.** "用红色标记颅内动脉（图14）、蓝色标记颈动脉（图15）、黄色标记主动脉（图16）、绿色标记骨" ("red marks intracranial arteries, blue the carotid/neck arteries, yellow the aorta, green bone").
   - No VA-specific class is shown. The standard does not fix a per-vessel label dictionary.
6. **Label format and tools.**
   - "必须以开放可读的格式（如DICOM）存储全部标注结果" ("all annotation results must be stored in an openly readable format, e.g. DICOM").
   - "现有的开源软件标记工具包括3Dslicer、MITK和ITK‑SNAP" ("existing open-source labelling tools include 3D Slicer, MITK and ITK-SNAP"); ITK-SNAP is the recommended tool.
7. **Purpose: regulatory, not research release.**
   - "数据集的用途包括AI产品的训练集、调优集、性能独立测试集、临床试验数据集、日常质控等" ("dataset uses include training, tuning and independent performance-test sets for AI products, clinical-trial data sets and routine QC").
   - "本部分的主要目标为建立配套的头颈动脉CTA影像标准数据库" ("the main goal of this part is to build a matching standard head-and-neck CTA image database").
8. **No data offer.** I searched the HTML and the PDF for 公开, 共享, 下载, 申请, 获取, 补充, 数据可用 and "data availab". The only hits are unrelated: "获取截面图像" (obtain cross-sections) and the site's own "PDF下载" link. No case count appears anywhere.

## Points where the searcher's entry is right or wrong

- **Right:** it is "not a dataset"; the Sogou snippet is genuine; the format is DICOM; the tools are 3D Slicer, MITK and ITK-SNAP.
- **Overstated:** "covers bilateral VA" is true of the vessel list. But the standard does not explicitly require labelling from the origin, and its figure label scheme lumps vessels into intracranial, neck, aorta and bone.
- **Not supported:** that the candidate "may provide public cervical VA geometry". It provides none.

## Access for a non-Chinese researcher

- **The text:** the document itself is freely readable (brainmed HTML and medtion PDF, no login).
- **The data:** there is nothing to obtain. The NIFDC "standard database" the consensus announces has no public download, application or data-use agreement that I could find. An NIFDC test bank for device registration would be institutional-only; that last point is an inference from absence.
- **The only practical route:** a collaboration with the authoring hospitals, for example Xuanwu (卢洁, imaginglu@hotmail.com). That route is not an advertised data offer.
