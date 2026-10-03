# Verify 6: 2024 head-and-neck CTA annotation consensus (an annotation standard, not a dataset)

**Candidate:** 「头颈动脉CT血管成像数据标注与质量控制专家共识（2024）」 (Expert consensus on data annotation and quality control for head-and-neck arterial CTA, 2024)
**Citation:** 中华放射学杂志 (Chinese Journal of Radiology) 2024;58(2):150-157, DOI 10.3760/cma.j.cn112149-20230522-00358. The authors are the 中华医学会放射学分会医学影像人工智能工作组 (CMA radiology AI working group), the 北京医学会放射学分会人工智能学组 (Beijing Medical Association radiology AI group) and the 中国食品药品检定研究院 (NIFDC).
**Verified on:** 2026-10-01

## What I opened

| URL | Result |
|---|---|
| http://www.brainmed.com/info/detail?id=47109 (脑医汇 repost) | **Opened.** `curl` returned the full article text (195 KB of HTML, no login). Its byline: 刘士远, 陈敏, 金征宇, 卢洁, 周俊林 (corresponding authors), with 李小虎, 萧毅, 郑敏文, 张苗 and 王浩 (NIFDC 医疗器械检定所). It was reposted on 2024-08-09. |
| https://doi.org/10.3760/cma.j.cn112149-20230522-00358 | It redirects to yiigle.com (中华医学期刊全文数据库). The page is rendered by JavaScript, and WebFetch got only "loading", so I could not open the publisher page. |
| Searches for a released 头颈动脉CTA 标准数据库 (CMA, NIFDC) | Nothing found. No release announcement, download page or access procedure turned up. |

## What the text says (verbatim, from the brainmed full text)

1. 「数据集图像特征中的血管包括主动脉弓、头臂干、双侧锁骨下动脉、双侧颈总动脉、双侧颈内动脉、基底动脉、双侧椎动脉、双侧大脑前动脉、双侧大脑中动脉、双侧大脑后动脉、前交通动脉和后交通动脉。」 ("The vessels in the dataset's image features include the aortic arch, brachiocephalic trunk, both subclavians, both CCAs, both ICAs, the basilar, **both vertebral arteries**, the ACAs, MCAs and PCAs, the AComm and the PComms.") **Confirmed.**
2. 「每个病例的扫描范围从颅顶至胸骨角水平，采用螺旋扫描方式从足侧向头侧扫描。」 ("Each case is scanned from the vertex to the level of the sternal angle, helically from feet to head.") **Confirmed.** This range covers the VA origins at the subclavians.
3. 「必须以开放可读的格式（如DICOM）存储全部标注结果。」 ("All annotation results must be stored in an openly readable format such as DICOM.") **Confirmed.** It means a file format that can be read, **not** open access.
4. 「数据集的用途包括AI产品的训练集、调优集、性能独立测试集、临床试验数据集、日常质控等。」 ("Uses of the dataset include training, tuning and independent performance-test sets for AI products, clinical-trial datasets, routine QC.") **Confirmed.**
5. 「本部分的主要目标为建立配套的头颈动脉CTA影像标准数据库，提高AI的算法泛化能力及效率」 ("The main aim of this part is to build a matching head-and-neck CTA **standard database** to improve AI generalisation and efficiency.") This is a stated **goal**. The text gives no case count, location or access route.
6. How vessels are labelled: 「应标注所有对比剂充盈区域，包括动脉瘤、侧支循环血管等，标注边缘贴近血管内壁，标注范围不能超过血管管腔」 ("Label every contrast-filled region, including aneurysms and collaterals, with the label edge on the inner vessel wall and never beyond the lumen.") Also: 「将头颈动脉分支血管按照头颈动脉解剖走行及结构进行逐层标注，不能出现漏标、错标、多标的情况」 ("Label the head-and-neck branches slice by slice along their anatomical course, with no missed, wrong or extra labels.") These describe lumen masks for each branch.
7. The only VA segment named is intracranial, and only for plaque: 「标注颅内动脉斑块时，仅对椎动脉颅内段、基底动脉…的钙化斑块进行标注」 ("For intracranial plaque, only calcified plaque in the **intracranial VA segment**, the basilar… is labelled.") The text has **no V1-V4 nomenclature** and no statement that the VA label starts at the subclavian origin.
8. Data sharing: the text never uses 公开, 共享 or 下载 about data. The only 下载 hits are the brainmed site's own "PDF下载" and "App 下载" buttons. Privacy: 「从头颈动脉CTA图像中收集的数据必须是临床脱敏数据，由医院伦理委员会批准」 ("Data must be de-identified clinical data approved by a hospital ethics committee.")

## Refutation result

- **Is it a dataset?** No. It is a standard for how to build and label datasets. It holds **0 cases**, and the text has no file list, label dictionary or download link.
- **Is the cervical VA labelled?** No data exists to be labelled. The standard does list 双侧椎动脉 among the vessels, on scans from the vertex to the sternal angle, with lumen masks for each branch. So a **compliant** private dataset would probably contain the whole VA from the origin. But the text never says that explicitly, and it names no VA segments other than the intracranial one.
- **Is the "standard database" public?** No evidence of it. The text states the database only as an aim. NIFDC databases of this kind are used to test AI medical devices for regulatory approval. I found no release, application process or terms, so access for a non-Chinese researcher is unknown. It is effectively institutional or regulator-held.
- **Is the article text itself open?** The brainmed repost reads openly. The publisher copy on yiigle could not be opened, because it is rendered by JavaScript and probably paywalled.

**Verdict:** the candidate's quotes are accurate, but it is **not a source of VA geometry**. It is useful only as context: it is the Chinese labelling convention that explains why the private corpora (HNCTA, HNA, Fu et al. 2020) label the VA together with the arch and subclavians. public = false; cervical_va_labelled = false.
