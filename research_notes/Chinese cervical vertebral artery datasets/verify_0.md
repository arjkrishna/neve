# Verify 0: Fu et al. 2020 / Xuanwu Hospital + Shukun "CerebralDoc" head-and-neck CTA cohort

Adversarial check of the claim that this cohort may provide public cervical vertebral artery (VA) geometry. Checked 2026-10-01.

**Verdict: not public, and no separate VA label.** The VA is inside the annotation from its subclavian origin, but it is merged into a four-class scheme (bone / aorta / "carotid artery" / intracranial artery). The images and masks are withheld "due to privacy protection". The only things offered on request are software access and "clinical evaluation images", which are 2D reconstructions. The view-only demo server no longer exists. The searcher was right about access. On labels the searcher was half right: the VA lumen is annotated from the arch, but it has no VA class of its own.

## Sources opened myself

| Source | Result |
|---|---|
| Europe PMC full-text XML, PMC7518426 (`ebi.ac.uk/europepmc/webservices/rest/PMC7518426/fullTextXML`) | Opened, 120 kB; read Methods, Data availability, licence |
| nature.com article page `s41467-020-18606-2` | Opened (cookie redirect). The supplementary section lists **four** files only |
| Peer Review File, MOESM1 (springer static-content) | Downloaded and read, 33 pp. **This holds the label-scheme clarification** |
| Supplementary Information, MOESM2 (PDF) | Read: Supp. Tables 1–2 only, no data |
| Supplementary Software, MOESM3 (DOCX) | Read: instructions for the web demo |
| Reporting Summary, MOESM4 (PDF) | Read: Software/code and Data sections |
| Europe PMC `/bin/` supplementary links | HTTP 403 |
| `test.platform.shukun.net` | DNS NXDOMAIN today; `platform.shukun.net` NXDOMAIN too |
| Wayback CDX for `test.platform.shukun.net*` | 10 captures, 2021-10-20 to 2024-08-10. Each is only a SPA shell titled 数坤数字人体平台 ("Shukun Digital Human Platform"), earlier 加菲智慧医疗平台 CareSphere |
| Zhihu `zhuanlan.zhihu.com/p/29607061020` ("HeadNeckCTA2025") | HTTP 403, unread (already covered in `community.md` / `challenges.md`) |
| WebSearch (zh) 宣武医院 头颈CTA 数据集 公开 下载 椎动脉 标注; 18766 头颈部CTA 数据集 下载; 卢洁 傅凡 头颈CTA 数据集 共享 申请; 数坤科技 开放数据集 头颈 血管 | No release, mirror or application page for this cohort |
| WebSearch (en) "Rapid vessel segmentation and reconstruction of head and neck angiograms" dataset request | Nothing beyond the paper |
| Europe PMC: follow-up Radiology 2023 (doi 10.1148/radiol.220996, same group) | Not open access; data statement not readable |

## Evidence

### Field of view and VA coverage (paper, Methods)
> "Each scan included the aorta, carotid artery, and intracranial artery with ~561–967 slices in the Z axis."

> "…complete the automatic framework of bone segmentation and vessel segmentation from the aortic arch to the intracranial artery…"

Fig. 5 legend: "c Left vertebral artery arising directly from the aorta … e Bone remains at the origin of the left vertebral artery. f Vein artifact at the origin of the right vertebral artery." So the VA origins are in the field of view and inside the segmentation.

### Label scheme (paper, "Radiologist annotations")
> "…prelabeled using ITKSNAP software with masks marking the areas of the bone, aorta, carotid artery, and intracranial artery. They were multiclass classification with four classes. We annotated the arteries including the aorta, subclavian artery, common carotid artery, vertebral artery, internal carotid artery, basilar artery, posterior cerebral artery, middle cerebral artery, and anterior cerebral artery. … each CTA scan was annotated with four masks…"

### Peer Review File: this settles the VA question
Reviewer #1 asked whether the vessels were "all defined as one class or not". The authors replied (Response to Reviewer #1, item 2):
> "instead of annotating all vessels, we only labeled the arteries including aorta, subclavian artery, common carotid artery, vertebral artery, internal carotid artery, basilar artery, posterior cerebral artery, middle cerebral artery and anterior cerebral artery. In addition, the bones and arteries were classified as four classes: bone, aorta, carotid artery and intracranial artery … The numerical performance assessment of vessels integrated aorta, carotid artery and intracranial artery."

Their schematic figure labels panels as "intracranial artery (a), carotid artery (b) and aorta (c)". The neck-level slice (b) shows the two carotids and two small paravertebral lumens, which I read as the VAs, all in the same cyan "carotid artery" colour.
To Reviewer #4 they added: "Dividing the total volume into 3 regions … since the sizes of the vessels were different in aorta, carotid and intracranial." The classes are therefore **body regions, not vessel identities**. The cervical VA (V1–V3) sits in the "carotid artery" (neck) class together with the subclavian, CCA and ICA. V4 and the basilar are presumably in "intracranial artery".
Acquisition: "The slice thickness was 0.625mm for the reconstruction and we uploaded the 2mm slice thickness to the PACs."

### Data availability (paper)
> "Currently, the source imaging data from five tertiary hospitals cannot be made publicly accessible due to privacy protection. The CerebralDoc is accessible from the webserver created for this study at http://test.platform.shukun.net/login. User name: test, password: 123456. Request for software access and the remaining clinical evaluation images generated by technologists can be submitted to F.F. The source data underlying Supplementary Table 2, Figs. 3a, 7, and 8 are provided as a Source Data file with this paper."

The nature.com page lists only Peer Review File, Supplementary Information, Supplementary Software and Reporting Summary. **No Source Data file is attached.** Even if one were, it would hold only figure statistics.

### Reporting Summary
- Software: "The CerebralDoc … is a commercial system and can not give people access to the source code."
- Data: "The imaging data cannot be made publicly accessible due to privacy protection."
- Sample: "18,766 raw contrast-enhanced head and neck CTA images collected … from January 2018 to February 2019". The paper says June 2017 – November 2018, so the two disagree.

### Supplementary Software (DOCX)
The demo was a web viewer: "These demo data contain a total of 164 clinical routine cases … users can view the processing results in real time." It shows VR/MIP/CPR/curved-MPR plus source slices with the segmentation overlaid. Nothing is said about downloading. The host is now gone (NXDOMAIN), and the Wayback copies hold only the JavaScript shell, no case data.

### Contacts
- First author: F.F. = Fan Fu (傅凡), ORCID 0000-0001-5313-8925, Xuanwu Hospital.
- Corresponding author: Jie Lu (卢洁), imaginglu@hotmail.com.
- Industry co-authors: Shukun (Beijing) Technology Co., Ltd.

## Assessment for the catheter-graft use

- **Public:** no. There is no download, no registration and no application procedure. The only on-request items are software access, which is commercial and closed-source, and the "clinical evaluation images generated by technologists", which are rendered reconstructions rather than volumes plus masks. No route is stated for anyone, Chinese or foreign. A collaboration with Xuanwu/Shukun would be the only path, and moving patient CTAs out of China would face extra hurdles.
- **Cervical VA labelled:** not as a VA. The VA lumen from the subclavian origin is included in the "carotid artery" region mask. Extracting it would need per-vessel separation (topological splitting at the subclavian and V4), which is relabelling work.
- **HeadNeckCTA2025:** consistent with sibling notes, it is a blog write-up of this cohort and adds no access.
- **Correction to the earlier notes:** `platforms.md` §1.1 reads the label list as VA-specific labels. `challenges.md` and `papers.md` are right that there are only four region classes. The Peer Review File confirms the latter.
