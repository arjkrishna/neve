# Verify 4: CereVessMRA, "annotation of vessels for 271 TOF-MRA volumes" (ScienceDB 10.57760/sciencedb.13880)

Adversarial check of the claim that this ScienceDB deposit may provide public cervical vertebral artery (VA) geometry. Checked 2026-10-01.

**Verdict: public, but useless for the cervical VA.** I downloaded the actual files anonymously. The labels are **binary** 0/1 "cerebral vascular region" masks with no artery classes, so there is no VA label of any kind. Every one of the 271 volumes is an intracranial TOF-MRA slab **73.6 to 89.6 mm tall**, whose lower edge sits at about the skull base (petrous/cavernous ICA, basilar trunk). At most the V4/vertebrobasilar junction falls inside the slab. V1 to V3 (subclavian origin to C1) are physically outside the field of view. The searcher's "none cervical" reading is confirmed and now measured, not inferred.

## Sources opened myself

| Source | Result |
|---|---|
| `https://www.scidb.cn/detail?dataSetId=d3a67e63c0054652b75bfcc3cf15ae3f` (curl, 788 kB SSR HTML) | Opened. Title, description, JSON-LD (`"conditionsOfAccess":"PUBLIC"`, licence `by-nc/4.0`, 5,966,980,025 bytes, V2). Metadata is English only; no Chinese description |
| WebFetch of the same page | Returned only the title (the page is a JS app) |
| DataCite `api.datacite.org/dois/10.57760/sciencedb.13880` | Opened. Creators Guo Bin, Chen Ying, Gong Qiyong, Bai Xiangzhi; IsSupplementTo 10.1038/s41467-024-53550-5 |
| ScienceDB file-tree API `POST /api/gin-sdb-filetree/public/file/childrenFileListByPath` (path `/V1`, `/V2`) | **One file**: `manual-mra-data.rar`, 5,966,980,025 bytes, md5 c8c0d888…, same file in V1 and V2 |
| Download `https://china.scidb.cn/download?fileId=e694bccc357258cefde08477f1cd9b29` | **Anonymous, no login, no cookies**: HTTP 206 with byte ranges. I pulled the RAR head, the RAR5 quick-open record at the tail, two complete label members and two complete image members |
| Nature Communications article (nature.com) | Cookie redirect; read instead via Europe PMC full-text XML PMC11511858 |
| GitHub `YingChen7/HQA_Cerebrovasculature` `data/2_CereVessSeg_CereVessPro/README.md` | Model code README; says only that two example MRA files with annotations are included. No label dictionary |
| WebSearch (zh) `CereVessMRA 数据集 TOF-MRA 脑血管 标注 科学数据银行` | No Chinese-language page about this deposit (results were COSTA) |

## Evidence

### What is in the archive (RAR5 quick-open listing, all 542 members)
- `manual-mra-data/raw/CereVess_<id>_0000.nii.gz` × 271 (images) and `manual-mra-data/gt/CereVess_<id>.nii.gz` × 271 (labels). There are no other files: no README, no label table, no centerlines, no meshes.
- IDs: `AB…` × 121 (Datian General Hospital, pathological), `Guys` × 57, `HH` × 40, `IOP` × 53 (IXI, healthy, UK), totalling 150 IXI.
- The labels total 19.8 MB packed (about 50 to 85 kB each), which by itself points to a single binary class.

### Labels are binary vessel masks (read from the files)
| Member | Image grid | Voxel (mm) | Axial extent | Label values |
|---|---|---|---|---|
| `gt/CereVess_AB10520` (Datian) | 401×449×112 | 0.469×0.469×0.80 | 89.6 mm | `0`: 20,086,810; `1`: 78,678 |
| `gt/CereVess_Guys016` (IXI) | 331×432×100 | 0.469×0.469×0.80 | 80.0 mm | `0`: 14,206,711; `1`: 92,489 |
| `gt/CereVess_HH012` (IXI) | 354×468×100 | 0.469×0.469×0.80 | 80.0 mm | `0`: 16,430,700; `1`: 136,500 |

There are no class IDs beyond 1. The paper agrees: "All TOF-MRA volumes were manually annotated cerebral vascular regions using an in-house LabelTool software", and it evaluates "the predicted binary segmentation results".

### Every volume is an intracranial slab (NIfTI headers of all 271 images)
I read the headers of the 228 images stored uncompressed in the RAR by byte-range, plus one compressed image extracted whole. The table is in `verify_4_figs/raw_headers.tsv`.

| Site | n | Slices | z spacing | Axial extent |
|---|---|---|---|---|
| Datian (AB) | 121 | 112 | 0.80 mm | **89.6 mm** |
| IXI Guys | 57 (23 read, 34 compressed) | 100 | 0.80 mm | **80.0 mm** |
| IXI HH | 40 (31 read, 9 compressed) | 100 | 0.80 mm | **80.0 mm** |
| IXI IOP | 53 | 92 | 0.80 mm | **73.6 mm** |

Check: 121×112 + 57×100 + 40×100 + 53×92 = **28,128 slices**, exactly the page's "including 28,128 slices". So no volume is taller than 89.6 mm. The cervical VA from the subclavian to C1 is roughly 15 to 20 cm long and lies below the skull base, so it cannot be inside any of these slabs.

### Where the slab sits (MIPs I rendered from the downloaded volumes)
`verify_4_figs/mip_AB10520.png`, `mip_AB10520_nolab.png`, `mip_Guys016.png`: coronal and sagittal MIPs, with the label shown in red. Both show the circle of Willis, MCA/ACA/PCA branches and the cortical arteries. The lower edge cuts through the petrous/cavernous ICA and the basilar trunk. No neck, no vertebral column and no V3 loop are visible. The red label covers the whole intracranial tree as one class.

### Page and paper text (verbatim)
- ScienceDB description: "We manually annotated 271 (healthy : pathological = 150 : 121) TOF-MRA volumes, including 28,128 slices. This annotated dataset, named CereVessMRA, supports the development of deep learning methods for cerebrovascular segmentation … This large-scale and multi-center cerebral arterial annotation dataset facilitated the development of generalizable deep learning models for automatic and precise cerebral arterial segmentation."
- Keywords: "MRA, Vessel Segmentation, Deep Learning". The words vertebral, neck, cervical and basilar do not appear on the page. In the paper, "neck" and "vertebral" are absent and "carotid" appears only in reference titles.
- Paper, Data availability: "The annotated CereVessMRA dataset has been deposited to the Science Data Bank under the Data 10.57760/sciencedb.13880 … Data of these CN subjects are not publicly available as a result of institutional restrictions for patient privacy protection." This refers to the larger unannotated cohort of 2,841 scans; the 121 annotated Datian volumes are inside the public RAR.

## Access
- **Open download.** ScienceDB `conditionsOfAccess: PUBLIC`, licence CC BY-NC 4.0 (non-commercial, attribution; cite Guo et al., Nat Commun 15, 9235, 2024). The direct link worked from this machine without login, Chinese ID or phone verification, and it supports HTTP ranges. A non-Chinese researcher can obtain it.
- 5.97 GB single RAR5 file. Images are NIfTI `.nii.gz`; labels are binary NIfTI `.nii.gz`.

## Assessment for the catheter-graft use
- **Public:** yes, openly downloadable.
- **Cervical VA labelled:** no. No VA class exists, the masks are binary, and the field of view ends at about the skull base, so V1 to V3 and the subclavian origin are absent from both images and labels. At most a short distal V4/basilar stub is inside the binary mask, already covered better by TopBrain (VA labels 23/24) in the English notes.
- **Correction to the claim:** none needed. `platforms.md` row "CereVessMRA … intracranial slab" is right; this note replaces the slices ÷ volumes estimate with the measured per-site extents (73.6/80.0/89.6 mm).

## Scratch artefacts
Session scratchpad (temporary): `qo.ps1` (RAR5 quick-open parser), `nii.ps1` (remote NIfTI header reader and label counter), `mip.ps1` (MIP renderer), `rar_members.tsv` (all 542 members with offsets).
