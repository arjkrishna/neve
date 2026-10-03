# Public cervical vertebral artery (VA) geometry: repositories and dataset papers (outside Zenodo and the big challenges)

Scope note: the Vascular Model Repository (VMR) claims below come from reading the VMR metadata CSV on GitHub. They also come from reading the ZIP central directories of the VMR downloads over HTTP range requests, and from pulling individual path files (.pth) and image headers (.vti) out of those ZIPs. Nothing was fully downloaded. Coordinates in the VMR files are in cm, as SimVascular writes them. The SPL Head-and-Neck Atlas was checked the same way, with its coordinates in mm.

## Q1. VMR (vascularmodel.com / SimVascular): which models carry the head-and-neck arteries with VAs, what the VA extent is, and the format and licence

### Takeaway
VMR has exactly ONE model that spans the aorta to the cerebral arteries: `0248_H_AOCERE_CAS`, a single CT patient ("R-1") released with 30 synthetic carotid-stenosis variants. It has left and right vertebral paths and segmentations. Both start at the subclavian arteries and run up to the basilar, so it is the best VMR source for grafting. The only other VMR models with a VA are four MR "vertebrobasilar" models (0050–0053), where the VA is cut off at the bottom of the field of view (upper cervical course only), and one paediatric coarctation model (0018) with a short proximal VA stub.

### Cited Findings
- The VMR metadata file `dataset-svprojects.csv` lists 316 projects. By anatomy: Aorta 91, Coronary 87, Pulmonary 48, Abdominal Aorta 43, Cerebral 26, Pulmonary Fontan 16, Pulmonary Glenn 4, and "Aortic Root - Cerebral Arteries" 1 — [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
- Anatomy code `AOCERE` = "Aortic Root - Cerebral Arteries" and disease code `CAS` = "Carotid Artery Stenosis" — [VMR abbreviations CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-abbreviations.csv)
- The only AOCERE entry is `0248_H_AOCERE_CAS` (legacy name "R-1"; source UCHealth University of Colorado Hospital; modality CT; added 5 May 2025; creators Ricardo Roopnarinesingh and Debanjan Mukherjee). Its notes: "CT-generated, heart-to-brain arterial models ranging from the aortic root to the A1, M1, and P1 cerebral arteries in the Circle of Willis. Differing degrees of bilateral stenosis were systematically added to a healthy patient model by manually decreasing segmentation sizes of the carotid vessels … resulting in 30 unique models with carotid stenosis degrees ranging from 10% to 100%." The flags Images/Paths/Segmentations/Models/Meshes/Simulations are all 1 — [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
- Associated papers for 0248: Roopnarinesingh et al., medRxiv 2024 ([doi:10.1101/2024.09.29.24314524](https://doi.org/10.1101/2024.09.29.24314524)) and Roopnarinesingh, Leppert & Mukherjee, J Am Heart Assoc 2023 ([doi:10.1161/JAHA.123.030792](https://doi.org/10.1161/JAHA.123.030792)) — [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
- Download files for 0248:
  - `svprojects/0248_H_AOCERE_CAS.zip` = the reference model, 716,998,427 bytes.
  - `svprojects/0248_H_AOCERE_CAS/0248_H_AOCERE_CAS_all.zip` = all variants, 17,263,201,305 bytes (~17.3 GB).
  - Simulation results: `svresults/…_3D_RIGID_REST_VTU_reference.zip` (~1.9 GB) and `…_VTU_all.zip` (~56 GB).
  - Source: [VMR file_sizes.csv](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/file_sizes.csv)
- The reference ZIP is publicly downloadable (HTTP 200, `Accept-Ranges: bytes`, no login) at https://www.vascularmodel.com/svprojects/0248_H_AOCERE_CAS.zip. Its central directory lists these files — [0248 ZIP](https://www.vascularmodel.com/svprojects/0248_H_AOCERE_CAS.zip):
  - SimVascular paths (`.pth`): `aorta`, `left_subclavian`, `right_subclavian`, `left_vertebral`, `right_vertebral`, `left_carotid`, `right_carotid`, `left_carotid_external`, `right_carotid_external`, `left_communicating`, `right_communicating`, `left_anterior`, `right_anterior`, `right_posterior`, `anterior_connector`.
  - Matching `.ctgr` segmentations, plus stenosis variants of the carotid segmentations (10/25/40/50/70/85/100%).
  - Model `Models/40Left85Right.vtp` (13 MB).
  - Volume mesh `Meshes/40Left85Right.vtu`.
  - Per-vessel mesh surfaces, e.g. `wall_left_vertebral.vtp`, `wall_right_vertebral.vtp`, `cap_left_vertebral.vtp`.
  - CT image `Images/p12full.vti` (180 MB).
  - `LICENSE`, `0248_H_AOCERE_CAS.pdf`, `…_ClinicalData.csv`.
- **VA extent in 0248, measured from the extracted `.pth` files** — [0248 ZIP](https://www.vascularmodel.com/svprojects/0248_H_AOCERE_CAS.zip):
  - `left_vertebral`: the first point is 0.13 cm from the `left_subclavian` path. It runs from z = −28.56 to z = −8.57 cm, a ~20 cm vertical span. Its upper part lies next to the start of the `right_posterior` (P1) path, which begins 0.07 cm from the left_vertebral path.
  - `right_vertebral`: the first point is 0.42 cm from the `right_subclavian` path. It runs from z = −28.56 to −10.15 cm and ends 0.05 cm from the left_vertebral path, i.e. it merges into the shared basilar segment.
  - Each VA path has 49 control points and 200 interpolated points.
- **CT image in 0248**, from the header of `p12full.vti`: 512×512×446 voxels, spacing 0.0400 × 0.0400 × 0.0700 cm (0.4 × 0.4 × 0.7 mm), origin (−20.46, 0, −31.15) cm, Int16 values from −1024 to 3071 (HU) — [0248 ZIP](https://www.vascularmodel.com/svprojects/0248_H_AOCERE_CAS.zip)
- The Stanford Digital Repository deposit for 0248 is "Roopnarinesingh, R., Jani, N., Leppert, M., and Mukherjee, D. (2025). 0248_H_AOCERE_CAS. Version 1. https://doi.org/10.25740/qy363kk7638". It is released under the VMR research licence — [SDR qy363kk7638](https://purl.stanford.edu/qy363kk7638)
- **VMR licence** (the `LICENSE` file inside each ZIP) — [0248 ZIP LICENSE](https://www.vascularmodel.com/svprojects/0248_H_AOCERE_CAS.zip):
  - "Permission is hereby granted, free of charge, to any person obtaining a copy of this data to use the data for research and development purposes".
  - The copyright/README notice must be kept.
  - Publications should acknowledge NLM R01LM013120 and NHLBI HHSN268201100035C, and/or cite Wilson et al., J Med Devices 7(4):040923 (2013), and/or reference vascularmodel.com.
  - The data are provided "AS IS".
  - The SDR landing pages add that content must not be used to identify individuals or infringe their privacy — [SDR 0050 page](https://purl.stanford.edu/jj935cd0824)
- **Cerebral models 0050–0053** (`0050_H_CERE_H` to `0053_H_CERE_H`): 4 healthy adults (ages 23–31), MR, complete projects (images/paths/segmentations/models/meshes/simulations), from Bockman et al., "Fluid Mechanics of Mixing in the Vertebrobasilar System: Comparison of Simulation and MRI", Cardiovasc Eng Tech 3:450–461 (2012) — [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
- The 0050 ZIP (96 MB) holds paths `LVA`, `RVA_2`, PICA/AICA/SCA/PCA branches (here `LSCA` means *superior cerebellar* artery, not subclavian), an MR image `OSMSC0078-cm.vti`, and model `0078_0001.vtp`. Measured from the files — [0050 ZIP](https://www.vascularmodel.com/svprojects/0050_H_CERE_H.zip):
  - The MR volume is 512×512×172, spacing 0.043×0.043×0.1 cm, z from −9.57 to +7.53 cm.
  - The LVA and RVA control points both start at z ≈ −9.42/−9.43 cm, 0.15 cm above the bottom edge of the volume.
  - The LVA runs up to z = −0.52 cm (~8.9 cm vertical span); the RVA path goes up to +4.47 cm.
- **Aorta models**: checking the path names of all 78 `*_H_AO_*` project ZIPs found only one with a vertebral path, `0018_H_AO_COA` (10-year-old female, severe native coarctation, MR, LaDisa et al. 2011) — [0018 ZIP](https://www.vascularmodel.com/svprojects/0018_H_AO_COA.zip); [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
  - Its `vertebral.pth` has 7 control points and is ~6.8 cm long, starting about 1 cm from the start of the `subclavian` path (z 6.88 → 13.57 cm).
  - The MR image extends to z ≈ 17.6 cm, so this VA is a short proximal stub that ends well inside the image — [0018 ZIP](https://www.vascularmodel.com/svprojects/0018_H_AO_COA.zip)
- **Cerebral aneurysm models 0199–0220** (22 models, Weiss et al., Front Physiol 2023): Paths = 0 for all of them, and several have images only. The aneurysms sit at the AComm, PComm, cavernous/supraclinoid ICA, ICA terminus and basilar apex — [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
- The VMR is also mirrored in the Stanford Digital Repository (one PURL/DOI per model, e.g. [0008_H_AO_H](https://purl.stanford.edu/wb108xr3896)). The vascularmodel.com dataset page itself renders the model list client-side, and WebFetch could not read it — [vascularmodel.com dataset page](https://www.vascularmodel.com/dataset.html)

### Inferences
- **0248_H_AOCERE_CAS is a directly graftable, real, cervical-complete VA pair.** SimVascular centerline paths, lofted per-vessel wall surfaces (.vtp) and the source CT (0.4×0.4×0.7 mm) are all included. Both VAs start at the subclavians and reach the vertebrobasilar confluence/basilar. It is only one anatomy, though: the 30 "models" are stenosis edits of the carotids of the same patient, and the VA geometry is the same in all of them.
- The surface-to-centerline distances suggest the SimVascular paths are drawn from inside the subclavian lumen, so the true V0 ostium is included in the lofted union surface.
- **Possible field-of-view mismatch.** The aorta path lies at z ≈ −36 to −38.5 cm, below the bottom of `p12full.vti` (z ≥ −31.15 cm), and the right subclavian path ends at the image x-edge (x = 0). The aortic-root part of the model may therefore come from a different series than the bundled image, or the image may be cropped. The VA origins (z −28.56) are inside the bundled CT. Check this before using the CT for re-segmentation.
- The Bockman vertebrobasilar models (0050–0053) cover only about the upper ~9 cm of the VA (upper V2, V3, V4) because the MR slab cuts them off. They do not start at the origin. Only 0050 was measured; 0051–0053 come from the same study and probably behave the same, but this was not verified.
- `0018_H_AO_COA` is paediatric and pathological (coarctation, where the VA can act as a collateral) and gives only ~7 cm of proximal VA. It is a poor graft source.

### Gaps
- VA path length on its own was not converted into physiological mm. The 49-point control polylines are noisy (arc length 35 cm for the left VA against a 20 cm vertical span), so use the lofted wall .vtp or re-extract centerlines with VMTK for accurate lengths and radii.
- The contents of `0248_H_AOCERE_CAS_all.zip` (17 GB) were not listed. It presumably holds the other 29 stenosis variants of the same anatomy.
- `0246_H_AO_AOD.zip` could not be parsed from its tail (0 path entries found), so it was not checked for vertebral paths.
- The 0248 PDF (`0248_H_AOCERE_CAS.pdf`) and ClinicalData.csv were not read. Patient age and sex are "-" in the metadata.

## Q2. TCIA (cancerimagingarchive.net): head-neck CTA collections with vessel or VA segmentations

### Takeaway
No TCIA collection with vertebral-artery (or any head-neck arterial tree) segmentations was found. TCIA's head-and-neck holdings are cancer/RT CT collections with tumour, organ-at-risk or body-composition labels.

### Cited Findings
- Searches for TCIA head-neck CTA with vessel labels surfaced only head-and-neck cancer CT resources, e.g. HNSCC with muscle/adipose segmentations at C3, and an orodental sub-volume annotation dataset. None had vessel labels — [medRxiv C3 sarcopenia segmentations](https://www.medrxiv.org/content/10.1101/2022.01.23.22269674.full.pdf); [medRxiv orodental CT annotation dataset](https://www.medrxiv.org/content/10.1101/2025.09.09.25333678.full.pdf)
- The AVT aortic dataset re-uses the TCIA RIDER Lung CT collection, but only for the aorta and its branches (see Q3) — [AVT Data in Brief (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8760499/)

### Inferences
- TCIA contrast neck CTs (e.g. HNSCC and other RT-planning CTs) might show the VAs in the image, but any VA geometry would have to be segmented by the user. There are no ready labels.

### Gaps
- The TCIA collection browser was not searched exhaustively (it is a dynamic app), and no "Analysis Results" collection with carotid/vertebral labels turned up. A direct check of TCIA's "CT angiography" modality filter is still to do.

## Q3. PhysioNet, OpenNeuro, Kaggle, Figshare, Mendeley, Dryad, Harvard Dataverse, Synapse: head-neck CTA/MRA with VA segmentations

### Takeaway
Two openly downloadable resources contain CERVICAL VA geometry starting at the subclavian, but both are single idealised or atlas subjects. The **SPL Head-and-Neck Atlas** (Open Anatomy) has VTK models of left and right VA, subclavians and aortic arch from one CTA. **BodyParts3D/Anatomography** has low-poly OBJ meshes of the left and right VA trunks, basilar and subclavians. No multi-patient head-neck CTA/MRA dataset with VA labels was found on Figshare, Kaggle, OpenNeuro, PhysioNet, Mendeley, Dryad, Dataverse or Synapse.

### Cited Findings
- **SPL Head and Neck Atlas** (Surgical Planning Laboratory, BWH/Harvard) is built on a reduced-resolution (256×256) copy of the MANIX CT dataset from the OsiriX sample library. It includes label maps and 3D models, is downloadable as `head-neck-2016-09.zip` (31.6 MB), and is licensed under "3D Slicer License section B" — [Open Anatomy SPL H&N Atlas](https://www.openanatomy.org/atlas-pages/atlas-spl-head-and-neck.html); [ZIP](https://www.openanatomy.org/atlases/nac/head-neck-2016-09.zip)
- The atlas ZIP contains `Model_105_right_vertebral_artery.vtk`, `Model_106_left_vertebral_artery.vtk`, `Model_103/104_right/left_subclavian_artery.vtk`, `Model_115_arch_of_aorta.vtk`, CCA/ICA/ECA and jugular/subclavian vein models, `HN-Atlas-labels.nrrd`, a colour table, `atlasStructure.json` and the grayscale `Osirix-Manix-255-res.nrrd`. `LICENSE.md` says: "released under the terms of the 3D Slicer License" — [SPL atlas ZIP](https://www.openanatomy.org/atlases/nac/head-neck-2016-09.zip)
- **SPL atlas extents, measured from the binary VTK points (mm)** — [SPL atlas ZIP](https://www.openanatomy.org/atlases/nac/head-neck-2016-09.zip):

  | Structure | Points | z range (mm) |
  |---|---|---|
  | Right VA | 3,728 | −91.7 to +48.2 |
  | Left VA | 3,398 | −88.3 to +48.1 |
  | Right subclavian | — | −128.9 to −82.7 |
  | Left subclavian | — | −132.8 to −78.4 |
  | Aortic arch | — | −161.2 to −129.8 |

  The lower end of each VA overlaps the z range of its subclavian, and the two VAs span ~14 cm vertically.
- **BodyParts3D** (DBCLS, Japan) is licensed CC BY-SA 2.1 Japan, requires attribution "BodyParts3D, Copyright© The Database Center for Life Science licensed by CC Attribution-Share Alike 2.1 Japan", and is distributed as OBJ meshes — [BodyParts3D download page](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html); [BodyParts3D README](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/README_e.html)
- The BodyParts3D/Anatomography 4.3 manifest, mirrored on GitHub, lists these parts — [olivercase/body_parts_3d_api MANIFEST.csv](https://github.com/olivercase/body_parts_3d_api):
  - "Trunk of right vertebral artery" (FJ1725, 800 faces; FJ5015, 1,096 faces)
  - "Trunk of left vertebral artery" (FJ1725M, 800 faces)
  - "Trunk of basilar artery" (244 faces)
  - Left and right subclavian artery trunks
- The older BodyParts3D 3.0 parts list (1,524 entries) contains subclavian arteries but no "vertebral artery" entry — [parts_list_e.txt](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/parts_list_e.txt)
- **AVT (Aortic Vessel Tree) on Figshare:**
  - Contents: 56 CTAs (KiTS19, RIDER Lung CT, Dongyang Hospital) with binary masks of "the aortas and its branches", CC BY 4.0, with the note that the "collection partly builds upon other public data collection, the appropriate licenses apply" — [Figshare 14806362](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362)
  - The data paper describes coverage of "the aortic arch and its branches". Its full text never mentions the vertebral arteries — [AVT Data in Brief (PMC8760499)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8760499/)
- **IXI:** ~600 healthy subjects including MRA, licensed CC BY-SA 3.0. The IXI site does not give the MRA field of view — [IXI dataset page](https://brain-development.org/ixi-dataset/)
- Papers using IXI report the TOF-MRA volumes as 512×512×100 voxels, with a median voxel size of 0.47×0.47×0.80 mm across sampled volumes, i.e. an ~8 cm slab around the Circle of Willis — [arXiv 2510.00665](https://arxiv.org/pdf/2510.00665); [Li et al., MRM 2026](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70072)
- **Figshare: intracranial-aneurysm CTA dataset** (Sci Data 2024, Anhui): 99 patients + 44 controls with raw CTA, 3D models and CFD, CC BY-NC-ND 4.0. It focuses on MCA aneurysms, and the "cerebral arterial vessels" were thresholded at 150–600 HU — [Sci Data 2024 (PMC11557944)](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11557944/fullTextXML); [Sci Data article](https://www.nature.com/articles/s41597-024-04056-8)
- **Kaggle:** searches found only carotid ultrasound datasets and the coronary ImageCAS; no head-neck CTA with VA labels — [Kaggle carotid (ultrasound) notebook](https://www.kaggle.com/code/orvile/carotid-artery-segmentation-attention-u-net/input)
- **OpenNeuro:** the relevant hit was CHUV/Lausanne TOF-MRA ds003949 (aneurysm/intracranial). OpenNeuro data are CC0 — [OpenNeuro ds003949](https://openneuro.org/datasets/ds003949/versions/1.0.1)

### Inferences
- **SPL atlas:** the best "starts at the subclavian origin" option among non-VMR open repositories, and ready to use (VTK surfaces in patient space, with the arch, subclavians and carotids in the same frame). It is one subject, and at reduced in-plane resolution the small VA lumen may be blocky, so smoothing or re-extracting centerlines with VMTK is advisable. Whether the VAs reach the vertebrobasilar confluence was not verified: their upper x-ranges approach the midline, but no basilar model was listed. The Slicer licence (BSD-style) is permissive. The underlying MANIX image comes from the OsiriX sample library, whose own terms were not checked.
- **BodyParts3D:** idealised, very low-poly meshes (~800–1,100 faces per VA trunk). Useful as a topological template (origin → V4 → basilar) but not as realistic patient geometry. CC BY-SA means any derived meshes you redistribute must also be share-alike.
- **AVT:** probably truncates the arch branches near their origins and does not label the VA. It is not a VA source.
- **IXI MRA:** ~8 cm TOF slab, so it captures at most V4 and the V3 loop. It is an exclusion for cervical grafting.

### Gaps
- PhysioNet, Mendeley Data, Dryad, Harvard Dataverse and Synapse had no hits for head-neck CTA/MRA with VA labels. These searches went through the general web-search engine, not each repository's own search, so a small or poorly indexed deposit could have been missed.
- The Dongyang subset of AVT was not opened to check whether any VA origins are included in the masks.

## Q4. Published dataset papers (Scientific Data, Data in Brief, MedIA, arXiv, Chinese releases): head-neck CTA with carotid and vertebral labels

### Takeaway
Several large Chinese head-neck CTA datasets with explicit VA labels exist, but every one found is in-house/private: HNCTA (358), HNA (800), Peking Univ. Third Hospital (115), and Fu et al. (18,766). The one academic CTA set that spans arch to cerebral vessels for thrombectomy navigation (KCL, 12 CTAs) is also not released. No open data descriptor for a "supra-aortic/cervical arterial tree including the VA" was found.

### Cited Findings
- **HNCTA:** "358 head and neck CT scans", with labels for nine vessel categories including left and right VA, ICA, CCA, PComm and basilar; split 242/58/58. It is described as a private head-and-neck artery dataset — [DeformCL, arXiv 2506.05820 (HTML)](https://arxiv.org/html/2506.05820)
- **HaN-Seg**, as used in the same paper: 42 CTs with only the left and right carotid arteries as vessel targets — [DeformCL, arXiv 2506.05820](https://arxiv.org/html/2506.05820)
- **HNA:** 800 head-neck CTAs with "cerebral, vertebral and carotid arteries annotated", used in graph-convolution vessel-segmentation work. The search summary did not establish public availability — [arXiv 2301.02393](https://arxiv.org/pdf/2301.02393); [LIVE-Net, Comput Biol Med 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523003517)
- **Fu et al., Nat Commun 2020**, "Rapid vessel segmentation and reconstruction of head and neck angiograms using 3D CNN": 18,766 head-neck CTAs from 5 Chinese tertiary hospitals (June 2017–Nov 2018). Europe PMC marks the article as having "supporting data", but the Data-availability text could not be retrieved (Nature/PMC fetch blocked) — [PMC7518426](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7518426/); [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41467-020-18606-2&resultType=core&format=json)
- **Peking University Third Hospital:** 115 head-and-neck CTAs (2020–2024) annotated "from the lower cervical/aortic-arch region to the intracranial ICA C6 segment". The source is an "in-house carotid dataset" in a withdrawn arXiv paper — [BiSegMamba, arXiv 2605.30972](https://arxiv.org/abs/2605.30972)
- **KCL mechanical-thrombectomy navigation work:** 12 CTAs "encompass[ing] the aortic arch and extend[ing] to the cerebral vessels" (REC 20/ES/0005), turned into 3D Slicer/VMTK surface meshes. A later paper uses 10 arch-to-cerebral CTAs (REC 24/LO/0057). Neither gives a public data link — [arXiv 2503.24140](https://arxiv.org/html/2503.24140); [arXiv 2509.25518](https://arxiv.org/html/2509.25518)
- **Other data descriptors found are intracranial only:**
  - "A Dataset of Annotated DICOM Images of Head CT Angiography for Intracranial Aneurysm Detection" (172 CTA series, aneurysm annotations) — [Data (MDPI) 11(4):74](https://www.mdpi.com/2306-5729/11/4/74)
  - The Sci Data 2024 MCA-aneurysm CTA/3D-model set (see Q3) — [Sci Data 2024](https://www.nature.com/articles/s41597-024-04056-8)

### Inferences
- For a public, multi-patient source with cervical VA labels from the origin, nothing outside Zenodo and the challenges was found. The Chinese head-neck CTA corpora that do label the VA along its whole course (HNCTA, HNA, Fu et al.) would need a data-use request to the authors.
- The KCL group's arch-to-cerebral meshes are the closest match in purpose (catheter navigation), so contacting that group could be worthwhile. Their papers only mention navigation targets in the ICA/MCA, so VA coverage is also unconfirmed.

### Gaps
- Whether the HNA (800) and Fu et al. (18,766) data can be requested, and on what terms, is unconfirmed because the full texts could not be fetched.
- Chinese-language repositories (e.g. Baidu/Tianchi or Science Data Bank) were not searched in Chinese, so a Chinese-hosted head-neck CTA release could have been missed.

## Q5. Candidate summary: name, URL, modality, cases, VA coverage, label format, licence, fit for grafting

### Takeaway
Ranked for grafting: (1) VMR 0248_H_AOCERE_CAS; (2) SPL Head-and-Neck Atlas; (3) BodyParts3D (template only); (4) VMR 0050–0053 (upper cervical VA only); (5) VMR 0018 (proximal stub only). All five are single-subject, or effectively single-subject, geometries.

### Cited Findings

| # | Name / URL | Modality | Cases | VA coverage | Format | Licence / access | Fit for grafting |
|---|---|---|---|---|---|---|---|
| 1 | VMR `0248_H_AOCERE_CAS` — [ZIP](https://www.vascularmodel.com/svprojects/0248_H_AOCERE_CAS.zip), [SDR DOI](https://doi.org/10.25740/qy363kk7638) | CT (UCHealth) | 1 anatomy × 30 carotid-stenosis variants | L and R VA from the subclavian origin (path start 0.13 / 0.42 cm from the subclavian path) to the basilar/P1 region; ~20 cm vertical | SimVascular `.pth` centerline paths, `.ctgr` contours, `.vtp` model and per-vessel wall surfaces, `.vtu` mesh, `.vti` CT | VMR research licence (free for research and development, keep notice, acknowledge NLM/NHLBI or cite Wilson 2013); open download, no login | **Best.** Real, complete V0–V4 pair in the same frame as the arch, subclavians and carotids |
| 2 | SPL Head and Neck Atlas — [page](https://www.openanatomy.org/atlas-pages/atlas-spl-head-and-neck.html), [ZIP](https://www.openanatomy.org/atlases/nac/head-neck-2016-09.zip) | CT (MANIX, OsiriX sample), 256×256 | 1 | L and R VA overlap the subclavian z range and run ~14 cm up; confluence not verified | `.vtk` surface models, `.nrrd` label map + image | 3D Slicer License (BSD-style) | Good as one realistic origin-to-upper-cervical VA pair; lower resolution |
| 3 | BodyParts3D / Anatomography — [download](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html), [4.3 mirror](https://github.com/olivercase/body_parts_3d_api) | Modelled atlas | 1 | "Trunk of L/R vertebral artery", "Trunk of basilar artery", subclavians | `.obj` meshes | CC BY-SA 2.1 JP (attribution + share-alike) | Template only (~800–1,100 faces per VA) |
| 4 | VMR `0050`–`0053_H_CERE_H` — e.g. [0050 ZIP](https://www.vascularmodel.com/svprojects/0050_H_CERE_H.zip) | MR (Bockman 2012) | 4 | VA cut off by the bottom of the MR slab; ~9 cm upper cervical plus intracranial (0050 measured) | `.pth`, `.ctgr`, `.vtp`, `.vtu`, `.vti` | VMR licence | Partial: V2 (upper)–V4 only |
| 5 | VMR `0018_H_AO_COA` — [ZIP](https://www.vascularmodel.com/svprojects/0018_H_AO_COA.zip) | MR, paediatric coarctation | 1 | ~6.8 cm proximal VA from near the subclavian | `.pth`, `.ctgr`, `.vtp` | VMR licence | Poor (short, paediatric, pathological) |

Sources for the table: [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv), the ZIPs linked in the table, [SPL atlas ZIP](https://www.openanatomy.org/atlases/nac/head-neck-2016-09.zip), and the [BodyParts3D README](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/README_e.html).

### Inferences
- A practical route: graft the 0248 VA pair (and/or the SPL pair) as V0–V3 onto the existing intracranial VA pieces, registering at the V3/V4 junction (foramen magnum) and the subclavian origin.
- This gives at most 2–3 real cervical VA geometries from open, non-Zenodo sources. For variety (dominance, tortuosity, origin variants such as a left VA arising from the arch), the private Chinese corpora or the challenge/Zenodo datasets covered by other researchers would be needed.

### Gaps
- Radii and tortuosity statistics of the 0248 and SPL VAs were not computed.

## Q6. Exclusions: datasets whose VA is intracranial only, or absent

### Takeaway
Most "cerebrovascular" datasets found cover only the intracranial VA (V4), or none at all. They cannot supply cervical geometry.

### Cited Findings
- **IXI MRA** (CC BY-SA 3.0): 512×512×100 TOF slab (0.47×0.47×0.8 mm, ~8 cm) centred on the Circle of Willis. Intracranial VA only — [IXI](https://brain-development.org/ixi-dataset/); [arXiv 2510.00665](https://arxiv.org/pdf/2510.00665)
- **OpenNeuro ds003949** (CHUV/Lausanne TOF-MRA aneurysm cohort): intracranial TOF — [OpenNeuro ds003949](https://openneuro.org/datasets/ds003949/versions/1.0.1)
- **VMR 0199–0220** (Weiss 2023 aneurysm pairs): intracranial aneurysm sites, no paths released — [VMR metadata CSV](https://raw.githubusercontent.com/SimVascular/vascularmodel/main/dataset/dataset-svprojects.csv)
- **Sci Data 2024 MCA-aneurysm CTA/3D models** (Figshare, CC BY-NC-ND): cerebral arterial models around MCA aneurysms — [Sci Data 2024](https://www.nature.com/articles/s41597-024-04056-8)
- **MDPI Data 2026, head CTA aneurysm detection** (172 series): aneurysm annotations only — [Data 11(4):74](https://www.mdpi.com/2306-5729/11/4/74)
- **AVT** (Figshare, CC BY 4.0, 56 CTAs): aorta and arch branches; the VA is never mentioned in the data paper — [PMC8760499](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8760499/)
- **HaN-Seg:** only the carotid arteries as vessel classes (it is also on Zenodo) — [DeformCL](https://arxiv.org/html/2506.05820)
- **TopBrain 2025** (challenge; Zenodo): labels "Right VA"/"Left VA" among 40 CTA and 42 MRA classes, 25 training patients, commercial use needs permission. The data page does not state the VA extent — [TopBrain data page](https://topbrain2025.grand-challenge.org/data/)
### Inferences
- TopBrain may include some extracranial VA on CTA, since it also labels external-carotid vessels on MRA. It belongs to the challenge/Zenodo scope covered by other researchers and should be checked there.

### Gaps
- A search-result summary mentioned the MICCAI 2009 carotid lumen segmentation challenge (56 head-neck CTAs, carotid-only labels). No primary page was fetched, so treat this as unverified. It is carotid-only and a challenge set in any case.
- The HNA (800 CTA) description comes from a search summary that pointed to arXiv 2301.02393 and LIVE-Net. Which paper introduced HNA, and whether it is public, was not confirmed.
