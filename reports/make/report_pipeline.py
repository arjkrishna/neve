#!/usr/bin/env python3
"""Report 1: the meshing pipeline analysis, as a readable PDF.

    REPORTLAB_PYLIB=<dir with reportlab> python reports/make/report_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdfkit import H1, H2, H3, P, Small, Mono, Call, Bul, Num, Img, Tbl, Sp, build, title_page, TEXT_W, PageBreak

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "reports", "figs")
OUT = os.path.join(ROOT, "reports", "Meshing_Pipeline_Analysis.pdf")


def fig(name, width=TEXT_W, caption=None):
    return Img(os.path.join(FIG, name), width, caption)


story = []
story += title_page(
    "From Segmentation to Simulator",
    "How the vessel collision meshes are built, where they lose fidelity, what was changed, and what it gained",
    ["Scope: both anatomy sets — the TopBrain siphons grafted onto the shipped host (set A, 49 anatomies) and the "
     "three-source carotid anatomies (set B, 215 → 223 anatomies).",
     "Every number in this report was measured on this repository's code and data on 2026-09-02/03. The scripts and raw "
     "outputs are in <b>saved/mesher_probe/</b>; the source notes are <b>MESHING_PIPELINE_ANALYSIS.md</b> and <b>V2_BUILD_PLAN.md</b>.",
     "A companion report, <b>Mesh_Construction_v1_v2_v3.pdf</b>, explains the three mesh constructions and their mathematics in detail."])

# ------------------------------------------------------------------ 1. summary
story += [H1("1. Summary in plain language")]
story += [P("Each training anatomy reaches the simulator (SOFA) as a <b>collision mesh</b>: a closed surface of triangles that the "
            "guidewire and catheter push against. This report is about how that surface is made from the patient data, and what "
            "gets lost on the way.")]
story += [P("The central finding is simple. The segmented vessel surface from the patient data is used only to find the <b>centerline</b> "
            "and a <b>radius</b> along it, and is then discarded. The mesh SOFA sees is rebuilt from that centerline and those radii as a "
            "circular tube. The tube-builder in use (the \"v1 mesher\") paints spheres into a voxel grid, blurs the grid, and traces a "
            "contour through the blurred values. The blur pulls every wall inward: measured on the shipped sets, the meshed lumen is a "
            "median <b>0.65 mm of radius narrower</b> than the declared radius, thin vessels below about 1.2 mm vanish altogether, and "
            "12 mm short of the terminus the lumen is 0.35 mm where 1.17 mm is declared — narrower than the catheter.")]
story += [Call("<b>Consequence.</b> Only <b>93 of 264</b> shipped anatomies (22 of 49 in set A, 71 of 215 in set B) are physically navigable "
               "to their terminus by a 0.35 mm-radius catheter once SOFA's 0.3 mm contact distance is counted. Targets sampled in the "
               "unnavigable stretch cannot be reached, and the training signal cannot tell that apart from a policy failure.")]
story += [P("Replacing the blurred-sphere construction with a <b>signed-distance</b> construction (the \"v2 mesher\") removes the loss: "
            "the meshed radius is within 0.06 mm of the declared one before any simplification, and 0.12 mm at the chosen triangle "
            "budget. Rebuilding both sets this way (v2) made <b>100 %</b> of anatomies navigable, restored the two set-A anatomies that "
            "had to be excluded, and let the radius \"floors\" that existed only to survive the old mesher drop from 1.6 to 1.0 mm, "
            "bringing carotid stenosis grades of up to ~60 % back into the data.")]
story += [P("A third construction (v3) then puts the <b>patients' real segmented surfaces</b> back into the mesh where the graft kept them "
            "— the ICA siphon from TopBrain and the carotid bifurcation lumen from the Zenodo database — so the training vessels carry "
            "the bulges and eccentric cross-sections the real test anatomy has, while keeping v2's navigability guarantee.")]

story += [H2("Headline numbers")]
story += [Tbl([["", "v1 (shipped)", "v2 (signed distance)", "v3 (v2 + real surfaces)"],
               ["set A navigable", "22 / 49", "49 / 49", "49 / 49"],
               ["set B navigable", "71 / 215", "223 / 223", "223 / 223"],
               ["radius deficit along the route (median)", "0.65 mm", "0.12 mm", "0.12 mm on tube sections, 0.04–0.06 on real ones"],
               ["minimum lumen radius, median (A / B)", "0.56 / 0.53 mm", "1.44 / 1.38 mm", "1.49 / 1.42 mm"],
               ["mesh pieces per anatomy", "3–4 (severed fragments)", "1", "1"],
               ["cross-section shape (longest wall ray / MISR)", "1.0 (circle)", "1.0 (circle)", "1.24 on real siphons, 1.13 on real carotids"],
               ["SOFA step time (relative)", "1.0", "1.55", "1.55"]],
              widths=[5.2 * 28.35, 3.0 * 28.35, 3.4 * 28.35, TEXT_W - 11.6 * 28.35])]

# ------------------------------------------------------------------ 2. vocabulary
story += [H1("2. The words used in this report")]
story += [fig("cross_section_schematic.png", TEXT_W, "Figure 1. A real vessel cross-section, the inscribed sphere the pipeline measures, and what a circular tube keeps.")]
story += [Tbl([["term", "meaning here"],
               ["label map / segmentation", "A 3-D image in which every voxel carries a class number (e.g. 4 = right internal carotid). This is what TopBrain ships. Voxels are 0.3 × 0.3 × 0.6 mm."],
               ["surface", "A closed mesh of triangles wrapped round one label. Made by tracing the boundary between labelled and unlabelled voxels (marching cubes) and smoothing the staircase away."],
               ["centerline", "A polyline running down the middle of the vessel. Extracted by VMTK from the surface. Every anatomy's route is a centerline; the simulator's targets are sampled from it."],
               ["MISR", "Maximum Inscribed Sphere Radius: at each centerline point, the radius of the largest sphere that fits inside the vessel there. It is the distance to the <i>nearest</i> wall — for an elliptical lumen, the short half-axis. This is the only radius the pipeline carries forward."],
               ["lumen", "The open channel of the vessel that the device travels in. 'Lumen radius' in this report always means the meshed wall distance the device actually meets, not the declared MISR."],
               ["calibre", "How wide a vessel is. For a circular tube it is one number; for a real lumen it depends on direction — which is why MISR, area-equivalent radius and longest-ray radius are all reported for v3."],
               ["collision mesh", "The triangle surface SOFA loads. The device collides with its triangles and edges; SOFA keeps the device 0.3 mm away from it (the <i>contact distance</i>)."],
               ["signed distance field (SDF)", "A number stored at every voxel: how far that voxel is from the vessel wall, positive inside and negative outside. The wall is exactly where the number is zero."],
               ["iso-surface / marching cubes", "Tracing the surface where a stored field crosses a chosen value (0.5 for the blurred field, 0 for the signed distance). Marching cubes does this one voxel-cube at a time and interpolates between voxel values, so the wall is placed between voxel centres, not on them."],
               ["decimation / triangle budget", "Reducing a surface of ~900,000 triangles to the few thousand the simulator can afford, by repeatedly merging neighbouring triangles where the shape changes least (quadric decimation)."],
               ["deficit", "declared MISR minus meshed lumen radius at the same centerline point. Positive means the mesh is narrower than the data says it should be."],
               ["navigable", "The meshed lumen radius along the whole route, minus the 0.3 mm contact distance, is at least the catheter's 0.35 mm radius."],
               ["neck", "A stretch where the label is only one or two voxels thin (inscribed radius under 0.6 mm). An internal carotid is never that narrow, so a neck is a segmentation error."]],
              widths=[3.6 * 28.35, TEXT_W - 3.6 * 28.35], zebra=True)]

# ------------------------------------------------------------------ 3. pipeline
story += [H1("3. The pipeline, stage by stage")]
story += [fig("pipeline_flow.png", TEXT_W, "Figure 2. The five stages. The mesh SOFA loads (D) is built from the centerline and radii of stage B, not from the surface of stage A.")]
story += [P("A common assumption is that VMTK makes the meshes. It does not. VMTK is used once, in stage B, to extract a centerline with "
            "inscribed-sphere radii from a surface, and the surface is discarded afterwards. Everything the simulator collides with is "
            "re-synthesised in stage D from that centerline and those radii.")]
story += [Tbl([["stage", "tool", "what it does, in words", "settings that matter"],
               ["A. label → surface", "topbrain_tools/mask_to_surface.py",
                "Keep one label (4 = right ICA, 6 = left ICA) and its largest connected piece. Trace the voxel boundary into triangles in world millimetres. Smooth the staircase with a windowed-sinc filter (20 passes). Find the two ends of the vessel by skeletonising it.",
                "voxels 0.297 × 0.297 × 0.6 mm; pass band 0.1"],
               ["B. surface → centerline", "topbrain_tools/vmtk_centerline.py",
                "VMTK's vmtkCenterlines with the two ends as seeds. It computes the medial path and, at every point, the MISR. The end points are trimmed 1–3 mm because VMTK tacks the seeds on slightly off-path.",
                "no resampling, no smoothing"],
               ["A'/B'. Zenodo carotids", "shipped in the database",
                "Each carotid comes with a closed lumen STL and a VMTK centerline tree (root-to-tip paths with MISR). The tree is split into CCA, ICA and ECA by their common prefix. The STL was only ever probed, never meshed — until v3.",
                "138 carotids; 65 long enough for the seam"],
               ["C. graft", "graft_siphon.py, graft_three.py",
                "Place each donor section on the host: rotate it so its start tangent and its 'up' direction match the host's (a frame match, which pins the roll), blend the radii across the seam, apply radius floors, trim ends, check clearance to neighbouring vessels.",
                "v1: route floor 1.60, ECA floor 1.6, distal trim 4 mm, fuse band 0.35"],
               ["D. centerline → mesh", "bake_meshes.py → generate_mesh() (v1)",
                "See §3.1. Paint spheres, blur twice, trace the 0.5 contour, simplify to 1 % of the triangles.",
                "grid 0.6 × 0.6 × 0.9 mm; σ = √2 voxels; level = (min+max)/2; decimate(0.99)"],
               ["E. into SOFA", "sofabeamadapter.py",
                "Load the .obj as a triangle + line collision model. The device is a zero-width line that must stay 0.3 mm from the wall, so the wall is effectively 0.3 mm thicker than the mesh.",
                "contactDistance 0.3, alarmDistance 0.5"]],
              widths=[2.6 * 28.35, 3.4 * 28.35, TEXT_W - 9.6 * 28.35, 3.6 * 28.35])]
story += [Sp(4)]
story += [P("The host patient itself is different again: its collision mesh is the real VMR segmented surface, simplified by 90 % and written "
            "as an unwelded 'triangle soup' (3,583 triangles, every edge open). So until v3 the one <i>test</i> anatomy was a real "
            "segmented surface while every <i>training</i> anatomy was a smooth circular tube.")]

story += [H2("3.1 What the v1 mesher does, step by step")]
story += [Num(["<b>Paint spheres.</b> A 3-D grid of voxels (0.6 mm across, 0.9 mm tall) is created round the whole tree. For every centerline point, "
               "every voxel whose centre lies within the declared radius is set to 1; everything else stays 0. The vessel is now a binary, "
               "blocky tube.",
               "<b>Blur, twice.</b> The grid is smoothed with a Gaussian of one voxel, applied two times (combined width √2 voxels ≈ 0.85 mm sideways "
               "and 1.27 mm vertically). Values near the wall become fractions between 0 and 1.",
               "<b>Trace the 0.5 contour.</b> Marching cubes draws the surface where the blurred value equals half. (The level is not actually "
               "fixed at 0.5; it is set to the midpoint of the grid's minimum and maximum, which happens to be 0.5 only because the aorta "
               "saturates the grid to 1.0.)",
               "<b>Simplify.</b> Quadric decimation removes 99 % of the triangles, leaving ~3,700 for the whole tree."])]
story += [fig("v1_blur_profiles.png", TEXT_W, "Figure 3. Why step 2 costs radius. Blurring a tube of radius r spreads its edge over ~0.85 mm; cutting at 0.5 lands inside the true wall, "
                                             "by roughly σ²/(2r). For a 1.2 mm vessel that is nearly half its radius; a vessel under ~1.2 σ never reaches 0.5 at all and disappears.")]
story += [P("Two more properties follow from the construction. Because σ is set in <i>voxels</i> and the voxels are 0.6 mm wide but 0.9 mm tall, "
            "the blur is physically stronger vertically, so a vessel running across the scan loses more radius than one running up it. "
            "And because the blur happens to a binary image, it cannot be undone by choosing a different threshold: any single level is "
            "right for one radius and wrong for every other.")]

# ------------------------------------------------------------------ 4. measurements
story += [H1("4. Where the fidelity goes — measured")]
story += [H2("4.1 Stages A and B are faithful")]
story += [P("For all 25 patients and both ICAs (50 vessels), the MISR that VMTK reads off the smoothed surface was compared with the raw "
            "label's own distance transform at the same centerline points — i.e. with the inscribed radius the voxels themselves imply.")]
story += [Tbl([["", "median (MISR − label EDT)", "5th percentile", "MISR below label"],
               ["50 vessels", "+0.00 … +0.03 mm", "−0.07 … −0.11 mm", "31–50 %"]], widths=[3 * 28.35, 4.5 * 28.35, 4 * 28.35, TEXT_W - 11.5 * 28.35])]
story += [P("The surface smoothing does not shrink the vessel, and MISR tracks the label to within a tenth of a millimetre. Nothing in "
            "A or B needs fixing for radius fidelity. What MISR cannot do, by definition, is describe a non-circular lumen — it is the "
            "distance to the <i>nearest</i> wall — which is the limitation v3 addresses.")]

story += [H2("4.2 The label maps are clean, but have necks")]
story += [fig("label_necks.png", TEXT_W, "Figure 4. Neck counts per vessel. Both anatomies the earlier checks rejected (mr_015, mr_003_L) and both they called borderline (mr_013, mr_014) are on this list.")]
story += [P("Every label is a single connected component with no holes. What 20 of the 50 vessels do have is a run of skeleton voxels whose "
            "inscribed radius is under 0.6 mm — the segmentation is one or two voxels thin there. That is a label error (an ICA is never "
            "that narrow) and it becomes a centerline radius dip that the v1 mesher turns into a pinched or severed lumen. Screening the "
            "labels for necks predicts those failures four stages earlier and at a fraction of the cost; the v2/v3 builds repair them with "
            "a 1.0 mm radius floor and reject the one vessel whose neck run is long (the 19.5 mm fragment of mr_006's left ICA).")]

story += [H2("4.3 Stage D is where the lumen is lost")]
story += [fig("tube_erosion_measured.png", TEXT_W, "Figure 5. Straight 60 mm tubes of known radius pushed through the real generate_mesh path, "
                                                  "and through a signed-distance field on the same grid. Inscribed radius measured as the device sees it: distance from the axis to the nearest triangle.")]
story += [Tbl([["declared r", "v1, vessel along z", "v1, vessel along x", "SDF, same grid", "SDF, 0.3 mm grid"],
               ["1.0", "absent", "absent", "0.92", "0.97"], ["1.2", "0.66", "absent", "1.14", "1.17"], ["1.4", "0.78", "0.47", "1.35", "1.37"],
               ["1.6", "1.11", "0.72", "1.54", "1.57"], ["2.0", "1.62", "1.41", "1.96", "1.97"], ["2.5", "2.22", "2.05", "2.47", "2.47"],
               ["3.0", "2.83", "2.66", "2.97", "2.96"], ["4.0", "3.69", "3.64", "3.98", "3.96"]])]
story += [P("Three things are visible: the v1 mesher removes 0.3–0.9 mm of radius; it removes more from vessels running in the axial plane; "
            "and the loss is not the grid's fault, because a signed-distance field on the identical grid is within 0.06 mm from r = 1.2 up.")]
story += [P("On two real anatomies, measured along the RCCA route (end caps excluded):")]
story += [Tbl([["mesh", "triangles", "pieces", "min lumen (mm)", "declared there", "median deficit"],
               ["<b>topcow_mr_001</b> — declared minimum 0.84 mm at 226 mm", "", "", "", "", ""],
               ["shipped .obj (v1)", "3,709", "3", "<b>0.35</b>", "1.17", "<b>0.64</b>"],
               ["v1 before decimation", "371,020", "3", "0.31", "0.84", "0.34"],
               ["SDF before decimation", "387,140", "1", "0.76", "0.84", "<b>0.06</b>"],
               ["SDF at the v1 triangle count", "3,708", "1", "0.24", "1.17", "0.49"],
               ["SDF, 0.45 mm isotropic grid", "924,328", "1", "0.82", "0.84", "0.03"],
               ["<b>case_k_004_left__topcow_mr_010</b> — declared minimum 1.52 mm at 238 mm", "", "", "", "", ""],
               ["shipped .obj (v1)", "3,756", "4", "<b>0.43</b>", "1.52", "<b>0.62</b>"],
               ["v1 before decimation", "375,704", "4", "0.86", "1.52", "0.30"],
               ["SDF before decimation", "392,084", "1", "1.45", "1.52", "<b>0.05</b>"],
               ["SDF at the v1 triangle count", "3,756", "1", "0.96", "1.52", "0.48"]],
              widths=[6.2 * 28.35, 2.0 * 28.35, 1.5 * 28.35, 2.6 * 28.35, 2.4 * 28.35, TEXT_W - 14.7 * 28.35])]
story += [Bul(["The shipped meshes carry a median radius deficit of 0.62–0.64 mm. At the tightest point, 12 mm short of the terminus, the lumen is 0.35 / 0.43 mm "
               "radius where 1.17 / 1.52 mm is declared. The catheter is 0.35 mm in radius and SOFA adds 0.3 mm of contact distance: the device cannot reach the last centimetre.",
               "The deficit splits roughly in half: ~0.3 mm from the double blur, ~0.3–0.45 mm from decimating to 1 %. The signed-distance mesh at the same 3.7 k budget still loses 0.48 mm — the budget alone does that.",
               "The signed-distance construction removes the blur's loss entirely (0.06 mm), produces one connected piece instead of 3–4 (the extra pieces are thin vessels the blur severed), and costs the same 8–9 s to build."])]
story += [Call("Set-wide, on all 264 shipped anatomies (<b>saved/mesher_probe/lumen_v1.json</b>): median minimum lumen 0.56 mm (A) / 0.53 mm (B), "
               "median deficit 0.65 mm in both, and <b>93 of 264</b> pass the navigability test. The pipeline's own fit check reads the "
               "<i>declared</i> radius and passed every one of them.")]

# ------------------------------------------------------------------ 5. defects
story += [H1("5. Defects in the shipped pipeline")]
story += [Num(["<b>Erosion by design (D).</b> A binary tube blurred at σ = √2 voxels and cut at 0.5 shrinks every convex surface by about σ²/(2r) and deletes anything thinner than ~1.2 σ. Every radius floor, distal trim and fusing band in stage C is a patch over this one choice.",
               "<b>Anisotropic erosion (D).</b> σ is in voxels on a 0.6 × 0.6 × 0.9 mm grid, so horizontal vessels — the arch, the RVA take-off, the ECA — lose up to 0.3 mm more radius than superior-running ones.",
               "<b>Data-dependent iso-level (D).</b> marching_cubes(level=None) uses the midpoint of the grid's values. It is 0.5 only because the aorta saturates the grid; any cube without a fat vessel would silently trace a lower level and come out fatter.",
               "<b>Decimation that ignores where the device goes (D).</b> decimate(0.99) minimises a global error, so it spends the budget on the 12 mm arch and starves the 1–2 mm siphon. ~3.7 k triangles over roughly a metre of vessel is 4 per millimetre; a 1.5 mm vessel needs ~10 per millimetre to hold its inscribed radius within 0.1 mm.",
               "<b>Severed strays.</b> The shipped meshes have 3–4 connected pieces; the extras are thin branches the blur cut into islands. They are the reason a component-aware route check had to be written.",
               "<b>The fit check tests the wrong radius (E).</b> It compares the catheter with the centerline radius. The meshed lumen at the same point can be a third of that, and with the contact distance the navigable radius is smaller still.",
               "<b>Real lumen shape discarded (A → D).</b> The segmented surface is used only to seed VMTK. Stenoses, elliptical sections and the carotid bulb become circular tubes — while the host test anatomy is its real surface: a train/test mismatch in wall geometry on top of the anatomy split.",
               "<b>End caps.</b> Spheres at the terminus blurred to nothing, hence a 4 mm distal trim. A signed-distance cap is a hemisphere of full radius; the trim is unnecessary.",
               "<b>Label necks not screened (A).</b> One-voxel necks predict both known failures; the pipeline discovered them four stages later.",
               "<b>Unwelded host mesh (E).</b> The shipped .obj is a triangle soup with 10,750 open edges. SOFA copes, but any manifold-dependent check on it is meaningless."])]

story += [H1("6. Inefficiencies")]
story += [Bul(["The whole grid (~40 million voxels) is filled and blurred twice to place a surface that occupies a thin shell; a band-limited field costs a fraction of that.",
               "The sphere painting is a Python loop per centerline point, each building index grids; it is vectorisable and irrelevant once the field is a signed distance.",
               "Two conda environments (one without VMTK, one without scipy/skimage) with a .vtp hand-off, only because the vmtk package was installed bare.",
               "Meshing at 0.6 × 0.6 × 0.9 mm 'to save cost' and then decimating 99 % spends resolution in the wrong place. Whether SOFA's collision detection even notices 4 k vs 20 k static triangles had never been measured — the 'collision cost is the bottleneck' line that fixed the budget was an assumption carried forward.",
               "The radius floors (1.60 mm) erase stenosis above 37 % to survive a mesher that would otherwise seal the vessel."])]

# ------------------------------------------------------------------ 7. VMTK
story += [H1("7. What VMTK provides, per stage")]
story += [P("The installed vmtk_env has 151 script modules. The ones relevant to this pipeline, with their options (full dump in "
            "<b>saved/mesher_probe/vmtk_options.txt</b>):")]
story += [H3("Label-map preprocessing (image stage)")]
story += [Tbl([["script", "options (defaults)", "use here"],
               ["vmtkimagereader", "Format (nifti via ITK), Flip, DesiredOrientation", "read the NIfTI directly"],
               ["vmtkimagebinarize", "Threshold, LowerLabel, UpperLabel", "isolate label 4 / 6"],
               ["vmtkimagemorphology", "Operation ∈ {dilate, erode, open, close}, BallRadius per axis", "<b>close</b> with radius (1,1,1) bridges one-voxel necks"],
               ["vmtkimagesmoothing", "gauss (σ in real units) or anisotropic diffusion", "smooth a signed distance, not the binary label"],
               ["vmtkimageinitialization", "isosurface / threshold / colliding fronts / fast marching / seeds", "seed a level set from the label surface"],
               ["vmtklevelsetsegmentation", "geodesic / curves / threshold / laplacian; propagation, curvature, advection weights", "regularise the label surface; the curvature term closes gaps without a Gaussian's shrink"],
               ["vmtkmarchingcubes", "Level, Connectivity", "iso-surface with the largest-component filter built in"],
               ["vmtksurfacetobinaryimage", "PolyDataToImageDataSpacing", "voxelise a surface onto the mesher grid"]],
              widths=[3.6 * 28.35, 6.2 * 28.35, TEXT_W - 9.8 * 28.35])]
story += [H3("Surface stage")]
story += [Tbl([["script", "options (defaults)", "use here"],
               ["vmtksurfacesmoothing", "taubin / laplace; iterations; pass band", "Taubin is volume-preserving; Laplace shrinks"],
               ["vmtksurfaceremeshing", "ElementSizeMode ∈ {area, edgelength, areaarray, <b>edgelengtharray</b>}; TargetEdgeLength; iterations; aspect ratio", "importance-weighted remesh (fine on the route, coarse elsewhere)"],
               ["vmtksurfacedecimation", "TargetReduction", "plain decimation"],
               ["vmtksurfacecapper", "simple / centerpoint / smooth / annular", "cap open ends"],
               ["vmtksurfaceconnectivity", "largest / closest / all", "drop severed strays"],
               ["vmtksurfaceclipper / endclipper", "box, sphere, centerline-normal clip", "clean cut faces"],
               ["vmtkflowextensions", "extension length / ratio / radius, transition", "extend a terminus by a few mm"],
               ["vmtksurfacebooleanoperation", "union / intersection / difference", "union surfaces (fragile on near-tangent surfaces)"],
               ["vmtksurfacedistance, vmtkdistancetocenterlines", "signed distance to a reference; tube-function evaluation", "QA"]],
              widths=[3.6 * 28.35, 6.2 * 28.35, TEXT_W - 9.8 * 28.35])]
story += [H3("Centerline stage")]
story += [Tbl([["script", "options (defaults)", "use here"],
               ["vmtkcenterlines", "SeedSelectorName ∈ {pickpoint, openprofiles, <b>carotidprofiles</b>, profileidlist, idlist, <b>pointlist</b>}; AppendEndPoints; Resampling + step; CostFunction 1/R", "carotidprofiles picks CCA/ICA/ECA automatically; Resampling replaces the ad-hoc resample"],
               ["vmtknetworkextraction / vmtkcenterlinesnetwork", "AdvancementRatio", "seed-free network, no MISR-quality radius"],
               ["vmtkcenterlineresampling / smoothing", "spline length; iterations, factor", "uniform spacing; moving-average smoothing"],
               ["vmtkcenterlineattributes", "abscissa, parallel-transport normals", "a roll-free frame along the vessel"],
               ["vmtkcenterlinegeometry", "curvature, torsion, tortuosity, Frenet frame", "the metrics the stats slide computes by hand"],
               ["vmtkbranchextractor + bifurcation reference systems", "GroupIds, Blanking, TractIds", "split a tree at bifurcations"],
               ["vmtkcenterlinesections", "section area, min/max diameter, shape index", "a real cross-section instead of MISR"],
               ["vmtkcenterlinemodeller / vmtkpolyballmodeller", "RadiusArrayName, SampleDimensions, ModelBounds", "the tube function on an image — the signed-distance mesher, packaged"]],
              widths=[3.6 * 28.35, 6.2 * 28.35, TEXT_W - 9.8 * 28.35])]

# ------------------------------------------------------------------ 8. what was changed
story += [H1("8. What was changed, and what it gained")]
story += [H2("8.1 The ranked changes")]
story += [Num(["<b>Mesh from a signed-distance field, iso-surface at zero, no blur</b> (v2 mesher, topbrain_tools/sdf_mesher.py). Deficit 0.64 → 0.06 mm before decimation; one connected piece.",
               "<b>Choose the triangle budget by measurement.</b> SOFA step time was timed at 3.7 k … 387 k triangles (Figure 6). It is flat from 12 k to 38 k, +25 % at 9 k, +5 % at 6 k. 20 k was chosen (0.12 mm deficit); the kept 60 k surface lets any budget be re-cut in seconds.",
               "<b>Test the meshed lumen, not the declared radius.</b> check_anatomies.py now reports the meshed lumen minus contact distance against the catheter radius.",
               "<b>Screen the labels for necks first</b> (topbrain_tools/label_necks.py) and repair them with a 1.0 mm floor.",
               "<b>Isotropic 0.45 mm voxels.</b> Cheap once the field is band-limited; 0.02–0.03 mm deficit before decimation.",
               "<b>Lower the floors</b> from 1.60 to 1.0 mm — a 1.0 mm tube now meshes at 0.97 — restoring stenosis grades up to ~60 %.",
               "<b>Drop the distal trim.</b> The signed-distance cap is a full-radius hemisphere.",
               "<b>Put the real surfaces back</b> (v3, topbrain_tools/sdf_union.py): union the transformed segmented surfaces with the tube field.",
               "<b>One environment</b> — not done; the two-env split remains."])]
story += [fig("sofa_cost_vs_budget.png", TEXT_W, "Figure 6. Left: simulator cost per step against triangle count, two contact regimes. Right: lumen fidelity against the same budget. The knee is at 6–9 k; from 12 k to 38 k the cost is flat.")]

story += [H2("8.2 v2 results")]
story += [P("Both sets were rebuilt into new folders (topbrain_data/anatomies_v2, carotid_data/anatomies_v2) with the same donors, pairing "
            "and host, so v1 → v2 is a controlled comparison. Set A's centerlines are identical to v1's on the shared prefix in 49/49, "
            "5 mm longer at the terminus, with six siphons lifted to 1.0 mm at their label necks; rise, kink and junction statistics are unchanged.")]
story += [fig("lumen_distributions.png", TEXT_W, "Figure 7. Minimum lumen radius on the route, every anatomy, by version. The dashed line is what the catheter needs after the contact distance.")]
story += [Tbl([["", "A v1 (49)", "A v2 (49)", "B v1 (215)", "B v2 (223)"],
               ["navigable", "22 (45 %)", "<b>49 (100 %)</b>", "71 (33 %)", "<b>223 (100 %)</b>"],
               ["minimum lumen, median (p10) mm", "0.56 (0.15)", "<b>1.44 (0.93)</b>", "0.53 (0.17)", "<b>1.38 (0.82)</b>"],
               ["median radius deficit, mm", "0.65", "<b>0.12</b>", "0.65", "<b>0.12</b>"],
               ["mesh pieces, median (max)", "3 (4)", "<b>1 (1)</b>", "3 (4)", "<b>1 (1)</b>"],
               ["per-anatomy min-lumen change", "", "+0.74 mm median, none worse", "", "+0.78 mm median, none worse"]])]
story += [P("Verification: set A 49/49 and set B 223/223 pass the static checks (route in one component, enclosure, targets, declared fit, "
            "meshed lumen); SOFA rollouts 6/6 and 9/9 with targets past the seam. Set B built 223 of 237 pairs against v1's 215: the lower "
            "ECA floor removed eight fusing rejections. The shipped host tree through the old mesher <i>fails</i> the meshed-lumen test "
            "(0.00 mm at its worst point, 5 pieces); through the v2 mesher it passes (0.99 mm, one piece) — the test separates the meshers, not the anatomies.")]
story += [Call("<b>Caught during the build.</b> The route floor was scoped to the donor section, so set B's TopBrain siphons carried their label "
               "necks unfloored; 13 anatomies baked at 0.30–0.63 mm. The meshed-lumen check flagged them on its first run; graft_three.py "
               "gained a siphon floor and the 22 affected pairs were regrafted. Keep that check in the bake, not only in the audit.")]

story += [H2("8.3 v3 results")]
story += [P("Same centerlines as v2 (verified byte-identical, 272/272); only the mesh changes. Each patient's real segmented surface — the "
            "TopBrain ICA label surface for the siphon; the Zenodo lumen STL for CCA, ICA and ECA in set B — is carried through the exact "
            "rotation, origin, anchor and mirror its centerline received in the graft, clipped to the part the graft kept, and unioned with "
            "the floored tube field. The companion report gives the mathematics.")]
story += [fig("shape_ratios.png", TEXT_W, "Figure 8. How non-circular the meshed lumen is, by section: the longest of 16 wall rays cast from the centerline, divided by MISR. Tubes read 1.02; real sections read 1.13–1.25.")]
story += [Tbl([["", "A v2", "A v3", "B v2", "B v3"],
               ["navigable", "49/49", "49/49", "223/223", "223/223"],
               ["minimum lumen, median (p10) mm", "1.44 (0.93)", "<b>1.49 (0.97)</b>", "1.38 (0.82)", "<b>1.42 (0.86)</b>"],
               ["deficit on tube sections / real sections", "0.12 / —", "0.12 / <b>0.04</b>", "0.12 / —", "0.12 / <b>0.06</b>"],
               ["shape of real siphon: longest ray / MISR, area-r / MISR, max/min", "—", "<b>1.25 / 1.10 / 1.26</b>", "—", "<b>1.24 / 1.09 / 1.25</b>"],
               ["shape of real CCA–ICA (set B)", "—", "—", "—", "<b>1.13 / 1.04 / 1.14</b>"],
               ["pieces / open edges (max)", "1 / 0", "1 / 5", "1 / 3", "1 / 9"]])]
story += [P("Verification: A 49/49 and B 223/223 pass; SOFA 6/6 and 10/10. One set B anatomy needed 30 k triangles instead of 20 k to keep "
            "its floored neck navigable after decimation; the bake now steps the budget up automatically and records it. Fragments of "
            "source surface clipped off by the capsule (8–2,300 triangles against ~900 k) are dropped and logged.")]
story += [fig("render_v2_vs_v3_mr001.png", TEXT_W * 0.92, "Figure 9. The same anatomy, v2 tube mesh (left) and v3 mesh with the real siphon surface (right), from 125 mm along the route.")]

# ------------------------------------------------------------------ 9. remaining
story += [H1("9. What remains")]
story += [Bul(["<b>The host arch, trunk and RVA stay tubes</b> in every version — the host is procedural and has no segmentation to use. The 34 synthetic ICA extensions on short carotid donors are tubes too.",
               "<b>The test anatomy is unchanged.</b> Its real VMR surface can go through the same signed-distance union so that train and test share construction exactly; the tube half of that (host_v2_control) already passes every check.",
               "<b>SOFA cost.</b> 20 k triangles is +55 % per simulator step over v1's 3.7 k. recut_obj.py re-cuts a whole set at 9 k (+25 %, 0.23 mm deficit) in seconds if throughput matters more than the last 0.1 mm.",
               "<b>73 carotids are too short</b> for the current seam design (ICA under 48 mm past the bifurcation). That is a grafting decision, not a meshing one.",
               "<b>One environment</b> for stages A–B, and VMTK's carotidprofiles / Resampling in place of the hand-rolled seeds and resampler.",
               "An analytic swept-tube mesh (exact radius, no voxels) was prototyped and set aside: without seam stitching at junctions it leaves gaps of up to 7 mm."])]

story += [H1("Appendix — files")]
story += [Tbl([["file", "what it is"],
               ["topbrain_tools/sdf_mesher.py", "v2 mesher: band-limited signed distance, iso-surface at zero, decimation, lumen measurement"],
               ["topbrain_tools/sdf_union.py, bake_meshes_v3.py", "v3: real-surface transform, capsule clip, union; per-section deficit and shape report"],
               ["topbrain_tools/bake_meshes_v2.py, recut_obj.py", "v2 bake; re-cut a set at another triangle budget from the kept 60 k surface"],
               ["topbrain_tools/label_necks.py", "neck screen over the raw label maps"],
               ["topbrain_tools/check_anatomies.py", "verification suite, now with the meshed-lumen column and --only"],
               ["graft_siphon.py, graft_three.py", "grafters with the floor / trim / band flags and the section transforms written out"],
               ["saved/mesher_probe/", "probe scripts, raw tables, SOFA timings, comparisons, renders"],
               ["MESHING_PIPELINE_ANALYSIS.md, V2_BUILD_PLAN.md", "source notes for this report; the bug → guard table"]],
              widths=[6.5 * 28.35, TEXT_W - 6.5 * 28.35])]

build(story, OUT, "From Segmentation to Simulator — meshing pipeline analysis")
print("wrote", OUT)
