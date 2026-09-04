#!/usr/bin/env python3
"""Report 2: v1, v2, v3 -- how each mesh is constructed, the mathematics, and
what changes for calibre, lumen, MISR and construction, against the host.

    REPORTLAB_PYLIB=<dir with reportlab> python reports/make/report_versions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdfkit import H1, H2, H3, P, Small, Mono, Call, Bul, Num, Img, Tbl, Sp, build, title_page, TEXT_W, PageBreak

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "reports", "figs")
OUT = os.path.join(ROOT, "reports", "Mesh_Construction_v1_v2_v3.pdf")
cm = 28.35


def fig(name, width=TEXT_W, caption=None):
    return Img(os.path.join(FIG, name), width, caption)


story = []
story += title_page(
    "Three Ways to Build the Same Vessel",
    "v1, v2 and v3 collision meshes: how each is constructed, the mathematics behind it, and what changes for calibre, lumen, "
    "MISR and construction — measured against the host test anatomy",
    ["Companion to <b>Meshing_Pipeline_Analysis.pdf</b>. Same data, same measurements (2026-09-02/03), organised around the three constructions.",
     "All three versions share the same centerlines and radii — v3's centerlines are byte-identical to v2's in 272/272 anatomies — so every "
     "difference described here is a difference in how the collision surface is built from them.",
     "Figures are drawn from the per-anatomy reports (mesh_v2.json, mesh_v3.json, lumen_v1.json) and from the tube and timing experiments in saved/mesher_probe/."])

# ------------------------------------------------------------------ 1
story += [H1("1. What is being compared")]
story += [P("An anatomy is stored as a set of <b>centerlines</b> — polylines down the middle of each vessel — with a <b>radius</b> at every point. "
            "That is what the graft produces, what the insertion point and targets are defined on, and what the simulator's path-finder "
            "reads. But the simulator does not collide the device with a centerline; it collides it with a closed surface of triangles, "
            "the <b>collision mesh</b>, that has to be constructed from those centerlines and radii. The three versions differ only in that "
            "construction:")]
story += [fig("construction_versions.png", TEXT_W, "Figure 1. One cross-section, three constructions. The grey dashed circle is the declared radius.")]
story += [Tbl([["", "v1 — shipped", "v2 — signed distance", "v3 — v2 + real surfaces"],
               ["input", "centerlines + MISR radii", "centerlines + MISR radii", "centerlines + MISR radii <b>and</b> the segmented surfaces"],
               ["shape of each vessel", "circular tube, blurred", "circular tube, exact", "the patient's real lumen where segmented; circular tube elsewhere"],
               ["grid", "0.6 × 0.6 × 0.9 mm", "0.45 mm isotropic", "0.45 mm isotropic"],
               ["surface placed at", "blurred value = 0.5", "signed distance = 0", "signed distance = 0 of the union"],
               ["triangles per anatomy", "~3,700", "20,000", "20,000 (30,000 where needed)"],
               ["radius floors / trim", "1.60 mm / 4 mm trim", "1.0 mm / no trim", "1.0 mm / no trim"]],
              widths=[3.2 * cm, 3.4 * cm, 3.6 * cm, TEXT_W - 10.2 * cm])]

# ------------------------------------------------------------------ 2
story += [H1("2. Four words, defined")]
story += [fig("cross_section_schematic.png", TEXT_W, "Figure 2. Lumen, MISR, and what a circular tube keeps of a real cross-section.")]
story += [Tbl([["term", "what it means", "how it is measured in this report"],
               ["<b>lumen</b>", "The open channel inside the vessel wall — the space the device moves in. Its cross-section is rarely a circle: the carotid bulb widens on one side, a stenosis narrows it eccentrically, the ICA terminus flares.",
                "<i>meshed lumen radius</i>: distance from a centerline point to the nearest triangle of the collision mesh. Its minimum along the route, minus SOFA's 0.3 mm contact distance, must exceed the catheter's 0.35 mm radius for the route to be <i>navigable</i>."],
               ["<b>calibre</b>", "How wide the vessel is. For a circular tube it is one number, the radius. For a real lumen it depends on direction.",
                "declared calibre = the MISR stored with the centerline; meshed calibre = what the mesh actually presents. Their difference is the <i>deficit</i>."],
               ["<b>MISR</b>", "Maximum Inscribed Sphere Radius. At a centerline point, the radius of the largest sphere that fits inside the vessel; equal to the distance to the <i>nearest</i> wall. VMTK computes it; it is the only radius the pipeline carries.",
                "It is the target every tube is built to, so a tube of radius MISR is, by construction, the largest <i>circle</i> that fits — never the full lumen."],
               ["<b>construction</b>", "What kind of object the collision surface is: a smooth analytic tube, a blurred one, or a segmented surface with real irregularities; welded and closed, or a loose triangle soup; one connected piece or several.",
                "connected pieces, open edges, triangle count, and the <i>shape ratios</i>: 16 rays cast from the centerline in the plane normal to it; longest ray / MISR, area-equivalent radius / MISR, longest / shortest ray. A circle scores 1.0 on all three."]],
              widths=[2.4 * cm, 7.4 * cm, TEXT_W - 9.8 * cm])]
story += [Call("<b>Why the shape ratios are needed.</b> On the centerline, the distance to the nearest wall <i>is</i> the MISR by definition, so a "
               "'deficit' of zero says nothing about whether the far wall, the bulge or the ellipse is there. Only a measurement that looks in "
               "every direction can tell a real lumen from a tube of the same MISR.")]

# ------------------------------------------------------------------ 3 v1
story += [H1("3. v1 — a blurred binary tube")]
story += [H2("3.1 Construction")]
story += [Num(["<b>A grid.</b> A 3-D array of voxels is laid over the whole vessel tree, 0.6 mm across and 0.9 mm tall (the shipped patient's scan spacing, inherited).",
               "<b>Paint.</b> For every centerline point c<sub>i</sub> with radius r<sub>i</sub>, every voxel whose centre lies within r<sub>i</sub> of c<sub>i</sub> is set to 1. Everything else is 0. The tree is now a blocky binary solid — the union of all those spheres.",
               "<b>Blur.</b> The array is smoothed with a Gaussian of one voxel, and then smoothed again. The result is a field between 0 and 1 whose transition from inside to outside is spread over about two voxels.",
               "<b>Trace.</b> Marching cubes draws the surface where the field crosses its mid-value (in the code, the midpoint of the array's minimum and maximum, which equals 0.5 because the aorta saturates the array).",
               "<b>Simplify.</b> Quadric decimation removes 99 % of the ~370,000 triangles, keeping about 3,700."])]
story += [H2("3.2 Mathematics")]
story += [Mono("indicator:  I(x) = 1 if some centerline point i has |x − c_i| < r_i, else 0\n"
               "blur:       B(x) = (G_σ ∗ G_σ ∗ I)(x) = (G_{σ√2} ∗ I)(x),   σ = 1 voxel\n"
               "surface:    { x : B(x) = ½ (min B + max B) }  ≈  { B = 0.5 }\n"
               "mesh:       quadric decimation of that surface to 1 %")]
story += [P("Two consequences follow directly. First, <b>the wall moves inward</b>. For a straight tube of radius r, the blurred field on the axis is "
            "1 − exp(−r²/2σ²) and the 0.5 level sits inside the true wall by roughly σ²/(2r): blurring a curved boundary always shifts the "
            "half-level toward the centre of curvature, more the tighter the curve. Second, <b>thin tubes vanish</b>: when the on-axis value "
            "never reaches 0.5 — i.e. r &lt; σ·√(2 ln 2) ≈ 1.18 σ — there is no contour to trace at all. With σ√2 = 0.85 mm across the "
            "scan and 1.27 mm along it, that threshold is about 1.0 mm for a vessel running up the body and 1.5 mm for one running across it.")]
story += [fig("v1_blur_profiles.png", TEXT_W, "Figure 3. The radial profile of a blurred tube for three radii (isotropic σ = 0.85 mm). The dashed line is the traced level; the wall lands inside the true radius, by 0.46 mm at r = 1.2.")]
story += [fig("tube_erosion_measured.png", TEXT_W, "Figure 4. The same effect measured through the real code with straight test tubes. The v1 mesher loses 0.3–0.9 mm; it loses more across the scan than along it; a signed-distance field on the identical grid loses almost nothing.")]
story += [H2("3.3 What this does to the anatomies")]
story += [Bul(["Along the shipped RCCA routes the meshed lumen is a median <b>0.65 mm of radius narrower</b> than the declared MISR (both sets).",
               "Twelve millimetres short of the terminus the lumen is 0.35–0.43 mm where 1.17–1.52 mm is declared. With the 0.3 mm contact distance, the catheter cannot get there. <b>93 of 264</b> shipped anatomies are navigable end to end.",
               "The thin cranial vessels are severed into 3–4 disconnected pieces per anatomy.",
               "Every stage-C constant that mentions the mesher — the 1.60 mm route floor (which erases stenosis above 37 %), the 1.6 mm ECA floor, the 4 mm distal trim, the 0.35 mm fusing band — exists to survive this construction."])]

# ------------------------------------------------------------------ 4 v2
story += [H1("4. v2 — a signed-distance tube")]
story += [H2("4.1 Construction")]
story += [Num(["<b>A finer, isotropic grid</b> — 0.45 mm in every direction — but only a thin <b>band</b> of it round each vessel is ever computed.",
               "<b>Store a distance, not a 0/1.</b> Every band voxel is given the signed distance to the vessel wall: how far inside it is (positive) or how far outside (negative). For a union of spheres this is simply the largest of (r<sub>i</sub> − distance to c<sub>i</sub>) over the nearby centerline samples.",
               "<b>Trace the zero.</b> Marching cubes draws the surface where the distance changes sign. Because it interpolates between neighbouring voxel values, and the distance field is (locally) linear, the crossing lands on the true wall — between voxel centres, not on them.",
               "<b>Simplify carefully.</b> Quadric decimation with volume preservation to a budget chosen by measuring the simulator: 20,000 triangles. The 60,000-triangle surface is kept so a set can be re-cut at any other budget in seconds."])]
story += [H2("4.2 Mathematics")]
story += [Mono("field:      f(x) = max_i ( r_i − |x − c_i| )     within ~2 mm of the tube\n"
               "surface:    { x : f(x) = 0 }        = the exact union of the spheres\n"
               "vertex:     between voxels a and b:  x = a + (b − a) · f(a) / (f(a) − f(b))\n"
               "mesh:       quadric decimation (volume-preserving) to 20,000 triangles")]
story += [P("The nearest-sample lookup is done <b>per branch</b> and then the maximum is taken over branches. Near a junction the nearest sample by "
            "distance can belong to the thin vessel while a farther, fatter one governs the union; a single global nearest-neighbour would "
            "carve a false cavity into the fat vessel.")]
story += [fig("v2_sdf_vs_blur.png", 0.8 * TEXT_W, "Figure 5. The two fields for a 1.2 mm vessel. The signed distance is a straight line through zero at the wall; the blurred field's 0.5 crossing is 0.46 mm inside it.")]
story += [H2("4.3 The triangle budget, chosen by measurement")]
story += [fig("sofa_cost_vs_budget.png", TEXT_W, "Figure 6. Simulator step time (left) and lumen fidelity (right) against triangle count, one anatomy.")]
story += [P("Simulator cost is flat from 12 k to 38 k triangles and rises steeply below 9 k; lumen fidelity keeps improving up to 20 k. So 20 k dominates "
            "every 12 k option at the same cost. It is +55 % per simulator step over v1's 3.7 k; 9 k is the cheaper compromise (+25 %, 0.23 mm deficit).")]
story += [H2("4.4 What this does to the anatomies")]
story += [Bul(["Deficit 0.65 → <b>0.12 mm</b> at the budget (0.06 mm before decimation). Every anatomy in both sets is navigable; the two set-A anatomies previously excluded for a pinched or severed lumen pass.",
               "One connected piece per anatomy instead of 3–4.",
               "The floors could drop to 1.0 mm because a 1.0 mm tube now meshes at 0.97 mm and 0.97 − 0.3 contact ≥ 0.35 catheter. Carotid stenosis grades up to ~60 % are back in the data; label necks (segmentation one voxel thin) are repaired at the same floor.",
               "No distal trim: the signed-distance cap at a terminus is a full-radius hemisphere."])]

# ------------------------------------------------------------------ 5 v3
story += [H1("5. v3 — the tube, plus the patient's real surface")]
story += [H2("5.1 Construction")]
story += [P("Every siphon has a real segmented surface (the TopBrain label surface that stage A built), and every set-B carotid has a real "
            "lumen surface (the Zenodo STL). v3 puts those back. Because the v2 field is a signed distance, adding a surface to it is one "
            "operation on the field, not a surgery on triangles:")]
story += [Num(["<b>Carry the surface where its centerline went.</b> The graft moved each section's centerline by a mirror (for left donors), a translation to the seam and a rotation that matches the section's start tangent and its 'up' direction to the host's. The grafters now record that map, and the surface is pushed through the same one.",
               "<b>Signed distance to the surface.</b> On the band voxels round the section, compute how far each voxel is from the transformed surface, positive inside (the sign comes from the surface's outward normals, which are re-derived after the transform because a mirror reverses them).",
               "<b>Clip to a capsule.</b> A generous tube round the <i>kept</i> centerline — 1.8 × MISR + 1 mm, tapered down to the tube radius over 8 mm before each seam — is intersected with the real surface. It removes what the graft never kept (the trimmed inlet, the label below the anchor trim, the ICA terminus flaring into the circle of Willis, the far side of a cut face) and makes the hand-over to the tube continuous.",
               "<b>Union with the tube.</b> Take the larger of the tube field and the clipped real field. Where the real lumen is wider than the tube, the real shape wins; where it pinches below the floored tube, the tube wins."])]
story += [H2("5.2 Mathematics")]
story += [Mono("transform:  v′ = R · (m ⊙ v − o) + a\n"
               "            m = mirror (±1, 1, 1)      o = trimmed proximal point\n            R = frame match (tangent→tangent, up→up)\n            a = seam point on the host\n"
               "real field: f_real(x) = − signed_distance(x, transformed surface)   (+ inside)\n"
               "capsule:    f_cap(x) = max_i ( ρ_i − |x − c_i| )\n            ρ_i = r_i + (1.8 r_i + 1 − r_i) · w_i,   w_i → 0 within 8 mm of a seam\n"
               "union:      f(x) = max( f_tube(x), min( f_real(x), f_cap(x) ) )\n"
               "surface:    { x : f(x) = 0 }")]
story += [fig("v3_union_schematic.png", TEXT_W, "Figure 7. The union in one cross-section. The neck on the lower left of the real lumen is filled by the floored tube; the bulge on the upper right survives; anything beyond the dashed capsule would be clipped.")]
story += [P("Two properties are guaranteed by the formula. Since f ≥ f<sub>tube</sub> everywhere, the v3 lumen <b>contains</b> the v2 tube: the navigability "
            "guarantee cannot be lost. And since min(f<sub>real</sub>, f<sub>cap</sub>) ≤ f<sub>cap</sub>, nothing from the source surface can appear "
            "outside the capsule: a side vessel in the label, or the part of a carotid STL the graft discarded, cannot leak into the anatomy. "
            "Two practical details were learned while building it: the marching-cubes band must be <i>dilated</i> rather than eroded (the union "
            "puts surface within a voxel of the band edge), and clipped side-vessel stubs leave tiny fragments (8–2,300 triangles against ~900,000) "
            "that are dropped and logged.")]
story += [H2("5.3 Measuring what came back: the shape ratios")]
story += [Mono("at each 4 mm station: cast 16 rays in the plane normal to the centerline,\n                      read the wall distance d_k along each\n"
               "longest / MISR = max_k d_k / MISR         (1.0 for a tube; > 1 with a far wall)\n"
               "area-r / MISR  = sqrt(mean_k d_k²) / MISR  (circle with the polygon's area)\n"
               "max / min      = max_k d_k / min_k d_k    (ellipticity)")]
story += [fig("shape_ratios.png", TEXT_W, "Figure 8. Longest wall ray over MISR, per anatomy, by section. Host tube sections read 1.02; real ICA siphons ~1.25; real carotid CCA–ICA ~1.13.")]

# ------------------------------------------------------------------ 6 effects
story += [H1("6. What changes, property by property")]
story += [Tbl([["property", "v1", "v2", "v3"],
               ["<b>calibre</b> vs declared MISR (median deficit)", "0.65 mm narrower", "0.12 mm narrower", "0.12 on tube sections; 0.04–0.06 on real sections"],
               ["<b>lumen</b>: minimum on the route, median (A / B)", "0.56 / 0.53 mm", "1.44 / 1.38 mm", "1.49 / 1.42 mm"],
               ["<b>lumen</b>: navigable anatomies", "22/49, 71/215 (35 %)", "49/49, 223/223", "49/49, 223/223"],
               ["<b>MISR</b>: how it is used", "tube radius, then eroded", "tube radius, kept exactly", "a floor under the real surface: the lumen is never narrower than max(MISR, 1.0)"],
               ["<b>construction</b>: cross-section", "circle, radius MISR − ~0.65", "circle, radius MISR", "real segmented section where available (longest ray 1.13–1.25 × MISR); circle elsewhere"],
               ["<b>construction</b>: topology", "3–4 pieces, some open edges", "1 piece, closed", "1 piece, closed (fragments dropped)"],
               ["thin vessels (< 1.2 mm)", "absent from the mesh", "present at declared radius", "present at declared radius"],
               ["stenosis (set B)", "capped at 37 % by the 1.60 floor", "up to ~60 %", "up to ~60 %, with the real eccentric shape"],
               ["terminus", "eroded; 4 mm trimmed", "full-radius cap, no trim", "real terminus, clipped at the capsule"],
               ["triangles / SOFA cost per step", "3.7 k / 1.0", "20 k / 1.55", "20–30 k / 1.55"]],
              widths=[4.6 * cm, 3.6 * cm, 3.6 * cm, TEXT_W - 11.8 * cm])]
story += [fig("lumen_distributions.png", TEXT_W, "Figure 9. Minimum lumen radius on the route, every anatomy. v1 straddles the 0.65 mm line the catheter needs; v2 and v3 clear it everywhere.")]
story += [fig("deficit_distributions.png", TEXT_W, "Figure 10. Median deficit per anatomy. v1 sits at 0.65 mm; v2 at 0.12; v3 lower still because the real sections add width beyond MISR in most directions.")]

# ------------------------------------------------------------------ 7 host
story += [H1("7. How each version relates to the host test anatomy")]
story += [P("The host is the one patient the shipped simulator was built on, and the one anatomy the policy is tested on. Its collision mesh "
            "is not a tube: it is the real VMR segmented surface, simplified by 90 % and written as an unwelded triangle soup — 3,584 "
            "triangles, 10,750 open edges. Measured the same way as the training sets, in the simulator's own frame:")]
story += [Tbl([["host test mesh", "value"],
               ["construction", "real segmented surface; decimate(0.9); triangle soup, every edge open"],
               ["deficit vs its own MISR, median (p10 … p90)", "−0.15 mm (−0.94 … +0.86): on average slightly <i>wider</i> than MISR, with real ±0.9 mm variation round the section"],
               ["minimum lumen on the route", "0.04 mm at 225 mm (distance-only; the surface is open, so point-in-solid is unreliable) — a genuine narrowing or a decimation artefact, worth a look"],
               ["the same tree through the v1 mesher", "deficit +0.76, 5 pieces, fails the meshed-lumen test"],
               ["the same tree through the v2 mesher (host_v2_control)", "deficit +0.14, minimum lumen 0.99, one piece, passes"]],
              widths=[6.0 * cm, TEXT_W - 6.0 * cm])]
story += [fig("host_vs_sets.png", 0.85 * TEXT_W, "Figure 11. Calibre relative to declared MISR: the host test mesh against the three training constructions.")]
story += [H2("Similar and dissimilar, version by version")]
story += [Tbl([["", "v1 training vs host test", "v2 training vs host test", "v3 training vs host test"],
               ["calibre", "<b>dissimilar</b>: training vessels ~0.8 mm of radius narrower than the equivalent host vessel", "<b>similar</b>: within ~0.3 mm of the host's average", "<b>similar</b>: within ~0.25 mm, and the direction of the difference now matches (real sections read wider than MISR, as the host does)"],
               ["lumen shape", "dissimilar: circles vs a real, irregular lumen", "dissimilar: circles vs a real lumen", "<b>similar where segmented</b>: real sections carry the same kind of ±20–25 % directional variation the host has; the host arch/trunk remain tubes"],
               ["MISR relation", "mesh well inside the inscribed sphere", "mesh = inscribed sphere", "inscribed sphere is a floor; lumen extends beyond it, as in the host"],
               ["topology / quality", "dissimilar, and worse: several pieces", "dissimilar, and <b>better</b> than the host: closed, welded, one piece", "same as v2"],
               ["triangle density", "similar (3.7 k vs 3.6 k)", "denser (20 k)", "denser (20–30 k)"],
               ["what the device feels", "a uniform tube narrower than the data", "a uniform tube at the data's calibre", "a real wall on the route's donor sections; a tube on the shared host part"]],
              widths=[2.4 * cm, 4.3 * cm, 4.3 * cm, TEXT_W - 11.0 * cm])]
story += [P("What is still dissimilar after v3 is confined to the host's own share of every anatomy — the arch, trunk, RVA and the other shipped "
            "branches — which are procedural and have no segmentation, and to the 34 synthetic ICA extensions. The remaining step, if train "
            "and test are to share construction exactly, is to put the host's real VMR surface through the same union so the test mesh is "
            "also closed, welded and at a known calibre; the tube half of that already passes every check.")]

# ------------------------------------------------------------------ 8 summary
story += [H1("8. Summary")]
story += [Tbl([["", "one-line description", "use it when"],
               ["v1", "a blurred binary tube; ~0.65 mm narrower than the data; thin vessels missing; 35 % of anatomies navigable", "reproducing earlier results only"],
               ["v2", "an exact signed-distance tube at the declared MISR; every anatomy navigable; floors at 1.0 mm; 1.55 × simulator cost", "the training set should be clean tubes at true calibre — e.g. to isolate calibre from shape"],
               ["v3", "v2 plus the patients' real segmented surfaces on the donor sections; real cross-sections; same guarantees", "the training set should look like the test anatomy: real walls, real bulges, real stenoses"]],
              widths=[1.2 * cm, 8.6 * cm, TEXT_W - 9.8 * cm])]
story += [Call("The one thing every version still shares with the host — and the one it cannot fix — is that the host arch and trunk are tubes "
               "in the training sets and a real surface in the test. That is now the only construction difference left, and it is on the "
               "part of the route every anatomy shares.")]

story += [H1("Appendix — the same anatomy in each version")]
story += [fig("fig_v1_mr001.png", TEXT_W, "A1. topcow_mr_001 and _001_L, v1 figures (tubes, blurred mesher; the distal siphon thins to a thread).")]
story += [fig("fig_v2_mr001.png", TEXT_W, "A2. The same two, v2 (signed-distance tubes at declared calibre).")]
story += [fig("fig_v3_mr001.png", TEXT_W, "A3. The same two, v3 (drawn from the baked mesh; the siphon carries the real surface, note the terminus bulge on the mirrored left ICA).")]
story += [fig("render_v2_vs_v3_carotid.png", TEXT_W, "A4. A set-B anatomy from 40 mm along the route: v2 tube mesh (left) and v3 with the real carotid lumen and siphon surfaces (right).")]

build(story, OUT, "Three Ways to Build the Same Vessel — v1, v2, v3")
print("wrote", OUT)
