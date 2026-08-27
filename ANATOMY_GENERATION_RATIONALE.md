# Anatomy generation — shortened Methods text, and the rationale behind it

Two things here: a condensed replacement for the §2.1 subsection (≈160 words instead of
≈400), and the reasoning that justifies each choice — the part a reviewer will actually
probe, and which the parameter list does not answer on its own.

---

## 1. Replacement text for §2.1

> **Anatomy generation.** Each synthetic anatomy is produced by deforming the navigated
> branch of a patient-derived arterial tree. The centerline is displaced by a small number
> of low-frequency modes with random amplitude and phase, applied in two directions
> transverse to the vessel axis, so that the course meanders smoothly rather than kinking.
> A per-anatomy tortuosity factor scales the displacement, spanning nearly straight to
> markedly tortuous vessels, and lumen calibre is scaled independently by a few percent.
> The deformation tapers to zero at both ends of the branch, pinning the ostium and the
> terminus so that anatomically fixed landmarks do not move and the perturbed vessel
> remains connected to the unmodified tree without a surface boolean operation. The
> bifurcation with the adjacent vertebral artery is perturbed independently, so the
> deflection required to select the correct daughter branch differs between anatomies.
> Surfaces are rebuilt from the deformed centerlines and audited for lumen clearance
> before use.

Exact frequencies, amplitudes, envelope widths and voxel spacings move to a parameter
table or to supplementary material. They are reproducibility detail, not argument.

---

## 2. Why these choices produce RCCA-like vessels

### 2.1 Why low-frequency modes, and why more than one

Over a branch of ~240 mm, the three modes correspond to wavelengths of roughly 340, 180
and 115 mm. **Nothing shorter than about a hand's width of vessel is allowed to vary.**
That is the deliberate part: arteries are smooth conduits whose course meanders over
centimetre scales. A displacement field containing millimetre-scale components would
produce kinks — geometry that is not anatomical, that no guidewire could follow, and that
is numerically hostile to the finite-element solver.

Three modes rather than one, at incommensurate frequencies, prevents the result from
being a recognisable sine wave: the superposition does not repeat over the length of the
branch, so successive anatomies are not scaled copies of a single template. Independent
random amplitude *and* phase per mode means variation in **where** the vessel bends, not
only **how much**.

Displacement is applied along two transverse directions rather than one because the
carotid course is genuinely non-planar; a planar deformation would produce vessels that
are all bent within a single plane, and a policy could exploit that regularity.

### 2.2 Why this displacement scale

The nominal displacement scale is ~4 mm against a vessel of 2.5–3.4 mm diameter — roughly
one vessel width of lateral wander. Large enough to change which way the device must be
steered; small enough that the vessel remains a plausible artery rather than a corkscrew.

The tortuosity multiplier spans 0.4 to 1.6, i.e. displacement scales of about 1.6–6.4 mm.
**This is the parameter that stands in for inter-patient variation**, and it is the right
axis to vary: carotid tortuosity differs markedly between patients, increases with age and
hypertension, and is a recognised predictor of procedural difficulty. Clipping the factor
at roughly ±2σ prevents both degenerate ends — a perfectly straight vessel that teaches
nothing, and an extreme one that would be anatomically implausible or impassable.

Calibre is varied only a few percent (1σ ≈ 7%), because vessel diameter varies far less
between patients than vessel course does. Making calibre the dominant axis of variation
would be modelling the wrong thing.

### 2.3 Why both ends are pinned — the anatomical reason, not just the engineering one

The engineering reason is easy: holding both ends fixed keeps the perturbed branch
connected to the rest of the tree, so no surface boolean is needed and no junction has to
be repaired.

The anatomical reason matters more. **The two pinned points are the two places that do not
vary in the way the middle does.** The ostium is where the branch leaves its parent — a
fixed landmark relative to the arch, and the point at which branch selection occurs. The
terminus connects to the Circle of Willis. What varies between patients is the *course
between them*: the cervical run and the cavernous curve. Confining deformation to that
segment means the generator varies what anatomy actually varies, and holds fixed what
anatomy actually holds fixed.

The envelope is C¹ (smoothstep rather than linear ramp) so that no curvature discontinuity
is introduced at the transition between the pinned and free regions — a kink there would
be an artifact of the method, and would sit exactly where the device is being steered into
the branch.

### 2.4 Why the neighbouring bifurcation is perturbed too

If the RCCA/vertebral bifurcation were identical in every anatomy, the deflection needed
to enter the correct daughter would be the same every episode. A policy could then learn a
fixed motor pattern at the fork and never learn to *perceive* the fork. Perturbing the
neighbouring branch independently makes branch selection a geometric problem that must be
solved from observation each time. This is the one part of the generator aimed specifically
at preventing memorisation rather than at producing realism.

### 2.5 Why width and tortuosity stay realistic *in each section*

This is the question the parameter list answers least well, and it has a single clean
answer: **per-section realism is inherited, not synthesised.** The generator never
constructs a radius profile or a curvature profile from scratch. It takes the patient's
and modifies it.

**Width.** The deformation does not assign radii; it *scales* the ones already there,
$\rho \leftarrow \rho\,(1 + (\sigma-1)w)$, by a single per-anatomy factor of a few percent.
Everything that makes the real vessel's calibre profile anatomical is therefore carried
through untouched: the wide proximal common carotid, the taper through the cervical
segment, the narrow distal course. Measured along the source branch, radius falls from
about 2.6 mm proximally to about 1.3 mm at the distal cervical/petrous transition before
widening slightly again — and every generated anatomy reproduces that same shape, uniformly
rescaled. A section is realistic in width because **its width was never generated**. Because
$\sigma$ is one scalar for the whole branch rather than an independent draw per section, the
generator also cannot produce the anatomically absurd case of a distal segment wider than
its parent.

**Tortuosity.** The same logic, one step less obvious. The displacement is *added* to the
existing centerline, and it is deliberately much gentler than the curvature already present
at the places where curvature is high. Where the native vessel is close to straight — the
cervical run — the added meander dominates, and tortuosity varies visibly between
anatomies. Where the native vessel is sharply curved — the cavernous genua, radii of a few
millimetres — the added field (radius of order 100 mm) is a small perturbation on top, and
the native geometry continues to dominate. Each section therefore keeps its own
*character*: the parts of the carotid that vary between patients are the parts that vary
here, and the sharply curved segment stays sharply curved rather than being smoothed away
or randomised into something no carotid does.

The envelope reinforces this at the boundaries, where both radius and course revert exactly
to the source anatomy, so the junctions a real vessel is constrained to meet are met.

This is the mechanism behind the clearance result in §4: generated anatomies match the
source patient's calibre because they *are* the source patient's calibre profile, rescaled
by a few percent — the audit confirms that the surface reconstruction preserved what the
centerline deformation had preserved.

**The same mechanism is the limitation.** Inheriting the siphon is why it stays realistic
and also why it barely varies (see §3). The generator gives genuine variation in the
sections whose native curvature is low, and near-replication in the section whose native
curvature is high — which is precisely the section that is hard.

### 2.6 Why the surfaces are audited

Rebuilding a surface from a deformed centerline narrows the lumen relative to the
centerline radii it was built from. Left unchecked this silently produces vessels the
device cannot physically enter — which measures the mesher rather than the policy. The
audit is what converts "we generated anatomies" into "we generated anatomies a guidewire
can traverse, of the same calibre as the source patient". It is also what prevents the
opposite failure: buying variation by quietly widening the vessel, which would make the
task easier and the generalization claim hollow.

---

## 3. Limitations to state, not hide

**The generator samples a neighbourhood of one patient, not a population.** Every anatomy
is a deformation of a single segmented tree. It produces vessels that plausibly *could* be
that patient under different tortuosity, not draws from the distribution of carotid
anatomies across people. Branching topology, ostium angle and overall arch configuration
never vary. The honest claim is generalization across vessel *course and calibre*, not
across anatomy in full.

**The added curvature is much gentler than the native curvature it is meant to vary.** At
the nominal displacement scale the deformation introduces bends of radius on the order of
100 mm. The cavernous genua in the source anatomy have radii of roughly 4–5 mm. **The
generator therefore varies the cervical course substantially while leaving the hardest
geometry — the siphon — essentially as it was in the source patient.** This is worth
stating plainly for two reasons: it bounds what the generalization result demonstrates,
and it is the most likely explanation for why the cavernous segment remains the failure
mode in every evaluation. Increasing the frequency content specifically in the distal
segment is the obvious next step, and it must be paired with the clearance audit, since
sharper bends are exactly where reconstruction narrows the lumen most.

**The transverse directions are defined from the branch's overall chord, not the local
tangent.** For a branch that curves substantially, "perpendicular" is therefore only
approximate, and the deformation is not purely lateral everywhere. It bends the vessel
rather than stretching it in practice, but the construction is an approximation.

**The parameters were chosen to span a plausible range, not fitted to a cohort.**
Frequencies, amplitudes and the tortuosity distribution were selected for anatomical
plausibility and verified after the fact by the clearance audit and by transfer to the
source patient. They were not estimated from a set of real carotid centerlines. Fitting the
mode spectrum to a cohort of segmented anatomies is the single clearest way to strengthen
this section, and would let the paper claim the generated distribution *matches* rather than
merely *plausibly resembles* real variation.

---

## 3b. Edited LaTeX version of the combined draft

Tightened from ~250 to ~215 words, LaTeX-ready.

```latex
\textbf{Anatomy generation.} Synthetic anatomies are produced by deforming a
patient-derived tree of the carotid and vertebral branches. Only the navigated right
common carotid--internal carotid axis (RCCA--RICA) and the proximal $35$\,mm of the
adjacent right vertebral artery (RVA) are modified. The navigated centerline is displaced
by three low-frequency sinusoidal modes in normalized arclength, each with random
amplitude and phase, applied along two directions transverse to the vessel axis so that
the course meanders smoothly rather than kinking. A per-anatomy tortuosity factor scales
the displacement; lumen calibre is scaled independently by a few percent. The deformation
tapers to zero at both ends, pinning the ostium and the terminus so that anatomically
fixed landmarks are preserved and the perturbed branch remains connected to the unmodified
tree. The RVA is perturbed independently, so the deflection required to select RCCA over
RVA at the bifurcation differs between anatomies. Surfaces are rebuilt from the deformed
centerlines each generation and audited for lumen clearance before use.

\textbf{Choice of scales.} The nominal displacement is $4$\,mm against a $2.5$--$3.4$\,mm
lumen---about one vessel width of lateral wander, enough to alter the required steering
while keeping the result a plausible artery. A tortuosity multiplier spanning $0.4$--$1.6$
gives displacement scales of $1.6$--$6.4$\,mm, standing in for inter-patient variation:
carotid tortuosity differs markedly between patients and is a recognized predictor of
procedural difficulty. Clipping at $\pm 2\sigma$ excludes both degenerate extremes---a
straight vessel that teaches nothing, and one too tortuous to be plausible or traversable.
Calibre varies by only a few percent ($1\sigma \approx 7\%$), since vessel diameter varies
far less between patients than vessel course does.
```

### What was changed and why

**The second heading was wrong.** It read *Envelope and anchoring*, but the paragraph
discusses displacement scale, the tortuosity multiplier and calibre --- nothing about the
envelope or the anchors. Renamed *Choice of scales*. Anchoring is already covered in the
first paragraph, so nothing is lost.

**Removed a duplicated opening.** The draft said *"Anatomies are generated by perturbing a
patient-derived arterial tree"* and then, three sentences later, *"Each synthetic anatomy
is produced by deforming the navigated branch of a patient-derived arterial tree."* Same
sentence twice. Likewise *"a sum of three sinusoidal modes"* and *"a small number of
low-frequency modes"* --- kept once, as *three low-frequency sinusoidal modes*, which is
both specific and short.

**Anatomical naming.** *"the navigated right common carotid artery to RICA"* is not a
standard construction; *right common carotid--internal carotid axis (RCCA--RICA)* is. The
fork is between RCCA and RVA, so *"(CCA vs VA)"* was made *"select RCCA over RVA"* --- the
laterality matters, since only the right side is navigated.

**Grammar.** *"comprises of"* $\rightarrow$ *"comprising"*. *"a perfectly straight vessel
an extreme one"* was missing its conjunction. *"recognised"* $\rightarrow$ *"recognized"*
for US spelling, per SPIE style.

**Stranded sentence.** *"The complete tree is re-meshed from centerlines at every
generation"* sat at the end of the scales paragraph, where it is unrelated to scales. Moved
into the first paragraph beside the clearance audit, which is what it sets up.

**LaTeX.** `~4 mm` $\rightarrow$ `$4$\,mm` (a bare `~` is a non-breaking space in LaTeX and
would silently vanish); `±2σ` $\rightarrow$ `$\pm 2\sigma$`; `1σ ≈ 7%` $\rightarrow$
`$1\sigma \approx 7\%$` (an unescaped `%` comments out the rest of the line); en-dashes as
`--`, em-dashes as `---`; thin spaces before all units.

---

## 4. If a reviewer asks "how do you know they're realistic?"

The three answers available, in descending strength:

1. **Calibre is matched to the source anatomy by direct measurement** — median clearance
   2.14 mm against 2.14 mm, and marginally tighter in the narrow tail. Generated vessels
   are not easier to traverse than the patient's own.
2. **A policy trained only on generated anatomies transfers to the real patient anatomy**
   without retraining. Whatever the generated vessels are, they are close enough to elicit
   behaviour that works on a real one.
3. **The deformation is constrained to be smooth, connected and low-curvature**, with the
   anatomically fixed landmarks held fixed — so the failure modes of naïve procedural
   generation (kinks, disconnections, implausible calibre) are excluded by construction.

Note that none of these is a demonstration that the generated *distribution* matches the
population distribution. If that claim is wanted, §3's last limitation is the work required.
