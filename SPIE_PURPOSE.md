# SPIE §1 — DESCRIPTION OF PURPOSE (draft 1)

Companion to `SPIE_ABSTRACT.md`. Same three-move structure as the exemplar, with the
"prior work + its limitation" paragraph replaced by the formulation argument, since this
is a first report rather than an extension.

---

## 1. DESCRIPTION OF PURPOSE

Endovascular thrombectomy is the standard of care for large-vessel-occlusion ischemic
stroke, with pooled trial evidence demonstrating substantial functional benefit over
medical management alone.<sup>1</sup> Delivery depends on manually navigating a guidewire
and catheter from peripheral access to the cerebral vasculature, a task that exposes the
operator to cumulative ionizing radiation and that depends on a limited pool of trained
neurointerventionalists, particularly outside major centres.<sup>2</sup> Robotic catheter
systems can execute device motions but remain teleoperated, so the navigation decisions
are still made by the physician. Reinforcement learning (RL) in physics simulation has
therefore been proposed for autonomous device navigation, and recent work demonstrates
single- and dual-device navigation of increasingly deep cerebral targets.<sup>2,3</sup>
The near-term clinical target is not full autonomy but supervised autonomy: the
interventionalist plans the route and oversees execution while the controller performs
the navigation.
However, the generalizability of learned navigation policies across patient anatomy
remains incompletely characterized. Agents trained on one vessel mesh, or on a small
fixed set of meshes, can attain high success rates while encoding a single route;
success measured on the anatomy used for training does not establish performance on
anatomy the agent has not seen.<sup>4</sup> Training and evaluating anatomy
generalization may benefit from large sets of vessels spanning a range of tortuosity,
lumen calibre, and branch geometry; assembling such sets exclusively from patient
segmentations is difficult.

A second difficulty is specific to the control problem and constrains which formulations
are appropriate. Navigation is not Markov with respect to the information available at
deployment. Fluoroscopy reports device shape, but not the elastic and torsional energy
stored along the shaft, the out-of-plane configuration lost to projection, or the
hysteretic stick–slip contact state at the vessel wall. These quantities govern how the
device responds to the next actuation, so visually identical configurations can evolve
differently under the same command; because they are path-dependent and the projection
is many-to-one, the corresponding belief is not reliably recoverable from observation
history. Policies conditioned on observation history<sup>5</sup> are therefore necessary
but not sufficient. Unobserved anatomy compounds this: each vessel constitutes a distinct
transition kernel, so anatomy generalization is a distributional problem rather than a
matter of model capacity.

The present work addresses both difficulties together — by synthesizing the anatomy
distribution rather than sampling it from segmentations, and by exploiting simulator
privilege during training rather than attempting to infer hidden state at inference. The
finite-element solver exposes contact forces, tensions, and the full device
configuration, which are Markov by construction and which mark the stress states
separating the two behaviors on which navigation depends: centerline following, which
governs the large majority of steps, and recovery from buckling and wrong-branch entry,
which governs the few steps that determine the outcome. Geometric classifiers derived
from the clinician's planned route co-vary with these privileged fields and are
available at deployment, providing deployable proxies for them. This is a property of
the intended workflow rather than an assumption of privileged test-time knowledge: under
supervised autonomy the route is supplied by the operator, and pre-procedural
angiography and intra-procedural roadmapping already make it available. The deployable
policy therefore observes only fluoroscopic device tracking and its relation to that
route.

The workflow comprises (1) procedural synthesis of patient-derived carotid anatomies
resampled during training; (2) a geometric clearance audit admitting only anatomies
whose lumen calibre matches the patient-derived vessel; (3) asymmetric actor–critic
training with a privileged teacher actor and planned-path proxies; and (4) evaluation on
held-out synthetic anatomies and on the patient-derived anatomy. The simulated action
space — translation and rotation of each device — is the same set of degrees of freedom
a bench actuator drives, so a learned policy is executable without re-parameterization.
The current study is a feasibility demonstration in simulation, intended as the
foundation for staged validation: robotic actuation of those degrees of freedom on a
vascular phantom, then in-vivo evaluation, before clinical translation.

---

# NOTES FOR THE AUTHORS (delete before submission)

## Mapping to the exemplar

| exemplar move | here |
|---|---|
| clinical value established, with trial citations | thrombectomy standard of care, pooled trial evidence |
| current practice widely available | robotic systems exist but teleoperated |
| recent learned methods promising | RL for autonomous navigation, single- and dual-device |
| **However, generalizability … incompletely characterized** | **However, generalizability across patient anatomy … incompletely characterized** |
| conventional metrics insufficient | success on training anatomy does not establish generalization |
| controlled datasets needed, hard to assemble from patients | large vessel sets needed, hard to assemble from segmentations |
| *prior work + its limitation* | **replaced** by the formulation argument (¶2) — this is a first report |
| present work overcomes … workflow comprises (1)–(4) | present work addresses both … workflow comprises (1)–(4) |
| feasibility demonstration, foundation for larger trials | feasibility demonstration, foundation for multi-patient/in-vitro |

## References — VERIFY BEFORE SUBMISSION

Verified by literature search this session:

3. Karstensen L, et al. *Learning-based autonomous navigation, benchmark environments and
   simulation framework for endovascular interventions.* Comput Biol Med 2025.
   [arXiv:2410.01956](https://arxiv.org/abs/2410.01956) — source for the radiation /
   physician-scarcity framing **and** the anatomy-generation precedent (ArchVariety).
4. Robertshaw H, et al. *Autonomous navigation of catheters and guidewires in mechanical
   thrombectomy using inverse reinforcement learning.* IJCARS 2025.
   [doi:10.1007/s11548-024-03208-w](https://link.springer.com/article/10.1007/s11548-024-03208-w)
   — dual-device, ICA→M1, 96% on fixed patient anatomies.
5. Robertshaw H, et al. *Toward AI Autonomous Navigation for Mechanical Thrombectomy using
   HM-MARL.* IEEE RA-L 2026. [arXiv:2602.18663](https://arxiv.org/abs/2602.18663v1) —
   the honest multi-anatomy comparator (56–80% multi-vasculature).
6. *Recurrent neural networks for generalization towards the vessel geometry in autonomous
   endovascular guidewire navigation in the aortic arch.* IJCARS 2023.
   [doi:10.1007/s11548-023-02938-7](https://link.springer.com/article/10.1007/s11548-023-02938-7)
   — cite at "policies conditioned on observation history"; it is the direct precedent
   for the alternative ¶2 argues past.

**NOT yet verified — do not submit without checking:**

1. Pooled thrombectomy trial evidence. Intended citation is the HERMES collaboration
   meta-analysis (Goyal M, et al., *Lancet* 2016) of five randomized trials. Confirm
   volume/pages and that the claim you make matches their endpoint (mRS shift at 90 days).
2. Operator radiation dose / neurointerventionalist scarcity. Ref. 3 states this framing
   and can carry it, but a dedicated occupational-dose or workforce citation is stronger.
   Search terms: "occupational radiation dose neurointerventional operator", "workforce
   shortage thrombectomy capacity".

Numbering above is provisional — renumber once the reference list is fixed.

## The supervised-autonomy framing — why it is load-bearing, not decoration

The stated translation path is **simulation → vascular phantom driven by a robotic
actuator → in-vivo (animal) → human**, with the deployed system navigating from
fluoroscopy under clinician supervision, the clinician supplying the planned route.
Four consequences for the paper:

0. **The phantom stage justifies the action space, and it has precedent.** The bench
   robot pushes/pulls and rotates each device — precisely the four degrees of freedom the
   simulated policy commands — so nothing has to be re-parameterized between simulation
   and bench. That is now stated in the closing paragraph, because it converts the action
   space from an arbitrary modelling choice into a hardware constraint the work was
   designed around. Ref. 3 (stEVE) is the precedent to cite: policies trained in that
   simulator transferred to physical test benches at up to 97/100, which is the strongest
   available evidence that this stage is reachable.

1. **It pre-empts the strongest methodological objection to this work.** A reviewer will
   ask whether planned-path features are privileged information leaking into the policy —
   i.e. whether the agent is told the answer. The answer is that the route is an *input
   of the clinical procedure*: the operator plans it, and pre-procedural CTA/MRA plus
   intra-procedural roadmapping are already standard in neurointervention. The paragraph
   now says this explicitly. Without it, the planned path reads as a convenient oracle;
   with it, the method is *shaped by* the deployment setting. **This is the single most
   important addition to §1.**
2. **It fixes what "deployable" means, and the observation split follows from it.** The
   policy sees fluoroscopic device tracking and its relation to the planned route —
   nothing else. Contact forces, tensions and full device configuration exist only in the
   simulator and are used only to supervise. That is exactly the asymmetric/teacher–
   student design, so the architecture is justified by the clinical target rather than
   by convenience.
3. **It sets the correct autonomy level.** Not full autonomy — supervised autonomy, with
   the operator planning and overseeing. This is the defensible near-term claim and
   avoids a reviewer reading the paper as proposing unsupervised robotic neurosurgery.
   If a citation is wanted, the surgical-autonomy-levels taxonomy (Yang et al.,
   *Sci Robot* 2017) is the standard reference — **verify before citing**.

Consider mirroring one clause of this in the Abstract (currently silent on deployment).
Minimal insertion, after the planned-path sentence:

> Because the route is supplied by the operator under supervised autonomy, these
> classifiers require no information the clinical workflow does not already provide.

## Wording choices worth keeping

- **"incompletely characterized"** rather than "unsolved" — mirrors the exemplar and is
  defensible; several groups report anatomy generalization, so a stronger claim invites
  a reviewer to cite ref. 5 against you.
- **"encoding a single route"** rather than "memorizing" — memorization is the
  interpretation; encoding-a-route is the observation.
- **"not reliably recoverable"** rather than "not recoverable" — the weaker claim is the
  true one and cannot be refuted by pointing at recurrent baselines. See the technical
  note in `SPIE_ABSTRACT.md`; the abstract never claims the problem "is not a POMDP",
  because formally it is one.
- **"necessary but not sufficient"** about history conditioning — concedes the prior work
  fairly while establishing why privileged training is needed.

## Deliberately absent

- The mesh-generation defect and its correction — appears only as the clearance audit,
  item (2), stated positively.
- Any "first" or "highest success" claim — see positioning constraints in
  `SPIE_ABSTRACT.md`.
- Why planned paths are clinically practical — deferred to Methods per author note.
