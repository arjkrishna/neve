# SPIE abstract — draft 3 (no prior-work claim; adds the formulation rationale)

**Working title**
Procedurally generated vessel anatomy and force-field–aligned path supervision for
anatomy-generalizable autonomous neurovascular navigation

---

## ABSTRACT (primary — 297 words)

Reinforcement learning is increasingly demonstrated for autonomous catheter and
guidewire navigation in simulation, including dual-device navigation of the cerebral
vasculature. However, generalization across patient anatomy remains unresolved; agents
trained on a small set of fixed vessel meshes can attain high success rates while
memorizing those meshes. Training and evaluating anatomy generalization may benefit from
large sets of vessels spanning a range of tortuosity, calibre, and branch geometry,
which are difficult to obtain from patient segmentations alone.

Navigation is also not Markov with respect to the information available at deployment.
Fluoroscopy reports device shape, but not the elastic and torsional energy stored along
the shaft, the out-of-plane configuration lost to projection, or the hysteretic
stick–slip contact state — quantities that govern how the device responds to the next
actuation, so that visually identical configurations can evolve differently. Because
these quantities are path-dependent and the projection is many-to-one, the corresponding
belief is not reliably recoverable from observation history, and unobserved anatomy
makes each vessel a distinct transition kernel.

We therefore exploit simulator privilege during training rather than attempting to infer
hidden state at inference. The finite-element solver exposes contact forces, tensions,
and the full device configuration, which are Markov by construction; an asymmetric
actor–critic and a privileged teacher actor use these to supervise a policy restricted
to deployable observations. A planned-path formulation supplies geometric classifiers
that co-vary with the privileged force fields, giving that policy proxies for the stress
states distinguishing the two behaviors navigation requires: centerline following, which
governs the large majority of steps, and recovery from buckling and wrong-branch entry,
which governs the few steps that determine the outcome.

Specifically, the workflow combines procedural synthesis of patient-derived carotid
anatomies, a geometric audit admitting only anatomies whose lumen clearance matches
patient calibre, privileged teacher–student training, and planned-path proxies. Results
are presented on held-out synthetic anatomies and on the patient-derived anatomy.

---

## VARIANT A — results stated (replace the final sentence)

> On held-out synthetic anatomies the resulting policy succeeds in 84.7% of trials,
> navigating the common carotid and mid–internal carotid without failure, and transfers
> without retraining to the patient-derived anatomy in 75.5% of trials.

## VARIANT B — clinical-burden opening (Karstensen/Robertshaw convention)

> Endovascular intervention is the standard of care for acute ischemic stroke, yet
> suffers from operator radiation exposure and a scarcity of proficient
> neurointerventionalists. Reinforcement learning is increasingly demonstrated for
> autonomous catheter and guidewire navigation in simulation, but generalization across
> patient anatomy remains unresolved; …

---

# NOTES FOR THE AUTHORS (delete before submission)

## On the "not Markov" sentences — what is and is not claimed

This wording was chosen to be **defensible against a reviewer who formalizes it**, so it
is worth knowing exactly where the line is.

**Strictly, the simulated system IS a POMDP.** The FEM state — nodal positions and
velocities of both device models plus the active contact constraints — is Markov, and
fluoroscopy is a partial observation of it. A referee could correctly object to a flat
claim of "not a POMDP." The abstract therefore never says that. It says two narrower
things, both true:

1. **Not Markov *in the deployable observation*.** Uncontroversial and exactly right.
   The hidden variables named are the physically correct ones:
   - *stored elastic and torsional energy* — proximal torque does not transmit
     instantaneously to the tip; the shaft winds up and releases (whip/snap-back), so
     identical tip appearance can precede opposite tip motion;
   - *out-of-plane configuration* — a single projection is many-to-one; the shaft's
     depth component is unrecovered;
   - *hysteretic stick–slip contact* — Coulomb friction is path-dependent; whether a
     contact is sticking or slipping depends on loading history, not on current pose.
2. **The belief is not *reliably recoverable* from observation history.** This is the
   substantive claim and the reason history-conditioning (e.g. recurrent policies) is
   necessary but not sufficient: hysteresis means the sufficient statistic is not a
   function of any fixed-length observation window, and the projection is non-injective.
   We say "not reliably recoverable", not "not recoverable" — the weaker claim is the
   true one and cannot be refuted by pointing at RNN baselines.

**Third clause, the contextual/hidden-parameter point.** "Unobserved anatomy makes each
vessel a distinct transition kernel" positions the setting as a hidden-parameter or
contextual MDP (each patient vessel = a different MDP, no anatomy label at input). This
is the formal reason anatomy generalization is a *distributional* problem and not a
capacity problem — and it is what licenses the procedural-generation half of the paper.
These three sentences are what make privileged training the *appropriate* response
rather than one option among many: if the belief were recoverable, a recurrent policy
would suffice and the teacher would be unnecessary.

## Structure of draft 3

| move | sentences |
|---|---|
| context | RL increasingly demonstrated… |
| gap | generalization unresolved; memorization; hard to obtain anatomies |
| **why this problem resists the standard formulation** | not Markov in deployable observation; belief not recoverable; anatomy = distinct kernel |
| **why our methods follow from that** | exploit simulator privilege; asymmetric critic + privileged teacher; planned-path proxies separate the two behaviors |
| what was built | workflow combines A, B, C, D |
| results | held-out synthetic + patient anatomy |

"In prior work" is removed throughout — this is presented as the first work.

## Field register, from the literature

- [Karstensen et al., stEVE (arXiv:2410.01956, Comput Biol Med 2025)](https://arxiv.org/abs/2410.01956) —
  opens on clinical burden, names benchmarks, reports x/100. Their ArchVariety benchmark
  is the anatomy-generation precedent.
- [Robertshaw et al., IRL dual-device (IJCARS 2025)](https://link.springer.com/article/10.1007/s11548-024-03208-w) —
  Purpose/Methods/Results with explicit numbers (96%, 22.6 s).
- [Robertshaw et al., HM-MARL (RA-L 2026)](https://arxiv.org/abs/2602.18663v1) — closest
  comparator on anatomy generalization: **56–80% multi-vasculature**.
- [Recurrent networks for aortic-arch generalization (IJCARS 2023)](https://link.springer.com/article/10.1007/s11548-023-02938-7) —
  frames the gap as "each patient has a unique vascular system"; also the most direct
  precedent for the "recurrent policy" alternative our paragraph 2 argues past.

**This field states numbers in the abstract — prefer Variant A** unless submission
precedes final results.

## Provenance of every number quoted

| claim | value | evidence |
|---|---|---|
| held-out synthetic | **84.7%** (83/98) | v1bp ckpt2002292, 50 procedurally-varied anatomies, deterministic policy, 600-step cap, 5 mm threshold |
| CCA / mid-ICA / siphon | 100% (26/26) / 100% (40/40) / 53.1% (17/32) | same run |
| patient anatomy, no retraining | **75.5%** (74/98) | same checkpoint, original segmented surface, never in training |
| patient CCA / mid-ICA / siphon | 100% / 90.2% / 33.3% | same run |

Robustness: second-best checkpoint (ckpt514264) gives 83.7% synthetic / 72.4% patient —
the headline is not a single-checkpoint artifact.

## Positioning constraints — DO NOT VIOLATE

- **No "highest success" claim.** Robertshaw (IJCARS 2025): 96% ICA→M1 dual-device on 12
  *fixed* patient anatomies.
- **No "first dual-device siphon" claim.** The same work traverses it.
- Honest comparator is **HM-MARL 56–80% multi-vasculature**; place 84.7% on *generated,
  held-out* anatomy beside it and state the protocol difference.
- Cross-system numbers are **not** head-to-head — thresholds, step caps and truncation
  rules differ and are often unstated.

## Deliberately absent

1. **The mesh-generation defect** we found and corrected — a correction to our own
   instrumentation, not a result. It belongs in Methods as the clearance audit.
2. **Freeze-collapse / dead-end narratives** — stories about our own failed
   configurations; they cannot earn space in four pages.
3. **Why planned paths are clinically practical** — deferred to Methods per author note.

## Open issue to resolve before submission

The reported models were **trained** on anatomies generated *before* the clearance
calibration and **evaluated** on calibrated ones — so this is "performance on anatomy
matched to patient calibre", not strict in-distribution generalization. Either retrain on
calibrated anatomy (one run, cleanest) or state the shift in one Methods sentence.
