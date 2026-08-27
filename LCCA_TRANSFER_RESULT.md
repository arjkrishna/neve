# LCCA different-vessel transfer — result

2026-08-26. Uncommitted. 98 episodes per arm, deterministic policy, 600-step cap, 5 mm
success threshold, real patient mesh with the original segmented surface pinned.

## Result

| arm | vessel | model | success | 95% CI (Wilson) |
|---|---|---|---|---|
| **1** | **LCCA** — different vessel | ckpt2002292 (best) | **0/98 = 0.0%** | [0.0, 3.8] |
| **2** | RCCA-internal — **control** | ckpt2002292 | **68/98 = 69.4%** | [59.6, 77.7] |
| **3** | **LCCA** — different vessel | ckpt514264 (earlier) | **13/98 = 13.3%** | [7.9, 21.4] |

Arm identity verified two independent ways, because a first pass at reading the result
files mislabelled them: (a) path_len span — LCCA targets reach 245.5 mm, RCCA-internal only
213.6 mm; (b) the branch actually navigated in the step logs — 57,930 LCCA steps in arm 1
versus 40,927 RCCA steps in arm 2.

## The control arm is what makes this readable

Starting inside a branch collapses the planned path from two segments to one. Verified in
source: `FixedPathfinder` takes its same-branch shortcut, so `trunk_idx ==
target_daughter_idx`; LocalGuidance feature 25 (`is_in_trunk`) pins at 1.0 and feature 26
(`is_on_target_daughter`) pins at 0.0 for the entire episode, while the privileged block
simultaneously asserts "tip is in the target daughter". Fourteen of 51 guidance dimensions
become episode-long constants, two of them inverted relative to training. That joint pattern
occurs nowhere in the training distribution, and nothing errors.

**Arm 2 makes that collapse common-mode.** Identical machinery, identical one-branch
topology, identical offset from the branch's proximal junction (19.34 mm vs 18.75 mm),
matched shortest task (74.0 mm) — on the vessel the policy was trained on. It scores 69.4%
against the published 75.5% from the normal (11)→RCCA insertion.

**So the collapse costs roughly 6 points, not the result.** The LCCA failure is therefore
attributable to the vessel, not to the experimental setup. Without arm 2 this would have
been uninterpretable.

## What it means

**The policy has not learned general CCA→ICA navigation.** It has learned the RCCA course.
On a different carotid of the *same patient* — same devices, same observation semantics,
same mesh, one branch away — the best checkpoint solves nothing at all.

The LCCA is not a degenerate target. It is passable end-to-end (0 of 252 stations below the
guidewire radius) and only slightly tighter than the RCCA branch (median clearance 1.98 mm
vs 2.14 mm). Episodes start correctly (`on_path=1`, `nearest_named=LCCA`) and then advance
only ~14–25 mm of a 103–144 mm path before arresting with cross-track at the wall.

**Continued training made transfer worse.** The earlier checkpoint scores 13.3% on the LCCA;
the later one, which is better on the RCCA (75.5% vs 72.4%) and on held-out synthetic
anatomy (84.7% vs 83.7%), scores **0%**. Fisher exact on 13/98 vs 0/98 gives p ≈ 0.0003.
Training past ~0.5M steps on RCCA-only anatomies bought RCCA competence and spent
cross-vessel generality.

## Consequence for the paper

The generalization claim needs re-scoping. Procedural variation *around one vessel* produced
competence *on that vessel* and on deformations of it — not transferable carotid navigation.
The held-out synthetic anatomies are deformations of the RCCA, so 84.7% measures robustness
to tortuosity and calibre variation of a single course, which is a weaker claim than
"generalizes across anatomy".

This also sharpens the case for the parked mesh-generator work in a new direction: the
generator varies one branch of one patient. Genuine anatomy generalization needs training
anatomies that differ in *course and topology*, not only in the tortuosity of a fixed course.

## Reproduce

```bash
docker run --rm -v "D:\Arjun\workspace\neve\eve:/opt/eve_training/eve" \
  -v "D:\Arjun\workspace\neve\eve_bench:/opt/eve_training/eve_bench" \
  -v "D:\Arjun\workspace\neve\saved:/opt/eve_training/results" \
  -v "D:\Arjun\workspace\neve\monitoring\lcca_preflight.py:/tmp/pf.py" \
  eve-training-fixed python3 /tmp/pf.py 2          # V0+V1 preflight, ~1 min

bash launch_lcca_transfer.sh lcca          <ckpt> 98
bash launch_lcca_transfer.sh rcca_internal <ckpt> 98   # the control — not optional
bash launch_lcca_transfer.sh rcca_baseline <ckpt> 98
```

Code (all default-off; RCCA behaviour byte-identical without the new flags):
`--insert_inside_branch {none,RCCA,LCCA}` and `--insert_point_idx` in
`training _scripts/eval_anatomies.py`; `monitoring/lcca_preflight.py`;
`launch_lcca_transfer.sh`.

## Open

- Arm 3's control (RCCA-internal with ckpt514264) was not run — worth ~1 h if the
  checkpoint-ordering claim is to be stated rigorously rather than by inference from arm 2.
- Failure arrests were not yet bucketed against `results/lcca_clearance_profile.npy`. The
  LCCA has tight bands at arclength 130, 223 and 266 mm; arrests clustering there would be
  geometry rather than failed transfer. Given arrests occur at ~20 mm, well proximal to all
  three, this is unlikely to change the conclusion — but it is the check that was skipped
  last time and it is cheap.
