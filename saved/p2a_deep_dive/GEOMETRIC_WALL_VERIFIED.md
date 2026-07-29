# The failure is a GEOMETRIC ARREST, not a policy failure — independently verified

2026-07-29. A 14-agent workflow proposed this; **I re-derived it from the raw eval logs
with my own parser before accepting it.** Verdict: the central claim is CONFIRMED and
stronger than stated for generated anatomies; one corollary is REFUTED.

## 1. CONFIRMED — the real-patient wall at proj_s ≈ 153.4 mm

Parsed per-episode max `proj_s` from the real-patient eval worker logs:

| model | n | succ | fail | failed arrest: p25 / median / p75 | in [152.5,154.5] | modal value |
|---|---|---|---|---|---|---|
| v1b ckpt757854 | 98 | 29 | 69 | 153.3 / **153.4** / 153.4 | **63/69 = 91%** | 153.4 (51×) |
| v1bp ckpt514264 | 98 | 29 | 69 | 153.3 / **153.3** / 153.3 | **63/69 = 91%** | 153.3 (61×) |
| **H0 = hand-written heuristic** | 98 | 24 | 74 | 37.7 / **153.3** / 153.3 | 41/74 = 55% | 153.3 (23×) + 153.4 (18×) |

**`max solved path_len = 156.2 mm` — IDENTICAL for all three models.**

Why this is decisive: **H0 is a hand-written centerline follower with no learned
parameters.** A geometric station that arrests the heuristic and two independently
trained policies at the same tenth of a millimetre cannot be a policy defect. The
arrest is a property of the environment.

**And it is NOT at the siphon.** 153.4 mm sits 7 mm past the ICA-mid cut (146 mm) and
**57 mm BEFORE the siphon band begins (210 mm)**. The headline "siphon 0/30" is
mislabelled: nothing ever reaches the siphon on the real patient to fail there.

## 2. CONFIRMED AND STRONGER — generated anatomies are walled too, per-anatomy

Pooled arrest on the 50-anatomy eval scatters (SD 34 mm), which looks like ordinary
navigation failure. It is not. Grouping failures by the anatomy branch-hash logged in
`EPISODE_START` shows each anatomy has its OWN deterministic station:

| run | anatomies w/ ≥2 deep failures | **walled** (within-anatomy SD < 2 mm) | failure mass in walled anatomies |
|---|---|---|---|
| v1b | 15 | **12 (80%)** | 24/53 = **45%** |
| v1bp | 16 | **11 (69%)** | 22/54 = **41%** |

Walled stations span 142.5–222.8 mm (median ~154.6) — one wall per anatomy, at
different places, which is exactly why the pooled distribution looked like scatter.

**The clinching detail: the same anatomies wall at the same station for BOTH models.**

| anatomy | v1b | v1bp |
|---|---|---|
| 11d068 | 152.1 ± 0.00 | 152.1 ± 0.00 |
| bfadcd | 166.3 ± 0.00 | 166.3 ± 0.00 |
| 8ba8a3 | 157.2 ± 0.00 | 157.2 ± 0.00 |
| dd7d9c | 160.8 ± 0.00 | 160.8 ± 0.00 |

Two differently-trained policies arresting at an identical station on the same mesh is
model-independent by construction. **~40–45% of the 50-anatomy failure mass is
environment, not policy.** (Caveat: n=2 failures per anatomy — a zero SD from two
samples is weak alone; the cross-model agreement is what carries this.)

## 3. REFUTED — the "clean cut" corollary

The workflow claimed *max solved path_len 156.2 / min failed 164.2 — a clean cut*,
i.e. every real-patient failure is a beyond-wall target. **False.** Measured min failed
`path_len` is **103.2 mm** (v1b and v1bp) and 79.1 mm (H0) — well short of the wall,
so those are genuine navigation failures. Accounting: of 69 real-patient failures,
**63 are wall-limited and ~6 are real policy failures at shallow depth.** The wall
dominates but does not explain everything, and the true RL headroom on the real
patient is small but non-zero.

## 4. What this means for every prior conclusion

- **The eval matrix measures the mesh as much as the policy.** v1b 57.1% vs v1bp 55.1%
  on 50 anatomies is ~40% composed of a shared geometric artifact. At n=98 (SE 5.0 pp
  per arm, 7.0 pp on a difference) that comparison was never resolvable anyway.
- **"Siphon 0/30" must be retired as a claim.** It is an arrest at 153.4 mm, before the
  siphon. The paper cannot state a siphon result on this evidence.
- **The reward-pair null and the dither null are both re-explained.** Neither could
  possibly have moved a geometric wall. They were well-designed tests aimed at a
  bottleneck that was not there.
- **The stuck/recovery analysis needs re-stratification.** The r = −0.82
  avoidance-not-escape finding pools walled and non-walled episodes; a "stall" against
  a hard geometric wall is not the same event as a stall in navigable vessel. That
  finding is now provisional until recomputed within the non-walled stratum.

## 5. Two candidate mechanisms, both indefensible regardless

Found by code audit this session (not yet discriminated — that is the next test):
1. **Collision discretization.** `jshaped.py:50` defaults
   `collis_edges_per_mm_straight=0.1`, unoverridden, so an 885 mm shaft gets 89
   collision edges = **9.94 mm straight chords**, collided with `proximity=0.0`,
   inside a 1.2–2.5 mm lumen. A straight 10 mm chord cannot follow a genu.
2. **Mesh erosion.** `meshing.py:41-56` voxelizes at [0.6,0.6,0.9] mm, applies
   `gaussian_smooth(1)` **twice**, then `decimate(0.99)` — offline replication measures
   25–43% distal lumen shrink (r_eff 0.52–0.62 mm where r_true is 1.25–1.40 mm).

Both predict an arrest near 153 mm, so the number alone cannot discriminate them.

## 6. Next test (cheap, decisive, no training)

**H0 arrest-station A/B on the real patient**: checkpoint0, ≥40 deep episodes per cell,
2×2 over `collis_edges_per_mm_straight {0.1, 0.5}` × mesher {current, fine}. Score the
**arrest station**, not success% — the station is deterministic to 0.1 mm, so a single
cell has effectively infinite power where success% at n=30 has almost none.
~4–6 GPU-hours, no training, invalidates no checkpoint.

**Kill criterion:** if neither cell moves p95 arrest more than 5 mm off 153.4, both
geometric hypotheses are dead → test device stiffness (both devices are at
`young_modulus` 1e3, ~80× softer than eve's own validated 17e3/80e3), and if that also
fails, re-scope the paper to cervical-to-petrous ICA with the cavernous siphon stated
as an open problem.
