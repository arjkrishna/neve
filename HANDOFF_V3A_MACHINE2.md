# Handoff — Run `rcca_procedural_v3a` on a second machine

**Audience:** an agent (or engineer) setting up and running the RL_IMPROV_16
Tier-A experiment on a second computer, while machine 1 keeps the `v2`
control run alive. Read top-to-bottom before launching. Author: the
RL_IMPROV_15/16 session, 2026-07-12. Repo branch **`rl_improv_15`**, at
commit **`0cd073c`** or later.

Companion docs (read in this order if you need depth):
[RL_IMPROV_16_EXPERIMENTS.md](RL_IMPROV_16_EXPERIMENTS.md) (the experiment
plan + gates), [RL_IMPROV_15_CHANGES.md](RL_IMPROV_15_CHANGES.md) (Gen-4 +
fix-package reference), `saved/monitor_rcca_procedural.md` (machine-1 only —
the running v1/v2 forensic log).

---

## 1. Context — what you are running and why

Autonomous **guidewire+catheter navigation** (4 continuous action dims) in a
**SOFA** physics sim. Each of 16 workers trains on a **procedurally varied
RCCA/siphon anatomy** (per-worker meshes, re-randomized every 10 episodes);
eval is a fixed **held-out** RCCA. Wire starts at the RCCA ostium; success =
deterministic policy reaches a deep target on