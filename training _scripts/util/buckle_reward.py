"""Gen-4 anti-buckle potential — privileged reward shaping (pure math).

Potential-based shaping term that prices wire buckling using two signals
the ENV can read exactly but a deployed system could not fully:

  slack_mm   — guidewire slack = inserted_gw - proj.s (mm). Positive when
               the wire has been fed further than its tip has advanced
               along the planned path: the stored bow of a forming buckle
               (or, off-path, fed length past the divergence point —
               intentional overlap with the Δoff_arc shaping: both are
               symmetric, so retract earns both back). Normal navigation
               sits at/below zero (chord-cutting makes the wire path
               SHORTER than the centerline), hence the dead-band.
  contact_mm — mean |position - free_position| over all SOFA beam nodes
               (mm): the collision-response constraint correction, i.e. a
               contact-impulse proxy (privileged tail dims 5-6). Chosen
               over dofs.force deliberately — force.value includes the
               INTERNAL elastic bending force, and high bending is
               REQUIRED to conform to the siphon; penalizing it would
               invert into "don't follow tortuosity". |pos - free_pos| is
               nonzero only where the wall pushes back.

The env adds  coef * (phi_t - phi_{t-1})  each step (delta / gamma=1
potential form). Consequences, by construction:
  - Any closed loop in (slack, contact) nets EXACTLY zero — the term
    cannot be farmed by oscillation (the arclength-progress 2x doubling
    lesson).
  - The episode sum telescopes to phi_end - phi_start: forming a buckle
    and recovering is net zero; STARTING buckled (stuck-pool restore)
    and unbuckling is net POSITIVE — the recovery incentive the audit
    found missing.
  - phi is bounded in [-(W_SLACK + W_CONTACT), 0] via input caps, so a
    single-step delta is bounded and a SOFA glitch cannot dominate the
    return. Caps are applied to the INPUTS (inside the potential), never
    to the delta — clipping the delta would break loop-neutrality
    (penalty clipped on the way in, full credit on the way out).

There is deliberately NO on-path/off-path gating: a state-classifier gate
would make phi jump on classifier flicker, which is farmable at the gate
boundary. phi is a pure function of physical state.
"""

# Dead-band: normal advance has slack <= ~0 (chord-cutting); small positive
# jitter from projection discretization must not be priced.
SLACK_DEADBAND_MM = 5.0
# Saturation: obs feature 43 normalizes slack by 50 mm; a 40 mm bow is a
# fully developed fold (fold detector fires after ~10 mm). Beyond the cap
# the gradient is zero — the fold truncation / MaxSteps prices the rest.
SLACK_CAP_MM = 40.0
# PrivilegedState dim 5 normalizes mean contact by 2 mm — same scale here.
CONTACT_CAP_MM = 2.0
# Equal weight to the two channels; phi in [-1, 0]. With coef=1.0 the slack
# channel is worth coef*W/CAP = 0.0125/mm — parity with the 0.01/mm
# arclength progress factor. Launchers currently run coef=0.5 (half-weight,
# conservative first run). The channels are complementary, not competing:
# progress grades TIP arclength motion, this grades stored slack — during a
# buckle the tip is stationary (progress ~0) while slack moves, and during
# clean advance slack is ~0 while the tip moves.
W_SLACK = 0.5
W_CONTACT = 0.5

# RL_IMPROV_16 (v3c) — CATHETER-slack channel. The v3a forensic measured the
# reward pricing GUIDEWIRE slack only, which left 700+ mm catheter coils
# reward-free (and knotting at the RVA shelf) while the taxed alternative
# (guidewire lead) was the winning behavior. cath_slack = inserted_cath -
# arc(cath tip): near zero in clean tracking, 50 mm+ only when the shaft is
# coiling (validated detector: >50 mm = knot at precision 0.97/recall 0.96).
# Dead-band 15 mm absorbs normal catheter lag + projection jitter; cap 150 mm
# keeps the single-step delta bounded (fully-developed coils run 250-740 mm —
# beyond the cap the gradient is zero and MaxSteps prices the rest). Weight 1.0
# at the channel level; the launcher scales the whole channel via its own
# coefficient (cath_slack_coef), separate from buckle_reward_coef, so the two
# shaping channels attribute independently.
CATH_SLACK_DEADBAND_MM = 15.0
# v3c run-1 finding — the ORIGINAL design capped the input at 150 mm (phi_c
# bottoming at -1.0), mirroring the gw channel. That was wrong here: the gw
# channel's tail is priced by the fold detector's truncation, but under
# --relax_failure_truncations NOTHING prices catheter slack past the cap, so
# beyond 150 mm the gradient is exactly zero. Run 1's eval-1 failures sat at
# ~708 mm slack (both devices railed to their ~898 mm insertion caps) — 4.7x
# into that dead zone, with no marginal deterrent.
# Fix: keep the SLOPE identical (1.0 potential per 150 mm of excess, i.e.
# 0.00333/mm at coef 0.5) so behavior below 150 mm is UNCHANGED — the v3c
# seed never exceeds 128.9 mm, so seed rewards are bit-identical and the
# cache stays valid — but extend the linear range to beyond the physical
# device limit (~898 mm insertion). The far clip is a glitch guard only; it
# never binds in reachable states, so the single-step delta stays bounded and
# the term remains a pure function of state (loop-neutral: forming a coil and
# pulling it back out still nets exactly zero).
#   phi_c(708mm) = -(708-15)/150 = -4.62  ->  x0.5 coef = -2.31 return
#   (was -0.5 flat under the old cap — a 4.6x stronger deterrent)
CATH_SLACK_SLOPE_MM = 150.0   # mm of excess slack per 1.0 of potential
CATH_SLACK_LIMIT_MM = 1000.0  # far input clip (glitch guard; > device max)


def buckle_potential(slack_mm: float, contact_mm: float) -> float:
    """Anti-buckle potential phi(slack, contact) in [-(W_SLACK+W_CONTACT), 0].

    Monotonically non-increasing in both inputs; flat inside the slack
    dead-band and beyond the caps. Inputs are raw mm (not normalized).
    """
    slack_ex = min(max(float(slack_mm) - SLACK_DEADBAND_MM, 0.0), SLACK_CAP_MM)
    contact = min(max(float(contact_mm), 0.0), CONTACT_CAP_MM)
    return -(
        W_SLACK * slack_ex / SLACK_CAP_MM
        + W_CONTACT * contact / CONTACT_CAP_MM
    )


def cath_slack_potential(cath_slack_mm: float) -> float:
    """Catheter-slack potential phi_c(cath_slack), <= 0, linear past the band.

    Linear in excess slack beyond the dead-band at a fixed slope (1.0 of
    potential per CATH_SLACK_SLOPE_MM), clipped only at CATH_SLACK_LIMIT_MM
    — far past the ~898 mm physical device limit, so the clip is a glitch
    guard that never binds in reachable states. Same construction rules as
    buckle_potential otherwise: monotonically non-increasing, flat inside the
    dead-band, clip on the INPUT (never the delta) so loop-neutrality holds —
    forming a coil and pulling it back out nets exactly zero, and an episode
    restored INTO a coiled state that un-coils nets positive.
    """
    ex = min(
        max(float(cath_slack_mm) - CATH_SLACK_DEADBAND_MM, 0.0),
        CATH_SLACK_LIMIT_MM,
    )
    return -(ex / CATH_SLACK_SLOPE_MM)
