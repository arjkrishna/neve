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
