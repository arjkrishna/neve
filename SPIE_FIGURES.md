# SPIE figure specifications

Four pages allows roughly 3–4 figures. Ordered as the author specified: anatomy
generation first, then the control problem, then behavior, then results.

---

## FIG. 1 — Procedural anatomy generation within a grounded vasculature
*Referenced from Methods §2.1. This is the paper's lead figure: it carries the first of
the two contributions and must be readable without the caption.*

**Layout: three panels, left to right.**

**(a) The grounded vasculature.** The full loaded arterial tree rendered faintly in grey,
with the navigated branch (RCCA → cervical ICA → cavernous segment) highlighted. Mark
and label the two pinned points: the ostium/bridge junction proximally and the terminus
distally. Caption these as *anchors* — everything outside them is byte-identical across
generations, which is what "grounded" means and what keeps the junctions anatomically
consistent.

**(b) The perturbation.** The navigated centerline drawn as a single bold curve with
4–6 generated variants overlaid in a light colour, showing the envelope opening from
zero at the proximal anchor, reaching full amplitude through the cavernous course, and
closing to zero at the distal anchor. A small inset showing the envelope weight *w* vs
arclength makes the mechanism legible at a glance. This panel is what a reader will
point at when they ask "how varied is it?"

**(c) Surface and audit.** One generated variant as a meshed surface, with a clearance
map along the navigated branch (colour = local lumen clearance) and the accept threshold
drawn as a line. Adjacent, a small histogram of clearance for accepted vs rejected
anatomies with the patient-derived vessel's clearance marked. This panel does the work of
the whole calibration argument — that variation was not bought by widening the vessel.

**Draft caption.** "Procedural generation of vessel anatomy. (a) The patient-derived
arterial tree is fixed except for the navigated branch; the ostium and terminus are
pinned so both junctions remain anatomically consistent. (b) The navigated centerline is
displaced by a band-limited field under an envelope that is zero at both anchors and
full through the cavernous course, yielding variation in tortuosity and course while
preserving connectivity; six generated variants shown. (c) Each generated surface is
admitted only if lumen clearance along the navigated branch meets the patient-derived
calibre, so anatomical variation is not obtained by widening the vessel."

> **Data to produce it:** variants from the generator at several seeds; clearance from
> `monitoring/mesh_clearance.py`; the patient reference from the original segmented
> surface.

---

## FIG. 2 — The control problem: devices, state, and reward
*Referenced from Methods §2.2. Explains what the agent sees and is paid for.*

**Layout: two panels.**

**(a) Devices in the vessel.** Guidewire and catheter rendered inside a cut-away vessel
at a tight curve, annotated with the four actuated degrees of freedom (translation and
rotation, per device) as arrows. Call out the deployable observation — projected device
tracking and its relation to the planned route — versus the privileged simulator
quantities (contact forces, tensions, full configuration) with a visual separator, e.g.
solid outline for what the policy sees, dashed for what only the critic/teacher sees.
**This panel is where the paper's central asymmetry becomes visible**, so the split must
be unmistakable.

**(b) Reward and progress.** A schematic of the navigated route with arclength progress
along it, showing the dense progress term, the shaping term, and the terminal success
condition at the target threshold. A small inset of a buckling event with the
corresponding reward trace makes the recovery case concrete.

**Draft caption.** "The dual-device control problem. (a) Guidewire and catheter are each
actuated in translation and rotation. The deployable policy observes projected device
tracking and its relation to the clinician's planned route (solid); contact forces,
tensions and full device configuration exist only in simulation and are used to
supervise (dashed). (b) Reward is dense in arclength progress along the planned route
with terminal reward at the target; the inset shows a buckling event, where progress
stalls while the device is still being advanced."

---

## FIG. 3 — The two navigation modes and their competition
*Referenced from Methods §2.4 / Results. This is the empirical motivation for the
teacher–student design.*

**Layout: one wide panel, two y-axes, plus a small companion.**

**(a)** Training progress on the x-axis. Episode success rising; per-event recovery
success falling; stalled-episode share falling. Three lines, clearly distinguished. The
crossing pattern is the entire argument — as path-following improves, stalls leave the
data and the recovery skill decays.

**(b)** Small adjacent panel: recovery's share of total successes over the same axis,
collapsing. Reinforces the same point in the units a clinician cares about (episodes,
not events).

**Draft caption.** "Competition between the two navigation behaviors. As centerline
following improves, episode success rises while the per-event recovery rate falls and
stalled episodes become rarer: the experience that teaches recovery is removed by the
success of the other behavior. (b) The share of successful episodes that required a
recovery falls correspondingly."

> ⚠ **Blocked on re-analysis.** The current numbers pool geometrically impassable
> ("walled") anatomies with passable ones — a stall against an impassable facet is not
> the same event as a stall in a navigable vessel. **Recompute within the passable
> stratum before this figure is drawn.** See `GEOMETRIC_WALL_VERIFIED.md` §4. If the
> effect does not survive, the figure and the claim both go.

---

## FIG. 4 — Results
*Bar or dot plot: success by depth band (CCA / mid-ICA / cavernous siphon) for held-out
synthetic anatomies and for the patient-derived anatomy, with binomial confidence
intervals. Two groups of three bars. Confidence intervals are not optional at n=98 —
the siphon band is n≈30 and its interval is wide.*

**Draft caption.** "Navigation success by vessel segment on held-out synthetic anatomies
and on the patient-derived anatomy, with 95% binomial confidence intervals. The common
carotid and mid–internal carotid are navigated without failure on synthetic anatomy;
remaining failures are confined to the cavernous segment."

---

## Priority if space forces a cut

Figs. 1 and 4 are load-bearing and cannot be cut — one carries the contribution, the
other the result. Fig. 2 can be reduced to panel (a) alone, since the observation split
is what the reader needs and the reward can be described in text. **Fig. 3 is the first
to go**, both because it is the least essential to the headline claim and because it is
the one currently blocked on re-analysis.
