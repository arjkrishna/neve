"""Gen-4 #6 — escapability decision for the stuck-state restore pool (pure).

The recovery-generalization audit flagged that stuck checkpoints are captured
at the fold==10 / off-path==25 triggers with NO guarantee the state is
recoverable. A genuinely wedged state in the restore pool teaches the critic
"recovery is impossible" (V(stuck) = failure-and-nothing-after), poisoning
the very skill the curriculum is meant to build.

screen_stuck_pool.py restores each checkpoint, runs a scripted pure-retract
for K steps, and passes the measured before/after signals here. This module
holds ONLY the verdict logic so it can be unit-tested without SOFA.

Signals (all mm; measured by the screener from the restored env):
  slack_start  = inserted_gw - proj.s at restore (the stored buckle bow)
  slack_end    = same after K retract steps
  retract_mm   = inserted_gw_start - inserted_gw_end (how much wire the
                 scripted retract actually pulled back; ~0 => mechanically
                 wedged, the below-zero mask or a physical jam ate the
                 command)
  reason       = the capture reason from the checkpoint sidecar
                 ("fold" | "offpath")

Verdict:
  - A state the retract cannot MOVE (retract_mm < MIN_RETRACT_MM) is
    mechanically wedged => NOT escapable, whatever the reason. These are the
    exact states that teach "recovery impossible".
  - fold: additionally the buckle must actually relax — slack drops by
    FOLD_SLACK_DROP_MM, or ends below FOLD_SLACK_ABS_MM. A wire that
    retracts but keeps its full bow is not recovering.
  - offpath: retractability is sufficient — from a state the wire can back
    out of, the policy can return toward the divergence fork and re-try.
    (Pure retract can't re-steer onto the path, so we don't require
    on_path_end; we only require that escape is mechanically possible.)
"""

# Minimum wire pull-back (mm over the K-step retract) for the state to be
# considered mechanically escapable rather than jammed.
MIN_RETRACT_MM = 2.0
# fold: the buckle bow must relax by this many mm ...
FOLD_SLACK_DROP_MM = 5.0
# ... or end below this absolute slack (already near-unbuckled).
FOLD_SLACK_ABS_MM = 8.0

# ----------------------------------------------------------------------------
# Restore-fidelity gate (the user's point: SOFA restore of a HIGH-ENERGY
# buckled state is imperfect — the bow springs during settle, contact isn't
# fully captured, the wire can pop free or penetrate). Before we even judge
# escapability we must confirm the restored state IS the captured state,
# else a sprung buckle reads as trivially "escapable" (false positive) and a
# popped/penetrated state as garbage. We compare the post-restore inserted
# length and slack against what the checkpoint recorded at capture.
# ----------------------------------------------------------------------------
# Inserted-length mismatch that means the restore did not land (e.g. the
# dof_positions never applied and the wire sits at the ostium baseline —
# hundreds of mm off, so this is a loose bound that only catches gross
# failure, not normal settle).
RESTORE_INSERTED_TOL_MM = 10.0
# Slack (the buckle bow) mismatch beyond normal settle relaxation. A faithful
# restore + 3 settle steps loses a few mm; springing from a 40 mm bow to ~5
# is gross infidelity.
RESTORE_SLACK_TOL_MM = 8.0


def restore_faithful(
    captured_inserted_mm: float,
    restored_inserted_mm: float,
    captured_slack_mm: float,
    restored_slack_mm: float,
    inserted_tol_mm: float = RESTORE_INSERTED_TOL_MM,
    slack_tol_mm: float = RESTORE_SLACK_TOL_MM,
) -> bool:
    """True if the restored wire matches the captured stuck state on both fed
    length and stored bow (slack). A mismatch means SOFA restore did not
    faithfully reproduce the buckle, so the checkpoint is unusable for the
    recovery curriculum regardless of its retract behavior."""
    return (
        abs(float(restored_inserted_mm) - float(captured_inserted_mm))
        <= inserted_tol_mm
        and abs(float(restored_slack_mm) - float(captured_slack_mm))
        <= slack_tol_mm
    )


def is_escapable(
    reason: str,
    slack_start: float,
    slack_end: float,
    retract_mm: float,
    on_path_end: bool = False,
) -> bool:
    """Return True if the scripted retract shows the stuck state is
    recoverable. See module docstring for the rationale per reason."""
    if retract_mm < MIN_RETRACT_MM:
        return False  # mechanically wedged
    if str(reason).lower().startswith("fold"):
        return (
            slack_end <= slack_start - FOLD_SLACK_DROP_MM
            or slack_end <= FOLD_SLACK_ABS_MM
        )
    # offpath / other: retractable is enough (or already back on path).
    return True


def escape_metrics(reason, slack_start, slack_end, retract_mm, on_path_end=False):
    """Verdict + the numbers behind it, for per-checkpoint logging."""
    return {
        "reason": reason,
        "slack_start": round(float(slack_start), 3),
        "slack_end": round(float(slack_end), 3),
        "slack_drop": round(float(slack_start) - float(slack_end), 3),
        "retract_mm": round(float(retract_mm), 3),
        "on_path_end": bool(on_path_end),
        "escapable": is_escapable(
            reason, slack_start, slack_end, retract_mm, on_path_end
        ),
    }
