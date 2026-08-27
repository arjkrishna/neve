#!/bin/bash
# ============================================================================
# LCCA DIFFERENT-VESSEL TRANSFER EXPERIMENT
#
# Question: can an RCCA-trained policy navigate a completely different CCA->ICA
# vessel (the LEFT common carotid of the same patient), standing in for "another
# patient"? Same mesh, same devices, same observation semantics — only the
# navigated vessel changes.
#
# ---------------------------------------------------------------------------
# WHY THERE IS A CONTROL ARM, AND WHY IT IS NOT OPTIONAL
# ---------------------------------------------------------------------------
# There is no bridge branch feeding the LCCA — its point[0] IS the aortic-arch
# junction, bit-identical to (11)[0] (verified, ||diff|| = 0.000000 mm). So the
# wire must start INSIDE the LCCA. That makes the planned path a SINGLE branch,
# where every RCCA training episode had TWO ((11) -> RCCA across a commit fork).
#
# Consequence, verified in the source: FixedPathfinder takes its same-branch
# shortcut, so trunk_idx == target_daughter_idx. LocalGuidance feature 25
# (is_in_trunk) pins at 1.0 and feature 26 (is_on_target_daughter) pins at 0.0
# for the whole episode — while the privileged block asserts "tip is in the
# target daughter". That joint pattern does not occur anywhere in training, and
# 14 of 51 guidance dims become episode-long constants. Nothing errors.
#
# ARM 2 makes that collapse COMMON-MODE: identical machinery, identical
# one-branch topology, identical offset from the branch's proximal junction —
# but on the RCCA, the vessel the policy was trained on. Then:
#   ARM 2 ~= published RCCA number -> the collapse is benign, and ARM 1 vs ARM 2
#                                     is a clean zero-shot vessel-transfer read
#   ARM 2 drops sharply            -> we measured the topology change, not the
#                                     vessel, and ARM 1 is uninterpretable
#
# Without ARM 2 the headline cannot be attributed to the vessel. Run it.
#
# ---------------------------------------------------------------------------
# PRIMARY COMPARISON: ARM 1 vs ARM 2, on path_len in [74, 218] mm.
# NOT against the published RCCA-from-(11) numbers (75.5% / 72.4%) — that
# comparison confounds "different vessel" with "path collapsed to one branch".
# ---------------------------------------------------------------------------
#
# Preflight (run first, ~1 min, no SOFA):
#   docker run --rm \
#     -v "D:\Arjun\workspace\neve\eve:/opt/eve_training/eve" \
#     -v "D:\Arjun\workspace\neve\eve_bench:/opt/eve_training/eve_bench" \
#     -v "D:\Arjun\workspace\neve\saved:/opt/eve_training/results" \
#     -v "D:\Arjun\workspace\neve\monitoring\lcca_preflight.py:/tmp/pf.py" \
#     eve-training-fixed python3 /tmp/pf.py 2
#
# Preflight results (2026-08-24), which set the constants below:
#   LCCA idx 2 : s=18.75mm r=4.15mm clearance=3.15mm  176 targets  74.0-248.1mm
#   RCCA idx 15: s=19.34mm r=2.50mm                   146 targets  74.0-218.2mm
#   LCCA passable end-to-end: 0 stations below the guidewire radius.
#     median clearance 1.98mm vs the RCCA branch's 2.14mm — LCCA is slightly
#     TIGHTER, so this is not an easier vessel.
#   Tight bands (<2x wire radius) at LCCA arclength 130, 223, 266 mm — bucket
#     arrests against results/lcca_clearance_profile.npy, NOT against the
#     RCCA-calibrated CCA/ICA-mid/siphon cuts, which are meaningless here.
#
# usage: bash launch_lcca_transfer.sh <arm> <checkpoint-in-container> [n_eps]
#        arm = lcca | rcca_internal | rcca_baseline
# ============================================================================
set -e
export MSYS_NO_PATHCONV=1

ARM="${1:?usage: launch_lcca_transfer.sh <lcca|rcca_internal|rcca_baseline> <ckpt> [n_eps]}"
CKPT="${2:?checkpoint path inside the container}"
N_EPISODES="${3:-98}"

case "$ARM" in
  lcca)
    FLAGS="--real_patient_anatomy --insert_inside_branch LCCA --insert_point_idx 2 --target_min_arclength_mm 92.5"
    NAME="${NAME:-lcca_transfer}" ;;
  rcca_internal)
    FLAGS="--real_patient_anatomy --insert_inside_branch RCCA --insert_point_idx 15 --target_min_arclength_mm 93.1"
    NAME="${NAME:-rcca_internal_control}" ;;
  rcca_baseline)
    FLAGS="--real_patient_anatomy"
    NAME="${NAME:-rcca_baseline}" ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac

# ENV_FLAGS deliberately left at the launcher default
# ("--relax_failure_truncations --buckle_reward_coef 0.5"), which is exactly
# what produced the published 75.5% / 72.4% real-patient numbers. The v1bp
# reward flags (--cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight
# 0.5) change the LOGGED REWARD only, never success or the observation, so
# adding them here would make reward incomparable to the prior runs for no gain.

echo "=========================================================================="
echo " ARM       : $ARM"
echo " FLAGS     : $FLAGS"
echo " CHECKPOINT: $CKPT"
echo " EPISODES  : $N_EPISODES"
echo "=========================================================================="
echo
echo " HARD GATE after the run — check BEFORE reading any success number:"
echo "   1. header contains '[eval-anat] NAV BRANCH=' and 'GUARD ok:'"
if [ "$ARM" != "rcca_baseline" ]; then
echo "   2. every episode path_len in [74, 250] mm."
echo "      < 70  => min_arclength did not take; targets sampled near the tip"
echo "      > 280 => insertion resolved to the arch; the whole path is wrong"
else
echo "   2. insertion prints as (11)[2] = (17.791, 14.554, 398.168)"
fi
echo
sleep 2

NAME="$NAME" EXTRA_FLAGS="$FLAGS" bash launch_eval_anatomies.sh "$CKPT" "$N_EPISODES"
