#!/bin/bash
# Render every run listed in a results.tsv into <out>/<tag>/..., frontier seeds only,
# then rebuild the cross-checkpoint by_seed index.
#   usage: render_campaign.sh <results.tsv> <out_dir> <seeds_csv> [jobs] [types]
set -u
cd "/d/neve/.claude/worktrees/rl_improv_18_p2"
RES="$1"; OUT="$2"; SEEDS="$3"; JOBS="${4:-1}"; TYPES="${5:-all}"
while IFS=$'\t' read -r tag name rep ckpt rec off t0 t1 rc; do
  [ -z "$tag" ] && continue
  host="${rec/\/opt\/eve_training\/results16/D:\/neve\/.claude\/worktrees\/rl_improv_16_resume\/saved}"
  host="${host/\/opt\/eve_training\/results/D:\/neve\/.claude\/worktrees\/rl_improv_18_p2\/saved}"
  if [ -d "$OUT/$tag" ]; then echo "skip $tag (rendered)"; continue; fi
  echo "=== render $tag from $host ==="
  python monitoring/traj/traj_render.py "$host" "$OUT" --block "$tag" --types "$TYPES" --seeds "$SEEDS" --movie gif --jobs "$JOBS" 2>&1 | grep -vE "Warning|warn" | tail -2
done < "$RES"
python monitoring/traj/traj_render.py --reindex "$OUT"
