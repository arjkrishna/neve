#!/bin/bash
# Sequential replay campaign driver.
#   usage: run_campaign.sh <manifest.tsv> <results.tsv> <logdir>
# manifest.tsv columns (TAB): name  launcher  checkpoint  seeds  reps  env(k=v;k=v or -)  extra_flags(or -)
# Each rep runs the launcher with SEED_LIST, NAME=eval_<name>_rep<k>, 16 workers, no snapshots,
# and appends: tag name rep checkpoint record_dir official_success wall_start wall_end to results.tsv
set -u
cd "/d/neve/.claude/worktrees/rl_improv_18_p2"
MANIFEST="$1"; RESULTS="$2"; LOGDIR="$3"; mkdir -p "$LOGDIR"
while IFS=$'\t' read -r name launcher ckpt seeds reps env extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  for r in $(seq 1 "$reps"); do
    tag="${name}_rep${r}"; log="$LOGDIR/$tag.log"; t0=$(date +%H:%M)
    echo "=== $tag  ($ckpt) start $t0 ==="
    (
      export SEED_LIST="$seeds" NAME="eval_$tag" N_WORKER="${N_WORKER:-16}" CHANGE_EVERY=1 SNAPSHOT_MODE=off
      if [ "$env" != "-" ]; then IFS=';' read -ra kvs <<< "$env"; for kv in "${kvs[@]}"; do export "$kv"; done; fi
      if [ "$extra" != "-" ]; then export EXTRA_FLAGS="$extra"; fi
      bash "$launcher" "$ckpt" 98 < /dev/null
    ) > "$log" 2>&1
    rc=$?
    rec=$(grep -oE "TRAJ_RECORD_DIR=[^ ]+" "$log" | head -1 | cut -d= -f2)
    off=$(grep -oE "OFFICIAL[^:]*: [0-9]+/[0-9]+ = [0-9.]+%" "$log" | head -1 | sed -E 's/.*: //')
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\trc=%s\n" "$tag" "$name" "$r" "$ckpt" "$rec" "${off:-?}" "$t0" "$(date +%H:%M)" "$rc" >> "$RESULTS"
    echo "    -> ${off:-no result} (rc=$rc) rec=$rec"
  done
done < "$MANIFEST"
echo "CAMPAIGN_DONE $(date +%H:%M)"
