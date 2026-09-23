#!/bin/bash
# Keep rendering campaign runs as their results lines appear (every 5 min), for all three parts.
# Stops after one final pass once chain2.log says ALL_PARTS_DONE, or when render_stop exists.
set -u
cd "/d/neve/.claude/worktrees/rl_improv_18_p2"
C="C:/Users/akrish41/AppData/Local/Temp/claude/d--neve--claude-worktrees-rl-improv-18-p2/950f7cc3-fea2-4106-bb7b-072e68d5b1da/scratchpad/campaign"
B="saved/eve_paper/neurovascular/full/mesh_ben/frontier_campaign"
FR="50,122,80,118,44,139,8,120,134,55,162,108,23,93,154,2,39,140,171,124,155,161,91,148,16,115,18,37,47,10,31,34,73,43,110,3,70,81,167,1,48,138,21,69,63,89,97"
HOST=$(seq -s, 900000 900097)
while true; do
  for part in "part1:anim_carotid:$FR" "part2:anim_host:$HOST" "part3:anim_procedural:$FR,$HOST"; do
    IFS=: read -r p out seeds <<< "$part"
    res="$C/${p}_results.tsv"
    [ -f "$res" ] || continue
    # anything not rendered yet?
    todo=0
    while IFS=$'\t' read -r tag rest; do [ -n "$tag" ] && [ ! -d "$B/$out/$tag" ] && todo=1; done < "$res"
    if [ "$todo" = 1 ]; then
      echo "=== $(date +%H:%M) pass $p -> $out ==="
      bash "$C/render_campaign.sh" "$res" "$B/$out" "$seeds" 1 all
    fi
  done
  if [ -f "$C/render_stop" ]; then echo "stopped by render_stop"; break; fi
  if grep -q PART3B_DONE "$C/chain3.log" 2>/dev/null; then
    # one more pass already happened above after the last results line; done
    echo "PART3B_DONE seen $(date +%H:%M)"; break
  fi
  sleep 300
done
echo "RENDER_LOOP_DONE $(date +%H:%M)"
