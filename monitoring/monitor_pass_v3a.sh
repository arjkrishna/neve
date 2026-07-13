#!/usr/bin/env bash
# RL_IMPROV_16 E8 — read-only verification scan for rcca_procedural_v3a.
# Run from the repo root on the machine hosting the container:
#     bash monitoring/monitor_pass_v3a.sh
# Makes NO changes to the run. Checks = the v2 fix-package set PLUS the
# Tier-A experiment gates (E1b weight spread, E2 aux loss, E3 stuck lane).
# Gates and abort criteria: RL_IMPROV_16_EXPERIMENTS.md (Tier A).
set -uo pipefail
C=${CONTAINER:-rcca_procedural_v3a}
BASE=/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben
GLOB="2026-*_rcca_procedural_v3a"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "===== MONITOR SCAN (v3a) ====="
docker inspect "$C" --format 'Running={{.State.Running}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}}' 2>&1

echo ""; echo "[liveness: CSV rows now vs 15s later]"
docker exec "$C" sh -c 'CSV=$(ls -t '"$BASE"'/'"$GLOB"'/diagnostics/csv/losses_trainer_synchron.csv 2>/dev/null | head -1); [ -n "$CSV" ] && wc -l < "$CSV" || echo "no CSV yet"' 2>&1
sleep 15
docker exec "$C" sh -c 'CSV=$(ls -t '"$BASE"'/'"$GLOB"'/diagnostics/csv/losses_trainer_synchron.csv 2>/dev/null | head -1); [ -n "$CSV" ] && wc -l < "$CSV" || echo "no CSV yet"' 2>&1

echo ""; echo "[A. last CSV row — v2 fix checks + NEW E-gates: p99p1 in [5,20], aux_loss O(1)]"
docker exec "$C" sh -c '
CSV=$(ls -t '"$BASE"'/'"$GLOB"'/diagnostics/csv/losses_trainer_synchron.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
  # header: ...,awac_weight_saturation(26),awac_weight_max(27),awac_weight_mean(28),awac_weight_p99p1(29),aux_loss(30),...
  tail -1 "$CSV" | awk -F, "{print \"u=\"\$2\"  alpha=\"\$7\"  ent=\"\$11\"  clamp=\"\$20\"  q1_mean=\"\$12\"  nf=\"\$23\$24\$25\"  awac_mean=\"\$28\"  awac_max=\"\$27\"  p99p1=\"\$29\"  aux_loss=\"\$30}"
  echo "--- alpha band violations (past pretrain) ---"
  awk -F, "NR>1 && \$2>10000 && (\$7<0.0066 || \$7>0.101)" "$CSV" | wc -l
  echo "--- E1b gate: p99p1 recent range (want 5-20; ~1.7 = BC-degenerate; >>20 over-sharp) ---"
  tail -5000 "$CSV" | awk -F, "{if(\$29!=\"\"&&\$29>mx)mx=\$29; if(mn==0||(\$29!=\"\"&&\$29<mn))mn=\$29} END{print \"min=\"mn\" max=\"mx}"
  echo "--- E2 gate: aux_loss recent mean (want O(0.1-2) when znorm on; ~0 = still dead) ---"
  tail -5000 "$CSV" | awk -F, "{if(\$30!=\"\"){s+=\$30;c++}} END{if(c)print s/c; else print \"n/a\"}"
fi
' 2>&1

echo ""; echo "[A2. anti-rail: CLEAN_RAIL_FILTER rejections (few % ok; >30% WATCH)]"
docker logs "$C" 2>&1 | grep -c "CLEAN_RAIL_FILTER"

echo ""; echo "[A3. deterministic probes — freeze + retract-vs-slack + aux R^2 (E8)]"
docker exec -i "$C" python3 - < "$HERE/probe_policy_v3.py" 2>&1 | grep -E "^PROBE|^  \(" || echo "(probe failed)"

echo ""; echo "[eval trajectory (v2 ref: 6.1% -> 30.6% -> 49.0%; GATE: eval1 >= 30.6%)]"
docker exec "$C" sh -c 'CSV=$(ls -t '"$BASE"'/*_rcca_procedural_v3a.csv 2>/dev/null | head -1); [ -n "$CSV" ] && tail -n +3 "$CSV" | awk -F";" "{print \"explore=\"\$2\"  quality=\"\$3\"  speed=\"\$7\"  traj=\"\$8\"  reward=\"\$9}"' 2>&1

echo ""; echo "[E4-prep: stuck-pool harvest count]"
docker exec "$C" sh -c 'ls /opt/eve_training/results/rcca_v3_stuck/*.npz 2>/dev/null | wc -l' 2>&1

echo ""; echo "[progress + recovery + diversity]"
docker logs "$C" 2>&1 | grep -oE "buffer_len=[0-9]+.*update_step=[0-9]+" | tail -1
docker logs "$C" 2>&1 | grep -oiE "WBT|grader_failure_timeout" | sort | uniq -c
docker logs "$C" --tail 3000 2>&1 | grep -oE "pid=[0-9]+" | sort -u | wc -l
docker logs "$C" --tail 25000 2>&1 | grep -oE "reason=[a-z_]+ \| is_clean=[01] \| grader_success=[01]" | sort | uniq -c | sort -rn | head -6

echo ""; echo "[IPC guard]"
docker logs "$C" 2>&1 | grep -c "IPC TIMEOUT"
echo "===== END SCAN ====="
