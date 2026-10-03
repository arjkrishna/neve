#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\graft_v5_reg_dryall.sh (= /sp/v5_va/graft_v5_reg_dryall.sh in the build containers)
# graft_reg: dry-run EVERY donor unit in v5_reg/donors against all 223 v4 anatomies (graft_v5_reg.py, default mode:
# follow handedness, all v4 + new gates). Idempotent: graft_v5_reg.py skips anatomies whose JSON exists (pass FORCE=1 to
# redo). Runs inside the agent's own container graft_reg_w (eve-training-fixed, --cpus 6). Log: v5_reg/dry/_dryall.log
# v5_assign extension (2026-10-02; defaults unchanged): CONTAINER (default graft_reg_w), DONORS (registry folder name under
# v5_reg, default donors), DRY (dry folder name under v5_reg, default dry), WORKERS (default 6); a per-unit done marker
# <DRY>/<unit>/_done.json is written by graft_v5_reg.py; units whose marker exists are skipped unless FORCE=1.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
CONTAINER="${CONTAINER:-graft_reg_w}"
DONORS="${DONORS:-donors}"
DRY="${DRY:-dry}"
WORKERS="${WORKERS:-6}"
REG=/d/Arjun/neve/carotid_data/va_sources/v5_reg
LOG="$REG/$DRY/_dryall.log"
mkdir -p "$(dirname "$LOG")"
UNITS="${UNITS:-$(ls $REG/$DONORS)}"
echo "== $(date '+%F %T') dryall start container $CONTAINER donors $DONORS dry $DRY units: $UNITS" >> "$LOG"
for u in $UNITS; do
  [ -f "$REG/$DONORS/$u/record.json" ] || continue
  if [ -z "$FORCE" ] && [ -f "$REG/$DRY/$u/_done.json" ]; then echo "   unit $u done earlier (skip)" >> "$LOG"; continue; fi
  t0=$(date +%s)
  docker exec -w /sp/v5_va "$CONTAINER" python3 /sp/v5_va/graft_v5_reg.py --unit "$u" --workers "$WORKERS" \
      --donor-root "/repo/carotid_data/va_sources/v5_reg/$DONORS" --dry-dir "/repo/carotid_data/va_sources/v5_reg/$DRY" \
      ${FORCE:+--force} >> "$LOG" 2>&1
  echo "   unit $u rc $? $(( $(date +%s) - t0 )) s" >> "$LOG"
done
echo "== $(date '+%F %T') dryall done" >> "$LOG"
