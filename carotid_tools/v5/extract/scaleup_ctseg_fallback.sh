#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ctseg_fallback.sh (= /sp/v5_va/scaleup_ctseg_fallback.sh in the build containers)
# scaleup_ctseg_fallback.sh <source> <case> [case ...]   (Git Bash on the host; scaleup_fixes, 2026-10-02 08:55) -- PIPELINE v2,
# the per-vessel FALLBACK of fix 2 (the ct_seg4 C2-exit pinch fix). Run it after "scaleup_v2_post.sh <src> post2 <case>"
# with the SAME environment (it calls scaleup_v2_post.sh again); on a case without the problem it only checks (~5 s).
#   check   for each VA of ct_seg/<PILOT_SEG_TAG>: the VMTK MISR centreline of the stage-A surface (<v>_stageA_vmtk.json) must
#           span >= 50 % of the label's s-range (seg_qc.json seeds_fix). A VA that fails it AND had pinch-fix events is
#           re-cast with the PILOT method (no pinch fix) -- Song 138 R: the fix extended the label 131 -> 154.5 mm but
#           vmtkCenterlines returned 0.7 mm, so measure saw no wall; without the fix the centreline is 131.9 mm.
#   act     writes ct_seg/pinch_skip_<tag>.json {"vessels": [...]} (read by scaleup_ct_seg.py, block v2-c), keeps the failed
#           wall as ct_seg/<tag>_pinchfail, then V2_FORCE="ctseg measure protrusion fig" scaleup_v2_post.sh <src>
#           "ctseg measure protrusion fig collect grade" <case>; re-checks and records ct_seg/pinch_fallback_<tag>.json.
# A VA that fails the check WITHOUT pinch events is reported only (not a pinch problem; the tally marks the wall invalid).
# Idempotent: a vessel already in the skip file is not re-done.
S=$1; shift
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPU=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
TAG=${PILOT_SEG_TAG:-ct_seg4}
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/$S
check() {   # $1 case -> prints "BAD <v> ..." / "OK <v> ..." / "NOPINCH <v> ..." lines
  docker run --rm --memory 512m -v "D:\\Arjun\\neve:/repo" python:3 python3 -c "
import json, os
d = '/repo/carotid_data/va_sources/derived/thin_slice/$S/$1/ct_seg/$TAG'
q = json.load(open(d + '/seg_qc.json'))
ev = (q.get('pinch_fix') or {}).get('events') or []
skip = set((q.get('pinch_fix') or {}).get('skipped_vessels_fallback') or [])
for v in ('rva', 'lva'):
    sf = (q.get('seeds_fix') or {}).get(v)
    if not sf:
        continue
    a, b = sf['label_s_range_mm']
    p = os.path.join(d, v + '_stageA_vmtk.json')
    L = json.load(open(p))['length_mm'] if os.path.isfile(p) else 0.0
    trig = any(e.get('vessel') == v for e in ev)
    ok = L >= 0.5 * (b - a)
    kind = 'OK' if ok else ('BAD' if trig and v not in skip else 'NOPINCH' if not trig else 'SKIPPED')
    print(kind, v, 'vmtk_mm %.1f label_s %.2f-%.2f pinch_events %s' % (L, a, b, trig))
"
}
for c in "$@"; do
  D=$DER_H/$c/ct_seg/$TAG
  [ -f $D/seg_qc.json ] || { echo "$c: no ct_seg/$TAG wall"; continue; }
  out=$(check $c); echo "$out" | sed "s/^/$c check: /"
  bad=$(echo "$out" | awk '$1=="BAD" {print $2}' | tr '\n' ' ')
  [ -z "$bad" ] && continue
  echo "== $(date '+%F %T') $c: pinch-fix fallback for: $bad"
  SKF=$DER_H/$c/ct_seg/pinch_skip_$TAG.json
  old=""; [ -f $SKF ] && old=$(grep -o '"r\?l\?va"' $SKF | tr -d '"' | tr '\n' ' ')
  vs=$(echo $old $bad | tr ' ' '\n' | grep -v '^$' | sort -u | awk '{printf "%s\"%s\"", (NR>1?", ":""), $1}')
  printf '{"vessels": [%s], "why": "VMTK centreline of the pinch-fixed wall < 50 %% of the label range: %s", "by": "scaleup_ctseg_fallback.sh", "when": "%s"}\n' \
    "$vs" "$(echo "$out" | awk '$1=="BAD"' | tr '\n' ';')" "$(date '+%F %T')" > $SKF
  rm -rf ${D}_pinchfail; mv $D ${D}_pinchfail
  V2_FORCE="ctseg measure protrusion fig" bash $SPU/scaleup_v2_post.sh $S "ctseg measure protrusion fig collect grade" $c
  after=$(check $c); echo "$after" | sed "s/^/$c after: /"
  printf '{"case": "%s", "tag": "%s", "fallback_vessels": [%s], "before": "%s", "after": "%s", "failed_wall_kept_as": "%s_pinchfail", "when": "%s"}\n' \
    "$c" "$TAG" "$vs" "$(echo "$out" | tr '\n' ';')" "$(echo "$after" | tr '\n' ';')" "$TAG" "$(date '+%F %T')" > $DER_H/$c/ct_seg/pinch_fallback_$TAG.json
done
