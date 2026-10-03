#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\swap.sh (= /sp/v5_va/swap13/swap.sh in the build containers)
# v5_swap13 step 1 (host, Git Bash). Idempotent: per-anatomy done-markers in swap13/done/.
#   A  backup  anatomies_v5/<name>/            -> backup_pre_route13/<name>/                (byte copy, sha256-verified)
#              iso_v5/<name>.vtp                -> backup_pre_route13/_iso_v5/<name>.vtp
#              bake_full/_done/<name>.{bake,measure}.json -> backup_pre_route13/_bake_full_done/
#   B  swap    route13/<variant>/<name>/        -> anatomies_v5/<name>/   (file lists must be identical; cp -p each file)
#              route13/<variant>_iso/<name>.vtp -> iso_v5/<name>.vtp
#              route13/<variant>_bf/_done/<name>.{bake,measure}.json -> bake_full/_done/
#              then sha256 of every file == the pick's.
#   bash swap.sh backup|swap|verify
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va
W=$SP/swap13
A5=/d/Arjun/neve/carotid_data/anatomies_v5
ISO=/d/Arjun/neve/carotid_data/va_sources/v5_reg/iso_v5
BFD=/d/Arjun/neve/carotid_data/va_sources/v5_reg/bake_full/_done
R13=/d/Arjun/neve/carotid_data/va_sources/v5_reg/route13
BK=$SP/backup_pre_route13
MODE=$1
mkdir -p $W/done $W/logs $W/manifests $BK/_iso_v5 $BK/_bake_full_done
log() { echo "$(date '+%F %T') $*" | tee -a $W/logs/swap.log; }
manifest() { (cd "$1" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do sha256sum "$f"; done); }
flist() { (cd "$1" && find . -type f | LC_ALL=C sort); }
h() { sha256sum "$1" | cut -d' ' -f1; }
FAIL=0
while IFS=$'\t' read -r N V; do
  N=$(echo "$N" | tr -d '\r'); V=$(echo "$V" | tr -d '\r'); [ -n "$N" ] || continue
  P=$R13/$V/$N
  if [ "$MODE" = backup ]; then
    if [ -f $W/done/$N.backup ]; then log "backup skip $N (done)"; continue; fi
    mkdir -p $BK/$N
    cp -a "$A5/$N/." "$BK/$N/" && cp -p "$ISO/$N.vtp" "$BK/_iso_v5/" \
      && cp -p "$BFD/$N.bake.json" "$BFD/$N.measure.json" "$BK/_bake_full_done/" || { log "backup COPY FAILED $N"; FAIL=1; continue; }
    manifest "$A5/$N" > $W/manifests/$N.v5_before.sha256
    manifest "$BK/$N" > $W/manifests/$N.backup.sha256
    ok=1
    cmp -s $W/manifests/$N.v5_before.sha256 $W/manifests/$N.backup.sha256 || ok=0
    for f in "$ISO/$N.vtp:$BK/_iso_v5/$N.vtp" "$BFD/$N.bake.json:$BK/_bake_full_done/$N.bake.json" "$BFD/$N.measure.json:$BK/_bake_full_done/$N.measure.json"; do
      a=${f%%:*}; b=${f#*:}; [ "$(h "$a")" = "$(h "$b")" ] || ok=0
      echo "$(h "$a")  $a" >> $W/manifests/$N.v5_before_extra.sha256
    done
    if [ $ok = 1 ]; then echo "$(date '+%F %T') $(wc -l < $W/manifests/$N.backup.sha256) files + iso + 2 markers" > $W/done/$N.backup; log "backup OK $N"
    else log "backup HASH MISMATCH $N"; FAIL=1; fi
  elif [ "$MODE" = swap ]; then
    if [ -f $W/done/$N.swap ]; then log "swap skip $N (done)"; continue; fi
    [ -f $W/done/$N.backup ] || { log "swap REFUSED $N: no backup marker"; FAIL=1; continue; }
    [ -d "$P" ] && [ -f "$R13/${V}_iso/$N.vtp" ] && [ -f "$R13/${V}_bf/_done/$N.bake.json" ] && [ -f "$R13/${V}_bf/_done/$N.measure.json" ] \
      || { log "swap REFUSED $N: pick incomplete"; FAIL=1; continue; }
    if [ "$(flist "$P")" != "$(flist "$A5/$N")" ]; then log "swap REFUSED $N: file lists differ"; FAIL=1; continue; fi
    (cd "$P" && find . -type f | LC_ALL=C sort) | while IFS= read -r f; do cp -p "$P/$f" "$A5/$N/$f" || echo COPYFAIL; done | grep -q COPYFAIL && { log "swap COPY FAILED $N"; FAIL=1; continue; }
    cp -p "$R13/${V}_iso/$N.vtp" "$ISO/$N.vtp" && cp -p "$R13/${V}_bf/_done/$N.bake.json" "$R13/${V}_bf/_done/$N.measure.json" "$BFD/" \
      || { log "swap COPY FAILED (iso/markers) $N"; FAIL=1; continue; }
    log "swap copied $N <- $V"
    touch $W/done/$N.copied
  fi
  if [ "$MODE" = swap ] || [ "$MODE" = verify ]; then
    [ "$MODE" = verify ] || true
    manifest "$P" > $W/manifests/$N.pick.sha256
    manifest "$A5/$N" > $W/manifests/$N.v5_after.sha256
    ok=1
    cmp -s $W/manifests/$N.pick.sha256 $W/manifests/$N.v5_after.sha256 || ok=0
    [ "$(h "$R13/${V}_iso/$N.vtp")" = "$(h "$ISO/$N.vtp")" ] || ok=0
    for k in bake measure; do [ "$(h "$R13/${V}_bf/_done/$N.$k.json")" = "$(h "$BFD/$N.$k.json")" ] || ok=0; done
    cmp -s $W/manifests/$N.v5_after.sha256 $W/manifests/$N.v5_before.sha256 && ok=0   # must have changed
    if [ $ok = 1 ]; then echo "$(date '+%F %T') $V $(wc -l < $W/manifests/$N.v5_after.sha256) files = pick; iso + markers = pick" > $W/done/$N.swap; log "$MODE OK $N"
    else log "$MODE HASH CHECK FAILED $N"; FAIL=1; rm -f $W/done/$N.swap; fi
  fi
done < $W/picks.tsv
log "$MODE finished FAIL=$FAIL"
exit $FAIL
