#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\copy_records_s3.sh (= /sp/v6/s3fix/copy_records_s3.sh in the build containers)
# s3_fix: add the s3_fix records to carotid_data/anatomies_v6/_records/ (Git Bash, host side). New files are copied; a
# file that already exists with another sha256 is REPLACED only if it is on the update list below (the scratchpad
# originals of notes that grew since s2_records copied them, e.g. DECISIONS_ADDENDUM.md); anything else differing is an
# error. Every replacement is logged (_records/results/s3_fix/records_replaced.txt). RECORDS.sha256 is rebuilt at the end
# over every file under _records/ except itself and README.md (as s2_records wrote it).
#   bash copy_records_s3.sh
set -euo pipefail
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
W=/d/Arjun/neve/carotid_data/v6_work
X=$W/s3fix
DST=/d/Arjun/neve/carotid_data/anatomies_v6/_records
UPDATABLE=" notes/DECISIONS_ADDENDUM.md notes/V5_BUILD_LOG.md notes/s3_fix.md README.md checks/build_record_check.json notes/s2_records.md notes/s2_images.md "
mkdir -p "$DST/results/s3_fix"
REPL=$DST/results/s3_fix/records_replaced.txt
n=0; skip=0; rep=0
cpf() {   # cpf <src file> <dst path relative to _records>
  local s="$1" d="$DST/$2"
  [ -f "$s" ] || { echo "MISSING $s" >&2; exit 3; }
  if [ -f "$d" ]; then
    if [ "$(sha256sum "$s" | cut -c1-64)" = "$(sha256sum "$d" | cut -c1-64)" ]; then skip=$((skip+1)); return; fi
    case "$UPDATABLE" in *" $2 "*) echo "$(date '+%F %T') $2 $(sha256sum "$d" | cut -c1-16) -> $(sha256sum "$s" | cut -c1-16)" >> $REPL
                                    cp -p "$s" "$d"; rep=$((rep+1)); return;; esac
    echo "DIFFERS (not overwritten): $d" >&2; exit 4
  fi
  mkdir -p "$(dirname "$d")"; cp -p "$s" "$d"; n=$((n+1))
}
cpdir() { local s="$1" d="$2"; shift 2
  while IFS= read -r f; do cpf "$s/$f" "$d/$f"; done < <(cd "$s" && find . -type f "$@" | sed 's|^\./||' | sort); }
# --- the s3_fix results
cpf "$SP/v6/results/v6_coverage_s3.json" "results/v6_coverage_s3.json"
cpf "$SP/v6/results/v6_s3_set.json" "results/s3_fix/set_checks_after_fix.json"
cpf "$X/sofa_summary_final.json" "results/s3_fix/sofa_rcca_summary.json"
cpf "$X/copy_verify_s3.json" "results/s3_fix/copy_verify_s3.json"
cpf "$X/anatomies_v6.sha256" "results/s3_fix/anatomies_v6.sha256"
cpf "$X/prov_final/stamp_prov_s3.json" "results/s3_fix/stamp_prov_s3.json"
cpf "$X/cov_v6_0_summary.txt" "results/s3_fix/coverage_before_fix_summary.txt"
cpf "$SP/v6/s3fix/final_rebake.txt" "results/s3_fix/final_rebake.txt"
cpf "$SP/v6/s3fix/final_other.txt" "results/s3_fix/final_other.txt"
cpf "$X/logs/final.batch.log" "results/s3_fix/final.batch.log"
cpf "$X/logs/final_other.batch.log" "results/s3_fix/final_other.batch.log"
for f in split_s3.json loader_s3.json; do [ -f "$X/set/$f" ] && cpf "$X/set/$f" "results/s3_fix/$f"; done
# --- per anatomy: the final checks records, the bake markers, the fold fix, SOFA, the coverage before the fix
cpdir "$X/checks/final" "per_anatomy/checks_s3" -name 'case_*.json'
cpdir "$X/bake/final/_done" "per_anatomy/bake_s3" -name '*.bake.json'
cpdir "$X/fold" "per_anatomy/fold_fix" -name 'case_*.json'
cpdir "$X/sofa/final" "sofa/rcca_v6_s3" -name 'case_*.json'
cpdir "$X/cov_v6_0" "checks/coverage_v1p6_prefix" -maxdepth 1 -name 'case_*.json'
# --- diagnoses (the lost-wall mechanism, the lattice screen, the neck test, the dev re-bake variants)
cpdir "$X/diag" "diag/s3_fix" -maxdepth 1 -name '*.json'
cpdir "$X/shift" "diag/s3_fix/shift" -maxdepth 1 -name 'case_*.json'
for t in A B C0 C1 C2 C3 D1 D2 D3; do [ -d "$X/checks/$t" ] && cpdir "$X/checks/$t" "diag/s3_fix/dev_rebake/$t" -name 'case_*.json'; done
# --- notes: this stage, the final verifier, and the grown originals
cpf "$SP/v6/notes/s3_fix.md" "notes/s3_fix.md"
cpf "$SP/v6/notes/verify_v6_final.md" "notes/verify_v6_final.md"
cpf "$SP/v6/notes/s2_images.md" "notes/s2_images.md"
cpf "$SP/v6/notes/s2_records.md" "notes/s2_records.md"
cpf "$SP/v6/DECISIONS_ADDENDUM.md" "notes/DECISIONS_ADDENDUM.md"
cpf "$SP/v6/records/out/build_record_check.json" "checks/build_record_check.json"
cpf "$SP/v6/s3fix/README_records_new.md" "README.md"
echo "copied $n, replaced $rep, already present $skip"
# --- RECORDS.sha256 over everything
( cd "$DST" && find . -type f ! -name RECORDS.sha256 ! -name RECORDS.sha256.tmp ! -path ./README.md | sed 's|^\./||' | sort | while IFS= read -r f; do sha256sum "$f"; done ) > "$X/RECORDS.sha256.tmp"
mv "$X/RECORDS.sha256.tmp" "$DST/RECORDS.sha256"
(cd "$DST" && sha256sum -c --quiet RECORDS.sha256) && echo "RECORDS.sha256: $(wc -l < $DST/RECORDS.sha256) files, sha256sum -c ok"
