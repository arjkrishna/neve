#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\finalize_tools.sh (= /sp/v6/records/finalize_tools.sh in the build containers)
# finalize_tools 1.1 (s3_fix, 2026-10-08; 1.0 s2_records = finalize_tools_v1_0.sh): (re)stage carotid_tools/v6 and install it.
# Host side (Git Bash). 1.1: the restore test runs md5sum -c on the FINAL manifest TOOLCHAIN_s3fix.md5 (2288f778) and counts
# the rows of every manifest that resolve (TOOLCHAINS.tsv).
#   1. make_tools_v6.py (container) -> records/tools_v6 (+ FILES.tsv, TOOLCHAINS.tsv)
#   2. + restore_layout.sh, MANIFEST.md (= MANIFEST_head.md + the per-file list from FILES.tsv)
#   3. restore test into records/restore_test: sha256 (restore_layout.sh), byte compare with the scratchpad, md5sum -c of
#      the final manifest's /sp entries -> records/out/restore_test.txt
#   4. install: D:\Arjun\neve\carotid_tools\v6 replaced by the staged copy (this agent's own folder), then compared
set -euo pipefail
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
R=$SP/v6/records; T=$R/tools_v6; RT=$R/restore_test; DST=/d/Arjun/neve/carotid_tools/v6
bash $R/dr_rec.sh python:3.11-slim python3 /sp/v6/records/make_tools_v6.py > $R/out/make_tools_v6.log 2>&1
cp $R/restore_layout.sh $T/restore_layout.sh
n=$(($(wc -l < $T/FILES.tsv) - 1))
{ sed "s/^- \*\*[0-9]* files\*\*:.*/- **$n files**: $(awk -F'\t' 'NR>1{c[$1]++} END{for(k in c) printf "%s %d\n", k, c[k]}' $T/FILES.tsv | sort -k2,2nr -k1,1 | awk '{printf "%s%s %s", (NR>1?", ":""), $1, $2}')./" $R/MANIFEST_head.md
  awk -F'\t' 'NR>1{print $1"\t"$3"\t"$9}' $T/FILES.tsv | awk -F'\t' '{ if ($1!=s) {print "\n### " $1 "/\n"; s=$1} w=$3;
     sub(/^selected \(.*\)$/,"",w); sub(/^listed in /,"first listed in ",w); printf "- `%s`%s\n", substr($2, length($1)+2), (w==""?"":" ("w")") }'
} > $T/MANIFEST.md
rm -rf $RT
{ bash $T/restore_layout.sh $RT
  bad=0; while IFS=$'\t' read -r st orig rest; do [ "$st" = stage ] && continue; cmp -s "$RT/$orig" "$SP/$orig" || { bad=$((bad+1)); echo "DIFF $orig"; }; done < $T/FILES.tsv
  echo "byte compare with the scratchpad: $n files, $bad differ"
  ( cd $RT && sed -n 's#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p' v6/build/TOOLCHAIN_s3fix.md5 | md5sum -c --quiet && echo "final manifest $(md5sum < v6/build/TOOLCHAIN_s3fix.md5 | cut -c1-8) (TOOLCHAIN_s3fix.md5): every /sp entry OK on the restored layout ($(sed -n 's#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p' v6/build/TOOLCHAIN_s3fix.md5 | wc -l) entries)" )
  echo "every manifest (TOOLCHAINS.tsv): $(awk -F'\t' 'NR>1 && ($5 == "current file" || $5 ~ /^frozen copy/ || $5 ~ /md5 matches/)' $T/TOOLCHAINS.tsv | wc -l) rows resolved, $(awk -F'\t' 'NR>1 && $5 == "NOT FOUND"' $T/TOOLCHAINS.tsv | wc -l) not preserved (named in MANIFEST.md)"
  echo "repo entries (TOOLCHAINS.tsv): $(awk -F'\t' '$5 ~ /repo module, md5 matches/' $T/TOOLCHAINS.tsv | wc -l) rows md5-equal, $(awk -F'\t' '$5 ~ /DIFFERS/' $T/TOOLCHAINS.tsv | wc -l) differ"
  date '+restore test %F %T'
} > $R/out/restore_test.txt 2>&1
cat $R/out/restore_test.txt
grep -q "sha256 mismatches: 0" $R/out/restore_test.txt && grep -q ", 0 differ" $R/out/restore_test.txt && grep -q "every /sp entry OK" $R/out/restore_test.txt && grep -q "^final manifest 2288f778" $R/out/restore_test.txt
case "$DST" in /d/Arjun/neve/carotid_tools/v6) ;; *) echo refusing; exit 2;; esac
rm -rf "$DST"; mkdir -p "$DST"; cp -rp $T/. "$DST/"
diff -r $T "$DST" && echo "installed: $(find $DST -type f | wc -l) files in $DST (identical to the staged copy)"
