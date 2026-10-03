#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\item5.sh (= /sp/v5_va/checks/item5.sh in the build containers)
# v5_checks item 5 (host, read-only git): anatomies_v4 against git HEAD.   bash item5.sh <out.json>
SPD=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/checks
OUT=${1:-$SPD/item5_v4_untouched_end.json}
cd /d/Arjun/neve
git ls-tree -r HEAD -- carotid_data/anatomies_v4 | awk -F'\t' '{split($1,a," "); print a[3]"\t"$2}' > $SPD/v4_head_end.tsv
cut -f2 $SPD/v4_head_end.tsv | git hash-object --stdin-paths > $SPD/v4_disk_end.hash
cat > $OUT <<EOT
{"head": "$(git rev-parse HEAD)", "checked_at": "$(date '+%F %T')",
 "git_status_porcelain_lines": $(git status --porcelain -- carotid_data/anatomies_v4 | wc -l),
 "git_diff_stat_HEAD_lines": $(git diff --stat HEAD -- carotid_data/anatomies_v4 | wc -l),
 "tracked_files": $(wc -l < $SPD/v4_head_end.tsv),
 "hash_object_vs_head_mismatches": $(paste <(cut -f1 $SPD/v4_head_end.tsv) $SPD/v4_disk_end.hash | awk -F'\t' '$1!=$2{n++} END{print n+0}'),
 "ignored_untracked_files": $(git status --porcelain --ignored -- carotid_data/anatomies_v4 | grep -c '^!!'),
 "untracked_not_ignored": $(git status --porcelain -- carotid_data/anatomies_v4 | grep -c '^??'),
 "newest_mtime_any_file": "$(find carotid_data/anatomies_v4 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS\n' | sort | tail -1)",
 "files_newer_than_2026-10-01": $(find carotid_data/anatomies_v4 -newermt "2026-10-01" | wc -l)}
EOT
cat $OUT
