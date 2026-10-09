#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\restore_layout.sh (= /sp/v6/records/restore_layout.sh in the build containers)
# restore_layout.sh <target_dir>
#
# Rebuilds the build-time layout of the v6 toolchain from this folder: every file listed in FILES.tsv is written to
# <target_dir>/<original path> (v6/..., v5_va/..., v4_build/...) with the one "[carotid_tools/v6 copy]" header line
# removed, so each restored file is byte-identical to the file the build ran (checked against the sha256 in FILES.tsv).
# <target_dir> plays the session scratchpad: mount it as /sp in the build containers so the scripts find themselves at
# /sp/v6/..., /sp/v5_va/... (see MANIFEST.md, "Re-running"). Then, for the final toolchain:
#   cd <target_dir> && sed -n 's#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p' v6/build/TOOLCHAIN_s2build.md5 | md5sum -c
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: restore_layout.sh <target_dir>   (e.g. <scratch>; becomes /sp)" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
TGT="$1"
mkdir -p "$TGT"
bad=0; n=0
while IFS=$'\t' read -r stage orig copy header sha_orig rest; do
  [ "$stage" = "stage" ] && continue
  out="$TGT/$orig"
  mkdir -p "$(dirname "$out")"
  if [ "$header" = "yes" ]; then
    awk 'BEGIN{done=0} { if (!done && index($0, "# [carotid_tools/v6 copy] source: ") == 1) { done=1; next } print }' \
      "$HERE/$copy" > "$out"
  else
    cp "$HERE/$copy" "$out"
  fi
  got=$(sha256sum "$out" | cut -c1-64)
  if [ "$got" != "$sha_orig" ]; then echo "MISMATCH $orig" >&2; bad=$((bad+1)); fi
  n=$((n+1))
done < "$HERE/FILES.tsv"
echo "restored $n files into $TGT; sha256 mismatches: $bad"
[ "$bad" -eq 0 ]
