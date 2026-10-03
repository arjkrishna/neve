#!/bin/bash
# restore_layout.sh <target_dir>
#
# Rebuilds the build-time layout of the v5 toolchain from this folder: every file listed in FILES.tsv is copied to
# <target_dir>/<original path under v5_va> with the one "[carotid_tools/v5 copy]" header line removed, so each restored
# file is byte-identical to the file the build ran (checked against the sha256 in FILES.tsv). Mount <target_dir>'s
# parent as /sp in the build containers so the scripts find themselves at /sp/v5_va/... (see MANIFEST.md, "Re-running").
#
# Donor unit records (graft/donor_records/) are reference copies; they are not restored into the layout.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: restore_layout.sh <target_dir>   (e.g. <scratch>/v5_va)" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
TGT="$1"
mkdir -p "$TGT"
bad=0; n=0
while IFS=$'\t' read -r stage orig copy header sha_orig rest; do
  [ "$stage" = "stage" ] && continue
  case "$orig" in carotid_data/*) continue ;; esac
  out="$TGT/$orig"
  mkdir -p "$(dirname "$out")"
  if [ "$header" = "yes" ]; then
    awk 'BEGIN{done=0} { if (!done && index($0, "# [carotid_tools/v5 copy] source: ") == 1) { done=1; next } print }' \
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
