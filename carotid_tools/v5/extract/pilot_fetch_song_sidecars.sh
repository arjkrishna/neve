#!/usr/bin/env bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_sidecars.sh (= /sp/v5_va/pilot_fetch_song_sidecars.sh in the build containers)
# pilot_fetch_song_sidecars.sh (pilot_fetch_song): fetch the dcm2niix BIDS sidecar (cta_images_head_<id>.json, RAR5
# method 3 = compressed) of every Song case without downloading the archives. Per case: range-read the archive's
# signature + main header (bytes 0..first file header - 1) and the sidecar's own file header + packed data, glue
# them, plus a RAR5 end-of-archive block, into a one-member RAR5 file and extract it with Windows' bsdtar
# (libarchive 3.8.8, RAR5 read support). The member is non-solid, so it decompresses on its own.
# Without the end block bsdtar exits 0 but writes zeros (first attempt, sidecars_attempt1_zeros.log), so
# pilot_fetch_song_meta.py re-checks every sidecar against the RAR5 CRC32 in the index.
# Input: _source_meta/sidecar_ranges.tsv (case url first_hdr hdr_off data_end member), written from
# results/pilot_fetch_song_index.json. Output: _source_meta/sidecars/<case>.json and sidecars.log
# Run on the HOST (Git Bash): bash /c/.../scratchpad/v5_va/pilot_fetch_song_sidecars.sh
set -u
M=/d/Arjun/neve/carotid_data/va_sources/thin_slice/song/_source_meta
BSDTAR=/c/Windows/System32/tar.exe
mkdir -p "$M/sidecars" "$M/tmp_rar"
cd "$M/tmp_rar" || exit 1
printf '\x1d\x77\x56\x51\x03\x05\x04\x00' > end.bin   # RAR5 end-of-archive header
declare -A HEAD
while IFS=$'\t' read -r CASE URL FIRST HOFF DEND MEMBER; do
  [ -s "$M/sidecars/$CASE.json" ] && continue
  if [ -z "${HEAD[$URL]:-}" ]; then
    curl -sfL -r "0-$((FIRST-1))" -o "head_$(basename "$URL").bin" "$URL" || { echo "$CASE head FAIL"; continue; }
    HEAD[$URL]="head_$(basename "$URL").bin"
  fi
  ok=0
  for t in 1 2 3 4 5; do
    if curl -sfL -r "$HOFF-$DEND" -o member.bin "$URL" && [ "$(stat -c %s member.bin)" -eq $((DEND-HOFF+1)) ]; then ok=1; break; fi
    sleep $((t*3))
  done
  [ $ok -eq 1 ] || { echo "$CASE member FAIL"; continue; }
  cat "${HEAD[$URL]}" member.bin end.bin > one.rar
  rm -rf x && mkdir x
  if "$BSDTAR" -xf one.rar -C x 2> err.txt; then st=ok; else st="bsdtar:$(tr '\n' ' ' < err.txt)"; fi
  f=$(find x -name '*.json' | head -1)
  if [ -n "$f" ] && [ -s "$f" ]; then cp "$f" "$M/sidecars/$CASE.json"; echo "$CASE ok $st"; else echo "$CASE FAIL $st"; fi
done < "$M/sidecar_ranges.tsv" | tee -a "$M/sidecars.log"
cd "$M" && rm -rf "$M/tmp_rar"
ls "$M/sidecars" | wc -l
