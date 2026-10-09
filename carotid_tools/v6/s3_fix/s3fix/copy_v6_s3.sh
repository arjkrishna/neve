#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\copy_v6_s3.sh (= /sp/v6/s3fix/copy_v6_s3.sh in the build containers)
# copy_v6 1.2 (s3_fix, 2026-10-08): replace ONLY the anatomies changed by s3_fix in carotid_data/anatomies_v6, with hash
# verification. Host-side (Git Bash). Per changed anatomy <n> (names in $ITEMS):
#   staged = v6_work/s3fix/staged/<n>/ = the final tree's files ($TREE/<n>: Centrelines_comb/, OBJ, collision_full.vtp,
#   mesh_v6.json) + the stamped provenance ($PROV/<n>/provenance.json, stamp_prov 1.1) + the licence files of the CURRENT
#   anatomies_v6/<n> (LICENSE_v6_donors.txt, LICENSE_VMR_0248.txt where present: donors unchanged). The staged file SET
#   must equal the current folder's. The current folder is moved to v6_work/s3fix/replaced/<n>/ (kept), the staged one
#   copied in. Manifest $MAN = the s2 build manifest (v6_work/full/anatomies_v6.sha256) with the changed anatomies' lines
#   replaced by the staged files' sha256; verified on anatomies_v6 with sha256sum -c plus an exact file-set comparison
#   (BUILD_v6.json and _records are outside the manifest, as in s2). Re-runnable: an anatomy whose folder already equals
#   its staged copy is skipped.
#     ITEMS=<names file> TREE=<final tree> PROV=<stamped prov dir> bash copy_v6_s3.sh
set -u
W=/d/Arjun/neve/carotid_data/v6_work/s3fix
DST=/d/Arjun/neve/carotid_data/anatomies_v6
OLDMAN=/d/Arjun/neve/carotid_data/v6_work/full/anatomies_v6.sha256
MAN=${MAN:-$W/anatomies_v6.sha256}; OUT=${OUT:-$W/copy_verify_s3.json}
ITEMS=${ITEMS:?}; TREE=${TREE:?}; PROV=${PROV:?}
NAMES=$(tr -d '\r' < $ITEMS | awk 'NF{print $1}')
mkdir -p $W/staged $W/replaced
for n in $NAMES; do
  [ -d "$DST/$n" ] || { echo "no $DST/$n"; exit 3; }
  [ -f "$PROV/$n/provenance.json" ] || { echo "no stamped provenance $n"; exit 3; }
  rm -rf "$W/staged/$n"; mkdir -p "$W/staged/$n"
  cp -r "$TREE/$n/Centrelines_comb" "$W/staged/$n/"
  for f in vessel_architecture_collision.obj collision_full.vtp mesh_v6.json; do cp "$TREE/$n/$f" "$W/staged/$n/"; done
  cp "$PROV/$n/provenance.json" "$W/staged/$n/"
  for f in LICENSE_v6_donors.txt LICENSE_VMR_0248.txt; do [ -f "$DST/$n/$f" ] && cp "$DST/$n/$f" "$W/staged/$n/"; done
  A=$(cd "$DST/$n" && find . -type f | sort | tr '\n' ' '); B=$(cd "$W/staged/$n" && find . -type f | sort | tr '\n' ' ')
  [ "$A" = "$B" ] || { echo "file set differs for $n: [$A] vs [$B]"; exit 4; }
done
# manifest: old lines of unchanged anatomies + staged lines of changed ones
grep -v -F -f <(for n in $NAMES; do echo " $n/"; echo "*$n/"; done) $OLDMAN > $MAN.tmp
for n in $NAMES; do (cd $W/staged && find "$n" -type f -print0 | sort -z | xargs -0 sha256sum) >> $MAN.tmp; done
sort -k2 $MAN.tmp > $MAN; rm -f $MAN.tmp
# replace
for n in $NAMES; do
  if diff -rq "$DST/$n" "$W/staged/$n" > /dev/null 2>&1; then echo "unchanged already: $n"; continue; fi
  rm -rf "$W/replaced/$n"; mv "$DST/$n" "$W/replaced/$n"
  mkdir -p "$DST/$n"; cp -r "$W/staged/$n/." "$DST/$n/"
  echo "replaced $n"
done
# verify the whole folder
(cd $DST && sha256sum -c --quiet $MAN) > $W/copy_verify_s3.log 2>&1; RC=$?
(cd $DST && find . -type f ! -path './_records/*' ! -name BUILD_v6.json | sed 's#^\./##' | sort) > $W/copy_files_dst.txt
sed -E 's/^[0-9a-f]{64} [ *]//' $MAN | sort > $W/copy_files_man.txt
EXTRA=$(comm -23 $W/copy_files_dst.txt $W/copy_files_man.txt | wc -l); MISS=$(comm -13 $W/copy_files_dst.txt $W/copy_files_man.txt | wc -l)
NF=$(wc -l < $W/copy_files_dst.txt); NM=$(wc -l < $MAN); NDIR=$(ls -d $DST/case_* | wc -l)
MSHA=$(sha256sum $MAN | cut -c1-64)
OK=false; [ $RC = 0 ] && [ $NF = $NM ] && [ $EXTRA = 0 ] && [ $MISS = 0 ] && [ $NDIR = 223 ] && OK=true
cat > $OUT <<EOF2
{"ok": $OK, "anatomies": $NDIR, "changed": $(echo $NAMES | wc -w), "files_dst": $NF, "files_manifest": $NM, "sha256sum_c_rc": $RC,
 "extra_in_dst": $EXTRA, "missing_in_dst": $MISS, "manifest": "${MAN#/d/Arjun/neve/}", "manifest_sha256": "$MSHA",
 "copy_v6": "1.2 (s3_fix)", "date": "$(date '+%F %T %Z')", "replaced_kept_in": "${W#/d/Arjun/neve/}/replaced"}
EOF2
cat $OUT
