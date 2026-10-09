#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\install_s3fix_tools.sh (= /sp/v6/s3fix/install_s3fix_tools.sh in the build containers)
# s3_fix (2026-10-08): install bake_v6 1.2 (L1 lattice shift + L2 route-island bridge, both opt-in; defaults = 1.1) and
# checks_v6 1.6 (parts coverage + folds), freeze 1.1 / 1.5, and write the NEW frozen manifest build/TOOLCHAIN_s3fix.md5 =
# every entry of the s2 manifest (TOOLCHAIN_s2build.md5, d65b7209; kept unchanged as the s2 record, its two changed
# entries now resolve to the frozen copies) with current md5s + the frozen copies + the s3_fix scripts. Run ONCE in a
# container (paths are container paths), when no worker is running the dev copies.
#   bash /sp/v6/s3fix/install_s3fix_tools.sh
set -e
V=/sp/v6
M2=$V/build/TOOLCHAIN_s2build.md5
M=$V/build/TOOLCHAIN_s3fix.md5
[ "$(md5sum < $M2 | cut -c1-8)" = d65b7209 ] || { echo "s2 manifest is not d65b7209"; exit 3; }
frz() {  # frz <live> <frozen> <expected md5 of live>
  [ -f "$2" ] || { [ "$(md5sum < "$1" | cut -c1-32)" = "$3" ] && cp -p "$1" "$2"; }
  [ -f "$2" ] || { echo "frozen copy $2 missing / unexpected md5 of $1"; exit 3; }
  [ "$(md5sum < "$2" | cut -c1-32)" = "$3" ] || { echo "frozen copy $2 has the wrong md5"; exit 3; }
}
frz $V/bake/bake_v6.py $V/bake/bake_v6_v1_1.py "$(grep ' \*/sp/v6/bake/bake_v6.py$' $M2 | cut -c1-32)"
frz $V/checks/checks_v6.py $V/checks/checks_v6_v1_5.py "$(grep ' \*/sp/v6/checks/checks_v6.py$' $M2 | cut -c1-32)"
cp $V/bake/bake_v6_1p2_dev.py $V/bake/bake_v6.py
cp $V/checks/checks_v6_1p6_dev.py $V/checks/checks_v6.py
cmp $V/bake/bake_v6.py $V/bake/bake_v6_1p2_dev.py && cmp $V/checks/checks_v6.py $V/checks/checks_v6_1p6_dev.py
{
  echo "# s3_fix frozen toolchain manifest ($(date '+%F %T %Z')): '<md5> *<container path>'. = the s2 manifest"
  echo "# TOOLCHAIN_s2build.md5 (d65b7209) with bake_v6 1.2 (L1 lattice shift + L2 route-island bridge, opt-in; defaults = 1.1)"
  echo "# and checks_v6 1.6 (parts coverage against the v4 / v5 shipped wall + folds); frozen bake_v6 1.1 / checks_v6 1.5"
  echo "# copies; the s3_fix scripts (fold_fix 1.0, stamp_prov 1.1, copy_v6 1.2, runners, diagnoses). Workers check it with md5sum -c."
  grep -v '^#' $M2 | sed 's/^[0-9a-f]* \*//' | while read -r p; do echo "$(md5sum < "$p" | cut -c1-32) *$p"; done
  for p in $V/bake/bake_v6_v1_1.py $V/checks/checks_v6_v1_5.py \
           $V/s3fix/dr_s3.sh $V/s3fix/p.sh $V/s3fix/batch_rebake.sh $V/s3fix/final_other.sh $V/s3fix/batch_sofa.sh \
           $V/s3fix/batch_cov.sh $V/s3fix/fold_fix.py $V/s3fix/stamp_prov_s3.py $V/s3fix/copy_v6_s3.sh \
           $V/s3fix/install_s3fix_tools.sh $V/s3fix/sum_cov.py $V/s3fix/sum_cov2.py \
           $V/s3fix/diag_frag.py $V/s3fix/diag_sign.py $V/s3fix/diag_ray.py $V/s3fix/diag_neck.py $V/s3fix/diag_shift.py \
           $V/s3fix/diag_island.py $V/s3fix/batch_frag.sh $V/s3fix/batch_shift.sh $V/s3fix/peek_att.py $V/s3fix/sum_frag.py \
           $V/s3fix/sum_shift.py; do
    echo "$(md5sum < "$p" | cut -c1-32) *$p"; done
} > $M.tmp
mv $M.tmp $M
grep -v '^#' $M | md5sum -c --quiet && echo "MANIFEST OK $(md5sum < $M | cut -c1-8) $(grep -vc '^#' $M) files"
echo "changed vs d65b7209:"; diff <(grep -v '^#' $M2) <(grep -v '^#' $M | head -n $(grep -vc '^#' $M2)) || true
