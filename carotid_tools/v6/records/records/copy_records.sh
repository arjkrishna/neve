#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\copy_records.sh (= /sp/v6/records/copy_records.sh in the build containers)
# s2_records: byte copies of the files CAROTID_V6.md cites into carotid_data/anatomies_v6/_records/ (Git Bash, host side).
# Idempotent: a file already present with the same sha256 is skipped; a different one is an error (never overwritten).
# Writes _records/RECORDS.sha256 (sha256 of every copied file, relative to _records/) at the end.
#   bash copy_records.sh            (README.md and checks/build_record_check.json are written separately)
set -euo pipefail
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
W=/d/Arjun/neve/carotid_data/v6_work
DST=/d/Arjun/neve/carotid_data/anatomies_v6/_records
mkdir -p "$DST"
n=0; skip=0
cpf() {   # cpf <src file> <dst path relative to _records>
  local s="$1" d="$DST/$2"
  [ -f "$s" ] || { echo "MISSING $s" >&2; exit 3; }
  if [ -f "$d" ]; then
    if [ "$(sha256sum "$s" | cut -c1-64)" = "$(sha256sum "$d" | cut -c1-64)" ]; then skip=$((skip+1)); return; fi
    echo "DIFFERS (not overwritten): $d" >&2; exit 4
  fi
  mkdir -p "$(dirname "$d")"; cp -p "$s" "$d"; n=$((n+1))
}
cpdir() { # cpdir <src dir> <dst dir relative> [find args...]   (files only, keeps sub-paths)
  local s="$1" d="$2"; shift 2
  while IFS= read -r f; do cpf "$s/$f" "$d/$f"; done < <(cd "$s" && find . -type f "$@" | sed 's|^\./||' | sort)
}
# --- results: the final records
for f in v6_final_summary.json v6_checks.json v6_build.json stage2_dry.json stage2_dry_auto.json stage2_dry2.json \
         stage2_dry2_auto.json pilot_summary.json repilot_summary.json repilot2_summary.json s2_harden_wall_bake.json; do
  cpf "$SP/v6/results/$f" "results/$f"; done
for f in copy_verify_final.json copy_verify.json copy_verify_staged.json anatomies_v6.sha256 anatomies_v6_staged.sha256 \
         check4_summary.txt tree3_merge.json verify_tree3.json cmp_build4.json cmp_build4_final2.json cmp_dry.json; do
  cpf "$W/full/$f" "results/build/$f"; done
cpf "$W/full/prov_final/stamp_prov.json" "results/build/stamp_prov.json"
cpf "$W/full/license/license_check.json" "results/build/license_check.json"
# --- set-level checks (s2_checks)
cpdir "$W/s2checks/split" "checks/split"
cpdir "$W/s2checks/loader" "checks/loader"
cpdir "$W/s2checks/orient" "checks/orient"
cpdir "$W/s2checks/lumen" "checks/lumen"
cpdir "$W/s2checks/entry_geom" "checks/entry_geom"
cpdir "$W/s2checks/ident" "checks/ident"
cpdir "$W/s2checks/route_spot" "checks/route_spot"
cpdir "$W/s2checks/diag_cov12" "checks/diag_cov12"
# --- gates and heading evidence (stage 1)
for f in selftest.txt test_api.txt orient_v3.json orient_v4.json orient_v5.json; do cpf "$SP/v6/gates/out/$f" "gates/$f"; done
for f in heading_evidence.txt heading_evidence.json zenodo_frames.json; do cpf "$SP/v6/heading/$f" "gates/$f"; done
# --- per anatomy: the final checks records (checks_v6; route_d22 1.4, k_011_left mr_012_L 1.5)
cpdir "$W/full/checks4" "per_anatomy/checks4" -not -path './_claim*'
cpdir "$W/full/diag/tubegenus_final" "per_anatomy/tube_genus_prebake" -name '*.json'
# --- SOFA (RCCA all 223 + v5 / v4 references; D14 on the 25-anatomy set)
for d in rcca_v6 rcca_v5 rcca_v4 d14_v6 d14_v5 d14_v4; do cpdir "$W/s2checks/sofa/$d" "sofa/$d"; done
for f in set25.json items.txt bif_angles_w042.json; do cpf "$W/s2checks/sofa/$f" "sofa/$f"; done
# --- diagnoses the doc cites (D26 / D27 / bake fusions / VA 1.4.2)
for f in d26eq_cmp.txt d26eq_cmp_checks4.txt d27eq_cmp.txt offboth_k011_mr012L_siphon.json own2_k011_mr012L_siphon.json \
         own_case_w_033_left__topcow_mr_007.json tube_m030_mr013L.json tube_w014R_mr010.json scan_self_final.json \
         d22_w008_mr001_fork.json d22_w012L_mr020_fork.json d22alt_w008_mr001.json d22dir_w008_mr001.json d22dir_w012L_mr020.json \
         d22dir_case_w_033_left__topcow_mr_007.json; do cpf "$W/full/diag/$f" "diag/$f"; done
cpdir "$W/full/diag" "diag" -maxdepth 1 -name 'd22alt_case_*.json'
cpdir "$W/full/diag/d27test" "diag/d27test"
cpdir "$W/full/diag/d26test" "diag/d26test" -name '*.json'
# --- notes: plan, decisions, build log, every stage and verifier, the audit's agent notes
for f in V6_BUILD_PLAN.md DECISIONS_ADDENDUM.md; do cpf "$SP/v6/$f" "notes/$f"; done
cpf "$SP/v5_va/v6/V6_ITEMS.md" "notes/V6_ITEMS.md"
cpf "$SP/v5_va/state/V5_BUILD_LOG.md" "notes/V5_BUILD_LOG.md"
for f in v6_plan v6_carotid v6_va v6_wall v6_pilot verify_v6_pilot s2_harden_carotid s2_harden_carotid2 s2_harden_va \
         s2_harden_wall_bake s2_integrate verify_s2_gate s2_integrate2 verify_s2_gate2 s2_build s2_checks; do
  cpf "$SP/v6/notes/$f.md" "notes/$f.md"; done
for f in tbo_history tbo_geom tbo_cause tbo_report va_narrow verify_tbo; do cpf "$SP/v5_va/notes/$f.md" "notes/audit/$f.md"; done
echo "copied $n, already present $skip"
