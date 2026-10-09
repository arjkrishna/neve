# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_bvr.py (= /sp/v6/s3fix/patch_bvr.py in the build containers)
import re, sys
F = sys.argv[1]
s = open(F).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, b)
rep('VERSION = "build_v6_record 1.0 (s2_records, 2026-10-07)"',
    'VERSION = "build_v6_record 1.1 (s3_fix, 2026-10-08; 1.0 s2_records 2026-10-07)"')
rep('ap.add_argument("--out", default="/sp/v6/records/out/BUILD_v6.json")',
    'ap.add_argument("--out", default="/sp/v6/records/out/BUILD_v6.json")\n'
    '# 1.1 (s3_fix): the set-level coverage result (checks_v6 1.6 on all 223) and the per-anatomy checks records of the\n'
    '# anatomies s3_fix did not re-stamp (for the orient:* derived flags only)\n'
    'ap.add_argument("--coverage", default="/sp/v6/results/v6_coverage_s3.json")\n'
    'ap.add_argument("--checks4", default="/repo/carotid_data/anatomies_v6/_records/per_anatomy/checks4")')
# derived flags + s3 per anatomy, before the licence block
rep('    # ---- licence (LICENSE files in the folder, cross-read with provenance)\n',
    '    # ---- 1.1: derived flags (the flags v6_build_collect derives outside provenance) and the s3_fix record\n'
    '    if isinstance(ck.get("derived_flags"), list):\n'
    '        dflags, dsrc = list(ck["derived_flags"]), "provenance v6_checks.derived_flags"\n'
    '    else:\n'
    '        dflags, dsrc = [], "provenance (left map deviation) + _records/per_anatomy/checks4 (orient accepted flags)"\n'
    '        c4p = os.path.join(a.checks4, n + ".json")\n'
    '        if os.path.exists(c4p):\n'
    '            dflags += ["orient:%s" % f_ for f_ in ((jl(c4p).get("orient") or {}).get("accepted_flags") or [])]\n'
    '        if lf.get("left_fork_pose") == "zenodo_map" and ldev is not None and ldev >= 0.01:\n'
    '            dflags.append("left_pose_map_bounded_dev")\n'
    '    C["derived_flags_source:%s" % dsrc] += 1\n'
    '    for f_ in dflags:\n'
    '        C["flag_derived:%s" % (f_ if not f_.startswith("orient:") else "orient:" + f_.split(":")[1])] += 1\n'
    '    s3 = ck.get("s3_fix")\n'
    '    if s3:\n'
    '        C["s3_fix_kind:%s" % s3.get("kind")] += 1\n'
    '        sh_ = [float(v) for v in (bk.get("lattice_shift_mm") or [0, 0, 0])]\n'
    '        rib = bk.get("route_island_bridge") or {}\n'
    '        L["s3_fix"].append({"anatomy": sn, "kind": s3.get("kind"), "lattice_shift_mm": sh_ if any(sh_) else None,\n'
    '                            "route_island_bridges": [[b_.get("section"), b_.get("rcca_s"), b_.get("volume_mm3"), b_.get("length_mm")]\n'
    '                                                     for b_ in (rib.get("bridged") or [])] or None,\n'
    '                            "obj_fold_fix": (bk.get("obj_fold_fix") or {}).get("result"),\n'
    '                            "coverage": ck.get("coverage_v4_v5_shipped"), "flags": [f_ for f_ in ck.get("flags") or []\n'
    '                                                                                  if f_.startswith(("d27_coverage", "d27_bake", "obj_fold_fix"))]})\n'
    '        C["s3_bake_lattice_shift"] += bool(any(sh_))\n'
    '        C["s3_bake_route_island_bridge"] += bool(rib.get("bridged"))\n'
    '        C["s3_route_islands_bridged"] += len(rib.get("bridged") or [])\n'
    '    # ---- licence (LICENSE files in the folder, cross-read with provenance)\n')
rep('        "checks_pass": bool(ck.get("pass")),\n',
    '        "checks_pass": bool(ck.get("pass")),\n'
    '        "derived_flags": dflags,\n'
    '        "s3_fix": (s3 or {}).get("kind"),\n')
rep('fin = jl(a.final)\n', 'fin = jl(a.final)\ncov = jl(a.coverage)\n')
open(F, "w").write(s)
print("ok")
s = open(F).read()
rep('    "one bake (bake_v6 1.1 = v5\'s handle rule + E1-E7, C4, D20) and one frozen toolchain; checks_v6 (D22 / D26 / D27)",\n]',
    '    "one bake (bake_v6 1.1 = v5\'s handle rule + E1-E7, C4, D20) and one frozen toolchain; checks_v6 (D22 / D26 / D27)",\n'
    '    "s3_fix (final verifier\'s must-fix list, D27): 8 right-route pieces that lost real scanned wall against v4\'s / v5\'s "\n'
    '    "shipped wall re-baked with bake_v6 1.2 (L1 lattice shift on 4, L2 route-island bridge on 4); 8 pieces with a "\n'
    '    "0.1-1 % deficit accepted and flagged (record only); one fold-over pair removed from m_022_left mr_021_L\'s OBJ; "\n'
    '    "checks_v6 1.6 adds the coverage gate (against the v4 / v5 shipped wall) and the OBJ fold-over gate",\n]')
rep('     "text": "no stopping for choices: residuals get the recommended minimal option + a flag; k_011_left mr_012_L\'s siphon deficit on the bake\'s repair capsule = a bake repair window, reported not gated"},',
    '     "text": "no stopping for choices: residuals get the recommended minimal option + a flag; k_011_left mr_012_L\'s siphon deficit on the bake\'s repair capsule = a bake repair window, reported not gated. Applied by s3_fix (2026-10-08) to the final verifier\'s coverage finding: a real-wall deficit >= 1 % of a piece takes a fix that restores the wall (a targeted re-bake: bake_v6 1.2 lattice shift L1, else route-island bridge L2), else v5\'s geometry for that piece; < 1 % is accepted with a flag"},')
rep('    "final": "carotid_tools/v6/build/build/TOOLCHAIN_s2build.md5 = d65b7209 (72 files; recorded in provenance v6_checks)",\n}',
    '    "final": "carotid_tools/v6/build/build/TOOLCHAIN_s2build.md5 = d65b7209 (72 files; recorded in provenance v6_checks of the 206 anatomies s3_fix did not change)",\n'
    '    "s3_fix": "carotid_tools/v6/build/build/TOOLCHAIN_s3fix.md5 (recorded in provenance v6_checks of the 17 anatomies s3_fix changed; = d65b7209\'s files with bake_v6 1.2 / checks_v6 1.6 + frozen 1.1 / 1.5 + the s3_fix scripts)",\n}')
rep('                 "coverage_lt_0.99_reported_not_gated": L["d22_coverage_lt_0.99"]}',
    '                 "coverage_lt_0.99_reported_not_gated": L["d22_coverage_lt_0.99"],\n'
    '                 "coverage_v4_v5_shipped": {"file": "_records/results/v6_coverage_s3.json (checks_v6 1.6 parts coverage + folds on all 223; set-level copy)",\n'
    '                                            "rule": cov.get("rule"), "after": cov.get("after"), "before": cov.get("before")}}')
rep('                "note": "counts of the flags as written in each provenance: v6_build.flags (composer + VA tool; va_right: / "\n'
    '                        "va_left: grouped by their first two parts), v6_checks.flags (D26 / D27 / coverage / D23), bake_v6.flags"}',
    '                "derived_flags": cnt("flag_derived:"), "derived_flags_source": cnt("derived_flags_source:"),\n'
    '                "note": "counts of the flags as written in each provenance: v6_build.flags (composer + VA tool; va_right: / "\n'
    '                        "va_left: grouped by their first two parts), v6_checks.flags (D26 / D27 / coverage / D23; s3_fix: "\n'
    '                        "d27_coverage_* / d27_bake_* / obj_fold_fix_*), bake_v6.flags. derived_flags: the flags the build "\n'
    '                        "collection derived outside provenance (left_pose_map_bounded_dev from the left fork\'s map deviation, "\n'
    '                        "orient:<accepted orientation flag>); in provenance v6_checks.derived_flags on the 17 anatomies s3_fix "\n'
    '                        "re-stamped, derived here for the others (per_anatomy.<name>.derived_flags)"}\n'
    'out["s3_fix"] = {"by": "s3_fix (2026-10-08), the final verifier\'s must-fix list (verify_v6_final NO-GO) under D27",\n'
    '                 "counts": cnt("s3_fix_kind:"), "bake_lattice_shift": C["s3_bake_lattice_shift"],\n'
    '                 "bake_route_island_bridge": C["s3_bake_route_island_bridge"], "route_islands_bridged": C["s3_route_islands_bridged"],\n'
    '                 "changed": L["s3_fix"],\n'
    '                 "notes": "_records/notes/s3_fix.md"}')
rep('    "doc": "CAROTID_V6.md",\n}', '    "doc": "CAROTID_V6.md",\n    "s3_fix": "_records/results/v6_coverage_s3.json, _records/results/s3_fix/ (records of the 17 changed anatomies), _records/notes/s3_fix.md",\n}')
open(F, "w").write(s)
print("ok2")
