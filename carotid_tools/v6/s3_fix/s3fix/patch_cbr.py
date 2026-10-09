# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_cbr.py (= /sp/v6/s3fix/patch_cbr.py in the build containers)
import sys
F = sys.argv[1]
s = open(F).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, b)
rep('VERSION = "check_build_record 1.0 (s2_records, 2026-10-07)"', 'VERSION = "check_build_record 1.1 (s3_fix, 2026-10-08; 1.0 s2_records 2026-10-07)"')
rep('ap.add_argument("--out", default="/sp/v6/records/out/build_record_check.json")',
    'ap.add_argument("--out", default="/sp/v6/records/out/build_record_check.json")\n'
    'ap.add_argument("--checks4", default="/repo/carotid_data/anatomies_v6/_records/per_anatomy/checks4")')
rep('n_bad = sum(1 for r in res if not r["ok"])\n',
    '# ---- 1.1 (s3_fix): the s3_fix block and the derived flags\n'
    'S3 = {n: get(p, "v6_checks.s3_fix") for n, p in P.items() if get(p, "v6_checks.s3_fix")}\n'
    'check("s3_fix.counts", B["s3_fix"]["counts"], tally(v.get("kind") for v in S3.values()))\n'
    'check("s3_fix.changed (names, kind)", sorted((x["anatomy"], x["kind"]) for x in B["s3_fix"]["changed"]),\n'
    '      sorted((sh(n), v.get("kind")) for n, v in S3.items()))\n'
    'check("s3_fix.bake_lattice_shift", B["s3_fix"]["bake_lattice_shift"],\n'
    '      sum(1 for n in S3 if any(float(x) for x in (get(P[n], "bake_v6.lattice_shift_mm") or [0, 0, 0]))))\n'
    'check("s3_fix.bake_route_island_bridge", B["s3_fix"]["bake_route_island_bridge"],\n'
    '      sum(1 for n in S3 if get(P[n], "bake_v6.route_island_bridge.bridged")))\n'
    'check("s3_fix.route_islands_bridged", B["s3_fix"]["route_islands_bridged"],\n'
    '      sum(len(get(P[n], "bake_v6.route_island_bridge.bridged") or []) for n in S3))\n'
    'for n in S3:\n'
    '    check("per_anatomy.%s.derived_flags == provenance v6_checks.derived_flags" % n, B["per_anatomy"][n]["derived_flags"],\n'
    '          get(P[n], "v6_checks.derived_flags"))\n'
    'orient_rc = collections.Counter()\n'
    'for n in names:\n'
    '    if n in S3:\n'
    '        fl = [f for f in get(P[n], "v6_checks.derived_flags") or [] if f.startswith("orient:")]\n'
    '    else:\n'
    '        c4p = os.path.join(a.checks4, n + ".json")\n'
    '        fl = ["orient:%s" % f for f in (get(load(c4p), "orient.accepted_flags") or [])] if os.path.exists(c4p) else []\n'
    '    for f in fl:\n'
    '        orient_rc["orient:" + f.split(":")[1]] += 1\n'
    'check("flags.derived_flags orient:*", {k: v for k, v in B["flags"]["derived_flags"].items() if k.startswith("orient:")},\n'
    '      dict(sorted(orient_rc.items())))\n'
    'check("flags.derived_flags left_pose_map_bounded_dev == carotid.left_fork.map_bounded_deviation",\n'
    '      B["flags"]["derived_flags"].get("left_pose_map_bounded_dev", 0), len(B["carotid"]["left_fork"]["map_bounded_deviation"]))\n'
    'check("per_anatomy derived_flags tally == flags.derived_flags",\n'
    '      dict(sorted(collections.Counter((f if not f.startswith("orient:") else "orient:" + f.split(":")[1])\n'
    '                                      for r in B["per_anatomy"].values() for f in r.get("derived_flags") or []).items())),\n'
    '      dict(sorted(B["flags"]["derived_flags"].items())))\n'
    'n_bad = sum(1 for r in res if not r["ok"])\n')
open(F, "w").write(s)
print("ok")
