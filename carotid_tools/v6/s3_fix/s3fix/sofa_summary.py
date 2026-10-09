#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\sofa_summary.py (= /sp/v6/s3fix/sofa_summary.py in the build containers)
"""s3_fix: SOFA RCCA (sofa_rcca_v6 1.0, seeds 777/778/779 x 300) after the fix. The set = s2_checks' records for every
anatomy (/w/s2checks/sofa/rcca_v6) with the re-run records of the anatomies whose geometry s3_fix changed (/s3/sofa/<tag>).
Reports: soundness (passed) per changed anatomy, reach per changed anatomy before / after, the all-223 totals before /
after, and the 25-anatomy set (s2_checks' set25) v6 before / after against v5.
    python3 sofa_summary.py <tag> <changed names file> <out.json>"""
import json
import os
import sys

tag, names_f, out = sys.argv[1:4]
S2 = "/w/s2checks/sofa/rcca_v6"
S2V5 = "/w/s2checks/sofa/rcca_v5"
S3 = "/s3/sofa/" + tag
changed = [l.split()[0] for l in open(names_f) if l.strip()]
names = sorted(f[:-5] for f in os.listdir(S2) if f.endswith(".json"))
K = ("past_seam1_20mm", "past_bifurcation", "reached_siphon", "reached_target")


def jl(p):
    return json.load(open(p))


def tot(recs):
    t = {k: sum(r["route"][k] for r in recs) for k in K}
    t.update({"anat_" + k: sum(1 for r in recs if r["route"][k] > 0) for k in K})
    t["seeds"] = sum(r["route"]["n_seeds"] for r in recs)
    t["passed"] = sum(1 for r in recs if r.get("passed"))
    t["anatomies"] = len(recs)
    t["tip_outside_max_mm"] = max(float(r.get("tip_outside_max_mm") or 0) for r in recs)
    return t


before = {n: jl(os.path.join(S2, n + ".json")) for n in names}
after = dict(before)
per = {}
for n in changed:
    p = os.path.join(S3, n + ".json")
    if not os.path.exists(p):
        per[n] = {"missing": True}
        continue
    r = jl(p)
    after[n] = r
    per[n] = {"passed": r.get("passed"), "exception": r.get("exception"), "loader_mesh_is_this_obj": r.get("loader_mesh_is_this_obj"),
              "tip_outside_max_mm": r.get("tip_outside_max_mm"), "route_after": r["route"], "route_before": before[n]["route"],
              "max_rcca_s_mm": [before[n].get("max_rcca_s_mm"), r.get("max_rcca_s_mm")], "anatomy_dir": r.get("anatomy_dir")}
set25 = jl("/w/s2checks/sofa/set25.json")["set25"]
res = {"tool": "sofa_rcca_v6 1.0", "changed": per, "all_before": tot(list(before.values())), "all_after": tot(list(after.values())),
       "set25_v6_before": tot([before[n] for n in set25]), "set25_v6_after": tot([after[n] for n in set25]),
       "set25_changed": [n for n in set25 if n in changed]}
v5 = [os.path.join(S2V5, n + ".json") for n in set25]
if all(os.path.exists(p) for p in v5):
    res["set25_v5"] = tot([jl(p) for p in v5])
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k != "changed"}, indent=1))
for n, v in per.items():
    print(n, {k: v.get(k) for k in ("passed", "tip_outside_max_mm", "max_rcca_s_mm")}, "before", v.get("route_before"), "after", v.get("route_after"))
