#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\cov_results.py (= /sp/v6/s3fix/cov_results.py in the build containers)
"""s3_fix: the set-level result of checks_v6 1.6's parts coverage + folds on the FINAL anatomies_v6 (all 223):
the 17 anatomies s3_fix changed from their final records (/s3/checks/final/<n>.json), the other 206 from the 1.6 run on
the pre-fix tree (/s3/cov_v6_0/<n>.json; their geometry and unions are unchanged, so the result holds), plus the
pre-fix values of the changed ones for the before / after table. Writes /sp/v6/results/v6_coverage_s3.json.
    python3 cov_results.py <changed items file (name kind ...)> [out]"""
import glob
import json
import os
import sys
import time

items = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
out = sys.argv[2] if len(sys.argv) > 2 else "/sp/v6/results/v6_coverage_s3.json"
changed = {f[0]: f[1] for f in items}
PRE, FIN = "/s3/cov_v6_0", "/s3/checks/final"


def jl(p):
    return json.load(open(p))


def row(d):
    c = d.get("coverage") or {}
    r = {}
    for g in ("fork", "siphon"):
        a, b = (c.get("v4") or {}).get(g) or {}, (c.get("v5") or {}).get(g) or {}
        r[g] = {"deficit_v4": a.get("deficit"), "deficit_v5": b.get("deficit"), "lost_mm2_v4": a.get("lost_mm2"),
                "lost_mm2_v5": b.get("lost_mm2"), "coverage_v4": a.get("coverage"), "coverage_v5": b.get("coverage"),
                "area_mm2_v4": a.get("area_mm2"), "lost_s_runs_v4": (a.get("lost") or {}).get("s_runs_v6"),
                "ok": (c.get(g) or {}).get("ok"), "flag": (c.get(g) or {}).get("flag"),
                "deficit_max": (c.get(g) or {}).get("deficit_max")}
    fo = d.get("folds") or {}
    r["folds"] = {k: (fo.get(k) or {}).get("fold_pairs") for k in ("obj", "full")}
    r["checks_v6"] = str(d.get("script", "")).split(" (")[0]
    r["pass_all_gates"] = d.get("pass")
    return r


names = sorted(os.path.basename(p)[:-5] for p in glob.glob(PRE + "/case_*.json"))
per, before = {}, {}
for n in names:
    pre = jl(os.path.join(PRE, n + ".json"))
    if n in changed:
        before[n] = row(pre)
        per[n] = dict(row(jl(os.path.join(FIN, n + ".json"))), source="s3_fix final record (%s)" % changed[n])
    else:
        per[n] = dict(row(pre), source="checks_v6 1.6 parts coverage,folds on the pre-fix anatomies_v6 (unchanged anatomy)")


def lst(rows, cond):
    return sorted([n, g, rows[n][g]["deficit_max"], rows[n][g]["lost_mm2_v4"]] for n in rows for g in ("fork", "siphon")
                  if cond(rows[n][g]))


res = {"tool": "checks_v6 1.6 parts coverage + folds", "date": time.strftime("%F %T"), "n": len(per),
       "rule": "lost real wall = v4 / v5 SHIPPED RCCA-owned wall of the piece, carried by its own transforms, > 0.3 mm from "
               "the v6 union, on the piece's own scanned surface and outside the union; deficit = lost / gated area; gate "
               "< 1 % (both references), flag >= 0.1 %",
       "after": {"pieces_ge_1pct": lst(per, lambda x: (x["deficit_max"] or 0) >= 0.01),
                 "pieces_flagged_0p1_to_1pct": lst(per, lambda x: 0.001 <= (x["deficit_max"] or 0) < 0.01),
                 "pieces_gt0_lt_0p1pct": len(lst(per, lambda x: 0 < (x["deficit_max"] or 0) < 0.001)),
                 "gate_pass_anatomies": sum(1 for n in per if per[n]["fork"]["ok"] and per[n]["siphon"]["ok"]),
                 "obj_fold_pairs_total": sum(per[n]["folds"]["obj"] or 0 for n in per),
                 "full_fold_pairs": sorted([n, per[n]["folds"]["full"]] for n in per if per[n]["folds"]["full"])},
       "before_on_changed": before, "changed": changed, "per_anatomy": per}
res["before"] = {"pieces_ge_1pct": lst({**{n: per[n] for n in per if n not in changed}, **before}, lambda x: (x["deficit_max"] or 0) >= 0.01)}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n", "after", "before")}, indent=1))
