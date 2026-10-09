#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\s3_set.py (= /sp/v6/s3fix/s3_set.py in the build containers)
"""s3_fix: the set-level checks after the fix in one small file (/sp/v6/results/v6_s3_set.json), for BUILD_v6.json's
s3_fix block and CAROTID_V6.md: split (v5_split exit, compared / lowered, as s2_checks reads it), the training loader over
the final folder (loader_set), and SOFA on the RCCA (sofa_summary.py's file: the 9 re-run anatomies, all-223 and the
25-anatomy totals)."""
import json

out = {}
setlog = open("/s3/set/set.log").read().splitlines()
sp = json.load(open("/s3/set/split_s3.json"))
b = sp.get("baseline") or {}
out["split"] = {"file": "results/s3_fix/split_s3.json", "run": [l for l in setlog if " split rc " in l][-1],
                "n_anatomies": sp.get("n_anatomies"), "topbrain_patients": sp.get("topbrain_patients"),
                "baseline": b.get("path"), "compared": b.get("compared"), "lowered": b.get("lowered")}
ld = json.load(open("/s3/set/loader_s3.json"))
out["loader"] = {"file": "results/s3_fix/loader_s3.json", "run": [l for l in setlog if " loader rc " in l][-1],
                 **{k: ld.get(k) for k in ("anatomy_count", "folders", "names_equal_folders", "fingerprints_unique",
                                           "schedule_covers_all_in_one_pass", "checked", "ok", "pass", "rcca_len_mm")}}
so = json.load(open("/s3/sofa_summary_final.json"))
out["sofa_rcca"] = {"file": "results/s3_fix/sofa_rcca_summary.json",
                    "changed_passed": {n: v.get("passed") for n, v in so["changed"].items()},
                    "all_before": so["all_before"], "all_after": so["all_after"], "set25_v6_before": so["set25_v6_before"],
                    "set25_v6_after": so["set25_v6_after"], "set25_v5": so.get("set25_v5"), "set25_changed": so["set25_changed"]}
json.dump(out, open("/sp/v6/results/v6_s3_set.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
