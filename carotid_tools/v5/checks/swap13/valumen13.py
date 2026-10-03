#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\valumen13.py (= /sp/v5_va/swap13/valumen13.py in the build containers)
"""v5_swap13: checks/va_lumen.py's one() UNCHANGED (VA lumen in the shipped OBJ, v4's route-check measures) on the 13
swapped anatomies, now (anatomies_v5) and before the swap (backup_pre_route13), with v4 alongside.
    python3 valumen13.py    -> /sp/v5_va/swap13/va_lumen13.json"""
import json
import sys
sys.path.insert(0, "/sp/v5_va/checks")
import va_lumen as L  # noqa: E402
names = [x.strip() for x in open("/sp/v5_va/swap13/names13.txt") if x.strip()]
out = {}
for n in names:
    L.A5 = "/repo/carotid_data/anatomies_v5"
    now = L.one(n)
    L.A5 = "/sp/v5_va/backup_pre_route13"
    before = L.one(n)
    out[n] = {"now": now, "before_swap": before}
    f = lambda r, t: (r[t]["whole"]["lumen_min_mm"], r[t]["whole"]["outside"], r[t]["whole"]["navigable_0.65"])
    print(n, "now RVA/LVA", f(now, "v5_RVA"), f(now, "v5_LVA"), "| before", f(before, "v5_RVA"), f(before, "v5_LVA"),
          "| v4", f(now, "v4_RVA"), f(now, "v4_LVA"), flush=True)
json.dump(out, open("/sp/v5_va/swap13/va_lumen13.json", "w"), indent=1)
print("DONE va_lumen13")
