# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\make_items.py (= /sp/v6/s2checks/make_items.py in the build containers)
# s2_checks: SOFA item list (order = priority): rcca v6 (all 223), d14 v6 / v5 / v4 (the v5 25-anatomy SOFA set), rcca v5 (25).
import json, os
sel = json.load(open("/sp/v5_va/checks/sofa_selection.json"))
set25 = sel["most_changed_15"] + sel["random_10"]
assert len(set25) == 25 and len(set(set25)) == 25
v6 = sorted(d for d in os.listdir("/repo/carotid_data/anatomies_v6") if d.startswith("case_"))
assert len(v6) == 223
for n in set25:
    for t in ("v6", "v5", "v4"):
        assert os.path.isdir("/repo/carotid_data/anatomies_%s/%s" % (t, n)), (t, n)
items = ["rcca:v6:" + n for n in v6] + ["d14:%s:%s" % (t, n) for t in ("v6", "v5", "v4") for n in set25] + ["rcca:v5:" + n for n in set25]
os.makedirs("/s2c/sofa", exist_ok=True)
open("/s2c/sofa/items.txt", "w").write("\n".join(items) + "\n")
json.dump({"set25": set25, "rule": sel["rule"], "source": "/sp/v5_va/checks/sofa_selection.json"}, open("/s2c/sofa/set25.json", "w"), indent=1)
print(len(items), "items")
