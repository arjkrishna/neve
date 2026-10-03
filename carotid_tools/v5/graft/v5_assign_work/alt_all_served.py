#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\alt_all_served.py (= /sp/v5_va/v5_assign_work/alt_all_served.py in the build containers)
"""v5_assign: alternative B -- PRIMARIES ONLY, every TopBrain patient must get a primary with >= 1 passing VA
(no reserve); same lexicographic order. Compared with the chosen plan (A: lexicographic optimum over primaries, the
patient it leaves unserved gets a reserve). Output /sp/v5_va/results/v5_assign_alt_all_served.json"""
import json, sys
import numpy as np
sys.path.insert(0, "/sp/v5_va")
import graft_v5_assign as GA
from scipy.optimize import LinearConstraint, Bounds, milp
GA.CFG.update(donor_root="/repo/carotid_data/va_sources/v5_reg/donors_v5", score="v5")
DRY = "/repo/carotid_data/va_sources/v5_reg/dry_v5"
info = GA.anatomies()
units = {u: v for u, v in GA.load_units().items() if u not in ("ia172_172", "song_AHMU1218138")}
U, pats, W, detail, missing = GA.score_matrix(units, info, DRY)
nu, npat = len(U), len(pats)
nv = nu * npat
rows, lb, ub = [], [], []
for i in range(nu):
    a = np.zeros(nv); a[i * npat:(i + 1) * npat] = 1; rows.append(a); lb.append(0); ub.append(1)
for j in range(npat):
    a = np.zeros(nv); a[j::npat] = 1; rows.append(a); lb.append(1); ub.append(1)        # every patient served
ubx = np.array([1.0 if detail[(u, P)]["vas_pass"] > 0 else 0.0 for u in U for P in pats])
opt = []
for t in GA.LEX_TERMS:
    c = np.array([float(detail[(u, P)][t]) for u in U for P in pats])
    res = milp(-c, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv), bounds=Bounds(0, ubx))
    if not res.success:
        print("infeasible at", t, res.message); sys.exit(1)
    x = np.round(res.x); o = float(c @ x); opt.append({"term": t, "optimum": round(o, 4)})
    rows.append(c); lb.append(o - (0.5 if t != "coarse_sum" else 1e-6)); ub.append(np.inf)
X = x.reshape(nu, npat)
asg = {U[i]: pats[j] for i in range(nu) for j in range(npat) if X[i, j] > 0.5}
A = json.load(open("/sp/v5_va/results/graft_reg_assign_v5.json"))
chosen = {t["unit"]: t["patient"] for t in A["assignment"]}
diff = {u: (chosen.get(u), asg.get(u)) for u in set(chosen) | set(asg) if chosen.get(u) != asg.get(u)}
out = {"what": __doc__, "optima_B": opt, "optima_A_primaries": A["lexicographic"]["optima"],
       "A_reserve": A["reserve"]["used"], "assignment_B": asg, "differences_A_vs_B": diff,
       "B_rows": {u: detail[(u, P)] for u, P in asg.items() if u in diff}}
json.dump(out, open("/sp/v5_va/results/v5_assign_alt_all_served.json", "w"), indent=1)
print("B optima:", opt); print("A optima (primaries):", A["lexicographic"]["optima"]); print("diff (A, B):", diff)
for u in diff:
    if u in asg: print("  B", u, asg[u], detail[(u, asg[u])])
