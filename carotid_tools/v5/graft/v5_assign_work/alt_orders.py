#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\alt_orders.py (= /sp/v5_va/v5_assign_work/alt_orders.py in the build containers)
"""v5_assign: sensitivity of the one-to-one assignment to the order of the objective (primaries only, every patient
served first, as the chosen plan). Variants: chosen (served, both, VAs, RVAs, coarse, unflag, take-off); C coarse
before gates (served, coarse, both, VAs, ...); D no RVA tie-break; E unflagged before coarseness.
Output /sp/v5_va/results/v5_assign_alt_orders.json"""
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
ALL = ("served", "anatomies_both_pass", "vas_pass", "rva_pass", "coarse_sum", "unflag_points", "takeoff_preserved")

def lex(order):
    nu, npat = len(U), len(pats); nv = nu * npat
    rows, lb, ub = [], [], []
    for i in range(nu):
        a = np.zeros(nv); a[i * npat:(i + 1) * npat] = 1; rows.append(a); lb.append(0); ub.append(1)
    for j in range(npat):
        a = np.zeros(nv); a[j::npat] = 1; rows.append(a); lb.append(0); ub.append(1)
    for t in order:
        c = np.array([float(detail[(u, P)][t]) for u in U for P in pats])
        res = milp(-c, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv), bounds=Bounds(0, 1))
        x = np.round(res.x); o = float(c @ x)
        rows.append(c); lb.append(o - (0.5 if t != "coarse_sum" else 1e-6)); ub.append(np.inf)
    X = x.reshape(nu, npat)
    asg = {U[i]: pats[j] for i in range(nu) for j in range(npat) if X[i, j] > 0.5 and detail[(U[i], pats[j])]["vas_pass"] > 0}
    tot = {t: round(sum(detail[(u, P)][t] for u, P in asg.items()), 4) for t in ALL}
    return asg, tot

variants = {"chosen": ALL,
            "C_coarse_before_gates": ("served", "coarse_sum", "anatomies_both_pass", "vas_pass", "rva_pass", "unflag_points", "takeoff_preserved"),
            "D_no_rva_tiebreak": ("served", "anatomies_both_pass", "vas_pass", "coarse_sum", "unflag_points", "takeoff_preserved"),
            "E_unflagged_before_coarse": ("served", "anatomies_both_pass", "vas_pass", "rva_pass", "unflag_points", "coarse_sum", "takeoff_preserved")}
out = {"what": __doc__}
base = None
for k, order in variants.items():
    asg, tot = lex(order)
    if base is None:
        base = asg
    out[k] = {"order": order, "totals": tot, "n_units_moved_vs_chosen": sum(1 for u in set(asg) | set(base) if asg.get(u) != base.get(u)),
              "moved": {u: (base.get(u), asg.get(u)) for u in set(asg) | set(base) if asg.get(u) != base.get(u)}}
    print("%-28s %s moved %d" % (k, tot, out[k]["n_units_moved_vs_chosen"]))
json.dump(out, open("/sp/v5_va/results/v5_assign_alt_orders.json", "w"), indent=1)
