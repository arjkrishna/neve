#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\graft_v5_assign.py (= /sp/v5_va/graft_v5_assign.py in the build containers)
"""graft_v5_assign.py -- ONE-TO-ONE assignment of donor units (donor patients: a VA pair, or one side) to TopBrain
patients, from the dry runs of graft_v5_reg.py, and the per-anatomy PLAN a bake driver consumes.

Why one-to-one: v4's maximal overlap-free train / test split (~170 = 153 / 17) groups anatomies by the parts they share.
Every anatomy of one TopBrain patient already shares that patient's ICA / VA pieces, so a donor used ONLY with one
TopBrain patient adds no new sharing edge and the split stays as in v4. A donor used with two patients would merge
their groups.

Inputs
  donors   /repo/carotid_data/va_sources/v5_reg/donors/<unit>/record.json (+ R.json / L.json)   graft_v5_reg_donor.py
  dry runs /repo/carotid_data/va_sources/v5_reg/dry/<unit>/<anatomy>.json                     graft_v5_reg.py --unit
           (every unit against every anatomy, follow handedness, every gate)
  patients provenance.json "patient" of each v4 anatomy (the TopBrain patient), "handedness"

Score of unit u at TopBrain patient P (over every anatomy of P and both host sides)
  W(u, P) = 1e4 * (VAs passing every gate) + 100 * (sum of the donor sides' coarseness score over those VAs)
            + 0.1 * (passing VAs whose take-off is preserved)
  i.e. (a) gate passes first, (b) coarseness (union > 0.3 mm on the CT-edge wall, contamination removed; tube 0),
  take-off preservation only as a weak tie-break.

Solvers
  strict (default; the brief's rule)  Hungarian (scipy linear_sum_assignment, maximise W) on units x patients: each
           unit to at most one patient, each patient at most one unit. Pairs with 0 passes are dropped.
  composite (reported beside it)      ILP (scipy milp): each unit to at most one patient; each patient at most one
           unit supplying a donor RIGHT VA and at most one supplying a donor LEFT VA -- so two half-pairs (a right-only
           and a left-only donor) can share a patient. Still split-safe (each donor patient used with one TopBrain
           patient). Scores are per-side additive (each half was dry-run with v4's other VA).
Then the PLAN: per anatomy and host side, the assigned donor side (follow rule) if its dry run passed, else v4's VA
kept (flag + reason). The plan is VERIFIED by re-running graft_v5_reg.build on every anatomy with the plan as a whole
(the other VA as finally written, which may come from another unit or be v4's); a side failing there is dropped to v4
and the anatomy re-run, until stable (<= 3 rounds).

Outputs
  /repo/carotid_data/va_sources/v5_reg/plan_<tag>.json     {"anatomies": {name: {"right": spec|null, "left": ...,
                                                             "flags": [...]}}, "assignment": ..., "version"}
                                                            -> graft_v5_reg.py --plan <file> --out <dir>; bake_v5.py
  /sp/v5_va/results/graft_reg_assign_<tag>.json             assignment table, scores, unassigned, v4-kept anatomies

    python3 /sp/v5_va/graft_v5_assign.py [--tag test] [--solver strict|composite] [--units a,b] [--no-verify]
eve-training-fixed (numpy, scipy >= 1.9).

v5_assign extension (1.1, 2026-10-02; the 1.0 behaviour is unchanged without the new options):
  --donor-root DIR     registry folder (default v5_reg/donors); --dry is already an option; the verification uses DIR too
  --score v5           lexicographic objective of the v5 assignment brief, per (unit, patient) over every anatomy of
                       the patient (follow rule; the unit dry run is the assembled anatomy for a strict pair):
                         0. TopBrain patients served (>= 1 passing VA) by a primary unit: reserves only where no
                            one-to-one assignment of the primaries can serve a patient
                         1e7 x anatomies with BOTH VAs passing every gate                                   (a)
                       + 1e5 x VAs passing + 2e3 x RVAs passing (the RVA is the training-reachable side)
                       + 1e2 x sum over passing VAs of the donor side's coarseness score (tally contact-free
                         union > 0.3 mm): coarser pairs go to the patients with more anatomies          (b)
                       + 1e-4 x sum over passing VAs of (2 - flag level) (0 unflagged, 1 review flags,
                         2 provisional-limit / truncated wall / V1 rescue / Siemens 1.0 mm)                 (c)
                       + 1e-6 x passing VAs with the take-off preserved                                 (tie-break)
                       solved EXACTLY in that order (solve_lex: milp, maximise one term, fix it, next term); a summed
                       weight cannot keep the order over 25 pairs, so the weights only rank pairs in the tables and
                       the summed-weight Hungarian is reported beside
  --reserve-units a,b  RESERVE units (ordered): used only for patients that the one-to-one optimum over the
                       primary units leaves with no gate-passing donor (second Hungarian, reserves x those patients)
  patient_key          enforced (verify_graft_reg item 10): two units with the same donor patient key are an error
  plan entries carry a "v5_assign" provenance block (patient, unit, class, sides served, flags) that
  graft_v5_reg.py --plan --out writes into provenance.v5_va.assignment
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/sp/v5_va")

VERSION = "graft_v5_assign 1.1 (2026-10-02; v5_assign: --donor-root, --score v5, --reserve-units, patient_key)"
ROOT = "/repo/carotid_data/va_sources/v5_reg"
V4 = "/repo/carotid_data/anatomies_v4"
RES = "/sp/v5_va/results"
W_PASS, W_COARSE, W_TAKEOFF = 1e4, 100.0, 0.1
# --score v5 (1.1)
W5 = {"both": 1e7, "va": 1e5, "rva": 2e3, "coarse": 1e2, "unflag": 1e-4, "takeoff": 1e-6}
CFG = {"donor_root": os.path.join(ROOT, "donors"), "score": "v1"}


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, p)


def anatomies():
    names = sorted(d for d in os.listdir(V4) if os.path.isdir(os.path.join(V4, d, "Centrelines_comb")))
    info = {}
    for n in names:
        p = jload(os.path.join(V4, n, "provenance.json"))
        info[n] = {"patient": p["patient"], "handedness": p["handedness"]}
    return info


def donor_side_for(host_side, mirrored):
    """follow rule: native anatomies take the same donor side; mirrored ones the opposite side, x-mirrored."""
    if mirrored:
        return ("L" if host_side == "right" else "R"), True
    return ("R" if host_side == "right" else "L"), False


def load_units(only=None, exclude_review=False):
    units = {}
    droot = CFG["donor_root"]
    for u in sorted(os.listdir(droot)):
        f = os.path.join(droot, u, "record.json")
        if not os.path.isfile(f) or (only and u not in only):
            continue
        rec = jload(f)
        sides = {}
        for s, v in rec["sides"].items():
            if v.get("built") and v.get("usable") and not (exclude_review and v.get("verdict") == "review"):
                sides[s] = v
        units[u] = {"record": rec, "sides": sides, "patient_key": None}
        for s in sides:
            sr = jload(os.path.join(droot, u, "%s.json" % s))
            units[u]["patient_key"] = sr.get("patient_key")
            units[u].setdefault("coarse", {})[s] = float(sr["coarseness"].get("score") or 0.0)
            units[u].setdefault("wall", {})[s] = sr["wall_class"]
            el = sr.get("eligible") or {}
            units[u].setdefault("flag_level", {})[s] = int(el.get("flag_level") or 0)
            units[u]["class"] = el.get("class")
    return {u: v for u, v in units.items() if v["sides"]}


def side_pass(row, side):
    sd = (row or {}).get("sides", {}).get(side) or {}
    return bool(sd.get("pass")), sd


def score_matrix(units, info, dry, pats=None):
    pats = pats or sorted({v["patient"] for v in info.values()})
    U = sorted(units)
    W = np.zeros((len(U), len(pats)))
    detail = {}
    missing = Counter()
    for i, u in enumerate(U):
        for j, P in enumerate(pats):
            npass, coarse, ntk, per = 0, 0.0, 0, {"R": 0, "L": 0}
            nvas, nboth, nrva, unflag, nanat = 0, 0, 0, 0, 0
            for n, inf in info.items():
                if inf["patient"] != P:
                    continue
                nanat += 1
                f = os.path.join(dry, u, n + ".json")
                row = jload(f) if os.path.isfile(f) else None
                if row is None:
                    missing[u] += 1
                both = True
                for side in ("right", "left"):
                    ds, _ = donor_side_for(side, inf["handedness"] == "mirrored")
                    if ds not in units[u]["sides"]:
                        both = False
                        continue
                    nvas += 1
                    ok, sd = side_pass(row, side)
                    if ok:
                        npass += 1
                        per[ds] += 1
                        nrva += side == "right"
                        coarse += units[u]["coarse"].get(ds, 0.0)
                        unflag += 2 - units[u].get("flag_level", {}).get(ds, 0)
                        ntk += 0 if sd.get("takeoff_not_preserved") else 1
                    else:
                        both = False
                nboth += both
            if CFG["score"] == "v5":
                W[i, j] = (W5["both"] * nboth + W5["va"] * npass + W5["rva"] * nrva + W5["coarse"] * coarse
                           + W5["unflag"] * unflag + W5["takeoff"] * ntk)
            else:
                W[i, j] = W_PASS * npass + W_COARSE * coarse + W_TAKEOFF * ntk
            detail[(u, P)] = {"anatomies": nanat, "anatomies_both_pass": nboth, "vas_candidate": nvas, "vas_pass": npass,
                              "rva_pass": nrva, "lva_pass": npass - nrva, "by_donor_side": per,
                              "coarse_sum": round(coarse, 4), "unflag_points": unflag, "takeoff_preserved": ntk,
                              "served": int(npass > 0), "W": round(float(W[i, j]), 6)}
    return U, pats, W, detail, missing


def solve_strict(U, pats, W, npass=None):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(W, maximize=True)
    if npass is not None:                       # 1.1: keep a pair only if at least one VA passes there
        return {U[i]: pats[j] for i, j in zip(r, c) if npass[i, j] > 0}
    return {U[i]: pats[j] for i, j in zip(r, c) if W[i, j] >= W_PASS}


# "served" first: primaries serve as many TopBrain patients as any one-to-one assignment can (V5_BUILD_LOG 14:25:
# "25 primary pairs for 25 patients; patients left without any passing donor: reserve, else keep v4"); then (a), (b), (c)
LEX_TERMS = ("served", "anatomies_both_pass", "vas_pass", "rva_pass", "coarse_sum", "unflag_points", "takeoff_preserved")


def solve_lex(U, pats, detail):
    """1.1 (--score v5): EXACT lexicographic one-to-one assignment (a summed float weight cannot keep the order over 25
    pairs): maximise term 1, fix it (>= optimum - tol), maximise term 2, ... (scipy milp, x_up binary, each unit <= 1
    patient, each patient <= 1 unit). Terms: LEX_TERMS. Returns (assignment of pairs with >= 1 passing VA, optima)."""
    from scipy.optimize import LinearConstraint, Bounds, milp
    nu, npat = len(U), len(pats)
    nv = nu * npat
    rows, lb, ub = [], [], []
    for i in range(nu):
        a_ = np.zeros(nv); a_[i * npat:(i + 1) * npat] = 1
        rows.append(a_); lb.append(0); ub.append(1)
    for j in range(npat):
        a_ = np.zeros(nv); a_[j::npat] = 1
        rows.append(a_); lb.append(0); ub.append(1)
    optima = []
    x = None
    for t in LEX_TERMS:
        c = np.array([float(detail[(u, P)][t]) for u in U for P in pats])
        res = milp(-c, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv),
                   bounds=Bounds(0, 1), options={"time_limit": 120})
        if not res.success:
            raise RuntimeError("milp failed at %s: %s" % (t, res.message))
        x = np.round(res.x)
        opt = float(c @ x)
        optima.append({"term": t, "optimum": round(opt, 6)})
        tol = 0.5 if t != "coarse_sum" else 1e-6
        rows.append(c); lb.append(opt - tol); ub.append(np.inf)
    X = x.reshape(nu, npat)
    assign = {U[i]: pats[j] for i in range(nu) for j in range(npat)
              if X[i, j] > 0.5 and detail[(U[i], pats[j])]["vas_pass"] > 0}
    return assign, optima


def solve_composite(U, pats, W, units):
    from scipy.optimize import LinearConstraint, Bounds, milp
    nu, npat = len(U), len(pats)
    nv = nu * npat
    c = -W.reshape(-1)
    rows, lb, ub = [], [], []
    for i in range(nu):                                   # each unit to at most one patient
        a = np.zeros(nv)
        a[i * npat:(i + 1) * npat] = 1
        rows.append(a); lb.append(0); ub.append(1)
    for j in range(npat):                                 # per patient: one donor-R supplier, one donor-L supplier
        for s in ("R", "L"):
            a = np.zeros(nv)
            for i, u in enumerate(U):
                if s in units[u]["sides"]:
                    a[i * npat + j] = 1
            rows.append(a); lb.append(0); ub.append(1)
    res = milp(c, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError("milp failed: %s" % res.message)
    x = np.round(res.x).reshape(nu, npat)
    return {U[i]: pats[j] for i in range(nu) for j in range(npat) if x[i, j] > 0.5 and W[i, j] >= W_PASS}


def make_plan(assign, units, info, dry):
    by_pat = defaultdict(list)
    for u, P in assign.items():
        by_pat[P].append(u)
    plan = {}
    for n, inf in sorted(info.items()):
        mir = inf["handedness"] == "mirrored"
        entry = {"patient": inf["patient"], "handedness": inf["handedness"], "flags": []}
        for side in ("right", "left"):
            ds, m = donor_side_for(side, mir)
            sup = [u for u in by_pat.get(inf["patient"], []) if ds in units[u]["sides"]]
            if not sup:
                entry[side] = None
                entry["flags"].append("%s: v4 VA kept -- no donor %s VA assigned to %s" % (side, ds, inf["patient"]))
                continue
            u = sup[0]
            f = os.path.join(dry, u, n + ".json")
            row = jload(f) if os.path.isfile(f) else None
            ok, sd = side_pass(row, side)
            if not ok:
                entry[side] = None
                why = sd.get("final_fail") or sd.get("why") or ("dry run missing" if row is None else "failed")
                entry["flags"].append("%s: v4 VA kept -- donor %s %s fails (%s)" % (side, u, ds, why))
                continue
            entry[side] = {"donor_id": u, "donor_side": ds, "mirror": m}
        sup_all = by_pat.get(inf["patient"], [])
        entry["v5_assign"] = {"tool": VERSION, "patient": inf["patient"], "handedness": inf["handedness"],
                              "assigned_units": sup_all,
                              "assigned_class": {u: units[u].get("class") for u in sup_all},
                              "rule": "strict one-to-one donor pair <-> TopBrain patient; follow handedness (native: "
                                      "donor R -> RVA, donor L -> LVA; mirrored: donor L mirrored -> RVA, donor R "
                                      "mirrored -> LVA); a side whose donor fails a gate keeps v4's VA (flagged)"}
        plan[n] = entry
    return plan


def verify_plan(plan, workers, rounds=3):
    """Re-run graft_v5_reg.build with each anatomy's whole plan; drop failing sides to v4 until stable."""
    import graft_v5_reg as GR
    from multiprocessing import Pool
    log = []
    todo = sorted(plan)
    final_rows = {}
    for rnd in range(rounds):
        jobs = [(n, {"right": plan[n]["right"], "left": plan[n]["left"]}, None, None) for n in todo]
        with Pool(workers, initializer=GR._init, initargs=({"compat": False, "hand": "follow",
                                                             "donor_root": CFG["donor_root"], "unit": None},)) as pool:
            rows = pool.map(GR.job, jobs, chunksize=1)
        changed = []
        for n, status, out in rows:
            final_rows[n] = out
            if status != "ok":
                for side in ("right", "left"):
                    if plan[n][side]:
                        plan[n][side] = None
                        plan[n]["flags"].append("%s: v4 VA kept -- verification error %s" % (side, out.get("error")))
                changed.append(n)
                continue
            for side in ("right", "left"):
                if plan[n][side] and not out["sides"][side].get("pass"):
                    why = out["sides"][side].get("final_fail") or out["sides"][side].get("why")
                    plan[n]["flags"].append("%s: v4 VA kept -- fails in the assembled anatomy (round %d: %s)" % (side, rnd + 1, why))
                    plan[n][side] = None
                    changed.append(n)
        log.append({"round": rnd + 1, "anatomies": len(todo), "changed": sorted(set(changed))})
        if not changed:
            break
        todo = sorted(set(changed))
    return plan, final_rows, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="test")
    ap.add_argument("--solver", default="strict", choices=("strict", "composite"))
    ap.add_argument("--units", default=None, help="comma list (default: every unit with a usable side)")
    ap.add_argument("--exclude-review", action="store_true", help="drop sides whose QC verdict is 'review'")
    ap.add_argument("--dry", default=os.path.join(ROOT, "dry"))
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--donor-root", default=os.path.join(ROOT, "donors"))
    ap.add_argument("--score", default="v1", choices=("v1", "v5"))
    ap.add_argument("--reserve-units", default=None, help="ordered comma list of RESERVE units (strict + v5 only)")
    a = ap.parse_args()
    CFG.update(donor_root=a.donor_root, score=a.score)
    t0 = time.time()
    info = anatomies()
    reserves = [u for u in (a.reserve_units or "").split(",") if u]
    only = set(a.units.split(",")) if a.units else None
    units_all = load_units(None if only is None else (only | set(reserves)), a.exclude_review)
    units = {u: v for u, v in units_all.items() if u not in reserves}
    keys = Counter(v["patient_key"] for v in units_all.values())
    dup = [k for k, c in keys.items() if c > 1]
    if dup and a.score == "v5":
        raise SystemExit("one donor patient under two unit ids (patient_key %s): refusing (split safety)" % dup)
    U, pats, W, detail, missing = score_matrix(units, info, a.dry)
    reserve_log = None
    lex = None
    if a.solver == "strict" and a.score == "v5":
        assign, lex = solve_lex(U, pats, detail)
        NP = np.array([[detail[(u, P)]["vas_pass"] for P in pats] for u in U])
        hung = solve_strict(U, pats, W, NP)               # the summed-weight Hungarian, reported beside
        lex = {"optima": lex, "hungarian_summed_weight_same_assignment": hung == assign,
               "hungarian_summed_weight": hung}
        unserved = [P for P in pats if P not in assign.values()]
        reserve_log = {"reserves": reserves, "patients_unserved_by_primaries": unserved, "used": {}}
        if reserves and unserved:
            runits = {u: units_all[u] for u in reserves if u in units_all}
            RU, rp, RW, rdetail, rmissing = score_matrix(runits, info, a.dry, pats=unserved)
            rassign, ropt = solve_lex(RU, rp, rdetail)
            reserve_log["used"] = rassign
            reserve_log["optima"] = ropt
            reserve_log["reserve_scores"] = {"%s@%s" % k: v for k, v in rdetail.items()}
            assign.update(rassign)
            detail.update(rdetail)
            missing.update(rmissing)
            units = dict(units, **{u: runits[u] for u in rassign})
    elif a.solver == "strict":
        assign = solve_strict(U, pats, W)
    else:
        assign = solve_composite(U, pats, W, units)
    if a.score == "v5":
        ks = Counter(units[u]["patient_key"] for u in assign)
        assert all(c == 1 for c in ks.values()), "a donor patient is assigned twice: %s" % ks
        assert len(set(assign.values())) == len(assign), "a TopBrain patient has two donors (strict)"
    # one-to-one / split safety
    used = Counter(assign.values())
    per_pat_sides = defaultdict(Counter)
    for u, P in assign.items():
        for s in units[u]["sides"]:
            per_pat_sides[P][s] += 1
    assert all(c <= 2 for c in used.values())
    assert all(c <= 1 for P in per_pat_sides for c in per_pat_sides[P].values())
    plan = make_plan(assign, units, info, a.dry)
    pre = Counter((side, plan[n][side] is not None) for n in plan for side in ("right", "left"))
    vlog, rows = [], {}
    if not a.no_verify:
        plan, rows, vlog = verify_plan(plan, a.workers)
    post = Counter((side, plan[n][side] is not None) for n in plan for side in ("right", "left"))
    for n in plan:                                       # 1.1: final sides into the provenance block
        if "v5_assign" in plan[n]:
            plan[n]["v5_assign"]["served"] = {s: plan[n][s] is not None for s in ("right", "left")}
            plan[n]["v5_assign"]["flags"] = list(plan[n]["flags"])
    table = []
    for u, P in sorted(assign.items(), key=lambda t: t[1]):
        d = detail[(u, P)]
        table.append({"unit": u, "patient": P, "sides": sorted(units[u]["sides"]),
                      "wall": units[u].get("wall"), "coarse": units[u].get("coarse"),
                      "n_anatomies": sum(1 for v in info.values() if v["patient"] == P), **d})
    pats_without = sorted(set(pats) - set(assign.values()))
    kept = {n: plan[n]["flags"] for n in plan if plan[n]["flags"]}
    full = sorted(n for n in plan if plan[n]["right"] and plan[n]["left"])
    none = sorted(n for n in plan if not plan[n]["right"] and not plan[n]["left"])
    summary = {"version": VERSION, "tag": a.tag, "solver": a.solver, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "units_considered": U, "n_units": len(U), "n_patients": len(pats), "assignment": table,
               "units_unassigned": sorted(set(U) - set(assign)), "patients_without_donor": pats_without,
               "duplicate_patient_keys": dup, "dry_missing_per_unit": dict(missing),
               "vas_replaced_before_verify": {"right": pre[("right", True)], "left": pre[("left", True)]},
               "vas_replaced_after_verify": {"right": post[("right", True)], "left": post[("left", True)]},
               "anatomies_both_sides_v5": len(full), "anatomies_one_side_v5": len(plan) - len(full) - len(none),
               "anatomies_v4_va_kept_both": len(none), "verify_log": vlog,
               "split_safety": {"units_per_patient_max": max(used.values()) if used else 0,
                                "each_unit_one_patient": True,
                                "note": "each donor unit (donor patient) is used with exactly one TopBrain patient"},
               "weights": ({"pass": W_PASS, "coarse": W_COARSE, "takeoff": W_TAKEOFF} if a.score == "v1" else W5),
               "score": a.score, "donor_root": a.donor_root, "dry": a.dry, "reserve": reserve_log, "lexicographic": lex,
               "units_class": {u: units_all[u].get("class") for u in units_all},
               "score_matrix": {"units": U, "patients": pats, "W": W.round(3).tolist(),
                                "vas_pass": [[detail[(u, P)]["vas_pass"] for P in pats] for u in U]},
               "flags_per_anatomy": kept, "seconds": round(time.time() - t0, 1)}
    plan_doc = {"version": VERSION, "tag": a.tag, "solver": a.solver, "hand": "follow", "score": a.score,
                "donor_root": a.donor_root,
                "assignment": {t["unit"]: t["patient"] for t in table}, "anatomies": plan,
                "graft_tool": "/sp/v5_va/graft_v5_reg.py --plan <this file> --out <dir>  (then /sp/v5_va/bake_v5.py --anatomies <dir>)"}
    jdump(plan_doc, os.path.join(ROOT, "plan_%s.json" % a.tag))
    jdump(summary, os.path.join(RES, "graft_reg_assign_%s.json" % a.tag))
    if rows:
        jdump({n: {s: {k: v for k, v in (r.get("sides", {}).get(s) or {}).items() if k != "base"} for s in ("right", "left")}
               for n, r in rows.items()}, os.path.join(ROOT, "plan_%s_verify_rows.json" % a.tag))
    print("%s / %s: %d units, %d patients; assigned %d; VAs v5 R %d L %d (before verify R %d L %d); anatomies both %d, "
          "one %d, none %d; %.0f s" % (a.tag, a.solver, len(U), len(pats), len(assign), post[("right", True)],
                                       post[("left", True)], pre[("right", True)], pre[("left", True)], len(full),
                                       len(plan) - len(full) - len(none), len(none), time.time() - t0))
    for t in table:
        print("  %-20s -> %-14s sides %-6s anatomies %2d  both %2s  VAs pass %2d / %2d (R %s L %s)  coarse_sum %.3f  W %.4f" % (
            t["unit"], t["patient"], ",".join(t["sides"]), t["n_anatomies"], t.get("anatomies_both_pass", "-"),
            t["vas_pass"], t["vas_candidate"], t.get("rva_pass", "-"), t.get("lva_pass", "-"), t["coarse_sum"], t["W"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
