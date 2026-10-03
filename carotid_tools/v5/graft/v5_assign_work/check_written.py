#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\check_written.py (= /sp/v5_va/v5_assign_work/check_written.py in the build containers)
"""v5_assign: check the written (unbaked) anatomies_v5 against anatomies_v4 and the plan.
 - 223 anatomies, same names as v4; nothing but Centrelines_comb/, provenance.json (+ LICENSE files) written;
 - every non-VA centreline byte-identical to v4; a side the plan leaves null (or that failed) byte-identical to v4;
 - a replaced RVA keeps v4's subclavian prefix (points before O_h) exactly;
 - xform.vaC_R / vaC_L present iff that side is replaced, surface exists and its md5 = record's;
 - provenance: v5_va.summary, v5_va.assignment, licences, NC, kept flags; everything else in provenance = v4's;
 - plan handedness: donor side by the follow rule; one donor unit per TopBrain patient.
Output /sp/v5_va/results/v5_assign_written_check.json"""
import hashlib, json, os, sys
import numpy as np
V4 = "/repo/carotid_data/anatomies_v4"
V5 = "/repo/carotid_data/anatomies_v5"
PLAN = "/repo/carotid_data/va_sources/v5_reg/plan_v5.json"
RVA, LVA = "Centerline curve - RVA.mrk.json", "Centerline curve - LVA.mrk.json"

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

sys.path.insert(0, "/repo/carotid_tools"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from graft_siphon import read_curve

def pts(p):
    return np.asarray(read_curve(p)[1], float)

plan = json.load(open(PLAN))["anatomies"]
names4 = sorted(d for d in os.listdir(V4) if os.path.isdir(os.path.join(V4, d, "Centrelines_comb")))
names5 = sorted(d for d in os.listdir(V5) if os.path.isdir(os.path.join(V5, d)))
bad, rows = [], {}
unit_pat = {}
for n in names4:
    r = {"ok": True, "why": []}
    d5, d4 = os.path.join(V5, n), os.path.join(V4, n)
    if not os.path.isdir(d5):
        bad.append((n, "missing")); continue
    extra = sorted(set(os.listdir(d5)) - {"Centrelines_comb", "provenance.json", "LICENSE_v5_donors.txt", "LICENSE_VMR_0248.txt"})
    if extra:
        r["why"].append("unexpected files %s" % extra)
    c4 = sorted(os.listdir(os.path.join(d4, "Centrelines_comb"))); c5 = sorted(os.listdir(os.path.join(d5, "Centrelines_comb")))
    if c4 != c5:
        r["why"].append("centreline file set differs")
    for f in c4:
        if f in (RVA, LVA):
            continue
        if md5(os.path.join(d4, "Centrelines_comb", f)) != md5(os.path.join(d5, "Centrelines_comb", f)):
            r["why"].append("non-VA file changed: %s" % f)
    p4 = json.load(open(os.path.join(d4, "provenance.json"))); p5 = json.load(open(os.path.join(d5, "provenance.json")))
    v5 = p5.get("v5_va") or {}
    summ = v5.get("summary") or {}
    if not summ or not v5.get("assignment"):
        r["why"].append("no v5_va.summary / assignment")
    for k in p4:
        if k == "xform":
            continue
        if p4[k] != p5.get(k):
            r["why"].append("provenance key %s changed" % k)
    x4, x5 = p4["xform"], p5["xform"]
    for k in x4:
        if k in ("va_R", "va_L"):
            continue
        if x4[k] != x5.get(k):
            r["why"].append("xform %s changed" % k)
    pl = plan[n]
    mir = p4["handedness"] == "mirrored"
    rep = {}
    for side, f, ck, vk in (("right", RVA, "vaC_R", "va_R"), ("left", LVA, "vaC_L", "va_L")):
        replaced = bool(summ.get("replaced", {}).get(side))
        rep[side] = replaced
        same = md5(os.path.join(d4, "Centrelines_comb", f)) == md5(os.path.join(d5, "Centrelines_comb", f))
        spec = pl.get(side)
        if spec:
            want = ("L" if side == "right" else "R", True) if mir else ("R" if side == "right" else "L", False)
            if (spec["donor_side"], bool(spec["mirror"])) != want:
                r["why"].append("%s breaks follow rule" % side)
            unit_pat.setdefault(spec["donor_id"], set()).add(p4["patient"])
        if not replaced:
            if not same:
                r["why"].append("%s kept but file differs from v4" % side)
            if ck in x5:
                r["why"].append("%s kept but %s present" % (side, ck))
            if x5.get(vk) != x4.get(vk):
                r["why"].append("%s kept but xform %s changed" % (side, vk))
            if spec:
                r["why"].append("%s planned but not replaced (final gate?)" % side)
            continue
        if not spec:
            r["why"].append("%s replaced without a plan" % side)
        if same:
            r["why"].append("%s replaced but identical to v4" % side)
        xc = x5.get(ck)
        if not xc:
            r["why"].append("%s replaced but no %s" % (side, ck)); continue
        if xc.get("real_surface") is not False:
            if not os.path.isfile(xc["surface"]) or md5(xc["surface"]) != xc["surface_md5"]:
                r["why"].append("%s surface missing / md5" % side)
        if spec and xc.get("donor_id") != spec["donor_id"]:
            r["why"].append("%s donor id differs from the plan" % side)
        if side == "right":
            a4, a5 = pts(os.path.join(d4, "Centrelines_comb", f)), pts(os.path.join(d5, "Centrelines_comb", f))
            # the host subclavian prefix: v4 and v5 agree up to the anchor O_h
            anc = np.asarray(xc["anchor"], float)
            k4 = int(np.argmin(np.linalg.norm(a4 - anc, axis=1)))
            if k4 > 0 and not np.array_equal(a4[:k4], a5[:k4]):
                r["why"].append("right subclavian prefix differs from v4")
    r["replaced"] = rep
    r["ok"] = not r["why"]
    if not r["ok"]:
        bad.append((n, r["why"]))
    rows[n] = r
multi = {u: sorted(p) for u, p in unit_pat.items() if len(p) > 1}
out = {"what": "v5_assign written-anatomy check (anatomies_v5 vs v4 and plan_v5)", "n_v4": len(names4), "n_v5": len(names5),
       "names_equal": names4 == [n for n in names5 if n in names4] and set(names5) == set(names4),
       "n_ok": sum(r["ok"] for r in rows.values()), "bad": bad, "units_on_several_patients": multi,
       "replaced_right": sum(r["replaced"]["right"] for r in rows.values() if "replaced" in r),
       "replaced_left": sum(r["replaced"]["left"] for r in rows.values() if "replaced" in r)}
json.dump(out, open("/sp/v5_va/results/v5_assign_written_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "bad"}), "bad:", bad[:10])
