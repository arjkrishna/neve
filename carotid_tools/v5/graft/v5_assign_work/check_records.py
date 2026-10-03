#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\check_records.py (= /sp/v5_va/v5_assign_work/check_records.py in the build containers)
"""v5_assign: check every donor record in v5_reg/donors_v5 against the final eligible list and for internal sanity.
Output: /sp/v5_va/results/v5_assign_records.json (+ a table on stdout)."""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, "/repo/carotid_tools"); sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from graft_siphon import arclength
ROOT = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
OLD = "/repo/carotid_data/va_sources/v5_reg/donors"
EL = json.load(open("/sp/v5_va/results/v5_eligible_final.json"))

def md5(p):
    h = hashlib.md5(); h.update(open(p, "rb").read()); return h.hexdigest()

def uid(p):
    return "vmr0248" if p["source"] == "vmr0248" else (p["case"] if p["case"].startswith(p["source"]) else "%s_%s" % (p["source"], p["case"]))

rows, bad = [], []
keys = {}
for p in EL["primary"] + EL["reserve"]:
    u = uid(p)
    keys.setdefault(p["patient_key"], []).append(u)
    unit = json.load(open(os.path.join(ROOT, u, "record.json")))
    for s in "RL":
        r = json.load(open(os.path.join(ROOT, u, s + ".json")))
        sd = p["sides"][s]
        chk, notes = {}, []
        surf = r["surface"]["path"]
        m_copy = md5(surf); m_src = md5(r["surface"]["src"])
        exp = (sd.get("wall_files") or {}).get("surface_md5")
        chk["surface_md5_copy_eq_record"] = m_copy == r["surface"]["md5"]
        chk["surface_md5_eq_source"] = m_copy == m_src
        chk["surface_md5_eq_eligible"] = (exp is None) or (m_copy == exp)
        chk["surface_src_eq_eligible_path"] = sd.get("wall_files", {}).get("surface_vtp") in (None, r["surface"]["src"])
        P = np.asarray(r["centerline"]["points"], float)
        L = float(arclength(P)[-1])
        chk["length_consistent"] = abs(L - r["centerline"]["length_mm"]) < 0.01
        lm = r["landmarks"]
        seq = [lm.get(k) for k in ("C6", "C2", "C1", "dura", "end")]
        have = [v for v in seq if v is not None]
        chk["landmarks_complete"] = all(v is not None for v in seq)
        chk["landmarks_ordered"] = all(a < b for a, b in zip(have, have[1:]))
        chk["splice_window_V3_exists"] = (lm.get("C2") is not None and lm.get("dura") is not None and
                                          min(lm["dura"], L - 10.0) - lm["C2"] >= 10.0)
        tk = r["takeoff_native"]
        chk["takeoff_native_finite"] = all(np.isfinite(tk[k]) for k in ("5", "10"))
        chk["chirality_plus1"] = r["frame"]["chirality_sign"] == 1.0
        qv = r["qc"]["verdict"]
        want = {"donor": "pass", "donor_review": "review", None: "pass"}.get(sd.get("qc_status"), "?")
        chk["qc_verdict_eq_eligible"] = qv == want
        chk["usable"] = unit["sides"][s].get("usable") is True
        chk["radii_positive"] = bool(np.all(np.asarray(r["centerline"]["radii_misr"], float) > 0))
        chk["ostium_to_axis_le_10mm"] = r["ostium"]["ostium_to_axis_mm"] <= 10.0
        if not all(chk.values()):
            bad.append((u, s, [k for k, v in chk.items() if not v]))
        row = {"unit": u, "side": s, "class": p["class"], "source": p["source"], "wall_tag": r["surface"]["seg_tag"],
               "surface_md5": m_copy, "len_mm": round(L, 1), "landmarks": {k: lm.get(k) for k in ("C6", "C2", "C1", "dura", "end")},
               "takeoff_native_10": round(tk["10"], 1), "takeoff_native_5": round(tk["5"], 1),
               "subclavian_elev": round(r["subclavian"]["elevation_fwd5_deg"], 1), "ostium_to_axis_mm": round(r["ostium"]["ostium_to_axis_mm"], 2),
               "r_misr_median": r["centerline"]["r_misr_median"], "genus": r["surface"].get("genus_measure_json"),
               "qc": qv, "score": r["coarseness"]["score"], "builder_score": r["coarseness"].get("builder_score"),
               "flag_level": (r.get("eligible") or {}).get("flag_level"), "subclavian_src": r["subclavian"]["src"][-80:],
               "checks": chk}
        # comparison with the earlier test record of the same unit (graft_reg 01:25-01:52), where one exists
        op = os.path.join(OLD, u, s + ".json")
        if os.path.isfile(op):
            o = json.load(open(op))
            Po = np.asarray(o["centerline"]["points"], float)
            row["vs_test_record"] = {"same_surface_md5": o["surface"]["md5"] == m_copy,
                                     "same_centerline": Po.shape == P.shape and bool(np.allclose(Po, P)),
                                     "takeoff10_old": round(o["takeoff_native"]["10"], 1), "len_old": round(o["centerline"]["length_mm"], 1),
                                     "seg_tag_old": o["surface"].get("seg_tag")}
        rows.append(row)
dupkeys = {k: v for k, v in keys.items() if len(v) > 1}
out = {"what": "v5_assign donor record checks (donors_v5 vs results/v5_eligible_final.json)", "n_units": len(EL["primary"]) + len(EL["reserve"]),
       "n_sides": len(rows), "n_sides_all_checks_pass": sum(all(r["checks"].values()) for r in rows), "failures": bad,
       "duplicate_patient_keys": dupkeys, "rows": rows}
json.dump(out, open("/sp/v5_va/results/v5_assign_records.json", "w"), indent=1)
print("sides %d, all checks pass %d; failures %s; dup keys %s" % (len(rows), out["n_sides_all_checks_pass"], bad, dupkeys))
for r in rows:
    v = r.get("vs_test_record")
    print("%-18s %s %-7s %-9s len %5.1f C6 %5s C2 %5s C1 %5s dura %6s end %5.1f tk10 %5.1f sub %5.1f oax %4.1f r %.2f g %4s qc %-6s sc %.4f fl %s %s" % (
        r["unit"], r["side"], r["class"][:3], r["wall_tag"], r["len_mm"], r["landmarks"]["C6"], r["landmarks"]["C2"], r["landmarks"]["C1"],
        r["landmarks"]["dura"], r["landmarks"]["end"], r["takeoff_native_10"], r["subclavian_elev"], r["ostium_to_axis_mm"], r["r_misr_median"],
        r["genus"], r["qc"], r["score"] or 0, r["flag_level"], ("old: md5 %s cl %s tk %s" % (v["same_surface_md5"], v["same_centerline"], v["takeoff10_old"])) if v else ""))
