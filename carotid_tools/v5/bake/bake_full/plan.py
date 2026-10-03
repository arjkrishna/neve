#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\plan.py (= /sp/v5_va/bake_full/plan.py in the build containers)
"""v5_bake_full: the bake order -- hardest first, so the ladder cases do not finish last.

Per anatomy: the donor sides (provenance xform vaC_R / vaC_L) and each side's stage-A surface genus
(donors_v5/<unit>/<side>.json surface), whether the probe (bake_probe_b) needed a local window for that donor side,
v4's bake seconds / attempts / far clamp. Order key (descending): number of donor sides that needed a window in the
probe, max donor genus, sum of donor genus, v4 bake seconds. Writes /sp/v5_va/bake_full/order.txt and plan.json.
    python3 plan.py      (neve-build-meshlab)
"""
import glob
import json
import os

A = "/repo/carotid_data/anatomies_v5"
V4 = "/repo/carotid_data/anatomies_v4"
DON = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
OUT = "/sp/v5_va/bake_full"
# probe B (results/v5_bake_probe.json): the donor sides that needed one local window
WINDOWED = {("song_AHMU1218094", "L"), ("song_AHMU1218002", "R"), ("ia172_015", "R"), ("ia172_159", "L")}


def donor_genus(unit, side):
    p = os.path.join(DON, unit, side + ".json")
    try:
        d = json.load(open(p))
    except Exception:                                                    # noqa: BLE001
        return None
    s = d.get("surface") or {}
    for k in ("genus_measure_json", "genus_if_one_component", "genus", "genus_max"):
        if isinstance(s.get(k), (int, float)):
            return s[k]
    g = s.get("genus_measure_json") or s.get("genus_measure")
    if isinstance(g, dict):
        for k in ("genus", "genus_max", "max_genus"):
            if isinstance(g.get(k), (int, float)):
                return g[k]
    if isinstance(g, list) and g and isinstance(g[0], dict):
        return max(c.get("genus", 0) for c in g)
    if isinstance(g, str) and os.path.exists(g.replace("/repo/", "/repo/")):
        try:
            gg = json.load(open(g))
            if isinstance(gg, dict):
                for k in ("genus", "genus_max", "max_genus"):
                    if isinstance(gg.get(k), (int, float)):
                        return gg[k]
                if isinstance(gg.get("components"), list):
                    return max(c.get("genus", 0) for c in gg["components"])
            if isinstance(gg, list):
                return max(c.get("genus", 0) for c in gg)
        except Exception:                                                # noqa: BLE001
            return None
    return None


rows = []
for d in sorted(glob.glob(A + "/*/provenance.json")):
    name = os.path.basename(os.path.dirname(d))
    p = json.load(open(d))
    x = p["xform"]
    m4 = json.load(open(os.path.join(V4, name, "mesh_v4.json")))
    sides = []
    for key in ("vaC_R", "vaC_L"):
        xf = x.get(key)
        if not xf:
            continue
        g = donor_genus(xf["donor_id"], xf["donor_side"])
        sides.append({"key": key, "donor": xf["donor_id"], "side": xf["donor_side"], "genus": g,
                      "real_surface": xf.get("real_surface"), "len_mm": round(xf["hi_mm"] - xf["lo_mm"], 1),
                      "windowed_in_probe": (xf["donor_id"], xf["donor_side"]) in WINDOWED})
    gs = [s["genus"] or 0 for s in sides]
    rows.append({"name": name, "sides": sides, "n_windowed": sum(s["windowed_in_probe"] for s in sides),
                 "max_genus": max(gs) if gs else 0, "sum_genus": sum(gs), "v4_seconds": m4.get("seconds", 0),
                 "v4_attempts": len(m4.get("attempts", [])),
                 "v4_far_clamp": bool((m4.get("far_clamp") or {}).get("used"))})
rows.sort(key=lambda r: (-r["n_windowed"], -r["max_genus"], -r["sum_genus"], -r["v4_seconds"], r["name"]))
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "order.txt"), "w") as f:
    f.write("\n".join(r["name"] for r in rows) + "\n")
json.dump(rows, open(os.path.join(OUT, "plan.json"), "w"), indent=1)
print("anatomies", len(rows), "one-sided", sum(1 for r in rows if len(r["sides"]) == 1),
      "no genus", sum(1 for r in rows for s in r["sides"] if s["genus"] is None))
print("n_windowed>0:", sum(1 for r in rows if r["n_windowed"]), " max_genus>=10:", sum(1 for r in rows if r["max_genus"] >= 10))
for r in rows[:12]:
    print(r["name"], r["n_windowed"], r["max_genus"], r["sum_genus"], r["v4_seconds"],
          [(s["donor"], s["side"], s["genus"]) for s in r["sides"]])
