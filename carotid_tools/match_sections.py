#!/usr/bin/env python3
"""Decide which lower section joins which TopBrain siphon.

Not all combinations: the point is a spread of anatomies, not a cross product,
and reusing one donor everywhere would let a policy memorise it. Caps:

    each lower section  <= 4 times
    each siphon         <= 3 times with LEFT lowers and 3 with RIGHT lowers

Within those caps the pairing minimises CALIBRE MISMATCH at the seam, because
that is the one thing the graft cannot hide: the radius ramp can ease a
difference away over 25 mm, but a 2 mm ICA tip meeting a 6 mm siphon leaves
either a step or an implausible taper.

Solved in ROUNDS, not as one expanded assignment. The expanded form let a lower
take the same siphon in several of its four slots, so the pair count overstated
the number of distinct anatomies. Each round is a strict one-to-one assignment
with already-used pairs forbidden, which makes every pair unique by
construction while the capacity counters still enforce the caps.

--exclude takes pairs that failed to graft, so the plan can be re-solved with
those forbidden and the freed capacity spent on alternatives.

    python carotid_tools/match_sections.py --out carotid_data/pairing.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/eve_training/topbrain_tools")

LOWER_CAP = 5
SIPHON_CAP_PER_SIDE = 4
# Right-side siphons excluded as defective. mr_015 is the known enclosure
# failure (distal 22 mm pinched shut, 7.19 mm beyond the wall); mr_013 and
# mr_014 carry the tolerated-but-real wall excursions; mr_025 needed an RVA
# deflection. Their LEFT-derived counterparts are unaffected and stay in.
DEFECTIVE_SIPHONS = {"topcow_mr_013", "topcow_mr_014",
                     "topcow_mr_015", "topcow_mr_025"}
# The catheter is 0.7 mm across, so an ICA tip under 2 mm is not a seam, it is
# an obstruction.
MIN_TIP_D = 2.0
# Beyond ~3 mm the radius ramp stops being a repair and becomes an invented
# taper, so forbid it and let the solver find another partner.
MAX_MISMATCH_MM = 3.0


def load_lowers(manifest, ext_dir):
    man = json.load(open(manifest, encoding="utf-8"))
    ext = {}
    for f in glob.glob(os.path.join(ext_dir, "*_ica_extended.json")):
        d = json.load(open(f, encoding="utf-8"))
        ext[d["name"]] = d
    out = []
    for m in man["models"]:
        if m["extend_mm"] > man["max_extend_mm"]:
            continue
        rec = {"name": m["name"], "side": m["side"], "path": m["path"],
               "cca_mm": m["cca_mm"], "eca_mm": m["eca_mm"],
               "bif_deg": m["bif_deg"], "stenosis_pct": m.get("stenosis_pct"),
               "extended": m["name"] in ext}
        if rec["extended"]:
            e = ext[m["name"]]
            rec["ica_mm"] = e["ica_mm_after"]
            r = e.get("radii")
            rec["tip_d"] = 2 * float(r[-1]) if r else 2 * (m.get("ica_tip_radius") or 0)
            rec["ext_json"] = os.path.join(ext_dir, m["name"] + "_ica_extended.json")
        else:
            rec["ica_mm"] = m["ica_mm"]
            rec["tip_d"] = 2 * (m.get("ica_tip_radius") or 0)
            rec["ext_json"] = None
        if rec["tip_d"] >= MIN_TIP_D:
            out.append(rec)
    return out


def load_siphons(dirs):
    from graft_siphon import prep_siphon, arclength
    out = []
    for d, tag in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*_ica.json"))):
            j = json.load(open(f, encoding="utf-8"))
            p, r = prep_siphon(np.array(j["points"], float), np.array(j["radii"], float))
            L = float(arclength(p)[-1])
            nm = os.path.basename(f).replace("_ica.json", "") + tag
            if L < 60 or nm in DEFECTIVE_SIPHONS:
                continue
            out.append({"name": nm,
                        "src": f, "mirror": tag == "_L", "len_mm": L,
                        "prox_d": 2 * float(r[0])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="carotid_data/lower_manifest.json")
    ap.add_argument("--extended", default="carotid_data/extended")
    ap.add_argument("--right", default="topbrain_data/centerlines")
    ap.add_argument("--left", default="topbrain_data/centerlines_left")
    ap.add_argument("--out", default="carotid_data/pairing.json")
    ap.add_argument("--exclude", default=None,
                    help="json list of [lower, siphon] pairs to forbid")
    a = ap.parse_args()

    low = load_lowers(a.manifest, a.extended)
    sip = load_siphons([(a.right, ""), (a.left, "_L")])
    banned = set()
    if a.exclude and os.path.exists(a.exclude):
        for l, s in json.load(open(a.exclude, encoding="utf-8")):
            banned.add((l, s))
    print("lowers: %d (tip >= %.1f mm, %d extended)   siphons: %d (4 defective dropped)   banned: %d"
          % (len(low), MIN_TIP_D, sum(1 for l in low if l["extended"]),
             len(sip), len(banned)))
    ld = np.array([l["tip_d"] for l in low])
    sd = np.array([s["prox_d"] for s in sip])
    print("  lower ICA tip calibre  : %.2f - %.2f mm (median %.2f)"
          % (ld.min(), ld.max(), np.median(ld)))
    print("  siphon proximal calibre: %.2f - %.2f mm (median %.2f)"
          % (sd.min(), sd.max(), np.median(sd)))

    from scipy.optimize import linear_sum_assignment
    BIG = 1e6
    used = set()
    cap_low = {i: LOWER_CAP for i in range(len(low))}
    cap_sip = {(j, side): SIPHON_CAP_PER_SIDE
               for j in range(len(sip)) for side in ("left", "right")}
    pairs = []
    for rnd in range(LOWER_CAP):
        li = [i for i in range(len(low)) if cap_low[i] > 0]
        sj = [key for key in cap_sip if cap_sip[key] > 0]
        if not li or not sj:
            break
        C = np.full((len(li), len(sj)), BIG)
        for x, i in enumerate(li):
            for y, key in enumerate(sj):
                j, side = key
                if side != low[i]["side"] or (i, j) in used:
                    continue
                if (low[i]["name"], sip[j]["name"]) in banned:
                    continue
                m = abs(low[i]["tip_d"] - sip[j]["prox_d"])
                if m <= MAX_MISMATCH_MM:
                    C[x, y] = m
        r, c = linear_sum_assignment(C)
        got = 0
        for x, y in zip(r, c):
            if C[x, y] >= BIG:
                continue
            i = li[x]
            j, side = sj[y]
            used.add((i, j))
            cap_low[i] -= 1
            cap_sip[(j, side)] -= 1
            got += 1
            pairs.append({"lower": low[i]["name"], "siphon": sip[j]["name"],
                          "mismatch_mm": float(C[x, y]),
                          "lower_side": low[i]["side"],
                          "lower_tip_d": low[i]["tip_d"],
                          "siphon_prox_d": sip[j]["prox_d"],
                          "lower_extended": low[i]["extended"]})
        print("  round %d: +%d pairs (running total %d)" % (rnd + 1, got, len(pairs)))
        if got == 0:
            break

    mm = np.array([p["mismatch_mm"] for p in pairs])
    lc = Counter(p["lower"] for p in pairs)
    sc = Counter(p["siphon"] for p in pairs)
    uniq = len({(p["lower"], p["siphon"]) for p in pairs})
    print("")
    print("paired: %d  (%d unique combinations)" % (len(pairs), uniq))
    print("  seam mismatch: %.2f - %.2f mm, median %.2f"
          % (mm.min(), mm.max(), np.median(mm)))
    print("  lower usage : max %d (cap %d), %d distinct"
          % (max(lc.values()), LOWER_CAP, len(lc)))
    print("  siphon usage: max %d (cap %d), %d distinct"
          % (max(sc.values()), 2 * SIPHON_CAP_PER_SIDE, len(sc)))

    plan = {"lowers": {l["name"]: l for l in low},
            "siphons": {s["name"]: s for s in sip},
            "pairs": pairs}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
