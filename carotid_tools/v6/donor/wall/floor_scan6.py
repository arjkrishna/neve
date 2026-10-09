#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\floor_scan6.py (= /sp/v6/wall/floor_scan6.py in the build containers)
# floor_scan6.py (v6_wall) = /sp/v5_va/tbo/va_narrow/floor_scan.py with the donor root (V6_DONORS) and the output (V6_SCAN_OUT)
# as env; the v6 records keep each side's seg dir in surface.src, so the station / seg_qc lookups follow the new wall.
# The used donor ranges are v5's (anatomies_v5 provenance): the same VAs, re-scanned on the new lines.
"""va_narrow: how common is the "floored MISR at bone contact" pattern over all v5 donor VAs? (eve-training-fixed)

Per donor side (donors_v5/<unit>/<side>.json): stretches of the donor centreline where the VMTK MISR < 1.0 mm (the
graft floors those radii to VA_MIN_R = 1.0, so the union's tube floor, not the donor wall, sets the lumen there); per
stretch: length, min MISR, the CT calibre there, and at the nearest ct_seg4 stations: open-bright flagged / fix-8
not-opened, mean count of bright-neighbour rays (kind 1 dip-before-bright + kind 3 bright-neighbour-upper, of 64).
Per v5 VA (provenance xform.vaC_* + v5_va.<side>.splice_on_donor_mm): which of those stretches lie in the used donor
range [0, splice], and the VA's OBJ lumen minimum (checks/va_lumen.json, final for all but the 13 moved anatomies).
    python3 floor_scan.py -> floor_scan.json
"""
import glob
import json
import os

import numpy as np

R5 = os.environ.get("V6_DONORS", "/repo/carotid_data/va_sources/v5_reg/donors_v5")
A5 = "/repo/carotid_data/anatomies_v5"
OUT = os.environ.get("V6_SCAN_OUT", "/v6w/scan") + "/floor_scan.json"
FLOOR = 1.0


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def in_runs(s, rr):
    return any(a <= s <= b for a, b in rr)


def side_scan(unit, side):
    f = os.path.join(R5, unit, "%s.json" % side)
    if not os.path.exists(f):
        return None
    rec = json.load(open(f))
    c = rec["centerline"]
    P = np.asarray(c["points"], float)
    m = np.asarray(c["radii_misr"], float)
    cal = np.asarray(c["radii_calibre"], float) if c.get("radii_calibre") else None
    s = arclen(P)
    src = rec["surface"]["src"]
    seg = os.path.dirname(src)
    vessel = "rva" if side == "R" else "lva"
    St, ob = None, {}
    try:
        St = json.load(open(os.path.join(seg, "stations_rays.json")))[vessel]
        ob = json.load(open(os.path.join(seg, "seg_qc.json")))["open_bright"].get(vessel, {})
    except Exception:   # noqa: BLE001  (a donor without ct_seg stations, e.g. a tube donor)
        pass
    flagged = ob.get("flagged_s_ranges_mm", []) if ob else []
    fix8 = (ob.get("keep_connected_fix8") or {}).get("runs_not_opened_s_mm", []) if ob else []
    stretches = []
    for a, b in runs(m < FLOOR):
        row = {"s_from": float(s[a]), "s_to": float(s[b]), "len_mm": float(s[b] - s[a] + (s[1] - s[0])),
               "misr_min": float(m[a:b + 1].min()), "calibre_median": float(np.median(cal[a:b + 1])) if cal is not None else None}
        if St is not None:
            sP, sS = np.asarray(St["P"], float), np.asarray(St["s"], float)
            ks = [int(np.argmin(np.linalg.norm(sP - P[i], axis=1))) for i in range(a, b + 1)]
            kinds = [np.asarray(St["kind"][k]) for k in ks]
            row.update({"bright_rays_mean": float(np.mean([((kk == 1) | (kk == 3)).sum() for kk in kinds])),
                        "bright_rays_max": int(max(((kk == 1) | (kk == 3)).sum() for kk in kinds)),
                        "open_bright_frac": float(np.mean([in_runs(float(sS[k]), flagged) for k in ks])),
                        "fix8_frac": float(np.mean([in_runs(float(sS[k]), fix8) for k in ks])),
                        "st_req_median": float(np.median([np.sqrt(np.mean(np.square(St["r_e"][k]))) for k in ks]))})
        stretches.append(row)
    return {"unit": unit, "side": side, "len_mm": float(s[-1]), "misr_median": float(np.median(m)),
            "floored_mm_total": float(sum(r["len_mm"] for r in stretches)), "stretches": stretches,
            "landmarks": {k: rec.get("landmarks", {}).get(k) for k in ("C6", "C2", "C1", "dura")}}


def main():
    donors = {}
    for d in sorted(glob.glob(os.path.join(R5, "*"))):
        u = os.path.basename(d)
        for sd in ("R", "L"):
            r = side_scan(u, sd)
            if r:
                donors["%s:%s" % (u, sd)] = r
    lum = {r["name"]: r for r in json.load(open("/sp/v5_va/checks/va_lumen.json"))}
    vas = []
    for root in sorted(glob.glob(os.path.join(A5, "case_*"))):
        name = os.path.basename(root)
        prov = json.load(open(os.path.join(root, "provenance.json")))
        for key, side, tag in (("vaC_R", "right", "RVA"), ("vaC_L", "left", "LVA")):
            xf = prov["xform"].get(key)
            if not xf:
                continue
            vv = prov["v5_va"][side]
            dk = "%s:%s" % (xf["donor_id"], xf["donor_side"])
            sp = float(vv.get("splice_on_donor_mm") or 1e9)
            adv = float(vv.get("ostium_advance_mm") or 0.0)
            used = [st for st in donors.get(dk, {}).get("stretches", []) if st["s_to"] >= adv and st["s_from"] <= sp]
            L = lum.get(name, {}).get("v5_" + tag, {}).get("whole", {})
            vas.append({"name": name, "va": tag, "donor": dk, "splice_on_donor_mm": sp,
                        "floored_stretches_used": len(used), "floored_mm_used": round(sum(u["len_mm"] for u in used), 1),
                        "floored_at_bone_contact": sum(1 for u in used if u.get("bright_rays_mean", 0) >= 6
                                                       or u.get("open_bright_frac", 0) > 0.5),
                        "obj_lumen_min_check_stage": L.get("lumen_min_mm"), "obj_min_at_mm": L.get("lumen_min_at_mm")})
    n_with = sum(1 for v in vas if v["floored_stretches_used"])
    n_bone = sum(1 for v in vas if v["floored_at_bone_contact"])
    summ = {"donor_sides": len(donors),
            "donor_sides_with_floored_stretch": sum(1 for d in donors.values() if d["stretches"]),
            "vas": len(vas), "vas_using_a_floored_stretch": n_with, "vas_using_a_floored_bone_contact_stretch": n_bone}
    json.dump({"summary": summ, "donors": donors, "vas": vas}, open(OUT, "w"), indent=1)
    print(json.dumps(summ))
    for k, d in donors.items():
        for st in d["stretches"]:
            if st["len_mm"] >= 1.5:
                print("%-24s L%5.0f  s %6.1f-%6.1f (%4.1f mm) misr_min %.2f cal %s bright %s ob %s fix8 %s st_req %s" % (
                    k, d["len_mm"], st["s_from"], st["s_to"], st["len_mm"], st["misr_min"],
                    "%.2f" % st["calibre_median"] if st["calibre_median"] else "-",
                    "%.1f" % st.get("bright_rays_mean", -1), "%.2f" % st.get("open_bright_frac", -1),
                    "%.2f" % st.get("fix8_frac", -1), "%.2f" % st.get("st_req_median", -1)))


if __name__ == "__main__":
    main()
