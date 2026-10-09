#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\wd_collect.py (= /sp/v6/wall/wd_collect.py in the build containers)
"""wd_collect.py (v6_wall gate 3): aggregate wd_proxy.py's per-donor-side 1 mm bins over every v5 donor VA's used
donor stretch and compare variants.

Per v5 donor VA (anatomies_v5/<name>/provenance.json: xform vaC_<R|L> donor_id / donor_side, v5_va.<side>
ostium_advance_mm, splice_on_donor_mm): donor stretch = donor arclength [advance + 2, splice - 25] (v5's donor_real:
2 mm core trim at the ostium, the 25 mm seam zone out; the bake's tube windows are not known before the bake and are
not removed). Share of wall area > 0.3 mm off the tube over those bins, per variant tag. The same stretch in donor mm is
used for every variant (the v6 VMTK lines differ from v5's by under ~1 mm of arclength).

Calibration: the 'v5' proxy against v5's measured donor_real (vvw/out/<name>.json exact.va.<side>.donor_real
frac_0.3mm, the wall verifier's exact-ownership measure on the undecimated union) -- same VAs, same convention.
    python3 wd_collect.py <tag,tag,...>  -> /v6w/wd/wd_collect.json + printed table
"""
import glob
import json
import os
import sys

import numpy as np

tags = sys.argv[1].split(",")
A5 = "/repo/carotid_data/anatomies_v5"
VVW = "/sp/v5_va/vvw/out"
bins = {}
for t in tags:
    for f in glob.glob("/v6w/wd/%s/*.json" % t):
        d = json.load(open(f))
        bins[(t, d["unit"], d["side"])] = d
vas = []
for root in sorted(glob.glob(os.path.join(A5, "case_*"))):
    name = os.path.basename(root)
    prov = json.load(open(os.path.join(root, "provenance.json")))
    vv = None
    vf = os.path.join(VVW, name + ".json")
    if os.path.exists(vf):
        vv = json.load(open(vf)).get("exact", {}).get("va", {})
    for key, side in (("vaC_R", "right"), ("vaC_L", "left")):
        xf = prov["xform"].get(key)
        if not xf:
            continue
        v5 = prov["v5_va"][side]
        adv = float(v5.get("ostium_advance_mm") or 0.0)
        sp = float(v5["splice_on_donor_mm"])
        a, b = adv + 2.0, sp - 25.0
        row = {"name": name, "va": side, "donor": "%s:%s" % (xf["donor_id"], xf["donor_side"]), "from": a, "to": b}
        meas = ((vv or {}).get(side) or {}).get("donor_real") or {}
        row["v5_measured_donor_real"] = meas.get("frac_0.3mm")
        for t in tags:
            d = bins.get((t, xf["donor_id"], xf["donor_side"]))
            if d is None:
                continue
            ar, a3 = np.asarray(d["bins_1mm"]["area"]), np.asarray(d["bins_1mm"]["a03"])
            fl = np.asarray(d["bins_1mm"]["floored_frac"])
            i0, i1 = int(np.floor(a)), int(np.ceil(b))
            i1 = min(i1, len(ar))
            if i1 <= i0:
                continue
            row[t] = float(a3[i0:i1].sum() / max(ar[i0:i1].sum(), 1e-9))
            row[t + "_floored_mm"] = float(fl[i0:i1].sum())
            lm = d["bins_1mm"].get("lumen_min_union")
            if lm:
                j0, j1 = int(np.floor(adv + 3.0)), min(int(np.ceil(sp)), len(lm))
                vals = [v for v in lm[j0:j1] if v is not None]
                if vals:
                    k = int(np.argmin(vals))
                    row[t + "_lumen_min"] = float(min(vals))
        vas.append(row)


def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    return [round(float(v), 4) for v in np.percentile(x, [10, 50, 90])] + [round(float(x.min()), 4), round(float(x.max()), 4)] if len(x) else None


summ = {"vas": len(vas)}
for t in tags:
    vals = [r.get(t) for r in vas]
    summ[t] = {"p10_med_p90_min_max": q(vals), "n": sum(1 for v in vals if v is not None),
               "floored_mm_per_va_med": float(np.median([r.get(t + "_floored_mm", np.nan) for r in vas if t + "_floored_mm" in r])) if any(t + "_floored_mm" in r for r in vas) else None,
               "vas_using_floor": sum(1 for r in vas if r.get(t + "_floored_mm", 0) > 0),
               "lumen_min_union_p10_med_p90_min_max": q([r.get(t + "_lumen_min") for r in vas]),
               "lumen_min_union_lt_0.9": sum(1 for r in vas if r.get(t + "_lumen_min", 9) < 0.9)}
m = [(r["v5_measured_donor_real"], r.get("v5")) for r in vas if r["v5_measured_donor_real"] is not None and r.get("v5") is not None]
if m:
    x, y = np.array(m).T
    summ["calibration_v5proxy_vs_measured"] = {"n": len(m), "measured_med": float(np.median(x)), "proxy_med": float(np.median(y)),
                                               "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                                               "abs_diff_med": float(np.median(np.abs(x - y))),
                                               "abs_diff_p90": float(np.percentile(np.abs(x - y), 90)),
                                               "proxy_minus_measured_med": float(np.median(y - x))}
# per donor side (median over its VAs)
per = {}
for r in vas:
    per.setdefault(r["donor"], []).append(r)
pd_ = {}
for k, rr in sorted(per.items()):
    pd_[k] = {"vas": len(rr), **{t: round(float(np.median([r[t] for r in rr if t in r])), 4) for t in tags if any(t in r for r in rr)},
              "v5_measured": round(float(np.median([r["v5_measured_donor_real"] for r in rr if r["v5_measured_donor_real"] is not None])), 4)
              if any(r["v5_measured_donor_real"] is not None for r in rr) else None}
json.dump({"summary": summ, "per_donor_side": pd_, "vas": vas}, open("/v6w/wd/wd_collect.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print("%-24s %4s %8s " % ("donor side", "VAs", "v5meas") + " ".join("%9s" % t for t in tags))
for k, v in pd_.items():
    print("%-24s %4d %8s " % (k, v["vas"], "%.3f" % v["v5_measured"] if v["v5_measured"] is not None else "-") +
          " ".join("%9s" % ("%.3f" % v[t] if t in v else "-") for t in tags))
