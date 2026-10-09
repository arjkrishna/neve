# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\heading\zen_frames.py (= /sp/v6/heading/zen_frames.py in the build containers)
"""Per-patient frame analysis of the Zenodo forks from heading_evidence.json (C rows)."""
import json, numpy as np, collections
H = json.load(open("/sp/v6/heading/heading_evidence.json"))
rows = H["C"]["rows"]
pats = collections.defaultdict(dict)
for n, r in rows.items():
    if "error" in r or not r["lm"]["w10_20"]:
        continue
    pats[r["patient"]][r["side"]] = r
print("group  patient        xdir  dx(L-R)   R:dy   R:dlatS  L:dy   L:dlatS   R:ccay  L:ccay")
tab = collections.defaultdict(list)
for p in sorted(pats):
    s = pats[p]
    grp = p.split("_")[1]
    if "left" in s and "right" in s:
        dx = s["left"]["lm"]["w10_20"]["bif"][0] - s["right"]["lm"]["w10_20"]["bif"][0]
        xdir = 1 if dx > 0 else -1          # +1: +x = patient left
    else:
        dx, xdir = float("nan"), 0
    vals = []
    for sd in ("right", "left"):
        if sd in s:
            d = s[sd]["lm"]["w10_20"]["eca_minus_ica"]
            lat_sign = (-1 if sd == "right" else 1) * (xdir if xdir else 1)   # lateral direction in x
            vals += [d[1], lat_sign * d[0], s[sd]["lm"]["w10_20"]["cca_dir"][1]]
            tab[(grp, xdir, sd)].append((d[1], lat_sign * d[0], s[sd]["lm"]["w10_20"]["cca_dir"][1], s[sd]["lm"]["w10_20"]["bif"]))
        else:
            vals += [np.nan] * 3
    print("%-5s  %-13s  %+d  %7.1f  %6.1f %7.1f %6.1f %7.1f   %6.2f %6.2f" % (grp, p, xdir, dx, vals[0], vals[1], vals[3], vals[4], vals[2], vals[5]))
print()
print("summary by (group, xdir, side): n, ECA dy>0, ECA lateral (own-patient x), CCA y-lean>0, mean bif xyz")
for k in sorted(tab):
    v = tab[k]
    dy = np.array([a[0] for a in v]); dl = np.array([a[1] for a in v]); cy = np.array([a[2] for a in v])
    b = np.array([a[3] for a in v]).mean(0).round(0)
    print(k, len(v), "dy>0 %d" % (dy > 0).sum(), "lat>0 %d" % (dl > 0).sum(), "ccay>0 %d" % (cy > 0).sum(), b)
