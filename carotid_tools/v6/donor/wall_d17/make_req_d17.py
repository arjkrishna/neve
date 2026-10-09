#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\make_req_d17.py (= /sp/v6/wall_d17/make_req_d17.py in the build containers)
"""make_req_d17.py (s2_harden_wall_bake, D17): the subclavian request entries for the D17 re-run = stage 1's entries
(wall/subclavian_requests.json, written by v6_va's make_request.py: same s_from, same ostium, same metadata) with the
centreline / radius replaced by the D17 path (paths_d17.py: v6_va's path with the 45 mm cap lifted -> to the end of
the donor's segmentation, FOV-clipped) and s_to = the path's end. Checks that the D17 path equals stage 1's
centreline over stage 1's whole length (so the proximal part, the ostium and s_from are unchanged).
Writes /v6w/d17/req/<donor_id>.json (list, one entry per side, as sub_batch2 expects) + /v6w/d17/req/_summary.json.
    python3 make_req_d17.py        (neve-build-stagea)"""
import json
import os

import numpy as np

REQ1 = "/sp/v6/wall/subclavian_requests.json"
PATHS = "/v6w/d17/paths"
OUT = "/v6w/d17/req"


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def main():
    os.makedirs(OUT, exist_ok=True)
    E = json.load(open(REQ1))["entries"]
    per, summ = {}, {}
    for e in E:
        k = "%s_%s" % (e["donor_id"], e["side"])
        pj = json.load(open(os.path.join(PATHS, k + ".json")))
        P = np.asarray(pj["path"], float)
        r = np.asarray(pj["radius_path"], float)
        C1 = np.asarray(e["centerline_ras_mm"], float)
        n = min(len(C1), len(P))
        dmax = float(np.abs(P[:n] - C1[:n]).max())
        n_need = int(np.searchsorted(arclen(C1), float(e["s_to_mm"])))   # stage 1 stretch end
        if n < n_need or dmax > 2e-3:
            raise SystemExit("%s: D17 path does not continue stage 1's centreline (n %d of %d, max diff %.4f)" % (k, n, len(C1), dmax))
        s = arclen(P)
        e2 = dict(e)
        e2.update(centerline_ras_mm=P.round(3).tolist(), radius_mm=r.round(3).tolist(), s_to_mm=round(float(s[-1]), 1),
                  out_dir_suggested="/repo/carotid_data/v6_work/wall/d17/local/%s/" % e["donor_id"],
                  d17={"stage1_s_to_mm": e["s_to_mm"], "stage1_points": len(C1), "equals_stage1_maxdiff_mm": dmax,
                       "path_end_how": pj["d17"]["end_how"], "past_ostium_mm": round(float(s[-1] - e["ostium_s_mm"]), 2),
                       "tube_extension_needed": pj["d17"]["tube_extension_needed"],
                       "note": "s_to = the end of the donor's own subclavian segmentation (D17: the (17)/(19) branch ends there as a cut end)"})
        per.setdefault(e["donor_id"], []).append(e2)
        summ[k] = {"s_from": e2["s_from_mm"], "s_to_stage1": e["s_to_mm"], "s_to_d17": e2["s_to_mm"],
                   "ostium": e2["ostium_s_mm"], "past_ostium_d17": e2["d17"]["past_ostium_mm"], "end_how": pj["d17"]["end_how"],
                   "tube": pj["d17"]["tube_extension_needed"]}
        print(k, json.dumps(summ[k]), flush=True)
    for u, rows in per.items():
        json.dump(rows, open(os.path.join(OUT, u + ".json"), "w"), indent=1)
    json.dump(summ, open(os.path.join(OUT, "_summary.json"), "w"), indent=1)
    print(len(summ), "sides,", len(per), "units")


if __name__ == "__main__":
    main()
