#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\va\paths_d17.py (= /sp/v6/va/paths_d17.py in the build containers)
"""paths_d17.py (s2_harden_va, 2026-10-04) -- D17: the donor's OWN subclavian path to the END of its segmentation.

D17 (user): no seam-2 bridge; the donor's subclavian continues toward the arm as the (17)/(19) branch to the end of the
donor's segmentation and ends as a cut end (a short tube only when < ~30 mm past the ostium).

Stage 1's paths (va/subclavian_paths.py) kept the Arterial chain only to KEEP_PAST_MM = 45 mm past the ostium. This
re-runs the SAME code (imported unchanged; same chain, same origin, same 0.5 mm resampling => the same arclength
origin, verified per side against the 45 mm record) with no distal cut, then ends the path where the donor data ends:
  * the end of the Arterial xSA chain (largest-radius continuation, as stage 1), and
  * the CT field of view (the path is cut 3 mm before it first leaves the CT box past the ostium; VMR 0248: z range).
Writes /s2/paths_d17/<donor>_<side>.json (stage-1 schema + "d17": {end_how, past_ostium_mm, prefix_dev_mm, ...}) and
/s2/paths_d17/_summary.json. The wall agent segments the distal extension along these paths (wall/requests.json).
Runs in neve-build-stagea. Read-only on /repo; writes /s2 only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/va")
import subclavian_paths as SPM  # noqa: E402

OUT = os.environ.get("D17_OUT", "/s2/paths_d17")
OLD = "/repo/carotid_data/v6_work/va/subclavian/paths"
FOV_MARGIN_MM = 3.0
SPM.KEEP_PAST_MM = 1e9
SPM.OUT = OUT


def fov_end(o):
    """Index of the last path point past the ostium before the path first leaves the CT box (minus the margin)."""
    P = np.asarray(o["path"], float)
    s = SPM.arclen(P)
    so = float(o["s_ostium_mm"])
    fov = o.get("fov") or {}
    if "lo" in fov:
        lo, hi = np.asarray(fov["lo"]), np.asarray(fov["hi"])
        inside = np.all((P >= lo + FOV_MARGIN_MM) & (P <= hi - FOV_MARGIN_MM), axis=1)
    elif "z_mm" in fov:
        z0, z1 = fov["z_mm"]
        inside = (P[:, 2] >= z0 + FOV_MARGIN_MM) & (P[:, 2] <= z1 - FOV_MARGIN_MM)
    else:
        return len(P) - 1, "no FOV record (Arterial chain end)"
    past = np.nonzero(s >= so)[0]
    if len(past) == 0:
        return len(P) - 1, "ostium at the path end"
    out = np.nonzero(~inside[past[0]:])[0]
    if len(out) == 0:
        return len(P) - 1, "Arterial xSA chain end (inside the CT)"
    k = past[0] + int(out[0]) - 1
    return max(k, past[0]), "CT field of view (cut %.0f mm inside the box)" % FOV_MARGIN_MM


def main():
    os.makedirs(OUT, exist_ok=True)
    units = sorted(d for d in os.listdir(SPM.DONORS) if os.path.isfile(os.path.join(SPM.DONORS, d, "record.json")))
    only = os.environ.get("ONLY")
    only = set(only.split(",")) if only else None
    summ = {}
    for u in units:
        ur = SPM.jload(os.path.join(SPM.DONORS, u, "record.json"))
        for side in ("R", "L"):
            key = "%s_%s" % (u, side)
            if only and key not in only:
                continue
            if not ((ur.get("sides") or {}).get(side) or {}).get("built"):
                continue
            if os.path.isfile(os.path.join(OUT, key + ".json")) and not only:
                summ[key] = SPM.jload(os.path.join(OUT, key + ".json")).get("d17") or {"status": "done before"}
                continue
            try:
                o = SPM.one(u, side)
            except Exception as e:  # noqa: BLE001
                o = {"donor_id": u, "side": side, "status": "error", "why": "%s: %s" % (type(e).__name__, e)}
            row = {"status": o.get("status")}
            if o.get("status") == "ok":
                P = np.asarray(o["path"], float)
                r = np.asarray(o["radius_path"], float)
                full_past = float(SPM.arclen(P)[-1] - o["s_ostium_mm"])
                k, how = fov_end(o)
                P, r = P[:k + 1], r[:k + 1]
                s = SPM.arclen(P)
                # same chain and arclength origin as stage 1's 45 mm record?
                oldf = os.path.join(OLD, key + ".json")
                dev = None
                if os.path.isfile(oldf):
                    q = np.asarray(SPM.jload(oldf)["path"], float)
                    n = min(len(q), len(P))
                    dev = float(np.linalg.norm(q[:n] - P[:n], axis=1).max())
                o.update({"path": P.round(3), "radius_path": r.round(3), "path_len_mm": float(s[-1]),
                          "past_ostium_mm": float(s[-1] - o["s_ostium_mm"])})
                o["d17"] = {"end_how": how, "past_ostium_mm": float(s[-1] - o["s_ostium_mm"]),
                            "arterial_chain_past_ostium_mm": full_past, "prefix_dev_vs_stage1_mm": dev,
                            "fov_margin_mm": FOV_MARGIN_MM, "tool": "paths_d17.py (s2_harden_va)"}
                row.update(o["d17"])
            else:
                row["why"] = o.get("why")
            SPM.jdump(o, os.path.join(OUT, key + ".json"))
            summ[key] = row
            print("%-22s %s" % (key, json.dumps(row, default=float)), flush=True)
    if not only:
        SPM.jdump(summ, os.path.join(OUT, "_summary.json"))


if __name__ == "__main__":
    main()
