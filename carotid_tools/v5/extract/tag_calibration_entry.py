#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_calibration_entry.py (= /sp/v5_va/tag_calibration_entry.py in the build containers)
"""tag_calibration_entry.py -- offline search for a corrected foramen-entry rule, on the per-level distance profiles
stored by tag_calibration_paths.py (profile_1mm: path resampled at 1 mm as ts_qc, distance d_k to each TS vertebra
mask C1..C7, 5 mm-ray enclosure).

    python3 tag_calibration_entry.py      -> /sp/v5_va/results/tag_calibration_entry.json     (eve-training-fixed)

Rule family R(d_thr, run_min, gap_min): for each level k, runs of path points with z inside the level's OWN z range
and d_k <= d_thr, kept if >= run_min mm. Entry = the lowest level (C7 .. C3) with a kept run, EXCEPT that a C7 run
counts only if at least gap_min mm of path lie between its end and the start of the first kept C6 run (a VA that
enters at C7 leaves the C7 foramen and crosses the C7/C6 intertransverse space before the C6 foramen; a C6 entry
grazes the C7 label at the lower rim of the C6 foramen, contiguous with the C6 run). gap_min = 0 is the plain run rule.
Truth: C6 on all four GT VAs (VMR 0248 R/L, TaG-Net A R/L), so every GT path and every trace of them at every quality
(their cervical course below C2 is correct at every quality) must read C6. TS sides are compared with the verifier's
independent axial threading test (verify_arterial_yield_out/qc.json: first threaded level, threaded_C7).
"""
import json

import numpy as np

RES = "/sp/v5_va/results"


def runs(mask):
    out, st = [], None
    for i, v in enumerate(mask):
        if v and st is None:
            st = i
        if (not v or i == len(mask) - 1) and st is not None:
            out.append((st, i if v else i - 1))
            st = None
    return out


def entry(prof, d_thr, run_min, gap_min, zr):
    s = np.asarray(prof["s"], float)
    z = np.asarray(prof["z"], float)
    kept = {}
    for k in range(7, 2, -1):
        L = "C%d" % k
        if L not in prof["d"] or L not in zr:
            continue
        d = np.asarray([np.inf if x is None else x for x in prof["d"][L]], float)
        own = (z >= zr[L][0]) & (z <= zr[L][1])
        kept[L] = [(s[a], s[b]) for a, b in runs(own & (d <= d_thr)) if s[b] - s[a] + 1.0 >= run_min]
    for k in range(7, 2, -1):
        L = "C%d" % k
        if not kept.get(L):
            continue
        if L == "C7" and gap_min > 0:
            c7_end = kept["C7"][0][1]
            c6 = [r for r in kept.get("C6", []) if r[0] > kept["C7"][0][0]]
            if c6 and c6[0][0] - c7_end - 1.0 < gap_min:
                continue
        return L, kept[L][0][0]
    return None, None


def main():
    rows = []
    for g in ("VMR", "A"):
        J = json.load(open(RES + "/tag_calibration_paths_%s.json" % g))
        for pid, row in J["paths"].items():
            for ms in ("total_v3", "total"):
                v = row.get(ms)
                if not isinstance(v, dict) or "profile_1mm" not in v:
                    continue
                zr = {L: p["z_range_mm"] for L, p in v["entry"]["per_level"].items()}
                rows.append({"set": g, "id": pid, "ms": ms, "kind": row["kind"], "truth": "C6", "prof": v["profile_1mm"], "zr": zr})
    import os
    T = json.load(open(RES + "/tag_calibration_paths_TS.json")) if os.path.isfile(RES + "/tag_calibration_paths_TS.json") else {"paths": {}}
    VQ = json.load(open("/sp/v5_va/verify_arterial_yield_out/qc.json"))
    for key, r in T["paths"].items():
        if not r.get("trace_ok"):
            continue
        sid, sd = key.split()
        q = VQ.get("ts_" + sid, {}).get("sides", {}).get(sd, {})
        thr = q.get("threaded_C6_C1") or []
        vf = "C7" if q.get("threaded_C7") else (thr[0] if thr else None)
        v = r["trace"]
        zr = {L: p["z_range_mm"] for L, p in v["entry"]["per_level"].items()}
        rows.append({"set": "TS", "id": key, "ms": "dataset", "kind": "trace", "truth": vf, "prof": v["profile_1mm"], "zr": zr,
                     "stored": (r.get("stored_ts_qc") or {}).get("foramen_entry_level")})
    grid = []
    for d_thr in (0.0, 0.5, 1.0):
        for run_min in (3.0, 5.0):
            for gap_min in (0.0, 2.0, 3.0):
                res = {"d_thr": d_thr, "run_min": run_min, "gap_min": gap_min}
                for sub in ("VMR", "A"):
                    for ms in ("total_v3", "total"):
                        rr_ = [x for x in rows if x["set"] == sub and x["ms"] == ms]
                        calls = [entry(x["prof"], d_thr, run_min, gap_min, x["zr"])[0] for x in rr_]
                        res["%s_%s_C6" % (sub, ms)] = "%d/%d" % (sum(c == "C6" for c in calls), len(calls))
                        res["%s_%s_wrong" % (sub, ms)] = sorted(set("%s=%s" % (x["id"], c) for x, c in zip(rr_, calls) if c != "C6"))[:6]
                ts = [x for x in rows if x["set"] == "TS"]
                calls = [entry(x["prof"], d_thr, run_min, gap_min, x["zr"])[0] for x in ts]
                res["TS_agree_verifier"] = "%d/%d" % (sum(c == x["truth"] for x, c in zip(ts, calls)), len(ts))
                res["TS_C7_calls"] = sum(c == "C7" for c in calls)
                hist = {}
                for c in calls:
                    hist[str(c)] = hist.get(str(c), 0) + 1
                res["TS_calls"] = hist
                res["TS_disagree"] = ["%s %s vs verifier %s" % (x["id"], c, x["truth"]) for x, c in zip(ts, calls) if c != x["truth"]]
                grid.append(res)
                print(json.dumps({k: v for k, v in res.items() if not k.endswith("_wrong") and k != "TS_disagree"}), flush=True)
    ts = [x for x in rows if x["set"] == "TS"]
    vh, sh = {}, {}
    for x in ts:
        vh[str(x["truth"])] = vh.get(str(x["truth"]), 0) + 1
        sh[str(x["stored"])] = sh.get(str(x["stored"]), 0) + 1
    out = {"about": __doc__.split("\n\n")[0], "grid": grid, "TS_verifier_first_threaded_hist": vh, "TS_stored_ts_qc_entry_hist": sh,
           "TS_stored_agree_verifier": "%d/%d" % (sum(x["stored"] == x["truth"] for x in ts), len(ts)),
           "n_rows": {g: sum(1 for x in rows if x["set"] == g) for g in ("VMR", "A", "TS")}}
    json.dump(out, open(RES + "/tag_calibration_entry.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}))


if __name__ == "__main__":
    main()
