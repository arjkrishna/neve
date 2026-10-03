#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_select.py (= /sp/v5_va/scaleup_select.py in the build containers)
"""scaleup_select.py <source> <case> -- PIPELINE v2 trace-set SELECTION per side (scaleup_fixes, 2026-10-02), both sources.

Writes qc/<SCALEUP_SELECT> (default select.json; the pilot re-runs use select_v2.json so the pilot's file is kept), read by
scaleup_ctseg_common_<src>.selected(), scaleup_v2_gross.py, scaleup_v2_fig.py, scaleup_v2_collect_<src>.py.
Candidates per side, each graded by the UNCHANGED pilot QC gate (pilot_proc_<src>_qc2.py):
  tsba     qc/qc.json          art_extract + the basilar fix (the protocol trace)
  ts       qc/qc_ts.json       the unchanged art_extract (ia172 gains this in v2: PILOT_TRACE_DIR=ts)
  tsba_v1r qc/qc_tsba_v1r.json / ts_v1r qc/qc_ts_v1r.json   the same tools on the V1-rescued mask (scaleup_v1rescue.py)
Level 1 = the Song pilot's rule, unchanged: among tsba / ts, rank = (passes_as_v5_donor, whole_va_pass, is tsba); best wins,
  tsba wins ties (on the 172-series pilot ts never beat tsba below the splice, so v2 = the pilot there).
Level 2 = the V1 rescue: when qc/v1rescue.json rescued this side from base trace set e and qc_<e>_v1r.json exists, the
  rescued trace REPLACES the level-1 choice if it passes as a v5 donor AND (the level-1 choice fails as a donor, OR it passes
  only with V1 bridge(s) <= 3 mm -- the provisional limit -- while the rescued trace has no V1 bridge left). Otherwise the
  level-1 choice stays and the rescue is recorded as "not used" with the reason.
Read-only apart from the selection file. Any python3 (json only).
"""
import json
import os
import sys

SRCN, CASE = sys.argv[1], sys.argv[2]
DER = "/repo/carotid_data/va_sources/derived/thin_slice/%s/%s" % (SRCN, CASE)
SEL = os.environ.get("SCALEUP_SELECT", "select.json")


def L(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def qf(ext):
    return "qc.json" if ext == "tsba" else "qc_%s.json" % ext


def v1b(r):
    return [b for b in (r.get("bridges") or []) if b.get("segment") == "V1"]


out = {"case": CASE, "source": SRCN, "pipeline": "v2 (scaleup_select.py)",
       "rule": "level 1: rank = (passes_as_v5_donor, whole_va_pass, is_tsba) over tsba / ts, tsba wins ties (the Song pilot "
               "rule); level 2: the V1-rescued trace <e>_v1r replaces it if it passes as a donor and (level 1 fails as a donor "
               "or passes only with V1 bridge(s) <= 3 mm while the rescued trace has none)", "sides": {}}
resc = L(os.path.join(DER, "qc/v1rescue.json")) or {}
for side in ("R", "L"):
    cands = []
    for e in ("tsba", "ts"):
        q = L(os.path.join(DER, "qc", qf(e)))
        if q and side in (q.get("sides") or {}):
            cands.append((e, q["sides"][side]))
    if not cands:
        raise SystemExit("no qc file for %s" % CASE)
    best = max(cands, key=lambda t: (bool(t[1].get("passes_as_v5_donor")), bool(t[1].get("whole_va_pass")), t[0] == "tsba"))
    rec = {"ext": best[0], "level1_ext": best[0], "qc_file": qf(best[0]), "route_rescue": best[0] != "tsba",
           "v1_rescue_used": False,
           "candidates": {e: {"donor": bool(r.get("passes_as_v5_donor")), "whole_va": bool(r.get("whole_va_pass")),
                              "status": r.get("status"), "why": r.get("gate_fail_reasons"),
                              "v1_bridges": [[b.get("s_mm"), b.get("length_mm"), b.get("HU_median_native")] for b in v1b(r)]}
                          for e, r in cands}}
    rs = (resc.get("sides") or {}).get(side) or {}
    if rs.get("rescued"):
        e0 = rs.get("base_trace_set") or "tsba"
        qv = L(os.path.join(DER, "qc", qf(e0 + "_v1r")))
        rv = (qv or {}).get("sides", {}).get(side) if qv else None
        if rv is None:
            rec["v1_rescue"] = "rescued mask exists but qc_%s_v1r.json is missing (re-trace / QC not run)" % e0
        else:
            rec["candidates"][e0 + "_v1r"] = {"donor": bool(rv.get("passes_as_v5_donor")), "whole_va": bool(rv.get("whole_va_pass")),
                                              "status": rv.get("status"), "why": rv.get("gate_fail_reasons"),
                                              "v1_bridges": [[b.get("s_mm"), b.get("length_mm"), b.get("HU_median_native")] for b in v1b(rv)]}
            b1 = best[1]
            prov = bool(b1.get("passes_as_v5_donor")) and bool(v1b(b1))
            use = bool(rv.get("passes_as_v5_donor")) and (not b1.get("passes_as_v5_donor") or (prov and not v1b(rv)))
            rec["v1_rescue"] = {"base_trace_set": e0, "rescued_trace_set": e0 + "_v1r", "rescued_status": rv.get("status"),
                                "rescued_fails": rv.get("gate_fail_reasons"), "level1_donor": bool(b1.get("passes_as_v5_donor")),
                                "level1_provisional_v1": prov, "used": use,
                                "why": ("level-1 trace fails as a donor; the rescued trace passes" if use and not b1.get("passes_as_v5_donor")
                                        else "level-1 trace passes only with V1 bridge(s) <= 3 mm; the rescued trace has none" if use
                                        else "rescued trace fails as a donor" if not rv.get("passes_as_v5_donor")
                                        else "level-1 trace already passes and the rescue does not remove its V1 bridges")}
            if use:
                rec.update(ext=e0 + "_v1r", qc_file=qf(e0 + "_v1r"), v1_rescue_used=True, route_rescue=e0 != "tsba")
    elif rs:
        rec["v1_rescue"] = "not applied: %s" % (rs.get("why") if not rs.get("applies") else "the rescue failed: %s" % [
            b.get("verdict") for b in rs.get("bridges_rescue", [])])
    out["sides"][side] = rec
    print("select %s %s: %s (level 1 %s) %s" % (CASE, side, rec["ext"], rec["level1_ext"],
          json.dumps(rec.get("v1_rescue")) if rec.get("v1_rescue") else ""), flush=True)
out["any_donor"] = any(out["sides"][sd]["candidates"].get(out["sides"][sd]["ext"], {}).get("donor") for sd in out["sides"])
json.dump(out, open(os.path.join(DER, "qc", SEL), "w"), indent=1)
