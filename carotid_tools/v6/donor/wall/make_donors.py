#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\make_donors.py (= /sp/v6/wall/make_donors.py in the build containers)
"""v6_wall: the per-donor ct_seg configuration that reproduces each v5 wall (from its seg_qc.json) -> wall/donors.json.
pilot walls (no seg_qc "pipeline"): pinch fix off, select.json (absent for the ia172 pilot cases -> tsba = the pilot).
pipeline v2 walls: pinch fix on, the recorded select file. PILOT_SEG_TAG = v5's tag (the pinch-skip file name)."""
import json, os
el = json.load(open("/sp/v5_va/results/v5_eligible_final.json"))
out = {}
for c in ("primary", "reserve"):
    for p in el.get(c, []):
        if p["source"] == "vmr0248":
            out["vmr0248"] = {"source": "vmr0248", "case": "VMR0248", "class": c, "note": "no NIfTI: vmr_ct_seg3 on p12full.vti"}
            continue
        sd = os.path.dirname(p["sides"]["R"]["wall_files"]["surface_vtp"])
        q = json.load(open(os.path.join(sd, "seg_qc.json")))
        pipe = q.get("pipeline")
        did = p["case"] if p["case"].startswith("ia172") else "song_" + p["case"]
        out[did] = {"source": p["source"], "case": p["case"], "class": c, "v5_seg_dir": sd, "v5_tag": os.path.basename(sd),
                    "env": {"PILOT_SOURCE": p["source"], "PILOT_CASE": p["case"], "PILOT_SEG_TAG": os.path.basename(sd),
                            "VMR_OPEN_BRIGHT": "1", "PILOT_CT": "shipped", "PILOT_KEEP_CONNECTED": "1",
                            "SCALEUP_PINCH_FIX": "1" if pipe else "0",
                            "SCALEUP_SELECT": (pipe or {}).get("select_file") or "select.json"},
                    "v5_pipeline": "v2" if pipe else "pilot", "v5_seconds": q.get("seconds"),
                    "sides": {s: {"trace_set": p["sides"][s].get("trace_set"), "wall_files": p["sides"][s].get("wall_files")}
                              for s in ("R", "L")}}
json.dump(out, open("/sp/v6/wall/donors.json", "w"), indent=1)
print(len(out), "units;", sum(1 for v in out.values() if v.get("v5_pipeline") == "pilot"), "pilot walls")
