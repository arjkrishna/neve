# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\heading\c6check.py (= /sp/v6/heading/c6check.py in the build containers)
import json, numpy as np
H = json.load(open("/sp/v6/heading/heading_evidence.json"))
rows = [r for r in H["D"]["rows"] if r["key"] == "lower"]
near = [r for r in rows if r["u_seam2"] is not None and r["u_seam2"] < 10]
print("RICA seams < 10 mm from midline:", len(near), "contralateral:", sum(r["contralateral"] for r in near),
      "| all right routes contralateral:", sum(r["contralateral"] for r in rows), "/", len(rows))
import collections
print("by lower:", collections.Counter(r["lower"] for r in near).most_common(12))
print("frames of near-midline lowers:", collections.Counter(r["frame"] for r in near))
