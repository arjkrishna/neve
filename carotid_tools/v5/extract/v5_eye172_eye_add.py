#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_eye172_eye_add.py (= /sp/v5_va/v5_eye172_eye_add.py in the build containers)
"""v5_eye172_eye_add.py -- (v5_eye172, 2026-10-02) append the by-eye verdicts of the 7 flagged 172-series pilot sides to
results/scaleup_eye_rejections.json (read-modify-write, same entry format; a backup of the file as it was is written to
results/v5_eye172/scaleup_eye_rejections.before_v5_eye172.json). Idempotent: an entry already present for the same
source/case/side by v5_eye172 is replaced, an entry by another agent is left alone (and reported)."""
import json
import os
import shutil
import time

F = "/sp/v5_va/results/scaleup_eye_rejections.json"
BK = "/sp/v5_va/results/v5_eye172/scaleup_eye_rejections.before_v5_eye172.json"
WHEN = time.strftime("%Y-%m-%d %H:%M")
L = "results/v5_eye172/look_%s_%s.png (CPRs + 18 axial native-CT crops + HU/CCA along the trace)"
NEW = [
    ("ia172_058", "L", "review = rule 6 only (Arterial VA label 0.89 within 2 mm, review-only; no bridges): " + L % ("ia172_058", "L") +
     " shows the trace leaving the left SCA (s 1-4), V1 posterior to the CCA, entry C6 (s 49), inside the C6..C2 transverse "
     "foramina, C1 foramen, V3 loop over the C1 posterior arch (s 172-189), dura (s 224); continuous round lumen in both CPR "
     "planes, HU/CCA 1.03 V1 / 1.08 V2 / 1.04 C2-C1 / 0.98 C1-dura; no carotid or vein contact, no shortcut. Kept."),
    ("ia172_039", "R", "review = bridged (max 11.1 mm): the only bridge is at the basilar confluence (raw s 240.7, z -508.3), "
     "above the splice, outside the donor stretch. " + L % ("ia172_039", "R") + ": trace from the right SCA (s 1-4), V1 behind "
     "the CCA, entry C6 (s 44), foramina C6..C2, C1 foramen (s 146), V3 loop over the C1 arch (s 154-170), dura (s 201); "
     "continuous lumen, HU/CCA 1.05 V1 / 1.04 V2 / 1.02 C2-C1 / 0.82 C1-dura; no shortcut. Kept."),
    ("ia172_039", "L", "review = rule 6 (0.87 within 2 mm) + bridged (11.1 mm at the basilar confluence, raw s 241.5, above the "
     "splice) + robust V3 excursion 13.1 mm (< the study's 15; recalibrated pass): " + L % ("ia172_039", "L") + " shows the "
     "trace from the left SCA (s 1-4; a 1-2 mm HU dip to 0.46 x CCA at s 5 is the origin funnel, the lumen is continuous in "
     "both CPRs), V1 behind the CCA, entry C6 (s 41), foramina C6..C2, C1 foramen ring (s 148), V3 loop around the C1 "
     "posterior arch (s 157-173, clearly a loop on the axial crops), dura (s 206); HU/CCA 1.06 V1 / 1.07 V2 / 0.98 C2-C1; no "
     "carotid or vein shortcut. Kept."),
    ("ia172_015", "R", "review = rule 6 (Arterial VA label 0.00 within 2 mm, review-only) + Erec entry C7 (pilot: check by "
     "eye): " + L % ("ia172_015", "R") + " shows the trace from the right SCA (s 1), V1 posterior to the CCA, entering the C7 "
     "transverse foramen (s ~30; a known VA entry variant) then the C6..C2 foramina, C1 foramen ring (s 144), V3 loop over the "
     "C1 arch (s 153-170), dura (s 203); continuous lumen, HU/CCA 0.92 V1 / 0.89 V2 / 0.92 C2-C1; Arterial labelled another "
     "vessel, the trace is the VA; no shortcut. Kept (flag: C7 entry)."),
    ("ia172_190", "R", "review = rule 6 (0.83 within 2 mm) + bridged (9.2 mm at the basilar confluence, raw s 234.9, z -580.6, "
     "above the splice): " + L % ("ia172_190", "R") + ": trace from the right SCA (s 1), V1 behind the CCA, entry C6 (s 43), "
     "foramina C6..C2, C1 foramen (s 138), V3 loop over the C1 arch (s 157-177), dura (s 197); continuous lumen, HU/CCA "
     "0.97-0.99 in every segment; no shortcut. Kept."),
    ("ia172_190", "L", "review = bridged (9.2 mm at the basilar confluence, raw s 231.4, above the splice; no V1-V3 bridge): " +
     L % ("ia172_190", "L") + ": trace starts inside the TS left-SCA label (s 0-2 in the CPR), V1 behind the CCA, entry C6 "
     "(s 44), foramina C6..C2, C1 foramen (s 137), V3 loop over the C1 arch (s 160), dura (s 194); HU/CCA 0.93 V1 / 1.02 V2 / "
     "0.99 C2-C1; no shortcut. Kept."),
    ("ia172_141", "R", "review = rule 6 only (0.90 within 2 mm, review-only; no bridges): " + L % ("ia172_141", "R") + ": trace "
     "leaves the medial end of the right SCA (s 1), V1 behind the CCA, entry C6 (s 43), foramina C6..C2, C1 foramen (s 138), "
     "V3 loop over the C1 arch (s 146-160), dura (s 189); continuous lumen, HU/CCA 1.02 V1 / 0.92 V2 / 0.90 C2-C1; no "
     "shortcut. Kept."),
]
D = json.load(open(F))
if not os.path.isfile(BK):
    shutil.copy2(F, BK)
rows = D["rejections"]
for case, side, why in NEW:
    key = ("ia172", case, side)
    others = [r for r in rows if (r.get("source"), r.get("case"), r.get("side")) == key and r.get("by") != "v5_eye172"]
    if others:
        print("NOTE: existing entry by %s for %s -- left alone, v5_eye172 entry added after it" % (others[0].get("by"), key))
    rows[:] = [r for r in rows if not ((r.get("source"), r.get("case"), r.get("side")) == key and r.get("by") == "v5_eye172")]
    rows.append({"source": "ia172", "case": case, "side": side, "verdict": "kept", "by": "v5_eye172", "when": WHEN, "why": why})
tmp = F + ".tmp_v5_eye172"
json.dump(D, open(tmp, "w"), indent=1)
os.replace(tmp, F)
print("entries now", len(rows), "; v5_eye172 entries", sum(r.get("by") == "v5_eye172" for r in rows))
