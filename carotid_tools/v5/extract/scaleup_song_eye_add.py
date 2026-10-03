#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_eye_add.py (= /sp/v5_va/scaleup_song_eye_add.py in the build containers)
"""scaleup_song_eye_add.py ENTRIES.json (scaleup_lane_song, 2026-10-02 relaunch; python:3)
Read-modify-write of the shared results/scaleup_eye_rejections.json (SCALEUP_PIPELINE.md sec 4): each entry of ENTRIES.json
(a list of {source, case, side, verdict?, why, by, when}) replaces an existing entry for the same source/case/side or is
appended. Lock: results/.eye_lock (mkdir) so the lanes do not overwrite each other."""
import json, os, sys, time
E = "/sp/v5_va/results/scaleup_eye_rejections.json"
LK = "/sp/v5_va/results/.eye_lock"
for _ in range(120):
    try:
        os.mkdir(LK); break
    except FileExistsError:
        time.sleep(1)
try:
    d = json.load(open(E))
    new = json.load(open(sys.argv[1]))
    rows = d.setdefault("rejections", [])
    for n in new:
        k = (n["source"], n["case"], n["side"])
        rows[:] = [r for r in rows if (r.get("source"), r.get("case"), r.get("side")) != k]
        rows.append(n)
    tmp = E + ".tmp_song"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, E)
    print("eye entries:", len(rows), "song:", sorted("%s %s %s" % (r["case"], r["side"], r.get("verdict", "reject")) for r in rows if r.get("source") == "song"))
finally:
    try:
        os.rmdir(LK)
    except OSError:
        pass
