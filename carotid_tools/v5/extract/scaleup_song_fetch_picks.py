#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_fetch_picks.py (= /sp/v5_va/scaleup_song_fetch_picks.py in the build containers)
# scaleup_song_fetch_picks.py (scaleup_lane_song, 2026-10-02) = pilot_fetch_song_picks.py with 3 changes: (1) the cases come from
#   argv (pilot: fixed list of 10); (2) the output MERGES into results/scaleup_song_fetch_picks.json (pilot: wrote
#   results/pilot_fetch_song_picks.json); (3) OVERRIDES holds the scale-up by-eye corrections (pilot: one 090 entry). Rules unchanged.
"""pilot_fetch_song_picks.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:36 EDT): CCA and subclavian (SCA) picks
per side from the qc_pass1 candidate blobs (ids as drawn on qc_axial.png), in the vertebral-column frame of
qc_va_check.json (qc_pass1's own picks used a canal finder that failed on most cases, and a midline that is 29 mm off
on AHMU1218008). Rules (dx = x - column x, +dx = patient right; dy = y - column y, +dy = anterior):
  CCA, levels L3 (arch+60) and L4 (arch+100): circularity >= 0.8, area 20-100 mm2, median HU 0.8-1.6 x aorta, not
      touching bone, dy +3..+70 mm, |dx| 8-60 mm; per side the LARGEST. At arch+100 the CCA may already have divided:
      then the pick is the larger of ICA/ECA (flagged later by eye).
  SCA, levels L1 (arch+15) and L2 (arch+35): circularity >= 0.45, area 30-250 mm2, median HU 0.75-1.6 x aorta, not
      touching bone, |dx| 5-90 mm, dy >= -15 mm; per side the most POSTERIOR (smallest dy).
Every pick is then confirmed or overridden by eye on qc_axial.png: overrides go in OVERRIDES below, with the reason.
Writes results/pilot_fetch_song_picks.json. Read-only on the case folders.
usage: pilot_fetch_song_picks.py   (neve-build-stagea)"""
import json, numpy as np
import sys, os
CASES = sys.argv[1:]
OUTJ = "/sp/v5_va/results/scaleup_song_fetch_picks.json"
IMG = "/repo/carotid_data/va_sources/thin_slice/song"
# (case, level prefix, kind, side) -> (blob id or None, reason); filled after the by-eye review
OVERRIDES = {
    # scale-up by-eye overrides go here: (case, level prefix, kind, side): (blob id or None, reason)
}
out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for c in CASES:
    q = json.load(open("%s/%s/qc_pass1.json" % (IMG, c)))
    vc = json.load(open("%s/%s/qc_va_check.json" % (IMG, c)))
    tr = np.array(vc["column"]["track_every_5mm"], float)
    ao = float(q["aorta_auto"]["aorta_hu_med"])
    rec = dict(aorta_hu_med=ao, picks={}, candidates={})
    for lv, cd in q["candidates"].items():
        p2 = lv[:2]
        i = int(np.argmin(np.abs(tr[:, 0] - cd["z"])))
        colx, coly = tr[i, 1], tr[i, 2]
        kind = "CCA" if p2 in ("L3", "L4") else "SCA"
        cands = []
        for b in cd["blobs"]:
            dx, dy = b["x"] - colx, b["y"] - coly
            r = b["hu_med"] / ao
            if kind == "CCA":
                ok = (b["circ"] >= 0.8 and 20 <= b["area_mm2"] <= 100 and 0.8 <= r <= 1.6 and not b["touch_bone"]
                      and 3 <= dy <= 70 and 8 <= abs(dx) <= 60)
            else:
                ok = (b["circ"] >= 0.45 and 30 <= b["area_mm2"] <= 250 and 0.75 <= r <= 1.6 and not b["touch_bone"]
                      and 5 <= abs(dx) <= 90 and dy >= -15)
            if ok:
                cands.append(dict(id=b["id"], side="R" if dx > 0 else "L", dx=round(dx, 1), dy=round(dy, 1),
                                  area_mm2=b["area_mm2"], circ=b["circ"], hu_med=b["hu_med"], ratio=round(r, 2)))
        rec["candidates"][lv] = dict(z=cd["z"], column=[round(colx, 1), round(coly, 1)], blobs=cands)
        for side in ("R", "L"):
            cs = [x for x in cands if x["side"] == side]
            key = (c, p2, kind, side)
            if key in OVERRIDES:
                bid, why = OVERRIDES[key]
                pick = None if bid is None else dict(next(dict(id=b["id"], hu_med=b["hu_med"], area_mm2=b["area_mm2"],
                                                             x=b["x"], y=b["y"]) for b in cd["blobs"] if b["id"] == bid),
                                                    override=why)
            elif not cs:
                pick = None
            elif kind == "CCA":
                pick = max(cs, key=lambda x: x["area_mm2"])
            else:
                pick = min(cs, key=lambda x: x["dy"])
            rec["picks"]["%s_%s_%s" % (kind, p2, side)] = pick
    out[c] = rec
    pk = rec["picks"]
    print(c, "aorta", ao, " ".join("%s=%s" % (k, "-" if v is None else "%d:%d" % (v["id"], v["hu_med"])) for k, v in sorted(pk.items())))
    for lv, cc in rec["candidates"].items():
        print("   ", lv, "col", cc["column"], [(b["id"], b["side"], b["dx"], b["dy"], b["area_mm2"], b["circ"], b["hu_med"]) for b in cc["blobs"]])
json.dump(out, open(OUTJ + ".tmp", "w"), indent=1); os.replace(OUTJ + ".tmp", OUTJ)
