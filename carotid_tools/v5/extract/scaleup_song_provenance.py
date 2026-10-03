#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_provenance.py (= /sp/v5_va/scaleup_song_provenance.py in the build containers)
"""scaleup_song_provenance.py (scaleup_lane_song, 2026-10-02; neve-build-stagea or python:3 with the same mounts)
Writes / replaces the section "## Scale-up cases (scaleup_lane_song, 2026-10-02)" at the end of
carotid_data/va_sources/thin_slice/song/PROVENANCE.md: one row per scale-up case fetched so far (every case with a
fetch.json listed in results/scaleup_song_fetch_review.json or in the lane's case lists), from <case>/meta.json,
<case>/fetch.json and the by-eye review. The pilot's part of the file is left as it is."""
import json, os
IMG = "/repo/carotid_data/va_sources/thin_slice/song"
RES = "/sp/v5_va/results"
LOGD = "/repo/carotid_data/va_sources/derived/thin_slice/song/_logs_scaleup"
P = os.path.join(IMG, "PROVENANCE.md")
HEAD = "## Scale-up cases (scaleup_lane_song, 2026-10-02)"
rv = json.load(open(os.path.join(RES, "scaleup_song_fetch_review.json")))
cases = []
for fn in ("batch1_cases.txt", "extension_cases.txt"):
    p = os.path.join(LOGD, fn)
    if os.path.exists(p):
        cases += open(p).read().split()
rows = []
for c in cases:
    fe = os.path.join(IMG, c, "fetch.json")
    if not os.path.exists(fe):
        continue
    f = json.load(open(fe))
    m = json.load(open(os.path.join(IMG, c, "meta.json"))) if os.path.exists(os.path.join(IMG, c, "meta.json")) else {}
    r = rv.get(c) or {}
    im = m.get("image") or {}
    sp = im.get("spacing_mm") or [None] * 3
    sc = m.get("scanner") or {}
    rows.append("| %s | %s | %s %s | %s / %s | %s | %s (%s) | %s | %s |" % (
        c, (m.get("group") or "").replace("aneurysm patient", "aneurysm"), m.get("sex"), m.get("age"),
        sp[2] if sp[2] is None else round(sp[2], 3), sp[0] if sp[0] is None else round(sp[0], 3),
        im.get("z_extent_mm"), sc.get("family"), sc.get("kernel"), f.get("md5"),
        r.get("verdict", "pending by-eye review") + (" (" + r["reject_reason"] + ")" if r.get("reject_reason") else "")))
sec = [HEAD, "",
       "Fetched by the scale-up's Song lane with the pilot's method, unchanged: RAR5 range read of the stored member + CRC32",
       "(`pilot_fetch_song_get.py`, `<case>/fetch.json`), QC pass 1 + MIPs + side files (`pilot_fetch_song_qc.py`), the",
       "column-frame VA check and zoomed VA crops (`pilot_fetch_song_va_check2.py`, `_va_zoom.py`), picks + `meta.json`",
       "(`scaleup_song_fetch_picks.py`, `scaleup_song_fetch_meta.py`: the pilot scripts with scale-up result paths). Where the",
       "automatic arch-top track failed, the crops were re-made with the TotalSegmentator aorta top (`scaleup_song_refqc.sh`;",
       "the first outputs are kept as `*_autoarch.*`). By-eye review: `scratchpad/v5_va/results/scaleup_song_fetch_review.json`;",
       "summary `results/scaleup_song_fetch.json`. Licence as above: CC BY 4.0; derived through Arterial = non-commercial.", "",
       "| case | group | sex / age | slice / in-plane (mm) | z (mm) | vendor (kernel) | md5 of ct.nii.gz | verdict |",
       "|---|---|---|---|---|---|---|---|"] + rows + [""]
txt = open(P).read()
if HEAD in txt:
    txt = txt[:txt.index(HEAD)].rstrip("\n") + "\n\n"
else:
    txt = txt.rstrip("\n") + "\n\n"
open(P + ".tmp", "w").write(txt + "\n".join(sec))
os.replace(P + ".tmp", P)
print("PROVENANCE.md: %d scale-up rows" % len(rows))
