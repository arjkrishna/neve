#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\paths_d17.py (= /sp/v6/wall_d17/paths_d17.py in the build containers)
"""paths_d17.py (s2_harden_wall_bake, user decision D17, 2026-10-04): the donor's OWN subclavian path to the END of
its segmentation (arm side), for the (17) / (19) branch that now ends as a cut end (no seam-2 bridge to the host).

= v6_va's subclavian_paths.py (imported UNCHANGED: Arterial labelled-centreline xSA chain, origin -> ostium -> distal
largest-radius continuation; TS / record fallback; VMR 0248 expert path) with its distal cap KEEP_PAST_MM (45 mm) lifted:
the path now runs to the end of the donor's segmentation:
  * Arterial sides: the end of the xSA chain (the walk stops where Arterial's subclavian label chain stops);
  * fallback sides (no Arterial chain): the record's subclavian (+-40 mm, v5's ct_seg4 station path) -- its end;
  * VMR 0248: the end of the expert SimVascular subclavian path;
then clipped to the donor CT field of view (3 mm inside its bounds) so the ct_seg request stays inside the CT.
Up to the old 45 mm cap every point is v6_va's path exactly (checked here, recorded).
Writes /v6w/d17/paths/<donor_id>_<side>.json (v6_va's schema + "d17") and /v6w/d17/paths/_summary.json.
    python3 paths_d17.py            (neve-build-stagea; reads /repo read-only)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/va")
import subclavian_paths as SPth  # noqa: E402

VERSION = "paths_d17 1.1 (2026-10-04, s2_harden_wall_bake, D17; 1.1: end clipped at a calibre collapse)"
OLD = "/repo/carotid_data/v6_work/va/subclavian/paths"
OUT = os.environ.get("D17_PATHS_OUT", "/v6w/d17/paths")
FOV_MARGIN = 3.0
TUBE_MIN_PAST = 30.0     # D17: label < ~30 mm past the ostium -> short tube extension (VA tool), flagged
COLLAPSE = 0.5          # calibre collapse at the chain end: radius < 0.5 x the subclavian calibre about the ostium


def main():
    SPth.KEEP_PAST_MM = 1.0e9
    SPth.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    post_only = os.environ.get("D17_POST_ONLY") == "1"
    if not post_only:
        SPth.main()      # writes OUT/<donor>_<side>.json + _summary.json (v6_va's code path, cap lifted)
    summ = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        p = os.path.join(OUT, f)
        o = json.load(open(p))
        if o.get("status") != "ok":
            summ[f[:-5]] = {"status": o.get("status")}
            continue
        if post_only:
            # re-post-process from the uncapped walk kept in the file (the 1.0 run did not clip anywhere: written
            # path == uncapped walk; checked by length)
            assert "path_uncapped" in o or abs(o["d17"]["path_len_uncapped_mm"] - o["path_len_mm"]) < 1e-6, f
            if "path_uncapped" in o:
                o["path"], o["radius_path"] = o["path_uncapped"], o["radius_uncapped"]
        else:
            o["path_uncapped"], o["radius_uncapped"] = o["path"], o["radius_path"]
        o.setdefault("path_uncapped", o["path"])
        o.setdefault("radius_uncapped", o["radius_path"])
        P = np.asarray(o["path"], float)
        r = np.asarray(o["radius_path"], float)
        s = SPth.arclen(P)
        s_o = float(o["s_ostium_mm"])
        d17 = {"script": VERSION, "path_len_uncapped_mm": float(s[-1])}
        # FOV clip (thin-slice: CT bounds; VMR: z range)
        fov = o.get("fov") or {}
        inside = np.ones(len(P), bool)
        if "lo" in fov:
            lo3, hi3 = np.asarray(fov["lo"]) + FOV_MARGIN, np.asarray(fov["hi"]) - FOV_MARGIN
            inside = np.all((P >= lo3) & (P <= hi3), axis=1)
        elif "z_mm" in fov:
            zlo, zhi = fov["z_mm"]
            inside = (P[:, 2] >= zlo + FOV_MARGIN) & (P[:, 2] <= zhi - FOV_MARGIN)
        k_end = len(P) - 1
        out_after = np.nonzero(~inside & (s > s_o))[0]
        if len(out_after):
            k_end = int(out_after[0]) - 1
            d17["fov_clipped"] = True
            d17["fov_clip_s_mm"] = float(s[k_end])
        else:
            d17["fov_clipped"] = False
        P, r, s = P[:k_end + 1], r[:k_end + 1], s[:k_end + 1]
        info = o.get("path_info") or {}
        how = ("Arterial xSA chain end" if "segments" in info else
               ("expert SimVascular path end" if "SimVascular" in str(info.get("src")) else "record subclavian end (fallback)"))
        d17["end_how"] = "CT field of view (clipped %.0f mm inside)" % FOV_MARGIN if d17["fov_clipped"] else how
        # calibre collapse at the chain end (Arterial's lumen fading out, or the walk's last step into a small branch):
        # the path ends where the 5 mm running median of the radius falls below COLLAPSE x the subclavian's own calibre
        # about the ostium [O - 5, O + 20] and stays below to the end (only beyond O + 25 mm)
        ref = float(np.median(r[(s >= s_o - 5.0) & (s <= s_o + 20.0)]))
        from scipy.ndimage import median_filter
        rm = median_filter(r, size=11, mode="nearest")
        low = rm < COLLAPSE * ref
        k_c = len(r)
        while k_c > 0 and low[k_c - 1]:
            k_c -= 1
        d17["calibre_ref_mm"] = round(ref, 3)
        if k_c < len(r) and s[min(k_c, len(s) - 1)] > s_o + 25.0:
            d17["calibre_collapse_clipped_mm"] = round(float(s[-1] - s[k_c]), 2)
            d17["end_how"] += "; clipped at the calibre collapse (radius < %.1f x %.2f mm)" % (COLLAPSE, ref)
            P, r, s = P[:k_c + 1], r[:k_c + 1], s[:k_c + 1]
        d17["past_ostium_mm"] = float(s[-1] - s_o)
        d17["tube_extension_needed"] = bool(d17["past_ostium_mm"] < TUBE_MIN_PAST)
        d17["tube_extension_mm_to_30"] = max(0.0, TUBE_MIN_PAST - d17["past_ostium_mm"])
        # identity with v6_va's capped path up to its end
        q = os.path.join(OLD, f)
        if os.path.isfile(q):
            Po = np.asarray(json.load(open(q))["path"], float)
            n = min(len(Po), len(P))
            d17["equals_v6va_path_first_n"] = n
            d17["equals_v6va_path_maxdiff_mm"] = float(np.abs(P[:n] - Po[:n]).max()) if n else None
            d17["v6va_path_len_mm"] = float(SPth.arclen(Po)[-1])
        o["path"], o["radius_path"] = P.round(3), r.round(3)
        o["path_len_mm"] = float(s[-1])
        o["past_ostium_mm"] = float(s[-1] - s_o)
        o["d17"] = d17
        SPth.jdump(o, p)
        summ[f[:-5]] = {"past_ostium_mm": round(d17["past_ostium_mm"], 1), "end_how": d17["end_how"],
                        "uncapped_mm": round(d17["path_len_uncapped_mm"], 1), "tube": d17["tube_extension_needed"],
                        "equals_v6va_maxdiff": d17.get("equals_v6va_path_maxdiff_mm"),
                        "r_end_mm": round(float(np.median(r[-10:])), 2)}
        print(f[:-5], json.dumps(summ[f[:-5]]), flush=True)
    SPth.jdump({"script": VERSION, "sides": summ}, os.path.join(OUT, "_d17_summary.json"))


if __name__ == "__main__":
    main()
