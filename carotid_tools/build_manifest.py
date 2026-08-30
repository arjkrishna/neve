#!/usr/bin/env python3
"""Measure every carotid bifurcation model and write a manifest.

Everything downstream keys off this: which models are long enough to reach the
TopBrain seam unaided, which need the 10 mm extension, and which are too short
to rescue. Also records the frame each model lives in, because these are CTA
patient coordinates and the graft has to know which way is superior, exactly as
it did for the TopBrain siphons.

Writes carotid_data/lower_manifest.json:
    {name, side, cca_mm, ica_mm, eca_mm, calibres, bif_deg, stenosis_pct,
     superior_axis, ica_tip, ica_tip_tangent, ...}

    python carotid_tools/build_manifest.py <db_dir> --out <manifest.json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_bifurcations import read_centerlines, split_tree, arclen, unit

# The host stays CCA-calibre to 72 mm and the TopBrain seam sits at 130 mm, so
# a lower section must carry 58 mm of ICA past its bifurcation to tile the gap
# unaided. See the tiling analysis.
NEED_ICA_MM = 58.0
MAX_EXTEND_MM = 10.0


def describe_full(path):
    paths = read_centerlines(path)
    cca, dau = split_tree(paths)
    if cca is None or len(dau) < 2:
        return None
    cca_p, cca_r = cca

    def calibre(r, frac=0.5):
        if r is None or not len(r):
            return 0.0
        n = max(int(len(r) * frac), 1)
        return float(np.median(r[:n]))

    dau = sorted(dau, key=lambda d: -calibre(d[1]))
    (ica_p, ica_r), (eca_p, eca_r) = dau[0], dau[1]

    # Superior: these are CTA patient coordinates, so rather than trust a tag,
    # take the direction the carotid actually runs. CCA to ICA tip is superior
    # by construction of the anatomy.
    sup = unit(ica_p[-1] - cca_p[0])

    def tangent(p, i, span=12):
        lo, hi = max(0, i - span), min(len(p) - 1, i + span)
        return unit(p[hi] - p[lo])

    ica_len = arclen(ica_p)
    out = {
        "name": os.path.basename(path).replace("_lumen_centerlines.vtp", ""),
        "path": path,
        "cca_mm": arclen(cca_p), "ica_mm": ica_len, "eca_mm": arclen(eca_p),
        "cca_d": 2 * calibre(cca_r), "ica_d": 2 * calibre(ica_r),
        "eca_d": 2 * calibre(eca_r),
        "n_daughters": len(dau),
        "superior": sup.tolist(),
        "bif_point": cca_p[-1].tolist(),
        "ica_tip": ica_p[-1].tolist(),
        "ica_tip_tangent": tangent(ica_p, len(ica_p) - 1).tolist(),
        "ica_tip_radius": float(ica_r[-1]) if ica_r is not None else None,
        "extend_mm": max(0.0, NEED_ICA_MM - ica_len),
    }
    out["side"] = "left" if "_left" in out["name"] else "right"
    if ica_r is not None and len(ica_r) > 10:
        distal = float(np.median(ica_r[-max(len(ica_r) // 3, 1):]))
        out["ica_min_d"] = 2 * float(ica_r.min())
        out["stenosis_pct"] = 100.0 * (1 - float(ica_r.min()) / max(distal, 1e-6))
    # bifurcation angle
    n = min(len(ica_p), len(eca_p), 40)
    va, vb = unit(ica_p[n - 1] - ica_p[0]), unit(eca_p[n - 1] - eca_p[0])
    out["bif_deg"] = float(np.degrees(np.arccos(np.clip(va @ vb, -1, 1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("database")
    ap.add_argument("--out", default="carotid_data/lower_manifest.json")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.database, "**", "*_centerlines.vtp"),
                             recursive=True))
    rows, bad = [], []
    for f in files:
        try:
            d = describe_full(f)
        except Exception as e:                                # noqa: BLE001
            bad.append((os.path.basename(f), str(e)[:50])); continue
        if d is None:
            bad.append((os.path.basename(f), "tree would not split")); continue
        rows.append(d)

    ica = np.array([r["ica_mm"] for r in rows])
    ready = [r for r in rows if r["extend_mm"] <= 0]
    extend = [r for r in rows if 0 < r["extend_mm"] <= MAX_EXTEND_MM]
    short = [r for r in rows if r["extend_mm"] > MAX_EXTEND_MM]

    print("parsed %d models (%d unparsed)" % (len(rows), len(bad)))
    print("  reach the %.0f mm seam unaided : %3d   <- extension templates come from these"
          % (NEED_ICA_MM, len(ready)))
    print("  need <= %.0f mm extension        : %3d" % (MAX_EXTEND_MM, len(extend)))
    print("  too short to rescue            : %3d" % len(short))
    print("  usable pool after extension    : %3d  -> x4 uses = %d anatomies"
          % (len(ready) + len(extend), 4 * (len(ready) + len(extend))))
    print("\n  ICA length: %.1f - %.1f mm, median %.1f" % (ica.min(), ica.max(), np.median(ica)))
    sides = {}
    for r in ready + extend:
        sides[r["side"]] = sides.get(r["side"], 0) + 1
    print("  usable by side: %s" % sides)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"need_ica_mm": NEED_ICA_MM, "max_extend_mm": MAX_EXTEND_MM,
                   "models": rows, "unparsed": bad}, fh, indent=1)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
