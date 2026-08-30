#!/usr/bin/env python3
"""What is actually in the CarotidAnalyzer bifurcation database, and is it graftable?

The database ships lumen surfaces, plaque surfaces and a centerline tree per
carotid. The centerline tree is a set of VMTK root-to-tip paths that all start
at the same inlet, so the CCA is their shared prefix and the daughters are what
remains after they diverge. That structure is what has to be recovered before
any of it can be spliced onto the host, because the host's RCCA is a single
continuous branch with no modelled bifurcation at all.

Measures, per model:
  CCA length and calibre below the bifurcation
  ICA and ECA lengths and calibres above it
  which daughter is the ICA (the larger one; the ECA is consistently smaller)
  bifurcation angle
  a stenosis proxy: narrowest ICA calibre against its own distal calibre

and compares the CCA calibre against the host's RCCA radius profile, which is
what decides whether a seam is even possible.

    python carotid_tools/analyze_bifurcations.py <database_dir> [--host <dir>]
"""
import argparse
import glob
import os
import sys

import numpy as np


def read_centerlines(path):
    """[(points, radii)] one entry per polyline cell, in traversal order."""
    import pyvista as pv
    m = pv.read(path)
    pts = np.asarray(m.points, float)
    rad = None
    for k in m.point_data.keys():
        if "MaximumInscribedSphere" in k:
            rad = np.asarray(m.point_data[k], float)
    out = []
    for c in range(m.n_cells):
        ids = m.GetCell(c).GetPointIds()
        idx = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if len(idx) < 2:
            continue
        out.append((pts[idx], None if rad is None else rad[idx]))
    return out


def arclen(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def split_tree(paths, tol=0.05):
    """(cca, daughters) by finding the prefix every path shares.

    All paths run from the same inlet, so the common prefix is the CCA and
    everything past the divergence is a daughter. Paths that stay identical
    past the split are duplicates of the same daughter and are collapsed.
    """
    if not paths:
        return None, []
    ref = paths[0][0]
    k = len(ref)
    for p, _ in paths[1:]:
        n = min(k, len(p))
        d = np.linalg.norm(ref[:n] - p[:n], axis=1)
        far = np.nonzero(d > tol)[0]
        k = min(k, int(far[0]) if len(far) else n)
    cca_p, cca_r = paths[0][0][:k], (paths[0][1][:k] if paths[0][1] is not None else None)

    tips, dau = [], []
    for p, r in paths:
        if len(p) <= k + 5:
            continue
        tip = p[-1]
        if any(np.linalg.norm(tip - t) < 1.0 for t in tips):
            continue                       # duplicate path to the same tip
        tips.append(tip)
        dau.append((p[k:], None if r is None else r[k:]))
    return (cca_p, cca_r), dau


def describe(path):
    paths = read_centerlines(path)
    cca, dau = split_tree(paths)
    if cca is None or len(dau) < 2:
        return None
    cca_p, cca_r = cca
    # ICA is the larger daughter; the ECA is consistently the smaller vessel
    def calibre(r):
        return float(np.median(r[:len(r) // 2])) if r is not None and len(r) else 0.0
    dau = sorted(dau, key=lambda d: -calibre(d[1]))
    ica, eca = dau[0], dau[1]

    def ang(a, b):
        n = min(len(a), len(b), 40)
        va, vb = unit(a[n - 1] - a[0]), unit(b[n - 1] - b[0])
        return float(np.degrees(np.arccos(np.clip(va @ vb, -1, 1))))

    out = {
        "n_paths": len(paths), "n_daughters": len(dau),
        "cca_mm": arclen(cca_p), "ica_mm": arclen(ica[0]), "eca_mm": arclen(eca[0]),
        "cca_d": 2 * calibre(cca_r), "ica_d": 2 * calibre(ica[1]),
        "eca_d": 2 * calibre(eca[1]),
        "bif_deg": ang(ica[0], eca[0]),
    }
    if ica[1] is not None and len(ica[1]) > 10:
        distal = float(np.median(ica[1][-len(ica[1]) // 3:]))
        out["ica_min_d"] = 2 * float(ica[1].min())
        out["stenosis_pct"] = 100.0 * (1 - ica[1].min() / max(distal, 1e-6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("database")
    ap.add_argument("--host", default="/opt/eve_training/eve_bench/data/dualdevicenav")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.database, "**", "*_centerlines.vtp"),
                             recursive=True))
    if a.limit:
        files = files[:a.limit]
    print("centerline models found: %d\n" % len(files), flush=True)

    rows, bad = [], []
    print("%-30s %5s %7s %7s %7s %6s %6s %6s %6s %7s"
          % ("model", "paths", "CCA_mm", "ICA_mm", "ECA_mm",
             "CCA_d", "ICA_d", "ECA_d", "bif", "sten%"))
    for f in files:
        try:
            d = describe(f)
        except Exception as e:                                # noqa: BLE001
            bad.append((os.path.basename(f), str(e)[:40])); continue
        if d is None:
            bad.append((os.path.basename(f), "could not split tree")); continue
        d["name"] = os.path.basename(f).replace("_lumen_centerlines.vtp", "")
        rows.append(d)
        print("%-30s %5d %7.1f %7.1f %7.1f %6.2f %6.2f %6.2f %6.0f %7.0f"
              % (d["name"][:30], d["n_paths"], d["cca_mm"], d["ica_mm"], d["eca_mm"],
                 d["cca_d"], d["ica_d"], d["eca_d"], d["bif_deg"],
                 d.get("stenosis_pct", float("nan"))))

    if not rows:
        print("nothing parsed"); return 1

    def col(k):
        return np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])

    print("\n%d models parsed, %d unparsed" % (len(rows), len(bad)))
    for k, lab in (("cca_mm", "CCA length"), ("ica_mm", "ICA length"),
                   ("eca_mm", "ECA length"), ("cca_d", "CCA diameter"),
                   ("ica_d", "ICA diameter"), ("eca_d", "ECA diameter"),
                   ("bif_deg", "bifurcation angle"), ("stenosis_pct", "ICA stenosis")):
        v = col(k)
        if len(v):
            print("  %-20s %6.1f - %6.1f   median %6.1f" % (lab, v.min(), v.max(), np.median(v)))

    # --- does the CCA calibre match anywhere on the host's RCCA?
    try:
        sys.path.insert(0, "/opt/eve_training/eve_bench")
        from eve_bench.dualdevicenav import load_branches
        br = [b for b in load_branches(os.path.join(a.host, "Centrelines_comb"))
              if "RCCA" in str(b.name).upper()][0]
        c = np.asarray(br.coordinates, float); r = np.asarray(br.radii, float)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        cca_d = col("cca_d")
        lo, hi = np.percentile(cca_d, [10, 90])
        ok = (2 * r >= lo) & (2 * r <= hi)
        print("\nhost RCCA vs this database's CCA calibre (%.2f-%.2f mm, 10-90th pct):" % (lo, hi))
        if ok.any():
            print("  host RCCA is in that range over %.0f-%.0f mm of its %.0f mm"
                  % (s[ok].min(), s[ok].max(), s[-1]))
        else:
            print("  host RCCA NEVER falls in that range (host %.2f-%.2f mm)"
                  % (2 * r.min(), 2 * r.max()))
    except Exception as e:                                    # noqa: BLE001
        print("\n(host comparison unavailable: %s)" % e)

    if bad:
        print("\nunparsed:")
        for n, why in bad[:10]:
            print("   %-40s %s" % (n[:40], why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
