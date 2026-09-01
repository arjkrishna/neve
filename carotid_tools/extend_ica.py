#!/usr/bin/env python3
"""Extend short cervical ICAs using the distal shape of the ones that are long enough.

34 bifurcation models fall short of the 58 mm needed to reach the TopBrain
seam, by 10 mm or less. Rather than continue them with a straight line, which
would put an unnaturally rigid segment exactly where the vessel should still be
curving, the extension is COPIED from the 31 models that do have that stretch.

Each donor contributes its own distal 10 mm expressed in a local frame at the
point 10 mm before its tip: tangent, the superior direction orthogonalised
against it, and their cross product. Mapping that into the recipient's tip
frame reproduces the donor's curvature and torsion SENSE relative to the
vessel, not merely its length, and keeps the join C1-continuous because the
first template point sits on the recipient's own tangent.

The reference axis is +z, the scanner's superior. It used to be the model's
own `superior` entry from the manifest, which is not an anatomical axis at all
but the CCA-inlet-to-ICA-tip CHORD -- that is, roughly the direction the
vessel already runs. Orthogonalising the tangent against a vector nearly
parallel to it leaves almost nothing to normalise, so the roll of the copied
segment was decided by numerical noise: three models sat under |n| = 0.15,
case_m_022_left at 0.047 (2.7 deg), and it duly turned up among the anatomies
with the worst rise. +z is both the correct anatomical reference and the
better-conditioned one for every model in the set.

Donors are drawn per recipient rather than averaged: averaging 31 real
continuations would produce one bland arc and erase the variation that is the
whole point of using real anatomy.

    python carotid_tools/extend_ica.py --manifest <json> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_bifurcations import read_centerlines, split_tree, unit

# scanner superior; see the module docstring for why this is not per-model
SUPERIOR = np.array([0.0, 0.0, 1.0])


def arclength(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample(p, r, step=0.5):
    s = arclength(p)
    n = max(int(round(s[-1] / step)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    q = np.stack([np.interp(t, s, p[:, i]) for i in range(3)], axis=1)
    return q, (np.interp(t, s, r) if r is not None else None)


def frame(t, up):
    """Rows [tangent, up-in-plane, binormal]; the same construction the graft uses."""
    t = unit(np.asarray(t, float))
    n = np.asarray(up, float) - t * float(np.asarray(up, float) @ t)
    if np.linalg.norm(n) < 1e-6:
        n = np.array([1.0, 0.0, 0.0]) - t * float(t[0])
    n = unit(n)
    return np.stack([t, n, np.cross(t, n)])


def ica_of(path):
    paths = read_centerlines(path)
    cca, dau = split_tree(paths)
    if cca is None or len(dau) < 2:
        return None, None, None

    def cal(r):
        return float(np.median(r[:max(len(r) // 2, 1)])) if r is not None and len(r) else 0.0

    dau = sorted(dau, key=lambda d: -cal(d[1]))
    return dau[0][0], dau[0][1], cca[0]


def tangent_at(p, i, span=12):
    lo, hi = max(0, i - span), min(len(p) - 1, i + span)
    return unit(p[hi] - p[lo])


def make_template(p, r, sup, length):
    """The distal `length` mm, in the local frame `length` mm before the tip."""
    s = arclength(p)
    if s[-1] < length + 5:
        return None
    i = int(np.searchsorted(s, s[-1] - length))
    F = frame(tangent_at(p, i), sup)
    local = (p[i:] - p[i]) @ F.T                      # rows of F are the basis
    rad = None if r is None else r[i:] / max(float(r[i]), 1e-6)
    return {"local": local, "radius_ratio": rad}


def apply_template(p, r, sup, tpl, need):
    """Append the first `need` mm of a template to the tip of (p, r)."""
    F = frame(tangent_at(p, len(p) - 1), sup)
    loc = tpl["local"]
    s = arclength(loc)
    keep = s <= need
    if keep.sum() < 2:
        return None, None
    seg = loc[keep]
    world = seg @ F + p[-1]                           # F rows are orthonormal
    new_p = np.vstack([p, world[1:]])
    if r is None or tpl["radius_ratio"] is None:
        return new_p, r
    ratio = tpl["radius_ratio"][keep]
    new_r = np.concatenate([r, r[-1] * ratio[1:]])
    return new_p, new_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="carotid_data/lower_manifest.json")
    ap.add_argument("--out", default="carotid_data/extended")
    ap.add_argument("--seed", type=int, default=20260830)
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))
    need_mm, max_ext = man["need_ica_mm"], man["max_extend_mm"]
    models = man["models"]
    donors = [m for m in models if m["extend_mm"] <= 0]
    recips = [m for m in models if 0 < m["extend_mm"] <= max_ext]
    print("donors (ICA >= %.0f mm): %d   recipients: %d" % (need_mm, len(donors), len(recips)))

    tpls = []
    for d in donors:
        p, r, _ = ica_of(d["path"])
        if p is None:
            continue
        p, r = resample(p, r)
        t = make_template(p, r, SUPERIOR, max_ext)
        if t is not None:
            tpls.append((d["name"], t))
    print("usable templates: %d" % len(tpls))
    if not tpls:
        print("no templates; aborting"); return 1

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ok, fail = 0, []
    print("\n%-26s %8s %8s %9s %-24s" % ("model", "was_mm", "now_mm", "added", "donor"))
    for m in recips:
        p, r, _ = ica_of(m["path"])
        if p is None:
            fail.append((m["name"], "no ICA")); continue
        p, r = resample(p, r)
        before = float(arclength(p)[-1])
        name, tpl = tpls[int(rng.integers(len(tpls)))]
        np_, nr_ = apply_template(p, r, SUPERIOR, tpl,
                                  m["extend_mm"] + 0.5)
        if np_ is None:
            fail.append((m["name"], "template too short")); continue
        after = float(arclength(np_)[-1])
        # a kink at the join would defeat the point of copying real curvature
        d = np.diff(np_, axis=0)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        kink = float(np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(axis=1), -1, 1))).max())
        out = {"name": m["name"], "source": m["path"], "donor": name,
               "ica_mm_before": before, "ica_mm_after": after,
               "max_kink_deg": kink,
               "points": np_.tolist(),
               "radii": None if nr_ is None else nr_.tolist()}
        with open(os.path.join(a.out, m["name"] + "_ica_extended.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh)
        ok += 1
        print("%-26s %8.1f %8.1f %9.1f %-24s%s"
              % (m["name"][:26], before, after, after - before, name[:24],
                 "  KINK %.0f" % kink if kink > 40 else ""))

    print("\nextended %d, failed %d" % (ok, len(fail)))
    for n, why in fail[:10]:
        print("   %-30s %s" % (n[:30], why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
