#!/usr/bin/env python3
"""Three-source graft: host arch -> real carotid bifurcation -> TopBrain siphon.

The existing 49 anatomies vary only in the siphon; every one of them shares a
single cervical carotid and has no carotid bifurcation at all, because the
host's RCCA is one continuous branch. This composes three real sources instead:

    host        ostium to the first seam, the arch and proximal CCA
    lower       a real CCA, a real ICA/ECA bifurcation, and cervical ICA
    siphon      a real TopBrain ICA from the skull base to the terminus

The ECA becomes a NEW branch in the tree. That is the point: entering the
external carotid is a real clinical error, and the current anatomy cannot
represent it because no such fork exists.

SEAM PLACEMENT. The siphon seam is pinned at 130 mm of composed arclength so
these anatomies stay comparable with the 49 that already exist, and the host
cut falls out of that: host_cut = 130 - (cca + ica). Clamped to [15, 72] mm,
since below 15 the host contributes nothing meaningful and above 72 it has
stopped being CCA-calibre. When a lower is long enough to overrun, its CCA is
trimmed proximally rather than the seam being moved.

Both seams get the frame match (tangent onto tangent, superior onto superior)
and the smoothstep radius ramp that graft_siphon.py established, for the same
reasons: a tangent-only match leaves the roll to chance, and an unramped
calibre step is a discontinuity the mesher turns into a shelf.

    python carotid_tools/graft_three.py --pairing <json> --out <dir> [--only NAME]
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graft_siphon import (RCCA_FILE, RVA_FILE, arclength, frame_rotation,
                          is_cranial_stub, kink_deg, min_clearance, prep_siphon,
                          read_curve, resample, rva_deflect, rva_shorten,
                          smoothstep, tangent_at, unit, write_curve,
                          json_to_branch, branch_to_json)
from analyze_bifurcations import read_centerlines, split_tree

SIPHON_SEAM_MM = 130.0
HOST_CUT_MIN, HOST_CUT_MAX = 15.0, 72.0
BLEND_MM = 25.0
RESAMPLE_MM = 1.0
DISTAL_TRIM_MM = 4.0
ECA_NAME = "Centerline curve - RECA.mrk.json"


def ramp_radius(s, r, join_s, target, blend=BLEND_MM):
    """Ease the calibre to `target` over the last `blend` mm before `join_s`."""
    r = np.asarray(r, float).copy()
    a = float(np.interp(max(join_s - blend, s[0]), s, r))
    w = smoothstep((s - (join_s - blend)) / max(blend, 1e-6))
    inside = s >= (join_s - blend)
    return np.where(inside, a + (target - a) * w, r)


def place(src_p, src_r, anchor_p, anchor_t, up=np.array([0.0, 0.0, 1.0])):
    """Translate src onto anchor_p and frame-match its start tangent to anchor_t."""
    t_src = tangent_at(src_p, 0)
    R = frame_rotation(t_src, anchor_t)
    return (src_p - src_p[0]) @ R.T + anchor_p, src_r


def load_lower(rec):
    """(cca, ica, eca) as (points, radii), ICA already extended if it was."""
    paths = read_centerlines(rec["path"])
    cca, dau = split_tree(paths)
    if cca is None or len(dau) < 2:
        return None

    def cal(r):
        return float(np.median(r[:max(len(r) // 2, 1)])) if r is not None and len(r) else 0.0

    dau = sorted(dau, key=lambda d: -cal(d[1]))
    ica, eca = dau[0], dau[1]
    if rec.get("ext_json"):
        e = json.load(open(rec["ext_json"], encoding="utf-8"))
        ica = (np.asarray(e["points"], float),
               None if e["radii"] is None else np.asarray(e["radii"], float))
    return cca, ica, eca


def compose(host_p, host_r, lower, siphon_p, siphon_r):
    """Return (route_points, route_radii, eca_points, eca_radii, diagnostics)."""
    (cca_p, cca_r), (ica_p, ica_r), (eca_p, eca_r) = lower
    cca_p, cca_r = resample(cca_p, cca_r, RESAMPLE_MM)
    ica_p, ica_r = resample(ica_p, ica_r, RESAMPLE_MM)
    eca_p, eca_r = resample(eca_p, eca_r, RESAMPLE_MM)

    cca_L, ica_L = arclength(cca_p)[-1], arclength(ica_p)[-1]
    host_cut = SIPHON_SEAM_MM - (cca_L + ica_L)
    trim_cca = 0.0
    if host_cut < HOST_CUT_MIN:                 # lower overruns: trim its CCA
        trim_cca = HOST_CUT_MIN - host_cut
        s = arclength(cca_p)
        if s[-1] - trim_cca < 5:
            return None, None, None, None, {"why": "CCA too short to trim"}
        keep = s >= trim_cca
        cca_p, cca_r = cca_p[keep], (None if cca_r is None else cca_r[keep])
        cca_L = arclength(cca_p)[-1]
        host_cut = HOST_CUT_MIN
    host_cut = float(min(host_cut, HOST_CUT_MAX))

    hs = arclength(host_p)
    k = int(np.searchsorted(hs, host_cut))
    k = min(max(k, 5), len(host_p) - 2)
    keep_p, keep_r = host_p[:k + 1].copy(), host_p[:k + 1].copy()
    keep_r = host_r[:k + 1].copy()

    # seam 1: host -> CCA
    keep_r = ramp_radius(hs[:k + 1], keep_r, hs[k], float(cca_r[0]))
    cca_moved, _ = place(cca_p, cca_r, keep_p[-1], tangent_at(keep_p, len(keep_p) - 1))
    # the ICA and ECA must move with their own CCA, so apply the same transform
    R = frame_rotation(tangent_at(cca_p, 0), tangent_at(keep_p, len(keep_p) - 1))
    def move(p):
        return (p - cca_p[0]) @ R.T + keep_p[-1]
    ica_moved, eca_moved = move(ica_p), move(eca_p)

    # seam 2: ICA tip -> siphon
    sip_moved, _ = place(siphon_p, siphon_r, ica_moved[-1],
                         tangent_at(ica_moved, len(ica_moved) - 1))
    ica_s = arclength(ica_moved)
    ica_r2 = ramp_radius(ica_s, ica_r, ica_s[-1], float(siphon_r[0]))

    route_p = np.vstack([keep_p, cca_moved[1:], ica_moved[1:], sip_moved[1:]])
    route_r = np.concatenate([keep_r, cca_r[1:], ica_r2[1:], siphon_r[1:]])
    route_p, route_r = resample(route_p, route_r, RESAMPLE_MM)
    s_end = arclength(route_p)
    trim = s_end <= (s_end[-1] - DISTAL_TRIM_MM)
    if trim.sum() >= 30:
        route_p, route_r = route_p[trim], route_r[trim]

    kk = kink_deg(route_p)
    diag = {"host_cut_mm": host_cut, "cca_mm": cca_L, "ica_mm": ica_L,
            "trim_cca_mm": trim_cca, "total_mm": float(arclength(route_p)[-1]),
            "max_kink": float(kk.max()),
            "seam1_kink": float(kk[max(k - 4, 0):k + 4].max()),
            "eca_mm": float(arclength(eca_moved)[-1])}
    return route_p, route_r, eca_moved, eca_r, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairing", default="carotid_data/pairing.json")
    ap.add_argument("--host", default="eve_bench/data/dualdevicenav/Centrelines_comb")
    ap.add_argument("--out", default="carotid_data/anatomies")
    ap.add_argument("--only", default=None, help="comma-separated pair indices or 'a:b' range")
    ap.add_argument("--max-kink", type=float, default=60.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave already-built anatomies alone, so a replacement "
                         "round costs only the new pairs")
    a = ap.parse_args()

    plan = json.load(open(a.pairing, encoding="utf-8"))
    pairs = plan["pairs"]
    if a.only:
        if ":" in a.only:
            lo, hi = a.only.split(":"); pairs = pairs[int(lo):int(hi)]
        else:
            pairs = [pairs[int(i)] for i in a.only.split(",")]

    tmpl, hp, hr = read_curve(os.path.join(a.host, RCCA_FILE))
    hp, hr = resample(hp, hr, RESAMPLE_MM)
    rva_tmpl, vp, vr = read_curve(os.path.join(a.host, RVA_FILE))
    rva_name = RVA_FILE.replace("Centerline curve", "cc").replace(".mrk.json", "")

    others, neigh = [], []
    for f in sorted(os.listdir(a.host)):
        if not f.endswith(".json") or f == RCCA_FILE:
            continue
        _, bp, br = read_curve(os.path.join(a.host, f))
        if is_cranial_stub(f, bp):
            continue
        others.append(f)
        if br is not None:
            neigh.append((f.replace("Centerline curve", "cc").replace(".mrk.json", ""), bp, br))

    os.makedirs(a.out, exist_ok=True)
    ok, fail = 0, []
    print("%-34s %8s %8s %7s %7s %s" % ("anatomy", "total", "host_cut", "kink", "clear", "note"))
    for idx, pr in enumerate(pairs):
        name = "%s__%s" % (pr["lower"], pr["siphon"])
        if a.skip_existing and os.path.isdir(os.path.join(a.out, name, "Centrelines_comb")):
            ok += 1
            continue
        rec = plan["lowers"][pr["lower"]]
        sinfo = plan["siphons"][pr["siphon"]]
        try:
            lower = load_lower(rec)
            if lower is None:
                raise ValueError("lower tree would not split")
            j = json.load(open(sinfo["src"], encoding="utf-8"))
            sp = np.array(j["points"], float)
            sr = np.array(j["radii"], float)
            if sinfo["mirror"]:
                sp = sp * np.array([-1.0, 1.0, 1.0])
            sp, sr = prep_siphon(sp, sr)
            rp, rr, ep, er, diag = compose(hp, hr, lower, sp, sr)
            if rp is None:
                raise ValueError(diag.get("why", "compose failed"))
            if diag["max_kink"] > a.max_kink:
                raise ValueError("kink %.0f deg" % diag["max_kink"])

            nb = [(n, p, q) for n, p, q in neigh]
            clear, who = min_clearance(rp, rr, nb, SIPHON_SEAM_MM)
            note = ""
            rva = None
            if clear < 0 and who == rva_name:
                s2 = int(np.searchsorted(arclength(rp), SIPHON_SEAM_MM))
                cut = rva_shorten(rp[s2:], rr[s2:], vp, vr)
                if cut is not None:
                    rva, note = (cut[0], cut[1]), "RVA shortened %.0f mm" % cut[2]
                else:
                    bent = rva_deflect(rp[s2:], rr[s2:], vp, vr,
                                       [n for n in nb if n[0] != rva_name])
                    if bent is not None:
                        rva, note = (bent[0], bent[1]), "RVA deflected %.1f mm" % bent[2]
                if rva is not None:
                    patched = [(n[0], rva[0], rva[1]) if n[0] == rva_name else n for n in nb]
                    clear, who = min_clearance(rp, rr, patched, SIPHON_SEAM_MM)
            if clear < 0:
                raise ValueError("overlap %.2f mm with %s" % (clear, who))

            folder = os.path.join(a.out, name, "Centrelines_comb")
            os.makedirs(folder, exist_ok=True)
            for o in others:
                if o == RVA_FILE and rva is not None:
                    write_curve(rva_tmpl, os.path.join(folder, o), rva[0], rva[1])
                else:
                    shutil.copy2(os.path.join(a.host, o), os.path.join(folder, o))
            write_curve(tmpl, os.path.join(folder, RCCA_FILE), rp, rr)
            write_curve(tmpl, os.path.join(folder, ECA_NAME), ep, er)
            with open(os.path.join(a.out, name, "provenance.json"), "w", encoding="utf-8") as fh:
                json.dump({"lower": pr["lower"], "siphon": pr["siphon"],
                           "mismatch_mm": pr["mismatch_mm"], **diag}, fh, indent=1)
            ok += 1
            print("%-34s %8.1f %8.1f %7.1f %+7.2f %s"
                  % (name[:34], diag["total_mm"], diag["host_cut_mm"],
                     diag["max_kink"], clear, note))
        except Exception as e:                                # noqa: BLE001
            fail.append((name, str(e)[:60]))
            print("%-34s %s" % (name[:34], "FAILED: " + str(e)[:50]))

    print("\nbuilt %d, failed %d" % (ok, len(fail)))
    for n, why in fail[:20]:
        print("   %-40s %s" % (n[:40], why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
