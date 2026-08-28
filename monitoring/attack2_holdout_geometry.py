#!/usr/bin/env python3
"""ATTACK 2: is the 4-anatomy holdout geometrically easier than the other 18?

Pure numpy, reads .mrk.json centerlines only. Read-only.
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANAT = os.path.join(ROOT, "topbrain_data", "anatomies")
HOST = os.path.join(ROOT, "eve_bench", "data", "dualdevicenav", "Centrelines_comb")
RCCA_FILE = "Centerline curve - RCCA.mrk.json"
EXCLUDE = {"topcow_mr_013", "topcow_mr_014", "topcow_mr_015"}
HOLDOUT = {"topcow_mr_004", "topcow_mr_008", "topcow_mr_017", "topcow_mr_023"}
CRANIAL_Z = 500.0
GRAFT_S = 133.0     # measured first >1mm departure from host course


def json_to_branch(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([y, -z, -x], axis=-1)


def read_curve(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    m = d["markups"][0]
    pos = np.array([cp["position"] for cp in m["controlPoints"]], float)
    rad = None
    for meas in m.get("measurements", []):
        if meas.get("name") == "Radius" and "controlPointValues" in meas:
            rad = np.array(meas["controlPointValues"], float)
    return json_to_branch(pos), rad


def arclength(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def kink_deg(p):
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-9)
    return np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(axis=1), -1, 1)))


def menger_curvature(p):
    """Exact circumscribed-circle curvature at each interior point (1/mm)."""
    a = p[:-2]; b = p[1:-1]; c = p[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    denom = ab * bc * ca
    k = np.where(denom > 1e-12, 4.0 * area / np.maximum(denom, 1e-12), 0.0)
    return k   # index i corresponds to point i+1


def resample(p, r, step=0.5):
    s = arclength(p)
    n = max(int(round(s[-1] / step)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(t, s, p[:, i]) for i in range(3)], axis=1)
    return out, np.interp(t, s, r), t


def load_branches(folder):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.mrk.json"))):
        name = os.path.basename(f)[:-len(".mrk.json")]
        p, r = read_curve(f)
        if r is None:
            continue
        out.append((name, p, r))
    return out


def is_cranial_stub(name, p):
    return " - " not in name and float(p[:, 2].min()) > CRANIAL_Z


def analyse(folder, tag):
    branches = load_branches(folder)
    rcca = [b for b in branches if b[0].endswith("- RCCA")]
    assert len(rcca) == 1, folder
    _, p, r = rcca[0]
    s = arclength(p)
    L = float(s[-1])

    # --- graft region mask on the NATIVE 1 mm stations (as shipped) ---
    g = s >= GRAFT_S
    graft_len = L - float(s[g][0]) if g.any() else 0.0

    # curvature on the native stations (they are ~1 mm apart post-resample)
    k_all = menger_curvature(p)                 # for points 1..n-2
    s_k = s[1:-1]
    gk = s_k >= GRAFT_S
    kg = k_all[gk]
    kink = kink_deg(p)                          # for points 1..n-2 as well
    s_kink = s[1:-1]
    gkk = s_kink >= GRAFT_S
    kkg = kink[gkk]

    rc_min = 1.0 / kg.max() if kg.size and kg.max() > 0 else float("inf")
    res = {
        "anatomy": tag,
        "rcca_len": L,
        "graft_len": graft_len,
        "graft_chord": float(np.linalg.norm(p[-1] - p[g][0])) if g.any() else 0.0,
        "Rc_min": float(rc_min),
        "kappa_p95": float(np.percentile(kg, 95)) if kg.size else float("nan"),
        "kappa_p05": float(np.percentile(kg, 5)) if kg.size else float("nan"),
        "Rc_p05": float(1.0 / np.percentile(kg, 95)) if kg.size and np.percentile(kg, 95) > 0 else float("nan"),
        "max_bend": float(kkg.max()) if kkg.size else float("nan"),
        "turn_cum": float(kkg.sum()) if kkg.size else float("nan"),
        "n_graft_stations": int(g.sum()),
    }
    res["graft_tort"] = res["graft_len"] / res["graft_chord"] if res["graft_chord"] > 0 else float("nan")

    # --- clearance: graft-region RCCA stations vs every OTHER kept branch ---
    others = [(n, bp, br) for (n, bp, br) in branches
              if not n.endswith("- RCCA") and not is_cranial_stub(n, bp)]
    q, qr = p[g], r[g]
    if others and q.size:
        gaps = np.full(len(q), np.inf)
        who = None
        for n, bp, br in others:
            d = np.linalg.norm(q[:, None, :] - bp[None, :, :], axis=2)
            gg = (d - qr[:, None] - br[None, :]).min(axis=1)
            gaps = np.minimum(gaps, gg)
        res["clear_min"] = float(gaps.min())
        res["clear_p05"] = float(np.percentile(gaps, 5))
        res["clear_n_lt_035"] = int((gaps < 0.35).sum())
        res["clear_n_lt_0"] = int((gaps < 0.0).sum())
    else:
        res["clear_min"] = res["clear_p05"] = float("nan")
        res["clear_n_lt_035"] = res["clear_n_lt_0"] = -1

    # --- radius-floor fraction, several candidate definitions ---
    for lab, mask in (("g133", s >= GRAFT_S), ("g130", s >= 130.0),
                      ("half", s >= 0.5 * L), ("g146", s >= 146.0)):
        rr = r[mask]
        res["floor_%s" % lab] = float((rr <= 2.0).mean()) if rr.size else float("nan")
    res["r_min_graft"] = float(r[g].min()) if g.any() else float("nan")
    res["r_med_graft"] = float(np.median(r[g])) if g.any() else float("nan")
    res["r_term"] = float(r[-1])
    return res


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ANAT, "topcow_mr_*"))):
        name = os.path.basename(d)
        rows.append(analyse(os.path.join(d, "Centrelines_comb"), name))
    host = analyse(HOST, "HOST")
    json.dump({"rows": rows, "host": host},
              open(os.path.join(ROOT, "monitoring", "attack2_geom.json"), "w"), indent=1)

    keys = ["floor_g133", "floor_g130", "floor_half", "floor_g146"]
    keep = [x for x in rows if x["anatomy"] not in EXCLUDE]
    print("calibration of the 2.0mm-floor fraction (target: 016=62.9%, 024=68.1%, median 12.6%)")
    for k in keys:
        v16 = [x[k] for x in rows if x["anatomy"] == "topcow_mr_016"][0]
        v24 = [x[k] for x in rows if x["anatomy"] == "topcow_mr_024"][0]
        med22 = float(np.median([x[k] for x in keep]))
        print("  %-11s 016=%.1f%%  024=%.1f%%  median22=%.1f%%" % (k, 100*v16, 100*v24, 100*med22))
    print()
    hdr = ("anat", "L", "gL", "Rcmin", "kp95", "bend", "turn", "tort",
           "clrmin", "clrp05", "n<.35", "flr%", "rmin")
    print("%-14s %6s %6s %7s %7s %6s %7s %6s %7s %7s %5s %6s %5s" % hdr)
    for x in sorted(rows, key=lambda z: z["anatomy"]):
        mark = "*" if x["anatomy"] in HOLDOUT else (" " if x["anatomy"] not in EXCLUDE else "x")
        print("%s%-13s %6.1f %6.1f %7.2f %7.4f %6.1f %7.0f %6.3f %7.2f %7.2f %5d %6.1f %5.2f" % (
            mark, x["anatomy"].replace("topcow_", ""), x["rcca_len"], x["graft_len"],
            x["Rc_min"], x["kappa_p95"], x["max_bend"], x["turn_cum"], x["graft_tort"],
            x["clear_min"], x["clear_p05"], x["clear_n_lt_035"], 100*x["floor_g133"],
            x["r_min_graft"]))
    x = host
    print("%s%-13s %6.1f %6.1f %7.2f %7.4f %6.1f %7.0f %6.3f %7.2f %7.2f %5d %6.1f %5.2f" % (
        "H", "HOST", x["rcca_len"], x["graft_len"], x["Rc_min"], x["kappa_p95"],
        x["max_bend"], x["turn_cum"], x["graft_tort"], x["clear_min"], x["clear_p05"],
        x["clear_n_lt_035"], 100*x["floor_g133"], x["r_min_graft"]))


main()
