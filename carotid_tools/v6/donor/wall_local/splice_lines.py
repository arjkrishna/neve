#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\splice_lines.py (= /sp/v6/wall_local/splice_lines.py in the build containers)
"""splice_lines.py 1.1 (s2_harden_wall_bake, D15): the local donor VA's centreline = v5's own VMTK MISR line outside the
artefact zones (where the local surface IS v5's surface), and the VMTK line of the local surface inside the zones.

Per side of /v6w/seg/local/<unit>: <v>_stageA_vmtk.json as written by vmr_vmtk.py on the local surface is kept as
<v>_stageA_vmtk_raw.json; v5's line = the v5 record's centerline src (donors_v5/<unit>/<side>.json, VMTK json); each
line point gets the weight w of its nearest v5 station (zones.json). Every maximal run of v5 points with w > 0 is
replaced by the raw points with w > 0 that lie between the projections of the run's two outside neighbours onto the raw
line (to the line end where the run reaches an end). Recorded: the raw line's deviation from v5's line OUTSIDE the zones
(VMTK on an identical surface stretch), the gaps at every splice junction, the lengths.
    python3 splice_lines.py [unit,...]       (any image with numpy / scipy)
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v6/wall_local")
import local_seg as LS  # noqa: E402

SEG = "/v6w/seg/local"
VP = {"R": "rva", "L": "lva"}


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def turn_1mm_max(P):
    """largest angle between consecutive 1 mm chords of the line resampled every 1 mm (deg)."""
    s = arclen(P)
    q = np.arange(0.0, s[-1], 1.0)
    if len(q) < 3:
        return 0.0
    Q = np.stack([np.interp(q, s, P[:, k]) for k in range(3)], 1)
    d = np.diff(Q, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)
    c = np.clip((d[:-1] * d[1:]).sum(1), -1, 1)
    return float(np.degrees(np.arccos(c)).max())


def proj_index(P, X):
    """index into polyline P of the vertex just before the projection of X, and the fractional position."""
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((X - a) * ab).sum(1) / L2, 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - X, axis=1)
    j = int(np.argmin(d))
    return j, float(t[j]), float(d[j])


def splice_side(unit, sd):
    v = VP[sd]
    d = os.path.join(SEG, unit)
    raw_p = os.path.join(d, "%s_stageA_vmtk_raw.json" % v)
    cur_p = os.path.join(d, "%s_stageA_vmtk.json" % v)
    if not os.path.exists(raw_p):
        if not os.path.exists(cur_p):
            return {"status": "no VMTK line"}
        shutil.copy2(cur_p, raw_p)
    raw = json.load(open(raw_p))
    rec5 = json.load(open(os.path.join(LS.R5, unit, "%s.json" % sd)))
    src5 = rec5["centerline"]["src"]
    v5 = json.load(open(src5))
    P5, R5_ = np.asarray(v5["points"], float), np.asarray(v5["radii"], float)
    Pr, Rr = np.asarray(raw["points"], float), np.asarray(raw["radii"], float)
    if np.linalg.norm(Pr[0] - P5[0]) > np.linalg.norm(Pr[-1] - P5[0]):
        Pr, Rr = Pr[::-1].copy(), Rr[::-1].copy()
    z = json.load(open(os.path.join(d, "zones.json")))[sd]
    st = json.load(open(os.path.join(LS.D[unit]["v5_seg_dir"], "stations_rays.json")))[v]
    Ps = np.asarray(st["P"], float)
    w = np.asarray(z["w_station"], float)
    n = min(len(Ps), len(w))
    tree = cKDTree(Ps[:n])
    w5 = w[tree.query(P5)[1]]
    wr = w[tree.query(Pr)[1]]
    out5 = w5 == 0
    # raw deviation from v5 outside the zones
    dev = cKDTree(P5).query(Pr[wr == 0])[0] if (wr == 0).any() else np.zeros(1)
    pts, rad, gaps, runs_, lats = [], [], [], [], []
    i, N = 0, len(P5)
    while i < N:
        if out5[i]:
            pts.append(P5[i])
            rad.append(R5_[i])
            i += 1
            continue
        j = i
        while j + 1 < N and not out5[j + 1]:
            j += 1
        # raw stretch between the projections of P5[i-1] and P5[j+1]
        ja = proj_index(Pr, P5[i - 1][None])[0] + 1 if i > 0 else 0
        jb = proj_index(Pr, P5[j + 1][None])[0] if j + 1 < N else len(Pr) - 1
        seg = [k for k in range(ja, jb + 1) if wr[k] > 0] if jb >= ja else []
        if not seg:  # zone shorter than the raw sampling: keep v5's points there
            seg_p, seg_r = P5[i:j + 1], R5_[i:j + 1]
            how = "v5 kept (no raw point inside)"
        else:
            seg_p, seg_r = Pr[seg[0]:seg[-1] + 1].copy(), Rr[seg[0]:seg[-1] + 1].copy()
            how = "raw"
            # 1.1: the line follows the surface blend -- each raw point is blended with its projection on v5's line by
            # the zone weight w of its nearest station: P = (1 - w) Q5 + w P_raw (w = 1 in the core: the raw line;
            # w -> 0 at the blend ends: v5's line), the MISR likewise; continuous with v5's line at both junctions
            wseg = wr[seg[0]:seg[-1] + 1]
            for q in range(len(seg_p)):
                if wseg[q] >= 1.0:
                    continue
                jq, tq, _ = proj_index(P5, seg_p[q][None])
                Q5 = P5[jq] + tq * (P5[jq + 1] - P5[jq])
                R5q = R5_[jq] + tq * (R5_[jq + 1] - R5_[jq])
                seg_p[q] = (1.0 - wseg[q]) * Q5 + wseg[q] * seg_p[q]
                seg_r[q] = (1.0 - wseg[q]) * R5q + wseg[q] * seg_r[q]
            how = "raw, blended by w"
        g0 = float(np.linalg.norm(seg_p[0] - P5[i - 1])) if i > 0 else None
        g1 = float(np.linalg.norm(seg_p[-1] - P5[j + 1])) if j + 1 < N else None
        gaps += [g for g in (g0, g1) if g is not None]
        lat = [proj_index(P5, seg_p[0][None])[2] if i > 0 else None, proj_index(P5, seg_p[-1][None])[2] if j + 1 < N else None]
        lats += [x for x in lat if x is not None]
        runs_.append({"v5_idx": [i, j], "v5_s_mm": [float(arclen(P5)[i]), float(arclen(P5)[j])], "raw_points": len(seg_p),
                      "how": how, "gap_in_mm": g0, "gap_out_mm": g1, "lateral_in_out_mm": lat})
        pts.extend(seg_p)
        rad.extend(seg_r)
        i = j + 1
    P = np.asarray(pts)
    R = np.asarray(rad)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    P, R = P[keep], R[keep]
    o = dict(raw)
    o.update(points=P.tolist(), radii=R.tolist(), length_mm=float(arclen(P)[-1]), n=int(len(P)),
             note=(raw.get("note") or "") + "; D15 splice: v5's line outside the zones, the local surface's VMTK line inside",
             splice={"v5_line": src5, "raw": raw_p, "runs": runs_, "junction_gap_max_mm": max(gaps) if gaps else 0.0,
                     "junction_lateral_max_mm": max(lats) if lats else 0.0,
                     "raw_outside_zones_dev_max_mm": float(dev.max()), "raw_outside_zones_dev_p99_mm": float(np.percentile(dev, 99)),
                     "length_mm": {"v5": float(arclen(P5)[-1]), "raw": float(arclen(Pr)[-1]), "spliced": float(arclen(P)[-1])},
                     "turn_1mm_max_deg": {"v5": turn_1mm_max(P5), "raw": turn_1mm_max(Pr), "spliced": turn_1mm_max(P)},
                     "points_from_v5": int(out5.sum()), "misr_lt_1_mm": {
                         "v5": float(np.r_[np.diff(arclen(P5)), 0][R5_ < 1.0].sum()),
                         "spliced": float(np.r_[np.diff(arclen(P)), 0][R < 1.0].sum())}})
    json.dump(o, open(cur_p + ".tmp", "w"))
    os.replace(cur_p + ".tmp", cur_p)
    return {"status": "spliced", **{k: o["splice"][k] for k in ("junction_gap_max_mm", "junction_lateral_max_mm", "raw_outside_zones_dev_max_mm", "turn_1mm_max_deg",
                                                                 "length_mm", "misr_lt_1_mm")}, "runs": len(runs_)}


def main():
    units = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(u for u in os.listdir(SEG) if not u.startswith("_"))
    allr = {}
    sf = os.path.join(SEG, "_splice_summary.json")
    if os.path.exists(sf):
        allr = json.load(open(sf))
    for u in units:
        if not os.path.exists(os.path.join(SEG, u, ".seg_done")):
            continue
        for sd in ("R", "L"):
            if not os.path.exists(os.path.join(SEG, u, "%s_stageA.vtp" % VP[sd])):
                continue
            try:
                r = splice_side(u, sd)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                r = {"status": "error %r" % e}
            allr["%s:%s" % (u, sd)] = r
            print(u, sd, json.dumps(r), flush=True)
    json.dump(allr, open(sf, "w"), indent=1)


if __name__ == "__main__":
    main()
