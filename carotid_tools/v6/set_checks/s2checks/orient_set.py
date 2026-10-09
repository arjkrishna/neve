#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\orient_set.py (= /sp/v6/s2checks/orient_set.py in the build containers)
"""orient_set.py 1.0 (s2_checks, 2026-10-07): set-level ORIENTATION GATES, FORK HEADINGS and ICA-TERMINUS AP OFFSETS on a
whole anatomy tree (anatomies_v6, and anatomies_v5 for the comparison), with the FROZEN gates/orient_gates.py of the
build manifest (d65b7209), read-only.

Per tree: OG.measure_tree(root, fork_gates=False) (checks_v6 part orient's call; the fork gates are informational, D11),
then checks_v6 part orient's post-processing copied verbatim (D8: va_target 20-30 accepted + flag, > 30 accepted only on
D9 rung 3; D23: pieces listed in provenance v6_build.d23_accepted keep tier A, accepted + flagged).
ICA-terminus AP offset (D11 record) = compose_v6.ica_ap re-implemented on the TREE (the placed piece end and its
back-mapped end from the tree's own curves + xform, measured by OG): target = C_HOST + A0 (q - mid), q = the back-mapped
piece end (D12), mid = the patient's two raw native VA ends (x-mirrored by the piece's recorded mirror); post_mm > 0 =
the terminus lies POSTERIOR of its target; flag > 40 mm. v5 reference count (compose_v6 summary): 108 anatomies
(RICA 66, LICA 59) -- reproduced here as a check of the re-implementation.
    python3 orient_set.py <tag> <root> <out.json>
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/gates")
import orient_gates as OG  # noqa: E402

VERSION = "orient_set 1.0 (s2_checks, 2026-10-07)"
AP_FLAG_MM = 40.0


def post_orient(r, prov):
    """checks_v6 1.5 part_orient post-processing (verbatim logic)."""
    va = (r.get("pairs") or {}).get("va") or {}
    blocking = [list(b) for b in r.get("blocking", [])]
    after, flags = [], []
    for p, f in blocking:
        if f == "va_target":
            side = "R" if p.startswith("R") else "L"
            tm = va.get("tip_move_anat_%s_mm" % side)
            rung = None
            for k in ("va_R" if side == "R" else "va_L",):
                xf = (prov.get("xform") or {}).get(k) or {}
                rung = xf.get("va_rung") or xf.get("rung") or (xf.get("v6") or {}).get("rung")
            if tm is not None and tm <= 30.0:
                flags.append("va_target_20_30:%s" % p)
                continue
            if str(rung) in ("3", "v5_top", "rung3"):
                flags.append("va_target_gt30_rung3:%s" % p)
                continue
        after.append([p, f])
    d23 = ((prov.get("v6_build") or {}).get("d23_accepted") or {})
    acc = set(d23.get("pieces") or [])
    pair_ok = set(d23.get("pair_flags") or [])
    tier_a = list(r.get("tierA_pieces") or [])
    if acc:
        after2 = []
        for p, f in after:
            if p in acc or (f in pair_ok and p.endswith("VA_top")):
                flags.append("D23:%s:%s" % (p, f))
                continue
            after2.append([p, f])
        after = after2
    tier_a_after = [p for p in tier_a if p not in acc]
    return {"tierA_pieces": tier_a, "tierA_pieces_after_D23": tier_a_after, "d23_accepted": sorted(acc & set(tier_a)),
            "v6_blocking": blocking, "v6_blocking_after_D8": after, "accepted_flags": flags}


def ica_ap(m, pid, mirror, tb_root=OG.TB_ROOT):
    fa = "%s/centerlines_rva/%s_va.json" % (tb_root, pid)
    fb = "%s/centerlines_lva/%s_va.json" % (tb_root, pid)
    if not (os.path.exists(fa) and os.path.exists(fb)):
        return {"why": "patient VA ends unknown"}
    mir = np.asarray(mirror, float)
    mid = 0.5 * (OG.load_src(fa)[-1] + OG.load_src(fb)[-1]) * mir
    q, e = np.asarray(m["_Q_end"], float), np.asarray(m["end"], float)
    t = OG.C_HOST + OG.A0 @ (q - mid)
    d = e - t
    return {"post_mm": round(float(d[1]), 2), "err_mm": round(float(np.linalg.norm(d)), 2)}


def main():
    tag, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    res, errs = OG.measure_tree(root, OG.TB_ROOT, None, False)
    per = {}
    C = collections.Counter()
    fork_yaw = collections.defaultdict(list)
    for n, r in sorted(res.items()):
        prov = json.load(open(os.path.join(root, n, "provenance.json"), encoding="utf-8"))
        po = post_orient(r, prov)
        ap = {}
        for p in ("RICA", "LICA"):
            m = r["pieces"].get(p)
            if m and "error" not in m and "_Q_end" in m:
                ap[p] = ica_ap(m, r["patient"], m.get("mirror", [1, 1, 1]))
        forks = {}
        for k, f in (r.get("forks") or {}).items():
            forks[k] = {"yaw_deg": f.get("yaw_deg"), "up_tilt_deg": f.get("up_tilt_deg"), "flags": f.get("flags"),
                        "placement": f.get("placement"), "mirrored": f.get("mirrored"), "contralateral": f.get("contralateral"),
                        "contralateral_effective": f.get("contralateral_effective"), "eca_anterior_mm": f.get("eca_anterior_mm"),
                        "frame": f.get("frame")}
            if f.get("yaw_deg") is not None:
                fork_yaw[k].append(abs(f["yaw_deg"]))
        per[n] = {**po, "classes": r.get("classes"), "primary_pieces": r.get("primary_pieces"), "ica_ap": ap, "forks": forks,
                  "va_pair": {k: ((r.get("pairs") or {}).get("va") or {}).get(k) for k in
                              ("tip_move_anat_R_mm", "tip_move_anat_L_mm", "lumen_gap_mm")}}
        C["anatomies"] += 1
        C["tierA_any"] += bool(po["tierA_pieces"])
        C["tierA_after_D23"] += bool(po["tierA_pieces_after_D23"])
        C["blocking_any"] += bool(po["v6_blocking"])
        C["blocking_after_D8_D23"] += bool(po["v6_blocking_after_D8"])
        C["va_target_20_30_flag"] += sum(1 for x in po["accepted_flags"] if x.startswith("va_target_20_30"))
        C["d23_flags"] += sum(1 for x in po["accepted_flags"] if x.startswith("D23"))
        fl = [(p, (a or {}).get("post_mm")) for p, a in ap.items()]
        C["ica_ap_gt40_any"] += any(v is not None and v > AP_FLAG_MM for _, v in fl)
        C["ica_ap_gt40_RICA"] += any(p == "RICA" and v is not None and v > AP_FLAG_MM for p, v in fl)
        C["ica_ap_gt40_LICA"] += any(p == "LICA" and v is not None and v > AP_FLAG_MM for p, v in fl)
        for k, f in forks.items():
            fl_ = f.get("flags") or []
            C["%s_fork_heading_gt45" % k] += "fork_heading" in fl_
            C["%s_fork_tilt_gt40" % k] += "fork_tilt" in fl_
            C["%s_contralateral_effective" % k] += bool(f.get("contralateral_effective"))
            C["%s_mirrored" % k] += bool(f.get("mirrored"))
            C["%s_eca_anterior" % k] += (f.get("eca_anterior_mm") or 0) > 0
    apR = [v["ica_ap"]["RICA"]["post_mm"] for v in per.values() if (v["ica_ap"].get("RICA") or {}).get("post_mm") is not None]
    apL = [v["ica_ap"]["LICA"]["post_mm"] for v in per.values() if (v["ica_ap"].get("LICA") or {}).get("post_mm") is not None]

    def st(x):
        x = np.asarray(x, float)
        return {"n": int(len(x)), "min": float(x.min()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                "max": float(x.max())} if len(x) else {"n": 0}
    summary = {"version": VERSION, "orient_gates": OG.VERSION, "tag": tag, "root": root, "errors": errs,
               "counts": dict(C), "og_counts": OG.counts(res),
               "ica_ap_post_mm": {"RICA": st(apR), "LICA": st(apL)},
               "fork_abs_yaw_deg": {k: st(v) for k, v in fork_yaw.items()},
               "anatomies": per}
    json.dump(summary, open(out + ".tmp", "w"), indent=0, default=float)
    os.replace(out + ".tmp", out)
    print(json.dumps({"tag": tag, "counts": dict(C), "errors": len(errs), "ica_ap": summary["ica_ap_post_mm"]}, default=float))


if __name__ == "__main__":
    main()
