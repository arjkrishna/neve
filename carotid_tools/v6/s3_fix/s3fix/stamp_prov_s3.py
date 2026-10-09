#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\stamp_prov_s3.py (= /sp/v6/s3fix/stamp_prov_s3.py in the build containers)
"""stamp_prov 1.1 (s3_fix, 2026-10-08) -- the final provenance of each anatomy CHANGED by s3_fix = its final tree's
provenance.json + ONE key "v6_checks" (as stamp_prov 1.0 for s2_build), now from checks_v6 1.6 records:
  * everything stamp_prov 1.0 wrote (verdict, part versions, gates, D22 per piece, D26 / D27 / D22-coverage / D23 flags);
  * coverage against the v4 / v5 SHIPPED wall per piece (checks_v6 1.6 part coverage: deficit per reference, lost area,
    ok, flag) and fold-over counts (part folds);
  * derived_flags: the flags v6_build_collect derives outside provenance (orient:<accepted orientation flag> from the
    orient part, left_pose_map_bounded_dev from the left fork's map deviation) -- written here for the stamped anatomies;
  * s3_fix flags (D27, USER rule 2026-10-07): d27_coverage_<piece>_real_wall_loss_<p>pct_accepted (0.1 % <= deficit
    < 1 %), d27_bake_lattice_shift_<sx>_<sy>_<sz> (bake_v6 1.2 L1), d27_bake_route_island_bridge_<n> (bake_v6 1.2 L2),
    obj_fold_fix_<method> (fold_fix 1.0), and the block s3_fix (what changed and why).
Asserted per anatomy: the stamped dict minus v6_checks == the source provenance; the source provenance differs from
full/tree3's only in the keys the s3_fix step owns (re-bake: bake_v6; fold fix: bake_v6.obj_fold_fix / obj_tris; record
only: nothing).
    python3 stamp_prov_s3.py --items items.tsv --out DIR [--tree3 /full/tree3] [--manifest MAN]
items.tsv: <name> <kind: rebake|fold|record> <final tree dir> <checks record json> [<checks4 done json>]
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

VERSION = "stamp_prov 1.1 (s3_fix, 2026-10-08)"
ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tree3", default="/full/tree3")
ap.add_argument("--manifest", default="/sp/v6/build/TOOLCHAIN_s3fix.md5")
a = ap.parse_args()
RZ = np.diag([-1.0, -1.0, 1.0])


def jl(p):
    with open(p) as f:
        return json.load(f)


def md5b(b):
    return hashlib.md5(b).hexdigest()


def map_dev(p):
    """= build_v6_record's rule (provenance only): xform.lower_L.R against the per-patient Zenodo map recorded in
    v6_carotid.left_fork.map.frame (I for LPS-like, Rz(180) for RAS-like); deg."""
    fr = (((p.get("v6_carotid") or {}).get("left_fork") or {}).get("map") or {}).get("frame")
    xl = (p.get("xform") or {}).get("lower_L") or {}
    if fr not in ("LPS", "RAS") or "R" not in xl:
        return None
    A0 = np.eye(3) if fr == "LPS" else RZ
    R = np.asarray(xl["R"], float)
    return round(float(np.degrees(np.arccos(np.clip((np.trace(R @ A0.T) - 1) / 2, -1, 1)))), 3)


man = md5b(open(a.manifest, "rb").read()) if os.path.exists(a.manifest) else None
os.makedirs(a.out, exist_ok=True)
res = {}
for line in open(a.items):
    f = line.split()
    if not f or f[0].startswith("#"):
        continue
    n, kind, tree, ckp = f[:4]
    src = os.path.join(tree, n, "provenance.json")
    raw = open(src, "rb").read()
    p = json.loads(raw)
    p3 = jl(os.path.join(a.tree3, n, "provenance.json"))
    # the source differs from tree3 only in what this step owns
    if kind == "rebake":
        own = {"bake_v6"}
    elif kind == "fold":
        own = set()
        b6, b3 = dict(p["bake_v6"]), dict(p3["bake_v6"])
        assert "obj_fold_fix" in b6, n
        b6.pop("obj_fold_fix"); b6.pop("obj_tris", None); b3.pop("obj_tris", None)
        assert b6 == b3, n + ": bake_v6 differs beyond obj_fold_fix / obj_tris"
        own = {"bake_v6"}
    else:
        own = set()
    q3 = {k: v for k, v in p3.items() if k not in own}
    qs = {k: v for k, v in p.items() if k not in own}
    assert q3 == qs, "%s: provenance differs from tree3 outside %s" % (n, sorted(own))
    rec = jl(ckp)
    d22 = (rec.get("route") or {}).get("d22") or {}
    flags, pieces = [], {}
    for g in ("fork", "siphon"):
        x = d22.get(g) or {}
        rw = x.get("d22_repair_window") or {}
        pieces[g] = {"reference": x.get("d22_reference"), "value": x.get("value"), "value_v4": x.get("value_v4"),
                     "value_own": x.get("value_own"), "value_before_d27": x.get("value_before_d27"),
                     "no_wall_lost": x.get("no_wall_lost"), "coverage": (x.get("coverage") or {}).get("within_0.3"),
                     "d26_flag": bool(x.get("d26_flag")), "d27_flag": bool(x.get("d27_flag"))}
        if x.get("d26_flag"):
            flags.append("d26_d22_%s_reference_own_surface" % g)
        if x.get("d27_flag"):
            pieces[g]["d22_repair_window"] = {k: rw.get(k) for k in ("cells", "area_mm2", "s_range_mm", "lumen_beyond_capsule_mm",
                                                                     "share_of_off_both_area", "clipped_in_v4_world_frac")}
            flags.append("d27_d22_%s_repair_window_%.2f_mm2" % (g, float(rw.get("area_mm2") or 0)))
        if x.get("coverage_flag"):
            flags.append("d22_%s_coverage_lt_0.99" % g)
    orr = rec.get("orient") or {}
    if orr.get("d23_accepted"):
        flags.append("d23_tierA_accepted")
    # 1.1: coverage against the v4 / v5 shipped wall (checks_v6 1.6) and folds
    cv = rec.get("coverage") or {}
    cov = {}
    for g in ("fork", "siphon"):
        c = cv.get(g) or {}
        cov[g] = {"deficit_v4": (cv.get("v4", {}).get(g) or {}).get("deficit"),
                  "deficit_v5": (cv.get("v5", {}).get(g) or {}).get("deficit"),
                  "lost_mm2_max": c.get("lost_mm2_max"), "ok": c.get("ok"), "flag": c.get("flag")}
        if c.get("flag"):
            flags.append("d27_coverage_%s_real_wall_loss_%.2fpct_accepted" % (g, 100.0 * float(c.get("deficit_max") or 0)))
    fo = rec.get("folds") or {}
    folds = {k: (fo.get(k) or {}).get("fold_pairs") for k in ("obj", "full")}
    bk = p.get("bake_v6") or {}
    sh = [float(v) for v in (bk.get("lattice_shift_mm") or [0, 0, 0])]
    if any(sh):
        flags.append("d27_bake_lattice_shift_%s" % "_".join(("%g" % v) for v in sh))
    rib = bk.get("route_island_bridge") or {}
    if rib.get("bridged"):
        flags.append("d27_bake_route_island_bridge_%d" % len(rib["bridged"]))
    if bk.get("obj_fold_fix"):
        flags.append("obj_fold_fix_%s" % str(bk["obj_fold_fix"].get("result", "")).split(" (")[0].replace(" ", "_"))
    derived = ["orient:%s" % x for x in (orr.get("accepted_flags") or [])]
    ldev = map_dev(p)
    lpose = ((((p.get("v6_build") or {}).get("carotid") or {}).get("left_fork") or {}).get("left_fork_pose"))
    if lpose == "zenodo_map" and ldev is not None and ldev >= 0.01:
        derived.append("left_pose_map_bounded_dev")
    pv_ = dict(rec.get("part_versions") or {})
    blk = {"version": VERSION, "date": time.strftime("%F %T"), "checks_v6": str(rec.get("script") or "").split(" (")[0],
           "part_versions": pv_, "pass": bool(rec.get("pass")), "gates": rec.get("gates"),
           "d22_baseline": d22.get("baseline"), "d22": pieces, "coverage_v4_v5_shipped": cov, "folds": folds,
           "carried_from_source": sorted(p_ for p_, v_ in pv_.items() if not str(v_).startswith("checks_v6 1.6")),
           "flags": flags, "derived_flags": derived,
           "s3_fix": {"kind": kind, "why": {"rebake": "final verifier must-fix 1: real scanned wall lost >= 1 % vs v4 / v5 "
                                                      "shipped wall; D27 re-bake (bake_v6 1.2)",
                                            "fold": "final verifier should-fix 3: fold-over pair on the training OBJ (fold_fix 1.0)",
                                            "record": "D27: real-wall deficit 0.1-1 % accepted with a flag (record only)"}[kind]},
           "toolchain_manifest_md5": man,
           "note": "check-level flags; compose flags are in v6_build.flags; derived_flags = flags v6_build_collect derives "
                   "outside provenance. Full records: v6_work/s3fix/checks/final/<n>.json"}
    q = dict(p)
    q.pop("v6_checks", None)
    q["v6_checks"] = blk
    chk = dict(q)
    chk.pop("v6_checks")
    pp = dict(p)
    pp.pop("v6_checks", None)
    assert chk == pp, n
    out = json.dumps(q, indent=1, default=str).encode()
    os.makedirs(os.path.join(a.out, n), exist_ok=True)
    dst = os.path.join(a.out, n, "provenance.json")
    with open(dst + ".tmp", "wb") as fh:
        fh.write(out)
    os.replace(dst + ".tmp", dst)
    res[n] = {"kind": kind, "src_md5": md5b(raw), "stamped_md5": md5b(out), "pass": blk["pass"], "flags": flags,
              "derived_flags": derived}
summ = {"version": VERSION, "n": len(res), "pass": sum(1 for r in res.values() if r["pass"]),
        "not_pass": [n for n, r in res.items() if not r["pass"]], "per_anatomy": res}
with open(os.path.join(a.out, "stamp_prov_s3.json"), "w") as fh:
    json.dump(summ, fh, indent=1)
print(json.dumps({k: summ[k] for k in ("n", "pass", "not_pass")}))
