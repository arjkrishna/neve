#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\check_build_record_v1_0.py (= /sp/v6/records/check_build_record_v1_0.py in the build containers)
"""s2_records: check that BUILD_v6.json equals the per-anatomy provenance on every count (V6_ITEMS (d): "the record
equals the per-anatomy provenance on every count").

Recounts with code separate from build_v6_record.py (dotted-path reads, its own grouping), straight from
anatomies_v6/<name>/{provenance.json, mesh_v6.json, LICENSE_*} and compares every count, every list's membership and the
provenance digest. Also reports, for information, how the record compares with the build's own collection
(results/v6_build.json, s2_build) and the set checks (results/v6_final_summary.json, s2_checks).

    python3 check_build_record.py [--record BUILD_v6.json] [--root anatomies_v6] [--out build_record_check.json]
exit 0 = every count equal; 1 = a difference (listed)."""
import argparse
import collections
import hashlib
import json
import os
import sys

VERSION = "check_build_record 1.0 (s2_records, 2026-10-07)"
ap = argparse.ArgumentParser()
ap.add_argument("--record", default="/sp/v6/records/out/BUILD_v6.json")
ap.add_argument("--root", default="/repo/carotid_data/anatomies_v6")
ap.add_argument("--collection", default="/sp/v6/results/v6_build.json")
ap.add_argument("--final", default="/sp/v6/results/v6_final_summary.json")
ap.add_argument("--out", default="/sp/v6/records/out/build_record_check.json")
a = ap.parse_args()


def load(p):
    with open(p) as f:
        return json.load(f)


def get(o, path, default=None):
    for k in path.split("."):
        if not isinstance(o, dict) or k not in o:
            return default
        o = o[k]
    return o


def tally(values):
    c = collections.Counter(str(v) for v in values)
    return dict(sorted(c.items()))


B = load(a.record)
names = sorted(x for x in os.listdir(a.root) if x.startswith("case_"))
P = {n: load(os.path.join(a.root, n, "provenance.json")) for n in names}
M = {n: load(os.path.join(a.root, n, "mesh_v6.json")) for n in names}
F = {n: set(os.listdir(os.path.join(a.root, n))) for n in names}
sh = lambda n: n[5:].replace("__topcow_", " ")
res = []


def check(label, rec, recount):
    ok = json.dumps(rec, sort_keys=True, default=str) == json.dumps(recount, sort_keys=True, default=str)
    res.append({"check": label, "ok": ok, **({} if ok else {"record": rec, "recount": recount})})


def stringify_keys(d):
    return {str(k): v for k, v in (d or {}).items()}


# --- identity and digest
lines = ""
for n in names:
    h = hashlib.sha256(open(os.path.join(a.root, n, "provenance.json"), "rb").read()).hexdigest()
    lines += "%s %s\n" % (n, h)
check("provenance_digest", B["provenance_digest"]["sha256"], hashlib.sha256(lines.encode()).hexdigest())
check("anatomies.v6", B["anatomies"]["v6"], len(names))
check("anatomies.by_handedness", B["anatomies"]["by_handedness"], tally(p.get("handedness") for p in P.values()))
check("per_anatomy.names", sorted(B["per_anatomy"]), names)

# --- right / left fork
rfp = lambda p: get(p, "v6_build.carotid.right_fork")
lfp = lambda p: get(p, "v6_build.carotid.left_fork")
check("carotid.right_fork.rung", stringify_keys(B["carotid"]["right_fork"]["rung"]), tally(rfp(p)["right_fork_rung"] for p in P.values()))
check("carotid.right_fork.pose", B["carotid"]["right_fork"]["pose"], tally(rfp(p)["right_fork_pose"] for p in P.values()))
check("carotid.right_fork.placement", B["carotid"]["right_fork"]["placement"], tally(rfp(p)["right_fork_placement"] for p in P.values()))
check("carotid.right_fork.mirrored_D20", B["carotid"]["right_fork"]["mirrored_D20"], sum(1 for p in P.values() if rfp(p)["right_fork_mirrored"] is True))
check("carotid.right_fork.host_cut_shifted", B["carotid"]["right_fork"]["host_cut_shifted"], sum(1 for p in P.values() if rfp(p).get("host_cut_shift_mm")))
check("carotid.right_fork.seam1_bend_gt45 (names)", sorted(x[0] for x in B["carotid"]["right_fork"]["seam1_bend_gt45"]),
      sorted(sh(n) for n, p in P.items() if (rfp(p).get("right_seam1_bend_deg") or 0) > 45))
for r in (3, 4, 5):
    key = {3: "R3_bend_up_to_60", 4: "R4_bounded_deviation", 5: "R5_v5_pose"}[r]
    check("carotid.right_fork.%s (names)" % key, sorted(x[0] for x in B["carotid"]["right_fork"][key]),
          sorted(sh(n) for n, p in P.items() if rfp(p)["right_fork_rung"] == r))
check("carotid.left_fork.decision", B["carotid"]["left_fork"]["decision"], tally(lfp(p)["left_fork_decision"] for p in P.values()))
check("carotid.left_fork.rung", stringify_keys(B["carotid"]["left_fork"]["rung"]), tally(lfp(p)["fork_rung"] for p in P.values()))
check("carotid.left_fork.pose", B["carotid"]["left_fork"]["pose"], tally(lfp(p)["left_fork_pose"] for p in P.values()))
check("carotid.left_fork.mirrored_D20", B["carotid"]["left_fork"]["mirrored_D20"], sum(1 for p in P.values() if lfp(p)["left_fork_mirrored"] is True))
check("carotid.left_fork.ica_tubed", B["carotid"]["left_fork"]["ica_tubed_D1_rung5_6"], sum(1 for p in P.values() if (lfp(p).get("tubed_ica_mm") or 0) > 0))
check("carotid.left_fork.substituted (names)", sorted(x[0] for x in B["carotid"]["left_fork"]["substituted"]),
      sorted(sh(n) for n, p in P.items() if lfp(p)["left_fork_decision"] == "substituted"))
check("carotid.left_fork.not_on_map (names)", sorted(x[0] for x in B["carotid"]["left_fork"]["not_on_map"]),
      sorted(sh(n) for n, p in P.items() if lfp(p)["left_fork_pose"] != "zenodo_map"))
check("carotid.ica_ap gt40 R/L", [B["carotid"]["ica_terminus_ap_mm"]["gt40_R"], B["carotid"]["ica_terminus_ap_mm"]["gt40_L"]],
      [sum(bool(get(p, "v6_build.carotid.ica_ap_flag_R")) for p in P.values()), sum(bool(get(p, "v6_build.carotid.ica_ap_flag_L")) for p in P.values())])
check("carotid.eca.tip_skipped", B["carotid"]["eca"]["tip_skipped"],
      tally(t["side"] for p in P.values() for t in (get(p, "v6_build.carotid.skipped_eca_tips") or [])))
check("carotid.eca.right_mode", B["carotid"]["eca"]["right_mode"], tally(get(p, "v6_build.carotid.eca_right") for p in P.values()))
check("carotid.route_baseline_D13", B["carotid"]["route_baseline_D13"], tally(get(p, "v6_build.carotid.route_baseline") for p in P.values()))
check("carotid.route13_moved (names)", [x[0] for x in B["carotid"]["route13_moved"]], [sh(n) for n, p in P.items() if get(p, "v6_build.route13")])

# --- VA
for side in ("right", "left"):
    vp = lambda p, s=side: get(p, "v6_build.va." + s)
    check("va.rung.%s" % side, B["va"]["rung"][side], tally(vp(p)["rung"] for p in P.values()))
    check("va.junction.%s" % side, B["va"]["junction"][side], tally(vp(p)["junction_type"] for p in P.values()))
    check("va.flags.%s" % side, B["va"]["flags"][side], tally(f for p in P.values() for f in vp(p).get("flags") or []))
    check("va.va_target_band.%s" % side, B["va"]["va_target_band"][side], tally(vp(p).get("va_target_band") for p in P.values()))
    check("donors.va_donor_sides.%s" % side, B["donors"]["va_donor_sides"][side], sum(1 for p in P.values() if vp(p).get("donor_id")))
check("va.host_no_donor_D21", sorted(map(tuple, B["va"]["host_no_donor_D21"])),
      sorted((sh(n), s) for n, p in P.items() for s in ("right", "left") if not get(p, "v6_build.va.%s.donor_id" % s)))
check("va.d18_v5_junction (names, sides)", sorted((x[0], x[1]) for x in B["va"]["d18_v5_junction"]),
      sorted((sh(n), s) for n, p in P.items() for s in ("right", "left") if get(p, "v6_build.va.%s.rung" % s) == 2))
check("va.d9_rung3 (names, sides)", sorted((x[0], x[1]) for x in B["va"]["d9_rung3_v5_top"]),
      sorted((sh(n), s) for n, p in P.items() for s in ("right", "left") if get(p, "v6_build.va.%s.rung" % s) == 3))
check("va.d16_tube (names, sides)", sorted((x[0], x[1]) for x in B["va"]["d16_tube"]),
      sorted((sh(n), s) for n, p in P.items() for s in ("right", "left") if get(p, "v6_build.va.%s.d16_tube_mm" % s)))
check("va.d17_tube (names, sides)", sorted((x[0], x[1]) for x in B["va"]["d17_tube"]),
      sorted((sh(n), s) for n, p in P.items() for s in ("right", "left") if "d17_tube" in (get(p, "v6_build.va.%s.flags" % s) or [])))

# --- donors: one-to-one, per unit
unit_pat = collections.defaultdict(set)
for p in P.values():
    for u in get(p, "v5_va.assignment.assigned_units") or []:
        unit_pat[u].add(p["patient"])
check("donors.units", B["donors"]["units"], len(unit_pat))
check("donors.one_to_one.ok", B["donors"]["one_to_one"]["ok"],
      all(len(s) == 1 for s in unit_pat.values()) and len({next(iter(s)) for s in unit_pat.values()}) == len(unit_pat))
for row in B["donors"]["table"]:
    u = row["unit"]
    sides = [(s, get(p, "v6_build.va.%s.rung" % s)) for p in P.values() for s in ("right", "left") if get(p, "v6_build.va.%s.donor_id" % s) == u]
    rc = {"anatomies": sum(1 for p in P.values() if u in (get(p, "v5_va.assignment.assigned_units") or [])),
          "rva_donor_sides": sum(1 for s, _ in sides if s == "right"), "lva_donor_sides": sum(1 for s, _ in sides if s == "left"),
          "rung1_own_junction": sum(1 for _, r in sides if r == 1), "rung2_v5_junction": sum(1 for _, r in sides if r == 2),
          "rung3_v5_top": sum(1 for _, r in sides if r == 3)}
    check("donors.table.%s" % u, {k: row[k] for k in rc}, rc)

# --- licences
check("licences.v6_file", B["licences"]["anatomies_with_LICENSE_v6_donors_txt"], sum("LICENSE_v6_donors.txt" in f for f in F.values()))
check("licences.vmr_file", B["licences"]["anatomies_with_LICENSE_VMR_0248_txt"], sum("LICENSE_VMR_0248.txt" in f for f in F.values()))
check("licences.non_commercial", B["licences"]["non_commercial_anatomies"], sum(bool(get(p, "v5_va.summary.non_commercial")) for p in P.values()))

# --- bake / mesh
check("bake.rung", B["bake"]["rung"], tally(get(p, "bake_v6.rung") for p in P.values()))
check("bake.rung == mesh_v6 ladder rung", B["bake"]["rung"], tally(get(m, "ladder_v5b.rung") for m in M.values()))
check("bake.pass", B["bake"]["pass"], sum(1 for n in names if get(P[n], "bake_v6.pass") and M[n].get("pass")))
check("bake.carotid_margin_mm", B["bake"]["carotid_margin_mm"], tally(get(p, "bake_v6.carotid_margin_mm") for p in P.values()))
check("bake.va_margin_mm", B["bake"]["va_margin_mm"], tally(get(p, "bake_v6.va_margin_mm") for p in P.values()))
check("bake.flags", B["bake"]["flags"], tally(f for p in P.values() for f in get(p, "bake_v6.flags") or []))
check("bake.anatomies_with_tube_windows", B["bake"]["anatomies_with_tube_windows"],
      sum(1 for p in P.values() if any(v.get("n_windows") for v in (get(p, "bake_v6.va_tubed") or {}).values())))
check("bake.window_sections", B["bake"]["window_sections"],
      tally(k for p in P.values() for k, v in (get(p, "bake_v6.va_tubed") or {}).items() if v.get("n_windows")))
check("bake.c4 applied (names)", sorted(B["bake"]["c4_mr021_lva_pinch_applied"]),
      sorted(sh(n) for n, p in P.items() if get(p, "bake_v6.c4.applied")))
check("bake.e7", B["bake"]["e7_va_aware_obj"], tally(get(p, "bake_v6.e7.result") for p in P.values()))
check("bake.far_clamp (names)", sorted(x[0] for x in B["bake"]["far_clamp"]), sorted(sh(n) for n, p in P.items() if get(p, "bake_v6.far_clamp.used")))
check("bake.sections_tube_only", sorted(map(tuple, B["bake"]["sections_tube_only_list"])),
      sorted((sh(n), s) for n, p in P.items() for s in get(p, "bake_v6.sections_tube_only") or []))
check("bake.obj_genus0_closed", B["bake"]["obj_genus0_closed"],
      sum(1 for m in M.values() if all(g["genus"] == 0 and g["boundary_loops"] == 0 for g in m["obj"]["genus"])))
check("bake.centreline_pts_outside_total", B["bake"]["centreline_pts_outside_total"], sum(int(m.get("centerline_pts_outside") or 0) for m in M.values()))
check("bake.obj_tris range", [B["bake"]["obj_tris"]["min"], B["bake"]["obj_tris"]["max"]],
      [float(min(m["obj"]["tris"] for m in M.values())), float(max(m["obj"]["tris"] for m in M.values()))])

# --- checks (v6_checks stamped by the build)
check("checks.pass", B["checks"]["pass"], sum(bool(get(p, "v6_checks.pass")) for p in P.values()))
gates = collections.Counter()
for p in P.values():
    for g, v in (get(p, "v6_checks.gates") or {}).items():
        gates[g] += bool(v)
check("checks.gates", B["checks"]["gates"], dict(sorted(gates.items())))
for g in ("fork", "siphon"):
    check("checks.d22_reference.%s" % g, B["checks"]["d22_reference"][g], tally(get(p, "v6_checks.d22.%s.reference" % g) for p in P.values()))
check("checks.d26 (names, pieces)", sorted((x[0], x[1]) for x in B["checks"]["d26_own_surface"]),
      sorted((sh(n), g) for n, p in P.items() for g in ("fork", "siphon") if get(p, "v6_checks.d22.%s.d26_flag" % g)))
check("checks.d27 (names, pieces)", sorted((x[0], x[1]) for x in B["checks"]["d27_repair_window"]),
      sorted((sh(n), g) for n, p in P.items() for g in ("fork", "siphon") if get(p, "v6_checks.d22.%s.d27_flag" % g)))
check("checks.coverage_lt_0.99 (names, pieces)", sorted((x[0], x[1]) for x in B["checks"]["coverage_lt_0.99_reported_not_gated"]),
      sorted((sh(n), g) for n, p in P.items() for g in ("fork", "siphon") if (get(p, "v6_checks.d22.%s.coverage" % g) or 1.0) < 0.99))
check("checks.d13_baseline", B["checks"]["d13_baseline"], tally(get(p, "v6_checks.d22_baseline") for p in P.values()))

# --- flags as written
fc = collections.Counter()
for p in P.values():
    for f in get(p, "v6_build.flags") or []:
        fc[":".join(f.split(":")[:2]) if f.split(":")[0] in ("va_right", "va_left") else f] += 1
check("flags.compose", B["flags"]["compose_v6_build_flags"], dict(sorted(fc.items())))
kc = collections.Counter()
for p in P.values():
    for f in get(p, "v6_checks.flags") or []:
        kc[f[:f.index("_repair_window") + len("_repair_window")] if "_repair_window" in f else f] += 1
check("flags.checks", B["flags"]["checks_v6_checks_flags"], dict(sorted(kc.items())))
check("flags.bake", B["flags"]["bake_flags"], tally(f for p in P.values() for f in get(p, "bake_v6.flags") or []))

# --- toolchain recorded in every anatomy
check("toolchain.compose_manifest_md5", B["toolchain"]["compose_manifest_md5"], tally(get(p, "v6_build.toolchain.manifest_md5") for p in P.values()))
check("toolchain.checks_manifest_md5", B["toolchain"]["checks_manifest_md5"], tally(get(p, "v6_checks.toolchain_manifest_md5") for p in P.values()))

n_bad = sum(1 for r in res if not r["ok"])

# --- information: the record against the build collection (s2_build) and the set checks (s2_checks)
info = []
col = load(a.collection)
cc = col.get("counts") or {}
for k, v in sorted(B["flags"]["compose_v6_build_flags"].items()):
    info.append(["flag:" + k, v, cc.get("flag:" + k)])
for k, v in sorted(B["flags"]["checks_v6_checks_flags"].items()):
    alt = {"d23_tierA_accepted": "flag:D23_tierA_accepted:RVA_top"}.get(k, "flag:" + k)
    info.append([k, v, cc.get(alt)])
for g in ("fork", "siphon"):
    for r, v in B["checks"]["d22_reference"][g].items():
        info.append(["d22_%s_reference:%s" % (g, r), v, cc.get("d22_%s_reference:%s" % (g, r))])
for r, v in B["bake"]["rung"].items():
    info.append(["bake_rung:" + r, v, cc.get("bake_rung:" + r)])
info.append(["checks_pass", B["checks"]["pass"], cc.get("checks_pass")])
info.append(["left_pose_map_bounded_dev", len(B["carotid"]["left_fork"]["map_bounded_deviation"]), cc.get("flag:left_pose_map_bounded_dev")])
info.append(["orient va_target 20-30 (band 20-30, both sides)", B["va"]["va_target_band"]["right"].get("20-30", 0) + B["va"]["va_target_band"]["left"].get("20-30", 0), cc.get("flag:orient:va_target_20_30")])
fin = load(a.final)
info.append(["final.d26 list", sorted({x[0] for x in B["checks"]["d26_own_surface"]}), sorted(get(fin, "right_route.d26") or [])])
info.append(["final.d27 list", sorted({x[0] for x in B["checks"]["d27_repair_window"]}), sorted(get(fin, "right_route.d27") or [])])
info.append(["final.reference", B["checks"]["d22_reference"], get(fin, "right_route.reference")])
info.append(["final.ica_ap_gt40_R", B["carotid"]["ica_terminus_ap_mm"]["gt40_R"], get(fin, "ica_ap_gt40.v6.ica_ap_gt40_RICA")])
EXPLAIN = {"orient va_target 20-30 (band 20-30, both sides)":
           "the collection's orient:va_target_20_30 (11) counts the VA flag va_target_20_30 (right 9 + left 2, also in this "
           "record under va.flags); one of the 9 (case_w_022_right__topcow_mr_026 RVA) has va_target_mm 19.999, so its "
           "recorded band is '<=20' while the post-bake measure flagged it: a boundary case, both values in provenance"}
info_rows = [{"what": w, "record": r, "other": o, "equal": json.dumps(r, sort_keys=True) == json.dumps(o, sort_keys=True),
              **({"why": EXPLAIN[w]} if w in EXPLAIN else {})} for w, r, o in info]
out = {"version": VERSION, "record": a.record, "root": a.root, "n_anatomies": len(names), "checks": len(res), "differences": n_bad,
       "verdict": "BUILD_v6.json == provenance on every count" if n_bad == 0 else "DIFFERENCES", "results": res,
       "information_vs_build_collection_and_set_checks": {"rows": info_rows, "not_equal": [r for r in info_rows if not r["equal"]]}}
with open(a.out, "w") as f:
    json.dump(out, f, indent=1, default=str)
print(json.dumps({"checks": len(res), "differences": n_bad, "info_not_equal": [r["what"] for r in info_rows if not r["equal"]]}))
sys.exit(0 if n_bad == 0 else 1)
