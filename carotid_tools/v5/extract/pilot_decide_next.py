#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_decide_next.py (= /sp/v5_va/pilot_decide_next.py in the build containers)
"""pilot_decide_next.py -- scale-up: pair yield per processed case with honest intervals, cases still needed for
need + 5 spares, the ordered selection list of the next cases, and the per-case cost with TotalSegmentator --ml.

Read-only on every input (python:3 container). Inputs: results/pilot_decide_tables.json (pilot_decide_collect.py),
results/pilot_fetch_song_table.json (143 Song cases), results/pilot_ia172_candidates.json (172 series + duplicate
screen), results/pilot_proc_{song,ia172}_summary.json (timing_s), Song / ia172 fetch records.
Output: results/pilot_next_cases.json; prints a summary.
"""
import json
import math

SP = "/sp/v5_va"
RES = SP + "/results"
NEED_TOTAL = 25
SPARES = 5
HOST_DENSE = 0.027


def L(p):
    return json.load(open(p))


def wilson(x, n, z=1.96):
    if n == 0:
        return [None, None]
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def bb_pmf(m, x, n):
    """beta-binomial predictive for m new cases, Jeffreys posterior Beta(x + .5, n - x + .5)"""
    a, b = x + 0.5, n - x + 0.5
    out = []
    for k in range(m + 1):
        lc = math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
        out.append(math.exp(lc + lbeta(k + a, m - k + b) - lbeta(a, b)))
    return out


def conv(p, q):
    r = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def plan_cases(need, rates, pools, order=("song", "ia172"), qs=(0.5, 0.8, 0.9), mmax=400):
    """smallest total m (alternating sources 1:1 in `order`, spilling to the other source when a pool runs out)
    with P(pairs >= need) >= q; also the central (expected-value) m."""
    res = {}
    for m in range(0, mmax + 1):
        ms = {s: 0 for s in order}
        k = 0
        for i in range(m):
            s = order[i % 2]
            if ms[s] >= pools[s]:
                s = order[(i + 1) % 2]
            if ms[s] >= pools[s]:
                break
            ms[s] += 1
            k += 1
        if k < m:
            break
        dist = [1.0]
        for s in order:
            dist = conv(dist, bb_pmf(ms[s], *rates[s]))
        pge = sum(dist[need:]) if need < len(dist) else 0.0
        exp = sum(ms[s] * rates[s][0] / rates[s][1] for s in order)
        if "central" not in res and exp >= need:
            res["central"] = {"m": m, "split": dict(ms), "expected_pairs": round(exp, 1), "P_reach": round(pge, 3)}
        for q in qs:
            if "P%d" % int(q * 100) not in res and pge >= q:
                res["P%d" % int(q * 100)] = {"m": m, "split": dict(ms), "expected_pairs": round(exp, 1), "P_reach": round(pge, 3)}
        if all(("P%d" % int(q * 100)) in res for q in qs) and "central" in res:
            break
    res["pool_cap"] = sum(pools.values())
    for q in qs:
        res.setdefault("P%d" % int(q * 100), "not reached within the pools (%d cases)" % sum(pools.values()))
    res.setdefault("central", "not reached within the pools")
    return res


def main():
    T = L(RES + "/pilot_decide_tables.json")
    sides = T["sides"]

    # ---- per-case pair outcomes (ia172, song; AortaSeg exhausted)
    def pair_ok(src, case, key):
        ss = [s for s in sides if s["source"] == src and s["case"] == case]
        return len(ss) == 2 and all(s["qc"][key] for s in ss)

    def coarse_ok(src, case):
        ss = [s for s in sides if s["source"] == src and s["case"] == case]
        return pair_ok(src, case, "recal_pass") and all((s.get("donor_stretch") or {}).get("u03_contact_free") is not None and
                                                        s["donor_stretch"]["u03_contact_free"] >= HOST_DENSE for s in ss)
    cases = {src: sorted(set(s["case"] for s in sides if s["source"] == src)) for src in ("ia172", "song")}
    rates = {}
    for key, fn in (("recalibrated", lambda s, c: pair_ok(s, c, "recal_pass")),
                    ("backed_only", lambda s, c: pair_ok(s, c, "backed_only_pass")),
                    ("draft", lambda s, c: pair_ok(s, c, "draft_pass")),
                    ("coarse_enough_recal", coarse_ok)):
        rates[key] = {}
        tx = tn = 0
        for src in ("ia172", "song"):
            x = sum(bool(fn(src, c)) for c in cases[src])
            n = len(cases[src])
            tx += x
            tn += n
            rates[key][src] = {"pairs": x, "cases": n, "rate": round(x / n, 3), "wilson95": wilson(x, n),
                               "which": [c for c in cases[src] if fn(src, c)]}
        rates[key]["combined"] = {"pairs": tx, "cases": tn, "rate": round(tx / tn, 3), "wilson95": wilson(tx, tn)}

    # ---- age (exploratory) on pilot cases
    st = L(RES + "/pilot_fetch_song_table.json")
    cand = L(RES + "/pilot_ia172_candidates.json")
    C = cand["candidates"]

    def age_of(src, c):
        return st[c]["age"] if src == "song" else C[c.replace("ia172_", "")]["age"]
    age_tab = {}
    for band, f in (("<75", lambda a: a < 75), (">=75", lambda a: a >= 75), ("<60", lambda a: a < 60), (">=60", lambda a: a >= 60)):
        xs = [(src, c) for src in ("ia172", "song") for c in cases[src] if f(age_of(src, c))]
        age_tab[band] = {"cases": len(xs), "pairs_recal": sum(pair_ok(s, c, "recal_pass") for s, c in xs),
                         "which": ["%s(%d)" % (c, age_of(s, c)) for s, c in xs]}

    # ---- need
    vmr = 1   # VMR 0248 pair (eligible)
    need_target = NEED_TOTAL + SPARES
    have = {"recalibrated": rates["recalibrated"]["combined"]["pairs"] + vmr,
            "backed_only": rates["backed_only"]["combined"]["pairs"] + vmr,
            "draft": rates["draft"]["combined"]["pairs"] + vmr,
            "coarse_enough_recal": rates["coarse_enough_recal"]["combined"]["pairs"] + vmr}

    # ---- pools
    processed_song = set(cases["song"])
    song_pool = sorted([k for k, v in st.items() if v.get("ndim") == 3 and (v.get("z_mm") or 0) >= 300
                        and k not in processed_song and k not in ("AHMU1218077", "AHMU1218097")])
    # Song duplicate screen: identical sex, age, grid shape and voxel size (one CTA per subject is expected)
    sig = {}
    for k, v in st.items():
        sig.setdefault((v.get("sex"), v.get("age"), tuple(v.get("shape") or []), tuple(round(z, 4) for z in (v.get("zooms") or []))), []).append(k)
    song_dups = [ks for ks in sig.values() if len(ks) > 1]
    song_same_demo = {}
    for k, v in st.items():
        song_same_demo.setdefault((v.get("group"), v.get("sex"), v.get("age"), v.get("slice_class")), []).append(k)
    song_same_demo = [ks for ks in song_same_demo.values() if len(ks) > 1]
    processed_ia = set(c.replace("ia172_", "") for c in cases["ia172"]) | {"166"}
    dup_pairs = cand["duplicate_screen"]["same_demographics_scanner_table"]
    POST = ("BA", "PCA", "VA", "PICA", "SCA", "AICA")
    ia_rows = []
    dropped = {"posterior_aneurysm": [], "duplicate_suspect_second_member": [], "too_thick": [], "not_HN_or_short": []}
    # same-demographics pairs -> connected components; keep at most ONE member per component: none if a member was
    # already processed, else the first member that is otherwise eligible
    comp = {}
    for a, b in dup_pairs:
        ca, cb = comp.get(a, {a}), comp.get(b, {b})
        u = ca | cb
        for m in u:
            comp[m] = u

    def eligible_basic(sid):
        x = C[sid]
        arts = [a.split(":")[0] for a in (x.get("arteries") or [])] + [x.get("artery") or ""]
        return (x.get("head_neck") == "1" and (x.get("z") or 0) >= 300 and (x.get("thick") or 9) <= 0.75 and
                not any(any(p in a for p in POST) for a in arts if a and a != "NA"))
    skip_second = {}
    for u in {frozenset(v) for v in comp.values()}:
        mem = sorted(u)
        if any(m in processed_ia for m in mem):
            keeper = [m for m in mem if m in processed_ia][0]
        else:
            el = [m for m in mem if eligible_basic(m)]
            keeper = el[0] if el else None
        for m in mem:
            if m != keeper and m not in processed_ia:
                skip_second[m] = keeper or "no eligible member"
    for sid, x in sorted(C.items()):
        if sid in processed_ia:
            continue
        if x.get("head_neck") != "1" or (x.get("z") or 0) < 300:
            dropped["not_HN_or_short"].append(sid)
            continue
        if (x.get("thick") or 9) > 0.75:
            dropped["too_thick"].append(sid)
            continue
        arts = [a.split(":")[0] for a in (x.get("arteries") or [])] + [x.get("artery") or ""]
        if any(any(p in a for p in POST) for a in arts if a and a != "NA"):
            dropped["posterior_aneurysm"].append(sid + ":" + ",".join(sorted(set(a for a in arts if a and a != "NA"))))
            continue
        if sid in skip_second:
            dropped["duplicate_suspect_second_member"].append("%s (kept/blocked by %s)" % (sid, skip_second[sid]))
            continue
        ia_rows.append(x)
    pools = {"song": len(song_pool), "ia172": len(ia_rows)}

    # ---- scale-up scenarios
    def rate_pair(key):
        return {s: (rates[key][s]["pairs"], rates[key][s]["cases"]) for s in ("song", "ia172")}
    scen = {}
    for key in ("recalibrated", "backed_only", "coarse_enough_recal", "draft"):
        need_more = max(0, need_target - have[key])
        scen[key] = {"have_pairs_incl_VMR": have[key], "need_more_for_30": need_more,
                     "need_more_for_25": max(0, NEED_TOTAL - have[key]),
                     "cases_for_30": plan_cases(need_more, rate_pair(key), pools),
                     "cases_for_25": plan_cases(max(0, NEED_TOTAL - have[key]), rate_pair(key), pools)}
    # with the V1 mask-break rescue (untested): +ia172_061 and +AHMU1218141 (their failing side breaks only through
    # enhanced lumen, the other side passes); future rate +1 per source-sample
    resc = {"ia172": (rates["recalibrated"]["ia172"]["pairs"] + 1, rates["recalibrated"]["ia172"]["cases"]),
            "song": (rates["recalibrated"]["song"]["pairs"] + 1, rates["recalibrated"]["song"]["cases"])}
    have_r = have["recalibrated"] + 2
    scen["recalibrated_with_V1_rescue_untested"] = {
        "assumes": "re-joining Arterial V1 mask breaks through enhanced lumen turns ia172_061 and AHMU1218141 into pairs "
                   "(AHMU1218044 R also fails the origin rule; ia172_041 L bridges non-lumen) and lifts each source's rate "
                   "by one case in its pilot sample",
        "have_pairs_incl_VMR": have_r, "need_more_for_30": max(0, need_target - have_r),
        "cases_for_30": plan_cases(max(0, need_target - have_r), resc, pools)}

    # ---- ordered selection list
    def song_key(k):
        v = st[k]
        return (v["age"] >= 75, -(v.get("z_mm") or 0))
    strata = {"siemens_0.7": [], "ge_0.625": [], "philips_0.45": [], "other": []}
    for k in song_pool:
        v = st[k]
        sc = v.get("slice_class")
        fam = v.get("scanner_family") or ""
        if sc == "0.7" and fam.startswith("Siemens"):
            strata["siemens_0.7"].append(k)
        elif sc == "0.625" and fam.startswith("GE"):
            strata["ge_0.625"].append(k)
        elif sc == "0.45" and fam.startswith("Philips"):
            strata["philips_0.45"].append(k)
        else:
            strata["other"].append(k)
    for s in strata:
        strata[s].sort(key=song_key)
    song_order = list(strata["siemens_0.7"])
    ge, ph = list(strata["ge_0.625"]), list(strata["philips_0.45"])
    # interleave GE : Philips by pool share (about 5 : 2), so neither scanner family is starved
    while ge or ph:
        for _ in range(5):
            if ge:
                song_order.append(ge.pop(0))
        for _ in range(2):
            if ph:
                song_order.append(ph.pop(0))
    song_order += strata["other"]

    reserves = ["062", "155", "160", "067", "060", "119", "052", "131"]

    def ia_tier(x):
        t = 0 if x["sid"] in reserves else 1
        if x["sid"] not in reserves:
            if x.get("thick") == 0.5 and x.get("kernel") == "FC43" and x.get("ia") == "0":
                t = 1
            elif x.get("thick") == 0.5 and x.get("kernel") == "FC43":
                t = 2
            elif x.get("thick") == 0.5:
                t = 3
            else:
                t = 4
        return t
    # young (< 75) reserves and 0.5 mm FC43 controls first, then the >= 75 ones of those two tiers, then tiers 2-4
    ia_rows.sort(key=lambda x: (0 if ia_tier(x) <= 1 else ia_tier(x), (x.get("age") or 0) >= 75, ia_tier(x),
                                round(x.get("ps") or 9, 2), -(x.get("z") or 0)))
    tier_why = {0: "fetch agent's named reserve (control, 0.5 mm FC43, z >= 330, pre-screened)",
                1: "control, 0.5 mm Toshiba FC43 (the pilot protocol)",
                2: "anterior-circulation aneurysm, 0.5 mm FC43",
                3: "0.5 mm, other kernel (FC04)",
                4: "0.6-0.75 mm non-Toshiba series (acquisition voxel still < 1.15 mm)"}

    def song_why(k):
        v = st[k]
        s = next(n for n, l in strata.items() if k in l)
        w = {"siemens_0.7": "Siemens 0.7 mm: pilot 3/3 cases gave pairs, sharpest edges (FWHM 0.95-1.07 non-DE)",
             "ge_0.625": "GE 0.625 mm (largest stratum; pilot 2/3 pairs)",
             "philips_0.45": "Philips 0.45 mm (pilot 1/4 pairs; failures were V1 mask breaks and a rotated head)",
             "other": "remaining stratum"}[s]
        return w + ("; age >= 75, placed last in stratum (pilot: 1/5 cases >= 75 gave a pair vs 10/17 younger)" if v["age"] >= 75 else "")
    # global order: alternate Song and 172-series 1:1, Song first (plan: Song first; keep both sources represented)
    sel = []
    i = j = 0
    while i < len(song_order) or j < len(ia_rows):
        if i < len(song_order):
            k = song_order[i]
            v = st[k]
            sel.append({"source": "song", "case": k, "group": v["group"], "sex": v["sex"], "age": v["age"],
                        "z_mm": v["z_mm"], "slice_mm": v["slice_class"], "inplane_mm": round(v["zooms"][0], 3),
                        "scanner": v["scanner_family"].split(" (")[0], "aneurysm": v.get("aneurysm_locations"),
                        "why": song_why(k), "prefetch_check": "none needed (Song covers arch to vertex; z >= 300)",
                        "arterial_input": "ct_negzero_air1000 re-window (0-HU clipped release)"})
            i += 1
        if j < len(ia_rows):
            x = ia_rows[j]
            t = ia_tier(x)
            sel.append({"source": "ia172", "case": "ia172_" + x["sid"], "sex": x["sex"], "age": x["age"], "z_mm": x["z"],
                        "slice_mm": x["thick"], "inplane_mm": x["ps"], "scanner": "%s %s %s" % (x["manu"], x["model"], x["kernel"]),
                        "aneurysm": x.get("arteries"), "tier": t,
                        "why": tier_why[t] + ("; age >= 75, placed last in tier" if (x.get("age") or 0) >= 75 else ""),
                        "prefetch_check": "range-read the lowest slice: require two aortic limbs > 150 HU (166 / 090 / 118 lesson)",
                        "arterial_input": "CT clipped to [-1024, 3071]",
                        "duplicate_note": ("member of a same-demographics pair; the other member is excluded" if any(x["sid"] in p for p in dup_pairs) else None)})
            j += 1
    for n, r in enumerate(sel):
        r["order"] = n + 1
    m80 = scen["recalibrated"]["cases_for_30"].get("P80")
    m90 = scen["recalibrated"]["cases_for_30"].get("P90")

    # ---- per-case cost (TotalSegmentator --ml + CPU split)
    S = {s: L(RES + "/pilot_proc_%s_summary.json" % s)["timing_s"] for s in ("song", "ia172")}
    GPU = {"ts_total_v3_ml_totalseg", "ts_total_v3_ml", "arterial_attempt1"}
    SKIP = ("shipped", "arttest", "negzero", "phantom", "rescue")
    rows = []
    for src, tt in S.items():
        for c, d in tt.items():
            if src == "ia172" and "ts_total_v3_ml" not in d:
                continue   # only --ml cases
            gpu = cpu = wall_cpu = 0
            for k, v in d.items():
                if any(x in k for x in SKIP) or k == "ts_total_v3" or not isinstance(v, (int, float)):
                    continue
                if k in GPU:
                    gpu += v
                else:
                    cpu += v
                    if k.startswith(("ctseg", "pilot_proc")):
                        wall_cpu += v
            rows.append({"source": src, "case": c, "gpu_s": gpu, "cpu_s": cpu, "wall_stage_cpu_s": wall_cpu,
                         "arterial_s": d.get("arterial_attempt1"),
                         "ts_ml_s": d.get("ts_total_v3_ml_totalseg", d.get("ts_total_v3_ml"))})

    def q(xs, p):
        xs = sorted(xs)
        if not xs:
            return None
        k = (len(xs) - 1) * p
        f = int(k)
        return round(xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f), 0)
    cost = {}
    for src in ("song", "ia172", "all"):
        rr_ = [r for r in rows if src == "all" or r["source"] == src]
        cost[src] = {"n_cases": len(rr_), "cases": [r["case"] for r in rr_],
                     "gpu_s_median": q([r["gpu_s"] for r in rr_], .5), "gpu_s_max": q([r["gpu_s"] for r in rr_], 1),
                     "cpu_s_median": q([r["cpu_s"] for r in rr_], .5), "cpu_s_max": q([r["cpu_s"] for r in rr_], 1),
                     "ts_ml_s_range": [q([r["ts_ml_s"] for r in rr_], 0), q([r["ts_ml_s"] for r in rr_], 1)],
                     "arterial_s_range": [q([r["arterial_s"] for r in rr_], 0), q([r["arterial_s"] for r in rr_], 1)]}
    # throughput: one GPU (TotalSegmentator --ml + Arterial under the lock) shared by one Song lane and one 172 lane,
    # each lane one heavy CPU container at a time (the agents' rule)
    gs, cs = cost["song"]["gpu_s_median"], cost["song"]["cpu_s_median"]
    gi, ci = cost["ia172"]["gpu_s_median"], cost["ia172"]["cpu_s_median"]
    thr = {}
    for key in ("central", "P80", "P90"):
        pl = scen["recalibrated"]["cases_for_30"].get(key)
        if isinstance(pl, dict):
            ns, ni = pl["split"]["song"], pl["split"]["ia172"]
            lane_s, lane_i = ns * (gs + cs + 60) / 3600, ni * (gi + ci + 150) / 3600
            gpu_h = (ns * gs + ni * gi) / 3600
            thr[key] = {"cases": ns + ni, "song_lane_h": round(lane_s, 1), "ia172_lane_h": round(lane_i, 1),
                        "gpu_busy_h": round(gpu_h, 1), "wall_h_two_lanes": round(max(lane_s, lane_i, gpu_h), 1),
                        "serial_one_lane_h": round(lane_s + lane_i, 1)}
    fetch = {"song_range_read_s": [20, 71], "song_MB_per_case": [60, 109],
             "ia172_fetch_s": [31, 56], "ia172_convert_s": [33, 100], "ia172_MB_per_case_dicom_plus_nifti": [273, 611],
             "derived_MB_per_case": {"song": [392, 781], "ia172": [428, 519]},
             "D_free_GB_at_02:45": 505}

    out = {"what": "pilot_decide scale-up: yield per case, cases still needed, the next-case selection list, cost",
           "need": {"distinct_TopBrain_patients": NEED_TOTAL, "spares": SPARES, "target_pairs": need_target,
                    "eligible_now_incl_VMR": have},
           "pair_rate_per_case": rates, "age_exploratory": age_tab,
           "pools_remaining": {"song_eligible_3D_z300_unprocessed": pools["song"],
                               "ia172_HN_z300_le0.75mm_no_posterior_aneurysm_deduplicated": pools["ia172"],
                               "ia172_dropped": dropped, "song_strata_sizes": {k: len(v) for k, v in strata.items()}},
           "scenarios": scen,
           "selection_rule": ("alternate Song and 172-series 1:1 (Song first), so both sources stay represented; Song: "
                              "Siemens 0.7 mm first, then GE 0.625 : Philips 0.45 interleaved 5 : 2 (pool share); within a "
                              "stratum age < 75 first, then longer z. 172-series: named reserves, then 0.5 mm FC43 controls, "
                              "then anterior-aneurysm 0.5 mm, then FC04, then 0.6-0.75 mm series; within a tier age < 75 "
                              "first, finer in-plane pixel, longer z; posterior-circulation aneurysms excluded; one member "
                              "per same-demographics pair. Process in this order and STOP once eligible pairs (QC-passing, "
                              "both walls valid) reach 30, re-ranking after each batch."),
           "batch_plan": {"first_batch": m80 if isinstance(m80, dict) else None,
                          "list_length": len(sel), "covers_P90": m90},
           "selection": sel,
           "cost_per_case_ts_ml": cost, "cost_rows": rows, "fetch_and_disk": fetch, "throughput_estimate": thr,
           "song_duplicate_screen": {"identical_sex_age_shape_voxel": song_dups,
                                     "same_group_sex_age_slice (weak; not excluded)": song_same_demo}}
    json.dump(out, open(RES + "/pilot_next_cases.json", "w"), indent=1)
    print("RATES", json.dumps({k: {s: (v[s]["pairs"], v[s]["cases"], v[s]["wilson95"]) for s in v} for k, v in rates.items()}))
    print("AGE", json.dumps(age_tab))
    print("HAVE", have, "POOLS", pools, json.dumps({k: len(v) for k, v in dropped.items()}), json.dumps({k: len(v) for k, v in strata.items()}))
    for k, v in scen.items():
        print("SCEN", k, json.dumps(v))
    print("COST", json.dumps(cost))
    print("THROUGHPUT", json.dumps(thr))
    print("SONG DUPS", song_dups, "same-demo groups", len(song_same_demo))
    print("FIRST 20", [(r["order"], r["case"], r["age"]) for r in sel[:20]])
    print("DROPPED", json.dumps(dropped))


if __name__ == "__main__":
    main()
