import json
import numpy as np

D = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality.json"))
D2 = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality2.json"))
HOSTS = ["HOST_COLLISION", "HOST_VISUAL"]
COH = [k for k in D if k not in HOSTS]
SEGS = ["ALL", "CORE_no_last2", "TRUNK_s_lt_130", "DISTAL_HALF", "GRAFT_s_ge_130", "GRAFT_core"]


def row(tag, seg):
    return D[tag]["segments"][seg]


print("=" * 150)
print("T1  PER-ANATOMY, WHOLE RCCA ROUTE (own centerline stations)")
print("=" * 150)
h = ("anat", "n", "len", "tort", "turn", "medcl", "p05", "p01", "min", "nW", "nC",
     "reffmed", "reffmin", "Rcmin", "Rcp05", "bendmax", "T&B", "dist_min", "dist_med")
print(("%-15s" + "%6s" * 3 + "%7s" + "%7s%7s%7s%7s" + "%4s%4s" + "%8s%8s" +
       "%7s%7s%8s%5s%9s%8s") % h)
for t in HOSTS + COH:
    a = row(t, "ALL")
    dh = row(t, "DISTAL_HALF")
    print(("%-15s%6d%6.1f%6.2f%7.0f%7.3f%7.3f%7.3f%7.3f%4d%4d%8.3f%8.3f%7.1f%7.1f%8.1f%5d"
           "%9.3f%8.3f")
          % (t, D[t]["n_stations"], D[t]["route_len"], D[t]["tortuosity"],
             D[t]["total_turning_deg"], a["clear"]["med"], a["clear"]["p05"],
             a["clear"]["p01"], a["clear"]["min"], a["n_blk_wire"], a["n_blk_cath"],
             a["reff"]["med"], a["reff"]["min"], a["Rc_min"], a["Rc_p05"], a["bend_max"],
             a["n_tight_bent"], dh["clear"]["min"], dh["clear"]["med"]))

print()
print("=" * 150)
print("T2  SEGMENT BREAKDOWN  (med / p05 / min clearance, reff med, nWire, nCath, tight&bent)")
print("=" * 150)
for seg in ["TRUNK_s_lt_130", "GRAFT_s_ge_130", "GRAFT_core"]:
    print("-- %s" % seg)
    print("%-15s%5s%8s%8s%8s%9s%9s%5s%5s%5s%8s%8s%8s"
          % ("anat", "n", "medcl", "p05", "min", "reffmed", "reff/r", "nW", "nC", "T&B",
             "Rcmin", "Rcp05", "bendmax"))
    for t in HOSTS + COH:
        s = row(t, seg)
        if s.get("n", 0) == 0:
            continue
        print("%-15s%5d%8.3f%8.3f%8.3f%9.3f%9.3f%5d%5d%5d%8.1f%8.1f%8.1f"
              % (t, s["n"], s["clear"]["med"], s["clear"]["p05"], s["clear"]["min"],
                 s["reff"]["med"], s["reff_over_stated"], s["n_blk_wire"], s["n_blk_cath"],
                 s["n_tight_bent"], s["Rc_min"], s["Rc_p05"], s["bend_max"]))
    print()


def agg(tags, seg, path):
    v = []
    for t in tags:
        s = row(t, seg)
        if s.get("n", 0) == 0:
            continue
        x = s
        for p in path:
            x = x[p]
        v.append(x)
    return np.array(v, float)


print("=" * 150)
print("T3  COHORT vs HOST SUMMARY  (cohort = median across the 22, [min,max] in brackets)")
print("=" * 150)
metrics = [("median clearance", ["clear", "med"]), ("p05 clearance", ["clear", "p05"]),
           ("p01 clearance", ["clear", "p01"]), ("min clearance", ["clear", "min"]),
           ("median r_eff", ["reff", "med"]), ("min r_eff", ["reff", "min"]),
           ("r_eff/stated_r", ["reff_over_stated"]), ("n blocked wire", ["n_blk_wire"]),
           ("n blocked cath", ["n_blk_cath"]), ("n tight&bent", ["n_tight_bent"]),
           ("Rc min", ["Rc_min"]), ("Rc p05", ["Rc_p05"]), ("bend max", ["bend_max"])]
for seg in ["ALL", "CORE_no_last2", "TRUNK_s_lt_130", "DISTAL_HALF", "GRAFT_s_ge_130"]:
    print("-- SEGMENT %s" % seg)
    print("%-20s%12s%12s%14s%12s%12s" % ("metric", "HOSTcoll", "HOSTvis", "cohort_med",
                                         "coh-HOSTc", "coh-HOSTv"))
    for nm, p in metrics:
        try:
            hc = agg(["HOST_COLLISION"], seg, p)[0]
            hv = agg(["HOST_VISUAL"], seg, p)[0]
            cv = agg(COH, seg, p)
        except Exception:
            continue
        print("%-20s%12.3f%12.3f%14s%12.3f%12.3f"
              % (nm, hc, hv, "%.3f[%.3f,%.3f]" % (np.median(cv), cv.min(), cv.max()),
                 np.median(cv) - hc, np.median(cv) - hv))
    print()

print("=" * 150)
print("T4  DIFFERENCE-IN-DIFFERENCES: graft-vs-trunk within-anatomy, cohort vs host")
print("    D = (graft - trunk) per anatomy; anything left after subtracting the host's own D")
print("    is REAL ANATOMY, since the surface offset cancels within an anatomy.")
print("=" * 150)
for nm, p in [("median clearance", ["clear", "med"]), ("p05 clearance", ["clear", "p05"]),
              ("min clearance", ["clear", "min"]), ("median r_eff", ["reff", "med"]),
              ("min r_eff", ["reff", "min"]), ("Rc p05", ["Rc_p05"]),
              ("bend max", ["bend_max"])]:
    g_h = agg(["HOST_COLLISION"], "GRAFT_s_ge_130", p)[0]
    t_h = agg(["HOST_COLLISION"], "TRUNK_s_lt_130", p)[0]
    g_v = agg(["HOST_VISUAL"], "GRAFT_s_ge_130", p)[0]
    t_v = agg(["HOST_VISUAL"], "TRUNK_s_lt_130", p)[0]
    g_c = agg(COH, "GRAFT_s_ge_130", p)
    t_c = agg(COH, "TRUNK_s_lt_130", p)
    dc = g_c - t_c
    print("%-18s host_coll D=%+8.3f  host_vis D=%+8.3f  cohort D med=%+8.3f [%+.3f,%+.3f]"
          "   DiD_vs_coll=%+8.3f DiD_vs_vis=%+8.3f"
          % (nm, g_h - t_h, g_v - t_v, np.median(dc), dc.min(), dc.max(),
             np.median(dc) - (g_h - t_h), np.median(dc) - (g_v - t_v)))

print()
print("=" * 150)
print("T5  SHARED-TRUNK SURFACE CONTROL (pass 2): all surfaces sampled on ONE common")
print("    polyline = host RCCA trunk resampled at 0.5 mm, 260 stations, s = 0 - 129.5 mm")
print("=" * 150)
P = D2["per"]
print("%-16s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s"
      % ("surface", "devmed", "devmax", "cl_med", "cl_p05", "cl_p01", "cl_min",
         "reffmed", "reffmin", "cl/reff", "coff"))
rows = {}
for t in ["HOST_COLLISION", "HOST_VISUAL"] + COH:
    d = P[t]
    cl = np.array(d["cl"])
    re = np.array(d["reff"])
    co = np.array(d["coff"])
    fin = np.isfinite(re)
    rows[t] = (np.median(cl), np.percentile(cl, 5), np.percentile(cl, 1), cl.min(),
               np.nanmedian(re), np.nanmin(re), np.nanmedian(cl[fin] / re[fin]),
               np.nanmedian(co))
    print("%-16s%9.4f%9.4f%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f"
          % (t, d["dev_med"], d["dev_max"], rows[t][0], rows[t][1], rows[t][2], rows[t][3],
             rows[t][4], rows[t][5], rows[t][6], rows[t][7]))
ch = np.array([rows[t] for t in COH])
print("%-16s%9s%9s%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f"
      % ("COHORT median", "-", "-", *np.median(ch, 0)))
lab = ["cl_med", "cl_p05", "cl_p01", "cl_min", "reff_med", "reff_min", "cl/reff", "coff"]
cm = np.median(ch, 0)
print("\n  HEADLINE surface-only deltas (cohort median - host), anatomy held constant:")
for i, L in enumerate(lab):
    print("    %-9s cohort %8.3f   vs HOSTcoll %8.3f (%+7.3f, %+6.1f%%)   "
          "vs HOSTvis %8.3f (%+7.3f, %+6.1f%%)"
          % (L, cm[i], rows["HOST_COLLISION"][i], cm[i] - rows["HOST_COLLISION"][i],
             100 * (cm[i] / rows["HOST_COLLISION"][i] - 1), rows["HOST_VISUAL"][i],
             cm[i] - rows["HOST_VISUAL"][i], 100 * (cm[i] / rows["HOST_VISUAL"][i] - 1)))

print("\n  per-station paired difference on the common curve (cohort median profile - host):")
CL = np.array([P[t]["cl"] for t in COH])
RE = np.array([P[t]["reff"] for t in COH])
cmed = np.median(CL, 0)
rmed = np.nanmedian(RE, 0)
for hn in HOSTS:
    dd = cmed - np.array(P[hn]["cl"])
    rr = rmed - np.array(P[hn]["reff"])
    print("    vs %-15s clearance diff: med %+0.3f  p05 %+0.3f  p95 %+0.3f  frac_stations_"
          "cohort_tighter %.2f | reff diff med %+0.3f  frac_narrower %.2f"
          % (hn, np.median(dd), np.percentile(dd, 5), np.percentile(dd, 95), (dd < 0).mean(),
             np.nanmedian(rr), np.mean(rr < 0)))

print()
print("=" * 150)
print("T6  BLOCKED / NEAR-BLOCKED STATIONS (own centerline), exact signed distance")
print("=" * 150)
for t in HOSTS + COH:
    s = np.array(D[t]["profile"]["s"])
    cl = np.array(D[t]["profile"]["clear"])
    rc = np.array(D[t]["profile"]["Rc"])
    w = np.where(cl < 0.18)[0]
    c = np.where(cl < 0.35)[0]
    tb = np.where((rc > 0) & (rc < 8.0) & (cl < 0.6))[0]
    print("%-16s wire<0.18: %-38s cath<0.35: %-30s tight&bent: %s"
          % (t, [("%.0fmm/%.3f" % (s[i], cl[i])) for i in w],
             [("%.0fmm/%.3f" % (s[i], cl[i])) for i in c],
             [("%.0fmm" % s[i]) for i in tb][:8]))
