import json
import numpy as np

D = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality.json"))
D2 = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality2.json"))
HOSTS = ["HOST_COLLISION", "HOST_VISUAL"]
COH = [k for k in D if k not in HOSTS]
P = D2["per"]
QS = np.array(D2["common_s"])

print("=" * 120)
print("A. WHERE the host surfaces dip in the SHARED TRUNK (common curve), and what the "
      "cohort does at those same stations")
print("=" * 120)
clv = np.array(P["HOST_VISUAL"]["cl"])
clc = np.array(P["HOST_COLLISION"]["cl"])
CL = np.array([P[t]["cl"] for t in COH])
cmed = np.median(CL, 0)
cmin = CL.min(0)
for nm, arr in (("HOST_VISUAL", clv), ("HOST_COLLISION", clc)):
    k = np.argsort(arr)[:8]
    k = np.sort(k)
    print(" %s 8 tightest trunk stations:" % nm)
    for i in k:
        print("   s=%6.1f  host=%.3f   cohort med=%.3f  cohort min=%.3f  cohort max=%.3f"
              % (QS[i], arr[i], cmed[i], cmin[i], CL[:, i].max()))

print()
print("=" * 120)
print("B. RATIO form of the shared-trunk surface effect (per-station, cohort median / host)")
print("=" * 120)
RE = np.array([P[t]["reff"] for t in COH])
rmed = np.nanmedian(RE, 0)
for nm in HOSTS:
    hr = np.array(P[nm]["reff"])
    hc = np.array(P[nm]["cl"])
    print("  vs %-15s  reff ratio med %.3f [p05 %.3f p95 %.3f] | clearance ratio med %.3f "
          "[p05 %.3f p95 %.3f]"
          % (nm, np.nanmedian(rmed / hr), np.nanpercentile(rmed / hr, 5),
             np.nanpercentile(rmed / hr, 95), np.median(cmed / hc),
             np.percentile(cmed / hc, 5), np.percentile(cmed / hc, 95)))
sr = np.array(P["HOST_VISUAL"]["stated_r"])
print("  reff/stated_r on the common trunk curve: host_vis %.3f  host_coll %.3f  cohort "
      "med %.3f" % (np.nanmedian(np.array(P["HOST_VISUAL"]["reff"]) / sr),
                    np.nanmedian(np.array(P["HOST_COLLISION"]["reff"]) / sr),
                    np.nanmedian(rmed / sr)))

print()
print("=" * 120)
print("C. PER-ANATOMY VERDICT on the DISTAL/GRAFT segment vs each host surface")
print("=" * 120)


def g(t, k, path):
    x = D[t]["segments"][k]
    for p in path:
        x = x[p]
    return x


hdr = ("anat", "gr_medcl", "gr_p05", "gr_min", "gr_reffmed", "n_TB", "nW", "nC",
       "vs_VIS_min", "vs_VIS_med", "vs_COL_min")
print(("%-15s" + "%10s" * 4 + "%5s%4s%4s" + "%12s%12s%12s") % hdr)
hv = (g("HOST_VISUAL", "GRAFT_s_ge_130", ["clear", "med"]),
      g("HOST_VISUAL", "GRAFT_s_ge_130", ["clear", "min"]))
hc2 = (g("HOST_COLLISION", "GRAFT_s_ge_130", ["clear", "med"]),
       g("HOST_COLLISION", "GRAFT_s_ge_130", ["clear", "min"]))
n_worse_min_v = n_worse_med_v = n_worse_min_c = 0
for t in COH:
    m = g(t, "GRAFT_s_ge_130", ["clear", "med"])
    p5 = g(t, "GRAFT_s_ge_130", ["clear", "p05"])
    mn = g(t, "GRAFT_s_ge_130", ["clear", "min"])
    re = g(t, "GRAFT_s_ge_130", ["reff", "med"])
    tb = g(t, "GRAFT_s_ge_130", ["n_tight_bent"])
    nw = g(t, "GRAFT_s_ge_130", ["n_blk_wire"])
    nc = g(t, "GRAFT_s_ge_130", ["n_blk_cath"])
    n_worse_min_v += mn < hv[1]
    n_worse_med_v += m < hv[0]
    n_worse_min_c += mn < hc2[1]
    print("%-15s%10.3f%10.3f%10.3f%10.3f%5d%4d%4d%12.3f%12.3f%12.3f"
          % (t, m, p5, mn, re, tb, nw, nc, mn - hv[1], m - hv[0], mn - hc2[1]))
print("HOST_VISUAL graft: med %.3f min %.3f | HOST_COLLISION graft: med %.3f min %.3f"
      % (hv[0], hv[1], hc2[0], hc2[1]))
print("cohort anatomies with graft MIN tighter than host_visual: %d/22" % n_worse_min_v)
print("cohort anatomies with graft MEDIAN tighter than host_visual: %d/22" % n_worse_med_v)
print("cohort anatomies with graft MIN tighter than host_collision(0.0): %d/22" % n_worse_min_c)

print()
print("=" * 120)
print("D. SURFACE-CORRECTED graft comparison. The shared-trunk control gives the surface's")
print("   per-station multiplicative effect on r_eff; apply it to the cohort graft r_eff to")
print("   estimate what the SAME anatomy would measure if baked by the host's pipeline.")
print("=" * 120)
kmed = np.nanmedian(rmed / np.array(P["HOST_VISUAL"]["reff"]))
for nm, path in (("graft reff med", ["reff", "med"]), ("graft reff min", ["reff", "min"])):
    cv = np.array([g(t, "GRAFT_s_ge_130", path) for t in COH])
    print("  %-15s cohort med %.3f -> surface-corrected %.3f (/%.3f)   host_visual %.3f"
          "   host_collision %.3f"
          % (nm, np.median(cv), np.median(cv) / kmed, kmed,
             g("HOST_VISUAL", "GRAFT_s_ge_130", path),
             g("HOST_COLLISION", "GRAFT_s_ge_130", path)))
print("  (surface factor k = cohort/host_visual r_eff on the shared trunk = %.4f)" % kmed)

print()
print("=" * 120)
print("E. GEOMETRY-ONLY (surface independent) difficulty: centerline shape, graft segment")
print("=" * 120)
print("%-15s%10s%10s%10s%10s%10s%10s" % ("anat", "Rc_min", "Rc_p05", "bend_max", "bend_p95",
                                         "tort_all", "turn_all"))
for t in ["HOST_COLLISION"] + COH:
    s = D[t]["segments"]["GRAFT_s_ge_130"]
    print("%-15s%10.2f%10.2f%10.1f%10.1f%10.3f%10.0f"
          % (t, s["Rc_min"], s["Rc_p05"], s["bend_max"], s["bend_p95"],
             D[t]["tortuosity"], D[t]["total_turning_deg"]))
cv = {k: np.array([D[t]["segments"]["GRAFT_s_ge_130"][k] for t in COH])
      for k in ["Rc_min", "Rc_p05", "bend_max", "bend_p95"]}
cv["tort"] = np.array([D[t]["tortuosity"] for t in COH])
cv["turn"] = np.array([D[t]["total_turning_deg"] for t in COH])
h = D["HOST_COLLISION"]
for k in cv:
    hv2 = (h["segments"]["GRAFT_s_ge_130"][k] if k in h["segments"]["GRAFT_s_ge_130"]
           else (h["tortuosity"] if k == "tort" else h["total_turning_deg"]))
    worse = ((cv[k] < hv2).sum() if k.startswith("Rc")
             else (cv[k] > hv2).sum())
    print("  %-10s host %8.2f | cohort med %8.2f [%.2f,%.2f] | n_cohort_HARDER_than_host %d/22"
          % (k, hv2, np.median(cv[k]), cv[k].min(), cv[k].max(), worse))
