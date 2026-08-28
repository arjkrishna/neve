import json
import numpy as np

D = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality.json"))
D2 = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality2.json"))
HOSTS = ["HOST_COLLISION", "HOST_VISUAL"]
COH = [k for k in D if k not in HOSTS]
P = D2["per"]
QS = np.array(D2["common_s"])
CL = np.array([P[t]["cl"] for t in COH])
RE = np.array([P[t]["reff"] for t in COH])
cmed, rmed = np.median(CL, 0), np.nanmedian(RE, 0)

print("A. r_eff around the host's distal-trunk narrowing (s 112-129, the zone whose declared")
print("   radii were overwritten by smoothstep):")
m = (QS >= 112) & (QS <= 129.5)
print("   host_visual reff med %.3f min %.3f | host_coll reff med %.3f | cohort reff med "
      "%.3f | stated_r med %.3f"
      % (np.nanmedian(np.array(P["HOST_VISUAL"]["reff"])[m]),
         np.nanmin(np.array(P["HOST_VISUAL"]["reff"])[m]),
         np.nanmedian(np.array(P["HOST_COLLISION"]["reff"])[m]),
         np.nanmedian(rmed[m]), np.median(np.array(P["HOST_VISUAL"]["stated_r"])[m])))

print()
print("B. Shared-trunk control restricted to s < 105 mm (EXCLUDES the overwritten-radius zone)")
m2 = QS < 105
for nm in HOSTS:
    h = np.array(P[nm]["cl"])[m2]
    hr = np.array(P[nm]["reff"])[m2]
    print("   vs %-15s cohortmed cl %.3f vs %.3f (%+.3f, %+.1f%%) | cl p05 %.3f vs %.3f | "
          "cl min %.3f vs %.3f | reff %.3f vs %.3f (%+.1f%%) | frac cohort tighter %.2f"
          % (nm, np.median(cmed[m2]), np.median(h), np.median(cmed[m2]) - np.median(h),
             100 * (np.median(cmed[m2]) / np.median(h) - 1),
             np.percentile(cmed[m2], 5), np.percentile(h, 5), cmed[m2].min(), h.min(),
             np.nanmedian(rmed[m2]), np.nanmedian(hr), 100 * (np.nanmedian(rmed[m2]) /
                                                              np.nanmedian(hr) - 1),
             float((cmed[m2] - h < 0).mean())))
print("   per-anatomy cl_min on s<105: cohort %s" %
      np.round([CL[i][m2].min() for i in range(len(COH))], 3).tolist())

print()
print("C. Is the host's high total turning real curvature or centerline noise?")


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def turning(C, step):
    S = arclen(C)
    q = np.arange(0, S[-1], step)
    R = np.stack([np.interp(q, S, C[:, k]) for k in range(3)], 1)
    d = np.diff(R, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1
    d /= n
    return float(np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1, 1))).sum())


for st in [0.5, 1.0, 2.0, 4.0, 8.0]:
    hc = turning(np.array(D["HOST_COLLISION"]["profile"]["C"]), st)
    cc = np.array([turning(np.array(D[t]["profile"]["C"]), st) for t in COH])
    print("   step %4.1f mm : host %7.0f deg | cohort med %7.0f [%.0f,%.0f] | ratio %.2f"
          % (st, hc, np.median(cc), cc.min(), cc.max(), np.median(cc) / hc))

print()
print("D. station spacing (own centerlines): host %.4f mm | cohort med %.4f mm"
      % (D["HOST_COLLISION"]["station_spacing_med"],
         np.median([D[t]["station_spacing_med"] for t in COH])))

print()
print("E. FRACTION of the whole route below thresholds (own centerline, core = drop last 2)")
print("   %-16s %8s %8s %8s %8s" % ("anat", "f<0.35", "f<0.6", "f<1.0", "f<1.5"))
for t in HOSTS + COH:
    cl = np.array(D[t]["profile"]["clear"])[:-2]
    print("   %-16s %8.3f %8.3f %8.3f %8.3f"
          % (t, (cl < 0.35).mean(), (cl < 0.6).mean(), (cl < 1.0).mean(), (cl < 1.5).mean()))
cohf = np.array([[(np.array(D[t]["profile"]["clear"])[:-2] < x).mean()
                  for x in (0.35, 0.6, 1.0, 1.5)] for t in COH])
print("   %-16s %8.3f %8.3f %8.3f %8.3f" % ("COHORT median", *np.median(cohf, 0)))
