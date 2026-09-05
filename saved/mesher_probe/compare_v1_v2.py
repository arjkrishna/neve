"""v1 vs v2, set-wide: meshed lumen, deficit, components, navigability."""
import glob, json, os, sys, numpy as np
v1 = json.load(open("saved/mesher_probe/lumen_v1.json"))
def load_v2(root):
    out = {}
    for f in glob.glob(os.path.join(root, "*", "mesh_v2.json")):
        out[os.path.basename(os.path.dirname(f))] = json.load(open(f))["obj"]
    return out
def summ(S):
    lm = np.array([v.get("lumen_min_mm", 0) for v in S.values()]); df = np.array([v.get("median_deficit_mm", 0) for v in S.values()])
    comps = np.array([v.get("comps", 0) for v in S.values()]); nav = sum(1 for v in S.values() if v.get("navigable"))
    return "n=%3d  navigable %3d (%3.0f%%)  min-lumen median %.2f p10 %.2f  deficit median %.2f  components median %d max %d" % (
        len(S), nav, 100.0 * nav / max(len(S), 1), np.median(lm), np.percentile(lm, 10), np.median(df), np.median(comps), comps.max())
for tag, prefix, root in (("A TopBrain", "topcow", "topbrain_data/anatomies_v2"), ("B carotid", "case_", "carotid_data/anatomies_v2")):
    a1 = {k: v for k, v in v1.items() if k.startswith(prefix)}; a2 = load_v2(root)
    print("%s\n   v1  %s\n   v2  %s" % (tag, summ(a1), summ(a2)))
    both = sorted(set(a1) & set(a2))
    if both:
        d = np.array([a2[k].get("lumen_min_mm", 0) - a1[k].get("lumen_min_mm", 0) for k in both])
        print("   per-anatomy min-lumen gain (v2 - v1) over %d shared: median %+.2f mm, min %+.2f, worse in %d" % (len(both), np.median(d), d.min(), int((d < 0).sum())))
    bad = [k for k, v in a2.items() if not v.get("navigable")]
    if bad: print("   v2 NOT navigable: %s" % ", ".join(bad[:12]) + (" ..." if len(bad) > 12 else ""))
